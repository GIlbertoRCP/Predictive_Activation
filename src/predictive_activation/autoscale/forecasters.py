"""Forecasters for the load a newly started pod will face.

Target: the peak request rate over the window [t + lead, t + lead + window),
where `lead` is the pod startup delay in steps. A pod requested at step t is
only useful from t + lead onward, so that peak is what capacity must cover.

All forecasters share one contract: `predict(rps, ...)[t]` uses only rps[:t]
(observations strictly before t, matching the one-step metrics lag in the
simulator). Predictions are computed for the whole series in one pass, which
keeps evaluation fast and makes look-ahead leaks easy to test.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from predictive_activation.autoscale.traces import DAY_S


def window_peak_target(rps, lead, window):
    """y[t] = max(rps[t+lead : t+lead+window]); NaN where the window runs off the end."""
    s = pd.Series(rps)
    # rolling max ending at i covers [i-window+1, i]; shift so it starts at t+lead
    fwd = s.rolling(window, min_periods=window).max().shift(-(lead + window - 1))
    return fwd.to_numpy()


def _past(rps):
    """Series where index t holds rps[t-1]: the latest value visible at decision time."""
    return pd.Series(rps).shift(1)


class Forecaster:
    name = "base"

    def fit(self, rps, step_s, lead, window):
        return self

    def predict(self, rps, step_s, lead, window):
        raise NotImplementedError


class LastValue(Forecaster):
    """Persistence: assume the most recent rate continues."""
    name = "last_value"

    def predict(self, rps, step_s, lead, window):
        return _past(rps).bfill().to_numpy()


class SeasonalNaive(Forecaster):
    """Yesterday's peak over the same window, rescaled by today's level vs yesterday's."""
    name = "seasonal_naive"

    def __init__(self, level_window_s=1800):
        self.level_window_s = level_window_s

    def predict(self, rps, step_s, lead, window):
        day = DAY_S // step_s
        past = _past(rps)
        yesterday_peak = pd.Series(window_peak_target(rps, lead, window)).shift(day)
        lw = max(1, self.level_window_s // step_s)
        level_now = past.rolling(lw, min_periods=1).mean()
        level_yday = level_now.shift(day)
        ratio = (level_now / level_yday).clip(0.5, 2.0)
        pred = yesterday_peak * ratio
        # First day has no history: fall back to persistence
        return pred.fillna(past).bfill().to_numpy()


class Holt(Forecaster):
    """Double exponential smoothing (level + trend), extrapolated to the window end."""
    name = "holt"

    def __init__(self, alpha=0.3, beta=0.05):
        self.alpha, self.beta = alpha, beta

    def predict(self, rps, step_s, lead, window):
        x = np.asarray(rps, dtype=float)
        n = len(x)
        out = np.empty(n)
        level, trend = x[0], 0.0
        out[0] = x[0]
        for t in range(1, n):
            # update with x[t-1], then forecast for t
            prev = level
            level = self.alpha * x[t - 1] + (1 - self.alpha) * (level + trend)
            trend = self.beta * (level - prev) + (1 - self.beta) * trend
            h = lead + window
            out[t] = max(level + max(trend, 0.0) * h, level)  # never forecast a decline into the window
        return out


class GBMQuantile(Forecaster):
    """Gradient-boosted quantile regression on lag, rolling and calendar features.

    Predicts a high quantile (default p90) of the window peak, so the scaler
    provisions for a bad-but-plausible case rather than the average.
    """
    name = "gbm_q90"

    def __init__(self, quantile=0.9, max_iter=300, seed=0):
        self.quantile = quantile
        self.model = HistGradientBoostingRegressor(loss="quantile", quantile=quantile, max_iter=max_iter,
                                                   learning_rate=0.05, random_state=seed)
        self.name = f"gbm_q{int(quantile * 100)}"

    @staticmethod
    def features(rps, step_s, lead, window):
        day = DAY_S // step_s
        past = _past(rps)
        logp = np.log1p(past)
        feats = {}
        for k in (1, 2, 3, 4, 8):
            feats[f"lag{k}"] = logp.shift(k - 1)
        for secs in (60, 300, 900, 3600):
            w = max(1, secs // step_s)
            feats[f"mean_{secs}s"] = logp.rolling(w, min_periods=1).mean()
            feats[f"max_{secs}s"] = logp.rolling(w, min_periods=1).max()
        feats["slope_300s"] = logp - logp.shift(max(1, 300 // step_s))
        peak = pd.Series(np.log1p(window_peak_target(rps, lead, window)))
        # Sub-hourly periodicity (cron-style jobs every 5/10/15/30/60 min): the
        # window peak observed that long ago. Shifts are at least lead + window
        # so the window had fully happened by decision time.
        horizon = lead + window
        for minutes in (5, 10, 15, 30, 60):
            k = minutes * 60 // step_s
            if k >= horizon:
                feats[f"peak_{minutes}min_ago"] = peak.shift(k)
        feats["peak_yesterday"] = peak.shift(day)
        feats["peak_last_week"] = peak.shift(7 * day)
        feats["level_vs_yesterday"] = feats["mean_900s"] - feats["mean_900s"].shift(day)
        t = np.arange(len(rps)) * step_s
        tod = 2 * np.pi * (t % DAY_S) / DAY_S
        feats["tod_sin"], feats["tod_cos"] = np.sin(tod), np.cos(tod)
        feats["weekend"] = ((t // DAY_S) % 7 >= 5).astype(float)
        return pd.DataFrame(feats)

    def fit(self, rps, step_s, lead, window):
        X = self.features(rps, step_s, lead, window)
        y = np.log1p(window_peak_target(rps, lead, window))
        ok = np.isfinite(y)
        self.model.fit(X[ok], y[ok])
        return self

    def predict(self, rps, step_s, lead, window):
        X = self.features(rps, step_s, lead, window)
        return np.expm1(self.model.predict(X))


class Oracle(Forecaster):
    """Knows the future. An upper bound, not a real forecaster."""
    name = "oracle"

    def predict(self, rps, step_s, lead, window):
        y = window_peak_target(rps, lead, window)
        return pd.Series(y).ffill().to_numpy()


def evaluate_forecast(y_true, y_pred, quantile=0.9):
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    yt, yp = y_true[ok], y_pred[ok]
    err = yt - yp
    return {
        "mae": float(np.abs(err).mean()),
        "mape_pct": float(100 * np.mean(np.abs(err) / np.maximum(yt, 1e-9))),
        "under_forecast_pct": float(100 * np.mean(yt > yp)),  # when this happens, pods arrive short
        "pinball_q90": float(np.mean(np.maximum(quantile * err, (quantile - 1) * err))),
    }
