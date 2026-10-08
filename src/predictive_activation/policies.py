"""Control policies.

Every policy answers two questions each tick: *how often should I poll* and *which flow should I move*.

Placement policies all share one optimizer (`Placement`). It takes a forecast of per-flow activity over
the next HORIZON ticks, predicts the load on each core link, and makes the single flow move that most
reduces predicted overload. Policies differ ONLY in the forecast they plug in:

    persistence forecast  -> "Reactive load balancer"   (assumes each flow keeps doing what it does now)
    learned forecast      -> "Predictive"               (gradient-boosted per-flow activity forecaster)
    true schedule         -> "Clairvoyant upper bound"  (reads the simulator's job timers)

so any gap between them is attributable to forecast quality, not to different control logic.
"""
from __future__ import annotations

import numpy as np

from .config import SimConfig
from .engine import Action, View
from .features import window_features

HORIZON = 10             # ticks ahead that forecasts and labels look
COOLDOWN = 20            # ticks before the same flow may be moved again
ACTIVE_FRAC = 0.30       # a flow counts as "active" above 30% of link capacity
FLOW_HORIZONS = (2, 4, 7, 10)   # horizons the learned flow forecaster is trained on
HEADROOM = 0.92          # predicted load above this fraction of capacity counts as overload


# ---------------------------------------------------------------------------------- baselines
class NoControl:
    """Static ECMP placement; slow heartbeat polling. The 'do nothing' baseline."""
    clairvoyant = False

    def __init__(self, poll: int = 30):
        self.poll = poll
        self.name = f"No control (poll {poll}s)"

    def reset(self, cfg: SimConfig, seed: int) -> None: ...
    def act(self, view: View) -> Action:
        return Action(self.poll)


class ThresholdReactive:
    """The classic monitoring loop: poll, and once a link's *observed* queue is high move its heaviest flow."""
    clairvoyant = False

    def __init__(self, poll: int = 30, queue_thresh: float = 0.35):
        self.poll, self.thresh = poll, queue_thresh
        self.name = f"Threshold reroute (poll {poll}s)"

    def reset(self, cfg: SimConfig, seed: int) -> None: ...

    def act(self, view: View) -> Action:
        moves = []
        if view.polled:
            q = view.windows[:, -1, 1]
            for a in np.flatnonzero(q >= self.thresh):
                b = int(np.argmin(q))
                cand = [j for j in range(len(view.flow_link))
                        if view.flow_link[j] == a and view.flow_rate[j] > 150
                        and view.t - view.last_move_t[j] >= COOLDOWN]
                if cand and b != a and q[b] < self.thresh:
                    moves.append((max(cand, key=lambda j: view.flow_rate[j]), b))
        return Action(self.poll, moves)


# ---------------------------------------------------------------------------------- flow forecasts
class PersistenceForecast:
    """Reactive view of the world: a flow active now stays active; an idle flow stays idle."""
    name = "persistence"

    def predict(self, view: View) -> np.ndarray:
        active = (view.flow_windows[:, -1] > ACTIVE_FRAC).astype(float)
        return np.repeat(active[:, None], HORIZON, axis=1)


class LearnedFlowForecast:
    """P(flow active at t+k) from the flow's own recent (stale) rate history.

    New information only arrives at a poll, so the model is evaluated at polls and the forecast is
    advanced (shifted) by the number of ticks elapsed in between.
    """
    name = "learned"

    def __init__(self, model):
        self.model = model
        self._p, self._t0 = None, 0

    def reset(self) -> None:
        self._p = None

    def _evaluate(self, view: View) -> np.ndarray:
        n = view.flow_windows.shape[0]
        win = np.zeros((n, view.flow_windows.shape[1], 4), np.float32)
        win[:, :, 0] = view.flow_windows
        base = window_features(win, np.full(n, view.age[0]), np.full(n, view.interval))
        X = np.concatenate([np.hstack([base, np.full((n, 1), k, np.float32)]) for k in FLOW_HORIZONS])
        p = self.model.risk(X).reshape(len(FLOW_HORIZONS), n)
        ks = np.arange(1, HORIZON + 1)
        return np.stack([np.interp(ks, FLOW_HORIZONS, p[:, j]) for j in range(n)])

    def predict(self, view: View) -> np.ndarray:
        if self._p is None or view.polled:
            self._p, self._t0 = self._evaluate(view), view.t
        e = min(view.t - self._t0, HORIZON - 1)
        if e == 0:
            return self._p
        return np.concatenate([self._p[:, e:], np.repeat(self._p[:, -1:], e, axis=1)], axis=1)


class TrueForecast:
    """Clairvoyant: reads the simulator's job timers. Not deployable; defines the achievable ceiling."""
    name = "true schedule"

    def predict(self, view: View) -> np.ndarray:
        fab = view.fabric
        return (fab.future_job_load(HORIZON) / fab.peak[:, None] > 0.5).astype(float)


# ---------------------------------------------------------------------------------- polling schedules
class FixedPoll:
    def __init__(self, k: int):
        self.k, self.name = k, f"poll {k}s"

    def reset(self) -> None: ...
    def interval(self, view: View, predicted_overload: float) -> int:
        return self.k


class AdaptivePoll:
    """Cheap heartbeat polls; drop to 1 s whenever the forecast predicts a link overload (a collision)."""

    def __init__(self, base: int = 10, calm_ticks: int = 6, overload_trigger: float = 0.0):
        self.base, self.calm_ticks, self.trigger = base, calm_ticks, overload_trigger
        self.name = f"adaptive poll ({base}s/1s)"
        self._calm = 0

    def reset(self) -> None:
        self._calm = 0

    def interval(self, view: View, predicted_overload: float) -> int:
        busy = predicted_overload > self.trigger or view.windows[:, -1, 1].max() > 0.1
        self._calm = 0 if busy else self._calm + 1
        return 1 if (busy or self._calm < self.calm_ticks) else self.base


# ---------------------------------------------------------------------------------- placement
class Placement:
    clairvoyant = False

    def __init__(self, forecast, polling, min_gain: float = 60.0, headroom: float = HEADROOM, name: str | None = None):
        self.forecast, self.polling, self.min_gain, self.headroom = forecast, polling, min_gain, headroom
        self.clairvoyant = isinstance(forecast, TrueForecast)
        self.name = name or f"{forecast.name} placement, {polling.name}"

    def reset(self, cfg: SimConfig, seed: int) -> None:
        self.polling.reset()
        if hasattr(self.forecast, "reset"):
            self.forecast.reset()

    def _cost(self, p: np.ndarray, peak: np.ndarray, assign: np.ndarray, cfg: SimConfig) -> tuple[float, float]:
        over, worst = 0.0, 0.0
        for l in range(cfg.n_links):
            load = (p[assign == l] * peak[assign == l, None]).sum(axis=0) + cfg.bg_mean_mbps
            over += np.clip(load - self.headroom * cfg.capacity_mbps, 0, None).sum()
            worst = max(worst, load.max() / cfg.capacity_mbps)
        return over, worst

    def act(self, view: View) -> Action:
        cfg = view.cfg
        p = self.forecast.predict(view)                                   # (jobs, HORIZON)
        peak = np.where(view.flow_peak > 100, view.flow_peak, 500.0)      # prior until a burst has been seen
        assign = view.flow_link.copy()
        base, worst = self._cost(p, peak, assign, cfg)

        moves = []
        if base > 0:
            best, best_cost = None, base - self.min_gain
            for j in range(len(assign)):
                if view.t - view.last_move_t[j] < COOLDOWN or p[j].max() < 0.2:
                    continue
                for l in range(cfg.n_links):
                    if l == assign[j]:
                        continue
                    trial = assign.copy()
                    trial[j] = l
                    c, _ = self._cost(p, peak, trial, cfg)
                    if c < best_cost:
                        best, best_cost = (j, l), c
            if best:
                moves.append(best)
        return Action(self.polling.interval(view, base), moves)
