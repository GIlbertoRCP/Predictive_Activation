"""Request-rate workload traces.

A trace is a request rate (requests/second) sampled every `step_s` seconds.
Synthetic scenarios cover the shapes autoscalers struggle with; the Azure
Functions 2019 loader provides real per-minute invocation counts.
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

DAY_S = 86_400
SCENARIOS = ("diurnal", "flash_crowd", "ramp", "noisy")


@dataclass
class Trace:
    name: str
    rps: np.ndarray  # requests/second at each step
    step_s: int
    meta: dict = field(default_factory=dict)

    @property
    def steps_per_day(self):
        return DAY_S // self.step_s

    def split(self, train_days):
        """Split into (train, test) traces at a day boundary."""
        cut = train_days * self.steps_per_day
        return (Trace(f"{self.name}[train]", self.rps[:cut], self.step_s, self.meta),
                Trace(f"{self.name}[test]", self.rps[cut:], self.step_s, self.meta))

    def resample(self, step_s):
        """Linear interpolation to a finer step (e.g. 60s data -> 15s decisions)."""
        if step_s == self.step_s:
            return self
        t_old = np.arange(len(self.rps)) * self.step_s
        t_new = np.arange(0, t_old[-1] + self.step_s, step_s)
        return Trace(self.name, np.interp(t_new, t_old, self.rps), step_s, self.meta)


def _ar1_noise(rng, n, sigma, phi):
    """Autocorrelated multiplicative noise (log-space AR(1)), mean ~1."""
    eps = rng.normal(0, sigma * np.sqrt(1 - phi**2), n)
    x = np.empty(n)
    x[0] = rng.normal(0, sigma)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + eps[i]
    return np.exp(x - sigma**2 / 2)


def _flash_crowds(rng, n, step_s, per_day, peak_range=(3.0, 8.0)):
    """Multiplier with sudden spikes: ~1-3 min rise, 10-40 min exponential decay."""
    mult = np.ones(n)
    days = n * step_s / DAY_S
    for _ in range(rng.poisson(per_day * days)):
        start = rng.integers(0, n)
        peak = rng.uniform(*peak_range)
        rise = max(1, int(rng.uniform(60, 180) / step_s))
        decay = rng.uniform(600, 2400) / step_s
        idx = np.arange(start, n)
        k = idx - start
        shape = np.where(k < rise, (k + 1) / rise, np.exp(-(k - rise) / decay))
        mult[idx] = np.maximum(mult[idx], 1 + (peak - 1) * shape)
    return mult


def synthetic_trace(scenario="diurnal", days=14, step_s=60, base_rps=400.0, seed=0):
    """Generate a synthetic request-rate trace.

    diurnal     daily + weekly cycle with mild noise
    flash_crowd diurnal plus a few sudden 3-8x spikes per day
    ramp        diurnal on top of steady growth (launch / viral product)
    noisy       weak seasonality, heavy autocorrelated noise
    """
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario {scenario!r}; choose from {SCENARIOS}")
    rng = np.random.default_rng(seed)
    n = days * DAY_S // step_s
    t = np.arange(n) * step_s
    hour = (t % DAY_S) / 3600
    weekday = (t // DAY_S) % 7

    # Two-hump business-day shape: peaks ~11:00 and ~20:00, trough ~04:00
    daily = (0.35
             + 0.45 * np.exp(-((hour - 11) ** 2) / 8)
             + 0.55 * np.exp(-((hour - 20) ** 2) / 6)
             + 0.10 * np.exp(-((hour - 14) ** 2) / 20))
    weekly = np.where(weekday >= 5, 0.7, 1.0)

    if scenario == "diurnal":
        rps = base_rps * daily * weekly * _ar1_noise(rng, n, 0.06, 0.9)
    elif scenario == "flash_crowd":
        rps = base_rps * daily * weekly * _ar1_noise(rng, n, 0.06, 0.9) * _flash_crowds(rng, n, step_s, per_day=2)
    elif scenario == "ramp":
        growth = 1 + 2.0 * (t / t[-1])  # 3x over the trace
        rps = base_rps * daily * weekly * growth * _ar1_noise(rng, n, 0.08, 0.9)
    else:  # noisy
        flat = 0.7 + 0.3 * daily / daily.max()
        rps = base_rps * flat * _ar1_noise(rng, n, 0.35, 0.97)

    return Trace(scenario, rps.astype(np.float64), step_s, {"seed": seed, "days": days, "base_rps": base_rps})


def load_azure_functions(data_dir, top_n_apps=20, days=None):
    """Aggregate the Azure Functions 2019 invocation trace into one request-rate series.

    Expects the extracted files `invocations_per_function_md.anon.d01.csv` ...
    from https://github.com/Azure/AzurePublicDataset (AzureFunctionsDataset2019).
    Each file is one day; columns '1'..'1440' are per-minute invocation counts.
    The `top_n_apps` busiest apps are summed so the result looks like one
    service's traffic rather than a single bursty function.
    """
    files = sorted(Path(data_dir).glob("invocations_per_function_md.anon.d*.csv"))
    if days:
        files = files[:days]
    if not files:
        raise FileNotFoundError(f"No invocations_per_function_md.anon.d*.csv files in {data_dir}")

    minute_cols = [str(i) for i in range(1, 1441)]
    per_day = [pd.read_csv(f, usecols=["HashApp"] + minute_cols).groupby("HashApp")[minute_cols].sum() for f in files]

    # Choose apps that appear every day, ranked by total volume
    common = set(per_day[0].index)
    for d in per_day[1:]:
        common &= set(d.index)
    totals = sum(d.loc[sorted(common)].sum(axis=1) for d in per_day)
    apps = totals.sort_values(ascending=False).index[:top_n_apps]

    counts = np.concatenate([d.loc[apps].sum(axis=0).to_numpy(dtype=float) for d in per_day])
    return Trace("azure_functions", counts / 60.0, 60, {"apps": len(apps), "days": len(files)})


# --- Azure per-app traces -----------------------------------------------------
# Parsing 1.9 GB of CSVs takes about a minute, so the busiest apps are cached once.

AZURE_DIR = Path("data/azure_functions_2019")
AZURE_CACHE = AZURE_DIR / "top_apps_per_minute.npz"


def build_azure_cache(data_dir=AZURE_DIR, top_n=200, out=None):
    """Cache per-minute invocation counts for the `top_n` busiest apps present on
    every day, with each app's dominant trigger type."""
    data_dir = Path(data_dir)
    files = sorted(data_dir.glob("invocations_per_function_md.anon.d*.csv"))
    if not files:
        raise FileNotFoundError(f"No invocations_per_function_md.anon.d*.csv files in {data_dir}")
    minute_cols = [str(i) for i in range(1, 1441)]
    per_day, triggers = [], []
    for f in files:
        df = pd.read_csv(f, usecols=["HashApp", "Trigger"] + minute_cols, dtype={c: "float32" for c in minute_cols})
        per_day.append(df.groupby("HashApp")[minute_cols].sum())
        triggers.append(df[["HashApp", "Trigger"]])
    common = set(per_day[0].index)
    for d in per_day[1:]:
        common &= set(d.index)
    common = sorted(common)
    totals = sum(d.loc[common].sum(axis=1) for d in per_day).sort_values(ascending=False)
    apps = totals.index[:top_n]
    counts = np.stack([np.concatenate([d.loc[a].to_numpy() for d in per_day]) for a in apps])
    trig = pd.concat(triggers)
    dominant = trig[trig.HashApp.isin(apps)].groupby("HashApp").Trigger.agg(lambda s: s.value_counts().index[0])
    out = Path(out or data_dir / AZURE_CACHE.name)
    np.savez_compressed(out, counts=counts, apps=np.array(apps), triggers=np.array([dominant[a] for a in apps]))
    return out


def select_azure_apps(cache=AZURE_CACHE, trigger="http", min_active_pct=95.0, busiest=8, burstiest=4):
    """Fixed selection rule (decided before looking at results):
    among apps with the given trigger and traffic in >= min_active_pct of minutes,
    take the `busiest` by volume and the `burstiest` by the 99.9th percentile of
    minute-to-minute jumps relative to the mean. Returns {name: per-minute counts}."""
    z = np.load(cache, allow_pickle=True)
    counts, apps, trig = z["counts"], z["apps"], z["triggers"]
    rows = []
    for rank, (app, x, t) in enumerate(zip(apps, counts, trig), start=1):
        if t != trigger or 100 * (x > 0).mean() < min_active_pct:
            continue
        rows.append((rank, app, x, np.percentile(np.abs(np.diff(x)), 99.9) / x.mean()))
    chosen = {}
    for rank, _app, x, _ in rows[:busiest]:  # rows are already in volume order
        chosen[f"az_busy_{rank:03d}"] = x
    busy_ranks = {r[0] for r in rows[:busiest]}
    bursty = [r for r in sorted(rows, key=lambda r: -r[3]) if r[0] not in busy_ranks]
    for rank, _app, x, _ in bursty[:burstiest]:
        chosen[f"az_burst_{rank:03d}"] = x
    return chosen


def azure_trace(name, cache=AZURE_CACHE, mean_rps=250.0):
    """One selected app as a Trace, rescaled to `mean_rps` (shape is preserved)."""
    x = select_azure_apps(cache)[name].astype(np.float64) / 60.0
    return Trace(name, x * (mean_rps / x.mean()), 60, {"source": "azure_functions_2019", "mean_rps": mean_rps})


if __name__ == "__main__":
    print(f"Cached to {build_azure_cache()}")
    for name, x in select_azure_apps().items():
        r = x / 60
        print(f"{name}: mean {r.mean():.1f} req/s, peak/mean {r.max() / r.mean():.1f}")
