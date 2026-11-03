"""Benchmark scaling policies across scenarios, seeds and pod startup delays.

    python -m predictive_activation.autoscale.evaluate            # full grid
    python -m predictive_activation.autoscale.evaluate --quick    # small grid for a smoke run

Each (scenario, seed) generates 14 days of traffic: forecasters train on the
first 7 days and every policy is scored on the last 7. Every policy is run at
several target utilizations, which traces a cost-vs-SLO curve; policies are
then compared at equal cost (pod-hours) by interpolating along those curves.

Outputs go to benchmarks/autoscale/: runs.csv (one row per simulation),
forecasts.csv, matched_cost.csv, summary.json and PNG charts.
"""

import argparse
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace

import numpy as np
import pandas as pd

from predictive_activation import config
from predictive_activation.autoscale import forecasters as fc
from predictive_activation.autoscale.policies import Predictive, ReactiveHPA, ReactiveRPS
from predictive_activation.autoscale.simulator import ClusterConfig, sample_arrivals, simulate
from predictive_activation.autoscale.traces import SCENARIOS, azure_trace, select_azure_apps, synthetic_trace

OUT_DIR = config.BENCHMARKS_DIR / "autoscale"
AZURE_OUT_DIR = config.BENCHMARKS_DIR / "autoscale_azure"
# The burstiest Azure apps peak at ~40x their mean, so the replica cap is raised for them.
AZURE_CLUSTER = ClusterConfig(max_replicas=1000)
TRAIN_DAYS, TEST_DAYS = 7, 7
WINDOW_STEPS = 4  # forecast the peak over the first minute a new pod is ready
TARGETS = (0.4, 0.5, 0.6, 0.7, 0.8)
STARTUPS_S = (15, 30, 60, 120, 180)
REFERENCE_POLICY, REFERENCE_TARGET = "reactive_rps", 0.6


def load_trace(scenario, seed, step_s):
    """Synthetic scenario (traffic shape depends on seed) or a selected Azure app
    named az_* (fixed shape; the seed only changes the Poisson arrivals)."""
    if scenario.startswith("az_"):
        trace = azure_trace(scenario)
    else:
        trace = synthetic_trace(scenario, days=TRAIN_DAYS + TEST_DAYS, seed=seed)
    return trace.resample(step_s)


def run_case(scenario, seed, startups=STARTUPS_S, targets=TARGETS, base_cfg=None):
    """All policies x targets x startup delays for one (scenario, seed)."""
    base_cfg = base_cfg or ClusterConfig()
    trace = load_trace(scenario, seed, base_cfg.step_s)
    arrivals = sample_arrivals(trace.rps, base_cfg.step_s, seed=seed)
    observed = arrivals / base_cfg.step_s
    cut = TRAIN_DAYS * trace.steps_per_day
    test_arrivals = arrivals[cut:]

    runs, forecast_rows = [], []
    for startup in startups:
        cfg = replace(base_cfg, startup_s=startup)
        lead = math.ceil(startup / cfg.step_s)
        y_true = fc.window_peak_target(observed, lead, WINDOW_STEPS)[cut:]

        forecasts = {}
        for f in (fc.LastValue(), fc.SeasonalNaive(), fc.Holt(), fc.GBMQuantile(seed=seed), fc.Oracle()):
            f.fit(observed[:cut], cfg.step_s, lead, WINDOW_STEPS)
            forecasts[f.name] = f.predict(observed, cfg.step_s, lead, WINDOW_STEPS)[cut:]
            forecast_rows.append({"scenario": scenario, "seed": seed, "startup_s": startup, "forecaster": f.name,
                                  **fc.evaluate_forecast(y_true, forecasts[f.name])})

        for target in targets:
            policies = [
                ReactiveHPA(target=target),
                ReactiveRPS(target=target),
                # Ablation: the predictive controller (no HPA rate limit) with a naive forecast.
                # Any gap between this and predictive_gbm is the value of forecasting itself.
                Predictive(forecasts["last_value"], target=target, name="predictive_naive"),
                Predictive(forecasts["holt"], target=target, name="predictive_holt"),
                Predictive(forecasts["gbm_q90"], target=target, name="predictive_gbm"),
                Predictive(forecasts["gbm_q90"], target=target, reactive_floor=True, name="hybrid_gbm"),
                Predictive(forecasts["oracle"], target=target, name="oracle"),
            ]
            for policy in policies:
                r = simulate(test_arrivals, policy, cfg, trace_name=scenario)
                runs.append({"scenario": scenario, "seed": seed, "startup_s": startup, "policy": policy.name,
                             "target": target, **r.metrics})
    return runs, forecast_rows


def matched_cost(runs, metric="slo_violation_min"):
    """Interpolate each policy's `metric` at the reference policy's cost.

    For every (scenario, seed, startup), the reference cost is the pod-hours of
    REFERENCE_POLICY at REFERENCE_TARGET. Each policy's (pod_hours, metric)
    points across targets are sorted by cost and linearly interpolated there.
    """
    rows = []
    for key, g in runs.groupby(["scenario", "seed", "startup_s"]):
        ref = g[(g.policy == REFERENCE_POLICY) & (g.target == REFERENCE_TARGET)]
        if ref.empty:
            continue
        ref_cost = float(ref.pod_hours.iloc[0])
        for policy, pg in g.groupby("policy"):
            pg = pg.sort_values("pod_hours")
            in_range = pg.pod_hours.min() <= ref_cost <= pg.pod_hours.max()
            value = float(np.interp(ref_cost, pg.pod_hours, pg[metric])) if in_range else np.nan
            rows.append(dict(zip(["scenario", "seed", "startup_s"], key), policy=policy,
                             ref_pod_hours=ref_cost, **{metric: value}))
    return pd.DataFrame(rows)


def headline_startup(runs):
    """60 s if it was simulated, otherwise the middle startup delay of the grid."""
    startups = sorted(runs.startup_s.unique())
    return 60 if 60 in startups else int(startups[len(startups) // 2])


def plot(runs, matched, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    policies = ["hpa_cpu", "reactive_rps", "predictive_naive", "predictive_holt", "predictive_gbm", "hybrid_gbm", "oracle"]
    colors = dict(zip(policies, ["#9e9e9e", "#5b8def", "#b39ddb", "#f4a259", "#2a9d8f", "#264653", "#c0392b"]))
    scenarios = list(dict.fromkeys(runs.scenario))
    ncols = min(4, len(scenarios))
    nrows = math.ceil(len(scenarios) / ncols)

    def grid(height):
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, height * nrows), squeeze=False)
        flat = axes.ravel()
        for ax in flat[len(scenarios):]:
            ax.set_visible(False)
        return fig, flat

    # 1. Cost vs SLO frontier per scenario (mean over seeds)
    fig, flat = grid(3.8)
    startup_for_frontier = headline_startup(runs)
    sub = runs[runs.startup_s == startup_for_frontier]
    for ax, scen in zip(flat, scenarios):
        s = sub[sub.scenario == scen].groupby(["policy", "target"])[["pod_hours", "slo_violation_min"]].mean().reset_index()
        for p in policies:
            pts = s[s.policy == p].sort_values("pod_hours")
            if not pts.empty:
                ax.plot(pts.pod_hours, pts.slo_violation_min, marker="o", ms=3, lw=1.5, color=colors[p], label=p,
                        ls="--" if p == "oracle" else "-")
        ax.set_title(scen)
        ax.set_xlabel("pod-hours (7 days)")
        ax.set_yscale("symlog", linthresh=1)
        ax.grid(alpha=0.3)
    flat[0].set_ylabel("SLO-violation minutes")
    flat[len(scenarios) - 1].legend(fontsize=7, loc="upper right")
    fig.suptitle(f"Cost vs SLO violations, pod startup {startup_for_frontier}s (lower-left is better)")
    fig.tight_layout()
    fig.savefig(out_dir / "frontier.png", dpi=130)
    plt.close(fig)

    # 2. Violation minutes at matched cost vs startup delay
    fig, flat = grid(3.6)
    m = matched.groupby(["scenario", "startup_s", "policy"]).slo_violation_min.mean().reset_index()
    for ax, scen in zip(flat, scenarios):
        s = m[m.scenario == scen]
        for p in policies:
            pts = s[s.policy == p].sort_values("startup_s")
            if not pts.empty:
                ax.plot(pts.startup_s, pts.slo_violation_min, marker="o", ms=3, color=colors[p], label=p,
                        ls="--" if p == "oracle" else "-")
        ax.set_title(scen)
        ax.set_xlabel("pod startup delay (s)")
        ax.grid(alpha=0.3)
    flat[0].set_ylabel(f"SLO-violation min @ cost of {REFERENCE_POLICY} t={REFERENCE_TARGET}")
    flat[len(scenarios) - 1].legend(fontsize=7)
    fig.suptitle("When does prediction pay off? Violations at equal cost vs pod startup delay")
    fig.tight_layout()
    fig.savefig(out_dir / "startup_sweep.png", dpi=130)
    plt.close(fig)


def plot_timeline(out_dir, scenario="flash_crowd", seed=0, startup=60, target=0.6, day=2, span_min=90, base_cfg=None):
    """Zoom on the biggest spike of one test day: load vs ready capacity and p95 latency."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = replace(base_cfg or ClusterConfig(), startup_s=startup)
    trace = load_trace(scenario, seed, cfg.step_s)
    arrivals = sample_arrivals(trace.rps, cfg.step_s, seed=seed)
    observed = arrivals / cfg.step_s
    cut = TRAIN_DAYS * trace.steps_per_day
    lead = math.ceil(startup / cfg.step_s)
    gbm = fc.GBMQuantile(seed=seed).fit(observed[:cut], cfg.step_s, lead, WINDOW_STEPS)
    forecast = gbm.predict(observed, cfg.step_s, lead, WINDOW_STEPS)[cut:]

    results = {p.name: simulate(arrivals[cut:], p, cfg) for p in
               (ReactiveRPS(target=target), Predictive(forecast, target=target, name="predictive_gbm"))}
    spd = trace.steps_per_day
    test_obs = observed[cut:]
    peak = day * spd + int(np.argmax(test_obs[day * spd:(day + 1) * spd]))
    half = span_min * 60 // cfg.step_s // 2
    sl = slice(max(0, peak - half // 3), peak + 2 * half - half // 3)
    minutes = (np.arange(sl.start, sl.stop) - peak) * cfg.step_s / 60

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 5.5), sharex=True)
    ax1.plot(minutes, test_obs[sl], color="#444", lw=0.8, label="request rate")
    for name, color in (("reactive_rps", "#5b8def"), ("predictive_gbm", "#2a9d8f")):
        r = results[name]
        ax1.plot(minutes, r.ready[sl] * cfg.pod_capacity_rps, color=color, lw=1.2, label=f"{name} capacity")
        ax2.plot(minutes, r.p95_ms[sl], color=color, lw=1, label=name)
    ax1.set_ylabel("requests / s")
    ax1.legend(fontsize=8)
    ax2.axhline(cfg.slo_p95_ms, color="#c0392b", ls="--", lw=1, label="SLO")
    ax2.set_yscale("log")
    ax2.set_ylabel("p95 latency (ms)")
    ax2.set_xlabel("minutes from spike peak")
    ax2.legend(fontsize=8)
    fig.suptitle(f"{scenario}: largest spike of test day {day + 1}, pod startup {startup}s, target utilization {target}")
    fig.tight_layout()
    fig.savefig(out_dir / "timeline.png", dpi=130)
    plt.close(fig)


def summarize(runs, forecasts, matched, base_cfg=None):
    hs = headline_startup(runs)
    at_ref = matched[matched.startup_s == hs].groupby(["scenario", "policy"]).slo_violation_min.mean().unstack()
    return {
        "reference": f"{REFERENCE_POLICY} at target {REFERENCE_TARGET}",
        "headline_startup_s": hs,
        "violation_min_at_matched_cost": at_ref.round(2).to_dict(orient="index"),
        "forecast_metrics": forecasts[forecasts.startup_s == hs]
        .groupby(["scenario", "forecaster"])[["mae", "mape_pct", "under_forecast_pct", "pinball_q90"]]
        .mean().round(3).reset_index().to_dict(orient="records"),
        "cluster": asdict(base_cfg or ClusterConfig()),
        "grid": {"scenarios": sorted(runs.scenario.unique()), "seeds": sorted(int(s) for s in runs.seed.unique()),
                 "startups_s": sorted(int(s) for s in runs.startup_s.unique()),
                 "targets": sorted(float(t) for t in runs.target.unique()),
                 "train_days": TRAIN_DAYS, "test_days": TEST_DAYS},
    }


def main(scenarios=SCENARIOS, seeds=(0, 1, 2), startups=STARTUPS_S, targets=TARGETS, out_dir=OUT_DIR, workers=None,
         base_cfg=None, timeline_scenario="flash_crowd"):
    out_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()
    cases = [(s, seed) for s in scenarios for seed in seeds]
    workers = workers or min(len(cases), os.cpu_count() or 1)
    runs, forecasts = [], []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_case, s, seed, startups, targets, base_cfg) for s, seed in cases]
        for f in futures:
            r, fr = f.result()
            runs += r
            forecasts += fr
    runs, forecasts = pd.DataFrame(runs), pd.DataFrame(forecasts)
    matched = matched_cost(runs)

    runs.to_csv(out_dir / "runs.csv", index=False)
    forecasts.to_csv(out_dir / "forecasts.csv", index=False)
    matched.to_csv(out_dir / "matched_cost.csv", index=False)
    summary = summarize(runs, forecasts, matched, base_cfg)
    summary["runtime_s"] = round(time.time() - start, 1)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    plot(runs, matched, out_dir)
    plot_timeline(out_dir, scenario=timeline_scenario, base_cfg=base_cfg)
    print(f"{len(runs)} simulations in {summary['runtime_s']}s -> {out_dir}")
    print(f"SLO-violation minutes at matched cost, startup {summary['headline_startup_s']}s:")
    print(pd.DataFrame(summary["violation_min_at_matched_cost"]).T.round(1).to_string())
    return runs, forecasts, matched


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="1 seed, 2 startups, 3 targets")
    parser.add_argument("--azure", action="store_true",
                        help="run on selected Azure Functions 2019 apps (needs data/azure_functions_2019, see AUTOSCALING.md)")
    args = parser.parse_args()
    kwargs = {}
    if args.azure:
        apps = list(select_azure_apps())
        # Timeline shows the cron-like app (bursts every 10 min): the clearest case for forecasting
        timeline = "az_busy_024" if "az_busy_024" in apps else apps[0]
        kwargs = dict(scenarios=apps, out_dir=AZURE_OUT_DIR, base_cfg=AZURE_CLUSTER, timeline_scenario=timeline)
    if args.quick:
        kwargs.update(seeds=(0,), startups=(30, 120), targets=(0.5, 0.6, 0.7))
    main(**kwargs)
