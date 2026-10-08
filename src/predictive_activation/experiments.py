"""Reproducible experiment pipeline.  `python -m predictive_activation.experiments [--quick]`

Seed discipline (no overlap, ever):
    train       0-59          forecaster training data
    validation  100-105       choose each controller's headroom (the only tuned knob)
    test        1000-1029     everything reported in the README (30 paired episodes)
Every policy sees the *same* test episodes (same job jitter, same noise stream), so comparisons are paired.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from .config import SimConfig
from .dataset import collect, collect_flows
from .engine import run_episode
from .forecasters import ScoreBaseline, SkForecaster
from .policies import (ACTIVE_FRAC, AdaptivePoll, FixedPoll, LearnedFlowForecast, NoControl, PersistenceForecast,
                       Placement, ThresholdReactive, TrueForecast)

RESULTS = Path(__file__).resolve().parents[2] / "results"
_STATE: dict = {}


def make_policy(spec: dict):
    kind = spec["kind"]
    if kind == "none":
        return NoControl(spec.get("poll", 30))
    if kind == "threshold":
        return ThresholdReactive(spec["poll"])
    polling = AdaptivePoll(spec.get("base", 10)) if spec.get("adaptive") else FixedPoll(spec.get("poll", 1))
    forecast = {"reactive": PersistenceForecast, "true": TrueForecast}.get(kind)
    forecast = forecast() if forecast else LearnedFlowForecast(_STATE["flow_model"])
    return Placement(forecast, polling, headroom=spec.get("hr", 0.92), name=spec["name"])


def _init_worker(flow_model):
    _STATE["flow_model"] = flow_model


def _run(job):
    spec, seed, cfg = job
    return run_episode(cfg, seed, make_policy(spec)).metrics


def evaluate(specs, seeds, cfg, procs=2):
    jobs = [(s, seed, cfg) for s in specs for seed in seeds]
    with mp.get_context("spawn").Pool(procs, initializer=_init_worker, initargs=(_STATE["flow_model"],)) as pool:
        out = pool.map(_run, jobs, chunksize=1)
    res = {}
    for i, s in enumerate(specs):
        res[s["name"]] = out[i * len(seeds):(i + 1) * len(seeds)]
    return res


def mean(res, name, key):
    return float(np.nanmean([m[key] for m in res[name]]))


def forecasting_benchmark(cfg, link_models, flow_model, quick):
    n_test = 6 if quick else 20
    out = {"link_onset": {}, "flow_activity": {}}
    for iv in (1, 5, 10):
        te = collect(cfg, range(1000, 1000 + n_test), interval=iv)
        rows = {}
        for m in link_models:
            r = m.risk(te.X)
            p, rc, _ = precision_recall_curve(te.y, r)[0:3]
            mask = p[:-1] >= 0.5
            rows[m.name] = {"pr_auc": float(average_precision_score(te.y, r)),
                            "roc_auc": float(roc_auc_score(te.y, r)),
                            "recall_at_precision_0.5": float(rc[:-1][mask].max()) if mask.any() else 0.0}
        rows["_prevalence"] = float(te.y.mean())
        out["link_onset"][str(iv)] = rows
        ft = collect_flows(cfg, range(1000, 1000 + n_test), interval=iv, stride=3)
        persist = (ft.X[:, 0] > ACTIVE_FRAC).astype(float)   # feature 0 = latest observed flow rate / capacity
        out["flow_activity"][str(iv)] = {
            "persistence_pr_auc": float(average_precision_score(ft.y, persist)),
            "gbm_pr_auc": float(average_precision_score(ft.y, flow_model.risk(ft.X))),
            "prevalence": float(ft.y.mean())}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="tiny run for CI / smoke tests")
    ap.add_argument("--procs", type=int, default=2)
    args = ap.parse_args()
    q = args.quick
    RESULTS.mkdir(exist_ok=True, parents=True)
    cfg = SimConfig(ticks=600 if q else 1800)
    t0 = time.time()
    log = lambda s: print(f"[{time.time() - t0:6.0f}s] {s}", flush=True)

    log("training forecasters")
    tr_seeds = range(0, 6 if q else 60)
    link_tr = collect(cfg, tr_seeds, stride=2)
    link_models = [ScoreBaseline("queue_last"), ScoreBaseline("util_last"),
                   SkForecaster("logreg").fit(link_tr.X, link_tr.y), SkForecaster("gbm").fit(link_tr.X, link_tr.y)]
    flow_tr = collect_flows(cfg, range(0, 6 if q else 40))
    flow_model = SkForecaster("gbm").fit(flow_tr.X, flow_tr.y)
    _STATE["flow_model"] = flow_model

    log("forecasting benchmark")
    fb = forecasting_benchmark(cfg, link_models, flow_model, q)
    (RESULTS / "forecasting.json").write_text(json.dumps(fb, indent=2))

    val, test = range(100, 102 if q else 106), range(1000, 1003 if q else 1030)
    grid = (0.75, 0.92) if q else (0.65, 0.75, 0.85, 0.92)
    log("tuning headroom on validation seeds")
    chosen = {}
    for kind, poll in (("reactive", 1), ("learned", 5), ("true", 1)):
        specs = [dict(kind=kind, poll=poll, hr=h, name=f"{kind}-{h}") for h in grid]
        r = evaluate(specs, val, cfg, args.procs)
        best = min(grid, key=lambda h: mean(r, f"{kind}-{h}", "congested_pct") + 10 * mean(r, f"{kind}-{h}", "drop_pct"))
        chosen[kind] = best
        log(f"  {kind}: headroom={best}")
    (RESULTS / "tuning.json").write_text(json.dumps(chosen, indent=2))

    polls = (1, 5) if q else (1, 2, 5, 10, 20)
    specs = [dict(kind="none", poll=30, name="No control (poll 30s)"),
             dict(kind="threshold", poll=30, name="Threshold reroute (poll 30s)"),
             dict(kind="threshold", poll=1, name="Threshold reroute (poll 1s)")]
    for p in polls:
        specs.append(dict(kind="reactive", poll=p, hr=chosen["reactive"], name=f"Reactive balancer (poll {p}s)"))
        specs.append(dict(kind="learned", poll=p, hr=chosen["learned"], name=f"Predictive (poll {p}s)"))
    specs.append(dict(kind="true", poll=1, hr=chosen["true"], name="Clairvoyant (upper bound)"))
    specs.append(dict(kind="learned", adaptive=True, base=10, hr=chosen["learned"], name="Predictive (adaptive poll 10s/1s)"))
    specs.append(dict(kind="reactive", adaptive=True, base=10, hr=chosen["reactive"], name="Reactive balancer (adaptive poll 10s/1s)"))
    log(f"main comparison: {len(specs)} policies x {len(test)} episodes")
    main_res = evaluate(specs, test, cfg, args.procs)
    (RESULTS / "main.json").write_text(json.dumps({"seeds": list(test), "specs": specs, "metrics": main_res}, indent=2))

    if not q:
        log("sensitivity sweeps")
        sens = {}
        sseeds = range(1000, 1010)
        variants = {"control_delay": [("1 tick", dict(control_delay=1)), ("2 ticks (default)", {}), ("5 ticks", dict(control_delay=5))],
                    "telemetry_noise": [("3% (default)", {}), ("10%", dict(telemetry_noise=0.10)), ("20%", dict(telemetry_noise=0.20))],
                    "load": [("0.8x", dict(load_scale=0.8)), ("1.0x (default)", {}), ("1.2x", dict(load_scale=1.2))]}
        sspecs = [dict(kind="none", poll=30, name="No control"),
                  dict(kind="reactive", poll=1, hr=chosen["reactive"], name="Reactive balancer (poll 1s)"),
                  dict(kind="learned", poll=5, hr=chosen["learned"], name="Predictive (poll 5s)")]
        for axis, vs in variants.items():
            sens[axis] = {}
            for label, kw in vs:
                c2 = SimConfig(**kw)
                r = evaluate(sspecs, sseeds, c2, args.procs)
                sens[axis][label] = {n: {k: mean(r, n, k) for k in ("congested_pct", "drop_pct", "mean_slowdown")} for n in r}
                log(f"  {axis} {label}")
        (RESULTS / "sensitivity.json").write_text(json.dumps(sens, indent=2))
    log("done")


if __name__ == "__main__":
    main()
