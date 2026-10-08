"""Turn results/*.json into figures and results/RESULTS.md.  `python -m predictive_activation.report`"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import SimConfig
from .engine import run_episode
from .experiments import RESULTS, _STATE, make_policy
from .dataset import collect_flows
from .forecasters import SkForecaster

INK, INK2, MUTED, SURF = "#0b0b0b", "#52514e", "#898781", "#fcfcfb"
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#898781"
FIGS = RESULTS / "figures"

plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
                     "text.color": INK, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.edgecolor": "#d9d8d3", "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#ececE8", "grid.linewidth": 0.8, "font.size": 10.5,
                     "axes.titleweight": "bold", "axes.titlesize": 11.5, "axes.titlelocation": "left"})


def ci(x):
    x = np.asarray(x, float)
    return x.mean(), 1.96 * x.std(ddof=1) / np.sqrt(len(x))


def col(res, name, key):
    return [m[key] for m in res[name]]


def paired(res, a, b, key):
    d = np.array(col(res, a, key)) - np.array(col(res, b, key))
    m, h = d.mean(), 1.96 * d.std(ddof=1) / np.sqrt(len(d))
    return m, h


def fig_pareto(main):
    res = main["metrics"]
    polls = sorted({s["poll"] for s in main["specs"] if s["kind"] in ("reactive", "learned") and not s.get("adaptive")})
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.1))
    for ax, key, title in zip(axes, ("congested_pct", "drop_pct", "mean_slowdown"),
                              ("Link-time congested (%)", "Traffic dropped (%)", "Iteration slowdown (x ideal)")):
        for kind, color, label, prefix in (("reactive", ORANGE, "Reactive balancer", "Reactive balancer"),
                                           ("learned", BLUE, "Predictive (learned)", "Predictive")):
            xs, ys, es = [], [], []
            for p in polls:
                name = f"{prefix} (poll {p}s)"
                m, h = ci(col(res, name, key))
                xs.append(np.mean(col(res, name, "poll_msgs_per_min"))); ys.append(m); es.append(h)
            ax.errorbar(xs, ys, yerr=es, color=color, marker="o", ms=6, lw=2, capsize=3, label=label,
                        markeredgecolor=SURF, markeredgewidth=1.5)
            ax.annotate(f"{polls[0]}s", (xs[0], ys[0]), textcoords="offset points", xytext=(-4, 9), fontsize=8.5, color=INK2, ha="right")
            ax.annotate(f"{polls[-1]}s", (xs[-1], ys[-1]), textcoords="offset points", xytext=(4, 9), fontsize=8.5, color=INK2)
        nm = ci(col(res, "No control (poll 30s)", key))[0]
        cv = ci(col(res, "Clairvoyant (upper bound)", key))[0]
        ax.axhline(nm, color=GRAY, ls="--", lw=1.4); ax.axhline(cv, color=AQUA, ls=":", lw=1.8)
        ax.text(ax.get_xlim()[1], nm, " no control", color=INK2, fontsize=8.5, va="bottom", ha="right")
        ax.text(ax.get_xlim()[1], cv, " clairvoyant", color=INK2, fontsize=8.5, va="bottom", ha="right")
        ax.set_xscale("log"); ax.set_xlabel("Telemetry cost: polling requests / min (log)"); ax.set_title(title)
    axes[0].legend(frameon=False, loc="upper left", fontsize=9)
    fig.suptitle("Same controller logic, different forecast: prediction buys accuracy at a fraction of the telemetry",
                 x=0.012, ha="left", fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(FIGS / "pareto.png", dpi=170); plt.close(fig)


def fig_forecast(fc):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.9))
    ax = axes[0]
    names = ["threshold on util_last", "Logistic regression", "Gradient-boosted trees"]
    cols = [GRAY, ORANGE, BLUE]
    ivs = ["1", "5", "10"]
    w = 0.26
    for i, (n, c) in enumerate(zip(names, cols)):
        v = [fc["link_onset"][iv][n]["pr_auc"] for iv in ivs]
        ax.bar(np.arange(3) + (i - 1) * w, v, w * 0.92, color=c, label=n.replace("threshold on util_last", "Utilization threshold"))
    prev = fc["link_onset"]["1"]["_prevalence"]
    ax.axhline(prev, color=INK2, ls="--", lw=1.2); ax.text(2.45, prev + 0.01, "random guess", fontsize=8.5, color=INK2, ha="right")
    ax.set_xticks(range(3)); ax.set_xticklabels([f"poll {i}s" for i in ivs]); ax.set_ylim(0, 1)
    ax.set_title("Link congestion 10 s ahead (PR-AUC)"); ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    ax = axes[1]
    pa = [fc["flow_activity"][iv]["persistence_pr_auc"] for iv in ivs]
    gb = [fc["flow_activity"][iv]["gbm_pr_auc"] for iv in ivs]
    ax.bar(np.arange(3) - 0.2, pa, 0.38, color=ORANGE, label="Persistence (reactive view)")
    ax.bar(np.arange(3) + 0.2, gb, 0.38, color=BLUE, label="Learned forecaster")
    ax.axhline(fc["flow_activity"]["1"]["prevalence"], color=INK2, ls="--", lw=1.2)
    ax.set_xticks(range(3)); ax.set_xticklabels([f"poll {i}s" for i in ivs]); ax.set_ylim(0, 1)
    ax.set_title("Will this flow be bursting in 2-10 s? (PR-AUC)"); ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    fig.tight_layout(); fig.savefig(FIGS / "forecasting.png", dpi=170); plt.close(fig)


def fig_sensitivity(sens):
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8), sharey=True)
    series = [("No control", GRAY), ("Reactive balancer (poll 1s)", ORANGE), ("Predictive (poll 5s)", BLUE)]
    for ax, (axis, title) in zip(axes, (("control_delay", "Control-plane delay"), ("telemetry_noise", "Telemetry noise"), ("load", "Offered load"))):
        labels = list(sens[axis])
        for i, (n, c) in enumerate(series):
            ax.bar(np.arange(len(labels)) + (i - 1) * 0.27, [sens[axis][l][n]["congested_pct"] for l in labels], 0.25, color=c,
                   label=n.replace(" (poll 1s)", ", 1s").replace(" (poll 5s)", ", 5s"))
        ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=8.5); ax.set_title(title)
    axes[0].set_ylabel("Link-time congested (%)"); axes[0].legend(frameon=False, fontsize=8.5)
    fig.tight_layout(); fig.savefig(FIGS / "sensitivity.png", dpi=170); plt.close(fig)


def fig_timeline(cfg):
    flow_tr = collect_flows(cfg, range(0, 40))
    _STATE["flow_model"] = SkForecaster("gbm").fit(flow_tr.X, flow_tr.y)
    tuning = json.loads((RESULTS / "tuning.json").read_text())
    seed = 1003
    runs = [("No control", dict(kind="none", poll=30, name="none"), GRAY),
            ("Reactive balancer, 1 s polling", dict(kind="reactive", poll=1, hr=tuning["reactive"], name="r"), ORANGE),
            ("Predictive, 5 s polling", dict(kind="learned", poll=5, hr=tuning["learned"], name="p"), BLUE)]
    fig, axes = plt.subplots(3, 1, figsize=(11.5, 6.0), sharex=True, sharey=True)
    for ax, (title, spec, color) in zip(axes, runs):
        res = run_episode(cfg, seed, make_policy(spec), record=True)
        q = res.trace["queue"].max(axis=1) / cfg.buffer_mb * 100
        t = np.arange(len(q))
        ax.fill_between(t, 0, q, color=color, alpha=0.9, lw=0)
        ax.axhline(cfg.congestion_frac * 100, color=INK2, ls="--", lw=1)
        for (tm, job, link) in res.trace["moves"]:
            ax.plot([tm], [-6], marker="v", color=INK2, ms=4, clip_on=False)
        ax.set_ylabel("worst queue %"); ax.set_xlim(0, 600 if True else len(q)); ax.set_ylim(0, 105)
        ax.set_title(f"{title} - {res.metrics['congested_pct']:.1f}% link-time congested, {res.metrics['reroutes']} reroutes", fontsize=10.5)
    axes[-1].set_xlabel("time (s)"); axes[0].text(598, 52, "SLO line (50% of buffer)", ha="right", fontsize=8.5, color=INK2)
    fig.tight_layout(); fig.savefig(FIGS / "timeline.png", dpi=170); plt.close(fig)


def write_markdown(main, fc, sens, tuning):
    res = main["metrics"]
    names = [s["name"] for s in main["specs"]]
    rows = ["| Policy | Congested link-time % | Dropped % | Slowdown | Polls/min | Reroutes |", "|---|---|---|---|---|---|"]
    for n in names:
        f = lambda k, d=2: "%.*f ± %.*f" % ((d,) + (ci(col(res, n, k))[0],) + (d, ci(col(res, n, k))[1]))
        rows.append(f"| {n} | {f('congested_pct')} | {f('drop_pct', 3)} | {f('mean_slowdown', 3)} | {np.mean(col(res, n, 'poll_msgs_per_min')):.0f} | {np.mean(col(res, n, 'reroutes')):.0f} |")
    comps = [("Predictive (poll 5s)", "Reactive balancer (poll 1s)"), ("Predictive (poll 5s)", "Reactive balancer (poll 5s)"),
             ("Predictive (poll 5s)", "No control (poll 30s)")]
    lines = ["", "Paired differences (A - B, mean ± 95% CI over %d episodes; negative = A better):" % len(main["seeds"]), ""]
    for a, b in comps:
        parts = []
        for k, lab, d in (("congested_pct", "congested pp", 2), ("drop_pct", "dropped pp", 3), ("mean_slowdown", "slowdown", 3)):
            m, h = paired(res, a, b, k)
            parts.append(f"{lab} {m:+.{d}f} ± {h:.{d}f}")
        lines.append(f"- **{a}** vs **{b}**: " + "; ".join(parts))
    md = ["# Results (auto-generated by `python -m predictive_activation.report`)", "",
          f"30 held-out test episodes x 30 simulated minutes, 4 spine links, 8 training jobs. Headroom tuned on validation seeds: `{tuning}`.", "",
          *rows, *lines, ""]
    (RESULTS / "RESULTS.md").write_text("\n".join(md))


def main():
    FIGS.mkdir(parents=True, exist_ok=True)
    main_r = json.loads((RESULTS / "main.json").read_text())
    fc = json.loads((RESULTS / "forecasting.json").read_text())
    tuning = json.loads((RESULTS / "tuning.json").read_text())
    fig_pareto(main_r); fig_forecast(fc)
    if (RESULTS / "sensitivity.json").exists():
        sens = json.loads((RESULTS / "sensitivity.json").read_text())
        fig_sensitivity(sens)
    else:
        sens = None
    write_markdown(main_r, fc, sens, tuning)
    fig_timeline(SimConfig())


if __name__ == "__main__":
    main()
