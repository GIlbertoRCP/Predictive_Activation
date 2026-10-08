"""Closed-loop episode runner: fabric + sampled telemetry + a control policy.

The policy never sees ground truth. It sees (a) zero-order-held samples from the last poll, (b) the age
of those samples, and (c) per-flow rate stats from the same poll, i.e. what an OpenFlow controller
actually receives from PortStats / FlowStats replies. Only the clairvoyant oracle gets a fabric handle.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from .config import SimConfig
from .fabric import Fabric, TickRecord
from .features import sliding_windows

WINDOW = 120  # ticks of telemetry history exposed to policies


@dataclass
class Action:
    poll_interval: int
    moves: list = field(default_factory=list)  # [(job_index, target_link)]


@dataclass
class View:
    t: int
    polled: bool
    windows: np.ndarray          # (L, WINDOW, 4) zero-order-held telemetry, newest last
    age: np.ndarray              # (L,) ticks since the last real sample
    interval: int                # polling interval currently in force
    flow_peak: np.ndarray        # (n_jobs,) decayed max of polled per-flow rates (Mbps)
    flow_rate: np.ndarray        # (n_jobs,) per-flow rate at the last poll (Mbps)
    flow_windows: np.ndarray     # (n_jobs, WINDOW) zero-order-held per-flow rate / capacity
    flow_link: np.ndarray        # (n_jobs,) link each flow is (planned to be) on
    last_move_t: np.ndarray      # (n_jobs,) tick of the last move (for cooldowns)
    cfg: SimConfig
    fabric: Fabric | None = None  # populated only for clairvoyant policies


class Policy(Protocol):
    name: str
    clairvoyant: bool

    def reset(self, cfg: SimConfig, seed: int) -> None: ...
    def act(self, view: View) -> Action: ...


@dataclass
class EpisodeResult:
    metrics: dict
    trace: dict | None = None


def _channels(rec: TickRecord, cfg: SimConfig) -> np.ndarray:
    util = rec.offered / cfg.capacity_mbps
    queue = rec.queue / cfg.buffer_mb
    drop = rec.drops / np.maximum(rec.offered, 1e-9)
    lat = rec.latency_ms / 100.0
    return np.stack([util, queue, drop, lat], axis=1)  # (L, 4)


def run_episode(cfg: SimConfig, seed: int, policy: Policy, record: bool = False) -> EpisodeResult:
    fab = Fabric(cfg, seed)
    rng = np.random.default_rng(seed + 7_919)
    L, n_jobs, T = cfg.n_links, len(cfg.jobs), cfg.ticks
    policy.reset(cfg, seed)

    hist = np.zeros((T + WINDOW, L, 4), np.float32)   # zero-order-held observations, left-padded
    ages = np.zeros((T, L))
    intervals = np.zeros(T, int)
    fhist = np.zeros((T + WINDOW, n_jobs), np.float32)
    demand_truth = np.zeros((T, n_jobs))
    flow_peak = np.zeros(n_jobs)
    flow_rate = np.zeros(n_jobs)
    flow_link = fab.link_of.copy()
    last_move = np.full(n_jobs, -10_000)
    held = np.zeros((L, 4), np.float32)
    fheld = np.zeros(n_jobs, np.float32)
    last_poll = -10_000
    interval = 1
    polls = 0

    truth_cong = np.zeros((T, L), bool)
    truth_queue = np.zeros((T, L))
    offered_sum = drops_sum = 0.0
    lat_all = []
    poll_ticks, move_ticks = [], []
    moves_total = flaps = 0
    last_job_move_t = np.full(n_jobs, -10_000)

    for t in range(T):
        rec = fab.step()
        truth_cong[t] = rec.congested
        truth_queue[t] = rec.queue
        offered_sum += rec.offered.sum()
        drops_sum += rec.drops.sum()
        lat_all.append(rec.latency_ms.copy())

        polled = (t - last_poll) >= interval
        if polled:
            noise = 1.0 + rng.normal(0.0, cfg.telemetry_noise, size=(L, 4))
            held = np.clip(_channels(rec, cfg) * noise, 0.0, None).astype(np.float32)
            rate = rec.demand * (1.0 + rng.normal(0.0, cfg.telemetry_noise, size=n_jobs))
            flow_rate = np.clip(rate, 0.0, None)
            flow_peak = np.maximum(flow_rate, 0.98 * flow_peak)
            fheld = (flow_rate / cfg.capacity_mbps).astype(np.float32)
            last_poll = t
            polls += 1
            poll_ticks.append(t)
        hist[t + WINDOW - 1] = held
        fhist[t + WINDOW - 1] = fheld
        demand_truth[t] = rec.demand
        ages[t] = t - last_poll
        intervals[t] = interval

        view = View(
            t=t, polled=polled,
            windows=hist[t : t + WINDOW].transpose(1, 0, 2),
            age=np.full(L, t - last_poll), interval=interval,
            flow_peak=flow_peak, flow_rate=flow_rate, flow_windows=fhist[t : t + WINDOW].T, flow_link=flow_link.copy(), last_move_t=last_move.copy(),
            cfg=cfg, fabric=fab if getattr(policy, "clairvoyant", False) else None,
        )
        act = policy.act(view)
        interval = int(max(1, act.poll_interval))
        for job, link in act.moves:
            if flow_link[job] == link:
                continue
            if t - last_job_move_t[job] < 30:
                flaps += 1
            fab.schedule_move(job, link)
            flow_link[job] = link
            last_move[job] = t
            last_job_move_t[job] = t
            moves_total += 1
            move_ticks.append((t, job, link))

    lat = np.concatenate(lat_all)
    slow = [ct / ideal for (_, ct, ideal) in fab.iterations]
    metrics = {
        "congested_pct": 100.0 * truth_cong.mean(),
        "drop_pct": 100.0 * drops_sum / max(offered_sum, 1e-9),
        "p99_latency_ms": float(np.percentile(lat, 99)),
        "mean_slowdown": float(np.mean(slow)) if slow else float("nan"),
        "p95_slowdown": float(np.percentile(slow, 95)) if slow else float("nan"),
        "iterations": len(slow),
        "poll_msgs_per_min": polls * L / (T / 60.0),
        "reroutes": moves_total,
        "flaps": flaps,
    }
    trace = None
    if record:
        trace = {
            "congested": truth_cong, "queue": truth_queue, "poll_ticks": np.array(poll_ticks),
            "moves": move_ticks, "windows_hist": hist, "ages": ages, "intervals": intervals,
            "flow_hist": fhist, "demand": demand_truth,
        }
    return EpisodeResult(metrics, trace)
