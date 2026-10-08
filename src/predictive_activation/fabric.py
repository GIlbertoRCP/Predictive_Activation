"""Fluid-queue model of the core links of an AI-cluster fabric.

Why a fluid model: the congestion that matters for distributed training is *incast / collision*
of synchronized all-reduce bursts on a shared core link. That is a queueing phenomenon with a
physically meaningful lead time (demand ramps up, the queue fills, then the buffer overflows), so it
can be forecast from telemetry, unlike the echo-RTT artefacts in the original Mininet capture.

Model, per 1-second tick, per link:
    offered  = sum(demand of jobs routed on the link) + background(AR(1))
    queue'   = clip(queue + offered - capacity, 0, buffer);  drops = overflow
    latency  = base + queue / capacity   (queueing delay)
Each job alternates a compute phase and a communication phase that must move a fixed volume. In the
communication phase it ramps to its peak rate; congestion slows it (bandwidth share) and drops force
retransmission, which stretches the iteration. That feedback is what we measure as "slowdown".
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import SimConfig


@dataclass
class TickRecord:
    """Ground truth for one tick (what the simulator knows; the controller only sees polled samples)."""

    t: int
    demand: np.ndarray        # (n_jobs,) Mbps offered by each job
    link_of: np.ndarray       # (n_jobs,) link each job is routed on this tick
    offered: np.ndarray       # (L,)
    served: np.ndarray        # (L,)
    queue: np.ndarray         # (L,) Mb
    drops: np.ndarray         # (L,) Mb dropped this tick
    latency_ms: np.ndarray    # (L,)
    congested: np.ndarray     # (L,) bool


@dataclass
class Fabric:
    cfg: SimConfig
    seed: int
    rng: np.random.Generator = field(init=False)

    def __post_init__(self) -> None:
        cfg = self.cfg
        self.rng = np.random.default_rng(self.seed)
        n, L = len(cfg.jobs), cfg.n_links
        self.peak = np.array([j.peak_mbps for j in cfg.jobs]) * cfg.load_scale
        self.volume = np.array([j.volume_mb for j in cfg.jobs]) * cfg.load_scale
        self.ideal_comm = np.array([j.comm_seconds + cfg.ramp_ticks / 2 for j in cfg.jobs])
        self.link_of = np.array([j.home_link for j in cfg.jobs], dtype=int)
        self.in_comm = np.zeros(n, dtype=bool)
        # Random initial phase so jobs are not synchronized at t=0.
        self.timer = np.array([self.rng.uniform(1.0, j.compute_mean_s + j.comm_seconds) for j in cfg.jobs])
        self.remaining = np.zeros(n)
        self.ramp = np.zeros(n)
        self.comm_elapsed = np.zeros(n)
        self.backoff = np.ones(n)  # AIMD-style rate factor: halves on loss, recovers additively
        self.queue = np.zeros(L)
        self.bg = np.full(L, cfg.bg_mean_mbps)
        self.t = 0
        self.pending: list[tuple[int, int, int]] = []
        self.iterations: list[tuple[int, float, float]] = []  # (job, comm_time, ideal_time)
        self.moves_applied = 0

    # ------------------------------------------------------------------ control plane
    def schedule_move(self, job: int, link: int) -> None:
        """Install a flow rule moving `job` to `link`; takes effect after the control delay."""
        self.pending.append((self.t + self.cfg.control_delay, int(job), int(link)))

    def _apply_pending(self) -> None:
        keep = []
        for t_eff, job, link in self.pending:
            if t_eff <= self.t:
                if self.link_of[job] != link:
                    self.link_of[job] = link
                    self.moves_applied += 1
            else:
                keep.append((t_eff, job, link))
        self.pending = keep

    # ------------------------------------------------------------------ data plane
    def _demand(self) -> np.ndarray:
        d = np.zeros(len(self.peak))
        for j in range(len(d)):
            if self.in_comm[j]:
                frac = min(1.0, (self.ramp[j] + 1) / self.cfg.ramp_ticks)
                d[j] = min(self.peak[j] * frac * self.backoff[j], max(self.remaining[j], 0.0))
            else:
                d[j] = 2.0  # idle control chatter
        return d

    def step(self) -> TickRecord:
        cfg = self.cfg
        L = cfg.n_links
        self._apply_pending()
        demand = self._demand()
        link_of = self.link_of.copy()

        self.bg = cfg.bg_mean_mbps + cfg.bg_rho * (self.bg - cfg.bg_mean_mbps) + self.rng.normal(0, cfg.bg_sigma_mbps, L)
        self.bg = np.clip(self.bg, 0.0, None)
        offered = np.array([demand[link_of == l].sum() for l in range(L)]) + self.bg

        total = self.queue + offered
        served = np.minimum(total, cfg.capacity_mbps)          # Mb per 1 s tick
        queue_new = np.clip(total - cfg.capacity_mbps, 0.0, None)
        drops = np.clip(queue_new - cfg.buffer_mb, 0.0, None)
        queue_new = np.minimum(queue_new, cfg.buffer_mb)
        self.queue = queue_new
        latency = cfg.base_latency_ms + queue_new / cfg.capacity_mbps * 1000.0
        congested = queue_new >= cfg.congestion_queue_mb

        # Per-job progress: bandwidth share of the link, minus retransmission of dropped data.
        for j in range(len(demand)):
            if not self.in_comm[j]:
                continue
            l = link_of[j]
            share = min(1.0, cfg.capacity_mbps / max(offered[l], 1e-9))
            drop_frac = drops[l] / max(offered[l], 1e-9)
            progress = demand[j] * share
            self.remaining[j] -= progress
            self.remaining[j] += demand[j] * drop_frac        # retransmit what was lost
            if drops[l] > 0:
                self.backoff[j] = max(0.25, self.backoff[j] * 0.5)
            else:
                self.backoff[j] = min(1.0, self.backoff[j] + 0.1)
            self.ramp[j] += 1
            self.comm_elapsed[j] += 1
            if self.remaining[j] <= 0:
                self.iterations.append((j, self.comm_elapsed[j], self.ideal_comm[j]))
                self.in_comm[j] = False
                spec = cfg.jobs[j]
                self.timer[j] = max(3.0, self.rng.normal(spec.compute_mean_s, spec.compute_jitter_s))

        for j in range(len(demand)):
            if not self.in_comm[j]:
                self.timer[j] -= 1
                if self.timer[j] <= 0:
                    self.in_comm[j] = True
                    self.remaining[j] = self.volume[j]
                    self.ramp[j] = 0
                    self.comm_elapsed[j] = 0

        rec = TickRecord(self.t, demand, link_of, offered, served, queue_new.copy(), drops, latency, congested)
        self.t += 1
        return rec

    # ------------------------------------------------------------------ clairvoyant access (oracle only)
    def future_job_load(self, horizon: int) -> np.ndarray:
        """(n_jobs, horizon) expected demand of each job over the next `horizon` ticks.

        Uses the true job timers. Only the clairvoyant upper-bound policy may call this.
        """
        n = len(self.peak)
        out = np.zeros((n, horizon))
        for j in range(n):
            spec = self.cfg.jobs[j]
            if self.in_comm[j]:
                dur = int(np.ceil(max(self.remaining[j], 0.0) / self.peak[j]))
                out[j, : min(horizon, dur)] = self.peak[j]
            else:
                start = int(np.ceil(max(self.timer[j], 0.0)))
                dur = int(np.ceil(spec.comm_seconds))
                if start < horizon:
                    out[j, start : min(horizon, start + dur)] = self.peak[j]
        return out
