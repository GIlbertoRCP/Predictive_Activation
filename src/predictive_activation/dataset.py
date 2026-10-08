"""Build labelled forecasting datasets from simulated episodes.

Task: given the telemetry a controller would actually hold at tick t for a link that is *not yet
congested*, predict whether the link becomes congested within the next HORIZON ticks (early warning).
Episodes are generated under random routing perturbations and random polling intervals so the
forecaster sees the staleness and the routing shifts it will encounter in closed loop.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import SimConfig
from .engine import WINDOW, Action, View, run_episode
from .features import FEATURE_NAMES, sliding_windows, window_features
from .policies import HORIZON


class Collector:
    """Data-collection policy: random polling-interval schedule + occasional random re-routes."""
    clairvoyant = False
    name = "collector"

    def __init__(self, interval: int | None = None, move_prob: float = 0.015):
        self.fixed, self.move_prob = interval, move_prob

    def reset(self, cfg: SimConfig, seed: int) -> None:
        self.rng = np.random.default_rng(seed + 31)
        self.interval = self.fixed or 1
        self.next_switch = 0

    def act(self, view: View) -> Action:
        if self.fixed is None and view.t >= self.next_switch:
            self.interval = int(self.rng.choice([1, 1, 2, 5, 10, 15]))
            self.next_switch = view.t + int(self.rng.integers(20, 80))
        moves = []
        if self.rng.random() < self.move_prob:
            j = int(self.rng.integers(len(view.flow_link)))
            if view.t - view.last_move_t[j] >= 15:
                others = [l for l in range(view.cfg.n_links) if l != view.flow_link[j]]
                moves.append((j, int(self.rng.choice(others))))
        return Action(self.interval, moves)


@dataclass
class Dataset:
    X: np.ndarray        # (N, F)
    y: np.ndarray        # (N,) onset within HORIZON
    seed: np.ndarray
    tick: np.ndarray
    link: np.ndarray
    interval: np.ndarray
    cong_now: np.ndarray  # truth: already congested at t (excluded from X/y; kept for audit)


def collect(cfg: SimConfig, seeds, interval: int | None = None, stride: int = 1) -> Dataset:
    Xs, ys, ss, ts, ls, iv, cn = [], [], [], [], [], [], []
    for seed in seeds:
        res = run_episode(cfg, seed, Collector(interval), record=True)
        tr = res.trace
        cong, hist, ages, intervals = tr["congested"], tr["windows_hist"], tr["ages"], tr["intervals"]
        T, L = cong.shape
        ticks = np.arange(WINDOW // 2, T - HORIZON, stride)
        for l in range(L):
            fut = np.stack([cong[ticks + k + 1, l] for k in range(HORIZON)], axis=1).any(axis=1)
            keep = ~cong[ticks, l]                               # early-warning task: not yet congested
            tk = ticks[keep]
            win = sliding_windows(hist[WINDOW - 1:, l, :], tk, WINDOW)
            X = window_features(win, ages[tk, l], intervals[tk])
            Xs.append(X); ys.append(fut[keep])
            ss.append(np.full(len(tk), seed)); ts.append(tk); ls.append(np.full(len(tk), l))
            iv.append(intervals[tk]); cn.append(np.zeros(len(tk), bool))
    cat = np.concatenate
    return Dataset(cat(Xs), cat(ys).astype(int), cat(ss), cat(ts), cat(ls), cat(iv), cat(cn))


def collect_flows(cfg: SimConfig, seeds, interval: int | None = None, stride: int = 2) -> Dataset:
    """Per-flow activity dataset: P(flow j demands > 30% of capacity at t+k), k in FLOW_HORIZONS."""
    from .policies import ACTIVE_FRAC, FLOW_HORIZONS

    Xs, ys = [], []
    for seed in seeds:
        tr = run_episode(cfg, seed, Collector(interval), record=True).trace
        flow_hist, ages, intervals, demand = tr["flow_hist"], tr["ages"], tr["intervals"], tr["demand"]
        T, n = demand.shape
        ticks = np.arange(WINDOW // 2, T - max(FLOW_HORIZONS), stride)
        for j in range(n):
            series = np.zeros((T, 4), np.float32)
            series[:, 0] = flow_hist[WINDOW - 1:WINDOW - 1 + T, j]
            win = sliding_windows(series, ticks, WINDOW)
            base = window_features(win, ages[ticks, 0], intervals[ticks])
            for k in FLOW_HORIZONS:
                y = demand[ticks + k, j] > ACTIVE_FRAC * cfg.capacity_mbps
                Xs.append(np.hstack([base, np.full((len(ticks), 1), k, np.float32)])); ys.append(y)
    X, y = np.concatenate(Xs), np.concatenate(ys).astype(int)
    z = np.zeros(len(y), int)
    return Dataset(X, y, z, z, z, z, z.astype(bool))
