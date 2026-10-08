"""Feature engineering for congestion-onset forecasting from (possibly stale) switch telemetry.

A telemetry window has shape (N, W, 4) with channels:
    0 util        offered load / capacity          (rx_mbps analogue)
    1 queue       queue depth / buffer             (queue-depth analogue)
    2 drop        dropped / offered                (rx_loss analogue)
    3 latency     queueing delay in units of 100 ms
Samples are zero-order-held between polls, so `age` (ticks since the last real sample) and the
current polling `interval` are explicit features: the model knows how stale its input is.
"""
from __future__ import annotations

import numpy as np

CHANNELS = ("util", "queue", "drop", "latency")
BURST_THRESHOLD = 0.30  # util above this == an all-reduce burst is on this link

FEATURE_NAMES = (
    "util_last", "queue_last", "drop_last", "latency_last",
    "util_mean5", "util_mean15", "util_meanW",
    "util_slope3", "util_slope8", "queue_slope3", "queue_slope8",
    "util_maxW", "queue_maxW", "queue_max10",
    "in_burst", "since_onset", "burst_len", "since_burst_end",
    "onset_interval", "next_onset_eta", "n_onsets",
    "age", "interval",
)


def _last_index(mask: np.ndarray) -> np.ndarray:
    """Index of the last True along axis 1, or -1 if none. mask: (N, W) bool."""
    W = mask.shape[1]
    idx = np.where(mask, np.arange(W)[None, :], -1)
    return idx.max(axis=1)


def window_features(win: np.ndarray, age: np.ndarray, interval: np.ndarray) -> np.ndarray:
    """(N, W, 4) windows -> (N, len(FEATURE_NAMES)) float32 matrix (NaN = undefined)."""
    N, W, _ = win.shape
    u, q, d, lat = win[:, :, 0], win[:, :, 1], win[:, :, 2], win[:, :, 3]
    last = W - 1

    above = u > BURST_THRESHOLD
    onset = above & ~np.concatenate([np.zeros((N, 1), bool), above[:, :-1]], axis=1)
    offset = ~above & np.concatenate([np.zeros((N, 1), bool), above[:, :-1]], axis=1)

    last_on = _last_index(onset)
    masked = onset.copy()
    rows = np.arange(N)
    masked[rows[last_on >= 0], last_on[last_on >= 0]] = False
    prev_on = _last_index(masked)
    last_off = _last_index(offset)

    in_burst = above[:, last]
    since_onset = np.where(last_on >= 0, last - last_on, np.nan)
    burst_len = np.where(in_burst & (last_on >= 0), last - last_on, 0.0)
    since_end = np.where(~in_burst & (last_off >= 0), last - last_off, np.nan)
    onset_interval = np.where((last_on >= 0) & (prev_on >= 0), last_on - prev_on, np.nan)
    eta = onset_interval - since_onset  # ticks until the next onset if the cycle repeats
    n_onsets = onset.sum(axis=1).astype(float)

    f = np.stack(
        [
            u[:, last], q[:, last], d[:, last], lat[:, last],
            u[:, -5:].mean(axis=1), u[:, -15:].mean(axis=1), u.mean(axis=1),
            u[:, last] - u[:, last - 3], u[:, last] - u[:, last - 8],
            q[:, last] - q[:, last - 3], q[:, last] - q[:, last - 8],
            u.max(axis=1), q.max(axis=1), q[:, -10:].max(axis=1),
            in_burst.astype(float), since_onset, burst_len, since_end,
            onset_interval, eta, n_onsets,
            age.astype(float), interval.astype(float),
        ],
        axis=1,
    )
    return f.astype(np.float32)


def sliding_windows(series: np.ndarray, idx: np.ndarray, W: int) -> np.ndarray:
    """series (T, 4) zero-padded on the left; return windows ending at each index in idx."""
    pad = np.concatenate([np.zeros((W - 1, series.shape[1]), series.dtype), series], axis=0)
    return pad[idx[:, None] + np.arange(W)[None, :]]
