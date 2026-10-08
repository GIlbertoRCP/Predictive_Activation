"""Optional PyTorch sequence forecasters (LSTM / GRU) for link-level congestion onset.

Same task, same data and same `risk(features)`-style evaluation as the scikit-learn models, so a recurrent
network can be compared head-to-head:  `python -m predictive_activation.torch_models`.
Requires `pip install torch` (not needed for anything else in this repository).

NOTE: written against the same windows the GBM uses, but not exercised in the CI environment
(torch is not installed there). Run it locally and compare PR-AUC before quoting results.
"""
from __future__ import annotations

import numpy as np

from .config import SimConfig
from .engine import WINDOW, run_episode
from .dataset import Collector
from .features import sliding_windows
from .policies import HORIZON

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover
    torch = None


class SeqForecaster:
    def __init__(self, cell: str = "lstm", hidden: int = 64, epochs: int = 8, seed: int = 0):
        if torch is None:
            raise ImportError("PyTorch is required for SeqForecaster (pip install torch)")
        torch.manual_seed(seed)
        rnn = {"lstm": nn.LSTM, "gru": nn.GRU}[cell]
        self.name = f"{cell.upper()} (PyTorch)"
        self.rnn = rnn(5, hidden, batch_first=True)         # 4 telemetry channels + staleness channel
        self.head = nn.Linear(hidden, 1)
        self.epochs = epochs

    def _forward(self, x):
        out, _ = self.rnn(x)
        return self.head(out[:, -1]).squeeze(-1)

    def fit(self, X: np.ndarray, y: np.ndarray):
        X, y = torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
        params = list(self.rnn.parameters()) + list(self.head.parameters())
        opt = torch.optim.Adam(params, lr=2e-3)
        pos_w = torch.tensor((1 - y.mean()) / y.mean().clamp(min=1e-3))
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)
        for _ in range(self.epochs):
            perm = torch.randperm(len(X))
            for i in range(0, len(X), 256):
                idx = perm[i:i + 256]
                opt.zero_grad()
                loss_fn(self._forward(X[idx]), y[idx]).backward()
                opt.step()
        return self

    def risk(self, X: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return torch.sigmoid(self._forward(torch.tensor(X, dtype=torch.float32))).numpy()


def sequence_dataset(cfg: SimConfig, seeds, interval=None, stride=2, tail=40):
    """(N, tail, 5) raw telemetry tails + staleness channel, and onset labels (not-yet-congested ticks)."""
    Xs, ys = [], []
    for seed in seeds:
        tr = run_episode(cfg, seed, Collector(interval), record=True).trace
        cong, hist, ages = tr["congested"], tr["windows_hist"], tr["ages"]
        T, L = cong.shape
        ticks = np.arange(WINDOW // 2, T - HORIZON, stride)
        for l in range(L):
            keep = ~cong[ticks, l]
            tk = ticks[keep]
            fut = np.stack([cong[tk + k + 1, l] for k in range(HORIZON)], axis=1).any(axis=1)
            series = np.concatenate([hist[WINDOW - 1:WINDOW - 1 + T, l, :], ages[:, l:l + 1] / 15.0], axis=1)
            Xs.append(sliding_windows(series, tk, tail)); ys.append(fut)
    return np.concatenate(Xs).astype(np.float32), np.concatenate(ys).astype(np.float32)


if __name__ == "__main__":
    from sklearn.metrics import average_precision_score
    cfg = SimConfig()
    Xtr, ytr = sequence_dataset(cfg, range(0, 40))
    for cell in ("lstm", "gru"):
        m = SeqForecaster(cell).fit(Xtr, ytr)
        for iv in (1, 5, 10):
            Xte, yte = sequence_dataset(cfg, range(1000, 1010), interval=iv, stride=3)
            print(f"{m.name} polled every {iv}s: PR-AUC {average_precision_score(yte, m.risk(Xte)):.3f}")
