"""Train the LSTM or GRU congestion forecaster.

    python -m predictive_activation.train.forecaster --arch lstm
"""

import argparse
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, TensorDataset

from predictive_activation import config
from predictive_activation.data import build_sequences
from predictive_activation.models import ARCHITECTURES


def train(arch, epochs=10, batch_size=64, lr=1e-3, out_path=None):
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)

    print("Loading and preprocessing dataset...")
    X, y, _ = build_sequences()
    X, y = torch.from_numpy(X), torch.from_numpy(y).unsqueeze(1)
    pos_rate = y.mean().item()
    print(f"  Total sequences: {len(X)}")
    print(f"  Positive class rate (future_congestion=1): {pos_rate*100:.2f}%")

    # TODO(phase 1): overlapping windows are shuffled before the split, so test
    # windows share rows with training windows. Switch to a time-based split.
    g = torch.Generator().manual_seed(config.SEED)
    perm = torch.randperm(len(X), generator=g)
    X, y = X[perm], y[perm]
    split_idx = int(len(X) * 0.8)
    train_loader = DataLoader(TensorDataset(X[:split_idx], y[:split_idx]), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(X[split_idx:], y[split_idx:]), batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ARCHITECTURES[arch]().to(device)

    # Class-imbalance weighting. forward() ends in Sigmoid, which the rest of
    # the pipeline depends on, so use BCELoss with manual per-sample weights.
    pos_weight = (1.0 - pos_rate) / max(pos_rate, 1e-6)
    print(f"  Class-imbalance pos_weight: {pos_weight:.3f}")
    criterion = nn.BCELoss(reduction="none")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"\nTraining {arch.upper()} on {device}...")
    start = time.time()
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            weights = torch.where(batch_y > 0.5, torch.tensor(pos_weight, device=device), torch.tensor(1.0, device=device))
            loss = (criterion(outputs, batch_y) * weights).mean()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"  Epoch [{epoch+1}/{epochs}] | Loss: {total_loss/len(train_loader):.4f}")
    print(f"Training time: {time.time() - start:.2f}s")

    # Imbalanced classification: accuracy alone is misleading, report the full suite.
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            outputs = model(batch_X.to(device)).cpu().numpy().flatten()
            preds.extend((outputs > 0.5).astype(int).tolist())
            targets.extend(batch_y.numpy().astype(int).flatten().tolist())

    print(f"\n  Accuracy:  {accuracy_score(targets, preds)*100:.2f}%")
    print(f"  Precision: {precision_score(targets, preds, zero_division=0)*100:.2f}%")
    print(f"  Recall:    {recall_score(targets, preds, zero_division=0)*100:.2f}%")
    print(f"  F1 Score:  {f1_score(targets, preds, zero_division=0)*100:.2f}%")
    print(f"  Confusion Matrix [[TN, FP], [FN, TP]]:\n{confusion_matrix(targets, preds)}")

    out_path = out_path or (config.LSTM_PATH if arch == "lstm" else config.GRU_PATH)
    torch.save(model.state_dict(), out_path)
    print(f"\nModel weights saved to {out_path}")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arch", choices=sorted(ARCHITECTURES), default="lstm")
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()
    train(args.arch, epochs=args.epochs)
