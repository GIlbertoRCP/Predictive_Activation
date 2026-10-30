"""Dataset loading and sequence construction shared by training and evaluation."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from predictive_activation import config

GROUP_KEYS = ["switch_id", "port_no"]


def load_telemetry(csv_path=config.DATASET_PATH):
    df = pd.read_csv(csv_path)
    return df.sort_values(by=GROUP_KEYS + ["timestamp"]).reset_index(drop=True)


def fit_scaler(df):
    scaler = MinMaxScaler()
    # Fit on a bare array: the service transforms raw arrays from Kafka.
    scaler.fit(df[config.FEATURES].to_numpy())
    return scaler


def add_labels(df, latency_threshold_ms=config.LATENCY_THRESHOLD_MS):
    """Label congestion from RAW latency in ms, before any scaling.

    Scaling first squashes latency into [0, 1] so a millisecond threshold is
    never exceeded and every label comes out 0 (an earlier bug in this repo).
    `future_congestion` is the next step's label within the same (switch, port).
    """
    df = df.copy()
    df["is_congested"] = (df["latency_ms"] > latency_threshold_ms).astype(int)
    df["future_congestion"] = df.groupby(GROUP_KEYS)["is_congested"].shift(-1)
    return df


def build_sequences(csv_path=config.DATASET_PATH, sequence_length=config.SEQUENCE_LENGTH,
                    latency_threshold_ms=config.LATENCY_THRESHOLD_MS):
    """Return (X, y, scaler): X is (N, sequence_length, n_features) scaled
    windows, y is the congestion label for the step after each window.
    Windows never cross (switch, port) boundaries."""
    df = add_labels(load_telemetry(csv_path), latency_threshold_ms)
    df = df.dropna(subset=["future_congestion"]).reset_index(drop=True)
    df["future_congestion"] = df["future_congestion"].astype(int)

    scaler = fit_scaler(df)
    df[config.FEATURES] = scaler.transform(df[config.FEATURES].to_numpy())

    X, y = [], []
    for _, group in df.groupby(GROUP_KEYS):
        feats = group[config.FEATURES].values
        target = group["future_congestion"].values
        for i in range(len(group) - sequence_length + 1):
            X.append(feats[i:i + sequence_length])
            y.append(target[i + sequence_length - 1])

    X = np.asarray(X, dtype=np.float32).reshape(-1, sequence_length, len(config.FEATURES))
    y = np.asarray(y, dtype=np.float32)
    return X, y, scaler
