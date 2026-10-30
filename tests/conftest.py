import numpy as np
import pandas as pd
import pytest
import torch

from predictive_activation import config
from predictive_activation.data import fit_scaler
from predictive_activation.models import LSTMPredictor


def make_telemetry(n_steps=20, ports=((1, 1), (1, 2), (2, 1)), congested_every=7, seed=0):
    """Small synthetic telemetry frame: latency spikes above the threshold on
    every `congested_every`-th step of each port."""
    rng = np.random.default_rng(seed)
    rows = []
    for switch_id, port_no in ports:
        for t in range(n_steps):
            spike = t % congested_every == congested_every - 1
            rows.append({
                "timestamp": 1000.0 + t,
                "switch_id": switch_id,
                "port_no": port_no,
                "rx_mbps": rng.uniform(0, 100),
                "tx_mbps": rng.uniform(0, 100),
                "rx_loss": rng.uniform(0, 0.05),
                "tx_loss": rng.uniform(0, 0.05),
                "avg_queue_depth": rng.integers(0, 200),
                "latency_ms": 150.0 if spike else rng.uniform(1, 50),
            })
    # Shuffle so tests prove loaders sort correctly
    return pd.DataFrame(rows).sample(frac=1.0, random_state=seed).reset_index(drop=True)


@pytest.fixture
def telemetry_df():
    return make_telemetry()


@pytest.fixture
def telemetry_csv(tmp_path, telemetry_df):
    path = tmp_path / "telemetry.csv"
    telemetry_df.to_csv(path, index=False)
    return path


@pytest.fixture
def scaler(telemetry_df):
    return fit_scaler(telemetry_df)


@pytest.fixture
def lstm_model():
    torch.manual_seed(0)
    return LSTMPredictor().eval()


class FixedActionAgent:
    """Stand-in for the SB3 DQN: always returns the same action."""

    def __init__(self, action=1):
        self.action = action
        self.states = []

    def predict(self, state, deterministic=True):
        self.states.append(state)
        return np.int64(self.action), None


@pytest.fixture
def fixed_agent():
    return FixedActionAgent()


@pytest.fixture
def event_stream(telemetry_df):
    """Telemetry rows as Kafka-style dicts in timestamp order."""
    df = telemetry_df.sort_values("timestamp")
    return [{k: (v.item() if hasattr(v, "item") else v) for k, v in row.items()}
            for row in df[["timestamp", "switch_id", "port_no"] + config.FEATURES].to_dict("records")]
