"""Shared constants and paths. Every module imports from here so the
training scripts, inference service and dashboard can't drift apart."""

import os
from pathlib import Path

# Paths resolve against PA_ROOT (default: current working directory), so the
# same code works from the repo root locally and from /app in Docker.
ROOT = Path(os.getenv("PA_ROOT", Path.cwd()))
MODELS_DIR = Path(os.getenv("PA_MODELS_DIR", ROOT / "models"))
DATA_DIR = Path(os.getenv("PA_DATA_DIR", ROOT / "data"))
BENCHMARKS_DIR = Path(os.getenv("PA_BENCHMARKS_DIR", ROOT / "benchmarks"))

DATASET_PATH = DATA_DIR / "telemetry_dataset.csv"
LSTM_PATH = MODELS_DIR / "lstm_forecaster.pth"
GRU_PATH = MODELS_DIR / "gru_forecaster.pth"
LSTM_ONNX_PATH = MODELS_DIR / "lstm_forecaster.onnx"
LSTM_QUANT_PATH = MODELS_DIR / "lstm_forecaster_quant.onnx"
RL_AGENT_PATH = MODELS_DIR / "dqn_telemetry_agent"  # SB3 appends .zip
BENCHMARK_RESULTS_PATH = BENCHMARKS_DIR / "results.json"

# Model / feature contract
FEATURES = ["rx_mbps", "tx_mbps", "rx_loss", "tx_loss", "avg_queue_depth", "latency_ms"]
SEQUENCE_LENGTH = 5
LATENCY_THRESHOLD_MS = 100.0  # raw latency above this counts as congested
RISK_THRESHOLD = 0.70  # predicted probability above this triggers a reroute
SEED = 42

# RL polling actions: index -> polling interval in seconds
ACTION_MAPPING = {0: 30, 1: 10, 2: 1}

# Kafka
BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
TELEMETRY_TOPIC = "network-telemetry"
CONTROL_TOPIC = "network-control"
SIMULATION_TOPIC = "network-simulation"
