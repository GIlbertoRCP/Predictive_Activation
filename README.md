# Predictive Network Telemetry & Self-Healing SDN System

An event-driven, distributed SDN monitoring and traffic engineering system that uses LSTM networks and Reinforcement Learning to move from reactive to **proactive, self-healing** network management. 

By decoupling network state monitoring, ML inference, and control plane execution using **Apache Kafka**, the system balances data collection overhead, predictive congestion detection, and automated network rerouting in real-time.

**Team:** Diego Alas, Gilberto Romero-Cano, Corey Green, JJ Wagner  
**Course:** CSCI 4930 HL1 @ CU Denver  

[![CI](https://github.com/GIlbertoRCP/Predictive_Activation/actions/workflows/ci.yml/badge.svg)](https://github.com/GIlbertoRCP/Predictive_Activation/actions/workflows/ci.yml)

> **Status (Oct 2026):** The original SDN course project is preserved unchanged on the [`sdn-course-project`](https://github.com/GIlbertoRCP/Predictive_Activation/tree/sdn-course-project) branch. `main` has been restructured into a tested package as groundwork for the next phase. See [Results](#results) for what the measurements actually show.

---

## System Architecture

The project implements a decoupled, event-driven streaming pipeline:

```
┌────────────────────────────────────────┐
│            SDN Control Plane           │
│     (Mininet Emulation & os-ken)       │
│                                        │
│          s3 ──► s1 ──► s4 (Primary)    │
│           └───► s2 ────┘  (Backup)     │
└──────────────────┬─────────────────────┘
                   │
                   │ Publish Telemetry (JSON)
                   ▼
┌────────────────────────────────────────┐
│            Apache Kafka                │
│    Topic: [network-telemetry]          │
└──────────────────┬─────────────────────┘
                   │
                   │ Consume Telemetry Stream
                   ▼
┌────────────────────────────────────────┐
│        FastAPI Inference Service       │
│    - Fits MinMaxScaler on CSV          │
│    - Runs PyTorch LSTM Forecasts       │
│    - Executes DQN RL Polling Decision  │
│    - Reroutes Path (s1 vs s2)          │
└──────────────────┬─────────────────────┘
                   │
                   │ Publish Control Commands (JSON)
                   ▼
┌────────────────────────────────────────┐
│            Apache Kafka                │
│     Topic: [network-control]           │
└──────────────────┬───────────────┬─────┘
                   │               │
                   │ Consume       │ Consume
                   ▼               ▼
┌───────────────────────────┐ ┌───────────────────────────┐
│     SDN Control Plane     │ │    Streamlit Dashboard    │
│  - Adjust Polling Rate    │ │  - Real-time Graphviz Map │
│  - Apply OpenFlow Rules   │ │  - Live Rolling Charts   │
│    to reroute flows       │ │  - Self-Healing Event Logs│
└───────────────────────────┘ └───────────────────────────┘
```

1. **Lightweight SDN Controller**: The os-ken controller polls OpenFlow switches for port/queue metrics, publishes raw telemetry to the `network-telemetry` topic, and subscribes to the `network-control` topic. It offloads all heavy ML calculations, keeping memory overhead under 30MB.
2. **Inference Microservice**: A FastAPI application consumes telemetry messages, maintains port-specific sliding windows, runs the LSTM to predict congestion, runs the DQN agent to select polling intervals, and decides when to reroute traffic from Core 1 (`s1`) to Core 2 (`s2`).
3. **Closed-Loop Self-Healing**: When the LSTM congestion forecast exceeds **70%**, the microservice commands the controller to:
   - Snaps the polling rate to intensive mode (1s).
   - Installs high-priority (priority 10) flow rules on aggregation switches to steer traffic flows to the backup path `s2` to prevent service degradation.
4. **Interactive Dashboard**: A Streamlit UI displays live metrics, plots throughput/latency/risk charts, lists self-healing logs, and visualizes the network topology in real-time.

---

## Results

All numbers come from [BENCHMARKS.md](BENCHMARKS.md), produced by `python -m predictive_activation.optimize` on a Ryzen 7 7700X.

| Finding | Number |
| :--- | :--- |
| LSTM next-step congestion F1 | 99.14% |
| Persistence baseline (next step = current step) | **99.30%**. The label changes on only 0.6% of steps, so the LSTM adds nothing here. |
| DQN polling agent vs a one-line threshold rule | 5,056 vs 4,990 mean reward (indistinguishable) |
| INT8 LSTM size | 792.6 KB → 210.6 KB (−73.4%) |
| INT8 reroute decisions matching FP32 | 99.92% of 24,909 windows |
| Batch-1 inference, PyTorch → ONNX Runtime | 0.22 ms → 0.054 ms (4.2×) |
| Per-event cost, batch 1 → batch 1024 (INT8) | 54 µs → 2.2 µs |

The streaming and inference engineering holds up. The prediction problem is too easy for the models to matter, which is what the next phase addresses.

---

## Project Layout

```
src/predictive_activation/
  config.py            constants, paths, Kafka topics (single source of truth)
  models.py            LSTM / GRU forecasters
  data.py              dataset loading, labels, sliding windows
  service.py           FastAPI + Kafka inference service (InferenceEngine core)
  dashboard.py         Streamlit dashboard
  optimize.py          ONNX export, INT8 quantization, benchmarks
  replay_simulator.py  replays the CSV into Kafka
  stress_generator.py  synthetic Kafka load generator
  train/forecaster.py  trains the LSTM or GRU (--arch)
  train/rl_agent.py    trains the DQN polling agent
models/                trained checkpoints and ONNX graphs
data/                  recorded Mininet telemetry
benchmarks/            results.json (raw benchmark output)
legacy/sdn/            os-ken controller + Mininet topology (Linux only)
tests/                 pytest suite (no Kafka or Mininet needed)
```

---

## Running It

### 1. Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]" --extra-index-url https://download.pytorch.org/whl/cpu
```

Use the `--extra-index-url` to get the CPU-only PyTorch build. Leave it out if you want CUDA.

### 2. Option A: Local simulation (any OS)

Mininet needs Linux kernel namespaces, so this mode replays recorded telemetry into the same streaming pipeline instead.

```bash
docker compose up -d kafka
python -m predictive_activation.service                      # FastAPI on :8000
python -m predictive_activation.replay_simulator --speed 2.0
streamlit run src/predictive_activation/dashboard.py
```

Open http://localhost:8501 and choose **Live Streaming & Self-Healing**. Or run everything in containers with `docker compose up --build`, then start the replay simulator against `localhost:9092`.

### 3. Option B: Real SDN (Linux with Mininet + Open vSwitch)

```bash
pip install -e ".[sdn]"
python -m predictive_activation.service
os-ken-manager legacy/sdn/simple_switch_13.py legacy/sdn/predictive_controller.py
sudo python3 legacy/sdn/custom_topo.py
./legacy/sdn/traffic.sh
streamlit run src/predictive_activation/dashboard.py
```

### 4. Retrain and re-benchmark

```bash
python -m predictive_activation.train.forecaster --arch lstm
python -m predictive_activation.train.forecaster --arch gru
python -m predictive_activation.train.rl_agent
python -m predictive_activation.optimize          # writes benchmarks/results.json
```

### 5. Tests

```bash
ruff check src tests
pytest
```

CI runs both on every push and pull request.

---

## Dashboard Controls

* **Live engine swapping:** on the **MLOps & Edge Optimization** tab, switch between PyTorch, ONNX FP32 and ONNX INT8. The switch is a `/set-engine` call to the running service.
* **Congestion spike injection:** the sidebar button publishes to `network-simulation`. The replay simulator then injects 15 readings with latency above 150 ms, so you can watch the reroute to `s2`.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Runtime | Python 3.12 |
| Event streaming | Apache Kafka (Confluent 7.5.0, KRaft mode) via Docker Compose |
| Service | FastAPI + Uvicorn, Prometheus client |
| Deep learning | PyTorch 2.10 (LSTM / GRU) |
| Reinforcement learning | Stable-Baselines3 DQN, Gymnasium |
| Inference optimization | ONNX, ONNX Runtime dynamic INT8 quantization |
| Dashboard | Streamlit, Altair, Graphviz |
| SDN (legacy) | os-ken 3.1.1, Mininet, Open vSwitch |

---

## License
MIT. See [LICENSE](LICENSE) for details.
