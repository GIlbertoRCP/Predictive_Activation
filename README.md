# Predictive Network Telemetry & Self-Healing SDN System

An event-driven, distributed SDN monitoring and traffic engineering system that uses LSTM networks and Reinforcement Learning to move from reactive to **proactive, self-healing** network management. 

By decoupling network state monitoring, ML inference, and control plane execution using **Apache Kafka**, the system balances data collection overhead, predictive congestion detection, and automated network rerouting in real-time.

**Team:** Diego Alas, Gilberto Romero-Cano, Corey Green, JJ Wagner  
**Course:** CSCI 4930 HL1 @ CU Denver  

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
2. **Inference Microservice**: A FastAPI application consumes telemetry messages, maintains port-specific sliding windows, runs LSTM models to predict congestion, runs the DQN agent to select polling intervals, and decides when to reroute traffic from Core 1 (`s1`) to Core 2 (`s2`).
3. **Closed-Loop Self-Healing**: When the LSTM congestion forecast exceeds **70%**, the microservice commands the controller to:
   - Snaps the polling rate to intensive mode (1s).
   - Installs high-priority (priority 10) flow rules on aggregation switches to steer traffic flows to the backup path `s2` to prevent service degradation.
4. **Interactive Dashboard**: A Streamlit UI displays live metrics, plots throughput/latency/risk charts, lists self-healing logs, and visualizes the network topology in real-time.

---

## Execution Modes & Instructions

### 1. Setup Environment
First, clone the repository, set up a Python virtual environment, and install the dependencies:

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: source .venv/Scripts/activate

# Install dependencies
pip install -r requirements.txt
```

Start the Apache Kafka message broker in the background using Docker:
```bash
docker compose up -d
```

---

### Option A: Local Simulation Mode (macOS / Dev)
Since Mininet requires Linux kernel namespaces and cannot run natively on macOS, use the simulator to replay historical data into the streaming pipeline:

```bash
# 1. Start the FastAPI ML Inference Service (loads LSTM/DQN models)
python3 inference_service.py

# 2. Run the Telemetry Simulator (streams CSV data into Kafka in a loop)
python3 replay_simulator.py --speed 2.0

# 3. Launch the Interactive Dashboard
streamlit run dashboard.py --server.headless=true
```
Navigate to **http://localhost:8501** and select the **Live Streaming & Self-Healing** mode on the sidebar to watch the system dynamically predict congestion, trigger path adjustments, and display self-healing alerts on the visual map.

---

### Option B: Real SDN Mode (Linux / AWS EC2)
If running on an Ubuntu VM with Mininet and Open vSwitch installed:

```bash
# 1. Start the FastAPI ML Inference Service
python3 inference_service.py

# 2. Launch the os-ken SDN controller (handles forwarding and telemetry collection)
os-ken-manager simple_switch_13.py predictive_controller.py

# 3. Start the Mininet network topology
sudo python3 custom_topo.py

# 4. Generate traffic in the Mininet CLI (or run traffic script)
chmod +x traffic.sh
./traffic.sh

# 5. Launch the Dashboard
streamlit run dashboard.py --server.headless=true
```

---

## Edge Model Optimization & MLOps

To prove edge readiness and low-latency operation, the system features an edge optimization suite utilizing **ONNX Runtime** and **Dynamic INT8 Quantization**:

### 1. Compile & Quantize Models
Convert PyTorch `.pth` weights to optimized ONNX graphs and generate quantized versions by running:
```bash
python3 optimize_models.py
```
This performs a full size and latency benchmark run, exporting a performance report to `onnx_benchmarks.json`.

### 2. Edge Optimization Benchmarks
* **PyTorch Baseline**: ~791 KB model size, ~0.175ms inference latency.
* **ONNX Runtime (FP32)**: ~795 KB model size, ~0.058ms inference latency (**66.8% speedup**).
* **ONNX Runtime Quantized (INT8)**: **~214 KB** model size (**72.9% footprint reduction**), **~0.054ms** inference latency (**69.1% speedup**).

### 3. Interactive MLOps Dashboard Controls
* **Live Engine Swapping**: Under the **MLOps & Edge Optimization (ONNX / INT8)** tab, choose between PyTorch, ONNX, and ONNX Quantized execution backends. Swapping triggers an instant REST call (`/set-engine`) to update the live FastAPI inference thread.
* **Congestion Spike Injection**: Click the **"🔥 Inject Congestion Spike"** button in the sidebar to simulate high-load scenarios. FastAPI sends a command to the `network-simulation` topic, prompting the telemetry simulator to inject 15 readings of latency > 150ms and high queue depths, demonstrating real-time path self-healing.

---

## Tech Stack

Every component has been verified for compatibility on **Python 3.12.x**:

| Layer | Technology | Version | Notes |
|-------|-----------|---------|-------|
| Runtime | Python | `3.12` | Broad library support, security patches |
| Infrastructure | Docker & Compose | `v2` | Simplifies running Kafka in KRaft mode |
| Event Streaming | Apache Kafka | `7.5.0` (Confluent) | KRaft mode (Zookeeper-less single-node broker) |
| Microservices | FastAPI & Uvicorn | `0.138.0` / `0.49.0` | Decoupled async inference backend |
| SDN Controller | **os-ken** | `3.1.1` | Maintained Ryu fork (compatible with Python 3.12) |
| Deep Learning | PyTorch | `2.10.0` | LSTM model for forecasting congestion |
| Reinforcement Learning | Stable Baselines3 | `2.9.0` | DQN agent for polling rate escalation |
| Visualization | Graphviz & Altair | `0.21.0` / `6.2.1` | Renders dynamic topological path switches |
| Dashboard | Streamlit | `1.58.0` | Real-time comparative dashboard |

---

## requirements.txt

```text
# Deep Learning & RL
torch==2.10.0
stable-baselines3[extra]>=2.0.0
gymnasium>=0.29.1

# ML & Data
scikit-learn==1.6.1
numpy>=2.0,<3.0
pandas>=2.2

# SDN Controller (WSL/Linux Only)
os-ken==3.1.1
eventlet==0.35.2

# Dashboard & UI
streamlit>=1.30.0
altair>=5.0.0
matplotlib>=3.9
pyyaml>=6.0

# Streaming & Microservices
kafka-python-ng>=2.2.2
fastapi>=0.110.0
uvicorn>=0.28.0
requests>=2.31.0
graphviz>=0.20.0
onnx>=1.15.0
onnxruntime>=1.17.0
onnxscript>=0.1.0
```

---

## License
MIT — see [LICENSE](LICENSE) for details.
