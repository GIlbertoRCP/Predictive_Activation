# Predictive Network Telemetry System

An SDN-based monitoring system that uses LSTM/GRU networks and Reinforcement Learning to move from reactive to **proactive** network telemetry. Instead of polling at a fixed rate, the system predicts congestion before it happens and only activates high-fidelity monitoring when it's needed.

**Team:** Diego Alas, Gilberto Romero-Cano, Corey Green, JJ Wagner
**Course:** CSCI 4930 HL1 @ CU Denver

---

## Project Goals

The core idea is **Predictive Activation** — the system operates in two modes:

- **Heartbeat mode** (every 30s): Default low-overhead polling. Used when the LSTM predicts less than 70% chance of congestion.
- **Intensive mode** (every 1s): High-fidelity polling. Triggered when the LSTM predicts 70%+ congestion probability.

A Reinforcement Learning agent learns *when* to switch between these modes, balancing three competing objectives via the reward function:

```
Reward = -(Monitoring_Cost) - α(Congestion_Penalty) + β(Detection_Accuracy)
```

The target is to match the detection accuracy of always-on intensive polling while reducing monitoring overhead by 60% or more.


## Reproducibility & Execution Instructions

To satisfy reproducibility requirements, we have provided a pre-generated network dataset (telemetry_dataset.csv). Any team member and user (Mac/Windows/Linux) can run the Machine Learning and Dashboard components.  


### Option A: Bash Script pipeline 
We built an automated Bash script that trains the LSTM, trains the Reinforcement Learning agent, and launches the interactive dashboard automatically.

```bash
# 1. Create and activate a virtual environment 
python3 -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Make the script executable and run the entire pipeline
chmod +x run_pipeline.sh
./run_pipeline.sh
```

### Option B: Docker Container 
If you do not want to configure a local Python environment, you can run the pre-configured Dashboard using Docker. Live code changes should instantly reflect in the app.

```bash
# Build the image
docker build -t sdn-dashboard .

# Run the container with live-reloading enabled
docker run -p 8501:8501 -v "$(pwd):/app" sdn-dashboard

```
Navigate to http://localhost:8501 to view the multi-tab architecture breakdown and telemetry dynamics

---
### Manual Execution 
If you prefer to run the pipeline phases individually: 
* **Phase 2 (Forecasting):** Train LSTM: `python3 train_lstm.py`
  * Train GRU (Ablation Comparison): `python3 train_gru.py`
* **Phase 3 (RL Agent):** `python3 train_rl_agent.py`
* **Phase 5 (Dashboard):** `streamlit run dashboard.py`


## Tech Stack

Every component has been audited for mutual compatibility on **Python 3.12.10**. Here are the key choices and why we made them.

| Layer | Technology | Version | Notes |
|-------|-----------|---------|-------|
| Runtime | Python | `3.12.10` | Broad ML library support; security patches through Oct 2028 |
| Infrastructure | AWS EC2 (Ubuntu 22.04) | Jammy | Team is on Windows/Mac — Mininet requires Linux kernel namespaces |
| Network Emulation | Mininet | `2.3.1b4` | Install from source on 22.04 |
| SDN Controller | **os-ken** | `3.1.1` | Maintained Ryu fork — see note below |
| Virtual Switch | Open vSwitch | `2.17.9` | OpenFlow 1.3, from Ubuntu default repos |
| Deep Learning | PyTorch | `2.10.0` | LSTM/GRU congestion prediction |
| Reinforcement Learning | Stable Baselines3 | `2.7.1` | DQN/PPO agent for telemetry decisions |
| Anomaly Detection | Scikit-learn | `1.6.1` | DBSCAN clustering |
| Dashboard | Streamlit | `1.54.0` | Real-time comparison of reactive vs. predictive |


### Why os-ken instead of Ryu?
Ryu is unmaintained (last release: May 2020) and **broken on Python 3.12** — it depends on `distutils`, `asynchat`, and `ssl.wrap_socket()`, all of which were removed in 3.12. os-ken is OpenStack's actively maintained fork with a near-identical API. Migration from any Ryu code or tutorial is a namespace find-and-replace:
```python
# Ryu (broken)                        # os-ken (works)
from ryu.base import app_manager  →   from os_ken.base import app_manager
from ryu.controller import ofp_event  →   from os_ken.controller import ofp_event
```

### Why PyTorch over TensorFlow?
Stable Baselines3 is built on PyTorch — it's a hard dependency (`torch>=2.3.0`). Using TensorFlow would mean abandoning SB3 or installing both frameworks. PyTorch also has broader Python 3.12 support and dominates the modern RL ecosystem.


**requirements.txt:**

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

# Dashboard
streamlit==1.54.0

# Utilities
matplotlib>=3.9
pyyaml>=6.0
```

## License

MIT — see [LICENSE](LICENSE) for details.
