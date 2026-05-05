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

To satisfy reproducibility requirements, the project is divided into Data/Infrastructure generation (Linux required) and Machine Learning/Visualization (Cross-Platform).

### System Requirements
* **Python Version:** Python 3.12.10
* **Operating System:** * **Machine Learning & Dashboard:** Windows, macOS, or Linux.
  * **Mininet Network Emulation:** Ubuntu/Debian Linux (or Windows WSL2) is **strictly required**. macOS is not supported for Mininet kernel namespaces.

### 1. Local Environment Setup (For All Team Members)
To run the machine learning models and the Streamlit dashboard, clone this repository and set up a virtual environment:

```bash
# Clone the repository
git clone [https://github.com/](https://github.com/)[YOUR-USERNAME]/Predictive_Activation.git
cd Predictive_Activation

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

```
### 2. Running the ML & Dashboard (Cross-Platform)
We have provided a pre-generated network dataset (`telemetry_dataset.csv`). Any team member can retrain the neural networks locally:

* **Phase 2 (Forecasting):** * Train LSTM: `python3 train_lstm.py`
  * Train GRU: `python3 train_gru.py`
* **Phase 3 (RL Agent):** Train the Deep Q-Network (DQN) on the generated dataset:
  ```bash
  python3 train_rl_agent.py
  ```
* **Phase 5 (Dashboard):** Visualize the overhead comparison and real-time telemetry:
    ```
    streamlit run dashboard.py
    ```

### 3. Running the Infrastructure (Linux/WSL Only)
*Note: This section requires Mininet and Open vSwitch installed.*

1. Start the Predictive Telemetry Controller:
   ```bash
   python3 -m os_ken.cmd.manager predictive_controller.py simple_switch_13.py
2. In a separate terminal, launch the Fat-Tree topology: 
    ```
    sudo python3 custom_topo.py
    ```
3. Trigger the dataset generation traffic script via Mininet CLI:
    ```
    mininet > source traffic.sh
    ```



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


### Why PyTorch over TensorFlow?
Ryu is unmaintained (last release: May 2020) and **broken on Python 3.12** — it depends on `distutils`, `asynchat`, and `ssl.wrap_socket()`, all of which were removed in 3.12. os-ken is OpenStack's actively maintained fork with a near-identical API. Migration from any Ryu code or tutorial is a namespace find-and-replace:



Stable Baselines3 is built on PyTorch — it's a hard dependency (`torch>=2.3.0`). Using TensorFlow would mean abandoning SB3 or installing both frameworks. PyTorch also has broader Python 3.12 support and dominates the modern RL ecosystem.

## Development Roadmap (15 Weeks)

| Phase | Weeks | Focus |
|-------|-------|-------|
| **1 — Infrastructure** | 5-6 | EC2 environment, Mininet + os-ken setup, data collection pipeline |
| **2 — Prediction** | 7-8 | LSTM/GRU model training (target: >85% accuracy) |
| **3 — Decision Engine** | 9-10 | RL agent: environment, reward function, training |
| **4 — Integration** | 11-13 | Close the loop: predictor → RL agent → os-ken controller |
| **5 — Demo** | 14–15 | Streamlit dashboard, baseline comparisons, final report |


**requirements.txt:**

```
# Deep Learning & RL
torch==2.10.0
stable-baselines3==2.7.1
gymnasium>=0.29.1

# ML & Data
scikit-learn==1.6.1
numpy>=2.0,<3.0
pandas>=2.2

# SDN Controller
os-ken==3.1.1

# Dashboard
streamlit==1.54.0

# Utilities
matplotlib>=3.9
pyyaml>=6.0
```

## License

MIT — see [LICENSE](LICENSE) for details.
