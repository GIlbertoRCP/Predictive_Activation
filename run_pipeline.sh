#!/bin/bash

echo "===================================================="
echo "Predictive Activation: Automated ML Pipeline"
echo "===================================================="

# Activate the virtual environment so the script can see your installed packages
source .venv/bin/activate

echo -e "\n[1/3] Phase 2: Training the LSTM Congestion Predictor..."
python3 train_lstm.py

echo -e "\n[2/3] Phase 3: Training the DQN Reinforcement Learning Agent..."
echo "      (This agent is learning to optimize the Reward Function)"
python3 train_rl_agent.py

echo -e "\n[3/3] Phase 5: Launching the Streamlit Dashboard..."
echo "      (Opening on http://localhost:8501)"
streamlit run dashboard.py