import gymnasium as gym
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from sklearn.preprocessing import MinMaxScaler

# 1. Re-initialize the LSTM to generate predictions for our RL agent
class CongestionPredictor(nn.Module):
    def __init__(self, input_size=6, hidden_size=128, num_layers=2):
        super(CongestionPredictor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return self.sigmoid(out)

# 2. Build the Custom SDN Telemetry Environment
class SDNTelemetryEnv(gym.Env):
    def __init__(self, csv_path, model_path, alpha=1.0, beta=1.0):
        super(SDNTelemetryEnv, self).__init__()
        
        print("Loading dataset and pre-computing LSTM predictions for the RL environment...")
        self.df = pd.read_csv(csv_path).sort_values(by=['switch_id', 'port_no', 'timestamp'])
        
        # Action Space: 0 = 30s (Low), 1 = 10s (Medium), 2 = 1s (High)
        self.action_space = spaces.Discrete(3)
        self.intervals = {0: 30, 1: 10, 2: 1}
        self.costs = {0: 1, 1: 3, 2: 30} # Cost penalty for each mode
        
        # Observation Space: [LSTM_Prob, Utilization, Current_Interval_Idx]
        self.observation_space = spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32)
        
        self.alpha = alpha
        self.beta = beta
        
        # Load the LSTM Model to act as our oracle
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.lstm_model = CongestionPredictor().to(self.device)
        self.lstm_model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        self.lstm_model.eval()

        self._preprocess_data()
        self.current_step = 5 # Start after the first 5-second window
        self.max_steps = len(self.df) - 1
        self.current_interval_idx = 0 # Default to 30s

    def _preprocess_data(self):
        features = ['rx_mbps', 'tx_mbps', 'rx_loss', 'tx_loss', 'avg_queue_depth', 'latency_ms']
        scaler = MinMaxScaler()
        self.df[features] = scaler.fit_transform(self.df[features])
        # Define ground truth congestion (utilization > 80% or latency spike)
        self.df['is_congested'] = (self.df['latency_ms'] > 0.5).astype(int) 

    def _get_lstm_prediction(self, step):
        # Grab the last 5 seconds of data to feed the LSTM
        window = self.df.iloc[step-5:step][['rx_mbps', 'tx_mbps', 'rx_loss', 'tx_loss', 'avg_queue_depth', 'latency_ms']].values
        x_tensor = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            prob = self.lstm_model(x_tensor).item()
        return prob

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.current_step = 5
        self.current_interval_idx = 0
        
        lstm_prob = self._get_lstm_prediction(self.current_step)
        utilization = self.df.iloc[self.current_step]['rx_mbps']
        
        return np.array([lstm_prob, utilization, self.current_interval_idx], dtype=np.float32), {}

    def step(self, action):
        # 1. Execute Action & Calculate Cost
        polling_rate = self.intervals[action]
        cost = self.costs[action]
        
        # 2. Advance time based on the polling rate chosen
        next_step = self.current_step + polling_rate
        
        # Check if we hit the end of the dataset
        terminated = next_step >= self.max_steps
        if terminated:
            next_step = self.max_steps - 1

        # 3. Check what actually happened in the network during that blind spot
        # Did congestion occur while we were sleeping?
        window_truth = self.df.iloc[self.current_step:next_step]['is_congested'].max()
        
        # 4. Calculate the Reward
        penalty = 0
        accuracy_reward = 0
        
        # If congestion happened but we weren't in 1s High-Fidelity mode -> Massive Penalty!
        if window_truth == 1 and action != 2:
            penalty = 100
            
        # If we went into High-Fidelity mode and successfully caught the congestion -> Reward!
        if action == 2 and window_truth == 1:
            accuracy_reward = 50
            
        # The Custom Reward Function from the Proposal
        reward = -(cost) - (self.alpha * penalty) + (self.beta * accuracy_reward)
        
        # 5. Get the Next State
        self.current_step = next_step
        self.current_interval_idx = action
        
        lstm_prob = self._get_lstm_prediction(self.current_step)
        utilization = self.df.iloc[self.current_step]['rx_mbps']
        next_state = np.array([lstm_prob, utilization, self.current_interval_idx], dtype=np.float32)
        
        return next_state, reward, terminated, False, {}

# 3. Train the DQN Agent
if __name__ == "__main__":
    print("Initializing SDN Telemetry Environment...")
    env = SDNTelemetryEnv(csv_path='telemetry_dataset.csv', model_path='lstm_forecaster.pth')
    
    print("Building Deep Q-Network (DQN) Agent...")
    model = DQN("MlpPolicy", env, verbose=1, exploration_fraction=0.2, learning_starts=1000)
    
    print("Training Agent for 20,000 timesteps...")
    model.learn(total_timesteps=20000, progress_bar=True)
    
    print("Evaluating Trained Policy...")
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"Mean Reward per Episode: {mean_reward:.2f} +/- {std_reward:.2f}")
    
    # Save the RL Brain
    model.save("dqn_telemetry_agent")
    print("\n✅ RL Agent saved successfully to 'dqn_telemetry_agent.zip'")