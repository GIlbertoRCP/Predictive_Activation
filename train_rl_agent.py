import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from sklearn.preprocessing import MinMaxScaler

# Constants kept consistent with train_lstm.py / dashboard.py
SEQUENCE_LENGTH = 6  # 5 history rows + 1 current
LATENCY_THRESHOLD_MS = 100.0
FEATURES = ['rx_mbps', 'tx_mbps', 'rx_loss', 'tx_loss', 'avg_queue_depth', 'latency_ms']
SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)


# --- LSTM architecture (matches train_lstm.py exactly so the checkpoint loads) ---
class CongestionPredictor(nn.Module):
    def __init__(self, input_size=6, hidden_size=128, num_layers=2):
        super(CongestionPredictor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.sigmoid(self.fc(out[:, -1, :]))


class SDNTelemetryEnv(gym.Env):
    """
    Polling-rate selection environment.

    BUG FIXES vs previous version
    -----------------------------
    1. Congestion ground truth now matches train_lstm.py: `latency_ms > 100 ms`
       on the next step. Previously the env used `scaled_latency_ms > 0.5`
       (≈ raw latency > 1855 ms), which disagreed with what the LSTM was
       trained to predict.

    2. Sliding windows no longer cross (switch, port) boundaries. Previously
       `self.df.iloc[step-5:step]` would mix rows from different switches/ports
       at group boundaries because the dataframe is sorted by
       (switch_id, port_no, timestamp). Each episode now walks a single
       (switch, port) trace; on reset, we sample a different trace.
    """

    metadata = {"render_modes": []}

    def __init__(self, csv_path, model_path, alpha=1.0, beta=1.0):
        super().__init__()
        print("Loading dataset and preparing per-(switch, port) traces...")
        df = pd.read_csv(csv_path)
        df = df.sort_values(by=['switch_id', 'port_no', 'timestamp']).reset_index(drop=True)

        # Ground-truth congestion in RAW ms, BEFORE scaling
        df['is_congested'] = (df['latency_ms'] > LATENCY_THRESHOLD_MS).astype(int)

        # Scale features for LSTM input (matches training)
        self.scaler = MinMaxScaler()
        df[FEATURES] = self.scaler.fit_transform(df[FEATURES])

        # Build per-(switch, port) traces; reject any trace too short to play.
        self.traces = []
        for _, group in df.groupby(['switch_id', 'port_no']):
            group = group.reset_index(drop=True)
            if len(group) > SEQUENCE_LENGTH + 30:  # need room to step forward
                self.traces.append({
                    'features': group[FEATURES].values.astype(np.float32),
                    'is_congested': group['is_congested'].values.astype(np.int64),
                })
        if not self.traces:
            raise RuntimeError("No (switch, port) trace is long enough for training.")
        print(f"  Loaded {len(self.traces)} traces (each >= {SEQUENCE_LENGTH + 30} steps)")

        # Action Space: 0 = 30s (Low), 1 = 10s (Medium), 2 = 1s (High)
        self.action_space = spaces.Discrete(3)
        self.intervals = {0: 30, 1: 10, 2: 1}
        self.costs = {0: 1, 1: 3, 2: 30}

        # Observation Space: [LSTM_Prob, Utilization, Current_Interval_Idx]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 2.0], dtype=np.float32),
            dtype=np.float32,
        )

        self.alpha = alpha
        self.beta = beta

        # Load LSTM oracle (matches train_lstm.py architecture)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.lstm_model = CongestionPredictor().to(self.device)
        self.lstm_model.load_state_dict(
            torch.load(model_path, map_location=self.device, weights_only=True)
        )
        self.lstm_model.eval()

        # Episode state
        self.current_trace = None
        self.current_step = 0
        self.current_interval_idx = 0
        self.rng = np.random.default_rng(SEED)

    def _get_lstm_prediction(self, features, step):
        """SEQUENCE_LENGTH-1 history rows ending at `step`. Within-trace only."""
        window = features[step - (SEQUENCE_LENGTH - 1):step + 1]
        x_tensor = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.lstm_model(x_tensor).item()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Sample a trace to play this episode
        self.current_trace = self.traces[self.rng.integers(0, len(self.traces))]
        self.current_step = SEQUENCE_LENGTH - 1  # earliest index with full history
        self.current_interval_idx = 0

        lstm_prob = self._get_lstm_prediction(
            self.current_trace['features'], self.current_step
        )
        utilization = float(self.current_trace['features'][self.current_step, 0])  # rx_mbps
        return np.array([lstm_prob, utilization, 0.0], dtype=np.float32), {}

    def step(self, action):
        features = self.current_trace['features']
        is_congested = self.current_trace['is_congested']

        polling_rate = self.intervals[int(action)]
        cost = self.costs[int(action)]

        next_step = self.current_step + polling_rate
        terminated = next_step >= len(features) - 1
        if terminated:
            next_step = len(features) - 1

        # Did congestion occur in the blind spot we just skipped over?
        window_truth = int(is_congested[self.current_step:next_step].max())

        penalty = 100 if (window_truth == 1 and int(action) != 2) else 0
        accuracy_reward = 50 if (int(action) == 2 and window_truth == 1) else 0
        reward = -(cost) - (self.alpha * penalty) + (self.beta * accuracy_reward)

        # Advance state
        self.current_step = next_step
        self.current_interval_idx = int(action)

        lstm_prob = self._get_lstm_prediction(features, self.current_step)
        utilization = float(features[self.current_step, 0])
        next_state = np.array(
            [lstm_prob, utilization, float(self.current_interval_idx)],
            dtype=np.float32,
        )
        return next_state, float(reward), bool(terminated), False, {}


if __name__ == "__main__":
    print("Initializing SDN Telemetry Environment...")
    env = SDNTelemetryEnv(
        csv_path='telemetry_dataset.csv',
        model_path='lstm_forecaster.pth',
    )

    print("Building Deep Q-Network (DQN) Agent...")
    model = DQN(
        "MlpPolicy", env, verbose=1,
        exploration_fraction=0.2,
        learning_starts=1000,
        seed=SEED,
    )

    print("Training Agent for 20,000 timesteps...")
    model.learn(total_timesteps=20000, progress_bar=False)

    print("\nEvaluating Trained Policy...")
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"Mean Reward per Episode: {mean_reward:.2f} +/- {std_reward:.2f}")

    model.save("dqn_telemetry_agent")
    print("\nRL Agent saved to 'dqn_telemetry_agent.zip'")
