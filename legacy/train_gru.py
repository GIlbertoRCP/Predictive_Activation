import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
import time

# Reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)


# --- 1. GRU Architecture (unchanged) ---
class CongestionPredictor(nn.Module):
    def __init__(self, input_size=6, hidden_size=128, num_layers=2):
        super(CongestionPredictor, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])
        return self.sigmoid(out)


# --- 2. Data Preparation ---
def prepare_data(csv_path, sequence_length=5, latency_threshold_ms=100.0):
    """
    Same fix as train_lstm.py: compute the congestion label from RAW latency_ms
    in milliseconds BEFORE scaling. Previously, MinMaxScaler forced latency_ms
    into [0, 1], making the threshold `20.0 / 1.0 = 20` impossible to exceed
    and producing 100% negative labels.
    """
    df = pd.read_csv(csv_path)
    df = df.sort_values(by=['switch_id', 'port_no', 'timestamp']).reset_index(drop=True)

    features = ['rx_mbps', 'tx_mbps', 'rx_loss', 'tx_loss', 'avg_queue_depth', 'latency_ms']

    # Label from raw milliseconds, BEFORE scaling
    df['is_congested'] = (df['latency_ms'] > latency_threshold_ms).astype(int)
    df['future_congestion'] = (
        df.groupby(['switch_id', 'port_no'])['is_congested'].shift(-1)
    )
    df = df.dropna(subset=['future_congestion']).reset_index(drop=True)
    df['future_congestion'] = df['future_congestion'].astype(int)

    # Scale features AFTER labels are set
    scaler = MinMaxScaler()
    df[features] = scaler.fit_transform(df[features])

    X, y = [], []
    for _, group in df.groupby(['switch_id', 'port_no']):
        group_features = group[features].values
        group_target = group['future_congestion'].values
        for i in range(len(group) - sequence_length):
            X.append(group_features[i:(i + sequence_length)])
            y.append(group_target[i + sequence_length - 1])

    X = torch.tensor(np.array(X), dtype=torch.float32)
    y = torch.tensor(np.array(y), dtype=torch.float32).unsqueeze(1)

    pos_rate = y.mean().item()
    print(f"  Total sequences: {len(X)}")
    print(f"  Positive class rate (future_congestion=1): {pos_rate*100:.2f}%")
    return X, y, pos_rate


# --- 3. Training Loop ---
if __name__ == "__main__":
    print("Loading dataset for GRU training...")
    X, y, pos_rate = prepare_data('telemetry_dataset.csv', latency_threshold_ms=100.0)

    g = torch.Generator().manual_seed(SEED)
    perm = torch.randperm(len(X), generator=g)
    X, y = X[perm], y[perm]

    split_idx = int(len(X) * 0.8)
    train_loader = DataLoader(TensorDataset(X[:split_idx], y[:split_idx]),
                              batch_size=64, shuffle=True)
    test_loader = DataLoader(TensorDataset(X[split_idx:], y[split_idx:]),
                             batch_size=64, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CongestionPredictor().to(device)

    pos_weight = (1.0 - pos_rate) / max(pos_rate, 1e-6)
    print(f"  Class-imbalance pos_weight: {pos_weight:.3f}")
    criterion = nn.BCELoss(reduction='none')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print(f"\nStarting GRU Training on {device}...")
    start_time = time.time()

    for epoch in range(10):
        model.train()
        total_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            weights = torch.where(batch_y > 0.5,
                                  torch.tensor(pos_weight, device=device),
                                  torch.tensor(1.0, device=device))
            loss = (criterion(outputs, batch_y) * weights).mean()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"  Epoch [{epoch+1}/10] | Loss: {total_loss/len(train_loader):.4f}")

    training_time = time.time() - start_time
    print(f"\nTotal Training Time: {training_time:.2f} seconds")

    # --- 4. Honest Evaluation ---
    print("\nEvaluating GRU on Test Data...")
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X).cpu().numpy().flatten()
            all_preds.extend((outputs > 0.5).astype(int).tolist())
            all_targets.extend(batch_y.numpy().astype(int).flatten().tolist())

    acc  = accuracy_score(all_targets, all_preds)
    prec = precision_score(all_targets, all_preds, zero_division=0)
    rec  = recall_score(all_targets, all_preds, zero_division=0)
    f1   = f1_score(all_targets, all_preds, zero_division=0)
    cm   = confusion_matrix(all_targets, all_preds)

    print(f"\n  Accuracy:  {acc*100:.2f}%")
    print(f"  Precision: {prec*100:.2f}%")
    print(f"  Recall:    {rec*100:.2f}%")
    print(f"  F1 Score:  {f1*100:.2f}%")
    print(f"  Confusion Matrix [[TN, FP], [FN, TP]]:\n{cm}")

    torch.save(model.state_dict(), 'gru_forecaster.pth')
    print("\nModel weights saved to 'gru_forecaster.pth'")
