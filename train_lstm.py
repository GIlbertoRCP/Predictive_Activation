import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

# Reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# --- 1. Model Architecture ---
# Kept identical to the previous version so existing model checkpoints stay
# compatible with predictive_controller.py and train_rl_agent.py.
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


# --- 2. Data Preparation ---
def prepare_data(csv_path, sequence_length=5, latency_threshold_ms=100.0):
    """
    Builds (X, y) sequences for next-step congestion prediction.

    BUG FIX vs previous version
    ---------------------------
    Previously, MinMaxScaler.fit_transform was applied to `latency_ms` BEFORE
    the congestion label was computed. That forced `latency_ms` into [0, 1],
    so the threshold `latency_threshold / df['latency_ms'].max()` evaluated to
    `20.0 / 1.0 = 20` and was never exceeded by any scaled value. Result:
    every label was 0, the dataset was 100% negative class, and the model
    trivially learned to always output ~0. Verified empirically: the saved
    `lstm_forecaster.pth` weights produce ~4e-6 on every input.

    Fix: compute the label from RAW latency in milliseconds first, then scale
    features for the network input.
    """
    print("Loading and preprocessing dataset...")
    df = pd.read_csv(csv_path)
    df = df.sort_values(by=['switch_id', 'port_no', 'timestamp']).reset_index(drop=True)

    features = ['rx_mbps', 'tx_mbps', 'rx_loss', 'tx_loss', 'avg_queue_depth', 'latency_ms']

    # Step 1: label congestion from RAW latency_ms (milliseconds), BEFORE scaling
    df['is_congested'] = (df['latency_ms'] > latency_threshold_ms).astype(int)

    # Step 2: shift within each (switch, port) group to get the NEXT-step label
    df['future_congestion'] = (
        df.groupby(['switch_id', 'port_no'])['is_congested'].shift(-1)
    )
    df = df.dropna(subset=['future_congestion']).reset_index(drop=True)
    df['future_congestion'] = df['future_congestion'].astype(int)

    # Step 3: NOW scale features for the network input
    scaler = MinMaxScaler()
    df[features] = scaler.fit_transform(df[features])

    # Step 4: build per-(switch, port) sequences (never cross group boundaries)
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
    X, y, pos_rate = prepare_data('telemetry_dataset.csv', latency_threshold_ms=100.0)

    # Shuffle then split (each sequence is independent within its group)
    g = torch.Generator().manual_seed(SEED)
    perm = torch.randperm(len(X), generator=g)
    X, y = X[perm], y[perm]

    split_idx = int(len(X) * 0.8)
    X_train, y_train = X[:split_idx], y[:split_idx]
    X_test,  y_test  = X[split_idx:], y[split_idx:]

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
    test_loader  = DataLoader(TensorDataset(X_test,  y_test),  batch_size=64, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CongestionPredictor().to(device)

    # Class-imbalance weighting. Architecture keeps Sigmoid in forward(), which
    # the rest of the pipeline depends on, so we stay with BCELoss and apply
    # per-sample weights manually instead of switching to BCEWithLogitsLoss.
    pos_weight = (1.0 - pos_rate) / max(pos_rate, 1e-6)
    print(f"  Class-imbalance pos_weight: {pos_weight:.3f}")
    criterion = nn.BCELoss(reduction='none')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print(f"\nStarting Training on {device}...")
    epochs = 10
    for epoch in range(epochs):
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
        print(f"  Epoch [{epoch+1}/{epochs}] | Loss: {total_loss/len(train_loader):.4f}")

    # --- 4. Honest Evaluation ---
    # Imbalanced classification: accuracy alone is misleading. Report the full
    # suite so the dashboard and the report can show real numbers.
    print("\nEvaluating Model on Test Data...")
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

    if f1 >= 0.70:
        print("\nSuccess: Model produces meaningful predictions on both classes.")
    else:
        print("\nWarning: F1 below 0.70. Consider tuning the threshold, "
              "sequence length, or number of epochs.")

    torch.save(model.state_dict(), 'lstm_forecaster.pth')
    print("\nModel weights saved to 'lstm_forecaster.pth'")
