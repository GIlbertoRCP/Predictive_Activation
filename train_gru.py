import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import time

# --- 1. The GRU Architecture ---
class CongestionPredictor(nn.Module):
    def __init__(self, input_size=6, hidden_size=128, num_layers=2):
        super(CongestionPredictor, self).__init__()
        # Using GRU instead of LSTM!
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # GRU returns (output, h_n) instead of (output, (h_n, c_n))
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])
        return self.sigmoid(out)

# --- 2. Data Preparation ---
def prepare_data(csv_path, sequence_length=5, latency_threshold=20.0):
    df = pd.read_csv(csv_path)
    df = df.sort_values(by=['switch_id', 'port_no', 'timestamp'])
    
    features = ['rx_mbps', 'tx_mbps', 'rx_loss', 'tx_loss', 'avg_queue_depth', 'latency_ms']
    scaler = MinMaxScaler()
    df[features] = scaler.fit_transform(df[features])
    
    df['future_congestion'] = (df['latency_ms'] > (latency_threshold / df['latency_ms'].max())).astype(int)
    df['future_congestion'] = df.groupby(['switch_id', 'port_no'])['future_congestion'].shift(-1)
    df = df.dropna()

    X, y = [], []
    for _, group in df.groupby(['switch_id', 'port_no']):
        group_features = group[features].values
        group_target = group['future_congestion'].values
        for i in range(len(group) - sequence_length):
            X.append(group_features[i:(i + sequence_length)])
            y.append(group_target[i + sequence_length - 1])
            
    return torch.tensor(np.array(X), dtype=torch.float32), torch.tensor(np.array(y), dtype=torch.float32).unsqueeze(1)

# --- 3. Training Loop ---
if __name__ == "__main__":
    print("Loading dataset for GRU training...")
    X, y = prepare_data('telemetry_dataset.csv')
    
    split_idx = int(len(X) * 0.8)
    train_loader = DataLoader(TensorDataset(X[:split_idx], y[:split_idx]), batch_size=64, shuffle=True)
    test_loader = DataLoader(TensorDataset(X[split_idx:], y[split_idx:]), batch_size=64, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CongestionPredictor().to(device)
    criterion = nn.BCELoss() 
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print(f"\nStarting GRU Training on {device}...")
    start_time = time.time()
    
    for epoch in range(10):
        model.train()
        total_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch+1}/10] | Loss: {total_loss/len(train_loader):.4f}")
        
    training_time = time.time() - start_time
    print(f"\nTotal Training Time: {training_time:.2f} seconds")
    
    # --- 4. Evaluation ---
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            predicted = (outputs > 0.5).float()
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
            
    print(f"Final Test Accuracy: {100 * correct / total:.2f}%")
    torch.save(model.state_dict(), 'gru_forecaster.pth')
    print("Model weights saved to 'gru_forecaster.pth'")