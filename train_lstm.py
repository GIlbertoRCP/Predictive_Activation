import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler

# --- 1. Model Architecture ---
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
def prepare_data(csv_path, sequence_length=5, latency_threshold=20.0):
    print("Loading and preprocessing dataset...")
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
    X, y = prepare_data('telemetry_dataset.csv')
    
    # Split: 80% Training, 20% Testing
    split_idx = int(len(X) * 0.8)
    X_train, y_train = X[:split_idx], y[:split_idx]
    X_test, y_test = X[split_idx:], y[split_idx:]
    
    # Load into PyTorch batches
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=64, shuffle=False)
    
    # Initialize Model, Loss Function, and Optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CongestionPredictor().to(device)
    
    # Binary Cross Entropy (Perfect for 0 or 1 predictions)
    criterion = nn.BCELoss() 
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print(f"\nStarting Training on {device}...")
    epochs = 10
    
    for epoch in range(epochs):
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
            
        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {total_loss/len(train_loader):.4f}")
        
    # --- 4. Evaluation ---
    print("\nEvaluating Model on Test Data...")
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            
            # If probability is > 50%, predict Congestion (1)
            predicted = (outputs > 0.5).float()
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
            
    accuracy = 100 * correct / total
    print(f"Final Test Accuracy: {accuracy:.2f}%")
    
    if accuracy >= 85.0:
        print("✅ Success: Exceeded the 85% Phase 2 benchmark!")
    else:
        print("⚠️ Warning: Fell short of the 85% benchmark. May need more epochs or data.")
        
    # Save the model for Phase 4
    torch.save(model.state_dict(), 'lstm_forecaster.pth')
    print("\nModel weights saved to 'lstm_forecaster.pth'")