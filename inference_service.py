import os
import json
import time
import threading
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from fastapi import FastAPI
from kafka import KafkaConsumer, KafkaProducer
from sklearn.preprocessing import MinMaxScaler
from stable_baselines3 import DQN
import onnxruntime as ort

# --- FastAPI Initialization ---
app = FastAPI(title="SDN Predictive Telemetry Inference Service")

# --- Constants & Config ---
BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
TELEMETRY_TOPIC = "network-telemetry"
CONTROL_TOPIC = "network-control"
MODEL_PATH = "lstm_forecaster.pth"
RL_MODEL_PATH = "dqn_telemetry_agent"
CSV_DATASET_PATH = "telemetry_dataset.csv"
SEQUENCE_LENGTH = 5
RISK_THRESHOLD = 0.70

FEATURES = ['rx_mbps', 'tx_mbps', 'rx_loss', 'tx_loss', 'avg_queue_depth', 'latency_ms']
ACTION_MAPPING = {0: 30, 1: 10, 2: 1} # 0=Low (30s), 1=Medium (10s), 2=High (1s)

# Global states
latest_telemetry = {}
latest_decisions = {}
history_buffer = {} # (switch_id, port_no) -> list of last 5 scaled feature vectors
current_action_idx = {} # dpid -> int (0, 1, or 2)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
scaler = MinMaxScaler()
lstm_model = None
rl_agent = None
producer = None

# ONNX Sessions
lstm_onnx_session = None
lstm_quant_session = None
current_engine = "pytorch" # pytorch | onnx | onnx_quantized
last_prediction_latency = 0.0

# --- LSTM Model Definition ---
class CongestionPredictor(nn.Module):
    def __init__(self, input_size=6, hidden_size=128, num_layers=2):
        super(CongestionPredictor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x)
        # Sliced at static index 4 to match ONNX compile
        out_last = out[:, 4, :]
        return self.sigmoid(self.fc(out_last))

# --- Startup Initialization ---
@app.on_event("startup")
def startup_event():
    global scaler, lstm_model, rl_agent, producer, lstm_onnx_session, lstm_quant_session
    
    print("Fitting MinMaxScaler on telemetry dataset...")
    if os.path.exists(CSV_DATASET_PATH):
        df = pd.read_csv(CSV_DATASET_PATH)
        scaler.fit(df[FEATURES])
        print("Scaler fitted successfully.")
    else:
        print(f"Warning: {CSV_DATASET_PATH} not found. Using default scaler ranges.")
        scaler.fit(np.zeros((2, 6)))

    print(f"Loading PyTorch LSTM model from {MODEL_PATH} on {device}...")
    lstm_model = CongestionPredictor().to(device)
    lstm_model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    lstm_model.eval()
    print("PyTorch model loaded.")

    print("Loading ONNX sessions...")
    try:
        lstm_onnx_session = ort.InferenceSession("lstm_forecaster.onnx", providers=['CPUExecutionProvider'])
        lstm_quant_session = ort.InferenceSession("lstm_forecaster_quant.onnx", providers=['CPUExecutionProvider'])
        print("ONNX models loaded successfully.")
    except Exception as e:
        print(f"Warning: ONNX models failed to load. Only PyTorch will be available. Error: {e}")

    print(f"Loading RL agent from {RL_MODEL_PATH}...")
    rl_agent = DQN.load(RL_MODEL_PATH)
    print("RL Agent loaded.")

    # Initialize Producer
    for i in range(10):
        try:
            producer = KafkaProducer(
                bootstrap_servers=[BOOTSTRAP_SERVERS],
                value_serializer=lambda v: json.dumps(v).encode('utf-8')
            )
            print("Kafka Producer initialized.")
            break
        except Exception as e:
            print(f"Waiting for Kafka Producer ({i+1}/10): {e}")
            time.sleep(3)

    # Start Kafka Consumer Thread
    consumer_thread = threading.Thread(target=consume_telemetry, daemon=True)
    consumer_thread.start()
    print("Kafka Consumer Thread started.")

# --- Telemetry Consumer Loop ---
def consume_telemetry():
    global latest_telemetry, latest_decisions, history_buffer, current_action_idx, last_prediction_latency
    print(f"Connecting Consumer to {BOOTSTRAP_SERVERS} for topic {TELEMETRY_TOPIC}...")
    consumer = None
    for i in range(15):
        try:
            consumer = KafkaConsumer(
                TELEMETRY_TOPIC,
                bootstrap_servers=[BOOTSTRAP_SERVERS],
                value_deserializer=lambda x: json.loads(x.decode('utf-8')),
                auto_offset_reset='latest'
            )
            print("Kafka Consumer connected successfully.")
            break
        except Exception as e:
            print(f"Waiting for Kafka Consumer ({i+1}/15): {e}")
            time.sleep(3)

    if not consumer:
        print("Failed to start Kafka Consumer. Thread exiting.")
        return

    for message in consumer:
        data = message.value
        dpid = data['switch_id']
        port_no = data['port_no']
        port_key = f"{dpid}_{port_no}"
        
        # Save latest telemetry for dashboard endpoint
        latest_telemetry[port_key] = data

        # Extract features
        features = [
            data['rx_mbps'],
            data['tx_mbps'],
            data['rx_loss'],
            data['tx_loss'],
            data['avg_queue_depth'],
            data['latency_ms']
        ]

        # Scale features using correct DataFrame columns to prevent sklearn warnings
        features_df = pd.DataFrame([features], columns=FEATURES)
        scaled_feats = scaler.transform(features_df)[0]

        # Update sliding window
        if port_key not in history_buffer:
            history_buffer[port_key] = []
        history_buffer[port_key].append(scaled_feats.tolist())

        if len(history_buffer[port_key]) > SEQUENCE_LENGTH:
            history_buffer[port_key].pop(0)

        # Run inference if window is complete
        if len(history_buffer[port_key]) == SEQUENCE_LENGTH:
            start_pred_time = time.perf_counter()
            
            # Predict Congestion using the selected engine
            if current_engine == "pytorch" or lstm_onnx_session is None:
                window_tensor = torch.tensor([history_buffer[port_key]], dtype=torch.float32).to(device)
                with torch.no_grad():
                    lstm_prob = lstm_model(window_tensor).item()
            elif current_engine == "onnx":
                window_numpy = np.array([history_buffer[port_key]], dtype=np.float32)
                outputs = lstm_onnx_session.run(['output'], {'input': window_numpy})
                lstm_prob = float(outputs[0][0][0])
            else: # onnx_quantized
                window_numpy = np.array([history_buffer[port_key]], dtype=np.float32)
                outputs = lstm_quant_session.run(['output'], {'input': window_numpy})
                lstm_prob = float(outputs[0][0][0])
                
            last_prediction_latency = (time.perf_counter() - start_pred_time) * 1000.0 # ms
            
            curr_action = current_action_idx.get(dpid, 0)
            
            # Observe: [LSTM Congestion Risk, Port RX utilization (scaled), Current Polling Interval Action]
            rl_state = np.array([lstm_prob, scaled_feats[0], float(curr_action)], dtype=np.float32)
            action, _ = rl_agent.predict(rl_state, deterministic=True)
            action = int(action)
            new_interval = ACTION_MAPPING[action]
            
            # Update action index state
            current_action_idx[dpid] = action

            # Self-healing routing decision
            routing_path = "s2" if lstm_prob > RISK_THRESHOLD else "s1"

            decision = {
                'timestamp': time.time(),
                'switch_id': dpid,
                'port_no': port_no,
                'lstm_prob': lstm_prob,
                'action_idx': action,
                'polling_interval': new_interval,
                'routing_path': routing_path,
                'engine': current_engine,
                'latency_ms': last_prediction_latency
            }

            latest_decisions[port_key] = decision

            # Publish decision back to Kafka control topic
            if producer:
                producer.send(CONTROL_TOPIC, value=decision)
                producer.flush()

# --- REST Endpoints ---
@app.get("/status")
def get_status():
    return {
        "status": "Inference service online",
        "device": str(device),
        "bootstrap_servers": BOOTSTRAP_SERVERS,
        "current_engine": current_engine,
        "models_loaded": {
            "pytorch": MODEL_PATH,
            "onnx": "lstm_forecaster.onnx" if lstm_onnx_session else None,
            "onnx_quantized": "lstm_forecaster_quant.onnx" if lstm_quant_session else None
        }
    }

@app.post("/set-engine")
def set_engine(engine: str):
    global current_engine
    if engine in ["pytorch", "onnx", "onnx_quantized"]:
        current_engine = engine
        print(f"--- Swapped Inference Engine to: {engine} ---")
        return {"status": "success", "engine": engine}
    return {"status": "error", "message": f"Invalid engine choice. Must be one of: pytorch, onnx, onnx_quantized"}

@app.get("/latest")
def get_latest():
    return {
        "telemetry": latest_telemetry,
        "decisions": latest_decisions,
        "current_engine": current_engine,
        "last_latency_ms": last_prediction_latency
    }

@app.post("/trigger-spike")
def trigger_spike():
    global producer
    if producer:
        producer.send("network-simulation", value={"command": "spike"})
        producer.flush()
        print("--- Broadcasted Spike Command to Kafka: network-simulation ---")
        return {"status": "success", "message": "Congestion spike triggered"}
    return {"status": "error", "message": "Kafka producer not connected"}


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
