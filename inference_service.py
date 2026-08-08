import os
import json
import time
import threading
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from fastapi import FastAPI, Response
from kafka import KafkaConsumer, KafkaProducer
from sklearn.preprocessing import MinMaxScaler
from stable_baselines3 import DQN
import onnxruntime as ort
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

# --- FastAPI Initialization ---
app = FastAPI(title="SDN Predictive Telemetry Inference Service")

# --- Prometheus Metrics ---
TELEMETRY_EVENTS = Counter('telemetry_events_total', 'Total raw telemetry events ingested')
DECISIONS_PRODUCED = Counter('decisions_produced_total', 'Total network control decisions produced')
INFERENCE_LATENCY = Histogram(
    'inference_latency_seconds',
    'Latency of neural network inference in seconds',
    buckets=[0.00005, 0.0001, 0.00025, 0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05]
)
ACTIVE_ENGINE = Gauge('active_inference_engine', 'Active inference backend (0=PyTorch, 1=ONNX, 2=ONNX_INT8)')

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
recent_latencies = [] # Rolling buffer for p50/p90/p99 calculations

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

    ACTIVE_ENGINE.set(0) # Default: PyTorch

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

# --- Micro-Batched Telemetry Consumer Loop ---
def consume_telemetry():
    global latest_telemetry, latest_decisions, history_buffer, current_action_idx
    global last_prediction_latency, recent_latencies
    
    print(f"Connecting Vectorized Consumer to {BOOTSTRAP_SERVERS} for topic {TELEMETRY_TOPIC}...")
    consumer = None
    for i in range(15):
        try:
            consumer = KafkaConsumer(
                TELEMETRY_TOPIC,
                bootstrap_servers=[BOOTSTRAP_SERVERS],
                value_deserializer=lambda x: json.loads(x.decode('utf-8')),
                auto_offset_reset='latest',
                enable_auto_commit=True,
                max_poll_records=500
            )
            print("Kafka Consumer connected successfully.")
            break
        except Exception as e:
            print(f"Waiting for Kafka Consumer ({i+1}/15): {e}")
            time.sleep(3)

    if not consumer:
        print("Failed to start Kafka Consumer. Thread exiting.")
        return

    while True:
        # Micro-batch polling (non-blocking 5ms poll window)
        records_dict = consumer.poll(timeout_ms=5, max_records=500)
        if not records_dict:
            time.sleep(0.001)
            continue

        raw_events = []
        for tp, messages in records_dict.items():
            for message in messages:
                raw_events.append(message.value)

        if not raw_events:
            continue

        TELEMETRY_EVENTS.inc(len(raw_events))

        # Vectorized feature matrix processing
        port_keys = []
        features_list = []
        scaled_last_rx = []

        for data in raw_events:
            dpid = data['switch_id']
            port_no = data['port_no']
            port_key = f"{dpid}_{port_no}"
            latest_telemetry[port_key] = data

            port_keys.append(port_key)
            features_list.append([
                data['rx_mbps'],
                data['tx_mbps'],
                data['rx_loss'],
                data['tx_loss'],
                data['avg_queue_depth'],
                data['latency_ms']
            ])

        feats_matrix = np.array(features_list, dtype=np.float32)
        scaled_matrix = scaler.transform(feats_matrix)

        # Update sliding window buffers and filter batch candidates
        ready_keys = []
        ready_windows = []
        ready_last_rx = []
        ready_dpids = []
        ready_ports = []

        for i, port_key in enumerate(port_keys):
            scaled_row = scaled_matrix[i].tolist()
            if port_key not in history_buffer:
                history_buffer[port_key] = []
            history_buffer[port_key].append(scaled_row)

            if len(history_buffer[port_key]) > SEQUENCE_LENGTH:
                history_buffer[port_key].pop(0)

            if len(history_buffer[port_key]) == SEQUENCE_LENGTH:
                ready_keys.append(port_key)
                ready_windows.append(history_buffer[port_key])
                ready_last_rx.append(scaled_row[0])
                parts = port_key.split("_")
                ready_dpids.append(int(parts[0]))
                ready_ports.append(int(parts[1]))

        if not ready_windows:
            continue

        # Zero-Copy Vectorized Batch Inference
        start_pred_time = time.perf_counter()
        batch_numpy = np.array(ready_windows, dtype=np.float32) # (N, 5, 6)

        if current_engine == "pytorch" or lstm_onnx_session is None:
            batch_tensor = torch.tensor(batch_numpy, dtype=torch.float32).to(device)
            with torch.no_grad():
                probs = lstm_model(batch_tensor).cpu().numpy().flatten()
        elif current_engine == "onnx":
            outputs = lstm_onnx_session.run(['output'], {'input': batch_numpy})
            probs = outputs[0].flatten()
        else: # onnx_quantized
            outputs = lstm_quant_session.run(['output'], {'input': batch_numpy})
            probs = outputs[0].flatten()

        batch_latency_sec = time.perf_counter() - start_pred_time
        per_event_latency_ms = (batch_latency_sec / len(ready_windows)) * 1000.0
        last_prediction_latency = per_event_latency_ms

        INFERENCE_LATENCY.observe(batch_latency_sec / len(ready_windows))
        recent_latencies.append(per_event_latency_ms)
        if len(recent_latencies) > 1000:
            recent_latencies.pop(0)

        # Process micro-batched control decisions
        for idx in range(len(ready_windows)):
            dpid = ready_dpids[idx]
            port_no = ready_ports[idx]
            port_key = ready_keys[idx]
            lstm_prob = float(probs[idx])

            curr_action = current_action_idx.get(dpid, 0)
            rl_state = np.array([lstm_prob, ready_last_rx[idx], float(curr_action)], dtype=np.float32)
            action, _ = rl_agent.predict(rl_state, deterministic=True)
            action = int(action)
            new_interval = ACTION_MAPPING[action]
            current_action_idx[dpid] = action

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
                'latency_ms': round(per_event_latency_ms, 4)
            }

            latest_decisions[port_key] = decision
            if producer:
                producer.send(CONTROL_TOPIC, value=decision)

        if producer:
            producer.flush()

        DECISIONS_PRODUCED.inc(len(ready_windows))

# --- REST Endpoints ---
@app.get("/status")
def get_status():
    p50 = float(np.percentile(recent_latencies, 50)) if recent_latencies else 0.0
    p90 = float(np.percentile(recent_latencies, 90)) if recent_latencies else 0.0
    p99 = float(np.percentile(recent_latencies, 99)) if recent_latencies else 0.0

    return {
        "status": "Vectorized ONNX Inference service online",
        "device": str(device),
        "bootstrap_servers": BOOTSTRAP_SERVERS,
        "current_engine": current_engine,
        "latency_percentiles_ms": {
            "p50": round(p50, 4),
            "p90": round(p90, 4),
            "p99": round(p99, 4)
        },
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
        engine_code = 0 if engine == "pytorch" else (1 if engine == "onnx" else 2)
        ACTIVE_ENGINE.set(engine_code)
        print(f"--- Swapped Vectorized Inference Engine to: {engine} ---")
        return {"status": "success", "engine": engine}
    return {"status": "error", "message": "Invalid engine choice. Must be one of: pytorch, onnx, onnx_quantized"}

@app.get("/latest")
def get_latest():
    p50 = float(np.percentile(recent_latencies, 50)) if recent_latencies else 0.0
    p99 = float(np.percentile(recent_latencies, 99)) if recent_latencies else 0.0
    return {
        "telemetry": latest_telemetry,
        "decisions": latest_decisions,
        "current_engine": current_engine,
        "last_latency_ms": round(last_prediction_latency, 4),
        "p50_latency_ms": round(p50, 4),
        "p99_latency_ms": round(p99, 4)
    }

@app.get("/metrics")
def get_prometheus_metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

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
