import os
import json
import time
import threading
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from fastapi import FastAPI, Body
from pydantic import BaseModel
from sklearn.preprocessing import MinMaxScaler
from stable_baselines3 import DQN
import onnxruntime as ort

from traffic_engine import AITrafficEngine

# --- FastAPI Initialization ---
app = FastAPI(
    title="NeuroFlow: Autonomous AI Cluster Traffic Engineering Service",
    description="Proactive Incast Mitigation, Dynamic Flow Steering, and Adaptive Telemetry for AI Clusters"
)

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
ACTION_MAPPING = {0: 30, 1: 10, 2: 1}  # 0=Low (30s), 1=Medium (10s), 2=High (1s)

# Global states
latest_telemetry: Dict[str, Any] = {}
latest_decisions: Dict[str, Any] = {}
history_buffer: Dict[str, List[List[float]]] = {}
current_action_idx: Dict[int, int] = {}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
scaler = MinMaxScaler()
lstm_model = None
rl_agent = None
producer = None
kafka_available = False

# Traffic Engine
traffic_engine = AITrafficEngine()

# ONNX Sessions
lstm_onnx_session = None
lstm_quant_session = None
current_engine = "pytorch"  # pytorch | onnx | onnx_quantized
last_prediction_latency = 0.0

# Embedded generator reference
embedded_generator = None


# --- Model Definition (Matches trained checkpoint) ---
class CongestionPredictor(nn.Module):
    def __init__(self, input_size=6, hidden_size=128, num_layers=2):
        super(CongestionPredictor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x)
        out_last = out[:, 4, :]
        return self.sigmoid(self.fc(out_last))


# --- Request Schemas ---
class TelemetryBatchRequest(BaseModel):
    batch: List[Dict[str, Any]]


# --- Startup Initialization ---
@app.on_event("startup")
def startup_event():
    global scaler, lstm_model, rl_agent, producer, lstm_onnx_session, lstm_quant_session, kafka_available
    
    print("[Inference Service] Fitting MinMaxScaler on telemetry dataset...")
    if os.path.exists(CSV_DATASET_PATH):
        df = pd.read_csv(CSV_DATASET_PATH)
        scaler.fit(df[FEATURES])
        print("[Inference Service] Scaler fitted successfully.")
    else:
        scaler.fit(np.zeros((2, 6)))

    print(f"[Inference Service] Loading PyTorch LSTM model from {MODEL_PATH} on {device}...")
    try:
        lstm_model = CongestionPredictor().to(device)
        lstm_model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
        lstm_model.eval()
        print("[Inference Service] PyTorch model loaded.")
    except Exception as e:
        print(f"[Inference Service] Error loading PyTorch model: {e}")

    print("[Inference Service] Loading ONNX inference sessions...")
    try:
        if os.path.exists("lstm_forecaster.onnx"):
            lstm_onnx_session = ort.InferenceSession("lstm_forecaster.onnx", providers=['CPUExecutionProvider'])
            print("[Inference Service] ONNX FP32 session loaded.")
        if os.path.exists("lstm_forecaster_quant.onnx"):
            lstm_quant_session = ort.InferenceSession("lstm_forecaster_quant.onnx", providers=['CPUExecutionProvider'])
            print("[Inference Service] ONNX INT8 Quantized session loaded.")
    except Exception as e:
        print(f"[Inference Service] ONNX load warning: {e}")

    print(f"[Inference Service] Loading RL agent from {RL_MODEL_PATH}...")
    try:
        rl_agent = DQN.load(RL_MODEL_PATH)
        print("[Inference Service] RL DQN Agent loaded.")
    except Exception as e:
        print(f"[Inference Service] RL Agent load warning: {e}")

    # Initialize Kafka Producer (non-blocking)
    threading.Thread(target=_init_kafka, daemon=True).start()


def _init_kafka():
    global producer, kafka_available
    try:
        from kafka import KafkaProducer, KafkaConsumer
        for i in range(3):
            try:
                producer = KafkaProducer(
                    bootstrap_servers=[BOOTSTRAP_SERVERS],
                    value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                    request_timeout_ms=1000
                )
                kafka_available = True
                print("[Inference Service] Connected to Kafka Producer.")
                
                # Start consumer thread
                threading.Thread(target=_consume_kafka_telemetry, daemon=True).start()
                return
            except Exception:
                time.sleep(1)
    except ImportError:
        pass
    print("[Inference Service] Operating in standalone / direct ingestion mode (Kafka offline).")


def _consume_kafka_telemetry():
    from kafka import KafkaConsumer
    try:
        consumer = KafkaConsumer(
            TELEMETRY_TOPIC,
            bootstrap_servers=[BOOTSTRAP_SERVERS],
            value_deserializer=lambda x: json.loads(x.decode('utf-8')),
            auto_offset_reset='latest'
        )
        for msg in consumer:
            process_telemetry_item(msg.value)
    except Exception as e:
        print(f"[Inference Service] Kafka consumer stopped: {e}")


def process_telemetry_item(data: Dict[str, Any]):
    """Core telemetry processing pipeline shared across Kafka and REST ingest."""
    global latest_telemetry, latest_decisions, history_buffer, current_action_idx, last_prediction_latency
    
    dpid = data['switch_id']
    port_no = data['port_no']
    port_key = f"{dpid}_{port_no}"
    
    latest_telemetry[port_key] = data

    # Feed into traffic engine topology stats
    traffic_engine.set_measured_telemetry(
        switch_id=dpid,
        port_no=port_no,
        rx_mbps=data.get('rx_mbps', 0.0),
        queue_depth=data.get('avg_queue_depth', 0.0),
        latency_ms=data.get('latency_ms', 0.0)
    )

    # Route simulated active flow if flow metadata is present
    src_host = data.get('src_host')
    dst_host = data.get('dst_host')
    flow_type = data.get('flow_type', 'elephant')
    if src_host and dst_host:
        flow_id = f"{src_host}_{dst_host}"
        traffic_engine.route_flow(
            flow_id=flow_id,
            src=src_host,
            dst=dst_host,
            flow_type=flow_type,
            bandwidth_mbps=data.get('rx_mbps', 50.0)
        )

    # Feature vector extraction
    features = [
        data['rx_mbps'],
        data['tx_mbps'],
        data['rx_loss'],
        data['tx_loss'],
        data['avg_queue_depth'],
        data['latency_ms']
    ]

    features_df = pd.DataFrame([features], columns=FEATURES)
    scaled_feats = scaler.transform(features_df)[0]

    # Update sliding window
    if port_key not in history_buffer:
        history_buffer[port_key] = []
    history_buffer[port_key].append(scaled_feats.tolist())

    if len(history_buffer[port_key]) > SEQUENCE_LENGTH:
        history_buffer[port_key].pop(0)

    # Inference step
    if len(history_buffer[port_key]) == SEQUENCE_LENGTH:
        start_pred_time = time.perf_counter()
        
        lstm_prob = 0.0
        if current_engine == "pytorch" or (lstm_onnx_session is None and lstm_quant_session is None):
            if lstm_model:
                window_tensor = torch.tensor([history_buffer[port_key]], dtype=torch.float32).to(device)
                with torch.no_grad():
                    lstm_prob = lstm_model(window_tensor).item()
        elif current_engine == "onnx" and lstm_onnx_session:
            window_numpy = np.array([history_buffer[port_key]], dtype=np.float32)
            outputs = lstm_onnx_session.run(['output'], {'input': window_numpy})
            lstm_prob = float(outputs[0][0][0])
        elif current_engine == "onnx_quantized" and lstm_quant_session:
            window_numpy = np.array([history_buffer[port_key]], dtype=np.float32)
            outputs = lstm_quant_session.run(['output'], {'input': window_numpy})
            lstm_prob = float(outputs[0][0][0])
            
        last_prediction_latency = (time.perf_counter() - start_pred_time) * 1000.0  # ms
        
        # RL Telemetry Scheduling
        curr_action = current_action_idx.get(dpid, 0)
        if rl_agent:
            rl_state = np.array([lstm_prob, scaled_feats[0], float(curr_action)], dtype=np.float32)
            action, _ = rl_agent.predict(rl_state, deterministic=True)
            action = int(action)
        else:
            action = 2 if lstm_prob > RISK_THRESHOLD else 0
            
        new_interval = ACTION_MAPPING.get(action, 30)
        current_action_idx[dpid] = action

        # Update Traffic Engine with node incast risk
        traffic_engine.update_congestion_prediction(switch_id=str(dpid), incast_risk=lstm_prob, threshold=RISK_THRESHOLD)

        # High-level routing path
        routing_path = "s2" if lstm_prob > RISK_THRESHOLD else "s1"

        decision = {
            'timestamp': time.time(),
            'switch_id': dpid,
            'port_no': port_no,
            'lstm_prob': round(lstm_prob, 4),
            'action_idx': action,
            'polling_interval': new_interval,
            'routing_path': routing_path,
            'engine': current_engine,
            'latency_ms': round(last_prediction_latency, 3),
            'training_phase': data.get('training_phase', 'COMPUTE')
        }

        latest_decisions[port_key] = decision

        # Publish decision to Kafka if producer is active
        if producer:
            try:
                producer.send(CONTROL_TOPIC, value=decision)
            except Exception:
                pass


# --- REST Endpoints ---
@app.get("/status")
def get_status():
    return {
        "status": "Inference and Traffic Engineering service online",
        "device": str(device),
        "kafka_connected": kafka_available,
        "bootstrap_servers": BOOTSTRAP_SERVERS,
        "current_engine": current_engine,
        "models_loaded": {
            "pytorch": MODEL_PATH if lstm_model else None,
            "onnx": "lstm_forecaster.onnx" if lstm_onnx_session else None,
            "onnx_quantized": "lstm_forecaster_quant.onnx" if lstm_quant_session else None,
            "rl_agent": RL_MODEL_PATH if rl_agent else None
        },
        "cluster_metrics": traffic_engine.get_cluster_metrics()
    }


@app.post("/set-engine")
def set_engine(engine: str):
    global current_engine
    if engine in ["pytorch", "onnx", "onnx_quantized"]:
        current_engine = engine
        print(f"[Inference Service] Swapped inference engine to: {engine}")
        return {"status": "success", "engine": engine}
    return {"status": "error", "message": f"Invalid engine choice: {engine}"}


@app.post("/ingest-telemetry")
def ingest_telemetry(payload: TelemetryBatchRequest):
    """Direct ingestion endpoint for local simulation without Kafka."""
    for item in payload.batch:
        process_telemetry_item(item)
    return {"status": "success", "processed_count": len(payload.batch)}


@app.get("/latest")
def get_latest():
    traffic_engine.clear_finished_flows(max_age_sec=20.0)
    return {
        "telemetry": latest_telemetry,
        "decisions": latest_decisions,
        "current_engine": current_engine,
        "last_latency_ms": last_prediction_latency,
        "cluster_metrics": traffic_engine.get_cluster_metrics(),
        "active_flows": list(traffic_engine.active_flows.values())[:15],
        "congested_nodes": list(traffic_engine.congested_nodes)
    }


@app.post("/trigger-burst")
@app.post("/trigger-spike")
def trigger_burst():
    """Triggers an immediate Incast Bottleneck / Distributed All-Reduce Burst."""
    # 1. If Kafka is connected, send trigger
    if producer:
        try:
            producer.send("network-simulation", value={"command": "incast"})
            producer.flush()
        except Exception:
            pass

    # 2. If embedded generator exists
    global embedded_generator
    if embedded_generator:
        embedded_generator.trigger_incast(rounds=12)

    return {"status": "success", "message": "High-Volume Distributed Incast Burst Triggered"}


@app.post("/start-embedded-workload")
def start_embedded_workload(speed: float = 1.0):
    """Starts an in-process AI workload simulator for standalone local testing."""
    global embedded_generator
    if embedded_generator and embedded_generator.running:
        return {"status": "already_running"}

    from ai_workload_generator import AIClusterWorkloadGenerator
    embedded_generator = AIClusterWorkloadGenerator(kafka_broker=None, fastapi_url="http://localhost:8000")

    def run_sim():
        while embedded_generator.running:
            batch = embedded_generator.generate_tick()
            for item in batch:
                process_telemetry_item(item)
            time.sleep(max(0.2, 1.0 / speed))

    embedded_generator.running = True
    threading.Thread(target=run_sim, daemon=True).start()
    print("[Inference Service] Embedded AI Workload Generator started.")
    return {"status": "started", "speed": speed}


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
