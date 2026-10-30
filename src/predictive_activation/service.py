"""FastAPI inference service: consumes telemetry from Kafka, runs batched
LSTM forecasts + the DQN polling policy, and publishes control decisions.

    python -m predictive_activation.service
"""

import json
import threading
import time
from collections import deque
from contextlib import asynccontextmanager

import numpy as np
import onnxruntime as ort
import torch
from fastapi import FastAPI, Response
from kafka import KafkaConsumer, KafkaProducer
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from stable_baselines3 import DQN

from predictive_activation import config
from predictive_activation.data import fit_scaler, load_telemetry
from predictive_activation.models import load_model

# --- Prometheus Metrics ---
TELEMETRY_EVENTS = Counter('telemetry_events_total', 'Total raw telemetry events ingested')
DECISIONS_PRODUCED = Counter('decisions_produced_total', 'Total network control decisions produced')
INFERENCE_LATENCY = Histogram(
    'inference_latency_seconds',
    'Latency of neural network inference in seconds',
    buckets=[0.00005, 0.0001, 0.00025, 0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05]
)
ACTIVE_ENGINE = Gauge('active_inference_engine', 'Active inference backend (0=PyTorch, 1=ONNX, 2=ONNX_INT8)')

ENGINES = ["pytorch", "onnx", "onnx_quantized"]


class InferenceEngine:
    """Kafka-free core of the service: sliding windows, batched inference and
    decision logic. Kept separate so it can be tested without a broker."""

    def __init__(self, scaler, lstm_model, rl_agent, onnx_sessions=None, device="cpu"):
        self.scaler = scaler
        self.lstm_model = lstm_model
        self.rl_agent = rl_agent
        self.onnx_sessions = onnx_sessions or {}  # engine name -> ort.InferenceSession
        self.device = device
        self.engine = "pytorch"

        self.latest_telemetry = {}
        self.latest_decisions = {}
        self.history = {}  # "dpid_port" -> deque of last SEQUENCE_LENGTH scaled feature vectors
        self.current_action_idx = {}  # dpid -> int (0, 1, or 2)
        self.last_latency_ms = 0.0
        self.recent_latencies = deque(maxlen=1000)  # rolling buffer for p50/p90/p99

    def set_engine(self, engine):
        if engine not in ENGINES:
            raise ValueError(f"Invalid engine {engine!r}. Must be one of: {', '.join(ENGINES)}")
        self.engine = engine

    def predict(self, batch):
        """batch: float32 array (N, SEQUENCE_LENGTH, n_features) -> (N,) probabilities."""
        session = self.onnx_sessions.get(self.engine)
        if session is None:
            with torch.no_grad():
                return self.lstm_model(torch.from_numpy(batch).to(self.device)).cpu().numpy().flatten()
        return session.run(['output'], {'input': batch})[0].flatten()

    def latency_percentiles(self):
        if not self.recent_latencies:
            return {"p50": 0.0, "p90": 0.0, "p99": 0.0}
        p50, p90, p99 = np.percentile(self.recent_latencies, [50, 90, 99])
        return {"p50": round(float(p50), 4), "p90": round(float(p90), 4), "p99": round(float(p99), 4)}

    def process(self, events):
        """Ingest raw telemetry events and return a decision for every port
        whose window is full."""
        if not events:
            return []

        port_keys = []
        for data in events:
            port_key = f"{data['switch_id']}_{data['port_no']}"
            self.latest_telemetry[port_key] = data
            port_keys.append(port_key)

        feats_matrix = np.array([[data[f] for f in config.FEATURES] for data in events], dtype=np.float32)
        scaled_matrix = self.scaler.transform(feats_matrix)

        # Update sliding window buffers and collect ports ready for inference
        ready = []  # (port_key, window, scaled_rx)
        for port_key, scaled_row in zip(port_keys, scaled_matrix):
            window = self.history.setdefault(port_key, deque(maxlen=config.SEQUENCE_LENGTH))
            window.append(scaled_row)
            if len(window) == config.SEQUENCE_LENGTH:
                ready.append((port_key, np.array(window), float(scaled_row[0])))

        if not ready:
            return []

        # Vectorized batch inference
        start = time.perf_counter()
        probs = self.predict(np.stack([w for _, w, _ in ready]).astype(np.float32))
        per_event_sec = (time.perf_counter() - start) / len(ready)
        self.last_latency_ms = per_event_sec * 1000.0
        INFERENCE_LATENCY.observe(per_event_sec)
        self.recent_latencies.append(self.last_latency_ms)

        decisions = []
        for (port_key, _, scaled_rx), prob in zip(ready, probs):
            dpid, port_no = (int(p) for p in port_key.split("_"))
            lstm_prob = float(prob)

            curr_action = self.current_action_idx.get(dpid, 0)
            rl_state = np.array([lstm_prob, scaled_rx, float(curr_action)], dtype=np.float32)
            action, _ = self.rl_agent.predict(rl_state, deterministic=True)
            action = int(action)
            self.current_action_idx[dpid] = action

            decision = {
                'timestamp': time.time(),
                'switch_id': dpid,
                'port_no': port_no,
                'lstm_prob': lstm_prob,
                'action_idx': action,
                'polling_interval': config.ACTION_MAPPING[action],
                'routing_path': "s2" if lstm_prob > config.RISK_THRESHOLD else "s1",
                'engine': self.engine,
                'latency_ms': round(self.last_latency_ms, 4)
            }
            self.latest_decisions[port_key] = decision
            decisions.append(decision)
        return decisions


inference: InferenceEngine | None = None
producer: KafkaProducer | None = None


def build_engine():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Fitting MinMaxScaler on {config.DATASET_PATH}...")
    scaler = fit_scaler(load_telemetry(config.DATASET_PATH))

    print(f"Loading PyTorch LSTM model from {config.LSTM_PATH} on {device}...")
    lstm_model = load_model("lstm", config.LSTM_PATH, device)

    onnx_sessions = {}
    try:
        onnx_sessions["onnx"] = ort.InferenceSession(str(config.LSTM_ONNX_PATH), providers=['CPUExecutionProvider'])
        onnx_sessions["onnx_quantized"] = ort.InferenceSession(str(config.LSTM_QUANT_PATH), providers=['CPUExecutionProvider'])
        print("ONNX models loaded successfully.")
    except Exception as e:
        print(f"Warning: ONNX models failed to load. Only PyTorch will be available. Error: {e}")

    print(f"Loading RL agent from {config.RL_AGENT_PATH}...")
    rl_agent = DQN.load(config.RL_AGENT_PATH)
    return InferenceEngine(scaler, lstm_model, rl_agent, onnx_sessions, device)


def connect_producer(attempts=10):
    for i in range(attempts):
        try:
            p = KafkaProducer(
                bootstrap_servers=[config.BOOTSTRAP_SERVERS],
                value_serializer=lambda v: json.dumps(v).encode('utf-8')
            )
            print("Kafka Producer initialized.")
            return p
        except Exception as e:
            print(f"Waiting for Kafka Producer ({i+1}/{attempts}): {e}")
            time.sleep(3)
    return None


def consume_telemetry():
    print(f"Connecting consumer to {config.BOOTSTRAP_SERVERS} for topic {config.TELEMETRY_TOPIC}...")
    consumer = None
    for i in range(15):
        try:
            consumer = KafkaConsumer(
                config.TELEMETRY_TOPIC,
                bootstrap_servers=[config.BOOTSTRAP_SERVERS],
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
        # Micro-batch polling (5ms poll window)
        records = consumer.poll(timeout_ms=5, max_records=500)
        events = [m.value for messages in records.values() for m in messages]
        if not events:
            time.sleep(0.001)
            continue

        TELEMETRY_EVENTS.inc(len(events))
        decisions = inference.process(events)
        if producer:
            for decision in decisions:
                producer.send(config.CONTROL_TOPIC, value=decision)
            producer.flush()
        DECISIONS_PRODUCED.inc(len(decisions))


@asynccontextmanager
async def lifespan(_app):
    global inference, producer
    inference = build_engine()
    ACTIVE_ENGINE.set(0)
    producer = connect_producer()
    threading.Thread(target=consume_telemetry, daemon=True).start()
    print("Kafka Consumer Thread started.")
    yield


app = FastAPI(title="SDN Predictive Telemetry Inference Service", lifespan=lifespan)


@app.get("/status")
def get_status():
    return {
        "status": "Vectorized ONNX Inference service online",
        "device": str(inference.device),
        "bootstrap_servers": config.BOOTSTRAP_SERVERS,
        "current_engine": inference.engine,
        "latency_percentiles_ms": inference.latency_percentiles(),
        "models_loaded": {
            "pytorch": str(config.LSTM_PATH),
            "onnx": str(config.LSTM_ONNX_PATH) if "onnx" in inference.onnx_sessions else None,
            "onnx_quantized": str(config.LSTM_QUANT_PATH) if "onnx_quantized" in inference.onnx_sessions else None,
        }
    }


@app.post("/set-engine")
def set_engine(engine: str):
    try:
        inference.set_engine(engine)
    except ValueError as e:
        return {"status": "error", "message": str(e)}
    ACTIVE_ENGINE.set(ENGINES.index(engine))
    print(f"--- Swapped Vectorized Inference Engine to: {engine} ---")
    return {"status": "success", "engine": engine}


@app.get("/latest")
def get_latest():
    pct = inference.latency_percentiles()
    return {
        "telemetry": inference.latest_telemetry,
        "decisions": inference.latest_decisions,
        "current_engine": inference.engine,
        "last_latency_ms": round(inference.last_latency_ms, 4),
        "p50_latency_ms": pct["p50"],
        "p99_latency_ms": pct["p99"],
    }


@app.get("/metrics")
def get_prometheus_metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/trigger-spike")
def trigger_spike():
    if producer:
        producer.send(config.SIMULATION_TOPIC, value={"command": "spike"})
        producer.flush()
        print("--- Broadcasted Spike Command to Kafka: network-simulation ---")
        return {"status": "success", "message": "Congestion spike triggered"}
    return {"status": "error", "message": "Kafka producer not connected"}


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
