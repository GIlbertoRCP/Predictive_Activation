"""
ai_workload_generator.py - Realistic AI Distributed Training Workload & Telemetry Generator

Simulates high-performance GPU cluster fabrics (16 GPUs on Spine-Leaf fabric)
executing Distributed Data Parallel (DDP) / Ring All-Reduce training rounds.
Generates multi-class telemetry (Elephant tensor flows vs Mice control flows)
and simulates incast buffer queue saturation events.
"""

import time
import json
import random
import argparse
import threading
from typing import Dict, List, Any, Optional
import requests

try:
    from kafka import KafkaProducer, KafkaConsumer
    HAS_KAFKA = True
except ImportError:
    HAS_KAFKA = False


class AIClusterWorkloadGenerator:
    """
    Simulates distributed AI training workloads across 16 GPU workers (h1..h16).
    Emits switch port telemetry matching the trained PyTorch/ONNX models.
    """

    def __init__(self, kafka_broker: Optional[str] = "localhost:9092", 
                 fastapi_url: str = "http://localhost:8000"):
        self.kafka_broker = kafka_broker
        self.fastapi_url = fastapi_url
        self.producer = None
        self.kafka_connected = False
        
        # State tracking
        self.phase = "COMPUTE"  # COMPUTE, ALL_REDUCE, INCAST_SPIKE
        self.phase_counter = 0
        self.incast_trigger_counter = 0
        self.running = False
        
        # Topology switch ports mapping (dpid -> active ports)
        # s1, s2 (Core Spines): ports 1, 2
        # s3, s4 (Aggregations): ports 1, 2 (to cores), 3, 4 (to ToRs)
        # s5..s8 (Edge ToRs): port 1 (to Agg), ports 2..5 (to GPUs)
        self.switches = {
            1: [1, 2],
            2: [1, 2],
            3: [1, 2, 3, 4],
            4: [1, 2, 3, 4],
            5: [1, 2, 3, 4, 5],
            6: [1, 2, 3, 4, 5],
            7: [1, 2, 3, 4, 5],
            8: [1, 2, 3, 4, 5]
        }
        
        self._init_kafka()

    def _init_kafka(self):
        if not HAS_KAFKA or not self.kafka_broker:
            return
        try:
            self.producer = KafkaProducer(
                bootstrap_servers=[self.kafka_broker],
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                request_timeout_ms=1000
            )
            self.kafka_connected = True
            print(f"[AI Workload Generator] Successfully connected to Kafka at {self.kafka_broker}")
            
            # Start simulation commands listener
            threading.Thread(target=self._listen_for_kafka_triggers, daemon=True).start()
        except Exception as e:
            print(f"[AI Workload Generator] Kafka unavailable ({e}). Defaulting to HTTP streaming to {self.fastapi_url}")
            self.kafka_connected = False

    def _listen_for_kafka_triggers(self):
        try:
            consumer = KafkaConsumer(
                'network-simulation',
                bootstrap_servers=[self.kafka_broker],
                value_deserializer=lambda x: json.loads(x.decode('utf-8')),
                auto_offset_reset='latest'
            )
            for msg in consumer:
                cmd = msg.value.get("command")
                if cmd in ["spike", "all_reduce", "incast"]:
                    self.trigger_incast(rounds=12)
        except Exception:
            pass

    def trigger_incast(self, rounds: int = 15):
        """Immediately forces the workload generator into an Incast Buffer Saturation phase."""
        self.incast_trigger_counter = rounds
        print(f"\n🔥 [BURST INJECTED] High-Volume Incast Saturation Triggered for {rounds} cycles! 🔥\n")

    def generate_tick(self) -> List[Dict[str, Any]]:
        """
        Generates one simulation tick across the AI cluster fabric.
        Returns a list of port telemetry records.
        """
        now = time.time()
        self.phase_counter += 1

        # Phase state machine:
        if self.incast_trigger_counter > 0:
            current_phase = "INCAST_SPIKE"
            self.incast_trigger_counter -= 1
        elif self.phase_counter % 25 < 14:
            current_phase = "COMPUTE"
        else:
            current_phase = "ALL_REDUCE"

        self.phase = current_phase
        telemetry_batch = []

        for dpid, ports in self.switches.items():
            for port in ports:
                record = self._generate_port_metric(dpid, port, current_phase, now)
                telemetry_batch.append(record)

        return telemetry_batch

    def _generate_port_metric(self, dpid: int, port: int, phase: str, timestamp: float) -> Dict[str, Any]:
        """Generates realistic telemetry features conditioned on the active AI training phase."""
        is_core_s1 = (dpid == 1)

        if phase == "COMPUTE":
            # GPUs performing forward/backward pass. Low network traffic, light mice control flows.
            rx_mbps = max(0.5, random.gauss(12.0, 3.0))
            tx_mbps = max(0.5, random.gauss(11.5, 3.0))
            rx_loss = 0.0
            tx_loss = 0.0
            queue_depth = max(100, int(random.gauss(2500, 500)))
            latency_ms = max(0.8, random.gauss(2.5, 0.4))
            flow_type = "mice"

        elif phase == "ALL_REDUCE":
            # Distributed All-Reduce parameter exchange. High throughput, balanced queues.
            rx_mbps = max(30.0, random.gauss(72.0, 8.0))
            tx_mbps = max(30.0, random.gauss(70.0, 8.0))
            rx_loss = max(0.0, random.gauss(0.00005, 0.00002))
            tx_loss = 0.0
            queue_depth = int(random.gauss(28000, 4000))
            latency_ms = max(5.0, random.gauss(15.0, 3.0))
            flow_type = "elephant"

        else: # INCAST_SPIKE
            # Simultaneous tensor transfer collision on Spine 1 (s1)
            if is_core_s1:
                rx_mbps = max(88.0, random.gauss(96.0, 3.0))
                tx_mbps = max(85.0, random.gauss(94.0, 3.0))
                rx_loss = max(0.01, random.gauss(0.045, 0.01))
                tx_loss = max(0.005, random.gauss(0.02, 0.005))
                queue_depth = int(random.gauss(185000, 20000))
                latency_ms = max(120.0, random.gauss(175.0, 25.0)) # Exceeds 100ms threshold!
            else:
                # Other switches see moderate/rerouted load
                rx_mbps = max(20.0, random.gauss(55.0, 10.0))
                tx_mbps = max(20.0, random.gauss(52.0, 10.0))
                rx_loss = 0.0001
                tx_loss = 0.0
                queue_depth = int(random.gauss(35000, 5000))
                latency_ms = max(4.0, random.gauss(12.0, 2.0))
            flow_type = "elephant"

        # Pair a simulated host flow
        src_gpu = f"h{random.randint(1, 8)}"
        dst_gpu = f"h{random.randint(9, 16)}"

        return {
            'timestamp': timestamp,
            'switch_id': dpid,
            'port_no': port,
            'rx_mbps': round(rx_mbps, 4),
            'tx_mbps': round(tx_mbps, 4),
            'rx_loss': round(rx_loss, 6),
            'tx_loss': round(tx_loss, 6),
            'avg_queue_depth': queue_depth,
            'latency_ms': round(latency_ms, 4),
            'flow_type': flow_type,
            'src_host': src_gpu,
            'dst_host': dst_gpu,
            'training_phase': phase
        }

    def emit_batch(self, batch: List[Dict[str, Any]]):
        """Dispatches telemetry batch to Kafka if online, or directly to FastAPI inference service."""
        # 1. Try Kafka
        if self.kafka_connected and self.producer:
            try:
                for item in batch:
                    self.producer.send('network-telemetry', value=item)
                self.producer.flush()
                return
            except Exception:
                self.kafka_connected = False

        # 2. Direct HTTP streaming fallback to FastAPI
        try:
            requests.post(
                f"{self.fastapi_url}/ingest-telemetry", 
                json={"batch": batch}, 
                timeout=0.8
            )
        except Exception:
            pass

    def run_loop(self, interval: float = 1.0):
        """Continuous simulation loop."""
        self.running = True
        print(f"[AI Workload Generator] Streaming AI Cluster Telemetry (interval={interval}s)...")
        while self.running:
            start_time = time.time()
            batch = self.generate_tick()
            self.emit_batch(batch)
            elapsed = time.time() - start_time
            sleep_time = max(0.05, interval - elapsed)
            time.sleep(sleep_time)

    def stop(self):
        self.running = False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="AI Cluster Workload Generator")
    parser.add_argument("--broker", default="localhost:9092", help="Kafka broker address")
    parser.add_argument("--fastapi", default="http://localhost:8000", help="FastAPI backend URL")
    parser.add_argument("--speed", type=float, default=1.0, help="Stream speed multiplier")
    args = parser.parse_args()

    generator = AIClusterWorkloadGenerator(kafka_broker=args.broker, fastapi_url=args.fastapi)
    interval = max(0.2, 1.0 / args.speed)
    generator.run_loop(interval=interval)
