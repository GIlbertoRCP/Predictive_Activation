#!/usr/bin/env python3
"""
High-Throughput Synthetic Kafka Telemetry Stress Generator
Floods Kafka topic [network-telemetry] with up to 50,000+ telemetry events/second
to benchmark thread contention, batching throughput, and memory allocations.
"""

import os
import sys
import time
import json
import random
import argparse
import multiprocessing as mp
from kafka import KafkaProducer

def worker_producer(worker_id, target_rate_per_worker, duration, bootstrap_servers, topic, batch_size):
    """
    Worker process that generates pre-serialized telemetry events and streams them to Kafka.
    """
    try:
        producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            acks=0,  # High-throughput asynchronous sending
            batch_size=batch_size,
            linger_ms=1,  # Micro-batching buffer
            compression_type='gzip',
            value_serializer=None  # Pre-serialized bytes
        )
    except Exception as e:
        print(f"[Worker {worker_id}] Error connecting to Kafka broker ({bootstrap_servers}): {e}")
        return 0, 0.0

    # Pre-generate candidate JSON byte strings to avoid serialization bottlenecks during loop
    switches = [1, 2, 3, 4]
    ports = [1, 2, 3, 4]
    pre_encoded_samples = []
    
    for _ in range(1000):
        dpid = random.choice(switches)
        port_no = random.choice(ports)
        payload = {
            'timestamp': time.time(),
            'switch_id': dpid,
            'port_no': port_no,
            'rx_mbps': round(random.uniform(10.0, 950.0), 4),
            'tx_mbps': round(random.uniform(10.0, 950.0), 4),
            'rx_loss': round(random.uniform(0.0, 0.05), 6),
            'tx_loss': round(random.uniform(0.0, 0.05), 6),
            'avg_queue_depth': random.randint(0, 150),
            'latency_ms': round(random.uniform(0.5, 45.0), 4)
        }
        pre_encoded_samples.append(json.dumps(payload).encode('utf-8'))

    sample_count = len(pre_encoded_samples)
    total_sent = 0
    start_time = time.perf_counter()
    end_time = start_time + duration

    # Rate timing mechanism
    interval = 0.005 # 5ms burst loops
    events_per_burst = int(target_rate_per_worker * interval) if target_rate_per_worker > 0 else 5000

    idx = 0
    while time.perf_counter() < end_time:
        burst_start = time.perf_counter()
        for _ in range(events_per_burst):
            msg_bytes = pre_encoded_samples[idx % sample_count]
            producer.send(topic, value=msg_bytes)
            idx += 1
            total_sent += 1
        
        # Pacing yield
        if target_rate_per_worker > 0:
            elapsed = time.perf_counter() - burst_start
            sleep_time = interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    producer.flush()
    producer.close()
    elapsed_total = time.perf_counter() - start_time
    return total_sent, elapsed_total


def run_stress_test(target_rate=50000, duration=10, bootstrap_servers="localhost:9092", topic="network-telemetry", num_workers=4, batch_size=65536):
    print("=" * 70)
    print(f"   KAFKA TELEMETRY STRESS GENERATOR (Target: {target_rate:,} events/sec)")
    print("=" * 70)
    print(f" Broker: {bootstrap_servers}")
    print(f" Topic: {topic}")
    print(f" Target Duration: {duration}s")
    print(f" Worker Processes: {num_workers}")
    print(f" Batch Size: {batch_size / 1024:.1f} KB")
    print("-" * 70)

    rate_per_worker = target_rate // num_workers if num_workers > 0 else target_rate

    pool = mp.Pool(processes=num_workers)
    tasks = []
    for w in range(num_workers):
        t = pool.apply_async(worker_producer, (w, rate_per_worker, duration, bootstrap_servers, topic, batch_size))
        tasks.append(t)

    pool.close()
    pool.join()

    total_events = 0
    max_elapsed = 0.0
    for t in tasks:
        sent, elapsed = t.get()
        total_events += sent
        if elapsed > max_elapsed:
            max_elapsed = elapsed

    actual_rate = total_events / max_elapsed if max_elapsed > 0 else 0
    print("\n" + "=" * 70)
    print("   STRESS TEST RESULTS")
    print("=" * 70)
    print(f" Total Telemetry Events Sent: {total_events:,}")
    print(f" Effective Benchmark Time:   {max_elapsed:.3f} seconds")
    print(f" Achieved Ingestion Rate:     {actual_rate:,.2f} events/second")
    print(f" Target Rate Satisfaction:   {(actual_rate / target_rate) * 100:.1f}%")
    print("=" * 70)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Synthetic Kafka Telemetry Stress Generator")
    parser.add_argument("--rate", type=int, default=50000, help="Target telemetry events per second (default: 50000)")
    parser.add_argument("--duration", type=float, default=10.0, help="Duration of stress test in seconds (default: 10.0)")
    parser.add_argument("--bootstrap-servers", type=str, default="localhost:9092", help="Kafka bootstrap servers")
    parser.add_argument("--topic", type=str, default="network-telemetry", help="Target Kafka topic")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel worker processes (default: 4)")
    parser.add_argument("--batch-size", type=int, default=65536, help="Kafka producer batch size in bytes (default: 65536)")

    args = parser.parse_args()
    run_stress_test(
        target_rate=args.rate,
        duration=args.duration,
        bootstrap_servers=args.bootstrap_servers,
        topic=args.topic,
        num_workers=args.workers,
        batch_size=args.batch_size
    )
