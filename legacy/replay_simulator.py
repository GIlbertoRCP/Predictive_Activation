import csv
import json
import time
import argparse
import threading
import random
from kafka import KafkaProducer, KafkaConsumer

# Global state for simulation triggers
spike_counter = 0

def listen_for_simulation_commands(broker):
    global spike_counter
    print(f"Connecting Simulation Command Consumer to {broker}...")
    consumer = None
    for i in range(15):
        try:
            consumer = KafkaConsumer(
                'network-simulation',
                bootstrap_servers=[broker],
                value_deserializer=lambda x: json.loads(x.decode('utf-8')),
                auto_offset_reset='latest'
            )
            print("Simulation Command Consumer connected successfully.")
            break
        except Exception as e:
            print(f"Waiting for Simulation Consumer ({i+1}/15): {e}")
            time.sleep(3)
            
    if not consumer:
        print("Error: Could not start Simulation Command Consumer thread.")
        return

    for message in consumer:
        cmd = message.value
        if cmd.get("command") == "spike":
            print("\n🔥 TRIGGER RECEIVED: Injecting simulated congestion traffic spike for next 15 readings! 🔥")
            spike_counter = 15

def run_simulator(csv_path, broker, topic, speed, loop):
    global spike_counter
    print(f"Connecting to Kafka broker at {broker}...")
    producer = None
    for i in range(15):
        try:
            producer = KafkaProducer(
                bootstrap_servers=[broker],
                value_serializer=lambda v: json.dumps(v).encode('utf-8')
            )
            print("Successfully connected to Kafka.")
            break
        except Exception as e:
            print(f"Waiting for Kafka broker ({i+1}/15): {e}")
            time.sleep(3)
    
    if not producer:
        print("Error: Could not connect to Kafka. Exiting.")
        return

    # Start simulation commands listener thread
    listener_thread = threading.Thread(
        target=listen_for_simulation_commands, 
        args=(broker,), 
        daemon=True
    )
    listener_thread.start()

    print(f"Reading dataset from {csv_path}...")
    try:
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    except FileNotFoundError:
        print(f"Error: {csv_path} not found.")
        return
    
    # Sort rows by timestamp to preserve order
    rows.sort(key=lambda r: float(r['timestamp']))
    print(f"Loaded {len(rows)} rows.")

    iteration = 1
    while True:
        print(f"Starting playback iteration {iteration}...")
        prev_timestamp = None
        
        for row in rows:
            try:
                original_ts = float(row['timestamp'])
                
                if prev_timestamp is not None:
                    time_diff = original_ts - prev_timestamp
                    if time_diff > 0:
                        time.sleep(time_diff / speed)
                
                # Override timestamp to current system time
                current_time = time.time()
                
                if spike_counter > 0:
                    # Override telemetry values with anomalous congestion metrics
                    rx_val = float(92.0 + random.uniform(-3, 3))
                    tx_val = float(88.0 + random.uniform(-3, 3))
                    loss_rx = float(0.04 + random.uniform(0, 0.02))
                    loss_tx = float(0.03 + random.uniform(0, 0.02))
                    queue_val = int(8500 + random.randint(-500, 500))
                    latency_val = float(160.0 + random.uniform(-15, 15))
                    
                    payload = {
                        'timestamp': current_time,
                        'switch_id': int(row['switch_id']),
                        'port_no': int(row['port_no']),
                        'rx_mbps': rx_val,
                        'tx_mbps': tx_val,
                        'rx_loss': loss_rx,
                        'tx_loss': loss_tx,
                        'avg_queue_depth': queue_val,
                        'latency_ms': latency_val
                    }
                    spike_counter -= 1
                    print(f"  [SPIKE ACTIVE ({spike_counter} left)] - Injecting congestion on SW:{payload['switch_id']} P:{payload['port_no']}")
                else:
                    # standard telemetry values from CSV
                    payload = {
                        'timestamp': current_time,
                        'switch_id': int(row['switch_id']),
                        'port_no': int(row['port_no']),
                        'rx_mbps': float(row['rx_mbps']),
                        'tx_mbps': float(row['tx_mbps']),
                        'rx_loss': float(row['rx_loss']),
                        'tx_loss': float(row['tx_loss']),
                        'avg_queue_depth': float(row['avg_queue_depth']),
                        'latency_ms': float(row['latency_ms'])
                    }
                
                producer.send(topic, value=payload)
                prev_timestamp = original_ts
                
            except (ValueError, KeyError) as e:
                # Skip malformed rows
                continue
        
        if not loop:
            break
        iteration += 1
        print("Replay finished, looping back to the beginning.")

    # Flush remaining messages
    producer.flush()
    print("Simulator finished.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Replay telemetry dataset into Kafka.")
    parser.add_argument('--csv', type=str, default='telemetry_dataset.csv', help='Path to telemetry CSV file.')
    parser.add_argument('--broker', type=str, default='localhost:9092', help='Kafka bootstrap broker.')
    parser.add_argument('--topic', type=str, default='network-telemetry', help='Target Kafka topic.')
    parser.add_argument('--speed', type=float, default=2.0, help='Playback speed multiplier.')
    parser.add_argument('--no-loop', dest='loop', action='store_false', help='Do not loop indefinitely.')
    parser.set_defaults(loop=True)
    
    args = parser.parse_args()
    run_simulator(args.csv, args.broker, args.topic, args.speed, args.loop)
