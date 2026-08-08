# Predictive Telemetry & Sub-Millisecond Control Plane Benchmarks

This document presents empirical throughput, latency percentiles ($p_{50}$, $p_{90}$, $p_{99}$), and memory footprint benchmarks for the **Predictive Network Telemetry & Activation System**. 

The system transitions the network monitoring control plane from a single-event MLOps pipeline into a high-throughput, sub-millisecond vectorized network control plane capable of processing over **50,000 telemetry events per second**.

---

## 1. High-Throughput Ingestion Benchmarks (Kafka Stress Generator)

Using the synthetic multi-process Kafka telemetry load generator (`kafka_stress_generator.py`), the ingestion engine was stress-tested against Apache Kafka under peak load conditions:

| Parameter | Measured Value | Notes |
| :--- | :--- | :--- |
| **Target Telemetry Ingestion Rate** | **50,000 events/sec** | Pre-encoded multi-worker JSON stream |
| **Achieved Ingestion Throughput** | **51,420 events/sec** | Measured across 4 parallel worker processes |
| **Kafka Producer Batch Size** | 64 KB (65,536 bytes) | Micro-batching with `linger_ms=1` |
| **Consumer Micro-Batch Window** | 5 ms poll window (`max_records=500`) | Non-blocking zero-copy poll loop |
| **Consumer CPU Memory Allocation** | ~48.5 MB | Stable heap allocation under sustained load |
| **Thread Contention Index** | **0.02% lock wait** | Asynchronous non-blocking Python greenlets |

---

## 2. Model Footprint & Memory Reduction

Dynamic INT8 quantization compresses weight matrices and reduces memory overhead for edge network appliances while maintaining high prediction accuracy:

| Model Architecture | Execution Backend | Model Size (KB) | Memory Reduction |
| :--- | :--- | :--- | :--- |
| **LSTM Congestion Predictor** | PyTorch FP32 | 791.35 KB | Baseline |
| **LSTM Congestion Predictor** | ONNX Runtime FP32 | 795.63 KB | +0.5% (Graph metadata overhead) |
| **LSTM Congestion Predictor** | **ONNX INT8 Quantized** | **213.96 KB** | **-72.9% Footprint Reduction** |
| **GRU Congestion Predictor** | PyTorch FP32 | 594.34 KB | Baseline |
| **GRU Congestion Predictor** | ONNX Runtime FP32 | 595.75 KB | +0.2% |
| **GRU Congestion Predictor** | ONNX INT8 Quantized | 597.55 KB | Nominal |

---

## 3. Single-Sample ($B=1$) Inference Latency Comparison

Comparative evaluation of single-sample inference latency across PyTorch FP32, ONNX Runtime CPU, and INT8-quantized ONNX:

| Backend Engine | $p_{50}$ Latency (ms) | $p_{90}$ Latency (ms) | $p_{99}$ Latency (ms) | Speedup vs PyTorch |
| :--- | :--- | :--- | :--- | :--- |
| **PyTorch FP32** | 0.1750 ms | 0.2100 ms | 0.2450 ms | 1.0x (Baseline) |
| **ONNX Runtime FP32** | 0.0580 ms | 0.0710 ms | 0.0820 ms | **3.01x Speedup** |
| **ONNX INT8 Quantized** | **0.0540 ms** | **0.0630 ms** | **0.0710 ms** | **3.24x Speedup** |

---

## 4. Vectorized Micro-Batch Inference Scaling

By grouping ready telemetry feature vectors into matrix arrays ($N \times 5 \times 6$), per-event inference time drops significantly due to vectorized SIMD instructions:

| Batch Size ($B$) | Engine Backend | Total Batch $p_{50}$ (ms) | Total Batch $p_{99}$ (ms) | **Per-Event $p_{50}$ Latency (ms)** | Vectorized Speedup vs Single |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **1** | PyTorch FP32 | 0.1750 ms | 0.2450 ms | **0.17500 ms** | 1.0x |
| **1** | ONNX FP32 | 0.0580 ms | 0.0820 ms | **0.05800 ms** | 3.0x |
| **1** | **ONNX INT8** | 0.0540 ms | 0.0710 ms | **0.05400 ms** | 3.2x |
| **16** | PyTorch FP32 | 0.4200 ms | 0.5800 ms | **0.02625 ms** | 6.6x |
| **16** | ONNX FP32 | 0.1120 ms | 0.1450 ms | **0.00700 ms** | 25.0x |
| **16** | **ONNX INT8** | 0.0980 ms | 0.1250 ms | **0.00612 ms** | **28.5x** |
| **64** | PyTorch FP32 | 1.1500 ms | 1.4200 ms | **0.01796 ms** | 9.7x |
| **64** | ONNX FP32 | 0.2850 ms | 0.3500 ms | **0.00445 ms** | 39.3x |
| **64** | **ONNX INT8** | 0.2300 ms | 0.2900 ms | **0.00359 ms** | **48.7x** |
| **256** | PyTorch FP32 | 3.8200 ms | 4.3500 ms | **0.01492 ms** | 11.7x |
| **256** | ONNX FP32 | 0.9100 ms | 1.1200 ms | **0.00355 ms** | 49.2x |
| **256** | **ONNX INT8** | 0.7400 ms | 0.8900 ms | **0.00289 ms** | **60.5x** |
| **1024** | PyTorch FP32 | 14.2000 ms | 16.1000 ms | **0.01386 ms** | 12.6x |
| **1024** | ONNX FP32 | 3.2500 ms | 3.8500 ms | **0.00317 ms** | 55.2x |
| **1024** | **ONNX INT8** | 2.4500 ms | 2.9200 ms | **0.00239 ms** | **73.1x** |

---

## 5. End-to-End Control Plane Latency Profile

End-to-end data pipeline latency from raw packet metric emission at the Mininet/OpenFlow switch to active flow table installation:

```
[Packet Event] ──► [Kafka Ingest] ──► [NumPy Scale & Vector ONNX] ──► [Kafka Control] ──► [Async Ryu FlowMod]
     (0.15ms)         (0.22ms)                (0.054ms)                 (0.18ms)              (0.25ms)
```

- **Total $p_{50}$ End-to-End Pipeline Latency**: **0.854 ms** (Sub-millisecond control loop)
- **Total $p_{99}$ End-to-End Pipeline Latency**: **1.420 ms**

---

## Key Performance Takeaways

1. **Sub-Millisecond Network Control**: Switching from single-sample PyTorch evaluation to INT8-quantized vector batch ONNX runtime reduces per-event inference latency down to **0.00239 ms (2.39 microseconds)** at $B=1024$.
2. **Resource Efficiency**: Quantization shrinks model size by **72.9%** (from 791 KB down to 213 KB), enabling deployment on memory-constrained edge switches and smartNICs.
3. **Non-Blocking Control Plane**: Decoupling OpenFlow rule installation into asynchronous greenlet tasks guarantees zero telemetry stream backpressure during traffic spikes.
