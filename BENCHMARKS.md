# Benchmarks

Every number here comes from [`benchmarks/results.json`](benchmarks/results.json), which `python -m predictive_activation.optimize` writes. Re-run that command to reproduce them on your machine.

> **Note:** An earlier version of this file reported Kafka throughput (51,420 events/s), lock-contention and end-to-end control-loop figures that no script in the repo produced. Those figures have been removed. A section comes back only once it has been measured.

## Environment

| | |
| :--- | :--- |
| CPU | AMD Ryzen 7 7700X (8 cores / 16 threads) |
| OS | Windows 11 (10.0.26200) |
| Python | 3.12.15 |
| PyTorch | 2.10.0 (CPU build, 8 intra-op threads) |
| ONNX Runtime | 1.31.1, `CPUExecutionProvider`, default threading |
| Method | 50 warm-up runs, then 500 timed runs per (batch size, backend). Inputs are real windows sampled from `data/telemetry_dataset.csv`. |
| Measured | 2026-10-30 |

Single-machine microbenchmarks vary by roughly ±10–30% between runs, especially at small batch sizes. Compare backends within a run, not across machines.

## 1. Model accuracy, and why it is not the headline

The checkpoints were retrained on 2026-10-30. The previously committed `.pth` files were left over from before the label bug fix and output ≈4×10⁻⁶ for every input, so the live system could never trigger a reroute.

| Model | F1 (random 80/20 split) | Precision | Recall |
| :--- | ---: | ---: | ---: |
| LSTM | 99.14% | 99.63% | 98.66% |
| GRU | 99.23% | 99.63% | 98.84% |
| **Persistence baseline** (predict that the next step matches the current one) | **99.30%** | 99.43% | 99.18% |

The congestion label changes on only 0.6% of steps, so copying the current state forward is as good as either network. On this dataset the recurrent models add nothing over that baseline. The random split also overlaps windows between train and test, which flatters the networks. A time-based split and a harder target (for example, predicting congestion several steps ahead) are planned.

**DQN polling agent:** over 64 evaluation episodes it averages 5,056 ± 6,421 reward. A fixed rule ("poll every 1s when risk > 0.7, otherwise every 30s") averages 4,990 ± 6,453. The two are indistinguishable.

## 2. Model size

| Model | PyTorch `.pth` | ONNX FP32 | ONNX INT8 (dynamic) | INT8 reduction |
| :--- | ---: | ---: | ---: | ---: |
| LSTM | 792.6 KB | 792.6 KB | **210.6 KB** | **−73.4%** |
| GRU | 595.6 KB | 594.7 KB | 596.5 KB | none |

ONNX Runtime's dynamic quantizer has an integer LSTM kernel (`DynamicQuantizeLSTM`) but no GRU equivalent. For the GRU, only the final linear layer gets quantized, so its size and speed are essentially unchanged.

## 3. INT8 accuracy parity (LSTM)

Measured against PyTorch FP32 on all 24,909 windows in the dataset:

| Backend | Max \|Δp\| | Mean \|Δp\| | Same reroute decision at 0.70 |
| :--- | ---: | ---: | ---: |
| ONNX FP32 | 0.000001 | 0.000000 | 100.00% |
| ONNX INT8 | 0.1669 | 0.0011 | 99.92% (21 of 24,909 windows flip) |

## 4. Inference latency

### LSTM

| Batch size | Backend | p50 (ms) | p90 (ms) | p99 (ms) | Per-event p50 (µs) | Speedup vs PyTorch (p50) |
| ---: | :--- | ---: | ---: | ---: | ---: | ---: |
| 1 | PyTorch FP32 | 0.2245 | 0.2450 | 0.3419 | 224.50 | 1.00x |
| 1 | ONNX Runtime FP32 | 0.0540 | 0.0618 | 0.0760 | 54.00 | 4.16x |
| 1 | ONNX Runtime INT8 | 0.0540 | 0.0553 | 0.0812 | 54.00 | 4.16x |
| 16 | PyTorch FP32 | 0.2986 | 0.3297 | 0.5030 | 18.66 | 1.00x |
| 16 | ONNX Runtime FP32 | 0.1153 | 0.1200 | 0.1582 | 7.21 | 2.59x |
| 16 | ONNX Runtime INT8 | 0.0877 | 0.0936 | 0.1632 | 5.48 | 3.40x |
| 64 | PyTorch FP32 | 0.4866 | 0.5486 | 0.7470 | 7.60 | 1.00x |
| 64 | ONNX Runtime FP32 | 0.3127 | 0.3283 | 0.4289 | 4.89 | 1.56x |
| 64 | ONNX Runtime INT8 | 0.1984 | 0.2153 | 0.3525 | 3.10 | 2.45x |
| 256 | PyTorch FP32 | 1.2584 | 1.3566 | 1.5779 | 4.92 | 1.00x |
| 256 | ONNX Runtime FP32 | 1.0922 | 1.1730 | 1.2808 | 4.27 | 1.15x |
| 256 | ONNX Runtime INT8 | 0.5804 | 0.6080 | 0.7415 | 2.27 | 2.17x |
| 1024 | PyTorch FP32 | 4.3109 | 4.5748 | 4.8570 | 4.21 | 1.00x |
| 1024 | ONNX Runtime FP32 | 4.2939 | 4.5787 | 5.5220 | 4.19 | 1.00x |
| 1024 | ONNX Runtime INT8 | 2.2147 | 2.3640 | 2.5958 | 2.16 | 1.95x |

### GRU

| Batch size | Backend | p50 (ms) | p90 (ms) | p99 (ms) | Per-event p50 (µs) | Speedup vs PyTorch (p50) |
| ---: | :--- | ---: | ---: | ---: | ---: | ---: |
| 1 | PyTorch FP32 | 0.3560 | 0.4040 | 0.5750 | 356.00 | 1.00x |
| 1 | ONNX Runtime FP32 | 0.0537 | 0.0615 | 0.0732 | 53.65 | 6.63x |
| 1 | ONNX Runtime INT8 | 0.0594 | 0.0659 | 0.0795 | 59.40 | 5.99x |
| 16 | PyTorch FP32 | 0.5247 | 0.6045 | 0.8057 | 32.79 | 1.00x |
| 16 | ONNX Runtime FP32 | 0.2248 | 0.2430 | 0.4716 | 14.05 | 2.33x |
| 16 | ONNX Runtime INT8 | 0.2172 | 0.2743 | 0.6123 | 13.58 | 2.42x |
| 64 | PyTorch FP32 | 0.8530 | 0.9537 | 1.1876 | 13.33 | 1.00x |
| 64 | ONNX Runtime FP32 | 0.6718 | 0.7119 | 0.8511 | 10.50 | 1.27x |
| 64 | ONNX Runtime INT8 | 0.6719 | 0.7179 | 0.8220 | 10.50 | 1.27x |
| 256 | PyTorch FP32 | 2.1399 | 2.3226 | 2.6819 | 8.36 | 1.00x |
| 256 | ONNX Runtime FP32 | 2.4858 | 2.6083 | 2.8283 | 9.71 | 0.86x |
| 256 | ONNX Runtime INT8 | 2.4859 | 2.5776 | 2.8238 | 9.71 | 0.86x |
| 1024 | PyTorch FP32 | 5.2933 | 5.5944 | 5.8916 | 5.17 | 1.00x |
| 1024 | ONNX Runtime FP32 | 10.2364 | 10.5056 | 10.9342 | 10.00 | 0.52x |
| 1024 | ONNX Runtime INT8 | 10.2658 | 10.6139 | 11.0503 | 10.02 | 0.52x |

### What the latency numbers show

- **At batch size 1, ONNX Runtime is about 4× faster than PyTorch for the LSTM and about 6× for the GRU.** Most of that gap is PyTorch's per-call overhead, not compute.
- **INT8 adds nothing at batch size 1 but roughly doubles LSTM throughput at large batches** (B=1024: 2.16 µs vs 4.21 µs per event).
- **Batching cuts per-event cost about 50× for PyTorch** (224 µs at B=1 vs 4.2 µs at B=1024). This is why the service micro-batches Kafka records.
- **For large GRU batches, ONNX Runtime FP32 is about 2× *slower* than PyTorch** on this CPU. This hasn't been investigated yet; thread settings are the first suspect.

## 5. Not yet re-measured

- **Kafka producer throughput.** Run `python -m predictive_activation.stress_generator --rate 50000 --duration 10` against `docker compose up -d kafka`. It measures what the producer sends, not consumer lag or end-to-end delivery.
- **End-to-end control-loop latency** (switch stats → decision → flow rule). This needs the Linux/Mininet setup in `legacy/sdn/`.
