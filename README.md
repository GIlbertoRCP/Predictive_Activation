# Predictive Activation

**Can forecasting congestion beat reacting to it, and how much telemetry does it save?**
A queueing simulator of an AI-training fabric (8 all-reduce jobs sharing 4 spine links), a forecaster that predicts
bursts from *stale* switch telemetry, and a controller that re-places flows before collisions happen. Reactive
and predictive controllers share identical placement logic; they differ only in the forecast they use.

Originally a CU Denver CSCI 4930 team project (Diego Alas, Gilberto Romero-Cano, Corey Green, JJ Wagner). Version 1.0
re-frames it around a measurable question and a reproducible benchmark.

## Headline results

**Forecasting** (held-out test episodes, PR-AUC; random guessing = 0.21 link / 0.29 flow):

| Telemetry polled every | 1 s | 5 s | 10 s |
|---|---|---|---|
| Link congestion 10 s ahead: utilization-threshold alarm | 0.59 | 0.45 | 0.36 |
| Link congestion 10 s ahead: logistic regression | 0.72 | 0.63 | 0.58 |
| Link congestion 10 s ahead: **gradient-boosted trees** | **0.79** | **0.69** | **0.63** |
| Flow burst 2-10 s ahead: persistence (what a reactive system assumes) | 0.41 | 0.34 | 0.29 |
| Flow burst 2-10 s ahead: **learned forecaster** | **0.84** | **0.79** | **0.72** |

Stale telemetry degrades every model, which is the cost/benefit tradeoff the controller exploits.

**Closed-loop control** (preliminary: 6 validation episodes, 30 simulated minutes each; the full 30-episode test
run is `make experiments`):

| Policy | Link-time congested | Traffic dropped | Polls / min |
|---|---|---|---|
| No control | ~4.2% | ~0.39% | 8 |
| Reactive balancer, poll 1 s | 2.7% | 0.27% | 240 |
| **Predictive, poll 5 s** | **2.1%** | **0.12%** | **48** |
| Clairvoyant upper bound | ~2.4% | ~0.13% | 240 |

Predictive control with 5x less telemetry beat the reactive balancer polling every second. Treat these as
preliminary until the full run is reproduced; the clairvoyant row used a less-tuned setting, so it is not a strict ceiling.

**Negative results (kept on purpose):** risk-adaptive polling (10 s heartbeat, escalate to 1 s) did *not* beat fixed
5 s polling, and a threshold-triggered reroute made things worse than doing nothing.

## Why the simulator? (a finding about the original data)

The original Mininet capture (`data/mininet_capture_2026-04.csv`) has zero packet loss and zero queue depth in every
row, and its "latency" is controller echo-RTT. Predicting "next step congested iff current step congested" scores
F1 = 0.993, so the old LSTM score measured autocorrelation, not forecasting. The new simulator
(`src/predictive_activation/fabric.py`) models real queue dynamics: bursts ramp, buffers fill, drops trigger back-off.

## Layout

```
src/predictive_activation/   fabric (simulator) · features · forecasters · policies · engine · experiments · report
                             torch_models.py  optional LSTM/GRU on the same task (see note)
tests/                       12 unit tests (determinism, queue bounds, feature correctness, no-fabric-access for non-oracle policies)
legacy/                      original Mininet/os-ken/Kafka/FastAPI/Streamlit stack + caveats (legacy/README.md)
data/                        original capture, kept for the finding above
```

## Run it

```bash
pip install -r requirements.txt
make test          # seconds
make quick         # ~2 min smoke run of the whole pipeline
make experiments   # full benchmark, ~1 hour on 2 cores; writes results/*.json
make report        # figures + results/RESULTS.md
```

Seeds are split (train 0-59, validation 100-105, test 1000-1029); each policy sees the same test episodes.

## Limitations (read before quoting)

* Results come from a simulator, not hardware; it is a fluid model with 1 s ticks and synthetic periodic job traffic.
* Controller numbers above are from validation seeds; headroom for predictive/clairvoyant landed on the edge of the tuning grid (0.65).
* `torch_models.py` was not run in the build environment (no PyTorch); run `python -m predictive_activation.torch_models` and compare before citing.
* Next step: drive the same policy from the Mininet/os-ken controller in `legacy/`.

MIT License.
