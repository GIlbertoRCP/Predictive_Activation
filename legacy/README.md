# legacy/ - the original course-project stack (CSCI 4930, CU Denver)

The first version of this project: Mininet fat-tree + os-ken controller -> Kafka -> FastAPI inference
(PyTorch LSTM + DQN) -> Streamlit dashboard. It is kept because it is a working end-to-end integration
and the starting point for a hardware-in-the-loop version of the controller (see the main README roadmap).

Read these caveats before quoting anything from it:

* **`telemetry_dataset.csv` (now `../data/mininet_capture_2026-04.csv`) cannot support an ML claim.** Every row has
  `rx_loss = tx_loss = avg_queue_depth = 0`; "latency" is the OpenFlow echo RTT of the *switch*, copied to all its
  ports, and reaches 3.7 s. A rule that predicts "next step is congested iff the current step is" scores
  **F1 = 0.993** on it, so the LSTM F1 in the old README measured autocorrelation, not forecasting skill.
  This finding motivated the queueing simulator in `src/`.
* The `p50/p95/p99_latency_ms`, `packets_saved` and `collisions_avoided` values in `traffic_engine.py` are
  **hard-coded placeholders for the demo dashboard, not measurements.**
* ONNX / INT8 numbers in `onnx_benchmarks.json` are real but concern a model that predicts a trivial label.
