import numpy as np
import onnxruntime as ort
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic

from predictive_activation import config
from predictive_activation.optimize import accuracy_parity, benchmark_batch, export_onnx


def test_onnx_export_matches_pytorch_with_dynamic_batch(lstm_model, tmp_path):
    onnx_path = tmp_path / "lstm.onnx"
    export_onnx(lstm_model, onnx_path)
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    for batch in (1, 7, 64):
        x = np.random.default_rng(batch).random((batch, config.SEQUENCE_LENGTH, len(config.FEATURES)), dtype=np.float32)
        with torch.no_grad():
            expected = lstm_model(torch.from_numpy(x)).numpy()
        np.testing.assert_allclose(session.run(["output"], {"input": x})[0], expected, atol=1e-5)


def test_int8_quantization_stays_close(lstm_model, tmp_path):
    onnx_path, quant_path = tmp_path / "lstm.onnx", tmp_path / "lstm_q.onnx"
    export_onnx(lstm_model, onnx_path)
    quantize_dynamic(onnx_path, quant_path, weight_type=QuantType.QInt8)
    assert quant_path.stat().st_size < onnx_path.stat().st_size * 0.5

    session = ort.InferenceSession(str(quant_path), providers=["CPUExecutionProvider"])
    x = np.random.default_rng(0).random((256, config.SEQUENCE_LENGTH, len(config.FEATURES)), dtype=np.float32)

    def run_pt(a):
        with torch.no_grad():
            return lstm_model(torch.from_numpy(a)).numpy()

    parity = accuracy_parity(run_pt, lambda a: session.run(["output"], {"input": a})[0], x)
    assert parity["n_windows"] == 256
    assert parity["max_abs_diff"] < 0.05


def test_benchmark_batch_reports_ordered_percentiles():
    stats = benchmark_batch(lambda a: a.sum(), np.ones((4, 5, 6), dtype=np.float32), num_runs=50, warmup=5)
    assert stats["p50_ms"] <= stats["p90_ms"] <= stats["p99_ms"]
    assert stats["runs"] == 50
    # p50_ms is rounded to 4 decimals, so compare within that rounding error.
    assert abs(stats["per_event_p50_ms"] * 4 - stats["p50_ms"]) <= 1e-4
