"""Export the forecasters to ONNX, quantize them to INT8, and benchmark all
three backends. Writes benchmarks/results.json.

    python -m predictive_activation.optimize
"""

import json
import os
import platform
import time
from datetime import datetime, timezone

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic

from predictive_activation import config
from predictive_activation.data import build_sequences
from predictive_activation.models import load_model

BATCH_SIZES = [1, 16, 64, 256, 1024]


def cpu_name():
    if platform.system() == "Windows":
        import subprocess
        try:
            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
                capture_output=True, text=True, timeout=10,
            )
            if out.stdout.strip():
                return out.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            pass
    if os.path.exists("/proc/cpuinfo"):
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def environment_info():
    return {
        "measured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "cpu": cpu_name(),
        "logical_cores": os.cpu_count(),
        "os": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "onnxruntime": ort.__version__,
        "onnx": onnx.__version__,
    }


def benchmark_batch(runner_func, input_data, num_runs=500, warmup=50):
    for _ in range(warmup):
        runner_func(input_data)

    latencies = np.empty(num_runs)
    for i in range(num_runs):
        t0 = time.perf_counter()
        runner_func(input_data)
        latencies[i] = (time.perf_counter() - t0) * 1000.0

    p50, p90, p99 = np.percentile(latencies, [50, 90, 99])
    return {
        "mean_ms": round(float(latencies.mean()), 4),
        "p50_ms": round(float(p50), 4),
        "p90_ms": round(float(p90), 4),
        "p99_ms": round(float(p99), 4),
        "per_event_p50_ms": round(float(p50) / len(input_data), 6),
        "runs": num_runs,
    }


def export_onnx(model, onnx_path):
    # Legacy TorchScript exporter: writes a single self-contained .onnx file
    # with a dynamic batch axis.
    torch.onnx.export(
        model,
        torch.randn(1, config.SEQUENCE_LENGTH, len(config.FEATURES)),
        onnx_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        dynamo=False,
    )


def accuracy_parity(reference_fn, candidate_fn, windows):
    """Compare a candidate backend against FP32 PyTorch on real windows."""
    ref = reference_fn(windows).flatten()
    cand = candidate_fn(windows).flatten()
    ref_dec = ref > config.RISK_THRESHOLD
    cand_dec = cand > config.RISK_THRESHOLD
    return {
        "n_windows": int(len(windows)),
        "max_abs_diff": round(float(np.abs(ref - cand).max()), 6),
        "mean_abs_diff": round(float(np.abs(ref - cand).mean()), 6),
        "decision_agreement": round(float((ref_dec == cand_dec).mean()), 6),
        "positive_decisions_fp32": int(ref_dec.sum()),
    }


def export_and_benchmark(arch, pth_path, onnx_path, quant_path, real_windows, num_runs=500):
    print(f"\n=== {arch.upper()}: {pth_path} ===")
    if not os.path.exists(pth_path):
        print(f"Skipping {pth_path} (file not found).")
        return None

    model = load_model(arch, pth_path)
    export_onnx(model, onnx_path)
    quantize_dynamic(onnx_path, quant_path, weight_type=QuantType.QInt8)
    print(f"Exported {onnx_path} and quantized {quant_path}")

    sizes = {
        "pytorch": round(os.path.getsize(pth_path) / 1024.0, 2),
        "onnx": round(os.path.getsize(onnx_path) / 1024.0, 2),
        "onnx_quantized": round(os.path.getsize(quant_path) / 1024.0, 2),
    }
    print(f"File sizes (KB): {sizes}")

    ort_session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_quant_session = ort.InferenceSession(str(quant_path), providers=["CPUExecutionProvider"])
    # Which op types actually got quantized tells you whether INT8 did anything.
    quant_ops = sorted({n.op_type for n in onnx.load(str(quant_path)).graph.node
                        if "Integer" in n.op_type or "Quant" in n.op_type})

    def run_pytorch(x_np):
        with torch.no_grad():
            return model(torch.from_numpy(x_np)).numpy()

    def run_onnx(x_np):
        return ort_session.run(["output"], {"input": x_np})[0]

    def run_quant(x_np):
        return ort_quant_session.run(["output"], {"input": x_np})[0]

    runners = {"pytorch": run_pytorch, "onnx": run_onnx, "onnx_quantized": run_quant}
    rng = np.random.default_rng(config.SEED)
    batch_results = {}
    for b in BATCH_SIZES:
        idx = rng.choice(len(real_windows), size=b, replace=len(real_windows) < b)
        x = np.ascontiguousarray(real_windows[idx])
        batch_results[str(b)] = {name: benchmark_batch(fn, x, num_runs) for name, fn in runners.items()}
        row = " | ".join(f"{name} p50={r['p50_ms']:.4f}ms" for name, r in batch_results[str(b)].items())
        print(f"B={b:4d} | {row}")

    parity = {
        "onnx": accuracy_parity(run_pytorch, run_onnx, real_windows),
        "onnx_quantized": accuracy_parity(run_pytorch, run_quant, real_windows),
    }
    print(f"INT8 parity: {parity['onnx_quantized']}")

    return {
        "file_sizes_kb": sizes,
        "quantized_op_types": quant_ops,
        "batch_benchmarks": batch_results,
        "accuracy_parity": parity["onnx_quantized"],
        "accuracy_parity_by_engine": parity,
    }


def main(num_runs=500):
    real_windows, _, _ = build_sequences()
    results = {"environment": environment_info()}
    print(f"Environment: {results['environment']}")

    for arch, pth in (("lstm", config.LSTM_PATH), ("gru", config.GRU_PATH)):
        onnx_path = config.MODELS_DIR / f"{arch}_forecaster.onnx"
        quant_path = config.MODELS_DIR / f"{arch}_forecaster_quant.onnx"
        res = export_and_benchmark(arch, pth, onnx_path, quant_path, real_windows, num_runs)
        if res:
            results[arch] = res

    config.BENCHMARKS_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.BENCHMARK_RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {config.BENCHMARK_RESULTS_PATH}")
    return results


if __name__ == "__main__":
    main()
