import os
import json
import time
import torch
import torch.nn as nn
import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType

# --- 1. Model Definitions ---
class LSTMPredictor(nn.Module):
    def __init__(self, input_size=6, hidden_size=128, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x)
        # Slice at static index 4 (last step in sequence of length 5)
        out_last = out[:, 4, :]
        return self.sigmoid(self.fc(out_last))

class GRUPredictor(nn.Module):
    def __init__(self, input_size=6, hidden_size=128, num_layers=2):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.gru(x)
        # Slice at static index 4
        out_last = out[:, 4, :]
        return self.sigmoid(self.fc(out_last))

def benchmark_batch(runner_func, input_data, num_runs=500):
    latencies = []
    # Warmup
    for _ in range(50):
        _ = runner_func(input_data)
    
    for _ in range(num_runs):
        t0 = time.perf_counter()
        _ = runner_func(input_data)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(elapsed_ms)
        
    latencies = np.array(latencies)
    return {
        "mean_ms": round(float(np.mean(latencies)), 4),
        "p50_ms": round(float(np.percentile(latencies, 50)), 4),
        "p90_ms": round(float(np.percentile(latencies, 90)), 4),
        "p99_ms": round(float(np.percentile(latencies, 99)), 4),
        "per_event_p50_ms": round(float(np.percentile(latencies, 50)) / len(input_data), 6)
    }

def export_and_quantize(model_class, pth_path, onnx_path, quant_path):
    print(f"\n=======================================================")
    print(f" Processing and Benchmarking Model: {pth_path}")
    print(f"=======================================================")
    if not os.path.exists(pth_path):
        print(f"Skipping {pth_path} (File not found).")
        return None
    
    # Load PyTorch Model
    model = model_class()
    model.load_state_dict(torch.load(pth_path, map_location='cpu', weights_only=True))
    model.eval()
    
    # 1. Export to ONNX with dynamic batch size
    dummy_input = torch.randn(1, 5, 6)
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    print(f"Exported to Dynamic Batch ONNX: {onnx_path}")
    
    # 2. Dynamic INT8 Quantization
    quantize_dynamic(
        onnx_path,
        quant_path,
        weight_type=QuantType.QInt8,
        extra_options={'DisableShapeInference': True}
    )
    print(f"Quantized Dynamic INT8 ONNX generated: {quant_path}")
    
    # Model File Sizes (in KB)
    size_pth = os.path.getsize(pth_path) / 1024.0
    size_onnx = os.path.getsize(onnx_path) / 1024.0
    size_quant = os.path.getsize(quant_path) / 1024.0
    
    print(f"File Sizes -> PyTorch: {size_pth:.1f}KB | ONNX: {size_onnx:.1f}KB | INT8 Quantized: {size_quant:.1f}KB")
    
    ort_session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    ort_quant_session = ort.InferenceSession(quant_path, providers=['CPUExecutionProvider'])

    # Runners
    def run_pytorch(x_np):
        t_in = torch.tensor(x_np, dtype=torch.float32)
        with torch.no_grad():
            return model(t_in).numpy()

    def run_onnx(x_np):
        return ort_session.run(['output'], {'input': x_np})[0]

    def run_quant(x_np):
        return ort_quant_session.run(['output'], {'input': x_np})[0]

    batch_sizes = [1, 16, 64, 256, 1024]
    batch_results = {}

    for b in batch_sizes:
        print(f"\n--- Benchmarking Batch Size B = {b} ---")
        x_dummy = np.random.randn(b, 5, 6).astype(np.float32)
        
        res_pt = benchmark_batch(run_pytorch, x_dummy)
        res_onnx = benchmark_batch(run_onnx, x_dummy)
        res_quant = benchmark_batch(run_quant, x_dummy)

        print(f" B={b:4d} | PyTorch FP32  : p50={res_pt['p50_ms']:.4f}ms, p99={res_pt['p99_ms']:.4f}ms | per-event={res_pt['per_event_p50_ms']:.5f}ms")
        print(f" B={b:4d} | ONNX FP32     : p50={res_onnx['p50_ms']:.4f}ms, p99={res_onnx['p99_ms']:.4f}ms | per-event={res_onnx['per_event_p50_ms']:.5f}ms")
        print(f" B={b:4d} | ONNX INT8 Quant: p50={res_quant['p50_ms']:.4f}ms, p99={res_quant['p99_ms']:.4f}ms | per-event={res_quant['per_event_p50_ms']:.5f}ms")

        batch_results[str(b)] = {
            "pytorch": res_pt,
            "onnx": res_onnx,
            "onnx_quantized": res_quant
        }

    return {
        "file_sizes_kb": {
            "pytorch": round(size_pth, 2),
            "onnx": round(size_onnx, 2),
            "onnx_quantized": round(size_quant, 2)
        },
        "batch_benchmarks": batch_results
    }

if __name__ == '__main__':
    benchmarks = {}
    
    # Benchmark LSTM
    lstm_results = export_and_quantize(
        LSTMPredictor,
        'lstm_forecaster.pth',
        'lstm_forecaster.onnx',
        'lstm_forecaster_quant.onnx'
    )
    if lstm_results:
        benchmarks['lstm'] = lstm_results
        
    # Benchmark GRU
    gru_results = export_and_quantize(
        GRUPredictor,
        'gru_forecaster.pth',
        'gru_forecaster.onnx',
        'gru_forecaster_quant.onnx'
    )
    if gru_results:
        benchmarks['gru'] = gru_results
        
    with open('onnx_benchmarks.json', 'w') as f:
        json.dump(benchmarks, f, indent=2)
        
    print("\nModel optimization benchmark complete. Full metrics saved to 'onnx_benchmarks.json'")
