import os
import json
import time
import torch
import torch.nn as nn
import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType

# --- 1. Model Definitions (Matching the project architecture) ---
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


def export_and_quantize(model_class, pth_path, onnx_path, quant_path):
    print(f"\n--- Processing model: {pth_path} ---")
    if not os.path.exists(pth_path):
        print(f"Skipping {pth_path} (File not found).")
        return None
    
    # Load PyTorch Model
    model = model_class()
    model.load_state_dict(torch.load(pth_path, map_location='cpu', weights_only=True))
    model.eval()
    
    # 1. Export to ONNX
    # Input shape: (batch_size, sequence_length, features) -> (1, 5, 6)
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
        dynamo=False
    )
    print(f"Exported to ONNX: {onnx_path}")
    
    # 2. Dynamic INT8 Quantization
    quantize_dynamic(
        onnx_path,
        quant_path,
        weight_type=QuantType.QInt8,
        extra_options={'DisableShapeInference': True}
    )
    print(f"Quantized ONNX model generated: {quant_path}")
    
    # 3. Benchmarking
    # Retrieve file sizes (in KB)
    size_pth = os.path.getsize(pth_path) / 1024.0
    size_onnx = os.path.getsize(onnx_path) / 1024.0
    size_quant = os.path.getsize(quant_path) / 1024.0
    
    print(f"Sizes -> PyTorch: {size_pth:.1f}KB, ONNX: {size_onnx:.1f}KB, Quant: {size_quant:.1f}KB")
    
    # Measure execution latency
    num_runs = 1000
    dummy_numpy = np.random.randn(1, 5, 6).astype(np.float32)
    
    # PyTorch latency
    print("Benchmarking PyTorch FP32...")
    with torch.no_grad():
        for _ in range(100):  # Warmup
            _ = model(dummy_input)
        
        start_t = time.perf_counter()
        for _ in range(num_runs):
            _ = model(dummy_input)
        pytorch_latency = ((time.perf_counter() - start_t) / num_runs) * 1000.0 # ms
        
    # ONNX FP32 latency
    print("Benchmarking ONNX FP32...")
    ort_session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    for _ in range(100):  # Warmup
        _ = ort_session.run(['output'], {'input': dummy_numpy})
    
    start_t = time.perf_counter()
    for _ in range(num_runs):
        _ = ort_session.run(['output'], {'input': dummy_numpy})
    onnx_latency = ((time.perf_counter() - start_t) / num_runs) * 1000.0 # ms
    
    # ONNX Quantized latency
    print("Benchmarking ONNX INT8 Quantized...")
    ort_quant_session = ort.InferenceSession(quant_path, providers=['CPUExecutionProvider'])
    for _ in range(100):  # Warmup
        _ = ort_quant_session.run(['output'], {'input': dummy_numpy})
        
    start_t = time.perf_counter()
    for _ in range(num_runs):
        _ = ort_quant_session.run(['output'], {'input': dummy_numpy})
    quant_latency = ((time.perf_counter() - start_t) / num_runs) * 1000.0 # ms
    
    print(f"Latencies -> PyTorch: {pytorch_latency:.3f}ms, ONNX: {onnx_latency:.3f}ms, Quant: {quant_latency:.3f}ms")
    
    return {
        "pytorch": {"size_kb": round(size_pth, 2), "latency_ms": round(pytorch_latency, 3)},
        "onnx": {"size_kb": round(size_onnx, 2), "latency_ms": round(onnx_latency, 3)},
        "onnx_quantized": {"size_kb": round(size_quant, 2), "latency_ms": round(quant_latency, 3)}
    }

if __name__ == '__main__':
    benchmarks = {}
    
    # Process LSTM
    lstm_results = export_and_quantize(
        LSTMPredictor,
        'lstm_forecaster.pth',
        'lstm_forecaster.onnx',
        'lstm_forecaster_quant.onnx'
    )
    if lstm_results:
        benchmarks['lstm'] = lstm_results
        
    # Process GRU
    gru_results = export_and_quantize(
        GRUPredictor,
        'gru_forecaster.pth',
        'gru_forecaster.onnx',
        'gru_forecaster_quant.onnx'
    )
    if gru_results:
        benchmarks['gru'] = gru_results
        
    # Export benchmarks to JSON
    with open('onnx_benchmarks.json', 'w') as f:
        json.dump(benchmarks, f, indent=2)
        
    print("\nModel optimization complete. Results written to 'onnx_benchmarks.json'")
