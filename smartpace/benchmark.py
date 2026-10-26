"""SmartPace Live — High-Precision FP32 vs ONNX INT8 Benchmark Service.

Provides reproducible, wall-clock timing comparing native PyTorch FP32 inference
against optimized ONNX Runtime INT8 quantized execution on identical preprocessed ECG windows.
"""
import os
import time
from typing import Any, Dict, List, Optional

import numpy as np
import onnxruntime as ort
import torch
from scipy.special import softmax

from . import config as C
from . import train


def run_fp32_vs_int8_benchmark(
    n_windows: int = 50,
    repetitions: int = 3,
    warmup_runs: int = 10,
    model_path: Optional[str] = None,
    onnx_path: Optional[str] = None,
    data_path: Optional[str] = None,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """Executes a live, paired latency and accuracy comparison between PyTorch FP32 and ONNX INT8.

    Args:
        n_windows: Number of test windows to evaluate (10 to 200).
        repetitions: Number of repeated timed passes per window.
        warmup_runs: Inferences discarded prior to measurement to eliminate cold-start transients.
        model_path: Checkpoint path for PyTorch FP32 model.
        onnx_path: Filepath to quantized INT8 ONNX graph.
        data_path: Path to authentic preprocessed test dataset (.npz).
        device: Device string ('cpu' or 'cuda').

    Returns:
        Structured dictionary containing measured latency percentiles, speedup,
        prediction agreement, and probability drift.
    """
    pt_path = model_path or os.path.join("runs", "first", "models", "resnet1d_AB_s0.pt")
    int8_path = onnx_path or os.path.join("runs", "first", "edge", "resnet1d_int8.onnx")
    ds_path = data_path or os.path.join("runs", "first", "edge", "test_subset.npz")
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Verify existence of required artifacts
    if not os.path.exists(pt_path):
        raise FileNotFoundError(f"PyTorch model checkpoint not found: {pt_path}")
    if not os.path.exists(int8_path):
        raise FileNotFoundError(f"ONNX INT8 model artifact not found: {int8_path}")

    # 2. Load model artifacts outside timed region
    pt_size_kb = os.path.getsize(pt_path) / 1024.0
    int8_size_kb = os.path.getsize(int8_path) / 1024.0

    model, meta = train.load_model(pt_path)
    model = model.to(dev).eval()
    for param in model.parameters():
        param.requires_grad = False

    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 4
    sess = ort.InferenceSession(int8_path, sess_options=sess_options, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    onnx_providers = sess.get_providers()

    # 3. Load or prepare representative test windows
    if os.path.exists(ds_path):
        npz = np.load(ds_path)
        x_data = npz["x"]  # (N, 1, 2500) float32
        n_total = len(x_data)
        n_use = min(n_windows, n_total)
        test_windows = x_data[:n_use]
    else:
        # Fallback to authentic records from mitbih.npz with prepare_input
        mit_path = os.path.join("data", "cache", "mitbih.npz")
        if os.path.exists(mit_path):
            from . import preprocess
            npz = np.load(mit_path)
            raw_X = npz["X"]
            n_use = min(n_windows, len(raw_X))
            test_windows = preprocess.prepare_input(raw_X[:n_use], filt=True)
        else:
            raise FileNotFoundError("No benchmark dataset found in test_subset.npz or mitbih.npz")

    # 4. Warm-up Phase (excluded from measurements)
    dummy_pt = torch.from_numpy(test_windows[0:1]).to(dev)
    dummy_np = test_windows[0:1]
    for _ in range(warmup_runs):
        with torch.no_grad():
            _ = model(dummy_pt)
        _ = sess.run(None, {input_name: dummy_np})

    # Synchronize if GPU
    if dev == "cuda":
        torch.cuda.synchronize()

    # 5. Timed Inference Loop
    pt_latencies_ms: List[float] = []
    int8_latencies_ms: List[float] = []
    pt_probs_list: List[np.ndarray] = []
    int8_probs_list: List[np.ndarray] = []

    for w_idx in range(len(test_windows)):
        w = test_windows[w_idx:w_idx + 1]  # (1, 1, 2500) float32
        wt = torch.from_numpy(w).to(dev)

        # PyTorch FP32 Timed Run
        for _ in range(repetitions):
            t0 = time.perf_counter_ns()
            with torch.no_grad():
                out_pt = model(wt)
                if dev == "cuda":
                    torch.cuda.synchronize()
            dt_ms = (time.perf_counter_ns() - t0) / 1e6
            pt_latencies_ms.append(dt_ms)

        # Compute probabilities for accuracy & drift metrics
        with torch.no_grad():
            logits_pt = model(wt).cpu().numpy()
        probs_pt = softmax(logits_pt, axis=1)[0]
        pt_probs_list.append(probs_pt)

        # ONNX INT8 Timed Run
        for _ in range(repetitions):
            t0 = time.perf_counter_ns()
            out_onnx = sess.run(None, {input_name: w})[0]
            dt_ms = (time.perf_counter_ns() - t0) / 1e6
            int8_latencies_ms.append(dt_ms)

        probs_onnx = softmax(out_onnx, axis=1)[0]
        int8_probs_list.append(probs_onnx)

    # 6. Aggregate Statistical Metrics
    pt_arr = np.array(pt_latencies_ms, dtype=np.float64)
    int8_arr = np.array(int8_latencies_ms, dtype=np.float64)
    pt_probs = np.array(pt_probs_list, dtype=np.float64)      # (N, C)
    int8_probs = np.array(int8_probs_list, dtype=np.float64)  # (N, C)

    pt_preds = np.argmax(pt_probs, axis=1)
    int8_preds = np.argmax(int8_probs, axis=1)

    matching_predictions = int(np.sum(pt_preds == int8_preds))
    agreement_rate = float(matching_predictions / len(pt_preds))
    mean_prob_drift = float(np.mean(np.abs(pt_probs - int8_probs)))
    max_prob_drift = float(np.max(np.abs(pt_probs - int8_probs)))

    fp32_median = float(np.median(pt_arr))
    fp32_mean = float(np.mean(pt_arr))
    fp32_p95 = float(np.percentile(pt_arr, 95))
    fp32_p99 = float(np.percentile(pt_arr, 99))

    int8_median = float(np.median(int8_arr))
    int8_mean = float(np.mean(int8_arr))
    int8_p95 = float(np.percentile(int8_arr, 95))
    int8_p99 = float(np.percentile(int8_arr, 99))

    speedup_median = float(fp32_median / max(int8_median, 1e-4))
    speedup_mean = float(fp32_mean / max(int8_mean, 1e-4))
    compression_ratio = float(pt_size_kb / max(int8_size_kb, 1e-4))

    return {
        "status": "completed",
        "benchmark_type": "model_only",
        "test_windows_evaluated": len(test_windows),
        "repetitions_per_window": repetitions,
        "total_timed_inferences": len(pt_arr),
        "warmup_runs": warmup_runs,
        "timing_unit": "milliseconds",
        "pytorch_fp32": {
            "model_file": os.path.basename(pt_path),
            "size_kb": round(pt_size_kb, 1),
            "device": dev,
            "mean_latency_ms": round(fp32_mean, 3),
            "median_latency_ms": round(fp32_median, 3),
            "p95_latency_ms": round(fp32_p95, 3),
            "p99_latency_ms": round(fp32_p99, 3),
            "min_latency_ms": round(float(np.min(pt_arr)), 3),
            "max_latency_ms": round(float(np.max(pt_arr)), 3),
        },
        "onnx_int8": {
            "model_file": os.path.basename(int8_path),
            "size_kb": round(int8_size_kb, 1),
            "execution_provider": onnx_providers[0] if onnx_providers else "CPUExecutionProvider",
            "all_providers": onnx_providers,
            "mean_latency_ms": round(int8_mean, 3),
            "median_latency_ms": round(int8_median, 3),
            "p95_latency_ms": round(int8_p95, 3),
            "p99_latency_ms": round(int8_p99, 3),
            "min_latency_ms": round(float(np.min(int8_arr)), 3),
            "max_latency_ms": round(float(np.max(int8_arr)), 3),
        },
        "comparison": {
            "speedup_ratio_median": round(speedup_median, 2),
            "speedup_ratio_mean": round(speedup_mean, 2),
            "compression_ratio": round(compression_ratio, 2),
            "prediction_agreement_rate": round(agreement_rate, 4),
            "prediction_agreement_pct": round(agreement_rate * 100.0, 2),
            "matching_windows": matching_predictions,
            "mean_probability_drift": round(mean_prob_drift, 5),
            "max_probability_drift": round(max_prob_drift, 5),
        },
        "scientific_disclaimer": "Research benchmark on verified model checkpoints. Not a clinical device diagnostic measurement.",
    }
