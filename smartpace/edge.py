"""Edge model optimisation (INT8 quantisation, magnitude pruning) and benchmarking.

Workflow: PyTorch model -> ONNX (FP32) -> static INT8 quantisation (onnxruntime, QDQ format).
Pruning: global unstructured L1 pruning + short fine-tuning, then ONNX export.

Honest caveat: unstructured pruning zeroes weights but does not shrink the dense ONNX file nor
speed up dense CPU kernels. We therefore report parameter sparsity and the gzip-compressed size
next to the plain file size; only quantisation reduces the dense model size/latency.
"""
import gzip
import json
import os
import platform
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

from .train import TrainCfg, get_device


# ----------------------------------------------------------------------------- export
def export_onnx(model, path, length=2500):
    model = model.cpu().eval()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    dummy = torch.zeros(1, 1, length)
    kw = dict(input_names=["input"], output_names=["logits"], opset_version=13,
              dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}})
    try:
        torch.onnx.export(model, dummy, path, dynamo=False, **kw)
    except TypeError:                      # older torch without the `dynamo` argument
        torch.onnx.export(model, dummy, path, **kw)
    return path


def quantize_int8(fp32_path, int8_path, calib_x, n_calib=200):
    """Static post-training INT8 quantisation (QDQ) calibrated on `calib_x` (N,1,L)."""
    from onnxruntime.quantization import (CalibrationDataReader, QuantFormat, QuantType,
                                          quantize_static)
    try:
        from onnxruntime.quantization.shape_inference import quant_pre_process
        pre = fp32_path.replace(".onnx", "_pre.onnx")
        quant_pre_process(fp32_path, pre)
        src = pre
    except Exception:
        src = fp32_path

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.it = iter([{"input": calib_x[i:i + 1].astype(np.float32)} for i in range(min(n_calib, len(calib_x)))])

        def get_next(self):
            return next(self.it, None)

    quantize_static(src, int8_path, Reader(), quant_format=QuantFormat.QDQ,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
                    per_channel=True)
    return int8_path


# ----------------------------------------------------------------------------- pruning
def prune_model(model, sparsity, finetune=None, device=None):
    """Global L1 unstructured pruning of all Conv1d/Linear weights, optional fine-tuning with the
    mask kept, then made permanent. `finetune` = (x_train, y_train, epochs)."""
    import copy
    m = copy.deepcopy(model).cpu()
    params = [(mod, "weight") for mod in m.modules() if isinstance(mod, (nn.Conv1d, nn.Linear))]
    prune.global_unstructured(params, pruning_method=prune.L1Unstructured, amount=sparsity)
    if finetune is not None:
        x, y, epochs = finetune
        device = device or get_device()
        m = m.to(device).train()
        opt = torch.optim.AdamW(m.parameters(), lr=3e-4)
        lossf = nn.CrossEntropyLoss()
        xt, yt = torch.from_numpy(x), torch.from_numpy(y).long()
        for _ in range(epochs):
            perm = torch.randperm(len(yt))
            for i in range(0, len(perm), 128):
                b = perm[i:i + 128]
                if len(b) < 2:
                    continue
                opt.zero_grad()
                lossf(m(xt[b].to(device)), yt[b].to(device)).backward()
                opt.step()
        m = m.cpu()
    for mod, name in params:
        prune.remove(mod, name)
    return m.eval()


def sparsity_stats(model):
    tot = nz = 0
    for p in model.parameters():
        tot += p.numel()
        nz += int((p != 0).sum())
    return {"params": tot, "nonzero_params": nz, "sparsity": 1 - nz / tot}


# ----------------------------------------------------------------------------- benchmark
def file_size_mb(path):
    return os.path.getsize(path) / 1e6


def gzip_size_mb(path):
    with open(path, "rb") as f:
        return len(gzip.compress(f.read(), 6)) / 1e6


def ort_session(path, threads=1):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])


def onnx_logits(path, x, bs=256):
    sess = ort_session(path, threads=1)
    out = []
    for i in range(0, len(x), bs):
        out.append(sess.run(None, {"input": x[i:i + bs].astype(np.float32)})[0])
    return np.concatenate(out)


def latency_ms(path, length=2500, n_warmup=20, n_runs=200, threads=1, blocks=5):
    """Single-window (batch 1) latency in milliseconds: mean, median, p95.
    The runs are split into `blocks`; the block with the lowest median is reported, which reduces
    the effect of other programs using the CPU during the measurement."""
    sess = ort_session(path, threads)
    x = np.random.default_rng(0).standard_normal((1, 1, length)).astype(np.float32)
    for _ in range(n_warmup):
        sess.run(None, {"input": x})
    per = max(1, n_runs // blocks)
    best = None
    for _ in range(blocks):
        t = []
        for _ in range(per):
            t0 = time.perf_counter()
            sess.run(None, {"input": x})
            t.append((time.perf_counter() - t0) * 1000)
        t = np.array(t)
        if best is None or np.median(t) < np.median(best):
            best = t
    return {"lat_mean_ms": float(best.mean()), "lat_median_ms": float(np.median(best)),
            "lat_p95_ms": float(np.percentile(best, 95))}


_MEM_SCRIPT = r"""
import sys, os, numpy as np, psutil
import onnxruntime as ort

def peak_mb():
    mi = psutil.Process(os.getpid()).memory_info()
    if hasattr(mi, "peak_wset"):                         # Windows: peak working set
        return mi.peak_wset / 1e6
    try:                                                 # Linux: high-water mark of resident memory
        return [int(l.split()[1]) for l in open("/proc/self/status") if l.startswith("VmHWM")][0] / 1024.0
    except Exception:                                    # macOS
        import resource
        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return r / 1e6 if sys.platform == "darwin" else r / 1024.0

p = psutil.Process(os.getpid())
base = p.memory_info().rss
so = ort.SessionOptions(); so.intra_op_num_threads = int(sys.argv[3]); so.inter_op_num_threads = 1
s = ort.InferenceSession(sys.argv[1], so, providers=["CPUExecutionProvider"])
x = np.zeros((1, 1, int(sys.argv[2])), dtype=np.float32)
for _ in range(50):
    s.run(None, {"input": x})
print(peak_mb(), (p.memory_info().rss - base) / 1e6)
"""


def memory_mb(path, length=2500, threads=1):
    """Peak memory (MB) of a fresh process that loads the model and runs 50 inferences, and the
    resident memory added by loading + running it (excluding the interpreter/onnxruntime baseline)."""
    r = subprocess.run([sys.executable, "-c", _MEM_SCRIPT, path, str(length), str(threads)],
                       capture_output=True, text=True)
    try:
        peak, delta = r.stdout.strip().split()
        return {"peak_rss_mb": float(peak), "model_rss_delta_mb": float(delta)}
    except Exception:
        print("  memory measurement failed:", (r.stderr or "")[-300:])
        return {"peak_rss_mb": float("nan"), "model_rss_delta_mb": float("nan")}


def hardware_info():
    info = {"platform": platform.platform(), "machine": platform.machine(),
            "python": sys.version.split()[0], "cpu_count": os.cpu_count()}
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                low = line.lower()
                val = line.split(":", 1)[1].strip() if ":" in line else ""
                if low.startswith(("model name", "hardware")) or (low.startswith("model") and "raspberry" in val.lower()):
                    info["cpu"] = val
                    break
    except Exception:
        pass
    try:
        import onnxruntime
        info["onnxruntime"] = onnxruntime.__version__
    except Exception:
        pass
    return info


def benchmark_file(path, length=2500, threads=1):
    row = {"file_mb": file_size_mb(path), "gzip_mb": gzip_size_mb(path)}
    row.update(latency_ms(path, length, threads=threads))
    row.update(memory_mb(path, length, threads))
    return row


if __name__ == "__main__":                 # python -m smartpace.edge model1.onnx model2.onnx ...
    print(json.dumps(hardware_info(), indent=2))
    for p in sys.argv[1:]:
        print(p, json.dumps(benchmark_file(p), indent=2))
