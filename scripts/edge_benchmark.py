#!/usr/bin/env python
"""Standalone on-device benchmark (copy this file + the .onnx files + test_subset.npz to the device).

Needs only:  pip install numpy onnxruntime psutil
Usage:       python edge_benchmark.py edge/*.onnx --test edge/test_subset.npz --threads 1
Reports per model: macro-F1 on the 500-window test subset, file size, gzip size, batch-1 latency
(mean / median / p95 over 200 runs after 20 warm-up runs) and peak process memory (VmHWM).
"""
import argparse, gzip, json, os, platform, subprocess, sys, time
import numpy as np

MEM = r'''
import sys, os, numpy as np, psutil, onnxruntime as ort
def peak_mb():
    mi = psutil.Process(os.getpid()).memory_info()
    if hasattr(mi, "peak_wset"):
        return mi.peak_wset / 1e6
    try:
        return [int(l.split()[1]) for l in open("/proc/self/status") if l.startswith("VmHWM")][0] / 1024.0
    except Exception:
        import resource
        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return r / 1e6 if sys.platform == "darwin" else r / 1024.0
so = ort.SessionOptions(); so.intra_op_num_threads = int(sys.argv[3]); so.inter_op_num_threads = 1
s = ort.InferenceSession(sys.argv[1], so, providers=["CPUExecutionProvider"])
x = np.zeros((1, 1, int(sys.argv[2])), dtype=np.float32)
for _ in range(50): s.run(None, {"input": x})
print(peak_mb())
'''


def hw():
    info = {"platform": platform.platform(), "machine": platform.machine(), "python": sys.version.split()[0],
            "cpu_count": os.cpu_count()}
    try:
        for line in open("/proc/cpuinfo"):
            low, val = line.lower(), (line.split(":", 1)[1].strip() if ":" in line else "")
            if low.startswith(("model name", "hardware")) or (low.startswith("model") and "raspberry" in val.lower()):
                info["cpu"] = val; break
    except Exception:
        pass
    import onnxruntime; info["onnxruntime"] = onnxruntime.__version__
    return info


def macro_f1(y, p):
    f = []
    for c in np.unique(y):
        tp = ((p == c) & (y == c)).sum(); fp = ((p == c) & (y != c)).sum(); fn = ((p != c) & (y == c)).sum()
        f.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    return float(np.mean(f))


def session(path, threads):
    import onnxruntime as ort
    so = ort.SessionOptions(); so.intra_op_num_threads = threads; so.inter_op_num_threads = 1
    return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+"); ap.add_argument("--test", default=None)
    ap.add_argument("--threads", type=int, default=1); ap.add_argument("--runs", type=int, default=200)
    a = ap.parse_args()
    print(json.dumps(hw(), indent=2))
    test = np.load(a.test) if a.test else None
    rows = []
    for path in a.models:
        sess = session(path, a.threads)
        x1 = np.random.default_rng(0).standard_normal((1, 1, 2500)).astype(np.float32)
        for _ in range(20): sess.run(None, {"input": x1})
        t = []
        for _ in range(a.runs):
            t0 = time.perf_counter(); sess.run(None, {"input": x1}); t.append((time.perf_counter() - t0) * 1000)
        t = np.array(t)
        row = {"model": os.path.basename(path), "file_mb": round(os.path.getsize(path) / 1e6, 3),
               "gzip_mb": round(len(gzip.compress(open(path, "rb").read(), 6)) / 1e6, 3),
               "lat_mean_ms": round(float(t.mean()), 3), "lat_median_ms": round(float(np.median(t)), 3),
               "lat_p95_ms": round(float(np.percentile(t, 95)), 3)}
        if test is not None:
            lg = np.concatenate([sess.run(None, {"input": test["x"][i:i + 64]})[0] for i in range(0, len(test["x"]), 64)])
            row["macro_f1_subset"] = round(macro_f1(test["y"], lg.argmax(1)), 4)
        r = subprocess.run([sys.executable, "-c", MEM, path, "2500", str(a.threads)], capture_output=True, text=True)
        row["peak_rss_mb"] = round(float(r.stdout.strip()), 1) if r.stdout.strip() else None
        rows.append(row); print(json.dumps(row))
    out = os.path.join(os.path.dirname(os.path.abspath(a.models[0])), "device_benchmark.json")
    json.dump({"hardware": hw(), "threads": a.threads, "results": rows}, open(out, "w"), indent=2)
    print("saved", out)


if __name__ == "__main__":
    main()
