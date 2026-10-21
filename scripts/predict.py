#!/usr/bin/env python
"""Run the trained SmartPace ensemble on NEW ECG data.

Outputs, per 10-s window: predicted rhythm (or "UNRELIABLE"), calibrated probabilities, uncertainty,
signal-quality flags. Research prototype - NOT a medical device.

Inputs (choose one)
  --wfdb  PATH      a WFDB record without extension, e.g. data/mitdb/100 (any sampling rate)
  --csv   FILE      one column (or a column chosen with --col) of single-lead samples; needs --fs
  --npy   FILE      1-D array of samples (one long recording); needs --fs
  --npz   FILE      pre-windowed data: X (N, L) and optional y (N,) with the unified class ids
                    0=NSR 1=SBRAD 2=STACH 3=AF 4=PVC (-1 = ignore); needs --fs (sampling rate of X)

Examples
  python scripts/predict.py --run runs/first --wfdb data/mitdb/100
  python scripts/predict.py --run runs/first --csv my_ecg.csv --fs 500 --explain 3
  python scripts/predict.py --run runs/first --npz other_dataset.npz --fs 360     (also reports F1 if y given)

Use a lead similar to lead II (MLII). Units do not matter (each window is z-scored), polarity and lead do.
"""
import argparse
import glob
import json
import os
import sys
from argparse import Namespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from smartpace import config as C  # noqa: E402
from smartpace import evaluation as E  # noqa: E402
from smartpace.datasets import resample_to_fs  # noqa: E402
from smartpace.preprocess import SQIGate, prepare_input, sqi_features  # noqa: E402
from smartpace.train import ensemble_logits, load_model  # noqa: E402


def load_calibration(run, data_dir):
    """T, uncertainty threshold and SQI thresholds. Computed once from the validation data
    (needs the cached datasets in data_dir) and stored in <run>/calibration.json."""
    path = os.path.join(run, "calibration.json")
    if os.path.exists(path):
        return json.load(open(path, encoding="utf-8"))
    info = json.load(open(os.path.join(run, "results", "run_info.json"), encoding="utf-8"))
    args = Namespace(**info["args"])
    args.out, args.data_dir = run, data_dir
    from smartpace.experiments import Ctx
    print("First use: computing calibration from the validation data (a minute or two) ...", flush=True)
    ctx = Ctx(args)
    cal = {"main_train": ctx.main, "T": ctx.temperature(), "tau": ctx.tau_unc(),
           "sqi_thr": ctx.gate().thr, "reject_coverage": args.reject_coverage}
    json.dump(cal, open(path, "w", encoding="utf-8"), indent=2)
    return cal


def read_input(a):
    if a.wfdb:
        import wfdb
        from smartpace.datasets import _pick_lead
        rec = wfdb.rdrecord(a.wfdb)
        ch = a.lead if a.lead is not None else _pick_lead(rec)
        print(f"WFDB record {a.wfdb}: {rec.sig_name}, fs={rec.fs}, using channel {ch} ({rec.sig_name[ch]})")
        return np.nan_to_num(rec.p_signal[:, ch].astype(np.float64)), int(rec.fs), None
    if a.csv:
        import pandas as pd
        df = pd.read_csv(a.csv)
        col = df.columns[a.col] if isinstance(a.col, int) else a.col
        return df[col].to_numpy(dtype=np.float64), a.fs, None
    if a.npy:
        return np.load(a.npy).astype(np.float64).ravel(), a.fs, None
    z = np.load(a.npz)
    return None, a.fs, (z["X"].astype(np.float64), z["y"] if "y" in z.files else None)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="runs/first")
    ap.add_argument("--data-dir", default="data")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--wfdb"); g.add_argument("--csv"); g.add_argument("--npy"); g.add_argument("--npz")
    ap.add_argument("--fs", type=int, help="sampling rate of the input (Hz)")
    ap.add_argument("--lead", type=int, default=None, help="WFDB channel index (default: MLII / II)")
    ap.add_argument("--col", default=0, help="CSV column name or index")
    ap.add_argument("--out", default=None, help="output CSV (default: <run>/predictions.csv)")
    ap.add_argument("--explain", type=int, default=0, help="save Integrated Gradients plots for the first N reliable windows")
    a = ap.parse_args()
    if (a.csv or a.npy or a.npz) and not a.fs:
        ap.error("--fs is required for csv/npy/npz input")
    if isinstance(a.col, str) and a.col.isdigit():
        a.col = int(a.col)

    cal = load_calibration(a.run, a.data_dir)
    models = [load_model(f)[0] for f in sorted(glob.glob(os.path.join(a.run, "models", f"resnet1d_{cal['main_train']}_s*.pt")))]
    if not models:
        sys.exit("No trained models found in " + os.path.join(a.run, "models"))
    gate = SQIGate(use_bw=False)
    gate.thr = cal["sqi_thr"]

    sig, fs, pre = read_input(a)
    y = None
    if pre is None:
        x250 = resample_to_fs(sig, fs).astype(np.float32)
        n = len(x250) // C.WIN
        if n == 0:
            sys.exit(f"Recording shorter than {C.WIN_S} s.")
        X = x250[:n * C.WIN].reshape(n, C.WIN)
        t0 = np.arange(n) * float(C.WIN_S)
    else:
        X, y = pre
        if X.shape[1] != fs * C.WIN_S:
            print(f"note: windows have {X.shape[1]} samples; at {fs} Hz that is {X.shape[1] / fs:.1f} s (expected {C.WIN_S} s)")
        X = np.stack([resample_to_fs(w, fs) for w in X]).astype(np.float32)
        X = X[:, :C.WIN] if X.shape[1] >= C.WIN else np.pad(X, ((0, 0), (0, C.WIN - X.shape[1])), mode="edge")
        t0 = np.arange(len(X)) * float(C.WIN_S)

    x = prepare_input(X, True)
    logits = ensemble_logits(models, x).mean(0)
    p = E.apply_temperature(logits, cal["T"])
    pred = p.argmax(1)
    ent = E.entropy(p)
    sqi = sqi_features(X)
    q_bad = gate.flag(sqi)
    u_bad = ent > cal["tau"]
    unreliable = q_bad | u_bad

    import pandas as pd
    df = pd.DataFrame({"window": np.arange(len(X)), "start_s": t0,
                       "prediction": np.where(unreliable, "UNRELIABLE", [C.CLASSES[i] for i in pred]),
                       "top_class": [C.CLASSES[i] for i in pred], "confidence": p.max(1).round(3),
                       "entropy": ent.round(3), "uncertain": u_bad, "poor_signal": q_bad})
    for i, c in enumerate(C.CLASSES):
        df["p_" + c] = p[:, i].round(3)
    out = a.out or os.path.join(a.run, "predictions.csv")
    df.to_csv(out, index=False)
    print(f"\n{len(df)} windows ({len(df) * C.WIN_S / 60:.1f} min) -> {out}")
    print("model: ensemble of", len(models), "ResNet1D | uncertainty threshold %.3f | T=%.2f" % (cal["tau"], cal["T"]))
    print(df["prediction"].value_counts().to_string())
    print(f"unreliable: {unreliable.mean():.1%} (uncertain {u_bad.mean():.1%}, poor signal {q_bad.mean():.1%})")

    if y is not None:
        m = y >= 0
        print("\n--- evaluation against the provided labels (windows with y>=0) ---")
        print(f"macro-F1 (all):      {E.macro_f1(y[m], pred[m]):.3f}   accuracy {E.accuracy(y[m], pred[m]):.3f}")
        k = m & ~unreliable
        if k.any():
            print(f"macro-F1 (accepted): {E.macro_f1(y[k], pred[k]):.3f}   accuracy {E.accuracy(y[k], pred[k]):.3f}   "
                  f"({k.sum()}/{m.sum()} windows accepted)")
        print("per-class F1:", {c: round(v, 3) for c, v in E.per_class_f1(y[m], pred[m]).items()})
        print("ECE: %.3f" % E.ece(p[m], y[m]))

    if a.explain:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from smartpace.explain import integrated_gradients, smooth_abs
        idx = np.where(~unreliable)[0][:a.explain]
        if len(idx):
            attr = smooth_abs(integrated_gradients(models[0], x[idx], pred[idx], steps=32))
            tt = np.arange(C.WIN) / C.FS
            for j, i in enumerate(idx):
                fig, ax = plt.subplots(figsize=(10, 2.6))
                ax.plot(tt, x[i, 0], "k", lw=.8)
                ax2 = ax.twinx()
                ax2.fill_between(tt, 0, attr[j] / (attr[j].max() + 1e-12), color="r", alpha=.3)
                ax2.set_yticks([])
                ax.set_title(f"window {i} ({t0[i]:.0f}s): {C.CLASSES[pred[i]]}, confidence {p[i].max():.2f}", fontsize=9)
                ax.set_xlabel("time (s)")
                fig.tight_layout()
                f = os.path.join(a.run, f"explain_window_{i}.png")
                fig.savefig(f, dpi=120)
                plt.close(fig)
                print("saved", f)


if __name__ == "__main__":
    main()
