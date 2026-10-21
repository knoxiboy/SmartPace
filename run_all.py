#!/usr/bin/env python
"""Run the whole SmartPace study.

Real data (needs internet, ~2 GB):
    python run_all.py --data-dir data --out runs/full
Quick offline smoke test on SYNTHETIC ECG (numbers are meaningless, checks the pipeline):
    python run_all.py --synthetic --quick --out runs/smoke
"""
import argparse
import json
import os
import time

from smartpace.experiments import ALL, Ctx
from smartpace.report import make_report


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default="runs/full")
    ap.add_argument("--synthetic", action="store_true", help="use synthetic ECG (smoke test only)")
    ap.add_argument("--quick", action="store_true", help="tiny settings for a fast check")
    ap.add_argument("--steps", default="1,2,3,4,5,6", help="comma-separated experiments to run")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--mc-T", type=int, default=20, help="MC-dropout passes")
    ap.add_argument("--reject-coverage", type=float, default=0.95,
                    help="share of CLEAN validation windows that must be accepted by the uncertainty gate")
    ap.add_argument("--ptbxl-max", type=int, default=None, help="subsample PTB-XL to N records (default: all)")
    ap.add_argument("--ptbxl-res", type=int, default=500, choices=[100, 500])
    ap.add_argument("--with-transformer", action="store_true", help="also train ablation B3 (CNN-Transformer)")
    ap.add_argument("--expl-per-class", type=int, default=20)
    ap.add_argument("--ig-steps", type=int, default=32)
    ap.add_argument("--noise-n", type=int, default=1500, help="test windows used in the noise study")
    ap.add_argument("--prune-epochs", type=int, default=3)
    ap.add_argument("--prune-n", type=int, default=6000)
    ap.add_argument("--sqi-q", type=float, default=1.0, help="percentile of clean validation windows used for SQI thresholds")
    ap.add_argument("--threads", type=int, default=1, help="onnxruntime threads for latency")
    ap.add_argument("--main-train", default="AB", choices=["A", "B", "AB"],
                    help="training set of the main model used in experiments 3-6 (AB = MIT-BIH + PTB-XL)")
    ap.add_argument("--sqi-n", type=int, default=2000, help="training windows (many patients) used to fit the SQI gate")
    ap.add_argument("--burst-snr", type=float, default=0.0, help="SNR (dB) of the noise bursts in experiment 6")
    a = ap.parse_args()
    if a.quick:
        a.seeds, a.epochs, a.mc_T, a.expl_per_class, a.ig_steps = 2, 3, 5, 4, 8
        a.noise_n, a.prune_epochs, a.prune_n, a.patience = 150, 1, 300, 3
    return a


if __name__ == "__main__":
    a = parse()
    t0 = time.time()
    ctx = Ctx(a)
    for s in a.steps.split(","):
        t1 = time.time()
        ALL[s.strip()](ctx)
        print(f"-- experiment {s} done in {time.time() - t1:.0f}s", flush=True)
    ctx.info.update({"args": vars(a), "total_seconds": round(time.time() - t0)})
    with open(os.path.join(a.out, "results", "run_info.json"), "w") as f:
        json.dump(ctx.info, f, indent=2, default=str)
    make_report(a.out)
    print(f"\nAll done in {time.time() - t0:.0f}s. See {a.out}/REPORT.md")
