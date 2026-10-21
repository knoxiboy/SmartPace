#!/usr/bin/env python
"""Read-only integrity check of the downloaded datasets and their caches.
Safe to run while an experiment is running (it only reads files).

    python scripts/check_data.py --data-dir data
"""
import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import wfdb  # noqa: E402

from smartpace import config as C  # noqa: E402
from smartpace.datasets import MITDB_RECORDS  # noqa: E402


def mitdb(dd):
    d = os.path.join(dd, "mitdb")
    print("\n== MIT-BIH (data/mitdb) ==")
    missing, bad = [], []
    for r in MITDB_RECORDS:
        p = os.path.join(d, r)
        if not all(os.path.exists(p + e) for e in (".dat", ".hea", ".atr")):
            missing.append(r)
            continue
        try:
            h = wfdb.rdheader(p)
            a = wfdb.rdann(p, "atr")
            if h.sig_len != 650000 or os.path.getsize(p + ".dat") != 1950000 or len(a.sample) < 100:
                bad.append(r)
        except Exception:
            bad.append(r)
    ok = len(MITDB_RECORDS) - len(missing) - len(bad)
    print(f"complete and readable: {ok} / {len(MITDB_RECORDS)}")
    if missing:
        print("missing records:", ", ".join(missing))
    if bad:
        print("CORRUPT/TRUNCATED records (delete these files):", ", ".join(bad))
    if os.path.exists(os.path.join(d, "x.part")) or glob.glob(os.path.join(d, "*.part")):
        print("leftover .part files found (interrupted downloads)")


def nstdb(dd):
    d = os.path.join(dd, "nstdb")
    print("\n== NSTDB noise records (data/nstdb) ==")
    if not os.path.isdir(d):
        print("folder not there yet (it is downloaded after the datasets are loaded)")
        return
    for k in C.NOISE_KINDS:
        p = os.path.join(d, k)
        if not all(os.path.exists(p + e) for e in (".dat", ".hea")):
            print(f"{k}: MISSING")
            continue
        try:
            rec = wfdb.rdrecord(p)
            sig = rec.p_signal
            fine = sig.shape == (650000, 2) and np.isfinite(sig).all() and sig[:, 0].std() > 0
            print(f"{k} ({C.NOISE_NAMES[k]}): shape {sig.shape}, fs {rec.fs}, std {sig[:, 0].std():.3f} -> "
                  f"{'OK' if fine else 'SUSPICIOUS'}")
        except Exception as e:
            print(f"{k}: UNREADABLE ({type(e).__name__}) - delete bw/em/ma files and re-run")


def ptbxl(dd):
    d = os.path.join(dd, "ptb-xl")
    print("\n== PTB-XL (data/ptb-xl) ==")
    for f in ("ptbxl_database.csv", "scp_statements.csv"):
        print(f"{f}: {'present' if os.path.exists(os.path.join(d, f)) else 'MISSING'}")
    for res, folder in ((100, "records100"), (500, "records500")):
        heas = glob.glob(os.path.join(d, folder, "**", "*.hea"), recursive=True)
        if not heas:
            continue
        bad, n_ok = [], 0
        want = 1000 if res == 100 else 5000
        for h in heas:
            dat = h[:-4] + ".dat"
            try:
                rec = wfdb.rdrecord(h[:-4])
                if os.path.exists(dat) and rec.p_signal.shape == (want, 12) and np.isfinite(rec.p_signal).all():
                    n_ok += 1
                else:
                    bad.append(os.path.basename(h))
            except Exception:
                bad.append(os.path.basename(h))
        print(f"{folder}: {n_ok} readable records of {len(heas)} header files "
              f"(expected shape ({want}, 12))")
        if bad:
            print("  bad/incomplete (first 10):", bad[:10], "-> delete their .hea/.dat and re-run")
        for pa in glob.glob(os.path.join(d, folder, "**", "*.part"), recursive=True)[:5]:
            print("  leftover partial download:", pa)


def caches(dd):
    print("\n== Processed caches (data/cache) ==")
    files = sorted(glob.glob(os.path.join(dd, "cache", "*.npz")))
    if not files:
        print("no cache files yet")
    for f in files:
        z = np.load(f, allow_pickle=True)
        y, s, rec = z["y"], z["split"], z["rec"].astype(str)
        print(f"\n{os.path.basename(f)}: {len(y)} windows, {len(set(rec))} records/patients")
        for name, sid in (("train", 0), ("val", 1), ("test", 2)):
            m = s == sid
            cnt = {c: int((y[m] == i).sum()) for i, c in enumerate(C.CLASSES)}
            cnt["OTHER"] = int((y[m] < 0).sum())
            print(f"  {name:5s} n={int(m.sum()):6d} records={len(set(rec[m])):5d}  {cnt}")
        if os.path.basename(f) == "mitbih.npz":
            n = len(set(rec))
            print(f"  -> {'OK' if n == 44 else 'WARNING'}: expected 44 records "
                  f"(18 train / 4 val / 22 test), found {n}")
        miss = [c for i, c in enumerate(C.CLASSES) if (y[s == 2] == i).sum() == 0]
        if miss:
            print("  note: classes absent from the TEST split:", miss)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    a = ap.parse_args()
    mitdb(a.data_dir)
    ptbxl(a.data_dir)
    nstdb(a.data_dir)
    caches(a.data_dir)
