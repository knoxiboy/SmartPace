"""Dataset builders with the unified 5-class label protocol (see config.py).

Every builder returns a dict with arrays
    X (N, 2500) float32 raw (resampled, unfiltered) lead-II windows at 250 Hz
    y (N,)      unified class index, or -1 (OTHER) for windows outside the five classes
    rec (N,)    record / patient id (used for patient-wise splits)
    t0 (N,)     window start time in seconds within the record
    split (N,)  0 = train, 1 = val, 2 = test
and, for MIT-BIH / synthetic data, a `replay` dict (test records at a dense stride) used by the
decision-support simulator.
"""
import ast
import os
import time
from concurrent.futures import ThreadPoolExecutor
from math import gcd

import numpy as np
from scipy import signal as sps

from . import config as C
from . import synthetic


# ============================================================================= helpers
def resample_to_fs(x, fs_in, fs_out=C.FS):
    if int(fs_in) == fs_out:
        return x
    g = gcd(int(fs_in), fs_out)
    return sps.resample_poly(x, fs_out // g, int(fs_in) // g, axis=0)


def _pick_lead(rec):
    names = [n.upper() for n in rec.sig_name]
    for want in ("MLII", "II"):
        if want in names:
            return names.index(want)
    return 0


# ============================================================================= MIT-BIH
MITDB_RECORDS = ["100", "101", "102", "103", "104", "105", "106", "107", "108", "109", "111", "112",
                 "113", "114", "115", "116", "117", "118", "119", "121", "122", "123", "124", "200",
                 "201", "202", "203", "205", "207", "208", "209", "210", "212", "213", "214", "215",
                 "217", "219", "220", "221", "222", "223", "228", "230", "231", "232", "233", "234"]


def dl_with_retry(db, dest, fname, tries=8):
    """Download https://physionet.org/files/<db>/<fname> into dest/<fname>, retrying with back-off.
    `db` includes the version, e.g. 'mitdb/1.0.0'. Partial downloads are never kept."""
    import requests
    path = os.path.join(dest, fname)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    url = f"https://physionet.org/files/{db}/{fname}"
    for k in range(tries):
        try:
            r = requests.get(url, stream=True, timeout=60)
            if r.status_code == 404:
                raise FileNotFoundError(f"404 Not Found: {url}")
            r.raise_for_status()
            with open(path + ".part", "wb") as f:
                for chunk in r.iter_content(1 << 16):
                    f.write(chunk)
            os.replace(path + ".part", path)
            return
        except FileNotFoundError:
            raise
        except Exception as e:
            if os.path.exists(path + ".part"):
                os.remove(path + ".part")
            print(f"  retry {k + 1}/{tries} for {fname}: {type(e).__name__}", flush=True)
            time.sleep(min(30, 2 ** k))
    raise RuntimeError(f"Could not download {url}. Check your connection or download it manually.")


def download_mitbih(data_dir):
    d = os.path.join(data_dir, "mitdb")
    os.makedirs(d, exist_ok=True)
    for r in MITDB_RECORDS:
        for ext in ("dat", "hea", "atr"):
            dl_with_retry("mitdb/1.0.0", d, f"{r}.{ext}")
    return d


def _rhythm_intervals(ann, n_samples):
    """List of (start, end, rhythm_label) in original sample indices."""
    marks = [(s, a) for s, a in zip(ann.sample, ann.aux_note) if a and a.strip("\x00").startswith("(")]
    marks = [(int(s), a.strip("\x00").strip()) for s, a in marks]
    if not marks or marks[0][0] > 0:
        marks.insert(0, (0, "(N"))
    out = []
    for i, (s, a) in enumerate(marks):
        e = marks[i + 1][0] if i + 1 < len(marks) else n_samples
        out.append((s, e, a))
    return out


def label_mitbih_window(s, e, intervals, beat_samples, beat_symbols, fs=360):
    """Unified label of the window [s, e) (original sample indices). See config.py."""
    wl = e - s
    af = oth = 0
    for a, b, r in intervals:
        ov = max(0, min(e, b) - max(s, a))
        if ov == 0:
            continue
        if r in C.AF_RHYTHMS:
            af += ov
        elif r not in C.NORMAL_RHYTHMS:
            oth += ov
    if af >= 0.5 * wl:
        return C.AF
    if oth >= 0.5 * wl:
        return C.OTHER
    m = (beat_samples >= s) & (beat_samples < e)
    syms = beat_symbols[m]
    if any(x in C.PACED_SYMBOLS for x in syms):
        return C.OTHER
    if any(x in C.PVC_SYMBOLS for x in syms):
        return C.PVC
    bs = beat_samples[m]
    if len(bs) < 3:
        return C.OTHER
    hr = 60.0 / (np.median(np.diff(bs)) / fs)
    if hr < C.BRADY_BPM:
        return C.SBRAD
    if hr > C.TACHY_BPM:
        return C.STACH
    return C.NSR


def windows_from_mitbih_record(path, record, stride_s):
    """All windows of one record -> (X, y, t0)."""
    import wfdb
    rec = wfdb.rdrecord(os.path.join(path, record))
    ann = wfdb.rdann(os.path.join(path, record), "atr")
    ch = _pick_lead(rec)
    sig = rec.p_signal[:, ch].astype(np.float64)
    sig = np.nan_to_num(sig)
    fs = int(rec.fs)
    n = len(sig)
    x250 = resample_to_fs(sig, fs).astype(np.float32)
    intervals = _rhythm_intervals(ann, n)
    keep = np.array([s in C.BEAT_SYMBOLS for s in ann.symbol])
    bsamp = np.asarray(ann.sample)[keep]
    bsym = np.array(ann.symbol)[keep]
    wl, st = C.WIN_S * fs, int(stride_s * fs)
    X, y, t0 = [], [], []
    for s in range(0, n - wl + 1, st):
        s250 = int(round(s * C.FS / fs))
        if s250 + C.WIN > len(x250):
            break
        X.append(x250[s250:s250 + C.WIN])
        y.append(label_mitbih_window(s, s + wl, intervals, bsamp, bsym, fs))
        t0.append(s / fs)
    return np.stack(X), np.array(y), np.array(t0)


def build_mitbih(data_dir, replay_stride_s=2.5):
    path = download_mitbih(data_dir)
    parts = []
    spec = ([(r, 0, 5.0) for r in C.MITBIH_TRAIN] + [(r, 1, 10.0) for r in C.MITBIH_VAL] +
            [(r, 2, 10.0) for r in C.MITBIH_DS2])
    for r, split, stride in spec:
        if r in C.MITBIH_PACED or not os.path.exists(os.path.join(path, r + ".dat")):
            continue
        X, y, t0 = windows_from_mitbih_record(path, r, stride)
        parts.append((X, y, np.full(len(y), r), t0, np.full(len(y), split)))
    out = {k: np.concatenate([p[i] for p in parts]) for i, k in
           enumerate(["X", "y", "rec", "t0", "split"])}
    rp = []
    for r in C.MITBIH_DS2:
        if r in C.MITBIH_PACED or not os.path.exists(os.path.join(path, r + ".dat")):
            continue
        X, y, t0 = windows_from_mitbih_record(path, r, replay_stride_s)
        rp.append((X, y, np.full(len(y), r), t0))
    out["replay"] = {"X": np.concatenate([p[0] for p in rp]), "y": np.concatenate([p[1] for p in rp]),
                     "rec": np.concatenate([p[2] for p in rp]),
                     "t0": np.concatenate([p[3] for p in rp]), "stride_s": replay_stride_s}
    return out


# ============================================================================= PTB-XL
def label_ptbxl(scp_codes):
    """Unified label from a PTB-XL `scp_codes` dict. Priority: AF > PVC > SBRAD/STACH > NSR."""
    codes = set(scp_codes)
    if codes & {"AFIB", "AFLT"}:
        return C.AF
    if "PVC" in codes:
        return C.PVC
    if "SBRAD" in codes:
        return C.SBRAD
    if "STACH" in codes:
        return C.STACH
    if "SR" in codes and "NORM" in codes:
        return C.NSR
    return C.OTHER


def download_ptbxl_meta(data_dir):
    d = os.path.join(data_dir, "ptb-xl")
    os.makedirs(d, exist_ok=True)
    for f in ("ptbxl_database.csv", "scp_statements.csv"):
        dl_with_retry("ptb-xl/" + C.PTBXL_VERSION, d, f)
    return d


def _dl_record(args):
    d, rel = args
    for ext in (".dat", ".hea"):
        dl_with_retry("ptb-xl/" + C.PTBXL_VERSION, d, rel + ext)


def build_ptbxl(data_dir, res=500, max_records=None, seed=0, workers=8):
    """Build the PTB-XL window set. `max_records` subsamples (stratified by fold) for quick runs;
    `res` in {100, 500} selects records100 / records500 (both are resampled to 250 Hz)."""
    import pandas as pd
    import wfdb
    d = download_ptbxl_meta(data_dir)
    df = pd.read_csv(os.path.join(d, "ptbxl_database.csv"), index_col="ecg_id")
    df["scp_codes"] = df["scp_codes"].apply(ast.literal_eval)
    df["y"] = df["scp_codes"].apply(label_ptbxl)
    if max_records is not None and max_records < len(df):
        per_fold = max(1, max_records // 10)
        df = pd.concat([g.sample(min(len(g), per_fold), random_state=seed)
                        for _, g in df.groupby("strat_fold")])
    col = "filename_hr" if res == 500 else "filename_lr"
    rels = df[col].tolist()
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(_dl_record, [(d, r) for r in rels]))
    X, keep = [], []
    for i, rel in enumerate(rels):
        rec = wfdb.rdrecord(os.path.join(d, rel))
        names = [n.upper() for n in rec.sig_name]
        sig = np.nan_to_num(rec.p_signal[:, names.index("II")].astype(np.float64))
        x = resample_to_fs(sig, rec.fs)
        x = x[:C.WIN] if len(x) >= C.WIN else np.pad(x, (0, C.WIN - len(x)), mode="edge")
        X.append(x.astype(np.float32))
    folds = df["strat_fold"].to_numpy()
    split = np.where(folds <= 8, 0, np.where(folds == C.PTBXL_VAL_FOLD, 1, 2))
    return {"X": np.stack(X), "y": df["y"].to_numpy(), "rec": df["patient_id"].astype(int).astype(str).to_numpy(),
            "t0": np.zeros(len(df)), "split": split}


# ============================================================================= cache / API
def _plain(v):
    """numpy array without object dtype (strings -> unicode) so the cache never needs pickle."""
    v = np.asarray(v)
    return v.astype(str) if v.dtype == object else v


def _save(path, ds):
    flat = {k: _plain(v) for k, v in ds.items() if k != "replay"}
    if "replay" in ds:
        for k, v in ds["replay"].items():
            flat["replay_" + k] = _plain(v)
    np.savez_compressed(path, **flat)


def _load(path):
    z = np.load(path, allow_pickle=True)          # our own cache; older caches may hold object arrays
    ds = {k: _plain(z[k]) for k in z.files if not k.startswith("replay_")}
    rp = {k[7:]: _plain(z[k]) for k in z.files if k.startswith("replay_")}
    if rp:
        rp["stride_s"] = float(rp["stride_s"])
        ds["replay"] = rp
    return ds


def load_dataset(name, data_dir="data", synthetic_mode=False, ptbxl_max=None, ptbxl_res=500, rebuild=False):
    """name in {'mitbih', 'ptbxl'}. Windows are cached under <data_dir>/cache/."""
    if synthetic_mode:
        return synthetic.build_synthetic_dataset(style="A" if name == "mitbih" else "B",
                                                 seed=1 if name == "mitbih" else 2)
    os.makedirs(os.path.join(data_dir, "cache"), exist_ok=True)
    tag = name if name == "mitbih" else f"ptbxl_{ptbxl_res}_{ptbxl_max or 'all'}"
    cache = os.path.join(data_dir, "cache", tag + ".npz")
    if os.path.exists(cache) and not rebuild:
        return _load(cache)
    ds = build_mitbih(data_dir) if name == "mitbih" else build_ptbxl(data_dir, ptbxl_res, ptbxl_max)
    _save(cache, ds)
    return ds


def get_split(ds, split, drop_other=True):
    """split in {'train','val','test'} -> (X, y, rec, t0)."""
    sid = {"train": 0, "val": 1, "test": 2}[split]
    m = ds["split"] == sid
    if drop_other:
        m &= ds["y"] >= 0
    return ds["X"][m], ds["y"][m], ds["rec"][m], ds["t0"][m]


def merge(ds_a, ds_b):
    """Concatenate two datasets (for combined-training experiments). Replay is dropped."""
    out = {k: np.concatenate([ds_a[k], ds_b[k]]) for k in ["X", "y", "rec", "t0", "split"]}
    out["rec"] = out["rec"].astype(str)
    return out


def class_counts(y):
    return {c: int((y == i).sum()) for i, c in enumerate(C.CLASSES)} | {"OTHER": int((y < 0).sum())}
