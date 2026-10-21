"""Synthetic ECG generator.

PURPOSE: smoke-testing the pipeline offline (no PhysioNet access) and unit tests.
It is NOT a substitute for MIT-BIH / PTB-XL: any number obtained from synthetic data says
nothing about real-world performance. Two "styles" imitate the dataset shift between the
two real datasets (different amplitude scale, morphology and noise level).
"""
import numpy as np

from .config import FS, WIN, WIN_S, NSR, SBRAD, STACH, AF, PVC, N_CLASSES

# (offset s, amplitude, width s) of P, Q, R, S, T waves
_WAVES = [(-0.18, 0.15, 0.025), (-0.035, -0.12, 0.010), (0.0, 1.0, 0.012),
          (0.035, -0.22, 0.012), (0.26, 0.30, 0.050)]


def _subject(rng, style):
    s = {"amp": rng.uniform(0.8, 1.3), "t_amp": rng.uniform(0.7, 1.4),
         "r_w": rng.uniform(0.010, 0.016), "p_amp": rng.uniform(0.7, 1.3),
         "noise": rng.uniform(0.005, 0.02)}
    if style == "B":      # "PTB-XL-like": different scale, deeper S, taller T, noisier
        s["amp"] *= 0.45
        s["t_amp"] *= 1.5
        s["noise"] *= 2.5
        s["s_amp"] = 1.6
    else:
        s["s_amp"] = 1.0
    return s


def _add_beat(sig, t, r_t, sub, pvc=False, p_wave=True):
    if pvc:
        comps = [(0.0, 1.5 * sub["amp"], 0.040), (0.12, -0.5 * sub["amp"], 0.050),
                 (0.30, -0.35 * sub["amp"] * sub["t_amp"], 0.070)]
    else:
        comps = []
        for k, (off, a, w) in enumerate(_WAVES):
            if k == 0 and not p_wave:
                continue
            a = a * sub["amp"]
            if k == 0:
                a *= sub["p_amp"]
            if k == 3:
                a *= sub["s_amp"]
            if k == 4:
                a *= sub["t_amp"]
            if k == 2:
                w = sub["r_w"]
            comps.append((off, a, w))
    lo = int(max(0, (r_t - 0.6) * FS))
    hi = int(min(len(t), (r_t + 0.7) * FS))
    tt = t[lo:hi]
    for off, a, w in comps:
        sig[lo:hi] += a * np.exp(-0.5 * ((tt - (r_t + off)) / w) ** 2)


def _rr_sequence(rng, cls):
    """R-peak times (s) for one 10 s window of the given rhythm class."""
    times, pvc_flags = [], []
    t = rng.uniform(0.1, 0.6)
    if cls == AF:
        while t < WIN_S + 0.7:
            times.append(t); pvc_flags.append(False)
            t += rng.uniform(0.35, 1.1)
        return np.array(times), pvc_flags
    hr = {NSR: rng.uniform(62, 98), SBRAD: rng.uniform(38, 57),
          STACH: rng.uniform(105, 150), PVC: rng.uniform(62, 98)}[cls]
    rr0 = 60.0 / hr
    n_pvc = rng.integers(1, 4) if cls == PVC else 0
    pvc_pos = set(rng.choice(np.arange(2, max(3, int(WIN_S / rr0) - 1)), size=min(n_pvc, 2),
                             replace=False)) if cls == PVC else set()
    k = 0
    while t < WIN_S + 0.7:
        if k in pvc_pos:
            t_pvc = t - rr0 + 0.62 * rr0 * rng.uniform(0.9, 1.1)   # premature
            if times and t_pvc > times[-1] + 0.25:
                times.append(t_pvc); pvc_flags.append(True)
                t = t_pvc + 1.35 * rr0                              # compensatory pause
                k += 1
                continue
        times.append(t); pvc_flags.append(False)
        t += rr0 * (1 + rng.normal(0, 0.02))
        k += 1
    return np.array(times), pvc_flags


def make_window(rng, cls, sub):
    t = np.arange(WIN) / FS
    sig = np.zeros(WIN)
    times, flags = _rr_sequence(rng, cls)
    for rt, pv in zip(times, flags):
        _add_beat(sig, t, rt, sub, pvc=pv, p_wave=(cls != AF))
    if cls == AF:   # fibrillatory baseline
        f = rng.uniform(5, 7)
        sig += 0.04 * sub["amp"] * np.sin(2 * np.pi * f * t + rng.uniform(0, 6)) * rng.uniform(0.6, 1.4)
    sig += 0.1 * sub["amp"] * np.sin(2 * np.pi * rng.uniform(0.15, 0.35) * t + rng.uniform(0, 6))
    sig += rng.normal(0, sub["noise"], WIN)
    return sig.astype(np.float32)


def make_records(n_records, windows_per_record, style="A", seed=0, prefix="S"):
    """Contiguous synthetic 'records' with rhythm episodes (Markov chain over classes).

    Returns dict with X (N, WIN), y, rec (str array), widx (window index within record)."""
    rng = np.random.default_rng(seed)
    X, y, rec, widx = [], [], [], []
    for r in range(n_records):
        sub = _subject(rng, style)
        cls = int(rng.choice(N_CLASSES, p=[0.4, 0.15, 0.15, 0.15, 0.15]))
        for k in range(windows_per_record):
            if rng.random() > 0.8:                     # episode change
                cls = int(rng.choice(N_CLASSES, p=[0.4, 0.15, 0.15, 0.15, 0.15]))
            X.append(make_window(rng, cls, sub))
            y.append(cls); rec.append(f"{prefix}{r:02d}"); widx.append(k)
    return {"X": np.stack(X), "y": np.array(y), "rec": np.array(rec), "widx": np.array(widx)}


def build_synthetic_dataset(style="A", n_train=12, n_val=4, n_test=6, wpr=40, seed=0):
    """Returns dict of arrays with a `split` column (0 train / 1 val / 2 test) and
    the same fields as the real dataset builders. `replay` = test records in order."""
    parts = []
    for split, n, off in [(0, n_train, 0), (1, n_val, 100), (2, n_test, 200)]:
        d = make_records(n, wpr, style=style, seed=seed + off, prefix=f"{style}{split}_")
        d["split"] = np.full(len(d["y"]), split)
        d["t0"] = d["widx"] * float(WIN_S)
        parts.append(d)
    out = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    te = out["split"] == 2
    out["replay"] = {"X": out["X"][te], "y": out["y"][te], "rec": out["rec"][te],
                     "t0": out["t0"][te], "stride_s": float(WIN_S)}
    return out
