"""Noise sources and SNR-controlled corruption.

Real noise: MIT-BIH Noise Stress Test Database (NSTDB) records `bw` (baseline wander),
`em` (electrode motion) and `ma` (muscle artifact). To avoid training and testing on the
same noise samples, the first half of each noise recording is used for training
augmentation and the second half for evaluation.

Fallback: synthetic noise generators (used only for offline smoke tests).
"""
import os

import numpy as np
from scipy import signal as sps

from .config import FS, NOISE_KINDS


def _resample(x, fs_in):
    if fs_in == FS:
        return x
    from math import gcd
    g = gcd(int(fs_in), FS)
    return sps.resample_poly(x, FS // g, int(fs_in) // g)


def synth_noise(kind, n, seed=0):
    """Synthetic noise of `n` samples at FS (unit-ish scale)."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / FS
    if kind == "bw":      # slow drifts: sum of sinusoids (0.05-0.6 Hz) + random walk
        x = sum(rng.uniform(0.5, 1.5) * np.sin(2 * np.pi * rng.uniform(0.05, 0.6) * t +
                                               rng.uniform(0, 6)) for _ in range(5))
        return x
    if kind == "ma":      # muscle: broadband EMG-like, high-pass 20 Hz
        sos = sps.butter(4, 20, btype="high", fs=FS, output="sos")
        return sps.sosfilt(sos, rng.normal(0, 1, n))
    if kind == "em":      # electrode motion: random large transients with exponential decay
        x = np.zeros(n)
        for _ in range(max(1, n // (3 * FS))):
            c = rng.integers(0, n)
            w = rng.uniform(0.2, 1.0) * FS
            a = rng.normal(0, 3)
            idx = np.arange(max(0, c - int(3 * w)), min(n, c + int(3 * w)))
            x[idx] += a * np.exp(-np.abs(idx - c) / w) * np.sin(2 * np.pi * rng.uniform(1, 6) *
                                                               (idx - c) / FS)
        return x + rng.normal(0, 0.2, n)
    raise ValueError(kind)


class NoiseBank:
    """Holds long noise signals per kind and serves random segments."""

    def __init__(self, signals):
        self.sig = signals            # kind -> 1D array at FS

    @classmethod
    def from_nstdb(cls, nstdb_dir):
        import wfdb
        sig = {}
        for k in NOISE_KINDS:
            rec = wfdb.rdrecord(os.path.join(nstdb_dir, k))
            x = rec.p_signal[:, 0].astype(np.float64)
            sig[k] = _resample(x, rec.fs)
        return cls(sig)

    @classmethod
    def synthetic(cls, minutes=10, seed=123):
        n = int(minutes * 60 * FS)
        return cls({k: synth_noise(k, n, seed + i) for i, k in enumerate(NOISE_KINDS)})

    def _span(self, kind, part):
        n = len(self.sig[kind])
        return (0, n // 2) if part == "train" else (n // 2, n)

    def sample(self, kind, n, length, rng, part="test"):
        lo, hi = self._span(kind, part)
        starts = rng.integers(lo, hi - length, size=n)
        seg = np.stack([self.sig[kind][s:s + length] for s in starts])
        return seg - seg.mean(axis=1, keepdims=True)

    def corrupt(self, X, kind, snr_db, rng, part="test"):
        """Add noise of `kind` to every window at the given SNR (dB, per-window power)."""
        noise = self.sample(kind, len(X), X.shape[1], rng, part)
        px = X.var(axis=1, keepdims=True) + 1e-12
        pn = noise.var(axis=1, keepdims=True) + 1e-12
        alpha = np.sqrt(px / (pn * 10 ** (snr_db / 10.0)))
        return (X + alpha * noise).astype(np.float32)

    def corrupt_mixed(self, X, rng, snr_range=(6, 24), part="train"):
        """Random noise kind and SNR per window (training augmentation)."""
        out = X.copy()
        kinds = rng.choice(NOISE_KINDS, size=len(X))
        snrs = rng.uniform(*snr_range, size=len(X))
        for k in NOISE_KINDS:
            idx = np.where(kinds == k)[0]
            for s in np.unique(np.round(snrs[idx])):
                sub = idx[np.round(snrs[idx]) == s]
                out[sub] = self.corrupt(X[sub], k, float(s), rng, part)
        return out


def measured_snr_db(clean, noisy):
    n = noisy - clean
    return 10 * np.log10(clean.var(axis=1) / (n.var(axis=1) + 1e-12))


def download_nstdb(data_dir):
    """Download the three NSTDB noise records (bw, em, ma)."""
    from .datasets import dl_with_retry
    d = os.path.join(data_dir, "nstdb")
    os.makedirs(d, exist_ok=True)
    for k in NOISE_KINDS:
        for e in ("dat", "hea"):
            dl_with_retry("nstdb/1.0.0", d, f"{k}.{e}")
    return d
