"""Signal preprocessing: filtering, normalisation, R-peak detection, HRV features,
and a simple signal-quality index (SQI)."""
import numpy as np
from scipy import signal as sps
from scipy.stats import kurtosis

from .config import FS

_SOS_BP = sps.butter(3, [0.5, 40.0], btype="band", fs=FS, output="sos")
_SOS_QRS = sps.butter(2, [5.0, 15.0], btype="band", fs=FS, output="sos")


def bandpass(x, axis=-1):
    """Zero-phase 0.5-40 Hz band-pass (removes baseline wander and HF noise)."""
    return sps.sosfiltfilt(_SOS_BP, x, axis=axis)


def zscore(x, axis=-1, eps=1e-6):
    m = x.mean(axis=axis, keepdims=True)
    s = x.std(axis=axis, keepdims=True)
    return (x - m) / (s + eps)


def prepare_input(raw, filt=True):
    """raw (N, L) -> model input (N, 1, L) float32: [band-pass] + per-window z-score."""
    x = np.asarray(raw, dtype=np.float64)
    if filt:
        x = bandpass(x, axis=-1)
    x = zscore(x, axis=-1)
    return x[:, None, :].astype(np.float32)


# ----------------------------------------------------------------------------- R-peaks
def detect_rpeaks(x, fs=FS):
    """Pan-Tompkins-style R-peak detector for one window.

    band-pass 5-15 Hz -> derivative -> squaring -> moving-window integration ->
    adaptive threshold + refractory period -> refinement on the filtered signal.
    Returns sample indices (int array).
    """
    x = np.asarray(x, dtype=np.float64)
    if not np.isfinite(x).all() or x.std() < 1e-9:
        return np.array([], dtype=int)
    f = sps.sosfiltfilt(_SOS_QRS, x)
    d = np.gradient(f)
    w = max(int(0.15 * fs), 1)
    mwi = np.convolve(d ** 2, np.ones(w) / w, mode="same")
    ref = np.percentile(mwi, 99)
    if ref <= 0:
        return np.array([], dtype=int)
    pk, _ = sps.find_peaks(mwi, height=0.3 * ref, distance=int(0.25 * fs))
    if len(pk) == 0:
        return np.array([], dtype=int)
    af = np.abs(f)
    r = int(0.08 * fs)
    out = []
    for p in pk:
        lo, hi = max(0, p - r), min(len(af), p + r + 1)
        out.append(lo + int(np.argmax(af[lo:hi])))
    out = np.unique(np.array(out, dtype=int))
    keep = [out[0]]                      # enforce refractory period after refinement
    for p in out[1:]:
        if p - keep[-1] >= int(0.25 * fs):
            keep.append(p)
    return np.array(keep, dtype=int)


FEATURE_NAMES = ["n_beats", "hr", "mean_rr", "sdnn", "rmssd", "pnn50", "cv_rr", "min_rr",
                 "max_rr", "max_min_ratio", "mad_rr", "frac_premature", "frac_pause",
                 "amp_cv", "width_ms", "width_cv"]


def hrv_features(x, fs=FS):
    """Hand-crafted rhythm + morphology features used by baseline B2."""
    r = detect_rpeaks(x, fs)
    feat = np.zeros(len(FEATURE_NAMES), dtype=np.float32)
    feat[0] = len(r)
    if len(r) < 3:
        return feat
    rr = np.diff(r) / fs
    med = np.median(rr)
    f = sps.sosfiltfilt(_SOS_QRS, np.asarray(x, dtype=np.float64))
    af = np.abs(f)
    amp = af[r]
    half = int(0.12 * fs)
    widths = []
    for p, a in zip(r, amp):
        seg = af[max(0, p - half): p + half + 1]
        widths.append((seg > 0.5 * a).sum() / fs * 1000.0)
    widths = np.array(widths)
    drr = np.abs(np.diff(rr))
    feat[1:] = [60.0 / rr.mean(), rr.mean(), rr.std(), np.sqrt(np.mean(np.diff(rr) ** 2)),
                (drr > 0.05).mean() if len(drr) else 0.0, rr.std() / (rr.mean() + 1e-9),
                rr.min(), rr.max(), rr.max() / (rr.min() + 1e-9),
                np.median(np.abs(rr - med)), (rr < 0.8 * med).mean(), (rr > 1.2 * med).mean(),
                amp.std() / (amp.mean() + 1e-9), widths.mean(),
                widths.std() / (widths.mean() + 1e-9)]
    return np.nan_to_num(feat)


def feature_matrix(raw, fs=FS):
    return np.stack([hrv_features(w, fs) for w in raw])


def heart_rate(x, fs=FS):
    """Mean heart rate (bpm) of a window from detected R-peaks; nan if < 3 beats."""
    r = detect_rpeaks(x, fs)
    if len(r) < 3:
        return float("nan")
    return 60.0 / (np.diff(r) / fs).mean()


def rr_cv(x, fs=FS):
    """Coefficient of variation of R-R intervals (irregularity measure); nan if < 4 beats."""
    r = detect_rpeaks(x, fs)
    if len(r) < 4:
        return float("nan")
    rr = np.diff(r) / fs
    return rr.std() / (rr.mean() + 1e-9)


# ----------------------------------------------------------------------------- SQI
SQI_NAMES = ["kurt", "hf_ratio", "bw_ratio", "rr_valid", "flat"]


def sqi_features(raw, fs=FS):
    """Per-window signal-quality features computed on the *raw* (unfiltered) window.

    kurt     : kurtosis of the band-passed signal (clean ECG is spiky -> high kurtosis)
    hf_ratio : power above 40 Hz relative to 0.5-40 Hz power (muscle noise raises it)
    bw_ratio : power below 0.5 Hz relative to total power (baseline wander raises it)
    rr_valid : fraction of detected R-R intervals within [0.25, 2.0] s
    flat     : 1 if the window is (almost) flat / non-finite
    """
    raw = np.asarray(raw, dtype=np.float64)
    out = np.zeros((len(raw), len(SQI_NAMES)), dtype=np.float32)
    freqs = np.fft.rfftfreq(raw.shape[1], 1.0 / fs)
    P = np.abs(np.fft.rfft(raw - raw.mean(axis=1, keepdims=True), axis=1)) ** 2
    inband = P[:, (freqs >= 0.5) & (freqs <= 40)].sum(1) + 1e-12
    hf = P[:, freqs > 40].sum(1)
    lf = P[:, freqs < 0.5].sum(1)
    tot = P.sum(1) + 1e-12
    filt = bandpass(raw, axis=-1)
    for i in range(len(raw)):
        flat = float((not np.isfinite(raw[i]).all()) or raw[i].std() < 1e-6)
        k = kurtosis(filt[i], fisher=False) if not flat else 0.0
        r = detect_rpeaks(raw[i], fs)
        if len(r) >= 2:
            rr = np.diff(r) / fs
            rv = float(((rr > 0.25) & (rr < 2.0)).mean())
        else:
            rv = 0.0
        out[i] = [k, hf[i] / inband[i], lf[i] / tot[i], rv, flat]
    return out


class SQIGate:
    """Flags a window as unreliable. Thresholds are fitted on clean validation data."""

    def __init__(self, q=1.0, use_bw=False):
        """use_bw=False: baseline wander is NOT a rejection criterion because the 0.5 Hz high-pass
        in the pipeline removes it; set True to also reject on low-frequency power."""
        self.q = q
        self.use_bw = use_bw
        self.thr = None

    def fit(self, sqi_clean):
        q = self.q
        self.thr = {
            "kurt": float(np.percentile(sqi_clean[:, 0], q)),
            "hf_ratio": float(np.percentile(sqi_clean[:, 1], 100 - q)),
            "bw_ratio": float(np.percentile(sqi_clean[:, 2], 100 - q)),
            "rr_valid": 0.7,
        }
        return self

    def flag(self, sqi):
        t = self.thr
        bad = (sqi[:, 0] < t["kurt"]) | (sqi[:, 1] > t["hf_ratio"]) | (sqi[:, 3] < t["rr_valid"]) | (sqi[:, 4] > 0.5)
        if self.use_bw:
            bad = bad | (sqi[:, 2] > t["bw_ratio"])
        return bad
