"""SIMULATED decision-support framework.

>>> This is NOT a clinically validated pacemaker controller. <<<
It replays recorded ECG windows through predefined policies and measures decision-level metrics
(false-trigger rate, detection delay, missed episodes). Actions are software labels, not therapy.

Actions
    0 NONE          reliable, normal
    1 LOG           reliable, low risk -> log and keep monitoring
    2 RATE_SUPPORT  reliable, bradycardia with high calibrated risk -> simulated rate-support recommendation
    3 ALERT         reliable, tachycardia / AF / PVC with high calibrated risk -> simulated clinician alert
    4 UNRELIABLE    signal or prediction unreliable -> "signal unreliable" alert, hold last safe state
"""
import numpy as np

from .config import AF, NSR, PVC, SBRAD, STACH
from .preprocess import heart_rate, rr_cv

NONE, LOG, RATE_SUPPORT, ALERT, UNRELIABLE = range(5)
ACTION_NAMES = ["none", "log", "rate_support", "alert", "unreliable"]


# ----------------------------------------------------------------------------- policies
def policy_ml(probs, unreliable, tau_risk, use_reject=True):
    """probs: calibrated class probabilities (N,5); unreliable: bool (N,)."""
    risk = 1.0 - probs[:, NSR]
    pred = probs.argmax(1)
    act = np.full(len(probs), NONE)
    abn = pred != NSR
    high = abn & (risk >= tau_risk)
    act[abn & ~high] = LOG
    act[high & (pred == SBRAD)] = RATE_SUPPORT
    act[high & np.isin(pred, [STACH, AF, PVC])] = ALERT
    if use_reject:
        act[unreliable] = UNRELIABLE
    return act


def policy_hr(hr, lo, hi):
    """Fixed-threshold baseline 1: heart rate only."""
    act = np.full(len(hr), NONE)
    bad = (hr < lo) | (hr > hi)
    act[bad & (hr < lo)] = RATE_SUPPORT
    act[bad & (hr > hi)] = ALERT
    return act


def policy_hr_rr(hr, cv, lo, hi, cv_thr):
    """Fixed-threshold baseline 2: heart rate + R-R irregularity (AF-style) rule."""
    act = policy_hr(hr, lo, hi)
    act[(cv > cv_thr) & (act == NONE)] = ALERT
    return act


def window_rule_features(X):
    hr = np.array([heart_rate(w) for w in X])
    cv = np.array([rr_cv(w) for w in X])
    return np.nan_to_num(hr, nan=75.0), np.nan_to_num(cv, nan=0.0)


def is_alert(act):
    return (act == RATE_SUPPORT) | (act == ALERT)


# ----------------------------------------------------------------------------- scoring
def _episodes(y, rec, t0, stride_s):
    """Index ranges (record-wise, contiguous in time) of consecutive abnormal windows."""
    eps = []
    order = np.lexsort((t0, rec))
    cur = None
    for k, i in enumerate(order):
        abn = y[i] > 0
        cont = (cur is not None and rec[i] == rec[order[k - 1]] and abs(t0[i] - t0[order[k - 1]] - stride_s) < 1e-6)
        if abn and cont:
            cur.append(i)
        elif abn:
            if cur:
                eps.append(cur)
            cur = [i]
        else:
            if cur:
                eps.append(cur)
            cur = None
    if cur:
        eps.append(cur)
    return eps


def _dilate(abn, rec, t0, k):
    """Mark windows within k windows (same record) of an abnormal window."""
    order = np.lexsort((t0, rec))
    ab, r = abn[order], rec[order]
    out = ab.copy()
    for s_ in range(1, k + 1):
        if s_ >= len(ab):
            break
        same = r[s_:] == r[:-s_]
        out[s_:] |= ab[:-s_] & same
        out[:-s_] |= ab[s_:] & same
    res = np.empty_like(out)
    res[order] = out
    return res


def _count_events(mask, rec, t0):
    """Number of runs of consecutive True windows (per record)."""
    order = np.lexsort((t0, rec))
    ev, prev = 0, False
    for k, i in enumerate(order):
        cur = bool(mask[i])
        same = k > 0 and rec[i] == rec[order[k - 1]]
        if cur and not (prev and same):
            ev += 1
        prev = cur
    return ev


def score(act, y, rec, t0, stride_s, tol_s=10.0):
    """Decision-level metrics for one policy on a replay set.

    A window labelled NSR that lies within `tol_s` seconds of an abnormal window is ambiguous (the
    10-s window only partly overlaps the episode), so the primary false-trigger metric ignores
    alerts there ("tolerant"). The `_strict` variants count every alert on an NSR-labelled window."""
    alert = is_alert(act)
    scored = y >= 0
    hours = scored.sum() * stride_s / 3600.0
    abn = y > 0
    near = _dilate(abn, rec, t0, int(round(tol_s / stride_s)))
    calm = (y == 0) & ~near
    strict = y == 0
    ft = _count_events(alert & calm, rec, t0)
    ft_strict = _count_events(alert & strict, rec, t0)
    delays, missed = [], 0
    eps = _episodes(y, rec, t0, stride_s)
    for ep in eps:
        hit = [j for j, i in enumerate(ep) if alert[i]]
        if hit:
            delays.append(hit[0] * stride_s)
        else:
            missed += 1
    nan = float("nan")
    return {"false_triggers_per_hour": ft / hours if hours else nan,
            "false_triggers_strict_per_hour": ft_strict / hours if hours else nan,
            "episodes": len(eps), "missed_episode_rate": missed / len(eps) if eps else nan,
            "delay_mean_s": float(np.mean(delays)) if delays else nan,
            "delay_median_s": float(np.median(delays)) if delays else nan,
            "window_sensitivity": float(alert[abn].mean()) if abn.any() else nan,
            "window_false_alert_rate": float(alert[calm].mean()) if calm.any() else nan,
            "window_false_alert_rate_strict": float(alert[strict].mean()) if strict.any() else nan,
            "unreliable_time_frac": float((act[scored] == UNRELIABLE).mean())}


def tune_fixed_thresholds(hr, cv, y):
    """Grid-search baseline thresholds on VALIDATION windows (maximise F1 of alert vs abnormal)."""
    m = y >= 0
    hr, cv, y = hr[m], cv[m], y[m]
    abn = y > 0
    best1 = (-1, None)
    best2 = (-1, None)

    def f1(alert):
        tp = (alert & abn).sum(); fp = (alert & ~abn).sum(); fn = (~alert & abn).sum()
        return 2 * tp / (2 * tp + fp + fn + 1e-9)

    for lo in range(40, 62, 2):
        for hi in range(95, 160, 5):
            a = (hr < lo) | (hr > hi)
            s = f1(a)
            if s > best1[0]:
                best1 = (s, (lo, hi))
            for c in (0.08, 0.1, 0.15, 0.2, 0.25, 0.3):
                s2 = f1(a | (cv > c))
                if s2 > best2[0]:
                    best2 = (s2, (lo, hi, c))
    return {"hr": best1[1], "hr_rr": best2[1]}


def tune_tau(probs, unreliable, y, grid=np.linspace(0.5, 0.99, 25)):
    """Choose the risk threshold maximising window-level F1 of alert-vs-abnormal on validation."""
    m = y >= 0
    abn = (y > 0)[m]
    best, bt = -1, 0.5
    for t in grid:
        a = is_alert(policy_ml(probs[m], unreliable[m], t))
        tp = (a & abn).sum(); fp = (a & ~abn).sum(); fn = (~a & abn).sum()
        s = 2 * tp / (2 * tp + fp + fn + 1e-9)
        if s > best:
            best, bt = s, float(t)
    return bt
