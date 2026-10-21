"""Metrics, calibration (temperature scaling, ECE, Brier) and uncertainty / rejection utilities."""
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax, softmax
from sklearn.metrics import f1_score, roc_auc_score

from .config import CLASSES, N_CLASSES


# ----------------------------------------------------------------------------- classification
def macro_f1(y, pred):
    """Macro-F1 over the classes present in the ground truth (absent classes would give 0/0)."""
    labels = np.unique(y)
    return float(f1_score(y, pred, labels=labels, average="macro", zero_division=0))


def per_class_f1(y, pred):
    f = f1_score(y, pred, labels=list(range(N_CLASSES)), average=None, zero_division=0)
    present = np.isin(np.arange(N_CLASSES), np.unique(y))
    return {c: (float(v) if present[i] else float("nan")) for i, (c, v) in enumerate(zip(CLASSES, f))}


def macro_auroc(y, probs):
    """One-vs-rest macro AUROC over classes present in y (nan if < 2 classes)."""
    aucs = []
    for c in np.unique(y):
        if (y == c).sum() > 0 and (y != c).sum() > 0:
            aucs.append(roc_auc_score((y == c).astype(int), probs[:, c]))
    return float(np.mean(aucs)) if aucs else float("nan")


def accuracy(y, pred):
    return float((y == pred).mean())


# ----------------------------------------------------------------------------- calibration
def ece(probs, y, n_bins=15):
    """Expected calibration error (top-label, equal-width bins)."""
    conf = probs.max(1)
    pred = probs.argmax(1)
    acc = (pred == y).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(e)


def brier(probs, y):
    onehot = np.eye(probs.shape[1])[y]
    return float(((probs - onehot) ** 2).sum(1).mean())


def reliability(probs, y, n_bins=10):
    conf, pred = probs.max(1), probs.argmax(1)
    acc = (pred == y).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        rows.append((float((lo + hi) / 2), float(conf[m].mean()) if m.any() else np.nan,
                     float(acc[m].mean()) if m.any() else np.nan, int(m.sum())))
    return rows


def fit_temperature(logits, y):
    """Temperature scaling: T minimising the NLL on a validation set."""
    lp = lambda T: -log_softmax(logits / T, axis=1)[np.arange(len(y)), y].mean()
    r = minimize_scalar(lp, bounds=(0.05, 20.0), method="bounded")
    return float(r.x)


def apply_temperature(logits, T):
    return softmax(logits / T, axis=1)


# ----------------------------------------------------------------------------- uncertainty
def entropy(probs):
    return -(probs * np.log(probs + 1e-12)).sum(1)


def mutual_information(prob_samples):
    """prob_samples (M, N, C): epistemic uncertainty = H[mean p] - mean H[p]."""
    mean = prob_samples.mean(0)
    return entropy(mean) - np.mean([entropy(p) for p in prob_samples], axis=0)


def rejection_curve(y, pred, unc, coverages=(1.0, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5)):
    """Keep the most certain `coverage` fraction; report accuracy and macro-F1 on the kept set."""
    order = np.argsort(unc)
    rows = []
    for c in coverages:
        k = max(1, int(round(c * len(y))))
        idx = order[:k]
        rows.append({"coverage": c, "accuracy": accuracy(y[idx], pred[idx]),
                     "macro_f1": macro_f1(y[idx], pred[idx])})
    return rows


def aurc(y, pred, unc):
    """Area under the risk-coverage curve (lower is better)."""
    order = np.argsort(unc)
    err = (y[order] != pred[order]).astype(float)
    risk = np.cumsum(err) / np.arange(1, len(err) + 1)
    return float(risk.mean())


def threshold_for_coverage(unc_val, coverage):
    return float(np.quantile(unc_val, coverage))


def reject_stats(y, pred, reject):
    """Accuracy/F1 on accepted vs rejected windows and the rejection rate."""
    acc_m, rej_m = ~reject, reject
    out = {"reject_rate": float(reject.mean())}
    out["acc_accepted"] = accuracy(y[acc_m], pred[acc_m]) if acc_m.any() else float("nan")
    out["f1_accepted"] = macro_f1(y[acc_m], pred[acc_m]) if acc_m.any() else float("nan")
    out["acc_rejected"] = accuracy(y[rej_m], pred[rej_m]) if rej_m.any() else float("nan")
    return out
