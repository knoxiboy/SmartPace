"""Reliability-aware noise rejection policies for ECG classification.

Implements and evaluates five rejection policies under changing signal quality and noise:
1. Signal-quality gate only (SQI only)
2. Predictive uncertainty gate only (Uncertainty only)
3. Fixed dual-gate baseline (SQI gate + fixed entropy threshold)
4. Validation-calibrated dual gate (calibrated on validation data for target coverage/risk)
5. Adaptive reliability policy (joint continuous SQI + uncertainty scoring with abnormal-rhythm sensitivity)

Provides complete trade-off curve generation (coverage vs error rate, macro-F1, per-class recall,
false negative rate, clean input rejection, and cross-noise transfer analysis).
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.metrics import recall_score

from .config import CLASSES, N_CLASSES, NOISE_KINDS
from .evaluation import accuracy, entropy, macro_f1
from .preprocess import SQIGate, sqi_features


def compute_sqi_degradation_index(sqi_feats: np.ndarray, sqi_clean_val: np.ndarray) -> np.ndarray:
    """Computes a continuous normalized SQI degradation score in [0, 1].
    
    Higher score means more severe signal corruption.
    sqi_feats columns: [0: kurtosis, 1: hf_ratio, 2: bw_ratio, 3: rr_valid, 4: flat]
    """
    feats = np.asarray(sqi_feats, dtype=np.float32)
    clean = np.asarray(sqi_clean_val, dtype=np.float32)

    # Reference statistics from clean validation set
    clean_kurt_p5 = np.percentile(clean[:, 0], 5.0)
    clean_kurt_p50 = np.median(clean[:, 0])
    clean_hf_p50 = np.median(clean[:, 1])
    clean_hf_p95 = np.percentile(clean[:, 1], 95.0) + 1e-6
    clean_bw_p50 = np.median(clean[:, 2])
    clean_bw_p95 = np.percentile(clean[:, 2], 95.0) + 1e-6

    # 1. Kurtosis loss (kurtosis lower than clean median indicates flattened spikes / Gaussian noise)
    kurt_deg = np.clip((clean_kurt_p50 - feats[:, 0]) / (clean_kurt_p50 - clean_kurt_p5 + 1e-6), 0.0, 3.0) / 3.0

    # 2. High-frequency excess (EMG / muscle noise)
    hf_deg = np.clip((feats[:, 1] - clean_hf_p50) / (clean_hf_p95 - clean_hf_p50 + 1e-6), 0.0, 3.0) / 3.0

    # 3. Baseline wander excess (low-frequency drift)
    bw_deg = np.clip((feats[:, 2] - clean_bw_p50) / (clean_bw_p95 - clean_bw_p50 + 1e-6), 0.0, 3.0) / 3.0

    # 4. R-R invalidity (fraction of R-peaks outside physiological 0.25-2.0s)
    rr_deg = np.clip(1.0 - feats[:, 3], 0.0, 1.0)

    # 5. Flatline or non-finite indicator
    flat_deg = np.clip(feats[:, 4], 0.0, 1.0)

    # Composite degradation index
    deg_index = 0.30 * hf_deg + 0.25 * kurt_deg + 0.15 * bw_deg + 0.20 * rr_deg + 0.10 * flat_deg
    return np.clip(deg_index, 0.0, 1.0)


@dataclass
class RejectionEvaluationResult:
    coverage: float
    error_rate: float
    accuracy: float
    macro_f1: float
    abnormal_recall: float
    recall_per_class: Dict[str, float]
    arrhythmia_fn_rate: float
    clean_rejection_rate: float
    rejection_rate_overall: float


class BaseNoisePolicy:
    """Abstract base class for noise-aware rejection policies."""
    
    def score_unreliability(self, probs: np.ndarray, sqi_feats: np.ndarray) -> np.ndarray:
        """Higher score indicates higher unreliability / candidates for abstention."""
        raise NotImplementedError

    def decide(self, probs: np.ndarray, sqi_feats: np.ndarray, threshold: float) -> np.ndarray:
        """Returns boolean array of rejection decisions (True = reject, False = accept)."""
        scores = self.score_unreliability(probs, sqi_feats)
        return scores > threshold


class SQIOnlyPolicy(BaseNoisePolicy):
    """Rejection based solely on signal quality indices."""

    def __init__(self, sqi_clean_val: np.ndarray):
        self.sqi_clean_val = sqi_clean_val

    def score_unreliability(self, probs: np.ndarray, sqi_feats: np.ndarray) -> np.ndarray:
        return compute_sqi_degradation_index(sqi_feats, self.sqi_clean_val)


class UncertaintyOnlyPolicy(BaseNoisePolicy):
    """Rejection based solely on predictive uncertainty (Shannon entropy)."""

    def score_unreliability(self, probs: np.ndarray, sqi_feats: np.ndarray) -> np.ndarray:
        H = entropy(probs)
        max_ent = np.log(probs.shape[1] + 1e-12)
        return np.clip(H / max_ent, 0.0, 1.0)


class FixedDualGatePolicy(BaseNoisePolicy):
    """SmartPace baseline: Fixed dual-gate (SQIGate + fixed entropy tau)."""

    def __init__(self, sqi_gate: SQIGate, tau_unc: float):
        self.sqi_gate = sqi_gate
        self.tau_unc = tau_unc

    def score_unreliability(self, probs: np.ndarray, sqi_feats: np.ndarray) -> np.ndarray:
        H = entropy(probs)
        q_bad = self.sqi_gate.flag(sqi_feats).astype(float)
        # Scaled so that q_bad forces score above 1.0, and H is normalized
        return q_bad * 1.5 + (H / self.tau_unc) * 0.5

    def decide(self, probs: np.ndarray, sqi_feats: np.ndarray, threshold: float = 1.0) -> np.ndarray:
        q_bad = self.sqi_gate.flag(sqi_feats)
        unc_bad = entropy(probs) > self.tau_unc
        return q_bad | unc_bad


class ValCalibratedDualGatePolicy(BaseNoisePolicy):
    """Dual gate calibrated on validation data to meet target coverage/risk constraints."""

    def __init__(self, sqi_clean_val: np.ndarray):
        self.sqi_clean_val = sqi_clean_val
        self.tau_sqi: float = 0.5
        self.tau_unc: float = 0.5

    def calibrate_for_coverage(self, val_probs: np.ndarray, val_sqi: np.ndarray, target_coverage: float):
        """Finds thresholds (tau_sqi, tau_unc) on validation data that yield target coverage."""
        q_scores = compute_sqi_degradation_index(val_sqi, self.sqi_clean_val)
        H_scores = entropy(val_probs) / np.log(val_probs.shape[1] + 1e-12)

        # Equal percentile split on unreliability
        # Rejection happens if q_score > tau_sqi OR H_scores > tau_unc
        # We sweep candidate percentiles p in [target_coverage, 1.0]
        best_p = target_coverage
        min_diff = 1e9
        for p in np.linspace(target_coverage, 1.0, 50):
            t_q = float(np.quantile(q_scores, p))
            t_u = float(np.quantile(H_scores, p))
            cov = float((~((q_scores > t_q) | (H_scores > t_u))).mean())
            diff = abs(cov - target_coverage)
            if diff < min_diff:
                min_diff = diff
                best_p = p
                self.tau_sqi = t_q
                self.tau_unc = t_u

    def score_unreliability(self, probs: np.ndarray, sqi_feats: np.ndarray) -> np.ndarray:
        q_scores = compute_sqi_degradation_index(sqi_feats, self.sqi_clean_val)
        H_scores = entropy(probs) / np.log(probs.shape[1] + 1e-12)
        # Unified unreliability ratio relative to calibrated thresholds
        ratio_q = q_scores / (self.tau_sqi + 1e-6)
        ratio_u = H_scores / (self.tau_unc + 1e-6)
        return np.maximum(ratio_q, ratio_u)

    def decide(self, probs: np.ndarray, sqi_feats: np.ndarray, threshold: float = 1.0) -> np.ndarray:
        scores = self.score_unreliability(probs, sqi_feats)
        return scores > threshold


class AdaptiveReliabilityPolicy(BaseNoisePolicy):
    """Adaptive reliability policy combining quality, uncertainty, and abnormal rhythm risk.
    
    Balances:
    - Detecting abnormal rhythms (avoids falsely dropping borderline arrhythmias)
    - Avoiding unreliable predictions (rejects corrupted or highly ambiguous waveforms)
    - Preserving sufficient coverage for clinical utility.
    """

    def __init__(self, sqi_clean_val: np.ndarray, w_unc: float = 0.5, w_sqi: float = 0.35,
                 w_interaction: float = 0.25, beta_abnormal: float = 0.15):
        self.sqi_clean_val = sqi_clean_val
        self.w_unc = w_unc
        self.w_sqi = w_sqi
        self.w_interaction = w_interaction
        self.beta_abnormal = beta_abnormal

    def score_unreliability(self, probs: np.ndarray, sqi_feats: np.ndarray) -> np.ndarray:
        q_deg = compute_sqi_degradation_index(sqi_feats, self.sqi_clean_val)
        H_norm = entropy(probs) / np.log(probs.shape[1] + 1e-12)

        # Abnormal probability (classes 1..4: SBRAD, STACH, AF, PVC)
        p_abnormal = probs[:, 1:].sum(axis=1) if probs.shape[1] > 1 else np.zeros(len(probs))

        # Composite unreliability score
        # Interaction term compounds severe noise with high ambiguity
        score = (self.w_unc * H_norm +
                 self.w_sqi * q_deg +
                 self.w_interaction * (H_norm * q_deg) -
                 self.beta_abnormal * p_abnormal)
        return np.asarray(score, dtype=np.float32)


def evaluate_policy_operating_point(
    y_true: np.ndarray,
    probs: np.ndarray,
    rejections: np.ndarray,
    clean_mask: Optional[np.ndarray] = None,
) -> RejectionEvaluationResult:
    """Computes clinical performance metrics for a specific binary rejection decision mask."""
    y = np.asarray(y_true, dtype=np.int64)
    pred = probs.argmax(axis=1)
    acc_mask = ~rejections
    coverage = float(acc_mask.mean())
    rejection_rate = float(rejections.mean())

    if not acc_mask.any():
        return RejectionEvaluationResult(
            coverage=0.0,
            error_rate=1.0,
            accuracy=0.0,
            macro_f1=0.0,
            abnormal_recall=0.0,
            recall_per_class={c: 0.0 for c in CLASSES},
            arrhythmia_fn_rate=0.0,
            clean_rejection_rate=1.0,
            rejection_rate_overall=1.0,
        )

    y_acc = y[acc_mask]
    pred_acc = pred[acc_mask]

    acc = accuracy(y_acc, pred_acc)
    err = 1.0 - acc
    f1 = macro_f1(y_acc, pred_acc)

    # Abnormal rhythm recall on accepted set
    # Abnormal classes are indices 1..4 (SBRAD, STACH, AF, PVC)
    abnormal_mask = y_acc > 0
    if abnormal_mask.any():
        abnormal_rec = float((pred_acc[abnormal_mask] > 0).mean())
    else:
        abnormal_rec = 1.0

    # Per-class recall
    rec_dict = {}
    for c_idx, c_name in enumerate(CLASSES):
        cls_mask = y_acc == c_idx
        if cls_mask.any():
            rec_dict[c_name] = float((pred_acc[cls_mask] == c_idx).mean())
        else:
            rec_dict[c_name] = float("nan")

    # False negative rate: true abnormal beats predicted as normal (0 / NSR)
    true_abnormal_in_acc = (y_acc > 0).sum()
    if true_abnormal_in_acc > 0:
        fn_count = int(((y_acc > 0) & (pred_acc == 0)).sum())
        fn_rate = float(fn_count / true_abnormal_in_acc)
    else:
        fn_rate = 0.0

    # Clean rejection rate
    if clean_mask is not None and clean_mask.any():
        clean_rej = float(rejections[clean_mask].mean())
    else:
        clean_rej = 0.0

    return RejectionEvaluationResult(
        coverage=coverage,
        error_rate=err,
        accuracy=acc,
        macro_f1=f1,
        abnormal_recall=abnormal_rec,
        recall_per_class=rec_dict,
        arrhythmia_fn_rate=fn_rate,
        clean_rejection_rate=clean_rej,
        rejection_rate_overall=rejection_rate,
    )


def generate_tradeoff_curve(
    policy: BaseNoisePolicy,
    probs: np.ndarray,
    sqi_feats: np.ndarray,
    y_true: np.ndarray,
    clean_mask: Optional[np.ndarray] = None,
    target_coverages: Optional[List[float]] = None,
) -> List[Dict]:
    """Sweeps target coverage levels to generate complete risk-coverage and F1 trade-off points."""
    if target_coverages is None:
        target_coverages = [0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0]

    scores = policy.score_unreliability(probs, sqi_feats)
    rows = []

    for cov in target_coverages:
        if isinstance(policy, FixedDualGatePolicy):
            # Fixed dual gate has an intrinsic operating point, but can also be modulated by threshold
            rejs = policy.decide(probs, sqi_feats)
        else:
            # Threshold chosen to match exact coverage quantile
            if cov >= 1.0:
                rejs = np.zeros(len(probs), dtype=bool)
            else:
                thr = float(np.quantile(scores, cov))
                rejs = scores > thr

        res = evaluate_policy_operating_point(y_true, probs, rejs, clean_mask=clean_mask)
        rows.append({
            "target_coverage": cov,
            "actual_coverage": res.coverage,
            "error_rate": res.error_rate,
            "accuracy": res.accuracy,
            "macro_f1": res.macro_f1,
            "abnormal_recall": res.abnormal_recall,
            "arrhythmia_fn_rate": res.arrhythmia_fn_rate,
            "clean_rejection_rate": res.clean_rejection_rate,
            "rejection_rate": res.rejection_rate_overall,
            "recall_nsr": res.recall_per_class.get("NSR", np.nan),
            "recall_sbrad": res.recall_per_class.get("SBRAD", np.nan),
            "recall_stach": res.recall_per_class.get("STACH", np.nan),
            "recall_af": res.recall_per_class.get("AF", np.nan),
            "recall_pvc": res.recall_per_class.get("PVC", np.nan),
        })

    return rows


def evaluate_noise_type_breakdown(
    policy: BaseNoisePolicy,
    probs: np.ndarray,
    sqi_feats: np.ndarray,
    noise_kinds: np.ndarray,
    snr_levels: np.ndarray,
    threshold: Optional[float] = None,
    target_coverage: float = 0.8,
) -> List[Dict]:
    """Computes rejection rate per noise type ('bw', 'em', 'ma') across SNR levels."""
    scores = policy.score_unreliability(probs, sqi_feats)
    if threshold is None:
        threshold = float(np.quantile(scores, target_coverage))
    rejections = scores > threshold

    rows = []
    for k in NOISE_KINDS:
        for snr in np.unique(snr_levels):
            if np.isnan(snr):
                continue
            m = (noise_kinds == k) & (snr_levels == snr)
            if m.any():
                rej_rate = float(rejections[m].mean())
                rows.append({"noise_kind": k, "snr_db": snr, "rejection_rate": rej_rate, "n_samples": int(m.sum())})

    # Clean condition
    clean_m = noise_kinds == "clean"
    if clean_m.any():
        rows.append({"noise_kind": "clean", "snr_db": np.nan, "rejection_rate": float(rejections[clean_m].mean()), "n_samples": int(clean_m.sum())})

    return rows
