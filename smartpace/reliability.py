"""Adaptive reliability and abstention module.

Combines three complementary signals for safe cardiac decision-support:
1. Signal Quality Gate (SQI): Detects corrupted, flat, or artifact-laden ECG signals.
2. Predictive Uncertainty: Detects ambiguous rhythms where model confidence is low.
3. Distribution Shift Detector: Detects unfamiliar waveforms differing from the training distribution
   using Mahalanobis distance on learned feature representations.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import roc_auc_score

from .config import CLASSES, N_CLASSES
from .evaluation import accuracy, ece, entropy, macro_f1


# -----------------------------------------------------------------------------
# Distribution Shift Detectors (Mahalanobis & Euclidean Baselines)
# -----------------------------------------------------------------------------
class MahalanobisShiftDetector:
    """Detects distribution shift / unfamiliar inputs using class-conditional Mahalanobis distance.

    Fits class-conditional centroids and a shared regularized covariance matrix on learned
    penultimate feature representations from in-distribution training data.
    """

    def __init__(self, alpha: float = 0.05, eps: float = 1e-4):
        """
        alpha: shrinkage factor towards diagonal spherical covariance (Ledoit-Wolf style regularization).
        eps: jitter added to diagonal to ensure strict positive-definiteness.
        """
        self.alpha = alpha
        self.eps = eps
        self.means_: Optional[Dict[int, np.ndarray]] = None
        self.precision_: Optional[np.ndarray] = None
        self.classes_: Optional[np.ndarray] = None
        self.threshold_: float = np.inf

    def fit(self, features: np.ndarray, labels: np.ndarray) -> "MahalanobisShiftDetector":
        """Fit class centroids and tied regularized precision matrix."""
        features = np.asarray(features, dtype=np.float64)
        labels = np.asarray(labels, dtype=int)
        classes = np.unique(labels)
        self.classes_ = classes
        d = features.shape[1]

        self.means_ = {}
        centered_list = []
        for c in classes:
            idx = np.where(labels == c)[0]
            if len(idx) > 0:
                mean_c = features[idx].mean(axis=0)
                self.means_[c] = mean_c
                centered_list.append(features[idx] - mean_c)

        # Pooled within-class covariance
        centered = np.vstack(centered_list)
        n = len(centered)
        cov = (centered.T @ centered) / max(1, n - len(classes))

        # Shrinkage regularization
        trace_avg = np.trace(cov) / d
        cov_reg = (1.0 - self.alpha) * cov + self.alpha * trace_avg * np.eye(d) + self.eps * np.eye(d)

        # Invert to obtain precision matrix
        try:
            self.precision_ = np.linalg.inv(cov_reg)
        except np.linalg.LinAlgError:
            self.precision_ = np.linalg.pinv(cov_reg)

        return self

    def score(self, features: np.ndarray) -> np.ndarray:
        """Compute minimum Mahalanobis distance across all learned class centroids.

        Higher score = more distant from in-distribution training manifold (higher shift).
        """
        if self.means_ is None or self.precision_ is None:
            raise RuntimeError("MahalanobisShiftDetector must be fitted before calling score().")

        features = np.asarray(features, dtype=np.float64)
        n = len(features)
        min_dists = np.full(n, np.inf, dtype=np.float64)

        for c, mu in self.means_.items():
            diff = features - mu
            # Mahalanobis distance squared: (x - mu) @ P @ (x - mu).T
            # Computed efficiently via batch quadratic form
            dist_sq = np.sum((diff @ self.precision_) * diff, axis=1)
            min_dists = np.minimum(min_dists, dist_sq)

        return np.sqrt(np.maximum(0.0, min_dists))

    def calibrate(self, val_features: np.ndarray, target_coverage: float = 0.95) -> float:
        """Set distance threshold so that target_coverage fraction of validation data is accepted."""
        scores = self.score(val_features)
        self.threshold_ = float(np.quantile(scores, target_coverage))
        return self.threshold_


class EuclideanShiftDetector:
    """Simpler baseline shift detector using Euclidean distance to the nearest class centroid."""

    def __init__(self):
        self.means_: Optional[Dict[int, np.ndarray]] = None
        self.threshold_: float = np.inf

    def fit(self, features: np.ndarray, labels: np.ndarray) -> "EuclideanShiftDetector":
        features = np.asarray(features, dtype=np.float64)
        labels = np.asarray(labels, dtype=int)
        self.means_ = {c: features[labels == c].mean(axis=0) for c in np.unique(labels)}
        return self

    def score(self, features: np.ndarray) -> np.ndarray:
        features = np.asarray(features, dtype=np.float64)
        centroid_mat = np.stack(list(self.means_.values()), axis=0)
        dists = cdist(features, centroid_mat, metric="euclidean")
        return dists.min(axis=1)

    def calibrate(self, val_features: np.ndarray, target_coverage: float = 0.95) -> float:
        scores = self.score(val_features)
        self.threshold_ = float(np.quantile(scores, target_coverage))
        return self.threshold_


# -----------------------------------------------------------------------------
# Adaptive Reliability & Abstention Module (Tri-Gate Framework)
# -----------------------------------------------------------------------------
@dataclass
class ReliabilityDecision:
    """Decision output for a single or batch of ECG windows."""
    action: np.ndarray                   # "ACCEPT", "REJECT_QUALITY", "ABSTAIN_UNCERTAIN", "ABSTAIN_UNFAMILIAR"
    is_accepted: np.ndarray              # bool mask (True if prediction should be utilized)
    rejection_reason: List[str]          # human-interpretable diagnostics
    pred_labels: np.ndarray              # predicted rhythm indices (0..4)
    probs: np.ndarray                    # calibrated class probability distribution
    confidence: np.ndarray               # top-class probability
    uncertainty: np.ndarray              # predictive entropy
    shift_score: np.ndarray              # Mahalanobis distance
    sqi_flags: np.ndarray                # True if failed SQI quality check


class AdaptiveAbstentionModule:
    """Integrates Signal Quality, Predictive Uncertainty, and Distribution Shift detection.

    Knows when the model does not know:
    - Poor signal quality -> Reject (Request clean signal)
    - High predictive uncertainty -> Abstain (Flag for review)
    - Distribution shift / unfamiliar -> Abstain (Flag for review)
    - Otherwise -> Accept prediction with confidence and explanation
    """

    def __init__(self, sqi_gate=None, shift_detector: Optional[MahalanobisShiftDetector] = None,
                 tau_unc: float = np.inf, tau_shift: float = np.inf):
        self.sqi_gate = sqi_gate
        self.shift_detector = shift_detector or MahalanobisShiftDetector()
        self.tau_unc = tau_unc
        self.tau_shift = tau_shift

    def calibrate(self, sqi_features_clean: np.ndarray, val_features: np.ndarray,
                  val_labels: np.ndarray, val_probs: np.ndarray,
                  unc_coverage: float = 0.95, shift_coverage: float = 0.95):
        """Calibrate thresholds systematically on validation data."""
        # 1. Calibrate shift detector
        self.shift_detector.fit(val_features, val_labels)
        self.tau_shift = self.shift_detector.calibrate(val_features, target_coverage=shift_coverage)

        # 2. Calibrate uncertainty threshold
        val_unc = entropy(val_probs)
        self.tau_unc = float(np.quantile(val_unc, unc_coverage))
        return self

    def decide(self, probs: np.ndarray, features: np.ndarray, sqi_features: Optional[np.ndarray] = None) -> ReliabilityDecision:
        """Evaluate input batch and return structured reliability decisions."""
        probs = np.asarray(probs, dtype=np.float32)
        features = np.asarray(features, dtype=np.float32)
        n = len(probs)

        pred_labels = probs.argmax(axis=1)
        conf = probs.max(axis=1)
        unc = entropy(probs)
        shift_scores = self.shift_detector.score(features)

        # 1. SQI Signal Quality check
        sqi_bad = np.zeros(n, dtype=bool)
        if self.sqi_gate is not None and sqi_features is not None:
            sqi_bad = self.sqi_gate.flag(sqi_features)

        # 2. Uncertainty check
        unc_bad = unc > self.tau_unc

        # 3. Distribution shift check
        shift_bad = shift_scores > self.tau_shift

        # Multi-tiered decision logic
        actions = np.full(n, "ACCEPT", dtype=object)
        reasons = []

        for i in range(n):
            r = []
            if sqi_bad[i]:
                actions[i] = "REJECT_QUALITY"
                r.append("Signal quality below acceptable clinical threshold (SQI)")
            elif shift_bad[i]:
                actions[i] = "ABSTAIN_UNFAMILIAR"
                r.append(f"Distribution shift / unfamiliar waveform (dist={shift_scores[i]:.2f} > tau={self.tau_shift:.2f})")
            elif unc_bad[i]:
                actions[i] = "ABSTAIN_UNCERTAIN"
                r.append(f"High predictive uncertainty (entropy={unc[i]:.2f} > tau={self.tau_unc:.2f})")
            else:
                actions[i] = "ACCEPT"
                r.append("Reliable prediction accepted")
            reasons.append("; ".join(r))

        is_accepted = (actions == "ACCEPT")

        return ReliabilityDecision(
            action=actions,
            is_accepted=is_accepted,
            rejection_reason=reasons,
            pred_labels=pred_labels,
            probs=probs,
            confidence=conf,
            uncertainty=unc,
            shift_score=shift_scores,
            sqi_flags=sqi_bad,
        )


# -----------------------------------------------------------------------------
# Benchmark Evaluation Across 4 Configurations
# -----------------------------------------------------------------------------
def evaluate_four_configurations(
    y_true: np.ndarray,
    probs: np.ndarray,
    features: np.ndarray,
    sqi_features: Optional[np.ndarray],
    abstention_module: AdaptiveAbstentionModule,
    ood_features: Optional[np.ndarray] = None,
) -> Dict[str, Dict[str, float]]:
    """Compare 4 configurations on the same evaluation dataset:
    1. Baseline classifier (no rejection)
    2. Classifier + Uncertainty estimation only
    3. Classifier + Signal Quality rejection only
    4. Full Adaptive System (Uncertainty + Signal Quality + Distribution Shift)

    Returns dictionary of metrics for each configuration.
    """
    y_true = np.asarray(y_true, dtype=int)
    probs = np.asarray(probs, dtype=np.float32)
    pred = probs.argmax(axis=1)
    unc = entropy(probs)
    n = len(y_true)

    shift_scores = abstention_module.shift_detector.score(features)
    sqi_bad = np.zeros(n, dtype=bool)
    if abstention_module.sqi_gate is not None and sqi_features is not None:
        sqi_bad = abstention_module.sqi_gate.flag(sqi_features)

    # Calculate OOD detection AUROC if OOD features provided
    ood_auroc_shift = float("nan")
    ood_auroc_unc = float("nan")
    if ood_features is not None and len(ood_features) > 0:
        ood_shift = abstention_module.shift_detector.score(ood_features)
        y_ood = np.concatenate([np.zeros(n, dtype=int), np.ones(len(ood_features), dtype=int)])
        combined_shift = np.concatenate([shift_scores, ood_shift])
        ood_auroc_shift = float(roc_auc_score(y_ood, combined_shift))

    # Error detection AUROC
    is_error = (pred != y_true).astype(int)
    def calc_error_auroc(score_vec):
        if 0 < is_error.sum() < len(is_error):
            return float(roc_auc_score(is_error, score_vec))
        return float("nan")

    # Define masks for the 4 configurations
    configs = {
        "1. Baseline Classifier": np.ones(n, dtype=bool),
        "2. Classifier + Uncertainty Gate": unc <= abstention_module.tau_unc,
        "3. Classifier + Signal Quality Gate": ~sqi_bad,
        "4. Full Adaptive Tri-Gate (SmartPace)": (~sqi_bad) & (unc <= abstention_module.tau_unc) & (shift_scores <= abstention_module.tau_shift),
    }

    # Combined ranking score for error detection
    composite_score = unc / (abstention_module.tau_unc + 1e-12) + shift_scores / (abstention_module.tau_shift + 1e-12)
    score_for_config = {
        "1. Baseline Classifier": unc,
        "2. Classifier + Uncertainty Gate": unc,
        "3. Classifier + Signal Quality Gate": sqi_bad.astype(float),
        "4. Full Adaptive Tri-Gate (SmartPace)": composite_score,
    }

    results = {}
    for cfg_name, accept_mask in configs.items():
        n_acc = int(accept_mask.sum())
        cov = float(n_acc / n)

        if n_acc > 0:
            y_acc = y_true[accept_mask]
            pred_acc = pred[accept_mask]
            probs_acc = probs[accept_mask]

            f1 = macro_f1(y_acc, pred_acc)
            acc = accuracy(y_acc, pred_acc)
            risk = float(1.0 - acc)
            cal_err = ece(probs_acc, y_acc)
        else:
            f1, acc, risk, cal_err = float("nan"), float("nan"), float("nan"), float("nan")

        err_auroc = calc_error_auroc(score_for_config[cfg_name])

        results[cfg_name] = {
            "macro_f1": f1,
            "accuracy": acc,
            "coverage": cov,
            "risk_at_coverage": risk,
            "auroc_error_detection": err_auroc,
            "auroc_ood_detection": ood_auroc_shift if "Adaptive" in cfg_name else float("nan"),
            "ece": cal_err,
            "n_accepted": n_acc,
            "n_total": n,
        }

    return results
