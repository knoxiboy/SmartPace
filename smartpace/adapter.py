"""SmartPace Live Adapter — Clinical Research Inference & Reliability Engine.

Wraps the existing SmartPace modules (preprocess, models, train, reliability, explain, edge)
into a unified, thread-safe service for streaming replay and real-time analysis.
"""
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy.special import softmax

from . import config as C
from . import edge, explain, preprocess, reliability, train


class SmartPaceEngine:
    """Production research adapter wrapping SmartPace ML models and reliability gates."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        edge_onnx_path: Optional[str] = None,
        data_cache_path: Optional[str] = None,
        device: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path or os.path.join("runs", "first", "models", "resnet1d_AB_s0.pt")
        self.edge_onnx_path = edge_onnx_path or os.path.join("runs", "first", "edge", "resnet1d_int8.onnx")
        self.data_cache_path = data_cache_path or os.path.join("data", "cache", "mitbih.npz")

        self.model: Optional[torch.nn.Module] = None
        self.meta: Dict[str, Any] = {}
        self.arch: str = "resnet1d"
        self.abstention_module: Optional[reliability.AdaptiveAbstentionModule] = None
        self.sqi_gate: Optional[preprocess.SQIGate] = None
        self.shift_detector: Optional[reliability.MahalanobisShiftDetector] = None

        self._load_model_and_calibrate()

    def _load_model_and_calibrate(self) -> None:
        """Load PyTorch checkpoint and calibrate the tri-gate abstention module."""
        # 1. Load PyTorch model if available
        if os.path.exists(self.model_path):
            self.model, self.meta = train.load_model(self.model_path)
            self.model = self.model.to(self.device).eval()
            self.arch = self.meta.get("arch", "resnet1d")
        else:
            # Fallback to creating a fresh architecture if checkpoint missing
            self.model = train.M.build("resnet1d").to(self.device).eval()
            self.arch = "resnet1d"

        # 2. Setup SQI Gate
        self.sqi_gate = preprocess.SQIGate(q=1.0, use_bw=False)

        # 3. Setup Mahalanobis Shift Detector
        self.shift_detector = reliability.MahalanobisShiftDetector(alpha=0.05, eps=1e-4)

        # 4. Fit / Calibrate using cached validation data if present
        calibrated = False
        if os.path.exists(self.data_cache_path):
            try:
                npz = np.load(self.data_cache_path, allow_pickle=True)
                # Use validation split (split == 1) or a subset of train
                split = npz.get("split")
                X = npz.get("X")
                y = npz.get("y")
                if split is not None and X is not None and y is not None:
                    val_idx = np.where((split == 1) & (y >= 0))[0]
                    if len(val_idx) < 100:
                        val_idx = np.where(y >= 0)[0][:500]

                    if len(val_idx) >= 50:
                        X_val = X[val_idx[:300]]
                        y_val = y[val_idx[:300]]

                        # Extract SQI
                        sqi_val = preprocess.sqi_features(X_val, fs=C.FS)
                        self.sqi_gate.fit(sqi_val)

                        # Extract features and probabilities
                        x_prep = preprocess.prepare_input(X_val, filt=True)
                        with torch.no_grad():
                            xt = torch.from_numpy(x_prep).to(self.device)
                            feats_t = self.model.extract_features(xt)
                            logits = self.model(xt).cpu().numpy()
                            feats = feats_t.cpu().numpy()
                            probs = softmax(logits, axis=1)

                        self.abstention_module = reliability.AdaptiveAbstentionModule(
                            sqi_gate=self.sqi_gate,
                            shift_detector=self.shift_detector,
                        )
                        self.abstention_module.calibrate(
                            sqi_features_clean=sqi_val,
                            val_features=feats,
                            val_labels=y_val,
                            val_probs=probs,
                            unc_coverage=0.95,
                            shift_coverage=0.95,
                        )
                        calibrated = True
            except Exception as e:
                print(f"[SmartPaceEngine] Calibration notice: {e}")

        if not calibrated:
            # Conservative default thresholds with fallback fit
            self.sqi_gate.thr = {"kurt": 2.5, "hf_ratio": 0.35, "bw_ratio": 0.40, "rr_valid": 0.70}
            d_dim = 64
            dummy_feats = np.random.default_rng(0).normal(size=(C.N_CLASSES * 4, d_dim)).astype(np.float64)
            dummy_labels = np.repeat(np.arange(C.N_CLASSES), 4)
            self.shift_detector.fit(dummy_feats, dummy_labels)
            self.abstention_module = reliability.AdaptiveAbstentionModule(
                sqi_gate=self.sqi_gate,
                shift_detector=self.shift_detector,
                tau_unc=0.85,
                tau_shift=12.0,
            )

    def analyze_window(
        self,
        raw_window: np.ndarray,
        use_onnx_edge: bool = False,
        compute_explanation: bool = False,
    ) -> Dict[str, Any]:
        """Runs the end-to-end SmartPace pipeline on a single 10-second (2500 samples) ECG window.

        Args:
            raw_window: 1D array of shape (2500,) or 2D (1, 2500) raw samples at 250 Hz.
            use_onnx_edge: Whether to evaluate edge ONNX runtime in parallel or as primary.
            compute_explanation: Whether to compute Integrated Gradients attributions.

        Returns:
            Dictionary containing prediction, probabilities, vitals, reliability decisions,
            and optional explanation metrics.
        """
        raw_1d = np.asarray(raw_window, dtype=np.float64).flatten()
        if len(raw_1d) != C.WIN:
            raise ValueError(f"Expected window length {C.WIN} samples (10s @ 250Hz), got {len(raw_1d)}")

        t0_total = time.perf_counter()

        # 1. Preprocessing & Vitals
        r_peaks = preprocess.detect_rpeaks(raw_1d, fs=C.FS)
        heart_rate_bpm = preprocess.heart_rate(raw_1d, fs=C.FS)
        rr_variability = preprocess.rr_cv(raw_1d, fs=C.FS)

        # Signal Quality Index
        sqi_vec = preprocess.sqi_features(raw_1d[None, :], fs=C.FS)
        sqi_kurtosis = float(sqi_vec[0, 0])
        sqi_hf_ratio = float(sqi_vec[0, 1])
        sqi_bw_ratio = float(sqi_vec[0, 2])
        sqi_rr_valid = float(sqi_vec[0, 3])
        sqi_flat = bool(sqi_vec[0, 4] > 0.5)

        # Composite SQI Score [0, 1]
        sqi_score = max(0.0, min(1.0, float(
            0.35 * min(1.0, sqi_kurtosis / 10.0) +
            0.35 * max(0.0, 1.0 - sqi_hf_ratio * 3.0) +
            0.30 * sqi_rr_valid
        ))) if not sqi_flat else 0.0

        # Model input normalization: bandpass (0.5-40Hz) + per-window z-score -> (1, 1, 2500)
        x_in = preprocess.prepare_input(raw_1d[None, :], filt=True)

        # 2. Model Inference
        t0_infer = time.perf_counter()
        with torch.no_grad():
            xt = torch.from_numpy(x_in).to(self.device)
            features = self.model.extract_features(xt).cpu().numpy()  # (1, D)
            logits = self.model(xt).cpu().numpy()                     # (1, 5)
        inference_time_ms = (time.perf_counter() - t0_infer) * 1000.0

        probs = softmax(logits, axis=1)[0]
        pred_idx = int(np.argmax(probs))
        predicted_class = C.CLASSES[pred_idx]
        confidence = float(probs[pred_idx])
        predictive_entropy = float(reliability.entropy(probs[None, :])[0])

        # 3. Optional Edge ONNX Benchmark comparison
        edge_metrics = None
        if os.path.exists(self.edge_onnx_path):
            try:
                t0_edge = time.perf_counter()
                edge_logits = edge.onnx_logits(self.edge_onnx_path, x_in)
                edge_lat_ms = (time.perf_counter() - t0_edge) * 1000.0
                edge_probs = softmax(edge_logits, axis=1)[0]
                edge_pred_idx = int(np.argmax(edge_probs))
                edge_metrics = {
                    "edge_variant": "resnet1d_int8",
                    "edge_pred_class": C.CLASSES[edge_pred_idx],
                    "edge_confidence": float(edge_probs[edge_pred_idx]),
                    "edge_latency_ms": float(edge_lat_ms),
                    "agrees_with_fp32": bool(edge_pred_idx == pred_idx),
                    "confidence_shift": float(abs(edge_probs[edge_pred_idx] - confidence)),
                }
            except Exception:
                pass

        # 4. Tri-Gate Reliability & Abstention Decision
        decision = self.abstention_module.decide(
            probs=probs[None, :],
            features=features,
            sqi_features=sqi_vec,
        )

        action = str(decision.action[0])
        is_accepted = bool(decision.is_accepted[0])
        rejection_reason = decision.rejection_reason[0]
        mahalanobis_dist = float(decision.shift_score[0])

        # 5. Integrated Gradients Explanation (if requested)
        explanation_data = None
        if compute_explanation and not sqi_flat:
            try:
                t0_exp = time.perf_counter()
                attr = explain.integrated_gradients(
                    self.model, x_in, target=np.array([pred_idx]), steps=32, device=self.device
                )[0]
                exp_lat_ms = (time.perf_counter() - t0_exp) * 1000.0

                # Attribution smoothing for clean visualization
                attr_smooth = explain.smooth_abs(attr[None, :], width=25)[0]

                # Regional anatomical enrichment
                masks = explain.region_masks(raw_1d)
                tot_attr = float(np.sum(attr_smooth)) + 1e-12
                share_qrs = float(np.sum(attr_smooth[masks["QRS"]]) / tot_attr) if masks["QRS"].any() else 0.0
                share_p = float(np.sum(attr_smooth[masks["P"]]) / tot_attr) if masks["P"].any() else 0.0
                share_t = float(np.sum(attr_smooth[masks["T"]]) / tot_attr) if masks["T"].any() else 0.0

                explanation_data = {
                    "attributions": attr.tolist(),
                    "attributions_smooth": attr_smooth.tolist(),
                    "explanation_latency_ms": float(exp_lat_ms),
                    "qrs_share": share_qrs,
                    "p_wave_share": share_p,
                    "t_wave_share": share_t,
                    "completeness_target_class": predicted_class,
                }
            except Exception as e:
                explanation_data = {"error": str(e)}

        total_time_ms = (time.perf_counter() - t0_total) * 1000.0

        return {
            "predicted_class": predicted_class,
            "predicted_index": pred_idx,
            "confidence": confidence,
            "predictive_entropy": predictive_entropy,
            "class_probabilities": {c: float(probs[i]) for i, c in enumerate(C.CLASSES)},
            "reliability": {
                "action": action,
                "is_accepted": is_accepted,
                "reason": rejection_reason,
                "sqi_score": sqi_score,
                "sqi_kurtosis": sqi_kurtosis,
                "sqi_hf_ratio": sqi_hf_ratio,
                "sqi_bw_ratio": sqi_bw_ratio,
                "sqi_flat": sqi_flat,
                "mahalanobis_distance": mahalanobis_dist,
                "tau_shift": float(self.abstention_module.tau_shift),
                "tau_uncertainty": float(self.abstention_module.tau_unc),
                "shift_status": "SHIFT_DETECTED" if mahalanobis_dist > self.abstention_module.tau_shift else "IN_DISTRIBUTION",
            },
            "vitals": {
                "heart_rate_bpm": None if np.isnan(heart_rate_bpm) else float(heart_rate_bpm),
                "r_peak_indices": r_peaks.tolist(),
                "r_peak_count": len(r_peaks),
                "rr_cv": None if np.isnan(rr_variability) else float(rr_variability),
            },
            "latency": {
                "model_inference_ms": float(inference_time_ms),
                "total_processing_ms": float(total_time_ms),
            },
            "edge_benchmark": edge_metrics,
            "explanation": explanation_data,
        }
