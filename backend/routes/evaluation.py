"""Research Evaluation REST Endpoints.

Serves all empirically measured benchmark datasets and research artifacts from runs/first/results/.
Never invents or fabricates benchmark numbers.
"""
import json
import os
from typing import Any, Dict, List

import pandas as pd
from fastapi import APIRouter, HTTPException

router = APIRouter(tags=["Research Evaluation"])

RESULTS_DIR = os.path.join("runs", "first", "results")


import math

def _clean_val(v):
    if v is None:
        return None
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return v


def _read_csv_safe(filename: str) -> List[Dict[str, Any]]:
    path = os.path.join(RESULTS_DIR, filename)
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
            records = df.to_dict(orient="records")
            return [{k: _clean_val(v) for k, v in row.items()} for row in records]
        except Exception:
            return []
    return []


@router.get("/api/evaluation/overview")
async def get_evaluation_overview():
    """Summary of all 6 research evaluation experiments and their artifact status."""
    info_path = os.path.join(RESULTS_DIR, "run_info.json")
    info = {}
    if os.path.exists(info_path):
        with open(info_path, "r", encoding="utf-8") as f:
            info = json.load(f)

    return {
        "run_info": info,
        "available_experiments": [
            {"id": "exp1", "name": "Detection, Uncertainty & Selective Rejection", "ready": os.path.exists(os.path.join(RESULTS_DIR, "exp1_summary.csv"))},
            {"id": "exp2", "name": "Cross-Dataset Generalization & Adaptation", "ready": os.path.exists(os.path.join(RESULTS_DIR, "exp2_summary.csv"))},
            {"id": "exp3", "name": "Calibration & Explanation Stress-Testing", "ready": os.path.exists(os.path.join(RESULTS_DIR, "exp3_calibration.csv"))},
            {"id": "exp4", "name": "Reliability-Aware Noise Rejection (5 Policies)", "ready": os.path.exists(os.path.join(RESULTS_DIR, "exp4_noise.csv"))},
            {"id": "exp5", "name": "Edge Quantization, Pruning & Confidence Distortion", "ready": os.path.exists(os.path.join(RESULTS_DIR, "exp5_edge.csv"))},
            {"id": "exp6", "name": "Closed-Loop Alarm Fatigue Decision Simulator", "ready": os.path.exists(os.path.join(RESULTS_DIR, "exp6_simulator.csv"))},
        ],
    }


@router.get("/api/evaluation/exp1")
async def get_exp1_results():
    """Experiment 1: Uncertainty estimation, Mahalanobis detector, and selective prediction."""
    return {
        "summary": _read_csv_safe("exp1_summary.csv"),
        "rejection_curves": _read_csv_safe("exp1_rejection_curves.csv"),
        "shift_detector_comparison": _read_csv_safe("exp1_shift_detector_comparison.csv"),
        "reliability_comparison": _read_csv_safe("exp1_reliability_comparison.csv"),
    }


@router.get("/api/evaluation/exp2")
async def get_exp2_results():
    """Experiment 2: Cross-dataset generalisation, few-shot linear probing, and selective cross-rejection."""
    return {
        "summary": _read_csv_safe("exp2_summary.csv"),
        "adaptation": _read_csv_safe("exp2_adaptation.csv"),
        "sampling_control": _read_csv_safe("exp2_sampling_control.csv"),
        "cross_rejection": _read_csv_safe("exp2_cross_rejection.csv"),
    }


@router.get("/api/evaluation/exp3")
async def get_exp3_results():
    """Experiment 3: Calibration, Integrated Gradients completeness, and failure mode AUROC."""
    exp_json = os.path.join(RESULTS_DIR, "exp3_explanation.json")
    diagnostics = json.load(open(exp_json, "r", encoding="utf-8")) if os.path.exists(exp_json) else {}
    return {
        "calibration": _read_csv_safe("exp3_calibration.csv"),
        "error_detection_auroc": _read_csv_safe("exp3_error_detection_auroc.csv"),
        "plausibility": _read_csv_safe("exp3_plausibility.csv"),
        "diagnostics": diagnostics,
    }


@router.get("/api/evaluation/exp4")
async def get_exp4_results():
    """Experiment 4: Noise robustness across SNR levels, 5-policy risk-coverage curves, and cross-noise transfer."""
    return {
        "noise_performance": _read_csv_safe("exp4_noise.csv"),
        "tradeoff_curves": _read_csv_safe("exp4_tradeoff_curves.csv"),
        "noise_type_rejection": _read_csv_safe("exp4_noise_type_rejection.csv"),
        "cross_noise_transfer": _read_csv_safe("exp4_cross_noise_transfer.csv"),
    }


@router.get("/api/evaluation/exp5")
async def get_exp5_results():
    """Experiment 5: INT8 quantization and pruning edge benchmark with confidence distortion."""
    return {
        "edge_benchmark": _read_csv_safe("exp5_edge.csv"),
        "confidence_distortion": _read_csv_safe("exp5_confidence_distortion.csv"),
    }


@router.get("/api/evaluation/exp6")
async def get_exp6_results():
    """Experiment 6: Closed-loop clinical replay simulation and alarm fatigue reduction."""
    return {
        "simulator_policies": _read_csv_safe("exp6_simulator.csv"),
    }


@router.get("/api/evaluation/hardware")
async def get_hardware_info():
    """Hardware specifications and inference runtime environment."""
    hw_path = os.path.join(RESULTS_DIR, "hardware.json")
    if os.path.exists(hw_path):
        with open(hw_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"status": "Hardware metadata file not found"}
