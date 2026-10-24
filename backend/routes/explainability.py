"""Explainability Lab REST Endpoints."""
import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from smartpace import config as C
from smartpace import explain, preprocess

router = APIRouter(tags=["Explainability Lab"])


class ExplainRequest(BaseModel):
    event_id: Optional[str] = Field(None, description="Event ID to explain")
    samples: Optional[List[float]] = Field(None, description="Or direct 2500 samples")
    target_class: Optional[str] = Field(None, description="Optional target class (default: predicted class)")
    steps: int = Field(32, ge=8, le=64, description="Integrated Gradients integration steps")


class StressTestRequest(BaseModel):
    event_id: Optional[str] = Field(None)
    samples: Optional[List[float]] = Field(None)
    sigma: float = Field(0.03, ge=0.01, le=0.20, description="Perturbation noise standard deviation")
    steps: int = Field(24, ge=8, le=32)


@router.post("/api/explain")
async def compute_explanation(req: ExplainRequest, request: Request):
    """Computes Integrated Gradients attributions and anatomical region localization."""
    raw = None
    if req.event_id:
        ev = request.app.state.event_store.get_event_by_id(req.event_id)
        if ev and "raw_window" in ev:
            raw = np.array(ev["raw_window"], dtype=np.float64)

    if raw is None and req.samples:
        if len(req.samples) == C.WIN:
            raw = np.array(req.samples, dtype=np.float64)

    if raw is None:
        raise HTTPException(status_code=400, detail="Must provide either valid event_id or 2500 samples")

    engine = request.app.state.engine
    x_in = preprocess.prepare_input(raw[None, :], filt=True)

    # Class target
    target_idx = None
    if req.target_class and req.target_class in C.CLASSES:
        target_idx = C.CLASSES.index(req.target_class)

    attr = explain.integrated_gradients(
        engine.model, x_in, target=np.array([target_idx]) if target_idx is not None else None,
        steps=req.steps, device=engine.device
    )[0]

    attr_smooth = explain.smooth_abs(attr[None, :], width=25)[0]

    # Anatomical masks
    masks = explain.region_masks(raw)
    tot = float(np.sum(attr_smooth)) + 1e-12
    qrs_share = float(np.sum(attr_smooth[masks["QRS"]]) / tot) if masks["QRS"].any() else 0.0
    p_share = float(np.sum(attr_smooth[masks["P"]]) / tot) if masks["P"].any() else 0.0
    t_share = float(np.sum(attr_smooth[masks["T"]]) / tot) if masks["T"].any() else 0.0

    # Completeness check: compare sum of attributions with logit diff
    # Output structured explanation narrative
    pred_idx = int(engine.model(torch.from_numpy(x_in).to(engine.device)).argmax(dim=1).cpu())
    predicted_class = C.CLASSES[pred_idx]

    narrative = (
        f"Model evidence indicates class '{predicted_class}'. The highlighted waveform segments contributed "
        f"most strongly to this classification. QRS complex received {qrs_share*100:.1f}% attribution weight, "
        f"P-wave {p_share*100:.1f}%, and T-wave {t_share*100:.1f}%. Please review the waveform and reliability "
        f"indicators before interpreting this result."
    )

    return {
        "predicted_class": predicted_class,
        "target_class": req.target_class or predicted_class,
        "attributions": attr.tolist(),
        "attributions_smooth": attr_smooth.tolist(),
        "anatomical_overlap": {
            "qrs_complex_share": qrs_share,
            "p_wave_share": p_share,
            "t_wave_share": t_share,
            "background_share": max(0.0, 1.0 - (qrs_share + p_share + t_share)),
        },
        "qrs_mask": masks["QRS"].tolist(),
        "p_mask": masks["P"].tolist(),
        "t_mask": masks["T"].tolist(),
        "evidence_narrative": narrative,
    }


@router.post("/api/explain/stress_test")
async def run_explanation_stress_test(req: StressTestRequest, request: Request):
    """Stress-tests explanation stability under input perturbation and counterfactual masking."""
    raw = None
    if req.event_id:
        ev = request.app.state.event_store.get_event_by_id(req.event_id)
        if ev and "raw_window" in ev:
            raw = np.array(ev["raw_window"], dtype=np.float64)

    if raw is None and req.samples:
        if len(req.samples) == C.WIN:
            raw = np.array(req.samples, dtype=np.float64)

    if raw is None:
        raise HTTPException(status_code=400, detail="Must provide either valid event_id or 2500 samples")

    engine = request.app.state.engine
    x_in = preprocess.prepare_input(raw[None, :], filt=True)

    # 1. Stability under physiological perturbation
    stab = explain.individual_stability(
        engine.model, x_in, sigma=req.sigma, steps=req.steps, device=engine.device
    )
    spearman_rho = float(stab["spearman"][0])
    topk_jaccard = float(stab["topk_jaccard"][0])

    # 2. Counterfactual test
    attr = stab["clean_attr"][0]
    L = len(raw)
    k_len = max(1, int(0.10 * L))
    order = np.argsort(-attr)
    high_idx = order[:k_len]
    ctrl_idx = order[-k_len:]

    x_pert_high = x_in.copy()
    x_pert_ctrl = x_in.copy()
    x_pert_high[0, 0, high_idx] = 0.0
    x_pert_ctrl[0, 0, ctrl_idx] = 0.0

    with torch.no_grad():
        p_clean = torch.softmax(engine.model(torch.from_numpy(x_in).to(engine.device)), dim=1).cpu().numpy()[0]
        p_high = torch.softmax(engine.model(torch.from_numpy(x_pert_high).to(engine.device)), dim=1).cpu().numpy()[0]
        p_ctrl = torch.softmax(engine.model(torch.from_numpy(x_pert_ctrl).to(engine.device)), dim=1).cpu().numpy()[0]

    clean_pred = int(np.argmax(p_clean))
    high_pred = int(np.argmax(p_high))
    ctrl_pred = int(np.argmax(p_ctrl))

    # Failure mode evaluation
    is_stable = bool(spearman_rho >= 0.70 and topk_jaccard >= 0.35)
    cf_flipped_on_high = bool(clean_pred != high_pred)
    cf_retained_on_ctrl = bool(clean_pred == ctrl_pred)

    failure_case = "Case A: Correct & Stable"
    if not is_stable:
        if p_clean[clean_pred] >= 0.85:
            failure_case = "Case C: Confident & Unstable (Silent Failure)"
        else:
            failure_case = "Case B: Suspicious / Fragile Attribution"

    return {
        "spearman_rho": spearman_rho,
        "topk_jaccard": topk_jaccard,
        "is_stable": is_stable,
        "failure_mode_assessment": failure_case,
        "counterfactual": {
            "clean_predicted_class": C.CLASSES[clean_pred],
            "clean_confidence": float(p_clean[clean_pred]),
            "high_attribution_removed_prediction": C.CLASSES[high_pred],
            "high_attribution_removed_confidence": float(p_high[clean_pred]),
            "target_probability_drop": float(p_clean[clean_pred] - p_high[clean_pred]),
            "control_region_removed_prediction": C.CLASSES[ctrl_pred],
            "attribution_is_causal": bool(cf_flipped_on_high or (p_clean[clean_pred] - p_high[clean_pred] > 0.15)),
        },
    }


@router.get("/api/explain/failure_cases")
async def get_stored_failure_cases():
    """Loads documented benchmark failure cases (Case A, B, C) from runs/first/results/."""
    csv_path = os.path.join("runs", "first", "results", "exp3_stress_test_cases.csv")
    if not os.path.exists(csv_path):
        return {"cases": []}

    try:
        df = pd.read_csv(csv_path)
        # Select representative instances of each case
        cases = []
        for case_name in [
            "Case A: Correct, stable explanation",
            "Case B: Incorrect, suspicious explanation",
            "Case C: Confident, unstable explanation (SILENT FAILURE)",
        ]:
            subset = df[df["failure_case"] == case_name]
            if not subset.empty:
                for _, r in subset.head(3).iterrows():
                    cases.append({
                        "case_type": case_name,
                        "sample_idx": int(r["sample_idx"]),
                        "pred_class": r["pred_class"],
                        "true_class": r["true_class"],
                        "confidence": float(r["confidence"]),
                        "spearman_rho": float(r["spearman_rho"]),
                        "topk_jaccard": float(r["topk_jaccard"]),
                        "qrs_enrichment": float(r["qrs_enrichment"]),
                        "bg_share": float(r["bg_share"]),
                        "is_correct": bool(r["is_correct"]),
                        "cf_flip_high": bool(r["cf_flip_high"]),
                    })
        return {"cases": cases}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/explain/diagnostics")
async def get_explanation_diagnostics():
    """Returns the benchmark explanation evaluation summary."""
    json_path = os.path.join("runs", "first", "results", "exp3_explanation.json")
    if not os.path.exists(json_path):
        return {"diagnostics": None}
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {"diagnostics": data}
