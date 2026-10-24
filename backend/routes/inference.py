"""Inference & Event History REST Endpoints."""
import os
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from smartpace import config as C
from smartpace.models import ARCHS

router = APIRouter(tags=["Inference & Events"])


class WindowInferenceRequest(BaseModel):
    samples: List[float] = Field(..., description="2500 samples of lead-II ECG at 250 Hz")
    compute_explanation: bool = Field(False, description="Whether to compute Integrated Gradients attribution")
    use_edge_onnx: bool = Field(True, description="Whether to also run ONNX edge benchmark")


class ModelSelectRequest(BaseModel):
    model_name: str = Field(..., description="Checkpoint filename, e.g. 'resnet1d_AB_s0.pt' or 'cnn1d_AB_s0.pt'")


@router.get("/api/events")
async def get_recent_events(request: Request, limit: int = 50):
    """Returns recent streaming inference decisions."""
    event_store = request.app.state.event_store
    return {"events": event_store.get_recent_events(limit=limit)}


@router.get("/api/events/statistics")
async def get_event_statistics(request: Request):
    """Returns aggregated summary metrics of live inference events."""
    event_store = request.app.state.event_store
    return event_store.get_statistics()


@router.get("/api/events/{event_id}")
async def get_event_detail(event_id: str, request: Request):
    """Retrieves full details of a specific inference event."""
    event_store = request.app.state.event_store
    ev = event_store.get_event_by_id(event_id)
    if not ev:
        raise HTTPException(status_code=404, detail=f"Event '{event_id}' not found")
    return ev


@router.post("/api/inference/window")
async def analyze_single_window(req: WindowInferenceRequest, request: Request):
    """Runs the real SmartPace ML pipeline on an uploaded/specified 10-second ECG window."""
    if len(req.samples) != C.WIN:
        raise HTTPException(status_code=400, detail=f"Expected {C.WIN} samples (10s @ 250Hz), got {len(req.samples)}")

    engine = request.app.state.engine
    raw = np.array(req.samples, dtype=np.float64)
    res = engine.analyze_window(
        raw_window=raw,
        use_onnx_edge=req.use_edge_onnx,
        compute_explanation=req.compute_explanation,
    )
    return res


@router.get("/api/models")
async def list_models(request: Request):
    """Lists all available model checkpoints and edge ONNX variants."""
    models_dir = os.path.join("runs", "first", "models")
    edge_dir = os.path.join("runs", "first", "edge")

    checkpoints = []
    if os.path.exists(models_dir):
        for f in sorted(os.listdir(models_dir)):
            if f.endswith(".pt"):
                arch_key = f.split("_")[0]
                checkpoints.append({
                    "name": f,
                    "arch": arch_key,
                    "path": os.path.join(models_dir, f),
                    "size_kb": round(os.path.getsize(os.path.join(models_dir, f)) / 1024, 1),
                    "is_active": (f == os.path.basename(request.app.state.engine.model_path)),
                })

    edge_variants = []
    if os.path.exists(edge_dir):
        for f in sorted(os.listdir(edge_dir)):
            if f.endswith(".onnx"):
                edge_variants.append({
                    "name": f,
                    "path": os.path.join(edge_dir, f),
                    "size_kb": round(os.path.getsize(os.path.join(edge_dir, f)) / 1024, 1),
                })

    return {
        "active_model": os.path.basename(request.app.state.engine.model_path),
        "active_arch": request.app.state.engine.arch,
        "classes": C.CLASSES,
        "checkpoints": checkpoints,
        "edge_variants": edge_variants,
    }


@router.post("/api/models/select")
async def select_model(req: ModelSelectRequest, request: Request):
    """Switches active PyTorch model checkpoint."""
    models_dir = os.path.join("runs", "first", "models")
    target_path = os.path.join(models_dir, req.model_name)
    if not os.path.exists(target_path):
        raise HTTPException(status_code=404, detail=f"Model checkpoint '{req.model_name}' not found")

    try:
        from smartpace.adapter import SmartPaceEngine
        request.app.state.engine = SmartPaceEngine(model_path=target_path)
        request.app.state.stream_manager.engine = request.app.state.engine
        return {
            "status": "ok",
            "active_model": req.model_name,
            "message": f"Successfully loaded and calibrated {req.model_name}",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")
