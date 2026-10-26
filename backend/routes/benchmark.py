"""Live PyTorch FP32 vs ONNX INT8 Benchmark Endpoint."""
import asyncio
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from smartpace.benchmark import run_fp32_vs_int8_benchmark

router = APIRouter(tags=["Live Benchmark"])

_latest_benchmark_result: Optional[Dict[str, Any]] = None


class BenchmarkRunRequest(BaseModel):
    n_windows: int = Field(50, ge=5, le=200, description="Number of ECG windows to benchmark")
    repetitions: int = Field(3, ge=1, le=10, description="Timing repetitions per window")
    warmup_runs: int = Field(10, ge=1, le=30, description="Warm-up runs excluded from timing")


@router.post("/api/benchmark/run")
async def run_benchmark(req: BenchmarkRunRequest):
    """Executes live measured PyTorch FP32 vs ONNX INT8 benchmark off the event loop."""
    global _latest_benchmark_result
    try:
        # Offload CPU-intensive benchmark to worker thread to keep WebSockets responsive
        result = await asyncio.to_thread(
            run_fp32_vs_int8_benchmark,
            n_windows=req.n_windows,
            repetitions=req.repetitions,
            warmup_runs=req.warmup_runs,
        )
        _latest_benchmark_result = result
        return result
    except FileNotFoundError as fnf:
        raise HTTPException(status_code=404, detail=str(fnf))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Benchmark execution failed: {e}")


@router.get("/api/benchmark/latest")
async def get_latest_benchmark():
    """Returns the most recent measured benchmark result."""
    global _latest_benchmark_result
    if _latest_benchmark_result is not None:
        return _latest_benchmark_result

    # Run initial baseline benchmark if none exists
    try:
        res = await asyncio.to_thread(
            run_fp32_vs_int8_benchmark,
            n_windows=20,
            repetitions=2,
            warmup_runs=5,
        )
        _latest_benchmark_result = res
        return res
    except Exception as e:
        return {"status": "unavailable", "message": str(e)}
