"""SmartPace Live Backend Entry Point.

FastAPI application delivering real-time streaming ECG replay,
tri-gate reliability analysis, Integrated Gradients explanations,
and research evaluation benchmarks.
"""
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from smartpace.adapter import SmartPaceEngine
from .services.ecg_source import ECGSourceManager
from .services.event_store import EventStore
from .services.stream_manager import StreamManager

from .routes import stream, inference, explainability, stress_test, evaluation, system_info, benchmark


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes models and stream managers on startup, cleans up on shutdown."""
    print("[SmartPace Live] Initializing ML Engine and Data Sources...", flush=True)
    engine = SmartPaceEngine()
    ecg_source = ECGSourceManager()
    event_store = EventStore()
    stream_manager = StreamManager(ecg_source=ecg_source, engine=engine, event_store=event_store)

    app.state.engine = engine
    app.state.ecg_source = ecg_source
    app.state.event_store = event_store
    app.state.stream_manager = stream_manager

    print("[SmartPace Live] Engine ready! Model:", engine.arch, "Device:", engine.device, flush=True)
    yield
    print("[SmartPace Live] Shutting down stream manager...", flush=True)
    await stream_manager.stop()


app = FastAPI(
    title="SmartPace Live — Research API",
    description="Real-Time ECG Monitoring, Reliability AI & Explainability Platform",
    version="2.0.0-research",
    lifespan=lifespan,
)

# Enable CORS for local Vite dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API Routers
app.include_router(stream.router)
app.include_router(inference.router)
app.include_router(benchmark.router)
app.include_router(explainability.router)
app.include_router(stress_test.router)
app.include_router(evaluation.router)
app.include_router(system_info.router)


@app.get("/api/health")
async def health_check():
    """Health check endpoint confirming engine and dataset availability."""
    return {
        "status": "healthy",
        "service": "SmartPace Live Backend",
        "active_model": os.path.basename(app.state.engine.model_path) if hasattr(app.state, "engine") else None,
        "streaming_active": app.state.stream_manager.is_playing if hasattr(app.state, "stream_manager") else False,
    }


# Mount built static frontend if present
frontend_dist = os.path.join(os.path.dirname(__file__), "..", "frontend", "dist")
if os.path.exists(frontend_dist):
    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
