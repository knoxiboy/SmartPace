"""Automated Tests for SmartPace Live Backend, Streaming Engine & ML Adapter."""
import asyncio
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from smartpace import config as C
from smartpace.adapter import SmartPaceEngine
from backend.services.ecg_source import ECGSourceManager
from backend.services.event_store import EventStore
from backend.services.stream_manager import StreamManager


@pytest.fixture(scope="module")
def client():
    """Initializes FastAPI test client with lifespan context."""
    with TestClient(app) as c:
        yield c


def test_api_health_endpoint(client):
    """Test health check endpoint."""
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["service"] == "SmartPace Live Backend"


def test_list_records_endpoint(client):
    """Test listing available authentic clinical ECG records."""
    res = client.get("/api/records")
    assert res.status_code == 200
    data = res.json()
    assert "records" in data
    assert len(data["records"]) > 0
    # MIT-BIH 100 should be present
    record_ids = [r["id"] for r in data["records"]]
    assert "mitdb_100" in record_ids


def test_ecg_source_manager_loads_real_record():
    """Test that ECGSourceManager loads physical MIT-BIH samples with proper resampling to 250 Hz."""
    mgr = ECGSourceManager()
    rec = mgr.load_record("mitdb_100")
    assert rec["fs"] == C.FS
    assert rec["total_samples"] > 10000
    assert len(rec["samples"]) == rec["total_samples"]
    # Check that it's physical float data, not constant or NaN
    assert np.isfinite(rec["samples"]).all()
    assert rec["samples"].std() > 0.05


def test_smartpace_engine_window_analysis():
    """Test that SmartPaceEngine executes the real pipeline with SQI, R-peaks, and Tri-Gate."""
    engine = SmartPaceEngine()
    # Create clean synthetic baseline window
    t = np.linspace(0, 10, C.WIN)
    clean_ecg = np.sin(2 * np.pi * 1.2 * t) + 0.5 * np.cos(2 * np.pi * 2.4 * t)
    res = engine.analyze_window(clean_ecg, compute_explanation=True)

    assert "predicted_class" in res
    assert res["predicted_class"] in C.CLASSES
    assert 0.0 <= res["confidence"] <= 1.0
    assert "reliability" in res
    assert res["reliability"]["action"] in ["ACCEPT", "REJECT_QUALITY", "ABSTAIN_UNCERTAIN", "ABSTAIN_UNFAMILIAR"]
    assert "vitals" in res
    assert "latency" in res
    assert res["latency"]["model_inference_ms"] > 0
    assert "explanation" in res
    if res["explanation"]:
        assert "attributions" in res["explanation"]
        assert len(res["explanation"]["attributions"]) == C.WIN


def test_event_store_lifecycle():
    """Test that EventStore records, indexes, and computes statistics over events."""
    store = EventStore(max_memory_events=10, db_path=None)
    store.add_event({
        "event_id": "ev_test_1",
        "predicted_class": "NSR",
        "confidence": 0.95,
        "reliability": {"is_accepted": True, "action": "ACCEPT"},
        "latency": {"total_processing_ms": 3.2},
    })
    store.add_event({
        "event_id": "ev_test_2",
        "predicted_class": "AF",
        "confidence": 0.65,
        "reliability": {"is_accepted": False, "action": "ABSTAIN_UNCERTAIN"},
        "latency": {"total_processing_ms": 4.1},
    })

    recent = store.get_recent_events(limit=5)
    assert len(recent) == 2
    assert recent[0]["event_id"] == "ev_test_2"  # Reverse chronological

    stats = store.get_statistics()
    assert stats["total_events"] == 2
    assert stats["accepted_count"] == 1
    assert stats["abstained_count"] == 1
    assert stats["acceptance_rate"] == 0.5


def test_models_endpoint(client):
    """Test model registry endpoint."""
    res = client.get("/api/models")
    assert res.status_code == 200
    data = res.json()
    assert "checkpoints" in data
    assert len(data["checkpoints"]) > 0
    assert "active_model" in data


def test_system_info_endpoint(client):
    """Test full system transparency disclosure endpoint."""
    res = client.get("/api/system/info")
    assert res.status_code == 200
    data = res.json()
    assert "disclaimer" in data["project"]
    assert "model_architecture" in data
    assert "reliability_framework" in data
    assert "edge_deployment" in data


def test_evaluation_endpoints(client):
    """Test that research evaluation endpoints correctly serve experiment results."""
    res_overview = client.get("/api/evaluation/overview")
    assert res_overview.status_code == 200
    data_ov = res_overview.json()
    assert len(data_ov["available_experiments"]) == 6

    res_exp4 = client.get("/api/evaluation/exp4")
    assert res_exp4.status_code == 200
    data_exp4 = res_exp4.json()
    assert "noise_performance" in data_exp4
    assert "tradeoff_curves" in data_exp4

    res_exp5 = client.get("/api/evaluation/exp5")
    assert res_exp5.status_code == 200
    data_exp5 = res_exp5.json()
    assert "edge_benchmark" in data_exp5
    assert len(data_exp5["edge_benchmark"]) > 0


def test_noise_stress_corruption_api(client):
    """Test that noise stress corrupts input without modifying original data."""
    dummy_clean = np.sin(np.linspace(0, 10, C.WIN)).tolist()
    res = client.post("/api/stress/corrupt", json={
        "samples": dummy_clean,
        "noise_type": "ma",
        "snr_db": 0.0,
    })
    assert res.status_code == 200
    data = res.json()
    assert "corrupted_samples" in data
    assert len(data["corrupted_samples"]) == C.WIN
    # Corrupted should differ from clean
    assert data["corrupted_samples"] != dummy_clean
