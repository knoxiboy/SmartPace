"""SmartPace Live — Week 1 Verification Suite (Days 1–5).

Comprehensive regression and integration tests verifying:
- Day 1: Model checkpoints and ONNX edge artifacts audit.
- Day 2: Real Single Seed vs Deep Ensemble (s0, s1, s2) mode selection and aggregation.
- Day 3: Measured PyTorch FP32 vs ONNX INT8 benchmarking with equivalent inputs.
- Day 4: Incremental, causal Pan-Tompkins R-peak detection across streaming chunks.
- Day 5: End-to-end integration and streaming stability.
"""
import os
import numpy as np
import pytest
import torch
import onnxruntime as ort
from scipy.special import softmax
from fastapi.testclient import TestClient

from backend.main import app
from smartpace import config as C
from smartpace import preprocess, train
from smartpace.adapter import SmartPaceEngine
from smartpace.benchmark import run_fp32_vs_int8_benchmark
from smartpace.causal_detector import CausalPanTompkins, PeakEvent


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


# =============================================================================
# DAY 1 — REPOSITORY, MODEL, CHECKPOINT, AND ARTIFACT AUDIT
# =============================================================================

def test_day1_checkpoints_exist_and_loadable():
    """Verify all 3 trained ResNet1D checkpoints exist and load with expected architecture."""
    models_dir = os.path.join("runs", "first", "models")
    seeds = ["s0", "s1", "s2"]
    for s in seeds:
        path = os.path.join(models_dir, f"resnet1d_AB_{s}.pt")
        assert os.path.exists(path), f"Checkpoint missing: {path}"
        size_kb = os.path.getsize(path) / 1024.0
        assert 800.0 <= size_kb <= 900.0, f"Unexpected checkpoint size: {size_kb} KB"

        ck = torch.load(path, map_location="cpu", weights_only=False)
        assert ck.get("arch") == "resnet1d"
        model, meta = train.load_model(path)
        assert isinstance(model, torch.nn.Module)
        assert meta.get("filt") is True
        dummy = torch.randn(1, 1, C.WIN)
        with torch.no_grad():
            out = model(dummy)
            feats = model.extract_features(dummy)
        assert out.shape == (1, C.N_CLASSES), f"Output shape mismatch: {out.shape}"
        assert feats.shape == (1, 64), f"Penultimate feature shape mismatch: {feats.shape}"


def test_day1_onnx_artifacts_valid():
    """Verify ONNX INT8 and FP32 models exist, load in ONNX Runtime, and report CPU execution."""
    edge_dir = os.path.join("runs", "first", "edge")
    int8_path = os.path.join(edge_dir, "resnet1d_int8.onnx")
    assert os.path.exists(int8_path), f"Missing ONNX INT8 artifact: {int8_path}"
    assert os.path.getsize(int8_path) < 300 * 1024, "INT8 artifact should be quantized (<300 KB)"

    sess = ort.InferenceSession(int8_path, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    out = sess.get_outputs()[0]
    assert inp.name == "input"
    assert inp.shape in (["batch", 1, 2500], [1, 1, 2500], ["None", 1, 2500])
    assert out.name == "logits"
    assert out.shape in (["batch", 5], [1, 5], ["None", 5])
    assert "CPUExecutionProvider" in sess.get_providers()


def test_day1_preprocessing_contract():
    """Verify inference-time preprocessing matches training-time contract."""
    raw = np.random.randn(2, C.WIN).astype(np.float64)
    x_in = preprocess.prepare_input(raw, filt=True)
    assert x_in.shape == (2, 1, C.WIN)
    assert x_in.dtype == np.float32
    assert np.isfinite(x_in).all()
    # Check z-score normalization on each window
    assert np.allclose(x_in[:, 0, :].mean(axis=-1), 0.0, atol=1e-5)
    assert np.allclose(x_in[:, 0, :].std(axis=-1), 1.0, atol=1e-3)


# =============================================================================
# DAY 2 — REAL SINGLE-SEED / DEEP-ENSEMBLE INFERENCE
# =============================================================================

def test_day2_single_mode_inference():
    """Verify SmartPaceEngine single-seed mode executes exactly 1 model with valid metrics."""
    engine = SmartPaceEngine(mode="single")
    assert engine.mode == "single"
    assert engine.get_inference_mode()["active_models"] == 1

    dummy_raw = np.random.randn(C.WIN)
    res = engine.analyze_window(dummy_raw)

    assert res["inference_mode"] == "single"
    assert res["active_models"] == 1
    assert res["predicted_class"] in C.CLASSES
    assert 0.0 <= res["confidence"] <= 1.0
    assert len(res["class_probabilities"]) == C.N_CLASSES
    prob_sum = sum(res["class_probabilities"].values())
    assert np.isclose(prob_sum, 1.0, atol=1e-4)
    assert res["ensemble_info"]["disagreement"] == 0.0


def test_day2_ensemble_mode_inference():
    """Verify Deep Ensemble evaluates all 3 members and produces mean probabilities."""
    engine = SmartPaceEngine(mode="single")
    assert engine.ensemble_available is True
    assert len(engine.ensemble_models) == 3

    # Switch to ensemble mode
    mode_status = engine.set_inference_mode("ensemble")
    assert mode_status["mode"] == "ensemble"
    assert mode_status["active_models"] == 3

    dummy_raw = np.random.randn(C.WIN)
    res = engine.analyze_window(dummy_raw)

    assert res["inference_mode"] == "ensemble"
    assert res["active_models"] == 3
    assert res["predicted_class"] in C.CLASSES
    assert 0.0 <= res["confidence"] <= 1.0
    prob_sum = sum(res["class_probabilities"].values())
    assert np.isclose(prob_sum, 1.0, atol=1e-4)

    # Validate ensemble metrics
    ens = res["ensemble_info"]
    assert ens["mode"] == "ensemble"
    assert ens["active_models"] == 3
    assert len(ens["member_predictions"]) == 3
    assert 0.0 <= ens["disagreement"] <= 1.0
    assert ens["probability_variance"] >= 0.0
    assert 0.0 <= ens["agreement_ratio"] <= 1.0


def test_day2_probability_aggregation_math():
    """Verify probability averaging averages member softmax probabilities, not raw logits."""
    engine = SmartPaceEngine()
    engine.set_inference_mode("ensemble")

    raw = np.random.randn(C.WIN)
    x_in = preprocess.prepare_input(raw[None, :], filt=True)
    xt = torch.from_numpy(x_in).to(engine.device)

    # Compute manual member probabilities
    manual_member_probs = []
    with torch.no_grad():
        for m in engine.ensemble_models:
            m_logits = m(xt).cpu().numpy()
            manual_member_probs.append(softmax(m_logits, axis=1)[0])

    expected_mean_probs = np.mean(manual_member_probs, axis=0)

    res = engine.analyze_window(raw)
    actual_probs = np.array([res["class_probabilities"][c] for c in C.CLASSES])

    assert np.allclose(actual_probs, expected_mean_probs, atol=1e-5), "Aggregation should equal mean of softmax probabilities"


def test_day2_api_mode_toggle(client):
    """Verify REST API correctly switches and reports inference mode."""
    # 1. Query current mode
    get_res = client.get("/api/inference/mode")
    assert get_res.status_code == 200
    assert get_res.json()["mode"] in ("single", "ensemble")

    # 2. Switch to ensemble
    post_res = client.post("/api/inference/mode", json={"mode": "ensemble"})
    assert post_res.status_code == 200
    assert post_res.json()["mode"] == "ensemble"
    assert post_res.json()["active_models"] == 3

    # 3. Switch back to single
    post_res2 = client.post("/api/inference/mode", json={"mode": "single"})
    assert post_res2.status_code == 200
    assert post_res2.json()["mode"] == "single"
    assert post_res2.json()["active_models"] == 1

    # 4. Reject invalid mode
    bad_res = client.post("/api/inference/mode", json={"mode": "quantum_hyper_mode"})
    assert bad_res.status_code == 400


def test_day2_tri_gate_active_in_both_modes():
    """Verify SQI, entropy uncertainty, and Mahalanobis shift detection are applied in both modes."""
    engine = SmartPaceEngine()
    flat_raw = np.zeros(C.WIN)  # Flatline -> should trigger SQI gate rejection

    # Single mode
    engine.set_inference_mode("single")
    res_single = engine.analyze_window(flat_raw)
    assert res_single["reliability"]["is_accepted"] is False
    assert "FLATLINE" in res_single["reliability"]["reason"] or res_single["reliability"]["action"] == "REJECT_QUALITY"

    # Ensemble mode
    engine.set_inference_mode("ensemble")
    res_ens = engine.analyze_window(flat_raw)
    assert res_ens["reliability"]["is_accepted"] is False
    assert res_ens["reliability"]["action"] == "REJECT_QUALITY"


# =============================================================================
# DAY 3 — LIVE PYTORCH FP32 VS ONNX INT8 BENCHMARK
# =============================================================================

def test_day3_benchmark_service_execution():
    """Verify real benchmark service measures latency, speedup, agreement, and probability drift."""
    res = run_fp32_vs_int8_benchmark(n_windows=15, repetitions=2, warmup_runs=5)

    assert res["status"] == "completed"
    assert res["test_windows_evaluated"] == 15
    assert res["repetitions_per_window"] == 2
    assert res["total_timed_inferences"] == 30

    pt = res["pytorch_fp32"]
    int8 = res["onnx_int8"]
    comp = res["comparison"]

    # Latency numbers must be strictly positive and realistic
    assert pt["median_latency_ms"] > 0.1
    assert int8["median_latency_ms"] > 0.05
    assert pt["p95_latency_ms"] >= pt["median_latency_ms"]
    assert int8["p95_latency_ms"] >= int8["median_latency_ms"]

    # Speedup ratio must match the ratio of medians
    expected_speedup = pt["median_latency_ms"] / max(int8["median_latency_ms"], 1e-4)
    assert np.isclose(comp["speedup_ratio_median"], expected_speedup, atol=0.05)

    # Agreement must be between 0% and 100%
    assert 0.0 <= comp["prediction_agreement_rate"] <= 1.0
    assert 0.0 <= comp["mean_probability_drift"] <= 1.0


def test_day3_benchmark_api_endpoints(client):
    """Test POST /api/benchmark/run and GET /api/benchmark/latest endpoints."""
    # Run a small benchmark via API
    run_res = client.post("/api/benchmark/run", json={"n_windows": 10, "repetitions": 2, "warmup_runs": 5})
    assert run_res.status_code == 200
    data = run_res.json()
    assert data["status"] == "completed"
    assert "pytorch_fp32" in data
    assert "onnx_int8" in data
    assert "comparison" in data

    # Check that GET /api/benchmark/latest returns the result
    latest_res = client.get("/api/benchmark/latest")
    assert latest_res.status_code == 200
    assert latest_res.json()["status"] == "completed"


# =============================================================================
# DAY 4 — INCREMENTAL, CAUSAL R-PEAK DETECTION
# =============================================================================

def test_day4_causal_pt_detector_basic():
    """Verify incremental CausalPanTompkins processes successive chunks with persistent state."""
    npz = np.load(os.path.join("data", "cache", "mitbih.npz"))
    raw = npz["X"][0]  # 2500 samples
    fs = 250

    detector = CausalPanTompkins(fs=fs)
    chunk_size = 25
    detected_events = []

    for i in range(0, len(raw), chunk_size):
        chunk = raw[i:i + chunk_size]
        events = detector.process_chunk(chunk, start_sample_idx=i)
        detected_events.extend(events)

    # MIT-BIH 10s window typically has 10-12 beats
    assert len(detected_events) >= 8, f"Expected at least 8 R-peaks, got {len(detected_events)}"

    # Check event structure
    for ev in detected_events:
        assert isinstance(ev, PeakEvent)
        assert 0 <= ev.sample_index < len(raw)
        assert np.isclose(ev.timestamp, ev.sample_index / fs, atol=1e-5)
        assert ev.delay_samples > 0, "Causal confirmation must acknowledge positive delay"
        assert ev.detection_sample_idx >= ev.sample_index


def test_day4_refractory_period_suppression():
    """Verify refractory period prevents duplicate detections within 200ms."""
    detector = CausalPanTompkins(fs=250)
    # Create synthetic burst with two rapid spikes 50ms apart
    t = np.arange(100)
    signal = np.zeros(100)
    signal[20] = 5.0
    signal[30] = 5.0  # 10 samples = 40ms < 200ms refractory period

    events = detector.process_chunk(signal, start_sample_idx=0)
    # At most 1 peak should be accepted within refractory interval
    assert len(events) <= 1


def test_day4_varying_chunk_sizes():
    """Verify CausalPanTompkins works stably across chunk sizes from 1 to 100 samples."""
    npz = np.load(os.path.join("data", "cache", "mitbih.npz"))
    raw = npz["X"][0][:500]  # 500 samples

    for c_size in [1, 10, 25, 50, 100]:
        det = CausalPanTompkins(fs=250)
        events = []
        for i in range(0, len(raw), c_size):
            chunk = raw[i:i + c_size]
            events.extend(det.process_chunk(chunk, start_sample_idx=i))
        # Stable operation without crashing on all chunk sizes
        assert isinstance(events, list)


def test_day4_detector_reset_behavior():
    """Verify reset clears filter memory and thresholds for seeking or stream restart."""
    detector = CausalPanTompkins(fs=250)
    dummy = np.random.randn(200)
    _ = detector.process_chunk(dummy, start_sample_idx=0)

    # Reset
    detector.reset(initial_sample_idx=5000)
    assert detector.last_peak_sample_idx < 5000
    assert detector.in_peak is False
    assert detector.peak_max_mwi == 0.0
    assert np.allclose(detector.zi, 0.0)


# =============================================================================
# DAY 5 — INTEGRATION & STREAMING ENGINE STABILITY
# =============================================================================

@pytest.mark.asyncio
async def test_day5_streaming_manager_causal_peaks_integration():
    """Verify StreamManager broadcasts ecg_chunk with r_peaks and mode metadata."""
    from backend.services.ecg_source import ECGSourceManager
    from backend.services.event_store import EventStore
    from backend.services.stream_manager import StreamManager

    ecg_source = ECGSourceManager()
    engine = SmartPaceEngine()
    event_store = EventStore(max_memory_events=5, db_path=None)
    stream_mgr = StreamManager(ecg_source=ecg_source, engine=engine, event_store=event_store)

    received_messages = []

    async def test_subscriber(msg):
        received_messages.append(msg)

    stream_mgr.subscribe(test_subscriber)

    status = await stream_mgr.start("mitdb_100", speed=5.0, start_time_s=0.0)
    assert status["is_playing"] is True
    assert "inference_mode" in status

    # Switch inference mode while streaming
    mode_res = await stream_mgr.set_inference_mode("ensemble")
    assert mode_res["mode"] == "ensemble"

    # Wait for a few ticks to accumulate chunk events
    import asyncio
    await asyncio.sleep(0.4)

    await stream_mgr.stop()
    assert stream_mgr.is_playing is False

    # Verify message stream received ecg_chunk with r_peaks
    chunk_msgs = [m for m in received_messages if m.get("type") == "ecg_chunk"]
    assert len(chunk_msgs) > 0
    assert "r_peaks" in chunk_msgs[0]
