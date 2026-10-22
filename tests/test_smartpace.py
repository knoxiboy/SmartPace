import numpy as np
import pytest
import torch

from smartpace import config as C
from smartpace import evaluation as E
from smartpace import simulator as S
from smartpace.datasets import label_mitbih_window, label_ptbxl
from smartpace.explain import completeness_gap, integrated_gradients
from smartpace.models import CNN1D, CNNTransformer, ResNet1D, n_params
from smartpace.noise import NoiseBank, measured_snr_db
from smartpace.preprocess import detect_rpeaks, heart_rate, prepare_input
from smartpace.synthetic import build_synthetic_dataset, make_window, _subject


def test_ptbxl_label_priority():
    assert label_ptbxl({"AFIB": 100, "PVC": 0}) == C.AF
    assert label_ptbxl({"PVC": 0, "SBRAD": 0}) == C.PVC
    assert label_ptbxl({"SBRAD": 0, "SR": 0}) == C.SBRAD
    assert label_ptbxl({"STACH": 0}) == C.STACH
    assert label_ptbxl({"NORM": 100, "SR": 0}) == C.NSR
    assert label_ptbxl({"IMI": 100}) == C.OTHER


def _lab(s, e, intervals, beats, syms):
    return label_mitbih_window(s, e, intervals, np.array(beats), np.array(syms))


def test_mitbih_window_labels():
    beats_n = list(range(100, 3600, 360))          # 60 bpm at 360 Hz -> 1.0 s apart
    n = ["N"] * len(beats_n)
    ivs = [(0, 7200, "(N")]
    assert _lab(0, 3600, ivs, beats_n, n) == C.NSR
    fast = list(range(0, 3600, 200))               # 108 bpm
    assert _lab(0, 3600, ivs, fast, ["N"] * len(fast)) == C.STACH
    slow = list(range(0, 3600, 480))               # 45 bpm
    assert _lab(0, 3600, ivs, slow, ["N"] * len(slow)) == C.SBRAD
    sy = ["N"] * len(beats_n); sy[3] = "V"
    assert _lab(0, 3600, ivs, beats_n, sy) == C.PVC
    assert _lab(0, 3600, [(0, 7200, "(AFIB")], beats_n, n) == C.AF
    assert _lab(0, 3600, [(0, 7200, "(VT")], beats_n, n) == C.OTHER
    assert _lab(0, 3600, ivs, beats_n, ["/"] * len(beats_n)) == C.OTHER
    assert _lab(0, 3600, ivs, [100, 200], ["N", "N"]) == C.OTHER      # too few beats


def test_noise_snr_exact():
    ds = build_synthetic_dataset()
    X = ds["X"][:40]
    nb = NoiseBank.synthetic()
    for k in C.NOISE_KINDS:
        for snr in (18, 6, -6):
            n = nb.corrupt(X, k, snr, np.random.default_rng(0))
            assert abs(measured_snr_db(X, n).mean() - snr) < 0.05


def test_noise_train_test_parts_disjoint():
    nb = NoiseBank.synthetic()
    lo_tr, hi_tr = nb._span("bw", "train")
    lo_te, hi_te = nb._span("bw", "test")
    assert hi_tr <= lo_te


def test_rpeak_heart_rate():
    rng = np.random.default_rng(0)
    sub = _subject(rng, "A")
    w = make_window(rng, C.NSR, sub)
    assert 55 < heart_rate(w) < 105
    assert len(detect_rpeaks(np.zeros(C.WIN))) == 0


def test_ece_and_temperature():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 5, 3000)
    logits = np.eye(5)[y] * 2 + rng.normal(0, 1.2, (3000, 5))
    over = logits * 4                                   # overconfident
    T = E.fit_temperature(over, y)
    assert T > 1.5
    assert E.ece(E.apply_temperature(over, T), y) < E.ece(E.apply_temperature(over, 1.0), y)
    perfect = np.eye(5)[y] * 0.999 + 0.001 / 5
    assert E.ece(perfect / perfect.sum(1, keepdims=True), y) < 0.01


def test_rejection_curve_monotone_for_informative_uncertainty():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 5, 2000)
    wrong = rng.random(2000) < 0.3
    pred = np.where(wrong, (y + 1) % 5, y)
    unc = wrong + rng.random(2000) * 0.5
    acc = [r["accuracy"] for r in E.rejection_curve(y, pred, unc)]
    assert acc[-1] > acc[0]


def test_models_forward_and_size():
    x = torch.randn(2, 1, C.WIN)
    for M in (ResNet1D, CNN1D, CNNTransformer):
        assert M()(x).shape == (2, C.N_CLASSES)
    assert 1e5 < n_params(ResNet1D()) < 5e5            # "lightweight": a few hundred thousand params


def test_integrated_gradients_completeness():
    torch.manual_seed(0)
    m = ResNet1D().eval()
    x = prepare_input(build_synthetic_dataset()["X"][:4])
    assert completeness_gap(m, x, steps=96) < 0.02


def test_simulator_scoring_toy():
    # one record, 12 windows, stride 10 s; abnormal episode = windows 4..6
    y = np.array([0] * 4 + [3] * 3 + [0] * 5)
    rec = np.array(["r"] * 12)
    t0 = np.arange(12) * 10.0
    act = np.zeros(12, int)
    act[5] = S.ALERT            # detect 1 window late
    act[9] = S.ALERT            # one false trigger
    m = S.score(act, y, rec, t0, 10.0)
    assert m["episodes"] == 1 and m["delay_mean_s"] == 10.0 and m["missed_episode_rate"] == 0
    assert abs(m["false_triggers_per_hour"] - 1 / (12 * 10 / 3600)) < 1e-6


def test_policy_ml_reject_overrides():
    p = np.array([[0.05, 0.9, 0.03, 0.01, 0.01], [0.9, 0.05, 0.03, 0.01, 0.01]])
    act = S.policy_ml(p, np.array([True, False]), 0.6)
    assert act[0] == S.UNRELIABLE and act[1] == S.NONE
    assert S.policy_ml(p, np.array([False, False]), 0.6)[0] == S.RATE_SUPPORT


def _write_wfdb(path_no_ext, sig, fs, names):
    import os
    import wfdb
    d, n = os.path.split(path_no_ext)
    os.makedirs(d, exist_ok=True)
    wfdb.wrsamp(n, fs=fs, units=["mV"] * len(names), sig_name=names, p_signal=sig, fmt=["16"] * len(names),
                write_dir=d)


def test_ptbxl_loader_on_fake_local_copy(tmp_path):
    """Exercises build_ptbxl() parsing (labels, folds, lead II, 500->250 Hz) on a tiny fake PTB-XL tree."""
    import pandas as pd
    from smartpace.datasets import build_ptbxl
    root = tmp_path / "ptb-xl"
    names = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    rng = np.random.default_rng(0)
    sub = _subject(rng, "B")
    rows = []
    for i, (cls, codes, fold) in enumerate([(C.NSR, {"NORM": 100, "SR": 0}, 1), (C.AF, {"AFIB": 0}, 9),
                                            (C.PVC, {"PVC": 0}, 10), (C.OTHER, {"IMI": 50}, 10)]):
        w = make_window(rng, max(cls, 0), sub)
        w500 = np.repeat(w, 2)                                  # crude 250 -> 500 Hz
        sig = np.zeros((5000, 12)); sig[:, 1] = w500; sig[:, 0] = 99.0
        rel = f"records500/00000/0000{i}_hr"
        _write_wfdb(str(root / rel), sig, 500, names)
        rows.append({"ecg_id": i + 1, "patient_id": 100 + i, "strat_fold": fold, "scp_codes": str(codes),
                     "filename_hr": rel, "filename_lr": rel})
    pd.DataFrame(rows).to_csv(root / "ptbxl_database.csv", index=False)
    (root / "scp_statements.csv").write_text("x\n")
    ds = build_ptbxl(str(tmp_path), res=500)
    assert ds["X"].shape == (4, C.WIN)
    assert list(ds["y"]) == [C.NSR, C.AF, C.PVC, C.OTHER]
    assert list(ds["split"]) == [0, 1, 2, 2]
    assert np.abs(ds["X"][0] - 99).max() > 0.01 and ds["X"][0].max() < 5   # lead II used, not lead I


def test_nstdb_loader_on_fake_files(tmp_path):
    for k in C.NOISE_KINDS:
        _write_wfdb(str(tmp_path / k), np.random.default_rng(0).normal(0, 1, (3600, 2)), 360, ["MLII", "V1"])
    nb = NoiseBank.from_nstdb(str(tmp_path))
    assert all(len(nb.sig[k]) == 2500 for k in C.NOISE_KINDS)       # 10 s at 360 Hz -> 250 Hz


def test_ptbxl_subsample_keeps_columns(tmp_path):
    """--ptbxl-max path: per-fold subsampling must keep strat_fold (regression test)."""
    import pandas as pd
    from smartpace.datasets import build_ptbxl
    root = tmp_path / "ptb-xl"
    names = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    rng = np.random.default_rng(0)
    sub = _subject(rng, "B")
    rows = []
    for i in range(30):
        w = make_window(rng, C.NSR, sub)
        sig = np.zeros((1000, 12)); sig[:, 1] = np.repeat(w, 2)[:1000] if False else np.resize(np.repeat(w, 2), 1000) * 0 + 1
        rel = f"records100/00000/{i:05d}_lr"
        _write_wfdb(str(root / rel), np.zeros((1000, 12)) + 1.0, 100, names)
        rows.append({"ecg_id": i + 1, "patient_id": 100 + i, "strat_fold": i % 10 + 1,
                     "scp_codes": str({"NORM": 100, "SR": 0}), "filename_hr": rel, "filename_lr": rel})
    pd.DataFrame(rows).to_csv(root / "ptbxl_database.csv", index=False)
    (root / "scp_statements.csv").write_text("x\n")
    ds = build_ptbxl(str(tmp_path), res=100, max_records=10)
    assert len(ds["y"]) == 10 and set(ds["split"]) <= {0, 1, 2}
    assert ds["X"].shape == (10, C.WIN)


def test_cache_roundtrip_handles_object_arrays(tmp_path):
    """Regression: object-dtype id arrays (newer pandas) must save and load back as plain strings."""
    from smartpace.datasets import _load, _save
    ds = {"X": np.zeros((3, 4), np.float32), "y": np.array([0, 1, 2]),
          "rec": np.array(["a", "b", "c"], dtype=object), "t0": np.zeros(3), "split": np.array([0, 1, 2])}
    _save(str(tmp_path / "c.npz"), ds)
    out = _load(str(tmp_path / "c.npz"))
    assert out["rec"].dtype.kind == "U" and list(out["rec"]) == ["a", "b", "c"]
    # an OLD cache that was written with an object array must also load
    np.savez_compressed(str(tmp_path / "old.npz"), X=ds["X"], y=ds["y"], rec=ds["rec"], t0=ds["t0"], split=ds["split"])
    assert list(_load(str(tmp_path / "old.npz"))["rec"]) == ["a", "b", "c"]


def test_simulator_tolerance_ignores_boundary_alerts():
    y = np.array([0] * 4 + [3] * 3 + [0] * 5)
    rec = np.array(["r"] * 12)
    t0 = np.arange(12) * 10.0
    act = np.zeros(12, int)
    act[5] = S.ALERT
    act[7] = S.ALERT                      # right after the episode: boundary window, ambiguous
    m = S.score(act, y, rec, t0, 10.0)
    assert m["false_triggers_per_hour"] == 0 and m["false_triggers_strict_per_hour"] > 0


def test_sqi_gate_ignores_baseline_wander_by_default():
    from smartpace.preprocess import SQIGate
    clean = np.column_stack([np.full(50, 8.0), np.full(50, 0.01), np.full(50, 0.01), np.ones(50), np.zeros(50)])
    g = SQIGate().fit(clean)
    bw = np.array([[8.0, 0.01, 0.9, 1.0, 0.0]])
    assert not g.flag(bw)[0]
    assert SQIGate(use_bw=True).fit(clean).flag(bw)[0]


def test_edge_memory_measurement_works(tmp_path):
    from smartpace import edge
    p = edge.export_onnx(CNN1D().eval(), str(tmp_path / "m.onnx"))
    r = edge.memory_mb(p)
    assert r["peak_rss_mb"] > 10


def test_model_extract_features():
    from smartpace.models import CNN1D, CNNTransformer, ResNet1D
    from smartpace.train import extract_features
    x = np.random.randn(4, 1, 2500).astype(np.float32)
    for model in [ResNet1D(), CNN1D(), CNNTransformer()]:
        model.eval()
        feat = extract_features(model, x)
        assert feat.shape[0] == 4 and feat.shape[1] > 0
        logits = model(torch.from_numpy(x))
        assert logits.shape == (4, C.N_CLASSES)


def test_mahalanobis_and_euclidean_shift_detectors():
    from smartpace.reliability import EuclideanShiftDetector, MahalanobisShiftDetector
    rng = np.random.default_rng(0)
    # Generate 3 distinct class clusters in 16-D feature space
    F_c0 = rng.normal(loc=-2.0, scale=0.5, size=(50, 16))
    F_c1 = rng.normal(loc=2.0, scale=0.5, size=(50, 16))
    features = np.vstack([F_c0, F_c1])
    labels = np.array([0] * 50 + [1] * 50)

    # In-distribution validation samples
    val_feat = np.vstack([rng.normal(loc=-2.0, scale=0.5, size=(20, 16)),
                          rng.normal(loc=2.0, scale=0.5, size=(20, 16))])

    # OOD samples far from both clusters
    ood_feat = rng.normal(loc=15.0, scale=1.0, size=(20, 16))

    # Test Mahalanobis detector
    mah = MahalanobisShiftDetector().fit(features, labels)
    tau_mah = mah.calibrate(val_feat, target_coverage=0.90)
    val_scores = mah.score(val_feat)
    ood_scores = mah.score(ood_feat)

    assert (val_scores <= tau_mah).mean() >= 0.85
    assert ood_scores.min() > val_scores.max()

    # Test Euclidean detector
    euc = EuclideanShiftDetector().fit(features, labels)
    tau_euc = euc.calibrate(val_feat, target_coverage=0.90)
    assert euc.score(ood_feat).min() > euc.score(val_feat).max()


def test_adaptive_abstention_module_and_four_configurations():
    from smartpace.preprocess import SQIGate
    from smartpace.reliability import AdaptiveAbstentionModule, MahalanobisShiftDetector, evaluate_four_configurations

    rng = np.random.default_rng(42)
    n = 60
    # Synthetic embeddings, labels, and calibrated probabilities
    labels = rng.choice([0, 1, 2], size=n)
    features = rng.normal(loc=labels[:, None] * 3.0, scale=0.5, size=(n, 8))
    # Probabilities
    probs = np.full((n, C.N_CLASSES), 0.05, dtype=np.float32)
    for i in range(n):
        probs[i, labels[i]] = 0.8
    # Add some uncertain samples
    probs[:10] = 1.0 / C.N_CLASSES

    # Synthetic clean SQI features (5 columns)
    sqi = np.column_stack([np.full(n, 8.0), np.full(n, 0.01), np.full(n, 0.01), np.ones(n), np.zeros(n)])
    # Make a few samples have bad SQI
    sqi[45:50, 0] = 0.0 # flat kurtosis
    sqi[45:50, 4] = 1.0 # flat flag

    sqi_gate = SQIGate().fit(sqi)
    shift_det = MahalanobisShiftDetector().fit(features, labels)
    shift_det.calibrate(features, target_coverage=0.95)

    abstention = AdaptiveAbstentionModule(
        sqi_gate=sqi_gate,
        shift_detector=shift_det,
        tau_unc=0.8,
        tau_shift=shift_det.threshold_,
    )

    dec = abstention.decide(probs, features, sqi)
    assert len(dec.action) == n
    assert "REJECT_QUALITY" in dec.action
    assert "ABSTAIN_UNCERTAIN" in dec.action
    assert "ACCEPT" in dec.action

    # Test four configurations evaluation
    ood_feat = rng.normal(loc=30.0, scale=1.0, size=(20, 8))
    res = evaluate_four_configurations(
        y_true=labels,
        probs=probs,
        features=features,
        sqi_features=sqi,
        abstention_module=abstention,
        ood_features=ood_feat,
    )

    assert "1. Baseline Classifier" in res
    assert "2. Classifier + Uncertainty Gate" in res
    assert "3. Classifier + Signal Quality Gate" in res
    assert "4. Full Adaptive Tri-Gate (SmartPace)" in res

    # Baseline coverage must be 1.0
    assert res["1. Baseline Classifier"]["coverage"] == 1.0
    # Full Adaptive Tri-Gate must have risk <= baseline risk (or nan if coverage 0)
    assert res["4. Full Adaptive Tri-Gate (SmartPace)"]["risk_at_coverage"] <= res["1. Baseline Classifier"]["risk_at_coverage"]
    assert res["4. Full Adaptive Tri-Gate (SmartPace)"]["auroc_ood_detection"] > 0.8


def test_adapt_linear_probe():
    from smartpace.adaptation import adapt_linear_probe
    rng = np.random.default_rng(0)
    F_src = rng.normal(size=(60, 16))
    y_src = rng.choice([0, 1, 2], size=60)
    F_adapt = rng.normal(loc=1.0, size=(20, 16))
    y_adapt = rng.choice([0, 1, 2], size=20)
    F_te = rng.normal(loc=1.0, size=(30, 16))

    probs, preds = adapt_linear_probe(
        features_source_train=F_src,
        y_source_train=y_src,
        features_target_adapt=F_adapt,
        y_target_adapt=y_adapt,
        features_target_test=F_te,
    )
    assert probs.shape == (30, C.N_CLASSES)
    assert len(preds) == 30
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)


def test_sampling_rate_control_and_cross_rejection():
    from smartpace.adaptation import evaluate_cross_dataset_rejection, evaluate_sampling_rate_control
    from smartpace.models import CNN1D
    from smartpace.reliability import AdaptiveAbstentionModule, MahalanobisShiftDetector

    rng = np.random.default_rng(1)
    n = 10
    x = rng.normal(size=(n, 1, 2500)).astype(np.float32)
    y = rng.choice([0, 1, 2], size=n)
    model = CNN1D().eval()

    def dummy_pred(m, batch):
        return rng.normal(size=(len(batch), C.N_CLASSES)).astype(np.float32)

    # 1. Test sampling rate control
    samp_res = evaluate_sampling_rate_control(model, x, y, dummy_pred)
    assert "1. Standard Harmonized (250 Hz)" in samp_res
    assert "2. Bandwidth-Limited (100 Hz Sim)" in samp_res
    assert "3. High-Rate Reconstructed (500 Hz Sim)" in samp_res

    # 2. Test cross rejection
    features = rng.normal(size=(n, 16))
    probs = np.full((n, C.N_CLASSES), 0.2, dtype=np.float32)
    shift_det = MahalanobisShiftDetector().fit(features, y)
    abstention = AdaptiveAbstentionModule(shift_detector=shift_det, tau_unc=1.0, tau_shift=10.0)

    curves = evaluate_cross_dataset_rejection(y, probs, features, abstention)
    assert len(curves) > 0
    assert "coverage" in curves[0]
    assert "macro_f1" in curves[0]


def test_noise_rejection_policies():
    from smartpace.noise_policy import (
        AdaptiveReliabilityPolicy,
        FixedDualGatePolicy,
        SQIOnlyPolicy,
        UncertaintyOnlyPolicy,
        ValCalibratedDualGatePolicy,
        compute_sqi_degradation_index,
        evaluate_noise_type_breakdown,
        evaluate_policy_operating_point,
        generate_tradeoff_curve,
    )
    from smartpace.preprocess import SQIGate

    rng = np.random.default_rng(42)
    n = 60
    # Dummy SQI features: [kurt, hf_ratio, bw_ratio, rr_valid, flat]
    clean_sqi = np.column_stack([
        rng.uniform(3.0, 6.0, size=n),
        rng.uniform(0.01, 0.05, size=n),
        rng.uniform(0.01, 0.05, size=n),
        np.ones(n),
        np.zeros(n),
    ]).astype(np.float32)

    noisy_sqi = np.column_stack([
        rng.uniform(0.5, 2.0, size=n),
        rng.uniform(0.2, 0.8, size=n),
        rng.uniform(0.1, 0.6, size=n),
        rng.uniform(0.0, 0.5, size=n),
        rng.choice([0.0, 1.0], size=n, p=[0.8, 0.2]),
    ]).astype(np.float32)

    deg = compute_sqi_degradation_index(noisy_sqi, clean_sqi)
    assert deg.shape == (n,)
    assert (deg >= 0.0).all() and (deg <= 1.0).all()

    # Probabilities
    probs = rng.uniform(0.01, 1.0, size=(n, C.N_CLASSES))
    probs /= probs.sum(axis=1, keepdims=True)
    y_true = rng.choice(list(range(C.N_CLASSES)), size=n)

    # 1. SQI Only Policy
    sqi_pol = SQIOnlyPolicy(clean_sqi)
    sqi_scores = sqi_pol.score_unreliability(probs, noisy_sqi)
    assert sqi_scores.shape == (n,)

    # 2. Uncertainty Only Policy
    unc_pol = UncertaintyOnlyPolicy()
    unc_scores = unc_pol.score_unreliability(probs, noisy_sqi)
    assert unc_scores.shape == (n,)

    # 3. Fixed Dual Gate Policy
    sqi_gate = SQIGate(q=5.0).fit(clean_sqi)
    fixed_pol = FixedDualGatePolicy(sqi_gate, tau_unc=1.0)
    fixed_rejs = fixed_pol.decide(probs, noisy_sqi)
    assert fixed_rejs.shape == (n,)
    assert fixed_rejs.dtype == bool

    # 4. Validation-Calibrated Policy
    val_pol = ValCalibratedDualGatePolicy(clean_sqi)
    val_pol.calibrate_for_coverage(probs, noisy_sqi, target_coverage=0.7)
    val_rejs = val_pol.decide(probs, noisy_sqi)
    assert val_rejs.shape == (n,)

    # 5. Adaptive Reliability Policy
    adapt_pol = AdaptiveReliabilityPolicy(clean_sqi)
    adapt_scores = adapt_pol.score_unreliability(probs, noisy_sqi)
    assert adapt_scores.shape == (n,)

    # Tradeoff curve generation
    tradeoff = generate_tradeoff_curve(
        adapt_pol,
        probs,
        noisy_sqi,
        y_true,
        target_coverages=[0.5, 0.8, 1.0],
    )
    assert len(tradeoff) == 3
    assert "error_rate" in tradeoff[0]
    assert "abnormal_recall" in tradeoff[0]
    assert "arrhythmia_fn_rate" in tradeoff[0]

    # Noise breakdown evaluation
    noise_kinds = np.array(["bw"] * 20 + ["em"] * 20 + ["ma"] * 20)
    snrs = np.array([12] * 30 + [-6] * 30)
    breakdown = evaluate_noise_type_breakdown(adapt_pol, probs, noisy_sqi, noise_kinds, snrs, target_coverage=0.7)
    assert len(breakdown) > 0
    assert "noise_kind" in breakdown[0]
    assert "rejection_rate" in breakdown[0]


