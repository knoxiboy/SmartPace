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
