"""Collect result CSVs into a single REPORT.md."""
import json
import os

import pandas as pd


def _md(df, floatfmt=3):
    df = df.copy()
    for c in df.columns:
        if df[c].dtype.kind == "f":
            df[c] = df[c].map(lambda v: "" if pd.isna(v) else f"{v:.{floatfmt}f}")
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    body = ["| " + " | ".join(str(v) for v in r) + " |" for r in df.to_numpy()]
    return "\n".join([head, sep] + body)


def _read(out, name):
    p = os.path.join(out, "results", name)
    return pd.read_csv(p) if os.path.exists(p) else None


def make_report(out):
    info = {}
    p = os.path.join(out, "results", "run_info.json")
    if os.path.exists(p):
        info = json.load(open(p, encoding="utf-8"))
    L = ["# SmartPace - results report", ""]
    if info.get("synthetic"):
        L += ["> **SYNTHETIC DATA RUN.** These numbers come from generated ECG and only prove that the "
              "pipeline executes. They say nothing about real-world performance.", ""]
    L += ["> The decision-support simulator is **not a clinically validated pacemaker controller**.", ""]
    if info:
        L += [f"- Noise source: {info.get('noise_source')}", f"- Total run time: {info.get('total_seconds')} s", ""]
        mt = info.get("main_train")
        if mt and mt != "A":
            L += [f"- Experiments 3-6 use the ResNet1D trained on **{mt}** (MIT-BIH + PTB-XL training splits). Their "
                  "test sets are therefore in-domain for both datasets; the cross-dataset study is Section 2.", ""]

    def section(title, files, note=None, figs=()):
        L.append(f"## {title}")
        if note:
            L.append(note)
        for name, cap in files:
            df = _read(out, name)
            if df is not None:
                L.extend(["", f"**{cap}**", "", _md(df), ""])
        for f in figs:
            if os.path.exists(os.path.join(out, "figures", f)):
                L.extend([f"![{f}](figures/{f})", ""])

    section("1. Detection, uncertainty and rejection (remark 1)",
            [("exp1_summary.csv", "MIT-BIH test (inter-patient), mean ± std over seeds"),
             ("exp1_uncertainty.csv", "Uncertainty methods (ResNet1D)"),
             ("exp1_reliability_comparison.csv", "Adaptive Abstention: 4-Configuration Comparison (Signal Quality + Uncertainty + Distribution Shift)"),
             ("exp1_shift_detector_comparison.csv", "Distribution Shift Detectors: Mahalanobis vs Centroid Baseline")],
            figs=["exp1_rejection.png", "exp1_adaptive_abstention.png"])
    section("2. Cross-dataset generalisation (remark 2)",
            [("exp2_label_distribution.csv", "Unified-label window counts"),
             ("exp2_summary.csv", "Train / test combinations"),
             ("exp2_adaptation.csv", "Experiment E: Limited Target Adaptation vs Cross-Dataset Transfer"),
             ("exp2_sampling_control.csv", "Sampling Rate & Bandwidth Control Experiment"),
             ("exp2_cross_rejection.csv", "Cross-Dataset Rejection: Selective Prediction under Domain Shift")],
            figs=["exp2_confusion.png", "exp2_cross_rejection.png"])
    section("3. Calibration and explanations (remark 3 & Improvement 4)",
            [("exp3_calibration.csv", "Calibration"),
             ("exp3_plausibility.csv", "Attribution enrichment inside P / QRS / T regions (>1 = concentrated there)"),
             ("exp3_stress_test_cases.csv", "Explanation Stress Testing & Failure Mode Classification (Cases A, B, C)"),
             ("exp3_error_detection_auroc.csv", "Error Detection AUROC: Confidence vs. Explanation Instability vs. Combined")],
            figs=["exp3_reliability.png", "exp3_attributions.png", "exp3_failure_cases.png"],
            note="Improvement 4 stress-tests explanations under physiological perturbations and counterfactuals, revealing Case C silent failures "
                 "where confidence is high but attributions are fragile.")
    ex = os.path.join(out, "results", "exp3_explanation.json")
    if os.path.exists(ex):
        s = json.load(open(ex, encoding="utf-8"))["summary"]
        L += ["**Explanation metrics**", "", "```", json.dumps(s, indent=1), "```", ""]
    section("4. Noise robustness and rejection (remark 4 & Improvement 3)",
            [("exp4_noise.csv", "Performance vs. noise type and SNR"),
             ("exp4_tradeoff_curves.csv", "5-Policy Risk-Coverage and Macro-F1 Trade-off Curves"),
             ("exp4_noise_type_rejection.csv", "Rejection Rates across Noise Types and SNR Levels"),
             ("exp4_cross_noise_transfer.csv", "Cross-Noise Transfer Generalization Evaluation")],
            figs=["exp4_noise.png", "exp4_rejection_rate.png", "exp4_tradeoff_curves.png", "exp4_noise_type_rejection.png"],
            note="Improvement 3 compares 5 abstention policies (SQI-only, Uncertainty-only, Fixed Dual-Gate, Val-Calibrated Dual Gate, "
                 "and Adaptive Reliability Policy) across complete risk-coverage trade-off curves, abnormal rhythm sensitivity, and cross-noise transfer.")
    section("5. Edge model benchmark (remark 5 & Improvement 5)",
            [("exp5_edge.csv", "FP32 vs INT8 vs Pruned Performance and Resource Profile"),
             ("exp5_confidence_distortion.csv", "Edge Calibration and Confidence Distortion Analysis")],
            figs=["exp5_reliability_resource_tradeoff.png"],
            note="Improvement 5 evaluates the reliability-resource trade-off: latency, file size, and peak memory "
                 "benchmarked against calibration (ECE, Brier score), signed confidence distortion, and abstention discordance.")
    section("6. SIMULATED decision support (remark 6)", [("exp6_simulator.csv", "Policy comparison")],
            note="False triggers are counted as events per hour of replayed recording; delay is the time from "
                 "episode start to the first correct simulated action.", figs=["exp6_timeline.png"])
    with open(os.path.join(out, "REPORT.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))
