"""The six experiments that answer the six review remarks (see README.md).

exp1  detection + uncertainty + rejection vs baselines            (remark 1)
exp2  cross-dataset generalisation MIT-BIH <-> PTB-XL             (remark 2)
exp3  calibration + Integrated Gradients evaluation               (remark 3)
exp4  noise robustness + uncertainty/quality-based rejection      (remark 4)
exp5  edge model: INT8 / pruning benchmark                        (remark 5)
exp6  SIMULATED decision support vs fixed-threshold baselines     (remark 6)
"""
import json
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.ensemble import HistGradientBoostingClassifier

from . import config as C
from . import edge, explain, simulator as S
from . import evaluation as E
from .datasets import get_split, load_dataset, merge, class_counts
from .noise import NoiseBank, download_nstdb
from .preprocess import SQIGate, feature_matrix, prepare_input, sqi_features
from .train import (TrainCfg, ensemble_logits, get_or_train, load_model, mc_dropout_probs,
                    predict_logits)

warnings.filterwarnings("ignore")


# ============================================================================= context
class Ctx:
    """Holds datasets, noise bank, models and caches shared by all experiments."""

    def __init__(self, a):
        self.a = a
        self.out = a.out
        for d in ("results", "figures", "models", "edge"):
            os.makedirs(os.path.join(self.out, d), exist_ok=True)
        self.cfg = TrainCfg(epochs=a.epochs, batch_size=a.batch_size, patience=a.patience, aug_noise_p=0.3)
        self.info = {"synthetic": bool(a.synthetic)}
        print("Loading datasets ...", flush=True)
        self.A = load_dataset("mitbih", a.data_dir, a.synthetic)
        self.B = load_dataset("ptbxl", a.data_dir, a.synthetic, a.ptbxl_max, a.ptbxl_res)
        self.AB = None
        if a.synthetic:
            self.noise = NoiseBank.synthetic()
            self.info["noise_source"] = "synthetic"
        else:
            try:
                self.noise = NoiseBank.from_nstdb(download_nstdb(a.data_dir))
                self.info["noise_source"] = "NSTDB"
            except Exception as e:                                   # pragma: no cover
                print("WARNING: NSTDB unavailable (%s); using synthetic noise." % e)
                self.noise = NoiseBank.synthetic()
                self.info["noise_source"] = "synthetic (NSTDB unavailable)"
        self.main = a.main_train
        self.info["main_train"] = self.main
        self._T = {}
        self._gate = None
        self._feat = {}
        self._xcache = {}

    # ---- paths / io
    def p(self, *parts):
        return os.path.join(self.out, *parts)

    def save_csv(self, df, name):
        df.to_csv(self.p("results", name), index=False)
        return df

    # ---- data
    def ds(self, which):
        if which == "AB":
            if self.AB is None:
                self.AB = merge(self.A, self.B)
            return self.AB
        return {"A": self.A, "B": self.B}[which]

    def split(self, which, split):
        return get_split(self.ds(which), split)

    def inp(self, which, split, filt=True):
        key = (which, split, filt)
        if key not in self._xcache:
            self._xcache[key] = prepare_input(self.split(which, split)[0], filt)
        return self._xcache[key]

    # ---- models
    def models(self, arch, trainset="A", filt=True, n=None):
        Xtr, ytr, _, _ = self.split(trainset, "train")
        Xva, yva, _, _ = self.split(trainset, "val")
        n = n or self.a.seeds
        out = []
        for s in range(n):
            name = f"{arch}_{trainset}{'' if filt else '_nofilt'}_s{s}"
            out.append(get_or_train(self.p("models"), name, arch, Xtr, ytr, Xva, yva, self.cfg,
                                    self.noise, filt, s))
        return out

    # ---- features for the feature-based baseline (cached)
    def features(self, which, split):
        key = (which, split)
        if key not in self._feat:
            if which == "AB":          # merge() concatenates A then B, so the order is preserved
                self._feat[key] = np.vstack([self.features("A", split), self.features("B", split)])
            else:
                self._feat[key] = feature_matrix(self.split(which, split)[0])
        return self._feat[key]

    # ---- calibration / gate
    def temperature(self, ts=None):
        """Temperature fitted on the validation set of training set `ts` (default: main model)."""
        ts = ts or self.main
        if ts not in self._T:
            ms = self.models("resnet1d", ts)
            _, yva, _, _ = self.split(ts, "val")
            lg = ensemble_logits(ms, self.inp(ts, "val")).mean(0)
            self._T[ts] = E.fit_temperature(lg, yva)
        return self._T[ts]

    def gate(self):
        """Signal-quality gate fitted on CLEAN validation windows plus a sample of training windows
        (many patients), so thresholds do not depend on just a few validation records."""
        if self._gate is None:
            Xv = self.split(self.main, "val")[0]
            Xt = self.split(self.main, "train")[0]
            rng = np.random.default_rng(0)
            Xt = Xt[rng.permutation(len(Xt))[:self.a.sqi_n]]
            self._gate = SQIGate(q=self.a.sqi_q, use_bw=False).fit(sqi_features(np.concatenate([Xv, Xt])))
        return self._gate

    def tau_unc(self, ts=None):
        """Entropy threshold giving `reject_coverage` acceptance on clean validation data."""
        ts = ts or self.main
        ms = self.models("resnet1d", ts)
        p = E.apply_temperature(ensemble_logits(ms, self.inp(ts, "val")).mean(0), self.temperature(ts))
        return E.threshold_for_coverage(E.entropy(p), self.a.reject_coverage)

    def ens_probs(self, ms, x, calibrated=True, ts=None):
        lg = ensemble_logits(ms, x).mean(0)
        return E.apply_temperature(lg, self.temperature(ts) if calibrated else 1.0)


def _fmt(v):
    v = np.asarray(v, float)
    return f"{v[0]:.4f}" if len(v) == 1 else f"{np.mean(v):.4f} ± {np.std(v):.4f}"


def _metrics(y, probs):
    pred = probs.argmax(1)
    d = {"macro_f1": E.macro_f1(y, pred), "accuracy": E.accuracy(y, pred),
         "auroc": E.macro_auroc(y, probs)}
    d.update({"f1_" + k: v for k, v in E.per_class_f1(y, pred).items()})
    return d


# ============================================================================= exp1
def exp1_detection(ctx):
    print("\n=== EXP1: detection, uncertainty and rejection ===", flush=True)
    Xte_raw, yte, _, _ = ctx.split("A", "test")
    xte = ctx.inp("A", "test")
    rows = []
    archs = ["cnn1d", "resnet1d"] + (["cnntransformer"] if ctx.a.with_transformer else [])
    allm = {}
    for arch in archs:
        ms = ctx.models(arch, "A", n=ctx.a.seeds if arch != "cnntransformer" else 1)
        allm[arch] = ms
        for s, m in enumerate(ms):
            pr = softmax(predict_logits(m, xte), 1)
            rows.append({"model": arch, "seed": s, **_metrics(yte, pr)})
    # B2: HRV + morphology features with gradient boosting
    Xtr_raw, ytr, _, _ = ctx.split("A", "train")
    print("  extracting features for baseline B2 ...", flush=True)
    Ftr, Fte = feature_matrix(Xtr_raw), feature_matrix(Xte_raw)
    for s in range(1):          # deterministic baseline: one run (a std over seeds would be 0)
        gb = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, class_weight="balanced",
                                            random_state=s).fit(Ftr, ytr)
        pr = np.zeros((len(Fte), C.N_CLASSES))
        pr[:, gb.classes_] = gb.predict_proba(Fte)
        rows.append({"model": "features_gbm", "seed": s, **_metrics(yte, pr)})
    df = ctx.save_csv(pd.DataFrame(rows), "exp1_runs.csv")
    summ = []
    for m, g in df.groupby("model"):
        r = {"model": m, "n_seeds": len(g)}
        for k in ["macro_f1", "accuracy", "auroc"] + ["f1_" + c for c in C.CLASSES]:
            r[k] = _fmt(g[k].dropna()) if g[k].notna().any() else "n/a"
        summ.append(r)
    ctx.save_csv(pd.DataFrame(summ), "exp1_summary.csv")

    # ---- uncertainty methods on the ResNet1D
    ms = allm["resnet1d"]
    xva = ctx.inp("A", "val")
    _, yva, _, _ = ctx.split("A", "val")

    def method(name, x):
        if name == "softmax":
            p = softmax(predict_logits(ms[0], x), 1)
            return p, E.entropy(p)
        if name == "mc_dropout":
            ps = mc_dropout_probs(ms[0], x, T=ctx.a.mc_T)
            p = ps.mean(0)
            return p, E.entropy(p)
        ps = np.stack([softmax(predict_logits(m, x), 1) for m in ms])
        p = ps.mean(0)
        return p, E.entropy(p)

    curves, unc_rows = [], []
    for name in ["softmax", "mc_dropout", "ensemble"]:
        pv, uv = method(name, xva)
        pt, ut = method(name, xte)
        pred = pt.argmax(1)
        tau = E.threshold_for_coverage(uv, ctx.a.reject_coverage)
        rs = E.reject_stats(yte, pred, ut > tau)
        unc_rows.append({"method": name, "macro_f1_all": E.macro_f1(yte, pred),
                         "aurc": E.aurc(yte, pred, ut),
                         "auroc_error_detection": _error_auroc(yte, pred, ut), **rs})
        for r in E.rejection_curve(yte, pred, ut):
            curves.append({"method": name, **r})
    ctx.save_csv(pd.DataFrame(unc_rows), "exp1_uncertainty.csv")
    cdf = ctx.save_csv(pd.DataFrame(curves), "exp1_rejection_curves.csv")
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
    for name, g in cdf.groupby("method"):
        ax[0].plot(g["coverage"], g["accuracy"], marker="o", label=name)
        ax[1].plot(g["coverage"], g["macro_f1"], marker="o", label=name)
    for a_, t in zip(ax, ["Accuracy on accepted", "Macro-F1 on accepted"]):
        a_.set_xlabel("coverage (fraction kept)"); a_.set_title(t); a_.invert_xaxis(); a_.grid(alpha=.3)
    ax[0].legend(); fig.tight_layout(); fig.savefig(ctx.p("figures", "exp1_rejection.png"), dpi=130); plt.close(fig)
    print(pd.DataFrame(summ)[["model", "macro_f1", "accuracy"]].to_string(index=False))


def _error_auroc(y, pred, unc):
    from sklearn.metrics import roc_auc_score
    err = (y != pred).astype(int)
    return float(roc_auc_score(err, unc)) if 0 < err.sum() < len(err) else float("nan")


# ============================================================================= exp2
def exp2_cross_dataset(ctx):
    print("\n=== EXP2: cross-dataset generalisation ===", flush=True)
    dist = []
    for name, ds in [("MIT-BIH", ctx.A), ("PTB-XL", ctx.B)]:
        for sp, sid in [("train", 0), ("val", 1), ("test", 2)]:
            dist.append({"dataset": name, "split": sp, **class_counts(ds["y"][ds["split"] == sid])})
    ctx.save_csv(pd.DataFrame(dist), "exp2_label_distribution.csv")
    names = {"A": "MIT-BIH", "B": "PTB-XL", "AB": "MIT-BIH+PTB-XL"}
    rows = []
    for tr in ["A", "B", "AB"]:
        for arch in ["resnet1d", "cnn1d"]:
            for s, m in enumerate(ctx.models(arch, tr)):
                for te in ["A", "B"]:
                    y = ctx.split(te, "test")[1]
                    pr = softmax(predict_logits(m, ctx.inp(te, "test")), 1)
                    rows.append({"model": arch, "train": names[tr], "test": names[te], "seed": s, **_metrics(y, pr)})
        # feature-based baseline trained on the same training set (fair comparison)
        ytr = ctx.split(tr, "train")[1]
        Ftr = ctx.features(tr, "train")
        for s in range(1):      # deterministic baseline: one run
            gb = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, class_weight="balanced",
                                                random_state=s).fit(Ftr, ytr)
            for te in ["A", "B"]:
                y = ctx.split(te, "test")[1]
                pr = np.zeros((len(y), C.N_CLASSES))
                pr[:, gb.classes_] = gb.predict_proba(ctx.features(te, "test"))
                rows.append({"model": "features_gbm", "train": names[tr], "test": names[te], "seed": s, **_metrics(y, pr)})
    df = ctx.save_csv(pd.DataFrame(rows), "exp2_runs.csv")
    summ = []
    for (mdl, tr, te), g in df.groupby(["model", "train", "test"], sort=False):
        r = {"model": mdl, "train": tr, "test": te, "setting": "within-dataset" if tr == te else
             ("combined" if "+" in tr else "cross-dataset")}
        for k in ["macro_f1", "accuracy", "auroc"] + ["f1_" + c for c in C.CLASSES]:
            r[k] = _fmt(g[k].dropna()) if g[k].notna().any() else "n/a"
        summ.append(r)
    sdf = ctx.save_csv(pd.DataFrame(summ), "exp2_summary.csv")
    # confusion matrices of the seed-0 MIT-BIH-trained ResNet on both test sets
    from sklearn.metrics import confusion_matrix
    m0 = ctx.models("resnet1d", "A")[0]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for a_, te, nm in zip(ax, ["A", "B"], ["MIT-BIH test", "PTB-XL test (cross-dataset)"]):
        y = ctx.split(te, "test")[1]
        pred = predict_logits(m0, ctx.inp(te, "test")).argmax(1)
        cm = confusion_matrix(y, pred, labels=list(range(C.N_CLASSES)))
        cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
        a_.imshow(cmn, vmin=0, vmax=1, cmap="Blues")
        a_.set_xticks(range(5)); a_.set_xticklabels(C.CLASSES, rotation=45); a_.set_yticks(range(5)); a_.set_yticklabels(C.CLASSES)
        for i in range(5):
            for j in range(5):
                a_.text(j, i, cm[i, j], ha="center", va="center", fontsize=8, color="white" if cmn[i, j] > .5 else "black")
        a_.set_title("Trained on MIT-BIH -> " + nm, fontsize=9); a_.set_xlabel("predicted"); a_.set_ylabel("true")
    fig.tight_layout(); fig.savefig(ctx.p("figures", "exp2_confusion.png"), dpi=130); plt.close(fig)
    print(sdf[["model", "train", "test", "macro_f1", "accuracy"]].to_string(index=False))


# ============================================================================= exp3
def exp3_calibration_explain(ctx):
    print("\n=== EXP3: calibration and explanations ===", flush=True)
    rows, rel = [], []
    sets = [("A", "MIT-BIH-trained")] + ([(ctx.main, f"main ({ctx.main}-trained)")] if ctx.main != "A" else [])
    for ts, ts_name in sets:
        ms_ = ctx.models("resnet1d", ts)
        T = ctx.temperature(ts)
        for te, nm in [("A", "MIT-BIH test"), ("B", "PTB-XL test")]:
            y = ctx.split(te, "test")[1]
            lg = ensemble_logits(ms_, ctx.inp(te, "test")).mean(0)
            variants = {"uncalibrated": softmax(lg, 1), f"temperature (T={T:.2f})": E.apply_temperature(lg, T)}
            if ts == "A" and te == "B":
                yb = ctx.split("B", "val")[1]
                Tb = E.fit_temperature(ensemble_logits(ms_, ctx.inp("B", "val")).mean(0), yb)
                variants[f"temperature (T={Tb:.2f}, refit on PTB-XL val)"] = E.apply_temperature(lg, Tb)
            shift = "in-domain" if (ts == te or ts == "AB") else "cross-dataset"
            for vn, p in variants.items():
                rows.append({"model": ts_name, "test_set": f"{nm} ({shift})", "calibration": vn, "ece": E.ece(p, y),
                             "brier": E.brier(p, y), "nll": float(-np.log(p[np.arange(len(y)), y] + 1e-12).mean()),
                             "accuracy": E.accuracy(y, p.argmax(1))})
                for lo, conf, acc, n in E.reliability(p, y):
                    rel.append({"model": ts_name, "test_set": f"{nm} ({shift})", "calibration": vn,
                                "bin_center": lo, "mean_conf": conf, "acc": acc, "n": n})
    ctx.save_csv(pd.DataFrame(rows), "exp3_calibration.csv")
    rdf = pd.DataFrame(rel)
    keys = list(dict.fromkeys(zip(rdf["model"], rdf["test_set"])))
    fig, axes = plt.subplots(1, len(keys), figsize=(4.5 * len(keys), 4), squeeze=False)
    for a_, (mn, tn) in zip(axes[0], keys):
        g = rdf[(rdf.model == mn) & (rdf.test_set == tn)]
        a_.plot([0, 1], [0, 1], "k--", lw=1)
        for vn, h in g.groupby("calibration", sort=False):
            h = h.dropna()
            a_.plot(h["mean_conf"], h["acc"], marker="o", label=vn[:30])
        a_.set_title(f"{mn}\n{tn}", fontsize=8); a_.set_xlabel("confidence"); a_.set_ylabel("accuracy")
        a_.grid(alpha=.3); a_.legend(fontsize=6)
    fig.tight_layout(); fig.savefig(ctx.p("figures", "exp3_reliability.png"), dpi=130); plt.close(fig)

    # ---- explanations on a class-balanced subset of the MIT-BIH test set (main model)
    ms = ctx.models("resnet1d", ctx.main)
    Xraw, y, _, _ = ctx.split("A", "test")
    x = ctx.inp("A", "test")
    rng = np.random.default_rng(0)
    idx = np.concatenate([rng.permutation(np.where(y == c)[0])[:ctx.a.expl_per_class] for c in range(C.N_CLASSES)])
    xs, rs, ys = x[idx], Xraw[idx], y[idx]
    m0 = ms[0]
    preds = predict_logits(m0, xs).argmax(1)
    attr = explain.integrated_gradients(m0, xs, preds, steps=ctx.a.ig_steps)
    stab = explain.stability(m0, xs, steps=ctx.a.ig_steps)
    faith = explain.deletion_insertion(m0, xs, attr)
    plaus = explain.plausibility(attr, rs, preds)
    gap = explain.completeness_gap(m0, xs[:16], steps=max(64, ctx.a.ig_steps))
    out = {"n_windows": int(len(idx)), "ig_steps": ctx.a.ig_steps, "completeness_gap": gap, **stab, **faith}
    with open(ctx.p("results", "exp3_explanation.json"), "w") as f:
        json.dump({"summary": out, "plausibility_enrichment": plaus}, f, indent=2)
    prow = [{"predicted_class": c, **v} for c, v in plaus.items()]
    ctx.save_csv(pd.DataFrame(prow), "exp3_plausibility.csv")
    # figure: one example per class
    fig, axes = plt.subplots(C.N_CLASSES, 1, figsize=(10, 9), sharex=True)
    t = np.arange(C.WIN) / C.FS
    sm = explain.smooth_abs(attr)
    for c in range(C.N_CLASSES):
        w = np.where(preds == c)[0]
        ax = axes[c]
        ax.set_ylabel(C.CLASSES[c])
        if len(w):
            i = w[0]
            ax.plot(t, xs[i, 0], lw=.8, color="k")
            ax2 = ax.twinx(); ax2.fill_between(t, 0, sm[i] / (sm[i].max() + 1e-12), color="r", alpha=.3)
            ax2.set_yticks([])
    axes[-1].set_xlabel("time (s)"); axes[0].set_title("Integrated Gradients (red = attribution) on predicted class")
    fig.tight_layout(); fig.savefig(ctx.p("figures", "exp3_attributions.png"), dpi=130); plt.close(fig)
    print(json.dumps(out, indent=1))


# ============================================================================= exp4
def exp4_noise(ctx):
    print("\n=== EXP4: noise robustness and rejection ===", flush=True)
    ms = ctx.models("resnet1d", ctx.main)
    b1 = ctx.models("cnn1d", ctx.main, n=1)[0]
    nof = ctx.models("resnet1d", ctx.main, filt=False, n=1)[0]
    Xraw, y, _, _ = ctx.split("A", "test")
    rng = np.random.default_rng(1)
    sel = rng.permutation(len(y))[:ctx.a.noise_n]
    Xr, ys = Xraw[sel], y[sel]
    gate, tau = ctx.gate(), ctx.tau_unc()
    conds = [("clean", None, np.nan)] + [(k, k, s) for k in C.NOISE_KINDS for s in C.SNR_LEVELS]
    rows = []
    for name, kind, snr in conds:
        Xn = Xr if kind is None else ctx.noise.corrupt(Xr, kind, snr, np.random.default_rng(7), part="test")
        x = prepare_input(Xn, True)
        p = ctx.ens_probs(ms, x)
        pred = p.argmax(1)
        unc_rej = E.entropy(p) > tau
        q_rej = gate.flag(sqi_features(Xn))
        rej = unc_rej | q_rej
        rs = E.reject_stats(ys, pred, rej)
        rows.append({"noise": name, "snr_db": snr,
                     "macro_f1_resnet_ens": E.macro_f1(ys, pred),
                     "macro_f1_resnet_seed0": E.macro_f1(ys, predict_logits(ms[0], x).argmax(1)),
                     "macro_f1_cnn1d": E.macro_f1(ys, predict_logits(b1, x).argmax(1)),
                     "macro_f1_resnet_nofilter": E.macro_f1(ys, predict_logits(nof, prepare_input(Xn, False)).argmax(1)),
                     "ece": E.ece(p, ys), "reject_rate": rs["reject_rate"],
                     "reject_rate_uncertainty_only": float(unc_rej.mean()),
                     "reject_rate_quality_only": float(q_rej.mean()),
                     "macro_f1_accepted": rs["f1_accepted"], "acc_accepted": rs["acc_accepted"],
                     "acc_rejected": rs["acc_rejected"], "acc_all": E.accuracy(ys, pred)})
        print(f"  {name:6s} snr {snr:>5}: F1 {rows[-1]['macro_f1_resnet_ens']:.3f}  "
              f"F1(accepted) {rs['f1_accepted']:.3f}  reject {rs['reject_rate']:.2f}", flush=True)
    df = ctx.save_csv(pd.DataFrame(rows), "exp4_noise.csv")
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
    clean = df[df.noise == "clean"].iloc[0]
    for a_, k in zip(ax, C.NOISE_KINDS):
        g = df[df.noise == k].sort_values("snr_db")
        a_.plot(g.snr_db, g.macro_f1_resnet_ens, "o-", label="ResNet1D ensemble (all)")
        a_.plot(g.snr_db, g.macro_f1_accepted, "s-", label="ResNet1D (accepted only)")
        a_.plot(g.snr_db, g.macro_f1_resnet_seed0, "d-.", label="ResNet1D seed 0 (filtered)")
        a_.plot(g.snr_db, g.macro_f1_cnn1d, "^--", label="CNN1D baseline")
        a_.plot(g.snr_db, g.macro_f1_resnet_nofilter, "v:", label="ResNet1D, no filter")
        a_.axhline(clean.macro_f1_resnet_ens, color="gray", lw=.8, ls=":")
        a_.set_title(C.NOISE_NAMES[k]); a_.set_xlabel("SNR (dB)"); a_.invert_xaxis(); a_.grid(alpha=.3)
    ax[0].set_ylabel("macro-F1"); ax[0].legend(fontsize=6)
    fig.tight_layout(); fig.savefig(ctx.p("figures", "exp4_noise.png"), dpi=130); plt.close(fig)
    fig, ax = plt.subplots(figsize=(5, 3.4))
    for k in C.NOISE_KINDS:
        g = df[df.noise == k].sort_values("snr_db")
        ax.plot(g.snr_db, g.reject_rate, "o-", label=C.NOISE_NAMES[k])
    ax.axhline(clean.reject_rate, color="gray", ls=":", label="clean")
    ax.set_xlabel("SNR (dB)"); ax.set_ylabel("rejection rate"); ax.invert_xaxis(); ax.legend(fontsize=7); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(ctx.p("figures", "exp4_rejection_rate.png"), dpi=130); plt.close(fig)


# ============================================================================= exp5
def exp5_edge(ctx):
    print("\n=== EXP5: edge model benchmark ===", flush=True)
    ed = ctx.p("edge")
    ms = ctx.models("resnet1d", ctx.main)
    m = ms[0]
    b1 = ctx.models("cnn1d", ctx.main, n=1)[0]
    Xtr, ytr, _, _ = ctx.split(ctx.main, "train")
    xtr = ctx.inp(ctx.main, "train")
    Xva = ctx.inp(ctx.main, "val")
    xte, yte = ctx.inp("A", "test"), ctx.split("A", "test")[1]
    sub = np.random.default_rng(0).permutation(len(ytr))[:ctx.a.prune_n]
    ft = lambda: (xtr[sub], ytr[sub], ctx.a.prune_epochs)
    variants, torch_models = {}, {}
    variants["resnet1d_fp32"] = edge.export_onnx(m, os.path.join(ed, "resnet1d_fp32.onnx")); torch_models["resnet1d_fp32"] = m
    ctrl = edge.prune_model(m, 0.0, finetune=ft())      # control: identical fine-tuning, nothing pruned
    cn = "resnet1d_finetuned_fp32 (control, no pruning)"
    variants[cn] = edge.export_onnx(ctrl, os.path.join(ed, "resnet1d_finetuned_fp32.onnx")); torch_models[cn] = ctrl
    variants["cnn1d_fp32 (baseline B1)"] = edge.export_onnx(b1, os.path.join(ed, "cnn1d_fp32.onnx")); torch_models["cnn1d_fp32 (baseline B1)"] = b1
    try:
        variants["resnet1d_int8"] = edge.quantize_int8(variants["resnet1d_fp32"], os.path.join(ed, "resnet1d_int8.onnx"), Xva)
        torch_models["resnet1d_int8"] = m
    except Exception as e:                                         # pragma: no cover
        print("  INT8 quantisation failed:", e)
    for sp in (0.5, 0.8):
        pm = edge.prune_model(m, sp, finetune=ft())
        nm = f"resnet1d_pruned{int(sp * 100)}_fp32"
        variants[nm] = edge.export_onnx(pm, os.path.join(ed, nm + ".onnx")); torch_models[nm] = pm
        if True:
            try:
                nm2 = f"resnet1d_pruned{int(sp * 100)}_int8"
                variants[nm2] = edge.quantize_int8(variants[nm], os.path.join(ed, nm2 + ".onnx"), Xva); torch_models[nm2] = pm
            except Exception as e:                                 # pragma: no cover
                print("  pruned INT8 failed:", e)
    np.savez_compressed(os.path.join(ed, "test_subset.npz"), x=xte[:500], y=yte[:500])   # for the on-device run
    rows = []
    for name, path in variants.items():
        lg = edge.onnx_logits(path, xte)
        st = edge.sparsity_stats(torch_models[name])
        row = {"variant": name, "macro_f1": E.macro_f1(yte, lg.argmax(1)), "accuracy": E.accuracy(yte, lg.argmax(1)),
               "params": st["params"], "nonzero_params": st["nonzero_params"], **edge.benchmark_file(path, threads=ctx.a.threads)}
        rows.append(row)
        print(f"  {name:28s} F1 {row['macro_f1']:.4f}  {row['file_mb']:.3f} MB  {row['lat_mean_ms']:.2f} ms  {row['peak_rss_mb']:.0f} MB", flush=True)
    ctx.save_csv(pd.DataFrame(rows), "exp5_edge.csv")
    with open(ctx.p("results", "hardware.json"), "w") as f:
        json.dump(edge.hardware_info(), f, indent=2)


# ============================================================================= exp6
def _corrupt_blocks(ctx, X, rec, frac=0.3, block=24, seed=3, snr=6.0):
    """Corrupt random contiguous blocks (default 24 windows = 60 s) of each record with mixed noise
    at `snr` dB SNR to emulate bursts of bad signal quality."""
    rng = np.random.default_rng(seed)
    Xn = X.copy()
    bad = np.zeros(len(X), bool)
    for r in np.unique(rec):
        ii = np.where(rec == r)[0]
        nblk = max(1, int(frac * len(ii) / block))
        for _ in range(nblk):
            s = rng.integers(0, max(1, len(ii) - block))
            sel = ii[s:s + block]
            kind = C.NOISE_KINDS[rng.integers(0, 3)]
            Xn[sel] = ctx.noise.corrupt(X[sel], kind, snr, rng, part="test")
            bad[sel] = True
    return Xn, bad


def exp6_simulator(ctx):
    print("\n=== EXP6: SIMULATED decision support (not a clinical controller) ===", flush=True)
    rp = ctx.A["replay"]
    Xr, yr, recr, t0r, stride = rp["X"], rp["y"], rp["rec"], rp["t0"], float(rp["stride_s"])
    ms = ctx.models("resnet1d", ctx.main)
    gate, tau_u = ctx.gate(), ctx.tau_unc()
    # ---- tune everything on VALIDATION windows only
    Xva_raw, yva, _, _ = get_split(ctx.A, "val", drop_other=False)
    hr_v, cv_v = S.window_rule_features(Xva_raw)
    thr = S.tune_fixed_thresholds(hr_v, cv_v, yva)
    pv = ctx.ens_probs(ms, prepare_input(Xva_raw, True))
    unrel_v = (E.entropy(pv) > tau_u) | gate.flag(sqi_features(Xva_raw))
    tau_risk = S.tune_tau(pv, unrel_v, yva)
    with open(ctx.p("results", "exp6_tuned_thresholds.json"), "w") as f:
        json.dump({"hr_rule(lo,hi)": thr["hr"], "hr_rr_rule(lo,hi,cv)": thr["hr_rr"], "tau_risk": tau_risk,
                   "tau_uncertainty": tau_u}, f, indent=2)
    rows = []
    timeline = None
    for scen in ["clean", "noisy_bursts"]:
        X = Xr if scen == "clean" else _corrupt_blocks(ctx, Xr, recr, snr=ctx.a.burst_snr)[0]
        hr, cv = S.window_rule_features(X)
        p = ctx.ens_probs(ms, prepare_input(X, True))
        unrel = (E.entropy(p) > tau_u) | gate.flag(sqi_features(X))
        acts = {"fixed: heart-rate rule": S.policy_hr(hr, *thr["hr"]),
                "fixed: heart-rate + R-R irregularity rule": S.policy_hr_rr(hr, cv, *thr["hr_rr"]),
                "SmartPace policy (no reject)": S.policy_ml(p, unrel, tau_risk, use_reject=False),
                "SmartPace policy (+ uncertainty/quality reject)": S.policy_ml(p, unrel, tau_risk, use_reject=True)}
        for name, act in acts.items():
            rows.append({"scenario": scen, "policy": name, **S.score(act, yr, recr, t0r, stride)})
        if scen == "noisy_bursts":
            timeline = (yr, acts, recr, t0r)
    df = ctx.save_csv(pd.DataFrame(rows), "exp6_simulator.csv")
    # timeline figure for the first record
    yr_, acts, recr_, t0_ = timeline
    r0 = np.unique(recr_)[0]
    ii = np.where(recr_ == r0)[0][:400]
    fig, ax = plt.subplots(3, 1, figsize=(11, 4.5), sharex=True)
    ax[0].step(t0_[ii] / 60, yr_[ii], where="post"); ax[0].set_yticks(range(-1, 5)); ax[0].set_ylabel("true class")
    for a_, nm in zip(ax[1:], ["fixed: heart-rate rule", "SmartPace policy (+ uncertainty/quality reject)"]):
        a_.step(t0_[ii] / 60, acts[nm][ii], where="post", color="C1"); a_.set_ylabel("action"); a_.set_title(nm, fontsize=8)
        a_.set_yticks(range(5)); a_.set_yticklabels(S.ACTION_NAMES, fontsize=6)
    ax[-1].set_xlabel("time (min) - SIMULATION, record %s, noisy bursts" % r0)
    fig.tight_layout(); fig.savefig(ctx.p("figures", "exp6_timeline.png"), dpi=130); plt.close(fig)
    print(df[["scenario", "policy", "false_triggers_per_hour", "delay_mean_s", "missed_episode_rate",
              "unreliable_time_frac"]].round(3).to_string(index=False))


ALL = {"1": exp1_detection, "2": exp2_cross_dataset, "3": exp3_calibration_explain,
       "4": exp4_noise, "5": exp5_edge, "6": exp6_simulator}
