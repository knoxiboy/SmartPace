"""Integrated Gradients and the three explanation evaluations required by the project:
stability, faithfulness (deletion / insertion) and physiological plausibility (P / QRS / T overlap)."""
import numpy as np
import torch
from scipy.stats import spearmanr

from .config import CLASSES, FS
from .preprocess import detect_rpeaks
from .train import get_device, predict_logits

_trapz = getattr(np, "trapezoid", None) or np.trapz


def integrated_gradients(model, x, target=None, steps=32, device=None, bs=32):
    """Integrated Gradients w.r.t. the model input (zero baseline = the z-score mean).

    x: np (N,1,L). target: class index per window (default: predicted class).
    Returns attributions np (N, L) of the target-class logit."""
    device = device or get_device()
    model = model.to(device).eval()
    if target is None:
        target = predict_logits(model, x, device).argmax(1)
    out = []
    alphas = (torch.arange(steps, dtype=torch.float32, device=device) + 0.5) / steps   # midpoint rule
    for i in range(0, len(x), bs):
        xb = torch.from_numpy(x[i:i + bs]).to(device)
        tb = torch.from_numpy(np.asarray(target[i:i + bs])).long().to(device)
        grad_sum = torch.zeros_like(xb)
        for a in alphas:
            xi = (a * xb).requires_grad_(True)
            logit = model(xi).gather(1, tb[:, None]).sum()
            g, = torch.autograd.grad(logit, xi)
            grad_sum += g
        out.append((xb * grad_sum / steps).squeeze(1).detach().cpu().numpy())
    return np.concatenate(out)


def smooth_abs(a, width=25):
    k = np.ones(width) / width
    return np.stack([np.convolve(np.abs(r), k, mode="same") for r in a])


# ----------------------------------------------------------------------------- stability
def stability(model, x, sigma=0.05, steps=32, topk_frac=0.1, seed=0):
    """Attribution stability under small input noise (sigma in z-score units).
    Returns mean Spearman correlation and mean top-k Jaccard overlap between the attribution
    maps of the clean and the perturbed window (same target class)."""
    rng = np.random.default_rng(seed)
    pred = predict_logits(model, x).argmax(1)
    a1 = smooth_abs(integrated_gradients(model, x, pred, steps))
    xn = (x + sigma * rng.standard_normal(x.shape)).astype(np.float32)
    a2 = smooth_abs(integrated_gradients(model, xn, pred, steps))
    k = int(topk_frac * x.shape[-1])
    rho, jac = [], []
    for u, v in zip(a1, a2):
        rho.append(spearmanr(u, v)[0])
        tu, tv = set(np.argsort(-u)[:k]), set(np.argsort(-v)[:k])
        jac.append(len(tu & tv) / len(tu | tv))
    return {"spearman": float(np.nanmean(rho)), "topk_jaccard": float(np.mean(jac))}


# ----------------------------------------------------------------------------- faithfulness
def deletion_insertion(model, x, attr, fractions=None, seed=0):
    """Deletion: zero out the most attributed samples first; the probability of the originally
    predicted class should fall quickly. Insertion: start from zeros, insert the most attributed
    samples; the probability should rise quickly. Compared with a random-order control.
    Returns dict of curve AUCs (mean over windows)."""
    fractions = np.linspace(0, 0.5, 11) if fractions is None else fractions
    rng = np.random.default_rng(seed)
    pred = predict_logits(model, x).argmax(1)
    score = smooth_abs(attr)
    L = x.shape[-1]

    def curve(order_fn, mode):
        res = []
        for f in fractions:
            xm = x.copy()
            for i in range(len(x)):
                idx = order_fn(i)[:int(f * L)]
                if mode == "del":
                    xm[i, 0, idx] = 0.0
                else:
                    z = np.zeros_like(xm[i, 0])
                    z[idx] = x[i, 0, idx]
                    xm[i, 0] = z
            lg = predict_logits(model, xm)
            p = torch.softmax(torch.from_numpy(lg), 1).numpy()
            res.append(p[np.arange(len(x)), pred])
        return np.array(res)          # (F, N)

    ig_order = lambda i: np.argsort(-score[i])
    rnd = [rng.permutation(L) for _ in range(len(x))]
    rnd_order = lambda i: rnd[i]
    auc = lambda c: float(_trapz(c.mean(1), fractions) / (fractions[-1] - fractions[0]))
    return {"deletion_auc_ig": auc(curve(ig_order, "del")), "deletion_auc_random": auc(curve(rnd_order, "del")),
            "insertion_auc_ig": auc(curve(ig_order, "ins")), "insertion_auc_random": auc(curve(rnd_order, "ins"))}


# ----------------------------------------------------------------------------- plausibility
REGIONS = {"P": (-0.20, -0.08), "QRS": (-0.06, 0.06), "T": (0.10, 0.40)}   # seconds relative to R


def region_masks(raw_window, fs=FS):
    """Boolean masks (P, QRS, T) over the window, built from detected R-peaks of the RAW window."""
    L = len(raw_window)
    masks = {k: np.zeros(L, bool) for k in REGIONS}
    for r in detect_rpeaks(raw_window, fs):
        for k, (a, b) in REGIONS.items():
            lo, hi = max(0, int(r + a * fs)), min(L, int(r + b * fs))
            if hi > lo:
                masks[k][lo:hi] = True
    return masks


def plausibility(attr, raw, preds):
    """Enrichment of attribution mass inside P / QRS / T regions, per predicted class.
    enrichment = (share of |attribution| inside region) / (share of samples inside region);
    > 1 means the explanation concentrates on that waveform region."""
    rows = {}
    for i in range(len(attr)):
        m = region_masks(raw[i])
        a = np.abs(attr[i])
        tot = a.sum() + 1e-12
        for k in REGIONS:
            area = m[k].mean()
            if area == 0:
                continue
            rows.setdefault(CLASSES[preds[i]], {}).setdefault(k, []).append((a[m[k]].sum() / tot) / area)
    return {c: {k: float(np.mean(v)) for k, v in d.items()} for c, d in rows.items()}


def completeness_gap(model, x, steps=64):
    """Sanity check of IG: sum(attr) should equal logit(x) - logit(0). Returns mean relative gap."""
    pred = predict_logits(model, x).argmax(1)
    a = integrated_gradients(model, x, pred, steps)
    fx = predict_logits(model, x)[np.arange(len(x)), pred]
    f0 = predict_logits(model, np.zeros_like(x))[np.arange(len(x)), pred]
    return float(np.mean(np.abs(a.sum(1) - (fx - f0)) / (np.abs(fx - f0) + 1e-6)))


# -----------------------------------------------------------------------------
# Improvement 4: Reliability-Tested Explanations & Stress Testing
# -----------------------------------------------------------------------------
def individual_stability(model, x, target=None, sigma=0.03, steps=32, topk_frac=0.1, seed=0, device=None):
    """Computes per-window attribution stability metrics under physiologically plausible perturbations.
    
    Returns array of Spearman correlations and top-k Jaccard overlaps for each window,
    along with clean and perturbed smoothed attributions.
    """
    rng = np.random.default_rng(seed)
    if target is None:
        target = predict_logits(model, x, device).argmax(1)
    a1 = smooth_abs(integrated_gradients(model, x, target, steps, device))
    xn = (x + sigma * rng.standard_normal(x.shape)).astype(np.float32)
    a2 = smooth_abs(integrated_gradients(model, xn, target, steps, device))
    k = max(1, int(topk_frac * x.shape[-1]))

    rhos, jacs = [], []
    for u, v in zip(a1, a2):
        r, _ = spearmanr(u, v)
        rhos.append(float(r) if np.isfinite(r) else 0.0)
        tu, tv = set(np.argsort(-u)[:k]), set(np.argsort(-v)[:k])
        jac = len(tu & tv) / (len(tu | tv) + 1e-12)
        jacs.append(float(jac))

    return {
        "spearman": np.array(rhos, dtype=np.float32),
        "topk_jaccard": np.array(jacs, dtype=np.float32),
        "clean_attr": a1,
        "perturbed_attr": a2,
    }


def counterfactual_stress_test(
    model,
    x,
    raw_windows,
    y_true,
    probs=None,
    steps=32,
    topk_frac=0.10,
    sigma=0.03,
    seed=0,
    device=None,
):
    """Performs per-sample explanation stress testing and categorizes failure modes.

    Categorizes each window into:
    - Case A: Correct prediction, stable explanation
    - Case B: Incorrect prediction, suspicious explanation (unstable or concentrates on background)
    - Case C: Confident prediction, unstable explanation (high softmax confidence, but fragile explanation)
    - Case D: Other / borderline
    """
    device = device or get_device()
    n = len(x)
    y_true = np.asarray(y_true, dtype=np.int64)

    if probs is None:
        lg = predict_logits(model, x, device)
        probs = torch.softmax(torch.from_numpy(lg), dim=1).numpy()

    preds = probs.argmax(axis=1)
    confs = probs.max(axis=1)

    # 1. Per-window stability
    stab = individual_stability(model, x, target=preds, sigma=sigma, steps=steps, topk_frac=topk_frac, seed=seed, device=device)
    rhos = stab["spearman"]
    jacs = stab["topk_jaccard"]
    attr = stab["clean_attr"]

    # 2. Counterfactual region perturbation test
    # Highlighted region perturbation (top k% most attributed) vs control region (bottom k%)
    L = x.shape[-1]
    k_len = max(1, int(topk_frac * L))
    x_pert_high = x.copy()
    x_pert_ctrl = x.copy()

    for i in range(n):
        order = np.argsort(-attr[i])
        high_idx = order[:k_len]
        ctrl_idx = order[-k_len:]
        x_pert_high[i, 0, high_idx] = 0.0
        x_pert_ctrl[i, 0, ctrl_idx] = 0.0

    p_high = torch.softmax(torch.from_numpy(predict_logits(model, x_pert_high, device)), dim=1).numpy()
    p_ctrl = torch.softmax(torch.from_numpy(predict_logits(model, x_pert_ctrl, device)), dim=1).numpy()
    pred_high = p_high.argmax(axis=1)
    pred_ctrl = p_ctrl.argmax(axis=1)

    # 3. Physiological region localization and overlap
    records = []
    for i in range(n):
        m = region_masks(raw_windows[i])
        a = attr[i]
        tot = a.sum() + 1e-12

        share_qrs = float(a[m["QRS"]].sum() / tot) if m["QRS"].any() else 0.0
        share_p = float(a[m["P"]].sum() / tot) if m["P"].any() else 0.0
        share_t = float(a[m["T"]].sum() / tot) if m["T"].any() else 0.0
        phys_share = share_qrs + share_p + share_t
        bg_share = 1.0 - phys_share

        qrs_enr = (share_qrs / m["QRS"].mean()) if m["QRS"].mean() > 0 else 0.0
        p_enr = (share_p / m["P"].mean()) if m["P"].mean() > 0 else 0.0
        t_enr = (share_t / m["T"].mean()) if m["T"].mean() > 0 else 0.0

        is_corr = bool(preds[i] == y_true[i])
        conf_i = float(confs[i])
        rho_i = float(rhos[i])
        jac_i = float(jacs[i])

        cf_flip_high = bool(pred_high[i] != preds[i])
        cf_flip_ctrl = bool(pred_ctrl[i] != preds[i])

        # Failure mode categorization
        if is_corr and (rho_i >= 0.70) and (phys_share >= 0.25):
            case = "Case A: Correct, stable explanation"
        elif (not is_corr) and ((bg_share >= 0.70) or (rho_i < 0.60)):
            case = "Case B: Incorrect, suspicious explanation"
        elif (conf_i >= 0.85) and ((rho_i < 0.60) or (jac_i < 0.35)):
            case = "Case C: Confident, unstable explanation"
        else:
            case = "Case D: Standard / intermediate"

        records.append({
            "window_idx": i,
            "true_class": CLASSES[y_true[i]],
            "predicted_class": CLASSES[preds[i]],
            "confidence": conf_i,
            "is_correct": is_corr,
            "stability_spearman": rho_i,
            "topk_jaccard": jac_i,
            "share_qrs": share_qrs,
            "share_p": share_p,
            "share_t": share_t,
            "physiological_share": phys_share,
            "background_share": bg_share,
            "qrs_enrichment": float(qrs_enr),
            "p_enrichment": float(p_enr),
            "t_enrichment": float(t_enr),
            "counterfactual_flip_highlighted": cf_flip_high,
            "counterfactual_flip_control": cf_flip_ctrl,
            "failure_case": case,
        })

    return records, stab


def evaluate_explanation_error_warning(y_true, probs, stability_spearman):
    """Evaluates whether explanation instability improves error detection AUROC over confidence alone."""
    from sklearn.metrics import roc_auc_score

    y_true = np.asarray(y_true)
    preds = probs.argmax(axis=1)
    is_error = (preds != y_true).astype(int)

    if len(np.unique(is_error)) < 2:
        return {"auroc_confidence": 0.5, "auroc_explanation_instability": 0.5, "auroc_combined": 0.5}

    confs = probs.max(axis=1)
    unc_score = 1.0 - confs
    instability_score = 1.0 - np.clip(stability_spearman, -1.0, 1.0) / 2.0  # mapped to [0, 1]

    # Normalize scores
    u_norm = (unc_score - unc_score.min()) / (unc_score.max() - unc_score.min() + 1e-12)
    s_norm = (instability_score - instability_score.min()) / (instability_score.max() - instability_score.min() + 1e-12)
    comb_score = 0.65 * u_norm + 0.35 * s_norm

    auc_conf = float(roc_auc_score(is_error, unc_score))
    auc_instab = float(roc_auc_score(is_error, instability_score))
    auc_comb = float(roc_auc_score(is_error, comb_score))

    return {
        "auroc_confidence_only": auc_conf,
        "auroc_explanation_instability_only": auc_instab,
        "auroc_combined_warning": auc_comb,
        "auroc_gain_over_confidence": auc_comb - auc_conf,
    }
