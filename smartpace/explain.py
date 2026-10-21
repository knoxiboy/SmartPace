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
