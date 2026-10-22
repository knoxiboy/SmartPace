"""Training, inference (plain / MC-dropout / ensemble) and model registry."""
import copy
import os
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from scipy.special import softmax

from . import models as M
from .config import N_CLASSES
from .evaluation import macro_f1
from .preprocess import prepare_input


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class TrainCfg:
    epochs: int = 30
    batch_size: int = 128
    lr: float = 2e-3
    weight_decay: float = 1e-4
    aug_noise_p: float = 0.3        # fraction of training windows corrupted with noise each epoch
    patience: int = 8
    verbose: bool = True


def class_weights(y):
    cnt = np.bincount(y, minlength=N_CLASSES).astype(float)
    w = np.where(cnt > 0, 1.0 / np.sqrt(np.maximum(cnt, 1)), 0.0)
    return torch.tensor(w / w[cnt > 0].mean(), dtype=torch.float32)


@torch.no_grad()
def predict_logits(model, x, device=None, bs=512):
    """x: np (N,1,L) float32 -> logits np (N,C). Deterministic (eval mode)."""
    device = device or get_device()
    model = model.to(device).eval()
    out = []
    for i in range(0, len(x), bs):
        out.append(model(torch.from_numpy(x[i:i + bs]).to(device)).cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def extract_features(model, x, device=None, bs=512):
    """x: np (N,1,L) float32 -> features np (N,D). Deterministic (eval mode)."""
    device = device or get_device()
    model = model.to(device).eval()
    out = []
    for i in range(0, len(x), bs):
        batch = torch.from_numpy(x[i:i + bs]).to(device)
        if hasattr(model, "extract_features"):
            feat = model.extract_features(batch)
        else:
            feat = model(batch)
        out.append(feat.cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def mc_dropout_probs(model, x, T=20, device=None, bs=512):
    """Monte-Carlo dropout: T stochastic passes with dropout on, BatchNorm in eval mode.
    Returns probs (T, N, C)."""
    device = device or get_device()
    model = model.to(device).eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()
    res = []
    for _ in range(T):
        out = []
        for i in range(0, len(x), bs):
            out.append(torch.softmax(model(torch.from_numpy(x[i:i + bs]).to(device)), 1).cpu().numpy())
        res.append(np.concatenate(out))
    model.eval()
    return np.stack(res)


def train_model(arch, Xtr, ytr, Xva, yva, cfg=None, noise_bank=None, filt=True, seed=0, device=None):
    """Train one model. Xtr/Xva are RAW windows (N, L); filtering + z-score are applied here."""
    cfg = cfg or TrainCfg()
    device = device or get_device()
    set_seed(seed)
    rng = np.random.default_rng(seed)
    model = M.build(arch).to(device)
    x_clean = prepare_input(Xtr, filt)
    x_val = prepare_input(Xva, filt)
    yt = torch.from_numpy(ytr).long()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=cfg.lr, epochs=cfg.epochs, steps_per_epoch=int(np.ceil(len(ytr) / cfg.batch_size)))
    lossf = nn.CrossEntropyLoss(weight=class_weights(ytr).to(device))
    best, best_state, bad, hist = -1, None, 0, []
    for ep in range(cfg.epochs):
        x_ep = x_clean
        if noise_bank is not None and cfg.aug_noise_p > 0:
            idx = np.where(rng.random(len(Xtr)) < cfg.aug_noise_p)[0]
            if len(idx):
                x_ep = x_clean.copy()
                x_ep[idx] = prepare_input(noise_bank.corrupt_mixed(Xtr[idx], rng, part="train"), filt)
        xt = torch.from_numpy(x_ep)
        model.train()
        perm = torch.randperm(len(yt))
        tot = 0.0
        for i in range(0, len(perm), cfg.batch_size):
            b = perm[i:i + cfg.batch_size]
            if len(b) < 2:
                continue
            xb, yb = xt[b].to(device), yt[b].to(device)
            xb = xb * (0.8 + 0.4 * torch.rand(len(b), 1, 1, device=device))          # amplitude
            shift = int(np.random.randint(-100, 101))
            xb = torch.roll(xb, shift, dims=2)                                       # time shift
            xb = xb + 0.02 * torch.randn_like(xb)
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(b)
        val_pred = predict_logits(model, x_val, device).argmax(1)
        f1 = macro_f1(yva, val_pred)
        hist.append({"epoch": ep + 1, "loss": tot / len(yt), "val_macro_f1": f1})
        if cfg.verbose:
            print(f"  [{arch} s{seed}] ep {ep + 1:02d} loss {tot / len(yt):.4f} val_macroF1 {f1:.4f}", flush=True)
        if f1 > best:
            best, bad, best_state = f1, 0, copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    model.load_state_dict(best_state)
    return model.cpu().eval(), hist


# ----------------------------------------------------------------------------- registry
def save_model(path, model, arch, meta=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"arch": arch, "state": model.state_dict(), "meta": meta or {}}, path)


def load_model(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = M.build(ck["arch"])
    m.load_state_dict(ck["state"])
    return m.eval(), ck.get("meta", {})


def get_or_train(model_dir, name, arch, Xtr, ytr, Xva, yva, cfg, noise_bank=None, filt=True, seed=0):
    path = os.path.join(model_dir, name + ".pt")
    if os.path.exists(path):
        return load_model(path)[0]
    model, hist = train_model(arch, Xtr, ytr, Xva, yva, cfg, noise_bank, filt, seed)
    save_model(path, model, arch, {"history": hist, "filt": filt, "seed": seed})
    return model


def ensemble_logits(models_, x):
    """list of models -> logits (M, N, C)."""
    return np.stack([predict_logits(m, x) for m in models_])


def probs_from_logits(logits):
    return softmax(logits, axis=-1)
