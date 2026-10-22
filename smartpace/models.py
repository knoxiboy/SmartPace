"""Neural architectures: ResNet1D (proposed), CNN1D (baseline B1), CNNTransformer (ablation B3)."""
import torch
import torch.nn as nn

from .config import N_CLASSES


class ResBlock(nn.Module):
    def __init__(self, cin, cout, stride, k=7, p=0.1):
        super().__init__()
        self.c1 = nn.Conv1d(cin, cout, k, stride, padding=k // 2, bias=False)
        self.b1 = nn.BatchNorm1d(cout)
        self.drop = nn.Dropout(p)                      # kept active for MC-dropout
        self.c2 = nn.Conv1d(cout, cout, k, 1, padding=k // 2, bias=False)
        self.b2 = nn.BatchNorm1d(cout)
        self.short = None
        if stride != 1 or cin != cout:
            self.short = nn.Sequential(nn.Conv1d(cin, cout, 1, stride, bias=False), nn.BatchNorm1d(cout))
        self.act = nn.ReLU()

    def forward(self, x):
        y = self.act(self.b1(self.c1(x)))
        y = self.b2(self.c2(self.drop(y)))
        return self.act(y + (x if self.short is None else self.short(x)))


class ResNet1D(nn.Module):
    """Lightweight residual 1D CNN (~0.2 M parameters)."""

    def __init__(self, n_classes=N_CLASSES, widths=(16, 32, 48, 64), blocks=(2, 2, 2, 2),
                 k=7, p_block=0.1, p_head=0.3):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(1, widths[0], 15, 2, padding=7, bias=False),
                                  nn.BatchNorm1d(widths[0]), nn.ReLU())
        layers, cin = [], widths[0]
        for i, (w, n) in enumerate(zip(widths, blocks)):
            for j in range(n):
                layers.append(ResBlock(cin, w, stride=(2 if (j == 0 and i > 0) else 1), k=k, p=p_block))
                cin = w
        self.body = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.drop = nn.Dropout(p_head)
        self.fc = nn.Linear(cin, n_classes)

    def extract_features(self, x):
        """Extract penultimate feature embedding (B, D)."""
        x = self.body(self.stem(x))
        return self.pool(x).flatten(1)

    def forward(self, x):
        feat = self.extract_features(x)
        return self.fc(self.drop(feat))


class CNN1D(nn.Module):
    """Shallow baseline B1."""

    def __init__(self, n_classes=N_CLASSES, p=0.3):
        super().__init__()

        def blk(ci, co, k):
            return nn.Sequential(nn.Conv1d(ci, co, k, padding=k // 2), nn.BatchNorm1d(co), nn.ReLU(),
                                 nn.MaxPool1d(4))
        self.f = nn.Sequential(blk(1, 16, 7), blk(16, 32, 5), blk(32, 64, 5),
                               nn.Conv1d(64, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool1d(1))
        self.drop = nn.Dropout(p)
        self.fc = nn.Linear(64, n_classes)

    def extract_features(self, x):
        """Extract penultimate feature embedding (B, 64)."""
        return self.f(x).flatten(1)

    def forward(self, x):
        return self.fc(self.drop(self.extract_features(x)))


class CNNTransformer(nn.Module):
    """Conv front-end + Transformer encoder (the original proposal's architecture, ablation B3)."""

    def __init__(self, n_classes=N_CLASSES, d=64, heads=4, layers=2, p=0.1):
        super().__init__()
        self.front = nn.Sequential(nn.Conv1d(1, 32, 15, 4, padding=7), nn.BatchNorm1d(32), nn.ReLU(),
                                   nn.Conv1d(32, d, 7, 4, padding=3), nn.BatchNorm1d(d), nn.ReLU())
        self.pos = nn.Parameter(torch.zeros(1, 200, d))
        enc = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout=p, batch_first=True)
        self.enc = nn.TransformerEncoder(enc, layers)
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(d, n_classes)

    def extract_features(self, x):
        """Extract penultimate feature embedding (B, d)."""
        z = self.front(x).transpose(1, 2)                  # (B, T, d)
        z = z + self.pos[:, :z.shape[1]]
        return self.enc(z).mean(1)

    def forward(self, x):
        return self.fc(self.drop(self.extract_features(x)))


ARCHS = {"resnet1d": ResNet1D, "cnn1d": CNN1D, "cnntransformer": CNNTransformer}


def build(arch):
    return ARCHS[arch]()


def n_params(model):
    return sum(p.numel() for p in model.parameters())
