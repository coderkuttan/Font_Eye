"""Font embedding network + ArcFace head."""
import math

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from fontset import ROOT, load_classes  # noqa: F401  (re-exported)

EMB_DIM = 256


class FontEmbedder(nn.Module):
    """64x128 grayscale patch -> L2-normalised 256-d font fingerprint."""

    def __init__(self, backbone="resnet18", pretrained=True):
        super().__init__()
        self.net = timm.create_model(backbone, pretrained=pretrained, in_chans=1, num_classes=0)
        if hasattr(self.net, "maxpool"):
            self.net.maxpool = nn.Identity()   # keep spatial detail: glyph strokes are thin
        self.proj = nn.Sequential(nn.Linear(self.net.num_features, EMB_DIM), nn.BatchNorm1d(EMB_DIM))

    def forward(self, x):                      # x in [0,1], shape (N,1,64,128)
        x = (x - 0.5) / 0.5
        return F.normalize(self.proj(self.net(x)), dim=1)


class ArcFace(nn.Module):
    def __init__(self, n_classes, s=30.0, m=0.30):
        super().__init__()
        self.W = nn.Parameter(torch.randn(n_classes, EMB_DIM) * 0.01)
        self.s, self.m = s, m

    def forward(self, emb, labels=None):
        cos = F.linear(emb, F.normalize(self.W, dim=1)).clamp(-1 + 1e-7, 1 - 1e-7)
        if labels is None:
            return cos * self.s
        theta = torch.acos(cos)
        target = torch.cos(theta + self.m)
        # keep the margin monotonic when theta + m > pi
        target = torch.where(theta + self.m < math.pi, target, cos - self.m * math.sin(self.m))
        onehot = F.one_hot(labels, cos.shape[1]).bool()
        return torch.where(onehot, target, cos) * self.s
