"""
model.py — Frozen ResNet-18 backbone with an incrementally expanding linear head.
Only the head is trained; the backbone is fixed throughout.
"""

import torch
import torch.nn as nn
from torchvision import models
from data import CLASSES_PER_TASK


class ContinualResNet(nn.Module):
    """
    ResNet-18 with frozen pretrained backbone.
    The classifier head expands by CLASSES_PER_TASK neurons after each task.
    """

    def __init__(self, classes_per_task: int = CLASSES_PER_TASK):
        super().__init__()
        self.classes_per_task = classes_per_task

        # ── backbone (frozen) ────────────────────────────────────────────────
        # Try pretrained weights; fall back to random init if offline
        try:
            backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            print("Loaded pretrained ResNet-18 backbone.")
        except Exception:
            backbone = models.resnet18(weights=None)
            print("WARNING: pretrained weights unavailable — using random init.")
        self.feature_dim = backbone.fc.in_features   # 512

        # Remove the original classification head
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        for param in self.backbone.parameters():
            param.requires_grad = False

        # ── head (trained) ───────────────────────────────────────────────────
        # Starts empty; call expand_head() before each task
        self.head = nn.Linear(self.feature_dim, 0)
        self._num_classes = 0

    # ── helpers ──────────────────────────────────────────────────────────────

    def expand_head(self, task_id: int):
        """
        Add CLASSES_PER_TASK output neurons for task_id.
        Previous weights are preserved exactly.
        """
        new_total = (task_id + 1) * self.classes_per_task
        if new_total <= self._num_classes:
            return  # already expanded

        old_head = self.head
        new_head = nn.Linear(self.feature_dim, new_total)

        # Copy old weights
        if self._num_classes > 0:
            with torch.no_grad():
                new_head.weight[:self._num_classes] = old_head.weight
                new_head.bias[:self._num_classes]   = old_head.bias

        self.head = new_head
        self._num_classes = new_total

    def get_trainable_params(self):
        """Only the head parameters are trainable."""
        return list(self.head.parameters())

    # ── forward ──────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            feats = self.backbone(x)          # (B, 512, 1, 1)
        feats = feats.flatten(1)              # (B, 512)
        return self.head(feats)               # (B, num_classes_so_far)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return backbone features (no grad)."""
        with torch.no_grad():
            feats = self.backbone(x)
        return feats.flatten(1)


if __name__ == "__main__":
    model = ContinualResNet()
    for t in range(5):
        model.expand_head(t)
        x = torch.randn(4, 3, 32, 32)
        out = model(x)
        print(f"Task {t+1}: head output shape = {out.shape}")
    print("Model OK.")
