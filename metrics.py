"""
metrics.py — Evaluation metrics for continual learning.
    • Average Accuracy (AA)
    • Backward Transfer (BWT)
    • Peak RAM per task
    • Wall-clock time per task
"""

import time
import psutil
import os
import torch
import numpy as np
from torch.utils.data import DataLoader


# ── accuracy ─────────────────────────────────────────────────────────────────

def evaluate(model, loader: DataLoader, device: torch.device) -> float:
    """Top-1 accuracy on a DataLoader (global class indices)."""
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total   += y.size(0)
    return correct / total if total > 0 else 0.0


def compute_aa_bwt(acc_matrix: np.ndarray) -> tuple[float, float]:
    """
    acc_matrix[i, j] = accuracy on task j after training on task i.
    Shape: (num_tasks, num_tasks), lower triangle only filled.

    AA  = mean of the last row (performance after all tasks)
    BWT = mean of (acc_matrix[T-1, j] - acc_matrix[j, j]) for j < T-1
    """
    T = acc_matrix.shape[0]
    aa  = float(np.mean(acc_matrix[T - 1, :T]))
    bwt = float(np.mean([
        acc_matrix[T - 1, j] - acc_matrix[j, j]
        for j in range(T - 1)
    ])) if T > 1 else 0.0
    return aa, bwt


# ── resource tracking ─────────────────────────────────────────────────────────

class ResourceTracker:
    """Context manager that records peak RAM and elapsed time."""

    def __init__(self):
        self._proc = psutil.Process(os.getpid())
        self.peak_ram_mb = 0.0
        self.elapsed_sec = 0.0
        self._start_time = None
        self._start_ram  = 0.0

    def __enter__(self):
        self._start_time = time.perf_counter()
        self._start_ram  = self._proc.memory_info().rss / 1e6
        return self

    def __exit__(self, *_):
        self.elapsed_sec = time.perf_counter() - self._start_time
        end_ram = self._proc.memory_info().rss / 1e6
        self.peak_ram_mb = end_ram - self._start_ram

    @property
    def summary(self) -> dict:
        return {
            "time_sec":    round(self.elapsed_sec, 2),
            "ram_delta_mb": round(self.peak_ram_mb, 2),
        }
