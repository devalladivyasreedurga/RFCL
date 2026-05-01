"""
EWC — Elastic Weight Consolidation [Kirkpatrick et al. 2017]

After each task, computes the Fisher Information Matrix diagonal to identify
important weights, then penalises large changes to them during future tasks.

Run standalone:
    python methods/ewc.py
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data  import get_cifar100_tasks, CLASSES_PER_TASK
from eval  import evaluate
from model import ContinualModel

# ── config ────────────────────────────────────────────────────────────────────
NUM_TASKS   = 5
BATCH_SIZE  = 64
EPOCHS      = 5
LR_HEAD     = 1e-3
LR_BACKBONE = 1e-4
EWC_LAMBDA  = 10000       # regularisation strength — higher = less forgetting
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def get_device():
    if torch.cuda.is_available():         return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


# ── EWC class ─────────────────────────────────────────────────────────────────

class EWC:
    """
    Stores the optimal weights and Fisher diagonal after each task.
    Accumulates across tasks so all previous tasks are protected.
    """
    def __init__(self):
        self._params  = []   # list of param snapshots, one per task
        self._fishers = []   # list of Fisher diagonals, one per task

    def update(self, model, dataloader, device):
        """Call once after finishing a task."""
        params  = {n: p.clone().detach() for n, p in model.named_parameters() if p.requires_grad}
        fishers = {n: torch.zeros_like(p) for n, p in model.named_parameters() if p.requires_grad}

        model.eval()
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            model.zero_grad()
            F.cross_entropy(model(x), y).backward()
            for n, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fishers[n] += p.grad.data.pow(2)

        for n in fishers:
            fishers[n] /= len(dataloader)

        self._params.append(params)
        self._fishers.append(fishers)

    def penalty(self, model) -> torch.Tensor:
        """Sum of Fisher-weighted squared deviations from all past optima."""
        loss = torch.tensor(0.0, device=next(model.parameters()).device)
        for params, fishers in zip(self._params, self._fishers):
            for n, p in model.named_parameters():
                if n in params:
                    loss += (fishers[n] * (p - params[n]).pow(2)).sum()
        return loss


# ── training ──────────────────────────────────────────────────────────────────

def train_task(model, loader, optimizer, device, ewc, task_id):
    model.train()
    for epoch in range(EPOCHS):
        total = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            loss = F.cross_entropy(model(x), y)
            loss = loss + EWC_LAMBDA * ewc.penalty(model)

            loss.backward()
            optimizer.step()
            total += loss.item()
        print(f"  Epoch {epoch+1}/{EPOCHS}  loss={total/len(loader):.4f}")


# ── main run ──────────────────────────────────────────────────────────────────

def run(tasks=None, device=None):
    if device is None: device = get_device()
    if tasks  is None: tasks  = get_cifar100_tasks(NUM_TASKS)

    print(f"\n{'='*50}\n  Method: EWC\n{'='*50}")

    model         = ContinualModel().to(device)
    ewc           = EWC()
    train_loaders = []
    test_loaders  = []
    acc_matrix    = np.zeros((NUM_TASKS, NUM_TASKS))
    task_times    = []

    for task_id, (train_ds, test_ds) in enumerate(tasks):
        print(f"\n── Task {task_id+1}/{NUM_TASKS}  "
              f"(classes {task_id*CLASSES_PER_TASK}–{(task_id+1)*CLASSES_PER_TASK-1}) ──")
        t0 = time.time()

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
        test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
        train_loaders.append(train_loader)
        test_loaders.append(test_loader)

        optimizer = torch.optim.Adam(model.classifier.parameters(), lr=LR_HEAD)

        train_task(model, train_loader, optimizer, device, ewc, task_id)

        # Compute Fisher and snapshot weights after this task
        ewc.update(model, train_loader, device)

        model.eval()
        print("  Accuracies →", end="")
        for prev in range(task_id + 1):
            acc = evaluate(model, [test_loaders[prev]], device)[0]
            acc_matrix[task_id, prev] = acc
            print(f"  T{prev+1}: {acc*100:.1f}%", end="")
        print()

        elapsed = time.time() - t0
        task_times.append(elapsed)
        print(f"  Time: {elapsed:.1f}s")

    T   = NUM_TASKS
    aa  = float(np.mean(acc_matrix[T-1, :T]))
    bwt = float(np.mean([acc_matrix[T-1, j] - acc_matrix[j, j] for j in range(T-1)]))

    print(f"\n  AA: {aa*100:.2f}%   BWT: {bwt*100:.2f}%")

    results = {
        "method": "ewc",
        "acc_matrix": acc_matrix.tolist(),
        "aa":  round(aa*100, 2),
        "bwt": round(bwt*100, 2),
        "task_times": [round(t, 2) for t in task_times],
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "ewc.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved → {RESULTS_DIR}/ewc.json")
    return results


if __name__ == "__main__":
    run()
