"""
Naive Baseline — cross-entropy only, no forgetting prevention.
Establishes the worst-case forgetting reference.

Run standalone:
    python methods/naive.py
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
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def get_device():
    if torch.cuda.is_available():       return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


# ── training ──────────────────────────────────────────────────────────────────

def train_task(model, loader, optimizer, device, task_id):
    model.train()
    for epoch in range(EPOCHS):
        total = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            optimizer.step()
            total += loss.item()
        print(f"  Epoch {epoch+1}/{EPOCHS}  loss={total/len(loader):.4f}")


# ── main run ──────────────────────────────────────────────────────────────────

def run(tasks=None, device=None):
    if device is None: device = get_device()
    if tasks  is None: tasks  = get_cifar100_tasks(NUM_TASKS)

    print(f"\n{'='*50}\n  Method: NAIVE\n{'='*50}")

    model         = ContinualModel().to(device)
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

        train_task(model, train_loader, optimizer, device, task_id)

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
        "method": "naive",
        "acc_matrix": acc_matrix.tolist(),
        "aa":  round(aa*100, 2),
        "bwt": round(bwt*100, 2),
        "task_times": [round(t, 2) for t in task_times],
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "naive.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved → {RESULTS_DIR}/naive.json")
    return results


if __name__ == "__main__":
    run()
