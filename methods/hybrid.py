"""
Hybrid — EWC + LwF + Experience Replay

Combines three complementary forgetting-prevention mechanisms:
  1. EWC  — penalises changes to weights important for old tasks
  2. LwF  — distils old predictions so new model mimics the old one
  3. Replay — stores 100 examples per past task and mixes them into training

Run standalone:
    python methods/hybrid.py
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import copy
import json
import time
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data  import get_cifar100_tasks, CLASSES_PER_TASK
from eval  import evaluate
from model import ContinualModel

# ── config ────────────────────────────────────────────────────────────────────
NUM_TASKS    = 5
BATCH_SIZE   = 64
EPOCHS       = 5
LR_HEAD      = 1e-3
EWC_LAMBDA   = 10000
LWF_ALPHA    = 2.0
KD_TEMP      = 2.0
REPLAY_SIZE  = 100   # examples stored per completed task
RESULTS_DIR  = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def get_device():
    if torch.cuda.is_available():         return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


# ── EWC ───────────────────────────────────────────────────────────────────────

class EWC:
    def __init__(self):
        self._params  = []
        self._fishers = []

    def update(self, model, dataloader, device):
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
        loss = torch.tensor(0.0, device=next(model.parameters()).device)
        for params, fishers in zip(self._params, self._fishers):
            for n, p in model.named_parameters():
                if n in params:
                    loss += (fishers[n] * (p - params[n]).pow(2)).sum()
        return loss


# ── Replay Buffer ─────────────────────────────────────────────────────────────

class ReplayBuffer:
    """Stores a fixed number of (x, y) examples per completed task."""
    def __init__(self):
        self.xs = []   # list of tensors (C, H, W)
        self.ys = []   # list of int labels

    def add_task(self, dataset, n=REPLAY_SIZE):
        indices = torch.randperm(len(dataset))[:n].tolist()
        for idx in indices:
            x, y = dataset[idx]
            self.xs.append(x)
            self.ys.append(int(y))

    def sample(self, n):
        """Returns a random batch of (xs, ys) tensors, or (None, None) if empty."""
        if not self.xs:
            return None, None
        n = min(n, len(self.xs))
        idx = random.sample(range(len(self.xs)), n)
        xs = torch.stack([self.xs[i] for i in idx])
        ys = torch.tensor([self.ys[i] for i in idx], dtype=torch.long)
        return xs, ys

    def __len__(self):
        return len(self.xs)


# ── training ──────────────────────────────────────────────────────────────────

def train_task(model, loader, optimizer, device, teacher, n_old, ewc, replay):
    model.train()
    for epoch in range(EPOCHS):
        total = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            # Mix in replay examples if buffer has any
            rx, ry = replay.sample(BATCH_SIZE // 2)
            if rx is not None:
                rx, ry   = rx.to(device), ry.to(device)
                x_all    = torch.cat([x, rx])
                y_all    = torch.cat([y, ry])
            else:
                x_all, y_all = x, y

            outputs = model(x_all)

            # CE loss on current task + replay examples (full 100 classes)
            loss = F.cross_entropy(outputs, y_all)

            # LwF — distil old class predictions on the full batch
            if teacher is not None and n_old > 0:
                with torch.no_grad():
                    teacher_out = teacher(x_all)
                kd = F.kl_div(
                    F.log_softmax(outputs[:, :n_old]      / KD_TEMP, dim=1),
                    F.softmax(teacher_out[:, :n_old]      / KD_TEMP, dim=1),
                    reduction="batchmean",
                ) * (KD_TEMP ** 2)
                loss = loss + LWF_ALPHA * kd

            # EWC — penalise changes to important weights
            loss = loss + EWC_LAMBDA * ewc.penalty(model)

            loss.backward()
            optimizer.step()
            total += loss.item()
        print(f"  Epoch {epoch+1}/{EPOCHS}  loss={total/len(loader):.4f}")


# ── main run ──────────────────────────────────────────────────────────────────

def run(tasks=None, device=None):
    if device is None: device = get_device()
    if tasks  is None: tasks  = get_cifar100_tasks(NUM_TASKS)

    print(f"\n{'='*50}\n  Method: HYBRID (EWC + LwF + Replay)\n{'='*50}")

    model         = ContinualModel().to(device)
    ewc           = EWC()
    replay        = ReplayBuffer()
    teacher       = None
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

        n_old = task_id * CLASSES_PER_TASK
        train_task(model, train_loader, optimizer, device, teacher, n_old, ewc, replay)

        # Update EWC, teacher, and replay buffer after each task
        ewc.update(model, train_loader, device)

        teacher = copy.deepcopy(model)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)

        replay.add_task(train_ds)
        print(f"  Replay buffer: {len(replay)} examples total")

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
        "method": "hybrid",
        "acc_matrix": acc_matrix.tolist(),
        "aa":  round(aa*100, 2),
        "bwt": round(bwt*100, 2),
        "task_times": [round(t, 2) for t in task_times],
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "hybrid.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved → {RESULTS_DIR}/hybrid.json")
    return results


if __name__ == "__main__":
    run()
