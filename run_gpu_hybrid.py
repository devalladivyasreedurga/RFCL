"""
Single-file LwF + EWC continual learning on CIFAR-100.
Optimized for GPU — just run: python run_gpu_hybrid.py
"""

import copy
import json
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
NUM_TASKS   = 5
EPOCHS      = 5
BATCH_SIZE  = 128
LR_HEAD     = 1e-3
LR_BACKBONE = 1e-4
LWF_ALPHA   = 1.0     # FIX: was 2.0 — 1.0 is more stable
TEMPERATURE = 2.0     # FIX: was 3.0
EWC_LAMBDA  = 1000    # FIX: was 10 — too weak to prevent forgetting
NUM_WORKERS = 4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ──────────────────────────────────────────────
# DATA  (FIX: load test split, add augmentation)
# ──────────────────────────────────────────────
CLASSES_PER_TASK = 100 // NUM_TASKS

def get_cifar100_tasks(num_tasks=NUM_TASKS):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    test_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    train_data = datasets.CIFAR100(root='./data', train=True,  download=True, transform=train_transform)
    test_data  = datasets.CIFAR100(root='./data', train=False, download=True, transform=test_transform)

    tasks = []
    for t in range(num_tasks):
        s, e = t * CLASSES_PER_TASK, (t + 1) * CLASSES_PER_TASK
        train_idx = [i for i, (_, y) in enumerate(train_data) if s <= y < e]
        test_idx  = [i for i, (_, y) in enumerate(test_data)  if s <= y < e]
        tasks.append((Subset(train_data, train_idx), Subset(test_data, test_idx)))
    return tasks

# ──────────────────────────────────────────────
# MODEL  (backbone fully frozen — only classifier head is trained)
# ──────────────────────────────────────────────
class ContinualModel(nn.Module):
    def __init__(self, num_classes=100):          # FIX: was _init_
        super().__init__()                         # FIX: was _init_
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])
        self.feature_dim = backbone.fc.in_features

        # Freeze entire backbone — stable features means EWC+LwF work much better
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def forward(self, x):
        features = self.feature_extractor(x).view(-1, self.feature_dim)
        features = F.normalize(features, dim=1)
        weights  = F.normalize(self.classifier.weight, dim=1)
        return F.linear(features, weights)

    def extract_features(self, x):
        with torch.no_grad():
            return self.feature_extractor(x).view(-1, self.feature_dim)

# ──────────────────────────────────────────────
# EWC  (FIX: accumulates across ALL tasks, not just last)
# ──────────────────────────────────────────────
class EWC:
    def __init__(self):                            # FIX: was _init_
        self._params  = []
        self._fishers = []

    def update(self, model, dataloader, device):
        """Call once after each task finishes."""
        params  = {n: p.clone().detach() for n, p in model.named_parameters() if p.requires_grad}
        fishers = {n: torch.zeros_like(p)  for n, p in model.named_parameters() if p.requires_grad}

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

# ──────────────────────────────────────────────
# TRAIN  (FIX: LwF applied every epoch, mixed precision used properly)
# ──────────────────────────────────────────────
def train_task(model, train_loader, optimizer, device,
               teacher_model=None, ewc=None, n_old=0, scaler=None):
    model.train()
    use_amp = scaler is not None

    for epoch in range(EPOCHS):
        total = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(x)
                loss    = F.cross_entropy(outputs, y)

                # FIX: was skipped on epoch 0 — now applied every epoch
                if teacher_model is not None and n_old > 0:
                    with torch.no_grad():
                        teacher_out = teacher_model(x)
                    kd = F.kl_div(
                        F.log_softmax(outputs[:, :n_old]    / TEMPERATURE, dim=1),
                        F.softmax(teacher_out[:, :n_old]    / TEMPERATURE, dim=1),
                        reduction='batchmean',
                    ) * (TEMPERATURE ** 2)
                    loss = loss + LWF_ALPHA * kd

                if ewc is not None:
                    loss = loss + EWC_LAMBDA * ewc.penalty(model)

            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            total += loss.item()
        print(f"  Epoch {epoch+1}/{EPOCHS}  loss={total/len(train_loader):.4f}")

# ──────────────────────────────────────────────
# EVAL
# ──────────────────────────────────────────────
def evaluate(model, loaders, device):
    model.eval()
    results = []
    with torch.no_grad():
        for loader in loaders:
            correct = total = 0
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                correct += (model(x).argmax(dim=1) == y).sum().item()
                total   += y.size(0)
            results.append(correct / total)
    return results

# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
if __name__ == "__main__":                         # FIX: was _name_ / _main_
    import numpy as np

    tasks  = get_cifar100_tasks()
    model  = ContinualModel().to(device)
    ewc    = EWC()                                 # FIX: persistent across tasks
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    train_loaders = []
    test_loaders  = []
    teacher_model = None

    acc_matrix = np.zeros((NUM_TASKS, NUM_TASKS))

    for task_id, (train_ds, test_ds) in enumerate(tasks):
        print(f"\n{'='*40}")
        print(f"  Task {task_id+1}/{NUM_TASKS}  "
              f"(classes {task_id*CLASSES_PER_TASK}–{(task_id+1)*CLASSES_PER_TASK-1})")
        print(f"{'='*40}")
        start = time.time()

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=NUM_WORKERS, pin_memory=True)
        test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                                  num_workers=NUM_WORKERS, pin_memory=True)
        train_loaders.append(train_loader)
        test_loaders.append(test_loader)

        # backbone is frozen — only train classifier head
        optimizer = torch.optim.Adam(model.classifier.parameters(), lr=LR_HEAD)

        n_old = task_id * CLASSES_PER_TASK
        train_task(model, train_loader, optimizer, device,
                   teacher_model=teacher_model, ewc=ewc, n_old=n_old, scaler=scaler)

        # FIX: accumulate EWC across tasks
        ewc.update(model, train_loader, device)

        teacher_model = copy.deepcopy(model)
        teacher_model.eval()
        for p in teacher_model.parameters():
            p.requires_grad_(False)

        # FIX: evaluate on test data
        model.eval()
        print("  Accuracies →", end="")
        for prev in range(task_id + 1):
            acc = evaluate(model, [test_loaders[prev]], device)[0]
            acc_matrix[task_id, prev] = acc
            print(f"  T{prev+1}: {acc*100:.1f}%", end="")
        print(f"\n  Time: {time.time()-start:.1f}s")

    # ── metrics ──
    T   = NUM_TASKS
    aa  = float(np.mean(acc_matrix[T-1, :T]))
    bwt = float(np.mean([acc_matrix[T-1, j] - acc_matrix[j, j] for j in range(T-1)]))

    print(f"\n{'='*40}")
    print(f"  Average Accuracy (AA):   {aa*100:.2f}%")
    print(f"  Backward Transfer (BWT): {bwt*100:.2f}%")
    print(f"  Accuracy matrix:\n{np.round(acc_matrix*100, 1)}")

    with open("results_hybrid_gpu.json", "w") as f:
        json.dump({"method": "hybrid", "acc_matrix": acc_matrix.tolist(),
                   "aa": round(aa*100, 2), "bwt": round(bwt*100, 2)}, f, indent=2)
    print("  Saved → results_hybrid_gpu.json")
