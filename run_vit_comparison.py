"""
ViT-based Continual Learning Comparison on Split-CIFAR-100.

Compares 4 methods all using the same frozen ViT-B/16 backbone:
  1. Naive  — linear classifier, full CE, no forgetting prevention
  2. EWC    — elastic weight consolidation on classifier weights
  3. LwF    — knowledge distillation from old model
  4. ADAM   — per-task adapters + NCM classifier (replay-free, no forgetting)

python run_vit_comparison.py
"""

import copy, json, time, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.models import vit_b_16, ViT_B_16_Weights
from torch.utils.data import DataLoader, Subset

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
NUM_TASKS        = 5
CLASSES_PER_TASK = 20
BATCH_SIZE       = 64
NUM_WORKERS      = 4

# Shared classifier methods (Naive / EWC / LwF)
EPOCHS           = 5
LR               = 1e-3
EWC_LAMBDA       = 10000
LWF_ALPHA        = 2.0
KD_TEMP          = 2.0

# ADAM-specific
ADAPTER_EPOCHS   = 8
ADAPTER_LR       = 1e-3
ADAPTER_DIM      = 128
TUKEY_ALPHA      = 0.5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ──────────────────────────────────────────────
# DATA
# ──────────────────────────────────────────────
def get_tasks():
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    test_tf = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    train_data = datasets.CIFAR100(root='./data', train=True,  download=True, transform=train_tf)
    test_data  = datasets.CIFAR100(root='./data', train=False, download=True, transform=test_tf)
    tasks = []
    for t in range(NUM_TASKS):
        s, e = t * CLASSES_PER_TASK, (t + 1) * CLASSES_PER_TASK
        tr = [i for i, (_, y) in enumerate(train_data) if s <= y < e]
        te = [i for i, (_, y) in enumerate(test_data)  if s <= y < e]
        tasks.append((Subset(train_data, tr), Subset(test_data, te)))
    return tasks

# ──────────────────────────────────────────────
# SHARED: ViT-B/16 BACKBONE (frozen)
# ──────────────────────────────────────────────
def build_backbone():
    vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
    feat_dim = vit.heads.head.in_features  # 768
    vit.heads = nn.Identity()
    for p in vit.parameters():
        p.requires_grad = False
    vit = vit.to(device).eval()
    return vit, feat_dim

# ──────────────────────────────────────────────
# SHARED: EVALUATION (class-incremental full argmax)
# ──────────────────────────────────────────────
def evaluate_linear(model, loaders):
    model.eval()
    results = []
    with torch.no_grad():
        for loader in loaders:
            correct = total = 0
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                preds = model(x).argmax(dim=1)
                correct += (preds == y).sum().item()
                total   += y.size(0)
            results.append(correct / total)
    return results

# ──────────────────────────────────────────────
# SHARED: LINEAR MODEL (backbone + classifier head)
# ──────────────────────────────────────────────
class LinearModel(nn.Module):
    def __init__(self, backbone, feat_dim, num_classes=100):
        super().__init__()
        self.backbone   = backbone
        self.classifier = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        with torch.no_grad():
            feat = self.backbone(x)
        return self.classifier(feat)

# ──────────────────────────────────────────────
# METHOD 1: NAIVE
# ──────────────────────────────────────────────
def run_naive(tasks, backbone, feat_dim):
    print(f"\n{'='*50}\n  Method: NAIVE (ViT)\n{'='*50}")
    model = LinearModel(backbone, feat_dim).to(device)
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=LR)

    test_loaders = []
    acc_matrix   = np.zeros((NUM_TASKS, NUM_TASKS))
    task_times   = []

    for task_id, (train_ds, test_ds) in enumerate(tasks):
        print(f"\n── Task {task_id+1}/{NUM_TASKS}  "
              f"(classes {task_id*CLASSES_PER_TASK}–{(task_id+1)*CLASSES_PER_TASK-1}) ──")
        t0 = time.time()

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS, pin_memory=True)
        test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
        test_loaders.append(test_loader)

        model.train()
        for epoch in range(EPOCHS):
            total = 0.0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                loss = F.cross_entropy(model(x), y)
                loss.backward()
                optimizer.step()
                total += loss.item()
            print(f"  Epoch {epoch+1}/{EPOCHS}  loss={total/len(train_loader):.4f}")

        accs = evaluate_linear(model, test_loaders)
        for prev, acc in enumerate(accs):
            acc_matrix[task_id, prev] = acc
        print("  Accuracies →", "  ".join(f"T{i+1}: {a*100:.1f}%" for i, a in enumerate(accs)))
        elapsed = time.time() - t0
        task_times.append(elapsed)
        print(f"  Time: {elapsed:.1f}s")

    return _summarize("naive_vit", acc_matrix, task_times)

# ──────────────────────────────────────────────
# METHOD 2: EWC
# ──────────────────────────────────────────────
class EWC:
    def __init__(self):
        self._params  = []
        self._fishers = []

    def update(self, model, loader):
        params  = {n: p.clone().detach() for n, p in model.named_parameters() if p.requires_grad}
        fishers = {n: torch.zeros_like(p) for n, p in model.named_parameters() if p.requires_grad}
        model.eval()
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            model.zero_grad()
            F.cross_entropy(model(x), y).backward()
            for n, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fishers[n] += p.grad.data.pow(2)
        for n in fishers:
            fishers[n] /= len(loader)
        self._params.append(params)
        self._fishers.append(fishers)

    def penalty(self, model):
        loss = torch.tensor(0.0, device=device)
        for params, fishers in zip(self._params, self._fishers):
            for n, p in model.named_parameters():
                if n in params:
                    loss += (fishers[n] * (p - params[n]).pow(2)).sum()
        return loss

def run_ewc(tasks, backbone, feat_dim):
    print(f"\n{'='*50}\n  Method: EWC (ViT)\n{'='*50}")
    model     = LinearModel(backbone, feat_dim).to(device)
    ewc       = EWC()
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=LR)

    test_loaders = []
    acc_matrix   = np.zeros((NUM_TASKS, NUM_TASKS))
    task_times   = []

    for task_id, (train_ds, test_ds) in enumerate(tasks):
        print(f"\n── Task {task_id+1}/{NUM_TASKS}  "
              f"(classes {task_id*CLASSES_PER_TASK}–{(task_id+1)*CLASSES_PER_TASK-1}) ──")
        t0 = time.time()

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS, pin_memory=True)
        test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
        test_loaders.append(test_loader)

        model.train()
        for epoch in range(EPOCHS):
            total = 0.0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                loss = F.cross_entropy(model(x), y) + EWC_LAMBDA * ewc.penalty(model)
                loss.backward()
                optimizer.step()
                total += loss.item()
            print(f"  Epoch {epoch+1}/{EPOCHS}  loss={total/len(train_loader):.4f}")

        ewc.update(model, train_loader)

        accs = evaluate_linear(model, test_loaders)
        for prev, acc in enumerate(accs):
            acc_matrix[task_id, prev] = acc
        print("  Accuracies →", "  ".join(f"T{i+1}: {a*100:.1f}%" for i, a in enumerate(accs)))
        elapsed = time.time() - t0
        task_times.append(elapsed)
        print(f"  Time: {elapsed:.1f}s")

    return _summarize("ewc_vit", acc_matrix, task_times)

# ──────────────────────────────────────────────
# METHOD 3: LwF
# ──────────────────────────────────────────────
def run_lwf(tasks, backbone, feat_dim):
    print(f"\n{'='*50}\n  Method: LwF (ViT)\n{'='*50}")
    model     = LinearModel(backbone, feat_dim).to(device)
    teacher   = None
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=LR)

    test_loaders = []
    acc_matrix   = np.zeros((NUM_TASKS, NUM_TASKS))
    task_times   = []

    for task_id, (train_ds, test_ds) in enumerate(tasks):
        n_old = task_id * CLASSES_PER_TASK
        print(f"\n── Task {task_id+1}/{NUM_TASKS}  "
              f"(classes {n_old}–{n_old+CLASSES_PER_TASK-1}) ──")
        t0 = time.time()

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS, pin_memory=True)
        test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
        test_loaders.append(test_loader)

        model.train()
        for epoch in range(EPOCHS):
            total = 0.0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                outputs = model(x)
                loss    = F.cross_entropy(outputs, y)
                if teacher is not None and n_old > 0:
                    with torch.no_grad():
                        t_out = teacher(x)
                    kd = F.kl_div(
                        F.log_softmax(outputs[:, :n_old] / KD_TEMP, dim=1),
                        F.softmax(t_out[:, :n_old]       / KD_TEMP, dim=1),
                        reduction="batchmean",
                    ) * (KD_TEMP ** 2)
                    loss = loss + LWF_ALPHA * kd
                loss.backward()
                optimizer.step()
                total += loss.item()
            print(f"  Epoch {epoch+1}/{EPOCHS}  loss={total/len(train_loader):.4f}")

        teacher = copy.deepcopy(model).eval()
        for p in teacher.parameters():
            p.requires_grad_(False)

        accs = evaluate_linear(model, test_loaders)
        for prev, acc in enumerate(accs):
            acc_matrix[task_id, prev] = acc
        print("  Accuracies →", "  ".join(f"T{i+1}: {a*100:.1f}%" for i, a in enumerate(accs)))
        elapsed = time.time() - t0
        task_times.append(elapsed)
        print(f"  Time: {elapsed:.1f}s")

    return _summarize("lwf_vit", acc_matrix, task_times)

# ──────────────────────────────────────────────
# METHOD 4: ADAM (per-task adapters + NCM)
# ──────────────────────────────────────────────
class TaskAdapter(nn.Module):
    def __init__(self, in_dim, out_dim=ADAPTER_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim * 2),
            nn.BatchNorm1d(out_dim * 2),
            nn.GELU(),
            nn.Linear(out_dim * 2, out_dim),
        )
    def forward(self, x):
        return self.net(x)

def extract_features(backbone, adapters, x):
    with torch.no_grad():
        base  = backbone(x.to(device))
        parts = [base] + [a(base) for a in adapters]
        feat  = torch.cat(parts, dim=1)
        feat  = feat.relu().pow(TUKEY_ALPHA)
        return F.normalize(feat, dim=1)

def ncm_predict(feats, prototypes, proto_labels):
    dists = torch.cdist(feats, prototypes.to(device))
    return proto_labels.to(device)[dists.argmin(dim=1)]

def run_adam(tasks, backbone, feat_dim):
    print(f"\n{'='*50}\n  Method: ADAM (ViT + Adapters + NCM)\n{'='*50}")

    adapters     = []
    prototypes   = {}   # class → mean feature vector
    tasks_data   = []
    test_loaders = []
    acc_matrix   = np.zeros((NUM_TASKS, NUM_TASKS))
    task_times   = []

    for task_id, (train_ds, test_ds) in enumerate(tasks):
        n_old    = task_id * CLASSES_PER_TASK
        n_newend = n_old + CLASSES_PER_TASK
        print(f"\n── Task {task_id+1}/{NUM_TASKS}  (classes {n_old}–{n_newend-1}) ──")
        t0 = time.time()

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS, pin_memory=True)
        test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
        test_loaders.append(test_loader)
        tasks_data.append((train_ds, test_ds))

        # Step 1: Train adapter for this task
        print("  Training adapter...")
        adapter  = TaskAdapter(feat_dim, ADAPTER_DIM).to(device)
        temp_clf = nn.Linear(ADAPTER_DIM, CLASSES_PER_TASK).to(device)
        opt      = torch.optim.AdamW(
            list(adapter.parameters()) + list(temp_clf.parameters()),
            lr=ADAPTER_LR, weight_decay=1e-4,
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=ADAPTER_EPOCHS)

        adapter.train()
        for epoch in range(ADAPTER_EPOCHS):
            total = correct = seen = 0
            for x, y in train_loader:
                x, y   = x.to(device), y.to(device)
                y_local = y - n_old
                with torch.no_grad():
                    base = backbone(x)
                logits = temp_clf(adapter(base))
                loss   = F.cross_entropy(logits, y_local)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(adapter.parameters(), 5.0)
                opt.step()
                total   += loss.item()
                correct += (logits.argmax(1) == y_local).sum().item()
                seen    += y.size(0)
            sched.step()
            print(f"    Epoch {epoch+1}/{ADAPTER_EPOCHS}  loss={total/len(train_loader):.4f}  acc={correct/seen*100:.1f}%")

        adapter.eval()
        for p in adapter.parameters():
            p.requires_grad_(False)
        adapters.append(adapter)
        del temp_clf

        # Step 2: Recompute ALL prototypes with expanded feature space
        print("  Recomputing prototypes...")
        all_feats = {c: [] for c in range(n_newend)}
        for tr_ds, _ in tasks_data:
            loader = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=False,
                                num_workers=NUM_WORKERS, pin_memory=True)
            for x, y in loader:
                feats = extract_features(backbone, adapters, x)
                for i, label in enumerate(y):
                    c = label.item()
                    if c in all_feats:
                        all_feats[c].append(feats[i].cpu())
        for c, fl in all_feats.items():
            if fl:
                prototypes[c] = torch.stack(fl).mean(dim=0)

        # Step 3: NCM evaluation
        classes   = sorted(prototypes.keys())
        proto_mat = torch.stack([prototypes[c] for c in classes])
        proto_lbl = torch.tensor(classes, dtype=torch.long)

        print("  Accuracies →", end="")
        for prev in range(task_id + 1):
            correct = total = 0
            for x, y in test_loaders[prev]:
                feats = extract_features(backbone, adapters, x)
                preds = ncm_predict(feats, proto_mat, proto_lbl)
                correct += (preds == y.to(device)).sum().item()
                total   += y.size(0)
            acc = correct / total
            acc_matrix[task_id, prev] = acc
            print(f"  T{prev+1}: {acc*100:.1f}%", end="")
        print()
        elapsed = time.time() - t0
        task_times.append(elapsed)
        print(f"  Time: {elapsed:.1f}s")

    return _summarize("adam_vit", acc_matrix, task_times)

# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────
def _summarize(name, acc_matrix, task_times):
    T   = NUM_TASKS
    aa  = float(np.mean(acc_matrix[T-1, :T]))
    bwt = float(np.mean([acc_matrix[T-1, j] - acc_matrix[j, j] for j in range(T-1)]))
    print(f"\n  AA: {aa*100:.2f}%   BWT: {bwt*100:.2f}%")
    results = {
        "method": name,
        "acc_matrix": acc_matrix.tolist(),
        "aa":  round(aa*100, 2),
        "bwt": round(bwt*100, 2),
        "task_times": [round(t, 2) for t in task_times],
    }
    os.makedirs("results", exist_ok=True)
    with open(f"results/{name}.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved → results/{name}.json")
    return results

def print_comparison(all_results):
    print(f"\n{'='*62}")
    print("  ViT Comparison Table")
    print(f"{'='*62}")
    print(f"  {'Method':<18} {'AA (%)':>8} {'BWT (%)':>9} {'Avg time/task':>15}")
    print(f"  {'-'*56}")
    for name, r in all_results.items():
        avg_t = round(float(np.mean(r["task_times"])), 1)
        print(f"  {name:<18} {r['aa']:>8.2f} {r['bwt']:>9.2f} {avg_t:>13.1f}s")
    print()
    best_aa  = max(all_results, key=lambda m: all_results[m]["aa"])
    best_bwt = max(all_results, key=lambda m: all_results[m]["bwt"])
    print(f"  Best AA:  {best_aa}  ({all_results[best_aa]['aa']:.2f}%)")
    print(f"  Best BWT: {best_bwt}  ({all_results[best_bwt]['bwt']:.2f}%)")

# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
import os

if __name__ == "__main__":
    print("Loading data...")
    tasks = get_tasks()

    print("Loading ViT-B/16 backbone (shared across all methods)...")
    backbone, feat_dim = build_backbone()
    print(f"Feature dim: {feat_dim}")

    all_results = {}
    all_results["Naive"]  = run_naive(tasks, backbone, feat_dim)
    all_results["EWC"]    = run_ewc(tasks, backbone, feat_dim)
    all_results["LwF"]    = run_lwf(tasks, backbone, feat_dim)
    all_results["ADAM"]   = run_adam(tasks, backbone, feat_dim)

    print_comparison(all_results)
