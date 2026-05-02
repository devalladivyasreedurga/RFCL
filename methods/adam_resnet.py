"""
ADAM — Adapter-based Continual Learning (ResNet-18)
Per-task MLP adapters + Tukey transform + NCM classifier

Same approach as ADAM ViT but with ResNet-18 backbone (512-dim features).
Demonstrates that the adapter approach works on weaker backbones too,
though ViT features give significantly better performance.

Results from notebook:
  AA: 47.63%  BWT: -15.54%

Run standalone:
    python methods/adam_resnet.py
"""

import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset

# ── config ────────────────────────────────────────────────────────────────────
NUM_TASKS        = 5
CLASSES_PER_TASK = 20
BATCH_SIZE       = 128
NUM_WORKERS      = 4
ADAPTER_EPOCHS   = 8
ADAPTER_LR       = 1e-3
ADAPTER_DIM      = 128
TUKEY_ALPHA      = 0.5
RESULTS_DIR      = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


# ── data ──────────────────────────────────────────────────────────────────────
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


# ── backbone ──────────────────────────────────────────────────────────────────
def build_backbone():
    resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    feat_dim = resnet.fc.in_features   # 512
    backbone = nn.Sequential(*list(resnet.children())[:-1])
    for p in backbone.parameters():
        p.requires_grad = False
    return backbone.to(device).eval(), feat_dim


# ── adapter ───────────────────────────────────────────────────────────────────
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


# ── feature extraction ────────────────────────────────────────────────────────
def extract_features(backbone, adapters, x):
    with torch.no_grad():
        base  = backbone(x.to(device)).flatten(1)         # (B, 512)
        parts = [base] + [a(base) for a in adapters]
        feat  = torch.cat(parts, dim=1)                   # (B, 512 + N*128)
        feat  = feat.relu().pow(TUKEY_ALPHA)
        return F.normalize(feat, dim=1)


# ── NCM evaluation ────────────────────────────────────────────────────────────
def evaluate_ncm(backbone, adapters, prototypes, loaders):
    classes   = sorted(prototypes.keys())
    proto_mat = torch.stack([prototypes[c] for c in classes]).to(device)
    proto_lbl = torch.tensor(classes, dtype=torch.long).to(device)
    results   = []
    for loader in loaders:
        correct = total = 0
        for x, y in loader:
            feats = extract_features(backbone, adapters, x)
            dists = torch.cdist(feats, proto_mat)
            preds = proto_lbl[dists.argmin(dim=1)]
            correct += (preds == y.to(device)).sum().item()
            total   += y.size(0)
        results.append(correct / total)
    return results


# ── main run ──────────────────────────────────────────────────────────────────
def run(tasks=None):
    if tasks is None:
        tasks = get_tasks()

    print(f"\n{'='*50}\n  Method: ADAM (ResNet-18)\n{'='*50}")

    backbone, feat_dim = build_backbone()
    adapters     = []
    prototypes   = {}
    tasks_data   = []
    test_loaders = []
    acc_matrix   = np.zeros((NUM_TASKS, NUM_TASKS))
    task_times   = []

    for task_id, (train_ds, test_ds) in enumerate(tasks):
        n_old    = task_id * CLASSES_PER_TASK
        n_newend = n_old + CLASSES_PER_TASK
        print(f"\n-- Task {task_id+1}/{NUM_TASKS} "
              f"(classes {n_old}-{n_newend-1}) --")
        t0 = time.time()

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=NUM_WORKERS, pin_memory=True)
        test_loaders.append(DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                                       num_workers=NUM_WORKERS, pin_memory=True))
        tasks_data.append((train_ds, test_ds))

        # Step 1: Train adapter for this task
        print("  Training adapter...")
        adapter  = TaskAdapter(feat_dim, ADAPTER_DIM).to(device)
        temp_clf = nn.Linear(ADAPTER_DIM, CLASSES_PER_TASK).to(device)
        opt      = torch.optim.AdamW(
            list(adapter.parameters()) + list(temp_clf.parameters()),
            lr=ADAPTER_LR, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=ADAPTER_EPOCHS)

        adapter.train()
        for epoch in range(ADAPTER_EPOCHS):
            total = correct = seen = 0
            for x, y in train_loader:
                x, y    = x.to(device), y.to(device)
                y_local = y - n_old
                with torch.no_grad():
                    base = backbone(x).flatten(1)
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
            print(f"    Epoch {epoch+1}/{ADAPTER_EPOCHS}  "
                  f"loss={total/len(train_loader):.4f}  acc={correct/seen*100:.1f}%")

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

        # Step 3: Evaluate with NCM
        accs = evaluate_ncm(backbone, adapters, prototypes, test_loaders)
        for prev, acc in enumerate(accs):
            acc_matrix[task_id, prev] = acc
        print("  Accs:", "  ".join(f"T{i+1}: {a*100:.1f}%" for i, a in enumerate(accs)))

        elapsed = time.time() - t0
        task_times.append(elapsed)
        print(f"  Time: {elapsed:.1f}s")

    T   = NUM_TASKS
    aa  = float(np.mean(acc_matrix[T-1, :T]))
    bwt = float(np.mean([acc_matrix[T-1, j] - acc_matrix[j, j] for j in range(T-1)]))
    print(f"\n  AA: {aa*100:.2f}%   BWT: {bwt*100:.2f}%")

    results = {
        "method": "adam_resnet",
        "acc_matrix": acc_matrix.tolist(),
        "aa":  round(aa*100, 2),
        "bwt": round(bwt*100, 2),
        "task_times": [round(t, 2) for t in task_times],
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "adam_resnet.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved to {RESULTS_DIR}/adam_resnet.json")
    return results


if __name__ == "__main__":
    run()
