"""
CORe50 Evaluation — ADAM vs Naive
NC (New Classes) scenario: 5 tasks x 10 classes
ViT-B/16 frozen backbone (same setup as run_vit_comparison.py)

CORe50 dataset:
  - 50 object categories, 11 sessions (different backgrounds/lighting)
  - Train sessions: s1, s2, s4, s5, s6, s8, s9, s11
  - Test  sessions: s3, s7, s10
  - NC split: Task 1 = classes 0-9, Task 2 = 10-19, ..., Task 5 = 40-49

Download (run once):
  python core50_eval/run_core50.py --download
"""

import os, sys, argparse, json, time, copy, zipfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from torchvision.models import vit_b_16, ViT_B_16_Weights
from PIL import Image

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
NUM_TASKS        = 5
CLASSES_PER_TASK = 10
TOTAL_CLASSES    = 50
BATCH_SIZE       = 64
NUM_WORKERS      = 4
EPOCHS           = 5
LR               = 1e-3
ADAPTER_DIM      = 128
ADAPTER_EPOCHS   = 8
ADAPTER_LR       = 1e-3
TUKEY_ALPHA      = 0.5

DATA_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
CORE50_URL  = "http://bias.csr.unibo.it/maltoni/download/core50/core50_128x128.zip"
CORE50_DIR  = os.path.join(DATA_DIR, "core50_128x128")

# Official train/test session split
TRAIN_SESSIONS = ["s1", "s2", "s4", "s5", "s6", "s8", "s9", "s11"]
TEST_SESSIONS  = ["s3", "s7", "s10"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ──────────────────────────────────────────────
# DOWNLOAD
# ──────────────────────────────────────────────
def download_core50():
    os.makedirs(DATA_DIR, exist_ok=True)
    zip_path = os.path.join(DATA_DIR, "core50_128x128.zip")

    if os.path.exists(CORE50_DIR):
        print("CORe50 already downloaded.")
        return

    if not HAS_REQUESTS:
        print("Install requests: pip install requests")
        print(f"Or manually download from:\n  {CORE50_URL}")
        print(f"And extract to: {DATA_DIR}")
        return

    print(f"Downloading CORe50 (~3.8 GB)...")
    r = requests.get(CORE50_URL, stream=True)
    total = int(r.headers.get("content-length", 0))
    downloaded = 0
    with open(zip_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            f.write(chunk)
            downloaded += len(chunk)
            pct = downloaded / total * 100 if total else 0
            print(f"\r  {pct:.1f}%", end="", flush=True)
    print("\nExtracting...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(DATA_DIR)
    os.remove(zip_path)
    print("Done.")

# ──────────────────────────────────────────────
# DATASET
# ──────────────────────────────────────────────
class CORe50Dataset(Dataset):
    """
    Loads CORe50 images from disk.
    Label = object index 0-49 (o1 -> 0, o2 -> 1, ..., o50 -> 49)
    """
    def __init__(self, sessions, transform=None):
        self.samples   = []   # list of (img_path, label)
        self.transform = transform

        for session in sessions:
            session_dir = os.path.join(CORE50_DIR, session)
            if not os.path.exists(session_dir):
                raise FileNotFoundError(f"Session dir not found: {session_dir}\n"
                                        f"Run with --download first.")
            for obj_name in sorted(os.listdir(session_dir)):
                if not obj_name.startswith("o"):
                    continue
                label     = int(obj_name[1:]) - 1   # o1 -> 0, o50 -> 49
                obj_dir   = os.path.join(session_dir, obj_name)
                for fname in os.listdir(obj_dir):
                    if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                        self.samples.append((os.path.join(obj_dir, fname), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def get_tasks():
    train_tf = transforms.Compose([
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

    train_full = CORe50Dataset(TRAIN_SESSIONS, transform=train_tf)
    test_full  = CORe50Dataset(TEST_SESSIONS,  transform=test_tf)

    tasks = []
    for t in range(NUM_TASKS):
        s, e = t * CLASSES_PER_TASK, (t + 1) * CLASSES_PER_TASK
        tr_idx = [i for i, (_, y) in enumerate(train_full.samples) if s <= y < e]
        te_idx = [i for i, (_, y) in enumerate(test_full.samples)  if s <= y < e]
        tasks.append((Subset(train_full, tr_idx), Subset(test_full, te_idx)))

    print(f"Tasks created: {NUM_TASKS} tasks x {CLASSES_PER_TASK} classes")
    for t, (tr, te) in enumerate(tasks):
        print(f"  Task {t+1}: {len(tr)} train, {len(te)} test samples")
    return tasks

# ──────────────────────────────────────────────
# BACKBONE
# ──────────────────────────────────────────────
def build_backbone():
    vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
    feat_dim = vit.heads.head.in_features
    vit.heads = nn.Identity()
    for p in vit.parameters():
        p.requires_grad = False
    return vit.to(device).eval(), feat_dim

# ──────────────────────────────────────────────
# LINEAR MODEL (for Naive)
# ──────────────────────────────────────────────
class LinearModel(nn.Module):
    def __init__(self, backbone, feat_dim):
        super().__init__()
        self.backbone   = backbone
        self.classifier = nn.Linear(feat_dim, TOTAL_CLASSES)

    def forward(self, x):
        with torch.no_grad():
            feat = self.backbone(x)
        return self.classifier(feat)

# ──────────────────────────────────────────────
# EVALUATION
# ──────────────────────────────────────────────
def evaluate_linear(model, loaders):
    model.eval()
    results = []
    with torch.no_grad():
        for loader in loaders:
            correct = total = 0
            for x, y in loader:
                x, y  = x.to(device), y.to(device)
                preds = model(x).argmax(dim=1)
                correct += (preds == y).sum().item()
                total   += y.size(0)
            results.append(correct / total)
    return results

def evaluate_ncm(adapters, backbone, prototypes, loaders):
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

# ──────────────────────────────────────────────
# ADAM HELPERS
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

# ──────────────────────────────────────────────
# SUMMARIZE
# ──────────────────────────────────────────────
def summarize(name, acc_matrix, task_times):
    T   = NUM_TASKS
    aa  = float(np.mean(acc_matrix[T-1, :T]))
    bwt = float(np.mean([acc_matrix[T-1, j] - acc_matrix[j, j] for j in range(T-1)]))
    print(f"\n  AA: {aa*100:.2f}%   BWT: {bwt*100:.2f}%")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, f"{name}.json"), "w") as f:
        json.dump({"method": name, "acc_matrix": acc_matrix.tolist(),
                   "aa": round(aa*100, 2), "bwt": round(bwt*100, 2),
                   "task_times": [round(t, 2) for t in task_times]}, f, indent=2)
    print(f"  Saved to {RESULTS_DIR}/{name}.json")
    return {"aa": round(aa*100, 2), "bwt": round(bwt*100, 2), "task_times": task_times}

# ──────────────────────────────────────────────
# METHOD 1: NAIVE
# ──────────────────────────────────────────────
def run_naive(tasks, backbone, feat_dim):
    print(f"\n{'='*50}\n  Method: NAIVE\n{'='*50}")
    model     = LinearModel(backbone, feat_dim).to(device)
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=LR)
    test_loaders = []
    acc_matrix   = np.zeros((NUM_TASKS, NUM_TASKS))
    task_times   = []

    for task_id, (train_ds, test_ds) in enumerate(tasks):
        print(f"\n-- Task {task_id+1}/{NUM_TASKS} (classes {task_id*CLASSES_PER_TASK}-{(task_id+1)*CLASSES_PER_TASK-1}) --")
        t0 = time.time()
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=NUM_WORKERS, pin_memory=True)
        test_loaders.append(DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                                       num_workers=NUM_WORKERS, pin_memory=True))
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
        print("  Accs:", "  ".join(f"T{i+1}:{a*100:.1f}%" for i, a in enumerate(accs)))
        elapsed = time.time() - t0
        task_times.append(elapsed)
        print(f"  Time: {elapsed:.1f}s")

    return summarize("naive_core50", acc_matrix, task_times)

# ──────────────────────────────────────────────
# METHOD 2: ADAM
# ──────────────────────────────────────────────
def run_adam(tasks, backbone, feat_dim):
    print(f"\n{'='*50}\n  Method: ADAM (Adapters + NCM)\n{'='*50}")
    adapters   = []
    prototypes = {}
    tasks_data = []
    test_loaders = []
    acc_matrix   = np.zeros((NUM_TASKS, NUM_TASKS))
    task_times   = []

    for task_id, (train_ds, test_ds) in enumerate(tasks):
        n_old    = task_id * CLASSES_PER_TASK
        n_newend = n_old + CLASSES_PER_TASK
        print(f"\n-- Task {task_id+1}/{NUM_TASKS} (classes {n_old}-{n_newend-1}) --")
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
            lr=ADAPTER_LR, weight_decay=1e-4,
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=ADAPTER_EPOCHS)

        adapter.train()
        for epoch in range(ADAPTER_EPOCHS):
            total = correct = seen = 0
            for x, y in train_loader:
                x, y    = x.to(device), y.to(device)
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
            print(f"    Epoch {epoch+1}/{ADAPTER_EPOCHS}  loss={total/len(train_loader):.4f}  "
                  f"acc={correct/seen*100:.1f}%")

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
        accs = evaluate_ncm(adapters, backbone, prototypes, test_loaders)
        for prev, acc in enumerate(accs):
            acc_matrix[task_id, prev] = acc
        print("  Accs:", "  ".join(f"T{i+1}:{a*100:.1f}%" for i, a in enumerate(accs)))
        elapsed = time.time() - t0
        task_times.append(elapsed)
        print(f"  Time: {elapsed:.1f}s")

    return summarize("adam_core50", acc_matrix, task_times)

# ──────────────────────────────────────────────
# COMPARISON TABLE
# ──────────────────────────────────────────────
def print_comparison(results):
    print(f"\n{'='*60}")
    print("  CORe50 NC Comparison (5 tasks x 10 classes)")
    print(f"{'='*60}")
    print(f"  {'Method':<20} {'AA (%)':>8} {'BWT (%)':>9} {'Avg time/task':>14}")
    print(f"  {'-'*55}")
    for name, r in results.items():
        avg_t = round(float(np.mean(r["task_times"])), 1)
        print(f"  {name:<20} {r['aa']:>8.2f} {r['bwt']:>9.2f} {avg_t:>12.1f}s")

# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true", help="Download CORe50 dataset")
    args = parser.parse_args()

    if args.download:
        download_core50()
        sys.exit(0)

    print("Loading CORe50 tasks...")
    tasks = get_tasks()

    print("\nLoading ViT-B/16 backbone...")
    backbone, feat_dim = build_backbone()
    print(f"Feature dim: {feat_dim}")

    results = {}
    results["Naive"] = run_naive(tasks, backbone, feat_dim)
    results["ADAM"]  = run_adam(tasks, backbone, feat_dim)

    print_comparison(results)
