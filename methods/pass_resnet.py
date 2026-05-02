"""
PASS — Prototype Augmentation with Self-Supervision (ResNet-18)
Best variant: frozen backbone + k-means prototypes (K=5) + Tukey transform

Key ideas:
  1. Frozen ResNet-18 + projection head extracts features
  2. K-means clusters each class into K prototypes
  3. Augment old classes by adding Gaussian noise to stored prototypes
  4. NCM (cosine similarity) for classification — no softmax head

Results from notebook:
  AA: 57.20%  BWT: -10.79%

Run standalone:
    python methods/pass_resnet.py
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
BATCH_SIZE       = 256
NUM_WORKERS      = 4
EPOCHS           = 5
LR               = 1e-3
PROTO_AUG_N      = 50        # augmented samples per class per batch
K_PROTOTYPES     = 5         # k-means clusters per class
TUKEY_ALPHA      = 0.5       # Tukey power transform exponent
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


# ── model ─────────────────────────────────────────────────────────────────────
class FeatureExtractor(nn.Module):
    """Frozen ResNet-18 + trainable projection head."""
    def __init__(self):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])   # drop FC
        self.feat_dim = backbone.fc.in_features                           # 512
        for p in self.backbone.parameters():
            p.requires_grad = False

        self.projector = nn.Sequential(
            nn.Linear(self.feat_dim, self.feat_dim * 2),
            nn.BatchNorm1d(self.feat_dim * 2),
            nn.ReLU(),
            nn.Linear(self.feat_dim * 2, self.feat_dim),
        )

    def extract_raw(self, x):
        with torch.no_grad():
            f = self.backbone(x).flatten(1)
        return self.projector(f)

    def forward(self, x):
        return self.extract_raw(x)


# ── prototype store ───────────────────────────────────────────────────────────
class PrototypeStore:
    """Stores K cluster centres + diagonal covariance per class."""
    def __init__(self):
        self.centers  = {}   # class → (K, D) tensor
        self.cov_diag = {}   # class → (D,) tensor  (diagonal covariance)

    def _kmeans(self, feats, K, n_iter=30):
        """Simple k-means on CPU."""
        feats = feats.cpu()
        idx   = torch.randperm(len(feats))[:K]
        centers = feats[idx].clone()
        for _ in range(n_iter):
            dists   = torch.cdist(feats, centers)
            assigns = dists.argmin(dim=1)
            new_c   = torch.stack([feats[assigns == k].mean(0)
                                   if (assigns == k).any()
                                   else centers[k]
                                   for k in range(K)])
            centers = new_c
        return centers

    def update(self, extractor, loader, task_id):
        """Extract features for all classes in the task and compute prototypes."""
        extractor.eval()
        all_feats = {}
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device)
                feats = extractor.extract_raw(x)
                feats = feats.relu().pow(TUKEY_ALPHA)        # Tukey transform
                feats = F.normalize(feats, dim=1)
                for i, label in enumerate(y):
                    c = label.item()
                    all_feats.setdefault(c, []).append(feats[i].cpu())

        for c, fl in all_feats.items():
            fm = torch.stack(fl)                             # (N, D)
            centers  = self._kmeans(fm, K=min(K_PROTOTYPES, len(fm)))
            cov_diag = fm.var(dim=0).clamp(min=1e-6)
            self.centers[c]  = centers
            self.cov_diag[c] = cov_diag

    def augment(self, n=PROTO_AUG_N):
        """
        Sample n augmented features by adding Gaussian noise to random prototypes.
        Returns (xs, ys) tensors.
        """
        if not self.centers:
            return None, None
        xs, ys = [], []
        classes = list(self.centers.keys())
        per_class = max(1, n // len(classes))
        for c in classes:
            centers  = self.centers[c]                       # (K, D)
            cov_diag = self.cov_diag[c]                      # (D,)
            for _ in range(per_class):
                k    = torch.randint(len(centers), (1,)).item()
                noise = torch.randn_like(centers[k]) * cov_diag.sqrt() * 0.1
                xs.append(centers[k] + noise)
                ys.append(c)
        return torch.stack(xs), torch.tensor(ys, dtype=torch.long)

    def get_all(self):
        """Returns stacked (proto_matrix, labels) for NCM."""
        classes = sorted(self.centers.keys())
        protos, labels = [], []
        for c in classes:
            for ctr in self.centers[c]:
                protos.append(ctr)
                labels.append(c)
        return torch.stack(protos), torch.tensor(labels, dtype=torch.long)


# ── NCM evaluation ────────────────────────────────────────────────────────────
def evaluate_ncm(extractor, proto_store, loaders):
    extractor.eval()
    proto_mat, proto_lbl = proto_store.get_all()
    proto_mat = F.normalize(proto_mat.to(device), dim=1)
    proto_lbl = proto_lbl.to(device)
    results = []
    with torch.no_grad():
        for loader in loaders:
            correct = total = 0
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                feats = extractor.extract_raw(x)
                feats = feats.relu().pow(TUKEY_ALPHA)
                feats = F.normalize(feats, dim=1)
                sims  = feats @ proto_mat.T
                preds = proto_lbl[sims.argmax(dim=1)]
                correct += (preds == y).sum().item()
                total   += y.size(0)
            results.append(correct / total)
    return results


# ── training ──────────────────────────────────────────────────────────────────
def train_task(extractor, loader, optimizer, proto_store, task_id):
    """Train projector with CE on current task + augmented old classes."""
    n_old    = task_id * CLASSES_PER_TASK
    n_newend = (task_id + 1) * CLASSES_PER_TASK

    # Temporary classifier for current training
    temp_clf = nn.Linear(extractor.feat_dim, n_newend).to(device)
    nn.init.xavier_uniform_(temp_clf.weight)
    opt_all  = torch.optim.Adam(
        list(extractor.projector.parameters()) + list(temp_clf.parameters()), lr=LR)

    extractor.train()
    for epoch in range(EPOCHS):
        total = correct = seen = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt_all.zero_grad()

            feats  = extractor.extract_raw(x)
            feats_t = feats.relu().pow(TUKEY_ALPHA)

            # Augment old classes
            ax, ay = proto_store.augment(PROTO_AUG_N)
            if ax is not None:
                ax, ay = ax.to(device), ay.to(device)
                feats_t = torch.cat([feats_t, ax])
                y_all   = torch.cat([y, ay])
            else:
                y_all = y

            loss = F.cross_entropy(temp_clf(feats_t), y_all)
            loss.backward()
            opt_all.step()
            total   += loss.item()
            correct += (temp_clf(feats_t).argmax(1) == y_all).sum().item()
            seen    += y_all.size(0)

        print(f"  Epoch {epoch+1}/{EPOCHS}  loss={total/len(loader):.4f}  "
              f"acc={correct/seen*100:.1f}%")
    del temp_clf


# ── main run ──────────────────────────────────────────────────────────────────
def run(tasks=None):
    if tasks is None:
        tasks = get_tasks()

    print(f"\n{'='*50}\n  Method: PASS (ResNet-18)\n{'='*50}")

    extractor   = FeatureExtractor().to(device)
    proto_store = PrototypeStore()
    optimizer   = torch.optim.Adam(extractor.projector.parameters(), lr=LR)
    test_loaders = []
    acc_matrix   = np.zeros((NUM_TASKS, NUM_TASKS))
    task_times   = []

    for task_id, (train_ds, test_ds) in enumerate(tasks):
        print(f"\n-- Task {task_id+1}/{NUM_TASKS} "
              f"(classes {task_id*CLASSES_PER_TASK}-{(task_id+1)*CLASSES_PER_TASK-1}) --")
        t0 = time.time()

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=NUM_WORKERS, pin_memory=True)
        test_loaders.append(DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                                       num_workers=NUM_WORKERS, pin_memory=True))

        train_task(extractor, train_loader, optimizer, proto_store, task_id)

        print("  Updating prototypes...")
        proto_store.update(extractor, train_loader, task_id)

        accs = evaluate_ncm(extractor, proto_store, test_loaders)
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
        "method": "pass_resnet",
        "acc_matrix": acc_matrix.tolist(),
        "aa":  round(aa*100, 2),
        "bwt": round(bwt*100, 2),
        "task_times": [round(t, 2) for t in task_times],
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "pass_resnet.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved to {RESULTS_DIR}/pass_resnet.json")
    return results


if __name__ == "__main__":
    run()
