"""
data.py — Split-CIFAR-100 data pipeline
Splits CIFAR-100 into 5 sequential tasks of 20 classes each.
"""

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np

# ── reproducibility ──────────────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ── constants ────────────────────────────────────────────────────────────────
NUM_TASKS   = 5
CLASSES_PER_TASK = 20          # 100 classes / 5 tasks
BATCH_SIZE  = 64
DATA_DIR    = "./data"

# Standard CIFAR-100 normalisation
CIFAR_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR_STD  = (0.2675, 0.2565, 0.2761)

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
])


def get_task_classes(task_id: int) -> list[int]:
    """Return the 20 class indices assigned to task_id (0-indexed)."""
    start = task_id * CLASSES_PER_TASK
    return list(range(start, start + CLASSES_PER_TASK))


def _filter_dataset(dataset, class_ids: list[int]):
    """Return a Subset containing only samples whose label is in class_ids."""
    targets = np.array(dataset.targets)
    mask = np.isin(targets, class_ids)
    indices = np.where(mask)[0]
    return Subset(dataset, indices)


def get_task_loaders(task_id: int):
    """
    Returns (train_loader, test_loader) for a single task.
    Labels are remapped to [0, CLASSES_PER_TASK) within the task.
    The head is expanded externally; raw CIFAR labels are preserved here
    so the expanding-head model can use global class indices directly.
    """
    full_train = datasets.CIFAR100(DATA_DIR, train=True,  download=True, transform=train_transform)
    full_test  = datasets.CIFAR100(DATA_DIR, train=False, download=True, transform=test_transform)

    class_ids = get_task_classes(task_id)
    train_sub = _filter_dataset(full_train, class_ids)
    test_sub  = _filter_dataset(full_test,  class_ids)

    train_loader = DataLoader(train_sub, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_sub,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, test_loader


def get_all_seen_test_loader(up_to_task: int):
    """
    Returns a test loader covering all classes seen so far (tasks 0..up_to_task).
    Used to compute Average Accuracy after each task.
    """
    full_test = datasets.CIFAR100(DATA_DIR, train=False, download=True, transform=test_transform)
    seen_classes = []
    for t in range(up_to_task + 1):
        seen_classes.extend(get_task_classes(t))
    sub = _filter_dataset(full_test, seen_classes)
    return DataLoader(sub, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)


if __name__ == "__main__":
    print("Downloading CIFAR-100 and verifying splits...")
    for t in range(NUM_TASKS):
        tr, te = get_task_loaders(t)
        classes = get_task_classes(t)
        print(f"  Task {t+1}: classes {classes[0]}–{classes[-1]} | "
              f"train batches={len(tr)} | test batches={len(te)}")
    print("Data pipeline OK.")
