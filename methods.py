"""
methods.py — Continual learning loss functions and per-task update logic.

Four methods implemented:
    1. Naive         — cross-entropy only (catastrophic forgetting baseline)
    2. EWC           — Elastic Weight Consolidation [Kirkpatrick et al. 2017]
    3. LwF           — Learning without Forgetting [Li & Hoiem 2016]
    4. Hybrid        — EWC + LwF with task-adaptive distillation weight
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ── hyper-parameters (can be overridden from train.py) ───────────────────────
EWC_LAMBDA   = 1000.0   # regularisation strength for EWC
LWF_LAMBDA0  = 1.0      # base distillation weight for LwF / Hybrid
KD_TEMP      = 2.0      # knowledge-distillation temperature


# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────

def _cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, targets)


def _kd_loss(logits_new: torch.Tensor,
             logits_old: torch.Tensor,
             T: float = KD_TEMP) -> torch.Tensor:
    """
    Soft-target knowledge distillation loss (KL divergence).
    logits_old: logits from the saved old model (no grad).
    """
    p_old = F.softmax(logits_old / T, dim=1)
    log_p_new = F.log_softmax(logits_new / T, dim=1)
    # Only distil over the classes the old model knew
    n_old = logits_old.size(1)
    log_p_new_old = log_p_new[:, :n_old]
    return F.kl_div(log_p_new_old, p_old, reduction="batchmean") * (T ** 2)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Naive fine-tuning
# ─────────────────────────────────────────────────────────────────────────────

class NaiveMethod:
    name = "Naive"

    def __init__(self):
        pass

    def before_task(self, model, task_id, train_loader, device):
        """Called once before training on a new task."""
        pass

    def loss(self, model, x, y, task_id) -> torch.Tensor:
        logits = model(x)
        return _cross_entropy(logits, y)

    def after_task(self, model, task_id, train_loader, device):
        """Called once after training on a task (for EWC fisher computation etc.)."""
        pass


# ─────────────────────────────────────────────────────────────────────────────
# 2. EWC
# ─────────────────────────────────────────────────────────────────────────────

class EWCMethod:
    name = "EWC"

    def __init__(self, ewc_lambda: float = EWC_LAMBDA):
        self.ewc_lambda = ewc_lambda
        # Lists accumulate across tasks
        self._means:   list[dict[str, torch.Tensor]] = []
        self._fishers: list[dict[str, torch.Tensor]] = []

    def before_task(self, model, task_id, train_loader, device):
        pass

    def loss(self, model, x, y, task_id) -> torch.Tensor:
        logits = model(x)
        ce = _cross_entropy(logits, y)
        ewc = self._ewc_penalty(model)
        return ce + self.ewc_lambda * ewc

    def after_task(self, model, task_id, train_loader, device):
        """Compute Fisher Information Matrix diagonal over the training set."""
        means   = {n: p.data.clone() for n, p in model.head.named_parameters()}
        fishers = {n: torch.zeros_like(p) for n, p in model.head.named_parameters()}

        model.eval()
        count = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            model.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            for n, p in model.head.named_parameters():
                if p.grad is not None:
                    fishers[n] += p.grad.data.clone().pow(2)
            count += 1

        for n in fishers:
            fishers[n] /= count          # average over batches

        self._means.append(means)
        self._fishers.append(fishers)

    def _ewc_penalty(self, model) -> torch.Tensor:
        penalty = torch.tensor(0.0, device=next(model.head.parameters()).device)
        for means, fishers in zip(self._means, self._fishers):
            for n, p in model.head.named_parameters():
                if n in means:
                    # Only penalise weights that existed at snapshot time
                    old_size = means[n].shape[0]
                    delta = p[:old_size] - means[n]
                    penalty = penalty + (fishers[n] * delta.pow(2)).sum()
        return penalty


# ─────────────────────────────────────────────────────────────────────────────
# 3. LwF
# ─────────────────────────────────────────────────────────────────────────────

class LwFMethod:
    name = "LwF"

    def __init__(self, lambda0: float = LWF_LAMBDA0, T: float = KD_TEMP):
        self.lambda0  = lambda0
        self.T        = T
        self._old_model = None   # snapshot from before current task

    def before_task(self, model, task_id, train_loader, device):
        if task_id == 0:
            self._old_model = None
            return
        self._old_model = copy.deepcopy(model)
        self._old_model.eval()
        for p in self._old_model.parameters():
            p.requires_grad_(False)

    def loss(self, model, x, y, task_id) -> torch.Tensor:
        logits = model(x)
        ce = _cross_entropy(logits, y)
        if self._old_model is None:
            return ce
        with torch.no_grad():
            old_logits = self._old_model(x)
        kd = _kd_loss(logits, old_logits, self.T)
        return ce + self.lambda0 * kd

    def after_task(self, model, task_id, train_loader, device):
        pass


# ─────────────────────────────────────────────────────────────────────────────
# 4. Hybrid (EWC + LwF with task-adaptive distillation weight)
# ─────────────────────────────────────────────────────────────────────────────

class HybridMethod:
    """
    L = CE + λ_ewc * L_EWC + (t / T) * λ0 * L_KD
    Distillation weight grows as more tasks are learned.
    """
    name = "Hybrid"

    def __init__(self,
                 num_tasks:  int   = 5,
                 ewc_lambda: float = EWC_LAMBDA,
                 lambda0:    float = LWF_LAMBDA0,
                 T:          float = KD_TEMP):
        self.num_tasks  = num_tasks
        self.ewc_lambda = ewc_lambda
        self.lambda0    = lambda0
        self.T          = T

        self._ewc  = EWCMethod(ewc_lambda)
        self._lwf  = LwFMethod(lambda0, T)

    def before_task(self, model, task_id, train_loader, device):
        self._current_task = task_id
        self._lwf.before_task(model, task_id, train_loader, device)
        # EWC needs no before_task action

    def loss(self, model, x, y, task_id) -> torch.Tensor:
        logits = model(x)
        ce = _cross_entropy(logits, y)

        ewc_pen = self._ewc._ewc_penalty(model)

        kd = torch.tensor(0.0, device=logits.device)
        if self._lwf._old_model is not None:
            with torch.no_grad():
                old_logits = self._lwf._old_model(x)
            kd = _kd_loss(logits, old_logits, self.T)

        dist_weight = (task_id / self.num_tasks) * self.lambda0

        return ce + self.ewc_lambda * ewc_pen + dist_weight * kd

    def after_task(self, model, task_id, train_loader, device):
        self._ewc.after_task(model, task_id, train_loader, device)
        # LwF doesn't need after_task


# ─────────────────────────────────────────────────────────────────────────────
# registry
# ─────────────────────────────────────────────────────────────────────────────

METHOD_REGISTRY = {
    "naive":  NaiveMethod,
    "ewc":    EWCMethod,
    "lwf":    LwFMethod,
    "hybrid": HybridMethod,
}
