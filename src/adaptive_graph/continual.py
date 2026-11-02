"""Online EWC and bounded historical graph-task rehearsal."""
from __future__ import annotations
from collections import deque
import torch
from torch.nn import functional as F


class OnlineEWC:
    def __init__(self, decay=0.9):
        self.decay = decay
        self.fisher = {}
        self.theta = {}

    def penalty(self, model):
        first = next(model.parameters())
        loss = torch.zeros((), dtype=first.dtype, device=first.device)
        for name, p in model.named_parameters():
            if name in self.fisher:
                loss = loss + (self.fisher[name] * (p - self.theta[name]).square()).sum()
        return loss / 2

    def consolidate(self, model, loss_fn):
        """Average squared minibatch gradients as a diagonal Fisher proxy.

        `loss_fn` may be one callable or a sequence of small-batch callbacks.
        This is empirical and approximate; full per-example Fisher is costlier.
        """
        functions = [loss_fn] if callable(loss_fn) else list(loss_fn)
        if not functions:
            return
        accumulated = {}
        for fn in functions:
            model.zero_grad(set_to_none=True)
            fn().backward()
            for name, p in model.named_parameters():
                if p.grad is not None:
                    grad_sq = p.grad.detach().square().clone()
                    accumulated[name] = accumulated.get(name, torch.zeros_like(grad_sq)) + grad_sq
        for name, p in model.named_parameters():
            if name not in accumulated:
                continue
            contribution = accumulated[name] / len(functions)
            self.fisher[name] = (self.decay * self.fisher[name] + contribution
                                 if name in self.fisher else contribution)
            self.theta[name] = p.detach().clone()
        model.zero_grad(set_to_none=True)


class TaskReplay:
    def __init__(self, capacity=3):
        self.items = deque(maxlen=capacity)

    def add(self, task):
        self.items.append(task)

    def sample(self, rng, how_many=2):
        if not self.items:
            return []
        indices = rng.choice(len(self.items), size=min(how_many, len(self.items)), replace=False)
        return [self.items[int(i)] for i in np_atleast(indices)]


def np_atleast(x):
    import numpy as np
    return np.atleast_1d(x)


def train_gcn(model, task, optimizer, epochs=8, replay=None, ewc=None, ewc_lambda=1.0,
              rng=None, replay_weight=0.5, max_train_pairs=2048,
              project_current_gradient=False, gradient_stats=None):
    import numpy as np
    rng = rng if rng is not None else np.random.default_rng(7)
    if not len(task.task.train_idx):
        return float("nan")
    last = float("nan")
    for _ in range(epochs):
        model.train()
        ids = task.subset("train")
        if len(ids) > max_train_pairs:
            ids = ids[torch.as_tensor(rng.choice(len(ids), max_train_pairs, replace=False),
                                      dtype=torch.long, device=ids.device)]
        optimizer.zero_grad(set_to_none=True)
        logits = model(task, ids)
        current_loss = F.binary_cross_entropy_with_logits(logits, task.labels[ids])
        loss = current_loss
        reference_loss = None
        if replay is not None:
            if hasattr(replay, "sample_pairs"):
                # E2: fixed number of past training examples across snapshots.
                # A per-example mean makes replay weight independent of how many
                # historical periods happened to contribute examples.
                groups = replay.sample_pairs(rng, current_time=task.task.time_index)
                old_sum = torch.zeros((), device=logits.device)
                old_count = 0
                for old, old_ids in groups:
                    old_sum = old_sum + F.binary_cross_entropy_with_logits(
                        model(old, old_ids), old.labels[old_ids], reduction="sum")
                    old_count += len(old_ids)
                if old_count:
                    reference_loss = old_sum / old_count
                    loss = loss + replay_weight * reference_loss
            else:
                for old in replay.sample(rng, 1):
                    old_ids = old.subset("train")
                    if len(old_ids) > max_train_pairs:
                        old_ids = old_ids[:max_train_pairs]
                    if len(old_ids):
                        old_loss = F.binary_cross_entropy_with_logits(model(old, old_ids),
                                                                       old.labels[old_ids])
                        loss = loss + replay_weight * old_loss
        if ewc is not None:
            loss = loss + ewc_lambda * ewc.penalty(model)
        # Optional A-GEM-style projection of only the current-year gradient.
        # It is an extra action in E3.2; all prior action paths are unchanged.
        correction = None
        params = [p for p in model.parameters() if p.requires_grad]
        if project_current_gradient and reference_loss is not None:
            from .gradient_conflict import projection_coefficient
            g_new = torch.autograd.grad(current_loss, params, retain_graph=True,
                                        allow_unused=True)
            g_old = torch.autograd.grad(reference_loss, params, retain_graph=True,
                                        allow_unused=True)
            coefficient = projection_coefficient(g_new, g_old)
            correction = (coefficient, g_old)
            if gradient_stats is not None:
                gradient_stats['probe_steps'] = gradient_stats.get('probe_steps', 0) + 1
                gradient_stats['projected_steps'] = gradient_stats.get('projected_steps', 0) + int(coefficient > 0)
        loss.backward()
        if correction is not None:
            coefficient, old_grads = correction
            if coefficient:
                for p, old_g in zip(params, old_grads):
                    if old_g is not None and p.grad is not None:
                        p.grad.add_(old_g.detach(), alpha=coefficient)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        last = float(loss.detach())
    return last


def consolidate_gcn(model, task, ewc, max_examples=128, batch_size=16):
    ids = task.subset("train")[:max_examples]
    if not len(ids):
        return
    model.eval()
    functions = []
    for begin in range(0, len(ids), batch_size):
        batch = ids[begin:begin+batch_size]
        functions.append(lambda batch=batch: F.binary_cross_entropy_with_logits(
            model(task, batch), task.labels[batch]))
    ewc.consolidate(model, functions)
