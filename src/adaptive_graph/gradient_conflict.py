"""E3.2: history-only gradient-interference measurement and A-GEM-style projection.

The reference gradient uses *stored learner-training pairs*, never controller
validation, future-year, or frozen test labels. Probe evaluation does not touch
optimizer, RNG, or replay sampler/counters. The projection is a first-order
heuristic, not a guarantee of lower historical task loss after an optimizer step.
"""
from collections import defaultdict
import numpy as np
import torch
from torch.nn import functional as F


def reference_groups(memory, max_pairs=16):
    if not memory or not memory.items:
        return []
    # Deterministically choose across the existing memory inventory, including
    # older periods. This does not mutate memory or use the replay RNG.
    ordered = sorted(memory.items, key=lambda e: (e.time_index, e.index, e.label))
    count = min(int(max_pairs), len(ordered))
    chosen = [ordered[int(i)] for i in np.linspace(0, len(ordered) - 1, count, dtype=int)]
    groups = defaultdict(list)
    for e in chosen:
        groups[id(e.prepared)].append(e)
    return [(g[0].prepared, [x.index for x in g]) for g in groups.values()]


def flat_dot(first, second):
    return sum((a.detach() * b.detach()).sum() for a,b in zip(first, second)
               if a is not None and b is not None)


def probe_gradient_conflict(model, current_pt, memory, max_pairs=16):
    """Return pre-action cosine and reference/current gradient norm ratio.

    Uses only current revealed `train_idx` and previously stored *masked* train
    examples. Does not modify model parameters or .grad buffers.
    """
    groups = reference_groups(memory, max_pairs)
    if not groups:
        return {'available': 0, 'cosine': 0., 'norm_ratio': 0.,
                'reference_pairs': 0, 'conflict': 0}
    params = [p for p in model.parameters() if p.requires_grad]
    was_training = model.training
    model.eval()
    try:
        ids = current_pt.subset('train')[:max_pairs]
        if not len(ids):
            return {'available': 0, 'cosine': 0., 'norm_ratio': 0.,
                    'reference_pairs': 0, 'conflict': 0}
        current_loss = F.binary_cross_entropy_with_logits(
            model(current_pt, ids), current_pt.labels[ids])
        new_g = torch.autograd.grad(current_loss, params, allow_unused=True)
        old_sum = None
        n = 0
        for pt, indices in groups:
            ix = torch.as_tensor(indices, dtype=torch.long, device=pt.labels.device)
            cur = F.binary_cross_entropy_with_logits(
                model(pt, ix), pt.labels[ix], reduction='sum')
            old_sum = cur if old_sum is None else old_sum + cur
            n += len(ix)
        ref_g = torch.autograd.grad(old_sum / n, params, allow_unused=True)
        dot = sum(float((a.detach() * b.detach()).sum()) for a,b in zip(new_g, ref_g)
                  if a is not None and b is not None)
        new_sq = sum(float(a.detach().square().sum()) for a in new_g if a is not None)
        old_sq = sum(float(b.detach().square().sum()) for b in ref_g if b is not None)
        cosine = dot / (np.sqrt(new_sq * old_sq) + 1e-12)
        return {'available': int(new_sq > 0 and old_sq > 0),
                'cosine': float(np.clip(cosine, -1., 1.)),
                'norm_ratio': float(np.sqrt(old_sq) / (np.sqrt(new_sq) + 1e-12)),
                'reference_pairs': n, 'conflict': int(dot < 0 and old_sq > 0)}
    finally:
        model.train(was_training)


def projection_coefficient(current_grads, reference_grads, epsilon=1e-12):
    """Return >=0 factor c so g_new + c*g_reference is nonconflicting."""
    dot = sum(float((a.detach() * b.detach()).sum())
              for a,b in zip(current_grads, reference_grads)
              if a is not None and b is not None)
    norm_sq = sum(float(b.detach().square().sum()) for b in reference_grads
                  if b is not None)
    return max(0., -dot / (norm_sq + epsilon))
