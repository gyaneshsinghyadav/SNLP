"""Matched, training-era immediate action probes for E5.

Each candidate branch starts from the same learner parameters, optimizer state,
EWC state, replay examples and random draws. Probes only use CURRENT revealed
training labels and CURRENT disjoint controller validation labels, never
sealed task.eval_idx or labels from later years. This is a one-step
counterfactual *surrogate*, NOT a next-year causal reward estimator.
"""
from __future__ import annotations
from copy import deepcopy
import time
import numpy as np
import torch

from .continual import train_gcn
from .temporal_control import model_ap


def choose_probe_actions(allowed_mask, chosen_index, maximum, seed):
    """Deterministic candidate list including the actual chosen action."""
    candidates = np.flatnonzero(allowed_mask).tolist()
    if chosen_index not in candidates:
        raise ValueError('Chosen action is not allowed')
    if maximum < 2:
        raise ValueError('Need at least two counterfactual candidates')
    if len(candidates) <= maximum:
        return candidates
    rng = np.random.default_rng(seed)
    others = [i for i in candidates if i != chosen_index]
    selected = rng.choice(others, size=maximum - 1, replace=False).tolist()
    return sorted([chosen_index, *map(int, selected)])


def matched_action_probes(model, optimizer, memory, ewc, pt, val_idx,
                          history_probes, actions, action_ids, cfg, options,
                          seed):
    """Compute comparable immediate validation outcomes on cloned branches.

    No optimizer/model/memory RNG state changes persist in the real learner.
    Each branch uses identical NumPy & PyTorch RNG streams. Cost is reported
    descriptively but is NOT used as a reward since runtime is nondeterministic.
    """
    if len(val_idx) == 0:
        raise ValueError('Counterfactual comparison needs revealed controller validation labels')
    if set(map(int, val_idx)) & set(map(int, pt.task.train_idx)):
        raise ValueError('Controller validation must be disjoint from learner training')
    if set(map(int, val_idx)) & set(map(int, pt.task.eval_idx)):
        raise ValueError('Sealed evaluations cannot be used for counterfactual training')
    initial_ap = model_ap(model, pt, val_idx)
    old_scores = [model_ap(model, old_pt, old_ids) for old_pt, old_ids in history_probes]
    results = []
    for idx in action_ids:
        action = actions[idx]
        # deepcopy the actual model and initialize a new optimizer bound to it.
        clone = deepcopy(model)
        opt = torch.optim.Adam(clone.parameters(), lr=cfg.lr, weight_decay=1e-5)
        opt.load_state_dict(deepcopy(optimizer.state_dict()))
        clone_memory = deepcopy(memory)
        clone_ewc = deepcopy(ewc)
        # IMPORTANT: use configured base maximum replay budget; memory.batch_size
        # may have been set by previous year's action.
        base_budget = options.e5_full_replay_batch
        budget = (min(clone_memory.capacity, max(2, int(round(
                    base_budget * action.replay_fraction))))
                  if action.replay_scale > 0 else 0)
        clone_memory.configure(action.replay_mode, budget or base_budget)
        local_rng = np.random.default_rng(seed)
        started = time.perf_counter()
        # GPU is not part of E5 comparison protocol; fork RNG for CPU and CUDA
        # so probe dropout never advances the main learner's random stream.
        cuda_devices = list(range(torch.cuda.device_count())) if str(cfg.device).startswith('cuda') else []
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(seed)
            loss = train_gcn(
                clone, pt, opt, epochs=cfg.epochs,
                replay=clone_memory if action.replay_scale and len(clone_memory.items) else None,
                ewc=clone_ewc if action.ewc_scale and clone_ewc.fisher else None,
                ewc_lambda=cfg.ewc_lambda * action.ewc_scale,
                replay_weight=cfg.replay_weight * action.replay_scale,
                max_train_pairs=cfg.max_train_pairs,
                rng=local_rng,
                project_current_gradient=bool(action.projected),
            )
        val_after = model_ap(clone, pt, val_idx)
        new_scores = [model_ap(clone, old_pt, old_ids) for old_pt, old_ids in history_probes]
        drops = [max(0., prev - after) for prev, after in zip(old_scores, new_scores)
                 if np.isfinite(prev) and np.isfinite(after)]
        mean_drop = float(np.mean(drops)) if drops else 0.
        peak_drop = max(drops, default=0.)
        # Deliberately no wall-clock cost in supervised action target.
        reward = (float(val_after - initial_ap)
                  - options.forgetting_lambda * mean_drop
                  - options.peak_forgetting_lambda * peak_drop)
        results.append({
            'action_index': int(idx), 'action': action.name,
            'immediate_val_ap_before': initial_ap, 'immediate_val_ap_after': float(val_after),
            'immediate_val_ap_gain': float(val_after - initial_ap),
            'mean_historical_ap_drop': mean_drop, 'peak_historical_ap_drop': peak_drop,
            'surrogate_reward': reward, 'probe_loss': loss,
            'probe_seconds': time.perf_counter() - started,
            'probe_budget': budget,
            'probe_train_count': len(pt.task.train_idx),
            'probe_val_count': len(val_idx),
        })
    return results
