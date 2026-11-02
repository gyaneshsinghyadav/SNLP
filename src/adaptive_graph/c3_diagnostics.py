"""C3: matched-state adaptation action study, restricted to TRAINING-era rewards.

This is an *offline diagnostic*, not a deployable oracle policy or DQN training.
Every action at year t branches from identical pre-update GCN+optimizer, EWC,
replay and observed task. Rewards use the SAME t+1 controller validation pairs.
A fixed balanced reference trajectory determines each branching origin. Holding
that reference fixed isolates the immediate adaptation effect but does NOT
measure a learned controller's long-horizon return.
"""
from __future__ import annotations
from collections import deque
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
import json
import time
import hashlib

import numpy as np
import pandas as pd
import torch

from .continual import OnlineEWC, TaskReplay, consolidate_gcn, train_gcn
from .data import create_task
from .experiment import Config
from .meta_environment import historical_partition, make_group_network
from .models import HybridGCN, prepare
from .temporal_control import (
    ACTIONS, ControllerOptions, model_ap, split_controller_validation,
    reward_components, c2_observed_state,
)


def _prepare_year(net, cfg, options, ti):
    task = create_task(net, ti, seed=cfg.seed, max_positives=cfg.max_positives,
                       negative_ratio=cfg.negative_ratio, eval_fraction=cfg.eval_fraction)
    partition = split_controller_validation(task, options.controller_val_fraction, cfg.seed)
    if partition is None or len(task.eval_idx) == 0:
        raise ValueError(f"Year {task.time_value} has too few labeled candidates")
    train_idx, val_idx = partition
    if set(train_idx) & set(val_idx) or set(val_idx) & set(task.eval_idx):
        raise RuntimeError("Controller-validation leakage")
    pt = prepare(replace(task, train_idx=train_idx), net.n, net.node_features,
                 cfg.feature_track, cfg.device)
    return task, pt, val_idx


def _matched_group(net, cfg, options, group, train_end):
    torch.manual_seed(cfg.seed + 202700)
    rng_seed = cfg.seed + 31009
    baseline_rng = np.random.default_rng(rng_seed)
    base_model = None
    optimizer = None
    ewc = OnlineEWC()
    replay = TaskReplay(cfg.memory_tasks)
    probes = deque(maxlen=cfg.max_retention_tasks)
    rows = []
    n_trained = 0
    previous_val_ap = float('nan')
    previous_forgetting = 0.0
    previous_loss = 0.0
    previous_history_edges = 0
    for ti in range(cfg.start_time_index, train_end):
        task, pt, val_idx = _prepare_year(net, cfg, options, ti)
        if base_model is None:
            base_model = HybridGCN(pt.x.shape[1], cfg.hidden, use_gcn=True,
                                   use_heuristics=cfg.use_heuristics).to(cfg.device)
            optimizer = torch.optim.Adam(base_model.parameters(), lr=cfg.lr,
                                         weight_decay=1e-5)
        # This representation is computed BEFORE the action using historical
        # graph information and current TRAINING-era revealed labels only.
        observed_ap = model_ap(base_model, pt, val_idx) if n_trained else float('nan')
        state = c2_observed_state(task, pt.task.train_idx, observed_ap,
                                  previous_val_ap, previous_forgetting, previous_loss,
                                  len(replay.items), bool(ewc.fisher), net.n,
                                  len(net.times), previous_history_edges)
        previous_val_ap = observed_ap
        previous_history_edges = len(task.historical_edges)
        # Only fork once the GCN has completed warm-up and a next TRAINING year exists.
        fork = n_trained >= cfg.warmup and ti + 1 < train_end
        if fork:
            next_task, next_pt, next_val = _prepare_year(net, cfg, options, ti + 1)
            frozen_ap = model_ap(base_model, next_pt, next_val)
            pre_scores = [model_ap(base_model, old_pt, old_val) for old_pt, old_val in probes]
            start_digest = hashlib.sha256(b''.join(
                p.detach().cpu().numpy().tobytes() for p in base_model.parameters()
            )).hexdigest()[:16]
            next_val_size = len(next_val)
        else:
            frozen_ap = float('nan')
        balanced_model = None
        balanced_opt = None
        balanced_loss, balanced_forgetting = 0., 0.
        if fork:
            for action in ACTIONS:
                # Fork model AND Adam state before every action. No action is allowed
                # to inherit the weight updates of any preceding branch.
                branch = deepcopy(base_model)
                branch_opt = torch.optim.Adam(branch.parameters(), lr=cfg.lr,
                                              weight_decay=1e-5)
                branch_opt.load_state_dict(deepcopy(optimizer.state_dict()))
                branch_rng = np.random.default_rng(rng_seed + ti * 173)
                started = time.perf_counter()
                loss = train_gcn(
                    branch, pt, branch_opt, epochs=cfg.epochs,
                    replay=replay if action.replay_scale > 0 and replay.items else None,
                    ewc=ewc if action.ewc_scale > 0 and ewc.fisher else None,
                    ewc_lambda=cfg.ewc_lambda * action.ewc_scale,
                    replay_weight=cfg.replay_weight * action.replay_scale,
                    max_train_pairs=cfg.max_train_pairs, rng=branch_rng)
                seconds = time.perf_counter() - started
                after_scores = [model_ap(branch, old_pt, old_val) for old_pt, old_val in probes]
                drops = [max(0., pre - post) for pre, post in zip(pre_scores, after_scores)
                         if np.isfinite(pre) and np.isfinite(post)]
                forgetting = float(np.mean(drops)) if drops else 0.
                updated_ap = model_ap(branch, next_pt, next_val)
                parts = reward_components(updated_ap, frozen_ap, forgetting, seconds, options)
                rows.append({
                    'group': group, 'action_year': str(task.time_value),
                    'reward_year': str(next_task.time_value),
                    'action_time_index': ti, 'reward_time_index': ti + 1,
                    'action': action.name, 'reference_policy': 'fixed_balanced',
                    'same_starting_model': True, 'starting_model_sha256_16': start_digest,
                    'validation_candidates': next_val_size,
                    'prior_probe_count': len(pre_scores), 'train_candidates': len(task.train_idx),
                    **{f'state_{j}': float(v) for j, v in enumerate(state)},
                    'gcn_train_loss': loss, 'updated_ap': updated_ap,
                    'frozen_ap': frozen_ap, **parts,
                })
                if action.name == 'balanced':
                    balanced_model, balanced_opt = branch, branch_opt
                    balanced_loss, balanced_forgetting = loss, forgetting
            if balanced_model is None:
                raise RuntimeError('Missing reference-policy branch')
            base_model, optimizer = balanced_model, balanced_opt
        else:
            # Warm-up and final train year: balanced reference update but no
            # rewarded action (would require revealing sealed test-year labels).
            balanced_loss = train_gcn(base_model, pt, optimizer, epochs=cfg.epochs,
                      replay=replay if replay.items else None,
                      ewc=ewc if ewc.fisher else None,
                      ewc_lambda=cfg.ewc_lambda,
                      replay_weight=cfg.replay_weight,
                      max_train_pairs=cfg.max_train_pairs, rng=baseline_rng)
        previous_loss = balanced_loss
        previous_forgetting = balanced_forgetting
        if cfg.use_ewc:
            consolidate_gcn(base_model, pt, ewc)
        if cfg.use_replay:
            replay.add(pt)
        probes.append((pt, val_idx))
        n_trained += 1
    return rows


def summarize_action_study(df, forgetting_weights, cost_weights):
    """Evaluate reward grids only on TRAINING-era, matched-action observations."""
    if df.empty:
        raise ValueError('No matched action comparisons were created')
    group_keys = ['group', 'action_time_index']
    counts = df.groupby(group_keys).action.nunique()
    if (counts != len(ACTIONS)).any():
        raise RuntimeError('Missing one or more matched actions at a decision point')
    sensitivity, rankings = [], []
    for fw in forgetting_weights:
        for cw in cost_weights:
            scored = df.assign(score=(df.delta_ap - fw * df.forgetting_ap -
                                       cw * df.adaptation_seconds))
            winners = []
            for _, part in scored.groupby(group_keys, sort=True):
                part = part.sort_values(['score', 'action'], ascending=[False, True])
                top = part.iloc[0]
                margin = float(top.score - part.iloc[1].score)
                winners.append({'forgetting_lambda': fw, 'cost_lambda': cw,
                                'group': int(top.group),
                                'action_year': top.action_year,
                                'winner': top.action,
                                'winner_margin': margin,
                                'near_tie_1e_4': bool(margin < 1e-4),
                                'best_reward': float(top.score),
                                'worst_reward': float(part.iloc[-1].score)})
            rankings.extend(winners)
            for action, part in scored.groupby('action'):
                sensitivity.append({'forgetting_lambda': fw, 'cost_lambda': cw,
                                    'action': action,
                                    'matched_decision_points': len(part),
                                    'mean_reward': float(part.score.mean()),
                                    'mean_delta_ap': float(part.delta_ap.mean()),
                                    'mean_forgetting_ap': float(part.forgetting_ap.mean()),
                                    'mean_adaptation_seconds': float(part.adaptation_seconds.mean()),
                                    'win_count': sum(w['winner'] == action for w in winners)})
    return pd.DataFrame(sensitivity), pd.DataFrame(rankings)


def run_c3_action_study(net, cfg: Config, output_dir, options=None,
                        groups=3, holdout_groups=1,
                        forgetting_weights=(0., .25, 1.), cost_weights=(0., .001)):
    options = options or ControllerOptions(state_version='c2')
    options.validate()
    if cfg.feature_track != 'history':
        raise ValueError('C3 requires --feature-track history to avoid unknown feature-time leakage')
    if not cfg.use_gcn or not cfg.use_ewc or not cfg.use_replay:
        raise ValueError('C3 action comparison needs GCN, EWC and replay enabled')
    if not (2 <= groups <= net.n // 2 and 1 <= holdout_groups < groups):
        raise ValueError('Invalid communities/holdout-communities')
    if not forgetting_weights or not cost_weights:
        raise ValueError('Reward grids cannot be empty')
    if any(not np.isfinite(x) or x < 0 for x in (*forgetting_weights, *cost_weights)):
        raise ValueError('Reward weights must be finite and nonnegative')
    max_t = len(net.times) if cfg.max_snapshots <= 0 else min(cfg.max_snapshots, len(net.times))
    train_end = max_t - options.test_last
    if train_end - cfg.start_time_index < cfg.warmup + 2:
        raise ValueError('Not enough training years for C3 delayed matched-action rewards')
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    nodes = historical_partition(net, groups, cfg.start_time_index, cfg.seed)
    permutation = np.random.default_rng(cfg.seed + 7129).permutation(groups).tolist()
    train_ids = sorted(permutation[:groups - holdout_groups])
    holdout_ids = sorted(permutation[groups - holdout_groups:])
    all_rows = []
    for group in train_ids:
        print(f'[C3] matched-action study on meta-training group {group}', flush=True)
        subgraph = make_group_network(net, nodes[group], group)
        # Match C2's per-group seeding protocol.
        per_group_cfg = replace(cfg, seed=cfg.seed + 1009 * group)
        all_rows.extend(_matched_group(subgraph, per_group_cfg, options, group, train_end))
    df = pd.DataFrame(all_rows)
    if df.empty or int(df.reward_time_index.max()) >= train_end:
        raise RuntimeError('No valid training-era action pairs or leaked test-era reward')
    sensitivity, rankings = summarize_action_study(df, forgetting_weights, cost_weights)
    df.to_csv(output / 'matched_action_outcomes.csv', index=False)
    sensitivity.to_csv(output / 'reward_sensitivity.csv', index=False)
    rankings.to_csv(output / 'action_rankings.csv', index=False)
    manifest = {
        'protocol': 'C3 matched-state single-step fork; balanced reference trajectory; training-only t+1 validation',
        'dataset': net.name, 'config': asdict(cfg), 'options': asdict(options),
        'partition_history_cutoff_index': cfg.start_time_index,
        'training_years': [str(x) for x in net.times[cfg.start_time_index:train_end]],
        'sealed_test_years': [str(x) for x in net.times[train_end:max_t]],
        'meta_train_groups': train_ids, 'meta_holdout_groups_never_processed': holdout_ids,
        'group_node_counts': [len(ids) for ids in nodes],
        'cross_group_edges_omitted': True,
        'reward_grid_forgetting': list(forgetting_weights),
        'reward_grid_cost': list(cost_weights),
        'limitations': [
            'Offline diagnostic only: the best next-year action is a hindsight oracle, not a deployable policy',
            'Action effects are one-step effects from a fixed balanced reference path, not long-horizon returns',
            'All comparisons share the same pre-update model/optimizer/EWC/replay at each year',
            'Wall-clock cost is hardware/order-dependent; compare reward with cost_lambda=0 as sensitivity',
            'The meta-holdout groups and sealed test years are not read for reward or model selection',
            'Training-year validation labels must be revealed before calculating delayed reward',
            'Partitioning removes cross-group links and changes the benchmark population',
        ],
    }
    (output / 'c3_manifest.json').write_text(json.dumps(manifest, indent=2))
    summary = {
        'matched_decision_points': len(df) // len(ACTIONS),
        'matched_action_outcomes': len(df),
        'train_groups': train_ids, 'untouched_meta_holdout_groups': holdout_ids,
        'last_reward_time_index': int(df.reward_time_index.max()),
        'first_sealed_test_time_index': train_end,
        'mean_action_delta_ap': df.groupby('action').delta_ap.mean().to_dict(),
        'mean_action_forgetting_ap': df.groupby('action').forgetting_ap.mean().to_dict(),
        'reward_grid_settings': len(forgetting_weights) * len(cost_weights),
        'matched_decisions_with_distinct_delta_ap_gt_0_001': int(
            (df.groupby(['group', 'action_time_index']).delta_ap
               .agg(lambda x: float(x.max() - x.min())) > .001).sum()),
        'reward_settings_near_tie_fraction': float(rankings.near_tie_1e_4.mean()),
        'controller_trained': False,
        'final_test_accessed': False,
    }
    (output / 'c3_summary.json').write_text(json.dumps(summary, indent=2))
    return summary
