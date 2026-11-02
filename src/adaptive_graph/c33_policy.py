"""C3.3: regularized one-step contextual policy vs temporal DQN in closed loop.

The contextual policy is learned ONLY from C3 matched-action, training-era
outcomes on a fixed balanced reference trajectory.  A multi-output ridge model
predicts all four one-step rewards from the 16 C2 history/revealed-label state
features.  Alpha is selected with leave-one-training-group-out validation.

The DQN uses actual chronological action -> next-year state transitions from
fixed-policy off-policy episodes and online training episodes on the same
meta-training groups, never using final-test rewards.  The held-out author
partition is not used for either policy's training.  Both controllers are
frozen in that partition while their GCN learners still adapt using permissible
training-year labels.  The last test years have no GCN/policy adaptation.

Matched-action labels are NOT valid causal multi-step DQN transitions; this
module deliberately does not feed them into the DQN.
"""
from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
import csv
import json

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .c3_diagnostics import _matched_group
from .experiment import Config
from .meta_environment import historical_partition, make_group_network
from .temporal_control import (ACTIONS, ACTION_INDEX, C2_STATE_SIZE,
                               ControllerOptions, TemporalDQN,
                               run_temporal_controller)

FEATURES = [f"state_{i}" for i in range(C2_STATE_SIZE)]
ACTIONS_ORDERED = [a.name for a in ACTIONS]


class ContextualRidge:
    """Frozen multi-output ridge Q(s, action) for immediate *reward*, not returns."""
    def __init__(self, fitted):
        self.fitted = fitted

    def predict_rewards(self, state):
        x = np.asarray(state, dtype=float).reshape(1, -1)
        if x.shape[1] != C2_STATE_SIZE or not np.isfinite(x).all():
            raise ValueError('Expected finite C2 state of 16 features')
        scores = self.fitted.predict(x).reshape(-1)
        if scores.size != len(ACTIONS):
            raise RuntimeError('Contextual policy returned wrong number of actions')
        return scores


def _paired_action_data(df, forgetting_lambda: float, cost_lambda: float):
    """One row per decision, four observed rewards; no hindsight winner leakage."""
    needed = {'group', 'action_time_index', 'action', 'reward_time_index',
              'delta_ap', 'forgetting_ap', 'adaptation_seconds', *FEATURES}
    if missing := needed.difference(df.columns):
        raise ValueError(f'Missing causal C3 observation fields: {sorted(missing)}. '
                         'Regenerate C3 outcomes using the C3.3 upgrade.')
    if df.empty or df[list(FEATURES)].isna().any().any():
        raise ValueError('Missing / NaN C3 matched-action states')
    states, rewards, groups, years = [], [], [], []
    for (group, year), part in df.groupby(['group', 'action_time_index'], sort=True):
        if len(part) != len(ACTIONS) or set(part.action) != set(ACTIONS_ORDERED):
            raise ValueError('Each decision must have exactly four distinct actions')
        if part.reward_time_index.nunique() != 1 or not (part.reward_time_index.iloc[0] == year + 1):
            raise ValueError('Matched rewards must refer to the next chronological year')
        vectors = part[FEATURES].to_numpy(dtype=float)
        if not np.allclose(vectors, vectors[0], rtol=0, atol=0):
            raise ValueError('Matched actions must have identical decision state')
        sorted_part = part.set_index('action').loc[ACTIONS_ORDERED]
        r = (sorted_part.delta_ap.to_numpy(float)
             - forgetting_lambda * sorted_part.forgetting_ap.to_numpy(float)
             - cost_lambda * sorted_part.adaptation_seconds.to_numpy(float))
        if not np.isfinite(r).all() or not np.isfinite(vectors).all():
            raise ValueError('All rewards and state values must be finite')
        states.append(vectors[0]); rewards.append(r)
        groups.append(int(group)); years.append(int(year))
    return (np.stack(states), np.stack(rewards), np.array(groups), np.array(years))


def train_contextual_policy(df: pd.DataFrame, forgetting_lambda=.25,
                            cost_lambda=.001, alphas=(.01, .1, 1., 10., 100.)):
    """Select ridge alpha on TRAIN-GROUP CV only; no held-out scores consulted."""
    if forgetting_lambda < 0 or cost_lambda < 0:
        raise ValueError('Reward weights must be nonnegative')
    x, y, groups, years = _paired_action_data(df, forgetting_lambda, cost_lambda)
    if len(x) < 2:
        raise ValueError('At least two matched training-era decisions are needed')
    unique_groups = np.unique(groups)
    cv = []
    for alpha in alphas:
        if alpha <= 0 or not np.isfinite(alpha):
            raise ValueError('Ridge strengths must be finite and positive')
        errors = []
        if len(unique_groups) > 1:
            for holdout in unique_groups:
                tr, va = (groups != holdout), (groups == holdout)
                model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
                model.fit(x[tr], y[tr])
                predictions = model.predict(x[va])
                errors.extend(np.mean((predictions - y[va]) ** 2, axis=1).tolist())
        else:
            # No independent group CV possible: avoid bogus in-sample alpha choice.
            errors = [float('nan')]
        cv.append({'alpha': float(alpha),
                   'group_cv_mse': float(np.mean(errors)),
                   'n_groups': int(len(unique_groups)), 'n_decisions': int(len(x))})
    if len(unique_groups) > 1:
        best_alpha = min(cv, key=lambda row: (row['group_cv_mse'], row['alpha']))['alpha']
        selection = 'leave_one_training_group_out_mse'
    else:
        best_alpha = 10.
        selection = 'fixed_alpha_insufficient_training_groups'
    fitted = make_pipeline(StandardScaler(), Ridge(alpha=best_alpha))
    fitted.fit(x, y)
    predicted = fitted.predict(x)
    train_diag = {
        'matched_training_decisions': len(x),
        'matched_training_action_outcomes': len(x) * len(ACTIONS),
        'training_groups': unique_groups.tolist(),
        'ridge_alpha': float(best_alpha), 'ridge_selection': selection,
        'train_reward_mse_in_sample': float(np.mean((predicted-y)**2)),
        'train_action_match_in_sample': float(np.mean(predicted.argmax(1) == y.argmax(1))),
        'training_decision_year_indices': sorted(np.unique(years).tolist()),
        'limitation': 'In-sample diagnostics do not measure generalization; C3 data come from balanced reference states.'
    }
    return ContextualRidge(fitted), pd.DataFrame(cv), train_diag


def _write_csv(path, records):
    if not records:
        return
    pd.DataFrame(records).to_csv(path, index=False)


def _retention_mean(path):
    loc = Path(path) / 'retention.csv'
    if not loc.exists():
        return None
    retention = pd.read_csv(loc)
    return float(retention.drop_ap.mean()) if len(retention) else None


def run_c33(net, cfg: Config, output_dir, options=None, groups=3,
            holdout_groups=1, ridge_alphas=(.01, .1, 1., 10., 100.),
            dqn_offpolicy=True, independent_validation=False):
    """Full training-only contextual/DQN fit + frozen-policy closed-loop comparison."""
    options = options or ControllerOptions(state_version='c2', policy='dqn')
    options.validate()
    if options.state_version != 'c2' or options.policy != 'dqn':
        raise ValueError('C3.3 requires C2 observations and a DQN policy')
    if cfg.feature_track != 'history':
        raise ValueError('C3.3 requires history-only features')
    if not cfg.use_gcn or not cfg.use_ewc or not cfg.use_replay:
        raise ValueError('C3.3 requires GCN, EWC, and replay enabled')
    if not (2 <= groups <= net.n // 2 and 1 <= holdout_groups < groups):
        raise ValueError('Invalid groups or heldout group count')
    max_t = len(net.times) if cfg.max_snapshots <= 0 else min(len(net.times), cfg.max_snapshots)
    train_end = max_t - options.test_last
    if train_end - cfg.start_time_index < cfg.warmup + 2:
        raise ValueError('Not enough chronology for training-era controller rewards')
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    nodes = historical_partition(net, groups, cfg.start_time_index, cfg.seed)
    permutation = np.random.default_rng(cfg.seed + 7129).permutation(groups).tolist()
    train_ids = sorted(permutation[:groups-holdout_groups])
    holdout_ids = sorted(permutation[groups-holdout_groups:])
    subgraphs = [make_group_network(net, n, i) for i, n in enumerate(nodes)]
    manifest = {
        'protocol': 'C3.3 frozen contextual-ridge / temporal DQN / four fixed-policy closed-loop comparison',
        'seed': cfg.seed,
        'dataset': net.name,
        'training_groups': train_ids,
        'policy_holdout_groups': holdout_ids,
        'history_cutoff': cfg.start_time_index,
        'training_years': [str(t) for t in net.times[cfg.start_time_index:train_end]],
        'frozen_test_years': [str(t) for t in net.times[train_end:max_t]],
        'original_events': len(net.events),
        'within_group_events': sum(len(g.events) for g in subgraphs),
        'cross_group_edges_omitted': True,
        'config': asdict(cfg),
        'controller_options': asdict(options),
        'dqn_offpolicy_rollouts': dqn_offpolicy,
        'limitations': [
            ('Independent evaluation requires new, unseen years and locked configuration.' if independent_validation else 'DBLP 2009-2010 were used during previous development: this is NOT an untouched final test.'),
            'Matched-action contextual training labels come from a fixed balanced reference state distribution.',
            'DQN off-policy episodes represent correlated temporal experiences, not independent graph histories.',
            'Wall-time cost is hardware-dependent; hold cost penalty fixed across comparison.',
            'Node-disjoint groups remove cross-group links and change the evaluation population.',
            'Model/hyperparameter selection cannot use held-out test AP.',
        ],
    }
    (out / 'c33_manifest.json').write_text(json.dumps(manifest, indent=2))

    matched = []
    for i in train_ids:
        print(f'[C3.3] matched action training group {i}', flush=True)
        local_cfg = replace(cfg, seed=cfg.seed + 1009*i)
        matched.extend(_matched_group(subgraphs[i], local_cfg, options, i, train_end))
    df = pd.DataFrame(matched)
    if df.empty or int(df.reward_time_index.max()) >= train_end:
        raise RuntimeError('C3.3 requires matched training-year rewards only')
    df.to_csv(out / 'matched_action_training.csv', index=False)
    context, cv, context_diag = train_contextual_policy(
        df, options.forgetting_lambda, options.cost_lambda, ridge_alphas)
    cv.to_csv(out / 'contextual_group_cv.csv', index=False)
    (out / 'contextual_training.json').write_text(json.dumps(context_diag, indent=2))
    scaler, reg = context.fitted.steps[0][1], context.fitted.steps[1][1]
    np.savez(out / 'contextual_model.npz', mean=scaler.mean_, scale=scaler.scale_,
             coef=reg.coef_, intercept=reg.intercept_,
             alpha=np.array([context_diag['ridge_alpha']]))

    # DQN training uses *real* chronological transitions from complete fixed
    # trajectories followed by online DQN trajectories, not C3 matched branches.
    torch.manual_seed(cfg.seed + 35001)
    dqn = TemporalDQN(options, cfg.seed, cfg.device)
    dqn_rows = []
    offpolicy_count = 0
    for i in train_ids:
        per_cfg = replace(cfg, seed=cfg.seed + 1009*i)
        if dqn_offpolicy:
            for action in ACTION_INDEX:
                transitions = []
                fixed_options = replace(options, policy='fixed', fixed_action=action)
                result = run_temporal_controller(
                    subgraphs[i], per_cfg, out / f'dqn_offpolicy_group_{i}' / action,
                    fixed_options, train_policy=False, transition_sink=transitions,
                    evaluate_test=False)
                if result['reward_transitions'] != len(transitions):
                    raise RuntimeError('DQN off-policy transition audit mismatch')
                for s, a, reward, ns, terminal in transitions:
                    dqn.observe(s, a, reward, ns, terminal)
                    offpolicy_count += 1
                dqn_rows.append({'group': i, 'episode': f'fixed_{action}',
                                 'reward_transitions': len(transitions),
                                 'policy_grad_updates_cumulative': dqn.update_count})
        learned = run_temporal_controller(
            subgraphs[i], per_cfg, out / f'dqn_online_group_{i}', options,
            controller=dqn, train_policy=True, evaluate_test=False)
        dqn_rows.append({'group': i, 'episode': 'dqn_online',
                         'reward_transitions': learned['reward_transitions'],
                         'policy_grad_updates_cumulative': dqn.update_count})
    dqn_training_updates = dqn.update_count
    _write_csv(out / 'dqn_training_episodes.csv', dqn_rows)
    torch.save({'policy_q': dqn.q.state_dict(), 'target_q': dqn.target.state_dict(),
                'trained_only_on_groups': train_ids, 'state_version': 'c2'},
               out / 'dqn_frozen.pt')

    # The heldout policy weights remain frozen, but the GCN trains using the
    # permissible years in each heldout group. All methods use identical GCN
    # initialization per group and the same sampled candidate protocol.
    comparison = []
    for i in holdout_ids:
        per_cfg = replace(cfg, seed=cfg.seed + 1009*i)
        variants = [('contextual_ridge', replace(options, policy='fixed'), context),
                    ('dqn_frozen', options, None),
                    ('random', replace(options, policy='random'), None)] + [
                    (f'fixed_{a}', replace(options, policy='fixed', fixed_action=a), None)
                    for a in ACTION_INDEX]
        for label, opts, selector in variants:
            print(f'[C3.3] heldout group {i}: {label}', flush=True)
            folder = out / f'holdout_group_{i}' / label
            run = run_temporal_controller(
                subgraphs[i], per_cfg, folder, opts,
                controller=dqn if label == 'dqn_frozen' else None,
                train_policy=False, greedy_policy=True, policy_selector=selector)
            if dqn.update_count != dqn_training_updates:
                raise RuntimeError('Heldout group updated DQN weights')
            if run['controller_gradient_updates'] != 0:
                raise RuntimeError('Controller gradients occurred during holdout')
            comparison.append({'seed': cfg.seed, 'group': i, 'policy': label,
                               'test_mean_ap': run['test_mean_ap'],
                               'test_mean_roc_auc': run['test_mean_roc_auc'],
                               'train_prequential_mean_ap': run['train_prequential_mean_ap'],
                               'mean_historical_ap_drop': _retention_mean(folder),
                               'policy_actions': run['policy_actions'],
                               'chosen_actions': json.dumps(run['chosen_actions']),
                               'controller_gradient_updates_holdout': run['controller_gradient_updates']})
    _write_csv(out / 'holdout_comparison.csv', comparison)
    summary = {
        'seed': cfg.seed,
        'training_groups': train_ids, 'heldout_groups': holdout_ids,
        'matched_training_decisions': context_diag['matched_training_decisions'],
        'contextual_cv_selected_alpha': context_diag['ridge_alpha'],
        'dqn_offpolicy_transitions': offpolicy_count,
        'dqn_online_transitions': int(sum(r['reward_transitions'] for r in dqn_rows
                                          if r['episode'] == 'dqn_online')),
        'dqn_training_gradient_updates': dqn_training_updates,
        'heldout_controller_updates': 0,
        'heldout_comparison': comparison,
        'first_frozen_test_index': train_end,
        'last_train_reward_time_index': int(df.reward_time_index.max()),
        'not_fresh_publication_test': not independent_validation,
    }
    (out / 'c33_summary.json').write_text(json.dumps(summary, indent=2))
    return summary
