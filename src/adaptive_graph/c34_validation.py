"""C3.4: preregistered, audit-oriented independent temporal evaluation.

The lock sees year values and file bytes, never graph labels/score results. Evaluation
loads a locked CSV and rejects modified code/config/data before running C3.3's
chronological, held-out-author 7-policy comparison. No policy selection is made
using final-test results. Statistical comparisons pair policies within seeds.

A lock is NOT a proof of scientific independence: users must select a genuinely
unseen version/source/time period, and never iterate settings on its test scores.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from itertools import product
from pathlib import Path
import json
import math
import os

import numpy as np
import pandas as pd
from scipy.stats import t as t_distribution

from .c33_policy import run_c33
from .data import TemporalNetwork
from .experiment import Config
from .temporal_control import ControllerOptions

# SHA-256 of code files: lock and execute must use exactly the same sources.
SOURCE_FILES = (
    'data.py', 'features.py', 'models.py', 'metrics.py', 'continual.py',
    'experiment.py', 'rl.py', 'temporal_control.py', 'meta_environment.py',
    'c3_diagnostics.py', 'c33_policy.py', 'c34_validation.py', 'cli.py',
)
POLICIES = ('contextual_ridge', 'dqn_frozen', 'random', 'fixed_plastic',
            'fixed_balanced', 'fixed_ewc_focus', 'fixed_replay_focus')


@dataclass(frozen=True)
class FrozenProtocol:
    seeds: tuple = (7, 13, 42, 101, 202)
    start_time_index: int = 3
    max_nodes: int = 5000
    communities: int = 3
    holdout_communities: int = 1
    max_positives: int = 500
    max_train_pairs: int = 1024
    epochs: int = 3
    hidden: int = 32
    warmup: int = 1
    gamma: float = .9
    controller_updates: int = 2
    reward_forgetting_lambda: float = .25
    reward_cost_lambda: float = 0.
    test_last: int = 2
    ridge_alphas: tuple = (.01, .1, 1., 10., 100.)

    def validate(self):
        if len(self.seeds) < 2 or len(set(self.seeds)) != len(self.seeds):
            raise ValueError('Need at least two distinct predeclared seeds')
        if self.start_time_index < 1 or self.epochs < 1 or self.max_nodes < 0:
            raise ValueError('Invalid cutoff/epochs/max_nodes')
        if self.test_last != 2:
            raise ValueError('C3.4 currently requires exactly two final years')
        if not 2 <= self.communities or not 1 <= self.holdout_communities < self.communities:
            raise ValueError('Invalid author-group configuration')
        if not 0 <= self.gamma < 1 or self.reward_forgetting_lambda < 0 or self.reward_cost_lambda < 0:
            raise ValueError('Invalid reward/controller options')
        if self.hidden < 1 or self.max_positives < 1 or self.max_train_pairs < 1:
            raise ValueError('Invalid training sizes')
        if not self.ridge_alphas or min(self.ridge_alphas) <= 0:
            raise ValueError('Ridge alphas must be positive')


def file_sha256(path):
    h = sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def code_hashes():
    root = Path(__file__).resolve().parent
    return {p: file_sha256(root / p) for p in SOURCE_FILES}


def scan_years(path, column='timestamp'):
    """Read only the year column; does not examine edge endpoints or labels."""
    counts = {}
    for chunk in pd.read_csv(path, usecols=[column], chunksize=200_000):
        numeric = pd.to_numeric(chunk[column], errors='raise')
        if numeric.isna().any() or (numeric % 1 != 0).any():
            raise ValueError('C3.4 requires integer publication-year timestamps')
        if ((numeric < 1800) | (numeric > 2200)).any():
            raise ValueError('Years outside expected publication range')
        for year, count in numeric.astype(int).value_counts().items():
            counts[int(year)] = counts.get(int(year), 0) + int(count)
    if len(counts) < 5:
        raise ValueError('Need at least five distinct annual periods')
    return dict(sorted(counts.items()))


def create_lock(csv_path, lock_path, *, source, seen_through=2010,
                protocol=None, time_column='timestamp'):
    """Create immutable experiment preregistration before test scoring."""
    protocol = protocol or FrozenProtocol()
    protocol.validate()
    csv_path = Path(csv_path)
    lock_path = Path(lock_path)
    if lock_path.exists():
        raise FileExistsError(f'Lock already exists: {lock_path}')
    counts = scan_years(csv_path, time_column)
    years = list(counts)
    final_years = years[-protocol.test_last:]
    if min(final_years) <= seen_through:
        raise ValueError(f'Final years {final_years} were previously used (seen-through={seen_through}); provide new data')
    if final_years[1] != final_years[0] + 1:
        raise ValueError('Final test years must be consecutive')
    # All events are available in the supplied source; the chronologically
    # final two years stay sealed within the C3.3 training/evaluation engine.
    if len(years) - protocol.test_last - protocol.start_time_index < protocol.warmup + 2:
        raise ValueError('Too few historical years for chronological rewards')
    lock = {
        'version': 'c34-protocol-v1',
        'source': str(source),
        'csv_basename': csv_path.name,
        'csv_sha256': file_sha256(csv_path),
        'time_column': time_column,
        # Only year existence, not held-out positive counts, is preregistered.
        'observed_years': years,
        'training_year_counts': {str(k): v for k, v in counts.items() if k not in final_years},
        'final_test_years': final_years,
        'seen_through': int(seen_through),
        'protocol': asdict(protocol),
        'code_sha256': code_hashes(),
        'primary_metric': 'test_mean_ap',
        'primary_comparison': 'dqn_frozen - fixed_ewc_focus',
        'secondary_comparisons': ['dqn_frozen - fixed_plastic',
                                  'dqn_frozen - random',
                                  'contextual_ridge - fixed_ewc_focus'],
        'train_only_selection': 'training-author group CV for ridge; no test-year policy tuning',
        'safety_notes': [
            'Prior DBLP 2009-2010 scores are development, not independent test.',
            'The chosen final years must truly be unseen to the researcher.',
            'Historical author-subgraph selection produces biased exploratory results.',
            'Only one locked final evaluation should be used for research claims.',
            'All policies must use identical candidate sets and initialization within each seed/group.',
            'Five seeds give large uncertainty; do not interpret p > .05 as equivalence.',
            'Runtime cost estimates are hardware-dependent.',
        ],
    }
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(json.dumps(lock, indent=2))
    return lock


def verify_lock(csv_path, lock_path):
    lock = json.loads(Path(lock_path).read_text())
    if lock.get('version') != 'c34-protocol-v1':
        raise ValueError('Not a C3.4 lock file')
    if file_sha256(csv_path) != lock['csv_sha256']:
        raise ValueError('Dataset changed after preregistration; final evaluation blocked')
    if code_hashes() != lock['code_sha256']:
        raise ValueError('Source code changed after preregistration; final evaluation blocked')
    observed = scan_years(csv_path, lock['time_column'])
    if list(observed) != lock['observed_years'] or {
            str(k): v for k, v in observed.items() if k not in lock['final_test_years']} != lock['training_year_counts']:
        raise ValueError('Year chronology or training-year counts changed')
    protocol = FrozenProtocol(**{**lock['protocol'],
                                 'seeds': tuple(lock['protocol']['seeds']),
                                 'ridge_alphas': tuple(lock['protocol']['ridge_alphas'])})
    protocol.validate()
    return lock, protocol


def _paired_comparisons(rows):
    """Seed-level pairing; not misleading row-level pseudoreplication."""
    df = pd.DataFrame(rows)
    required = {'seed', 'policy', 'test_mean_ap', 'test_mean_roc_auc', 'mean_historical_ap_drop'}
    if missing := required - set(df):
        raise ValueError(f'Missing evaluation columns: {sorted(missing)}')
    if df[['seed', 'policy']].duplicated().any():
        raise ValueError('More than one heldout group per policy/seed requires explicit group aggregation')
    wide = df.pivot(index='seed', columns='policy', values='test_mean_ap')
    if wide.isna().any().any() or set(wide.columns) != set(POLICIES):
        raise ValueError('Missing policy score for a seed')
    out = []
    comparisons = [('dqn_frozen','fixed_ewc_focus'),('dqn_frozen','fixed_plastic'),
                   ('dqn_frozen','random'),('contextual_ridge','fixed_ewc_focus')]
    for a,b in comparisons:
        diff = (wide[a] - wide[b]).to_numpy(float)
        n = len(diff)
        average = float(np.mean(diff))
        std = float(np.std(diff, ddof=1)) if n > 1 else float('nan')
        margin = float(t_distribution.ppf(.975, n-1) * std / math.sqrt(n)) if n > 1 else float('nan')
        # Exact sign-flip permutation over seeds (descriptive, extremely low power for n=5).
        signed_means = [abs(float(np.mean(diff * np.array(signs))))
                        for signs in product((-1,1), repeat=n)]
        p_value = sum(v >= abs(average)-1e-12 for v in signed_means) / len(signed_means)
        out.append({'policy_a': a, 'policy_b': b, 'n_paired_seeds': n,
                    'mean_ap_difference': average, 'sd_paired_difference': std,
                    'ci95_low': average-margin, 'ci95_high': average+margin,
                    'exact_sign_flip_p_two_sided': p_value,
                    'win_seeds': int((diff > 0).sum()), 'tie_seeds': int((diff == 0).sum()),
                    'loss_seeds': int((diff < 0).sum())})
    return df, pd.DataFrame(out)


def evaluate_locked(csv_path, lock_path, output_dir, device='cpu'):
    """Only executable protocol: frozen config, all seven policies, declared seeds."""
    lock, spec = verify_lock(csv_path, lock_path)
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f'Final evaluation output exists: {output_dir}. Never overwrite a registered final test.')
    output_dir.mkdir(parents=True)
    (output_dir / 'registered_protocol.json').write_text(json.dumps(lock, indent=2))
    net = TemporalNetwork.from_csv(csv_path, time=lock['time_column'], time_bucket='year')
    if [int(y) for y in net.times[-2:]] != lock['final_test_years']:
        raise ValueError('Final test chronology differs from lock')
    opts = ControllerOptions(policy='dqn', state_version='c2',
                             gamma=spec.gamma, test_last=spec.test_last,
                             controller_updates=spec.controller_updates,
                             forgetting_lambda=spec.reward_forgetting_lambda,
                             cost_lambda=spec.reward_cost_lambda)
    all_rows = []
    for seed in spec.seeds:
        print(f'[C3.4] registered seed={seed}, final={lock["final_test_years"]}', flush=True)
        sub = net.induced_subset(spec.max_nodes, seed=seed,
                                 selection_history_idx=spec.start_time_index)
        cfg = Config(seed=seed, start_time_index=spec.start_time_index,
                     epochs=spec.epochs, hidden=spec.hidden, warmup=spec.warmup,
                     max_positives=spec.max_positives,
                     max_train_pairs=spec.max_train_pairs,
                     feature_track='history', device=device)
        summary = run_c33(sub, cfg, output_dir / f'seed_{seed}', opts,
                          groups=spec.communities,
                          holdout_groups=spec.holdout_communities,
                          ridge_alphas=spec.ridge_alphas,
                          dqn_offpolicy=True, independent_validation=True)
        all_rows.extend(summary['heldout_comparison'])
    df, pairs = _paired_comparisons(all_rows)
    df.to_csv(output_dir / 'c34_seed_comparison.csv', index=False)
    pairs.to_csv(output_dir / 'c34_paired_differences.csv', index=False)
    aggregate = df.groupby('policy', as_index=False).agg(
        n_seed_runs=('seed','nunique'),
        mean_test_ap=('test_mean_ap','mean'), sd_test_ap=('test_mean_ap','std'),
        mean_test_auc=('test_mean_roc_auc','mean'),
        mean_historical_ap_drop=('mean_historical_ap_drop','mean'))
    aggregate.to_csv(output_dir / 'c34_policy_summary.csv', index=False)
    results = {'protocol': 'C3.4 pre-registered independent evaluation',
               'dataset_sha256': lock['csv_sha256'],
               'final_test_years': lock['final_test_years'],
               'n_seeds': len(spec.seeds),
               'primary_comparison': lock['primary_comparison'],
               'test_observed_once': True,
               'warnings': lock['safety_notes']}
    (output_dir / 'c34_summary.json').write_text(json.dumps(results, indent=2))
    return results
