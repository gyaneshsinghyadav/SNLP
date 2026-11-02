#!/usr/bin/env python3
"""E6-C full structural-temporal fusion, matched E6-B temporal DBLP protocol.

Trainable TGAT/structural encoders + gated cross-interactions + prequential LR
logit anchor. All fitting, replay, model selection use training-era labels only.
The last two years are test-only; model and LR anchor weights remain frozen.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import torch
from torch.nn import functional as F

from adaptive_graph.data import create_task
from adaptive_graph.tgat import TemporalEventStream
from adaptive_graph.e6_structural import HistoricalPairFeatures, FEATURE_NAMES
from adaptive_graph.e6_fusion import FullStructuralTemporalFusion
from adaptive_graph.temporal_control import split_controller_validation
from e6_run_suite import DEFAULT_E4_SHA, digest_candidates, sha

MODES = ('fusion', 'structural', 'temporal', 'fusion_no_recency', 'fusion_no_anchor')


@dataclass(frozen=True)
class Settings:
    start_time_index: int = 3
    test_last: int = 2
    max_positives: int = 200
    negative_ratio: float = 1.
    eval_fraction: float = .3
    controller_val_fraction: float = .25
    decay: float = .7
    neighbors: int = 12
    heads: int = 4
    hidden: int = 32
    fusion_dim: int = 48
    dropout: float = .1
    epochs: int = 6
    patience: int = 3
    batch_size: int = 128
    max_replay_pairs: int = 256
    replay_years: int = 5
    lr_tgat: float = .0007
    lr_decoder: float = .002
    weight_decay: float = .001
    gradient_clip: float = 2.
    logistic_c: float = 1.
    pairwise_rank_weight: float = .05


@dataclass
class Snapshot:
    year: str
    ti: int
    pt: object
    structural: torch.Tensor
    y: np.ndarray
    fit_idx: np.ndarray
    val_idx: np.ndarray
    prior: dict[str, torch.Tensor]


def _seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _lr_features(x: np.ndarray, mode: str) -> np.ndarray:
    if mode == 'fusion_no_recency':
        out = x.copy()
        out[:, 8:] = 0.
        return out
    return x


def _prior_logits(history_x: list[np.ndarray], history_y: list[np.ndarray],
                  x: np.ndarray, seed: int, mode: str) -> np.ndarray:
    if mode in ('temporal', 'fusion_no_anchor'):
        return np.zeros(len(x), dtype=np.float32)
    if not history_x:
        return np.zeros(len(x), dtype=np.float32)
    xx = np.concatenate(history_x)
    yy = np.concatenate(history_y)
    if len(np.unique(yy)) < 2:
        return np.zeros(len(x), dtype=np.float32)
    prior = make_pipeline(StandardScaler(), LogisticRegression(
        C=1., max_iter=350, solver='lbfgs', random_state=seed))
    prior.fit(_lr_features(xx, mode), yy)
    # sklearn decision_function is in log-odds, hence valid as a residual anchor.
    return np.asarray(prior.decision_function(_lr_features(x, mode)), dtype=np.float32)


def make_snapshots(stream: TemporalEventStream, seed: int, settings: Settings,
                   modes: tuple[str, ...], device: str):
    """Build E6-B-exact task indices and *prequential* LR priors.

    A year t prior is fitted only on fitting-subset labels observed in years < t.
    Its values remain cached unchanged even after labels from t become known.
    """
    net = stream.network
    train_end = len(net.times) - settings.test_last
    if train_end <= settings.start_time_index + 2:
        raise ValueError('Insufficient historical years')
    histories_x, histories_y = [], []
    snapshots, audit, keys = [], [], []
    for ti in range(settings.start_time_index, len(net.times)):
        phase = 'test' if ti >= train_end else 'train'
        task = create_task(net, ti, seed=seed, max_positives=settings.max_positives,
                           negative_ratio=settings.negative_ratio,
                           eval_fraction=settings.eval_fraction)
        part = (split_controller_validation(task, settings.controller_val_fraction, seed)
                if phase == 'train' else None)
        if phase == 'train' and part is None:
            raise RuntimeError(f'Insufficient train/validation candidates at {task.time_value}')
        fit_idx, val_idx = (part if part is not None else (
            np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)))
        used = set(fit_idx.tolist()) | set(val_idx.tolist())
        if len(used) != len(fit_idx) + len(val_idx) or used & set(task.eval_idx.tolist()):
            raise RuntimeError('Training/validation/evaluation index overlap')
        hist = HistoricalPairFeatures(net, ti, decay=settings.decay)
        structural = hist.transform(task.candidates)
        pt = stream.prepare(task, neighbors=settings.neighbors, device=device)
        prior = {
            m: torch.tensor(_prior_logits(histories_x, histories_y, structural, seed, m),
                            dtype=torch.float32, device=device) for m in modes
        }
        seen = {tuple(row) for row in task.historical_edges.tolist()}
        overlap = sum(tuple(sorted(map(int, row))) in seen for row in task.candidates)
        if overlap:
            raise RuntimeError(f'Historical candidate leakage at {task.time_value}')
        if pt.historical_event_count > int((stream.timestamps < pt.cutoff).sum()):
            raise RuntimeError('Unexpected temporal event count')
        snapshots.append(Snapshot(str(task.time_value), ti, pt,
                                  torch.as_tensor(structural, device=device),
                                  task.labels.copy(), fit_idx, val_idx, prior))
        audit.append({'seed':seed, 'time':str(task.time_value), 'phase':phase,
                      'history_events':hist.historical_event_count,
                      'tgat_history_events':pt.historical_event_count,
                      'train_candidates':int(len(fit_idx)),
                      'controller_validation_candidates':int(len(val_idx)),
                      'sealed_test_candidates':int(len(task.eval_idx)),
                      'positive_count':int(task.labels.sum()),
                      'candidate_count':int(len(task.labels)),
                      'label_overlap_count':0, 'historical_candidate_overlap':overlap})
        keys.append({'seed':seed,'time':str(task.time_value),
                     'candidate_task_sha256':digest_candidates(task)})
        if phase == 'train':
            # Only the fitting subset enters any anchor model, never eval or val.
            histories_x.append(structural[fit_idx].copy())
            histories_y.append(task.labels[fit_idx].copy())
    return snapshots, pd.DataFrame(audit), pd.DataFrame(keys)


def _subsample_history(records: list[Snapshot], seed: int, ti: int,
                       settings: Settings):
    records = records[-settings.replay_years:]
    if not records or settings.max_replay_pairs == 0:
        return []
    quota = max(1, settings.max_replay_pairs // len(records))
    result=[]
    for old in records:
        rng = np.random.default_rng(seed * 10009 + ti * 347 + old.ti)
        ids = old.fit_idx
        if len(ids) > quota:
            # A reproducible sample: earlier years only, never validation.
            ids = np.sort(rng.choice(ids, size=quota, replace=False))
        result.append((old, ids))
    return result


def _predict(model, snapshot: Snapshot, mode: str, idx: np.ndarray):
    model.eval()
    ids = torch.as_tensor(idx, dtype=torch.long, device=snapshot.pt.x.device)
    with torch.no_grad():
        prediction = []
        for part in ids.split(256):
            prediction.extend(torch.sigmoid(model(snapshot.pt,snapshot.structural,
                            snapshot.prior[mode],ids=part)).cpu().numpy().tolist())
    return np.array(prediction, dtype=np.float64)


def _optimize(model, optimizer, current: Snapshot, replay, mode: str,
              settings: Settings, seed: int):
    """Update after evaluation; early stopping sees only current train-validation."""
    from copy import deepcopy
    best_state=deepcopy(model.state_dict())
    best_val=-float('inf')
    stagnant=0
    total_updates=0
    epoch_log=[]
    base=[(current, current.fit_idx)] + replay
    for epoch in range(settings.epochs):
        model.train()
        rng=np.random.default_rng(seed * 1613 + current.ti * 23 + epoch)
        # Shuffle time contexts; each train candidate is visited once per epoch.
        chunks=[]
        for snap, indices in base:
            shuffled=indices.copy()
            rng.shuffle(shuffled)
            for batch in np.array_split(shuffled, max(1, (len(shuffled) + settings.batch_size - 1)
                                                // settings.batch_size)):
                if len(batch): chunks.append((snap,batch))
        rng.shuffle(chunks)
        losses=[]
        for snap, batch in chunks:
            logits=model(snap.pt, snap.structural, snap.prior[mode], ids=batch)
            target=torch.as_tensor(snap.y[batch],dtype=torch.float32,device=logits.device)
            loss=F.binary_cross_entropy_with_logits(logits,target)
            pos=logits[target > 0.5]
            neg=logits[target < 0.5]
            if len(pos) and len(neg) and settings.pairwise_rank_weight:
                # AP/ranking-aware auxiliary objective, with only fitting labels.
                loss=loss + settings.pairwise_rank_weight * F.softplus(
                    neg.unsqueeze(0) - pos.unsqueeze(1)).mean()
            # Modest residual penalty to prevent residual path from overwhelming
            # the structural-LR anchor with little data.
            if mode not in ('temporal', 'fusion_no_anchor'):
                loss=loss + .0005 * model.residual_log_scale.sigmoid().square()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), settings.gradient_clip)
            optimizer.step()
            losses.append(float(loss.detach()))
            total_updates+=1
        # Same-year controller-validation is in the revealed training portion,
        # and completely disjoint from task.eval_idx.
        valid_scores=_predict(model,current,mode,current.val_idx)
        valid_true=current.y[current.val_idx]
        val=float(average_precision_score(valid_true,valid_scores))
        epoch_log.append({'time':current.year,'epoch':epoch+1,'validation_ap':val,
                          'train_loss':float(np.mean(losses)), 'mode':mode})
        if val > best_val + 1e-6:
            best_val=val
            best_state=deepcopy(model.state_dict())
            stagnant=0
        else:
            stagnant+=1
            if stagnant >= settings.patience:
                break
    model.load_state_dict(best_state)
    return epoch_log,total_updates


def run_seed(stream: TemporalEventStream, seed: int, dest: Path,
             settings: Settings, modes=MODES, device='cpu', e6b_root: Path | None = None):
    start=time.perf_counter()
    _seed(seed)
    snapshots,audit,keys=make_snapshots(stream,seed,settings,tuple(modes),device)
    if e6b_root:
        old_file=e6b_root/f'e6_dblp_seed{seed}'/'candidate_fingerprints.csv'
        if old_file.is_file():
            original=pd.read_csv(old_file, dtype={'time':str})
            if not original.equals(keys):
                raise RuntimeError('Candidate fingerprint mismatch with E6-B')
        else:
            print(f'[E6-C] E6-B candidate fingerprints missing for seed {seed}; '
                  'results comparable by protocol but not pair-verified',flush=True)
    n_train=len(snapshots)-settings.test_last
    summary=[]; all_scores=[]; training=[]; diagnostics=[]
    # The exact E6-B structural Logistic Regression reference is independently
    # scored inside this run, on IDENTICAL candidates and frozen snapshots.
    for i,snap in enumerate(snapshots):
        if i==0:continue
        phase='train' if i<n_train else 'test'
        ids=snap.pt.task.eval_idx
        pred=snap.prior['fusion'][ids].cpu().numpy()
        true=snap.y[ids]
        all_scores.append({'model':'logistic_anchor','seed':seed,'time':snap.year,
            'phase':phase,'ap':float(average_precision_score(true,pred)),
            'roc_auc':float(roc_auc_score(true,pred)),
            'candidate_count':len(ids),'positive_count':int(true.sum()),
            'model_fit_year_through':snapshots[n_train-1].year if phase=='test'
            else snapshots[i-1].year})
    anchors=[r for r in all_scores if r['model']=='logistic_anchor' and r['phase']=='test']
    summary.append({'seed':seed,'model':'logistic_anchor',
                    'test_mean_ap':float(np.mean([r['ap'] for r in anchors])),
                    'test_mean_roc_auc':float(np.mean([r['roc_auc'] for r in anchors])),
                    'test_candidate_total':int(sum(r['candidate_count'] for r in anchors)),
                    'train_updates':0})
    for mode in modes:
        _seed(seed)
        model=FullStructuralTemporalFusion(
            snapshots[0].pt.x.shape[1],hidden=settings.hidden,
            heads=settings.heads,fusion_dim=settings.fusion_dim,
            dropout=settings.dropout,mode=mode).to(device)
        # Independent optimizers for temporal and decoder parameters.
        temporal_parameters=list(model.temporal.parameters())
        temporal_ids={id(p) for p in temporal_parameters}
        fusion_parameters=[p for p in model.parameters() if id(p) not in temporal_ids]
        optimizer=torch.optim.AdamW([
            {'params':temporal_parameters,'lr':settings.lr_tgat},
            {'params':fusion_parameters,'lr':settings.lr_decoder}],
            weight_decay=settings.weight_decay)
        updates=0
        completed=[]
        for i,snap in enumerate(snapshots):
            phase='train' if i<n_train else 'test'
            if i>0:
                ids=snap.pt.task.eval_idx
                pred=_predict(model,snap,mode,ids)
                true=snap.y[ids]
                all_scores.append({'model':mode,'seed':seed,'time':snap.year,
                    'phase':phase,'ap':float(average_precision_score(true,pred)),
                    'roc_auc':float(roc_auc_score(true,pred)),
                    'candidate_count':len(ids),'positive_count':int(true.sum()),
                    'model_fit_year_through':snapshots[n_train-1].year if phase=='test'
                    else snapshots[i-1].year})
            if phase=='train':
                replay=_subsample_history(completed,seed,snap.ti,settings)
                history,steps=_optimize(model,optimizer,snap,replay,mode,settings,seed)
                updates+=steps
                training.extend({'seed':seed,**row} for row in history)
                completed.append(snap)
            else:
                # Train is never invoked in this branch. No gradient or refit
                # occurs in 2015 or 2016. 2015 events enter 2016's history only.
                pass
            if mode in ('fusion','structural','fusion_no_recency'):
                with torch.no_grad():
                    _,parts=model(snap.pt,snap.structural,snap.prior[mode],
                                  ids=snap.pt.task.eval_idx[:min(16,len(snap.pt.task.eval_idx))],
                                  return_parts=True)
                diagnostics.append({'seed':seed,'model':mode,'time':snap.year,
                    'phase':phase,'residual_scale':float(parts.get('scale', model.residual_log_scale.sigmoid()).detach()),
                    'mean_residual':float(parts['correction'].abs().mean().detach()) if 'correction' in parts else float('nan'),
                    'gate_mean':float(parts['gate_mean'].detach()) if 'gate_mean' in parts else None})
            print(f'[E6-C] seed={seed} model={mode} year={snap.year} '
                  f'phase={phase}',flush=True)
        test_rows=[r for r in all_scores if r['model']==mode and r['phase']=='test']
        if len(test_rows) != settings.test_last:
            raise RuntimeError('Missing frozen test evaluation')
        summary.append({'seed':seed,'model':mode,
                        'test_mean_ap':float(np.mean([r['ap'] for r in test_rows])),
                        'test_mean_roc_auc':float(np.mean([r['roc_auc'] for r in test_rows])),
                        'test_candidate_total':int(sum(r['candidate_count'] for r in test_rows)),
                        'train_updates':updates})
        # Archive frozen checkpoint at end of training; do not modify it during test.
        dest.mkdir(parents=True,exist_ok=True)
        torch.save({'model':model.state_dict(),'mode':mode,'seed':seed,
                    'settings':asdict(settings),
                    'fit_year_through':snapshots[n_train-1].year},dest/f'{mode}_frozen.pt')
    pd.DataFrame(all_scores).to_csv(dest/'prequential.csv',index=False)
    pd.DataFrame(training).to_csv(dest/'training_history.csv',index=False)
    pd.DataFrame(diagnostics).to_csv(dest/'fusion_diagnostics.csv',index=False)
    audit.to_csv(dest/'task_audit.csv',index=False)
    keys.to_csv(dest/'candidate_fingerprints.csv',index=False)
    pd.DataFrame(summary).to_csv(dest/'model_summary.csv',index=False)
    (dest/'manifest.json').write_text(json.dumps({'seed':seed,'modes':list(modes),
        'settings':asdict(settings),'train_years':[s.year for s in snapshots[:n_train]],
        'test_years':[s.year for s in snapshots[n_train:]],
        'protocol':'E6-B new-link prediction with frozen 2015–2016 weights; '
                   'prequential LR anchors and train-only replay',
        'time_seconds':time.perf_counter()-start},indent=2)+'\n')
    return summary


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--csv',required=True,type=Path)
    ap.add_argument('--seeds',default='7')
    ap.add_argument('--results-dir',type=Path,default=Path('results'))
    ap.add_argument('--expected-sha',default=DEFAULT_E4_SHA)
    ap.add_argument('--e6b-results-dir',type=Path,default=None)
    ap.add_argument('--device',default='cpu',choices=['cpu','cuda'])
    ap.add_argument('--modes',default=','.join(MODES))
    ap.add_argument('--epochs',type=int,default=Settings.epochs)
    ap.add_argument('--patience',type=int,default=Settings.patience)
    ap.add_argument('--max-replay-pairs',type=int,default=Settings.max_replay_pairs)
    args=ap.parse_args(argv)
    csv=args.csv.resolve()
    if not csv.is_file(): ap.error(f'CSV missing: {csv}')
    fingerprint=sha(csv)
    if fingerprint!=args.expected_sha:
        ap.error(f'CSV SHA256 mismatch: {fingerprint}; expected {args.expected_sha}')
    try: seeds=[int(s.strip()) for s in args.seeds.split(',')]
    except ValueError: ap.error('Invalid comma-separated seeds')
    if not seeds or len(set(seeds))!=len(seeds) or any(s<0 for s in seeds):
        ap.error('Seeds must be distinct non-negative integers')
    modes=tuple(args.modes.split(','))
    if not modes or len(set(modes))!=len(modes) or any(m not in MODES for m in modes):
        ap.error('Unknown or repeated modes')
    if args.epochs<1 or args.patience<1 or args.max_replay_pairs<0:
        ap.error('Invalid settings')
    settings=Settings(epochs=args.epochs,patience=args.patience,
                      max_replay_pairs=args.max_replay_pairs)
    if args.device=='cuda' and not torch.cuda.is_available():
        ap.error('CUDA requested but not available')
    torch.set_num_threads(min(torch.get_num_threads(),2))
    root=Path(__file__).resolve().parents[1]
    out=(args.results_dir if args.results_dir.is_absolute()
         else root/args.results_dir)
    out.mkdir(parents=True,exist_ok=True)
    files=['src/adaptive_graph/e6_fusion.py','src/adaptive_graph/e6_structural.py',
           'src/adaptive_graph/tgat.py','src/adaptive_graph/data.py',
           'scripts/e6_run_suite.py','scripts/e6c_run_suite.py']
    protocol={'phase':'E6-C','csv_sha256':fingerprint,
              'code_sha256':{f:sha(root/f) for f in files},
              'settings':asdict(settings),'modes':list(modes)}
    lock=out/'e6c_dblp_protocol.json'
    if lock.exists():
        old=json.loads(lock.read_text())
        if old!=protocol:
            raise RuntimeError('E6-C run settings or code changed; use a new results directory')
    else:
        lock.write_text(json.dumps(protocol,indent=2)+'\n')
    stream=TemporalEventStream.from_csv(csv,bucket='year',undirected=True)
    if [str(t) for t in stream.network.times[-2:]]!=['2015','2016']:
        ap.error('Expected final test years 2015 and 2016')
    print(f'E6-C DBLP CSV SHA {fingerprint}; nodes={stream.network.n} '
          f'events={len(stream.network.events)}',flush=True)
    for seed in seeds:
        folder=out/f'e6c_dblp_seed{seed}'
        if folder.exists():
            summary=folder/'model_summary.csv'
            keys=folder/'candidate_fingerprints.csv'
            if not summary.is_file() or not keys.is_file():
                raise RuntimeError(f'Incomplete folder: {folder}')
            observed=set(pd.read_csv(summary)['model'])
            if observed!=set(modes):
                raise RuntimeError(f'Incompatible modes in folder {folder}')
            print('REUSED:',folder.name,flush=True)
            continue
        try:
            run_seed(stream,seed,folder,settings,modes,args.device,args.e6b_results_dir)
        except Exception:
            # A failed run may have written partial checkpoints; refuse silent reuse.
            raise
    print('E6-C complete. Aggregate with scripts/e6c_aggregate.py',flush=True)

if __name__=='__main__':
    main()
