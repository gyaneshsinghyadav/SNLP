#!/usr/bin/env python3
"""E6-A/B: verify E4-compatible input; evaluate history-only structural models.

Uses the E4 task generator and sample split, including 210 revealed training pairs
and 70 controller-validation pairs per training year for the DBLP data. Never fits
on eval_idx or sealed years; all models and hyperparameters are predeclared.
This is the E4 TGAT-subgraph benchmark, NOT the C3.4 heldout-author protocol.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from adaptive_graph.data import TemporalNetwork, create_task
from adaptive_graph.temporal_control import split_controller_validation
from adaptive_graph.e6_structural import HistoricalPairFeatures, FEATURE_NAMES

DEFAULT_E4_SHA = '3cade711abbe2452f1bf22b1ed938ed46640d58651114676e85aea4198578744'
SEEDS = (7, 13, 42, 101, 202)
SCORERS = ('adamic_adar', 'logistic', 'boosted')
CONFIG = {'time_bucket': 'year', 'undirected': True, 'start_time_index': 3,
          'test_last': 2, 'max_positives': 200, 'negative_ratio': 1.,
          'eval_fraction': .3, 'controller_val_fraction': .25,
          'decay': .7, 'logistic_c': 1., 'boosted_max_iter': 80,
          'boosted_max_leaf_nodes': 7, 'boosted_min_samples_leaf': 15,
          'random_seed_offset': 0}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def digest_candidates(task):
    h=hashlib.sha256()
    for array in (task.candidates, task.labels, task.train_idx, task.eval_idx):
        z=np.ascontiguousarray(array,dtype='<i8')
        h.update(z.shape[0].to_bytes(8, 'little'))
        h.update(z.tobytes())
    return h.hexdigest()


def model_for(name, seed):
    if name == 'logistic':
        return make_pipeline(StandardScaler(),LogisticRegression(
            C=CONFIG['logistic_c'],max_iter=350,solver='lbfgs',random_state=seed))
    if name == 'boosted':
        return HistGradientBoostingClassifier(max_iter=CONFIG['boosted_max_iter'],
            max_leaf_nodes=CONFIG['boosted_max_leaf_nodes'], learning_rate=.05,
            min_samples_leaf=CONFIG['boosted_min_samples_leaf'], l2_regularization=1.,
            random_state=seed, early_stopping=False)
    raise ValueError(name)


def score_model(model, x):
    return model.predict_proba(x)[:, 1]


def run_seed(net, seed, dest):
    """Prequential evaluation: t scores before any revealed train labels from t."""
    max_time=len(net.times); train_end=max_time-CONFIG['test_last']
    if train_end <= CONFIG['start_time_index']+2:
        raise ValueError('Insufficient training years')
    train_x=[]; train_y=[]; models={}; scores=[]; audits=[]; keys=[]
    for ti in range(CONFIG['start_time_index'],max_time):
        phase='train' if ti<train_end else 'test'
        task=create_task(net,ti,seed=seed,max_positives=CONFIG['max_positives'],
             negative_ratio=CONFIG['negative_ratio'],eval_fraction=CONFIG['eval_fraction'])
        partition=(split_controller_validation(task,CONFIG['controller_val_fraction'],seed)
                   if phase=='train' else None)
        if phase=='train' and partition is None:
            raise RuntimeError(f'Insufficient per-class samples at {task.time_value}')
        fit_idx=(partition[0] if partition is not None else np.empty(0,dtype=np.int64))
        validate_idx=(partition[1] if partition is not None else np.empty(0,dtype=np.int64))
        if (set(fit_idx)&set(task.eval_idx) or
            set(validate_idx)&set(task.eval_idx) or set(fit_idx)&set(validate_idx)):
            raise RuntimeError('Candidate index overlap')
        hist=HistoricalPairFeatures(net,ti,decay=CONFIG['decay'])
        x=hist.transform(task.candidates)
        y=task.labels
        # Detect any candidate edges that already existed in history.
        # New-link targets must explicitly exclude repeat collaborations.
        seen={tuple(row) for row in task.historical_edges.tolist()}
        overlap=sum(tuple(sorted(map(int,row))) in seen for row in task.candidates)
        if overlap:
            raise RuntimeError(f'Historical candidate overlap in year {task.time_value}: {overlap}')
        audits.append({'seed':seed,'time':str(task.time_value),'phase':phase,
            'history_edges':len(task.historical_edges), 'history_events':hist.historical_event_count,
            'new_links':task.new_link_count,'repeat_contacts':task.repeat_contact_count,
            'cold_start':task.cold_start_count,'train_candidates':len(fit_idx),
            'controller_validation_candidates':len(validate_idx),
            'sealed_test_candidates':len(task.eval_idx),'label_overlap_count':0,
            'historical_candidate_overlap':overlap})
        keys.append({'seed':seed,'time':str(task.time_value),
                     'candidate_task_sha256':digest_candidates(task)})
        # E4 evaluates training snapshots after warmup=1; final two test
        # years are scored with learner weights frozen at end of 2014.
        if train_x and len(task.eval_idx):
            eval_x=x[task.eval_idx]
            eval_y=y[task.eval_idx]
            for name in SCORERS:
                if name=='adamic_adar':
                    prediction=eval_x[:, FEATURE_NAMES.index('adamic_adar')]
                else:
                    prediction=score_model(models[name],eval_x)
                scores.append({'seed':seed,'time':str(task.time_value),'phase':phase,
                    'model':name,'ap':float(average_precision_score(eval_y,prediction)),
                    'roc_auc':float(roc_auc_score(eval_y,prediction)),
                    'candidate_count':len(eval_y),
                    'positive_count':int(eval_y.sum()),'model_fit_year_through':
                        str(net.times[train_end-1]) if phase=='test' else str(net.times[ti-1])})
        if phase=='train':
            # Never use task.eval_idx or controller-validation indices for training.
            train_x.append(x[fit_idx]); train_y.append(y[fit_idx])
            xx=np.concatenate(train_x); yy=np.concatenate(train_y)
            if len(np.unique(yy)) != 2:
                raise ValueError('Need both positive and negative training examples')
            for name in SCORERS:
                if name!='adamic_adar':
                    model=model_for(name,seed)
                    model.fit(xx,yy)
                    models[name]=model
        print(f'[E6] seed={seed} year={task.time_value} phase={phase} '
              f'candidates={len(task.eval_idx)} history={hist.historical_event_count}',flush=True)
    frame=pd.DataFrame(scores)
    audit=pd.DataFrame(audits); keyframe=pd.DataFrame(keys)
    if len(frame.query("phase == 'test'")) != len(SCORERS)*CONFIG['test_last']:
        raise RuntimeError('Missing test observations')
    dest.mkdir(parents=True,exist_ok=False)
    frame.to_csv(dest/'prequential.csv',index=False)
    audit.to_csv(dest/'task_audit.csv',index=False)
    keyframe.to_csv(dest/'candidate_fingerprints.csv',index=False)
    test=frame[frame.phase=='test']
    summary=[]
    for name,g in test.groupby('model'):
        summary.append({'model':name,'seed':seed,'test_mean_ap':float(g.ap.mean()),
                        'test_mean_roc_auc':float(g.roc_auc.mean()),
                        'test_candidate_total':int(g.candidate_count.sum()),
                        'train_pairs_total':int(audit.query("phase == 'train'").train_candidates.sum())})
    pd.DataFrame(summary).to_csv(dest/'model_summary.csv',index=False)
    (dest/'manifest.json').write_text(json.dumps({'seed':seed,'features':list(FEATURE_NAMES),
      'config':CONFIG, 'times':[str(t) for t in net.times],
      'time_split':{'train_end_exclusive':str(net.times[train_end]),
                    'test_times':[str(t) for t in net.times[train_end:]]},
      'note':'E4-style new-link candidates; 2015 edges are observable history for 2016, '
             'but models stay frozen in both years'},indent=2)+'\n')
    return summary


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--csv',type=Path,required=True)
    p.add_argument('--seeds',default='7')
    p.add_argument('--results-dir',type=Path,default=Path('results'))
    p.add_argument('--expected-sha',default=DEFAULT_E4_SHA,
      help='Expected SHA256 of the original E4 DBLP subgraph. Set explicitly only for synthetic tests.')
    args=p.parse_args(argv)
    csv=args.csv.resolve()
    if not csv.is_file(): p.error(f'Missing dataset: {csv}')
    actual=sha(csv)
    if args.expected_sha != actual:
        p.error(f'CSV SHA256 mismatch: actual {actual}, E4 expected {args.expected_sha}. '
                'Do not compare against E4 without the identical input bytes.')
    try: seeds=[int(s.strip()) for s in args.seeds.split(',')]
    except ValueError: p.error('--seeds needs comma-separated integers')
    if not seeds or min(seeds)<0 or len(set(seeds))!=len(seeds):
        p.error('Seeds must be distinct non-negative integers')
    root=Path(__file__).resolve().parents[1]
    dest=args.results_dir if args.results_dir.is_absolute() else root/args.results_dir
    dest.mkdir(parents=True,exist_ok=True)
    # The actual task construction code must be pinned for future comparisons.
    code_files=['src/adaptive_graph/data.py','src/adaptive_graph/temporal_control.py',
        'src/adaptive_graph/e6_structural.py','scripts/e6_run_suite.py']
    code_sha={x:sha(root/x) for x in code_files}
    protocol={'phase':'E6-B','baseline':'E4 DBLP new-link protocol',
       'csv_sha256':actual,'code_sha256':code_sha,'config':CONFIG,'seeds':sorted(seeds)}
    lock=dest/'e6_dblp_protocol.json'
    if lock.exists():
        prior=json.loads(lock.read_text())
        for key in ('csv_sha256','code_sha256','config','baseline'):
            if protocol[key]!=prior[key]:
                raise RuntimeError(f'E6 protocol changed: {key}. Use another results directory.')
        protocol['seeds']=sorted(set(protocol['seeds'])|set(prior['seeds']))
    lock.write_text(json.dumps(protocol,indent=2)+'\n')
    net=TemporalNetwork.from_csv(csv,time_bucket='year',directed=False)
    if str(net.times[-2:]) != '[2015, 2016]':
        p.error('Expected 2015 and 2016 as final two test years')
    for seed in seeds:
        folder=dest/f'e6_dblp_seed{seed}'
        if folder.exists():
            summary=folder/'model_summary.csv'; keys=folder/'candidate_fingerprints.csv'
            if not summary.exists() or not keys.exists():
                raise RuntimeError(f'Incomplete results folder {folder}, inspect manually')
            expected=set(SCORERS); observed=set(pd.read_csv(summary).model)
            if expected!=observed or len(pd.read_csv(keys))!=len(net.times)-3:
                raise RuntimeError(f'Incomplete results folder {folder}')
            print('REUSED:',folder.name,flush=True)
            continue
        run_seed(net,seed,folder)
    print('E6-B complete; run scripts/e6_aggregate.py for comparison.',flush=True)

if __name__=='__main__': main()
