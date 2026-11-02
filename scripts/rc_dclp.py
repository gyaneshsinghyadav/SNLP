#!/usr/bin/env python3
"""Supported interface for the finalized RC-DCLP (E4-on-E3) research model.

The unchanged, hash-locked experimental implementation lives in
``scripts/e6g_run_suite.py``. This facade pins its E4-on-E3 arm.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
from e6_run_suite import DEFAULT_E4_SHA, sha
from e6g_run_suite import main as original_train, E6GSettings
from e6c_run_suite import Settings, make_snapshots
from e6d_run_suite import E6DSettings, _packets, _score
from adaptive_graph.tgat import TemporalEventStream
from adaptive_graph.e6_higher_order import HigherOrderStructuralTemporalFusion

SEEDS = (7,13,42,101,202)


def archive_dir(seed: int) -> Path:
    return ROOT/'recorded'/'e6g_checkpoint'/'e4_retention'/f'e6g_dblp_seed{seed}'


def checkpoint_dir(seed: int, root: Path|None) -> Path:
    return (root/'e4_retention'/f'e6g_dblp_seed{seed}') if root is not None else archive_dir(seed)


def load_checkpoint(path: Path, device: str='cpu') -> dict:
    if not path.is_file(): raise FileNotFoundError(f'Checkpoint missing: {path}')
    ck = torch.load(path, map_location=device, weights_only=True)
    if ck.get('backbone')!='structural_subgraph' or ck.get('policy')!='dqn':
        raise ValueError(f'Unexpected checkpoint backbone/policy: {path}')
    s=ck.get('settings',{})
    if s.get('variant')!='e3' or s.get('checkpoint_mode')!='retention':
        raise ValueError('Not the accepted E4-on-E3 model; E5/E6b/other arms are excluded')
    if ck.get('fit_year_through')!= '2014' and str(ck.get('fit_year_through'))!='2014':
        raise ValueError('Expected frozen 2014-trained predictor')
    return ck


def verify_saved() -> None:
    # The original completed DBLP experiment locked the exact training source
    # files. This confirms the release runs the same implementation.
    protocol=json.loads((ROOT/'recorded'/'e6g_checkpoint'/'e4_retention'/'e6g_protocol.json').read_text())
    if protocol.get('dataset_sha256') != DEFAULT_E4_SHA:
        raise RuntimeError('Recorded protocol has a different dataset fingerprint')
    for rel, expected in protocol['code_sha256'].items():
        actual=sha(ROOT/rel)
        if actual != expected:
            raise RuntimeError(f'Core code differs from archived successful run: {rel}')
    print(f"PASS: all {len(protocol['code_sha256'])} original experiment code hashes match exactly")
    summary=[]
    for seed in SEEDS:
        folder=archive_dir(seed)
        model=load_checkpoint(folder/'structural_subgraph_dqn_frozen.pt')
        controller=torch.load(folder/'structural_subgraph_dqn_controller.pt',map_location='cpu',weights_only=True)
        assert controller['state_size']==25 and len(controller['actions'])==6
        assert controller['fit_year_through']== '2014' or str(controller['fit_year_through'])=='2014'
        assert all(torch.isfinite(v).all() for v in model['model'].values())
        assert len(model['model'])>20
        meta=json.loads((folder/'manifest.json').read_text())
        assert meta['settings']['variant']=='e3' and meta['settings']['checkpoint_mode']=='retention'
        assert meta['seed']==seed
        cand=pd.read_csv(folder/'candidate_fingerprints.csv', dtype={'time':str})
        assert len(cand)==14 and cand.candidate_task_sha256.nunique()==14
        scores=pd.read_csv(folder/'model_summary.csv')
        assert len(scores)==1 and int(scores.seed.iloc[0])==seed
        audit=pd.read_csv(folder/'task_audit.csv')
        assert audit.label_overlap_count.eq(0).all()
        summary.append(scores.iloc[0])
    df=pd.DataFrame(summary)
    print('PASS: five E4-on-E3 GNN checkpoints and five DQN checkpoints load with finite weights')
    print('PASS: five manifests, candidate hashes and leakage audit CSVs are consistent')
    print(f"Recorded test AP={df.test_mean_ap.mean()*100:.3f}%, ROC-AUC={df.test_mean_roc_auc.mean()*100:.3f}%")
    print(f"Recorded historical-validation mean forgetting={df.mean_forgetting.mean()*100:.3f} pp")
    print('Note: this verifies ARCHIVED outputs, not a new DBLP training run.')


def do_train(args):
    argv=['--csv',str(args.csv),'--seeds',args.seeds,'--arms','e4_retention',
          '--results-dir',str(args.results_dir),'--device',args.device]
    if args.e6d_reference is not None:argv.extend(['--e6d-reference',str(args.e6d_reference)])
    original_train(argv)


def do_predict(args):
    csv=Path(args.csv).expanduser().resolve()
    if not csv.is_file(): raise FileNotFoundError(csv)
    digest=sha(csv)
    if digest != DEFAULT_E4_SHA:
        raise RuntimeError('This saved DBLP checkpoint requires the ORIGINAL CSV. '
                           f'Expected SHA256 {DEFAULT_E4_SHA}, got {digest}. '
                           'Retrain on any different dataset.')
    if args.device=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable')
    seed=args.seed
    folder=checkpoint_dir(seed,args.results_dir)
    ck=load_checkpoint(folder/'structural_subgraph_dqn_frozen.pt',device=args.device)
    base=Settings(**ck['settings']['base'])
    torch.set_num_threads(min(torch.get_num_threads(),2))
    stream=TemporalEventStream.from_csv(csv,bucket='year',undirected=True)
    snaps,audit,keys=make_snapshots(stream,seed,base,('fusion',),args.device)
    expected=pd.read_csv(folder/'candidate_fingerprints.csv', dtype={'time':str})
    pd.testing.assert_frame_equal(keys,expected,check_exact=True)
    years=[int(x) for x in args.years.split(',')]
    if not years or not set(years)<=set((2015,2016)):raise ValueError('Only original frozen 2015–2016 evaluation is supported')
    model=HigherOrderStructuralTemporalFusion(snaps[0].pt.x.shape[1],hidden=base.hidden,
        heads=base.heads,fusion_dim=base.fusion_dim,dropout=base.dropout,
        mode='structural_subgraph').to(args.device)
    model.load_state_dict(ck['model'], strict=True)
    model.eval()
    packets=_packets(stream,snaps,E6DSettings(base=base,max_nodes=ck['settings']['max_nodes'],
        first_hop_limit=ck['settings']['first_hop_limit']),args.device)
    out=[]
    for snap,packet in zip(snaps,packets):
        if int(snap.year) not in years:continue
        ids=snap.pt.task.eval_idx
        pred=_score(model,snap,packet,ids)
        pair=snap.pt.task.candidates[ids]
        for (u,v),score in zip(pair,pred):
            out.append({'seed':seed,'prediction_year':snap.year,'source_internal_id':int(u),
                        'target_internal_id':int(v),'probability':float(score)})
        if args.evaluate:
            label=snap.y[ids]
            print(f'{snap.year}: AP={average_precision_score(label,pred):.6f} '
                  f'AUC={roc_auc_score(label,pred):.6f} n={len(ids)}')
    dest=Path(args.output);dest.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(out).to_csv(dest,index=False)
    print(f'Saved {len(out)} reproducible frozen-candidate scores to {dest}')
    print('Node IDs refer to the factorization of the EXACT original dataset CSV.')


def do_compare(args):
    from e6g_compare import compare
    root=args.results_dir or ROOT/'recorded'/'e6g_checkpoint'
    root=root if root.is_absolute() else ROOT/root
    reference=ROOT/'docs'/'e6g_e3_reference'
    compare(root,args.output,reference_root=reference,require_five=True)
    print(f'Compared against archived matched unconstrained DQN–RRD control: {args.output}')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    sub.add_parser('verify',help='Inspect all actual saved GNN/DQN weights and recorded metrics')
    train=sub.add_parser('train',help='Reproduce the final E4-on-E3 five-seed experiment')
    train.add_argument('--csv',required=True,type=Path)
    train.add_argument('--seeds',default='7,13,42,101,202')
    train.add_argument('--results-dir',type=Path,default=Path('runs/rc_dclp'))
    train.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    train.add_argument('--e6d-reference',default=None)
    pred=sub.add_parser('predict',help='Score frozen DBLP evaluation candidates with saved model weights')
    pred.add_argument('--csv',required=True,type=Path)
    pred.add_argument('--seed',type=int,choices=SEEDS,default=7)
    pred.add_argument('--years',default='2015,2016')
    pred.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    pred.add_argument('--results-dir',type=Path,default=None,
                      help='Optional newly trained experiment root, otherwise bundled checkpoints')
    pred.add_argument('--evaluate',action='store_true',help='Report metrics using labels (post-hoc only)')
    pred.add_argument('--output',type=Path,default=Path('outputs/frozen_predictions.csv'))
    comp=sub.add_parser('compare',help='Compare the recorded E4 output with matched E3 reference')
    comp.add_argument('--results-dir',type=Path,default=None)
    comp.add_argument('--output',type=Path,default=Path('outputs/paired_comparison'))
    args=p.parse_args()
    if args.command=='verify':verify_saved()
    elif args.command=='train':do_train(args)
    elif args.command=='predict':do_predict(args)
    elif args.command=='compare':do_compare(args)

if __name__=='__main__':main()
