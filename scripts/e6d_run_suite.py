#!/usr/bin/env python3
"""E6-D: five-seed history-only higher-order structural-temporal link prediction.

Respects the exact E6-C candidate protocol and holds test-period weights fixed.
The named E6-C reference is loaded only after candidate fingerprint validation.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
from dataclasses import asdict,dataclass
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score,roc_auc_score

from adaptive_graph.tgat import TemporalEventStream
from adaptive_graph.e6_structural import HistoricalPairFeatures
from adaptive_graph.e6_subgraph import HistoricalEnclosingSubgraphs
from adaptive_graph.e6_higher_order import HigherOrderStructuralTemporalFusion,MODES
from e6c_run_suite import Settings,make_snapshots,_subsample_history,_seed
from e6_run_suite import DEFAULT_E4_SHA,sha


@dataclass(frozen=True)
class E6DSettings:
    # All inherited task splitting/temporal training settings are E6-C defaults.
    base: Settings = Settings()
    max_nodes: int = 32
    first_hop_limit: int = 20


def _packets(stream,snapshots,settings:E6DSettings,device):
    packets=[]
    for snap in snapshots:
        hist=HistoricalPairFeatures(stream.network,snap.ti,decay=settings.base.decay)
        pack=HistoricalEnclosingSubgraphs(hist,max_nodes=settings.max_nodes,
                         first_hop_limit=settings.first_hop_limit).transform(snap.pt.task.candidates)
        packets.append(pack.on(device))
    return packets


def _score(model,snap,packet,idx):
    model.eval()
    predictions=[]
    for part in torch.as_tensor(idx,dtype=torch.long,device=snap.pt.x.device).split(128):
        with torch.no_grad():
            logits=model(snap.pt,snap.structural,snap.prior['fusion'],packet,ids=part)
            predictions.extend(torch.sigmoid(logits).cpu().numpy().tolist())
    return np.asarray(predictions,dtype=np.float64)


def _train(model,opt,current,packet,replay,packets_by_ti,settings,seed):
    base=settings.base
    best=deepcopy(model.state_dict());best_ap=-np.inf;stale=0;history=[];steps=0
    candidates=[(current,current.fit_idx,packet)] + [
        (past,idx,packets_by_ti[past.ti]) for past,idx in replay]
    for epoch in range(base.epochs):
        model.train()
        rng=np.random.default_rng(seed*1613+current.ti*23+epoch)
        chunks=[]
        for snap,ids,pkt in candidates:
            shuffled=ids.copy();rng.shuffle(shuffled)
            for chunk in np.array_split(shuffled,max(1,(len(shuffled)+base.batch_size-1)//base.batch_size)):
                if len(chunk):chunks.append((snap,chunk,pkt))
        rng.shuffle(chunks)
        losses=[]
        for snap,ids,pkt in chunks:
            logits=model(snap.pt,snap.structural,snap.prior['fusion'],pkt,ids=ids)
            target=torch.as_tensor(snap.y[ids],dtype=torch.float32,device=logits.device)
            loss=F.binary_cross_entropy_with_logits(logits,target)
            pos=logits[target>0.5];neg=logits[target<0.5]
            if len(pos) and len(neg) and base.pairwise_rank_weight:
                loss=loss+base.pairwise_rank_weight*F.softplus(
                    neg.unsqueeze(0)-pos.unsqueeze(1)).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),base.gradient_clip)
            opt.step();steps+=1;losses.append(float(loss.detach()))
        pred=_score(model,current,packet,current.val_idx)
        ap=float(average_precision_score(current.y[current.val_idx],pred))
        history.append({'time':current.year,'epoch':epoch+1,'validation_ap':ap,
                        'train_loss':float(np.mean(losses))})
        if ap>best_ap+1e-6:
            best_ap=ap;best=deepcopy(model.state_dict());stale=0
        else:
            stale+=1
            if stale>=base.patience:break
    model.load_state_dict(best)
    return history,steps


def run_seed(stream,seed:int,dest:Path,settings:E6DSettings,
             modes=MODES,device='cpu',e6c_results_dir:Path|None=None):
    start=time.perf_counter();_seed(seed)
    snapshots,audit,keys=make_snapshots(stream,seed,settings.base,('fusion',),device)
    if e6c_results_dir is not None:
        old=e6c_results_dir/f'e6c_dblp_seed{seed}'
        candidate=old/'candidate_fingerprints.csv'
        baseline=old/'model_summary.csv'
        if not candidate.is_file() or not baseline.is_file():
            raise RuntimeError(f'Missing E6-C reference data for seed {seed}: {old}')
        oldkeys=pd.read_csv(candidate,dtype={'time':str})
        if not keys.equals(oldkeys):
            raise RuntimeError(f'E6-C candidate fingerprint mismatch at seed {seed}')
        oldsummary=pd.read_csv(baseline)
        if not {'fusion','logistic_anchor'}.issubset(set(oldsummary.model)):
            raise RuntimeError('Missing E6-C fusion and anchor reference')
    else:
        oldsummary=None
    packets=_packets(stream,snapshots,settings,device)
    packets_by_ti={snap.ti:pkt for snap,pkt in zip(snapshots,packets)}
    n_train=len(snapshots)-settings.base.test_last
    scores=[];training=[];diagnostics=[];summary=[]
    for i,snap in enumerate(snapshots):
        if i==0:continue
        phase='train' if i<n_train else 'test'
        ids=snap.pt.task.eval_idx
        prior=snap.prior['fusion'][ids].cpu().numpy()
        y=snap.y[ids]
        scores.append({'seed':seed,'model':'logistic_anchor','time':snap.year,
                       'phase':phase,'ap':float(average_precision_score(y,prior)),
                       'roc_auc':float(roc_auc_score(y,prior)),
                       'candidate_count':len(ids),'positive_count':int(y.sum()),
                       'model_fit_year_through':snapshots[n_train-1].year if phase=='test' else snapshots[i-1].year})
    anchor=[r for r in scores if r['model']=='logistic_anchor' and r['phase']=='test']
    summary.append({'seed':seed,'model':'logistic_anchor',
                    'test_mean_ap':float(np.mean([r['ap'] for r in anchor])),
                    'test_mean_roc_auc':float(np.mean([r['roc_auc'] for r in anchor])),
                    'test_candidate_total':int(sum(r['candidate_count'] for r in anchor)),
                    'train_updates':0})
    if oldsummary is not None:
        assert abs(float(oldsummary.loc[oldsummary.model=='logistic_anchor','test_mean_ap'].iloc[0]) - summary[0]['test_mean_ap'])<1e-6, 'E6-C anchor mismatch'
    for mode in modes:
        _seed(seed)
        model=HigherOrderStructuralTemporalFusion(snapshots[0].pt.x.shape[1],
                   hidden=settings.base.hidden,heads=settings.base.heads,
                   fusion_dim=settings.base.fusion_dim,dropout=settings.base.dropout,mode=mode).to(device)
        parameters=list(model.base.temporal.parameters())
        temporal_ids={id(p) for p in parameters}
        others=[p for p in model.parameters() if id(p) not in temporal_ids]
        opt=torch.optim.AdamW([{'params':parameters,'lr':settings.base.lr_tgat},
                              {'params':others,'lr':settings.base.lr_decoder}],
                             weight_decay=settings.base.weight_decay)
        completed=[];updates=0
        for i,(snap,packet) in enumerate(zip(snapshots,packets)):
            phase='train' if i<n_train else 'test'
            if i>0:
                ids=snap.pt.task.eval_idx
                pred=_score(model,snap,packet,ids);y=snap.y[ids]
                scores.append({'seed':seed,'model':mode,'time':snap.year,'phase':phase,
                               'ap':float(average_precision_score(y,pred)),
                               'roc_auc':float(roc_auc_score(y,pred)),
                               'candidate_count':len(ids),'positive_count':int(y.sum()),
                               'model_fit_year_through':snapshots[n_train-1].year if phase=='test' else snapshots[i-1].year})
            if phase=='train':
                replay=_subsample_history(completed,seed,snap.ti,settings.base)
                log,nsteps=_train(model,opt,snap,packet,replay,packets_by_ti,settings,seed)
                updates+=nsteps
                training.extend({'seed':seed,'model':mode,**r} for r in log)
                completed.append(snap)
            with torch.no_grad():
                model.eval()
                _,parts=model(snap.pt,snap.structural,snap.prior['fusion'],packet,
                              ids=snap.pt.task.eval_idx[:min(16,len(snap.pt.task.eval_idx))],
                              return_parts=True)
            diagnostics.append({'seed':seed,'model':mode,'time':snap.year,'phase':phase,
                                **{k:float(v.detach().mean()) for k,v in parts.items()}})
            print(f'[E6-D] seed={seed} model={mode} year={snap.year} {phase}',flush=True)
        test=[r for r in scores if r['model']==mode and r['phase']=='test']
        if len(test)!=settings.base.test_last:
            raise RuntimeError('Missing frozen test rows')
        summary.append({'seed':seed,'model':mode,
                        'test_mean_ap':float(np.mean([r['ap'] for r in test])),
                        'test_mean_roc_auc':float(np.mean([r['roc_auc'] for r in test])),
                        'test_candidate_total':int(sum(r['candidate_count'] for r in test)),
                        'train_updates':updates})
        dest.mkdir(parents=True,exist_ok=True)
        torch.save({'model':model.state_dict(),'seed':seed,'mode':mode,
                    'settings':asdict(settings),
                    'fit_year_through':snapshots[n_train-1].year},dest/f'{mode}_frozen.pt')
    dest.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(scores).to_csv(dest/'prequential.csv',index=False)
    pd.DataFrame(summary).to_csv(dest/'model_summary.csv',index=False)
    pd.DataFrame(training).to_csv(dest/'training_history.csv',index=False)
    pd.DataFrame(diagnostics).to_csv(dest/'subgraph_diagnostics.csv',index=False)
    audit.to_csv(dest/'task_audit.csv',index=False)
    keys.to_csv(dest/'candidate_fingerprints.csv',index=False)
    manifest={'phase':'E6-D','seed':seed,'modes':list(modes),
              'settings':asdict(settings),'train_years':[s.year for s in snapshots[:n_train]],
              'test_years':[s.year for s in snapshots[n_train:]],
              'matched_e6c_reference':e6c_results_dir is not None,
              'runtime_sec':time.perf_counter()-start}
    (dest/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return summary


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--csv',type=Path,required=True)
    ap.add_argument('--expected-sha',default=DEFAULT_E4_SHA)
    ap.add_argument('--seeds',default='7')
    ap.add_argument('--results-dir',type=Path,default=Path('results'))
    ap.add_argument('--e6c-results-dir',type=Path,default=None)
    ap.add_argument('--modes',default=','.join(MODES))
    ap.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    ap.add_argument('--epochs',type=int,default=Settings.epochs)
    ap.add_argument('--patience',type=int,default=Settings.patience)
    ap.add_argument('--max-replay-pairs',type=int,default=Settings.max_replay_pairs)
    ap.add_argument('--max-subgraph-nodes',type=int,default=32)
    ap.add_argument('--first-hop-limit',type=int,default=20)
    args=ap.parse_args(argv)
    csv=args.csv.resolve()
    if not csv.is_file():ap.error(f'Missing CSV: {csv}')
    digest=sha(csv)
    if digest!=args.expected_sha:ap.error(f'CSV SHA256 mismatch: {digest}; expected {args.expected_sha}')
    try:seeds=[int(x.strip()) for x in args.seeds.split(',')]
    except ValueError:ap.error('Invalid seed specification')
    if not seeds or len(set(seeds))!=len(seeds) or any(s<0 for s in seeds):ap.error('Seeds must be unique nonnegative integers')
    modes=tuple(args.modes.split(','))
    if not modes or len(set(modes))!=len(modes) or not set(modes).issubset(MODES):ap.error('Invalid model modes')
    if args.epochs<1 or args.patience<1 or args.max_replay_pairs<0 or args.max_subgraph_nodes<4 or args.first_hop_limit<2:
        ap.error('Invalid optimization or subgraph settings')
    if args.device=='cuda' and not torch.cuda.is_available():ap.error('CUDA is unavailable')
    torch.set_num_threads(min(torch.get_num_threads(),2))
    settings=E6DSettings(base=Settings(epochs=args.epochs,patience=args.patience,
                                  max_replay_pairs=args.max_replay_pairs),
                         max_nodes=args.max_subgraph_nodes,
                         first_hop_limit=args.first_hop_limit)
    root=Path(__file__).resolve().parents[1]
    out=args.results_dir if args.results_dir.is_absolute() else root/args.results_dir
    out.mkdir(parents=True,exist_ok=True)
    files=['src/adaptive_graph/e6_subgraph.py','src/adaptive_graph/e6_higher_order.py',
           'src/adaptive_graph/e6_fusion.py','src/adaptive_graph/e6_structural.py',
           'src/adaptive_graph/tgat.py','src/adaptive_graph/data.py',
           'scripts/e6c_run_suite.py','scripts/e6d_run_suite.py']
    protocol={'phase':'E6-D','csv_sha256':digest,
              'code_sha256':{f:sha(root/f) for f in files},
              'settings':asdict(settings),'modes':list(modes)}
    lock=out/'e6d_dblp_protocol.json'
    if lock.exists():
        previous=json.loads(lock.read_text())
        if previous!=protocol:
            raise RuntimeError('Existing E6-D results use different code/settings. Use a new --results-dir')
    else:
        lock.write_text(json.dumps(protocol,indent=2)+'\n')
    stream=TemporalEventStream.from_csv(csv,bucket='year',undirected=True)
    if list(map(str,stream.network.times[-2:]))!=['2015','2016']:
        ap.error('Expected final test years 2015 and 2016')
    ref=args.e6c_results_dir
    if ref is not None:
        ref=ref if ref.is_absolute() else root/ref
    print(f'E6-D DBLP sha={digest} nodes={stream.network.n} events={len(stream.network.events)}',flush=True)
    for seed in seeds:
        folder=out/f'e6d_dblp_seed{seed}'
        if folder.exists():
            summary=folder/'model_summary.csv';candidates=folder/'candidate_fingerprints.csv'
            if not summary.is_file() or not candidates.is_file():
                raise RuntimeError(f'Incomplete E6-D results folder: {folder}')
            observed=set(pd.read_csv(summary)['model'])
            if observed!=set(modes)|{'logistic_anchor'}:
                raise RuntimeError(f'Mismatched E6-D models in {folder}')
            if ref is not None:
                old=ref/f'e6c_dblp_seed{seed}'/'candidate_fingerprints.csv'
                if not old.is_file() or not pd.read_csv(old,dtype={'time':str}).equals(
                                   pd.read_csv(candidates,dtype={'time':str})):
                    raise RuntimeError(f'E6-C/E6-D candidate mismatch for reused seed {seed}')
            print('REUSED:',folder.name,flush=True)
        else:
            run_seed(stream,seed,folder,settings,modes,args.device,ref)
    print('E6-D run complete; aggregate with scripts/e6d_aggregate.py')


if __name__=='__main__': main()
