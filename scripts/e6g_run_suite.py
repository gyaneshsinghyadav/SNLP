#!/usr/bin/env python3
"""E6-G: controlled E3 vs E4 retention-aware checkpoint selection on E6-F GNN+DQN.

Strict prequential protocol: score year t before learning its fit labels; use
only fit subsets for model gradients, memory and EWC; same-year controller
validation is disjoint from fit and sealed eval; freeze during final test years.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from copy import deepcopy
from dataclasses import asdict,dataclass,replace
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from sklearn.metrics import average_precision_score,roc_auc_score

from adaptive_graph.tgat import TemporalEventStream
from adaptive_graph.e6_higher_order import HigherOrderStructuralTemporalFusion
from adaptive_graph.e6e_continual import E6ReplayMemory,memory_loss,policy_state
from adaptive_graph.e6f_memory import ReplayWithLogitsMemory,RepresentativeReplayMemory,label_and_distill_loss
from adaptive_graph.continual import OnlineEWC
from adaptive_graph.temporal_control import ControllerOptions,TemporalDQN, E32_ACTIONS
from adaptive_graph.gradient_conflict import projection_coefficient
from e6c_run_suite import Settings,make_snapshots,_seed,_prior_logits
from e6d_run_suite import E6DSettings,_packets,_score
from e6_run_suite import DEFAULT_E4_SHA,sha

POLICIES=('dqn',)
BACKBONES=('structural_subgraph','fusion_subgraph')
ARMS=('e3_control','e4_retention')

@dataclass(frozen=True)
class IntegrationSettings:
    base:Settings=Settings()
    capacity:int=256
    reserve_per_period:int=4
    full_replay_budget:int=64
    light_replay_budget:int=16
    replay_weight:float=.5
    ewc_lambda:float=.2
    ewc_decay:float=.9
    reward_forgetting_lambda:float=.75
    reward_peak_lambda:float=.25
    reward_cost_lambda:float=.001
    epsilon:float=.35
    gamma:float=.9
    max_nodes:int=32
    first_hop_limit:int=20
    probe_pairs:int=16
    conflict_guard_threshold:float=-.05
    retention_guard_threshold:float=.02


@dataclass(frozen=True)
class E6GSettings(IntegrationSettings):
    variant:str='e3'  # Representative memory and historical-logit distillation always enabled.
    distill_alpha:float=0.1  # Keep the measured E3 configuration identical.
    checkpoint_mode:str='current'  # 'current' E3 control, 'retention' E4.
    retention_tolerance:float=0.02  # Maximum allowed incremental AP damage per old task.



def build_memory(settings,seed,packets):
    kw=dict(capacity=settings.capacity,reserve_per_period=settings.reserve_per_period,seed=seed)
    if settings.variant=='e0':
        return E6ReplayMemory(**kw)
    if settings.variant=='e1':
        memory=RepresentativeReplayMemory(**kw,distill=False)
    elif settings.variant=='e2':
        memory=ReplayWithLogitsMemory(**kw,distill=True)
    elif settings.variant=='e3':
        memory=RepresentativeReplayMemory(**kw,distill=True)
    else:
        raise ValueError('Unknown E6-F experimental variant')
    if isinstance(memory,RepresentativeReplayMemory):memory.set_packets(packets)
    return memory


def model_init(input_dim,settings,backbone,device):
    base=settings.base
    model=HigherOrderStructuralTemporalFusion(input_dim,hidden=base.hidden,heads=base.heads,
        fusion_dim=base.fusion_dim,dropout=base.dropout,mode=backbone).to(device)
    temporal=list(model.base.temporal.parameters())
    tids={id(p) for p in temporal}
    rest=[p for p in model.parameters() if id(p) not in tids]
    opt=torch.optim.AdamW([{'params':temporal,'lr':base.lr_tgat},
                           {'params':rest,'lr':base.lr_decoder}],
                          weight_decay=base.weight_decay)
    return model,opt


def probe(model,snap,packet,memory,packets,max_pairs=16):
    if not memory.items:
        return {'available':0,'cosine':0.,'norm_ratio':0.,'reference_pairs':0,'conflict':0}
    items=sorted(memory.items,key=lambda e:(e.snapshot.ti,e.index))
    sel=[items[int(i)] for i in np.linspace(0,len(items)-1,min(max_pairs,len(items)),dtype=int)]
    grouped=defaultdict(list);contexts={}
    for e in sel:
        grouped[e.snapshot.ti].append(e.index)
        contexts[e.snapshot.ti]=e.snapshot
    groups=[(contexts[t],np.asarray(idx,dtype=np.int64)) for t,idx in sorted(grouped.items())]
    indices=snap.fit_idx[:max_pairs]
    if not len(indices):return {'available':0,'cosine':0.,'norm_ratio':0.,'reference_pairs':0,'conflict':0}
    params=[p for p in model.parameters() if p.requires_grad]
    was_training=model.training
    model.eval()
    try:
        cur=model(snap.pt,snap.structural,snap.prior['fusion'],packet,ids=indices)
        target=torch.tensor(snap.y[indices],device=cur.device,dtype=torch.float32)
        cg=torch.autograd.grad(F.binary_cross_entropy_with_logits(cur,target),params,allow_unused=True)
        old_loss=memory_loss(model,groups,packets)
        og=torch.autograd.grad(old_loss,params,allow_unused=True)
        dot=sum(float((a.detach()*b.detach()).sum()) for a,b in zip(cg,og) if a is not None and b is not None)
        an=sum(float(a.detach().square().sum()) for a in cg if a is not None)
        bn=sum(float(b.detach().square().sum()) for b in og if b is not None)
        valid=an>1e-20 and bn>1e-20
        cosine=float(np.clip(dot/(np.sqrt(an*bn)+1e-12),-1,1)) if valid else 0.
        return {'available':int(valid),'cosine':cosine,
                'norm_ratio':float(np.sqrt(bn)/(np.sqrt(an)+1e-12)) if valid else 0.,
                'reference_pairs':len(sel),'conflict':int(valid and dot<0)}
    finally:
        model.train(was_training)


def _historical_checkpoint_scores(model, completed, packets):
    """Non-mutating score of *past training years' validation subsets*.

    Never calls historical_retention, which updates the best-so-far dictionary.
    The eval_idx of every period remains untouched by checkpoint selection.
    """
    return {int(old.ti): float(average_precision_score(
        old.y[old.val_idx], _score(model,old,packets[old.ti],old.val_idx)))
        for old in completed}


def train_year(model,opt,snap,packet,memory,packets,action,ewc,settings,seed,
               completed=()):
    base=settings.base
    best_state=deepcopy(model.state_dict());best_opt=deepcopy(opt.state_dict())
    use_retention=settings.checkpoint_mode=='retention'
    if settings.checkpoint_mode not in ('current','retention'):
        raise ValueError('Invalid checkpoint mode')
    # Only historical training-era validation is used to choose checkpoints.
    # Baseline is measured BEFORE any optimizer update, and stays eligible.
    initial_ap=(float(average_precision_score(snap.y[snap.val_idx],
        _score(model,snap,packet,snap.val_idx))) if use_retention else float('nan'))
    old_before=(_historical_checkpoint_scores(model,completed,packets)
                if use_retention and completed else {})
    best_ap=initial_ap if use_retention else -np.inf
    selected_epoch=0 if use_retention else None
    eligible_epochs=0;rejected_epochs=0
    stale=0;history=[];updates=0;projected=0;replayed=0
    start=time.perf_counter()
    for epoch in range(base.epochs):
        model.train();rng=np.random.default_rng(seed*1613+snap.ti*23+epoch)
        order=snap.fit_idx.copy();rng.shuffle(order)
        chunks=np.array_split(order,max(1,(len(order)+base.batch_size-1)//base.batch_size))
        losses=[]
        for idx in chunks:
            if not len(idx):continue
            logits=model(snap.pt,snap.structural,snap.prior['fusion'],packet,ids=idx)
            target=torch.tensor(snap.y[idx],dtype=torch.float32,device=logits.device)
            cur_loss=F.binary_cross_entropy_with_logits(logits,target)
            pos=logits[target>.5];neg=logits[target<.5]
            if len(pos) and len(neg) and base.pairwise_rank_weight:
                cur_loss=cur_loss+base.pairwise_rank_weight*F.softplus(neg[None,:]-pos[:,None]).mean()
            budget=(0 if action is None else settings.full_replay_budget if action.replay_fraction==1
                    else settings.light_replay_budget if action.replay_fraction>0 else 0)
            groups=memory.sample(budget,action.replay_mode if action else 'uniform',snap.ti) if budget else []
            replayed+=sum(len(ij) for _,ij in groups)
            distill_loss=None
            if settings.variant in ('e2','e3') and groups:
                old_loss,distill_loss=label_and_distill_loss(
                    model,groups,packets,memory,settings.distill_alpha)
            else:
                old_loss=memory_loss(model,groups,packets) if groups else None
            loss=cur_loss
            if old_loss is not None:
                loss=loss+settings.replay_weight*old_loss
            if distill_loss is not None:
                loss=loss+settings.distill_alpha*distill_loss
            if action is not None and action.ewc_scale>0 and ewc.fisher:
                loss=loss+settings.ewc_lambda*action.ewc_scale*ewc.penalty(model)
            correction=None
            if action is not None and action.projected and old_loss is not None:
                params=[p for p in model.parameters() if p.requires_grad]
                grad_new=torch.autograd.grad(cur_loss,params,retain_graph=True,allow_unused=True)
                grad_old=torch.autograd.grad(old_loss,params,retain_graph=True,allow_unused=True)
                coeff=projection_coefficient(grad_new,grad_old)
                correction=(coeff,params,grad_old)
                projected+=int(coeff>0)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if correction is not None:
                coeff,params,grad_old=correction
                if coeff:
                    for p,g in zip(params,grad_old):
                        if p.grad is not None and g is not None:
                            p.grad.add_(g.detach(),alpha=coeff)
            torch.nn.utils.clip_grad_norm_(model.parameters(),base.gradient_clip)
            opt.step();updates+=1;losses.append(float(loss.detach()))
        pred=_score(model,snap,packet,snap.val_idx)
        ap=float(average_precision_score(snap.y[snap.val_idx],pred))
        old_after=(_historical_checkpoint_scores(model,completed,packets)
                   if use_retention and completed else {})
        declines={ti:max(0.,old_before[ti]-old_after[ti]) for ti in old_before}
        worst=max(declines.values(),default=0.)
        mean=float(np.mean(list(declines.values()))) if declines else 0.
        eligible=(not use_retention or worst<=settings.retention_tolerance+1e-12)
        eligible_epochs+=int(eligible);rejected_epochs+=int(not eligible)
        history.append({'time':snap.year,'epoch':epoch+1,'validation_ap':ap,
                        'train_loss':float(np.mean(losses)),'action':action.name if action else 'no_replay',
                        'distill_alpha':settings.distill_alpha if settings.variant in ('e2','e3') else 0.,
                        'checkpoint_eligible':bool(eligible),
                        'incremental_mean_old_drop':mean,
                        'incremental_worst_old_drop':worst})
        if eligible and ap>best_ap+1e-6:
            best_ap=ap;best_state=deepcopy(model.state_dict());best_opt=deepcopy(opt.state_dict());
            selected_epoch=epoch+1;stale=0
        else:
            stale+=1
            if stale>=base.patience:break
    model.load_state_dict(best_state);opt.load_state_dict(best_opt)
    chosen_old=(_historical_checkpoint_scores(model,completed,packets)
                if use_retention and completed else {})
    chosen_declines=[max(0.,old_before[ti]-chosen_old[ti]) for ti in old_before]
    return history,{'updates':updates,'replayed_pairs':replayed,
                    'projected_steps':projected,'cost':time.perf_counter()-start,
                    'checkpoint_mode':settings.checkpoint_mode,
                    'selected_epoch':selected_epoch,
                    'pre_current_ap':initial_ap,
                    'selected_current_ap':best_ap,
                    'eligible_epochs':eligible_epochs,
                    'rejected_epochs':rejected_epochs,
                    'rolled_back':bool(use_retention and selected_epoch==0),
                    'selected_mean_incremental_drop':float(np.mean(chosen_declines)) if chosen_declines else 0.,
                    'selected_worst_incremental_drop':max(chosen_declines,default=0.)}


def consolidate(model,snap,packet,ewc):
    """Estimate EWC curvature from revealed fit examples, never val or eval."""
    ids=snap.fit_idx[:min(32,len(snap.fit_idx))]
    if not len(ids):return
    was_training=model.training;model.eval()
    try:
        jobs=[]
        for part in np.array_split(ids,max(1,(len(ids)+15)//16)):
            def callback(part=part):
                score=model(snap.pt,snap.structural,snap.prior['fusion'],packet,ids=part)
                target=torch.tensor(snap.y[part],device=score.device,dtype=torch.float32)
                return F.binary_cross_entropy_with_logits(score,target)
            jobs.append(callback)
        ewc.consolidate(model,jobs)
    finally:model.train(was_training)


def historical_retention(model,completed,packets,best_ap):
    rows=[]
    for old in completed:
        pred=_score(model,old,packets[old.ti],old.val_idx)
        score=float(average_precision_score(old.y[old.val_idx],pred))
        ref=best_ap.get(old.ti,score)
        drop=max(0.,ref-score)
        best_ap[old.ti]=max(ref,score)
        rows.append({'historical_year':old.year,'historical_ap':score,
                     'historical_best_ap':best_ap[old.ti],'ap_drop':drop})
    mean=float(np.mean([r['ap_drop'] for r in rows])) if rows else 0.
    peak=float(max((r['ap_drop'] for r in rows),default=0.))
    mean_ap=float(np.mean([r['historical_ap'] for r in rows])) if rows else 0.
    return rows,mean,peak,mean_ap


def run_seed(stream,seed,dest,settings,policies=POLICIES,backbones=('structural_subgraph',),
             device='cpu',e6d_reference=None):
    if settings.reserve_per_period * (len(stream.network.times)-settings.base.test_last-settings.base.start_time_index)>settings.capacity:
        raise ValueError('Replay reserve exceeds capacity for all training periods')
    start=time.perf_counter();_seed(seed)
    snaps,audit,keys=make_snapshots(stream,seed,settings.base,('fusion',),device)
    if e6d_reference is not None:
        archived=e6d_reference/f'e6d_dblp_seed{seed}'/'candidate_fingerprints.csv'
        if not archived.is_file():raise FileNotFoundError(f'Missing E6-D reference: {archived}')
        old=pd.read_csv(archived,dtype={'time':str})
        if not old.equals(keys):raise RuntimeError('E6-D candidate fingerprints differ: invalid comparison')
    e6d_settings=E6DSettings(base=settings.base,max_nodes=settings.max_nodes,
                             first_hop_limit=settings.first_hop_limit)
    packets=_packets(stream,snaps,e6d_settings,device)
    packet_by_ti={s.ti:p for s,p in zip(snaps,packets)}
    n_train=len(snaps)-settings.base.test_last
    results=[];scores=[];history=[];decisions=[];retention=[];memory_rows=[];rewards=[];checkpoints=[]
    for backbone in backbones:
        for policy in policies:
            _seed(seed)
            model,opt=model_init(snaps[0].pt.x.shape[1],settings,backbone,device)
            ewc=OnlineEWC(decay=settings.ewc_decay)
            memory=build_memory(settings,seed,packet_by_ti)
            dqn=None
            if policy=='dqn':
                opts=ControllerOptions(policy='dqn',fixed_action='adaptive_full',action_space='e32',
                    state_version='e32',gamma=settings.gamma,controller_updates=2,
                    retention_guard_threshold=settings.retention_guard_threshold,
                    conflict_guard_threshold=settings.conflict_guard_threshold)
                dqn=TemporalDQN(opts,seed,device)
            completed=[];best_ap={};prev={};pending=None
            frozen_prior=None
            if policy=='frozen':
                # Truly frozen LR anchor too: later scores use an LR fitted only
                # on first revealed year's learner-fit examples, not the usual
                # prequential anchor (which refits through 2014).
                first=snaps[0]
                xx=first.structural[first.fit_idx].detach().cpu().numpy()
                yy=first.y[first.fit_idx]
                frozen_prior={s.ti:torch.tensor(_prior_logits([xx],[yy],
                    s.structural.detach().cpu().numpy(),seed,'fusion'),
                    dtype=torch.float32,device=s.pt.x.device) for s in snaps[1:]}
            update_count=0;total_replay=0;total_project=0
            for i,(snap,packet) in enumerate(zip(snaps,packets)):
                phase='train' if i<n_train else 'test'
                if i>0:
                    ids=snap.pt.task.eval_idx
                    scored_snap=(replace(snap,prior={'fusion':frozen_prior[snap.ti]})
                                 if policy=='frozen' else snap)
                    pred=_score(model,scored_snap,packet,ids)
                    scores.append({'seed':seed,'backbone':backbone,'policy':policy,
                        'time':snap.year,'phase':phase,
                        'ap':float(average_precision_score(snap.y[ids],pred)),
                        'roc_auc':float(roc_auc_score(snap.y[ids],pred)),
                        'candidate_count':len(ids), 'positive_count':int(snap.y[ids].sum()),
                        'model_fit_year_through':(snaps[0].year if policy=='frozen' else
                            snaps[n_train-1].year if phase=='test' else snaps[i-1].year)})
                if phase=='test':
                    continue  # No training, policy updates, EWC, or replay admission.
                if i==0:
                    action=None;trace={};state=None;pre=0.;pr={'available':0,'cosine':0.,'norm_ratio':0,'reference_pairs':0,'conflict':0}
                elif policy=='frozen':
                    continue  # Frozen after first train-year warmup.
                else:
                    pre=float(average_precision_score(snap.y[snap.val_idx],
                        _score(model,snap,packet,snap.val_idx)))
                    pr=probe(model,snap,packet,memory,packet_by_ti,settings.probe_pairs)
                    state=policy_state(snap,stream.network,memory,pre,prev,pr,bool(ewc.fisher))
                    if policy=='dqn':
                        if pending is not None:
                            loss=dqn.observe(pending['state'],pending['action'],pending['reward'],state)
                            rewards.append({'seed':seed,'backbone':backbone,'policy':policy,
                                'credited_year':pending['year'],'td_loss':loss,'terminal':False,
                                'reward':pending['reward']})
                        ai,trace=dqn.choose_with_trace(state,epsilon=settings.epsilon)
                        action=E32_ACTIONS[ai]
                    else:
                        trace={};action=next((a for a in E32_ACTIONS if a.name==policy),None)
                        if policy=='no_replay':action=None
                        elif action is None:raise ValueError(f'Invalid policy: {policy}')
                h,metrics=train_year(model,opt,snap,packet,memory,packet_by_ti,action,ewc,settings,seed,
                                     completed=completed)
                checkpoints.append({'seed':seed,'backbone':backbone,'policy':policy,'time':snap.year,
                    'action':action.name if action else 'no_replay',
                    'checkpoint_mode':metrics['checkpoint_mode'],
                    'selected_epoch':metrics['selected_epoch'],
                    'rolled_back':metrics['rolled_back'],
                    'pre_current_ap':metrics['pre_current_ap'],
                    'selected_current_ap':metrics['selected_current_ap'],
                    'eligible_epochs':metrics['eligible_epochs'],
                    'rejected_epochs':metrics['rejected_epochs'],
                    'selected_mean_incremental_drop':metrics['selected_mean_incremental_drop'],
                    'selected_worst_incremental_drop':metrics['selected_worst_incremental_drop']})
                update_count+=metrics['updates'];total_replay+=metrics['replayed_pairs'];total_project+=metrics['projected_steps']
                history.extend({'seed':seed,'backbone':backbone,'policy':policy,**row} for row in h)
                # Historical reference values are measured ONLY on training-era
                # controller-validation subsets. No sealed eval labels enter reward.
                old_rows,mean_drop,peak_drop,historical_ap=historical_retention(model,completed,packet_by_ti,best_ap)
                retention.extend({'seed':seed,'backbone':backbone,'policy':policy,
                                  'after_year':snap.year,**r} for r in old_rows)
                current_after=float(average_precision_score(snap.y[snap.val_idx],
                                        _score(model,snap,packet,snap.val_idx)))
                if i>0:
                    reward=(current_after-pre-settings.reward_forgetting_lambda*mean_drop
                            -settings.reward_peak_lambda*peak_drop
                            -settings.reward_cost_lambda*(metrics['updates']+metrics['replayed_pairs']/64.))
                    decisions.append({'seed':seed,'backbone':backbone,'policy':policy,
                        'time':snap.year,'action':action.name if action else 'no_replay',
                        'current_val_before':pre,'current_val_after':current_after,
                        'current_val_gain':current_after-pre,'mean_forgetting':mean_drop,
                        'peak_forgetting':peak_drop,'historical_val_ap':historical_ap,
                        'reward':reward,'compute_cost_units':metrics['updates']+metrics['replayed_pairs']/64.,'gradient_cosine':pr['cosine'],
                        'gradient_conflict':pr['conflict'],'gradient_probe_available':pr['available'],
                        'replayed_pairs':metrics['replayed_pairs'],
                        'projected_steps':metrics['projected_steps'],
                        'cost_seconds':metrics['cost'],
                        'explored':bool(trace.get('explored',False)),
                        'guard_active':bool(trace.get('guard_active',False))})
                    if policy=='dqn':pending={'state':state.copy(),'action':ai,'reward':reward,'year':snap.year}
                    prev={'mean_forgetting':mean_drop,'peak_forgetting':peak_drop,
                          'historical_ap':historical_ap,'val_gain':current_after-pre,
                          'cost':metrics['cost'],'action_index':
                          next((j for j,a in enumerate(E32_ACTIONS) if action and a.name==action.name),0)}
                else:
                    prev={'mean_forgetting':mean_drop,'peak_forgetting':peak_drop,
                          'historical_ap':historical_ap,'val_gain':0.,'cost':metrics['cost']}
                completed.append(snap)
                best_ap[snap.ti]=current_after
                # Fitting-only EWC and memory admission; after action and reward.
                consolidate(model,snap,packet,ewc)
                memory.add(model,snap,packet,action.replay_mode if action else 'uniform')
                memory_rows.append({'seed':seed,'backbone':backbone,'policy':policy,'after_year':snap.year,
                    'stored_pairs':len(memory.items),'periods':memory.coverage(),
                    'positive_share':memory.positive_share(),'cumulative_replayed_pairs':memory.replayed_pairs,
                    'teacher_count':len(getattr(memory,'teacher_logits',{}))})
                print(f'[E6-F] seed={seed} backbone={backbone} policy={policy} year={snap.year} done',flush=True)
            if dqn and pending is not None:
                # Last action can receive its realized training-year reward,
                # but never any transition or reward from frozen test years.
                loss=dqn.observe(pending['state'],pending['action'],pending['reward'],
                                 np.zeros_like(pending['state']),terminal=True)
                rewards.append({'seed':seed,'backbone':backbone,'policy':policy,
                    'credited_year':pending['year'],'td_loss':loss,'terminal':True,
                    'reward':pending['reward']})
            test=[r for r in scores if r['seed']==seed and r['backbone']==backbone
                  and r['policy']==policy and r['phase']=='test']
            if len(test)!=settings.base.test_last:raise RuntimeError('Missing sealed test scores')
            ret=[r['ap_drop'] for r in retention if r['seed']==seed and r['backbone']==backbone and r['policy']==policy]
            results.append({'seed':seed,'backbone':backbone,'policy':policy,
                'test_mean_ap':float(np.mean([r['ap'] for r in test])),
                'test_mean_roc_auc':float(np.mean([r['roc_auc'] for r in test])),
                'mean_forgetting':float(np.mean(ret)) if ret else 0.,
                'peak_forgetting':float(max(ret,default=0.)),
                'replayed_pairs':total_replay,'projected_steps':total_project,
                'train_updates':update_count,
                'rollback_years':sum(int(r['rolled_back']) for r in checkpoints
                    if r['seed']==seed and r['backbone']==backbone and r['policy']==policy),
                'rejected_epochs':sum(int(r['rejected_epochs']) for r in checkpoints
                    if r['seed']==seed and r['backbone']==backbone and r['policy']==policy)})
            dest.mkdir(parents=True,exist_ok=True)
            torch.save({'model':model.state_dict(),'backbone':backbone,'policy':policy,
                'seed':seed,'settings':asdict(settings),
                'fit_year_through':snaps[n_train-1].year if policy!='frozen' else snaps[0].year},
                dest/f'{backbone}_{policy}_frozen.pt')
            if dqn:
                torch.save({'policy':dqn.q.state_dict(),'state_size':25,
                    'actions':[a.name for a in E32_ACTIONS],
                    'fit_year_through':snaps[n_train-1].year},
                    dest/f'{backbone}_dqn_controller.pt')
    dest.mkdir(parents=True,exist_ok=True)
    for name,rows in [('model_summary',results),('prequential',scores),('training_history',history),
                      ('controller_decisions',decisions),('controller_rewards',rewards),
                      ('retention',retention),('replay_memory',memory_rows),
                      ('checkpoint_selection',checkpoints)]:
        pd.DataFrame(rows,columns=(['seed','backbone','policy','time','action'] if name=='controller_decisions' and not rows else
                                  ['seed','backbone','policy','credited_year','reward'] if name=='controller_rewards' and not rows else None)).to_csv(
                                      dest/f'{name}.csv',index=False)
    audit.to_csv(dest/'task_audit.csv',index=False)
    keys.to_csv(dest/'candidate_fingerprints.csv',index=False)
    (dest/'manifest.json').write_text(json.dumps({'phase':'E6-G','seed':seed,
        'settings':asdict(settings),'policies':list(policies),'backbones':list(backbones),
        'train_years':[s.year for s in snaps[:n_train]],'test_years':[s.year for s in snaps[n_train:]],
        'e6d_candidate_matched':e6d_reference is not None,'elapsed_seconds':time.perf_counter()-start},indent=2)+'\n')
    return results


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--csv',type=Path,required=True)
    p.add_argument('--expected-sha',default=DEFAULT_E4_SHA)
    p.add_argument('--seeds',default='7,13,42,101,202')
    p.add_argument('--arms',default='e3_control,e4_retention')
    p.add_argument('--results-dir',type=Path,default=Path('results/e6g_checkpoint'))
    p.add_argument('--e6d-reference',type=Path,default=Path('docs/e6e_e6d_reference'))
    p.add_argument('--distill-alpha',type=float,default=.1)
    p.add_argument('--retention-tolerance-pp',type=float,default=2.)
    p.add_argument('--epochs',type=int,default=Settings.epochs)
    p.add_argument('--patience',type=int,default=Settings.patience)
    p.add_argument('--max-subgraph-nodes',type=int,default=32)
    p.add_argument('--first-hop-limit',type=int,default=20)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    a=p.parse_args(argv)
    if not a.csv.is_file():p.error(f'Dataset missing: {a.csv}')
    digest=sha(a.csv)
    if digest!=a.expected_sha:p.error(f'Dataset fingerprint mismatch: {digest}')
    seeds=[int(x.strip()) for x in a.seeds.split(',')]
    arms=tuple(x.strip() for x in a.arms.split(','))
    if (not seeds or len(seeds)!=len(set(seeds)) or not arms or len(arms)!=len(set(arms))
        or any(arm not in ARMS for arm in arms)):
        p.error('Invalid seeds or checkpoint arms')
    if (a.epochs<1 or a.patience<1 or not 0<=a.distill_alpha<=5
        or not 0<=a.retention_tolerance_pp<=100):
        p.error('Invalid training parameters')
    if a.device=='cuda' and not torch.cuda.is_available():p.error('CUDA requested but unavailable')
    torch.set_num_threads(min(torch.get_num_threads(),2))
    root=Path(__file__).resolve().parents[1]
    out=a.results_dir if a.results_dir.is_absolute() else root/a.results_dir
    ref=(None if str(a.e6d_reference).lower()=='none' else
         a.e6d_reference if a.e6d_reference.is_absolute() else root/a.e6d_reference)
    files=['src/adaptive_graph/e6e_continual.py','src/adaptive_graph/e6f_memory.py',
           'scripts/e6f_run_suite.py','scripts/e6g_run_suite.py',
           'src/adaptive_graph/e6_higher_order.py','src/adaptive_graph/e6_subgraph.py',
           'src/adaptive_graph/e6_fusion.py','src/adaptive_graph/temporal_control.py',
           'src/adaptive_graph/continual.py','scripts/e6c_run_suite.py','scripts/e6d_run_suite.py']
    stream=TemporalEventStream.from_csv(a.csv,bucket='year',undirected=True)
    if list(map(str,stream.network.times[-2:]))!=['2015','2016']:
        p.error('Expected final frozen 2015 and 2016 test years')
    print(f'[E6-G] dataset SHA256 {digest}; nodes={stream.network.n}; events={len(stream.network.events)}',flush=True)
    for arm in arms:
        settings=E6GSettings(base=Settings(epochs=a.epochs,patience=a.patience),
            max_nodes=a.max_subgraph_nodes,first_hop_limit=a.first_hop_limit,
            variant='e3',distill_alpha=a.distill_alpha,
            checkpoint_mode='current' if arm=='e3_control' else 'retention',
            retention_tolerance=a.retention_tolerance_pp/100.)
        arm_root=out/arm
        arm_root.mkdir(parents=True,exist_ok=True)
        protocol={'phase':'E6-G','dataset_sha256':digest,
                  'code_sha256':{f:sha(root/f) for f in files},
                  'settings':asdict(settings),'policies':['dqn'],
                  'backbones':['structural_subgraph']}
        lock=arm_root/'e6g_protocol.json'
        if lock.exists() and json.loads(lock.read_text())!=json.loads(json.dumps(protocol)):
            raise RuntimeError(f'Incompatible E6-G run: {arm_root}. Choose a new --results-dir')
        if not lock.exists():lock.write_text(json.dumps(protocol,indent=2)+'\n')
        for seed in seeds:
            dest=arm_root/f'e6g_dblp_seed{seed}'
            if dest.exists():
                m=dest/'manifest.json';summary=dest/'model_summary.csv'
                if not m.is_file() or not summary.is_file():
                    raise RuntimeError(f'Incomplete E6-G output: {dest}')
                recorded=json.loads(m.read_text())
                if (recorded['settings']!=asdict(settings) or
                    recorded['policies']!=['dqn'] or recorded['backbones']!=['structural_subgraph']):
                    raise RuntimeError(f'Incompatible preexisting output: {dest}')
                print('REUSED:',dest);continue
            run_seed(stream,seed,dest,settings,('dqn',),('structural_subgraph',),a.device,ref)
    print('Completed. Aggregate using scripts/e6g_compare.py',flush=True)

if __name__=='__main__':main()
