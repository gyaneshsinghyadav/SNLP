"""E6-G E4 retention-aware checkpoint selection tests."""
import copy
from dataclasses import replace
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from test_e6f_forgetting import snapshot_setup
from e6f_run_suite import E6FSettings,run_seed as run_e6f
from e6g_run_suite import (E6GSettings,run_seed as run_e6g,
    train_year as train_e6g, _historical_checkpoint_scores)


def test_e3_control_matches_original_e6f_exactly(snapshot_setup,tmp_path):
    stream,s,_,_=snapshot_setup
    cfg_f=replace(s,variant='e3')
    cfg_g=E6GSettings(base=s.base,capacity=s.capacity,
        reserve_per_period=s.reserve_per_period,full_replay_budget=s.full_replay_budget,
        light_replay_budget=s.light_replay_budget,max_nodes=s.max_nodes,
        first_hop_limit=s.first_hop_limit,variant='e3',checkpoint_mode='current')
    a=tmp_path/'e6f';b=tmp_path/'e6g'
    run_e6f(stream,7,a,cfg_f,policies=('dqn',),backbones=('structural_subgraph',))
    run_e6g(stream,7,b,cfg_g,policies=('dqn',),backbones=('structural_subgraph',))
    for table in ('model_summary','prequential','retention','controller_decisions','replay_memory'):
        left=pd.read_csv(a/f'{table}.csv');right=pd.read_csv(b/f'{table}.csv')
        shared=[c for c in left if c in right and c not in ('cost_seconds','elapsed_seconds')]
        pd.testing.assert_frame_equal(left[shared],right[shared],atol=1e-5,rtol=1e-5)


def test_e4_retention_constraint_and_frozen_final_years(snapshot_setup,tmp_path):
    stream,s,_,_=snapshot_setup
    cfg=E6GSettings(base=replace(s.base,epochs=2,patience=2),capacity=s.capacity,
        reserve_per_period=s.reserve_per_period,full_replay_budget=s.full_replay_budget,
        light_replay_budget=s.light_replay_budget,max_nodes=s.max_nodes,
        first_hop_limit=s.first_hop_limit,checkpoint_mode='retention',
        retention_tolerance=0.0)
    dest=tmp_path/'e4'
    run_e6g(stream,7,dest,cfg,policies=('dqn',),backbones=('structural_subgraph',))
    ck=pd.read_csv(dest/'checkpoint_selection.csv')
    assert ck.checkpoint_mode.eq('retention').all()
    assert ck.selected_worst_incremental_drop.max() <= 1e-8
    assert (ck.selected_epoch>=0).all()
    assert ck.rejected_epochs.ge(0).all()
    assert ck.eligible_epochs.ge(0).all()
    assert (ck.rolled_back==(ck.selected_epoch==0)).all()
    pre=pd.read_csv(dest/'prequential.csv')
    assert set(pre.query("phase=='test'").time.astype(str))=={'2015','2016'}
    assert set(pre.query("phase=='test'").model_fit_year_through.astype(str))=={'2014'}
    dec=pd.read_csv(dest/'controller_decisions.csv')
    assert not dec.time.astype(str).isin(['2015','2016']).any()
    assert pd.read_csv(dest/'task_audit.csv').label_overlap_count.eq(0).all()
    assert pd.read_csv(dest/'replay_memory.csv').teacher_count.gt(0).all()


def test_e4_sealed_future_edges_cannot_affect_trained_weights(snapshot_setup,tmp_path):
    stream,s,_,_=snapshot_setup
    changed=copy.deepcopy(stream)
    events=changed.network.events.copy()
    mask=events[:,0]>=8
    events[mask,2]=(events[mask,2]+1)%changed.network.n
    changed.network.events=events[events[:,1]!=events[:,2]]
    cfg=E6GSettings(base=s.base,capacity=s.capacity,reserve_per_period=s.reserve_per_period,
        full_replay_budget=s.full_replay_budget,light_replay_budget=s.light_replay_budget,
        max_nodes=s.max_nodes,first_hop_limit=s.first_hop_limit,checkpoint_mode='retention')
    a=tmp_path/'reference';b=tmp_path/'modified_future'
    run_e6g(stream,7,a,cfg,policies=('dqn',),backbones=('structural_subgraph',))
    run_e6g(changed,7,b,cfg,policies=('dqn',),backbones=('structural_subgraph',))
    x=torch.load(a/'structural_subgraph_dqn_frozen.pt',weights_only=True)
    y=torch.load(b/'structural_subgraph_dqn_frozen.pt',weights_only=True)
    for key in x['model']:
        assert torch.equal(x['model'][key],y['model'][key]),key
    aa=pd.read_csv(a/'checkpoint_selection.csv')
    bb=pd.read_csv(b/'checkpoint_selection.csv')
    pd.testing.assert_frame_equal(aa,bb,check_exact=False,rtol=1e-5,atol=1e-5)


def test_checkpoint_scoring_does_not_mutate_historical_records(snapshot_setup):
    from e6g_run_suite import model_init
    _,s,shots,packs=snapshot_setup
    model,_=model_init(shots[0].pt.x.shape[1],s,'structural_subgraph','cpu')
    before=[snap.y.copy() for snap in shots]
    old=_historical_checkpoint_scores(model,shots[:2],{ss.ti:pp for ss,pp in zip(shots,packs)})
    assert len(old)==2
    assert all(0<=x<=1 for x in old.values())
    assert all(np.array_equal(y0,snap.y) for y0,snap in zip(before,shots))


def test_rejected_update_restores_model_and_optimizer(snapshot_setup,monkeypatch):
    """If every epoch harms old validation > tolerance, both states roll back."""
    import e6g_run_suite as runner
    from adaptive_graph.continual import OnlineEWC
    stream,s,shots,packs=snapshot_setup
    cfg=E6GSettings(base=replace(s.base,epochs=1,patience=1),capacity=s.capacity,
        reserve_per_period=s.reserve_per_period,full_replay_budget=s.full_replay_budget,
        light_replay_budget=s.light_replay_budget,max_nodes=s.max_nodes,
        first_hop_limit=s.first_hop_limit,checkpoint_mode='retention',
        retention_tolerance=0.)
    model,opt=runner.model_init(shots[0].pt.x.shape[1],cfg,'structural_subgraph','cpu')
    packet_by_ti={ss.ti:pp for ss,pp in zip(shots,packs)}
    mem=runner.build_memory(cfg,7,packet_by_ti)
    before={k:v.detach().clone() for k,v in model.state_dict().items()}
    before_opt=copy.deepcopy(opt.state_dict())

    def fake_retention_scores(m, completed, packets):
        unmodified=all(torch.equal(m.state_dict()[k],v) for k,v in before.items())
        return {ss.ti:(1. if unmodified else 0.) for ss in completed}
    monkeypatch.setattr(runner,'_historical_checkpoint_scores',fake_retention_scores)
    hist,metric=runner.train_year(model,opt,shots[1],packs[1],mem,packet_by_ti,
        None,OnlineEWC(),cfg,7,completed=[shots[0]])
    assert metric['rolled_back'] and metric['selected_epoch']==0
    assert metric['selected_worst_incremental_drop']==0.
    assert metric['rejected_epochs']>=1
    for k,v in before.items():assert torch.equal(model.state_dict()[k],v),k
    assert opt.state_dict()==before_opt
