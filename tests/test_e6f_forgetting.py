"""Controlled E6-F changes: no sealed-label usage, parity, teachers and chronology."""
import sys
from pathlib import Path
from dataclasses import replace
import numpy as np
import pandas as pd
import torch
import pytest

from adaptive_graph.data import synthetic_network
from adaptive_graph.tgat import TemporalEventStream
from adaptive_graph.e6f_memory import (ReplayWithLogitsMemory,RepresentativeReplayMemory,
                                       label_and_distill_loss)
from adaptive_graph.e6e_continual import E6ReplayMemory

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from e6c_run_suite import Settings,make_snapshots
from e6d_run_suite import E6DSettings,_packets
from e6f_run_suite import E6FSettings,run_seed,model_init,build_memory
from e6e_run_suite import run_seed as original_run_seed

@pytest.fixture(scope='module')
def snapshot_setup(tmp_path_factory):
    torch.set_num_threads(2)
    net=synthetic_network(n=54,periods=10,seed=37)
    net.times=list(range(2007,2017))
    path=tmp_path_factory.mktemp('e6f')/'graph.csv'
    pd.DataFrame([(int(u),int(v),int(net.times[t])) for t,u,v in net.events],
                 columns=['source','target','timestamp']).to_csv(path,index=False)
    stream=TemporalEventStream.from_csv(path,bucket='year',undirected=True)
    base=Settings(epochs=1,patience=1,hidden=16,fusion_dim=16,
                  neighbors=4,heads=4,batch_size=128)
    settings=E6FSettings(base=base,max_nodes=12,first_hop_limit=8,
                         full_replay_budget=8,light_replay_budget=4)
    shots,_,_=make_snapshots(stream,7,base,('fusion',),'cpu')
    packets=_packets(stream,shots,E6DSettings(base=base,max_nodes=12,first_hop_limit=8),'cpu')
    return stream,settings,shots,packets


def test_representative_admission_stratifies_by_year_and_class(snapshot_setup):
    _,s,shots,packs=snapshot_setup
    model,_=model_init(shots[0].pt.x.shape[1],s,'structural_subgraph','cpu')
    memory=RepresentativeReplayMemory(capacity=48,reserve_per_period=2,seed=7)
    memory.set_packets({snap.ti:pk for snap,pk in zip(shots,packs)})
    for snap,packet in zip(shots[:5],packs[:5]):
        memory.add(model,snap,packet,mode='adaptive')
    assert len(memory.items)==48
    assert memory.coverage()==5
    assert all(item.index in set(item.snapshot.fit_idx.tolist()) for item in memory.items)
    assert all(item.index not in set(item.snapshot.val_idx.tolist()) for item in memory.items)
    assert all(item.index not in set(item.snapshot.pt.task.eval_idx.tolist()) for item in memory.items)
    counts=pd.DataFrame([(i.snapshot.ti,i.label) for i in memory.items],columns=['time','label'])
    for _,grp in counts.groupby('time'):
        assert abs(int((grp.label==1).sum())-int((grp.label==0).sum()))<=2
    sample=memory.sample(16,'adaptive',shots[5].ti)
    assert len({snap.ti for snap,ids in sample})==5
    assert sum(len(ids) for _,ids in sample)==16


def test_distillation_teacher_is_immutable_and_uses_fit_only(snapshot_setup):
    _,s,shots,packs=snapshot_setup
    model,_=model_init(shots[0].pt.x.shape[1],s,'structural_subgraph','cpu')
    memory=ReplayWithLogitsMemory(capacity=80,seed=7,distill=True)
    memory.add(model,shots[0],packs[0])
    old=dict(memory.teacher_logits)
    assert len(old)==len(memory.items)>0
    assert set(x[1] for x in old).issubset(set(shots[0].fit_idx.tolist()))
    for p in model.parameters():
        if p.requires_grad:
            with torch.no_grad():p.add_(torch.ones_like(p)*0.001)
            break
    memory.add(model,shots[1],packs[1])
    assert all(memory.teacher_logits[k]==v for k,v in old.items() if k in memory.teacher_logits)
    groups=memory.sample(8,'adaptive',shots[2].ti)
    loss,diff=label_and_distill_loss(model,groups,
        {snap.ti:p for snap,p in zip(shots,packs)},memory,0.1)
    assert torch.isfinite(loss) and torch.isfinite(diff)
    assert diff.item()>=0
    assert diff.requires_grad


def test_e0_is_original_runner_on_same_synthetic_tasks(snapshot_setup,tmp_path):
    stream,s,_,_=snapshot_setup
    old=replace(s,variant='e0')
    original=type(s).__mro__[1](base=s.base,capacity=s.capacity,
        reserve_per_period=s.reserve_per_period,full_replay_budget=s.full_replay_budget,
        light_replay_budget=s.light_replay_budget,max_nodes=s.max_nodes,
        first_hop_limit=s.first_hop_limit)
    a=tmp_path/'original';b=tmp_path/'new'
    original_run_seed(stream,7,a,original,policies=('dqn',),backbones=('structural_subgraph',))
    run_seed(stream,7,b,old,policies=('dqn',),backbones=('structural_subgraph',))
    for table in ('model_summary','prequential','retention','controller_decisions'):
        left=pd.read_csv(a/f'{table}.csv');right=pd.read_csv(b/f'{table}.csv')
        # Runtime is machine-noisy; compare every reproducible output.
        common=[c for c in left.columns if c in right.columns and
                c not in ('cost_seconds','elapsed_seconds')]
        if 'compute_cost_units' in common:pass
        pd.testing.assert_frame_equal(left[common],right[common],atol=1e-5,rtol=1e-5)


@pytest.mark.parametrize('variant',['e1','e2','e3'])
def test_all_changes_keep_dqn_and_frozen_test_years(snapshot_setup,tmp_path,variant):
    stream,s,_,_=snapshot_setup
    run_seed(stream,7,tmp_path/variant,replace(s,variant=variant),
             policies=('dqn',),backbones=('structural_subgraph',))
    out=tmp_path/variant
    pre=pd.read_csv(out/'prequential.csv')
    assert set(pre.query("phase=='test'").time.astype(str))=={'2015','2016'}
    assert set(pre.query("phase=='test'").model_fit_year_through.astype(str))=={'2014'}
    decisions=pd.read_csv(out/'controller_decisions.csv')
    assert decisions.policy.eq('dqn').all()
    assert decisions.time.astype(str).isin(['2015','2016']).sum()==0
    assert pd.read_csv(out/'task_audit.csv').label_overlap_count.eq(0).all()
    memory=pd.read_csv(out/'replay_memory.csv')
    assert memory.stored_pairs.le(s.capacity).all()
    if variant in ('e2','e3'):
        assert (memory.teacher_count==memory.stored_pairs).all()
    else:
        assert (memory.teacher_count==0).all()


def test_e3_training_is_independent_of_sealed_future_edges(snapshot_setup,tmp_path):
    import copy
    stream,s,_,_=snapshot_setup
    changed=copy.deepcopy(stream)
    net=changed.network
    events=net.events.copy()
    mask=events[:,0]>=8
    events[mask,2]=(events[mask,2]+1)%net.n
    net.events=events[events[:,1]!=events[:,2]]
    cfg=replace(s,variant='e3')
    first=tmp_path/'historical';second=tmp_path/'changed_future'
    run_seed(stream,7,first,cfg,policies=('dqn',),backbones=('structural_subgraph',))
    run_seed(changed,7,second,cfg,policies=('dqn',),backbones=('structural_subgraph',))
    a=pd.read_csv(first/'controller_decisions.csv')
    b=pd.read_csv(second/'controller_decisions.csv')
    pd.testing.assert_frame_equal(a[['time','action','current_val_after','mean_forgetting','replayed_pairs']],
                                  b[['time','action','current_val_after','mean_forgetting','replayed_pairs']],
                                  atol=1e-5,rtol=1e-5)
    x=torch.load(first/'structural_subgraph_dqn_frozen.pt',weights_only=True)
    y=torch.load(second/'structural_subgraph_dqn_frozen.pt',weights_only=True)
    for name in x['model']:
        assert torch.equal(x['model'][name],y['model'][name]),name
