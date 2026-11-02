"""E6-E integration checks: memory separation, EWC, DQN, frozen test protocol."""
import sys
from pathlib import Path
from dataclasses import replace
import numpy as np
import pandas as pd
import pytest
import torch
from adaptive_graph.data import synthetic_network
from adaptive_graph.tgat import TemporalEventStream
from adaptive_graph.e6e_continual import E6ReplayMemory,policy_state
from adaptive_graph.temporal_control import E32_ACTIONS
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from e6c_run_suite import Settings,make_snapshots
from e6d_run_suite import E6DSettings,_packets
from e6e_run_suite import IntegrationSettings,run_seed,model_init,probe

@pytest.fixture
def fixture_stream(tmp_path):
    net=synthetic_network(n=54,periods=10,seed=37)
    net.times=list(range(2007,2017))
    path=tmp_path/'graph.csv'
    pd.DataFrame([(int(u),int(v),int(net.times[t])) for t,u,v in net.events],
                 columns=['source','target','timestamp']).to_csv(path,index=False)
    return TemporalEventStream.from_csv(path,bucket='year',undirected=True)

@pytest.fixture
def setup(fixture_stream):
    torch.set_num_threads(2)
    settings=IntegrationSettings(base=Settings(epochs=1,patience=1,hidden=16,
        fusion_dim=16,neighbors=4,heads=4,batch_size=128),max_nodes=12,first_hop_limit=8,
        full_replay_budget=8,light_replay_budget=4)
    snaps,audit,keys=make_snapshots(fixture_stream,7,settings.base,('fusion',),'cpu')
    packets=_packets(fixture_stream,snaps,E6DSettings(base=settings.base,max_nodes=12,first_hop_limit=8),'cpu')
    return fixture_stream,settings,snaps,packets

def test_replay_cannot_accept_validation_or_eval_labels(setup):
    stream,s,shots,packs=setup
    model,_=model_init(shots[0].pt.x.shape[1],s,'structural_subgraph','cpu')
    memory=E6ReplayMemory(capacity=48,reserve_per_period=2,seed=7)
    for snap,pack in zip(shots[:5],packs[:5]):
        memory.add(model,snap,pack,mode='adaptive')
    assert len(memory.items)==48
    assert memory.coverage()==5
    assert all(e.index in set(e.snapshot.fit_idx) for e in memory.items)
    assert all(e.index not in set(e.snapshot.val_idx) for e in memory.items)
    assert all(e.index not in set(e.snapshot.pt.task.eval_idx) for e in memory.items)
    before=set((e.snapshot.ti,e.index) for e in memory.items)
    _=memory.sample(16,'adaptive',shots[5].ti)
    assert set((e.snapshot.ti,e.index) for e in memory.items)==before

def test_pre_action_state_and_probe_have_no_eval_labels(setup):
    stream,s,shots,packs=setup
    model,_=model_init(shots[0].pt.x.shape[1],s,'structural_subgraph','cpu')
    mem=E6ReplayMemory(capacity=128,seed=7)
    mem.add(model,shots[0],packs[0])
    packs_by={k.ti:v for k,v in zip(shots,packs)}
    a=probe(model,shots[1],packs[1],mem,packs_by,max_pairs=4)
    val=0.5
    first=policy_state(shots[1],stream.network,mem,val,{},a,True)
    alternate=replace(shots[1],y=shots[1].y.copy())
    alternate.y[shots[1].pt.task.eval_idx]=1-alternate.y[shots[1].pt.task.eval_idx]
    second=policy_state(alternate,stream.network,mem,val,{},a,True)
    assert np.array_equal(first,second)
    assert first.shape==(25,)

def test_full_integration_is_frozen_and_dqn_runs(setup,tmp_path):
    stream,s,_,_=setup
    out=tmp_path/'runs'
    result=run_seed(stream,7,out,s,
        policies=('uniform_full','adaptive_ewc','dqn'),
        backbones=('structural_subgraph',))
    assert len(result)==3
    pre=pd.read_csv(out/'prequential.csv')
    test=pre.query("phase == 'test'")
    assert test.groupby('policy').size().eq(2).all()
    assert set(test.model_fit_year_through.astype(str))=={'2014'}
    ret=pd.read_csv(out/'retention.csv')
    assert not ret.empty
    actions=pd.read_csv(out/'controller_decisions.csv')
    assert set(actions.policy)=={'uniform_full','adaptive_ewc','dqn'}
    assert set(actions.time.astype(str)).isdisjoint({'2015','2016'})
    assert actions.query("policy == 'dqn'").action.isin([a.name for a in E32_ACTIONS]).all()
    assert (pd.read_csv(out/'task_audit.csv').label_overlap_count==0).all()
    assert (out/'structural_subgraph_dqn_controller.pt').exists()
    weights=torch.load(out/'structural_subgraph_dqn_frozen.pt',map_location='cpu',weights_only=True)
    assert weights['fit_year_through']=='2014'

def test_changes_in_sealed_future_do_not_affect_training(setup,tmp_path):
    import copy
    stream,s,_,_=setup
    other=copy.deepcopy(stream)
    net=other.network
    # Change only future target edges; preserve chronology and node universe.
    ev=net.events.copy();mask=ev[:,0]>=8
    ev[mask,2]=(ev[mask,2]+1)%net.n
    net.events=ev[ev[:,1]!=ev[:,2]]
    out1=tmp_path/'a';out2=tmp_path/'b'
    run_seed(stream,7,out1,s,policies=('uniform_full',),backbones=('structural_subgraph',))
    run_seed(other,7,out2,s,policies=('uniform_full',),backbones=('structural_subgraph',))
    left=pd.read_csv(out1/'controller_decisions.csv');right=pd.read_csv(out2/'controller_decisions.csv')
    # Wall-clock cost varies, but training-era labels and metrics must not.
    columns=['time','action','current_val_before','current_val_after',
             'mean_forgetting','peak_forgetting','replayed_pairs']
    pd.testing.assert_frame_equal(left[columns],right[columns],check_exact=False,atol=1e-5,rtol=1e-5)

def test_gradient_projection_and_full_fusion_path(setup):
    from e6e_run_suite import train_year,consolidate
    from adaptive_graph.continual import OnlineEWC
    stream,s,snaps,packs=setup
    model,opt=model_init(snaps[0].pt.x.shape[1],s,'fusion_subgraph','cpu')
    old=snaps[0];new=snaps[1]
    memory=E6ReplayMemory(capacity=80,seed=7)
    packets={snap.ti:pack for snap,pack in zip(snaps,packs)}
    consolidate(model,old,packs[0],ewc:=OnlineEWC())
    assert ewc.fisher
    memory.add(model,old,packs[0])
    projected=next(a for a in E32_ACTIONS if a.name=='adaptive_projected')
    old_params={k:v.detach().clone() for k,v in model.state_dict().items()}
    logs,m=train_year(model,opt,new,packs[1],memory,packets,projected,ewc,s,7)
    assert m['replayed_pairs']>0
    assert m['updates']>0
    assert logs
    assert any(not torch.equal(v,old_params[k]) for k,v in model.state_dict().items())
    regularized=next(a for a in E32_ACTIONS if a.name=='adaptive_ewc')
    _,stats=train_year(model,opt,new,packs[1],memory,packets,regularized,ewc,s,7)
    assert stats['replayed_pairs']>0


def test_freeze_policy_and_symmetric_subgraph(setup,tmp_path):
    stream,s,_,_=setup
    out=tmp_path/'freeze'
    run_seed(stream,7,out,s,policies=('frozen',),backbones=('structural_subgraph',))
    scores=pd.read_csv(out/'prequential.csv')
    assert set(scores.model_fit_year_through.astype(str))=={'2010'}
    assert set(scores.query("phase=='test'").time.astype(str))=={'2015','2016'}
    assert pd.read_csv(out/'controller_decisions.csv').empty
