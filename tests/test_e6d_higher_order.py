"""E6-D causal subgraph, symmetry, gradient flow and frozen split regression."""
from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest
import torch
from adaptive_graph.data import synthetic_network,create_task
from adaptive_graph.tgat import TemporalEventStream
from adaptive_graph.e6_structural import HistoricalPairFeatures
from adaptive_graph.e6_subgraph import HistoricalEnclosingSubgraphs,RootedSubgraphEncoder
from adaptive_graph.e6_higher_order import HigherOrderStructuralTemporalFusion
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from e6c_run_suite import Settings,make_snapshots
from e6d_run_suite import E6DSettings,run_seed,_packets


@pytest.fixture
def stream(tmp_path):
    net=synthetic_network(n=68,periods=10,seed=37)
    net.times=list(range(2007,2017))
    path=tmp_path/'events.csv'
    pd.DataFrame([(int(u),int(v),int(net.times[t])) for t,u,v in net.events],
                 columns=['source','target','timestamp']).to_csv(path,index=False)
    return TemporalEventStream.from_csv(path,bucket='year',undirected=True)


def test_subgraph_reversal_and_bounded_features(stream):
    historical=HistoricalPairFeatures(stream.network,4)
    enc=HistoricalEnclosingSubgraphs(historical,max_nodes=18,first_hop_limit=10)
    p=np.array([[1,2],[2,1],[3,5]],dtype=np.int64)
    batch=enc.transform(p)
    assert batch.nodes.shape==(3,18,8)
    assert batch.weighted_adj.shape==(3,18,18)
    assert torch.equal(batch.nodes[0],batch.nodes[1])
    assert torch.equal(batch.weighted_adj[0],batch.weighted_adj[1])
    assert torch.allclose(batch.weighted_adj[0].sum(1)[batch.mask[0]],torch.ones(int(batch.mask[0].sum())))
    assert torch.isfinite(batch.nodes).all()


def test_gradients_reach_all_modalities(stream):
    task=create_task(stream.network,4,seed=7,max_positives=20)
    pt=stream.prepare(task,neighbors=4)
    hist=HistoricalPairFeatures(stream.network,4)
    b=HistoricalEnclosingSubgraphs(hist,max_nodes=18).transform(task.candidates)
    x=torch.tensor(hist.transform(task.candidates))
    model=HigherOrderStructuralTemporalFusion(pt.x.shape[1],hidden=16,heads=4,fusion_dim=16,dropout=0)
    y=model(pt,x,torch.zeros(len(task.labels)),b,ids=task.train_idx)
    torch.nn.functional.binary_cross_entropy_with_logits(y,pt.labels[task.train_idx]).backward()
    assert torch.isfinite(y).all()
    assert model.subgraph.project.weight.grad.abs().sum()>0
    assert model.base.struct_encoder.in_proj.weight.grad.abs().sum()>0
    assert model.base.temporal.node_proj.weight.grad.abs().sum()>0
    assert model.trimodal_gate[0].weight.grad.abs().sum()>0


def test_subgraph_is_history_only(stream):
    import copy
    original=stream.network
    new=copy.deepcopy(original)
    new.events=new.events.copy()
    later=new.events[:,0]>=7
    new.events[later,2]=(new.events[later,2]+3)%new.n
    new.events=new.events[new.events[:,1]!=new.events[:,2]]
    # Graph constructed before index 7 cannot inspect mutated t>=7 events.
    before=HistoricalEnclosingSubgraphs(HistoricalPairFeatures(original,6),max_nodes=16)
    after=HistoricalEnclosingSubgraphs(HistoricalPairFeatures(new,6),max_nodes=16)
    p=np.array([[1,2],[2,4],[5,8]],dtype=np.int64)
    b1,b2=before.transform(p),after.transform(p)
    assert torch.equal(b1.nodes,b2.nodes)
    assert torch.equal(b1.weighted_adj,b2.weighted_adj)
    assert torch.equal(b1.static_adj,b2.static_adj)


def test_exclude_target_period_edges(stream):
    net=stream.network
    hist=HistoricalPairFeatures(net,4)
    assert hist.historical_event_count==int((net.events[:,0]<4).sum())
    graph=HistoricalEnclosingSubgraphs(hist,max_nodes=16)
    task=create_task(net,4,seed=7,max_positives=10)
    result=graph.transform(task.candidates)
    assert torch.isfinite(result.weighted_adj).all()


def test_complete_training_and_causal_freeze(stream,tmp_path):
    torch.set_num_threads(2)
    settings=E6DSettings(base=Settings(epochs=1,patience=1,neighbors=4,hidden=16,
                              fusion_dim=16,replay_years=2,max_replay_pairs=8),
                          max_nodes=16,first_hop_limit=10)
    out=tmp_path/'out'
    run_seed(stream,7,out,settings,modes=('fusion_subgraph','subgraph_only'))
    summary=pd.read_csv(out/'model_summary.csv')
    assert set(summary.model)=={'fusion_subgraph','subgraph_only','logistic_anchor'}
    per=pd.read_csv(out/'prequential.csv')
    assert per.query("phase == 'test'").groupby('model').size().eq(2).all()
    assert set(per.query("phase == 'test'").model_fit_year_through.astype(str))=={'2014'}
    ckpt=torch.load(out/'fusion_subgraph_frozen.pt',map_location='cpu',weights_only=True)
    assert ckpt['fit_year_through']=='2014'
    assert (pd.read_csv(out/'task_audit.csv').label_overlap_count==0).all()
    assert torch.isfinite(torch.stack([v.abs().max() for v in ckpt['model'].values()])).all()


def test_bad_inputs(stream):
    with pytest.raises(ValueError):HigherOrderStructuralTemporalFusion(10,mode='hgtlp_exact')
    hist=HistoricalPairFeatures(stream.network,4)
    with pytest.raises(ValueError):HistoricalEnclosingSubgraphs(hist,max_nodes=3)
    with pytest.raises(ValueError):HistoricalEnclosingSubgraphs(hist).transform([[2,2]])
