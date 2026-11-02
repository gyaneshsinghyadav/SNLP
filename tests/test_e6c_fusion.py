"""E6-C tests: both modalities learn, symmetric scores, causal features & priors."""
import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from adaptive_graph.data import synthetic_network, TemporalNetwork, create_task
from adaptive_graph.tgat import TemporalEventStream
from adaptive_graph.e6_structural import HistoricalPairFeatures, FEATURE_NAMES
from adaptive_graph.e6_fusion import FullStructuralTemporalFusion

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from e6c_run_suite import Settings, _prior_logits, make_snapshots, run_seed


@pytest.fixture
def csv(tmp_path):
    net=synthetic_network(n=78,periods=10,seed=19)
    net.times=list(range(2007,2017))
    frame=pd.DataFrame([(int(u),int(v),int(net.times[t])) for t,u,v in net.events],
                        columns=['source','target','timestamp'])
    path=tmp_path/'history.csv';frame.to_csv(path,index=False)
    return path


def test_fusion_is_trainable_in_both_branches(csv):
    stream=TemporalEventStream.from_csv(csv,bucket='year',undirected=True)
    task=create_task(stream.network,4,seed=7,max_positives=20)
    pt=stream.prepare(task,neighbors=4)
    features=torch.tensor(HistoricalPairFeatures(stream.network,4).transform(task.candidates))
    model=FullStructuralTemporalFusion(pt.x.shape[1],hidden=16,heads=4,fusion_dim=16,dropout=0)
    output=model(pt,features,ids=task.train_idx)
    target=pt.labels[task.train_idx]
    torch.nn.functional.binary_cross_entropy_with_logits(output,target).backward()
    assert torch.isfinite(output).all()
    assert model.struct_encoder.in_proj.weight.grad.abs().sum()>0
    assert model.temporal.node_proj.weight.grad.abs().sum()>0
    assert model.feature_gate[0].weight.grad.abs().sum()>0


def test_undirected_pair_reverse_has_identical_predictions(csv):
    s=TemporalEventStream.from_csv(csv,bucket='year',undirected=True)
    pt=s.prepare(create_task(s.network,4,seed=7,max_positives=20),neighbors=4)
    model=FullStructuralTemporalFusion(pt.x.shape[1],hidden=16,heads=4,dropout=0)
    model.eval()
    x=HistoricalPairFeatures(s.network,4).transform(pt.task.candidates)
    pair=pt.candidates.clone()
    reverse=pair.flip(1)
    from dataclasses import replace
    alternate=replace(pt,candidates=reverse)
    x_reverse=HistoricalPairFeatures(s.network,4).transform(reverse.numpy())
    with torch.no_grad():
        a=model(pt,torch.as_tensor(x))
        b=model(alternate,torch.as_tensor(x_reverse))
    assert torch.allclose(a,b,atol=1e-5)


def test_anchor_uses_only_previous_revealed_fit_labels(csv):
    s=TemporalEventStream.from_csv(csv,bucket='year',undirected=True)
    snapshots,audit,keys=make_snapshots(s,7,Settings(neighbors=3),('fusion','fusion_no_recency'),'cpu')
    assert set(audit.label_overlap_count)=={0}
    assert set(audit.historical_candidate_overlap)=={0}
    assert torch.allclose(snapshots[0].prior['fusion'],torch.zeros_like(snapshots[0].prior['fusion']))
    # Prior is not allowed to fit on labels in the same target year; changing
    # that year's labels cannot alter its already-computed feature or prior.
    all_same=[]
    for snap in snapshots:
        prior=snap.prior['fusion']
        assert torch.isfinite(prior).all()
        all_same.append(prior.shape==(len(snap.y),))
    assert all(all_same)
    assert len(keys)==len(s.network.times)-Settings.start_time_index
    assert all(snap.fit_idx.size==0 for snap in snapshots[-2:])


def test_sealed_future_events_do_not_change_train_causal_inputs(csv,tmp_path):
    frame=pd.read_csv(csv)
    # Only replace events for final two years; do not modify node set.
    earlier=frame[frame.timestamp<2015]
    late=frame[frame.timestamp>=2015].copy()
    late['target']=np.roll(late['target'].to_numpy(),3)
    late=late[late.source!=late.target]
    changed=tmp_path/'mutated.csv'
    pd.concat([earlier,late],ignore_index=True).to_csv(changed,index=False)
    original=TemporalEventStream.from_csv(csv,bucket='year',undirected=True)
    mutation=TemporalEventStream.from_csv(changed,bucket='year',undirected=True)
    before,_,_=make_snapshots(original,7,Settings(neighbors=4),('fusion',),'cpu')
    after,_,_=make_snapshots(mutation,7,Settings(neighbors=4),('fusion',),'cpu')
    # Factorization must not change; candidate generation has same prior graph.
    assert original.network.n == mutation.network.n
    for a,b in zip(before[:-2],after[:-2]):
        assert a.year==b.year
        assert np.array_equal(a.pt.candidates.numpy(),b.pt.candidates.numpy())
        assert np.array_equal(a.y,b.y)
        assert torch.equal(a.structural,b.structural)
        assert torch.equal(a.prior['fusion'],b.prior['fusion'])
        assert torch.equal(a.pt.neighbor_nodes,b.pt.neighbor_nodes)
        assert torch.equal(a.pt.valid,b.pt.valid)
        assert torch.allclose(a.pt.x,b.pt.x)


def test_no_recency_branch_masks_last_four_inputs(csv):
    s=TemporalEventStream.from_csv(csv,bucket='year',undirected=True)
    snap=make_snapshots(s,7,Settings(neighbors=3),('fusion','fusion_no_recency'),'cpu')[0][2]
    model=FullStructuralTemporalFusion(snap.pt.x.shape[1],hidden=16,heads=4,
                                       fusion_dim=12,dropout=0,mode='fusion_no_recency')
    model.eval()
    with torch.no_grad():
        p1=model(snap.pt,snap.structural,snap.prior['fusion_no_recency'])
        alt=snap.structural.clone();alt[:,8:]+=9000.
        p2=model(snap.pt,alt,snap.prior['fusion_no_recency'])
    assert torch.allclose(p1,p2)


def test_complete_training_freezes_final_test_years(csv,tmp_path):
    stream=TemporalEventStream.from_csv(csv,bucket='year',undirected=True)
    torch.set_num_threads(2)
    settings=Settings(epochs=1,patience=1,neighbors=4,hidden=16,fusion_dim=16,
                      replay_years=2,max_replay_pairs=8)
    dest=tmp_path/'result'
    run_seed(stream,7,dest,settings,modes=('fusion','temporal','structural'))
    stats=pd.read_csv(dest/'model_summary.csv')
    assert set(stats.model)=={'logistic_anchor','fusion','temporal','structural'}
    audit=pd.read_csv(dest/'task_audit.csv')
    assert (audit.label_overlap_count==0).all()
    assert (audit.historical_candidate_overlap==0).all()
    pred=pd.read_csv(dest/'prequential.csv')
    assert pred.query("phase == 'test'").groupby('model').size().eq(2).all()
    assert set(pred.query("phase == 'test'").model_fit_year_through.astype(str))=={'2014'}
    ckpt=torch.load(dest/'fusion_frozen.pt',map_location='cpu',weights_only=True)
    assert ckpt['fit_year_through']=='2014'
    assert all(torch.isfinite(v).all() for v in ckpt['model'].values())


def test_input_contract_and_modes():
    with pytest.raises(ValueError):FullStructuralTemporalFusion(5,mode='wrong')
    assert len(FEATURE_NAMES)==12
    assert np.allclose(_prior_logits([],[],np.ones((4,12),dtype=np.float32),7,'fusion'),0)
