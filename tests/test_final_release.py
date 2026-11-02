"""Checks tied to bundled real DBLP E4-on-E3 artifacts and supported facade."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from rc_dclp import SEEDS,archive_dir,load_checkpoint,verify_saved
from adaptive_graph.e6_higher_order import HigherOrderStructuralTemporalFusion
from e6c_run_suite import Settings
from test_e6f_forgetting import snapshot_setup


def test_five_saved_checkpoints_are_exclusively_e4_on_e3():
    verify_saved()
    for seed in SEEDS:
        path=archive_dir(seed)
        ck=load_checkpoint(path/'structural_subgraph_dqn_frozen.pt')
        ctrl=torch.load(path/'structural_subgraph_dqn_controller.pt',weights_only=True,map_location='cpu')
        assert ctrl['state_size']==25 and len(ctrl['actions'])==6
        assert ck['settings']['variant']=='e3'
        assert ck['settings']['checkpoint_mode']=='retention'
        assert ck['settings']['distill_alpha']==0.1
        assert ck['settings']['retention_tolerance']==0.02
        model=HigherOrderStructuralTemporalFusion(6,hidden=ck['settings']['base']['hidden'],
            heads=ck['settings']['base']['heads'],fusion_dim=ck['settings']['base']['fusion_dim'],
            dropout=ck['settings']['base']['dropout'],mode='structural_subgraph')
        model.load_state_dict(ck['model'],strict=True)
        assert all(torch.isfinite(x).all() for x in model.state_dict().values())


def test_recorded_metrics_and_historical_reference():
    d=ROOT/'recorded'/'e6g_checkpoint_analysis'/'five_seed_summary.csv'
    summary=pd.read_csv(d)
    e3=summary.query("variant=='e3_control'").iloc[0]
    e4=summary.query("variant=='e4_retention'").iloc[0]
    assert abs(e4.test_mean_ap_mean-.90529)<.0002
    assert abs(e4.mean_forgetting_mean-.00450)<.0002
    assert e4.mean_forgetting_mean<e3.mean_forgetting_mean
    assert (e3.test_mean_ap_mean-e4.test_mean_ap_mean)<.001
    assert len(SEEDS)==5


def test_archived_weights_run_real_forward_path_on_synthetic_history(snapshot_setup):
    """Execute archived frozen weights through the actual structural GNN/anchor path."""
    _,_,shots,packets=snapshot_setup
    ck=load_checkpoint(archive_dir(7)/'structural_subgraph_dqn_frozen.pt')
    base=Settings(**ck['settings']['base'])
    model=HigherOrderStructuralTemporalFusion(shots[0].pt.x.shape[1],
        hidden=base.hidden,heads=base.heads,fusion_dim=base.fusion_dim,
        dropout=base.dropout,mode='structural_subgraph')
    model.load_state_dict(ck['model'],strict=True)
    model.eval()
    with torch.no_grad():
        logits=model(shots[0].pt,shots[0].structural,shots[0].prior['fusion'],
                     packets[0],ids=np.arange(4))
    assert logits.shape==(4,)
    assert torch.isfinite(logits).all()
    assert ((logits.sigmoid()>=0)&(logits.sigmoid()<=1)).all()
