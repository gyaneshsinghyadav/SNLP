"""E6-D trimodal decoder: E6-C temporal/structural fusion plus rooted 2-hop GNN.

Prequential LR anchoring and historical candidate generation are unchanged.
This is a SEAL/HGTLP-inspired architectural experiment, not an exact reproduction.
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .e6_fusion import FullStructuralTemporalFusion
from .e6_subgraph import RootedSubgraphEncoder, SubgraphBatch

MODES=('fusion_subgraph','subgraph_only','fusion_subgraph_static','structural_subgraph')


class HigherOrderStructuralTemporalFusion(nn.Module):
    def __init__(self,input_dim,hidden=32,heads=4,fusion_dim=48,dropout=.1,
                 mode='fusion_subgraph'):
        super().__init__()
        if mode not in MODES:
            raise ValueError(f'Unknown E6-D mode: {mode}')
        self.mode=mode
        self.base=FullStructuralTemporalFusion(input_dim,hidden=hidden,heads=heads,
                                               fusion_dim=fusion_dim,dropout=dropout,
                                               mode='fusion')
        self.subgraph=RootedSubgraphEncoder(hidden=fusion_dim,dropout=dropout)
        self.trimodal_gate=nn.Sequential(nn.Linear(3*fusion_dim,3*fusion_dim),nn.GELU(),
                                         nn.Linear(3*fusion_dim,3))
        self.mixer=nn.Sequential(nn.Linear(6*fusion_dim,fusion_dim),
                                nn.LayerNorm(fusion_dim),nn.GELU(),nn.Dropout(dropout),
                                nn.Linear(fusion_dim,fusion_dim),nn.GELU())
        self.delta=nn.Linear(fusion_dim,1)
        self.subgraph_head=nn.Linear(fusion_dim,1)
        # Conservative addition to the already strong E6-C anchor.
        self.log_scale=nn.Parameter(torch.tensor(-3.2))
        nn.init.normal_(self.delta.weight,std=.01)
        nn.init.zeros_(self.delta.bias)

    def forward(self,prepared,structural_features,prior_logits,
                subgraphs:SubgraphBatch,ids=None,return_parts=False):
        if ids is None:
            ids=torch.arange(len(prepared.labels),device=prepared.x.device)
        elif not torch.is_tensor(ids):
            ids=torch.as_tensor(ids,dtype=torch.long,device=prepared.x.device)
        if len(structural_features)!=len(prepared.labels) or len(subgraphs.nodes)!=len(prepared.labels):
            raise ValueError('Candidate-aligned input arrays required')
        if not torch.isfinite(structural_features).all():
            raise ValueError('Structural input contains non-finite values')
        g=self.subgraph(subgraphs,ids,unweighted=self.mode=='fusion_subgraph_static')
        if self.mode=='subgraph_only':
            logits=self.subgraph_head(g).flatten()
            return (logits,{'scale':torch.zeros((),device=logits.device),
                            'delta':logits,'subgraph_norm':g.norm(dim=-1).mean()}) if return_parts else logits
        s=self.base.struct_encoder(structural_features[ids])
        if self.mode=='structural_subgraph':
            t=torch.zeros_like(s)
            anchor=prior_logits[ids]
            base=anchor+torch.sigmoid(self.base.residual_log_scale)*self.base.struct_head(s).flatten()
        else:
            t=self.base.temporal_encoder(self.base.temporal.states(prepared,ids))
            gate=self.base.feature_gate(torch.cat((s,t),-1))
            combined=torch.cat(((1-gate)*s,gate*t,s*t,(s-t).abs()),-1)
            base=prior_logits[ids]+torch.sigmoid(self.base.residual_log_scale)*self.base.residual_head(
                self.base.cross(combined)).flatten()
        mix=torch.softmax(self.trimodal_gate(torch.cat((s,t,g),-1)),dim=-1)
        ms=mix[:,0:1]*s;mt=mix[:,1:2]*t;mg=mix[:,2:3]*g
        z=self.mixer(torch.cat((ms,mt,mg,s*g,t*g,(s-g).abs()),-1))
        delta=self.delta(z).flatten()
        scale=torch.sigmoid(self.log_scale)
        output=base+scale*delta
        if return_parts:
            return output,{'scale':scale,'delta':delta,'gate_struct':mix[:,0].mean(),
                           'gate_temporal':mix[:,1].mean(),'gate_subgraph':mix[:,2].mean(),
                           'subgraph_norm':g.norm(dim=-1).mean()}
        return output
