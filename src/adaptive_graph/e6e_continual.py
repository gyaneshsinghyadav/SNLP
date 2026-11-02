"""E6-E: bounded history-only replay and policy observations for E6-D predictors.

Every retained index belongs to a historical fit subset. No sealed evaluation
or held-out controller-validation labels can enter optimizer/replay/EWC losses.
"""
from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn.functional as F

@dataclass
class Item:
    snapshot: object
    index: int
    label: int
    difficulty: float
    priority: float
    tie_breaker: float

class E6ReplayMemory:
    def __init__(self, capacity=256, reserve_per_period=4, seed=7):
        if capacity < 2 or not 0 <= reserve_per_period <= capacity:
            raise ValueError('Invalid replay memory capacity or historical reserve')
        self.capacity=int(capacity)
        self.reserve=int(reserve_per_period)
        self.items=[]
        self.rng=np.random.default_rng(seed+3391)
        self.seen=0
        self.replayed_pairs=0

    def add(self, model, snap, packet, mode='adaptive'):
        if mode not in ('uniform','adaptive'):
            raise ValueError('Unknown memory admission mode')
        ids=np.asarray(snap.fit_idx,dtype=np.int64)
        if not len(ids):return
        model.eval()
        for part in np.array_split(ids,max(1,(len(ids)+127)//128)):
            with torch.no_grad():
                logits=model(snap.pt,snap.structural,snap.prior['fusion'],packet,ids=part)
                labels=torch.tensor(snap.y[part],dtype=torch.float32,device=logits.device)
                losses=F.binary_cross_entropy_with_logits(logits,labels,reduction='none').detach().cpu().numpy()
            for idx,loss in zip(part,losses):
                self.seen+=1
                self.items.append(Item(snap,int(idx),int(snap.y[idx]),float(loss),
                                       float(np.clip(loss,0,20)),float(self.rng.random())))
        if len(self.items)>self.capacity:self._trim(mode)
        assert len(self.items)<=self.capacity
        assert all(e.index in set(e.snapshot.fit_idx.tolist()) for e in self.items)

    def _trim(self,mode):
        groups=defaultdict(list)
        for item in self.items:groups[item.snapshot.ti].append(item)
        # Reserve a small, equal number from every historical period, then fill
        # remaining capacity by uniform reservoir keys or training loss priority.
        keep=[];rest=[]
        periods=sorted(groups)
        if len(periods)*self.reserve>self.capacity:
            raise ValueError('Memory capacity cannot cover all historical periods')
        def key(e):return (e.priority if mode=='adaptive' else e.tie_breaker,e.tie_breaker)
        for year in periods:
            ranked=sorted(groups[year],key=key,reverse=True)
            keep+=ranked[:self.reserve];rest+=ranked[self.reserve:]
        keep+=sorted(rest,key=key,reverse=True)[:self.capacity-len(keep)]
        self.items=keep

    def sample(self, budget, mode, now):
        if budget<=0 or not self.items:return []
        if mode not in ('uniform','adaptive'):
            raise ValueError('Unknown replay selection mode')
        n=min(int(budget),len(self.items))
        if mode=='uniform':
            selected=self.rng.choice(len(self.items),size=n,replace=False)
        else:
            difficulty=np.array([e.difficulty for e in self.items])
            recency=np.exp(-np.log(2)*np.array([max(0,now-e.snapshot.ti) for e in self.items])/3.)
            score=.75*np.clip(difficulty,0,10)+.25*recency
            score=np.maximum(score,1e-5);score=score/score.sum()
            selected=self.rng.choice(len(self.items),size=n,replace=False,p=score)
        self.replayed_pairs+=n
        groups=defaultdict(list);snaps={}
        for i in np.atleast_1d(selected):
            e=self.items[int(i)]
            groups[e.snapshot.ti].append(e.index)
            snaps[e.snapshot.ti]=e.snapshot
        return [(snaps[ti],np.array(sorted(indices),dtype=np.int64))
                for ti,indices in sorted(groups.items())]

    def coverage(self):return len({x.snapshot.ti for x in self.items})
    def positive_share(self):return float(np.mean([x.label for x in self.items])) if self.items else 0.


def memory_loss(model,groups,packets):
    numerator=None;count=0
    for snap,ids in groups:
        logits=model(snap.pt,snap.structural,snap.prior['fusion'],packets[snap.ti],ids=ids)
        target=torch.tensor(snap.y[ids],device=logits.device,dtype=torch.float32)
        val=F.binary_cross_entropy_with_logits(logits,target,reduction='sum')
        numerator=val if numerator is None else numerator+val
        count+=len(ids)
    return numerator/max(1,count) if count else None


def policy_state(snap,network,memory,current_val_ap,previous,probe,ewc_ready):
    """25 history-only or currently revealed fit/validation inputs.

    Index 8 is last historical forgetting (E3.2 retention guard),
    22 is pre-action gradient cosine, 24 is gradient-probe availability.
    Never uses task.eval_idx labels or target-year link totals.
    """
    x=snap.structural.detach().cpu().numpy()[snap.fit_idx]
    y=snap.y[snap.fit_idx]
    hist=np.asarray(snap.pt.task.historical_edges).reshape(-1,2)
    degree=np.bincount(hist.ravel(),minlength=network.n) if len(hist) else np.zeros(network.n)
    previous=previous or {}
    v=lambda k:float(previous.get(k,0.))
    arr=np.array([
        snap.ti/max(1,len(network.times)-1),
        np.log1p(len(hist))/17.,
        float(np.count_nonzero(degree))/max(1,network.n),
        float(np.mean(y)),
        np.log1p(len(y))/8.,
        np.log1p(np.maximum(x[:,0],0)).mean()/8.,
        np.log1p(np.maximum(x[:,8],0)).mean()/8.,
        float(current_val_ap),
        v('mean_forgetting'),
        len(memory.items)/max(1,memory.capacity),
        memory.coverage()/max(1,len(network.times)),
        v('cost')/30.,
        v('val_gain'),
        float(ewc_ready),
        memory.positive_share(),
        float(np.std(x[:,0]))/12.,
        np.log1p(degree[degree>0]).mean()/10. if degree.any() else 0.,
        v('peak_forgetting'),
        v('mean_forgetting'),
        v('historical_ap'),
        len(memory.items)/256.,
        v('action_index')/5.,
        float(probe.get('cosine',0)),
        float(np.clip(probe.get('norm_ratio',0),0,10))/10.,
        float(probe.get('available',0)),
    ],dtype=np.float32)
    if arr.shape!=(25,) or not np.isfinite(arr).all():
        raise ValueError('Invalid continual controller state')
    return arr
