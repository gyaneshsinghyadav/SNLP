"""E6-F controlled ablations: representative rehearsal and historical-logit distillation.

E0 uses the original E6ReplayMemory without modifications. This module adds:
  E1: balanced per-year/label admission + feature diversity + refreshed difficulty;
  E2: original admission with immutable teacher logits from historical checkpoints;
  E3: E1 + E2.
All teachers and admission labels derive exclusively from historical *fit* indices.
"""
from __future__ import annotations

from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F

from .e6e_continual import E6ReplayMemory


def _key(item):
    return (int(item.snapshot.ti), int(item.index))


class ReplayWithLogitsMemory(E6ReplayMemory):
    def __init__(self, *args, distill=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.distill = bool(distill)
        self.teacher_logits = {}

    def add(self, model, snap, packet, mode='adaptive'):
        super().add(model, snap, packet, mode)
        if not self.distill:
            return
        new = [e for e in self.items if e.snapshot.ti == snap.ti and _key(e) not in self.teacher_logits]
        if new:
            was_training = model.training
            model.eval()
            try:
                for chunk in np.array_split(new, max(1, (len(new)+127)//128)):
                    ids=np.array([e.index for e in chunk], dtype=np.int64)
                    with torch.no_grad():
                        outputs=model(snap.pt,snap.structural,snap.prior['fusion'],packet,ids=ids)
                    for e,value in zip(chunk, outputs.detach().cpu().tolist()):
                        self.teacher_logits[_key(e)] = float(value)
            finally:
                model.train(was_training)
        live = {_key(e) for e in self.items}
        self.teacher_logits = {k:v for k,v in self.teacher_logits.items() if k in live}
        assert len(self.teacher_logits)==len(self.items)
        assert all(np.isfinite(v) for v in self.teacher_logits.values())


class RepresentativeReplayMemory(ReplayWithLogitsMemory):
    """Balanced historical year/label storage, with structural diversity.

    Work inside fit-only indices. Uniform admission from original archive is
    *not* modified. Selection prioritizes history coverage, then difficulty.
    """
    def add(self, model, snap, packet, mode='adaptive'):
        # Refresh *difficulty only*, never the immutable teacher. Uses old fit
        # labels against each item's original historical context + LR prior.
        if self.items:
            self.refresh_difficulty(model, {snap.ti: packet})
        super().add(model, snap, packet, mode)

    def refresh_difficulty(self, model, extra_packets=None):
        extra_packets = extra_packets or {}
        by_year=defaultdict(list)
        for e in self.items: by_year[e.snapshot.ti].append(e)
        was_training=model.training
        model.eval()
        try:
            for ti,items in by_year.items():
                snap=items[0].snapshot
                packet=extra_packets.get(ti, getattr(self, '_historical_packets', {}).get(ti))
                if packet is None:
                    # Preserve the feature context using the registered archive
                    # populated after every admission.
                    raise RuntimeError(f'Missing historical subgraph packet for year {ti}')
                for chunk in np.array_split(items,max(1,(len(items)+127)//128)):
                    ids=np.array([e.index for e in chunk],dtype=np.int64)
                    with torch.no_grad():
                        logits=model(snap.pt,snap.structural,snap.prior['fusion'],packet,ids=ids)
                        labels=torch.as_tensor(snap.y[ids],dtype=torch.float32,device=logits.device)
                        losses=F.binary_cross_entropy_with_logits(logits,labels,reduction='none')
                    for e,loss in zip(chunk,losses.detach().cpu().tolist()):
                        e.difficulty=float(loss)
                        e.priority=float(np.clip(loss,0.,20.))
        finally:
            model.train(was_training)

    def set_packets(self, packets):
        self._historical_packets=packets

    @staticmethod
    def _diverse_pick(pool, n):
        if n>=len(pool): return list(pool)
        if n<=0: return []
        # Feature basis is pair-only and historical; no future-year statistics.
        x=np.stack([e.snapshot.structural[e.index].detach().cpu().numpy()
                    for e in pool],axis=0).astype(np.float64)
        x=np.log1p(np.maximum(x,0.))
        lo=x.min(axis=0);rng=np.maximum(x.max(axis=0)-lo,1e-6)
        x=np.clip((x-lo)/rng,0,1)
        diff=np.log1p(np.array([e.difficulty for e in pool],dtype=np.float64))
        diff/=max(1e-6,float(diff.max()))
        # Begin with hardest example, then favor distance from represented cases
        # without allowing difficulty to dominate all future selections.
        selected=[int(np.argmax(diff))]
        distance=np.linalg.norm(x-x[selected[0]],axis=1)
        while len(selected)<n:
            merit=.75*distance/max(1e-6,np.sqrt(x.shape[1]))+.25*diff
            merit[selected]=-np.inf
            j=int(np.argmax(merit));selected.append(j)
            distance=np.minimum(distance,np.linalg.norm(x-x[j],axis=1))
        return [pool[i] for i in selected]

    def _trim(self, mode):
        grouped=defaultdict(list)
        for e in self.items:grouped[int(e.snapshot.ti)].append(e)
        years=sorted(grouped)
        if len(years)*self.reserve>self.capacity:
            raise ValueError('Insufficient memory for per-year reserves')
        allocations={year:0 for year in years}
        # Fair water-fill to year capacity; no implicit recency bias.
        remaining=self.capacity
        while remaining:
            progress=False
            for year in years:
                if allocations[year]<len(grouped[year]) and remaining:
                    allocations[year]+=1;remaining-=1;progress=True
            if not progress:break
        result=[]
        for year in years:
            members=grouped[year];budget=allocations[year]
            pos=[e for e in members if e.label==1]
            neg=[e for e in members if e.label==0]
            # Allocate to labels alternately, spill to whichever is available.
            allocations_by_label={0:0,1:0}
            for _ in range(budget):
                eligible=[lab for lab,pool in ((0,neg),(1,pos))
                          if allocations_by_label[lab]<len(pool)]
                lab=min(eligible,key=lambda k:(allocations_by_label[k], k))
                allocations_by_label[lab]+=1
            result+=self._diverse_pick(neg,allocations_by_label[0])
            result+=self._diverse_pick(pos,allocations_by_label[1])
        self.items=result
        assert len(self.items)<=self.capacity

    def sample(self, budget, mode, now):
        if budget<=0 or not self.items:return []
        if mode not in ('adaptive','uniform'):raise ValueError('Unknown sampling mode')
        n=min(int(budget),len(self.items));groups=defaultdict(list)
        for i,e in enumerate(self.items): groups[(e.snapshot.ti,e.label)].append(i)
        chosen=[];selected=set()
        # Guarantee historical-year coverage as far as the budget permits.
        years=sorted({key[0] for key in groups})
        for year in years:
            if len(chosen)>=n:break
            for label in (0,1):
                if len(chosen)>=n:break
                candidates=[i for i in groups.get((year,label),[]) if i not in selected]
                if candidates:
                    k=self._weighted_pick(candidates,mode,now)
                    chosen.append(k);selected.add(k)
        while len(chosen)<n:
            available=[i for i in range(len(self.items)) if i not in selected]
            k=self._weighted_pick(available,mode,now)
            chosen.append(k);selected.add(k)
        self.replayed_pairs+=n
        output=defaultdict(list);snaps={}
        for i in chosen:
            e=self.items[i]
            output[e.snapshot.ti].append(e.index);snaps[e.snapshot.ti]=e.snapshot
        return [(snaps[ti],np.asarray(sorted(ids),dtype=np.int64)) for ti,ids in sorted(output.items())]

    def _weighted_pick(self,indices,mode,now):
        if mode=='uniform':return int(self.rng.choice(indices))
        arr=[self.items[i] for i in indices]
        difficulty=np.array([e.difficulty for e in arr])
        age=np.array([max(0,now-e.snapshot.ti) for e in arr])
        scores=.75*np.clip(difficulty,0,10)+.25*np.exp(-np.log(2)*age/3.)
        scores=np.maximum(scores,1e-5);scores/=scores.sum()
        return int(self.rng.choice(indices,p=scores))


def label_and_distill_loss(model,groups,packets,memory,distill_weight):
    """Fit-only rehearsal BCE and optional immutable-logit MSE, normalized per pair."""
    if not groups:return None,None
    supervised=None;distillation=None;total=0
    for snap,ids in groups:
        outputs=model(snap.pt,snap.structural,snap.prior['fusion'],packets[snap.ti],ids=ids)
        targets=torch.as_tensor(snap.y[ids],device=outputs.device,dtype=torch.float32)
        loss=F.binary_cross_entropy_with_logits(outputs,targets,reduction='sum')
        supervised=loss if supervised is None else supervised+loss
        if distill_weight>0:
            assert memory.distill, 'Distillation requires immutable teacher store'
            teacher=torch.as_tensor([memory.teacher_logits[(int(snap.ti),int(idx))] for idx in ids],
                                     device=outputs.device,dtype=outputs.dtype)
            drift=F.mse_loss(outputs,teacher,reduction='sum')
            distillation=drift if distillation is None else distillation+drift
        total+=len(ids)
    return supervised/total,(distillation/total if distillation is not None else None)
