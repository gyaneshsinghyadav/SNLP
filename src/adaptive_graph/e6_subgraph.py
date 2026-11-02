"""E6-D: causal, rooted two-hop enclosing-subgraph tensors for DBLP.

No candidate labels or events at/after ``prediction_index`` enter extraction.
Uses a deterministic bounded 2-hop induced subgraph, not a reproduction of SEAL.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .e6_structural import HistoricalPairFeatures

NODE_FEATURE_DIM = 8


@dataclass
class SubgraphBatch:
    nodes: torch.Tensor  # [candidate, max_nodes, 8]
    weighted_adj: torch.Tensor  # [candidate, max_nodes, max_nodes]
    static_adj: torch.Tensor
    mask: torch.Tensor  # Boolean valid node mask

    def on(self, device: str):
        return SubgraphBatch(self.nodes.to(device), self.weighted_adj.to(device),
                             self.static_adj.to(device), self.mask.to(device))


def _neighbors(csr, u: int) -> np.ndarray:
    return csr.indices[csr.indptr[u]:csr.indptr[u + 1]]


def _distances(a, start: int, max_hops=3):
    # Graph is historical only; shortest paths capped at 3 hops.
    distances = {start: 0}
    frontier = [start]
    for hop in range(1, max_hops + 1):
        next_frontier=[]
        for node in frontier:
            for nbr in _neighbors(a, node):
                j=int(nbr)
                if j not in distances:
                    distances[j]=hop
                    next_frontier.append(j)
        frontier=next_frontier
        if not frontier:
            break
    return distances


class HistoricalEnclosingSubgraphs:
    """Induced two-hop neighborhoods, bounded deterministically by historical relevance.

    Candidate endpoints are canonicalized, so reversing an undirected pair
    yields identical tensors. The target edge is not injected into the graph.
    """

    def __init__(self, historical: HistoricalPairFeatures, max_nodes=32,
                 first_hop_limit=20):
        if max_nodes < 4 or first_hop_limit < 2:
            raise ValueError('Need max_nodes>=4 and first_hop_limit>=2')
        self.hist=historical
        self.max_nodes=int(max_nodes)
        self.first_hop_limit=int(first_hop_limit)
        self.a=historical.a
        self.w=historical.w
        self.degree=historical.degree
        self.recent_degree=historical.recent_degree

    def _one(self,u,v):
        u,v=sorted((int(u),int(v)))
        if u==v:
            raise ValueError('Self candidate cannot form a rooted enclosing graph')
        au=set(map(int,_neighbors(self.a,u)))
        av=set(map(int,_neighbors(self.a,v)))
        roots={u,v}
        # Prioritize shared neighbors; ties break consistently by historical strength.
        def score(n):
            w_u=float(self.w[u,n]); w_v=float(self.w[v,n])
            return (int(n in au and n in av), w_u+w_v,
                    float(self.recent_degree[n]),-int(n))
        first=sorted((au|av)-roots,key=score,reverse=True)[:min(self.first_hop_limit,self.max_nodes-2)]
        chosen=[u,v]+first
        # Fill remaining slots using 2-hop neighbors reached through first-hop nodes.
        second_counts={}
        for node in first:
            for nbr in _neighbors(self.a,node):
                n=int(nbr)
                if n not in chosen:
                    second_counts[n]=second_counts.get(n,0)+1
        second=sorted(second_counts, key=lambda n:(second_counts[n],
                      float(self.recent_degree[n]),-n),reverse=True)
        chosen.extend(second[:self.max_nodes-len(chosen)])
        count=len(chosen)
        a=self.a[chosen,:][:,chosen].toarray().astype(np.float32)
        w=self.w[chosen,:][:,chosen].toarray().astype(np.float32)
        # Root distances on the induced subgraph (max 3; unreachable=4),
        # sorted distance pair makes undirected endpoint order irrelevant.
        def bfs(idx):
            dist=np.full(count,4.,dtype=np.float32)
            dist[idx]=0
            frontier=[idx]
            for d in range(1,4):
                nxt=[]
                for i in frontier:
                    for j in np.flatnonzero(a[i]>0):
                        if dist[j]>d:
                            dist[j]=d
                            nxt.append(int(j))
                frontier=nxt
            return dist
        du,dv=bfs(0),bfs(1)
        mind=np.minimum(du,dv);maxd=np.maximum(du,dv)
        feat=np.zeros((self.max_nodes,NODE_FEATURE_DIM),dtype=np.float32)
        for i,n in enumerate(chosen):
            feat[i]=[
                float(i < 2), mind[i]/4., maxd[i]/4.,
                float(du[i]==1 and dv[i]==1),
                float(du[i]<=2 and dv[i]<=2),
                np.log1p(self.degree[n])/8.,
                np.log1p(self.recent_degree[n])/8.,
                min(float(a[i,0]+a[i,1]),2.)/2.,
            ]
        mask=np.zeros(self.max_nodes,dtype=bool);mask[:count]=True
        def normalize(mat):
            # Edges with nonnegative historical strength. Add self-loops only for
            # valid nodes, then row-normalize; padding never contributes.
            mat=np.maximum(mat,0.)
            mat=mat+np.eye(count,dtype=np.float32)
            mat=mat/np.maximum(mat.sum(axis=1,keepdims=True),1e-6)
            output=np.zeros((self.max_nodes,self.max_nodes),dtype=np.float32)
            output[:count,:count]=mat
            return output
        return feat,normalize(np.log1p(w)),normalize(a),mask

    def transform(self,pairs) -> SubgraphBatch:
        pairs=np.asarray(pairs,dtype=np.int64).reshape(-1,2)
        if (pairs<0).any() or (pairs>=self.a.shape[0]).any():
            raise ValueError('Candidate outside historical node universe')
        features=[]; weighted=[]; static=[]; masks=[]
        for u,v in pairs:
            f,w,a,m=self._one(int(u),int(v))
            features.append(f);weighted.append(w);static.append(a);masks.append(m)
        n=len(pairs);k=self.max_nodes
        def stack(xs,shape,dtype):
            return np.stack(xs) if xs else np.empty(shape,dtype=dtype)
        return SubgraphBatch(torch.from_numpy(stack(features,(0,k,NODE_FEATURE_DIM),np.float32)),
                    torch.from_numpy(stack(weighted,(0,k,k),np.float32)),
                    torch.from_numpy(stack(static,(0,k,k),np.float32)),
                    torch.from_numpy(stack(masks,(0,k),bool)))


class RootedSubgraphEncoder(nn.Module):
    """Trainable gated two-layer message passing over induced historical subgraphs."""
    def __init__(self,hidden=48,dropout=.1):
        super().__init__()
        self.project=nn.Linear(NODE_FEATURE_DIM,hidden)
        self.updates=nn.ModuleList([nn.Linear(2*hidden,hidden) for _ in range(2)])
        self.gates=nn.ModuleList([nn.Linear(2*hidden,hidden) for _ in range(2)])
        self.norms=nn.ModuleList([nn.LayerNorm(hidden) for _ in range(2)])
        self.attn=nn.Linear(hidden,1)
        self.pool=nn.Sequential(nn.Linear(3*hidden,hidden),nn.LayerNorm(hidden),nn.GELU())
        self.dropout=dropout

    def forward(self,batch:SubgraphBatch,ids,unweighted=False):
        x=batch.nodes[ids]
        a=(batch.static_adj if unweighted else batch.weighted_adj)[ids]
        mask=batch.mask[ids]
        h=F.gelu(self.project(x)) * mask.unsqueeze(-1)
        for update,gate,norm in zip(self.updates,self.gates,self.norms):
            messages=torch.bmm(a,h)
            joined=torch.cat((h,messages),-1)
            step=F.gelu(update(joined))
            weight=torch.sigmoid(gate(joined))
            h=norm(h+F.dropout(step*weight,self.dropout,training=self.training))
            h=h*mask.unsqueeze(-1)
        roots=(h[:,0,:]+h[:,1,:])/2
        difference=(h[:,0,:]-h[:,1,:]).abs()
        weight=self.attn(h).squeeze(-1).masked_fill(~mask,-1e4)
        pooled=(torch.softmax(weight,-1).unsqueeze(-1)*h).sum(1)
        return self.pool(torch.cat((roots,difference,pooled),-1))
