"""Sparse GCN and supervised hybrid link predictor. No torch-geometric needed."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import scipy.sparse as sp
import torch
from torch import nn
from torch.nn import functional as F
from .data import Task, undirected_csr, directed_csr
from .features import (structural_features, transform_pair_features, node_inputs,
                       directed_node_inputs, directed_structural_features)


def make_sparse_adj(a: sp.csr_matrix, device="cpu"):
    n = a.shape[0]
    m = a + sp.eye(n, dtype=np.float32, format="csr")
    deg = np.asarray(m.sum(1)).ravel()
    scale = np.power(np.maximum(deg, 1), -0.5)
    norm = sp.diags(scale) @ m @ sp.diags(scale)
    coo = norm.tocoo()
    ix = np.stack((coo.row, coo.col))
    return torch.sparse_coo_tensor(torch.tensor(ix, dtype=torch.long, device=device),
                                   torch.tensor(coo.data, dtype=torch.float32, device=device),
                                   (n, n), device=device).coalesce()


@dataclass
class PreparedTask:
    task: Task
    adj: torch.Tensor
    x: torch.Tensor
    structural: torch.Tensor
    candidates: torch.Tensor
    labels: torch.Tensor

    def subset(self, split="train"):
        ids = self.task.train_idx if split == "train" else self.task.eval_idx
        return torch.as_tensor(ids, dtype=torch.long, device=self.x.device)


def prepare(task: Task, n, original_features=None, track="history", device="cpu"):
    a = undirected_csr(task.historical_edges, n)
    if task.directed:
        directed_a = directed_csr(task.historical_edges, n)
        x = directed_node_inputs(directed_a)
        if track == "original":
            raise ValueError("Original node features are not available for the Bitcoin directed track")
        heuristic = transform_pair_features(directed_structural_features(directed_a, task.candidates))
    else:
        x = node_inputs(a, original_features=original_features, track=track)
        heuristic = transform_pair_features(structural_features(a, task.candidates))
    return PreparedTask(task, make_sparse_adj(a, device),
                        torch.as_tensor(x, device=device),
                        torch.as_tensor(heuristic, device=device),
                        torch.tensor(task.candidates, dtype=torch.long, device=device),
                        torch.tensor(task.labels, dtype=torch.float32, device=device))


class HybridGCN(nn.Module):
    def __init__(self, input_dim, hidden=32, structure_dim=9, dropout=0.1, use_gcn=True,
                 use_heuristics=True):
        super().__init__()
        self.use_gcn = use_gcn
        self.use_heuristics = use_heuristics
        self.dropout = dropout
        if use_gcn:
            self.conv1 = nn.Linear(input_dim, hidden)
            self.conv2 = nn.Linear(hidden, hidden)
        dim = (2 * hidden if use_gcn else 0) + (structure_dim if use_heuristics else 0)
        if not dim:
            raise ValueError("GCN and heuristics cannot both be disabled")
        self.state_dim = dim
        self.head = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(hidden, 1))

    def encode(self, pt: PreparedTask):
        if not self.use_gcn:
            return None
        h = F.relu(self.conv1(torch.sparse.mm(pt.adj, pt.x)))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.conv2(torch.sparse.mm(pt.adj, h))
        return h

    def states(self, pt: PreparedTask, ids: torch.Tensor | None = None, detach=False):
        if ids is None:
            ids = torch.arange(len(pt.labels), device=pt.labels.device)
        parts = []
        if self.use_gcn:
            h = self.encode(pt)
            links = pt.candidates[ids]
            hu, hv = h[links[:, 0]], h[links[:, 1]]
            parts.extend([hu, hv] if pt.task.directed else
                         [hu * hv, torch.abs(hu - hv)])
        if self.use_heuristics:
            parts.append(pt.structural[ids])
        result = torch.cat(parts, dim=1)
        return result.detach() if detach else result

    def forward(self, pt: PreparedTask, ids: torch.Tensor | None = None):
        return self.head(self.states(pt, ids)).flatten()
