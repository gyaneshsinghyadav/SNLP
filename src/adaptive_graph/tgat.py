"""Causal, one-hop TGAT-style encoder for directed temporal link prediction.

This is a compact TGAT-inspired temporal-neighbor attention architecture, NOT a
reproduction of the original multi-layer ICLR 2020 TGAT implementation.

All node inputs, neighbor events and edge ratings are frozen at the START of the
prediction bucket (strictly earlier timestamps). Quarter-level train labels
become visible only after the snapshot's prequential evaluation. Edge sign is
an interaction attribute, not a non-link label.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from .data import TemporalNetwork, Task, directed_csr, undirected_csr
from .features import directed_node_inputs


def _unix_seconds(values: pd.Series) -> np.ndarray:
    if pd.api.types.is_numeric_dtype(values):
        if values.between(1800, 2200).all():
            dt = pd.to_datetime(values.astype(int).astype(str), format="%Y", utc=True)
        else:
            dt = pd.to_datetime(values, unit="s", utc=True, errors="raise")
    else:
        dt = pd.to_datetime(values, utc=True, errors="raise", format="mixed")
    return (dt.astype("int64").to_numpy() / 1e9).astype(np.float64)


def _bucket_start(label, bucket: str) -> float:
    if bucket == "quarter":
        stamp = pd.Period(str(label), freq="Q").start_time
    elif bucket == "year":
        stamp = pd.Timestamp(f"{label}-01-01")
    elif bucket == "month":
        stamp = pd.Period(str(label), freq="M").start_time
    else:
        raise ValueError("TGAT supports year, quarter and month buckets")
    return float(stamp.tz_localize("UTC").timestamp())


@dataclass
class TemporalEventStream:
    network: TemporalNetwork
    # Individual interactions, including repeated ratings in one time bucket.
    timestamps: np.ndarray
    source: np.ndarray
    target: np.ndarray
    rating: np.ndarray
    bucket: str

    @classmethod
    def from_csv(cls, path, bucket="quarter", rating_column="rating", undirected=False):
        if bucket not in {"year", "quarter", "month"}:
            raise ValueError("TGAT time bucket must be year, quarter, or month")
        raw = pd.read_csv(path)
        for c in ("source", "target", "timestamp"):
            if c not in raw.columns:
                raise ValueError(f"Missing column: {c}")
        raw = raw.dropna(subset=["source", "target", "timestamp"]).copy()
        if not len(raw):
            raise ValueError("Empty temporal dataset")
        nodes = pd.Index(pd.concat([raw.source, raw.target], ignore_index=True).unique())
        mapping = {node: idx for idx, node in enumerate(nodes)}
        # Match the TemporalNetwork.from_csv node factorization exactly.
        net = TemporalNetwork.from_csv(path, time_bucket=bucket, directed=not undirected)
        if net.n != len(nodes):
            raise ValueError("Unexpected node map mismatch")
        timestamps = _unix_seconds(raw.timestamp)
        if rating_column in raw.columns:
            rating = pd.to_numeric(raw[rating_column], errors="coerce").fillna(0).to_numpy(np.float32)
            rating = np.clip(rating / 10.0, -1.0, 1.0)
        else:
            rating = np.ones(len(raw), dtype=np.float32)
        source = raw.source.map(mapping).to_numpy(np.int64)
        target = raw.target.map(mapping).to_numpy(np.int64)
        keep = source != target
        return cls(net, timestamps[keep], source[keep], target[keep], rating[keep], bucket)

    def prepare(self, task: Task, neighbors=12, device="cpu"):
        """Construct source-outgoing / target-incoming causal event neighbors."""
        if neighbors < 1:
            raise ValueError("neighbors must be >= 1")
        cutoff = _bucket_start(task.time_value, self.bucket)
        allowed = self.timestamps < cutoff
        times = self.timestamps[allowed]
        src = self.source[allowed]
        dst = self.target[allowed]
        ratings = self.rating[allowed]
        n = self.network.n
        # Use histories exclusively earlier than the current prediction cutoff.
        historical_edges = np.column_stack([src, dst]) if len(src) else np.empty((0, 2), np.int64)
        x = directed_node_inputs(undirected_csr(historical_edges, n) if not self.network.directed
                                 else directed_csr(historical_edges, n))
        neighbor_nodes = np.zeros((n, 2, neighbors), dtype=np.int64)
        age_days = np.zeros((n, 2, neighbors), dtype=np.float32)
        neighbor_ratings = np.zeros((n, 2, neighbors), dtype=np.float32)
        valid = np.zeros((n, 2, neighbors), dtype=bool)
        # Input files may not be time-sorted, so use a stable sorted order.
        order = np.argsort(times, kind="stable")
        from collections import defaultdict, deque
        histories = defaultdict(lambda: deque(maxlen=neighbors))
        for ix in order:
            s, d = int(src[ix]), int(dst[ix])
            delta = max(0.0, (cutoff - float(times[ix])) / 86400.0)
            rating = float(ratings[ix])
            histories[(s, 0)].append((d, delta, rating))  # outgoing
            histories[(d, 1)].append((s, delta, rating))  # incoming
            if not self.network.directed:
                # Both endpoints are neighbors in an undirected coauthor graph.
                histories[(d, 0)].append((s, delta, rating))
                histories[(s, 1)].append((d, delta, rating))
        for (node, role), events in histories.items():
            for j, (other, age, rating) in enumerate(reversed(events)):
                neighbor_nodes[node, role, j] = other
                age_days[node, role, j] = min(age, 36500.0)
                neighbor_ratings[node, role, j] = rating
                valid[node, role, j] = True
        return TemporalPrepared(task=task, x=torch.as_tensor(x, dtype=torch.float32, device=device),
                                candidates=torch.as_tensor(task.candidates, dtype=torch.long, device=device),
                                labels=torch.as_tensor(task.labels, dtype=torch.float32, device=device),
                                neighbor_nodes=torch.as_tensor(neighbor_nodes, dtype=torch.long, device=device),
                                age_days=torch.as_tensor(age_days, dtype=torch.float32, device=device),
                                ratings=torch.as_tensor(neighbor_ratings, dtype=torch.float32, device=device),
                                valid=torch.as_tensor(valid, dtype=torch.bool, device=device),
                                cutoff=cutoff, historical_event_count=int(allowed.sum()),
                                undirected=not self.network.directed)


@dataclass
class TemporalPrepared:
    task: Task
    x: torch.Tensor
    candidates: torch.Tensor
    labels: torch.Tensor
    neighbor_nodes: torch.Tensor
    age_days: torch.Tensor
    ratings: torch.Tensor
    valid: torch.Tensor
    cutoff: float
    historical_event_count: int
    undirected: bool = False

    def subset(self, split="train"):
        ids = self.task.train_idx if split == "train" else self.task.eval_idx
        return torch.as_tensor(ids, dtype=torch.long, device=self.x.device)


class FunctionalTimeEncoding(nn.Module):
    """Cosine relative-time embedding with trainable frequencies and phases."""
    def __init__(self, dimension):
        super().__init__()
        self.frequencies = nn.Parameter(torch.logspace(-3.5, -0.4, dimension))
        self.phase = nn.Parameter(torch.zeros(dimension))

    def forward(self, elapsed_days):
        return torch.cos(elapsed_days.unsqueeze(-1) * self.frequencies + self.phase)


class TemporalAttentionLinkPredictor(nn.Module):
    """One-hop direction-aware temporal attention + asymmetric link decoder.

    Source nodes aggregate outgoing past contacts; target nodes aggregate
    incoming past contacts. No current/future event enters those tables.
    """
    def __init__(self, input_dim, hidden=32, heads=4, dropout=0.1):
        super().__init__()
        if hidden % heads:
            raise ValueError("hidden must be divisible by attention heads")
        self.hidden = hidden
        self.heads = heads
        self.dropout = dropout
        self.node_proj = nn.Linear(input_dim, hidden)
        self.time = FunctionalTimeEncoding(hidden)
        self.query = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(2)])
        self.key = nn.ModuleList([nn.Linear(hidden * 2 + 1, hidden) for _ in range(2)])
        self.value = nn.ModuleList([nn.Linear(hidden * 2 + 1, hidden) for _ in range(2)])
        self.update = nn.ModuleList([nn.Linear(hidden * 2, hidden) for _ in range(2)])
        self.state_dim = 4 * hidden
        self.head = nn.Sequential(nn.Linear(self.state_dim, hidden), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(hidden, 1))

    def _embed(self, pt: TemporalPrepared, nodes: torch.Tensor, role: int, embedded):
        root = embedded[nodes]
        nbr = pt.neighbor_nodes[nodes, role]
        neighbor_emb = embedded[nbr]
        tenc = self.time(pt.age_days[nodes, role])
        attr = pt.ratings[nodes, role].unsqueeze(-1)
        events = torch.cat((neighbor_emb, tenc, attr), dim=-1)
        bs, k = nbr.shape
        q = self.query[role](root).reshape(bs, self.heads, -1)
        keys = self.key[role](events).reshape(bs, k, self.heads, -1)
        values = self.value[role](events).reshape(bs, k, self.heads, -1)
        logits = (q.unsqueeze(1) * keys).sum(-1) / (self.hidden // self.heads) ** 0.5
        mask = pt.valid[nodes, role].unsqueeze(-1)
        logits = logits.masked_fill(~mask, -1e4)
        weights = torch.softmax(logits, dim=1) * mask.float()
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-9)
        context = (weights.unsqueeze(-1) * values).sum(dim=1).reshape(bs, self.hidden)
        result = F.relu(self.update[role](torch.cat((root, context), dim=-1)))
        return F.dropout(result, p=self.dropout, training=self.training)

    def states(self, pt: TemporalPrepared, ids=None, detach=False):
        if ids is None:
            ids = torch.arange(len(pt.labels), device=pt.labels.device)
        pair = pt.candidates[ids]
        x = F.relu(self.node_proj(pt.x))
        source = self._embed(pt, pair[:, 0], 0, x)
        target = self._embed(pt, pair[:, 1], 1, x)
        result = torch.cat((source, target, source * target, torch.abs(source - target)), dim=-1)
        if pt.undirected:
            # Symmetric features: predicting (A,B) must equal predicting (B,A).
            reverse_source = self._embed(pt, pair[:, 1], 0, x)
            reverse_target = self._embed(pt, pair[:, 0], 1, x)
            reverse = torch.cat((reverse_source, reverse_target,
                                 reverse_source * reverse_target,
                                 torch.abs(reverse_source - reverse_target)), dim=-1)
            result = 0.5 * (result + reverse)
        return result.detach() if detach else result

    def forward(self, pt, ids=None):
        return self.head(self.states(pt, ids)).flatten()
