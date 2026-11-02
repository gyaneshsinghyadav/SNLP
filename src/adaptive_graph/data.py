"""Chronological edge events and leakage-safe directed/undirected tasks."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp


def canonical(pairs: np.ndarray) -> np.ndarray:
    a = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    return np.sort(a, axis=1)


def unique_pairs(pairs: np.ndarray, directed: bool = False) -> np.ndarray:
    a = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    return np.unique(a if directed else canonical(a), axis=0)


def pair_keys(pairs: np.ndarray, n: int, directed: bool = False) -> np.ndarray:
    p = np.asarray(pairs, dtype=np.int64).reshape(-1, 2) if directed else canonical(pairs)
    return p[:, 0] * np.int64(n) + p[:, 1]


@dataclass
class TemporalNetwork:
    """Events: (time-index, src, dst); repeated contacts across times are retained."""
    events: np.ndarray
    n: int
    times: list
    node_features: np.ndarray | None = None
    name: str = "dataset"
    directed: bool = False

    def __post_init__(self):
        e = np.asarray(self.events, dtype=np.int64).reshape(-1, 3).copy()
        if self.n < 2 or not len(e):
            raise ValueError("Network needs >=2 nodes and nonempty edge events")
        if np.any(e[:, 1:] < 0) or np.any(e[:, 1:] >= self.n):
            raise ValueError("Out-of-range node ID")
        if np.any(e[:, 0] < 0) or np.any(e[:, 0] >= len(self.times)):
            raise ValueError("Invalid time index")
        if not self.directed:
            e[:, 1:] = canonical(e[:, 1:])
        e = e[e[:, 1] != e[:, 2]]
        e = np.unique(e, axis=0)  # duplicate contacts within the same snapshot
        self.events = e[np.argsort(e[:, 0], kind="stable")]
        if self.node_features is not None:
            self.node_features = np.asarray(self.node_features, dtype=np.float32)
            if self.node_features.ndim != 2 or self.node_features.shape[0] != self.n:
                raise ValueError("node_features must have n rows")

    @classmethod
    def from_csv(cls, path: str | Path, src="source", dst="target", time="timestamp", name=None, time_bucket="none", directed=False):
        df = pd.read_csv(path)
        if any(c not in df.columns for c in (src, dst, time)):
            raise ValueError(f"CSV needs columns {src!r}, {dst!r}, {time!r}")
        df = df.dropna(subset=[src, dst, time]).copy()
        if time_bucket not in {"none", "year", "quarter", "month", "week", "day"}:
            raise ValueError("time_bucket must be none/year/quarter/month/week/day")
        if time_bucket != "none":
            values = df[time]
            if pd.api.types.is_numeric_dtype(values) and values.between(1800, 2200).all():
                datetimes = pd.to_datetime(values.astype(str), format="%Y", utc=True)
            elif pd.api.types.is_numeric_dtype(values):
                datetimes = pd.to_datetime(values, unit="s", utc=True, errors="raise")
            else:
                datetimes = pd.to_datetime(values, utc=True, errors="raise", format="mixed")
            if time_bucket == "year":
                df[time] = datetimes.dt.year
            elif time_bucket == "quarter":
                df[time] = datetimes.dt.strftime("%Y-") + "Q" + datetimes.dt.quarter.astype(str)
            elif time_bucket == "month":
                df[time] = datetimes.dt.strftime("%Y-%m")
            elif time_bucket == "week":
                df[time] = datetimes.dt.strftime("%G-W%V")
            else:
                df[time] = datetimes.dt.strftime("%Y-%m-%d")
        # Consistent factorization for both columns, including string node IDs.
        nodes = pd.Index(pd.concat([df[src], df[dst]], ignore_index=True).unique())
        lookup = {v: i for i, v in enumerate(nodes)}
        times = sorted(df[time].unique().tolist())
        tlookup = {t: i for i, t in enumerate(times)}
        events = np.column_stack((df[time].map(tlookup).to_numpy(dtype=np.int64),
                                  df[src].map(lookup).to_numpy(dtype=np.int64),
                                  df[dst].map(lookup).to_numpy(dtype=np.int64)))
        return cls(events, len(nodes), times, name=name or Path(path).stem, directed=directed)

    def induced_subset(self, max_nodes: int | None, seed=7, selection_history_idx=1):
        """Degree-seeded BFS using ONLY history prior to selection_history_idx.

        The evaluated task indices must be >= selection_history_idx to avoid
        future-informed sampling. This is exploratory, biased subgraph sampling.
        """
        if not max_nodes or max_nodes >= self.n:
            return self
        rng = np.random.default_rng(seed)
        if selection_history_idx < 1 or selection_history_idx >= len(self.times):
            raise ValueError("selection_history_idx must be >=1 and < number of times")
        # No target or future edges enter either the subset sampling or features.
        a = undirected_csr(self.get_history(selection_history_idx), self.n)
        deg = np.diff(a.indptr)
        seed_node = int(np.argmax(deg))
        selected = {seed_node}
        frontier = [seed_node]
        while len(selected) < max_nodes:
            if not frontier:
                rest = np.setdiff1d(np.arange(self.n), list(selected))
                if not len(rest):
                    break
                frontier.append(int(rng.choice(rest)))
            u = frontier.pop(0)
            neighbors = a.indices[a.indptr[u]:a.indptr[u+1]].copy()
            rng.shuffle(neighbors)
            for v in neighbors:
                if int(v) not in selected:
                    selected.add(int(v))
                    frontier.append(int(v))
                    if len(selected) >= max_nodes:
                        break
        ids = np.array(sorted(selected), dtype=np.int64)
        remap = np.full(self.n, -1, dtype=np.int64)
        remap[ids] = np.arange(len(ids))
        ev = self.events.copy()
        ev[:, 1:] = remap[ev[:, 1:]]
        ev = ev[np.all(ev[:, 1:] >= 0, axis=1)]
        if not len(ev):
            raise ValueError("Node subset has no edges: choose larger --max-nodes")
        features = self.node_features[ids] if self.node_features is not None else None
        return TemporalNetwork(ev, len(ids), self.times, features, self.name + "_subset", self.directed)

    def get_history(self, target_time_idx: int) -> np.ndarray:
        """ONLY edges strictly earlier than target time. No future data consulted."""
        return unique_pairs(self.events[self.events[:, 0] < target_time_idx, 1:], directed=self.directed)

    def get_current(self, target_time_idx: int) -> np.ndarray:
        return unique_pairs(self.events[self.events[:, 0] == target_time_idx, 1:], directed=self.directed)


def directed_csr(edges: np.ndarray, n: int) -> sp.csr_matrix:
    if not len(edges):
        return sp.csr_matrix((n, n), dtype=np.float32)
    a = sp.coo_matrix((np.ones(len(edges), np.float32),
                       (edges[:, 0], edges[:, 1])), shape=(n, n)).tocsr()
    a.setdiag(0); a.eliminate_zeros(); a.data[:] = 1
    return a


def undirected_csr(edges: np.ndarray, n: int) -> sp.csr_matrix:
    if len(edges) == 0:
        return sp.csr_matrix((n, n), dtype=np.float32)
    u, v = edges[:, 0], edges[:, 1]
    a = sp.coo_matrix((np.ones(2 * len(edges), np.float32),
                       (np.r_[u, v], np.r_[v, u])), shape=(n, n)).tocsr()
    a.setdiag(0)
    a.eliminate_zeros()
    a.data[:] = 1
    return a


@dataclass
class Task:
    time_index: int
    time_value: object
    historical_edges: np.ndarray
    candidates: np.ndarray
    labels: np.ndarray
    train_idx: np.ndarray
    eval_idx: np.ndarray
    history_node_count: int
    new_link_count: int
    repeat_contact_count: int
    cold_start_count: int
    directed: bool = False


def create_task(net: TemporalNetwork, time_index: int, seed=7,
                max_positives=4000, negative_ratio=1.0, eval_fraction=0.3) -> Task:
    """Candidate construction only references past and target truth, not later years.

    Labels at target time are used to create an *offline evaluation dataset*, never
    as model inputs for prequential prediction; train_idx is revealed afterwards.
    """
    if time_index < 1 or time_index >= len(net.times):
        raise ValueError("time_index must be between 1 and last observed time")
    rng = np.random.default_rng(seed + time_index * 1777)
    hist = net.get_history(time_index)
    now = net.get_current(time_index)
    known = np.unique(hist)
    known_mask = np.zeros(net.n, dtype=bool)
    known_mask[known] = True
    past_keys = set(pair_keys(hist, net.n, net.directed).tolist())
    now_keys = pair_keys(now, net.n, net.directed)
    old = np.isin(now_keys, list(past_keys))
    novel = now[~old]
    cold = ~(known_mask[novel[:, 0]] & known_mask[novel[:, 1]])
    positives = novel[~cold]
    if max_positives and len(positives) > max_positives:
        positives = positives[rng.choice(len(positives), max_positives, replace=False)]
    forbidden = past_keys | set(now_keys.tolist())
    # Only previously observed nodes are candidates; ordered pairs matter for directed tasks.
    known_set = set(known.tolist())
    forbidden_known = sum(1 for k in forbidden if k // net.n in known_set and k % net.n in known_set)
    max_available = (len(known) * (len(known) - 1) if net.directed else
                     len(known) * (len(known) - 1) // 2) - forbidden_known
    desired = min(int(round(len(positives) * negative_ratio)), max(0, max_available))
    sampled = set()
    attempts = 0
    while len(sampled) < desired and attempts < max(1000, desired * 50):
        u, v = rng.choice(known, 2, replace=False)
        key = (int(u) * net.n + int(v) if net.directed else
               min(int(u), int(v)) * net.n + max(int(u), int(v)))
        if key not in forbidden:
            sampled.add(key)
        attempts += 1
    if len(sampled) < desired:
        for i, u in enumerate(known):
            for v in (known if net.directed else known[i + 1:]):
                if int(u) == int(v):
                    continue
                key = int(u) * net.n + int(v)
                if key not in forbidden:
                    sampled.add(key)
                if len(sampled) == desired:
                    break
            if len(sampled) == desired:
                break
    negatives = np.array([(key // net.n, key % net.n) for key in sorted(sampled)], dtype=np.int64).reshape(-1, 2)
    candidates = np.concatenate([positives, negatives], axis=0)
    labels = np.r_[np.ones(len(positives), np.int64), np.zeros(len(negatives), np.int64)]
    # Stratified, fixed held-out task evaluation, unused in training and Fisher.
    train, hold = [], []
    for c in (0, 1):
        ids = np.flatnonzero(labels == c)
        rng.shuffle(ids)
        n_eval = min(max(1, int(round(len(ids) * eval_fraction))), max(0, len(ids) - 1))
        hold.extend(ids[:n_eval]); train.extend(ids[n_eval:])
    return Task(time_index, net.times[time_index], hist, candidates, labels,
                np.array(train, dtype=np.int64), np.array(hold, dtype=np.int64),
                len(known), len(positives), int(old.sum()), int(cold.sum()), net.directed)


def synthetic_network(n=100, periods=9, seed=7) -> TemporalNetwork:
    """Reproducible evolving stochastic block network; not research evidence."""
    rng = np.random.default_rng(seed)
    membership = np.arange(n) % 4
    events = []
    ever = set()
    for t in range(periods):
        target = 150 if t == 0 else 90
        tries = 0
        added = 0
        while added < target and tries < target * 100:
            tries += 1
            u = int(rng.integers(n))
            same = rng.random() < (0.8 if t < periods // 2 else 0.58)
            if same:
                group = (membership == membership[u]).nonzero()[0]
            else:
                group = (membership != membership[u]).nonzero()[0]
            v = int(rng.choice(group))
            if u == v:
                continue
            a, b = sorted((u, v))
            if (t, a, b) in ever:
                continue
            # Controlled repeated contacts, so repeat and new-link tasks differ.
            ever.add((t, a, b))
            events.append((t, a, b)); added += 1
    return TemporalNetwork(np.asarray(events, dtype=np.int64), n, list(range(periods)), name="synthetic")
