"""Library-agnostic ogbl-collab adapter: original features and temporal edge years."""
from __future__ import annotations
from contextlib import contextmanager
import numpy as np
import torch
from .data import TemporalNetwork


@contextmanager
def trusted_ogb_legacy_load(enabled=False):
    """Opt-in compatibility with older OGB .pt files. TRUSTED LOCAL FILES ONLY."""
    if not enabled:
        yield
        return
    original = torch.load
    def legacy_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original(*args, **kwargs)
    torch.load = legacy_load
    try:
        yield
    finally:
        torch.load = original


def load_ogbl_collab(root="datasets/ogb", trust_ogb_pickle=False):
    try:
        from ogb.linkproppred import LinkPropPredDataset
    except ImportError as e:
        raise ImportError("Install with: pip install -e '.[ogb]' (no PyG required)") from e
    with trusted_ogb_legacy_load(trust_ogb_pickle):
        ds = LinkPropPredDataset(name="ogbl-collab", root=root)
        split = ds.get_edge_split()
        graph = ds[0]
    chunks = []
    for partition in ("train", "valid", "test"):
        d = split[partition]
        if "edge" not in d or "year" not in d:
            raise ValueError(f"ogbl-collab {partition} missing edge/year fields")
        edges = np.asarray(d["edge"], dtype=np.int64).reshape(-1, 2)
        years = np.asarray(d["year"], dtype=np.int64).reshape(-1)
        if len(edges) != len(years):
            raise ValueError("Split edge and year lengths mismatch")
        chunks.append(np.column_stack((years, edges)))
    combined = np.concatenate(chunks, axis=0)
    years = sorted(np.unique(combined[:, 0]).tolist())
    labels = np.searchsorted(np.asarray(years), combined[:, 0])
    events = np.column_stack((labels, combined[:, 1:]))
    x = graph.get("node_feat")
    if x is not None:
        x = np.asarray(x, dtype=np.float32)
    return TemporalNetwork(events, int(graph["num_nodes"]), years,
                           node_features=x, name="ogbl-collab")
