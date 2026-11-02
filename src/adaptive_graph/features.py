"""Historical-only structural and node features (undirected graph)."""
from __future__ import annotations
import numpy as np
import scipy.sparse as sp

FEATURE_NAMES = ["common_neighbors", "jaccard", "adamic_adar", "katz_l2_l3",
                 "pagerank_product", "preferential_attachment", "degree_min",
                 "degree_max", "degree_difference"]
DIRECTED_FEATURE_NAMES = ["directed_two_hop", "shared_targets", "shared_sources",
                          "historical_reverse_edge", "source_out_degree", "target_in_degree",
                          "source_out_target_in_product", "directed_pagerank_product",
                          "target_out_plus_source_in_degree"]


def pagerank(a: sp.csr_matrix, damping=0.85, iterations=40):
    n = a.shape[0]
    degree = np.asarray(a.sum(axis=1)).ravel()
    p = np.full(n, 1.0 / n)
    for _ in range(iterations):
        next_p = damping * np.asarray(a.T @ (p / np.maximum(degree, 1))).ravel()
        next_p += damping * p[degree == 0].sum() / n + (1 - damping) / n
        p = next_p
    return p.astype(np.float32)


def node_inputs(a: sp.csr_matrix, original_features=None, track="history"):
    deg = np.diff(a.indptr).astype(np.float32)
    pr = pagerank(a)
    x = np.column_stack([np.log1p(deg), np.sqrt(deg), (deg > 0).astype(np.float32),
                         (pr * a.shape[0]).astype(np.float32),
                         1.0 / np.sqrt(deg + 1), np.ones(a.shape[0], np.float32)]).astype(np.float32)
    if track == "original":
        if original_features is None:
            raise ValueError("Original feature track requested, but no original node features exist")
        original = np.nan_to_num(np.asarray(original_features, dtype=np.float32),
                                 nan=0.0, posinf=0.0, neginf=0.0)
        # Per-row normalization independent of held-out edge labels; the OGB
        # original features themselves may encode non-historical information.
        norms = np.linalg.norm(original, axis=1, keepdims=True)
        x = np.column_stack((x, original / np.maximum(norms, 1e-6)))
    return x


def _common(a: sp.csr_matrix, u, v, degree, invlog):
    common = a[u].multiply(a[v])
    cn = float(common.sum())
    aa = float(np.asarray(common @ invlog).reshape(-1)[0])
    return cn, aa


def structural_features(a: sp.csr_matrix, pairs: np.ndarray, katz_beta=0.05,
                        path_budget=128) -> np.ndarray:
    """Compute 9 structural features from the historical graph only.

    Katz is truncated at walks of length 2 and 3; for high-degree nodes,
    at most `path_budget` neighbors of u are used in the length-3 sum.
    For smaller graphs this computation is exact, with no future adjacency.
    """
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    deg = np.diff(a.indptr).astype(float)
    invlog = np.zeros(len(deg))
    valid = deg > 1
    invlog[valid] = 1.0 / np.log(deg[valid])
    pr = pagerank(a)
    result = np.empty((len(pairs), len(FEATURE_NAMES)), dtype=np.float32)
    for i, (u, v) in enumerate(pairs):
        u, v = int(u), int(v)
        cn, aa = _common(a, u, v, deg, invlog)
        union = deg[u] + deg[v] - cn
        # Number of walks of length three: sum_{w neighbor(u)} CN(w, v).
        u_neighbors = a.indices[a.indptr[u]:a.indptr[u + 1]]
        if len(u_neighbors) > path_budget:
            u_neighbors = u_neighbors[:path_budget]
        walks3 = 0.0
        if len(u_neighbors):
            walks3 = float(a[u_neighbors].multiply(a[v]).sum())
        result[i] = (cn, cn / max(union, 1), aa,
                     katz_beta**2 * cn + katz_beta**3 * walks3,
                     float(pr[u] * pr[v]) * len(deg)**2,
                     deg[u] * deg[v], min(deg[u], deg[v]),
                     max(deg[u], deg[v]), abs(deg[u] - deg[v]))
    return result


def transform_pair_features(raw: np.ndarray) -> np.ndarray:
    """Monotone bounded scaling determined without inspecting target labels."""
    x = np.maximum(np.asarray(raw, dtype=np.float32), 0)
    x = np.log1p(x)
    x = np.clip(x, 0, 10)
    return x


def directed_node_inputs(a: sp.csr_matrix) -> np.ndarray:
    """Direction-aware history features: sender and receiver roles are distinct."""
    out_deg = np.asarray(a.sum(axis=1)).ravel().astype(np.float32)
    in_deg = np.asarray(a.sum(axis=0)).ravel().astype(np.float32)
    pr = pagerank(a)
    return np.column_stack((np.log1p(out_deg), np.log1p(in_deg),
                            np.sqrt(out_deg), np.sqrt(in_deg),
                            (pr * len(out_deg)).astype(np.float32),
                            np.ones(len(out_deg), np.float32))).astype(np.float32)


def directed_structural_features(a: sp.csr_matrix, pairs: np.ndarray) -> np.ndarray:
    """Nine directional statistics from past adjacency, without future edges."""
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    out_deg = np.asarray(a.sum(axis=1)).ravel()
    in_deg = np.asarray(a.sum(axis=0)).ravel()
    pr = pagerank(a)
    at = a.T.tocsr()
    features = np.zeros((len(pairs), 9), dtype=np.float32)
    for i, (u, v) in enumerate(pairs):
        u, v = int(u), int(v)
        # (A^2)[u,v] is the directed count u -> middle -> v.
        two_hop = float(a.getrow(u).multiply(at.getrow(v)).sum())
        shared_targets = float(a.getrow(u).multiply(a.getrow(v)).sum())
        shared_sources = float(at.getrow(u).multiply(at.getrow(v)).sum())
        reverse = float(a[v, u] > 0)
        features[i] = [two_hop, shared_targets, shared_sources, reverse,
                       out_deg[u], in_deg[v], out_deg[u] * in_deg[v],
                       pr[u] * pr[v] * len(out_deg)**2,
                       out_deg[v] + in_deg[u]]
    return features
