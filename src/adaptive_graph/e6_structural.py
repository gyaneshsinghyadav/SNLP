"""E6-B: history-only, pair-specific structural features on a temporal graph.

All state used in the feature vector comes from edges whose time index is STRICTLY
below the task's prediction index. Candidate labels must never enter this class.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from .data import TemporalNetwork, undirected_csr

FEATURE_NAMES = (
    'common_neighbors', 'adamic_adar', 'resource_allocation', 'jaccard',
    'preferential_attachment', 'degree_min', 'degree_max', 'degree_difference',
    'two_hop_weighted_common_neighbors', 'temporal_weighted_adamic_adar',
    'recent_degree_min', 'recent_degree_max',
)


class HistoricalPairFeatures:
    def __init__(self, net: TemporalNetwork, prediction_index: int, decay: float = 0.70):
        if net.directed:
            raise ValueError('E6 DBLP features require an undirected coauthor graph')
        if not 1 <= prediction_index < len(net.times):
            raise ValueError('Prediction index must follow at least one history snapshot')
        if not 0 < decay <= 1:
            raise ValueError('Decay must be in (0,1]')
        self.prediction_index = prediction_index
        # Earlier events ONLY. Note that this uses exactly the previous years of
        # the graph, even if the model parameters stay frozen at test time.
        historical = net.events[net.events[:, 0] < prediction_index]
        edges = historical[:, 1:]
        self.a = undirected_csr(edges, net.n)
        self.degree = np.diff(self.a.indptr).astype(np.float64)
        self.invlog_degree = np.divide(1., np.log(np.maximum(self.degree, 2)),
                                       out=np.zeros_like(self.degree),
                                       where=self.degree > 1)
        self.inverse_degree = 1. / np.maximum(self.degree, 1)
        # Time-aware, weighted historical coauthor edges. Coauthors with more
        # recent collaborations receive larger weights, without future events.
        weights = np.power(decay, prediction_index - 1 - historical[:, 0].astype(int))
        u, v = edges[:, 0], edges[:, 1]
        self.w = sp.coo_matrix((np.r_[weights, weights], (np.r_[u,v], np.r_[v,u])),
                               shape=(net.n, net.n), dtype=np.float64).tocsr()
        self.w.sum_duplicates()
        self.recent_degree = np.asarray(self.w.sum(axis=1)).ravel()
        self.historical_event_count = int(len(historical))

    def transform(self, pairs: np.ndarray) -> np.ndarray:
        pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
        if (pairs < 0).any() or (pairs >= self.a.shape[0]).any():
            raise ValueError('Out-of-bounds candidate node')
        out = np.empty((len(pairs), len(FEATURE_NAMES)), dtype=np.float32)
        for i, (u, v) in enumerate(pairs):
            u, v = int(u), int(v)
            if u == v:
                raise ValueError('Self-pairs are invalid')
            nu = self.a.indices[self.a.indptr[u]:self.a.indptr[u + 1]]
            nv = self.a.indices[self.a.indptr[v]:self.a.indptr[v + 1]]
            common = np.intersect1d(nu, nv, assume_unique=True)
            cn = float(len(common))
            aa = float(self.invlog_degree[common].sum())
            ra = float(self.inverse_degree[common].sum())
            du, dv = self.degree[u], self.degree[v]
            union = du + dv - cn
            # Pair-wise weighted common neighbor score for a two-hop path.
            if len(common):
                wu = np.asarray(self.w[u, common].toarray()).ravel()
                wv = np.asarray(self.w[v, common].toarray()).ravel()
                two_hop_weights = np.sqrt(np.maximum(wu * wv, 0))
                weighted_cn = float(two_hop_weights.sum())
                weighted_aa = float((two_hop_weights * self.invlog_degree[common]).sum())
            else:
                weighted_cn = weighted_aa = 0.
            rdu, rdv = self.recent_degree[u], self.recent_degree[v]
            out[i] = [cn, aa, ra, cn / max(union, 1.), du * dv,
                      min(du, dv), max(du, dv), abs(du - dv),
                      weighted_cn, weighted_aa, min(rdu, rdv), max(rdu, rdv)]
        # Fixed, label-independent transform. No future statistics or scaler fit.
        out = np.log1p(np.maximum(out, 0.))
        return np.nan_to_num(out, copy=False, nan=0., posinf=10., neginf=0.)
