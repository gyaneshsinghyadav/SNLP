"""Binary and sampled-candidate ranking metrics."""
import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score,
                             precision_score, recall_score, f1_score, roc_auc_score)


def calculate_metrics(y, scores, pairs=None, k=10):
    y = np.asarray(y, dtype=int).reshape(-1)
    s = np.asarray(scores, dtype=float).reshape(-1)
    if len(y) != len(s) or not len(y):
        return {}
    # Scores are logits or Q(action=link)-Q(action=non-link), threshold at 0.
    pred = (s >= 0).astype(int)
    out = {"accuracy": float(accuracy_score(y, pred)),
           "precision": float(precision_score(y, pred, zero_division=0)),
           "recall": float(recall_score(y, pred, zero_division=0)),
           "f1": float(f1_score(y, pred, zero_division=0)),
           "ap": float(average_precision_score(y, s)) if y.sum() else float("nan")}
    out["roc_auc"] = float(roc_auc_score(y, s)) if np.unique(y).size == 2 else float("nan")
    top = np.argsort(-s, kind="stable")[:min(k, len(s))]
    out[f"precision_at_{k}"] = float(np.mean(y[top]))
    # Query-level ranks against sampled negatives with the SAME source node.
    if pairs is not None and y.sum():
        pairs = np.asarray(pairs)
        ranks = []
        for i in np.flatnonzero(y):
            negatives = (pairs[:, 0] == pairs[i, 0]) & (y == 0)
            ranks.append(1 + int(np.sum(s[negatives] >= s[i])))
        out["mrr"] = float(np.mean(1.0 / np.asarray(ranks)))
        out[f"hits_at_{k}"] = float(np.mean(np.asarray(ranks) <= k))
    return out
