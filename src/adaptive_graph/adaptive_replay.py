"""Phase E2: bounded sample-level replay for temporal link prediction.

Uniform is a reservoir-sampled control; adaptive retains a weighted mixture of
hard-to-predict, diverse, and recent TRAINING examples. No held-out labels are
examined, stored or used for difficulty scoring. This is a heuristic priority
policy, not yet an RL-learned sampler.
"""
from __future__ import annotations
from dataclasses import dataclass, replace
from collections import defaultdict
import numpy as np
import torch
from torch.nn import functional as F


@dataclass
class ReplayExample:
    prepared: object
    index: int
    label: int
    time_index: int
    difficulty: float
    novelty: float
    feature: np.ndarray


class PairReplayMemory:
    def __init__(self, mode="adaptive", capacity=256, batch_size=64,
                 loss_weight=0.45, diversity_weight=0.35, recency_weight=0.20,
                 recency_half_life=3.0, seed=7, min_period_examples=0):
        if mode not in {"adaptive", "uniform"}:
            raise ValueError("Replay mode must be uniform or adaptive")
        if capacity < 2 or batch_size < 1 or recency_half_life <= 0:
            raise ValueError("Replay capacity >=2, batch >=1, half-life >0 required")
        weights = (loss_weight, diversity_weight, recency_weight)
        if min(weights) < 0 or sum(weights) <= 0:
            raise ValueError("Replay weights must be nonnegative with positive sum")
        self.mode, self.capacity, self.batch_size = mode, capacity, batch_size
        self.weights = np.asarray(weights, dtype=np.float64) / sum(weights)
        self.half_life = float(recency_half_life)
        if min_period_examples < 0 or min_period_examples * 2 > capacity:
            raise ValueError("min_period_examples must be >=0 and <= half the memory capacity")
        self.min_period_examples = int(min_period_examples)
        self.items: list[ReplayExample] = []
        self.seen = 0
        self.rng = np.random.default_rng(seed + 3337)
        self.replay_calls = 0
        self.replayed_pairs = 0
        self.last_replay_labels = []

    def configure(self, mode: str, batch_size: int):
        """E3 changes only current-year admission/sampling; never resets memory."""
        if mode not in {"uniform", "adaptive"} or not 1 <= batch_size <= self.capacity:
            raise ValueError("Mode must be uniform/adaptive; batch must fit memory")
        self.mode, self.batch_size = mode, int(batch_size)

    def __len__(self):
        return len(self.items)

    def _scores(self, now):
        if not self.items:
            return np.empty(0, dtype=float)
        decay = np.array([np.exp(-np.log(2) * max(0, now - e.time_index) /
                                 self.half_life) for e in self.items], dtype=float)
        losses = np.array([e.difficulty for e in self.items])
        diversity = np.array([e.novelty for e in self.items])
        return self.weights[0] * losses + self.weights[1] * diversity + self.weights[2] * decay

    @staticmethod
    def _masked_training_copy(pt):
        """Store only labels revealed for this task's learner-training subset."""
        ids = np.asarray(pt.task.train_idx, dtype=np.int64)
        masked_labels = np.zeros_like(pt.task.labels)
        masked_labels[ids] = pt.task.labels[ids]
        masked = torch.zeros_like(pt.labels)
        masked[torch.as_tensor(ids, dtype=torch.long, device=masked.device)] = pt.labels[
            torch.as_tensor(ids, dtype=torch.long, device=masked.device)]
        task = replace(pt.task, labels=masked_labels,
                       eval_idx=np.empty(0, dtype=np.int64),
                       new_link_count=0, repeat_contact_count=0, cold_start_count=0)
        return replace(pt, task=task, labels=masked)

    @staticmethod
    def _candidate_stats(model, pt, indices, chunk=128):
        device = pt.labels.device
        losses, features = [], []
        training = model.training
        model.eval()
        try:
            with torch.no_grad():
                for part in np.array_split(np.asarray(indices, dtype=np.int64),
                                           max(1, (len(indices) + chunk - 1) // chunk)):
                    if not len(part):
                        continue
                    ids = torch.as_tensor(part, dtype=torch.long, device=device)
                    logits = model(pt, ids)
                    target = pt.labels[ids]
                    losses.append(F.binary_cross_entropy_with_logits(
                        logits, target, reduction="none").detach().cpu().numpy())
                    # History-derived embeddings define structural diversity.
                    if hasattr(model, "states"):
                        vec = model.states(pt, ids, detach=True)
                    else:
                        pairs = pt.candidates[ids]
                        vec = torch.cat((pt.x[pairs[:, 0]], pt.x[pairs[:, 1]]), -1)
                    vec = F.normalize(vec.float(), dim=-1, eps=1e-8)
                    features.append(vec.cpu().numpy().astype(np.float32))
        finally:
            model.train(training)
        return np.concatenate(losses), np.concatenate(features)

    def add(self, pt, model):
        """Observe one revealed training task; candidate test labels are never read.

        Uniform memory uses unbiased reservoir sampling. Adaptive memory keeps
        high-priority examples using a frozen evaluation of training-only pairs.
        """
        indices = np.asarray(pt.task.train_idx, dtype=np.int64)
        if not len(indices):
            return
        snapshot = self._masked_training_copy(pt)
        losses, feats = self._candidate_stats(model, snapshot, indices)
        labels = snapshot.task.labels[indices]
        # Clamp hardness into [0, 1]; this resists noisy outlier losses.
        hardness = np.clip(losses / 2.0, 0.0, 1.0)
        now = int(pt.task.time_index)
        order = self.rng.permutation(len(indices))
        for k in order:
            k = int(k)
            self.seen += 1
            vec = feats[k]
            if not self.items:
                novelty = 1.0
            else:
                current = np.stack([x.feature for x in self.items])
                cos = np.clip(current @ vec, -1, 1)
                novelty = float(np.clip(1 - max(0.0, float(np.max(cos))), 0, 1))
            entry = ReplayExample(snapshot, int(indices[k]), int(labels[k]), now,
                                  float(hardness[k]), novelty, vec.copy())
            if self.mode == "uniform":
                if len(self.items) < self.capacity:
                    self.items.append(entry)
                elif self.min_period_examples:
                    # E3.1's period reserve must survive DQN changes from
                    # adaptive to uniform and back. The uniform mode is therefore
                    # a coverage-constrained reservoir, not a pure reservoir.
                    from collections import Counter
                    periods = Counter(x.time_index for x in self.items)
                    candidates = [i for i, old in enumerate(self.items)
                                  if old.time_index == entry.time_index or
                                  periods[old.time_index] > self.min_period_examples]
                    if not candidates:
                        continue
                    needs_period = periods[entry.time_index] < self.min_period_examples
                    j = int(self.rng.integers(self.seen))
                    if needs_period:
                        # Directly reserve an entry for each newly observed year.
                        self.items[int(self.rng.choice(candidates))] = entry
                    elif j < self.capacity and j in candidates:
                        self.items[j] = entry
                else:
                    j = int(self.rng.integers(self.seen))
                    if j < self.capacity:
                        self.items[j] = entry
            else:
                if len(self.items) < self.capacity:
                    self.items.append(entry)
                else:
                    new_score = (self.weights[0] * entry.difficulty +
                                 self.weights[1] * entry.novelty + self.weights[2])
                    scores = self._scores(now)
                    # New entries compete directly with the worst retained item.
                    counts = {label: sum(x.label == label for x in self.items)
                              for label in (0, 1)}
                    minimum = max(1, int(0.2 * self.capacity))
                    # Maintain a minimum minority-class share when both labels
                    # have been observed; priority alone can collapse to one label.
                    from collections import Counter
                    periods = Counter(x.time_index for x in self.items)
                    # Preserve older training periods as the graph changes;
                    # never use sealed evaluation/test targets for admission.
                    eligible = [i for i, old in enumerate(self.items)
                                if (old.label == entry.label or
                                    counts[old.label] > minimum) and
                                (not self.min_period_examples or
                                 old.time_index == entry.time_index or
                                 periods[old.time_index] > self.min_period_examples)]
                    if not eligible:
                        continue
                    worst = min(eligible, key=lambda i: scores[i])
                    # Underrepresented classes enter regardless of hardness.
                    needs_class = counts[entry.label] < minimum
                    needs_period = (self.min_period_examples > 0 and
                                    periods[entry.time_index] < self.min_period_examples and
                                    self.items[worst].time_index != entry.time_index)
                    if needs_class or needs_period or new_score > scores[worst]:
                        self.items[worst] = entry

    def sample_pairs(self, rng, how_many=None, current_time=None):
        if not self.items:
            return []
        n = min(how_many or self.batch_size, len(self.items))
        now = max(x.time_index for x in self.items) if current_time is None else int(current_time)
        if self.mode == "uniform":
            probs = np.full(len(self.items), 1 / len(self.items), dtype=float)
        else:
            priorities = self._scores(now)
            # Positive probability for every example, without replacement.
            logits = (priorities - priorities.max()) / 0.25
            probs = np.exp(logits)
            probs /= probs.sum()
        # When possible, guarantee both link and non-link examples in each batch.
        labels = np.array([e.label for e in self.items], dtype=int)
        chosen = []
        if n >= 2 and len(set(labels)) == 2:
            for target in (0, 1):
                pool = np.flatnonzero(labels == target)
                p = probs[pool] / probs[pool].sum()
                chosen.append(int(rng.choice(pool, p=p)))
        # Reserve a small part of a batch for the earliest observed period.
        # The class-balanced samples above remain mandatory.
        if self.min_period_examples and n >= 4:
            earliest = min(e.time_index for e in self.items)
            if not any(self.items[i].time_index == earliest for i in chosen):
                pool = np.array([i for i, e in enumerate(self.items)
                                 if e.time_index == earliest and i not in chosen])
                if len(pool):
                    p = probs[pool] / probs[pool].sum()
                    chosen.append(int(rng.choice(pool, p=p)))
        rest = n - len(chosen)
        if rest:
            pool = np.array([i for i in range(len(self.items)) if i not in chosen])
            p = probs[pool] / probs[pool].sum()
            chosen.extend(int(i) for i in rng.choice(pool, size=rest, replace=False, p=p))
        self.replay_calls += 1
        self.replayed_pairs += len(chosen)
        self.last_replay_labels = [self.items[j].label for j in chosen]
        groups = defaultdict(list)
        for j in chosen:
            e = self.items[j]
            groups[id(e.prepared)].append(e)
        batches = []
        for entries in groups.values():
            pt = entries[0].prepared
            ids = torch.as_tensor([e.index for e in entries],
                                  dtype=torch.long, device=pt.labels.device)
            batches.append((pt, ids))
        return batches

    def diagnostics(self, now):
        scores = self._scores(now)
        pos = sum(x.label == 1 for x in self.items)
        return {
            "mode": self.mode, "capacity": self.capacity,
            "stored_pairs": len(self.items), "stored_positive": pos,
            "stored_negative": len(self.items) - pos,
            "stored_periods": len({x.time_index for x in self.items}),
            "min_period_examples": self.min_period_examples,
            "oldest_stored_period": min((x.time_index for x in self.items), default=None),
            "youngest_stored_period": max((x.time_index for x in self.items), default=None),
            "mean_difficulty": float(np.mean([x.difficulty for x in self.items])) if self.items else 0.,
            "mean_novelty": float(np.mean([x.novelty for x in self.items])) if self.items else 0.,
            "mean_priority": float(scores.mean()) if len(scores) else 0.,
            "seen_training_pairs": self.seen,
            "replay_calls": self.replay_calls,
            "replayed_pairs": self.replayed_pairs,
        }
