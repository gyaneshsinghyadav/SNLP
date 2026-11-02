"""Temporal RL for deciding *how to adapt* a GCN, not which links exist.

Protocol (target year t):
  1. Score t before learning from t's labels (prequential evaluation).
  2. Reveal t's training-only labels; score a disjoint controller-validation subset.
  3. Credit the adaptation action from t-1 using next-year validation AP
     compared with a frozen pre-adaptation counterfactual, plus old-task
     validation forgetting and adaptation wall-clock cost.
  4. Choose an adaptation strategy using observations available at t.
  5. Update the GCN and record an unfinished transition for t+1.

The final `test_last` time periods have frozen model and policy weights. Their
labels are never used to update the GCN, policy, EWC, or replay. For intermediate
training years, validation labels are revealed *only at their own year*.

This is a small, experimental learner-control MDP/POMDP; a single temporal
trajectory is not sufficient evidence of optimal or generalizable RL behavior.
"""
from __future__ import annotations
from collections import deque
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from pathlib import Path
import csv
import json
import random
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .continual import OnlineEWC, TaskReplay, consolidate_gcn, train_gcn
from .data import TemporalNetwork, create_task
from .experiment import Config
from .metrics import calculate_metrics
from .models import HybridGCN, prepare


@dataclass(frozen=True)
class Adaptation:
    name: str
    ewc_scale: float
    replay_scale: float


# Scales multiply the global cfg.ewc_lambda and cfg.replay_weight.
ACTIONS = (
    Adaptation("plastic", 0.0, 0.0),
    Adaptation("balanced", 1.0, 1.0),
    Adaptation("ewc_focus", 3.0, 0.0),
    Adaptation("replay_focus", 0.0, 2.0),
)
ACTION_INDEX = {a.name: i for i, a in enumerate(ACTIONS)}
# E3 controls a compact 6-way combination of sampler, per-epoch replay budget,
# and regularization. replay_scale=2.0 matches E1's replay_focus strength.
@dataclass(frozen=True)
class E3Action:
    name: str
    ewc_scale: float
    replay_scale: float
    replay_mode: str
    replay_fraction: float
    projected: bool = False

E3_ACTIONS = (
    E3Action("no_replay", 0.0, 0.0, "adaptive", 0.0),
    E3Action("uniform_light", 0.0, 2.0, "uniform", 0.25),
    E3Action("adaptive_light", 0.0, 2.0, "adaptive", 0.25),
    E3Action("uniform_full", 0.0, 2.0, "uniform", 1.0),
    E3Action("adaptive_full", 0.0, 2.0, "adaptive", 1.0),
    E3Action("adaptive_ewc", 1.0, 2.0, "adaptive", 1.0),
)
# E3.1 prevents the no-replay collapse observed in the DBLP seed-7 pilot.
# Its entire action space rehearses examples whenever stored memory is available.
E31_ACTIONS = tuple(a for a in E3_ACTIONS if a.replay_fraction > 0)
E32_ACTIONS = E31_ACTIONS + (E3Action("adaptive_projected", 0.0, 2.0, "adaptive", 1.0, True),)
STATE_SIZE = 10
C2_STATE_SIZE = 16
E3_STATE_SIZE = 19
E31_STATE_SIZE = 22
E32_STATE_SIZE = 25  # E5 deliberately reuses the same observation vector



def action_catalog(options):
    return (E32_ACTIONS if options.action_space in {"e32", "e5"} else
            E31_ACTIONS if options.action_space == "e31" else
            E3_ACTIONS if options.action_space == "e3" else ACTIONS)


def action_mask(options, state):
    """Allowed E3.1 actions from history-only state; usable by policy and Bellman target.

    Position 8 is *previous training step's* observed historical AP drop (C2
    state definition). Never consult labels from the task being predicted.
    """
    actions = action_catalog(options)
    mask = np.ones(len(actions), dtype=bool)
    if options.action_space in {"e31", "e32", "e5"} and float(state[8]) >= options.retention_guard_threshold:
        mask = np.array([a.replay_fraction >= 1.0 for a in actions], dtype=bool)
    if options.action_space in {"e32", "e5"} and len(state) >= E32_STATE_SIZE and float(state[24]) > 0:
        # Pre-action *training-only* gradient conflict can proactively forbid
        # light replay, even before a historical AP drop is observed.
        if float(state[22]) < options.conflict_guard_threshold:
            mask &= np.array([a.replay_fraction >= 1.0 for a in actions], dtype=bool)
    assert mask.any()
    return mask


@dataclass
class ControllerOptions:
    policy: str = "dqn"  # dqn, fixed, random
    fixed_action: str = "balanced"
    gamma: float = 0.9
    test_last: int = 2
    controller_val_fraction: float = 0.25
    forgetting_lambda: float = 0.25
    cost_lambda: float = 0.001
    controller_lr: float = 0.001
    controller_updates: int = 8
    controller_batch_size: int = 16
    controller_memory: int = 256
    target_sync: int = 12
    state_version: str = "v1"  # v1 retained for Phase B/C1; c2 history-only features
    action_space: str = "legacy"  # separate controllers: legacy / e3 / e31
    peak_forgetting_lambda: float = 0.0
    retention_guard_threshold: float = 0.02  # AP units (0.02 = two pp)
    conflict_guard_threshold: float = -0.05
    gradient_probe_pairs: int = 16
    e5_counterfactual_actions: int = 4
    e5_counterfactual_updates: int = 4
    e5_counterfactual_weight: float = 0.5
    e5_full_replay_batch: int = 64

    def validate(self):
        if self.action_space not in {"legacy", "e3", "e31", "e32", "e5"}:
            raise ValueError("action_space must be legacy, e3, e31, e32 or e5")
        if self.state_version not in {"v1", "c2", "e3", "e31", "e32", "e5"}:
            raise ValueError("state_version must be v1, c2, e3, e31, e32 or e5")
        if self.action_space in {"e3", "e31", "e32", "e5"} and self.state_version != self.action_space:
            raise ValueError("E3/E3.1 action space requires matching controller state")
        if self.action_space == "legacy" and self.state_version in {"e3", "e31", "e32", "e5"}:
            raise ValueError("E3/E3.1 states require matching action space")
        if self.peak_forgetting_lambda < 0 or not 0 <= self.retention_guard_threshold <= 1:
            raise ValueError("Retention guard and peak-forgetting penalty must be nonnegative and <=1 guard")
        if self.gradient_probe_pairs < 2 or not -1 <= self.conflict_guard_threshold <= 0:
            raise ValueError("E3.2 probe pairs >=2 and gradient conflict threshold in [-1,0]")
        if not (2 <= self.e5_counterfactual_actions <= len(E32_ACTIONS)
                and self.e5_counterfactual_updates >= 1
                and 0 <= self.e5_counterfactual_weight <= 1
                and self.e5_full_replay_batch >= 2):
            raise ValueError("Invalid E5 counterfactual hyperparameters")
        if self.policy not in {"dqn", "fixed", "random"}:
            raise ValueError("policy must be dqn, fixed, or random")
        if self.fixed_action not in {a.name for a in action_catalog(self)}:
            raise ValueError(f"Unknown action {self.fixed_action!r}")
        if not 0 <= self.gamma < 1:
            raise ValueError("gamma must be in [0, 1)")
        if self.test_last < 1:
            raise ValueError("test_last must be >=1")
        if not 0 < self.controller_val_fraction < 0.5:
            raise ValueError("controller_val_fraction must be in (0, .5)")
        if self.forgetting_lambda < 0 or self.cost_lambda < 0:
            raise ValueError("reward penalties must be nonnegative")
        if self.controller_updates < 1 or self.controller_batch_size < 1:
            raise ValueError("controller updates/batch size must be positive")
        if self.controller_memory < 1 or self.target_sync < 1:
            raise ValueError("controller memory/sync interval must be positive")


class ControllerQ(nn.Module):
    def __init__(self, state_dim=STATE_SIZE, hidden=32, n_actions=len(ACTIONS)):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, n_actions))

    def forward(self, state):
        return self.net(state)


class TemporalDQN:
    """One action per temporal training task; next state is the next YEAR."""
    def __init__(self, options: ControllerOptions, seed: int, device: str):
        self.options = options
        self.rng = np.random.default_rng(seed + 144)
        self.device = device
        self.actions = action_catalog(options)
        self.state_dim = (E32_STATE_SIZE if options.state_version in {"e32", "e5"} else
                          E31_STATE_SIZE if options.state_version == "e31" else
                          E3_STATE_SIZE if options.state_version == "e3" else
                          C2_STATE_SIZE if options.state_version == "c2" else STATE_SIZE)
        self.q = ControllerQ(state_dim=self.state_dim, n_actions=len(self.actions)).to(device)
        self.target = deepcopy(self.q).eval()
        self.optimizer = torch.optim.Adam(self.q.parameters(), lr=options.controller_lr)
        self.memory = deque(maxlen=options.controller_memory)
        self.update_count = 0
        self.cf_model = ControllerQ(state_dim=self.state_dim, n_actions=len(self.actions)).to(device) if options.action_space == "e5" else None
        self.cf_optimizer = (torch.optim.Adam(self.cf_model.parameters(), lr=options.controller_lr)
                             if self.cf_model is not None else None)
        self.cf_memory = deque(maxlen=options.controller_memory)
        self.cf_updates = 0

    def q_values(self, state: np.ndarray) -> np.ndarray:
        state = np.asarray(state, dtype=np.float32).reshape(-1)
        if state.size != self.state_dim:
            raise ValueError(f"Expected controller state length {self.state_dim}, got {state.size}")
        with torch.no_grad():
            values = self.q(torch.as_tensor(state[None], dtype=torch.float32,
                                            device=self.device))
        return values.squeeze(0).detach().cpu().numpy()

    def counterfactual_values(self, state):
        if self.cf_model is None or not self.cf_memory:
            return np.zeros(len(self.actions), dtype=np.float32)
        with torch.no_grad():
            tensor = torch.as_tensor(np.asarray(state, np.float32)[None], device=self.device)
            return self.cf_model(tensor).squeeze(0).cpu().numpy()

    def train_counterfactual(self, state, rows):
        """Learn immediate surrogate outcomes, AFTER this year's action was chosen.

        Separate critic, separate optimizer: delayed DQN TD targets are never
        replaced by one-step surrogate rewards. Input rows are already masked
        to training-era validation only.
        """
        if self.cf_model is None:
            return float("nan")
        for item in rows:
            self.cf_memory.append((np.asarray(state, np.float32).copy(),
                                   int(item['action_index']), float(item['surrogate_reward'])))
        loss_value = float("nan")
        for _ in range(self.options.e5_counterfactual_updates):
            ids = self.rng.choice(len(self.cf_memory),
                                  size=min(self.options.controller_batch_size, len(self.cf_memory)),
                                  replace=False)
            batch = [self.cf_memory[int(i)] for i in np.atleast_1d(ids)]
            states = torch.tensor(np.stack([b[0] for b in batch]), device=self.device)
            act = torch.tensor([b[1] for b in batch], dtype=torch.long, device=self.device)
            targets = torch.tensor([b[2] for b in batch], dtype=torch.float32, device=self.device)
            estimates = self.cf_model(states).gather(1, act[:, None]).flatten()
            loss = F.smooth_l1_loss(estimates, targets)
            self.cf_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.cf_model.parameters(), 5.0)
            self.cf_optimizer.step()
            self.cf_updates += 1
            loss_value = float(loss.detach())
        return loss_value

    def choose_with_trace(self, state: np.ndarray, epsilon: float = 0.25):
        """Trace PRE-action Q-values and whether epsilon caused exploration."""
        raw_values = self.q_values(state)
        cf_values = self.counterfactual_values(state)
        values = raw_values + (self.options.e5_counterfactual_weight * cf_values
                               if self.cf_memory else 0)
        allowed = action_mask(self.options, state)
        allowed_ids = np.flatnonzero(allowed)
        greedy = int(allowed_ids[np.argmax(values[allowed])])
        explored = bool(self.rng.random() < epsilon)
        action = int(self.rng.choice(allowed_ids)) if explored else greedy
        available = np.sort(values[allowed])
        margin = float(available[-1] - available[-2]) if len(available) >= 2 else 0.0
        return action, {"q_values": values, "cf_values": cf_values, "greedy_action": self.actions[greedy].name,
                        "explored": explored, "epsilon": float(epsilon),
                        "q_margin": margin, "guard_active": bool(not allowed.all()),
                        "allowed_actions": [self.actions[i].name for i in allowed_ids]}

    def choose(self, state: np.ndarray, epsilon: float = 0.25) -> int:
        return self.choose_with_trace(state, epsilon)[0]

    def observe(self, state, action, reward, next_state, terminal=False) -> float:
        self.memory.append((np.asarray(state, np.float32).copy(), int(action), float(reward),
                            np.asarray(next_state, np.float32).copy(), bool(terminal)))
        last = float("nan")
        for _ in range(self.options.controller_updates):
            indices = self.rng.choice(len(self.memory),
                                      size=min(self.options.controller_batch_size,
                                               len(self.memory)), replace=False)
            batch = [self.memory[int(i)] for i in np.atleast_1d(indices)]
            states = torch.tensor(np.stack([x[0] for x in batch]), device=self.device)
            actions = torch.tensor([x[1] for x in batch], dtype=torch.long, device=self.device)
            rewards = torch.tensor([x[2] for x in batch], device=self.device)
            next_states = torch.tensor(np.stack([x[3] for x in batch]), device=self.device)
            terminals = torch.tensor([x[4] for x in batch], dtype=torch.float32,
                                     device=self.device)
            pred = self.q(states).gather(1, actions[:, None]).flatten()
            with torch.no_grad():
                # Double-DQN style action selection avoids argmax on a moving target.
                future_q = self.q(next_states)
                if self.options.action_space in {"e31", "e32", "e5"}:
                    # Bootstrap over feasible next actions only; otherwise the
                    # learner would optimize an action it cannot execute.
                    permitted = torch.as_tensor(np.stack([
                        action_mask(self.options, s) for s in np.stack([x[3] for x in batch])
                    ]), dtype=torch.bool, device=self.device)
                    future_q = future_q.masked_fill(~permitted, float("-inf"))
                future_actions = future_q.argmax(dim=1, keepdim=True)
                next_value = self.target(next_states).gather(1, future_actions).flatten()
                target = rewards + self.options.gamma * (1.0 - terminals) * next_value
            loss = F.smooth_l1_loss(pred, target)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q.parameters(), 5.0)
            self.optimizer.step()
            self.update_count += 1
            if self.update_count % self.options.target_sync == 0:
                self.target.load_state_dict(self.q.state_dict())
            last = float(loss.detach().cpu())
        return last


def split_controller_validation(task, fraction, seed):
    """Disjoint stratified GCN-train and controller-validation indices.

    Both are subsets of task.train_idx. task.eval_idx remains completely sealed.
    """
    rng = np.random.default_rng(seed + 61019 + task.time_index * 37)
    train, val = [], []
    for label in (0, 1):
        candidates = np.asarray(task.train_idx)[task.labels[task.train_idx] == label].copy()
        rng.shuffle(candidates)
        count = min(max(1, int(round(len(candidates) * fraction))), len(candidates) - 1)
        if count < 1:
            return None
        val.extend(candidates[:count].tolist())
        train.extend(candidates[count:].tolist())
    train = np.array(sorted(train), dtype=np.int64)
    val = np.array(sorted(val), dtype=np.int64)
    assert not (set(train) & set(val))
    assert not (set(val) & set(task.eval_idx))
    return train, val


def model_ap(model, pt, indices: np.ndarray):
    if model is None or len(indices) == 0:
        return float("nan")
    model.eval()
    ix = torch.as_tensor(indices, dtype=torch.long, device=pt.x.device)
    with torch.no_grad():
        prediction = model(pt, ix).detach().cpu().numpy()
    return calculate_metrics(pt.task.labels[indices], prediction).get("ap", float("nan"))


def model_metrics(model, pt):
    if model is None:
        return {}
    model.eval()
    ix = pt.subset("eval")
    with torch.no_grad():
        predictions = model(pt, ix).detach().cpu().numpy()
    return calculate_metrics(pt.task.labels[pt.task.eval_idx], predictions,
                             pt.task.candidates[pt.task.eval_idx])


def observed_state(task, val_ap, last_forgetting, last_loss,
                   replay_size, ewc_ready, net_nodes, total_periods):
    """Every input is available when target-year t labels have been revealed."""
    n = max(1, net_nodes)
    pairs = max(1, len(task.train_idx))
    state = np.array([
        task.time_index / max(1, total_periods - 1),
        np.log1p(len(task.historical_edges)) / 17.0,
        task.history_node_count / n,
        np.log1p(task.new_link_count) / 12.0,
        task.repeat_contact_count / max(1, task.new_link_count + task.repeat_contact_count),
        float(task.labels[task.train_idx].sum()) / pairs,
        float(np.clip(val_ap if np.isfinite(val_ap) else 0.0, 0.0, 1.0)),
        float(np.clip(last_forgetting, 0.0, 1.0)),
        float(np.clip(last_loss if np.isfinite(last_loss) else 0.0, 0.0, 10.0)) / 10.0,
        (min(replay_size, 5) / 5.0 if ewc_ready else 0.0),
    ], dtype=np.float32)
    return state


def c2_observed_state(task, train_idx, val_ap, previous_val_ap,
                      last_forgetting, last_loss, replay_size, ewc_ready,
                      net_nodes, total_periods, previous_historical_edges):
    """16 history-only / TRAIN-revealed observation fields.

    All labels used here are indexed by `train_idx`, which excludes both the
    controller-validation and sealed evaluation pairs. Controller validation
    AP is supplied separately and revealed only at the current training year.
    In particular, `task.new_link_count` and `task.repeat_contact_count` are
    forbidden: they summarize target-year edges outside the revealed subset.
    """
    idx = np.asarray(train_idx, dtype=np.int64)
    if len(idx) == 0:
        raise ValueError("C2 controller needs current-year training observations")
    hist = np.asarray(task.historical_edges, dtype=np.int64).reshape(-1, 2)
    degrees = np.bincount(hist.ravel(), minlength=net_nodes) if len(hist) else np.zeros(net_nodes)
    active = degrees[degrees > 0]
    current_ap = float(val_ap) if np.isfinite(val_ap) else 0.0
    previous_ap = float(previous_val_ap) if np.isfinite(previous_val_ap) else current_ap
    history_count = len(hist)
    growth = (history_count - previous_historical_edges) / max(1, previous_historical_edges)
    pos_fraction = float(np.mean(task.labels[idx] == 1))
    positive_count = int(np.count_nonzero(task.labels[idx] == 1))
    # Labels for sealed task.eval_idx and the rest of task.train_idx are never consulted.
    return np.array([
        task.time_index / max(1, total_periods - 1),
        np.log1p(history_count) / 17.0,
        task.history_node_count / max(1, net_nodes),
        np.log1p(positive_count) / 12.0,
        pos_fraction,
        float(np.clip(current_ap, 0, 1)),
        float(np.clip(previous_ap, 0, 1)),
        float(np.clip(current_ap - previous_ap, -1, 1)),
        float(np.clip(last_forgetting, 0, 1)),
        float(np.clip(last_loss if np.isfinite(last_loss) else 0, 0, 10)) / 10,
        min(replay_size, 5) / 5.0,
        float(bool(ewc_ready)),
        float(np.clip(growth, 0, 5)) / 5,
        float(np.clip((2 * history_count) / max(1, len(active)), 0, 100)) / 100,
        float(np.mean(active <= 2)) if len(active) else 0.0,
        float(np.clip(np.std(active) / (np.mean(active) + 1e-6), 0, 10)) / 10
        if len(active) else 0.0,
    ], dtype=np.float32)


def reward_components(next_ap, frozen_ap, forgetting, seconds, options,
                      peak_forgetting=0.0):
    """Same-target next-year delta AP against a no-update counterfactual."""
    if not np.isfinite(next_ap) or not np.isfinite(frozen_ap):
        raise ValueError("Missing next-year controller-validation AP")
    delta_ap = float(next_ap - frozen_ap)
    forgetting = float(max(0.0, forgetting))
    peak_forgetting = float(max(0.0, peak_forgetting))
    seconds = float(max(0.0, seconds))
    return {
        "delta_ap": delta_ap,
        "forgetting_ap": forgetting,
        "peak_forgetting_ap": peak_forgetting,
        "adaptation_seconds": seconds,
        "reward": delta_ap - options.forgetting_lambda * forgetting
                  - options.peak_forgetting_lambda * peak_forgetting
                  - options.cost_lambda * seconds,
    }


def _csv(path: Path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, obj):
    path.write_text(json.dumps(obj, indent=2, default=str, allow_nan=False))


def run_temporal_controller(net: TemporalNetwork, cfg: Config, output_dir,
                            options: ControllerOptions | None = None,
                            controller: TemporalDQN | None = None,
                            train_policy: bool = True,
                            greedy_policy: bool = False,
                            transition_sink: list | None = None,
                            policy_selector=None, evaluate_test: bool = True,
                            prepare_builder=None, model_builder=None,
                            encoder_label: str = "gcn", replay_options: dict | None = None):

    options = options or ControllerOptions()
    options.validate()
    actions = action_catalog(options)
    if options.action_space in {"e3", "e31", "e32", "e5"}:
        if replay_options is None or replay_options.get("mode") not in {"uniform", "adaptive"}:
            raise ValueError("E3/E3.1/E3.2/E5 needs sample replay: --replay-mode adaptive (or uniform)")
        if policy_selector is not None:
            raise ValueError("E3/E3.1/E3.2/E5 has no contextual ridge selector")
        if not cfg.use_replay:
            raise ValueError("E3/E3.1/E3.2/E5 requires replay enabled")
    if not cfg.use_gcn:
        raise ValueError("The adaptation controller requires a GCN (--no-gcn is invalid)")
    if cfg.feature_track not in {"history", "original"}:
        raise ValueError("Invalid feature track")
    if cfg.start_time_index < 1:
        raise ValueError("start_time_index must be >= 1")
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    torch.set_num_threads(1)
    rng = np.random.default_rng(cfg.seed)
    max_t = (len(net.times) if cfg.max_snapshots <= 0
             else min(cfg.max_snapshots, len(net.times)))
    train_end = max_t - options.test_last
    if train_end - cfg.start_time_index < cfg.warmup + 2:
        raise ValueError("Insufficient training years: need >= warmup + 2 training "
                         "periods before frozen test years; reduce warmup/test_last "
                         "or include more snapshots")
    if options.action_space in {"e31", "e32", "e5"}:
        period_quota = replay_options.get("min_period_examples", 0)
        active_periods = train_end - cfg.start_time_index
        if period_quota * active_periods > replay_options["capacity"]:
            raise ValueError("E3.1 period coverage quota exceeds capacity for training periods; "
                             "increase --replay-capacity or decrease --min-period-examples")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if replay_options is not None:
        from .adaptive_replay import PairReplayMemory
        memory = PairReplayMemory(seed=cfg.seed, **replay_options)
    else:
        memory = TaskReplay(cfg.memory_tasks)
    _write_json(out / "manifest.json", {
        "dataset": net.name, "encoder": encoder_label, "nodes": net.n, "events": len(net.events),
        "times": [str(x) for x in net.times],
        "gcn_config": asdict(cfg), "controller_options": asdict(options),
        "actions": [asdict(a) for a in actions],
        "controller_action_space": options.action_space,
        "train_time_indices": list(range(cfg.start_time_index, train_end)),
        "frozen_test_time_indices": list(range(train_end, max_t)),
        "protocol": "t prediction before reveal; action(t) after reveal; "
                    "reward(t) from validation at t+1; controller frozen on test years",
        "counterfactual": "frozen pre-action learner on identical next-year validation candidates (DQN TD reward)",
        "e5_auxiliary_counterfactual": (
            "immediate same-year training-era validation gain from cloned branches, "
            "computed strictly after action selection; not the next-year reward"
            if options.action_space == "e5" else None),
        "controller_training_enabled": train_policy,
        "controller_state_version": options.state_version,
        "controller_greedy_actions": greedy_policy,
        "replay_options": replay_options or {"mode": "task", "capacity_tasks": cfg.memory_tasks},
        "caveats": ["single timeline gives very few action-reward transitions",
                    "cost reward is hardware-dependent wall time",
                    "test metrics are sampled-candidate, not official OGB benchmark",
                    "history-selected subgraphs are exploratory, not full-graph results"],
    })
    if controller is None:
        controller = TemporalDQN(options, cfg.seed, cfg.device)
    elif controller.options != options:
        raise ValueError("Shared controller options must match the episode options")
    updates_at_start = controller.update_count
    model = None
    optimizer = None
    ewc = OnlineEWC()
    probes = deque(maxlen=cfg.max_retention_tasks)
    decisions, rewards, metrics, losses, retention, audit = [], [], [], [], [], []
    replay_rows = []
    gradient_rows = []
    counterfactual_rows = []
    counterfactual_diagnostics = []
    e5_probe_seconds = 0.
    last_replay_budget = 0
    pending = None
    n_trained = 0
    n_chosen = 0
    prev_loss = 0.0
    prev_forgetting = 0.0
    prev_prev_forgetting = 0.0
    prev_peak_forgetting = 0.0
    prev_val_ap = float("nan")
    prev_history_count = 0
    initial_time = time.perf_counter()

    for ti in range(cfg.start_time_index, max_t if evaluate_test else train_end):
        is_train = ti < train_end
        task = create_task(net, ti, seed=cfg.seed, max_positives=cfg.max_positives,
                           negative_ratio=cfg.negative_ratio,
                           eval_fraction=cfg.eval_fraction)
        partition = split_controller_validation(task, options.controller_val_fraction,
                                                cfg.seed) if is_train else None
        if not len(task.eval_idx) or (is_train and partition is None):
            # Gaps cannot be silently linked as consecutive RL transitions.
            raise ValueError(f"Snapshot {task.time_value}: insufficient stratified "
                             "candidates for GCN/controller validation. Use a larger "
                             "graph or larger --max-positives/--max-nodes.")
        train_idx, val_idx = (partition if is_train else
                              (np.asarray(task.train_idx, dtype=np.int64),
                               np.array([], dtype=np.int64)))
        train_task = replace(task, train_idx=train_idx)
        pt = (prepare_builder(train_task) if prepare_builder is not None else
              prepare(train_task, net.n, net.node_features, cfg.feature_track, cfg.device))
        audit.append({"time": str(task.time_value), "phase": "train" if is_train else "test",
                      "history_edges": len(task.historical_edges),
                      "new_links": task.new_link_count, "repeat_contacts": task.repeat_contact_count,
                      "cold_start": task.cold_start_count, "train_candidates": len(train_idx),
                      "controller_validation_candidates": len(val_idx),
                      "sealed_test_candidates": len(task.eval_idx),
                      "label_overlap_count": len(set(train_idx) & set(val_idx))
                                             + len(set(val_idx) & set(task.eval_idx))
                                             + len(set(train_idx) & set(task.eval_idx))})
        if model is None:
            # Isolate GCN initialization from controller construction (shared
            # DQN versus fresh fixed controller), ensuring fair comparisons.
            torch.manual_seed(cfg.seed + 202700)
            model = (model_builder(pt) if model_builder is not None else
                     HybridGCN(pt.x.shape[1], cfg.hidden, use_gcn=True,
                               use_heuristics=cfg.use_heuristics)).to(cfg.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr,
                                         weight_decay=1e-5)
        # Scores are computed BEFORE training from year t, so no t-label leakage.
        if n_trained >= cfg.warmup:
            scores = model_metrics(model, pt)
            metrics.append({"time": str(task.time_value),
                            "phase": "train" if is_train else "test", **scores})
        if not is_train:
            # Absolutely no test-year labels are used for controller reward,
            # controller-state construction, or GCN adaptation.
            print(f"[temporal-{options.policy}] frozen test year={task.time_value}", flush=True)
            continue

        observed_ap = model_ap(model, pt, val_idx) if n_trained else float("nan")
        if options.state_version in {"c2", "e3", "e31", "e32", "e5"}:
            state = c2_observed_state(task, train_idx, observed_ap, prev_val_ap,
                                      prev_forgetting, prev_loss, len(memory.items),
                                      bool(ewc.fisher), net.n, max_t,
                                      prev_history_count)
            if options.state_version in {"e3", "e31", "e32", "e5"}:
                # All three fields describe previously observed training memory,
                # never a future/test target or its label.
                mem_size = max(1, len(memory.items))
                positive = sum(x.label == 1 for x in memory.items)
                class_imbalance = abs(2 * positive - len(memory.items)) / mem_size
                state = np.concatenate((state, np.array([
                    float(memory.mode == "adaptive"),
                    last_replay_budget / max(1, replay_options["batch_size"]),
                    class_imbalance], dtype=np.float32)))
                if options.state_version in {"e31", "e32", "e5"}:
                    # Previous observed train-only retention trend and coverage.
                    # Epoch t's training labels haven't influenced these features.
                    period_coverage = len({x.time_index for x in memory.items}) / max(1, n_trained)
                    state = np.concatenate((state, np.array([
                        np.clip(prev_forgetting - prev_prev_forgetting, -1, 1),
                        np.clip(prev_peak_forgetting, 0, 1),
                        np.clip(period_coverage, 0, 1)], dtype=np.float32)))
                if options.state_version in {"e32", "e5"}:
                    from .gradient_conflict import probe_gradient_conflict
                    probe_started = time.perf_counter()
                    gradient_diagnostic = probe_gradient_conflict(
                        model, pt, memory, options.gradient_probe_pairs)
                    gradient_probe_seconds = time.perf_counter() - probe_started
                    state = np.concatenate((state, np.array([
                        gradient_diagnostic['cosine'],
                        min(5., gradient_diagnostic['norm_ratio']) / 5.,
                        float(gradient_diagnostic['available'])], dtype=np.float32)))
                    gradient_rows.append({"time": str(task.time_value),
                                          "gradient_probe_seconds": gradient_probe_seconds,
                                          **gradient_diagnostic})
        else:
            state = observed_state(task, observed_ap, prev_forgetting, prev_loss,
                                   len(memory.items), bool(ewc.fisher), net.n, max_t)
        prev_val_ap = observed_ap
        prev_history_count = len(task.historical_edges)
        if pending is not None:
            frozen_ap = model_ap(pending["frozen"], pt, val_idx)
            reward = reward_components(observed_ap, frozen_ap,
                                       pending["forgetting"], pending["seconds"], options,
                                       pending["peak_forgetting"])
            terminal = (ti == train_end - 1)
            if transition_sink is not None:
                transition_sink.append((pending["state"].copy(), pending["action"],
                                        reward["reward"], state.copy(), terminal))
            loss_q = (controller.observe(pending["state"], pending["action"],
                                         reward["reward"], state, terminal)
                      if options.policy == "dqn" and train_policy else float("nan"))
            rewards.append({"action_year": pending["year"],
                            "reward_year": str(task.time_value),
                            "action": actions[pending["action"]].name,
                            "updated_ap": observed_ap, "frozen_no_update_ap": frozen_ap,
                            "terminal": terminal, "q_td_loss": loss_q, **reward})
            pending = None

        # No reward is available for an action on the LAST training year,
        # because the following years are a sealed, controller-frozen test set.
        eligible_action = (n_trained >= cfg.warmup and ti < train_end - 1)
        trace = None
        if eligible_action and policy_selector is not None:
            values = np.asarray(policy_selector.predict_rewards(state), dtype=np.float64)
            if values.shape != (len(actions),) or not np.isfinite(values).all():
                raise RuntimeError("Contextual selector returned invalid action scores")
            index = int(values.argmax())
            trace = {"estimated_rewards": values}
        elif eligible_action and options.policy == "dqn":
            index, trace = controller.choose_with_trace(
                state, 0.0 if greedy_policy else cfg.epsilon)
        elif eligible_action and options.policy == "random":
            permitted = np.flatnonzero(action_mask(options, state))
            index = int(controller.rng.choice(permitted))
        else:
            index = next(i for i, a in enumerate(actions) if a.name == options.fixed_action)
        action = actions[index]
        if options.action_space in {"e3", "e31", "e32", "e5"}:
            # Only the current training task's decision can change sampler and
            # batch budget. Admission happens after current train labels reveal.
            last_replay_budget = (min(memory.capacity, max(2, int(round(
                                 replay_options["batch_size"] * action.replay_fraction))))
                                 if action.replay_scale > 0 else 0)
            memory.configure(action.replay_mode,
                             min(memory.capacity, last_replay_budget or replay_options["batch_size"]))
        if eligible_action:
            n_chosen += 1
        baseline_model = deepcopy(model).eval() if eligible_action else None
        if (options.action_space == "e5" and options.policy == "dqn"
                and train_policy and eligible_action):
            from .e5_counterfactual import choose_probe_actions, matched_action_probes
            allowed = action_mask(options, state)
            arm_ids = choose_probe_actions(allowed, index,
                                            options.e5_counterfactual_actions,
                                            seed=cfg.seed + ti * 1013)
            # The decision has ALREADY been made; these same-year outcomes can
            # only train the auxiliary critic for FUTURE action decisions.
            counterfactual = matched_action_probes(
                model, optimizer, memory, ewc, pt, val_idx, list(probes),
                actions, arm_ids, cfg, options, seed=cfg.seed + ti * 1013 + 97)
            e5_probe_seconds += sum(x['probe_seconds'] for x in counterfactual)
            for row in counterfactual:
                counterfactual_rows.append({"time": str(task.time_value),
                                            "selected_action": action.name,
                                            "state_seen_before_probe": True,
                                            **row})
            cf_fit_loss = controller.train_counterfactual(state, counterfactual)
            counterfactual_diagnostics.append({"time": str(task.time_value),
                                               "probe_count": len(counterfactual),
                                               "cf_fit_loss": cf_fit_loss,
                                               "critic_training_examples": len(controller.cf_memory)})
        old_scores = [model_ap(model, old_pt, old_idx) for old_pt, old_idx in probes]
        started = time.perf_counter()
        gradient_stats = {}
        loss = train_gcn(
            model, pt, optimizer, epochs=cfg.epochs,
            replay=memory if action.replay_scale > 0 and len(memory.items) else None,
            ewc=ewc if action.ewc_scale > 0 and ewc.fisher else None,
            ewc_lambda=cfg.ewc_lambda * action.ewc_scale,
            replay_weight=cfg.replay_weight * action.replay_scale,
            max_train_pairs=cfg.max_train_pairs, rng=rng,
            project_current_gradient=bool(getattr(action, "projected", False)),
            gradient_stats=gradient_stats)
        seconds = time.perf_counter() - started
        if options.action_space in {"e32", "e5"}:
            # Controller cost penalty must charge the shared diagnostic overhead.
            seconds += gradient_probe_seconds
        new_scores = [model_ap(model, old_pt, old_idx) for old_pt, old_idx in probes]
        drops = [max(0.0, before - after) for before, after in zip(old_scores, new_scores)
                 if np.isfinite(before) and np.isfinite(after)]
        forgetting = float(np.mean(drops)) if drops else 0.0
        peak_forgetting = float(max(drops)) if drops else 0.0
        # Consolidate EWC using only task training labels, regardless of action.
        if cfg.use_ewc:
            consolidate_gcn(model, pt, ewc)
        if cfg.use_replay:
            if replay_options is not None:
                memory_add_start = time.perf_counter()
                memory.add(pt, model)
                selection_seconds = time.perf_counter() - memory_add_start
                # E2 reward cost should include priority computation and
                # selection, not just optimizer time (task mode unchanged).
                seconds += selection_seconds
                replay_rows.append({"time": str(task.time_value),
                                    "memory_selection_seconds": selection_seconds,
                                    **memory.diagnostics(task.time_index)})
            else:
                memory.add(pt)
        probes.append((pt, val_idx))
        prev_loss = loss
        prev_prev_forgetting = prev_forgetting
        prev_forgetting = forgetting
        prev_peak_forgetting = peak_forgetting
        n_trained += 1
        decision = {"time": str(task.time_value), "action": action.name,
                    "policy_selected": eligible_action,
                    "ewc_lambda": cfg.ewc_lambda * action.ewc_scale,
                    "replay_weight": cfg.replay_weight * action.replay_scale,
                    "train_seconds": seconds, "old_validation_forgetting_ap": forgetting,
                    "peak_old_validation_forgetting_ap": peak_forgetting,
                    "observed_validation_ap": observed_ap}
        if options.action_space in {"e3", "e31", "e32", "e5"}:
            decision.update({"replay_mode": action.replay_mode if last_replay_budget else "off",
                             "replay_batch_budget": last_replay_budget,
                             "replay_memory_capacity": memory.capacity,
                             "effective_ewc_scale": action.ewc_scale})
        if trace is not None and "estimated_rewards" in trace:
            decision.update({f"predicted_reward_{a.name}": float(trace["estimated_rewards"][i])
                             for i, a in enumerate(actions)})
            decision["greedy_action"] = action.name
            decision["explored"] = False
        elif trace is not None:
            decision.update({"greedy_action": trace["greedy_action"],
                             "explored": trace["explored"], "epsilon": trace["epsilon"],
                             "q_margin": trace["q_margin"],
                             "retention_guard_active": trace["guard_active"],
                             "allowed_actions": "|".join(trace["allowed_actions"])})
            if options.action_space == "e5":
                decision.update({f"cf_{a.name}": float(trace["cf_values"][i])
                                 for i, a in enumerate(actions)})
            decision.update({f"q_{a.name}": float(trace["q_values"][i])
                             for i, a in enumerate(actions)})
            decision.update({f"state_{i}": float(v) for i, v in enumerate(state)})
        if options.action_space in {"e32", "e5"}:
            decision.update({"gradient_probe_cosine": gradient_diagnostic["cosine"],
                             "gradient_probe_available": gradient_diagnostic["available"],
                             "gradient_probe_reference_pairs": gradient_diagnostic["reference_pairs"],
                             "gradient_projected_steps": gradient_stats.get("projected_steps", 0)})
            gradient_rows[-1].update({"action": action.name,
                                      "gradient_projected_steps": gradient_stats.get("projected_steps", 0)})
        decisions.append(decision)
        losses.append({"time": str(task.time_value), "gcn_train_loss": loss})
        for (old_pt, _), before, after in zip(list(probes)[:-1], old_scores, new_scores):
            retention.append({"after_time": str(task.time_value),
                              "old_task_time": str(old_pt.task.time_value),
                              "ap_before": before, "ap_after": after,
                              "drop_ap": max(0.0, before - after)})
        if eligible_action:
            pending = {"year": str(task.time_value), "state": state.copy(),
                       "action": index, "frozen": baseline_model,
                       "forgetting": forgetting, "peak_forgetting": peak_forgetting,
                       "seconds": seconds}
        print(f"[temporal-{options.policy}] train year={task.time_value} "
              f"action={action.name} loss={loss:.4f} "
              f"validation_forgetting={forgetting:.4f}", flush=True)
    if pending is not None:
        raise RuntimeError("Unresolved training transition before sealed test split")
    for name, rows in (("controller_decisions.csv", decisions),
                       ("controller_rewards.csv", rewards),
                       ("prequential.csv", metrics),
                       ("losses.csv", losses),
                       ("retention.csv", retention),
                       ("task_audit.csv", audit),
                       ("replay_memory.csv", replay_rows),
                       ("gradient_interference.csv", gradient_rows),
                       ("counterfactual_actions.csv", counterfactual_rows),
                       ("counterfactual_training.csv", counterfactual_diagnostics)):
        _csv(out / name, rows)
    test_rows = [x for x in metrics if x["phase"] == "test"]
    train_rows = [x for x in metrics if x["phase"] == "train"]
    def mean(rows, key):
        numbers = [float(r[key]) for r in rows if key in r and np.isfinite(r[key])]
        return float(np.mean(numbers)) if numbers else None
    summary = {
        "dataset": net.name, "encoder": encoder_label,
        "policy": "contextual_ridge" if policy_selector is not None else options.policy,
        "fixed_action": options.fixed_action,
        "gamma": options.gamma,
        "training_updates": n_trained,
        "policy_actions": n_chosen,
        "reward_transitions": len(rewards),
        "controller_gradient_updates": controller.update_count - updates_at_start,
        "controller_gradient_updates_cumulative": controller.update_count,
        "controller_training_enabled": train_policy,
        "controller_state_version": options.state_version,
        "replay_mode": replay_options["mode"] if replay_options else "task",
        "replay_capacity": memory.capacity if replay_options else cfg.memory_tasks,
        "replay_stored_pairs": len(memory.items) if replay_options else None,
        "replay_calls": memory.replay_calls if replay_options else None,
        "replayed_pairs": memory.replayed_pairs if replay_options else None,
        "controller_action_space": options.action_space,
        "mean_gradient_probe_cosine": mean(gradient_rows, "cosine") if gradient_rows else None,
        "gradient_conflict_periods": sum(x["conflict"] for x in gradient_rows),
        "projected_gradient_steps": sum(x.get("gradient_projected_steps", 0) for x in gradient_rows),
        "mean_historical_ap_drop": mean(retention, "drop_ap"),
        "replay_stored_periods": (len({x.time_index for x in memory.items})
                                  if replay_options else None),
        "chosen_actions": {a.name: sum(d["action"] == a.name and d["policy_selected"]
                                        for d in decisions) for a in actions},
        "test_years": [r["time"] for r in test_rows],
        "test_mean_ap": mean(test_rows, "ap"),
        "test_mean_roc_auc": mean(test_rows, "roc_auc"),
        "train_prequential_mean_ap": mean(train_rows, "ap"),
        "mean_observed_reward": mean(rewards, "reward"),
        "counterfactual_comparisons": len(counterfactual_rows),
        "counterfactual_critic_updates": controller.cf_updates,
        "counterfactual_critic_examples": len(controller.cf_memory),
        "counterfactual_probe_seconds": e5_probe_seconds,
        "counterfactual_weight": options.e5_counterfactual_weight if options.action_space == "e5" else None,
        "runtime_seconds": time.perf_counter() - initial_time,
    }
    _write_json(out / "summary.json", summary)
    torch.save({"gcn": model.state_dict(), "controller": controller.q.state_dict(),
                "counterfactual_critic": controller.cf_model.state_dict() if controller.cf_model is not None else None,
                "gcn_config": asdict(cfg), "options": asdict(options),
                "gcn_input_dim": pt.x.shape[1]}, out / "temporal_checkpoint.pt")
    return summary
