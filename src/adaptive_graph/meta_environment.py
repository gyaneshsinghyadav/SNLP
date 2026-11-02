"""C1: node-disjoint, chronology-preserving meta-training episodes.

Partition nodes ONLY from the graph before the first training task. Episodes
receive fresh GCN/replay/EWC states but share one DQN. The held-out node group
never contributes DQN gradients or replay samples. All episodes reserve the
last test_last years for strictly frozen GCN/policy evaluation.

Subgraphs omit cross-partition edges: benchmark is exploratory and is NOT the
same test population as the original full-/induced-graph Phase B experiment.
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, replace
from pathlib import Path
import csv
import json

import numpy as np
import torch

from .data import TemporalNetwork, undirected_csr
from .experiment import Config
from .temporal_control import ControllerOptions, TemporalDQN, run_temporal_controller, ACTION_INDEX


def historical_partition(net: TemporalNetwork, groups: int, history_cutoff: int, seed: int):
    """Deterministic, balanced region growing; reads NO edge at/after cutoff.

    Returns node ID arrays. Later labels must NEVER affect community membership.
    We do not promise statistical independence between correlated author groups.
    """
    if not (2 <= groups <= net.n // 2):
        raise ValueError("Number of groups must be between 2 and n_nodes // 2")
    if not (1 <= history_cutoff < len(net.times)):
        raise ValueError("history_cutoff must be a valid positive time index")
    a = undirected_csr(net.get_history(history_cutoff), net.n)
    degree = np.diff(a.indptr)
    rng = np.random.default_rng(seed)
    order = np.lexsort((rng.random(net.n), -degree))
    # Pick high-degree seeds that aren't immediate neighbors of existing seeds.
    seeds = []
    for node in order:
        node = int(node)
        if any(node == s or a[node, s] != 0 for s in seeds):
            continue
        seeds.append(node)
        if len(seeds) == groups:
            break
    for node in order:
        if len(seeds) == groups:
            break
        if int(node) not in seeds:
            seeds.append(int(node))
    target = [net.n // groups + (i < net.n % groups) for i in range(groups)]
    assigned = np.full(net.n, -1, dtype=np.int32)
    queues = [deque([seed]) for seed in seeds]
    sizes = [1] * groups
    for i, node in enumerate(seeds):
        assigned[node] = i
    remaining_order = iter(int(x) for x in order)
    while any(sizes[i] < target[i] for i in range(groups)):
        for i in range(groups):
            if sizes[i] >= target[i]:
                continue
            chosen = None
            # Drain previously discovered frontiers, retaining only newly
            # eligible nodes; prefer within-group edges from early history.
            while queues[i] and chosen is None:
                u = queues[i].popleft()
                neighbors = a.indices[a.indptr[u]:a.indptr[u+1]]
                candidates = [int(v) for v in neighbors if assigned[v] == -1]
                if candidates:
                    chosen = candidates[0]
                    queues[i].extend(candidates[1:])
            if chosen is None:
                for u in remaining_order:
                    if assigned[u] == -1:
                        chosen = u
                        break
            if chosen is None:
                # If the fallback iterator has been consumed by another group,
                # remaining nodes can still be found directly.
                unassigned = np.flatnonzero(assigned == -1)
                if not len(unassigned):
                    raise RuntimeError("Unable to fill balanced node groups")
                chosen = int(unassigned[0])
            assigned[chosen] = i
            sizes[i] += 1
            queues[i].append(chosen)
    assert np.all(assigned >= 0) and len(set(sizes)) <= 2
    return [np.flatnonzero(assigned == i) for i in range(groups)]


def make_group_network(net: TemporalNetwork, nodes: np.ndarray, index: int):
    """Reindex edges induced by nodes, keeping EXACT original year indices."""
    ids = np.asarray(nodes, dtype=np.int64)
    mapping = np.full(net.n, -1, dtype=np.int64)
    mapping[ids] = np.arange(len(ids), dtype=np.int64)
    remapped = mapping[net.events[:, 1:]]
    mask = np.all(remapped >= 0, axis=1)
    if not mask.any():
        raise ValueError(f"Group {index} has no internal events; use fewer groups")
    events = np.column_stack((net.events[mask, 0], remapped[mask]))
    features = net.node_features[ids] if net.node_features is not None else None
    subgraph = TemporalNetwork(events, len(ids), list(net.times), features,
                               name=f"{net.name}_group_{index}", directed=net.directed)
    return subgraph


def run_meta_episodes(net: TemporalNetwork, cfg: Config, output_dir,
                      options: ControllerOptions | None = None,
                      groups: int = 4, holdout_groups: int = 1,
                      compare_fixed: bool = True,
                      counterfactual_rollouts: bool = False):
    """Train DQN across node-disjoint histories, evaluate frozen policy on unseen nodes.

    Policy is never optimized on the heldout groups, even in their train years.
    Each group's GCN has a fresh initialization and its own rehearsal/EWC.
    Identical chronological time boundaries are used everywhere.
    """
    options = options or ControllerOptions()
    options.validate()
    if options.policy != "dqn":
        raise ValueError("C1 requires controller=dqn; fixed controls are evaluated automatically")
    if cfg.feature_track != "history":
        raise ValueError("C1 forbids potentially future-derived original node features")
    if not (1 <= holdout_groups < groups):
        raise ValueError("Holdout groups must be >=1 and less than total groups")
    max_t = len(net.times) if cfg.max_snapshots <= 0 else min(cfg.max_snapshots, len(net.times))
    train_end = max_t - options.test_last
    if train_end - cfg.start_time_index < cfg.warmup + 2:
        raise ValueError("Not enough chronological training periods for rewarded transitions")
    if counterfactual_rollouts and options.state_version != "c2":
        raise ValueError("Counterfactual action rollouts require state_version=c2")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    groups_nodes = historical_partition(net, groups, cfg.start_time_index, cfg.seed)
    group_graphs = [make_group_network(net, ids, i) for i, ids in enumerate(groups_nodes)]
    # Indices and assignment are determined by early historical structure only.
    # A holdout is pre-registered and never selected using future performance.
    # Pre-register held-out authors by seeded draw, NOT by test outcomes.
    permutation = np.random.default_rng(cfg.seed + 7129).permutation(groups).tolist()
    train_ids = sorted(permutation[:groups - holdout_groups])
    holdout_ids = sorted(permutation[groups - holdout_groups:])
    node_rows = []
    event_rows = []
    retained = 0
    for i, (ids, graph) in enumerate(zip(groups_nodes, group_graphs)):
        role = "meta_train" if i in train_ids else "meta_holdout"
        node_rows.extend({"original_node_id": int(n), "group": i, "role": role}
                         for n in ids)
        retained += len(graph.events)
        event_rows.append({"group": i, "role": role, "nodes": len(ids),
                           "events_all_years": len(graph.events),
                           "events_before_cutoff": int((graph.events[:, 0] < cfg.start_time_index).sum()),
                           "events_training_era": int((graph.events[:, 0] < train_end).sum()),
                           "events_frozen_test_era": int((graph.events[:, 0] >= train_end).sum())})
    def write_csv(name, rows):
        with (out / name).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
    write_csv("node_partitions.csv", node_rows)
    write_csv("group_audit.csv", event_rows)
    metadata = {
        "protocol": "C1 disjoint-node multi-episode meta-training / frozen holdout-policy transfer",
        "partition_only_years_before": str(net.times[cfg.start_time_index]),
        "history_cutoff_index": cfg.start_time_index,
        "train_years": [str(x) for x in net.times[cfg.start_time_index:train_end]],
        "frozen_test_years": [str(x) for x in net.times[train_end:max_t]],
        "train_groups": train_ids, "holdout_groups": holdout_ids,
        "total_original_events": len(net.events), "internal_events_kept": retained,
        "cross_group_events_omitted": len(net.events) - retained,
        "group_audit": event_rows,
        "config": asdict(cfg), "controller_options": asdict(options),
        "limitations": [
            "Node-disjoint groups are not necessarily statistically independent",
            "Partitioning removes between-group collaborations and changes evaluation population",
            "One chronological transition per year per training group, not per epoch",
            "Repeated updates over a small replay buffer do not create independent samples",
            "Full-scale DBLP results and controller cross-network generalization are not established",
        ],
    }
    (out / "meta_manifest.json").write_text(json.dumps(metadata, indent=2))
    torch.manual_seed(cfg.seed)
    controller = TemporalDQN(options, cfg.seed, cfg.device)
    episode_rows = []
    counterfactual_rows = []
    # For comparability, use identical per-group GCN seeds for DQN and fixed controls.
    for i in train_ids:
        print(f"[C1] meta-train group {i} ({len(groups_nodes[i])} nodes)", flush=True)
        episode_seed = cfg.seed + 1009 * i
        if counterfactual_rollouts:
            # Train-only off-policy data: four full policy trajectories, each
            # with its OWN independent GCN/EWC/replay state. No future-year
            # labels may enter a transition before its reward_year. All years
            # belong to the meta-training era and never the frozen test era.
            for action_name in ACTION_INDEX:
                rollout_options = replace(options, policy="fixed", fixed_action=action_name)
                transitions = []
                print(f"[C2] group {i}: collect train-only rollout {action_name}", flush=True)
                rollout_summary = run_temporal_controller(
                    group_graphs[i], replace(cfg, seed=episode_seed),
                    out / f"counterfactual_group_{i}" / action_name,
                    rollout_options, train_policy=False, transition_sink=transitions)
                if len(transitions) != rollout_summary["reward_transitions"]:
                    raise RuntimeError("Counterfactual collection lost temporal transitions")
                with (out / f"counterfactual_group_{i}" / action_name /
                      "controller_rewards.csv").open(newline="") as reward_file:
                    reward_records = list(csv.DictReader(reward_file))
                if len(reward_records) != len(transitions):
                    raise RuntimeError("Counterfactual reward audit differs from replay buffer")
                for (state, act, reward, next_state, terminal), record in zip(transitions, reward_records):
                    if record["reward_year"] not in metadata["train_years"]:
                        raise RuntimeError("Forbidden frozen-test reward in controller training")
                    controller.observe(state, act, reward, next_state, terminal)
                    counterfactual_rows.append({"group": i, "fixed_action": action_name,
                                                "action_year": record["action_year"],
                                                "reward_year": record["reward_year"],
                                                "transition": len(counterfactual_rows),
                                                "reward": float(reward), "terminal": terminal})
        result = run_temporal_controller(group_graphs[i], replace(cfg, seed=episode_seed),
                                         out / f"train_group_{i}", options,
                                         controller=controller, train_policy=True)
        episode_rows.append({**result, "group": i, "role": "meta_train",
                             "policy": "dqn"})
    training_updates = controller.update_count
    torch.save({"policy_q": controller.q.state_dict(),
                "target_q": controller.target.state_dict(),
                "controller_options": asdict(options),
                "trained_on_groups": train_ids}, out / "meta_controller.pt")
    for i in holdout_ids:
        print(f"[C1] held-out group {i} (policy FROZEN)", flush=True)
        episode_seed = cfg.seed + 1009 * i
        result = run_temporal_controller(group_graphs[i], replace(cfg, seed=episode_seed),
                                         out / f"holdout_group_{i}" / "dqn_frozen", options,
                                         controller=controller, train_policy=False,
                                         greedy_policy=True)
        assert controller.update_count == training_updates
        assert result["controller_gradient_updates"] == 0
        episode_rows.append({**result, "group": i, "role": "meta_holdout",
                             "policy": "dqn_frozen"})
        if compare_fixed:
            for action in ACTION_INDEX:
                fixed_opt = replace(options, policy="fixed", fixed_action=action)
                print(f"[C1] held-out group {i}: fixed {action}", flush=True)
                fixed_result = run_temporal_controller(
                    group_graphs[i], replace(cfg, seed=episode_seed),
                    out / f"holdout_group_{i}" / f"fixed_{action}", fixed_opt)
                episode_rows.append({**fixed_result, "group": i, "role": "meta_holdout",
                                     "policy": f"fixed_{action}"})
    write_csv("episode_summary.csv", episode_rows)
    if counterfactual_rows:
        write_csv("counterfactual_transition_audit.csv", counterfactual_rows)
    heldout = [row for row in episode_rows if row["role"] == "meta_holdout"]
    def aggregate(rows, key):
        values = [float(r[key]) for r in rows if r[key] is not None and np.isfinite(r[key])]
        return float(np.mean(values)) if values else None
    comparison = []
    for label in sorted(set(x["policy"] for x in heldout)):
        rows = [x for x in heldout if x["policy"] == label]
        comparison.append({"policy": label, "n_heldout_groups": len(rows),
                           "mean_test_ap": aggregate(rows, "test_mean_ap"),
                           "mean_test_roc_auc": aggregate(rows, "test_mean_roc_auc"),
                           "mean_prequential_train_ap": aggregate(rows, "train_prequential_mean_ap"),
                           "controller_grad_updates_on_holdout": sum(r["controller_gradient_updates"] for r in rows)})
    write_csv("holdout_comparison.csv", comparison)
    summary = {"train_groups": len(train_ids), "heldout_groups": len(holdout_ids),
               "controller_state_version": options.state_version,
               "counterfactual_rollouts": counterfactual_rollouts,
               "counterfactual_train_transitions": len(counterfactual_rows),
               "total_controller_training_transitions": len(counterfactual_rows) +
                   sum(r["reward_transitions"] for r in episode_rows if r["role"] == "meta_train"),
               "training_reward_transitions": sum(r["reward_transitions"] for r in episode_rows if r["role"] == "meta_train"),
               "controller_gradient_updates_train": training_updates,
               "controller_gradient_updates_holdout": 0,
               "internal_events_kept": retained,
               "cross_group_events_omitted": len(net.events) - retained,
               "holdout_comparison": comparison}
    (out / "meta_summary.json").write_text(json.dumps(summary, indent=2))
    return summary
