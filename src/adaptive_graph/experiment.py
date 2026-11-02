"""Prequential chronological experiments and reproducible artifact creation."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import csv
import json
import random
import time
import numpy as np
import torch
from .data import TemporalNetwork, create_task
from .features import FEATURE_NAMES, DIRECTED_FEATURE_NAMES
from .metrics import calculate_metrics
from .models import HybridGCN, prepare
from .continual import OnlineEWC, TaskReplay, train_gcn, consolidate_gcn
from .rl import QNetwork, TransitionMemory, train_dqn, consolidate_dqn, q_scores


@dataclass
class Config:
    mode: str = "gcn-rl-cl"    # baseline, gcn, gcn-rl, gcn-rl-cl
    feature_track: str = "history"  # history, original
    seed: int = 7
    epochs: int = 8
    rl_epochs: int = 10
    hidden: int = 32
    warmup: int = 2
    lr: float = 0.003
    ewc_lambda: float = 2.0
    replay_weight: float = 0.5
    memory_tasks: int = 3
    memory_transitions: int = 3000
    max_positives: int = 250
    negative_ratio: float = 1.0
    eval_fraction: float = 0.3
    max_snapshots: int = 0
    start_time_index: int = 1
    device: str = "cpu"
    gamma: float = 0.0
    epsilon: float = 0.25
    use_gcn: bool = True
    use_heuristics: bool = True
    use_ewc: bool = True
    use_replay: bool = True
    max_train_pairs: int = 512
    max_retention_tasks: int = 5
    save_tasks: bool = True


def _predict_gcn(model, pt, split="eval"):
    model.eval()
    ids = pt.subset(split)
    with torch.no_grad():
        return model(pt, ids).cpu().numpy()


def _predict_rl(model, q, pt, split="eval"):
    model.eval()
    ids = pt.subset(split)
    with torch.no_grad():
        states = model.states(pt, ids, detach=True)
        return q_scores(q, states)


def _metrics_on(pt, scores, split="eval"):
    idx = pt.task.eval_idx if split == "eval" else pt.task.train_idx
    return calculate_metrics(pt.task.labels[idx], scores, pt.task.candidates[idx])


def _json_write(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, default=str, allow_nan=True), encoding="utf-8")


def _save_rows(path, rows):
    if not rows:
        return
    cols = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("w", encoding="utf8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader(); writer.writerows(rows)


def run_experiment(net: TemporalNetwork, cfg: Config, output_dir: str | Path):
    if cfg.mode not in {"baseline", "gcn", "gcn-rl", "gcn-rl-cl"}:
        raise ValueError(f"Unknown mode: {cfg.mode}")
    if cfg.feature_track not in {"history", "original"}:
        raise ValueError("Invalid feature track")
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.set_num_threads(1)
    rng = np.random.default_rng(cfg.seed)
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    manifest = {"dataset": net.name, "nodes": net.n, "events": len(net.events),
                "times": [str(t) for t in net.times], "config": asdict(cfg),
                "temporal_protocol": "history < target year; evaluate before update",
                "edge_task": ("previously unseen directed edges, existing nodes only" if net.directed
                              else "previously unseen undirected edges, existing nodes only"),
                "negative_sampling": "balanced observed-node nonedges; fixed held-out per task",
                "directed": net.directed,
                "structural_feature_names": DIRECTED_FEATURE_NAMES if net.directed else FEATURE_NAMES,
                "original_feature_warning": ("OGB feature embeddings may use non-historical papers"
                                             if cfg.feature_track == "original" else None)}
    _json_write(out / "manifest.json", manifest)
    max_t = len(net.times) if cfg.max_snapshots <= 0 else min(len(net.times), cfg.max_snapshots)
    all_tasks = []
    baseline_rows = []
    baseline_history = {i: [] for i in range(6)}
    prediction_rows = []
    retention_rows = []
    loss_rows = []
    model = None
    q = None
    optimizer = None
    qopt = None
    gcn_ewc = OnlineEWC()
    rl_ewc = OnlineEWC()
    past_tasks = TaskReplay(cfg.memory_tasks)
    transitions = TransitionMemory(cfg.memory_transitions)
    past_holdouts = []
    best_ap = {}
    trained_steps = 0
    for ti in range(cfg.start_time_index, max_t):
        task = create_task(net, ti, seed=cfg.seed, max_positives=cfg.max_positives,
                           negative_ratio=cfg.negative_ratio, eval_fraction=cfg.eval_fraction)
        if not len(task.train_idx) or not len(task.eval_idx) or task.new_link_count < 2:
            all_tasks.append({"time": str(task.time_value), "status": "skipped_insufficient_candidates",
                              "novel": task.new_link_count, "repeat": task.repeat_contact_count,
                              "cold_start": task.cold_start_count})
            continue
        pt = prepare(task, net.n, net.node_features, cfg.feature_track, cfg.device)
        if cfg.save_tasks:
            task_dir = out / "task_samples"
            task_dir.mkdir(exist_ok=True)
            np.savez_compressed(task_dir / f"time_index_{ti}.npz",
                                candidates=task.candidates, labels=task.labels,
                                train_idx=task.train_idx, eval_idx=task.eval_idx,
                                history_time_exclusive=ti,
                                # Historical edges are reconstructed from events.
                                history_edge_count=len(task.historical_edges))
        all_tasks.append({"time": str(task.time_value), "status": "used",
                          "history_edges": len(task.historical_edges),
                          "history_nodes": task.history_node_count,
                          "novel": task.new_link_count,
                          "repeat": task.repeat_contact_count,
                          "cold_start": task.cold_start_count,
                          "train_candidates": len(task.train_idx),
                          "heldout_candidates": len(task.eval_idx)})
        # Baselines use only historical graph features; no label data in scoring.
        baseline_labels = (("D2Hop", "SharedTargets", "SharedSources", "ReverseEdge", "OutDegree", "InDegree")
                           if net.directed else ("CN", "Jaccard", "Adamic-Adar", "Katz-2/3", "PageRank", "PA"))
        for col, baseline in enumerate(baseline_labels):
            ids = task.eval_idx
            score = pt.structural[ids, col].cpu().numpy()
            # Binary decision cutoffs are fitted ONLY on earlier revealed
            # training pairs. Ranking/AP/AUC are invariant to this shift.
            past = baseline_history[col]
            if past:
                values = np.concatenate([item[0] for item in past])
                truths = np.concatenate([item[1] for item in past])
                from sklearn.metrics import f1_score
                options = np.unique(np.quantile(values, np.linspace(0, 1, 31)))
                threshold = float(max(options, key=lambda th: f1_score(
                    truths, (values >= th).astype(int), zero_division=0)))
            else:
                threshold = 1e-6
            met = calculate_metrics(task.labels[ids], score - threshold, task.candidates[ids])
            baseline_rows.append({"time": str(task.time_value), "method": baseline,
                                  "historical_threshold": threshold, **met})
        for col in range(6):
            baseline_history[col].append((pt.structural[task.train_idx, col].cpu().numpy(),
                                           task.labels[task.train_idx]))
        if cfg.mode == "baseline":
            continue
        if model is None:
            model = HybridGCN(pt.x.shape[1], cfg.hidden,
                              use_gcn=cfg.use_gcn, use_heuristics=cfg.use_heuristics).to(cfg.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=1e-5)
            if cfg.mode in {"gcn-rl", "gcn-rl-cl"}:
                q = QNetwork(model.state_dim, cfg.hidden).to(cfg.device)
                qopt = torch.optim.Adam(q.parameters(), lr=cfg.lr)
        # Strict walk-forward: evaluate target BEFORE any of its labels train either model.
        if trained_steps >= cfg.warmup:
            gc_scores = _predict_gcn(model, pt)
            prediction_rows.append({"time": str(task.time_value), "method": "GCN",
                                     **_metrics_on(pt, gc_scores)})
            if q is not None and len(transitions):
                rl_scores = _predict_rl(model, q, pt)
                prediction_rows.append({"time": str(task.time_value), "method": "GCN+Q",
                                         **_metrics_on(pt, rl_scores)})
        use_cl = cfg.mode == "gcn-rl-cl"
        loss = train_gcn(model, pt, optimizer, epochs=cfg.epochs,
                         replay=past_tasks if use_cl and cfg.use_replay else None,
                         ewc=gcn_ewc if use_cl and cfg.use_ewc else None,
                         ewc_lambda=cfg.ewc_lambda, rng=rng,
                         replay_weight=cfg.replay_weight, max_train_pairs=cfg.max_train_pairs)
        rloss = None
        fresh_transitions = []
        if q is not None:
            model.eval()
            with torch.no_grad():
                ids = pt.subset("train")
                state = model.states(pt, ids, detach=True)
            rloss, fresh_transitions = train_dqn(
                q, qopt, state, pt.labels[ids], transitions, rng,
                ewc=rl_ewc if use_cl and cfg.use_ewc else None, ewc_lambda=cfg.ewc_lambda,
                epochs=cfg.rl_epochs, epsilon=cfg.epsilon, gamma=cfg.gamma,
                replay=use_cl and cfg.use_replay)
        if use_cl:
            if cfg.use_ewc:
                consolidate_gcn(model, pt, gcn_ewc)
                if fresh_transitions:
                    consolidate_dqn(q, fresh_transitions, rl_ewc, gamma=cfg.gamma)
            if cfg.use_replay:
                past_tasks.add(pt)
        trained_steps += 1
        loss_rows.append({"time": str(task.time_value), "gcn_loss": loss, "q_loss": rloss})
        # Held-out previous task examples stay frozen: an independent forgetting probe.
        past_holdouts.append(pt)
        if cfg.max_retention_tasks >= 2 and len(past_holdouts) > cfg.max_retention_tasks:
            # Keep earliest anchor and recent tasks, with bounded graph memory.
            past_holdouts = [past_holdouts[0]] + past_holdouts[-(cfg.max_retention_tasks - 1):]
        for old in past_holdouts:
            scores = (_predict_rl(model, q, old) if q is not None
                      else _predict_gcn(model, old))
            met = _metrics_on(old, scores)
            key = old.task.time_index
            ap = met.get("ap", float("nan"))
            if np.isfinite(ap):
                best_ap[key] = max(ap, best_ap.get(key, float("-inf")))
            retention_rows.append({"after_time": str(task.time_value),
                                   "task_time": str(old.task.time_value),
                                   "ap": ap,
                                   "forgetting_ap": max(0, best_ap.get(key, ap) - ap),
                                   "roc_auc": met.get("roc_auc")})
        print(f"[{cfg.mode}] time={task.time_value} train_pairs={len(task.train_idx)} "
              f"history_edges={len(task.historical_edges)} train_loss={loss:.4f}", flush=True)
    _save_rows(out / "task_audit.csv", all_tasks)
    _save_rows(out / "baselines.csv", baseline_rows)
    _save_rows(out / "predictions.csv", prediction_rows)
    _save_rows(out / "retention.csv", retention_rows)
    _save_rows(out / "losses.csv", loss_rows)
    summary = {"dataset": net.name, "mode": cfg.mode, "track": cfg.feature_track,
               "seed": cfg.seed, "trained_tasks": trained_steps,
               "prequential_evaluations": len({r['time'] for r in prediction_rows}),
               "runtime_seconds": round(time.perf_counter() - started, 2),
               "baseline_mean_ap": {}, "model_mean_ap": {},
               "final_mean_prior_task_forgetting_ap": None}
    # Compare AP over identical post-warmup target times, even for
    # baseline-only runs (which do not instantiate a trainable model).
    eligible_times = [row["time"] for row in all_tasks if row["status"] == "used"]
    comparable_times = set(eligible_times[cfg.warmup:])
    fair_baseline_rows = [row for row in baseline_rows if row["time"] in comparable_times]
    for kind, rows, key in (("baseline", fair_baseline_rows, "baseline_mean_ap"),
                            ("model", prediction_rows, "model_mean_ap")):
        for method in sorted({x["method"] for x in rows}):
            ap = [x["ap"] for x in rows if x["method"] == method and np.isfinite(x["ap"])]
            summary[key][method] = round(float(np.mean(ap)), 4) if ap else None
    if retention_rows:
        last_time = retention_rows[-1]["after_time"]
        r = [x["forgetting_ap"] for x in retention_rows
             if x["after_time"] == last_time and x["task_time"] != last_time]
        summary["final_mean_prior_task_forgetting_ap"] = float(np.mean(r)) if r else None
    _json_write(out / "summary.json", summary)
    if model is not None:
        torch.save({"model": model.state_dict(), "config": asdict(cfg),
                    "input_dim": next(iter(past_holdouts)).x.shape[1],
                    "state_dim": model.state_dim,
                    "q": q.state_dict() if q is not None else None}, out / "checkpoint.pt")
    return summary
