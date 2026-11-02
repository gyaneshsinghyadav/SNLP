"""Console entrypoint: generate-demo, audit, run, suite, report."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .data import TemporalNetwork, synthetic_network
from .experiment import Config, run_experiment
from .report import summarize, summarize_suite


def _load(args):
    if args.dataset == "demo":
        net = synthetic_network(n=args.demo_nodes, periods=args.demo_periods, seed=args.seed)
    elif args.dataset == "csv":
        if not args.csv:
            raise ValueError("--csv is required with --dataset csv")
        net = TemporalNetwork.from_csv(args.csv, args.source_column,
                                       args.target_column, args.time_column, time_bucket=args.time_bucket,
                                       directed=args.directed)
    else:
        from .ogb_adapter import load_ogbl_collab
        net = load_ogbl_collab(args.ogb_root, args.trust_ogb_pickle)
    if args.max_nodes:
        net = net.induced_subset(args.max_nodes, seed=args.seed,
                                 selection_history_idx=args.start_time_index)
    return net


def _data_args(p):
    p.add_argument("--dataset", choices=["demo", "csv", "ogbl-collab"], default="demo")
    p.add_argument("--csv", type=str)
    p.add_argument("--source-column", default="source")
    p.add_argument("--target-column", default="target")
    p.add_argument("--time-column", default="timestamp")
    p.add_argument("--time-bucket", choices=["none", "year", "quarter", "month", "week", "day"], default="none")
    p.add_argument("--directed", action="store_true", help="Keep source->target orientation for prediction")
    p.add_argument("--demo-nodes", type=int, default=90)
    p.add_argument("--demo-periods", type=int, default=8)
    p.add_argument("--ogb-root", default="datasets/ogb")
    p.add_argument("--trust-ogb-pickle", action="store_true",
                   help="UNSAFE on untrusted files; allow legacy OGB torch.load artifacts")
    p.add_argument("--max-nodes", type=int, default=0,
                   help="Induced history-seeded subgraph for quick tests. 0 = all nodes")
    p.add_argument("--start-time-index", type=int, default=1,
                   help="First eligible task; also the exclusive cutoff for subgraph selection")
    p.add_argument("--seed", type=int, default=7)


def _train_args(p):
    p.add_argument("--mode", choices=["baseline", "gcn", "gcn-rl", "gcn-rl-cl"],
                   default="gcn-rl-cl")
    p.add_argument("--feature-track", choices=["history", "original"], default="history")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--rl-epochs", type=int, default=10)
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--lr", type=float, default=0.003)
    p.add_argument("--ewc-lambda", type=float, default=2.0)
    p.add_argument("--replay-weight", type=float, default=0.5)
    p.add_argument("--memory-tasks", type=int, default=3)
    p.add_argument("--memory-transitions", type=int, default=3000)
    p.add_argument("--max-positives", type=int, default=250)
    p.add_argument("--negative-ratio", type=float, default=1)
    p.add_argument("--eval-fraction", type=float, default=0.30)
    p.add_argument("--max-snapshots", type=int, default=0)
    p.add_argument("--max-train-pairs", type=int, default=512)
    p.add_argument("--max-retention-tasks", type=int, default=5)
    p.add_argument("--no-save-tasks", action="store_true")
    p.add_argument("--device", default="cpu", help="cpu, mps or cuda")
    p.add_argument("--gamma", type=float, default=0)
    p.add_argument("--epsilon", type=float, default=0.25)
    p.add_argument("--no-gcn", action="store_true")
    p.add_argument("--no-heuristics", action="store_true")
    p.add_argument("--no-ewc", action="store_true")
    p.add_argument("--no-replay", action="store_true")
    p.add_argument("--output", default="results/demo")


def _config(args, **replacements):
    keys = Config.__dataclass_fields__.keys()
    opts = {k: getattr(args, k) for k in keys if hasattr(args, k)}
    opts["use_gcn"] = not args.no_gcn
    opts["use_heuristics"] = not args.no_heuristics
    opts["use_ewc"] = not args.no_ewc
    opts["use_replay"] = not args.no_replay
    opts["save_tasks"] = not args.no_save_tasks
    opts.update(replacements)
    return Config(**opts)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="adaptive-graph",
                                     description="Leakage-safe GCN-RL-CL dynamic link prediction")
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("generate-demo", help="Save synthetic events to a CSV")
    demo.add_argument("--nodes", type=int, default=90)
    demo.add_argument("--periods", type=int, default=8)
    demo.add_argument("--seed", type=int, default=7)
    demo.add_argument("--output", default="examples/synthetic_edges.csv")
    audit = sub.add_parser("audit", help="Print temporal snapshot statistics")
    _data_args(audit)
    run = sub.add_parser("run", help="Run one chronological experiment")
    _data_args(run); _train_args(run)
    suite = sub.add_parser("suite", help="Compare baseline, GCN, RL and continual learning")
    _data_args(suite); _train_args(suite)
    sweep = sub.add_parser("sweep", help="Repeat one model over multiple seeds with confidence intervals")
    _data_args(sweep); _train_args(sweep)
    sweep.add_argument("--seeds", default="7,13,42,101,202")
    def add_temporal_args(p):
        _data_args(p); _train_args(p)
        # Candidate classification in the original `run` has gamma=0.
        # Adaptation decisions in temporal-rl have real year-to-year transitions.
        p.set_defaults(gamma=0.9, output="results/temporal_rl")
        p.add_argument("--controller", choices=["dqn", "fixed", "random"], default="dqn")
        p.add_argument("--fixed-action", choices=["plastic", "balanced", "ewc_focus", "replay_focus"],
                       default="balanced")
        p.add_argument("--test-last", type=int, default=2,
                       help="Last N periods frozen for final evaluation")
        p.add_argument("--controller-val-fraction", type=float, default=0.25)
        p.add_argument("--reward-forgetting-lambda", type=float, default=0.25)
        p.add_argument("--reward-cost-lambda", type=float, default=0.001)
        p.add_argument("--controller-lr", type=float, default=0.001)
        p.add_argument("--controller-updates", type=int, default=8)

    temporal = sub.add_parser("temporal-rl", help="Learn action-dependent temporal GCN adaptation policy")
    add_temporal_args(temporal)
    meta = sub.add_parser("temporal-rl-meta", help="C1: train DQN over node-disjoint histories; test frozen policy on unseen authors")
    add_temporal_args(meta)
    meta.add_argument("--communities", type=int, default=4, help="Number of history-defined node-disjoint graph histories")
    meta.add_argument("--holdout-communities", type=int, default=1, help="Communities unseen by controller training")
    meta.add_argument("--no-fixed-comparison", action="store_true", help="Skip four fixed-policy heldout comparisons")
    c2 = sub.add_parser("temporal-rl-c2", help="C2: history-only state, action diagnostics, and train-only off-policy rollouts")
    add_temporal_args(c2)
    c2.add_argument("--communities", type=int, default=3)
    c2.add_argument("--holdout-communities", type=int, default=1)
    c2.add_argument("--no-fixed-comparison", action="store_true")
    c2.add_argument("--no-counterfactual-rollouts", action="store_true",
                    help="C2 ablation: disable off-policy fixed-strategy training trajectories")
    c2.set_defaults(output="results/dblp_c2")
    c2_sweep = sub.add_parser("temporal-rl-c2-sweep", help="C2: multi-seed paired holdout comparisons and optional rollout ablation")
    add_temporal_args(c2_sweep)
    c2_sweep.add_argument("--communities", type=int, default=3)
    c2_sweep.add_argument("--holdout-communities", type=int, default=1)
    c2_sweep.add_argument("--seeds", default="7,13,42")
    c2_sweep.add_argument("--include-ablation", action="store_true",
                          help="Repeat C2 without fixed-policy off-policy rollouts")
    c2_sweep.set_defaults(output="results/dblp_c2_sweep")
    c3 = sub.add_parser("temporal-rl-c3", help="C3: matched action effects and training-only reward sensitivity")
    add_temporal_args(c3)
    c3.add_argument("--communities", type=int, default=3)
    c3.add_argument("--holdout-communities", type=int, default=1)
    c3.add_argument("--forgetting-grid", default="0,0.25,1")
    c3.add_argument("--cost-grid", default="0,0.001")
    c3.set_defaults(output="results/dblp_c3")
    c3_sweep = sub.add_parser("temporal-rl-c3-sweep", help="C3: matched action study across seeds, no final-test tuning")
    add_temporal_args(c3_sweep)
    c3_sweep.add_argument("--communities", type=int, default=3)
    c3_sweep.add_argument("--holdout-communities", type=int, default=1)
    c3_sweep.add_argument("--forgetting-grid", default="0,0.25,1")
    c3_sweep.add_argument("--cost-grid", default="0,0.001")
    c3_sweep.add_argument("--seeds", default="7,13,42")
    c3_sweep.set_defaults(output="results/dblp_c3_sweep")
    c33 = sub.add_parser("temporal-rl-c33", help="C3.3: contextual ridge vs DQN vs fixed policies on frozen author holdout")
    add_temporal_args(c33)
    c33.add_argument("--communities", type=int, default=3)
    c33.add_argument("--holdout-communities", type=int, default=1)
    c33.add_argument("--ridge-alphas", default="0.01,0.1,1,10,100")
    c33.add_argument("--no-dqn-offpolicy", action="store_true")
    c33.set_defaults(output="results/dblp_c33_seed7")
    c33_sweep = sub.add_parser("temporal-rl-c33-sweep", help="C3.3: multiple paired seeds for contextual vs DQN vs fixed policies")
    add_temporal_args(c33_sweep)
    c33_sweep.add_argument("--communities", type=int, default=3)
    c33_sweep.add_argument("--holdout-communities", type=int, default=1)
    c33_sweep.add_argument("--ridge-alphas", default="0.01,0.1,1,10,100")
    c33_sweep.add_argument("--no-dqn-offpolicy", action="store_true")
    c33_sweep.add_argument("--seeds", default="7,13,42")
    c33_sweep.set_defaults(output="results/dblp_c33_sweep")
    temporal_suite = sub.add_parser("temporal-rl-suite", help="Compare temporal DQN against fixed adaptation policies over seeds")
    add_temporal_args(temporal_suite)
    temporal_suite.add_argument("--seeds", default="7,13,42")
    temporal_suite.add_argument("--policies", default="dqn,fixed:balanced,fixed:plastic,fixed:ewc_focus,fixed:replay_focus")
    c34_lock = sub.add_parser('temporal-rl-c34-lock', help='C3.4: preregister fresh temporal evaluation BEFORE any test scoring')
    c34_lock.add_argument('--csv', required=True)
    c34_lock.add_argument('--output', default='results/c34_protocol_lock.json')
    c34_lock.add_argument('--source', required=True, help='Pinned dataset source URL or archive identifier')
    c34_lock.add_argument('--seen-through', type=int, default=2010, help='Last previously examined final-test year')
    c34_lock.add_argument('--seeds', default='7,13,42,101,202')
    c34_lock.add_argument('--max-nodes', type=int, default=5000)
    c34_lock.add_argument('--start-time-index', type=int, default=3)
    c34_eval = sub.add_parser('temporal-rl-c34-evaluate', help='C3.4: execute preregistered seven-policy test exactly once')
    c34_eval.add_argument('--csv', required=True)
    c34_eval.add_argument('--lock', required=True)
    c34_eval.add_argument('--device', default='cpu')
    c34_eval.add_argument('--output', default='results/dblp_c34_final')
    # A separate experimental architecture. This does not change the frozen C3.4 code path.
    tgat = sub.add_parser("tgat-rl-cl", help="TGAT-style time-aware encoder with DQN, EWC and replay")
    tgat.add_argument("--csv", required=True, help="Raw timestamped interactions with source,target,timestamp[,rating]")
    tgat.add_argument("--time-bucket", choices=["quarter", "month", "year"], default="quarter")
    tgat.add_argument("--rating-column", default="rating")
    tgat.add_argument("--undirected", action="store_true", help="Undirected coauthorship: symmetric neighbor history, candidate pairs and predictions")
    tgat.add_argument("--controller", choices=["dqn", "fixed", "random"], default="fixed")
    tgat.add_argument("--fixed-action", default=None,
                      help="Fixed action name: E1/E2 replay_focus or E3 adaptive_full, etc.")
    tgat.add_argument("--action-space", choices=["legacy", "e3", "e31", "e32", "e5"], default="legacy",
                      help="E3.1: retention-aware five-action DQN, replay coverage and guard")
    tgat.add_argument("--neighbors", type=int, default=12)
    tgat.add_argument("--heads", type=int, default=4)
    tgat.add_argument("--hidden", type=int, default=32)
    tgat.add_argument("--start-time-index", type=int, default=3)
    tgat.add_argument("--test-last", type=int, default=2)
    tgat.add_argument("--warmup", type=int, default=1)
    tgat.add_argument("--epochs", type=int, default=3)
    tgat.add_argument("--gamma", type=float, default=0.9)
    tgat.add_argument("--controller-updates", type=int, default=2)
    tgat.add_argument("--controller-val-fraction", type=float, default=0.25)
    tgat.add_argument("--reward-forgetting-lambda", type=float, default=None,
                      help="Mean forgetting penalty: default .25 for E3, .75 for E3.1")
    tgat.add_argument("--reward-peak-forgetting-lambda", type=float, default=None,
                      help="Peak old-task forgetting penalty: default .25 for E3.1, 0 otherwise")
    tgat.add_argument("--retention-guard-threshold", type=float, default=0.02,
                      help="E3.1: if previous historical AP drop >= threshold, DQN must use full replay")
    tgat.add_argument("--counterfactual-actions", type=int, default=4,
                      help="E5: allowed actions probed from matched training-era model states (2-6)")
    tgat.add_argument("--counterfactual-updates", type=int, default=4,
                      help="E5: auxiliary critic updates after action decision")
    tgat.add_argument("--counterfactual-weight", type=float, default=0.5,
                      help="E5: weight of learned immediate-reward critic in action selection")
    tgat.add_argument("--gradient-probe-pairs", type=int, default=16,
                      help="E3.2: bounded current/history gradient probe pairs")
    tgat.add_argument("--conflict-guard-threshold", type=float, default=-0.05,
                      help="E3.2: forbid light replay if pre-action cosine < threshold")
    tgat.add_argument("--min-period-examples", type=int, default=None,
                      help="E3.1: minimum examples reserved per historical training period (default 4)")
    tgat.add_argument("--reward-cost-lambda", type=float, default=0)
    tgat.add_argument("--max-positives", type=int, default=200)
    tgat.add_argument("--max-train-pairs", type=int, default=512)
    tgat.add_argument("--lr", type=float, default=0.003)
    tgat.add_argument("--ewc-lambda", type=float, default=2.0)
    tgat.add_argument("--replay-weight", type=float, default=0.5)
    tgat.add_argument("--memory-tasks", type=int, default=3)
    tgat.add_argument("--replay-mode", choices=["task", "uniform", "adaptive"], default="task",
                      help="E1 task rehearsal (default), or E2 matched-budget uniform/adaptive sample replay")
    tgat.add_argument("--replay-capacity", type=int, default=256, help="E2 stored training pairs")
    tgat.add_argument("--replay-batch-size", type=int, default=64, help="E2 replayed training pairs per epoch")
    tgat.add_argument("--replay-loss-weight", type=float, default=0.45)
    tgat.add_argument("--replay-diversity-weight", type=float, default=0.35)
    tgat.add_argument("--replay-recency-weight", type=float, default=0.20)
    tgat.add_argument("--replay-recency-half-life", type=float, default=3.0)
    tgat.add_argument("--seed", type=int, default=7)
    tgat.add_argument("--epsilon", type=float, default=0.25)
    tgat.add_argument("--device", default="cpu")
    tgat.add_argument("--output", default="results/tgat_bitcoin_pilot")
    report = sub.add_parser("report", help="Summarize run and create plots")
    report.add_argument("--run", default="results/demo")
    args = parser.parse_args(argv)
    if args.command == "tgat-rl-cl":
        from .tgat import TemporalEventStream, TemporalAttentionLinkPredictor
        from .temporal_control import ControllerOptions, run_temporal_controller
        stream = TemporalEventStream.from_csv(args.csv, bucket=args.time_bucket,
                                              rating_column=args.rating_column,
                                              undirected=args.undirected)
        if args.hidden % args.heads:
            raise ValueError("--hidden must be divisible by --heads")
        config = Config(seed=args.seed, mode="gcn-rl-cl", epochs=args.epochs,
                        hidden=args.hidden, warmup=args.warmup, lr=args.lr,
                        ewc_lambda=args.ewc_lambda, replay_weight=args.replay_weight,
                        memory_tasks=args.memory_tasks, start_time_index=args.start_time_index,
                        max_positives=args.max_positives, max_train_pairs=args.max_train_pairs,
                        gamma=args.gamma, epsilon=args.epsilon, device=args.device)
        if args.action_space in {"e3", "e31", "e32", "e5"} and args.replay_mode == "task":
            parser.error("--action-space e3/e31/e32 requires --replay-mode adaptive or uniform")
        fixed = args.fixed_action or ("adaptive_full" if args.action_space in {"e3", "e31", "e32", "e5"}
                                       else "replay_focus")
        options = ControllerOptions(policy=args.controller, fixed_action=fixed,
                                    action_space=args.action_space,
                                    state_version=args.action_space if args.action_space in {"e3", "e31", "e32", "e5"} else "c2",
                                    gamma=args.gamma,
                                    controller_updates=args.controller_updates,
                                    controller_val_fraction=args.controller_val_fraction,
                                    test_last=args.test_last,
                                    forgetting_lambda=(args.reward_forgetting_lambda if
                                        args.reward_forgetting_lambda is not None else
                                        (0.75 if args.action_space in {"e31", "e32", "e5"} else 0.25)),
                                    peak_forgetting_lambda=(args.reward_peak_forgetting_lambda if
                                        args.reward_peak_forgetting_lambda is not None else
                                        (0.25 if args.action_space in {"e31", "e32", "e5"} else 0.0)),
                                    retention_guard_threshold=args.retention_guard_threshold,
                                    conflict_guard_threshold=args.conflict_guard_threshold,
                                    gradient_probe_pairs=args.gradient_probe_pairs,
                                    e5_counterfactual_actions=args.counterfactual_actions,
                                    e5_counterfactual_updates=args.counterfactual_updates,
                                    e5_counterfactual_weight=args.counterfactual_weight,
                                    e5_full_replay_batch=args.replay_batch_size,
                                    cost_lambda=args.reward_cost_lambda)
        def prep(task):
            return stream.prepare(task, neighbors=args.neighbors, device=args.device)
        def build(pt):
            return TemporalAttentionLinkPredictor(pt.x.shape[1], hidden=args.hidden,
                                                   heads=args.heads)
        replay_options = None if args.replay_mode == "task" else {
            "mode": args.replay_mode,
            "capacity": args.replay_capacity,
            "batch_size": args.replay_batch_size,
            "loss_weight": args.replay_loss_weight,
            "diversity_weight": args.replay_diversity_weight,
            "recency_weight": args.replay_recency_weight,
            "recency_half_life": args.replay_recency_half_life,
            "min_period_examples": (args.min_period_examples if args.min_period_examples is not None
                                    else (4 if args.action_space in {"e31", "e32", "e5"} else 0)),
        }
        summary = run_temporal_controller(stream.network, config, args.output, options,
                                          prepare_builder=prep, model_builder=build,
                                          encoder_label="one_hop_tgat_style",
                                          replay_options=replay_options)
        print(json.dumps(summary, indent=2))
        return
    if args.command == 'temporal-rl-c34-lock':
        from .c34_validation import FrozenProtocol, create_lock
        seeds = tuple(int(x.strip()) for x in args.seeds.split(',') if x.strip())
        lock = create_lock(args.csv, args.output, source=args.source,
                           seen_through=args.seen_through,
                           protocol=FrozenProtocol(seeds=seeds, max_nodes=args.max_nodes,
                                                   start_time_index=args.start_time_index))
        print(json.dumps({'protocol_lock': args.output, 'final_test_years': lock['final_test_years'],
                          'seeds': lock['protocol']['seeds'], 'source': lock['source']}, indent=2))
        return
    if args.command == 'temporal-rl-c34-evaluate':
        from .c34_validation import evaluate_locked
        result = evaluate_locked(args.csv, args.lock, args.output, args.device)
        print(json.dumps(result, indent=2))
        return
    if args.command == "generate-demo":
        net = synthetic_network(args.nodes, args.periods, args.seed)
        output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(net.events, columns=["timestamp", "source", "target"])
        df.to_csv(output, index=False)
        print(f"Wrote {len(df)} events to {output}")
        return
    if args.command == "report":
        summarize(args.run); return
    net = _load(args)
    if args.command == "audit":
        rows = []
        observed = set()
        for i, t in enumerate(net.times):
            e = net.get_current(i)
            from .data import pair_keys
            keys = set(pair_keys(e, net.n, net.directed).tolist())
            rows.append({"time": str(t), "edges": len(e),
                         "new_pairs": len(keys - observed),
                         "repeat_pairs": len(keys & observed),
                         "cumulative_unique": len(keys | observed)})
            observed.update(keys)
        print(f"Dataset={net.name} directed={net.directed} nodes={net.n} events={len(net.events)}")
        print(pd.DataFrame(rows).to_string(index=False))
        return
    if args.command in {"temporal-rl-c33", "temporal-rl-c33-sweep"}:
        from .c33_policy import run_c33
        from .temporal_control import ControllerOptions
        alphas = [float(x.strip()) for x in args.ridge_alphas.split(",") if x.strip()]
        opts = ControllerOptions(policy="dqn", state_version="c2", gamma=args.gamma,
                                 test_last=args.test_last,
                                 controller_val_fraction=args.controller_val_fraction,
                                 forgetting_lambda=args.reward_forgetting_lambda,
                                 cost_lambda=args.reward_cost_lambda,
                                 controller_lr=args.controller_lr,
                                 controller_updates=args.controller_updates)
        seeds = ([int(v.strip()) for v in args.seeds.split(",") if v.strip()]
                 if args.command.endswith("sweep") else [args.seed])
        if not seeds or len(set(seeds)) != len(seeds):
            raise ValueError("--seeds requires unique integer values")
        results = []
        out = Path(args.output)
        for seed in seeds:
            print(f"[C3.3] research pilot seed {seed}", flush=True)
            seed_net = _load(argparse.Namespace(**{**vars(args), "seed": seed})) if args.max_nodes else net
            folder = out / f"seed_{seed}" if args.command.endswith("sweep") else out
            result = run_c33(seed_net, _config(args, seed=seed), folder, opts,
                             groups=args.communities, holdout_groups=args.holdout_communities,
                             ridge_alphas=alphas,
                             dqn_offpolicy=not args.no_dqn_offpolicy)
            results.extend(result["heldout_comparison"])
        if args.command.endswith("sweep"):
            out.mkdir(parents=True, exist_ok=True)
            df = pd.DataFrame(results)
            df.to_csv(out / "c33_seed_comparison.csv", index=False)
            aggregate = df.groupby("policy", as_index=False).agg(
                mean_test_ap=("test_mean_ap", "mean"),
                sd_test_ap=("test_mean_ap", "std"),
                mean_test_roc_auc=("test_mean_roc_auc", "mean"),
                mean_historical_ap_drop=("mean_historical_ap_drop", "mean"),
                n_heldout_runs=("test_mean_ap", "count"))
            aggregate.to_csv(out / "c33_summary.csv", index=False)
            print(aggregate.to_string(index=False))
            return aggregate.to_dict(orient="records")
        print(json.dumps(result, indent=2))
        return result
    if args.command == "temporal-rl-c3-sweep":
        from .c3_diagnostics import run_c3_action_study
        from .temporal_control import ControllerOptions
        seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
        if not seeds or len(set(seeds)) != len(seeds):
            raise ValueError("--seeds requires unique integer values")
        weights_f = [float(x) for x in args.forgetting_grid.split(",") if x.strip()]
        weights_c = [float(x) for x in args.cost_grid.split(",") if x.strip()]
        opts = ControllerOptions(state_version="c2", test_last=args.test_last,
                                 controller_val_fraction=args.controller_val_fraction)
        out = Path(args.output)
        joined = []
        joined_grid = []
        for seed in seeds:
            print(f"[C3] analyzing seed {seed}", flush=True)
            net_for_seed = (_load(argparse.Namespace(**{**vars(args), "seed": seed}))
                            if args.max_nodes else net)
            run_dir = out / f"seed_{seed}"
            run_c3_action_study(net_for_seed, _config(args, seed=seed), run_dir, opts,
                                groups=args.communities, holdout_groups=args.holdout_communities,
                                forgetting_weights=weights_f, cost_weights=weights_c)
            detail = pd.read_csv(run_dir / "matched_action_outcomes.csv")
            detail.insert(0, "seed", seed)
            joined.append(detail)
            grid = pd.read_csv(run_dir / "reward_sensitivity.csv")
            grid.insert(0, "seed", seed)
            joined_grid.append(grid)
        out.mkdir(parents=True, exist_ok=True)
        details = pd.concat(joined, ignore_index=True)
        details.to_csv(out / "c3_seed_outcomes.csv", index=False)
        grids = pd.concat(joined_grid, ignore_index=True)
        grids.to_csv(out / "c3_seed_grid.csv", index=False)
        summary = details.groupby("action", as_index=False).agg(
            mean_delta_ap=("delta_ap", "mean"),
            mean_forgetting_ap=("forgetting_ap", "mean"),
            matched_outcomes=("delta_ap", "count"))
        summary["n_seeds"] = len(seeds)
        summary.to_csv(out / "c3_action_summary.csv", index=False)
        grid_summary = grids.groupby(["forgetting_lambda", "cost_lambda", "action"],
                                      as_index=False).agg(
            mean_reward=("mean_reward", "mean"),
            sd_across_seeds=("mean_reward", "std"),
            mean_win_count=("win_count", "mean"))
        grid_summary.to_csv(out / "c3_grid_summary.csv", index=False)
        print(summary.to_string(index=False))
        return summary.to_dict(orient="records")
    if args.command == "temporal-rl-c3":
        from .c3_diagnostics import run_c3_action_study
        from .temporal_control import ControllerOptions
        def parse_grid(raw):
            vals = [float(x.strip()) for x in raw.split(",") if x.strip()]
            if not vals or not all(np.isfinite(x) and x >= 0 for x in vals):
                raise ValueError("Reward weight grids require finite nonnegative numbers")
            return vals
        options = ControllerOptions(state_version="c2", test_last=args.test_last,
                                    controller_val_fraction=args.controller_val_fraction,
                                    forgetting_lambda=args.reward_forgetting_lambda,
                                    cost_lambda=args.reward_cost_lambda)
        result = run_c3_action_study(net, _config(args), args.output, options,
                                     groups=args.communities, holdout_groups=args.holdout_communities,
                                     forgetting_weights=parse_grid(args.forgetting_grid),
                                     cost_weights=parse_grid(args.cost_grid))
        print(json.dumps(result, indent=2))
        return result
    if args.command in {"temporal-rl", "temporal-rl-suite", "temporal-rl-meta", "temporal-rl-c2", "temporal-rl-c2-sweep"}:
        from .temporal_control import ControllerOptions, run_temporal_controller, ACTION_INDEX
        def get_options(policy, action):
            return ControllerOptions(
                policy=policy, fixed_action=action,
                gamma=args.gamma, test_last=args.test_last,
                controller_val_fraction=args.controller_val_fraction,
                forgetting_lambda=args.reward_forgetting_lambda,
                cost_lambda=args.reward_cost_lambda,
                controller_lr=args.controller_lr,
                controller_updates=args.controller_updates,
                state_version="c2" if args.command in {"temporal-rl-c2", "temporal-rl-c2-sweep"} else "v1")
        if args.command == "temporal-rl-c2-sweep":
            from .meta_environment import run_meta_episodes
            seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
            if not seeds or len(set(seeds)) != len(seeds):
                raise ValueError("--seeds requires unique integer values")
            out = Path(args.output)
            observations = []
            scenarios = ["rollouts", "no_rollouts"] if args.include_ablation else ["rollouts"]
            for seed in seeds:
                for scenario in scenarios:
                    result = run_meta_episodes(
                        net if not args.max_nodes else _load(argparse.Namespace(**{**vars(args), "seed": seed})),
                        _config(args, seed=seed), out / f"seed_{seed}" / scenario,
                        get_options("dqn", args.fixed_action),
                        groups=args.communities, holdout_groups=args.holdout_communities,
                        compare_fixed=True, counterfactual_rollouts=(scenario == "rollouts"))
                    for comparison in result["holdout_comparison"]:
                        observations.append({"seed": seed, "scenario": scenario,
                                             "rollout_transitions": result["counterfactual_train_transitions"],
                                             **comparison})
            out.mkdir(parents=True, exist_ok=True)
            data = pd.DataFrame(observations)
            data.to_csv(out / "c2_seed_comparison.csv", index=False)
            aggregate = (data.groupby(["scenario", "policy"], as_index=False)
                         .agg(n_seeds=("seed", "nunique"),
                              mean_ap=("mean_test_ap", "mean"),
                              sd_ap=("mean_test_ap", "std"),
                              mean_roc_auc=("mean_test_roc_auc", "mean")))
            aggregate.to_csv(out / "c2_summary.csv", index=False)
            print(aggregate.to_string(index=False))
            return aggregate.to_dict(orient="records")
        if args.command in {"temporal-rl-meta", "temporal-rl-c2", "temporal-rl-c2-sweep"}:
            from .meta_environment import run_meta_episodes
            result = run_meta_episodes(net, _config(args), args.output,
                                       get_options(args.controller, args.fixed_action),
                                       groups=args.communities,
                                       holdout_groups=args.holdout_communities,
                                       compare_fixed=not args.no_fixed_comparison,
                                       counterfactual_rollouts=(args.command == "temporal-rl-c2"
                                          and not args.no_counterfactual_rollouts))
            print(json.dumps(result, indent=2))
            return result
        if args.command == "temporal-rl":
            result = run_temporal_controller(
                net, _config(args), args.output,
                get_options(args.controller, args.fixed_action))
            print(json.dumps(result, indent=2))
            return result
        seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
        if not seeds or len(set(seeds)) != len(seeds):
            raise ValueError("--seeds requires distinct integers")
        policy_specs = [x.strip() for x in args.policies.split(",") if x.strip()]
        policies = []
        for spec in policy_specs:
            policy, _, action = spec.partition(":")
            if policy not in {"dqn", "fixed", "random"}:
                raise ValueError(f"Invalid policy spec: {spec}")
            action = action or args.fixed_action
            if action not in ACTION_INDEX:
                raise ValueError(f"Invalid adaptation action: {action}")
            policies.append((spec, policy, action))
        directory = Path(args.output)
        rows = []
        for seed in seeds:
            for label, policy, action in policies:
                safe = label.replace(":", "_")
                print(f"\nTemporal RL: seed={seed}, policy={label}", flush=True)
                summary = run_temporal_controller(
                    net, _config(args, seed=seed), directory / f"seed_{seed}" / safe,
                    get_options(policy, action))
                rows.append({**summary, "seed": seed, "policy": label})
        directory.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(rows)
        df.to_csv(directory / "temporal_comparison.csv", index=False)
        stats = (df.groupby("policy", as_index=False)
                   .agg(n_seeds=("seed", "count"),
                        mean_test_ap=("test_mean_ap", "mean"),
                        sd_test_ap=("test_mean_ap", "std"),
                        mean_test_auc=("test_mean_roc_auc", "mean"),
                        mean_reward=("mean_observed_reward", "mean")))
        stats.to_csv(directory / "temporal_summary.csv", index=False)
        print(stats.to_string(index=False))
        return
    if args.command == "run":
        result = run_experiment(net, _config(args), args.output)
        summarize(args.output)
        return result
    if args.command == "sweep":
        from scipy.stats import t as student_t
        seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
        if len(set(seeds)) != len(seeds) or not seeds:
            raise ValueError("--seeds must contain distinct integers")
        directory = Path(args.output)
        rows = []
        for seed in seeds:
            summary = run_experiment(net, _config(args, seed=seed), directory / f"seed_{seed}")
            for method, ap in summary["model_mean_ap"].items():
                rows.append({"seed": seed, "model": method, "mean_ap": ap,
                             "forgetting_ap": summary["final_mean_prior_task_forgetting_ap"]})
        if rows:
            df = pd.DataFrame(rows)
            directory.mkdir(parents=True, exist_ok=True)
            df.to_csv(directory / "seed_sweep.csv", index=False)
            ci_rows = []
            for method, part in df.groupby("model"):
                vals = part["mean_ap"].dropna().to_numpy(dtype=float)
                if len(vals) < 2:
                    continue
                mean = float(np.mean(vals))
                ci = float(student_t.ppf(0.975, len(vals)-1) * np.std(vals, ddof=1) / np.sqrt(len(vals)))
                ci_rows.append({"method": method, "n_seeds": len(vals),
                                "ap_mean": mean, "ap_95ci_low": mean-ci,
                                "ap_95ci_high": mean+ci})
            if ci_rows:
                pd.DataFrame(ci_rows).to_csv(directory / "seed_summary.csv", index=False)
                print(pd.DataFrame(ci_rows).to_string(index=False))
        return
    if args.command == "suite":
        directory = Path(args.output)
        variants = ["baseline", "gcn", "gcn-rl", "gcn-rl-cl",
                    "gcn-rl-cl-no-gcn", "gcn-rl-cl-no-heuristics",
                    "gcn-rl-cl-no-ewc", "gcn-rl-cl-no-replay"]
        for name in variants:
            actual_mode = name.split("-no-")[0]
            path = directory / name
            print(f"\nRunning {name}")
            cfg = _config(args, mode=actual_mode,
                          use_gcn="no-gcn" not in name,
                          use_heuristics="no-heuristics" not in name,
                          use_ewc="no-ewc" not in name,
                          use_replay="no-replay" not in name)
            run_experiment(net, cfg, path)
            from .report import plots
            plots(path)
        summarize_suite(directory)
        return


if __name__ == "__main__":
    main()
