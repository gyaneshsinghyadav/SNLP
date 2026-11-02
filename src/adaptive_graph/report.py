"""Plot metrics and generate simple experiment reports."""
from pathlib import Path
import json
import pandas as pd


def summarize(run_dir):
    path = Path(run_dir)
    summary = json.loads((path / "summary.json").read_text())
    print(json.dumps(summary, indent=2))
    plots(path)


def plots(path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    path = Path(path)
    for filename, metric, output, x in (("predictions.csv", "ap", "prediction_ap.png", "time"),
                                         ("retention.csv", "ap", "retention_ap.png", "after_time"),
                                         ("baselines.csv", "ap", "baseline_ap.png", "time")):
        source = path / filename
        if not source.exists() or source.stat().st_size == 0:
            continue
        df = pd.read_csv(source)
        if df.empty or metric not in df:
            continue
        fig, ax = plt.subplots(figsize=(8, 4.5))
        group = "task_time" if filename == "retention.csv" else "method"
        for name, segment in df.groupby(group, sort=False):
            ax.plot(segment[x].astype(str), segment[metric], marker="o", label=str(name))
        ax.set(title=f"{metric.upper()} — {filename}", xlabel=x, ylabel=metric)
        ax.set_ylim(bottom=0, top=1)
        ax.legend(fontsize=8, loc="best", ncol=2)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(path / output, dpi=130)
        plt.close(fig)


def summarize_suite(suite_dir):
    path = Path(suite_dir)
    rows = []
    for file in path.glob("*/summary.json"):
        s = json.loads(file.read_text())
        for model, ap in s.get("model_mean_ap", {}).items():
            rows.append({"experiment": file.parent.name, "dataset": s["dataset"],
                         "mode": s["mode"], "track": s["track"], "seed": s["seed"],
                         "model": model, "mean_ap": ap,
                         "forgetting_ap": s["final_mean_prior_task_forgetting_ap"]})
        if s.get("mode") == "baseline":
            for model, ap in s.get("baseline_mean_ap", {}).items():
                rows.append({"experiment": file.parent.name, "dataset": s["dataset"],
                             "mode": s["mode"], "track": s["track"], "seed": s["seed"],
                             "model": model, "mean_ap": ap, "forgetting_ap": None})
    if rows:
        pd.DataFrame(rows).to_csv(path / "suite_comparison.csv", index=False)
        print(pd.DataFrame(rows).to_string(index=False))
