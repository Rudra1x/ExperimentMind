import os
from datetime import datetime
from strands import tool
from db.schema import get_session, Experiment, Metric


@tool
def query_all_runs(experiment_name: str) -> dict:
    """Query all completed runs for a given experiment from the database.
    Returns runs sorted by creation date (oldest first)."""
    try:
        session = get_session()
        runs = session.query(Experiment).filter(
            Experiment.experiment_name == experiment_name,
            Experiment.status == "completed"
        ).order_by(Experiment.created_at.asc()).all()

        results = []
        for run in runs:
            metrics = session.query(Metric).filter(
                Metric.run_id == run.run_id,
                Metric.step == -1
            ).all()
            results.append({
                "run_id":         run.run_id,
                "experiment_name": run.experiment_name,
                "primary_metric": run.primary_metric,
                "status":         run.status,
                "created_at":     run.created_at.isoformat() if run.created_at else None,
                "metrics":        {m.metric_name: m.value for m in metrics}
            })

        session.close()
        return {
            "success":         True,
            "experiment_name": experiment_name,
            "total_runs":      len(results),
            "runs":            results
        }
    except Exception as e:
        return {"success": False, "error": str(e), "runs": []}


@tool
def compute_leaderboard(runs: list, primary_metric: str) -> dict:
    """Rank experiment runs by their primary metric.
    For accuracy/f1/precision/recall/r2 — higher is better.
    For loss/mae/mse/rmse — lower is better.
    Returns ranked leaderboard with deltas vs best."""

    higher_is_better = ["val_accuracy", "accuracy", "f1", "val_f1",
                         "precision", "recall", "r2"]
    reverse = primary_metric in higher_is_better

    # Filter runs that have the primary metric
    valid_runs = [
        r for r in runs
        if primary_metric in r.get("metrics", {})
    ]

    if not valid_runs:
        return {"success": False, "error": f"No runs with metric: {primary_metric}"}

    # Sort by primary metric
    sorted_runs = sorted(
        valid_runs,
        key=lambda r: r["metrics"][primary_metric],
        reverse=reverse
    )

    best_value = sorted_runs[0]["metrics"][primary_metric]

    leaderboard = []
    for rank, run in enumerate(sorted_runs, 1):
        metric_value = run["metrics"][primary_metric]
        delta        = metric_value - best_value if not reverse else metric_value - best_value

        leaderboard.append({
            "rank":          rank,
            "run_id":        run["run_id"],
            "metric_value":  round(metric_value, 4),
            "delta_vs_best": round(delta, 4),
            "is_best":       rank == 1,
            "created_at":    run.get("created_at"),
            "all_metrics":   run.get("metrics", {})
        })

    return {
        "success":        True,
        "primary_metric": primary_metric,
        "higher_is_better": reverse,
        "best_value":     round(best_value, 4),
        "best_run_id":    sorted_runs[0]["run_id"],
        "total_ranked":   len(leaderboard),
        "leaderboard":    leaderboard
    }


@tool
def detect_regression(runs: list, primary_metric: str, window: int = 3) -> dict:
    """Check if the last N runs show a declining trend in the primary metric.
    Returns regression flag, trend direction, and magnitude."""

    higher_is_better = ["val_accuracy", "accuracy", "f1", "val_f1",
                         "precision", "recall", "r2"]
    is_higher_better = primary_metric in higher_is_better

    # Filter and sort by date
    valid_runs = sorted(
        [r for r in runs if primary_metric in r.get("metrics", {})],
        key=lambda r: r.get("created_at", "")
    )

    if len(valid_runs) < 2:
        return {"regression": False, "reason": "Not enough runs to detect trend"}

    recent = valid_runs[-window:]
    values = [r["metrics"][primary_metric] for r in recent]

    # Simple trend: compare first and last in window
    trend_delta = values[-1] - values[0]
    is_regression = (trend_delta < 0 and is_higher_better) or \
                    (trend_delta > 0 and not is_higher_better)

    return {
        "regression":        is_regression,
        "trend_delta":       round(trend_delta, 4),
        "direction":         "declining" if is_regression else "improving",
        "window_values":     [round(v, 4) for v in values],
        "window_size":       len(recent),
        "primary_metric":    primary_metric,
        "is_higher_better":  is_higher_better
    }


@tool
def get_experiment_summary(experiment_name: str) -> dict:
    """Get a high-level summary of an experiment's progress.
    Returns total runs, best result, improvement over baseline, and trend."""
    try:
        session = get_session()
        runs = session.query(Experiment).filter(
            Experiment.experiment_name == experiment_name,
            Experiment.status == "completed"
        ).order_by(Experiment.created_at.asc()).all()

        if not runs:
            session.close()
            return {"success": False, "error": "No completed runs found"}

        primary_metric = runs[0].primary_metric
        higher_is_better = primary_metric in [
            "val_accuracy", "accuracy", "f1", "val_f1",
            "precision", "recall", "r2"
        ]

        run_metrics = []
        for run in runs:
            metrics = session.query(Metric).filter(
                Metric.run_id == run.run_id,
                Metric.metric_name == primary_metric,
                Metric.step == -1
            ).first()
            if metrics:
                run_metrics.append({
                    "run_id":     run.run_id,
                    "value":      metrics.value,
                    "created_at": run.created_at.isoformat() if run.created_at else None
                })

        session.close()

        if not run_metrics:
            return {"success": False, "error": "No metric data found"}

        values       = [r["value"] for r in run_metrics]
        best_value   = max(values) if higher_is_better else min(values)
        worst_value  = min(values) if higher_is_better else max(values)
        baseline     = values[0]
        latest       = values[-1]
        improvement  = best_value - baseline if higher_is_better else baseline - best_value

        return {
            "success":         True,
            "experiment_name": experiment_name,
            "primary_metric":  primary_metric,
            "total_runs":      len(run_metrics),
            "baseline_value":  round(baseline, 4),
            "best_value":      round(best_value, 4),
            "latest_value":    round(latest, 4),
            "total_improvement": round(improvement, 4),
            "improvement_pct": round((improvement / baseline) * 100, 1),
            "best_run_id":     run_metrics[values.index(best_value)]["run_id"]
        }
    except Exception as e:
        return {"success": False, "error": str(e)}