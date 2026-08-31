import os
import re
import json
import shutil
from datetime import datetime
from strands import tool
from db.schema import get_session, Experiment, Metric


@tool
def parse_metrics_from_log(log_path: str) -> dict:
    """Parse final training metrics from a completed job log file.
    First looks for FINAL_METRICS JSON line, then falls back to regex.
    Returns dict of metric names and values."""
    if not os.path.exists(log_path):
        return {"error": f"Log not found: {log_path}", "metrics": {}}

    with open(log_path) as f:
        content = f.read()

    # Primary: parse FINAL_METRICS JSON line
    match = re.search(r'FINAL_METRICS:\s*({.+})', content)
    if match:
        try:
            metrics = json.loads(match.group(1))
            return {"source": "FINAL_METRICS", "metrics": metrics, "log_path": log_path}
        except json.JSONDecodeError:
            pass

    # Fallback: regex for individual metric lines
    pattern = re.compile(
        r'(val_accuracy|accuracy|val_loss|loss|f1|val_f1|precision|recall|mae|mse|rmse|r2)'
        r'\s*[=:]\s*([0-9.]+)',
        re.IGNORECASE
    )
    metrics = {}
    for m in pattern.finditer(content):
        metrics[m.group(1).lower()] = float(m.group(2))

    if not metrics:
        return {"error": "No metrics found in log", "metrics": {}, "log_path": log_path}

    return {"source": "regex", "metrics": metrics, "log_path": log_path}


@tool
def write_run_to_db(
    run_id:          str,
    experiment_name: str,
    config_path:     str,
    metrics:         dict,
    primary_metric:  str,
    status:          str = "completed"
) -> dict:
    """Write a completed experiment run and its metrics to the SQLite database."""
    try:
        session = get_session()

        experiment = Experiment(
            run_id          = run_id,
            experiment_name = experiment_name,
            config_path     = config_path,
            status          = status,
            primary_metric  = primary_metric,
            completed_at    = datetime.now()
        )
        session.merge(experiment)

        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, (int, float)):
                session.add(Metric(
                    run_id      = run_id,
                    metric_name = metric_name,
                    value       = float(metric_value),
                    step        = -1  # -1 = final value
                ))

        session.commit()
        session.close()

        return {
            "success":         True,
            "run_id":          run_id,
            "metrics_written": len(metrics),
            "message":         f"Run {run_id} stored with {len(metrics)} metrics"
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@tool
def move_config_to_done(config_path: str, run_id: str) -> dict:
    """Archive a completed experiment config from running/ to done/ folder."""
    try:
        done_dir = "experiments/done"
        os.makedirs(done_dir, exist_ok=True)

        filename     = os.path.basename(config_path)
        timestamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_filename = f"{timestamp}_DONE_{run_id[:20]}_{filename}"
        dest_path    = os.path.join(done_dir, new_filename)

        if os.path.exists(config_path):
            shutil.move(config_path, dest_path)

        return {"success": True, "done_path": dest_path}
    except Exception as e:
        return {"success": False, "error": str(e)}


@tool
def query_recent_runs(limit: int = 10) -> dict:
    """Query the database for recent experiment runs with their metrics."""
    try:
        session = get_session()
        runs    = session.query(Experiment)\
                         .order_by(Experiment.created_at.desc())\
                         .limit(limit).all()

        results = []
        for run in runs:
            metrics = session.query(Metric)\
                             .filter(Metric.run_id == run.run_id, Metric.step == -1)\
                             .all()
            results.append({
                "run_id":          run.run_id,
                "experiment_name": run.experiment_name,
                "status":          run.status,
                "primary_metric":  run.primary_metric,
                "metrics":         {m.metric_name: m.value for m in metrics},
                "created_at":      run.created_at.isoformat() if run.created_at else None
            })

        session.close()
        return {"success": True, "count": len(results), "runs": results}
    except Exception as e:
        return {"success": False, "error": str(e)}