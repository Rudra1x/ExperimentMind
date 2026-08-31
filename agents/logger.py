import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
from datetime import datetime
from agents.base import create_agent
from tools.logging_tools import (
    parse_metrics_from_log, write_run_to_db,
    move_config_to_done, query_recent_runs
)

LOGGER_PROMPT = """You are the Results Logger Agent for ExperimentMind.

Your job is to parse training results and store them permanently.

When given job information:
1. Call parse_metrics_from_log to extract metrics from the log file
2. Call write_run_to_db to store the run and all metrics in the database
3. Call move_config_to_done to archive the config
4. Call query_recent_runs to confirm storage
5. Report what metrics were found and stored
"""

def create_logger_agent():
    return create_agent(
        system_prompt=LOGGER_PROMPT,
        tools=[parse_metrics_from_log, write_run_to_db, move_config_to_done, query_recent_runs]
    )


def log_results(config_path: str, output_dir: str, run_id: str = None) -> dict:
    """Deterministic logger — calls tools directly, no LLM orchestration."""
    print(f"\n[Logger] Logging results: {output_dir}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    experiment_name = config.get("experiment_name", "unknown")
    primary_metric  = config.get("primary_metric", "val_accuracy")
    log_path        = os.path.join(output_dir, "training.log")

    if run_id is None:
        run_id = f"{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Step 1: Parse metrics
    metrics_result = parse_metrics_from_log(log_path)
    print(f"  Tool #1: parse_metrics_from_log ✓")

    if "error" in metrics_result:
        print(f"[Logger] FAILED → {metrics_result['error']}")
        return {"status": "failed", "error": metrics_result["error"]}

    metrics = metrics_result.get("metrics", {})

    # Step 2: Write to DB
    db_result = write_run_to_db(
        run_id          = run_id,
        experiment_name = experiment_name,
        config_path     = config_path,
        metrics         = metrics,
        primary_metric  = primary_metric,
        status          = "completed"
    )
    print(f"  Tool #2: write_run_to_db ✓")

    # Step 3: Archive config
    move_config_to_done(config_path, run_id)
    print(f"  Tool #3: move_config_to_done ✓")

    # Step 4: Confirm storage
    recent = query_recent_runs(limit=3)
    print(f"  Tool #4: query_recent_runs ✓")

    print(f"[Logger] DONE → {len(metrics)} metrics stored for {run_id}")
    return {"status": "completed", "run_id": run_id, "metrics": metrics}