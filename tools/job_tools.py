import os
import sys
import subprocess
import json
import time
from datetime import datetime
from strands import tool


@tool
def run_local_job(config_path: str, experiment_name: str, output_dir: str) -> dict:
    """Launch a training job locally using subprocess.
    Executes the dummy trainer with the config file.
    Returns run_id, pid, and log path."""
    try:
        os.makedirs(output_dir, exist_ok=True)
        log_path = os.path.join(output_dir, "training.log")

        cmd = [sys.executable, "tests/dummy_trainer.py", config_path]
        process = subprocess.Popen(
            cmd,
            stdout=open(log_path, 'w'),
            stderr=subprocess.STDOUT,
            cwd=os.getcwd()
        )

        run_id = f"{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        tracking = {
            "run_id":          run_id,
            "pid":             process.pid,
            "config_path":     config_path,
            "experiment_name": experiment_name,
            "output_dir":      output_dir,
            "log_path":        log_path,
            "started_at":      datetime.now().isoformat(),
            "status":          "running"
        }

        tracking_path = os.path.join(output_dir, "run_tracking.json")
        with open(tracking_path, 'w') as f:
            json.dump(tracking, f, indent=2)

        return {
            "success":       True,
            "run_id":        run_id,
            "pid":           process.pid,
            "log_path":      log_path,
            "tracking_path": tracking_path,
            "message":       f"Job launched with PID {process.pid}"
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@tool
def wait_for_completion(output_dir: str, timeout_seconds: int = 120) -> dict:
    """Wait for a training job to finish by polling the log file.
    Checks every 2 seconds for STATUS: completed or STATUS: failed.
    Returns final status and elapsed time."""
    log_path  = os.path.join(output_dir, "training.log")
    start     = time.time()

    while True:
        elapsed = time.time() - start

        if elapsed > timeout_seconds:
            return {
                "status":          "timeout",
                "elapsed_seconds": int(elapsed),
                "message":         f"Job timed out after {timeout_seconds}s"
            }

        if os.path.exists(log_path):
            with open(log_path) as f:
                content = f.read()
            if "STATUS: completed" in content:
                return {"status": "completed", "elapsed_seconds": int(elapsed), "log_path": log_path}
            if "STATUS: failed" in content:
                return {"status": "failed",    "elapsed_seconds": int(elapsed), "log_path": log_path}

        time.sleep(2)


@tool
def tail_logs(log_path: str, n: int = 20) -> dict:
    """Return the last N lines of a training log file."""
    if not os.path.exists(log_path):
        return {"error": f"Log not found: {log_path}", "lines": []}

    with open(log_path) as f:
        lines = f.readlines()

    last_n = lines[-n:] if len(lines) >= n else lines
    return {
        "log_path":      log_path,
        "total_lines":   len(lines),
        "last_n_lines":  [l.rstrip() for l in last_n]
    }


@tool
def get_running_configs() -> dict:
    """List all YAML config files in the experiments/running folder."""
    running_dir = "experiments/running"
    os.makedirs(running_dir, exist_ok=True)

    configs = [
        os.path.join(running_dir, f)
        for f in os.listdir(running_dir)
        if f.endswith('.yaml') or f.endswith('.yml')
    ]
    return {"count": len(configs), "configs": configs}