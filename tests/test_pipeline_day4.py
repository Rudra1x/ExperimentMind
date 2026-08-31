import sys
import os
import shutil
import subprocess
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
from db.schema import init_db
from agents.validator import validate_config
from agents.logger import log_results
from tools.logging_tools import query_recent_runs


def run_pipeline():
    print("\n" + "="*50)
    print("ExperimentMind — Full Pipeline Test (Day 4)")
    print("="*50)

    # Setup
    init_db()
    for d in ["experiments/queue", "experiments/running",
              "experiments/done", "experiments/failed"]:
        os.makedirs(d, exist_ok=True)

    shutil.copy("tests/config_valid.yaml", "experiments/queue/pipeline_test.yaml")
    config_queue = "experiments/queue/pipeline_test.yaml"
    print(f"[Setup] Config queued: {config_queue}")

    # Step 1: Validate
    print("\n--- Step 1: Validate ---")
    validate_config(config_queue)

    running_configs = [
        os.path.join("experiments/running", f)
        for f in os.listdir("experiments/running")
        if "pipeline_test" in f and f.endswith('.yaml')
    ]

    if not running_configs:
        print("FAILED: Config not in running/ after validation")
        return

    running_config = sorted(running_configs)[-1]
    print(f"[Step 1] PASSED → {running_config}")

    # Step 2: Launch training
    print("\n--- Step 2: Launch Training ---")
    with open(running_config) as f:
        config = yaml.safe_load(f)

    output_dir = config.get("output_dir", "experiments/outputs/test_baseline_v1")
    os.makedirs(output_dir, exist_ok=True)

    result = subprocess.run(
        [sys.executable, "tests/dummy_trainer.py", running_config],
        capture_output=True, text=True
    )
    print(result.stdout.strip())

    if result.returncode != 0:
        print(f"FAILED: Trainer error → {result.stderr}")
        return

    print(f"[Step 2] PASSED → log at {output_dir}/training.log")

    # Step 3: Log results
    print("\n--- Step 3: Log Results ---")
    log_results(running_config, output_dir)

    # Step 4: Verify DB
    print("\n--- Verification: DB Query ---")
    runs = query_recent_runs(limit=5)

    print("\n" + "="*50)
    print("FINAL RESULTS")
    print("="*50)

    if runs.get("success") and runs.get("count", 0) > 0:
        for run in runs["runs"]:
            print(f"\n✅ Run stored in DB:")
            print(f"   run_id:     {run['run_id']}")
            print(f"   experiment: {run['experiment_name']}")
            print(f"   status:     {run['status']}")
            print(f"   metrics:")
            for k, v in run['metrics'].items():
                print(f"     {k}: {v}")
    else:
        print("❌ No runs found in DB - check logger output above")

    print(f"\n📁 experiments/running: {len(os.listdir('experiments/running'))} files")
    print(f"📁 experiments/done:    {len(os.listdir('experiments/done'))} files")
    print("="*50)


if __name__ == "__main__":
    run_pipeline()