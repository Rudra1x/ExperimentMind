import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
from agents.base import create_agent
from tools.job_tools import run_local_job, wait_for_completion, tail_logs, get_running_configs

LAUNCHER_PROMPT = """You are the Job Launcher Agent for ExperimentMind.

Your job is to execute ML training jobs from validated experiment configs.

When given a config path, experiment name, and output directory:
1. Call run_local_job to start the training process
2. Call wait_for_completion to wait until the job finishes
3. Call tail_logs to verify the final output
4. Report SUCCESS or FAILED with the run_id and log_path
"""

def create_launcher_agent():
    return create_agent(
        system_prompt=LAUNCHER_PROMPT,
        tools=[run_local_job, wait_for_completion, tail_logs, get_running_configs]
    )


def launch_job(config_path: str) -> dict:
    """Launch a training job for a validated config file."""
    print(f"\n[Launcher] Starting job: {config_path}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    experiment_name = config.get("experiment_name", "unknown")
    output_dir      = config.get("output_dir", f"experiments/outputs/{experiment_name}")

    agent  = create_launcher_agent()
    prompt = f"""
    Launch and monitor this training job:
    - config_path:      {config_path}
    - experiment_name:  {experiment_name}
    - output_dir:       {output_dir}

    Run local job → wait for completion → tail logs → report SUCCESS or FAILED.
    """

    response = agent(prompt)
    print(f"[Launcher] Done: {response}")
    return {"config_path": config_path, "experiment_name": experiment_name,
            "output_dir": output_dir, "response": str(response)}