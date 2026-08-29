import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.base import create_agent
from tools.config_tools import (
    read_yaml_config,
    validate_required_fields,
    check_paths,
    move_config_to_running,
    move_config_to_failed
)

VALIDATOR_PROMPT = """You are the Validator Agent for ExperimentMind, an autonomous 
ML experiment management system.

Your job is to validate experiment configuration files before they are executed.

When given a config file path, you must:
1. Call read_yaml_config to load the config
2. Call validate_required_fields to check all required fields are present
3. If fields are valid, call check_paths to verify dataset and output paths
4. If ALL checks pass: call move_config_to_running and return success
5. If ANY check fails: collect ALL errors, call move_config_to_failed, return failure

Always collect all errors before deciding — don't stop at the first error.
Return a clear, structured summary of what you found and what action you took.
"""

def create_validator_agent():
    return create_agent(
        system_prompt=VALIDATOR_PROMPT,
        tools=[
            read_yaml_config,
            validate_required_fields,
            check_paths,
            move_config_to_running,
            move_config_to_failed
        ]
    )


def validate_config(config_path: str) -> dict:
    """
    Validate a single experiment config file.
    Returns structured result with status, errors, and next path.
    """
    print(f"\n[Validator] Checking: {config_path}")
    agent = create_validator_agent()

    prompt = f"""
    Validate this experiment config file: {config_path}
    
    Follow these steps in order:
    1. Read the config file
    2. Validate required fields
    3. If fields are valid, check paths
    4. Move to running (if valid) or failed (if invalid)
    5. Report: PASSED or FAILED, with all errors listed
    """

    response = agent(prompt)
    print(f"[Validator] Result: {response}")
    return {"config_path": config_path, "response": str(response)}


if __name__ == "__main__":
    # Quick test
    if len(sys.argv) > 1:
        validate_config(sys.argv[1])
    else:
        print("Usage: python agents/validator.py <config_path>")