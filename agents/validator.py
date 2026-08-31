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
    Deterministic pipeline — calls tools directly, no LLM orchestration.
    LLM reasoning is reserved for agents that actually need it (Analyzer).
    """
    print(f"\n[Validator] Checking: {config_path}")

    # Step 1: Read config
    config = read_yaml_config(config_path)
    if "error" in config:
        errors = [config["error"]]
        move_config_to_failed(config_path, errors)
        print(f"[Validator] FAILED → {errors}")
        return {"status": "failed", "errors": errors}

    print(f"  Tool #1: read_yaml_config ✓")

    # Step 2: Validate required fields
    field_result = validate_required_fields(config)
    errors = field_result.get("errors", [])
    print(f"  Tool #2: validate_required_fields ✓")

    # Step 3: Check paths only if fields passed
    if field_result.get("valid"):
        path_result = check_paths(
            config.get("dataset_path", ""),
            config.get("output_dir", "")
        )
        print(f"  Tool #3: check_paths ✓")
        if not path_result.get("paths_valid"):
            errors.extend(path_result.get("errors", []))
    else:
        print(f"  Tool #3: check_paths SKIPPED (field errors found)")

    # Step 4: Route based on result
    if not errors:
        move_result = move_config_to_running(
            config_path,
            config.get("experiment_name", "unknown")
        )
        print(f"  Tool #4: move_config_to_running ✓")
        print(f"[Validator] PASSED → {move_result.get('new_path')}")
        return {
            "status":       "passed",
            "config":       config,
            "running_path": move_result.get("new_path")
        }
    else:
        move_config_to_failed(config_path, errors)
        print(f"  Tool #4: move_config_to_failed ✓")
        print(f"[Validator] FAILED → {len(errors)} error(s): {errors}")
        return {"status": "failed", "errors": errors}


if __name__ == "__main__":
    # Quick test
    if len(sys.argv) > 1:
        validate_config(sys.argv[1])
    else:
        print("Usage: python agents/validator.py <config_path>")