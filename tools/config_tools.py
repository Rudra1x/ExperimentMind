import os
import yaml
import shutil
import hashlib
from datetime import datetime
from strands import tool


@tool
def read_yaml_config(config_path: str) -> dict:
    """Read and parse a YAML experiment config file from disk.
    Returns the config as a dictionary or an error dict if parsing fails."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        if not isinstance(config, dict):
            return {"error": "Config file is not a valid YAML mapping"}
        config["_config_path"] = config_path
        config["_config_hash"] = hashlib.md5(
            open(config_path, 'rb').read()
        ).hexdigest()[:8]
        return config
    except FileNotFoundError:
        return {"error": f"File not found: {config_path}"}
    except yaml.YAMLError as e:
        return {"error": f"YAML parse error: {str(e)}"}


@tool
def validate_required_fields(config: dict) -> dict:
    """Check that all required fields are present in the experiment config.
    Returns a validation result with valid=True/False and list of errors."""
    required_fields = [
        "experiment_name",
        "model_type",
        "dataset_path",
        "output_dir",
        "primary_metric",
        "executor",
        "hyperparams"
    ]
    valid_executors = ["local", "kaggle"]
    valid_metrics = [
        "val_accuracy", "accuracy", "val_loss", "loss",
        "f1", "val_f1", "precision", "recall",
        "mae", "mse", "rmse", "r2"
    ]

    errors = []

    # Check required fields
    for field in required_fields:
        if field not in config:
            errors.append(f"Missing required field: '{field}'")

    # Check executor value
    if "executor" in config and config["executor"] not in valid_executors:
        errors.append(
            f"Invalid executor '{config['executor']}'. "
            f"Must be one of: {valid_executors}"
        )

    # Check primary_metric value
    if "primary_metric" in config and config["primary_metric"] not in valid_metrics:
        errors.append(
            f"Invalid primary_metric '{config['primary_metric']}'. "
            f"Must be one of: {valid_metrics}"
        )

    # Check hyperparams is a dict
    if "hyperparams" in config and not isinstance(config["hyperparams"], dict):
        errors.append("'hyperparams' must be a dictionary of key-value pairs")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "fields_checked": required_fields
    }


@tool
def check_paths(dataset_path: str, output_dir: str) -> dict:
    """Verify that the dataset path exists and the output directory is writable.
    Returns path validation results with specific error messages."""
    errors = []

    # Check dataset exists
    if not os.path.exists(dataset_path):
        errors.append(
            f"Dataset not found: '{dataset_path}'. "
            f"Check the path is correct and the file exists."
        )

    # Check or create output directory
    try:
        os.makedirs(output_dir, exist_ok=True)
        # Test write permission
        test_file = os.path.join(output_dir, ".write_test")
        with open(test_file, 'w') as f:
            f.write("test")
        os.remove(test_file)
    except PermissionError:
        errors.append(
            f"Output directory not writable: '{output_dir}'"
        )
    except Exception as e:
        errors.append(f"Output directory error: {str(e)}")

    return {
        "paths_valid": len(errors) == 0,
        "dataset_path": dataset_path,
        "output_dir": output_dir,
        "errors": errors
    }


@tool
def move_config_to_running(config_path: str, experiment_name: str) -> dict:
    """Move a validated config file from queue to running folder.
    Returns the new path of the moved config."""
    try:
        running_dir = "experiments/running"
        os.makedirs(running_dir, exist_ok=True)

        filename = os.path.basename(config_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_filename = f"{timestamp}_{experiment_name}_{filename}"
        dest_path = os.path.join(running_dir, new_filename)

        shutil.move(config_path, dest_path)
        return {
            "success": True,
            "new_path": dest_path,
            "message": f"Config moved to running: {dest_path}"
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@tool
def move_config_to_failed(config_path: str, errors: list) -> dict:
    """Move an invalid config to the failed folder and write an error report.
    Returns the paths of the moved config and error report."""
    try:
        failed_dir = "experiments/failed"
        os.makedirs(failed_dir, exist_ok=True)

        filename = os.path.basename(config_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_filename = f"{timestamp}_FAILED_{filename}"
        dest_path = os.path.join(failed_dir, new_filename)

        shutil.move(config_path, dest_path)

        # Write error report alongside the failed config
        report_path = dest_path.replace(".yaml", "_error_report.txt")
        with open(report_path, 'w') as f:
            f.write(f"ExperimentMind Validation Report\n")
            f.write(f"{'='*40}\n")
            f.write(f"Config: {filename}\n")
            f.write(f"Time: {datetime.now().isoformat()}\n")
            f.write(f"Status: FAILED\n\n")
            f.write(f"Errors found ({len(errors)}):\n")
            for i, error in enumerate(errors, 1):
                f.write(f"  {i}. {error}\n")
            f.write(f"\nFix these errors and re-submit the config.\n")

        return {
            "success": True,
            "failed_config_path": dest_path,
            "error_report_path": report_path,
            "error_count": len(errors)
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@tool
def list_queue_configs() -> dict:
    """List all YAML config files currently in the experiments queue folder.
    Returns a list of config file paths waiting to be processed."""
    queue_dir = "experiments/queue"
    os.makedirs(queue_dir, exist_ok=True)

    configs = [
        os.path.join(queue_dir, f)
        for f in os.listdir(queue_dir)
        if f.endswith('.yaml') or f.endswith('.yml')
    ]

    return {
        "queue_count": len(configs),
        "configs": configs,
        "queue_dir": queue_dir
    }