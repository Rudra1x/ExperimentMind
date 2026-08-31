"""
Dummy ML trainer for ExperimentMind testing.
Reads a YAML config, simulates training, writes metrics to log.
"""
import yaml
import sys
import json
import time
import os
import random


def train(config_path: str) -> int:
    with open(config_path) as f:
        config = yaml.safe_load(f)

    experiment_name = config.get('experiment_name', 'unknown')
    output_dir      = config.get('output_dir', 'experiments/outputs/default')
    hyperparams     = config.get('hyperparams', {})
    C               = float(hyperparams.get('C', 1.0))

    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, 'training.log')

    random.seed(int(C * 100))
    print(f"Training started: {experiment_name}")
    print(f"Log: {log_path}")

    with open(log_path, 'w') as log:
        log.write(f"ExperimentMind Training Log\n")
        log.write(f"{'='*40}\n")
        log.write(f"Experiment: {experiment_name}\n")
        log.write(f"Config:     {config_path}\n")
        log.write(f"Params:     {json.dumps(hyperparams)}\n")
        log.write(f"{'='*40}\n\n")

        val_accuracy = 0.72 + (C * 0.05)
        val_loss     = 1.0

        for epoch in range(1, 6):
            val_accuracy = min(val_accuracy + random.uniform(0.01, 0.04), 0.99)
            val_loss     = max(val_loss     - random.uniform(0.05, 0.12), 0.05)

            log.write(f"Epoch {epoch}/5\n")
            log.write(f"  val_accuracy: {val_accuracy:.4f}\n")
            log.write(f"  val_loss:     {val_loss:.4f}\n\n")
            log.flush()
            time.sleep(0.3)

        final_metrics = {
            "val_accuracy":     round(val_accuracy, 4),
            "val_loss":         round(val_loss, 4),
            "f1":               round(val_accuracy * 0.97, 4),
            "epochs_completed": 5
        }

        log.write(f"\n{'='*40}\n")
        log.write(f"FINAL_METRICS: {json.dumps(final_metrics)}\n")
        log.write(f"STATUS: completed\n")

    print(f"Training complete. val_accuracy={val_accuracy:.4f}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tests/dummy_trainer.py <config_path>")
        sys.exit(1)
    sys.exit(train(sys.argv[1]))