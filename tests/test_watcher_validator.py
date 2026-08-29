import sys
import os
import shutil
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.validator import validate_config


def setup():
    """Reset experiments folders before testing."""
    for folder in ["experiments/queue", "experiments/running",
                   "experiments/failed", "experiments/outputs"]:
        os.makedirs(folder, exist_ok=True)

    # Copy test configs to queue
    test_configs = [
        "tests/config_valid.yaml",
        "tests/config_missing_fields.yaml",
        "tests/config_bad_path.yaml"
    ]
    for config in test_configs:
        if os.path.exists(config):
            dest = os.path.join("experiments/queue", os.path.basename(config))
            shutil.copy(config, dest)
            print(f"[Test] Queued: {dest}")

    return [
        os.path.join("experiments/queue", os.path.basename(c))
        for c in test_configs
    ]


def run_tests():
    print("\n" + "="*50)
    print("ExperimentMind — Watcher + Validator Test")
    print("="*50)

    configs = setup()

    results = {"passed": 0, "failed": 0}

    for config_path in configs:
        if not os.path.exists(config_path):
            print(f"[Test] Skipping (already processed): {config_path}")
            continue

        print(f"\n[Test] Processing: {config_path}")
        print("-" * 40)
        validate_config(config_path)

    # Check routing results
    print("\n" + "="*50)
    print("ROUTING RESULTS")
    print("="*50)

    running = os.listdir("experiments/running")
    failed = os.listdir("experiments/failed")

    print(f"\n Moved to running/ ({len([f for f in running if f.endswith('.yaml')])} configs):")
    for f in running:
        if f.endswith('.yaml'):
            print(f"   {f}")

    print(f"\n Moved to failed/ ({len([f for f in failed if f.endswith('.yaml')])} configs):")
    for f in failed:
        if f.endswith('.yaml'):
            print(f"   {f}")

    print(f"\n Error reports written:")
    for f in failed:
        if f.endswith('.txt'):
            print(f"   {f}")
            with open(os.path.join("experiments/failed", f)) as report:
                print(report.read())

    print("="*50)
    print("Test complete. Check experiments/running/ and experiments/failed/")


if __name__ == "__main__":
    run_tests()