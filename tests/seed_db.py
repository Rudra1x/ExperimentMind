"""
Seed the DB with realistic mock experiment runs for Analyzer testing.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime, timedelta
from db.schema import init_db, get_session, Experiment, Metric

MOCK_RUNS = [
    {
        "run_id":          "baseline_lr001_bs32_20260825_080000",
        "experiment_name": "iris_classifier",
        "primary_metric":  "val_accuracy",
        "status":          "completed",
        "days_ago":        6,
        "metrics": {
            "val_accuracy": 0.810,
            "val_loss":     0.521,
            "f1":           0.798,
            "epochs_completed": 5
        }
    },
    {
        "run_id":          "exp_lr005_bs32_20260826_090000",
        "experiment_name": "iris_classifier",
        "primary_metric":  "val_accuracy",
        "status":          "completed",
        "days_ago":        5,
        "metrics": {
            "val_accuracy": 0.843,
            "val_loss":     0.471,
            "f1":           0.831,
            "epochs_completed": 5
        }
    },
    {
        "run_id":          "exp_lr005_bs64_20260827_100000",
        "experiment_name": "iris_classifier",
        "primary_metric":  "val_accuracy",
        "status":          "completed",
        "days_ago":        4,
        "metrics": {
            "val_accuracy": 0.832,
            "val_loss":     0.489,
            "f1":           0.819,
            "epochs_completed": 5
        }
    },
    {
        "run_id":          "exp_lr001_bs16_20260828_110000",
        "experiment_name": "iris_classifier",
        "primary_metric":  "val_accuracy",
        "status":          "completed",
        "days_ago":        3,
        "metrics": {
            "val_accuracy": 0.871,
            "val_loss":     0.398,
            "f1":           0.863,
            "epochs_completed": 5
        }
    },
    {
        "run_id":          "exp_lr0001_bs32_20260829_120000",
        "experiment_name": "iris_classifier",
        "primary_metric":  "val_accuracy",
        "status":          "completed",
        "days_ago":        2,
        "metrics": {
            "val_accuracy": 0.856,
            "val_loss":     0.431,
            "f1":           0.844,
            "epochs_completed": 5
        }
    },
    {
        "run_id":          "exp_lr001_bs32_dropout_20260830_130000",
        "experiment_name": "iris_classifier",
        "primary_metric":  "val_accuracy",
        "status":          "completed",
        "days_ago":        1,
        "metrics": {
            "val_accuracy": 0.896,
            "val_loss":     0.312,
            "f1":           0.889,
            "epochs_completed": 5
        }
    },
]


def seed():
    init_db()
    session = get_session()

    # Clear existing test data
    session.query(Metric).filter(
        Metric.run_id.like("%iris_classifier%")
    ).delete(synchronize_session=False)
    session.query(Experiment).filter(
        Experiment.experiment_name == "iris_classifier"
    ).delete(synchronize_session=False)
    session.commit()

    for run_data in MOCK_RUNS:
        created_at = datetime.now() - timedelta(days=run_data["days_ago"])

        exp = Experiment(
            run_id          = run_data["run_id"],
            experiment_name = run_data["experiment_name"],
            config_path     = f"experiments/done/{run_data['run_id']}.yaml",
            status          = run_data["status"],
            primary_metric  = run_data["primary_metric"],
            created_at      = created_at,
            completed_at    = created_at + timedelta(minutes=5)
        )
        session.merge(exp)

        for metric_name, value in run_data["metrics"].items():
            session.add(Metric(
                run_id      = run_data["run_id"],
                metric_name = metric_name,
                value       = float(value),
                step        = -1
            ))

    session.commit()
    session.close()
    print(f"[Seed] {len(MOCK_RUNS)} runs seeded for experiment: iris_classifier")
    print("[Seed] Done. Ready for Analyzer testing.")


if __name__ == "__main__":
    seed()