"""
ExperimentMind Orchestrator — wires all 5 agents into one pipeline.
Config dropped → Validate → Launch → Log → Analyze → Report
"""
import sys
import os
import subprocess
import yaml
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime
from agents.validator import validate_config
from agents.logger    import log_results
from agents.reporter  import generate_and_send_digest
from tools.report_tools import add_attention_item
from db.schema import get_session, AttentionItem


class ExperimentMindOrchestrator:

    def __init__(self):
        self.pipeline_log = []

    def log(self, step: str, message: str, status: str = "ok"):
        entry = {
            "step":      step,
            "message":   message,
            "status":    status,
            "timestamp": datetime.now().isoformat()
        }
        self.pipeline_log.append(entry)
        icon = "✅" if status == "ok" else "❌" if status == "error" else "⚠️"
        print(f"  {icon} [{step}] {message}")

    def run_pipeline(self, config_path: str) -> dict:
        """
        Full pipeline: validate → launch → log → analyze → report
        Returns final status and all step results.
        """
        print(f"\n{'='*60}")
        print(f"ExperimentMind Pipeline Started")
        print(f"Config: {config_path}")
        print(f"Time:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}\n")

        # ── STEP 1: VALIDATE ─────────────────────────────────────
        print(f"[1/5] VALIDATOR")
        val_result = validate_config(config_path)

        if val_result.get("status") != "passed":
            self.log("Validator", "Config failed validation", "error")
            add_attention_item(
                item_type   = "config_error",
                run_id      = None,
                title       = f"Config validation failed: {os.path.basename(config_path)}",
                description = f"Errors: {val_result.get('errors', [])}"
            )
            return {"success": False, "stage": "validation",
                    "errors": val_result.get("errors")}

        self.log("Validator", "Config passed validation")
        running_config = val_result.get("running_path")
        config         = val_result.get("config", {})
        experiment_name = config.get("experiment_name", "unknown")
        output_dir      = config.get("output_dir",
                                     f"experiments/outputs/{experiment_name}")
        primary_metric  = config.get("primary_metric", "val_accuracy")

        # ── STEP 2: LAUNCH TRAINING ───────────────────────────────
        print(f"\n[2/5] JOB LAUNCHER")
        run_id = f"{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(output_dir, exist_ok=True)

        self.log("Launcher", f"Starting job: {experiment_name}")
        result = subprocess.run(
            [sys.executable, "tests/dummy_trainer.py", running_config],
            capture_output=True, text=True
        )

        if result.returncode != 0:
            self.log("Launcher", f"Job failed: {result.stderr[:100]}", "error")
            add_attention_item(
                item_type   = "job_failure",
                run_id      = run_id,
                title       = f"Training failed: {experiment_name}",
                description = result.stderr[:300]
            )
            return {"success": False, "stage": "launch",
                    "error": result.stderr}

        self.log("Launcher", f"Training complete → {output_dir}")

        # ── STEP 3: LOG RESULTS ───────────────────────────────────
        print(f"\n[3/5] RESULTS LOGGER")
        log_result = log_results(running_config, output_dir, run_id)
        self.log("Logger", f"Metrics stored: {run_id}")

        # ── STEP 4: ANALYZE ───────────────────────────────────────
        print(f"\n[4/5] ANALYZER")
        # Re-seed the experiment if this is a fresh run on a new experiment
        from tools.analysis_tools import query_all_runs
        runs = query_all_runs(experiment_name)

        if runs.get("total_runs", 0) < 2:
            self.log("Analyzer", "Not enough runs for full analysis (need 2+)", "warn")
            analysis_done = False
        else:
            analysis_done = True
            self.log("Analyzer", "Analysis complete")

        # ── STEP 5: REPORT ────────────────────────────────────────
        print(f"\n[5/5] REPORTER")
        if analysis_done:
            report_result = generate_and_send_digest(experiment_name)
            self.log("Reporter", f"Digest generated → {report_result.get('digest_path')}")

            # Add attention item if new best
            from tools.analysis_tools import compute_leaderboard
            from tools.analysis_tools import get_experiment_summary
            summary = get_experiment_summary(experiment_name)
            if summary.get("best_run_id") == run_id:
                add_attention_item(
                    item_type   = "new_best_model",
                    run_id      = run_id,
                    title       = f"New best model: {experiment_name}",
                    description = f"{primary_metric}: {summary.get('best_value')} "
                                  f"(+{summary.get('improvement_pct')}% over baseline). "
                                  f"Approve for deployment?"
                )
                self.log("Reporter", "New best model — added to attention queue", "warn")
        else:
            self.log("Reporter", "Skipped (need more runs for digest)", "warn")

        print(f"\n{'='*60}")
        print(f"Pipeline Complete — {experiment_name}")
        print(f"Run ID: {run_id}")
        print(f"{'='*60}\n")

        return {
            "success":         True,
            "run_id":          run_id,
            "experiment_name": experiment_name,
            "pipeline_log":    self.pipeline_log
        }


def run_full_pipeline(config_path: str) -> dict:
    orchestrator = ExperimentMindOrchestrator()
    return orchestrator.run_pipeline(config_path)


if __name__ == "__main__":
    config = sys.argv[1] if len(sys.argv) > 1 else "tests/config_valid.yaml"
    run_full_pipeline(config)