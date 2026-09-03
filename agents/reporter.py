import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime
from agents.analyzer import analyze_experiment
from tools.report_tools import (
    format_digest, save_digest,
    send_email_digest, send_slack_digest,
    add_attention_item, get_attention_items
)


def generate_and_send_digest(experiment_name: str) -> dict:
    """
    Full reporter pipeline — deterministic, no LLM needed here.
    Analyzer already handled the LLM step.
    """
    print(f"\n[Reporter] Generating digest for: {experiment_name}")

    # Step 1: Run full analysis
    analysis = analyze_experiment(experiment_name)
    if not analysis.get("success"):
        print(f"[Reporter] Analysis failed: {analysis.get('error')}")
        return {"success": False, "error": analysis.get("error")}

    # Step 2: Format digest
    digest_result = format_digest(
        experiment_name      = experiment_name,
        leaderboard          = analysis["leaderboard"],
        summary              = analysis["summary"],
        trend                = analysis["trend"],
        regression           = analysis["regression"],
        nl_interpretation    = analysis["nl_interpretation"],
        runs_completed_today = 1,
        runs_failed_today    = 0
    )
    print(f"  Step 1: Digest formatted ✓")

    content = digest_result["content"]
    date    = digest_result["date"]
    subject = f"ExperimentMind Digest — {date} — Best: {analysis['best_value']}"

    # Step 3: Save to file + DB
    save_result = save_digest(content, date)
    print(f"  Step 2: Digest saved → {save_result.get('path')} ✓")

    # Step 4: Send email (skipped if not configured)
    email_result = send_email_digest(content, subject)
    if email_result.get("skipped"):
        print(f"  Step 3: Email skipped ({email_result.get('reason')})")
    elif email_result.get("success"):
        print(f"  Step 3: Email sent ✓")
    else:
        print(f"  Step 3: Email failed — {email_result.get('error')}")

    # Step 5: Send Slack (skipped if not configured)
    slack_result = send_slack_digest(content, subject)
    if slack_result.get("skipped"):
        print(f"  Step 4: Slack skipped (webhook not configured)")
    elif slack_result.get("success"):
        print(f"  Step 4: Slack sent ✓")

    # Step 6: Add attention item if new best or regression
    if analysis.get("regression"):
        add_attention_item(
            item_type   = "regression",
            run_id      = analysis["best_run_id"],
            title       = f"Regression detected: {experiment_name}",
            description = f"Performance declining in last 3 runs. "
                          f"Best: {analysis['best_value']}"
        )
        print(f"  Step 5: Regression alert added to attention queue ✓")

    print(f"\n[Reporter] Digest complete.")
    print(f"  File: {save_result.get('path')}")
    print(f"\n{'='*55}")
    print(content[:800] + "..." if len(content) > 800 else content)
    print(f"{'='*55}")

    return {
        "success":     True,
        "digest_path": save_result.get("path"),
        "subject":     subject,
        "content":     content
    }


if __name__ == "__main__":
    experiment = sys.argv[1] if len(sys.argv) > 1 else "iris_classifier"
    generate_and_send_digest(experiment)