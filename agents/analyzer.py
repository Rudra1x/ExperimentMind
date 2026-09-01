import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.base import create_agent
from tools.analysis_tools import (
    query_all_runs,
    compute_leaderboard,
    detect_regression,
    get_experiment_summary
)

ANALYZER_PROMPT = """You are the Analyzer Agent for ExperimentMind.

You receive structured experiment data — a leaderboard, regression analysis, 
and experiment summary. Your job is to write a clear, insightful 3-5 sentence 
natural language interpretation for a data scientist.

Cover:
1. What the current best result is and which run achieved it
2. Whether performance is trending up or showing regression
3. The most significant improvement since baseline
4. One concrete recommendation for the next experiment

Be specific with numbers. Be direct. No fluff.
"""


def create_analyzer_agent():
    return create_agent(
        system_prompt=ANALYZER_PROMPT,
        tools=[]  # Analyzer LLM call is for NL generation only — no tools needed
    )


def analyze_experiment(experiment_name: str) -> dict:
    """
    Full deterministic analysis pipeline + LLM natural language summary.
    """
    print(f"\n[Analyzer] Analyzing: {experiment_name}")

    # Step 1: Query all runs (deterministic)
    runs_data = query_all_runs(experiment_name)
    if not runs_data.get("success") or runs_data.get("total_runs", 0) == 0:
        return {"success": False, "error": "No runs found for this experiment"}

    runs           = runs_data["runs"]
    primary_metric = runs[0]["primary_metric"]
    print(f"  Step 1: Queried {len(runs)} runs ✓")

    # Step 2: Compute leaderboard (deterministic)
    leaderboard_data = compute_leaderboard(runs, primary_metric)
    if not leaderboard_data.get("success"):
        return {"success": False, "error": "Leaderboard computation failed"}
    print(f"  Step 2: Leaderboard computed ✓")

    # Step 3: Detect regression (deterministic)
    regression_data = detect_regression(runs, primary_metric)
    print(f"  Step 3: Regression analysis ✓")

    # Step 4: Get summary stats (deterministic)
    summary_data = get_experiment_summary(experiment_name)
    print(f"  Step 4: Summary computed ✓")

    # Step 5: LLM generates natural language interpretation
    print(f"  Step 5: Generating NL interpretation...")

    leaderboard = leaderboard_data["leaderboard"]
    top3 = leaderboard[:3]

    llm_prompt = f"""
Experiment: {experiment_name}
Primary metric: {primary_metric} ({'higher is better' if leaderboard_data['higher_is_better'] else 'lower is better'})

LEADERBOARD (top 3 of {len(leaderboard)} runs):
{chr(10).join([f"  #{r['rank']} {r['run_id'][-30:]}: {primary_metric}={r['metric_value']}" for r in top3])}

TREND ANALYSIS:
  Direction: {regression_data['direction']}
  Last {regression_data['window_size']} runs: {regression_data['window_values']}
  Regression detected: {regression_data['regression']}

SUMMARY:
  Total runs: {summary_data.get('total_runs', 0)}
  Baseline: {summary_data.get('baseline_value')}
  Best: {summary_data.get('best_value')}
  Improvement: +{summary_data.get('improvement_pct')}% since baseline
  Latest run: {summary_data.get('latest_value')}

Write a 3-5 sentence analysis for the data scientist. Be specific with numbers.
"""

    agent    = create_analyzer_agent()
    response = agent(llm_prompt)
    nl_summary = str(response).strip()
    print(f"  Step 5: NL summary generated ✓")

    # Build final output
    result = {
        "success":         True,
        "experiment_name": experiment_name,
        "primary_metric":  primary_metric,
        "leaderboard":     leaderboard,
        "best_run_id":     leaderboard_data["best_run_id"],
        "best_value":      leaderboard_data["best_value"],
        "regression":      regression_data["regression"],
        "trend":           regression_data["direction"],
        "summary":         summary_data,
        "nl_interpretation": nl_summary
    }

    # Print formatted output
    print(f"\n{'='*55}")
    print(f"ANALYSIS RESULTS — {experiment_name}")
    print(f"{'='*55}")
    print(f"\n📊 LEADERBOARD ({primary_metric}):")
    for entry in leaderboard:
        flag = "🥇" if entry["is_best"] else f"#{entry['rank']} "
        print(f"  {flag} {entry['run_id'][-35:]}")
        print(f"     {primary_metric}: {entry['metric_value']}  "
              f"(delta: {entry['delta_vs_best']:+.4f})")

    print(f"\n📈 TREND: {regression_data['direction'].upper()}")
    print(f"   Last {regression_data['window_size']} values: "
          f"{regression_data['window_values']}")
    if regression_data["regression"]:
        print(f"   ⚠️  REGRESSION DETECTED")

    print(f"\n📋 SUMMARY:")
    print(f"   Total runs:   {summary_data.get('total_runs')}")
    print(f"   Baseline:     {summary_data.get('baseline_value')}")
    print(f"   Best:         {summary_data.get('best_value')}")
    print(f"   Improvement:  +{summary_data.get('improvement_pct')}%")

    print(f"\n🤖 INTERPRETATION:")
    print(f"   {nl_summary}")
    print(f"{'='*55}")

    return result


if __name__ == "__main__":
    experiment = sys.argv[1] if len(sys.argv) > 1 else "iris_classifier"
    analyze_experiment(experiment)