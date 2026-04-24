"""
run_baseline_measurement.py
==========================

Tiny baseline harness for FoodOps Incident Lab scenario JSON files.

This is intentionally simple. It gives us a reproducible "before training"
number even before the full environment loop is wired up.

Current baseline policy: complaint-only lexical heuristic.
That makes it weak on purpose, which is fine — we mainly want a stable
reference point we can improve on later.

Usage:
    python scripts/run_baseline_measurement.py scenarios/
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def load_scenarios(folder: Path):
    for path in sorted(folder.glob("*.json")):
        if path.name.startswith("scenario.schema"):
            continue
        yield path.name, json.loads(path.read_text())


def complaint_only_policy(complaint: str) -> dict:
    text = complaint.lower()
    if any(token in text for token in ["discount", "promo"]):
        return {
            "predicted_root_cause": "business_config_drift",
            "predicted_terminal_action": "submit_report_no_fix",
        }
    if any(token in text for token in ["commission", "payout", "margin"]):
        return {
            "predicted_root_cause": "external_business_reality",
            "predicted_terminal_action": "submit_report_no_fix",
        }
    if any(token in text for token in ["api", "broken", "wrong", "aggregation", "pipeline"]):
        return {
            "predicted_root_cause": "pipeline_failure",
            "predicted_terminal_action": "submit_report_after_refresh",
        }
    return {
        "predicted_root_cause": "pipeline_failure",
        "predicted_terminal_action": "submit_report_after_refresh",
    }


def score_prediction(prediction: dict, scenario: dict) -> dict:
    gt = scenario.get("ground_truth", {})
    true_root_cause = gt.get("true_root_cause")
    if isinstance(true_root_cause, list):
        root_match = prediction["predicted_root_cause"] in true_root_cause
    else:
        root_match = prediction["predicted_root_cause"] == true_root_cause
    action_match = prediction["predicted_terminal_action"] == gt.get("correct_terminal_action")
    return {
        "root_cause_match": root_match,
        "terminal_action_match": action_match,
        "full_match": root_match and action_match,
    }


def main() -> int:
    folder = Path(sys.argv[1] if len(sys.argv) > 1 else "scenarios")
    scenarios = list(load_scenarios(folder))
    if not scenarios:
        print(f"no scenarios found in {folder}")
        return 1

    total = 0
    root_hits = 0
    action_hits = 0
    full_hits = 0

    print(f"baseline evaluation over {len(scenarios)} scenarios\n")
    print("=" * 72)
    for name, scenario in scenarios:
        complaint = scenario["observation"]["user_complaint"]
        prediction = complaint_only_policy(complaint)
        score = score_prediction(prediction, scenario)
        total += 1
        root_hits += int(score["root_cause_match"])
        action_hits += int(score["terminal_action_match"])
        full_hits += int(score["full_match"])
        print(f"\n{name}")
        print(f"  complaint: {complaint}")
        print(f"  predicted_root_cause:   {prediction['predicted_root_cause']}")
        print(f"  predicted_terminal:     {prediction['predicted_terminal_action']}")
        print(f"  root_cause_match:       {score['root_cause_match']}")
        print(f"  terminal_action_match:  {score['terminal_action_match']}")
        print(f"  full_match:             {score['full_match']}")

    print("\n" + "=" * 72)
    print(f"root cause accuracy:      {root_hits}/{total} = {root_hits / total:.1%}")
    print(f"terminal action accuracy: {action_hits}/{total} = {action_hits / total:.1%}")
    print(f"full exact accuracy:      {full_hits}/{total} = {full_hits / total:.1%}")
    print("\nNote: this is a complaint-only lexical baseline, not the final agent baseline.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
