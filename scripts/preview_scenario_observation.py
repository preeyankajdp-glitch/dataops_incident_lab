"""
preview_scenario_observation.py
===============================

Small dry-run helper for FoodOps Incident Lab scenarios.

It prints only the agent-visible observation block and a simple prompt preview,
so we can sanity-check the spec/observation split before wiring scenarios into
the full environment.

Usage:
    python scripts/preview_scenario_observation.py scenarios/stuck_promo_after_campaign_end.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def build_prompt(observation: dict) -> str:
    user_role = observation.get("user_role", "Unknown user")
    complaint = observation.get("user_complaint", "")
    shown_kpis = observation.get("shown_kpis", [])
    shown_time_range = observation.get("shown_time_range", "")
    return (
        "You are the incident agent for FoodOps Incident Lab.\n"
        "Only use the user-visible observation below.\n\n"
        f"user_role: {user_role}\n"
        f"user_complaint: {complaint}\n"
        f"shown_kpis: {shown_kpis}\n"
        f"shown_time_range: {shown_time_range}\n"
    )


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: python scripts/preview_scenario_observation.py <scenario.json>")
        return 1

    scenario_path = Path(sys.argv[1])
    scenario = json.loads(scenario_path.read_text())
    observation = scenario.get("observation", {})

    print("=== Agent-visible observation ===")
    print(json.dumps(observation, indent=2))
    print()
    print("=== Prompt preview ===")
    print(build_prompt(observation))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
