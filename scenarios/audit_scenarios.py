"""
audit_scenarios.py
==================

Reads every scenario JSON and prints ONLY what the agent will see
(observation.user_complaint + observation.shown_kpis).

Then flags common leakage patterns — e.g., the complaint text mentioning
ground-truth-only information like 'promo was supposed to end on X'.

Run this after every round of scenario authoring. If you can solve any
scenario from the printed output alone, there's leakage.

Usage:
    python audit_scenarios.py scenarios/
"""

import json
import re
import sys
from pathlib import Path

try:
    import jsonschema
except Exception:
    jsonschema = None


# Phrases that suggest the complaint is leaking hidden ground truth.
# These are HEURISTIC — not every hit is a real leak, but each deserves review.
LEAKAGE_PATTERNS = [
    (r"\bstale\b", "mentions 'stale' — agent should diagnose this, not be told"),
    (r"\bschema drift\b", "mentions schema drift as cause — huge leak"),
    (r"\baggregation (job|pipeline) (failed|broken)\b", "names the pipeline failure mode"),
    (r"\bis_live\b|\bpromo_active\b", "references an internal config column name"),
    (r"\bchannel_brand_config\b|\border_daily_metrics\b|\bchannel_daily_metrics\b",
        "references an internal table name"),
    (r"\bcommission.*(changed|renegotiated) (on|at)\b",
        "narrates a business event the agent should infer"),
    (r"\bsupposed to (end|stop) (on|at)\b", "reveals campaign timing ground truth"),
    (r"\bmenu sync (is |was )(stale|broken|behind)\b", "names menu sync as the cause"),
    (r"\brenamed\b|\bdropped column\b", "reveals a schema change"),
    (r"\bdouble[- ]count\b", "names the bug directly"),
]


def load_scenarios(folder: Path):
    files = sorted(folder.glob("*.json"))
    scenarios = []
    for f in files:
        if f.name.startswith("scenario.schema"):
            continue
        with f.open() as fh:
            scenarios.append((f.name, json.load(fh)))
    return scenarios


def load_schema(folder: Path):
    schema_path = folder / "scenario.schema.json"
    if not schema_path.exists():
        return None
    with schema_path.open() as fh:
        return json.load(fh)


def check_structure(name, scn):
    """Ensure observation section has no extraneous keys that could leak state."""
    allowed = {"user_role", "user_complaint", "shown_kpis", "shown_time_range"}
    obs = scn.get("observation", {})
    extra = set(obs.keys()) - allowed
    if extra:
        return [f"observation contains non-allowlisted keys: {extra}"]
    if not obs.get("user_complaint"):
        return ["observation.user_complaint is missing or empty"]
    return []


def check_leakage(name, scn):
    """Scan the user complaint for phrases that leak hidden ground truth."""
    complaint = scn.get("observation", {}).get("user_complaint", "").lower()
    hits = []
    for pat, msg in LEAKAGE_PATTERNS:
        if re.search(pat, complaint):
            hits.append(f"possible leak: {msg!r} (matched /{pat}/)")
    return hits


def check_ground_truth_coverage(name, scn):
    """Every scenario must have a true_root_cause and at least one required_evidence entry."""
    gt = scn.get("ground_truth", {})
    issues = []
    if not gt.get("true_root_cause"):
        issues.append("ground_truth.true_root_cause missing")
    if not gt.get("relevant_tools"):
        issues.append("ground_truth.relevant_tools missing — tool_coverage reward won't work")
    if scn.get("is_null_incident") and gt.get("correct_terminal_action") != "submit_report_no_fix":
        issues.append("is_null_incident=true but correct_terminal_action is not submit_report_no_fix")
    return issues


def check_schema(name, scn, schema):
    if schema is None:
        return []
    if jsonschema is None:
        return ["jsonschema is not installed; schema validation was skipped"]
    try:
        jsonschema.validate(scn, schema)
        return []
    except Exception as exc:
        return [f"schema validation failed: {exc}"]


def audit(folder: Path) -> int:
    scenarios = load_scenarios(folder)
    schema = load_schema(folder)
    if not scenarios:
        print(f"no scenarios found in {folder}")
        return 1

    total_issues = 0
    print(f"auditing {len(scenarios)} scenarios in {folder}\n")
    print("=" * 72)

    for name, scn in scenarios:
        print(f"\n── {name} ──")
        obs = scn.get("observation", {})
        print(f"  user_role:       {obs.get('user_role', '(none)')}")
        print(f"  user_complaint:  {obs.get('user_complaint', '(MISSING)')}")
        print(f"  shown_kpis:      {obs.get('shown_kpis', '(none)')}")
        print(f"  time_range:      {obs.get('shown_time_range', '(none)')}")

        issues = []
        issues += check_schema(name, scn, schema)
        issues += check_structure(name, scn)
        issues += check_leakage(name, scn)
        issues += check_ground_truth_coverage(name, scn)

        if issues:
            print("  ⚠ issues:")
            for i in issues:
                print(f"    - {i}")
            total_issues += len(issues)
        else:
            print("  ✓ clean")

    print("\n" + "=" * 72)
    print(f"total issues: {total_issues}")
    print("\nManual check: read the 'user_complaint' lines above. If you can")
    print("solve any scenario from the complaint alone without needing tools,")
    print("that complaint has a leak even if no pattern matched.\n")
    return 0 if total_issues == 0 else 1


if __name__ == "__main__":
    folder = Path(sys.argv[1] if len(sys.argv) > 1 else "scenarios")
    sys.exit(audit(folder))
