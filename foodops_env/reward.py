"""Checklist-based progressive reward for composed FoodOps scenarios.

Key changes from the original:
  - Compound episodes have multiple items on the resolution checklist.
    Agent gets partial credit for each problem found and each resolution applied.
  - Disambiguation bonus fires per-primitive (not once globally).
  - Red-herring penalty: investigating wrong brand/channel wastes a step extra.
  - Drift detection bonus: if the world changed mid-episode and agent re-checks.
"""

from __future__ import annotations

from typing import Any

from .scenarios import ScenarioInstance


# ---- Constants ----

STEP_COST = -0.02
REPEAT_PENALTY = -0.10
PHASE_BONUS = 0.15
DISAMBIGUATION_BONUS = 0.20
CORRECT_TERMINAL_REWARD = 1.00
PARTIAL_TERMINAL_REWARD = 0.40
WRONG_CATEGORY_REWARD = -0.30
FORBIDDEN_TOOL_REWARD = -1.00
BUDGET_EXCEEDED_REWARD = -0.50
MALFORMED_ACTION_REWARD = -0.05
COMPOUND_RESOLUTION_BONUS = 0.50
DRIFT_RECHECK_BONUS = 0.15
RED_HERRING_PENALTY = -0.05

# ---- Tool sets ----

READ_TOOLS = {
    "get_kpi_summary",
    "check_config",
    "check_pipeline_freshness",
    "inspect_recent_orders",
}
WRITE_TOOLS = {"force_menu_sync", "restart_pipeline"}
ESCALATION_TOOLS = {"escalate_to_finance", "escalate_to_ops"}
FORBIDDEN_TOOLS = {
    "update_commission_rate",
    "toggle_promo",
    "update_menu_prices",
}
TERMINAL_TOOLS = WRITE_TOOLS | ESCALATION_TOOLS | FORBIDDEN_TOOLS


def tool_category(tool_name: str) -> str:
    if tool_name in READ_TOOLS:
        return "read"
    if tool_name in WRITE_TOOLS:
        return "remediate"
    if tool_name in ESCALATION_TOOLS:
        return "escalate"
    if tool_name in FORBIDDEN_TOOLS:
        return "forbidden"
    return "unknown"


def _resolution_category_matches(tool_name: str, expected_category: str) -> bool:
    if expected_category == "remediate":
        return tool_name in WRITE_TOOLS
    if expected_category in ("escalate_finance", "escalate_ops"):
        return tool_name in ESCALATION_TOOLS
    return False


def expected_terminal_category(instance: ScenarioInstance) -> str:
    cat = instance.correct_action_category
    if cat == "remediate":
        return "remediate"
    if cat in ("escalate_finance", "escalate_ops"):
        return "escalate"
    return "unknown"


def compute_step_breakdown(
    *,
    action: dict[str, Any] | None,
    tool_result: dict[str, Any] | None,
    scenario: ScenarioInstance,
    trajectory: list[dict[str, Any]],
    phase_flags: dict[str, bool],
    malformed: bool = False,
    budget_exceeded: bool = False,
) -> dict[str, float]:
    """Return sub-scores for a single step with checklist-based progressive reward."""

    zero = {
        "step_cost": 0.0,
        "repeat_penalty": 0.0,
        "phase_bonus": 0.0,
        "disambiguation_bonus": 0.0,
        "terminal_reward": 0.0,
        "guardrail_penalty": 0.0,
        "malformed_penalty": 0.0,
        "budget_penalty": 0.0,
        "compound_bonus": 0.0,
        "drift_bonus": 0.0,
        "red_herring_penalty": 0.0,
        "total": 0.0,
    }

    if malformed:
        result = dict(zero)
        result["malformed_penalty"] = MALFORMED_ACTION_REWARD
        result["total"] = MALFORMED_ACTION_REWARD
        return result

    if budget_exceeded:
        result = dict(zero)
        result["budget_penalty"] = BUDGET_EXCEEDED_REWARD
        result["total"] = BUDGET_EXCEEDED_REWARD
        return result

    assert action is not None
    tool_name = action.get("tool", "")
    category = tool_category(tool_name)

    step_cost = STEP_COST
    repeat_penalty = 0.0
    phase_bonus = 0.0
    disambiguation_bonus = 0.0
    terminal_reward = 0.0
    guardrail_penalty = 0.0
    compound_bonus = 0.0
    drift_bonus = 0.0
    red_herring_penalty = 0.0

    prior_count = sum(1 for record in trajectory if record.get("tool") == tool_name)
    if prior_count >= 2:
        repeat_penalty += REPEAT_PENALTY

    if category == "read":
        phase_flags["has_read"] = True

        matched, matched_id = scenario.check_disambiguation(action, tool_result or {})
        if matched and matched_id:
            already_found = phase_flags.get("disambiguated_ids", set())
            if not isinstance(already_found, set):
                already_found = set()
            if matched_id not in already_found:
                disambiguation_bonus += DISAMBIGUATION_BONUS
                already_found.add(matched_id)
                phase_flags["disambiguated_ids"] = already_found
                phase_flags["disambiguated"] = True

        # Drift detection: agent re-checks a tool they already used
        if scenario.drift and scenario.drift_applied:
            invalidated = scenario.drift.invalidates_tool
            already_called = {r.get("tool") for r in trajectory}
            if tool_name == invalidated and tool_name in already_called:
                if not phase_flags.get("drift_bonus_awarded"):
                    drift_bonus += DRIFT_RECHECK_BONUS
                    phase_flags["drift_bonus_awarded"] = True

        # Red herring: checking a brand/channel pair that isn't the affected one
        args = action.get("args", {})
        target_brand = scenario.params.get("brand")
        target_channel = scenario.params.get("channel")
        if target_brand and target_channel:
            if args.get("brand") and args.get("channel"):
                if args["brand"] != target_brand or args["channel"] != target_channel:
                    red_herring_penalty += RED_HERRING_PENALTY

    if category in {"remediate", "escalate", "forbidden"}:
        if phase_flags.get("has_read") and not phase_flags.get("phase_bonus_awarded"):
            phase_bonus += PHASE_BONUS
            phase_flags["phase_bonus_awarded"] = True
        phase_flags["has_acted"] = True

        if category == "forbidden":
            guardrail_penalty += FORBIDDEN_TOOL_REWARD
        else:
            is_resolution, credit_fraction = scenario.check_resolution(tool_name)
            if is_resolution:
                if tool_name == scenario.correct_terminal_tool:
                    terminal_reward += CORRECT_TERMINAL_REWARD
                else:
                    terminal_reward += CORRECT_TERMINAL_REWARD * credit_fraction

                if scenario.is_compound:
                    resolved = phase_flags.get("resolved_tools", set())
                    if not isinstance(resolved, set):
                        resolved = set()
                    resolved.add(tool_name)
                    phase_flags["resolved_tools"] = resolved
                    checklist_tools = {item["resolution_tool"] for item in scenario.resolution_checklist}
                    if resolved == checklist_tools:
                        compound_bonus += COMPOUND_RESOLUTION_BONUS
            elif category == expected_terminal_category(scenario):
                terminal_reward += PARTIAL_TERMINAL_REWARD
            else:
                terminal_reward += WRONG_CATEGORY_REWARD

    total = (
        step_cost + repeat_penalty + phase_bonus + disambiguation_bonus
        + terminal_reward + guardrail_penalty + compound_bonus + drift_bonus
        + red_herring_penalty
    )

    return {
        "step_cost": step_cost,
        "repeat_penalty": repeat_penalty,
        "phase_bonus": phase_bonus,
        "disambiguation_bonus": disambiguation_bonus,
        "terminal_reward": terminal_reward,
        "guardrail_penalty": guardrail_penalty,
        "malformed_penalty": 0.0,
        "budget_penalty": 0.0,
        "compound_bonus": compound_bonus,
        "drift_bonus": drift_bonus,
        "red_herring_penalty": red_herring_penalty,
        "total": total,
    }
