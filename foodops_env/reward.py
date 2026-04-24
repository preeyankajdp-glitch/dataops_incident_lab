"""Reward helpers for the FoodOps RL environment."""

from __future__ import annotations

from typing import Any

from .scenarios import ScenarioInstance


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


def expected_terminal_category(instance: ScenarioInstance) -> str:
    if instance.correct_terminal_tool in WRITE_TOOLS:
        return "remediate"
    if instance.correct_terminal_tool in ESCALATION_TOOLS:
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
    """Return sub-scores for a single step."""
    if malformed:
        return {
            "step_cost": 0.0,
            "repeat_penalty": 0.0,
            "phase_bonus": 0.0,
            "disambiguation_bonus": 0.0,
            "terminal_reward": 0.0,
            "guardrail_penalty": 0.0,
            "malformed_penalty": MALFORMED_ACTION_REWARD,
            "budget_penalty": 0.0,
            "total": MALFORMED_ACTION_REWARD,
        }

    if budget_exceeded:
        return {
            "step_cost": 0.0,
            "repeat_penalty": 0.0,
            "phase_bonus": 0.0,
            "disambiguation_bonus": 0.0,
            "terminal_reward": 0.0,
            "guardrail_penalty": 0.0,
            "malformed_penalty": 0.0,
            "budget_penalty": BUDGET_EXCEEDED_REWARD,
            "total": BUDGET_EXCEEDED_REWARD,
        }

    assert action is not None
    tool_name = action.get("tool", "")
    category = tool_category(tool_name)

    step_cost = STEP_COST
    repeat_penalty = 0.0
    phase_bonus = 0.0
    disambiguation_bonus = 0.0
    terminal_reward = 0.0
    guardrail_penalty = 0.0

    prior_count = sum(1 for record in trajectory if record.get("tool") == tool_name)
    if prior_count >= 2:
        repeat_penalty += REPEAT_PENALTY

    if category == "read":
        phase_flags["has_read"] = True
        if (
            not phase_flags.get("disambiguated", False)
            and scenario.template.disambiguating_matcher(action, tool_result or {}, scenario)
        ):
            disambiguation_bonus += DISAMBIGUATION_BONUS
            phase_flags["disambiguated"] = True

    if category in {"remediate", "escalate", "forbidden"}:
        if phase_flags.get("has_read") and not phase_flags.get("phase_bonus_awarded"):
            phase_bonus += PHASE_BONUS
            phase_flags["phase_bonus_awarded"] = True
        phase_flags["has_acted"] = True

        if category == "forbidden":
            guardrail_penalty += FORBIDDEN_TOOL_REWARD
        elif tool_name == scenario.correct_terminal_tool:
            terminal_reward += CORRECT_TERMINAL_REWARD
        elif category == expected_terminal_category(scenario):
            terminal_reward += PARTIAL_TERMINAL_REWARD
        else:
            terminal_reward += WRONG_CATEGORY_REWARD

    total = step_cost + repeat_penalty + phase_bonus + disambiguation_bonus + terminal_reward + guardrail_penalty
    return {
        "step_cost": step_cost,
        "repeat_penalty": repeat_penalty,
        "phase_bonus": phase_bonus,
        "disambiguation_bonus": disambiguation_bonus,
        "terminal_reward": terminal_reward,
        "guardrail_penalty": guardrail_penalty,
        "malformed_penalty": 0.0,
        "budget_penalty": 0.0,
        "total": total,
    }

