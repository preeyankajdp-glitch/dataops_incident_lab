from __future__ import annotations

import random

from foodops_env.env import FoodOpsEnv
from foodops_env.reward import compute_step_breakdown
from foodops_env.scenarios import SCENARIO_BY_ID, SCENARIO_TEMPLATES, sample_scenario


def _seed_for_scenario(target_id: str, limit: int = 500) -> int:
    for seed in range(limit):
        instance = sample_scenario(random.Random(seed))
        if instance.scenario_id == target_id:
            return seed
    raise AssertionError(f"Could not find seed for scenario {target_id}")


def test_reset_seed_is_deterministic():
    env_a = FoodOpsEnv()
    env_b = FoodOpsEnv()
    obs_a, info_a = env_a.reset(seed=42)
    obs_b, info_b = env_b.reset(seed=42)
    assert obs_a == obs_b
    assert info_a["scenario_id"] == info_b["scenario_id"]


def test_each_scenario_type_is_reachable():
    seen = {sample_scenario(random.Random(seed)).scenario_id for seed in range(200)}
    assert seen == {template.scenario_id for template in SCENARIO_TEMPLATES}


def test_guardrail_penalty_fires_for_forbidden_tool():
    seed = _seed_for_scenario("commission_drift")
    env = FoodOpsEnv()
    env.reset(seed=seed)
    _, reward, terminated, truncated, info = env.step(
        {"tool": "update_commission_rate", "args": {"channel": "Zomato", "rate": 19.0}}
    )
    assert terminated is True
    assert truncated is False
    assert info["reward_breakdown"]["guardrail_penalty"] == -1.0
    assert reward <= -1.0


def test_phase_bonus_fires_once_not_twice():
    scenario = sample_scenario(random.Random(42))
    phase_flags = {
        "has_read": False,
        "has_acted": False,
        "phase_bonus_awarded": False,
        "disambiguated": False,
    }
    trajectory: list[dict] = []

    read_action = {"tool": "get_kpi_summary", "args": {"brand": scenario.params["brand"], "channel": scenario.params["channel"]}}
    read_breakdown = compute_step_breakdown(
        action=read_action,
        tool_result={},
        scenario=scenario,
        trajectory=trajectory,
        phase_flags=phase_flags,
    )
    trajectory.append({"tool": read_action["tool"]})
    assert read_breakdown["phase_bonus"] == 0.0

    first_terminal = {"tool": "escalate_to_ops", "args": {"reason": "checking one-shot bonus"}}
    first_terminal_breakdown = compute_step_breakdown(
        action=first_terminal,
        tool_result={},
        scenario=scenario,
        trajectory=trajectory,
        phase_flags=phase_flags,
    )
    trajectory.append({"tool": first_terminal["tool"]})
    assert first_terminal_breakdown["phase_bonus"] == 0.15

    second_terminal_breakdown = compute_step_breakdown(
        action=first_terminal,
        tool_result={},
        scenario=scenario,
        trajectory=trajectory,
        phase_flags=phase_flags,
    )
    assert second_terminal_breakdown["phase_bonus"] == 0.0


def test_disambiguation_bonus_only_for_correct_tool():
    seed = _seed_for_scenario("stale_menu_sync")
    env = FoodOpsEnv()
    env.reset(seed=seed)
    brand = env.anomaly.params["brand"]
    channel = env.anomaly.params["channel"]

    wrong_obs, wrong_reward, wrong_terminated, wrong_truncated, wrong_info = env.step(
        {"tool": "check_config", "args": {"brand": brand, "channel": channel}}
    )
    assert wrong_terminated is False
    assert wrong_truncated is False
    assert wrong_info["reward_breakdown"]["disambiguation_bonus"] == 0.0

    env = FoodOpsEnv()
    env.reset(seed=seed)
    _, _, _, _, right_info = env.step(
        {
            "tool": "check_pipeline_freshness",
            "args": {"table_name": "menu_sync", "brand": brand, "channel": channel},
        }
    )
    assert right_info["reward_breakdown"]["disambiguation_bonus"] == 0.20


def test_same_complaint_can_map_to_multiple_scenarios():
    complaint_to_scenarios: dict[str, set[str]] = {}
    for seed in range(300):
        instance = sample_scenario(random.Random(seed))
        complaint_to_scenarios.setdefault(instance.user_complaint, set()).add(instance.scenario_id)
    assert any(len(scenarios) >= 2 for scenarios in complaint_to_scenarios.values())


def test_random_agent_reward_bounds():
    all_rewards = []
    for scenario_id in SCENARIO_BY_ID:
        seed_start = _seed_for_scenario(scenario_id)
        for seed in range(seed_start, seed_start + 3):
            env = FoodOpsEnv()
            env.reset(seed=seed)
            brand = env.anomaly.params["brand"]
            channel = env.anomaly.params["channel"]
            actions = [
                {"tool": "get_kpi_summary", "args": {"brand": brand, "channel": channel, "range": "current"}},
                {"tool": "check_config", "args": {"brand": brand, "channel": channel}},
                {"tool": "escalate_to_ops", "args": {"reason": "random policy escalation"}},
            ]
            reward_total = 0.0
            for action in actions:
                _, reward, terminated, truncated, _ = env.step(action)
                reward_total += reward
                if terminated or truncated:
                    break
            all_rewards.append(reward_total)
    mean_reward = sum(all_rewards) / len(all_rewards)
    assert -1.2 <= mean_reward <= 1.2

