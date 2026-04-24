from __future__ import annotations

import random

from foodops_env.env import FoodOpsEnv
from foodops_env.primitives import PRIMITIVES, PRIMITIVE_BY_ID, COMPATIBLE_COMPOUNDS
from foodops_env.reward import compute_step_breakdown, FORBIDDEN_TOOLS
from foodops_env.scenarios import (
    ScenarioInstance,
    sample_scenario,
    compose_single,
    compose_compound,
    compose_with_drift,
)


def _seed_for_scenario(target_id: str, limit: int = 500) -> int:
    """Find a seed that produces a scenario containing the given primitive."""
    for seed in range(limit):
        instance = sample_scenario(random.Random(seed))
        if target_id in instance.scenario_id:
            return seed
    raise AssertionError(f"Could not find seed for scenario {target_id}")


def test_reset_seed_is_deterministic():
    env_a = FoodOpsEnv()
    env_b = FoodOpsEnv()
    obs_a, info_a = env_a.reset(seed=42)
    obs_b, info_b = env_b.reset(seed=42)
    assert obs_a == obs_b
    assert info_a["scenario_id"] == info_b["scenario_id"]


def test_all_12_primitives_are_registered():
    assert len(PRIMITIVES) == 12
    ids = {p.primitive_id for p in PRIMITIVES}
    expected = {
        "stuck_promo", "stale_menu_sync", "commission_drift",
        "ingestion_lag", "duplicate_orders", "missing_feed",
        "brand_paused", "discount_spike", "refund_spike",
        "aov_collapse", "partial_aggregation", "false_alarm",
    }
    assert ids == expected


def test_single_scenarios_are_reachable():
    seen = set()
    for seed in range(500):
        instance = compose_single(random.Random(seed))
        seen.add(instance.scenario_id)
    assert len(seen) == 12


def test_compound_scenarios_produce_multiple_primitives():
    for seed in range(50):
        instance = compose_compound(random.Random(seed))
        assert instance.is_compound
        assert len(instance.primitives) == 2
        assert len(instance.resolution_checklist) >= 1


def test_drift_scenarios_have_drift():
    for seed in range(20):
        instance = compose_with_drift(random.Random(seed))
        assert instance.drift is not None
        assert not instance.drift_applied


def test_composition_distribution():
    """50% single, 30% compound, 20% drift."""
    counts = {"single": 0, "compound": 0, "drift": 0}
    for seed in range(1000):
        instance = sample_scenario(random.Random(seed))
        if instance.drift:
            counts["drift"] += 1
        elif instance.is_compound:
            counts["compound"] += 1
        else:
            counts["single"] += 1
    assert counts["single"] > 350
    assert counts["compound"] > 200
    assert counts["drift"] > 100


def test_guardrail_penalty_fires_for_forbidden_tool():
    env = FoodOpsEnv()
    env.reset(seed=42)
    _, reward, terminated, truncated, info = env.step(
        {"tool": "update_commission_rate", "args": {"channel": "Zomato", "rate": 19.0}}
    )
    assert terminated is True
    assert truncated is False
    assert info["reward_breakdown"]["guardrail_penalty"] == -1.0
    assert reward <= -1.0


def test_phase_bonus_fires_once():
    env = FoodOpsEnv()
    env.reset(seed=42)
    brand = env.anomaly.primitive_params[0]["brand"]
    channel = env.anomaly.primitive_params[0]["channel"]

    env.step({"tool": "check_config", "args": {"brand": brand, "channel": channel}})
    _, _, _, _, info = env.step(
        {"tool": "escalate_to_ops", "args": {"reason": "test"}}
    )
    assert info["reward_breakdown"]["phase_bonus"] == 0.15


def test_same_complaint_maps_to_multiple_scenarios():
    complaint_to_scenarios: dict[str, set[str]] = {}
    for seed in range(500):
        instance = sample_scenario(random.Random(seed))
        complaint_to_scenarios.setdefault(instance.user_complaint, set()).add(instance.scenario_id)
    assert any(len(s) >= 2 for s in complaint_to_scenarios.values())


def test_compound_episode_does_not_terminate_on_first_resolution():
    """For a compound episode with 2 distinct resolution tools, first terminal action should not end episode."""
    for seed in range(200):
        instance = compose_compound(random.Random(seed))
        checklist_tools = {item["resolution_tool"] for item in instance.resolution_checklist}
        if len(checklist_tools) >= 2:
            env = FoodOpsEnv()
            env.reset(seed=seed)
            # Force the env to use this compound scenario
            env.anomaly = instance
            for prim, params in zip(instance.primitives, instance.primitive_params):
                prim.inject_fn(env.world, params, env.rng)
            env._pending_resolutions = set(checklist_tools)

            brand = instance.primitive_params[0]["brand"]
            channel = instance.primitive_params[0]["channel"]

            env.step({"tool": "check_config", "args": {"brand": brand, "channel": channel}})
            first_tool = list(checklist_tools)[0]
            if first_tool == "force_menu_sync":
                _, _, terminated, _, _ = env.step(
                    {"tool": first_tool, "args": {"brand": brand, "channel": channel}}
                )
            elif first_tool == "restart_pipeline":
                _, _, terminated, _, _ = env.step(
                    {"tool": first_tool, "args": {"pipeline_name": "orders_ingest"}}
                )
            else:
                _, _, terminated, _, _ = env.step(
                    {"tool": first_tool, "args": {"reason": "compound test"}}
                )

            assert not terminated, f"Episode terminated on first resolution of compound with tools {checklist_tools}"
            return
    raise AssertionError("No compound scenario with 2 distinct resolution tools found")


def test_drift_changes_world_state():
    env = FoodOpsEnv()
    env.reset(seed=42)
    brand = env.anomaly.primitive_params[0]["brand"]
    channel = env.anomaly.primitive_params[0]["channel"]

    from foodops_env.scenarios import DriftEvent, _drift_commission_bump
    drift = DriftEvent(
        drift_id="test_drift",
        description="test",
        trigger_step=2,
        apply_fn=_drift_commission_bump,
        invalidates_tool="check_config",
    )
    env.anomaly.drift = drift

    config_before = env.world["channel_brand_config"][(brand, channel)]["effective_commission_pct"]
    env.step({"tool": "check_config", "args": {"brand": brand, "channel": channel}})
    env.step({"tool": "get_kpi_summary", "args": {"brand": brand, "channel": channel, "range": "current"}})
    config_after = env.world["channel_brand_config"][(brand, channel)]["effective_commission_pct"]
    assert config_after > config_before


def test_red_herring_penalty():
    env = FoodOpsEnv()
    env.reset(seed=42)
    target_brand = env.anomaly.primitive_params[0]["brand"]
    target_channel = env.anomaly.primitive_params[0]["channel"]

    wrong_brand = [b for b in ["Behrouz Biryani", "Faasos", "Oven Story Pizza"] if b != target_brand][0]
    _, _, _, _, info = env.step(
        {"tool": "check_config", "args": {"brand": wrong_brand, "channel": target_channel}}
    )
    assert info["reward_breakdown"]["red_herring_penalty"] < 0


def test_random_agent_reward_bounds():
    all_rewards = []
    for seed in range(20):
        env = FoodOpsEnv()
        obs, info = env.reset(seed=seed)
        brand = obs["dashboard_kpis"]["brand"]
        channel = obs["dashboard_kpis"]["channel"]
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
    assert -2.0 <= mean_reward <= 2.0


def test_scenario_instance_backward_compat():
    """The new ScenarioInstance still exposes the old-style properties."""
    instance = compose_single(random.Random(42))
    assert instance.template is instance
    assert "brand" in instance.params
    assert "channel" in instance.params
    assert isinstance(instance.disambiguating_tool, dict)
    assert instance.correct_terminal_tool is not None
    assert instance.correct_action_category is not None
