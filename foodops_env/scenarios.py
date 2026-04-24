"""Scenario templates for the FoodOps RL environment.

The core mechanic here is ambiguity by construction:
the same complaint pool is reused across every scenario, so the complaint
alone never identifies the root cause. Only tool use can disambiguate.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import random
from typing import Any, Callable


SHARED_COMPLAINT_POOL = [
    "Net payout on {brand} × {channel} dropped this week. Is this real or is the dashboard off?",
    "{brand} × {channel} looks weaker than expected. Margins are down and I can't tell why.",
    "Revenue from {brand} on {channel} is below where I thought it'd be. Can you dig in?",
    "{brand} × {channel} numbers feel off — payout per order is lower than last month.",
]

BRANDS = [
    "Behrouz Biryani",
    "Oven Story Pizza",
    "Faasos",
    "Sweet Truth",
    "The Bowl Company",
]

CHANNELS = ["Zomato", "Swiggy"]


WorldState = dict[str, Any]
ScenarioParameterizationFn = Callable[[random.Random], dict[str, Any]]
ScenarioInjectFn = Callable[[WorldState, dict[str, Any]], None]
ScenarioMatcherFn = Callable[[dict[str, Any], dict[str, Any], "ScenarioInstance"], bool]


@dataclass(frozen=True)
class ScenarioTemplate:
    scenario_id: str
    anomaly_category: str
    correct_terminal_tool: str
    correct_action_category: str
    disambiguating_tool: dict[str, Any]
    parameterization_fn: ScenarioParameterizationFn
    inject_fn: ScenarioInjectFn
    story_seed: str
    disambiguating_matcher: ScenarioMatcherFn


@dataclass(frozen=True)
class ScenarioInstance:
    template: ScenarioTemplate
    params: dict[str, Any]
    user_complaint: str

    @property
    def scenario_id(self) -> str:
        return self.template.scenario_id

    @property
    def anomaly_category(self) -> str:
        return self.template.anomaly_category

    @property
    def correct_terminal_tool(self) -> str:
        return self.template.correct_terminal_tool

    @property
    def correct_action_category(self) -> str:
        return self.template.correct_action_category

    @property
    def disambiguating_tool(self) -> dict[str, Any]:
        return self.template.disambiguating_tool

    @property
    def story_seed(self) -> str:
        return self.template.story_seed


def _pick_brand_channel(rng: random.Random) -> tuple[str, str]:
    return rng.choice(BRANDS), rng.choice(CHANNELS)


def _param_stuck_promo(rng: random.Random) -> dict[str, Any]:
    brand, channel = _pick_brand_channel(rng)
    return {
        "brand": brand,
        "channel": channel,
        "promo_discount_pct": rng.randint(10, 25),
        "days_overdue": rng.randint(3, 14),
        "payout_drop_pct": round(rng.uniform(0.10, 0.18), 4),
    }


def _param_stale_menu_sync(rng: random.Random) -> dict[str, Any]:
    brand, channel = _pick_brand_channel(rng)
    return {
        "brand": brand,
        "channel": channel,
        "hours_stale": rng.randint(24, 96),
        "payout_drop_pct": round(rng.uniform(0.10, 0.18), 4),
    }


def _param_commission_drift(rng: random.Random) -> dict[str, Any]:
    brand, channel = _pick_brand_channel(rng)
    return {
        "brand": brand,
        "channel": channel,
        "commission_delta": round(rng.uniform(2.0, 5.0), 2),
        "payout_drop_pct": round(rng.uniform(0.10, 0.18), 4),
    }


def _make_complaint(rng: random.Random, brand: str, channel: str) -> str:
    return rng.choice(SHARED_COMPLAINT_POOL).format(brand=brand, channel=channel)


def _current_pair(world: WorldState, brand: str, channel: str) -> dict[str, Any]:
    return world["current_reference"][(brand, channel)]


def _baseline_pair(world: WorldState, brand: str, channel: str) -> dict[str, Any]:
    return world["baseline_reference"][(brand, channel)]


def _rebuild_recent_orders(
    world: WorldState,
    brand: str,
    channel: str,
    *,
    gross_sales: float,
    discount_amount: float,
    net_sales: float,
    net_payout: float,
    total_orders: int,
) -> None:
    current_orders = world["recent_orders"][(brand, channel)]
    if not current_orders:
        return
    total_orders = max(1, total_orders)
    gross_per_order = gross_sales / total_orders
    discount_per_order = discount_amount / total_orders
    net_per_order = net_sales / total_orders
    payout_per_order = net_payout / total_orders
    for idx, order in enumerate(current_orders):
        order["gross_amount"] = round(gross_per_order, 2)
        order["discount_amount"] = round(discount_per_order, 2)
        order["net_sales"] = round(net_per_order, 2)
        order["net_payout"] = round(payout_per_order, 2)
        order["customer_segment"] = "repeat" if idx % 3 else "new"


def _inject_stuck_promo(world: WorldState, params: dict[str, Any]) -> None:
    brand = params["brand"]
    channel = params["channel"]
    drop_pct = params["payout_drop_pct"]
    baseline = _baseline_pair(world, brand, channel)
    current = _current_pair(world, brand, channel)
    config = world["channel_brand_config"][(brand, channel)]

    config["promo_active"] = True
    config["promo_discount_pct"] = float(params["promo_discount_pct"])
    config["promo_expected_to_end_at"] = world["now"] - timedelta(days=params["days_overdue"])
    config["promo_overdue_days"] = params["days_overdue"]

    current["total_orders"] = max(1, round(baseline["total_orders"] * (1 + world["rng"].uniform(-0.06, 0.06))))
    current["gross_sales"] = round(baseline["gross_sales"] * (1 + world["rng"].uniform(-0.05, 0.05)), 2)
    current["discount_amount"] = round(
        baseline["discount_amount"] + (current["gross_sales"] * config["promo_discount_pct"] / 100.0),
        2,
    )
    current["net_sales"] = round(max(0.0, baseline["net_sales"] - (baseline["net_payout"] * drop_pct * 0.9)), 2)
    current["net_payout"] = round(baseline["net_payout"] * (1.0 - drop_pct), 2)
    current["aov"] = round(current["gross_sales"] / max(current["total_orders"], 1), 2)
    _rebuild_recent_orders(
        world,
        brand,
        channel,
        gross_sales=current["gross_sales"],
        discount_amount=current["discount_amount"],
        net_sales=current["net_sales"],
        net_payout=current["net_payout"],
        total_orders=current["total_orders"],
    )


def _inject_stale_menu_sync(world: WorldState, params: dict[str, Any]) -> None:
    brand = params["brand"]
    channel = params["channel"]
    drop_pct = params["payout_drop_pct"]
    baseline = _baseline_pair(world, brand, channel)
    current = _current_pair(world, brand, channel)
    config = world["channel_brand_config"][(brand, channel)]

    config["promo_active"] = False
    config["promo_discount_pct"] = 0.0
    config["menu_last_synced_at"] = world["now"] - timedelta(hours=params["hours_stale"])
    world["pipeline_freshness"]["menu_sync"][(brand, channel)] = config["menu_last_synced_at"]

    current["total_orders"] = max(1, round(baseline["total_orders"] * (1 + world["rng"].uniform(-0.08, 0.08))))
    current["gross_sales"] = round(baseline["gross_sales"] * (1 + world["rng"].uniform(-0.05, 0.02)), 2)
    current["aov"] = round(baseline["aov"] * (1.0 - drop_pct * 0.75), 2)
    current["net_sales"] = round(baseline["net_sales"] * (1.0 - drop_pct * 0.7), 2)
    current["net_payout"] = round(baseline["net_payout"] * (1.0 - drop_pct), 2)
    current["discount_amount"] = round(max(0.0, current["gross_sales"] - current["net_sales"]), 2)
    _rebuild_recent_orders(
        world,
        brand,
        channel,
        gross_sales=current["gross_sales"],
        discount_amount=current["discount_amount"],
        net_sales=current["net_sales"],
        net_payout=current["net_payout"],
        total_orders=current["total_orders"],
    )


def _inject_commission_drift(world: WorldState, params: dict[str, Any]) -> None:
    brand = params["brand"]
    channel = params["channel"]
    drop_pct = params["payout_drop_pct"]
    baseline = _baseline_pair(world, brand, channel)
    current = _current_pair(world, brand, channel)
    config = world["channel_brand_config"][(brand, channel)]

    config["promo_active"] = False
    config["promo_discount_pct"] = 0.0
    config["effective_commission_pct"] = round(config["baseline_commission_pct"] + params["commission_delta"], 2)

    current["total_orders"] = max(1, round(baseline["total_orders"] * (1 + world["rng"].uniform(-0.05, 0.05))))
    current["gross_sales"] = round(baseline["gross_sales"] * (1 + world["rng"].uniform(-0.04, 0.04)), 2)
    current["discount_amount"] = baseline["discount_amount"]
    current["net_sales"] = round(baseline["net_sales"] * (1 + world["rng"].uniform(-0.04, 0.04)), 2)
    current["net_payout"] = round(baseline["net_payout"] * (1.0 - drop_pct), 2)
    current["aov"] = round(current["gross_sales"] / max(current["total_orders"], 1), 2)
    current["effective_commission_pct"] = config["effective_commission_pct"]
    _rebuild_recent_orders(
        world,
        brand,
        channel,
        gross_sales=current["gross_sales"],
        discount_amount=current["discount_amount"],
        net_sales=current["net_sales"],
        net_payout=current["net_payout"],
        total_orders=current["total_orders"],
    )


def _matches_stuck_promo(action: dict[str, Any], result: dict[str, Any], instance: ScenarioInstance) -> bool:
    return (
        action.get("tool") == "check_config"
        and action.get("args", {}).get("brand") == instance.params["brand"]
        and action.get("args", {}).get("channel") == instance.params["channel"]
        and bool(result.get("promo_active"))
        and float(result.get("promo_discount_pct", 0.0)) > 0.0
    )


def _matches_stale_menu_sync(action: dict[str, Any], result: dict[str, Any], instance: ScenarioInstance) -> bool:
    freshness = result.get("freshness_at")
    if action.get("tool") != "check_pipeline_freshness":
        return False
    if action.get("args", {}).get("table_name") != "menu_sync":
        return False
    return (
        action.get("args", {}).get("brand") == instance.params["brand"]
        and action.get("args", {}).get("channel") == instance.params["channel"]
        and bool(result.get("is_stale"))
        and freshness is not None
    )


def _matches_commission_drift(action: dict[str, Any], result: dict[str, Any], instance: ScenarioInstance) -> bool:
    return (
        action.get("tool") == "check_config"
        and action.get("args", {}).get("brand") == instance.params["brand"]
        and action.get("args", {}).get("channel") == instance.params["channel"]
        and float(result.get("effective_commission_pct", 0.0)) > float(result.get("baseline_commission_pct", 0.0))
    )


SCENARIO_TEMPLATES = [
    ScenarioTemplate(
        scenario_id="stuck_promo",
        anomaly_category="business",
        correct_terminal_tool="escalate_to_finance",
        correct_action_category="escalate_finance",
        disambiguating_tool={
            "tool": "check_config",
            "field": "promo_active",
            "brand_param": "brand",
            "channel_param": "channel",
        },
        parameterization_fn=_param_stuck_promo,
        inject_fn=_inject_stuck_promo,
        story_seed="Promo flag stayed on after the campaign ended.",
        disambiguating_matcher=_matches_stuck_promo,
    ),
    ScenarioTemplate(
        scenario_id="stale_menu_sync",
        anomaly_category="pipeline",
        correct_terminal_tool="force_menu_sync",
        correct_action_category="remediate",
        disambiguating_tool={
            "tool": "check_pipeline_freshness",
            "field": "menu_last_synced_at",
            "table_name": "menu_sync",
            "brand_param": "brand",
            "channel_param": "channel",
        },
        parameterization_fn=_param_stale_menu_sync,
        inject_fn=_inject_stale_menu_sync,
        story_seed="Menu sync fell behind and pricing drifted.",
        disambiguating_matcher=_matches_stale_menu_sync,
    ),
    ScenarioTemplate(
        scenario_id="commission_drift",
        anomaly_category="business",
        correct_terminal_tool="escalate_to_finance",
        correct_action_category="escalate_finance",
        disambiguating_tool={
            "tool": "check_config",
            "field": "effective_commission_pct",
            "brand_param": "brand",
            "channel_param": "channel",
        },
        parameterization_fn=_param_commission_drift,
        inject_fn=_inject_commission_drift,
        story_seed="Channel commission changed quietly and margin shrank.",
        disambiguating_matcher=_matches_commission_drift,
    ),
]

SCENARIO_BY_ID = {template.scenario_id: template for template in SCENARIO_TEMPLATES}


def build_scenario_instance(template: ScenarioTemplate, rng: random.Random) -> ScenarioInstance:
    params = template.parameterization_fn(rng)
    complaint = _make_complaint(rng, params["brand"], params["channel"])
    return ScenarioInstance(template=template, params=params, user_complaint=complaint)


def sample_scenario(rng: random.Random) -> ScenarioInstance:
    """Sample one of the three scenario families uniformly."""
    template = rng.choice(SCENARIO_TEMPLATES)
    return build_scenario_instance(template, rng)
