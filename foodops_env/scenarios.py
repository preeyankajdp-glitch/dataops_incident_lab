"""Scenario composition engine for procedural episode generation.

Replaces the old 3-template system with a composable engine that:
  - Picks 1-3 anomaly primitives per episode
  - Supports compound faults (one masking another)
  - Supports schema drift (world mutates mid-episode)
  - Maps complaint families to ambiguous complaint pools
  - Generates ground-truth checklists for progressive reward
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
import random
from typing import Any

from .primitives import (
    COMPATIBLE_COMPOUNDS,
    PRIMITIVES,
    PRIMITIVE_BY_ID,
    AnomalyPrimitive,
    WorldState,
)


BRANDS = [
    "Behrouz Biryani",
    "Oven Story Pizza",
    "Faasos",
    "Sweet Truth",
    "The Bowl Company",
]

CHANNELS = ["Zomato", "Swiggy"]

COMPLAINT_FAMILIES: dict[str, list[str]] = {
    "payout_drop": [
        "Net payout on {brand} × {channel} dropped this week. Is this real or is the dashboard off?",
        "{brand} × {channel} looks weaker than expected. Margins are down and I can't tell why.",
        "Revenue from {brand} on {channel} is below where I thought it'd be. Can you dig in?",
        "{brand} × {channel} numbers feel off — payout per order is lower than last month.",
    ],
    "numbers_off": [
        "Something looks off with {brand} × {channel} numbers. Orders don't match what I expected.",
        "{brand} on {channel} has weird order data today. Can you check if the feed is okay?",
        "Dashboard for {brand} × {channel} looks inconsistent. Are we counting everything right?",
    ],
    "brand_gone": [
        "{brand} seems to have disappeared from {channel}. Zero orders showing up.",
        "Why is {brand} showing no activity on {channel}? Did someone pause it?",
    ],
    "revenue_flat": [
        "Revenue on {brand} × {channel} is flat even though orders went up. What happened?",
        "Orders up but net sales barely moved for {brand} on {channel}. Something is off.",
    ],
}

# ---- Schema drift events: mid-episode world mutations ----

@dataclass(frozen=True)
class DriftEvent:
    drift_id: str
    description: str
    trigger_step: int  # inject after this many agent steps
    apply_fn: Any  # Callable[[WorldState, str, str, random.Random], None]
    invalidates_tool: str  # which tool's prior results become stale


def _drift_commission_bump(world: WorldState, brand: str, channel: str, rng: random.Random) -> None:
    config = world["channel_brand_config"][(brand, channel)]
    bump = round(rng.uniform(1.5, 3.5), 2)
    config["effective_commission_pct"] = round(config["effective_commission_pct"] + bump, 2)
    current = world["current_reference"][(brand, channel)]
    current["effective_commission_pct"] = config["effective_commission_pct"]
    current["net_payout"] = round(current["net_sales"] * (1.0 - config["effective_commission_pct"] / 100.0), 2)


def _drift_promo_toggled(world: WorldState, brand: str, channel: str, rng: random.Random) -> None:
    config = world["channel_brand_config"][(brand, channel)]
    config["promo_active"] = not config["promo_active"]
    if config["promo_active"]:
        config["promo_discount_pct"] = float(rng.randint(8, 20))
    else:
        config["promo_discount_pct"] = 0.0


def _drift_feed_recovered(world: WorldState, brand: str, channel: str, rng: random.Random) -> None:
    world["pipeline_freshness"]["orders"] = world["now"]
    world["pipeline_freshness"]["menu_sync"][(brand, channel)] = world["now"]
    config = world["channel_brand_config"][(brand, channel)]
    config["menu_last_synced_at"] = world["now"]


DRIFT_EVENTS = [
    DriftEvent("commission_bump", "Channel silently raised commission mid-investigation",
               trigger_step=4, apply_fn=_drift_commission_bump, invalidates_tool="check_config"),
    DriftEvent("promo_toggled", "Someone toggled the promo flag while you were investigating",
               trigger_step=3, apply_fn=_drift_promo_toggled, invalidates_tool="check_config"),
    DriftEvent("feed_recovered", "The stale pipeline recovered on its own mid-investigation",
               trigger_step=5, apply_fn=_drift_feed_recovered, invalidates_tool="check_pipeline_freshness"),
]


# ---- Composed scenario instance ----

@dataclass
class ScenarioInstance:
    primitives: list[AnomalyPrimitive]
    primitive_params: list[dict[str, Any]]
    user_complaint: str
    drift: DriftEvent | None = None
    drift_applied: bool = False

    @property
    def scenario_id(self) -> str:
        ids = [p.primitive_id for p in self.primitives]
        return "+".join(ids)

    @property
    def is_compound(self) -> bool:
        return len(self.primitives) > 1

    @property
    def difficulty(self) -> str:
        n = len(self.primitives)
        if self.drift:
            return "hard"
        if n >= 2:
            return "medium"
        return "easy"

    @property
    def anomaly_category(self) -> str:
        categories = {p.category for p in self.primitives}
        if len(categories) == 1:
            return next(iter(categories))
        return "compound"

    @property
    def resolution_checklist(self) -> list[dict[str, str]]:
        seen: set[str] = set()
        checklist: list[dict[str, str]] = []
        for prim, params in zip(self.primitives, self.primitive_params):
            key = prim.resolution_tool
            if key not in seen:
                seen.add(key)
                checklist.append({
                    "primitive_id": prim.primitive_id,
                    "resolution_tool": prim.resolution_tool,
                    "resolution_category": prim.resolution_category,
                    "diagnostic_tool": prim.diagnostic_tool,
                })
        return checklist

    @property
    def diagnostic_checklist(self) -> list[dict[str, str]]:
        return [
            {
                "primitive_id": prim.primitive_id,
                "diagnostic_tool": prim.diagnostic_tool,
            }
            for prim in self.primitives
        ]

    @property
    def correct_terminal_tool(self) -> str:
        return self.primitives[0].resolution_tool

    @property
    def correct_action_category(self) -> str:
        return self.primitives[0].resolution_category

    @property
    def story_seed(self) -> str:
        return " + ".join(p.story_seed for p in self.primitives)

    # Backward-compat shims for reward.py and env.py
    @property
    def template(self) -> ScenarioInstance:
        return self

    @property
    def params(self) -> dict[str, Any]:
        return self.primitive_params[0]

    @property
    def disambiguating_tool(self) -> dict[str, Any]:
        p = self.primitives[0]
        return {"tool": p.diagnostic_tool}

    def disambiguating_matcher(self, action: dict[str, Any], result: dict[str, Any], _instance: Any) -> bool:
        for prim, params in zip(self.primitives, self.primitive_params):
            if prim.matcher_fn(action, result, params):
                return True
        return False

    def check_disambiguation(self, action: dict[str, Any], result: dict[str, Any]) -> tuple[bool, str | None]:
        for prim, params in zip(self.primitives, self.primitive_params):
            if prim.matcher_fn(action, result, params):
                return True, prim.primitive_id
        return False, None

    def check_resolution(self, tool_name: str) -> tuple[bool, float]:
        for item in self.resolution_checklist:
            if item["resolution_tool"] == tool_name:
                if len(self.resolution_checklist) == 1:
                    return True, 1.0
                return True, 1.0 / len(self.resolution_checklist)
        return False, 0.0


def _pick_brand_channel(rng: random.Random) -> tuple[str, str]:
    return rng.choice(BRANDS), rng.choice(CHANNELS)


def _make_complaint(rng: random.Random, brand: str, channel: str, families: set[str]) -> str:
    available_pools: list[str] = []
    for fam in families:
        available_pools.extend(COMPLAINT_FAMILIES.get(fam, []))
    if not available_pools:
        available_pools = COMPLAINT_FAMILIES["payout_drop"]
    return rng.choice(available_pools).format(brand=brand, channel=channel)


# ---- Episode composition ----

def compose_single(rng: random.Random) -> ScenarioInstance:
    prim = rng.choice(PRIMITIVES)
    brand, channel = _pick_brand_channel(rng)
    params = prim.param_fn(rng, brand, channel)
    complaint = _make_complaint(rng, brand, channel, {prim.complaint_family})
    return ScenarioInstance(
        primitives=[prim],
        primitive_params=[params],
        user_complaint=complaint,
    )


def compose_compound(rng: random.Random) -> ScenarioInstance:
    pair = rng.choice(COMPATIBLE_COMPOUNDS)
    prim_a = PRIMITIVE_BY_ID[pair[0]]
    prim_b = PRIMITIVE_BY_ID[pair[1]]
    brand, channel = _pick_brand_channel(rng)
    params_a = prim_a.param_fn(rng, brand, channel)
    params_b = prim_b.param_fn(rng, brand, channel)
    families = {prim_a.complaint_family, prim_b.complaint_family}
    complaint = _make_complaint(rng, brand, channel, families)
    return ScenarioInstance(
        primitives=[prim_a, prim_b],
        primitive_params=[params_a, params_b],
        user_complaint=complaint,
    )


def compose_with_drift(rng: random.Random) -> ScenarioInstance:
    instance = compose_single(rng)
    drift = rng.choice(DRIFT_EVENTS)
    instance.drift = drift
    return instance


def sample_scenario(rng: random.Random) -> ScenarioInstance:
    roll = rng.random()
    if roll < 0.50:
        return compose_single(rng)
    elif roll < 0.80:
        return compose_compound(rng)
    else:
        return compose_with_drift(rng)


def build_scenario_instance(template: Any, rng: random.Random) -> ScenarioInstance:
    """Backward-compat: build from an old-style template or just sample fresh."""
    return sample_scenario(rng)


# Re-export for backward compat
SCENARIO_TEMPLATES = PRIMITIVES[:3]
SCENARIO_BY_ID = {p.primitive_id: p for p in PRIMITIVES}
