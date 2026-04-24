"""Atomic anomaly primitives for procedural episode generation.

Each primitive is a small, composable unit that:
  1. Mutates one specific thing in the world state
  2. Declares which diagnostic tool reveals it
  3. Declares which terminal tool resolves it
  4. Provides a matcher that awards disambiguation bonus

Primitives are composed by the scenario engine into single- or multi-fault episodes.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
import random
from typing import Any, Callable

WorldState = dict[str, Any]


@dataclass(frozen=True)
class AnomalyPrimitive:
    primitive_id: str
    category: str  # "pipeline" | "business" | "financial"
    severity_range: tuple[float, float]  # (min_payout_drop, max_payout_drop)
    diagnostic_tool: str  # tool name that reveals this anomaly
    resolution_tool: str  # tool name that fixes / escalates it
    resolution_category: str  # "remediate" | "escalate_finance" | "escalate_ops"
    inject_fn: Callable[[WorldState, dict[str, Any], random.Random], None]
    param_fn: Callable[[random.Random, str, str], dict[str, Any]]
    matcher_fn: Callable[[dict[str, Any], dict[str, Any], dict[str, Any]], bool]
    complaint_family: str  # which complaint pool this maps to
    story_seed: str


# ---------------------------------------------------------------------------
# Helper: pair accessors
# ---------------------------------------------------------------------------

def _cfg(world: WorldState, brand: str, channel: str) -> dict[str, Any]:
    return world["channel_brand_config"][(brand, channel)]


def _cur(world: WorldState, brand: str, channel: str) -> dict[str, Any]:
    return world["current_reference"][(brand, channel)]


def _base(world: WorldState, brand: str, channel: str) -> dict[str, Any]:
    return world["baseline_reference"][(brand, channel)]


def _rebuild_orders(
    world: WorldState, brand: str, channel: str, current: dict[str, Any],
) -> None:
    orders = world["recent_orders"][(brand, channel)]
    if not orders:
        return
    n = max(1, current["total_orders"])
    for order in orders:
        order["gross_amount"] = round(current["gross_sales"] / n, 2)
        order["discount_amount"] = round(current["discount_amount"] / n, 2)
        order["net_sales"] = round(current["net_sales"] / n, 2)
        order["net_payout"] = round(current["net_payout"] / n, 2)


# ---------------------------------------------------------------------------
# 1. STUCK PROMO — promo flag left on after campaign ended
# ---------------------------------------------------------------------------

def _param_stuck_promo(rng: random.Random, brand: str, channel: str) -> dict[str, Any]:
    return {
        "brand": brand, "channel": channel,
        "promo_discount_pct": rng.randint(10, 25),
        "days_overdue": rng.randint(3, 14),
        "payout_drop_pct": round(rng.uniform(0.10, 0.18), 4),
    }


def _inject_stuck_promo(world: WorldState, params: dict[str, Any], rng: random.Random) -> None:
    b, c = params["brand"], params["channel"]
    drop = params["payout_drop_pct"]
    baseline, current, config = _base(world, b, c), _cur(world, b, c), _cfg(world, b, c)

    config["promo_active"] = True
    config["promo_discount_pct"] = float(params["promo_discount_pct"])
    config["promo_expected_to_end_at"] = world["now"] - timedelta(days=params["days_overdue"])
    config["promo_overdue_days"] = params["days_overdue"]

    current["total_orders"] = max(1, round(baseline["total_orders"] * (1 + rng.uniform(-0.06, 0.06))))
    current["gross_sales"] = round(baseline["gross_sales"] * (1 + rng.uniform(-0.05, 0.05)), 2)
    current["discount_amount"] = round(
        baseline["discount_amount"] + (current["gross_sales"] * config["promo_discount_pct"] / 100.0), 2,
    )
    current["net_sales"] = round(max(0.0, baseline["net_sales"] - (baseline["net_payout"] * drop * 0.9)), 2)
    current["net_payout"] = round(baseline["net_payout"] * (1.0 - drop), 2)
    current["aov"] = round(current["gross_sales"] / max(current["total_orders"], 1), 2)
    _rebuild_orders(world, b, c, current)


def _match_stuck_promo(action: dict[str, Any], result: dict[str, Any], params: dict[str, Any]) -> bool:
    return (
        action.get("tool") == "check_config"
        and action.get("args", {}).get("brand") == params["brand"]
        and action.get("args", {}).get("channel") == params["channel"]
        and bool(result.get("promo_active"))
        and float(result.get("promo_discount_pct", 0)) > 0
    )


# ---------------------------------------------------------------------------
# 2. STALE MENU SYNC — menu pipeline hasn't run, prices are stale
# ---------------------------------------------------------------------------

def _param_stale_menu(rng: random.Random, brand: str, channel: str) -> dict[str, Any]:
    return {
        "brand": brand, "channel": channel,
        "hours_stale": rng.randint(24, 96),
        "payout_drop_pct": round(rng.uniform(0.10, 0.18), 4),
    }


def _inject_stale_menu(world: WorldState, params: dict[str, Any], rng: random.Random) -> None:
    b, c = params["brand"], params["channel"]
    drop = params["payout_drop_pct"]
    baseline, current, config = _base(world, b, c), _cur(world, b, c), _cfg(world, b, c)

    config["promo_active"] = False
    config["promo_discount_pct"] = 0.0
    config["menu_last_synced_at"] = world["now"] - timedelta(hours=params["hours_stale"])
    world["pipeline_freshness"]["menu_sync"][(b, c)] = config["menu_last_synced_at"]

    current["total_orders"] = max(1, round(baseline["total_orders"] * (1 + rng.uniform(-0.08, 0.08))))
    current["gross_sales"] = round(baseline["gross_sales"] * (1 + rng.uniform(-0.05, 0.02)), 2)
    current["aov"] = round(baseline["aov"] * (1.0 - drop * 0.75), 2)
    current["net_sales"] = round(baseline["net_sales"] * (1.0 - drop * 0.7), 2)
    current["net_payout"] = round(baseline["net_payout"] * (1.0 - drop), 2)
    current["discount_amount"] = round(max(0.0, current["gross_sales"] - current["net_sales"]), 2)
    _rebuild_orders(world, b, c, current)


def _match_stale_menu(action: dict[str, Any], result: dict[str, Any], params: dict[str, Any]) -> bool:
    return (
        action.get("tool") == "check_pipeline_freshness"
        and action.get("args", {}).get("table_name") == "menu_sync"
        and action.get("args", {}).get("brand") == params["brand"]
        and action.get("args", {}).get("channel") == params["channel"]
        and bool(result.get("is_stale"))
    )


# ---------------------------------------------------------------------------
# 3. COMMISSION DRIFT — channel raised commission quietly
# ---------------------------------------------------------------------------

def _param_commission_drift(rng: random.Random, brand: str, channel: str) -> dict[str, Any]:
    return {
        "brand": brand, "channel": channel,
        "commission_delta": round(rng.uniform(2.0, 5.0), 2),
        "payout_drop_pct": round(rng.uniform(0.10, 0.18), 4),
    }


def _inject_commission_drift(world: WorldState, params: dict[str, Any], rng: random.Random) -> None:
    b, c = params["brand"], params["channel"]
    drop = params["payout_drop_pct"]
    baseline, current, config = _base(world, b, c), _cur(world, b, c), _cfg(world, b, c)

    config["promo_active"] = False
    config["promo_discount_pct"] = 0.0
    config["effective_commission_pct"] = round(config["baseline_commission_pct"] + params["commission_delta"], 2)

    current["total_orders"] = max(1, round(baseline["total_orders"] * (1 + rng.uniform(-0.05, 0.05))))
    current["gross_sales"] = round(baseline["gross_sales"] * (1 + rng.uniform(-0.04, 0.04)), 2)
    current["discount_amount"] = baseline["discount_amount"]
    current["net_sales"] = round(baseline["net_sales"] * (1 + rng.uniform(-0.04, 0.04)), 2)
    current["net_payout"] = round(baseline["net_payout"] * (1.0 - drop), 2)
    current["aov"] = round(current["gross_sales"] / max(current["total_orders"], 1), 2)
    current["effective_commission_pct"] = config["effective_commission_pct"]
    _rebuild_orders(world, b, c, current)


def _match_commission_drift(action: dict[str, Any], result: dict[str, Any], params: dict[str, Any]) -> bool:
    return (
        action.get("tool") == "check_config"
        and action.get("args", {}).get("brand") == params["brand"]
        and action.get("args", {}).get("channel") == params["channel"]
        and float(result.get("effective_commission_pct", 0)) > float(result.get("baseline_commission_pct", 0))
    )


# ---------------------------------------------------------------------------
# 4. ORDER INGESTION LAG — ingested_at delayed by hours, metrics look stale
# ---------------------------------------------------------------------------

def _param_ingestion_lag(rng: random.Random, brand: str, channel: str) -> dict[str, Any]:
    return {
        "brand": brand, "channel": channel,
        "lag_hours": rng.randint(4, 18),
        "payout_drop_pct": round(rng.uniform(0.08, 0.15), 4),
    }


def _inject_ingestion_lag(world: WorldState, params: dict[str, Any], rng: random.Random) -> None:
    b, c = params["brand"], params["channel"]
    drop = params["payout_drop_pct"]
    baseline, current = _base(world, b, c), _cur(world, b, c)

    world["pipeline_freshness"]["orders"] = world["now"] - timedelta(hours=params["lag_hours"])

    current["total_orders"] = max(1, round(baseline["total_orders"] * (1.0 - drop * 0.8)))
    current["gross_sales"] = round(baseline["gross_sales"] * (1.0 - drop * 0.8), 2)
    current["net_sales"] = round(baseline["net_sales"] * (1.0 - drop * 0.8), 2)
    current["net_payout"] = round(baseline["net_payout"] * (1.0 - drop), 2)
    current["aov"] = round(current["gross_sales"] / max(current["total_orders"], 1), 2)
    current["discount_amount"] = round(max(0.0, current["gross_sales"] - current["net_sales"]), 2)

    for order in world["recent_orders"][(b, c)][:max(1, params["lag_hours"])]:
        order["ingestion_lag_hours"] = params["lag_hours"]
    _rebuild_orders(world, b, c, current)


def _match_ingestion_lag(action: dict[str, Any], result: dict[str, Any], params: dict[str, Any]) -> bool:
    return (
        action.get("tool") == "check_pipeline_freshness"
        and action.get("args", {}).get("table_name") == "orders"
        and bool(result.get("is_stale"))
    )


# ---------------------------------------------------------------------------
# 5. DUPLICATE ORDERS — orders double-counted, revenue inflated
# ---------------------------------------------------------------------------

def _param_duplicate_orders(rng: random.Random, brand: str, channel: str) -> dict[str, Any]:
    return {
        "brand": brand, "channel": channel,
        "dup_fraction": round(rng.uniform(0.15, 0.35), 2),
    }


def _inject_duplicate_orders(world: WorldState, params: dict[str, Any], rng: random.Random) -> None:
    b, c = params["brand"], params["channel"]
    frac = params["dup_fraction"]
    baseline, current = _base(world, b, c), _cur(world, b, c)

    extra_orders = max(1, round(baseline["total_orders"] * frac))
    current["total_orders"] = baseline["total_orders"] + extra_orders
    current["gross_sales"] = round(baseline["gross_sales"] * (1 + frac), 2)
    current["discount_amount"] = round(baseline["discount_amount"] * (1 + frac), 2)
    current["net_sales"] = round(baseline["net_sales"] * (1 + frac), 2)
    current["net_payout"] = round(baseline["net_payout"] * (1 + frac), 2)
    current["aov"] = baseline["aov"]

    orders = world["recent_orders"][(b, c)]
    n_dup = min(len(orders), max(1, round(len(orders) * frac)))
    for i in range(n_dup):
        dup = dict(orders[i])
        dup["order_id"] = dup["order_id"] + "-DUP"
        dup["is_duplicate"] = True
        orders.append(dup)
    _rebuild_orders(world, b, c, current)


def _match_duplicate_orders(action: dict[str, Any], result: dict[str, Any], params: dict[str, Any]) -> bool:
    if action.get("tool") != "inspect_recent_orders":
        return False
    if action.get("args", {}).get("brand") != params["brand"]:
        return False
    if action.get("args", {}).get("channel") != params["channel"]:
        return False
    rows = result.get("rows", [])
    return any(r.get("is_duplicate") or str(r.get("order_id", "")).endswith("-DUP") for r in rows)


# ---------------------------------------------------------------------------
# 6. MISSING FEED — all recent orders for one channel vanished
# ---------------------------------------------------------------------------

def _param_missing_feed(rng: random.Random, brand: str, channel: str) -> dict[str, Any]:
    return {"brand": brand, "channel": channel}


def _inject_missing_feed(world: WorldState, params: dict[str, Any], rng: random.Random) -> None:
    b, c = params["brand"], params["channel"]
    current = _cur(world, b, c)

    current["total_orders"] = 0
    current["gross_sales"] = 0.0
    current["discount_amount"] = 0.0
    current["net_sales"] = 0.0
    current["net_payout"] = 0.0
    current["aov"] = 0.0
    world["recent_orders"][(b, c)] = []


def _match_missing_feed(action: dict[str, Any], result: dict[str, Any], params: dict[str, Any]) -> bool:
    if action.get("tool") != "inspect_recent_orders":
        return False
    if action.get("args", {}).get("brand") != params["brand"]:
        return False
    if action.get("args", {}).get("channel") != params["channel"]:
        return False
    return len(result.get("rows", [])) == 0


# ---------------------------------------------------------------------------
# 7. BRAND PAUSED — brand goes offline on one channel
# ---------------------------------------------------------------------------

def _param_brand_paused(rng: random.Random, brand: str, channel: str) -> dict[str, Any]:
    return {"brand": brand, "channel": channel}


def _inject_brand_paused(world: WorldState, params: dict[str, Any], rng: random.Random) -> None:
    b, c = params["brand"], params["channel"]
    config = _cfg(world, b, c)
    current = _cur(world, b, c)

    config["is_live"] = False
    current["total_orders"] = 0
    current["gross_sales"] = 0.0
    current["discount_amount"] = 0.0
    current["net_sales"] = 0.0
    current["net_payout"] = 0.0
    current["aov"] = 0.0


def _match_brand_paused(action: dict[str, Any], result: dict[str, Any], params: dict[str, Any]) -> bool:
    return (
        action.get("tool") == "check_config"
        and action.get("args", {}).get("brand") == params["brand"]
        and action.get("args", {}).get("channel") == params["channel"]
        and result.get("is_live") is False
    )


# ---------------------------------------------------------------------------
# 8. DISCOUNT SPIKE — unexpected discounts without a promo campaign
# ---------------------------------------------------------------------------

def _param_discount_spike(rng: random.Random, brand: str, channel: str) -> dict[str, Any]:
    return {
        "brand": brand, "channel": channel,
        "spike_pct": round(rng.uniform(0.12, 0.30), 2),
        "payout_drop_pct": round(rng.uniform(0.08, 0.15), 4),
    }


def _inject_discount_spike(world: WorldState, params: dict[str, Any], rng: random.Random) -> None:
    b, c = params["brand"], params["channel"]
    drop = params["payout_drop_pct"]
    baseline, current, config = _base(world, b, c), _cur(world, b, c), _cfg(world, b, c)

    config["promo_active"] = False
    config["promo_discount_pct"] = 0.0

    current["total_orders"] = max(1, round(baseline["total_orders"] * (1 + rng.uniform(-0.05, 0.05))))
    current["gross_sales"] = round(baseline["gross_sales"] * (1 + rng.uniform(-0.03, 0.03)), 2)
    current["discount_amount"] = round(current["gross_sales"] * params["spike_pct"], 2)
    current["net_sales"] = round(current["gross_sales"] - current["discount_amount"], 2)
    current["net_payout"] = round(baseline["net_payout"] * (1.0 - drop), 2)
    current["aov"] = round(current["gross_sales"] / max(current["total_orders"], 1), 2)
    _rebuild_orders(world, b, c, current)


def _match_discount_spike(action: dict[str, Any], result: dict[str, Any], params: dict[str, Any]) -> bool:
    if action.get("tool") != "inspect_recent_orders":
        return False
    if action.get("args", {}).get("brand") != params["brand"]:
        return False
    if action.get("args", {}).get("channel") != params["channel"]:
        return False
    rows = result.get("rows", [])
    if not rows:
        return False
    avg_discount = sum(float(r.get("discount_amount", 0)) for r in rows) / len(rows)
    avg_gross = sum(float(r.get("gross_amount", 1)) for r in rows) / len(rows)
    return avg_gross > 0 and (avg_discount / avg_gross) > 0.10


# ---------------------------------------------------------------------------
# 9. REFUND SPIKE — abnormal refund rate
# ---------------------------------------------------------------------------

def _param_refund_spike(rng: random.Random, brand: str, channel: str) -> dict[str, Any]:
    return {
        "brand": brand, "channel": channel,
        "refund_rate": round(rng.uniform(0.12, 0.30), 2),
        "payout_drop_pct": round(rng.uniform(0.10, 0.20), 4),
    }


def _inject_refund_spike(world: WorldState, params: dict[str, Any], rng: random.Random) -> None:
    b, c = params["brand"], params["channel"]
    drop = params["payout_drop_pct"]
    baseline, current = _base(world, b, c), _cur(world, b, c)

    current["total_orders"] = baseline["total_orders"]
    current["gross_sales"] = baseline["gross_sales"]
    current["discount_amount"] = baseline["discount_amount"]
    refund = round(current["gross_sales"] * params["refund_rate"], 2)
    current["refund_amount"] = refund
    current["net_sales"] = round(max(0, current["gross_sales"] - current["discount_amount"] - refund), 2)
    current["net_payout"] = round(baseline["net_payout"] * (1.0 - drop), 2)
    current["aov"] = baseline["aov"]

    for order in world["recent_orders"][(b, c)]:
        if rng.random() < params["refund_rate"]:
            order["status"] = "refunded"
            order["refund_amount"] = order.get("gross_amount", 0)
    _rebuild_orders(world, b, c, current)


def _match_refund_spike(action: dict[str, Any], result: dict[str, Any], params: dict[str, Any]) -> bool:
    if action.get("tool") != "inspect_recent_orders":
        return False
    if action.get("args", {}).get("brand") != params["brand"]:
        return False
    if action.get("args", {}).get("channel") != params["channel"]:
        return False
    rows = result.get("rows", [])
    if not rows:
        return False
    refund_count = sum(1 for r in rows if r.get("status") == "refunded")
    return refund_count / len(rows) > 0.10


# ---------------------------------------------------------------------------
# 10. AOV COLLAPSE — cheap items dominate, average order value drops
# ---------------------------------------------------------------------------

def _param_aov_collapse(rng: random.Random, brand: str, channel: str) -> dict[str, Any]:
    return {
        "brand": brand, "channel": channel,
        "aov_drop_factor": round(rng.uniform(0.35, 0.55), 2),
    }


def _inject_aov_collapse(world: WorldState, params: dict[str, Any], rng: random.Random) -> None:
    b, c = params["brand"], params["channel"]
    factor = params["aov_drop_factor"]
    baseline, current = _base(world, b, c), _cur(world, b, c)

    current["total_orders"] = max(1, round(baseline["total_orders"] * (1 + rng.uniform(0.10, 0.30))))
    current["aov"] = round(baseline["aov"] * (1.0 - factor), 2)
    current["gross_sales"] = round(current["total_orders"] * current["aov"], 2)
    current["discount_amount"] = round(baseline["discount_amount"] * (1 + rng.uniform(-0.05, 0.05)), 2)
    current["net_sales"] = round(current["gross_sales"] - current["discount_amount"], 2)
    commission_pct = _cfg(world, b, c)["effective_commission_pct"] / 100.0
    current["net_payout"] = round(current["net_sales"] * (1.0 - commission_pct), 2)
    _rebuild_orders(world, b, c, current)


def _match_aov_collapse(action: dict[str, Any], result: dict[str, Any], params: dict[str, Any]) -> bool:
    if action.get("tool") == "get_kpi_summary":
        summary = result.get("summary", result)
        baseline_aov = params.get("_baseline_aov", 300)
        current_aov = float(summary.get("aov", 0))
        return current_aov < baseline_aov * 0.7
    if action.get("tool") == "inspect_recent_orders":
        if action.get("args", {}).get("brand") != params["brand"]:
            return False
        if action.get("args", {}).get("channel") != params["channel"]:
            return False
        rows = result.get("rows", [])
        if not rows:
            return False
        avg_gross = sum(float(r.get("gross_amount", 0)) for r in rows) / len(rows)
        return avg_gross < params.get("_baseline_aov", 300) * 0.7
    return False


# ---------------------------------------------------------------------------
# 11. PARTIAL AGGREGATION — metrics refreshed for some brands, not others
# ---------------------------------------------------------------------------

def _param_partial_agg(rng: random.Random, brand: str, channel: str) -> dict[str, Any]:
    return {
        "brand": brand, "channel": channel,
        "stale_hours": rng.randint(26, 72),
    }


def _inject_partial_agg(world: WorldState, params: dict[str, Any], rng: random.Random) -> None:
    b, c = params["brand"], params["channel"]
    baseline, current = _base(world, b, c), _cur(world, b, c)

    world["pipeline_freshness"]["channel_daily_metrics"] = world["now"] - timedelta(hours=params["stale_hours"])

    scale = round(rng.uniform(0.70, 0.85), 2)
    current["total_orders"] = max(1, round(baseline["total_orders"] * scale))
    current["gross_sales"] = round(baseline["gross_sales"] * scale, 2)
    current["net_sales"] = round(baseline["net_sales"] * scale, 2)
    current["net_payout"] = round(baseline["net_payout"] * scale, 2)
    current["discount_amount"] = round(baseline["discount_amount"] * scale, 2)
    current["aov"] = baseline["aov"]
    _rebuild_orders(world, b, c, current)


def _match_partial_agg(action: dict[str, Any], result: dict[str, Any], params: dict[str, Any]) -> bool:
    return (
        action.get("tool") == "check_pipeline_freshness"
        and action.get("args", {}).get("table_name") == "channel_daily_metrics"
        and bool(result.get("is_stale"))
    )


# ---------------------------------------------------------------------------
# 12. FALSE ALARM — metrics dipped due to weekend / seasonality, nothing broken
# ---------------------------------------------------------------------------

def _param_false_alarm(rng: random.Random, brand: str, channel: str) -> dict[str, Any]:
    return {
        "brand": brand, "channel": channel,
        "dip_pct": round(rng.uniform(0.05, 0.12), 4),
    }


def _inject_false_alarm(world: WorldState, params: dict[str, Any], rng: random.Random) -> None:
    b, c = params["brand"], params["channel"]
    dip = params["dip_pct"]
    baseline, current = _base(world, b, c), _cur(world, b, c)

    current["total_orders"] = max(1, round(baseline["total_orders"] * (1.0 - dip)))
    current["gross_sales"] = round(baseline["gross_sales"] * (1.0 - dip), 2)
    current["discount_amount"] = round(baseline["discount_amount"] * (1.0 - dip), 2)
    current["net_sales"] = round(baseline["net_sales"] * (1.0 - dip), 2)
    current["net_payout"] = round(baseline["net_payout"] * (1.0 - dip), 2)
    current["aov"] = baseline["aov"]
    _rebuild_orders(world, b, c, current)


def _match_false_alarm(action: dict[str, Any], result: dict[str, Any], params: dict[str, Any]) -> bool:
    if action.get("tool") == "check_config":
        if action.get("args", {}).get("brand") != params["brand"]:
            return False
        if action.get("args", {}).get("channel") != params["channel"]:
            return False
        return (
            result.get("is_live") is True
            and not result.get("promo_active")
            and float(result.get("effective_commission_pct", 0)) <= float(result.get("baseline_commission_pct", 99))
        )
    return False


# ---------------------------------------------------------------------------
# PRIMITIVE REGISTRY
# ---------------------------------------------------------------------------

PRIMITIVES: list[AnomalyPrimitive] = [
    AnomalyPrimitive(
        primitive_id="stuck_promo",
        category="business",
        severity_range=(0.10, 0.18),
        diagnostic_tool="check_config",
        resolution_tool="escalate_to_finance",
        resolution_category="escalate_finance",
        inject_fn=_inject_stuck_promo,
        param_fn=_param_stuck_promo,
        matcher_fn=_match_stuck_promo,
        complaint_family="payout_drop",
        story_seed="Promo flag stayed on after the campaign ended.",
    ),
    AnomalyPrimitive(
        primitive_id="stale_menu_sync",
        category="pipeline",
        severity_range=(0.10, 0.18),
        diagnostic_tool="check_pipeline_freshness",
        resolution_tool="force_menu_sync",
        resolution_category="remediate",
        inject_fn=_inject_stale_menu,
        param_fn=_param_stale_menu,
        matcher_fn=_match_stale_menu,
        complaint_family="payout_drop",
        story_seed="Menu sync fell behind and pricing drifted.",
    ),
    AnomalyPrimitive(
        primitive_id="commission_drift",
        category="business",
        severity_range=(0.10, 0.18),
        diagnostic_tool="check_config",
        resolution_tool="escalate_to_finance",
        resolution_category="escalate_finance",
        inject_fn=_inject_commission_drift,
        param_fn=_param_commission_drift,
        matcher_fn=_match_commission_drift,
        complaint_family="payout_drop",
        story_seed="Channel commission changed quietly and margin shrank.",
    ),
    AnomalyPrimitive(
        primitive_id="ingestion_lag",
        category="pipeline",
        severity_range=(0.08, 0.15),
        diagnostic_tool="check_pipeline_freshness",
        resolution_tool="restart_pipeline",
        resolution_category="remediate",
        inject_fn=_inject_ingestion_lag,
        param_fn=_param_ingestion_lag,
        matcher_fn=_match_ingestion_lag,
        complaint_family="numbers_off",
        story_seed="Order ingestion pipeline running hours behind.",
    ),
    AnomalyPrimitive(
        primitive_id="duplicate_orders",
        category="pipeline",
        severity_range=(0.15, 0.35),
        diagnostic_tool="inspect_recent_orders",
        resolution_tool="restart_pipeline",
        resolution_category="remediate",
        inject_fn=_inject_duplicate_orders,
        param_fn=_param_duplicate_orders,
        matcher_fn=_match_duplicate_orders,
        complaint_family="numbers_off",
        story_seed="Duplicate events inflated order counts.",
    ),
    AnomalyPrimitive(
        primitive_id="missing_feed",
        category="pipeline",
        severity_range=(0.90, 1.00),
        diagnostic_tool="inspect_recent_orders",
        resolution_tool="restart_pipeline",
        resolution_category="remediate",
        inject_fn=_inject_missing_feed,
        param_fn=_param_missing_feed,
        matcher_fn=_match_missing_feed,
        complaint_family="numbers_off",
        story_seed="Feed from channel stopped entirely.",
    ),
    AnomalyPrimitive(
        primitive_id="brand_paused",
        category="business",
        severity_range=(0.90, 1.00),
        diagnostic_tool="check_config",
        resolution_tool="escalate_to_ops",
        resolution_category="escalate_ops",
        inject_fn=_inject_brand_paused,
        param_fn=_param_brand_paused,
        matcher_fn=_match_brand_paused,
        complaint_family="brand_gone",
        story_seed="Brand was taken offline on this channel.",
    ),
    AnomalyPrimitive(
        primitive_id="discount_spike",
        category="business",
        severity_range=(0.08, 0.15),
        diagnostic_tool="inspect_recent_orders",
        resolution_tool="escalate_to_finance",
        resolution_category="escalate_finance",
        inject_fn=_inject_discount_spike,
        param_fn=_param_discount_spike,
        matcher_fn=_match_discount_spike,
        complaint_family="payout_drop",
        story_seed="Unexpected discounts eating into margins.",
    ),
    AnomalyPrimitive(
        primitive_id="refund_spike",
        category="financial",
        severity_range=(0.10, 0.20),
        diagnostic_tool="inspect_recent_orders",
        resolution_tool="escalate_to_ops",
        resolution_category="escalate_ops",
        inject_fn=_inject_refund_spike,
        param_fn=_param_refund_spike,
        matcher_fn=_match_refund_spike,
        complaint_family="payout_drop",
        story_seed="Abnormal refund rate eroding net payout.",
    ),
    AnomalyPrimitive(
        primitive_id="aov_collapse",
        category="business",
        severity_range=(0.35, 0.55),
        diagnostic_tool="inspect_recent_orders",
        resolution_tool="escalate_to_ops",
        resolution_category="escalate_ops",
        inject_fn=_inject_aov_collapse,
        param_fn=_param_aov_collapse,
        matcher_fn=_match_aov_collapse,
        complaint_family="revenue_flat",
        story_seed="Average order value collapsed — cheap items dominating.",
    ),
    AnomalyPrimitive(
        primitive_id="partial_aggregation",
        category="pipeline",
        severity_range=(0.15, 0.30),
        diagnostic_tool="check_pipeline_freshness",
        resolution_tool="restart_pipeline",
        resolution_category="remediate",
        inject_fn=_inject_partial_agg,
        param_fn=_param_partial_agg,
        matcher_fn=_match_partial_agg,
        complaint_family="numbers_off",
        story_seed="Aggregation pipeline only partially refreshed.",
    ),
    AnomalyPrimitive(
        primitive_id="false_alarm",
        category="none",
        severity_range=(0.05, 0.12),
        diagnostic_tool="check_config",
        resolution_tool="escalate_to_ops",
        resolution_category="escalate_ops",
        inject_fn=_inject_false_alarm,
        param_fn=_param_false_alarm,
        matcher_fn=_match_false_alarm,
        complaint_family="payout_drop",
        story_seed="Normal seasonal dip — no action needed.",
    ),
]

PRIMITIVE_BY_ID: dict[str, AnomalyPrimitive] = {p.primitive_id: p for p in PRIMITIVES}

COMPATIBLE_COMPOUNDS: list[tuple[str, str]] = [
    ("stuck_promo", "stale_menu_sync"),
    ("stuck_promo", "commission_drift"),
    ("stale_menu_sync", "commission_drift"),
    ("commission_drift", "ingestion_lag"),
    ("stuck_promo", "ingestion_lag"),
    ("duplicate_orders", "commission_drift"),
    ("stale_menu_sync", "discount_spike"),
    ("ingestion_lag", "refund_spike"),
    ("aov_collapse", "commission_drift"),
    ("partial_aggregation", "stuck_promo"),
    ("partial_aggregation", "commission_drift"),
]
