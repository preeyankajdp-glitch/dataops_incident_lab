"""Food brand x channel incident lab with a guided Gradio UI."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
import json
import os
from pathlib import Path
import random
import re
from typing import Any

import duckdb
import gradio as gr
from openai import OpenAI


APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data" / "foodops_lab"
DB_PATH = DATA_DIR / "foodops.duckdb"
LOG_PATH = DATA_DIR / "logs" / "pipeline_runs.json"
DEFAULT_SEED = 20260424
DAYS_OF_DATA = 14
SERVER_PORT = 7863
HF_BASE_URL = os.getenv("API_BASE_URL") or os.getenv("HF_API_BASE_URL") or "https://router.huggingface.co/v1"
HF_MODEL = os.getenv("HF_CHAT_MODEL") or os.getenv("HF_MODEL") or os.getenv("OPENAI_MODEL") or "Qwen/Qwen2.5-7B-Instruct"


BRANDS = [
    {"brand_id": "behrouz", "brand_name": "Behrouz Biryani", "cuisine_type": "Biryani"},
    {"brand_id": "ovenstory", "brand_name": "Oven Story Pizza", "cuisine_type": "Pizza"},
    {"brand_id": "faasos", "brand_name": "Faasos", "cuisine_type": "Wraps"},
    {"brand_id": "sweettruth", "brand_name": "Sweet Truth", "cuisine_type": "Desserts"},
    {"brand_id": "thebowlcompany", "brand_name": "The Bowl Company", "cuisine_type": "Rice Bowls"},
]

CHANNELS = [
    {"channel_id": "zomato", "channel_name": "Zomato", "channel_type": "Aggregator", "default_commission_pct": 22.0, "feed_delay_minutes": 20},
    {"channel_id": "swiggy", "channel_name": "Swiggy", "channel_type": "Aggregator", "default_commission_pct": 24.0, "feed_delay_minutes": 35},
    {"channel_id": "ondc", "channel_name": "ONDC", "channel_type": "Marketplace", "default_commission_pct": 12.0, "feed_delay_minutes": 55},
    {"channel_id": "ownapp", "channel_name": "Own App", "channel_type": "Direct", "default_commission_pct": 5.0, "feed_delay_minutes": 5},
]

CONFIG_ROWS = [
    ("behrouz", "zomato", True, False, 0.0, 22.0),
    ("behrouz", "swiggy", True, False, 0.0, 24.0),
    ("behrouz", "ownapp", True, False, 0.0, 5.0),
    ("ovenstory", "zomato", True, False, 0.0, 21.0),
    ("ovenstory", "swiggy", True, False, 0.0, 24.0),
    ("ovenstory", "ondc", True, False, 0.0, 13.0),
    ("faasos", "zomato", True, False, 0.0, 22.0),
    ("faasos", "swiggy", True, False, 0.0, 23.0),
    ("faasos", "ownapp", True, False, 0.0, 5.0),
    ("sweettruth", "zomato", True, False, 0.0, 23.0),
    ("sweettruth", "swiggy", True, False, 0.0, 24.0),
    ("thebowlcompany", "swiggy", True, False, 0.0, 24.0),
    ("thebowlcompany", "ondc", True, False, 0.0, 12.5),
    ("thebowlcompany", "ownapp", True, False, 0.0, 5.0),
]

BRAND_BASES = {
    "behrouz": {"aov": 520, "orders": 9},
    "ovenstory": {"aov": 480, "orders": 8},
    "faasos": {"aov": 310, "orders": 10},
    "sweettruth": {"aov": 260, "orders": 7},
    "thebowlcompany": {"aov": 340, "orders": 8},
}

CHANNEL_MULTIPLIERS = {
    "zomato": {"orders": 1.0, "aov": 1.04},
    "swiggy": {"orders": 0.95, "aov": 1.0},
    "ondc": {"orders": 0.45, "aov": 0.94},
    "ownapp": {"orders": 0.52, "aov": 1.09},
}

KPI_DEFINITIONS = {
    "total_orders": {
        "label": "Total Orders",
        "meaning": "How many orders the brand-channel network created.",
        "look_next": ["delivered_orders", "cancelled_orders", "gross_sales"],
    },
    "gross_sales": {
        "label": "Gross Sales",
        "meaning": "Topline order value before discounts, refunds, and commissions.",
        "look_next": ["discount_amount", "net_sales", "commission_amount"],
    },
    "discount_amount": {
        "label": "Discount Amount",
        "meaning": "How much demand was bought through promotions or markdowns.",
        "look_next": ["promo_active", "net_sales", "aov"],
    },
    "net_sales": {
        "label": "Net Sales",
        "meaning": "Delivered revenue after discounts and refunds. This is what the business really realized.",
        "look_next": ["gross_sales", "refund_amount", "cancellation_rate"],
    },
    "net_payout": {
        "label": "Net Payout",
        "meaning": "Net sales after channel commissions. This is the revenue pool that actually pays rent.",
        "look_next": ["commission_amount", "effective_commission_pct", "channel_name"],
    },
    "aov": {
        "label": "AOV",
        "meaning": "Average realized revenue per delivered order. It reflects basket quality and discount intensity.",
        "look_next": ["total_orders", "discount_amount", "gross_sales"],
    },
    "brand_channel_coverage": {
        "label": "Brand x Channel Coverage",
        "meaning": "Share of brand-channel combinations currently live.",
        "look_next": ["is_live", "channel_brand_config", "orders_by_channel"],
    },
}


# KPI_RELATIONS is the single source of truth for KPI algebra.
# Each entry states: the formula, the components that feed it, and the
# typical decomposition an analyst (or agent) should walk through when
# the KPI behaves oddly. This is used by:
#   1. the assistant/agent as a "get_kpi_relations" tool,
#   2. the dashboard to show "why this KPI moved" context,
#   3. the reward function to score whether the agent's reasoning
#      chain cites a valid decomposition.
KPI_RELATIONS = {
    "total_orders": {
        "formula": "count(orders) where status != 'cancelled'",
        "components": ["orders.status", "orders.brand_id", "orders.channel_id"],
        "decomposition": (
            "If orders dropped on one channel but not others, suspect channel-level "
            "pause (is_live=false), feed delay, or ingestion issue. If drop is global, "
            "suspect raw ingest or aggregation staleness."
        ),
    },
    "gross_sales": {
        "formula": "sum(gross_amount)",
        "components": ["total_orders", "aov"],
        "decomposition": (
            "gross_sales ~ total_orders * aov. If gross flat while orders up, AOV fell. "
            "If gross up while orders flat, AOV rose (price change or mix shift)."
        ),
    },
    "discount_amount": {
        "formula": "sum(discount_amount) ~ sum(gross * promo_discount_pct) when promo_active",
        "components": ["promo_active", "promo_discount_pct", "gross_sales"],
        "decomposition": (
            "Discount up without a live promo campaign in ops comms -> config drift "
            "(promo_active stuck true). Discount up with active promo -> expected."
        ),
    },
    "net_sales": {
        "formula": "sum(delivered_amount) - sum(refund_amount)",
        "components": ["gross_sales", "discount_amount", "refund_amount", "cancellation_rate"],
        "decomposition": (
            "net_sales ~ gross_sales - discount_amount - refund_amount (on delivered). "
            "If net fell but gross flat, check discount rise or refund rise. "
            "If net fell proportional to orders, it is a volume story not a quality story."
        ),
    },
    "net_payout": {
        "formula": "net_sales * (1 - effective_commission_pct / 100)",
        "components": ["net_sales", "effective_commission_pct"],
        "decomposition": (
            "net_payout is the money that actually pays rent. "
            "If payout flat while gross/net up -> commission moved (renegotiation or drift). "
            "If payout down proportional to net -> volume or quality problem, not commission."
        ),
    },
    "aov": {
        "formula": "gross_sales / total_orders",
        "components": ["gross_sales", "total_orders"],
        "decomposition": (
            "Orders up but gross flat -> AOV collapse (low-price SKU mix, heavy discount, "
            "cheap channel skew). AOV up suddenly for one brand-channel -> menu sync stale "
            "or price update not propagated."
        ),
    },
    "brand_channel_coverage": {
        "formula": "count(is_live=true) / count(*) over channel_brand_config",
        "components": ["channel_brand_config.is_live"],
        "decomposition": (
            "Coverage drop means one or more brand-channel listings went dark. "
            "Check channel_brand_config for is_live=false and recent notes."
        ),
    },
}

SCENARIOS = {
    "Baseline": {
        "kind": "baseline",
        "headline": "Healthy baseline",
        "summary": "All live brands are synced, no campaign is stuck, and the network behaves normally.",
        "apply": [],
    },
    "Stuck Promo After Campaign End": {
        "kind": "business_config",
        "headline": "Behrouz on Swiggy is still discounting after the promo ended",
        "summary": "Discounts stay elevated for one brand-channel pair even though the campaign should have ended.",
        "apply": ["stuck_promo"],
    },
    "Brand Paused On One Channel": {
        "kind": "business_config",
        "headline": "Oven Story disappeared from Zomato yesterday",
        "summary": "Orders collapse only on one channel because the listing was toggled off.",
        "apply": ["pause_brand_channel"],
    },
    "Commission Drift Hiding Margin Pressure": {
        "kind": "business_config",
        "headline": "Gross is up but payout is flat",
        "summary": "Everything in the pipeline is correct; the channel commission changed and margins shrank.",
        "apply": ["commission_drift"],
    },
    "Menu Sync Staleness": {
        "kind": "business_config",
        "headline": "Faasos Swiggy AOV jumped overnight",
        "summary": "Menu sync is stale, so pricing on one aggregator is out of date.",
        "apply": ["menu_sync_stale"],
    },
    "Orders Up, Net Sales Flat (AOV Collapse)": {
        "kind": "business_config",
        "headline": "Order volume rose but net sales did not",
        "summary": "No pipeline bug. A cheap-SKU mix shift pulled AOV down, so more orders did not translate to more revenue.",
        "apply": ["aov_collapse"],
    },
    "Discount Spike Without Promo Campaign": {
        "kind": "business_config",
        "headline": "Discount amount is up, but ops says no campaign is live",
        "summary": "No pipeline failure. The discount rise reflects config drift on one brand-channel rather than a real campaign.",
        "apply": ["stealth_discount"],
    },
    "Stale Aggregate After Swiggy Feed": {
        "kind": "pipeline_data",
        "headline": "Raw Swiggy orders landed, but the dashboard never caught up",
        "summary": "Fresh raw orders were inserted for Behrouz × Swiggy, but aggregation did not rerun, so the dashboard is stale.",
        "apply": ["stale_aggregate_after_swiggy_feed"],
    },
    "Partial Aggregate Refresh (Zomato Missing)": {
        "kind": "pipeline_data",
        "headline": "Zomato vanished after the refresh",
        "summary": "The refresh completed only partially and dropped Zomato partitions from the aggregate table.",
        "apply": ["partial_refresh_drop_zomato"],
    },
    "Duplicate ONDC Events": {
        "kind": "pipeline_data",
        "headline": "ONDC counts look almost double",
        "summary": "The connector replayed ONDC events without deduplication, inflating orders and revenue in raw and aggregate.",
        "apply": ["duplicate_ondc_events"],
    },
    "Late Swiggy Feed Skew": {
        "kind": "pipeline_data",
        "headline": "Swiggy looks low in the morning and catches up later",
        "summary": "The Swiggy feed arrived after the dashboard build, so the morning aggregate is incomplete.",
        "apply": ["late_swiggy_feed_skew"],
    },
    "Cancelled Orders Missing From Aggregate": {
        "kind": "pipeline_data",
        "headline": "Cancellation rate looks impossibly clean",
        "summary": "The aggregate transform dropped cancelled orders for one slice, making the dashboard look healthier than reality.",
        "apply": ["cancelled_orders_dropped_from_aggregate"],
    },
    "Mixed: Promo + Stale Aggregate": {
        "kind": "mixed",
        "headline": "Behrouz on Swiggy has a real promo problem and a stale dashboard",
        "summary": "The promo flag is stuck on, raw orders changed, and the aggregate never refreshed. Both causes are true.",
        "apply": ["mixed_promo_stale_aggregate"],
    },
    "Mixed: Listing Pause + Stale Aggregate": {
        "kind": "mixed",
        "headline": "Oven Story was paused, but the dashboard still shows sales",
        "summary": "The listing really was paused, but the aggregate is stale, so the dashboard keeps showing old Zomato revenue.",
        "apply": ["mixed_pause_stale_aggregate"],
    },
    "Compound: Promo + Menu Sync": {
        "kind": "mixed",
        "headline": "Behrouz on Zomato is bleeding margin for two reasons",
        "summary": "Promo stayed on, menu sync is old, and the resulting revenue story is messy.",
        "apply": ["compound_promo_sync"],
    },
}

SCENARIO_QUESTIONS = {
    "Baseline": "Give me a plain-English view of network health for yesterday.",
    "Stuck Promo After Campaign End": "Why is discount amount still so high for Behrouz on Swiggy?",
    "Brand Paused On One Channel": "Oven Story orders crashed on Zomato but Swiggy looks fine. What happened?",
    "Commission Drift Hiding Margin Pressure": "Gross revenue is up, so why is payout flat?",
    "Menu Sync Staleness": "Why did Faasos Swiggy AOV jump so sharply overnight?",
    "Orders Up, Net Sales Flat (AOV Collapse)": "We shipped more orders than last week but net sales didn't move. What's eating it?",
    "Discount Spike Without Promo Campaign": "Discount amount is higher than expected even though ops says no campaign is live. Is the dashboard wrong?",
    "Stale Aggregate After Swiggy Feed": "Swiggy raw orders landed, but the dashboard still looks stale. What broke?",
    "Partial Aggregate Refresh (Zomato Missing)": "Zomato vanished from today's dashboard after the refresh. Is this real demand loss or a pipeline problem?",
    "Duplicate ONDC Events": "Why does ONDC look nearly double what ops expected?",
    "Late Swiggy Feed Skew": "Every morning Swiggy looks too low and then catches up later. What's happening?",
    "Cancelled Orders Missing From Aggregate": "Cancellation rate looks impossibly clean on Faasos x Swiggy. Did we really have zero cancels?",
    "Mixed: Promo + Stale Aggregate": "Behrouz on Swiggy still looks off after the promo ended. Is it just the campaign, or is the dashboard stale too?",
    "Mixed: Listing Pause + Stale Aggregate": "Oven Story was paused on Zomato, but the dashboard still shows sales there. What's going on?",
    "Compound: Promo + Menu Sync": "Is Behrouz on Zomato losing money because of promo leakage, sync issues, or both?",
}


# Per-scenario storybook content. The Scenarios tab renders this as three acts:
#   Act 1 — what the ops user was looking at before
#   Act 2 — what got injected into the data world
#   Act 3 — what to do next (go to Dashboard and ask the assistant)
# Keep these short. They are story beats, not documentation.
SCENARIO_STORYBOOK = {
    "Baseline": {
        "persona": "You are the ops lead at a cloud kitchen company.",
        "before": "You opened the dashboard for a routine check. All KPIs look normal. Orders, discounts, commissions, and payouts are in their usual ranges.",
        "injection_summary": "No changes. This is the clean control state — a healthy network.",
        "what_to_ask": "Use this as your baseline reference. Load a scenario to see how a real incident would look.",
    },
    "Stuck Promo After Campaign End": {
        "persona": "You are the ops lead at a cloud kitchen company.",
        "before": "Marketing ran a Diwali promo on Behrouz × Swiggy last week. The campaign was supposed to end. You thought discount amount would come back down.",
        "injection_summary": "The `promo_active` flag was never flipped off in `channel_brand_config`. Orders on that brand × channel continued to record a ~18% discount even though ops considers the campaign over.",
        "what_to_ask": "Go to the Dashboard tab. You will see discount amount still elevated. Ask the assistant why.",
    },
    "Brand Paused On One Channel": {
        "persona": "You are the ops lead at a cloud kitchen company.",
        "before": "Oven Story has been selling steadily on both Zomato and Swiggy. Nothing was announced about pausing listings.",
        "injection_summary": "Someone toggled `is_live = false` on Oven Story × Zomato in `channel_brand_config`. Orders on that channel are effectively zero from the change forward; Swiggy is untouched.",
        "what_to_ask": "Go to the Dashboard. Zomato numbers for Oven Story will look broken. Ask the assistant if it is a pipeline issue or something else.",
    },
    "Commission Drift Hiding Margin Pressure": {
        "persona": "You are the ops lead at a cloud kitchen company.",
        "before": "Gross sales have been trending up. Finance was expecting net payout to move up in line with gross.",
        "injection_summary": "The platform renegotiated and bumped `effective_commission_pct` on one brand × channel. Data is correct, pipeline is healthy — margins actually shrank.",
        "what_to_ask": "Go to the Dashboard. Gross is up but payout is flat. Ask the assistant what is eating margin. This is a null-incident case — the right answer is 'not a pipeline bug.'",
    },
    "Menu Sync Staleness": {
        "persona": "You are the ops lead at a cloud kitchen company.",
        "before": "Prices were updated internally a few days ago. You expected Swiggy to reflect the new pricing.",
        "injection_summary": "The `menu_last_synced_at` on Faasos × Swiggy is days behind the last internal update. Customers are being served outdated prices. AOV looks artificially high.",
        "what_to_ask": "Go to the Dashboard. AOV on that pair will look off. Ask the assistant to dig in.",
    },
    "Orders Up, Net Sales Flat (AOV Collapse)": {
        "persona": "You are the ops lead at a cloud kitchen company.",
        "before": "You launched a new low-priced SKU and orders have been climbing. You expected net sales to climb with them.",
        "injection_summary": "55 extra orders were added at a much lower ticket size for The Bowl Company × Swiggy. No pipeline bug. AOV collapsed because the mix shifted to cheap SKUs.",
        "what_to_ask": "Go to the Dashboard. Orders are up, net sales are flat. Ask the assistant what is eating revenue. The right answer is 'business reality, not a bug.'",
    },
    "Discount Spike Without Promo Campaign": {
        "persona": "You are the ops lead at a cloud kitchen company.",
        "before": "Ops comms say no active campaign on Sweet Truth × Zomato. You expect discount_amount to sit near zero for that pair.",
        "injection_summary": "The `promo_active` flag was quietly flipped true on that pair in `channel_brand_config` with a 12% discount. Orders that day were re-written with the discount applied. Pipeline is healthy.",
        "what_to_ask": "Go to the Dashboard. Discount amount is nonzero for a pair that should not be discounting. Ask the assistant why. Another config-drift case, not a pipeline bug.",
    },
    "Stale Aggregate After Swiggy Feed": {
        "persona": "You are the ops lead at a cloud kitchen company.",
        "before": "Swiggy orders landed in raw for Behrouz, and ops expected the dashboard to move right away.",
        "injection_summary": "Fresh Behrouz × Swiggy orders were inserted into `orders`, but the aggregate table was not rebuilt. Raw is newer than dashboard state.",
        "what_to_ask": "Go to the Dashboard. Ask whether the issue is a real demand change or a stale aggregate.",
    },
    "Partial Aggregate Refresh (Zomato Missing)": {
        "persona": "You are the ops lead at a cloud kitchen company.",
        "before": "The morning refresh ran, but the Zomato view suddenly looked empty across brands.",
        "injection_summary": "The refresh only partially completed and dropped Zomato partitions from `channel_daily_metrics`. Raw data still exists.",
        "what_to_ask": "Go to the Dashboard. Ask why Zomato disappeared after the refresh.",
    },
    "Duplicate ONDC Events": {
        "persona": "You are the ops lead at a cloud kitchen company.",
        "before": "ONDC was stable yesterday. Today, order count looks suspiciously high.",
        "injection_summary": "The ONDC connector replayed the same order events twice without a dedup step, so both raw and aggregate are inflated.",
        "what_to_ask": "Go to the Dashboard. Ask whether ONDC growth is real or a pipeline/data issue.",
    },
    "Late Swiggy Feed Skew": {
        "persona": "You are the ops lead at a cloud kitchen company.",
        "before": "Every morning, Swiggy starts low and catches up by lunch. The team is not sure if this is a pipeline issue or just delayed feed timing.",
        "injection_summary": "The Swiggy feed arrived after the dashboard build window. Orders exist in raw with later `ingested_at`, but the aggregate snapshot is missing them.",
        "what_to_ask": "Go to the Dashboard. Ask why Swiggy looks low early in the day.",
    },
    "Cancelled Orders Missing From Aggregate": {
        "persona": "You are the ops lead at a cloud kitchen company.",
        "before": "Customer support saw cancellations on Faasos × Swiggy, but the dashboard still shows a very clean cancellation rate.",
        "injection_summary": "The aggregate transform lost cancelled orders for that slice, so the dashboard under-reports cancellation pressure.",
        "what_to_ask": "Go to the Dashboard. Ask if the clean cancellation rate is trustworthy.",
    },
    "Mixed: Promo + Stale Aggregate": {
        "persona": "You are the ops lead at a cloud kitchen company.",
        "before": "A promo should have ended on Behrouz × Swiggy, but the revenue picture still looks wrong even after ops checked the campaign.",
        "injection_summary": "Two things are true: the promo flag is stuck on, and the aggregate did not refresh after raw orders changed. The business issue is real, and the dashboard is stale too.",
        "what_to_ask": "Go to the Dashboard. Ask whether this is just promo leakage or a stale-data problem as well.",
    },
    "Mixed: Listing Pause + Stale Aggregate": {
        "persona": "You are the ops lead at a cloud kitchen company.",
        "before": "Oven Story × Zomato was paused, but finance still sees Zomato sales on the dashboard.",
        "injection_summary": "The listing really is paused in config and raw orders were removed, but the aggregate was not rebuilt, so the dashboard still shows the old sales.",
        "what_to_ask": "Go to the Dashboard. Ask why a paused listing still looks live in the reported numbers.",
    },
    "Compound: Promo + Menu Sync": {
        "persona": "You are the ops lead at a cloud kitchen company.",
        "before": "Behrouz × Zomato has been the star performer. Finance was surprised net revenue dropped sharply.",
        "injection_summary": "Two things at once on Behrouz × Zomato: the Diwali promo never got turned off, AND the menu sync is five days behind on old higher prices. Both are real, both contribute to the revenue story.",
        "what_to_ask": "Go to the Dashboard. Ask the assistant why Behrouz × Zomato is bleeding margin. A strong answer separates both contributing causes.",
    },
}


APP_CSS = """
/* ============================================================
   FoodOps Incident Lab — design tokens
   ============================================================ */
:root {
  --bg-page:        #0F1115;
  --bg-surface:     #1A1D26;
  --bg-surface-2:   #242833;
  --border:         #2E3440;
  --border-soft:    #242833;
  --text-primary:   #F0F2F7;
  --text-secondary: #A8AEBF;
  --text-muted:     #6B7180;
  --accent:         #FF6B35;
  --accent-soft:    rgba(255,107,53,0.14);
  --accent-border:  rgba(255,107,53,0.45);
  --signal-ok:      #4ADE80;
  --signal-warn:    #FBBF24;
  --signal-bad:     #F87171;
  --chip-brand:     #60A5FA;
  --chip-channel:   #A78BFA;
}

.app-shell { max-width: 1480px; margin: 0 auto; }

/* --- Hero -------------------------------------------------- */
.hero {
  padding: 22px 26px 16px 26px;
  border-bottom: 1px solid var(--border);
  margin-bottom: 6px;
  background: linear-gradient(180deg, var(--bg-surface) 0%, var(--bg-page) 100%);
}
.hero h1 {
  margin: 0;
  font-size: 1.85rem;
  font-weight: 700;
  letter-spacing: -0.01em;
  color: var(--text-primary);
}
.hero p {
  margin: 6px 0 0 0;
  color: var(--text-secondary);
  font-size: 0.95rem;
}

/* --- Status strip (persistent across tabs) ---------------- */
.status-strip {
  display: flex;
  gap: 20px;
  align-items: center;
  flex-wrap: wrap;
  padding: 12px 20px;
  margin: 8px 0 14px 0;
  border: 1px solid var(--border);
  border-radius: 10px;
  background: var(--bg-surface);
  font-size: 0.92rem;
}
.status-strip.perturbed {
  background: var(--accent-soft);
  border-color: var(--accent-border);
}
.status-strip .status-item {
  display: flex;
  flex-direction: column;
  line-height: 1.25;
}
.status-strip .status-label {
  font-size: 0.72rem;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--text-muted);
  margin-bottom: 2px;
}
.status-strip .status-value {
  color: var(--text-primary);
  font-weight: 600;
}
.status-strip .status-headline {
  flex: 1;
  min-width: 240px;
  color: var(--text-primary);
  font-weight: 500;
}
.status-strip .status-pill {
  display: inline-block;
  padding: 2px 10px;
  border-radius: 999px;
  font-size: 0.75rem;
  font-weight: 600;
  letter-spacing: 0.02em;
}
.status-strip .pill-baseline { background: rgba(74,222,128,0.14); color: var(--signal-ok); }
.status-strip .pill-perturbed { background: rgba(255,107,53,0.18); color: var(--accent); }
.status-strip .pill-kpi { background: rgba(96,165,250,0.14); color: var(--chip-brand); }
.status-strip .pill-data { background: rgba(251,191,36,0.14); color: var(--signal-warn); }
.status-strip .pill-compound { background: rgba(248,113,113,0.14); color: var(--signal-bad); }

/* --- Metric cards ----------------------------------------- */
.metric-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 12px;
}
.metric-card {
  position: relative;
  border: 1px solid var(--border);
  border-left: 3px solid var(--border);
  border-radius: 10px;
  padding: 14px 16px;
  background: var(--bg-surface);
  transition: border-color 120ms ease;
}
.metric-card:hover { border-color: var(--accent-border); }
.metric-card.ok    { border-left-color: var(--signal-ok); }
.metric-card.warn  { border-left-color: var(--signal-warn); }
.metric-card.bad   { border-left-color: var(--signal-bad); }
.metric-label {
  font-size: 0.78rem;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--text-muted);
  margin-bottom: 6px;
}
.metric-value {
  font-size: 1.7rem;
  font-weight: 700;
  color: var(--text-primary);
  line-height: 1.1;
}
.metric-sub {
  font-size: 0.82rem;
  color: var(--text-secondary);
  margin-top: 6px;
}

/* --- Story/helper boxes ----------------------------------- */
.story-box {
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 14px 16px;
  background: var(--bg-surface);
  color: var(--text-primary);
}
.story-box h3 { margin: 0 0 8px 0; color: var(--text-primary); }

/* --- KPI reference list ----------------------------------- */
.kpi-ref-list {
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 12px 16px;
  background: var(--bg-surface);
  color: var(--text-secondary);
  font-size: 0.9rem;
  line-height: 1.55;
}
.kpi-ref-list strong { color: var(--text-primary); }

/* --- Gradio component overrides --------------------------- */
.gradio-container { background: var(--bg-page) !important; }
.tab-nav button { font-weight: 600; letter-spacing: 0.01em; }

/* --- Scenario diff view ----------------------------------- */
.diff-header {
  padding: 12px 16px;
  background: var(--bg-surface);
  border: 1px solid var(--border);
  border-radius: 10px;
  margin-bottom: 12px;
  color: var(--text-primary);
}
.diff-header .diff-sub { color: var(--text-muted); margin-left: 8px; font-weight: 400; }

.diff-empty {
  padding: 18px 20px;
  background: var(--bg-surface);
  border: 1px dashed var(--border);
  border-radius: 10px;
  color: var(--text-secondary);
}
.diff-empty strong { color: var(--text-primary); }
.diff-empty p { margin: 6px 0 0 0; }

.diff-table-card {
  border: 1px solid var(--border);
  border-radius: 10px;
  background: var(--bg-surface);
  margin-bottom: 12px;
  overflow: hidden;
}
.diff-table-header {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 10px 16px;
  background: var(--bg-surface-2);
  border-bottom: 1px solid var(--border);
  flex-wrap: wrap;
}
.diff-table-name {
  font-family: "SF Mono", "Menlo", monospace;
  font-weight: 600;
  color: var(--accent);
}
.diff-table-counts { color: var(--text-secondary); font-size: 0.88rem; }
.diff-badges { margin-left: auto; display: flex; gap: 6px; }
.diff-badge {
  padding: 2px 10px;
  border-radius: 999px;
  font-size: 0.72rem;
  font-weight: 600;
  letter-spacing: 0.02em;
}
.badge-changed { background: rgba(251,191,36,0.16); color: var(--signal-warn); }
.badge-added   { background: rgba(74,222,128,0.16); color: var(--signal-ok); }
.badge-removed { background: rgba(248,113,113,0.16); color: var(--signal-bad); }
.badge-none    { background: rgba(168,174,191,0.12); color: var(--text-muted); }

.diff-section-label {
  padding: 10px 16px 6px 16px;
  font-size: 0.74rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--text-muted);
}

.diff-change-card {
  margin: 0 12px 10px 12px;
  padding: 10px 12px;
  background: var(--bg-surface-2);
  border-radius: 6px;
  border-left: 3px solid var(--signal-warn);
}
.diff-key {
  font-family: "SF Mono", "Menlo", monospace;
  color: var(--chip-brand);
  font-size: 0.85rem;
  margin-bottom: 6px;
}
.diff-line {
  display: flex;
  gap: 10px;
  align-items: center;
  padding: 3px 0;
  font-size: 0.88rem;
}
.diff-col {
  min-width: 180px;
  color: var(--text-secondary);
  font-family: "SF Mono", "Menlo", monospace;
}
.diff-before { color: var(--signal-bad); text-decoration: line-through; }
.diff-arrow { color: var(--text-muted); }
.diff-after { color: var(--signal-ok); font-weight: 600; }

.diff-table {
  width: calc(100% - 24px);
  margin: 0 12px 12px 12px;
  border-collapse: collapse;
  font-size: 0.85rem;
}
.diff-table th {
  background: var(--bg-surface-2);
  color: var(--text-secondary);
  text-align: left;
  padding: 6px 10px;
  font-weight: 600;
  font-size: 0.75rem;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  border-bottom: 1px solid var(--border);
}
.diff-table td {
  padding: 6px 10px;
  border-bottom: 1px solid var(--border-soft);
  color: var(--text-primary);
}
.diff-table tr:hover td { background: var(--bg-surface-2); }
.diff-overflow {
  color: var(--text-muted);
  font-size: 0.82rem;
  padding: 6px 16px 10px 16px;
}

/* --- Pipeline state panel --------------------------------- */
.pipeline-panel {
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 14px 16px;
  background: var(--bg-surface);
  margin-bottom: 14px;
}
.pipeline-panel.pipeline-stale { border-color: var(--signal-bad); background: rgba(248,113,113,0.06); }
.pipeline-header {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 10px;
}
.pipeline-title {
  font-size: 0.78rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--text-muted);
}
.pipeline-badge {
  font-size: 0.72rem;
  font-weight: 700;
  letter-spacing: 0.08em;
}
.pipeline-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 10px;
  margin-bottom: 12px;
}
.pipeline-kv {
  background: var(--bg-surface-2);
  border-radius: 6px;
  padding: 8px 10px;
}
.pipeline-k {
  font-size: 0.7rem;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--text-muted);
  margin-bottom: 2px;
}
.pipeline-v { color: var(--text-primary); font-weight: 600; font-size: 0.95rem; }
.pipeline-tables {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.86rem;
}
.pipeline-tables th {
  color: var(--text-muted);
  text-align: left;
  padding: 4px 10px;
  font-size: 0.72rem;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  border-bottom: 1px solid var(--border);
}
.pipeline-tables td {
  padding: 4px 10px;
  border-bottom: 1px solid var(--border-soft);
  color: var(--text-primary);
  font-family: "SF Mono", "Menlo", monospace;
}

/* --- Tool trace timeline ---------------------------------- */
.trace-timeline {
  display: flex;
  flex-direction: column;
  gap: 8px;
  max-height: 280px;
  overflow-y: auto;
}
.trace-empty {
  color: var(--text-muted);
  padding: 12px;
  background: var(--bg-surface);
  border: 1px dashed var(--border);
  border-radius: 8px;
  font-size: 0.9rem;
}
.trace-raw {
  background: var(--bg-surface);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 10px 12px;
  font-family: "SF Mono", "Menlo", monospace;
  font-size: 0.82rem;
  color: var(--text-secondary);
  overflow-x: auto;
  max-height: 280px;
}
.trace-event {
  display: flex;
  gap: 10px;
  align-items: flex-start;
  padding: 8px 10px;
  background: var(--bg-surface);
  border: 1px solid var(--border);
  border-radius: 8px;
}

/* --- Chat window ------------------------------------------ */
.chat-window,
.chat-window > div {
  background: var(--bg-surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: 10px !important;
}
.chat-window [data-testid="chatbot"],
.chat-window .message-wrap,
.chat-window .message-row {
  background: transparent !important;
}

/* --- Dark dropdowns -------------------------------------- */
.dark-dropdown,
.dark-dropdown > div {
  background: transparent !important;
}
.dark-dropdown button,
.dark-dropdown input,
.dark-dropdown select,
.dark-dropdown [role="combobox"] {
  background: var(--bg-surface) !important;
  color: var(--text-primary) !important;
  border: 1px solid var(--border) !important;
  border-radius: 10px !important;
  box-shadow: none !important;
}
.dark-dropdown span,
.dark-dropdown label,
.dark-dropdown svg {
  color: var(--text-primary) !important;
  fill: var(--text-primary) !important;
}
#scenario-dropdown,
#overview-date-dropdown {
  background: transparent !important;
}
#scenario-dropdown *,
#overview-date-dropdown * {
  color: var(--text-primary) !important;
}
#scenario-dropdown button,
#scenario-dropdown input,
#scenario-dropdown select,
#scenario-dropdown [role="combobox"],
#scenario-dropdown [data-testid="dropdown"],
#scenario-dropdown [data-slot="input"],
#overview-date-dropdown button,
#overview-date-dropdown input,
#overview-date-dropdown select,
#overview-date-dropdown [role="combobox"],
#overview-date-dropdown [data-testid="dropdown"],
#overview-date-dropdown [data-slot="input"] {
  background: var(--bg-surface) !important;
  color: var(--text-primary) !important;
  border: 1px solid var(--border) !important;
  border-radius: 10px !important;
  box-shadow: none !important;
}
.trace-step {
  flex: 0 0 24px;
  height: 24px;
  border-radius: 999px;
  background: var(--accent-soft);
  color: var(--accent);
  font-size: 0.8rem;
  font-weight: 700;
  display: flex;
  align-items: center;
  justify-content: center;
}
.trace-body { flex: 1; min-width: 0; }
.trace-tool {
  font-family: "SF Mono", "Menlo", monospace;
  color: var(--chip-brand);
  font-weight: 600;
}
.trace-args {
  color: var(--text-secondary);
  font-family: "SF Mono", "Menlo", monospace;
  font-size: 0.85rem;
}
.trace-result {
  margin-top: 4px;
  color: var(--text-muted);
  font-family: "SF Mono", "Menlo", monospace;
  font-size: 0.8rem;
  white-space: pre-wrap;
  overflow-wrap: break-word;
}

/* --- Storybook (Scenarios tab) ---------------------------- */
.storybook {
  display: flex;
  flex-direction: column;
  gap: 4px;
  max-width: 820px;
  margin: 0 auto;
}
.story-act {
  background: var(--bg-surface);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 18px 22px;
}
.story-act-injection { border-left: 3px solid var(--signal-warn); }
.story-act-next      { border-left: 3px solid var(--accent); }
.story-act-head {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 10px;
}
.story-num {
  flex: 0 0 28px;
  height: 28px;
  border-radius: 999px;
  background: var(--accent-soft);
  color: var(--accent);
  font-weight: 700;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 0.95rem;
}
.story-title {
  font-size: 0.76rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--text-muted);
  font-weight: 600;
}
.story-body p {
  margin: 6px 0;
  color: var(--text-primary);
  line-height: 1.55;
  font-size: 0.95rem;
}
.story-persona {
  color: var(--text-secondary);
  font-style: italic;
  font-size: 0.88rem;
  margin-bottom: 6px;
}
.story-changes {
  margin-top: 10px;
  padding: 8px 12px;
  background: var(--bg-surface-2);
  border-radius: 6px;
  color: var(--text-secondary);
  font-size: 0.85rem;
  font-family: "SF Mono", "Menlo", monospace;
}
.story-changes code {
  color: var(--accent);
  background: transparent;
}
.story-suggested {
  margin-top: 10px;
  padding: 10px 12px;
  background: var(--bg-surface-2);
  border-radius: 6px;
  border-left: 2px solid var(--chip-brand);
}
.story-suggested-label {
  font-size: 0.7rem;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--text-muted);
}
.story-suggested-text {
  margin-top: 4px;
  color: var(--text-primary);
  font-style: italic;
}
.story-cta {
  margin-top: 12px;
  padding: 10px 12px;
  background: var(--accent-soft);
  border-radius: 6px;
  color: var(--text-primary);
  font-size: 0.92rem;
}
.story-arrow {
  text-align: center;
  color: var(--text-muted);
  font-size: 1.4rem;
  line-height: 1;
  padding: 2px 0;
}

/* --- Clean dashboard helpers ------------------------------ */
.dashboard-intro {
  color: var(--text-secondary);
  font-size: 0.9rem;
  margin-bottom: 4px;
}
.dashboard-freshness {
  color: var(--text-muted);
  font-size: 0.85rem;
}

/* --- Pipeline DAG ----------------------------------------- */
.dag-container {
  border: 1px solid var(--border);
  border-radius: 12px;
  background: var(--bg-surface);
  padding: 14px 18px;
  margin-bottom: 12px;
}
.dag-header {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 8px;
}
.dag-title {
  font-size: 0.78rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--text-muted);
}
.dag-status {
  padding: 2px 10px;
  border-radius: 999px;
  font-size: 0.7rem;
  font-weight: 700;
  letter-spacing: 0.05em;
}
.dag-status-healthy {
  background: rgba(74,222,128,0.16);
  color: var(--signal-ok);
}
.dag-status-broken {
  background: rgba(248,113,113,0.16);
  color: var(--signal-bad);
}
.pipeline-dag {
  display: block;
  margin: 4px 0;
}
.dag-legend {
  color: var(--text-muted);
  font-size: 0.8rem;
  margin-top: 6px;
  font-style: italic;
}

/* --- Table content panel (shown when you click a node) ---- */
.table-panel {
  border: 1px solid var(--border);
  border-radius: 10px;
  background: var(--bg-surface);
  overflow: hidden;
}
.table-panel-empty {
  padding: 18px;
  color: var(--text-muted);
  background: var(--bg-surface);
  border: 1px dashed var(--border);
  border-radius: 10px;
  text-align: center;
  font-size: 0.9rem;
}
.table-panel-error {
  padding: 12px;
  color: var(--signal-bad);
  background: rgba(248,113,113,0.08);
  border: 1px solid var(--signal-bad);
  border-radius: 10px;
  font-family: "SF Mono", "Menlo", monospace;
  font-size: 0.85rem;
}
.table-panel-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 10px 16px;
  background: var(--bg-surface-2);
  border-bottom: 1px solid var(--border);
}
.table-panel-name {
  font-family: "SF Mono", "Menlo", monospace;
  font-weight: 600;
  color: var(--accent);
}
.table-panel-count {
  color: var(--text-secondary);
  font-size: 0.85rem;
}
.table-panel-scroll {
  max-height: 360px;
  overflow: auto;
}
.table-panel-overflow {
  padding: 8px 16px;
  color: var(--text-muted);
  font-size: 0.82rem;
  background: var(--bg-surface-2);
  border-top: 1px solid var(--border);
}

/* --- Node-click buttons (sit invisibly over DAG nodes) ---- */
.node-click-row {
  display: flex;
  gap: 8px;
  margin-top: -6px;
  padding: 0 20px;
}
.node-click-row button {
  flex: 1;
  background: transparent !important;
  border: 1px dashed var(--border) !important;
  color: var(--text-secondary) !important;
  font-size: 0.78rem !important;
  padding: 4px 8px !important;
}
.node-click-row button:hover {
  border-color: var(--accent) !important;
  color: var(--accent) !important;
}

/* --- Reward / scoring card ----------------------------- */
.reward-empty {
  padding: 18px;
  border: 1px dashed var(--border);
  border-radius: 10px;
  color: var(--text-secondary);
  background: var(--bg-surface);
  font-size: 0.9rem;
}
.reward-card {
  border: 1px solid var(--border);
  border-radius: 12px;
  background: var(--bg-surface);
  padding: 14px 18px;
}
.reward-header {
  display: flex;
  align-items: center;
  gap: 14px;
  margin-bottom: 12px;
}
.reward-title {
  font-size: 0.78rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--text-muted);
}
.reward-badge {
  padding: 2px 10px;
  border-radius: 999px;
  font-size: 0.72rem;
  font-weight: 700;
  letter-spacing: 0.06em;
  background: var(--bg-surface-2);
}
.reward-total {
  margin-left: auto;
  font-size: 1.8rem;
  font-weight: 700;
  font-family: "SF Mono", "Menlo", monospace;
}
.reward-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.9rem;
  margin-bottom: 10px;
}
.reward-table th {
  text-align: left;
  padding: 6px 8px;
  color: var(--text-muted);
  font-size: 0.72rem;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  border-bottom: 1px solid var(--border);
}
.reward-table td {
  padding: 6px 8px;
  border-bottom: 1px solid var(--border-soft);
  color: var(--text-primary);
}
.reward-explain {
  padding: 10px 12px;
  background: var(--bg-surface-2);
  border-radius: 6px;
  color: var(--text-secondary);
  font-size: 0.88rem;
  line-height: 1.55;
}

/* --- Markdown / form readability ------------------------- */
.gradio-container .prose,
.gradio-container .prose p,
.gradio-container .prose li,
.gradio-container .prose span {
  color: var(--text-secondary) !important;
}
.gradio-container .prose h1,
.gradio-container .prose h2,
.gradio-container .prose h3,
.gradio-container .prose h4,
.gradio-container .prose strong,
.gradio-container .prose b {
  color: var(--text-primary) !important;
}
.gradio-container label,
.gradio-container .block-title,
.gradio-container .form label,
.gradio-container .gr-block-label {
  color: var(--text-primary) !important;
}
.gradio-container input,
.gradio-container textarea,
.gradio-container select {
  color: var(--text-primary) !important;
}
.gradio-container input::placeholder,
.gradio-container textarea::placeholder {
  color: var(--text-muted) !important;
  opacity: 1 !important;
}
.gradio-container input,
.gradio-container textarea {
  caret-color: var(--text-primary) !important;
}

/* --- Simple dark text input ------------------------------ */
.simple-text-input textarea,
.simple-text-input input {
  background: var(--bg-surface) !important;
  color: var(--text-primary) !important;
  border: 1px solid var(--border) !important;
  border-radius: 10px !important;
  padding: 12px 14px !important;
  min-height: 56px !important;
  box-shadow: none !important;
}
.simple-text-input textarea::placeholder,
.simple-text-input input::placeholder {
  color: var(--text-muted) !important;
  opacity: 1 !important;
}

/* --- Incident report panel ------------------------------- */
.incident-form-shell {
  margin-top: 10px;
  padding: 14px 0 4px 0;
}
.incident-guide {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 10px;
  margin: 10px 0 14px 0;
}
.incident-guide-card {
  border: 1px solid var(--border);
  border-radius: 10px;
  background: var(--bg-surface);
  padding: 12px 14px;
}
.incident-guide-title {
  color: var(--text-primary);
  font-weight: 700;
  font-family: "SF Mono", "Menlo", monospace;
  font-size: 0.82rem;
  margin-bottom: 6px;
}
.incident-guide-copy {
  color: var(--text-secondary);
  font-size: 0.84rem;
  line-height: 1.45;
}

/* --- Scenario catalog ------------------------------------ */
.scenario-catalog {
  display: flex;
  flex-direction: column;
  gap: 14px;
  margin: 8px 0 16px 0;
}
.scenario-group h4 {
  margin: 0 0 8px 0;
  color: var(--text-primary);
  font-size: 0.92rem;
}
.scenario-group-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 10px;
}
.scenario-catalog-card {
  border: 1px solid var(--border);
  border-radius: 10px;
  background: var(--bg-surface);
  padding: 12px 14px;
  min-height: 84px;
}
.scenario-catalog-title {
  color: var(--text-primary);
  font-weight: 600;
  margin-bottom: 6px;
}
.scenario-catalog-copy {
  color: var(--text-secondary);
  font-size: 0.84rem;
  line-height: 1.45;
}

@media (max-width: 980px) {
  .incident-guide,
  .scenario-group-grid {
    grid-template-columns: 1fr;
  }
}
"""

TABLE_SQL_REFERENCE = {
    "brands": """
create table brands (
  brand_id varchar primary key,
  brand_name varchar,
  cuisine_type varchar,
  launched_at date,
  is_active boolean
)
""".strip(),
    "channels": """
create table channels (
  channel_id varchar primary key,
  channel_name varchar,
  channel_type varchar,
  default_commission_pct double,
  feed_delay_minutes integer,
  is_active boolean
)
""".strip(),
    "channel_brand_config": """
create table channel_brand_config (
  brand_id varchar,
  channel_id varchar,
  is_live boolean,
  promo_active boolean,
  promo_discount_pct double,
  effective_commission_pct double,
  menu_last_synced_at timestamp,
  notes varchar
)
""".strip(),
    "orders": """
create table orders (
  order_id varchar,
  ordered_at timestamp,
  ingested_at timestamp,
  brand_id varchar,
  channel_id varchar,
  gross_amount double,
  discount_amount double,
  delivered_amount double,
  refund_amount double,
  status varchar,
  customer_phone varchar
)
""".strip(),
    "channel_daily_metrics": """
create or replace table channel_daily_metrics as
select
  cast(o.ordered_at as date) as metric_date,
  b.brand_name,
  c.channel_name,
  cfg.is_live,
  cfg.promo_active,
  cfg.effective_commission_pct,
  count(*) as total_orders,
  round(sum(o.gross_amount), 2) as gross_sales,
  round(sum(o.delivered_amount) - sum(o.refund_amount), 2) as net_sales,
  round(
    (sum(o.delivered_amount) - sum(o.refund_amount))
    - ((sum(o.delivered_amount) - sum(o.refund_amount)) * (cfg.effective_commission_pct / 100.0)),
    2
  ) as net_payout
from orders o
left join brands b on b.brand_id = o.brand_id
left join channels c on c.channel_id = o.channel_id
left join channel_brand_config cfg on cfg.brand_id = o.brand_id and cfg.channel_id = o.channel_id
group by 1, 2, 3, 4, 5, 6
""".strip(),
}


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


def connect() -> duckdb.DuckDBPyConnection:
    ensure_dirs()
    return duckdb.connect(str(DB_PATH))


def write_log(action: str, message: str, details: dict[str, Any] | None = None) -> None:
    ensure_dirs()
    payload = {"runs": []}
    if LOG_PATH.exists():
        try:
            payload = json.loads(LOG_PATH.read_text())
        except json.JSONDecodeError:
            payload = {"runs": []}
    payload.setdefault("runs", []).append(
        {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "action": action,
            "message": message,
            "details": details or {},
        }
    )
    payload["runs"] = payload["runs"][-120:]
    LOG_PATH.write_text(json.dumps(payload, indent=2))


def reset_logs() -> None:
    ensure_dirs()
    LOG_PATH.write_text(json.dumps({"runs": []}, indent=2))


def sql_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def initialize_database(seed: int = DEFAULT_SEED) -> str:
    ensure_dirs()
    rng = random.Random(seed)
    reset_logs()
    with connect() as conn:
        conn.execute("drop table if exists channel_daily_metrics")
        conn.execute("drop table if exists orders")
        conn.execute("drop table if exists channel_brand_config")
        conn.execute("drop table if exists channels")
        conn.execute("drop table if exists brands")

        conn.execute(
            """
            create table brands (
              brand_id varchar primary key,
              brand_name varchar,
              cuisine_type varchar,
              launched_at date,
              is_active boolean
            )
            """
        )
        conn.execute(
            """
            create table channels (
              channel_id varchar primary key,
              channel_name varchar,
              channel_type varchar,
              default_commission_pct double,
              feed_delay_minutes integer,
              is_active boolean
            )
            """
        )
        conn.execute(
            """
            create table channel_brand_config (
              brand_id varchar,
              channel_id varchar,
              is_live boolean,
              promo_active boolean,
              promo_discount_pct double,
              effective_commission_pct double,
              menu_last_synced_at timestamp,
              notes varchar
            )
            """
        )
        conn.execute(
            """
            create table orders (
              order_id varchar,
              ordered_at timestamp,
              ingested_at timestamp,
              brand_id varchar,
              channel_id varchar,
              gross_amount double,
              discount_amount double,
              delivered_amount double,
              refund_amount double,
              status varchar,
              customer_phone varchar
            )
            """
        )

        today = date.today()
        for idx, brand in enumerate(BRANDS):
            launched = today - timedelta(days=400 + (idx * 70))
            conn.execute(
                "insert into brands values (?, ?, ?, ?, ?)",
                [brand["brand_id"], brand["brand_name"], brand["cuisine_type"], launched, True],
            )
        for channel in CHANNELS:
            conn.execute(
                "insert into channels values (?, ?, ?, ?, ?, ?)",
                [
                    channel["channel_id"],
                    channel["channel_name"],
                    channel["channel_type"],
                    channel["default_commission_pct"],
                    channel["feed_delay_minutes"],
                    True,
                ],
            )

        sync_base = datetime.combine(today, time(10, 0))
        for idx, row in enumerate(CONFIG_ROWS):
            brand_id, channel_id, is_live, promo_active, promo_discount_pct, commission_pct = row
            synced_at = sync_base - timedelta(hours=(idx % 4) * 8)
            conn.execute(
                "insert into channel_brand_config values (?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    brand_id,
                    channel_id,
                    is_live,
                    promo_active,
                    promo_discount_pct,
                    commission_pct,
                    synced_at,
                    "baseline",
                ],
            )

        order_counter = 1
        for offset in range(DAYS_OF_DATA):
            order_date = today - timedelta(days=DAYS_OF_DATA - 1 - offset)
            weekday_boost = 1.15 if order_date.weekday() in (4, 5, 6) else 1.0
            for brand_id, channel_id, is_live, promo_active, promo_discount_pct, _ in CONFIG_ROWS:
                if not is_live:
                    continue
                base_orders = BRAND_BASES[brand_id]["orders"]
                order_target = max(
                    2,
                    round(
                        base_orders
                        * CHANNEL_MULTIPLIERS[channel_id]["orders"]
                        * weekday_boost
                        * rng.uniform(0.82, 1.18)
                    ),
                )
                for _ in range(order_target):
                    aov = BRAND_BASES[brand_id]["aov"] * CHANNEL_MULTIPLIERS[channel_id]["aov"] * rng.uniform(0.84, 1.18)
                    gross_amount = round(aov, 2)
                    discount_ratio = rng.uniform(0.04, 0.12) + (promo_discount_pct / 100 if promo_active else 0)
                    discount_amount = round(gross_amount * min(discount_ratio, 0.35), 2)
                    status_roll = rng.random()
                    if status_roll < 0.13:
                        status = "cancelled"
                        delivered_amount = 0.0
                    else:
                        status = "delivered"
                        delivered_amount = round(gross_amount - discount_amount, 2)
                    refund_amount = round(delivered_amount * rng.uniform(0.35, 1.0), 2) if status == "delivered" and rng.random() < 0.06 else 0.0
                    ordered_at = datetime.combine(order_date, time(hour=rng.randint(8, 23), minute=rng.randint(0, 59)))
                    ingested_at = ordered_at + timedelta(minutes=CHANNELS[[c["channel_id"] for c in CHANNELS].index(channel_id)]["feed_delay_minutes"] + rng.randint(0, 25))
                    phone_suffix = rng.randint(1000, 9999)
                    conn.execute(
                        "insert into orders values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        [
                            f"ORD-{order_counter:06d}",
                            ordered_at,
                            ingested_at,
                            brand_id,
                            channel_id,
                            gross_amount,
                            discount_amount,
                            delivered_amount,
                            refund_amount,
                            status,
                            f"98{phone_suffix:08d}"[:10],
                        ],
                    )
                    order_counter += 1

        refresh_aggregation(conn, log_action=False)
    write_log("reset_baseline", f"Rebuilt the FoodOps baseline with seed {seed}.", {"seed": seed})
    return f"Baseline rebuilt with seed {seed}. Network is back to a clean starting point."


def refresh_aggregation(conn: duckdb.DuckDBPyConnection | None = None, *, log_action: bool = True) -> str:
    owns_conn = conn is None
    if conn is None:
        conn = connect()
    conn.execute(
        """
        create or replace table channel_daily_metrics as
        with coverage as (
          select
            sum(case when is_live then 1 else 0 end)::double / nullif(count(*), 0) as coverage_ratio
          from channel_brand_config
        )
        select
          cast(o.ordered_at as date) as metric_date,
          b.brand_id,
          b.brand_name,
          c.channel_id,
          c.channel_name,
          cfg.is_live,
          cfg.promo_active,
          cfg.promo_discount_pct,
          cfg.effective_commission_pct,
          cfg.menu_last_synced_at,
          count(*) as total_orders,
          sum(case when o.status = 'delivered' then 1 else 0 end) as delivered_orders,
          sum(case when o.status = 'cancelled' then 1 else 0 end) as cancelled_orders,
          round(sum(o.gross_amount), 2) as gross_sales,
          round(sum(o.discount_amount), 2) as discount_amount,
          round(sum(o.refund_amount), 2) as refund_amount,
          round(sum(o.delivered_amount) - sum(o.refund_amount), 2) as net_sales,
          round(
            (sum(o.delivered_amount) - sum(o.refund_amount))
            - ((sum(o.delivered_amount) - sum(o.refund_amount)) * (cfg.effective_commission_pct / 100.0)),
            2
          ) as net_payout,
          round(
            ((sum(o.delivered_amount) - sum(o.refund_amount)) * (cfg.effective_commission_pct / 100.0)),
            2
          ) as commission_amount,
          round(
            (sum(o.delivered_amount) - sum(o.refund_amount)) / nullif(sum(case when o.status = 'delivered' then 1 else 0 end), 0),
            2
          ) as aov,
          round(sum(case when o.status = 'cancelled' then 1 else 0 end)::double / nullif(count(*), 0), 4) as cancellation_rate,
          round((select coverage_ratio from coverage), 4) as brand_channel_coverage,
          max(o.ingested_at) as latest_ingested_at
        from orders o
        left join brands b on b.brand_id = o.brand_id
        left join channels c on c.channel_id = o.channel_id
        left join channel_brand_config cfg on cfg.brand_id = o.brand_id and cfg.channel_id = o.channel_id
        group by
          cast(o.ordered_at as date),
          b.brand_id,
          b.brand_name,
          c.channel_id,
          c.channel_name,
          cfg.is_live,
          cfg.promo_active,
          cfg.promo_discount_pct,
          cfg.effective_commission_pct,
          cfg.menu_last_synced_at
        order by metric_date, brand_name, channel_name
        """
    )
    if log_action:
        write_log("refresh_aggregation", "Rebuilt channel_daily_metrics from the latest orders and config state.")
    if owns_conn:
        conn.close()
    return "Aggregation refreshed. Overview and assistant now read the updated joined metrics."


def get_recent_dates() -> list[str]:
    with connect() as conn:
        rows = conn.execute("select distinct metric_date from channel_daily_metrics order by metric_date desc limit 14").fetchall()
    return [str(row[0]) for row in rows]


def get_default_focus_date() -> str:
    dates = get_recent_dates()
    return dates[0] if dates else date.today().isoformat()


def metric_cards_html(metric_date: str) -> str:
    with connect() as conn:
        row = conn.execute(
            """
            select
              coalesce(sum(total_orders), 0),
              coalesce(sum(gross_sales), 0),
              coalesce(sum(net_sales), 0),
              coalesce(sum(net_payout), 0),
              coalesce(sum(discount_amount), 0),
              coalesce(avg(aov), 0),
              max(brand_channel_coverage),
              max(latest_ingested_at)
            from channel_daily_metrics
            where metric_date = ?
            """,
            [metric_date],
        ).fetchone()
    total_orders, gross_sales, net_sales, net_payout, discount_amount, aov, coverage, latest_ingested = row
    cards = [
        ("Total Orders", f"{int(total_orders)}", "Network order volume"),
        ("Gross Sales", f"Rs {gross_sales:,.0f}", "Before discounts and refunds"),
        ("Net Sales", f"Rs {net_sales:,.0f}", "Delivered revenue after refunds"),
        ("Net Payout", f"Rs {net_payout:,.0f}", "After channel commissions"),
        ("Discounts", f"Rs {discount_amount:,.0f}", "Promo pressure on the day"),
        ("AOV", f"Rs {aov:,.0f}", "Average realized value"),
        ("Coverage", f"{(coverage or 0) * 100:.0f}%", "Brand x channel rows currently live"),
        ("Latest Ingest", latest_ingested.strftime("%d %b, %H:%M") if latest_ingested else "-", "Freshness of joined metrics"),
    ]
    pieces = ["<div class='metric-grid'>"]
    for label, value, sub in cards:
        pieces.append(
            f"<div class='metric-card'><div class='metric-label'>{label}</div><div class='metric-value'>{value}</div><div class='metric-sub'>{sub}</div></div>"
        )
    pieces.append("</div>")
    return "".join(pieces)


def get_network_tables(metric_date: str) -> tuple[list[list[Any]], list[list[Any]], list[list[Any]], list[list[Any]]]:
    with connect() as conn:
        brand_rows = conn.execute(
            """
            select
              brand_name,
              sum(total_orders) as total_orders,
              round(sum(gross_sales), 2) as gross_sales,
              round(sum(net_sales), 2) as net_sales,
              round(sum(net_payout), 2) as net_payout,
              round(avg(aov), 2) as aov
            from channel_daily_metrics
            where metric_date = ?
            group by 1
            order by total_orders desc, brand_name
            """,
            [metric_date],
        ).fetchall()
        channel_rows = conn.execute(
            """
            select
              channel_name,
              sum(total_orders) as total_orders,
              round(sum(gross_sales), 2) as gross_sales,
              round(sum(net_sales), 2) as net_sales,
              round(sum(net_payout), 2) as net_payout,
              round(avg(effective_commission_pct), 2) as avg_commission_pct
            from channel_daily_metrics
            where metric_date = ?
            group by 1
            order by total_orders desc, channel_name
            """,
            [metric_date],
        ).fetchall()
        config_rows = conn.execute(
            """
            select
              b.brand_name,
              c.channel_name,
              cfg.is_live,
              cfg.promo_active,
              cfg.promo_discount_pct,
              cfg.effective_commission_pct,
              cfg.menu_last_synced_at,
              cfg.notes
            from channel_brand_config cfg
            join brands b on b.brand_id = cfg.brand_id
            join channels c on c.channel_id = cfg.channel_id
            order by b.brand_name, c.channel_name
            """
        ).fetchall()
        story_rows = conn.execute(
            """
            select
              brand_name,
              channel_name,
              total_orders,
              round(net_sales, 2) as net_sales,
              round(net_payout, 2) as net_payout,
              promo_active,
              round(effective_commission_pct, 1) as commission_pct
            from channel_daily_metrics
            where metric_date = ?
            order by net_sales desc
            limit 12
            """,
            [metric_date],
        ).fetchall()
    return brand_rows, channel_rows, config_rows, story_rows


def get_pipeline_log_rows() -> list[list[Any]]:
    ensure_dirs()
    if not LOG_PATH.exists():
        return []
    payload = json.loads(LOG_PATH.read_text())
    rows = []
    for run in reversed(payload.get("runs", [])):
        rows.append([run["timestamp"], run["action"], run["message"]])
    return rows[:20]


def get_table_counts() -> list[list[Any]]:
    with connect() as conn:
        counts = []
        for table_name in ["brands", "channels", "channel_brand_config", "orders", "channel_daily_metrics"]:
            count = conn.execute(f"select count(*) from {table_name}").fetchone()[0]
            counts.append([table_name, count])
    return counts


def get_table_preview(table_name: str) -> tuple[list[str], list[list[Any]]]:
    with connect() as conn:
        result = conn.execute(f"select * from {table_name} limit 12")
        columns = [item[0] for item in result.description]
        rows = result.fetchall()
    cleaned_rows = [[value.isoformat(sep=" ", timespec="minutes") if isinstance(value, datetime) else value for value in row] for row in rows]
    return columns, cleaned_rows


def get_table_sql(table_name: str) -> str:
    return TABLE_SQL_REFERENCE.get(table_name, "-- no SQL reference available")


def scenario_story_markdown(scenario_name: str) -> str:
    scenario = SCENARIOS[scenario_name]
    prompt = SCENARIO_QUESTIONS.get(scenario_name, "")
    return (
        f"### {scenario['headline']}\n"
        + ("This is the clean control state before we introduce any incident.\n\n" if scenario_name == "Baseline" else "")
        + f"{scenario['summary']}\n\n"
        f"**Try asking the assistant:** `{prompt}`"
    )


def render_scenario_catalog() -> str:
    groups = [
        ("Business / config reasoning", "business_config"),
        ("Pipeline / data failures", "pipeline_data"),
        ("Mixed / multi-cause", "mixed"),
    ]
    parts = ["<div class='scenario-catalog'>"]
    for title, kind in groups:
        parts.append(f"<div class='scenario-group'><h4>{title}</h4><div class='scenario-group-grid'>")
        for name, scenario in SCENARIOS.items():
            if scenario.get("kind") != kind:
                continue
            parts.append(
                "<div class='scenario-catalog-card'>"
                f"<div class='scenario-catalog-title'>{name}</div>"
                f"<div class='scenario-catalog-copy'>{scenario['summary']}</div>"
                "</div>"
            )
        parts.append("</div></div>")
    parts.append("</div>")
    return "".join(parts)


INCIDENT_REPORT_GUIDE_HTML = """
<div class='incident-guide'>
  <div class='incident-guide-card'>
    <div class='incident-guide-title'>business_config_drift</div>
    <div class='incident-guide-copy'>Use for promo flags, commission overrides, listing state, or menu sync issues.</div>
  </div>
  <div class='incident-guide-card'>
    <div class='incident-guide-title'>pipeline_failure</div>
    <div class='incident-guide-copy'>Use for stale aggregates, partial refreshes, duplicate ingest, or broken transforms.</div>
  </div>
  <div class='incident-guide-card'>
    <div class='incident-guide-title'>external_business_reality</div>
    <div class='incident-guide-copy'>Use when nothing is broken and the KPI move reflects real business change.</div>
  </div>
  <div class='incident-guide-card'>
    <div class='incident-guide-title'>compound</div>
    <div class='incident-guide-copy'>Use when more than one cause is true at the same time.</div>
  </div>
</div>
"""


def apply_stuck_promo(target_date: str) -> None:
    with connect() as conn:
        conn.execute(
            """
            update channel_brand_config
            set promo_active = true,
                promo_discount_pct = 18.0,
                notes = 'promo should have ended but stayed active'
            where brand_id = 'behrouz' and channel_id = 'swiggy'
            """
        )
        conn.execute(
            """
            update orders
            set discount_amount = round(gross_amount * 0.18, 2),
                delivered_amount = case when status = 'delivered' then round(gross_amount - round(gross_amount * 0.18, 2), 2) else 0 end
            where cast(ordered_at as date) = ?
              and brand_id = 'behrouz'
              and channel_id = 'swiggy'
            """,
            [target_date],
        )
        refresh_aggregation(conn, log_action=False)
    write_log("scenario", "Applied stuck promo scenario to Behrouz x Swiggy.", {"target_date": target_date})


def apply_pause_brand_channel(target_date: str) -> None:
    with connect() as conn:
        conn.execute(
            """
            update channel_brand_config
            set is_live = false,
                notes = 'listing paused by mistake'
            where brand_id = 'ovenstory' and channel_id = 'zomato'
            """
        )
        conn.execute(
            """
            delete from orders
            where cast(ordered_at as date) = ?
              and brand_id = 'ovenstory'
              and channel_id = 'zomato'
            """,
            [target_date],
        )
        refresh_aggregation(conn, log_action=False)
    write_log("scenario", "Paused Oven Story on Zomato and removed target-date orders.", {"target_date": target_date})


def apply_commission_drift(target_date: str) -> None:
    with connect() as conn:
        conn.execute(
            """
            update channel_brand_config
            set effective_commission_pct = 28.0,
                notes = 'commission revised upward'
            where channel_id = 'zomato'
            """
        )
        refresh_aggregation(conn, log_action=False)
    write_log("scenario", "Raised Zomato commission to show payout pressure with healthy topline.", {"target_date": target_date})


def apply_menu_sync_stale(target_date: str) -> None:
    with connect() as conn:
        conn.execute(
            """
            update channel_brand_config
            set menu_last_synced_at = current_timestamp - interval 4 day,
                notes = 'menu sync stale'
            where brand_id = 'faasos' and channel_id = 'swiggy'
            """
        )
        conn.execute(
            """
            update orders
            set gross_amount = round(gross_amount + 60, 2),
                delivered_amount = case when status = 'delivered' then round((gross_amount + 60) - discount_amount, 2) else 0 end
            where cast(ordered_at as date) = ?
              and brand_id = 'faasos'
              and channel_id = 'swiggy'
            """,
            [target_date],
        )
        refresh_aggregation(conn, log_action=False)
    write_log("scenario", "Made Faasos x Swiggy menu sync stale and shifted ticket size upward.", {"target_date": target_date})


def apply_compound_promo_sync(target_date: str) -> None:
    with connect() as conn:
        conn.execute(
            """
            update channel_brand_config
            set promo_active = true,
                promo_discount_pct = 20.0,
                menu_last_synced_at = current_timestamp - interval 5 day,
                effective_commission_pct = 24.5,
                notes = 'promo stayed on and menu sync is stale'
            where brand_id = 'behrouz' and channel_id = 'zomato'
            """
        )
        conn.execute(
            """
            update orders
            set gross_amount = round(gross_amount + 45, 2),
                discount_amount = round((gross_amount + 45) * 0.20, 2),
                delivered_amount = case when status = 'delivered' then round((gross_amount + 45) - round((gross_amount + 45) * 0.20, 2), 2) else 0 end
            where cast(ordered_at as date) = ?
              and brand_id = 'behrouz'
              and channel_id = 'zomato'
            """,
            [target_date],
        )
        refresh_aggregation(conn, log_action=False)
    write_log("scenario", "Applied compound promo + menu sync pressure to Behrouz x Zomato.", {"target_date": target_date})


def apply_aov_collapse(target_date: str) -> None:
    """Pure KPI scenario: orders rise but gross_sales stays flat because ticket size dropped.
    No config change, no pipeline failure. The data is correct; the business reality shifted."""
    with connect() as conn:
        # Add ~40% more orders for The Bowl Company on Swiggy at much lower ticket size
        conn.execute(
            """
            insert into orders
            select
              'extra_' || row_number() over () || '_' || ? as order_id,
              cast(? as timestamp) + (row_number() over () * interval 7 minute) as ordered_at,
              cast(? as timestamp) + (row_number() over () * interval 7 minute) + interval 25 minute as ingested_at,
              'thebowlcompany' as brand_id,
              'swiggy' as channel_id,
              round(150 + random() * 40, 2) as gross_amount,
              0.0 as discount_amount,
              round(150 + random() * 40, 2) as delivered_amount,
              0.0 as refund_amount,
              'delivered' as status,
              '9' || cast(floor(random() * 1000000000) as varchar) as customer_phone
            from range(0, 55)
            """,
            [target_date, target_date, target_date],
        )
        conn.execute(
            """
            update channel_brand_config
            set notes = 'new cheap SKU launched, AOV dropped by mix'
            where brand_id = 'thebowlcompany' and channel_id = 'swiggy'
            """
        )
        refresh_aggregation(conn, log_action=False)
    write_log("scenario", "Introduced cheap-SKU mix shift for The Bowl Company x Swiggy (AOV collapse).", {"target_date": target_date})


def apply_stealth_discount(target_date: str) -> None:
    """Pure KPI scenario: discount_amount rises because the config has a stale promo flag,
    but there is no current campaign in ops comms. No pipeline bug; this is config drift."""
    with connect() as conn:
        conn.execute(
            """
            update channel_brand_config
            set promo_active = true,
                promo_discount_pct = 12.0,
                notes = 'no active campaign in ops comms, but flag is on'
            where brand_id = 'sweettruth' and channel_id = 'zomato'
            """
        )
        conn.execute(
            """
            update orders
            set discount_amount = round(gross_amount * 0.12, 2),
                delivered_amount = case when status = 'delivered' then round(gross_amount - round(gross_amount * 0.12, 2), 2) else 0 end
            where cast(ordered_at as date) = ?
              and brand_id = 'sweettruth'
              and channel_id = 'zomato'
            """,
            [target_date],
        )
        refresh_aggregation(conn, log_action=False)
    write_log("scenario", "Enabled stealth discount flag on Sweet Truth x Zomato.", {"target_date": target_date})


def apply_stale_aggregate_after_swiggy_feed(target_date: str) -> None:
    with connect() as conn:
        conn.execute(
            """
            insert into orders
            select
              'RAWLATE-' || cast(row_number() over () as varchar) || '-' || ? as order_id,
              cast(? as timestamp) + (row_number() over () * interval 9 minute) as ordered_at,
              current_timestamp as ingested_at,
              'behrouz' as brand_id,
              'swiggy' as channel_id,
              round(480 + random() * 120, 2) as gross_amount,
              0.0 as discount_amount,
              round(480 + random() * 120, 2) as delivered_amount,
              0.0 as refund_amount,
              'delivered' as status,
              '9' || cast(floor(random() * 1000000000) as varchar) as customer_phone
            from range(0, 18)
            """,
            [target_date, target_date],
        )
        conn.execute(
            """
            update channel_brand_config
            set notes = 'raw orders landed after build; aggregate still stale'
            where brand_id = 'behrouz' and channel_id = 'swiggy'
            """
        )
    write_log("scenario", "Inserted fresh Behrouz x Swiggy raw orders without rebuilding the aggregate.", {"target_date": target_date})
    write_log("pipeline_failure", "Raw feed landed after the dashboard build; aggregation refresh did not run.", {"target_date": target_date, "scope": "Behrouz x Swiggy"})


def apply_partial_refresh_drop_zomato(target_date: str) -> None:
    with connect() as conn:
        refresh_aggregation(conn, log_action=False)
        conn.execute(
            """
            delete from channel_daily_metrics
            where metric_date = ?
              and channel_name = 'Zomato'
            """,
            [target_date],
        )
    write_log("scenario", "Dropped Zomato rows from the aggregate after a partial refresh.", {"target_date": target_date})
    write_log("pipeline_failure", "The refresh only completed part-way; Zomato partitions are missing from channel_daily_metrics.", {"target_date": target_date})


def apply_duplicate_ondc_events(target_date: str) -> None:
    with connect() as conn:
        conn.execute(
            """
            insert into orders
            select
              'DUP-' || order_id as order_id,
              ordered_at,
              ingested_at + interval 2 minute,
              brand_id,
              channel_id,
              gross_amount,
              discount_amount,
              delivered_amount,
              refund_amount,
              status,
              customer_phone
            from orders
            where cast(ordered_at as date) = ?
              and channel_id = 'ondc'
            limit 18
            """,
            [target_date],
        )
        refresh_aggregation(conn, log_action=False)
        conn.execute(
            """
            update channel_brand_config
            set notes = 'connector replayed ONDC events without dedupe'
            where channel_id = 'ondc'
            """
        )
    write_log("scenario", "Duplicated ONDC raw events and rebuilt the aggregate.", {"target_date": target_date})
    write_log("pipeline_failure", "ONDC replayed confirm events without a dedup step, inflating raw and aggregate counts.", {"target_date": target_date})


def apply_late_swiggy_feed_skew(target_date: str) -> None:
    with connect() as conn:
        conn.execute(
            """
            update orders
            set ingested_at = cast(? as timestamp) + interval 1 day + interval 6 hour
            where cast(ordered_at as date) = ?
              and channel_id = 'swiggy'
            """,
            [target_date, target_date],
        )
        refresh_aggregation(conn, log_action=False)
        conn.execute(
            """
            delete from channel_daily_metrics
            where metric_date = ?
              and channel_name = 'Swiggy'
            """,
            [target_date],
        )
        conn.execute(
            """
            update channel_brand_config
            set notes = 'feed landed after morning build'
            where channel_id = 'swiggy'
            """
        )
    write_log("scenario", "Shifted Swiggy ingest later than the build window and removed Swiggy rows from the aggregate.", {"target_date": target_date})
    write_log("pipeline_failure", "Swiggy feed arrived after the dashboard build, so the aggregate snapshot is incomplete.", {"target_date": target_date})


def apply_cancelled_orders_dropped_from_aggregate(target_date: str) -> None:
    with connect() as conn:
        refresh_aggregation(conn, log_action=False)
        conn.execute(
            """
            update channel_daily_metrics
            set total_orders = delivered_orders,
                cancelled_orders = 0,
                cancellation_rate = 0.0
            where metric_date = ?
              and brand_name = 'Faasos'
              and channel_name = 'Swiggy'
            """,
            [target_date],
        )
    write_log("scenario", "Tampered the aggregate so cancelled orders disappear for Faasos x Swiggy.", {"target_date": target_date})
    write_log("pipeline_failure", "A transform bug dropped cancelled orders from the aggregate output for one slice.", {"target_date": target_date, "scope": "Faasos x Swiggy"})


def apply_mixed_promo_stale_aggregate(target_date: str) -> None:
    with connect() as conn:
        conn.execute(
            """
            update channel_brand_config
            set promo_active = true,
                promo_discount_pct = 18.0,
                notes = 'promo stuck on; aggregate still stale'
            where brand_id = 'behrouz' and channel_id = 'swiggy'
            """
        )
        conn.execute(
            """
            update orders
            set discount_amount = round(gross_amount * 0.18, 2),
                delivered_amount = case when status = 'delivered' then round(gross_amount - round(gross_amount * 0.18, 2), 2) else 0 end
            where cast(ordered_at as date) = ?
              and brand_id = 'behrouz'
              and channel_id = 'swiggy'
            """,
            [target_date],
        )
    write_log("scenario", "Applied stuck promo to Behrouz x Swiggy but left the aggregate stale.", {"target_date": target_date})
    write_log("pipeline_failure", "Config and raw changed, but aggregation did not rerun, so the dashboard is stale too.", {"target_date": target_date})


def apply_mixed_pause_stale_aggregate(target_date: str) -> None:
    with connect() as conn:
        conn.execute(
            """
            update channel_brand_config
            set is_live = false,
                notes = 'listing paused but aggregate not refreshed'
            where brand_id = 'ovenstory' and channel_id = 'zomato'
            """
        )
        conn.execute(
            """
            delete from orders
            where cast(ordered_at as date) = ?
              and brand_id = 'ovenstory'
              and channel_id = 'zomato'
            """,
            [target_date],
        )
    write_log("scenario", "Paused Oven Story x Zomato in config and raw, but left the aggregate unchanged.", {"target_date": target_date})
    write_log("pipeline_failure", "The listing is truly paused, but the aggregate is stale and still shows yesterday's sales.", {"target_date": target_date})


SCENARIO_APPLIERS = {
    "stuck_promo": apply_stuck_promo,
    "pause_brand_channel": apply_pause_brand_channel,
    "commission_drift": apply_commission_drift,
    "menu_sync_stale": apply_menu_sync_stale,
    "compound_promo_sync": apply_compound_promo_sync,
    "aov_collapse": apply_aov_collapse,
    "stealth_discount": apply_stealth_discount,
    "stale_aggregate_after_swiggy_feed": apply_stale_aggregate_after_swiggy_feed,
    "partial_refresh_drop_zomato": apply_partial_refresh_drop_zomato,
    "duplicate_ondc_events": apply_duplicate_ondc_events,
    "late_swiggy_feed_skew": apply_late_swiggy_feed_skew,
    "cancelled_orders_dropped_from_aggregate": apply_cancelled_orders_dropped_from_aggregate,
    "mixed_promo_stale_aggregate": apply_mixed_promo_stale_aggregate,
    "mixed_pause_stale_aggregate": apply_mixed_pause_stale_aggregate,
}


# ----------------------------------------------------------------------
# REWARD / BENCHMARK SCORING
# ----------------------------------------------------------------------
# Each scenario has a hidden ground truth. When the assistant (or a trained
# agent) calls submit_incident_report at the end of an episode, we compare
# its report against this truth and compute a reward.
#
# The reward is bounded in [-0.5, 1.0] and broken into sub-scores so we can
# see WHY the reward changed during training. A dict of sub-scores gets
# logged with every submission.

VALID_KPIS = set(KPI_DEFINITIONS.keys())

# Canonical root-cause categories the agent is expected to pick from.
ROOT_CAUSE_CATEGORIES = [
    "business_config_drift",     # promo / commission / sync flag issues
    "pipeline_failure",          # stale aggregate, failed refresh, schema drift
    "external_business_reality", # mix shift, seasonality - not a bug
    "compound",                  # multiple true causes
]

SCENARIO_GROUND_TRUTH = {
    "Baseline": {
        "true_root_cause": "external_business_reality",  # "nothing broken"
        "true_affected_kpis": set(),
        "relevant_tools": {"get_metric_snapshot", "list_kpis"},
        "is_null_incident": True,
    },
    "Stuck Promo After Campaign End": {
        "true_root_cause": "business_config_drift",
        "true_affected_kpis": {"discount_amount", "net_sales", "net_payout"},
        "relevant_tools": {"get_brand_channel_config", "get_metric_snapshot", "inspect_pipeline_logs"},
        "is_null_incident": False,
    },
    "Brand Paused On One Channel": {
        "true_root_cause": "business_config_drift",
        "true_affected_kpis": {"total_orders", "gross_sales", "net_sales"},
        "relevant_tools": {"get_brand_channel_config", "get_metric_snapshot", "run_sql"},
        "is_null_incident": False,
    },
    "Commission Drift Hiding Margin Pressure": {
        "true_root_cause": "external_business_reality",
        "true_affected_kpis": {"net_payout"},
        "relevant_tools": {"get_brand_channel_config", "get_metric_snapshot", "get_kpi_relations"},
        "is_null_incident": True,
    },
    "Menu Sync Staleness": {
        "true_root_cause": "business_config_drift",
        "true_affected_kpis": {"aov", "gross_sales"},
        "relevant_tools": {"get_brand_channel_config", "get_metric_snapshot"},
        "is_null_incident": False,
    },
    "Orders Up, Net Sales Flat (AOV Collapse)": {
        "true_root_cause": "external_business_reality",
        "true_affected_kpis": {"aov", "total_orders"},
        "relevant_tools": {"get_metric_snapshot", "run_sql", "get_kpi_relations"},
        "is_null_incident": True,
    },
    "Discount Spike Without Promo Campaign": {
        "true_root_cause": "business_config_drift",
        "true_affected_kpis": {"discount_amount", "net_sales"},
        "relevant_tools": {"get_brand_channel_config", "get_metric_snapshot"},
        "is_null_incident": False,
    },
    "Stale Aggregate After Swiggy Feed": {
        "true_root_cause": "pipeline_failure",
        "true_affected_kpis": {"total_orders", "gross_sales", "net_sales", "net_payout"},
        "relevant_tools": {"inspect_pipeline_logs", "get_table_freshness", "run_sql"},
        "is_null_incident": False,
    },
    "Partial Aggregate Refresh (Zomato Missing)": {
        "true_root_cause": "pipeline_failure",
        "true_affected_kpis": {"total_orders", "gross_sales", "net_sales"},
        "relevant_tools": {"inspect_pipeline_logs", "run_sql", "get_metric_snapshot"},
        "is_null_incident": False,
    },
    "Duplicate ONDC Events": {
        "true_root_cause": "pipeline_failure",
        "true_affected_kpis": {"total_orders", "gross_sales", "net_sales", "net_payout"},
        "relevant_tools": {"inspect_pipeline_logs", "run_sql", "get_metric_snapshot"},
        "is_null_incident": False,
    },
    "Late Swiggy Feed Skew": {
        "true_root_cause": "pipeline_failure",
        "true_affected_kpis": {"total_orders", "gross_sales", "net_sales"},
        "relevant_tools": {"inspect_pipeline_logs", "get_table_freshness", "run_sql"},
        "is_null_incident": False,
    },
    "Cancelled Orders Missing From Aggregate": {
        "true_root_cause": "pipeline_failure",
        "true_affected_kpis": {"total_orders", "cancellation_rate"},
        "relevant_tools": {"inspect_pipeline_logs", "run_sql", "get_metric_snapshot"},
        "is_null_incident": False,
    },
    "Mixed: Promo + Stale Aggregate": {
        "true_root_cause": "compound",
        "true_affected_kpis": {"discount_amount", "net_sales", "net_payout"},
        "relevant_tools": {"get_brand_channel_config", "inspect_pipeline_logs", "get_table_freshness"},
        "is_null_incident": False,
    },
    "Mixed: Listing Pause + Stale Aggregate": {
        "true_root_cause": "compound",
        "true_affected_kpis": {"total_orders", "gross_sales", "net_sales"},
        "relevant_tools": {"get_brand_channel_config", "inspect_pipeline_logs", "get_table_freshness"},
        "is_null_incident": False,
    },
    "Compound: Promo + Menu Sync": {
        "true_root_cause": "compound",
        "true_affected_kpis": {"discount_amount", "net_sales", "net_payout", "aov"},
        "relevant_tools": {"get_brand_channel_config", "get_metric_snapshot", "inspect_pipeline_logs"},
        "is_null_incident": False,
    },
}

# Tools the agent might call that count as "inspection" (vs "mutation")
INSPECTION_TOOLS = {
    "list_kpis", "get_kpi_relations", "get_metric_snapshot",
    "get_brand_channel_config", "get_table_freshness",
    "inspect_pipeline_logs", "run_sql",
}
MUTATION_TOOLS = {"refresh_aggregation"}

REWARD_WEIGHTS = {
    "root_cause":     0.50,
    "tool_coverage":  0.20,
    "tool_order":     0.10,
    "grounded_kpis":  0.10,
    "not_premature":  0.10,
}
REWARD_PENALTIES = {
    "hallucination":  0.30,
    "premature":      0.20,
}


def compute_scenario_reward(
    trace: list[dict[str, Any]],
    report: dict[str, Any] | None,
    scenario_name: str,
) -> dict[str, Any]:
    """Score an episode. Returns a dict of sub-scores plus a total.

    Inputs:
      trace: list of tool-call records (same format as the UI trace)
      report: the submitted incident report ({root_cause, affected_kpis, ...})
              or None if the agent never submitted
      scenario_name: which scenario the episode was on

    Output dict keys: root_cause, tool_coverage, tool_order, grounded_kpis,
    not_premature, hallucination_pen, premature_pen, total, explanation.
    """
    truth = SCENARIO_GROUND_TRUTH.get(scenario_name)
    if truth is None:
        return {"total": 0.0, "explanation": "No ground truth for this scenario."}

    tool_names = [ev.get("tool") for ev in trace if ev.get("tool")]

    # 1) Root-cause correctness
    r_root = 0.0
    if report:
        if report.get("root_cause_category") == truth["true_root_cause"]:
            r_root = 1.0
        elif truth["true_root_cause"] == "compound":
            # partial credit if agent named one of the compound's true sub-causes
            if report.get("root_cause_category") in {"business_config_drift", "pipeline_failure"}:
                r_root = 0.5

    # 2) Tool coverage — did they call the relevant tools for this scenario?
    relevant = truth["relevant_tools"]
    if relevant:
        used = set(tool_names) & relevant
        r_coverage = len(used) / len(relevant)
    else:
        r_coverage = 1.0

    # 3) Tool order — inspected before submitting / acting?
    r_order = 0.0
    submit_idx = next((i for i, t in enumerate(tool_names) if t == "submit_incident_report"), None)
    if submit_idx is not None:
        inspected_before = any(t in INSPECTION_TOOLS for t in tool_names[:submit_idx])
        if inspected_before:
            r_order += 0.5
    else:
        if any(t in INSPECTION_TOOLS for t in tool_names):
            r_order += 0.25
    first_mutation = next((i for i, t in enumerate(tool_names) if t in MUTATION_TOOLS), None)
    if first_mutation is None:
        r_order += 0.5
    else:
        if any(t in INSPECTION_TOOLS for t in tool_names[:first_mutation]):
            r_order += 0.5

    # 4) Grounded KPIs — did they reference only real KPIs?
    r_grounded = 0.5  # cautious default if no KPIs referenced
    p_halluc = 0.0
    if report and report.get("affected_kpis"):
        referenced = set(report["affected_kpis"])
        if referenced <= VALID_KPIS:
            r_grounded = 1.0
        else:
            fake = referenced - VALID_KPIS
            frac_fake = len(fake) / len(referenced)
            r_grounded = 0.0
            p_halluc = REWARD_PENALTIES["hallucination"] * frac_fake

    # 5) Not premature — did they inspect before submitting?
    r_not_prem = 0.0
    p_premature = 0.0
    if submit_idx is not None:
        inspected = any(t in INSPECTION_TOOLS for t in tool_names[:submit_idx])
        if inspected:
            r_not_prem = 1.0
        if submit_idx == 0 or not tool_names[:submit_idx]:
            p_premature = REWARD_PENALTIES["premature"]

    positive = (
        REWARD_WEIGHTS["root_cause"]    * r_root +
        REWARD_WEIGHTS["tool_coverage"] * r_coverage +
        REWARD_WEIGHTS["tool_order"]    * r_order +
        REWARD_WEIGHTS["grounded_kpis"] * r_grounded +
        REWARD_WEIGHTS["not_premature"] * r_not_prem
    )
    total = positive - p_halluc - p_premature

    # Plain-English explanation
    lines = []
    if report is None:
        lines.append("No incident report submitted.")
    else:
        if r_root == 1.0:
            lines.append(f"✓ Correctly identified root cause as **{truth['true_root_cause']}**.")
        elif r_root > 0:
            lines.append(f"~ Partially correct root cause (compound scenario — got one sub-cause).")
        else:
            lines.append(
                f"✗ Reported **{report.get('root_cause_category')}** but truth is "
                f"**{truth['true_root_cause']}**."
            )
        if r_grounded == 1.0:
            lines.append("✓ All referenced KPIs are valid.")
        elif p_halluc > 0:
            fake = set(report.get("affected_kpis", [])) - VALID_KPIS
            lines.append(f"✗ Hallucinated KPIs: {', '.join(sorted(fake))}")
    lines.append(f"Tool coverage: {r_coverage:.0%} of relevant tools used.")
    lines.append(f"Tool order: {r_order:.0%} (inspected before concluding/acting).")
    if p_premature > 0:
        lines.append("✗ Submitted too early without inspecting.")

    return {
        "total": round(total, 3),
        "root_cause": round(r_root, 3),
        "tool_coverage": round(r_coverage, 3),
        "tool_order": round(r_order, 3),
        "grounded_kpis": round(r_grounded, 3),
        "not_premature": round(r_not_prem, 3),
        "hallucination_pen": round(p_halluc, 3),
        "premature_pen": round(p_premature, 3),
        "explanation": "\n".join(lines),
        "scenario_name": scenario_name,
        "truth_root_cause": truth["true_root_cause"],
    }


def render_reward_card(reward: dict[str, Any]) -> str:
    """Render a reward breakdown as a scorecard HTML block."""
    total = reward.get("total", 0.0)
    if total >= 0.8:
        color, label = "var(--signal-ok)", "STRONG"
    elif total >= 0.5:
        color, label = "var(--signal-warn)", "PARTIAL"
    elif total >= 0.0:
        color, label = "var(--text-muted)", "WEAK"
    else:
        color, label = "var(--signal-bad)", "INCORRECT"

    explanation = reward.get("explanation", "").replace("\n", "<br/>")
    rows = [
        ("Root cause", reward.get("root_cause", 0), 0.50),
        ("Tool coverage", reward.get("tool_coverage", 0), 0.20),
        ("Tool order", reward.get("tool_order", 0), 0.10),
        ("Grounded KPIs", reward.get("grounded_kpis", 0), 0.10),
        ("Not premature", reward.get("not_premature", 0), 0.10),
    ]
    row_html = "".join(
        f"<tr><td>{name}</td>"
        f"<td style='text-align:right;color:var(--text-secondary);'>{score:.2f}</td>"
        f"<td style='text-align:right;color:var(--text-muted);'>× {weight:.2f}</td>"
        f"<td style='text-align:right;font-weight:600;'>{score * weight:.3f}</td></tr>"
        for name, score, weight in rows
    )
    penalty_html = ""
    p_halluc = reward.get("hallucination_pen", 0)
    p_prem = reward.get("premature_pen", 0)
    if p_halluc > 0:
        penalty_html += f"<tr style='color:var(--signal-bad);'><td colspan='3'>Hallucination penalty</td><td style='text-align:right;'>− {p_halluc:.3f}</td></tr>"
    if p_prem > 0:
        penalty_html += f"<tr style='color:var(--signal-bad);'><td colspan='3'>Premature submission</td><td style='text-align:right;'>− {p_prem:.3f}</td></tr>"

    return f"""
<div class='reward-card'>
  <div class='reward-header'>
    <span class='reward-title'>Episode reward</span>
    <span class='reward-badge' style='color:{color}'>{label}</span>
    <span class='reward-total' style='color:{color}'>{total:+.3f}</span>
  </div>
  <table class='reward-table'>
    <thead><tr><th>Component</th><th>Score</th><th>Weight</th><th>Contribution</th></tr></thead>
    <tbody>{row_html}{penalty_html}</tbody>
  </table>
  <div class='reward-explain'>{explanation}</div>
</div>
"""


# --- Scenario diff capture ---------------------------------------------------
# Each scenario function ("applier") declares which tables it touches. We snapshot
# those tables before applying, then after, and surface the diff in the UI.
# The Scenarios tab becomes a "what changed in the system" report rather than
# requiring the user to go hunt through Debug.

SCENARIO_TOUCHED_TABLES = {
    "stuck_promo":         ["channel_brand_config", "orders", "channel_daily_metrics"],
    "pause_brand_channel": ["channel_brand_config", "orders", "channel_daily_metrics"],
    "commission_drift":    ["channel_brand_config", "channel_daily_metrics"],
    "menu_sync_stale":     ["channel_brand_config", "orders", "channel_daily_metrics"],
    "compound_promo_sync": ["channel_brand_config", "orders", "channel_daily_metrics"],
    "aov_collapse":        ["channel_brand_config", "orders", "channel_daily_metrics"],
    "stealth_discount":    ["channel_brand_config", "orders", "channel_daily_metrics"],
    "stale_aggregate_after_swiggy_feed": ["channel_brand_config", "orders", "channel_daily_metrics"],
    "partial_refresh_drop_zomato": ["channel_daily_metrics"],
    "duplicate_ondc_events": ["channel_brand_config", "orders", "channel_daily_metrics"],
    "late_swiggy_feed_skew": ["channel_brand_config", "orders", "channel_daily_metrics"],
    "cancelled_orders_dropped_from_aggregate": ["channel_daily_metrics"],
    "mixed_promo_stale_aggregate": ["channel_brand_config", "orders", "channel_daily_metrics"],
    "mixed_pause_stale_aggregate": ["channel_brand_config", "orders", "channel_daily_metrics"],
}


def _snapshot_table(conn: duckdb.DuckDBPyConnection, table: str) -> tuple[list[str], list[tuple]]:
    """Fetch (columns, rows) for a table. Used to capture before/after state."""
    result = conn.execute(f"select * from {table}").fetchall()
    columns = [d[0] for d in conn.description]
    return columns, [tuple(r) for r in result]


def snapshot_tables(tables: list[str]) -> dict[str, tuple[list[str], list[tuple]]]:
    """Capture current state of a set of tables. Keyed by table name."""
    out: dict[str, tuple[list[str], list[tuple]]] = {}
    with connect() as conn:
        for table in tables:
            try:
                out[table] = _snapshot_table(conn, table)
            except Exception:
                out[table] = ([], [])
    return out


def diff_snapshots(
    before: dict[str, tuple[list[str], list[tuple]]],
    after: dict[str, tuple[list[str], list[tuple]]],
) -> dict[str, dict[str, Any]]:
    """For each table, return added, removed, and changed rows.
    We key rows by a primary-ish identifier per table, falling back to
    full-row hash for tables without one."""
    keyers = {
        "channel_brand_config": lambda r, cols: (r[cols.index("brand_id")], r[cols.index("channel_id")]),
        "orders": lambda r, cols: (r[cols.index("order_id")],),
        "channel_daily_metrics": lambda r, cols: (
            str(r[cols.index("metric_date")]),
            r[cols.index("brand_name")],
            r[cols.index("channel_name")],
        ),
    }
    result: dict[str, dict[str, Any]] = {}
    for table in before.keys() | after.keys():
        cols_before, rows_before = before.get(table, ([], []))
        cols_after, rows_after = after.get(table, ([], []))
        cols = cols_after or cols_before
        if not cols:
            result[table] = {"columns": [], "added": [], "removed": [], "changed": [], "total_before": 0, "total_after": 0}
            continue
        key_fn = keyers.get(table, lambda r, c: tuple(r))
        try:
            by_key_before = {key_fn(r, cols_before): r for r in rows_before}
            by_key_after = {key_fn(r, cols_after): r for r in rows_after}
        except Exception:
            by_key_before = {tuple(r): r for r in rows_before}
            by_key_after = {tuple(r): r for r in rows_after}
        added_keys = set(by_key_after) - set(by_key_before)
        removed_keys = set(by_key_before) - set(by_key_after)
        changed = []
        for k in set(by_key_before) & set(by_key_after):
            if by_key_before[k] != by_key_after[k]:
                changed.append({"key": k, "before": by_key_before[k], "after": by_key_after[k]})
        result[table] = {
            "columns": cols,
            "added": [by_key_after[k] for k in added_keys],
            "removed": [by_key_before[k] for k in removed_keys],
            "changed": changed,
            "total_before": len(rows_before),
            "total_after": len(rows_after),
        }
    return result


def get_pipeline_state_snapshot() -> dict[str, Any]:
    """Snapshot pipeline health: freshness, last refresh time, row counts per table.
    Always shown on Scenarios tab alongside the data diff."""
    state: dict[str, Any] = {"tables": [], "freshness": {}, "last_refresh": None}
    with connect() as conn:
        for table in ["brands", "channels", "channel_brand_config", "orders", "channel_daily_metrics"]:
            try:
                count = conn.execute(f"select count(*) from {table}").fetchone()[0]
                state["tables"].append({"table": table, "rows": int(count)})
            except Exception:
                state["tables"].append({"table": table, "rows": 0})
        try:
            raw_max = conn.execute("select max(ingested_at) from orders").fetchone()[0]
            agg_max = conn.execute("select max(latest_ingested_at) from channel_daily_metrics").fetchone()[0]
            state["freshness"]["orders_latest_ingest"] = raw_max.strftime("%Y-%m-%d %H:%M") if raw_max else "—"
            state["freshness"]["aggregate_latest_ingest"] = agg_max.strftime("%Y-%m-%d %H:%M") if agg_max else "—"
            if raw_max and agg_max:
                lag_seconds = (raw_max - agg_max).total_seconds() if raw_max > agg_max else 0
                state["freshness"]["aggregate_lag_minutes"] = round(lag_seconds / 60, 1)
                state["freshness"]["is_stale"] = lag_seconds > 60
            else:
                state["freshness"]["aggregate_lag_minutes"] = 0
                state["freshness"]["is_stale"] = False
        except Exception:
            pass
    try:
        if LOG_PATH.exists():
            with LOG_PATH.open() as fh:
                entries = json.load(fh)
            refresh_entries = [e for e in entries if e.get("action") == "refresh_aggregation"]
            if refresh_entries:
                state["last_refresh"] = refresh_entries[-1].get("timestamp")
    except Exception:
        pass
    return state


def run_scenario_with_diff(scenario_name: str, seed: int) -> dict[str, Any]:
    """Apply a scenario and return a diff report:
      - which tables were touched
      - before/after row snapshots for changed rows only
      - pipeline state (freshness, row counts, last refresh)
      - summary action log of what the scenario did
    This is what the Scenarios tab renders.
    """
    # Always reset to clean baseline so diffs are meaningful
    initialize_database(seed)
    target_date = get_default_focus_date()

    steps = SCENARIOS[scenario_name]["apply"]
    touched: set[str] = set()
    for step in steps:
        touched.update(SCENARIO_TOUCHED_TABLES.get(step, []))
    touched_list = sorted(touched)

    before = snapshot_tables(touched_list) if touched_list else {}
    for step in steps:
        SCENARIO_APPLIERS[step](target_date)
    after = snapshot_tables(touched_list) if touched_list else {}
    diff = diff_snapshots(before, after) if touched_list else {}

    return {
        "scenario_name": scenario_name,
        "target_date": target_date,
        "steps": steps,
        "touched_tables": touched_list,
        "diff": diff,
        "pipeline_state": get_pipeline_state_snapshot(),
    }


# --- HTML renderers for the Scenarios tab -----------------------------------

def _format_cell(value: Any) -> str:
    """Render a cell value safely. Shows timestamps short, booleans pretty,
    floats rounded, Nones dimmed."""
    if value is None:
        return "<span style='color:var(--text-muted)'>NULL</span>"
    if isinstance(value, bool):
        color = "var(--signal-ok)" if value else "var(--signal-bad)"
        return f"<span style='color:{color};font-weight:600'>{value}</span>"
    if isinstance(value, float):
        return f"{value:,.2f}"
    s = str(value)
    if len(s) > 60:
        s = s[:57] + "..."
    return s


def _render_row_table(columns: list[str], rows: list[tuple], max_rows: int = 20) -> str:
    """Render rows as a compact HTML table."""
    if not rows:
        return "<div style='color:var(--text-muted);padding:8px;'>no rows</div>"
    head = "".join(f"<th>{c}</th>" for c in columns)
    body_rows = []
    for r in rows[:max_rows]:
        body_rows.append("<tr>" + "".join(f"<td>{_format_cell(v)}</td>" for v in r) + "</tr>")
    overflow = f"<div class='diff-overflow'>... and {len(rows) - max_rows} more rows</div>" if len(rows) > max_rows else ""
    return (
        "<table class='diff-table'>"
        f"<thead><tr>{head}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody>"
        f"</table>{overflow}"
    )


def _render_changed_rows(columns: list[str], changed: list[dict]) -> str:
    """For changed rows, show which columns actually changed, side by side."""
    if not changed:
        return "<div style='color:var(--text-muted);padding:8px;'>no changed rows</div>"
    pieces = []
    for item in changed[:10]:
        before_row = item["before"]
        after_row = item["after"]
        changed_cols = [i for i in range(len(columns)) if before_row[i] != after_row[i]]
        if not changed_cols:
            continue
        # Show a short "BEFORE → AFTER" line for each changed cell
        key_label = " / ".join(str(k) for k in item["key"])
        lines = [f"<div class='diff-key'>{key_label}</div>"]
        for idx in changed_cols:
            col = columns[idx]
            lines.append(
                f"<div class='diff-line'>"
                f"<span class='diff-col'>{col}</span>"
                f"<span class='diff-before'>{_format_cell(before_row[idx])}</span>"
                f"<span class='diff-arrow'>→</span>"
                f"<span class='diff-after'>{_format_cell(after_row[idx])}</span>"
                f"</div>"
            )
        pieces.append("<div class='diff-change-card'>" + "".join(lines) + "</div>")
    overflow = f"<div class='diff-overflow'>... and {len(changed) - 10} more changed rows</div>" if len(changed) > 10 else ""
    return "".join(pieces) + overflow


def render_scenario_diff(report: dict[str, Any]) -> str:
    """Render the scenario diff as a single HTML block for the Scenarios tab."""
    if not report.get("touched_tables"):
        return (
            "<div class='diff-empty'>"
            "<strong>Baseline loaded — no scenario perturbations applied.</strong>"
            "<p>Pick a scenario above to see exactly what changes in the system.</p>"
            "</div>"
        )

    parts = [
        f"<div class='diff-header'>"
        f"<strong>Scenario applied:</strong> {report['scenario_name']} "
        f"<span class='diff-sub'>(target date: {report['target_date']})</span>"
        f"</div>"
    ]

    for table in report["touched_tables"]:
        d = report["diff"].get(table, {})
        cols = d.get("columns", [])
        added, removed, changed = d.get("added", []), d.get("removed", []), d.get("changed", [])
        total_before = d.get("total_before", 0)
        total_after = d.get("total_after", 0)

        badge_parts = []
        if changed:
            badge_parts.append(f"<span class='diff-badge badge-changed'>{len(changed)} changed</span>")
        if added:
            badge_parts.append(f"<span class='diff-badge badge-added'>+{len(added)} added</span>")
        if removed:
            badge_parts.append(f"<span class='diff-badge badge-removed'>−{len(removed)} removed</span>")
        if not badge_parts:
            badge_parts.append("<span class='diff-badge badge-none'>no changes</span>")
        badges = " ".join(badge_parts)

        parts.append(f"""
<div class='diff-table-card'>
  <div class='diff-table-header'>
    <span class='diff-table-name'>{table}</span>
    <span class='diff-table-counts'>{total_before} → {total_after} rows</span>
    <span class='diff-badges'>{badges}</span>
  </div>
""")

        if changed:
            parts.append("<div class='diff-section-label'>Changed rows (before → after)</div>")
            parts.append(_render_changed_rows(cols, changed))

        if added:
            parts.append("<div class='diff-section-label'>Added rows</div>")
            parts.append(_render_row_table(cols, added, max_rows=10))

        if removed:
            parts.append("<div class='diff-section-label'>Removed rows</div>")
            parts.append(_render_row_table(cols, removed, max_rows=10))

        parts.append("</div>")

    return "".join(parts)


def render_storybook_acts(scenario_name: str, report: dict[str, Any]) -> str:
    """Render the three-act storybook for a scenario:
      Act 1 — what the user was looking at
      Act 2 — what got injected (with concrete counts)
      Act 3 — next step (ask the assistant on the Dashboard tab)
    Returns a single HTML block.
    """
    book = SCENARIO_STORYBOOK.get(scenario_name, SCENARIO_STORYBOOK["Baseline"])
    suggested_question = SCENARIO_QUESTIONS.get(scenario_name, "")

    # Build a concrete "here is what actually changed" line from the diff report
    if scenario_name == "Baseline" or not report.get("touched_tables"):
        change_summary = "Baseline re-seeded. All tables are at their clean starting state."
    else:
        pieces = []
        for t in report["touched_tables"]:
            d = report["diff"].get(t, {})
            bits = []
            if d.get("changed"):
                bits.append(f"{len(d['changed'])} changed")
            if d.get("added"):
                bits.append(f"+{len(d['added'])} added")
            if d.get("removed"):
                bits.append(f"-{len(d['removed'])} removed")
            if bits:
                pieces.append(f"<code>{t}</code>: {', '.join(bits)} rows")
        change_summary = " · ".join(pieces) if pieces else "no row-level changes detected"

    return f"""
<div class='storybook'>
  <div class='story-act'>
    <div class='story-act-head'>
      <span class='story-num'>1</span>
      <span class='story-title'>What the user was looking at</span>
    </div>
    <div class='story-body'>
      <div class='story-persona'>{book['persona']}</div>
      <p>{book['before']}</p>
    </div>
  </div>

  <div class='story-arrow'>↓</div>

  <div class='story-act story-act-injection'>
    <div class='story-act-head'>
      <span class='story-num'>2</span>
      <span class='story-title'>What got injected into the data world</span>
    </div>
    <div class='story-body'>
      <p>{book['injection_summary']}</p>
      <div class='story-changes'>{change_summary}</div>
    </div>
  </div>

  <div class='story-arrow'>↓</div>

  <div class='story-act story-act-next'>
    <div class='story-act-head'>
      <span class='story-num'>3</span>
      <span class='story-title'>What to do next</span>
    </div>
    <div class='story-body'>
      <p>{book['what_to_ask']}</p>
      {f"<div class='story-suggested'><span class='story-suggested-label'>Suggested question</span><div class='story-suggested-text'>“{suggested_question}”</div></div>" if suggested_question else ""}
      <div class='story-cta'>Now switch to the <strong>Dashboard</strong> tab and ask the assistant.</div>
    </div>
  </div>
</div>
"""


PIPELINE_DAG_NODES = [
    {"id": "app",        "label": "App",              "sub": "orders stream",    "x": 60,  "table": None},
    {"id": "raw",        "label": "Raw table",        "sub": "orders",           "x": 240, "table": "orders"},
    {"id": "aggregate",  "label": "Aggregated",       "sub": "channel_daily_metrics", "x": 440, "table": "channel_daily_metrics"},
    {"id": "dashboard",  "label": "Dashboard",        "sub": "KPIs",             "x": 640, "table": None},
]
PIPELINE_DAG_SIDECARS = [
    {"id": "config",     "label": "Config tables",    "sub": "brands, channels,\nchannel_brand_config", "x": 340, "table": "channel_brand_config"},
]


# Map each scenario step ("applier") to which DAG node(s) or edge(s) it breaks.
# This is what makes the DAG light up after a scenario is applied. Notes are the
# short caption shown under the DAG when that node/edge is broken.
SCENARIO_DAG_IMPACT = {
    "stuck_promo": {
        "broken_nodes": {"config": "promo_active flag stuck on"},
        "broken_edges": {},
    },
    "pause_brand_channel": {
        "broken_nodes": {"config": "is_live flipped off"},
        "broken_edges": {"app->raw": "no new orders for that pair"},
    },
    "commission_drift": {
        "broken_nodes": {"config": "commission_pct changed"},
        "broken_edges": {},
    },
    "menu_sync_stale": {
        "broken_nodes": {"config": "menu_last_synced_at behind"},
        "broken_edges": {},
    },
    "compound_promo_sync": {
        "broken_nodes": {"config": "promo stuck on AND menu sync stale"},
        "broken_edges": {},
    },
    "aov_collapse": {
        "broken_nodes": {},
        "broken_edges": {"app->raw": "cheap SKU mix shift (not a bug)"},
    },
    "stealth_discount": {
        "broken_nodes": {"config": "promo_active silently flipped on"},
        "broken_edges": {},
    },
    "stale_aggregate_after_swiggy_feed": {
        "broken_nodes": {"aggregate": "raw is ahead of aggregate"},
        "broken_edges": {"raw->agg": "refresh skipped after raw ingest"},
    },
    "partial_refresh_drop_zomato": {
        "broken_nodes": {"aggregate": "partial refresh dropped Zomato"},
        "broken_edges": {"raw->agg": "aggregate missing one partition"},
    },
    "duplicate_ondc_events": {
        "broken_nodes": {"raw": "duplicate ONDC events present"},
        "broken_edges": {"app->raw": "connector replayed events", "raw->agg": "no dedupe before rollup"},
    },
    "late_swiggy_feed_skew": {
        "broken_nodes": {"raw": "late feed landed after build", "aggregate": "morning snapshot incomplete"},
        "broken_edges": {"app->raw": "vendor feed delay", "raw->agg": "build happened too early"},
    },
    "cancelled_orders_dropped_from_aggregate": {
        "broken_nodes": {"aggregate": "cancelled orders filtered out"},
        "broken_edges": {"raw->agg": "status transform bug"},
    },
    "mixed_promo_stale_aggregate": {
        "broken_nodes": {"config": "promo flag stuck on", "aggregate": "dashboard still stale"},
        "broken_edges": {"raw->agg": "refresh skipped after raw/config change"},
    },
    "mixed_pause_stale_aggregate": {
        "broken_nodes": {"config": "listing paused", "aggregate": "old sales still visible"},
        "broken_edges": {"raw->agg": "refresh skipped after pause"},
    },
}


def compute_dag_state(scenario_name: str) -> dict[str, Any]:
    """Given the current scenario, return the DAG state: which nodes are broken,
    which edges are broken, and the captions to show.

    For baseline, everything is healthy. For a scenario, we union the impacts
    of all its applier steps."""
    state = {
        "broken_nodes": {},   # node_id -> caption
        "broken_edges": {},   # edge_id -> caption  (edges are "source->target")
        "active_nodes": set(), # for live pulse hook points (populated by LLM traversal)
    }
    scenario = SCENARIOS.get(scenario_name, {})
    for step in scenario.get("apply", []):
        impact = SCENARIO_DAG_IMPACT.get(step, {})
        for nid, caption in impact.get("broken_nodes", {}).items():
            if nid in state["broken_nodes"]:
                state["broken_nodes"][nid] += f"; {caption}"
            else:
                state["broken_nodes"][nid] = caption
        for eid, caption in impact.get("broken_edges", {}).items():
            if eid in state["broken_edges"]:
                state["broken_edges"][eid] += f"; {caption}"
            else:
                state["broken_edges"][eid] = caption
    return state


def _dag_node_svg(node: dict[str, Any], is_broken: bool, is_active: bool, y: int) -> str:
    """Render one DAG node as SVG."""
    x = node["x"]
    fill = "#2a1818" if is_broken else "#1A1D26"
    stroke = "#F87171" if is_broken else ("#FF6B35" if is_active else "#2E3440")
    stroke_width = 2 if (is_broken or is_active) else 1
    label_color = "#F87171" if is_broken else "#F0F2F7"
    sub_color = "#A8AEBF"
    width, height = 140, 62
    node_x = x - width // 2

    pulse = ""
    if is_active:
        pulse = f"""<animate attributeName='stroke-opacity' values='1;0.4;1' dur='1.4s' repeatCount='indefinite' />"""

    sub_lines = node["sub"].split("\n")
    sub_tspans = "".join(
        f"<tspan x='{x}' dy='{12 if i > 0 else 0}'>{line}</tspan>"
        for i, line in enumerate(sub_lines)
    )

    return f"""
<g class='dag-node dag-node-{node["id"]}' data-node-id='{node["id"]}'>
  <rect x='{node_x}' y='{y}' width='{width}' height='{height}' rx='8' ry='8'
        fill='{fill}' stroke='{stroke}' stroke-width='{stroke_width}'>{pulse}</rect>
  <text x='{x}' y='{y + 24}' text-anchor='middle' fill='{label_color}' font-size='14' font-weight='600'>{node["label"]}</text>
  <text x='{x}' y='{y + 42}' text-anchor='middle' fill='{sub_color}' font-size='10' font-family='SF Mono, Menlo, monospace'>{sub_tspans}</text>
</g>
"""


def _dag_edge_svg(x1: int, y1: int, x2: int, y2: int, edge_id: str, is_broken: bool, caption: str | None = None) -> str:
    """Render a directed edge between two x,y points with an arrowhead."""
    stroke = "#F87171" if is_broken else "#4A5062"
    dash = "stroke-dasharray='6,4'" if is_broken else ""
    arrow_color = stroke
    label = ""
    if caption and is_broken:
        mx, my = (x1 + x2) // 2, (y1 + y2) // 2 - 6
        label = f"<text x='{mx}' y='{my}' text-anchor='middle' fill='#F87171' font-size='10' font-style='italic'>{caption}</text>"
    return f"""
<g class='dag-edge dag-edge-{edge_id}'>
  <line x1='{x1}' y1='{y1}' x2='{x2}' y2='{y2}' stroke='{stroke}' stroke-width='2' {dash}
        marker-end='url(#arrow-{("bad" if is_broken else "ok")})' />
  {label}
</g>
"""


def render_pipeline_dag(scenario_name: str, dag_state: dict[str, Any] | None = None) -> str:
    """Render the full pipeline DAG as SVG.
    Nodes: App -> Raw -> Aggregate -> Dashboard
    Sidecar: Config tables (feed into Raw->Aggregate edge)

    Broken nodes get a red border and caption; broken edges are dashed red.
    The 'active_nodes' field is the pulse hook (unused for now; reserved for
    future live-pulse of tool calls).
    """
    if dag_state is None:
        dag_state = compute_dag_state(scenario_name)

    broken_nodes = dag_state.get("broken_nodes", {})
    broken_edges = dag_state.get("broken_edges", {})
    active_nodes = dag_state.get("active_nodes", set())

    width, height = 720, 260
    main_y = 100   # y-coordinate for main row of nodes
    config_y = 20  # y-coordinate for sidecar (above main row)

    parts = [
        f"<svg class='pipeline-dag' viewBox='0 0 {width} {height}' xmlns='http://www.w3.org/2000/svg' style='width:100%;height:auto;'>",
        # Arrow marker defs (ok and bad variants)
        """<defs>
             <marker id='arrow-ok' viewBox='0 0 10 10' refX='9' refY='5' markerWidth='6' markerHeight='6' orient='auto'>
               <path d='M 0 0 L 10 5 L 0 10 z' fill='#4A5062' />
             </marker>
             <marker id='arrow-bad' viewBox='0 0 10 10' refX='9' refY='5' markerWidth='6' markerHeight='6' orient='auto'>
               <path d='M 0 0 L 10 5 L 0 10 z' fill='#F87171' />
             </marker>
           </defs>""",
    ]

    # Main-row edges (between successive main nodes)
    for i in range(len(PIPELINE_DAG_NODES) - 1):
        src = PIPELINE_DAG_NODES[i]
        dst = PIPELINE_DAG_NODES[i + 1]
        edge_id = f"{src['id']}->{dst['id']}"
        is_broken = edge_id in broken_edges
        caption = broken_edges.get(edge_id)
        parts.append(_dag_edge_svg(
            src["x"] + 70, main_y + 31,
            dst["x"] - 70, main_y + 31,
            edge_id, is_broken, caption,
        ))

    # Sidecar -> main edge (config feeds into raw->aggregate edge)
    sidecar = PIPELINE_DAG_SIDECARS[0]
    parts.append(_dag_edge_svg(
        sidecar["x"], config_y + 62,
        sidecar["x"], main_y,
        f"config->{sidecar['id']}",
        sidecar["id"] in broken_nodes,
        None,
    ))

    # Main nodes
    for node in PIPELINE_DAG_NODES:
        parts.append(_dag_node_svg(
            node, is_broken=(node["id"] in broken_nodes),
            is_active=(node["id"] in active_nodes), y=main_y,
        ))
    # Sidecar
    parts.append(_dag_node_svg(
        sidecar, is_broken=(sidecar["id"] in broken_nodes),
        is_active=(sidecar["id"] in active_nodes), y=config_y,
    ))

    # Broken-node captions below main row
    caption_parts = []
    for nid, cap in broken_nodes.items():
        node = next((n for n in PIPELINE_DAG_NODES + PIPELINE_DAG_SIDECARS if n["id"] == nid), None)
        if node:
            cy = (config_y + 70) if node in PIPELINE_DAG_SIDECARS else (main_y + 80)
            caption_parts.append(
                f"<text x='{node['x']}' y='{cy}' text-anchor='middle' fill='#F87171' "
                f"font-size='10' font-style='italic'>{cap}</text>"
            )
    parts.extend(caption_parts)

    parts.append("</svg>")

    # Legend / overall status
    if broken_nodes or broken_edges:
        badge = "<span class='dag-status dag-status-broken'>PIPELINE IMPACTED</span>"
    else:
        badge = "<span class='dag-status dag-status-healthy'>PIPELINE HEALTHY</span>"

    return f"""
<div class='dag-container'>
  <div class='dag-header'>
    <span class='dag-title'>Pipeline</span>
    {badge}
  </div>
  <div class='dag-canvas'>{''.join(parts)}</div>
  <div class='dag-legend'>
    Click a node below to inspect the table contents. Nodes turn red when a
    scenario impacts them; dashed red edges mean an upstream step is the cause.
  </div>
</div>
"""


def render_table_contents_panel(table_name: str, max_rows: int = 25) -> str:
    """Render a table's current contents as an HTML panel. Used when the user
    clicks a DAG node."""
    if not table_name:
        return "<div class='table-panel-empty'>Click a node in the pipeline above to see its contents.</div>"
    try:
        with connect() as conn:
            rows = conn.execute(f"select * from {table_name} limit {max_rows + 1}").fetchall()
            cols = [d[0] for d in conn.description]
            total = conn.execute(f"select count(*) from {table_name}").fetchone()[0]
    except Exception as exc:
        return f"<div class='table-panel-error'>Could not read {table_name}: {exc}</div>"

    shown = rows[:max_rows]
    head = "".join(f"<th>{c}</th>" for c in cols)
    body = "".join(
        "<tr>" + "".join(f"<td>{_format_cell(v)}</td>" for v in r) + "</tr>"
        for r in shown
    )
    overflow = f"<div class='table-panel-overflow'>showing first {max_rows} of {total:,} rows</div>" if total > max_rows else f"<div class='table-panel-overflow'>{total:,} rows total</div>"

    return f"""
<div class='table-panel'>
  <div class='table-panel-header'>
    <span class='table-panel-name'>{table_name}</span>
    <span class='table-panel-count'>{total:,} rows</span>
  </div>
  <div class='table-panel-scroll'>
    <table class='diff-table'>
      <thead><tr>{head}</tr></thead>
      <tbody>{body}</tbody>
    </table>
  </div>
  {overflow}
</div>
"""


def render_pipeline_state(state: dict[str, Any]) -> str:
    """Render pipeline freshness + table counts as a persistent panel."""
    freshness = state.get("freshness", {})
    is_stale = freshness.get("is_stale", False)
    stale_class = "pipeline-stale" if is_stale else "pipeline-fresh"
    stale_label = "STALE" if is_stale else "FRESH"
    stale_color = "var(--signal-bad)" if is_stale else "var(--signal-ok)"

    table_rows = "".join(
        f"<tr><td>{t['table']}</td><td style='text-align:right;'>{t['rows']:,}</td></tr>"
        for t in state.get("tables", [])
    )

    last_refresh = state.get("last_refresh") or "—"

    return f"""
<div class='pipeline-panel {stale_class}'>
  <div class='pipeline-header'>
    <span class='pipeline-title'>Pipeline state</span>
    <span class='pipeline-badge' style='color:{stale_color}'>{stale_label}</span>
  </div>
  <div class='pipeline-grid'>
    <div class='pipeline-kv'>
      <div class='pipeline-k'>Raw latest ingest</div>
      <div class='pipeline-v'>{freshness.get("orders_latest_ingest", "—")}</div>
    </div>
    <div class='pipeline-kv'>
      <div class='pipeline-k'>Aggregate latest</div>
      <div class='pipeline-v'>{freshness.get("aggregate_latest_ingest", "—")}</div>
    </div>
    <div class='pipeline-kv'>
      <div class='pipeline-k'>Aggregate lag</div>
      <div class='pipeline-v'>{freshness.get("aggregate_lag_minutes", 0)} min</div>
    </div>
    <div class='pipeline-kv'>
      <div class='pipeline-k'>Last refresh</div>
      <div class='pipeline-v'>{last_refresh}</div>
    </div>
  </div>
  <table class='pipeline-tables'>
    <thead><tr><th>Table</th><th style='text-align:right;'>Rows</th></tr></thead>
    <tbody>{table_rows}</tbody>
  </table>
</div>
"""


def run_scenario_flow(scenario_name: str, seed: int) -> tuple[str, str, str, list[list[Any]], list[list[Any]], list[list[Any]], list[list[Any]], list[list[Any]], str]:
    baseline_message = initialize_database(seed)
    target_date = get_default_focus_date()
    for step_name in SCENARIOS[scenario_name]["apply"]:
        SCENARIO_APPLIERS[step_name](target_date)
    story = scenario_story_markdown(scenario_name)
    cards = metric_cards_html(target_date)
    brand_rows, channel_rows, config_rows, story_rows = get_network_tables(target_date)
    logs = get_pipeline_log_rows()
    status = f"{baseline_message} Scenario loaded: {scenario_name}. Focus date is {target_date}."
    return target_date, cards, story, brand_rows, channel_rows, config_rows, story_rows, logs, status


def tool_list_kpis() -> dict[str, Any]:
    return {"kpis": [{**{"key": key}, **value} for key, value in KPI_DEFINITIONS.items()]}


def tool_get_kpi_relations(kpi: str | None = None) -> dict[str, Any]:
    """Expose KPI algebra (formulas + decompositions) to the agent.
    If kpi is provided, return only that KPI's relation; else return all.
    """
    if kpi and kpi in KPI_RELATIONS:
        payload = {kpi: {**KPI_RELATIONS[kpi], "label": KPI_DEFINITIONS.get(kpi, {}).get("label", kpi)}}
    else:
        payload = {
            key: {**value, "label": KPI_DEFINITIONS.get(key, {}).get("label", key)}
            for key, value in KPI_RELATIONS.items()
        }
    return {"relations": payload}


def build_status_strip(scenario_name: str, metric_date: str) -> str:
    """Persistent top strip showing current world state at a glance.
    Shown across all tabs so the user always knows what they are looking at.
    """
    scenario = SCENARIOS.get(scenario_name, SCENARIOS["Baseline"])
    kind = scenario.get("kind", "baseline")
    headline = scenario.get("headline", "Healthy baseline")

    is_perturbed = kind != "baseline"
    strip_class = "status-strip perturbed" if is_perturbed else "status-strip"

    kind_labels = {
        "baseline": ("Baseline", "pill-baseline"),
        "business_config": ("Business / config", "pill-kpi"),
        "pipeline_data": ("Pipeline / data", "pill-data"),
        "mixed": ("Mixed", "pill-compound"),
    }
    kind_label, kind_class = kind_labels.get(kind, ("Baseline", "pill-baseline"))

    try:
        with connect() as conn:
            last_ingest = conn.execute(
                "select max(latest_ingested_at) from channel_daily_metrics where metric_date = ?",
                [metric_date],
            ).fetchone()[0]
        freshness = last_ingest.strftime("%d %b, %H:%M") if last_ingest else "—"
    except Exception:
        freshness = "—"

    return f"""
<div class="{strip_class}">
  <div class="status-item">
    <span class="status-label">Scenario</span>
    <span class="status-value">{scenario_name}</span>
  </div>
  <div class="status-item">
    <span class="status-label">Type</span>
    <span class="status-pill {kind_class}">{kind_label}</span>
  </div>
  <div class="status-item">
    <span class="status-label">Focus date</span>
    <span class="status-value">{metric_date}</span>
  </div>
  <div class="status-item">
    <span class="status-label">Aggregate freshness</span>
    <span class="status-value">{freshness}</span>
  </div>
  <div class="status-headline">{headline}</div>
</div>
"""


def tool_get_metric_snapshot(metric_date: str, brand_name: str | None = None, channel_name: str | None = None) -> dict[str, Any]:
    filters = ["metric_date = ?"]
    params: list[Any] = [metric_date]
    if brand_name:
        filters.append("brand_name = ?")
        params.append(brand_name)
    if channel_name:
        filters.append("channel_name = ?")
        params.append(channel_name)
    where_clause = " and ".join(filters)
    with connect() as conn:
        row = conn.execute(
            f"""
            select
              coalesce(sum(total_orders), 0) as total_orders,
              coalesce(sum(delivered_orders), 0) as delivered_orders,
              coalesce(sum(cancelled_orders), 0) as cancelled_orders,
              round(coalesce(sum(gross_sales), 0), 2) as gross_sales,
              round(coalesce(sum(discount_amount), 0), 2) as discount_amount,
              round(coalesce(sum(refund_amount), 0), 2) as refund_amount,
              round(coalesce(sum(net_sales), 0), 2) as net_sales,
              round(coalesce(sum(net_payout), 0), 2) as net_payout,
              round(coalesce(avg(aov), 0), 2) as aov,
              round(coalesce(avg(effective_commission_pct), 0), 2) as effective_commission_pct,
              round(coalesce(avg(cancellation_rate), 0), 4) as cancellation_rate
            from channel_daily_metrics
            where {where_clause}
            """,
            params,
        ).fetchone()
    columns = [
        "total_orders",
        "delivered_orders",
        "cancelled_orders",
        "gross_sales",
        "discount_amount",
        "refund_amount",
        "net_sales",
        "net_payout",
        "aov",
        "effective_commission_pct",
        "cancellation_rate",
    ]
    return {
        "metric_date": metric_date,
        "brand_name": brand_name,
        "channel_name": channel_name,
        "summary": dict(zip(columns, row)),
    }


def tool_compare_metric_snapshots(
    metric_date_a: str,
    metric_date_b: str,
    brand_name: str | None = None,
    channel_name: str | None = None,
) -> dict[str, Any]:
    snapshot_a = tool_get_metric_snapshot(metric_date_a, brand_name, channel_name)
    snapshot_b = tool_get_metric_snapshot(metric_date_b, brand_name, channel_name)

    deltas: dict[str, Any] = {}
    for key, value_a in snapshot_a["summary"].items():
        value_b = snapshot_b["summary"].get(key)
        if isinstance(value_a, (int, float)) and isinstance(value_b, (int, float)):
            absolute_change = round(value_b - value_a, 4)
            percent_change = None if value_a == 0 else round(((value_b - value_a) / value_a) * 100, 2)
            deltas[key] = {
                "from": value_a,
                "to": value_b,
                "absolute_change": absolute_change,
                "percent_change": percent_change,
            }

    return {
        "metric_date_a": metric_date_a,
        "metric_date_b": metric_date_b,
        "brand_name": brand_name,
        "channel_name": channel_name,
        "snapshot_a": snapshot_a["summary"],
        "snapshot_b": snapshot_b["summary"],
        "delta": deltas,
    }


def tool_get_brand_channel_config(
    brand_name: str | None = None,
    channel_name: str | None = None,
    *,
    include_notes: bool = True,
    limit: int | None = None,
    anomalies_only: bool = False,
) -> dict[str, Any]:
    filters = []
    params: list[Any] = []
    if brand_name:
        filters.append("b.brand_name = ?")
        params.append(brand_name)
    if channel_name:
        filters.append("c.channel_name = ?")
        params.append(channel_name)
    if anomalies_only:
        filters.append(
            "("
            "cfg.is_live = false "
            "or cfg.promo_active = true "
            "or cfg.menu_last_synced_at < current_timestamp - interval '24 hours'"
            ")"
        )
    where_clause = f"where {' and '.join(filters)}" if filters else ""
    limit_clause = f"limit {int(limit)}" if limit and limit > 0 else ""
    with connect() as conn:
        rows = conn.execute(
            f"""
            select
              b.brand_name,
              c.channel_name,
              cfg.is_live,
              cfg.promo_active,
              cfg.promo_discount_pct,
              cfg.effective_commission_pct,
              cfg.menu_last_synced_at,
              cfg.notes
            from channel_brand_config cfg
            join brands b on b.brand_id = cfg.brand_id
            join channels c on c.channel_id = cfg.channel_id
            {where_clause}
            order by b.brand_name, c.channel_name
            {limit_clause}
            """,
            params,
        ).fetchall()
    return {
        "rows": [
            {
                "brand_name": row[0],
                "channel_name": row[1],
                "is_live": row[2],
                "promo_active": row[3],
                "promo_discount_pct": row[4],
                "effective_commission_pct": row[5],
                "menu_last_synced_at": row[6].isoformat(sep=" ", timespec="minutes") if row[6] else None,
                **({"notes": row[7]} if include_notes else {}),
            }
            for row in rows
        ]
    }


def tool_check_duplicate_orders(
    metric_date: str,
    brand_name: str | None = None,
    channel_name: str | None = None,
) -> dict[str, Any]:
    filters = ["cast(o.ordered_at as date) = ?"]
    params: list[Any] = [metric_date]
    if brand_name:
        filters.append("b.brand_name = ?")
        params.append(brand_name)
    if channel_name:
        filters.append("c.channel_name = ?")
        params.append(channel_name)

    where_clause = " and ".join(filters)

    with connect() as conn:
        summary_row = conn.execute(
            f"""
            select
              count(*) as total_rows,
              count(distinct o.order_id) as distinct_order_ids,
              count(*) - count(distinct o.order_id) as duplicate_rows
            from orders o
            left join brands b on b.brand_id = o.brand_id
            left join channels c on c.channel_id = o.channel_id
            where {where_clause}
            """,
            params,
        ).fetchone()

        duplicate_rows = conn.execute(
            f"""
            select
              o.order_id,
              count(*) as occurrences,
              min(o.ordered_at) as first_ordered_at,
              max(o.ingested_at) as last_ingested_at,
              min(b.brand_name) as brand_name,
              min(c.channel_name) as channel_name
            from orders o
            left join brands b on b.brand_id = o.brand_id
            left join channels c on c.channel_id = o.channel_id
            where {where_clause}
            group by o.order_id
            having count(*) > 1
            order by occurrences desc, o.order_id
            limit 10
            """,
            params,
        ).fetchall()

    total_rows = summary_row[0] or 0
    distinct_order_ids = summary_row[1] or 0
    duplicate_count = summary_row[2] or 0
    duplicate_rate_pct = round((duplicate_count / total_rows) * 100, 2) if total_rows else 0.0

    return {
        "metric_date": metric_date,
        "brand_name": brand_name,
        "channel_name": channel_name,
        "total_rows": total_rows,
        "distinct_order_ids": distinct_order_ids,
        "duplicate_rows": duplicate_count,
        "duplicate_rate_pct": duplicate_rate_pct,
        "sample_duplicates": [
            {
                "order_id": row[0],
                "occurrences": row[1],
                "first_ordered_at": row[2].isoformat(sep=" ", timespec="minutes") if row[2] else None,
                "last_ingested_at": row[3].isoformat(sep=" ", timespec="minutes") if row[3] else None,
                "brand_name": row[4],
                "channel_name": row[5],
            }
            for row in duplicate_rows
        ],
    }


def tool_get_table_freshness(table_name: str) -> dict[str, Any]:
    with connect() as conn:
        if table_name == "orders":
            row = conn.execute(
                "select count(*), min(cast(ordered_at as date)), max(cast(ordered_at as date)), max(ingested_at) from orders"
            ).fetchone()
            return {
                "table": "orders",
                "row_count": row[0],
                "min_order_date": str(row[1]),
                "max_order_date": str(row[2]),
                "latest_ingested_at": row[3].isoformat(sep=" ", timespec="minutes") if row[3] else None,
            }
        row = conn.execute(
            "select count(*), min(metric_date), max(metric_date), max(latest_ingested_at) from channel_daily_metrics"
        ).fetchone()
        return {
            "table": "channel_daily_metrics",
            "row_count": row[0],
            "min_metric_date": str(row[1]),
            "max_metric_date": str(row[2]),
            "latest_ingested_at": row[3].isoformat(sep=" ", timespec="minutes") if row[3] else None,
        }


def sanitize_pipeline_log_message(action: str, message: str) -> str:
    action_key = (action or "").strip().lower()
    if action_key == "reset_baseline":
        return "Baseline was rebuilt."
    if action_key == "scenario":
        return "A state change was applied to the environment."
    if action_key == "pipeline_failure":
        return "A pipeline/data-path anomaly was recorded."
    if action_key == "refresh_aggregation":
        return "Aggregation was rebuilt."
    if action_key == "run_sql":
        return "Ad hoc SQL was executed for inspection."
    return message


def tool_inspect_pipeline_logs(limit: int = 8, *, sanitize: bool = False) -> dict[str, Any]:
    if not LOG_PATH.exists():
        return {"runs": []}
    payload = json.loads(LOG_PATH.read_text())
    runs = payload.get("runs", [])[-limit:]
    return {
        "runs": [
            {
                "timestamp": run["timestamp"],
                "action": run["action"],
                "message": sanitize_pipeline_log_message(run["action"], run["message"]) if sanitize else run["message"],
            }
            for run in runs
        ]
    }


def tool_run_sql(query: str) -> dict[str, Any]:
    with connect() as conn:
        result = conn.execute(query)
        columns = [item[0] for item in result.description] if result.description else []
        rows = result.fetchmany(20) if columns else []
    write_log("run_sql", "Assistant executed ad hoc SQL for inspection.", {"query": query[:200]})
    return {"columns": columns, "rows": rows}


def tool_refresh_aggregation() -> dict[str, Any]:
    message = refresh_aggregation()
    return {"status": "ok", "message": message}


def tool_submit_incident_report(
    root_cause: str,
    fix: str,
    root_cause_category: str | None = None,
    affected_kpis: list[str] | None = None,
) -> dict[str, Any]:
    """Submit an incident report. Terminal action in an episode.
    Triggers reward computation against the current scenario's ground truth.

    Arguments:
      root_cause: free-text description
      fix: proposed remediation
      root_cause_category: one of ROOT_CAUSE_CATEGORIES (for scoring)
      affected_kpis: list of KPI keys the agent believes are impacted
    """
    global _LATEST_TRACE_FOR_SCORING  # populated by llm_assistant before this is called

    # If the caller didn't provide a category, try to infer from the root_cause text
    if root_cause_category is None:
        root_cause_category = _infer_root_cause_category(root_cause)

    report = {
        "root_cause": root_cause,
        "root_cause_category": root_cause_category,
        "affected_kpis": affected_kpis or [],
        "fix": fix,
    }

    # Score against current scenario's ground truth
    trace = _LATEST_TRACE_FOR_SCORING or []
    reward = compute_scenario_reward(trace, report, ACTIVE_SCENARIO_NAME)

    payload = {
        "status": "submitted",
        "root_cause": root_cause,
        "root_cause_category": root_cause_category,
        "affected_kpis": affected_kpis or [],
        "fix": fix,
        "submitted_at": datetime.now().isoformat(timespec="seconds"),
        "reward": reward,
    }
    write_log("submit_incident_report", f"Reward: {reward.get('total', 0):+.3f}", payload)
    # Stash latest reward for the UI to pick up
    global _LATEST_REWARD
    _LATEST_REWARD = reward
    return payload


def _infer_root_cause_category(text: str) -> str:
    """Heuristic: map free-text root cause to one of the canonical categories."""
    t = (text or "").lower()
    if any(k in t for k in ["promo", "commission", "sync", "is_live", "config", "flag"]):
        return "business_config_drift"
    if any(k in t for k in ["stale", "aggregate", "schema", "pipeline", "refresh", "ingest"]):
        return "pipeline_failure"
    if any(k in t for k in ["mix shift", "seasonality", "demand", "not a bug", "business reality"]):
        return "external_business_reality"
    if "both" in t or "multiple" in t or "compound" in t:
        return "compound"
    return "business_config_drift"  # most common default in this world


# Module-level state for scoring. Set when a scenario is loaded and when the
# assistant runs, so submit_incident_report can look it up without plumbing.
ACTIVE_SCENARIO_NAME: str = "Baseline"
_LATEST_TRACE_FOR_SCORING: list[dict[str, Any]] = []
_LATEST_REWARD: dict[str, Any] | None = None


def render_tool_trace_html(trace_text: str) -> str:
    """Turn the raw JSON tool trace into a human-readable timeline.
    Expects trace_text to be either a JSON list of {tool, args, result} dicts,
    or JSON lines. Falls back to raw JSON in a <pre> if structure is unknown.
    """
    if not trace_text or trace_text.strip() in ("", "{}", "[]"):
        return "<div class='trace-empty'>No tool calls yet. Ask a question above.</div>"

    try:
        data = json.loads(trace_text)
    except Exception:
        return f"<pre class='trace-raw'>{trace_text}</pre>"

    # Normalize to a list of events
    if isinstance(data, dict):
        events = data.get("trace") or data.get("tool_calls") or [data]
    elif isinstance(data, list):
        events = data
    else:
        return f"<pre class='trace-raw'>{trace_text}</pre>"

    if not events:
        return "<div class='trace-empty'>No tool calls recorded.</div>"

    pieces = ["<div class='trace-timeline'>"]
    for i, ev in enumerate(events, 1):
        if not isinstance(ev, dict):
            continue
        tool = ev.get("tool") or ev.get("name") or "unknown"
        args = ev.get("args") or ev.get("arguments") or ev.get("input") or {}
        result = ev.get("result") or ev.get("output") or ev.get("response")

        args_str = json.dumps(args, default=str) if args else ""
        if len(args_str) > 140:
            args_str = args_str[:137] + "..."

        result_preview = ""
        if result is not None:
            result_str = json.dumps(result, default=str)[:220]
            result_preview = f"<div class='trace-result'>↳ {result_str}</div>"

        pieces.append(f"""
<div class='trace-event'>
  <div class='trace-step'>{i}</div>
  <div class='trace-body'>
    <div class='trace-call'><code class='trace-tool'>{tool}</code><span class='trace-args'>({args_str})</span></div>
    {result_preview}
  </div>
</div>
""")
    pieces.append("</div>")
    return "".join(pieces)


def find_hf_token() -> str | None:
    api_key = (
        os.getenv("HF_TOKEN")
        or os.getenv("HUGGINGFACE_API_KEY")
        or os.getenv("HUGGINGFACEHUB_API_TOKEN")
        or os.getenv("HF_API_TOKEN")
    )
    if api_key:
        return api_key

    try:
        from huggingface_hub import get_token

        api_key = get_token()
        if api_key:
            return api_key
    except Exception:
        pass

    project_root = APP_DIR.parent
    candidate_files = [
        project_root / "myfirstenv" / "inference.py",
        project_root / "storeops_env" / "inference.py",
        project_root / "sudoku_rl" / "llm_inference.py",
    ]
    for candidate in candidate_files:
        if not candidate.exists():
            continue
        match = re.search(r"hf_[A-Za-z0-9]{20,}", candidate.read_text(errors="ignore"))
        if match:
            return match.group(0)
    return None


def get_hf_client() -> OpenAI | None:
    api_key = find_hf_token()
    if not api_key:
        return None
    return OpenAI(api_key=api_key, base_url=HF_BASE_URL)


BRAND_ALIASES = {
    "behrouz": "Behrouz Biryani",
    "behrouz biryani": "Behrouz Biryani",
    "oven story": "Oven Story Pizza",
    "ovenstory": "Oven Story Pizza",
    "oven story pizza": "Oven Story Pizza",
    "faasos": "Faasos",
    "sweet truth": "Sweet Truth",
    "sweettruth": "Sweet Truth",
    "bowl company": "The Bowl Company",
    "the bowl company": "The Bowl Company",
}
CHANNEL_ALIASES = {
    "zomato": "Zomato",
    "swiggy": "Swiggy",
    "ondc": "ONDC",
    "own app": "Own App",
    "ownapp": "Own App",
}


def extract_date(text: str, fallback: str) -> str:
    match = re.search(r"\d{4}-\d{2}-\d{2}", text)
    if match:
        return match.group(0)
    if "yesterday" in text.lower():
        return (date.today() - timedelta(days=1)).isoformat()
    if "today" in text.lower():
        return date.today().isoformat()
    return fallback


def extract_brand_channel(text: str) -> tuple[str | None, str | None]:
    lowered = text.lower()
    brand_name = next((value for key, value in sorted(BRAND_ALIASES.items(), key=lambda item: len(item[0]), reverse=True) if key in lowered), None)
    channel_name = next((value for key, value in sorted(CHANNEL_ALIASES.items(), key=lambda item: len(item[0]), reverse=True) if key in lowered), None)
    return brand_name, channel_name


def reason_from_snapshot(snapshot: dict[str, Any], config: dict[str, Any]) -> str:
    summary = snapshot["summary"]
    config_rows = config.get("rows", [])
    if not summary:
        return "I could not find matching metrics for that slice yet."
    if config_rows:
        row = config_rows[0]
        if not row["is_live"]:
            return f"{row['brand_name']} on {row['channel_name']} is not live right now, which explains the sudden order drop without a pipeline failure."
        if row["promo_active"] and row["notes"] and "sync" in row["notes"]:
            return (
                f"Both signals are real here: {row['brand_name']} on {row['channel_name']} still has an active "
                f"{row['promo_discount_pct']:.0f}% promo, and the menu sync is stale, so discount leakage and stale pricing are combining to pressure margin."
            )
        if row["promo_active"] and summary["discount_amount"] > 0:
            return (
                f"The main story is promo pressure: {row['brand_name']} on {row['channel_name']} still has an active discount of "
                f"{row['promo_discount_pct']:.0f}%, so revenue is being bought through discounting rather than real demand growth."
            )
        if row["notes"] and "sync" in row["notes"]:
            return (
                f"The brand-channel config shows stale menu sync. Higher AOV here looks operational, not analytical: the aggregator menu is out of date."
            )
        if row["notes"] and "commission" in row["notes"]:
            return (
                f"This is mainly a margin story, not a data failure: commission was revised upward on {row['channel_name']}, so payout is flattening even when sales look healthy."
            )
    if summary["net_payout"] < summary["net_sales"] * 0.8:
        return "Sales are healthy, but payout is compressed because commission is taking a larger share than usual."
    if summary["cancelled_orders"] and summary["cancelled_orders"] >= max(2, summary["total_orders"] * 0.2):
        return "The biggest drag is conversion leakage: too many created orders are cancelling before they become delivered revenue."
    if summary["discount_amount"] > max(1, summary["gross_sales"] * 0.15):
        return "Revenue is soft mainly because discount spend is too heavy relative to topline."
    return "The network looks broadly healthy. Nothing in the joined data points to an active data incident."


def heuristic_assistant(message: str, history: list[dict[str, str]], focus_date: str) -> tuple[list[dict[str, str]], str, str]:
    brand_name, channel_name = extract_brand_channel(message)
    metric_date = extract_date(message, focus_date)
    trace: list[dict[str, Any]] = []

    snapshot = tool_get_metric_snapshot(metric_date, brand_name, channel_name)
    trace.append({"tool": "get_metric_snapshot", "arguments": {"metric_date": metric_date, "brand_name": brand_name, "channel_name": channel_name}, "result": snapshot})

    config = tool_get_brand_channel_config(brand_name, channel_name)
    trace.append({"tool": "get_brand_channel_config", "arguments": {"brand_name": brand_name, "channel_name": channel_name}, "result": config})

    freshness = tool_get_table_freshness("channel_daily_metrics")
    trace.append({"tool": "get_table_freshness", "arguments": {"table_name": "channel_daily_metrics"}, "result": freshness})

    if any(token in message.lower() for token in ["broken", "wrong", "incident", "stale", "pipeline", "api", "sync"]):
        logs = tool_inspect_pipeline_logs()
        trace.append({"tool": "inspect_pipeline_logs", "arguments": {"limit": 8}, "result": logs})

    summary = snapshot["summary"]
    opening = []
    if brand_name or channel_name:
        slice_name = " / ".join([part for part in [brand_name, channel_name] if part])
        opening.append(f"For {slice_name} on {metric_date},")
    else:
        opening.append(f"For the network on {metric_date},")

    opening.append(
        f"we saw {summary['total_orders']} total orders, gross sales of Rs {summary['gross_sales']:,.0f}, "
        f"net sales of Rs {summary['net_sales']:,.0f}, and payout of Rs {summary['net_payout']:,.0f}."
    )
    business_reason = reason_from_snapshot(snapshot, config)
    next_hint = " Next best KPIs to inspect would be "
    if "payout" in message.lower():
        next_hint += "`commission_amount` and `effective_commission_pct`."
    elif "revenue" in message.lower() or "sales" in message.lower():
        next_hint += "`discount_amount`, `aov`, and `cancellation_rate`."
    else:
        next_hint += "`net_sales`, `net_payout`, and `brand_channel_coverage`."

    answer = " ".join(opening) + " " + business_reason + next_hint

    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": answer},
    ]
    latest_result = json.dumps(snapshot, indent=2, default=str)
    global _LATEST_TRACE_FOR_SCORING
    _LATEST_TRACE_FOR_SCORING = trace
    return history, answer, json.dumps(trace, indent=2, default=str), latest_result


def normalize_chat_history(history: Any) -> list[dict[str, str]]:
    if not history:
        return []
    normalized: list[dict[str, str]] = []
    if isinstance(history, list):
        for item in history:
            if isinstance(item, dict) and "role" in item and "content" in item:
                normalized.append({"role": item["role"], "content": item["content"]})
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                user_msg, assistant_msg = item
                if user_msg:
                    normalized.append({"role": "user", "content": str(user_msg)})
                if assistant_msg:
                    normalized.append({"role": "assistant", "content": str(assistant_msg)})
    return normalized


def extract_json_object(text: str) -> dict[str, Any]:
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        cleaned = cleaned.strip()

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    decoder = json.JSONDecoder()
    for idx, char in enumerate(cleaned):
        if char != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(cleaned[idx:])
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            continue

    type_match = re.search(r'"type"\s*:\s*"([^"]+)"', cleaned, flags=re.DOTALL)
    if type_match:
        parsed: dict[str, Any] = {"type": type_match.group(1)}
        tool_match = re.search(r'"tool"\s*:\s*"([^"]+)"', cleaned, flags=re.DOTALL)
        if tool_match:
            parsed["tool"] = tool_match.group(1)
        answer_match = re.search(r'"answer"\s*:\s*"([\s\S]*)"\s*}\s*$', cleaned, flags=re.DOTALL)
        if answer_match:
            parsed["answer"] = answer_match.group(1).replace('\\"', '"').strip()
            return parsed
    raise ValueError("Model did not return a valid JSON object.")


def build_agent_tool_descriptions(default_focus_date: str) -> str:
    return (
        "Available tools:\n"
        "- list_kpis(): list KPI definitions and what they mean.\n"
        "- get_kpi_relations(kpi?): get formulas/decomposition for one KPI or all KPIs.\n"
        f"- get_metric_snapshot(metric_date?, brand_name?, channel_name?): get one aggregated snapshot. Default metric_date is {default_focus_date}.\n"
        f"- compare_metric_snapshots(metric_date_a?, metric_date_b?, brand_name?, channel_name?): compare two snapshots and return deltas. If omitted, metric_date_b defaults to {default_focus_date} and metric_date_a should usually be the prior day.\n"
        "- get_brand_channel_config(brand_name?, channel_name?, limit?, anomalies_only?): inspect config rows. Avoid broad dumps. If brand/channel is unknown, prefer anomalies_only=true and a small limit.\n"
        "- get_table_freshness(table_name): inspect raw or aggregate freshness for 'orders' or 'channel_daily_metrics'.\n"
        "- inspect_pipeline_logs(limit?): get sanitized recent pipeline activity.\n"
        "- check_duplicate_orders(metric_date?, brand_name?, channel_name?): compare total rows vs distinct order_id rows in raw orders for duplicate/replay checks.\n"
        "- run_sql(query): run a focused SQL query when a structured tool is insufficient. Keep it inspection-only.\n"
        "- refresh_aggregation(): rebuild the aggregate if you have evidence the aggregate is stale.\n"
    )


def dispatch_agent_tool(tool_name: str, arguments: dict[str, Any], focus_date: str) -> dict[str, Any]:
    args = dict(arguments or {})

    if tool_name == "list_kpis":
        return tool_list_kpis()
    if tool_name == "get_kpi_relations":
        return tool_get_kpi_relations(args.get("kpi"))
    if tool_name == "get_metric_snapshot":
        return tool_get_metric_snapshot(
            args.get("metric_date") or focus_date,
            args.get("brand_name"),
            args.get("channel_name"),
        )
    if tool_name == "compare_metric_snapshots":
        metric_date_b = args.get("metric_date_b") or focus_date
        metric_date_a = args.get("metric_date_a") or (date.fromisoformat(metric_date_b) - timedelta(days=1)).isoformat()
        return tool_compare_metric_snapshots(
            metric_date_a,
            metric_date_b,
            args.get("brand_name"),
            args.get("channel_name"),
        )
    if tool_name == "get_brand_channel_config":
        limit = args.get("limit")
        anomalies_only = bool(args.get("anomalies_only", False))
        if not args.get("brand_name") and not args.get("channel_name"):
            limit = limit or 6
            anomalies_only = True if "anomalies_only" not in args else anomalies_only
        return tool_get_brand_channel_config(
            args.get("brand_name"),
            args.get("channel_name"),
            include_notes=False,
            limit=limit,
            anomalies_only=anomalies_only,
        )
    if tool_name == "get_table_freshness":
        return tool_get_table_freshness(args.get("table_name") or "channel_daily_metrics")
    if tool_name == "inspect_pipeline_logs":
        return tool_inspect_pipeline_logs(limit=int(args.get("limit", 8)), sanitize=True)
    if tool_name == "check_duplicate_orders":
        return tool_check_duplicate_orders(
            args.get("metric_date") or focus_date,
            args.get("brand_name"),
            args.get("channel_name"),
        )
    if tool_name == "run_sql":
        query = (args.get("query") or "").strip()
        if not query:
            return {"error": "query is required"}
        return tool_run_sql(query)
    if tool_name == "refresh_aggregation":
        return tool_refresh_aggregation()
    return {"error": f"Unsupported tool request: {tool_name}"}


def llm_assistant(message: str, history: list[dict[str, str]], focus_date: str) -> tuple[list[dict[str, str]], str, str]:
    client = get_hf_client()
    if client is None:
        return heuristic_assistant(message, history, focus_date)

    brand_name, channel_name = extract_brand_channel(message)
    metric_date = extract_date(message, focus_date)
    trace: list[dict[str, Any]] = []
    latest_result = "{}"

    comparisonish = bool(
        re.search(r"\b(up|down|flat|increase|decrease|higher|lower|compared?|versus|vs\.?|double|2x|twice|expected)\b", message.lower())
    )
    duplicateish = any(token in message.lower() for token in ["duplicate", "double", "2x", "twice", "replay", "expected"])

    system_prompt = (
        "You are a tool-using FoodOps incident analyst. "
        "You must reason step by step and inspect evidence before answering. "
        "Do not assume facts just because the user phrased them. Treat the user's wording as a hypothesis. "
        "Do not claim a metric is up/down/flat without direct comparison evidence. "
        "Do not explain a duplicate/replay anomaly without checking duplicate evidence first. "
        "Avoid broad config dumps when you do not know the brand/channel; prefer targeted or anomaly-only requests. "
        "Never mention hidden scenario state, labels, or notes. Use only tool outputs. "
        "Respond with JSON only. Valid response shapes:\n"
        '{"type":"tool","tool":"get_metric_snapshot","arguments":{"metric_date":"2026-04-24","brand_name":null,"channel_name":"ONDC"}}\n'
        '{"type":"final","answer":"Observed: ...\\nLikely driver: ...\\nMissing check: ..."}\n'
        "Use exactly one tool per step. When you have enough evidence, return type=final."
    )
    kickoff = (
        f"User question: {message}\n"
        f"Default focus date: {metric_date}\n"
        f"Inferred brand from question: {brand_name or 'unknown'}\n"
        f"Inferred channel from question: {channel_name or 'unknown'}\n"
        f"Question flags: comparison={comparisonish}, duplicate_or_2x={duplicateish}\n\n"
        f"{build_agent_tool_descriptions(metric_date)}"
    )

    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
    messages.extend(history[-6:])
    messages.append({"role": "user", "content": kickoff})

    try:
        final_answer = "I could not complete the request."
        for step in range(8):
            response = client.chat.completions.create(
                model=HF_MODEL,
                messages=messages,
                temperature=0.0,
            )
            content = response.choices[0].message.content or ""
            decision = extract_json_object(content)

            decision_type = (decision.get("type") or "").strip().lower()
            decision_tool = (decision.get("tool") or "").strip()

            if decision_type == "final" or decision_tool == "final":
                used_tools = {event.get("tool") for event in trace}
                if comparisonish and "compare_metric_snapshots" not in used_tools:
                    messages.append({"role": "assistant", "content": json.dumps(decision, ensure_ascii=True)})
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "You are trying to answer a comparison-style question without direct comparison evidence. "
                                "Call compare_metric_snapshots first, then answer."
                            ),
                        }
                    )
                    continue
                if duplicateish and "check_duplicate_orders" not in used_tools:
                    messages.append({"role": "assistant", "content": json.dumps(decision, ensure_ascii=True)})
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "This question suggests a possible duplicate/replay anomaly. "
                                "Call check_duplicate_orders before returning a final answer."
                            ),
                        }
                    )
                    continue
                final_answer = decision.get("answer") or "I could not generate a grounded answer."
                trace.append(
                    {
                        "tool": "hf_chat_completion",
                        "arguments": {"model": HF_MODEL, "base_url": HF_BASE_URL, "step": step + 1},
                        "result": {"status": "ok", "decision": "final"},
                    }
                )
                break

            if decision_type != "tool" and not decision_tool:
                raise ValueError(f"Unexpected model response: {content}")

            tool_name = decision_tool
            arguments = decision.get("arguments") or {}
            tool_result = dispatch_agent_tool(tool_name, arguments, metric_date)
            trace.append({"tool": tool_name, "arguments": arguments, "result": tool_result})
            latest_result = json.dumps(tool_result, indent=2, default=str)

            messages.append({"role": "assistant", "content": json.dumps(decision, ensure_ascii=True)})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"Tool result for {tool_name}:\n"
                        f"{json.dumps(tool_result, indent=2, default=str)}\n\n"
                        "Continue. Either call one more tool or return a final answer."
                    ),
                }
            )
        else:
            final_answer = (
                "Observed: I gathered some evidence but did not converge cleanly within the tool budget.\n"
                "Likely driver: uncertain.\n"
                "Missing check: the tool loop hit its step limit before a grounded conclusion."
            )

        history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": final_answer},
        ]
        global _LATEST_TRACE_FOR_SCORING
        _LATEST_TRACE_FOR_SCORING = trace
        return history, final_answer, json.dumps(trace, indent=2, default=str), latest_result
    except Exception as exc:
        trace.append(
            {
                "tool": "hf_chat_completion",
                "arguments": {"model": HF_MODEL, "base_url": HF_BASE_URL},
                "result": {"status": "fallback", "error": str(exc)},
            }
        )
        fallback_history, fallback_answer, fallback_trace, latest_result = heuristic_assistant(message, history, focus_date)
        merged_trace = trace + json.loads(fallback_trace)
        # Label the fallback so the user knows they are NOT seeing real LLM reasoning
        labeled_answer = (
            "⚠️ **Heuristic fallback** (Hugging Face provider unavailable — token, quota, or network). "
            "The response below is from a deterministic backup, not a real LLM.\n\n"
            f"{fallback_answer}"
        )
        if fallback_history and fallback_history[-1].get("role") == "assistant":
            fallback_history[-1]["content"] = labeled_answer
        _LATEST_TRACE_FOR_SCORING = merged_trace
        return fallback_history, labeled_answer, json.dumps(merged_trace, indent=2, default=str), latest_result


def refresh_everything(metric_date: str) -> tuple[str, list[list[Any]], list[list[Any]], list[list[Any]], list[list[Any]], list[list[Any]]]:
    cards = metric_cards_html(metric_date)
    brand_rows, channel_rows, config_rows, story_rows = get_network_tables(metric_date)
    logs = get_pipeline_log_rows()
    return cards, brand_rows, channel_rows, config_rows, story_rows, logs


def build_demo() -> gr.Blocks:
    initial_date = get_default_focus_date() if DB_PATH.exists() else date.today().isoformat()
    with gr.Blocks(title="FoodOps Incident Lab") as demo:
        # Hero — intentionally minimal, like a normal product
        gr.HTML(
            """
            <div class="app-shell">
              <div class="hero">
                <h1>FoodOps</h1>
                <p>Brand × channel performance across Zomato, Swiggy, ONDC, and Own App.</p>
              </div>
            </div>
            """
        )

        focus_date = gr.State(initial_date)
        active_scenario = gr.State("Baseline")

        # ================================================================
        # TAB 1 — DASHBOARD  (looks like a real business dashboard)
        # ================================================================
        with gr.Tab("Dashboard"):
            with gr.Row():
                overview_date = gr.Dropdown(
                    label="Date",
                    choices=get_recent_dates() or [initial_date],
                    value=initial_date,
                    elem_id="overview-date-dropdown",
                    elem_classes=["dark-dropdown"],
                    scale=2,
                )
                gr.HTML(
                    "<div class='dashboard-freshness' style='padding-top:28px;'>"
                    "Live figures for the selected day, rolled up from raw orders."
                    "</div>",
                    scale=5,
                )
            scenario_status_html = gr.HTML(build_status_strip("Baseline", initial_date))

            # --- KPI strip (full width, top) ---
            metric_cards = gr.HTML(metric_cards_html(initial_date) if DB_PATH.exists() else "")

            # --- Tables (full width, below KPIs) ---
            with gr.Row():
                brand_table = gr.Dataframe(
                    headers=["Brand", "Orders", "Gross Sales", "Net Sales", "Net Payout", "AOV"],
                    interactive=False,
                    label="By Brand",
                )
                channel_table = gr.Dataframe(
                    headers=["Channel", "Orders", "Gross Sales", "Net Sales", "Net Payout", "Avg Commission %"],
                    interactive=False,
                    label="By Channel",
                )
            spotlight_table = gr.Dataframe(
                headers=["Brand", "Channel", "Orders", "Net Sales", "Net Payout", "Promo Active", "Commission %"],
                interactive=False,
                label="Top Brand × Channel Combinations",
            )
            with gr.Accordion("Brand × Channel configuration", open=False):
                config_table = gr.Dataframe(
                    headers=["Brand", "Channel", "Is Live", "Promo Active", "Promo %", "Commission %", "Menu Last Synced", "Ops Notes"],
                    interactive=False,
                    label=None,
                )

            # --- Chat + JSON side by side (full width, bottom) ---
            gr.HTML("<div style='border-top:1px solid var(--border); margin-top:18px; padding-top:14px;'></div>")
            gr.Markdown("### Ask the assistant")
            gr.Markdown(
                "Questions about any brand, channel, KPI, or unusual number. "
                "Every tool the assistant calls to answer is shown on the right."
            )
            with gr.Row():
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(label="Chat", height=360, elem_classes=["chat-window"])
                    with gr.Row():
                        prompt_box = gr.Textbox(
                            label="Ask a question",
                            placeholder="e.g. Why is discount amount up on Behrouz × Swiggy?",
                            elem_classes=["simple-text-input"],
                            interactive=True,
                            lines=2,
                            max_lines=4,
                            scale=5,
                        )
                        send_btn = gr.Button("Send", variant="primary", scale=1)
                with gr.Column(scale=2):
                    gr.Markdown("**Tool trace — what the assistant did**")
                    tool_trace_html = gr.HTML(
                        "<div class='trace-empty'>No tool calls yet. Ask a question on the left.</div>"
                    )
                    with gr.Accordion("Raw JSON", open=False):
                        tool_trace = gr.Code(label="Tool Trace JSON", language="json")
                        latest_result = gr.Code(label="Latest Tool Result", language="json")

            with gr.Accordion("KPI reference", open=False):
                kpi_meaning = gr.HTML(
                    "<div class='kpi-ref-list'>"
                    + "".join(
                        f"<p><strong>{value['label']}</strong> — {value['meaning']}"
                        + (
                            f"<br/><span style='color:var(--text-muted)'>Formula:</span> "
                            f"<code>{KPI_RELATIONS[key]['formula']}</code>"
                            f"<br/><span style='color:var(--text-muted)'>Decomposition:</span> "
                            f"{KPI_RELATIONS[key]['decomposition']}"
                            if key in KPI_RELATIONS else ""
                        )
                        + "</p>"
                        for key, value in KPI_DEFINITIONS.items()
                    )
                    + "</div>"
                )

        # ================================================================
        # TAB 2 — SCENARIO STUDIO  (storybook: before → injection → next step)
        # ================================================================
        with gr.Tab("Scenario Studio"):
            gr.Markdown(
                "### Simulate a real-world incident\n"
                "We now split scenarios into three buckets: **Business / config**, **Pipeline / data**, "
                "and **Mixed**. Pick a scenario, watch which part of the system changes, then head to "
                "the Dashboard and ask the assistant to diagnose it."
            )
            gr.HTML(render_scenario_catalog())
            with gr.Row():
                scenario_name = gr.Dropdown(
                    label="Scenario",
                    choices=list(SCENARIOS),
                    value="Baseline",
                    elem_id="scenario-dropdown",
                    elem_classes=["dark-dropdown"],
                    scale=3,
                )
                load_btn = gr.Button("Run Scenario", variant="primary", scale=1)
            status_bar = gr.Markdown(
                "Ready. Pick a scenario above and click **Run Scenario**. "
                "The Dashboard tab will reflect the changes."
            )

            # Pipeline DAG — the visual centerpiece. Turns red on broken nodes/edges.
            studio_dag_html = gr.HTML(render_pipeline_dag("Baseline"))

            # Clickable node buttons (rendered as a row, positioned below the DAG)
            # Clicking a button opens a table-contents panel for that node's table.
            gr.HTML("<div class='node-click-row-label' style='color:var(--text-muted);"
                    "font-size:0.78rem;margin:6px 0 4px 0;'>Inspect a pipeline node</div>")
            with gr.Row(elem_classes="node-click-row"):
                node_app_btn       = gr.Button("App (stream)")
                node_raw_btn       = gr.Button("Raw: orders")
                node_config_btn    = gr.Button("Config: channel_brand_config")
                node_agg_btn       = gr.Button("Aggregate: channel_daily_metrics")
                node_dash_btn      = gr.Button("Dashboard (KPIs)")
            studio_table_panel_html = gr.HTML(
                "<div class='table-panel-empty'>Click a node button above to inspect its table contents.</div>"
            )

            # Main storybook narrative — the three acts, below the DAG
            storybook_html = gr.HTML(
                render_storybook_acts(
                    "Baseline",
                    {"touched_tables": [], "diff": {}, "pipeline_state": {}},
                )
            )

            # Collapsibles for the curious: raw diff and pipeline state
            with gr.Accordion("Show me exactly which rows changed", open=False):
                scenario_diff_html = gr.HTML(
                    "<div class='diff-empty'>"
                    "<strong>Baseline — no rows have been modified.</strong>"
                    "<p>Run a scenario to see the before/after diff.</p>"
                    "</div>"
                )
            with gr.Accordion("Pipeline state (freshness, row counts)", open=False):
                pipeline_state_html = gr.HTML(
                    render_pipeline_state(get_pipeline_state_snapshot()) if DB_PATH.exists() else ""
                )
            with gr.Accordion("Recent pipeline / scenario actions", open=False):
                scenario_actions = gr.Dataframe(
                    headers=["Timestamp", "Action", "Message"],
                    interactive=False,
                    label=None,
                )

        # ================================================================
        # TAB 3 — DATA EXPLORER  (engineer's back door)
        # ================================================================
        with gr.Tab("Data Explorer"):
            gr.Markdown(
                "### Engineer view\n"
                "Reset the baseline, rebuild aggregates, inspect any table, run SQL. "
                "The pipeline diagram below is always healthy here — this tab is the "
                "back door into the system, independent of any scenario."
            )

            # DAG view — always healthy in Data Explorer
            explorer_dag_html = gr.HTML(render_pipeline_dag("Baseline"))

            with gr.Row():
                seed_input = gr.Number(label="Seed", value=DEFAULT_SEED, precision=0)
                reset_btn = gr.Button("Reset to Baseline", variant="secondary")
                rebuild_btn = gr.Button("Refresh Aggregate", variant="secondary")
            table_counts = gr.Dataframe(
                headers=["Table", "Row Count"], interactive=False, label="Table row counts"
            )
            with gr.Row():
                table_name = gr.Dropdown(
                    label="Inspect Table",
                    choices=["brands", "channels", "channel_brand_config", "orders", "channel_daily_metrics"],
                    value="orders",
                )
                inspect_table_btn = gr.Button("Inspect", variant="secondary")

            # Table preview comes FIRST so results of Inspect/Run SQL are visible immediately
            table_preview = gr.Dataframe(interactive=False, label="Table Preview")

            # SQL editors BELOW the preview, so running them scrolls down naturally
            gr.Markdown(
                "**SQL sandbox.** The left editor shows the current CREATE / aggregation SQL "
                "for reference. Edit the right editor to run your own query. "
                "⚠️ Running the aggregation SQL will rebuild the aggregate table — "
                "avoid this mid-demo."
            )
            with gr.Row():
                table_sql = gr.Code(label="Create / Aggregate SQL (read-only reference)", language="sql")
                sql_runner = gr.Code(label="Your Query (editable)", language="sql")
            execute_sql_btn = gr.Button("Run Query", variant="primary")
            sql_status = gr.Markdown("SQL runner ready.")

            pipeline_logs = gr.Dataframe(
                headers=["Timestamp", "Action", "Message"],
                interactive=False,
                label="Pipeline Logs",
            )

        # ================================================================
        # WIRING
        # ================================================================
        def sync_from_date(metric_date: str):
            cards, brands, channels, configs, story_rows, logs = refresh_everything(metric_date)
            return (
                cards, brands, channels, configs, story_rows, logs,
                get_table_counts(), metric_date, build_status_strip(ACTIVE_SCENARIO_NAME, metric_date),
            )

        overview_date.change(
            sync_from_date,
            inputs=[overview_date],
            outputs=[
                metric_cards, brand_table, channel_table, config_table,
                spotlight_table, scenario_actions, table_counts, focus_date, scenario_status_html,
            ],
        )

        def handle_reset(seed: float):
            message = initialize_database(int(seed))
            metric_date = get_default_focus_date()
            cards, brands, channels, configs, story_rows, logs = refresh_everything(metric_date)
            baseline_report = {"touched_tables": [], "diff": {}, "pipeline_state": get_pipeline_state_snapshot()}
            story_html_val = render_storybook_acts("Baseline", baseline_report)
            baseline_diff_html = (
                "<div class='diff-empty'>"
                "<strong>Baseline — no rows have been modified.</strong>"
                "<p>Run a scenario to see the before/after diff.</p>"
                "</div>"
            )
            pipeline_html = render_pipeline_state(baseline_report["pipeline_state"])
            dag_html_val = render_pipeline_dag("Baseline")
            empty_panel = "<div class='table-panel-empty'>Click a node button above to inspect its table contents.</div>"
            return (
                message, metric_date, cards, brands, channels, configs,
                story_rows, logs, get_table_counts(),
                story_html_val, "Baseline", "Baseline", build_status_strip("Baseline", metric_date),
                baseline_diff_html, pipeline_html,
                dag_html_val, empty_panel,
            )

        reset_btn.click(
            handle_reset,
            inputs=[seed_input],
            outputs=[
                sql_status, overview_date, metric_cards, brand_table, channel_table,
                config_table, spotlight_table, scenario_actions, table_counts,
                storybook_html, active_scenario, scenario_name, scenario_status_html,
                scenario_diff_html, pipeline_state_html,
                studio_dag_html, studio_table_panel_html,
            ],
        )

        def handle_load_scenario(scenario_name_value: str, seed: float):
            """Apply the scenario, refresh dashboard data, and render the storybook
            on the Scenario Studio tab. The Dashboard tab will now show the
            post-scenario numbers without any 'scenario loaded' hint."""
            global ACTIVE_SCENARIO_NAME, _LATEST_REWARD
            ACTIVE_SCENARIO_NAME = scenario_name_value
            _LATEST_REWARD = None  # new episode, new reward
            report = run_scenario_with_diff(scenario_name_value, int(seed))
            metric_date = report["target_date"]

            # Refresh the dashboard-side views
            cards, brand_rows, channel_rows, config_rows, story_rows, logs = refresh_everything(metric_date)

            # Scenario-studio renderings
            story_html_val = render_storybook_acts(scenario_name_value, report)
            diff_html_val = render_scenario_diff(report)
            pipeline_html_val = render_pipeline_state(report["pipeline_state"])
            dag_html_val = render_pipeline_dag(scenario_name_value)

            # Short status line
            if scenario_name_value == "Baseline":
                status_msg = "Baseline re-seeded. Dashboard is clean."
            else:
                touched = ", ".join(report["touched_tables"]) or "no tables"
                status_msg = (
                    f"Scenario **{scenario_name_value}** applied. "
                    f"Changes in: {touched}. Now go to the **Dashboard** tab."
                )

            return (
                metric_date, cards, brand_rows, channel_rows, config_rows,
                story_rows, logs, status_msg, build_status_strip(scenario_name_value, metric_date),
                story_html_val, diff_html_val, pipeline_html_val,
                scenario_name_value,
                dag_html_val,
            )

        load_btn.click(
            handle_load_scenario,
            inputs=[scenario_name, seed_input],
            outputs=[
                overview_date, metric_cards, brand_table, channel_table,
                config_table, spotlight_table, scenario_actions, status_bar, scenario_status_html,
                storybook_html, scenario_diff_html, pipeline_state_html,
                active_scenario,
                studio_dag_html,
            ],
        )

        # Node-click handlers: each button shows that node's table contents in a panel below the DAG
        node_app_btn.click(
            lambda: "<div class='table-panel-empty'>The App is the event source — orders stream in. There is no table to inspect at this stage; the Raw table is the first materialization.</div>",
            outputs=[studio_table_panel_html],
        )
        node_raw_btn.click(
            lambda: render_table_contents_panel("orders"),
            outputs=[studio_table_panel_html],
        )
        node_config_btn.click(
            lambda: render_table_contents_panel("channel_brand_config"),
            outputs=[studio_table_panel_html],
        )
        node_agg_btn.click(
            lambda: render_table_contents_panel("channel_daily_metrics"),
            outputs=[studio_table_panel_html],
        )
        node_dash_btn.click(
            lambda: "<div class='table-panel-empty'>The Dashboard is the presentation layer — it reads from the Aggregated table. Head to the Dashboard tab to see the rendered KPIs.</div>",
            outputs=[studio_table_panel_html],
        )

        def handle_rebuild(metric_date: str):
            msg = refresh_aggregation()
            cards, brands, channels, configs, story_rows, logs = refresh_everything(metric_date)
            pipeline_html = render_pipeline_state(get_pipeline_state_snapshot())
            return (
                msg, cards, brands, channels, configs, story_rows, logs,
                get_table_counts(), pipeline_html, build_status_strip(ACTIVE_SCENARIO_NAME, metric_date),
            )

        rebuild_btn.click(
            handle_rebuild,
            inputs=[overview_date],
            outputs=[
                sql_status, metric_cards, brand_table, channel_table, config_table,
                spotlight_table, scenario_actions, table_counts,
                pipeline_state_html, scenario_status_html,
            ],
        )

        def handle_assistant(message: str, history, metric_date: str):
            history = normalize_chat_history(history)
            try:
                new_history, answer, trace_text, latest_text = llm_assistant(message, history, metric_date)
                trace_html_val = render_tool_trace_html(trace_text)
                return new_history, trace_text, latest_text, "", trace_html_val
            except Exception as exc:
                fallback_history = history + [
                    {"role": "user", "content": message},
                    {"role": "assistant", "content": f"I hit an app-side error while answering: {exc}"},
                ]
                err_trace = json.dumps({"error": str(exc)}, indent=2)
                return (
                    fallback_history, err_trace, "{}", "",
                    f"<div class='trace-empty'>Error: {exc}</div>",
                )

        send_btn.click(
            handle_assistant,
            inputs=[prompt_box, chatbot, overview_date],
            outputs=[chatbot, tool_trace, latest_result, prompt_box, tool_trace_html],
        )
        prompt_box.submit(
            handle_assistant,
            inputs=[prompt_box, chatbot, overview_date],
            outputs=[chatbot, tool_trace, latest_result, prompt_box, tool_trace_html],
        )

        def inspect_table(table_name_value: str):
            """Show the table's CREATE/aggregation SQL in the reference pane,
            and put a SAFE preview query in the editable pane so running it
            by accident does not mutate state."""
            columns, rows = get_table_preview(table_name_value)
            safe_query = f"select * from {table_name_value} limit 20"
            return (
                get_table_sql(table_name_value),       # reference (read-only look)
                safe_query,                            # editable — safe default
                gr.update(headers=columns, value=rows),
                get_pipeline_log_rows(),
                get_table_counts(),
            )

        inspect_table_btn.click(
            inspect_table,
            inputs=[table_name],
            outputs=[table_sql, sql_runner, table_preview, pipeline_logs, table_counts],
        )

        def execute_sql(query: str, current_table: str):
            """Run the query, then reset the editor to a harmless preview of the
            currently-selected table so subsequent accidental clicks are safe."""
            result = tool_run_sql(query)
            columns = result.get("columns") or ["result"]
            rows = result.get("rows") or [["Statement executed."]]
            safe_query = f"select * from {current_table} limit 20"
            return (
                "Query executed. Editor reset to a safe preview.",
                gr.update(headers=columns, value=rows),
                get_pipeline_log_rows(),
                get_table_counts(),
                safe_query,
            )

        execute_sql_btn.click(
            execute_sql,
            inputs=[sql_runner, table_name],
            outputs=[sql_status, table_preview, pipeline_logs, table_counts, sql_runner],
        )

        # Initial paint
        if not DB_PATH.exists():
            initialize_database(DEFAULT_SEED)
        cards, brands, channels, configs, story_rows, logs = refresh_everything(initial_date)
        metric_cards.value = cards
        brand_table.value = brands
        channel_table.value = channels
        config_table.value = configs
        spotlight_table.value = story_rows
        scenario_actions.value = logs
        table_counts.value = get_table_counts()
        sql_text, safe_initial_sql, preview_update, pipeline_log_rows, count_rows = inspect_table("orders")
        table_sql.value = sql_text
        sql_runner.value = safe_initial_sql
        table_preview.headers = preview_update["headers"]
        table_preview.value = preview_update["value"]
        pipeline_logs.value = pipeline_log_rows
        table_counts.value = count_rows

    return demo


def main() -> None:
    if not DB_PATH.exists():
        initialize_database(DEFAULT_SEED)
    demo = build_demo()
    demo.launch(
        server_name="127.0.0.1",
        server_port=SERVER_PORT,
        share=False,
        css=APP_CSS,
        theme=gr.themes.Base(),
    )


if __name__ == "__main__":
    main()
