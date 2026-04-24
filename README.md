# FoodOps Incident Lab

This sandbox is now a guided **food brand x delivery channel** environment rather than a generic KPI table demo.

It is designed to feel like a real cloud-kitchen / food delivery ops cockpit:

```text
brands
channels
channel_brand_config
orders
  -> channel_daily_metrics
  -> Control Room + Assistant
```

## Why this version exists

The old single-table sandbox was useful for early thinking, but the UI and world model were getting too abstract.

This redesign focuses on:

1. intuitive UI first
2. realistic Indian food brand / aggregator context
3. incidents that can be either business-real or data-real
4. KPI explanations that sound like operations analysis, not formula dumps

## Data model

### Raw tables

- `brands`
  - brand catalog such as Behrouz Biryani, Oven Story Pizza, Faasos
- `channels`
  - delivery / commerce channels such as Zomato, Swiggy, ONDC, Own App
- `channel_brand_config`
  - the important many-to-many ops table:
    - `is_live`
    - `promo_active`
    - `promo_discount_pct`
    - `effective_commission_pct`
    - `menu_last_synced_at`
    - `notes`
- `orders`
  - order facts with:
    - `ordered_at`
    - `ingested_at`
    - `brand_id`
    - `channel_id`
    - `gross_amount`
    - `discount_amount`
    - `delivered_amount`
    - `refund_amount`
    - `status`

### Aggregate table

- `channel_daily_metrics`
  - daily brand x channel rollup used by the UI and assistant

## KPI layer

The lab keeps the KPI set intentionally small but meaningful:

- `total_orders`
- `gross_sales`
- `discount_amount`
- `net_sales`
- `net_payout`
- `aov`
- `brand_channel_coverage`

These are not treated as just formulas. The UI and assistant use them as business signals:

- orders up + payout flat -> margin pressure
- discount up -> demand may be bought, not organic
- payout down with healthy sales -> commissions may be the issue
- one channel dropping while another stays healthy -> likely config / channel issue

## Main tabs

### 1. Control Room

The high-level read:

- KPI cards
- brand summary
- channel summary
- a “where the story lives” table for top brand-channel combinations

### 2. Network Map

The live brand x channel config matrix:

- which combos are live
- which promos are active
- commission settings
- menu sync freshness

### 3. Scenario Studio

The guided workflow:

1. reset baseline
2. pick a scenario
3. load it into the same network
4. inspect the resulting story

### 4. Assistant

A grounded analyst-style assistant with tool trace.

Right now it uses a deterministic heuristic reasoning path so the lab still works even when external model credits are unavailable.

## Built-in scenarios

- `Baseline`
- `Stuck Promo After Campaign End`
- `Brand Paused On One Channel`
- `Commission Drift Hiding Margin Pressure`
- `Menu Sync Staleness`
- `Compound: Promo + Menu Sync`

These are meant to be intuitive:

- one brand on one channel can be wrong while everything else stays healthy
- some scenarios are real business issues, not pipeline failures
- the compound scenario is intentionally messy because that is where the benchmark gets interesting

## Running locally

```bash
cd /Users/priyankajain/Documents/scaler_project
PYTHONPATH=/Users/priyankajain/Documents/scaler_project \
  /Users/priyankajain/Documents/scaler_project/storeops_env/.venv/bin/python \
  /Users/priyankajain/Documents/scaler_project/dataops_incident_lab/order_dashboard_app.py
```

Expected local URL:

```text
http://127.0.0.1:7863
```

## Current design stance

This version is optimized for:

- clarity of story
- realism of incidents
- ease of demo
- a better foundation for future training / OpenEnv wrapping

It is **not** yet the final benchmark environment.

Still pending:

- stricter scenario-spec vs agent-observation separation
- richer seeded scenario families
- explicit tool budgets / reward logic
- a trained policy replacing the heuristic assistant

## Short summary

This is now a cleaner FoodOps incident sandbox: a brand x channel control room where promos, commissions, live listings, and sync drift can all change the KPI story in believable ways without overwhelming the UI.
