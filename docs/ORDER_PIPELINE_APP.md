# Order Pipeline App

This document explains what we are trying to build with `order_dashboard_app.py`
and why it matters for the larger DataOps Incident Gym project.

## Goal

We want a small, realistic app that shows the full path from business activity
to dashboard KPI:

```text
business/app behavior
  -> raw orders table
  -> aggregate table
  -> dashboard metrics
```

The important point is that the dashboard should not change magically. It
should change only after the pipeline runs.

## Why We Are Building This

The main OpenEnv environment is about incident response: a model sees a broken
business KPI, investigates data systems, applies the right fix, and restores
the dashboard.

This app is the simplest visual version of that world.

It gives us:

- a concrete raw table: `orders`
- a concrete aggregate table: `order_daily_metrics`
- dashboard metrics that read from the aggregate table
- a visible pipeline step between raw data and dashboard
- a place to simulate KPI manipulation and pipeline failures

So instead of only saying "revenue changed because of data," we can show the
actual flow end to end.

## Core Idea

The app starts empty.

Before the user runs anything from the pipeline tab:

- `orders` has zero rows
- `order_daily_metrics` has zero rows
- dashboard tabs show no real data yet

This is intentional. It makes the pipeline actions meaningful.

The user must explicitly:

1. generate raw table SQL
2. execute it to populate `orders`
3. generate aggregate SQL
4. execute it to populate `order_daily_metrics`
5. view the dashboard after the aggregate exists

That mirrors how real systems work better than preloading tables on startup.

## What The App Has

### 1. Data Pipeline Tab

This is the operational console.

It now has three main buttons:

- `Generate Raw Table`
- `Generate Agg Table`
- `Execute Query`

Behavior:

- `Generate Raw Table` fills the SQL editor with default SQL that creates the
  raw `orders` table and inserts synthetic order rows.
- `Generate Agg Table` fills the SQL editor with default SQL that creates or
  rebuilds `order_daily_metrics` from `orders`.
- `Execute Query` runs whatever is currently in the SQL editor.

This means the user can also modify the SQL and run custom pipeline logic such
as:

- `ALTER TABLE`
- `SELECT`
- `CREATE TABLE`
- `INSERT`
- `DELETE`
- aggregate rebuild SQL

So the tab is not just a toy button flow. It is a small pipeline SQL console.

### 2. Metric Manipulator

Below the pipeline console, we still keep the metric manipulation workflow.

Its job is to create controlled business changes for a single day, such as:

- net AOV drop
- gross AOV increase
- discount amount increase

The manipulator generates exact raw order rows, then upserts those rows into
`orders`, and finally rebuilds `order_daily_metrics`.

This is useful because it lets us simulate business-side KPI movement in a
structured way without manually editing all order rows.

### 3. Dashboard Views

The app also has tabs that show:

- aggregated metrics
- raw orders
- metric knowledge graph
- aggregate SQL
- custom query view
- user-facing dashboard view

These tabs help us inspect each layer of the system from different angles.

## Data Model

### Raw Table

`orders`

Columns:

- `order_ts`
- `order_id`
- `cart_amount`
- `discount_amount`
- `total_amount`
- `order_status`
- `refund_created`
- `refund_created_at`
- `payment_method`
- `city`

This table represents order-level business events.

### Aggregate Table

`order_daily_metrics`

Columns include:

- `metric_date`
- `total_orders`
- `delivered_orders`
- `cancelled_orders`
- `refund_orders`
- `gross_sales`
- `discount_amount`
- `delivered_sales`
- `refunded_amount`
- `net_sales`
- `aov`
- `refund_rate`
- `cancellation_rate`

This table represents the dashboard-ready daily KPI layer.

## End-to-End Flow

### Current intended flow

```text
Generate Raw Table
  -> execute SQL
  -> orders gets populated

Generate Agg Table
  -> execute SQL
  -> order_daily_metrics gets populated

Dashboard
  -> reads aggregate table
```

### Metric manipulation flow

```text
choose a date and metric change
  -> generate synthetic rows
  -> upsert rows into orders
  -> rebuild order_daily_metrics
  -> dashboard reflects the new KPI pattern
```

## Why This Helps The RL / OpenEnv Story

This app gives us a miniature world where an agent can eventually learn:

- raw data and dashboard data are different layers
- aggregate rebuilds are explicit actions
- KPI changes can come from either business behavior or pipeline logic
- SQL and pipeline actions affect what the dashboard shows

That makes it much easier to justify the larger DataOps environment:

```text
business KPI problem
  -> inspect dashboard
  -> inspect raw / aggregate tables
  -> inspect pipeline logic
  -> run the right fix
  -> verify restored KPI
```

## What We Are Trying To Demonstrate

At a product/demo level, we are trying to show:

1. business metrics live on top of a pipeline
2. raw data creation and aggregate creation are separate steps
3. dashboard metrics depend on aggregate tables, not directly on raw events
4. a controlled manipulation of raw data changes the dashboard only after the
   pipeline runs
5. this gives us the foundation for DataOps incident investigation and agentic
   policy learning

## What Comes Next

The most natural next improvements are:

1. add a pipeline logs tab
2. store pipeline run history in DuckDB and/or a log file
3. show a simple DAG:
   `orders -> order_daily_metrics -> dashboard`
4. add data quality checks
5. connect this app more directly to incident scenarios in the OpenEnv demo

## One-Line Summary

We are building a small end-to-end data app where business events create raw
rows, pipeline SQL builds aggregate metrics, the dashboard reads those metrics,
and nothing appears until the pipeline is explicitly run.
