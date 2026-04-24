# Instructions for Codex: generating FoodOps Incident Lab scenarios

You are authoring scenario JSON files for the FoodOps Incident Lab.

## Required reading first
1. `scenarios/scenario.schema.json` — JSON schema every scenario must validate against.
2. `scenarios/stuck_promo_after_campaign_end.json` — the reference scenario. Match its structure exactly.
3. `scenarios/audit_scenarios.py` — run this after authoring; every scenario must pass with zero issues.

## What to generate

Create one file per scenario in `scenarios/`. Filename = `scenario_id` + `.json`.

Generate these 10 scenarios for this round:

| scenario_id | family | difficulty | is_null_incident |
|---|---|---|---|
| stuck_promo_after_campaign_end | business_config_drift | medium | false |
| brand_paused_on_one_channel | business_config_drift | medium | false |
| menu_sync_staleness | business_config_drift | hard | false |
| commission_renegotiation_not_a_bug | external_business_reality | hard | true |
| stale_aggregate_single_channel | pipeline_failure | easy | false |
| partial_aggregate_refresh | pipeline_failure | medium | false |
| ondc_double_count | cross_table_join_bug | hard | false |
| orphan_brand_after_catalog_cleanup | cross_table_join_bug | hard | false |
| seasonal_delivery_slowdown | external_business_reality | medium | true |
| compound_promo_plus_menu_sync | compound | very_hard | false |

## Hard rules — do not violate

1. **Nothing in `world_state`, `ground_truth`, or `narrative_for_humans` may be referenced in `observation.user_complaint`.** The complaint is what the agent sees. Everything else is hidden. The audit script enforces this for common patterns — but you must also apply judgment.

2. **`user_complaint` must sound like something a non-technical ops lead would say.** It should contain a symptom and an ambient question ("is the pipeline broken or is this real?"), not a diagnosis. Think Slack message, not JIRA ticket.

3. **`observation` must contain only these keys:** `user_role`, `user_complaint`, `shown_kpis`, `shown_time_range`. Nothing else — the audit will reject additions.

4. **Every scenario must have at least one entry in `ground_truth.required_evidence`.** This drives tool_coverage and tool_order rewards. If a scenario has no required evidence, training can't score it.

5. **Null-incident scenarios (`is_null_incident=true`) must have `correct_terminal_action = submit_report_no_fix`** and must list any mutation (`refresh_aggregation`, etc.) in `incorrect_terminal_actions_with_reason`.

6. **Compound scenarios use a list for `true_root_cause`** (one string per true cause), not a single string. The reward function handles proportional scoring.

7. **Use placeholders `{{brand}}`, `{{channel}}`, `{{promo_discount_pct}}`, etc. for parameterized values.** These get resolved at reset() from the `parameterization` block. Do not hardcode names like "Behrouz Biryani" in ground truth — hardcode them in the `pool` arrays and reference via placeholders.

## Brand and channel pools to use

Brands pool: `["Behrouz Biryani", "Oven Story", "Faasos", "Sweet Truth", "The Good Bowl", "Lunchbox", "Mandarin Oak"]`
Channels pool: `["Swiggy", "Zomato", "ONDC", "Own App"]`

Not every scenario needs to support all brands/channels. For example, `ondc_double_count` only makes sense on ONDC, so `channel.pool = ["ONDC"]`.

## Validation step

After generating all 10, run:

```bash
python scenarios/audit_scenarios.py scenarios/
```

You must see `total issues: 0` before handing off. If any scenario has issues, fix them.

Also verify:
- Every file parses as JSON (no trailing commas, no comments outside narrative fields).
- `scenario_id` matches the filename.
- `scenario_family` is one of the allowlisted values in the schema.

## Tone guidance for user_complaint

Good examples:
- "Net payout for Behrouz on Swiggy looks way down this past week even though orders are steady. Discounts look bigger than expected. Is the pipeline broken or is this real?"
- "Why did Oven Story orders drop 60% on Zomato overnight? Swiggy looks fine. Is Zomato's API acting up?"
- "Average delivery days went from 3.2 to 4.1. Is our aggregation wrong?"

Bad examples (leaky):
- "The promo_active flag is stuck on for Behrouz-Swiggy." (names a column)
- "The aggregation job failed at 2am." (diagnoses the cause)
- "Schema drift on the channel_brand_config table." (gives the answer)

If you can solve the scenario from the complaint alone, rewrite it.
