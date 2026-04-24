"""Gymnasium-compatible environment for FoodOps ambiguity-first incidents.

V2: Supports compound episodes, schema drift, and checklist-based resolution.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import random
from typing import Any

try:  # pragma: no cover - optional dependency
    import gymnasium as gym
except Exception:  # pragma: no cover
    class _FallbackEnv:
        metadata: dict[str, Any] = {}

    class _FallbackGym:
        Env = _FallbackEnv

    gym = _FallbackGym()  # type: ignore[assignment]

from .primitives import WorldState
from .reward import (
    BUDGET_EXCEEDED_REWARD,
    FORBIDDEN_TOOLS,
    MALFORMED_ACTION_REWARD,
    READ_TOOLS,
    TERMINAL_TOOLS,
    WRITE_TOOLS,
    ESCALATION_TOOLS,
    compute_step_breakdown,
)
from .scenarios import (
    BRANDS,
    CHANNELS,
    ScenarioInstance,
    sample_scenario,
)


MAX_STEPS = 12
MAX_INVALID_ACTION_STREAK = 3
DEFAULT_NOW = datetime(2026, 4, 24, 10, 0, 0)


@dataclass(frozen=True)
class PairState:
    brand: str
    channel: str


class FoodOpsEnv(gym.Env):  # type: ignore[misc]
    """Compound-aware environment with schema drift and checklist-based resolution."""

    metadata = {"render_modes": ["ansi"]}

    def __init__(self) -> None:
        self.rng = random.Random()
        self.world: WorldState = {}
        self.anomaly: ScenarioInstance | None = None
        self.trajectory: list[dict[str, Any]] = []
        self.phase_flags: dict[str, Any] = {
            "has_read": False,
            "has_acted": False,
            "phase_bonus_awarded": False,
            "disambiguated": False,
            "disambiguated_ids": set(),
            "resolved_tools": set(),
            "drift_bonus_awarded": False,
        }
        self.step_count = 0
        self.done = False
        self.terminated_reason = ""
        self.invalid_action_streak = 0
        self.last_tool_call: dict[str, Any] | None = None
        self.last_tool_result: dict[str, Any] | None = None
        self._pending_resolutions: set[str] = set()

    def reset(self, seed: int | None = None, **_: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        if seed is not None:
            self.rng.seed(seed)
        scenario_rng = random.Random(seed) if seed is not None else self.rng
        self.step_count = 0
        self.done = False
        self.terminated_reason = ""
        self.invalid_action_streak = 0
        self.trajectory = []
        self.phase_flags = {
            "has_read": False,
            "has_acted": False,
            "phase_bonus_awarded": False,
            "disambiguated": False,
            "disambiguated_ids": set(),
            "resolved_tools": set(),
            "drift_bonus_awarded": False,
        }
        self.last_tool_call = None
        self.last_tool_result = None

        self.world = self._build_base_world()
        self.anomaly = sample_scenario(scenario_rng)

        # Inject all primitives
        for prim, params in zip(self.anomaly.primitives, self.anomaly.primitive_params):
            prim.inject_fn(self.world, params, self.rng)

        # Store AOV baseline for matchers that need it
        for params in self.anomaly.primitive_params:
            b, c = params["brand"], params["channel"]
            params["_baseline_aov"] = self.world["baseline_reference"][(b, c)]["aov"]

        self.world["active_pair"] = (
            self.anomaly.primitive_params[0]["brand"],
            self.anomaly.primitive_params[0]["channel"],
        )

        # Track which resolution tools are still pending for compound episodes
        self._pending_resolutions = {
            item["resolution_tool"] for item in self.anomaly.resolution_checklist
        }

        obs = self._observation()
        info = {
            "scenario_id": self.anomaly.scenario_id,
            "anomaly_category": self.anomaly.anomaly_category,
            "story_seed": self.anomaly.story_seed,
            "difficulty": self.anomaly.difficulty,
            "is_compound": self.anomaly.is_compound,
            "num_primitives": len(self.anomaly.primitives),
            "has_drift": self.anomaly.drift is not None,
            "scenario_params": deepcopy(self.anomaly.primitive_params[0]),
        }
        return obs, info

    def step(self, action: dict[str, Any]) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        if self.anomaly is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        if self.done:
            return self._observation(), 0.0, True, False, {"terminated_reason": self.terminated_reason}

        self.step_count += 1

        # ---- Schema drift injection ----
        if (
            self.anomaly.drift
            and not self.anomaly.drift_applied
            and self.step_count >= self.anomaly.drift.trigger_step
        ):
            brand = self.anomaly.primitive_params[0]["brand"]
            channel = self.anomaly.primitive_params[0]["channel"]
            self.anomaly.drift.apply_fn(self.world, brand, channel, self.rng)
            self.anomaly.drift_applied = True

        # ---- Malformed action handling ----
        if not isinstance(action, dict) or "tool" not in action:
            self.invalid_action_streak += 1
            malformed_result = {
                "error": "malformed action",
                "raw": action.get("raw", "") if isinstance(action, dict) else str(action),
                "invalid_action_streak": self.invalid_action_streak,
            }
            self.last_tool_call = action if isinstance(action, dict) else {"tool": None, "args": {}}
            self.last_tool_result = malformed_result
            reward_breakdown = compute_step_breakdown(
                action=None, tool_result=malformed_result, scenario=self.anomaly,
                trajectory=self.trajectory, phase_flags=self.phase_flags, malformed=True,
            )
            terminated = self.invalid_action_streak >= MAX_INVALID_ACTION_STREAK
            if terminated:
                self.terminated_reason = "too_many_invalid_actions"
            self.trajectory.append({
                "tool": "__malformed__", "args": {}, "result": deepcopy(malformed_result),
                "step_reward": reward_breakdown["total"], "reward_breakdown": reward_breakdown,
            })
            self.done = terminated
            info = {"reward_breakdown": reward_breakdown, "terminated_reason": self.terminated_reason,
                    "invalid_action_streak": self.invalid_action_streak}
            return self._observation(), reward_breakdown["total"], terminated, False, info

        tool_name = str(action.get("tool"))
        args = action.get("args") if isinstance(action.get("args"), dict) else {}
        normalized_action = {"tool": tool_name, "args": args}
        tool_result = self._dispatch_tool(tool_name, args)

        # ---- Unknown tool handling ----
        if tool_result.get("error") == "unknown tool":
            self.invalid_action_streak += 1
            invalid_result = {"error": "unknown tool", "tool": tool_name,
                              "invalid_action_streak": self.invalid_action_streak}
            self.last_tool_call = normalized_action
            self.last_tool_result = invalid_result
            reward_breakdown = compute_step_breakdown(
                action=None, tool_result=invalid_result, scenario=self.anomaly,
                trajectory=self.trajectory, phase_flags=self.phase_flags, malformed=True,
            )
            terminated = self.invalid_action_streak >= MAX_INVALID_ACTION_STREAK
            if terminated:
                self.terminated_reason = "too_many_invalid_actions"
            self.trajectory.append({
                "tool": "__unknown__", "args": deepcopy(args), "result": deepcopy(invalid_result),
                "step_reward": reward_breakdown["total"], "reward_breakdown": reward_breakdown,
            })
            self.done = terminated
            info = {"reward_breakdown": reward_breakdown, "terminated_reason": self.terminated_reason,
                    "invalid_action_streak": self.invalid_action_streak}
            return self._observation(), reward_breakdown["total"], terminated, False, info

        self.invalid_action_streak = 0

        reward_breakdown = compute_step_breakdown(
            action=normalized_action, tool_result=tool_result, scenario=self.anomaly,
            trajectory=self.trajectory, phase_flags=self.phase_flags,
        )

        # ---- Terminal logic with compound support ----
        terminated = False
        truncated = False

        if tool_name in TERMINAL_TOOLS:
            if tool_name in FORBIDDEN_TOOLS:
                terminated = True
                self.terminated_reason = "forbidden_tool"
            else:
                self._pending_resolutions.discard(tool_name)
                if not self._pending_resolutions:
                    terminated = True
                    self.terminated_reason = "all_resolved"
                elif len(self._pending_resolutions) == 0:
                    terminated = True
                    self.terminated_reason = "terminal_action"
                else:
                    # Compound: first resolution done, more remain — don't terminate
                    pass

        if self.step_count >= MAX_STEPS and not terminated:
            budget_breakdown = compute_step_breakdown(
                action=normalized_action, tool_result=tool_result, scenario=self.anomaly,
                trajectory=self.trajectory, phase_flags=self.phase_flags, budget_exceeded=True,
            )
            reward_breakdown = self._merge_breakdowns(reward_breakdown, budget_breakdown)
            truncated = True
            self.terminated_reason = "budget_exceeded"

        self.last_tool_call = normalized_action
        self.last_tool_result = tool_result
        self.trajectory.append({
            "tool": tool_name, "args": deepcopy(args), "result": deepcopy(tool_result),
            "step_reward": reward_breakdown["total"], "reward_breakdown": reward_breakdown,
        })

        self.done = terminated or truncated
        obs = self._observation()
        info = {
            "reward_breakdown": reward_breakdown,
            "terminated_reason": self.terminated_reason,
            "scenario_id": self.anomaly.scenario_id,
            "pending_resolutions": len(self._pending_resolutions),
        }
        return obs, reward_breakdown["total"], terminated, truncated, info

    def render(self) -> str:
        if self.anomaly is None:
            return "FoodOpsEnv not initialized."
        lines = [
            f"Complaint: {self.anomaly.user_complaint}",
            f"Scenario: {self.anomaly.scenario_id} (difficulty={self.anomaly.difficulty})",
            f"Done: {self.done} ({self.terminated_reason or 'active'})",
        ]
        for idx, record in enumerate(self.trajectory, start=1):
            lines.append(
                f"{idx}. {record['tool']}({json.dumps(record['args'], sort_keys=True)}) -> "
                f"{record['result']} | reward={record['step_reward']:+.2f}"
            )
        return "\n".join(lines)

    def close(self) -> None:
        return None

    def get_state(self) -> dict[str, Any]:
        if self.anomaly is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        return {
            "world": deepcopy(self.world),
            "anomaly": {
                "scenario_id": self.anomaly.scenario_id,
                "params": deepcopy(self.anomaly.primitive_params[0]),
                "user_complaint": self.anomaly.user_complaint,
                "story_seed": self.anomaly.story_seed,
                "difficulty": self.anomaly.difficulty,
                "is_compound": self.anomaly.is_compound,
                "num_primitives": len(self.anomaly.primitives),
                "has_drift": self.anomaly.drift is not None,
                "drift_applied": self.anomaly.drift_applied if self.anomaly.drift else False,
            },
            "trajectory": deepcopy(self.trajectory),
            "phase_flags": {
                k: list(v) if isinstance(v, set) else v
                for k, v in self.phase_flags.items()
            },
            "step_count": self.step_count,
            "done": self.done,
            "terminated_reason": self.terminated_reason,
            "invalid_action_streak": self.invalid_action_streak,
            "pending_resolutions": len(self._pending_resolutions),
        }

    # ---- Observation ----

    def _observation(self) -> dict[str, Any]:
        assert self.anomaly is not None
        brand = self.anomaly.primitive_params[0]["brand"]
        channel = self.anomaly.primitive_params[0]["channel"]
        dashboard_kpis = self._summarize_pair(brand, channel, range_name="current")
        dashboard_kpis["brand"] = brand
        dashboard_kpis["channel"] = channel

        obs = {
            "user_complaint": self.anomaly.user_complaint,
            "dashboard_kpis": dashboard_kpis,
            "last_tool_call": deepcopy(self.last_tool_call),
            "last_tool_result": deepcopy(self.last_tool_result),
            "steps_remaining": max(0, MAX_STEPS - self.step_count),
            "tools_available": [
                "get_kpi_summary",
                "check_config",
                "check_pipeline_freshness",
                "inspect_recent_orders",
                "force_menu_sync",
                "restart_pipeline",
                "escalate_to_finance",
                "escalate_to_ops",
                "update_commission_rate",
                "toggle_promo",
                "update_menu_prices",
            ],
        }

        # Hint for compound episodes: how many issues remain
        if self.anomaly.is_compound:
            obs["pending_resolutions"] = len(self._pending_resolutions)

        return obs

    # ---- Tool dispatch ----

    def _dispatch_tool(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        if tool_name == "get_kpi_summary":
            return self._tool_get_kpi_summary(
                brand=args.get("brand"), channel=args.get("channel"),
                range_name=str(args.get("range", "current")),
            )
        if tool_name == "check_config":
            return self._tool_check_config(args.get("brand"), args.get("channel"))
        if tool_name == "check_pipeline_freshness":
            return self._tool_check_pipeline_freshness(
                table_name=str(args.get("table_name", "orders")),
                brand=args.get("brand"), channel=args.get("channel"),
            )
        if tool_name == "inspect_recent_orders":
            return self._tool_inspect_recent_orders(
                brand=args.get("brand"), channel=args.get("channel"),
                limit=int(args.get("limit", 10)),
            )
        if tool_name == "force_menu_sync":
            return self._tool_force_menu_sync(args.get("brand"), args.get("channel"))
        if tool_name == "restart_pipeline":
            return self._tool_restart_pipeline(str(args.get("pipeline_name", "unknown")))
        if tool_name == "escalate_to_finance":
            return {"status": "escalated", "team": "finance", "reason": args.get("reason", "")}
        if tool_name == "escalate_to_ops":
            return {"status": "escalated", "team": "ops", "reason": args.get("reason", "")}
        if tool_name == "update_commission_rate":
            return {"status": "forbidden", "message": "Commission changes are owned by finance/BD."}
        if tool_name == "toggle_promo":
            return {"status": "forbidden", "message": "Promo flags are not agent-owned in this benchmark."}
        if tool_name == "update_menu_prices":
            return {"status": "forbidden", "message": "Menu pricing is not agent-owned in this benchmark."}
        return {"error": "unknown tool"}

    # ---- Tool implementations ----

    def _tool_get_kpi_summary(self, brand: str | None, channel: str | None, range_name: str) -> dict[str, Any]:
        if range_name not in {"current", "baseline"}:
            return {"error": f"unknown range: {range_name}"}
        source = self.world["current_reference"] if range_name == "current" else self.world["baseline_reference"]
        summary = self._aggregate_pairs(source, brand=brand, channel=channel)
        return {"range": range_name, "brand": brand, "channel": channel, "summary": summary}

    def _tool_check_config(self, brand: str | None, channel: str | None) -> dict[str, Any]:
        if not brand or not channel:
            return {"error": "brand and channel are required"}
        config = deepcopy(self.world["channel_brand_config"][(brand, channel)])
        config["brand"] = brand
        config["channel"] = channel
        config["menu_last_synced_at"] = config["menu_last_synced_at"].isoformat()
        if config.get("promo_expected_to_end_at"):
            config["promo_expected_to_end_at"] = config["promo_expected_to_end_at"].isoformat()
        return config

    def _tool_check_pipeline_freshness(
        self, table_name: str, brand: str | None = None, channel: str | None = None,
    ) -> dict[str, Any]:
        if table_name == "menu_sync":
            if not brand or not channel:
                return {"error": "brand and channel are required for menu_sync freshness"}
            freshness_at = self.world["pipeline_freshness"]["menu_sync"][(brand, channel)]
            is_stale = freshness_at <= self.world["now"] - timedelta(hours=24)
            return {
                "table_name": table_name, "brand": brand, "channel": channel,
                "freshness_at": freshness_at.isoformat(), "is_stale": is_stale,
            }

        freshness_at = self.world["pipeline_freshness"].get(table_name)
        if freshness_at is None:
            return {"error": f"unknown table or pipeline: {table_name}"}
        is_stale = freshness_at <= self.world["now"] - timedelta(hours=24)
        return {"table_name": table_name, "freshness_at": freshness_at.isoformat(), "is_stale": is_stale}

    def _tool_inspect_recent_orders(self, brand: str | None, channel: str | None, limit: int) -> dict[str, Any]:
        if not brand or not channel:
            return {"error": "brand and channel are required"}
        rows = deepcopy(self.world["recent_orders"][(brand, channel)][: max(1, min(limit, 10))])
        return {"brand": brand, "channel": channel, "rows": rows}

    def _tool_force_menu_sync(self, brand: str | None, channel: str | None) -> dict[str, Any]:
        if not brand or not channel:
            return {"status": "failed", "message": "brand and channel are required"}
        freshness = self.world["pipeline_freshness"]["menu_sync"][(brand, channel)]
        self.world["pipeline_freshness"]["menu_sync"][(brand, channel)] = max(freshness, self.world["now"])
        self.world["channel_brand_config"][(brand, channel)]["menu_last_synced_at"] = self.world["now"]
        return {
            "status": "queued", "pipeline": "menu_sync",
            "brand": brand, "channel": channel, "queued_at": self.world["now"].isoformat(),
        }

    def _tool_restart_pipeline(self, pipeline_name: str) -> dict[str, Any]:
        return {"status": "queued", "pipeline_name": pipeline_name, "queued_at": self.world["now"].isoformat()}

    # ---- World building ----

    def _build_base_world(self) -> WorldState:
        world: WorldState = {
            "rng": self.rng,
            "now": DEFAULT_NOW,
            "brands": list(BRANDS),
            "channels": list(CHANNELS),
            "channel_brand_config": {},
            "baseline_reference": {},
            "current_reference": {},
            "recent_orders": {},
            "pipeline_freshness": {},
        }

        world["pipeline_freshness"]["orders"] = DEFAULT_NOW - timedelta(minutes=self.rng.randint(10, 40))
        world["pipeline_freshness"]["channel_daily_metrics"] = DEFAULT_NOW - timedelta(minutes=self.rng.randint(10, 40))
        world["pipeline_freshness"]["menu_sync"] = {}

        for brand_idx, brand in enumerate(BRANDS):
            for channel_idx, channel in enumerate(CHANNELS):
                pair = (brand, channel)
                total_orders = 70 + brand_idx * 9 + channel_idx * 7 + self.rng.randint(0, 14)
                aov = 320 + brand_idx * 24 + channel_idx * 18 + self.rng.randint(-10, 15)
                gross_sales = round(total_orders * aov, 2)
                discount_rate = 0.08 + (brand_idx * 0.005) + (channel_idx * 0.007)
                discount_amount = round(gross_sales * discount_rate, 2)
                refund_amount = round(gross_sales * (0.01 + channel_idx * 0.004), 2)
                net_sales = round(gross_sales - discount_amount - refund_amount, 2)
                baseline_commission_pct = round(18.0 + channel_idx * 3.5 + brand_idx * 0.4, 2)
                net_payout = round(net_sales * (1.0 - baseline_commission_pct / 100.0), 2)
                baseline = {
                    "total_orders": total_orders,
                    "gross_sales": gross_sales,
                    "discount_amount": discount_amount,
                    "refund_amount": refund_amount,
                    "net_sales": net_sales,
                    "net_payout": net_payout,
                    "aov": round(gross_sales / total_orders, 2),
                    "effective_commission_pct": baseline_commission_pct,
                }
                world["baseline_reference"][pair] = baseline
                world["current_reference"][pair] = deepcopy(baseline)
                world["channel_brand_config"][pair] = {
                    "is_live": True,
                    "promo_active": False,
                    "promo_discount_pct": 0.0,
                    "promo_expected_to_end_at": None,
                    "promo_overdue_days": 0,
                    "effective_commission_pct": baseline_commission_pct,
                    "baseline_commission_pct": baseline_commission_pct,
                    "menu_last_synced_at": DEFAULT_NOW - timedelta(hours=self.rng.randint(1, 6)),
                }
                world["pipeline_freshness"]["menu_sync"][pair] = world["channel_brand_config"][pair]["menu_last_synced_at"]
                world["recent_orders"][pair] = self._build_recent_orders(brand, channel, baseline, self.rng)
        return world

    def _build_recent_orders(
        self, brand: str, channel: str, baseline: dict[str, Any], rng: random.Random,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        sample_orders = min(12, baseline["total_orders"])
        gross_per_order = baseline["gross_sales"] / sample_orders
        discount_per_order = baseline["discount_amount"] / sample_orders
        net_sales_per_order = baseline["net_sales"] / sample_orders
        net_payout_per_order = baseline["net_payout"] / sample_orders
        for idx in range(sample_orders):
            ordered_at = DEFAULT_NOW - timedelta(hours=idx * 2)
            rows.append({
                "order_id": f"{brand[:3].upper()}-{channel[:3].upper()}-{idx:03d}",
                "brand": brand, "channel": channel,
                "ordered_at": ordered_at.isoformat(),
                "gross_amount": round(gross_per_order, 2),
                "discount_amount": round(discount_per_order, 2),
                "net_sales": round(net_sales_per_order, 2),
                "net_payout": round(net_payout_per_order, 2),
                "customer_segment": "repeat" if idx % 3 else "new",
            })
        return rows

    def _summarize_pair(self, brand: str, channel: str, range_name: str) -> dict[str, Any]:
        source = self.world["current_reference"] if range_name == "current" else self.world["baseline_reference"]
        summary = deepcopy(source[(brand, channel)])
        if range_name == "current":
            summary["current_commission_pct"] = self.world["channel_brand_config"][(brand, channel)]["effective_commission_pct"]
        return summary

    def _aggregate_pairs(
        self, source: dict[tuple[str, str], dict[str, Any]], brand: str | None, channel: str | None,
    ) -> dict[str, Any]:
        matching = [
            row for (row_brand, row_channel), row in source.items()
            if (brand is None or row_brand == brand) and (channel is None or row_channel == channel)
        ]
        if not matching:
            return {}
        total_orders = sum(row["total_orders"] for row in matching)
        gross_sales = round(sum(row["gross_sales"] for row in matching), 2)
        discount_amount = round(sum(row["discount_amount"] for row in matching), 2)
        refund_amount = round(sum(row.get("refund_amount", 0) for row in matching), 2)
        net_sales = round(sum(row["net_sales"] for row in matching), 2)
        net_payout = round(sum(row["net_payout"] for row in matching), 2)
        weighted_commission = round(sum(row["effective_commission_pct"] for row in matching) / len(matching), 2)
        return {
            "total_orders": total_orders,
            "gross_sales": gross_sales,
            "discount_amount": discount_amount,
            "refund_amount": refund_amount,
            "net_sales": net_sales,
            "net_payout": net_payout,
            "aov": round(gross_sales / max(total_orders, 1), 2),
            "effective_commission_pct": weighted_commission,
        }

    @staticmethod
    def _merge_breakdowns(primary: dict[str, float], extra: dict[str, float]) -> dict[str, float]:
        merged = {key: float(primary.get(key, 0.0)) + float(extra.get(key, 0.0)) for key in set(primary) | set(extra)}
        merged["total"] = sum(merged.get(key, 0.0) for key in merged if key != "total")
        return merged
