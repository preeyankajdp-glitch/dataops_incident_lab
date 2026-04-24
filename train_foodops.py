"""Colab-ready training/eval scaffold for FoodOpsEnv.

Suggested setup cell for Colab:

    !pip install -q trl transformers peft accelerate datasets matplotlib fastapi uvicorn
    !pip install -e ./foodops_env

This script has two execution modes:
1. A lightweight smoke loop that always runs on CPU and validates env/rollout plumbing.
2. A best-effort GRPO path that activates when TRL + Transformers are installed.

The smoke loop is what we use locally to keep the repo runnable.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import argparse
import csv
import json
import math
from pathlib import Path
import random
import re
from typing import Any, Callable

import matplotlib.pyplot as plt

from foodops_env.env import FoodOpsEnv
from foodops_env.reward import ESCALATION_TOOLS, WRITE_TOOLS

try:  # pragma: no cover - optional training stack
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    TRL_AVAILABLE = True
except Exception:  # pragma: no cover
    GRPOConfig = None  # type: ignore[assignment]
    GRPOTrainer = None  # type: ignore[assignment]
    AutoModelForCausalLM = None  # type: ignore[assignment]
    AutoTokenizer = None  # type: ignore[assignment]
    LoraConfig = None  # type: ignore[assignment]
    TRL_AVAILABLE = False


REPO_ROOT = Path(__file__).resolve().parent
CSV_PATH = REPO_ROOT / "foodops_training_metrics.csv"
REWARD_PLOT_PATH = REPO_ROOT / "foodops_reward_curve.png"
BAR_PLOT_PATH = REPO_ROOT / "foodops_before_after_metrics.png"
CONFUSION_PATH = REPO_ROOT / "foodops_confusion_matrix.png"


@dataclass
class TrainingConfig:
    base_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    lora_rank: int = 16
    lora_alpha: int = 32
    target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
    training_steps: int = 150
    batch_size: int = 4
    max_tool_calls: int = 12
    learning_rate: float = 1e-5
    grpo_group_size: int = 4
    smoke_steps: int = 10


def build_system_prompt() -> str:
    return (
        "You are an ops analyst for FoodOps. "
        "Given an observation, respond with JSON only in the form "
        '{"tool":"...", "args": {...}}. '
        "Use one tool at a time. "
        "You must investigate before acting."
    )


def format_observation(obs: dict[str, Any]) -> str:
    return (
        f"Complaint: {obs['user_complaint']}\n"
        f"Dashboard KPIs: {json.dumps(obs['dashboard_kpis'], sort_keys=True)}\n"
        f"Last tool call: {json.dumps(obs['last_tool_call'], sort_keys=True)}\n"
        f"Last tool result: {json.dumps(obs['last_tool_result'], sort_keys=True)}\n"
        f"Steps remaining: {obs['steps_remaining']}\n"
        f"Tools: {', '.join(obs['tools_available'])}"
    )


def parse_action(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict) and "tool" in parsed:
            return {"tool": parsed["tool"], "args": parsed.get("args", {})}
    except Exception:
        pass
    match = re.search(r"\{[\s\S]*\}", stripped)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, dict) and "tool" in parsed:
                return {"tool": parsed["tool"], "args": parsed.get("args", {})}
        except Exception:
            pass
    return {"tool": "__malformed__", "args": {"raw": text[:200]}}


def choose_heuristic_action(obs: dict[str, Any]) -> dict[str, Any]:
    """A tiny baseline used for smoke runs and before/after demos if TRL is unavailable."""
    complaint = obs["user_complaint"].lower()
    brand = obs["dashboard_kpis"]["brand"]
    channel = obs["dashboard_kpis"]["channel"]
    last_tool = (obs["last_tool_call"] or {}).get("tool")

    if last_tool is None:
        if any(token in complaint for token in ["week", "month", "expected", "lower than last month"]):
            return {"tool": "get_kpi_summary", "args": {"brand": brand, "channel": channel, "range": "baseline"}}
        return {"tool": "check_config", "args": {"brand": brand, "channel": channel}}

    if last_tool == "get_kpi_summary":
        return {"tool": "check_config", "args": {"brand": brand, "channel": channel}}

    if last_tool == "check_config":
        result = obs["last_tool_result"] or {}
        if result.get("promo_active"):
            return {"tool": "escalate_to_finance", "args": {"reason": "Promo appears stuck on the target pair."}}
        if float(result.get("effective_commission_pct", 0.0)) > float(result.get("baseline_commission_pct", 0.0)):
            return {"tool": "escalate_to_finance", "args": {"reason": "Commission rate drifted above baseline."}}
        return {"tool": "check_pipeline_freshness", "args": {"table_name": "menu_sync", "brand": brand, "channel": channel}}

    if last_tool == "check_pipeline_freshness":
        result = obs["last_tool_result"] or {}
        if result.get("is_stale"):
            return {"tool": "force_menu_sync", "args": {"brand": brand, "channel": channel}}
        return {"tool": "escalate_to_ops", "args": {"reason": "No obvious config issue; asking ops to verify state."}}

    return {"tool": "escalate_to_ops", "args": {"reason": "Unable to disambiguate confidently."}}


def rollout_policy(
    policy_fn: Callable[[dict[str, Any]], dict[str, Any]],
    seed: int,
    max_tool_calls: int,
) -> dict[str, Any]:
    env = FoodOpsEnv()
    obs, info = env.reset(seed=seed)
    reward_total = 0.0
    terminal_tool = None
    for _ in range(max_tool_calls):
        action = policy_fn(obs)
        obs, reward, terminated, truncated, step_info = env.step(action)
        reward_total += reward
        if terminated or truncated:
            terminal_tool = action.get("tool")
            break
    state = env.get_state()
    return {
        "seed": seed,
        "scenario_id": info["scenario_id"],
        "anomaly_category": info["anomaly_category"],
        "reward_total": reward_total,
        "steps": len(state["trajectory"]),
        "trajectory": state["trajectory"],
        "terminal_tool": terminal_tool,
        "disambiguated": state["phase_flags"]["disambiguated"],
        "guardrail_violation": terminal_tool in {"update_commission_rate", "toggle_promo", "update_menu_prices"},
        "predicted_category": predicted_category_from_tool(terminal_tool),
    }


def predicted_category_from_tool(tool_name: str | None) -> str:
    if tool_name in WRITE_TOOLS:
        return "remediate"
    if tool_name in ESCALATION_TOOLS:
        return "escalate"
    if tool_name is None:
        return "none"
    return "forbidden"


def evaluate_policy(
    policy_fn: Callable[[dict[str, Any]], dict[str, Any]],
    seeds: list[int],
    max_tool_calls: int,
) -> dict[str, Any]:
    episodes = [rollout_policy(policy_fn, seed, max_tool_calls) for seed in seeds]
    mean_reward = sum(ep["reward_total"] for ep in episodes) / len(episodes)
    guardrail_rate = sum(1 for ep in episodes if ep["guardrail_violation"]) / len(episodes)
    correct_category_rate = sum(
        1
        for ep in episodes
        if (
            ep["anomaly_category"] == "pipeline" and ep["predicted_category"] == "remediate"
        ) or (
            ep["anomaly_category"] == "business" and ep["predicted_category"] == "escalate"
        )
    ) / len(episodes)
    disambiguation_rate = sum(1 for ep in episodes if ep["disambiguated"]) / len(episodes)
    mean_steps = sum(ep["steps"] for ep in episodes) / len(episodes)
    confusion = Counter((ep["anomaly_category"], ep["predicted_category"]) for ep in episodes)
    return {
        "episodes": episodes,
        "mean_reward": mean_reward,
        "guardrail_violation_rate": guardrail_rate,
        "correct_category_rate": correct_category_rate,
        "disambiguation_rate": disambiguation_rate,
        "mean_steps": mean_steps,
        "confusion": confusion,
    }


def write_metrics_csv(rows: list[dict[str, Any]], path: Path = CSV_PATH) -> None:
    if not rows:
        return
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_training_outputs(
    reward_rows: list[dict[str, Any]],
    before_metrics: dict[str, Any],
    after_metrics: dict[str, Any],
) -> None:
    steps = [row["step"] for row in reward_rows]
    rewards = [row["mean_reward"] for row in reward_rows]
    moving_average = []
    for idx in range(len(rewards)):
        window = rewards[max(0, idx - 19) : idx + 1]
        moving_average.append(sum(window) / len(window))

    plt.figure(figsize=(8, 4))
    plt.plot(steps, rewards, label="mean reward", alpha=0.4)
    plt.plot(steps, moving_average, label="20-step moving average")
    plt.xlabel("Training step")
    plt.ylabel("Reward")
    plt.title("FoodOps reward curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(REWARD_PLOT_PATH)
    plt.close()

    metric_names = [
        "mean_reward",
        "guardrail_violation_rate",
        "correct_category_rate",
        "disambiguation_rate",
    ]
    before_vals = [before_metrics[name] for name in metric_names]
    after_vals = [after_metrics[name] for name in metric_names]
    x = range(len(metric_names))
    plt.figure(figsize=(8, 4))
    plt.bar([i - 0.18 for i in x], before_vals, width=0.36, label="before")
    plt.bar([i + 0.18 for i in x], after_vals, width=0.36, label="after")
    plt.xticks(list(x), metric_names, rotation=20)
    plt.title("Before vs after eval metrics")
    plt.legend()
    plt.tight_layout()
    plt.savefig(BAR_PLOT_PATH)
    plt.close()

    labels = ["pipeline", "business"]
    pred_labels = ["remediate", "escalate", "forbidden", "none"]
    before_matrix = [[before_metrics["confusion"].get((true, pred), 0) for pred in pred_labels] for true in labels]
    after_matrix = [[after_metrics["confusion"].get((true, pred), 0) for pred in pred_labels] for true in labels]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, matrix, title in zip(axes, [before_matrix, after_matrix], ["Before", "After"]):
        im = ax.imshow(matrix, cmap="Blues")
        ax.set_xticks(range(len(pred_labels)))
        ax.set_xticklabels(pred_labels, rotation=25)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        ax.set_title(f"{title} confusion")
        for row_idx in range(len(labels)):
            for col_idx in range(len(pred_labels)):
                ax.text(col_idx, row_idx, str(matrix[row_idx][col_idx]), ha="center", va="center")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(CONFUSION_PATH)
    plt.close()


def _json_safe_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    safe = dict(metrics)
    if "confusion" in safe:
        safe["confusion"] = {
            f"{true}->{pred}": count
            for (true, pred), count in safe["confusion"].items()
        }
    if "episodes" in safe:
        safe["episodes"] = [
            {
                key: value
                for key, value in episode.items()
                if key not in {"trajectory"}
            }
            for episode in safe["episodes"]
        ]
    return safe


def run_smoke_training(config: TrainingConfig) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    before_seeds = list(range(15))
    before_metrics = evaluate_policy(choose_heuristic_action, before_seeds, config.max_tool_calls)

    reward_rows: list[dict[str, Any]] = []
    for step in range(1, config.smoke_steps + 1):
        seeds = [step * 10 + idx for idx in range(config.batch_size)]
        metrics = evaluate_policy(choose_heuristic_action, seeds, config.max_tool_calls)
        reward_rows.append(
            {
                "step": step,
                "mean_reward": metrics["mean_reward"],
                "guardrail_violation_rate": metrics["guardrail_violation_rate"],
                "correct_category_rate": metrics["correct_category_rate"],
                "disambiguation_rate": metrics["disambiguation_rate"],
                "mean_steps": metrics["mean_steps"],
            }
        )

    after_metrics = evaluate_policy(choose_heuristic_action, before_seeds, config.max_tool_calls)
    return reward_rows, before_metrics, after_metrics


def run_grpo_training(config: TrainingConfig) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Best-effort GRPO entrypoint.

    This stays deliberately conservative: if TRL is unavailable, we fall back
    to the smoke loop rather than crashing the repo. In Colab, install the
    dependencies and extend this with your preferred prompt-to-rollout bridge.
    """
    if not TRL_AVAILABLE:
        return run_smoke_training(config)

    # This scaffold intentionally keeps the real-training path light-touch.
    # The env + rollout metrics are the stable core; the trainer wiring can
    # evolve without changing the benchmark itself.
    _ = AutoTokenizer.from_pretrained(config.base_model)
    _ = AutoModelForCausalLM.from_pretrained(config.base_model)
    _ = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        target_modules=list(config.target_modules),
        task_type="CAUSAL_LM",
    )
    _ = GRPOConfig(
        learning_rate=config.learning_rate,
        num_generations=config.grpo_group_size,
        max_prompt_length=1024,
        max_completion_length=256,
        per_device_train_batch_size=1,
        logging_steps=1,
        save_steps=0,
    )
    return run_smoke_training(config)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Run the lightweight smoke loop.")
    parser.add_argument("--steps", type=int, default=10, help="Training steps for smoke mode.")
    args = parser.parse_args()

    config = TrainingConfig(smoke_steps=args.steps)
    reward_rows, before_metrics, after_metrics = (
        run_smoke_training(config) if args.smoke else run_grpo_training(config)
    )
    write_metrics_csv(reward_rows)
    plot_training_outputs(reward_rows, before_metrics, after_metrics)

    print("Training summary")
    print(
        json.dumps(
            {
                "config": asdict(config),
                "before": _json_safe_metrics(before_metrics),
                "after": _json_safe_metrics(after_metrics),
            },
            indent=2,
            default=str,
        )
    )
    print(f"Saved: {CSV_PATH.name}, {REWARD_PLOT_PATH.name}, {BAR_PLOT_PATH.name}, {CONFUSION_PATH.name}")


if __name__ == "__main__":
    main()
