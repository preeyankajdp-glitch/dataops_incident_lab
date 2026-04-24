"""REINFORCE trainer for FoodOps Incident Lab.

# Colab setup:
#   !pip install -q torch transformers peft accelerate matplotlib
#   !pip install -e ./foodops_env
#   %env HF_TOKEN=<your-token-if-needed>
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
import argparse
import csv
import json
import math
import os
from pathlib import Path
import random
import re
from typing import Any, Callable

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer

from foodops_env.env import FoodOpsEnv
from foodops_env.reward import ESCALATION_TOOLS, FORBIDDEN_TOOLS, WRITE_TOOLS
from foodops_env.primitives import PRIMITIVES


HF_TOKEN = os.environ.get("HF_TOKEN", "")
ACTION_PREFIX = '{"tool": "'
SYSTEM_PROMPT = """You are a FoodOps incident analyst.

You must output exactly one valid JSON tool call.
No prose. No markdown. No explanation.

Rules:
- Use one tool only per step.
- Investigate with read tools before acting.
- Some incidents have multiple root causes — fix each one.
- If the world state may have changed, re-check before deciding.
- Never change commission rates, promo flags, or menu prices directly.

Valid format:
{"tool": "check_config", "args": {"brand": "Behrouz Biryani", "channel": "Swiggy"}}

More valid examples:
{"tool": "check_pipeline_freshness", "args": {"table_name": "menu_sync", "brand": "Behrouz Biryani", "channel": "Swiggy"}}
{"tool": "force_menu_sync", "args": {"brand": "Behrouz Biryani", "channel": "Swiggy"}}
{"tool": "escalate_to_finance", "args": {"reason": "Commission drift above baseline."}}
{"tool": "inspect_recent_orders", "args": {"brand": "Faasos", "channel": "Zomato", "limit": 10}}
{"tool": "restart_pipeline", "args": {"pipeline_name": "orders_ingest"}}
"""


@dataclass
class TrainingConfig:
    base_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    lora_rank: int = 16
    lora_alpha: int = 32
    target_modules: tuple = ("q_proj", "k_proj", "v_proj", "o_proj")
    training_steps: int = 150
    max_tool_calls_per_episode: int = 12
    learning_rate: float = 1e-5
    max_new_tokens: int = 64
    temperature: float = 0.2
    baseline_ema: float = 0.9
    eval_seeds: list[int] = field(default_factory=lambda: list(range(1000, 1006)))
    checkpoint_every: int = 50
    smoke_steps: int = 3
    hf_token: str = HF_TOKEN
    batch_size: int = 4


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_model_dtype(device: torch.device) -> torch.dtype:
    if device.type == "cuda":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    if device.type == "mps":
        return torch.float16
    return torch.float32


def _resolve_target_modules(model: torch.nn.Module, requested: tuple[str, ...]) -> list[str]:
    available = {name.split(".")[-1] for name, _ in model.named_modules()}
    matched = [name for name in requested if name in available]
    if matched:
        return matched
    fallbacks = [name for name in ("c_attn", "c_proj") if name in available]
    if fallbacks:
        return fallbacks
    return [name for name in sorted(available) if name.endswith("proj")][:4]


def count_trainable_parameters(model: torch.nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def load_model(model_name: str, config: TrainingConfig | None = None) -> tuple[torch.nn.Module, Any]:
    cfg = config or TrainingConfig()
    device = get_device()
    dtype = get_model_dtype(device)
    token = cfg.hf_token or None

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=token,
        trust_remote_code=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    target_modules = _resolve_target_modules(model, tuple(cfg.target_modules))
    lora_config = LoraConfig(
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.to(device)
    model.train()
    model._foodops_device = device  # type: ignore[attr-defined]
    return model, tokenizer


def build_prompt(tokenizer: Any, obs: dict[str, Any]) -> str:
    user_prompt = format_observation(obs)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + ACTION_PREFIX
    return f"{SYSTEM_PROMPT}\n\nUser:\n{user_prompt}\n\nAssistant:\n{ACTION_PREFIX}"


def _sanitize_last_tool_call(last_tool_call: Any) -> Any:
    if not isinstance(last_tool_call, dict):
        return last_tool_call
    if "raw" in last_tool_call and "tool" not in last_tool_call:
        return {"status": "invalid_previous_action"}
    return {key: value for key, value in last_tool_call.items() if key != "raw"}


def _sanitize_last_tool_result(last_tool_result: Any) -> Any:
    if not isinstance(last_tool_result, dict):
        return last_tool_result
    sanitized = dict(last_tool_result)
    if "error" in sanitized:
        sanitized.pop("raw", None)
    return sanitized


def format_observation(obs: dict[str, Any]) -> str:
    dashboard = obs["dashboard_kpis"]
    compact_dashboard = {
        "brand": dashboard.get("brand"),
        "channel": dashboard.get("channel"),
        "net_payout": dashboard.get("net_payout"),
        "gross_sales": dashboard.get("gross_sales"),
        "discount_amount": dashboard.get("discount_amount"),
        "effective_commission_pct": dashboard.get("effective_commission_pct"),
        "aov": dashboard.get("aov"),
        "total_orders": dashboard.get("total_orders"),
    }
    return (
        f"Complaint: {obs['user_complaint']}\n"
        f"Dashboard KPIs: {json.dumps(compact_dashboard, sort_keys=True)}\n"
        f"Last tool call: {json.dumps(_sanitize_last_tool_call(obs['last_tool_call']), sort_keys=True)}\n"
        f"Last tool result: {json.dumps(_sanitize_last_tool_result(obs['last_tool_result']), sort_keys=True)}\n"
        f"Steps remaining: {obs['steps_remaining']}\n"
        f"Tools available: {', '.join(obs['tools_available'])}"
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
    return {"_malformed": True, "raw": text[:200]}


def predicted_category_from_tool(tool_name: str | None) -> str:
    if tool_name in WRITE_TOOLS:
        return "remediate"
    if tool_name in ESCALATION_TOOLS:
        return "escalate"
    if tool_name in FORBIDDEN_TOOLS:
        return "forbidden"
    return "none"


def is_malformed_action(action: Any) -> bool:
    return isinstance(action, dict) and bool(action.get("_malformed"))


def to_env_action(action: Any, fallback_raw: str = "") -> dict[str, Any]:
    if isinstance(action, dict) and "tool" in action and not action.get("_malformed"):
        return action
    if is_malformed_action(action):
        return {"raw": action.get("raw", fallback_raw)}
    return {"raw": fallback_raw or json.dumps(action, default=str)}


def random_policy(obs: dict[str, Any], rng: random.Random | None = None) -> dict[str, Any]:
    local_rng = rng or random
    brand = obs["dashboard_kpis"].get("brand")
    channel = obs["dashboard_kpis"].get("channel")
    tool = local_rng.choice(obs["tools_available"])
    if tool == "get_kpi_summary":
        return {"tool": tool, "args": {"brand": brand, "channel": channel, "range": local_rng.choice(["current", "baseline"])}}
    if tool == "check_config":
        return {"tool": tool, "args": {"brand": brand, "channel": channel}}
    if tool == "check_pipeline_freshness":
        table_name = local_rng.choice(["orders", "menu_sync", "channel_daily_metrics"])
        args = {"table_name": table_name}
        if table_name == "menu_sync":
            args.update({"brand": brand, "channel": channel})
        return {"tool": tool, "args": args}
    if tool == "inspect_recent_orders":
        return {"tool": tool, "args": {"brand": brand, "channel": channel, "limit": local_rng.choice([5, 10])}}
    if tool == "force_menu_sync":
        return {"tool": tool, "args": {"brand": brand, "channel": channel}}
    if tool == "restart_pipeline":
        return {"tool": tool, "args": {"pipeline_name": local_rng.choice(["orders_ingest", "menu_sync", "daily_metrics"]) }}
    if tool in {"escalate_to_finance", "escalate_to_ops"}:
        return {"tool": tool, "args": {"reason": "random guess"}}
    if tool == "update_commission_rate":
        return {"tool": tool, "args": {"channel": channel, "rate": local_rng.choice([18, 20, 22, 24])}}
    if tool == "toggle_promo":
        return {"tool": tool, "args": {"brand": brand, "channel": channel, "active": local_rng.choice([True, False])}}
    if tool == "update_menu_prices":
        return {"tool": tool, "args": {"brand": brand, "channel": channel, "prices": {"sample_sku": 199}}}
    return {"_malformed": True, "raw": "random policy malformed"}


def choose_heuristic_action(obs: dict[str, Any]) -> dict[str, Any]:
    complaint = obs["user_complaint"].lower()
    brand = obs["dashboard_kpis"]["brand"]
    channel = obs["dashboard_kpis"]["channel"]
    last_tool = (obs["last_tool_call"] or {}).get("tool")
    result = obs["last_tool_result"] or {}
    tools_used = set()
    # Build history of tools used from the trajectory (via observation chain)
    if last_tool:
        tools_used.add(last_tool)

    if last_tool is None:
        return {"tool": "check_config", "args": {"brand": brand, "channel": channel}}

    if last_tool == "check_config":
        if result.get("is_live") is False:
            return {"tool": "escalate_to_ops", "args": {"reason": "Brand is paused on this channel."}}
        if result.get("promo_active"):
            return {"tool": "escalate_to_finance", "args": {"reason": "Promo appears stuck on the target pair."}}
        if float(result.get("effective_commission_pct", 0.0)) > float(result.get("baseline_commission_pct", 0.0)):
            return {"tool": "escalate_to_finance", "args": {"reason": "Commission rate drifted above baseline."}}
        return {"tool": "check_pipeline_freshness", "args": {"table_name": "menu_sync", "brand": brand, "channel": channel}}

    if last_tool == "check_pipeline_freshness":
        if result.get("is_stale"):
            table = result.get("table_name", "")
            if table == "menu_sync":
                return {"tool": "force_menu_sync", "args": {"brand": brand, "channel": channel}}
            return {"tool": "restart_pipeline", "args": {"pipeline_name": table or "orders_ingest"}}
        return {"tool": "inspect_recent_orders", "args": {"brand": brand, "channel": channel, "limit": 10}}

    if last_tool == "inspect_recent_orders":
        rows = result.get("rows", [])
        if not rows:
            return {"tool": "restart_pipeline", "args": {"pipeline_name": "orders_ingest"}}
        if any(str(r.get("order_id", "")).endswith("-DUP") or r.get("is_duplicate") for r in rows):
            return {"tool": "restart_pipeline", "args": {"pipeline_name": "dedup"}}
        if any(r.get("status") == "refunded" for r in rows):
            refund_rate = sum(1 for r in rows if r.get("status") == "refunded") / len(rows)
            if refund_rate > 0.10:
                return {"tool": "escalate_to_ops", "args": {"reason": f"High refund rate ({refund_rate:.0%})."}}
        return {"tool": "escalate_to_ops", "args": {"reason": "No obvious issue found in recent orders."}}

    if last_tool == "get_kpi_summary":
        return {"tool": "check_config", "args": {"brand": brand, "channel": channel}}

    if last_tool in ("force_menu_sync", "restart_pipeline"):
        # After a remediation, check if more issues remain
        pending = obs.get("pending_resolutions", 0)
        if pending and pending > 0:
            return {"tool": "check_config", "args": {"brand": brand, "channel": channel}}
        return {"tool": "escalate_to_ops", "args": {"reason": "Pipeline remediated, verifying state."}}

    return {"tool": "escalate_to_ops", "args": {"reason": "Unable to disambiguate confidently."}}


def rollout_policy(
    policy_fn: Callable[[dict[str, Any]], Any],
    seed: int,
    max_tool_calls: int,
) -> dict[str, Any]:
    env = FoodOpsEnv()
    obs, info = env.reset(seed=seed)
    reward_total = 0.0
    terminal_tool = None
    malformed_actions = 0
    for _ in range(max_tool_calls):
        policy_output = policy_fn(obs)
        action = policy_output[0] if isinstance(policy_output, tuple) else policy_output
        env_action = to_env_action(action)
        obs, reward, terminated, truncated, _ = env.step(env_action)
        reward_total += reward
        if env_action.get("tool") is None:
            malformed_actions += 1
        if terminated or truncated:
            terminal_tool = env_action.get("tool")
            break
    state = env.get_state()
    return {
        "seed": seed,
        "scenario_id": info["scenario_id"],
        "anomaly_category": info["anomaly_category"],
        "difficulty": info.get("difficulty", "easy"),
        "is_compound": info.get("is_compound", False),
        "has_drift": info.get("has_drift", False),
        "reward_total": reward_total,
        "steps": len(state["trajectory"]),
        "trajectory": state["trajectory"],
        "terminal_tool": terminal_tool,
        "disambiguated": state["phase_flags"]["disambiguated"],
        "guardrail_violation": terminal_tool in FORBIDDEN_TOOLS,
        "predicted_category": predicted_category_from_tool(terminal_tool),
        "malformed_actions": malformed_actions,
    }


def evaluate_policy(
    policy_fn: Callable[[dict[str, Any]], Any],
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
    malformed_action_rate = sum(ep["malformed_actions"] for ep in episodes) / len(episodes)
    confusion = Counter((ep["anomaly_category"], ep["predicted_category"]) for ep in episodes)
    per_scenario: dict[str, dict[str, float]] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for episode in episodes:
        grouped[episode["scenario_id"]].append(episode)
    for scenario_id, rows in grouped.items():
        per_scenario[scenario_id] = {
            "mean_reward": sum(row["reward_total"] for row in rows) / len(rows),
            "correct_category_rate": sum(
                1
                for row in rows
                if (
                    row["anomaly_category"] == "pipeline" and row["predicted_category"] == "remediate"
                ) or (
                    row["anomaly_category"] == "business" and row["predicted_category"] == "escalate"
                )
            ) / len(rows),
            "disambiguation_rate": sum(1 for row in rows if row["disambiguated"]) / len(rows),
        }

    per_difficulty: dict[str, dict[str, float]] = {}
    diff_grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for episode in episodes:
        diff_grouped[episode["difficulty"]].append(episode)
    for difficulty, rows in diff_grouped.items():
        per_difficulty[difficulty] = {
            "count": len(rows),
            "mean_reward": sum(row["reward_total"] for row in rows) / len(rows),
            "disambiguation_rate": sum(1 for row in rows if row["disambiguated"]) / len(rows),
        }

    return {
        "episodes": episodes,
        "mean_reward": mean_reward,
        "guardrail_violation_rate": guardrail_rate,
        "correct_category_rate": correct_category_rate,
        "disambiguation_rate": disambiguation_rate,
        "mean_steps": mean_steps,
        "mean_malformed_actions": malformed_action_rate,
        "confusion": confusion,
        "per_scenario": per_scenario,
        "per_difficulty": per_difficulty,
    }


def benchmark_action_format(
    model: torch.nn.Module,
    tokenizer: Any,
    config: TrainingConfig,
    seeds: list[int],
    *,
    sample: bool,
    max_examples: int = 10,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    malformed_count = 0
    valid_json_count = 0
    valid_tool_count = 0

    for seed in seeds:
        env = FoodOpsEnv()
        obs, info = env.reset(seed=seed)
        action, _, decoded, prompt = qwen_policy(
            obs,
            model,
            tokenizer,
            config,
            sample=sample,
            debug_shapes=False,
        )
        is_malformed = is_malformed_action(action)
        valid_json = not is_malformed
        tool_name = action.get("tool") if isinstance(action, dict) else None
        valid_tool = valid_json and tool_name in obs["tools_available"]
        malformed_count += int(is_malformed)
        valid_json_count += int(valid_json)
        valid_tool_count += int(valid_tool)

        if len(rows) < max_examples:
            rows.append(
                {
                    "seed": seed,
                    "scenario_id": info["scenario_id"],
                    "brand": obs["dashboard_kpis"].get("brand"),
                    "channel": obs["dashboard_kpis"].get("channel"),
                    "valid_json": valid_json,
                    "valid_tool": valid_tool,
                    "tool": tool_name,
                    "decoded": decoded,
                    "raw": action.get("raw", decoded) if isinstance(action, dict) else decoded,
                    "prompt": prompt,
                }
            )

    total = max(1, len(seeds))
    return {
        "sample": sample,
        "seed_count": len(seeds),
        "valid_json_rate": valid_json_count / total,
        "valid_tool_rate": valid_tool_count / total,
        "malformed_rate": malformed_count / total,
        "examples": rows,
    }


def make_model_policy(model: torch.nn.Module, tokenizer: Any, config: TrainingConfig) -> Callable[[dict[str, Any]], dict[str, Any]]:
    def _policy(obs: dict[str, Any]) -> dict[str, Any]:
        prompt = build_prompt(tokenizer, obs)
        device = next(model.parameters()).device
        encoded = tokenizer(prompt, return_tensors="pt")
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        with torch.no_grad():
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=config.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=False,
            )
        generated_ids = generated[:, input_ids.shape[1] :]
        decoded = ACTION_PREFIX + tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        return parse_action(decoded)

    return _policy


def generate_with_logprobs(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    config: TrainingConfig,
    *,
    sample: bool,
    debug_shapes: bool,
) -> tuple[torch.Tensor, torch.Tensor, str]:
    device = next(model.parameters()).device
    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    generation_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": config.max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "return_dict_in_generate": True,
        "output_scores": True,
    }
    if sample and config.temperature > 0:
        generation_kwargs.update({"do_sample": True, "temperature": config.temperature, "top_p": 0.95})
    else:
        generation_kwargs.update({"do_sample": False})

    with torch.no_grad():
        generated = model.generate(**generation_kwargs)

    generated_ids = generated.sequences[:, input_ids.shape[1] :]
    if generated_ids.shape[1] == 0:
        empty = torch.zeros(0, device=device, dtype=torch.float32)
        return generated_ids.squeeze(0), empty, ""

    debug_log_probs = []
    for step_scores, token_id in zip(generated.scores, generated_ids[0]):
        debug_log_probs.append(torch.log_softmax(step_scores[0].float(), dim=-1)[token_id].detach())
    debug_log_probs_tensor = torch.stack(debug_log_probs).to(torch.float32)

    if debug_shapes:
        print(
            f"generate_with_logprobs: tokens={tuple(generated_ids.squeeze(0).shape)} "
            f"log_probs={tuple(debug_log_probs_tensor.shape)} dtype={debug_log_probs_tensor.dtype}"
        )
        print(f"generate debug scores shape: {tuple(debug_log_probs_tensor.shape)} dtype={debug_log_probs_tensor.dtype}")

    decoded = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return generated_ids.squeeze(0), debug_log_probs_tensor, decoded


def compute_response_logprobs(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    response_token_ids: list[int],
) -> torch.Tensor:
    device = next(model.parameters()).device
    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    response_ids = torch.tensor(response_token_ids, device=device, dtype=torch.long).unsqueeze(0)
    if response_ids.shape[1] == 0:
        return torch.zeros(0, device=device, dtype=torch.float32)

    full_ids = torch.cat([input_ids, response_ids], dim=1)
    full_attention = torch.ones_like(full_ids)
    outputs = model(input_ids=full_ids[:, :-1], attention_mask=full_attention[:, :-1])
    prompt_len = input_ids.shape[1]
    gen_len = response_ids.shape[1]
    token_logits = outputs.logits[:, prompt_len - 1 : prompt_len - 1 + gen_len, :]
    return (
        F.log_softmax(token_logits.float(), dim=-1)
        .gather(-1, response_ids.unsqueeze(-1))
        .squeeze(-1)
        .squeeze(0)
        .to(torch.float32)
    )


def qwen_policy(
    obs: dict[str, Any],
    model: torch.nn.Module,
    tokenizer: Any,
    config: TrainingConfig,
    *,
    sample: bool,
    debug_shapes: bool,
) -> tuple[dict[str, Any], torch.Tensor, str, str]:
    prompt = build_prompt(tokenizer, obs)
    response_ids, log_probs, decoded = generate_with_logprobs(
        model,
        tokenizer,
        prompt,
        config,
        sample=sample,
        debug_shapes=debug_shapes,
    )
    full_decoded = ACTION_PREFIX + decoded
    action = parse_action(full_decoded)
    return action, response_ids, full_decoded, prompt


def save_checkpoint(model: torch.nn.Module, tokenizer: Any, output_dir: Path, step: int) -> None:
    checkpoint_dir = output_dir / "checkpoints" / f"step_{step}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)


def train_reinforce(
    model: torch.nn.Module,
    tokenizer: Any,
    config: TrainingConfig,
    output_dir: Path,
    *,
    smoke: bool,
    seed: int,
) -> list[dict[str, Any]]:
    device = next(model.parameters()).device
    optimizer = AdamW((param for param in model.parameters() if param.requires_grad), lr=config.learning_rate)
    optimizer.zero_grad(set_to_none=True)
    running_baseline = 0.0
    metrics_log: list[dict[str, Any]] = []
    malformed_samples: list[dict[str, Any]] = []
    total_steps = config.smoke_steps if smoke else config.training_steps
    first_debug = True

    for step in range(total_steps):
        episode_seed = seed + step
        env = FoodOpsEnv()
        obs, info = env.reset(seed=episode_seed)
        trajectory: list[dict[str, Any]] = []
        terminated = False
        truncated = False
        while not (terminated or truncated) and len(trajectory) < config.max_tool_calls_per_episode:
            action, response_ids, decoded, prompt = qwen_policy(
                obs,
                model,
                tokenizer,
                config,
                sample=True,
                debug_shapes=smoke and first_debug,
            )
            first_debug = False
            env_action = to_env_action(action, decoded)
            obs, reward, terminated, truncated, step_info = env.step(env_action)
            if is_malformed_action(action) and len(malformed_samples) < 20:
                malformed_sample = {
                    "training_step": step,
                    "episode_seed": episode_seed,
                    "prompt": prompt,
                    "decoded": decoded,
                    "raw": action.get("raw", decoded),
                }
                malformed_samples.append(malformed_sample)
                if smoke:
                    print("Malformed action sample:", json.dumps(malformed_sample, indent=2)[:1200])
            trajectory.append(
                {
                    "prompt": prompt,
                    "action": action,
                    "decoded": decoded,
                    "response_token_ids": response_ids.detach().cpu().tolist(),
                    "reward": reward,
                }
            )

        total_reward = sum(item["reward"] for item in trajectory)
        advantage = total_reward - running_baseline
        valid_items = [item for item in trajectory if item["response_token_ids"]]
        if valid_items:
            loss_total = 0.0
            scale = -float(advantage) / (len(valid_items) * float(config.batch_size))
            for item in valid_items:
                token_log_probs = compute_response_logprobs(
                    model,
                    tokenizer,
                    item["prompt"],
                    item["response_token_ids"],
                )
                step_loss = token_log_probs.sum() * scale
                step_loss.backward()
                loss_total += float(step_loss.detach().cpu().item())
                del token_log_probs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            loss_value = loss_total
        else:
            loss_value = 0.0

        if (step + 1) % config.batch_size == 0 or step == total_steps - 1:
            torch.nn.utils.clip_grad_norm_(
                [param for param in model.parameters() if param.requires_grad],
                max_norm=1.0,
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        running_baseline = (config.baseline_ema * running_baseline) + ((1.0 - config.baseline_ema) * total_reward)
        metrics_log.append(
            {
                "step": step,
                "seed": episode_seed,
                "mean_reward": total_reward,
                "loss": loss_value,
                "running_baseline": running_baseline,
                "trajectory_length": len(trajectory),
                "scenario_id": info["scenario_id"],
                "terminated": terminated,
                "truncated": truncated,
            }
        )

        if (not smoke) and config.checkpoint_every > 0 and (step + 1) % config.checkpoint_every == 0:
            save_checkpoint(model, tokenizer, output_dir, step + 1)

    malformed_path = output_dir / "malformed_action_samples.json"
    malformed_path.write_text(json.dumps(malformed_samples, indent=2, default=str))

    return metrics_log


def write_metrics_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_training_outputs(
    reward_rows: list[dict[str, Any]],
    random_metrics: dict[str, Any],
    trained_metrics: dict[str, Any],
    heuristic_metrics: dict[str, Any],
    output_dir: Path,
) -> None:
    reward_plot = output_dir / "reward_curve.png"
    bars_plot = output_dir / "before_after_bars.png"
    confusion_plot = output_dir / "confusion_matrix.png"

    steps = [row["step"] for row in reward_rows]
    rewards = [row["mean_reward"] for row in reward_rows]
    moving_average = []
    for idx in range(len(rewards)):
        window = rewards[max(0, idx - 19) : idx + 1]
        moving_average.append(sum(window) / len(window))

    plt.figure(figsize=(8, 4))
    plt.plot(steps, rewards, label="trajectory reward", alpha=0.4)
    plt.plot(steps, moving_average, label="20-step moving average")
    plt.xlabel("Training step")
    plt.ylabel("Reward")
    plt.title("FoodOps reward curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(reward_plot)
    plt.close()

    metric_names = [
        "mean_reward",
        "guardrail_violation_rate",
        "correct_category_rate",
        "disambiguation_rate",
    ]
    series = [
        ("Random", random_metrics),
        ("Trained", trained_metrics),
        ("Heuristic", heuristic_metrics),
    ]
    x = list(range(len(metric_names)))
    width = 0.24
    plt.figure(figsize=(9, 4))
    offsets = [-width, 0.0, width]
    for (label, metrics), offset in zip(series, offsets):
        plt.bar(
            [item + offset for item in x],
            [metrics[name] for name in metric_names],
            width=width,
            label=label,
        )
    plt.xticks(x, metric_names, rotation=20)
    plt.title("Random vs trained vs heuristic")
    plt.legend()
    plt.tight_layout()
    plt.savefig(bars_plot)
    plt.close()

    labels = ["pipeline", "business", "financial", "compound", "none"]
    pred_labels = ["remediate", "escalate", "forbidden", "none"]
    before_matrix = [[random_metrics["confusion"].get((true, pred), 0) for pred in pred_labels] for true in labels]
    after_matrix = [[trained_metrics["confusion"].get((true, pred), 0) for pred in pred_labels] for true in labels]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, matrix, title in zip(axes, [before_matrix, after_matrix], ["Before (random)", "After (trained)"]):
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
    plt.savefig(confusion_plot)
    plt.close()


def write_eval_report(
    random_metrics: dict[str, Any],
    trained_metrics: dict[str, Any],
    heuristic_metrics: dict[str, Any],
    action_benchmark: dict[str, Any],
    reward_rows: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    def _json_safe(metrics: dict[str, Any]) -> dict[str, Any]:
        safe = dict(metrics)
        safe["confusion"] = {f"{true}->{pred}": count for (true, pred), count in metrics["confusion"].items()}
        safe["episodes"] = [
            {key: value for key, value in episode.items() if key != "trajectory"}
            for episode in metrics["episodes"]
        ]
        return safe

    report_path = output_dir / "eval_report.json"
    report_path.write_text(
        json.dumps(
            {
                "random": _json_safe(random_metrics),
                "trained": _json_safe(trained_metrics),
                "heuristic": _json_safe(heuristic_metrics),
                "pretrain_action_format": action_benchmark,
                "training": {
                    "steps": len(reward_rows),
                    "best_reward": max((row["mean_reward"] for row in reward_rows), default=0.0),
                    "last_reward": reward_rows[-1]["mean_reward"] if reward_rows else 0.0,
                },
            },
            indent=2,
            default=str,
        )
    )

    summary_path = output_dir / "run_summary.md"
    summary_path.write_text(
        "\n".join(
            [
                "# FoodOps REINFORCE Run",
                "",
                f"- Random mean reward: {random_metrics['mean_reward']:+.3f}",
                f"- Trained mean reward: {trained_metrics['mean_reward']:+.3f}",
                f"- Heuristic mean reward: {heuristic_metrics['mean_reward']:+.3f}",
                f"- Pre-train valid JSON rate: {action_benchmark['valid_json_rate']:.2%}",
                f"- Pre-train valid tool rate: {action_benchmark['valid_tool_rate']:.2%}",
                f"- Pre-train malformed rate: {action_benchmark['malformed_rate']:.2%}",
                f"- Trained mean malformed actions: {trained_metrics['mean_malformed_actions']:.2f}",
                f"- Trained disambiguation rate: {trained_metrics['disambiguation_rate']:.2%}",
                f"- Trained guardrail violation rate: {trained_metrics['guardrail_violation_rate']:.2%}",
                "",
                "Per-scenario trained metrics:",
                json.dumps(trained_metrics["per_scenario"], indent=2),
                "",
                "Per-difficulty trained metrics:",
                json.dumps(trained_metrics.get("per_difficulty", {}), indent=2),
            ]
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Run a short CPU-friendly pipeline check.")
    parser.add_argument("--steps", type=int, default=None, help="Override the number of training trajectories.")
    parser.add_argument("--seed", type=int, default=7, help="Base seed for training rollouts.")
    parser.add_argument("--output-dir", default="./training_output", help="Where to write plots, CSVs, and reports.")
    parser.add_argument("--model", default=None, help="Optional model override.")
    parser.add_argument("--smoke-eval-seeds", type=int, default=3, help="Number of eval seeds to use during smoke mode.")
    args = parser.parse_args()

    config = TrainingConfig()
    if args.steps is not None:
        if args.smoke:
            config.smoke_steps = args.steps
        else:
            config.training_steps = args.steps

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_name = args.model or config.base_model
    if args.smoke:
        config.max_new_tokens = 32

    model, tokenizer = load_model(model_name, config)
    print(f"Loaded: {getattr(model.config, 'model_type', 'unknown')}")
    print(f"LoRA params: {count_trainable_parameters(model)}")

    random_rng = random.Random(args.seed)
    random_policy_fn = lambda obs: random_policy(obs, random_rng)
    heuristic_policy_fn = choose_heuristic_action
    trained_policy_fn = make_model_policy(model, tokenizer, config)

    eval_seeds = config.eval_seeds if not args.smoke else config.eval_seeds[: args.smoke_eval_seeds]
    action_benchmark = benchmark_action_format(
        model,
        tokenizer,
        config,
        eval_seeds,
        sample=False,
    )
    benchmark_path = output_dir / "action_format_benchmark.json"
    benchmark_path.write_text(json.dumps(action_benchmark, indent=2, default=str))
    print(
        "Pre-train action benchmark:",
        json.dumps(
            {
                "valid_json_rate": round(action_benchmark["valid_json_rate"], 4),
                "valid_tool_rate": round(action_benchmark["valid_tool_rate"], 4),
                "malformed_rate": round(action_benchmark["malformed_rate"], 4),
            },
            indent=2,
        ),
    )
    random_before = evaluate_policy(random_policy_fn, eval_seeds, config.max_tool_calls_per_episode)
    heuristic_ref = evaluate_policy(heuristic_policy_fn, eval_seeds, config.max_tool_calls_per_episode)

    reward_rows = train_reinforce(
        model,
        tokenizer,
        config,
        output_dir,
        smoke=args.smoke,
        seed=args.seed,
    )

    trained_after = evaluate_policy(trained_policy_fn, eval_seeds, config.max_tool_calls_per_episode)

    csv_path = output_dir / "training_metrics.csv"
    if not args.smoke:
        write_metrics_csv(reward_rows, csv_path)
        plot_training_outputs(reward_rows, random_before, trained_after, heuristic_ref, output_dir)
    write_eval_report(random_before, trained_after, heuristic_ref, action_benchmark, reward_rows, output_dir)

    if args.smoke:
        print("Smoke per-step rewards:", [round(row["mean_reward"], 4) for row in reward_rows])

    summary = {
        "config": asdict(config),
        "model_name": model_name,
        "device": str(get_device()),
        "random_mean_reward": random_before["mean_reward"],
        "trained_mean_reward": trained_after["mean_reward"],
        "heuristic_mean_reward": heuristic_ref["mean_reward"],
        "pretrain_valid_json_rate": action_benchmark["valid_json_rate"],
        "pretrain_valid_tool_rate": action_benchmark["valid_tool_rate"],
        "action_benchmark_path": str(benchmark_path),
        "csv_path": str(csv_path if not args.smoke else ""),
        "plots_saved": not args.smoke,
    }
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
