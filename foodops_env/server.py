"""Minimal FastAPI wrapper for FoodOpsEnv."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .env import FoodOpsEnv


app = FastAPI(title="FoodOps Incident Env")
_SESSIONS: dict[str, FoodOpsEnv] = {}


class ResetRequest(BaseModel):
    seed: int | None = None
    session_id: str | None = None


class StepRequest(BaseModel):
    action: dict[str, Any] = Field(default_factory=dict)
    session_id: str | None = None


def _resolve_session(session_id: str | None) -> tuple[str, FoodOpsEnv]:
    resolved = session_id or "default"
    env = _SESSIONS.get(resolved)
    if env is None:
        env = FoodOpsEnv()
        _SESSIONS[resolved] = env
    return resolved, env


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/reset")
def reset(req: ResetRequest) -> dict[str, Any]:
    session_id, env = _resolve_session(req.session_id)
    obs, info = env.reset(seed=req.seed)
    return {"session_id": session_id, "observation": obs, "info": info}


@app.post("/step")
def step(req: StepRequest) -> dict[str, Any]:
    session_id, env = _resolve_session(req.session_id)
    try:
        obs, reward, terminated, truncated, info = env.step(req.action)
    except RuntimeError as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "session_id": session_id,
        "observation": obs,
        "reward": reward,
        "terminated": terminated,
        "truncated": truncated,
        "info": info,
    }


@app.get("/state")
def state(session_id: str | None = None) -> dict[str, Any]:
    session_id, env = _resolve_session(session_id)
    return {"session_id": session_id, "state": env.get_state()}


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":  # pragma: no cover
    main()

