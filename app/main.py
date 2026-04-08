import uuid
from pathlib import Path
from typing import Dict

import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.models import TaskConfig, AutoscalerAction
from app.env import AutoscalerEnv
from app.grader import grade_episode


# ---------------------------------------------------------------------------
# App Setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Cloud Autoscaler OpenEnv",
    description=(
        "An OpenEnv-compliant cloud autoscaling environment. "
        "The agent controls server count to balance latency, cost, and stability."
    ),
    version="1.0.0",
)

# Locate tasks/ relative to this file (cloud-rl/tasks/)
TASKS_DIR = Path(__file__).parent.parent / "tasks"

# In-memory store for task configs and active sessions
TASK_CONFIGS: Dict[str, TaskConfig] = {}
SESSIONS: Dict[str, AutoscalerEnv] = {}


# ---------------------------------------------------------------------------
# Startup: load all task YAML configs
# ---------------------------------------------------------------------------

@app.on_event("startup")
def load_tasks():
    """Parse every *.yaml in tasks/ into a typed TaskConfig at server start."""
    for yaml_file in sorted(TASKS_DIR.glob("*.yaml")):
        with open(yaml_file) as f:
            raw = yaml.safe_load(f)
        config = TaskConfig(**raw)
        TASK_CONFIGS[config.task_id] = config
    print(f"[startup] Loaded {len(TASK_CONFIGS)} tasks: {list(TASK_CONFIGS.keys())}")


# ---------------------------------------------------------------------------
# Request / Response schemas (HTTP-layer only, not part of MDP)
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_id: str

class StepRequest(BaseModel):
    session_id: str
    action: AutoscalerAction


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", tags=["System"])
def root():
    """
    Root endpoint required for Hugging Face Spaces health checks.
    """
    return {
        "status": "ok", 
        "message": "Cloud Autoscaler API is running",
        "documentation": "/docs"
    }
def health():
    """Ping endpoint — used by the validation script to confirm the server is live."""
    return {"status": "ok", "tasks_loaded": len(TASK_CONFIGS)}


@app.get("/tasks", tags=["Environment"])
def list_tasks():
    """
    Return all available tasks and the action schema.

    The inference script calls this first to discover what task IDs exist
    and what actions are valid.
    """
    return {
        "tasks": [
            {
                "task_id":     cfg.task_id,
                "description": cfg.description,
                "max_steps":   cfg.max_steps,
                "difficulty":  cfg.task_id,   # easy / medium / hard
            }
            for cfg in TASK_CONFIGS.values()
        ],
        "action_schema": {
            "scale_change": "int, one of [-1, 0, 1]"
        },
    }


@app.post("/reset", tags=["Environment"])
def reset(req: ResetRequest):
    """
    Start a new episode for the requested task.

    Creates a fresh AutoscalerEnv, generates the workload sequence,
    and returns a session_id + the initial observation.
    """
    if req.task_id not in TASK_CONFIGS:
        raise HTTPException(
            status_code=404,
            detail=f"Task '{req.task_id}' not found. "
                   f"Available: {list(TASK_CONFIGS.keys())}",
        )

    session_id = str(uuid.uuid4())
    env = AutoscalerEnv()
    obs = env.reset(TASK_CONFIGS[req.task_id])
    SESSIONS[session_id] = env

    return {
        "session_id":  session_id,
        "task_id":     req.task_id,
        "observation": obs.model_dump(),
    }


@app.post("/step", tags=["Environment"])
def step(req: StepRequest):
    """
    Apply one action to the environment and return the next state.

    Returns: observation, reward, done, info.
    When done=True, call /grader to retrieve the episode score.
    """
    env = _get_session(req.session_id)
    result = env.step(req.action)
    return result.model_dump()


@app.get("/state", tags=["Environment"])
def state(session_id: str):
    """
    Return the current observation without advancing time.
    Safe to call anytime after /reset.
    """
    env = _get_session(session_id)
    return {"observation": env.state().model_dump()}


@app.get("/grader", tags=["Evaluation"])
def grader(session_id: str):
    """
    Score the completed episode and return a breakdown.

    Returns:
      score         float [0.0, 1.0]
      avg_latency   average per-step latency proxy
      avg_cost      average servers used
      instability   average server change per step
      sla_violations fraction of steps exceeding the SLA threshold
    """
    env = _get_session(session_id)

    if not env.episode_history:
        raise HTTPException(
            status_code=400,
            detail="No episode history found. Run at least one /step call first.",
        )

    return grade_episode(env.episode_history, env.config)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _get_session(session_id: str) -> AutoscalerEnv:
    """Look up an active session or raise 404."""
    env = SESSIONS.get(session_id)
    if env is None:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{session_id}' not found. Call /reset first.",
        )
    return env
