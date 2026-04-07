"""
app/grader.py — Official grader for the Cloud Autoscaler OpenEnv.

Reads bounds and weights from the TaskConfig (sourced from tasks/*.yaml),
and consumes the episode_history produced by AutoscalerEnv.

Called by main.py at GET /grader after an episode is complete.
"""

from typing import List, Dict, Any
from app.models import TaskConfig


def grade_episode(episode_history: List[Dict[str, Any]], config: TaskConfig) -> Dict:
    """
    Score a completed episode using the normalised penalty formula:

        score = 1.0 - (w_L * L̂ + w_C * Ĉ + w_I * Î + w_V * V̂)

    Each component is the *episode-level* average, normalised to [0, 1]
    using the task-specific bounds from the YAML config.

    Args:
        episode_history: list of step dicts from AutoscalerEnv.episode_history.
                         Each entry has keys: observation, action, reward, info.
        config:          TaskConfig loaded from tasks/<task_id>.yaml.

    Returns:
        dict with score [0.0, 1.0] and a full breakdown of raw/normalised metrics.
    """
    bounds  = config.grader_bounds
    weights = config.grader_weights
    T = len(episode_history)

    if T == 0:
        return {"score": 0.0, "error": "Empty episode history"}

    # ------------------------------------------------------------------
    # Accumulate raw metrics across all steps
    # ------------------------------------------------------------------
    total_latency    = 0.0
    total_servers    = 0.0
    total_instability = 0.0
    sla_violations   = 0

    for i, step in enumerate(episode_history):
        obs  = step["observation"]
        info = step["info"]

        # Latency proxy: recorded per-step in info (L_t = q_t + overflow)
        total_latency += info["latency"]

        # Cost proxy: active servers at this step
        total_servers += obs.active_servers

        # SLA violation: latency exceeded the threshold this step
        if info["latency"] > bounds.L_SLA:
            sla_violations += 1

        # Instability: server count change (already computed in simulator)
        total_instability += info["instability"]

    # ------------------------------------------------------------------
    # Compute episode-level averages
    # ------------------------------------------------------------------
    avg_latency    = total_latency / T
    avg_cost       = total_servers / T
    avg_instability = total_instability / max(T - 1, 1)   # avoid div-by-zero on 1-step episodes
    sla_rate       = sla_violations / T                    # fraction of steps with violations

    # ------------------------------------------------------------------
    # Normalise each component to [0, 1]
    # ------------------------------------------------------------------
    L_hat = min(1.0, avg_latency     / bounds.L_max)
    C_hat = min(1.0, avg_cost        / bounds.S_max)
    I_hat = min(1.0, avg_instability / max(bounds.I_max, 1))
    V_hat = min(1.0, sla_rate)                             # already in [0, 1]

    # ------------------------------------------------------------------
    # Compute final score
    # ------------------------------------------------------------------
    penalty = (
        weights.w_L * L_hat +
        weights.w_C * C_hat +
        weights.w_I * I_hat +
        weights.w_V * V_hat
    )
    final_score = round(max(0.0, min(1.0, 1.0 - penalty)), 4)

    return {
        "task":            config.task_id,
        "score":           final_score,
        # Raw episode averages
        "avg_latency":     round(avg_latency, 4),
        "avg_cost":        round(avg_cost, 4),
        "avg_instability": round(avg_instability, 4),
        "sla_violations":  sla_violations,
        "sla_rate":        round(sla_rate, 4),
        # Normalised components (useful for debugging)
        "L_hat":           round(L_hat, 4),
        "C_hat":           round(C_hat, 4),
        "I_hat":           round(I_hat, 4),
        "V_hat":           round(V_hat, 4),
        "steps":           T,
    }
