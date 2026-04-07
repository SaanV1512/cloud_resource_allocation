from typing import List, Dict, Any, Optional

from app.models import (
    TaskConfig,
    AutoscalerAction,
    AutoscalerObservation,
    StepResult,
)
from app.simulator import CloudSimulator


class AutoscalerEnv:
    """
    OpenEnv-compliant glue layer for the Cloud Autoscaler MDP.

    Responsibilities:
      - reset() / step() / state() interface
      - Packaging raw simulator output into Pydantic models
      - Computing per-step reward
      - Tracking episode history for the grader

    Does NOT handle HTTP, sessions, or task loading.
    That is main.py's job.
    """

    def __init__(self):
        self.simulator: Optional[CloudSimulator] = None
        self.config: Optional[TaskConfig] = None
        self.episode_history: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # reset()
    # ------------------------------------------------------------------

    def reset(self, config: TaskConfig) -> AutoscalerObservation:
        """
        Initialise a fresh episode for the given task config.

        - Creates a new CloudSimulator (pre-generates full workload)
        - Clears episode history
        - Returns the first observation (before any action is taken)
        """
        self.config = config
        self.simulator = CloudSimulator(config)
        self.episode_history = []

        return self._build_observation()

    # ------------------------------------------------------------------
    # step()
    # ------------------------------------------------------------------

    def step(self, action: AutoscalerAction) -> StepResult:
        """
        Apply one MDP transition and return a fully typed StepResult.

        Flow:
          1. Pass scale_change to the simulator
          2. Build AutoscalerObservation from raw output
          3. Compute reward using normalised task-specific penalties
          4. Log to episode_history for the grader
          5. Return StepResult
        """
        self._require_reset()

        raw    = self.simulator.step(action.scale_change)
        obs    = self._obs_from_raw(raw)
        reward = self._compute_reward(raw)

        info = {
            "step":           raw["step"],
            "capacity":       raw["capacity"],
            "latency":        raw["latency"],
            "overflow":       raw["overflow"],
            "instability":    raw["instability"],
            "reward":         reward,
            # Mismatch 2 fix: cost and sla_violations needed by agent.learn()
            "cost":           round(raw["active_servers"] / self.config.max_servers, 4),
            "sla_violations": 1 if raw["latency"] > self.config.grader_bounds.L_SLA else 0,
        }

        result = StepResult(
            observation=obs,
            reward=reward,
            done=raw["done"],
            info=info,
        )

        # Record every step so the grader has the full episode
        self.episode_history.append({
            "observation": obs,
            "action":      action.scale_change,
            "reward":      reward,
            "info":        info,
        })

        return result

    # ------------------------------------------------------------------
    # state()
    # ------------------------------------------------------------------

    def state(self) -> AutoscalerObservation:
        """
        Return the current observation without advancing time.
        Zero side effects — safe to call as many times as needed.
        """
        self._require_reset()
        return self._build_observation()

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _compute_reward(self, raw: Dict[str, Any]) -> float:
        """
        Reward function (consistent with grader formula):

          R_t = -(w_L * L̂ + w_C * Ĉ + w_I * Î + w_V * V̂)

        Each penalty is normalised to [0, 1] using task-specific bounds,
        so reward is always in [-1, 0]. Agent maximises toward 0.

          L̂ = min(1, latency  / L_max)   ← latency penalty
          Ĉ = min(1, servers  / S_max)   ← cost penalty
          Î = min(1, |Δs|     / I_max)   ← instability penalty
          V̂ = 1.0 if latency > L_SLA else 0.0  ← SLA violation
        """
        bounds  = self.config.grader_bounds
        weights = self.config.grader_weights

        L_hat = min(1.0, raw["latency"]      / bounds.L_max)
        C_hat = min(1.0, raw["active_servers"] / bounds.S_max)
        I_hat = min(1.0, raw["instability"]  / max(bounds.I_max, 1))
        V_hat = 1.0 if raw["latency"] > bounds.L_SLA else 0.0

        reward = -(
            weights.w_L * L_hat +
            weights.w_C * C_hat +
            weights.w_I * I_hat +
            weights.w_V * V_hat
        )

        return round(reward, 6)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_observation(self) -> AutoscalerObservation:
        """Build an observation from current simulator state."""
        raw = self.simulator.get_state()
        return self._obs_from_raw(raw)

    def _obs_from_raw(self, raw: Dict[str, Any]) -> AutoscalerObservation:
        """Convert a raw simulator dict into a typed AutoscalerObservation."""
        return AutoscalerObservation(
            current_requests=raw["current_requests"],
            previous_requests=raw["previous_requests"],
            active_servers=raw["active_servers"],
            cpu_utilization=raw["cpu_utilization"],
            queue_length=raw["queue_length"],
        )

    def _require_reset(self):
        """Guard against calling step() or state() before reset()."""
        if self.simulator is None or self.config is None:
            raise RuntimeError(
                "Environment not initialised. Call reset(config) first."
            )
