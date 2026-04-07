import numpy as np
from typing import List
from app.models import TaskConfig, WorkloadPattern


class CloudSimulator:
    """
    Pure math engine for the cloud autoscaling MDP.

    Handles:
      - Pre-generating the full noisy workload sequence (deterministic via seed)
      - Applying the 1-step scaling delay
      - Queue dynamics: q_{t+1} = max(0, q_t + d_{t+1} - capacity_t)
      - CPU utilization: u_{t+1} = min(1.5, d_{t+1} / capacity_t)
      - Computing latency proxy L_t and instability for the reward

    Does NOT know about HTTP, FastAPI, rewards, or OpenEnv conventions.
    That is env.py's job.
    """

    def __init__(self, config: TaskConfig):
        self.config = config

        # Internal MDP state
        self.current_step: int = 0
        self.active_servers: int = config.initial_servers
        self.queue_length: int = 0
        self.current_requests: int = config.base_load
        self.previous_requests: int = config.base_load

        # 1-step delay buffer: action taken now applies NEXT step
        self.pending_action: int = 0

        # Pre-generate the entire workload at construction time
        # Same seed → same sequence → fully deterministic across all runs
        self.workload_sequence: List[int] = self._generate_workload()

    # ------------------------------------------------------------------
    # Workload Generation
    # ------------------------------------------------------------------

    def _generate_workload(self) -> List[int]:
        """
        Generate the full noisy workload sequence once at reset time.

        Pattern options:
          steady     → flat base load with light Gaussian noise
          diurnal    → sine wave oscillation (low → peak → low)
          flash_crowd → calm baseline with sudden massive spikes
        """
        rng = np.random.default_rng(seed=self.config.seed)
        steps = self.config.max_steps
        base = self.config.base_load
        std = self.config.noise_std
        max_cap = self.config.max_servers * self.config.capacity_per_server

        pattern = self.config.workload_pattern

        if pattern == WorkloadPattern.steady:
            # Flat baseline — agent should maintain minimal servers
            base_sequence = np.full(steps, base, dtype=float)

        elif pattern == WorkloadPattern.diurnal:
            # Sine wave: rises from base to base + 4×base then drops back
            # Agent must anticipate peaks due to 1-step delay
            t = np.arange(steps)
            amplitude = 4 * base
            base_sequence = base + amplitude * np.sin(2 * np.pi * t / steps)
            base_sequence = np.maximum(base_sequence, 0.0)

        elif pattern == WorkloadPattern.flash_crowd:
            # Structure: calm → spike → recovery → spike → calm
            # Spike loads hit 90% of max capacity to test limits
            calm1    = int(steps * 0.25)
            spike1   = int(steps * 0.15)
            recovery = int(steps * 0.20)
            spike2   = int(steps * 0.15)
            calm2    = steps - calm1 - spike1 - recovery - spike2

            spike_load = max_cap * 0.9         # 90% of total max capacity
            calm2_load = base * 1.2            # slightly elevated after chaos

            base_sequence = np.array(
                [base]        * calm1    +
                [spike_load]  * spike1   +
                [base]        * recovery +
                [spike_load * 0.85] * spike2 +  # second spike is smaller
                [calm2_load]  * calm2,
                dtype=float
            )

        else:
            base_sequence = np.full(steps, base, dtype=float)

        # Add reproducible Gaussian noise and floor at 0
        noise = rng.normal(0.0, std, size=steps)
        noisy = np.clip(base_sequence + noise, 0.0, None)
        return noisy.astype(int).tolist()

    # ------------------------------------------------------------------
    # MDP Step
    # ------------------------------------------------------------------

    def step(self, scale_change: int) -> dict:
        """
        Apply one MDP transition step and return raw components.

        Transition equations:
          s_{t+1} = clip(s_t + a_{t-1}, 1, S_max)   ← 1-step delay
          q_{t+1} = max(0, q_t + d_{t+1} - capacity_t)
          u_{t+1} = min(1.5, d_{t+1} / capacity_t)
          L_t     = q_t + max(0, d_t - capacity_t)   ← latency proxy

        Returns a raw dict — env.py is responsible for building
        StepResult, computing the reward, and tracking episode history.
        """
        prev_servers = self.active_servers

        # Apply the action buffered from the PREVIOUS step (1-step delay)
        new_servers = prev_servers + self.pending_action
        new_servers = int(np.clip(new_servers, 1, self.config.max_servers))

        # Buffer the current action for the NEXT step
        self.pending_action = scale_change

        # Read next demand FIRST — makes data flow explicit and safe
        d_next = self.workload_sequence[self.current_step]

        # Capacity is based on the NEW server count (takes effect this step)
        capacity = new_servers * self.config.capacity_per_server

        # Queue dynamics: overflow piles into the next step's backlog
        overflow  = max(0, d_next - capacity)
        new_queue = max(0, self.queue_length + d_next - capacity)

        # CPU utilization: clamped to [0.0, 1.5] per model constraint
        cpu_util = round(min(1.5, d_next / max(capacity, 1)), 4)

        # Latency proxy L_t = q_t + max(0, d_t - capacity_t)
        # Uses OLD queue (self.queue_length = q_t) — correct MDP timing
        latency = self.queue_length + overflow

        # Instability: how much did server count actually change this step?
        instability = abs(new_servers - prev_servers)

        # Commit demand to state AFTER all calculations use d_next
        self.previous_requests = self.current_requests
        self.current_requests  = d_next

        # Commit state
        self.active_servers = new_servers
        self.queue_length   = new_queue
        self.current_step  += 1

        done = self.current_step >= self.config.max_steps

        return {
            "current_requests":  self.current_requests,
            "previous_requests": self.previous_requests,
            "active_servers":    self.active_servers,
            "cpu_utilization":   cpu_util,
            "queue_length":      self.queue_length,
            "latency":           latency,
            "overflow":          overflow,
            "instability":       instability,
            "capacity":          capacity,
            "done":              done,
            "step":              self.current_step,
        }

    # ------------------------------------------------------------------
    # Read-only State Snapshot
    # ------------------------------------------------------------------

    def get_state(self) -> dict:
        """
        Return the current MDP state without advancing time.
        Called by env.state() — must have zero side effects.
        """
        capacity = self.active_servers * self.config.capacity_per_server
        return {
            "current_requests":  self.current_requests,
            "previous_requests": self.previous_requests,
            "active_servers":    self.active_servers,
            "cpu_utilization":   round(min(1.5, self.current_requests / max(capacity, 1)), 4),
            "queue_length":      self.queue_length,
        }
