"""
Test script: runs the AdaptiveAgent against AutoscalerEnv
for all 3 tasks and prints step-by-step results + grader scores.

Usage:
    cd cloud-rl
    python test_env.py
"""

import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent))

from app.models import TaskConfig, AutoscalerAction
from app.env import AutoscalerEnv
from app.grader import grade_episode
from agent import AdaptiveAgent


def load_task(task_id: str) -> TaskConfig:
    yaml_path = Path(__file__).parent / "tasks" / f"{task_id}.yaml"
    with open(yaml_path) as f:
        raw = yaml.safe_load(f)
    return TaskConfig(**raw)


def run_task(task_id: str):
    config = load_task(task_id)
    env    = AutoscalerEnv()
    agent  = AdaptiveAgent()

    # reset() returns an AutoscalerObservation — passed directly to agent
    obs  = env.reset(config)
    done = False
    step_num = 0

    print(f"\n{'='*65}")
    print(f"  TASK: {task_id.upper()} — {config.description}")
    print(f"  Steps: {config.max_steps} | Pattern: {config.workload_pattern.value}")
    print(f"{'='*65}")
    print(f"{'Step':>4} | {'Demand':>6} | {'Servers':>7} | {'Queue':>5} | {'CPU':>5} | {'Reward':>7} | {'Action':>6} | {'SLA':>3}")
    print(f"{'-'*4}-+-{'-'*6}-+-{'-'*7}-+-{'-'*5}-+-{'-'*5}-+-{'-'*7}-+-{'-'*6}-+-{'-'*3}")

    while not done:
        # Mismatch 1 fix: pass obs object directly (no dict mapping needed)
        scale_change = agent.act(obs)

        action = AutoscalerAction(scale_change=scale_change)
        result = env.step(action)

        step_num += 1
        obs  = result.observation
        done = result.done

        # Mismatch 2 fix: agent.learn() now gets cost + sla_violations
        agent.learn(result.reward, result.info)

        sla_flag = "❌" if result.info["sla_violations"] else "✅"

        print(
            f"{step_num:>4} | "
            f"{obs.current_requests:>6} | "
            f"{obs.active_servers:>7} | "
            f"{obs.queue_length:>5} | "
            f"{obs.cpu_utilization:>5.2f} | "
            f"{result.reward:>7.4f} | "
            f"{scale_change:>+6d} | "
            f"{sla_flag}"
        )

    # --- Grader Score (Mismatch 3 fix: uses app/grader.py) ---
    scores = grade_episode(env.episode_history, config)

    print(f"\n--- Episode Summary ---")
    print(f"  Score:          {scores['score']:.4f}  (0=worst, 1=best)")
    print(f"  Avg latency:    {scores['avg_latency']:.2f}  (normalised: {scores['L_hat']:.4f})")
    print(f"  Avg servers:    {scores['avg_cost']:.2f}  (normalised: {scores['C_hat']:.4f})")
    print(f"  Avg instability:{scores['avg_instability']:.4f}  (normalised: {scores['I_hat']:.4f})")
    print(f"  SLA violations: {scores['sla_violations']} / {scores['steps']} steps  ({scores['sla_rate']*100:.1f}%)")

    return scores


def main():
    print("Cloud Autoscaler — Environment + Grader Test")
    print("Testing AdaptiveAgent on all 3 tasks...\n")

    all_scores = []
    for task_id in ["easy", "medium", "hard"]:
        try:
            scores = run_task(task_id)
            all_scores.append(scores)
        except Exception as e:
            print(f"\n❌ TASK {task_id.upper()} FAILED: {e}")
            import traceback
            traceback.print_exc()

    # Aggregate score (matches teammate's TASK_WEIGHTS)
    if len(all_scores) == 3:
        task_weights = {"easy": 0.2, "medium": 0.3, "hard": 0.5}
        aggregate = sum(task_weights[s["task"]] * s["score"] for s in all_scores)
        print(f"\n{'='*65}")
        print(f"  AGGREGATE SCORE: {aggregate:.4f}")
        for s in all_scores:
            print(f"    {s['task']:6} → {s['score']:.4f} (weight: {task_weights[s['task']]})")
        print(f"{'='*65}")


if __name__ == "__main__":
    main()
