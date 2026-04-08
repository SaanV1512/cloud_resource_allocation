import os
import requests
from agent import AdaptiveAgent
from app.models import AutoscalerObservation

# --- MANDATORY CONFIGURATION ---
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
MODEL_NAME = os.getenv("MODEL_NAME", "CloudScaler-RuleBased-Baseline")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or "dummy_token"
BENCHMARK = "cloud_resource_allocation"
SUCCESS_SCORE_THRESHOLD = 0.5

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: str = "null") -> None:
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: list) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    success_val = str(success).lower()
    print(f"[END] success={success_val} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

def run_task(task_id):
    # --- Initialization (Fixed Scope) ---
    agent = AdaptiveAgent()
    agent.epsilon = 0 
    
    rewards = []
    step_count = 0
    score = 0.0
    success = False
    session_id = None
    started_logging = False

    try:
        # 1. Reset Environment
        reset_res = requests.post(
            f"{API_BASE_URL}/reset", 
            json={"task_id": task_id}, 
            timeout=10
        )
        reset_res.raise_for_status()
        
        data = reset_res.json()
        session_id = data["session_id"]
        obs_data = data["observation"]
        obs = AutoscalerObservation(**obs_data)

        # Only start logging once the environment actually responds
        log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)
        started_logging = True

        done = False
        while not done:
            step_count += 1
            action = agent.act(obs)
            error_val = "null"

            # 2. Step Environment
            step_res = requests.post(
                f"{API_BASE_URL}/step",
                json={
                    "session_id": session_id,
                    "action": {"scale_change": action}
                },
                timeout=5
            )

            if step_res.status_code != 200:
                error_val = f"HTTP_{step_res.status_code}"
                # Log the error step and exit loop
                log_step(step=step_count, action=str(action), reward=0.0, done=True, error=error_val)
                break
            
            step_data = step_res.json()
            obs = AutoscalerObservation(**step_data["observation"])
            reward = step_data.get("reward", 0.0)
            done = step_data.get("done", False)
            info = step_data.get("info", {})

            # Agent Learning & Memory
            agent.learn(reward, info)
            rewards.append(reward)

            # Mandatory [STEP] log
            log_step(step=step_count, action=str(action), reward=reward, done=done, error=error_val)

        # 3. Final Grader Call
        if session_id:
            grader_res = requests.get(
                f"{API_BASE_URL}/grader", 
                params={"session_id": session_id},
                timeout=5
            )
            if grader_res.status_code == 200:
                result_data = grader_res.json()
                score = result_data.get("score", 0.0)
                success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] Task {task_id} failed: {e}", flush=True)
        # score, success, and step_count remain at initial values

    finally:
        # Guarantee the [END] log is emitted if [START] was reached
        if started_logging:
            log_end(success=success, steps=step_count, score=score, rewards=rewards)

if __name__ == "__main__":
    try:
        # Fetch task list from the environment
        tasks_res = requests.get(f"{API_BASE_URL}/tasks", timeout=5)
        tasks_res.raise_for_status()
        available_tasks = tasks_res.json()["tasks"]
        
        for t in available_tasks:
            run_task(t["task_id"])
            
    except Exception as e:
        print(f"[CRITICAL] Failed to connect to Environment at {API_BASE_URL}: {e}")