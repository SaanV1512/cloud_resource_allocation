import os
import time
import requests
import textwrap
from openai import OpenAI
from app.models import AutoscalerObservation

# The endpoint where your FastAPI environment is running
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:7860")

# The endpoint for the LLM provider (separate from the environment!)
LLM_API_BASE = os.getenv("LLM_API_BASE") or os.getenv("OPENAI_API_BASE")

MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
BENCHMARK = "cloud_resource_allocation"
SUCCESS_SCORE_THRESHOLD = 0.5 

# MANDATORY: Initialize OpenAI Client (Pointing to LLM provider, NOT Environment)
client = OpenAI(base_url=LLM_API_BASE, api_key=HF_TOKEN)

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an AI managing a cloud autoscaling cluster.
    At each step, you are given the current demand (requests), queue length, active servers, and CPU utilization.
    Your goal is to maximize total reward by balancing server cost and latency SLA.
    You can take one action:
    -1 (scale down by 1 server)
    0 (do nothing)
    1 (scale up by 1 server)
    
    Reply with EXACTLY ONE NUMBER: -1, 0, or 1. Do not output any other text, prefixes, or explanation.
    """
).strip()


def log_start(task: str) -> None:
    print(f"[START] task={task}", flush=True)

def log_step(step: int, reward: float) -> None:
    print(f"[STEP] step={step} reward={reward:.2f}", flush=True)

def log_end(task: str, score: float, steps: int) -> None:
    print(f"[END] task={task} score={score:.3f} steps={steps}", flush=True)


def build_user_prompt(step: int, obs: AutoscalerObservation, last_reward: float) -> str:
    return textwrap.dedent(
        f"""
        Step: {step}
        Demand: {obs.current_requests}
        Queue Length: {obs.queue_length}
        Active Servers: {obs.active_servers}
        CPU Utilization: {obs.cpu_utilization:.2f}
        Previous Step Reward: {last_reward:.2f}
        
        Send your next action choice (-1, 0, or 1):
        """
    ).strip()

def get_action_from_llm(step: int, obs: AutoscalerObservation, last_reward: float) -> int:
    user_prompt = build_user_prompt(step, obs, last_reward)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=10,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        
        # Safely parse the exact digit from text
        if "-1" in text:
            return -1
        elif "1" in text:
            return 1
        else:
            return 0
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return 0  # Fallback safe action


def run_task(task_id):
    # MANDATORY: Start logging IMMEDIATELY so validator sees the [START] block
    log_start(task=task_id)

    rewards = []
    step_count = 0
    score = 0.0
    success = False
    session_id = None

    try:
        # 1. Reset Environment
        reset_res = requests.post(
            f"{API_BASE_URL}/reset", 
            json={"task_id": task_id}, 
            timeout=15
        )
        reset_res.raise_for_status()
        
        data = reset_res.json()
        session_id = data["session_id"]
        obs_data = data["observation"]
        obs_data.setdefault("previous_requests", 0)  # Handle missing previous_requests
        obs = AutoscalerObservation(**obs_data)

        done = False
        last_reward = 0.0
        
        while not done:
            step_count += 1
            
            # --- LLM INFERENCE ---
            action = get_action_from_llm(step_count, obs, last_reward)
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

            last_reward = reward
            rewards.append(reward)

            # Mandatory [STEP] log
            log_step(step=step_count, reward=reward)

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

    finally:
        # Final block: emit [END]
        log_end(task=task_id, score=score, steps=step_count)


if __name__ == "__main__":
    print(f"# Attempting to connect to Environment at {API_BASE_URL}...", flush=True)

    # 1. Wait for server (up to 30s)
    env_ready = False
    for i in range(15):
        try:
            # Try both /health and just the root (HF Spaces might only respond to /)
            requests.get(f"{API_BASE_URL}/health", timeout=2)
            env_ready = True
            print(f"# Environment is LIVE at {API_BASE_URL}", flush=True)
            break
        except Exception as e:
            print(f"# Waiting for environment... ({i+1}/15) {e}", flush=True)
            time.sleep(2)

    available_tasks = []
    try:
        # 2. Fetch task list from the environment
        tasks_res = requests.get(f"{API_BASE_URL}/tasks", timeout=10)
        tasks_res.raise_for_status()
        available_tasks = tasks_res.json().get("tasks", [])
    except Exception as e:
        print(f"# Task discovery failed: {e}", flush=True)

    # 3. Fallback: If no tasks found, try "easy" anyway
    if not available_tasks:
        print("# No tasks discovered via API. Falling back to default 'easy' task.", flush=True)
        available_tasks = [{"task_id": "easy"}]
        
    for t in available_tasks:
        try:
            run_task(t["task_id"])
        except Exception as e:
            print(f"# Task {t['task_id']} failed at runtime: {e}", flush=True)