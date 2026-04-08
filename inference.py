import os
import requests
from agent import AdaptiveAgent
BASE_URL = "http://localhost:8000"

print("[START]")

# 1. Get available tasks
tasks_res = requests.get(f"{BASE_URL}/tasks")
tasks = tasks_res.json()["tasks"]

# pick first task (you can change later)
task_id = tasks[1]["task_id"]
print(f"[INFO] Using task: {task_id}")

# 2. Reset environment
reset_res = requests.post(
    f"{BASE_URL}/reset",
    json={"task_id": task_id}
)
data = reset_res.json()

session_id = data["session_id"]
state = data["observation"]
from app.models import AutoscalerObservation
agent  = AdaptiveAgent()
done = False

# 3. Run episode
step_count = 0
obs = AutoscalerObservation(**state)
while not done:
    action = agent.act(obs)  # Use the adaptive agent to choose an action

    print(f"[STEP {step_count}] action={action}")

    step_res = requests.post(
        f"{BASE_URL}/step",
        json={
            "session_id": session_id,
            "action": {"scale_change": action}
        }
    )

    data = step_res.json()
    obs = AutoscalerObservation(**data["observation"])
    #state = data["observation"]
    reward = data["reward"]
    done = data["done"]
    info = data["info"]

    agent.learn(reward, info)  # Update the agent based on feedback

    if step_count%10 == 0:
        print(f"[Step {step_count}] Action: {action}, Reward: {reward}, Servers: {obs.active_servers}")

    step_count += 1

print("[END]")

# 4. Get final score
grader_res = requests.get(
    f"{BASE_URL}/grader",
    params={"session_id": session_id}
)

print("[RESULT]", grader_res.json())