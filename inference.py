import requests

BASE_URL = "http://localhost:8000"

print("[START]")

# 1. Get available tasks
tasks_res = requests.get(f"{BASE_URL}/tasks")
tasks = tasks_res.json()["tasks"]

# pick first task (you can change later)
task_id = tasks[0]["task_id"]
print(f"[INFO] Using task: {task_id}")

# 2. Reset environment
reset_res = requests.post(
    f"{BASE_URL}/reset",
    json={"task_id": task_id}
)
data = reset_res.json()

session_id = data["session_id"]
state = data["observation"]

done = False

# 3. Run episode
step_count = 0

while not done:
    action = 0  # dummy action for now

    print(f"[STEP {step_count}] action={action}")

    step_res = requests.post(
        f"{BASE_URL}/step",
        json={
            "session_id": session_id,
            "action": {"scale_change": action}
        }
    )

    data = step_res.json()

    state = data["observation"]
    reward = data["reward"]
    done = data["done"]

    print(f"   reward={reward}")

    step_count += 1

print("[END]")

# 4. Get final score
grader_res = requests.get(
    f"{BASE_URL}/grader",
    params={"session_id": session_id}
)

print("[RESULT]", grader_res.json())