from typing import List, Dict
# -------------------------------
# CONFIG (can move to YAML later)
# -------------------------------

TASK_CONFIG = {
    "easy": {
        "weights": {"L": 0.35, "C": 0.40, "I": 0.10, "V": 0.15},
        "L_max": 500,
        "S_max": 10,
        "I_max": 1.0,
        "sla_threshold": 200,
    },
    "medium": {
        "weights": {"L": 0.45, "C": 0.25, "I": 0.15, "V": 0.15},
        "L_max": 800,
        "S_max": 10,
        "I_max": 1.0,
        "sla_threshold": 300,
    },
    "hard": {
        "weights": {"L": 0.55, "C": 0.15, "I": 0.10, "V": 0.20},
        "L_max": 1200,
        "S_max": 10,
        "I_max": 1.0,
        "sla_threshold": 400,
    },
}

def compute_metrics(logs: List[Dict], task_name: str) -> Dict:
    """
    logs: list of dicts with keys:
        - state (State object)
        - action
        - reward
    """

    config = TASK_CONFIG[task_name]
    sla_threshold = config["sla_threshold"]

    total_latency = 0.0
    total_servers = 0.0
    instability = 0.0
    sla_violations = 0.0

    T = len(logs)

    for i in range(T):
        state = logs[i]["state"]

        # latency proxy
        latency = state.queue
        total_latency += latency

        # cost proxy
        total_servers += state.servers

        # SLA violation
        if latency > sla_threshold:
            sla_violations += 1

        # instability (change in servers)
        if i > 0:
            prev_servers = logs[i - 1]["state"].servers
            instability += abs(state.servers - prev_servers)

    avg_latency = total_latency / T
    avg_cost = total_servers / T
    instability = instability / max(1, (T - 1))
    sla_violations = sla_violations / T

    return {
        "avg_latency": avg_latency,
        "avg_cost": avg_cost,
        "instability": instability,
        "sla_violations": sla_violations,
    }


def normalize(metrics: Dict, task_name: str) -> Dict:
    config = TASK_CONFIG[task_name]

    L = min(1.0, metrics["avg_latency"] / config["L_max"])
    C = min(1.0, metrics["avg_cost"] / config["S_max"])
    I = min(1.0, metrics["instability"] / config["I_max"])
    V = metrics["sla_violations"]  # already 0–1

    return {"L": L, "C": C, "I": I, "V": V}


def score(metrics: Dict, task_name: str) -> float:
    config = TASK_CONFIG[task_name]
    weights = config["weights"]

    norm = normalize(metrics, task_name)

    loss = (
        weights["L"] * norm["L"]
        + weights["C"] * norm["C"]
        + weights["I"] * norm["I"]
        + weights["V"] * norm["V"]
    )

    final_score = max(0.0, min(1.0, 1.0 - loss))
    return final_score


def grade_episode(logs: List[Dict], task_name: str) -> Dict:
    """
    Returns full evaluation for one episode
    """
    metrics = compute_metrics(logs, task_name)
    final_score = score(metrics, task_name)

    return {
        "task": task_name,
        "score": final_score,
        **metrics,
    }


TASK_WEIGHTS = {
    "easy": 0.2,
    "medium": 0.3,
    "hard": 0.5,
}


def aggregate_scores(results: List[Dict]) -> float:
    """
    results: list of outputs from grade_episode()
    """
    total = 0.0

    for r in results:
        task = r["task"]
        total += TASK_WEIGHTS[task] * r["score"]

    return total
