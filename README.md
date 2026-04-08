---
title: Cloud Resource Allocation
emoji: 🚀
colorFrom: blue
colorTo: red
sdk: docker
sdk_version: 3.11
app_port: 7860
pinned: false
tags:
- openenv
---

# Cloud Resource Allocation

A FastAPI-based cloud autoscaling environment for benchmarking autoscaler agents against synthetic workloads.

The repo provides:
- A reusable OpenEnv-style environment for cloud autoscaling (`app/env.py`)
- A deterministic workload simulator with steady, diurnal, and flash-crowd patterns (`app/simulator.py`)
- A FastAPI server exposing task discovery, reset, step, and grading endpoints (`app/main.py`)
- A rule-based baseline autoscaler agent (`agent.py`)
- An example inference client that runs an episode against the server (`inference.py`)
- Task definitions for `easy`, `medium`, and `hard` scenarios in `tasks/*.yaml`

---

## Features

- `FastAPI` environment server with REST endpoints for interactive agent control
- Task configs driven by YAML files for easy scenario prototyping
- Step-level reward and episode scoring with latency, cost, instability, and SLA penalties
- Baseline adaptive agent that scales servers using a simple threshold policy
- Deterministic workloads seeded for reproducible evaluation

---

## Repository Structure

- `app/main.py` - FastAPI application and HTTP endpoint definitions
- `app/env.py` - OpenEnv-style environment wrapper around the simulator
- `app/simulator.py` - Cloud autoscaling simulator and workload generator
- `app/models.py` - Pydantic request/response and task schemas
- `app/grader.py` - Episode grader and scoring logic
- `agent.py` - Example adaptive autoscaler agent
- `inference.py` - Example client that interacts with the server
- `tasks/` - Task configuration YAML files: `easy.yaml`, `medium.yaml`, `hard.yaml`
- `requirements.txt` - Python dependencies

---

## Requirements

- Python 3.11+ recommended
- `fastapi`
- `uvicorn[standard]`
- `pydantic`
- `numpy`
- `pyyaml`

Install dependencies with:

```bash
pip install -r requirements.txt
```

---

## Running the Server

Start the FastAPI server from the repository root:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The server exposes the following endpoints:

- `GET /health` - health check and task count
- `GET /tasks` - list available tasks and action schema
- `POST /reset` - create a new episode for a selected task
- `POST /step` - advance the environment by one action
- `GET /state` - read the current observation without stepping
- `GET /grader` - score the completed episode

---

## Example Inference Flow

Use `inference.py` as a simple example client to run an episode.

1. Start the API server.
2. Run:

```bash
python inference.py
```

This script:
- fetches available tasks
- resets the environment for the first task
- repeatedly calls `/step` with the `AdaptiveAgent` action
- updates the agent using feedback from each step
- retrieves the final graded episode score

---

## Task Definitions

The environment supports three pre-configured scenarios:

- `easy` — steady workload for basic stability
- `medium` — diurnal workload for proactive scaling
- `hard` — flash crowd workload for rapid reaction

Each task file includes:
- workload settings
- server capacity limits
- grading bounds and weights
- maximum episode length

---

## Agent Interface

Agents should use the environment via the following flow:

```python
from app.models import AutoscalerObservation
from agent import AdaptiveAgent
from app.env import AutoscalerEnv
from app.models import TaskConfig

agent = AdaptiveAgent()
# config = load from task YAML or via server
# env = AutoscalerEnv()
# obs = env.reset(config)

# while not done:
#     action = agent.act(obs)
#     result = env.step(action)
#     agent.learn(result.reward, result.info)
#     obs = result.observation
```

The allowed actions are:
- `-1` — scale down
- `0` — no change
- `1` — scale up

---

## Observation Space

The observation is an `AutoscalerObservation` object with the following fields:

- `current_requests` (int): Demand at current timestep
- `previous_requests` (int): Demand at previous timestep  
- `active_servers` (int): Supply of active servers
- `cpu_utilization` (float): Load per server (0.0 to 1.5)
- `queue_length` (int): Backlog of requests

---

## Scoring

The grader computes a final score in `[0.0, 1.0]` using normalized penalties for:
- latency proxy
- server cost
- instability
- SLA violations

The same normalized components are used by the environment reward function, so agents can learn against a consistent objective.

---

## Baseline Scores

The included `AdaptiveAgent` (rule-based autoscaler) achieves the following scores:

- `easy`: 0.9200
- `medium`: 0.8843  
- `hard`: 0.7224

These scores represent a reasonable starting point for reinforcement learning agents.

---

## Notes

- The cloud simulator uses a 1-step scaling delay: actions taken at step `t` apply at step `t+1`.
- Workloads are deterministic for a given task seed, enabling reproducible trials.

---

## License

No license is specified in the repo. Add one if you want to share or publish this project.
