# Multi-Agent RL MAPF Drone System
**Instant-alignment drone AI for safe, adaptive, real-time flight control.**

Production-ready multi agent drone AI with PPO, FastAPI backend, Docker deployment, and monitoring.

---

## Overview  
This system combines Reinforcement Learning (RL), Multi-Agent Path Finding (MAPF), and real-time alignment scoring to control drones safely. Instead of waiting for delayed success signals, the system scores each action instantly against internal principles (the “Constitution”) so it can act safely and efficiently in real time.

The architecture integrates:  
- A custom drone environment  
- A PPO agent in PyTorch  
- Alignment scoring for safety and smoothness  
- API endpoints for predictions, health, and metrics  
- Monitoring hooks for production environments  

---

## System Architecture

### Core Agents
- **Ingestion Agent**  
  Collects raw sensor data (camera, GPS, IMU, barometer) at fixed frequency and queues it.  

- **Preprocess Agent**  
  Filters and compresses raw inputs into compact features.  

- **Prediction Agent**  
  Uses the PPO policy to select the drone’s next action (hover, climb, move, turn).  

### Alignment Scoring
Each action is scored instantly against values:  
- Maintain safety margins  
- Save energy  
- Move smoothly  

Weighted scores are combined and tested for stability. The Alignment Policy selects the safest, most stable action.  

### Safety and Oversight
- **Safety Controller** – vetoes unsafe actions (for example, entering a no-fly zone).  
- **Supervisor** – starts/stops agents, restarts failures, quarantines unstable modules.  

---

## Repository Structure
    multi-agent-rl-mapf-drone-system/
    ├── configs/              # Training and environment configs
    ├── docker/               # Docker and Kubernetes deployment files
    ├── docs/                 # Architecture and deployment documentation
    ├── monitoring/           # Prometheus, Grafana, Alertmanager configs
    ├── scripts/              # Helper scripts (run, train, deploy)
    ├── src/                  # Source code (agents, API, env, utils)
    └── tests/                # Unit, integration, and load tests

---

## System Design Files
- High-Level Diagram: `drone_high_lv_system_design.png`  
- Low-Level Diagram: `drone_low_lv_system_design.png`  
- Reward Patterns Reference: `drones_matrix_RL.png`  
- Detailed Specs: `low_level_design/`  

---

## Installation

Clone the repository and install dependencies (ensure you are using Python 3.10):

    git clone <repo-url>
    cd multi-agent-rl-mapf-drone-system
    pip install -r requirements.txt

Or, with pyproject.toml:

    pip install .

---

## Usage

Train the model:

    python src/main.py --config configs/train.yaml

Run the API server:

    uvicorn src.api.app:app --reload

Example request:

    curl -X POST http://localhost:8000/predict          -H "Content-Type: application/json"          -d '{"state": {"drone": [0,0], "goal": [5,5]}}'

Example response:

    {"action": "move_up"}

---

## Deployment

Using Docker:

    docker build -t drone-system -f docker/Dockerfile.prod .
    docker run -p 8080:8080 drone-system

Using Kubernetes:

    kubectl apply -f docker/k8s/deployment.yaml
    kubectl apply -f docker/k8s/service.yaml

---

## Monitoring
- Metrics: `/metrics` endpoint (Prometheus)  
- Health: `/healthz` endpoint (Kubernetes probes)  
- Dashboard: `monitoring/grafana-dashboard.json`  
- Alerting: `monitoring/alertmanager.yml`  

Key Grafana panels:  
- Request latency (p95)  
- Requests by endpoint  
- Training reward distribution  
- Error rate  

---

## Tests

Run all tests:

    pytest --cov=src

Covers:  
- Unit tests (`tests/test_api.py`, `tests/test_training.py`)  
- Integrity tests (`tests/test_integrity.py`)  
- Integration tests (`tests/test_integration.py`)  
- Load tests (`tests/test_load.py`)  

---

## Contributing  
See [CONTRIBUTING.md](CONTRIBUTING.md).  

---

## License  
Apache-2.0 License – see [LICENSE](LICENSE).  
