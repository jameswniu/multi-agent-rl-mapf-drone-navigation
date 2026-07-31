# Architecture

- **Agents**: a shared-weight PPO policy driving every drone.
- **Environment**: `DroneEnv`, a gymnasium grid world with obstacles and local sensing.
- **Path finding**: vertex, swap and stationary conflicts detected and refused each step.
- **Integrity**: validators that report drift and hallucination, plus a Safety Controller that vetoes.
- **API**: FastAPI serving predictions, metrics and health.
- **Monitoring**: Prometheus and Grafana.
- **Deployment**: Docker, Compose, Kubernetes.

The full design, including the parts not implemented, is in
[`architecture/low_level_design.txt`](../architecture/low_level_design.txt) and
rendered in the README.
