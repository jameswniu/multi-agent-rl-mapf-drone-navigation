# app.py
# ----------------------------
# This file defines the FastAPI application that serves predictions
# from the PPO drone agent. In production, we add:
#  - Structured logging (so logs are consistent across modules)
#  - Metrics (so Prometheus can monitor performance)
#  - Error handling (so the API fails gracefully)
#  - Health check endpoint (so Kubernetes can verify the service is alive)

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse, JSONResponse
from pydantic import BaseModel
from typing import List
import time

# Prometheus metrics utilities
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from utils.metrics import REQUEST_COUNT, REQUEST_LATENCY

# Central logging utility
from utils.logger import get_logger

# Custom error handling
from utils.errors import APIError, error_handler

# Domain-specific code: environment and agent
from env.drone_env import DroneEnv
from agents.ppo_agent import PPOAgent


class StateInput(BaseModel):
    """Schema for the prediction request body.

    ``state`` is one environment observation: a row per drone, each row being
    ``[x, y, goal_x, goal_y, steps_remaining, blocked_up, blocked_down,
    blocked_left, blocked_right]``. It is a list of rows rather than a mapping
    because that is what the policy consumes; ``PPOAgent.predict`` builds a
    tensor straight from it.
    """

    state: List[List[float]]


# -------------------------------------------------
# Setup
# -------------------------------------------------
app = FastAPI(title="Drone Pathfinding API")
logger = get_logger(__name__)

# Register custom error handler
app.add_exception_handler(APIError, error_handler)

# Load environment and agent once at startup
env = DroneEnv("configs/env.yaml")
agent = PPOAgent(env)


@app.on_event("startup")
def load_agent_weights():
    """Load agent weights if available on startup."""
    try:
        agent.load("models/ppo_drone.pt")
    except FileNotFoundError:
        logger.warning(
            "Model weights not found at 'models/ppo_drone.pt';"
            " starting without pretrained weights."
        )


# -------------------------------------------------
# Middleware: automatically runs before/after each request
# - Records request latency
# - Increments counters
# - Logs request info
# -------------------------------------------------
@app.middleware("http")
async def add_metrics(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time

    endpoint = request.url.path
    REQUEST_COUNT.labels(method=request.method, endpoint=endpoint).inc()
    REQUEST_LATENCY.labels(method=request.method, endpoint=endpoint).observe(process_time)

    logger.info(f"{request.method} {endpoint} completed in {process_time:.3f}s")
    return response


# -------------------------------------------------
# Routes
# -------------------------------------------------

@app.post("/predict")
def predict(payload: StateInput):
    """
    Accepts a payload matching :class:`StateInput` and returns the agent's action.

    Example request:  { "state": [[0, 0, 19, 19, 200, 1, 1, 1, 0]] }
    Example response: { "actions": [4], "action_names": ["right"] }

    Actions are returned both ways on purpose. The indices are what the
    environment's ``step()`` accepts; the names are what a human reads in a log.
    """
    logger.info("Received predict request")

    rows, cols = env.observation_space.shape
    # A malformed observation is a client error, not a server fault, so it is
    # rejected before it can reach the policy as an opaque tensor failure.
    if len(payload.state) != rows:
        raise HTTPException(
            status_code=422,
            detail=f"state must have exactly {rows} rows, one per drone, got {len(payload.state)}",
        )
    bad = [i for i, row in enumerate(payload.state) if len(row) != cols]
    if bad:
        raise HTTPException(
            status_code=422,
            detail=f"each row must have exactly {cols} values; row {bad[0]} has {len(payload.state[bad[0]])}",
        )

    try:
        action = agent.predict(payload.state)
    except Exception as e:
        # Wrap raw exceptions in a clean APIError
        raise APIError(f"Prediction failed: {str(e)}", status_code=500)
    actions = [int(a) for a in action]
    return {"actions": actions, "action_names": [env.action_map[a] for a in actions]}


@app.get("/metrics")
def metrics():
    """
    Exposes Prometheus metrics.
    Monitoring systems scrape this endpoint automatically.
    """
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/healthz")
def healthz():
    """
    Simple health check endpoint.
    Returns 200 OK if service is alive.
    Used by Kubernetes liveness/readiness probes.
    """
    return JSONResponse(content={"status": "ok"})
