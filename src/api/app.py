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
from src.utils.metrics import REQUEST_COUNT, REQUEST_LATENCY

# Central logging utility
from src.utils.logger import get_logger

# Custom error handling
from src.utils.errors import APIError, error_handler

# Domain-specific code: environment and agent
from src.env.drone_env import DroneEnv
from src.agents.ppo_agent import PPOAgent


class StateInput(BaseModel):
    """Schema for the prediction request body.

    ``state`` is one environment observation, flat and in the order the
    environment emits it: ``[x, y, goal_x, goal_y, steps_remaining]``. It is a
    list of numbers rather than a mapping because that is what the policy
    consumes; ``PPOAgent.predict`` builds a tensor straight from it.
    """

    state: List[float]


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

    Example request:  { "state": [0, 0, 19, 19, 200] }
    Example response: { "action": 4, "action_name": "right" }

    The action is returned both ways on purpose. The index is what the
    environment's ``step()`` accepts; the name is what a human reads in a log.
    """
    logger.info("Received predict request")

    expected = int(env.observation_space.shape[0])
    if len(payload.state) != expected:
        # A wrong-length observation is a client error, not a server fault, so
        # it is rejected before it can reach the policy as an opaque failure.
        raise HTTPException(
            status_code=422,
            detail=f"state must have exactly {expected} values, got {len(payload.state)}",
        )

    try:
        action = agent.predict(payload.state)
    except Exception as e:
        # Wrap raw exceptions in a clean APIError
        raise APIError(f"Prediction failed: {str(e)}", status_code=500)
    return {"action": action, "action_name": env.action_map[action]}


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
