"""
API tests, run against the real agent.

These deliberately do not stub PPOAgent. An earlier version of this file
replaced the agent with a dummy that returned a constant, which meant the
/predict contract could be broken without a single test noticing, and it was.
"""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api.app import app, env


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_predict_returns_a_legal_action_per_drone(client):
    """A real observation produces one legal action for every drone."""
    obs, _ = env.reset()
    response = client.post("/predict", json={"state": obs.tolist()})

    assert response.status_code == 200
    body = response.json()
    assert len(body["actions"]) == env.num_drones
    assert env.action_space.contains(np.asarray(body["actions"], dtype=np.int64))
    assert body["action_names"] == [env.action_map[a] for a in body["actions"]]


def test_predict_rejects_a_wrong_row_count(client):
    """One row per drone; too few is a client error, not a policy failure."""
    response = client.post("/predict", json={"state": [[0] * 9]})
    assert response.status_code == 422


def test_predict_rejects_a_wrong_row_width(client):
    """Right number of drones, wrong number of features."""
    response = client.post("/predict", json={"state": [[0, 0, 1]] * env.num_drones})
    assert response.status_code == 422


def test_predict_rejects_a_mapping_state(client):
    """The old contract accepted a dict here; it never worked, so it is refused."""
    response = client.post("/predict", json={"state": {"x": 1, "y": 2}})
    assert response.status_code == 422


def test_healthz(client):
    assert client.get("/healthz").json() == {"status": "ok"}
