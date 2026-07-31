"""
API tests, run against the real agent.

These deliberately do not stub PPOAgent. An earlier version of this file
replaced the agent with a dummy that returned a constant, which meant the
/predict contract could be broken without a single test noticing, and it was.
"""

import pytest
from fastapi.testclient import TestClient

from src.api.app import app, env


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_predict_returns_a_legal_action(client):
    """A real observation produces an action inside the action space."""
    obs, _ = env.reset()
    response = client.post("/predict", json={"state": obs.tolist()})

    assert response.status_code == 200
    body = response.json()
    assert body["action"] in range(env.action_space.n)
    assert body["action_name"] == env.action_map[body["action"]]


def test_predict_rejects_a_wrong_length_state(client):
    """Too few values is a client error, not a policy failure."""
    response = client.post("/predict", json={"state": [0, 0, 1]})
    assert response.status_code == 422


def test_predict_rejects_a_mapping_state(client):
    """The old contract accepted a dict here; it never worked, so it is refused."""
    response = client.post("/predict", json={"state": {"x": 1, "y": 2}})
    assert response.status_code == 422


def test_healthz(client):
    assert client.get("/healthz").json() == {"status": "ok"}
