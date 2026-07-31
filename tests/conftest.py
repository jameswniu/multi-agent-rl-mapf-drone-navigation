"""
PyTest Fixtures
---------------
Reusable fixtures for the test suite.

This file used to wrap its imports in ``try/except Exception`` and substitute
stub classes when the real ones failed to import. That masked two separate
production bugs: a broken gymnasium import surfaced as a confusing
``AttributeError`` on a stub, and a broken /predict contract passed its tests
because the stub returned a constant. The imports are now plain, so a missing
dependency fails loudly at collection where it belongs.
"""

import pytest

from env.drone_env import DroneEnv
from agents.ppo_agent import PPOAgent


@pytest.fixture(scope="function")
def env():
    """A fresh DroneEnv per test, closed afterwards."""
    e = DroneEnv()
    try:
        yield e
    finally:
        e.close()


@pytest.fixture(scope="function")
def agent():
    """A PPOAgent tied to its own fresh DroneEnv."""
    e = DroneEnv()
    a = PPOAgent(e)
    try:
        yield a
    finally:
        e.close()


@pytest.fixture(scope="function")
def model_path(tmp_path):
    """A temporary path for saving and loading model weights."""
    return tmp_path / "ppo_test_model.pt"
