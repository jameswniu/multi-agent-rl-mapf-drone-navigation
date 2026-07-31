"""
Integrity Validator Tests
-------------------------
These tests confirm that our integrity validators are working.

What are we testing?
- The environment validator should NOT raise errors in normal runs.
- The policy validator SHOULD raise errors when given bad inputs.
"""

import torch
from env.drone_env import DroneEnv
from agents.ppo_agent import PPOAgent
from integrity_validators import PolicyIntegrityValidator


def test_env_integrity_validator_clean_run():
    """
    Test that DroneEnv runs normally without errors.

    Steps:
    1. Create environment.
    2. Reset it.
    3. Take one legal step.
    4. Assert there are no integrity errors in the info dictionary.
    """
    env = DroneEnv()
    _, _ = env.reset()
    # One action per drone; hovering is always legal.
    _, _, _, _, info = env.step([0] * env.num_drones)
    assert "integrity_errors" not in info
    env.close()


def test_policy_integrity_validator_catches_invalid():
    """
    Test that the policy validator flags bad outputs.

    Steps:
    1. Create environment + agent.
    2. Create a policy validator manually.
    3. Feed it bad inputs:
       - action_probs that sum to more than 1
       - a value that is NaN
       - an invalid action index
    4. Assert that errors are returned.
    """
    env = DroneEnv()
    agent = PPOAgent(env)
    validator = PolicyIntegrityValidator(env.action_space)

    errors = validator.validate(
        action_probs=torch.tensor([0.5, 0.6]),   # invalid: sums > 1
        value=torch.tensor(float("nan")),        # invalid: NaN
        action=999                               # invalid: not in action space
    )

    # At least one drift or hallucination error should be present
    assert any(err["type"] in ["drift", "hallucination"] for err in errors)

    env.close()
