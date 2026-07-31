"""
Integrity Validators
--------------------
This file defines two validator classes:
1. IntegrityValidator -> for the environment (DroneEnv)
2. PolicyIntegrityValidator -> for the PPO agent

What do they do?
-> They catch two kinds of problems:
   - Drift: values that go outside expected ranges (e.g. obs out of bounds, NaN rewards).
   - Hallucination: invalid outputs that should never happen (e.g. illegal actions).
-> They return errors in a structured format: {"type": ..., "field": ..., "msg": ...}
   so other code (like IntegrityStats) can count and report them.
"""

import numpy as np
import torch


# ---------------- Environment Validator ----------------

class IntegrityValidator:
    """
    Validator for the environment side (DroneEnv).
    Checks:
    - Observations are inside the observation space.
    - Actions are inside the action space.
    - Rewards are finite numbers.
    """

    def __init__(self, action_space, observation_space):
        self.action_space = action_space
        self.observation_space = observation_space

    def _as_action(self, action):
        """Shape an action the way its space expects it.

        A MultiDiscrete space holds one move per drone and wants an integer
        vector; a Discrete space wants a single integer. Casting a vector with
        ``int()`` raises, which previously made every legal multi-drone action
        look like a hallucination.
        """
        if hasattr(self.action_space, "nvec"):
            return np.asarray(action, dtype=np.int64).reshape(self.action_space.nvec.shape)
        return int(action)

    def validate(self, obs, action, reward):
        errors = []

        # Observation drift check
        try:
            # Cast to expected dtype to reduce false positives
            obs = np.asarray(obs, dtype=self.observation_space.dtype)
            if not self.observation_space.contains(obs):
                errors.append({"type": "drift", "field": "obs", "msg": "observation out of bounds"})
        except Exception:
            errors.append({"type": "drift", "field": "obs", "msg": "invalid observation format"})

        # Action hallucination check
        try:
            if not self.action_space.contains(self._as_action(action)):
                errors.append({"type": "hallucination", "field": "action", "msg": "invalid action"})
        except Exception:
            errors.append({"type": "hallucination", "field": "action", "msg": "invalid action"})

        # Reward sanity check
        if not np.isfinite(reward):
            errors.append({"type": "drift", "field": "reward", "msg": "non-finite reward"})

        return errors


# ---------------- Policy Validator ----------------

class PolicyIntegrityValidator:
    """
    Validator for the agent side (PPO policy).
    Checks:
    - Action probabilities are non-negative and sum to ~1.
    - Value predictions are finite numbers.
    - Chosen action is legal.
    """

    def __init__(self, action_space, strict=False):
        self.action_space = action_space
        self.strict = strict  # strict=True -> tighter tolerance for prob sums

    def _as_action(self, action):
        """Match the action to the shape its space declares. See IntegrityValidator."""
        if hasattr(self.action_space, "nvec"):
            return np.asarray(action, dtype=np.int64).reshape(self.action_space.nvec.shape)
        return int(action)

    def validate(self, action_probs, value, action):
        errors = []

        # Check for negative probabilities
        if (action_probs < 0).any():
            errors.append({"type": "drift", "field": "probs", "msg": "negative action probability"})

        # Check that probs sum to ~1. With one row per drone, every row is its
        # own distribution and each must sum to 1 independently; summing the
        # whole tensor would total the number of drones and hide a broken row.
        sum_tol = 1e-2 if not self.strict else 1e-5
        totals = action_probs.sum(dim=-1) if action_probs.dim() > 1 else action_probs.sum()
        if not torch.isclose(totals, torch.ones_like(totals), atol=sum_tol).all():
            errors.append({"type": "drift", "field": "probs", "msg": "action probabilities do not sum to 1"})

        # Value sanity
        if not torch.isfinite(value).all():
            errors.append({"type": "drift", "field": "value", "msg": "non-finite value estimate"})

        # Action hallucination check
        try:
            if not self.action_space.contains(self._as_action(action)):
                errors.append({"type": "hallucination", "field": "action", "msg": "invalid action index"})
        except Exception:
            errors.append({"type": "hallucination", "field": "action", "msg": "invalid action index"})

        return errors
