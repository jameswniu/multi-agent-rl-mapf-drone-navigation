"""Drone environment for reinforcement learning."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, Dict, Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces

try:  # pragma: no cover - optional dependency
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None

from integrity_validators import IntegrityValidator


class DroneEnv(gym.Env):
    """Simple grid-based drone environment.

    Parameters
    ----------
    config_path: str | Path, optional
        Path to the YAML configuration file. If omitted, uses ``configs/env.yaml``.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, config_path: str | Path = "configs/env.yaml"):
        super().__init__()
        self.config = self._load_config(config_path)
        self.grid_size: int = int(self.config.get("grid_size", 10))
        self.num_drones: int = int(self.config.get("num_drones", 1))
        self.obstacle_density: float = float(self.config.get("obstacle_density", 0.0))
        self.max_steps: int = int(self.config.get("max_steps", 100))

        # Observation: [x, y, goal_x, goal_y, steps_remaining]
        # Bounds are per-dimension: the four coordinates are clamped to the grid,
        # while steps_remaining counts down from max_steps. A single scalar bound
        # would put every steps_remaining > grid_size outside the declared space.
        self.observation_space = spaces.Box(
            low=np.zeros(5, dtype=np.float32),
            high=np.array(
                [
                    self.grid_size - 1,
                    self.grid_size - 1,
                    self.grid_size - 1,
                    self.grid_size - 1,
                    self.max_steps,
                ],
                dtype=np.float32,
            ),
            dtype=np.float32,
        )

        # Actions: 0 hover, 1 up, 2 down, 3 left, 4 right
        self.action_space = spaces.Discrete(5)
        self.action_map = {
            0: "hover",
            1: "up",
            2: "down",
            3: "left",
            4: "right",
        }

        self.validator = IntegrityValidator(self.action_space, self.observation_space)

        self.position = np.zeros(2, dtype=np.float32)
        self.goal = np.array([self.grid_size - 1, self.grid_size - 1], dtype=np.float32)
        self.steps = 0

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------
    def _load_config(self, path: str | Path) -> Dict[str, Any]:
        path = Path(path)
        if not path.is_absolute():
            # Resolve relative to project root (one level above src)
            path = Path(__file__).resolve().parents[2] / path
        with path.open("r") as f:
            if yaml is not None:
                return yaml.safe_load(f) or {}
            # Fallback: basic YAML parser for key: value lines
            data: Dict[str, Any] = {}
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if ":" in line:
                    key, value = line.split(":", 1)
                    data[key.strip()] = float(value) if "." in value else int(value)
            return data

    def _get_obs(self) -> np.ndarray:
        steps_remaining = self.max_steps - self.steps
        return np.array(
            [self.position[0], self.position[1], self.goal[0], self.goal[1], steps_remaining],
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self.position = np.zeros(2, dtype=np.float32)
        self.goal = np.array([self.grid_size - 1, self.grid_size - 1], dtype=np.float32)
        self.steps = 0
        obs = self._get_obs()
        info: Dict[str, Any] = {}
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self.steps += 1

        # Move drone
        if action == 1:  # up
            self.position[1] = min(self.grid_size - 1, self.position[1] + 1)
        elif action == 2:  # down
            self.position[1] = max(0, self.position[1] - 1)
        elif action == 3:  # left
            self.position[0] = max(0, self.position[0] - 1)
        elif action == 4:  # right
            self.position[0] = min(self.grid_size - 1, self.position[0] + 1)
        # action 0 -> hover

        obs = self._get_obs()

        terminated = bool(np.array_equal(self.position, self.goal))
        reward = 10.0 if terminated else -1.0
        truncated = bool(self.steps >= self.max_steps)

        info: Dict[str, Any] = {}
        errors = self.validator.validate(obs, action, reward)
        if errors:
            info["integrity_errors"] = errors

        return obs, reward, terminated, truncated, info

    def close(self):
        """Cleanup resources (none for this simple env)."""
        pass
