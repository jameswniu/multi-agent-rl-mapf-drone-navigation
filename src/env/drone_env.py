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

        # Observation: [x, y, goal_x, goal_y, steps_remaining,
        #               blocked_up, blocked_down, blocked_left, blocked_right]
        # Bounds are per-dimension: the coordinates are clamped to the grid, while
        # steps_remaining counts down from max_steps. A single scalar bound would
        # put every steps_remaining > grid_size outside the declared space.
        # The four trailing flags are local sensing. Without them a drone cannot
        # see an obstacle before hitting it, so avoidance is not learnable and the
        # obstacles would only ever be a tax on a blind policy.
        self.observation_space = spaces.Box(
            low=np.zeros(9, dtype=np.float32),
            high=np.array(
                [
                    self.grid_size - 1,
                    self.grid_size - 1,
                    self.grid_size - 1,
                    self.grid_size - 1,
                    self.max_steps,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
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
        # Populated on reset, once the seeded RNG exists.
        self.obstacles = np.zeros((self.grid_size, self.grid_size), dtype=bool)

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

    def _generate_obstacles(self) -> np.ndarray:
        """Draw a fresh obstacle grid from the seeded RNG.

        The start and goal cells are always cleared. A blocked start would make
        the episode meaningless, and a blocked goal would make it unwinnable, so
        neither is left to chance.
        """
        grid = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        if self.obstacle_density <= 0:
            return grid

        grid = self.np_random.random((self.grid_size, self.grid_size)) < self.obstacle_density
        grid[0, 0] = False
        grid[self.grid_size - 1, self.grid_size - 1] = False
        return grid

    def _is_obstacle(self, x: int, y: int) -> bool:
        """True when the cell holds an obstacle. Out of bounds is not an obstacle."""
        if not (0 <= x < self.grid_size and 0 <= y < self.grid_size):
            return False
        return bool(self.obstacles[x, y])

    def _is_impassable(self, x: int, y: int) -> bool:
        """What the drone's local sensor reports: a wall reads the same as an obstacle."""
        if not (0 <= x < self.grid_size and 0 <= y < self.grid_size):
            return True
        return bool(self.obstacles[x, y])

    def _sensor_flags(self) -> list:
        """Blocked flags for the four moves, in action order: up, down, left, right."""
        x, y = int(self.position[0]), int(self.position[1])
        neighbours = [(x, y + 1), (x, y - 1), (x - 1, y), (x + 1, y)]
        return [1.0 if self._is_impassable(nx, ny) else 0.0 for nx, ny in neighbours]

    def _get_obs(self) -> np.ndarray:
        steps_remaining = self.max_steps - self.steps
        return np.array(
            [
                self.position[0],
                self.position[1],
                self.goal[0],
                self.goal[1],
                steps_remaining,
                *self._sensor_flags(),
            ],
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self.obstacles = self._generate_obstacles()
        self.position = np.zeros(2, dtype=np.float32)
        self.goal = np.array([self.grid_size - 1, self.grid_size - 1], dtype=np.float32)
        self.steps = 0
        obs = self._get_obs()
        info: Dict[str, Any] = {}
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self.steps += 1

        # Where the action would take the drone. Edges clamp, so a move into a
        # wall is absorbed rather than rejected.
        x, y = float(self.position[0]), float(self.position[1])
        if action == 1:  # up
            y = min(self.grid_size - 1, y + 1)
        elif action == 2:  # down
            y = max(0, y - 1)
        elif action == 3:  # left
            x = max(0, x - 1)
        elif action == 4:  # right
            x = min(self.grid_size - 1, x + 1)
        # action 0 -> hover

        # An obstacle refuses the move outright: the drone stays where it was.
        collided = self._is_obstacle(int(x), int(y))
        if not collided:
            self.position[0], self.position[1] = x, y

        obs = self._get_obs()

        terminated = bool(np.array_equal(self.position, self.goal))
        if terminated:
            reward = 10.0
        else:
            # A collision costs more than a wasted step, otherwise there is no
            # gradient telling the policy to route around anything.
            reward = -2.0 if collided else -1.0
        truncated = bool(self.steps >= self.max_steps)

        info: Dict[str, Any] = {}
        if collided:
            info["collision"] = True
        errors = self.validator.validate(obs, action, reward)
        if errors:
            info["integrity_errors"] = errors

        return obs, reward, terminated, truncated, info

    def close(self):
        """Cleanup resources (none for this simple env)."""
        pass
