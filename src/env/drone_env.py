"""Multi-drone environment for reinforcement learning and path finding."""

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
from safety_controller import SafetyController

# Observation values per drone, before the four sensor flags.
_BASE_FEATURES = 5
_SENSOR_FLAGS = 4
_FEATURES_PER_DRONE = _BASE_FEATURES + _SENSOR_FLAGS


class DroneEnv(gym.Env):
    """Grid world holding one or more drones.

    Every drone is described by the same nine features, so the observation is
    ``(num_drones, 9)`` and the action is one discrete move per drone. A single
    drone is simply the degenerate case of the same shape.

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
        self.num_drones: int = max(1, int(self.config.get("num_drones", 1)))
        self.obstacle_density: float = float(self.config.get("obstacle_density", 0.0))
        self.max_steps: int = int(self.config.get("max_steps", 100))
        # Potential-based reward shaping. Off by default because it changes the
        # reward a run reports, and a number that silently means something else
        # is worse than a hard task.
        self.reward_shaping: bool = bool(self.config.get("reward_shaping", False))
        # A fixed layout makes the task stationary: the same starts and goals
        # every episode. Random placement re-poses the problem on every reset,
        # which is a much harder thing to learn and hides whether the learner
        # works at all. Off by default; the demo profile turns it on.
        self.fixed_layout: bool = bool(self.config.get("fixed_layout", False))
        self.shaping_gamma: float = float(self.config.get("shaping_gamma", 0.99))

        # Per drone: [x, y, goal_x, goal_y, steps_remaining,
        #             blocked_up, blocked_down, blocked_left, blocked_right]
        # Bounds are per-dimension. The coordinates are clamped to the grid while
        # steps_remaining counts down from max_steps, so one scalar bound cannot
        # describe both. The trailing flags are local sensing: without them a
        # drone cannot see an obstacle or a neighbour before moving into one.
        row_high = np.array(
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
        )
        self.observation_space = spaces.Box(
            low=np.zeros((self.num_drones, _FEATURES_PER_DRONE), dtype=np.float32),
            high=np.tile(row_high, (self.num_drones, 1)),
            dtype=np.float32,
        )

        # One move per drone: 0 hover, 1 up, 2 down, 3 left, 4 right.
        self.action_space = spaces.MultiDiscrete([5] * self.num_drones)
        self.action_map = {0: "hover", 1: "up", 2: "down", 3: "left", 4: "right"}

        self.validator = IntegrityValidator(self.action_space, self.observation_space)
        # The validators report; this one refuses. It is the only component here
        # permitted to change what happens rather than just describe it.
        self.safety = SafetyController(self.grid_size, self.config.get("safety"))

        self.positions = np.zeros((self.num_drones, 2), dtype=np.float32)
        self.goals = np.zeros((self.num_drones, 2), dtype=np.float32)
        self.steps = 0
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
        """Draw a fresh obstacle grid from the seeded RNG."""
        grid = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        if self.obstacle_density <= 0:
            return grid
        return self.np_random.random((self.grid_size, self.grid_size)) < self.obstacle_density

    def _free_cells(self) -> np.ndarray:
        """Every cell not holding an obstacle, as an (n, 2) array."""
        xs, ys = np.where(~self.obstacles)
        return np.stack([xs, ys], axis=1)

    def _place(self) -> None:
        """Choose distinct start and goal cells for every drone.

        Starts must be distinct from each other, or two drones would begin
        stacked in one cell. Goals must be distinct too, or drones would be
        rewarded for crowding a single square.
        """
        free = self._free_cells()
        # Never spawn a drone somewhere the geofence would immediately trap it.
        free = np.array([c for c in free if self.safety.in_geofence(int(c[0]), int(c[1]))])

        if self.fixed_layout:
            # Deterministic and reproducible: the first free cells become starts,
            # the last become goals, so drones begin far from where they finish.
            if len(free) < 2 * self.num_drones:
                raise ValueError("not enough free cells for a fixed layout")
            order = np.lexsort((free[:, 1], free[:, 0]))
            ordered = free[order]
            self.positions = ordered[: self.num_drones].astype(np.float32)
            self.goals = ordered[-self.num_drones :][::-1].astype(np.float32)
            return
        needed = 2 * self.num_drones
        if len(free) < needed:
            raise ValueError(
                f"grid has {len(free)} free cells but {self.num_drones} drones need {needed}; "
                "lower obstacle_density or raise grid_size"
            )
        chosen = self.np_random.choice(len(free), size=needed, replace=False)
        picks = free[chosen]
        self.positions = picks[: self.num_drones].astype(np.float32)
        self.goals = picks[self.num_drones :].astype(np.float32)

    def _occupied(self) -> set:
        return {(int(p[0]), int(p[1])) for p in self.positions}

    def _is_obstacle(self, x: int, y: int) -> bool:
        if not (0 <= x < self.grid_size and 0 <= y < self.grid_size):
            return False
        return bool(self.obstacles[x, y])

    def _is_impassable(self, x: int, y: int, occupied: set) -> bool:
        """What a drone's sensor reports: wall, obstacle and neighbour all read alike."""
        if not (0 <= x < self.grid_size and 0 <= y < self.grid_size):
            return True
        if self.obstacles[x, y]:
            return True
        return (x, y) in occupied

    def _sensor_flags(self, index: int, occupied: set) -> list:
        """Blocked flags for one drone, in action order: up, down, left, right."""
        x, y = int(self.positions[index][0]), int(self.positions[index][1])
        others = occupied - {(x, y)}
        neighbours = [(x, y + 1), (x, y - 1), (x - 1, y), (x + 1, y)]
        return [1.0 if self._is_impassable(nx, ny, others) else 0.0 for nx, ny in neighbours]

    def _get_obs(self) -> np.ndarray:
        steps_remaining = self.max_steps - self.steps
        occupied = self._occupied()
        rows = []
        for i in range(self.num_drones):
            rows.append(
                [
                    self.positions[i][0],
                    self.positions[i][1],
                    self.goals[i][0],
                    self.goals[i][1],
                    steps_remaining,
                    *self._sensor_flags(i, occupied),
                ]
            )
        return np.array(rows, dtype=np.float32)

    def _potential(self, positions) -> np.ndarray:
        """Negative Manhattan distance to each drone's goal.

        Used as the shaping potential. Closer to the goal is a higher potential,
        so moving toward it earns a small positive shaping term.
        """
        return -np.abs(positions - self.goals).sum(axis=1)

    def _target(self, index: int, action: int) -> Tuple[float, float]:
        """Where one action would take one drone, clamped at the grid edge."""
        x, y = float(self.positions[index][0]), float(self.positions[index][1])
        if action == 1:
            y = min(self.grid_size - 1, y + 1)
        elif action == 2:
            y = max(0, y - 1)
        elif action == 3:
            x = max(0, x - 1)
        elif action == 4:
            x = min(self.grid_size - 1, x + 1)
        return x, y

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self.obstacles = self._generate_obstacles()
        self._place()
        self.steps = 0
        return self._get_obs(), {}

    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Advance every drone by one move, resolving conflicts by refusal.

        Three kinds of conflict are refused rather than resolved by priority,
        because a priority rule would quietly teach the policy that some drones
        always win: two drones claiming one cell, two drones swapping cells, and
        a drone moving into a cell whose occupant is staying put.
        """
        self.steps += 1
        actions = np.atleast_1d(np.asarray(action)).astype(int).ravel()
        if actions.size != self.num_drones:
            raise ValueError(f"expected {self.num_drones} actions, got {actions.size}")

        current = [(int(p[0]), int(p[1])) for p in self.positions]
        intended = []
        collided = [False] * self.num_drones

        # Pass 1: an obstacle refuses the move outright.
        for i in range(self.num_drones):
            tx, ty = self._target(i, int(actions[i]))
            if self._is_obstacle(int(tx), int(ty)):
                intended.append(current[i])
                collided[i] = True
            else:
                intended.append((int(tx), int(ty)))

        # The Safety Controller arbitrates before drones are compared with each
        # other, so a move it refuses never reaches conflict resolution at all.
        intended, veto_reasons = self.safety.review(current, intended)
        for idx in veto_reasons:
            collided[idx] = True

        # Pass 2: drone against drone, repeated until nothing else gives way.
        for _ in range(self.num_drones + 1):
            changed = False
            for i in range(self.num_drones):
                if intended[i] == current[i]:
                    continue
                for j in range(self.num_drones):
                    if i == j:
                        continue
                    # A drone that is staying put covers the stationary case too:
                    # its intended cell is its current cell, so "same cell" catches
                    # a mover walking into it.
                    j_is_moving = intended[j] != current[j]
                    same_cell = intended[i] == intended[j]
                    swapping = intended[i] == current[j] and intended[j] == current[i]
                    if same_cell or swapping:
                        intended[i] = current[i]
                        collided[i] = True
                        # Only fault the other drone if it was also trying to move.
                        # Penalising one for holding its ground would teach the
                        # policy that hovering is risky, which is the opposite of
                        # what a yielding manoeuvre should cost.
                        if j_is_moving:
                            intended[j] = current[j]
                            collided[j] = True
                        changed = True
                        break
            if not changed:
                break

        self.positions = np.array(intended, dtype=np.float32)

        at_goal = np.all(self.positions == self.goals, axis=1)
        rewards = np.where(at_goal, 10.0, np.where(collided, -2.0, -1.0))

        if self.reward_shaping:
            # F = gamma * phi(s') - phi(s). Ng, Harada and Russell (1999) show
            # this leaves the optimal policy unchanged, so it guides exploration
            # without redefining what a good route is. A sparse +10 at the goal
            # is almost never stumbled upon on a large grid, which is why an
            # unshaped run looks like it is not learning at all.
            before = self._potential(np.array(current, dtype=np.float32))
            after = self._potential(self.positions)
            rewards = rewards + self.shaping_gamma * after - before

        reward = float(rewards.sum())

        terminated = bool(at_goal.all())
        truncated = bool(self.steps >= self.max_steps)

        obs = self._get_obs()
        info: Dict[str, Any] = {"at_goal": int(at_goal.sum())}
        if any(collided):
            info["collisions"] = int(sum(collided))
        if veto_reasons:
            info["safety_vetoes"] = len(veto_reasons)
            info["veto_reasons"] = sorted(set(veto_reasons.values()))

        errors = self.validator.validate(obs, actions, reward)
        if errors:
            info["integrity_errors"] = errors

        return obs, reward, terminated, truncated, info

    def render(self):
        """Draw the grid as text.

        ``metadata`` has always advertised a human render mode without providing
        one. Letters are drones, digits are their goals, and ``#`` is an
        obstacle. Row order is flipped so y increases upward, which is what the
        coordinates say and not what list order gives you.
        """
        grid = [["." for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                if self.obstacles[x, y]:
                    grid[y][x] = "#"
        for i, (gx, gy) in enumerate(self.goals.astype(int)):
            grid[gy][gx] = str(i % 10)
        for i, (x, y) in enumerate(self.positions.astype(int)):
            grid[y][x] = chr(ord("A") + i % 26)
        return "\n".join(" ".join(row) for row in reversed(grid))

    def close(self):
        """Cleanup resources (none for this simple env)."""
        pass
