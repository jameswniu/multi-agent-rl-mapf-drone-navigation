"""Multi-drone environment for reinforcement learning and path finding."""

from __future__ import annotations

from pathlib import Path
from collections import deque
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

# Observation values per drone.
#
# The two flag groups are deliberately separate. A wall is permanent and knowable
# alone, so it is safe to remove that move from the policy's choices outright. A
# neighbouring drone is neither: it may move away next step, and masking against
# it manufactures deadlock, since two drones facing each other would each have
# their only useful move removed and neither could ever yield. Worse, a drone
# parked on its goal would become a permanent wall for every other drone. So peer
# occupancy is reported as information the policy can weigh, never as a
# constraint it cannot violate.
_BASE_FEATURES = 5
_STATIC_FLAGS = 4       # impassable at my current altitude: masked
_PEER_FLAGS = 4         # another drone right now: observed only
_ALTITUDE_FEATURES = 1  # my own altitude
_CLEAR_FLAGS = 4        # blocked now, but passable if I climb: observed only
_VERTICAL_FLAGS = 2     # climb and descend legality: masked
_FEATURES_PER_DRONE = (
    _BASE_FEATURES + _STATIC_FLAGS + _PEER_FLAGS
    + _ALTITUDE_FEATURES + _CLEAR_FLAGS + _VERTICAL_FLAGS
)

# Obstacle heights. A drone occupies a cell only when its altitude is at least
# the cell's height, so height is literally how high you must fly to pass.
_CLEAR = 0    # open ground
_LOW = 1      # low enough to fly over
_TALL = 2     # taller than any drone can climb

# Cells also have a ceiling: the highest altitude allowed inside them. Almost
# everywhere that is the drone's own limit and means nothing. A gap cut through a
# solid wall is the exception, and it has to be, because a cell that is merely
# clear is passable from any altitude: an early course put an opening in a wall
# the drone could not climb, and the drone simply climbed once at the start and
# flew through the opening without ever coming down. A ceiling of zero is what
# actually makes a route require the ground.
_TUNNEL = 0


class DroneEnv(gym.Env):
    """Grid world holding one or more drones.

    Every drone is described by the same thirteen features, so the observation
    is ``(num_drones, 13)`` and the action is one discrete move per drone. A
    single drone is simply the degenerate case of the same shape.

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
        # Dense peer term, g(coop_quality). The completion bonus pays for
        # cooperation as an event; this pays for its quality every step, which
        # is the second axis of the reward-density matrix in the README. Off by
        # default so the shipped reward is unchanged.
        self.peer_shaping: bool = bool(self.config.get("peer_shaping", False))
        self.peer_weight: float = float(self.config.get("peer_weight", 0.5))

        # Reward terms. Named and configurable because their relative sizes are
        # the whole design: the completion bonus has to outweigh what a policy
        # could earn by stranding a drone, or stranding one becomes optimal.
        self.step_penalty: float = float(self.config.get("step_penalty", 1.0))
        self.collision_penalty: float = float(self.config.get("collision_penalty", 2.0))
        self.arrival_bonus: float = float(self.config.get("arrival_bonus", 10.0))
        # Flying has to cost something. Climbing once and staying up would
        # otherwise make every low obstacle free, and the choice between going
        # over and going around would stop being a choice.
        self.altitude_penalty: float = float(self.config.get("altitude_penalty", 0.5))
        self.completion_bonus: float = float(self.config.get("completion_bonus", 50.0))
        # A fixed layout makes the task stationary: the same starts and goals
        # every episode. Random placement re-poses the problem on every reset,
        # which is a much harder thing to learn and hides whether the learner
        # works at all. Off by default; the demo profile turns it on.
        self.fixed_layout: bool = bool(self.config.get("fixed_layout", False))
        self.shaping_gamma: float = float(self.config.get("shaping_gamma", 0.99))
        # Which distance the shaping potential measures. "manhattan" ignores
        # terrain completely, which is fine on an open board and actively
        # misleading once walls exist: it points a drone straight at a barrier
        # and never credits the climb that gets it across. "terrain" measures
        # the real walking distance, treating anything the drone could fly over
        # as passable. That is a relaxation of the task, not a solution to it:
        # it says which way the goal is, and still leaves the policy to work out
        # when climbing is worth paying for.
        self.shaping_potential: str = str(self.config.get("shaping_potential", "manhattan"))
        # How high a drone may climb, and what share of obstacles are low enough
        # to clear. Both default to the flat world, so every existing config
        # keeps its exact behaviour: no climbing, and every obstacle solid.
        self.max_altitude: int = int(self.config.get("max_altitude", 0))
        self.low_obstacle_ratio: float = float(self.config.get("low_obstacle_ratio", 0.0))
        # Terrain layout. Scattered obstacles almost never force a choice about
        # altitude: measured on one such board the trained drone reached the goal
        # at the true optimum having never once climbed, because a clear ground
        # route existed. A ridge puts a barrier between start and goal on purpose,
        # so going over and going around are the only two options.
        self.terrain: str = str(self.config.get("terrain", "scatter"))
        self.ridges: int = max(1, int(self.config.get("ridges", 1)))
        # Openings per solid wall. One is a hard serialisation point: every drone
        # must cross that single cell, so N drones cost at least N steps of pure
        # queueing however well they coordinate. Fine for a small fleet, and the
        # thing that jams a large one.
        self.tunnels_per_wall: int = max(1, int(self.config.get("tunnels_per_wall", 1)))

        # Per drone: [x, y, goal_x, goal_y, steps_remaining,
        #             blocked_up, blocked_down, blocked_left, blocked_right,
        #             peer_up, peer_down, peer_left, peer_right]
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
                1.0, 1.0, 1.0, 1.0,          # blocked_up/down/left/right
                1.0, 1.0, 1.0, 1.0,          # peer_up/down/left/right
                max(1.0, float(self.max_altitude)),   # own altitude
                1.0, 1.0, 1.0, 1.0,          # clearable_up/down/left/right
                1.0, 1.0,                    # blocked_climb, blocked_descend
            ],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=np.zeros((self.num_drones, _FEATURES_PER_DRONE), dtype=np.float32),
            high=np.tile(row_high, (self.num_drones, 1)),
            dtype=np.float32,
        )

        # One move per drone. The vertical pair exists whatever max_altitude is,
        # so the action space has a single shape; when climbing is disabled they
        # are simply always masked out. A conditional space would leak into the
        # network shape, saved weights and every test.
        self.action_space = spaces.MultiDiscrete([7] * self.num_drones)
        self.action_map = {
            0: "hover", 1: "up", 2: "down", 3: "left", 4: "right",
            5: "climb", 6: "descend",
        }

        self.validator = IntegrityValidator(self.action_space, self.observation_space)
        # The validators report; this one refuses. It is the only component here
        # permitted to change what happens rather than just describe it.
        self.safety = SafetyController(self.grid_size, self.config.get("safety"))

        self.positions = np.zeros((self.num_drones, 2), dtype=np.float32)
        self.altitudes = np.zeros(self.num_drones, dtype=np.int32)
        # Which drones have ever reached their goal this episode, so the arrival
        # bonus pays once rather than every step the drone sits there.
        self._reached = np.zeros(self.num_drones, dtype=bool)
        self._distance_cache: Dict[int, Any] = {}
        self.goals = np.zeros((self.num_drones, 2), dtype=np.float32)
        self.steps = 0
        self.heights = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        self.ceilings = np.full((self.grid_size, self.grid_size), self.max_altitude, dtype=np.int32)

    @property
    def obstacles(self) -> np.ndarray:
        """Any cell that is not open ground, as the old boolean grid.

        Kept because placement and most callers only ever asked "is something
        here", and because a config that knows nothing about heights should
        behave exactly as it always did.

        Returned read-only on purpose. A derived array cannot be written through:
        ``env.obstacles[2, 2] = True`` would build a temporary, set a bit on it
        and discard it, changing nothing and reporting nothing. Marking it
        unwritable turns that into an immediate error instead of a test that
        quietly stops testing what it names. Write to ``heights`` instead, which
        is the real terrain and can say how tall an obstacle is.
        """
        derived = self.heights > _CLEAR
        derived.flags.writeable = False
        return derived

    @obstacles.setter
    def obstacles(self, value) -> None:
        # Assigning booleans yields the flat world: every obstacle solid.
        self.heights = (np.asarray(value).astype(bool)).astype(np.int32) * _TALL

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

    def _generate_ceilings(self) -> np.ndarray:
        """Default headroom everywhere; only a course cuts tunnels into it."""
        return np.full((self.grid_size, self.grid_size), self.max_altitude, dtype=np.int32)

    def _generate_heights(self) -> np.ndarray:
        """Draw a fresh obstacle-height grid from the seeded RNG.

        With ``low_obstacle_ratio`` at zero every obstacle is solid, which is the
        flat world the earlier configs describe. Above zero, that share of them
        become low enough to fly over, and the drone has a real choice between
        climbing and detouring.
        """
        heights = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        if self.terrain == "ridge":
            return self._generate_ridge(heights)
        if self.terrain == "course":
            return self._generate_course(heights)
        if self.obstacle_density <= 0:
            return heights
        occupied = self.np_random.random((self.grid_size, self.grid_size)) < self.obstacle_density
        low = self.np_random.random((self.grid_size, self.grid_size)) < self.low_obstacle_ratio
        heights[occupied] = _TALL
        heights[occupied & low] = _LOW
        return heights

    def _ridge_columns(self) -> list:
        """Evenly spaced barrier columns, with clear ground on both outsides."""
        n = self.ridges
        return [int(round(self.grid_size * (k + 1) / (n + 1))) for k in range(n)]

    def _generate_ridge(self, heights: np.ndarray) -> np.ndarray:
        """Barriers across the board, mostly low, partly solid.

        Each wall spans the full height, so a drone travelling from the low-x
        side to the high-x side has to cross every one of them and cannot reach
        its goal without deciding how.

        Solid segments stop climbing from collapsing into a single reflex. Over a
        low span flying is cheap; at a solid one the only answer is to go around,
        so the drone has to read the terrain rather than learn one habit.

        With more than one wall the interesting question moves to the gap between
        them: hold altitude across it, or drop to the ground and climb again. That
        is a genuine trade rather than a flourish, and which way it goes is set by
        altitude_penalty against the width of the gap. See _describe_economics.
        """
        for mid in self._ridge_columns():
            for y in range(self.grid_size):
                solid = self.np_random.random() >= self.low_obstacle_ratio
                heights[mid, y] = _TALL if solid else _LOW
            # One guaranteed low cell per wall, so the board stays crossable by
            # climbing even when every random draw came up solid.
            heights[mid, int(self.np_random.integers(0, self.grid_size))] = _LOW
        return heights

    def _generate_course(self, heights: np.ndarray) -> np.ndarray:
        """Alternating barriers: low walls to fly over, tall walls to walk around.

        Two crossings of one kind is not a course, it is the same move twice, and
        on a board of only low walls the drone has no reason to come back down
        between them. Holding altitude across a gap of g columns costs
        (g + 1)(1 + p) against 3(1 + p) + g for dropping and re-climbing, so
        descending only pays when g > (2 + 2p) / p. Buying that with a large
        altitude_penalty was measured and does not work: at a penalty high enough
        to matter, crossing is worth about -9.5 in shaped reward against +2 of
        progress, the agent correctly concludes crossing is bad, and five seeds
        never solve it.

        A solid wall settles it without touching the economics. It is taller than
        any drone can climb, so the only way past is the one gap in it, and the
        drone has to be on the ground to use it. Descending stops being a trade
        and becomes a requirement, which is what makes the course alternate:
        climb over the low wall, come down for the solid one, walk to its gap,
        then climb again.
        """
        for index, mid in enumerate(self._ridge_columns()):
            if index % 2 == 0:
                # Low the whole way across: no way around, so it must be flown.
                heights[mid, :] = _LOW
            else:
                # Solid the whole way across but for one opening: no way over,
                # so it must be walked around.
                heights[mid, :] = _TALL
                # Spread the openings evenly rather than placing them at random.
                # A gap at the board edge forces a detour the full height of the
                # grid twice over, which turned a 10 wide course into a 31 action
                # route and put it out of reach. Evenly spaced also means the
                # queues that form are the same size.
                for k in range(self.tunnels_per_wall):
                    gap = int(round(self.grid_size * (k + 1) / (self.tunnels_per_wall + 1)))
                    gap = min(self.grid_size - 1, max(0, gap))
                    heights[mid, gap] = _CLEAR
                    # An opening is a tunnel, not a doorway in the sky. Without
                    # the ceiling the drone climbs once at the start and flies
                    # through still airborne, and the wall stops asking anything.
                    self.ceilings[mid, gap] = _TUNNEL
        return heights

    def describe_economics(self) -> str:
        """Whether dropping between two walls is cheaper than staying airborne.

        Crossing a gap of ``g`` clear columns and stepping onto the next wall
        costs ``(g + 1)(1 + p)`` if altitude is held. Dropping costs one airborne
        step onto the first gap cell, a descent, ``g - 1`` ground steps, a climb,
        and an airborne step onto the wall: ``3(1 + p) + g``. Descending wins
        exactly when ``g > (2 + 2p) / p``.

        The three ``(1 + p)`` terms are the part that is easy to get wrong. An
        earlier version of this counted only two and reported that dropping was
        optimal on a board where the shortest path plainly held altitude, so the
        helper contradicted the search that scored the runs.
        """
        p = self.altitude_penalty
        if p <= 0:
            return "altitude is free, so the drone should climb once and stay up"
        cols = self._ridge_columns()
        if len(cols) < 2:
            return "one wall, so there is no gap to decide about"
        gap = cols[1] - cols[0] - 1
        need = (2 + 2 * p) / p
        verdict = "drop and re-climb" if gap > need else "stay airborne"
        return (f"gap {gap} columns, altitude_penalty {p}, threshold {need:.1f} "
                f"-> optimal is to {verdict}")

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

        if self.terrain in ("ridge", "course"):
            # Straddle every barrier explicitly. The generic split sorts free cells
            # and halves them, which has no idea a wall exists: on an eight wide
            # board it put the goal one column short of the ridge, on the same
            # side as the start, and the drone reached it without ever meeting
            # the terrain the profile was built to pose. A course leaving this
            # out is worse still: the goal landed between the first and second
            # walls, so the run never reached the solid wall it exists to show.
            cols = self._ridge_columns()
            left = np.array([c for c in free if c[0] < cols[0]])
            right = np.array([c for c in free if c[0] > cols[-1]])
            if len(left) < self.num_drones or len(right) < self.num_drones:
                raise ValueError("ridge terrain needs room on both sides of the wall")
            lo = np.lexsort((left[:, 1], left[:, 0]))
            ro = np.lexsort((right[:, 1], right[:, 0]))
            starts = left[lo][np.linspace(0, len(left) - 1, self.num_drones).astype(int)]
            goals = right[ro][np.linspace(0, len(right) - 1, self.num_drones).astype(int)]
            self.positions = starts.astype(np.float32)
            self.goals = goals.astype(np.float32)
            return

        if self.fixed_layout:
            # Deterministic and reproducible: the first free cells become starts,
            # the last become goals, so drones begin far from where they finish.
            if len(free) < 2 * self.num_drones:
                raise ValueError("not enough free cells for a fixed layout")
            order = np.lexsort((free[:, 1], free[:, 0]))
            ordered = free[order]
            half = len(ordered) // 2
            # Spread evenly through each half rather than taking a contiguous
            # block. Consecutive cells in sorted order form a single column, so
            # taking the first and last N stacked every drone and every goal on
            # top of each other, which is both a degenerate task and unreadable
            # when drawn.
            start_idx = np.linspace(0, half - 1, self.num_drones).astype(int)
            goal_idx = np.linspace(half, len(ordered) - 1, self.num_drones).astype(int)
            self.positions = ordered[start_idx].astype(np.float32)
            self.goals = ordered[goal_idx][::-1].astype(np.float32)
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
        # Altitude is part of the key. Two drones sharing a column at different
        # heights are not in conflict, which is both physically true and the
        # thing that lets one drone pass over another that is stuck.
        return {(int(p[0]), int(p[1]), int(a))
                for p, a in zip(self.positions, self.altitudes)}

    def _is_obstacle(self, x: int, y: int, altitude: int = 0) -> bool:
        """Whether the cell blocks a drone flying at this altitude.

        Blocked from below by its height and from above by its ceiling. Both are
        needed: height alone lets a drone fly over everything a course puts in
        front of it, including the gaps meant to be walked through.
        """
        if not (0 <= x < self.grid_size and 0 <= y < self.grid_size):
            return False
        return bool(self.heights[x, y] > altitude or altitude > self.ceilings[x, y])

    def _is_impassable(self, x: int, y: int, occupied: set, altitude: int = 0) -> bool:
        """Whether a cell cannot be entered from this altitude, this turn.

        Off the grid, or standing taller than the drone is currently flying. It
        is still safe to mask on: both facts are knowable alone and true for the
        whole turn. A cell that is only blocked because the drone is low can be
        reached by climbing first, and that shows up in the clearance flags
        rather than being silently permitted here.

        A cell holding another drone is NOT impassable. It is occupied right now
        and may be free next step, and treating the two alike is what turned the
        action mask into a deadlock generator.
        """
        if not (0 <= x < self.grid_size and 0 <= y < self.grid_size):
            return True
        return bool(self.heights[x, y] > altitude or altitude > self.ceilings[x, y])

    def _sensor_flags(self, index: int, occupied: set) -> list:
        """Fifteen flags for one drone; each group runs up, down, left, right.

        Which of them are allowed to drive the action mask is the whole point of
        splitting them. Masking is sound only for a fact that is permanent for
        the turn and knowable by this drone alone.

        * ``blocked``   impassable from the current altitude, so masked.
        * ``peer``      a neighbour is there this instant. Never masked; it may
                        move, and masking it manufactures deadlock.
        * ``clearable`` blocked now but reachable after climbing. Never masked,
                        because the drone can choose to climb; this is the flag
                        that makes "over or around" a decision rather than a wall.
        * ``vertical``  whether climbing and descending are legal at all, which
                        is knowable and permanent, so masked.
        """
        x, y = int(self.positions[index][0]), int(self.positions[index][1])
        alt = int(self.altitudes[index])
        others = occupied - {(x, y, alt)}
        neighbours = [(x, y + 1), (x, y - 1), (x - 1, y), (x + 1, y)]

        blocked = [1.0 if self._is_impassable(nx, ny, others, alt) else 0.0
                   for nx, ny in neighbours]
        peers = [1.0 if (nx, ny, alt) in others else 0.0 for nx, ny in neighbours]
        clearable = [
            1.0 if (self._in_bounds(nx, ny)
                    and alt < self.heights[nx, ny] <= self.max_altitude) else 0.0
            for nx, ny in neighbours
        ]
        # Climbing is illegal at the ceiling as well as at the drone's own limit,
        # so a drone standing in a tunnel cannot rise inside it.
        blocked_climb = 1.0 if (alt >= self.max_altitude or alt + 1 > self.ceilings[x, y]) else 0.0
        # Never descend into something you are flying over.
        blocked_descend = 1.0 if (alt <= 0 or self.heights[x, y] > alt - 1) else 0.0

        return blocked + peers + [float(alt)] + clearable + [blocked_climb, blocked_descend]

    def _in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.grid_size and 0 <= y < self.grid_size

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

    def _coop_quality(self) -> np.ndarray:
        """g(coop_quality): per-drone clearance from peers, in [0, 1].

        One graded number per drone per step, measuring how much room it is
        leaving its neighbours. A drone with all four adjacent cells free
        scores 1.0; one boxed in by peers scores 0.0. This is deliberately a
        quality signal rather than an event: the completion bonus already pays
        for cooperation happening, and the point of the second axis is to pay
        for how well it is happening while it happens.

        Crowding is measured on plan coordinates. Two drones at the same
        location but different altitudes are still competing for the same
        ground track, which is what a conflict at a choke point looks like.
        """
        n = len(self.positions)
        if n < 2:
            return np.ones(n, dtype=np.float32)

        out = np.zeros(n, dtype=np.float32)
        occupied = {(float(p[0]), float(p[1])) for p in self.positions}
        for i, pos in enumerate(self.positions):
            x, y = float(pos[0]), float(pos[1])
            others = occupied - {(x, y)}
            free = sum(
                1.0
                for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))
                if (nx, ny) not in others
            )
            out[i] = free / 4.0
        return out

    def _potential(self, positions) -> np.ndarray:
        """Negative distance to each drone's goal, higher being closer.

        Ng, Harada and Russell (1999) show any potential leaves the optimal
        policy unchanged, so the choice here is purely about how much of a hint
        the shaping carries.
        """
        if self.shaping_potential != "terrain":
            return -np.abs(positions - self.goals).sum(axis=1)

        out = np.zeros(len(positions), dtype=np.float32)
        for i, pos in enumerate(positions):
            field = self._distance_field(i)
            x, y = int(pos[0]), int(pos[1])
            out[i] = -field.get((x, y), self.grid_size * 2)
        return out

    def _distance_field(self, index: int):
        """Steps from every cell to drone ``index``'s goal, cached per episode.

        Anything the drone could fly over counts as passable, so the field knows
        a low wall is crossable without saying how much that costs or when to
        bother. Only genuinely solid terrain blocks it. Computed once per goal
        because a fixed layout never moves, and the walls never move either.
        """
        cached = self._distance_cache.get(index)
        if cached is not None:
            return cached

        gx, gy = int(self.goals[index][0]), int(self.goals[index][1])
        field = {(gx, gy): 0}
        queue = deque([(gx, gy)])
        while queue:
            x, y = queue.popleft()
            for nx, ny in ((x, y + 1), (x, y - 1), (x - 1, y), (x + 1, y)):
                if not self._in_bounds(nx, ny) or (nx, ny) in field:
                    continue
                if self.heights[nx, ny] > self.max_altitude:
                    continue
                field[(nx, ny)] = field[(x, y)] + 1
                queue.append((nx, ny))

        self._distance_cache[index] = field
        return field

    def _target(self, index: int, action: int) -> Tuple[float, float]:
        """Where one action would take one drone, clamped at the grid edge.

        Climbing and descending hold position: a turn spent changing altitude is
        a turn not spent covering ground, and that is exactly the cost that makes
        flying over an obstacle a trade rather than a free pass.
        """
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

    def _target_altitude(self, index: int, action: int) -> int:
        """Altitude one action would leave a drone at, refusing illegal changes."""
        alt = int(self.altitudes[index])
        if action == 5:
            x, y = int(self.positions[index][0]), int(self.positions[index][1])
            if alt + 1 > self.ceilings[x, y]:
                return alt          # no headroom here
            return min(self.max_altitude, alt + 1)
        if action == 6:
            x, y = int(self.positions[index][0]), int(self.positions[index][1])
            below = alt - 1
            # Descending into the obstacle you are flying over is not a move.
            if below < 0 or self.heights[x, y] > below:
                return alt
            return below
        return alt

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self.ceilings = self._generate_ceilings()
        self.heights = self._generate_heights()   # may cut tunnels into ceilings
        self._place()
        self.steps = 0
        self._reached = np.zeros(self.num_drones, dtype=bool)
        self.altitudes = np.zeros(self.num_drones, dtype=np.int32)
        self._distance_cache = {}   # goals and terrain are redrawn on reset
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

        # Cells carry altitude. Two drones sharing a column at different heights
        # are not in conflict, so the whole comparison below has to be in three
        # dimensions or a drone could never pass over one that is stuck.
        current = [(int(p[0]), int(p[1]), int(a))
                   for p, a in zip(self.positions, self.altitudes)]
        intended = []
        collided = [False] * self.num_drones

        # Pass 1: terrain refuses the move outright, judged at the altitude the
        # drone would arrive at rather than the one it left.
        for i in range(self.num_drones):
            act = int(actions[i])
            tx, ty = self._target(i, act)
            talt = self._target_altitude(i, act)
            if self._is_obstacle(int(tx), int(ty), talt):
                intended.append(current[i])
                collided[i] = True
            else:
                intended.append((int(tx), int(ty), talt))

        # The Safety Controller reasons in the plane, so it is handed plan-view
        # cells and its refusals are mapped back onto the full move. A refused
        # drone holds its altitude too; being sent up or down by a rule that
        # never considered height would be a decision nobody made.
        plan_current = [(c[0], c[1]) for c in current]
        plan_intended = [(t[0], t[1]) for t in intended]
        reviewed, veto_reasons = self.safety.review(plan_current, plan_intended)
        for idx, cell in enumerate(reviewed):
            if cell != plan_intended[idx]:
                intended[idx] = current[idx]
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

        self.positions = np.array([(t[0], t[1]) for t in intended], dtype=np.float32)
        self.altitudes = np.array([t[2] for t in intended], dtype=np.int32)

        # Arriving means landing on the goal, not hovering above it. Without the
        # altitude term a drone could park in the air over its target and count
        # as home, which is neither what a delivery is nor what the picture shows.
        at_goal = np.all(self.positions == self.goals, axis=1) & (self.altitudes == 0)

        # A drone standing on its goal earns nothing further. Paying it every
        # step made camping an income stream, and because the episode ends only
        # when everyone arrives, finishing the task switched that income off.
        # Measured on the four-drone profile: bringing three drones home scored
        # +500 while bringing all four home scored -40, so the optimal policy
        # under the old reward was to strand one drone deliberately. The agent
        # was not failing to learn that; it was learning it correctly.
        rewards = np.where(collided, -self.collision_penalty, -self.step_penalty)
        # Height costs fuel. Without this a drone climbs once, treats every low
        # obstacle as absent for the rest of the episode, and the choice between
        # going over and going around stops existing.
        rewards = rewards - self.altitude_penalty * self.altitudes
        rewards = np.where(at_goal, 0.0, rewards)

        # Arrival pays once, on the step a drone first reaches its goal. Leaving
        # and returning does not pay again, which would be a second income loop.
        newly_home = at_goal & ~self._reached
        rewards = rewards + newly_home * self.arrival_bonus
        self._reached = self._reached | at_goal

        if self.reward_shaping:
            # F = gamma * phi(s') - phi(s). Ng, Harada and Russell (1999) show
            # this leaves the optimal policy unchanged, so it guides exploration
            # without redefining what a good route is. A sparse +10 at the goal
            # is almost never stumbled upon on a large grid, which is why an
            # unshaped run looks like it is not learning at all.
            # Plan coordinates only. The potential measures ground distance to
            # the goal, and cells now carry altitude as a third element.
            before = self._potential(
                np.array([(c[0], c[1]) for c in current], dtype=np.float32)
            )
            after = self._potential(self.positions)
            rewards = rewards + self.shaping_gamma * after - before

        if self.peer_shaping:
            # Paid every step, to every drone that is not already home. A drone
            # sitting on its goal is excluded so that clearance cannot become a
            # second camping income, which is the exact failure the arrival
            # bonus had to be repaired for.
            rewards = rewards + np.where(
                at_goal, 0.0, self.peer_weight * self._coop_quality()
            )

        terminated = bool(at_goal.all())

        # The completion bonus is what makes solving the task beat solving most
        # of it. It is paid to every drone, because arriving is a team outcome:
        # a drone that yields a corridor so another can pass has contributed to
        # it, and a purely per-drone bonus would not price that contribution.
        if terminated:
            rewards = rewards + self.completion_bonus

        reward = float(rewards.sum())

        # Per-drone rewards alongside the scalar. The scalar is what the gym API
        # requires, but training on it alone gives every drone the same signal:
        # one that flew a clean route and one that drove into a wall are told the
        # same thing. Credit assignment needs the vector.
        per_drone = [float(r) for r in rewards]
        truncated = bool(self.steps >= self.max_steps)

        obs = self._get_obs()
        info: Dict[str, Any] = {
            "at_goal": int(at_goal.sum()),
            "rewards": per_drone,
            "altitudes": [int(a) for a in self.altitudes],
        }
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
        one. Letters are drones, digits are their goals, ``o`` is an obstacle low
        enough to fly over and ``#`` is one that is not. Row order is flipped so
        y increases upward, which is what the coordinates say and not what list
        order gives you.
        """
        grid = [["." for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                height = int(self.heights[x, y])
                if height > _CLEAR:
                    grid[y][x] = "o" if height <= self.max_altitude else "#"
                elif self.ceilings[x, y] < self.max_altitude:
                    grid[y][x] = "n"      # tunnel: passable only on the ground
        for i, (gx, gy) in enumerate(self.goals.astype(int)):
            grid[gy][gx] = str(i % 10)
        for i, (x, y) in enumerate(self.positions.astype(int)):
            # Airborne drones render lowercase, so altitude is visible in text.
            letter = chr(ord("A") + i % 26)
            grid[y][x] = letter.lower() if self.altitudes[i] > 0 else letter
        return "\n".join(" ".join(row) for row in reversed(grid))

    def close(self):
        """Cleanup resources (none for this simple env)."""
        pass
