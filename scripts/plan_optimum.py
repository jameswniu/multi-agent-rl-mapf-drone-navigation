"""What is the best a central planner can find on these boards?

Two jobs. It answers whether a board can be solved at all, which is worth
knowing before blaming the learner: the eight drone course sat unsolved for a
long time under theories about congestion and packing, and this says the board
is clearable in 17 of the 60 steps allowed, so none of those theories could have
been the whole story. And it gives the number the learned policy is measured
against, which is why claims elsewhere quote a step count and a planned makespan
side by side rather than a step count alone. That number is a bound, not an
optimum, and should be described as one: see below.

Prioritised planning with space-time A*: drones are routed one at a time and each
treats the already-routed ones as moving obstacles. It is incomplete, so a
failure here is not a proof that a board is impossible; a success is a proof that
it is possible. Orderings are reshuffled across restarts because one ordering can
fail on an instance another ordering solves.

Reservations ignore altitude while the environment does not, so a drone may pass
over a parked one in the simulator but not in this planner. That makes the
planner strictly more conservative and its makespan an upper bound on the true
optimum, which is the safe direction for a benchmark to err in.
"""

import heapq
import random
import sys

sys.path.insert(0, "src")

from env.drone_env import DroneEnv  # noqa: E402

MOVES = [(0, 0, 0), (0, -1, 0), (0, 1, 0), (-1, 0, 0), (1, 0, 0), (0, 0, 1), (0, 0, -1)]


def plan(env, start, goal, reserved, horizon):
    """Space-time A* for one drone around cells already reserved by others.

    ``reserved`` holds vertex reservations keyed by (time, x, y) and edge
    reservations keyed by (time, from, to) so a swap is caught as well as a
    head-on claim on the same cell.
    """
    sx, sy = start
    gx, gy = goal

    def heuristic(x, y, z):
        return abs(x - gx) + abs(y - gy) + z

    open_set = [(heuristic(sx, sy, 0), 0, (sx, sy, 0), None)]
    seen = {}
    while open_set:
        _, time, node, parent = heapq.heappop(open_set)
        if node in seen.get(time, ()):
            continue
        seen.setdefault(time, {})[node] = parent
        x, y, z = node
        if (x, y, z) == (gx, gy, 0):
            path, cursor, at = [], node, time
            while at >= 0:
                path.append(cursor)
                cursor = seen[at][cursor]
                at -= 1
                if cursor is None:
                    break
            return path[::-1]
        if time >= horizon:
            continue
        for dx, dy, dz in MOVES:
            nx, ny, nz = x + dx, y + dy, z + dz
            if not (0 <= nx < env.grid_size and 0 <= ny < env.grid_size):
                continue
            if not (0 <= nz <= env.max_altitude):
                continue
            if env._is_impassable(nx, ny, set(), nz):
                continue
            if (time + 1, nx, ny) in reserved:
                continue
            if (time + 1, (nx, ny), (x, y)) in reserved:
                continue
            step = time + 1
            if (nx, ny, nz) in seen.get(step, ()):
                continue
            heapq.heappush(open_set, (step + heuristic(nx, ny, nz), step, (nx, ny, nz), node))
    return None


def solve(env, horizon, order):
    reserved = set()
    paths = {}
    for i in order:
        start = tuple(int(v) for v in env.positions[i])
        goal = tuple(int(v) for v in env.goals[i])
        path = plan(env, start, goal, reserved, horizon)
        if path is None:
            return None
        for time, (x, y, _) in enumerate(path):
            reserved.add((time, x, y))
            if time:
                px, py, _ = path[time - 1]
                reserved.add((time, (px, py), (x, y)))
        # A parked drone stays put and keeps blocking its goal cell.
        x, y, _ = path[-1]
        for time in range(len(path), horizon + 1):
            reserved.add((time, x, y))
        paths[i] = path
    return paths


for cfg in ("configs/fly-fleet.yaml", "configs/fly-fleet8.yaml"):
    env = DroneEnv(cfg)
    env.reset(seed=7)
    fleet = list(range(env.num_drones))
    best = None
    rng = random.Random(0)
    for attempt in range(400):
        order = fleet if attempt == 0 else rng.sample(fleet, len(fleet))
        paths = solve(env, env.max_steps, order)
        if paths:
            span = max(len(p) for p in paths.values()) - 1
            if best is None or span < best:
                best = span
    label = f"{env.num_drones} drones"
    if best is None:
        print(f"{label:<12} NO joint plan found in {env.max_steps} steps (400 orderings)", flush=True)
    else:
        print(f"{label:<12} solvable, best makespan {best} of {env.max_steps} allowed", flush=True)
    env.close()
