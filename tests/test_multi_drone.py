"""
Multi-drone and conflict tests.

num_drones sat in configs/env.yaml being read and never used; the environment
tracked a single position vector. These cover the fleet contract and the three
conflicts a path-finding step has to refuse.

Conflicts are resolved by refusing both movers rather than by priority. A
priority rule would quietly teach the policy that some drones always win, which
is a coordination bug disguised as a tie-break.
"""

import numpy as np
import pytest

from env.drone_env import DroneEnv


@pytest.fixture
def pair(tmp_path):
    """Two drones on a clear grid, positioned by hand per test."""
    cfg = tmp_path / "env.yaml"
    cfg.write_text("grid_size: 8\nnum_drones: 2\nobstacle_density: 0.0\nmax_steps: 50\n")
    e = DroneEnv(str(cfg))
    e.reset(seed=0)
    e.obstacles[:] = False
    try:
        yield e
    finally:
        e.close()


def test_fleet_size_comes_from_config():
    e = DroneEnv()
    assert e.num_drones == 10
    obs, _ = e.reset(seed=0)
    assert obs.shape == (10, 9)
    assert list(e.action_space.nvec) == [5] * 10
    e.close()


def test_every_drone_starts_and_aims_somewhere_distinct():
    e = DroneEnv()
    e.reset(seed=3)
    starts = {tuple(p) for p in e.positions.tolist()}
    goals = {tuple(g) for g in e.goals.tolist()}
    assert len(starts) == e.num_drones
    assert len(goals) == e.num_drones
    e.close()


def test_wrong_number_of_actions_is_refused(pair):
    with pytest.raises(ValueError):
        pair.step([1])


def test_vertex_conflict_refuses_both(pair):
    """Two drones claiming one cell: neither is allowed to take it."""
    pair.positions = np.array([[2.0, 2.0], [4.0, 2.0]], dtype=np.float32)
    pair.goals = np.array([[0.0, 0.0], [7.0, 7.0]], dtype=np.float32)

    _, reward, _, _, info = pair.step([4, 3])  # right, left -> both want (3, 2)

    assert np.array_equal(pair.positions, np.array([[2.0, 2.0], [4.0, 2.0]], dtype=np.float32))
    assert info["collisions"] == 2
    assert reward == -4.0


def test_swap_conflict_refuses_both(pair):
    """Adjacent drones cannot trade places; they would pass through each other."""
    pair.positions = np.array([[2.0, 2.0], [3.0, 2.0]], dtype=np.float32)
    pair.goals = np.array([[0.0, 0.0], [7.0, 7.0]], dtype=np.float32)

    _, _, _, _, info = pair.step([4, 3])  # right, left

    assert np.array_equal(pair.positions, np.array([[2.0, 2.0], [3.0, 2.0]], dtype=np.float32))
    assert info["collisions"] == 2


def test_moving_into_a_stationary_drone_is_refused(pair):
    """Only the mover is penalised; the drone that held its ground is not."""
    pair.positions = np.array([[2.0, 2.0], [3.0, 2.0]], dtype=np.float32)
    pair.goals = np.array([[0.0, 0.0], [7.0, 7.0]], dtype=np.float32)

    _, _, _, _, info = pair.step([4, 0])  # right into a hovering neighbour

    assert np.array_equal(pair.positions, np.array([[2.0, 2.0], [3.0, 2.0]], dtype=np.float32))
    assert info["collisions"] == 1


def test_drones_see_each_other(pair):
    """A neighbour reads as blocked, the same as a wall or an obstacle."""
    pair.positions = np.array([[2.0, 2.0], [2.0, 3.0]], dtype=np.float32)
    obs = pair._get_obs()
    assert obs[0][5] == 1.0   # drone 0 looking up at drone 1
    assert obs[1][6] == 1.0   # drone 1 looking down at drone 0


def test_independent_moves_both_succeed(pair):
    pair.positions = np.array([[1.0, 1.0], [5.0, 5.0]], dtype=np.float32)
    pair.goals = np.array([[0.0, 0.0], [7.0, 7.0]], dtype=np.float32)

    _, reward, _, _, info = pair.step([4, 4])  # both right, far apart

    assert np.array_equal(pair.positions, np.array([[2.0, 1.0], [6.0, 5.0]], dtype=np.float32))
    assert "collisions" not in info
    assert reward == -2.0  # one per drone


def test_episode_ends_only_when_every_drone_is_home(pair):
    pair.positions = np.array([[1.0, 1.0], [5.0, 5.0]], dtype=np.float32)
    pair.goals = np.array([[2.0, 1.0], [7.0, 7.0]], dtype=np.float32)

    _, _, terminated, _, info = pair.step([4, 0])  # only drone 0 arrives
    assert info["at_goal"] == 1
    assert not terminated

    pair.positions = np.array([[2.0, 1.0], [7.0, 7.0]], dtype=np.float32)
    _, reward, terminated, _, info = pair.step([0, 0])
    assert info["at_goal"] == 2
    assert terminated
    assert reward == 20.0  # +10 each
