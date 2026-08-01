"""
Obstacle tests.

obstacle_density sat in configs/env.yaml being read into an attribute and never
used. These cover what that field now buys: obstacles appear at roughly the
configured rate, they refuse movement, they cost more than an ordinary step,
and a drone can sense one before walking into it.
"""

import numpy as np
import pytest

from env.drone_env import DroneEnv


@pytest.fixture
def env():
    e = DroneEnv()
    try:
        yield e
    finally:
        e.close()


@pytest.fixture
def solo(tmp_path):
    """A one-drone, obstacle-free grid, for testing a single mechanic in isolation."""
    cfg = tmp_path / "env.yaml"
    cfg.write_text("grid_size: 8\nnum_drones: 1\nobstacle_density: 0.0\nmax_steps: 50\n")
    e = DroneEnv(str(cfg))
    e.reset(seed=0)
    try:
        yield e
    finally:
        e.close()


def test_observation_is_one_row_of_twenty_per_drone(env):
    obs, _ = env.reset(seed=0)
    # Five state values, four blocked flags, four peer flags, altitude,
    # four clearance flags, and the two vertical legality flags.
    assert obs.shape == (env.num_drones, 20)
    assert env.observation_space.contains(obs)


def test_no_drone_starts_or_aims_at_an_obstacle(env):
    """Starts and goals are drawn from free cells, so neither can be blocked."""
    for seed in range(20):
        env.reset(seed=seed)
        for pos in env.positions:
            assert not env.obstacles[int(pos[0]), int(pos[1])]
        for goal in env.goals:
            assert not env.obstacles[int(goal[0]), int(goal[1])]


def test_obstacle_rate_tracks_the_configured_density(env):
    rates = []
    for seed in range(30):
        env.reset(seed=seed)
        rates.append(env.obstacles.mean())
    assert env.obstacle_density == pytest.approx(0.1)
    assert np.mean(rates) == pytest.approx(0.1, abs=0.03)


def test_same_seed_gives_the_same_obstacles(env):
    env.reset(seed=7)
    first = env.obstacles.copy()
    env.reset(seed=7)
    assert np.array_equal(first, env.obstacles)


def test_moving_into_an_obstacle_is_refused_and_costs_extra(solo):
    """Position unchanged, the step flagged, and dearer than a plain move."""
    solo.positions = np.array([[2.0, 2.0]], dtype=np.float32)
    solo.goals = np.array([[7.0, 7.0]], dtype=np.float32)
    solo.heights[:] = 0
    solo.heights[2, 3] = 2  # directly above

    obs, reward, terminated, truncated, info = solo.step([1])  # up

    assert np.array_equal(solo.positions, np.array([[2.0, 2.0]], dtype=np.float32))
    assert info.get("collisions") == 1
    assert reward == -2.0
    assert not terminated


def test_a_clear_move_is_not_flagged_and_costs_one(solo):
    solo.positions = np.array([[2.0, 2.0]], dtype=np.float32)
    solo.goals = np.array([[7.0, 7.0]], dtype=np.float32)
    solo.heights[:] = 0

    obs, reward, terminated, truncated, info = solo.step([1])  # up

    assert np.array_equal(solo.positions, np.array([[2.0, 3.0]], dtype=np.float32))
    assert "collisions" not in info
    assert reward == -1.0


def test_sensor_flags_report_obstacles_and_edges(solo):
    """The four trailing values per row are up, down, left, right."""
    solo.positions = np.array([[0.0, 0.0]], dtype=np.float32)
    solo.heights[:] = 0
    solo.heights[0, 1] = 2  # directly above the corner

    up, down, left, right = solo._get_obs()[0][5:9]

    assert up == 1.0      # the obstacle
    assert down == 1.0    # the grid edge reads as impassable too
    assert left == 1.0    # edge
    assert right == 0.0   # clear


def test_zero_density_produces_no_obstacles(solo):
    assert not solo.obstacles.any()
