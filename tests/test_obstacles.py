"""
Obstacle tests.

obstacle_density sat in configs/env.yaml being read into an attribute and never
used. These cover the behaviour that field now buys: obstacles are generated at
roughly the configured rate, they refuse movement, they cost more than an
ordinary step, and the drone can sense them before walking into them.
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


def test_observation_is_nine_dimensional_and_in_space(env):
    obs, _ = env.reset(seed=0)
    assert obs.shape == (9,)
    assert env.observation_space.contains(obs)


def test_start_and_goal_are_never_blocked(env):
    """A blocked start is meaningless and a blocked goal is unwinnable."""
    for seed in range(25):
        env.reset(seed=seed)
        assert not env.obstacles[0, 0]
        assert not env.obstacles[env.grid_size - 1, env.grid_size - 1]


def test_obstacle_rate_tracks_the_configured_density(env):
    """Averaged over seeds, the draw should sit near obstacle_density."""
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


def test_moving_into_an_obstacle_is_refused_and_costs_extra(env):
    """Position is unchanged, the step is flagged, and it costs more than a plain move."""
    env.reset(seed=0)
    # Plant an obstacle directly above the drone, then try to move up.
    env.obstacles[:] = False
    env.obstacles[0, 1] = True
    before = env.position.copy()

    obs, reward, terminated, truncated, info = env.step(1)  # up

    assert np.array_equal(env.position, before)
    assert info.get("collision") is True
    assert reward == -2.0
    assert not terminated


def test_a_clear_move_is_not_flagged_and_costs_one(env):
    env.reset(seed=0)
    env.obstacles[:] = False
    before = env.position.copy()

    obs, reward, terminated, truncated, info = env.step(1)  # up

    assert not np.array_equal(env.position, before)
    assert "collision" not in info
    assert reward == -1.0


def test_sensor_flags_report_adjacent_obstacles(env):
    """The four trailing observation values are up, down, left, right."""
    env.reset(seed=0)
    env.obstacles[:] = False
    env.obstacles[0, 1] = True  # directly above the start at (0, 0)

    obs = env._get_obs()
    up, down, left, right = obs[5], obs[6], obs[7], obs[8]

    assert up == 1.0        # the obstacle
    assert down == 1.0      # the grid edge reads as impassable too
    assert left == 1.0      # edge
    assert right == 0.0     # clear


def test_zero_density_produces_no_obstacles(tmp_path):
    """A density of zero must leave the grid completely clear."""
    config = tmp_path / "env.yaml"
    config.write_text("grid_size: 10\nnum_drones: 1\nobstacle_density: 0.0\nmax_steps: 50\n")
    e = DroneEnv(str(config))
    e.reset(seed=3)
    assert not e.obstacles.any()
    e.close()
