"""
Tests for the learning-path fixes.

Three separate defects lived here. A one-step episode produced a single return,
whose unbiased standard deviation is nan, which poisoned every weight in the
network permanently and looked from the outside like a policy that had quietly
stopped learning. The policy loss back-propagated through the value head. And a
sparse goal reward left exploration with nothing to follow.
"""

import numpy as np
import pytest
import torch

from env.drone_env import DroneEnv
from agents.ppo_agent import PPOAgent


def test_single_sample_std_is_nan_which_is_why_the_guard_exists():
    """The property the guard defends against, stated outright."""
    assert torch.isnan(torch.tensor([5.0]).std())


def test_a_one_step_episode_does_not_poison_the_network(tmp_path):
    """
    A drone spawning one move from its goal ends the episode in a single step.
    Before the guard, the resulting nan spread to every weight and never left.
    """
    cfg = tmp_path / "e.yaml"
    cfg.write_text("grid_size: 4\nnum_drones: 1\nobstacle_density: 0.0\nmax_steps: 20\n")
    env = DroneEnv(str(cfg))
    env.reset(seed=0)
    # Place the drone adjacent to its goal so one correct move ends the episode.
    env.goals = np.array([[1.0, 0.0]], dtype=np.float32)
    env.positions = np.array([[0.0, 0.0]], dtype=np.float32)

    agent = PPOAgent(env)
    agent.train(num_episodes=3)

    for name, param in agent.policy.named_parameters():
        assert torch.isfinite(param).all(), f"{name} went non-finite"


def test_non_finite_policy_output_is_reported_as_drift(tmp_path):
    """
    Categorical validates the simplex itself and raises on a nan, which would
    pre-empt the integrity layer. The check runs first so this reads as drift.
    """
    cfg = tmp_path / "e.yaml"
    cfg.write_text("grid_size: 4\nnum_drones: 1\nobstacle_density: 0.0\nmax_steps: 20\n")
    env = DroneEnv(str(cfg))
    obs, _ = env.reset(seed=0)
    agent = PPOAgent(env)

    with torch.no_grad():
        for p in agent.policy.parameters():
            p.fill_(float("nan"))

    with pytest.raises(ValueError, match="non-finite"):
        agent.predict(obs)


def test_shaping_is_off_by_default_and_changes_reward_when_on(tmp_path):
    """A run's reported reward must not silently mean something different."""
    plain = tmp_path / "plain.yaml"
    plain.write_text("grid_size: 6\nnum_drones: 1\nobstacle_density: 0.0\nmax_steps: 20\nfixed_layout: true\n")
    shaped = tmp_path / "shaped.yaml"
    shaped.write_text(
        "grid_size: 6\nnum_drones: 1\nobstacle_density: 0.0\nmax_steps: 20\n"
        "fixed_layout: true\nreward_shaping: true\n"
    )

    a, b = DroneEnv(str(plain)), DroneEnv(str(shaped))
    assert a.reward_shaping is False
    assert b.reward_shaping is True

    a.reset(seed=0)
    b.reset(seed=0)
    _, ra, _, _, _ = a.step([4])
    _, rb, _, _, _ = b.step([4])
    assert ra != rb


def test_fixed_layout_is_identical_across_resets_and_seeds(tmp_path):
    """The demo depends on this: a moving target is not learnable in 200 episodes."""
    cfg = tmp_path / "e.yaml"
    cfg.write_text("grid_size: 6\nnum_drones: 2\nobstacle_density: 0.0\nmax_steps: 20\nfixed_layout: true\n")
    env = DroneEnv(str(cfg))

    env.reset(seed=0)
    first_pos, first_goal = env.positions.copy(), env.goals.copy()
    env.reset(seed=99)
    assert np.array_equal(env.positions, first_pos)
    assert np.array_equal(env.goals, first_goal)
    # Starts and goals must still be distinct from each other.
    assert not np.array_equal(first_pos, first_goal)


def test_render_draws_drones_goals_and_obstacles(tmp_path):
    """metadata advertised a human render mode with nothing behind it."""
    cfg = tmp_path / "e.yaml"
    cfg.write_text("grid_size: 5\nnum_drones: 2\nobstacle_density: 0.0\nmax_steps: 20\nfixed_layout: true\n")
    env = DroneEnv(str(cfg))
    env.reset(seed=0)
    env.obstacles[2, 2] = True

    art = env.render()
    rows = art.splitlines()
    assert len(rows) == env.grid_size
    assert all(len(r.split()) == env.grid_size for r in rows)
    assert "A" in art and "B" in art   # drones
    assert "#" in art                  # the obstacle


def test_demo_runs_end_to_end(capsys):
    """
    A smoke test at tiny scale. The demo is the repository's front door, so a
    crash there is worse than a crash in a library corner nobody runs.
    """
    from main import run_demo

    run_demo(blocks=1, per_block=2, fleet_steps=2)
    out = capsys.readouterr().out

    assert "PART 1" in out and "PART 2" in out
    assert "WHAT THIS DOES NOT SHOW" in out
    assert "conflicting moves refused" in out
    assert "Drones sharing a cell at any point: 0" in out
