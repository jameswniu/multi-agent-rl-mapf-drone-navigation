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
import torch

from env.drone_env import DroneEnv


@pytest.fixture
def pair(tmp_path):
    """Two drones on a clear grid, positioned by hand per test."""
    cfg = tmp_path / "env.yaml"
    cfg.write_text("grid_size: 8\nnum_drones: 2\nobstacle_density: 0.0\nmax_steps: 50\n")
    e = DroneEnv(str(cfg))
    e.reset(seed=0)
    e.heights[:] = 0
    try:
        yield e
    finally:
        e.close()


def test_fleet_size_comes_from_config():
    e = DroneEnv()
    assert e.num_drones == 10
    obs, _ = e.reset(seed=0)
    assert obs.shape == (10, 20)
    # Seven: hover, the four planar moves, climb and descend. The vertical
    # pair exists whatever max_altitude is, so the space has one shape and
    # they are simply masked out on a board with no third dimension.
    assert list(e.action_space.nvec) == [7] * 10
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


def test_drones_see_each_other_as_peers_not_as_walls(pair):
    """A neighbour reports in the peer flags, never in the blocked flags.

    The distinction is load-bearing rather than cosmetic. Blocked flags drive the
    action mask, so anything reported there is a move the policy cannot take at
    all. When a neighbour read as blocked, two drones facing each other each had
    their only useful move removed and neither could yield, and a drone parked on
    its goal became a permanent wall for everyone else.
    """
    pair.positions = np.array([[2.0, 2.0], [2.0, 3.0]], dtype=np.float32)
    obs = pair._get_obs()

    assert obs[0][9] == 1.0, "drone 0 should see a peer above it"
    assert obs[1][10] == 1.0, "drone 1 should see a peer below it"

    assert obs[0][5] == 0.0, "a neighbour is not a wall and must not be masked"
    assert obs[1][6] == 0.0, "a neighbour is not a wall and must not be masked"


def test_a_peer_never_removes_a_move_from_the_policy(pair):
    """The action mask must leave a peer-occupied direction available.

    Regression test for a deadlock the mask itself created. Masking is only sound
    for facts that are permanent and knowable alone; an occupied cell is neither,
    because the occupant may move next step.
    """
    from agents.ppo_agent import PPOAgent

    pair.positions = np.array([[2.0, 2.0], [2.0, 3.0]], dtype=np.float32)
    agent = PPOAgent(pair, action_masking=True)

    obs = pair._get_obs()
    probs = torch.ones(pair.num_drones, 7) / 7.0
    masked = agent._mask_probs(obs, probs)

    # Action 1 is "up", the direction drone 1 currently occupies.
    assert masked[0][1] > 0.0, "moving toward a peer must stay possible"


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

    # Drone 0 already collected its arrival bonus on the previous step, so only
    # drone 1 is newly home. Both then receive the completion bonus.
    expected = pair.arrival_bonus + 2 * pair.completion_bonus
    assert reward == expected


def test_finishing_beats_stranding_a_drone(pair):
    """Completing the task must pay more than deliberately not completing it.

    This is a regression test for a real defect rather than a hypothetical. A
    drone on its goal used to earn a bonus on every step, while the episode ends
    only once every drone is home, so finishing switched that income off. On the
    four-drone profile bringing three drones home scored +500 and bringing all
    four home scored -40, which made stranding a drone the optimal policy. The
    agent was not failing to learn that; it was learning it correctly.
    """
    pair.positions = np.array([[2.0, 1.0], [7.0, 7.0]], dtype=np.float32)
    pair.goals = np.array([[2.0, 1.0], [7.0, 7.0]], dtype=np.float32)

    _, solved, terminated, _, _ = pair.step([0, 0])
    assert terminated

    # Same board, but drone 1 held one cell short of its goal.
    pair.reset(seed=0)
    pair.heights[:] = 0
    pair.positions = np.array([[2.0, 1.0], [6.0, 7.0]], dtype=np.float32)
    pair.goals = np.array([[2.0, 1.0], [7.0, 7.0]], dtype=np.float32)

    _, stranded, terminated, _, _ = pair.step([0, 0])
    assert not terminated

    assert solved > stranded, (
        f"completing the task scored {solved} against {stranded} for leaving a "
        "drone behind, so the reward prefers partial success"
    )


def test_sitting_on_a_goal_pays_once(pair):
    """A parked drone must not draw an income stream.

    Per-step goal pay is what made stranding profitable, so the fix is only real
    if a drone that stays put stops earning.
    """
    pair.positions = np.array([[2.0, 1.0], [5.0, 5.0]], dtype=np.float32)
    pair.goals = np.array([[2.0, 1.0], [7.0, 7.0]], dtype=np.float32)

    _, first, _, _, _ = pair.step([0, 0])
    _, second, _, _, _ = pair.step([0, 0])
    _, third, _, _, _ = pair.step([0, 0])

    assert first > second, "the arrival bonus should be paid on arrival"
    assert second == third, "a parked drone must earn the same nothing every step"
