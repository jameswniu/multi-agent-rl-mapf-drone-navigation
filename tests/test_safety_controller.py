"""
Safety Controller tests.

Everything else in the integrity layer reports and steps aside. This is the one
component allowed to refuse, so these cover both that it refuses when it should
and, just as importantly, that it stays out of the way when it should not.
"""

import numpy as np
import pytest

from env.drone_env import DroneEnv
from safety_controller import SafetyController


def _env(tmp_path, drones=2, margin=0, separation=0, grid=10):
    cfg = tmp_path / "env.yaml"
    cfg.write_text(
        f"grid_size: {grid}\nnum_drones: {drones}\nobstacle_density: 0.0\n"
        f"max_steps: 50\nsafety:\n  geofence_margin: {margin}\n  min_separation: {separation}\n"
    )
    e = DroneEnv(str(cfg))
    e.reset(seed=0)
    e.obstacles[:] = False
    return e


def test_defaults_are_permissive():
    """Installing the controller must not change behaviour on its own."""
    c = SafetyController(grid_size=20)
    assert c.geofence_margin == 0
    assert c.min_separation == 0


def test_shipped_config_is_wired_in():
    e = DroneEnv()
    assert isinstance(e.safety, SafetyController)
    assert e.safety.geofence_margin == 0
    assert e.safety.min_separation == 0
    e.close()


def test_separation_defaults_to_zero_so_it_cannot_race_the_conflict_rule():
    """
    A separation of 1 would fire on two drones entering one cell, which is
    already the vertex-conflict rule's job. Whichever ran first would decide,
    handing one drone the cell by loop order rather than by any rule.
    """
    c = SafetyController(grid_size=10)
    approved, reasons = c.review([(2, 2), (4, 2)], [(3, 2), (3, 2)])
    assert reasons == {}          # the controller stays out of it
    assert approved == [(3, 2), (3, 2)]  # left for conflict resolution to refuse


def test_geofence_refuses_a_move_into_the_margin(tmp_path):
    e = _env(tmp_path, drones=1, margin=2)
    e.positions = np.array([[2.0, 5.0]], dtype=np.float32)
    e.goals = np.array([[7.0, 7.0]], dtype=np.float32)

    _, _, _, _, info = e.step([3])  # left, into the 2-cell margin

    assert np.array_equal(e.positions, np.array([[2.0, 5.0]], dtype=np.float32))
    assert info["safety_vetoes"] == 1
    assert info["veto_reasons"] == ["geofence"]
    e.close()


def test_geofence_allows_a_move_that_stays_inside(tmp_path):
    e = _env(tmp_path, drones=1, margin=2)
    e.positions = np.array([[5.0, 5.0]], dtype=np.float32)
    e.goals = np.array([[7.0, 7.0]], dtype=np.float32)

    _, _, _, _, info = e.step([3])  # left, well clear of the margin

    assert np.array_equal(e.positions, np.array([[4.0, 5.0]], dtype=np.float32))
    assert "safety_vetoes" not in info
    e.close()


def test_drones_never_spawn_inside_the_geofence(tmp_path):
    e = _env(tmp_path, drones=3, margin=3, grid=12)
    for seed in range(15):
        e.reset(seed=seed)
        for pos in e.positions:
            assert e.safety.in_geofence(int(pos[0]), int(pos[1]))
        for goal in e.goals:
            assert e.safety.in_geofence(int(goal[0]), int(goal[1]))
    e.close()


def test_separation_refuses_both_movers(tmp_path):
    """Symmetry matters: vetoing only the drone checked first is a priority rule."""
    e = _env(tmp_path, drones=2, separation=2)
    e.positions = np.array([[2.0, 2.0], [5.0, 2.0]], dtype=np.float32)
    e.goals = np.array([[0.0, 0.0], [9.0, 9.0]], dtype=np.float32)

    # They close to (3,2) and (4,2), one apart, which breaches a separation of 2.
    _, _, _, _, info = e.step([4, 3])

    assert np.array_equal(e.positions, np.array([[2.0, 2.0], [5.0, 2.0]], dtype=np.float32))
    assert info["safety_vetoes"] == 2
    assert info["veto_reasons"] == ["separation"]
    e.close()


def test_separation_faults_only_the_mover_against_a_stationary_drone(tmp_path):
    """A drone holding its ground is not at fault for someone else approaching."""
    e = _env(tmp_path, drones=2, separation=2)
    e.positions = np.array([[3.0, 2.0], [5.0, 2.0]], dtype=np.float32)
    e.goals = np.array([[0.0, 0.0], [9.0, 9.0]], dtype=np.float32)

    # The mover would land on (4,2), one cell from a drone that never moved.
    _, _, _, _, info = e.step([4, 0])

    assert np.array_equal(e.positions, np.array([[3.0, 2.0], [5.0, 2.0]], dtype=np.float32))
    assert info["safety_vetoes"] == 1
    e.close()


def test_holding_position_is_never_vetoed(tmp_path):
    """
    A drone already inside a forbidden region is left where it is. Moving it
    because its current cell became illegal would be a worse outcome than
    leaving it put, and the controller must never manufacture a move.
    """
    c = SafetyController(grid_size=10, config={"geofence_margin": 3})
    current = [(0, 0)]  # already outside the fence
    approved, reasons = c.review(current, [(0, 0)])
    assert approved == [(0, 0)]
    assert reasons == {}
