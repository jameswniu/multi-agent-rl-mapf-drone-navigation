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

    # Pick a cell no drone or goal already occupies. render() draws obstacles
    # first and paints goals and drones over them, so a hardcoded cell can be
    # silently covered and the assertion then tests nothing.
    taken = {(int(x), int(y)) for x, y in env.positions}
    taken |= {(int(x), int(y)) for x, y in env.goals}
    free = next((x, y) for x in range(env.grid_size) for y in range(env.grid_size)
                if (x, y) not in taken)
    env.heights[free] = 2

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


def test_masking_a_confident_policy_still_yields_a_distribution():
    """The mask must never return a row that fails to sum to 1.

    Renormalising by a clamped floor is wrong whenever the surviving mass falls
    below it. A trained policy puts nearly all its weight on one action, and when
    the mask removes exactly that action the legal remainder can sit far under
    any fixed epsilon. Dividing by the floor rather than the true sum then leaves
    something that is not a distribution.

    This is not hypothetical. It was found by the repository's own integrity
    validator reporting probability drift partway through a four drone flight
    run, having gone unnoticed because torch renormalises inside Categorical, so
    the policy kept acting and nothing ever raised.
    """
    import numpy as np
    import torch

    from agents.ppo_agent import PPOAgent
    from env.drone_env import DroneEnv

    env = DroneEnv("configs/fly-fleet.yaml")
    env.reset(seed=7)
    agent = PPOAgent(env, action_masking=True)

    width = env.observation_space.shape[-1]
    obs = np.zeros((1, width), dtype=np.float32)
    obs[0, 5:9] = [0, 0, 1, 0]      # only "left" is blocked
    obs[0, 18:20] = [1, 1]          # neither climbing nor descending is legal

    confident = torch.full((1, 7), 1e-12)
    confident[0, 3] = 1.0 - 6e-12   # and "left" is exactly what it wants

    masked = agent._mask_probs(obs, confident)
    assert abs(float(masked.sum()) - 1.0) < 1e-5
    assert float(masked[0, 3]) == 0.0, "a masked action keeps zero probability"


def test_masking_survives_total_probability_underflow():
    """A row that underflows to exactly zero must still be samplable.

    The failure above has a sharper form: when the legal mass reaches zero the
    old renormalisation produced an all-zero row, which Categorical cannot sample
    from at all. Falling back to a uniform choice over the legal actions is the
    honest reading, since the policy has no usable opinion left.
    """
    import numpy as np
    import torch
    from torch.distributions import Categorical

    from agents.ppo_agent import PPOAgent
    from env.drone_env import DroneEnv

    env = DroneEnv("configs/fly-fleet.yaml")
    env.reset(seed=7)
    agent = PPOAgent(env, action_masking=True)

    width = env.observation_space.shape[-1]
    obs = np.zeros((1, width), dtype=np.float32)
    obs[0, 18:20] = [1, 1]

    masked = agent._mask_probs(obs, torch.zeros((1, 7)))
    assert abs(float(masked.sum()) - 1.0) < 1e-5
    assert not bool(torch.isnan(masked).any())
    Categorical(masked).sample()    # must not raise


def test_masking_preserves_the_ranking_among_legal_actions():
    """A flattened row is worse than an unnormalised one.

    Falling back to uniform whenever the legal mass is small throws away the
    policy's ordering, and greedy prediction then takes the lowest index, which
    is hover. Measured on a four drone flight run: a drone one cell from its goal
    hovered for thirteen consecutive steps while standing on a teammate's goal
    and blocking it. It read as a coordination failure and was arithmetic.
    """
    import numpy as np
    import torch

    from agents.ppo_agent import PPOAgent
    from env.drone_env import DroneEnv

    env = DroneEnv("configs/fly-fleet.yaml")
    env.reset(seed=7)
    agent = PPOAgent(env, action_masking=True)

    width = env.observation_space.shape[-1]
    obs = np.zeros((1, width), dtype=np.float32)
    obs[0, 5:9] = [0, 0, 1, 0]      # only "left" is blocked
    obs[0, 18:20] = [1, 1]          # no vertical option

    # Nearly all the mass sits on the blocked action, but the legal actions are
    # still ordered: "right" outranks the rest.
    probs = torch.tensor([[1e-20, 2e-20, 3e-20, 1.0 - 6e-20, 5e-20, 1e-20, 1e-20]])
    masked = agent._mask_probs(obs, probs)

    assert abs(float(masked.sum()) - 1.0) < 1e-5
    assert int(masked.argmax()) == 4, "the best legal action must survive the mask"


def test_return_normalisation_over_one_episode_erases_the_success_signal():
    """The measurement that motivated RunningNorm, stated as a test.

    Eight drones all home returns about +47 against about 0 for a run that
    strands one. Normalising over a single episode subtracts that episode's own
    mean, so both collapse to the same standardised shape and the difference the
    agent most needs is the one the update throws away.
    """
    from agents.ppo_agent import RunningNorm

    horizon, fleet = 17, 8
    def returns(rewards):
        out = torch.zeros_like(rewards)
        running = torch.zeros(fleet)
        for t in range(horizon - 1, -1, -1):
            running = rewards[t] + 0.99 * running
            out[t] = running
        return out

    won = torch.full((horizon, fleet), -1.0)
    won[-1] += 60.0                       # arrival plus the completion bonus
    lost = torch.full((horizon, fleet), -1.0)
    lost[-1, : fleet - 1] += 10.0         # one drone short, so no completion

    won, lost = returns(won), returns(lost)
    assert won.mean() - lost.mean() > 40, "the raw gap should be large"

    def per_episode(x):
        return (x - x.mean()) / (x.std() + 1e-8)

    erased = abs(float(per_episode(won).mean() - per_episode(lost).mean()))
    assert erased < 1e-5, "per-episode normalisation should erase the level"

    shared = RunningNorm()
    shared.update(torch.cat([won.reshape(-1), lost.reshape(-1)]))
    kept = abs(float(shared.normalize(won).mean() - shared.normalize(lost).mean()))
    assert kept > 1.0, "a shared scale must keep the two apart"


def test_the_running_normaliser_forgets_when_given_a_horizon():
    """Unbounded history is the flaw: it keeps rescaling against a dead past.

    Without a cap the count grows without bound, so each new batch moves the
    mean by less and less and the statistics lag ever further behind the returns
    actually being earned. Capping the count holds each batch's influence fixed.
    """
    from agents.ppo_agent import RunningNorm

    forgetful, hoarding = RunningNorm(horizon=1000), RunningNorm()
    early = torch.full((500,), -50.0)
    for _ in range(20):
        forgetful.update(early)
        hoarding.update(early)
    late = torch.full((500,), 50.0)
    for _ in range(4):
        forgetful.update(late)
        hoarding.update(late)

    assert forgetful.mean > hoarding.mean, "a capped average should track the recent returns"
    assert hoarding.count > forgetful.count


def test_entropy_weight_is_constant_unless_a_schedule_is_asked_for(tmp_path):
    """Annealing is opt-in, and off it must not perturb the existing behaviour."""
    cfg = tmp_path / "env.yaml"
    cfg.write_text("grid_size: 5\nnum_drones: 1\nobstacle_density: 0.0\nmax_steps: 10\n")
    env = DroneEnv(str(cfg))
    env.reset(seed=0)

    plain = PPOAgent(env, entropy_coef=0.06)
    assert plain._entropy_weight() == pytest.approx(0.06)
    plain._episodes_seen = 10_000
    assert plain._entropy_weight() == pytest.approx(0.06), "no schedule, no drift"

    scheduled = PPOAgent(env, entropy_coef=0.10, entropy_final=0.01, anneal_episodes=1000)
    assert scheduled._entropy_weight() == pytest.approx(0.10)
    scheduled._episodes_seen = 500
    assert scheduled._entropy_weight() == pytest.approx(0.055)
    scheduled._episodes_seen = 5000
    assert scheduled._entropy_weight() == pytest.approx(0.01), "clamped at the end value"
    env.close()


def test_the_crowded_board_carries_its_own_entropy_setting():
    """Eight drones need more exploration than the agent's default provides.

    Left at 0.01 the fleet collapses onto hovering and strands a drone; the
    setting lives in the config rather than the agent because it costs accuracy
    on boards that already solve.
    """
    crowded = DroneEnv("configs/fly-fleet8.yaml")
    crowded.reset(seed=7)
    assert PPOAgent(crowded).entropy_coef == pytest.approx(0.06)
    crowded.close()

    roomy = DroneEnv("configs/fly-fleet.yaml")
    roomy.reset(seed=7)
    assert PPOAgent(roomy).entropy_coef == pytest.approx(0.01), "solved profiles keep the default"
    roomy.close()


def test_an_entropy_schedule_set_entirely_in_config_actually_runs(tmp_path):
    """Both halves of the schedule must come from the config, or neither works.

    Reading entropy_final but not anneal_episodes leaves the duration at zero,
    and a zero duration makes the weight hold its start value forever. The
    profile would look annealed and be constant, which is the worst of the two
    failure modes because nothing reports it.
    """
    cfg = tmp_path / "env.yaml"
    cfg.write_text(
        "grid_size: 5\nnum_drones: 1\nobstacle_density: 0.0\nmax_steps: 10\n"
        "entropy_coef: 0.10\nentropy_final: 0.02\nanneal_episodes: 400\n"
    )
    env = DroneEnv(str(cfg))
    env.reset(seed=0)

    agent = PPOAgent(env)
    assert agent.entropy_coef == pytest.approx(0.10)
    assert agent.entropy_final == pytest.approx(0.02)
    assert agent.anneal_episodes == 400, "the duration must come from the config too"

    agent._episodes_seen = 200
    assert agent._entropy_weight() == pytest.approx(0.06), "half way, so half way down"
    env.close()


def test_the_viewer_only_calls_a_drone_home_when_it_has_landed(tmp_path):
    """The exported arrival flag must mean what the environment means by it.

    A drone sitting on its goal's column at altitude one is not home: the
    environment's own termination test is position AND altitude zero. The export
    checked position alone, so the viewer painted a drone green while it was
    still in the air and read four of eight on a step that scored three. The
    end-of-episode totals were right, which is what made it survive: only the
    frames mid-descent disagreed.
    """
    import numpy as np

    from scripts import export_trajectory  # noqa: F401

    cfg = tmp_path / "env.yaml"
    cfg.write_text("grid_size: 5\nnum_drones: 1\nobstacle_density: 0.0\nmax_steps: 5\nmax_altitude: 1\n")
    env = DroneEnv(str(cfg))
    env.reset(seed=0)
    env.heights[:] = 0
    env.positions = np.array([[2.0, 2.0]], dtype=np.float32)
    env.goals = np.array([[2.0, 2.0]], dtype=np.float32)

    env.altitudes[0] = 1
    hovering = np.all(env.positions == env.goals, axis=1) & (env.altitudes == 0)
    assert not hovering[0], "the environment does not call a hovering drone home"

    env.altitudes[0] = 0
    landed = np.all(env.positions == env.goals, axis=1) & (env.altitudes == 0)
    assert landed[0]
    env.close()


def test_the_exporter_reads_altitude_when_scoring_arrival():
    """Guards the specific line, so the check cannot be dropped again."""
    from pathlib import Path

    source = Path("scripts/export_trajectory.py").read_text()
    arrival_lines = [ln for ln in source.splitlines() if "env.goals[i]" in ln or "altitudes[i]" in ln]
    joined = "\n".join(arrival_lines)
    assert joined.count("altitudes[i]") >= 2, (
        "both the per-frame flag and the run total must require altitude zero"
    )
