"""Export real rollouts to JSON for the browser viewer in docs/sim.html.

The viewer draws whatever this writes and invents nothing, so every position and
every number on screen came out of an actual episode.

Two scenarios are exported, because the honest claim about this repository is a
split one: the small profile learns and the multi-drone profile does not. Showing
only one of them would be a choice about which half to hide.

Two kinds of evidence per scenario, and they answer different questions.

A learning curve, averaged over several seeds. One seed is not a result here. An
early single-seed export showed the multi-drone profile scoring better untrained
than after two thousand episodes, which was network initialisation luck rather
than anything about learning. A band across seeds shows the trend and shows how
wide the spread is, which is the part a single line hides.

Trajectories at four checkpoints, from one named seed, because positions cannot
be averaged. A mean of two routes is not a route anybody flew.

Evaluation is greedy throughout. Training reward is noisy while the policy is
still exploring, and the question a reader is actually asking is what the policy
would do if you asked it right now.

Usage:
    python scripts/export_trajectory.py --out docs/trajectory.json
"""

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agents.ppo_agent import PPOAgent  # noqa: E402
from env.drone_env import DroneEnv  # noqa: E402

SCENARIOS = [
    {
        "key": "solo",
        "name": "One drone, 5x5",
        "config": "configs/demo.yaml",
        "episodes": 2000,
        "blurb": "Solved by every seed, on the optimal four-move route.",
    },
    {
        "key": "fleet",
        "name": "Four drones, 8x8",
        "config": "configs/sim.yaml",
        # Twelve thousand, not eight. Widening the observation for altitude made
        # this profile slower to learn without making it harder: eight thousand
        # solved three seeds of five, twelve thousand solves all five and lands
        # on the same fourteen step schedule as before.
        "episodes": 12000,
        "blurb": "Solved by every seed, on the shortest schedule that exists.",
    },
    {
        "key": "fly",
        "name": "One drone, obstacle course",
        "config": "configs/fly.yaml",
        "episodes": 4000,
        "blurb": "Low walls to fly over, a solid one to walk through.",
    },
    {
        "key": "fly-fleet",
        "name": "Four drones, obstacle course",
        "config": "configs/fly-fleet.yaml",
        "episodes": 30000,
        "blurb": "Four rows, one tunnel, so the routes converge and have to queue.",
    },
]


def rollout(env, agent, seed, record=True):
    """Play one greedy episode.

    With ``record`` false only the summary is kept. Carrying every frame for
    every sampled point on every seed would inflate the payload by an order of
    magnitude for data that nothing draws.
    """
    state, _ = env.reset(seed=seed)
    frames = []
    total = 0.0
    refused = 0

    for _ in range(env.max_steps):
        action = agent.predict(state)
        state, reward, terminated, truncated, info = env.step(action)
        total += float(reward)
        refused += int(info.get("collisions", 0))

        if record:
            frames.append(
                {
                    "positions": [[int(p[0]), int(p[1])] for p in env.positions],
                    "altitudes": [int(a) for a in env.altitudes],
                    "atGoal": [
                        bool(np.array_equal(env.positions[i], env.goals[i]))
                        for i in range(env.num_drones)
                    ],
                    "refused": int(info.get("collisions", 0)),
                    "reward": round(float(reward), 3),
                    "cumulative": round(total, 3),
                }
            )
        if terminated or truncated:
            break

    arrived = sum(
        bool(np.array_equal(env.positions[i], env.goals[i])) for i in range(env.num_drones)
    )
    out = {"totalReward": round(total, 2), "arrived": int(arrived), "refusedTotal": refused}
    if record:
        out["frames"] = frames
    return out


def one_seed(spec, root, env_seed, torch_seed, sample_at, checkpoints, record):
    """Train a single agent, evaluating greedily at every sampled point."""
    torch.manual_seed(torch_seed)
    env = DroneEnv(str(root / spec["config"]))
    env.reset(seed=env_seed)

    # Read the board now. After a rollout env.positions holds where the drones
    # ended up, which is emphatically not where they started.
    layout = {
        "gridSize": env.grid_size,
        "numDrones": env.num_drones,
        "maxSteps": env.max_steps,
        # Height, not just presence. A low obstacle is one a drone can fly over,
        # and drawing the two alike would hide the entire decision.
        "obstacles": [
            [int(x), int(y), int(env.heights[x, y])]
            for x, y in zip(*np.where(env.heights > 0))
        ],
        "maxAltitude": int(env.max_altitude),
        # Cells a drone may only occupy on the ground. They are not obstacles and
        # would otherwise draw as bare floor, which hides the reason the route
        # comes back down at all.
        "tunnels": [
            [int(x), int(y)]
            for x, y in zip(*np.where(env.ceilings < env.max_altitude))
        ],
        "goals": [[int(g[0]), int(g[1])] for g in env.goals],
        "starts": [[int(p[0]), int(p[1])] for p in env.positions],
    }

    agent = PPOAgent(env, action_masking=True)
    points, runs, trained = [], [], 0

    for target in sample_at:
        if target > trained:
            with contextlib.redirect_stdout(io.StringIO()):
                agent.train(num_episodes=target - trained)
            trained = target

        want = record and target in checkpoints
        run = rollout(env, agent, env_seed, record=want)
        points.append((target, run["totalReward"], run["arrived"]))
        if want:
            runs.append(
                {
                    "label": "Untrained" if target == 0 else f"{target} episodes",
                    "episodes": target,
                    **run,
                }
            )

    return layout, points, runs


def build(spec, root, env_seed, seeds):
    total = spec["episodes"]
    stride = max(1, total // 40)
    checkpoints = [0, total // 4, total // 2, total]
    sample_at = sorted(set(list(range(0, total + 1, stride)) + checkpoints))

    print(f"\n{spec['name']}  ({total} episodes, {len(seeds)} seeds, sampling every {stride})")

    per_seed, per_runs, layout = [], [], None
    for ts in seeds:
        lay, points, r = one_seed(spec, root, env_seed, ts, sample_at, checkpoints, record=True)
        per_seed.append(points)
        per_runs.append(r)
        layout = layout or lay
        print(f"  seed {ts}: {points[0][1]:>9.2f} to {points[-1][1]:>9.2f}")

    # Replay the median seed by final reward, not seed 0. An arbitrary pick can
    # land on the worst run, which then contradicts the averaged headline beside
    # it and reads as a bug rather than as variance.
    finals = [pts[-1][1] for pts in per_seed]
    order = sorted(range(len(finals)), key=lambda k: finals[k])
    pick = order[len(order) // 2]
    runs = per_runs[pick]
    print(f"  replaying seed {seeds[pick]} (median of {len(seeds)}, final {finals[pick]:.2f})")

    # Band across seeds. The spread is the honest part; a lone mean line would
    # imply a confidence the data does not support.
    curve = []
    for idx, target in enumerate(sample_at):
        rewards = [s[idx][1] for s in per_seed]
        arrived = [s[idx][2] for s in per_seed]
        curve.append(
            {
                "episodes": target,
                "mean": round(float(np.mean(rewards)), 2),
                "lo": round(float(np.min(rewards)), 2),
                "hi": round(float(np.max(rewards)), 2),
                "arrived": round(float(np.mean(arrived)), 2),
            }
        )

    print(f"  mean {curve[0]['mean']:.2f} to {curve[-1]['mean']:.2f}"
          f"   arrived {curve[0]['arrived']:.2f} to {curve[-1]['arrived']:.2f}"
          f" of {layout['numDrones']}")

    return {
        "key": spec["key"],
        "name": spec["name"],
        "blurb": spec["blurb"],
        "config": spec["config"],
        "episodes": total,
        "seeds": len(seeds),
        "replaySeed": seeds[pick],
        **layout,
        "curve": curve,
        "runs": runs,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7, help="environment layout seed")
    parser.add_argument("--seeds", type=int, default=5, help="how many network seeds to average")
    parser.add_argument("--out", default="docs/trajectory.json")
    args = parser.parse_args(argv)

    root = Path(__file__).resolve().parents[1]
    seeds = list(range(args.seeds))
    payload = {
        "envSeed": args.seed,
        "scenarios": [build(s, root, args.seed, seeds) for s in SCENARIOS],
    }

    out = root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, separators=(",", ":")))
    print(f"\nwrote {out} ({out.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
