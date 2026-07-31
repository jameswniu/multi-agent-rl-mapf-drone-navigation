"""Export real rollouts to JSON for the browser viewer in docs/sim.html.

The viewer draws whatever this writes and invents nothing, so every position,
refusal and veto on screen came out of an actual episode. Two rollouts are
recorded against the same fixed layout: one from an untrained policy and one
after training. Holding the task still is what makes the comparison mean
anything, since otherwise the second run could simply have drawn an easier map.

Usage:
    python scripts/export_trajectory.py --episodes 600 --out docs/trajectory.json
"""

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agents.ppo_agent import PPOAgent  # noqa: E402
from env.drone_env import DroneEnv  # noqa: E402


def rollout(env, agent, seed):
    """Play one greedy episode and record every step.

    Greedy rather than sampled on purpose. Training reward is noisy because the
    policy is still exploring; what a viewer wants to see is what the policy
    would actually do if asked, which is the argmax.
    """
    state, _ = env.reset(seed=seed)
    frames = []
    total = 0.0

    for _ in range(env.max_steps):
        action = agent.predict(state)
        state, reward, terminated, truncated, info = env.step(action)
        total += float(reward)

        at_goal = [
            bool(np.array_equal(env.positions[i], env.goals[i]))
            for i in range(env.num_drones)
        ]
        frames.append(
            {
                "positions": [[int(p[0]), int(p[1])] for p in env.positions],
                "atGoal": at_goal,
                "refused": int(info.get("collisions", 0)),
                "vetoed": int(info.get("safety_vetoes", 0)),
                "reward": round(float(reward), 3),
                "cumulative": round(total, 3),
            }
        )
        if terminated or truncated:
            break

    return {
        "frames": frames,
        "totalReward": round(total, 2),
        "refusedTotal": sum(f["refused"] for f in frames),
        "arrived": sum(frames[-1]["atGoal"]) if frames else 0,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/sim.yaml")
    parser.add_argument("--episodes", type=int, default=600)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", default="docs/trajectory.json")
    args = parser.parse_args(argv)

    root = Path(__file__).resolve().parents[1]
    env = DroneEnv(str(root / args.config))
    env.reset(seed=args.seed)

    # Captured before any training so the layout recorded here is the layout both
    # rollouts run against.
    obstacles = [
        [int(x), int(y)] for x, y in zip(*np.where(env.obstacles))
    ]
    goals = [[int(g[0]), int(g[1])] for g in env.goals]
    starts = [[int(p[0]), int(p[1])] for p in env.positions]

    agent = PPOAgent(env, action_masking=True)

    # Snapshots rather than a before and after pair, because a single jump hides
    # whether the middle was monotonic. It generally is not.
    checkpoints = [0, args.episodes // 4, args.episodes // 2, args.episodes]
    runs = []
    trained = 0

    print(f"{env.grid_size}x{env.grid_size} grid, {env.num_drones} drones, {args.episodes} episodes")
    for target in checkpoints:
        if target > trained:
            with contextlib.redirect_stdout(io.StringIO()):
                agent.train(num_episodes=target - trained)
            trained = target

        run = rollout(env, agent, args.seed)
        runs.append(
            {
                "label": "Untrained" if target == 0 else f"{target} episodes",
                "episodes": target,
                "detail": (
                    "randomly initialised policy"
                    if target == 0
                    else f"after {target} training episodes"
                ),
                **run,
            }
        )
        print(
            f"  {runs[-1]['label']:<16} reward {run['totalReward']:>9.2f}"
            f"   arrived {run['arrived']}/{env.num_drones}"
            f"   refused {run['refusedTotal']:>3}"
        )

    payload = {
        "gridSize": env.grid_size,
        "numDrones": env.num_drones,
        "maxSteps": env.max_steps,
        "obstacles": obstacles,
        "goals": goals,
        "starts": starts,
        "episodes": args.episodes,
        "runs": runs,
    }

    out = root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, separators=(",", ":")))

    print(f"\nwrote {out} ({out.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
