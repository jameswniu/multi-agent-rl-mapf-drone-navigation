#!/usr/bin/env python3
"""Four-quadrant reward-density sweep.

Same environment, same PPO hyperparameters, same seeds. The only thing that
varies across the four cells is which reward-density switches are on, so any
behavioural difference is attributable to reward shape rather than to the task,
the algorithm, or the initialisation.

Metrics per run, all from greedy evaluation after training:
  drones_home     mean drones home per episode, the headline task metric
  stranded_rate   fraction of episodes ending with 1 to n-1 drones home, which
                  is the signature of the completion-trading exploit
  solved_rate     fraction of episodes with every drone home
  collisions      mean refused moves per episode, the peer-conflict metric
  clearance       mean g(coop_quality) over all steps, peer-interaction quality
"""
import argparse, json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import torch
from env.drone_env import DroneEnv
from agents.ppo_agent import PPOAgent

QUADRANTS = ["sparse-sparse", "dense-sparse", "sparse-dense", "dense-dense"]


def evaluate(agent, env, episodes=100, seed=0):
    home, stranded, solved, collisions, clearance = [], 0, 0, [], []
    n = env.num_drones
    for ep in range(episodes):
        state, _ = env.reset(seed=seed * 10_000 + ep)
        done = False
        refused = 0
        steps = 0
        while not done:
            with torch.no_grad():
                action = agent.predict(state)
            state, _, terminated, truncated, info = env.step(action)
            refused += int(np.sum(info.get("collided", np.zeros(n))))
            clearance.append(float(np.mean(env._coop_quality())))
            steps += 1
            done = terminated or truncated
        at_goal = int(np.sum(np.all(env.positions == env.goals, axis=1) & (env.altitudes == 0)))
        home.append(at_goal)
        if at_goal == n:
            solved += 1
        elif at_goal > 0:
            stranded += 1
        collisions.append(refused)
    return {
        "drones_home": float(np.mean(home)),
        "stranded_rate": stranded / episodes,
        "solved_rate": solved / episodes,
        "collisions": float(np.mean(collisions)),
        "clearance": float(np.mean(clearance)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=2000)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--eval-episodes", type=int, default=100)
    ap.add_argument("--out", default=str(ROOT / "paper" / "results" / "sweep.json"))
    args = ap.parse_args()

    results = {}
    t0 = time.time()
    for quad in QUADRANTS:
        results[quad] = []
        cfg = str(ROOT / "configs" / "quadrants" / f"{quad}.yaml")
        for seed in range(args.seeds):
            torch.manual_seed(seed)
            np.random.seed(seed)
            env = DroneEnv(config_path=cfg)
            agent = PPOAgent(env)
            agent.train(num_episodes=args.episodes)
            m = evaluate(agent, env, episodes=args.eval_episodes, seed=seed)
            m["seed"] = seed
            results[quad].append(m)
            print(f"[{time.time()-t0:7.0f}s] {quad:<14} seed {seed}  "
                  f"home {m['drones_home']:.2f}  stranded {m['stranded_rate']:.2f}  "
                  f"solved {m['solved_rate']:.2f}  coll {m['collisions']:.1f}  "
                  f"clear {m['clearance']:.2f}", flush=True)

    os.makedirs(Path(args.out).parent, exist_ok=True)
    payload = {
        "config": {"episodes": args.episodes, "seeds": args.seeds,
                   "eval_episodes": args.eval_episodes,
                   "elapsed_seconds": round(time.time() - t0, 1)},
        "runs": results,
    }
    Path(args.out).write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {args.out} in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
