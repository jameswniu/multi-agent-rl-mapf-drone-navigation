"""
Main Training Script
--------------------
This file is the "orchestrator":
- It creates the environment (DroneEnv).
- It instantiates the PPO agent.
- It runs training for a few episodes.
- It saves the model to disk.
- It runs a quick inference demo.

Why separate this?
-> By keeping main.py clean and explicit, the workflow is easy to follow.
-> Environment logic stays in drone_env.py, agent logic stays in ppo_agent.py.
-> This separation mirrors good software engineering practice.
"""

import argparse
import os
from pathlib import Path
from typing import Any, Dict

from env.drone_env import DroneEnv
from agents.ppo_agent import PPOAgent
from integrity_stats import IntegrityStats  # tracks drift vs hallucination stats

try:  # pragma: no cover - optional dependency
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None


def load_train_config(config_path=None) -> Dict[str, Any]:
    """
    Read training hyperparameters from a YAML file.

    Returns an empty dict when no path is given or the file is absent, so the
    caller keeps its own defaults rather than crashing. Passing a path that
    does not exist is a caller mistake and is reported as one.
    """
    if config_path is None:
        return {}

    path = Path(config_path)
    if not path.is_absolute():
        # Resolve relative to the project root, one level above src, so the
        # same invocation works from the repo root and from inside src.
        path = Path(__file__).resolve().parents[1] / path
    if not path.exists():
        raise FileNotFoundError(f"training config not found: {config_path}")

    with path.open("r") as f:
        if yaml is not None:
            return yaml.safe_load(f) or {}
        # Fallback parser for key: value lines, matching DroneEnv's behaviour.
        data: Dict[str, Any] = {}
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or ":" not in line:
                continue
            key, value = line.split(":", 1)
            value = value.strip()
            data[key.strip()] = float(value) if "." in value else int(value)
        return data


def train_and_save(model_path="models/ppo_drone.pt", num_episodes=10, config=None):
    """
    End-to-end training routine.

    Steps:
    1. Create environment.
    2. Create PPO agent, using any hyperparameters supplied in ``config``.
    3. Train agent for a given number of episodes.
    4. Save the trained model to disk.
    5. Print an integrity report.
    """
    stats = IntegrityStats()
    config = config or {}

    # Step 1 -> Environment
    env = DroneEnv()

    # Step 2 -> Agent. Anything absent from the config keeps the PPOAgent default.
    agent_kwargs = {}
    if "learning_rate" in config:
        agent_kwargs["lr"] = float(config["learning_rate"])
    if "gamma" in config:
        agent_kwargs["gamma"] = float(config["gamma"])
    agent = PPOAgent(env, **agent_kwargs)

    # Monkey-patch validator to record stats after each check
    original_validate = agent.validator.validate
    def wrapped_validate(probs, value, action=None):
        errors = original_validate(probs, value, action)
        stats.record_policy(errors)
        return errors
    agent.validator.validate = wrapped_validate

    # Step 3 -> Train
    print(f"Starting training for {num_episodes} episodes...")
    agent.train(num_episodes=num_episodes)

    # Step 4 -> Save
    os.makedirs("models", exist_ok=True)
    agent.save(model_path)
    print(f"Model saved to {model_path}")

    # Step 5 -> Report integrity stats
    stats.report(prefix="[Training Integrity Report]")

    return agent, env


def run_inference(agent, env, rollout_len=5):
    """
    Run a quick inference demo.
    - Reset the environment.
    - Let the trained agent pick greedy actions.
    - Print out actions and rewards.
    """
    stats = IntegrityStats()

    state, _ = env.reset()
    total_reward = 0.0

    for t in range(rollout_len):
        # Agent predicts best action (greedy)
        action = agent.predict(state)
        state, reward, terminated, truncated, info = env.step(action)

        # Record env-level integrity stats
        stats.record_env(info)
        total_reward += reward

        print(f"Step {t+1}: action={env.action_map[action]}, reward={reward:.2f}")

        if terminated or truncated:
            break

    print(f"Total reward over {t+1} steps = {total_reward:.2f}")
    stats.report(prefix="[Inference Integrity Report]")


def parse_args(argv=None):
    """Command line surface for a training run."""
    parser = argparse.ArgumentParser(
        prog="main",
        description="Train the PPO drone agent, save it, then run a short inference rollout.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="YAML file of training hyperparameters, for example configs/train.yaml",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Number of training episodes. Overrides num_episodes from the config file.",
    )
    parser.add_argument(
        "--model-path",
        default="models/ppo_drone.pt",
        help="Where to write the trained weights.",
    )
    parser.add_argument(
        "--rollout-len",
        type=int,
        default=5,
        help="Number of inference steps to run after training.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    config = load_train_config(args.config)

    # Precedence: explicit flag, then config file, then the built-in default.
    episodes = args.episodes
    if episodes is None:
        episodes = int(config.get("num_episodes", 10))

    agent, env = train_and_save(
        model_path=args.model_path, num_episodes=episodes, config=config
    )
    run_inference(agent, env, rollout_len=args.rollout_len)


if __name__ == "__main__":
    main()
