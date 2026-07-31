"""
Generate the README media from real runs.

Nothing here is staged. The progression GIF snapshots the live policy at
intervals during one training run and replays an episode from each snapshot, so
the improvement on screen is the improvement that actually happened.

    python scripts/make_media.py
"""

import copy
import io
import contextlib
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import PillowWriter

from env.drone_env import DroneEnv
from agents.ppo_agent import PPOAgent

BG, PANEL, GRIDC = "#0b120e", "#172018", "#2e3830"
SILVER, MUTED, DIM = "#e8ebe9", "#a5aca7", "#757c77"
ACCENT, ALERT = "#6faa85", "#ff6b4a"

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(ROOT, "assets")

# The single-drone profile is the one this agent genuinely SOLVES: reward +10
# means it reached the goal. Multi-drone reward improves under shaping without
# the drones actually arriving, so animating that as "getting better" would be
# showing a number rise rather than a task being done.
SOLVE_CFG = """grid_size: 7
num_drones: 1
obstacle_density: 0.10
max_steps: 40
reward_shaping: true
fixed_layout: true
"""

# Used only for the conflict panel, where the point is the safety machinery and
# not the policy's competence.
FLEET_CFG = """grid_size: 12
num_drones: 6
obstacle_density: 0.10
max_steps: 60
fixed_layout: false
"""


def _cfg_path(body, name="_media_env.yaml"):
    p = os.path.join(ASSETS, name)
    with open(p, "w") as f:
        f.write(body)
    return p


def _frame(ax, env, refused, trail, title, sub):
    ax.clear()
    ax.set_facecolor(PANEL)
    n = env.grid_size
    ax.set_xlim(-0.5, n - 0.5); ax.set_ylim(-0.5, n - 0.5)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(GRIDC)
    for i in range(n + 1):
        ax.axhline(i - 0.5, color=GRIDC, lw=0.4, zorder=0)
        ax.axvline(i - 0.5, color=GRIDC, lw=0.4, zorder=0)

    xs, ys = np.where(env.obstacles)
    ax.scatter(xs, ys, marker="s", s=210, c="#0c110d", edgecolors=GRIDC, lw=0.8, zorder=1)
    for i, (gx, gy) in enumerate(env.goals):
        ax.plot([env.positions[i][0], gx], [env.positions[i][1], gy],
                color=ACCENT, lw=0.7, alpha=0.25, zorder=1)
    ax.scatter(env.goals[:, 0], env.goals[:, 1], marker="s", s=170,
               facecolors="none", edgecolors=ACCENT, lw=1.7, zorder=2)

    if trail and len(trail) > 1:
        tr = np.array(trail)
        ax.plot(tr[:, 0], tr[:, 1], color=SILVER, lw=1.6, alpha=0.35, zorder=3)

    home = np.all(env.positions == env.goals, axis=1)
    ok = [i for i in range(env.num_drones) if i not in refused and not home[i]]
    if ok:
        p = env.positions[ok]
        ax.scatter(p[:, 0], p[:, 1], s=220, c=SILVER, edgecolors="#5a625c", lw=0.9, zorder=4)
    if home.any():
        p = env.positions[home]
        ax.scatter(p[:, 0], p[:, 1], s=220, c=ACCENT, edgecolors=SILVER, lw=1.2, zorder=5)
    if refused:
        p = env.positions[list(refused)]
        ax.scatter(p[:, 0], p[:, 1], s=420, facecolors="none", edgecolors=ALERT, lw=2.6, zorder=6)
        ax.scatter(p[:, 0], p[:, 1], s=220, c=ALERT, zorder=7)

    # Title and subtitle as figure text, so they cannot overlap each other.
    ax.set_title(title, color=SILVER, fontsize=15, fontweight="bold", loc="left", pad=26)
    ax.text(0, 1.020, sub, transform=ax.transAxes, color=DIM, fontsize=10.5, va="bottom")


def progression_gif(path, episodes=600, snap_every=150, replay=34):
    """Snapshot the live policy during one run and replay a greedy episode from each.

    Single drone, because this is the profile the agent actually solves. The
    trail shows where it has been, so a wandering policy and a direct one are
    distinguishable at a glance rather than only in the reward number.
    """
    env = DroneEnv(_cfg_path(SOLVE_CFG))
    agent = PPOAgent(env)
    snaps = [(0, copy.deepcopy(agent.policy.state_dict()))]
    for block in range(episodes // snap_every):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            agent.train(num_episodes=snap_every)
        snaps.append(((block + 1) * snap_every, copy.deepcopy(agent.policy.state_dict())))

    fig, ax = plt.subplots(figsize=(6.4, 6.9), dpi=105)
    fig.patch.set_facecolor(BG)
    fig.subplots_adjust(left=0.06, right=0.94, top=0.82, bottom=0.05)

    writer = PillowWriter(fps=6)
    summary = []
    with writer.saving(fig, path, dpi=105):
        for ep, state in snaps:
            agent.policy.load_state_dict(state)
            obs, _ = env.reset(seed=0)
            trail = [tuple(env.positions[0])]
            arrived_at = None
            for t_i in range(replay):
                obs, _, term, trunc, info = env.step(agent.predict(obs))
                trail.append(tuple(env.positions[0]))
                if term and arrived_at is None:
                    arrived_at = t_i + 1
                tag = "before training" if ep == 0 else f"after {ep} episodes"
                status = f"ARRIVED in {arrived_at} moves" if arrived_at else f"step {t_i+1}/{replay}, still searching"
                _frame(ax, env, [], trail, f"1 drone, 7x7 grid  |  {tag}", status)
                writer.grab_frame(facecolor=BG)
                if term:
                    for _ in range(10):
                        writer.grab_frame(facecolor=BG)
                    break
            summary.append((ep, arrived_at))
    plt.close(fig)
    print(f"  wrote {path}")
    for ep, arr in summary:
        label = "untrained" if ep == 0 else f"ep {ep}"
        print(f"    {label:>10}: {'arrived in ' + str(arr) + ' moves' if arr else 'never arrived'}")


def learning_curve(path, seeds=(0, 1, 2), episodes=600):
    runs = []
    for sd in seeds:
        torch.manual_seed(sd)
        env = DroneEnv(_cfg_path(SOLVE_CFG))
        agent = PPOAgent(env)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            agent.train(num_episodes=episodes)
        runs.append([float(x.split("=")[1]) for x in buf.getvalue().splitlines() if "total reward" in x])
    runs = np.array(runs)
    sm = lambda a, k=15: np.convolve(a, np.ones(k) / k, mode="valid")

    fig, ax = plt.subplots(figsize=(9, 4.6), dpi=120)
    fig.patch.set_facecolor(BG); ax.set_facecolor(PANEL)
    for r in runs:
        ax.plot(sm(r), color=SILVER, lw=1.0, alpha=0.35)
    ax.plot(sm(runs.mean(axis=0)), color=ACCENT, lw=2.6, label=f"mean of {len(seeds)} seeds")
    ax.set_xlabel("episode", color=MUTED, fontsize=10)
    ax.set_ylabel("episode reward (15-ep mean)", color=MUTED, fontsize=10)
    ax.set_title("Learning to reach the goal, 1 drone on 7x7", color=SILVER,
                 fontsize=14, fontweight="bold", loc="left", pad=14)
    ax.text(0, 1.03, "thin lines are individual seeds; every one improves",
            transform=ax.transAxes, color=DIM, fontsize=10, va="bottom")
    ax.tick_params(colors=DIM, labelsize=9)
    for s in ax.spines.values():
        s.set_color(GRIDC)
    ax.grid(color=GRIDC, lw=0.5, alpha=0.6)
    ax.legend(facecolor=PANEL, edgecolor=GRIDC, labelcolor=MUTED, fontsize=9.5)
    fig.tight_layout(); fig.savefig(path, facecolor=BG); plt.close(fig)
    print(f"  wrote {path}  (first40 {runs[:, :40].mean():.1f} -> last40 {runs[:, -40:].mean():.1f})")


if __name__ == "__main__":
    os.makedirs(ASSETS, exist_ok=True)
    print("generating README media from real runs")
    progression_gif(os.path.join(ASSETS, "learning-progression.gif"))
    learning_curve(os.path.join(ASSETS, "learning-curve.png"))
    os.remove(os.path.join(ASSETS, "_media_env.yaml"))
