"""
PPO Agent with Integrity Validation
-----------------------------------
This file defines a Proximal Policy Optimization (PPO) agent in PyTorch.

Why PPO?
-> PPO is one of the most widely used reinforcement learning algorithms.
-> It improves stability using a "clipping trick" that prevents huge policy updates.
-> It separates policy (actions) from value (baseline), which reduces variance.

Why integrity validation?
-> After every policy decision, we check the probabilities, value estimates,
   and chosen actions to make sure they are valid.
-> This prevents silent bugs (like NaNs, negative probabilities, or invalid actions).
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical

from integrity_validators import PolicyIntegrityValidator  # schema-based validator
from utils.metrics import TRAINING_REWARD  # episode reward, for Prometheus


# ---------------- Policy Network ----------------

class PPOPolicy(nn.Module):
    """
    The policy and value networks share a common backbone.

    - Shared layers -> extract features from the state.
    - Policy head -> outputs action probabilities.
    - Value head -> predicts the baseline value of the state.

    This "actor-critic" design is standard in modern RL.
    """

    def __init__(self, obs_dim, act_dim):
        super().__init__()

        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
        )

        # Policy head: probability distribution over actions
        self.policy_head = nn.Sequential(
            nn.Linear(64, act_dim),
            nn.Softmax(dim=-1),  # ensures outputs are valid probabilities
        )

        # Value head: scalar baseline for this state
        self.value_head = nn.Linear(64, 1)

    def forward(self, x):
        features = self.shared(x)
        action_probs = self.policy_head(features)
        value = self.value_head(features)
        return action_probs, value


# ---------------- PPO Agent ----------------

class PPOAgent:
    """
    Encapsulates PPO training and inference.

    Main methods:
    - select_action(state) -> sample an action for exploration
    - train(num_episodes) -> run episodes and update policy
    - predict(state) -> pick greedy action (for inference/production)
    - save(path)/load(path) -> persist model weights
    """

    def __init__(self, env, lr=3e-4, gamma=0.99, eps_clip=0.2, epochs=3, action_masking=None):
        self.env = env
        # Invalid action masking. The observation already reports which moves are
        # blocked, so a drone should never have to walk into a wall to find out.
        # Masking removes those actions from the distribution outright, which is
        # the difference between a rule the policy must learn and a rule it
        # cannot break. Only unilaterally knowable cases are maskable: a wall, an
        # obstacle, or a neighbour standing still. Two drones entering the same
        # cell is a joint event and stays with the environment's conflict rule.
        if action_masking is None:
            action_masking = bool(getattr(env, "config", {}).get("action_masking", True))
        self.action_masking = action_masking
        self.gamma = gamma       # discount factor
        self.eps_clip = eps_clip # PPO clipping parameter
        self.epochs = epochs     # policy update iterations

        # One row of features per drone, and the same move set for each. The
        # policy weights are shared across drones: a single trunk is applied to
        # every row, so the network size does not grow with the fleet and a
        # lesson learned by one drone is available to all of them.
        raw_dim = int(env.observation_space.shape[-1])
        act_dim = int(env.action_space.nvec[0]) if hasattr(env.action_space, "nvec") else int(env.action_space.n)

        # Scales for the input transform below. Read from the environment rather
        # than hard-coded so a different grid does not silently change what the
        # network sees.
        self._grid = float(getattr(env, "grid_size", 1) or 1)
        self._horizon = float(getattr(env, "max_steps", 1) or 1)

        # Two extra inputs: the offset from the drone to its own goal. See
        # _features for why that is worth more than it looks.
        obs_dim = raw_dim + 2

        # Initialize policy network and optimizer
        self.policy = PPOPolicy(obs_dim, act_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        # Attach validator for drift/hallucination checks
        self.validator = PolicyIntegrityValidator(env.action_space)

    def _features(self, state):
        """Turn a raw observation into what the network actually consumes.

        Two changes, both about making the task easier to represent rather than
        easier to solve.

        The offset from a drone to its goal is supplied directly. The raw
        observation gives absolute position and absolute goal, so a policy has to
        learn subtraction before it can learn navigation, and it has to learn it
        separately for every region of the grid. Handing over the difference
        makes the policy translation invariant: "goal is three cells north" means
        the same thing everywhere on the board, so an episode spent in one corner
        teaches something usable in the other.

        Everything is then scaled to roughly the unit interval. Raw inputs mixed
        coordinates in 0 to 7 with a step counter in 0 to 40 and flags in 0 to 1,
        so the first layer was dominated by whichever number happened to be
        largest, and the step counter drowned out the sensors.
        """
        arr = np.atleast_2d(np.asarray(state, dtype=np.float32))
        x, y = arr[..., 0], arr[..., 1]
        gx, gy = arr[..., 2], arr[..., 3]
        steps = arr[..., 4]
        flags = arr[..., 5:]

        out = np.concatenate(
            [
                np.stack([
                    x / self._grid,
                    y / self._grid,
                    gx / self._grid,
                    gy / self._grid,
                    (gx - x) / self._grid,   # offset to goal, the useful part
                    (gy - y) / self._grid,
                    steps / self._horizon,
                ], axis=-1),
                flags,                        # already 0 or 1
            ],
            axis=-1,
        )
        return torch.as_tensor(out, dtype=torch.float32).reshape(
            *np.asarray(state).shape[:-1], out.shape[-1]
        )

    def _mask_probs(self, state, probs):
        """Zero out moves the observation already reports as blocked.

        Observation columns 5 to 8 are blocked_up, blocked_down, blocked_left
        and blocked_right, matching action indices 1 to 4. Hover is index 0 and
        is always legal, so the renormalised distribution can never be empty.
        """
        if not self.action_masking:
            return probs
        flags = torch.as_tensor(np.atleast_2d(np.asarray(state))[..., 5:9], dtype=probs.dtype)
        mask = torch.ones_like(probs)
        mask[..., 1:5] = 1.0 - flags.reshape(mask[..., 1:5].shape)
        masked = probs * mask
        return masked / masked.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    def _guard_probs(self, probs, value):
        """Report a non-finite policy output as drift before it can crash."""
        if torch.isfinite(probs).all() and torch.isfinite(value).all():
            return
        errors = [{"type": "drift", "field": "probs", "msg": "non-finite policy output"}]
        for e in errors:
            print(f"[Integrity Warning] {e['type']} on {e['field']}: {e['msg']}")
        raise ValueError(
            "policy produced non-finite outputs; the network has diverged. "
            "This is drift, not a crash: check the loss for a nan before this point."
        )

    def select_action(self, state):
        """
        Pick an action from the current policy.
        -> During training, we sample from the distribution (exploration).
        """
        raw = np.asarray(state, dtype=np.float32)
        probs, value = self.policy(self._features(raw))
        probs = self._mask_probs(raw, probs)
        # Check before Categorical is constructed. Categorical validates the
        # simplex itself and raises on a nan, which would pre-empt the validator
        # and surface a torch constraint error instead of the drift report this
        # layer exists to produce.
        self._guard_probs(probs, value)
        m = Categorical(probs)
        action = m.sample()

        actions = action.detach().numpy()
        errors = self.validator.validate(probs, value.mean(), actions)
        if errors:
            for e in errors:
                print(f"[Integrity Warning] {e['type']} on {e['field']}: {e['msg']}")

        # One log probability PER DRONE, not summed. Summing made the fleet's
        # move a single joint event, so all drones shared one advantage and the
        # drone that flew a clean route was told exactly what the drone that
        # drove into a wall was told. Keeping them separate is what lets credit
        # land on whichever drone earned it. For one drone this is unchanged.
        return actions, m.log_prob(action)

    def train(self, num_episodes=100):
        """
        Train the PPO agent for a number of episodes.

        Every quantity below is per drone, shape ``(timesteps, num_drones)``,
        rather than one number per timestep. That distinction is the difference
        between a fleet that coordinates and one that cannot: with a single
        shared advantage, a drone is credited for outcomes it had no part in and
        blamed for failures it did not cause, so its gradient is mostly other
        drones' noise. A single drone is the degenerate case and is unaffected.

        The policy itself is shared across drones. Each drone sees only its own
        observation row and contributes its own transitions, which is parameter
        sharing with independent credit.
        """
        for ep in range(num_episodes):
            state, _ = self.env.reset()
            log_probs, rewards, states, actions = [], [], [], []
            terminated, truncated = False, False

            # Rollout one episode
            while not (terminated or truncated):
                action, log_prob = self.select_action(state)
                next_state, reward, terminated, truncated, info = self.env.step(action)

                per_drone = info.get("rewards")
                if per_drone is None:
                    # An environment that reports only a team scalar cannot say
                    # who earned what, so the best available split is an even
                    # one. Credit assignment is lost here, not silently faked.
                    n = int(np.atleast_1d(np.asarray(action)).size)
                    per_drone = [float(reward) / n] * n

                states.append(state)
                actions.append(action)
                rewards.append(per_drone)
                log_probs.append(log_prob)

                state = next_state

            # Discounted returns, computed down each drone's own timeline.
            rewards_t = torch.tensor(np.array(rewards), dtype=torch.float32)
            discounted = torch.zeros_like(rewards_t)
            running = torch.zeros(rewards_t.shape[1])
            for t in reversed(range(rewards_t.shape[0])):
                running = rewards_t[t] + self.gamma * running
                discounted[t] = running

            # Normalize returns -> improves training stability.
            # Only when there are at least two samples. The unbiased std of a
            # single value is nan, and a one-step episode is not hypothetical:
            # a drone can spawn one move from its goal, reach it immediately,
            # and produce exactly one return. That nan propagates into the loss
            # and every weight in the network is permanently poisoned, which
            # looks from the outside like a policy that silently stopped
            # learning rather than like a crash.
            if discounted.numel() > 1:
                discounted = (discounted - discounted.mean()) / (discounted.std() + 1e-8)

            # Policy update for several epochs
            for _ in range(self.epochs):
                states_np = np.array(states, dtype=np.float32)
                actions_t = torch.tensor(np.array(actions), dtype=torch.long)
                old_log_probs = torch.stack(log_probs)

                # Forward pass
                probs, values = self.policy(self._features(states_np))
                # The same mask must be applied here. Sampling from a masked
                # distribution and then scoring those actions against an
                # unmasked one makes the PPO ratio compare two different
                # distributions, which silently corrupts every update.
                probs = self._mask_probs(states_np, probs)
                m = Categorical(probs)
                new_log_probs = m.log_prob(actions_t)
                entropy = m.entropy().mean()  # encourages exploration

                # One baseline per drone. Averaging the critic across drones
                # gave every drone the fleet's expected value, so a drone in a
                # good position and one in a hopeless corner shared a baseline
                # and neither advantage meant anything local.
                state_values = values.squeeze(-1)

                # Advantage = return - baseline, detached.
                # Without the detach the policy loss back-propagates through the
                # value head, so the critic is pulled by the actor's objective
                # rather than only by its own regression target.
                advantages = (discounted - state_values).detach()
                if advantages.numel() > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # PPO surrogate loss
                ratio = (new_log_probs - old_log_probs.detach()).exp()
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

                # Value loss (MSE)
                value_loss = (discounted - state_values) ** 2

                # Total loss = policy + value - entropy
                loss = -torch.min(surr1, surr2).mean() \
                       + 0.5 * value_loss.mean() \
                       - 0.01 * entropy

                # Gradient update
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            episode_reward = float(rewards_t.sum())
            # Declared in utils/metrics.py and, until now, never written to, so
            # the Grafana training-reward panel had no series behind it.
            TRAINING_REWARD.observe(episode_reward)
            print(f"Episode {ep+1}, total reward={episode_reward:.2f}")

    def save(self, path):
        """Save model weights to disk."""
        torch.save(self.policy.state_dict(), path)

    def load(self, path):
        """Load model weights from disk."""
        self.policy.load_state_dict(torch.load(path))

    def predict(self, state):
        """
        Greedy action selection (for inference).
        -> Use after training when running in production.
        """
        raw = np.asarray(state, dtype=np.float32)
        probs, value = self.policy(self._features(raw))
        probs = self._mask_probs(raw, probs)
        self._guard_probs(probs, value)
        action = torch.argmax(probs, dim=-1).detach().numpy()

        # Integrity check
        errors = self.validator.validate(probs, value.mean(), action)
        if errors:
            for e in errors:
                print(f"[Integrity Warning] {e['type']} on {e['field']}: {e['msg']}")

        return action
