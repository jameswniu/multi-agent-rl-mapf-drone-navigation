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


class RunningNorm:
    """Mean and variance of every return seen so far, updated in one pass.

    Returns have to be rescaled before the critic regresses on them or the loss
    is dominated by whichever reward term happens to be largest. Doing that per
    batch is the obvious way and it is wrong here, because the batch is one
    episode: subtracting that episode's own mean removes exactly the thing worth
    knowing, which is whether this episode went better than episodes generally
    go. Measured on the eight drone board, an episode bringing every drone home
    returns +46.89 against -0.45 for one that strands a drone, and per-episode
    normalisation shrinks that gap of 47.3 to 0.24.

    It also gives the critic a target that stops moving. Under per-batch
    normalisation the critic is asked to predict a z-score against a ruler that
    is rebuilt from scratch every episode, so there is nothing stable to
    converge to. That is worth naming because it is what made generalised
    advantage estimation fail here: GAE bootstraps through the critic, so a
    critic trained against a moving ruler poisons every advantage it touches.

    Chan's parallel update, so a batch of any size folds in without keeping the
    samples around.

    ``horizon`` caps how much history that average is allowed to hold. Keeping
    all of it sounds more principled and measurably is not: early training is
    almost entirely failures, so an unbounded average keeps rescaling the
    returns of a competent policy against a distribution that stopped being true
    thousands of episodes ago, and the signal fades as the agent improves. On
    the four drone course an unbounded normaliser stalls flat at half the fleet
    home and never moves off it, at any budget out to 30000 episodes. Capping
    the count keeps the statistics tracking the returns actually being earned.
    """

    def __init__(self, horizon=None):
        self.mean = 0.0
        self.var = 1.0
        self.count = 1e-4
        self.horizon = horizon

    def update(self, values):
        batch_count = values.numel()
        if not batch_count:
            return
        batch_mean = float(values.mean())
        batch_var = float(values.var(unbiased=False))
        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean += delta * batch_count / total
        self.var = (
            self.var * self.count
            + batch_var * batch_count
            + delta * delta * self.count * batch_count / total
        ) / total
        # Forgetting is the whole point of the cap: holding the count at the
        # horizon makes each new batch keep a fixed share of the average rather
        # than a share that shrinks to nothing as training goes on.
        self.count = min(total, self.horizon) if self.horizon else total

    def normalize(self, values):
        return (values - self.mean) / (self.var**0.5 + 1e-8)


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

    def __init__(self, env, lr=3e-4, gamma=0.99, eps_clip=0.2, epochs=3, action_masking=None,
                 batch_episodes=None, return_norm=None, norm_horizon=None,
                 entropy_coef=None, entropy_final=None, anneal_episodes=None):
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
        # How many episodes to gather before updating. One is the smallest batch
        # PPO can run on and it shows: advantages get normalised over a single
        # rollout, three epochs of gradient steps are taken on it, and it is
        # thrown away. Across tens of thousands of updates that random-walks the
        # policy rather than improving it. Measured on the four drone flight
        # profile, longer training made things steadily worse, solving 3 seeds of
        # 5 at 8000 episodes, 1 at 20000 and 0 at 40000.
        if batch_episodes is None:
            batch_episodes = int(getattr(env, "config", {}).get("batch_episodes", 1))
        self.batch_episodes = max(1, int(batch_episodes))

        # How returns are rescaled before the critic regresses on them. "batch"
        # uses the current batch's own mean and variance, which is the usual
        # choice and is safe when a batch holds many episodes. At a batch of one
        # it erases the difference between a good episode and a bad one, and it
        # hands the critic a target that is rescaled every episode; see
        # RunningNorm for the measurement.
        if return_norm is None:
            return_norm = str(getattr(env, "config", {}).get("return_norm", "batch"))
        if return_norm not in ("batch", "running"):
            raise ValueError(f"return_norm must be 'batch' or 'running', got {return_norm!r}")
        self.return_norm = return_norm
        if norm_horizon is None:
            norm_horizon = getattr(env, "config", {}).get("norm_horizon", 100000)
        self.norm_horizon = int(norm_horizon) or None
        self._returns = RunningNorm(self.norm_horizon) if return_norm == "running" else None

        # Weight on the entropy term. The default 0.01 is the usual starting
        # point and is not always enough here: a drone that has learned to yield
        # in a crowd can collapse onto hovering, and once hover holds most of
        # the probability mass sampling stops trying anything else, so the
        # collapse maintains itself. Measured on the eight drone board, the
        # stranded drone sat on hover at 0.83 against 0.17 for the move that
        # would have freed it.
        if entropy_coef is None:
            entropy_coef = float(getattr(env, "config", {}).get("entropy_coef", 0.01))
        self.entropy_coef = float(entropy_coef)

        # Optional decay of that weight over training. Exploration and sharpness
        # are wanted at different times rather than at once: enough noise early
        # to break out of a bad habit, and little enough late that the greedy
        # policy the evaluation actually runs is decisive. Holding one value for
        # the whole run has to compromise between the two. Left equal to
        # entropy_coef, nothing anneals and the behaviour is unchanged.
        if entropy_final is None:
            entropy_final = getattr(env, "config", {}).get("entropy_final", None)
        self.entropy_final = float(entropy_final) if entropy_final is not None else self.entropy_coef
        # Read from the config for the same reason entropy_final is. A profile
        # that sets one and not the other gets no schedule at all, silently:
        # anneal_episodes of zero makes _entropy_weight hold the start value
        # forever, so the setting would look applied and do nothing.
        if anneal_episodes is None:
            anneal_episodes = getattr(env, "config", {}).get("anneal_episodes", None)
        self.anneal_episodes = int(anneal_episodes) if anneal_episodes else 0
        self._episodes_seen = 0

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
        self._ceiling = float(max(1, getattr(env, "max_altitude", 0) or 1))

        # Two extra inputs: the offset from the drone to its own goal. See
        # _features for why that is worth more than it looks.
        obs_dim = raw_dim + 2

        # Initialize policy network and optimizer
        self.policy = PPOPolicy(obs_dim, act_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        # Attach validator for drift/hallucination checks
        self.validator = PolicyIntegrityValidator(env.action_space)

    def _entropy_weight(self):
        """Entropy weight for the update about to be applied.

        Interpolates on episodes seen so far rather than on updates, so the
        schedule does not silently change shape when the batch size does.
        """
        if not self.anneal_episodes or self.entropy_final == self.entropy_coef:
            return self.entropy_coef
        progress = min(1.0, self._episodes_seen / self.anneal_episodes)
        return self.entropy_coef + (self.entropy_final - self.entropy_coef) * progress


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
        arr = np.atleast_2d(np.asarray(state, dtype=np.float32)).copy()
        x, y = arr[..., 0], arr[..., 1]
        gx, gy = arr[..., 2], arr[..., 3]
        steps = arr[..., 4]
        # Column 13 is altitude, a count rather than a flag, so it is scaled like
        # the coordinates. Left raw it would grow with the ceiling and start
        # outweighing the sensors, which is the same failure the step counter had.
        if arr.shape[-1] >= 14:
            arr[..., 13] = arr[..., 13] / self._ceiling
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
        """Zero out moves the observation already reports as impossible.

        Columns 5 to 8 are blocked_up, blocked_down, blocked_left, blocked_right
        and match action indices 1 to 4. Columns 18 and 19 say whether climbing
        and descending are legal, matching action indices 5 and 6. Hover is index
        0 and is always legal, so the renormalised distribution is never empty.

        What is deliberately NOT masked is the clearance group at columns 14 to
        17, which marks a neighbour that is blocked from the current altitude but
        reachable after climbing. Masking those would delete the decision this
        whole feature exists to pose: go over, or go around.
        """
        if not self.action_masking:
            return probs
        arr = np.atleast_2d(np.asarray(state))
        mask = torch.ones_like(probs)

        planar = torch.as_tensor(arr[..., 5:9], dtype=probs.dtype)
        mask[..., 1:5] = 1.0 - planar.reshape(mask[..., 1:5].shape)

        # Older observations stop before the vertical pair; those environments
        # have no third dimension, so there is nothing to mask.
        if arr.shape[-1] >= 20:
            vertical = torch.as_tensor(arr[..., 18:20], dtype=probs.dtype)
            mask[..., 5:7] = 1.0 - vertical.reshape(mask[..., 5:7].shape)

        masked = probs * mask

        # Normalising this row is fiddlier than it looks, and both obvious ways
        # are wrong.
        #
        # Dividing by a clamped floor fails when the surviving mass falls under
        # the floor: a confident policy puts nearly all its weight on one action,
        # and if the mask removes exactly that action the legal remainder can be
        # far below any fixed epsilon, so dividing by the floor rather than the
        # true sum leaves a row that is not a distribution. Torch renormalises
        # inside Categorical, so nothing raises and the policy keeps acting;
        # the integrity validator is what reported the drift.
        #
        # Flattening the row to uniform whenever that happens is worse, and in a
        # way that is invisible in the loss. It throws away the policy's ranking
        # among the actions that ARE legal, and greedy prediction then just takes
        # the lowest index, which is hover. Measured on a four drone flight run:
        # a drone one cell from its goal hovered for thirteen consecutive steps
        # while standing on a teammate's goal and blocking it, purely because its
        # distribution had been flattened. Two seeds of five failed that way and
        # both looked like a coordination failure rather than an arithmetic one.
        #
        # Rescaling by the row's own peak keeps the ordering and makes the sum
        # safe to divide by, because the largest legal entry becomes exactly 1.
        # Only a row whose legal mass is exactly zero falls back to uniform, and
        # there the policy genuinely has no preference left to preserve.
        tiny = torch.finfo(masked.dtype).tiny
        peak = masked.max(dim=-1, keepdim=True).values
        scaled = torch.where(peak > 0, masked / peak.clamp_min(tiny), mask)
        return scaled / scaled.sum(dim=-1, keepdim=True).clamp_min(tiny)

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

        Episodes are gathered into a batch of ``batch_episodes`` before the
        policy is updated. Every per-timestep quantity below is per drone, shape
        ``(timesteps, num_drones)``, rather than one number per timestep: with a
        single shared advantage a drone is credited for outcomes it had no part
        in and blamed for failures it did not cause, so its gradient is mostly
        other drones' noise. A single drone is the degenerate case, unaffected.

        The policy itself is shared across drones. Each drone sees only its own
        observation row and contributes its own transitions, which is parameter
        sharing with independent credit.
        """
        pending = []
        for ep in range(num_episodes):
            self._episodes_seen += 1
            state, _ = self.env.reset()
            log_probs, rewards, states, actions = [], [], [], []
            terminated, truncated = False, False

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

            episode_reward = float(np.array(rewards).sum())
            # Declared in utils/metrics.py and, until now, never written to, so
            # the Grafana training-reward panel had no series behind it.
            TRAINING_REWARD.observe(episode_reward)
            print(f"Episode {ep+1}, total reward={episode_reward:.2f}")

            # Discounted returns run down each drone's own timeline, and must be
            # computed per episode: a batch holds several, and carrying a return
            # across the boundary would credit one episode's actions with the
            # next one's outcome.
            rewards_t = torch.tensor(np.array(rewards), dtype=torch.float32)
            discounted = torch.zeros_like(rewards_t)
            running = torch.zeros(rewards_t.shape[1])
            for t in reversed(range(rewards_t.shape[0])):
                running = rewards_t[t] + self.gamma * running
                discounted[t] = running

            pending.append((states, actions, log_probs, discounted))
            if len(pending) < self.batch_episodes and ep < num_episodes - 1:
                continue

            states_np = np.array([s for e in pending for s in e[0]], dtype=np.float32)
            actions_t = torch.tensor(np.array([a for e in pending for a in e[1]]), dtype=torch.long)
            old_log_probs = torch.stack([lp for e in pending for lp in e[2]])
            discounted = torch.cat([e[3] for e in pending], dim=0)
            pending = []

            # Normalize returns -> improves training stability.
            # Only when there are at least two samples. The unbiased std of a
            # single value is nan, and a one-step episode is not hypothetical:
            # a drone can spawn one move from its goal, reach it immediately,
            # and produce exactly one return. That nan propagates into the loss
            # and every weight in the network is permanently poisoned, which
            # looks from the outside like a policy that silently stopped
            # learning rather than like a crash.
            if self._returns is not None:
                # Fold this batch in first, then rescale by the statistics of
                # every return seen so far. A single sample is fine here, which
                # is why this branch carries no size guard: the running variance
                # is never the nan that an unbiased std of one value would give.
                self._returns.update(discounted)
                discounted = self._returns.normalize(discounted)
            elif discounted.numel() > 1:
                discounted = (discounted - discounted.mean()) / (discounted.std() + 1e-8)

            for _ in range(self.epochs):
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

                ratio = (new_log_probs - old_log_probs.detach()).exp()
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
                value_loss = (discounted - state_values) ** 2

                loss = -torch.min(surr1, surr2).mean() \
                       + 0.5 * value_loss.mean() \
                       - self._entropy_weight() * entropy

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

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
