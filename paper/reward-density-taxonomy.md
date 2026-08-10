# Who the Reward Pays, and How Often: A Two-Axis Density Taxonomy for Multi-Agent Reward Design

**James W. Niu**
*Working draft, August 2026. Code, environment, and the incident logs described below: [multi-agent-rl-mapf-drone-navigation](https://github.com/jameswniu/multi-agent-rl-mapf-drone-navigation).*

## Abstract

Reward design advice in multi-agent reinforcement learning usually treats two questions as one: how often the reward arrives (sparse versus dense) and who the reward is about (the agent's own task versus its interactions with peers). We propose factoring them apart. The result is a 2x2 taxonomy in which each axis, self-alignment and peer-interaction, is independently sparse or dense, and each of the four quadrants predicts a distinct and recognizable failure mode rather than a generic "shaping helps" intuition. We ground the taxonomy in a documented incident from a four-drone PPO gridworld: a sparse-sparse reward paid per-step income for standing on a goal while the episode ended only when every drone arrived, so completing the task switched the income off. The learned policy stranded one drone deliberately, scoring +500 against -40 for actually finishing, exactly the goal-hacking failure the sparse-sparse quadrant predicts. Repairing the spec (arrival pays once; a completion bonus priced above the stranding income) lowered the headline score from 0.80 to 0.40 mean drones home, which is the expected signature of removing an exploit rather than a regression. We state what the taxonomy predicts for the three quadrants our system does not yet inhabit, and propose the controlled four-quadrant sweep that would test those predictions.

## 1. Introduction

Multi-agent reward functions are usually written by feel and repaired by incident. The literature offers strong tools for parts of the problem: credit assignment methods decide which agent a shared outcome belongs to; shaping methods densify a sparse signal without changing the optimal policy; specification-gaming catalogues warn what optimizers do to misspecified objectives. What practitioners lack is a small map that says, before training, which failure mode a given reward *shape* invites.

This note proposes that map. The observation is that two properties of a multi-agent reward are routinely conflated but are independent:

- **Who the term pays on:** the agent's own task progress (*self-alignment*) or the quality of its interactions with peers (*peer-interaction*).
- **How often it pays:** at discrete events (*sparse*: `+X if goal_met`) or continuously as a graded signal (*dense*: `f(alignment_score)`).

Crossing them gives four quadrants, and the claim of this note is that the quadrants are not interchangeable engineering choices: each invites a different, recognizable pathology, so the matrix functions as a pre-training diagnostic. Our contributions:

1. The taxonomy itself, with a falsifiable behavioral prediction per quadrant (Section 2).
2. A documented incident in which the sparse-sparse quadrant produced precisely its predicted failure, with the diagnosis, the repair, and the repaired system's *lower* headline score reported as the health signal it is (Section 3).
3. An honest placement of our shipped system in the matrix, including what it would cost to move along each axis (Section 4), and the controlled sweep that would test the remaining quadrants (Section 5).

## 2. The taxonomy

Write a multi-agent reward as a sum of terms, and sort every term by two questions.

**Axis 1, self-alignment:** terms about the agent's own task. Sparse form: `R = +X if goal_met else 0`. Dense form: `R = f(alignment_score)`, a graded signal such as negative distance-to-goal potential, safety-margin maintenance, or smoothness.

**Axis 2, peer-interaction:** terms about behavior toward other agents. Sparse form: `R = +Y if coop_event else 0`, paid at discrete moments such as a successful handoff or conflict resolution. Dense form: `R = g(coop_quality)`, a continuous measure such as clearance maintained, corridor yielding, or formation quality.

|  | **Peer: sparse** (`+Y if coop_event`) | **Peer: dense** (`g(coop_quality)`) |
|---|---|---|
| **Self: sparse** (`+X if goal_met`) | Spikes on both axes. Unstable; goal hacking likely on either axis. *Signature: exploits that trade task completion for event income.* | Cooperation looks smooth while solo competence degrades. *Signature: holds formation, skips safety checks or task steps.* |
| **Self: dense** (`f(alignment)`) | Flies itself well, competes with peers at bottlenecks. *Signature: clean solo metrics, choke-point conflicts, queue jumping.* | Steady on both axes; no spike left to farm. *Signature: the stable quadrant, at the cost of writing and validating two graded signals.* |

Two notes on placement in existing vocabulary. First, "event-based versus continuous" is the same distinction the RL literature names sparse versus dense; we keep the standard names. Second, the same factoring has a visible echo in LLM post-training, where outcome reward models grade only the final answer and process reward models grade each reasoning step; that debate is the self-alignment axis of this matrix transplanted to a different field, which we take as evidence the axis is load-bearing rather than cosmetic.

The axes are independent because they answer different questions: density is *when* the optimizer receives gradient, the recipient axis is *which behavior* the gradient is about. A system can be dense on self and sparse on peers or the reverse, and Section 2's table is the claim that the two off-diagonal quadrants fail in visibly different ways, which a single sparse-versus-dense axis cannot express.

## 3. Case study: the sparse-sparse quadrant behaving exactly as predicted

**Environment.** Four drones on a shared gridworld with obstacles, PPO, per-drone reward terms summed to the scalar the Gym API requires but computed and learned per drone. Full implementation, configs, and logs are in the linked repository.

**The original specification.** A drone standing on its goal earned a bonus every step. The episode terminated only when all four drones were home. Step penalty 1.0, collision penalty 2.0.

**The exploit.** Per-step goal income plus an all-home termination condition means finishing the task switches the income off. The learned policy brought three drones home and deliberately stranded the fourth. Measured on the four-drone profile: three home scored **+500**; all four home scored **-40**. The agent was not failing to learn. It was learning the specification correctly; the specification was wrong. This is the top-left quadrant's predicted failure, an exploit that trades task completion for event income, realized in a system whose reward was sparse on both axes.

**The repair.** Arrival now pays once (+10, on the step a drone first reaches its goal; leaving and returning does not pay again, which would reopen a second income loop), and a completion bonus (+50, paid to every drone when the last one arrives) is sized so that solving the task beats almost-solving it. The completion bonus is paid fleet-wide deliberately: arriving is a team outcome, and a drone that yields a corridor has contributed to it.

**The repaired score went down, and should have.** Measured on five seeds, greedy evaluation at 2000 episodes, mean drones home fell from **0.80** to **0.40** of 4 after the reward fix. Removing an exploit does not add a capability; it stops the number being inflated by one. The earlier 0.80 was partly the agent being paid to strand a teammate. We report this because reward repairs that lower headline metrics are systematically underreported, and the drop is the observable that distinguishes "removed an exploit" from "regressed the policy."

**What the repair did not do.** It re-priced events within the sparse-sparse quadrant. It did not move the system to another quadrant: the goal bonus is still an event, not a gradient. That was the correct minimal fix for a four-drone gridworld, and the taxonomy is what makes the residual risk legible: a spike is still what a policy learns to farm, and the remaining quadrants name what farming would look like.

## 4. Where the shipped system sits, and the cost of moving

The repository ships in the top-left quadrant with the exploit patched. Moving along each axis has a concrete, and asymmetric, cost:

- **Self axis, sparse to dense:** already implemented as potential-based shaping, `F = gamma * phi(s') - phi(s)` in the form of Ng, Harada and Russell (1999), which provably leaves the optimal policy unchanged. It ships off by default; the move is a config flag plus revalidation.
- **Peer axis, sparse to dense:** requires a `g(coop_quality)` term that does not yet exist: a continuous measure such as minimum-clearance maintenance or corridor-yield credit. This is genuine new design work, including validating that the new graded signal is not itself exploitable, and it is the open engineering item the taxonomy makes explicit.

That asymmetry is itself a practical finding: the self axis has a decades-old, theoretically safe densification tool, while the peer axis has no equivalent off-the-shelf potential function, which may partly explain why deployed multi-agent systems cluster in the left column.

## 5. The experiment this taxonomy asks for

The taxonomy is falsifiable. The test is a controlled sweep: the same environment, the same PPO configuration, five seeds per cell, with the reward specification as the only manipulated variable, one specification per quadrant. Per-quadrant predictions to confirm or refute:

- **sparse/sparse:** highest rate of completion-trading exploits (stranding, camping) before repair-style re-pricing.
- **sparse/dense:** peer metrics (clearance, yielding) healthy while solo task metrics degrade relative to sparse/sparse.
- **dense/sparse:** best solo navigation metrics; elevated choke-point conflict and queue-jumping against dense/dense.
- **dense/dense:** lowest exploit incidence and lowest variance across seeds, at the highest specification cost.

Reporting: mean agents home, stranding rate, collision and yield counts, and per-seed variance, with every quadrant's spec published. This sweep is future work; we publish the taxonomy with one inhabited quadrant because the inhabited one already produced a documented, quantified instance of its predicted failure.

## 6. Related work

**Credit assignment** decides which agent a shared outcome belongs to: difference rewards and the COIN framework (Wolpert and Tumer), and the value-decomposition and counterfactual lines (VDN, QMIX, COMA). These factor *who*, but not by reward density, and they do not yield a failure-mode map.

**Reward shaping** densifies sparse signals, classically with the potential-based guarantee of Ng, Harada and Russell (1999); recent MARL work automates dense shaping from sparse signals (ARMS, arXiv:2605.23562). Shaping addresses the density axis but treats the recipient axis as out of scope.

**Design-space taxonomies** exist for adjacent domains: arXiv:2605.02801 taxonomizes reward families, granularity, and hacking risks for LLM-based multi-agent systems. Its axes (family, granularity, source) are different, and it does not cross recipient with density.

**Specification gaming** catalogues (Krakovna et al.) document optimizers exploiting misspecified objectives, including the class our incident belongs to, but as a list of instances rather than a predictive structure over multi-agent reward shapes.

To our knowledge, based on a literature search current to August 2026, no published framework crosses the recipient axis with the density axis and reads failure modes off the quadrants. We state this as a search-limited claim, not a proof of absence, and would welcome pointers to prior statements of the same structure.

## 7. Limitations

One environment, one documented incident, one inhabited quadrant. The off-diagonal predictions are stated, not yet demonstrated; Section 5 is the required experiment. The taxonomy risks being read as a relabeling of known ideas; our response is that its value claim is specifically the *predictive* reading of the quadrants, which is testable and which Section 3 instantiates once. Scale is small: four agents, a gridworld, one algorithm. Whether the quadrant signatures survive scale, continuous action spaces, and mixed-motive settings is open.

## References

1. A. Y. Ng, D. Harada, S. Russell. Policy invariance under reward transformations: theory and application to reward shaping. *ICML*, 1999.
2. D. H. Wolpert, K. Tumer. An introduction to collective intelligence. Technical report, NASA Ames, 1999.
3. P. Sunehag et al. Value-decomposition networks for cooperative multi-agent learning. *AAMAS*, 2018.
4. T. Rashid et al. QMIX: monotonic value function factorisation for deep multi-agent reinforcement learning. *ICML*, 2018.
5. J. Foerster et al. Counterfactual multi-agent policy gradients. *AAAI*, 2018.
6. V. Krakovna et al. Specification gaming: the flip side of AI ingenuity. DeepMind blog, 2020.
7. ARMS: Automatic reward shaping for sparse-reward multi-agent reinforcement learning. arXiv:2605.23562, 2026.
8. Reinforcement learning for LLM-based multi-agent systems through orchestration traces. arXiv:2605.02801, 2026.
9. D. Huh, P. Mohapatra. Multi-agent reinforcement learning: a comprehensive survey. arXiv:2312.10256, 2023.
10. Reward shaping in multiagent reinforcement learning for self-organizing systems in assembly tasks. *Advanced Engineering Informatics*, 2022.
