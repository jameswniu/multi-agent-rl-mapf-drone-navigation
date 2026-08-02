# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- Eight drones on the 8x8 obstacle course, solved: 10 of 10 seeds bring every
  drone home and hold, in 21 steps against a best-found central plan of 17.
  The fix was exploration, `entropy_coef: 0.06` on that profile alone, after
  feasibility, congestion, geometry, longer training, curriculum transfer and
  epoch count were each measured and ruled out; the negative results live in
  `configs/fly-fleet8.yaml`.
- Arrival now means arrived and stayed. A drone that leaves a goal it reached
  is disqualified for the episode, shown amber in the viewer, and counted
  separately as occupying; end-of-run labels, live readout, and board colors
  all answer the same question.
- `scripts/plan_optimum.py`: space-time A* over prioritised orderings, the
  feasibility bound the learned policies are measured against.
- Browser viewer rebuilt as an industrial yard flown by a real drone model
  (VR Drone by Dave404, CC-BY, with a procedural quadcopter as fallback):
  runtime canvas textures, sun shadows, occlusion fade kept, helipads with
  owner rings, X, Y, Z and Default camera flights, eased continuous zoom.
  The dusk-city variant is preserved on the `viewer-city-b` branch.
- README landing: recorded eight-drone rollout as a looping inline animation,
  plus stills shot from the shipped build.
- Opt-in training knobs, off by default and measured before shelving: running
  return normalisation with a forgetting horizon, entropy annealing.

- Multiple drones sharing one grid, sized by `num_drones`, with shared policy weights.
- Path finding: vertex, swap and stationary conflicts detected and refused each step.
- Obstacles drawn from `obstacle_density`, with four local sensor flags per drone.
- Safety Controller with geofence and separation rules, the only component permitted to veto.
- Training configuration via `--config`, applied to the agent rather than discarded.
- `configs/env-prod.yaml` as a loadable production profile.

### Changed
- `/predict` takes one observation row per drone and returns action indices and names.
- Environment migrated from the deprecated `gym` package to `gymnasium`.
- Observation space bounds are per-dimension.
- Imports normalised to a single package layout, which unbroke the production image.

### Fixed
- CI on all three workflows: the `gymnasium` import, an `httpx` incompatibility, and the missing editable install in the Docker image.
- `observation_space` declared one scalar bound across every dimension, so every step reported a spurious drift error.
- `/predict` accepted no valid input: the schema wanted a mapping, the policy wanted numbers.
- `tests/conftest.py` swallowed import errors and substituted stubs, masking both bugs above.
- `training_reward` was declared but never observed, leaving its Grafana panel empty.

## [0.1.0] - 2025-09-08

### Added
- Initial production baseline, observability, deployment configs and utilities.

---

[Unreleased]: https://github.com/jameswniu/multi-agent-rl-mapf-drone-navigation/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/jameswniu/multi-agent-rl-mapf-drone-navigation/releases/tag/v0.1.0
