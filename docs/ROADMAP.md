# Roadmap

You found the quiet page. Nothing here is promised, dated, or linked from the
front door; it is the honest list of what this repository does not do yet,
kept where only the curious end up.

## Where things stand

Every shipped profile trains and evaluates on a fixed layout, on purpose. A
fixed course is a reproducible task: it can be measured against an exact
optimum, replayed step for step in the viewer, and a claim about it can be
checked by anyone with one command. That is the point of the repo as it
stands, and it is done: all five profiles solve on every seed, up to eight
drones sharing one board.

## The next mountain: layouts that fight back

The frontier is generalisation across layouts, not more drones on the same
one.

- **Randomised courses.** Obstacles appear at random and take squares off the
  map, a fresh board every episode, so the policy has to read the world it
  wakes up in instead of memorising one. The sensing already supports this,
  four local flags per drone; nothing about the training does.
- **Layout holdout.** Train on one distribution of boards, evaluate on boards
  never seen. The gap between those two numbers is the honest measure of
  whether anything generalised.
- **Scale after generality.** Ten drones on 20x20 was measured once and it
  defeats this implementation. There is no point revisiting scale until a
  policy survives a board it has not seen; a bigger memorised course is still
  a memorised course.
- **Moving obstacles, eventually.** A square that disappears mid-episode is
  the same sensing problem as a peer drone, which the fleet already handles.
  That symmetry is the reason to believe this rung is reachable.

## Why this is not in the README

The README describes what is, measured and replayable. This page describes
what might be. Keeping the two apart is a feature.
