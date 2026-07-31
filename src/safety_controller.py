"""
Safety Controller
-----------------
The first component in this repository allowed to *veto*.

Everything else in the integrity layer reports and steps aside: a validator
counts a bad value and the run continues, which is the right call for a
training loop and the wrong one for anything that flies. The Safety Controller
is the other half of that idea. It sits between the policy's proposal and the
environment's movement resolution, and a move it refuses does not happen.

Rules are deliberately few and geometric, because a rule that cannot be checked
cheaply on every tick is a rule that will be skipped on a busy one.
"""

from typing import Any, Dict, List, Tuple


class SafetyController:
    """Final arbiter over proposed moves.

    Parameters
    ----------
    grid_size:
        Width of the square grid, used by the geofence.
    config:
        Optional mapping with ``geofence_margin`` and ``min_separation``.

    The defaults are permissive on purpose: ``geofence_margin`` of 0 leaves the
    whole grid usable and ``min_separation`` of 0 disables spacing entirely. A
    controller that changed behaviour the moment it was installed would make its
    effect impossible to separate from the policy's.

    Separation defaults to 0 rather than 1 for a sharper reason. Two drones in
    the same cell is distance 0, and that case already belongs to the
    environment's vertex-conflict rule, which refuses both movers. A separation
    rule of 1 would race that rule and settle it first, letting whichever drone
    was checked first proceed. That is a priority tie-break wearing a safety
    rule's clothes, and it teaches the policy that some drones always win.
    """

    def __init__(self, grid_size: int, config: Dict[str, Any] | None = None):
        config = config or {}
        self.grid_size = int(grid_size)
        self.geofence_margin = int(config.get("geofence_margin", 0))
        self.min_separation = int(config.get("min_separation", 0))

    # ------------------------------------------------------------------
    # Rules
    # ------------------------------------------------------------------
    def in_geofence(self, x: int, y: int) -> bool:
        """True when the cell is inside the permitted region."""
        m = self.geofence_margin
        return m <= x < self.grid_size - m and m <= y < self.grid_size - m

    @staticmethod
    def _chebyshev(a: Tuple[int, int], b: Tuple[int, int]) -> int:
        """Chebyshev distance: diagonal neighbours count as one apart."""
        return max(abs(a[0] - b[0]), abs(a[1] - b[1]))

    # ------------------------------------------------------------------
    # Arbitration
    # ------------------------------------------------------------------
    def review(self, current: List[Tuple[int, int]], proposed: List[Tuple[int, int]]):
        """Approve or refuse each proposed cell.

        Returns the approved cells and one reason per vetoed drone. A refused
        drone holds its current cell, which is always treated as safe: sending
        a drone somewhere else because where it already is has become illegal
        would be a worse outcome than leaving it put.
        """
        approved = list(proposed)
        reasons: Dict[int, str] = {}

        for i, cell in enumerate(proposed):
            if cell == current[i]:
                continue  # holding position is never vetoed
            if not self.in_geofence(*cell):
                approved[i] = current[i]
                reasons[i] = "geofence"

        # Separation is checked after the geofence, and against the cells that
        # survived it, so one veto cannot cascade into a phantom breach.
        if self.min_separation > 0:
            for _ in range(len(proposed) + 1):
                changed = False
                for i in range(len(approved)):
                    if approved[i] == current[i]:
                        continue
                    for j in range(len(approved)):
                        if i == j:
                            continue
                        if self._chebyshev(approved[i], approved[j]) >= self.min_separation:
                            continue
                        # Refuse both movers, never just the one examined first.
                        # Vetoing only drone i would hand drone j the cell purely
                        # because of loop order, which is a priority rule by
                        # accident and the thing this controller must not become.
                        approved[i] = current[i]
                        reasons[i] = "separation"
                        if approved[j] != current[j]:
                            approved[j] = current[j]
                            reasons[j] = "separation"
                        changed = True
                        break
                    if changed:
                        break
                if not changed:
                    break

        return approved, reasons
