"""
solver.py — Automated solvability checker for Worm Escape levels.

Uses a Directed Dependency Graph approach:

  1. Build the grid occupancy from the current set of worms.
  2. For each worm, trace forward from its head along its direction.
     If the trace hits another worm before leaving the grid, record a
     directed edge:  this_worm  --blocked-by-->  that_worm.
  3. Any worm with in-degree 0 in this "blocked-by" graph has a clear
     path to the grid edge and can be extracted.
  4. Remove all 0-dependency worms, then rebuild the graph on the
     remaining worms.  Repeat until:
       (a) all worms are extracted  →  solvable  (True)
       (b) no worm has 0 dependencies  →  cyclic deadlock  (False)

The function also returns a human-readable report describing each
resolution round or the detected deadlock for editor display.

Complexity: O(W² · max(R, C)) per round, where W = worm count and
R, C = grid dimensions.  Negligible for typical puzzle sizes (≤ 20×20,
≤ 26 worms).
"""

from __future__ import annotations

import copy
from typing import Tuple

from worm_escape.constants import DIR_DELTA
from worm_escape.entities import Worm
from worm_escape.level_manager import load_level


# ═══════════════════════════════════════════════════════════════════════════
#  INTERNAL HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _build_occupancy(worms: list[Worm]) -> dict[tuple, str]:
    """
    Build a cell → worm_id mapping for every occupied cell.
    """
    occ = {}
    for w in worms:
        for seg in w.segments:
            occ[seg] = w.worm_id
    return occ


def _first_blocker(worm: Worm, occupancy: dict[tuple, str],
                   rows: int, cols: int) -> str | None:
    """
    Trace forward from *worm*'s head along its direction vector.

    Returns the worm_id of the **first** other worm whose body is in
    the path, or None if the path is clear all the way to the grid edge.
    """
    dr, dc = DIR_DELTA[worm.direction]
    r, c = worm.head

    while True:
        r += dr
        c += dc

        # Reached the grid edge — path clear.
        if not (0 <= r < rows and 0 <= c < cols):
            return None

        cell_owner = occupancy.get((r, c))
        if cell_owner is not None and cell_owner != worm.worm_id:
            return cell_owner

    # (unreachable, but keeps linters happy)
    return None


def _build_dependency_graph(worms: list[Worm], rows: int, cols: int
                            ) -> dict[str, set[str]]:
    """
    Build the directed dependency graph.

    Returns
    -------
    blocked_by : dict[str, set[str]]
        blocked_by["A"] = {"B"} means A is blocked by B.
        A worm with an empty set has a clear extraction path.
    """
    occupancy = _build_occupancy(worms)
    blocked_by: dict[str, set[str]] = {w.worm_id: set() for w in worms}

    for w in worms:
        blocker = _first_blocker(w, occupancy, rows, cols)
        if blocker is not None:
            blocked_by[w.worm_id].add(blocker)

    return blocked_by


# ═══════════════════════════════════════════════════════════════════════════
#  PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════

def is_solvable(level_data: dict) -> Tuple[bool, list[str]]:
    """
    Determine whether *level_data* can be cleared by successive
    worm extractions.

    Parameters
    ----------
    level_data : dict
        The same schema accepted by ``load_level()``.

    Returns
    -------
    (solvable: bool, report: list[str])
        *report* is a list of human-readable lines describing the
        resolution process or the deadlock.
    """
    rows, cols, name, worms = load_level(level_data)
    report: list[str] = []
    extraction_order: list[str] = []
    round_num = 0

    while worms:
        round_num += 1
        graph = _build_dependency_graph(worms, rows, cols)

        # ── Find worms with 0 dependencies (free to extract) ──
        free = sorted(wid for wid, deps in graph.items() if len(deps) == 0)

        if not free:
            # ── Deadlock: every remaining worm is blocked ──
            report.append(f"Round {round_num}: DEADLOCK")
            for wid in sorted(graph):
                deps = ", ".join(sorted(graph[wid]))
                report.append(f"  [{wid}] blocked by: {deps}")
            report.append("")
            report.append(
                f"Remaining worms ({len(worms)}): "
                f"{', '.join(sorted(w.worm_id for w in worms))}"
            )
            report.append("Verdict: UNSOLVABLE — cyclic dependency detected.")
            return False, report

        # ── Extract all free worms in this round ──
        free_str = ", ".join(f"[{wid}]" for wid in free)
        report.append(f"Round {round_num}: extract {free_str}")
        extraction_order.extend(free)

        # Log blocked worms in this round for context.
        blocked = sorted(wid for wid, deps in graph.items() if len(deps) > 0)
        for wid in blocked:
            deps = ", ".join(sorted(graph[wid]))
            report.append(f"  [{wid}] still blocked by: {deps}")

        # Remove extracted worms from the list.
        worms = [w for w in worms if w.worm_id not in free]

    report.append("")
    report.append(
        f"Extraction order: {' → '.join(extraction_order)}  "
        f"({round_num} round{'s' if round_num != 1 else ''})"
    )
    report.append("Verdict: SOLVABLE")
    return True, report


def solve_report(level_data: dict) -> str:
    """
    Convenience wrapper: returns the full report as a single string.
    """
    solvable, report = is_solvable(level_data)
    return "\n".join(report)
