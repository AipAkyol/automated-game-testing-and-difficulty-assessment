"""
engine.py — Core game logic for Worm Escape.

Contains collision detection, single-step movement, and the
multi-step extraction driver.  No I/O or rendering happens here;
the optional *on_step* callback lets callers hook in animation.
"""

import time

from worm_escape.constants import ANIM_DELAY, DIR_DELTA
from worm_escape.entities import Worm


def _occupied_by_others(worms: list, exclude_id: str) -> set:
    """Return every cell occupied by worms other than *exclude_id*."""
    cells = set()
    for w in worms:
        if w.worm_id != exclude_id:
            cells.update(w.segments)
    return cells


def _step_forward(worm: Worm, worms: list, rows: int, cols: int) -> str:
    """
    Attempt to move *worm* one cell in its facing direction.

    Returns
    -------
    'extracted' – head would leave the grid → worm should be removed.
    'blocked'   – cell ahead is occupied by another worm → no movement.
    'moved'     – worm slid forward by one cell.
    """
    dr, dc = DIR_DELTA[worm.direction]
    hr, hc = worm.head
    nr, nc = hr + dr, hc + dc

    # Off the board → extraction
    if not (0 <= nr < rows and 0 <= nc < cols):
        return "extracted"

    # Collision with another worm
    if (nr, nc) in _occupied_by_others(worms, worm.worm_id):
        return "blocked"

    # Slide: drop tail, add new head position
    worm.segments.pop(0)
    worm.segments.append((nr, nc))
    return "moved"


def attempt_extraction(worm, worms, rows, cols, on_step=None):
    """
    Slide *worm* forward repeatedly until it is blocked or extracted.

    Parameters
    ----------
    on_step : callable or None
        Invoked after each successful one-cell slide (for animation).

    Returns
    -------
    (extracted: bool, steps_moved: int)
    """
    steps = 0
    while True:
        result = _step_forward(worm, worms, rows, cols)
        if result == "moved":
            steps += 1
            if on_step:
                on_step()
                time.sleep(ANIM_DELAY)
        elif result == "extracted":
            return True, steps
        else:   # blocked
            return False, steps
