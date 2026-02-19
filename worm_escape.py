#!/usr/bin/env python3
"""
Worm Escape — A console-based directional extraction puzzle game.

HOW TO PLAY
-----------
  Each worm on the board has a fixed facing direction (shown by its head
  arrow: ^ v < >).  Type that worm's letter ID to slide it forward.
  The worm moves one cell at a time until it either:
    • exits the grid boundary  →  extracted!  (removed from the board)
    • hits another worm's body →  blocked!    (stays where it stopped)

  Extract ALL worms to win the level.

LEVEL FORMAT
------------
  Levels are plain Python dicts.  See the LEVELS list (or the load_level
  docstring) for the full specification.  This data-driven design lets a
  separate Level Editor tool export directly to the same format.

Requires Python 3.6+ and a terminal with ANSI color support.
"""

import os
import sys
import time


# ═══════════════════════════════════════════════════════════════════════════
#  ANSI CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════

ANSI_RESET = "\033[0m"
ANSI_BOLD  = "\033[1m"
ANSI_DIM   = "\033[2m"

ANSI_COLORS = {
    "red":     "\033[91m",
    "green":   "\033[92m",
    "yellow":  "\033[93m",
    "blue":    "\033[94m",
    "magenta": "\033[95m",
    "cyan":    "\033[96m",
    "white":   "\033[97m",
}

# Directional arrows used to render each worm's head.
HEAD_ARROWS = {"up": "^", "down": "v", "left": "<", "right": ">"}

# Row / column deltas for each compass direction.
DIR_DELTA = {
    "up":    (-1,  0),
    "down":  ( 1,  0),
    "left":  ( 0, -1),
    "right": ( 0,  1),
}

# Seconds to pause between animation frames when a worm slides.
ANIM_DELAY = 0.18


# ═══════════════════════════════════════════════════════════════════════════
#  WORM CLASS
# ═══════════════════════════════════════════════════════════════════════════

class Worm:
    """
    A single worm entity on the grid.

    Attributes
    ----------
    worm_id   : str   – single uppercase letter, e.g. 'A'.
    segments  : list  – [(row, col), ...] ordered **tail → head**.
                        segments[-1] is always the HEAD.
    direction : str   – 'up' | 'down' | 'left' | 'right'.
    color     : str   – key into ANSI_COLORS.
    """

    def __init__(self, worm_id: str, segments: list, direction: str, color: str):
        self.worm_id   = worm_id
        self.segments  = list(segments)
        self.direction = direction
        self.color     = color

    @property
    def head(self):
        """The (row, col) of the worm's head (frontmost segment)."""
        return self.segments[-1]


# ═══════════════════════════════════════════════════════════════════════════
#  LEVEL LOADER  (data-driven, editor-friendly format)
# ═══════════════════════════════════════════════════════════════════════════

def load_level(level_data: dict):
    """
    Parse a level-data dictionary into game objects.

    Expected schema
    ---------------
    {
        "name": str,              # human-readable title (optional)
        "rows": int,              # grid height
        "cols": int,              # grid width
        "grid": [str, ...],       # one string per row (len == cols)
                                  #   '.'  →  empty cell
                                  #   'X'  →  HEAD  of worm X  (uppercase)
                                  #   'x'  →  BODY  of worm X  (lowercase)
        "worms": {                # metadata keyed by uppercase letter
            "A": {
                "color":     str, # key in ANSI_COLORS
                "direction": str  # 'up' | 'down' | 'left' | 'right'
            },
            ...
        }
    }

    Parsing algorithm
    -----------------
    1) Scan every cell of *grid*.  Group (row, col) coordinates by worm
       letter (case-insensitive).  Uppercase marks the **head**; lowercase
       marks **body** segments.

    2) For each worm, collect all coordinates and sort them so the list
       runs from **tail → head** (i.e. segments[-1] == head).  The sort
       key depends on the worm's facing direction:

         direction │ sort key            │ result: head is …
         ──────────┼─────────────────────┼──────────────────
         right     │  ascending  column  │ rightmost cell
         left      │  descending column  │ leftmost  cell
         down      │  ascending  row     │ bottommost cell
         up        │  descending row     │ topmost   cell

    3) Construct Worm objects.

    Returns
    -------
    (rows: int, cols: int, name: str, worms: list[Worm])
    """
    rows      = level_data["rows"]
    cols      = level_data["cols"]
    name      = level_data.get("name", "Unnamed Level")
    grid_strs = level_data["grid"]
    worm_meta = level_data["worms"]

    # ── Step 1: scan the text grid and group coordinates by worm ID ──
    #
    # cell_map example after scanning:
    #   { 'A': {'head': (3, 2), 'body': [(3, 0), (3, 1)]},
    #     'B': {'head': (2, 3), 'body': [(3, 3), (4, 3)]},
    #     'C': {'head': (1, 3), 'body': [(1, 4)]}           }
    cell_map = {}
    for r, row_str in enumerate(grid_strs):
        for c, ch in enumerate(row_str):
            if ch in ('.', ' '):
                continue
            wid = ch.upper()
            entry = cell_map.setdefault(wid, {"head": None, "body": []})
            if ch.isupper():
                if entry["head"] is not None:
                    raise ValueError(
                        f"Worm '{wid}': multiple heads found on the grid.")
                entry["head"] = (r, c)
            else:
                entry["body"].append((r, c))

    # ── Step 2 & 3: sort segments tail→head and build Worm objects ──
    sort_keys = {
        "right": lambda p:  p[1],   # ascending column
        "left":  lambda p: -p[1],   # descending column
        "down":  lambda p:  p[0],   # ascending row
        "up":    lambda p: -p[0],   # descending row
    }
    worms = []
    for wid, meta in worm_meta.items():
        if wid not in cell_map:
            raise ValueError(
                f"Worm '{wid}' declared in metadata but not found on grid.")
        info = cell_map[wid]
        if info["head"] is None:
            raise ValueError(f"Worm '{wid}' has no head (uppercase) on grid.")

        direction = meta["direction"]
        color     = meta.get("color", "white")
        coords    = info["body"] + [info["head"]]
        coords.sort(key=sort_keys[direction])

        worms.append(Worm(wid, coords, direction, color))

    return rows, cols, name, worms


# ═══════════════════════════════════════════════════════════════════════════
#  RENDERING
# ═══════════════════════════════════════════════════════════════════════════

def clear_screen():
    """Clear the terminal (cross-platform)."""
    os.system("cls" if os.name == "nt" else "clear")


def render(rows: int, cols: int, worms: list, title: str = "", moves: int = 0):
    """Print the full game board with ANSI-colored worms."""

    # Build fast lookup: (r, c) → (display_char, ansi_color_code)
    cell_map = {}
    for w in worms:
        ansi = ANSI_COLORS.get(w.color, "")
        for idx, (r, c) in enumerate(w.segments):
            is_head = (idx == len(w.segments) - 1)
            ch = HEAD_ARROWS[w.direction] if is_head else w.worm_id.lower()
            cell_map[(r, c)] = (ch, ansi)

    # ── Header ──
    print()
    print(f"  {ANSI_BOLD}========== WORM ESCAPE =========={ANSI_RESET}")
    if title:
        print(f"  {title}")
    print(f"  Moves: {moves}")
    print()

    # ── Column numbers ──
    #    Aligned to match the cell positions inside the grid row.
    header = "    " + " ".join(f"{c}" for c in range(cols))
    print(f"  {ANSI_DIM}{header}{ANSI_RESET}")

    # ── Top border ──
    inner_width = cols * 2 + 1
    print(f"    +{'-' * inner_width}+")

    # ── Grid rows ──
    for r in range(rows):
        parts = []
        for c in range(cols):
            if (r, c) in cell_map:
                ch, ansi = cell_map[(r, c)]
                parts.append(f"{ansi}{ANSI_BOLD}{ch}{ANSI_RESET}")
            else:
                parts.append(f"{ANSI_DIM}.{ANSI_RESET}")
        row_str = " ".join(parts)
        print(f"  {r} | {row_str} |")

    # ── Bottom border ──
    print(f"    +{'-' * inner_width}+")

    # ── Legend: worms still on the board ──
    print()
    if worms:
        print("  Worms remaining:")
        for w in sorted(worms, key=lambda w: w.worm_id):
            ansi  = ANSI_COLORS.get(w.color, "")
            arrow = HEAD_ARROWS[w.direction]
            print(
                f"    {ansi}{ANSI_BOLD}[{w.worm_id}]{ANSI_RESET}"
                f"  {w.color.capitalize():8s}"
                f"  {arrow} {w.direction:5s}"
                f"  ({len(w.segments)} segments)"
            )
    else:
        print(f"  {ANSI_DIM}  No worms remain — board is clear!{ANSI_RESET}")
    print()


# ═══════════════════════════════════════════════════════════════════════════
#  GAME LOGIC
# ═══════════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════════
#  LEVEL DATA
# ═══════════════════════════════════════════════════════════════════════════
#
# Grid encoding rules:
#   '.'  =  empty cell
#   Uppercase letter  =  the HEAD of that worm
#   Lowercase letter  =  a BODY segment of that worm  (same letter)
#
# The "worms" dictionary supplies each worm's display color and its fixed
# movement direction.  The level parser automatically orders the segments
# so that segments[-1] is the head (leading in the movement direction).
#
# ── Level 1 — "First Steps" ─────────────────────────────────────────────
#
# Intended solution order: C → B → A  (minimum 3 moves)
#
#       Col  0   1   2   3   4   5
#  Row 0   [ .   .   .   .   .   . ]
#  Row 1   [ .   .   .   C   c   . ]   C (green)  faces LEFT  — clear path
#  Row 2   [ .   .   .   B   .   . ]   B (blue)   faces UP    — blocked by C
#  Row 3   [ a   a   A   b   .   . ]   A (red)    faces RIGHT — blocked by B
#  Row 4   [ .   .   .   b   .   . ]
#  Row 5   [ .   .   .   .   .   . ]
#
# Why this forces C → B → A:
#   • A (→) can't move right because B's body sits at (3,3).
#   • B (^) can't move up because C's head sits at (1,3).
#   • C (←) has a clear path to the left edge — extract C first.
#   • After C is gone, B slides up and out.
#   • After B is gone, A slides right and out.
# ─────────────────────────────────────────────────────────────────────────

LEVELS = [
    {
        "name": "Level 1 — First Steps",
        "rows": 6,
        "cols": 6,
        "grid": [
            "......",
            "...Cc.",
            "...B..",
            "aaAb..",
            "...b..",
            "......",
        ],
        "worms": {
            "A": {"color": "red",   "direction": "right"},
            "B": {"color": "blue",  "direction": "up"},
            "C": {"color": "green", "direction": "left"},
        },
    },
]


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN GAME LOOP
# ═══════════════════════════════════════════════════════════════════════════

def enable_windows_ansi():
    """Enable virtual-terminal processing on Windows 10+ for ANSI codes."""
    if os.name == "nt":
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            # ENABLE_PROCESSED_OUTPUT | ENABLE_WRAP_AT_EOL | ENABLE_VIRTUAL_TERMINAL
            kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
        except Exception:
            pass   # graceful fallback — colors may not render


def main():
    enable_windows_ansi()

    # ── Select level (default 1; pass number as CLI arg) ──
    level_idx = 0
    if len(sys.argv) > 1:
        try:
            level_idx = int(sys.argv[1]) - 1
        except ValueError:
            pass
    if not (0 <= level_idx < len(LEVELS)):
        print(f"Level {level_idx + 1} not found.  Available: 1-{len(LEVELS)}")
        sys.exit(1)

    rows, cols, name, worms = load_level(LEVELS[level_idx])
    moves = 0

    # Helper: redraw the whole screen.
    def refresh():
        clear_screen()
        render(rows, cols, worms, title=name, moves=moves)

    refresh()

    # ── Main input loop — runs until every worm is extracted ──
    while worms:
        valid_ids = {w.worm_id for w in worms}
        prompt = (
            f"  Select worm to extract "
            f"[{'/'.join(sorted(valid_ids))}]"
            f"  (q = quit): "
        )
        try:
            choice = input(prompt).strip().upper()
        except (EOFError, KeyboardInterrupt):
            print("\n  Goodbye!")
            sys.exit(0)

        if choice in ("Q", "QUIT", "EXIT"):
            print("  Goodbye!")
            sys.exit(0)

        if choice not in valid_ids:
            input(f"  '{choice}' is not a valid worm on the board.  Press Enter...")
            refresh()
            continue

        # ── Attempt to slide the chosen worm ──
        worm = next(w for w in worms if w.worm_id == choice)
        extracted, steps = attempt_extraction(
            worm, worms, rows, cols, on_step=refresh
        )
        moves += 1

        if extracted:
            worms.remove(worm)
            refresh()
            ansi = ANSI_COLORS.get(worm.color, "")
            print(
                f"  {ansi}{ANSI_BOLD}[{worm.worm_id}]{ANSI_RESET}"
                f"  Extracted!  (slid {steps} cell(s) then exited)"
            )
        else:
            refresh()
            if steps == 0:
                print(f"  [{choice}] is completely blocked — cannot move!")
            else:
                print(
                    f"  [{choice}] slid {steps} cell(s) but is now blocked."
                )

    # ── Victory ──
    bar = "=" * 42
    print(f"\n  {ANSI_BOLD}{bar}")
    print(f"  *  CONGRATULATIONS!  Board cleared!     *")
    print(f"  *  Total moves: {moves:<25}*")
    print(f"  {bar}{ANSI_RESET}\n")


if __name__ == "__main__":
    main()
