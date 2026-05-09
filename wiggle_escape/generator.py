#!/usr/bin/env python3
"""
generator.py — Procedural level generator for Worm Escape.

Uses "Reverse Construction" to guarantee that 100% of generated levels
are solvable.  After generation the level is double-checked by the
dependency-graph solver as a safety net.

Algorithm overview
------------------
  Instead of placing worms randomly and hoping they don't deadlock, we
  build the puzzle *backward* — each worm is placed with a guaranteed
  clear exit, and later worms intentionally grow their bodies into
  earlier worms' exit corridors to create blocking dependencies.

  Placement order (1 … N) → Solve order is REVERSE (N … 1):

    1.  Place worm 1: its exit corridor (head → grid edge) is empty.
        This worm is extracted LAST in gameplay.
    2.  Place worm 2: grow its body through worm 1's corridor.
        Now worm 1 depends on worm 2.
    3.  Place worm 3: grow its body through worm 2's corridor.
    …   and so on.
    N.  Place worm N: nothing is placed after it, so its corridor stays
        clear.  This worm is extracted FIRST in gameplay.

  Why deadlocks are impossible:
    • Worm K's corridor was verified empty at placement time.
    • Only worms K+1…N (placed later) can put body segments into K's
      corridor.
    • Removing worms in reverse order (N, N-1, …, 1) clears each
      corridor before it's needed.
    • Cyclic "A blocks B, B blocks A" is structurally impossible:
      if A was placed before B, B's corridor was checked against A's
      cells — A's body cannot be in B's corridor.

Prompt corrections
------------------
  1. The task description says to place the head "on the edge" facing
     outward.  If the head is literally at the boundary, the worm exits
     in one step and there are ZERO cells in its exit corridor for other
     worms to block.  This implementation places the head 1–N cells
     INWARD from the edge (still facing toward it) to create a
     meaningful, blockable corridor.

  2. Body growth must respect the "head at leading edge" rule enforced
     by load_level().  No body segment may extend further in the head's
     movement direction than the head itself.  The random walk enforces
     this constraint at every step.

Run from the repository root:
    python generator.py
"""

import os
import sys
import random

from worm_escape.constants import (
    ANSI_BOLD,
    ANSI_COLORS,
    ANSI_DIM,
    ANSI_RESET,
    DIR_DELTA,
    HEAD_ARROWS,
)
from worm_escape.level_manager import load_level
from worm_escape.renderer import clear_screen, render
from worm_escape.solver import is_solvable


# ═══════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════

COLOR_CYCLE = ["red", "blue", "green", "yellow", "magenta", "cyan", "white"]


# ═══════════════════════════════════════════════════════════════════════════
#  EXIT CORRIDOR
# ═══════════════════════════════════════════════════════════════════════════

def _exit_corridor(head_r: int, head_c: int, direction: str,
                   rows: int, cols: int) -> set:
    """
    Compute the exit corridor: every cell between the head (exclusive)
    and the grid edge in the head's forward direction.

    These are the cells the worm must slide through to be extracted.
    If another worm's body sits in any of them, extraction is blocked.

    Example
    -------
    head at (3, 2) facing UP in a 6×6 grid →
        corridor = {(2,2), (1,2), (0,2)}
    """
    dr, dc = DIR_DELTA[direction]
    corridor = set()
    r, c = head_r + dr, head_c + dc
    while 0 <= r < rows and 0 <= c < cols:
        corridor.add((r, c))
        r += dr
        c += dc
    return corridor


# ═══════════════════════════════════════════════════════════════════════════
#  HEAD PLACEMENT
# ═══════════════════════════════════════════════════════════════════════════

def _pick_head(rows: int, cols: int, occupied: set,
               min_depth: int = 1, max_depth: int | None = None):
    """
    Pick a random grid edge, a random position along it, and a random
    "depth" (how many cells inward the head sits from the edge).

    The head's direction always faces TOWARD the chosen edge so the worm
    exits by sliding in that direction.

    Constraints checked:
      • Head cell must be unoccupied.
      • The entire exit corridor (head → edge) must be free of existing
        worms.  This guarantees the worm CAN exit once all later-placed
        blockers are removed.

    Returns
    -------
    (head_r, head_c, direction, corridor)  or  None
    """
    # ── Build all (edge_cell, direction) candidates ──
    #   top    edge → worm faces  UP     (exits upward)
    #   bottom edge → worm faces  DOWN   (exits downward)
    #   left   edge → worm faces  LEFT   (exits leftward)
    #   right  edge → worm faces  RIGHT  (exits rightward)
    edges = []
    for c in range(cols):
        edges.append((0, c, "up"))
        edges.append((rows - 1, c, "down"))
    for r in range(rows):
        edges.append((r, 0, "left"))
        edges.append((r, cols - 1, "right"))

    random.shuffle(edges)

    if max_depth is None:
        max_depth = max(rows, cols) - 2

    for edge_r, edge_c, direction in edges:
        # "Inward" = opposite of the worm's facing direction.
        opp_dr = -DIR_DELTA[direction][0]
        opp_dc = -DIR_DELTA[direction][1]

        depths = list(range(min_depth, max_depth + 1))
        random.shuffle(depths)

        for depth in depths:
            hr = edge_r + depth * opp_dr
            hc = edge_c + depth * opp_dc

            if not (0 <= hr < rows and 0 <= hc < cols):
                continue
            if (hr, hc) in occupied:
                continue

            corridor = _exit_corridor(hr, hc, direction, rows, cols)
            if corridor & occupied:
                continue       # corridor blocked by existing worms

            return hr, hc, direction, corridor

    return None     # grid is too crowded for another worm


# ═══════════════════════════════════════════════════════════════════════════
#  BODY GROWTH  (self-avoiding biased random walk)
# ═══════════════════════════════════════════════════════════════════════════

def _grow_body(head_r: int, head_c: int, direction: str,
               occupied: set, target_cells: set,
               min_segs: int, max_segs: int,
               rows: int, cols: int) -> list | None:
    """
    Grow body segments starting behind the head using a self-avoiding
    random walk.

    Walk rules
    ----------
    •  Each new segment must be **4-connected** to the previous one
       (up/down/left/right — no diagonals).
    •  The walk can go straight back, turn 90°, or even reverse,
       forming L-shapes, U-shapes, or S-shapes.
    •  A candidate cell is REJECTED if it:
       – is outside the grid,
       – is already occupied (by this or any other worm),
       – would extend further in the head's movement direction than
         the head itself (violates the "head at leading edge" rule
         enforced by load_level()).
    •  When multiple neighbours are valid, cells that fall in EARLIER
       worms' exit corridors receive 5× the selection weight.  This
       biased walk maximises the chance of creating the blocking
       dependencies that make the puzzle interesting.

    Parameters
    ----------
    target_cells : set
        Union of all earlier worms' exit corridors (minus already-
        occupied cells).  The walk is drawn toward these.

    Returns
    -------
    List of (row, col) body positions ordered **tail → closest-to-head**,
    or None if the walk couldn't produce at least *min_segs* segments.
    """
    num_segs = random.randint(min_segs, max_segs)
    if num_segs == 0:
        return []

    # ── Constraint function: body must NOT go further in the head's
    #    movement direction than the head.
    #      RIGHT → no body column > head_c
    #      LEFT  → no body column < head_c
    #      DOWN  → no body row    > head_r
    #      UP    → no body row    < head_r
    def _valid(r, c):
        if direction == "right" and c > head_c:
            return False
        if direction == "left"  and c < head_c:
            return False
        if direction == "down"  and r > head_r:
            return False
        if direction == "up"    and r < head_r:
            return False
        return True

    ALL_DIRS = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    body = []
    taken = occupied | {(head_r, head_c)}
    cur_r, cur_c = head_r, head_c          # walk starts at the head

    for _ in range(num_segs):
        # Enumerate valid 4-connected neighbours of the current cell.
        candidates = []
        for dr, dc in ALL_DIRS:
            nr, nc = cur_r + dr, cur_c + dc
            if (0 <= nr < rows and 0 <= nc < cols
                    and (nr, nc) not in taken
                    and _valid(nr, nc)):
                candidates.append((nr, nc))

        if not candidates:
            break                              # dead end — keep what we have

        # ── Weighted selection: bias toward earlier worms' corridors ──
        # Cells in target_cells get 5× the probability.  This encourages
        # the body to cross exit paths, creating blocking dependencies
        # that make the puzzle require a specific solve order.
        weighted = []
        for pos in candidates:
            weight = 5 if pos in target_cells else 1
            weighted.extend([pos] * weight)

        nr, nc = random.choice(weighted)
        body.append((nr, nc))
        taken.add((nr, nc))
        cur_r, cur_c = nr, nc

    if len(body) < min_segs:
        return None                            # couldn't grow enough

    # Reverse so the list runs  tail → … → closest-to-head.
    # After appending the head, the full segments list will be:
    #   [tail, ..., body_near_head, head]
    body.reverse()
    return body


# ═══════════════════════════════════════════════════════════════════════════
#  LEVEL GENERATION  (Reverse Construction core loop)
# ═══════════════════════════════════════════════════════════════════════════

def generate_level(rows: int, cols: int, num_worms: int,
                   min_len: int, max_len: int,
                   seed: int | None = None):
    """
    Generate a solvable Worm Escape level.

    For each worm (1 … N):
      1. Pick a head position whose exit corridor is completely free.
      2. Grow a body via biased random walk, preferring cells in earlier
         worms' corridors (to create blocking dependencies).

    The reverse-placement guarantee ensures the solver can always clear
    the board in reverse order.  The is_solvable() check at the end acts
    as a belt-and-suspenders verification.

    Parameters
    ----------
    seed : optional RNG seed for reproducible generation.

    Returns
    -------
    (level_data, placement_order, message)
        level_data      – LEVEL_DATA dict, or None on failure
        placement_order – worm IDs in the order they were placed
        message         – status or error string
    """
    if seed is not None:
        random.seed(seed)

    occupied     = set()       # all cells currently taken
    worms_placed = []          # list of placement records
    corridors    = []          # one set of cells per placed worm

    min_body = max(min_len - 1, 0)     # head counts as 1 segment
    max_body = max_len - 1

    for i in range(num_worms):
        wid = chr(ord('A') + i)        # 'A', 'B', 'C', …

        # ── Collect earlier worms' corridors as walk-bias targets ──
        target_cells = set()
        for corr in corridors:
            target_cells |= corr
        target_cells -= occupied       # no point targeting taken cells

        # Allow deeper heads for the first worm (longer corridor = more
        # cells for later worms to block).  Tighten for later worms to
        # keep the puzzle compact.
        max_depth = max(rows, cols) - 2
        if i > 0:
            max_depth = max(rows, cols) // 2 + 1

        placed = False
        for _attempt in range(300):
            result = _pick_head(rows, cols, occupied,
                                min_depth=1, max_depth=max_depth)
            if result is None:
                break            # no valid head spot exists — grid full

            hr, hc, direction, corridor = result

            body = _grow_body(hr, hc, direction,
                              occupied, target_cells,
                              min_body, max_body,
                              rows, cols)
            if body is None:
                continue        # body too short — try a different head

            # ── Success: record this worm ───────────────────────────
            color    = COLOR_CYCLE[i % len(COLOR_CYCLE)]
            segments = body + [(hr, hc)]      # tail → head order

            worms_placed.append({
                "worm_id":   wid,
                "head":      (hr, hc),
                "direction": direction,
                "body":      list(body),
                "color":     color,
                "segments":  segments,
            })
            occupied.update(segments)
            corridors.append(corridor)
            placed = True
            break

        if not placed:
            if i == 0:
                return None, [], (
                    "Failed to place even the first worm — "
                    "grid too small for the requested worm length."
                )
            break       # accept however many we managed

    if not worms_placed:
        return None, [], "No worms could be placed."

    # ── Build the LEVEL_DATA dictionary ──────────────────────────────
    grid = [['.' for _ in range(cols)] for _ in range(rows)]
    worms_meta = {}

    for wd in worms_placed:
        wid    = wd["worm_id"]
        hr, hc = wd["head"]
        grid[hr][hc] = wid.upper()         # head = uppercase
        for br, bc in wd["body"]:
            grid[br][bc] = wid.lower()      # body = lowercase
        worms_meta[wid] = {
            "color":     wd["color"],
            "direction": wd["direction"],
        }

    level_data = {
        "name":  f"Generated {rows}x{cols} ({len(worms_placed)} worms)",
        "rows":  rows,
        "cols":  cols,
        "grid":  ["".join(row) for row in grid],
        "worms": worms_meta,
    }

    placement_order = [wd["worm_id"] for wd in worms_placed]
    actual_placed   = len(worms_placed)
    if actual_placed < num_worms:
        msg = (
            f"Placed {actual_placed} of {num_worms} worms "
            f"(grid too crowded for more)."
        )
    else:
        msg = "OK"

    return level_data, placement_order, msg


# ═══════════════════════════════════════════════════════════════════════════
#  OUTPUT FORMATTING  (matches editor.py format for consistency)
# ═══════════════════════════════════════════════════════════════════════════

def _format_level_data(ld: dict) -> str:
    """Pretty-print a LEVEL_DATA dict as a copy-pasteable Python literal."""
    lines = []
    lines.append("LEVEL_DATA = {")
    lines.append(f'    "name": "{ld["name"]}",')
    lines.append(f'    "rows": {ld["rows"]},')
    lines.append(f'    "cols": {ld["cols"]},')
    lines.append(f'    "grid": [')
    for row_str in ld["grid"]:
        lines.append(f'        "{row_str}",')
    lines.append(f'    ],')
    lines.append(f'    "worms": {{')
    for wid, meta in ld["worms"].items():
        color     = meta["color"]
        direction = meta["direction"]
        lines.append(
            f'        "{wid}": {{"color": "{color}",'
            f'   "direction": "{direction}"}},'
        )
    lines.append(f'    }},')
    lines.append(f'}}')
    return "\n".join(lines)


def _register_in_init(filename: str):
    """
    Append an import + LEVELS entry for the new level file in
    worm_escape/levels/__init__.py.  Idempotent — safe to call twice
    for the same filename.
    """
    module_name = filename.replace(".py", "")
    alias       = "_" + module_name.upper()

    init_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "worm_escape", "levels", "__init__.py",
    )

    with open(init_path, "r", encoding="utf-8") as f:
        content = f.read()

    import_line = (
        f"from worm_escape.levels.{module_name} "
        f"import LEVEL_DATA as {alias}"
    )
    if import_line in content:
        return                                # already registered

    lines      = content.splitlines()
    insert_idx = None
    levels_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith("from worm_escape.levels."):
            insert_idx = i
        if line.strip().startswith("LEVELS"):
            levels_idx = i

    if insert_idx is None or levels_idx is None:
        content += f"\n{import_line}\n"
        content = content.replace("]\n", f"    {alias},\n]\n", 1)
    else:
        lines.insert(insert_idx + 1, import_line)
        for i in range(len(lines) - 1, -1, -1):
            if lines[i].strip() == "]":
                lines.insert(i, f"    {alias},")
                break
        content = "\n".join(lines) + "\n"

    with open(init_path, "w", encoding="utf-8") as f:
        f.write(content)


# ═══════════════════════════════════════════════════════════════════════════
#  SAVE
# ═══════════════════════════════════════════════════════════════════════════

def _next_level_number() -> int:
    """Find the next available level_XX number in the levels directory."""
    levels_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "worm_escape", "levels",
    )
    existing = set()
    if os.path.isdir(levels_dir):
        for fname in os.listdir(levels_dir):
            if fname.startswith("level_") and fname.endswith(".py"):
                try:
                    num = int(fname[6:-3])
                    existing.add(num)
                except ValueError:
                    pass
    n = 1
    while n in existing:
        n += 1
    return n


def _count_level_files() -> int:
    """Count how many level_XX.py files exist in the levels directory."""
    levels_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "worm_escape", "levels",
    )
    if not os.path.isdir(levels_dir):
        return 0
    return sum(
        1 for f in os.listdir(levels_dir)
        if f.startswith("level_") and f.endswith(".py")
    )


def _save_level(level_data: dict, formatted: str, seed: int,
                placement_order: list):
    """Save the generated level to a .py file and register in __init__."""
    next_num = _next_level_number()
    default  = f"level_{next_num:02d}"

    filename = input(f"  Filename [{default}]: ").strip()
    if not filename:
        filename = default
    if not filename.endswith(".py"):
        filename += ".py"

    filepath = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "worm_escape", "levels", filename,
    )
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # Write level file with metadata header.
    name  = level_data["name"]
    solve = " -> ".join(reversed(placement_order))
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f'"""\n')
        f.write(f'{filename} — {name}\n')
        f.write(f'\n')
        f.write(f'Auto-generated by generator.py  (seed {seed})\n')
        f.write(f'Solve order: {solve}\n')
        f.write(f'"""\n\n')
        f.write(formatted)
        f.write("\n")

    _register_in_init(filename)

    level_pos = _count_level_files()
    print(f"\n  Saved to {filepath}")
    print(f"  __init__.py updated.")
    print(f"  Play it:  python main.py {level_pos}")


# ═══════════════════════════════════════════════════════════════════════════
#  WINDOWS ANSI
# ═══════════════════════════════════════════════════════════════════════════

def _enable_windows_ansi():
    if os.name == "nt":
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    _enable_windows_ansi()

    print()
    print(f"  {ANSI_BOLD}═══════ WORM ESCAPE — LEVEL GENERATOR ═══════{ANSI_RESET}")
    print(f"  {ANSI_DIM}Reverse Construction — every level is guaranteed solvable.{ANSI_RESET}")
    print()

    # ── Gather parameters interactively ──────────────────────────────

    def ask_int(prompt, lo, hi, default=None):
        while True:
            dflt = f" [{default}]" if default is not None else ""
            raw  = input(f"  {prompt} ({lo}-{hi}){dflt}: ").strip()
            if not raw and default is not None:
                return default
            try:
                val = int(raw)
                if lo <= val <= hi:
                    return val
            except ValueError:
                pass
            print(f"    Enter an integer between {lo} and {hi}.")

    g_cols    = ask_int("Grid width  (columns)", 3, 20, default=6)
    g_rows    = ask_int("Grid height (rows)",    3, 20, default=6)
    n_worms   = ask_int("Number of worms",       1, 26, default=3)
    min_len   = ask_int("Minimum worm length",   1, max(g_rows, g_cols), default=2)
    max_len   = ask_int("Maximum worm length",   min_len, max(g_rows, g_cols), default=4)

    print()

    # ── Generate / preview / save loop ───────────────────────────────

    seed = random.randrange(2**32)

    while True:
        print(f"  {ANSI_DIM}Seed: {seed}{ANSI_RESET}")
        ld, order, msg = generate_level(g_rows, g_cols, n_worms,
                                        min_len, max_len, seed)

        if ld is None:
            print(
                f"\n  {ANSI_COLORS['red']}{ANSI_BOLD}"
                f"Generation failed: {msg}"
                f"{ANSI_RESET}"
            )
            retry = input("  Try again with a different seed? (y/n): ").strip().lower()
            if retry in ("y", "yes", ""):
                seed = random.randrange(2**32)
                continue
            else:
                print("  Goodbye!")
                sys.exit(0)

        # ── Status ───────────────────────────────────────────────────
        if msg != "OK":
            print(f"  {ANSI_COLORS['yellow']}{msg}{ANSI_RESET}")

        solve_order = list(reversed(order))
        print(f"  Placement order : {' -> '.join(order)}")
        print(f"  Expected solve  : {' -> '.join(solve_order)}")
        print()

        # ── Live preview using the game renderer ─────────────────────
        try:
            p_rows, p_cols, p_name, p_worms = load_level(ld)
            render(p_rows, p_cols, p_worms, title=ld["name"])
        except Exception as e:
            print(f"  {ANSI_COLORS['red']}Preview error: {e}{ANSI_RESET}")

        # ── Solvability verification (belt-and-suspenders) ───────────
        try:
            solvable, report = is_solvable(ld)
            report_str = "\n    ".join(report)
            if solvable:
                print(
                    f"  {ANSI_COLORS['green']}{ANSI_BOLD}"
                    f"Solver: SOLVABLE"
                    f"{ANSI_RESET}"
                )
                print(f"    {report_str}")
            else:
                print(
                    f"  {ANSI_COLORS['red']}{ANSI_BOLD}"
                    f"Solver: UNSOLVABLE (unexpected — possible bug)"
                    f"{ANSI_RESET}"
                )
                print(f"    {report_str}")
        except Exception as e:
            print(f"  {ANSI_COLORS['red']}Solver error: {e}{ANSI_RESET}")

        # ── Show the LEVEL_DATA dict ─────────────────────────────────
        formatted = _format_level_data(ld)
        print()
        print(f"  {'─' * 55}")
        print(formatted)
        print(f"  {'─' * 55}")
        print()

        # ── Save / regenerate / quit ─────────────────────────────────
        choice = input(
            "  [S]ave  /  [R]egenerate  /  [Q]uit : "
        ).strip().upper()

        if choice in ("S", "SAVE", ""):
            _save_level(ld, formatted, seed, order)
            break
        elif choice in ("R", "REGEN", "REGENERATE"):
            seed = random.randrange(2**32)
            print()
            continue
        elif choice in ("Q", "QUIT", "EXIT"):
            print("  Goodbye!")
            sys.exit(0)
        else:
            print(f"  Unknown option '{choice}'. Regenerating...")
            seed = random.randrange(2**32)
            continue

    print()


if __name__ == "__main__":
    main()
