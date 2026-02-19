#!/usr/bin/env python3
"""
editor.py — Interactive CLI Level Editor for Worm Escape.

Run from the repository root:
    python editor.py

Workflow
--------
  1. Enter grid dimensions (rows x cols).
  2. Move a cursor with WASD.  Place worm heads and body segments.
  3. When done, press S to export a copy-pasteable LEVEL_DATA dict
     (or save it directly to a .py file in worm_escape/levels/).

Commands  (type a letter then press Enter)
------------------------------------------
  W / A / S_move / D  – move cursor Up / Left / Down / Right
                         (shown as 'I/J/K/L' to avoid clash with Save)
  N        – start a New Worm  (auto-assigns next letter: A, B, C …)
  H        – place the current worm's HEAD at the cursor
  B        – place a BODY segment for the current worm at the cursor
  X        – erase whatever is at the cursor
  U        – Undo the last placement / erase action
  C        – Change the current worm's color (cycles through palette)
  T        – Test — run the level through load_level() to validate it
  S        – Save / export the level
  Q        – Quit without saving

The editor re-renders the full grid after every command so you always
have a live preview of the level you're building.
"""

import os
import sys
import textwrap

# ── Import shared constants from the game package ──────────────────────
from worm_escape.constants import (
    ANSI_BOLD,
    ANSI_COLORS,
    ANSI_DIM,
    ANSI_RESET,
    HEAD_ARROWS,
)

# ═══════════════════════════════════════════════════════════════════════════
#  COLOUR PALETTE  (cycles through these when assigning worms)
# ═══════════════════════════════════════════════════════════════════════════

COLOR_CYCLE = ["red", "blue", "green", "yellow", "magenta", "cyan", "white"]

# ANSI code for the cursor highlight (inverted / reverse video).
ANSI_CURSOR_BG = "\033[7m"   # reverse video

# Direction shorthand mapping used when the user picks a head direction.
DIR_MAP = {"U": "up", "D": "down", "L": "left", "R": "right"}


# ═══════════════════════════════════════════════════════════════════════════
#  EDITOR STATE
# ═══════════════════════════════════════════════════════════════════════════

class EditorState:
    """
    Holds the entire mutable state of the level being edited.

    cells : dict[(r,c)] → {"worm_id": str, "kind": "head"|"body"}
        Every occupied cell on the grid.

    worm_meta : dict[str] → {"color": str, "direction": str|None}
        Metadata for each worm (keyed by uppercase letter).

    cur_worm : str or None
        The worm ID currently being edited.

    cursor : (int, int)
        Current (row, col) of the editing cursor.

    undo_stack : list
        Snapshots of (cells_copy, worm_meta_copy) for undo.
    """

    def __init__(self, rows: int, cols: int):
        self.rows       = rows
        self.cols       = cols
        self.cells      = {}          # (r,c) → {"worm_id", "kind"}
        self.worm_meta  = {}          # "A" → {"color", "direction"}
        self.cur_worm   = None
        self.cursor     = (0, 0)
        self.undo_stack = []
        self._next_id   = 0           # index into A-Z

    # ── helpers ──────────────────────────────────────────────────────

    def _push_undo(self):
        """Snapshot current cells + meta for undo."""
        import copy
        self.undo_stack.append(
            (copy.deepcopy(self.cells), copy.deepcopy(self.worm_meta))
        )

    def undo(self) -> str:
        if not self.undo_stack:
            return "Nothing to undo."
        self.cells, self.worm_meta = self.undo_stack.pop()
        return "Undone."

    def next_worm_id(self) -> str:
        """Return the next available uppercase letter (A, B, C …)."""
        wid = chr(ord("A") + self._next_id)
        self._next_id += 1
        return wid

    def occupied(self, r: int, c: int) -> bool:
        return (r, c) in self.cells

    def move_cursor(self, dr: int, dc: int) -> str:
        r, c = self.cursor
        nr, nc = r + dr, c + dc
        if 0 <= nr < self.rows and 0 <= nc < self.cols:
            self.cursor = (nr, nc)
            return ""
        return "Cursor at grid edge."

    # ── placement ────────────────────────────────────────────────────

    def new_worm(self) -> str:
        """Start editing a brand-new worm."""
        if self._next_id >= 26:
            return "Maximum 26 worms (A-Z) reached."
        wid = self.next_worm_id()
        color = COLOR_CYCLE[(self._next_id - 1) % len(COLOR_CYCLE)]
        self.worm_meta[wid] = {"color": color, "direction": None}
        self.cur_worm = wid
        ansi = ANSI_COLORS.get(color, "")
        return (
            f"New worm {ansi}{ANSI_BOLD}[{wid}]{ANSI_RESET} "
            f"({color}).  Place its Head first (H)."
        )

    def place_head(self, direction: str) -> str:
        """Place the current worm's head at the cursor."""
        if self.cur_worm is None:
            return "No active worm. Press N to create one first."
        wid = self.cur_worm
        r, c = self.cursor

        # Check if this worm already has a head.
        for pos, info in self.cells.items():
            if info["worm_id"] == wid and info["kind"] == "head":
                return (
                    f"Worm [{wid}] already has a head at "
                    f"({pos[0]},{pos[1]}).  Erase it first (X) to move it."
                )

        if self.occupied(r, c):
            existing = self.cells[(r, c)]
            return (
                f"Cell ({r},{c}) already occupied by "
                f"worm [{existing['worm_id']}].  Erase first (X)."
            )

        self._push_undo()
        self.cells[(r, c)] = {"worm_id": wid, "kind": "head"}
        self.worm_meta[wid]["direction"] = direction
        arrow = HEAD_ARROWS[direction]
        return f"Placed [{wid}] head {arrow} ({direction}) at ({r},{c})."

    def place_body(self) -> str:
        """Place a body segment for the current worm at the cursor."""
        if self.cur_worm is None:
            return "No active worm. Press N to create one first."
        wid = self.cur_worm

        # Must have a head already.
        has_head = any(
            info["worm_id"] == wid and info["kind"] == "head"
            for info in self.cells.values()
        )
        if not has_head:
            return f"Worm [{wid}] has no head yet.  Place one first (H)."

        r, c = self.cursor
        if self.occupied(r, c):
            existing = self.cells[(r, c)]
            return (
                f"Cell ({r},{c}) already occupied by "
                f"worm [{existing['worm_id']}].  Erase first (X)."
            )

        # Adjacency check: segment must touch an existing segment of
        # the same worm (4-connected).
        own_cells = {
            pos for pos, info in self.cells.items()
            if info["worm_id"] == wid
        }
        neighbors = {(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)}
        if not neighbors & own_cells:
            return (
                f"Body must be adjacent to an existing [{wid}] segment.  "
                f"Move cursor next to one."
            )

        self._push_undo()
        self.cells[(r, c)] = {"worm_id": wid, "kind": "body"}
        return f"Placed [{wid}] body at ({r},{c})."

    def erase(self) -> str:
        """Remove whatever is at the cursor."""
        r, c = self.cursor
        if not self.occupied(r, c):
            return "Cell is already empty."
        self._push_undo()
        info = self.cells.pop((r, c))
        return f"Erased [{info['worm_id']}] {info['kind']} from ({r},{c})."

    def cycle_color(self) -> str:
        """Cycle the active worm's color through the palette."""
        if self.cur_worm is None:
            return "No active worm."
        meta = self.worm_meta[self.cur_worm]
        idx = COLOR_CYCLE.index(meta["color"]) if meta["color"] in COLOR_CYCLE else -1
        meta["color"] = COLOR_CYCLE[(idx + 1) % len(COLOR_CYCLE)]
        ansi = ANSI_COLORS.get(meta["color"], "")
        return (
            f"Worm [{self.cur_worm}] color → "
            f"{ansi}{ANSI_BOLD}{meta['color']}{ANSI_RESET}"
        )

    def select_worm(self, wid: str) -> str:
        """Switch the active worm to an existing one."""
        if wid not in self.worm_meta:
            return f"Worm [{wid}] does not exist."
        self.cur_worm = wid
        ansi = ANSI_COLORS.get(self.worm_meta[wid]["color"], "")
        return f"Switched to worm {ansi}{ANSI_BOLD}[{wid}]{ANSI_RESET}."


# ═══════════════════════════════════════════════════════════════════════════
#  RENDERING  (editor-specific — adds cursor highlight)
# ═══════════════════════════════════════════════════════════════════════════

def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


def render_editor(state: EditorState, message: str = ""):
    """Draw the editor grid with cursor and status bar."""

    clear_screen()

    rows, cols = state.rows, state.cols
    cr, cc = state.cursor

    # Build cell lookup: (r,c) → (char, ansi_color)
    cell_vis = {}
    for (r, c), info in state.cells.items():
        wid   = info["worm_id"]
        meta  = state.worm_meta[wid]
        ansi  = ANSI_COLORS.get(meta["color"], "")
        if info["kind"] == "head":
            d = meta.get("direction")
            ch = HEAD_ARROWS.get(d, "?") if d else "?"
        else:
            ch = wid.lower()
        cell_vis[(r, c)] = (ch, ansi)

    # ── Header ──
    print()
    print(f"  {ANSI_BOLD}═══════ WORM ESCAPE — LEVEL EDITOR ═══════{ANSI_RESET}")
    worm_label = "None"
    if state.cur_worm:
        wm = state.worm_meta[state.cur_worm]
        ansi = ANSI_COLORS.get(wm["color"], "")
        d = wm.get("direction") or "?"
        worm_label = f"{ansi}{ANSI_BOLD}[{state.cur_worm}]{ANSI_RESET} {wm['color']} {d}"
    print(f"  Grid: {rows}x{cols}   Cursor: ({cr},{cc})   Active worm: {worm_label}")
    print()

    # ── Column numbers ──
    header = "    " + " ".join(f"{c}" for c in range(cols))
    print(f"  {ANSI_DIM}{header}{ANSI_RESET}")

    # ── Top border ──
    inner_width = cols * 2 + 1
    print(f"    +{'-' * inner_width}+")

    # ── Grid rows ──
    for r in range(rows):
        parts = []
        for c in range(cols):
            is_cursor = (r == cr and c == cc)

            if (r, c) in cell_vis:
                ch, ansi = cell_vis[(r, c)]
                if is_cursor:
                    parts.append(f"{ANSI_CURSOR_BG}{ansi}{ANSI_BOLD}{ch}{ANSI_RESET}")
                else:
                    parts.append(f"{ansi}{ANSI_BOLD}{ch}{ANSI_RESET}")
            else:
                if is_cursor:
                    parts.append(f"{ANSI_CURSOR_BG}{ANSI_BOLD}+{ANSI_RESET}")
                else:
                    parts.append(f"{ANSI_DIM}.{ANSI_RESET}")
        row_str = " ".join(parts)
        print(f"  {r} | {row_str} |")

    # ── Bottom border ──
    print(f"    +{'-' * inner_width}+")

    # ── Worm list ──
    print()
    if state.worm_meta:
        print("  Worms defined:")
        for wid in sorted(state.worm_meta):
            meta  = state.worm_meta[wid]
            ansi  = ANSI_COLORS.get(meta["color"], "")
            d     = meta.get("direction") or "—"
            arrow = HEAD_ARROWS.get(d, "—")
            segs  = sum(1 for info in state.cells.values() if info["worm_id"] == wid)
            marker = " ◄" if wid == state.cur_worm else ""
            print(
                f"    {ansi}{ANSI_BOLD}[{wid}]{ANSI_RESET}"
                f"  {meta['color']:8s}  {arrow} {d:5s}"
                f"  ({segs} segment{'s' if segs != 1 else ''})"
                f"{marker}"
            )
    else:
        print(f"  {ANSI_DIM}  No worms yet.  Press N to create one.{ANSI_RESET}")
    print()

    # ── Command help ──
    print(f"  {ANSI_DIM}Commands:  I/J/K/L=move cursor   N=new worm   H=head   B=body{ANSI_RESET}")
    print(f"  {ANSI_DIM}          X=erase   C=cycle color   E=select existing worm{ANSI_RESET}")
    print(f"  {ANSI_DIM}          U=undo    T=test/validate   S=save/export   Q=quit{ANSI_RESET}")

    # ── Message line ──
    if message:
        print(f"\n  >> {message}")
    print()


# ═══════════════════════════════════════════════════════════════════════════
#  VALIDATION
# ═══════════════════════════════════════════════════════════════════════════

def validate(state: EditorState) -> list[str]:
    """
    Return a list of error strings.  Empty list = level is valid.
    """
    errors = []

    if not state.worm_meta:
        errors.append("No worms defined.")
        return errors

    for wid, meta in sorted(state.worm_meta.items()):
        # Must have exactly one head.
        heads = [
            pos for pos, info in state.cells.items()
            if info["worm_id"] == wid and info["kind"] == "head"
        ]
        bodies = [
            pos for pos, info in state.cells.items()
            if info["worm_id"] == wid and info["kind"] == "body"
        ]
        if len(heads) == 0:
            errors.append(f"[{wid}] has no head.")
        elif len(heads) > 1:
            errors.append(f"[{wid}] has multiple heads: {heads}.")
        if not meta.get("direction"):
            errors.append(f"[{wid}] has no direction set.")
        if len(heads) + len(bodies) < 1:
            errors.append(f"[{wid}] has no segments at all.")

        # Head must be at the leading edge for its direction.
        # e.g. a worm facing LEFT must have its head as the leftmost segment.
        direction = meta.get("direction")
        if direction and len(heads) == 1 and bodies:
            head_r, head_c = heads[0]
            for br, bc in bodies:
                if direction == "left"  and bc < head_c:
                    errors.append(
                        f"[{wid}] head at ({head_r},{head_c}) faces LEFT "
                        f"but body at ({br},{bc}) is further left.  "
                        f"Head must be at the leading edge.")
                    break
                if direction == "right" and bc > head_c:
                    errors.append(
                        f"[{wid}] head at ({head_r},{head_c}) faces RIGHT "
                        f"but body at ({br},{bc}) is further right.  "
                        f"Head must be at the leading edge.")
                    break
                if direction == "up"    and br < head_r:
                    errors.append(
                        f"[{wid}] head at ({head_r},{head_c}) faces UP "
                        f"but body at ({br},{bc}) is further up.  "
                        f"Head must be at the leading edge.")
                    break
                if direction == "down"  and br > head_r:
                    errors.append(
                        f"[{wid}] head at ({head_r},{head_c}) faces DOWN "
                        f"but body at ({br},{bc}) is further down.  "
                        f"Head must be at the leading edge.")
                    break

        # Connectivity: all segments of this worm should be 4-connected.
        all_segs = set(heads + bodies)
        if all_segs:
            visited = set()
            stack = [next(iter(all_segs))]
            while stack:
                p = stack.pop()
                if p in visited:
                    continue
                visited.add(p)
                r, c = p
                for nr, nc in [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]:
                    if (nr, nc) in all_segs and (nr, nc) not in visited:
                        stack.append((nr, nc))
            if visited != all_segs:
                errors.append(
                    f"[{wid}] segments are not all connected.  "
                    f"Disconnected: {all_segs - visited}."
                )

    # Try a full load_level round-trip to be sure.
    if not errors:
        try:
            from worm_escape.level_manager import load_level
            ld = _build_level_data(state, "validation_test")
            load_level(ld)
        except Exception as e:
            errors.append(f"load_level() failed: {e}")

    return errors


# ═══════════════════════════════════════════════════════════════════════════
#  EXPORT
# ═══════════════════════════════════════════════════════════════════════════

def _build_level_data(state: EditorState, name: str = "Untitled") -> dict:
    """Convert editor state into the LEVEL_DATA dict format."""
    rows, cols = state.rows, state.cols

    # Build grid strings.
    grid = []
    for r in range(rows):
        row_chars = []
        for c in range(cols):
            if (r, c) in state.cells:
                info = state.cells[(r, c)]
                wid  = info["worm_id"]
                if info["kind"] == "head":
                    row_chars.append(wid.upper())
                else:
                    row_chars.append(wid.lower())
            else:
                row_chars.append(".")
        grid.append("".join(row_chars))

    # Build worms metadata.
    worms = {}
    for wid in sorted(state.worm_meta):
        meta = state.worm_meta[wid]
        # Only include worms that have at least one cell on the grid.
        has_cells = any(info["worm_id"] == wid for info in state.cells.values())
        if has_cells:
            worms[wid] = {
                "color":     meta["color"],
                "direction": meta["direction"],
            }

    return {
        "name": name,
        "rows": rows,
        "cols": cols,
        "grid": grid,
        "worms": worms,
    }


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
        color = meta["color"]
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
    Update worm_escape/levels/__init__.py to import the newly saved level
    and append it to the LEVELS list.

    Strategy: read the existing __init__.py, parse out imports and the
    LEVELS list, then rewrite the file with the new entry appended.
    This is idempotent — if the module is already imported, it's skipped.
    """
    module_name = filename.replace(".py", "")         # e.g. "level_02"
    alias       = "_" + module_name.upper()            # e.g. "_LEVEL_02"

    init_path = os.path.join(
        os.path.dirname(__file__),
        "worm_escape", "levels", "__init__.py",
    )

    with open(init_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Skip if this module is already registered.
    import_line = f"from worm_escape.levels.{module_name} import LEVEL_DATA as {alias}"
    if import_line in content:
        return

    # ── Append import line after the last existing import ──
    # Find the position right before the LEVELS = [ line.
    lines = content.splitlines()
    insert_idx = None
    levels_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith("from worm_escape.levels."):
            insert_idx = i   # track last import line
        if line.strip().startswith("LEVELS"):
            levels_idx = i

    if insert_idx is None or levels_idx is None:
        # Fallback: just append at the end (shouldn't normally happen).
        content += f"\n{import_line}\n"
        content = content.replace("]\n", f"    {alias},\n]\n", 1)
    else:
        # Insert the new import after the last existing import.
        lines.insert(insert_idx + 1, import_line)

        # Now find the closing ']' of the LEVELS list and insert before it.
        # Re-scan because indices shifted by 1.
        for i in range(len(lines) - 1, -1, -1):
            if lines[i].strip() == "]":
                lines.insert(i, f"    {alias},")
                break

        content = "\n".join(lines) + "\n"

    with open(init_path, "w", encoding="utf-8") as f:
        f.write(content)


def export_level(state: EditorState):
    """
    Validate, then either print the level dict to console or save to file.
    Returns a status message string.
    """
    errors = validate(state)
    if errors:
        return "Cannot export — fix these errors first:\n    " + "\n    ".join(errors)

    name = input("  Level name (e.g. 'Level 2 — Crossroads'): ").strip()
    if not name:
        name = "Untitled"

    ld = _build_level_data(state, name)
    formatted = _format_level_data(ld)

    print(f"\n{'─' * 60}")
    print(formatted)
    print(f"{'─' * 60}\n")

    save_choice = input("  Save to file? (y/n): ").strip().lower()
    if save_choice in ("y", "yes"):
        filename = input("  Filename (e.g. level_02): ").strip()
        if not filename:
            filename = "level_new"
        if not filename.endswith(".py"):
            filename += ".py"

        filepath = os.path.join(
            os.path.dirname(__file__),
            "worm_escape", "levels", filename,
        )
        # Ensure directory exists.
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f'"""\n{filename} — {name}\n"""\n\n')
            f.write(formatted)
            f.write("\n")

        # ── Auto-update __init__.py so the new level is importable ──
        _register_in_init(filename)

        return f"Saved to {filepath}\n    __init__.py updated — level is now accessible from main.py."
    else:
        return "Level printed above (not saved to file).  Copy-paste as needed."


# ═══════════════════════════════════════════════════════════════════════════
#  WINDOWS ANSI SUPPORT
# ═══════════════════════════════════════════════════════════════════════════

def enable_windows_ansi():
    if os.name == "nt":
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN LOOP
# ═══════════════════════════════════════════════════════════════════════════

def main():
    enable_windows_ansi()

    print()
    print(f"  {ANSI_BOLD}═══════ WORM ESCAPE — LEVEL EDITOR ═══════{ANSI_RESET}")
    print()

    # ── Grid size ──
    while True:
        size_str = input("  Grid size (rows cols, e.g. '6 6'): ").strip()
        parts = size_str.replace(",", " ").replace("x", " ").split()
        if len(parts) == 2:
            try:
                rows, cols = int(parts[0]), int(parts[1])
                if 2 <= rows <= 20 and 2 <= cols <= 20:
                    break
            except ValueError:
                pass
        print("  Enter two integers between 2 and 20.")

    state = EditorState(rows, cols)
    msg = "Grid created.  Press N to start your first worm."
    render_editor(state, msg)

    # ── Command loop ──
    while True:
        try:
            cmd = input("  > ").strip().upper()
        except (EOFError, KeyboardInterrupt):
            print("\n  Goodbye!")
            sys.exit(0)

        if not cmd:
            render_editor(state)
            continue

        # ── Cursor movement (IJKL) ──────────────────────────────────
        if cmd == "I":
            msg = state.move_cursor(-1, 0)
        elif cmd == "K":
            msg = state.move_cursor(1, 0)
        elif cmd == "J":
            msg = state.move_cursor(0, -1)
        elif cmd == "L":
            msg = state.move_cursor(0, 1)

        # ── Worm creation ───────────────────────────────────────────
        elif cmd == "N":
            msg = state.new_worm()

        # ── Head placement ──────────────────────────────────────────
        elif cmd == "H":
            if state.cur_worm is None:
                msg = "No active worm.  Press N first."
            else:
                render_editor(state, "Direction for head?  U=up  D=down  L=left  R=right")
                d_input = input("  Direction> ").strip().upper()
                if d_input in DIR_MAP:
                    msg = state.place_head(DIR_MAP[d_input])
                else:
                    msg = f"Invalid direction '{d_input}'.  Use U/D/L/R."

        # ── Body placement ──────────────────────────────────────────
        elif cmd == "B":
            msg = state.place_body()

        # ── Erase ──────────────────────────────────────────────────
        elif cmd == "X":
            msg = state.erase()

        # ── Undo ───────────────────────────────────────────────────
        elif cmd == "U":
            msg = state.undo()

        # ── Cycle color ────────────────────────────────────────────
        elif cmd == "C":
            msg = state.cycle_color()

        # ── Select existing worm ───────────────────────────────────
        elif cmd == "E":
            if not state.worm_meta:
                msg = "No worms to select.  Press N to create one."
            else:
                ids = "/".join(sorted(state.worm_meta))
                render_editor(state, f"Switch to which worm? [{ids}]")
                w_input = input("  Worm ID> ").strip().upper()
                msg = state.select_worm(w_input)

        # ── Test / validate ────────────────────────────────────────
        elif cmd == "T":
            errors = validate(state)
            if errors:
                msg = "Validation FAILED:\n    " + "\n    ".join(errors)
            else:
                msg = "Validation PASSED — level is valid!"

        # ── Save / export ──────────────────────────────────────────
        elif cmd == "S":
            msg = export_level(state)

        # ── Quit ───────────────────────────────────────────────────
        elif cmd == "Q":
            confirm = input("  Quit without saving? (y/n): ").strip().lower()
            if confirm in ("y", "yes"):
                print("  Goodbye!")
                sys.exit(0)
            msg = "Cancelled."

        # ── Unknown ────────────────────────────────────────────────
        else:
            msg = f"Unknown command '{cmd}'.  See help below."

        render_editor(state, msg)


if __name__ == "__main__":
    main()
