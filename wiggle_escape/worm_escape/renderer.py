"""
renderer.py — Terminal rendering for Worm Escape.

Handles screen clearing and drawing the ANSI-colored game board.
"""

import os

from worm_escape.constants import (
    ANSI_BOLD, ANSI_COLORS, ANSI_DIM, ANSI_RESET, HEAD_ARROWS,
)


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
