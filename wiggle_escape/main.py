#!/usr/bin/env python3
"""
main.py — Entry point for Worm Escape.

Run from the repository root:
    python main.py          # plays Level 1
    python main.py 2        # plays Level 2 (when available)
"""

import os
import sys

from worm_escape.constants import ANSI_BOLD, ANSI_COLORS, ANSI_RESET
from worm_escape.engine import attempt_extraction
from worm_escape.level_manager import load_level
from worm_escape.levels import LEVELS
from worm_escape.renderer import clear_screen, render


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
