"""
level_01.py — Level 1: "First Steps"

Grid encoding rules:
  '.'  =  empty cell
  Uppercase letter  =  the HEAD of that worm
  Lowercase letter  =  a BODY segment of that worm  (same letter)

The "worms" dictionary supplies each worm's display color and its fixed
movement direction.  The level parser automatically orders the segments
so that segments[-1] is the head (leading in the movement direction).

Intended solution order: C → B → A  (minimum 3 moves)

      Col  0   1   2   3   4   5
 Row 0   [ .   .   .   .   .   . ]
 Row 1   [ .   .   .   C   c   . ]   C (green)  faces LEFT  — clear path
 Row 2   [ .   .   .   B   .   . ]   B (blue)   faces UP    — blocked by C
 Row 3   [ a   a   A   b   .   . ]   A (red)    faces RIGHT — blocked by B
 Row 4   [ .   .   .   b   .   . ]
 Row 5   [ .   .   .   .   .   . ]

Why this forces C → B → A:
  • A (→) can't move right because B's body sits at (3,3).
  • B (^) can't move up because C's head sits at (1,3).
  • C (←) has a clear path to the left edge — extract C first.
  • After C is gone, B slides up and out.
  • After B is gone, A slides right and out.
"""

LEVEL_DATA = {
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
}
