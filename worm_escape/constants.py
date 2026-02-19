"""
constants.py — Shared constants for Worm Escape.

All ANSI escape codes, directional data, grid‐character mappings,
and timing values live here so every other module can import them
from a single authoritative source.
"""

# ═══════════════════════════════════════════════════════════════════════════
#  ANSI ESCAPE CODES
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

# ═══════════════════════════════════════════════════════════════════════════
#  DIRECTIONAL DATA
# ═══════════════════════════════════════════════════════════════════════════

# Directional arrows used to render each worm's head.
HEAD_ARROWS = {"up": "^", "down": "v", "left": "<", "right": ">"}

# Row / column deltas for each compass direction.
DIR_DELTA = {
    "up":    (-1,  0),
    "down":  ( 1,  0),
    "left":  ( 0, -1),
    "right": ( 0,  1),
}

# ═══════════════════════════════════════════════════════════════════════════
#  TIMING
# ═══════════════════════════════════════════════════════════════════════════

# Seconds to pause between animation frames when a worm slides.
ANIM_DELAY = 0.18
