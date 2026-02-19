"""
level_manager.py — Data-driven level loader for Worm Escape.

The load_level() function translates a plain-dict level description
into the grid dimensions and a list of Worm objects.  This format is
intentionally simple so that a separate Level Editor tool can export
directly to the same schema.
"""

from worm_escape.entities import Worm


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

        # Validate: head must be at the leading edge for its direction.
        # e.g. a LEFT-facing worm's head must be its leftmost segment.
        head_r, head_c = info["head"]
        for br, bc in info["body"]:
            if (direction == "left"  and bc < head_c or
                direction == "right" and bc > head_c or
                direction == "up"    and br < head_r or
                direction == "down"  and br > head_r):
                raise ValueError(
                    f"Worm '{wid}': head at ({head_r},{head_c}) faces "
                    f"{direction.upper()} but body at ({br},{bc}) is "
                    f"further in that direction.  "
                    f"The head must be at the leading edge of the worm.")

        coords    = info["body"] + [info["head"]]
        coords.sort(key=sort_keys[direction])

        worms.append(Worm(wid, coords, direction, color))

    return rows, cols, name, worms
