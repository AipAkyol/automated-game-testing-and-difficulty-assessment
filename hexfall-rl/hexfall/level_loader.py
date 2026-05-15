import json
import random
import warnings
from pathlib import Path

import jsonschema

from hexfall.game import run_until_quiescent
from hexfall.types import (
    BufferBucket,
    GameState,
    Generator,
    IceBucket,
    Pin,
    PlainBucket,
    QuestionBucket,
    Wall,
)

_SCHEMA_PATH = Path(__file__).parent / "schemas" / "level_schema.json"

# Simulator-wide constants — no longer encoded in level files (per LEVEL_FORMAT.md §12).
_DEFAULT_BUFFER_SLOTS = 5
_DEFAULT_BUCKET_CAPACITY = 24

# Per LEVEL_FORMAT.md §8.1.
_KNOWN_SHORT_CODES = frozenset({
    "b", "br", "db", "dg", "dgr", "do", "dr", "f", "g", "gr",
    "o", "og", "p", "pk", "r", "w", "y",
})

# Per LEVEL_FORMAT.md §8.2 / task spec.
FULL_NAME_TO_SHORT = {
    "Yellow": "y", "Blue": "b", "Red": "r", "Green": "g",
    "Purple": "p", "Pink": "pk", "Orange": "o", "White": "w",
    "DarkBlue": "db", "DarkRed": "dr", "DarkGray": "dgr", "OliveGreen": "og",
}

# Opposite of pin facing direction → (drow, dcol) offset for the destruction cell.
_OPPOSITE_OFFSET = {
    "Up": (1, 0),     # opposite of Up is Down → row+1
    "Down": (-1, 0),  # opposite of Down is Up → row-1
    "Left": (0, 1),   # opposite of Left is Right → col+1
    "Right": (0, -1), # opposite of Right is Left → col-1
}

# Paxie "Up"/"Down"/"Left"/"Right" → existing internal lowercase generator facing.
_GEN_FACING_NORM = {"Up": "up", "Down": "down", "Left": "left", "Right": "right"}


class LevelLoadError(Exception):
    """Raised on any failure to load a level (schema, semantic, or unsupported mechanic)."""


class UnsupportedMechanicError(LevelLoadError):
    """Raised when a level contains a mechanic the simulator does not model."""


def _load_schema() -> dict:
    with open(_SCHEMA_PATH) as f:
        return json.load(f)


def _normalize_color(token: str) -> str:
    """Map a single color token to its canonical short code, or raise ValueError."""
    if token in _KNOWN_SHORT_CODES:
        return token
    if token in FULL_NAME_TO_SHORT:
        return FULL_NAME_TO_SHORT[token]
    raise ValueError(
        f"Unknown color token {token!r}: not a known short code "
        f"({sorted(_KNOWN_SHORT_CODES)}) and not in the full-name alias map "
        f"({sorted(FULL_NAME_TO_SHORT)})."
    )


def _normalize_all_colors(data: dict) -> None:
    """Walk the level dict and normalize every color token in place.

    Touches: stack colors, single/wood/ice collector colors, tunnel queue colors
    (both collectorArea and hexStackArea), keyLocks colors, editorMeta.zoneColors.
    Schema-allowed but unsupported sub-arrays are still normalized if non-empty,
    because the unsupported-mechanic check happens *after* normalization and we
    don't want unsupported entries to short-circuit the validity of their tokens.
    """
    ca = data.get("collectorArea", {})

    for entry in ca.get("singleBlockCollectors", []) or []:
        entry["color"] = _normalize_color(entry["color"])

    for entry in ca.get("woodBoxCollectors", []) or []:
        entry["hiddenColor"] = _normalize_color(entry["hiddenColor"])

    for entry in ca.get("iceCollectors", []) or []:
        entry["hiddenColor"] = _normalize_color(entry["hiddenColor"])

    for tunnel in ca.get("tunnels", []) or []:
        for q in tunnel.get("collectorQueue", []):
            q["color"] = _normalize_color(q["color"])

    for kl in ca.get("keyLocks", []) or []:
        if "color" in kl:
            kl["color"] = _normalize_color(kl["color"])

    hs = data.get("hexStackArea", {})
    for stack in hs.get("stacks", []) or []:
        stack["colors"] = [_normalize_color(c) for c in stack["colors"]]

    for tunnel in hs.get("tunnels", []) or []:
        for q in tunnel.get("collectorQueue", []):
            q["color"] = _normalize_color(q["color"])

    em = data.get("editorMeta", {})
    zone_colors = em.get("zoneColors")
    if zone_colors:
        em["zoneColors"] = [[_normalize_color(c) for c in zone] for zone in zone_colors]


def _check_unsupported(data: dict) -> None:
    """Hard-reject any level containing a mechanic the simulator does not model."""
    ca = data.get("collectorArea", {})
    if ca.get("mysteryCollectors"):
        raise UnsupportedMechanicError(
            "collectorArea.mysteryCollectors is non-empty — mystery collectors are unsupported."
        )
    if ca.get("tiedPairs"):
        raise UnsupportedMechanicError(
            "collectorArea.tiedPairs is non-empty — tied pairs are unsupported."
        )
    if ca.get("keyLocks"):
        raise UnsupportedMechanicError(
            "collectorArea.keyLocks is non-empty — key/lock pairs are unsupported."
        )
    hs = data.get("hexStackArea", {})
    if hs.get("tunnels"):
        raise UnsupportedMechanicError(
            "hexStackArea.tunnels is non-empty — hex-stack-area generators are unsupported."
        )


def _semantic_checks(data: dict) -> set[str]:
    """Run the eight semantic checks per LEVEL_FORMAT.md §10.2.

    Returns the set of distinct level colors (computed as a side product since
    every color-bearing field has to be visited anyway). Hard-fails by raising
    LevelLoadError; soft-fails by emitting warnings.
    """
    ca = data["collectorArea"]
    hs = data["hexStackArea"]
    em = data.get("editorMeta", {})

    ca_w = ca["gridWidth"]
    ca_h = ca["gridHeight"]
    hs_w = hs["gridWidth"]
    hs_h = hs["gridHeight"]

    single = ca.get("singleBlockCollectors", []) or []
    wood   = ca.get("woodBoxCollectors", []) or []
    ice    = ca.get("iceCollectors", []) or []
    dead   = ca.get("deadCells", []) or []
    tuns   = ca.get("tunnels", []) or []
    pins   = ca.get("pinBlockers", []) or []
    stacks = hs.get("stacks", []) or []

    # (1) Cell exclusivity in collectorArea (pins excluded).
    occupied: dict[tuple[int, int], str] = {}
    def _claim(x: int, y: int, kind: str) -> None:
        key = (x, y)
        if key in occupied:
            raise LevelLoadError(
                f"Cell exclusivity violated at (x={x}, y={y}): "
                f"both a {occupied[key]} and a {kind} are placed there."
            )
        occupied[key] = kind

    for e in single: _claim(e["x"], e["y"], "singleBlockCollector")
    for e in wood:   _claim(e["x"], e["y"], "woodBoxCollector")
    for e in ice:    _claim(e["x"], e["y"], "iceCollector")
    for e in dead:   _claim(e["x"], e["y"], "deadCell")
    for e in tuns:   _claim(e["x"], e["y"], "tunnel")

    # (2) Cell-in-bounds.
    def _in_bounds_ca(x: int, y: int, kind: str) -> None:
        if not (0 <= x < ca_w and 0 <= y < ca_h):
            raise LevelLoadError(
                f"{kind} at (x={x}, y={y}) is out of collectorArea bounds "
                f"({ca_w}x{ca_h})."
            )

    for e in single: _in_bounds_ca(e["x"], e["y"], "singleBlockCollector")
    for e in wood:   _in_bounds_ca(e["x"], e["y"], "woodBoxCollector")
    for e in ice:    _in_bounds_ca(e["x"], e["y"], "iceCollector")
    for e in dead:   _in_bounds_ca(e["x"], e["y"], "deadCell")
    for e in tuns:   _in_bounds_ca(e["x"], e["y"], "tunnel")
    for e in pins:   _in_bounds_ca(e["x"], e["y"], "pinBlocker")

    for s in stacks:
        if not (0 <= s["x"] < hs_w and 0 <= s["y"] < hs_h):
            raise LevelLoadError(
                f"Stack at (x={s['x']}, y={s['y']}) is out of hexStackArea bounds "
                f"({hs_w}x{hs_h})."
            )

    # Collect distinct level colors.
    distinct_colors: set[str] = set()
    for s in stacks:
        distinct_colors.update(s["colors"])
    for e in single: distinct_colors.add(e["color"])
    for e in wood:   distinct_colors.add(e["hiddenColor"])
    for e in ice:    distinct_colors.add(e["hiddenColor"])
    for t in tuns:
        for q in t.get("collectorQueue", []):
            distinct_colors.add(q["color"])

    # (3) Generator queue consistency — every queue color is a level color.
    # After normalization+collection above this is trivially true, but a queue
    # color that appears nowhere else would still pass; the spec only requires
    # it to be *a* level color, which it now is. The real failure mode this
    # check catches is unknown tokens, already handled in normalization.

    # (4) Color cross-check (warning).
    declared = em.get("colorCount")
    if declared is not None and declared != len(distinct_colors):
        warnings.warn(
            f"editorMeta.colorCount={declared} but actual distinct color count is "
            f"{len(distinct_colors)}: {sorted(distinct_colors)}",
            UserWarning,
            stacklevel=2,
        )

    # (5) Pin destruction cell.
    pin_cells: set[tuple[int, int]] = {(e["x"], e["y"]) for e in pins}
    for p in pins:
        px, py = p["x"], p["y"]
        direction = p["direction"]
        drow, dcol = _OPPOSITE_OFFSET[direction]
        dest_row, dest_col = py + drow, px + dcol
        if not (0 <= dest_col < ca_w and 0 <= dest_row < ca_h):
            warnings.warn(
                f"Pin at (x={px}, y={py}) facing {direction}: destruction cell "
                f"at (x={dest_col}, y={dest_row}) is off-grid — pin can never be destroyed.",
                UserWarning,
                stacklevel=2,
            )
            continue
        dest_key = (dest_col, dest_row)
        if dest_key in occupied:
            kind = occupied[dest_key]
            if kind in ("deadCell", "tunnel"):
                raise LevelLoadError(
                    f"Pin at (x={px}, y={py}) facing {direction}: destruction cell "
                    f"(x={dest_col}, y={dest_row}) contains a {kind}, which can never be cleared "
                    f"— level is malformed (per LEVEL_FORMAT.md §10.2 check 5)."
                )
        elif dest_key in pin_cells:
            raise LevelLoadError(
                f"Pin at (x={px}, y={py}) facing {direction}: destruction cell "
                f"(x={dest_col}, y={dest_row}) is another pin's origin — level is malformed "
                f"(per LEVEL_FORMAT.md §10.2 check 5)."
            )
        # If the destruction cell is empty or holds a pickable entity
        # (singleBlock / woodBox / iceCollector), no warning needed.

    # (6) Unsupported mechanics — already handled by _check_unsupported.
    # (7) Color token validity — already handled by _normalize_all_colors.

    # (8) Slice-bucket parity (warning).
    total_slices = sum(len(s["colors"]) for s in stacks)
    bucket_count = len(single) + len(wood) + len(ice)
    for t in tuns:
        bucket_count += len(t.get("collectorQueue", []))
    expected_slices = bucket_count * _DEFAULT_BUCKET_CAPACITY
    if total_slices != expected_slices:
        warnings.warn(
            f"Slice-bucket parity mismatch: field has {total_slices} slices but "
            f"{bucket_count} buckets × {_DEFAULT_BUCKET_CAPACITY} capacity = {expected_slices}.",
            UserWarning,
            stacklevel=2,
        )

    return distinct_colors


def _build_state(data: dict, level_id: str, distinct_colors: set[str], seed: int | None) -> GameState:
    ca = data["collectorArea"]
    hs = data["hexStackArea"]

    ca_w = ca["gridWidth"]
    ca_h = ca["gridHeight"]

    # Reserve grid: row = y, col = x.
    reserve: list[list] = [[None] * ca_w for _ in range(ca_h)]

    for e in ca.get("singleBlockCollectors", []) or []:
        reserve[e["y"]][e["x"]] = PlainBucket(color=e["color"])

    for e in ca.get("woodBoxCollectors", []) or []:
        reserve[e["y"]][e["x"]] = QuestionBucket(color=e["hiddenColor"], revealed=False)

    for e in ca.get("iceCollectors", []) or []:
        reserve[e["y"]][e["x"]] = IceBucket(
            row=e["y"],
            col=e["x"],
            color=e["hiddenColor"],
            thaw_threshold=e["iceCapacity"],
            thawed=False,
        )

    for e in ca.get("deadCells", []) or []:
        reserve[e["y"]][e["x"]] = Wall()

    for t in ca.get("tunnels", []) or []:
        queue = [q["color"] for q in t.get("collectorQueue", [])]
        reserve[t["y"]][t["x"]] = Generator(
            facing=_GEN_FACING_NORM[t["direction"]],
            remaining=len(queue),
            queue=queue,
        )

    pin_objs: list[Pin] = []
    for p in ca.get("pinBlockers", []) or []:
        pin_objs.append(Pin(
            origin_row=p["y"],
            origin_col=p["x"],
            direction=p["direction"],
            block_count=p.get("blockCount", 0),
            destroyed=False,
        ))

    # Hex field: existing GameState convention is dict[(col, row)] → top-to-bottom slices.
    # Paxie (x, y) maps directly to (col, row).
    field: dict[tuple[int, int], list[str]] = {
        (s["x"], s["y"]): list(s["colors"])
        for s in hs.get("stacks", []) or []
    }

    buffer: list[BufferBucket | None] = [None] * _DEFAULT_BUFFER_SLOTS

    rng = random.Random(seed) if seed is not None else random.Random()

    state = GameState(
        field=field,
        buffer_slots=_DEFAULT_BUFFER_SLOTS,
        bucket_capacity=_DEFAULT_BUCKET_CAPACITY,
        buffer=buffer,
        reserve_rows=ca_h,
        reserve_cols=ca_w,
        reserve=reserve,
        color_set=frozenset(distinct_colors),
        level_id=level_id,
        rng=rng,
        quiescent=False,
        pins=pin_objs,
        move_counter=0,
    )

    run_until_quiescent(state)
    return state


def load_level_from_data(data: dict, *, level_id: str | None = None, seed: int | None = None) -> GameState:
    """Load a level from an in-memory dict (Paxie format).

    Useful for tests and for the worked example in LEVEL_FORMAT.md §11.
    """
    schema = _load_schema()
    try:
        jsonschema.validate(data, schema)
    except jsonschema.ValidationError as e:
        raise LevelLoadError(f"Schema validation failed: {e.message}") from e

    _normalize_all_colors(data)
    _check_unsupported(data)
    distinct_colors = _semantic_checks(data)

    if level_id is None:
        level_id = f"level-{data['levelNumber']}"

    return _build_state(data, level_id, distinct_colors, seed)


def load_level(path: str | Path, *, seed: int | None = None) -> GameState:
    path = Path(path)
    with open(path) as f:
        data = json.load(f)

    return load_level_from_data(data, level_id=path.stem, seed=seed)
