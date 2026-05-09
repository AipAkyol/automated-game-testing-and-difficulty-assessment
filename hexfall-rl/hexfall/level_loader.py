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
    PlainBucket,
    QuestionBucket,
    Wall,
)

_SCHEMA_PATH = Path(__file__).parent / "schemas" / "level_schema.json"


class LevelLoadError(Exception):
    pass


def _load_schema() -> dict:
    with open(_SCHEMA_PATH) as f:
        return json.load(f)


def load_level(path: str | Path, *, seed: int | None = None) -> GameState:
    path = Path(path)
    with open(path) as f:
        data = json.load(f)

    schema = _load_schema()
    try:
        jsonschema.validate(data, schema)
    except jsonschema.ValidationError as e:
        raise LevelLoadError(f"Schema validation failed: {e.message}") from e

    reserve_cells = data["reserve"]["cells"]

    # Invariant (a): generator remaining == len(queue)
    for cell in reserve_cells:
        if cell["type"] == "generator":
            if cell["remaining"] != len(cell["queue"]):
                raise LevelLoadError(
                    f"Generator at row={cell['row']} col={cell['col']}: "
                    f"remaining={cell['remaining']} but len(queue)={len(cell['queue'])}"
                )

    # Invariant (b): meta.color_count == len(distinct_colors)
    distinct_colors: set[str] = set()
    for stack in data["field"]["stacks"]:
        distinct_colors.update(stack["slices"])
    for cell in reserve_cells:
        if cell["type"] in ("plain_bucket", "question_bucket"):
            distinct_colors.add(cell["color"])
        elif cell["type"] == "generator":
            distinct_colors.update(cell["queue"])

    expected_color_count = data["meta"]["color_count"]
    if expected_color_count != len(distinct_colors):
        raise LevelLoadError(
            f"meta.color_count={expected_color_count} does not match "
            f"actual distinct color count={len(distinct_colors)}: "
            f"{sorted(distinct_colors)}"
        )

    # Invariant (c): slice-bucket parity (warn only)
    total_slices = sum(len(stack["slices"]) for stack in data["field"]["stacks"])
    bucket_capacity = data["buffer"]["bucket_capacity"]
    plain_count = sum(1 for c in reserve_cells if c["type"] == "plain_bucket")
    question_count = sum(1 for c in reserve_cells if c["type"] == "question_bucket")
    gen_remaining = sum(c["remaining"] for c in reserve_cells if c["type"] == "generator")
    total_eventual_buckets = plain_count + question_count + gen_remaining
    expected_slices = bucket_capacity * total_eventual_buckets
    if total_slices != expected_slices:
        warnings.warn(
            f"Slice-bucket parity mismatch: field has {total_slices} slices but "
            f"{total_eventual_buckets} buckets × {bucket_capacity} capacity = {expected_slices}",
            UserWarning,
            stacklevel=2,
        )

    # Build GameState
    field: dict[tuple[int, int], list[str]] = {
        (stack["col"], stack["row"]): list(stack["slices"])
        for stack in data["field"]["stacks"]
    }

    buffer_slots = data["buffer"]["slots"]
    buffer: list[BufferBucket | None] = [None] * buffer_slots

    reserve_rows = data["reserve"]["rows"]
    reserve_cols = data["reserve"]["cols"]
    reserve: list[list] = [[None] * reserve_cols for _ in range(reserve_rows)]

    for cell in reserve_cells:
        r, c = cell["row"], cell["col"]
        cell_type = cell["type"]
        if cell_type == "plain_bucket":
            reserve[r][c] = PlainBucket(color=cell["color"])
        elif cell_type == "question_bucket":
            reserve[r][c] = QuestionBucket(color=cell["color"], revealed=False)
        elif cell_type == "generator":
            reserve[r][c] = Generator(
                facing=cell["facing"],
                remaining=cell["remaining"],
                queue=list(cell["queue"]),
            )
        elif cell_type == "wall":
            reserve[r][c] = Wall()

    rng = random.Random(seed) if seed is not None else random.Random()

    state = GameState(
        field=field,
        buffer_slots=buffer_slots,
        bucket_capacity=bucket_capacity,
        buffer=buffer,
        reserve_rows=reserve_rows,
        reserve_cols=reserve_cols,
        reserve=reserve,
        color_set=frozenset(distinct_colors),
        level_id=data["meta"]["id"],
        rng=rng,
        quiescent=False,
    )

    run_until_quiescent(state)

    return state
