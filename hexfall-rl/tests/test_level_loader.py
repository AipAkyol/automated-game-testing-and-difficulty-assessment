import json
from pathlib import Path

import pytest

from hexfall.level_loader import LevelLoadError, load_level
from hexfall.types import Generator, PlainBucket, QuestionBucket, Wall

LEVELS_DIR = Path(__file__).parent.parent / "levels"
TINY = LEVELS_DIR / "tiny_solvable.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write(tmp_path: Path, data: dict) -> Path:
    p = tmp_path / "level.json"
    p.write_text(json.dumps(data))
    return p


def _minimal(*, color_count=2, slices=None, capacity=4, cells=None, rows=1, cols=2):
    """Return a minimal valid level dict. Parity holds by default."""
    if slices is None:
        slices = [["red", "red", "red", "red"], ["blue", "blue", "blue", "blue"]]
    if cells is None:
        cells = [
            {"row": 0, "col": 0, "type": "plain_bucket", "color": "red"},
            {"row": 0, "col": 1, "type": "plain_bucket", "color": "blue"},
        ]
    stacks = [
        {"col": i, "row": 0, "slices": s}
        for i, s in enumerate(slices)
    ]
    return {
        "meta": {"id": "test-level", "name": "Test", "version": 1, "color_count": color_count},
        "field": {"stacks": stacks},
        "buffer": {"slots": 5, "bucket_capacity": capacity},
        "reserve": {"rows": rows, "cols": cols, "cells": cells},
    }


# ---------------------------------------------------------------------------
# Test 1
# ---------------------------------------------------------------------------

def test_load_tiny_solvable():
    state = load_level(TINY)

    assert (0, 0) in state.field
    assert (1, 0) in state.field
    assert state.field[(0, 0)] == ["red", "red", "red", "red"]
    assert state.field[(1, 0)] == ["blue", "blue", "blue", "blue"]

    assert state.buffer == [None, None, None, None, None]
    assert state.bucket_capacity == 4
    assert state.buffer_slots == 5

    assert state.reserve_rows == 1
    assert state.reserve_cols == 2
    assert state.reserve[0][0] == PlainBucket(color="red")
    assert state.reserve[0][1] == PlainBucket(color="blue")

    assert state.color_set == frozenset({"red", "blue"})
    assert state.level_id == "tiny-solvable"
    assert state.quiescent is True


# ---------------------------------------------------------------------------
# Test 2
# ---------------------------------------------------------------------------

def test_seed_determinism():
    s1 = load_level(TINY, seed=42)
    s2 = load_level(TINY, seed=42)
    assert s1.rng.randint(0, 1_000_000) == s2.rng.randint(0, 1_000_000)


# ---------------------------------------------------------------------------
# Test 3
# ---------------------------------------------------------------------------

def test_seed_none_independent():
    s1 = load_level(TINY, seed=None)
    s2 = load_level(TINY, seed=None)
    draws = [(s1.rng.randint(0, 1_000_000), s2.rng.randint(0, 1_000_000)) for _ in range(10)]
    assert any(a != b for a, b in draws), "Two seed=None RNG instances produced identical sequences"


# ---------------------------------------------------------------------------
# Test 4
# ---------------------------------------------------------------------------

def test_invalid_schema_missing_meta(tmp_path):
    bad = {
        "field": {"stacks": []},
        "buffer": {"slots": 5, "bucket_capacity": 25},
        "reserve": {"rows": 1, "cols": 1, "cells": []},
    }
    p = _write(tmp_path, bad)
    with pytest.raises(LevelLoadError):
        load_level(p)


# ---------------------------------------------------------------------------
# Test 5
# ---------------------------------------------------------------------------

def test_invalid_schema_unknown_cell_type(tmp_path):
    data = _minimal(
        color_count=2,
        cells=[{"row": 0, "col": 0, "type": "wormhole"}],
        rows=1,
        cols=2,
    )
    p = _write(tmp_path, data)
    with pytest.raises(LevelLoadError):
        load_level(p)


# ---------------------------------------------------------------------------
# Test 6
# ---------------------------------------------------------------------------

def test_generator_remaining_mismatch(tmp_path):
    data = _minimal(
        color_count=2,
        slices=[["red"], ["blue"]],
        capacity=1,
        cells=[
            {
                "row": 0, "col": 0,
                "type": "generator",
                "facing": "right",
                "remaining": 3,
                "queue": ["red", "blue"],  # len=2, not 3
            }
        ],
        rows=1,
        cols=2,
    )
    p = _write(tmp_path, data)
    with pytest.raises(LevelLoadError):
        load_level(p)


# ---------------------------------------------------------------------------
# Test 7
# ---------------------------------------------------------------------------

def test_color_count_mismatch(tmp_path):
    data = _minimal(color_count=3)  # actual distinct colors = 2 (red, blue)
    p = _write(tmp_path, data)
    with pytest.raises(LevelLoadError):
        load_level(p)


# ---------------------------------------------------------------------------
# Test 8
# ---------------------------------------------------------------------------

def test_implicit_empty_reserve_cells(tmp_path):
    data = {
        "meta": {"id": "test-empty", "name": "Test Empty", "version": 1, "color_count": 1},
        "field": {"stacks": [{"col": 0, "row": 0, "slices": ["red", "red", "red", "red"]}]},
        "buffer": {"slots": 5, "bucket_capacity": 4},
        "reserve": {
            "rows": 3,
            "cols": 3,
            "cells": [{"row": 0, "col": 0, "type": "plain_bucket", "color": "red"}],
        },
    }
    p = _write(tmp_path, data)
    state = load_level(p)

    assert state.reserve_rows == 3
    assert state.reserve_cols == 3
    assert state.reserve[0][0] == PlainBucket(color="red")

    none_count = sum(
        1
        for r in range(3)
        for c in range(3)
        if not (r == 0 and c == 0)
        if state.reserve[r][c] is None
    )
    assert none_count == 8


# ---------------------------------------------------------------------------
# Test 9
# ---------------------------------------------------------------------------

def test_walls_and_question_buckets_load(tmp_path):
    data = {
        "meta": {"id": "test-wq", "name": "Test WQ", "version": 1, "color_count": 2},
        "field": {"stacks": [
            {"col": 0, "row": 0, "slices": ["red"]},
            {"col": 1, "row": 0, "slices": ["blue"]},
        ]},
        "buffer": {"slots": 5, "bucket_capacity": 1},
        "reserve": {
            "rows": 2,
            "cols": 2,
            "cells": [
                {"row": 0, "col": 0, "type": "wall"},
                {"row": 0, "col": 1, "type": "question_bucket", "color": "red"},
                {"row": 1, "col": 0, "type": "plain_bucket", "color": "blue"},
            ],
        },
    }
    p = _write(tmp_path, data)
    state = load_level(p)

    assert isinstance(state.reserve[0][0], Wall)
    # ?-bucket in top row becomes reachable at load, so reveal phase flips revealed=True.
    assert state.reserve[0][1] == QuestionBucket(color="red", revealed=True)


# ---------------------------------------------------------------------------
# Test 10
# ---------------------------------------------------------------------------

def test_generator_loads(tmp_path):
    data = {
        "meta": {"id": "test-gen", "name": "Test Gen", "version": 1, "color_count": 2},
        "field": {"stacks": [
            {"col": 0, "row": 0, "slices": ["red"]},
            {"col": 1, "row": 0, "slices": ["blue"]},
        ]},
        "buffer": {"slots": 5, "bucket_capacity": 1},
        "reserve": {
            "rows": 1,
            "cols": 3,
            "cells": [
                {
                    "row": 0, "col": 0,
                    "type": "generator",
                    "facing": "right",
                    "remaining": 2,
                    "queue": ["red", "blue"],
                }
            ],
        },
    }
    p = _write(tmp_path, data)
    state = load_level(p)

    # Generator faces right into (0, 1) which is empty at load → it fires once.
    # After firing: produces "red" into (0, 1), remaining=1, queue=["blue"].
    # It can't fire again because (0, 1) is now occupied.
    assert state.reserve[0][0] == Generator(facing="right", remaining=1, queue=["blue"])
    assert state.reserve[0][1] == PlainBucket(color="red")


# ---------------------------------------------------------------------------
# Test 11
# ---------------------------------------------------------------------------

def test_parity_warning(tmp_path):
    data = {
        "meta": {"id": "test-parity", "name": "Test Parity", "version": 1, "color_count": 2},
        "field": {"stacks": [
            {"col": 0, "row": 0, "slices": ["red", "red", "red"]},  # 3 slices, parity expects 8
        ]},
        "buffer": {"slots": 5, "bucket_capacity": 4},
        "reserve": {
            "rows": 1,
            "cols": 2,
            "cells": [
                {"row": 0, "col": 0, "type": "plain_bucket", "color": "red"},
                {"row": 0, "col": 1, "type": "plain_bucket", "color": "blue"},
            ],
        },
    }
    p = _write(tmp_path, data)
    with pytest.warns(UserWarning):
        state = load_level(p)
    assert state is not None
