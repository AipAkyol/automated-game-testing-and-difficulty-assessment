import json
import random
import warnings
from pathlib import Path

import pytest

from hexfall.game import (
    compute_reachability,
    is_terminal,
    legal_actions_mask,
    run_until_quiescent,
    step,
)
from hexfall.level_loader import load_level
from hexfall.types import (
    BufferBucket,
    GameState,
    Generator,
    PlainBucket,
    QuestionBucket,
    Wall,
)

LEVELS_DIR = Path(__file__).parent.parent / "levels"
TINY = LEVELS_DIR / "tiny_solvable.json"


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def make_state(
    *,
    field=None,
    buffer=None,
    reserve=None,
    reserve_rows=None,
    reserve_cols=None,
    buffer_slots=5,
    bucket_capacity=4,
    color_set=None,
    level_id="test",
    seed=0,
):
    """Construct a GameState directly without JSON loading."""
    if reserve is not None:
        reserve_rows = len(reserve)
        reserve_cols = len(reserve[0]) if reserve else 0
    else:
        reserve_rows = reserve_rows if reserve_rows is not None else 1
        reserve_cols = reserve_cols if reserve_cols is not None else 1
        reserve = [[None] * reserve_cols for _ in range(reserve_rows)]

    if buffer is None:
        buffer = [None] * buffer_slots
    field = dict(field) if field else {}
    color_set = color_set or frozenset()

    return GameState(
        field=field,
        buffer_slots=buffer_slots,
        bucket_capacity=bucket_capacity,
        buffer=list(buffer),
        reserve_rows=reserve_rows,
        reserve_cols=reserve_cols,
        reserve=reserve,
        color_set=color_set,
        level_id=level_id,
        rng=random.Random(seed),
        quiescent=False,
    )


def _write_level(tmp_path: Path, data: dict, name: str = "level.json") -> Path:
    p = tmp_path / name
    p.write_text(json.dumps(data))
    return p


# ===========================================================================
# Reachability
# ===========================================================================

def test_reachability_top_row_buckets():
    state = make_state(
        reserve=[[PlainBucket("red"), PlainBucket("blue"), PlainBucket("green")]],
    )
    reach = compute_reachability(state)
    assert reach == [[True, True, True]]


def test_reachability_top_row_wall_not_pickable():
    state = make_state(
        reserve=[[PlainBucket("red"), Wall(), PlainBucket("blue")]],
    )
    reach = compute_reachability(state)
    assert reach == [[True, False, True]]


def test_reachability_through_top_row_empty():
    state = make_state(
        reserve=[
            [None, PlainBucket("red")],
            [PlainBucket("blue"), PlainBucket("green")],
        ],
    )
    reach = compute_reachability(state)
    # Top row [0][0] is None: not pickable. [0][1] reachable (top row).
    # [1][0] reachable: 4-neighbor (0,0) is in top_empty.
    # [1][1] reachable: 4-neighbor (0,1)? (0,1) is a bucket, not empty.
    #   Other neighbors of (1,1): (1,0) bucket, (0,1) bucket. So NOT reachable via empties.
    assert reach[0][0] is False
    assert reach[0][1] is True
    assert reach[1][0] is True
    assert reach[1][1] is False


def test_reachability_isolated_empty_does_not_grant_reachability():
    """Critical: guards against 'any empty neighbor' misreading.

    A mid-grid empty cell surrounded by buckets must NOT mark adjacent buckets
    reachable, because the empty cell does not trace back to the top edge.
    """
    state = make_state(
        reserve=[
            [PlainBucket("red"), PlainBucket("red"), PlainBucket("red")],
            [PlainBucket("red"), None, PlainBucket("red")],
            [PlainBucket("red"), PlainBucket("red"), PlainBucket("red")],
        ],
    )
    reach = compute_reachability(state)
    # Top row reachable.
    assert reach[0] == [True, True, True]
    # Buckets adjacent to isolated empty (1,1) must NOT be reachable just from that.
    # (1, 0): neighbors (0,0) bucket, (2,0) bucket, (1,1) empty (not top-connected).
    # No top_empty neighbor → False.
    assert reach[1][0] is False
    assert reach[1][2] is False
    assert reach[2][1] is False


def test_reachability_wall_blocks_propagation():
    # 3x1: empty top row, wall middle, plain bucket bottom. Wall blocks propagation.
    state = make_state(
        reserve=[
            [None],
            [Wall()],
            [PlainBucket("red")],
        ],
    )
    reach = compute_reachability(state)
    assert reach[2][0] is False

    # Sanity: without the wall, propagation reaches the bucket.
    state2 = make_state(
        reserve=[
            [None],
            [None],
            [PlainBucket("red")],
        ],
    )
    assert compute_reachability(state2)[2][0] is True


def test_reachability_generator_blocks_propagation():
    state = make_state(
        reserve=[
            [None],
            [Generator(facing="up", remaining=2, queue=["red", "red"])],
            [PlainBucket("red")],
        ],
    )
    reach = compute_reachability(state)
    assert reach[2][0] is False


def test_reachability_after_pick():
    state = make_state(
        field={},  # empty field → step terminates with "win", but we check reach mid-state
        reserve=[[PlainBucket("red")], [PlainBucket("red")]],
        bucket_capacity=1,
        color_set=frozenset({"red"}),
    )
    assert compute_reachability(state)[1][0] is False
    info = step(state, (0, 0))
    # field was empty so termination is "win"
    assert info["termination_reason"] == "win"
    # The cell below the picked bucket is now reachable.
    reach_after = compute_reachability(state)
    assert reach_after[1][0] is True


def test_reachability_demo_all_types_layout(tmp_path):
    """Reproduces LEVEL_FORMAT.md §10. After load (with on-load generator firing),
    only top row is reachable."""
    data = {
        "meta": {
            "id": "demo-all-types",
            "name": "Demo: All Cell Types",
            "version": 1,
            "color_count": 3,
        },
        "field": {
            "stacks": [
                {"col": 0, "row": 0, "slices": ["red", "blue", "green"]},
                {"col": 1, "row": 0, "slices": ["blue", "red"]},
                {"col": 0, "row": 1, "slices": ["green", "red", "blue", "green"]},
                {"col": 1, "row": 1, "slices": ["red"]},
            ]
        },
        "buffer": {"slots": 5, "bucket_capacity": 25},
        "reserve": {
            "rows": 3,
            "cols": 4,
            "cells": [
                {"row": 0, "col": 0, "type": "plain_bucket", "color": "red"},
                {"row": 0, "col": 1, "type": "plain_bucket", "color": "blue"},
                {"row": 0, "col": 2, "type": "question_bucket", "color": "green"},
                {"row": 0, "col": 3, "type": "wall"},
                {"row": 1, "col": 0, "type": "plain_bucket", "color": "green"},
                {
                    "row": 1, "col": 2, "type": "generator",
                    "facing": "left", "remaining": 2,
                    "queue": ["red", "blue"],
                },
                {"row": 2, "col": 1, "type": "plain_bucket", "color": "red"},
            ],
        },
    }
    p = _write_level(tmp_path, data)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # suppress parity warning
        state = load_level(p)

    # Generator fired on load; (1,1) is now a plain red bucket.
    assert state.reserve[1][1] == PlainBucket(color="red")
    gen = state.reserve[1][2]
    assert isinstance(gen, Generator)
    assert gen.remaining == 1
    assert gen.queue == ["blue"]

    reach = compute_reachability(state)
    # Only top-row pickables reachable. Wall at (0,3) is not pickable.
    assert reach[0][0] is True
    assert reach[0][1] is True
    assert reach[0][2] is True
    assert reach[0][3] is False
    for r in (1, 2):
        for c in range(4):
            assert reach[r][c] is False, f"unexpected reach at ({r}, {c})"


# ===========================================================================
# Buffer pulls and same-color collision
# ===========================================================================

def test_pull_drains_matching_stack_over_ticks():
    state = make_state(
        field={(0, 0): ["red", "red", "red", "red"]},
        reserve=[[PlainBucket("red")]],
        bucket_capacity=4,
        color_set=frozenset({"red"}),
    )
    info = step(state, (0, 0))
    # All 4 reds pulled into the bucket; bucket filled and removed; field empty.
    assert all(slot is None for slot in state.buffer)
    assert all(not s for s in state.field.values())
    assert info["termination_reason"] == "win"


def test_same_color_fuller_pulls_first():
    fuller = BufferBucket(color="red", capacity=10, fill=5)
    less = BufferBucket(color="red", capacity=10, fill=1)
    state = make_state(
        field={(0, 0): ["red", "red", "red", "red"]},
        buffer=[fuller, less, None, None, None],
        reserve=[[None]],
        bucket_capacity=10,
        color_set=frozenset({"red"}),
    )
    run_until_quiescent(state)
    # Fuller (started at 5) pulls first to fill: needed 5 more. less stayed at 1
    # while fuller was active. Once fuller hits 10 and leaves (in fill check),
    # less can start pulling. But the field has only 4 slices; fuller drains
    # them all (1 → 5 reds = 4 pulls bringing fill to 9; only 4 in field).
    # Actually fuller starts at 5, capacity 10, needs 5 more to fill. Field has
    # 4 → fuller hits 9, never fills. less stays at 1 the whole time.
    assert state.buffer[0] == BufferBucket(color="red", capacity=10, fill=9)
    assert state.buffer[1] == BufferBucket(color="red", capacity=10, fill=1)
    assert (0, 0) not in state.field or not state.field.get((0, 0))


def test_same_color_less_full_blocked_at_different_stack():
    """Critical: guards against local-only misreading of same-color collision."""
    fuller = BufferBucket(color="red", capacity=10, fill=5)
    less = BufferBucket(color="red", capacity=10, fill=1)
    state = make_state(
        field={
            (0, 0): ["red"],
            (1, 0): ["red"],  # different stack, also red top
        },
        buffer=[fuller, less, None, None, None],
        reserve=[[None]],
        bucket_capacity=10,
        color_set=frozenset({"red"}),
    )
    run_until_quiescent(state)
    # Less-full bucket pulls NOTHING while fuller is in buffer, even though a
    # matching slice is exposed at a different stack.
    # Fuller started at 5 → pulls 2 reds (one from each stack over 2 ticks) → fill=7.
    # Field totally drains because fuller pulls one per tick from sorted (col,row) order.
    assert state.buffer[1].fill == 1
    assert state.buffer[0].fill == 7


def test_distinct_colors_pull_same_tick():
    red = BufferBucket(color="red", capacity=1, fill=0)
    blue = BufferBucket(color="blue", capacity=1, fill=0)
    state = make_state(
        field={(0, 0): ["red"], (1, 0): ["blue"]},
        buffer=[red, blue, None, None, None],
        reserve=[[None, None]],
        bucket_capacity=1,
        color_set=frozenset({"red", "blue"}),
    )
    run_until_quiescent(state)
    # Both pulled and filled and were removed.
    assert state.buffer[0] is None
    assert state.buffer[1] is None
    assert all(not s for s in state.field.values())


# ===========================================================================
# Stack clear and fall
# ===========================================================================

def test_fall_one_candidate_does_not_call_rng():
    state = make_state(
        field={(0, 0): ["red"], (0, 1): ["red"]},  # (0,1) is bottom
        reserve=[[PlainBucket("red")]],
        bucket_capacity=2,
        color_set=frozenset({"red"}),
        seed=42,
    )
    rng_before = state.rng.getstate()
    step(state, (0, 0))
    rng_after = state.rng.getstate()
    assert rng_before == rng_after, "1-candidate fall must not consume RNG"


def _make_two_candidate_state(seed):
    return make_state(
        field={(0, 0): ["red"], (1, 0): ["blue"], (0, 1): ["red"]},
        reserve=[[PlainBucket("red")]],
        bucket_capacity=1,
        color_set=frozenset({"red", "blue"}),
        seed=seed,
    )


def test_fall_two_candidates_uses_rng_deterministically():
    s1 = _make_two_candidate_state(seed=42)
    s2 = _make_two_candidate_state(seed=42)
    step(s1, (0, 0))
    step(s2, (0, 0))
    assert dict(s1.field) == dict(s2.field)

    # Different seeds can produce different choice. We try a range of seeds and
    # require at least one outcome differs.
    found_diff = False
    for alt in range(1, 100):
        if alt == 42:
            continue
        s = _make_two_candidate_state(seed=alt)
        step(s, (0, 0))
        if dict(s.field) != dict(s1.field):
            found_diff = True
            break
    assert found_diff, "no seed in [1, 100) produced a different fall choice — RNG may be ignored"


def test_fall_zero_candidates_removes_position():
    state = make_state(
        field={(0, 0): ["red"]},
        reserve=[[PlainBucket("red")]],
        bucket_capacity=1,
        color_set=frozenset({"red"}),
    )
    step(state, (0, 0))
    assert (0, 0) not in state.field


def test_fall_multiple_clears_same_tick_independent_rolls():
    # Two distinct-color stacks both clear in the same tick. Each fall is
    # 2-candidate, so RNG advances exactly twice.
    state = make_state(
        field={
            (0, 0): ["red"], (1, 0): ["red"],
            (2, 0): ["blue"], (3, 0): ["blue"],
            (0, 1): ["red"],
            (2, 1): ["blue"],
        },
        buffer=[
            BufferBucket("red", 1, 0),
            BufferBucket("blue", 1, 0),
            None, None, None,
        ],
        reserve=[[None]],
        bucket_capacity=1,
        color_set=frozenset({"red", "blue"}),
        seed=42,
    )
    run_until_quiescent(state)

    # Compare to a fresh Random(42) advanced by exactly 2 randint calls.
    expected = random.Random(42)
    expected.randint(0, 1)
    expected.randint(0, 1)
    assert state.rng.getstate() == expected.getstate()


# ===========================================================================
# Generator
# ===========================================================================

def test_generator_fires_on_load_when_facing_empty(tmp_path):
    data = {
        "meta": {"id": "gl1", "name": "GL1", "version": 1, "color_count": 2},
        "field": {"stacks": [{"col": 0, "row": 0, "slices": ["blue"]}]},
        "buffer": {"slots": 5, "bucket_capacity": 1},
        "reserve": {
            "rows": 1, "cols": 2,
            "cells": [
                {
                    "row": 0, "col": 1, "type": "generator",
                    "facing": "left", "remaining": 1, "queue": ["red"],
                }
            ],
        },
    }
    p = _write_level(tmp_path, data)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        state = load_level(p)
    assert state.reserve[0][0] == PlainBucket(color="red")
    gen = state.reserve[0][1]
    assert isinstance(gen, Generator)
    assert gen.remaining == 0
    assert gen.queue == []


def test_generator_does_not_fire_on_load_when_facing_occupied(tmp_path):
    data = {
        "meta": {"id": "gl2", "name": "GL2", "version": 1, "color_count": 2},
        "field": {"stacks": [{"col": 0, "row": 0, "slices": ["red"]}]},
        "buffer": {"slots": 5, "bucket_capacity": 1},
        "reserve": {
            "rows": 1, "cols": 2,
            "cells": [
                {"row": 0, "col": 0, "type": "plain_bucket", "color": "red"},
                {
                    "row": 0, "col": 1, "type": "generator",
                    "facing": "left", "remaining": 1, "queue": ["blue"],
                },
            ],
        },
    }
    p = _write_level(tmp_path, data)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        state = load_level(p)
    assert state.reserve[0][0] == PlainBucket(color="red")
    gen = state.reserve[0][1]
    assert gen.remaining == 1
    assert gen.queue == ["blue"]


def test_generator_fires_when_facing_freed_during_play():
    state = make_state(
        field={(0, 0): ["red"]},
        reserve=[[
            PlainBucket("red"),
            Generator(facing="left", remaining=1, queue=["blue"]),
        ]],
        bucket_capacity=1,
        color_set=frozenset({"red", "blue"}),
    )
    step(state, (0, 0))
    assert state.reserve[0][0] == PlainBucket(color="blue")
    gen = state.reserve[0][1]
    assert gen.remaining == 0


def test_exhausted_generator_blocks_reachability():
    state = make_state(
        reserve=[
            [None],
            [Generator(facing="up", remaining=0, queue=[])],
            [PlainBucket("red")],
        ],
    )
    reach = compute_reachability(state)
    assert reach[2][0] is False


@pytest.mark.parametrize("facing,offset", [
    ("up", (-1, 0)),
    ("down", (1, 0)),
    ("left", (0, -1)),
    ("right", (0, 1)),
])
def test_generator_facing_directions(facing, offset):
    rows, cols = 3, 3
    reserve = [[None] * cols for _ in range(rows)]
    reserve[1][1] = Generator(facing=facing, remaining=1, queue=["red"])
    state = make_state(
        reserve=reserve,
        bucket_capacity=1,
        color_set=frozenset({"red"}),
    )
    run_until_quiescent(state)
    tr, tc = 1 + offset[0], 1 + offset[1]
    assert state.reserve[tr][tc] == PlainBucket(color="red")
    gen = state.reserve[1][1]
    assert gen.remaining == 0


# ===========================================================================
# Win and lose
# ===========================================================================

def test_win_minimal_level():
    state = load_level(TINY)
    info1 = step(state, (0, 0))
    assert info1["termination_reason"] is None
    info2 = step(state, (0, 1))
    assert info2["termination_reason"] == "win"
    assert is_terminal(state) == "win"


def test_deadlock_buffer_full_no_consumable():
    state = make_state(
        field={(0, 0): ["blue", "blue", "blue", "blue"]},
        buffer=[BufferBucket("red", 4, 0) for _ in range(5)],
        reserve=[[None]],
        bucket_capacity=4,
        color_set=frozenset({"red", "blue"}),
    )
    run_until_quiescent(state)
    assert is_terminal(state) == "deadlock"


def test_fallback_buffer_not_full_no_reachable():
    state = make_state(
        field={(0, 0): ["blue"]},  # blue slice; bucket is red, can't consume
        reserve=[[PlainBucket("red")]],
        bucket_capacity=4,
        color_set=frozenset({"red", "blue"}),
    )
    with pytest.warns(RuntimeWarning):
        info = step(state, (0, 0))
    assert info["termination_reason"] == "fallback"
    assert is_terminal(state) == "fallback"


# ===========================================================================
# Determinism and RNG isolation
# ===========================================================================

def test_replay_determinism():
    actions = [(0, 0), (0, 1)]
    s1 = load_level(TINY, seed=42)
    s2 = load_level(TINY, seed=42)
    assert dict(s1.field) == dict(s2.field)
    assert s1.buffer == s2.buffer
    assert s1.reserve == s2.reserve
    for a in actions:
        step(s1, a)
        step(s2, a)
        assert dict(s1.field) == dict(s2.field)
        assert s1.buffer == s2.buffer
        assert s1.reserve == s2.reserve


def test_rng_only_used_for_fall_direction():
    # Single-column field — every fall has at most 1 candidate, so no RNG.
    state = make_state(
        field={(0, 0): ["red"], (0, 1): ["red"], (0, 2): ["red"]},
        reserve=[[PlainBucket("red")]],
        bucket_capacity=3,
        color_set=frozenset({"red"}),
        seed=42,
    )
    fresh = random.Random(42)
    step(state, (0, 0))
    assert state.rng.getstate() == fresh.getstate(), \
        "RNG was consumed outside the 2-candidate fall path"


# ===========================================================================
# Tick ordering
# ===========================================================================

def test_pick_freeing_generator_facing_triggers_fire_same_step():
    # Generator's facing cell holds a bucket. Picking it must, within the same
    # step's auto-update loop, fire the generator into the freed cell.
    state = make_state(
        field={(0, 0): ["blue"]},
        reserve=[[
            PlainBucket("blue"),
            Generator(facing="left", remaining=1, queue=["red"]),
        ]],
        bucket_capacity=1,
        color_set=frozenset({"red", "blue"}),
    )
    step(state, (0, 0))
    assert state.reserve[0][0] == PlainBucket(color="red")
    gen = state.reserve[0][1]
    assert gen.remaining == 0
