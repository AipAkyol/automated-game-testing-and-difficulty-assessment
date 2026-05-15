import random
import warnings
from pathlib import Path

import pytest

from hexfall.game import (
    compute_reachability,
    is_terminal,
    legal_actions_mask,
    pin_blocked_cells,
    run_until_quiescent,
    step,
)
from hexfall.level_loader import load_level
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
    pins=None,
    buffer_slots=5,
    bucket_capacity=4,
    color_set=None,
    level_id="test",
    seed=0,
    move_counter=0,
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
    pins = list(pins) if pins else []

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
        pins=pins,
        move_counter=move_counter,
    )


def _silenced_load(path: str | Path, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return load_level(path, **kwargs)


# ===========================================================================
# Reachability — baseline
# ===========================================================================

def test_reachability_top_row_buckets():
    state = make_state(
        reserve=[[PlainBucket("r"), PlainBucket("b"), PlainBucket("g")]],
    )
    assert compute_reachability(state) == [[True, True, True]]


def test_reachability_top_row_wall_not_pickable():
    state = make_state(reserve=[[PlainBucket("r"), Wall(), PlainBucket("b")]])
    assert compute_reachability(state) == [[True, False, True]]


def test_reachability_through_top_row_empty():
    state = make_state(
        reserve=[
            [None, PlainBucket("r")],
            [PlainBucket("b"), PlainBucket("g")],
        ],
    )
    reach = compute_reachability(state)
    assert reach[0][0] is False
    assert reach[0][1] is True
    assert reach[1][0] is True
    assert reach[1][1] is False


def test_reachability_isolated_empty_does_not_grant_reachability():
    state = make_state(
        reserve=[
            [PlainBucket("r"), PlainBucket("r"), PlainBucket("r")],
            [PlainBucket("r"), None, PlainBucket("r")],
            [PlainBucket("r"), PlainBucket("r"), PlainBucket("r")],
        ],
    )
    reach = compute_reachability(state)
    assert reach[0] == [True, True, True]
    assert reach[1][0] is False
    assert reach[1][2] is False
    assert reach[2][1] is False


def test_reachability_wall_blocks_propagation():
    state = make_state(
        reserve=[[None], [Wall()], [PlainBucket("r")]],
    )
    assert compute_reachability(state)[2][0] is False

    state2 = make_state(
        reserve=[[None], [None], [PlainBucket("r")]],
    )
    assert compute_reachability(state2)[2][0] is True


def test_reachability_generator_blocks_propagation():
    state = make_state(
        reserve=[
            [None],
            [Generator(facing="up", remaining=2, queue=["r", "r"])],
            [PlainBucket("r")],
        ],
    )
    assert compute_reachability(state)[2][0] is False


def test_reachability_after_pick():
    state = make_state(
        field={},
        reserve=[[PlainBucket("r")], [PlainBucket("r")]],
        bucket_capacity=1,
        color_set=frozenset({"r"}),
    )
    assert compute_reachability(state)[1][0] is False
    info = step(state, (0, 0))
    assert info["termination_reason"] == "win"
    assert compute_reachability(state)[1][0] is True


# ===========================================================================
# Reachability — ice and pins
# ===========================================================================

def test_frozen_ice_bucket_is_reachable_but_not_pickable():
    state = make_state(
        reserve=[[
            PlainBucket("r"),
            IceBucket(row=0, col=1, color="b", thaw_threshold=2, thawed=False),
        ]],
        bucket_capacity=1,
        color_set=frozenset({"r", "b"}),
    )
    reach = compute_reachability(state)
    assert reach[0][1] is True   # top-row → reachable
    mask = legal_actions_mask(state)
    assert mask[0][0] is True
    assert mask[0][1] is False   # frozen → not pickable


def test_thawed_ice_bucket_is_pickable():
    state = make_state(
        reserve=[[
            IceBucket(row=0, col=0, color="r", thaw_threshold=1, thawed=True),
        ]],
        bucket_capacity=1,
        color_set=frozenset({"r"}),
    )
    assert legal_actions_mask(state)[0][0] is True


def test_pin_ray_cell_is_illegal_to_pick():
    """A plain bucket sitting under a pin ray is non-pickable regardless of reachability."""
    pin = Pin(origin_row=0, origin_col=0, direction="Right", block_count=1)
    state = make_state(
        reserve=[[PlainBucket("r"), PlainBucket("r")]],
        pins=[pin],
        bucket_capacity=1,
        color_set=frozenset({"r"}),
    )
    mask = legal_actions_mask(state)
    # Both top-row cells are inside the pin's ray (origin + 1) → illegal.
    assert mask[0][0] is False
    assert mask[0][1] is False


def test_pin_ray_blocks_reachability_propagation():
    """Pin ray cells act as opaque obstacles for reachability BFS."""
    # 3×1 reserve: empty top, pin in middle (covering middle cell), bucket at bottom.
    pin = Pin(origin_row=1, origin_col=0, direction="Down", block_count=0)
    state = make_state(
        reserve=[[None], [None], [PlainBucket("r")]],
        pins=[pin],
    )
    # Without pin, the bucket would be reachable.
    # With pin covering (1,0) and (2,0), the bucket itself is in the pin ray.
    assert pin_blocked_cells(state) == {(1, 0), (2, 0)}
    assert compute_reachability(state)[2][0] is False


# ===========================================================================
# Ice thaw — timing tests
# ===========================================================================

def test_ice_thaws_at_threshold():
    ice = IceBucket(row=0, col=1, color="b", thaw_threshold=1, thawed=False)
    state = make_state(
        field={(0, 0): ["r"]},
        reserve=[[PlainBucket("r"), ice]],
        bucket_capacity=1,
        color_set=frozenset({"r", "b"}),
    )
    # Before any pick: frozen.
    assert ice.thawed is False
    step(state, (0, 0))
    # After 1 pick, ice_thaw_phase runs at start of next tick — threshold met → thawed.
    assert ice.thawed is True


def test_ice_does_not_thaw_before_threshold():
    ice = IceBucket(row=0, col=1, color="b", thaw_threshold=3, thawed=False)
    state = make_state(
        field={(0, 0): ["r"]},
        reserve=[[PlainBucket("r"), ice]],
        bucket_capacity=1,
        color_set=frozenset({"r", "b"}),
    )
    step(state, (0, 0))
    assert state.move_counter == 1
    assert ice.thawed is False


def test_ice_test_level_solves_deterministically():
    state = _silenced_load(LEVELS_DIR / "ice_test.json", seed=0)

    # Only (0, 0) plain bucket is legal at load.
    assert legal_actions_mask(state)[0][0] is True
    assert legal_actions_mask(state)[0][1] is False
    assert legal_actions_mask(state)[0][2] is False

    step(state, (0, 0))
    # Ice at (0, 1) thaws (threshold 1); (0, 2) still frozen.
    assert legal_actions_mask(state)[0][1] is True
    assert legal_actions_mask(state)[0][2] is False

    step(state, (0, 1))
    # Ice at (0, 2) thaws (threshold 2).
    assert legal_actions_mask(state)[0][2] is True

    info = step(state, (0, 2))
    assert info["termination_reason"] == "win"


# ===========================================================================
# Pin destruction
# ===========================================================================

def test_pin_destroyed_when_destruction_cell_emptied():
    """Pin facing Right at (0, 1): destruction cell = (0, 0). Picking (0, 0) destroys it."""
    pin = Pin(origin_row=0, origin_col=1, direction="Right", block_count=1)
    state = make_state(
        field={(0, 0): ["r"]},
        reserve=[[PlainBucket("r"), None, None]],
        pins=[pin],
        bucket_capacity=1,
        color_set=frozenset({"r"}),
    )
    # Pin's ray initially covers (0, 1), (0, 2).
    assert pin_blocked_cells(state) == {(0, 1), (0, 2)}
    step(state, (0, 0))
    # After pick, destruction cell (0, 0) is None → pin destroys.
    assert pin.destroyed is True
    assert pin_blocked_cells(state) == set()


def test_pin_not_destroyed_when_generator_refills_destruction_cell():
    """Refill protection: generator firing into destruction cell on same tick blocks destruction."""
    # Pin facing Right at (0, 1), destruction cell (0, 0).
    pin = Pin(origin_row=0, origin_col=1, direction="Right", block_count=1)
    # Generator at (1, 0) facing up to (0, 0). After pick of (0, 0), the cell is empty
    # → generator fires into it (step 2.v) before pin destruction check (step 2.vii).
    state = make_state(
        field={(0, 0): ["b"]},
        reserve=[
            [PlainBucket("r"), None, None],
            [Generator(facing="up", remaining=1, queue=["b"]), None, None],
        ],
        pins=[pin],
        bucket_capacity=1,
        color_set=frozenset({"r", "b"}),
    )
    step(state, (0, 0))
    # Generator refilled (0, 0). Pin should NOT be destroyed.
    assert pin.destroyed is False
    assert isinstance(state.reserve[0][0], PlainBucket)
    assert state.reserve[0][0].color == "b"


def test_two_pins_cascade_in_single_tick():
    """Destroying pin A clears a cell that is pin B's destruction cell → B also destroys."""
    state = _silenced_load(LEVELS_DIR / "pin_test.json", seed=0)
    pin_a, pin_b = state.pins
    assert pin_a.destroyed is False
    assert pin_b.destroyed is False
    info = step(state, (1, 2))  # picks plain bucket at (row=1, col=2) — pin A's destruction cell
    # Both pins must end this step destroyed.
    assert pin_a.destroyed is True
    assert pin_b.destroyed is True
    assert info["termination_reason"] == "win"


def test_pin_ray_block_count_zero_extends_to_grid_edge():
    """blockCount=0 means the ray extends from origin to the grid edge in the facing direction."""
    pin = Pin(origin_row=0, origin_col=0, direction="Right", block_count=0)
    state = make_state(
        reserve=[[None, None, None, None]],
        pins=[pin],
    )
    assert pin_blocked_cells(state) == {(0, 0), (0, 1), (0, 2), (0, 3)}


def test_pin_ray_block_count_n_covers_n_plus_1_cells():
    pin = Pin(origin_row=0, origin_col=0, direction="Right", block_count=2)
    state = make_state(
        reserve=[[None, None, None, None, None]],
        pins=[pin],
    )
    assert pin_blocked_cells(state) == {(0, 0), (0, 1), (0, 2)}


# ===========================================================================
# Buffer pulls and same-color collision
# ===========================================================================

def test_pull_drains_matching_stack_over_ticks():
    state = make_state(
        field={(0, 0): ["r", "r", "r", "r"]},
        reserve=[[PlainBucket("r")]],
        bucket_capacity=4,
        color_set=frozenset({"r"}),
    )
    info = step(state, (0, 0))
    assert all(slot is None for slot in state.buffer)
    assert all(not s for s in state.field.values())
    assert info["termination_reason"] == "win"


def test_same_color_fuller_pulls_first():
    fuller = BufferBucket(color="r", capacity=10, fill=5)
    less = BufferBucket(color="r", capacity=10, fill=1)
    state = make_state(
        field={(0, 0): ["r", "r", "r", "r"]},
        buffer=[fuller, less, None, None, None],
        reserve=[[None]],
        bucket_capacity=10,
        color_set=frozenset({"r"}),
    )
    run_until_quiescent(state)
    assert state.buffer[0] == BufferBucket(color="r", capacity=10, fill=9)
    assert state.buffer[1] == BufferBucket(color="r", capacity=10, fill=1)


def test_distinct_colors_pull_same_tick():
    red = BufferBucket(color="r", capacity=1, fill=0)
    blue = BufferBucket(color="b", capacity=1, fill=0)
    state = make_state(
        field={(0, 0): ["r"], (1, 0): ["b"]},
        buffer=[red, blue, None, None, None],
        reserve=[[None, None]],
        bucket_capacity=1,
        color_set=frozenset({"r", "b"}),
    )
    run_until_quiescent(state)
    assert state.buffer[0] is None
    assert state.buffer[1] is None
    assert all(not s for s in state.field.values())


# ===========================================================================
# Stack clear and fall
# ===========================================================================

def test_fall_one_candidate_does_not_call_rng():
    state = make_state(
        field={(0, 0): ["r"], (0, 1): ["r"]},
        reserve=[[PlainBucket("r")]],
        bucket_capacity=2,
        color_set=frozenset({"r"}),
        seed=42,
    )
    rng_before = state.rng.getstate()
    step(state, (0, 0))
    assert state.rng.getstate() == rng_before


def _make_two_candidate_state(seed):
    return make_state(
        field={(0, 0): ["r"], (1, 0): ["b"], (0, 1): ["r"]},
        reserve=[[PlainBucket("r")]],
        bucket_capacity=1,
        color_set=frozenset({"r", "b"}),
        seed=seed,
    )


def test_fall_two_candidates_uses_rng_deterministically():
    s1 = _make_two_candidate_state(seed=42)
    s2 = _make_two_candidate_state(seed=42)
    step(s1, (0, 0))
    step(s2, (0, 0))
    assert dict(s1.field) == dict(s2.field)

    found_diff = False
    for alt in range(1, 100):
        if alt == 42:
            continue
        s = _make_two_candidate_state(seed=alt)
        step(s, (0, 0))
        if dict(s.field) != dict(s1.field):
            found_diff = True
            break
    assert found_diff


# ===========================================================================
# Generator
# ===========================================================================

def test_generator_fires_when_facing_freed_during_play():
    state = make_state(
        field={(0, 0): ["r"]},
        reserve=[[
            PlainBucket("r"),
            Generator(facing="left", remaining=1, queue=["b"]),
        ]],
        bucket_capacity=1,
        color_set=frozenset({"r", "b"}),
    )
    step(state, (0, 0))
    assert state.reserve[0][0] == PlainBucket(color="b")
    assert state.reserve[0][1].remaining == 0


def test_exhausted_generator_blocks_reachability():
    state = make_state(
        reserve=[
            [None],
            [Generator(facing="up", remaining=0, queue=[])],
            [PlainBucket("r")],
        ],
    )
    assert compute_reachability(state)[2][0] is False


@pytest.mark.parametrize("facing,offset", [
    ("up", (-1, 0)),
    ("down", (1, 0)),
    ("left", (0, -1)),
    ("right", (0, 1)),
])
def test_generator_facing_directions(facing, offset):
    rows, cols = 3, 3
    reserve = [[None] * cols for _ in range(rows)]
    reserve[1][1] = Generator(facing=facing, remaining=1, queue=["r"])
    state = make_state(
        reserve=reserve,
        bucket_capacity=1,
        color_set=frozenset({"r"}),
    )
    run_until_quiescent(state)
    tr, tc = 1 + offset[0], 1 + offset[1]
    assert state.reserve[tr][tc] == PlainBucket(color="r")
    assert state.reserve[1][1].remaining == 0


def test_generator_fires_on_load_when_facing_empty():
    # generator_test.json has generator at (y=0, x=1) facing Down into (y=1, x=1) which is empty.
    state = _silenced_load(LEVELS_DIR / "generator_test.json", seed=0)
    assert state.reserve[1][1] == PlainBucket(color="g")
    gen = state.reserve[0][1]
    assert isinstance(gen, Generator)
    assert gen.remaining == 1
    assert gen.queue == ["g"]


# ===========================================================================
# Win and lose
# ===========================================================================

def test_win_tiny_solvable():
    state = _silenced_load(TINY)
    info1 = step(state, (0, 0))
    assert info1["termination_reason"] is None
    info2 = step(state, (0, 1))
    assert info2["termination_reason"] == "win"
    assert is_terminal(state) == "win"


def test_deadlock_buffer_full_no_consumable():
    state = make_state(
        field={(0, 0): ["b", "b", "b", "b"]},
        buffer=[BufferBucket("r", 4, 0) for _ in range(5)],
        reserve=[[None]],
        bucket_capacity=4,
        color_set=frozenset({"r", "b"}),
    )
    run_until_quiescent(state)
    assert is_terminal(state) == "deadlock"


def test_fallback_buffer_not_full_no_reachable():
    state = make_state(
        field={(0, 0): ["b"]},
        reserve=[[PlainBucket("r")]],
        bucket_capacity=4,
        color_set=frozenset({"r", "b"}),
    )
    with pytest.warns(RuntimeWarning):
        info = step(state, (0, 0))
    assert info["termination_reason"] == "fallback"


# ===========================================================================
# Determinism, RNG isolation, and move counter
# ===========================================================================

def test_replay_determinism():
    actions = [(0, 0), (0, 1)]
    s1 = _silenced_load(TINY, seed=42)
    s2 = _silenced_load(TINY, seed=42)
    assert dict(s1.field) == dict(s2.field)
    assert s1.reserve == s2.reserve
    for a in actions:
        step(s1, a)
        step(s2, a)
        assert dict(s1.field) == dict(s2.field)
        assert s1.reserve == s2.reserve


def test_move_counter_increments_per_action():
    state = _silenced_load(TINY)
    assert state.move_counter == 0
    step(state, (0, 0))
    assert state.move_counter == 1
    step(state, (0, 1))
    assert state.move_counter == 2


def test_move_counter_unaffected_by_automatic_updates():
    """Generator firing on load is an automatic update, not a player action."""
    state = _silenced_load(LEVELS_DIR / "generator_test.json")
    # Generator fired on load → automatic update, not a player action.
    assert state.move_counter == 0


def test_rng_only_used_for_fall_direction():
    state = make_state(
        field={(0, 0): ["r"], (0, 1): ["r"], (0, 2): ["r"]},
        reserve=[[PlainBucket("r")]],
        bucket_capacity=3,
        color_set=frozenset({"r"}),
        seed=42,
    )
    fresh = random.Random(42)
    step(state, (0, 0))
    assert state.rng.getstate() == fresh.getstate()


# ===========================================================================
# Tick ordering
# ===========================================================================

def test_pick_freeing_generator_facing_triggers_fire_same_step():
    state = make_state(
        field={(0, 0): ["b"]},
        reserve=[[
            PlainBucket("b"),
            Generator(facing="left", remaining=1, queue=["r"]),
        ]],
        bucket_capacity=1,
        color_set=frozenset({"r", "b"}),
    )
    step(state, (0, 0))
    assert state.reserve[0][0] == PlainBucket(color="r")
    assert state.reserve[0][1].remaining == 0
