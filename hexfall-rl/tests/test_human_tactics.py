"""Tests for HumanTacticsPlayer (Issue G).

Covers the Player-protocol contract, determinism, end-to-end runs on the tiny
hand-built level and a real Paxie level, and a constructed case showing an
immediately-matched pick beats a speculative one.
"""
from __future__ import annotations

import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from hexfall.env import HexFallEnv
from hexfall.players import HumanTacticsPlayer, Player, evaluate
from hexfall.players.human_tactics import (
    _bottom_row_top_colors,
    _concrete_field_colors,
    _sentinels,
)

REPO = Path(__file__).resolve().parents[1]
LEVELS_DIR = REPO / "levels"
TINY = LEVELS_DIR / "tiny_solvable.json"
# Hand-built fixture: at reset exactly two actions are legal — the destruction
# cell of a Down-facing pin (a pickable 'r' bucket) and an immediate bottom-row
# 'b' match. Used to assert pin_setup (+4) outranks an available match (+3).
PIN_FIXTURE = LEVELS_DIR / "pin_vs_match_fixture.json"
# A real (CLASSIFIED) mid-difficulty Paxie level; skipped if the data is absent.
PAXIE_LEVEL = REPO / "CLASSIFIED.paxie_data" / "level_data" / "level50.json"


@pytest.fixture(autouse=True)
def _silence_parity_warnings():
    """Hand-built levels emit slice-bucket-parity UserWarnings on load; ignore them."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        yield


def _make_obs(
    *,
    field_visible,
    field_heights,
    reserve_color,
    reserve_cell_type,
    action_mask,
    buffer_occupied=None,
    buffer_colors=None,
    buffer_fills=None,
    buffer_capacities=None,
):
    """Build a minimal obs dict covering every key HumanTacticsPlayer.act reads.

    Defaults give an empty 5-slot buffer, no generators, no ice, and no pins.
    """
    i32 = np.int32
    fv = np.asarray(field_visible, dtype=i32)
    fh = np.asarray(field_heights, dtype=i32)
    rc = np.asarray(reserve_color, dtype=i32)
    R, C = rc.shape
    no_id = int(max(fv.max(), rc.max())) + 1  # NO_COLOR_ID for the empty buffer
    if buffer_occupied is None:
        buffer_occupied = np.zeros(5, dtype=i32)
    if buffer_colors is None:
        buffer_colors = np.full(5, no_id, dtype=i32)
    if buffer_fills is None:
        buffer_fills = np.zeros(5, dtype=i32)
    if buffer_capacities is None:
        buffer_capacities = np.full(5, 24, dtype=i32)
    return {
        "field_visible": fv,
        "field_heights": fh,
        "buffer_occupied": np.asarray(buffer_occupied, dtype=i32),
        "buffer_colors": np.asarray(buffer_colors, dtype=i32),
        "buffer_fills": np.asarray(buffer_fills, dtype=i32),
        "buffer_capacities": np.asarray(buffer_capacities, dtype=i32),
        "reserve_cell_type": np.asarray(reserve_cell_type, dtype=i32),
        "reserve_color": rc,
        "reserve_generator_facing": np.zeros((R, C), dtype=i32),
        "reserve_generator_remaining": np.zeros((R, C), dtype=i32),
        "reserve_remaining_thaw": np.zeros((R, C), dtype=i32),
        "action_mask": np.asarray(action_mask, dtype=i32),
        "pins_origin_row": np.zeros(0, dtype=i32),
        "pins_origin_col": np.zeros(0, dtype=i32),
        "pins_direction": np.zeros(0, dtype=i32),
        "pins_destroyed": np.zeros(0, dtype=i32),
    }


# ---------------------------------------------------------------------------
# Test 1: protocol compliance + returns a legal action for a valid obs
# ---------------------------------------------------------------------------

def test_human_tactics_implements_protocol_and_returns_legal_action():
    assert isinstance(HumanTacticsPlayer(), Player)
    env = HexFallEnv(level_path=str(TINY))
    obs, _ = env.reset(seed=0)
    action = HumanTacticsPlayer().act(obs, env)
    assert isinstance(action, int)
    assert obs["action_mask"][action] == 1


# ---------------------------------------------------------------------------
# Test 2: determinism — same obs (and env) => same action
# ---------------------------------------------------------------------------

def test_human_tactics_is_deterministic():
    env = HexFallEnv(level_path=str(TINY))
    obs, _ = env.reset(seed=0)
    p = HumanTacticsPlayer()
    first = p.act(obs, env)
    assert all(p.act(obs, env) == first for _ in range(5))
    # A fresh instance with no shared state agrees too.
    assert HumanTacticsPlayer().act(obs, env) == first


# ---------------------------------------------------------------------------
# Test 3: end-to-end on tiny_solvable — completes, all actions legal, wins
# ---------------------------------------------------------------------------

def test_human_tactics_solves_tiny_solvable():
    # evaluate() steps every returned action through the env, which raises on an
    # illegal action; a clean float return therefore proves no illegal action.
    wr = evaluate(HumanTacticsPlayer(), TINY, n_episodes=10, seed=42)
    assert isinstance(wr, float)
    assert 0.0 <= wr <= 1.0
    assert wr == 1.0  # tiny_solvable is always solvable


# ---------------------------------------------------------------------------
# Test 4: end-to-end on a real Paxie level — completes without illegal actions
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not PAXIE_LEVEL.exists(), reason="CLASSIFIED Paxie data not present")
def test_human_tactics_runs_on_real_paxie_level():
    wr = evaluate(HumanTacticsPlayer(), PAXIE_LEVEL, n_episodes=3, seed=0)
    assert isinstance(wr, float)
    assert 0.0 <= wr <= 1.0


# ---------------------------------------------------------------------------
# Test 5: constructed case — an immediately-matched pick beats a speculative one
# ---------------------------------------------------------------------------

def test_matched_pick_beats_speculative_pick():
    # 2 colours (0, 1); the field's single bottom stack shows colour 0 only, so
    # colour 1 appears nowhere visible. NO_COLOR_ID = 3 marks padding/empty.
    obs = _make_obs(
        field_visible=[[[0], [3]]],          # (0,0) top=colour 0; (0,1) empty
        field_heights=[[1, 0]],
        reserve_color=[[0, 1]],              # action 0 -> colour 0; action 1 -> colour 1
        reserve_cell_type=[[1, 1]],          # both plain buckets
        action_mask=[1, 1],
    )
    player = HumanTacticsPlayer()
    # The matched pick (action 0) wins.
    assert player.act(obs, None) == 0

    hidden_id, no_id = _sentinels(obs, None)
    concrete = _concrete_field_colors(obs, hidden_id, no_id)
    assert concrete == {0}  # colour 1 is invisible -> speculative

    ctx = {
        "bottom_counts": Counter([0]),
        "buffer_buckets": [],
        "post_occ": 1 / 5,
        "concrete_field": concrete,
        "gen_targets": set(),
        "pin_dest": {},
        "ice_blocked": False,
        "nearest_thaw": 0,
    }
    matched = player._component_values(0, 0, 0, ctx)
    speculative = player._component_values(1, 0, 1, ctx)
    assert matched["matched_now"] == 1.0 and matched["speculation"] == 0.0
    assert speculative["matched_now"] == 0.0 and speculative["speculation"] > 0.0
    assert player._score(0, 0, 0, ctx) > player._score(1, 0, 1, ctx)


# ---------------------------------------------------------------------------
# Test 6: pin_setup (+4) outranks an available immediate match (+3)
# ---------------------------------------------------------------------------

def test_pin_setup_destruction_cell_beats_match():
    # game.py's actual offset map locates the destruction cell — no
    # reimplementation in the test, so this also guards the player's own copy.
    from hexfall.game import _PIN_OPPOSITE_OFFSET

    env = HexFallEnv(level_path=str(PIN_FIXTURE))
    obs, _ = env.reset(seed=0)
    n_cols = obs["reserve_color"].shape[1]

    pins = env._state.pins
    assert len(pins) == 1, "fixture must have exactly one pin"
    pin = pins[0]
    drow, dcol = _PIN_OPPOSITE_OFFSET[pin.direction]
    dest_rc = (pin.origin_row + drow, pin.origin_col + dcol)
    dest_action = dest_rc[0] * n_cols + dest_rc[1]

    # The fixture must genuinely offer both options: the destruction-cell pick
    # and at least one *other* legal pick whose colour is consumable now (a +3
    # immediate match). Otherwise the assertion below would be vacuous.
    mask = obs["action_mask"]
    legal = [a for a in range(mask.shape[0]) if mask[a]]
    assert dest_action in legal, "destruction cell must be a legal pick"
    bottom_tops = set(_bottom_row_top_colors(obs))
    match_actions = [
        a for a in legal
        if a != dest_action and int(obs["reserve_color"][divmod(a, n_cols)]) in bottom_tops
    ]
    assert match_actions, "fixture must also present an immediate-match pick"

    # pin_setup (+4) must beat the immediate match (+3).
    chosen = HumanTacticsPlayer().act(obs, env)
    assert chosen == dest_action

    # Cross-check via the real engine: stepping the chosen action destroys the
    # pin, which proves the cell genuinely was the destruction cell.
    obs_after, _, _, _, _ = env.step(chosen)
    assert obs_after["pins_destroyed"][0] == 1
