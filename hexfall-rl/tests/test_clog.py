"""Tests for the Clog difficulty-feature extractors (Issue H Phase 2)."""
from __future__ import annotations

import random
from pathlib import Path

import pytest

from hexfall.clog import ctd, extract_clog_features, fcc, rhc
from hexfall.level_loader import load_level
from hexfall.types import (
    GameState,
    Generator,
    IceBucket,
    PlainBucket,
    QuestionBucket,
)

REPO = Path(__file__).resolve().parents[1]
# Repo-tracked fixture (NOT a classified Paxie level) — two monochrome stacks
# ["r","r","r","r"] and ["b","b","b","b"], so it is a monochrome-per-stack level.
TINY = REPO / "levels" / "tiny_solvable.json"


def _make_state(field, reserve) -> GameState:
    """Build a GameState directly (no JSON load, no quiescence) for exact-value tests."""
    rows = len(reserve)
    cols = len(reserve[0]) if reserve else 0
    return GameState(
        field=dict(field),
        buffer_slots=5,
        bucket_capacity=24,
        buffer=[None] * 5,
        reserve_rows=rows,
        reserve_cols=cols,
        reserve=reserve,
        color_set=frozenset(),  # unused by clog (C is recomputed from contents)
        level_id="clog-test",
        rng=random.Random(0),
        quiescent=True,
        pins=[],
        move_counter=0,
    )


def _rich_state() -> GameState:
    """A hand-built state with hand-computable feature values.

    field (col,row)->top-to-bottom slices:
        (0,0)=["r","r"]       h2, 0 transitions  -> ctd ratio 0/1 = 0.0
        (0,1)=["r","b","r"]   h3, 2 transitions  -> ctd ratio 2/2 = 1.0
        (1,1)=["g"]           h1, excluded from ctd
      bottom_row = 1; bottom-row stacks (0,1),(1,1) tops {"r","g"} = 2 distinct.
    reserve: plain r, ?-bucket b, frozen ice g, generator queue ["y","y"].
      C (oracle color_count) = {r,b,g} (field) U {r,b,g,y} (reserve) = 4.
      fcc = 2/4 = 0.5
      ctd = mean(0.0, 1.0) = 0.5
      rhc = (wood 1 + frozen-ice 1 + gen-queued 2) / (plain 1 + wood 1 + ice 1 + gen-queued 2)
          = 4/5 = 0.8
    """
    field = {
        (0, 0): ["r", "r"],
        (0, 1): ["r", "b", "r"],
        (1, 1): ["g"],
    }
    reserve = [
        [PlainBucket("r"), QuestionBucket("b", revealed=False),
         IceBucket(row=0, col=2, color="g", thaw_threshold=5, thawed=False)],
        [Generator(facing="up", remaining=2, queue=["y", "y"]), None, None],
    ]
    return _make_state(field, reserve)


def test_extract_returns_expected_keys_and_types():
    feats = extract_clog_features(load_level(TINY, seed=0))
    assert set(feats.keys()) == {"fcc", "ctd", "rhc"}
    assert all(isinstance(v, float) for v in feats.values())


def test_all_features_in_unit_interval():
    for state in (load_level(TINY, seed=0), _rich_state()):
        feats = extract_clog_features(state)
        for name, value in feats.items():
            assert 0.0 <= value <= 1.0, f"{name}={value} out of [0,1]"


def test_ctd_zero_on_monochrome_per_stack():
    # tiny_solvable: every stack is a single colour -> zero transitions everywhere.
    assert ctd(load_level(TINY, seed=0)) == 0.0
    # And a hand-built monochrome-per-stack state (mixed heights, still no transitions).
    mono = _make_state(
        {(0, 0): ["r", "r", "r"], (1, 0): ["b", "b"], (2, 0): ["g"]},
        [[PlainBucket("r"), PlainBucket("b"), PlainBucket("g")]],
    )
    assert ctd(mono) == 0.0


def test_extract_is_deterministic():
    a = extract_clog_features(load_level(TINY, seed=0))
    b = extract_clog_features(load_level(TINY, seed=0))
    assert a == b


def test_known_values_hand_built_state():
    state = _rich_state()
    assert fcc(state) == pytest.approx(0.5)
    assert ctd(state) == pytest.approx(0.5)
    assert rhc(state) == pytest.approx(0.8)
