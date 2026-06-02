"""Tests for the difficulty oracle (Issue C)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hexfall.level_loader import load_level
from hexfall.oracle import (
    EXPANDED_FEATURE_NAMES,
    FEATURE_NAMES,
    STRUCTURAL_FEATURE_NAMES,
    Oracle,
    extract_expanded_features,
    extract_structural_features,
)

REPO = Path(__file__).resolve().parents[1]
# A repo-tracked fixture (NOT a classified Paxie level) keeps the suite fast:
# MCTS-in-predict stays sub-second on this tiny level.
TINY_LEVEL = REPO / "levels" / "tiny_solvable.json"
ICE_LEVEL = REPO / "levels" / "ice_test.json"
WALL_LEVEL = REPO / "levels" / "wall_test.json"
GENERATOR_LEVEL = REPO / "levels" / "generator_test.json"
PIN_LEVEL = REPO / "levels" / "pin_test.json"


def _make_records(n: int = 12):
    """Synthetic fit records: one tiny loaded level + 3 winrates each."""
    state = load_level(TINY_LEVEL)
    records, targets = [], []
    for i in range(n):
        # vary player winrates and target deterministically (no RNG)
        g = (i % 4) / 4.0
        records.append(
            {
                "state": state,
                "greedy_winrate": g,
                "lookahead_winrate": min(1.0, g + 0.1),
                "mcts_winrate": min(1.0, g + 0.2),
            }
        )
        targets.append(min(1.0, 0.3 + g * 0.5))
    return records, targets


def test_structural_extractor_returns_five():
    state = load_level(TINY_LEVEL)
    feats = extract_structural_features(state)
    assert len(feats) == 5
    assert len(STRUCTURAL_FEATURE_NAMES) == 5


def test_full_feature_vector_is_eight():
    records, _ = _make_records(1)
    vec = Oracle._record_to_features(records[0])
    assert len(vec) == 8
    assert len(FEATURE_NAMES) == 8


def test_fit_is_deterministic():
    records, targets = _make_records()
    o1 = Oracle().fit(records, targets)
    o2 = Oracle().fit(records, targets)
    c1 = np.array([o1.coefficients[n] for n in FEATURE_NAMES])
    c2 = np.array([o2.coefficients[n] for n in FEATURE_NAMES])
    np.testing.assert_array_equal(c1, c2)


def test_predict_clamped_to_unit_interval():
    records, targets = _make_records()
    oracle = Oracle().fit(records, targets)
    pred = oracle.predict(TINY_LEVEL)
    assert 0.0 <= pred <= 1.0


def test_predict_is_deterministic():
    records, targets = _make_records()
    oracle = Oracle().fit(records, targets)
    p1 = oracle.predict(TINY_LEVEL)
    p2 = oracle.predict(TINY_LEVEL)
    assert p1 == p2


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError):
        Oracle().predict(TINY_LEVEL)


# --- Issue I: expanded structural-count features -----------------------------

def test_expanded_extractor_returns_eight_keys_with_numeric_types():
    feats = extract_expanded_features(load_level(TINY_LEVEL))
    assert set(feats.keys()) == set(EXPANDED_FEATURE_NAMES)
    assert len(EXPANDED_FEATURE_NAMES) == 8
    assert all(isinstance(v, (int, float)) for v in feats.values())


def test_expanded_features_are_non_negative():
    for level in (TINY_LEVEL, ICE_LEVEL, WALL_LEVEL, GENERATOR_LEVEL, PIN_LEVEL):
        feats = extract_expanded_features(load_level(level))
        assert all(v >= 0 for v in feats.values()), level


def test_expanded_extractor_is_deterministic():
    # Same level loaded twice must yield an identical dict.
    f1 = extract_expanded_features(load_level(GENERATOR_LEVEL))
    f2 = extract_expanded_features(load_level(GENERATOR_LEVEL))
    assert f1 == f2


def test_expanded_features_hand_checked_values():
    # tiny_solvable: 2 plain buckets, two height-4 field stacks, no mechanics.
    tiny = extract_expanded_features(load_level(TINY_LEVEL))
    assert tiny["generator_count"] == 0
    assert tiny["wall_count"] == 0
    assert tiny["woodbox_count"] == 0
    assert tiny["reserve_bucket_count"] == 2
    assert tiny["generator_queue_total_length"] == 0
    assert tiny["avg_field_stack_height"] == 4.0
    assert tiny["max_field_stack_height"] == 4
    assert tiny["mechanic_diversity"] == 1  # plain only

    # ice_test: 1 plain + 2 ice buckets, one height-3 field stack.
    ice = extract_expanded_features(load_level(ICE_LEVEL))
    assert ice["reserve_bucket_count"] == 3  # plain + ice counted as pickable
    assert ice["wall_count"] == 0
    assert ice["generator_count"] == 0
    assert ice["mechanic_diversity"] == 2  # plain + ice

    # wall_test: 6 plain + 1 wall, two height-6 field stacks.
    wall = extract_expanded_features(load_level(WALL_LEVEL))
    assert wall["wall_count"] == 1
    assert wall["reserve_bucket_count"] == 6  # walls are NOT pickable buckets
    assert wall["mechanic_diversity"] == 2  # plain + wall
    assert wall["max_field_stack_height"] == 6

    # generator_test: 1 generator (remaining=1 post-quiescence) + 3 plain.
    gen = extract_expanded_features(load_level(GENERATOR_LEVEL))
    assert gen["generator_count"] == 1
    assert gen["generator_queue_total_length"] == 1
    assert gen["mechanic_diversity"] == 2  # plain + generator

    # pin_test: 2 plain buckets + 2 pins overlaying the grid.
    pin = extract_expanded_features(load_level(PIN_LEVEL))
    assert pin["generator_count"] == 0
    assert pin["wall_count"] == 0
    assert pin["mechanic_diversity"] == 2  # plain + pin


def test_structural_only_path_unchanged_by_expanded():
    # The default Oracle assembly must remain the canonical 8-vector; the
    # expanded block only appears when explicitly opted in.
    records, _ = _make_records(1)
    default_vec = Oracle._record_to_features(records[0])
    assert len(default_vec) == 8
    assert Oracle().include_expanded is False

    expanded_vec = Oracle._record_to_features(records[0], include_expanded=True)
    assert len(expanded_vec) == 8 + 8  # 5 structural + 8 expanded + 3 player
    # The default vector is a prefix-preserving subset: structural block then
    # player block, with the expanded block spliced between in the opt-in form.
    assert expanded_vec[:5] == default_vec[:5]           # structural unchanged
    assert expanded_vec[-3:] == default_vec[-3:]         # player block unchanged
