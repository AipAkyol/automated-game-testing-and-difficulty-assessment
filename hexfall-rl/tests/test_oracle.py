"""Tests for the difficulty oracle (Issue C)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hexfall.level_loader import load_level
from hexfall.oracle import (
    FEATURE_NAMES,
    STRUCTURAL_FEATURE_NAMES,
    Oracle,
    extract_structural_features,
)

REPO = Path(__file__).resolve().parents[1]
# A repo-tracked fixture (NOT a classified Paxie level) keeps the suite fast:
# MCTS-in-predict stays sub-second on this tiny level.
TINY_LEVEL = REPO / "levels" / "tiny_solvable.json"


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
