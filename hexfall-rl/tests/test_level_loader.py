import copy
import json
import warnings
from pathlib import Path

import pytest

from hexfall.level_loader import (
    LevelLoadError,
    UnsupportedMechanicError,
    load_level,
    load_level_from_data,
)
from hexfall.types import (
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
# Helpers
# ---------------------------------------------------------------------------

def _minimal_paxie() -> dict:
    """Smallest valid Paxie-format level — 1 plain bucket + 1 slice."""
    return {
        "levelNumber": 1,
        "levelVersionCode": 1,
        "collectorArea": {
            "gridWidth": 1,
            "gridHeight": 1,
            "singleBlockCollectors": [{"x": 0, "y": 0, "color": "r"}],
            "woodBoxCollectors": [],
            "iceCollectors": [],
            "deadCells": [],
            "tunnels": [],
            "pinBlockers": [],
            "mysteryCollectors": [],
            "tiedPairs": [],
            "keyLocks": [],
        },
        "hexStackArea": {
            "gridWidth": 1,
            "gridHeight": 1,
            "stacks": [{"x": 0, "y": 0, "colors": ["r"]}],
            "tunnels": [],
        },
        "editorMeta": {
            "totalBlocks": 1,
            "colorCount": 1,
            "maxColorsPerStack": 1,
            "heightMin": 1,
            "heightMax": 1,
            "randomness": 0.0,
            "verticalPercent": 0.0,
            "horizontalPercent": 0.0,
            "mysteryPercent": 0.0,
        },
    }


def _silenced(data: dict, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return load_level_from_data(data, **kwargs)


# ===========================================================================
# Basic loading
# ===========================================================================

def test_load_tiny_solvable_file():
    state = _silenced(json.loads(TINY.read_text()), level_id="tiny_solvable")

    assert state.reserve_rows == 1
    assert state.reserve_cols == 2
    assert state.reserve[0][0] == PlainBucket(color="r")
    assert state.reserve[0][1] == PlainBucket(color="b")
    assert state.color_set == frozenset({"r", "b"})
    assert state.level_id == "tiny_solvable"
    assert state.quiescent is True
    assert state.move_counter == 0
    assert state.pins == []


def test_seed_determinism():
    s1 = load_level(TINY, seed=42)
    s2 = load_level(TINY, seed=42)
    assert s1.rng.randint(0, 1_000_000) == s2.rng.randint(0, 1_000_000)


def test_seed_none_independent():
    s1 = load_level(TINY, seed=None)
    s2 = load_level(TINY, seed=None)
    draws = [(s1.rng.randint(0, 1_000_000), s2.rng.randint(0, 1_000_000)) for _ in range(10)]
    assert any(a != b for a, b in draws)


def test_load_from_dict_assigns_default_level_id():
    state = _silenced(_minimal_paxie())
    assert state.level_id == "level-1"


# ===========================================================================
# Schema validation
# ===========================================================================

def test_schema_missing_top_level_key():
    data = _minimal_paxie()
    del data["collectorArea"]
    with pytest.raises(LevelLoadError):
        load_level_from_data(data)


def test_schema_unknown_top_level_key():
    data = _minimal_paxie()
    data["weirdField"] = []
    with pytest.raises(LevelLoadError):
        load_level_from_data(data)


def test_schema_invalid_direction():
    data = _minimal_paxie()
    data["collectorArea"]["singleBlockCollectors"] = []
    data["collectorArea"]["tunnels"] = [{
        "x": 0, "y": 0, "direction": "Sideways", "collectorQueue": [{"color": "r"}],
    }]
    with pytest.raises(LevelLoadError):
        load_level_from_data(data)


# ===========================================================================
# Unsupported mechanics — one test per type
# ===========================================================================

def test_unsupported_mystery_collectors():
    data = _minimal_paxie()
    data["collectorArea"]["mysteryCollectors"] = [{"x": 0, "y": 0}]
    with pytest.raises(UnsupportedMechanicError):
        load_level_from_data(data)


def test_unsupported_tied_pairs():
    data = _minimal_paxie()
    data["collectorArea"]["tiedPairs"] = [{"x": 0, "y": 0}]
    with pytest.raises(UnsupportedMechanicError):
        load_level_from_data(data)


def test_unsupported_key_locks():
    data = _minimal_paxie()
    data["collectorArea"]["keyLocks"] = [{
        "color": "r", "keyX": 0, "keyY": 0, "lockX": 0, "lockY": 0,
    }]
    with pytest.raises(UnsupportedMechanicError):
        load_level_from_data(data)


def test_unsupported_hex_stack_area_tunnels():
    data = _minimal_paxie()
    data["hexStackArea"]["tunnels"] = [{
        "x": 0, "y": 0, "direction": "Up", "collectorQueue": [{"color": "r"}],
    }]
    with pytest.raises(UnsupportedMechanicError):
        load_level_from_data(data)


# ===========================================================================
# Color normalization
# ===========================================================================

def test_color_normalization_full_name_to_short():
    data = _minimal_paxie()
    data["collectorArea"]["singleBlockCollectors"] = [{"x": 0, "y": 0, "color": "Yellow"}]
    data["hexStackArea"]["stacks"] = [{"x": 0, "y": 0, "colors": ["Yellow"]}]
    state = _silenced(data)
    assert state.reserve[0][0] == PlainBucket(color="y")
    assert state.color_set == frozenset({"y"})


def test_color_normalization_unknown_token_raises():
    data = _minimal_paxie()
    data["collectorArea"]["singleBlockCollectors"][0]["color"] = "Magenta"
    with pytest.raises(ValueError):
        load_level_from_data(data)


def test_color_normalization_short_code_passthrough():
    data = _minimal_paxie()
    # Short code 'dgr' is in the known short-code set.
    data["collectorArea"]["singleBlockCollectors"][0]["color"] = "dgr"
    data["hexStackArea"]["stacks"] = [{"x": 0, "y": 0, "colors": ["dgr"]}]
    state = _silenced(data)
    assert state.reserve[0][0].color == "dgr"


# ===========================================================================
# Semantic check (1): cell exclusivity
# ===========================================================================

def test_semantic_cell_exclusivity_violated():
    data = _minimal_paxie()
    # Place a single AND a wall at the same (x, y).
    data["collectorArea"]["singleBlockCollectors"] = [{"x": 0, "y": 0, "color": "r"}]
    data["collectorArea"]["deadCells"] = [{"x": 0, "y": 0}]
    with pytest.raises(LevelLoadError, match="Cell exclusivity"):
        load_level_from_data(data)


# ===========================================================================
# Semantic check (2): cell-in-bounds
# ===========================================================================

def test_semantic_cell_out_of_bounds_collector_area():
    data = _minimal_paxie()
    data["collectorArea"]["singleBlockCollectors"] = [{"x": 5, "y": 0, "color": "r"}]
    with pytest.raises(LevelLoadError, match="out of collectorArea bounds"):
        load_level_from_data(data)


def test_semantic_cell_out_of_bounds_hex_stack_area():
    data = _minimal_paxie()
    data["hexStackArea"]["stacks"] = [{"x": 9, "y": 0, "colors": ["r"]}]
    with pytest.raises(LevelLoadError, match="out of hexStackArea bounds"):
        load_level_from_data(data)


# ===========================================================================
# Semantic check (4): color cross-check (warning, not error)
# ===========================================================================

def test_semantic_color_count_mismatch_warns():
    data = _minimal_paxie()
    data["editorMeta"]["colorCount"] = 5  # actually 1 distinct color
    with pytest.warns(UserWarning, match="colorCount"):
        state = load_level_from_data(data)
    assert state is not None


# ===========================================================================
# Semantic check (5): pin destruction cell
# ===========================================================================

def test_semantic_pin_destruction_off_grid_warns():
    """Pin facing Right at (0, 2) — destruction cell (-1, 2) is off-grid → warn (per §11)."""
    data = _minimal_paxie()
    data["collectorArea"]["gridWidth"] = 2
    data["collectorArea"]["gridHeight"] = 3
    data["collectorArea"]["singleBlockCollectors"] = [{"x": 0, "y": 0, "color": "r"}]
    data["collectorArea"]["pinBlockers"] = [{"x": 0, "y": 2, "direction": "Right"}]
    with pytest.warns(UserWarning, match="off-grid"):
        state = load_level_from_data(data)
    assert len(state.pins) == 1


def test_semantic_pin_destruction_wall_errors():
    """Pin destruction cell holds a wall → load error."""
    data = _minimal_paxie()
    data["collectorArea"]["gridWidth"] = 2
    data["collectorArea"]["gridHeight"] = 2
    data["collectorArea"]["singleBlockCollectors"] = [{"x": 1, "y": 0, "color": "r"}]
    data["collectorArea"]["deadCells"] = [{"x": 0, "y": 0}]
    # Pin facing Right at (1, 1): destruction cell = (0, 1). Place a wall there instead.
    data["collectorArea"]["deadCells"].append({"x": 0, "y": 1})
    data["collectorArea"]["pinBlockers"] = [{"x": 1, "y": 1, "direction": "Right"}]
    # Pin's facing direction extends Right from (1,1) → off-grid for gridWidth=2.
    # Destruction cell is opposite-of-Right = Left → (0, 1) which is now a wall → error.
    with pytest.raises(LevelLoadError, match="contains a deadCell"):
        load_level_from_data(data)


def test_semantic_pin_destruction_generator_errors():
    data = _minimal_paxie()
    data["collectorArea"]["gridWidth"] = 2
    data["collectorArea"]["gridHeight"] = 2
    data["collectorArea"]["singleBlockCollectors"] = [{"x": 1, "y": 0, "color": "r"}]
    data["collectorArea"]["tunnels"] = [{
        "x": 0, "y": 1, "direction": "Up", "collectorQueue": [{"color": "r"}],
    }]
    data["collectorArea"]["pinBlockers"] = [{"x": 1, "y": 1, "direction": "Right"}]
    with pytest.raises(LevelLoadError, match="contains a tunnel"):
        load_level_from_data(data)


def test_semantic_pin_destruction_another_pin_errors():
    data = _minimal_paxie()
    data["collectorArea"]["gridWidth"] = 2
    data["collectorArea"]["gridHeight"] = 2
    data["collectorArea"]["singleBlockCollectors"] = [
        {"x": 0, "y": 0, "color": "r"},
    ]
    # Pin A at (1, 1) facing Right: destruction = (0, 1). Pin B origin at (0, 1) → conflict.
    data["collectorArea"]["pinBlockers"] = [
        {"x": 1, "y": 1, "direction": "Right"},
        {"x": 0, "y": 1, "direction": "Up"},  # pin at A's destruction cell
    ]
    with pytest.raises(LevelLoadError, match="another pin"):
        load_level_from_data(data)


# ===========================================================================
# Semantic check (8): slice-bucket parity (warning)
# ===========================================================================

def test_semantic_parity_mismatch_warns():
    data = _minimal_paxie()
    # 1 bucket × 25 capacity = 25 expected; field has 1 slice → mismatch.
    with pytest.warns(UserWarning, match="parity"):
        state = load_level_from_data(data)
    assert state is not None


# ===========================================================================
# Type construction — all 6 supported cell types
# ===========================================================================

def test_all_six_cell_types_construct():
    """Loads the worked example from LEVEL_FORMAT.md §11 and verifies type construction."""
    data = {
        "levelNumber": 9001,
        "levelVersionCode": 1,
        "collectorArea": {
            "gridWidth": 4,
            "gridHeight": 3,
            "singleBlockCollectors": [
                {"x": 0, "y": 0, "color": "r"},
                {"x": 1, "y": 0, "color": "b"},
                {"x": 0, "y": 1, "color": "g"},
            ],
            "woodBoxCollectors": [{"x": 2, "y": 0, "hiddenColor": "g"}],
            "iceCollectors": [{"x": 2, "y": 1, "hiddenColor": "r", "iceCapacity": 3}],
            "deadCells": [{"x": 3, "y": 0}],
            "tunnels": [{
                "x": 1, "y": 2, "direction": "Up",
                "collectorQueue": [{"color": "r"}, {"color": "b"}],
            }],
            "pinBlockers": [{"x": 0, "y": 2, "direction": "Right", "blockCount": 0}],
            "mysteryCollectors": [],
            "tiedPairs": [],
            "keyLocks": [],
        },
        "hexStackArea": {
            "gridWidth": 4, "gridHeight": 2,
            "stacks": [{"x": 0, "y": 0, "colors": ["r", "b", "g"]}],
            "tunnels": [],
        },
        "editorMeta": {
            "totalBlocks": 3, "colorCount": 3, "maxColorsPerStack": 2,
            "heightMin": 1, "heightMax": 3, "randomness": 0.0,
            "verticalPercent": 0.0, "horizontalPercent": 0.0, "mysteryPercent": 0.0,
        },
    }
    state = _silenced(data)

    assert isinstance(state.reserve[0][0], PlainBucket)
    assert state.reserve[0][0].color == "r"
    assert isinstance(state.reserve[0][2], QuestionBucket)
    assert state.reserve[0][2].color == "g"
    # Top-row ?-bucket is revealed on load-time reachability.
    assert state.reserve[0][2].revealed is True

    assert isinstance(state.reserve[1][2], IceBucket)
    assert state.reserve[1][2].color == "r"
    assert state.reserve[1][2].thaw_threshold == 3
    assert state.reserve[1][2].thawed is False

    assert isinstance(state.reserve[0][3], Wall)

    # Generator fires on load into (1, 1), producing a plain red.
    gen = state.reserve[2][1]
    assert isinstance(gen, Generator)
    assert gen.remaining == 1
    assert gen.queue == ["b"]
    assert state.reserve[1][1] == PlainBucket(color="r")

    assert len(state.pins) == 1
    pin = state.pins[0]
    assert isinstance(pin, Pin)
    assert pin.origin_row == 2 and pin.origin_col == 0
    assert pin.direction == "Right"
    assert pin.block_count == 0
    assert pin.destroyed is False
