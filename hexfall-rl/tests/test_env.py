import json
import warnings
from pathlib import Path

import numpy as np
import pytest

from hexfall.env import HexFallEnv
from hexfall.game import legal_actions_mask
from hexfall.level_loader import load_level


def _obs_equal(a: dict, b: dict) -> bool:
    if a.keys() != b.keys():
        return False
    for k in a:
        if not np.array_equal(a[k], b[k]):
            return False
    return True

LEVELS_DIR = Path(__file__).parent.parent / "levels"
TINY = LEVELS_DIR / "tiny_solvable.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write(tmp_path: Path, data: dict, name: str = "level.json") -> Path:
    p = tmp_path / name
    p.write_text(json.dumps(data))
    return p


def _open_env(path: Path, seed: int = 0) -> tuple[HexFallEnv, dict]:
    env = HexFallEnv(path, seed=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        obs, _ = env.reset()
    return env, obs


# ---------------------------------------------------------------------------
# Basic protocol
# ---------------------------------------------------------------------------

def test_reset_returns_obs_and_info():
    env = HexFallEnv(TINY, seed=0)
    result = env.reset()
    assert isinstance(result, tuple) and len(result) == 2
    obs, info = result
    assert isinstance(obs, dict)
    assert isinstance(info, dict)


def test_step_returns_5_tuple():
    env = HexFallEnv(TINY, seed=0)
    obs, _ = env.reset()
    legal = [i for i, m in enumerate(obs["action_mask"]) if m]
    result = env.step(legal[0])
    assert isinstance(result, tuple) and len(result) == 5
    obs2, reward, terminated, truncated, info = result
    assert isinstance(obs2, dict)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


def test_action_mask_dimension_and_consistency():
    env = HexFallEnv(TINY, seed=0)
    obs, _ = env.reset()
    rows = env._state.reserve_rows
    cols = env._state.reserve_cols
    assert len(obs["action_mask"]) == rows * cols

    expected_2d = legal_actions_mask(env._state)
    expected_flat = [int(expected_2d[r][c]) for r in range(rows) for c in range(cols)]
    assert list(obs["action_mask"]) == expected_flat


def test_action_index_decoding():
    env = HexFallEnv(TINY, seed=42)
    obs, _ = env.reset()
    cols = env._state.reserve_cols
    legal = [i for i, m in enumerate(obs["action_mask"]) if m]
    flat = legal[0]
    r, c = flat // cols, flat % cols

    state_ref = load_level(TINY, seed=42)
    from hexfall.game import step as game_step
    game_step(state_ref, (r, c))

    env.step(flat)
    assert env._state.field == state_ref.field
    assert env._state.buffer == state_ref.buffer
    assert env._state.reserve == state_ref.reserve


def test_illegal_action_raises_value_error():
    env = HexFallEnv(TINY, seed=0)
    env.reset()
    env.step(0)
    with pytest.raises(ValueError):
        env.step(0)


# ---------------------------------------------------------------------------
# Observation: move counter and pins
# ---------------------------------------------------------------------------

def test_move_counter_not_in_obs():
    # Per HEXFALL_MDP_SPEC §3.6 the raw move counter is NOT part of the agent
    # observation. Increment behavior is asserted in test_game.py.
    env = HexFallEnv(TINY, seed=0)
    obs, _ = env.reset()
    assert "move_counter" not in obs


def test_obs_includes_pins_arrays():
    env, obs = _open_env(LEVELS_DIR / "pin_test.json")
    assert obs["pins_destroyed"].shape == (2,)
    assert int((obs["pins_destroyed"] == 0).sum()) == 2
    # Required parallel arrays all present and same length.
    for k in ("pins_origin_row", "pins_origin_col", "pins_direction",
              "pins_block_count", "pins_destroyed"):
        assert obs[k].shape == (2,)


def test_destroyed_pins_marked_in_observation():
    env, obs = _open_env(LEVELS_DIR / "pin_test.json")
    assert int((obs["pins_destroyed"] == 0).sum()) == 2
    # The (1, 2) pick destroys both pins (cascade) and wins the level.
    legal = [i for i, m in enumerate(obs["action_mask"]) if m]
    assert legal == [legal_actions_mask(env._state)[1].index(True) + 1 * env._reserve_cols]
    obs2, _, terminated, _, _ = env.step(legal[0])
    assert terminated
    # Slots are not freed: shape stays (2,), but both pins are destroyed.
    assert obs2["pins_destroyed"].shape == (2,)
    assert int(obs2["pins_destroyed"].sum()) == 2


# ---------------------------------------------------------------------------
# Observation: ice bucket encoding
# ---------------------------------------------------------------------------

def test_ice_bucket_frozen_hides_color():
    env, obs = _open_env(LEVELS_DIR / "ice_test.json")
    assert int(obs["reserve_cell_type"][0, 1]) == env.CELL_TYPE_IDS["ice_bucket_frozen"]
    assert int(obs["reserve_color"][0, 1]) == env.HIDDEN_COLOR_ID
    assert int(obs["reserve_thaw_threshold"][0, 1]) == 1
    assert int(obs["reserve_remaining_thaw"][0, 1]) == 1


def test_ice_bucket_remaining_decrements_with_move_counter():
    env, obs = _open_env(LEVELS_DIR / "ice_test.json")
    assert int(obs["reserve_remaining_thaw"][0, 2]) == 2  # threshold 2, counter 0
    obs2, _, _, _, _ = env.step(0)  # pick (0, 0) plain
    # (0, 1) was threshold 1 → thawed now.
    assert int(obs2["reserve_cell_type"][0, 1]) == env.CELL_TYPE_IDS["ice_bucket_thawed"]
    assert int(obs2["reserve_color"][0, 1]) == env.color_to_id["r"]
    # (0, 2) was threshold 2 → still frozen, remaining = 1.
    assert int(obs2["reserve_cell_type"][0, 2]) == env.CELL_TYPE_IDS["ice_bucket_frozen"]
    assert int(obs2["reserve_remaining_thaw"][0, 2]) == 1


def test_ice_bucket_thaws_reveals_color():
    env, obs = _open_env(LEVELS_DIR / "ice_test.json")
    env.step(0)
    obs2 = env.get_obs()
    assert int(obs2["reserve_color"][0, 1]) == env.color_to_id["r"]
    assert int(obs2["reserve_cell_type"][0, 1]) == env.CELL_TYPE_IDS["ice_bucket_thawed"]


# ---------------------------------------------------------------------------
# Observation: pin overlay encoding
# ---------------------------------------------------------------------------

def test_pin_ray_overlay_with_underneath():
    env, obs = _open_env(LEVELS_DIR / "pin_test.json")
    # Reserve (1, 0) is in pin A's ray and has a plain red bucket underneath.
    assert int(obs["reserve_pin_ray_overlay"][1, 0]) == 1
    assert int(obs["reserve_cell_type"][1, 0]) == env.CELL_TYPE_IDS["plain_bucket"]
    assert int(obs["reserve_color"][1, 0]) == env.color_to_id["r"]


def test_pin_ray_overlay_clears_after_destruction():
    env, obs = _open_env(LEVELS_DIR / "pin_test.json")
    # Pick (1, 2) → destroys both pins via cascade.
    flat = 1 * env._reserve_cols + 2
    env.step(flat)
    obs2 = env.get_obs()
    # No cells should report pin_ray overlay anymore.
    assert int(obs2["reserve_pin_ray_overlay"].sum()) == 0


# ---------------------------------------------------------------------------
# Visibility
# ---------------------------------------------------------------------------

def _vis_path(tmp_path: Path, stacks: list[dict], reserve_cells: list[dict],
              gridWidth: int, name: str = "vis.json") -> Path:
    colors: set[str] = set()
    for s in stacks:
        colors.update(s["colors"])
    for c in reserve_cells:
        if "color" in c:
            colors.add(c["color"])
    data = {
        "levelNumber": 1, "levelVersionCode": 1,
        "collectorArea": {
            "gridWidth": gridWidth, "gridHeight": 1,
            "singleBlockCollectors": reserve_cells,
            "woodBoxCollectors": [], "iceCollectors": [],
            "deadCells": [], "tunnels": [], "pinBlockers": [],
            "mysteryCollectors": [], "tiedPairs": [], "keyLocks": [],
        },
        "hexStackArea": {
            "gridWidth": max(s["x"] for s in stacks) + 1,
            "gridHeight": max(s["y"] for s in stacks) + 1,
            "stacks": stacks, "tunnels": [],
        },
        "editorMeta": {
            "totalBlocks": sum(len(s["colors"]) for s in stacks),
            "colorCount": max(1, len(colors)), "maxColorsPerStack": 4,
            "heightMin": 1, "heightMax": 6, "randomness": 0.0,
            "verticalPercent": 0.0, "horizontalPercent": 0.0, "mysteryPercent": 0.0,
        },
    }
    return _write(tmp_path, data, name)


def _vis_stack(env: HexFallEnv, obs: dict, col: int, row: int) -> list[str]:
    """Decode dense field_visible[row, col, :height] back to label strings."""
    height = int(obs["field_heights"][row, col])
    out: list[str] = []
    for d in range(height):
        cid = int(obs["field_visible"][row, col, d])
        if cid == env.HIDDEN_COLOR_ID:
            out.append("?")
        elif cid == env.NO_COLOR_ID:
            out.append("?")
        else:
            out.append(env.id_to_color[cid])
    return out


def test_visibility_top_slice_only_when_covered(tmp_path):
    path = _vis_path(
        tmp_path,
        stacks=[
            {"x": 0, "y": 0, "colors": ["r", "r", "r", "r"]},
            {"x": 0, "y": 1, "colors": ["r", "r", "r", "r"]},
        ],
        reserve_cells=[
            {"x": 0, "y": 0, "color": "r"},
            {"x": 1, "y": 0, "color": "r"},
        ],
        gridWidth=2,
    )
    env, obs = _open_env(path)
    vis = _vis_stack(env, obs, 0, 0)
    assert vis[0] == "r"
    assert vis[1:] == ["?", "?", "?"]


def test_visibility_bottom_row_fully_visible(tmp_path):
    path = _vis_path(
        tmp_path,
        stacks=[{"x": 0, "y": 0, "colors": ["r", "b", "g", "y"]}],
        reserve_cells=[
            {"x": 0, "y": 0, "color": "r"},
            {"x": 1, "y": 0, "color": "b"},
            {"x": 2, "y": 0, "color": "g"},
            {"x": 3, "y": 0, "color": "y"},
        ],
        gridWidth=4,
    )
    env, obs = _open_env(path)
    vis = _vis_stack(env, obs, 0, 0)
    assert "?" not in vis
    assert vis == ["r", "b", "g", "y"]


def test_visibility_shoulder_exposed(tmp_path):
    path = _vis_path(
        tmp_path,
        stacks=[
            {"x": 0, "y": 0, "colors": ["r", "b", "g", "y"]},
            {"x": 0, "y": 1, "colors": ["r", "r"]},
        ],
        reserve_cells=[
            {"x": 0, "y": 0, "color": "r"},
            {"x": 1, "y": 0, "color": "b"},
            {"x": 2, "y": 0, "color": "g"},
        ],
        gridWidth=3,
    )
    env, obs = _open_env(path)
    vis = _vis_stack(env, obs, 0, 0)
    assert vis[0] == "r"
    assert vis[1] == "b"
    assert vis[2] == "?"
    assert vis[3] == "?"


def test_visibility_field_heights_always_correct(tmp_path):
    path = _vis_path(
        tmp_path,
        stacks=[
            {"x": 0, "y": 0, "colors": ["r", "r", "r", "r"]},
            {"x": 0, "y": 1, "colors": ["r", "r"]},
        ],
        reserve_cells=[
            {"x": 0, "y": 0, "color": "r"},
            {"x": 1, "y": 0, "color": "r"},
        ],
        gridWidth=2,
    )
    env, obs = _open_env(path)
    assert int(obs["field_heights"][0, 0]) == 4
    assert int(obs["field_heights"][1, 0]) == 2


# ---------------------------------------------------------------------------
# Reserve observation
# ---------------------------------------------------------------------------

def test_question_bucket_hidden_then_revealed():
    env, obs = _open_env(LEVELS_DIR / "hidden_test.json")

    # Reserve layout (row=y, col=x):
    #   (0,0)=plain r   (0,1)=plain r
    #   (1,0)=?-bucket  (1,1)=plain b
    # The ?-bucket at (1, 0) has (0, 0) above (occupied) → not reachable → hidden.
    assert int(obs["reserve_cell_type"][1, 0]) == env.CELL_TYPE_IDS["question_bucket"]
    assert int(obs["reserve_color"][1, 0]) == env.HIDDEN_COLOR_ID
    assert int(obs["reserve_revealed"][1, 0]) == 0

    # Pick the plain bucket at (0, 0) — flat = 0 * cols + 0 = 0.
    obs2, _, _, _, _ = env.step(0)
    assert int(obs2["reserve_color"][1, 0]) == env.color_to_id["b"]
    assert int(obs2["reserve_revealed"][1, 0]) == 1


def test_generator_observation_has_no_queue():
    env, obs = _open_env(LEVELS_DIR / "generator_test.json")
    # Generator at reserve (0, 1): typed channels expose facing + remaining,
    # but there is no queue channel in the observation.
    assert int(obs["reserve_cell_type"][0, 1]) == env.CELL_TYPE_IDS["generator"]
    assert "reserve_generator_facing" in obs
    assert "reserve_generator_remaining" in obs
    assert "reserve_generator_queue" not in obs


# ---------------------------------------------------------------------------
# Win and lose rewards
# ---------------------------------------------------------------------------

def test_win_reward_plus_one():
    env = HexFallEnv(TINY, seed=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        obs, _ = env.reset()

    legal = [i for i, m in enumerate(obs["action_mask"]) if m]
    obs, reward, terminated, _, info = env.step(legal[0])
    assert not terminated
    assert reward == 0.0

    legal2 = [i for i, m in enumerate(obs["action_mask"]) if m]
    obs2, reward2, terminated2, _, info2 = env.step(legal2[0])
    assert terminated2
    assert reward2 == 1.0
    assert info2["termination_reason"] == "win"


def test_deadlock_reward_minus_one():
    env, obs = _open_env(LEVELS_DIR / "forced_lose.json")
    # Pick all 5 blue buckets in sequence; each enters a buffer slot but none can
    # pull from the red-only field → buffer fills → deadlock.
    reward = 0.0
    terminated = False
    info: dict = {}
    for _ in range(5):
        legal = [i for i, m in enumerate(obs["action_mask"]) if m]
        if not legal:
            break
        obs, reward, terminated, _, info = env.step(legal[0])
        if terminated:
            break
    assert terminated
    assert reward == -1.0
    assert info["termination_reason"] == "deadlock"


# ---------------------------------------------------------------------------
# Determinism and seed
# ---------------------------------------------------------------------------

def test_determinism_replay():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        env1 = HexFallEnv(TINY, seed=7)
        obs1_0, _ = env1.reset()
        env2 = HexFallEnv(TINY, seed=7)
        obs2_0, _ = env2.reset()
    assert _obs_equal(obs1_0, obs2_0)

    for action in [0, 1]:
        o1, r1, t1, tr1, i1 = env1.step(action)
        o2, r2, t2, tr2, i2 = env2.step(action)
        assert _obs_equal(o1, o2)
        assert r1 == r2
        assert t1 == t2


def test_reset_seed_overrides_init_seed():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        env_ref = HexFallEnv(TINY, seed=2)
        obs_ref, _ = env_ref.reset()

        env = HexFallEnv(TINY, seed=1)
        env.reset()
        obs_override, _ = env.reset(seed=2)

    assert _obs_equal(obs_override, obs_ref)
