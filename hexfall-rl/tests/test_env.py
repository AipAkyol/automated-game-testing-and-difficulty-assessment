import copy
import json
import random
import warnings
from pathlib import Path

import pytest

from hexfall.env import HexFallEnv
from hexfall.game import legal_actions_mask
from hexfall.level_loader import load_level

LEVELS_DIR = Path(__file__).parent.parent / "levels"
TINY = LEVELS_DIR / "tiny_solvable.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write(tmp_path: Path, data: dict, name: str = "level.json") -> Path:
    p = tmp_path / name
    p.write_text(json.dumps(data))
    return p


def _vis_level(
    tmp_path: Path,
    stacks: list[dict],
    *,
    reserve_rows: int = 1,
    reserve_cols: int = 1,
    reserve_cells: list[dict] | None = None,
    cap: int = 25,
    name: str = "vis.json",
) -> Path:
    """Build a minimal level JSON for visibility tests."""
    if reserve_cells is None:
        reserve_cells = [{"row": 0, "col": 0, "type": "plain_bucket", "color": "red"}]
    colors: set[str] = set()
    for s in stacks:
        colors.update(s["slices"])
    for c in reserve_cells:
        if "color" in c:
            colors.add(c["color"])
    data = {
        "meta": {"id": "vis-test", "name": "V", "version": 1, "color_count": len(colors)},
        "field": {"stacks": stacks},
        "buffer": {"slots": 5, "bucket_capacity": cap},
        "reserve": {"rows": reserve_rows, "cols": reserve_cols, "cells": reserve_cells},
    }
    return _write(tmp_path, data, name)


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

    # Must match game.legal_actions_mask flattened row-major.
    expected_2d = legal_actions_mask(env._state)
    expected_flat = [expected_2d[r][c] for r in range(rows) for c in range(cols)]
    assert obs["action_mask"] == expected_flat


def test_action_index_decoding():
    env = HexFallEnv(TINY, seed=42)
    obs, _ = env.reset()
    cols = env._state.reserve_cols
    legal = [i for i, m in enumerate(obs["action_mask"]) if m]
    flat = legal[0]
    r, c = flat // cols, flat % cols

    # Replay on a fresh state using game.step directly.
    state_ref = load_level(TINY, seed=42)
    from hexfall.game import step as game_step
    game_step(state_ref, (r, c))

    # env.step should yield same internal state.
    env.step(flat)
    assert env._state.field == state_ref.field
    assert env._state.buffer == state_ref.buffer
    assert env._state.reserve == state_ref.reserve


def test_illegal_action_raises_value_error():
    env = HexFallEnv(TINY, seed=0)
    env.reset()
    # Pick action 0 (red bucket at (0,0)). The cell becomes empty.
    env.step(0)
    # Action 0 now points to an empty cell → illegal.
    with pytest.raises(ValueError):
        env.step(0)


# ---------------------------------------------------------------------------
# Visibility
# ---------------------------------------------------------------------------

def test_visibility_top_slice_only_when_covered(tmp_path):
    # (0,0) height 4, lower neighbor (0,1) height 4.
    # shoulder = 4 - 4 = 0. Only d=0 visible.
    path = _vis_level(
        tmp_path,
        stacks=[
            {"col": 0, "row": 0, "slices": ["red", "red", "red", "red"]},
            {"col": 0, "row": 1, "slices": ["red", "red", "red", "red"]},
        ],
        reserve_cells=[
            {"row": 0, "col": 0, "type": "plain_bucket", "color": "red"},
            {"row": 0, "col": 1, "type": "plain_bucket", "color": "red"},
        ],
        reserve_cols=2,
    )
    env, obs = _open_env(path)
    vis = obs["field_visible"][(0, 0)]
    assert vis[0] == "red"
    assert vis[1:] == ["?", "?", "?"]


def test_visibility_bottom_row_fully_visible(tmp_path):
    # Single stack with no lower neighbors → all slices visible.
    path = _vis_level(
        tmp_path,
        stacks=[{"col": 0, "row": 0, "slices": ["red", "blue", "green", "yellow"]}],
        reserve_cells=[{"row": 0, "col": 0, "type": "plain_bucket", "color": "red"},
                       {"row": 0, "col": 1, "type": "plain_bucket", "color": "blue"},
                       {"row": 0, "col": 2, "type": "plain_bucket", "color": "green"},
                       {"row": 0, "col": 3, "type": "plain_bucket", "color": "yellow"}],
        reserve_cols=4,
    )
    env, obs = _open_env(path)
    vis = obs["field_visible"][(0, 0)]
    assert "?" not in vis
    assert vis == ["red", "blue", "green", "yellow"]


def test_visibility_shoulder_exposed(tmp_path):
    # (0,0) height 4, lower neighbor (0,1) height 2.
    # shoulder = 4 - 2 = 2. d<2 visible, d>=2 hidden.
    path = _vis_level(
        tmp_path,
        stacks=[
            {"col": 0, "row": 0, "slices": ["red", "blue", "green", "yellow"]},
            {"col": 0, "row": 1, "slices": ["red", "red"]},
        ],
        reserve_cells=[
            {"row": 0, "col": 0, "type": "plain_bucket", "color": "red"},
            {"row": 0, "col": 1, "type": "plain_bucket", "color": "blue"},
            {"row": 0, "col": 2, "type": "plain_bucket", "color": "green"},
        ],
        reserve_cols=3,
    )
    env, obs = _open_env(path)
    vis = obs["field_visible"][(0, 0)]
    assert vis[0] == "red"
    assert vis[1] == "blue"
    assert vis[2] == "?"
    assert vis[3] == "?"


def test_visibility_field_heights_always_correct(tmp_path):
    # Heights must match actual stack sizes even when slices are hidden.
    path = _vis_level(
        tmp_path,
        stacks=[
            {"col": 0, "row": 0, "slices": ["red", "red", "red", "red"]},
            {"col": 0, "row": 1, "slices": ["red", "red"]},
        ],
        reserve_cells=[
            {"row": 0, "col": 0, "type": "plain_bucket", "color": "red"},
            {"row": 0, "col": 1, "type": "plain_bucket", "color": "red"},
        ],
        reserve_cols=2,
    )
    env, obs = _open_env(path)
    assert obs["field_heights"][(0, 0)] == 4
    assert obs["field_heights"][(0, 1)] == 2


# ---------------------------------------------------------------------------
# Reserve observation
# ---------------------------------------------------------------------------

def test_question_bucket_hidden_then_revealed(tmp_path):
    # Layout:
    #   row 0: [plain(red), question(blue)]   — both top-row, both revealed at load
    #   row 1: [question(green), None]        — unreachable at load; (1,0) has no empty top-neighbor
    data = {
        "meta": {"id": "qb", "name": "QB", "version": 1, "color_count": 3},
        "field": {"stacks": [
            {"col": 0, "row": 0, "slices": ["red"]},
            {"col": 1, "row": 0, "slices": ["blue"]},
            {"col": 0, "row": 1, "slices": ["green"]},
        ]},
        "buffer": {"slots": 5, "bucket_capacity": 1},
        "reserve": {
            "rows": 2, "cols": 2,
            "cells": [
                {"row": 0, "col": 0, "type": "plain_bucket", "color": "red"},
                {"row": 0, "col": 1, "type": "question_bucket", "color": "blue"},
                {"row": 1, "col": 0, "type": "question_bucket", "color": "green"},
            ],
        },
    }
    path = _write(tmp_path, data)
    env = HexFallEnv(path, seed=0)
    obs, _ = env.reset()

    # (0,1) question in top row → revealed at load.
    assert obs["reserve"][0][1]["color"] == "blue"
    assert obs["reserve"][0][1]["revealed"] is True

    # (1,0) question: (0,0) is a plain bucket (non-empty) → not top-connected → hidden.
    assert obs["reserve"][1][0]["color"] == "?"
    assert obs["reserve"][1][0]["revealed"] is False

    # Pick (0,0) plain red → empties that slot → (1,0) becomes reachable.
    obs2, _, _, _, _ = env.step(0)
    assert obs2["reserve"][1][0]["color"] == "green"
    assert obs2["reserve"][1][0]["revealed"] is True


def test_generator_observation_has_no_queue(tmp_path):
    data = {
        "meta": {"id": "gen-obs", "name": "GO", "version": 1, "color_count": 2},
        "field": {"stacks": [{"col": 0, "row": 0, "slices": ["red", "blue"]}]},
        "buffer": {"slots": 5, "bucket_capacity": 1},
        "reserve": {
            "rows": 1, "cols": 2,
            "cells": [
                {"row": 0, "col": 0, "type": "plain_bucket", "color": "red"},
                {
                    "row": 0, "col": 1,
                    "type": "generator", "facing": "left",
                    "remaining": 0, "queue": [],
                },
            ],
        },
    }
    path = _write(tmp_path, data)
    env = HexFallEnv(path, seed=0)
    obs, _ = env.reset()
    gen_desc = obs["reserve"][0][1]
    assert gen_desc["type"] == "generator"
    assert "facing" in gen_desc
    assert "remaining" in gen_desc
    assert "queue" not in gen_desc


# ---------------------------------------------------------------------------
# Win and lose rewards
# ---------------------------------------------------------------------------

def test_win_reward_plus_one():
    env = HexFallEnv(TINY, seed=0)
    obs, _ = env.reset()
    cols = env._state.reserve_cols

    # First pick (red bucket at (0,0)).
    legal = [i for i, m in enumerate(obs["action_mask"]) if m]
    obs, reward, terminated, _, info = env.step(legal[0])
    assert not terminated
    assert reward == 0.0

    # Second pick (blue bucket at (0,1)).
    legal2 = [i for i, m in enumerate(obs["action_mask"]) if m]
    obs2, reward2, terminated2, _, info2 = env.step(legal2[0])
    assert terminated2
    assert reward2 == 1.0
    assert info2["termination_reason"] == "win"


def test_deadlock_reward_minus_one(tmp_path):
    # 5 red reserve buckets, 1 blue field slice — picking all 5 reds fills buffer
    # with wrong-color buckets, leaving the blue slice unconsumed → deadlock.
    data = {
        "meta": {"id": "dl", "name": "DL", "version": 1, "color_count": 2},
        "field": {"stacks": [{"col": 0, "row": 0, "slices": ["blue"] * 5}]},
        "buffer": {"slots": 5, "bucket_capacity": 1},
        "reserve": {
            "rows": 1, "cols": 5,
            "cells": [
                {"row": 0, "col": i, "type": "plain_bucket", "color": "red"}
                for i in range(5)
            ],
        },
    }
    path = _write(tmp_path, data)
    env = HexFallEnv(path, seed=0)
    obs, _ = env.reset()

    reward, terminated, info = 0.0, False, {}
    for _ in range(10):
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
    env1 = HexFallEnv(TINY, seed=7)
    obs1_0, _ = env1.reset()

    env2 = HexFallEnv(TINY, seed=7)
    obs2_0, _ = env2.reset()
    assert obs1_0 == obs2_0

    # Solve tiny_solvable deterministically (pick action 0 then 1).
    for action in [0, 1]:
        o1, r1, t1, tr1, i1 = env1.step(action)
        o2, r2, t2, tr2, i2 = env2.step(action)
        assert o1 == o2
        assert r1 == r2
        assert t1 == t2


def test_reset_seed_overrides_init_seed():
    # Env created with seed=1, reset with seed=2. State must match seed=2.
    env_ref = HexFallEnv(TINY, seed=2)
    obs_ref, _ = env_ref.reset()

    env = HexFallEnv(TINY, seed=1)
    env.reset()
    obs_override, _ = env.reset(seed=2)

    assert obs_override == obs_ref
