from pathlib import Path
from typing import Any

import gymnasium as gym
from gymnasium import spaces

from hexfall.game import compute_reachability, is_terminal, legal_actions_mask
from hexfall.game import step as _game_step
from hexfall.level_loader import load_level
from hexfall.types import Generator, PlainBucket, QuestionBucket, Wall


class HexFallEnv(gym.Env):
    """Gymnasium wrapper for the Hex Fall simulator."""

    metadata = {"render_modes": []}

    def __init__(self, level_path: str | Path, *, seed: int | None = None):
        super().__init__()
        self._level_path = Path(level_path)
        self._init_seed = seed

        # Load once to read grid dimensions for space declarations.
        state = load_level(self._level_path, seed=seed)
        self._reserve_rows = state.reserve_rows
        self._reserve_cols = state.reserve_cols
        self._state = state

        self.action_space = spaces.Discrete(self._reserve_rows * self._reserve_cols)

        # Typed-space declaration for a dict-of-dicts with variable shapes is
        # awkward, and downstream RL training will flatten/encode anyway. Defer.
        self.observation_space = spaces.Dict({})

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[dict, dict]:
        super().reset(seed=seed)
        use_seed = seed if seed is not None else self._init_seed
        self._state = load_level(self._level_path, seed=use_seed)
        return self._get_obs(), {}

    def step(self, action: int) -> tuple[dict, float, bool, bool, dict]:
        row = action // self._reserve_cols
        col = action % self._reserve_cols
        # Propagates ValueError if action is illegal — do NOT swallow.
        info = _game_step(self._state, (row, col))
        reason = info["termination_reason"]
        terminated = reason is not None
        if reason == "win":
            reward = 1.0
        elif reason in ("deadlock", "fallback"):
            reward = -1.0
        else:
            reward = 0.0
        return self._get_obs(), reward, terminated, False, info

    def render(self) -> None:
        return None

    # ------------------------------------------------------------------
    # Observation construction
    # ------------------------------------------------------------------

    def _get_obs(self) -> dict[str, Any]:
        state = self._state
        field_visible: dict[tuple[int, int], list[str]] = {}
        field_heights: dict[tuple[int, int], int] = {}
        for (col, row), slices in state.field.items():
            field_heights[(col, row)] = len(slices)
            field_visible[(col, row)] = _visible_slices(col, row, slices, state.field)

        buffer_obs: list[dict | None] = []
        for slot in state.buffer:
            if slot is None:
                buffer_obs.append(None)
            else:
                buffer_obs.append(
                    {"color": slot.color, "capacity": slot.capacity, "fill": slot.fill}
                )

        reserve_obs: list[list[dict]] = []
        for r in range(state.reserve_rows):
            row_obs = []
            for c in range(state.reserve_cols):
                row_obs.append(_cell_descriptor(state.reserve[r][c]))
            reserve_obs.append(row_obs)

        reach = compute_reachability(state)
        mask_2d = legal_actions_mask(state)
        action_mask = [
            mask_2d[r][c]
            for r in range(state.reserve_rows)
            for c in range(state.reserve_cols)
        ]

        return {
            "field_visible": field_visible,
            "field_heights": field_heights,
            "buffer": buffer_obs,
            "reserve": reserve_obs,
            "reachability": reach,
            "action_mask": action_mask,
        }


# ------------------------------------------------------------------
# Visibility helpers
# ------------------------------------------------------------------

def _lower_neighbors(col: int, row: int) -> list[tuple[int, int]]:
    """Lower-neighbor (col, row) positions in odd-r offset."""
    if row & 1:  # odd row
        return [(col, row + 1), (col + 1, row + 1)]
    else:  # even row
        return [(col - 1, row + 1), (col, row + 1)]


def _visible_slices(
    col: int,
    row: int,
    slices: list[str],
    field: dict[tuple[int, int], list[str]],
) -> list[str]:
    """Return the per-slice visibility list for one stack.

    Hidden slices are represented as "?".

    Rules (MDP §4.1, RULES §3):
      d=0          → always visible (top slice).
      no live lower neighbor → bottom-row stack; all visible.
      d < h - min_lower_height → exposed shoulder; visible.

    # TODO: the "edge columns of alternating rows" visibility rule from
    # RULES §3 is not implemented.  Omission means the agent sees more "?"
    # than strictly necessary but never a slice that should be hidden.
    """
    h = len(slices)
    live_lower = [
        (lc, lr)
        for (lc, lr) in _lower_neighbors(col, row)
        if field.get((lc, lr))  # non-empty stack exists
    ]
    if not live_lower:
        return list(slices)

    min_lower_h = min(len(field[(lc, lr)]) for (lc, lr) in live_lower)
    shoulder = h - min_lower_h
    return [
        color if (d == 0 or d < shoulder) else "?"
        for d, color in enumerate(slices)
    ]


def _cell_descriptor(cell) -> dict:
    if cell is None:
        return {"type": "empty"}
    if isinstance(cell, PlainBucket):
        return {"type": "plain_bucket", "color": cell.color}
    if isinstance(cell, QuestionBucket):
        if cell.revealed:
            return {"type": "question_bucket", "color": cell.color, "revealed": True}
        return {"type": "question_bucket", "color": "?", "revealed": False}
    if isinstance(cell, Generator):
        # Generator output queue is hidden per MDP §4.3.
        return {"type": "generator", "facing": cell.facing, "remaining": cell.remaining}
    if isinstance(cell, Wall):
        return {"type": "wall"}
    raise ValueError(f"Unknown cell type: {type(cell)}")
