import copy
import random
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from hexfall.game import (
    compute_reachability,
    is_terminal,
    legal_actions_mask,
    pin_blocked_cells,
)
from hexfall.game import step as _game_step
from hexfall.level_loader import load_level
from hexfall.types import Generator, IceBucket, PlainBucket, QuestionBucket, Wall


# Reserve cell type IDs (pin overlay lives in a separate channel, not as a cell type).
CELL_TYPE_IDS: dict[str, int] = {
    "empty": 0,
    "plain_bucket": 1,
    "question_bucket": 2,
    "ice_bucket_frozen": 3,
    "ice_bucket_thawed": 4,
    "generator": 5,
    "wall": 6,
}

# Canonical direction set covering both pins ("Up"/"Down"/...) and generators
# ("up"/"down"/...). The two domains use different casings; we normalize to
# lowercase for the integer encoding.
_DIRECTIONS_CANONICAL = ("up", "down", "left", "right")
DIRECTION_TO_ID: dict[str, int] = {d: i for i, d in enumerate(sorted(_DIRECTIONS_CANONICAL))}
ID_TO_DIRECTION: list[str] = sorted(_DIRECTIONS_CANONICAL)


def _normalize_direction(d: str) -> str:
    return d.lower()


class HexFallEnv(gym.Env):
    """Gymnasium wrapper for the Hex Fall simulator.

    Sizing strategy (Option C — fixed-at-load):
        Every per-level dimension — field_rows, field_cols, max_stack_height,
        reserve_rows, reserve_cols, buffer_slots, num_pins, num_colors — is
        determined once in ``__init__`` from the loaded ``GameState`` and held
        constant across all subsequent ``reset()`` calls. Multi-level training
        will need a wrapper (vectorized envs padding to a common shape, or a
        level-sampling wrapper that rebuilds ``observation_space`` per
        episode); that is out of scope for this env.

    Sentinel color IDs:
        ``HIDDEN_COLOR_ID = num_colors`` marks information the agent cannot
        see (hidden ``?``-bucket color, frozen ice color, occluded field
        slice rendered as "?").
        ``NO_COLOR_ID = num_colors + 1`` marks the absence of a color
        concept (empty cells, padding in dense field stacks beyond actual
        height, empty buffer slots, generators / walls / pins lacking an
        underlying color).

    Pin slot semantics:
        ``num_pins = len(state.pins)`` is fixed at level load. Destruction
        sets ``pins_destroyed[i] = 1`` but the slot is never freed or
        reordered; an episode can only shrink the live-pin count.

    Reserve encoding note:
        The typed reserve channels deviate structurally from the previous
        dict-based encoding: the former ``"pin_ray"`` cell type is unrolled
        into a separate ``reserve_pin_ray_overlay`` channel, and the
        underlying cell's normal attributes describe whatever sits beneath
        the ray (``CELL_TYPE_IDS["empty"]`` when nothing).
    """

    metadata = {"render_modes": []}

    CELL_TYPE_IDS = CELL_TYPE_IDS
    DIRECTION_TO_ID = DIRECTION_TO_ID
    ID_TO_DIRECTION = ID_TO_DIRECTION

    def __init__(self, level_path: str | Path, *, seed: int | None = None):
        super().__init__()
        self._level_path = Path(level_path)
        self._init_seed = seed

        state = load_level(self._level_path, seed=seed)
        self._state = state

        # --- Reserve / action space -------------------------------------
        self._reserve_rows = state.reserve_rows
        self._reserve_cols = state.reserve_cols
        R, C = self._reserve_rows, self._reserve_cols

        # --- Color encoding --------------------------------------------
        self._colors_sorted: list[str] = sorted(state.color_set)
        self.color_to_id: dict[str, int] = {c: i for i, c in enumerate(self._colors_sorted)}
        self.id_to_color: list[str] = list(self._colors_sorted)
        num_colors = len(self._colors_sorted)
        self.HIDDEN_COLOR_ID = num_colors
        self.NO_COLOR_ID = num_colors + 1
        color_high = self.NO_COLOR_ID

        # --- Field dimensions ------------------------------------------
        if state.field:
            self._field_rows = max(row for (_, row) in state.field.keys()) + 1
            self._field_cols = max(col for (col, _) in state.field.keys()) + 1
            self._max_stack_height = max(len(s) for s in state.field.values())
        else:
            self._field_rows = 0
            self._field_cols = 0
            self._max_stack_height = 0

        # --- Buffer ----------------------------------------------------
        self._buffer_slots = state.buffer_slots
        cap_high = max(64, state.bucket_capacity)

        # --- Reserve thaw / generator bounds ---------------------------
        max_thaw = 0
        max_gen_remaining = 0
        for row in state.reserve:
            for cell in row:
                if isinstance(cell, IceBucket):
                    max_thaw = max(max_thaw, cell.thaw_threshold)
                elif isinstance(cell, Generator):
                    max_gen_remaining = max(max_gen_remaining, cell.remaining)
        self._max_thaw = max_thaw
        thaw_high = max(max_thaw, 1)
        gen_high = max(max_gen_remaining, 1)

        # --- Pins ------------------------------------------------------
        self._num_pins = len(state.pins)
        for pin in state.pins:
            if _normalize_direction(pin.direction) not in DIRECTION_TO_ID:
                raise ValueError(f"Unknown pin direction: {pin.direction!r}")
        # block_count can range up to grid extent; 0 means "to grid edge".
        pin_block_high = max(R, C, 1)

        # --- Action space ----------------------------------------------
        self.action_space = spaces.Discrete(R * C)

        # --- Observation space -----------------------------------------
        i32 = np.int32

        def box(shape, high):
            return spaces.Box(low=0, high=high, shape=shape, dtype=i32)

        fh = max(self._field_rows, 0)
        fc = max(self._field_cols, 0)
        msh = max(self._max_stack_height, 0)

        self.observation_space = spaces.Dict({
            "field_visible": box((fh, fc, msh), color_high),
            "field_heights": box((fh, fc), max(msh, 1)),
            "buffer_occupied": box((self._buffer_slots,), 1),
            "buffer_colors": box((self._buffer_slots,), color_high),
            "buffer_capacities": box((self._buffer_slots,), cap_high),
            "buffer_fills": box((self._buffer_slots,), cap_high),
            "reserve_cell_type": box((R, C), max(CELL_TYPE_IDS.values())),
            "reserve_color": box((R, C), color_high),
            "reserve_revealed": box((R, C), 1),
            "reserve_thaw_threshold": box((R, C), thaw_high),
            "reserve_remaining_thaw": box((R, C), thaw_high),
            "reserve_generator_facing": box((R, C), len(ID_TO_DIRECTION) - 1 if ID_TO_DIRECTION else 0),
            "reserve_generator_remaining": box((R, C), gen_high),
            "reserve_pin_ray_overlay": box((R, C), 1),
            "reachability": box((R, C), 1),
            "action_mask": box((R * C,), 1),
            "pins_origin_row": box((self._num_pins,), max(R - 1, 0)),
            "pins_origin_col": box((self._num_pins,), max(C - 1, 0)),
            "pins_direction": box((self._num_pins,), len(ID_TO_DIRECTION) - 1 if ID_TO_DIRECTION else 0),
            "pins_block_count": box((self._num_pins,), pin_block_high),
            "pins_destroyed": box((self._num_pins,), 1),
        })

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[dict, dict]:
        super().reset(seed=seed)
        use_seed = seed if seed is not None else self._init_seed
        self._state = load_level(self._level_path, seed=use_seed)
        return self.get_obs(), {}

    def step(self, action: int) -> tuple[dict, float, bool, bool, dict]:
        row = action // self._reserve_cols
        col = action % self._reserve_cols
        info = _game_step(self._state, (row, col))
        reason = info["termination_reason"]
        terminated = reason is not None
        if reason == "win":
            reward = 1.0
        elif reason in ("deadlock", "fallback"):
            reward = -1.0
        else:
            reward = 0.0
        return self.get_obs(), reward, terminated, False, info

    def render(self) -> None:
        return None

    # ------------------------------------------------------------------
    # State forking (for lookahead players)
    # ------------------------------------------------------------------

    def fork(self) -> "HexFallEnv":
        """Return a new HexFallEnv with an independent deep copy of the current state.

        The fork can be stepped independently of the original: it owns its
        ``GameState`` (field, buffer, reserve, pins, move counter) and its own
        RNG, so ``forked.step(action)`` never mutates the original env.

        The env holds only pure-Python data (no file handles, no GPU tensors),
        so ``copy.deepcopy`` is safe and self-contained. The fork is produced by
        copying rather than via ``gym.make``, so it is not registered with any
        Gymnasium env infrastructure (``spec`` stays ``None``).

        Note on stochasticity: deepcopy clones the RNG in its *current* state, so
        a freshly forked env reproduces the original's fall decisions. To
        enumerate distinct stochastic outcomes, call :meth:`reseed_rng` on the
        fork before stepping it (see ``LookaheadPlayer``).

        Usage::

            forked = env.fork()
            forked.reseed_rng(0)
            obs, rew, term, trunc, info = forked.step(action)
        """
        return copy.deepcopy(self)

    def reseed_rng(self, seed: int) -> None:
        """Re-seed the internal fall-direction RNG.

        The simulator's only stochastic choice is the hex fall direction
        (HEXFALL_MDP_SPEC.md §3.4), drawn from ``GameState.rng`` — a
        ``random.Random`` (not numpy). Replacing it with a freshly seeded
        ``random.Random`` lets a caller enumerate distinct fall outcomes from a
        forked env via ``fork(); reseed_rng(k); step(a)``.
        """
        self._state.rng = random.Random(seed)

    # ------------------------------------------------------------------
    # Observation construction
    # ------------------------------------------------------------------

    def get_obs(self) -> dict[str, Any]:
        state = self._state
        R, C = self._reserve_rows, self._reserve_cols
        i32 = np.int32
        NO = self.NO_COLOR_ID
        HID = self.HIDDEN_COLOR_ID

        # --- Field (dense from sparse stacks) --------------------------
        fh, fc, msh = self._field_rows, self._field_cols, self._max_stack_height
        field_visible = np.full((fh, fc, msh), NO, dtype=i32)
        field_heights = np.zeros((fh, fc), dtype=i32)
        for (col, row), slices in state.field.items():
            if not (0 <= row < fh and 0 <= col < fc):
                continue
            field_heights[row, col] = len(slices)
            vis = visible_slices(col, row, slices, state.field)
            for d, label in enumerate(vis[:msh]):
                if label == "?":
                    field_visible[row, col, d] = HID
                else:
                    field_visible[row, col, d] = self.color_to_id.get(label, NO)

        # --- Buffer ----------------------------------------------------
        slots = self._buffer_slots
        buffer_occupied = np.zeros((slots,), dtype=i32)
        buffer_colors = np.full((slots,), NO, dtype=i32)
        buffer_capacities = np.zeros((slots,), dtype=i32)
        buffer_fills = np.zeros((slots,), dtype=i32)
        for i, slot in enumerate(state.buffer):
            if slot is None:
                continue
            buffer_occupied[i] = 1
            buffer_colors[i] = self.color_to_id.get(slot.color, NO)
            buffer_capacities[i] = slot.capacity
            buffer_fills[i] = slot.fill

        # --- Reserve ---------------------------------------------------
        reserve_cell_type = np.zeros((R, C), dtype=i32)
        reserve_color = np.full((R, C), NO, dtype=i32)
        reserve_revealed = np.zeros((R, C), dtype=i32)
        reserve_thaw_threshold = np.zeros((R, C), dtype=i32)
        reserve_remaining_thaw = np.zeros((R, C), dtype=i32)
        reserve_generator_facing = np.zeros((R, C), dtype=i32)
        reserve_generator_remaining = np.zeros((R, C), dtype=i32)
        reserve_pin_ray_overlay = np.zeros((R, C), dtype=i32)

        blocked = pin_blocked_cells(state)
        mc = state.move_counter
        for r in range(R):
            for c in range(C):
                cell = state.reserve[r][c]
                if cell is None:
                    reserve_cell_type[r, c] = CELL_TYPE_IDS["empty"]
                elif isinstance(cell, PlainBucket):
                    reserve_cell_type[r, c] = CELL_TYPE_IDS["plain_bucket"]
                    reserve_color[r, c] = self.color_to_id.get(cell.color, NO)
                elif isinstance(cell, QuestionBucket):
                    reserve_cell_type[r, c] = CELL_TYPE_IDS["question_bucket"]
                    if cell.revealed:
                        reserve_color[r, c] = self.color_to_id.get(cell.color, NO)
                        reserve_revealed[r, c] = 1
                    else:
                        reserve_color[r, c] = HID
                        reserve_revealed[r, c] = 0
                elif isinstance(cell, IceBucket):
                    if cell.thawed:
                        reserve_cell_type[r, c] = CELL_TYPE_IDS["ice_bucket_thawed"]
                        reserve_color[r, c] = self.color_to_id.get(cell.color, NO)
                        reserve_revealed[r, c] = 1
                        reserve_thaw_threshold[r, c] = cell.thaw_threshold
                        reserve_remaining_thaw[r, c] = 0
                    else:
                        reserve_cell_type[r, c] = CELL_TYPE_IDS["ice_bucket_frozen"]
                        reserve_color[r, c] = HID
                        reserve_revealed[r, c] = 0
                        reserve_thaw_threshold[r, c] = cell.thaw_threshold
                        reserve_remaining_thaw[r, c] = max(0, cell.thaw_threshold - mc)
                elif isinstance(cell, Generator):
                    reserve_cell_type[r, c] = CELL_TYPE_IDS["generator"]
                    reserve_generator_facing[r, c] = DIRECTION_TO_ID[_normalize_direction(cell.facing)]
                    reserve_generator_remaining[r, c] = cell.remaining
                elif isinstance(cell, Wall):
                    reserve_cell_type[r, c] = CELL_TYPE_IDS["wall"]
                if (r, c) in blocked:
                    reserve_pin_ray_overlay[r, c] = 1

        # --- Reachability + action mask --------------------------------
        reach_2d = compute_reachability(state)
        reachability = np.zeros((R, C), dtype=i32)
        for r in range(R):
            for c in range(C):
                reachability[r, c] = 1 if reach_2d[r][c] else 0

        mask_2d = legal_actions_mask(state)
        action_mask = np.zeros((R * C,), dtype=i32)
        for r in range(R):
            for c in range(C):
                action_mask[r * C + c] = 1 if mask_2d[r][c] else 0

        # --- Pins ------------------------------------------------------
        n = self._num_pins
        pins_origin_row = np.zeros((n,), dtype=i32)
        pins_origin_col = np.zeros((n,), dtype=i32)
        pins_direction = np.zeros((n,), dtype=i32)
        pins_block_count = np.zeros((n,), dtype=i32)
        pins_destroyed = np.zeros((n,), dtype=i32)
        for i, pin in enumerate(state.pins):
            pins_origin_row[i] = pin.origin_row
            pins_origin_col[i] = pin.origin_col
            pins_direction[i] = DIRECTION_TO_ID[_normalize_direction(pin.direction)]
            pins_block_count[i] = pin.block_count
            pins_destroyed[i] = 1 if pin.destroyed else 0

        return {
            "field_visible": field_visible,
            "field_heights": field_heights,
            "buffer_occupied": buffer_occupied,
            "buffer_colors": buffer_colors,
            "buffer_capacities": buffer_capacities,
            "buffer_fills": buffer_fills,
            "reserve_cell_type": reserve_cell_type,
            "reserve_color": reserve_color,
            "reserve_revealed": reserve_revealed,
            "reserve_thaw_threshold": reserve_thaw_threshold,
            "reserve_remaining_thaw": reserve_remaining_thaw,
            "reserve_generator_facing": reserve_generator_facing,
            "reserve_generator_remaining": reserve_generator_remaining,
            "reserve_pin_ray_overlay": reserve_pin_ray_overlay,
            "reachability": reachability,
            "action_mask": action_mask,
            "pins_origin_row": pins_origin_row,
            "pins_origin_col": pins_origin_col,
            "pins_direction": pins_direction,
            "pins_block_count": pins_block_count,
            "pins_destroyed": pins_destroyed,
        }


# ------------------------------------------------------------------
# Visibility helpers
# ------------------------------------------------------------------

def _lower_neighbors(col: int, row: int) -> list[tuple[int, int]]:
    """Lower-neighbor (col, row) positions in odd-r offset."""
    if row & 1:
        return [(col, row + 1), (col + 1, row + 1)]
    else:
        return [(col - 1, row + 1), (col, row + 1)]


def visible_slices(
    col: int,
    row: int,
    slices: list[str],
    field: dict[tuple[int, int], list[str]],
) -> list[str]:
    """Return the per-slice visibility list for one stack.

    Hidden slices are represented as "?".
    """
    h = len(slices)
    live_lower = [
        (lc, lr)
        for (lc, lr) in _lower_neighbors(col, row)
        if field.get((lc, lr))
    ]
    if not live_lower:
        return list(slices)

    min_lower_h = min(len(field[(lc, lr)]) for (lc, lr) in live_lower)
    shoulder = h - min_lower_h
    return [
        color if (d == 0 or d < shoulder) else "?"
        for d, color in enumerate(slices)
    ]
