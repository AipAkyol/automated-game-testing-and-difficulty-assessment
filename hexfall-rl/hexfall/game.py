import warnings
from collections import deque
from typing import Any, Optional

from hexfall.types import (
    BufferBucket,
    GameState,
    Generator,
    PlainBucket,
    QuestionBucket,
)


_DIRECTIONS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}


def upper_neighbors(col: int, row: int) -> list[tuple[int, int]]:
    """Up to 2 upper-neighbor (col, row) pairs in odd-r offset.

    Caller filters out-of-bounds and missing stacks.
    """
    if row & 1:
        return [(col, row - 1), (col + 1, row - 1)]
    else:
        return [(col - 1, row - 1), (col, row - 1)]


def reserve_neighbors(row: int, col: int, rows: int, cols: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = row + dr, col + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            out.append((nr, nc))
    return out


def compute_reachability(state: GameState) -> list[list[bool]]:
    rows, cols = state.reserve_rows, state.reserve_cols

    top_empty: set[tuple[int, int]] = set()
    queue: deque[tuple[int, int]] = deque()
    for c in range(cols):
        if state.reserve[0][c] is None:
            top_empty.add((0, c))
            queue.append((0, c))
    while queue:
        r, c = queue.popleft()
        for nr, nc in reserve_neighbors(r, c, rows, cols):
            if (nr, nc) in top_empty:
                continue
            if state.reserve[nr][nc] is None:
                top_empty.add((nr, nc))
                queue.append((nr, nc))

    reach = [[False] * cols for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            cell = state.reserve[r][c]
            if not isinstance(cell, (PlainBucket, QuestionBucket)):
                continue
            if r == 0:
                reach[r][c] = True
                continue
            for nr, nc in reserve_neighbors(r, c, rows, cols):
                if (nr, nc) in top_empty:
                    reach[r][c] = True
                    break
    return reach


def legal_actions_mask(state: GameState) -> list[list[bool]]:
    rows, cols = state.reserve_rows, state.reserve_cols
    has_empty_slot = any(slot is None for slot in state.buffer)
    if not has_empty_slot:
        return [[False] * cols for _ in range(rows)]
    reach = compute_reachability(state)
    mask = [[False] * cols for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            if not reach[r][c]:
                continue
            cell = state.reserve[r][c]
            if isinstance(cell, (PlainBucket, QuestionBucket)):
                mask[r][c] = True
    return mask


def _pull_phase(state: GameState) -> bool:
    non_empty = [(c, r) for (c, r), s in state.field.items() if s]
    if not non_empty:
        return False
    bottom_row = max(r for _, r in non_empty)

    by_color: dict[str, list[tuple[int, BufferBucket]]] = {}
    for i, slot in enumerate(state.buffer):
        if slot is None or slot.fill >= slot.capacity:
            continue
        by_color.setdefault(slot.color, []).append((i, slot))

    any_pull = False
    for color, buckets in by_color.items():
        # Active bucket per same-color collision rule: highest fill, then lowest
        # buffer slot index. Tiebreak by slot index is a spec-level choice.
        _, active_bucket = max(buckets, key=lambda ib: (ib[1].fill, -ib[0]))
        candidates = sorted(
            (c, r) for (c, r), s in state.field.items()
            if r == bottom_row and s and s[0] == color
        )
        if not candidates:
            continue
        target = candidates[0]
        state.field[target].pop(0)
        active_bucket.fill += 1
        any_pull = True
    return any_pull


def _fill_check_phase(state: GameState) -> bool:
    any_left = False
    for i, slot in enumerate(state.buffer):
        if slot is not None and slot.fill >= slot.capacity:
            state.buffer[i] = None
            any_left = True
    return any_left


def _fall_phase(state: GameState) -> bool:
    if not state.field:
        return False
    # Bottom row is the max row index across all field keys, including positions
    # that were just emptied by pull this tick — those are the falls to resolve.
    bottom_row = max(r for (_, r) in state.field.keys())
    empty_positions = sorted(
        (c, r) for (c, r), s in state.field.items()
        if r == bottom_row and not s
    )
    if not empty_positions:
        return False

    any_change = False
    for col, row in empty_positions:
        if (col, row) not in state.field:
            continue
        candidates = [
            (uc, ur) for (uc, ur) in upper_neighbors(col, row)
            if (uc, ur) in state.field and state.field[(uc, ur)]
        ]
        if len(candidates) == 0:
            del state.field[(col, row)]
            any_change = True
        elif len(candidates) == 1:
            # Deterministic single-candidate fall. NO RNG call.
            uc, ur = candidates[0]
            state.field[(col, row)] = state.field.pop((uc, ur))
            any_change = True
        else:
            # Two candidates: the only place state.rng is consumed in the simulator.
            idx = state.rng.randint(0, 1)
            uc, ur = candidates[idx]
            state.field[(col, row)] = state.field.pop((uc, ur))
            any_change = True
    return any_change


def _generator_phase(state: GameState) -> bool:
    any_fired = False
    rows, cols = state.reserve_rows, state.reserve_cols
    for r in range(rows):
        for c in range(cols):
            cell = state.reserve[r][c]
            if not isinstance(cell, Generator):
                continue
            if cell.remaining <= 0:
                continue
            dr, dc = _DIRECTIONS[cell.facing]
            fr, fc = r + dr, c + dc
            if not (0 <= fr < rows and 0 <= fc < cols):
                continue
            if state.reserve[fr][fc] is not None:
                continue
            color = cell.queue.pop(0)
            state.reserve[fr][fc] = PlainBucket(color=color)
            cell.remaining -= 1
            any_fired = True
    return any_fired


def _reachability_and_reveal_phase(state: GameState) -> None:
    reach = compute_reachability(state)
    for r in range(state.reserve_rows):
        for c in range(state.reserve_cols):
            cell = state.reserve[r][c]
            if isinstance(cell, QuestionBucket) and reach[r][c] and not cell.revealed:
                cell.revealed = True


def run_until_quiescent(state: GameState) -> None:
    while True:
        any_change = False
        any_change |= _pull_phase(state)
        any_change |= _fill_check_phase(state)
        any_change |= _fall_phase(state)
        any_change |= _generator_phase(state)
        _reachability_and_reveal_phase(state)
        if not any_change:
            break
    state.quiescent = True


def is_terminal(state: GameState) -> Optional[str]:
    if all(not s for s in state.field.values()):
        return "win"
    mask = legal_actions_mask(state)
    if not any(any(row) for row in mask):
        if all(slot is not None for slot in state.buffer):
            return "deadlock"
        return "fallback"
    return None


def step(state: GameState, action: tuple[int, int]) -> dict[str, Any]:
    r, c = action
    rows, cols = state.reserve_rows, state.reserve_cols
    if not (0 <= r < rows and 0 <= c < cols):
        raise ValueError(f"Illegal action {action}: out of bounds")
    mask = legal_actions_mask(state)
    if not mask[r][c]:
        raise ValueError(f"Illegal action {action}")

    cell = state.reserve[r][c]
    new_bucket = BufferBucket(color=cell.color, capacity=state.bucket_capacity, fill=0)
    state.reserve[r][c] = None

    for i, slot in enumerate(state.buffer):
        if slot is None:
            state.buffer[i] = new_bucket
            break

    state.quiescent = False
    run_until_quiescent(state)

    reason = is_terminal(state)
    info: dict[str, Any] = {"termination_reason": reason}
    if reason == "fallback":
        warnings.warn(
            f"Fallback termination on level {state.level_id}",
            RuntimeWarning,
            stacklevel=2,
        )
    return info
