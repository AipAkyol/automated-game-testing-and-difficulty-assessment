import warnings
from collections import deque
from typing import Any, Optional

from hexfall.types import (
    BufferBucket,
    GameState,
    Generator,
    IceBucket,
    Pin,
    PlainBucket,
    QuestionBucket,
)


_DIRECTIONS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}

# Pin facing → forward (drow, dcol) offset, used to walk the ray from origin.
_PIN_DIR_OFFSET = {
    "Up": (-1, 0),
    "Down": (1, 0),
    "Left": (0, -1),
    "Right": (0, 1),
}

# Pin facing → opposite (drow, dcol), used to locate the destruction cell.
_PIN_OPPOSITE_OFFSET = {
    "Up": (1, 0),
    "Down": (-1, 0),
    "Left": (0, 1),
    "Right": (0, -1),
}


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


def _pin_ray_cells(pin: Pin, rows: int, cols: int) -> list[tuple[int, int]]:
    """Return (row, col) cells occupied by this pin's ray, clipped to the grid."""
    drow, dcol = _PIN_DIR_OFFSET[pin.direction]
    cells: list[tuple[int, int]] = []
    if pin.block_count == 0:
        # Ray extends from origin to the grid edge in the facing direction.
        r, c = pin.origin_row, pin.origin_col
        while 0 <= r < rows and 0 <= c < cols:
            cells.append((r, c))
            r += drow
            c += dcol
    else:
        # Ray is origin + block_count more cells, total block_count+1.
        for i in range(pin.block_count + 1):
            r = pin.origin_row + i * drow
            c = pin.origin_col + i * dcol
            if 0 <= r < rows and 0 <= c < cols:
                cells.append((r, c))
    return cells


def pin_blocked_cells(state: GameState) -> set[tuple[int, int]]:
    """All (row, col) cells covered by undestroyed pin rays."""
    blocked: set[tuple[int, int]] = set()
    for pin in state.pins:
        if pin.destroyed:
            continue
        for cell in _pin_ray_cells(pin, state.reserve_rows, state.reserve_cols):
            blocked.add(cell)
    return blocked


def compute_reachability(state: GameState) -> list[list[bool]]:
    rows, cols = state.reserve_rows, state.reserve_cols
    blocked = pin_blocked_cells(state)

    # Top-connected empty cells: BFS from the top row through None cells, treating
    # pin-ray cells as opaque regardless of the cell content underneath.
    top_empty: set[tuple[int, int]] = set()
    queue: deque[tuple[int, int]] = deque()
    for c in range(cols):
        if (0, c) in blocked:
            continue
        if state.reserve[0][c] is None:
            top_empty.add((0, c))
            queue.append((0, c))
    while queue:
        r, c = queue.popleft()
        for nr, nc in reserve_neighbors(r, c, rows, cols):
            if (nr, nc) in top_empty:
                continue
            if (nr, nc) in blocked:
                continue
            if state.reserve[nr][nc] is None:
                top_empty.add((nr, nc))
                queue.append((nr, nc))

    reach = [[False] * cols for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            if (r, c) in blocked:
                continue
            cell = state.reserve[r][c]
            # Per MDP §4.5, reachability is a property of the cell — ice buckets
            # can be reachable even while frozen (legality, not reachability, is
            # what gates picking).
            if not isinstance(cell, (PlainBucket, QuestionBucket, IceBucket)):
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
    blocked = pin_blocked_cells(state)
    mask = [[False] * cols for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            if (r, c) in blocked:
                continue
            if not reach[r][c]:
                continue
            cell = state.reserve[r][c]
            if isinstance(cell, (PlainBucket, QuestionBucket)):
                mask[r][c] = True
            elif isinstance(cell, IceBucket) and cell.thawed:
                mask[r][c] = True
    return mask


def _ice_thaw_phase(state: GameState) -> bool:
    """Thaw any frozen ice bucket whose threshold the move counter has reached.

    Spec: MDP §5.4 step 2.i — runs at the start of every tick.
    """
    any_thaw = False
    for r in range(state.reserve_rows):
        for c in range(state.reserve_cols):
            cell = state.reserve[r][c]
            if isinstance(cell, IceBucket) and not cell.thawed:
                if state.move_counter >= cell.thaw_threshold:
                    cell.thawed = True
                    any_thaw = True
    return any_thaw


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
            uc, ur = candidates[0]
            state.field[(col, row)] = state.field.pop((uc, ur))
            any_change = True
        else:
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


def _pin_destruction_phase(state: GameState) -> bool:
    """Destroy pins whose destruction cell ended this tick empty; cascade.

    Spec: MDP §5.4 step 2.vii. The destruction cell is one step opposite the
    pin's facing. If it is None (and in-bounds), the pin destroys: all ray
    cells become None (the underlying cell content is also cleared — pins
    overlay state, but per MDP §3.3 a pin sits atop content like a wall;
    Paxie levels never place a wall underneath a pin per §5.6 invariants).
    After any destruction, reachability is recomputed; cascading destructions
    within the same tick are resolved in this loop.
    """
    any_destroyed = False
    while True:
        destroyed_this_round = False
        for pin in state.pins:
            if pin.destroyed:
                continue
            drow, dcol = _PIN_OPPOSITE_OFFSET[pin.direction]
            dest_row = pin.origin_row + drow
            dest_col = pin.origin_col + dcol
            if not (0 <= dest_row < state.reserve_rows and 0 <= dest_col < state.reserve_cols):
                continue
            if state.reserve[dest_row][dest_col] is not None:
                continue
            pin.destroyed = True
            for r, c in _pin_ray_cells(pin, state.reserve_rows, state.reserve_cols):
                state.reserve[r][c] = None
            destroyed_this_round = True
            any_destroyed = True
        if not destroyed_this_round:
            break
        # Per MDP §5.4 step 2.vii: re-run reachability recomputation after destruction.
        _reachability_and_reveal_phase(state)
    return any_destroyed


def run_until_quiescent(state: GameState) -> None:
    """Run the automatic-update loop until no further changes (MDP §5.4 step 2)."""
    while True:
        any_change = False
        # Tick order per MDP §5.4 step 2:
        # i. Ice thaw checks (deterministic, no RNG)
        any_change |= _ice_thaw_phase(state)
        # ii. Buffer pulls
        any_change |= _pull_phase(state)
        # iii. Bucket fill checks
        any_change |= _fill_check_phase(state)
        # iv. Stack clear / fall (only RNG consumer)
        any_change |= _fall_phase(state)
        # v. Generator firing
        any_change |= _generator_phase(state)
        # vi. Reachability + ?-bucket reveal
        _reachability_and_reveal_phase(state)
        # vii. Pin destruction (cascading; re-runs reachability internally)
        any_change |= _pin_destruction_phase(state)
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
    # All pickable cells (PlainBucket, QuestionBucket, thawed IceBucket) carry a `color` field.
    new_bucket = BufferBucket(color=cell.color, capacity=state.bucket_capacity, fill=0)
    state.reserve[r][c] = None
    state.move_counter += 1

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
