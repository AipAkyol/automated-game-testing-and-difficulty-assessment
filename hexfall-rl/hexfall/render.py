from hexfall.env import HexFallEnv
from hexfall.game import compute_reachability, pin_blocked_cells
from hexfall.types import Generator, IceBucket, PlainBucket, QuestionBucket, Wall


_COLOR_W = 6


def _trunc(s: str) -> str:
    return s if len(s) <= _COLOR_W else s[:_COLOR_W]


def _render_field_agent(env: HexFallEnv, obs: dict) -> list[str]:
    field_visible = obs["field_visible"]
    field_heights = obs["field_heights"]
    HID = env.HIDDEN_COLOR_ID
    stacks: dict[tuple[int, int], tuple[list[str], int]] = {}
    fh, fc = field_heights.shape
    for r in range(fh):
        for c in range(fc):
            h = int(field_heights[r, c])
            if h == 0:
                continue
            labels: list[str] = []
            for d in range(h):
                cid = int(field_visible[r, c, d])
                if cid == HID:
                    labels.append("?")
                else:
                    labels.append(env.id_to_color[cid])
            stacks[(c, r)] = (labels, h)
    return _render_field_from_stacks(stacks)


def _render_field_full(state) -> list[str]:
    return _render_field_from_stacks(
        {pos: (slices, len(slices)) for pos, slices in state.field.items() if slices}
    )


def _render_field_from_stacks(
    stacks: dict[tuple[int, int], tuple[list[str], int]],
) -> list[str]:
    """stacks maps (col,row) -> (slice_labels_top_to_bottom, height).

    A label of "?" means a hidden slice; rendered as "??".
    """
    lines: list[str] = ["[1] HEX FIELD"]
    if not stacks:
        lines.append("  (empty)")
        return lines

    max_row = max(r for (_, r) in stacks.keys())
    for r in range(max_row + 1):
        row_positions = sorted(
            [(c, rr) for (c, rr) in stacks.keys() if rr == r]
        )
        indent = "  " if (r & 1) else ""
        if not row_positions:
            lines.append(f"{indent}Row {r}: (empty)")
            continue
        lines.append(f"{indent}Row {r}:")
        for (c, rr) in row_positions:
            slices, _height = stacks[(c, rr)]
            if not slices:
                lines.append(f"{indent}  ({c},{rr}): ---")
                continue
            parts = ["??" if s == "?" else _trunc(s) for s in slices]
            lines.append(f"{indent}  ({c},{rr}): " + ", ".join(parts))
    return lines


def _render_buffer(state) -> list[str]:
    lines = [
        "[2] BUFFER",
        "  (slots only persist between observations when bucket_capacity > pulls/loop; "
        "small-capacity levels typically show empty)",
    ]
    for i, slot in enumerate(state.buffer):
        if slot is None:
            lines.append(f"  Slot {i}: [EMPTY]")
        else:
            lines.append(
                f"  Slot {i}: {_trunc(slot.color)} ({slot.fill}/{slot.capacity})"
            )
    return lines


def _render_reserve(state, mode: str) -> list[str]:
    reach = compute_reachability(state)
    blocked = pin_blocked_cells(state)
    rows, cols = state.reserve_rows, state.reserve_cols

    grid: list[list[str]] = []
    for r in range(rows):
        row_cells: list[str] = []
        for c in range(cols):
            cell = state.reserve[r][c]
            underneath = _format_reserve_cell(cell, reach[r][c], mode, state.move_counter)
            if (r, c) in blocked:
                row_cells.append(f"PIN[{underneath}]")
            else:
                row_cells.append(underneath)
        grid.append(row_cells)

    col_widths = [
        max(len(grid[r][c]) for r in range(rows))
        for c in range(cols)
    ]

    lines = ["[3] RESERVE GRID"]
    for r in range(rows):
        cells = [grid[r][c].ljust(col_widths[c]) for c in range(cols)]
        lines.append("  " + " | ".join(cells))
    lines.append("  (R)=reachable, (-)=not reachable, generators/walls never pickable")
    return lines


def _format_reserve_cell(cell, reachable: bool, mode: str, move_counter: int) -> str:
    if cell is None:
        return "...."
    if isinstance(cell, Wall):
        return "WALL"
    flag = "R" if reachable else "-"
    if isinstance(cell, PlainBucket):
        return f"PL:{_trunc(cell.color)}({flag})"
    if isinstance(cell, QuestionBucket):
        if mode == "full":
            return f"QB:{_trunc(cell.color)}({flag})"
        if cell.revealed:
            return f"QB:{_trunc(cell.color)}({flag})"
        return f"QB:???({flag})"
    if isinstance(cell, IceBucket):
        remaining = max(0, cell.thaw_threshold - move_counter)
        if cell.thawed:
            return f"IC:{_trunc(cell.color)}/T({flag})"
        if mode == "full":
            return f"IC:{_trunc(cell.color)}/F{remaining}({flag})"
        return f"IC:???/F{remaining}({flag})"
    if isinstance(cell, Generator):
        if mode == "full":
            queue_str = "[" + ",".join(_trunc(q) for q in cell.queue) + "]"
            return f"GN:{cell.facing},{cell.remaining},{queue_str}"
        return f"GN:{cell.facing},{cell.remaining}"
    return "?CELL?"


def main(argv: list[str] | None = None) -> int:
    """CLI: render the initial state of a level."""
    import argparse
    parser = argparse.ArgumentParser(
        description="Render the initial observation of a Hex Fall level."
    )
    parser.add_argument("--level", dest="level_path", required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--mode", choices=("agent", "full"), default="agent",
        help="agent = observation-limited; full = full simulator state.",
    )
    args = parser.parse_args(argv)

    env = HexFallEnv(args.level_path, seed=args.seed)
    env.reset()
    print(render(env, args.mode))
    return 0


def render(env: HexFallEnv, mode: str = "agent") -> str:
    if mode not in ("agent", "full"):
        raise ValueError(f"mode must be 'agent' or 'full', got {mode!r}")

    state = env._state
    if mode == "agent":
        obs = env.get_obs()
        field_lines = _render_field_agent(env, obs)
    else:
        field_lines = _render_field_full(state)

    buffer_lines = _render_buffer(state)
    reserve_lines = _render_reserve(state, mode)

    return "\n".join(field_lines + [""] + buffer_lines + [""] + reserve_lines)
