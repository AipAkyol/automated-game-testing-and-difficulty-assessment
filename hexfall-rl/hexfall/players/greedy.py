"""Greedy (depth-0) player.

Heuristic: **maximize buffer-color matchedness to the colors currently
consumable at the bottom row of the hex field.** A buffer bucket whose color is
exposed as the top slice of a bottom-row stack can make progress on the next
tick; one whose color is not currently exposed sits idle. Keeping the 5-slot
buffer "productive" is a cheap proxy for forward progress and avoids committing
the buffer to a color that cannot currently drain (the core deadlock risk per
HEXFALL_RULES.md §4, §7).

The player is obs-only: it ignores ``env`` and runs no simulation. Cost per
decision is O(n_legal * n_buffer) plus O(field_width) to collect the consumable
colors.
"""
from __future__ import annotations

from collections.abc import Iterable

from hexfall.env import HexFallEnv


def _visible_bottom_colors(obs: dict) -> set[int]:
    """Color ids that are the top (consumable) slice of a bottom-row stack.

    The bottom row is the largest field-row index that still holds a stack
    (Paxie convention: highest ``y``; see LEVEL_FORMAT.md §6.1). Bottom-row
    stacks are always fully visible (HEXFALL_RULES.md §3), so the depth-0 entry
    is a concrete color id, never the hidden sentinel.
    """
    heights = obs["field_heights"]
    visible = obs["field_visible"]
    if heights.size == 0:
        return set()
    fh, fc = heights.shape
    rows_with_stack = [r for r in range(fh) if bool((heights[r] > 0).any())]
    if not rows_with_stack:
        return set()
    bottom_row = max(rows_with_stack)
    colors: set[int] = set()
    for c in range(fc):
        if heights[bottom_row, c] > 0:
            colors.add(int(visible[bottom_row, c, 0]))
    return colors


def _buffer_match(color_ids: Iterable[int], consumable: set[int]) -> int:
    """Count how many of the given buffer colors are currently consumable."""
    return sum(1 for cid in color_ids if cid in consumable)


def _occupied_buffer_colors(obs: dict) -> list[int]:
    """Color ids of the currently-occupied buffer slots."""
    occupied = obs["buffer_occupied"]
    buffer_colors = obs["buffer_colors"]
    return [int(buffer_colors[i]) for i in range(len(occupied)) if occupied[i]]


class GreedyPlayer:
    """Depth-0 heuristic player. Obs-only: ``env`` is accepted but ignored."""

    def act(self, obs: dict, env: HexFallEnv | None = None) -> int:
        mask = obs["action_mask"]
        reserve_color = obs["reserve_color"]
        n_cols = reserve_color.shape[1]
        consumable = _visible_bottom_colors(obs)
        base = _occupied_buffer_colors(obs)

        best_action: int | None = None
        best_score: int | None = None
        for a in range(mask.shape[0]):
            if not mask[a]:
                continue
            # action -> (row, col) per HexFallEnv.step: row = a // cols, col = a % cols.
            r, c = divmod(a, n_cols)
            new_color = int(reserve_color[r, c])
            # Score the hypothetical buffer = current buckets + the one we'd pick.
            score = _buffer_match(base + [new_color], consumable)
            # Strict ">" keeps the lowest-index action on ties.
            if best_score is None or score > best_score:
                best_score = score
                best_action = a

        # Contract: act() is only called at states with >= 1 legal action. Fall
        # back to 0 rather than returning None if a caller violates that.
        return best_action if best_action is not None else 0

    def score_state(self, obs: dict) -> float:
        """Heuristic value of a state: the number of occupied buffer buckets
        whose color is currently consumable at a bottom-row stack top.

        Used as the lookahead leaf value. Unlike :meth:`act` (which scores a
        *hypothetical* post-pick buffer), this scores the buffer as-is.
        """
        consumable = _visible_bottom_colors(obs)
        return float(_buffer_match(_occupied_buffer_colors(obs), consumable))
