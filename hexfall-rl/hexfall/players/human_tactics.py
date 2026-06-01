"""Human-tactics (depth-0) player — the *policy* axis of the solver suite.

Like :class:`~hexfall.players.greedy.GreedyPlayer` this is a depth-0, obs-only
heuristic: no search, no lookahead, no internal RNG (depth is already covered by
``LookaheadPlayer`` / ``MCTSPlayer``). The only difference from greedy is a
richer, multi-component scoring function meant to encode the kinds of tactical
reasoning a *human* applies when picking the next bucket. Each LEGAL pick is
scored; the argmax is taken with a deterministic lowest-index tie-break, so
``act`` is a pure (deterministic) function of the observation.

Scoring components (each individually weighted; signs and weights below were
chosen ONCE from the rules, NOT tuned to winrate or oracle correlation):

1. ``matched_now``  (+, ``W_MATCHED_NOW = 3.0``) — dominant progress signal.
   Number of bottom-row stacks whose exposed top slice is the candidate's
   colour, i.e. how much immediately-consumable work this pick has. A pick that
   can start draining the bottom row keeps the 5-slot buffer productive. Per the
   §4 same-colour collision rule, if a same-colour bucket is *already* in the
   buffer and is not nearly full, those slices are already claimed by it and the
   new bucket would sit globally idle — so ``matched_now`` is zeroed in that
   case (the matches are not *this* pick's to make).

2. ``same_color_idle`` (−, ``W_SAME_COLOR_IDLE = 2.0``) — penalise committing a
   slot to a colour already handled by a non-full, *not-nearly-full* same-colour
   bucket. Per §4 the less-full bucket pulls from nowhere until the fuller one
   fills, so the new bucket burns a buffer slot doing nothing — pure deadlock
   risk. (No penalty when the existing bucket is nearly full: it will clear soon
   and the new bucket then takes over, so the commit is reasonable.)

3. ``speculation`` (−, ``W_SPECULATION = 1.5``, scales with buffer occupancy) —
   penalise committing to a colour that appears *nowhere* in the currently
   visible field (neither a bottom-row top nor a partially-visible upper slice,
   §3 visibility). The colour may be buried or absent; gambling a slot on it is
   fine with an empty buffer but dangerous with a full one, so the penalty
   scales with the post-pick occupancy.

4. ``buffer_pressure`` (−, ``W_BUFFER_PRESSURE = 2.0``, scales with occupancy) —
   penalise *any* immediately-unproductive pick (``matched_now == 0``), scaled
   by post-pick occupancy. Dominant in the ``>= 4/5`` region, where committing a
   non-draining bucket walks the buffer toward the structural deadlock lose
   condition (§7).

5. ``pin_setup`` (+, ``W_PIN_SETUP = 4.0``) — reward picking the *destruction
   cell* of an active pin: the cell one step opposite the pin's facing (§5).
   Emptying it destroys the pin **iff** no live generator refills it on the same
   tick (destruction is quiescence-driven, §5), opening previously-unreachable
   reserve regions. A high-value strategic play humans prioritise, weighted to
   outrank a single immediate match.

6. ``ice_timing`` (+, ``W_ICE_TIMING = 0.5``, small / conservative) — when the
   field shows a colour obtainable only from a still-frozen ice bucket (a colour
   in the visible field that no currently pickable or buffered bucket can
   supply), slightly prefer immediately-productive picks. Every pick advances the
   move counter by one toward the nearest thaw threshold (§5 ice); the way to
   *keep* advancing without deadlocking is to drain what is consumable, so the
   nudge goes to ``matched_now > 0`` picks and grows as the nearest thaw nears.
   This is the fuzziest component and is deliberately the smallest weight.

Cost per decision is O(reserve_area + n_legal * (n_buffer + n_pins)) — the same
order as greedy, plus cheap pin/generator/ice bookkeeping computed once.
"""
from __future__ import annotations

from collections import Counter

import numpy as np

from hexfall.env import CELL_TYPE_IDS, ID_TO_DIRECTION, HexFallEnv

# Reserve-grid movement offsets (row, col). ``ID_TO_DIRECTION`` yields lowercase
# direction names, so both maps below are keyed lowercase.
_DIRECTIONS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}

# A pin is destroyed when the cell one step OPPOSITE its facing ends a tick
# empty (§5). These offsets land on that destruction cell from the pin origin.
_PIN_BEHIND_OFFSET = {"up": (1, 0), "down": (-1, 0), "left": (0, 1), "right": (0, -1)}


# ---------------------------------------------------------------------------
# Observation helpers (module-level, mirroring greedy.py's structure)
# ---------------------------------------------------------------------------

def _sentinels(obs: dict, env: HexFallEnv | None) -> tuple[int, int]:
    """Return ``(HIDDEN_COLOR_ID, NO_COLOR_ID)`` for this level.

    These are fixed at level load (``num_colors`` / ``num_colors + 1``), so
    reading them off ``env`` is deterministic. When ``env`` is not supplied we
    fall back to inferring them from the colour arrays: ``NO_COLOR_ID`` padding
    is virtually always present (empty reserve cells, empty buffer slots, field
    padding) so the global max colour id is ``NO_COLOR_ID`` and ``HIDDEN`` is one
    below. Only the small ``ice_timing`` component depends on this distinction.
    """
    if env is not None and hasattr(env, "HIDDEN_COLOR_ID") and hasattr(env, "NO_COLOR_ID"):
        return int(env.HIDDEN_COLOR_ID), int(env.NO_COLOR_ID)
    seen: list[int] = []
    for key in ("field_visible", "reserve_color", "buffer_colors"):
        arr = obs.get(key)
        if arr is not None and arr.size:
            seen.append(int(arr.max()))
    no_id = max(seen) if seen else 0
    return no_id - 1, no_id


def _bottom_row_top_colors(obs: dict) -> list[int]:
    """Top (consumable) colour id of each bottom-row stack, with multiplicity.

    Mirrors :func:`hexfall.players.greedy._visible_bottom_colors` but returns a
    list (not a set) so callers can count how many bottom-row stacks expose a
    given colour. Bottom-row stacks are always fully visible (§3), so every entry
    is a concrete colour id, never a sentinel.
    """
    heights = obs["field_heights"]
    visible = obs["field_visible"]
    if heights.size == 0:
        return []
    fh, fc = heights.shape
    rows_with_stack = [r for r in range(fh) if bool((heights[r] > 0).any())]
    if not rows_with_stack:
        return []
    bottom_row = max(rows_with_stack)
    out: list[int] = []
    for c in range(fc):
        if heights[bottom_row, c] > 0:
            out.append(int(visible[bottom_row, c, 0]))
    return out


def _concrete_field_colors(obs: dict, hidden_id: int, no_id: int) -> set[int]:
    """Set of concrete colour ids visible anywhere in the field (sentinels dropped)."""
    fv = obs["field_visible"]
    if fv.size == 0:
        return set()
    return {int(v) for v in np.unique(fv) if v != hidden_id and v != no_id}


def _occupied_buffer_buckets(obs: dict) -> list[tuple[int, int, int]]:
    """``(color, fill, capacity)`` for each occupied buffer slot."""
    occ = obs["buffer_occupied"]
    colors = obs["buffer_colors"]
    fills = obs["buffer_fills"]
    caps = obs["buffer_capacities"]
    out: list[tuple[int, int, int]] = []
    for i in range(len(occ)):
        if occ[i]:
            out.append((int(colors[i]), int(fills[i]), int(caps[i])))
    return out


def _generator_target_cells(obs: dict) -> set[tuple[int, int]]:
    """Cells a live generator (``remaining > 0``) faces into.

    If the candidate pick empties such a cell, the generator refills it on the
    same tick, so a pin behind it is *not* destroyed (§5 quiescence rule).
    """
    cell_type = obs["reserve_cell_type"]
    facing = obs["reserve_generator_facing"]
    remaining = obs["reserve_generator_remaining"]
    R, C = cell_type.shape
    gen_id = CELL_TYPE_IDS["generator"]
    targets: set[tuple[int, int]] = set()
    for r in range(R):
        for c in range(C):
            if cell_type[r, c] != gen_id or remaining[r, c] <= 0:
                continue
            dr, dc = _DIRECTIONS[ID_TO_DIRECTION[int(facing[r, c])]]
            tr, tc = r + dr, c + dc
            if 0 <= tr < R and 0 <= tc < C:
                targets.add((tr, tc))
    return targets


def _pin_destruction_cells(obs: dict) -> dict[tuple[int, int], int]:
    """Map each active pin's destruction cell -> number of pins destroyed there."""
    origin_r = obs["pins_origin_row"]
    origin_c = obs["pins_origin_col"]
    direction = obs["pins_direction"]
    destroyed = obs["pins_destroyed"]
    dest: dict[tuple[int, int], int] = {}
    for i in range(len(origin_r)):
        if destroyed[i]:
            continue
        dr, dc = _PIN_BEHIND_OFFSET[ID_TO_DIRECTION[int(direction[i])]]
        cell = (int(origin_r[i]) + dr, int(origin_c[i]) + dc)
        dest[cell] = dest.get(cell, 0) + 1
    return dest


def _frozen_ice_remaining_thaw(obs: dict) -> list[int]:
    """Remaining-thaw counts of all still-frozen ice buckets."""
    cell_type = obs["reserve_cell_type"]
    remaining_thaw = obs["reserve_remaining_thaw"]
    frozen_id = CELL_TYPE_IDS["ice_bucket_frozen"]
    R, C = cell_type.shape
    out: list[int] = []
    for r in range(R):
        for c in range(C):
            if cell_type[r, c] == frozen_id:
                out.append(int(remaining_thaw[r, c]))
    return out


class HumanTacticsPlayer:
    """Depth-0 human-tactics heuristic player. Obs-only; deterministic.

    ``env`` is read only for the two (fixed-at-load) sentinel colour ids; it is
    never forked or stepped, and no RNG is used, so ``act`` is a deterministic
    function of the observation. See the module docstring for the full component
    and weight rationale.
    """

    # Component weights — pre-committed, NOT tuned (see module docstring).
    W_MATCHED_NOW = 3.0
    W_SAME_COLOR_IDLE = 2.0
    W_SPECULATION = 1.5
    W_BUFFER_PRESSURE = 2.0
    W_PIN_SETUP = 4.0
    W_ICE_TIMING = 0.5

    #: A same-colour buffer bucket within this many slices of full is treated as
    #: "nearly full": it will clear soon, so a new same-colour pick is not idle.
    NEARLY_FULL_MARGIN = 2

    def act(self, obs: dict, env: HexFallEnv | None = None) -> int:
        mask = obs["action_mask"]
        reserve_color = obs["reserve_color"]
        n_cols = reserve_color.shape[1]

        hidden_id, no_id = _sentinels(obs, env)

        # --- State-level quantities, computed once (constant across candidates).
        bottom_counts = Counter(_bottom_row_top_colors(obs))
        buffer_buckets = _occupied_buffer_buckets(obs)
        n_slots = int(obs["buffer_occupied"].shape[0])
        n_occupied = int(obs["buffer_occupied"].sum())
        post_occ = (n_occupied + 1) / n_slots if n_slots else 1.0
        concrete_field = _concrete_field_colors(obs, hidden_id, no_id)
        gen_targets = _generator_target_cells(obs)
        pin_dest = _pin_destruction_cells(obs)

        legal = [a for a in range(mask.shape[0]) if mask[a]]

        # Ice context: is any visible field colour unobtainable from the buckets
        # we can deploy right now (legal picks + non-full buffer buckets)?
        frozen_thaw = _frozen_ice_remaining_thaw(obs)
        obtainable: set[int] = {int(reserve_color[divmod(a, n_cols)]) for a in legal}
        obtainable |= {col for (col, fill, cap) in buffer_buckets if fill < cap}
        ice_blocked = bool(frozen_thaw) and bool(concrete_field - obtainable)
        nearest_thaw = min(frozen_thaw) if frozen_thaw else 0

        ctx = {
            "bottom_counts": bottom_counts,
            "buffer_buckets": buffer_buckets,
            "post_occ": post_occ,
            "concrete_field": concrete_field,
            "gen_targets": gen_targets,
            "pin_dest": pin_dest,
            "ice_blocked": ice_blocked,
            "nearest_thaw": nearest_thaw,
        }

        best_action: int | None = None
        best_score: float | None = None
        for a in legal:
            r, c = divmod(a, n_cols)
            color = int(reserve_color[r, c])
            score = self._score(color, r, c, ctx)
            # Strict ">" keeps the lowest-index action on ties (deterministic).
            if best_score is None or score > best_score:
                best_score = score
                best_action = a

        # Contract: act() is only called with >= 1 legal action; fall back to 0.
        return best_action if best_action is not None else 0

    def _score(self, color: int, r: int, c: int, ctx: dict) -> float:
        """Weighted sum of the six components for one candidate pick."""
        comp = self._component_values(color, r, c, ctx)
        return (
            self.W_MATCHED_NOW * comp["matched_now"]
            - self.W_SAME_COLOR_IDLE * comp["same_color_idle"]
            - self.W_SPECULATION * comp["speculation"]
            - self.W_BUFFER_PRESSURE * comp["buffer_pressure"]
            + self.W_PIN_SETUP * comp["pin_setup"]
            + self.W_ICE_TIMING * comp["ice_timing"]
        )

    def _component_values(self, color: int, r: int, c: int, ctx: dict) -> dict[str, float]:
        """Raw (unweighted, non-negative) value of each component for one pick.

        Penalties are returned as non-negative magnitudes; their negative sign is
        applied in :meth:`_score`. Exposed as its own method for transparency and
        white-box testing.
        """
        # Existing non-full same-colour buffer buckets, and whether any of them
        # is NOT nearly full (so it will keep claiming this colour's slices).
        same_color_rooms = [
            cap - fill
            for (col, fill, cap) in ctx["buffer_buckets"]
            if col == color and fill < cap
        ]
        blocked = any(room > self.NEARLY_FULL_MARGIN for room in same_color_rooms)

        # 1. matched_now — bottom-row demand for this colour, unless already claimed.
        matched_now = 0.0 if blocked else float(ctx["bottom_counts"].get(color, 0))

        # 2. same_color_idle — committing behind a not-nearly-full same-colour bucket.
        same_color_idle = 1.0 if blocked else 0.0

        # 3. speculation — colour invisible anywhere in the field, scaled by occupancy.
        speculative = color not in ctx["concrete_field"]
        speculation = ctx["post_occ"] if speculative else 0.0

        # 4. buffer_pressure — unproductive pick, scaled by occupancy.
        buffer_pressure = ctx["post_occ"] if matched_now == 0.0 else 0.0

        # 5. pin_setup — destruction cell of an active pin with no live refill.
        if (r, c) in ctx["pin_dest"] and (r, c) not in ctx["gen_targets"]:
            pin_setup = float(ctx["pin_dest"][(r, c)])
        else:
            pin_setup = 0.0

        # 6. ice_timing — small nudge to drain while a needed colour stays frozen.
        if ctx["ice_blocked"] and matched_now > 0.0:
            ice_timing = 1.0 / (1.0 + ctx["nearest_thaw"])
        else:
            ice_timing = 0.0

        return {
            "matched_now": matched_now,
            "same_color_idle": same_color_idle,
            "speculation": speculation,
            "buffer_pressure": buffer_pressure,
            "pin_setup": pin_setup,
            "ice_timing": ice_timing,
        }
