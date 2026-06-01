"""Clog difficulty features (Issue H) — pure structural extractors.

Three candidate difficulty features computed purely and deterministically from a
*loaded* GameState. Kept in a separate module from ``oracle.py`` so they can be
ablated cleanly in Phase 3. Every value is read from loaded contents (field
slices + reserve cells) — never from ``editorMeta``, whose ``colorCount`` /
``maxColorsPerStack`` knobs are known to mismatch actual contents.

  fcc — Frontier Colour Concentration:
        distinct top-slice colours across the bottom-row stacks / C, where C is
        the loaded-state colour count (reused from
        ``oracle.extract_structural_features`` so it matches the oracle exactly).
  ctd — Colour Transition Density:
        mean over stacks of height >= 2 of (adjacent unequal-colour transitions)
        / (height - 1); 0.0 when no stack has height >= 2. Transitions are
        counted directly, so the value is correct for 1, 2, or 3+ distinct
        colours in a stack (no binary shortcut).
  rhc — Reserve Hardness Composite:
        (woodBox + frozen-ice + generator-queued)
        / (plain + woodBox + ice + generator-queued).

All three lie in [0, 1].
"""
from __future__ import annotations

from hexfall.oracle import STRUCTURAL_FEATURE_NAMES, extract_structural_features
from hexfall.types import (
    GameState,
    Generator,
    IceBucket,
    PlainBucket,
    QuestionBucket,
)


def _color_count(state: GameState) -> int:
    """C — distinct colours in the loaded state, reusing the oracle's definition.

    ``extract_structural_features`` computes ``color_count`` as the distinct
    colours across field slices + reserve bucket colours + generator-queue
    colours. Reusing it keeps C identical to the oracle's existing feature.
    """
    feats = dict(zip(STRUCTURAL_FEATURE_NAMES, extract_structural_features(state)))
    return int(feats["color_count"])


def fcc(state: GameState) -> float:
    """Frontier Colour Concentration: distinct bottom-row top-slice colours / C."""
    # Bottom row per hexfall/game.py _pull_phase: the max row-index among
    # non-empty stacks. fcc needs only the bottom row, so no hex-adjacency
    # convention is introduced.
    non_empty = [(col, row) for (col, row), slices in state.field.items() if slices]
    if not non_empty:
        return 0.0
    bottom_row = max(row for _, row in non_empty)
    top_colours = {
        state.field[(col, row)][0]
        for (col, row) in non_empty
        if row == bottom_row
    }
    c = _color_count(state)
    return len(top_colours) / c if c else 0.0


def ctd(state: GameState) -> float:
    """Colour Transition Density: mean over height>=2 stacks of transitions/(h-1)."""
    ratios: list[float] = []
    for slices in state.field.values():
        height = len(slices)
        if height >= 2:
            transitions = sum(
                1 for i in range(height - 1) if slices[i] != slices[i + 1]
            )
            ratios.append(transitions / (height - 1))
    return sum(ratios) / len(ratios) if ratios else 0.0


def rhc(state: GameState) -> float:
    """Reserve Hardness Composite over reserve bucket sources."""
    plain = wood = ice_total = ice_frozen = gen_queued = 0
    for row in state.reserve:
        for cell in row:
            if isinstance(cell, PlainBucket):
                plain += 1
            elif isinstance(cell, QuestionBucket):
                wood += 1
            elif isinstance(cell, IceBucket):
                ice_total += 1
                if not cell.thawed:
                    ice_frozen += 1
            elif isinstance(cell, Generator):
                gen_queued += len(cell.queue)
    denom = plain + wood + ice_total + gen_queued
    if denom == 0:
        return 0.0
    return (wood + ice_frozen + gen_queued) / denom


def extract_clog_features(state: GameState) -> dict[str, float]:
    """Return all three clog features keyed by name."""
    return {"fcc": fcc(state), "ctd": ctd(state), "rhc": rhc(state)}
