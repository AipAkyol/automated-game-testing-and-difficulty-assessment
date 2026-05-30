"""Difficulty oracle: structural features + player winrates -> human winrate.

The oracle predicts a level's *human* win rate (Anil's "% Win Rate") from a
small, interpretable feature vector:

    [5 structural features]  +  [3 player winrates]  =  8 features

Structural features are extracted from a *loaded* GameState (via the level
loader, never by hand-parsing JSON). Player winrates are the win fractions of
the greedy / lookahead(depth-2) / mcts agents.

Asymmetry between fit and predict (intentional, documented):
  - ``fit`` reads the THREE player winrates straight off each record. Those are
    the 20-episode winrates from ``outputs/eval_matrix.csv`` (precomputed; fit
    does NOT re-run any agent).
  - ``predict`` takes a BARE level and computes the three winrates fresh by
    running the agents via :func:`evaluate` with ``n_episodes=10`` and a FIXED
    seed. Ten episodes (vs. the matrix's twenty) halves predict cost for Issue
    D's generator loop; the extra sampling variance is an accepted tradeoff.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from hexfall.level_loader import load_level
from hexfall.types import GameState, Generator, IceBucket, PlainBucket, QuestionBucket

# --- Configuration -----------------------------------------------------------

#: Structural feature names, in vector order. The full feature vector appends
#: the three player-winrate columns (greedy, lookahead, mcts) after these.
STRUCTURAL_FEATURE_NAMES = [
    "color_count",
    "total_slice_count",
    "pin_count",
    "ice_count",
    "reserve_area",
]
PLAYER_FEATURE_NAMES = ["greedy_winrate", "lookahead_winrate", "mcts_winrate"]
FEATURE_NAMES = STRUCTURAL_FEATURE_NAMES + PLAYER_FEATURE_NAMES

#: Ridge regularization strength. PRE-COMMITTED — not tuned to any metric.
ALPHA = 1.0

#: predict() agent-evaluation budget. Deliberately lighter than the eval
#: matrix (20 episodes) to keep the Issue-D generator loop affordable.
PREDICT_N_EPISODES = 10
PREDICT_SEED = 0


# --- Structural feature extraction -------------------------------------------

def extract_structural_features(state: GameState) -> list[float]:
    """Return the 5 structural features of a loaded level, in vector order.

    All features are read from the post-quiescence loaded state, never from
    declared level metadata.
    """
    # color_count: distinct colors ACTUALLY PRESENT in the loaded state (field
    # slices + reserve cells), NOT the declared editorMeta.colorCount. Reading
    # the actual present set is robust to any declared-vs-actual count
    # discrepancy (a trap flagged on levels 87 and 91).
    colors: set[str] = set()
    for slices in state.field.values():
        colors.update(slices)
    for row in state.reserve:
        for cell in row:
            if isinstance(cell, (PlainBucket, QuestionBucket, IceBucket)):
                colors.add(cell.color)
            elif isinstance(cell, Generator):
                colors.update(cell.queue)
    color_count = len(colors)

    # total_slice_count: total colored slices in the starting hex field.
    total_slice_count = sum(len(slices) for slices in state.field.values())

    # pin_count: number of Pin objects overlaying the reserve grid.
    pin_count = len(state.pins)

    # ice_count: number of frozen IceBucket collectors (loaded state => all
    # ice starts frozen, but guard on .thawed to be explicit).
    ice_count = sum(
        1
        for row in state.reserve
        for cell in row
        if isinstance(cell, IceBucket) and not cell.thawed
    )

    # reserve_area: collector-area footprint as ONE scalar (width x height).
    reserve_area = state.reserve_cols * state.reserve_rows

    return [
        float(color_count),
        float(total_slice_count),
        float(pin_count),
        float(ice_count),
        float(reserve_area),
    ]


def _coerce_state(level: str | Path | GameState, *, seed: int | None = None) -> GameState:
    """Accept either an already-loaded GameState or a path, return a GameState."""
    if isinstance(level, GameState):
        return level
    return load_level(level, seed=seed)


def _player_winrates(level_path: str | Path) -> list[float]:
    """Compute the three player winrates fresh (10 episodes, fixed seed)."""
    # Lazy import: avoids dragging the env/players import graph into callers
    # that only want structural features.
    from hexfall.players import GreedyPlayer, LookaheadPlayer, MCTSPlayer, evaluate

    players = [
        GreedyPlayer(),
        LookaheadPlayer(depth=2),  # matches the eval matrix's "lookahead" == depth-2
        MCTSPlayer(n_rollouts=100),
    ]
    return [
        evaluate(p, level_path, n_episodes=PREDICT_N_EPISODES, seed=PREDICT_SEED)
        for p in players
    ]


# --- Oracle ------------------------------------------------------------------

class Oracle:
    """Ridge regression over structural features + player winrates.

    Internals are an sklearn ``Pipeline(StandardScaler -> Ridge(alpha=1.0))``.
    Ridge penalizes coefficient magnitude and is therefore scale-sensitive, so
    standardization inside the pipeline is mandatory (not optional polish).
    """

    def __init__(self, alpha: float = ALPHA):
        self.alpha = alpha
        self.pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=alpha)),
            ]
        )
        self._fitted = False

    @staticmethod
    def _record_to_features(record) -> list[float]:
        """Build an 8-feature vector from a fit record.

        ``record`` carries a loaded level plus its three precomputed player
        winrates. Accepts either a mapping with keys ``state``/``greedy_winrate``
        /``lookahead_winrate``/``mcts_winrate`` or an object with matching
        attributes.
        """
        get = (record.get if isinstance(record, dict) else lambda k: getattr(record, k))
        state = get("state")
        structural = extract_structural_features(state)
        players = [
            float(get("greedy_winrate")),
            float(get("lookahead_winrate")),
            float(get("mcts_winrate")),
        ]
        return structural + players

    def fit(self, levels, winrates) -> "Oracle":
        """Fit on records that already carry their player winrates.

        Args:
            levels: sequence of records; each carries a loaded level + the three
                precomputed player winrates (from the eval matrix). Player
                winrates are READ from the record — no agent is run here.
            winrates: human "% Win Rate" targets, aligned with ``levels``.
        """
        X = np.array([self._record_to_features(r) for r in levels], dtype=float)
        y = np.asarray(winrates, dtype=float)
        self.pipeline.fit(X, y)
        self._fitted = True
        return self

    def predict(self, level: str | Path | GameState) -> float:
        """Predict the human winrate for a BARE level (path or loaded state).

        Extracts the 5 structural features and computes the 3 player winrates
        fresh (10 episodes, fixed seed — see module docstring on the fit/predict
        asymmetry). Output is clamped to [0, 1].
        """
        if not self._fitted:
            raise RuntimeError("Oracle.predict called before fit()")

        state = _coerce_state(level)
        structural = extract_structural_features(state)

        # Player winrates need a file path to spin up the env; if we were handed
        # a loaded GameState we cannot re-evaluate, so require a path here.
        if isinstance(level, GameState):
            raise ValueError(
                "predict() needs a level path to compute player winrates; "
                "pass the path, not a loaded GameState"
            )
        players = _player_winrates(level)

        X = np.array([structural + players], dtype=float)
        raw = float(self.pipeline.predict(X)[0])
        return max(0.0, min(1.0, raw))

    @property
    def coefficients(self) -> dict[str, float]:
        """Fitted Ridge coefficients keyed by feature name (post-scaling)."""
        if not self._fitted:
            raise RuntimeError("coefficients requested before fit()")
        ridge: Ridge = self.pipeline.named_steps["ridge"]
        return dict(zip(FEATURE_NAMES, (float(c) for c in ridge.coef_)))
