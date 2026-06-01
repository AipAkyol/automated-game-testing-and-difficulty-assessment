"""Run a player on a level for N episodes and report outcomes.

Two entry points share one deterministic episode loop:

- :func:`evaluate` returns the scalar win fraction (the original API; oracle.py
  and run_eval_matrix.py depend on this exact signature and return type).
- :func:`evaluate_graded` returns a richer dict with per-episode results
  (win, slices-cleared fraction, moves survived) plus per-(player, level)
  aggregates — for testing whether graded solver outcomes carry predictive
  signal beyond binary winrate (Issue F).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from hexfall.env import HexFallEnv
from hexfall.players.base import Player


def _field_slice_count(env: HexFallEnv) -> int:
    """Total slices currently in the hex field (sum of stack heights)."""
    return sum(len(slices) for slices in env._state.field.values())


def _run_episodes(
    player: Player,
    level_path: str | Path,
    n_episodes: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Run ``n_episodes`` and return one result dict per episode.

    Each result dict has keys ``win`` (bool), ``slices_cleared_fraction``
    (float in [0, 1]) and ``moves_survived`` (int). A single env is reused
    across episodes; ``reset(seed=seed + i)`` rebuilds the full state from the
    level file plus the seed, so RNG state never bleeds between episodes and
    the per-episode results are bit-exact across runs of the same arguments.
    """
    if n_episodes < 1:
        raise ValueError(f"n_episodes must be >= 1, got {n_episodes}")

    env = HexFallEnv(level_path=str(level_path))
    results: list[dict[str, Any]] = []
    for i in range(n_episodes):
        obs, _info = env.reset(seed=seed + i)
        # "loaded level before any moves": measured at the post-load quiescent
        # state (move_counter == 0), before the player picks anything.
        initial_total_slices = _field_slice_count(env)
        terminated = truncated = False
        reward = 0.0
        while not (terminated or truncated):
            # Defensive: a quiescent state with no legal action is a terminal
            # lose reached without an explicit step. Well-formed levels reach
            # terminal states through step(); this guard just avoids handing an
            # empty action set to the player.
            if not obs["action_mask"].any():
                break
            action = player.act(obs, env)
            obs, reward, terminated, truncated, _info = env.step(action)

        win = bool(terminated and reward > 0)
        slices_cleared = initial_total_slices - _field_slice_count(env)
        if initial_total_slices > 0:
            fraction = slices_cleared / initial_total_slices
        else:
            # Degenerate empty-field level: nothing to clear; treat as fully
            # cleared rather than dividing by zero.
            fraction = 1.0
        fraction = min(1.0, max(0.0, fraction))
        results.append(
            {
                "win": win,
                "slices_cleared_fraction": fraction,
                # move_counter at terminal state == full episode length on a
                # win, no special-casing (HEXFALL_MDP_SPEC.md §3.6).
                "moves_survived": int(env._state.move_counter),
            }
        )
    return results


def evaluate(
    player: Player,
    level_path: str | Path,
    n_episodes: int,
    seed: int = 42,
) -> float:
    """Win fraction of ``player`` on ``level_path`` over ``n_episodes`` episodes.

    Args:
        player: Any :class:`~hexfall.players.base.Player` implementation.
        level_path: Path to a Paxie-format level JSON file.
        n_episodes: Number of episodes to run (>= 1).
        seed: Base seed. Episode ``i`` resets the env with ``seed + i``, so the
            fall-direction RNG — the only stochastic element
            (HEXFALL_MDP_SPEC.md §3.4) — varies per episode while staying
            reproducible.

    Returns:
        Float in ``[0.0, 1.0]`` — the fraction of episodes won.

    Determinism: given the same ``(player, level_path, n_episodes, seed)`` the
    return value is bit-exact across runs. See :func:`evaluate_graded` for
    per-episode detail (slices cleared, moves survived) over the same run.
    """
    results = _run_episodes(player, level_path, n_episodes, seed)
    wins = sum(1 for r in results if r["win"])
    return wins / n_episodes


def evaluate_graded(
    player: Player,
    level_path: str | Path,
    n_episodes: int,
    seed: int = 42,
) -> dict[str, Any]:
    """Graded evaluation of ``player`` on ``level_path`` over ``n_episodes``.

    Same deterministic episode loop as :func:`evaluate`, but returns per-episode
    outcomes and per-(player, level) aggregates instead of just the win
    fraction. Used to test whether graded outcomes carry predictive signal
    beyond binary winrate (Issue F).

    Args:
        player: Any :class:`~hexfall.players.base.Player` implementation.
        level_path: Path to a Paxie-format level JSON file.
        n_episodes: Number of episodes to run (>= 1).
        seed: Base seed; episode ``i`` resets with ``seed + i`` (see
            :func:`evaluate`).

    Returns:
        A dict with keys:

        - ``player``: ``type(player).__name__``.
        - ``level``: ``str(level_path)``.
        - ``n_episodes`` / ``seed``: the call parameters.
        - ``winrate``: float in [0, 1] — identical to :func:`evaluate`.
        - ``mean_slices_cleared_fraction``: mean over episodes of
          ``slices_cleared / initial_total_slices`` (each clamped to [0, 1]).
        - ``mean_moves_survived``: mean over episodes of the terminal
          move_counter.
        - ``episodes``: list of per-episode dicts, each with keys ``win``
          (bool), ``slices_cleared_fraction`` (float in [0, 1]) and
          ``moves_survived`` (int). On a win, ``slices_cleared_fraction`` is
          1.0 (the field is empty, HEXFALL_MDP_SPEC.md §7.1).

    Determinism: bit-exact across runs of the same arguments.
    """
    episodes = _run_episodes(player, level_path, n_episodes, seed)
    n = len(episodes)
    wins = sum(1 for e in episodes if e["win"])
    return {
        "player": type(player).__name__,
        "level": str(level_path),
        "n_episodes": n_episodes,
        "seed": seed,
        "winrate": wins / n,
        "mean_slices_cleared_fraction": sum(e["slices_cleared_fraction"] for e in episodes) / n,
        "mean_moves_survived": sum(e["moves_survived"] for e in episodes) / n,
        "episodes": episodes,
    }
