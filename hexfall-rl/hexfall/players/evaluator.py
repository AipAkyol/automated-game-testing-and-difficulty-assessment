"""Run a player on a level for N episodes and return its win fraction."""
from __future__ import annotations

from pathlib import Path

from hexfall.env import HexFallEnv
from hexfall.players.base import Player


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
    return value is bit-exact across runs. A single env is reused across
    episodes; ``reset(seed=...)`` rebuilds the full game state from the level
    file plus the seed, so RNG state never bleeds between episodes.
    """
    if n_episodes < 1:
        raise ValueError(f"n_episodes must be >= 1, got {n_episodes}")

    env = HexFallEnv(level_path=str(level_path))
    wins = 0
    for i in range(n_episodes):
        obs, _info = env.reset(seed=seed + i)
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
        if terminated and reward > 0:
            wins += 1
    return wins / n_episodes
