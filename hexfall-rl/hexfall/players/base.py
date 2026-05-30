"""The :class:`Player` protocol — the common interface every Hex Fall player implements."""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from hexfall.env import HexFallEnv


@runtime_checkable
class Player(Protocol):
    """A policy that selects a legal action from an observation.

    Players are bounded-rationality solvers used to estimate per-level winrates
    (to be fit against human winrate data). They are deterministic heuristics
    evaluated by :func:`hexfall.players.evaluator.evaluate`, not RL agents.
    """

    def act(self, obs: dict, env: HexFallEnv) -> int:
        """Select a legal action for the current quiescent state.

        Args:
            obs: The observation dict returned by ``env.reset()`` or
                ``env.step()``. Guaranteed to be at a quiescent state with at
                least one legal action (``obs["action_mask"]`` has >= 1 set bit).
            env: The live env. Players that do lookahead may call ``env.fork()``
                to branch and simulate; obs-only players (``GreedyPlayer``)
                ignore it.

        Returns:
            An integer action index ``a`` (into the flattened reserve grid) that
            is legal in the current state, i.e. ``obs["action_mask"][a] == 1``.
        """
        ...
