"""Bounded-rationality players for estimating per-level winrates.

Exposes the :class:`Player` protocol, the :class:`GreedyPlayer` (depth-0
heuristic), :class:`LookaheadPlayer` (depth-k forward search) and
:class:`MCTSPlayer` (UCT search), and the :func:`evaluate` helper that runs a
player on a level and returns its winrate.
"""
from hexfall.players.base import Player
from hexfall.players.evaluator import evaluate
from hexfall.players.greedy import GreedyPlayer
from hexfall.players.lookahead import LookaheadPlayer
from hexfall.players.mcts import MCTSPlayer

__all__ = ["Player", "GreedyPlayer", "LookaheadPlayer", "MCTSPlayer", "evaluate"]
