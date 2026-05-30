"""Depth-k lookahead player.

Enumerates legal actions, forks the env to simulate each, takes an expectation
over fall-direction stochasticity by sampling a small fixed set of RNG seeds,
and recurses to the requested depth. Leaf states are scored with the greedy
heuristic; terminal states score 1.0 (win) or 0.0 (lose).

Determinism: the fall samples are the fixed seeds ``range(n_fall_samples)`` and
ties in the argmax are broken toward the lowest action index, so a given
``(state, depth)`` always yields the same action. This keeps
:func:`hexfall.players.evaluator.evaluate` bit-exact reproducible.

Cost: at ``depth=1`` each decision costs ``O(n_legal * n_fall_samples)`` env
steps; at ``depth=2`` roughly ``O(n_legal**2 * n_fall_samples**2)``. This is
fine for small reserves; it is not MCTS and does no parallelism (see Issue B).
"""
from __future__ import annotations

from hexfall.env import HexFallEnv
from hexfall.players.greedy import GreedyPlayer


class LookaheadPlayer:
    """Depth-k lookahead player.

    Args:
        depth: Lookahead depth. ``depth=1`` looks one action ahead; ``depth=2``
            recurses once more. ``depth=0`` degenerates to pure greedy.
        n_fall_samples: Number of RNG seeds sampled to approximate the
            fall-direction expectation at each fork. Default 2.
    """

    def __init__(self, depth: int = 1, n_fall_samples: int = 2):
        self._depth = depth
        self._n_fall_samples = n_fall_samples
        self._greedy = GreedyPlayer()

    def act(self, obs: dict, env: HexFallEnv) -> int:
        mask = obs["action_mask"]
        legal = [a for a in range(mask.shape[0]) if mask[a]]
        if not legal:
            return 0
        if self._depth <= 0:
            return self._greedy.act(obs, env)

        best_action = legal[0]
        best_score: float | None = None
        for a in legal:
            score = self._evaluate_action(a, env, self._depth)
            # Strict ">" keeps the lowest-index action on ties.
            if best_score is None or score > best_score:
                best_score = score
                best_action = a
        return best_action

    def _evaluate_action(self, action: int, env: HexFallEnv, depth: int) -> float:
        """Expected value of taking ``action`` in ``env``, averaged over fall samples."""
        scores: list[float] = []
        for seed in range(self._n_fall_samples):
            fork = env.fork()
            fork.reseed_rng(seed)
            next_obs, reward, terminated, truncated, _info = fork.step(action)
            if terminated or truncated:
                # Terminal value: win = 1.0, lose = 0.0 (reward is +1/-1).
                scores.append(1.0 if reward > 0 else 0.0)
            elif depth <= 1:
                scores.append(self._heuristic_score(next_obs))
            else:
                next_mask = next_obs["action_mask"]
                next_legal = [a for a in range(next_mask.shape[0]) if next_mask[a]]
                if not next_legal:
                    scores.append(0.0)
                else:
                    scores.append(max(
                        self._evaluate_action(a, fork, depth - 1) for a in next_legal
                    ))
        return sum(scores) / len(scores) if scores else 0.0

    def _heuristic_score(self, obs: dict) -> float:
        """Greedy heuristic score of a leaf state (reuses GreedyPlayer internals)."""
        return self._greedy.score_state(obs)
