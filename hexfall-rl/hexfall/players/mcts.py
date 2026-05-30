"""Monte-Carlo Tree Search (UCT) player.

Builds a UCT search tree rooted at the current quiescent state, running
``n_rollouts`` simulations per decision. Each simulation walks the tree by the
UCT rule, expands one untried action, then plays a *rollout policy* (greedy by
default) to a terminal state; the win/lose outcome is backpropagated up the
visited path. The returned action is the root child with the most visits.

Interface match
---------------
This player follows the :class:`~hexfall.players.base.Player` protocol exactly:
``act(obs, env)`` receives the live env, so MCTS forks *that* env for its tree
and rollouts — there is no separate ``register_env`` step. Forking is a deep
copy (:meth:`HexFallEnv.fork`); the live env passed by
:func:`~hexfall.players.evaluator.evaluate` is never mutated.

Determinism
-----------
``act`` is a pure function of the env state: the root is ``env.fork()`` (a deep
copy that reproduces the original RNG), each rollout reseeds its own fork with a
fixed seed sequence (``i * 7919 + 1``), tree expansion order is fixed (untried
actions popped from the end of an index-ordered list), and argmax ties break to
the lowest key. So two ``act`` calls on the same state return the same action,
keeping :func:`evaluate` bit-exact reproducible.

The only stochastic element of the simulator is the hex fall direction
(HEXFALL_MDP_SPEC.md §3.4); reseeding each rollout fork samples distinct fall
outcomes while staying reproducible across runs.
"""
from __future__ import annotations

import math

from hexfall.env import HexFallEnv
from hexfall.players.base import Player


def _legal_actions(obs: dict) -> list[int]:
    mask = obs["action_mask"]
    return [a for a in range(mask.shape[0]) if mask[a]]


class MCTSNode:
    """A node in the UCT tree, owning a forked env at this node's state."""

    def __init__(
        self,
        env: HexFallEnv,
        parent: "MCTSNode | None" = None,
        action: int | None = None,
    ):
        self.env = env                    # HexFallEnv deep-copied to this state
        self.parent = parent
        self.action = action              # action that led here (None at root)
        self.children: dict[int, MCTSNode] = {}
        self.visits = 0
        self.wins = 0.0
        # A terminal node carries its outcome directly (set by the expander)
        # so repeated selection credits the true win/lose value rather than
        # re-rolling out from a finished episode.
        self.terminal = False
        self.terminal_value = 0.0
        # Legal actions not yet expanded. Popped from the end during expansion.
        self.untried_actions = _legal_actions(env.get_obs())

    def uct_score(self, exploration: float) -> float:
        if self.visits == 0:
            return float("inf")
        return (self.wins / self.visits) + exploration * math.sqrt(
            math.log(self.parent.visits) / self.visits
        )

    def is_fully_expanded(self) -> bool:
        return len(self.untried_actions) == 0


class MCTSPlayer:
    """UCT player. Obeys the ``Player`` protocol: ``act(obs, env)``.

    Args:
        n_rollouts: Number of UCT simulations run per decision.
        exploration: UCT exploration constant (``c`` in ``sqrt(2)`` by default).
        rollout_policy: A ``Player`` used to play out simulations. Defaults to a
            fresh :class:`~hexfall.players.greedy.GreedyPlayer` (lazily imported
            to avoid an import cycle).
        max_rollout_steps: Safety cap on rollout length, so a non-terminating
            policy can't loop forever. A capped rollout scores as a loss.
    """

    def __init__(
        self,
        n_rollouts: int = 100,
        exploration: float = math.sqrt(2),
        rollout_policy: Player | None = None,
        max_rollout_steps: int = 200,
    ):
        self.n_rollouts = n_rollouts
        self.exploration = exploration
        self._rollout_policy = rollout_policy
        self._max_rollout_steps = max_rollout_steps

    def act(self, obs: dict, env: HexFallEnv) -> int:
        """Run UCT from the current state; return the most-visited root action."""
        rollout_policy = self._rollout_policy
        if rollout_policy is None:
            from hexfall.players.greedy import GreedyPlayer  # lazy: avoid cycle

            rollout_policy = GreedyPlayer()

        root = MCTSNode(env.fork())

        for i in range(self.n_rollouts):
            # 1. Selection -------------------------------------------------
            node = self._select(root)

            # A selected terminal leaf: credit its known outcome, no rollout.
            if node.terminal:
                self._backprop(node, node.terminal_value)
                continue

            # 2. Expansion -------------------------------------------------
            if node.untried_actions:
                action = node.untried_actions.pop()
                child_env = node.env.fork()
                _obs, reward, terminated, truncated, _info = child_env.step(action)
                child = MCTSNode(child_env, parent=node, action=action)
                node.children[action] = child
                node = child
                if terminated or truncated:
                    node.terminal = True
                    node.terminal_value = 1.0 if reward > 0 else 0.0
                    self._backprop(node, node.terminal_value)
                    continue

            # 3. Rollout ---------------------------------------------------
            result = self._rollout(node, rollout_policy, seed=i * 7919 + 1)

            # 4. Backpropagation ------------------------------------------
            self._backprop(node, result)

        if not root.children:
            # No expansion happened (e.g. root already terminal): fall back to
            # any legal action, mirroring the GreedyPlayer contract.
            legal = _legal_actions(obs)
            return legal[0] if legal else 0

        # Most-visited root action; ties break to the lowest action index.
        return max(
            root.children.items(),
            key=lambda kv: (kv[1].visits, -kv[0]),
        )[0]

    def _select(self, node: MCTSNode) -> MCTSNode:
        """Descend by UCT until reaching a node with untried actions or a leaf."""
        while node.is_fully_expanded() and node.children:
            node = max(
                node.children.values(),
                key=lambda n: n.uct_score(self.exploration),
            )
        return node

    def _rollout(self, node: MCTSNode, rollout_policy: Player, seed: int) -> float:
        """Play ``rollout_policy`` from ``node`` to a terminal; 1.0 win else 0.0."""
        sim_env = node.env.fork()
        sim_env.reseed_rng(seed)
        obs = sim_env.get_obs()

        done = False
        reward = 0.0
        steps = 0
        while not done and steps < self._max_rollout_steps:
            if not obs["action_mask"].any():
                # Quiescent state with no legal action: a dead end, scored as
                # a loss (matches evaluator's defensive break semantics).
                break
            action = rollout_policy.act(obs, sim_env)
            obs, reward, terminated, truncated, _info = sim_env.step(action)
            done = terminated or truncated
            steps += 1

        return 1.0 if (done and reward > 0) else 0.0

    def _backprop(self, node: MCTSNode | None, result: float) -> None:
        while node is not None:
            node.visits += 1
            node.wins += result
            node = node.parent
