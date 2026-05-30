"""Tests for the MCTS (UCT) player.

Adapted to the repo's real API: players implement ``act(obs, env)`` (no
``register_env``), and ``evaluate`` takes ``seed=`` (not ``seed_base=``).
"""
import warnings
from pathlib import Path

import pytest

from hexfall.env import HexFallEnv
from hexfall.level_loader import UnsupportedMechanicError
from hexfall.players import GreedyPlayer, MCTSPlayer, Player, evaluate

LEVELS_DIR = Path(__file__).parent.parent / "levels"
TINY = LEVELS_DIR / "tiny_solvable.json"
PAXIE_DIR = Path(__file__).parent.parent / "CLASSIFIED.paxie_data" / "level_data"


@pytest.fixture(autouse=True)
def _silence_parity_warnings():
    """Hand-built levels emit slice-bucket-parity UserWarnings on load; ignore them."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        yield


def _legal_actions(obs) -> list[int]:
    mask = obs["action_mask"]
    return [a for a in range(mask.shape[0]) if mask[a]]


# ---------------------------------------------------------------------------
# Test 1: Protocol compliance
# ---------------------------------------------------------------------------

def test_mcts_protocol_compliance():
    player = MCTSPlayer()
    assert isinstance(player, Player)
    assert callable(player.act)

    env = HexFallEnv(level_path=str(TINY))
    obs, _ = env.reset(seed=0)
    action = MCTSPlayer(n_rollouts=5).act(obs, env)
    assert isinstance(action, int)


# ---------------------------------------------------------------------------
# Test 2: returns a legal action
# ---------------------------------------------------------------------------

def test_mcts_returns_legal_action():
    env = HexFallEnv(level_path=str(TINY))
    obs, _ = env.reset(seed=0)
    action = MCTSPlayer(n_rollouts=5).act(obs, env)
    assert isinstance(action, int)
    assert obs["action_mask"][action] == 1


# ---------------------------------------------------------------------------
# Test 3: MCTS winrate >= greedy on tiny_solvable
# ---------------------------------------------------------------------------

def test_mcts_winrate_ge_greedy_on_tiny_solvable():
    greedy_wr = evaluate(GreedyPlayer(), TINY, n_episodes=10, seed=42)
    mcts_wr = evaluate(MCTSPlayer(n_rollouts=20), TINY, n_episodes=10, seed=42)
    assert mcts_wr >= greedy_wr


# ---------------------------------------------------------------------------
# Test 4: no crash on a sample of Paxie levels (slow / data-dependent)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not PAXIE_DIR.exists(), reason="Paxie level data not present")
def test_mcts_no_crash_on_paxie_sample():
    paths = sorted(PAXIE_DIR.glob("level*.json"))[:5]
    assert paths, "expected at least one Paxie level"
    ran = 0
    for level_path in paths:
        try:
            wr = evaluate(MCTSPlayer(n_rollouts=10), level_path, n_episodes=1, seed=0)
        except UnsupportedMechanicError:
            # Some levels (e.g. level100) use mechanics the simulator rejects at
            # load; the full eval matrix excludes them. Skipping here is correct.
            continue
        assert isinstance(wr, float)
        assert 0.0 <= wr <= 1.0
        ran += 1
    assert ran > 0, "no supported levels in the sample"


# ---------------------------------------------------------------------------
# Test 5: deterministic (same env state -> same action)
# ---------------------------------------------------------------------------

def test_mcts_deterministic():
    env = HexFallEnv(level_path=str(TINY))
    obs, _ = env.reset(seed=0)
    player = MCTSPlayer(n_rollouts=20)
    a1 = player.act(obs, env)
    a2 = player.act(obs, env)
    assert a1 == a2
    # And the live env was not mutated by act().
    assert env._state.move_counter == 0
