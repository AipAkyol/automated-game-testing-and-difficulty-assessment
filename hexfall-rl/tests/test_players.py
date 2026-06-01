import warnings
from pathlib import Path

import numpy as np
import pytest

from hexfall.env import HexFallEnv
from hexfall.players import (
    GreedyPlayer,
    LookaheadPlayer,
    Player,
    evaluate,
    evaluate_graded,
)

LEVELS_DIR = Path(__file__).parent.parent / "levels"
TINY = LEVELS_DIR / "tiny_solvable.json"
FORCED_LOSE = LEVELS_DIR / "forced_lose.json"


@pytest.fixture(autouse=True)
def _silence_parity_warnings():
    """Hand-built levels emit slice-bucket-parity UserWarnings on load; ignore them."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        yield


def _legal_actions(obs) -> list[int]:
    mask = obs["action_mask"]
    return [a for a in range(mask.shape[0]) if mask[a]]


def _obs_equal(a: dict, b: dict) -> bool:
    if a.keys() != b.keys():
        return False
    return all(np.array_equal(a[k], b[k]) for k in a)


# ---------------------------------------------------------------------------
# Test 1: Protocol shape
# ---------------------------------------------------------------------------

def test_players_implement_protocol():
    assert isinstance(GreedyPlayer(), Player)
    assert isinstance(LookaheadPlayer(), Player)
    assert isinstance(LookaheadPlayer(depth=2), Player)
    assert callable(GreedyPlayer().act)
    assert callable(LookaheadPlayer().act)


# ---------------------------------------------------------------------------
# Test 2: Evaluator determinism
# ---------------------------------------------------------------------------

def test_evaluator_determinism():
    r1 = evaluate(GreedyPlayer(), TINY, n_episodes=5, seed=99)
    r2 = evaluate(GreedyPlayer(), TINY, n_episodes=5, seed=99)
    assert r1 == r2


# ---------------------------------------------------------------------------
# Test 3: Greedy winrate > 0 on tiny_solvable (always solvable)
# ---------------------------------------------------------------------------

def test_greedy_winrate_positive_on_tiny_solvable():
    wr = evaluate(GreedyPlayer(), TINY, n_episodes=10, seed=42)
    assert wr > 0.0


# ---------------------------------------------------------------------------
# Test 4 / 5: Lookahead depth-1 and depth-2 run without crashing
# ---------------------------------------------------------------------------

def test_lookahead1_runs_on_tiny_solvable():
    wr = evaluate(LookaheadPlayer(depth=1), TINY, n_episodes=5, seed=0)
    assert isinstance(wr, float)
    assert 0.0 <= wr <= 1.0


def test_lookahead2_runs_on_tiny_solvable():
    wr = evaluate(LookaheadPlayer(depth=2), TINY, n_episodes=5, seed=0)
    assert isinstance(wr, float)
    assert 0.0 <= wr <= 1.0


# ---------------------------------------------------------------------------
# Test 6: env.fork() independence
# ---------------------------------------------------------------------------

def test_fork_is_independent_of_original():
    env = HexFallEnv(level_path=str(TINY))
    obs, _ = env.reset(seed=0)
    before = env.get_obs()

    forked = env.fork()
    assert forked._state is not env._state  # deep copy, not a shared reference

    action = _legal_actions(obs)[0]
    forked.step(action)  # must not raise and must not mutate the original

    after = env.get_obs()
    assert _obs_equal(before, after)
    assert env._state.move_counter == 0      # original untouched
    assert forked._state.move_counter == 1   # fork advanced


# ---------------------------------------------------------------------------
# Test 7: fork + reseed produces valid observations (no crash; do not assert
# they differ — the level may not trigger a stochastic fall)
# ---------------------------------------------------------------------------

def test_fork_reseed_produces_valid_obs():
    env = HexFallEnv(level_path=str(TINY))
    obs, _ = env.reset(seed=0)
    action = _legal_actions(obs)[0]

    fork0 = env.fork()
    fork0.reseed_rng(0)
    fork1 = env.fork()
    fork1.reseed_rng(1)

    obs0, *_ = fork0.step(action)
    obs1, *_ = fork1.step(action)

    assert obs0.keys() == obs.keys()
    assert obs1.keys() == obs.keys()
    assert "action_mask" in obs0 and "action_mask" in obs1
    # Stepping the original was never done; both forks remain independent.
    assert env._state.move_counter == 0


# ---------------------------------------------------------------------------
# Test 8 / 9: players always return a legal action
# ---------------------------------------------------------------------------

def test_greedy_returns_legal_action():
    env = HexFallEnv(level_path=str(TINY))
    obs, _ = env.reset(seed=0)
    action = GreedyPlayer().act(obs, env)
    assert isinstance(action, int)
    assert obs["action_mask"][action] == 1


def test_lookahead_returns_legal_action():
    env = HexFallEnv(level_path=str(TINY))
    obs, _ = env.reset(seed=0)
    action = LookaheadPlayer(depth=1).act(obs, env)
    assert isinstance(action, int)
    assert obs["action_mask"][action] == 1


# ---------------------------------------------------------------------------
# Test 10: graded evaluation — per-episode keys and types (Issue F)
# ---------------------------------------------------------------------------

def test_evaluate_graded_episode_keys_and_types():
    result = evaluate_graded(GreedyPlayer(), TINY, n_episodes=5, seed=42)

    # Aggregate shape.
    assert result["n_episodes"] == 5
    assert len(result["episodes"]) == 5
    assert 0.0 <= result["winrate"] <= 1.0
    assert 0.0 <= result["mean_slices_cleared_fraction"] <= 1.0
    assert result["mean_moves_survived"] >= 0.0
    # winrate aggregate matches the scalar evaluate() path exactly.
    assert result["winrate"] == evaluate(GreedyPlayer(), TINY, n_episodes=5, seed=42)

    for ep in result["episodes"]:
        assert set(ep) >= {"win", "slices_cleared_fraction", "moves_survived"}
        assert isinstance(ep["win"], bool)
        assert isinstance(ep["slices_cleared_fraction"], float)
        assert isinstance(ep["moves_survived"], int)


# ---------------------------------------------------------------------------
# Test 11: slices_cleared_fraction stays within [0, 1] (win and lose levels)
# ---------------------------------------------------------------------------

def test_evaluate_graded_fraction_in_unit_interval():
    for level in (TINY, FORCED_LOSE):
        result = evaluate_graded(GreedyPlayer(), level, n_episodes=5, seed=42)
        for ep in result["episodes"]:
            assert 0.0 <= ep["slices_cleared_fraction"] <= 1.0


# ---------------------------------------------------------------------------
# Test 12: a fully-solved (won) episode clears every slice (fraction == 1.0)
# ---------------------------------------------------------------------------

def test_evaluate_graded_win_clears_all_slices():
    # tiny_solvable is always solvable; greedy wins every episode here.
    result = evaluate_graded(GreedyPlayer(), TINY, n_episodes=10, seed=42)
    won = [ep for ep in result["episodes"] if ep["win"]]
    assert won, "expected at least one won episode on tiny_solvable"
    for ep in won:
        # Win <=> empty field (HEXFALL_MDP_SPEC.md §7.1) => fraction == 1.0.
        assert ep["slices_cleared_fraction"] == pytest.approx(1.0)
        assert ep["moves_survived"] > 0


# ---------------------------------------------------------------------------
# Test 13: graded evaluation is deterministic under a fixed seed
# ---------------------------------------------------------------------------

def test_evaluate_graded_determinism():
    r1 = evaluate_graded(GreedyPlayer(), TINY, n_episodes=5, seed=99)
    r2 = evaluate_graded(GreedyPlayer(), TINY, n_episodes=5, seed=99)
    assert r1 == r2
