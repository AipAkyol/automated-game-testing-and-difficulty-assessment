"""Quick smoke run: run all three players on a couple of levels, print winrates.

Usage:
    python scripts/run_players.py
"""
import warnings
from pathlib import Path

from hexfall.players import GreedyPlayer, LookaheadPlayer, evaluate

LEVELS = [
    ("tiny_solvable", "levels/tiny_solvable.json"),
    ("level50", "CLASSIFIED.paxie_data/level_data/level50.json"),
]
PLAYERS = [
    ("greedy", GreedyPlayer()),
    ("lookahead-1", LookaheadPlayer(depth=1)),
    ("lookahead-2", LookaheadPlayer(depth=2)),
]
N_EPISODES = 20
SEED = 42


def main() -> None:
    # Level loading emits slice-bucket-parity warnings on hand-built levels;
    # silence them for a clean smoke-run report.
    warnings.simplefilter("ignore")
    for level_name, level_path in LEVELS:
        if not Path(level_path).exists():
            print(f"SKIP {level_name}: file not found at {level_path}")
            continue
        print(f"\n=== {level_name} ===")
        for player_name, player in PLAYERS:
            wr = evaluate(player, level_path, n_episodes=N_EPISODES, seed=SEED)
            print(f"  {player_name:15s}: {wr:.3f} ({int(round(wr * N_EPISODES))}/{N_EPISODES} wins)")


if __name__ == "__main__":
    main()
