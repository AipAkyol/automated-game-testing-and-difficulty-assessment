"""Spot-check: time a single MCTS episode on level50.json to set N for full matrix."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from hexfall.players import MCTSPlayer, evaluate  # noqa: E402

LEVEL = "CLASSIFIED.paxie_data/level_data/level50.json"


def spotcheck(n):
    player = MCTSPlayer(n_rollouts=n)
    t0 = time.time()
    wr = evaluate(player, LEVEL, n_episodes=1, seed=0)
    elapsed = time.time() - t0
    print(f"N={n}: 1 episode on level50 took {elapsed:.1f}s | winrate={wr:.2f}")
    return elapsed


if __name__ == "__main__":
    for n in [100, 50, 25]:
        elapsed = spotcheck(n)
        if elapsed <= 60:
            print(f"  -> N={n} is safe for full matrix. Locking N={n}.")
            break
    else:
        print("WARNING: even N=25 exceeds 60s per episode. Escalate before running full matrix.")
