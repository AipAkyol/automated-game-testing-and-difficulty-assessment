"""Run a 3-player x 99-level x 20-episode evaluation matrix.

Outputs outputs/eval_matrix.csv (297 data rows + header). N_MCTS is hardcoded
from the spot-check (scripts/mcts_spotcheck.py): N=100 timed at ~1-10s per
episode on level50/level10, well under the 60s/episode budget.

Adapted to the repo API: evaluate(player, level_path, n_episodes, seed=...).
"""
import csv
import math
import multiprocessing
import os
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# --- CONFIG ---
N_EPISODES = 20
SEED = 0
N_MCTS = 100            # locked from spot-check
N_PROCS = max(1, (os.cpu_count() or 4) - 2)  # leave 2 cores free
PAXIE_DIR = Path("CLASSIFIED.paxie_data/level_data")
OUTPUT_CSV = Path("outputs/eval_matrix.csv")
# --------------


def get_level_paths():
    """Return sorted list of level paths for levels 1-99 (excludes level100)."""
    paths = sorted(PAXIE_DIR.glob("level*.json"))
    paths = [p for p in paths if p.stem != "level100"]  # unsupported mechanic
    assert len(paths) == 99, f"Expected 99 levels, got {len(paths)}"
    return paths


def evaluate_task(args):
    """Worker function -- runs in a fresh spawned process."""
    warnings.simplefilter("ignore")  # quiet the hand-built-level parity warnings
    player_name, level_path_str, n_episodes, seed, n_mcts = args
    from hexfall.level_loader import UnsupportedMechanicError
    from hexfall.players import (
        GreedyPlayer,
        LookaheadPlayer,
        MCTSPlayer,
        evaluate,
    )

    if player_name == "greedy":
        player = GreedyPlayer()
    elif player_name == "lookahead":
        player = LookaheadPlayer(depth=2)
    elif player_name == "mcts":
        player = MCTSPlayer(n_rollouts=n_mcts)
    else:
        raise ValueError(f"Unknown player: {player_name}")

    level_path = str(level_path_str)
    level_id = Path(level_path).stem

    t0 = time.time()
    try:
        winrate = evaluate(player, level_path, n_episodes=n_episodes, seed=seed)
    except UnsupportedMechanicError as e:
        print(f"SKIP (unsupported): {player_name} on {level_id}: {e}", flush=True)
        winrate = float("nan")
    except Exception as e:  # noqa: BLE001 -- never abort the whole matrix
        print(f"ERROR: {player_name} on {level_id}: {e}", flush=True)
        winrate = float("nan")
    elapsed = time.time() - t0

    print(
        f"  {player_name:10s} {level_id:12s} winrate={winrate:.3f} t={elapsed:.1f}s",
        flush=True,
    )
    return (level_id, player_name, winrate, n_episodes, seed, round(elapsed, 2))


def main():
    level_paths = get_level_paths()
    print(f"Running eval matrix: 3 players x {len(level_paths)} levels x {N_EPISODES} episodes")
    print(f"Parallelism: {N_PROCS} processes | N_MCTS={N_MCTS}")

    tasks = []
    for player_name in ["greedy", "lookahead", "mcts"]:
        for level_path in level_paths:
            tasks.append((player_name, str(level_path), N_EPISODES, SEED, N_MCTS))

    print(f"Total tasks: {len(tasks)}")

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    ctx = multiprocessing.get_context("spawn")
    t_start = time.time()
    with ctx.Pool(processes=N_PROCS) as pool:
        results = pool.map(evaluate_task, tasks)
    t_total = time.time() - t_start

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["level_id", "player", "winrate", "n_episodes", "seed_base", "wallclock_seconds"]
        )
        writer.writerows(results)

    print(f"\nCSV written to {OUTPUT_CSV}")
    print(f"Total wallclock: {t_total / 60:.1f} minutes")

    player_winrates = defaultdict(list)
    for level_id, player_name, winrate, n_ep, sb, wc in results:
        if not math.isnan(winrate):
            player_winrates[player_name].append(winrate)

    print("\nPer-player aggregate winrate (mean across levels):")
    for player_name in ["greedy", "lookahead", "mcts"]:
        wrs = player_winrates[player_name]
        if wrs:
            print(f"  {player_name:10s}: {sum(wrs) / len(wrs):.3f}  (n={len(wrs)} levels)")
        else:
            print(f"  {player_name:10s}: no data")

    all_means = [sum(v) / len(v) for v in player_winrates.values() if v]
    if all_means and all(m < 0.02 for m in all_means):
        print("\nWARNING: All players at ~0% aggregate winrate. May indicate a "
              "simulator or evaluate() bug. Escalate before proceeding to Issue C.")
    if all_means and all(m > 0.98 for m in all_means):
        print("\nWARNING: All players at ~100% aggregate winrate. May indicate all "
              "levels are trivially easy. Escalate before proceeding to Issue C.")


if __name__ == "__main__":
    main()
