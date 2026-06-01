"""Run a *graded* 3-player x 99-level x 20-episode evaluation matrix (Issue F).

Sibling of scripts/run_eval_matrix.py. Identical configuration — same 3 players
(greedy, lookahead-2, mcts), same 99 levels (1-99, level100 excluded for its
unsupported mechanic), same 20 episodes, same SEED base — but calls
``evaluate_graded`` instead of ``evaluate`` and writes the richer schema to a
NEW file, ``outputs/eval_matrix_graded.csv``. The original
``outputs/eval_matrix.csv`` is never read or written here.

Because the seed base matches the original run, the per-(level, player) winrate
in this file is bit-identical to the original eval_matrix.csv (same seeds ->
same fall directions -> same wins). That equality is the proof the re-run is a
consistent replay, not a fresh random draw.

CSV schema (one row per (level_id, player)):
    level_id, player, n_episodes, seed_base, winrate,
    mean_slices_cleared_fraction, mean_moves_survived, wallclock_seconds
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

# --- CONFIG (mirrors run_eval_matrix.py so the two runs are comparable) ---
N_EPISODES = 20
SEED = 0
N_MCTS = 100            # locked from spot-check
N_PROCS = max(1, (os.cpu_count() or 4) - 2)  # leave 2 cores free
PAXIE_DIR = Path("CLASSIFIED.paxie_data/level_data")
OUTPUT_CSV = Path("outputs/eval_matrix_graded.csv")  # NEW file; not the original
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
        evaluate_graded,
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

    nan = float("nan")
    t0 = time.time()
    try:
        graded = evaluate_graded(player, level_path, n_episodes=n_episodes, seed=seed)
        winrate = graded["winrate"]
        mean_frac = graded["mean_slices_cleared_fraction"]
        mean_moves = graded["mean_moves_survived"]
    except UnsupportedMechanicError as e:
        print(f"SKIP (unsupported): {player_name} on {level_id}: {e}", flush=True)
        winrate = mean_frac = mean_moves = nan
    except Exception as e:  # noqa: BLE001 -- never abort the whole matrix
        print(f"ERROR: {player_name} on {level_id}: {e}", flush=True)
        winrate = mean_frac = mean_moves = nan
    elapsed = time.time() - t0

    print(
        f"  {player_name:10s} {level_id:12s} winrate={winrate:.3f} "
        f"frac={mean_frac:.3f} moves={mean_moves:.1f} t={elapsed:.1f}s",
        flush=True,
    )
    return (
        level_id,
        player_name,
        n_episodes,
        seed,
        winrate,
        mean_frac,
        mean_moves,
        round(elapsed, 2),
    )


def main():
    level_paths = get_level_paths()
    print(f"Running GRADED eval matrix: 3 players x {len(level_paths)} levels x {N_EPISODES} episodes")
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
            [
                "level_id",
                "player",
                "n_episodes",
                "seed_base",
                "winrate",
                "mean_slices_cleared_fraction",
                "mean_moves_survived",
                "wallclock_seconds",
            ]
        )
        writer.writerows(results)

    print(f"\nCSV written to {OUTPUT_CSV}")
    print(f"Total wallclock: {t_total / 60:.1f} minutes")

    player_wr = defaultdict(list)
    player_frac = defaultdict(list)
    player_moves = defaultdict(list)
    for level_id, player_name, n_ep, sb, winrate, mean_frac, mean_moves, wc in results:
        if not math.isnan(winrate):
            player_wr[player_name].append(winrate)
            player_frac[player_name].append(mean_frac)
            player_moves[player_name].append(mean_moves)

    print("\nPer-player aggregate (mean across levels):")
    print(f"  {'player':10s} {'winrate':>8s} {'frac_cleared':>13s} {'moves':>8s}  levels")
    for player_name in ["greedy", "lookahead", "mcts"]:
        wrs = player_wr[player_name]
        if wrs:
            print(
                f"  {player_name:10s} "
                f"{sum(wrs) / len(wrs):8.3f} "
                f"{sum(player_frac[player_name]) / len(wrs):13.3f} "
                f"{sum(player_moves[player_name]) / len(wrs):8.1f}  (n={len(wrs)})"
            )
        else:
            print(f"  {player_name:10s}: no data")


if __name__ == "__main__":
    main()
