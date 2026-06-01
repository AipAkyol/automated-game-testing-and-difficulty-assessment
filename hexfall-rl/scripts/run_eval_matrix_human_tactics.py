"""Run a 1-player (human_tactics) x 99-level x 20-episode evaluation matrix.

Sibling of scripts/run_eval_matrix.py (Issue G Phase 2). Identical configuration
— same 99 levels (1-99; level100 excluded for its unsupported mechanic), same 20
episodes, same SEED base, same per-level seed scheme (episode i resets with
seed + i) — so the resulting winrate column is reproducible and joinable to
outputs/eval_matrix.csv by level_id. The single player is HumanTacticsPlayer
(depth-0, deterministic), labelled "human_tactics".

Writes outputs/eval_matrix_human_tactics.csv with a schema IDENTICAL to
outputs/eval_matrix.csv:
    level_id, player, winrate, n_episodes, seed_base, wallclock_seconds

The original outputs/eval_matrix.csv and the 3-player run are never read or
written here.
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

# --- CONFIG (mirrors run_eval_matrix.py so the column is comparable/joinable) ---
PLAYER_NAME = "human_tactics"
N_EPISODES = 20
SEED = 0
N_PROCS = max(1, (os.cpu_count() or 4) - 2)  # leave 2 cores free
PAXIE_DIR = Path("CLASSIFIED.paxie_data/level_data")
OUTPUT_CSV = Path("outputs/eval_matrix_human_tactics.csv")  # NEW file; not the original
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
    player_name, level_path_str, n_episodes, seed = args
    from hexfall.level_loader import UnsupportedMechanicError
    from hexfall.players import HumanTacticsPlayer, evaluate

    if player_name == "human_tactics":
        player = HumanTacticsPlayer()
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
        f"  {player_name:13s} {level_id:12s} winrate={winrate:.3f} t={elapsed:.1f}s",
        flush=True,
    )
    return (level_id, player_name, winrate, n_episodes, seed, round(elapsed, 2))


def main():
    level_paths = get_level_paths()
    print(f"Running eval matrix: 1 player ({PLAYER_NAME}) x {len(level_paths)} levels x {N_EPISODES} episodes")
    print(f"Parallelism: {N_PROCS} processes")

    tasks = []
    for level_path in level_paths:
        tasks.append((PLAYER_NAME, str(level_path), N_EPISODES, SEED))

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

    winrates = [wr for (_, _, wr, _, _, _) in results if not math.isnan(wr)]
    nonzero = [wr for wr in winrates if wr > 0.0]
    print("\nAggregate:")
    if winrates:
        print(f"  mean winrate across levels : {sum(winrates) / len(winrates):.3f}  (n={len(winrates)} levels)")
        print(f"  levels with winrate > 0    : {len(nonzero)} / {len(winrates)}")
        print(f"  min / max winrate          : {min(winrates):.3f} / {max(winrates):.3f}")
    else:
        print("  no data")

    if winrates and all(wr < 0.02 for wr in winrates):
        print("\nWARNING: human_tactics at ~0% winrate on every level. Degenerate "
              "(all-zero) column — investigate before joining to the oracle.")


if __name__ == "__main__":
    main()
