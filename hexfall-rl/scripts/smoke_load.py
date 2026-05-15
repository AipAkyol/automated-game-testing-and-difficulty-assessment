"""Smoke-load every Paxie level under a given directory.

Usage:
    python scripts/smoke_load.py <levels_dir>

Where <levels_dir> contains files named `level{N}.json` for N in 1..100
(matching the CLASSIFIED.paxie_data/level_data/ naming convention).

Output:
    Per-level line: [OK]/[WARN]/[UNSUPPORTED]/[ERROR] level_NN.json: <message>
    Aggregate counts at the end.
    For pins with blockCount > 0, print full ray geometry.
"""

import argparse
import warnings
from pathlib import Path

from hexfall.game import _pin_ray_cells
from hexfall.level_loader import (
    LevelLoadError,
    UnsupportedMechanicError,
    load_level,
)


def _scan_level(path: Path) -> tuple[str, str, list[str]]:
    """Return (status, message, pin_ray_lines).

    status ∈ {"OK", "WARN", "UNSUPPORTED", "ERROR"}.
    """
    pin_lines: list[str] = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            state = load_level(path)
        except UnsupportedMechanicError as e:
            return ("UNSUPPORTED", str(e), [])
        except (LevelLoadError, ValueError) as e:
            return ("ERROR", f"{type(e).__name__}: {e}", [])
        except Exception as e:
            return ("ERROR", f"{type(e).__name__}: {e}", [])

    # Print geometry for every pin in the level. blockCount=0 means
    # extend-to-edge — equally interesting for sanity-checking ray geometry.
    for pin in state.pins:
        ray = _pin_ray_cells(pin, state.reserve_rows, state.reserve_cols)
        pin_lines.append(
            f"  Pin origin=(row={pin.origin_row}, col={pin.origin_col}) "
            f"dir={pin.direction} blockCount={pin.block_count} ray={ray}"
        )

    warning_msgs = [str(w.message) for w in caught]
    if warning_msgs:
        # Surface the first warning as the line summary; the rest are appended.
        msg = warning_msgs[0]
        if len(warning_msgs) > 1:
            msg = f"{msg} (+{len(warning_msgs) - 1} more warnings)"
        return ("WARN", msg, pin_lines)

    return ("OK", "", pin_lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("levels_dir", type=Path,
                        help="Directory containing levelN.json files (N in 1..100).")
    parser.add_argument("--start", type=int, default=1, help="First level number to scan (default 1).")
    parser.add_argument("--end", type=int, default=100, help="Last level number to scan (default 100).")
    parser.add_argument("--quiet-warnings", action="store_true",
                        help="Suppress the warning-detail line for WARN-status levels.")
    args = parser.parse_args(argv)

    counts = {"OK": 0, "WARN": 0, "UNSUPPORTED": 0, "ERROR": 0, "MISSING": 0}

    for n in range(args.start, args.end + 1):
        path = args.levels_dir / f"level{n}.json"
        if not path.exists():
            counts["MISSING"] += 1
            print(f"[MISSING] level{n}.json")
            continue
        status, message, pin_lines = _scan_level(path)
        counts[status] = counts.get(status, 0) + 1
        suffix = f": {message}" if (message and not args.quiet_warnings) else ""
        print(f"[{status}] level{n}.json{suffix}")
        for line in pin_lines:
            print(line)

    print()
    print("=== Aggregate ===")
    for k in ("OK", "WARN", "UNSUPPORTED", "ERROR", "MISSING"):
        if counts.get(k, 0):
            print(f"  {k}: {counts[k]}")

    return 0 if counts["ERROR"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
