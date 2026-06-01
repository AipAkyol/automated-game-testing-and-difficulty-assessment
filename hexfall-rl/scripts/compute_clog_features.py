"""Compute Clog features (Issue H Phase 2) across all 99 Paxie levels.

Writes ``CLASSIFIED.paxie_data/oracle/clog_features.csv`` (gitignored) with
columns ``level_id, fcc, ctd, rhc, total_slice_count`` and prints:

  1. Orthogonality gate: |Pearson r| of each clog feature vs total_slice_count;
     hard threshold |r| < 0.7. Any feature with |r| >= 0.7 is flagged loudly and
     the script stops (exit 1) — that feature is slice count rebranded.
  2. Correlation matrix: clog features pairwise + each vs total_slice_count and
     the existing 5 structural features (reused from
     ``oracle.extract_structural_features``).
  3. Distribution summary (min / Q1 / median / Q3 / max) per clog feature,
     flagging any feature with near-zero spread.

Does NOT refit or modify the oracle (that is Phase 3). Reads contents only,
never editorMeta.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from hexfall.clog import extract_clog_features
from hexfall.level_loader import load_level
from hexfall.oracle import STRUCTURAL_FEATURE_NAMES, extract_structural_features

REPO = Path(__file__).resolve().parents[1]
LEVEL_DIR = REPO / "CLASSIFIED.paxie_data" / "level_data"
OUT_CSV = REPO / "CLASSIFIED.paxie_data" / "oracle" / "clog_features.csv"

CLOG_NAMES = ["fcc", "ctd", "rhc"]
GATE_THRESHOLD = 0.7
SEED = 0  # load() consumes no RNG at level load, but pin a seed for determinism.

_ABBR = {
    "color_count": "colorCnt", "total_slice_count": "totSlice",
    "pin_count": "pinCnt", "ice_count": "iceCnt", "reserve_area": "resArea",
}


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson r, or nan if either input is constant (correlation undefined)."""
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def main() -> None:
    rows = []
    skipped = []
    with warnings.catch_warnings():
        # Loader colorCount/parity/off-grid-pin warnings are validated in Phase 1
        # and not the concern here; silence them to keep the report readable.
        warnings.simplefilter("ignore")
        for n in range(1, 100):
            path = LEVEL_DIR / f"level{n}.json"
            if not path.exists():
                skipped.append((n, "missing file"))
                continue
            try:
                state = load_level(path, seed=SEED)
            except Exception as e:  # report and skip rather than abort the corpus
                skipped.append((n, f"{type(e).__name__}: {e}"))
                continue
            clog = extract_clog_features(state)
            struct = dict(zip(STRUCTURAL_FEATURE_NAMES,
                              extract_structural_features(state)))
            rows.append({"level_id": state.level_id, **clog, **struct})

    df = pd.DataFrame(rows)
    df["total_slice_count"] = df["total_slice_count"].astype(int)
    print(f"Loaded {len(df)}/99 levels.", "Skipped:" if skipped else "(none skipped)",
          skipped if skipped else "")

    # --- Write CSV (only the four required feature columns + level_id) -------
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    csv_cols = ["level_id", "fcc", "ctd", "rhc", "total_slice_count"]
    df[csv_cols].to_csv(OUT_CSV, index=False)
    print(f"Wrote {OUT_CSV}")
    print(f"  ({len(df)} rows; columns={csv_cols})\n")

    tsc = df["total_slice_count"].to_numpy(dtype=float)

    # --- 1. Orthogonality gate ----------------------------------------------
    print("=" * 70)
    print(f"1. ORTHOGONALITY GATE  -  |Pearson r| vs total_slice_count  "
          f"(need |r| < {GATE_THRESHOLD})")
    print("=" * 70)
    breaches = []
    for name in CLOG_NAMES:
        r = pearson(df[name].to_numpy(dtype=float), tsc)
        is_nan = np.isnan(r)
        breach = (not is_nan) and abs(r) >= GATE_THRESHOLD
        if breach:
            breaches.append((name, r))
        rtxt = "nan(const)" if is_nan else f"{r:+.4f}"
        artxt = "nan" if is_nan else f"{abs(r):.4f}"
        print(f"  {name:5s}  r = {rtxt:>11s}   |r| = {artxt:>7s}   "
              f"-> {'*** BREACH ***' if breach else 'OK'}")
    print()

    # --- 2. Correlation matrix ----------------------------------------------
    matrix_cols = CLOG_NAMES + STRUCTURAL_FEATURE_NAMES
    series = {c: df[c].to_numpy(dtype=float) for c in matrix_cols}
    print("=" * 70)
    print("2. CORRELATION MATRIX (Pearson r) - rows: clog; cols: clog + 5 structural")
    print("=" * 70)
    print("       " + "".join(f"{_ABBR.get(c, c):>9s}" for c in matrix_cols))
    for rn in CLOG_NAMES:
        line = f"{rn:>6s} "
        for cn in matrix_cols:
            r = pearson(series[rn], series[cn])
            line += f"{('nan' if np.isnan(r) else f'{r:+.3f}'):>9s}"
        print(line)
    print()

    # --- 3. Distribution summary --------------------------------------------
    print("=" * 70)
    print("3. DISTRIBUTION SUMMARY per clog feature (min / Q1 / median / Q3 / max)")
    print("=" * 70)
    for name in CLOG_NAMES:
        v = df[name].to_numpy(dtype=float)
        mn, q1, med, q3, mx = (
            np.min(v), np.percentile(v, 25), np.median(v),
            np.percentile(v, 75), np.max(v),
        )
        std = float(np.std(v))
        spread = mx - mn
        if spread < 1e-9:
            flag = "   <<< CONSTANT (zero spread)"
        elif std < 0.02:
            flag = f"   <<< near-zero spread (std={std:.4f})"
        else:
            flag = ""
        print(f"  {name:5s} min={mn:.4f}  Q1={q1:.4f}  med={med:.4f}  "
              f"Q3={q3:.4f}  max={mx:.4f}  (std={std:.4f}){flag}")
    print()

    # --- Verdict -------------------------------------------------------------
    print("=" * 70)
    if breaches:
        print("XXX  ORTHOGONALITY GATE FAILED  XXX")
        for name, r in breaches:
            print(f"   {name}: |r| = {abs(r):.4f} >= {GATE_THRESHOLD} vs "
                  f"total_slice_count -> SLICE COUNT REBRANDED; PRUNE before Phase 3.")
        print("STOPPING (exit 1). Do NOT carry the breaching feature(s) into the Phase 3 refit.")
        print("=" * 70)
        sys.exit(1)
    print(f"OK  ORTHOGONALITY GATE PASSED — all clog features |r| < {GATE_THRESHOLD} "
          "vs total_slice_count.")
    print("=" * 70)


if __name__ == "__main__":
    main()
