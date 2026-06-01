"""Issue F: refit the difficulty oracle with GRADED solver features.

Separate script (NOT a flag on fit_oracle.py) so Issue C's gate run, its
coefficients, and its scatter PNG are never touched. This script writes NO
artifacts -- it only prints. It does not read or write outputs/eval_matrix.csv
or CLASSIFIED.paxie_data/oracle/.

Graded features (6 = 2 metrics x 3 players), pivoted per level from
outputs/eval_matrix_graded.csv:
    {greedy,lookahead,mcts}_frac_cleared   (mean_slices_cleared_fraction)
    {greedy,lookahead,mcts}_moves_survived (mean_moves_survived)

Three ablation configs, each LOO-CV (leakage-free, Ridge(StandardScaler),
alpha=1.0 -- identical protocol to fit_oracle.py):
    (a) graded-only                          : 6 graded
    (b) graded + structural                  : 6 graded + 5 structural
    (c) graded + binary winrate + structural : 6 graded + 3 winrate + 5 structural

Binary winrates are read from the graded CSV's own ``winrate`` column, which was
verified bit-identical to outputs/eval_matrix.csv last session (same seeds).

Baseline to beat (Issue C, structural-only LOO-CV): Spearman 0.6422 / Pearson
0.6210. Reported in the table with each config's lift. Conclusions are out of
scope -- numbers only.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from hexfall.level_loader import load_level
from hexfall.oracle import (
    ALPHA,
    STRUCTURAL_FEATURE_NAMES,
    extract_structural_features,
)

REPO = Path(__file__).resolve().parents[1]
GRADED_MATRIX = REPO / "outputs" / "eval_matrix_graded.csv"
ANIL_CSV = REPO / "CLASSIFIED.paxie_data" / "user_data_hexa_fall_filtered.csv"
LEVEL_DIR = REPO / "CLASSIFIED.paxie_data" / "level_data"

TARGET_COL = "% Win Rate"  # pinned literal; NOT the adjacent "% Win User Rate"

# Issue C structural-only LOO-CV baseline (the number each config must beat).
BASELINE_SPEARMAN = 0.6422
BASELINE_PEARSON = 0.6210

PLAYERS = ["greedy", "lookahead", "mcts"]
GRADED_FRAC_NAMES = [f"{p}_frac_cleared" for p in PLAYERS]
GRADED_MOVES_NAMES = [f"{p}_moves_survived" for p in PLAYERS]
GRADED_FEATURE_NAMES = GRADED_FRAC_NAMES + GRADED_MOVES_NAMES  # 6
WINRATE_NAMES = [f"{p}_winrate" for p in PLAYERS]              # 3


def load_graded_features() -> pd.DataFrame:
    """Pivot the graded matrix long->wide: one row per level, graded + winrate cols."""
    df = pd.read_csv(GRADED_MATRIX)
    wide = df.pivot(
        index="level_id",
        columns="player",
        values=["mean_slices_cleared_fraction", "mean_moves_survived", "winrate"],
    )
    # Flatten the MultiIndex columns to the canonical feature names.
    rename = {}
    for p in PLAYERS:
        rename[("mean_slices_cleared_fraction", p)] = f"{p}_frac_cleared"
        rename[("mean_moves_survived", p)] = f"{p}_moves_survived"
        rename[("winrate", p)] = f"{p}_winrate"
    wide.columns = [rename[c] for c in wide.columns]
    wide = wide.reset_index()
    # Normalize join key: "level1" -> 1 (matches fit_oracle.py).
    wide["level_num"] = wide["level_id"].str.replace("level", "", regex=False).astype(int)
    return wide


def load_human_winrates() -> pd.DataFrame:
    """Load Anil's CSV (comma-separated). Same logic as fit_oracle.py."""
    df = pd.read_csv(ANIL_CSV, sep=",")
    assert df["Level"].is_unique, "Anil 'Level' column has duplicates; join would fan out"
    assert df[TARGET_COL].max() <= 1.0, f"{TARGET_COL} looks like a percent, not a fraction"
    return df[["Level", TARGET_COL]]


def make_pipeline(alpha: float) -> Pipeline:
    return Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=alpha))])


def loo_corr(X_sub: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Leakage-free LOO-CV out-of-fold Spearman + Pearson (alpha=1.0)."""
    oof = cross_val_predict(make_pipeline(ALPHA), X_sub, y, cv=LeaveOneOut())
    return spearmanr(y, oof)[0], pearsonr(y, oof)[0]


def main() -> None:
    graded = load_graded_features()
    human = load_human_winrates()

    joined = graded.merge(human, left_on="level_num", right_on="Level", how="inner")
    print(f"Join size: {len(joined)} rows "
          f"(graded levels={graded['level_num'].nunique()}, "
          f"human levels={human['Level'].nunique()})")
    if len(joined) < 90:
        raise SystemExit(f"STOP: join size {len(joined)} < 90 -> key mismatch, not a result.")

    # --- Assemble a named feature frame (one source of truth, slice by name) ---
    struct_rows = []
    for _, r in joined.iterrows():
        state = load_level(LEVEL_DIR / f"level{int(r['level_num'])}.json")
        struct_rows.append(extract_structural_features(state))
    struct_df = pd.DataFrame(struct_rows, columns=STRUCTURAL_FEATURE_NAMES, index=joined.index)

    feat = pd.concat(
        [joined[GRADED_FEATURE_NAMES + WINRATE_NAMES].reset_index(drop=True),
         struct_df.reset_index(drop=True)],
        axis=1,
    )
    assert not feat.isna().any().any(), "NaN in feature matrix; investigate before fitting"
    y = joined[TARGET_COL].to_numpy(dtype=float)

    def X(cols: list[str]) -> np.ndarray:
        return feat[cols].to_numpy(dtype=float)

    # --- Three ablation configs ---------------------------------------------
    cfg = {
        "(a) graded-only": GRADED_FEATURE_NAMES,
        "(b) graded + structural": GRADED_FEATURE_NAMES + STRUCTURAL_FEATURE_NAMES,
        "(c) graded + winrate + structural":
            GRADED_FEATURE_NAMES + WINRATE_NAMES + STRUCTURAL_FEATURE_NAMES,
    }
    results = {name: loo_corr(X(cols), y) for name, cols in cfg.items()}

    # --- Reproducibility check: recompute the structural-only baseline -------
    repro_sp, repro_pe = loo_corr(X(STRUCTURAL_FEATURE_NAMES), y)

    # --- Redundancy check: frac_cleared vs total_slice_count -----------------
    tsc = feat["total_slice_count"].to_numpy(dtype=float)
    redundancy = {
        p: pearsonr(feat[f"{p}_frac_cleared"].to_numpy(dtype=float), tsc)[0]
        for p in PLAYERS
    }

    # --- Results table -------------------------------------------------------
    print("\n" + "=" * 78)
    print("RESULTS  (LOO-CV, Ridge(StandardScaler), alpha=1.0; N = "
          f"{len(joined)} levels)")
    print("=" * 78)
    hdr = f"{'config':36s} {'n_feat':>6s} {'Spearman':>9s} {'Pearson':>8s} {'dRho':>7s} {'dR':>7s}"
    print(hdr)
    print("-" * 78)
    base_row = (f"{'structural-only baseline (Issue C)':36s} {5:>6d} "
                f"{BASELINE_SPEARMAN:>9.4f} {BASELINE_PEARSON:>8.4f} "
                f"{0.0:>7.4f} {0.0:>7.4f}")
    print(base_row)
    print("-" * 78)
    for name, cols in cfg.items():
        sp, pe = results[name]
        print(f"{name:36s} {len(cols):>6d} {sp:>9.4f} {pe:>8.4f} "
              f"{sp - BASELINE_SPEARMAN:>+7.4f} {pe - BASELINE_PEARSON:>+7.4f}")
    print("=" * 78)
    print("dRho / dR = lift over the structural-only baseline "
          "(Spearman / Pearson).")

    print(f"\nReproducibility: structural-only recomputed on this join = "
          f"Spearman {repro_sp:.4f} / Pearson {repro_pe:.4f} "
          f"(Issue C reported {BASELINE_SPEARMAN:.4f} / {BASELINE_PEARSON:.4f}).")

    print("\nRedundancy check -- Pearson(mean_slices_cleared_fraction, "
          "total_slice_count):")
    for p in PLAYERS:
        print(f"  {p:10s} frac_cleared vs total_slice_count : r = {redundancy[p]:+.4f}")


if __name__ == "__main__":
    main()
