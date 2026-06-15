"""Ephemeral: dump structural-only oracle OOF predictions + standardized coefs.

Structural-only (5 features, NO player winrates) Ridge + StandardScaler + LOO-CV.
Mirrors scripts/fit_oracle.py's pipeline, sliced to the structural block, for the
poster's hero scatter. Not committed; artifacts under CLASSIFIED.paxie_data/.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from hexfall.level_loader import load_level
from hexfall.oracle import ALPHA, STRUCTURAL_FEATURE_NAMES, extract_structural_features

REPO = Path(__file__).resolve().parents[1]
ANIL_CSV = REPO / "CLASSIFIED.paxie_data" / "user_data_hexa_fall_filtered.csv"
LEVEL_DIR = REPO / "CLASSIFIED.paxie_data" / "level_data"
OUT_DIR = REPO / "CLASSIFIED.paxie_data" / "oracle"
TARGET_COL = "% Win Rate"

EXPECTED_SP = 0.6422
EXPECTED_PE = 0.6210
TOL = 0.001


def main() -> None:
    # --- Human target -------------------------------------------------------
    human = pd.read_csv(ANIL_CSV, sep=",")[["Level", TARGET_COL]]
    assert human["Level"].is_unique, "Anil 'Level' column has duplicates"

    # --- Structural features for levels 1..99 -------------------------------
    rows = []
    for lvl in range(1, 100):  # skip level 100 (unsupported)
        state = load_level(LEVEL_DIR / f"level{lvl}.json")
        rows.append({"level_id": lvl, **dict(
            zip(STRUCTURAL_FEATURE_NAMES, extract_structural_features(state)))})
    feats = pd.DataFrame(rows)

    # --- Join + verify ------------------------------------------------------
    joined = feats.merge(human, left_on="level_id", right_on="Level", how="inner")
    if len(joined) != 99:
        raise SystemExit(f"STOP: expected 99 joined rows, got {len(joined)}")

    X = joined[STRUCTURAL_FEATURE_NAMES].to_numpy(dtype=float)
    y = joined[TARGET_COL].to_numpy(dtype=float)
    level_ids = joined["level_id"].to_numpy(dtype=int)

    def make_pipeline() -> Pipeline:
        return Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=ALPHA))])

    # --- Leakage-free LOO-CV (scaler refits per fold) -----------------------
    oof = cross_val_predict(make_pipeline(), X, y, cv=LeaveOneOut())
    sp_rho, _ = spearmanr(y, oof)
    pe_r, _ = pearsonr(y, oof)
    print(f"Joined rows: {len(joined)}")
    print(f"Structural-only LOO-CV Spearman: {sp_rho:.4f}  (expect ~{EXPECTED_SP})")
    print(f"Structural-only LOO-CV Pearson : {pe_r:.4f}  (expect ~{EXPECTED_PE})")

    if abs(sp_rho - EXPECTED_SP) > TOL or abs(pe_r - EXPECTED_PE) > TOL:
        raise SystemExit(
            f"ABORT: correlations off by >{TOL} from expected "
            f"(Spearman {sp_rho:.4f} vs {EXPECTED_SP}, Pearson {pe_r:.4f} vs {EXPECTED_PE}) "
            "-> check join / feature extraction.")

    # --- Full-data fit for standardized coefficients ------------------------
    full = make_pipeline().fit(X, y)
    coefs = full.named_steps["ridge"].coef_
    print("\nStandardized coefficients (full-data fit):")
    for name, c in zip(STRUCTURAL_FEATURE_NAMES, coefs):
        print(f"  {name:20s} {c:+.4f}")

    # --- Write CSVs ---------------------------------------------------------
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    oof_df = pd.DataFrame(
        {"level_id": level_ids, "real_winrate": y, "predicted": oof})
    coef_df = pd.DataFrame(
        {"feature": STRUCTURAL_FEATURE_NAMES, "standardized_coefficient": coefs})
    oof_path = OUT_DIR / "structural_only_oof_predictions.csv"
    coef_path = OUT_DIR / "structural_only_coefficients.csv"
    oof_df.to_csv(oof_path, index=False)
    coef_df.to_csv(coef_path, index=False)

    print(f"\nWrote {oof_path} ({len(oof_df)} rows)")
    print(oof_df.head().to_string(index=False))
    print(f"\nWrote {coef_path} ({len(coef_df)} rows)")
    print(coef_df.to_string(index=False))


if __name__ == "__main__":
    main()
