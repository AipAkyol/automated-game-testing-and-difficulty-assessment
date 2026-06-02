"""Issue I: refit the difficulty oracle with the 8 expanded structural-count features.

Final solver-revival experiment. Sibling of scripts/fit_oracle_clog.py (prints-only;
writes NO artifacts, reads NO oracle PNG). Does NOT modify hexfall/oracle.py's default
5-structural + 3-player (8-vector) behavior or its bit-identical
extract_structural_features (which backs the 0.6422 frame guard). The structural
extractor, the new extract_expanded_features sibling, ALPHA, and the
Ridge(StandardScaler) + leakage-free LOO-CV machinery are imported/reused from
oracle.py + sklearn, not forked. Clog's fcc/ctd/rhc are NOT wired in here.

Data join (99 levels), identical frame/seeds to fit_oracle.py / fit_oracle_clog.py so
configs are apples-to-apples:
  - greedy/lookahead/mcts winrates <- outputs/eval_matrix.csv
  - 5 structural + 8 expanded      <- computed on the same loaded GameState
  - target "% Win Rate"            <- Anil's CSV, joined by level_num

Three ablation configs, each LOO-CV (Ridge(StandardScaler), alpha=1.0):
  (a) structural-only                          [5]   (must reproduce 0.6422 bit-identically)
  (b) structural + 8 expanded                  [13]
  (c) structural + 8 expanded + 3 players      [16]

Decision: does any config clear 0.6422 + 0.05 = 0.6922 Spearman?
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
    EXPANDED_FEATURE_NAMES,
    STRUCTURAL_FEATURE_NAMES,
    extract_expanded_features,
    extract_structural_features,
)

REPO = Path(__file__).resolve().parents[1]
EVAL_MATRIX = REPO / "outputs" / "eval_matrix.csv"
ANIL_CSV = REPO / "CLASSIFIED.paxie_data" / "user_data_hexa_fall_filtered.csv"
LEVEL_DIR = REPO / "CLASSIFIED.paxie_data" / "level_data"

TARGET_COL = "% Win Rate"  # pinned literal; NOT the adjacent "% Win User Rate"

# Issue C structural-only LOO-CV baseline (the number each config must beat).
BASELINE_SPEARMAN = 0.6422
BASELINE_PEARSON = 0.6210
LIFT_THRESHOLD = 0.05
REPRO_TOL = 1e-4  # config (a) must reproduce the baseline to 4 dp (frame guard)

PLAYER3_NAMES = ["greedy_winrate", "lookahead_winrate", "mcts_winrate"]


def load_player3_winrates() -> pd.DataFrame:
    """Pivot eval_matrix long->wide: one row per level, 3 winrate columns.

    Mirrors scripts/fit_oracle.py.load_player_winrates exactly.
    """
    df = pd.read_csv(EVAL_MATRIX)
    wide = df.pivot(index="level_id", columns="player", values="winrate")
    wide = wide.rename(
        columns={
            "greedy": "greedy_winrate",
            "lookahead": "lookahead_winrate",
            "mcts": "mcts_winrate",
        }
    )
    wide = wide.reset_index()
    wide["level_num"] = wide["level_id"].str.replace("level", "", regex=False).astype(int)
    return wide[["level_id", "level_num"] + PLAYER3_NAMES]


def load_human_winrates() -> pd.DataFrame:
    """Load Anil's CSV (comma-separated). Same logic as fit_oracle.py."""
    df = pd.read_csv(ANIL_CSV, sep=",")
    assert df["Level"].is_unique, "Anil 'Level' column has duplicates; join would fan out"
    assert df[TARGET_COL].max() <= 1.0, f"{TARGET_COL} looks like a percent, not a fraction"
    return df[["Level", TARGET_COL]]


def make_pipeline(alpha: float) -> Pipeline:
    """Identical to hexfall.oracle.Oracle's internal pipeline (StandardScaler -> Ridge)."""
    return Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=alpha))])


def loo_corr(X_sub: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Leakage-free LOO-CV out-of-fold Spearman + Pearson (alpha=1.0)."""
    oof = cross_val_predict(make_pipeline(ALPHA), X_sub, y, cv=LeaveOneOut())
    return spearmanr(y, oof)[0], pearsonr(y, oof)[0]


def main() -> None:
    player3 = load_player3_winrates()
    human = load_human_winrates()

    # --- Join player frame to the human target (same as fit_oracle.py) -------
    joined = player3.merge(human, left_on="level_num", right_on="Level", how="inner")
    print(f"Join size: {len(joined)} rows "
          f"(eval levels={player3['level_num'].nunique()}, "
          f"human levels={human['Level'].nunique()})")
    if len(joined) < 90:
        raise SystemExit(f"STOP: join size {len(joined)} < 90 -> key mismatch, not a result.")

    # --- Assemble a named feature frame (one source of truth, slice by name) --
    # structural + expanded are computed on the SAME loaded state, so every
    # config shares one frame.
    struct_rows, exp_rows = [], []
    for _, r in joined.iterrows():
        state = load_level(LEVEL_DIR / f"level{int(r['level_num'])}.json")
        struct_rows.append(extract_structural_features(state))
        ef = extract_expanded_features(state)
        exp_rows.append([ef[n] for n in EXPANDED_FEATURE_NAMES])
    struct_df = pd.DataFrame(struct_rows, columns=STRUCTURAL_FEATURE_NAMES, index=joined.index)
    exp_df = pd.DataFrame(exp_rows, columns=EXPANDED_FEATURE_NAMES, index=joined.index)

    feat = pd.concat(
        [joined[PLAYER3_NAMES].reset_index(drop=True),
         struct_df.reset_index(drop=True),
         exp_df.reset_index(drop=True)],
        axis=1,
    )
    assert not feat.isna().any().any(), "NaN in feature matrix; investigate before fitting"
    y = joined[TARGET_COL].to_numpy(dtype=float)

    def X(cols: list[str]) -> np.ndarray:
        return feat[cols].to_numpy(dtype=float)

    # --- Three ablation configs ---------------------------------------------
    cfg = {
        "(a) structural-only": STRUCTURAL_FEATURE_NAMES,
        "(b) structural + 8 expanded": STRUCTURAL_FEATURE_NAMES + EXPANDED_FEATURE_NAMES,
        "(c) structural + 8 expanded + 3 players":
            STRUCTURAL_FEATURE_NAMES + EXPANDED_FEATURE_NAMES + PLAYER3_NAMES,
    }
    results = {name: loo_corr(X(cols), y) for name, cols in cfg.items()}

    # --- Frame-mismatch guard: (a) MUST reproduce the Issue-C baseline -------
    repro_sp, repro_pe = results["(a) structural-only"]
    print(f"\nFrame guard: structural-only (a) = Spearman {repro_sp:.4f} / "
          f"Pearson {repro_pe:.4f}  (Issue C: {BASELINE_SPEARMAN:.4f} / {BASELINE_PEARSON:.4f})")
    if (abs(repro_sp - BASELINE_SPEARMAN) > REPRO_TOL
            or abs(repro_pe - BASELINE_PEARSON) > REPRO_TOL):
        raise SystemExit(
            f"STOP: structural-only did NOT reproduce the Issue-C baseline within "
            f"{REPRO_TOL} (got Spearman {repro_sp:.6f} / Pearson {repro_pe:.6f}, "
            f"expected {BASELINE_SPEARMAN} / {BASELINE_PEARSON}). Frame mismatch — "
            "level set / join / eval_matrix differs from Issue C; results are NOT "
            "apples-to-apples. Aborting before reporting lifts."
        )
    print(f"  -> reproduced within {REPRO_TOL}: frame is apples-to-apples with Issue C.")

    # --- Results table -------------------------------------------------------
    print("\n" + "=" * 80)
    print(f"RESULTS  (LOO-CV, Ridge(StandardScaler), alpha={ALPHA}; N = {len(joined)} levels)")
    print("=" * 80)
    hdr = f"{'config':42s} {'n_feat':>6s} {'Spearman':>9s} {'Pearson':>8s} {'dRho':>8s} {'dR':>8s}"
    print(hdr)
    print("-" * 80)
    print(f"{'structural-only baseline (Issue C)':42s} {5:>6d} "
          f"{BASELINE_SPEARMAN:>9.4f} {BASELINE_PEARSON:>8.4f} {0.0:>8.4f} {0.0:>8.4f}")
    print("-" * 80)
    for name, cols in cfg.items():
        sp, pe = results[name]
        print(f"{name:42s} {len(cols):>6d} {sp:>9.4f} {pe:>8.4f} "
              f"{sp - BASELINE_SPEARMAN:>+8.4f} {pe - BASELINE_PEARSON:>+8.4f}")
    print("=" * 80)
    print("dRho / dR = lift over the structural-only baseline (Spearman / Pearson).")

    # --- config (b) fitted coefficient vector (full-data fit) ----------------
    cols_b = cfg["(b) structural + 8 expanded"]
    pipe_b = make_pipeline(ALPHA).fit(X(cols_b), y)
    coef_b = dict(zip(cols_b, (float(c) for c in pipe_b.named_steps["ridge"].coef_)))
    print("\nConfig (b) fitted Ridge coefficient vector [standardized features]:")
    for name in cols_b:
        tag = "  [expanded]" if name in EXPANDED_FEATURE_NAMES else "  [structural]"
        print(f"    {name:30s} {coef_b[name]:+.4f}{tag}")

    # --- Sanity check: each new feature vs total_slice_count -----------------
    # High correlation is EXPECTED (counts scale with level size) and is NOT
    # grounds to drop a feature — Ridge decides. Reported for transparency only.
    tsc = feat["total_slice_count"].to_numpy(dtype=float)
    print("\nSanity check — Pearson of each expanded feature vs total_slice_count "
          f"(N = {len(joined)}; high r is expected, not disqualifying):")
    for name in EXPANDED_FEATURE_NAMES:
        col = feat[name].to_numpy(dtype=float)
        if np.allclose(col, col[0]):
            print(f"  {name:30s} r =   n/a (constant column)")
        else:
            print(f"  {name:30s} r = {pearsonr(col, tsc)[0]:+.4f}")

    # --- Decision ------------------------------------------------------------
    bar = BASELINE_SPEARMAN + LIFT_THRESHOLD
    sp_b = results["(b) structural + 8 expanded"][0]
    sp_c = results["(c) structural + 8 expanded + 3 players"][0]
    best_sp = max(sp_b, sp_c)
    best_cfg = "(b)" if sp_b >= sp_c else "(c)"
    lift = best_sp - BASELINE_SPEARMAN
    print("\n" + "=" * 80)
    print(f"DECISION (threshold: Spearman lift >= {LIFT_THRESHOLD:.2f}, i.e. >= {bar:.4f}):")
    print(f"  best of (b),(c) = {best_cfg} with Spearman {best_sp:.4f} "
          f"(lift {lift:+.4f} over {BASELINE_SPEARMAN:.4f})")
    if best_sp >= bar:
        print(f"  => CLEARS the bar: config {best_cfg} reaches >= {bar:.4f}. "
              "Expanded structural counts add oracle signal.")
    else:
        print(f"  => HONEST NEGATIVE: no config clears {bar:.4f}. The 8 expanded "
              "structural counts add no oracle signal over the structural-only baseline.")
    print("=" * 80)


if __name__ == "__main__":
    main()
