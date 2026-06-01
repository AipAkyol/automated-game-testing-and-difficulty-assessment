"""Issue G Phase 3: refit the difficulty oracle with the human_tactics winrate.

Sibling of scripts/fit_oracle_graded.py (prints-only; writes NO artifacts, reads
NO oracle PNG). Does NOT modify hexfall/oracle.py's default 5-structural +
3-player (8-vector) behavior or scripts/fit_oracle.py — test_oracle.py's shape
assertions stay green. The player-column set is parametrized locally here; the
feature extractor, ALPHA, and the Ridge(StandardScaler) + leakage-free LOO-CV
machinery are imported/reused from oracle.py + sklearn, not forked.

Data join (99 levels):
  - greedy/lookahead/mcts winrates  <- outputs/eval_matrix.csv
  - human_tactics winrate           <- outputs/eval_matrix_human_tactics.csv
  joined by level_id, then to Anil's "% Win Rate" target exactly as fit_oracle.py.

Three ablation configs, each LOO-CV (Ridge(StandardScaler), alpha=1.0):
  (a) structural-only                                   [5]
  (b) structural + greedy,lookahead,mcts,human_tactics  [9]
  (c) structural + human_tactics ONLY                   [6]

Decision: >= 0.05 LOO-CV Spearman lift over the Issue C baseline (0.6422), i.e.
Spearman >= 0.6922, in config (b) OR (c) -> keep human_tactics; else call it noise.
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
EVAL_MATRIX = REPO / "outputs" / "eval_matrix.csv"
HT_MATRIX = REPO / "outputs" / "eval_matrix_human_tactics.csv"
ANIL_CSV = REPO / "CLASSIFIED.paxie_data" / "user_data_hexa_fall_filtered.csv"
LEVEL_DIR = REPO / "CLASSIFIED.paxie_data" / "level_data"

TARGET_COL = "% Win Rate"  # pinned literal; NOT the adjacent "% Win User Rate"

# Issue C structural-only LOO-CV baseline (the number each config must beat).
BASELINE_SPEARMAN = 0.6422
BASELINE_PEARSON = 0.6210
LIFT_THRESHOLD = 0.05

PLAYER3_NAMES = ["greedy_winrate", "lookahead_winrate", "mcts_winrate"]
HT_NAME = "human_tactics_winrate"


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


def load_human_tactics_winrates() -> pd.DataFrame:
    """One row per level: the human_tactics winrate."""
    df = pd.read_csv(HT_MATRIX)
    players = set(df["player"].unique())
    assert players == {"human_tactics"}, f"expected only human_tactics, got {players}"
    out = df[["level_id", "winrate"]].rename(columns={"winrate": HT_NAME})
    out["level_num"] = out["level_id"].str.replace("level", "", regex=False).astype(int)
    return out[["level_num", HT_NAME]]


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
    ht = load_human_tactics_winrates()
    human = load_human_winrates()

    # --- Join the two player frames, then the human target --------------------
    players = player3.merge(ht, on="level_num", how="inner")
    joined = players.merge(human, left_on="level_num", right_on="Level", how="inner")
    print(f"Join size: {len(joined)} rows "
          f"(eval3 levels={player3['level_num'].nunique()}, "
          f"human_tactics levels={ht['level_num'].nunique()}, "
          f"human levels={human['Level'].nunique()})")
    if len(joined) < 90:
        raise SystemExit(f"STOP: join size {len(joined)} < 90 -> key mismatch, not a result.")

    # --- Assemble a named feature frame (one source of truth, slice by name) --
    struct_rows = []
    for _, r in joined.iterrows():
        state = load_level(LEVEL_DIR / f"level{int(r['level_num'])}.json")
        struct_rows.append(extract_structural_features(state))
    struct_df = pd.DataFrame(struct_rows, columns=STRUCTURAL_FEATURE_NAMES, index=joined.index)

    feat = pd.concat(
        [joined[PLAYER3_NAMES + [HT_NAME]].reset_index(drop=True),
         struct_df.reset_index(drop=True)],
        axis=1,
    )
    assert not feat.isna().any().any(), "NaN in feature matrix; investigate before fitting"
    y = joined[TARGET_COL].to_numpy(dtype=float)

    def X(cols: list[str]) -> np.ndarray:
        return feat[cols].to_numpy(dtype=float)

    # --- Three ablation configs ----------------------------------------------
    cfg = {
        "(a) structural-only": STRUCTURAL_FEATURE_NAMES,
        "(b) structural + 4 players": STRUCTURAL_FEATURE_NAMES + PLAYER3_NAMES + [HT_NAME],
        "(c) structural + human_tactics": STRUCTURAL_FEATURE_NAMES + [HT_NAME],
    }
    results = {name: loo_corr(X(cols), y) for name, cols in cfg.items()}

    # --- human_tactics fitted coefficient in (b) and (c) (full-data fit) ------
    def ht_coef(cols: list[str]) -> tuple[float, dict[str, float]]:
        pipe = make_pipeline(ALPHA).fit(X(cols), y)
        coefs = dict(zip(cols, (float(c) for c in pipe.named_steps["ridge"].coef_)))
        return coefs[HT_NAME], coefs

    ht_coef_b, all_coef_b = ht_coef(cfg["(b) structural + 4 players"])
    ht_coef_c, all_coef_c = ht_coef(cfg["(c) structural + human_tactics"])

    # --- Redundancy checks ----------------------------------------------------
    htw = feat[HT_NAME].to_numpy(dtype=float)
    tsc = feat["total_slice_count"].to_numpy(dtype=float)
    gw = feat["greedy_winrate"].to_numpy(dtype=float)
    r_ht_tsc = pearsonr(htw, tsc)[0]
    r_ht_greedy = pearsonr(htw, gw)[0]

    # --- Reproducibility: recompute structural-only on this join --------------
    repro_sp, repro_pe = loo_corr(X(STRUCTURAL_FEATURE_NAMES), y)

    # --- Results table --------------------------------------------------------
    print("\n" + "=" * 80)
    print(f"RESULTS  (LOO-CV, Ridge(StandardScaler), alpha={ALPHA}; N = {len(joined)} levels)")
    print("=" * 80)
    hdr = f"{'config':34s} {'n_feat':>6s} {'Spearman':>9s} {'Pearson':>8s} {'dRho':>8s} {'dR':>8s}"
    print(hdr)
    print("-" * 80)
    print(f"{'structural-only baseline (Issue C)':34s} {5:>6d} "
          f"{BASELINE_SPEARMAN:>9.4f} {BASELINE_PEARSON:>8.4f} {0.0:>8.4f} {0.0:>8.4f}")
    print("-" * 80)
    for name, cols in cfg.items():
        sp, pe = results[name]
        print(f"{name:34s} {len(cols):>6d} {sp:>9.4f} {pe:>8.4f} "
              f"{sp - BASELINE_SPEARMAN:>+8.4f} {pe - BASELINE_PEARSON:>+8.4f}")
    print("=" * 80)
    print("dRho / dR = lift over the structural-only baseline (Spearman / Pearson).")

    print(f"\nReproducibility: structural-only recomputed on this join = "
          f"Spearman {repro_sp:.4f} / Pearson {repro_pe:.4f} "
          f"(Issue C reported {BASELINE_SPEARMAN:.4f} / {BASELINE_PEARSON:.4f}).")

    print("\nhuman_tactics fitted Ridge coefficient (standardized features):")
    print(f"  in config (b) [9-vector]: {ht_coef_b:+.4f}")
    print(f"  in config (c) [6-vector]: {ht_coef_c:+.4f}")
    print("\n  full coefficient vector, config (b):")
    for name in cfg["(b) structural + 4 players"]:
        print(f"    {name:22s} {all_coef_b[name]:+.4f}")

    print("\nRedundancy checks (Pearson on the 99-level joined frame):")
    print(f"  human_tactics_winrate vs total_slice_count : r = {r_ht_tsc:+.4f}   "
          "(the check that killed Issues C & F)")
    print(f"  human_tactics_winrate vs greedy_winrate    : r = {r_ht_greedy:+.4f}   "
          "(confirm the Phase 2 0.952 finding)")

    # --- Decision -------------------------------------------------------------
    sp_b = results["(b) structural + 4 players"][0]
    sp_c = results["(c) structural + human_tactics"][0]
    best_sp = max(sp_b, sp_c)
    best_cfg = "(b)" if sp_b >= sp_c else "(c)"
    lift = best_sp - BASELINE_SPEARMAN
    print("\n" + "=" * 80)
    print(f"DECISION (threshold: Spearman lift >= {LIFT_THRESHOLD:.2f}, i.e. "
          f">= {BASELINE_SPEARMAN + LIFT_THRESHOLD:.4f}):")
    print(f"  best config = {best_cfg} with Spearman {best_sp:.4f} "
          f"(lift {lift:+.4f} over {BASELINE_SPEARMAN:.4f})")
    if lift >= LIFT_THRESHOLD:
        print(f"  => KEEP human_tactics: config {best_cfg} clears the +{LIFT_THRESHOLD:.2f} bar.")
    else:
        print(f"  => NOISE: neither (b) nor (c) clears +{LIFT_THRESHOLD:.2f} over the "
              f"structural-only baseline. human_tactics adds no oracle signal.")
    print("=" * 80)


if __name__ == "__main__":
    main()
