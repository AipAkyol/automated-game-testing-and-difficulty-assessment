"""Fit the difficulty oracle and report leakage-free LOO-CV correlations.

Combines precomputed player winrates (outputs/eval_matrix.csv) with structural
features and fits Ridge against Anil's human "% Win Rate".

GATE: out-of-fold (LOO-CV) Spearman > 0.5 -> proceed to Issue D.

CLASSIFICATION: the joined table embeds Anil's % Win Rate, and the feature
table is derived from proprietary level designs. Neither goes in git. The
scatter plot is written under CLASSIFIED.paxie_data/oracle/ (inherits the
parent .gitignore), NOT outputs/ (which is committed). Code only lives in repo.
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
from hexfall.oracle import (
    ALPHA,
    FEATURE_NAMES,
    STRUCTURAL_FEATURE_NAMES,
    extract_structural_features,
)

REPO = Path(__file__).resolve().parents[1]
EVAL_MATRIX = REPO / "outputs" / "eval_matrix.csv"
ANIL_CSV = REPO / "CLASSIFIED.paxie_data" / "user_data_hexa_fall_filtered.csv"
LEVEL_DIR = REPO / "CLASSIFIED.paxie_data" / "level_data"
PLOT_OUT = REPO / "CLASSIFIED.paxie_data" / "oracle" / "predicted_vs_real.png"

TARGET_COL = "% Win Rate"  # pinned literal; NOT the adjacent "% Win User Rate"


def load_player_winrates() -> pd.DataFrame:
    """Pivot eval_matrix long->wide: one row per level_id, 3 winrate columns."""
    df = pd.read_csv(EVAL_MATRIX)
    wide = df.pivot(index="level_id", columns="player", values="winrate")
    # NOTE: the "lookahead" column is depth-2 (Issue B's lookahead-2 aggregate
    # 0.061 matches the matrix). Label it lookahead_winrate for poster accuracy.
    wide = wide.rename(
        columns={
            "greedy": "greedy_winrate",
            "lookahead": "lookahead_winrate",
            "mcts": "mcts_winrate",
        }
    )
    # Normalize join key: "level1" -> 1 (no leading zeros, no separator).
    wide = wide.reset_index()
    wide["level_num"] = wide["level_id"].str.replace("level", "", regex=False).astype(int)
    return wide


def load_human_winrates() -> pd.DataFrame:
    """Load Anil's CSV (comma-separated; tab claim was wrong — see Step 0)."""
    df = pd.read_csv(ANIL_CSV, sep=",")
    assert df["Level"].is_unique, "Anil 'Level' column has duplicates; join would fan out"
    assert df[TARGET_COL].max() <= 1.0, f"{TARGET_COL} looks like a percent, not a fraction"
    return df[["Level", TARGET_COL]]


def main() -> None:
    players = load_player_winrates()
    human = load_human_winrates()

    # --- Inner-join + sanity check ------------------------------------------
    joined = players.merge(human, left_on="level_num", right_on="Level", how="inner")
    eval_ids = set(players["level_num"])
    human_ids = set(human["Level"])
    dropped_from_eval = sorted(eval_ids - human_ids)  # in eval, no human row
    print(f"Join size: {len(joined)} rows "
          f"(eval levels={len(eval_ids)}, human levels={len(human_ids)})")
    print(f"Eval level IDs with no human match: {dropped_from_eval or 'none'}")
    if len(joined) < 90:
        raise SystemExit(
            f"STOP: join size {len(joined)} < 90 -> key mismatch, not a result."
        )

    # --- Assemble 8-feature matrix ------------------------------------------
    rows = []
    for _, r in joined.iterrows():
        level_path = LEVEL_DIR / f"level{int(r['level_num'])}.json"
        state = load_level(level_path)
        structural = extract_structural_features(state)
        features = structural + [
            float(r["greedy_winrate"]),
            float(r["lookahead_winrate"]),
            float(r["mcts_winrate"]),
        ]
        rows.append(features)
    X = np.array(rows, dtype=float)
    y = joined[TARGET_COL].to_numpy(dtype=float)
    print(f"Feature matrix X: {X.shape}, target y: {y.shape}")

    # --- Leakage-free LOO-CV (scaler refits per fold, inside the pipeline) ---
    def make_pipeline(alpha: float) -> Pipeline:
        return Pipeline(
            [("scaler", StandardScaler()), ("ridge", Ridge(alpha=alpha))]
        )

    oof = cross_val_predict(make_pipeline(ALPHA), X, y, cv=LeaveOneOut())
    sp_rho, _ = spearmanr(y, oof)
    pe_r, _ = pearsonr(y, oof)
    print(f"\n=== LOO-CV (alpha={ALPHA}, PRE-COMMITTED) — THE GATE NUMBERS ===")
    print(f"  Spearman: {sp_rho:.4f}")
    print(f"  Pearson : {pe_r:.4f}")

    # --- Ablation: variance partition (same pipeline, same LOO, alpha=1.0) ---
    # Column order is [5 structural] + [greedy, lookahead, mcts]. Slice to drop
    # one block; everything else (pipeline, CV, correlations) is identical.
    def loo_corr(X_sub: np.ndarray) -> tuple[float, float]:
        oof_sub = cross_val_predict(make_pipeline(ALPHA), X_sub, y, cv=LeaveOneOut())
        return spearmanr(y, oof_sub)[0], pearsonr(y, oof_sub)[0]

    struct_sp, struct_pe = loo_corr(X[:, :5])   # drop 3 player columns
    player_sp, player_pe = loo_corr(X[:, 5:])   # drop 5 structural columns
    print("\n=== Ablation (LOO-CV, alpha=1.0) — variance partition ===")
    print(f"  full model      : Spearman {sp_rho:.4f} / Pearson {pe_r:.4f}   [8 features]")
    print(f"  structural-only : Spearman {struct_sp:.4f} / Pearson {struct_pe:.4f}   [5 features]")
    print(f"  player-only     : Spearman {player_sp:.4f} / Pearson {player_pe:.4f}   [3 features]")

    # --- Alpha sensitivity sidebar (NOT used for the gate) ------------------
    print("\nAlpha sensitivity sidebar (informational only, gate uses alpha=1.0):")
    for a in (0.1, 1.0, 10.0):
        oof_a = cross_val_predict(make_pipeline(a), X, y, cv=LeaveOneOut())
        rho_a, _ = spearmanr(y, oof_a)
        print(f"  alpha={a:<5}: Spearman={rho_a:.4f}")

    # --- Full-data fit for interpretable coefficients -----------------------
    full = make_pipeline(ALPHA).fit(X, y)
    coefs = full.named_steps["ridge"].coef_
    print("\nFull-data fit coefficients (standardized features, for poster):")
    for name, c in zip(FEATURE_NAMES, coefs):
        tag = "  [structural]" if name in STRUCTURAL_FEATURE_NAMES else "  [player]"
        print(f"  {name:20s} {c:+.4f}{tag}")

    # --- Scatter plot (CLASSIFIED: embeds Anil's real winrate) --------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    PLOT_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y, oof, alpha=0.7, edgecolor="k", linewidth=0.3)
    lims = [min(y.min(), oof.min()), max(y.max(), oof.max())]
    ax.plot(lims, lims, "r--", linewidth=1, label="y = x")
    ax.set_xlabel("Real human winrate (% Win Rate)")
    ax.set_ylabel("Oracle out-of-fold prediction")
    ax.set_title("Oracle LOO-CV: predicted vs real")
    ax.annotate(
        f"Spearman = {sp_rho:.3f}\nPearson = {pe_r:.3f}",
        xy=(0.05, 0.95), xycoords="axes fraction", va="top",
        bbox=dict(boxstyle="round", fc="white", ec="gray"),
    )
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(PLOT_OUT, dpi=120)
    print(f"\nScatter saved to {PLOT_OUT} (CLASSIFIED — not committed)")

    # --- Gate verdict --------------------------------------------------------
    print("\n" + "=" * 60)
    if sp_rho > 0.5:
        print(f"GATE PASS: LOO-CV Spearman {sp_rho:.4f} > 0.5 -> PROCEED TO ISSUE D")
    else:
        print(f"GATE FAIL: LOO-CV Spearman {sp_rho:.4f} <= 0.5 -> "
              "FALL BACK, RECOMMEND ESCALATION")
    print("=" * 60)


if __name__ == "__main__":
    main()
