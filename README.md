# Automated Game Testing and Difficulty Assessment

### A Structural Difficulty Oracle for *Hex Fall*

---

A senior design project asking whether a model can predict per-level human winrate for a mobile puzzle game — without playtesting.

**It can — but mostly by counting how big the level is.** A 5-feature structural Ridge regression on 99 shipped Hex Fall levels predicts human winrate at **LOO-CV Spearman 0.6422** against real player data from Paxie Games. Four follow-up experiments tried to improve on that baseline using solver outcomes, an LLM-written human-tactics player, decision-structure features, and expanded structural counts. **All four closed below the baseline.** On this corpus, difficulty is dominated by structural complexity; behavioral signals from bounded-rationality agents add no predictive power.

The systematic negative — earned under pre-committed thresholds, frame guards, and hard orthogonality gates — is the project's main empirical contribution. For a studio that wants difficulty estimates without playtesting, it says something concrete: a five-number feature vector predicts difficulty as well as anything we built here, including agent simulation that is orders of magnitude more expensive.

## Headline numbers

| | |
|---|---|
| Spearman against `% Win Rate`, LOO-CV, 99 levels | **0.6422** |
| Pearson against `% Win Rate`, LOO-CV, 99 levels | **0.6210** |
| Tests passing | 130 |
| Ablation experiments — all below the +0.05 lift threshold | 4 of 4 |
| Simulator coverage | 99 of 100 commercial levels |

## Repository layout

```
.
├── hexfall-rl/        # Main codebase — simulator, players, oracle, ablation framework, tests
├── documents/         # Authoritative specs: game rules, MDP spec, level format, schema
├── final_report/      # CmpE 492 final report (LaTeX + compiled PDF)
├── midterm_report/    # March 29 midterm submission
├── poster/            # Boğaziçi poster session deliverables (June 9–10, 2026)
├── wiggle_escape/     # Earlier exploratory work on Worm Escape (Phase 1)
└── .github/           # Issue templates
```

Start with `final_report/` for the full story. The code is in `hexfall-rl/` and has its own `README.md` and `REPO_LAYOUT.md`.

## How it works

A bounded-rationality solver ensemble — greedy, depth-*k* lookahead, and Monte Carlo Tree Search — produces three clearly separated skill tiers (aggregate winrates 0.061, 0.121, 0.212). Their winrates are joined with five structural features (color count, total slice count, pin count, ice count, reserve area) and fit via Ridge regression with `StandardScaler` and leakage-free leave-one-out cross-validation against Paxie's real per-level human winrate data.

Methodological discipline was pre-committed: lift thresholds locked before experiments ran, frame guards required every ablation script to bit-reproduce the baseline before evaluating alternatives, hard orthogonality gates eliminated features structurally redundant with level size, and no post-hoc refits were allowed after negative results. At N = 99 it would be easy to manufacture spurious lift by tuning after the fact; the discipline is what makes the negative results defensible.

## The pivot

The project began as a curiosity-driven reinforcement learning framework for automated game testing across multiple Paxie titles. Following industry consultation in mid-April, the scope narrowed to a single game (Hex Fall) where the cleanest per-level human winrate data was available. Following advisor consultation on May 28, the RL framing was dropped over a *structural validity* concern: optimal agent winrate doesn't predict human winrate, which is the validation target. The bounded-rationality replacement methodology avoided this directly. The CleanRL PPO pipeline remains in `hexfall-rl/scripts/train_ppo.py` as evidence of the originally planned work and a possible extension path.

## Acknowledgments

- **Atay Özgövde** (Boğaziçi University, project supervisor) — for the May 28 consultation that surfaced the structural validity problem, and for the June 2 consultation that closed the empirical work with a clean honest result.
- **Hüseyin Anıl Özmen** (CTO, Paxie Games) — for the 99-level Hex Fall corpus and per-level human winrate data that made validation possible.

## Context

CmpE 492 Senior Design Project · Boğaziçi University · Spring 2026 · Alperen Akyol
