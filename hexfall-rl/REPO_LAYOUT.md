# Repository Layout

A map of the hexfall-rl/ repo for future workers. One-line descriptions for each
directory and key file so you can orient yourself without exploring from scratch.

```text
hexfall-rl/
├── pyproject.toml            # Project metadata, dependencies, console scripts (hexfall-render / -random-agent / -smoke-load)
├── README.md                 # Project overview
├── REPO_LAYOUT.md            # This file — directory map for future workers
├── documents/                # Local copies of spec docs for Claude Code reference
│   ├── HEXFALL_MDP_SPEC.md   # POMDP formalization: state, observation, action, reward
│   ├── HEXFALL_RULES.md      # Authoritative game mechanics spec
│   ├── LEVEL_FORMAT.md       # JSON level file format spec (Paxie native format)
│   └── level_schema.json     # JSON Schema mirror of hexfall/schemas/level_schema.json
├── hexfall/                  # Python package — simulator, env wrapper, types
│   ├── __init__.py           # Package init
│   ├── clog.py               # Clog structural features (Issue H): fcc/ctd/rhc extractors from a loaded GameState; pure, separate from oracle.py for ablation. ctd pruned at gate (slice-count redundant); fcc+rhc refit = honest negative
│   ├── env.py                # Gymnasium wrapper (reset, step, observation, action mask)
│   ├── game.py               # Core mechanics: ticks (ice thaw → pull → fill → fall → generator → reachability → pin destruction), reachability, action legality
│   ├── level_loader.py       # Paxie JSON → game state; schema + 8 semantic checks; hard-rejects unsupported mechanics
│   ├── oracle.py             # Difficulty oracle: 5 structural features + 3 player winrates → Ridge(StandardScaler) predicts human winrate; fit=20-ep matrix, predict=10-ep fresh
│   ├── render.py             # CLI renderer: agent-view / full-view text dump (handles ice + pin overlay); has main() for `hexfall-render`
│   ├── types.py              # Dataclasses: PlainBucket, QuestionBucket, IceBucket, Generator, Wall, Pin, BufferBucket, GameState
│   ├── players/              # Bounded-rationality players + evaluator (winrates to fit vs. human data)
│   │   ├── __init__.py       # Re-exports Player, GreedyPlayer, HumanTacticsPlayer, LookaheadPlayer, MCTSPlayer, evaluate, evaluate_graded
│   │   ├── base.py           # Player Protocol — act(obs, env) -> legal action index
│   │   ├── evaluator.py      # evaluate(...) -> winrate (scalar); evaluate_graded(...) -> per-episode (win, slices-cleared frac, moves_survived) + per-(player,level) aggregates; both deterministic, shared loop
│   │   ├── greedy.py         # GreedyPlayer: depth-0 heuristic (buffer colors vs. bottom-row tops)
│   │   ├── human_tactics.py  # HumanTacticsPlayer: depth-0 human-tactics heuristic — 6 weighted components (matched-now/same-color-idle/speculation/buffer-pressure/pin-setup/ice-timing); obs-only, deterministic (Issue G)
│   │   ├── lookahead.py      # LookaheadPlayer: depth-k env.fork() search, fall-sample expectation
│   │   └── mcts.py           # MCTSPlayer: UCT tree search, env.fork() rollouts (greedy default), deterministic seed seq
│   └── schemas/
│       └── level_schema.json # JSON Schema (draft-07) for Paxie-native level files
├── levels/                   # Hand-built level JSON files (all Paxie format)
│   ├── README.md             # Level descriptions, mechanics covered, expected outcomes
│   ├── tiny_solvable.json    # 2-color smoke test (always solvable)
│   ├── forced_lose.json      # Five wrong-color buckets → guaranteed deadlock
│   ├── generator_test.json   # Generator fire-at-load, mid-episode fire, exhaustion
│   ├── hidden_test.json      # ?-bucket reveal-on-reachable mechanic
│   ├── deadlock_test.json    # Buffer deadlock under bad play; solvable under good play
│   ├── wall_test.json        # Wall shaping of reachability graph
│   ├── ice_test.json         # Ice bucket thaw timing and frozen-cell legality
│   ├── pin_test.json         # Pin destruction + cascading destruction in a single tick
│   └── pin_vs_match_fixture.json # HumanTacticsPlayer pin_setup-vs-immediate-match test fixture (Issue G Phase 2)
├── scripts/
│   ├── __init__.py             # Package marker (needed for console-script entry points)
│   ├── run_players.py          # Smoke run: greedy + lookahead-1/-2 winrates on tiny_solvable & level50
│   ├── run_random_agent.py     # Runs a random agent end-to-end; prints renderer output per step
│   ├── smoke_load.py           # Loads every level{N}.json under a directory; reports OK/WARN/UNSUPPORTED/ERROR aggregate + pin geometry
│   ├── survey_paxie_levels.py  # Aggregate-only survey of CLASSIFIED.paxie_data/ → markdown report
│   ├── train_ppo.py            # PPO trainer: Dict-obs encoder, mask-in-forward policy, 30-min wall-clock checkpointing, --resume
│   ├── mcts_spotcheck.py       # Times one MCTS episode on level50 across N to pick rollout budget (locked N=100)
│   ├── run_eval_matrix.py      # 3-player × 99-level × 20-episode matrix (spawn Pool) → outputs/eval_matrix.csv
│   ├── run_eval_matrix_graded.py # Sibling of run_eval_matrix.py: identical config, calls evaluate_graded → outputs/eval_matrix_graded.csv (same seeds ⇒ winrate column bit-identical)
│   ├── run_eval_matrix_human_tactics.py # Sibling of run_eval_matrix.py: single player human_tactics × 99 × 20 → outputs/eval_matrix_human_tactics.csv (same seeds, identical schema) [Issue G]
│   ├── compute_clog_features.py # Issue H Phase 2: fcc/ctd/rhc clog features over 99 levels → CLASSIFIED.paxie_data/oracle/clog_features.csv; orthogonality gate vs total_slice_count (exits 1 if any |r| ≥ 0.7)
│   ├── fit_oracle.py           # Fits the oracle: pivots eval_matrix + joins Anıl CSV, LOO-CV Spearman/Pearson gate, ablation, scatter → CLASSIFIED.paxie_data/oracle/
│   ├── fit_oracle_graded.py    # Issue F refit: 6 graded features (frac-cleared + moves × 3 players) in 3 ablation configs, LOO-CV vs Issue-C baseline + slice-count redundancy check; prints only, no artifacts
│   ├── fit_oracle_human_tactics.py # Issue G refit: 3 configs (structural-only / +4 players / +human_tactics-only), LOO-CV vs Issue-C baseline + greedy & slice-count redundancy; prints only (verdict: human_tactics is noise)
│   └── fit_oracle_clog.py     # Issue H Phase 3 refit: 4 configs (structural-only / +fcc+rhc / +3 players / fcc+rhc-only), LOO-CV vs 0.6422 baseline, frame guard + (b) coefficients; prints only (verdict: fcc+rhc honest negative)
├── outputs/                  # Generated evaluation artifacts — all gitignored at parent level (working-tree only, none committed)
│   ├── eval_matrix.csv       # greedy/lookahead/mcts winrates per level (level_id,player,winrate,n_episodes,seed_base,wallclock_seconds)
│   ├── eval_matrix_graded.csv # Graded matrix (Issue F): level_id,player,n_episodes,seed_base,winrate,mean_slices_cleared_fraction,mean_moves_survived,wallclock_seconds
│   └── eval_matrix_human_tactics.csv # human_tactics winrate per level — same schema as eval_matrix.csv (Issue G)
├── tests/                    # pytest test suite (125 tests across 8 files)
│   ├── __init__.py
│   ├── test_env.py           # Tests for Gymnasium env wrapper (env.py)
│   ├── test_game.py          # Tests for core mechanics, ice thaw, pin destruction, cascade (game.py)
│   ├── test_level_loader.py  # Tests for level loader, schema, unsupported mechanics, semantic checks (level_loader.py)
│   ├── test_players.py       # Tests for players: Protocol, evaluator determinism, env.fork() independence, graded eval (keys/types, frac∈[0,1], win⇒frac=1.0, determinism)
│   ├── test_mcts.py          # Tests for MCTSPlayer: protocol, legal action, ≥greedy on tiny, Paxie sample, determinism
│   ├── test_human_tactics.py # Tests for HumanTacticsPlayer: protocol/legal-action, determinism, tiny+real-level e2e, matched>speculative, pin_setup>match (engine-verified) [Issue G]
│   ├── test_clog.py          # Tests for clog extractors (Issue H): keys/types, features ∈ [0,1], ctd=0 on monochrome fixture, determinism, hand-computed exact values
│   └── test_oracle.py        # Tests for oracle: 5-feature/8-vector shape, fit determinism, predict clamp [0,1], predict determinism
├── vendor/                   # Third-party reference code, kept byte-identical for upstream diffability
│   └── cleanrl_ppo_reference.py  # CleanRL ppo.py @ commit fe8d8a0 — template for scripts/train_ppo.py
└── runs/                     # (gitignored) TensorBoard event files + .pt checkpoints from train_ppo.py runs
```

Note: documents/ contains local copies of spec docs for Claude Code reference.
Canonical versions live in the Claude project files, not the repo.

Gitignored sibling directory: `CLASSIFIED.paxie_data/` holds Anıl's 100-level
dataset (`level_data/level1.json` … `level100.json`), per-level `.meta`
sidecars, the filtered user-data CSV (`user_data_hexa_fall_filtered.csv`,
comma-separated despite the name), the generated `survey_report.md`, and
`oracle/` (fit_oracle.py's `predicted_vs_real.png`, compute_clog_features.py's
`clog_features.csv`, and any derived feature/join tables — kept here because they
embed human winrates + proprietary level designs). The dataset is
classified; only the survey/smoke scripts, the oracle code, and the level
schema are committable — never the derived artifacts.

Gitignore lives in the parent directory (`../.gitignore`). It currently covers
`.venv/`, `.claude/`, `CLASSIFIED.paxie_data/`, `outputs/`, plus Python
build/cache patterns. `runs/` should be added there — TensorBoard event files
and multi-MB checkpoint `.pt` files accumulate fast and are not reproducible
build artifacts.
