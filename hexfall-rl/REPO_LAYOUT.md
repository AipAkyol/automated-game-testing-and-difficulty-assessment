# Repository Layout

A map of the hexfall-rl/ repo for future workers. One-line descriptions for each
directory and key file so you can orient yourself without exploring from scratch.

```text
hexfall-rl/
├── pyproject.toml            # Project metadata, dependencies, entry points
├── README.md                 # Project overview
├── REPO_LAYOUT.md            # This file — directory map for future workers
├── documents/                # Local copies of spec docs for Claude Code reference
│   ├── HEXFALL_MDP_SPEC.md   # POMDP formalization: state, observation, action, reward
│   ├── HEXFALL_RULES.md      # Authoritative game mechanics spec
│   └── LEVEL_FORMAT.md       # JSON level file format spec
├── hexfall/                  # Python package — simulator, env wrapper, types
│   ├── __init__.py           # Package init
│   ├── env.py                # Gymnasium wrapper (reset, step, observation, action mask)
│   ├── game.py               # Core Hex Fall mechanics (ticks, fall, generator, reachability)
│   ├── level_loader.py       # JSON → game state; schema validation + semantic invariants
│   ├── render.py             # CLI renderer: agent-view / full-view text dump of env state
│   ├── types.py              # Dataclasses for game state (per MDP spec §3)
│   └── schemas/
│       ├── level_schema.json        # JSON Schema (draft-07) for native level files
│       └── paxie_level_schema.json  # JSON Schema (draft-07) for Paxie editor format (Anıl's 100-level dataset)
├── levels/                   # Hand-built level JSON files used for testing
│   ├── README.md             # Level descriptions, mechanics covered, expected outcomes
│   ├── tiny_solvable.json    # Original smoke-test level (2 colors, always solvable)
│   ├── forced_lose.json      # Color mismatch → guaranteed lose from level start
│   ├── generator_test.json   # Generator fire-at-load, mid-episode fire, exhaustion
│   ├── hidden_test.json      # ?-bucket reveal-on-reachable mechanic
│   ├── deadlock_test.json    # Buffer deadlock under bad play; solvable under good play
│   └── wall_test.json        # Wall shaping of reachability graph
├── scripts/
│   ├── run_random_agent.py     # Runs a random agent end-to-end; prints renderer output per step
│   └── survey_paxie_levels.py  # Aggregate-only survey of CLASSIFIED.paxie_data/ → markdown report
└── tests/                    # pytest test suite (56 tests across 3 files)
    ├── __init__.py
    ├── test_env.py           # Tests for Gymnasium env wrapper (env.py)
    ├── test_game.py          # Tests for core mechanics (game.py)
    └── test_level_loader.py  # Tests for level loader (level_loader.py)
```

Note: documents/ contains local copies of spec docs for Claude Code reference.
Canonical versions live in the Claude project files, not the repo.

Gitignored sibling directory: `CLASSIFIED.paxie_data/` holds Anıl's 100-level
dataset (`level_data/level1.json` … `level100.json`) and the generated
`survey_report.md`. The dataset is classified; only the survey script and the
Paxie JSON Schema are committable.
