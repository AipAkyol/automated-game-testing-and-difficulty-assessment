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
│   ├── env.py                # Gymnasium wrapper (reset, step, observation, action mask)
│   ├── game.py               # Core mechanics: ticks (ice thaw → pull → fill → fall → generator → reachability → pin destruction), reachability, action legality
│   ├── level_loader.py       # Paxie JSON → game state; schema + 8 semantic checks; hard-rejects unsupported mechanics
│   ├── render.py             # CLI renderer: agent-view / full-view text dump (handles ice + pin overlay); has main() for `hexfall-render`
│   ├── types.py              # Dataclasses: PlainBucket, QuestionBucket, IceBucket, Generator, Wall, Pin, BufferBucket, GameState
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
│   └── pin_test.json         # Pin destruction + cascading destruction in a single tick
├── scripts/
│   ├── __init__.py             # Package marker (needed for console-script entry points)
│   ├── run_random_agent.py     # Runs a random agent end-to-end; prints renderer output per step
│   ├── smoke_load.py           # Loads every level{N}.json under a directory; reports OK/WARN/UNSUPPORTED/ERROR aggregate + pin geometry
│   └── survey_paxie_levels.py  # Aggregate-only survey of CLASSIFIED.paxie_data/ → markdown report
└── tests/                    # pytest test suite (86 tests across 3 files)
    ├── __init__.py
    ├── test_env.py           # Tests for Gymnasium env wrapper (env.py)
    ├── test_game.py          # Tests for core mechanics, ice thaw, pin destruction, cascade (game.py)
    └── test_level_loader.py  # Tests for level loader, schema, unsupported mechanics, semantic checks (level_loader.py)
```

Note: documents/ contains local copies of spec docs for Claude Code reference.
Canonical versions live in the Claude project files, not the repo.

Gitignored sibling directory: `CLASSIFIED.paxie_data/` holds Anıl's 100-level
dataset (`level_data/level1.json` … `level100.json`), per-level `.meta`
sidecars, the filtered user-data CSV, and the generated `survey_report.md`.
The dataset is classified; only the survey/smoke scripts and the level schema
are committable.
