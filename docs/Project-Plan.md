# Project Plan — Automated Game Testing and Difficulty Assessment

**Course:** CmpE 492 – Senior Design Project  
**Project Owner:** Alperen Akyol  
**Project Supervisor:** Atay Özgövde  
**Repository:** [AipAkyol/automated-game-testing-and-difficulty-assessment](https://github.com/AipAkyol/automated-game-testing-and-difficulty-assessment)

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Key Dates & Milestones](#2-key-dates--milestones)
3. [Phase 1 – Foundation & Midterm (up to 29 March)](#3-phase-1--foundation--midterm-up-to-29-march)
4. [Phase 2 – Core Research & Development (30 March – 31 May)](#4-phase-2--core-research--development-30-march--31-may)
5. [Phase 3 – Final Wrap-up & Presentations (1 June – 11 June)](#5-phase-3--final-wrap-up--presentations-1-june--11-june)
6. [Technical Architecture](#6-technical-architecture)
7. [Deliverables Summary](#7-deliverables-summary)
8. [Risks & Mitigations](#8-risks--mitigations)

---

## 1. Project Overview

This project investigates **automated playtesting and difficulty assessment** for tile-based puzzle games.  The subject game is **Worm Escape** — a sliding-puzzle where players extract coloured "worms" from a grid by sliding them out along their facing direction.

### Goals

| Goal | Description |
|------|-------------|
| **Automated Solver** | Build an algorithm that can determine whether any given puzzle configuration is solvable and produce a valid extraction order. |
| **Difficulty Metric** | Derive a quantitative difficulty score for each level based on solver analysis (number of resolution rounds, cyclic dependencies, look-ahead depth, etc.). |
| **Level Generation** | Procedurally generate new puzzle levels of controlled difficulty. |
| **Level Editor** | Provide an interactive editor so that new levels can be designed, validated, and saved manually. |
| **Evaluation** | Compare the automated difficulty estimates against human playtest data (move counts, completion time) to validate the metric. |

---

## 2. Key Dates & Milestones

| Date | Event |
|------|-------|
| **Start of Term** | Project kick-off; repository established |
| **Week 3 (mid-Feb)** | Game engine & level representation finalised |
| **Week 5 (end-Feb)** | Level editor and first set of hand-crafted levels ready |
| **Week 7 (mid-Mar)** | Automated solver (Directed Dependency Graph) implemented & unit-tested |
| **⭐ 29 March** | **Midterm Report due** |
| **Week 10 (mid-Apr)** | Difficulty scoring formula defined and calibrated |
| **Week 11 (end-Apr)** | Procedural level generator producing levels of target difficulty |
| **Week 13 (mid-May)** | Human playtest data collected; correlation analysis complete |
| **Week 15 (end-May)** | Full evaluation written; documentation and code clean-up finished |
| **⭐ 11 June** | **Final Presentations** |

---

## 3. Phase 1 – Foundation & Midterm (up to 29 March)

### 3.1 Objectives

- Establish a playable, fully-featured version of Worm Escape.
- Implement the core automated solver.
- Validate the solver on a representative set of hand-crafted levels.
- Draft the midterm report.

### 3.2 Tasks

| # | Task | Status |
|---|------|--------|
| 1.1 | Define grid representation and level schema (JSON/dict) | ✅ Done |
| 1.2 | Implement `worm_escape` game engine (`engine.py`, `entities.py`) | ✅ Done |
| 1.3 | Implement terminal renderer with ANSI colours (`renderer.py`) | ✅ Done |
| 1.4 | Implement level manager — load & validate levels (`level_manager.py`) | ✅ Done |
| 1.5 | Build interactive level editor (`editor.py`) | ✅ Done |
| 1.6 | Implement Directed Dependency Graph solver (`solver.py`) | ✅ Done |
| 1.7 | Write initial procedural level generator (`generator.py`) | ✅ Done |
| 1.8 | Create at least 5 hand-crafted test levels | 🔄 In progress |
| 1.9 | Write unit tests for solver correctness | 🔄 In progress |
| 1.10 | Draft and submit **Midterm Report** | 📅 Due 29 March |

### 3.3 Midterm Report Contents (due 29 March)

The midterm report must include:

1. **Problem Statement** – motivation, scope, and research questions.
2. **Related Work** – survey of automated game testing, PCG, and puzzle difficulty literature.
3. **System Architecture** – description of all implemented modules with diagrams.
4. **Solver Algorithm** – detailed explanation of the Directed Dependency Graph approach.
5. **Preliminary Results** – solver output on the initial level set; any early difficulty observations.
6. **Plan for Remaining Work** – what will be done in Phase 2 and Phase 3.

---

## 4. Phase 2 – Core Research & Development (30 March – 31 May)

### 4.1 Objectives

- Define a robust, explainable difficulty metric.
- Extend the level generator to hit target difficulty bands.
- Collect and analyse human playtest data.
- Correlate automated scores with human performance.

### 4.2 Tasks

| # | Task | Deadline |
|---|------|----------|
| 2.1 | Define difficulty features (rounds to solve, blocking depth, branching factor, …) | Week 9 |
| 2.2 | Implement scoring function combining difficulty features | Week 9 |
| 2.3 | Calibrate scoring weights against a labelled level set | Week 10 |
| 2.4 | Extend generator to accept difficulty targets (easy / medium / hard) | Week 11 |
| 2.5 | Design and run human playtest session (≥ 10 participants, ≥ 3 levels each) | Week 12 |
| 2.6 | Statistical correlation: automated score vs. human move-count / time | Week 13 |
| 2.7 | Iterate on metric if correlation is weak; re-test if needed | Week 14 |
| 2.8 | Write evaluation section of final report | Week 15 |

### 4.3 Difficulty Features (planned)

| Feature | Description |
|---------|-------------|
| `num_rounds` | Number of dependency-graph resolution rounds needed |
| `max_blocking_depth` | Longest dependency chain in the blocked-by graph |
| `deadlock_count` | Number of cyclic subgraphs (0 for solvable levels) |
| `avg_branching` | Average number of extractable worms per round |
| `total_worms` | Grid population (larger → generally harder) |
| `grid_density` | Fraction of grid cells occupied by worm segments |

---

## 5. Phase 3 – Final Wrap-up & Presentations (1 June – 11 June)

### 5.1 Objectives

- Polish code, documentation, and the final report.
- Prepare and deliver the final presentation.

### 5.2 Tasks

| # | Task | Deadline |
|---|------|----------|
| 3.1 | Code review and clean-up; ensure all modules are documented | 5 June |
| 3.2 | Complete and proofread final report | 7 June |
| 3.3 | Prepare presentation slides (12–15 slides) | 8 June |
| 3.4 | Rehearse presentation (demo + Q&A preparation) | 9–10 June |
| 3.5 | **Final Presentation** | ⭐ 11 June |

### 5.3 Final Presentation Outline (11 June)

| Slide(s) | Content |
|----------|---------|
| 1 | Title, team, supervisor |
| 2 | Problem motivation & research questions |
| 3 | Worm Escape game — rules and demo |
| 4–5 | System architecture diagram |
| 6–7 | Automated solver — algorithm walkthrough |
| 8–9 | Difficulty metric — features and scoring |
| 10 | Level generator — examples by difficulty band |
| 11–12 | Evaluation results — correlation with human data |
| 13 | Limitations and future work |
| 14 | Conclusions |
| 15 | Q&A |

---

## 6. Technical Architecture

```
automated-game-testing-and-difficulty-assessment/
│
├── main.py                   # CLI entry-point (human play)
├── editor.py                 # Interactive level editor
├── generator.py              # Procedural level generator
│
└── worm_escape/
    ├── __init__.py           # Public API surface
    ├── constants.py          # ANSI codes, direction deltas, timing
    ├── entities.py           # Worm dataclass
    ├── level_manager.py      # Load & validate level dicts
    ├── engine.py             # Slide / extraction logic
    ├── renderer.py           # Terminal renderer
    ├── solver.py             # Automated solver (DDG) + difficulty report
    └── levels/               # Hand-crafted & generated level files
```

### Module Responsibilities

| Module | Responsibility |
|--------|---------------|
| `engine.py` | Stateful game logic: slide a worm one cell at a time, detect extraction |
| `solver.py` | Stateless analysis: build blocked-by graph, determine solvability, generate round-by-round report |
| `generator.py` | Procedural content generation: place worms on empty grids subject to difficulty constraints |
| `editor.py` | Interactive TUI for placing, moving, and deleting worms; calls `solver.py` to validate on save |
| `level_manager.py` | Schema validation and normalisation for level dicts |

---

## 7. Deliverables Summary

| Deliverable | Due Date | Format |
|-------------|----------|--------|
| Working game prototype (play + editor + solver) | Before midterm | Code in repository |
| **Midterm Report** | **29 March** | PDF (≥ 15 pages) |
| Difficulty scoring module | Mid-April | Code in repository |
| Enhanced level generator (targeted difficulty) | End of April | Code in repository |
| Human playtest data & analysis | Mid-May | CSV + notebook/script |
| Final report | 7 June | PDF (≥ 30 pages) |
| **Final Presentation** | **11 June** | Slides + live demo |

---

## 8. Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Difficulty metric does not correlate well with human data | Medium | High | Start with multiple feature candidates; use regression to find best weights |
| Generator produces unsolvable levels | Low | Medium | Always validate generated levels through `solver.py` before saving |
| Insufficient playtest participants | Medium | Medium | Recruit from course cohort early; supplement with crowd-sourcing if needed |
| Scope creep (adding game modes / new game types) | Medium | Low | Keep scope locked to Worm Escape; note extensions as future work |
| Terminal rendering issues on non-UNIX systems | Low | Low | ANSI fallback already implemented; test on Windows VM if needed |
