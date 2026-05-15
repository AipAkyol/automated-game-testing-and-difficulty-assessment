# HEXFALL_RULES.md

Rules and mechanics of Hex Fall (Paxie Games), the test environment for this project. This document is the single source of truth for game behavior. Workers and Claude Code sessions should read this before designing, implementing, or analyzing anything that depends on game state.

This document describes the **game**, not the MDP. State representation, action space, and reward design live in a separate document (`HEXFALL_MDP_SPEC.md`, to be written).

---

## 1. Purpose and scope

Hex Fall is a hexagonal puzzle game where the player clears a stacked hex field by selecting colored buckets that consume slices from the bottom of the field. The challenge comes from a 5-slot buffer constraint: committing buckets that cannot be filled leads to deadlock.

This doc covers what the game *is*: zones, mechanics, win/lose conditions, what's deterministic vs. random, and the parameters that vary across levels. It does not cover UI elements (coins, power-ups, settings) since they do not affect gameplay logic.

---

## 2. Overview

Hex Fall has three zones:

- **Hex field (top):** a pile of hexagonal stacks, each composed of colored slices.
- **Buffer (middle):** 5 slots holding active buckets that consume slices from the field.
- **Bucket reserve (bottom):** a static grid of unselected buckets the player picks from.

The player picks reachable buckets from the reserve. Buckets enter the buffer and automatically consume slices from the bottom row of the hex field. When a bucket fills, it leaves the buffer. The level ends when the field is empty (win) or when the buffer is locked (lose).

---

## 3. The hex field

The hex field is a pile of **hexagonal stacks**. Each stack is composed of **slices** in various colors, stacked vertically. Only the top slice is currently *consumable*, but visibility of slices below the top depends on geometry (see Visibility below).

**Geometry:** Stacks are arranged in a hexagonal grid. Because of hex geometry, every stack has up to **two upper neighbors** and up to two lower neighbors, plus side neighbors.

**Consumption:** Only the **bottom row** of the field is consumable, and only the **top slice** of each bottom-row stack. Slices below the top slice of a bottom-row stack are not directly accessible — they are exposed as the top slice is consumed.

**Fall mechanic:** When a bottom-row stack is fully cleared (all slices consumed), one of its **two upper neighbors** falls down into the empty position. The choice between the two upper neighbors is **random**. This is the only source of randomness in the field itself.

**Visibility:** Because the field is rendered in 3D, slices below the top of a stack may be partially visible depending on geometry:

- **Bottom row:** all slices in a bottom-row stack are fully visible.
- **Stacks taller than their lower neighbor:** the portion sticking up above the lower neighbor is visible.
- **Edge columns of alternating rows:** stacks in the first and last columns of certain rows are fully visible from the side.
- **Otherwise:** only the top slice is visible.

These rules are derivable from stack heights and grid position, so the simulator computes visibility automatically. The implication for RL is that the agent has *partial* observability of stack composition, not zero — visibility is a function of geometry, not random.

**No move limit, no timer.** The player has unlimited time and unlimited "moves" in the sense of pick actions. The only constraint is buffer capacity and the reserve being finite.

---

## 4. The buffer

The buffer has **exactly 5 slots**. A slot holds one bucket.

**Slot ordering does not matter.** Slices flow into whichever bucket can accept them, regardless of slot position.

**Bucket capacity:** Each bucket holds a fixed number of slices (parameter, default **25**) before filling. When full, the bucket leaves the buffer and frees its slot.

**Consumption rule:** Each bucket pulls slices from the bottom row of the field whose color matches the bucket. A bucket can only consume the *top slice* of a bottom-row stack. If the top slice doesn't match any bucket in the buffer, no consumption happens at that position until the buffer changes.

**Multiple buckets pulling simultaneously:** Buckets pull whenever a matching slice is available. If multiple buckets in the buffer have matching slices exposed at different bottom-row stacks, they all pull on the same tick.

**Same-color collision rule:** If two buckets of the same color are in the buffer at once, the **fuller one fills first**. The other bucket only starts receiving slices once the fuller one is full and leaves. This applies globally — the less-full bucket does not pull from any bottom-row stack while the fuller bucket is in the buffer, even if a matching slice is exposed at a different stack.

**No manual removal:** A bucket cannot be removed from the buffer manually. Once committed, it stays until it fills.

---

## 5. The bucket reserve

The reserve is a **static, finite grid** of buckets arranged below the hex field. The reserve does not regenerate during a level — when the player picks a bucket, that position becomes empty for the rest of the level.

**Adjacency:** The reserve uses **4-adjacency** (up, down, left, right) despite buckets being hex-shaped visually. Buckets are arranged in a square grid layout.

**Reachability rules:**

Reachability is computed as **graph connectivity from the top edge** of the reserve grid. Specifically:

- The **top row of the reserve is always the source set** — its cells are reachable from the start (provided they contain something pickable; walls and generators are never picked even if reachable).
- A cell is **reachable** if it can be connected to the top edge by a path of adjacent **empty cells**, plus a final step to the cell itself.
- Reachability **propagates through emptied cells**: when a bucket is picked, its cell becomes empty, which may make neighboring cells reachable that previously were not.
- **A cell with empty neighbors is not automatically reachable.** The empty neighbors must themselves trace back through more empties to the top edge. Isolated emptiness does not confer reachability.

**Note on real Hex Fall levels:** Published Hex Fall levels do not contain initial empty cells in the reserve — every reserve cell at level start contains a bucket, generator, or wall. Under that constraint, the literal rule "reachable if at least one neighbor is empty" gives the same result as the graph-reachability rule, because empties only arise by picking, and picks always start from the top edge. The rule is stated in its general form here to support hand-built test levels and procedurally generated levels that may include initial empties, where the two readings diverge.

**Generators:** Some positions in the reserve hold **bucket generators** instead of plain buckets. A generator is identified by a number on the tile, indicating **how many buckets it has remaining to produce**.

- Generators have a **fixed orientation** (face a direction).
- When the cell in the generator's facing direction is freed, the generator produces a bucket into that cell, and its number decrements by 1.
- The generator **stays in place** after producing — it does not move or get consumed.
- When a generator's count reaches 0, it stops producing but remains in place. The cell is still considered occupied for adjacency/reachability purposes.
- Generators **block reachability** through their own cell, the same way an unpicked bucket does. Reachability does not propagate through a generator-occupied cell, even after the generator is exhausted.
- Generator output is **deterministic**: same color sequence on every replay of the same level.
- Generator output **queue is hidden** from the player. Only the remaining count and facing direction are visible; the colors of upcoming buckets are revealed as each bucket is produced.

**? buckets:** Some buckets have unknown color until reached. The color is **deterministic** (same color on every replay of the same level), but is hidden from the player until the bucket becomes reachable. Reaching here means becoming pickable, not necessarily picked.

**Ice buckets:** Some buckets start the level encased in ice and are unusable until they thaw. Each ice bucket has a fixed **thaw threshold** measured in player moves (the level data calls this `iceCapacity`). The level maintains a **move counter** that starts at 0 and increments by 1 each time the player picks a bucket from the reserve. When the move counter reaches an ice bucket's thaw threshold, the bucket thaws immediately and becomes a normal bucket with its (previously hidden) color revealed.

- Before thawing, an ice bucket is **not pickable** — it cannot be placed into the buffer regardless of reachability.
- Before thawing, an ice bucket **blocks reachability propagation through its cell**, same as an unpicked bucket, generator, or wall.
- The ice bucket's color is hidden from the player until it thaws. Like ?-buckets, this color is deterministic per level — same color on every replay.
- After thawing, the ice bucket behaves identically to a normal bucket: it is pickable when reachable, has the standard capacity (default 25), and contributes to slice-bucket parity exactly like any other bucket.
- Thawing is **deterministic** given the move counter: the same sequence of player moves yields the same set of thawed ice buckets at any point in the trajectory.

Ice buckets contribute to slice-bucket parity from level start — their capacity is counted even while they are still frozen. A level with ice buckets is therefore *unwinnable until the player makes enough moves to thaw the buckets whose colors are needed for the field's remaining slices*.

**Walls:** Some reserve cells contain **walls** instead of buckets, generators, or empty space. A wall is a permanent obstacle: it cannot be picked, never gets removed during play, and **blocks reachability propagation through its cell** (same as an unpicked bucket or a generator). Level designers use walls to shape the reachability graph — e.g., forcing the player around a wall to reach the buckets on the other side. Walls are static for the entire level.

**Pin blockers:** Pins are ray-shaped barriers placed on the reserve grid. Each pin is defined by an **origin cell** `(x, y)`, a **direction** (`Up`, `Down`, `Left`, or `Right`), and an optional **block count** `b ≥ 0`.

- The pin occupies a **ray of cells** starting at the origin and extending in the pin's direction:
  - If `b ≥ 1`, the ray spans **`b + 1` cells**: the origin plus `b` additional cells in the direction. (Example: a pin at `(5, 3)` with direction `Right` and `blockCount: 2` occupies cells `(5, 3)`, `(6, 3)`, `(7, 3)`.)
  - If `b = 0` or the `blockCount` field is absent, the ray extends from the origin all the way to the **grid edge** in the pin's direction.
- The ray **passes through walls**: a pin's ray can occupy the same cell as a wall (`deadCell`); the wall and the pin are stacked, both present until the pin is destroyed.
- Each cell in the pin's ray **cannot hold a bucket and cannot be picked**. The cell is functionally a wall for the pin's lifetime.
- Crucially, the pin **blocks reachability propagation along its ray**: a path of adjacency that would otherwise propagate from one side of the ray to the other is blocked. Phrased differently — reachability does not cross the pin's ray, on either of the two perpendicular sides. This is the gameplay purpose of pins: they partition the reserve into regions that are temporarily unreachable from the top edge.
- A pin is **destroyed** when the bucket in the cell **directly behind the origin** is cleared. "Behind" means one step in the direction *opposite* the pin's facing direction. (Example: a pin at `(5, 3)` facing `Right` is destroyed when the bucket at `(4, 3)` is cleared. A pin at `(5, 3)` facing `Down` is destroyed when the bucket at `(5, 2)` is cleared. Coordinate semantics follow the level data convention where `y` increases downward.)
- When destroyed, **all cells of the ray become empty**, just as if each had been a bucket that was picked. Reachability is recomputed; previously unreachable regions on the far side of the ray may now become reachable.
- The destruction cell itself (behind the origin) must hold a normal bucket — pins cannot be destroyed by clearing generators, walls, or other pins. If the cell behind the origin is empty, contains a wall, or is otherwise non-bucket, the pin is **permanent for the level** unless the cell later receives a generator-produced bucket that is then cleared.

Pins are static once placed: their origin, direction, and block count do not change during play. Destruction is the only state transition.

**Slice-bucket parity:** Levels are constructed such that the total slice count in the field equals the total slice capacity across all buckets in the reserve. A level is winnable in principle iff the player picks buckets in an order that respects buffer capacity throughout.

---

## 6. Game loop

The game alternates between **player actions** and **automatic state updates**.

**Player action:** The player picks one reachable bucket from the reserve. The bucket moves to an open buffer slot. If the buffer has no open slot, the action is unavailable.

**Automatic updates** (happen continuously between actions, but conceptually per tick):

1. Each bucket in the buffer attempts to pull a matching top slice from the bottom row of the field.
2. When a bucket fills, it leaves the buffer.
3. When a hex stack is cleared, one of its two upper neighbors falls into the empty position.
4. When a generator's facing cell is freed, the generator produces its next bucket into that cell.

The game waits for the next player action when no further automatic updates can fire.

---

## 7. Win and lose conditions

**Win:** The hex field is empty. By slice-bucket parity, this also means all reserve buckets have been consumed and the buffer is empty.

**Lose (deadlock):** All 5 buffer slots are occupied AND none of them can pull from the current bottom row AND no bucket in the reserve can be picked to displace the situation. There is no explicit "buffer overflow" in the sense of slices spilling — the lose state is structural: progress becomes impossible.

There is no time limit and no move limit. The player can think indefinitely.

---

## 8. Determinism and randomness

This section is critical for RL design. It enumerates exactly what's deterministic, what's stochastic, and what's hidden from the player.

**Deterministic across replays:**

- Hex field initial layout and slice composition
- Bucket reserve initial layout
- Generator positions, orientations, and output sequences
- ? bucket colors (hidden but fixed per-level)
- Ice bucket thaw thresholds and colors (hidden but fixed per-level; thaw timing is a deterministic function of the player's move sequence)
- Pin positions, directions, block counts, and destruction conditions

**Stochastic:**

- Hex fall direction: when a bottom-row stack clears, one of its two upper neighbors falls in. This choice is random and the only source of true randomness in gameplay.

**Hidden information (POMDP):**

- ? bucket colors are hidden until the bucket becomes reachable.
- Ice bucket colors are hidden until the bucket thaws. The thaw threshold (number of moves required) is visible to the player at all times.
- Slices in the hex field are *partially* visible based on the geometric rules in §3 (Visibility). Bottom-row slices are fully visible; otherwise visibility depends on stack height and grid position.

**Implication:** Hex Fall is a partially observable Markov decision process (POMDP) with a single source of stochasticity (the fall direction). An RL agent must reason under uncertainty about both hidden bucket colors and hidden slice compositions, plus stochastic transitions in the field.

---

## 9. Difficulty levers (informational)

These are hypotheses about what makes a level hard, not authoritative rules. They will be validated against agent performance once the simulator and RL pipeline exist. They are listed here because the CTO's framing ("level curve & strategy optimization") treats these as the parameters a level generator would tune.

**Level parameters:**

- **Bucket capacity** — default 25 slices per bucket. Higher capacity = more slices buffered = more error tolerance.
- **Color count per level** — number of distinct slice/bucket colors. More colors → higher chance of committing the buffer to a color that won't appear at the bottom soon. Typical range observed: **5–7 colors per level**. Treated as a parameter; exact distribution to be confirmed from level data.
- **Field dimensions** — number and arrangement of hex stacks.
- **Reserve dimensions** — size of the bucket grid and topology.

**Hypothesized difficulty drivers (ranked roughly by impact, unverified):**

1. **Buffer pressure** — how often the player is forced to commit buckets without knowing what colors will be exposed next. Driven by color count and stack composition.
2. **Intra-stack color mixing** — stacks with multiple colors interleaved create timing puzzles (a green slice buried under reds means a green bucket sits idle).
3. **Spatial coupling between reserve and field** — if reachable buckets don't match colors about to be exposed, the player is forced into bad commits.
4. **Hidden information density** — how many ? buckets exist and where they sit in the reserve graph.
5. **Generator placement** — generators that face into critical paths create timing pressure.
6. **Field stochasticity exposure** — levels where the random fall direction frequently determines win/loss are variance-driven, not skill-driven.

---

## 10. Color palette

Real Hex Fall levels use a fixed palette of color codes. Two encodings appear in the level data:

**Short codes** (used in 95 of 100 levels in the validation dataset):

`b`, `br`, `db`, `dg`, `dgr`, `do`, `dr`, `f`, `g`, `gr`, `o`, `og`, `p`, `pk`, `r`, `w`, `y`

The semantic meaning of each code is not documented by Paxie and is not required by the simulator — the simulator treats colors as opaque tokens. Pairwise distinctness is the only property the simulator cares about. Approximate human-readable interpretations (for debugging only):

| Code | Likely name | Code | Likely name |
|------|-------------|------|-------------|
| `r` | red | `dr` | dark red |
| `g` | green | `dg` | dark green |
| `b` | blue | `db` | dark blue |
| `y` | yellow | `dgr` | dark gray |
| `o` | orange | `do` | dark orange |
| `og` | olive green | `gr` | gray |
| `p` | purple | `pk` | pink |
| `w` | white | `br` | brown |
| `f` | (filler / sand?) | | |

**Full-name encoding** (used in 5 levels — 81, 82, 83, 85, 86):

A subset of levels uses full-word color tokens instead of short codes. These appear in stacks, collectors, and tunnel queues throughout those 5 levels (no mixed encoding within a single level). The encoding is per-level, not per-field. Loader normalizes full names to short codes at parse time using the following mapping:

| Full name | Short code |
|-----------|------------|
| `Yellow` | `y` |
| `Blue` | `b` |
| `Red` | `r` |
| `Green` | `g` |
| `Purple` | `p` |
| `Pink` | `pk` |
| `Orange` | `o` |
| `White` | `w` |
| `DarkBlue` | `db` |
| `DarkRed` | `dr` |
| `DarkGray` | `dgr` |
| `OliveGreen` | `og` |

Internally, the simulator and all downstream code work exclusively in short codes. The full-name encoding is an artifact of a different editor session/build used for levels 81–86 and is not preserved past the loader boundary.

**Per-level color count:** Levels in the validation dataset have 2–7 distinct colors, with median 5. Two levels (87 and 91) declare a lower `editorMeta.colorCount` than the number of distinct colors actually present in the level data — this is treated as a non-blocking warning at load time, not an error, since the actual color set is the authoritative source.

---

## 11. Open questions and TODOs

Most of the originally-flagged open questions were resolved on May 2, 2026. What remains:

- **Color count distribution across levels:** §9 notes a typical range of 5–7 colors per level based on the user's observation. Exact distribution should be empirically verified from level data once available.

All other previously-open questions (slice visibility, generator end-of-queue behavior, multi-bucket interpretation of "2"/"3" tiles, reachability through generators) have been resolved and folded into the relevant sections.

---

## 12. Revision history

- **May 2, 2026:** Initial version. 11 sections. Created in commander chat from prior conversation with user. Open questions in §10 are deliberate — they should be resolved before MDP design begins.
- **May 2, 2026 (second pass):** Resolved 4 of 5 originally-open questions in a follow-up exchange. Added Visibility subsection to §3 (slices below the top are partially visible based on stack-height and edge-column geometry — Hex Fall is a partial-observability POMDP, not zero-observability). Clarified §5 generator semantics: number = remaining bucket count, generators stay blocking after exhaustion, reachability does not propagate through generator cells. Updated §8 to reference geometric visibility rules. Pinned color count parameter to typical range 5–7 in §9. Only one empirical question remains in §10.
- **May 5, 2026:** Two clarifications flowing back from `HEXFALL_MDP_SPEC.md` commander review. §4 same-color collision rule now states explicitly that the less-full bucket waits globally, not just at the same stack. §5 generators now state that the output queue is hidden from the player (only count + facing direction visible). Both items were ambiguous in the prior version; the resolutions are confirmed game-mechanics facts, not interpretations.
- **May 5, 2026 (second pass):** Added walls to §5 as a fifth reserve cell type. Walls are permanent obstacles — never removed, never pickable, block reachability propagation. Initially missed in the prior versions of the rules doc; identified from a level 38 screenshot during planning of the level-format issue. This is a real game mechanic, not a speculative addition.
- **May 7, 2026:** Fixed reachability rule in §5. The previous wording ("reachable if at least one of its 4 neighbors is empty") was imprecise — it would mark isolated empties as reachable. Replaced with the correct graph-reachability formulation: the top edge is the source set, reachability propagates through chains of empty cells. Added a note that real Hex Fall levels never have initial empty reserve cells, so the imprecise rule and the correct rule agree on real published levels; the general form matters only for test/generated levels. Caught during commander review of the LEVEL_FORMAT.md worker deliverable, which produced an incorrect reachability walkthrough following the literal rule.
- **May 13, 2026:** Major rules expansion driven by survey of Anıl Özmen's 100-level Paxie dataset (`survey_report.md`, May 13). Two new reserve cell types added to §5: **ice buckets** (frozen buckets that thaw after N player moves, present in 38 levels starting at level 39) and **pin blockers** (ray-shaped barriers destroyed by clearing the bucket behind their origin, present in 22 levels starting at level 64). §8 (determinism) extended to cover these mechanics — both are deterministic given the player's move sequence and the level definition. New §10 (Color palette) added enumerating the 17 short-code colors observed in real levels plus the full-name normalization mapping for 5 outlier levels (81-86) authored with a different editor build. Renumbered prior §10 (Open questions) → §11 and §11 (Revision history) → §12. The following Paxie mechanics are deliberately **not modeled**: `mysteryCollectors` (empty in all 100 levels), `tiedPairs` (empty in all 100 levels), `keyLocks` (present only in level 100), and `hexStackArea.tunnels` (empty in all 100 levels) — see `DECISIONS.md` May 13 entry for the rationale.