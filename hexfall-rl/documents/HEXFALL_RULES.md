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

**Walls:** Some reserve cells contain **walls** instead of buckets, generators, or empty space. A wall is a permanent obstacle: it cannot be picked, never gets removed during play, and **blocks reachability propagation through its cell** (same as an unpicked bucket or a generator). Level designers use walls to shape the reachability graph — e.g., forcing the player around a wall to reach the buckets on the other side. Walls are static for the entire level.

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

**Stochastic:**

- Hex fall direction: when a bottom-row stack clears, one of its two upper neighbors falls in. This choice is random and the only source of true randomness in gameplay.

**Hidden information (POMDP):**

- ? bucket colors are hidden until the bucket becomes reachable.
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

## 10. Open questions and TODOs

Most of the originally-flagged open questions were resolved on May 2, 2026. What remains:

- **Color count distribution across levels:** §9 notes a typical range of 5–7 colors per level based on the user's observation. Exact distribution should be empirically verified from level data once available.

All other previously-open questions (slice visibility, generator end-of-queue behavior, multi-bucket interpretation of "2"/"3" tiles, reachability through generators) have been resolved and folded into the relevant sections.

---

## 11. Revision history

- **May 2, 2026:** Initial version. 11 sections. Created in commander chat from prior conversation with user. Open questions in §10 are deliberate — they should be resolved before MDP design begins.
- **May 2, 2026 (second pass):** Resolved 4 of 5 originally-open questions in a follow-up exchange. Added Visibility subsection to §3 (slices below the top are partially visible based on stack-height and edge-column geometry — Hex Fall is a partial-observability POMDP, not zero-observability). Clarified §5 generator semantics: number = remaining bucket count, generators stay blocking after exhaustion, reachability does not propagate through generator cells. Updated §8 to reference geometric visibility rules. Pinned color count parameter to typical range 5–7 in §9. Only one empirical question remains in §10.
- **May 5, 2026:** Two clarifications flowing back from `HEXFALL_MDP_SPEC.md` commander review. §4 same-color collision rule now states explicitly that the less-full bucket waits globally, not just at the same stack. §5 generators now state that the output queue is hidden from the player (only count + facing direction visible). Both items were ambiguous in the prior version; the resolutions are confirmed game-mechanics facts, not interpretations.
- **May 5, 2026 (second pass):** Added walls to §5 as a fifth reserve cell type. Walls are permanent obstacles — never removed, never pickable, block reachability propagation. Initially missed in the prior versions of the rules doc; identified from a level 38 screenshot during planning of the level-format issue. This is a real game mechanic, not a speculative addition.
- **May 7, 2026:** Fixed reachability rule in §5. The previous wording ("reachable if at least one of its 4 neighbors is empty") was imprecise — it would mark isolated empties as reachable. Replaced with the correct graph-reachability formulation: the top edge is the source set, reachability propagates through chains of empty cells. Added a note that real Hex Fall levels never have initial empty reserve cells, so the imprecise rule and the correct rule agree on real published levels; the general form matters only for test/generated levels. Caught during commander review of the LEVEL_FORMAT.md worker deliverable, which produced an incorrect reachability walkthrough following the literal rule.
