# LEVEL_FORMAT.md

Specification of the JSON file format for Hex Fall levels. This document is the authoritative contract between the level loader (`hexfall/level_loader.py`), hand-built test levels, transcribed real Hex Fall levels, and the future level generator.

Game mechanics live in `HEXFALL_RULES.md`. The POMDP state structure that this format maps onto lives in `HEXFALL_MDP_SPEC.md` §3. This document covers only the file format.

---

## 1. Overview

Each level is a single JSON file. The top-level object has four sections:

```
{
  "meta":    { ... },   // level identity and global parameters
  "field":   { ... },   // hex field: grid topology and stack contents
  "buffer":  { ... },   // buffer: slot count and bucket capacity
  "reserve": { ... }    // reserve: grid dimensions and cell contents
}
```

Every field described in this spec is **required** unless explicitly marked *optional*. Validators should reject files with missing required fields or unrecognized cell types.

---

## 2. Coordinate systems

### 2.1 Hex field: odd-r offset coordinates

The hex field uses **odd-r offset coordinates**: a standard human-readable system for hex grids laid out in rows.

- Columns are indexed left-to-right starting at 0.
- Rows are indexed top-to-bottom starting at 0 (row 0 is the topmost row of the field).
- In odd-numbered rows (row 1, 3, 5, …), the visual display shifts the stacks half a hex to the right compared to even-numbered rows. This is a display convention; the coordinate system itself is simply (col, row) integer pairs.

The simulator may convert to axial coordinates internally for neighbor lookups. The conversion from odd-r offset `(col, row)` to axial `(q, r)` is:

```
q = col - (row - (row & 1)) / 2
r = row
```

**Why odd-r?** It is the most common human-readable offset system and maps naturally to how Hex Fall levels are visually described (rows of stacks, with alternating rows shifted). Hand-built levels are easier to author in this system.

### 2.2 Reserve: row-major square grid

The reserve is a square grid with 4-adjacency. Cells are addressed by `[row, col]` integer pairs, with `[0, 0]` at the top-left. Rows index top-to-bottom; columns index left-to-right. The reserve grid dimensions are declared in `reserve.rows` and `reserve.cols`.

---

## 3. Top-level structure

```json
{
  "meta":    <MetaObject>,
  "field":   <FieldObject>,
  "buffer":  <BufferObject>,
  "reserve": <ReserveObject>
}
```

No additional top-level keys are permitted. Validators should reject unknown keys at the top level.

---

## 4. Meta object

```json
"meta": {
  "id":               <string>,
  "name":             <string>,
  "version":          <integer>,
  "color_count":      <integer>,
  "notes":            <string>   // optional
}
```

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique level identifier. Use lowercase kebab-case (e.g., `"tiny-solvable"`, `"level-038"`). Must be unique across all level files in the project. |
| `name` | string | Human-readable display name (e.g., `"Level 38"`). Not required to be unique. |
| `version` | integer | Format version for this file. Current version: **1**. Increment when the format changes in a backward-incompatible way. |
| `color_count` | integer | Number of distinct colors used in this level (slices + buckets). Must match the actual count of distinct color strings appearing in `field` stacks and `reserve` bucket cells. Checked by validator. |
| `notes` | string | *Optional.* Freeform notes for human readers (e.g., design intent, known quirks, transcription source). Not read by the simulator. |

---

## 5. Field object

```json
"field": {
  "stacks": [
    {
      "col":    <integer>,
      "row":    <integer>,
      "slices": [<string>, ...]
    },
    ...
  ]
}
```

`field.stacks` is a list of stack objects. Every stack that contains at least one slice must appear in this list. Empty positions (no stack) are simply absent.

| Field | Type | Description |
|-------|------|-------------|
| `col` | integer | Column index (odd-r offset, 0-based, left-to-right). |
| `row` | integer | Row index (odd-r offset, 0-based, top-to-bottom). |
| `slices` | array of strings | Colors of the slices in this stack, ordered **top-to-bottom** (index 0 = top slice = next to be consumed). Must be non-empty; a stack with zero slices must not appear in the list. Each string must be a color name appearing in `meta.color_count`'s implied set. |

**Bottom row:** The bottom row of the field is the highest row index present among all stacks. Stacks at this row are the ones actively consumed by the buffer. Stacks at other rows fall downward when a bottom-row neighbor clears (per `HEXFALL_RULES.md` §3).

**Slice order rationale:** Top-to-bottom matches human reading order (the top slice is what you see first) and matches consumption order (the simulator pops from index 0).

---

## 6. Buffer object

```json
"buffer": {
  "slots":           <integer>,
  "bucket_capacity": <integer>
}
```

| Field | Type | Description |
|-------|------|-------------|
| `slots` | integer | Number of buffer slots. Must be 5 for any level faithful to `HEXFALL_RULES.md` §4. Parameterized here rather than hardcoded so the simulator can validate and so future experimental levels can vary it. |
| `bucket_capacity` | integer | Number of slices required to fill one bucket. Default per `HEXFALL_RULES.md` §4 is **25**. Applies uniformly to all buckets in this level. |

---

## 7. Reserve object

```json
"reserve": {
  "rows": <integer>,
  "cols": <integer>,
  "cells": [
    {
      "row":  <integer>,
      "col":  <integer>,
      "type": <string>,
      ... type-specific fields ...
    },
    ...
  ]
}
```

| Field | Type | Description |
|-------|------|-------------|
| `rows` | integer | Number of rows in the reserve grid (height). |
| `cols` | integer | Number of columns in the reserve grid (width). |
| `cells` | array | List of non-empty cell objects. Cells not listed are implicitly empty. |

### 7.1 Cell types

The `type` field takes one of five string values. Each type has its own additional fields, described below.

#### `"plain_bucket"`

A bucket with a known color.

```json
{ "row": 0, "col": 2, "type": "plain_bucket", "color": "red" }
```

| Field | Type | Description |
|-------|------|-------------|
| `color` | string | Bucket color. Must be one of the colors contributing to `meta.color_count`. |

#### `"question_bucket"`

A bucket whose color is hidden from the player until it becomes reachable (per `HEXFALL_RULES.md` §5 and `HEXFALL_MDP_SPEC.md` §4.3–4.4). The color is deterministic and stored in the level file; the simulator withholds it from the agent observation until the reveal condition fires.

```json
{ "row": 1, "col": 0, "type": "question_bucket", "color": "blue" }
```

| Field | Type | Description |
|-------|------|-------------|
| `color` | string | True color of the bucket, stored in the level file for simulator use. Not surfaced in agent observations until revealed. Must be one of the level's colors. |

#### `"generator"`

A bucket generator (per `HEXFALL_RULES.md` §5). Stays in place after producing; blocks reachability through its cell at all times, including after exhaustion.

```json
{
  "row": 2, "col": 1,
  "type":      "generator",
  "facing":    "right",
  "remaining": 3,
  "queue":     ["green", "red", "green"]
}
```

| Field | Type | Description |
|-------|------|-------------|
| `facing` | string | Direction the generator faces. One of `"up"`, `"down"`, `"left"`, `"right"`. Determines which adjacent cell receives produced buckets. |
| `remaining` | integer | Number of buckets this generator will still produce. Must equal `queue.length`. Validated by schema. |
| `queue` | array of strings | Colors of buckets to be produced, in order (index 0 = next to be produced). All strings must be level colors. The queue is hidden from the agent (per `HEXFALL_MDP_SPEC.md` §4.3); only `remaining` and `facing` are observable. |

**Invariant:** `remaining == queue.length` at level load. The simulator enforces this. After each production, `remaining` decrements and the front of the queue is consumed.

#### `"wall"`

A permanent obstacle. Never removed, never picked, blocks reachability propagation through its cell (per `HEXFALL_RULES.md` §5).

```json
{ "row": 0, "col": 3, "type": "wall" }
```

No additional fields beyond `row`, `col`, and `type`.

#### `"empty"` (implicit)

Cells not listed in `reserve.cells` are implicitly empty. There is no need to list empty cells explicitly. The `"empty"` string must not appear as a `type` value in the cells array — validators should reject it.

**Why implicit empties?** Reserve grids are sparse in practice (many empty cells between buckets). Listing only occupied cells keeps files compact and avoids encoding redundant no-ops.

### 7.2 Reachability at level load

Per `HEXFALL_RULES.md` §5: cells in the **top row** of the reserve (`row == 0`) are reachable from the start, provided they contain a plain bucket or question bucket (not a wall or generator). Empty cells in the top row act as gaps through which reachability propagates to row 1 neighbors below them, per the 4-adjacency rules.

The simulator computes reachability from scratch at load time; no reachability state is stored in the level file.

---

## 8. Color names

Color values are arbitrary strings (e.g., `"red"`, `"blue"`, `"green"`, `"yellow"`, `"purple"`, `"orange"`). The level format imposes no fixed color palette. The `meta.color_count` field must match the count of distinct color strings actually appearing across all stack slices and bucket cells (plain, question, generator queue).

**Convention:** Use lowercase English color names. Transcribed real Hex Fall levels should use names matching the game's visual colors as closely as possible.

---

## 9. Validation rules (summary)

These are the constraints a validator must check. The JSON schema (`level.schema.json`) encodes them mechanically; this section is the human-readable equivalent.

1. All four top-level sections (`meta`, `field`, `buffer`, `reserve`) are present.
2. `meta.version == 1`.
3. `meta.color_count` matches the count of distinct color strings in the level.
4. Every `slices` array is non-empty.
5. No two stacks share the same `(col, row)`.
6. No two reserve cells share the same `(row, col)`.
7. All `(col, row)` values in `field.stacks` are non-negative integers.
8. All `(row, col)` values in `reserve.cells` are within `[0, reserve.rows)` × `[0, reserve.cols)`.
9. `buffer.slots >= 1`; `buffer.bucket_capacity >= 1`.
10. `generator.remaining == generator.queue.length`.
11. `"empty"` does not appear as a cell type in `reserve.cells`.
12. All color strings referenced anywhere in the level appear in the implied color set (i.e., all colors are consistent with `meta.color_count`).
13. No unknown keys at the top level or within typed cell objects.

**Slice-bucket parity** (sum of all slice counts == `buffer.bucket_capacity * total_bucket_count`) is a recommended warning, not a hard error, because procedurally generated levels under construction may temporarily violate it.

---

## 10. Worked example: `demo-all-types.json`

This example demonstrates all five reserve cell types and a small hex field. It is not intended to be a playable or balanced level — it exists to exercise the parser.

```json
{
  "meta": {
    "id":           "demo-all-types",
    "name":         "Demo: All Cell Types",
    "version":      1,
    "color_count":  3,
    "notes":        "Illustrative example only. Shows plain bucket, question bucket, generator, wall, and implicit empty cells. Not a balanced level."
  },

  "field": {
    "stacks": [
      { "col": 0, "row": 0, "slices": ["red", "blue", "green"] },
      { "col": 1, "row": 0, "slices": ["blue", "red"] },
      { "col": 0, "row": 1, "slices": ["green", "red", "blue", "green"] },
      { "col": 1, "row": 1, "slices": ["red"] }
    ]
  },

  "buffer": {
    "slots":           5,
    "bucket_capacity": 25
  },

  "reserve": {
    "rows": 3,
    "cols": 4,
    "cells": [
      { "row": 0, "col": 0, "type": "plain_bucket",    "color": "red"   },
      { "row": 0, "col": 1, "type": "plain_bucket",    "color": "blue"  },
      { "row": 0, "col": 2, "type": "question_bucket", "color": "green" },
      { "row": 0, "col": 3, "type": "wall"                              },
      { "row": 1, "col": 0, "type": "plain_bucket",    "color": "green" },
      {
        "row": 1, "col": 2,
        "type":      "generator",
        "facing":    "left",
        "remaining": 2,
        "queue":     ["red", "blue"]
      },
      { "row": 2, "col": 1, "type": "plain_bucket",    "color": "red"   }
    ]
  }
}
```

**Reserve layout (3 × 4 grid, `.` = implicit empty):**

```
col:    0            1            2               3
row 0:  plain(red)   plain(blue)  question(green) WALL
row 1:  plain(green) .            generator→left  .
row 2:  .            plain(red)   .               .
```

**Reachability at load:**

Per `HEXFALL_RULES.md` §5, reachability is graph connectivity from the top edge through empty cells. A cell is reachable if it is in the top row (and pickable), or if it has at least one empty neighbor that traces back to the top edge through more empties.

- `[0,0]`, `[0,1]`, `[0,2]` are in the top row and contain pickable cells (plain bucket, plain bucket, question bucket). All three are **reachable** from the start.
- `[0,3]` is in the top row but is a wall. Walls are never pickable, so although the cell is "reachable" in the graph sense, no action picks it.
- `[1,0]` (plain green): its top-row neighbor `[0,0]` is a bucket, not empty. So `[1,0]` is **not reachable at load**. It becomes reachable only after `[0,0]` is picked (which empties that cell, connecting `[1,0]` to the top edge).
- `[1,2]` (generator): generators are never picked. The cell blocks propagation through itself at all times, including after exhaustion.
- `[2,1]` (plain red): its 4 neighbors are `[1,1]` (empty), `[2,0]` (empty), `[2,2]` (empty), and out-of-grid (bottom edge). It has empty neighbors, but none of those empties is connected to the top edge through more empties — for example, `[1,1]` is surrounded by `[0,1]` (bucket), `[1,0]` (bucket), `[1,2]` (generator), so `[1,1]` is not connected to the top through empties. `[2,1]` is therefore **not reachable at load**. It becomes reachable only after the bucket chain leading to `[1,1]` (or one of its other empty neighbors) is opened up.

In summary, the only reachable cells at load are the three in the top row: `[0,0]`, `[0,1]`, `[0,2]`. Reachability expands as buckets are picked.

> **Note:** This example uses initial empty cells (`[1,1]`, `[2,0]`, `[2,2]`, `[2,3]`) to exercise the reachability rule's general form. Real Hex Fall levels do not contain initial empties — every reserve cell at level start contains a bucket, generator, or wall. The example is constructed to test the reachability semantics, not to represent a realistic Hex Fall level.

**Generator behavior on load:** Per `HEXFALL_MDP_SPEC.md` §5.4 (May 7, 2026 revision), level loading runs the same automatic-update loop as a normal transition. The generator at `[1,2]` faces left to `[1,1]`, which is initially empty. On the first update tick at load, the generator fires: it produces `"red"` into `[1,1]`, decrements `remaining` to 1, and advances the queue to `["blue"]`. The simulator then checks for further updates; with no other update conditions met, the state becomes quiescent and the agent receives the first observation.

After this load-time firing, the reserve state is:
- `[1,1]` now contains a plain red bucket (produced by the generator).
- `[2,0]`, `[2,2]`, `[2,3]` remain empty.
- All other cells unchanged from the file definition.

Reachability on this post-load state still gives only the top row as reachable, since `[1,1]` is now a bucket (not empty) and the surrounding empties don't connect to the top edge.

---

## 11. Relation to MDP spec §3

For reference, the mapping between `HEXFALL_MDP_SPEC.md` §3 sub-sections and this file format:

| MDP spec §3 component | Level file location |
|-----------------------|---------------------|
| §3.1 Hex field — grid topology | `field.stacks[*].col`, `.row` |
| §3.1 Hex field — stack contents | `field.stacks[*].slices` |
| §3.2 Buffer — slot count | `buffer.slots` |
| §3.2 Buffer — bucket capacity | `buffer.bucket_capacity` |
| §3.3 Reserve — grid topology | `reserve.rows`, `reserve.cols` |
| §3.3 Reserve — plain bucket | cell `type: "plain_bucket"`, `.color` |
| §3.3 Reserve — ?-bucket color + revealed flag | cell `type: "question_bucket"`, `.color`; `revealed` flag initialized to `false` at load |
| §3.3 Reserve — generator facing, count, queue | cell `type: "generator"`, `.facing`, `.remaining`, `.queue` |
| §3.3 Reserve — wall | cell `type: "wall"` |
| §3.4 RNG state | **Not in level file.** Provided at runtime (seed passed to simulator). |
| §3.5 Quiescence flag | **Not in level file.** Runtime simulator state. |

The level file encodes all *deterministic* initial state. Runtime state (RNG seed, quiescence, current fill counts, revealed flags) is initialized by the simulator at load time from the level data plus any externally supplied seed.

---

## 12. Revision history

- **May 7, 2026:** Initial version. Produced in worker chat for Week 1 issue (LEVEL_FORMAT.md). Format covers all five reserve cell types, odd-r offset hex coordinates, and all MDP spec §3 fields. One open question flagged: generator-on-load firing behavior (§10).
- **May 7, 2026 (commander review):** Fixed §10 worked example reachability walkthrough. The original walkthrough followed a literal reading of `HEXFALL_RULES.md` §5 ("reachable if at least one neighbor is empty") which is imprecise — it would mark isolated empties as reachable. Corrected to follow proper top-edge-rooted graph reachability. Rules doc §5 was updated in parallel to clarify the rule. Example now correctly shows only the top row as reachable at load, with `[1,0]` and `[2,1]` not reachable until upstream buckets are picked. Added a clarifying note that the example uses initial empties to exercise the general rule, even though real Hex Fall levels never contain initial empties. Also resolved the generator-on-load question: load-time runs the same automatic-update loop, so the generator at `[1,2]` fires on load, populating `[1,1]`. MDP spec §5.4 was updated in parallel with an "Initial load" paragraph making this explicit.
