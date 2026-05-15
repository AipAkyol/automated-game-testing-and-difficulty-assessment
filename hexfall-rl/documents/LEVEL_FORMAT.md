# LEVEL_FORMAT.md

Specification of the JSON file format for Hex Fall levels. This document is the authoritative contract between the level loader (`hexfall/level_loader.py`), hand-built test levels, the level files provided by Paxie Games, and the future level generator.

Game mechanics live in `HEXFALL_RULES.md`. The POMDP state structure that this format maps onto lives in `HEXFALL_MDP_SPEC.md` §3. This document covers only the file format.

This document was rewritten on May 13, 2026 following the decision to adopt Paxie's native level format directly, rather than maintaining a separate internal format and translator (see `DECISIONS.md` May 13 entry, and the survey of the 100-level Paxie dataset in `CLASSIFIED.paxie_data/survey_report.md`). The previous internal format (top-level `meta`/`field`/`buffer`/`reserve` with snake_case keys) is fully superseded.

---

## 1. Overview

Each level is a single JSON file conforming to `level_schema.json`. The top-level object has five required fields:
{
"levelNumber":      <integer>,         // 1-based level index
"levelVersionCode": <integer>,         // format version, currently 1
"collectorArea":    { ... },           // bucket reserve: grid + cell contents
"hexStackArea":     { ... },           // hex field: grid + stack contents
"editorMeta":       { ... }            // generator parameters (informational)
}

The format uses **Paxie's editor terminology** in the JSON keys (`collectorArea`, `singleBlockCollectors`, `deadCells`, etc.) because the file format is what Paxie produces. The simulator's internal vocabulary is different (bucket reserve, plain bucket, wall, etc.) — see §2 below for the mapping. The loader translates field names and color tokens at the file boundary; nothing internal to the simulator uses Paxie's terminology.

Every field described as required must be present. Validators reject files with missing required fields or unrecognized top-level keys.

---

## 2. Vocabulary glossary

Paxie's JSON keys map to our internal terminology as follows. Use the internal terminology in code, in docs (except this one), and in conversation. Paxie's terminology appears only at the file-format boundary.

| Paxie JSON key | Internal term | Description |
|----------------|---------------|-------------|
| `collectorArea` | reserve / bucket reserve | The static grid of buckets the player picks from. |
| `hexStackArea` | hex field | The hexagonal pile of stacked slices being cleared. |
| `singleBlockCollectors` | plain buckets | Standard pickable buckets with a known color. |
| `woodBoxCollectors` | ?-buckets (question buckets) | Buckets with hidden color revealed on reachability. |
| `iceCollectors` | ice buckets | Frozen buckets that thaw after N player moves. |
| `deadCells` | walls | Permanent obstacles that block reachability. |
| `tunnels` (inside `collectorArea`) | generators | Cells that produce buckets into a facing neighbor. |
| `pinBlockers` | pin blockers | Ray-shaped barriers destroyed by clearing the cell behind their origin. |
| `mysteryCollectors` | (unsupported) | Not modeled; empty in all 100 surveyed levels. |
| `tiedPairs` | (unsupported) | Not modeled; empty in all 100 surveyed levels. |
| `keyLocks` | (unsupported) | Not modeled; present only in level 100. |
| `hexStackArea.tunnels` | (unsupported) | Not modeled; always empty in surveyed levels. |
| `editorMeta` | generator metadata | Paxie editor parameters; informational, not used by simulator runtime. |

`hiddenColor` (field on `woodBoxCollectors` and `iceCollectors`) corresponds to the bucket's true color, stored in the file but withheld from the agent observation until the reveal condition fires. `collectorQueue` (field on `tunnels`) corresponds to the generator's output queue (per `HEXFALL_MDP_SPEC.md` §3.3).

---

## 3. Coordinate systems

Both grids in this format use `(x, y)` integer pairs with `y` increasing **downward** (row index from the top) and `x` increasing **rightward** (column index from the left). This is the convention used by Paxie's editor. Internally, the simulator stores reserve cells as `(row, col)` pairs with `row = y` and `col = x` — the conversion is trivial.

### 3.1 `collectorArea` grid

The reserve uses a square grid with **4-adjacency** (up/down/left/right). The grid dimensions are `collectorArea.gridWidth` (columns) × `collectorArea.gridHeight` (rows). Every cell-bearing entry inside `collectorArea` uses `(x, y)` with `0 ≤ x < gridWidth` and `0 ≤ y < gridHeight`.

### 3.2 `hexStackArea` grid

The hex field uses hex geometry per `HEXFALL_RULES.md` §3. The grid dimensions are `hexStackArea.gridWidth` (columns) × `hexStackArea.gridHeight` (rows). Stacks are placed at `(x, y)` positions; the visual offset for alternating rows is implicit in the renderer.

The simulator may convert hex `(x, y)` to axial `(q, r)` coordinates internally for neighbor lookups; the conversion is the standard odd-r-offset-to-axial formula and is handled by the loader.

---

## 4. Top-level structure

```json
{
  "levelNumber":      <integer>,
  "levelVersionCode": <integer>,
  "collectorArea":    <CollectorAreaObject>,
  "hexStackArea":     <HexStackAreaObject>,
  "editorMeta":       <EditorMetaObject>
}
```

| Field | Type | Description |
|-------|------|-------------|
| `levelNumber` | integer ≥ 1 | 1-based level index. Used by the loader for diagnostics; the level identity comes from the filename, not this field. |
| `levelVersionCode` | integer | Format version. Current value: `1`. |
| `collectorArea` | object | Reserve grid and cell contents. See §5. |
| `hexStackArea` | object | Hex field grid and stack contents. See §6. |
| `editorMeta` | object | Paxie editor metadata. Informational only. See §7. |

No other top-level keys are permitted. The schema rejects unknown keys at the top level.

---

## 5. The `collectorArea` object

```json
"collectorArea": {
  "gridWidth":              <integer>,
  "gridHeight":             <integer>,
  "singleBlockCollectors":  [ ... ],
  "woodBoxCollectors":      [ ... ],       // optional; absent or empty allowed
  "iceCollectors":          [ ... ],       // optional; absent or empty allowed
  "deadCells":              [ ... ],       // optional; absent or empty allowed
  "tunnels":                [ ... ],       // optional; absent or empty allowed
  "pinBlockers":            [ ... ],       // optional; absent or empty allowed
  "mysteryCollectors":      [ ... ],       // present in schema; must be empty (unsupported)
  "tiedPairs":              [ ... ] | null, // present in schema; must be empty or null (unsupported)
  "keyLocks":               [ ... ]        // optional; if present and non-empty, level is unsupported
}
```

`gridWidth` and `gridHeight` are required. The other arrays are optional; absent or empty arrays are equivalent. Each array is described below.

### 5.1 `singleBlockCollectors` (plain buckets)

```json
{ "x": <integer>, "y": <integer>, "color": <string> }
```

| Field | Type | Description |
|-------|------|-------------|
| `x`, `y` | integer | Cell coordinates in the reserve grid. |
| `color` | string | Bucket color. See §8 for color token rules. |

A `singleBlockCollector` at `(x, y)` is a plain bucket — pickable when reachable, color known to the player at all times.

### 5.2 `woodBoxCollectors` (?-buckets)

```json
{ "x": <integer>, "y": <integer>, "hiddenColor": <string> }
```

| Field | Type | Description |
|-------|------|-------------|
| `x`, `y` | integer | Cell coordinates in the reserve grid. |
| `hiddenColor` | string | The bucket's true color. Stored in the file but withheld from the agent observation until the bucket becomes reachable (per `HEXFALL_RULES.md` §5, `HEXFALL_MDP_SPEC.md` §4.4). |

### 5.3 `iceCollectors` (ice buckets)

```json
{ "x": <integer>, "y": <integer>, "hiddenColor": <string>, "iceCapacity": <integer> }
```

| Field | Type | Description |
|-------|------|-------------|
| `x`, `y` | integer | Cell coordinates in the reserve grid. |
| `hiddenColor` | string | The bucket's true color. Hidden from the agent observation until the bucket thaws. |
| `iceCapacity` | integer ≥ 1 | Thaw threshold. The bucket thaws when the level move counter (per `HEXFALL_MDP_SPEC.md` §3.6) reaches this value. Observed range in the dataset: 3–22, median 10. |

Per `HEXFALL_RULES.md` §5 ice bucket subsection: an ice bucket is unpickable while frozen, blocks reachability through its cell while frozen, and behaves as a plain bucket once thawed.

### 5.4 `deadCells` (walls)

```json
{ "x": <integer>, "y": <integer> }
```

| Field | Type | Description |
|-------|------|-------------|
| `x`, `y` | integer | Cell coordinates. The cell is a permanent wall per `HEXFALL_RULES.md` §5. |

Walls have no additional fields.

### 5.5 `tunnels` (generators)

```json
{
  "x": <integer>,
  "y": <integer>,
  "direction": "Up" | "Down" | "Left" | "Right",
  "collectorQueue": [ { "color": <string> }, ... ]
}
```

| Field | Type | Description |
|-------|------|-------------|
| `x`, `y` | integer | Cell coordinates of the generator itself. |
| `direction` | string | Direction the generator faces. **Case-sensitive: capitalized first letter.** One of `"Up"`, `"Down"`, `"Left"`, `"Right"`. |
| `collectorQueue` | array of objects | Buckets the generator will produce, in order. Index 0 is the next bucket. Each object has a `color` field (string, level color). Length of the queue equals the generator's remaining count. |

Per `HEXFALL_RULES.md` §5: the generator stays in place after exhaustion and continues to block reachability through its cell. The queue is hidden from the agent observation; only the remaining count and facing direction are visible.

### 5.6 `pinBlockers` (pin blockers)

```json
{
  "x": <integer>,
  "y": <integer>,
  "direction": "Up" | "Down" | "Left" | "Right",
  "blockCount": <integer>        // optional; default 0 if absent
}
```

| Field | Type | Description |
|-------|------|-------------|
| `x`, `y` | integer | Origin cell of the pin's ray. |
| `direction` | string | Direction the pin faces (the direction the ray extends). Case-sensitive capitalized first letter. |
| `blockCount` | integer ≥ 0 (optional) | Number of additional cells beyond the origin that the ray covers. If `blockCount: 2`, the ray covers `(origin, origin+1, origin+2)` — three cells total. If `blockCount: 0` or the field is absent, the ray extends from the origin to the grid edge in the direction. |

Per `HEXFALL_RULES.md` §5 pin blocker subsection: the ray occupies cells (which become unpickable), blocks reachability propagation along the ray, and overlays any cell content underneath (e.g., a wall co-located with the ray). The pin is destroyed when the cell directly opposite the origin (one step in the direction *opposite* the pin's facing) ends an automatic-update tick empty. Level-construction invariant: that destruction cell must contain a pickable bucket at level start — never a wall, never a generator.

### 5.7 Unsupported sub-arrays

The following sub-arrays are recognized by the schema but **not supported by the simulator**. The loader's behavior on encountering non-empty entries in any of these is to **raise an error** and refuse to load the level (see §10).

| Field | Schema requirement | Loader behavior |
|-------|--------------------|-----------------|
| `mysteryCollectors` | Must be empty array if present. Empty in all 100 surveyed levels. | Empty: ignored. Non-empty: load error. |
| `tiedPairs` | Empty array or `null`. Empty in all 100 surveyed levels. | Empty/null: ignored. Non-empty: load error. |
| `keyLocks` | Optional; schema allows entries with `color`, `keyX`, `keyY`, `lockX`, `lockY`. | Present and non-empty: load error. (In the dataset, only level 100 has this.) |

### 5.8 Cell occupancy rules

A reserve cell `(x, y)` is described by *at most one* of {singleBlockCollector, woodBoxCollector, iceCollector, deadCell, tunnel} — these are mutually exclusive cell contents. Pin rays may overlay any cell, including overlay onto a wall or onto a cell that is otherwise empty. The level's pin definitions are separate from the cell contents.

Cells not mentioned in any of the supported sub-arrays are **empty at level start**. Empty cells have no explicit representation in the JSON; absence is the encoding.

---

## 6. The `hexStackArea` object

```json
"hexStackArea": {
  "gridWidth":  <integer>,
  "gridHeight": <integer>,
  "stacks": [
    { "x": <integer>, "y": <integer>, "colors": [<string>, ...] },
    ...
  ],
  "tunnels": []                       // present in schema; must be empty (unsupported)
}
```

| Field | Type | Description |
|-------|------|-------------|
| `gridWidth`, `gridHeight` | integer | Hex grid dimensions. Observed in the dataset: width 13 (constant); height 8–33, median 16. |
| `stacks` | array of stack objects | Every position in the grid that contains a stack. Stacks not listed are absent (the position is empty hex space). |
| `tunnels` | array | Always empty in surveyed levels. Schema permits the field for forward compatibility; non-empty values trigger a load error. |

### 6.1 Stack objects

```json
{ "x": <integer>, "y": <integer>, "colors": [<string>, ...] }
```

| Field | Type | Description |
|-------|------|-------------|
| `x`, `y` | integer | Position of this stack in the hex grid. |
| `colors` | array of strings | Slice colors, ordered **top-to-bottom**: index 0 is the top slice (next to be consumed), index `len-1` is the bottom of the stack. Each string is a color token. Must be non-empty; stacks with zero slices are absent. |

**Bottom row:** The bottom row of the field, in the Paxie convention, is the highest `y` value among all stacks. The simulator extracts this as the consumable row per `HEXFALL_RULES.md` §3.

**Slice-order rationale:** Top-to-bottom matches consumption order (top slice consumed first) and matches the natural reading of a vertical stack.

---

## 7. The `editorMeta` object

```json
"editorMeta": {
  "totalBlocks":          <integer>,        // required
  "colorCount":           <integer>,        // required
  "maxColorsPerStack":    <integer>,        // required
  "heightMin":            <integer>,        // required
  "heightMax":            <integer>,        // required
  "randomness":           <number>,         // required
  "verticalPercent":      <number>,         // required
  "horizontalPercent":    <number>,         // required
  "mysteryPercent":       <number>,         // required
  "clusteringEnabled":    <boolean>,        // optional
  "clusteringPercent":    <number>,         // optional
  "colorsPerStackRatios": [<number>, ...],  // optional
  "zoneCount":            <integer>,        // optional
  "zoneColors":           [[<string>, ...], ...]  // optional
}
```

This object captures the parameters Paxie's level editor used to generate the level. The simulator's runtime does **not** read these values — they do not affect gameplay. They are preserved in the file because:

1. They are the **target output of our Week 5 generator**. Producing a level means producing both the gameplay state (collectors, stacks) *and* the editor parameters that "would have" generated it.
2. They serve as a difficulty hint for the difficulty oracle's analysis — e.g., `colorsPerStackRatios[0]` is Paxie's "% of stacks with maximum color mixing" knob, a documented difficulty driver.

Selected fields, briefly (the schema covers value ranges):

- **`totalBlocks`** — total slice count in the field. Observed range: 144–1272, median 657.
- **`colorCount`** — declared distinct color count. The loader cross-checks this against the actual count of distinct color tokens in the level data; mismatches produce a warning but do not block loading (two levels in the dataset — 87 and 91 — have this mismatch).
- **`maxColorsPerStack`** — maximum distinct colors per stack. Always 1 or 2 in the dataset.
- **`heightMin`, `heightMax`** — stack height range.
- **`randomness`** — generator randomness knob, range 0.0–0.65 in dataset.
- **`verticalPercent`, `horizontalPercent`** — generator-pattern knobs.
- **`clusteringEnabled`, `clusteringPercent`** — clustering pass parameters.
- **`colorsPerStackRatios`** — when present (only when `maxColorsPerStack == 2`), `colorsPerStackRatios[0]` is the target percentage of stacks with two distinct colors. Decoded empirically from the dataset: matches the actual ratio within ~2 percentage points across levels.
- **`zoneCount`, `zoneColors`** — multi-zone color partitioning parameters; mostly absent in the dataset.

The simulator parses and stores these fields but does not act on them at runtime.

---

## 8. Color tokens

Color values are short string tokens, treated by the simulator as opaque labels. Only distinctness matters for gameplay logic.

### 8.1 Short codes

The 17 short codes observed in the dataset, used by 95 of 100 levels:

`b`, `br`, `db`, `dg`, `dgr`, `do`, `dr`, `f`, `g`, `gr`, `o`, `og`, `p`, `pk`, `r`, `w`, `y`

Per `HEXFALL_RULES.md` §10, approximate human meanings are documented for debugging only.

### 8.2 Full-name normalization

A subset of levels (5 of 100 — levels 81, 82, 83, 85, 86) uses full-word color tokens instead of short codes. The encoding is per-level (no mixed-encoding levels exist in the dataset). The loader normalizes full names to short codes at parse time using this mapping:

| Full name | Short code | Full name | Short code |
|-----------|------------|-----------|------------|
| `Yellow` | `y` | `White` | `w` |
| `Blue` | `b` | `DarkBlue` | `db` |
| `Red` | `r` | `DarkRed` | `dr` |
| `Green` | `g` | `DarkGray` | `dgr` |
| `Purple` | `p` | `OliveGreen` | `og` |
| `Pink` | `pk` | | |
| `Orange` | `o` | | |

After normalization, all internal simulator state, observations, and code references use short codes exclusively.

### 8.3 Per-level palette

A level's palette is the set of distinct color tokens actually appearing in any of: stack slices, plain bucket colors, ?-bucket hidden colors, ice bucket hidden colors, and generator queue colors. The loader computes this set at load time and uses it to size any color-indexed observation features.

The `editorMeta.colorCount` field declares the same count, but the loader treats it as a cross-check, not a constraint — two levels in the dataset have `colorCount` mismatches and the loader emits a warning rather than an error.

---

## 9. Unsupported mechanics

The schema permits, but the simulator does not support:

| Mechanic | Schema allows | Loader behavior on encounter |
|----------|---------------|------------------------------|
| `collectorArea.mysteryCollectors` non-empty | Yes | **Load error.** |
| `collectorArea.tiedPairs` non-empty | Yes | **Load error.** |
| `collectorArea.keyLocks` non-empty | Yes | **Load error.** |
| `hexStackArea.tunnels` non-empty | Yes | **Load error.** |

Levels containing any of these are unloadable by the simulator. The training pipeline maintains a known-good list of level IDs that excludes these. Of the 100 levels in the Paxie dataset, **99 are loadable** under this policy (only level 100, which contains a `keyLock`, is excluded outright; six additional levels — 74, 84, 87, 89, 94, 98 — use the `pinBlockers.blockCount` field, which is supported).

**Rationale:** Hard rejection on unsupported mechanics is safer than silent skipping. Partial loading would lead to silent miscomputation of agent winrates on levels the simulator did not fully model.

---

## 10. Validation rules

Two layers of validation apply.

### 10.1 Schema validation (`level_schema.json`)

The JSON schema enforces structural rules: required fields, type correctness, value enumerations (e.g., `direction` is one of four strings), array item shape. A level file failing schema validation is rejected at load time before any semantic checks run.

### 10.2 Semantic validation (in `level_loader.py`)

The loader performs the following checks after schema validation passes:

1. **Cell exclusivity.** No two non-pin cell-bearing entries refer to the same `(x, y)` in `collectorArea` (per §5.8). Multiple pins overlapping a single cell are permitted.
2. **Cell-in-bounds.** All `(x, y)` coordinates in `collectorArea` lie within `[0, gridWidth) × [0, gridHeight)`. Same for `hexStackArea`.
3. **Generator queue consistency.** Every color in every `collectorQueue` is a level color.
4. **Color cross-check.** The `editorMeta.colorCount` value is compared to the actual distinct-color count. A mismatch produces a warning, not an error.
5. **Pin destruction cell.** For each pin, the cell directly opposite the origin (one step in the direction opposite the pin's facing) is inspected. Expected case: the cell contains a pickable entity at level start — a `singleBlockCollector`, `woodBoxCollector`, or `iceCollector`. Error cases (loader raises): the cell contains a wall, generator, or another pin. Warning case (loader logs but loads): the cell is off-grid (origin is on an edge of the grid such that "one step opposite the facing direction" leaves the grid). Off-grid destruction cells make the pin effectively permanent for the level; this is not malformed per se (the worked example in §11 exhibits this configuration deliberately) but is flagged because real Paxie levels do not produce this geometry.
6. **Unsupported mechanics.** Per §9: presence of any non-empty unsupported sub-array triggers an error.
7. **Color token validity.** Every color token in the level is either a known short code or a known full-name alias (per §8). Unknown tokens raise an error.
8. **Slice-bucket parity** (recommended warning, not hard error): the sum of all slice counts in `hexStackArea.stacks` should equal `bucket_count × bucket_capacity` (where `bucket_count` is the total of single + wood + ice collectors plus all queued generator outputs, and `bucket_capacity` is the simulator's per-bucket capacity, default 24 per `HEXFALL_RULES.md` §4). Real Paxie levels satisfy parity exactly at capacity 24 — empirically validated across all 99 supported levels in the dataset, zero exceptions. Procedurally generated levels under construction may not.

---

## 11. Worked example: `demo-supported-types.json`

This example is a small hand-built level demonstrating all six supported cell types in `collectorArea` plus a simple hex field. It is not a balanced or playable Hex Fall level — it exists to exercise the parser end-to-end.

```json
{
  "levelNumber": 9001,
  "levelVersionCode": 1,
  "collectorArea": {
    "gridWidth": 4,
    "gridHeight": 3,
    "singleBlockCollectors": [
      { "x": 0, "y": 0, "color": "r" },
      { "x": 1, "y": 0, "color": "b" },
      { "x": 0, "y": 1, "color": "g" }
    ],
    "woodBoxCollectors": [
      { "x": 2, "y": 0, "hiddenColor": "g" }
    ],
    "iceCollectors": [
      { "x": 2, "y": 1, "hiddenColor": "r", "iceCapacity": 3 }
    ],
    "deadCells": [
      { "x": 3, "y": 0 }
    ],
    "tunnels": [
      {
        "x": 1, "y": 2,
        "direction": "Up",
        "collectorQueue": [ { "color": "r" }, { "color": "b" } ]
      }
    ],
    "pinBlockers": [
      { "x": 0, "y": 2, "direction": "Right", "blockCount": 0 }
    ],
    "mysteryCollectors": [],
    "tiedPairs": [],
    "keyLocks": []
  },
  "hexStackArea": {
    "gridWidth": 4,
    "gridHeight": 2,
    "stacks": [
      { "x": 0, "y": 0, "colors": ["r", "b", "g"] },
      { "x": 2, "y": 0, "colors": ["b", "r"] },
      { "x": 0, "y": 1, "colors": ["g", "r", "b", "g"] },
      { "x": 2, "y": 1, "colors": ["r"] }
    ],
    "tunnels": []
  },
  "editorMeta": {
    "totalBlocks": 10,
    "colorCount": 3,
    "maxColorsPerStack": 2,
    "heightMin": 1,
    "heightMax": 4,
    "randomness": 0.0,
    "verticalPercent": 0.0,
    "horizontalPercent": 0.0,
    "mysteryPercent": 0.0
  }
}
```

### 11.1 Reserve layout (4 × 3 grid, `.` = empty)
y\x:  0              1              2                  3
0:    plain(r)       plain(b)       wood(g hidden)     wall
1:    plain(g)       .              ice(r hidden, cap=3)  .
2:    pin→Right (∞)  generator↑     .                  .

The pin at `(0, 2)` faces `Right` with `blockCount: 0`, so its ray extends from `(0, 2)` to the grid edge — covering cells `(0, 2)`, `(1, 2)`, `(2, 2)`, `(3, 2)`. Note that the generator at `(1, 2)` is also in the ray — the pin overlays it visually but the generator is the actual cell content underneath; the pin's ray makes the cell non-pickable (moot for a generator, which is never pickable anyway) and blocks reachability across the row.

### 11.2 Reachability at load

Per `HEXFALL_RULES.md` §5: cells in the top row that contain a pickable entity are reachable. Walls are in the top row but are not pickable. Reachability propagates through emptied cells.

- `(0,0)`, `(1,0)`, `(2,0)` — top row, pickable. **Reachable.** The ?-bucket at `(2,0)` has its hidden color revealed at load because it transitions from "not yet reachable" to "reachable" on the very first reachability computation.
- `(3,0)` — top row, but a wall. Not pickable.
- `(0,1)` — plain bucket. Its top-row neighbor `(0,0)` is a bucket, not empty, so `(0,1)` is **not reachable** at load. It becomes reachable after `(0,0)` is picked.
- `(2,1)` — ice bucket, frozen. Even setting reachability aside, it is unpickable while frozen.
- `(1,2)` — generator (with pin ray overlaid). Never pickable; reachability does not propagate through it; the pin ray also blocks propagation across the row.
- `(0,2)` — origin of the pin ray. Non-pickable for the pin's lifetime.

The pin's destruction cell is the cell one step opposite its facing direction. Facing `Right` means destruction cell is at `(-1, 2)` — off-grid. This pin can never be destroyed in this example. (A real level would not have this configuration; the example is constructed to exercise the parser, not to be playable.)

### 11.3 Generator behavior on load

Per `HEXFALL_MDP_SPEC.md` §5.4 ("Initial load"), the loader runs the automatic-update loop at level load. The generator at `(1, 2)` faces `Up` to `(1, 1)`, which is empty at level start. The generator fires on load: it produces `"r"` (the first color in the queue) into `(1, 1)`, decrements its remaining count to 1, and advances the queue to `[{"color": "b"}]`.

After the load-time tick, `(1, 1)` contains a plain red bucket. Reachability is recomputed: `(1, 1)` is now a bucket but it does not become reachable, because no path of empties connects it to the top edge (its top-row neighbor `(1, 0)` is a bucket).

### 11.4 Ice thaw timing

The ice bucket at `(2, 1)` has `iceCapacity: 3`. It thaws when the move counter reaches 3 — i.e., after the player's third pick of the episode. At the start of the fourth tick of the action loop (per `HEXFALL_MDP_SPEC.md` §5.4 step 2.i), the ice thaw check fires; the bucket becomes pickable from that point if it is also reachable.

---

## 12. Relation to MDP spec §3

Mapping between `HEXFALL_MDP_SPEC.md` §3 sub-sections and this file format:

| MDP spec §3 component | Level file location |
|-----------------------|---------------------|
| §3.1 Hex field — grid topology | `hexStackArea.gridWidth`, `.gridHeight`, `.stacks[*].x`, `.stacks[*].y` |
| §3.1 Hex field — stack contents | `hexStackArea.stacks[*].colors` |
| §3.2 Buffer — slot count | Not in level file. Constant 5 per `HEXFALL_RULES.md` §4. |
| §3.2 Buffer — bucket capacity | Not in level file. Default 25 per `HEXFALL_RULES.md` §4. Simulator-level parameter. |
| §3.3 Reserve — grid topology | `collectorArea.gridWidth`, `.gridHeight` |
| §3.3 Reserve — plain bucket | `collectorArea.singleBlockCollectors[*]` with `color` |
| §3.3 Reserve — ?-bucket | `collectorArea.woodBoxCollectors[*]` with `hiddenColor`; `revealed = false` at load |
| §3.3 Reserve — generator | `collectorArea.tunnels[*]` with `direction` and `collectorQueue` |
| §3.3 Reserve — wall | `collectorArea.deadCells[*]` |
| §3.3 Reserve — ice bucket | `collectorArea.iceCollectors[*]` with `hiddenColor` and `iceCapacity`; `thawed = false` at load |
| §3.3 Reserve — pin blockers | `collectorArea.pinBlockers[*]` with `direction` and optional `blockCount`; `destroyed = false` at load |
| §3.4 RNG state | Not in level file. Provided at runtime (seed passed to simulator). |
| §3.5 Quiescence flag | Not in level file. Runtime simulator state. |
| §3.6 Move counter | Not in level file. Initialized to 0 at load. |

Buffer parameters (slot count, bucket capacity) are no longer encoded in the level file — they are simulator-wide constants. This is a change from the previous internal format (which encoded them per-level) and matches Paxie's format, where these are not present in the level data.

The level file encodes all *deterministic* initial state. Runtime state (RNG seed, quiescence, current fill counts, revealed flags, thawed flags, move counter) is initialized by the simulator at load time from the level data plus any externally supplied seed.

---

## 13. Revision history

- **May 7, 2026:** Initial version. Produced in worker chat for Week 1 issue (LEVEL_FORMAT.md). Format covered five reserve cell types (plain bucket, ?-bucket, generator, wall, empty), odd-r offset hex coordinates, and all MDP spec §3 fields. Used internal top-level keys (`meta`/`field`/`buffer`/`reserve`) and snake_case throughout.
- **May 7, 2026 (commander review):** Fixed worked-example reachability walkthrough; resolved generator-on-load firing behavior in parallel with `HEXFALL_RULES.md` §5 and `HEXFALL_MDP_SPEC.md` §5.4. See prior revision-history line in the May 7 worker deliverable for details.
- **May 13, 2026:** Full rewrite. Document now describes Paxie's native level format directly, per the May 13 format-adoption decision (`DECISIONS.md`). The previous internal format (`meta`/`field`/`buffer`/`reserve` with snake_case keys and `plain_bucket`/`question_bucket`/`wall`/`generator` cell types) is fully superseded. Two new reserve cell types are documented: ice buckets (`iceCollectors`) and pin blockers (`pinBlockers`). A new §2 (Vocabulary glossary) maps Paxie's JSON keys to the simulator's internal terminology — internal code and other docs continue to use "bucket"/"wall"/"generator", while this document uses Paxie's terms because it describes the file format. §8 documents the color token palette and full-name normalization for the 5 outlier levels (81, 82, 83, 85, 86) authored with a different Paxie editor build. §9 documents the four unsupported mechanics (`mysteryCollectors`, `tiedPairs`, `keyLocks`, non-empty `hexStackArea.tunnels`) and the loader's hard-reject policy on them. Worked example in §11 replaced with a Paxie-format example demonstrating all six supported reserve cell types. Coordinate convention switched from `[row, col]` to `(x, y)` to match Paxie. `level_schema.json` was concurrently replaced with the Paxie-format schema (formerly `paxie_level_schema.json`) and the old schema deleted.
- **May 16, 2026:** Two corrections to §10.2 following issue #6 implementation. (a) Slice-bucket parity check: bucket capacity corrected from 25 to 24, with empirical validation note (all 99 supported Paxie levels satisfy parity exactly at capacity 24). (b) Pin destruction cell check: off-grid destruction cells now produce a warning, not an error — this reconciles §10.2 with the §11 worked example, which deliberately exhibits an off-grid pin for parser-exercise purposes. Wall/generator/other-pin destruction cells continue to be errors. See `DECISIONS.md` May 16 entry.
