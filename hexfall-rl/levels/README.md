# Test Levels

Hand-built fixture levels for verifying specific mechanics. Each is the smallest level
that clearly exercises its target mechanic. Not balanced gameplay — intentionally tiny.

Format: all eight levels are now in Paxie's native JSON format (per
`documents/LEVEL_FORMAT.md`). Coordinates are `(x, y)` with `x = col`, `y = row`,
y increasing downward.

All levels emit a slice-bucket parity warning at load: the simulator's bucket
capacity is 25 (per `HEXFALL_RULES.md` §4), but these fixtures are intentionally
tiny — they never come close to filling a bucket. The warning is expected and
does not affect the mechanic each level exercises.

---

## tiny_solvable.json
Smoke-test level. 2 colors, two single-color stacks. Solvable regardless of pick order.
- **Expected outcome:** WIN.

---

## forced_lose.json
**Mechanic:** Lose detection via structural color mismatch (rules-doc deadlock).

**Construction:** Reserve is 5×1 with five blue plain buckets. Field stack at
`(0, 0)` has only red slices. The agent can only pick blue buckets, none of
which can pull from the red field. After five picks the buffer is full of
permanently-stuck blues and no further legal action exists → `deadlock` lose.

- **Expected outcome:** LOSE, every seed. Deterministic.
- **Termination reason:** `deadlock`

---

## generator_test.json
**Mechanic:** Generator load-time firing, mid-episode firing, and exhaustion.

**Construction:** Reserve is 2×2. Generator at `(1, 0)` faces `Down` to `(1, 1)`
with queue `[g, g]`. `(1, 1)` is empty at load → the load-time auto-update loop
fires the generator immediately: `(1, 1)` receives a green bucket and the
generator's `remaining` decrements from 2 to 1.

The reachability path through the generator cell `(1, 0)` is permanently blocked.
`(1, 1)` is reachable only via `(0, 0) → (0, 1)` once those cells are cleared.

- **Expected outcome:** WIN or LOSE depending on pick order.

---

## hidden_test.json
**Mechanic:** ?-bucket reveal-on-reachable.

**Construction:** Reserve is 2×2 with no initial empty cells. `(0, 0)` and
`(1, 0)` are plain red. `(0, 1)` is a ?-bucket (true color: blue) — not reachable
at load because `(0, 0)` is occupied. When the agent picks `(0, 0)`, `(0, 1)`
becomes reachable and its color is immediately revealed in the next observation.

- **Expected outcome:** WIN every seed. Solvable regardless of pick order.
- **Termination reason:** `win`

---

## deadlock_test.json
**Mechanic:** Rules-doc buffer deadlock reachable under bad play; solvable under good play.

**Construction:** Reserve is 7×1 (all top row, all reachable at start). `(0, 0)`
through `(4, 0)`: five blue buckets. `(5, 0)`–`(6, 0)`: two red buckets. Field has
only red slices.

**Bad sequence (deadlock):** Pick the five blues first → buffer fills with blue
buckets that can never pull (no blue slices) → no legal actions → LOSE (deadlock).
**Good sequence (win):** Pick the two reds first → the red bucket drains all red
slices → field empty → WIN.

- **Expected outcome:** WIN or LOSE depending on pick order.

---

## wall_test.json
**Mechanic:** Walls shape the reachability graph.

**Construction:** Reserve is 3×3. Wall at `(1, 0)`. Without the wall, picking
`(1, 0)` would directly open `(1, 1)`. With the wall, `(1, 1)` is reachable only
via the side paths `(0, 0) → (0, 1)` or `(2, 0) → (2, 1)`. `(1, 2)` is reachable
only after `(1, 1)` is emptied. `(0, 2)` and `(2, 2)` are initially empty.

- **Expected outcome:** WIN or LOSE depending on pick order.

---

## ice_test.json
**Mechanic:** Ice bucket thaw timing and action legality.

**Construction:** Reserve is 3×1 (all top row). `(0, 0)` is a plain blue bucket;
`(1, 0)` and `(2, 0)` are ice buckets. The ice bucket at `(1, 0)` hides red with
`iceCapacity: 1`; the one at `(2, 0)` hides green with `iceCapacity: 2`. Field
stack at `(0, 0)` is `[b, r, g]` (top-to-bottom).

**What it exercises:**
1. **Legality while frozen.** At load, only `(0, 0)` is legal — both ice buckets
   are reachable (they're in the top row) but frozen, so `legal_actions_mask`
   excludes them.
2. **Thaw at threshold.** Picking `(0, 0)` increments the move counter to 1.
   The ice at `(1, 0)` thaws (threshold 1) at the start of the next tick. The
   thaw event reveals its color (red) in the observation.
3. **Threshold ordering.** After picking `(1, 0)`, move counter is 2; the ice at
   `(2, 0)` thaws.
4. **Deterministic solvability.** The level is solvable by the random agent
   because at each quiescent state only one action is legal.

- **Expected outcome:** WIN, every seed. Deterministic action sequence:
  `(0, 0) → (1, 0) → (2, 0)`.

---

## pin_test.json
**Mechanic:** Pin destruction, ray-cell unpickability, and cascading destruction.

**Construction:** Reserve is 3×3. Two plain red buckets at `(0, 1)` and `(2, 1)`.
Two pin blockers:
- **Pin A** at `(1, 1)` facing `Left`, `blockCount: 0` → ray covers `(1, 1)` and
  `(0, 1)`. Destruction cell (one step opposite-of-Left) is `(2, 1)`.
- **Pin B** at `(0, 2)` facing `Down`, `blockCount: 0` → ray covers `(0, 2)`.
  Destruction cell (one step opposite-of-Down) is `(0, 1)`.

Field stack at `(0, 0)` is `[r, r]`.

**What it exercises:**
1. **Pin ray non-pickability.** The plain bucket at `(0, 1)` sits underneath Pin
   A's ray. At load, picking `(0, 1)` is illegal regardless of reachability.
2. **Reachability propagation blocked.** Reachability cannot route through any
   cell in either pin's ray.
3. **Pin destruction.** Picking the destruction cell of Pin A — the plain bucket
   at `(2, 1)` — empties that cell. At the same tick's pin-destruction phase,
   Pin A destroys; its ray cells `(1, 1)` and `(0, 1)` become empty.
4. **Cascade.** Pin B's destruction cell is `(0, 1)`, which Pin A's destruction
   just cleared. On the next iteration of the destruction phase, Pin B destroys.
   Both pins are removed within the same step.

After the cascade the field's two red slices are pulled by the picked red bucket
and the field empties → WIN.

- **Expected outcome:** WIN every seed. Deterministic.
- **Refill protection** (generator firing into a destruction cell on the same
  tick) is not exercised by this level — that scenario is covered inline in
  `tests/test_game.py`.

---

## Coordinate cheat sheet

Paxie format uses `(x, y)` with `y` increasing downward. The loader converts at the
file boundary to internal `(row, col)` where `row = y`, `col = x`. Internal
simulator state (`state.reserve[row][col]`, action `(row, col)`) uses
`(row, col)`; the level JSON uses `(x, y)`.
