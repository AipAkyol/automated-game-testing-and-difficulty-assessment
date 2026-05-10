# Test Levels

Hand-built fixture levels for verifying specific mechanics. Each is the smallest level
that clearly exercises its target mechanic. Not balanced gameplay — intentionally tiny.

---

## tiny_solvable.json
Smoke-test level. 2 colors, small stacks. Run to confirm the simulator loads and the
random agent completes an episode without crashing.
- **Expected outcome:** WIN (always solvable regardless of pick order).

---

## forced_lose.json
**Mechanic:** Lose detection via structural color mismatch.

**Construction:** Field contains only `red` slices; reserve contains only a `blue` bucket.
No blue ever appears in the field, so the blue bucket can never fill. Agent picks the
single blue bucket (the only legal action), it enters the buffer, no consumption is
possible, reserve is now empty, zero legal actions → immediate deadlock-lose.

- **Expected outcome:** LOSE, every seed. Deterministic — no random element.
- **Termination reason:** `deadlock`

---

## generator_test.json
**Mechanic:** Generator load-time firing, mid-episode firing, and exhaustion.

**Construction:** Reserve is 2×2. Generator at `[0,1]` faces down to `[1,1]`. `[1,1]`
is absent from the level file (implicitly empty), so the load-time auto-update loop
fires the generator immediately: `[1,1]` receives a green bucket and generator
`remaining` decrements from 2 to 1. The agent's first observation already has `[1,1]`
populated.

**Mid-episode fire:** After the agent picks `[0,0]` (opening `[1,0]`) and then picks
`[1,0]` (opening `[1,1]`), the agent can pick `[1,1]` (freeing it) → generator fires
again into `[1,1]`, `remaining` → 0 (exhausted).

**Reachability note:** Reachability never propagates through the generator cell `[0,1]`.
`[1,1]` is reachable only via the side path `[1,0]` → `[0,0]` → top edge.

- **Expected outcome:** WIN or LOSE depending on pick order (random agent may deadlock).

---

## hidden_test.json
**Mechanic:** ?-bucket reveal-on-reachable.

**Construction:** Reserve is 2×2, no initial empty cells. `[0,0]` and `[0,1]` are plain
red buckets (reachable at start). `[1,0]` is a `?`-bucket (true color: blue) — not
reachable at load because `[0,0]` above it is occupied. When the agent picks `[0,0]`,
`[1,0]` becomes reachable and its color is **immediately revealed as blue** in the
next observation (before the agent picks it).

- **Expected outcome:** WIN every seed. Solvable regardless of pick order.
- **Termination reason:** `win`

---

## deadlock_test.json
**Mechanic:** Rules-doc buffer deadlock reachable under bad play; solvable under good play.

**Construction:** Reserve is 1×7 (all top row, all reachable at start). Cols 0–4: five
`blue` buckets. Cols 5–6: two `red` buckets. Field has only `red` slices.

**Bad sequence (deadlock):** Pick cols 0–4 (all 5 blues) → buffer full with blue buckets
→ field has no blue slices → no consumption ever → buffer cannot free a slot → zero
legal actions → LOSE (deadlock). Specific sequence: `[0,0], [0,1], [0,2], [0,3], [0,4]`.

**Good sequence (win):** Pick cols 5–6 (both reds) → buffer consumes all 4 red slices
→ both buckets fill → field cleared → WIN.

- **Deadlock seeds (random agent, seeds 0–19):** 3, 4, 5, 8, 11, 12, 13, 14
- **Win seeds:** 0, 1, 2, 6, 7, 9, 10, 15, 16, 17, 18, 19
- **Parity note:** 4 slices ≠ 7 buckets × 2 = 14 capacity. Expected warning — the 5
  blue buckets are intentionally never consumable.

---

## wall_test.json
**Mechanic:** Walls shaping the reachability graph.

**Construction:** Reserve is 3×3. Wall at `[0,1]`. Without the wall, picking `[0,1]`
would empty it and directly open `[1,1]` (its lower neighbor). With the wall, `[1,1]`
is only reachable via two alternative paths:
- Path A: pick `[0,0]` → `[1,0]` reachable; pick `[1,0]` → `[1,1]` reachable via left.
- Path B: pick `[0,2]` → `[1,2]` reachable; pick `[1,2]` → `[1,1]` reachable via right.

`[2,1]` (bottom-center) becomes reachable only after `[1,1]` is emptied.
`[2,0]` and `[2,2]` are initially empty (documented; used to keep grid small — these
empty cells don't connect to the top edge so they don't add reachability).

- **Expected outcome:** WIN or LOSE depending on pick order (random agent varies).
