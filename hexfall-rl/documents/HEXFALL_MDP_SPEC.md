# HEXFALL_MDP_SPEC.md

Formalization of Hex Fall as a partially observable Markov decision process (POMDP). This document is the authoritative contract between the game rules (`HEXFALL_RULES.md`) and the simulator / RL agent: it defines the state, observation, action, transition, reward, and termination structure that both must agree on.

This document does **not** redefine game mechanics — those live in `HEXFALL_RULES.md` and are referenced by section number throughout. Where this spec adds detail beyond the rules doc (e.g., tick ordering, replay determinism), those additions are flagged as "spec-level" and listed in §10.

---

## 1. Purpose and scope

This spec answers the questions a simulator implementer or RL practitioner needs answered to build against Hex Fall:

- What is the underlying state $s_t$, and what RNG state is needed for replay?
- What does the agent see at step $t$? What's hidden?
- What is the action space, and how is action legality determined per state?
- What is the transition function $P(s_{t+1} \mid s_t, a_t)$, including the order in which automatic updates fire between actions?
- What is the reward function (placeholder for now; see §6)?
- When does an episode terminate?

This spec covers the **MDP-level abstraction** of Hex Fall. It does not specify tensor shapes, neural-network architectures, or library bindings — those are downstream implementation choices.

---

## 2. POMDP framing

Hex Fall is a partially observable Markov decision process:

$$\langle S, A, O, P, \Omega, R, \gamma \rangle$$

- $S$: full game state (§3)
- $A$: action space (§5)
- $O$: observation space (§4)
- $P : S \times A \to \Delta(S)$: stochastic transition (§5.4)
- $\Omega : S \to O$: observation function — deterministic given state (§4)
- $R : S \times A \times S \to \mathbb{R}$: reward function (§6, placeholder)
- $\gamma$: discount factor — RL training detail, not specified here

Two sources of partial observability (per `HEXFALL_RULES.md` §8):

1. **Hidden bucket colors.** ?-buckets in the reserve hide their color until they become reachable. Color is fixed per-level (deterministic across replays) but unobservable until revealed.
2. **Hidden slice composition.** Slices not satisfying the geometric visibility rules in `HEXFALL_RULES.md` §3 are hidden from the agent. Bottom-row stacks are fully visible; otherwise visibility depends on stack height vs. lower neighbor and grid position.

One source of stochasticity:

3. **Hex fall direction.** When a bottom-row stack clears, one of its (up to two) upper neighbors falls into the empty slot, chosen uniformly at random.

Because Hex Fall is a POMDP with non-trivial hidden state, RL training will require a memory-equipped policy (e.g., LSTM or transformer) — the current observation is not a sufficient statistic for optimal action selection.

---

## 3. State

The full state $s_t \in S$ is the complete description of the game at step $t$. It is what the simulator stores internally and what is hidden from the agent.

A state consists of:

### 3.1 Hex field

- **Grid topology**: a hexagonal grid of stack positions (axial coordinates). Static across the episode (positions don't move; only contents change).
- **Stack contents**: for each grid position, an ordered list of slices from top to bottom, each slice carrying a color label. An empty stack is a position with no slices (functionally absent from the field).
- **Stack heights**: derivable from contents but cached for efficiency.

### 3.2 Buffer

- **Slot occupancy**: 5 slots, each either empty or holding a bucket.
- **Per-bucket data** (for occupied slots): color, capacity (slices required to fill, default 25 per `HEXFALL_RULES.md` §4), current fill count.

### 3.3 Reserve

- **Grid topology**: square grid of cells (4-adjacency, per `HEXFALL_RULES.md` §5). Static positions.
- **Per-cell content**: one of {empty, plain bucket, ?-bucket, generator, wall}.
  - Plain bucket: color (already known to simulator).
  - ?-bucket: color (fixed per-level, hidden from agent until revealed — see §4.3) and a "revealed" flag.
  - Generator: facing direction, remaining count, predetermined output queue (color sequence).
  - Wall: no per-cell data. Walls are permanent obstacles, never picked, never removed, and block reachability through their cell (per `HEXFALL_RULES.md` §5).

### 3.4 RNG state

A pseudo-random number generator state used exclusively for **fall direction** decisions (the only stochastic transition; see `HEXFALL_RULES.md` §8). Including the seed in $s_t$ makes the transition function deterministic conditional on the seed, which enables exact replay of trajectories — essential for debugging and for the difficulty-oracle workflow described in `PROJECT_STATE.md`.

Other stochastic-looking elements (?-bucket colors, generator outputs) are **not drawn from this RNG**; they are baked into the level definition.

### 3.5 Quiescence flag

A boolean indicating whether the simulator has finished applying all pending automatic updates and is awaiting a player action. The agent only acts at quiescent states (§5.4); non-quiescent states are intermediate and not observed.

---

## 4. Observation

The observation function $\Omega : S \to O$ produces what the agent sees from a state. It is deterministic given the state (no observation noise beyond what's already captured in hidden state).

### 4.1 Visible hex field

Per `HEXFALL_RULES.md` §3 (Visibility), each slice in the field is visible iff:

- It is in the bottom row of stacks (all slices in a bottom-row stack are visible), **or**
- It is in the portion of a stack that sticks up above its lower neighbor (the "exposed shoulder"), **or**
- It is in an edge-column stack of an alternating row that is fully visible from the side, **or**
- It is the top slice of any stack.

Slices not satisfying any of these conditions are hidden. The observation represents hidden slices with a distinguished "unknown" token, **not** by omission — the agent knows a slice exists at that position but not its color.

Stack heights are visible (the silhouette of the field is always observable).

### 4.2 Visible buffer

The buffer is fully observable: slot occupancy, bucket colors, capacities, and current fill counts.

### 4.3 Visible reserve

- Plain bucket cells: color visible.
- ?-bucket cells: color hidden ("?") if `revealed = false`; color visible if `revealed = true`.
- Generator cells: facing direction visible, remaining count visible. **Spec decision (flag in §10):** generator output queue (the colors of buckets it will produce next) is treated as **hidden** in the observation, since `HEXFALL_RULES.md` does not state that the player sees generator output ahead of time. Only the color of an already-produced bucket is visible.
- Wall cells: visible as walls. No hidden data.
- Empty cells: visible as empty.

### 4.4 ?-bucket revelation

Per `HEXFALL_RULES.md` §5: a ?-bucket's color is revealed at the moment it becomes **reachable** (pickable), not when it is picked. Operationally, the simulator sets `revealed = true` for any ?-bucket transitioning from unreachable to reachable, and the observation function reflects this in the next observation.

### 4.5 Reachability information

The observation includes, for each reserve cell, whether the cell is currently reachable (per `HEXFALL_RULES.md` §5 reachability rules). This is fully observable — reachability is a deterministic function of the visible reserve grid.

### 4.6 Action mask

The observation includes a boolean mask over the action space (§5) indicating which actions are legal in the current state. Equivalent information could be derived by the agent from the reachability info plus buffer state, but providing it explicitly is standard practice and removes a redundant learning burden.

---

## 5. Action

### 5.1 Action space

The action space $A$ is the set of all reserve grid cells:

$$A = \{(i, j) : 0 \le i < H_{\text{reserve}}, \, 0 \le j < W_{\text{reserve}}\}$$

where $H_{\text{reserve}}$ and $W_{\text{reserve}}$ are the reserve grid dimensions (level-dependent constants).

The action $(i, j)$ means "pick the bucket at reserve cell $(i, j)$."

The action space has fixed size per level. Across levels, the size varies — generalization to varying action-space size is a training concern, not an MDP-spec concern.

**Spec decision:** Fixed-size action space + action mask is preferred over dynamic action sets, because it integrates cleanly with PPO and standard RL libraries. The mask in §4.6 carries the legality information.

### 5.2 Action legality

An action $(i, j)$ is **legal** in state $s_t$ iff:

1. Cell $(i, j)$ contains a plain bucket or a ?-bucket (not empty, not a generator, not a wall), **and**
2. Cell $(i, j)$ is reachable per `HEXFALL_RULES.md` §5, **and**
3. The buffer has at least one empty slot.

Picking a generator is never legal (generators are not buckets). Picking a wall is never legal (walls are obstacles, not buckets). Picking an empty cell is never legal. Picking an unreachable cell is never legal. Picking when the buffer is full is never legal.

### 5.3 No-op

There is no no-op action. The agent only chooses actions at quiescent states (§3.5). Any quiescent state with zero legal actions is a terminal-lose state per §7.2, so the agent never faces a non-terminal quiescent state with nothing to do.

### 5.4 Transition

A single MDP step proceeds as follows:

1. **Action application.** The selected bucket is removed from the reserve and placed into an empty buffer slot. Slot choice is irrelevant per `HEXFALL_RULES.md` §4 (slot ordering does not matter).
2. **Automatic updates.** The simulator runs automatic updates until quiescence. Updates fire in the following order each tick (spec-level decision; see §10):
   1. **Buffer pulls.** Each bucket in the buffer attempts to pull a matching top slice from the bottom row. Same-color collision rule applies (`HEXFALL_RULES.md` §4): the fuller bucket of a same-color pair pulls before the less-full one. Multiple distinct-color buckets pulling from different stacks all pull on the same tick.
   2. **Bucket fill checks.** Any bucket that reached capacity leaves the buffer, freeing its slot.
   3. **Stack clear checks.** Any bottom-row stack that is now empty triggers a fall: one of its (up to two) upper neighbors is chosen uniformly at random (using the RNG in §3.4) and falls into the empty position.
   4. **Generator firing.** Any generator whose facing cell is now empty (and which has remaining count > 0) produces its next bucket into that cell, decrementing its count.
   5. **Reachability recomputation.** Reachability is recomputed; any newly-reachable ?-bucket has its color revealed.
3. **Quiescence check.** If any update fired in step 2, repeat step 2. Otherwise the state is quiescent and the agent observes and acts again.

The transition is stochastic only via step 2.iii (fall direction).

**Deterministic tie-breaks.** Two situations require a tie-break that the rules doc and earlier spec versions did not pin down. The simulator must use these specific rules to preserve replay determinism:

- **One bucket, multiple matching stacks.** If a single active bucket can pull from multiple bottom-row stacks on a tick (i.e., its color appears as the top slice on two or more bottom-row stacks), the simulator pulls from the stack with the smallest column index, breaking ties by smallest row index. No RNG is consumed.
- **Same-color buckets with equal fill counts.** If two buckets of the same color have identical fill counts (including both at zero), the bucket in the lower-indexed buffer slot is treated as the "fuller" one for the same-color collision rule (`HEXFALL_RULES.md` §4). No RNG is consumed.

Both rules are deterministic, preserve replay correctness, and were locked during simulator implementation (May 9, 2026). They are part of the MDP contract, not implementation freedom.

**Initial load.** Level loading runs the same automatic-update loop (step 2 above) before returning the first observation to the agent. This means the initial observed state is always quiescent. In particular:

- If a generator's facing cell is initially empty, it fires on load (per step 2.iv).
- If a generator's facing cell is initially occupied (e.g., by a bucket, wall, or another generator), it does **not** fire — the empty-facing-cell guard applies on load just as it does mid-episode.
- If any other automatic updates apply at load (e.g., reachability computation, ?-bucket revelation), they run as part of the same loop until quiescence.

Real Hex Fall levels do not contain initial empty cells in the reserve, so on real levels no generator fires at load. The load-time update loop matters for hand-built test levels and procedurally generated levels that may include initial empties.

---

## 6. Reward

**Placeholder. To be refined when RL training begins (see `WEEK_1_PLAN.md` step 2 and `PROJECT_STATE.md`).**

The minimal reward structure assumed for now:

- $+1$ on transition into a terminal-win state.
- $-1$ on transition into a terminal-lose state.
- $0$ on all other transitions.

This is intentionally sparse. Curiosity-driven intrinsic rewards (`RL_CONCEPTS_SUMMARY.md` §3) will supplement this signal during training; they are computed by the curiosity module and added to the extrinsic reward, but they are **not part of the environment's reward** — they are part of the agent's training pipeline. The MDP defined here returns only extrinsic reward.

Possible additions to consider during reward-shaping work (do **not** implement now):

- Small positive shaping for filling a bucket.
- Small positive shaping for clearing a stack.
- Small negative shaping per step (encourages efficiency).
- Negative shaping for committing a bucket whose color is rare in remaining slices (anti-deadlock pressure).

Decisions on shaping belong to the training-design phase, not this spec.

---

## 7. Termination

### 7.1 Win

Episode terminates with reward $+1$ when the hex field is empty. Per `HEXFALL_RULES.md` §7, by slice-bucket parity this also implies all reserve buckets have been consumed and the buffer is empty.

### 7.2 Lose

Episode terminates with reward $-1$ at any quiescent state with **zero legal actions**.

This single condition subsumes both forms of stuckness:

- **Rules-doc deadlock** (`HEXFALL_RULES.md` §7): all 5 buffer slots occupied, no buffer bucket can pull from the bottom row, no reserve bucket can be picked. Buffer-full implies no legal actions per §5.2 condition 3.
- **Defensive fallback:** any other quiescent state with no legal actions, e.g., buffer has empty slots but no reserve bucket is currently reachable and no automatic update will ever make one reachable. Should not occur on well-formed levels (slice-bucket parity + Paxie level construction discipline) but can occur on procedurally generated levels with bugs.

**Telemetry on fallback termination.** When a lose terminates an episode that does *not* match the rules-doc deadlock pattern (i.e., the buffer was not full), the simulator emits:

- A console warning ("fallback termination on level X at step N").
- A structured record in a persistent per-run log file (level ID, step number, buffer/reserve state snapshot).
- `info["termination_reason"] = "fallback"` in the Gymnasium `info` dict returned by `step()`. Normal-deadlock losses set `info["termination_reason"] = "deadlock"`; wins set `info["termination_reason"] = "win"`.

The telemetry exists because fallback firings are signals of bugged levels (especially during procedural level generation in later phases). It is not used by the agent's policy.

### 7.3 Episode time limit

**Spec decision (flag in §10):** No hard time limit is part of the MDP definition (`HEXFALL_RULES.md` §3 explicitly states no move limit). For training purposes, the simulator may impose a step-count cap that produces a "truncated" episode (in Gymnasium terminology: `truncated = True` rather than `terminated = True`). The cap value and whether truncation applies a reward penalty are training-design decisions, not MDP-spec decisions.

---

## 8. Determinism and replay

A trajectory is reproducible iff the following are recorded:

- The level definition (deterministic per `HEXFALL_RULES.md` §8: hex layout, slice composition, reserve layout, generator outputs, ?-bucket colors).
- The RNG seed for fall direction (§3.4).
- The sequence of agent actions.

Given these, replay is bit-exact. This property is required for:

- Debugging simulator bugs.
- Reproducing failure cases for the difficulty-oracle workflow.
- Comparing agent performance across training checkpoints on identical episodes.

Implementations must take care not to introduce hidden RNG calls (e.g., dict ordering, set iteration on Python objects with non-deterministic hash) that bypass the declared RNG. Tests should include "replay an action sequence twice with the same seed and verify identical state at every step."

---

## 9. Summary of MDP signature

For quick reference:

| Component | Type | Notes |
|-----------|------|-------|
| State $S$ | structured tuple | Field, buffer, reserve, RNG, quiescence flag (§3) |
| Observation $O$ | structured tuple | Visible field + full buffer + visible reserve + reachability + action mask (§4) |
| Action $A$ | discrete, fixed-size | Reserve grid cells, with action mask (§5.1) |
| Transition $P$ | stochastic | Single source: fall direction (§5.4) |
| Observation function $\Omega$ | deterministic | Geometric visibility + ?-bucket reveal (§4) |
| Reward $R$ | sparse | Placeholder: ±1 terminal, 0 else (§6) |
| Termination | win / lose / (truncated) | §7 |

---

## 10. Open questions and deferred decisions

This section, as drafted in the initial spec, flagged 9 items for commander review. As of the May 5, 2026 review, 6 are resolved and folded into the relevant sections; 4 remain deferred to later phases. The breakdown is below.

### 10.1 Resolved (May 5, 2026 commander review)

The following items from the initial draft are resolved. The listed sections of this spec reflect the resolutions:

1. **Tick ordering of automatic updates** — confirmed as drafted (§5.4): pull → fill check → fall → generator fire → reachability recomputation, repeating until quiescence.
2. **Generator output queue visibility** — not visible. Only the count and facing direction are observable; the upcoming color sequence is hidden state. Reflected in §4.3 and added to `HEXFALL_RULES.md` §5.
3. **Lose condition coverage** — resolved by simplification: §7.2 now defines lose as "zero legal actions at a quiescent state," subsuming the rules-doc deadlock as a special case. Telemetry distinguishes fallback firings from normal deadlocks for post-hoc level-quality analysis.
4. **Same-color same-tick pulls from different stacks** — confirmed: less-full bucket waits entirely while a fuller same-color bucket is in the buffer, even from a different stack. Clarified in `HEXFALL_RULES.md` §4.
5. **Multiple same-tick stack clears** — confirmed: each clearing stack independently rolls fall direction. No interaction between concurrent falls; they are independent random choices.
6. **Exhausted-generator reachability** — confirmed: an exhausted generator (count = 0) still occupies its cell and blocks reachability through it. Matches `HEXFALL_RULES.md` §5 and is unchanged here.

### 10.2 Deferred to later phases

These items are not blocking for the simulator implementation. They will be resolved when the relevant phase begins.

1. **Action space size across levels** (training-design concern; Week 3+). Cross-level training requires either padding the action space to a max size, masking, or alternative architectures. Decision belongs to training design.
2. **Truncation policy** (training-design concern; Week 3+). Step-count cap value and reward treatment for truncated episodes. Decision belongs to training design.
3. **RNG scope expansion** (deferred indefinitely). The spec assumes a single RNG used only for fall direction. If randomized initial conditions or other stochastic elements are added later (e.g., for cross-level generalization), the RNG model needs revisiting.
4. **Color count distribution per level** (empirical; carried forward from `HEXFALL_RULES.md` §10). The maximum color count per level must be known to the simulator at level-load time; the typical distribution will be confirmed empirically once level data is available.

---

## 11. Revision history

- **May 5, 2026:** Initial draft. 11 sections. Created in worker chat for the Week 1 MDP-spec issue. Reward function is a placeholder per the brief; final shaping deferred to Week 3 training-design phase. Six spec-level decisions (§10.2) and three rules-doc ambiguities (§10.3) flagged for commander review.
- **May 5, 2026 (commander review):** Resolved 6 of the 9 flagged items. §7.2 (lose) simplified to "zero legal actions at quiescent state," subsuming the rules-doc deadlock as a special case; telemetry added (console + log file + `info` dict) so fallback firings are detectable as signals of bugged levels. §5.3 updated to remove the §7.2 open-question reference. §4.3 (generator queue visibility) confirmed as drafted. §10 restructured: resolved items moved to §10.1 with cross-references; deferred items consolidated in §10.2. Two resolutions also flowed back to `HEXFALL_RULES.md` (generator queue visibility, same-color same-tick pull behavior).
- **May 5, 2026 (second pass):** Added walls as a fifth reserve cell type (§3.3, §4.3, §5.2). Walls are permanent obstacles, never picked, never removed, and block reachability through their cell. Initially missed in both `HEXFALL_RULES.md` and this spec; identified from a level 38 screenshot during planning of the level-format issue. Action legality (§5.2) explicitly excludes walls.
- **May 7, 2026:** Added "Initial load" paragraph to §5.4 clarifying that level loading runs the same automatic-update loop until quiescence, with the empty-facing-cell guard applying on load (a generator does not fire at load if its facing cell is occupied). Resolves an open question flagged in the LEVEL_FORMAT.md worker deliverable. Real Hex Fall levels never have initial empty reserve cells, so no generator fires at load on real levels; the clarification matters for hand-built and generated test levels.
- **May 9, 2026:** Added "Deterministic tie-breaks" paragraph to §5.4 documenting two rules locked during simulator implementation: (1) when one bucket has multiple matching stacks on a tick, pull from smallest-col then smallest-row; (2) when same-color buckets have tied fills, lowest-indexed buffer slot wins the same-color collision priority. Both rules preserve replay determinism without consuming RNG. Flowed back from simulator-implementation worker review (issue #3 closure).
