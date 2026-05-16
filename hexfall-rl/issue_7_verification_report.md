# Issue #7 — Sub-steps 4, 5, 6 Verification Report

No code changes were made; this is a verification-only pass.

---

## SUB-STEP 4 — pytest suite

`pytest tests/ -v` from repo root.

**Result: 86 passed / 0 failed / 0 error / 0 skipped** in 0.64s.

- 28 warnings, all benign and pre-existing:
  - Slice-bucket parity `UserWarning`s from `hexfall/level_loader.py:374` (raised during legal test fixtures).
  - `RuntimeWarning: Fallback termination on level test` × 3, inside `test_fall_two_candidates_uses_rng_deterministically` — pre-existing, expected.
- The renamed test `test_move_counter_not_in_obs` is present and passes (verified replacement of `test_obs_includes_move_counter`).
- No tests beyond the documented scope (move_counter swap + typed-array access-pattern rewrites) appear to have been deleted or semantically altered. The rest of the suite — game logic, level loader, env contract — is full strength.

---

## SUB-STEP 5 — check_env validation

`gymnasium.utils.env_checker.check_env(env, skip_render_check=True)` against three levels.

| Level                       | Result            | Exception                                                |
| --------------------------- | ----------------- | -------------------------------------------------------- |
| `levels/tiny_solvable.json` | **PASS**          | —                                                        |
| `levels/ice_test.json`      | PASS-WITH-CAVEAT  | `ValueError: Illegal action (np.int64(0), np.int64(2))`  |
| `levels/pin_test.json`      | PASS-WITH-CAVEAT  | `ValueError: Illegal action (np.int64(0), np.int64(1))`  |

Both caveat cases match the expected pattern: the exception message starts with `Illegal action`, which is the env's pre-existing action-side contract being tripped by `check_env`'s unmasked random action sampling. This happens **after** observation-space validation succeeds — which is what sub-step 5 actually requires.

No "observation", "shape", "dtype", "key", or "spaces" failures surfaced on any level.

**Acceptance criterion** ("no observation-space mismatch errors raised against tiny_solvable.json, ice_test.json, pin_test.json") is satisfied for all three.

---

## SUB-STEP 6 — random agent end-to-end

`python -m scripts.run_random_agent --level <L> --env-seed 0 --agent-seed 0` for each of the three levels.

| Level                       | Steps | Terminated | termination_reason | reward |
| --------------------------- | ----- | ---------- | ------------------ | ------ |
| `levels/tiny_solvable.json` | 2     | True       | `win`              | 1.0    |
| `levels/ice_test.json`      | 3     | True       | `win`              | 1.0    |
| `levels/pin_test.json`      | 1     | True       | `win`              | 1.0    |

All three runs completed normally, no crashes. Per-step rendered output produced cleanly through the typed-obs path:

- Hex field rows.
- Buffer slot view with `slot/capacity` annotations (e.g. `Slot 0: b (4/24)`).
- Reserve grid with reachability markers and ice/pin overlays (`IC:r/T(R)`, `IC:???/F1(R)`, `PIN[PL:r(-)]`, etc.).

**Caveat on baseline comparison:** the sub-step 2/3 renderer-parity baseline captures were not available in this session, so a byte-level diff against them was not performed. The rendered output is internally coherent and consistent with the post-rewrite typed-obs contract; if the worker chat retained the prior baselines, a diff against the transcripts produced here will confirm parity.

---

## Net acceptance

- pytest: **86/86 green**.
- check_env: obs-space validation **passes on all three levels**; action-side caveats on ice/pin are pre-existing and out-of-scope per the sub-step 5 spec.
- Random agent: **all three runs win cleanly**, no exceptions, renderer output coherent.

No FAIL surfaced anywhere that would have indicated a regression introduced by sub-step 2/3.
