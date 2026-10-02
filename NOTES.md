# NOTES.md — measurements and post-mortems

Evidence behind the rules in `CLAUDE.md`. Read this when you are tempted to change something that
looks redundant, over-engineered, or slow — most of it is here because the obvious simplification was
tried and measured worse, or because a defect returned a plausible wrong number rather than raising.

`CLAUDE.md` carries the rules and points here for the numbers. Nothing in this file is actionable on
its own; if a rule and a note disagree, the code is the arbiter and both are stale.

## Testing: why the suite is shaped the way it is

### The suite runs in ~6 s, not ~53 s, and the caches cannot mask a defect

`conftest.py` sets `MPLCONFIGDIR` and `JAX_COMPILATION_CACHE_DIR` (plus
`JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0`, required because the default 1 s threshold excludes
every kernel here — the largest single compile is ~0.44 s) before importing jax, with
`os.environ.setdefault` so a value you exported always wins. Both are speed-only, unlike the x64
flag: nothing depends on them, and a warm cache was verified *unable* to mask a defect — reverting
`_is_filler`'s `>> 7` to `>> 8` still fails the one test that catches it, since XLA keys on the
computation. Expect ~53 s on a first run while the caches populate.

**Don't chase test-body slowness.** Measured, 72% of an uncached run is XLA compilation plus a
matplotlib font-cache rebuild (`~/.matplotlib` is unwritable in a sandbox, and matplotlib's own
fallback is a fresh temp dir per process, which never warms). The `range(2000)` accumulation loops in
`test_ground_locg.py` that look expensive total 0.80 s, 2.3% of the suite — leave them alone.

### Why fixtures are built inside test bodies

Several tests pick a seed to produce a specific pathology (a decoupled seed state, a subspace that
splits into two blocks, 13 Z terms in one X group) and assert the fixture still has it. Moving draws
into `@pytest.fixture` generators would make RNG stream position depend on fixture ordering, which is
invisible at the call site — `conftest.collapsing_states`' docstring records a measured instance
(changing a preceding `real_pauli_strings` count from 5 to 6 moved a collapse from 7 uniques to 9).

Only `collapsing_states` asserts its own precondition: `np.unique` makes `unique_states` distinct by
construction, but its *row count* varies with the seed (7 draws over 4 qubits measured 3–7 distinct
rows across 200 seeds), so a caller needing a floor must assert it.

### The padding-test trap, worth reading before writing any "does X change the answer?" test

`test_states_size_padding_is_shape_invariant_only` compares padded `sqd` calls against an *unpadded*
one, and **cannot** catch a broken filler mask: its fixture is 12 random 4-bit states, which collapse
to fewer uniques, so even the "baseline" arm already carries filler slots and is corrupted
identically. Both sides drift together and it passes.

Measured — changing `_is_filler`'s `states_u[:, 0] >> 7` to `>> 8` (a uint8 shifted by 8 is 0, so
every filler reads as a genuine state) left the whole `sqd` suite green while `sqd` returned −1.2
against a true −0.8297058541. Same for deleting `run_sqd`'s filler-diagonal masking.
`test_filler_slots_are_excluded_against_a_dense_reference` closes both, using a fixture that is
*already unique* (so `states_size=None` is a genuinely filler-free control) and a dense reference.

So: **a "does X change the answer?" test needs an arm where X is truly absent — verify that, don't
assume it from the parameter being unset.**

### Mutation-testing recipe

The highest-yield tool in the repo — it found two silent coverage gaps and a false docstring claim in
one session. Copy the file (`cp rqutils/sqd.py /tmp/x.bak`), rewrite one line with a Python one-liner
that **asserts the anchor string exists** before substituting, run the suite, restore from the copy,
and `diff -q` to prove the restore took.

The assert is not optional: a silent no-match reports a false "no coverage", indistinguishable from
the finding you are looking for. Two further traps:

- For a `@jax.jit`-decorated function, mutate in a **fresh subprocess**. Patching in a live session
  reuses the compiled kernel and both arms return bit-identical numbers (`test_ground_locg.py`'s
  `TestBasisOrthogonality` records this).
- Check the mutant is *reachable* before concluding a guard is untested. Some survive because the
  fixture never exercises them, which is a fixture finding, not a missing test.

Verify a new test actually fails against the bug it targets by reverting the fix **in place**; a copy
of the repo does not work, since the venv holds an editable install pointing at the original.

**The same trap applies to patch scripts, not just mutations.** A `str.replace` whose anchor `assert`s
present can still write nothing you notice, because the anchor may no longer match what is on disk --
`ruff format` reflows lines, and a `CLAUDE.md`/`NOTES.md` split moves prose between files, so an anchor
copied from your own earlier draft goes stale. Two measured consequences from one session: a duplicate
test was deleted while the assertion meant to replace it never landed (a net coverage **loss** that
reported success), and a multi-anchor script aborted midway while I assumed the earlier anchors had
applied -- they had not, since the write happens at the end. So: put the `open(..., "w")` last so a
failed assert changes nothing, verify each edit landed rather than inferring it from "ok", and
mutation-test the *surviving* assertion afterwards. Note `pytest` prints "no tests ran" rather than
failing when the class path is wrong, so a mis-copied class name looks like a pass.

### The prefilter's only `sqd`-specific sharding case: padded subspace meets partitioned vector

2026-08-28. `test/sharded/locg_prefilter.py` covers the prefilter on a mesh, but only through
`ground_locg` with a dense `einsum` matvec on an unpadded power-of-two vector —
`markdown/locg-chebyshev-prefilter.md` said so and deferred the rest to `sqd`. That deferral is now closed
by `test/sharded/sqd_grid.py`.

What is only reachable through `sqd`: a **padded** subspace whose filler slots are masked to zero,
partitioned across a mesh, driven through `apply_h`'s gather-heavy irregular kernel rather than a dense
matmul — and the filter calls that matvec `cycles * (degree + 1)` times before the solver's first
iteration, so a fault there gets far more exposure than one LOBPCG step gives it. 37 genuine states pad
to 64, which 2 and 4 both divide; that arrangement cannot occur in the `ground_locg` harness.

**Value agreement proves nothing here.** All 18 energy cases agree to 4e-16 or better regardless of
whether partitioning survives, so the child also prints the prefilter's output spec and the test
asserts `filtered_spec == vinit_spec`, plus that the partitioned arm really is `P('x',)` — without that
last guard a harness that quietly stopped partitioning would pass. Both guards are mutation-verified
(a `jax.reshard(out, P(None))` in the filter; a harness that never partitions).

### A `maxiter=1000` non-convergence usually means a small gap, not a bad subspace

2026-08-28. Two of ~300 random stress fixtures raised `RuntimeError: LOBPCG did not converge`, which
read as a defect. It is not one, and the distinction is worth keeping because the error message's own
advice ("check that the subspace is well conditioned") pointed the wrong way.

The fixture: 4 Pauli terms over 6 qubits, 37-state subspace, **relative gap 5.5e-04** — the three
lowest excited states degenerate to 4e-16, sitting 3.2e-03 above the ground state. LOBPCG's
*eigenvalue* converges quadratically while its *eigenvector* converges at a rate set by the gap, so:

| iteration | `theta` error | residual | converged |
| --- | --- | --- | --- |
| 100 | 7.6e-05 | 2.1e-03 | no |
| 500 | **4.9e-12** | 6.9e-07 | no |
| 999 | **4.4e-16** | 2.4e-11 | no |
| 1091 | 4.4e-16 | 8.6e-14 | **yes** |

So the answer was already at machine precision by iteration ~500 and the residual test only cleared at
**1091**, just past the default cap. `maxiter=2000` returns it, correct to 4.4e-16. `scipy.eigsh(tol=0)`
agrees. The rate law predicts ~491 iterations from `ln(1e10) / (2*sqrt(relgap))`, same order as
observed (block-size-1 is slower than that two-sided bound).

**Rare rather than systematic:** 0 of 140 further random subspaces failed at the default, *including 18
with a relative gap below 1e-4*. So don't raise the default cap on the strength of this — raise
`maxiter` at the call site. The message now says so.

Pinned by `test_sqd.py::TestConvergenceIsReported::test_near_degenerate_subspace_needs_maxiter_above_the_default`,
with the 37 basis states written out as integers. A seed-based redraw does **not** work: a nearby seed
measured a relative gap of 8.7e-03, too well-gapped to reproduce the raise.

### Indexing a *sharded* array to read one element emits an `all-gather` of the whole vector

2026-08-28, found by a cleanup review of the `vinit_from_min_diag` fix. Writing the sign weight as
`seed.at[imin].add(jnp.sign(seed[imin]))` is correct arithmetic but reads one element out of a
partitioned array. Measured on a 4-device mesh with `PartitionSpec('x')`:

| form | `all-gather` | `dynamic-slice` | HLO lines |
| --- | --- | --- | --- |
| `seed[imin]` indexed | **3** | 5 | 70 |
| elementwise mask | **0** | 3 | 57 |

An `all-gather` materializes the entire `states_size` vector on every device — at `_MAX_STATES`
(2³¹−1) that is precisely the full-vector collective `ground_locg`'s single-vector memory budget
exists to avoid. End to end through `run_sqd` on a 4-device mesh: 27 → 24 all-gathers.

The fix is a `broadcasted_iota` mask and an elementwise `where`, bit-identical at every `imin` tried.
**Whole-suite A/B: 20.4 s indexed vs 20.1 s masked** — no cost. Beware when measuring this: switching
the form invalidates the compilation cache, and a cold run measured 125 s against a warm 20 s, which
reads as a catastrophic regression and is not one. A/B both arms warm.

### Validation belongs to the module that owns the gate

Same review. `_check_prefilter` lived in `rqutils/sqd.py`, so `sqd(prefilter=(2, -1))` raised while
`ground_locg(prefilter=(2, -1))` — the *published* entry point, and the one whose docstring tells
callers to A/B the option — absorbed it. Measured: `(1.5, 2)`, `(-4, 2)`, `(True, 2)` and `(2, -1)` all
silently no-op'd there, and `"32,2"` / `32` leaked the internal tuple-unpack error out of a public
entry point. Moved to `ground_locg.py`, which owns the `degree > 1 and cycles > 0` gate the check
compensates for; `sqd.py` imports it, matching the existing dependency direction.

The array path's auto-derived bound had the same asymmetry: gated only on `prefilter is not None`, it
computed the O(N²) Gershgorin reduction even for the degenerate no-ops, measured **+6.1%** on a
2048-dim solve at `prefilter=(16, 0)`. Both paths now use the same tighter predicate.

### `vinit_from_min_diag`'s weight must carry the seed's sign, or it cancels it

2026-08-28, found while validating the prefilter fix against sampled `Bx = 0` subspaces, and
**independent of the prefilter** — it affected every `sqd` call on every revision that has
`_spread_seed`.

The heuristic added a bare `+1.0` at `argmin(diagonal)` on top of the spread seed. That *subtracts*
wherever the seed component is negative, and `_spread_seed`'s Murmur-style mixer maps index 0 to
**exactly −1.0** at every `states_size` (the mixer fixes 0; `mixed * 2/2**32 − 1` sends that to −1.0).
So `argmin(diagonal) == 0` zeroed the component at precisely the index the heuristic had just declared
the best available guess, violating `ground_locg`'s non-vanishing-overlap precondition.

Measured: a 2-state subspace of the `Bx = 0` n=4 Heisenberg chain returns **−0.25 against a true
−0.75**, `converged=True` in **0 iterations** — the projected operator is `diag(−0.75, −0.25)`, so the
surviving component is already an eigenvector. 1 in 18 randomly sampled `Bx = 0` subspaces at n=4–8 hit
it; 0 of 676 after the fix.

Fixed as `seed.at[imin].add(jnp.sign(seed[imin]))`, so the update reinforces and `|vinit[imin]|` lands
in `[1, 2)` for any seed. `jnp.sign` not `copysign` — the seed is complex whenever `hamiltonian.c` is.

**Why structural rather than a special case on index 0:** exact cancellation is only reachable there,
but **511 of `2**20` indices carry a seed within 1e-3 of −1.0**, each of which would lose all but a
thousandth of the component. That is a slow-convergence or wrong-answer risk nothing would have
attributed to this line.

The `sign(0) == 0` fallback is **unreachable, and provably so**: the mixer is a bijection on uint32, so
exactly one index yields a 0.0 seed, and it is **3906290832** — above the `_MAX_STATES` ceiling of
`2**31 − 1` that both entry points enforce. A mutant removing it survives; that is not dead code, it is
a guard whose reachability the ceiling currently forecloses.

### No matvec-only upper bound on `λ_max` exists, so the prefilter takes one from structure

2026-08-28, fixing `markdown/spinchain/rqutils-prefilter-bug.md`. `_lambda_max_bound` used 10 power steps, which
converge to the largest-*magnitude* eigenvalue; on a negative-leaning spectrum that is `λ_min`, the
Chebyshev interval inverts, and the filter damps its own target — an **excited** eigenpair returned
with `converged=True` (n=2 Heisenberg: +0.25 for a true −0.75).

Candidates measured before settling on structural bounds:

| bound | XXZ 25 | adversarial | matvec-only |
| --- | --- | --- | --- |
| `abs(estimate)` (the report's own fallback) | 21/25 | — | yes |
| `sqrt` of power iteration on `A²` | 24/25 | — | yes |
| Lanczos `μ_max + β_k` | **25/25** | **988/1000** | yes |
| EVSL's `μ_max + |β_k s_k|` | — | **3433/4000** | yes |
| Gershgorin `max_i Σ_j |A_ij|` | 25/25 | rigorous | no |
| `Σ|c_k|` for a Pauli sum | 25/25 | rigorous | n/a |

Lanczos looking perfect on the physics cases and failing adversarially is the trap that let the
original bug ship. **The impossibility is a theorem** — Kuczyński & Woźniakowski, SIAM J. Matrix Anal.
Appl. 13(4):1094–1122 (1992): with fewer than `N` matvecs another operator consistent with every
observation has an arbitrarily larger `λ_max`. Constructively, block-diagonal `A` with a start vector
inside one block gives a true 1000.0 against a Lanczos bound of 4.68, and 16 random restarts do not
help (200/200 invalid) because they share the invariant subspace. Cauchy interlacing makes the top
Ritz value a *lower* bound, so no `abs()` converts it.

**Looseness: measure the overlap, not the iteration count.** Iteration counts were identical from 1×
to 1660× the coefficient sum, which read as "looseness is free" — twice, in two separate measurements.
It is not: the filtered vector's ground-state overlap falls 0.78 → 0.094 → 0.018 at 1.6× → 415× →
6600× `λ_max`, matching the `sqrt(width)` law for the degree needed. Iteration counts hid it because
the prefilter only has to reach the ground state's basin before LOBPCG takes over. Hence `Σ|c_k|`
(~1.8× loose) rather than a deliberately inflated fallback — but loose still beats tight, since
over-estimating degrades smoothly while under-estimating changes the answer.

Removing the estimate also removed its ~11 matvecs: the iteration reduction *improved* to a 3.29×
median from the 1.88× recorded with the unsound bound.

### A regression test for a filter bug needs the seed that fails

Same fix. `test_negative_leaning_spectrum_finds_the_ground_state` first used seed 20260828 and
**passed against the unfixed code**. The bound is invalid (−0.7125 against a true `λ_max` of +0.25) for
every seed tried, but whether the ground state also lands inside the damped band is what varies: seeds
0 and 2 return +0.25, seeds 1, 3 and 20260828 return the correct −0.75. Only a full revert of
`ground_locg.py` — not a hand-written mutant of the bound — surfaced this, because the mutant fed the
power iteration a different start vector than the original did.

### `sqd`'s filler slots are protected by unreachability, not by `_spread_seed`'s mask

Found while pinning `TestSqdPrefilter` (2026-08-28). Removing the `jnp.where(filler, 0, vec)` mask in
`_spread_seed` leaves **every** assertion in that class green, at every `states_size`. That is not a
missing test — it is the guard being redundant with something stronger.

Probing the padded operator directly (40 draws over 4 qubits collapsing to 14 uniques, padded to 64, so
50 filler slots):

- Both coupling blocks are **exactly 0.0** — genuine rows never pull from filler rows and vice versa, so
  the padded operator is block-diagonal and the filter cannot move weight across the boundary.
- `apply_h` is **asymmetric** on filler rows (98 entries where `|H - H.T| > 1e-12`).
- The filler block's own lowest eigenvalue is **−10.59**, far below the genuine block's **−4.33**.

The asymmetry is what saves it. A symmetric block at −10.59 would be a legitimate lower eigenvalue for
LOBPCG to find; because `apply_h` is asymmetric there, that spectrum is **unreachable** rather than
merely unfavoured. Measured: an iterate started *entirely* inside the filler block converges to −4.23
and reports `converged=False`, and the unmasked spread seed — which puts *more* weight on filler slots
(norm 4.47) than on genuine ones (2.15) — still returns the correct −4.330397418179033.

So the mask is defence in depth, and no energy assertion can pin it. Don't record it as dead code on
the strength of a green suite, and don't claim a filler test covers it.

### Comparing energies cannot prove an option is a no-op

Same session. `TestSqdPrefilter`'s degenerate-value test first asserted that `prefilter=(1, 4)`,
`(16, 0)` and `(0, 0)` gave the baseline energy. It passed — and pinned nothing: on that fixture a
*working* `(16, 1)` also returns a bit-identical energy, and only `(32, 2)` moves the last ulp
(−3.533932511396396 against −3.533932511396397). A mutant coercing `cycles=0` to `1` in `run_sqd`
survived the energy form and dies against `str(jax.make_jaxpr(...))` equality.

Two mutants also survived by being aimed at the wrong layer: a coercion added to `sqd` while the test
traced `run_sqd` directly, and a mutation of `vinit_nodiag` when that fixture takes
`vinit_from_min_diag`. Both read as missing coverage and were not.

### A green suite after reverting a fix means the test is missing, not that the guard is dead

Some guards are only reachable when *other* defects compound with them, so the end-to-end assertion
(theta matches `eigvalsh`) stays green while the invariant the guard protects is already destroyed.

`ground_locg`'s `_reorthogonalize` (audit item I5) is the worked example, and a cautionary one: it was
twice recorded as unpinnable — first from an A/B compromised by the live-session jit trap above, then
from a *correct* A/B whose conclusion was drawn from the wrong assertion. Theta *does* survive, because
the 1.0 drift needed I4's 2000-iteration runs. The test that discriminates asserts the **invariant**
(`|<x|y>|`, straight off the `debug=True` diagnostics) rather than the end result, and fails 3 of 4
arms.

Before recording a negative result, check whether a *more direct* assertion exists; reach for the
docstring note only once it does not.

### A surviving mutant can mean the line is redundant in the *library* (2026-09-20)

The fifth reason a mutant survives, alongside the four in `CLAUDE.md`: the mutated line is an
early-out whose condition a later line already covers, so removing it changes speed and not one
answer. Nothing to pin, and a test written to pin it would be asserting the behaviour the *next*
branch produces.

`_is_lex_sorted`'s duplicate-row test is the worked example. It reads

```python
if not bool(np.all(np.any(differs, axis=1))):  # <- the early-out
    return False
first = np.argmax(differs, axis=1)
return bool(np.all(lhs[rows, first] < rhs[rows, first]))
```

and for a duplicate pair the final comparison is `32 < 32`, already False. Measured — replacing the
early-out's condition with `False` leaves all of `test_sqd.py` green (224 passed).

What made this worth recording is that a test *claimed* to cover it and could not. Old
`test_two_filler_rows_are_still_rejected` built two all-`255` rows and asserted `not
_is_lex_sorted(...)`; fillers are caught by the **high-bit check two lines earlier**, which returns
before the strictness pass runs, so the test was a second copy of
`test_one_filler_row_is_rejected` under a name promising otherwise. Both assertions pass either way,
which is why it survived review. It is now `test_duplicate_rows_are_rejected`, on a filler-free
fixture (byte 0 < 128) that genuinely reaches the sortedness pass and asserts the rejection there.

So: when a mutant survives, check whether a **later line in the same function** subsumes the mutated
one before recording missing coverage — and check what the fixture actually reaches, since an
*earlier* guard returning first is the mirror-image trap. Neither shows up in a green suite.

### Collapsing duplicate validation tests: parametrize, and prove it with the same mutant (2026-09-20)

Nine test functions across two modules each exercised one shared guard with a different input, which
is boilerplate rather than coverage — `_check_states_shape`'s single `states.shape[1] != num_qubits`
comparison had four (packed via `sqd`, packed via `hproj`, transposed, mismatched Hamiltonian),
`_check_cache_level` had three parametrized tests all asserting `match="cache_level"`, and
`pack_states`' binary check had two, one hiding a second input inside a bare `for` loop where a
failure would not name which value broke.

Collapsed to three parametrized tests, −53/+69 lines. The check that makes this safe rather than
lossy: disable the guard in `rqutils/` and confirm the **same set of failures** before and after —
7 kills for the width guard, 10 for `cache_level`, 4 for the binary check, identical both ways. The
distinct inputs survive as parametrize ids, so a failure still names the shape that broke.

Worth distinguishing from the reasoned overlaps elsewhere in this suite, which are **not** the same
thing and should not be collapsed: `test_fully_cached_level_matches_dense` says "overlaps by design"
because it is the positive control for `test_omitting_states_raises`, and
`test_filler_slots_are_excluded_against_a_dense_reference` is the only arm surviving the `>> 7` →
`>> 8` mutation. Those pin separate mutants; the nine above pinned one apiece.

### Sweep `cache_level`, don't sample it

Three bugs hid behind a single-cell check, each masked by the one before it — every existing sharding
check ran only `sqd`'s default `(1, 0)`:

1. `_accumulate_diagonal`'s rank-2 spec on a rank-1 accumulator (failed all six).
2. Once fixed: `_spread_seed`'s `jnp.where` mixing a replicated predicate with a partitioned `vec`,
   because `run_sqd` reshards `states_u` only inside `if cache_level[0] == 1` (failed `(0, *)`).
3. Once the sweep reached a *complex* fixture: `vinit_from_min_diag` using `diagonals[0]` raw where
   the uncached branch took `.real` (failed `(*, 2)` on any odd-Y Hamiltonian, **single-device, no
   mesh**).

Fixing each only revealed the next, so "the mesh test passes" meant very little until the grid was
complete. Note the last needed the *fixture* varied, not the parameter: the suite's
`real_pauli_strings` keeps the Y count even, so `.c` stays float64 and a six-cell sweep still
reported six passes.

### Multi-device paths are testable on CPU

`XLA_FLAGS=--xla_force_host_platform_device_count=4` gives virtual devices that exercise every
sharding code path (mesh detection, `PartitionSpec` propagation, `jax.reshard`, `sqd`'s mesh-size
padding) with no GPU. Not hypothetical: the first run found `sqd` raising `ShardingTypeError` on *any*
mesh, because one scatter omitted `out_sharding` while every neighbouring op passed it. Timings under
virtual devices are meaningless (they share one CPU) — correctness only.

### Why `svsim`'s sharding coverage was added last, and what it found

`test_svsim.py::TestShardedOutput` (subprocessing `test/sharded/svsim.py`) was checked because the
same axis in `sqd` hid three defects. `svsim` had none: it takes `out_sharding` as an explicit
parameter and threads it through every array-creating op, rather than resharding conditionally partway
through as `run_sqd` does.

Its one limit is documented rather than fixed: **`mesh.size` must divide `2^num_qubits`**, so a 3- or
6-device mesh fails at *every* qubit count, not just small ones. A state vector cannot be padded the
way `sqd`'s state list can — its indices *are* the basis states — so there is nothing to pad and the
jax raise (which names both shapes) stands. `PartitionSpec(None)` replicates instead.

**Assert the sharding *spec*, not just the values.** An explicitly replicated `svsim` run agrees with
the single-device answer to exactly 0.0, so "correct but silently unsharded" is invisible to any value
comparison — the regression a dropped `out_sharding` would actually cause. `TestShardedOutput` asserts
both.

### `apply_h` under a mesh: placement shipped, rounding declined (2026-09-15)

`markdown/spinchain/rqutils-apply-h-mesh-request.md` asked for two things: place a host `vec` internally, and round the
length up to `mesh.size` as `sqd` does. Placement shipped. Rounding was built (`1a339e8`), measured, and
withdrawn (`82c204b`) — the response doc carries the full argument; the four facts worth keeping here:

**The error names the wrong array.** A host `vec` resolves to `P(None,)` on an *empty* mesh; the `P('x',)`
in `Resource axis: x of P('x',) is not found in mesh: ()` comes from the partitioned `xsource` index
array, validated against the operand's empty mesh. Reading it as "numpy has a bad spec" sends you to the
wrong place.

**`isinstance(vec, jax.Array)` is the wrong discriminator** — shipped first, killed by mutation testing. A
committed `jax.Array` carries an empty mesh exactly as a host array does. `jax.typeof` is load-bearing in
the replacement: a committed array's own `.sharding` is a `SingleDeviceSharding` with no `.mesh` at all,
and `jax.typeof` normalizes both cases to a `NamedSharding` over one empty singleton `AbstractMesh`.

**Rounding cannot be applied consistently**, which is why "both or neither" was declined despite being a
sound argument: `diagonals` is `(n_groups, n_states)` but `diag_signs` is `(n_states, n_zbytes)`, so no
single pad axis serves both. The built version worked for `zsignatures=` — the only strategy the request
exercised — and broke the other two with a broadcast `TypeError` from inside the scan, worse than the
error it replaced.

**Three defects the 9-line guard went through, all found by review after the fact.** Worth the space
because each is a distinct class: it checked `vec`'s length where the constraint binds `states` (every
test arm sized both together, so the substitution was invisible); it read `shape[0]` where the kernel
broadcasts over a leading batch axis, rejecting a valid `(2, 24)` call as "vec length 2"; and one shared
error message could not serve two callers, since `hproj` takes unpacked rows that `uniquify_states`' 255
filler would make non-binary. A small guard on a subtle invariant needs review rounds, not one.

### `hproj` does not support sharding, and rejecting is simpler than fixing (2026-09-15)

`hproj` failed under a mesh at **every** subspace size, divisible or not: `columns[valid]` is a
boolean-mask gather on the partitioned array `get_xsource` returns, and XLA cannot resolve an output spec
for it (`ShardingTypeError`). Pre-existing — confirmed against `1a339e8^`.

It was fixed first (host transfer before masking, plus a divisibility check), then the fix was
**withdrawn** for an explicit `ValueError`: `hproj` returns a host scipy matrix, `spinchain` never calls
it, and every in-tree caller — `poc/sharding.py` (7a, 7c), `poc/davidson_xxz`, `poc/prefilter_cycles_e2e`,
`test/sharded/eigvec_roundtrip.py` — already calls it *outside* its `with jax.set_mesh(...)` block as
the unsharded oracle. **Rejecting removed 17 lines and two bug classes**, one of them a limitation only
documentable, never testable: the host transfer is single-process by construction, and virtual devices are
one process.

The check precedes the `np.unique`/`_is_lex_sorted` pass, for the same reason `_MAX_STATES` does —
`get_abstract_mesh().empty` is O(1), so a doomed call should not pay the O(N) sort. 0.57 ms to reject 4096
states.

### Why `test/sharded/*.py` are files rather than inline strings

Not named `test_*`, so pytest does not collect them; `test_sqd_sharded.py::TestShardedSqd` subprocesses
`test/sharded/sqd_grid.py` under `XLA_FLAGS=--xla_force_host_platform_device_count=4`, because the
virtual device count must be set before jax initializes and `conftest.py` has already imported it by
collection time. They live in files so ruff and ty check them — as a `textwrap.dedent` blob an
`ImportError` would surface as a nonzero exit, indistinguishable from the sharding regression the test
exists to catch.

## Architecture: simplifications that were tried and measured worse

### `ground_locg`: the one-matmul `_compute_sas` form does not belong here

From the since-deleted MLX port. Measured **98.7 ms against the scatter form's 27.7 ms at N=16.8M**,
because stacking three huge vectors is two 402 MB temporaries per iteration — the copy this module
exists to avoid.

### `ground_locg`: `body_iter1`'s exclusion bound is not `body()`'s specialized

`body_iter1` uses `2|rho| + 1`; `body()` uses `max(diag) + sum(|diag|) + 1`. The general form
collapses to a constant `1.0` for the negative `rho` of a ground-state search. Both are *valid* (each
strictly exceeds the one retained entry, so the eigensolver still cannot pick the excluded slot), but
the unified form's margin stops tracking the operator scale — a poor trade in a routine whose other
guards exist because large shifts destroy precision. Don't unify without redoing the bound argument.

### `ground_locg`: every guard is load-bearing and was measured

`markdown/locg.md` catalogues seven defects (I1–I7) that each failed *silently*, returning a plausible
wrong number rather than raising. Don't "simplify" the balancing, the re-orthogonalizations, or the
zero-direction masks.

**`markdown/locg.md` is stale** — it audits the pre-rewrite module, so its line numbers, its "no pytest
suite exists" scope note, and its A1–A5 gaps (all since fixed) don't apply. Cite it for the I-numbers
and the measurements only; read the module docstring for what currently holds. One severity is partly
retracted there — I5; see the testing section above for the retraction and the test that pins it.

### `sqd`: `get_xsource` setup dominates a solve

Weighted by call count it is **66–97%** of a solve (3.1 s against 79 ms of matvec loop at 10
iterations, N=200k, J=50), so the `2N` sort was not merely the `N ≤ 2^31` ceiling but the main cost at
every size measured, while `matvec/J` is flat at ~0.16 ms and confirms the `O(J·N)` model.
Independently reproduced end-to-end at N=3k, n=12, J=23: `(0, 2)` is **10.9× slower** than `(1, 2)`
and `(0, 0)` is **7.2× slower** than `(1, 0)`, all four levels returning the same energy. Hence the
advice to prefer `cache_level[0] = 1`.

### `sqd`: `get_xsource` is a binary search, not a sort

12–19× faster on the J-fold precompute on CPU, which is why **`states` must be lex-sorted**. Always
required — the sort was equally wrong on unsorted input — but previously undocumented.
`hproj(unique_states=True)` skips its `np.unique` and so can violate it; that returns a non-symmetric
matrix and is pinned by `TestHproj::test_unsorted_input_with_unique_states_is_wrong`.

Two paths selected statically on width: `uint64` keys for `B ≤ 8` bytes, an explicit lexicographic
search beyond. That boundary is a **correctness** limit — a `uint64` key silently truncates a wider
row and aliases distinct states — so if you touch it, note that a test only catches the overrun when
the subspace's *leading* bytes collide and partners genuinely exist — and that low qubit indices land
in the *leading* bytes (see the packed-signature note below), which is easy to get backwards and makes
the test pass vacuously.

### `sqd`: why `apply_h`'s positional form was deleted rather than deprecated

`cache_level` selected positionally how `scanned`'s members were read, and nothing could check the two
agreed: measured **0.44 max abs error** from one mispairing, and at `n = 15` with 2 states even their
*shapes* collide at `(2, 2)`, so no assertion could have closed it. Both are integer arrays.

### `paulis/symplectic`: the packed-signature shift, and why `matmul` is gone

`packbits` fills each byte from the **most significant** end, so a signature's payload bits are
entries `1` through `num_qubits` of `np.unpackbits`, in string-character order (leftmost character =
index 1) — the reverse of the qubit numbering. Anything decoding a packed signature back to an integer
must shift by `8*nbytes - (num_qubits + 1)`, counting the pad bit; dropping the `+1` silently returns
a *permutation* of the right answer. Measured **2.07 max abs error** in the since-removed `matmul`,
which is why that method is gone.

### `paulis/symplectic`: why there is no `force_real` flag

None could work. `.c` narrows to float64 exactly when the folded phase is real, i.e. when every string
has an even Y count, and an odd-Y string is complex128 *by construction*. Check `.c.dtype` if you need
float64. The one in-tree caller that did was the deleted MLX bench harness, so nothing exercises that
path now.

### `svsim`: `sin` is complex128 and must stay so

It carries `i·(-i)^popcount(x&z)` — the rotation's leading `i` and the `(-i)^{x·z}` phase of the
`Q = (-i)^{x·z} Z^z X^x` convention, folded in at build time. Omitting that phase silently broke every
`y`/`ry` gate — the only gates with overlapping X/Z signatures — and so every transpiled circuit
(`markdown/skqd.md`).

### `sqd`: why the initial vector is a spread, not a one-hot

A one-hot cannot leave the connected component of the projected Hamiltonian that contains it, so a
subspace whose Hamiltonian splits into disconnected blocks silently returned that block's minimum with
`converged=True`. `vinit_from_min_diag` still weights the minimum-diagonal state heavily on top of the
spread. Don't "simplify" either back to a one-hot — `test/test_sqd.py::TestSqdInitialVector` covers
both failure modes.

### `qprint`: test the full `fmt` × `output` grid, not a diagonal of it

Four bugs lived in cells nothing exercised, including `fmt='matrix'` being un-instantiable for *every*
input (`QPrintMatrix` never implemented the abstract `_add_labels`, which it does not need — it
overrides `_make_lines` and positions terms by row/column). An amplitude of exactly `1` is suppressed,
which is right when a basis label follows and wrong when nothing does; text-mode labels also carry the
`*` separator as a prefix, so the two renderings can disagree while each looks fine alone.
Cross-rendering assertions are what catch that — see `test/test_qprint.py::TestAmplitudeAndSeparator`.

### `paulis/general`: the `npmod` gating bug

Gating Python-level shape inference on `if npmod is np:` broke the entire `npmod=jnp` path in three
separate places (`components` plus the since-removed `compose` and `truncate`, all raising
`TypeError: object of type 'int' has no len()` on a scalar `dim`). `_normalize_dim` is now called
unconditionally at every site for exactly this reason.

## The MLX port: deleted, and what it left behind

The JAX solver measured faster even on the MLX GPU backend, so the port had no performance case, and
nothing in the tree imported or ran it. Don't reintroduce a second solver implementation without that
measurement going the other way first.

`markdown/mlx-metal-kernels.md` is the historical record of the fused-Metal-kernel work — three kernels
written, **one measured slower and deleted, two verified negative** — kept so nobody re-derives them
from scratch. Read it before attempting anything in that direction. It is a record, not a guide: every
claim about what the port *offered* is superseded, and any revived kernel needs its static MSL guards
revived too, since a numpy shim never compiles MSL text.

**When you delete a comparison arm, check what it was incidentally covering.** Learned the hard way
there: the rank-aware-selection guard (I3) had been tested *only* as a side effect of comparing the
fused Metal eigensolve against the op-graph one, so deleting the kernel silently took its only test
with it. The same trap applies to `/simplify`-style cleanups generally — re-run coverage checks after
removing an arm rather than assuming the remaining arms overlap.

## Scaling POCs: baselines, and three ways to misread a GPU run

The six scaling POCs live under `poc/`, findings in `markdown/scaling-pocs.md`.

**The POCs no longer have a baseline in the library and must not be pointed at one.**
`searchsorted.xsource_sort_legacy` is a verbatim copy of the pre-23fb226 sort and is the timing baseline for
both POC 1 and POC 8; their *correctness* arms still compare against `get_xsource`, which is the point
(agreement with what ships is now a regression test). Point a timing arm at the library and you get
searchsorted-versus-searchsorted: the first GPU run of `poc/gpu_unverified.py` reported
1.002×/1.000×/1.000×, POC 1e read 0.26× "SLOWER", and `fmt_ratio` was correct every time — which is
what made it easy to misread as a GPU finding. Restoring the baseline recovers 12.1×/18.3× and
3.57×/3.18× for the lex variant.

Two related traps in that script, both fixed: `peak_bytes_in_use` is a high-water mark that never
decreases (and sampling `bytes_in_use` after `del` reads the post-free baseline), so a leak test built
on either cannot observe anything; and `--devices` sets `CUDA_VISIBLE_DEVICES`, a *filter* over what
the driver exposes, so it cannot create a second GPU — Claim 3 on a one-GPU box is unrun, not
unresolved.

**On GPU the speedup is 5.15×, not 12–25×** (NVIDIA GH200, N=64M single signature, `alpha` 1.09 vs
0.92, so still rising with N). Two other GPU numbers from the same run are launch-bound artifacts and
must not be quoted: POC 1c at J=50 reads 12.5–14× — deceptively close to the CPU figure — with a sort
arm *flat* at 1141/1239/1201 ms across a 5× N range, and POC 1b below N=1M reads 2.56×. The
launch-bound regime covers J=1 past N=1M *and* J=50 at N=500k, so it is per-call latency × call count,
not N alone; `--sweep-to` exists to escape it and `check_scaling` fits `alpha`, refusing to call a
ratio quotable below 0.6.

**The `lax.sort` GPU memory leak does not reproduce** (~0.95 GB of transients fully reclaimed every
rep at N=5M/B=4), so that note was stale — and since the sort left the library, it is now a claim
about `lax.sort` rather than about `sqd`. Multi-GPU speed remains **unrun**, needing a physically
multi-GPU box.

## Measurement hygiene

### Use `eigvalsh`, or sparse `eigsh(k=1)` — never `eigh`

`np.linalg.eigh` (values *and* vectors) costs **77 s** at n=18/dim=4000 against 0.2 s for `hproj`, and
`op.to_matrix()` on a `SparsePauliOp` builds the full `2^n × 2^n` dense array — 4.3 GB at n=14.
Measured: a full-space reference took **46.4 s** dense against **0.0 s** via
`eigsh(op.to_matrix(sparse=True).tocsr(), k=1, which='SA')`, agreeing to 1.8e-15; a projected
reference went **11.24 s → 0.02 s** at n=16/dim=3000, agreeing to 7.9e-08 (the Lanczos `tol`, three
orders below typical inter-arm differences). Two harnesses in one session each burned ~8 minutes
calling `eigh` per instance for a diagnostic that needed one column — hoist any full decomposition out
of inner loops.

### Verify the referent, not just that the pointer resolves

Two cross-references in `test/` were "fixed" by removing dead paths while their surrounding claims
stayed wrong: a `3.6e-15` agreement figure that was a one-off observed value rather than the actual
`1e-9` gate, and an `hproj` workaround described as live when the file it points at records the bug as
fixed. Read the target.

Same rule for cost figures — A/B the whole call against the pre-change revision in a worktree
(`git worktree add`, `PYTHONPATH` at it, since the venv's editable install otherwise serves HEAD to
both arms): timing a guard predicate alone measured 3.4–3.8% where the end-to-end cost was 12–14%.

### Writing `Raises:` sections finds bugs

You cannot document a raise without reading its condition, which caught three wrong claims in one
pass: `apply_h`'s `states` arg omitted `(1, 1)` from the no-states-needed set; `components`' documented
`ValueError` is gated on `npmod is np` (so under `npmod=jnp` a bad `dim` gives an opaque `dot_general`
TypeError instead); and `ground_locg` accepts a bare Python `int` for `xinit` despite reading `.dtype`
— both callers are `jax.jit`-wrapped, so it arrives as a 0-d tracer. Trigger every raise you document.

### Docstring hazards that no tool catches

`"""... :math:`\alpha` ..."""` compiles `\a` to a BEL byte: the rendered reference is corrupted while
ruff, `ty` and pytest all pass, since it is valid Python. Sweep after touching any docstring
containing a backslash — this found exactly one offender (`hproj`) across the package. The sweep:

```bash
uv run python -c "
import rqutils.sqd, rqutils.ground_locg, rqutils.svsim, rqutils.qprint
import rqutils.math as rm, rqutils.paulis.general as pg, rqutils.paulis.symplectic as ps
bad = [(m.__name__, n) for m in (rqutils.sqd, rqutils.ground_locg, rqutils.svsim, rqutils.qprint, rm, pg, ps)
       for n, o in list(vars(m).items()) + [('<module>', m)]
       if isinstance(getattr(o, '__doc__', None), str) and any(c in o.__doc__ for c in '\x07\x08\x0b\x0c')]
print('control-char docstrings:', bad)"
```

Also brace `:math:` exponents (`2^{31}`, not `2^31`); unbraced renders as `2³1` and nothing warns.

**A docstring's body indentation must be uniform, and no tool catches a break.** Writing 8-space
continuation lines into a 4-space docstring makes reST read the deeper lines as a **block quote**,
which nests `Args:`/`Returns:`/`Raises:` where napoleon cannot parse them -- so the published
reference for that function silently loses its parameter table. Measured: this happened to
`apply_h` while ruff, `ty`, pytest and the control-char sweep above all passed, and the docs build
still reported success (the sweep sees byte values, not layout). Print the indentation ladder
instead -- a healthy docstring shows one dominant level with deeper ones only for nested blocks:

```bash
awk '/r"""<first words of the summary>/,/^    """$/' rqutils/sqd.py \
  | awk '{match($0, /^ */); if (length($0)) print RLENGTH}' | sort -n | uniq -c
```

`.. autoclass::` needs `:members:` or member docstrings are unpublished — `PauliSumXZ`'s four
documented public members rendered nowhere until it was added (`CircuitXZ` is deliberately bare, so
the flag is a no-op there). Confirm by grepping the built HTML for an `id="...<name>"` anchor, not by
reading the source. The docs build has **one** standing warning (`rqutils.paulis.rst` not in any
toctree — a `sphinx-apidoc` package stub); anything beyond that is yours. Note `grep -c warning` on
the build output counts Sphinx's own summary line too.

## The `N ≤ 2^31 - 1` ceiling: why it is enforced where it is

Subspace positions are int32 throughout — `uniquify_states`' iota, and `get_xsource`'s output with
`-1` as the absent marker — so a size at or above `2^31` wrapped to `-2147483648` and returned a
corrupted *permutation* rather than raising. The wrapped value is not `-1`, so the absent-marker test
could not catch it either. Unreachable on real hardware (`2^31` states is 4.3 GB of packed states
before any vector), which is why the check is cheap insurance.

`hproj`'s guard sits **before** its O(N) sortedness scan deliberately: an O(1) look at a shape,
measured **0.23 s** there against **23 s** when placed after the scan.

The guard also sits on `uniquify_states`' **static** `states_size`, where the int32 iota is actually
created — `uniquify_states` and `get_xsource` are un-underscored and called directly by six
`poc/` scripts, i.e. exactly the code that pushes N, which reached the iota with neither
entry-point guard in the chain. Being static it fires at trace time and costs nothing per call. That
placement also made **both** sides of the boundary cheap to pin: `jax.eval_shape` traces the guard
without allocating (~5 ms per side), where reaching it through `hproj` cost 23 s. `TestInt32Ceiling`
covers both, so the off-by-one that used to survive (`>` → `>=`, rejecting the largest legal size) is
now caught.

`uniquify_states`' single-device `jax.lax.sort` is the reason for the limit's magnitude;
`get_xsource` no longer contributes — it is a binary search into the already-sorted list, not a sort
of a stacked `2N` array.

### Replacing that sort out-of-core: prototyped and rejected (2026-08-29)

Rejected: the word-packed chunked sort + merge at n=100, B=13, N=8M is 4.3× slower (31.3 s vs 7.3 s) with
2.9× the peak RSS (1048 vs 365 MB) of plain `np.unique`: packing widens rows, and chunking distributes
nothing. `poc/ooc-uniquify.md` §2–§3.

## Memory at scale: the source cache, the diagonals, and their pre-filters

### At n=100 the binding constraint is the xsources cache, not the sort (2026-08-29)

Scaling attention has been on `uniquify_states`' sort because it sets the `2^31` ceiling. But an actual
n=100 solve is dominated by something else. Budget at `B = 13`, `J = 50`, `cache_level[0] = 1`:

At `N = 2^28` the cache is 53.7 of 82.9 GB, and at `2^31` 429.5 of 663.6 GB; the per-term table is
`markdown/xsources-cache-budget.md` §1.

The cache is **65% of the footprint** at `J = 50` — the largest single object, 8× the state list. So
`CLAUDE.md`'s "prefer `cache_level[0] = 1`" is right at the sizes it was measured at and *becomes
impossible* exactly where scaling matters. On a 16 GB node at `N = 2^28`, 14 of 50 groups fit.

**This budget assumes `K = 1`, and a real Hamiltonian is not like that.** 1D Heisenberg at n=100 has
`K = 100`, where the diagonal arrays are 2-3x the source cache and `xsources` is **22%** of the `(1, 0)`
footprint rather than 65%. See "Measured on a real n=100 Hamiltonian" below, which supersedes the
`4 * J * N`-dominates framing in this section and the two that follow it.

**And the penalty for turning it off is much worse at n=100 than the recorded 7-11x.** Measured at
n=100, N=200k, J=16 with the word-based search: an uncached matvec is **59.8x** slower (106.4 ms against
1.8 ms), and the precompute breaks even after **1.5 matvecs** against a solve's 100-300. The gap widened
because the wide-row search is intrinsically dearer per call, so at n=100 the two `cache_level[0]`
settings are "does not fit" and "60x slower".

**The partial-J dial is the answer, and `markdown/scaling-pocs.md` §2 already scoped it: "only worth
building if the full cache genuinely does not fit — otherwise always cache everything."** At n=100 with
large N that condition is now met, which it was not when that POC ran. Re-measured with the current
implementation, caching `J'` of `J = 16` groups and recomputing the rest:

| `J'` | cache | matvec |
| --- | --- | --- |
| 0 | 0 MB | 106.0 ms |
| 4 | 3.2 MB | 79.6 ms |
| 8 | 6.4 MB | 54.6 ms |
| 12 | 9.6 MB | 28.2 ms |
| 16 | 12.8 MB | 1.7 ms |

Linear in `J'` as the POC found, so it is a genuine continuous dial rather than a step. The API shape
this wants is a **memory budget**, not a mode: cache `floor(budget / (4N))` groups. Note the last step
(12 -> 16) is disproportionate, so the endpoint is still special — a partial cache never reaches the
full-cache time.

**Four literature techniques do not transfer, all for the same reason.** Lin tables, DanceQ's
divide-and-conquer (`tmp/2407.14591v2.pdf`), selected-CI residue arrays and rank-select all assume the
basis is *characterized* — every state at fixed particle number — so "how many states precede this one"
is a closed-form combinatorial count (DanceQ Eqs. 15/19/20/23 are all `D_Q(L, n)` binomials). An SQD
subspace is a **sampled subset**: that count exists only in the sampled list, so the offsets and strides
those methods need do not exist and no subsystem partitioning creates them. DanceQ's §4.4 conclusion
does transfer though, and it is the one above: for a matrix-free matvec *"the memory footprint of each
worker process should be the guiding principle"*, because runtime depends only weakly on the lookup
scheme. For calibration, their state of the art is 46 spins on ~256 nodes at 512 GiB each.

### `xcache_groups`: an intermediate count can *raise* peak memory (2026-08-29)

Shipped in `ae4bdee`. The cache array shrinks linearly in `J'` — that part is exact arithmetic,
`4 * J' * states_size` — but **peak memory does not**, because a partial cache runs two matvec kernels
instead of one and the second one's intermediates are not free.

Measured from XLA's `memory_analysis().temp_size_in_bytes` at `J = 16`, `N = 28344`,
`states_size = 32768`:

| `J'` | cache array | XLA peak | vs full |
| --- | --- | --- | --- |
| `None` (full) | 2.1 MB | 9.0 MB | 1.00x |
| 0 | 0 MB | **7.7 MB** | 0.86x |
| 4 | 0.5 MB | 9.8 MB | 1.09x |
| 8 | 1.0 MB | **10.4 MB** | **1.15x** |
| 12 | 1.6 MB | 10.9 MB | 1.20x |

So at this `J` every intermediate value *costs* peak memory while appearing to save cache. `J' = 0`
always saves, because that arm is single-kernel with no tail tuple.

The crossover is in `J`, since the cache scales with `J` while one kernel's working set does not. Sweeping
at fixed `N = 28k`: at `J = 16` the full cache is 2.1 MB of a 9.0 MB peak and `J' = J/2` costs 1.3 MB
net; at `J = 48` the cache is 6.3 MB of 13.2 MB and `J' = J/2` saves 0.8 MB; at `J = 48, N = 114k` it is
25.2 MB of 53.0 MB and saves 3.1 MB. **Use the dial when the cache is a large fraction of the
footprint** — which is the condition `markdown/scaling-pocs.md` §2 already gates the whole idea on, and is
why the guidance survives even though the naive "memory is linear in `J'`" framing does not.

Both docstrings state this. Recorded here because the measurement is what makes it a rule rather than a
caveat, and because the formula is so clean that a reader will otherwise trust it.

### Measured on a real n=100 Hamiltonian: the diagonal axis dominates, not the source cache (2026-08-29)

Every memory figure in the sections below was derived at **K=1** — one Z signature per X group — because
the fixtures were random Pauli strings. A real Hamiltonian is not like that, and the difference inverts
the guidance.

**1D Heisenberg at n=100, periodic:** 300 terms group into **J=101 X groups with K=100 Z signatures
each**, B=13, and the coefficients come out `float64` (the Y terms pair up). Measured from XLA's own
`memory_analysis().temp_size_in_bytes`, per state *slot* (`states_size`, the power-of-two padded size —
which is itself a 40% inflation at N=24M, since 24M rounds to 33.6M):

| `cache_level` | B/slot | at N=24M unique |
| --- | --- | --- |
| **(0, 0)** | **120** | **4.0 GB** |
| (1, 0) | 492 | 16.5 GB |
| (0, 2) | 920 | 30.9 GB |
| (1, 2) | 1296 | 43.5 GB |
| (0, 1) | 1433 | 48.1 GB |
| **(1, 1)** | **1805** | **60.6 GB** |

Stable to ±1 B/slot across N=2000 and N=8000, so the linearity is real; the 24M column is extrapolated,
not run.

**The three terms, and which one wins:**

| array | shape | B/slot at J=101, K=100 |
| --- | --- | --- |
| `diag_signs` (`cache_level[1]==1`) | `[J, ceil(K/8), ss]` | **1313** |
| `diagonals` (`cache_level[1]==2`) | `[J, ss]` float64 | 808 |
| `xsources` (`cache_level[0]==1`) | `[J, ss]` int32 | 404 |

`diag_signs` alone nearly accounts for `(1, 1)`'s whole 1805. So on a real n=100 problem
**`cache_level[1]` is the expensive axis and `cache_level[0]` is the cheap one** — the reverse of the
K=1 picture, and the reverse of what motivated the partial-J work. `markdown/scaling-pocs.md` says the
diagonal axis "is where the real memory-versus-speed judgement lies"; at K=100 that is emphatically
true, and the 15x between `(0, 0)` and `(1, 1)` is available today with no new API.

**A prior extrapolation here was wrong and is superseded.** Fitting n=20, K=1 gave
`bytes/slot = 203 + 4*J`, i.e. 607 at J=101 — **23% too high for `(1, 0)` and 3.0x too low for
`(1, 1)`**. The `K`-dependent diagonal term was the dominant one and the fit had no way to see it. Do
not size hardware from a K=1 fit.

### What the Bloom pre-filter is actually worth here

The filter costs **0.029 GB (0.86 B/slot)** at N=24M, flat in `J`. It replaces the `xsources` cache by
making `cache_level[0] = 0` affordable in *time*, so the memory it saves is exactly that 404 B/slot term
— **12.4 GB at N=24M, the same in every row** — but the ratio depends entirely on the diagonal level:

| from | to | before | after | ratio |
| --- | --- | --- | --- | --- |
| (1, 0) | (0, 0) + BF | 16.5 GB | **4.1 GB** | **4.06x** |
| (1, 2) | (0, 2) + BF | 43.5 GB | 30.9 GB | 1.41x |
| (1, 1) | (0, 1) + BF | 60.6 GB | 48.1 GB | 1.26x |

So the honest framing on this Hamiltonian: **the filter is not a memory optimization, it is a speed
rescue for the cheap-memory setting.** `(0, 0)` already costs 4.0 GB today with no filter, no new API and
no `cap` hazard; `(0, 0) + BF` costs 4.1 GB. The filter's value there is buying back the ~60x penalty
that `cache_level[0] = 0` carries, for 0.029 GB — not shrinking the footprint.

Its relative worth would return on a Hamiltonian with **small K** (few Z terms per X group), where the
diagonal arrays shrink and `xsources` is once again the dominant term. Both readings are correct; which
one applies is a property of the Hamiltonian, so quote `K` alongside any of these figures.

### A Bloom filter for *input* dedup: better-targeted, still loses to `np.unique` (2026-08-29)

The narrowest and best-aimed version of the filter idea: `sqd` receives raw measured bitstrings with
duplicates and dedupes them inside `uniquify_states`' lexsort. Use a filter for that dedup instead.

**The duplicate rate makes this worth measuring.** Simulated Zipf-ish shot sampling, which is what a real
SQD workflow produces:

| n | support | shots | unique | duplicate rate | shots/unique |
| --- | --- | --- | --- | --- | --- |
| 30 | 50,000 | 2M | 47,995 | **97.6%** | 41.7 |
| 30 | 500,000 | 2M | 180,692 | **91.0%** | 11.1 |

So the array `sqd` pads to `states_size` and lexsorts is 11-42x larger than the unique set it produces.

**The error direction is also favourable, which is the interesting part.** For dedup a false positive
means "I think I have seen this" → the state is *dropped*. That loses a genuine basis vector, which is
**variationally safe**: a smaller subspace gives a *higher* energy, never a wrong one. Contrast the
pre-filter case, where an FP costs a wasted search and exactness is recovered — here exactness is lost,
but the loss is bounded and in a known direction.

**It still loses, for three reasons, and the third is the one that matters.**

| variant | vs `np.unique` | genuine states lost |
| --- | --- | --- |
| BF dedup replacing `np.unique` | **0.64-0.79x** | 0.28-0.31% |
| BF pre-reduce, then `np.unique` | **0.63-0.72x** | 0.30-0.31% |
| chunked `np.unique` + `union1d` (exact, no filter) | **0.77x** | 0 |

1. **The dedup is inherently sequential.** "Have I seen `x`?" depends on every earlier insertion, so it
   cannot be vectorized over the array. A blocked version tests a block then inserts it, so duplicates
   *within* a block survive — measured 106,455 kept for 47,848 distinct.
2. **`np.unique` on `uint64` is a radix sort**: one pass, fully vectorized C. Hard to beat from numpy.
3. **`get_xsource` needs lex-sorted *and* unique input, so the sort is mandatory regardless.** That makes
   the filter *extra* work rather than replacement work — the honest comparison is `BF + np.unique`
   against `np.unique`, which it loses on both time and exactness.

Worth keeping as the general lesson: a filter can only replace a sort when nothing downstream needs
**order**. Here `get_xsource` binary-searches the result, so order is not optional, and every
approximate-set structure loses by construction.

### Using a Bloom filter as the subspace *definition*: sound, and it loses past n~70 (2026-08-29)

A sharper version of the filter idea: stop treating false positives as wasted work and **accept them into
the subspace**. The subspace becomes `{x : BF accepts x}` — the sampled states plus whatever else the
filter admits — and `states` is never materialized.

**The physics is fine, which is why the idea is worth taking seriously.** SQD projects onto whatever
subspace it is given, and adding basis vectors can only *lower* the variational energy. False positives
are extra states, not wrong answers.

**Two things kill it, and only the second is fundamental.**

First, mechanically a filter cannot stand in for `states` at all: `get_xsource` needs the **rank** of
`S[i] ^ X` (a filter has no order), the diagonal builders need `popcount(S[i] & z)` (a filter stores no
bits), and `sqd` **returns the states** — `sqd.py:624` unpacks `states_u[:subspace_dim]` because the
eigenvector is indexed by position, so without the basis the eigenvector is uninterpretable. Accepting
false positives does not fix any of that; it changes which set is being represented, not what operations
are needed on it.

Second, and this is the one that generalizes: **the false-positive *count* scales with `2^n`, not with
`N`.** `|accepted| = N + p*(2^n - N)`. At n=100, N=24M, even a `p = 1e-12` filter admits `1.3e18` extra
states. To hold the false positives to 1% of `N` the filter must get bigger than the list it replaces:

| n | filter bits/item | explicit rows bits/item | smaller |
| --- | --- | --- | --- |
| 34 | 23.3 | 40 | filter |
| 58 | 57.9 | 64 | filter |
| 66 | 69.4 | 72 | filter |
| **74** | **81.0** | **80** | **rows** |
| 100 | 118.5 | 104 | rows |

The crossover is near **n ≈ 70**. The reason is information-theoretic rather than incidental: representing
an `N`-subset of `2^n` with false-positive rate `p` costs at least `N*log2(1/p)` bits, and driving the FP
*count* down forces `p → N/2^n`, at which point that bound approaches `N*log2(2^n/N)` — the cost of simply
listing the elements. **A filter only wins when a fixed FP *rate* is acceptable, never a fixed FP
*count*.**

So the version that survives is the original one: the filter as a *pre-filter* over an explicit sorted
list, where a false positive costs one wasted search and the exact answer is recovered by the equality
test. Not as the subspace's representation.

### Hoisting the precompute, and the per-group host sync: both affordable (2026-08-29)

Two open items against the pre-filter design, both measured.

**Hoisting the J-fold precompute out of `run_sqd`'s trace does not regress the shape pinning.** The
concern was `states_size`, which "exists solely to pin array shapes and prevent JIT recompilation"
(`CLAUDE.md`): `sqd` rounds the input length up to a power of two and pads with filler, so many input
lengths map to one traced shape. A hoisted precompute takes `states_u` (`[states_size, B]`) and returns
`[J, states_size]`, both functions of values that are already static, so the pinning survives — verified,
not argued: **three different input lengths at one `states_size` produce one compilation.**

What hoisting does cost is a **doubled compilation count** — two jitted functions instead of one. At
nq=12, nine input lengths spanning four distinct `states_size` values: `inside` compiles 4 variants,
hoisted compiles 4 + 4 = 8. Output identical. But the compilations are cheap and amortized:

| | inside (today) | hoisted | ratio |
| --- | --- | --- | --- |
| compile, once per `states_size` | 0.11 s | 0.11 s | **1.03×** |
| warm run, every solve | 73.9 ms | 75.0 ms | **1.01×** |

at nq=20, N=182k, `states_size` 262144, J=16. So the doubled variant count is a cache-occupancy fact
rather than a time cost.

**The per-group `int(ncand)` host sync costs 2.0% at J=50, and batching it recovers that.** At N=2M,
J=50:

| policy | time | overhead |
| --- | --- | --- |
| no sync (unsafe) | 609.6 ms | — |
| sync per group | 621.8 ms | **+2.0%** (244 us/group) |
| one batched sync (`jnp.max` over all `ncand`, one read) | 611.0 ms | **+0.2%** |

244 us per group is real but small against a ~12 ms per-group kernel. Batching keeps the check on *every*
group while paying for one device-to-host transfer instead of `J`.

**But batching constrains where the filter can live.** A batched check cannot retry a group until all `J`
have run, which is fine for a precompute — retry the offenders afterwards — and wrong inside a matvec
loop, where the result is consumed immediately. Combined with the fact that the retry policy is host-side
sequencing and cannot live inside one `jit` at all, this is the second independent reason to put the
filter on the **precompute** rather than the matvec.

### `jnp.nonzero` does not shard, and it does not matter — the mask is already replicated (2026-08-29)

The pre-filter's compaction is `jnp.nonzero(mask, size=cap)`, and the open question was whether it shards.
**It does not.** On a partitioned mask it raises `ShardingTypeError`: *"The input should be fully
replicated when axis is not specified to cumsum."*

The failure is structural, not a JAX gap. Compaction means "move element `i` to position `rank(i)`", and
`rank(i)` depends on every element before it — which lives on another device. Tested per ingredient on a
`P('x')` mask at N=1M:

| operation | partitioned mask |
| --- | --- |
| `mask.sum()` (the overflow check) | **OK**, 2 collectives |
| `jnp.where` (elementwise) | **OK**, 0 collectives |
| `jnp.cumsum` | `ShardingTypeError` |
| `jnp.argsort` | `ShardingTypeError` |
| `jax.lax.top_k` | `ShardingTypeError` |

Everything that *reorders or compacts* along the sharded axis fails; only elementwise ops and reductions
survive. Note the **free overflow check shards even though the compaction does not**, so the safety
mechanism is not the constrained part.

**This does not block the filter, because `get_xsource` already requires a replicated `states`.** A
partitioned `[N, B]` state array fails on the *baseline*, before any filter is involved — `ValueError:
Unmapped values passed to vmap cannot be sharded along the mesh axis you are vmapping over`. The library
knows this: `_spread_seed`'s comment (`sqd.py:788-790`) says `run_sqd` reshards `states_u` only inside
`if cache_level[0] == 1`, "because the uncached branch still needs the replicated array for the
`get_xsource` searches". So on the path the filter accelerates, the mask derived from those states is
replicated by construction and `cumsum` is satisfied.

Verified end-to-end under a 4-device mesh with states, targets and filter replicated — the sharding
`get_xsource` actually runs under:

| | baseline | BF-filtered |
| --- | --- | --- |
| `all-gather` / `all-reduce` / `collective-permute` / `all-to-all` | 0 / 0 / 0 / 0 | **0 / 0 / 0 / 0** |
| output spec | `P(None,)` | `P(None,)` |
| exact | — | **yes** |

So the filter is usable on the multi-device path. What it does **not** do is lift the replication
requirement: `states` still costs `13 * N` bytes on *every* device (27.9 GB per device at N=2^31), which
is a separate ceiling from the `xsources` cache the filter exists to shrink, and one no filter can touch.
The honest scope is "the filter shrinks the per-device cache", not "the filter makes the subspace
distributable".

### Deriving the pre-filter capacity, and why the check must be separate from it (2026-08-29)

The `cap` hazard blocked the whole pre-filter family: `jnp.nonzero(mask, size=cap)` needs a static size,
and an undersized one drops hits with **no error**. Both halves of the fix were built and measured, and
they are **not** the same mechanism.

**An analytic bound does not exist.** `candidates = hits + FP`, and the FP tail is beautifully tight — a
6-sigma binomial bound is within **0.1-6%** of the mean at these sizes, because `Binomial(N, p)`
concentrates hard. But it needs `hits`, which is the unknown being computed, and `hits` can legitimately
be `N` (a subspace closed under the hop has a 100% hit rate for that hop). So any bound not derived from
the data collapses to `cap = N`, which is correct and worthless.

**The overflow check is free; deriving the cap is not.** The check is `mask.sum()`, and the mask is
already computed:

Baseline `searchsorted` 67.2 ms; a given cap 24.8 ms both with and without the check (2.71×/2.70×, so the
check's -0.3% is noise); a derived cap 31.7 ms (2.12×, +27.6%). Table:
`markdown/xsources-cache-budget.md` §5.

Counting costs **0.04 ms** on top of the mask at N=4M. The 27.6% is not the sum — it is the *second
pass*, since deriving runs the mask once to count and again to search.

**So derive once per sweep, not per group — the naive version is slower than no filter at all.** Over a
J=16 sweep at N=4M:

| strategy | time | vs baseline |
| --- | --- | --- |
| 16 plain searches | 1083.8 ms | 1.00× |
| derive per group | 2403.8 ms | **0.45×** |
| derive once, check every group, retry on overflow | 518.5 ms | **2.09×** |

Per-group derivation loses because each distinct `cap` is a separate `jit` compilation. Deriving once
from the first group and letting the check catch a later miss is **4.64× better** and equally exact:
worst case is one extra kernel for the offending group, never a wrong answer.

**The retry path is verified, not assumed.** On a fixture mixing a dense half (closed under bit-0 flips)
with a sparse half, group 0 derives a 65,536 cap and group 1 needs 519,752 candidates: the check fires,
re-runs at 524,288, and the result is exact. Power-of-two rounding keeps the compilation count bounded —
a 16-group sweep on a uniform subspace hits **one** cap value.

Two behaviours to preserve in any implementation:

- **Raise, do not clamp.** An undersized explicit `cap` must raise — including off-by-one (55,347 against
  55,348 candidates) — and the message should name the deficit and the sufficient value.
- **`cap = N` is the safe degenerate case.** On a hop-closed subspace the derived cap clamps to `N`, the
  filter stops paying, and the answer stays correct. Verified: 2,000,000 of 2,000,000 candidates, exact.

### Partial-J plus a Bloom filter: the two compose, and the filter helps at every setting (2026-08-29)

The two ideas above are complementary — cache the `J'` groups that fit, BF-filter the recompute for the
rest — so they were measured together. n=100, N=600k, J=16, `p = 1%` filter at **0.72 MB** against a full
cache of 38.4 MB. Fully-cached matvec is the reference at 6.2 ms. **Every arm verified exact against it.**

Recompute plain → with the filter: 447.4 → 47.4 ms at `J' = 0` (9.43×, 7.71× the full cache), down to
112.6 → 20.0 ms at `J' = 12` (5.64×, 3.25×). Per-`J'` table: `markdown/xsources-cache-budget.md` §4.

**The filter earns its 0.72 MB at every point on the dial**, not just at `J' = 0`: 5.6–9.4× on whatever
portion is recomputed. And because it is built once per subspace and shared by every group, its cost does
not scale with `J'` — the `+BF` column is flat while the cache column grows linearly.

So the practical shape is a **memory budget**: cache `floor((budget - |BF|) / (4N))` groups and filter the
remainder. At n=100, J=50 that reads:

| N | full cache | 16 GB budget | 64 GB budget | 256 GB budget |
| --- | --- | --- | --- | --- |
| 24M | 4.8 GB | 50/50 cached | 50/50 | 50/50 |
| 268M | 53.7 GB | 14/50 cached, 36 filtered | 50/50 | 50/50 |
| 2^31 | 429.5 GB | 1/50 cached, 49 filtered | 7/50 | 29/50 |

**A caveat on predicting `J'` from a budget.** A linear model
(`t = J'*t_cached + (J-J')*t_bf`, with `t_bf/t_cached` measured at 7.6×) fits the endpoints exactly and
the middle to ~4-6%, but drifts to **17.5% at `J' = 12`** — the measured 20.0 ms against a predicted
16.5 ms. So a budget-based API can size the cache correctly (that is exact arithmetic) but should not
promise a runtime from the model alone.

**The `cap` failure is not hypothetical, and it inflates its own speedup.** A first run of this used
`ccap = 16384` against a true worst case of 17,913 candidates across the J groups, silently dropped 933
real hits, and reported **5.7× instead of the honest 8.0×** on the `J' = 0` arm — the truncated arm does
less work, so the bug flatters the result. Size the capacity from the worst case over *all* `J` groups,
not one, and verify against `get_xsource` rather than trusting a timing.

### A Bloom filter breaks the `2^n` dependency the exact bitmap could not (2026-08-29)

The exact membership bitmap in the rank-select family is `2^n / 8` bytes, so it dies at n≈34 no matter
how sparse the subspace is — it indexes the *Hilbert space*. **A Bloom filter sizes by `N` instead**, so
the `2^n` term disappears entirely:

At 1% FP: 9.59 bits/item, k = 7, 29 MB at `N = 24M` and 2.6 GB at `2^31`, against an exact bitmap of 2.15
GB at n=34 and **137 GB** at n=40. Table for 10%/1%/0.1%: `markdown/xsources-cache-budget.md` §3.

**False positives are safe here, and that is not generally true of a filter.** `get_xsource` ends with
an explicit equality test (`found = keys[pos] == target_keys`, and `jnp.all(W[pos] == Wt)` on the wide
path), so a false positive costs one wasted `searchsorted` that then correctly reports absent. The output
stays **exact**. False negatives would be fatal, and a Bloom filter cannot produce them. Verified
bit-identical against `get_xsource` at every setting measured below.

Measured, `jit`-compiled, splitmix64-style mixing (k hashes from one key, no tables):

| n | N | FP measured | filter memory | speedup | exact bitmap |
| --- | --- | --- | --- | --- | --- |
| 30 | 4M | 1.02% | 4.8 MB | 2.76× | 134 MB, 4.09× |
| 30 | 4M | 0.13% | 7.2 MB | 2.87× | 134 MB, 4.09× |
| **100** | 1M | 1.01% | **1.2 MB** | **4.62×** | **1.6e29 bytes — impossible** |
| **100** | 1M | 0.13% | **1.8 MB** | **4.50×** | **impossible** |

The n=100 rows use a subspace closed under one bit flip at fixed Hamming weight, giving a realistic
1.98% hit rate; a uniform-random fixture has ~0 partners and measures the easy case only (6.1–6.9×).

Speedup degrades monotonically with hit rate, and **break-even is ~55%**, above which plain
`searchsorted` wins. Measured FP holds at 1.00% throughout, independent of hit rate, as theory predicts:

| hit rate | 0.2% | 5.2% | 25.1% | 50.1% | 100% |
| --- | --- | --- | --- | --- | --- |
| speedup | 3.83× | 2.82× | 1.89× | 1.13× | 0.72× |

**Two caveats before this is worth building.** It shares the exact pre-filter's `cap` problem — the
compaction needs a static candidate capacity, an undersized one drops hits silently, and the bound must
now cover true hits *plus* false positives, so it is `N`-bounded in the worst case exactly as before.
That is the blocking issue, not the memory. And these are single-signature measurements: the filter is
built once per subspace and reused across all `J` groups, so a J-fold sweep should do better than the
per-call figures here, but that is unmeasured.

### Binary fuse filters: better query, unaffordable construction (2026-08-29)

`arxiv.org/abs/2201.01174`. Within **13%** of the storage lower bound against Bloom's 44%, and a query
is a fixed **3 gathers + 2 XORs + 1 compare** regardless of the false-positive rate, where Bloom needs
`k = -log2(p)` hashes. Implemented the 3-wise variant (host-side construction, `jit`ed query) and
verified it against the paper: **1.130n array, 9.04 bits/key, measured FP 0.389% against the 2^-8 =
0.391% theory, no false negatives.**

**The query is better than Bloom's, as advertised:**

| filter | memory | measured FP | speedup | exact? |
| --- | --- | --- | --- | --- |
| Bloom, k=7 | 4.8 MB | 1.02% | 2.76× | yes |
| **binary fuse, 8-bit** | **4.5 MB** | **0.388%** | **3.62×** | **yes** |

n=30, N=4M, hit rate 0.372%. Better speedup at a *lower* FP rate and slightly less memory — 22% less
than Bloom at equal FP, and the mask itself is only 11% of the filtered search. Everything the paper
claims held up.

**Construction is what kills it.** The peeling step is a sequential graph algorithm — pop a singleton
slot, remove its key, decrement three counters, and any counter reaching 1 becomes a new singleton — so
the dependency is data-carried and the trip count data-dependent. Measured **~2.4 us/key**, flat in N,
projecting to **~60 s at N=24M**. The entire `J = 50` `get_xsource` precompute it would accelerate costs
**~3.4 s**, so the build is ~18× more expensive than the work it saves.

**And it does not vectorize.** The obvious fix — peel all current singletons per round instead of one at
a time — degenerates. Measured at n=20k: round 1 peels 4612 of 20000, but by round 10 it is 160 per round
and by round 30 it is ~80, so peeling 20k keys needs *thousands* of `O(cap)` rounds and runs slower than
the sequential version. This is intrinsic, not a bug in the rounds: at 1.125n the hypergraph sits
deliberately near the peelability threshold, which is precisely what buys the 13% space overhead, and
being near the threshold means few singletons exist at any moment. **The space efficiency and the
sequential construction are the same design choice.**

So the verdict is the reverse of the usual one: the filter is better than Bloom on every axis that
matters at query time, and unusable because of a one-off cost. It would need a C or numba peeling loop
(the reference implementation is C) to be worth considering, and even then it inherits the `cap` problem
that blocks the whole pre-filter family. Bloom stays the better candidate here purely because its build
is one vectorized `np.bitwise_or.at` with no loop and no failure mode.

### The Bloom pre-filter is closed: it cannot reach the path that needs it (2026-08-30)

The six entries above measured the filter itself and settled every open mechanic — the capacity policy,
the sharding, the hoisted precompute, the composition with `xcache_groups`. What none of them measured
is the one thing `markdown/xsources-cache-budget.md` §7 flagged as missing: **"nothing measured through a
full `sqd()` solve."** Measured now, on 1D Heisenberg (`J = n`, `K = n-1`) with a fixture half-closed
under a weight-preserving hop, and the answer closes the line.

**Two structural facts, and either one alone is enough.**

**1. The filter can only attach to the precompute, which is a one-off.** The retry policy is host-side
sequencing (kernel → `int(ncand)` read → possibly a second kernel) and cannot live inside one `jit`.
The uncached recompute is at `sqd.py:1876`, inside `_apply_h_kernel`'s `lax.scan`, and on the partial
path it is `matvec`'s `scanned_tail` arm — called by `ground_locg` every iteration with no host
interposition available. So the only reachable site is the `jax.lax.scan` at `sqd.py:1008`.

Measured as a share of the solve it runs inside:

| n | J | N | precompute, once | `(1,0)` solve | share |
| --- | --- | --- | --- | --- | --- |
| 30 | 30 | 37,817 | 36.0 ms | 469.8 ms | **7.7%** |
| 40 | 40 | 75,241 | 99.9 ms | 1519.4 ms | **6.6%** |
| 60 | 60 | 75,295 | 142.6 ms | 3134.1 ms | **4.5%** |
| 80 | 80 | 75,187 | 356.3 ms | 4246.7 ms | **8.4%** |

Flat at 4.5–8.4% with no trend in `n`. **Amdahl caps the whole idea at 1.09×** end-to-end, and that is
with an infinitely fast filter. The hit rate is not the problem — it is 1.54–4.14%, deep inside the
filter's paying region (break-even ~55%).

**2. The filter is unreachable on the path the memory saving requires.** Saving memory means
`cache_level[0] = 0` or a low `xcache_groups`, i.e. the uncached arm — exactly the site (1) rules out.
So the filter accelerates the path that already fits and cannot touch the path that does not. The
`(1,0) → (0,0) + BF` row in the first Bloom entry above reads as a 4.06× memory win, but `(0,0)` alone
already delivers it: 4.0 GB against 4.1 GB with the filter. **The filter contributes 0.029 GB of
overhead and no memory saving.**

**Why the 5.6–9.4× figures do not transfer.** They are matvec-and-setup-path numbers, and they are
correct as such. Composed into a solve they multiply a 4.5–8.4% share.

### Two percentages that look contradictory and are both right (2026-08-30)

Worth stating separately, because reading one as the other is what made the filter look worth building
and cost a session:

- **"`get_xsource` setup is 66–97% of a solve"** (module docstring, `NOTES.md` above,
  `markdown/scaling-pocs.md`) is **weighted by call count**. It is the cost of paying the `J`-fold search
  *per matvec* against paying it once — i.e. what `cache_level[0] = 0` actually costs.
- **4.5–8.4%** is the precompute measured **once**, as a fraction of the `(1,*)` solve it runs inside.

Both describe `sqd.py:1008`; they differ in how many times the work is counted.
`markdown/skqd-sqd-solve-tolerance.md` already confirmed the first "correct as stated" and named this exact
trap — *"an earlier 3–23% figure measured one `get_xsource` call as a fraction of a solve — a different
quantity."* **Reconciled numerically**, which is what makes them one fact rather than two: at n=40,
J=40, `t(0,0)/t(1,0) = 8.76×` (on `NOTES.md`'s stated trend — 7.2× at J=23, 5.7× at J=12, 9.3× at
J=52) and `t(0,0)/t_precompute = 133`, i.e. the same work paid ~133 times against once.

**So there is no stale claim here to correct.** Quote the 66–97% for "should I turn source caching
off", never as headroom for accelerating the precompute — the second question needs the one-off share,
and Amdahl applies to that one.

### `cache_level[1] = 1` is dominated on both axes, and the "compress the diagonal" premise was wrong (2026-08-30)

Opened as "the diagonal axis is the memory lever, and it is uninvestigated" — which is true of the axis
and false of the framing. **`diag_signs` is already one bit per (state, Z term)**, so the 1313 B/slot at
`J=101, K=100` is not waste to be compressed; it is `J * ceil(K/8)`, the information-theoretic size of
what it stores. There is no redundancy for a general-purpose compressor to find. The finding is simpler.

**Level 1 loses to both of its neighbours, at every `K` measured.** End-to-end `sqd()` solves, n=22,
N=24,674, all six levels returning the same energy (agreement < 1e-8):

| K | J | `(0,0)` | `(0,1)` | `(0,2)` | `(1,0)` | `(1,1)` | `(1,2)` |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 16 | 4 | 1868.7 | 1920.0 | 1735.7 | 291.8 | **338.6** | **153.4** |
| 64 | 4 | 1232.7 | 1375.8 | 1060.4 | 344.3 | **407.2** | **95.7** |
| 128 | 4 | 486.0 | 533.3 | 340.4 | 175.1 | **228.9** | **29.9** |

ms. Holding `cache_level[0]` fixed and moving only the diagonal axis, level 1 is **16–31% slower than
level 0** — which stores *nothing* — and **2.2–7.6× slower than level 2**, on both rows.

**The mechanism, which is why this is structural rather than a tuning accident.** Per X group at
n=24, N=39,736:

| K | ceil(K/8) | L1 store | L2 store | L1/L2 | L0 build | L1 build | L1 *use* | L2 use |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 16 | 2 | 0.131 MB | 0.524 MB | 0.25 | 0.46 | 0.49 | 0.57 | 0 |
| 64 | 8 | 0.524 MB | 0.524 MB | **1.00** | 1.58 | 1.45 | 1.56 | 0 |
| 128 | 16 | 1.049 MB | 0.524 MB | **2.00** | 2.83 | 3.27 | 3.16 | 0 |

**`L1 use` ≈ `L0 build`** (3.16 ms against 2.83 at K=128): unpacking the cached bits costs about what
recomputing the parity from `popcount(state & z)` costs. So level 1 stores `J * ceil(K/8) * N` bytes to
avoid work it then substantially redoes, while level 2 stores the composed sum and pays nothing.

**And it is memory-dominated above an exact crossover.** Level 1 costs `ceil(K/8)` bytes/slot/group,
level 2 costs the coefficient itemsize, so they cross when `ceil(K/8) == itemsize`: **`K = 64` for
float64, `K = 128` for complex128** (an odd-Y string makes the folded coefficients complex — see
`PauliSumXZ`). The measured `L1/L2` column hits exactly 1.00 and 2.00 at those points; this is
arithmetic, not a fit. Below the crossover level 1 is the smaller array, which is its only surviving
claim — and level 0 is smaller still, at zero.

**Independently corroborated by the record.** `markdown/skqd-sqd-solve-tolerance.md` found `(1,1)` "the only
level on the `[0]=1` row slower than `(1,0)`" at n=14/18 and cited it as why `spinchain` exposes only two
of the six levels. That holds at n=22 and now has a mechanism rather than just an observation.

**So the guidance is: never select `cache_level[1] = 1`.** Use 2 for speed, 0 for minimum footprint.
`NOTES.md`'s own budget table above already showed `(0,2)` at 920 B/slot against `(0,1)`'s 1433 and
`(1,2)` at 1296 against `(1,1)`'s 1805 — the comparison had simply not been drawn. Level 1 stays in the
API because the `cache_level` sweep is load-bearing in the suite (three bugs hid behind the default
`(1,0)`, each masked by the one before), not because a caller should pick it.

**What this does not close.** Level 2 at `J=101` is still 808 B/slot, and *that* is irreducible in the
same sense — one composed value per (state, X group). Cutting it needs partial caching on the diagonal
axis, the sibling of `xcache_groups`, which does not exist: `cache_level[1]` is all-or-nothing across
groups. That is the open question this investigation actually surfaced.

**Caveats.** One laptop CPU, n=22–24, N=25k–40k. The fixtures place all Z terms in a single X group,
which isolates `K` cleanly but is not a physical Hamiltonian's structure — 1D Heisenberg spreads them
across `J = n` groups. The crossover arithmetic is structure-independent; the timing ratios are not.

### A partial *diagonal* cache works, composes into a solve, and selects a diagonal-only dial (2026-08-30)

Split at `J'`: `(1, 2)` over `J'` groups, `(1, 0)` over the rest. Real `sqd()` solves (n=100, J=100) give
**half the diagonal memory for 2.45×** (75% for 3.17×), bit-identical; it composes because the cache is
read ~129 times per solve. **Do not split both axes** (41×), and expect **one compile per distinct
`J'`**. `poc/diag-cache.md` §1, §2, §5.

### Sparse transition pairs beat every cache level on memory *and* speed (2026-09-25, prototype)

Item 8 of the 2026-09-25 ideas doc, on branch `sparse-pairs`. **Every table, fixture and caveat is in
`poc/sparse/pairs.md`** (from `poc/sparse/pairs.py`); this is the verdict. At `(1, *)` only 8--22% of the
`4·J` B/slot source cache is real transitions on spinchain's open XXZ; storing each XOR pair once, with its
diagonal computed once (`H_ji = conj(H_ij)` exactly per X signature), measured on CPU against `(1, 0)`:
**P0** 4.4--5.3× faster whole solves at −52--63% memory, **−43% setup-inclusive peak** at `2^21`; **C2R**
3.4--6.2× past the cache at a peak about `(1, 0)`'s. The win tracks the hit rate, not the model -- at high
`h` CSR costs memory. On a spinchain-style recovery-grown subspace (`h` 0.16--0.17 at n=60, 0.354 at
n=20) the memory win holds at `2^21` -- P0 −75%, C2R −34% of `(1, 0)`'s operator -- but P0's speed past
the cache thins to 1.20--1.51×. Two measured lessons: the speedup shrinks with N from **cache locality,
not threads** (CSR order recovers it), and the setup peak needed a **counting sort** by target, per-group
construction and no duplicate transients. **Shipped 2026-09-26 as `sqd(matvec="pairs"|"csr")`**,
single-device: warm `sqd` at n=60 `type1` `2^17` takes 1.26 s / 1.63 s against `"indices"`' 6.25 s
(`poc/sparse/pairs.md` §9). **Its successor is ELLC (2026-09-27):** rows bucketed by degree rounded to a
×1.25 grid, a gather-reduce per row instead of a scatter per entry — 2.0–2.4× C2R per whole solve at a
smaller operator, peak 1.3× C2R's from compiling 19 bucket scans (§10). **Shipped beside `"csr"` as
`sqd(matvec="ell")`** (§9). Open: GPU timing, sharding, a pruned recovery subspace.

### sqd sparse builders: the source search runs on the host, in threads (2026-09-28)

The per-group `get_xsource` loop was 93% of an n=100 `"ell"` build (9.07 of 9.74 s, `type2`, `2^20`
Néel-flip subspace, 200 groups): XLA runs that search on one core. `_host_sources` searches with numpy,
which releases the GIL, one group per thread; a state wider than one word folds each word's rank into its
row index, exact through a word comparison, at any width. `"ell"` 9.74 → 2.49 s, `"csr"` 9.88 → 2.36 s; at
n=60 the search alone 3.6 → 0.46 s. Two later fixes: groups run in a sliding window rather than batches
waiting on their slowest (a batched form cost the n=100 search 3.00 against 2.60 s), and a word an X
signature leaves unchanged takes the state's own rank without a search — 121 and 77 of 199 XXZ groups
leave one of the two words alone, so the n=100 `"pairs"` build fell 3.25 → 2.07 s. Rejected: a numpy hash table (slower than the ranks at `2^17`), and DuckDB, 2.0× faster at
n=100 `2^20` for the whole front end but a new dependency and a second code path. No committed script
reproduces these numbers.

### sqd sparse kernels: the residual check runs on the host (2026-09-28)

The in-jit `"onthefly"` check repeated the J-fold device search, 0.31 s of a 0.74 s `"ell"` solve (n=60
`type2`, `2^17`). `_sparse_residual` now runs after the solve, per group, from host-searched sources and
recomputed diagonals, with the operator freed first; warm `sqd` 0.857 → 0.629 s (`"ell"`), −0.23–0.25 s on
all three, energies bit-identical. It reuses the search as the dense kernels reuse cached xsources.
Rejected for `"ell"`'s per-subspace retrace: a per-`states_size` shape memory with headroom. Growing
Hamming-shell bases move rows to wider buckets (width 19: 1 → 4 → 8 pieces per +12%), so every growing
call still retraced at 1.25–1.5× headroom; constant-size 5% turnover retraced nothing without it. It
saved one compile in nine calls. No committed script reproduces these numbers.

### sqd sparse kernels: one host search for the build and the check (2026-10-02)

`_sparse_residual` re-ran the build's whole per-group host search. `sqd` now searches once
(`_group_pairs`) and the check rebuilds each group's sources from those pairs (`_pair_xsources`: both
directions per pair, own rows for an identity group), at ~8 B/pair of host memory kept through the solve.
Build plus check 1.41–1.53× on an M1 (n=60 `type1`/`type2`, `2^20`/`2^21`, `"pairs"` and `"ell"`, 5/5
rounds each), residual bit-identical; ≤ the 0.58 s check on the GH200 (`poc/sparse/gpu.md` §7), unmeasured
there. No committed script reproduces these numbers.

### sqd sparse kernels: pairs sorted by i (2026-09-28)

`"pairs"` stored its pairs group by group, each group's `i` spanning every state, so all four accesses
per pair (`vec[j]`, `out[i]`, `vec[i]`, `out[j]`) were random. Counting-sorted by `i` across groups
(`_sort_by_target(both=False)`), the `i` side is local: one batched matvec at n=60 `2^20` 151.4 → 69.5
ns/state (`type2`), 55.0 → 28.9 (`type1`), unchanged at `2^17`, where the vectors fit in cache. Whole
warm `sqd` 21.28 → 13.59 s at `2^20`, 1.107 → 1.077 s at `2^17`; same memory, `Hv` within 3.2e-14 from
the new summation order. `"csr"` and `"ell"` already store by target; the dense kernels sweep `out` in
order, their one random access the source gather. No committed script reproduces these numbers.

### Partial diagonal cache: *which* groups to cache barely matters, only how many (2026-09-25)

At equal bytes, largest-`K_g`-first beats a prefix by only **1.02–1.03×** (whole solve 1.023×, n=18 JW)
and gains nothing on spin chains, so an API exposes the count `J'`, not an order.
`poc/diag_cache_order.py`; `poc/diag-cache.md` §6.

### Partial diagonal cache on XXZ: the identity group is the best byte, not most of the win (2026-09-25)

Caching only the identity group (`J' = 1`, ~24 B/slot) gives **1.26× / 1.38×** at n=24/32 against `(1,
2)`'s 2.83× / 3.46×: a third of the recompute, at ~73 ms per MiB against the whole cache's ~29. On XXZ a
prefix already is largest-first. `poc/diag-cache.md` §7.

### The Z parity is already streaming-optimal on CPU: XOR-folding and one-pass forms lose (2026-09-27)

Neither faster form of `popcount(s & z) & 1` beats `get_diagonal`'s per-term loop. XOR-folding ties, or
loses 0.57× as `uint64` words. Reading the states once per group loses 4–9× at `2^15`–`2^21`, both as a
broadcast and as the int8 GEMM Ozaki scheme II suggests. On CPU the lever is caching the diagonal, not
recomputing it faster; the GPU int8 path is open. `poc/parity-xor.md` §2

### The diagonal split at large `N`: the overhead is a *ratio*, and "flat 1.1 MB" was an artifact (2026-08-30)

Peak temp is **16 B/slot**, linear in `N`: **4.0% of the memory saved** at every `states_size` up to
`2^21`, `4/J` in the group count (40% at J=10), net-negative below about `J = 7`. Matvec time holds at
2.6–2.8× up to N=501k; synthetic arrays gave a 25–39× phantom. `poc/diag-cache.md` §3, §4.

### `states_size`'s power-of-two padding: the default is right at small `N` and wastes GBs at large `N` (2026-08-30)

`states_size` rounds up to the next power of two, which inflates **every** per-slot term at once --
states, the solver's vectors, and every cache. `NOTES.md` above notes the 40% at N=24M in passing. The
question is whether a finer bucket is better; the answer is regime-dependent, and the existing default is
correct for the regime it was measured in.

**The default is deliberate and measured, not crude rounding.** SQD's normal access pattern is growing,
all-distinct dimensions (one per Krylov rung plus one per configuration-recovery round), so no two calls
share a size and an exact `states_size` retraces the solver every call. Power-of-two bucketing collapses
that to `O(log N)` traces -- measured 1.25× over five dimensions 60..260 at n=10 and 1.43× over five
rungs at n=13. `sqd`'s docstring already offers the escape hatch: *"Pass `states.shape[0]` for no padding
at all."* So no capability is missing.

**But the trade inverts with `N`, because a compile is a fixed cost and the waste is a fraction.**
Compile is ~0.4 s regardless of size; its share of a cold solve at n=20:

| `N` | cold | warm | compile share |
| --- | --- | --- | --- |
| 2,000 | 0.47 s | 0.01 s | **97.3%** |
| 8,000 | 0.45 s | 0.06 s | 86.8% |
| 30,000 | 0.98 s | 0.62 s | 37.5% |
| 150,000 | 1.19 s | 0.78 s | **34.7%** |

So bucket *count* dominates while solves are fast, and *waste* dominates once they are not.

**Measured both regimes against a `pow2/8` policy** -- round up to a multiple of the largest power of two
at or below `N/8`, capping waste near 12.5% instead of 100%. Real `sqd` sweeps over growing, all-distinct
dimensions:

| regime | policy | distinct sizes | waste | sweep |
| --- | --- | --- | --- | --- |
| n=12, dims 60..860 | `pow2` | 5 | 39.8% | **2.20 s** |
| n=12, dims 60..860 | `pow2/8` | 9 | 4.8% | 3.45 s (**+57%**) |
| n=20, dims 21k..144k | `pow2` | 4 | **62.4%** | 5.10 s |
| n=20, dims 21k..144k | `pow2/8` | 5 | **3.3%** | **5.09 s** |

At small dimensions the finer bucket costs 57% wall clock -- the coalescing is real and the default wins.
At large dimensions it costs **one extra compilation and zero measurable time** (5.09 against 5.10 s is
noise) while cutting waste from 62.4% to 3.3%.

**The waste is data-dependent, not monotonic in `N`** -- `pow2` is free when `N` lands just below a power
of two and costs up to 2× when it lands just above:

| `N` | `pow2` waste | `pow2/8` waste | GB saved at 920 B/slot |
| --- | --- | --- | --- |
| 10,000 | 63.8% | 2.4% | 0.01 |
| 144,000 | 82.0% | 2.4% | 0.11 |
| 1,000,000 | 4.9% | 4.9% | 0.00 |
| **24,000,000** | **39.8%** | **4.9%** | **7.72** |
| 2^28 | 0.0% | 0.0% | 0.00 |

So there is no single figure for what this costs; it depends on where `N` falls. At N=24M and
`cache_level=(0, 2)` it is **7.7 GB**, and a caller can be unlucky at any size.

**Mesh divisibility survives, by construction.** `sqd` rounds `states_size` up to a multiple of
`mesh.size`. A `pow2/8` bucket is a multiple of a power of two at or above `N/16`, hence divisible by
every smaller power of two, so any realistic mesh divides it exactly -- verified at N=144k, 24M and 2^28
for meshes 2/4/8/16/64. No interaction.

**Recommendation: do not change the default; document the large-`N` case.** The default is correct where
it was measured and a caller with growing small dimensions would regress by 57%. What is missing is not a
parameter -- `states_size` is already public and overridable -- but the *knowledge* that at large `N` the
padding is worth sizing by hand, and that the penalty depends on where `N` falls relative to a power of
two. A caller at N=24M passing `states_size=25_165_824` saves 7.7 GB at `(0, 2)` for one extra
compilation. **If a default ever changes, make it size-dependent** (power-of-two below ~10^5, finer
above) rather than replacing one fixed rule with another, since both regimes are measured and they
disagree.

**Caveats.** One laptop CPU, n=12 and n=20, dimensions to 144k; the large-`N` rows in the waste table are
arithmetic on the measured 920 B/slot, not runs. The crossover was located by compile-share, not by
bisecting sweeps, so ~10^5 is an order of magnitude rather than a boundary.

### f32 *storage* for the solver's carried vectors: `ax` cannot be demoted, and it is the one that matters (2026-08-30)

Rejected: an f32 `ax` raises the residual floor 3.1e6× above tolerance (cancellation in `r = Ax - θx`);
at best `r`, `y` save 8 of 120 B/slot (6.7%). The `(0, 0)` floor of 120 B/slot (258 GB at `2^31`) stands,
and POC 6's f32-arithmetic variant still reproduces its reject at 0.42× end-to-end.
`poc/mixed-precision.md` §1–§4.

### The eigensolver's `O(N)` vector count is at its algorithmic minimum; a smaller basis is a bad trade (2026-08-30)

Opened as the remaining lever on the `(0, 0)` floor, since that floor is 120 B/slot of which only 13 is
the Hamiltonian. Two findings: the count is **not** slack, and the one way to reduce it costs far more
than it saves.

**First, the count. Measured, not counted from the source** — `temp_size_in_bytes / (8N)` over a jitted
`ground_locg`:

| | vectors |
| --- | --- |
| total transient, generic tridiagonal matvec | **8.00** |
| same, pure-diagonal matvec (1 temp) | **7.00** |
| → the **solver's own** working set | **7** |

Exactly 8.00 at N = 2^16, 2^18, 2^20 and flat across `maxiter` 1/5/30/100, so it is a per-iteration
working set, not accumulation. One vector belongs to the operator; a real `apply_h` will differ there.

**Seven is the algorithmic minimum for a 3-dimensional Rayleigh–Ritz basis.** The step needs
`{x, y, p}` *and* each one's image `{ax, ay, ap}` live simultaneously to form `sas`, which is 6, plus
`r` to construct `p`. That is 7 — the measured number. **So this is a basis-size question, not a
buffer-reuse question**, and no scheduling or aliasing work can reduce it.

Note the module docstring's "three-vector memory budget" refers to the Rayleigh–Ritz basis `{x, y, p}`,
not the total footprint. It is easy to read as the latter.

**Second, the smaller basis already exists and is already tested.** `body_iter1` is a complete
2-dimensional `{x, p}` iteration (`eigenpair_2x2`, its own exclusion bound, `_project_out`), used for
exactly one step before `body()` takes over. So the variant did not need writing, only looping.

Looped and compared against `ground_locg` at N=16384, `tol = eps(f64)`, on symmetric tridiagonals with
the off-diagonal scaled to vary the gap. **Both converge to the same eigenvalue to every digit
printed:**

| off-diagonal | 3-dim iters | 2-dim iters | ratio | θ (both) |
| --- | --- | --- | --- | --- |
| 0.05 | 94 | 381 | **4.05×** | −3.759790199 |
| 0.20 | 130 | 1543 | **11.87×** | −3.872513788 |
| 0.50 | 115 | 543 | **4.72×** | −4.451636269 |
| 1.00 | 102 | 327 | **3.21×** | −6.289249277 |

**3.2–11.9× more iterations**, median ~4.4×. That is the `y` term earning its vector: dropping it turns
locally-optimal CG into steepest descent, whose rate depends on the condition number rather than its
square root.

**And the saving is smaller than the vector count suggests.** 7 → 5 is 29% of the *working set* but the
floor also carries states and the 4-vector carry, neither of which changes:

| | `O(N)` vectors | B/slot | at `2^31` |
| --- | --- | --- | --- |
| 3-dim (today) | 11 | 120 | **258 GB** |
| 2-dim | 9 | 104 | 223 GB |

**13.3% off the floor for ~4.4× the time.** 13% more `N` at fixed subspace density is worth well under
one extra qubit, so the trade is bad in both directions: as a capacity play it buys almost nothing, and
as a time play it is a large regression.

**Conclusion: closed. Do not pursue a smaller basis.** The 258 GB single-device floor stands, and it is
dominated by terms that are each individually irreducible — 13 B/slot of states that `get_xsource`
requires *replicated*, and a Rayleigh–Ritz basis at its minimum. Lowering it further needs a different
*kind* of change: either distributing `states` (which means replacing the binary search with something
shardable — `jnp.searchsorted` needs the sorted array replicated) or an out-of-core scheme that streams
vectors, neither of which is a tweak to this solver.

**Caveats.** One laptop CPU. The iteration-count comparison uses synthetic tridiagonals, not projected
SQD Hamiltonians; the *direction* is a property of the algorithms (CG versus steepest descent) but the
4.4× median is fixture-specific. The 2-dim loop is a standalone reimplementation using
`ground_locg`'s own primitives, not a patched `ground_locg`, so it shares the primitives but not the
prefilter or the `body_iter0` seeding. Vector counts at `2^31` are arithmetic on measured B/slot.

## Sharding `states` and `uniquify_states`

### Distributing `states` is feasible: hash-by-prefix ownership plus a local search (2026-08-30)

Only `searchsorted` fails on a partitioned `[N, B]`. Routing targets to the owning shard (Wietek &
Läuchli) turns 27.9 GB/device into `27.9/d`; a minimal version is bit-identical to `get_xsource` with
zero all-gathers. Viable only at `cache_level[0] = 1` (~650 GB routed once at d=4, against ~84 TB per
solve at `(0, *)`). DanceQ is not a precedent. `poc/partition-states.md` §1–§2.

### Hash-partitioning `states`: hash the whole key, not the prefix — the prefix collapses on real subspaces (2026-08-30)

Prefix hashing collapses to 16.00× at d=16 on a banded subspace (1 distinct prefix). A whole-key `mix64`
gives 1.01–1.14×, exact on every fixture, its imbalance balls-in-bins in `N/d` alone: keep `N/d` above
~1000 for a slack under 1.15×. `poc/partition-states.md` §3.

### The variable-length routing has a primitive, and the capacity bound is the one the Bloom work lacked (2026-08-30)

`ragged_all_to_all` fits (forward-only, which nothing needs to differentiate) but is UNIMPLEMENTED on
XLA:CPU, so unmeasured. Padded buffers sized by balls-in-bins cost 0.4% at `N = 2^31, d = 64` (26%
penalty in FESOM2-JAX), and that capacity is known at setup, unlike the pre-filter's: raise, don't clamp.
`poc/partition-states.md` §5.

### 1D XXZ is the case that decides it: prefix hashing fails at exactly `d`, and range splitting fails on the *targets* (2026-08-30)

On an XXZ Krylov subspace prefix hashing measures exactly `d` (4.00–256.00×), and range splitting
13.68–14.95× on high-order hops, 32.51× with stale splitters. Whole-key hashing stays at 1.03–1.11×,
exact over all 61 X groups at n=60, d=64. `poc/partition-states.md` §4.

### The JAX composition works and is exact: `poc/hash_partition_jax.py` (2026-08-30)

Dense `all_to_all` over fixed-capacity buckets is bit-identical to `get_xsource` at D=2/4 with only
`all-to-all` collectives; bucket with `D` cumsum passes (13 B/slot, against 24 for a one-hot). The free
overflow count coincides with inexactness (slack 0.9 gives 1284 dropped), so it is a sufficient guard.
`poc/partition-states.md` §6.

### Sharding `uniquify_states`: three gaps, all closable, and the hard one is a global prefix sum (2026-08-30)

Global order forces range partitioning. In-graph splitters must compare the full row (lead-word-only put
4000/4000 rows in one bucket), reassembly works via a drop-scatter, and the sharded `cumsum` needs a
two-level prefix sum: one `all_gather`, `O(d^2)`, independent of `N`. `poc/partition-states.md` §7.

### Composing sharded `uniquify_states`: the algorithm is exact, and it needs *two* routing rounds (2026-08-30)

Buckets' global offsets straddle output-shard boundaries (2 of 4 measured), so it is route → sort/dedupe
→ route again, and that second round is what makes `P('x', None)` honest. Declaring private blocks
replicated is rejected by `check_vma`, correctly; splitters are passed to `shard_map`, not closed over.
`poc/partition-states.md` §8.

### The two-round JAX composition works, and the capacity model for a range partition is *not* balls-in-bins (2026-08-30)

Bit-identical to `uniquify_states` at D=2/4: 157,051 unique rows, 1 all-gather and 24 all-to-all at D=4.
Capacity follows the splitter imbalance (slack 1.35), not balls-in-bins, and round 2 sizes off `ss/d`;
the overflow guard must count live elements only (763,677 false positives before that fix).
`poc/partition-states.md` §8.

### Widening the POC device sweep found a latent `all_to_all` bug a 4-device box hid (2026-08-30)

`run_case` took `mesh` and `num_shards` separately, and 4 devices made them coincide; at 8 it raised, and
`num_shards` now derives from `mesh.shape["x"]`. A hardcoded device count is a coincidence, not a
fixture. `poc/partition-states.md` §9.

### The popcount diagonal path already shards — verified, no work needed (2026-08-30)

The reduction runs over the unsharded byte axis, so all four builders and `apply_h` at `(1, 0)`/`(1, 1)`
keep `P('x', …)` with max diff 0.0 and zero collectives; `test/sharded/diagonals.py` pins it with a spec
assertion, since a value-only test passes the unsharded mutant. `poc/partition-states.md` §10.

### The range-partitioned shuffle works — `poc/range_partition.py` (2026-08-29)

Sample sort with data-derived splitters is bit-identical to `np.unique` with zero all-gather/all-reduce.
Equal-range splitting collapses (4.00×/8.00×), a global `argsort` defeats the design, a one-hot cumsum
costs 34 GB at `2^31`, and overflow is detectable; its two gaps are closed by `poc/uniquify_sharded.py`.
`poc/partition-states.md` §7.

## Convergence: the residual floor and `atol`/`rtol`

### A rounding-floor residual is not zero, and `== 0.0` is the wrong guard

2026-08-28, from `markdown/spinchain/rqutils-prefilter-dim2-request.md`. `body_iter1` formed its search direction as a
bare `normalize(rcurr, norm_r)`. An `xinit` that *is* an eigenvector in floating point leaves a residual
at the **rounding floor** — 3.1e-16 on `[[2.9, 1], [1, 2.9]]` — so the `norm_r == 0.0` guard missed it,
the division amplified pure noise until `tmp_p` came back **parallel to `xcurr`**, and `sas` degenerated
to `[[1.9, -1.9], [-1.9, 4.8]]` whose lowest eigenvalue is **0.96** against a true **1.9**. Iteration 0
still had theta correct with `converged=False`; iteration 1 destroyed it.

**Two plausible fixes measured and rejected, in order:**

1. **Masking `sas[1, 1]`** — the guard that already existed. It *fires correctly*
   (`sas[1, 1] = 4.8 = 2|rho| + 1`) and is still insufficient: the surviving off-diagonal keeps coupling
   `x` to the noise. Lifting a diagonal only works when the off-diagonals are already negligible, which
   is exactly what a parallel `tmp_p` breaks.
2. **A scale-relative residual floor**, `eps * dim * max(|rho|, 1)`. Fixed all 42 cells of the reporter's
   sweep, then **failed a cell that previously passed**: `|r| = 8.07e-16` against a floor of `7.99e-16`.
   A 1% margin deciding correctness. Loosening it pins `theta = rho` when the iterate is not an
   eigenvector — measured, that returned 0.96 *silently*, strictly worse than the reported raise.

**The fix needs no threshold**: `_project_out((xcurr,), rcurr)`, which `body()` has always used and
`body_iter1` was missing. It renormalizes, subtracts the basis again, and returns *exactly* zero when the
norm collapses below 0.99 — "this direction was rounding noise" expressed structurally rather than as a
tolerance. Also drops a redundant norm reduction.

**The defect was not dim-2-specific**, contrary to the report's framing: any near-exact-eigenvector
`xinit` hits it at any dimension (verified 0/120 failures across dims 2–40 after the fix, real and
complex). dim 2 is only where `sqd`'s prefilter lands on the eigenvector routinely.

**A test-isolation trap worth keeping.** The anti-vacuity arm — "a genuine direction must survive" —
cannot isolate `body_iter1`. `maxiter=0` returns `rho_init` and skips the step; at `maxiter=1` `body()`
recovers whatever `body_iter1` discarded. So a mutant zeroing `tmp_p` unconditionally survives that
class and is caught by `TestDtypes` instead. The test says so rather than implying coverage it lacks.

### The eigen-residual floor is `eps·‖H‖` with no dimension dependence, and `tol` is now absolute (2026-08-31)

From `markdown/spinchain/rqutils-tol-request.md` (the `spinchain` side asked for a `tol` that means the eigen-residual,
so their solver criterion and their `_RESIDUAL_TOLERANCE = 1e-6` guard would be one number). Shipped;
reply in `markdown/spinchain/rqutils-tol-response.md`.

**The question that had to be settled first.** An absolute `tol` is only safe if it stays satisfiable as
`N` grows. The old test was `‖r‖ < tol·(‖Ax‖ + |θ|)·N·10`, whose `N·10` factor *asserts* an `O(N)`
rounding budget. If that were real, an absolute bound would become unreachable at large `N`.

**Method.** `debug=True` switches `ground_locg` from `while_loop` to `scan`, so it runs the full
`maxiter` regardless of convergence; with `tol=0` the test is never satisfied and the per-iteration `r`
diagnostics expose the whole trajectory. The floor is the median of the last quartile — past convergence
the residual *oscillates* in rounding noise rather than settling, so a single final iterate reads as a 2x
trend that is pure noise.

**Result — 27 samples, `N = 70..32768`, `‖H‖₂` over six decades, float64 and complex128, dense and
matrix-free:**

| model | min | median | max | spread |
| --- | --- | --- | --- | --- |
| `floor / (eps·‖H‖)` | 0.494 | **0.839** | 1.260 | **2.6x** |
| `floor / (eps·‖H‖·N)` | 1.9e-05 | 2.4e-04 | 5.8e-03 | 306x |

```text
floor(‖Hv − Ev‖)  ≈  eps(dtype) · ‖H‖₂        no N dependence
```

The `N·10` factor was **slack, not a rounding budget** — 700x to 94600x looser than the floor across the
sweep. That is why no single value of the old relative `tol` could be both fast and admissible for a
caller with a fixed residual requirement: the same `tol` meant a different absolute residual at every
`N`. The request's own table shows it — `tol=1e-12` gave 5.0e-06 at one size and 4.4e-06 at another.

Three arms, because one sweep alone would not have distinguished the models:

- **Scale invariance** (the decisive one). Fixed `N = 800`, `H` scaled over 10⁻³..10³: `floor/(eps·‖H‖)`
  stayed in [0.51, 1.16] with no trend. Isolates the mechanism from the size sweep.
- **Matrix-free matches dense.** Same subspaces, `floor_mf/floor_dense` ∈ [0.66, 1.57] — straddling 1.0
  with no bias, i.e. two samples of one noise floor. So the packed-scan `Ax` carries the same constant as
  a dense matvec and the floor reached through `sqd()` is the one measured. Note `N` there is the
  *padded* `states_size`; had the floor been `O(N)` the padding would have shown as a systematic `> 1`.
- **A third, unlooked-for confirmation.** `poc/sharding` independently reports `‖Hv−ev‖/‖H‖` of 5.5e-16
  and 6.6e-16 — a script written for another purpose.

**Why `‖r‖` carries no `N`:** it is a vector *norm*, dominated by the per-element relative error in
computing `Ax`, not by a sum that grows with the element count.

**What shipped.** `‖r‖ < tol`; `tol=None` → `4·eps·max(‖Ax₀‖, 1)`; a below-floor `tol` raises from `sqd`
with `4·eps·Σ|c_k|` (`Σ|c_k|` is already computed for `prefilter_hi`, and is a measured 1.56–1.90x
over-estimate of `‖H‖₂` on 1D XXZ — the safe direction). Raised, not clamped: the floor is computable
from the operator alone. The guard cannot live beside the gate it guards — `converged` is a traced
boolean inside a `while_loop` — so it sits in `sqd`, the outermost point where `Σ|c_k|` is concrete.

**Two measurement errors made and corrected, both worth keeping.**

1. **A non-monotonic "floor" is an unconverged trajectory, not a floor.** The first large-`N` sweep
   reported constants from 1.2e3 to 4.8e7, non-monotonic across five decades — which no rounding model
   produces. `maxiter=120` had left the residual descending **799–2096x within its final quartile**, so
   the tail median sampled a live trajectory. Raising to 700–900 brought the descent factor to 0.57–1.27
   and the floor to the predicted value. A **plateau gate** (`r[75%]/r[-1] < 3`) now rejects such rows.
   Had only large `N` been run, the artifact would have read as "the matrix-free path has a much higher
   floor" — plausible and entirely wrong. This is "a broken arm flatters its own benchmark" in mirror
   image: the under-converged arm reports a *worse* number, which is why it was catchable.
2. **`tol` does not retrace the solver, and a cold call is not a measurement.** An earlier draft of the
   response claimed `tol` enters the jit cache key and used that to decline publishing a speedup. It is a
   **traced** argument — `run_sqd._cache_size()` stays at 1 across three values — and the 0.327s-vs-0.003s
   observation behind the claim was one cold call. Warm, all arms: **5.06 → 4.21 → 3.21 → 2.58 ms** for
   `None → 1e-12 → 1e-9 → 1e-6`, i.e. **1.96x** at `tol=1e-6`, monotonic. Verifying the claim produced
   the number.

**A claim deliberately left open.** Whether a residual-targeted `tol` compresses the reported 13.5x
draw-to-draw variance. Iterations-to-`‖r‖<1e-8` track the relative spectral gap (2.3x rise against a 2.0x
relgap fall) while `N` rises 16x — but relgap and `N` are correlated in the XXZ family, so this does *not*
separate them. Settling it needs a draw-to-draw sweep at fixed `N` and fixed Hamiltonian, i.e. their
fixture. Recorded as suggestive, not decisive.

**A mutation-survival mechanism distinct from the two already in `CLAUDE.md`.** Seven of nine new tests
passed against a mutant restoring the relative form — right layer, right branch, fixture too *small*: at
`N=21` the old threshold was ~4200x looser, so the solver's own overshoot satisfied the absolute
assertion. Only a fixture large enough for the scaling to bite (n=10, 200 states → 4.967e-05 against a
requested 1e-8) or one *varying* dimension (4.006e-05 vs 6.331e-11 from one `tol`) discriminated. If the
defect is in how something **scales**, the fixture must span that axis.

**The default got slower, and the "unchanged" claim was asserted rather than measured.** The first
write-up of this entry said `tol=None` behaviour was essentially unchanged. Measured (`tol=None` on both
sides, warm, best of 5, A/B against a worktree of `c400fae`): **1.18–1.49x slower, median 1.33x, and the
gap grows with `N`** — 1.22x at N=200, 1.33x at N=800, 1.41x at N=2898, 1.49x at N=9460. Energies
bit-identical; the residual goes from ~1e-10 to ~1e-14, and that is where the time goes.

| `N` | before | after | slower | resid before | resid after |
| --- | --- | --- | --- | --- | --- |
| 200 | 1.06 ms | 1.29 ms | 1.22x | 1.58e-11 | 8.82e-15 |
| 800 | 3.48 ms | 4.64 ms | 1.33x | 7.34e-11 | 1.02e-14 |
| 2898 | 18.68 ms | 26.32 ms | 1.41x | 2.35e-10 | 1.23e-14 |
| 9460 | 81.19 ms | 120.79 ms | 1.49x | — | — |

**The mechanism makes the direction inevitable, so no benchmark was needed to *suspect* it:** the old
default was `eps` compared against `tol·(‖Ax‖+|θ|)·N·10`, so its *effective* absolute bound carried an
`N` factor; the new default `4·eps·‖Ax₀‖` does not. The ratio is ~`N·10/4` — ~5,100x at N=800, ~10⁶ at
N=2e5. Two bounds differing by a factor of `N` cannot be equivalent, and the discrepancy has to grow
with `N`. **A claim that behaviour is unchanged is a claim about a measurement**; the repo rule to A/B
whole calls against a worktree of the pre-change revision existed and was not applied until asked.

The `N` factor was deliberately *not* put back into the default: a default meaning 1e-10 at one size and
1e-8 at another is the property the change existed to remove. That is a judgement call, and the reply
offers to reverse it as a two-line change if the caller prefers the old timing. The caller's fix is to
pass `tol` explicitly — at 1e-6 they are ~1.5x ahead of the *old* default, so the change only pays if
the parameter is used.

> **Superseded 2026-09-01.** The offer above was taken up, and then withdrawn on measurement. `tol` is
> gone; convergence is `max(atol, rtol·(‖Hv‖+|E|))`. The `N` factor came back with `rtol` and lasted one
> commit — see the next entry. The **floor measurement in this entry still stands** and is the evidence
> base for the new design; only the `tol`-shaped conclusions below it are stale.

**Also unaddressed, and stated in the reply:** `p_is_zero` still reports convergence regardless of `tol`
(the stationary-point route, pre-existing and by design), so a converged result *can* carry a residual
above `tol`. And a residual bound is not an accuracy guarantee — `|E − λ_min| ≲ ‖r‖²/gap`, so the energy
error depends on a gap the caller does not have; measured ΔE was 1–7 decades *better* than the residual at
every arm, but that is the well-conditioned regime, not a promise.

### `atol`/`rtol`: the pair is right, and `rtol`'s scale took two tries to get right (2026-09-01)

From `markdown/spinchain/rqutils-atol-rtol-request.md`; reply in `markdown/spinchain/rqutils-atol-rtol-response.md`. `tol` is gone,
convergence is `‖r‖ < max(atol, rtol·(‖Hv‖ + |E|))`, either arm sufficing.

**The pair itself was never in doubt** — a purely relative test cannot name a residual, a purely absolute
one cannot track an operator whose scale the caller does not know, and `max` makes both special cases. The
`max` is load-bearing: a `min` would require *both*, which is strictly less expressive than either alone.

**`rtol`'s scale shipped wrong once.** The first cut multiplied by `n · 10`, exactly as the request asked,
because that reproduces the pre-2026-08-31 relative `tol` bit-for-bit. Probing the parameter *across its
range* — which should have preceded the commit, not followed it — found three failures:

1. **The bound can exceed `‖H‖`.** Every normalized `v` has `‖Hv − Ev‖ ≤ ‖H‖`, so past that the test
   carries no information: `rtol=1e-8` at `n=2^20` gave a bound of **4.2 against `‖H‖ = 20`**, the first
   iterate reported convergence, and the eigenpair was arbitrary with `converged=True`.
2. **It saturated.** `rtol=1e-6` and `rtol=1e-4` returned bit-identical answers — 100x of dial, one
   outcome.
3. **It was unstatable.** `rtol=1e-8` meant 4.1e-3 at `n=1024` and 8.4e0 at `n=2^21`.

The scale is now `‖Hv‖ + |E|` alone. That factor is required — `‖r‖` has units of `‖H‖`, so a
dimensionless `rtol` needs it — and `n · 10` is not, the floor having no `n` term (previous entry).
Verified dimension-independent: one `rtol`, one Hamiltonian, four sizes gives **2.03e-14 at N=64 and
2.19e-14 at N=1024**, flat across 16x.

**The generalizable rule: a tolerance parameter must be statable without knowing the problem size.** If
its useful range moves with `n`, callers cannot reason about it and its top end silently becomes an
accept-anything. Folding a dimension count into something named "relative" is what made all three
failures possible at once.

**The cost is real and was accepted deliberately:** one `rtol` no longer scales across dimensions, which
is the property the requester wanted for an end-to-end lock spanning several subspace sizes. Their
constants will likely still fail. Three routes offered in the reply, the recommended one being a single
`atol` (the bound no longer varies, so there is nothing for the constants to track).

**A guard whose argument is about a derived quantity must be checked against every input reaching it.**
Both arms now reject accept-anything — `rtol >= 0.5`, `atol >= Σ|c_k|` — but the `atol` half was **missing
from the first implementation**, even though the guard's own error text argues from the *bound* ("every
normalized vector satisfies `‖Hv − Ev‖ ≤ ‖H‖`"), which says nothing about which parameter produced it.
Measured consequence: `atol=100` against `‖H‖ = 17` was accepted and converged in **one iteration**. It
was found only because someone asked whether `atol` had been reviewed as carefully as `rtol`; it had not.
The salience asymmetry is the lesson — `rtol` got scrutiny because its failure had just been *measured*,
while `atol` felt safe for having been designed rather than inherited.

**The below-floor guard is conditioned on `rtol == 0`.** With a live relative arm an unreachable `atol` is
harmless, so an unconditional check would fire on correct input — the defect class this repo already paid
for with an overflow count that included padding.

**`rtol=None` is kept, and the asymmetry with `atol` is principled.** `rtol`'s default is the *promoted*
dtype's epsilon and cannot be a literal: a hardcoded `4·eps(f64)` = 8.88e-16 converges in **28 iterations
at float64 and exhausts a 500-iteration cap at float32** (`converged=False`, so `sqd` would raise). `atol`
has no dtype-derived value a caller would want, so it takes `0.0` and rejects `None`.
`TestRtolNoneIsDtypeDerived` pins this, and its failure message says to revisit the asymmetry if it ever
starts passing. `rtol=None` resolves to `4·eps`, not `eps`: the scale is ~`2‖H‖`, so `eps` would target
only 2x the floor, inside the 0.49–1.26 spread of the floor's own constant.

**Two test defects, both of the "passes for the wrong reason" kind.**

- The dimension-property test compared fixtures differing in **both** qubit count and subspace size, so
  `‖H‖` moved alongside `N` — and it passed against a formula with no dimension term at all. A test that
  varies two axes cannot attribute a difference to either. Rewritten to move one at a time.
- A boundary test asserted `atol == Σ|c_k|` exactly and **did not raise**: `sqd` sums the *padded*
  coefficient rectangle, so its value is one ulp higher (`6.47309024676594` against the test's
  `6.473090246765939`). The test was asserting floating-point associativity, not the guard. Exact-boundary
  tests on independently-computed floats are unsound; test strictly inside the region.

**And a figure that did not survive the redesign.** The `atol=1e-6` speedup is **1.85x** against the new
default (4.60 → 2.49 ms warm, best of 5, N=800). An earlier draft carried over 1.96x, measured against the
`n · 10` default — a different quantity. A tolerance ratio is only meaningful beside the definition it was
taken under, which is the same trap the requester's own table fell into.

## The eigensolver: Davidson, re-orthogonalization, `Ax` reuse, restarts

### Davidson vs `ground_locg`: the two regimes split, and memory is the axis that decides it (2026-09-02)

Davidson's matvec advantage is bought with memory: 51 vectors per eigenpair at depth 25 against
`ground_locg`'s 8, 5.6× fewer matvecs for 5.4× the vectors. At matched memory (synthetic, n=512) it is
1.9× better diagonally dominant and 1.4× worse non-dominant, `sqd`'s regime; `ground_locg` also converges
at `4·eps` where Davidson does not, and `n_keep = max_dav - 1` stalls. `poc/davidson.md` §1–§2.

### Davidson vs `ground_locg` on `xxz_krylov` through `apply_h`, prefilter on both arms (2026-09-02)

The matched-memory result holds on a physical fixture (N=8000): at 8 vectors unfiltered is a near-tie
(112 vs 156, 125 vs 129) and filtered goes to `ground_locg` (57 vs 67, 48 vs 68); the shared 66-matvec
prefilter gives `ground_locg` 1.11–1.27× and every Davidson arm 0.46–0.96×. Count matvecs analytically,
`3 + 3*niter`. `poc/davidson.md` §3–§4.

### The fixed 2-pass re-orthogonalization beats `diaglib`'s adaptive loop on its own metric (2026-09-02)

`_project_out` runs `_subtract_projections` twice, then twice more; `_reorthogonalize` defaults
`passes=2`. Both counts are **fixed**, where the reference implementation for the paper this module
follows — `diaglib` (Molecolab-Pisa, LGPL, `diaglib.f90`) — iterates *adaptively* on a measured
orthogonality defect: `ortho(X)` loops `while ‖XᵀX − Id‖ > τ_ortho` and `ortho(X,Y)` loops
`while ‖YᵀX‖ > τ_ortho`. That is the one idea in arXiv:2305.06668 that is **not** block-only — at
`m = 1` the criterion is just `|⟨x|y⟩|` — so it needed measuring rather than dismissing.

**The fixed count already over-satisfies their threshold, by two orders of magnitude.** Achieved
`|⟨x|y⟩|` over the diagonal-shift axis (the one that stresses this: `_reorthogonalize`'s docstring
records `|⟨x|y⟩| = 1.0` at shift 1e9 *without* it), `dim=64`, 120 iterations each:

| shift | max `\|⟨x\|y⟩\|` | median |
| --- | --- | --- |
| 0 | 1.26e-16 | 1.39e-17 |
| 1e3 | 1.11e-16 | 2.71e-17 |
| 1e6 | 9.02e-17 | 1.39e-17 |
| 1e9 | 1.11e-16 | 2.78e-17 |
| 1e12 | 8.33e-17 | 1.74e-17 |

`diaglib`'s shipped threshold is `tol_ortho = 2·eps ≈ 4.4e-16` (**note: the paper states 1e-14, which
does not match the code** — 20000× looser than what it ships). So an adaptive loop here would exit at its
first check on every iteration measured, buying nothing and costing the check: an extra O(N) reduction per
call, in a routine that already returns its norm specifically so no second reduction is needed. **Do not
replace the fixed counts with a measured loop.**

**Their growth-factor optimization is real and structurally inapplicable.** `diaglib` avoids the recheck
the paper calls "wasteful" by *predicting* the defect instead of measuring it: `ortho_cd` accumulates
`growth = Π‖L⁻¹‖` and estimates `error = eps·κ(L)²`, then `ortho_vs_x` uses `xu_norm = growth·eps` in
place of a `dgemm`-plus-norm. Good engineering, but the thing it saves is an `m × k` `dgemm`; at `m = 1`
the overlap is one inner product whose norm `_project_out` already has. Nothing to avoid.

**And 2 cannot be reduced to 1.** The shift sweep above measures *identically* at one pass
(7.7e-17–1.4e-16), i.e. this fixture cannot see the difference — but the suite fails immediately on
`TestProjectOut::test_orthogonal_vector_is_not_normalized_to_unity` (`Norm 0.0 dropped below 0.99`). A
worked instance of CLAUDE.md's "fixture too small" trap: the defect is in a branch a well-conditioned
operator never reaches, so measure the *invariant the tests assert*, not just the quantity of interest.

**One transferable observation about tolerance design.** `diaglib`'s residual test is
`‖r‖/√n < tol .AND. max|r| < 10·tol` — **conjunctive**, and the `1/√n` makes the permitted `‖r‖` grow with
dimension (at their own full-CI size, 18 360 640 determinants, `tol=1e-9` permits `‖r‖ = 4.3e-6`, 4300×
looser than at `n=1000`). That looks like the exact defect the entry above records for `rtol`'s deleted
`n·10` factor, and it is **not**, because `max|r| ≥ ‖r‖/√n` always, so the dimension-*independent* arm is
binding at every size checked. The rule is therefore sharper than "no `n` in a tolerance": a
dimension-dependent arm is harmless in a **conjunction** (it can only tighten) and fatal in a
**disjunction** (it becomes accept-anything). `ground_locg` uses `max(atol, rtol·scale)` — a disjunction —
which is why the `n` factor had to go there.

### Reusing `Ax` to cut the matvec count is closed, and the canary that should have caught it was one seed (2026-09-02)

Opened from Nottoli/Giannì/Levitt/Lipparini, *Theor. Chem. Acc.* **142**:69 (2023) (= arXiv:2305.06668),
whose §3.1.2 "reuse of applications" obtains `AX^[k+1]` and `AP^[k+1]` as `(AV)u` instead of fresh
products. `body()` spends **3 matvecs per iteration** (`ay`, `ap`, `axnext`; jaxpr
`while.body_jaxpr: dot_general=3`) and the third is removable: `xnext = (x κ₀ + y κ₁ + p κ₂)/‖·‖`, so
`axnext = (ax κ₀ + ay κ₁ + ap κ₂)/‖·‖` from images already formed for `sas`.

**It is fast and it silently breaks the residual.** Warm, arrays as arguments:

| | baseline | 3→2 | speedup |
| --- | --- | --- | --- |
| dense N=2048 / 4096 | 376 / 1507 ms | 252 / 996 ms | 1.49× / 1.51× |
| `sqd` n=16 / n=18 | 1157 / 6609 ms | 885 / 4654 ms | 1.31× / 1.42× |

Iteration counts and eigenvalues are unchanged. But in float32 the **reported residual understates the
true one by 59×** (1.23e-08 against 7.23e-07, the true value recomputed in f64 from the returned `x`, `θ`),
and with `atol=rtol=0` it reaches exactly `0.0`. `r = axnext − θ·xnext` subtracts two quantities that
*share* the staleness in `axnext`, so the error cancels out of `r` while remaining in `Ax`. Injecting a
known `ε` into `ax` shows it directly: reported `‖r‖` tracks `ε` (1e-8 → 1.0e-08, 1e-6 → 1.0e-06) while
the true residual stays at 1.6e-15. Memory also rises **8.00 → 9.00 vectors** (64 → 72 B/slot): holding
both images until `axnext` extends their live ranges, which XLA had been reusing.

**The root cause, which subsumes every repair attempt below.** The reused image's error **originates in
the operator application, not the reconstruction**. Isolated: `‖f32 matvec(x) − f64 A·x‖ = 1.2127e-07`,
and `‖exact-f64 combination of the f32 images − A·x_next‖ = 1.4258e-07` — the drift is fully present with
an *exact* combination. It is the matvec error already inside `ax`/`ay`/`ap`, mapped through `κ`. So every
technique that improves *how the three terms are combined* operates on a stage where the error does not
live. That single fact predicts all of the arms below, and is the reason to stop rather than try another.

**This matches the published bounds, so it is not an artifact of this solver.** The mixed-precision CG
error analysis (arXiv:2510.11379; and van der Vorst's earlier statement of the same point) finds that the
**matvec's** rounding dominates the residual gap, that computing inner products in higher precision does
**not** substantially reduce it, and that "to reduce the size of the residual gap, it is necessary to
compute an accurate matrix-vector product."

**And the strongest form of compensation has been measured, by someone else, and it buys nothing.**
Mukunoki, Ozaki, Ogita & Iakymchuk (*HPCAsia 2021*, doi:10.1145/3432261.3432270) run CG with **every**
inner product *and every matrix-vector multiplication* computed **correctly rounded** via the Ozaki
scheme — error-free transformations, i.e. the ceiling of what any compensation can achieve, strictly
beyond Kahan or Dekker. Their Table 2(a), relative true residual against plain FP64 over 8 matrices:
**median 1.002×, range 0.930–1.147×** — i.e. *unchanged*, and **worse in 3 of 8 cases**. Iteration counts
do improve (median 1.067×, up to 1.385×) and reproducibility is achieved, which is the paper's actual
goal; attainable accuracy is not. So correctly-rounded arithmetic — the limit of the entire compensation
family — does not move the residual gap even when applied to the matvec itself.

The two levers are therefore a *fresh* matvec (baseline) or a *higher-precision* one, and the latter is
already closed here (`ax` demoted to f32 raises the residual floor 3.1e6×, entry above). The Ozaki /
error-free-transformation line (OzBLAS) is aimed at reproducibility and accurate BLAS, not at this gap.

**The metric that matters is honesty, not the spurious count.** Honesty is `TRUE‖r‖ / reported‖r‖` at the
exit iterate over 40 f32 seeds (1.0 = honest, >1 = understates); spurious is `converged=True` under
`rtol = 4·eps64` on a f32 operator, which must be unsatisfiable.

| arm | honesty (median) | worst | spurious | `sqd` n=18 |
| --- | --- | --- | --- | --- |
| **baseline** | **0.95** | 1.70 | **0/100** | 1.00× |
| naive `(AV)κ` | 9.44 | ∞ | 137/200 | 1.42× |
| `jnp.sum` over stacked axis | — | — | 66/100 | — |
| Kahan/Neumaier sums | 9.54 | ∞ | 49/200 | ~1.4× dense |
| Dekker/Veltkamp exact products + Kahan | 6.51 | ∞ | 140/200 | ~1.4× dense |
| widened to f64 (arithmetic ceiling) | — | — | 140/200 | — |
| `|κ|`-ascending order | 10.15 | 50.7 | 23/200 | — |
| signed-κ ascending | 11.07 | 41.6 | 23/200 | — |
| signed-κ descending | 9.98 | 505.3 | 38/200 | — |
| elementwise ascending order | 8.13 | ∞ | 57/200 | — |
| cycled order (`niter % 6`) | 13.60 | 354.0 | 43/200 | — |
| periodic refresh, k=8 | — | — | 58/100 | 1.35× |
| residual-growth trigger | — | — | 10/25 | — |
| Kahan + k=8 | — | — | 57/200 | ~1.3× |
| **θ-stagnation trigger** | — | — | **0/100** | **1.10–1.14×** |
| **Kahan + k=2** | — | — | **0/100** | **1.10×** |

**Every reuse arm sits at honesty 6.5–13.6× against baseline's 0.95, regardless of the arithmetic.**
Arithmetic accuracy and residual honesty are **uncorrelated**: the arm with the best possible arithmetic
(Dekker, matching `best(1 rounding)` exactly) is 6.51, the worst arithmetic (cycled order) is 13.60, and
`|κ|`-ascending is 10.15 despite a 26% error reduction. **Do not rank these arms by spurious count** — it
is threshold-dependent and rewards a residual that is inflated-but-wrong: `sortkappa`'s 23/200 is the
*best* spurious count and the *second-worst* honesty, and it still fails the canary on seed 17.

Findings that make each family structural rather than mistuned:

- **Sums: a 13% ceiling.** Exact f64 summation of the *same* three inputs improves the step error only
  13% (1.348e-07 → 1.177e-07). `jnp.sum` is **bitwise identical** to chained adds here — its advantage is
  reassociating a long axis, and a 3-long axis admits only `(a+b)+c` (measured gain 1.00× at 3 and 8
  terms, 3.41× at 64, 7.68× at 1024). `einsum`/`tensordot` are slightly *worse* (2.38e-07).
- **Products: the gap is real, closing it does not help.** Product rounding is **56.9% of the error by
  RMS** — a genuine gap that Kahan-on-sums leaves untouched. Dekker/Veltkamp two-product recovers it
  exactly (`p+e` reproduces `a*b` with **zero error on 100%** of elements) and the combination then
  matches `best(1 rounding)` (max 2.3785e-07 against naive's 3.5826e-07, a 43% reduction), at **no new
  N-element temporaries and no measurable time**. It is nonetheless *bit-identical to the f64-widened arm*
  and scores the same 140/200 — both reach the arithmetic ceiling, and the ceiling is not where the error
  is.
- **Ordering: exhausted, and uncorrelated.** Six orderings measured. Term order alone spans 2.64e-07 to
  3.58e-07 (a 35% effect); ascending is optimal and descending worst, as theory says. Signed-κ ordering is
  **not a distinct strategy** — it agrees with `|κ|`-ordering on **88.3%** of iterations (both put the
  dominant `κ₀` last) and is bit-identical on a realistic `κ`; signed-*descending* degenerates to naive
  exactly. Cycling all six permutations to *decorrelate* the error is the **worst** arm (13.60, max 354×):
  error coherence was helping, since a fixed order makes the drift a smooth function of the iterate that
  partly cancels in `r = ax − θx`, while varying it injects fresh noise each step.
- **A carried compensation term is well-defined, exactly propagable, and useless.** The error recurrence
  is *exactly linear in the same* `κ`: `d_next = (dx·κ₀ + dy·κ₁ + dp·κ₂)/ν`, verified to **3.6e-16**
  against `‖d‖ ≈ 8.6e-08`. But seeded from a genuinely fresh state (`d = 0`) it stays **exactly 0.0**
  while the true drift grows to 1.8e-07 — the recurrence transports *inherited* error perfectly and is
  blind to error *created* per step. Seeding it from the Dekker tails plus the Kahan compensation is still
  exactly 0.0, because with exact products and compensated sums **there is no rounding in the combination
  to capture**. To seed `d` at all you must compare a reused image against a measured one, i.e. pay the
  matvec — so the scheme degenerates to periodic refresh plus **+3 carried O(N) vectors** (8.00 → 11.00,
  +37.5% on the floor). Strictly dominated; do not build it.
- **`⟨x|r⟩` cannot serve as an in-solver drift detector.** The Rayleigh-Ritz invariant does hold — `θ`
  equals `ρ = ⟨x|Ax⟩` to ±8.9e-16, so `|⟨x|r⟩|/‖Ax‖` sits at ~1.2e-07 ≈ f32 eps in the baseline (normalize
  by `‖Ax‖`, **not** by `‖r‖`, which is eps/eps ≈ O(1) near convergence and reads as a spurious failure).
  But it is **flat across every arm** (median-of-max 3.74e-07 baseline against 3.50e-07 naive — the broken
  arm reads *lower*), and as a discriminator it points backwards: spurious runs median 4.0e-13, honest runs
  3.6e-07, ranges overlapping over seven decades. The reuse error is **overwhelmingly perpendicular** to
  `x` (‖err‖ 6.9e-07 = 2.0e-07 parallel + 6.7e-07 perpendicular), and `⟨x|r⟩` is blind to the
  perpendicular 71% by construction. Projecting the parallel part out
  (`rnext -= xnext·⟨xnext|rnext⟩`) takes 66/100 → 55/100 — it repairs 29% of a symptom — and is a harmless
  no-op on the honest baseline. The general rule: **an invariant that `θ` protects cannot report on
  `ax`'s staleness**, because `θ` is computed from `sas`, i.e. from the fresh-matvec side.
- **Kahan does not compose with refreshing.** Plain vs +Kahan at k=2/4/8/16: 0/0, 10/9, 58/57, 98/96.
  The refresh period alone decides the outcome; per-step compensation is irrelevant once error compounds
  over k iterations.
- **`‖κ‖ = 1` is necessary but not sufficient.** The paper's safety argument is that `u` is orthogonal so
  `(AV)u` loses no precision. `eigenpair_3x3` already returns a normalized vector — measured
  `‖κ‖ = 1.000000` at every iteration — so the condition *held* while the variant failed. The bound
  controls per-step **amplification** (‖u‖=1 → error stays O(ε); ‖u‖=51 → 50× worse), not **accumulation**
  across iterations. Their Algorithm 1 line 27, `AW^[k+1] = A W^[k+1]`, is a genuine matvec, so one fresh
  product per iteration re-injects exact information structurally — the role the baseline's third matvec
  plays here.

**The literature calls this "loss of attainable accuracy from recurrence-updated residuals"** (Greenbaum,
*SIAM J. Matrix Anal. Appl.* 18(3):535–551, 1997) and its remedy is **replacement, not compensation**
(van der Vorst & Ye, *SIAM J. Sci. Comput.* 22(3):835–852, 2000; Sleijpen & van der Vorst, *Computing*
56(2), 1996). Two things from that literature close the ledger: replacement is **total** — the pipelined
BiCGStab application (arXiv:1612.01395) resets six quantities, where refreshing only `axnext` leaves
`ay`/`ap` contaminated, i.e. the shape tested here was wrong — and it costs **+22.1% iterations**
("delayed convergence"), which is about what 3→2 saves. That cancellation is why both clean arms land at
~1.10×, from unrelated mechanisms. Their k is also chosen "ad hoc" and "relatively arbitrary", the same
objection that blocks the θ-trigger's `THETA_REL`.

**Verdict: do not reuse `Ax`.** The honest trade is ~1.10× through `apply_h` for a convergence test that
must be justified empirically. Both clean arms are also *slower* than the third matvec is expensive once
their iteration penalty is counted. Unbuilt, and the only thing worth measuring if it is ever reopened:
**full** replacement (refresh all three images on a rare trigger), the shape the literature actually
prescribes. Note the dense figures (1.5×) are real — this is wrong for *this* solver's cost profile, not
wrong in general.

**The canary was decided by luck, and is now a 20-seed sweep.**
`TestRtolNoneIsDtypeDerived::test_a_float64_literal_rtol_cannot_converge_in_float32` asserts a
float64-tight `rtol` cannot converge in float32. It ran on `seed=3` only, and **eight broken variants
passed it** — naive, Kahan, k=8, Kahan+k8, residual-growth, `|κ|`-sorted, signed-κ-sorted and
cycled-order — because seed 3 is favourable. Both arms now
`@pytest.mark.parametrize("seed", range(20))`: 40 tests, 0.37 s, and mutation-tested to fail all eight
(14, 7, 10, 10, 9, 1, 1, 3 of 20 seeds respectively) while both clean arms still pass 40/40. **The margin
varies a lot** — the two sorted arms fail on only **1 of 20**, so 20 seeds is not generous and a future
arm could still slip through; the honesty ratio above is the more sensitive instrument and the one to
reach for when auditing a change here. The generalizable rule: **a canary guarding a silent-wrong-answer
defect must sweep its fixture, not sample it** — a single draw gave eight broken implementations a ~40%
chance of clearing the gate, and CLAUDE.md's "sweep `cache_level`, don't sample it" is the same lesson on
a different axis.

**One reusable mechanical fact:** `lax.cond` restores XLA buffer reuse. The naive variant's +1 vector is a
*liveness* artifact, not an algorithmic requirement — every arm with a branch on the reuse path measures
8.00 vectors again.

**Also settled, on the way past:** compensating the solver's O(N) reductions (`⟨x|Ax⟩`, norms) is
pointless and unshardable. `jnp.sum` is already tree-reduced — relative error ~1.5e-16, flat in N, against
10–25× worse for naive sequential — and a blocked Kahan `compute_sas` leaves the residual floor
**bit-identical** (7.8019e-15 at dim=256, 1.9374e-06 at dim=1024) with identical per-seed iteration counts.
It also *fails* `test/sharded/locg_prefilter.py`: compensation needs sequential accumulation over blocks, so
it reshapes a partitioned axis and `lax.scan` raises `0th dimension of all xs should be replicated. Got
P('x',)`. **Any compensated reduction here is incompatible with the sharded path**, independent of whether
it helped. θ's error only rivals `‖r‖` at the floor anyway (4.4e-16 against 1.6e-15), and above it
contributes nothing.

**And a refinement to the `prefilter_hi` rule.** `pstuermer/LOBPCG` estimates `‖A‖` by 10 power
iterations — the construct `ground_locg` deleted for returning an excited eigenpair with
`converged=True`. Not a contradiction: theirs feeds only the residual-norm *denominator*, where
under-estimating merely tightens the test. The precise rule is that a power-iteration estimate is fine as
a **convergence-test scale** and unsafe only as a **spectral-filter bound**, which needs a true upper
bound (Kuczyński–Woźniakowski). Nothing else in that repo transfers: it is block C/OpenMP/MKL, so its
locking, SVQB, `ortho_drop` and Gram caching all need `m > 1`, and its memory work is manual-allocator
hygiene (64-byte alignment, `wrk3: 3ns → max(ns, 9s²)`) that a compiler-managed backend does not have.
Its `project_back` even carries an extra `size × sizeSub` buffer plus a `memcpy` where `ground_locg`
writes `xnext` directly. Its residual test `‖W‖/(ANorm + |λ|·BNorm)` is, with `B = I`, exactly this
module's `scale = ‖Ax‖ + |θ|` — independent corroboration of that choice.

### The zeroed-`p` restart question is closed: the branch is only reachable at `n = 2` (2026-09-02)

Prompted by reading two Julia LOBPCGs (`JuliaMolSim/LOBPCGEigensolver.jl`, i.e. DFTK's, and
`venkovic/julia-lobpcg`). DFTK's `ortho!` handles a collapsed direction by **randomizing** the
small-norm column, where `_project_out` zeroes it and `body()` reports convergence. That is the exact
fork `markdown/locg.md` I7 left open ("the alternative is to restart with a fresh random `p`... has not been
settled"). Settled now, and not by measuring which is better: **the alternative is unreachable.**

**Reachability first, because the suite already recorded that no fixture reaches this branch**
(`TestGroundLocg`'s class docstring, and `TestZeroSearchDirection`'s "they do not cover `body()`'s own
`p_is_zero` branch, and cannot"). Measured by patching `_project_out` in a fresh subprocess to report
its norm through `jax.debug.callback` — the norm *before* the 0.99 cut, so the threshold cannot hide
anything. Real and complex, 60 seeds each, `maxiter=80`:

| n | zeroed / `body()` calls | min raw \|p\| |
| --- | --- | --- |
| **2** | **120 / 120** | 3.5e-34 |
| 3 | 0 / 124 | 1 |
| 4 | 0 / 1491 | 1 |
| 5–200 | 0 / 24 000+ | 1 |

**`n = 2` always, nothing else ever.** And it is structural, not a threshold artifact: at `n = 2` the raw
norms are ~1e-31 (max 2.2e-31 over 30 seeds), not marginal values near 0.99, because `span{x, y}` *is*
the whole space — `r` has nowhere orthogonal to go. At `n ≥ 3` the norm is exactly 1 every time. So the
`0.99` postcondition is doing no work in selecting these cases; a threshold anywhere in `(1e-30, 1)`
gives the same partition.

**Which makes a restart strictly worse, not merely unnecessary.** At `n = 2` the early exit is already
exact: over 200 seeds, **0 wrong** at rel > 1e-13, with eigenvalue error 0.0–2.1e-15 and `‖r‖` at the
rounding floor (3.1e-17–2.0e-15) at the moment the branch fires, `niter = 1`. A fresh random `p` at
`n = 2` is necessarily a linear combination of `x` and `y`, so it would re-enter Rayleigh–Ritz with a
rank-deficient basis and buy iterations to rediscover a converged answer. **The branch is not a
premature exit; it is a proof that the space is exhausted.**

**Why DFTK needs the opposite behaviour, which is the whole of the difference.** It converges `m`
eigenpairs, so a dead column must be replaced or the block loses rank and one eigenpair stops
converging. At `m = 1` a dead direction means the single eigenpair is *done*. Same numerical event,
opposite correct response — the block-size-1 specialization is what flips it, not a disagreement about
numerics.

**`markdown/locg.md` I7's open question can be marked closed**, with the caveat that this closes it for
`ground_locg`'s block-size-1 form only. It says nothing about a future block variant, where DFTK's
randomization would become the right answer.

**Nothing else from either repo transfers, for the reason already recorded above for the block C
implementation**: SVQB with eigenvalue-floor regularization (Stathopoulos–Wu, up to 36 inner passes),
level-shifted `safe_cholesky` with SVD fallback, skip-ortho's `norm(VtV·hX)` trigger (Duersch 2018),
and locking all act on an `m × m` Gram matrix that does not exist at `m = 1`. Two independent
corroborations are worth keeping, though:

- **DFTK reuses matvecs "only with orthogonal transformations"** (`new_AX = AY·cX`, `cX` from `syevd`)
  and **never reuses the residual** — `new_R = new_AX − new_BX·λ'` is always formed fresh. That is
  precisely the line the `Ax`-reuse entry above draws between a legitimate `(AV)u` thick restart and the
  rejected in-loop reuse: safe requires `S` orthogonal **and** `V`/`AV` current. DFTK protects exactly
  the quantity the honesty metric measures. Independent support for a decision made here by measurement.
- **`safe_cholesky`'s a-posteriori error estimate** (`eps(κ(R)²)`, read off the factor it already has,
  rather than computing `‖X'X − I‖`) is the standard escape from the "adaptive costs an extra O(N)
  reduction" objection that fixes the re-orthogonalization passes at 2. It does not apply here: at
  `m = 1` there is no factor to read a condition number from, and `_project_out` already returns its
  norm so callers avoid a second reduction. **The 2-pass rule now has a second reason** — not just that
  a measured loop would exit at its first check, but that the cheap-adaptivity trick that would make
  such a loop affordable has no substrate at block size 1.

One deviation shows even a careful implementation carves out unproven exceptions: DFTK reuses the `B`
matrix without the orthogonality justification, commenting that it "seems to be OK even with very badly
conditioned B matrices". Not applicable at `B = I`.

## Multi-node and multi-process runs

### Multi-node, one GPU per node: five harness failures before the library's own surfaced (2026-09-04)

Five harness failures, each hiding the next, came before the library's own scalar-read defect surfaced:
no `jax.distributed.initialize`, `make_mesh` rejecting multi-slice, a module-scope `jnp`, closing over a
sharded array, and three wrong gathers. The working gather is a non-collective, index-sorted concat of
`addressable_shards`; `process_allgather` stalled at 2/4 tasks. `poc/sqd-multinode.md` §1.

### The multi-node correctness result, and the wall clock that came with it (2026-09-04)

Correctness is settled on 4 GPUs across 4 nodes: all six `cache_level` cells, worst `|sharded - single|`
= 4.441e-16, spec `P('x',)` asserted. Speed is not: 9010 ms against 279 ms single-device at N = 2^20 (32x
slower) at identical iteration counts, so it is pure communication — one size, one interconnect.
`poc/sqd-multinode.md` §2–3.

### `poc/sqd_multinode` anchored: memory shards 3.2x, wall clock is 4.06x underwater, first hop is the dear one (2026-09-07)

n=26, N=400000, 1/2/4 nodes: `temp MB` 87.02 → 48.28 → 27.16 (3.20x), a flat ~+5 MB excess from the
replicated `states`. Wall clock is 4.06x slower; 1→2 costs 2.37x and 2→4 only 1.71x, so collective count
dominates. `|dE|` = 7.1e-15. `poc/sqd-multinode.md` §4, §6.

### `poc/sqd_multinode` on real nodes: the scaling is negative, and the memory column was never measured (2026-09-05)

First run, 2 and 4 devices: 961.8 → 1728.6 ms. Its 1.80x-per-doubling reading is superseded by the
anchored sweep, and its `|dE|` = 0 was vacuous (no `--reference-energy`). The memory column's `0.0` was
an instrument failure, since fixed. `poc/sqd-multinode.md` §4–5.

### `sqd` could not return its own eigenvalue multi-process, and I audited past it once (2026-09-04)

`sqd.py`'s `eigval = float(result[0])` raises **"Fetching value for `jax.Array` that spans
non-addressable (non process local) devices"** on a multi-process mesh, and `bool(result[-1])` on the
next line does too. This is the **default** `return_eigvec=False` path: the library could not hand back
a scalar it had correctly computed. `main` and `metal` carry the identical line (`product` predates the
module entirely and has no `sqd.py`), and `main` is a strict ancestor of `dev`
(`git merge-base --is-ancestor`), so no branch ever fixed this.

**I concluded the opposite first, from two true observations.** The `return_eigvec=True` branch *does*
reshard `eigvec` and `states_u` to `PartitionSpec(None)`, and every `np.asarray` in the library *is* on
caller-supplied numpy or host-side circuit construction. Both hold. Neither covers `eigval`, which is
`ground_locg`'s scalar and never resharded -- a reduction over a partitioned vector yields a rank-0
array whose sharding still names the whole mesh. **Auditing the paths I already suspected verified those
paths, not the claim.** The claim needed the one thing virtual devices cannot produce: a genuinely
non-addressable array. Four multi-node runs had already passed `poc/sharding` without touching this, because
`poc/sharding` reads energies via `float(sqd(...))` at `return_eigvec=False`... on fixtures where it worked
single-process. The gap was found by running, not by reading.

**`jax.reshard` is not the fix.** The spec is already `P()`; the problem is *addressability*, not
layout. A replicated scalar holds the identical value on every device (`is_fully_replicated` True, all
four shards equal when checked), so `_host_scalar` reads its own **addressable shard** -- exact, and
deliberately **non-collective**, because a collective there would deadlock any rank that took a
different branch.

**Why this survived: the multi-node testing that was done covered `svsim`, not `sqd`.** `f15a02e`
(2026-07-17) added `examples/svsim.py`'s distributed path, and it is a *correct* implementation of this
exact hazard -- `jax.distributed.initialize(cluster_detection_method='mpi4py')`, then writing
**`final_state.addressable_shards` per rank** with `h5py`, serialized by `MPI.COMM_WORLD` token-passing
and gated on `jax.process_index()`, never fetching a global array. The asymmetry is the lesson:
`svsim` returns a large distributed array, so its author had to confront addressability to write it out
at all. `sqd` returns a **scalar**, which looks innocuous and silently is not. **The dangerous return
type is the small one**, because nothing about it prompts the question.

`_host_scalar` uses the same `addressable_shards` mechanism as that `svsim` code rather than inventing
a second convention. One thing `svsim` can do that `sqd` cannot: gate work on `process_index() == 0`.
Every rank needs the eigenvalue back, so a local read of a replicated value is the only shape available.

### What multi-process *does not* require of the library, and why the POCs still broke (2026-09-04)

Two properties the library already had, both for unrelated reasons, and both load-bearing here:

- **Arrays reach the kernel through `args`, not a closure.** `run_sqd` binds only `cache_level` (a
  static int tuple) via `functools.partial` and passes `args = (scanned, states_u, ...)`. Closing over
  a globally-sharded array is illegal across processes; the library avoids it because a traced
  `cache_level` would retrace the kernel every matvec. A performance requirement and a correctness
  requirement wanting the same shape is luck, not design -- worth knowing before anyone "simplifies" it.
- **`eigvec` and `states_u` are resharded to replicated before host conversion**, so `sqd`'s
  `np.array`/`np.asarray` operate on addressable data. `poc/sharding` (7c) confirms it on real nodes.

**The POCs broke by reaching *around* the library**, which is why "the examples all broke" was still the
wrong reason to change it -- `poc/prefilter_gpu` built its own matvec over `apply_h` instead of going through
`run_sqd`, and `poc/uniquify_sharded` is prototype code for something not in the library at all. The library's *own*
defect was the scalar read above, a different failure entirely.

Still unverified on real nodes: `hproj` (inherently host-side, returns a scipy `csr_array`) and
`svsim`'s `mesh.size | 2^num_qubits` constraint, exercised only on virtual devices.

### A local gather is not a gather: `addressable_shards` means something different per topology (2026-09-04)

`poc/uniquify_sharded`'s host read went through three wrong forms before a right one, and the third was the worst
because it **did not raise**. Recorded in full because the shape recurs anywhere a sharded array reaches
the host:

1. `np.asarray` on a globally-sharded array -- raises "spans non-addressable devices".
2. `process_allgather` with its default `tiled=False` -- **restacks a fully addressable array** into a
   new leading axis (`(4,2)` -> `(1,4,2)`), so `np.array_equal` against the reference returned False and
   exactness flipped True -> False at `d=2`. Single-process only; the docstring separates the
   addressable and non-addressable cases and only the latter ignores `tiled`.
3. `process_allgather` on a **sub-mesh** -- the sweep meshes over `jax.devices()[:num_shards]`, so at
   `d=2` on a 4-process job ranks 2 and 3 hold no shard and its internal `addressable_data(0)` raises
   `FullyReplicatedShard: Array has no addressable shards`. Ranks 0-1 printed `True`, ranks 2-3 died,
   shutdown barrier timed out at 2/4.
4. Concatenating `addressable_shards` -- correct single-process, **silently partial across processes**.
   `addressable_shards` holds *every* shard when one process owns the mesh and only *this rank's* shard
   when it does not, so the concatenation returned a fraction with no error: unique counts came back
   **50782 and 106269 against a true 157051, differing per rank**. A crash traded for a wrong answer.

The working form branches on `len(shards) == arr.sharding.num_devices`: concatenate locally when every
shard is addressable (sorting by `sh.index[0].start`, since arrival order is not global order),
otherwise replicate **inside the sub-mesh** with `jax.jit(out_shardings=P())`. Scoping to
`arr.sharding.mesh` is what makes it safe -- only ranks in that mesh reach the line, so it cannot
deadlock the ranks that skipped, which is precisely why the whole-world `process_allgather` could not
serve.

**The transferable rule: `addressable_shards` is topology-dependent, so any code reading it must assert
how many shards it expected.** A length check is the whole difference between exact and silently
partial.

## GPU runs: the prefilter optimum and compile memory

### The GPU prefilter sweep: the peak transfers, its location does not (2026-09-04)

On a GPU the prefilter peaks at about the CPU's gain (1.38x against the CPU median 1.36x) but not at the
CPU's setting: `(16, 4)` gives only 1.08x, and `sqd`'s `(32, 2)` 1.07x. The sweep first died partway, and
the two missing cells were the best two — print a SKIPPED row rather than dropping a cell.
`poc/prefilter-gpu.md` §1.

### The prefilter optimum is a ridge in `extra mv`, and `(32, 8)` was a boundary artifact (2026-09-12)

Both retracted: the `(32, 8)` optimum sat on the grid's corner, and a second Hamiltonian (`n=22`/`J=8`)
gives a near-transposed surface, so no `(degree, cycles)` is best on both. CPU and GPU iteration counts
are identical (138/115/125/106). `(32, 2)` stays at 1.07x/1.25x: never best, never a loss.
`poc/prefilter-gpu.md` §2–§3, §8.

### End-to-end through `sqd`, `(32, 4)` loses by 45%: a solver-side ratio can invert, not just shrink (2026-09-12)

Through `sqd` with setup included, `(32, 4)` measures 0.68–0.71x, winning 0 of 81 paired rounds: a 1.30x
on the solver becomes 0.69x end-to-end. Seed-independent at `rungs=4, cap=4000`. `poc/prefilter-gpu.md`
§4, harness defects §6.

### Sweeping a static argument in one process exhausts the GPU on compiled modules, not tensors (2026-09-04)

Each static `prefilter` value keeps one more executable: the 8th failed with "Failed to load in-memory
CUBIN" on a 71 GB GPU holding under 1 GB of tensors, HLO and temp size identical in all nine (1836504 B).
`jax.clear_caches()` between cells; sweeps only, since `sqd`'s cache stays at 1 entry.
`poc/prefilter-gpu.md` §5.

### `poc/gpu_unverified`'s re-run: the `lax.sort` leak still does not reproduce, and two claims stay open (2026-09-04)

The leak is flat on a second CUDA device (+0.000 GB retained and drift, 0.950 GB live). Claim 2's
16.0x/14.6x/10.8x are lower bounds under a ~1.46 s floor, so don't quote them; Claim 3 is unrun (one
device). `poc/prefilter-gpu.md` §7, §9.

### sqd sparse kernels on a GPU: `"pairs"` wins until the vectors leave L2 (2026-10-01)

GH200, n=60, pre-`0d25235` kernels: `"pairs"` led the sparse kernels, 2.6–9.3× `"indices"` through `2^19`,
gone by `2^21` and 0.49× at `2^22`, at −39% memory. Both per-state steps, read as L2, were the per-step
complex-carry split that `0d25235` fixes; "keep `"indices"`" is retracted. `poc/sparse/gpu.md` §3, `poc/sparse/split.md` §4

### sqd dense kernels on a GPU: `"indices"` is three quarters diagonal recompute, `"tables"` 2.9–5.6× its matvec (2026-10-02)

GH200 profile, `type1` `2^20`/`2^22`: no carry split in the dense kernels (they never scatter). `"indices"`
spends ~75% recomputing diagonals (179 term-passes, 241 host syncs per matvec); `"tables"` drops it, its
`(2, N)` matvec matching the fixed `"pairs"` at `2^22` — the mesh candidate, unmeasured in solves. `poc/sparse/gpu.md` §6

### sqd on a GPU after the carry fix: `"tables"` fastest end to end, `"pairs"` the memory option (2026-10-02)

GH200, `type1`/`type2` `2^20`–`2^22`, one process: `"tables"` 3.05–4.54× `"indices"` per whole `sqd` call at
~2.2× its memory, fastest in 5 of 6; `"pairs"` 2.44–3.94× at ~0.55×. `"ell"` loses to host build (44% at
`2^22`) and compile (0.84× `"indices"` cold). Superseded for `"pairs"` by the tuned run below. `poc/sparse/gpu.md` §7

### sqd on a GPU with the tuned sparse kernels: `"pairs"` fastest end to end (2026-10-02)

GH200, after the single search, the `2^19` chunk and the dropped hint: `"pairs"` 3.37–5.84× `"indices"`
per whole call, fastest in 5 of 6 (`"tables"` +7% at `type1` `2^22`, at 3.9× the memory), 1.35–1.68× its
previous call, 0.53–0.64× `"indices"`' memory. Its host build (42%; half search, half sort) is the next lever. `poc/sparse/gpu.md` §8, §9

### sqd sparse kernels on a GPU: `"pairs"` wants bigger chunks, and a sorted-scatter hint slows it (2026-10-02)

GH200, `type1` `2^20`/`2^22`: `_CHUNK` `2^15` → `2^19` is 2.71×/1.37× per iteration at +8 MiB temp, the
plateau's edge, so it ships as `_GPU_PAIRS_CHUNK` (GPU, `"pairs"`; `2^15` is the CPU's own optimum). Merging ties;
`indices_are_sorted` on `out[i]` is 0.37–0.64×, and dropping it from `"csr"` is 2.12–3.24× (1.00× on an M1), so
no scatter carries it; `float64` factors 0.37–0.94×. `poc/sparse/tune.md` §2, §3

### sqd dense kernels: a fixed-trip diagonal loop is 3.3–4.4× `"indices"` on a GPU, 1.8–1.9× on CPU (2026-10-02)

Replacing `get_diagonal`'s `while_loop` (a host sync per term on a GPU) with a fixed `kmax`-term sum, the
identity group's diagonal cached once per solve: GH200 3.26–4.44× per iteration, M1 1.75–1.94×, temp
68 → 12 MiB; the identity cache alone 1.23–1.27×. Shipped bucketed by term count (no padding): 1.66–1.96×
per CPU iteration, `type1`/`type2`. `unroll` adds speed at ~1–3 GB temp. `poc/dense-tune.md` §2–§4

### sqd sparse builds on a GPU host: half search, half sort; the device search loses (2026-10-02)

GH200, `2^20`/`2^22`: a `"pairs"` build is search 35–49% and the cross-group sort 48–63%, not "nearly all
search"; `"ell"`'s adds 2,049 factor calls (31%). `get_xsource` on the device is 0.57–0.91× the host search.
`"ell"` wants chunk `2^17` and a ×2 grid there (1.24–1.74×); `(N, 2)` vectors lose on GPU too. `poc/sparse/gpu.md` §8, `poc/sparse/tune.md` §3

### sqd `"pairs"` on a GPU: sort on the device, and scatter without atomics past `2^21` (2026-10-02)

GH200: a stable `jnp.argsort` for the cross-group sort is 1.12–1.50× per build-plus-solve, the operator
bit-identical; dropping the sort loses at `2^22`. One unscanned, unpadded group per scatter with
`unique_indices` is 2.26× `base@2^19` at `2^22` but 0.67× at `2^20`. The device sort ships (GPU only); the
atomic-free scatter does not. `poc/sparse/pairs-sort.md` §2, `poc/sparse/tune.md` §3

### sqd on a GPU with the shipped diagonals and sort: `"indices"` matches `"tables"` at half the memory (2026-10-02)

GH200, `type1`/`type2` `2^20`–`2^22`: `"indices"` 2.96–4.32× its previous iteration, 0.97–1.01× `"tables"`'
speed at 0.42–0.48× its memory, so a mesh wants `"indices"`; `"pairs"` fastest in all six, 1.08–1.82×
`"indices"`, its device sort +11–18% peak on `type2`. `unique-exact` has no switch point; not shipped. `poc/sparse/gpu.md` §8, §9

### sqd `"pairs"`: exact-zero entries dropped, 1.18–1.59× per solve on CPU (2026-10-02)

XX+YY hops cancel on aligned spins, so 86% of `type1`'s searched pairs and 32% of `type2`'s had a zero
factor. Dropping them after the factor pass (`_drop_zeros`) cuts the operator 23–64% at identical iteration
counts, eigenvalues within 7.1e-15 (not bit-identical: chunk boundaries move). M1 only. `poc/sparse/prune.md` §3

### sqd sparse kernels: `"csr"` and `"ell"` leave the library, `"pairs"` stays (2026-10-02)

GH200 tuned (`2^17`, ×2 grid), `"ell"` against `"pairs"` per iteration is 0.28×/1.50× (`type1` `2^20`/`2^22`)
and 1.25×/0.70× (`type2`), always at more memory and a 4.8 s build; on CPU a near-tie. `"csr"` is dominated
on both backends. Both are removed from `Matvec`; their builders and kernels live on in `poc/`. `poc/sparse/tune.md` §3

### sqd sparse kernels: a state-major `(N, 2)` gather layout loses on CPU (2026-10-01)

M1, n=60 `type1`, `2^17`/`2^19`: gathering from `(N, 2)` instead of `(2, N)` is 0.83–0.87× on the matvec
and 0.95–0.98× per solve iteration for all three sparse kernels, flat in N, bit-identical matvecs. A first
run's solve figures were void: a second `jax.jit` of one function reused the first's trace. `poc/sparse/layout.md` §2, §4

### sqd sparse kernels: a tiled `"pairs"` order is 1.08× per iteration on CPU (2026-10-01)

M1, n=60 `type1`, `2^17`/`2^19`: sorting pairs by `(i >> 12, j >> 12, i)` is 1.08× per solve iteration
(10/10 rounds), almost all from the 1-D matvec (1.20–1.27×; `(2, N)` 1.03–1.05×), at unchanged memory.
GH200: 1.00× before the carry fix (`0d25235`), 1.01–1.03× after (10/10); skipping the cross-group sort
loses there (0.95–0.98×), so it stays. Not shipped: the gain is below a `lexsort`'s build cost. `poc/sparse/tiles.md` §2, §3

### sqd sparse kernels on a GPU: the cliff is XLA splitting the complex scan carry every step (2026-10-01)

GH200 profile: 72–88% of a `"pairs"` matvec from `2^20` is `wrapped_real`/`imag`/`complex`, a full pass
over the complex carry per scan step, so cost is `O(N × steps)`. `_scan_add` now carries real and
imaginary parts on CUDA only: GH200 2.5–17.9× per iteration, no temp cost, every sparse kernel past
`"indices"` (cross-run); ungated, CPU ran 0.68–0.90× with +1 `out`, hence the gate. `poc/sparse/split.md` §1, §3, §4

## Warm starts, collectives and the eigenpair check (2026-09)

### Warm-starting the growing subspace: four hypotheses eliminated, and the fixture gate is the result (2026-09-17)

No verdict: warm beats cold 1.3–1.9× on iterations, but zero-padding, the shape
`markdown/skqd-sqd-solve-tolerance.md` §8 rejected, beat both in 19/19 rounds, so the `valid` gate
withheld it. Cause: the ground state is concentrated (top-256 = 98.8%); a real retest needs relgap ≲
1e-04, and this family saturates at 3.95e-02. `poc/warmstart.md` §1–§5.

### Warm starts on a recovery-grown sequence: ~1.15×, and only with a lighter prefilter (2026-09-26)

On a fixture where the answer moves every round (n=60 `type1`, recovery-grown `2^12` to `2^17`, 5–30%
weight on new states), zero-padding finally loses at the shipped `(32, 2)` (4 of 5 rounds), and there the
best warm start beats cold by only 1.02–1.10×. Warm start plus `(16, 1)` gives 1.13–1.26× in operator
applications, shrinking as recovery converges (~1.15× per run); a first-order start is the worst warm arm.
Every arm converged to the cold energy. `poc/warmstart-rounds.md` §1–§2.

### `body()`'s all-reduces are one chain; 13 -> 12 is the floor (2026-09-25)

`test/sharded/allreduce_count.py`, 4 virtual CPU devices. The loop body compiled to 13 all-reduces
(arities `[1x7, 2x3, 3x2, 5]`), and a reachability walk over the HLO found **every pair dependent** --
the combiner had already merged all independent reductions, so the "seven isolated norms" of
`markdown/sqd-locg-improvement-ideas.md` were sequential links, not missed merges. Inlining
`jnp.linalg.norm`'s formula left 13 (its `@jit` is inlined anyway), and so did reordering the source.
The one missed merge was `norm_s`/`norm_u`, independent but split across two links; a single stacked
reduction gives 12. Bit-identical over 18 arms (N 64/1000/5000, f32/f64/c128, batched or not), temp
flat within +192 B to N=1M. **Below 12 needs fewer sequential reductions**, i.e. dropping a
`_project_out` or re-orthogonalization pass, which are closed. Speed on real nodes is unmeasured.

### `process_allgather` gathers per leaf: a pytree does not merge `sqd`'s two host reads (2026-09-25)

`markdown/sqd-locg-improvement-ideas.md` §11 recorded, as verified, that passing `(eigval, converged)` to
`process_allgather` as one pytree would make one collective instead of two. Read from JAX 0.11.2's
source, it is `jax.tree.map(_pjit, in_tree)`: each leaf goes through `_handle_array_process_allgather`
separately, with its own `jit`. **One call, still two collectives** -- "accepts a pytree" was true and
answered a different question.

The per-leaf handler also has two branches, and the one `sqd` normally takes may move no data. A rank-0
result on the full mesh is `P()` but not fully addressable, so it goes through
`jit(identity, out_shardings=P())` onto the layout it already has. Only a **fully addressable** input --
a 1-device solve inside a multi-process world, `poc/sqd_multinode`'s first row -- builds a process-spanning array and
really gathers. Merging would need a single stacked `(2,)` array (`converged` cast to float is exact), and
it saves one dispatch per solve against a loop of hundreds of iterations at 12 all-reduces each. Not
built: the payoff is below noise, and the change sits on the host-read path that has already shipped
broken four times, which a laptop cannot exercise -- a two-rank localhost cluster
(`jax.distributed.initialize` plus gloo CPU collectives) was tried and the sandbox denies the
coordinator's `bind`. Such a cluster would be the first local multi-process test for `_host_scalar`
if run outside the sandbox.

### `EigenpairCheckError`: an independent residual after every `sqd` solve (2026-09-25)

`sqd` recomputes `‖Hv − Ev‖` and `‖Hv‖` after the solve (at `cache_level=(0, 0)` as first built; see the
revision below) and raises `EigenpairCheckError` (a `RuntimeError` subclass) above `10 × max(bound, residual_floor)`,
where `bound` is the solver's own `max(atol, rtol·(‖Hv‖+|E|))` with the exact `‖Hv‖`. Motivated by
spinchain's `sqd_backend._eigen_residual`, which repacked, re-uniquified and re-built the operator to
check the same thing; all of that was rqutils code except the `(0, 0)` kernel, so doing it here keeps
the independence and drops the rebuild.

- **Slack measured, not chosen**: over the full suite's 185 converged `sqd` solves the recomputed
  residual sits at **max 0.96, median 0.27** of `bound`. The tail is at the floor (`r = 2.9e-15`,
  `bound = 3.0e-15`, floor `4.2e-15`), where the recomputation's own rounding is the discrepancy --
  hence the `max(…, floor)` inside the slack rather than a bare multiple of `bound`.
- **The slack holds at N = 10⁷, and N does not enter it** (2026-09-25, for spinchain's
  `rqutils-eigenpair-check-request.md`; `poc/eigenpair_check_scale.py`). On spinchain's eight open-XXZ
  Hamiltonians at n = 30 (complex, `Σ|c_k|` 19–45) with Hamming-shell subspaces around both Néel states,
  default tolerance and prefilter, residual ÷ threshold is **0.040–0.097** over N = 10⁵–3·10⁶ and
  **0.091** at N = 10⁷ (`type1`, δ = 0.5), identical at `(1, 0)` and `(1, 2)`, with no trend in N.
  Structurally: each `P_k` is a signed permutation and projection only drops entries, so the error in
  `Hv` has norm at most `γ_m·Σ|c_k|` -- term count, not N -- which the floor term tracks.
- **Catches what it is for**: a dominant-component sign flip with `converged=True` reads **5.7e+00**
  against a threshold of 7.8e-14. A swap of components 0/1 was a no-op on that fixture (both ~1e-17).
- **Cost**, `J=120`, `N=30k`, warm, interleaved 9 rounds: `(1,0)` +3.2%, `(1,2)` +7.9%, `(0,0)` +1.5%
  (6/9, noise); XLA temp +0.25–0.60 MiB, i.e. about one vector. `(1,2)` pays most because its solve
  matvec is cheap and the check's `(0,0)` one is not.
- **Must not say "did not converge"**: spinchain's `sqd_with_retry` retries on that substring, and a
  wrong pair retried at `maxiter=100_000` is the outcome it forbids. Pinned by the test.
- Multi-process: both reads go through `_host_scalar`, replicated, so every rank raises together.
- **Revised the same day: the check reuses a full `xsources` cache.** Timing the candidate kernels on the
  fixture above (warm, 15 interleaved rounds) showed the `(0,0)` check matvec at **80.5 ms** against
  **7.7 ms** for `(1,0)` on the cached `xsources` and 2.2 ms for the `(1,2)` solve kernel: ~90% of the
  check was redoing the `J`-fold `get_xsource` search. So when every group's source index is cached the
  check runs `(1,0)` on that array; otherwise (`cache_level[0]=0`, or a partial `xcache_groups`) it keeps
  `(0,0)`. It always rebuilds the diagonal from signatures, so a cached diagonal never vouches for
  itself. Whole-solve cost after: `(1,0)` **+0.9%** (was 3.2%), `(1,1)` +0.3%, `(1,2)` **+0.6%** (was
  7.9%), temp +0.25–0.35 MiB. **What this gives up is small**: the `(0,0)` arm never had
  *function* independence -- it calls the same `get_xsource` and `get_diagonal` the solve does -- so the
  only thing no longer re-derived is the source-index array the precompute scan built. At `(1,0)` the
  check kernel equals the solve's, as `(0,0)`'s already did for a `(0,0)` solve. The check now runs on
  the solve's `states_u` layout, so a check-only sharded call no longer reshards `states_u` back.
  Pinned by `TestEigenpairCheck.test_the_check_reuses_cached_xsources_but_never_a_cached_diagonal`,
  which counts named `pjit` call sites; the always-`(0,0)` and solve's-own-kernel mutants each fail it.

## `precond` was removed; `sqd` defaults to `prefilter=(32, 2)`

2026-08-28, acting on the comparison below.

**`sqd(prefilter=...)` now defaults to `(32, 2)`** — 1.49× median end-to-end wall clock (min 1.15×,
max 1.70×, 6 sampled XXZ subspaces at n=14–18), every arm correct to <1e-9. `sqd` supplies the required
`Σ|c_k|` bound itself, so the option costs the caller nothing and there is no bound to get wrong. Pass
`None` to restore the old graph exactly. `ground_locg`'s own default stays `None` — it cannot derive a
bound from an opaque callable.

An unplanned benefit: the near-degenerate subspace that motivated the "raise `maxiter` first" error
message **converges within the default cap** with the filter on (4.4e-16 at `maxiter=1000`, against a
`RuntimeError` without it). `test_near_degenerate_subspace_needs_maxiter_above_the_default` now pins
`prefilter=None` explicitly, or it would assert nothing.

**`ground_locg(precond=...)` is deleted**, with its 6 tests and
`examples/scaling/poc10_deflation_precond.py`. Not because it did not work — on a *positive-definite*
operator it measured 2.76× median with 0 regressions, better than the 1.79× on record. It is deleted
because **no `sqd` caller can use it**: `sqd` solves the raw indefinite projected `H` (~50% of diagonal
entries ≤ 0), Jacobi needs positive-definiteness, and the shift required to get one is the closed
investigation below. Measured on the raw operator, literal Jacobi does not merely regress — it **fails
to converge at all** (8000-iteration cap, wrong answer, 3/3 sizes); `|diag|⁻¹` regresses to 0.20–0.37×.
So a "fall back to `precond` when no bound is available" convenience would have turned a clean
`ValueError` into a silent wrong answer.

What the deletion left behind: `body_iter1` still splits the raw residual from the search direction,
and that split is still load-bearing — `r_is_zero` feeds both the `sas[1, 1]` mask and `converged`, so a
reintroduced preconditioner must not touch it. The comment there says so. `markdown/deflation-preconditioner.md`
keeps the deflation verdict (0.68–0.98×, 8/8 losses) with all its tables; only the script is gone, and
it is recoverable from `26a9b7b`.

**If preconditioning is ever reconsidered:** the one route that works from `sqd` is shift-by-the-free-
`O(N)`-bound then Jacobi, measured 1.45× median with energies exact to 4.4e-15. It is *dominated* by the
prefilter's 1.49× and requires `sqd` to transform the operator, so it was not pursued.

## Prefilter vs precond: use the prefilter, and do not combine them

2026-08-28, prompted by "is there an opportunity for precond to improve `locg`?". Answered by
measurement rather than by re-reading the closed record below; the conclusion **agrees** with it and
adds two things it does not contain. One Hamiltonian family (XXZ, `Bx = 0.5`), sampled subspaces,
single-device CPU, best-of-3 warm.

**The recommendation is the prefilter, and the deciding factor is the precondition, not the margin.**
`precond` needs positive-definiteness, hence a shift `sqd` cannot produce — that is the structural
blocker recorded below. The prefilter needs only an upper bound on `λ_max`, which `sqd` has free as
`Σ|c_k|`. Wall-clock on the *shifted* operator, where `precond` is at its best, `ground_locg` dense:

| n | dim | plain | precond | prefilter | both |
| --- | --- | --- | --- | --- | --- |
| 16 | 1975 | 262.7 ms | 187.1 | **105.3** | 179.3 |
| 16 | 1970 | 155.8 | 105.9 | **61.5** | 78.3 |
| 18 | 3971 | 566.9 | 397.3 | **239.0** | 290.5 |
| 18 | 3965 | 1206.6 | 828.8 | **261.2** | 352.6 |

Prefilter wins 4/4 even where `precond` is legal.

**They anti-compose, robustly, and this is the finding worth keeping.** Adding `precond` to a
prefiltered run *halves* the gain from `(32, 2)` up. On the shifted n=18 dim-3965 instance, gains over
plain:

| prefilter | filter only | filter + precond |
| --- | --- | --- |
| (16, 2) | 2.19x | 2.77x — precond helps |
| (32, 2) | **11.27x** | 6.04x |
| (48, 2) | **18.78x** | 9.94x |
| (64, 2) | **28.17x** | 13.00x |

Only at low degree does `precond` add anything. No verified mechanism — the plausible one, that a
diagonal rescale partly undoes the residual enrichment `_chebyshev_prefilter`'s docstring describes, is
**speculation and was not tested**. Don't record it as established.

**Quote the end-to-end number, not the iteration count.** Three measures of the same prefilter benefit,
each shrinking as it gets closer to what a caller experiences:

| measure | median |
| --- | --- |
| iteration counts, dense `ground_locg` | 5.02x |
| wall-clock, dense `ground_locg` | 2.43x |
| **wall-clock through `sqd()`** | **1.49x** (min 1.15x, max 1.70x, 6 instances) |

The prefilter costs `cycles·(degree+1)` ≈ 66 matvecs up front, and `apply_h`'s sparse matvec is cheap,
so those cost proportionally more than on a dense operator. This is exactly the matvec-to-bookkeeping
ratio `markdown/locg-chebyshev-prefilter.md` names as the genuinely uncertain quantity, and the end-to-end
figure lands **below** the 1.88x that doc measured on dense `ground_locg` — which is why `sqd`'s
docstring tells callers to A/B on their own subspaces rather than trusting the published figure. Every
arm was correct to <1e-9 against `eigsh(tol=0)`.

**A measurement trap hit twice here.** `test/conftest.py`'s `project_dense` builds the *full* `2^n`
operator before slicing, so it cannot reach n=16 (68 GB) — an attempt to reproduce the 1.79x figure with
it died with no useful error. Use `hproj` for anything past n≈12. Separately, a first pass on small
dense chains measured 1.05x median *with a regression* and looked like a refutation of the shipped
`precond`; the documented instances are **sampled subspaces at n=16-18, dim 2000-4000**, where the same
code reproduces 2.76x median with 0 regressions. The fixture family, not the code, was wrong.

## Preconditioners and subspace selection: a closed investigation

Full record in `markdown/spinchain/rqutils-precond-request.md` and `markdown/sdp-lower-bound.md`. Summarized here
because the conclusion is easy to re-litigate.

**What shipped:** `ground_locg(precond=None | callable)`, an approximate inverse `M⁻¹` applied to the
residual where the search direction is formed. Static argument, so `None` is the identity path and
leaves the traced graph unchanged. Measured **1.79× median** fewer iterations on a 12-instance XXZ
batch (1.29–2.04×, 0 regressions) — **on a shifted operator** `A = H − (λ_min − 0.5)I`, using the
*projected* `λ_min`. A caller that knows its own spectral range can build that; `sqd` cannot.

**What is closed, and why.** `sqd` solves the raw projected `H`, which is **indefinite** (~50% of
diagonal entries ≤ 0), and Jacobi needs positive-definiteness. Six routes to a usable shift were
measured and rejected: a convenience flag on `sqd` (~3× slower, 8/8 worse); `M⁻¹ = |diag|⁻¹` (0.30×
median, 12/12 regressions); a structural row-sum bound (16.6–25.5× over-shift); and three routes
through the SCIP product-state solver, which optimizes over a manifold on the **wrong side** of
`λ_min` so every bound from it is an upper bound.

**The decisive argument is structural, not a measurement.** A level-1 SDP bound *is* valid and by far
the tightest available (0.64–1.06× over-shift against 4.14–8.14× for coefficient-sum), and yields
1.29× with 0/12 regressions — but the same 1.29× comes free from `σ = min(diag) − 2·max|diag|` at
`O(N)`. And no bound on `H` can do better in principle: `ground_locg` sees `hproj(H, subspace)`, whose
minimum sits 0.64–1.06× of the projected spectral width *above* `λ_min(H)`, and that gap is a property
of the random projection. **The remaining upside is in estimating the projected operator's minimum,
not in tightening a bound on `H`** — the untried candidate being the two-level/deflation
preconditioner.

A useful by-product: the SDP bound's `σ` is **exactly linear in `n`** (max residual 5.2e-08 over
n=4..14) with slope equal to the single-bond `λ_min`, so the conic solve recovers a linear function
two solves determine. Its value is establishing the constant for a new coupling family, not
per-instance evaluation.

**Subspace selection (weight shells + diagonal ranking) was measured, then rejected** by the user on
2026-08-25 — sound results, but every arm compares against *uniform random* subspaces where a real SQD
workflow's quantum-sampled subspaces are already ground-state-biased, so the practical payoff is
unestablished. It is also a change to how callers choose `states`, not an `sqd` change. Do not build
on it.

**Retractions on record in those docs**, kept because each was nearly believed: a confounded
Bloch-sampling result (9/9 cells, 13.6–50.3% "capture" — two arms differing in sampling *mechanism*
with identical distributions); a `p`-sweep optimum that was a `np.unique`+truncation artifact; and
"top-amplitude selection is a ceiling", which it is not — it maximizes *fidelity* where `λ_min` of a
projection is variational and rewards connectivity.

## Moved from code comments (2026-09-25)

Evidence cut from `rqutils/` comments to meet the two-line ceiling; each code site points to its subsection.

### svsim._GATE_XZ: one table for the gate set, and no `cz`

A second parameterized-gate tuple beside the dispatch could drift and leave `angle` silently stale from the
previous iteration. `cz` is absent deliberately: decomposed on the `QuantumCircuit` path only, rejected as
a raw gate spec (`test/test_svsim.py::TestCz::test_cz_as_a_gate_spec_is_rejected`).

### svsim.do_svsim: build the index iota inside the scan body

Closed over, XLA hoists the `2^n` int64 iota into the scan carry: peak temp **2.5x** the statevector (above the two complex128 buffers),
invariant in n, against **2.0x** built in the body (loop-invariant, so rematerialized free) — 8 GiB at
n=30, 32 GiB at n=32. Output bit-identical.

### svsim.to_circuitxz: the y/ry phase, and what omitting it cost

x, z, rx, rz, rzz and cz have `x·z == 0`, so the `(-i)^{x·z}` phase is y/ry's alone. Without it `ry` was
off by `i`, corrupting nearly every transpiled circuit: |overlap| **0.5** against qiskit on a 5-qubit GHZ,
**1e-16** on a 6-qubit 4-rep Trotter step. See also "`svsim`: `sin` is complex128 and must stay so".

### qprint._process: compress before phase work

Only terms above the cutoff print, so select before normalizing phases (paid on every lazy `__repr__`):
**25 ms -> 5.3 ms** for a 5-term printout of a dim-2^20 input, and the wrap-around loop and both
`normalize_phase` passes become bounded by the term count. `global_phase='mean'` averages every element,
so it is taken before compressing.

### qprint._qobj_data: two qutip defects

- `dims[0]` is the ket space, the trivial `[1]` for a bra: taken unconditionally it raised "Product of
  subsystem dimensions 1 and qobj dimension 3 do not match" for every bra.
- `Qobj.full()`, not `.data.data`: qutip 5 wraps the payload in its own Dense/CSR class, so every Qobj
  raised "'qutip.core.data.dense.Dense' object has no attribute 'data'"; `.full()` is dense in 4 and 5.

### qprint.QPrintBraKet._add_labels: unravel per term

A precomputed `len(dim) x objdim` table cost **168 MB and 16.7 ms** at dim 2^20 over 20 subsystems,
against **0.009 ms** to unravel only the printed indices.

### qprint.QPrintMatrix._make_lines: always emit the amplitude

Suppressing an exact `1` suits a labelled term (`\frac{IZ}{2}`, not `1\frac{IZ}{2}`), but a matrix cell has
no label, so it came out empty: `\begin{pmatrix} & 0 & 0 & 0 \\ ...`.

### paulis.symplectic.pack_states: the 0/1 check

Before `astype`, which erases the evidence (256 wraps to 0, -1 to 255). min/max rather than
`(states == 0) | (states == 1)`: **1.05 ms against 4.14 ms** at N=1M, n=32, equivalent for the integer
and bool dtypes it receives since only 0 and 1 lie in [0, 1].

### paulis.symplectic.pack_states: chunked by rows (2026-09-27)

`astype(np.uint8)` then `np.pad` held two full copies of the input on top of it: **242 MiB** transient
peak at `2^21` × 60 for a 120 MiB input and a 16 MiB output. Packing `2^16`-row chunks through one
zeroed `(chunk, n + 1)` buffer into a preallocated output peaks at **20.3 MiB**, output included, and is
not slower (48 against 55 ms), bit-identical across widths 1–100, four dtypes and forced chunk
boundaries. The binary check also skips `min()` for unsigned and bool input. Rejected alongside:
`np.unpackbits(..., count=1 + n)` in `unpack_states` trims the returned basis's hidden base by 5%
(128 → 122 MiB) but is **1.53× slower** (19.0 against 12.4 ms median, 1/25 paired wins), as `count`
leaves numpy's byte-table fast path.

### sqd: real X groups scanned as float64 (2026-09-27)

Built for `TABLES` only. Real-first groups held as float64 cut `TABLES` memory to 0.63–0.66× at equal or
better speed; `INDICES` is Hamiltonian-dependent at equal iterations (1.18× on `type2`, **0.92×** on
`type1`, cause not found), so it and `ONTHEFLY` stay unsplit. `poc/real-groups.md` §6

### paulis.symplectic.from_paulisum: no quadratic group-bys

- A one-hot matmul summing duplicate strings materialized a dense `(n_unique, n_terms)` mask: **64 MB,
  27.5 ms** at 4000 terms / 2000 groups, 400 MB at 10000/5000, OOM beyond, against **0.03 ms** for
  `np.add.at` (not `np.bincount`: `coeffs` is still complex there).
- Rescanning `indices` per X signature: **15.9 ms** at 5000 groups against **0.3 ms** for one stable
  sort of `indices`, already each term's group id, which also stopped a per-group uint8 conversion; cumulative counts give each group's slice.

### paulis.symplectic.from_paulisum: the Hermiticity tolerance

Checked once so both ingest paths agree (the tuple path used to warn and take `.real` under
`force_real=True`). Against `atol`, not zero: rounding sits at the 1e-16 level, and conjugating a Hermitian
matrix by a non-Clifford circuit was rejected by the exact test **18 times out of 18** (n = 3, 4, 5 x 6
seeds) at |imag| ≤ **3.3e-16**, while those operators' own hermiticity error reached **2.7e-15**. The
default sits ~4 orders above that; `atol=0.0` restores the exact test.

### paulis.symplectic.from_paulisum: why `atol` is keyword-only

Positionally, `from_paulisum(op, 1e-3)` reads as a `simplify` tolerance but raised the Hermiticity
threshold nine orders, and the `.real` that follows dropped the imaginary part:
`from_paulisum((["ZI", "IZ"], [1 + 1e-4j, 0.5]), 1e-3)` returned `c = [0.5, 1.0]`.

### paulis.symplectic: the pad bit is unconditional

As an opt-in flag nothing enforced agreement between the signature and state sides; disagreement put every
element in the wrong column, still symmetric, so eigvalsh gave a plausible wrong energy — how `hproj`
shipped broken. The X side goes through `pack_states`, the consumers' padder; the Z signatures
`(n_xgroups, n_zterms, n_qubits)` pad axis 2, not 1, so cannot reuse it. `svsim` builds `CircuitXZ` itself.

### paulis.general.paulis: cache the returned array itself

Caching a `.copy()` and returning the original kept a second writeable allocation per key: retained
**2.00x** the result at `dim=(2,)*6` (537 MB for a 268 MB basis) against **1.00x**, and warm returns were
writeable where cold ones were read-only.

### paulis.general.pauli_matrices: sparse is derived and frozen

- **Derived**: the old CSR branch re-spelled the shell ordering and `sqrt(2/(k(k+1)))` normalization by
  hand; verified identical (max abs diff 0.0, equal nnz) for dim 2 through 6 before the swap.
- **Frozen**: callers share the memoized instances; `pauli_matrices(3, sparse=True)[1] /= 2` shifted the
  cache by 0.5 max abs, still Hermitian, so every later `components()` was consistently wrong.
- **Read-only, not copied**: a copy per hit is **276 us against 0.10 us** (2698x) at dim=6; `setflags` on
  the three buffers costs 1.15 ms cold and blocks `/=`, `*=`, `data[i] = ...`, `mat[i, j] = ...`.

### paulis.general.components: ungated normalize_dim, required dim

Gated, `components(m, dim=3, npmod=jnp)` raised "object of type 'int' has no len()" (general case:
"`paulis/general`: the `npmod` gating bug"). `dim` is required: a 4x4 matrix inferred `(4,)` where
`(2, 2)` may be meant, both giving 16 valid coefficients, but `2**(len(dim) - 2)` is 0.5 against 1.0, so
the norms differ by sqrt(2) (measured 1.4142135623730951) with nothing to say which the caller got.

### paulis.general.labels: fold the affixes in

Two whole-array `np.char.add` passes over `np.full(out.shape, ...)` cost **47-58%** of the call at 10
qubits (the latex prefix alone a 25 MB array of one 7-char string); a scalar for `np.full` does not help,
since `np.char.add` densifies it.

### ground_locg._check_tols: the accept-anything cutoffs

`‖Hv − Ev‖ ≤ ‖H‖` for every normalized v, so a bound reaching `‖H‖` accepts the first iterate: `atol=100`
against `‖H‖=17` converged in one iteration, and the superseded `* n * 10` scale made `rtol=1e-8` at
`n=2^20` a bound of 4.2 against `‖H‖=20` ("`atol`/`rtol`: the pair is right, and `rtol`'s scale took two
tries to get right (2026-09-01)"). Hence `rtol >= 0.5` (scale ≤ `2‖H‖`), and `atol` against `Σ|c_k|`,
which over-estimates `‖H‖₂` by **1.56-1.90x** on 1D XXZ. Both deliberately loose.

### ground_locg debug diagnostics: `reltol` became `rtol_scale`

Renamed 2026-09-01: the value is the scale `‖Ax‖ + |θ|` (~2|λ_min|) that `rtol` multiplies, where the old
name promised a floor near eps·‖A‖ — off by ~1e16; `diag["reltol"]` now raises KeyError. Not bare `scale`:
a key is read out of context, from a dict of nine.

### ground_locg.body: the batched matvec pair

- **Speed**: 1.61-1.81x on the pair, bit-identical `theta`; all-gathers 3 -> 2 in the compiled loop body on a 4-device mesh (re-measured 2026-09-26; the recorded
  6 -> 3 does not reproduce), and
  `jnp.stack` of `P('x')` is `P(None, 'x')`. Off in `ground_locg`: a `mat` callable need not batch.
- **Memory, opposite signs**: +1 vector temp against an elementwise operator; **−16.00 B/slot** (one f64
  complex slot, 0.942x, N=4000..60000) against `sqd`'s, whose unbatched arm holds two gather results live against one `(2, N)` buffer.
- **Rounding may differ**: at atol=rtol=0, `debug=True`, dim=32 a dense einsum moves `y` by **1.1e-9**
  (elementwise: exactly 0.0); a near-degenerate `sqd` subspace moved `y` **0.56** with `theta` agreeing to
  **2.2e-15**. Judge by `theta`; compensated summation is closed (CLAUDE.md, "Closed investigations").

The three `debug=True` matvecs batch too, and cannot change the trajectory: `diagnostics` is `scan`'s
output, never its carry.

### ground_locg.body: the convergence test's scale

`max`, not `min`: either arm suffices, so a fixed `atol` (a downstream 1e-6 guard) holds at every
dimension, and `rtol` scales by `(‖Ax‖ + |θ|)` alone — the pre-2026-08-31 `* n * 10` factor made
`rtol=1e-8` at `n=2^20` a 4.2 bound against `‖A‖ = 20`. `abs(theta)` so the sum cannot cancel:
`norm(Ax) - theta` went **negative** on a positive-definite operator, making the test unsatisfiable.

### ground_locg: promote xinit up front

One promotion fixes two defects: a lower-precision `xinit` made `while_loop`'s carry disagree on `theta`
(seed step against loop), and a float64 `xinit` with a complex128 `mat` stayed real through
`compute_sas`'s scatter, silently dropping the projected matrix's imaginary part behind a ComplexWarning.
`eval_shape` reads the dtype without a matvec.

### ground_locg: the `rtol=None` default

`atol` defaults to 0.0, since a derived absolute bound is what the pair replaced. 4·eps, not eps: the scale
is ~2‖A‖, so eps targets 2·eps·‖A‖, only 2x the floor eps·‖A‖ whose constant spans 0.49-1.26 over 27
samples; 4·eps is 8x, `residual_floor`'s 3.2x margin over the worst constant. From `work_dtype`, not
`xinit`: a float32 `xinit` on a complex128 problem would loosen it nine orders.

### ground_locg._project_out: sources for the two-pass form

Interleaved normalization is Algorithm 5 ("Modified orthogonalization procedure") of Duersch, Shao, Yang &
Gu, *A Robust and Efficient Implementation of LOBPCG*, arXiv:1704.07458; at block size 1 their SVQB is
just `normalize`. Two passes is Kahan's "twice is enough" (Parlett, *The Symmetric Eigenvalue Problem*
(1980), Sec. 6.9; SLEPc STR-1, "Orthogonalization Routines in SLEPc", Hernandez, Roman, Tomas & Vidal,
2007, reached from "SLEPc Technical Reports" at <https://slepc.upv.es/documentation/> since
`/documentation/reports/str1.pdf` redirects). `_reorthogonalize` likewise: "The fixed 2-pass
re-orthogonalization beats `diaglib`'s adaptive loop on its own metric (2026-09-02)".

### ground_locg.eigenpair_3x3: the discriminant form

`disc` is Cardano's `p^3 - q^2` (`q = -13.5 * c0`, 182.25 == 13.5**2) expanded in `c1`, `c0`: **1.16e-16**
mean relative error against **1.92e-16** factored, over 200k random inputs against exact rationals, ~1.7x
down the whole near-degenerate sweep. `p = -3*c1` rounds once and cubing triples it; 27.0 is exact. Don't
simplify it into the textbook form.

### sqd._host_scalar: one path on every rank

- **Branch on `jax.process_count()`, never `addressable_shards`**: an early return on non-empty shards had
  two of 4 ranks print and two raise "spans non-addressable devices" (a hang, had they entered the
  collective); before that, empty shards read as host-side handed `float()` the unreadable array.
- **Safe only because every rank calling `sqd` reaches it**; `poc/uniquify_sharded.py`'s sub-mesh gather
  is the non-collective form.
- **`process_allgather(tiled=True)`**: a bare `PartitionSpec` raised "jit requires a non-empty mesh in
  context" outside `set_mesh`, and a `NamedSharding` on `value.sharding.mesh` replicated over one device,
  leaving 3 of 4 ranks an `IndexError`. `tiled=False` is rejected for non-addressable input.
- **Shape depends on addressability**: fully addressable input is expanded to `(1,)` and gathered to
  `(process_count,)` — (4,) on 4 nodes, where `float()` raised "only 0-dimensional arrays can be
  converted to Python scalars" — hence `.reshape(-1)[0]`.

Background: "`sqd` could not return its own eigenvalue multi-process, and I audited past it once
(2026-09-04)".

### sqd.sqd: `packed=True` skips the pack

`pack_states` is not idempotent (a second pass reads each byte as one bit), and a caller holding packed
states paid the unpacked `[N, num_qubits]` (~7.7x the width at n=100) with both live at once — an 8x
expansion, **2.40 GB against 0.31 GB** at n=100, N=24M. `run_sqd` always took packed.

### sqd.sqd: pad the input states too

`states_p`'s leading dimension is in `run_sqd`'s jit cache key, so the raw length retraced on every
distinct `len(states)`: **0.44 s per call against 0.064 s** once the shape repeats.

### sqd.hproj: `shape=` is mandatory

Inferred, a trailing uncoupled state is dropped: **41x41 for a 53-state subspace** (local two-site js
operators), still symmetric, so eigvalsh gives a plausible wrong energy. With nothing surviving scipy
raises "cannot infer dimensions from zero sized index arrays" on what is legitimately the zero matrix.

### sqd._spread_seed: reshard the filler mask

`vec` is always sharded, but `run_sqd` reshards `states_u` only at `cache_level[0] == 1` (the uncached
search needs it replicated), so at (0, 0), (0, 1), (0, 2) `jnp.where` raised "select `which` must be
scalar or have the same sharding as cases" on every mesh — masked by `_accumulate_diagonal`'s rank bug,
which failed all six earlier. Fixed here because this function decides `vec`'s sharding.

### sqd.vinit_from_min_diag: `.real` on both branches

`diagonals` has `hamiltonian.c`'s dtype, complex128 whenever a string has an odd Y count, though the
projected diagonal is real; at `cache_level[1] == 2` `jnp.max`/`argmin` raised `TypeError: lt does not
accept dtype complex128`, single-device, while the uncached branch already took `.real`.

### sqd.run_sqd: the initial-vector guards

The minimum-diagonal state is weighted heavily, with the spread seed underneath rather than a one-hot.

- **Sign**: "`vinit_from_min_diag`'s weight must carry the seed's sign, or it cancels it", including why
  the unreachable `sign == 0` branch stays.
- **One-hot**: "`sqd`: why the initial vector is a spread, not a one-hot". A 14-state subspace splitting
  4+10 returned **−1.293**, its seed block's exact minimum, against a true **−2.191**; `vinit_nodiag`'s
  `e_0` with state 0 decoupled is an eigenvector with eigenvalue 0, so `sqd` returned 0.0 with
  `converged=True` on a 9-state IIIX subspace whose answer is −1.
- **Mask, not index**: "Indexing a *sharded* array to read one element emits an `all-gather` of the whole
  vector" (3 all-gathers against 0, bit-identical at several `imin`).
- **`out_sharding` is mandatory**: the original scatter raised `ShardingTypeError` on any multi-device
  mesh (on CPU via `XLA_FLAGS=--xla_force_host_platform_device_count`, as `poc/sharding.py` does), and
  re-measured 2026-09-25 in the mask form, a replicated `broadcasted_iota` makes the `where` raise
  `ShardingTypeError: select 'which' must be scalar or have the same sharding as cases` on 4 devices.

### sqd.uniquify_states: lexsort on uint64 words

`lax.sort` compares one key operand at a time, so `num_keys=B` (13 at n=100) packed into `ceil(B/8)`
order-preserving words (2) gives the same permutation at **1.79x (n=30) through 5.06x (n=127)**, N=200k;
this sort dominates the function, itself 14-27x `get_xsource`. Cost: an extra `[N, ceil(B/8)]` uint64
buffer, rows widened `8*ceil(B/8) - B` bytes (worst n=64, B=9 -> 16, +7/row; free at n=127, B=16), 1.09-1.69x
temp at N=1M — 0.07-0.17 GB at N=24M but 6-15 GB at the 2^31 ceiling, the wrong trade out-of-core
("Replacing that sort out-of-core: prototyped and rejected (2026-08-29)").

### sqd._is_lex_sorted: the filler test

A single filler is strictly increasing and used to pass (two or more fail as duplicates), and `hproj` does
not mask fillers, so it became a spurious basis state: **−1.118034 against a true −1.0**. Tested on the
packed byte, where genuine states have byte 0 < 128 (unpacked at n=2 a filler is [1, 1], a legitimate
state). O(1) on the last row, since all-255 fillers sort last: a full `np.any(states[:, 0] >> 7)` is
**5.25 ms at N=10M against 0.00012 ms**, ~4.4x the adjacent-row pass.

### sqd.get_xsource: search on uint64 words

`ceil(B/8)` comparisons per level instead of B, 2 against 13 at n=100; the byte form had an ~8x per-state
cliff across B > 8 (**15 ns/state at n=60, 189 at n=100**). Invariant: `lo` counts rows strictly below the
target, so after `ceil(log2(N)) + 1` halvings it is the insertion point.

### sqd._accumulate_diagonal: the output sharding

The output is 1-D, `template.shape[0]` long, while the template may be the 2-D `(N, nbytes)` state list, and `jnp.zeros` rejected a
rank-2 spec ("Length of sharding.spec (2) must be equal to aval's ndim (1)") on every sharded `sqd` call at
every `cache_level`. `NamedSharding`, since a bare `PartitionSpec` is rejected with no mesh context.

### sqd.compute_diagonal: the bit offset wraps at 8

With `& 255` rather than `& 7` the shift `7 - ibit` goes negative from `iterm=8`, i.e. past 8 Z terms in an
X group: **0.71 absolute error on 9 terms, and a 25% error in the end-to-end eigenvalue**.

## Moved from docstrings (2026-09-27)

Rationale, measurements and history cut from `rqutils/sqd.py` and `rqutils/ground_locg.py` docstrings to
meet the 3–5-line paragraph ceiling; each docstring points to its subsection.

### ground_locg module: the polish repairs the eigenvalue, not the eigenvector

Being second order in the eigenvector *angle* error, the Rayleigh-quotient polish repairs θ and leaves
the eigenvector as computed. For a near-degenerate lowest pair the returned `v` can be nearly orthogonal
to the true eigenvector while θ is still accurate to ten digits (measured `|⟨v_true|v⟩| = 0.447` against
a θ error of 1.2e-10). `_nullvec_3x3`'s cross products are the fragile step; once they lose the
eigenvector the polish has nothing to recover from. So audit the eigenvector, not only θ — κ becomes the
next iteration's search direction. It has *not* been shown that the iteration ever builds such a
projected matrix, since `_project_out` keeps the basis orthonormal by construction.

The module's measurements were first recorded in `markdown/locg.md`, which describes the pre-rewrite
module: its line numbers, "no pytest suite exists" note and several severity claims no longer hold, and
at least one failure mode it measured is no longer reachable now that the defects it compounded with are
fixed (see "ground_locg._reorthogonalize: the measured drift").

### ground_locg module: Chebyshev prefilter provenance

The technique is Chebyshev-filtered subspace iteration (ChFSI), standard in large-scale
electronic-structure codes (Banerjee et al. 2016; Zhou et al. 2006; Banerjee et al. 2018, used in
production in DFT-FE and the provenance `markdown/locg-chebyshev-prefilter.md` cites). Those papers
filter a whole subspace inside a self-consistent loop. The two-level complementary-subspace method of
the 2018 paper is **not** implemented and would not fit: it filters a subspace and solves its
complement, against the module's three-vector memory budget. A two-level *preconditioner* was
separately measured and rejected (0.68-0.98x, `markdown/deflation-preconditioner.md`): it improved
conditioning without opening the gap, which is what the iteration count tracks. Neither load-bearing
property (the Rayleigh-quotient lower edge, the true upper bound) is inherited from the references.

### ground_locg module: sharding verification

The re-orthogonalization adds two inner products per iteration, following the same reduction pattern as
the existing ones. `poc/sharding.py` exercises the solver on a four-device mesh (virtual CPU devices via
`XLA_FLAGS=--xla_force_host_platform_device_count=4`), agreeing with the single-device result to
8.9e-16; real multi-GPU behaviour remains unverified.

### ground_locg.residual_floor: the measured constant

The floor `ε·‖A‖₂` is dimension-independent, measured over n = 70 to 32768 and six decades of `‖A‖₂`,
dense and matrix-free, both coefficient dtypes: 27 samples, the constant spanning 0.49-1.26 (median
0.84, a 2.6x spread) while the n-scaled form spans 306x and so is not the mechanism. The factor 4 is a 3.2x margin
over the worst constant. `Σ|c_k|`, which `sqd` passes as the bound, measured a 1.56-1.90x over-estimate
of `‖H‖₂` on 1D XXZ fixtures. Full record: "The eigen-residual floor is `eps·‖H‖` with no dimension
dependence, and `tol` is now absolute (2026-08-31)".

### ground_locg._check_tols: why only rtol takes None

`rtol`'s default is the *promoted operator dtype's* epsilon, which no literal in the signature can
express: a hardcoded `8.88e-16` is right for float64 and unsatisfiable by 1.3e8x on a float32 problem,
which `test/test_ground_locg.py` exercises. `atol` has no dtype-derived absolute residual a caller would
want, so it takes a plain 0.0 and `None` is an error rather than a synonym. The below-floor check is
conditioned on `rtol == 0` because this repo already paid for a guard that fired on correct input (an
overflow count that included discarded padding, reported 763,677 beside a bit-exact result).

### ground_locg._check_prefilter: what the gate absorbed

`_chebyshev_prefilter` runs only `if degree > 1 and cycles > 0`, a branch with an implicit `else`, so an
out-of-range value was absorbed into a silent no-op: `(2, -1)`, `(-4, 2)` and `(True, 2)` all returned
the exact unfiltered energy at zero speedup, which reads as "the prefilter does not help on my problem"
— the one misdiagnosis this option cannot afford, since callers are told to A/B it. Malformed *types*
were no better: `(2,)`, `"32,2"` and `32` reached `ground_locg`'s tuple unpack and surfaced its
`ValueError`/`TypeError` from inside a public entry point. The legal-no-op versus error distinction is
only expressible here, since the single gate cannot make it. See also "Validation belongs to the module
that owns the gate".

### ground_locg._gershgorin_bound: cost

One `O(N²)` reduction, measured 1.1-2.7x a single matvec at N=512-4096, against the 11 matvecs the
power iteration it replaced spent on an estimate that was not a bound at all. Sharding-preserving: an
elementwise `abs` and two reductions.

### ground_locg._chebyshev_prefilter: the measurements

- **The upper bound.** It came from 10 power-iteration steps, wrong twice over: power iteration converges
  to the largest-*magnitude* eigenvalue, so on a negative-leaning spectrum the interval **inverts**, and
  even sign-repaired a fixed step count under-estimates. Measured: the n=2 Heisenberg chain returned
  +0.25 for a true -0.75; the bound was invalid in 16 of 25 XXZ configurations, with wrong answers in 2
  (`markdown/spinchain/rqutils-prefilter-bug.md`). The impossibility is Kuczynski & Wozniakowski, SIAM J.
  Matrix Anal. Appl. 13(4):1094-1122, 1992; the candidate table and the adversarial construction are in
  "No matvec-only upper bound on `λ_max` exists, so the prefilter takes one from structure".
- **The lower edge.** An accurate `lambda_1` instead of the running Rayleigh quotient measured 8.1x at
  relgap 1.3e-2 but an energy off by 15, silently, at relgap 4.0e-05: the interval then begins at
  `lambda_0` and damps the ground state with the rest.
- **Filtering alone** plateaus around 1e-5 to 1e-7, the lower edge following θ onto its own target
  (`markdown/locg-chebyshev-prefilter.md`).
- **No depleted residual.** A power-iteration start is *worse* than a random one (measured 177 LOBPCG
  iterations against 77): it collapses onto the dominant direction, leaving a residual with nothing to
  expose. A polynomial filter suppresses the unwanted band multiplicatively and leaves the residual rich
  in the directions block-size-1 LOBPCG can search.

### ground_locg.ground_locg: prefilter tuning

- **Correctness with a valid bound**: the returned eigenpair is the one the solver was going to reach
  (eigenvector overlap 1.0000000 against the unfiltered result, energies agreeing with `eigsh(tol=0)` to
  2.8e-14). The ~11 matvecs of a `λ_max` estimate are gone, since `prefilter_hi` needs no iteration.
- **`(32, 2)`**: across 27 connected-subspace configurations (3 sizes x 3 seeds x 3 anisotropies, every
  arm converged and correct to <1e-9) a median **1.88x** wall-clock reduction, range 1.25-3.95x, at
  *fewer* matvecs than the alternatives.
- **The knobs**: amplification outside the band grows like `cosh(degree · arccosh|x|)`, roughly
  exponentially, for `degree` matvecs. Cycle 1 does most of the work (growth factor 1e8-1e12), cycle 2
  refines once, past that θ is near `λ_0`.
- **Medians**: `(16, 4)` 1.41x, `(32, 2)` 1.88x, `(48, 2)` 1.79x; on a narrower sweep `(64, 2)` 2.29x and
  `(128, 2)` 1.68x with a 1.01x floor. An independent `spinchain` sweep measured `degree=64` as the
  *weakest* arm on every path (median 1.35x through `sqd`, 0.74-0.80x dense), so only `(32, 2)` is
  recommended without qualification. Longer solves favour the higher end: the 249- and 573-iteration
  cases measured 3.6-5.1x at `degree` 48-96.
- All single-device CPU; `poc/prefilter_gpu.py` sweeps the grid on a GPU ("The GPU prefilter sweep: the
  peak transfers, its location does not (2026-09-04)").

### ground_locg.ground_locg: the out-of-range one-hot

The one-hot is built as `iota == xinit`, so an out-of-range (or negative) index matches nothing and
yields the **zero vector**, from which the solver returned `0.0` with `converged=True`: measured
`xinit=16` on a dimension-16 operator whose true minimum was -1.5. Without the `vspace` check, the
callable path failed with an opaque "NoneType is not subscriptable".

### ground_locg.ground_locg: the rtol_scale key

`rtol_scale` (named `reltol` before 2026-09-01) holds `‖Ax‖ + |θ|`. Once converged `x` is the ground
eigenvector, so this is ~`2|λ_min|` (verified 3.9990 against `|λ_min| = 2`), **not** `2‖A‖₂`; the two
coincide only when the ground state is also the largest-magnitude one. See "ground_locg debug
diagnostics: `reltol` became `rtol_scale`".

### ground_locg._reorthogonalize: the measured drift

Measured `|⟨x|y⟩| = 1.0` at shift 1e9 without it. Removing it degrades the worst `|⟨x|y⟩|` over 60
iterations from ~5e-17 to 2.5e-12 at shift 1e6 and **1.0e-08 at shift 1e9** — eight orders of
magnitude — and `TestBasisOrthogonality` fails 3 of its 4 arms. θ still matches `eigvalsh` throughout,
so nothing else in the suite notices: the drift is underway but has not collapsed the basis. The audit's
`|⟨x|y⟩| = 1.0` needed the 2000-iteration runs that the `reltol` sign error (item I4) used to force, where
the fixed solver converges in 8-46 — hence the direct assertion off the `debug=True` diagnostics. Both
callers are `@jax.jit`-decorated, so reassigning it in a live session reuses the compiled kernel and
both arms return bit-identical numbers that look like "no effect".

### ground_locg._subtract_projections: why not a matmul

Reassociating the summation order into a matmul measured consistently worse in the near-degenerate
regime: over 4000 adversarial cases with `r` almost entirely inside `span(x, y)` plus an orthogonal part
of 1e-14..1e-6, both forms hold residual orthogonality at machine epsilon, but the matmul's worst
`|⟨b|p⟩|` is 8.3e-17 against 6.2e-17. Neither is broken, so this is a judgement call: a few ops are not
worth a 33% erosion of the quantity these guards protect.

### sqd.sqd: why keyword-only

Everything after `states` used to be positional-or-keyword, which made `sqd(ham, states, True)` a valid
`states_size` of 1 (`True == 1`) rather than the `return_eigvec` the caller meant — no error, the array
pinned to one slot. The parameters are unrelated, so no bare positional was worth preserving; `hproj`'s
`unique_states` is keyword-only for the same reason.

### sqd.sqd: states_size on a mesh

Rounding up to a multiple of `mesh.size` widens the coalescing `states_size` exists for rather than
defeating it: on a 4-device mesh `states_size` 33 through 36 all share one compiled kernel (707 ms to
compile, then ~16 ms) while 37 rounds to 40 and compiles afresh. The large-`N` case (4.9% at N=1M, 39.8%
at N=24M, 82.0% at N=144k; 97% of a cold solve compiling at N=2000; `N/8` buckets at 5.09 s against 5.10
s; 7.7 GB at N=24M) is "`states_size`'s power-of-two padding".

### sqd.sqd: one packed flag for both directions

Before 2026-08-30 the returned basis was unpacked either way, so a caller passing `packed=True` and
comparing against an unpacked array now gets a shape mismatch. That fails loudly (`np.array_equal` is
`False` on differing shapes), which is why one flag governs both directions instead of a second
`return_packed`: two flags make four combinations, two of them format conversions, and `sqd` is not a
converter — `pack_states` and `unpack_states` are.

### sqd.sqd: the prefilter default

1.49x median end-to-end (range 1.15-1.70x, 6 sampled XXZ subspaces at n=14-18, dim 978-3982, every arm
correct to <1e-9 against `eigsh(tol=0)`). That is **below** the 1.88x median of dense `ground_locg`, and
the gap is the point: the filter spends `cycles * (degree + 1)` matvecs up front, and `apply_h`'s
gather-heavy kernel is cheap enough that they cost proportionally more. Counting *iterations* would
report 5.02x, the wrong unit for a caller. `None` restores the pre-2026-08-28 graph exactly. See
"`precond` was removed; `sqd` defaults to `prefilter=(32, 2)`" and "Prefilter vs precond".

### sqd.sqd: degenerate ground states

Anything not basis-independent (per-site occupancies from `|v_i|²`, say) gets an arbitrary member's
value rather than the eigenspace average. The solver cannot report the case: the Rayleigh-Ritz 3x3
spans the search basis `{x, y, p}`, whose spacing reflects that basis, not the multiplicity — measured
2.0, not 0, on a 2-fold degenerate operator. Deflate-and-resolve is the other second opinion. The
returned member moves with the seed, the prefilter default or the iteration, none treated as breaking.

### sqd.sqd: non-convergence raises

The convergence flag used to be discarded and the non-converged `state.theta` returned — a valid
variational **upper bound**, so finite, real and above the true minimum, indistinguishable from a
correct result. `markdown/locg.md` records that this absence "is the reason I4 could hide", a sign error
that made the convergence test unsatisfiable so the solver silently never converged. `run_sqd` is
jitted, so `converged` is traced there and `sqd` raises on it once concrete.

### sqd._check_states_shape: three mistakes, and the packed declaration

- **Re-feeding packed states.** `sqd` takes *unpacked* states while the intermediate a caller keeps
  (from `uniquify_states`, or `pack_states` directly) is *packed*; re-packing it is `astype(uint8)` (a
  no-op) then `packbits` reading each byte as one bit, so the subspace silently changes. Realistic loop:
  `sqd`, configuration recovery, `sqd` again. At `num_qubits <= 7` a packed row is one byte, so a 1-qubit
  Hamiltonian's shape genuinely matches; `pack_states`' binary check catches it, packed bytes exceeding 1.
- **A transposed array** `(num_qubits, subspace_dim)`, and **a mismatched Hamiltonian**.
- **`packed=True` is a declaration** because at `num_qubits == 1` the shapes are undecidable:
  `sqd(unpacked, packed=True)` returns `+1.0` where the truth is `-1.0`, since `[[0], [1]]` is a legal
  *packed* array meaning something else. The flag closes the one direction nothing else can.
- **Coercion**: both entry points document `states` "as an array of integers or booleans" and `hproj`
  accepted a list of lists; reading `.ndim` off the raw argument would narrow that with an
  `AttributeError`. It also made the entry points agree, where `sqd` used to require an array.

### sqd._hproj_cols_elems: module scope

Defined inside `hproj`, a fresh cache key per call meant a full retrace and XLA compile every call:
0.098 s against 0.0001 s warm, essentially `hproj`'s whole steady-state cost. Nothing else there is worth
hoisting (`PauliSumXZ.from_paulisum` 0.4 ms, `packbits` noise): compile cost versus host work. A captured
`states_p` is part of the function object too, so as an argument it is traced and keyed on shape and
dtype, which repeat.

### sqd._pack_state_keys: why B > 8 raises

The limit was previously only *described*, and the failure is worse than truncation: byte 0 is most
significant, so at `B = 9` its shift is `8 * (9 - 1) = 64` bits on a `uint64` and it vanishes. Measured,
two 9-byte rows differing only in byte 0 both pack to key `0`, aliasing distinct states and destroying
the lex-order equivalence the search depends on. `B = ceil((n + 1) / 8)`, so `n >= 64` reaches it.

### sqd._is_lex_sorted: cost

One vectorized pass with no early exit, so unsorted input costs the same as sorted: 12-14% of `hproj`
(A/B'd end-to-end; both `O(N)`, so the ratio is flat in N) and ~20 ms standalone at N=1M, flat in `B`.
Cheap enough to be unconditional on a reference path; `sqd` never reaches it. Two or more fillers fail
the strictness test as duplicates; the high-bit test is for the single one.

### sqd._pack_state_words: either padding end

Trailing-padding is also order-preserving: appending a constant number of zero bytes is a left-shift by
`8 * pad`, and a constant left-shift is monotonic. Verified exhaustively at `B = 3` and over 20000 random
pairs at `B = 9`; a mutation to the other end leaves the whole suite green.

### sqd._word_less_than: no cumprod prefix

Word-wise comparison measured 3.6-7.7x on the whole search across n=64..200, bit-identical to the
byte-wise form it replaced. That predecessor used a `jnp.cumprod` prefix and had to pin
`dtype=jnp.uint8`: `cumprod` rejects a bool accumulator and promotes to int64, materializing the
`[N, B]` mask at 8 bytes per element — 192 MB of transients against 23 MB at N=1M, B=12, and 1.53x on
the J-fold precompute. Two or four scalar comparisons need no prefix.

### sqd.get_xsource: why a search, not a sort

- Lex-sortedness was always required — the sort-based implementation also returned a wrong answer
  otherwise — but was never stated. `hproj(unique_states=True)` skipped its `np.unique` and returned a
  wrong, non-symmetric matrix on unsorted input; it now raises
  (`test_sqd_hproj.py::TestHproj::test_unsorted_input_with_unique_states_raises`).
- The former implementation sorted a concatenated `[2N, B]` array of `S` and `S ^ X`: the `2N`
  allocation capped `N` at `2^31` (the sort runs on one device), and it dominated runtime at 66-97% of a
  solve. A `searchsorted` is a pure gather, so it also shards. Measured on CPU at 12-25x per signature
  and 12-17x on the J-fold precompute, and **5.15x at N=64M on an NVIDIA GH200** (a GPU sort is well
  optimized relative to its gather, so the ratio compresses while the direction holds);
  `markdown/scaling-pocs.md`.
- `lax.sort` was observed to leak GPU memory (up to 5 GB at `(5M, 9)`); re-measured on that GH200 against
  a pinned copy of the old sort it **did not reproduce** (~0.95 GB of transients at `(5M, 4)` reclaimed
  every repetition). The removal never depended on it.
- The wide path first compared rows one **byte** at a time, ~8x per state across the boundary (15
  ns/state at n=60 against 189 at n=100, N=300k). Words make a level `ceil((n+1)/64)` comparisons,
  3.6-7.7x on the whole search across n=64..200; the `B <= 8` path was untouched (0.83-1.18x, noise).
- The previous implementation returned *assorted* negative values for absent sources (`I[k+1] - N`,
  computed unconditionally) rather than exactly `-1`; consumers could not tell, since `apply_xgrp`
  gathers with `mode="fill", wrap_negative_indices=False`.
