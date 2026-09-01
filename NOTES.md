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

2026-08-28. `tests/_sharded_prefilter.py` covers the prefilter on a mesh, but only through
`ground_locg` with a dense `einsum` matvec on an unpadded power-of-two vector —
`docs/locg-chebyshev-prefilter.md` said so and deferred the rest to `sqd`. That deferral is now closed
by `tests/_sharded_sqd_prefilter.py`.

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

2026-08-28, fixing `docs/rqutils-prefilter-bug.md`. `_lambda_max_bound` used 10 power steps, which
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

`test_svsim.py::TestShardedOutput` (subprocessing `tests/_sharded_svsim.py`) was checked because the
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

### Why `tests/_sharded_*.py` are files rather than inline strings

The leading underscore keeps them uncollected; `test_sqd.py::TestShardedCacheLevels` subprocesses
`_sharded_cache_levels.py` under `XLA_FLAGS=--xla_force_host_platform_device_count=4`, because the
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

`docs/locg.md` catalogues seven defects (I1–I7) that each failed *silently*, returning a plausible
wrong number rather than raising. Don't "simplify" the balancing, the re-orthogonalizations, or the
zero-direction masks.

**`docs/locg.md` is stale** — it audits the pre-rewrite module, so its line numbers, its "no pytest
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
(`docs/skqd.md`).

### `sqd`: why the initial vector is a spread, not a one-hot

A one-hot cannot leave the connected component of the projected Hamiltonian that contains it, so a
subspace whose Hamiltonian splits into disconnected blocks silently returned that block's minimum with
`converged=True`. `vinit_from_min_diag` still weights the minimum-diagonal state heavily on top of the
spread. Don't "simplify" either back to a one-hot — `tests/test_sqd.py::TestSqdInitialVector` covers
both failure modes.

### `qprint`: test the full `fmt` × `output` grid, not a diagonal of it

Four bugs lived in cells nothing exercised, including `fmt='matrix'` being un-instantiable for *every*
input (`QPrintMatrix` never implemented the abstract `_add_labels`, which it does not need — it
overrides `_make_lines` and positions terms by row/column). An amplitude of exactly `1` is suppressed,
which is right when a basis label follows and wrong when nothing does; text-mode labels also carry the
`*` separator as a prefix, so the two renderings can disagree while each looks fine alone.
Cross-rendering assertions are what catch that — see `tests/test_qprint.py::TestAmplitudeAndSeparator`.

### `paulis/general`: the `npmod` gating bug

Gating Python-level shape inference on `if npmod is np:` broke the entire `npmod=jnp` path in three
separate places (`components` plus the since-removed `compose` and `truncate`, all raising
`TypeError: object of type 'int' has no len()` on a scalar `dim`). `_normalize_dim` is now called
unconditionally at every site for exactly this reason.

## The MLX port: deleted, and what it left behind

The JAX solver measured faster even on the MLX GPU backend, so the port had no performance case, and
nothing in the tree imported or ran it. Don't reintroduce a second solver implementation without that
measurement going the other way first.

`docs/mlx-metal-kernels.md` is the historical record of the fused-Metal-kernel work — three kernels
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

The six scaling POCs live under `examples/scaling/`, findings in `docs/scaling-pocs.md`.

**The POCs no longer have a baseline in the library and must not be pointed at one.**
`poc1.xsource_sort_legacy` is a verbatim copy of the pre-23fb226 sort and is the timing baseline for
both POC 1 and POC 8; their *correctness* arms still compare against `get_xsource`, which is the point
(agreement with what ships is now a regression test). Point a timing arm at the library and you get
searchsorted-versus-searchsorted: the first GPU run of `poc8_gpu_unverified.py` reported
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

Two cross-references in `tests/` were "fixed" by removing dead paths while their surrounding claims
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
`examples/scaling/` scripts, i.e. exactly the code that pushes N, which reached the iota with neither
entry-point guard in the chain. Being static it fires at trace time and costs nothing per call. That
placement also made **both** sides of the boundary cheap to pin: `jax.eval_shape` traces the guard
without allocating (~5 ms per side), where reaching it through `hproj` cost 23 s. `TestInt32Ceiling`
covers both, so the off-by-one that used to survive (`>` → `>=`, rejecting the largest legal size) is
now caught.

`uniquify_states`' single-device `jax.lax.sort` is the reason for the limit's magnitude;
`get_xsource` no longer contributes — it is a binary search into the already-sorted list, not a sort
of a stacked `2N` array.

### Replacing that sort out-of-core: prototyped and rejected (2026-08-29)

The sort is still the ceiling, and the obvious move is `poc9_ooc_uniquify.py`'s chunk-sort-and-merge,
which bounds the working set by a chosen chunk size rather than by `N`. That POC bails out at `B > 8`
("no uint64 equivalence available"), so it never covered `n = 100`. `_pack_state_words` removes that
obstacle — wide rows pack into `ceil(B/8)` uint64 columns, and a structured-dtype view makes
`np.unique` / `np.union1d` lexicographic over them with no row comparator — so the wide case was built
and measured. **It loses on both axes it exists to win.** At n=100, B=13, N=8M host-side:

| approach | time | peak RSS |
| --- | --- | --- |
| `np.unique(rows, axis=0)` (incumbent shape) | 7.3 s | **365 MB** |
| word-packed `np.unique` | **4.9 s** | 1655 MB |
| chunked sort + merge tree on words | 31.3 s | 1048 MB |

4.3× slower and 2.9× more peak memory than plain `np.unique`. Output verified identical in all arms.

The reason is worth keeping, because it is the same fact that makes the *in-JAX* fix a good trade and
this one a bad trade: **packing widens the data.** `8*ceil(B/8) - B` bytes per row — +7 at n=64
(B=9→16), +3 at n=100, free at n=127. Speed bought with memory is right for the JAX sort, whose own
working set dominates, and exactly wrong for a design whose entire purpose is bounding memory. Do not
read the shipped `uniquify_states` change as a step toward an out-of-core one; they pull opposite ways.

Two things the POC's own docstring already says, confirmed here and worth not rediscovering: chunking
removes the single-device *sort* but **distributes nothing** (sequential merge, full result on one
host), and a real multi-node uniquify needs a range-partitioned shuffle, for which chunk-local sorting
is the per-node kernel and not the algorithm.

### At n=100 the binding constraint is the xsources cache, not the sort (2026-08-29)

Scaling attention has been on `uniquify_states`' sort because it sets the `2^31` ceiling. But an actual
n=100 solve is dominated by something else. Budget at `B = 13`, `J = 50`, `cache_level[0] = 1`:

| N | states | vectors (~6) | **xsources `[J,N]` int32** | total |
| --- | --- | --- | --- | --- |
| 2^24 (17M) | 0.2 GB | 1.6 GB | **3.4 GB** | 5.2 GB |
| 2^28 (268M) | 3.5 GB | 25.8 GB | **53.7 GB** | 82.9 GB |
| 2^31 (2147M) | 27.9 GB | 206.2 GB | **429.5 GB** | 663.6 GB |

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

**The partial-J dial is the answer, and `docs/scaling-pocs.md` §2 already scoped it: "only worth
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
footprint** — which is the condition `docs/scaling-pocs.md` §2 already gates the whole idea on, and is
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
K=1 picture, and the reverse of what motivated the partial-J work. `docs/scaling-pocs.md` says the
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

| variant | time | vs baseline | overhead |
| --- | --- | --- | --- |
| baseline `searchsorted` | 67.2 ms | 1.00× | — |
| BF, cap given, no check | 24.8 ms | 2.70× | — |
| BF, cap given, **with check** | 24.8 ms | 2.71× | **-0.3%** (noise) |
| BF, cap **derived** + check | 31.7 ms | 2.12× | +27.6% |

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

| cached `J'` | cache | recompute, plain | recompute, + BF | BF gain | vs full cache |
| --- | --- | --- | --- | --- | --- |
| 0 | 0 MB | 447.4 ms | **47.4 ms** | **9.43×** | 7.71× |
| 4 | 9.6 MB | 336.5 ms | **38.6 ms** | **8.73×** | 6.27× |
| 8 | 19.2 MB | 162.6 ms | **28.5 ms** | **5.70×** | 4.64× |
| 12 | 28.8 MB | 112.6 ms | **20.0 ms** | **5.64×** | 3.25× |

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

| target FP | bits/item | k | at N=24M | at N=2^31 | exact bitmap, any N |
| --- | --- | --- | --- | --- | --- |
| 10% | 4.79 | 3 | 14 MB | 1.3 GB | n=30: 0.13 GB |
| 1% | 9.59 | 7 | 29 MB | 2.6 GB | n=34: 2.15 GB |
| 0.1% | 14.38 | 10 | 43 MB | 3.9 GB | n=40: **137 GB** |

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
is the one thing `docs/xsources-cache-budget.md` §7 flagged as missing: **"nothing measured through a
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
  `docs/scaling-pocs.md`) is **weighted by call count**. It is the cost of paying the `J`-fold search
  *per matvec* against paying it once — i.e. what `cache_level[0] = 0` actually costs.
- **4.5–8.4%** is the precompute measured **once**, as a fraction of the `(1,*)` solve it runs inside.

Both describe `sqd.py:1008`; they differ in how many times the work is counted.
`docs/skqd-sqd-solve-tolerance.md` already confirmed the first "correct as stated" and named this exact
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

**Independently corroborated by the record.** `docs/skqd-sqd-solve-tolerance.md` found `(1,1)` "the only
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

The open question the entry above surfaced: `cache_level[1]` is all-or-nothing across X groups, so
level 2's 808 B/slot at `J=101` cannot be traded down. Prototyped outside the library as two summed
`_apply_h_kernel` calls -- `(1, 2)` over the first `J'` groups, `(1, 0)` over the rest -- and measured.
**It works, and unlike the Bloom pre-filter it survives composition into a solve.**

**Real `sqd()` solves**, n=100, J=100, K=99, N=50,072, `states_size` 65536. Every arm returned
`-52.23606798`, identical to 8 decimals:

| `J'` | diagonal store | memory saved | solve | vs all-cached |
| --- | --- | --- | --- | --- |
| 0 | 0 MB | 100% | 3243.8 ms | 5.13× |
| 25 | 13.1 MB | 75% | 2004.9 ms | **3.17×** |
| 50 | 26.2 MB | 50% | 1552.2 ms | **2.45×** |
| 75 | 39.3 MB | 25% | 1091.0 ms | 1.73× |
| 100 | 52.4 MB | 0% | 632.3 ms | 1.00× |

So **half the diagonal memory for 2.45×**, or 75% of it for 3.17×. At `J=101, K=100` that is
808 → 404 B/slot; against the 24M-slot budget above, `(1, 2)`'s 43.5 GB → ~24 GB.

**Why it composes where the filter did not, which is the transferable part.** The implied matvec count
is **129**, stable across n=40/60/100 (from `(s10 - s12) / (mv10 - mv12)`). The diagonal cache is
*consumed* ~129 times per solve, so a per-matvec saving multiplies; the filter targeted a one-off
precompute at 4.5–8.4% of a solve, so Amdahl divided it. Matvec ratios of 5.0–8.2× became solve ratios
of **3.59–5.13×** — compression, not collapse. **The question to ask of any `sqd` optimization is how
many times per solve its target is paid**, and it is the same "weighted by call count" distinction that
made the filter look worth building.

**Three ways this axis behaves better than `xcache_groups` does on its own:**

- **Linear and monotonic**, no cliff. The X axis "never reaches full-cache time" with a
  disproportionate last step; this one lands on it.
- **Peak temp memory does not regress.** 1.1 MB at an intermediate split against 0.0–0.5 MB at the
  endpoints (XLA `memory_analysis`), i.e. 3.5% of the 31.5 MB store at J=60 — where the X-axis split
  measured 9.0 MB against 10.4 MB, a large fraction of its cache. End-to-end the two-arm overhead is
  **nil**: `J' = J` measured 632.3 ms against the single-kernel 649.1 (1.027×, noise).
  **Superseded in the entry below**: that 1.1 MB is not flat in `N` — it is 16 B/slot, so it scales
  linearly. The invariant is the *ratio*, 4.0% of the memory the split gives back at every
  `states_size`, and `4/J` in the group count.
- **Bit-identical at every split** — `max |Δ|` exactly `0.00e+00` against the all-cached reference at
  every `J'`, at both n=60 and n=100.

**The projection model was validated against points it was not calibrated on**, since `NOTES.md`
records the analogous X-axis model drifting 17.5% mid-range. Calibrating
`solve(J') = fixed + niter * matvec(J')` on the two endpoints only (`niter = 129`, `fixed = 245.9 ms`)
and testing the middle:

| `J'` | projected | measured | residual |
| --- | --- | --- | --- |
| 25 | 2050.7 | 2004.9 | +2.3% |
| 50 | 1665.6 | 1552.2 | **+7.3%** |
| 75 | 1207.3 | 1091.0 | **+10.7%** |

Worst non-calibration residual **10.7%**, and it errs *pessimistic* — real solves beat the projection.
Better than the X axis's 17.5%, and the same rule follows: **a budget API can size the cache by exact
arithmetic and must not advertise a runtime.**

**Two constraints any design must respect, both measured.**

**Do not split both axes.** Not because of the arm count, but because `cache_level[0] = 0` is
catastrophic per matvec: over all J groups at n=60, `(0, 2)` costs **73.97 ms against `(1, 2)`'s
1.80** — a **41×** gap, consistent with the documented 59.8×. Any X-axis split reintroduces it, so a
shared split point across both axes would pay 41× to save diagonal bytes. A four-arm prototype measured
39.37 ms against the two-arm 4.97 at the same diagonal split, and the cause is that arm's presence, not
the arm count. **The two axes are not symmetric: the diagonal axis is cheap to split (5.1× spread), the
X axis is expensive (41×).** So the dial belongs on the diagonal axis alone, orthogonal to
`xcache_groups`, with `cache_level[0]` left at 1.

**One compiled variant per distinct `J'`, and power-of-two rounding does not help — it makes it worse.**
13 distinct splits produced 13 variants; rounding `J'` up to a power of two produced **16**, because
both arms' lengths vary together (`xs[:jr]` and `xs[jr:]` are two shapes, and the complement is not a
power of two). This is unlike the pre-filter's `cap`, where rounding bounded the count. Acceptable
because `J'` is static per solve, but a caller sweeping it pays one compile per value, and that must be
documented rather than discovered.

**Caveats.** One laptop CPU. `N` ≈ 50k with `states_size` 65536 in every run, so the peak-memory claim
was unconfirmed at large `N` — **now measured, in the entry below**: it holds as a ratio (4.0% of
savings, invariant in `N`) rather than as the absolute figure quoted here. Fixtures are 1D Heisenberg with a subspace half-closed under a weight-preserving hop, not
sampled from a circuit. The intermediate solves came from throwaway `run_sqd` instrumentation behind an
env var, since `sqd` has no such parameter; it was removed and the file restored from a `cp` backup
(626 passed after).


### The diagonal split at large `N`: the overhead is a *ratio*, and "flat 1.1 MB" was an artifact (2026-08-30)

The entry above left one load-bearing claim unmeasured — peak temp memory "flat at 1.1 MB across every
intermediate split", from a single `states_size` of 65536. Swept to 2^21 it is **not flat**, and the
corrected statement is stronger.

**Peak temp is exactly 16 B/slot, and it scales linearly with `N`:**

| `states_size` | full diagonal store | peak temp at `J/2` | peak B/slot |
| --- | --- | --- | --- |
| 2^16 | 52.4 MB | 1.05 MB | **16.0** |
| 2^18 | 209.7 MB | 4.20 MB | **16.0** |
| 2^20 | 838.9 MB | 16.78 MB | **16.0** |
| 2^21 | 1677.7 MB | 33.56 MB | **16.0** |

Two float64 output buffers, one per kernel arm — `O(N)`, not `O(J·N)`. **That is why the
`xcache_groups` hazard does not occur here**: its two-kernel overhead was a fraction of a `4·J·N`
int32 cache and could exceed the saving (9.0 MB against 10.4), while this overhead is `O(N)` against a
`8·J·N` store.

**So the invariant is a ratio, not an absolute.** Overhead against the memory the split gives back is
**4.0% at every `states_size`** — invariant because it is a quotient of two terms both linear in `N`,
which is a far stronger guarantee than a small number at one size.

**It depends on `J` as `4/J`, measured at `states_size = 2^18`, `J' = J/2`:**

| `J` | store B/slot | peak B/slot | overhead vs saved |
| --- | --- | --- | --- |
| 10 | 80 | 16.0 | **40.0%** |
| 25 | 200 | 16.0 | 16.0% |
| 50 | 400 | 16.0 | 8.0% |
| 100 | 800 | 16.0 | **4.0%** |
| 200 | 1600 | 16.0 | **2.0%** |

`16 / (4·J)` exactly. (A `2/J` prediction stated while working this out was wrong by 2× — it divided
the 16 B/slot by the *full* store rather than the half a `J' = J/2` split gives back. Measured beats
that prediction; the `1/J` shape was right, the constant was not.) **The dial is only cheap at large
`J`** — 40% overhead at `J = 10` — but `J = 10` is also where the whole feature is pointless.

**The time ratios hold at scale, on a real Hamiltonian.** 1D Heisenberg n=100, J=100, matvec:

| `N` | `states_size` | store | all-cached | `J/2` | none | `J/2` vs all | exact |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 50,072 | 65536 | 52.4 MB | 4.49 ms | 12.49 | 28.04 | **2.78×** | `0.0e+00` |
| 187,846 | 262144 | 209.7 MB | 14.70 ms | 38.21 | 79.80 | **2.60×** | `0.0e+00` |
| 501,157 | 524288 | 419.4 MB | 19.99 ms | 56.51 | 123.97 | **2.83×** | `0.0e+00` |

Stable across a 10× range in `N`, and slightly better than the 2.45× recorded above. Bit-identical at
every size.

**A trap worth recording, because it produced a 25–39× phantom.** Timing this on
*synthetic shape-only* arrays — the right approach for peak memory, since that depends on shapes alone —
measured the `J/2` split at 25–39× rather than 2.6–2.8×. The cause is `_accumulate_diagonal`'s early
exit: it stops at the first zero coefficient in a zero-padded Z group, and `rng.normal` coefficients are
never zero, so all `K = 99` terms ran per group where a real Hamiltonian's structured groups stop far
sooner. **Synthetic arrays are valid for memory and invalid for time** whenever a kernel's trip count is
data-dependent. `NOTES.md` already records the converse trap (a *broken* arm flatters its own benchmark
by doing less work); this is the same lesson from the other side — an unrealistically *dense* fixture
punishes the arm that depends on sparsity.


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

Proposed as the one lever on the `(0, 0)` floor. That floor is **120 B/slot** measured, of which only 13
is the Hamiltonian: 32 B/slot is `_State`'s four carried vectors (`x, y, r, ax`) and ~75 is transients.
At `2^31` slots the floor alone is **258 GB**, and no `cache_level` setting touches it.

**This is a different proposal from POC 6, which is already rejected.**
`examples/scaling/poc6_mixed_precision.py` runs the matvec *arithmetic* in f32 and casts back, keeping
f64 *storage* — a bandwidth optimization. Demoting a carried vector is a memory optimization. The POC's
verdict (`docs/scaling-pocs.md` §6, **reject**) does not cover it, and re-running the POC confirms that
verdict still reproduces: 1.17–1.30× at fixed iterations, three of four converged solves hitting
`maxiter=300`, **0.42× end-to-end**, and 6d's naive form converging in 9 iterations with
`converged=True` and a **4.42% relative error**.

**But the shared discriminator settles the storage variant too, and cheaply.** Take a converged
eigenpair (1D Heisenberg n=40, N≈20k, `theta = -28.236067977500`, 66 iterations, `‖A‖ ≤ Σ|c| = 117`)
and round individual operands of `r = Ax - θx` to f32:

| residual path | `‖r‖` | `‖r‖/‖A‖` |
| --- | --- | --- |
| f64 (today) | 2.813e-09 | 2.405e-11 |
| **`r` stored f32** | **2.813e-09** | **2.405e-11** |
| **`ax` stored f32** | **6.780e-07** | 5.795e-09 |

`ground_locg`'s default `tol` is `eps(f64) = 2.22e-16`, so an f32 `ax` puts the residual floor
**3.1e6× above the tolerance** — the convergence test becomes unsatisfiable and the solver runs to
`maxiter`, which is precisely POC 6's measured failure arriving by a different route.

**The asymmetry is the finding, and it is structural.** Storing `r` at f32 changes nothing, because `r`
is already `O(1e-9)` and f32 carries ~7 significant digits of *relative* precision. Storing `ax` is
fatal because `ax` is `O(‖A‖) ≈ 117` while `r` is `O(1e-9)`: the subtraction is a catastrophic
cancellation of nearly equal `O(100)` quantities, and rounding the operands destroys the digits the
cancellation depends on. **That is a property of the arithmetic, not a tunable tolerance** — no
`work_dtype` handling or looser `tol` recovers it without accepting POC 6d's silent error.

**What that leaves is not worth the risk.** `x` is the answer and also an operand of the same
subtraction; `ax` is ruled out above. So at best `r` and `y` could be demoted: **8 of 120 B/slot, 6.7%**,
or 17 GB of 258 at `2^31`. Achieving it needs a mixed-dtype `_State` — `while_loop` requires the carry
types to agree, and `ground_locg:951` derives `work_dtype` from `result_type(xinit, matvec output)` and
casts `xinit` to it, so a per-field dtype is a change to every operation in the iteration. Against
`docs/locg.md`'s seven defects that each failed *silently*, 6.7% of the floor is not a good trade.

**For scale, the levers already available at that same problem size:** `cache_level` from `(1, 1)` to
`(0, 0)` is 1805 → 120 B/slot, **15.0×**, no code change; hand-sizing `states_size` at N=24M saves
**7.7 GB** for one keyword. Both dwarf the 6.7% and neither risks a silent wrong answer.

So the `(0, 0)` floor stands at 120 B/slot, and **258 GB at `2^31` is the honest ceiling** for this
solver on one device. Lowering it needs a different eigensolver structure — fewer carried `O(N)`
vectors, or a restart scheme that trades vectors for matvecs — not a dtype change. That is a separate
investigation and nothing here bears on it.


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


### Distributing `states` is feasible: hash-by-prefix ownership plus a local search (2026-08-30)

The `13 * N` replicated state list is the one term the `(0, 0)` floor cannot shed — 27.9 GB **per
device** at `N = 2^31` — and `sqd.py:1042` records it as a hard requirement: a partitioned `[N, B]`
"fails outright". Investigated with a breaking change permitted. **It is not a wall, and the literature
has the established solution.**

**What actually fails is narrower than "states must be replicated."** Per ingredient, on a `P('x', None)`
state array under a 4-device mesh:

| operation | partitioned |
| --- | --- |
| `bitwise_xor` (build `S ^ X`) | **OK**, `P('x', None)` |
| `_pack_state_keys` | **OK**, `P('x',)` |
| `jnp.searchsorted(keys, targets)` | **fails** — "Unmapped values passed to vmap cannot be sharded" |
| `keys[pos]` gather | fixable with `out_sharding=` |

The *targets* shard fine. What fails is that a binary search needs the whole sorted haystack visible to
every query. That is a data dependency, not a JAX gap — and it is only unavoidable **given a
range-agnostic partition**.

**The established scheme is Wietek & Läuchli**, *Phys. Rev. E* **98**, 033309 (2018), described in
`awietek.github.io/assets/pdf/thesis_awietek.pdf` §3.3. Split each basis state into **prefix** and
**postfix** bits; states sharing a prefix live on one rank; **a hash of the prefix bits gives the owning
rank**, so — quoting — *"we also don't have to store any information about their distribution. This
information is all encoded in the hash function."* Within a rank states stay lexicographically ordered,
so the local lookup is still a binary search, over `N/d` rows. The matvec buffers `(target, coefficient)`
pairs locally, does one **`MPI_Alltoallv`**, then each rank searches its own list.

**Hashing rather than range-partitioning, deliberately.** `poc11_range_partition.py` builds ordered
buckets from data-derived splitters, which is the natural fit here and keeps each shard sorted for free.
But the paper warns against exactly that: a random distribution *"reduces load balance problems
significantly since the communication structure is randomized. This is in stark contrast to distributing
the basis states in a linear fashion,"* where single processes take a multiple of the workload. Range
splitters are the linear case.

**Not applicable: DanceQ's approach.** `arxiv.org/abs/2407.14591` reaches 46 spins over ~256 nodes and
120 TiB with *"thread-local lookup tables for fast and synchronization-free state-to-index mapping"* —
no routing at all. That works because its basis is a **complete** U(1) particle-number sector, so
state → index is a closed-form combinatorial map (enumerative encoding, Cover 1973). **rqutils' subspace
is an arbitrary sampled set**, for which no such formula exists — which is why `get_xsource` searches in
the first place. The distinction is structural; do not cite DanceQ as evidence that this is easy.

**All three ingredients work in JAX**, verified on a 4-device mesh: prefix-hash ownership is elementwise
(`P('x',)`, balance 1011/1041/1016/1028 over 4096), `jax.lax.all_to_all` composes inside
`jax.shard_map`, and — the load-bearing one — **`jnp.searchsorted` against the local slice inside
`shard_map` works**, which is what removes the replicated haystack.

**A minimal end-to-end version is exact.** Range-partitioned variant at n=30, N=2564, d=4, hit rate
40.4%: **bit-identical to `get_xsource`** (1036 hits both, `np.array_equal` True), load imbalance
**1.02×**, and a **4.0× per-device** reduction in the state array. The JAX form compiled to **zero
`all-gather` / `all-reduce` / `collective-permute`** and 18 `all-to-all`.

**The cost is traffic, and it composes the right way — but only at `cache_level[0] = 1`.** Routing a
target and its answer is ~12 B/target (8 out as a uint64 key, 4 back as an int32 index), asymptotically
in `d`. At `N = 2^31`, n=100:

| `d` | states/device | routed per group/device |
| --- | --- | --- |
| 4 | **6.98 GB** (from 27.9) | 6.44 GB |
| 16 | 1.74 GB | 1.61 GB |
| 64 | 0.44 GB | 0.40 GB |
| 256 | 0.11 GB | 0.10 GB |

**The multiplier is `J`, not `J × niter`** — `get_xsource` runs once per solve in the precompute at
`cache_level[0] = 1` (`sqd.py:1008`), not inside the matvec. So at `d = 4` it is ~650 GB routed **once**
against 20.9 GB/device of permanent residency reclaimed, and it lands on the precompute this session
measured at only 4.5–8.4% of a solve. At `cache_level[0] = 0` the recompute *is* per matvec and the
traffic becomes ~84 TB per solve at `d=4` — unusable. **So this is only viable together with the source
cache**, which is the opposite of the Bloom filter's constraint and worth stating plainly.

**Why it composes where the filter did not:** the target is paid **once per solve**, not once per
matvec. That is the same "how many times is it paid" question, and here the answer is favourable.

**What is not done.** No hash-partitioned variant was built (the exact prototype range-partitions, which
the paper says will imbalance on real data — the measured 1.02× is on an evenly-split *sorted* array and
says nothing about a real subspace). Nothing measured on real interconnect; virtual devices make timings
meaningless per `CLAUDE.md`, so **no speed claim is made**. `uniquify_states`' sort is a separate
blocker with its own partial answer in `poc11_range_partition.py` (2.19–1.74×, zero collectives, but
splitter selection is host-side numpy and reassembly into the `[states_size, B]` contract is
unimplemented). And the diagonal builders need `popcount(S[i] & z)`, i.e. the state *bits* — those shard
elementwise, but that was not verified end-to-end here.

**Verdict: the only lever left that moves the ceiling, and the first one this session that survives
scrutiny.** It converts a hard 27.9 GB/device wall into `27.9/d`. It is also much larger than anything
else attempted here — a new partitioning contract, a routed `get_xsource`, and `uniquify_states`
rebuilt on POC 11 — so it is a project, not a patch.


### Hash-partitioning `states`: hash the whole key, not the prefix — the prefix collapses on real subspaces (2026-08-30)

The entry above proposed Wietek & Läuchli's scheme and flagged the untested claim: the 1.02×
imbalance was measured on an evenly-split *sorted* array and said nothing about a real subspace. Built
the hash-partitioned variant and tested it on fixtures chosen to break it. **One change to the
published scheme is required, and the reason is specific to SQD.**

**Prefix hashing collapses completely on a structured subspace.** Imbalance at n=60, N=200k, d=16,
12 prefix bits:

| fixture | prefix-hash | whole-key hash | range-split | distinct prefixes |
| --- | --- | --- | --- | --- |
| uniform | 1.12× | 1.01× | 1.00× | 2048 |
| fixed-weight n/2 | 1.16× | 1.02× | 1.00× | 2048 |
| low-weight n/8 | **3.81×** | 1.01× | 1.00× | 779 |
| banded (last n/4) | **16.00×** — total collapse | **1.09×** | 1.01× | **1** |

The last column is the mechanism: when excitations are confined to low qubits, **every state shares the
same high bits**, so the prefix hash is constant and one shard takes everything. This is the same root
cause `poc11_range_partition.py` already recorded for equal-range splitting on the most-significant
word — *"at n=100 that word is 7 bytes of leading pad"* — namely that **the high bits of an SQD state
carry almost no entropy.** A banded subspace is not contrived: it is what a circuit acting on a subset
of qubits produces.

**Hashing the whole key fixes it and costs nothing.** The key is already computed by
`_pack_state_keys`, so it is one extra `mix64`. Wietek & Läuchli hash the *prefix* because their scheme
needs prefix-grouping for local enumeration; **rqutils has no such requirement**, so the constraint does
not carry over.

**The one thing whole-key hashing gives up is the free local sort.** A range split hands each shard a
contiguous sorted block; a hash hands it a scattered subset that must be sorted per shard. That is
affordable and already prototyped: `d` independent sorts of `N/d` rows is exactly
`poc11_range_partition.py`'s phase 3 (`vmap` over `lax.sort`, zero collectives), and it runs **once at
setup**, not per `get_xsource` call. So the design is *hash to assign owners → per-shard sort → local
binary search*, with ownership still metadata-free.

**Verified exact on every fixture** against `get_xsource`, at n=60, d=16, with the hop chosen to act on
qubits each fixture populates (a hop on qubits 0–1 gives the banded fixture a 0% hit rate and tests
nothing):

| fixture | N | hit rate | imbalance | bit-identical | per-device |
| --- | --- | --- | --- | --- | --- |
| uniform | 74,944 | 39.9% | 1.03× | **yes** | 16× less |
| fixed-weight | 75,156 | 40.3% | 1.02× | **yes** | 16× less |
| banded | 6,435 | 53.3% | 1.14× | **yes** | 16× less |

**Balance holds as `d` grows, and the residual is Poisson, not structural.** Including a Zipf-sampled
fixture (2M shots → 148,976 unique, a **92.6% duplicate rate**, matching the 91–97.6% `NOTES.md`
measured for real shot distributions):

| fixture | N | d=4 | d=16 | d=64 | d=256 | d=1024 |
| --- | --- | --- | --- | --- | --- | --- |
| fixed-weight | 200,000 | 1.01× | 1.03× | 1.04× | 1.08× | 1.21× |
| banded | 6,435 | 1.01× | 1.09× | 1.22× | 1.63× | 2.86× |
| zipf-sampled | 148,976 | 1.01× | 1.01× | 1.04× | 1.09× | 1.22× |

**Checked against the balls-in-bins bound** `1 + sqrt(2 ln d / (N/d))` rather than asserted: predicted
and measured agree in magnitude at every point and **both depend only on `N/d`**, not on the fixture.
Banded degrades because its `N` is 6,435, so `d=1024` leaves ~6 states per shard — not because its
structure survives. **That is the hash working: it has erased the structure that destroyed prefix
hashing, leaving ordinary sampling noise**, which a capacity slack absorbs exactly as `poc11`'s `slack`
parameter does.

**Practical rule:** size shard capacity from `N/d` via the balls-in-bins bound, and keep `N/d` above
~1000 for a slack under 1.15×. At `N = 2^31` that permits `d` up to ~2×10^6 — far beyond any real mesh.

**Still unbuilt.** The verified prototype is numpy, so it validates the *algorithm*, not a JAX
implementation; the three JAX ingredients were verified separately in the entry above but not composed
with hashing. No real-interconnect measurement and **no speed claim**. `uniquify_states` still needs
rebuilding on `poc11`, and the diagonal builders' `popcount(S[i] & z)` path was not verified end-to-end.


### The variable-length routing has a primitive, and the capacity bound is the one the Bloom work lacked (2026-08-30)

The gap left by the hash-partitioning entry above: the prototype is numpy, and a JAX implementation
needs a routing step where **each shard receives a different, data-dependent count**. MPI does this with
`Alltoallv`; JAX requires static shapes. Explored, and both routes are now characterized.

**`jax.lax.ragged_all_to_all` exists** (JAX 0.11.1) and is the direct analogue — it takes
`(operand, output, input_offsets, send_sizes, output_offsets, recv_sizes)` and ships exactly the
per-destination counts.

**Its known defect does not apply here, which is the useful part.** FESOM2-JAX
(`arxiv.org/abs/2608.01546`) reports it "has a defective reverse-mode rule in JAX 10.1 and is usable
**forward-only**", and they fell back to a padded variant to keep exact adjoints. `get_xsource` returns
**integer indices** and is never differentiated — `sqd` does not backprop through it — so rqutils can use
the ragged path they had to abandon.

**It cannot be verified in this environment**: `JaxRuntimeError: UNIMPLEMENTED: HLO opcode
`ragged-all-to-all` is not supported by XLA:CPU ThunkEmitter`. So it is a GPU/TPU-only path here, and
nothing about it is measured — only that the API exists and the published objection is irrelevant to
this use.

**The padded fallback is affordable, and its capacity is computable — unlike the Bloom filter's.** Size
each send buffer by the balls-in-bins bound `cap = m/d + sqrt(2·(m/d)·ln d)` with `m = N/d`:

| `N` | d=4 | d=16 | d=64 | d=256 |
| --- | --- | --- | --- | --- |
| 10^6 | 0.7% | 3.8% | 18.8% | **90.1%** |
| 24×10^6 | 0.1% | 0.8% | 3.8% | 17.4% |
| 2^31 | 0.0% | 0.1% | 0.4% | **1.8%** |

Padding overhead, as a fraction of the data sent. **It shrinks as `N` grows and grows with `d`** — the
same `N/d` dependence as the imbalance, and negligible exactly where the feature matters (0.4% at
`N = 2^31, d = 64`). FESOM2-JAX measured padded against ragged at **0.742 vs 0.589 s/step** on 64 GPUs,
a 26% penalty rather than a cliff, so the fallback is a real option and not a last resort.

**The contrast with the pre-filter's `cap` is the point, and it is structural.** `NOTES.md` records that
the capacity problem "blocks the whole pre-filter family": there `cap = hits + FP` and **`hits` is the
unknown being computed**, so any data-independent bound collapses to `cap = N`, "correct and worthless."
Here the capacity depends only on `N/d`, **known at setup**, so the bound is tight and derivable in
advance. The overflow check is equally free — sum the per-destination counts and compare — and the same
rule carries over: **raise, do not clamp**, since an undersized capacity drops states silently.

So the routing is not a blocker on either path. What remains unbuilt is the JAX composition itself, and
it cannot be validated on this hardware.


### 1D XXZ is the case that decides it: prefix hashing fails at exactly `d`, and range splitting fails on the *targets* (2026-08-30)

The hash-partitioning entry above used synthetic fixtures. Re-run on a real 1D XXZ subspace — built the
way SQD gets one, a Krylov expansion from |Néel⟩ under the nearest-neighbour hop graph, so it lives in
one magnetization sector and stays *local* rather than spread over the weight shell. **Both alternatives
to whole-key hashing fail, and one fails completely.**

**Prefix hashing measures exactly `d`.** At n=30, N=200k, 12 prefix bits:

| scheme | d=4 | d=16 | d=64 | d=256 |
| --- | --- | --- | --- | --- |
| prefix-hash | **4.00×** | **16.00×** | **64.00×** | **256.00×** |
| whole-key hash | 1.00× | 1.02× | 1.04× | 1.11× |

Imbalance equal to `d` at every device count means **one shard holds the entire subspace and the other
`d-1` hold nothing** — not degradation, complete failure. The cause is **1 distinct prefix**, and it is
an artifact of the packing width rather than the physics: `B = ceil(31/8) = 4` bytes leaves **33 leading
zero bits** in the uint64 key, so the top 12 bits are constant *by construction*. At n=60 and n=100 there
are 312 and 280 distinct prefixes and imbalance is still **2.05–118.98×**, because a
magnetization-conserving Krylov subspace concentrates near Néel and even the non-pad high bits barely
vary.

**Range splitting looked competitive and is not — measuring it on the states hides the failure.**
Splitters derived from the state list balance *that list* by construction (1.00–1.85× measured), but
**what gets routed is the targets `S ^ X`**. At n=60, d=64, per XXZ hop:

| hop | hit rate | range-split | whole-key hash |
| --- | --- | --- | --- |
| (45, 46) | 18.4% | 1.01× | 1.05× |
| (29, 30) | 18.5% | 2.48× | 1.03× |
| (15, 16) | 18.3% | 7.90× | 1.04× |
| (0, 1) | 18.4% | **13.68×** | 1.05× |
| (59, 0) | 18.5% | **14.95×** | 1.04× |

The failure is **hop-dependent**: swapping *high-order* bits moves a key far in lex order, piling targets
into few range buckets, while a swap deep in the string barely moves it. **XXZ has a hop on every bond,
so the worst case is always present** in the J-fold sweep.

**And splitters go stale where a hash cannot.** A real SQD run grows the subspace during configuration
recovery, so the set the splitters were derived from is not the set later queried. Splitters from half the
subspace applied to the full set: **32.51×**, against whole-key hashing's **1.04×**. A hash of the key
does not depend on the population at all, which is the property that matters here.

**Verified exact on XXZ across every X group.** n=60, J=61 groups, K=60, N=80,000, d=64: **61 groups, 0
mismatches**, hit rates spanning **2.2%–100.0%** (the 100% group is the one where the subspace is closed
under that hop), imbalance **1.04×–1.10×** throughout, 64× fewer state rows per device. The
100%-hit-rate group matters — it is the degenerate case where every target is present, and the routing
handles it at the same imbalance.

**So the design choice is settled by physics, not preference.** On the Hamiltonian this library is most
often pointed at, prefix hashing is unusable and range splitting is unusable; whole-key hashing is
1.03–1.11× everywhere and invariant to hop, to `d`, and to the subspace growing. `poc12` now carries the
XXZ fixture and asserts exactness over all 60 hops.


### The JAX composition works and is exact: `poc13_hash_partition_jax.py` (2026-08-30)

`poc12` validated the hash-partitioning *algorithm* in numpy and left the JAX implementation as the
open item, with the caveat that `ragged_all_to_all` is `UNIMPLEMENTED` on XLA:CPU. **The composition
turned out to be fully verifiable here anyway**, because the dense `all_to_all` over fixed-capacity
buckets carries the identical dataflow — ragged is a *bandwidth* optimization of the same routing step,
not a different algorithm — and the dense form runs on CPU.

**Bit-identical to `get_xsource`** on a 1D XXZ Krylov subspace, n=30, N=21,716, hit rate 24.9%:

| D | hop | cap | exact | overflow | all-gather / all-reduce / collective-permute / all-to-all |
| --- | --- | --- | --- | --- | --- |
| 2 | (0,1) | 8826 | **yes** | 0 | 0 / 0 / 0 / 8 |
| 2 | (15,16) | 8826 | **yes** | 0 | 0 / 0 / 0 / 8 |
| 2 | (29,0) | 8826 | **yes** | 0 | 0 / 0 / 0 / 8 |
| 4 | (0,1) | 2270 | **yes** | 0 | 0 / 0 / 0 / 12 |
| 4 | (15,16) | 2270 | **yes** | 0 | 0 / 0 / 0 / 12 |
| 4 | (29,0) | 2270 | **yes** | 0 | 0 / 0 / 0 / 12 |

The hops are chosen deliberately: `(0,1)` and `(29,0)` touch the **high-order** bits, which is exactly
where range splitting measured 13.68–14.95× (`poc12`). Under hashing they are indistinguishable from the
mid-string hop. **Zero all-gather, all-reduce and collective-permute** in every case — only the intended
`all_to_all`.

**Bucketing must be `D` passes of an `[n]` cumsum, not one `[n, D]` one-hot.** `poc11` rejected the
one-hot shape for costing `4*N*NSH` bytes; measured here at D=4, **24 B/slot for the one-hot against 13
for the loop**, and the gap widens in `D` since one is `O(N·D)` and the other `O(N)`. Both give identical
buckets — verified against a numpy reference before either was used.

**The capacity guard works, and the check is sufficient.** `cap` must be static (`all_to_all` needs a
fixed shape), derived from the balls-in-bins bound `mu + sqrt(2·mu·ln D)` with `mu = (N/D)/D`. The kernel
returns a free overflow count (a sum of a mask already computed):

| slack | cap | overflow | exact |
| --- | --- | --- | --- |
| 1.6 | 2270 | 0 | **yes** |
| 0.9 | 1277 | **1284** | **no** |

**Exactness and a zero overflow count coincide**, which is what makes the free check a sufficient guard
rather than a partial one — and it is why a caller must **raise, not clamp**. Note the contrast that
mattered: the Bloom pre-filter's `cap = hits + FP` needed `hits`, the unknown being computed, so it
collapsed to `cap = N`; this capacity depends only on `N/D`, known at setup.

**Two JAX mechanics worth recording.** A rank-0 output cannot be concatenated across the mesh —
`shard_map` rejects `out_specs=P('x')` on a scalar with "which has rank 0 (and 0 < 1)", so the overflow
count must be `.reshape(1)`. And the sentinel is `0xFFFF...`, which is unreachable by construction: the
packed keys carry a zero pad bit at position 0, so no real key is all-ones.

**Still not established.** Nothing on a real interconnect — virtual devices make timings meaningless per
`CLAUDE.md`, so **no speed claim is made**. The setup phase (owner assignment plus per-shard sort) is
host-side numpy; in the library it is `poc11`'s phase 3. `uniquify_states` remains a separate blocker,
and the diagonal builders' `popcount(S[i] & z)` path is still unverified end-to-end.


### Sharding `uniquify_states`: three gaps, all closable, and the hard one is a global prefix sum (2026-08-30)

`poc11_range_partition.py` made the *sort* shardable and named two gaps in its own docstring — host-side
splitter selection, and no reassembly into the `[states_size, B]` contract. Probing those found a third,
which is the real one. **All three are closable; each mechanism is verified separately, and the full
composition is not built.**

**`uniquify_states` cannot use the hash partitioning of `poc12`/`poc13`.** Its output feeds
`get_xsource`, which binary-searches, so the result must be **globally lex-sorted**. A hash destroys
global order by design. Range partitioning is therefore mandatory here — splitters guarantee bucket
`i` < bucket `i+1`, so concatenating in order is globally sorted. **That is a real asymmetry between the
two functions**: `get_xsource` needs balance and gets it from hashing; `uniquify_states` needs order and
must accept range splitting's imbalance.

**Gap 1 — in-graph splitters: works, but they must compare the full row.** A strided sample gathered with
`.at[idx].get(out_sharding=P(None, None))` lands replicated (the explicit spec is required; JAX cannot
infer it off a partitioned axis), and sorting `d*64` rows is negligible. **But ordering the sample by its
lead word alone collapses.** On a fixture with a constant lead word and the order carried entirely by the
tail: lead-word-only puts **4000 of 4000 rows in one bucket**, full-row lex gives 990/990/990/1030.
Neither produces an *ordering violation* — lead-word splitting is sound, never wrong — it simply cannot
see the tail, which is the same balance collapse `poc11` documented for equal-range on the
most-significant word.

**An n=100 XXZ fixture does not catch this**: 94.1% of its rows share a lead word with another row, yet
enough distinct lead values remain that both schemes measured 0 violations and the balance looked fine at
1.30×. The adversarial fixture was necessary to separate them.

**Gap 2 — reassembly into `[states_size, B]`: works.** Bucket `k`'s rows land at
`sum(unique counts of buckets < k)`, and those counts are *traced*. A scatter at traced offsets is
expressible: park dead slots at index `states_size` in a `states_size + 1` buffer and drop them
(`mode='drop'`), then slice. Verified — output globally sorted, unique count exact, filler correct.
`states_size` must be `static_argnums`, which it already is in the real function. One subtlety: after
dedupe the live rows inside a block are **not contiguous** (interior duplicates are blanked), so the
within-block destination needs a *re-rank* (`cumsum(live) - 1`), not the original slot index.

**Gap 3 — the global prefix sum, which is the blocker `poc11` did not reach.** Ranking a row within its
bucket is `cumsum(bucket == k)` over the **sharded** axis, and that raises:
`ShardingTypeError: The input should be fully replicated when axis is not specified to cumsum`. `NOTES.md`
already records the rule — *"everything that reorders or compacts along the sharded axis fails; only
elementwise ops and reductions survive."* This is why `poc13` worked and a naive port here does not: there
the cumsum ran **inside `shard_map`** over each shard's own slice, whereas these ranks must be *global* to
place rows in a globally sorted output.

**Resolved by a two-level prefix sum**, the standard distributed counting-sort structure: each shard
cumsums its own slice for a local rank, one `all_gather` of a `[d, d]` per-shard-per-bucket count matrix,
then add the exclusive prefix over lower-numbered shards. **Verified against a sequential reference** —
global within-bucket ranks bit-identical, per-shard counts summing to the bucket totals, and **1
`all_gather`, no all-reduce, no collective-permute, no all-to-all**. The communicated volume is `O(d^2)`,
**independent of `N`**.

**What is not built.** These are three verified mechanisms, not a working `uniquify_states` — the
composition was not assembled, so there is no end-to-end exactness check against the real function and
**no speed claim** (virtual devices, per `CLAUDE.md`). Also unaddressed: the input `[N, B]` must divide
`d` (`sqd` already rounds `states_size` to a multiple of `mesh.size`, so the machinery exists), and
`poc11`'s capacity `slack` remains a correctness parameter needing the raise-not-clamp treatment. Note
`uniquify_states`' word packing is documented as *"the wrong trade for an out-of-core design"* at the
`2^31` ceiling (6–15 GB of extra buffer); that judgement is unchanged by anything here.


### Composing sharded `uniquify_states`: the algorithm is exact, and it needs *two* routing rounds (2026-08-30)

The three mechanisms from the probe above compose into an algorithm that is **bit-identical to
`uniquify_states`** — verified in numpy at n=100, N=209,400 with duplicates, `states_size` 262,144, d=4:
**157,051 unique rows, `np.array_equal` True**, including the `[states_size, B]` uint8 layout and its 255
filler. Bucket imbalance 1.15%.

**But the composition needs a second routing round that the probe did not anticipate, and this is the
finding.** The output is a *sharded* `[states_size, B]` array, so shard `s` owns a contiguous block of
rows. Bucket `k`'s unique rows land at a **data-dependent** global offset — the prefix sum of earlier
buckets' unique counts — which does **not** align with output-shard boundaries. Measured on this fixture:
offsets `[0, 22279, 50782, 96614]` against a shard size of 65,536, so **2 of 4 buckets straddle a
boundary**. A bucket owner must therefore send rows to more than one output shard. Zero crossings would be
luck, not a property.

So the real structure is **route → sort/dedupe → route again**:

1. splitters from a strided sample, compared on the **full row** (the probe's gap 1)
2. `all_to_all` #1: rows to their bucket's owner
3. local sort + dedupe — each owner ends up holding one sorted unique run
4. `all_gather` of the `d` unique counts, giving each bucket's global offset
5. `all_to_all` #2: rows from bucket owner to output-shard owner

**Two dead ends worth recording, because both looked right.**

*Private per-shard blocks cannot be declared replicated.* Having every shard scatter into its own
`[d, cap]` block and giving `out_specs=P(None, None, None)` is rejected: *"implies that the corresponding
output value is replicated across mesh axis 'x', but could not infer replication over any axes."*
**`check_vma` was correct and the spec was the lie** — those blocks are per-shard *partial* results, so
declaring them replicated would need a cross-shard reduction to become true. Suppressing the check with
`check_vma=False` would have produced a silently wrong answer. Routing to a single owner per bucket
removes the reduction entirely, which is why step 2 exists.

*A value derived from an explicitly-sharded array cannot be closed over.*
`NotImplementedError: Closing over inputs to shard_map where the input is sharded on 'Explicit' axes is
not implemented.` The splitters are computed outside `shard_map` from the sharded `words`, so they must be
**passed as an argument** with `P(None, None)`, not captured. The error names the workaround.

**Status: algorithm verified, JAX composition not built.** The numpy version above is exact and is the
specification; the JAX form reached the two errors named here and was not completed. What it needs beyond
the probe's three mechanisms is the second `all_to_all` plus its destination arithmetic, and a capacity
for *that* buffer as well — the second round's per-destination counts are as data-dependent as the first's,
so `poc13`'s raise-not-clamp treatment applies twice. **No speed claim**: numpy, single process, and two
routing rounds is materially more communication than the one round this line assumed. `poc11`'s note that
the word packing is *"the wrong trade for an out-of-core design"* at the `2^31` ceiling still stands and is
unaffected.


### The two-round JAX composition works, and the capacity model for a range partition is *not* balls-in-bins (2026-08-30)

The composition above was specified in numpy and left unbuilt in JAX. Built now: **bit-identical to
`uniquify_states`** at n=100, N=209,400 with duplicates, `states_size` 262,144, at **D=2 and D=4** —
157,051 unique rows, `np.array_equal` True, including the `[states_size, B]` uint8 output with 255 filler.

**The structure that makes the output spec honest.** Round 2 exists so that after it, shard `s` holds
exactly output rows `[s·SS/d, (s+1)·SS/d)` — which makes `out_specs=P('x', None)` **true**. That is what
the earlier dead end was about: private per-shard `[d, cap]` blocks declared `P(None,None,None)` are a
lie, and `check_vma` rejects them. Routing to a single owner makes the declaration honest instead of
suppressing the check.

**Collectives, D=4:** 1 `all-gather`, 24 `all-to-all`, 4 `all-reduce`, 0 `collective-permute`.

The `all-reduce` needed explaining rather than accepting. Two of the four are the caller's own `.sum()`
over `P('x')` results — summing a sharded array across shards *is* a reduction. The rest is
`u64[256,2]`: the **splitter sample**. `words.at[idx].get(out_sharding=P(None, None))` gathers a
replicated sample from a partitioned array, and XLA implements that replication as an all-reduce. It is
4 KB and `O(nsample)`, **independent of `N`** — so the earlier claim that splitter selection is *free* was
wrong; it is *cheap*, which is a different statement.

**Two capacity defects found, and the second corrects a claim in the entries above.**

*The guard fired on correct input.* The first working version reported **overflow 763,677 beside a
bit-exact result**, because round 2 funnels every dead row to one bucket, which overflows by design and is
then dropped harmlessly. The count conflated discarded padding with lost data. Fixed by masking the count
to **live** elements. **A guard that fires on correct input is worse than none** — it trains a caller to
ignore the one signal that matters, which is how `poc11`'s `cap` bug shipped.

*Balls-in-bins is the wrong model here.* `poc13` sizes its capacity as `mu + sqrt(2·mu·ln d)`, which is
correct **for a hash**: each element picks its destination independently at random. **A range partition
violates that assumption** — the destination is the element's *value* bucket, and bucket sizes are set by
the data distribution. Measured: buckets `[44557, 57006, 47400, 60437]`, so a shard sends `60437/d ≈
15,109` rows to the largest bucket's owner, not the `N/d/d = 13,088` the bound predicts. A 1.2× slack over
that mean still overflowed by 42,712. **`poc11` already had the right rule** — *"slack must exceed the
splitter imbalance; 1.35 was sufficient in every fixture here"* — and 1.35 is exact here at both D.

Round 2 needs a different baseline again: a bucket's unique rows land **contiguously**, so they reach only
**1–2 output shards**, not all `d`. Its worst per-destination is therefore ~the whole bucket, so size off
`ss/d`, not `ss/d/d` — understating it by a factor of `d`.

**The guard is verified in both directions.** Undersizing either round makes overflow nonzero *and*
`exact` False; at 1.35 both are clean. Exactness and a zero count coincide, which is what makes the free
check sufficient — and a caller must **raise**, not clamp.

**Still not established.** Virtual devices, so **no speed claim**, and two routing rounds with 24
`all-to-all` at D=4 is materially more communication than the single round this line originally assumed —
whether it pays for the 27.9 GB/device it saves needs real interconnect measurement. The prototype lives
outside `rqutils/`; `states_size` and both capacities are `static_argnums`, and `N` must divide `d`.


### Widening the POC device sweep found a latent `all_to_all` bug a 4-device box hid (2026-08-30)

Reorganizing `poc13`/`poc14` to run on real GPUs replaced their hardcoded `XLA_FLAGS=...count=4` with
`poc8`'s `--devices` convention (argparse **before** `import jax`, since `CUDA_VISIBLE_DEVICES` and
`XLA_FLAGS` are both read at backend initialization) and derived the shard sweep from
`jax.device_count()` rather than pinning `(2, 4)`. **That immediately failed at 8 devices**, and the
cause was a real defect rather than a limitation:

`ValueError: The size of all_to_all split_axis (4) has to be divisible by the size of the named axis
x (8)`. `run_case` took `mesh` and `num_shards` as **independent parameters that had to agree**. The
send buffer got `num_shards` rows while the mesh axis had `jax.device_count()` devices, so routing is
only well-defined when the two are equal — and **a 4-device box makes them coincide**, which is why
every earlier run passed. Fixed by deriving `num_shards = mesh.shape["x"]` inside `run_case`, so the
pair cannot disagree.

**The transferable point: a hardcoded device count is not a fixture, it is a coincidence.** Sweeping
the parameter is what separated two quantities that had been silently identical — the same reason
`CLAUDE.md` says to sweep `cache_level` rather than sample it, and the same shape as the three bugs
that hid behind its default `(1, 0)`.

Also: `d = 1` needed handling in both scripts, for *different* reasons. `poc13` runs it as a
meaningful degenerate case (zero collectives, still exact) but must **skip its overflow section** —
with one bucket the derived capacity is ~`N`, so a 0.9x slack is still ample, nothing overflows, and
the section's premise is false rather than its assertion weak. `poc14` **raises** at `d = 1`: with one
bucket there is no second routing round, which is the mechanism it exists to verify.

### The popcount diagonal path already shards — verified, no work needed (2026-08-30)

Carried as the last open item of the distributed-`states` line ("the diagonal builders need
`popcount(S[i] & z)`, i.e. the state *bits* — those shard elementwise, but that was not verified
end-to-end"). **Verified now, and there is nothing to build:** every function on the path shards with
**zero collectives** and returns bit-identical values.

**Why it was never at risk, which is the point worth keeping.** `_z_parity` is
`sum(bitwise_count(states & z), axis=1) & 1`: two elementwise ops and a reduction along the **byte**
axis. For a `P('x', None)` state array the sharded axis is axis 0, so the reduction runs entirely within
each device's own rows. `NOTES.md`'s rule — *"everything that reorders or compacts along the sharded axis
fails; only elementwise ops and reductions survive"* — is satisfied in its easiest form: **the reduction
is over the unsharded axis.** Contrast `uniquify_states`, whose `cumsum` reduces *along* the sharded axis
and needed a two-level prefix sum.

Measured on 1D XXZ, `P('x', None)` states, 4 devices — asserting the **spec and the values together**,
since `NOTES.md` records that a replicated run agrees to exactly 0.0 and "correct but silently unsharded"
is invisible to value comparison alone:

| function | out spec | values | max diff |
| --- | --- | --- | --- |
| `_z_parity` | `P('x',)` | match | 0.00e+00 |
| `get_diag_signs` | `P('x', None)` | match | 0.00e+00 |
| `get_diagonal` | `P('x',)` | match | 0.00e+00 |
| `compute_diagonal` | `P('x',)` | match | 0.00e+00 |

**Swept over every X group and both coefficient dtypes**, at n=40 (J=41, K=40, N=9,220 for real; J=42 for
the complex case, where a single odd-Y string makes `.c` complex128): **0 failures out of 41 and 42
groups** for all three builders, counting a missing `'x'` in the output spec as a failure alongside a
value mismatch. And **`apply_h` end-to-end at both cache levels that use this path** — `(1, 0)` which
recomputes from `zsignatures`, and `(1, 1)` which unpacks cached `diag_signs` — output spec `P('x',)`
and `max|diff| = 0.00e+00` on both dtypes.

**One pre-existing behaviour found and correctly attributed.** `apply_h` with a **real** `vec` against
**complex128** coefficients raises `TypeError: scan body function carry input and carry output must have
equal types ... float64[N] but ... complex128[N]`. That looked like a sharding failure in the first run
and is not: it **reproduces on a single device with no mesh at all**. It is an undocumented dtype
contract — `vec` must be promotable to the coefficient dtype — and `PauliSumXZ` makes `.c` complex
whenever any Pauli string has an odd Y count. Worth knowing, unrelated to this work, and *not* fixed
here.

**Pinned by `tests/_sharded_diagonals.py` and mutation-verified.** Dropping the single
`out_sharding=jax.typeof(states).sharding` on `get_diag_signs`' `init` accumulator makes the whole
builder run **correctly but unsharded**: `bad_value = 0` — every value still bit-identical — while
`bad_spec` goes to 42/44. **A value-only test passes that mutant silently**, which is the concrete
demonstration of `CLAUDE.md`'s rule rather than a restatement of it. Placing the mutation *inside* the
scan instead raises on a carry-type mismatch, so the `init` line is the one that had to be mutated to
produce the silent form.

**So the distributed-`states` line has no remaining unverified mechanism.** `get_xsource` (`poc13`),
`uniquify_states` (`poc14`) and the diagonal path all have working, exact, sharded forms. What remains is
entirely the open question stated in those entries: whether the routing communication pays for the
27.9 GB/device it removes, which needs a real interconnect and cannot be answered on virtual devices.

### The range-partitioned shuffle works — `poc11_range_partition.py` (2026-08-29)

That shuffle was then built. **It is the one design that removes the single-device sort**, and it is
correct: output bit-identical to `np.unique(rows, axis=0)`, and **zero `all-gather` / `all-reduce` /
`collective-permute`** in the compiled HLO. Sample sort with data-derived splitters, fixed-capacity
buckets, then `NSH` independent `lax.sort`s under `vmap`. Concatenating buckets in splitter order is
globally sorted, so no cross-bucket comparison ever happens.

Four findings, each of which cost a wrong turn:

- **Equal-range splitting collapses.** Splitting the packed most-significant word's nominal `2^64`
  range gives **4.00x/8.00x imbalance at NSH=4/8** — i.e. every row in one bucket. At n=100 that word
  is 7 bytes of leading pad (a consequence of `_pack_state_words`' leading-pad choice), so its range
  carries almost no information. Data-derived quantiles give **1.07–1.19x** on both uniform and
  fixed-Hamming-weight fixtures.
- **A global `argsort` for within-bucket rank defeats the whole design.** The obvious way to compute
  "position among earlier rows in my bucket" is `argsort(bucket)` — which is a global sort, the exact
  thing being removed. Measured **256 ms against 8 ms** at N=2M, and it showed up as 208 `sort` ops in
  the HLO against the incumbent's 29.
- **The `[N, NSH]` one-hot cumsum is the wrong fix.** Same speed as the alternative but `4*N*NSH`
  bytes — **34 GB at `N = 2^31, NSH = 4`, 275 GB at NSH=32**. It grows *with* the device count, so
  adding devices to reach larger `N` makes it worse. Use `NSH` sequential `[N]` cumsums instead: `O(N)`
  memory, same time, identical result.
- **Capacity overflow is detectable, which is what makes this shippable.** Static shapes force a
  per-shard capacity and an undersized one drops rows (16,090 lost at slack 1.05). But the kernel
  *returns the overflow count*, so a caller can raise. Contrast the rank-select prototype
  (`docs/rqutils-multiobs-response.md` §5.3), whose analogous `cap` had no detectable failure mode —
  that is why one is a candidate and the other is not.

Timings are 1.70–2.32x against the incumbent on 4 virtual CPU devices, reported only to show the
structure is not pathologically slow; **virtual-device timings are meaningless** and the claim here is
shardability, not speed. Two pieces of real work remain before it could replace `uniquify_states`:
splitter selection is host-side numpy (one device sees the sample), and reassembly into the
`[states_size, B]` contract is not implemented — the POC returns `[NSH, cap, NW]` blocks.

### A rounding-floor residual is not zero, and `== 0.0` is the wrong guard

2026-08-28, from `docs/rqutils-prefilter-dim2-request.md`. `body_iter1` formed its search direction as a
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

From `docs/rqutils-tol-request.md` (the `spinchain` side asked for a `tol` that means the eigen-residual,
so their solver criterion and their `_RESIDUAL_TOLERANCE = 1e-6` guard would be one number). Shipped;
reply in `docs/rqutils-tol-response.md`.

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

```
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
- **A third, unlooked-for confirmation.** `poc7_sharding` independently reports `‖Hv−ev‖/‖H‖` of 5.5e-16
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

From `docs/rqutils-atol-rtol-request.md`; reply in `docs/rqutils-atol-rtol-response.md`. `tol` is gone,
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
It also *fails* `tests/_sharded_prefilter.py`: compensation needs sequential accumulation over blocks, so
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
reintroduced preconditioner must not touch it. The comment there says so. `docs/deflation-preconditioner.md`
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
ratio `docs/locg-chebyshev-prefilter.md` names as the genuinely uncertain quantity, and the end-to-end
figure lands **below** the 1.88x that doc measured on dense `ground_locg` — which is why `sqd`'s
docstring tells callers to A/B on their own subspaces rather than trusting the published figure. Every
arm was correct to <1e-9 against `eigsh(tol=0)`.

**A measurement trap hit twice here.** `tests/conftest.py`'s `project_dense` builds the *full* `2^n`
operator before slicing, so it cannot reach n=16 (68 GB) — an attempt to reproduce the 1.79x figure with
it died with no useful error. Use `hproj` for anything past n≈12. Separately, a first pass on small
dense chains measured 1.05x median *with a regression* and looked like a refutation of the shipped
`precond`; the documented instances are **sampled subspaces at n=16-18, dim 2000-4000**, where the same
code reproduces 2.76x median with 0 regressions. The fixture family, not the code, was wrong.

## Preconditioners and subspace selection: a closed investigation

Full record in `docs/rqutils-precond-request.md` and `docs/sdp-lower-bound.md`. Summarized here
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
