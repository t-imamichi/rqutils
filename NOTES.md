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
