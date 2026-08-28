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
