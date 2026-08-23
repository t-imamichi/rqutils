# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment & commands

Always use `uv run python` (not bare `python`) — the venv at `.venv` is managed by uv. No `timeout` on
macOS (it is GNU coreutils) — use the Bash tool's own timeout rather than wrapping a command in it.
The shell is fish: **quote grep globs** (`--include="*.py"`). Unquoted, fish fails with
`(eval):1: no matches found` before grep runs — which looks like "no results", not "no command".

**Check `git rev-list --left-right --count origin/<branch>...HEAD` before amending.** Work happens on
feature branches (`metal`, not `main`) that get pushed mid-session, so "my commits are still local"
goes stale within a turn — amend then and you rewrite published history. `git branch -r --contains
<sha>` is the per-commit check.

```bash
uv run python -c "import rqutils; print(rqutils.__version__)"

# Optional deps are extras, not installed by default. Pull them in per-invocation:
uv run --extra qiskit python examples/sqd.py 8 --num-paulis 10 --subspace-frac 0.5
uv run --extra qiskit python examples/svsim.py 12 --out /tmp/out.h5
uv run --extra mpl   python ...   # matplotlib, for qprint(output='mpl')
uv run --extra qutip python ...   # qutip Qobj input to qprint
```

Extras: `mpl`, `qutip`, `qiskit`, `docs`, `dev` (pytest + ruff + ty). `mpi4py` is imported by `examples/` but declared nowhere — install it manually if you need the multi-process path.

`examples/` holds the three library demos (`sqd.py`, `svsim.py`, `bench.py`) plus the scaling POCs
under `examples/scaling/` (findings in `docs/scaling-pocs.md`). The POCs reach `_scaling_common.py`
through `sys.path.insert(0, dirname(__file__))`, so they are run as scripts, not imported.

The MLX port that used to live under `examples/mlx/` (8 scripts, ~2.7k lines) has been **deleted** —
see the Architecture section for what it left behind.

Docs (regenerates `docs/source/apidoc/` via `sphinx-apidoc`, which is **not** committed):

```bash
cd docs && uv run --extra docs make html    # output in docs/build/html
cd docs && uv run --extra docs make clean   # also removes source/apidoc
```

## Linting and type checking

```bash
uv run --extra dev ruff check rqutils/ tests/ examples/     # lint
uv run --extra dev ruff format rqutils/ tests/ examples/    # format (line width 100)
uv run --extra dev ty check rqutils/ tests/ examples/       # type check
```

All three are clean; keep them that way. Config lives in `[tool.ruff]` / `[tool.ty.rules]` in
`pyproject.toml`, and every suppression there carries the reason it exists — read those comments
before adding another. Notebooks are excluded from both tools: they get their names from IPython
magics (`%aimport rqutils.paulis`) that static analysis cannot see.

Several `[tool.ty.rules]` categories are set to `ignore` because JAX ships almost no type
information and numpy's stubs are stricter than this library's dtype-generic `npmod` convention
allows. Those were triaged individually rather than blanket-disabled — prefer fixing a real finding
in the code over widening the ignore list. Pre-commit runs only whitespace/EOF/YAML/large-file
hooks; it does not run ruff or ty.

New scripts under `examples/` trip rules the existing tree does not: **B023** (a `lambda` in a `for`
loop capturing the loop variable — endemic to benchmark harnesses passing thunks to a timer; fix by
binding as a default arg, `lambda vec=vec: ...`, not by restructuring the loop) and **E402** (imports
after the mandatory `jax.config.update('jax_enable_x64', True)`, which needs `# noqa: E402`).
`ruff --fix` resolves neither.

## Testing

```bash
uv run --extra dev pytest              # whole suite
uv run --extra dev pytest -v -x        # verbose, stop at first failure
```

One `tests/test_<module>.py` per module. All seven are covered — `ground_locg`, `sqd`, `svsim`,
`paulis/general`, `paulis/symplectic`, `qprint`, `math`. Most of it is single-device; the multi-device
paths are covered by two subprocess tests plus a POC, since virtual devices must be requested before
jax initializes (see below). `sqd`'s mesh-size padding and, through it, `ground_locg`'s `out_sharding`
contract are covered by `test_sqd.py::TestShardedCacheLevels` and, more thoroughly, by
`examples/scaling/poc7_sharding.py` — run the POC after any change to `ground_locg`'s reductions or
helper signatures, not just after touching `sqd`. **`svsim`'s `out_sharding` is now covered** by
`test_svsim.py::TestShardedOutput` (subprocessing `tests/_sharded_svsim.py`), which was the last
untested sharding contract — checked because the same axis in `sqd` hid three defects. `svsim` had
none: it takes `out_sharding` as an explicit parameter and threads it through every array-creating op,
rather than resharding conditionally partway through as `run_sqd` does. Its one limit is documented
rather than fixed: **`mesh.size` must divide `2^num_qubits`**, so a 3- or 6-device mesh fails at
*every* qubit count, not just small ones. A state vector cannot be padded the way `sqd`'s state list
can — its indices *are* the basis states — so there is nothing to pad and the jax raise (which names
both shapes) stands. `PartitionSpec(None)` replicates instead.

**Assert the sharding *spec*, not just the values.** An explicitly replicated `svsim` run agrees with
the single-device answer to exactly 0.0, so "correct but silently unsharded" is invisible to any value
comparison — the regression a dropped `out_sharding` would actually cause. `TestShardedOutput` asserts
both.

`tests/` also
contains three Jupyter notebooks used as interactive scratchpads; pytest does not collect them.

`tests/_sharded_cache_levels.py` is a script, not a test module (the leading underscore keeps it
uncollected): `test_sqd.py::TestShardedCacheLevels` subprocesses it under
`XLA_FLAGS=--xla_force_host_platform_device_count=4`, because the virtual device count has to be set
before jax initializes and `conftest.py` has already imported it by collection time. It lives in a
file rather than an inline string so ruff and ty check it — as a `textwrap.dedent` blob an
`ImportError` there would surface as a nonzero exit, indistinguishable from the sharding regression
it exists to catch.

`tests/conftest.py` enables `jax_enable_x64` before any `rqutils` import — every tolerance in the
suite depends on it — and holds the shared reference helpers (dense Pauli sums, projections, gate
unitaries), each validated against qiskit before being trusted as a reference.

**The suite runs in ~6 s, not ~53 s — via caches `conftest.py` configures.** It sets `MPLCONFIGDIR`
and `JAX_COMPILATION_CACHE_DIR` (plus `JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0`, required
because the default 1 s threshold excludes every kernel here — the largest single compile is ~0.44 s)
before importing jax, with `os.environ.setdefault` so a value you exported always wins. Both are
speed-only, unlike the x64 flag: nothing depends on them, and a warm cache was verified *unable* to
mask a defect — reverting `_is_filler`'s `>> 7` to `>> 8` still fails the one test that catches it,
since XLA keys on the computation. Expect ~53 s on a first run while the caches populate.
**Don't chase test-body slowness**: measured, 72% of an uncached run is XLA compilation plus a
matplotlib font-cache rebuild (`~/.matplotlib` is unwritable in a sandbox, and matplotlib's own
fallback is a fresh temp dir per process, which never warms). The `range(2000)` accumulation loops in
`test_ground_locg.py` that look expensive total 0.80 s, 2.3% of the suite — leave them alone.

**Fixtures are built inside each test body, and there are no `@pytest.fixture` state generators.**
This is deliberate: several tests pick a seed to produce a specific pathology (a decoupled seed
state, a subspace that splits into two blocks, 13 Z terms in one X group) and assert the fixture
still has it. Moving draws into fixtures would make RNG stream position depend on fixture ordering,
which is invisible at the call site — `conftest.collapsing_states`' docstring records a measured
instance (changing a preceding `real_pauli_strings` count from 5 to 6 moved a collapse from 7 uniques
to 9). Keep new generators as plain functions taking `rng`; `unique_states`/`collapsing_states` are
the pattern. Note only `collapsing_states` asserts its own precondition — `np.unique` makes
`unique_states` distinct by construction, but its *row count* varies with the seed (7 draws over 4
qubits measured 3–7 distinct rows across 200 seeds), so a caller needing a floor must assert it.

**Tests are organized by defect.** Writing these suites found bugs in five of the seven modules, all
of the same character: a plausible finite answer rather than a raise or a `NaN`. So when adding a
test, name the defect it locks down and record the measured wrong value, and prefer an *independent*
reference (a dense construction, scipy, qiskit) over self-consistency — several bugs made every
internal code path agree on the same wrong number.

**A concrete instance of that trap, worth reading before writing a padding/shape test.**
`test_states_size_padding_is_shape_invariant_only` compares padded `sqd` calls against an
*unpadded* `sqd` call, and cannot catch a broken filler mask: its fixture is 12 random 4-bit states,
which collapse to fewer uniques, so even the "baseline" arm already carries filler slots and is corrupted
identically. Both sides drift together and it passes. Measured — changing `_is_filler`'s
`states_u[:, 0] >> 7` to `>> 8` (a uint8 shifted by 8 is 0, so every filler reads as a genuine state)
left the whole sqd suite green while `sqd` returned −1.2 against a true −0.8297058541. Same for deleting
`run_sqd`'s filler-diagonal masking. `test_filler_slots_are_excluded_against_a_dense_reference` closes
both, using a fixture that is *already unique* (so `states_size=None` is a genuinely filler-free
control) and a dense reference. **A "does X change the answer?" test needs an arm where X is truly
absent — verify that, don't assume it from the parameter being unset.** Verify a new test actually fails against the bug
it targets by reverting the fix in place; a copy of the repo does not work, since the venv holds an
editable install pointing at the original.

**Mutation-testing recipe**, since "revert the fix in place" is easier said than done and this is the
highest-yield tool in the repo — it found two silent coverage gaps and a false docstring claim in one
session. Copy the file (`cp rqutils/sqd.py /tmp/x.bak`), rewrite one line with a Python one-liner that
**asserts the anchor string exists** before substituting, run the suite, restore from the copy, and
`diff -q` to prove the restore took. The assert is not optional: a silent no-match reports a false
"no coverage", which is indistinguishable from the finding you are looking for. Two further traps.
For a `@jax.jit`-decorated function, mutate in a **fresh subprocess** — patching in a live session
reuses the compiled kernel and both arms return bit-identical numbers (`test_ground_locg.py`'s
`TestBasisOrthogonality` records this). And check the mutant is *reachable* before concluding a guard
is untested: some survive because the fixture never exercises them, which is a fixture finding, not a
missing test.

**A green suite after reverting a fix means the test is missing, not that the guard is dead.** Some
guards are only reachable when *other* defects compound with them, so the end-to-end assertion
(theta matches `eigvalsh`) stays green while the invariant the guard protects is already destroyed.
`ground_locg`'s `_reorthogonalize` (audit item I5) is the worked example, and it is a cautionary one:
it was twice recorded as unpinnable, first from a compromised A/B (patched in a live session, so both
`@jax.jit` arms reused one compiled kernel and returned bit-identical numbers) and then from a
correct A/B whose conclusion was drawn from the wrong assertion — theta *does* survive, because the
1.0 drift needed I4's 2000-iteration runs. The test that discriminates asserts the **invariant**
(`|<x|y>|`, straight off the `debug=True` diagnostics) rather than the end result, and fails 3 of 4
arms. So before recording a negative result, check whether a *more direct* assertion exists; reach
for the docstring note only once it does not.

**Multi-device paths are testable on CPU — use it.**
`XLA_FLAGS=--xla_force_host_platform_device_count=4` gives virtual devices that exercise every
sharding code path (mesh detection, `PartitionSpec` propagation, `jax.reshard`, `sqd`'s mesh-size
padding) with no GPU. This is not hypothetical: the first run found `sqd` raising `ShardingTypeError`
on *any* mesh, because one scatter omitted `out_sharding` while every neighbouring op passed it.
`examples/scaling/poc7_sharding.py` is the harness — sharded against single-device against a dense
`eigvalsh` reference, plus every residue of `N mod mesh.size` **and all six `cache_level`s**.

**Sweep `cache_level`, don't sample it.** Three bugs hid behind a single-cell check, and each was
masked by the one before it — every existing sharding check ran only `sqd`'s default `(1, 0)`:
`_accumulate_diagonal`'s rank-2 spec on a rank-1 accumulator (failed all six); then, once fixed,
`_spread_seed`'s `jnp.where` mixing a replicated predicate with a partitioned `vec`, because
`run_sqd` reshards `states_u` only inside `if cache_level[0] == 1` (failed `(0, *)`); then, once the
sweep reached a *complex* fixture, `vinit_from_min_diag` using `diagonals[0]` raw where the uncached
branch took `.real` (failed `(*, 2)` on any odd-Y Hamiltonian, **single-device, no mesh**). Fixing
each one only revealed the next, so "the mesh test passes" meant very little until the grid was
complete. Note the last needed the *fixture* varied, not the parameter: the suite's
`real_pauli_strings` keeps the Y count even, so `.c` stays float64 and a six-cell sweep still
reported six passes. Timings under virtual devices are
meaningless (they share one CPU), so use this for correctness only.

## Architecture

Seven largely independent modules under `rqutils/`; nothing but `sqd.py → {paulis/symplectic.py, ground_locg.py}` and `qprint.py → paulis/general.py` couples them.

**Two unrelated Pauli representations — do not confuse them:**

- `paulis/general.py` — dense generalized (Gell-Mann-like) basis for arbitrary dimension. The whole public surface is `paulis(dim)`, `pauli_matrices(dim)` (the single-subsystem primitive `paulis` builds on), `components()`, and `labels()`; `compose`, `truncate`, `l0_projector`, `symmetry` and `paulis_shape` were removed as dead code, so don't reintroduce a caller expecting them. Normalization is `tr(λ_k λ_l) = 2δ_kl`, so **`λ_0 = sqrt(2/n)·I`, not `I`** — the most bug-prone invariant here. Basis-index ordering is fixed by a shell-by-shell construction loop; `components` and `labels` both index by basis position, so a reordering would disagree with the labels users read while every function stayed self-consistent. Shapes: `paulis(dim)` → `(d1², …, D, D)` (basis axes *first*); component arrays → `(…, d1², …)` (component axes *last*). Everything is memoized in module-level dicts keyed by a `tuple(int)`-normalized `dim` (`_normalize_dim`).
- `paulis/symplectic.py` — `PauliSumXZ`, a bit-packed qubit-only form for JAX/GPU. Convention `Q = (-i)^{x·z} Z^z X^x` with **little-endian qubit ordering** (Qiskit's `.x`/`.z` get reversed on ingest). Terms are grouped by unique X signature, Z groups zero-padded to a rectangle, the `(-i)^{popcount(x&z)}` phase folded into the coefficients, then `np.packbits`. Signatures always reserve **one pad bit at position 0** (a dummy identity, aligning with the pad bit consumers put in their state bitstrings) — not optional, so alignment cannot be got wrong. Note that `packbits` fills each byte from the **most significant** end, so a signature's payload bits are entries `1` through `num_qubits` of `np.unpackbits`, in string-character order (leftmost character = index 1) — the reverse of the qubit numbering. Anything decoding a packed signature back to an integer must shift by `8*nbytes - (num_qubits + 1)`, counting the pad bit; dropping the `+1` silently returns a *permutation* of the right answer (measured 2.07 max abs error in the since-removed `matmul`, which is why that method is gone). The class is now just `from_paulisum` and the `arrays` property. **A complex coefficient raises** (non-Hermitian; both ingest branches share one check) — there is no `force_real` flag any more, because none could work: `.c` narrows to float64 exactly when the folded phase is real, i.e. when every string has an even Y count, and an odd-Y string is complex128 *by construction*. Check `.c.dtype` if you need float64 — a backend with no complex128, say. (The one in-tree caller that did was the deleted MLX bench harness, so nothing exercises that path now.)

**`sqd.py`** — sample-based quantum diagonalization: project a large Pauli-sum Hamiltonian onto the subspace spanned by a list of computational-basis bitstrings and solve matrix-free. `sqd(...)` is the entry point; `hproj(...)` is the dense/debug path. Two conventions dominate:

- States carry **one extra zero pad bit at position 0** before `packbits`. `PauliSumXZ` reserves the same bit in its signatures unconditionally, so the two are aligned by construction — this used to be an opt-in `add_padding` flag, and the two sides disagreeing is how `hproj` shipped broken. Filler slots produced by uniquification are `255`, detected via `states_u[:, 0] >> 7`.
- `cache_level=(source_indices, diagonals)` selects among six matvec strategies. Only the *diagonal* axis is a genuine memory-for-speed trade; the source-index axis is near-free to enable and very expensive to disable (see below). They are one kernel indexed by that 2×3 grid, reached two ways. The **public `apply_h` is keyword-only**: name the arrays you have (`xsources=`/`xsignatures=` and one of `diagonals=`/`diag_signs=`/`zsignatures=`, plus `coeffs=` for the two that compute a diagonal) and the strategy follows, so the six valid input sets are the only constructible ones. There is **no positional `(scanned, cache_level)` form** — it was deleted as a breaking change, because `cache_level` selected positionally how `scanned`'s members were read and nothing could check the two agreed (measured 0.44 max abs error from one mispairing; both are integer arrays, and at `n = 15` with 2 states even their *shapes* collide at `(2, 2)`, so no assertion could have closed it). Internally `run_sqd` calls the private `_apply_h_kernel` with an assembled tuple and `cache_level` bound via `functools.partial` — it **must** stay static there, since `ground_locg` splats `args` positionally and `static_argnames` would never see it. `states_size` exists solely to pin array shapes and prevent JIT recompilation.

Two measured facts about the cost, from the six scaling POCs under `examples/scaling/` (findings in
`docs/scaling-pocs.md`). **`get_xsource` setup dominates** — weighted by call count it is 66–97% of a
solve (3.1 s against 79 ms of matvec loop at 10 iterations, N=200k, J=50), so the `2N` sort is not
merely the `N ≤ 2^31` ceiling but the main cost at every size measured, while `matvec/J` is flat at
~0.16 ms and confirms the `O(J·N)` model. The module docstring's caching section now carries this
magnitude and the resulting advice (prefer `cache_level[0] = 1`), rather than byte counts alone —
independently reproduced end-to-end at N=3k, n=12, J=23: `(0, 2)` is 10.9× slower than `(1, 2)` and
`(0, 0)` is 7.2× slower than `(1, 0)`, all four levels returning the same energy.

Because that change landed, **the POCs under `examples/scaling/` no longer have a baseline in the
library, and must not be pointed at one.** `poc1.xsource_sort_legacy` is a verbatim copy of the
pre-23fb226 sort and is the timing baseline for both POC 1 and POC 8; their *correctness* arms still
compare against `get_xsource`, which is the point (agreement with what ships is now a regression test).
Point a timing arm at the library and you get searchsorted-versus-searchsorted: the first GPU run of
`poc8_gpu_unverified.py` reported 1.002×/1.000×/1.000×, POC 1e read 0.26× "SLOWER", and `fmt_ratio`
was correct every time — which is what made it easy to misread as a GPU finding. Restoring the
baseline recovers 12.1×/18.3× and 3.57×/3.18× for the lex variant. Two related traps in that script,
both fixed: `peak_bytes_in_use` is a high-water mark that never decreases (and sampling `bytes_in_use`
after `del` reads the post-free baseline), so a leak test built on either cannot observe anything; and
`--devices` sets `CUDA_VISIBLE_DEVICES`, a *filter* over what the driver exposes, so it cannot create a
second GPU — Claim 3 on a one-GPU box is unrun, not unresolved.

**On GPU the speedup is 5.15×, not 12–25×** (NVIDIA GH200, N=64M single signature, `alpha` 1.09 vs
0.92, so still rising with N). Two other GPU numbers from the same run are launch-bound artifacts and
must not be quoted: POC 1c at J=50 reads 12.5–14× — deceptively close to the CPU figure — with a sort
arm *flat* at 1141/1239/1201 ms across a 5× N range, and POC 1b below N=1M reads 2.56×. The
launch-bound regime covers J=1 past N=1M *and* J=50 at N=500k, so it is per-call latency × call count,
not N alone; `--sweep-to` exists to escape it and `check_scaling` fits `alpha`, refusing to call a
ratio quotable below 0.6. **Also measured: the `lax.sort` GPU memory leak does not reproduce** (~0.95 GB
of transients fully reclaimed every rep at N=5M/B=4), so that note was stale — and since the sort left
the library, it is now a claim about `lax.sort` rather than about `sqd`. Multi-GPU speed remains
**unrun**, needing a physically multi-GPU box.

**`get_xsource` is now a binary search, not a sort** (12–19× faster on the J-fold precompute on CPU),
which is why **`states` must be lex-sorted** — always required, since the sort was equally wrong on
unsorted input, but previously undocumented. `hproj(unique_states=True)` skips its `np.unique` and so can
violate it; that returns a non-symmetric matrix and is pinned by
`TestHproj::test_unsorted_input_with_unique_states_is_wrong`. Two paths selected statically on width:
`uint64` keys for `B ≤ 8` bytes, an explicit lexicographic search beyond. That boundary is a
**correctness** limit — a `uint64` key silently truncates a wider row and aliases distinct states —
so if you touch it, note that a test only catches the overrun when the subspace's *leading* bytes
collide and partners genuinely exist; `packbits` puts low qubit indices in the *leading* bytes, the
reverse of the qubit numbering, and getting that backwards makes the test pass vacuously.

**`ground_locg.py`** — single-vector (block-size-1) LOBPCG specialization used as `sqd`'s eigensolver, with the Rayleigh–Ritz step solved analytically (`eigenpair_2x2`, `eigenpair_3x3` via Cardano) instead of via `eigh`, to keep memory down for huge vectors. It is sharding-transparent **only if the `mat` callable preserves output sharding** — that contract is why every `apply_*` in `sqd.py` passes `out_sharding=jax.typeof(vec).sharding`. Every guard in it is load-bearing and was measured: `docs/locg.md` catalogues seven defects (I1–I7) that each failed *silently*, returning a plausible wrong number rather than raising. Don't "simplify" the balancing, the re-orthogonalizations, or the zero-direction masks. **`docs/locg.md` is stale** — it audits the pre-rewrite module, so its line numbers, its "no pytest suite exists" scope note, and its A1–A5 gaps (all since fixed) don't apply; cite it for the I-numbers and the measurements only, and read the module docstring for what currently holds. One severity is partly retracted there: I5's full 1.0 collapse needed I4's 2000-iteration runs to develop, so the *eigenvalue* stays correct without the guard — but the guard is pinned, by `TestBasisOrthogonality`, which asserts `|<x|y>|` off the `debug=True` diagnostics and fails 3 of 4 arms when it is neutered. Two traps when editing it. `body_iter1`'s exclusion bound (`2|rho| + 1`) is **not** `body()`'s (`max(diag) + sum(|diag|) + 1`) specialized to one entry — the general form collapses to a constant `1.0` for the negative `rho` of a ground-state search. Both are *valid* (each strictly exceeds the one retained entry, so the eigensolver still can't pick the excluded slot), but the unified form's margin stops tracking the operator scale, which is a poor trade in a routine whose other guards exist because large shifts destroy precision. Don't unify without redoing the bound argument. And the one-matmul `_compute_sas` form (from the since-deleted MLX port) does **not** belong here: measured 98.7 ms against the scatter form's 27.7 ms at N=16.8M, because stacking three huge vectors is two 402 MB temporaries per iteration — the copy this module exists to avoid. Don't reintroduce it.

**The MLX port is gone** — deleted, not deprecated-in-place. The JAX solver measured faster even on
the MLX GPU backend, so it had no performance case, and nothing in the tree imported or ran it. Don't
reintroduce a second solver implementation without that measurement going the other way first. Two
things it left behind that still apply:

- `docs/mlx-metal-kernels.md` is the historical record of the fused-Metal-kernel work — three kernels
  written, **one measured slower and deleted, two verified negative** — kept so nobody re-derives
  them from scratch. Read it before attempting anything in that direction. It is a record, not a
  guide: every claim about what the port *offered* is superseded, and any revived kernel needs its
  static MSL guards revived too, since a numpy shim never compiles MSL text.
- **When you delete a comparison arm, check what it was incidentally covering.** Learned the hard way
  there: the rank-aware-selection guard (I3) had been tested *only* as a side effect of comparing the
  fused Metal eigensolve against the op-graph one, so deleting the kernel silently took its only test
  with it. The same trap applies to this deletion and to `/simplify`-style cleanups generally — the
  fix is to re-run coverage checks after removing an arm, not to assume the remaining arms overlap.

**`svsim.py`** — JAX state-vector simulator. Gates are compiled to the same symplectic `CircuitXZ` (x, z, cos, sin) form and applied by a single `jax.lax.scan`, so only `x, y, z, cz, rx, ry, rz, rzz` are supported; transpile to `basis_gates=['rx','ry','rz','rzz']` first. **`sin` is complex128, not real**: it carries `i·(-i)^popcount(x&z)`, i.e. the rotation's leading `i` and the `(-i)^{x·z}` phase of the `Q = (-i)^{x·z} Z^z X^x` convention, folded in at build time. Omitting that phase silently broke every `y`/`ry` gate — the only gates with overlapping X/Z signatures — and so every transpiled circuit (`docs/skqd.md`). Don't narrow `sin` back to float64. `cz` is only decomposed on the `QuantumCircuit` path (not as a raw gate spec) and is correct only up to a uniform `exp(iπ/4)`, which its `rzz`+`rz` decomposition cannot express.

**`qprint.py`** — pretty-printer with two orthogonal axes: `fmt` picks the content class (`QPrintBraKet` / `QPrintPauli` / `QPrintMatrix`) and `output` picks the rendering (`'text'` returns the object for lazy `__repr__`, `'latex'` a string, `'mpl'` a Figure). `QPrintBase` owns all numerics; subclasses only override `_qobj_data`, `_add_labels`, `_format_lhs`. **Test the full `fmt` × `output` grid, not a diagonal of it**: four bugs lived in cells nothing exercised, including `fmt='matrix'` being un-instantiable for *every* input (`QPrintMatrix` never implemented the abstract `_add_labels`, which it does not need — it overrides `_make_lines` and positions terms by row/column). Note that an amplitude of exactly `1` is suppressed, which is right when a basis label follows and wrong when nothing does; text-mode labels also carry the `*` separator as a prefix, so the two renderings can disagree while each looks fine alone. Cross-rendering assertions are what catch that — see `tests/test_qprint.py::TestAmplitudeAndSeparator`.

## Conventions to follow when editing

**`npmod`** — numeric functions take `npmod: ModuleType = np` as the last kwarg so callers can pass `jax.numpy` for traceable execution. The rule: **validation and early returns must be gated on `if npmod is np:`**, but **Python-level shape inference must NOT be** — it operates on static values and is needed identically by both backends. Gating it is what broke `paulis/general.py`'s entire `npmod=jnp` path in three separate places (`components` plus the since-removed `compose` and `truncate`, all raising `TypeError: object of type 'int' has no len()` on a scalar `dim`) — `_normalize_dim` is now called unconditionally at every site for exactly this reason. Shape arithmetic should use `np` explicitly, not `npmod`: `jnp.sqrt` rejects the plain tuples that `array.shape` returns, where numpy accepts them. And prefer an unrolled Python loop over `jax.lax.fori_loop` when the trip count is static — a traced loop index cannot subscript a static dimension tuple (`TracerIntegerConversionError`). `components` is now the only `npmod` consumer in the module; `tests/test_paulis_general.py::TestNpmodParity` pins it, including under `jax.jit`.

**Optional dependencies** — uniform `try: import X / except ImportError: HAS_X = False / else: HAS_X = True` at module top, every use guarded by `HAS_X and isinstance(...)`. Type aliases are conditionally widened (`CircuitInput |= QuantumCircuit`), and the runtime path raises a `RuntimeError` rather than failing at import. `numpy`, `scipy`, `h5py`, and **`jax`** are hard dependencies.

**Sharding is implicit** — the library reads `jax.sharding.get_abstract_mesh()`; it is the *caller's* job to set the mesh. The examples establish the expected pattern: a single axis named `'x'` with `AxisType.Explicit`, plus `jax.config.update('jax_enable_x64', True)` (without x64 you silently get complex64/int32 — you'll see truncation warnings).

**Docstrings feed the published API reference.** Every module opens with a raw docstring that is a full reST document: over/underlined title, `.. currentmodule::`, prose with `.. math::` derivations of the normalization conventions, and an explicit API section (`.. autofunction::` / `.. autoclass::` / `.. autosummary::`). Function docstrings are Google-style (`Args:` / `Returns:` / `Raises:`) via napoleon. Adding a public module requires **both** those directives **and** a manual line in the `toctree` of `docs/source/index.rst`.

**Docstrings with LaTeX must be raw strings.** `"""... :math:`\alpha` ..."""` compiles `\a` to a BEL
byte: the rendered reference is corrupted while ruff, `ty` and pytest all pass, since it is valid
Python. Sweep after touching any docstring containing a backslash — this found exactly one offender
(`hproj`) across the package. Also brace `:math:` exponents (`2^{31}`, not `2^31`); unbraced renders
as `2³1` and nothing warns.

```bash
uv run python -c "
import rqutils.sqd, rqutils.ground_locg, rqutils.svsim, rqutils.qprint
import rqutils.math as rm, rqutils.paulis.general as pg, rqutils.paulis.symplectic as ps
bad = [(m.__name__, n) for m in (rqutils.sqd, rqutils.ground_locg, rqutils.svsim, rqutils.qprint, rm, pg, ps)
       for n, o in list(vars(m).items()) + [('<module>', m)]
       if isinstance(getattr(o, '__doc__', None), str) and any(c in o.__doc__ for c in '\x07\x08\x0b\x0c')]
print('control-char docstrings:', bad)"
```

**`.. autoclass::` needs `:members:` or member docstrings are unpublished.** `PauliSumXZ`'s four
documented public members rendered nowhere until it was added; `CircuitXZ` is deliberately bare (no
documented members, so the flag is a no-op). Confirm by grepping the built HTML for an
`id="...<name>"` anchor, not by reading the source. The docs build has **one** standing warning
(`rqutils.paulis.rst` not in any toctree — a `sphinx-apidoc` package stub); anything beyond that is
yours. Note `grep -c warning` on the build output counts Sphinx's own summary line too.

**Writing `Raises:` sections finds bugs.** You cannot document a raise without reading its condition,
which caught three wrong claims in one pass: `apply_h`'s `states` arg omitted `(1, 1)` from the
no-states-needed set, `components`' documented `ValueError` is gated on `npmod is np` (so under
`npmod=jnp` a bad `dim` gives an opaque `dot_general` TypeError instead), and `ground_locg` accepts a
bare Python `int` for `xinit` despite reading `.dtype` — both callers are `jax.jit`-wrapped, so it
arrives as a 0-d tracer. Trigger every raise you document.

**Verify the referent, not just that the pointer resolves.** Two cross-references in `tests/` were
"fixed" by removing dead paths while their surrounding claims stayed wrong: a `3.6e-15` agreement
figure that was a one-off observed value rather than the actual `1e-9` gate, and an `hproj` workaround
described as live when the file it points at records the bug as fixed. Read the target. Same rule for
cost figures — A/B the whole call against the pre-change revision in a worktree (`git worktree add`,
`PYTHONPATH` at it, since the venv's editable install otherwise serves HEAD to both arms): timing a
guard predicate alone measured 3.4–3.8% where the end-to-end cost was 12–14%.

## Known rough edges

- `paulis(dim)` for multiple subsystems uses `np.einsum` with 3 letters per subsystem, capping at ~17 subsystems; `sparse=True` for products raises `NotImplementedError`.
- `sqd` and `hproj` are limited to `N ≤ 2^31 - 1` subspace states, and **this is now enforced rather than only documented** (`_MAX_STATES`, checked in both entry points). Subspace positions are int32 throughout — `uniquify_states`' iota, and `get_xsource`'s output with `-1` as the absent marker — so a size at or above `2^31` wrapped to `-2147483648` and returned a corrupted *permutation* rather than raising. Note the wrapped value is not `-1`, so the absent-marker test could not catch it either. Unreachable on real hardware (`2^31` states is 4.3 GB of packed states before any vector), which is why the check is cheap insurance. `hproj`'s guard sits **before** its O(N) sortedness scan deliberately: it is an O(1) look at a shape, and measured 0.23 s there against 23 s when placed after the scan. `uniquify_states`' single-device `jax.lax.sort` is the reason for the limit's magnitude; `get_xsource` no longer contributes — it is a binary search into the already-sorted list, not a sort of a stacked `2N` array.
  **Known gap, deliberate**: nothing pins the *accept* side of the boundary, so relaxing either guard's `>` to `>=` leaves the suite green while rejecting the largest legal size. Closing it costs 23 s (the passing case runs the O(N) scan), and the mutant's only consequence is rejecting a call that would OOM anyway — see the comment in `tests/test_sqd.py::TestHproj`.
- **`hproj(unique_states=True)` now raises on unsorted or duplicate-containing input, where it used to return a wrong matrix.** A behavioural change, not just a docs fix — though "break" means they were silently receiving a non-symmetric projection. `get_xsource` binary-searches into `states`, so both halves of "uniquified and lex-sorted" are load-bearing; the check is host-side numpy at 12–14% of `hproj`, on that opt-in path only. Pass `np.unique(states, axis=0)`, or leave `unique_states=False`.
- **`apply_h` is keyword-only; its positional `(scanned, cache_level)` form is gone.** The second downstream-visible break. A call like `apply_h(vec, (xsources, diagonals), None, (1, 2))` now raises `TypeError: apply_h() takes 1 positional argument but 4 were given`; the replacement is `apply_h(vec, xsources=..., diagonals=...)`. Deliberately a hard break rather than a deprecation shim: the whole point is that the six valid input sets become the only constructible ones, and keeping the unpaired form alive would have preserved exactly the hazard it was removed for. Callers binding a matvec thunk should bind the *arrays* (`functools.partial(apply_h, xsources=xs, diagonals=dg)`) rather than the `cache_level` — the four `examples/scaling/` POCs show the migration, and it makes the thunk `lambda: mv(vec)`.

There is no CHANGELOG in this repo — if one is ever added, both of the above belong in it.
- `run_sqd`'s initial vector is a deterministic pseudo-random spread (`_spread_seed`), not a one-hot: a one-hot cannot leave the connected component of the projected Hamiltonian that contains it, so a subspace whose Hamiltonian splits into disconnected blocks silently returned that block's minimum with `converged=True`. `vinit_from_min_diag` still weights the minimum-diagonal state heavily on top of the spread. Don't "simplify" either back to a one-hot — `tests/test_sqd.py::TestSqdInitialVector` covers both failure modes.
