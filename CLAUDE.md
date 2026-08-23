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

`examples/` holds the three library demos (`sqd.py`, `svsim.py`, `bench.py`) plus the `scaling/`
subdirectory, six scaling POCs measured against `rqutils.sqd` and documented in
`docs/scaling-pocs.md`.

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
`paulis/general`, `paulis/symplectic`, `qprint`, `math` — but **only single-device**: no *test*
exercises a multi-device mesh. `sqd`'s mesh-size padding and, through it, `ground_locg`'s
`out_sharding` contract are now covered by `examples/scaling/poc7_sharding.py` (see below) rather than
by pytest — run it after any change to `ground_locg`'s reductions or helper signatures, not just
after touching `sqd`; **`svsim`'s `out_sharding` is still exercised by nothing.** `poc7_sharding.py` now sweeps
`cache_level=(1, 2)` alongside the default `(1, 0)`, because the two take different diagonal paths and
sweeping only the default is how `_diag_all_groups` first shipped with broken sharding while the POC
stayed green — **a passing sharding POC is not evidence for a path it does not select.** That blind
spot also hid a second, older bug: `vinit_from_min_diag` narrowed its diagonal to `.real` on one
branch and not the other, so `cache_level=(1, 2)` raised `TypeError: lt does not accept dtype
complex128` for any odd-Y operator with an all-identity X group (reproduced at `ac15362`). Both are
fixed and pinned. The POC now sweeps **all six** levels, and the three
`cache_level[0] == 0` ones — which raised `ShardingTypeError` on any mesh, at `e5703ac` and every
revision since — are fixed too. Root cause was **`_spread_seed`, not `xsource`** (an earlier note
here guessed wrong): `_run_sqd` defers `jax.reshard(states_u, sharding)` until after the last
`xsource`, which on `cache_level[0] == 0` happens *inside the matvec*, so the reshard never runs
before `_spread_seed` — leaving its `jnp.where` predicate unsharded against a sharded `vec`
("select `which` must be scalar or have the same sharding as cases"). `_spread_seed` now reshards the
small 0/1 predicate itself, which is cheaper than resharding the state list and leaves the binary
search's sortedness precondition untouched. **pytest is single-device, so `poc7_sharding.py` is the
only coverage** — run it after touching `_spread_seed`, the reshard placement, or any diagonal path.
`tests/` also contains three Jupyter notebooks used as interactive scratchpads; pytest does not
collect them.

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
`_run_sqd`'s filler-diagonal masking. `test_filler_slots_are_excluded_against_a_dense_reference` closes
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

**Three benchmarking lessons that outlived the code that taught them.** Each was measured on a port
that has since been deleted, but none depends on it:

- **When you delete a comparison arm, check what it was incidentally covering.** A guard was covered
  *only* as a side effect of comparing two implementations of an eigensolve; deleting one arm silently
  took the guard's only test with it. Applies to any A/B you remove.
- **Expect a ~3.9% noise floor when benchmarking a before/after on this machine.** Treat any
  difference under ~4% as unresolved, not as a result.
- **fp32 in an eigensolve is a dynamic-range trap**, not merely a precision one: a large shift
  destroys the small eigenvalue you are solving for. `examples/scaling/poc6_mixed_precision.py`
  measures it.

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
`eigvalsh` reference, plus every residue of `N mod mesh.size`. Timings under virtual devices are
meaningless (they share one CPU), so use this for correctness only.

## Architecture

Seven largely independent modules under `rqutils/`; nothing but `sqd.py → {paulis/symplectic.py, ground_locg.py}` and `qprint.py → paulis/general.py` couples them.

**Two unrelated Pauli representations — do not confuse them:**

- `paulis/general.py` — dense generalized (Gell-Mann-like) basis for arbitrary dimension. The whole public surface is `paulis(dim)`, `pauli_matrices(dim)` (the single-subsystem primitive `paulis` builds on), `components()`, and `labels()`; `compose`, `truncate`, `l0_projector`, `symmetry` and `paulis_shape` were removed as dead code, so don't reintroduce a caller expecting them. Normalization is `tr(λ_k λ_l) = 2δ_kl`, so **`λ_0 = sqrt(2/n)·I`, not `I`** — the most bug-prone invariant here. Basis-index ordering is fixed by a shell-by-shell construction loop; `components` and `labels` both index by basis position, so a reordering would disagree with the labels users read while every function stayed self-consistent. Shapes: `paulis(dim)` → `(d1², …, D, D)` (basis axes *first*); component arrays → `(…, d1², …)` (component axes *last*). Everything is memoized in module-level dicts keyed by a `tuple(int)`-normalized `dim` (`_normalize_dim`).
- `paulis/symplectic.py` — `PauliSumXZ`, a bit-packed qubit-only form for JAX/GPU. Convention `Q = (-i)^{x·z} Z^z X^x` with **little-endian qubit ordering** (Qiskit's `.x`/`.z` get reversed on ingest). Terms are grouped by unique X signature, Z groups zero-padded to a rectangle, the `(-i)^{popcount(x&z)}` phase folded into the coefficients, then `np.packbits`. Signatures always reserve **one pad bit at position 0** (a dummy identity, aligning with the pad bit consumers put in their state bitstrings) — not optional, so alignment cannot be got wrong. Note that `packbits` fills each byte from the **most significant** end, so a signature's payload bits are entries `1` through `num_qubits` of `np.unpackbits`, in string-character order (leftmost character = index 1) — the reverse of the qubit numbering. Anything decoding a packed signature back to an integer must shift by `8*nbytes - (num_qubits + 1)`, counting the pad bit; dropping the `+1` silently returns a *permutation* of the right answer (measured 2.07 max abs error in the since-removed `matmul`, which is why that method is gone). The class is now just `from_paulisum` and the `arrays` property. **A complex coefficient raises** (non-Hermitian; both ingest branches share one check) — there is no `force_real` flag any more, because none could work: `.c` narrows to float64 exactly when the folded phase is real, i.e. when every string has an even Y count, and an odd-Y string is complex128 *by construction*. Check `.c.dtype` if you need float64.

**`sqd.py`** — sample-based quantum diagonalization: project a large Pauli-sum Hamiltonian onto the subspace spanned by a list of computational-basis bitstrings and solve matrix-free. `sqd(...)` is the entry point; `hproj(...)` is the dense/debug path. The module's `__all__` (and its docstring's API section) publish the public surface in three tiers: solvers (`sqd`, `hproj`); a stable kernel API — `uniquify_states`, `xsource`, `diag_signs`, `apply_h` — verified self-sufficient to build a matvec by hand, since `diag_signs` feeds `apply_h` directly with no diagonal builder needed; and lower-level building blocks (`diagonals`, `apply_xgroup`) documented for the in-tree POCs but carrying no stability promise. `diagonals()` is one dispatcher over the three sign sources (a `diag_signs` array, a `(zsignatures, states)` pair, or the ragged `(zsignatures, group_ids, states, num_groups)` form) that replaced three separate builders; the kernels behind each branch — `_diag_from_signs`, `_diag_from_z`, `_diag_all_groups` — are private. `_run_sqd` (also private) is `sqd`'s jitted body. Two conventions dominate:

- States carry **one extra zero pad bit at position 0** before `packbits`. `PauliSumXZ` reserves the same bit in its signatures unconditionally, so the two are aligned by construction — this used to be an opt-in `add_padding` flag, and the two sides disagreeing is how `hproj` shipped broken. Filler slots produced by uniquification are `255`, detected via `states_u[:, 0] >> 7`.
- `cache_level=(source_indices, diagonals)` selects among six matvec strategies. Only the *diagonal* axis is a genuine memory-for-speed trade; the source-index axis is near-free to enable and very expensive to disable (see below). They are one kernel, `apply_h`, indexed by that 2×3 grid; `cache_level` **must** stay static (it is bound via `functools.partial`, since `ground_locg` splats `args` positionally and `static_argnames` would never see it). `states_size` exists solely to pin array shapes and prevent JIT recompilation.

Two measured facts about the cost, from the six scaling POCs under `examples/scaling/` (findings in
`docs/scaling-pocs.md`). **`xsource` setup dominates** — weighted by call count it is 66–97% of a
solve (3.1 s against 79 ms of matvec loop at 10 iterations, N=200k, J=50), so the `2N` sort is not
merely the `N ≤ 2^31` ceiling but the main cost at every size measured, while `matvec/J` is flat at
~0.16 ms and confirms the `O(J·N)` model. The module docstring's caching section now carries this
magnitude and the resulting advice (prefer `cache_level[0] = 1`), rather than byte counts alone —
independently reproduced end-to-end at N=3k, n=12, J=23: `(0, 2)` is 10.9× slower than `(1, 2)` and
`(0, 0)` is 7.2× slower than `(1, 0)`, all four levels returning the same energy.

Because that change landed, **the POCs under `examples/scaling/` no longer have a baseline in the
library, and must not be pointed at one.** `poc1.xsource_sort_legacy` is a verbatim copy of the
pre-23fb226 sort and is the timing baseline for both POC 1 and POC 8; their *correctness* arms still
compare against `xsource`, which is the point (agreement with what ships is now a regression test).
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

**`xsource` is now a binary search, not a sort** (12–19× faster on the J-fold precompute on CPU),
which is why **`states` must be lex-sorted** — always required, since the sort was equally wrong on
unsorted input, but previously undocumented. `hproj(unique_states=True)` skips its `np.unique` and so can
violate it; that returns a non-symmetric matrix and is pinned by
`TestHproj::test_unsorted_input_with_unique_states_is_wrong`. Two paths selected statically on width:
`uint64` keys for `B ≤ 8` bytes, an explicit lexicographic search beyond. That boundary is a
**correctness** limit — a `uint64` key silently truncates a wider row and aliases distinct states —
so if you touch it, note that a test only catches the overrun when the subspace's *leading* bytes
collide and partners genuinely exist; `packbits` puts low qubit indices in the *leading* bytes, the
reverse of the qubit numbering, and getting that backwards makes the test pass vacuously.

**`ground_locg.py`** — single-vector (block-size-1) LOBPCG specialization used as `sqd`'s eigensolver, with the Rayleigh–Ritz step solved analytically (`eigenpair_2x2`, `eigenpair_3x3` via Cardano) instead of via `eigh`, to keep memory down for huge vectors. It is sharding-transparent **only if the `mat` callable preserves output sharding** — that contract is why every `apply_*` in `sqd.py` passes `out_sharding=jax.typeof(vec).sharding`. Every guard in it is load-bearing and was measured: `docs/locg.md` catalogues seven defects (I1–I7) that each failed *silently*, returning a plausible wrong number rather than raising. Don't "simplify" the balancing, the re-orthogonalizations, or the zero-direction masks. **`docs/locg.md` is stale** — it audits the pre-rewrite module, so its line numbers, its "no pytest suite exists" scope note, and its A1–A5 gaps (all since fixed) don't apply; cite it for the I-numbers and the measurements only, and read the module docstring for what currently holds. One severity is partly retracted there: I5's full 1.0 collapse needed I4's 2000-iteration runs to develop, so the *eigenvalue* stays correct without the guard — but the guard is pinned, by `TestBasisOrthogonality`, which asserts `|<x|y>|` off the `debug=True` diagnostics and fails 3 of 4 arms when it is neutered. Two traps when editing it. `body_iter1`'s exclusion bound (`2|rho| + 1`) is **not** `body()`'s (`max(diag) + sum(|diag|) + 1`) specialized to one entry — the general form collapses to a constant `1.0` for the negative `rho` of a ground-state search. Both are *valid* (each strictly exceeds the one retained entry, so the eigensolver still can't pick the excluded slot), but the unified form's margin stops tracking the operator scale, which is a poor trade in a routine whose other guards exist because large shifts destroy precision. Don't unify without redoing the bound argument. A one-matmul `_compute_sas`, stacking the three vectors instead of scattering them, was measured 98.7 ms against the scatter form's 27.7 ms at N=16.8M, because stacking three huge vectors is two 402 MB temporaries per iteration — the copy this module exists to avoid.

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
- **`apply_h`'s single static `nterms` is wrong for high-skew operators, and that is a known limit rather than an oversight.** One trip count serves every X group, so each pays the widest group's extent. The deciding quantity is the *skew*, `max(nzterms)/mean(nzterms)` — **not** `J`, which an earlier revision swept and which is why the docstring once carried a blanket "do not unroll". Measured at N=4096: skew 4.2× → unroll 1.19×, break-even 2120 calls (scan wins); skew 12.8× → 7.88×, break-even 49; skew 25.6× → 23.26×, break-even 4. A solve runs tens to low hundreds of matvecs, so below ~6× skew keep the scan and above ~13× the unroll wins. Hamiltonians sit low (a NN chain plus one long-range term is ~9×), which is why the shipped default is `max(nzterms)`. **Rotated operators sit high** — a caller measured a 782×197 rectangle, median 2 terms/group, 1.4% of slots real, and routing a cost function through `apply_h` came out 2× slower at n=20 and 9.7× at n=100 than a hand-rolled nonzero-only contraction; they reverted it. **That regime is now closed by `_diag_all_groups`** (`sqd.py`), which takes `PauliSumXZ.flat_terms` — the padding-free `(z, c, group_ids)` layout derived from `nzterms` — and builds every group's diagonal in one sharded scatter-add over the real terms. Bit-identical to the scan (`maxdiff` exactly 0.0) and 1.4× at skew 4.2× rising to 148.7× at skew 42.8×, with compile time *lower* than the scan's at every size (52 ms against 282 ms at J=100) because it emits one kernel over the real terms rather than a scan over the rectangle. That flat compile curve is what makes it the right structure where unrolling the group loop was not. Differentiable w.r.t. coefficients with no special handling — verified against central finite differences to 1.2e-10 on a full `expval`. **Sharding-transparent, and it took two changes**: the per-term contribution is computed inside a `vmap` (so each term is a `(num_states,)` row inheriting `states`' sharding — multiplying outside against a stacked parity is a rank-mismatched `P(None, 'x')`/`P(None,)` broadcast jax refuses), and the reduction is an explicit `.at[ids].add(..., out_sharding=...)` rather than `jax.ops.segment_sum`, which allocates its accumulator internally and dropped the sharding even with correct inputs. Bit-identical either way; verified on 4 devices across every residue of `N mod mesh.size` to 2.2e-15. `_accumulate_diagonal`'s docstring carries the skew table, `_diag_all_groups`' carries the speedup table.
- `sqd` is limited to `N ≤ 2^31` subspace states because `uniquify_states` sorts the state list (`jax.lax.sort`) within a single device. `xsource` no longer contributes — it is a binary search into the already-sorted list, not a sort of a stacked `2N` array.
- **`hproj(unique_states=True)` now raises on unsorted or duplicate-containing input, where it used to return a wrong matrix.** A behavioural change, not just a docs fix, so it is the one thing in this cleanup that can break a downstream caller — though "break" means they were silently receiving a non-symmetric projection. `xsource` binary-searches into `states`, so both halves of "uniquified and lex-sorted" are load-bearing; the check is host-side numpy at 12–14% of `hproj` (re-A/B'd against `2a89f94~1`: 12.8–17.8%), on that opt-in path only. Pass `np.unique(states, axis=0)`, or leave `unique_states=False`. **Note the opt-in path is the *faster* one**, which the cost figure alone hides: the O(N) sortedness scan displaces `np.unique`'s O(N log N) sort, so `hproj` is 1.4× faster at N=466 rising to 2.6× at N=2545. Two wrong numbers to avoid re-deriving — a cold-versus-warm comparison reads 55×, and timing `_is_lex_sorted` alone reads 4.8–6.6% instead of the whole-call 12–18%. There is no CHANGELOG in this repo — if one is ever added, this belongs in it.
- `_run_sqd`'s initial vector is a deterministic pseudo-random spread (`_spread_seed`), not a one-hot: a one-hot cannot leave the connected component of the projected Hamiltonian that contains it, so a subspace whose Hamiltonian splits into disconnected blocks silently returned that block's minimum with `converged=True`. `vinit_from_min_diag` still weights the minimum-diagonal state heavily on top of the spread. Don't "simplify" either back to a one-hot — `tests/test_sqd.py::TestSqdInitialVector` covers both failure modes.
