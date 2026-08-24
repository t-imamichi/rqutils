# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

This file holds the **rules**. Their evidence — measurements, post-mortems, and the simplifications
that were tried and measured worse — lives in **`NOTES.md`**. Read that before changing anything here
that looks redundant or over-engineered: most of it is load-bearing for a reason someone had to find
the hard way.

## Environment & commands

Always use `uv run python` (not bare `python`) — the venv at `.venv` is managed by uv. No `timeout` on
macOS (it is GNU coreutils) — use the Bash tool's own timeout rather than wrapping a command in it.
The shell is fish: **quote grep globs** (`--include="*.py"`). Unquoted, fish fails with
`(eval):1: no matches found` before grep runs — which looks like "no results", not "no command".

**Check `git rev-list --left-right --count origin/<branch>...HEAD` before amending.** Work happens on
feature branches (`metal`, not `main`) that get pushed mid-session, so "my commits are still local"
goes stale within a turn — amend then and you rewrite published history. `git branch -r --contains
<sha>` is the per-commit check.

**`metal` is stale — `dev` is authoritative.** Confirmed 2026-08-25. The two lines diverged in both
directions (18 vs 28 commits) and several items were done *independently on both* — the MLX removal, the
`2^31` subspace ceiling, `apply_h`'s keyword-only rewrite (C1), C2 and C3 — so `git log --cherry-pick`
reports every `metal` commit as unique and the overlap is invisible from the log alone. Where they overlap,
`dev` is the more thorough: on the `cache_level[0] == 0` `ShardingTypeError` its fix carries a dedicated
`tests/_sharded_cache_levels.py` subprocess harness (231 insertions) against `metal`'s 52.

One thing is **only** on `metal` and would be lost if it is deleted: the **segment-sum diagonal builder for
ragged X groups** (`f29a3f8`, made sharding-transparent by `16f4646`, exposed via a `diagonals()` entry
point in `01e7bc0`) — `dev` has no `all_diagonals`, no `diagonals()` and no segment-sum path. Also
`metal`-only: an API reorganization into three documented tiers and a CHANGELOG. Cherry-pick from there
rather than rebasing the branch if any of that is wanted later.

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

One `tests/test_<module>.py` per module; all seven are covered. `tests/` also holds three Jupyter
notebooks used as scratchpads (pytest does not collect them) and `tests/_sharded_*.py` scripts — the
leading underscore keeps them uncollected, and they are subprocessed under
`XLA_FLAGS=--xla_force_host_platform_device_count=4` because the virtual device count must be set
before jax initializes.

`tests/conftest.py` enables `jax_enable_x64` before any `rqutils` import — every tolerance in the suite
depends on it — and holds the shared reference helpers (dense Pauli sums, projections, gate unitaries),
each validated against qiskit before being trusted. It also configures caches that take the suite from
~53 s to ~6 s; expect ~53 s on a first run.

Rules, each of which cost a defect to learn — **evidence in `NOTES.md`**:

- **Tests are organized by defect.** Name the defect a new test locks down and record the measured
  wrong value. Prefer an *independent* reference (dense construction, scipy, qiskit) over
  self-consistency: several bugs made every internal code path agree on the same wrong number.
- **A "does X change the answer?" test needs an arm where X is truly absent.** Verify that; don't
  assume it from the parameter being unset. The padding test that looked like such a control had
  filler slots in *both* arms and passed against a mutant returning −1.2 for a true −0.8297058541.
- **Verify a new test fails against the bug it targets by reverting the fix in place** — a copy of the
  repo does not work, since the venv holds an editable install pointing at the original. `NOTES.md`
  has the mutation-testing recipe; mutate `@jax.jit` code in a **fresh subprocess** or both arms reuse
  one compiled kernel.
- **A green suite after reverting a fix means the test is missing, not that the guard is dead.** Some
  guards are only reachable when other defects compound with them. Look for a *more direct* assertion
  (the invariant, off `debug=True` diagnostics) before recording a negative result.
- **Fixtures are built inside each test body; there are no `@pytest.fixture` state generators.**
  Deliberate — several tests pick a seed for a specific pathology and assert the fixture still has it,
  and moving draws into fixtures makes RNG stream position depend on fixture ordering. Keep new
  generators as plain functions taking `rng`; `unique_states`/`collapsing_states` are the pattern.
- **Sweep `cache_level`, don't sample it** — three bugs hid behind the default `(1, 0)`, each masked by
  the one before. One needed a *complex* fixture, not just the parameter varied.
- **Assert the sharding *spec*, not just the values.** A replicated run agrees with single-device to
  exactly 0.0, so "correct but silently unsharded" is invisible to value comparison.
- **Multi-device paths are testable on CPU** via `--xla_force_host_platform_device_count=4`. Use it for
  correctness only — timings under virtual devices are meaningless.
- `examples/scaling/poc7_sharding.py` is the fuller sharding harness. Run it after any change to
  `ground_locg`'s reductions or helper signatures, not just after touching `sqd`.
- `svsim` requires **`mesh.size` to divide `2^num_qubits`** — documented rather than fixed, since a
  state vector's indices *are* the basis states and cannot be padded. `PartitionSpec(None)` replicates.

## Architecture

Seven largely independent modules under `rqutils/`; nothing but `sqd.py → {paulis/symplectic.py,
ground_locg.py}` and `qprint.py → paulis/general.py` couples them. Plus `product.py`, a SCIP
product-state solver, which nothing else imports.

**Two unrelated Pauli representations — do not confuse them:**

- `paulis/general.py` — dense generalized (Gell-Mann-like) basis for arbitrary dimension. The whole
  public surface is `paulis(dim)`, `pauli_matrices(dim)`, `components()`, and `labels()`; `compose`,
  `truncate`, `l0_projector`, `symmetry` and `paulis_shape` were removed as dead code, so don't
  reintroduce a caller expecting them. Normalization is `tr(λ_k λ_l) = 2δ_kl`, so
  **`λ_0 = sqrt(2/n)·I`, not `I`** — the most bug-prone invariant here. Basis-index ordering is fixed
  by a shell-by-shell construction loop, and `components`/`labels` both index by basis position, so a
  reordering would disagree with the labels users read while every function stayed self-consistent.
  Shapes: `paulis(dim)` → `(d1², …, D, D)` (basis axes *first*); component arrays → `(…, d1², …)`
  (component axes *last*). Everything is memoized in module-level dicts keyed by a
  `tuple(int)`-normalized `dim` (`_normalize_dim`).
- `paulis/symplectic.py` — `PauliSumXZ`, a bit-packed qubit-only form for JAX/GPU. Convention
  `Q = (-i)^{x·z} Z^z X^x` with **little-endian qubit ordering** (Qiskit's `.x`/`.z` get reversed on
  ingest). Terms are grouped by unique X signature, Z groups zero-padded to a rectangle, the
  `(-i)^{popcount(x&z)}` phase folded into the coefficients, then `np.packbits`. Signatures always
  reserve **one pad bit at position 0** (a dummy identity, aligning with the pad bit consumers put in
  their state bitstrings) — not optional, so alignment cannot be got wrong. Decoding a packed signature
  back to an integer must shift by `8*nbytes - (num_qubits + 1)`; dropping the `+1` silently returns a
  *permutation* (`NOTES.md`). The class is now just `from_paulisum` and the `arrays` property. **A
  complex coefficient raises** (non-Hermitian); there is no `force_real` flag and none could work —
  check `.c.dtype` if you need float64.

**`sqd.py`** — sample-based quantum diagonalization: project a large Pauli-sum Hamiltonian onto the
subspace spanned by a list of computational-basis bitstrings and solve matrix-free. `sqd(...)` is the
entry point; `hproj(...)` is the dense/debug path. Two conventions dominate:

- States carry **one extra zero pad bit at position 0** before `packbits`. `PauliSumXZ` reserves the
  same bit unconditionally, so the two are aligned by construction — this used to be an opt-in
  `add_padding` flag, and the two sides disagreeing is how `hproj` shipped broken. Filler slots from
  uniquification are `255`, detected via `states_u[:, 0] >> 7`.
- `cache_level=(source_indices, diagonals)` selects among six matvec strategies. Only the *diagonal*
  axis is a genuine memory-for-speed trade; the source-index axis is near-free to enable and very
  expensive to disable — **prefer `cache_level[0] = 1`** (`get_xsource` setup is 66–97% of a solve; see
  `NOTES.md`). They are one kernel indexed by that 2×3 grid, reached two ways. The **public `apply_h`
  is keyword-only**: name the arrays you have and the strategy follows, so the six valid input sets are
  the only constructible ones. Internally `run_sqd` calls the private `_apply_h_kernel` with an
  assembled tuple and `cache_level` bound via `functools.partial` — it **must** stay static there,
  since `ground_locg` splats `args` positionally and `static_argnames` would never see it.
  `states_size` exists solely to pin array shapes and prevent JIT recompilation.

**`states` must be lex-sorted** — `get_xsource` is a binary search, not a sort. Always required (the
sort was equally wrong on unsorted input) but previously undocumented; `hproj(unique_states=True)`
skips its `np.unique` and so can violate it. Two paths selected statically on width: `uint64` keys for
`B ≤ 8` bytes, explicit lexicographic search beyond — a **correctness** boundary, not a performance
one.

**`run_sqd`'s initial vector is a deterministic pseudo-random spread (`_spread_seed`), not a one-hot.**
Don't "simplify" it back — a one-hot cannot leave its connected component, so a subspace whose
Hamiltonian splits into disconnected blocks silently returned that block's minimum with
`converged=True`.

**`ground_locg.py`** — single-vector (block-size-1) LOBPCG specialization used as `sqd`'s eigensolver,
with the Rayleigh–Ritz step solved analytically (`eigenpair_2x2`, `eigenpair_3x3` via Cardano) instead
of via `eigh`, to keep memory down for huge vectors. It is sharding-transparent **only if the `mat`
callable preserves output sharding** — that contract is why every `apply_*` in `sqd.py` passes
`out_sharding=jax.typeof(vec).sharding`. It also takes an optional `precond` (`None` | callable).

Every guard in it is load-bearing and was measured; `docs/locg.md` catalogues seven defects (I1–I7)
that each failed *silently*. **Don't "simplify" the balancing, the re-orthogonalizations, or the
zero-direction masks**, don't unify `body_iter1`'s exclusion bound with `body()`'s, and don't
reintroduce the one-matmul `_compute_sas` form. `NOTES.md` has the measurements and notes that
`docs/locg.md` is stale on scope and line numbers.

**The MLX port is gone** — deleted, not deprecated-in-place, because the JAX solver measured faster
even on the MLX GPU backend. Don't reintroduce a second solver implementation without that measurement
going the other way first. `NOTES.md` records what the deletion left behind, including the trap that
**deleting a comparison arm can remove the only test of something else**.

**`svsim.py`** — JAX state-vector simulator. Gates are compiled to the same symplectic `CircuitXZ`
(x, z, cos, sin) form and applied by a single `jax.lax.scan`, so only `x, y, z, cz, rx, ry, rz, rzz`
are supported; transpile to `basis_gates=['rx','ry','rz','rzz']` first. **`sin` is complex128, not
real** — it carries the rotation's leading `i` and the convention's `(-i)^{x·z}` phase, and narrowing
it silently breaks every `y`/`ry` gate and so every transpiled circuit. `cz` is only decomposed on the
`QuantumCircuit` path and is correct only up to a uniform `exp(iπ/4)`.

**`qprint.py`** — pretty-printer with two orthogonal axes: `fmt` picks the content class
(`QPrintBraKet` / `QPrintPauli` / `QPrintMatrix`) and `output` picks the rendering (`'text'` returns
the object for lazy `__repr__`, `'latex'` a string, `'mpl'` a Figure). `QPrintBase` owns all numerics;
subclasses only override `_qobj_data`, `_add_labels`, `_format_lhs`. **Test the full `fmt` × `output`
grid, not a diagonal of it** — four bugs lived in cells nothing exercised (`NOTES.md`).

**`product.py`** — SCIP product-state solver (`solve_product`), a hard `pyscipopt` dependency. Returns
both the product-state objective and SCIP's `lower_bound`; **both are *upper* bounds on the true ground
energy**, since product states are a strict subset of Hilbert space. `NoSolutionError` is distinct
because `getVal()` does *not* fail when no solution exists — it returns an unfinished search point.

## Conventions to follow when editing

**`npmod`** — numeric functions take `npmod: ModuleType = np` as the last kwarg so callers can pass
`jax.numpy` for traceable execution. The rule: **validation and early returns must be gated on
`if npmod is np:`**, but **Python-level shape inference must NOT be** — it operates on static values and
is needed identically by both backends. Gating it broke `paulis/general.py`'s entire `npmod=jnp` path in
three places, which is why `_normalize_dim` is now called unconditionally at every site. Shape
arithmetic should use `np` explicitly, not `npmod` (`jnp.sqrt` rejects the plain tuples that
`array.shape` returns). Prefer an unrolled Python loop over `jax.lax.fori_loop` when the trip count is
static — a traced loop index cannot subscript a static dimension tuple
(`TracerIntegerConversionError`). `components` is the only `npmod` consumer left;
`tests/test_paulis_general.py::TestNpmodParity` pins it, including under `jax.jit`.

**Optional dependencies** — uniform `try: import X / except ImportError: HAS_X = False / else:
HAS_X = True` at module top, every use guarded by `HAS_X and isinstance(...)`. Type aliases are
conditionally widened (`CircuitInput |= QuantumCircuit`), and the runtime path raises a `RuntimeError`
rather than failing at import. `numpy`, `scipy`, `h5py`, `pyscipopt`, and **`jax`** are hard
dependencies.

**Sharding is implicit** — the library reads `jax.sharding.get_abstract_mesh()`; it is the *caller's*
job to set the mesh. The examples establish the expected pattern: a single axis named `'x'` with
`AxisType.Explicit`, plus `jax.config.update('jax_enable_x64', True)` (without x64 you silently get
complex64/int32 — you'll see truncation warnings).

**Docstrings feed the published API reference.** Every module opens with a raw docstring that is a full
reST document: over/underlined title, `.. currentmodule::`, prose with `.. math::` derivations of the
normalization conventions, and an explicit API section (`.. autofunction::` / `.. autoclass::` /
`.. autosummary::`). Function docstrings are Google-style (`Args:` / `Returns:` / `Raises:`) via
napoleon. Adding a public module requires **both** those directives **and** a manual line in the
`toctree` of `docs/source/index.rst`.

**Docstrings with LaTeX must be raw strings**, `.. autoclass::` needs `:members:`, and `:math:`
exponents must be braced — three hazards that no tool catches, detailed with the sweep command in
`NOTES.md`.

**Writing `Raises:` sections finds bugs** — it caught three wrong claims in one pass. Trigger every
raise you document (`NOTES.md`).

**When measuring: use `eigvalsh` or sparse `eigsh(k=1)`, never `eigh`** (77 s vs 0.02 s at the sizes
here), **A/B whole calls against a worktree of the pre-change revision** rather than timing a predicate
in isolation, and **verify the referent of a cross-reference, not just that it resolves**. All three
with numbers in `NOTES.md`.

## Known rough edges

- `paulis(dim)` for multiple subsystems uses `np.einsum` with 3 letters per subsystem, capping at ~17
  subsystems; `sparse=True` for products raises `NotImplementedError`.
- `sqd` and `hproj` are limited to `N ≤ 2^31 - 1` subspace states, **enforced** (`_MAX_STATES`) in both
  entry points *and* on `uniquify_states`' static `states_size`, where the int32 iota is actually
  created — six `examples/scaling/` scripts call the un-underscored helpers directly and reached the
  iota with neither entry-point guard in the chain. `TestInt32Ceiling` covers both sides of the
  boundary. `NOTES.md` explains why the guard placement is what it is.
- **`hproj(unique_states=True)` now raises on unsorted or duplicate-containing input, where it used to
  return a wrong matrix.** A behavioural change, though "break" means callers were silently receiving a
  non-symmetric projection. `get_xsource` binary-searches into `states`, so both halves of "uniquified
  and lex-sorted" are load-bearing; the check is host-side numpy at 12–14% of `hproj`, on that opt-in
  path only. Pass `np.unique(states, axis=0)`, or leave `unique_states=False`.
- **`apply_h` is keyword-only; its positional `(scanned, cache_level)` form is gone.** A call like
  `apply_h(vec, (xsources, diagonals), None, (1, 2))` now raises `TypeError`; the replacement is
  `apply_h(vec, xsources=..., diagonals=...)`. Deliberately a hard break rather than a deprecation
  shim — the point is that the six valid input sets become the only constructible ones, and the unpaired
  form was unverifiable (`NOTES.md`). Callers binding a matvec thunk should bind the *arrays*
  (`functools.partial(apply_h, xsources=xs, diagonals=dg)`), not the `cache_level`; the four
  `examples/scaling/` POCs show the migration.

The last two are downstream-visible breaks. There is no CHANGELOG in this repo — if one is ever added,
both belong in it.

## Closed investigations — don't reopen without reading the record

- **Preconditioning `sqd` is closed.** `ground_locg(precond=...)` shipped and measures 1.79× median on
  a *shifted* operator, but `sqd` solves the raw indefinite projected `H` and six routes to a usable
  shift were measured and rejected — including a level-1 SDP bound that is valid, tightest-available,
  and still matched by a free `O(N)` diagonal bound. The blocking argument is structural: no bound on
  `H`, however tight, can reach the shift the 1.79× used. `docs/rqutils-precond-request.md`,
  `docs/sdp-lower-bound.md`, summary in `NOTES.md`.
- **Subspace selection by weight shell + diagonal ranking was measured, then rejected** (2026-08-25) —
  sound results against a *uniform random* baseline, which is not what a real SQD workflow produces.
  Do not build on it.
