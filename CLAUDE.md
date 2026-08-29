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

**Worktrees need `worktree.baseRef: "head"`** (in `.claude/settings.json`). The default `fresh`
branches from `origin/main`, 144 commits behind `dev` and predating the `dev` extra, so `uv run
--extra dev` fails outright. A fresh worktree also gets a bare venv: run the full
`--extra dev --extra qiskit --extra mpl --extra qutip` or 23 tests **silently skip**, including
the qiskit reference comparisons this file calls the trustworthy oracle.

**No network access for git in this environment** — `git fetch` fails (`ssh: connect to host
github.com port 22: Operation not permitted`). `origin/*` refs are whatever the last successful fetch
left, so `rev-list --left-right --count` can report `0 0` for commits that were never pushed. Merges
against `origin/dev` use the cached ref; re-fetch from a networked shell before trusting either.

**Check `git rev-list --left-right --count origin/<branch>...HEAD` before amending.** Work happens on
feature branches (`metal`, not `main`) that get pushed mid-session, so "my commits are still local"
goes stale within a turn — amend then and you rewrite published history. `git branch -r --contains
<sha>` is the per-commit check.

**`dev` is the only live branch. `metal` and `product` are both stale — reference only, do not merge.**
Confirmed 2026-08-25. Both are ahead of `dev` in commit count, which makes them look like work waiting to
be integrated; they are not.

**`product`** holds the `rqutils/product.py` (SCIP product-state solver) investigation and an SDP
lower-bound spike. Its **conclusions are already on `dev`**: `CLAUDE.md` and `NOTES.md` here are
byte-identical to `product`'s, and `docs/rqutils-precond-request.md` / `docs/sdp-lower-bound.md` carry
the findings. What lives *only* on `product` is the code — `rqutils/product.py` itself, its `pyscipopt`
and `clarabel` dependencies, and `examples/scaling/poc_sdp_*.py` — plus the commit-by-commit record.
**`dev` has no `rqutils/product.py`**, so ignore any prose here that describes it as a module; the
investigation is **closed** (see "Closed investigations" below) and the module was never brought over.

**`metal`** diverged in both
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
# Sphinx caches: a rebuild prints NO warnings even when they exist. `make clean` first, or a
# docstring regression reads as a clean build.
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

**A global ignore hides real defects; prefer a per-line `# ty: ignore[rule]`.** Three rules were
retired by fixing their few real sites, and two of the three had a genuine bug behind them — a `None`
angle reaching `angle / 2.0` in `conftest.gate_unitary`, and `QPrintBase._process` annotated
`list[list[Term]]` while returning a flat `list[Term]`. Count first: `ty check -c 'rules.X="error"'`
per rule, then read the diagnostics rather than the count. The two patterns that did the work are a
per-line suppression where the stub is genuinely wrong (`math.py`'s `1.0j * mat`) and `@overload` where
a runtime flag picks the return shape (`sqd`'s `return_eigvec`). `invalid-return-type` is the next
candidate at 7 sites; `invalid-argument-type` (96) is the one to leave alone.

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
- **Undo a mutation with `cp` from your own backup, never `git checkout <file>`.** The file usually
  holds unrelated uncommitted work; a checkout discards all of it and the suite passes either way, so
  nothing flags it. `NOTES.md`'s recipe says to copy first — the restore step is the half that matters.
- **Aim a mutant at the layer and branch the test actually exercises.** Two survived this way in one
  session: a coercion added to `sqd` while the test traced `run_sqd` directly, and a mutation of
  `vinit_nodiag` when the fixture takes `vinit_from_min_diag`. Both read as missing coverage and were
  not. Check which branch your fixture reaches *before* concluding anything.
- **To prove an option is a no-op, compare traced graphs, not energies.** On a well-conditioned
  fixture a *working* `prefilter=(16, 1)` returns a bit-identical energy to no prefilter at all (only
  `(32, 2)` moved the last ulp), so "same energy as baseline" is satisfied by both arms and pins
  nothing. `jax.make_jaxpr` string equality separates them; it caught a `cycles=0`→`1` coercion the
  energy form missed.
- **A green suite after reverting a fix means the test is missing, not that the guard is dead.** Some
  guards are only reachable when other defects compound with them. Look for a *more direct* assertion
  (the invariant, off `debug=True` diagnostics) before recording a negative result.
- **Fixtures are built inside each test body; there are no `@pytest.fixture` state generators.**
  Deliberate — several tests pick a seed for a specific pathology and assert the fixture still has it,
  and moving draws into fixtures makes RNG stream position depend on fixture ordering. Keep new
  generators as plain functions taking `rng`; `unique_states`/`collapsing_states` are the pattern.
  They take `rng` **last**: `real_pauli_strings(num_qubits, count, rng)` (returns strings only —
  draw coefficients separately) and `unique_states(num_draws, num_qubits, rng)`.
- **Sweep `cache_level`, don't sample it** — three bugs hid behind the default `(1, 0)`, each masked by
  the one before. One needed a *complex* fixture, not just the parameter varied.
- **Assert the sharding *spec*, not just the values.** A replicated run agrees with single-device to
  exactly 0.0, so "correct but silently unsharded" is invisible to value comparison.
- **Multi-device paths are testable on CPU** via `--xla_force_host_platform_device_count=4`. Use it for
  correctness only — timings under virtual devices are meaningless.
- **A guard on a sharding decision may be invisible single-device.** Deleting the `not partial_xcache`
  condition on `run_sqd`'s post-precompute reshard left all six in-process `TestPartialXCache` cases green
  and raised on every mesh (`get_xsource` needs `states` replicated). If a change touches resharding, add
  a `tests/_sharded_*.py` case and mutation-test it *there*; `conftest.run_sharded_child` is the driver.
- `examples/scaling/poc7_sharding.py` is the fuller sharding harness. Run it after any change to
  `ground_locg`'s reductions or helper signatures, not just after touching `sqd`.
- `svsim` requires **`mesh.size` to divide `2^num_qubits`** — documented rather than fixed, since a
  state vector's indices *are* the basis states and cannot be padded. `PartitionSpec(None)` replicates.

## Architecture

Seven largely independent modules under `rqutils/`; nothing but `sqd.py → {paulis/symplectic.py,
ground_locg.py}` and `qprint.py → paulis/general.py` couples them.

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
- `cache_level=(source_indices, diagonals)` selects among six matvec strategies. The source-index axis
  is near-free to enable and very expensive to disable — **prefer `cache_level[0] = 1`** (`get_xsource`
  setup is 66–97% of a solve; see `NOTES.md`). That figure is **weighted by call count** — the cost of
  paying the `J`-fold search per matvec against once — and is *not* headroom for accelerating the
  precompute, which is only 4.5–8.4% of a `(1,*)` solve. Reading it as the latter is a recorded trap
  (`NOTES.md`, "Two percentages that look contradictory"). **`xcache_groups=J'` makes that axis a
  dial** rather than a switch, caching `J'` of the `J` X groups; `None` (the default) is byte-identical to before. Two
  things measured after it shipped, both in `NOTES.md`: an *intermediate* `J'` can **raise** peak memory
  (two kernels instead of one — 9.0 MB against 10.4 MB at `J=16`), and **which axis dominates is set by
  `K`**, the Z signatures per X group — a property of the Hamiltonian, not the subspace. At `K=1` the
  source cache leads; at `J=101, K=100` (1D Heisenberg, n=100) `diag_signs` is 1313 B/slot against its
  404, so `(0, 0)` measures 4.0 GB against `(1, 1)`'s 60.6 GB and `cache_level[1]` is the bigger lever.
  **Quote `K` with any memory figure and never size from `4 * J * N` alone** — a fit taken at `K=1` was
  3.0x wrong for `(1, 1)`. `states_size` also rounds **up to a power of two**: N=24M allocates 33.6M
  slots.

  The six strategies are one kernel indexed by that 2×3 grid, reached two ways. The **public `apply_h`
  is keyword-only**: name the arrays you have and the strategy follows (see "Known rough edges" for the
  break this was). Internally `run_sqd` calls the private `_apply_h_kernel` with an assembled tuple and
  `cache_level` bound via `functools.partial` — it **must** stay static there, since `ground_locg`
  splats `args` positionally and `static_argnames` would never see it.
  `states_size` exists solely to pin array shapes and prevent JIT recompilation.

**The rule for a new solver option on `run_sqd`:** forwarded to `ground_locg` by *keyword* → put it
in `static_argnames`; riding inside the positionally-splatted `args` → bind it with
`functools.partial`. `cache_level` is the second case (hence the note above), `prefilter` the first —
so the `static_argnames` list is not evidence that `cache_level` could join it.

`sqd` forwards `prefilter` but **not `precond`** — preconditioning `sqd` is a closed investigation
(see below), so that gap is deliberate, not an oversight to fix. The prefilter's published 1.88x is
`ground_locg`-on-dense-operators; **it is unmeasured through `apply_h`**, so treat it as
off-by-default plumbing, not a recommended setting. Malformed values are rejected by
`_check_prefilter`, since the filter's own `degree > 1 and cycles > 0` gate would absorb them as a
silent no-op.

**`states` must be lex-sorted** — `get_xsource` is a binary search, not a sort. Always required (the
sort was equally wrong on unsorted input) but previously undocumented; `hproj(unique_states=True)`
skips its `np.unique` and so can violate it. Two paths selected statically on width: `uint64` keys for
`B ≤ 8` bytes, explicit lexicographic search beyond — a **correctness** boundary, not a performance
one.

**`vinit_from_min_diag`'s added weight must carry the seed component's own sign.** A bare `+1.0`
subtracts where the seed is negative, and `_spread_seed` maps index 0 to *exactly* −1.0 at every
`states_size`, so `argmin(diagonal) == 0` cancelled it to zero and `sqd` returned a wrong eigenvalue
with `converged=True` (`NOTES.md`). Don't replace `jnp.sign` with `copysign` — the seed is complex
whenever the coefficients are.

**`run_sqd`'s initial vector is a deterministic pseudo-random spread (`_spread_seed`), not a one-hot.**
Don't "simplify" it back — a one-hot cannot leave its connected component, so a subspace whose
Hamiltonian splits into disconnected blocks silently returned that block's minimum with
`converged=True`.

**`ground_locg.py`** — single-vector (block-size-1) LOBPCG specialization used as `sqd`'s eigensolver,
with the Rayleigh–Ritz step solved analytically (`eigenpair_2x2`, `eigenpair_3x3` via Cardano) instead
of via `eigh`, to keep memory down for huge vectors. It is sharding-transparent **only if the `mat`
callable preserves output sharding** — that contract is why every `apply_*` in `sqd.py` passes
`out_sharding=jax.typeof(vec).sharding`. It also takes an optional `prefilter` (`None` |
`(degree, cycles)` Chebyshev, use `(32, 2)`; its call site must stay **after** the dtype promotion and
**before** `body_iter0` — `docs/locg-chebyshev-prefilter.md` has the tables). **There is no `precond`
argument** — it was removed 2026-08-28; see below.

**`prefilter` needs `prefilter_hi`, a true upper bound on `λ_max`, and there is no way to compute one
from matvecs.** Kuczyński–Woźniakowski (1992) prove it; a block-diagonal operator with a start vector
in one block hides the other block entirely. The array path derives Gershgorin `max_i Σ_j |A_ij|`
automatically; a **callable raises** without it; `sqd` passes `Σ|c_k|`. Don't "fix" a missing bound
with a power-iteration or Lanczos estimate — that is exactly the defect in
`docs/rqutils-prefilter-bug.md`, where the estimate returned an **excited** eigenpair with
`converged=True`. Prefer a loose bound: over-estimating degrades resolution smoothly, under-estimating
changes the answer.

**`precond` is gone** (2026-08-28), with its tests and `examples/scaling/poc10_deflation_precond.py`.
It worked — 2.76× median on a *positive-definite* operator — but no `sqd` caller can reach that: `sqd`
solves the raw indefinite projected `H`, and on it literal Jacobi **fails to converge** (8000-iteration
cap, wrong answer). Don't reintroduce it as a fallback for a missing `prefilter_hi`; that trades a clean
`ValueError` for a silent wrong answer. `NOTES.md` has the measurements and the one route that would
work (shift-then-Jacobi, 1.45×, dominated by the prefilter).

**`sqd` defaults to `prefilter=(32, 2)`** — 1.49× median end-to-end (min 1.15×), and `sqd` derives the
required bound itself. Quote *that* figure, not the 2.43× dense wall-clock or the 5.02× iteration count.
`ground_locg` still defaults to `None`, since it cannot derive a bound from a callable.

**A validator belongs in the module that owns the gate it compensates for.** `_check_prefilter` lives
here, not in `sqd.py`, because `_chebyshev_prefilter`'s `degree > 1 and cycles > 0` is what silently
absorbs a malformed value; `sqd.py` imports it. Sited wrongly, the *published* entry point is the
unguarded one.

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

**`product.py` is not on this branch** (see the branch note above). Mentioned only because
`docs/rqutils-precond-request.md` and `docs/sdp-lower-bound.md` reference it: both values it returns are
*upper* bounds on the true ground energy, which is why the whole preconditioner-shift line is closed.

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
HAS_X = True` at module top, every use guarded by `HAS_X and isinstance(...)`, and the runtime path
raises a `RuntimeError` rather than failing at import. `numpy`, `scipy`, `h5py`, and **`jax`** are hard
dependencies (`pyscipopt` is not — it is only on the stale `product` branch).

**A type alias must name every arm in its `type` statement — never widen one afterwards with
`X |= OptionalType` under a `HAS_*` guard.** A `type` statement is evaluated *statically*, so the
augmented assignment mutates only the runtime object and the added arm is invisible to a type checker
whether or not the optional package is installed. All three aliases had this (`sqd.HamiltonianInput`,
`svsim.CircuitInput`, `qprint.PrintReturnType`); every correct `sqd(SparsePauliOp, ...)` call was an
`invalid-argument-type` error for a downstream caller who type-checks. Naming the arm unconditionally is
safe because a `type` statement is **lazy** — nothing reads `__value__`, so an annotation-only import is
never resolved at runtime. Import that arm under `TYPE_CHECKING` if it is not already needed at runtime
(`sqd`), and leave the existing runtime import alone if it is (`svsim` needs it for an `isinstance`).

Note **this repo cannot see that defect itself**: `invalid-argument-type` is `ignore` in
`[tool.ty.rules]`, so `ty check` passes against the broken form. The three regression tests re-enable
the rule per-invocation via `ty check -c` (`conftest.assert_type_checks`), and pin the
without-the-package import too (`conftest.assert_imports_without`). Two traps if you write another:
`ty` silently checks **nothing** for a file outside the project root, and it resolves an alias to an
opaque `TypeAliasType` without evaluating its value, so no runtime assertion can pin this.

**Don't index a sharded array to read one element.** `seed.at[i].add(f(seed[i]))` is correct
arithmetic but emits an `all-gather` per read — measured 3 on a 4-device mesh, each materializing the
whole vector on every device, which is what `ground_locg`'s single-vector budget exists to avoid. Use a
`broadcasted_iota` mask and an elementwise `where` instead; bit-identical, no collective. Check with
`.lower(...).compile().as_text().count("all-gather")`, not by reading the source.

**Sharding is implicit** — the library reads `jax.sharding.get_abstract_mesh()`; it is the *caller's*
job to set the mesh. The examples establish the expected pattern: a single axis named `'x'` with
`AxisType.Explicit`, plus `jax.config.update('jax_enable_x64', True)` (without x64 you silently get
complex64/int32 — you'll see truncation warnings).

**Code comments should be concise.** One line where one line will do; a short block only for a
non-obvious invariant or a defect the comment is there to prevent recurring. Prefer stating the
constraint over narrating the code — if a comment restates what the next line plainly says, delete it.
Long explanations belong in the docstring (user-facing) or `NOTES.md` (evidence, measurements,
post-mortems), not inline.

**A comment must earn its length; length alone is not the test.** Ask what a reader loses if it is
deleted. A block that records a measured defect earns any length — `sqd.py`'s 27 lines over one
statement carry four silent-wrong-answer failures with their measured values, and deleting them removes
the only thing stopping recurrence. A block that re-explains what a `docs/` file or `NOTES.md` already
says earns nothing at any length. Measured 2026-08-29: 41 blocks in `rqutils/` and `tests/` are longer
than the code beneath them, and the ones inspected were all the first kind. **Do not trim by ratio.**

Two failure modes to check for instead, both of which produced real bloat here:

- **Editing by appending.** Revisiting a comment and adding a paragraph instead of rewriting the
  existing one. `[tool.ty.rules]`'s comment reached 22 lines over 4 lines of config across three
  visits, with the same `jax.jit` explanation stated twice in one block.
- **Restating a document that already exists.** Before writing an inline block, check whether the
  content belongs in — or is already in — `NOTES.md` or `docs/`. That same comment duplicated
  `docs/typing-notes.md` in full.

The same applies to a rule stated near-identically in two places: prefer one statement plus a pointer.

**Docstrings feed the published API reference.** Every module opens with a raw docstring that is a full
reST document: over/underlined title, `.. currentmodule::`, prose with `.. math::` derivations of the
normalization conventions, and an explicit API section (`.. autofunction::` / `.. autoclass::` /
`.. autosummary::`). Function docstrings are Google-style (`Args:` / `Returns:` / `Raises:`) via
napoleon. Adding a public module requires **both** those directives **and** a manual line in the
`toctree` of `docs/source/index.rst`.

**Docstrings with LaTeX must be raw strings**, `.. autoclass::` needs `:members:`, `:math:`
exponents must be braced, and a docstring's body indentation must be **uniform** — writing
8-space continuations into a 4-space docstring makes reST read the deeper lines as a block quote
and nests `Args:`/`Returns:`/`Raises:` beyond napoleon's reach. Four hazards that no tool
catches: ruff, `ty`, pytest and the control-char sweep all pass. Detailed with both check
commands in `NOTES.md`.

**Writing `Raises:` sections finds bugs** — it caught three wrong claims in one pass. Trigger every
raise you document (`NOTES.md`).

**When measuring: use `eigvalsh` or sparse `eigsh(k=1)`, never `eigh`** (77 s vs 0.02 s at the sizes
here), **A/B whole calls against a worktree of the pre-change revision** rather than timing a predicate
in isolation, **A/B both arms warm** — changing a traced expression invalidates the compilation cache,
and one cold run measured 125 s against a warm 20 s, which reads as a catastrophic regression and is
not one — and **verify the referent of a cross-reference, not just that it resolves**. All four with
numbers in `NOTES.md`.

**Pass arrays as arguments to a `jit`ted benchmark, never close over them.** XLA constant-folds a
closed-over input, so the work happens once at trace time and every later call measures nothing —
0.175 ms against a true 40.8 ms for `uniquify_states` at N=200k, a 233x phantom speedup. The tell is a
`slow_operation_alarm ... Constant folding an instruction` line in the log, not an error.

**For memory and compilation counts, ask XLA rather than a formula.**
`fn.lower(*args).compile().memory_analysis().temp_size_in_bytes` gives peak temp bytes and
`fn._cache_size()` counts compiled variants. Both caught claims a byte count got wrong: `4 * J' * N`
predicts a linear saving where the measured peak *rose* (9.0 MB against 10.4 MB).

**A broken arm flatters its own benchmark.** An undersized capacity, a dropped term, a truncated
candidate list all do *less work*, so they report a *better* number. Two instances: a 16384-slot cap
against 17913 candidates reported 5.7x where the honest figure was 8.0x, and a `vmap` whose batch axis
came out last read as "numerically wrong" until the shapes were compared. Verify the output before
quoting the time.

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
  non-symmetric projection. Both halves of "uniquified and lex-sorted" are load-bearing (see the
  binary-search note above); the check is host-side numpy at 12–14% of `hproj`, on that opt-in path
  only. Pass `np.unique(states, axis=0)`, or leave `unique_states=False`.
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

- **Preconditioning `sqd` is closed, and `precond` has been deleted** (2026-08-28). It shipped and
  measured 1.79–2.76× median on a *shifted* operator, but `sqd` solves the raw indefinite projected
  `H`, where literal Jacobi **fails to converge**; six routes to a usable shift were measured and
  rejected — including a level-1 SDP bound that is valid, tightest-available, and still matched by a
  free `O(N)` diagonal bound. The blocking argument is structural: no bound on `H`, however tight, can
  reach the shift those figures used. The two-level/deflation candidate the record called "untried" was
  then tried and **rejected** (0.68–0.98×, `docs/deflation-preconditioner.md`), so the line is fully
  closed. `docs/rqutils-precond-request.md`,
  `docs/sdp-lower-bound.md`, summary in `NOTES.md`.
- **Subspace selection by weight shell + diagonal ranking was measured, then rejected** (2026-08-25) —
  sound results against a *uniform random* baseline, which is not what a real SQD workflow produces.
  Do not build on it.
- **The Bloom pre-filter for `get_xsource` is closed** (2026-08-30). Six prototypes measured it
  thoroughly and every mechanic was settled — capacity policy, sharding, hoisted precompute,
  composition with `xcache_groups` — and it is still not worth building, for two structural reasons.
  **It can only attach to the precompute** (the retry policy is host-side sequencing and cannot live
  inside one `jit`; the uncached recompute at `sqd.py:1876` is inside `_apply_h_kernel`'s scan, called
  every iteration), and that precompute is **4.5–8.4%** of a `(1,*)` solve, so Amdahl caps it at
  **1.09×**. And **the path that would save memory is the one it cannot reach**: `(0,0)` already costs
  4.0 GB against 4.1 GB with the filter, so the filter is 0.029 GB of overhead and no saving. The
  published 5.6–9.4× figures are matvec-path, not end-to-end. Three variants were separately rejected —
  as the subspace *definition* (dead past n≈70, FP count scales `2^n`), for input dedup (0.64–0.79×
  against `np.unique`), and binary fuse (better query, ~60 s build at N=24M).
  `docs/xsources-cache-budget.md`, detail in `NOTES.md`. **The memory lever is the diagonal axis, not this one** — `diag_signs` is
  1313 B/slot against `xsources`' 404 at `J=101, K=100`, and it is uninvestigated.
