# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**This file holds only the rules.** Every measurement, post-mortem, rejected alternative and dated
record lives in **`NOTES.md`** — read it before changing anything here that looks redundant or
over-engineered, because most of it is load-bearing for a reason someone had to find the hard way.
When a rule needs evidence, it points there rather than restating it.

## Environment

- **Always `uv run python`**, never bare `python` — the venv at `.venv` is managed by uv.
- **`qiskit` is a required dependency**; `mpl`, `qutip`, `mpi`, `docs` and `dev` (pytest + ruff + ty)
  are extras and are **not** installed by default. Pull those in per invocation:
  `uv run --extra mpl python examples/bench.py`. `qiskit` is required because it is a public *input*
  type (`PauliSumXZ.from_paulisum` takes a `SparsePauliOp`, `svsim` takes a `QuantumCircuit`) — the
  `HAS_QISKIT` guards stay regardless, since they turn a broken install into a `RuntimeError` at the
  call site rather than an `ImportError` at `import rqutils`. The `qiskit` extra survives as an **empty
  alias** so the `--extra qiskit` invocations throughout this file, `NOTES.md`, `docs/` and the
  `examples/scaling/` docstrings keep resolving; it installs nothing.
- **Every multi-process run needs `--extra mpi`.** `mpi4py` is an extra rather than a dependency because
  it builds against the host MPI, so requiring it would make every install depend on a system MPI, and
  nothing under `rqutils/` imports it — only `examples/sqd.py --gpus mpi` and
  `examples/scaling/*.py --devices mpi` do, via
  `jax.distributed.initialize(cluster_detection_method="mpi4py")`. `mpirun … --devices mpi` without the
  extra raises at that call.
- **No `timeout` on macOS** (it is GNU coreutils) — use the Bash tool's own timeout.
- **The shell is fish: quote grep globs** (`--include="*.py"`). Unquoted, fish fails with
  `(eval):1: no matches found` *before* grep runs, which reads as "no results" rather than an error.
- **macOS `sed -i` needs an explicit backup arg** (`sed -i '' 's/x/y/' f`). Without it BSD sed reads the
  filename as the suffix and fails — **while exiting 0**, so it looks like a successful no-op. Prefer a
  short `python3 -` heredoc for in-place edits, and assert the match count before writing.
- **Don't put `cd` in a backgrounded command.** "Session cwd remains" applies to *subsequent* commands,
  not the backgrounded one, so `cd /tmp && uv run ...` leaves the project and fails with `No module
  named jax`. Use `uv run --directory <project>`.

### Git

- **No network access**: `git fetch` fails. `origin/*` is whatever the last fetch left, so
  `rev-list --left-right --count` can report `0 0` for commits that were never pushed — and the reverse.
- **`git branch -r --contains <sha>` plus `git reflog show origin/<branch>` is the reliable check**
  before amending, rebasing, or any force-push. The reflog distinguishes `update by push` from a fetch.
- **`dev` is the only live branch.** `main` is the PR target and a strict ancestor holding nothing
  unique. `metal` and `product` are stale reference branches — do not merge them, and do not delete
  them: `metal` alone holds the segment-sum diagonal builder for ragged X groups. Both are *ahead* of
  `dev` in commit count, which makes them look like pending work; they are not.
- **Integrate with `git merge --ff`**, not `--no-ff`. Older history shows merge commits; that is not the
  convention to follow.
- **Worktrees need `worktree.baseRef: "head"`** (already set in `.claude/settings.json`). The default
  `fresh` branches from `origin/main`, which predates the `dev` extra, so `uv run --extra dev` fails
  outright.

### Docs

```bash
cd docs && uv run --extra docs make html    # output in docs/build/html
cd docs && uv run --extra docs make clean   # also removes source/apidoc (not committed)
```

**Sphinx caches: a rebuild prints no warnings even when they exist.** `make clean` first, or a docstring
regression reads as a clean build. Note `grep -c warning` on the output counts the "build succeeded, N
warnings" summary line too.

## Linting and type checking

```bash
uv run --extra dev ruff check rqutils/ tests/ examples/     # lint
uv run --extra dev ruff format rqutils/ tests/ examples/    # format (line width 100)
uv run --extra dev ty check rqutils/ tests/ examples/       # type check
```

All three are clean; keep them that way — **from a venv without the `mpi` extra**, which is what those
commands give. The two `# ty: ignore[unresolved-import]` on `mpi4py` are required there and are reported
as *unused* if `mpi4py` happens to be installed, so `ty check` cannot be clean in both states and the
no-`mpi` one is the contract. Don't "fix" that warning by deleting the suppressions: it breaks `ty` for
every normal install. (A global `unused-ignore-comment = "ignore"` is the wrong trade too — that rule is
what flags the other six suppressions going stale.)

Config is in `[tool.ruff]` / `[tool.ty.rules]` in `pyproject.toml`, and every suppression carries the
reason it exists — read those comments before adding another. Pre-commit runs only whitespace/EOF/YAML/large-file hooks, not ruff or ty. Notebooks are
excluded from both: they get names from IPython magics that static analysis cannot see.

- **A global ignore hides real defects; prefer a per-line `# ty: ignore[rule]`.** Several `ty` rules are
  `ignore`d because JAX ships almost no type information and numpy's stubs are stricter than the
  dtype-generic `npmod` convention allows — but prefer fixing a real finding over widening the list.
  Count first with `ty check -c 'rules.X="error"'`, then read the diagnostics rather than the count.
  The two patterns that work are a per-line suppression where the stub is genuinely wrong, and
  `@overload` where a runtime flag picks the return shape.
- **New scripts under `examples/` trip rules the library does not**: **B023** (a `lambda` in a `for` loop
  capturing the loop variable — endemic to benchmark harnesses; fix by binding as a default arg,
  `lambda vec=vec: ...`) and **E402** (imports after the mandatory
  `jax.config.update('jax_enable_x64', True)`, needing `# noqa: E402`). `ruff --fix` resolves neither.

## Testing

```bash
uv run --extra dev pytest              # whole suite
uv run --extra dev pytest -v -x        # verbose, stop at first failure
```

**Run the full extras** — `--extra dev --extra mpl --extra qutip` — or tests **silently skip**. The
qiskit reference comparisons this file treats as the trustworthy oracle no longer need an extra, since
`qiskit` is a required dependency; `mpl` and `qutip` still do. A fresh worktree gets a bare venv, so
this bites there first.

**Don't run the suite for a markdown-only change.** `testpaths = ["tests"]` and
`python_files = ["test_*.py"]`, so pytest never looks at `*.md` — the result is known before it runs, and
a green run that could not have been red dilutes the signal. `git status --short` is the check. Ruff and
`ty` likewise only target `rqutils/ tests/ examples/`. **Docstrings are the exception**: they live in
`.py`, so editing one *is* a code change for `ty` and the docs build.

`tests/conftest.py` enables `jax_enable_x64` before any `rqutils` import — every tolerance depends on it
— and holds the shared reference helpers, each validated against qiskit before being trusted. It also
configures caches taking the suite from ~53 s to ~6 s; expect ~53 s cold. One `tests/test_<module>.py`
per module. `tests/_sharded_*.py` are subprocessed under
`XLA_FLAGS=--xla_force_host_platform_device_count=4` (the device count must be set before jax
initializes); the leading underscore keeps them uncollected, as do the scratchpad notebooks.

### Writing tests

- **Tests are organized by defect.** Name the defect a new test locks down and record the measured wrong
  value. Prefer an *independent* reference (dense construction, scipy, qiskit) over self-consistency:
  several bugs made every internal path agree on the same wrong number.
- **A "does X change the answer?" test needs an arm where X is truly absent.** Verify that; don't infer
  it from the parameter being unset.
- **Fixtures are built inside each test body; there are no `@pytest.fixture` state generators.**
  Deliberate — tests pick a seed for a specific pathology and assert the fixture still has it, and
  moving draws into fixtures makes RNG stream position depend on fixture ordering. Keep new generators
  as plain functions taking `rng` **last**: `real_pauli_strings(num_qubits, count, rng)` (strings only —
  draw coefficients separately), `unique_states(num_draws, num_qubits, rng)`.
- **Sweep `cache_level`, don't sample it** — bugs hide behind the default `(1, 0)`, and one needed a
  *complex* fixture rather than just the parameter varied.
- **A physically-motivated fixture can invert a conclusion a synthetic one reaches.** Build it from the
  Hamiltonian's own structure; `examples/scaling/poc12`'s `xxz_krylov` is the pattern. And **check the
  fixture exercises the thing under test** — a hop on qubits 0-1 gives a band-limited subspace a 0% hit
  rate, so the search is never called.
- **Don't assert an exact float boundary against a separately computed value.** Two summation orders
  over the same numbers differ in the last ulp, so an `x == threshold` arm tests floating-point
  associativity rather than the code. Test strictly inside the region.

### Mutation testing

**Verify a new test fails against the bug it targets by reverting the fix in place.** A copy of the repo
does not work — the venv holds an editable install pointing at the original. Mutate `@jax.jit` code in a
**fresh subprocess** or both arms reuse one compiled kernel. `NOTES.md` has the recipe.

**Undo a mutation with `cp` from your own backup, never `git checkout <file>`.** The file usually holds
unrelated uncommitted work; a checkout discards all of it, and the suite passes either way so nothing
flags it. The restore step is the half that matters.

Four reasons a mutant survives that are *not* missing coverage:

- **Wrong layer or branch.** Check which branch your fixture reaches before concluding anything.
- **Fixture too small.** If the defect is in how something *scales*, the fixture must span that axis —
  magnitude, not just code path. A threshold thousands of times looser than the correct one is still
  satisfied by the solver's own overshoot on a small fixture.
- **Compounding guards.** Some guards are only reachable when other defects coincide. Look for a more
  direct assertion — the invariant itself, or `debug=True` diagnostics — before recording a negative.
- **The option is a genuine no-op on that fixture.** To prove an option does nothing, compare traced
  graphs (`jax.make_jaxpr` string equality), not energies: a *working* option can return a bit-identical
  energy, so "same as baseline" is satisfied by both arms and pins nothing.

### Sharding tests

- **Multi-device paths are testable on CPU** via `--xla_force_host_platform_device_count=4`. Correctness
  only — timings under virtual devices are meaningless.
- **Virtual devices cannot test multi-*process* at all** — they are one process, so every device is
  addressable and the entire class of "spans non-addressable devices" errors is unreachable. That is how
  `sqd` shipped unable to return its own eigenvalue on a 4-node mesh (`float()` on a rank-0 array whose
  sharding names the whole mesh). **Anything reading a device value on the host needs a real
  multi-process run**, or `_host_scalar`-style handling by construction. `NOTES.md` has the four wrong
  gathers this produced.
- **`addressable_shards` is topology-dependent, so assert its length.** It holds *every* shard when one
  process owns the mesh and only *this rank's* when it does not, so concatenating it is exact
  single-process and **silently partial** across processes — measured, 50782 rows against a true 157051,
  no error raised. Compare against `sharding.num_devices` before trusting it.
- **A collective inside a conditional deadlocks.** `process_allgather` is whole-world, so a rank that
  skipped the branch never arrives and the job dies on a shutdown barrier at `2/4 tasks`. Prefer a local
  read, or scope the collective to the sub-mesh (`jax.jit(out_shardings=P())` under
  `jax.set_mesh(arr.sharding.mesh)`).
- **Assert the sharding *spec*, not just the values.** A replicated run agrees with single-device to
  exactly 0.0, so "correct but silently unsharded" is invisible to value comparison.
- **A guard on a sharding decision may be invisible single-device.** If a change touches resharding, add
  a `tests/_sharded_*.py` case and mutation-test it *there*; `conftest.run_sharded_child` is the driver.
- **`examples/scaling/poc7_sharding.py` is the fuller harness.** Run it after any change to
  `ground_locg`'s reductions or helper signatures, not just after touching `sqd`.
- **`svsim` requires `mesh.size` to divide `2^num_qubits`** — documented rather than fixed, since a state
  vector's indices *are* the basis states and cannot be padded. `PartitionSpec(None)` replicates.

## Measuring

- **Use `eigvalsh` or sparse `eigsh(k=1)`, never `eigh`** — 77 s vs 0.02 s at the sizes here.
- **A/B whole calls against a worktree of the pre-change revision**, not a predicate in isolation. A
  predicate microbenchmark has twice reported a regression that whole-call timing showed to be zero.
- **A/B both arms warm.** Changing a traced expression invalidates the compilation cache; one cold run
  measured 125 s against a warm 20 s, which reads as catastrophic and is not.
- **Pass arrays as arguments to a `jit`ted benchmark, never close over them.** XLA constant-folds a
  closed-over input, so the work happens once at trace time — 0.175 ms against a true 40.8 ms, a 233x
  phantom speedup. The tell is a `slow_operation_alarm ... Constant folding` line, not an error.
- **Ask how many times per solve the target is paid, before believing any ratio.** A one-off precompute
  is 4.5–8.4% of a solve, so Amdahl capped one 5.6–9.4x optimization at 1.09x; a per-matvec target
  consumed ~129 times kept most of its 5.0–8.2x. Same solver, opposite outcomes, one discriminator.
- **A quantity measured at one size is not a law.** Sweep the parameter before calling something flat,
  and prefer a ratio of two measured terms over an absolute.
- **For memory and compilation counts, ask XLA rather than a formula.**
  `fn.lower(*args).compile().memory_analysis().temp_size_in_bytes` and `fn._cache_size()`. Byte-count
  formulas have predicted a saving where the measured peak *rose*.
- **A broken arm flatters its own benchmark.** An undersized capacity, a dropped term, a truncated
  candidate list all do *less work* and so report a *better* number. Verify the output before quoting
  the time.
- **Verify the referent of a cross-reference, not just that it resolves.**

## Architecture

Eight modules under `rqutils/`, largely independent: only `sqd.py → {paulis/symplectic.py,
ground_locg.py}` and `qprint.py → paulis/general.py` couple them. `_types.py` and `math.py` are support.

### Two unrelated Pauli representations — do not confuse them

**`paulis/general.py`** — dense generalized (Gell-Mann-like) basis for arbitrary dimension. Public
surface is `paulis(dim)`, `pauli_matrices(dim)`, `components()`, `labels()`.

- Normalization is `tr(λ_k λ_l) = 2δ_kl`, so **`λ_0 = sqrt(2/n)·I`, not `I`** — the most bug-prone
  invariant here.
- Basis-index ordering is fixed by a shell-by-shell construction loop, and `components`/`labels` both
  index by basis position, so a reordering would disagree with the labels users read while every
  function stayed self-consistent.
- Shapes: `paulis(dim)` → `(d1², …, D, D)` (basis axes *first*); component arrays → `(…, d1², …)`
  (component axes *last*). Everything is memoized keyed by a `tuple(int)`-normalized `dim`.

**`paulis/symplectic.py`** — `PauliSumXZ`, a bit-packed qubit-only form for JAX/GPU. Convention
`Q = (-i)^{x·z} Z^z X^x` with **little-endian qubit ordering** (Qiskit's `.x`/`.z` are reversed on
ingest). Terms are grouped by unique X signature, Z groups zero-padded to a rectangle, the
`(-i)^{popcount(x&z)}` phase folded into the coefficients, then `np.packbits`.

- Signatures always reserve **one pad bit at position 0**, aligning with the pad bit consumers put in
  their state bitstrings — not optional, so alignment cannot be got wrong.
- Decoding a packed signature back to an integer must shift by `8*nbytes - (num_qubits + 1)`; dropping
  the `+1` silently returns a *permutation*.
- **A complex coefficient raises** (non-Hermitian); there is no `force_real` flag and none could work.
  Check `.c.dtype` if you need float64.

### `sqd.py` — sample-based quantum diagonalization

Project a Pauli-sum Hamiltonian onto the subspace spanned by computational-basis bitstrings and solve
matrix-free. `sqd(...)` is the entry point, `hproj(...)` the dense/debug path.

**Return shapes.** `sqd` returns 3 values with `return_eigvec=True` (`eigval, eigvec, basis`) and a bare
`float` otherwise — not a 5-tuple; the convergence flag and subspace dim are consumed inside `sqd`,
which raises on non-convergence. `hproj` returns a scipy `csr_array`, so `np.asarray()` on it yields a
**0-d object array** and the failure surfaces frames later as `IndexError`. Use `.toarray()`.

**Two invariants that produced silent wrong answers:**

- **States carry one extra zero pad bit at position 0** before `packbits`, aligned with
  `PauliSumXZ`'s by construction. Filler slots from uniquification are `255`, detected via
  `states_u[:, 0] >> 7`.
- **`states` must be lex-sorted** — `get_xsource` is a binary search, not a sort. Always required;
  `hproj(unique_states=True)` skips its `np.unique` and so can violate it. Two paths selected statically
  on width (`uint64` keys for `B ≤ 8` bytes, explicit lexicographic beyond) — a **correctness** boundary,
  not a performance one.

**`cache_level=(source_indices, diagonals)`** selects among six matvec strategies, one kernel indexed by
that 2×3 grid.

- **Prefer `cache_level[0] = 1`.** `get_xsource` setup is 66–97% of a solve — a figure **weighted by call
  count** (the `J`-fold search per matvec against once), *not* headroom for accelerating the precompute,
  which is only 4.5–8.4% of a `(1,*)` solve. Misreading it as the latter is a recorded trap.
- **Never select `cache_level[1] = 1`; use 0 or 2.** It is slower than `[1] = 0`, which stores *nothing*,
  because unpacking the cached bits costs about what recomputing the parity costs — it buys bytes and
  redoes the time. It stays in the API because the sweep is load-bearing, not because it should be
  chosen.
- **Which axis dominates is set by `K`**, the Z signatures per X group — a property of the Hamiltonian,
  not the subspace. **Quote `K` with any memory figure and never size from `4 * J * N` alone.**
  `states_size` also rounds **up to a power of two**.
- **`diag_signs` is not a compression target** — it is already one bit per (state, Z term), so its size
  is information-theoretic, not slack.

**The public `apply_h` is keyword-only**: name the arrays you have and the strategy follows. Internally
`run_sqd` calls the private `_apply_h_kernel` with an assembled tuple and `cache_level` bound via
`functools.partial` — it **must** stay static there, since `ground_locg` splats `args` positionally and
`static_argnames` would never see it. `states_size` exists solely to pin array shapes against JIT
recompilation.

**Rule for a new solver option on `run_sqd`:** forwarded to `ground_locg` by *keyword* → `static_argnames`;
riding inside the positionally-splatted `args` → bind with `functools.partial`. `cache_level` is the
second case, `prefilter` the first — so the `static_argnames` list is not evidence that `cache_level`
could join it.

**`sqd` forwards `prefilter` but not `precond`** — that gap is deliberate (see Closed investigations).
The prefilter's published 1.88x is `ground_locg`-on-dense-operators and **unmeasured through `apply_h`**,
so treat it as off-by-default plumbing rather than a recommended setting.

**Two initial-vector invariants, both of which returned a wrong eigenvalue with `converged=True`:**

- **`run_sqd`'s initial vector is a deterministic pseudo-random spread (`_spread_seed`), not a one-hot.**
  A one-hot cannot leave its connected component, so a Hamiltonian splitting into disconnected blocks
  silently returned that block's minimum.
- **`vinit_from_min_diag`'s added weight must carry the seed component's own sign.** A bare `+1.0`
  subtracts where the seed is negative. Don't replace `jnp.sign` with `copysign` — the seed is complex
  whenever the coefficients are.

**`states` must be replicated today, but that is not fundamental.** The `13 * N` per-device cost is the
one term the `(0,0)` floor cannot shed. Only `searchsorted` fails on a partitioned `[N, B]`; a
hash-ownership-plus-local-search design is verified bit-identical at 16× less per-device memory, and the
popcount diagonal path already shards with zero collectives. **Hash the whole key, not a prefix, and not
a range split.** `uniquify_states` is the exception — its output feeds a binary search so it must stay
globally lex-sorted, which needs *range* partitioning; that is built and bit-identical too. What remains
unverified is whether the routing pays, which needs a real interconnect. `NOTES.md` has the cost tables,
the balance measurements, the dead ends, and what is unbuilt. **A sort is the anti-pattern for sharding**
— only elementwise ops and reductions survive a partitioned axis.

### `ground_locg.py` — the eigensolver

Single-vector (block-size-1) LOBPCG specialization used as `sqd`'s eigensolver, with the Rayleigh–Ritz
step solved analytically (`eigenpair_2x2`, `eigenpair_3x3` via Cardano) rather than via `eigh`, to keep
memory down for huge vectors.

**Why LOBPCG rather than Davidson, measured against `diaglib`'s implementation of both.** Davidson's
footprint is *linear in history depth* (51 vectors/eigenpair at its competitive depth 25, against
`ground_locg`'s measured 8), and it buys its matvec advantage with that memory almost linearly. At
*matched* memory the two regimes split: Davidson is ~1.9x better on a diagonally dominant operator, and
~1.4x worse on a non-dominant one. `sqd`'s projected `H` is **not** diagonally dominant — the same fact
that closes `precond` — and `N` is the binding constraint, so this is the regime where LOBPCG wins.
Accuracy is comparable, with `ground_locg` better at the residual floor. Don't switch algorithms without
redoing that measurement on a physical Hamiltonian.

**Every guard in it is load-bearing and was measured**; `docs/locg.md` catalogues seven defects that each
failed *silently* (it is stale on scope and line numbers). Don't "simplify" the balancing, the
re-orthogonalizations or the zero-direction masks, don't unify `body_iter1`'s exclusion bound with
`body()`'s, and don't reintroduce the one-matmul `_compute_sas` form.

**The re-orthogonalization pass counts are fixed at 2 deliberately — don't make them adaptive.** Measured
`|⟨x|y⟩| ≤ 1.3e-16` across diagonal shifts 0–1e12, against the `diaglib` reference implementation's
`tol_ortho = 2·eps ≈ 4.4e-16`: a measured loop would exit at its first check every iteration and cost an
extra O(N) reduction to learn that. Nor can 2 drop to 1 — a well-conditioned fixture cannot tell them
apart, but `TestProjectOut` fails at once.

**Sharding-transparent only if the `mat` callable preserves output sharding** — that contract is why
every `apply_*` in `sqd.py` passes `out_sharding=jax.typeof(vec).sharding`.

**Convergence is `‖r‖ < max(atol, rtol·(‖Ax‖ + |θ|))`** — either arm suffices.

- `atol` is an absolute residual bound, default `0.0` (no absolute arm). **`None` is rejected**: a
  derived absolute bound is the unintuitive construct this replaced.
- `rtol` is a fraction of the operator magnitude — the `np.allclose` meaning, dimension-independent.
  **`None` is accepted** and resolves to `4·eps` of the *promoted* dtype, which cannot be a literal: a
  hardcoded float64 value exhausts a 500-iteration cap at float32.
- **Do not put an `n` factor in `rtol`'s scale.** It makes the bound exceed `‖A‖` at large `n`, so the
  first iterate reports convergence on an arbitrary eigenpair. The reason is the **disjunction**: a
  dimension-dependent arm is fatal here because either arm suffices, but harmless in a conjunction, where
  it can only tighten. `diaglib` ships `‖r‖/√n < tol AND max|r| < 10·tol` and is safe for exactly that
  reason — don't read its `√n` as licence for one here.
- **Both arms are guarded against accept-anything** (`rtol >= 0.5`, `atol >= Σ|c_k|`), and a below-floor
  `atol` raises **only when `rtol == 0`** — with a live relative arm it is harmless, and rejecting it
  would fire on correct input. The floor is `4·eps·Σ|c_k|`, exposed as `residual_floor`.

**`prefilter` needs `prefilter_hi`, a true upper bound on `λ_max`, and there is no way to compute one
from matvecs.** Kuczyński–Woźniakowski (1992) prove it. The array path derives Gershgorin automatically;
a **callable raises** without it; `sqd` passes `Σ|c_k|`. Don't "fix" a missing bound with a
power-iteration or Lanczos estimate — that defect returned an **excited** eigenpair with
`converged=True`. Prefer a loose bound: over-estimating degrades resolution smoothly, under-estimating
changes the answer. Its call site must stay **after** the dtype promotion and **before** `body_iter0`.
The prohibition is on a power-iteration estimate as a **filter bound**, not on the estimate itself: as a
*convergence-test scale* it is safe, since under-estimating there only tightens the test. Other
implementations use it exactly that way.

**`sqd` defaults to `prefilter=(32, 2)`** — 1.49× median end-to-end (min 1.15×). Quote *that*, not the
2.43× dense wall-clock or the 5.02× iteration count. `ground_locg` defaults to `None`, since it cannot
derive a bound from a callable.

**`debug=True` runs the full `maxiter`** — it switches `while_loop` to `scan`, so it does *not* stop at
convergence, and appends a dict of per-iteration diagnostics (`x`, `y`, `r`, `theta`, `rho`, `kappa`,
`sas`, `rtol_scale`, `converged`; 2 extra leading rows for the seed steps). With `atol=rtol=0` that makes
the residual *trajectory* observable, which is how a floor or plateau gets measured. `sqd` rejects that
pair as unsatisfiable; a direct `ground_locg` call does not validate, which is what makes it usable as an
instrument.

**A validator belongs in the module that owns the gate it compensates for.** `_check_prefilter` and
`_check_tols` live here, not in `sqd.py`, because this module holds the branches that would otherwise
absorb a malformed value silently. `sqd.py` imports them — and is the caller for `_check_tols`, being the
outermost point where `Σ|c_k|` is concrete (`run_sqd` is jitted, and a traced value cannot raise).

### `svsim.py`

JAX state-vector simulator. Gates compile to the same symplectic form and apply via a single
`jax.lax.scan`, so only `x, y, z, cz, rx, ry, rz, rzz` are supported — transpile to
`basis_gates=['rx','ry','rz','rzz']` first. **`sin` is complex128, not real** — it carries the rotation's
leading `i` and the convention's `(-i)^{x·z}` phase, and narrowing it silently breaks every `y`/`ry` gate
and so every transpiled circuit. `cz` is only decomposed on the `QuantumCircuit` path, and is correct
only up to a uniform `exp(iπ/4)`.

**`examples/svsim.py` is the repo's reference for a correct multi-process path**, and the only one that
was tested on real nodes before 2026-09-04: it writes `final_state.addressable_shards` per rank with
`h5py`, serialized by `MPI.COMM_WORLD` token-passing and gated on `jax.process_index()`, never fetching a
global array. Follow it rather than inventing a second convention. The asymmetry that let `sqd`'s bug
survive is worth remembering — `svsim` returns a large distributed array, so addressability had to be
confronted to write it out at all, while `sqd` returns a **scalar** that looks innocuous. **The dangerous
return type is the small one.**

### `qprint.py`

Pretty-printer with two orthogonal axes: `fmt` picks the content class (`QPrintBraKet` / `QPrintPauli` /
`QPrintMatrix`) and `output` picks the rendering (`'text'` returns the object for lazy `__repr__`,
`'latex'` a string, `'mpl'` a Figure). `QPrintBase` owns all numerics; subclasses override only
`_qobj_data`, `_add_labels`, `_format_lhs`. **Test the full `fmt` × `output` grid, not a diagonal** —
bugs have lived in cells nothing exercised.

## Conventions when editing

**`npmod`** — numeric functions take `npmod: ModuleType = np` as the last kwarg so callers can pass
`jax.numpy` for traceable execution. **Validation and early returns must be gated on `if npmod is np:`;
Python-level shape inference must NOT be** — it operates on static values and both backends need it
identically. Use `np` explicitly for shape arithmetic, not `npmod` (`jnp.sqrt` rejects the plain tuples
`array.shape` returns). Prefer an unrolled Python loop over `jax.lax.fori_loop` when the trip count is
static — a traced index cannot subscript a static dimension tuple. `components` is the only consumer
left; `TestNpmodParity` pins it, including under `jax.jit`.

**Optional dependencies** — uniform `try: import X / except ImportError: HAS_X = False / else:
HAS_X = True` at module top, every use guarded by `HAS_X and isinstance(...)`, and the runtime path
raising `RuntimeError` rather than failing at import. `numpy`, `scipy`, `h5py` and **`jax`** are hard
dependencies.

**A type alias must name every arm in its `type` statement — never widen one afterwards with
`X |= OptionalType` under a `HAS_*` guard.** A `type` statement is evaluated statically, so the augmented
assignment mutates only the runtime object and the added arm is invisible to a type checker. Naming the
arm unconditionally is safe because a `type` statement is **lazy** — import it under `TYPE_CHECKING` if
not already needed at runtime. **This repo cannot see the defect itself**: `invalid-argument-type` is
`ignore`d, so `ty check` passes against the broken form. The regression tests re-enable the rule
per-invocation via `conftest.assert_type_checks`. Two traps: `ty` silently checks **nothing** for a file
outside the project root, and resolves an alias to an opaque `TypeAliasType` without evaluating it, so no
runtime assertion can pin this.

**Sharding is implicit** — the library reads `jax.sharding.get_abstract_mesh()`; setting the mesh is the
*caller's* job. Examples establish the pattern: a single axis named `'x'` with `AxisType.Explicit`, plus
`jax.config.update('jax_enable_x64', True)` (without x64 you silently get complex64/int32).

**Don't index a sharded array to read one element.** `seed.at[i].add(f(seed[i]))` is correct arithmetic
but emits an `all-gather` per read, each materializing the whole vector on every device — which is what
`ground_locg`'s single-vector budget exists to avoid. Use a `broadcasted_iota` mask and an elementwise
`where`; bit-identical, no collective. Check with
`.lower(...).compile().as_text().count("all-gather")`, not by reading the source.

### Comments

**Concise: one line where one line will do.** A short block only for a non-obvious invariant or a defect
the comment prevents recurring. Prefer stating the constraint over narrating the code. Long explanations
belong in the docstring (user-facing) or `NOTES.md` (evidence), not inline.

**A comment must earn its length; length alone is not the test.** Ask what a reader loses if it is
deleted. A block recording a measured defect earns any length — deleting it removes the only thing
stopping recurrence. A block re-explaining what `NOTES.md` or `docs/` already says earns nothing at any
length. **Do not trim by ratio.** Two failure modes to check for instead:

- **Editing by appending** — revisiting a comment and adding a paragraph instead of rewriting the
  existing one, leaving two explanations of one statement and often a now-false opening sentence.
- **Restating a document that already exists.** Check whether the content belongs in `NOTES.md` or
  `docs/` before writing it inline.

**A rule stated near-identically in two places: prefer one statement plus a pointer.**

### Docstrings

**They feed the published API reference.** Every module opens with a raw docstring that is a full reST
document: over/underlined title, `.. currentmodule::`, prose with `.. math::` derivations, and an
explicit API section. Function docstrings are Google-style (`Args:` / `Returns:` / `Raises:`) via
napoleon. Adding a public module requires **both** those directives **and** a manual `toctree` line in
`docs/source/index.rst`.

Four hazards no tool catches — ruff, `ty`, pytest and the control-char sweep all pass:

- **LaTeX needs a raw string.**
- **`.. autoclass::` needs `:members:`.**
- **`:math:` exponents must be braced.**
- **Body indentation must be uniform** — 8-space continuations in a 4-space docstring make reST read the
  deeper lines as a block quote, nesting `Args:`/`Returns:`/`Raises:` beyond napoleon's reach.

**One `Args:` entry per parameter.** Documenting two in one entry leaves the second absent from the
rendered reference.

**Writing `Raises:` sections finds bugs** — it caught three wrong claims in one pass. Trigger every raise
you document.

## Known rough edges

Downstream-visible breaks and hard limits. There is no CHANGELOG; if one is added, all of these belong
in it.

- **`paulis(dim)`** for multiple subsystems uses `np.einsum` with 3 letters per subsystem, capping at ~17
  subsystems. `sparse=True` for products raises `NotImplementedError`.
- **`N ≤ 2^31 - 1` subspace states**, enforced (`_MAX_STATES`) in `sqd` and `hproj` *and* on
  `uniquify_states`' static `states_size`, where the int32 iota is created — `examples/scaling/` scripts
  call the un-underscored helpers directly and reach the iota with neither entry-point guard in the
  chain. `TestInt32Ceiling` covers both sides.
- **`hproj(unique_states=True)` raises on unsorted or duplicate-containing input**, where it used to
  return a silently non-symmetric projection. The check is host-side numpy at 12–14% of `hproj`, on that
  opt-in path only. Pass `np.unique(states, axis=0)` or leave `unique_states=False`.
- **`apply_h` is keyword-only**; the positional `(scanned, cache_level)` form raises `TypeError`. A hard
  break rather than a shim, so the six valid input sets become the only constructible ones. Bind the
  *arrays* for a matvec thunk (`functools.partial(apply_h, xsources=xs, diagonals=dg)`), not the
  `cache_level`.
- **`sqd(..., packed=True)` returns *packed* states.** One flag governs both directions, so a round trip
  needs no re-pack — which also removes a hazard, `pack_states` not being idempotent. A caller comparing
  the result against an unpacked array breaks loudly on the shape mismatch. Both overloads annotate
  `StateList`, which cannot express the width, so **`ty` will not catch a caller assuming the wrong one**.
- **`tol` is gone**, replaced by `atol`/`rtol` (see `ground_locg.py` above). `tol=` raises `TypeError`
  with no alias, deliberately: it meant *relative* in one revision and *absolute* in the next, so
  silently resolving it to one of the pair would be the worst option. `tol=x` on the absolute form is
  `atol=x, rtol=0.0`; on the relative form there is **no exact equivalent**. Also an **arity** change —
  `ground_locg`'s positional signature gained a slot.
- **Multi-process host reads go through `_host_scalar`.** `float(eigval)`/`bool(converged)` raise
  "spans non-addressable devices" on a multi-process mesh, because a reduction over a partitioned vector
  is a rank-0 array whose sharding still names the whole mesh. Fixed, but the shape recurs: `jax.reshard`
  does **not** help (the spec is already `P()` — the problem is addressability, not layout), and the fix
  must stay non-collective so a rank on another branch cannot deadlock. `main` and `metal` still carry the
  unfixed line (`product` predates the module and has no `sqd.py`).
- **`apply_h` with a real `vec` and complex128 coefficients raises** on the scan carry dtype.
  Pre-existing and reproduces single-device: an undocumented contract that `vec` be promotable to `.c`'s
  dtype.

## Closed investigations — don't reopen without reading `NOTES.md`

- **Preconditioning `sqd`**, and `precond` is deleted. It measured well on a *shifted* operator, but
  `sqd` solves the raw indefinite projected `H` where literal Jacobi fails to converge, and the blocking
  argument is structural: no bound on `H`, however tight, reaches the shift those figures used. Six
  routes measured and rejected, including the deflation candidate. Don't reintroduce it as a fallback for
  a missing `prefilter_hi` — that trades a clean `ValueError` for a silent wrong answer.
- **Reducing `ground_locg`'s `O(N)` vector count.** The working set is 7 vectors, measured, and 7 is the
  *algorithmic minimum* for a 3-dim Rayleigh–Ritz basis — so it is a basis-size question, not buffer
  reuse, and no aliasing work helps. The 2-dim variant converges but takes 3.2–11.9× more iterations for
  13.3% off the floor. The module docstring's "three-vector memory budget" means the Rayleigh–Ritz basis,
  **not** the total footprint.
- **Reduced precision in `ground_locg`, both variants.** f32 arithmetic with f64 storage is 0.42×
  end-to-end. f32 *storage* of a carried vector is closed by one measurement: rounding `ax` to f32 raises
  the residual floor 3.1e6× above tolerance, from catastrophic cancellation in `r = Ax - θx`, not a
  tunable tolerance. **The `(0,0)` floor is 120 B/slot and that is the honest single-device ceiling** —
  lowering it needs a solver with fewer carried `O(N)` vectors, not a dtype change.
- **The Bloom pre-filter for `get_xsource`.** Every mechanic was settled and it is still not worth
  building: it can only attach to the precompute, which is 4.5–8.4% of a solve, so Amdahl caps it at
  1.09×; and the path that would save memory is the one it cannot reach. Three variants separately
  rejected. **The memory lever is the diagonal axis, not this one.**
- **A partial *diagonal* cache is measured and works but is not implemented** — half the diagonal memory
  for 2.45×, bit-identical energies. Two constraints if it is ever built: **do not split both axes**, and
  **one compiled variant per distinct `J'`** (power-of-two rounding makes that *worse*). The overhead is
  4.0% of the memory the split returns, so the dial is only cheap at large `J`.
- **Subspace selection by weight shell + diagonal ranking.** Sound against a *uniform random* baseline,
  which is not what a real SQD workflow produces. Do not build on it.
- **The MLX port**, deleted rather than deprecated because the JAX solver measured faster even on the
  MLX GPU backend. Don't reintroduce a second solver implementation without that measurement going the
  other way first.
- **Reusing `Ax` to cut `body()`'s 3 matvecs to 2.** Measured 1.31–1.42× through `apply_h` (1.5× dense)
  and it makes the **reported residual understate the true one by 59×** in float32, because
  `r = ax − θx` cancels against the staleness the two terms share. **The error originates in the operator
  application, not the reconstruction** — an exact f64 combination of f32 images still drifts 1.43e-07 —
  so every technique that improves how the terms are combined works on the wrong stage. Sixteen arms
  measured (compensated sums, Dekker exact products, f64 widening, six orderings, cycling, fixed and
  triggered refresh, a carried error term, an `⟨x|r⟩` detector): **all sit at honesty
  `TRUE‖r‖/reported‖r‖` = 6.5–13.6× against baseline's 0.95**, and arithmetic accuracy is *uncorrelated*
  with it. Judge any future arm by that ratio, not by spurious-convergence counts, which reward an
  inflated-but-wrong residual. The only clean arms cost their speedup back (~1.10×), matching the +22.1%
  "delayed convergence" the residual-replacement literature reports. Don't reopen without reading the
  entry; if you do, the untested shape is *total* replacement.
- **Compensated summation anywhere in `ground_locg`.** `jnp.sum` is already tree-reduced (~1.5e-16
  relative, flat in `N`), a blocked-Kahan `compute_sas` leaves the residual floor bit-identical, and
  compensation **breaks sharding** — sequential accumulation over blocks reshapes a partitioned axis and
  `lax.scan` raises `0th dimension of all xs should be replicated`. The whole family is closed by a
  published measurement: CG with *every* inner product **and matvec** correctly rounded (Ozaki scheme,
  doi:10.1145/3432261.3432270) leaves the true residual at **median 1.002×**, worse in 3 of 8 cases. It
  buys reproducibility and some iteration count, never attainable accuracy.
