# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment & commands

Always use `uv run python` (not bare `python`) — the venv at `.venv` is managed by uv. No `timeout` on
macOS (it is GNU coreutils) — use the Bash tool's own timeout rather than wrapping a command in it.

```bash
uv run python -c "import rqutils; print(rqutils.__version__)"

# Optional deps are extras, not installed by default. Pull them in per-invocation:
uv run --extra qiskit python examples/sqd.py 8 --num-paulis 10 --subspace-frac 0.5
uv run --extra qiskit python examples/svsim.py 12 --out /tmp/out.h5
uv run --extra mpl   python ...   # matplotlib, for qprint(output='mpl')
uv run --extra qutip python ...   # qutip Qobj input to qprint
```

Extras: `mpl`, `qutip`, `qiskit`, `docs`, `dev` (pytest + ruff + ty), `mlx` (darwin only). `mpi4py` is imported by `examples/` but declared nowhere — install it manually if you need the multi-process path.

`examples/` holds the three library demos (`sqd.py`, `svsim.py`, `bench.py`); everything for the
MLX/Metal port lives under `examples/mlx/`. Names there are **unqualified by design** — the
directory already says `mlx`, so `check_solver_headless.py` rather than the old
`check_ground_locg_mlx_mlx.py`. The suffix instead carries the fact that actually governs how you
run a script:

| | subject | |
|---|---|---|
| `bench.py` | the solver, timed | needs a device for the `mlx-*` arms |
| `check_bench.py` | `bench.py`'s JAX arms, gate, reporting | headless (subprocesses `bench.py`) |
| `check_bench_common.py` | `_bench_common.py`'s references | headless |
| `check_solver_headless.py` | `ground_locg_mlx.py` via a numpy shim | headless |
| `check_solver_device.py` | `ground_locg_mlx.py` on real MLX | **needs a Metal device** |
| `count_ops.py` | op constructions per iteration | headless |

Note `examples/mlx/bench.py` is a different file from `examples/bench.py` (the svsim one); every
in-tree reference resolves `__file__`-relative, so nothing depends on the basename being unique.

The seven scripts are a coupled set, not independent files: `bench.py`, `check_solver_headless.py`,
`check_solver_device.py` and `check_bench_common.py` all reach `_bench_common.py` through
`sys.path.insert(dirname(__file__))`; `check_solver_headless.py`/`count_ops.py` locate
`rqutils/ground_locg_mlx.py` by walking **two** levels up; and `check_bench.py` builds a path to
`bench.py` to subprocess it. Move or rename one and an import, a `SRC` path, or that subprocess
breaks — so change them together and re-run the four headless checks afterwards.

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
by pytest; **`svsim`'s `out_sharding` is still exercised by nothing.** `rqutils/ground_locg_mlx.py`
needs a Metal device and is checked by `examples/mlx/check_solver_headless.py` (numpy shim, runs
anywhere) and `examples/mlx/check_solver_device.py` (real device) instead. `tests/` also contains three Jupyter
notebooks used as interactive scratchpads; pytest does not collect them.

`tests/conftest.py` enables `jax_enable_x64` before any `rqutils` import — every tolerance in the
suite depends on it — and holds the shared reference helpers (dense Pauli sums, projections, gate
unitaries), each validated against qiskit before being trusted as a reference.

**Tests are organized by defect.** Writing these suites found bugs in five of the seven modules, all
of the same character: a plausible finite answer rather than a raise or a `NaN`. So when adding a
test, name the defect it locks down and record the measured wrong value, and prefer an *independent*
reference (a dense construction, scipy, qiskit) over self-consistency — several bugs made every
internal code path agree on the same wrong number. Verify a new test actually fails against the bug
it targets by reverting the fix in place; a copy of the repo does not work, since the venv holds an
editable install pointing at the original.

**Multi-device paths are testable on CPU — use it.**
`XLA_FLAGS=--xla_force_host_platform_device_count=4` gives virtual devices that exercise every
sharding code path (mesh detection, `PartitionSpec` propagation, `jax.reshard`, `sqd`'s mesh-size
padding) with no GPU. This is not hypothetical: the first run found `sqd` raising `ShardingTypeError`
on *any* mesh, because one scatter omitted `out_sharding` while every neighbouring op passed it.
`examples/scaling/poc7_sharding.py` is the harness — sharded against single-device against a dense
`eigvalsh` reference, plus every residue of `N mod mesh.size`. Timings under virtual devices are
meaningless (they share one CPU), so use this for correctness only.

## Architecture

Eight largely independent modules under `rqutils/`; nothing but `sqd.py → {paulis/symplectic.py, ground_locg.py}` and `qprint.py → paulis/general.py` couples them. `ground_locg_mlx.py` imports nothing from the package and nothing imports it.

**Two unrelated Pauli representations — do not confuse them:**

- `paulis/general.py` — dense generalized (Gell-Mann-like) basis for arbitrary dimension. The whole public surface is `paulis(dim)`, `pauli_matrices(dim)` (the single-subsystem primitive `paulis` builds on), `components()`, and `labels()`; `compose`, `truncate`, `l0_projector`, `symmetry` and `paulis_shape` were removed as dead code, so don't reintroduce a caller expecting them. Normalization is `tr(λ_k λ_l) = 2δ_kl`, so **`λ_0 = sqrt(2/n)·I`, not `I`** — the most bug-prone invariant here. Basis-index ordering is fixed by a shell-by-shell construction loop; `components` and `labels` both index by basis position, so a reordering would disagree with the labels users read while every function stayed self-consistent. Shapes: `paulis(dim)` → `(d1², …, D, D)` (basis axes *first*); component arrays → `(…, d1², …)` (component axes *last*). Everything is memoized in module-level dicts keyed by a `tuple(int)`-normalized `dim` (`_normalize_dim`).
- `paulis/symplectic.py` — `PauliSumXZ`, a bit-packed qubit-only form for JAX/GPU. Convention `Q = (-i)^{x·z} Z^z X^x` with **little-endian qubit ordering** (Qiskit's `.x`/`.z` get reversed on ingest). Terms are grouped by unique X signature, Z groups zero-padded to a rectangle, the `(-i)^{popcount(x&z)}` phase folded into the coefficients, then `np.packbits`. Signatures always reserve **one pad bit at position 0** (a dummy identity, aligning with the pad bit consumers put in their state bitstrings) — not optional, so alignment cannot be got wrong. Note that `packbits` fills each byte from the **most significant** end, so a signature's payload bits are entries `1` through `num_qubits` of `np.unpackbits`, in string-character order (leftmost character = index 1) — the reverse of the qubit numbering. Anything decoding a packed signature back to an integer must shift by `8*nbytes - (num_qubits + 1)`, counting the pad bit; dropping the `+1` silently returns a *permutation* of the right answer (measured 2.07 max abs error in the since-removed `matmul`, which is why that method is gone). The class is now just `from_paulisum` and the `arrays` property. **A complex coefficient raises** (non-Hermitian; both ingest branches share one check) — there is no `force_real` flag any more, because none could work: `.c` narrows to float64 exactly when the folded phase is real, i.e. when every string has an even Y count, and an odd-Y string is complex128 *by construction*. Check `.c.dtype` if you need float64; `examples/mlx/_bench_common.py` is the caller that does.

**`sqd.py`** — sample-based quantum diagonalization: project a large Pauli-sum Hamiltonian onto the subspace spanned by a list of computational-basis bitstrings and solve matrix-free. `sqd(...)` is the entry point; `hproj(...)` is the dense/debug path. Two conventions dominate:

- States carry **one extra zero pad bit at position 0** before `packbits`. `PauliSumXZ` reserves the same bit in its signatures unconditionally, so the two are aligned by construction — this used to be an opt-in `add_padding` flag, and the two sides disagreeing is how `hproj` shipped broken. Filler slots produced by uniquification are `255`, detected via `states_u[:, 0] >> 7`.
- `cache_level=(source_indices, diagonals)` selects among six matvec strategies trading memory for speed. They are one kernel, `apply_h`, indexed by that 2×3 grid; `cache_level` **must** stay static (it is bound via `functools.partial`, since `ground_locg` splats `args` positionally and `static_argnames` would never see it). `apply_h_xz_cached` is a named wrapper for `(1, 2)` because `examples/mlx/bench.py` and `ground_locg_mlx` both refer to it. `states_size` exists solely to pin array shapes and prevent JIT recompilation.

Two measured facts about the cost, from the six scaling POCs under `examples/scaling/` (findings in
`docs/scaling-pocs.md`). **`get_xsource` setup dominates** — weighted by call count it is 66–97% of a
solve (3.1 s against 79 ms of matvec loop at 10 iterations, N=200k, J=50), so the `2N` sort is not
merely the `N ≤ 2^31` ceiling but the main cost at every size measured, while `matvec/J` is flat at
~0.16 ms and confirms the `O(J·N)` model. The caching docstring above, which frames the tradeoff as
memory-versus-speed, will mislead you about where the time goes.

**`get_xsource` is now a binary search, not a sort** (12–19× faster on the J-fold precompute), which is
why **`states` must be lex-sorted** — always required, since the sort was equally wrong on unsorted
input, but previously undocumented. `hproj(unique_states=True)` skips its `np.unique` and so can
violate it; that returns a non-symmetric matrix and is pinned by
`TestHproj::test_unsorted_input_with_unique_states_is_wrong`. Two paths selected statically on width:
`uint64` keys for `B ≤ 8` bytes, an explicit lexicographic search beyond. That boundary is a
**correctness** limit — a `uint64` key silently truncates a wider row and aliases distinct states —
so if you touch it, note that a test only catches the overrun when the subspace's *leading* bytes
collide and partners genuinely exist; `packbits` puts low qubit indices in the *leading* bytes, the
reverse of the qubit numbering, and getting that backwards makes the test pass vacuously.

**`ground_locg.py`** — single-vector (block-size-1) LOBPCG specialization used as `sqd`'s eigensolver, with the Rayleigh–Ritz step solved analytically (`eigenpair_2x2`, `eigenpair_3x3` via Cardano) instead of via `eigh`, to keep memory down for huge vectors. It is sharding-transparent **only if the `mat` callable preserves output sharding** — that contract is why every `apply_*` in `sqd.py` passes `out_sharding=jax.typeof(vec).sharding`. Every guard in it is load-bearing and was measured: `docs/locg.md` catalogues seven defects (I1–I7) that each failed *silently*, returning a plausible wrong number rather than raising. Don't "simplify" the balancing, the re-orthogonalizations, or the zero-direction masks.

**`ground_locg_mlx.py`** — an MLX port of the above plus `sqd.py`'s cached matvec, for Apple GPUs. **It duplicates the algorithm deliberately; when you change one, change both.** It is real-symmetric only (MLX has no complex128, Metal no float64) and raises on complex input rather than silently dropping imaginary parts. `mlx` is a darwin-only extra, so importing this module without it raises `ImportError`. MLX cannot initialize without a Metal device, so headless verification goes through `examples/mlx/check_solver_headless.py`, which re-executes the module's source against a numpy shim; `examples/mlx/check_solver_device.py` is the real-device counterpart.

Its whole option surface is **one `device="cpu"|"gpu"` parameter**. `"gpu"` selects the two fused Metal kernels (`_apply_h_xz_metal`, `_eigenpair_3x3_metal`) and requires float32; `"cpu"` runs the portable op-graph path and is the **only** route for an f64 solve, since Metal has no float64. `mx.compile` is unconditional. It got this way by deletion: `sas=`, `eig=`, `compile_body=`/`compile_chunk=` and a three-way matvec choice each existed to isolate one optimization while it was being measured, and once measured, two axes had losing arms that existed only to be not-chosen. `docs/mlx-metal-kernels.md` records what each was worth and, more usefully, the two **negative** results — read it before adding a knob back or fusing something new.

Two consequences worth knowing. The `device="gpu"`+f64 guard lives in `ground_locg_mlx`, not in the kernel, because the `seed_converged` early return exits *without entering the loop*, so a pushed-down guard would let that configuration succeed silently. And with `tol != 0.` the convergence test runs at `_COMPILE_CHUNK` (10) boundaries, so a solve terminating on a zeroed search direction can report an iteration count rounded up to a multiple of 10 — never wrong, since no further iteration can lower θ. Don't "fix" that by threading an OR-accumulated flag through the compiled carry; that trade was already costed and declined.

"Change both" applies to the **algebra, not the op granularity** — and this is the one place the two files are meant to look different. In JAX an unrolled Python loop over scalars is free (XLA fuses it into the surrounding `jit`); in MLX every call is its own lazily-evaluated op, so a faithful transcription of `jnp.stack([normalize(c) for c in cands])` cost 18% of the port's entire per-iteration op count. The Rayleigh–Ritz step here therefore uses batched whole-array forms (one broadcast cross for all three complement candidates, one `(7, 3)` normalization, `mx.diagonal`/`mx.roll` over element-wise `mx.stack`) and hoists dtype-dependent constants into `_CONST_CACHE` rather than rebuilding them per iteration — plus `_compute_sas` taking all n² inner products from one stacked matmul, which is *also* more accurate than the n²-reductions form it replaced. 116 → 65.0 ops/iter, and 32.5 under `device="gpu"`, with the eigenvalue bit-identical. `_project_out` is deliberately **not** batched the same way: a matmul reassociates its summation order, and it measured consistently worse in the near-degenerate regime its I5/I6 guards exist for (8.3e-17 vs 6.2e-17 residual non-orthogonality) — small, but not worth ~4 ops. Its docstring records the comparison; re-run it before touching that function. Don't unroll these back for symmetry with the JAX file, and don't port them *to* JAX, where they are churn.

`examples/mlx/count_ops.py` measures the op count (`--by-op` for the per-op breakdown, which is what caught a "fix" that moved the count the wrong way because it allocated two constants per call). Two cautions. It counts **op constructions, not time** — third in the cost model, behind sync count and launch count — so quote a speedup only from `examples/mlx/bench.py` on real hardware. And its totals **exclude the matvec's own ops**, which are attributed to `apply_h_xz`/`_apply_h_xz_metal` and fall outside the body filter; that is why the `cpu` arm still reads 65.0 after the matvec changed. **Fewer ops ≠ less time** — the `_compute_sas` matmul was flat on the CPU backend and ~1.17× on the GPU, because it removes ~12 kernel launches and the CPU backend has no launches to remove. Measure the backend you care about; a flat CPU result does not mean a change is worthless. `docs/mlx-metal-kernels.md` has the full history by backend.

**Read `per_it_ms` (from `fixed_s`, `tol=0.0`), not `solve_s`, when the spread warning fires.** `timeit` runs twice per arm and the warning doesn't say which call tripped it; if the reported `min` matches `solve_s` rather than `fixed_s`, the disturbance is in the convergence-checking path, whose per-`_COMPILE_CHUNK` device sync has queue-state-dependent latency. `fixed_s` (`tol=0.0`) is now the only configuration that does **zero** syncs, and it stays stable — four processes agreed to 1.3% while the same runs warned at 64–70% on `solve()`.

When benchmarking a before/after, **revert `rqutils/ground_locg_mlx.py` and `examples/mlx/bench.py` together.** This is now true in both directions: a new bench passes `device=`, which older module versions lack, and an old bench passes `sas=`/`eig=`/`compile_body=`, which the current module rejects — either mismatch raises `TypeError`. Verify the checkout actually took with `grep -c _CONST_CACHE rqutils/ground_locg_mlx.py` (0 = baseline, nonzero = optimized); a silently no-op'd `git stash` produced two same-code runs that looked like a valid comparison. Those two runs did establish this arm's **noise floor at 3.9%**, so treat any difference under ~4% as unresolved.

Two Metal kernels exist, both reached via `device="gpu"`, and **whether fusing wins is decided by output parallelism, not launch count** — the one transferable lesson from them. `_apply_h_xz_metal` (N outputs, one thread each) was a large win. `_eigenpair_3x3_metal` wins because a 3×3 eigensolve has no N-scaling work to under-parallelize: ~34 launches of nine-number scalar arithmetic collapse to one thread in registers, measured **1.75× per-iteration and 1.69× end-to-end** with a bit-identical eigenvalue, and unlike the op-graph reductions that win survives compilation. A third kernel fusing the Rayleigh–Ritz inner products **measured slower and was deleted** — it under-parallelized an O(N) reduction into six threadgroups regardless of N. Don't rewrite it; `docs/mlx-metal-kernels.md` explains why it lost. Both surviving kernels are fp32-only, so the f64 arm can use neither, which is why the portable op-graph reductions matter independently. Two static guards (`metal::` qualification, MSL-reserved identifiers — including `half`, whose incident motivated the scan) cover compile risk the numpy shim is structurally blind to, since it never compiles the MSL text; both were verified to fire by breaking them.

**Device status: validated on an M1 after the option collapse** (2026-08-05, `check_solver_device.py`, 12 arms, `FAILURES: none`). `mlx-gpu-f32 --matvec metal` measured 0.224 ms/iter at 70 iterations — the bottom of the recorded 0.224–0.227 range, which is what confirms `device="gpu"` reaches both fused kernels rather than falling back to the op-graph path (a fallback reads ~0.46, so this is the cheap sanity check worth doing after touching the dispatch). Expect the eigenvalue `-5.3960409164`, the post-`_compute_sas` f32 value, **not** the `-5.3960399628` recorded against the eigensolve kernel before that change; the difference is ~1.8e-7 relative against an f32 eps of 1.19e-7 and is documented, not a regression.

When attributing an eigenvalue change to a kernel, **pin every problem parameter first**. An uncontrolled comparison here looked like an I5-class accuracy regression (energy below reference, 120 iterations instead of 70) and was purely a different `--num-qubits`. The correctness gate cannot catch this: it compares against a reference built from the same run's inputs, so it passes either way.

Beware one fp32 trap when writing eigensolve tests: with a large dynamic range the balancing step destroys the small eigenvalues outright — for `diag([-1.5, -1.0, 1e9])` the two small entries become *bit-identical* after dividing by `scale=1e9`, and both the op-graph and fused paths return `0.0`. That is an inherent fp32 limit, not a defect, so build such cases from the exclusion shift `iter_body` actually produces (`max(diag) + sum(|diag|) + 1`), and normalize residuals by ‖A‖ rather than |θ|.

**`svsim.py`** — JAX state-vector simulator. Gates are compiled to the same symplectic `CircuitXZ` (x, z, cos, sin) form and applied by a single `jax.lax.scan`, so only `x, y, z, cz, rx, ry, rz, rzz` are supported; transpile to `basis_gates=['rx','ry','rz','rzz']` first. **`sin` is complex128, not real**: it carries `i·(-i)^popcount(x&z)`, i.e. the rotation's leading `i` and the `(-i)^{x·z}` phase of the `Q = (-i)^{x·z} Z^z X^x` convention, folded in at build time. Omitting that phase silently broke every `y`/`ry` gate — the only gates with overlapping X/Z signatures — and so every transpiled circuit (`docs/skqd.md`). Don't narrow `sin` back to float64. `cz` is only decomposed on the `QuantumCircuit` path (not as a raw gate spec) and is correct only up to a uniform `exp(iπ/4)`, which its `rzz`+`rz` decomposition cannot express.

**`qprint.py`** — pretty-printer with two orthogonal axes: `fmt` picks the content class (`QPrintBraKet` / `QPrintPauli` / `QPrintMatrix`) and `output` picks the rendering (`'text'` returns the object for lazy `__repr__`, `'latex'` a string, `'mpl'` a Figure). `QPrintBase` owns all numerics; subclasses only override `_qobj_data`, `_add_labels`, `_format_lhs`. **Test the full `fmt` × `output` grid, not a diagonal of it**: four bugs lived in cells nothing exercised, including `fmt='matrix'` being un-instantiable for *every* input (`QPrintMatrix` never implemented the abstract `_add_labels`, which it does not need — it overrides `_make_lines` and positions terms by row/column). Note that an amplitude of exactly `1` is suppressed, which is right when a basis label follows and wrong when nothing does; text-mode labels also carry the `*` separator as a prefix, so the two renderings can disagree while each looks fine alone. Cross-rendering assertions are what catch that — see `tests/test_qprint.py::TestAmplitudeAndSeparator`.

## Conventions to follow when editing

**`npmod`** — numeric functions take `npmod: ModuleType = np` as the last kwarg so callers can pass `jax.numpy` for traceable execution. The rule: **validation and early returns must be gated on `if npmod is np:`**, but **Python-level shape inference must NOT be** — it operates on static values and is needed identically by both backends. Gating it is what broke `paulis/general.py`'s entire `npmod=jnp` path in three separate places (`components` plus the since-removed `compose` and `truncate`, all raising `TypeError: object of type 'int' has no len()` on a scalar `dim`) — `_normalize_dim` is now called unconditionally at every site for exactly this reason. Shape arithmetic should use `np` explicitly, not `npmod`: `jnp.sqrt` rejects the plain tuples that `array.shape` returns, where numpy accepts them. And prefer an unrolled Python loop over `jax.lax.fori_loop` when the trip count is static — a traced loop index cannot subscript a static dimension tuple (`TracerIntegerConversionError`). `components` is now the only `npmod` consumer in the module; `tests/test_paulis_general.py::TestNpmodParity` pins it, including under `jax.jit`.

**Optional dependencies** — uniform `try: import X / except ImportError: HAS_X = False / else: HAS_X = True` at module top, every use guarded by `HAS_X and isinstance(...)`. Type aliases are conditionally widened (`CircuitInput |= QuantumCircuit`), and the runtime path raises a `RuntimeError` rather than failing at import. `numpy`, `scipy`, `h5py`, and **`jax`** are hard dependencies.

**Sharding is implicit** — the library reads `jax.sharding.get_abstract_mesh()`; it is the *caller's* job to set the mesh. The examples establish the expected pattern: a single axis named `'x'` with `AxisType.Explicit`, plus `jax.config.update('jax_enable_x64', True)` (without x64 you silently get complex64/int32 — you'll see truncation warnings).

**Docstrings feed the published API reference.** Every module opens with a raw docstring that is a full reST document: over/underlined title, `.. currentmodule::`, prose with `.. math::` derivations of the normalization conventions, and an explicit API section (`.. autofunction::` / `.. autoclass::` / `.. autosummary::`). Function docstrings are Google-style (`Args:` / `Returns:` / `Raises:`) via napoleon. Adding a public module requires **both** those directives **and** a manual line in the `toctree` of `docs/source/index.rst`.

## Known rough edges

- `paulis(dim)` for multiple subsystems uses `np.einsum` with 3 letters per subsystem, capping at ~17 subsystems; `sparse=True` for products raises `NotImplementedError`.
- `sqd` is limited to `N ≤ 2^31` subspace states because `get_xsource` sorts a `2N` stack on a single device; that sort is also noted to leak GPU memory.
- `run_sqd`'s initial vector is a deterministic pseudo-random spread (`_spread_seed`), not a one-hot: a one-hot cannot leave the connected component of the projected Hamiltonian that contains it, so a subspace whose Hamiltonian splits into disconnected blocks silently returned that block's minimum with `converged=True`. `vinit_from_min_diag` still weights the minimum-diagonal state heavily on top of the spread. Don't "simplify" either back to a one-hot — `tests/test_sqd.py::TestSqdInitialVector` covers both failure modes.
