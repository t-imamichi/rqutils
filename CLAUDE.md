# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment & commands

Always use `uv run python` (not bare `python`) — the venv at `.venv` is managed by uv.

```bash
uv run python -c "import rqutils; print(rqutils.__version__)"

# Optional deps are extras, not installed by default. Pull them in per-invocation:
uv run --extra qiskit python examples/sqd.py 8 --num-paulis 10 --subspace-frac 0.5
uv run --extra qiskit python examples/svsim.py 12 --out /tmp/out.h5
uv run --extra mpl   python ...   # matplotlib, for qprint(output='mpl')
uv run --extra qutip python ...   # qutip Qobj input to qprint
```

Extras: `mpl`, `qutip`, `qiskit`, `docs`, `dev` (pytest + ruff + ty), `mlx` (darwin only). `mpi4py` is imported by `examples/` but declared nowhere — install it manually if you need the multi-process path.

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

## Testing

```bash
uv run --extra dev pytest              # whole suite
uv run --extra dev pytest -v -x        # verbose, stop at first failure
```

`tests/test_ground_locg.py` covers `rqutils/ground_locg.py` only — the other six modules have no
automated tests yet, so for those still write a throwaway `uv run python -c ...` script or run the
matching `examples/` script. `tests/` also contains three Jupyter notebooks (`paulis.ipynb`,
`qprint.ipynb`, `sqd.ipynb`) used as interactive scratchpads; pytest does not collect them.

`tests/conftest.py` enables `jax_enable_x64` before any `rqutils` import — every tolerance in the
suite depends on it. Tests are organized by defect and keyed to the audit items in `docs/locg.md`;
when adding one, name the item it locks down and assert on the eigenvector residual as well as the
eigenvalue (the audit's I3 returned an exact eigenvalue with a garbage eigenvector).

## Architecture

Eight largely independent modules under `rqutils/`; nothing but `sqd.py → {paulis/symplectic.py, ground_locg.py}` and `qprint.py → paulis/general.py` couples them. `ground_locg_mlx.py` imports nothing from the package and nothing imports it.

**Two unrelated Pauli representations — do not confuse them:**

- `paulis/general.py` — dense generalized (Gell-Mann-like) basis for arbitrary dimension. `paulis(dim)`, `components()`, `compose()`, `truncate()`, `labels()`. Normalization is `tr(λ_k λ_l) = 2δ_kl`, so **`λ_0 = sqrt(2/n)·I`, not `I`** — the most bug-prone invariant here. Basis-index ordering is fixed by a shell-by-shell construction loop and is relied on by `symmetry`, `l0_projector`, and `truncate`. Shapes: `paulis(dim)` → `(d1², …, D, D)` (basis axes *first*); component arrays → `(…, d1², …)` (component axes *last*). Everything is memoized in module-level dicts keyed by a `tuple(int)`-normalized `dim`.
- `paulis/symplectic.py` — `PauliSumXZ`, a bit-packed qubit-only form for JAX/GPU. Convention `Q = (-i)^{x·z} Z^z X^x` with **little-endian qubit ordering** (Qiskit's `.x`/`.z` get reversed on ingest). Terms are grouped by unique X signature, Z groups zero-padded to a rectangle, the `(-i)^{popcount(x&z)}` phase folded into the coefficients, then `np.packbits`.

**`sqd.py`** — sample-based quantum diagonalization: project a large Pauli-sum Hamiltonian onto the subspace spanned by a list of computational-basis bitstrings and solve matrix-free. `sqd(...)` is the entry point; `hproj(...)` is the dense/debug path. Two conventions dominate:

- States carry **one extra zero pad bit at position 0** before `packbits`, so the Hamiltonian must be built with `add_padding=True`. Filler slots produced by uniquification are `255`, detected via `states_u[:, 0] >> 7`.
- `cache_level=(source_indices, diagonals)` selects among six matvec kernels trading memory for speed; `states_size` exists solely to pin array shapes and prevent JIT recompilation.

**`ground_locg.py`** — single-vector (block-size-1) LOBPCG specialization used as `sqd`'s eigensolver, with the Rayleigh–Ritz step solved analytically (`eigenpair_2x2`, `eigenpair_3x3` via Cardano) instead of via `eigh`, to keep memory down for huge vectors. It is sharding-transparent **only if the `mat` callable preserves output sharding** — that contract is why every `apply_*` in `sqd.py` passes `out_sharding=jax.typeof(vec).sharding`. Every guard in it is load-bearing and was measured: `docs/locg.md` catalogues seven defects (I1–I7) that each failed *silently*, returning a plausible wrong number rather than raising. Don't "simplify" the balancing, the re-orthogonalizations, or the zero-direction masks.

**`ground_locg_mlx.py`** — an MLX port of the above plus `sqd.py`'s cached matvec, for Apple GPUs. **It duplicates the algorithm deliberately; when you change one, change both.** It is real-symmetric only (MLX has no complex128, Metal no float64) and raises on complex input rather than silently dropping imaginary parts. `mlx` is a darwin-only extra, so importing this module without it raises `ImportError`. MLX cannot initialize without a Metal device, so headless verification goes through `examples/check_ground_locg_mlx_static.py`, which re-executes the module's source against a numpy shim; `examples/check_ground_locg_mlx_mlx.py` is the real-device counterpart.

**`svsim.py`** — JAX state-vector simulator. Gates are compiled to the same symplectic `CircuitXZ` (x, z, cos, sin) form and applied by a single `jax.lax.scan`, so only `x, y, z, cz, rx, ry, rz, rzz` are supported; transpile to `basis_gates=['rx','ry','rz','rzz']` first. **`sin` is complex128, not real**: it carries `i·(-i)^popcount(x&z)`, i.e. the rotation's leading `i` and the `(-i)^{x·z}` phase of the `Q = (-i)^{x·z} Z^z X^x` convention, folded in at build time. Omitting that phase silently broke every `y`/`ry` gate — the only gates with overlapping X/Z signatures — and so every transpiled circuit (`docs/skqd.md`). Don't narrow `sin` back to float64. `cz` is only decomposed on the `QuantumCircuit` path (not as a raw gate spec) and is correct only up to a uniform `exp(iπ/4)`, which its `rzz`+`rz` decomposition cannot express.

**`qprint.py`** — pretty-printer with two orthogonal axes: `fmt` picks the content class (`QPrintBraKet` / `QPrintPauli` / `QPrintMatrix`) and `output` picks the rendering (`'text'` returns the object for lazy `__repr__`, `'latex'` a string, `'mpl'` a Figure). `QPrintBase` owns all numerics; subclasses only override `_qobj_data`, `_add_labels`, `_format_lhs`.

## Conventions to follow when editing

**`npmod`** — numeric functions take `npmod: ModuleType = np` as the last kwarg so callers can pass `jax.numpy` for traceable execution. The rule: **all validation, Python-level shape inference, and early returns must be gated on `if npmod is np:`**. Where control flow is data-dependent, add an explicit `if npmod is jnp:` branch using `jax.lax.fori_loop`/`cond` (see `truncate`).

**Optional dependencies** — uniform `try: import X / except ImportError: HAS_X = False / else: HAS_X = True` at module top, every use guarded by `HAS_X and isinstance(...)`. Type aliases are conditionally widened (`CircuitInput |= QuantumCircuit`), and the runtime path raises a `RuntimeError` rather than failing at import. `numpy`, `scipy`, `h5py`, and **`jax`** are hard dependencies.

**Sharding is implicit** — the library reads `jax.sharding.get_abstract_mesh()`; it is the *caller's* job to set the mesh. The examples establish the expected pattern: a single axis named `'x'` with `AxisType.Explicit`, plus `jax.config.update('jax_enable_x64', True)` (without x64 you silently get complex64/int32 — you'll see truncation warnings).

**Docstrings feed the published API reference.** Every module opens with a raw docstring that is a full reST document: over/underlined title, `.. currentmodule::`, prose with `.. math::` derivations of the normalization conventions, and an explicit API section (`.. autofunction::` / `.. autoclass::` / `.. autosummary::`). Function docstrings are Google-style (`Args:` / `Returns:` / `Raises:`) via napoleon. Adding a public module requires **both** those directives **and** a manual line in the `toctree` of `docs/source/index.rst`.

## Known rough edges

- `paulis(dim)` for multiple subsystems uses `np.einsum` with 3 letters per subsystem, capping at ~17 subsystems; `sparse=True` for products raises `NotImplementedError`.
- `sqd` is limited to `N ≤ 2^31` subspace states because `get_xsource` sorts a `2N` stack on a single device; that sort is also noted to leak GPU memory.
- `run_sqd`'s initial vector is a deterministic pseudo-random spread (`_spread_seed`), not a one-hot: a one-hot cannot leave the connected component of the projected Hamiltonian that contains it, so a subspace whose Hamiltonian splits into disconnected blocks silently returned that block's minimum with `converged=True`. `vinit_from_min_diag` still weights the minimum-diagonal state heavily on top of the spread. Don't "simplify" either back to a one-hot — `tests/test_sqd.py::TestSqdInitialVector` covers both failure modes.
