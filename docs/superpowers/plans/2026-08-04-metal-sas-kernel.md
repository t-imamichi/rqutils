# Metal `_compute_sas` Kernel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fuse `ground_locg_mlx`'s Rayleigh–Ritz inner products (`_compute_sas`, 16 op launches/iteration) into a single `mx.fast.metal_kernel` launch, and make the benchmark's correctness gate scale past N≈20 000 so the change can be measured where the GPU is not overhead-dominated.

**Architecture:** Add `_compute_sas_metal(vectors, mvs)` alongside the existing op-graph `_compute_sas` in `rqutils/ground_locg_mlx.py`. One threadgroup per (i,j) pair with i≤j; each thread strides over N accumulating in a register; a threadgroup-memory tree reduction collapses partials; thread 0 writes **both** `out[i*n+j]` and `out[j*n+i]` so symmetry is exact by construction and the `(sas + sas.T) * 0.5` op disappears. fp32-only (Metal has no float64), selected once before the solver loop so the dispatch cannot add a per-iteration sync. Separately, `examples/_bench_common.dense_reference` gains a sparse `scipy.sparse.linalg.eigsh` path so the gate is not capped by an (N,N) dense allocation.

**Tech Stack:** Python 3, MLX 0.32.0 (`mx.fast.metal_kernel`, Metal Shading Language), numpy, scipy (`sparse.linalg.eigsh`), JAX (reference arms only), `uv` for env management, ruff + ty for lint/types.

## Global Constraints

- Always use `uv run python`, never bare `python`. MLX needs `--extra mlx`; lint/types/tests need `--extra dev`.
- **This session has no Metal device.** `mlx.core` imports but `metal::load_device` fails on even `mx.array([1.0])`. Every task below is designed to be *implemented and validated headless* via the numpy shim. Tasks 5 and 6 (real-device check, benchmark sweep) **cannot be run by the implementing agent** and must be handed to the user.
- `ruff check`, `ruff format`, and `ty check` must stay clean over `rqutils/ examples/`. Line width 100.
- `uv run --extra dev pytest` must stay green. No existing test may regress.
- The numpy shim in `examples/check_ground_locg_mlx_static.py` validates caller wiring and indexing arithmetic **only**. It cannot prove the Metal source compiles or that barriers are correct. Never claim otherwise in a comment, docstring, or commit message.
- fp32-only guard message must contain the substring `float32` (the existing static check asserts on it).
- Existing measured results must stay reproducible: every new flag defaults to the current behaviour (`--sas ops`).
- Do not modify `rqutils/ground_locg.py` (JAX). This change is MLX-only, so the "change both" obligation in `CLAUDE.md` is not triggered.
- Do not touch `_project_out` or `eigenpair_3x3`. Both are explicitly out of scope per the spec; their guards are items I5/I6/I1/I2 of `docs/locg.md` and each was measured to fail *silently*.

---

### Task 1: Sparse eigsh path for the correctness gate

Unblocks large-N measurement. `dense_reference` allocates `(N,N)` float64 — 6 MB at N=893, 7.2 GB at N=30 000, 80 GB at N=1e5.

**Files:**
- Modify: `examples/_bench_common.py` (add `sparse_reference`, add threshold logic; `dense_reference` is at lines 171-188)
- Test: `examples/check_bench_common.py` (existing checker for this module)

**Already verified headless** (n=10, 20 paulis, 200 states, seed=1): the `coo_matrix` construction below reproduces `dense_reference`'s `np.add.at` result **bit-identically** (max diff exactly `0.0`), `eigsh(which="SA")` agrees with `eigvalsh` to `8.9e-16`, and the operator has 794 nonzeros versus 32 761 dense elements. So the approach is sound before implementation starts; the test below pins it.

**Interfaces:**
- Consumes: `SolverInputs` (fields `xsources`, `diagonals`, `vinit`, `subspace_dim`, `num_xgroups`), already defined at `examples/_bench_common.py:32`.
- Produces:
  - `sparse_reference(inputs: SolverInputs) -> float` — algebraically-smallest eigenvalue via `eigsh`.
  - `DENSE_REFERENCE_MAX_DIM: int = 5000` — module-level threshold constant.
  - `dense_reference` keeps its existing signature `(inputs) -> tuple[np.ndarray, float]`. Unchanged, so all existing callers keep working.

- [ ] **Step 1: Write the failing test**

Append to `examples/check_bench_common.py`. A mid-file `from _bench_common import ...` is fine and matches existing style in `examples/` (E402 is not in the repo's enabled ruff rules — 9 such imports already exist). Do **not** add a `# noqa: E402`.

```python
# Sparse gate path: eigsh on the same operator must agree with the dense eigvalsh it replaces.
# Validated against the dense path rather than trusted on arrival -- CLAUDE.md prefers an
# independent reference, and eigsh is a different algorithm in a different library.
from _bench_common import DENSE_REFERENCE_MAX_DIM, sparse_reference

ps_s, cs_s, states_s = generate_problem(10, 20, 200, seed=1)
inputs_s = build_solver_inputs(ps_s, cs_s, states_s)
_, dense_eig = dense_reference(inputs_s)
sparse_eig = sparse_reference(inputs_s)
assert np.isfinite(sparse_eig), f"sparse_reference returned non-finite {sparse_eig}"
err_sparse = abs(sparse_eig - dense_eig)
assert err_sparse < 1e-9 * max(1.0, abs(dense_eig)), (
    f"sparse_reference {sparse_eig} disagrees with dense_reference {dense_eig} by {err_sparse}"
)
print(f"OK  sparse_reference matches dense_reference (err {err_sparse:.2e})")
assert DENSE_REFERENCE_MAX_DIM >= 4000, (
    f"DENSE_REFERENCE_MAX_DIM={DENSE_REFERENCE_MAX_DIM} must not be below the bench default "
    "--num-states 4000, or existing invocations would silently change reference path"
)
print(f"OK  DENSE_REFERENCE_MAX_DIM={DENSE_REFERENCE_MAX_DIM} keeps the bench default dense")
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `uv run python examples/check_bench_common.py`
Expected: FAIL with `ImportError: cannot import name 'DENSE_REFERENCE_MAX_DIM'`

- [ ] **Step 3: Implement the minimal code to make the test pass**

In `examples/_bench_common.py`, add near the other module constants:

```python
# Above this subspace dimension, dense_reference's (N, N) float64 matrix stops being
# allocatable: 200 MB at N=5000, 7.2 GB at N=30000, 80 GB at N=1e5. The sparse path takes over
# there. Chosen above the bench default --num-states 4000 so existing invocations keep using
# the dense path and stay bit-for-bit reproducible.
DENSE_REFERENCE_MAX_DIM = 5000
```

And add after `dense_reference`:

```python
def sparse_reference(inputs: SolverInputs) -> float:
    """Ground energy of the projected Hamiltonian via sparse eigsh, for large subspaces.

    ``dense_reference`` allocates an (N, N) float64 matrix, which is 80 GB at N=1e5. The
    operator is sparse by construction -- ``np.add.at`` over J X-groups, so at most J*N
    nonzeros -- so the same eigenvalue is reachable without ever densifying it.

    An independent algorithm in an independent library, not a second run of the solver under
    test: CLAUDE.md records bugs where every internal code path agreed on the same wrong number,
    so self-consistency is not an acceptable reference.

    Raises:
        SystemExit: If eigsh fails to converge or returns a non-finite value. An unconverged
            reference must fail the gate loudly -- a NaN eigenvalue once passed this benchmark's
            gate silently, because every IEEE 754 NaN comparison is false.
    """
    import scipy.sparse
    import scipy.sparse.linalg

    dim = inputs.subspace_dim
    rows = np.tile(np.arange(dim), inputs.xsources.shape[0])
    cols = inputs.xsources.reshape(-1)
    data = inputs.diagonals.reshape(-1)
    # coo_matrix sums duplicate (row, col) entries, matching dense_reference's np.add.at.
    matrix = scipy.sparse.coo_matrix((data, (rows, cols)), shape=(dim, dim)).tocsr()

    try:
        # 'SA' = smallest algebraic. Not 'SM' (smallest magnitude), which would return the
        # eigenvalue nearest zero rather than the ground state.
        eigvals = scipy.sparse.linalg.eigsh(
            matrix, k=1, which="SA", return_eigenvectors=False, tol=0.0
        )
    except scipy.sparse.linalg.ArpackNoConvergence as exc:
        raise SystemExit(
            f"gate failed: sparse_reference eigsh did not converge at N={dim} ({exc}); the "
            "reference eigenvalue is unusable, so no timing row may be reported"
        ) from exc
    value = float(eigvals[0])
    if not np.isfinite(value):
        raise SystemExit(
            f"gate failed: sparse_reference returned non-finite {value} at N={dim}. Every NaN "
            "comparison is false, so this must be rejected explicitly rather than compared"
        )
    return value
```

- [ ] **Step 4: Run the tests and make sure they pass**

Run: `uv run python examples/check_bench_common.py`
Expected: PASS, printing `OK  sparse_reference matches dense_reference (err ...)`

Then confirm nothing else broke:
Run: `uv run --extra dev pytest -q && uv run --extra dev ruff check rqutils/ examples/ && uv run --extra dev ruff format --check rqutils/ examples/ && uv run --extra dev ty check rqutils/ examples/`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add examples/_bench_common.py examples/check_bench_common.py
git commit -m "Add a sparse eigsh reference so the gate scales past a dense (N,N) matrix"
```

---

### Task 2: `_compute_sas_metal` kernel and its float32 guard

**Files:**
- Modify: `rqutils/ground_locg_mlx.py` (add source string, builder, and public function after `apply_h_xz_mlx_metal`, which ends at line 230)
- Test: `examples/check_ground_locg_mlx_static.py`

**Interfaces:**
- Consumes: `mx.fast.metal_kernel` (signature confirmed: `name`, `input_names`, `output_names`, `source`, `header=''`, `ensure_row_contiguous=True`, `atomic_outputs=False`, `compile_options=None`). Existing `_compute_sas(vectors, mvs)` at `rqutils/ground_locg_mlx.py:453`.
- Produces:
  - `_compute_sas_metal(vectors, mvs, threadgroup=256)` — same `(vectors, mvs)` calling convention as `_compute_sas`; both are tuples of length 2 or 3. Returns an `(n, n)` float32 MLX array.
  - `_METAL_SAS_KERNEL` / `_get_metal_sas_kernel()` — module-private, mirroring the matvec's memo pattern.

- [ ] **Step 1: Write the failing test**

Append to `examples/check_ground_locg_mlx_static.py`, immediately before the final `print("\nALL STATIC CHECKS PASSED ...")`:

```python
# 3h. _compute_sas_metal's CALLER logic and arithmetic, for both basis sizes. Same standing
# caveat as 3g: the shim interprets the kernel in numpy, so this validates indexing, shapes,
# grid setup and the both-triangles symmetry claim -- NOT that the Metal source compiles.
_compute_sas = module._compute_sas
_compute_sas_metal = module._compute_sas_metal

rng_sas = np.random.default_rng(11)
for nbasis in (2, 3):
    vecs = [rng_sas.normal(size=inputs.subspace_dim) for _ in range(nbasis)]
    mvs = [np.asarray(apply_h_xz_mlx(vv, inputs.xsources, inputs.diagonals)) for vv in vecs]
    want = np.asarray(_compute_sas(tuple(vecs), tuple(mvs)))
    vecs32 = [vv.astype(np.float32) for vv in vecs]
    mvs32 = [mm.astype(np.float32) for mm in mvs]
    got_sas = np.asarray(_compute_sas_metal(tuple(vecs32), tuple(mvs32)))

    assert got_sas.shape == (nbasis, nbasis), f"n={nbasis}: shape {got_sas.shape}"
    # Exact symmetry, not symmetry-to-tolerance: thread 0 writes the same value to both
    # triangles, so the two must be bit-identical. This is stronger than the op-graph path's
    # (sas + sas.T) * 0.5, which averages two values differing by rounding.
    asym = np.abs(got_sas - got_sas.T).max()
    assert asym == 0.0, f"n={nbasis}: output not exactly symmetric (|S - S.T| = {asym})"
    err_sas = np.abs(got_sas - want).max()
    scale = max(1.0, np.abs(want).max())
    assert err_sas < 1e-4 * scale, f"n={nbasis}: disagrees with _compute_sas by {err_sas}"

    # Independent reference: the inner products computed directly, not via either code path.
    direct = np.array([[vv @ mm for mm in mvs] for vv in vecs])
    direct = (direct + direct.T) * 0.5
    err_direct = np.abs(got_sas - direct).max()
    assert err_direct < 1e-4 * scale, f"n={nbasis}: disagrees with direct v@m by {err_direct}"
    print(
        f"OK  _compute_sas_metal n={nbasis} matches _compute_sas ({err_sas:.2e}) and "
        f"direct v@m ({err_direct:.2e}), exactly symmetric"
    )

# The float64 guard must fire, mirroring apply_h_xz_mlx_metal's (Metal has no float64).
try:
    _compute_sas_metal((vecs[0], vecs[1]), (mvs[0], mvs[1]))
except ValueError as exc:
    assert "float32" in str(exc), f"unexpected guard message: {exc}"
    print("OK  _compute_sas_metal rejects float64 input (Metal has no float64)")
else:
    raise AssertionError("_compute_sas_metal accepted float64 input -- guard did not fire")
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `uv run python examples/check_ground_locg_mlx_static.py`
Expected: FAIL with `AttributeError: module 'ground_locg_mlx_shimmed' has no attribute '_compute_sas_metal'`

- [ ] **Step 3: Implement the minimal code to make the test pass**

Add to `rqutils/ground_locg_mlx.py` after `apply_h_xz_mlx_metal` (line 230). Note `T acc` is per-thread, `partials` is threadgroup-shared:

```python
_METAL_SAS_SOURCE = """
    // One threadgroup per (i, j) pair with i <= j; one thread per stride-slice of the vectors.
    uint pair = threadgroup_position_in_grid.x;
    uint lane = thread_position_in_threadgroup.x;
    uint lanes = threads_per_threadgroup.x;

    // Unrank `pair` into (i, j) with i <= j. n_basis is 2 or 3, so a short scan is cheaper
    // than any closed form and avoids integer-sqrt rounding concerns entirely.
    uint i = 0;
    uint j = 0;
    uint seen = 0;
    for (uint a = 0; a < n_basis; ++a) {
        for (uint b = a; b < n_basis; ++b) {
            if (seen == pair) {
                i = a;
                j = b;
            }
            seen += 1;
        }
    }

    // Strided partial sum in a register. Stride `lanes` keeps adjacent lanes on adjacent
    // addresses, so these loads coalesce.
    T acc = 0;
    for (uint k = lane; k < n_states; k += lanes) {
        acc += vectors[i * n_states + k] * mvs[j * n_states + k];
    }
    partials[lane] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tree reduction over the threadgroup. `lanes` is a power of two (the caller rounds down),
    // so the halving is exact and no lane reads past the written region.
    for (uint half = lanes / 2; half > 0; half /= 2) {
        if (lane < half) {
            partials[lane] += partials[lane + half];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lane == 0) {
        // Write BOTH triangles with the identical value, so the result is exactly symmetric by
        // construction and no separate symmetrization op is needed. For i == j this writes the
        // same slot twice, which is harmless.
        out[i * n_basis + j] = partials[0];
        out[j * n_basis + i] = partials[0];
    }
"""

_METAL_SAS_KERNEL = None


def _get_metal_sas_kernel():
    """Build (once) the fused Rayleigh-Ritz inner-product kernel."""
    global _METAL_SAS_KERNEL
    if _METAL_SAS_KERNEL is None:
        _METAL_SAS_KERNEL = mx.fast.metal_kernel(
            name="sqd_compute_sas",
            input_names=["vectors", "mvs", "n_basis", "n_states"],
            output_names=["out"],
            source=_METAL_SAS_SOURCE,
            header="#include <metal_stdlib>\n",
        )
    return _METAL_SAS_KERNEL


def _compute_sas_metal(vectors, mvs, threadgroup=256):
    """Return the (n x n) matrix of <v_i | A | v_j> in a single fused Metal launch.

    The op-graph :func:`_compute_sas` costs 16 op launches per iteration (measured) to produce
    nine numbers, of which only six are distinct. This computes all six distinct inner products
    in one launch: one threadgroup per (i, j) pair with i <= j, a strided per-thread partial sum,
    then a threadgroup-memory tree reduction.

    Thread 0 of each threadgroup writes **both** ``out[i*n + j]`` and ``out[j*n + i]``, so the
    result is exactly symmetric by construction -- stronger than the op-graph path's
    ``(sas + sas.T) * 0.5``, which averages two values that differ by rounding -- and the
    symmetrization op disappears rather than merely being fused.

    The tree reduction changes summation order relative to ``mx.sum``. For dot products of
    unit-norm vectors this is benign and typically *more* accurate than sequential summation
    (error growing as log n rather than n), but it is a change: see
    ``examples/check_ground_locg_mlx_static.py`` case 3h, which pins agreement with both the
    op-graph path and a direct ``v @ m``.

    Metal has no float64, so this path is f32-only, exactly like :func:`apply_h_xz_mlx_metal`.
    The f64 arms must keep using :func:`_compute_sas`.

    Args:
        vectors: Basis vectors, a tuple of 2 or 3 arrays of shape ``(N,)``, float32.
        mvs: Their images under A, same length and shapes, float32.
        threadgroup: Maximum threads per threadgroup. Rounded down to a power of two so the
            tree reduction halves exactly.

    Returns:
        The ``(n, n)`` matrix of ``<v_i | A | v_j>``, exactly symmetric.

    Raises:
        ValueError: If the inputs are not float32, since Metal has no float64.
        ValueError: If ``vectors`` and ``mvs`` differ in length, or the length is not 2 or 3.
    """
    if len(vectors) != len(mvs):
        raise ValueError(f"vectors and mvs must have equal length, got {len(vectors)}/{len(mvs)}")
    num_basis = len(vectors)
    if num_basis not in (2, 3):
        raise ValueError(f"_compute_sas_metal supports a basis of 2 or 3, got {num_basis}")
    for name, arrays in (("vectors", vectors), ("mvs", mvs)):
        for array in arrays:
            if array.dtype != mx.float32:
                raise ValueError(
                    f"_compute_sas_metal requires float32 (Metal has no float64), got "
                    f"{array.dtype} in {name}. Use _compute_sas for the f64 arms."
                )

    num_states = vectors[0].shape[0]
    stacked_v = mx.stack(vectors)
    stacked_m = mx.stack(mvs)
    num_pairs = num_basis * (num_basis + 1) // 2

    # Round the threadgroup size down to a power of two: the tree reduction halves `lanes` until
    # it reaches 1, which only visits every written lane exactly once if it starts as a power of
    # two. Also cap it at num_states so no lane sits idle with nothing to accumulate.
    lanes = 1
    while lanes * 2 <= min(threadgroup, num_states):
        lanes *= 2

    # No output initialization is needed (and no `init_value=` is passed -- that kwarg is not in
    # MLX 0.32.0's documented call signature): the pairs (i, j) with i <= j cover every slot of
    # the (n, n) output once the j > i writes are mirrored, so every element is written before it
    # is read. Do not "fix" this by zero-filling first; that would add back a launch.
    kernel = _get_metal_sas_kernel()
    outputs = kernel(
        inputs=[stacked_v, stacked_m, num_basis, num_states],
        template=[("T", mx.float32)],
        # grid is in THREADS, not threadgroups: one threadgroup of `lanes` threads per pair.
        grid=(num_pairs * lanes, 1, 1),
        threadgroup=(lanes, 1, 1),
        output_shapes=[(num_basis, num_basis)],
        output_dtypes=[mx.float32],
    )
    return outputs[0]
```

Then extend the shim so it dispatches on kernel name. In `examples/check_ground_locg_mlx_static.py`, replace the body of `_shim_metal_kernel` (lines 86-116) so `name` selects an implementation:

```python
def _shim_metal_kernel(name, input_names, output_names, source, **kwargs):
    """Stand in for mx.fast.metal_kernel by interpreting each kernel's arithmetic in numpy.

    Dispatches on `name`: this module now has two kernels. Reproducing their per-thread and
    per-threadgroup indexing here lets the static check verify the CALLERS (shapes, dtypes, grid
    setup, flat row-major indexing, threadgroup rounding) without a Metal device. It does NOT
    verify that the Metal source compiles, that the barriers are correct, or that it is right on
    device -- only the user's real-hardware run can.
    """

    def call_matvec(inputs, output_dtypes=None, **kw):
        vec, xsources, diagonals, num_groups, num_states = inputs
        xs_flat = np.asarray(xsources).reshape(-1)
        dg_flat = np.asarray(diagonals).reshape(-1)
        vec_np = np.asarray(vec)
        out = np.zeros(num_states, dtype=np.dtype(output_dtypes[0]))
        for j in range(num_groups):
            off = j * num_states
            out = out + vec_np[xs_flat[off : off + num_states]] * dg_flat[off : off + num_states]
        return [out]

    def call_sas(inputs, grid=None, threadgroup=None, output_dtypes=None, **kw):
        vectors, mvs, num_basis, num_states = inputs
        vec_np = np.asarray(vectors).reshape(-1)
        mv_np = np.asarray(mvs).reshape(-1)
        lanes = threadgroup[0]
        num_pairs = num_basis * (num_basis + 1) // 2
        assert grid[0] == num_pairs * lanes, (
            f"grid {grid[0]} != num_pairs*lanes {num_pairs * lanes}: the caller must launch one "
            "threadgroup per (i, j) pair"
        )
        assert lanes & (lanes - 1) == 0, f"threadgroup {lanes} is not a power of two"
        out = np.zeros((num_basis, num_basis), dtype=np.dtype(output_dtypes[0]))
        # Reproduce the kernel's own pair-unranking scan, strided partial sums, and tree
        # reduction -- not just the mathematical result -- so an indexing error here shows up.
        pairs = [(a, b) for a in range(num_basis) for b in range(a, num_basis)]
        for pair, (i, j) in enumerate(pairs):
            partials = np.zeros(lanes, dtype=np.dtype(output_dtypes[0]))
            for lane in range(lanes):
                acc = np.dtype(output_dtypes[0]).type(0)
                for k in range(lane, num_states, lanes):
                    acc += vec_np[i * num_states + k] * mv_np[j * num_states + k]
                partials[lane] = acc
            half = lanes // 2
            while half > 0:
                partials[:half] += partials[half : half * 2]
                half //= 2
            out[i, j] = partials[0]
            out[j, i] = partials[0]
        return [out]

    if name == "sqd_apply_h_xz":
        return call_matvec
    if name == "sqd_compute_sas":
        return call_sas
    raise AssertionError(f"no shim implementation for Metal kernel {name!r}")
```

- [ ] **Step 4: Run the tests and make sure they pass**

Run: `uv run python examples/check_ground_locg_mlx_static.py`
Expected: PASS, including the new `OK  _compute_sas_metal n=2 ...`, `n=3`, and the float64-guard lines. The pre-existing `3g` matvec lines must still pass — that proves the shim's name dispatch didn't break the first kernel.

Run: `uv run --extra dev ruff check rqutils/ examples/ && uv run --extra dev ruff format --check rqutils/ examples/ && uv run --extra dev ty check rqutils/ examples/ && uv run --extra dev pytest -q`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add rqutils/ground_locg_mlx.py examples/check_ground_locg_mlx_static.py
git commit -m "Add a fused Metal kernel for the Rayleigh-Ritz inner products"
```

---

### Task 3: Wire the kernel into `ground_locg_mlx`

**Files:**
- Modify: `rqutils/ground_locg_mlx.py` (`ground_locg_mlx` signature at line 233; call sites at lines 313 and 350)
- Test: `examples/check_ground_locg_mlx_static.py`

**Interfaces:**
- Consumes: `_compute_sas_metal(vectors, mvs, threadgroup=256)` from Task 2.
- Produces: `ground_locg_mlx(mat, xinit, args=(), maxiter=1000, tol=None, compile_body=False, compile_chunk=10, sas="ops")` — one new trailing keyword-only-by-convention parameter, default `"ops"` preserving current behaviour exactly.

**Why selection happens once, before the loop:** the spec flags that reading `.dtype` inside the compiled body might force a sync (unverifiable in a headless session). Selecting the implementation once, before iteration begins, sidesteps the question entirely and is correct regardless, since the dtype is fixed for the whole solve.

- [ ] **Step 1: Write the failing test**

Append to `examples/check_ground_locg_mlx_static.py`, before the final summary print:

```python
# 3i. sas="metal" must be a strict no-op on the trajectory relative to sas="ops" under the
# shim, where both reduce to numpy arithmetic. Fixed-iteration mode (tol=0.) so no convergence
# check can mask a divergence: same ops, same order, same iteration count.
inputs32_v = inputs.vinit.astype(np.float32)
xs32_i = inputs.xsources.astype(np.int32)
dg32_i = inputs.diagonals.astype(np.float32)
eig_ops, _, it_ops, _ = ground_locg_mlx(
    apply_h_xz_mlx, inputs32_v, args=(xs32_i, dg32_i), maxiter=30, tol=0.0, sas="ops"
)
eig_met, _, it_met, _ = ground_locg_mlx(
    apply_h_xz_mlx, inputs32_v, args=(xs32_i, dg32_i), maxiter=30, tol=0.0, sas="metal"
)
assert it_ops == it_met == 30, f"iteration counts differ: {it_ops} vs {it_met}"
scale_eig = max(1.0, abs(eig_ops))
assert abs(eig_met - eig_ops) < 1e-4 * scale_eig, (
    f"sas='metal' changed the fixed-iteration eigenvalue: {eig_met} vs {eig_ops}"
)
print(f"OK  sas='metal' tracks sas='ops' (eig {eig_met:.6f} vs {eig_ops:.6f}, {it_met} iters)")

# sas="metal" must refuse float64 rather than silently running a different kernel.
try:
    ground_locg_mlx(
        apply_h_xz_mlx, inputs.vinit, args=(inputs.xsources, inputs.diagonals), sas="metal"
    )
except ValueError as exc:
    assert "float32" in str(exc), f"unexpected guard message: {exc}"
    print("OK  ground_locg_mlx(sas='metal') rejects float64 input")
else:
    raise AssertionError("ground_locg_mlx(sas='metal') accepted float64 -- guard did not fire")

# An unknown sas value must fail loudly, not fall through to a default.
try:
    ground_locg_mlx(apply_h_xz_mlx, inputs32_v, args=(xs32_i, dg32_i), maxiter=2, sas="bogus")
except ValueError as exc:
    assert "bogus" in str(exc), f"unexpected message: {exc}"
    print("OK  ground_locg_mlx rejects an unknown sas value")
else:
    raise AssertionError("ground_locg_mlx accepted sas='bogus'")
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `uv run python examples/check_ground_locg_mlx_static.py`
Expected: FAIL with `TypeError: ground_locg_mlx() got an unexpected keyword argument 'sas'`

- [ ] **Step 3: Implement the minimal code to make the test pass**

Change the signature at `rqutils/ground_locg_mlx.py:233`:

```python
def ground_locg_mlx(
    mat,
    xinit,
    args=(),
    maxiter=1000,
    tol=None,
    compile_body=False,
    compile_chunk=10,
    sas="ops",
):
```

Add to the docstring's `Args:` block, after `compile_chunk`:

```
        sas: Which Rayleigh-Ritz inner-product implementation to use. ``"ops"`` (default) is
            the portable op-graph :func:`_compute_sas`, and reproduces this function's
            behaviour exactly as it was before this parameter existed. ``"metal"`` uses the
            fused single-launch :func:`_compute_sas_metal`, which requires float32 (Metal has
            no float64) and so raises on an f64 solve. The choice is made once here rather than
            per iteration, so it cannot introduce a per-iteration device sync.
```

Add to the `Raises:` block:

```
        ValueError: If ``sas`` is not ``"ops"`` or ``"metal"``, or if ``sas="metal"`` is
            combined with a non-float32 ``xinit``.
```

Then, immediately after the existing complex-input guard block (after line 279) and before `check_convergence = tol != 0.0`:

```python
    if sas not in ("ops", "metal"):
        raise ValueError(f"sas must be 'ops' or 'metal', got {sas!r}")
    if sas == "metal" and xinit.dtype != mx.float32:
        # Fail here rather than at the first iteration's kernel call, so the error names the
        # parameter the caller actually set. Metal has no float64.
        raise ValueError(
            f"sas='metal' requires float32 (Metal has no float64), got {xinit.dtype}. Use "
            "sas='ops' for an f64 solve."
        )
    # Bind the implementation once, before iterating: the dtype is fixed for the whole solve, so
    # a per-iteration dispatch would buy nothing and might force a device sync inside the
    # compiled body (see the design doc -- unverified, and avoided rather than risked).
    compute_sas = _compute_sas_metal if sas == "metal" else _compute_sas
```

Replace line 313:

```python
    sas = _compute_sas((xcurr, tmp_p), (ax, matvec(tmp_p)))
```

with:

```python
    sas_mat = compute_sas((xcurr, tmp_p), (ax, matvec(tmp_p)))
```

**Then rename every subsequent use of the local `sas` in the iteration-1 block to `sas_mat`** (the `if r_is_zero:` block that does `sas = sas + mx.stack(...)`, the `sas[1, 1]` read, the `mx.array(0.0, sas.dtype)` calls, and `eigenpair_2x2(sas)`). The parameter is now named `sas`, so leaving the local shadowing it would make the code unreadable and would break the `r_is_zero` path.

In `iter_body`, replace line 350:

```python
        sas = _compute_sas((xcurr, ycurr, tmp_p), (axcurr, matvec(ycurr), matvec(tmp_p)))
```

with:

```python
        sas = compute_sas((xcurr, ycurr, tmp_p), (axcurr, matvec(ycurr), matvec(tmp_p)))
```

`iter_body`'s local `sas` does not collide with the parameter (it is assigned before use in that scope), but rename it to `sas_mat` there too for consistency if ruff flags shadowing.

- [ ] **Step 4: Run the tests and make sure they pass**

Run: `uv run python examples/check_ground_locg_mlx_static.py`
Expected: PASS. Critically, the pre-existing `3b`/`3e`/`3f` checks must still report the **same eigenvalue and iteration count as before this task** — they exercise the default `sas="ops"` path, which must be byte-identical.

Run: `uv run --extra dev ruff check rqutils/ examples/ && uv run --extra dev ruff format --check rqutils/ examples/ && uv run --extra dev ty check rqutils/ examples/ && uv run --extra dev pytest -q`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add rqutils/ground_locg_mlx.py examples/check_ground_locg_mlx_static.py
git commit -m "Let ground_locg_mlx select the fused Metal inner-product kernel"
```

---

### Task 4: `--sas` benchmark flag

**Files:**
- Modify: `examples/bench_mlx.py` (argparse block around line 136; `_time_mlx` at line 425; the JAX rejection pattern at line 381; the result dict at line 537)
- Test: `examples/check_bench_mlx.py`

**Interfaces:**
- Consumes: `ground_locg_mlx(..., sas=...)` from Task 3.
- Produces: `options.sas` (`"ops"` | `"metal"`), threaded into both `ground_locg_mlx` call sites in `_time_mlx` (`fixed()` and `solve()`), and reported in the result dict as `"sas"`.

- [ ] **Step 1: Write the failing test**

**Match the existing file's design:** `examples/check_bench_mlx.py` drives `bench_mlx.py` through `subprocess` and never imports it, deliberately — "mlx.core loads a Metal device on import even for mx.cpu arrays, so those arms can only be verified on the human partner's hardware" (its own module docstring). Do **not** add a direct `from bench_mlx import ...`; that would make this headless checker require a GPU. Use the existing `BASE` command list and `subprocess.run`.

Append to `examples/check_bench_mlx.py`:

```python
# 6. --sas: a JAX arm must refuse it (MLX-only kernel), and the flag must be accepted by the
# parser. Driven through subprocess like every other check here -- importing bench_mlx would
# pull in mlx.core and need a Metal device, which is exactly what this file avoids.
out = subprocess.run(
    BASE + ["--arm", "jax-cpu-f32", "--sas", "metal"], capture_output=True, text=True, check=False
)
assert out.returncode != 0, f"jax-cpu-f32 accepted --sas metal:\n{out.stdout}"
assert "sas" in (out.stdout + out.stderr), (
    f"rejection did not mention sas:\n{out.stdout}\n{out.stderr}"
)
print("OK  --sas metal refused for a jax arm")

# --sas ops must be an explicit no-op: same eigenvalue as omitting the flag entirely, proving
# the default path is untouched.
out_default = subprocess.run(
    BASE + ["--arm", "jax-cpu-f64", "--json"], capture_output=True, text=True, check=False
)
out_ops = subprocess.run(
    BASE + ["--arm", "jax-cpu-f64", "--sas", "ops", "--json"],
    capture_output=True,
    text=True,
    check=False,
)
assert out_ops.returncode == 0, f"--sas ops rejected for a jax arm:\n{out_ops.stderr}"
row_default = json.loads(out_default.stdout)
row_default = row_default["results"][0] if "results" in row_default else row_default
row_ops = json.loads(out_ops.stdout)
row_ops = row_ops["results"][0] if "results" in row_ops else row_ops
assert row_ops["eigval"] == row_default["eigval"], (
    f"--sas ops changed a jax arm's eigenvalue: {row_ops['eigval']} vs {row_default['eigval']}"
)
print("OK  --sas ops is a no-op for a jax arm (identical eigenvalue)")

# An unknown --sas value must be rejected by argparse rather than falling through to a default.
out = subprocess.run(
    BASE + ["--arm", "jax-cpu-f64", "--sas", "bogus"], capture_output=True, text=True, check=False
)
assert out.returncode != 0, "--sas bogus was accepted"
print("OK  --sas bogus rejected")
```

Note this exercises `--sas` on JAX arms only, which is all that is possible headless. The `mlx-cpu-f64` + `--sas metal` rejection and the `mlx-*-f32` happy path are verifiable only on hardware; they are covered by Task 5's checker and Task 6's sweep.

- [ ] **Step 2: Run it to make sure it fails**

Run: `uv run python examples/check_bench_mlx.py`
Expected: FAIL at the first new assertion. `--sas` does not exist yet, so argparse exits non-zero for *every* invocation using it — including `--sas metal` on the JAX arm. That makes the first `assert out.returncode != 0` pass for the wrong reason, and the run then fails on the following `assert "sas" in ...` (argparse's message says `unrecognized arguments: --sas metal`, which does contain "sas") and finally on `assert out_ops.returncode == 0` with `--sas ops rejected for a jax arm`.

Because argparse's "unrecognized argument" error can accidentally satisfy the earlier assertions, confirm the flag is genuinely absent first:

Run: `uv run python examples/bench_mlx.py --help | grep -c -- --sas`
Expected: `0`

- [ ] **Step 3: Implement the minimal code to make the test pass**

In the argparse block of `examples/bench_mlx.py`, after `--compile-body`:

```python
    parser.add_argument(
        "--sas",
        choices=("ops", "metal"),
        default="ops",
        help="Rayleigh-Ritz inner-product kernel, MLX f32 arms only. \"ops\" "
        "(default) is the portable op-graph _compute_sas -- existing measured "
        "results are only reproducible with this default. \"metal\" fuses all "
        "six distinct inner products into one custom Metal launch, replacing "
        "16 op launches per iteration and eliminating the symmetrization.",
    )
```

In `run_arm`, beside the existing `--matvec metal` JAX rejection (near line 381):

```python
    if options.sas == "metal" and framework != "mlx":
        # Fail loudly rather than silently substituting a different kernel: a "metal" row that
        # actually timed the op-graph path would misreport what was measured.
        raise SystemExit(
            f"{arm}: --sas metal is an MLX-only custom Metal kernel and has no JAX equivalent."
        )
```

In `_time_mlx`, after the `matvec_fn` selection block:

```python
    if options.sas == "metal" and precision != "f32":
        raise SystemExit(
            f"{arm}: --sas metal requires float32 (Metal has no float64). Use an f32 arm, or "
            "--sas ops for the f64 arms."
        )
```

Add `sas=options.sas` to **both** `ground_locg_mlx` calls in `_time_mlx` (inside `fixed()` and inside `solve()`), and add `"sas": options.sas` to the returned result dict so every row records which kernel it timed.

- [ ] **Step 4: Run the tests and make sure they pass**

Run: `uv run python examples/check_bench_mlx.py`
Expected: PASS

Run: `uv run --extra dev ruff check rqutils/ examples/ && uv run --extra dev ruff format --check rqutils/ examples/ && uv run --extra dev ty check rqutils/ examples/ && uv run --extra dev pytest -q`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add examples/bench_mlx.py examples/check_bench_mlx.py
git commit -m "Add --sas to select the fused Metal inner-product kernel in the benchmark"
```

---

### Task 5: Real-device checker cases — **REQUIRES HARDWARE, hand to the user**

**Files:**
- Modify: `examples/check_ground_locg_mlx_mlx.py` (loop at lines 34-69)

**Interfaces:**
- Consumes: `ground_locg_mlx(..., sas=...)` from Task 3, `_compute_sas_metal` from Task 2.
- Produces: nothing consumed by later tasks.

The implementing agent **writes** this code but **cannot run it** — this session has no Metal device. Write it, commit it, and report to the user that it is unverified.

- [ ] **Step 1: Add the `sas="metal"` arm to the checker**

In `examples/check_ground_locg_mlx_mlx.py`, inside the existing device/dtype loop, after the current `ground_locg_mlx` call and its `ok` check, add:

```python
            # sas='metal' is f32-only (Metal has no float64), so only exercise it on the f32
            # arms. This is the ONLY check that can establish the Metal source actually
            # compiles and that its barriers are correct -- the numpy shim cannot.
            if dtname == "f32":
                eig_sas, _, iters_sas, _ = ground_locg_mlx(
                    apply_h_xz_mlx, v0, args=(xs, dg), sas="metal"
                )
                ok_sas = abs(eig_sas - ref) < rtol * max(1.0, abs(ref))
                print(
                    f"{arm} sas=metal: eig={eig_sas:.10f} iters={iters_sas} "
                    f"{'OK' if ok_sas else 'FAIL'}"
                )
                if not ok_sas:
                    failures.append(f"{arm}-sas-metal")
```

- [ ] **Step 2: Verify it at least parses and lints**

Run: `uv run --extra dev ruff check examples/check_ground_locg_mlx_mlx.py && uv run --extra dev ruff format --check examples/check_ground_locg_mlx_mlx.py && uv run python -c "import ast; ast.parse(open('examples/check_ground_locg_mlx_mlx.py').read())"`
Expected: PASS. This is the *only* verification available headless — it does not run the checker.

- [ ] **Step 3: Commit, marked unverified**

```bash
git add examples/check_ground_locg_mlx_mlx.py
git commit -m "Add a real-device check for the fused Metal inner-product kernel

Written but not executed: this session has no Metal device, so only the numpy-shim
counterpart validated the algorithm. Requires a user run to confirm the Metal
source compiles and the barriers are correct."
```

- [ ] **Step 4: Hand to the user**

Report, verbatim, that the following must be run on a Mac with GPU access, and that **until it passes, the kernel is unproven on device**:

```bash
uv run --extra mlx python examples/check_ground_locg_mlx_mlx.py
```

Expected output: `FAILURES: none`, with `sas=metal` lines for `mlx-cpu-f32` and `mlx-gpu-f32`.

---

### Task 6: The N sweep — **REQUIRES HARDWARE, hand to the user**

**Files:**
- Modify: `docs/superpowers/specs/2026-08-04-metal-sas-kernel-design.md` (the empty Results table)

**Interfaces:** Consumes Tasks 1-5. Produces the measurement the design's success criterion 6 requires.

- [ ] **Step 1: Hand the sweep to the user**

The implementing agent cannot run this. Give the user these commands, noting that `--sas ops` and `--sas metal` must be run pairwise at each N with everything else held constant:

```bash
for N in 893 4000 20000; do
  for SAS in ops metal; do
    uv run --extra mlx python examples/bench_mlx.py --arm mlx-gpu-f32 \
      --matvec metal --compile-body --sas $SAS \
      --num-qubits 14 --num-paulis 100 --num-states $N --json
  done
done
```

- [ ] **Step 2: Record the results honestly**

Fill the design doc's Results table from the user's output. Rules, from the spec:

- A row whose gate failed gets **no timing number** — record the gate failure instead. The f32 arms were measured failing above ~1000 states at n=14 (5.4e-3 vs rtol 1e-4), so some large-N rows may legitimately have no timing.
- Do not extrapolate or estimate any cell. Empty stays empty.
- If `metal` is slower than `ops` at some N, record that. The spec predicts the win shrinks with N; a null or negative result at large N **is** the finding, not a failure.
- Note which N values, if any, took the new sparse gate path (N > 5000).

- [ ] **Step 3: Commit the measurements**

```bash
git add docs/superpowers/specs/2026-08-04-metal-sas-kernel-design.md
git commit -m "Record the measured N sweep for the fused Metal inner-product kernel"
```

---

## Self-Review

**1. Spec coverage.**

| Spec section | Task |
|---|---|
| `_compute_sas_metal` kernel, threadgroup reduction, both-triangles write | Task 2 |
| Templating on basis size n ∈ {2,3} | Task 2 (`num_basis` input, pair-unranking scan) |
| fp32-only guard mirroring the matvec's | Task 2 (function) + Task 3 (call-site guard) |
| Both implementations stay in step, enforced by a check | Task 2 Step 1 (asserts `_compute_sas_metal` vs `_compute_sas`) |
| Dispatch composes with `mx.compile`; hoist if `.dtype` syncs | Task 3 (bound once before the loop — the spec's stated fallback, adopted unconditionally) |
| `_project_out` / `eigenpair_3x3` untouched | Global Constraints (explicit prohibition) |
| Sparse `eigsh` gate path, N=5000 threshold, both-paths agreement | Task 1 |
| `eigsh` non-convergence / non-finite handling | Task 1 (`ArpackNoConvergence` + `isfinite`) |
| Shim dispatches on kernel `name` | Task 2 Step 3 |
| Shim simulates per-thread *and* per-threadgroup indexing incl. tree | Task 2 Step 3 (`call_sas`) |
| Exact-symmetry assertion | Task 2 Step 1 (`asym == 0.0`) |
| float64-guard assertion | Task 2 Step 1, Task 3 Step 1 |
| Real-device checker cases | Task 5 |
| `--sas` flag, defaulting to `ops`, refused for JAX/f64 | Task 4 |
| N sweep, gate status per row, no estimates | Task 6 |
| Success criteria 1-5 (checkers, lint, types, pytest) | Step 4 of Tasks 1-4 |
| Success criterion 6 (measurement) | Task 6 |

No gaps.

**2. Placeholder scan.** No TBD/TODO. Every code step carries real code. Tasks 5 and 6 have no code to write beyond what is shown, and their "hand to the user" steps are concrete commands with expected output, not deferrals.

**3. Type consistency.**

- `_compute_sas_metal(vectors, mvs, threadgroup=256)` — same name and argument order in Task 2 (definition), Task 2 Step 1 (test), Task 3 (bound as `compute_sas`), Task 5 (imported).
- Kernel `name="sqd_compute_sas"` in Task 2's builder matches the shim's dispatch string in Task 2 Step 3.
- `input_names=["vectors", "mvs", "n_basis", "n_states"]` matches the Metal source's identifiers and the shim's unpack order `vectors, mvs, num_basis, num_states`.
- `sas` parameter spelled identically in Task 3 (`ground_locg_mlx(..., sas="ops")`), Task 4 (`options.sas`, `sas=options.sas`), Task 5 (`sas="metal"`).
- `sparse_reference(inputs) -> float` and `DENSE_REFERENCE_MAX_DIM` consistent between Task 1's definition and its test.
- One hazard flagged explicitly in Task 3 Step 3: the new `sas` **parameter** shadows the existing **local** `sas` in the iteration-1 block, which is why that step spells out the `sas` → `sas_mat` rename rather than leaving it to be discovered.
