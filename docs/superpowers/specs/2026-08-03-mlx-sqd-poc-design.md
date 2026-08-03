# PoC: port the SQD solver loop to MLX and benchmark CPU vs GPU

Date: 2026-08-03
Status: approved, not yet implemented

## Goal

Answer one question with measured numbers: **is MLX faster than JAX for the SQD
eigensolver loop on Apple silicon, and does the Metal GPU help?**

This is a proof of concept, not a port of `rqutils`. Nothing under `rqutils/`
changes. All new code lives in `examples/`, so the library gains no import-time
dependency on `mlx` and needs no docs/toctree entry.

## Scope

Ported to MLX: **the solver loop only** — `ground_locg` plus the
`apply_h_xz_cached` matvec (i.e. `cache_level=(1, 2)`).

Not ported: `uniquify_states`, `get_xsource`, `get_diagonal`,
`get_diag_signs`, and the other five matvec kernels. These need MLX
equivalents for `jnp.bitwise_count`, `jnp.bincount`, `jnp.searchsorted`, and
multi-key `jax.lax.sort` — none of which MLX has (verified against
`include/mlx/ops.h` in mlx 0.32.0). Adding them is a port project, not a PoC.

The `(1, 2)` cache level was chosen because `apply_h_xz_cached` is pure
gather–multiply–add: it needs none of the missing primitives, and it is the
O(maxiter) hot path that dominates a real solve.

## Verified facts this design rests on

Established by running code in the repo, not assumed:

1. **`ham.c` is complex128 even with `force_real=True`.** The `(-i)^{x·z}`
   phase in `PauliSumXZ.from_paulisum` makes any Pauli string with an odd
   number of `Y`s carry an imaginary coefficient. `force_real` only zeroes the
   imaginary part of the *input* coefficients, before phasing.
2. **Pauli strings with an even number of `Y`s give `c.dtype == float64`.**
   Confirmed by construction.
3. **MLX has no `complex128`.** Its `Dtype` enum (`include/mlx/dtype.h`)
   defines `complex64` only. `float64` appears under `backend/cpu/` but not
   `backend/metal/`.
4. **`hproj` is broken and cannot be the reference.** For n=8 it raises
   `TypeError: Cannot concatenate arrays with shapes that differ ...
   (137, 1), (137, 2)` — it builds the Hamiltonian with `add_padding=True`
   but packs states without the pad bit, so the byte counts disagree. This
   confirms and sharpens the note in CLAUDE.md.
5. **`sqd` itself is correct.** A brute-force check (build the full 2^n matrix
   from the Pauli strings, project onto the unique states, `eigvalsh`) gives
   -2.9998846658233185 against `sqd`'s -2.9998846658233163 — agreement to
   2e-15.
6. **`apply_h_xz_cached` equals the explicit matrix.** Building
   `H[i, xsources[j][i]] += diagonals[j][i]` and comparing `H @ v` to
   `apply_h_xz_cached(v, xs, dg)` gives max error 4.4e-16, and
   `|H - H†| == 0` exactly. This is the equivalence the correctness gate uses.

Consequence of 1–3: **a complex Hamiltonian cannot run in double precision
under MLX at all.** The benchmark therefore generates **even-Y Pauli strings**
so the Hamiltonian is real. This is not a contrivance to make MLX look good —
real Hamiltonians of exactly this form arise in practice (e.g. Jordan–Wigner
electronic structure with real orbitals) — but it is a genuine restriction on
what this PoC measures, and must be stated in the results.

## Architecture

Three files under `examples/`:

| file | contents |
|---|---|
| `_bench_common.py` | problem generation (seeded, even-Y), setup via JAX, `timeit` helpers |
| `ground_locg_mlx.py` | the port: MLX single-vector LOBPCG + `apply_h_xz_mlx` |
| `bench_mlx.py` | arm selection, correctness gate, timing, reporting |

`_bench_common.py` is shared so that every arm — and the existing JAX-only
`bench.py` — generates a byte-identical problem from the same seed. Duplicating
the generator is how the arms would silently drift out of comparability.

### Data flow

```
numpy: pauli strings (even-Y), coeffs, states
  |
  |-- PauliSumXZ.from_paulisum(force_real=True, add_padding=True)
  |
  |-- JAX, cache_level=(1,2), on CPU, x64:
  |      uniquify_states -> get_xsource per X group -> get_diagonal per group
  |      => xsources int32[J, N], diagonals float64[J, N], vinit float64[N]
  |
  |-- sanitize (see below), then np.asarray / mx.array, cast per arm dtype
  |
  '-- timed region: solver only
         JAX arms: ground_locg(apply_h_xz_cached, vinit, (xsources, diagonals))
         MLX arms: ground_locg_mlx(apply_h_xz_mlx, vinit, (xsources, diagonals))
```

Setup runs **once**, in JAX, and is **not timed**. Every arm receives identical
`xsources` / `diagonals` / `vinit`, so any timing difference is attributable to
the solver alone. `setup_s` is reported for context but labelled as shared and
untimed.

## The port

### `apply_h_xz_mlx`

```python
def apply_h_xz_mlx(vec, xsources, diagonals):
    out = mx.zeros_like(vec)
    for j in range(xsources.shape[0]):     # J is static and small -> plain loop
        out = out + mx.take(vec, xsources[j]) * diagonals[j]
    return out
```

`jax.lax.scan` becomes a Python `for`. J (distinct X signatures) is static and
typically tens, so unrolling into MLX's lazy graph is correct and cheap.

**Invalid-source handling.** JAX's
`.at[].get(mode='fill', fill_value=0., wrap_negative_indices=False)` maps
`xsource == -1` (no source state in the subspace) to `0.0`. MLX's `take` has no
fill mode, and its out-of-bounds behaviour is undocumented — `-1` would likely
wrap numpy-style, which is silently wrong. About 48% of entries are `-1` in a
representative instance, so this is not an edge case.

Fix, applied **at setup, outside the timed region**:

```python
valid = xsources >= 0
xsources_s = np.where(valid, xsources, 0)      # any in-bounds index
diagonals_s = np.where(valid, diagonals, 0.)   # multiplied by zero anyway
```

`take` then gathers an arbitrary element and multiplies it by `0.0`.
Algebraically identical to the JAX path, zero cost inside the loop, and it
removes any dependence on undocumented `take` semantics. Both JAX and MLX arms
use the sanitized arrays, so neither is advantaged.

### `ground_locg_mlx`

A transcription of `_ground_locg_callable`, preserving `_project_out`,
`eigenpair_2x2`, `eigenpair_3x3` (Cardano) and the
`body_iter0` / `body_iter1` / `body` structure. Signature matches
`ground_locg`: `mat` is a callable taking `(vec, *args)`.

Substitutions:

| JAX | MLX |
|---|---|
| `jnp.sum`, `jnp.sqrt`, `jnp.where` | `mx.sum`, `mx.sqrt`, `mx.where` |
| `jnp.linalg.norm` | `mx.linalg.norm` |
| `jnp.cross` | `mx.linalg.cross` |
| `sas.at[i, j].set(v)` | build the 3x3 by `mx.stack` of scalars |
| `jax.lax.while_loop` | Python `for` + `if float(converged): break` |
| `out_sharding=...` | dropped (MLX has unified memory, no sharding) |

`mx.linalg.cross` and `mx.linalg.norm` exist in `include/mlx/linalg.h`, so
`eigenpair_3x3` ports directly. MLX has no `eigh`, which is precisely why this
codebase hand-rolled the analytic Rayleigh–Ritz step — that choice is what
makes the port feasible.

**Loop strategy (decision A).** Direct transcription with one convergence sync
per iteration. MLX has no `while_loop`/`cond`/`scan` and is lazy-eval, so
reading `converged` in Python forces `mx.eval` — a full device sync. That cost
is real for any MLX user running this algorithm; it is not a measurement
artifact. Two modes keep it from contaminating the speed number:

- `tol=None`, fixed `maxiter`: the predicate is never evaluated, so nothing
  syncs until the end. Clean per-iteration cost.
- `tol=eps`: syncs each iteration, reports the true iteration count.

Deferred: amortizing the sync every k iterations (B), and `mx.compile` over
unrolled chunks (C). Both are follow-ups if A shows sync dominating.

## Benchmark arms

| arm | selection | dtype |
|---|---|---|
| `jax-cpu-f64` | `JAX_PLATFORMS=cpu`, x64 on | float64 |
| `jax-cpu-f32` | `JAX_PLATFORMS=cpu`, x64 off | float32 |
| `jax-metal-f32` | `JAX_PLATFORMS=metal`, x64 off | float32 |
| `mlx-cpu-f64` | `mx.set_default_device(mx.cpu)` | float64 |
| `mlx-cpu-f32` | `mx.set_default_device(mx.cpu)` | float32 |
| `mlx-gpu-f32` | `mx.set_default_device(mx.gpu)` | float32 |

JAX's platform is process-global and x64 must be set before import, so JAX arms
require separate processes. `bench_mlx.py --arm <name>` runs one arm;
`--all` re-executes itself as a subprocess per JAX arm and collates. MLX arms
switch device in-process.

`jax-metal-f32` is attempted and reported as `skipped: <reason>` when the
backend is unavailable — never a hard failure. `mlx-gpu-f64` does not exist:
Metal has no float64.

Including `jax-cpu-f32` is what separates "MLX won" from "fp32 won".

## Correctness gate

Runs on a small instance (n=10, ~200 subspace states) **before any timing**.
No arm's number is printed unless its arm passed.

1. **Reference eigenvalue, computed two independent ways**, used *instead of*
   `hproj`, which raises (finding 4):
   - *Brute force*: build the full 2^n matrix by Kronecker products from the
     Pauli strings, project onto the unique states, `np.linalg.eigvalsh`.
     Measured at 0.134 s for n=10, so it runs in the gate every time.
   - *From solver inputs*: build the dense `H` from the sanitized
     `xsources` / `diagonals` as `H[i, xsources[j][i]] += diagonals[j][i]`,
     assert `|H - H.T|` is at machine zero, then `eigvalsh`.

   Asserting these two agree validates the whole setup chain (packing, padding,
   uniquification, X-source lookup, diagonal composition) against an
   independent construction. Verified at n=10: brute force -2.496495741801
   vs `sqd` -2.4964957418006506. The projected brute-force matrix has
   `max|imag| == 0.0` exactly, confirming finding 2.
2. **Matvec equivalence**: assert `apply_h_xz_mlx(v) ~= H @ v` on a random `v`,
   isolating matvec bugs (notably the `take` fill issue) from solver bugs.
   Tolerance 1e-12 relative for fp64, 1e-5 for fp32.
3. **Eigenvalue agreement**: assert each arm's eigenvalue matches the reference
   within 1e-9 relative (fp64) or 1e-4 relative (fp32).
4. Only then time.

## Output

```
arm            setup_s  compile_s  fixed(100it)_s  per_it_ms  solve_s  iters  eigval
jax-cpu-f64      2.41      0.83        1.204         12.04     0.71     58   -12.4471829
mlx-gpu-f32      2.41      0.02        0.318          3.18     0.19     71   -12.447102
```

Both metrics are reported, per the design decision: `fixed(100it)_s` /
`per_it_ms` is the clean speed comparison (identical work per arm), and
`solve_s` with `iters` is the production-relevant number. Showing both makes it
visible when fp32 is faster per iteration but needs more iterations.
`compile_s` is separated from steady state, as `bench.py` already does.
`--json` emits machine-readable output for diffing runs.

**MLX timing correctness**: MLX is lazy, so the timer must call
`mx.eval(result)` before stopping — the analogue of `jax.block_until_ready`.
Omitting it would measure graph construction and manufacture a bogus MLX win.
This lives in the shared `timeit` so no arm can get it wrong.

## Known limitations, to be stated with any results

1. **Only the solver loop is ported.** Setup still runs in JAX. A full-pipeline
   verdict needs the missing primitives.
2. **The Hamiltonian is restricted to even-Y Pauli strings** so coefficients are
   real. A general Hamiltonian is complex, and MLX has no complex128.
3. **MLX-GPU is fp32-only.** `mlx-gpu-f32` is not numerically comparable to
   `jax-cpu-f64`; that is why the fp32 JAX arms exist.
4. **No sharding.** `rqutils`' multi-device path has no MLX counterpart. This
   PoC says nothing about the distributed case.
5. **The author of this spec could not execute MLX.** `mlx.core` loads a Metal
   device even for `mx.cpu` arrays, and the authoring session was headless
   (`RuntimeError: [metal::load_device] No Metal device available`). The port was
   written and checked statically against the mlx 0.32.0 headers; **no benchmark
   number in this design has been measured.** Expect a debug round-trip on
   first run.

## Out of scope

- Changing anything under `rqutils/`.
- Fixing `hproj` (finding 4) or the `ibit = iterm & 255` bug at
  `rqutils/sqd.py:544`. Both are pre-existing and worth separate issues; the
  `(1, 2)` cache path this PoC uses touches neither.
- Porting `svsim` to MLX.
- Making MLX a declared dependency. It is already in `pyproject.toml`
  uncommitted; whether that stays is a decision for after the numbers exist.
