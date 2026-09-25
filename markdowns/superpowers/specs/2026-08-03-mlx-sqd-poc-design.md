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
   written and checked statically against the mlx 0.32.0 headers. Every number in
   the Results section below was measured by the user on real hardware, not by the
   author; the debug round-trip this predicted did happen, and took two rounds
   (see Results → Correctness).
6. **The MLX arms rebuild their op graph on every timed call**, a cost JAX
   amortizes into `compile_s`. This biases the per-iteration numbers against MLX.
   Quantified in the Results section: it accounts for essentially all of the
   measured MLX time.

## Results

Measured by the user on an Apple M1 (7 GPU cores), mlx 0.32.0, jax 0.11.0, at
`--num-qubits 12 --num-paulis 100 --num-states 1000` (subspace dimension
N=893 after uniquification, J=100 distinct X signatures):

| arm | setup_s | compile_s | per_it_ms | solve_s | iters | matvec_err | eigval |
|---|---|---|---|---|---|---|---|
| jax-cpu-f64 | 0.221 | 0.263 | **0.232** | 0.050 | 217 | 3.55e-15 | -5.3960400377 |
| jax-cpu-f32 | 0.224 | 0.275 | **0.229** | 0.010 | 42 | 1.69e-06 | -5.3958568573 |
| jax-metal-f32 | — | — | skipped | — | — | — | jax-metal not installed |
| mlx-cpu-f64 | 0.237 | 1.487 | **7.608** | 4.623 | 652 | 5.33e-15 | -5.3960400377 |
| mlx-cpu-f32 | 0.249 | 1.509 | **7.874** | 0.314 | 42 | 1.20e-06 | -5.3958401680 |
| mlx-gpu-f32 | 0.226 | 1.501 | **8.354** | 0.344 | 42 | 1.20e-06 | -5.3958587646 |

`setup_s` is the shared, untimed JAX setup, shown for context only.

### Verdict (baseline, before optimization)

> **Superseded — see "Optimization results" below.** The unoptimized numbers in
> the table above led to "MLX is ~34× slower and the Metal GPU does not help."
> Both halves turned out to be artifacts of MLX's default execution model rather
> than properties of MLX or of the M1 GPU. The diagnosis in this subsection is
> what pointed at the fix, so it is kept; the conclusion it reached is not.

As measured with the default `--matvec loop` path, MLX was ~34× slower per
iteration than JAX and the Metal GPU was 6% slower than MLX's own CPU backend.

The *reason* mattered more than the ratio. Per-iteration cost is essentially flat
across the three MLX arms —
7.608 (cpu-f64), 7.874 (cpu-f32), 8.354 (gpu-f32) ms, a 9.8% spread. Note the
direction: **f32 is slower than f64, and the GPU is slower than the CPU.**
Compute-bound work cannot behave that way (f32 moves half the bytes; the GPU has
7 cores). That flatness is the signature of a fixed per-call cost, and it is
large enough to account for essentially all of the MLX time: a `fixed(100it)`
call constructs roughly 126,000 Python-level MLX ops (J=100 groups × 3 ops ×
~4 matvecs × 100 iterations), at ~6.25 µs each. JAX pays this once, because
`jax.lax.scan` traces the loop and then runs entirely in compiled code; MLX,
having no `scan`/`while_loop`, rebuilds the graph on every call.

So **these numbers largely measure Python graph-construction overhead, not MLX
kernel throughput.** Two things would be needed for a verdict on MLX itself:
`mx.compile` over the loop body (deferred by design — see the loop-strategy
decision above), and a substantially larger N. The problem here is tiny: an
893-element vector is ~7 KiB, L1-resident, far too small for a GPU to amortize
kernel-launch latency.

## Optimization results

Acting on that diagnosis, four optimizations were tried. All measurements are
`mlx-*-f32` at the same problem size (n=12, 100 paulis, 1000 states → N=893,
J=100), against `jax-cpu-f32` at **0.229 ms/iter**.

| config | per_it_ms | vs MLX baseline | vs JAX f32 | compile_s |
|---|---|---|---|---|
| cpu, `loop` (baseline) | 8.106 | 1.00× | 35.4× slower | 1.54 |
| cpu, `chunked` (chunk=16) | 4.393 | 1.85× | 19.2× slower | 0.38 |
| cpu, `chunked` + `compile_body` | 2.991 | 2.71× | 13.1× slower | 2.11 |
| cpu, `metal` | 1.404 | 5.77× | 6.1× slower | 0.15 |
| gpu, `metal` | 0.664 | 12.2× | 2.90× slower | 0.10 |
| **gpu, `metal` + `compile_body`** | **0.373** | **21.7×** | **1.63× slower** | 0.06 |
| gpu, `metal` + host Rayleigh–Ritz | 0.910 | 8.9× | 3.97× slower | 0.21 |

**Net: 21.7× faster than the baseline, closing the gap to JAX from 35× to 1.63×.**
Full solve went 0.317 s → 0.0296 s. `matvec_err` stayed at 1.69e-06, identical to
JAX f32's — the speedups are numerically exact, not traded against accuracy.

Two conclusions the baseline verdict got wrong:

1. **The Metal GPU does help — 2.11× over MLX's CPU backend** on the identical
   fused kernel (0.664 vs 1.404 ms). On the op-graph path the GPU was 1.06%
   *slower*. That inversion is direct evidence the original "GPU doesn't help"
   finding was measuring dispatch overhead, not the M1 GPU.
2. **MLX is not inherently ~34× slower.** Nearly all of that gap was MLX's
   default execution model — no `scan`/`while_loop`, so a Python-level loop
   rebuilds the op graph every call and launches one kernel per op.

### What actually governs performance: syncs, then launches

The optimization sequence taught a cost model that the initial op-count reasoning
got wrong, and the correction is the most transferable finding here.

Per-iteration MLX op launches, measured by instrumenting the module with a
counting shim, *after* the fused Metal matvec landed:

| component | ops/iter | share |
|---|---|---|
| `eigenpair_3x3` | 19 | 37.3% |
| `_compute_sas` | 16 | 31.4% |
| `_project_out` | 15 | 29.4% |
| matvec (`metal`) | **1** | **2.0%** |

The matvec — 89,300 gathers — became 2% of the launches, while the Rayleigh–Ritz
step (35 of 51 launches, 69%) does arithmetic on a 3×3 matrix, i.e. nine numbers.
At ~7.3 µs per launch, nine-number scalar math cost ~20× the entire gather. Note
the irony: `rqutils` hand-rolled the analytic 2×2/3×3 eigensolver to save memory
on huge vectors, which is exactly the wrong tradeoff once every op is a launch.

That suggested moving the 3×3 eigensolve to the host with `np.linalg.eigh` —
2 syncs replacing 35 launches, and `eigh` is also 2×10⁷ times more accurate than
the Cardano formulation on the near-degenerate matrices this solver produces
(1.8e-15 vs 3.6e-08), at 4.23 µs per call.

**It was 1.37× slower** (0.910 vs 0.664 ms), and the reason is the lesson: the
model priced a host sync at roughly one kernel launch (~7 µs); the measurement
implies **~389 µs each, ~53× a launch.** A launch is *pipelined* — MLX queues it
and continues — but a sync *drains the pipeline*: every queued op must retire
before the CPU can read the value. MLX being lazy, pulling the 3×3 to numpy
forces the whole iteration graph to complete, destroying the async overlap that
made the fused kernel fast. That option was reverted (commit `4dc8510`).

So: **sync count dominates, launch count is secondary, op-construction count is
tertiary.** This also explains why `compile_body` is the single best lever — it
removes syncs — and it predicts that any optimization moving per-iteration work
to the host will lose, however favourable its op-count arithmetic looks.

### Optimizations, and what remains untried

1. **Chunked gather** (`--matvec chunked --chunk N`, both frameworks) — process
   X-groups in chunks of N, cutting matvec ops from `3J` to `3⌈J/chunk⌉` (300 → 21
   at chunk=16). Chunked rather than fully batched because a full `(J, N)` gather
   costs ~8 GB at N=10⁷ versus ~80 MB, which would defeat SQD's matrix-free
   design. Worth 1.85× on MLX. **Applied to JAX too, where it is a net loss**
   (0.239 → 0.327–0.386 ms at f64): XLA already fuses the unrolled loop, so this
   is not a framework difference but an MLX-specific one.
2. **`mx.compile` on the iteration body** (`--compile-body`, MLX only) — traces
   the body once instead of rebuilding per call. Worth 1.95× alone. Convergence
   is checked between `compile_chunk`-sized batches of iterations, since
   `mx.compile` has no `while_loop` equivalent; that is why the compiled arms show
   50 iterations instead of 42 (overshoot up to `compile_chunk`), which
   incidentally lands *closer* to the f64 reference.
3. **Fused custom Metal kernel** (`--matvec metal`, MLX f32 only) — one
   `mx.fast.metal_kernel` launch computing
   `out[i] = Σⱼ vec[xsources[j,i]] · diagonals[j,i]` with the accumulator in a
   per-thread register: no intermediates, no atomics (thread `i` solely owns
   `out[i]`), coalesced index reads. The single largest win, 5.8× on CPU and
   12.2× on GPU.
4. **Host-side Rayleigh–Ritz** — tried, 1.37× slower, reverted. See above.

Untried, in the order worth trying:

- **Larger N.** Still the decisive measurement. At N=893 the vector is ~7 KiB and
  only 893 threads launch on a 7-core GPU, so every number here remains
  overhead-dominated at a size where the hardware cannot show up. Constraint:
  the f32 arms already fail the eigenvalue gate above ~1000 states at n=14
  (5.4e-3 relative error vs rtol 1e-4), and Metal is fp32-only — so fp32
  convergence at large subspace dimension is a real prerequisite, not just a
  benchmarking nuisance.
- **Fuse Rayleigh–Ritz into a Metal kernel.** The on-device version of
  optimization 4: zero syncs, so unlike the host version it *composes* with
  `compile_body`. Would mean writing the 3×3 eigensolve in Metal and inheriting
  the Cardano near-degeneracy fragility.
- **Tuning** `threadgroup` (256) and `compile_chunk` (10). Flag-only.

## Correctness

Applies to every run above, optimized and unoptimized: the correctness gate is
identical in all of them, and no measurement in this document comes from an arm
that failed it.

The port is faithful. `mlx-cpu-f64` reproduces `jax-cpu-f64`'s eigenvalue to all
ten printed digits (-5.3960400377), and **all three f32 arms converge in exactly
42 iterations** — JAX and MLX, CPU and GPU. At the smaller gate size
(n=10/200 states) both f32 arms converged in exactly 26 iterations, again
matching JAX f32.

Two bugs were found and one non-bug diagnosed along the way:

1. **Silent float64 → float32 truncation on MLX ingest.** `mx.array(x)` converts
   float64 numpy arrays to float32 (documented MLX behaviour), so the original
   `mx.array(x).astype(mx.float64)` upcast already-truncated data — the f64 arm
   ran float64 *arithmetic* on float32-*precision* data. Diagnosed from the
   `matvec_err` column: it read 2.07e-08 (the f32 error floor) on the f64 arm,
   identically to the f32 arms. Fixed by passing dtype at construction, plus a
   dtype assertion so a silent precision downgrade now fails loudly.
2. **A NaN could pass the correctness gate.** The gate was
   `if abs(eigval - reference) > rtol`, and IEEE 754 makes every NaN comparison
   false, so a non-converged NaN eigenvalue was reported as a clean result with a
   full timing row. Fixed with an explicit `np.isfinite` check.
3. **Not a port bug:** `mlx-cpu-f64` needing 652 iterations versus JAX's 217
   traces to `eigenpair_3x3`, whose Cardano formulation loses ~8 digits when two
   eigenvalues are nearly degenerate — the regime a converging LOBPCG enters.
   `rqutils.ground_locg.eigenpair_3x3` (the JAX original) exhibits *bit-identical*
   error, 3.616e-08 on the same 300 near-degenerate matrices, agreeing with the
   port to 15 digits. The port reproduced a pre-existing library characteristic.
   Why JAX tolerates it in 217 iterations while MLX needs 652 is most likely a
   difference in accumulation order (Python loop vs `jax.lax.scan`) changing which
   intermediate matrices each encounters. **This is worth a separate `rqutils`
   issue**: it is a robustness property of the library's Rayleigh–Ritz step, not
   of this port.

## What this does not tell you

Beyond the four limitations listed above, all of which held: the comparison is
fp32-only where the GPU is concerned, and covers only the solver loop.

**The subspace dimension is the big one.** Every number here is at N=893 — a
~7 KiB vector, L1-resident, launching 893 threads on a 7-core GPU. Even the best
configuration is still substantially launch-latency-bound, so none of these
measurements reflect what either framework does when the hardware is saturated.
The honest reading of the final 1.63× is "MLX, aggressively optimized, is within
striking distance of JAX on a problem too small to distinguish their kernels" —
not a verdict on either framework's throughput.

The strategic case for MLX on Apple silicon, which these numbers do support: it
is currently the *only* way to use the M1 GPU for this workload at all.
`jax-metal` could not be measured here — it is not installed, and is effectively
unmaintained — so the `jax-metal-f32` arm was skipped in every run. There is no
JAX-on-GPU number to compare against.

## Out of scope

- Changing anything under `rqutils/`.
- Fixing `hproj` (finding 4) or the `ibit = iterm & 255` bug at
  `rqutils/sqd.py:544`. Both are pre-existing and worth separate issues; the
  `(1, 2)` cache path this PoC uses touches neither.
- Porting `svsim` to MLX.
- Making MLX a declared dependency. It is already in `pyproject.toml`
  uncommitted; whether that stays is a decision for after the numbers exist.
