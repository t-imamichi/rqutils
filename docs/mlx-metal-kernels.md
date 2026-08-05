# Fused Metal kernels and op-count reduction for `rqutils/ground_locg_mlx.py`

Record date: 2026-08-05. Measurements taken on branch `metal` against an Apple M1 (7-core GPU)
unless stated otherwise. This document is the standing home for findings that outlived the code
that produced them: three kernels were written, **one was measured slower and deleted**, and two
option axes that existed only to select between a winner and a loser were collapsed into a single
`device="cpu"|"gpu"` parameter.

Read this before proposing a fusion, a host-side eigensolve, or an "obvious" simplification in
`ground_locg_mlx.py`. Several of the obvious ones were already tried and measured to lose.

## Short answer

**Whether fusing a computation into one Metal kernel wins is decided by output parallelism, not by
launch count.** That is the one transferable lesson here, and it is counterintuitive: the naive
model says fewer kernel launches is always better, and that model gets two of the three cases
wrong.

| fused kernel | outputs | verdict |
|---|---|---|
| matvec (`_apply_h_xz_metal`) | N, one thread each | **large win** — 21 launches → 1 |
| 3×3 eigensolve (`_eigenpair_3x3_metal`) | 4, no N-scaling work at all | **1.75× win** — ~34 launches → 1 |
| Rayleigh–Ritz inner products (`_compute_sas_metal`) | 9, but an O(N) reduction behind them | **slower — deleted** |

The reduction lost because nine outputs give it nothing to parallelize over: it launched six
threadgroups regardless of N, while MLX's own reduction kernels spread the same work across the
whole GPU. The eigensolve won *despite* having only four outputs because it has no N-scaling work
to under-parallelize — it is scalar arithmetic on nine numbers, and one thread does it in
registers.

## The cost model

Established in `superpowers/specs/2026-08-03-mlx-sqd-poc-design.md` and confirmed by every
measurement since. In descending order of cost:

1. **Device sync — ~389 µs, roughly 53× a kernel launch.** A sync *drains* the pipeline: every
   queued op must retire before the CPU can read a value, destroying async overlap. Anything that
   reads a value back per iteration inherits this.
2. **Kernel launch.** What fusing removes.
3. **Python-level op construction.** What the batched whole-array forms in `ground_locg_mlx.py`
   remove. Real but third-order.

The ranking is why `tol=0.0` (fixed-iteration, zero syncs) exists as a benchmarking mode at all,
and why `mx.compile` is now unconditional rather than optional.

## What shipped

### Unconditional `mx.compile`

The single largest lever: **2.71× on the MLX CPU backend** (4.393 → 2.991 ms/iter with the chunked
matvec). It traces the iteration body's op graph once instead of once per call. It used to be the
opt-in `compile_body=` parameter; it is now always on, because there is no configuration in which
not compiling is the right answer. The compiled trajectory was pinned *bit-for-bit* against the
uncompiled one before the uncompiled path was removed.

Because MLX has no `while_loop`/`cond`, a convergence check is necessarily a Python branch on a
synced value, which a compiled function cannot contain. So two loop shapes survive, keyed on
whether `tol == 0.0`:

- `tol=0.0` — compile the body, run `maxiter` iterations, **zero syncs**.
- otherwise — compile a chunk of `_COMPILE_CHUNK = 10` iterations, sync once per chunk.

### The fused matvec (`_apply_h_xz_metal`)

`out[i] = sum_j vec[xsources[j, i]] * diagonals[j, i]` in one launch, accumulator in a per-thread
register, no intermediates. One thread owns one output element, so no atomics are needed.

At J=100, N=893 this replaces 21 op launches (the chunked path at `chunk=16`) plus ~0.4 MB of
intermediate traffic with a single launch and none. Measured **2.13× faster than MLX's CPU
backend** (0.452 vs 0.961 ms/iter). Device-validated: `matvec_err` 1.69e-06, exactly the fp32 floor
and identical to the JAX f32 reference.

### The fused 3×3 eigensolve (`_eigenpair_3x3_metal`)

Balancing, Cardano, the rank-aware seven-candidate null-vector search, and the closing Rayleigh
quotient, all in one launch. Takes the LOBPCG body from **65.0 to 32.5 op constructions per
iteration** (−50%), the eigensolve itself from ~34.5 ops to exactly 1.

Controlled measurement (M1, n=12/p=100/s=1000, `--matvec metal`, only the eigensolve differing):

| | per iteration | end-to-end |
|---|---|---|
| op-graph | 0.465 ms | 0.0465 s |
| fused | 0.265 ms | 0.0275 s |
| | **1.75×** | **1.69×** |

The eigenvalue was **bit-identical** (−5.3960399628) and the iteration count unchanged at 70. The
two ratios agree because `iters` does not move. The 1.75× *exceeds* what the −50% launch reduction
alone predicts, because the fused kernel also removes ~34 intermediate allocations per iteration.
Unlike the op-graph reductions below, **this win survives compilation** — it is on top of the 2.71×,
not masked by it.

## Negative results — do not rediscover these

### `sas="metal"`: fusing the Rayleigh–Ritz reduction was *slower*

Measured **0.697 ms/iter versus 0.593** for the op-graph path at N≈800, against a launch-count
model that predicted 0.419. The kernel was correct and fully tested; it was deleted anyway, because
correctness was never the problem.

Why it lost, and why the identical argument *won* for the matvec: the kernel launches **one
threadgroup per (i, j) pair — six, for n=3 — regardless of N.** Serial work per thread therefore
grows linearly with N (3.1 steps at N=800, 45.1 at N=11533) while parallelism stays pinned at 1536
threads. The op-graph path calls MLX's own reduction kernels, which spread a reduction across the
whole GPU. Trading 15 launches for a drastically under-parallelized reduction lost at every
measurable N. The matvec avoids this because it has N outputs and one thread per output; the
Rayleigh–Ritz step has nine, so there is no output parallelism to exploit.

Full sweep in `superpowers/specs/2026-08-04-metal-sas-kernel-design.md`, including why N above
~4000 is unmeasurable here (fp32 fails the solver's convergence gate, and Metal offers no float64).

Two things that kernel genuinely did better, given up knowingly:

- **Exact symmetry by construction.** Thread 0 wrote both `out[i*n+j]` and `out[j*n+i]` with the
  identical value — stronger than the op-graph `(sas + sas.T) * 0.5`, which averages two values
  differing by rounding.
- **A tree reduction**, whose error grows as log n rather than n.

One incident from that kernel outlived it and is worth keeping in mind when writing MSL: **`half`
is a reserved built-in scalar type** (16-bit float), so `uint half = lanes / 2` fails to compile
with "cannot combine with previous 'type-name' declaration specifier" and cascades into eight
further errors. Neither the numpy shim nor any static check could catch this — it was found only by
a real-device run. `check_solver_headless.py` still scans every kernel source for MSL-reserved
identifiers as a result, and that scan is retained even though the kernel that motivated it is
gone.

### The unchunked group-at-a-time matvec: 8.106 ms/iter

The original port's matvec, one `take`+multiply+add per X-group, i.e. 3·J op constructions per
matvec. **Strictly dominated — it was the baseline every other configuration beat, by up to
21.7×**, and it is deleted. `apply_h_xz` gathers `chunk` groups per flat `take`, cutting the count
to roughly 3·⌈J/chunk⌉.

But **`chunk=J` (full batching) is not the answer either**, which is why `apply_h_xz` keeps a
`chunk` parameter rather than batching unconditionally. `chunk` bounds the gathered temporary to
`chunk * N` elements. At the large N this design ultimately targets (SQD is matrix-free precisely
so it can reach N ~ 10⁷), a full `(J, N)` gather would cost **~8 GB versus ~80 MB** for the
group-at-a-time loop. The op-count/memory tradeoff at J=100:

| chunk | ops per matvec | temporary |
|---|---|---|
| 1 | 300 | N |
| 8 | 39 (7.7× fewer) | 8N |
| **16** | **21 (14.3× fewer)** | **16N** ← default |
| 32 | 12 (25× fewer) | 32N |

### Host-side `eigh` per iteration: 1.37× slower

Moving the 3×3 Rayleigh–Ritz eigensolve to the host with `np.linalg.eigh` replaced 35 op launches
with 2 device syncs and measured **0.910 versus 0.664 ms/iter**. Reverted in `4dc8510`. This is the
cleanest demonstration of the cost model's top item: two syncs cost more than 35 launches.

The same argument applies to `mx.linalg.eigh`, which *does* exist and takes exactly the input this
solver produces (real symmetric, lower triangle, ascending eigenvalues, so `w[0]`/`v[:, 0]` would
*be* the ground pair). Earlier versions of the module docstring said "MLX has no `eigh`"; that was
wrong. The analytic Cardano route is kept on **performance** grounds, not for lack of an
alternative. MLX's own docstring example for `eigh` passes `stream=mx.cpu`, which is suggestive of a
CPU-only implementation but not proof — unverified on device. Even if it runs on Metal with no sync,
the comparison is one general `eigh` launch against a single fused launch already at the launch
floor and hand-specialized to nine numbers: break-even is the *optimistic* case, against a measured
1.75× that would be given up.

**The one place `eigh` might still win, unexplored.** What it would buy is accuracy: it is ~2×10⁷
times more accurate than Cardano on the near-degenerate matrices a converging LOBPCG produces
(1.8e-15 versus 3.6e-08). That is the documented cause of `mlx-cpu-f64` needing **217 iterations
where JAX's f64 needs 89**. An `eigh`-backed path confined to the f64 CPU arm — which can use no
Metal kernel and crosses no device boundary — is therefore a plausibly *large* win via fewer
iterations rather than faster ones. It is not implemented, and would need measuring on real
hardware before being believed.

## Op-count reduction: fewer ops is not automatically less time

In JAX an unrolled Python loop over scalars is free — XLA fuses it into the surrounding `jit`. In
MLX every call is its own lazily-evaluated op, so a faithful transcription of
`jnp.stack([normalize(c) for c in cands])` cost **18% of the port's entire per-iteration op
count**. The module therefore uses batched whole-array forms where the JAX original unrolls, and
hoists dtype-dependent constants into `_CONST_CACHE`. Every one of those was verified
*bit-identical* to the form it replaced, not merely close.

The history, and what each step was actually worth:

| body ops/iter | change | measured |
|---|---|---|
| 116 | starting point | — |
| 76.3 | batched candidates, `_CONST_CACHE`, `diagonal`/`roll` | ~1.05–1.10× on `mlx-cpu-f64` (2.928 → 2.689 ms/iter) |
| 65.0 | `_compute_sas` from one stacked matmul | **backend-split — see below** |
| 32.5 | fused eigensolve (`device="gpu"`) | 1.75×, and it survives compilation |

**The `_compute_sas` step is the instructive one.** It is flat on the MLX CPU backend (2.742 →
2.725 ms/iter, 0.6%, inside that arm's noise floor) but worth **~1.17× on the GPU** (0.265 →
0.224–0.227 ms/iter, measured across four independent processes agreeing to 1.3%). On the GPU each
`mx.sum` is its own kernel launch, so collapsing 9 reductions plus 4.5 stacks into one matmul
removes ~12 launches; the CPU backend has no launches to remove, and the reductions were never the
bottleneck at N~1000. **Measure the backend you care about — a flat CPU result does not mean a
change is worthless.**

That matmul is also *more accurate* than what it replaced, independently of speed and on every
backend: against a longdouble reference over 3000 random unit-norm triples at N=900, the matmul is
exact (0.0 max error) while the n²-sums version carries up to 7.1e-15, because BLAS accumulates
pairwise (error ~ log N) where `mx.sum` over a product array accumulates sequentially.

**`_project_out` is the deliberate exception** and must not receive the same treatment. Its eight
reductions could become one stacked `B @ v` per pass, but a matmul reassociates the summation
order, and this function's whole purpose is resisting catastrophic cancellation (`locg.md` items
I5/I6, both measured to fail *silently*). Tested rather than assumed: over 4000 adversarial cases,
both forms hold residual orthogonality at machine epsilon, but the matmul is consistently *worse* —
worst `|<b|p>|` of **8.3e-17 versus 6.2e-17**. Neither is broken, so this is a judgement call, not
a measured failure: ~4 ops/iteration is not worth a 33% erosion of the quantity these guards
protect.

One consequence of the accumulation-order change, expected and benign: **f32 eigenvalues shift in
their last one or two digits** relative to runs recorded before it (e.g. −5.3960409164 versus
−5.3960399628, ~1.8e-7 relative against an f32 eps of 1.19e-7). Iteration counts are unchanged and
the correctness gate passes with a >500× margin at `rtol=1e-4`. f64 is unaffected at printed
precision. **Do not treat an f32 last-digit difference against a pre-matmul recorded value as a
regression.**

## Verification, and what it cannot reach

`examples/mlx/check_solver_headless.py` re-executes the module's source against a numpy shim bound
to the name `mx`, so it validates the algorithm and the caller-facing contract with no MLX and no
GPU. **It never compiles the MSL text.** The shim reimplements each kernel's *intended* per-thread
indexing in numpy, so it catches a divergence between the caller's contract (shapes, dtypes,
grid/threadgroup setup, flat row-major indexing) and that intent — but it is structurally blind to a
bug inside the Metal source string, which it does not read.

Two static guards cover part of that gap, both verified to fire by breaking them deliberately:

- every kernel source must qualify its math calls with `metal::` (MLX emits no
  `using namespace metal;`, so an unqualified `sqrt` fails to compile);
- no kernel may declare an MSL-reserved identifier (the `half` incident above).

`examples/mlx/check_solver_device.py` is the real-device counterpart and the only thing that can
establish that the MSL compiles and is correct on hardware. As of 2026-08-05 all arms pass on an
M1 with `FAILURES: none`, and the fused eigensolve agrees with `numpy.linalg.eigh` to 2.98e-07
(GPU) and 4.00e-07 (CPU) over 40 random symmetric matrices.

**Re-run after the option collapse (2026-08-05, same M1), all 12 arms `FAILURES: none`.** This is
the run that validates the `device=`-selected *combinations*, not just each kernel in isolation.
Three things it pinned:

- `mlx-gpu-f32 --matvec metal` measured **0.224 ms/iter at 70 iterations** with
  `matvec_err` 1.69e-06 — the bottom of the 0.224–0.227 range recorded for this configuration
  above, which is what confirms `device="gpu"` actually reaches both fused kernels rather than
  silently falling back to the op-graph path. A fallback would have read ~0.46.
- The eigenvalue was **−5.3960409164**, i.e. the *post*-`_compute_sas` value, not the
  −5.3960399628 in the eigensolve table. Both are correct; see the f32 last-digit paragraph above
  before treating the difference as a regression.
- `mlx-cpu-f64` reported `matvec_err` **exactly 0.0**, better than the 3.55e-15 recorded for the
  f64 arms previously. Structural rather than lucky: at this problem's J≤20 with `chunk=16` the
  chunked gather is one flat `take` plus a single `mx.sum`, so there is no cross-group `out = out +
  ...` accumulation left to round. The deleted group-at-a-time matvec accumulated J times, which is
  where the 3.55e-15 came from.

One combination is now exercised **only** by this checker: `mlx-cpu-f32` with `device="gpu"` (fused
eigensolve, op-graph matvec on the CPU backend). `bench.py` derives `device` from the arm name, so
it cannot produce that pairing. It passes at `rtol=1e-4`, and its eigenvalue legitimately differs
from the same arm's `metal-both` in the last two f32 digits — different arithmetic runs in different
places — so do not expect bit-equality between those two rows.

## Scope and limitations

- **One machine.** Every timing here is an Apple M1 with a 7-core GPU. Ratios on other Apple
  silicon are unmeasured, and the GPU/CPU-backend crossover in particular should be expected to
  move.
- **fp32 only above N≈4000.** Metal has no float64, and fp32 fails the solver's convergence gate
  past roughly that size, so the large-N end of every GPU claim is extrapolation.
- **The f64 arm can use no Metal kernel at all.** This is why the portable op-graph reductions
  matter independently of the fused kernels, and why `device="cpu"` remains the only route to an
  f64 solve.
- **Noise floor 3.9%** on the `mlx-cpu-f64` arm, established by two runs of identical code. Treat
  any difference under ~4% on that arm as unresolved.
- When attributing an eigenvalue change to a kernel, **pin every problem parameter first.** An
  uncontrolled comparison here once looked like an accuracy regression (energy below reference, 120
  iterations instead of 70) and was purely a different `--num-qubits`. The correctness gate cannot
  catch this: it builds its reference from the same run's inputs, so it passes either way.
- Quote a speedup only from `examples/mlx/bench.py` on real hardware. `examples/mlx/count_ops.py`
  counts **op constructions, not time** — third in the cost model above. It is a tool for finding
  *where* the launches are, not for claiming a win.
