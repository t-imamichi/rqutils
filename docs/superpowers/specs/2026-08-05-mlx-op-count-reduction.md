# Reducing per-iteration op launches in the MLX LOBPCG body

Date: 2026-08-05
Extends `docs/superpowers/specs/2026-08-03-mlx-sqd-poc-design.md` (the POC doc) and
`docs/superpowers/specs/2026-08-04-metal-sas-kernel-design.md` (the sas-kernel doc).

## Goal

Cut the op-construction count of `rqutils.ground_locg_mlx`'s LOBPCG iteration
body, targeting the Rayleigh–Ritz step that the POC's cost model identified as
dominating launches once the matvec was fused.

## Update (same day): the third Metal kernel was written after all

The section below deferred the fused eigensolve for lack of hardware. The user
then ran `examples/bench_mlx.py` on an M1, which **executed both pre-existing
Metal kernels on a device for the first time** — they passed every gate
(`matvec_err` 1.69e-06, the documented fp32 floor; eigenvalue matching the
recorded reference), and GPU measured 2.13× MLX-CPU (0.452 vs 0.961 ms/iter).
That removed the "unverifiable" half of the objection, so `eigenpair_3x3_metal`
(`eig="metal"`) was implemented.

It fuses balancing, Cardano, the rank-aware seven-candidate null-vector search
and the closing Rayleigh polish into one launch: **77.3 → 44.3 ops/iter (−43%)**,
with the eigensolve itself going from 34.5 ops to exactly 1. Cumulative with the
op-graph work below: **116 → 44.3, −62%.**

Why this wins where `_compute_sas_metal` lost — and the `_compute_sas` negative
result is what makes the argument precise rather than hopeful: the discriminator
is **output parallelism, not launch count.** `_compute_sas` fused an O(N)
reduction into six threadgroups and under-parallelized what MLX's own kernels
spread across the GPU. The eigensolve has *no* N-scaling work to under-parallelize
— it is scalar arithmetic on nine numbers — so a single thread in registers
replaces ~34 launches whose entire content is that arithmetic.

Two static guards were added, because this is the first kernel here to call any
math function and the shim never compiles MSL:

- every kernel source must qualify math calls with `metal::` (MLX emits no
  `using namespace metal;`);
- the pre-existing MSL-reserved-identifier check now covers all three sources.

Both were verified to fire by breaking them deliberately, and the eigenpair
comparison was verified to catch an injected `c0` permutation error (2.32e-04).

Its arithmetic is validated against the op-graph path and independently against
`numpy.linalg.eigh` (worst 7.03e-06 relative under the shim, consistent with fp32
Cardano) over 51 matrices spanning generic, large-trace, exactly-degenerate,
near-degenerate, identity and zero cases.

### Device validation and measurement (M1, 2026-08-05)

`examples/check_ground_locg_mlx_mlx.py` now passes on hardware — all 12 arms,
`FAILURES: none`. The kernel's `metal::sqrt`/`cos`/`sin`/`atan2` calls compile, and
on-device it agrees with `numpy.linalg.eigh` to 2.98e-07 (GPU) / 4.00e-07 (CPU)
over 40 random symmetric matrices.

**Controlled benchmark, n=12/p=100/s=1000, `--matvec metal --compile-body`, only
`--eig` differing:**

| | `eig=ops` | `eig=metal` | |
|---|---|---|---|
| `per_it_ms` | 0.465 | 0.265 | **1.75×** |
| `solve_s` | 0.0465 | 0.0275 | **1.69×** |
| `iters` | 70 | 70 | unchanged |
| `eigval` | -5.3960399628 | -5.3960399628 | bit-identical |

Three points worth keeping:

- **End-to-end tracks per-iteration** because `iters` does not move, so the win is
  real production time, not just a cheaper iteration.
- **1.75× exceeds the −43% launch reduction's prediction** (~1.4–1.5×). The extra is
  attributable to the fused kernel also eliminating ~34 intermediate allocations per
  iteration. Recorded as measured, not derived.
- **The win is on top of `--compile-body`**, unlike the op-graph reductions above,
  which `compile_body` largely masks.

A methodological note: an *uncontrolled* first run showed `eigval -5.4262146950`
in 120 iterations and looked like an accuracy regression — the f32 Cardano
fragility plus a below-reference energy is exactly the `docs/locg.md` I5 signature.
It was a different problem size (`--num-qubits 14` defaults vs the pinned 12), not a
regression. The correctness gate cannot catch this class of confusion, because it
compares against a reference derived from the same run's inputs and so passes either
way. **Pin every problem parameter before attributing an eigenvalue change to a
kernel.**

### One finding worth recording

An early test case, `diag([-1.5, -1.0, 1e9])`, failed — and the cause was the
test, not the kernel. **The op-graph `eigenpair_3x3` returns `0.0` for that input
too.** After balancing by `scale=1e9`, `-1.5` and `-1.0` become bit-identical in
float32 (measured difference exactly `0.0`, against eps 1.19e-07), so the true
minimum is unrecoverable in fp32 by *any* algorithm using this balancing. The case
was rewritten to use the exclusion shift `iter_body` actually generates
(`max(diag) + sum(|diag|) + 1`, bounded by the matrix scale), and the residual
tolerance was corrected to normalize by ‖A‖ rather than |θ| — a residual is an
absolute quantity whose fp32 floor scales with the matrix norm. Recorded because
it is a real documented limit of the fp32 path, and because it would be easy to
misread as a kernel defect.

## Why this target, and why not a third Metal kernel

The POC's measured cost model is **sync count dominates, launch count is
secondary, op-construction count is tertiary**. After the fused Metal matvec
landed, the matvec was 2% of per-iteration launches and the Rayleigh–Ritz step
was the other 98% — doing scalar arithmetic on a 3×3 matrix, i.e. nine numbers.

The POC's top-priority untried item was fusing the 3×3 eigensolve into a Metal
kernel. This work deliberately did **not** do that, for two reasons:

1. **It cannot be verified here.** `rqutils/ground_locg_mlx.py` states that no
   Metal kernel in it has ever been executed: the numpy shim reimplements each
   kernel's *intended* indexing rather than compiling the MSL text, and the
   real-device checker has never been run. A third unverifiable kernel — this one
   containing Cardano's method, whose fp32 near-degeneracy fragility the POC
   measured at 3.6e-08 vs `eigh`'s 1.8e-15 — adds risk that cannot be discharged
   without hardware. (The MSL `half` incident recorded in `_METAL_SAS_SOURCE`
   shows the shim's blind spot is not hypothetical.)
2. **The op-graph path is where the portable win is.** A Metal kernel is fp32-only
   (Metal has no float64), so it cannot help the f64 CPU arm at all. The
   reductions below help every arm and every precision.

The measurement environment also forced this: MLX cannot initialize without a
Metal device, and the session this was done in had no GPU access, so no timing
number could be produced. Op-construction count is what *was* measurable, and the
POC's cost model is what licenses reading it as a proxy — a weaker claim than a
wall-clock measurement, and flagged as such below.

## The finding

**In JAX an unrolled Python loop over scalars is free; in MLX it is not.** XLA
fuses `jnp.stack([normalize(c) for c in cands])` into the surrounding `jit`, so
`rqutils.ground_locg` pays nothing for building its seven null-vector candidates
one at a time. MLX constructs one lazily-evaluated op per call, so the *identical
source shape*, faithfully transcribed, became 18% of this port's entire
per-iteration op count.

This is a general hazard for the "when you change one, change both" rule that
governs these two files: transcribing JAX source into MLX preserves the algebra
and silently discards the fusion assumption the JAX source was written under.

The same asymmetry applies to constants. Every `mx.array(...)` is a fresh
allocation and another graph node, so the dtype-dependent scalars the eigenpair
kernels use were being rebuilt every iteration — `eigenpair_3x3` alone
constructed 7 per call, making `array` its single largest op contributor.

## Changes

All in `rqutils/ground_locg_mlx.py`. Each was verified **bit-identical** to the
form it replaced, not merely close — they change how many ops are launched, never
the arithmetic.

| change | ops/iter |
|---|---|
| `_nullvec_3x3`: stack the 7 candidates, normalize the `(7, 3)` block in one pass | 39 → 20 |
| `_nullvec_3x3`: one broadcast cross for the 3 complements (`col × e_k`), one batched cross for the 3 rank-2 pairs | 20 → 16 |
| `_nullvec_3x3`: reuse the pre-normalization norms for the `alive` mask | 16 → 13.2 |
| `eigenpair_3x3`: `mx.diagonal`/`mx.roll`/`mx.prod` instead of element-wise `mx.stack` | (see below) |
| `eigenpair_3x3`/`eigenpair_2x2`/`iter_body`/`normalize`/`_project_out`: per-dtype constant cache | 29 → 21.3 |
| `iter_body`: cached `p_mask` instead of `zeros_like` + in-place write; `mx.diagonal(...)[:2]` | 12 → 10 |

Net: **116 → 77.3 op constructions per iteration, −33%.**

Note the `eigenpair_3x3` row: the `diagonal`/`roll`/`prod` reformulation *by
itself* moved 26 → 29, i.e. **the wrong way**, because the two coefficient vectors
it introduced were being reallocated per call. The constant cache is what turned
it into a win. This is worth recording as a method point: the per-op breakdown
(`examples/count_mlx_ops.py --by-op`) is what exposed `array:7` as the real cost;
the function-level total alone would have suggested reverting a change that was
actually correct but incompletely applied.

Incidentally, the reformulation also brings the MLX source *closer* to the JAX
original, which already used `diagonal`/`roll`/`prod`.

`_normalize_or_zero` became dead and was removed.

## Verification

- `examples/check_ground_locg_mlx_static.py`: all checks pass, and critically the
  solve is unchanged — eigenvalue `-2.496495741801` to all 12 printed digits and
  **89 iterations**, identical before and after, on the default path, the
  `compile_body=True` path (bit-for-bit), and the `sas='metal'` path.
- Equivalence of each reformulation checked independently in numpy against the
  form it replaced: 0.0 max difference over 2000–5000 random matrices spanning
  scales 1e-12..1e12, including rank-deficient and exact-zero cases (the ones the
  guards exist for).
- `pytest`: 343 passed. `ruff check`, `ruff format --check`, `ty check`: clean.

## What this does not tell you

**No timing measurement was taken, on either backend.** MLX requires a Metal
device to initialize and none was reachable, so every number here is an
op-construction count, not a wall-clock figure. The POC's cost model ranks
op-construction count *third*, behind sync count and launch count — and this
change touches neither of the first two: it removes no syncs and the iteration
count is unchanged. So the honest prediction is a modest improvement on the
uncompiled MLX path, largest where per-op overhead dominates (small N), and
smaller under `--compile-body`, which already amortizes graph construction.
**Someone with a Metal device should run `examples/bench_mlx.py` before any
speedup factor is quoted.**

`examples/count_mlx_ops.py` is new and exists to make this class of regression
visible: it re-executes the module's own source against a counting numpy shim, so
its counts cannot drift from the real file.
