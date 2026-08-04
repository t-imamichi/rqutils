# Fusing Rayleigh–Ritz inner products into a Metal kernel

Date: 2026-08-04
Branch: `metal`
Supersedes nothing; extends `docs/superpowers/specs/2026-08-03-mlx-sqd-poc-design.md`
("the POC doc") — specifically the second entry in its "Untried, in the order
worth trying" list.

## Goal

Reduce per-iteration MLX op launches in `rqutils.ground_locg_mlx`'s LOBPCG body by
fusing `_compute_sas` into a single `mx.fast.metal_kernel` launch, and make
large-N benchmarking possible so the change can be measured where the hardware
is not overhead-dominated.

## Starting point

`mx.fast.metal_kernel` is already used in this repo, once:
`apply_h_xz_mlx_metal` (`rqutils/ground_locg_mlx.py:181`), the fused
gather-multiply-accumulate matvec. It was the POC's single largest win — 5.8× on
MLX's CPU backend, 12.2× on GPU — and with `--compile-body` took the MLX arm from
35.4× slower than `jax-cpu-f32` to 1.63× slower.

After that kernel landed, the POC measured per-iteration op launches:

| component | ops/iter | share |
|---|---|---|
| `eigenpair_3x3` | 19 | 37.3% |
| `_compute_sas` | 16 | 31.4% |
| `_project_out` | 15 | 29.4% |
| matvec (`metal`) | 1 | 2.0% |

The matvec is now 2% of launches. The remaining 98% is the Rayleigh–Ritz step.

## Scope decisions, and why

### Target `_compute_sas`, not `eigenpair_3x3`

The POC names "fuse Rayleigh–Ritz into a Metal kernel" as the next candidate and
means the 3×3 eigensolve. This design targets `_compute_sas` instead.

`eigenpair_3x3` does arithmetic on nine numbers. Fusing it is a pure launch-count
win: it does not scale with N, so its payoff shrinks to nothing as the problem
grows, and it would mean reimplementing Cardano's method in Metal fp32 and
inheriting the near-degeneracy fragility the POC documented (3.6e-08 error vs
`eigh`'s 1.8e-15, and the reason `mlx-cpu-f64` needs 652 iterations to JAX's
217).

`_compute_sas` is O(N) work: the POC measured 16 launches per iteration for it,
producing 6 distinct inner products. Most of those are full-vector passes (the
n² elementwise multiplies and reductions), the remainder being the two `mx.stack`
levels and the closing symmetrization. Fusing replaces all 16 with one launch and
performs six passes over memory instead of nine, so unlike the eigensolve its
value does not evaporate at large N.

(The exact 16 is the POC's instrumented measurement, not a derivation — a naive
count of the `mx.*` calls in the current source gives 24, so MLX evidently does
not construct one op per call here. Re-instrument before quoting a
launches-removed number in any result; the claim this design rests on is
"16 → 1", which holds either way.)

### Leave `_project_out` alone

`_project_out` is 15 ops (29.4%) and also O(N), so it looks like an equally good
target. It is deliberately excluded.

Its two passes, intermediate normalizations, and terminal subtraction are
load-bearing against catastrophic cancellation — items I5/I6 of `docs/locg.md`,
each measured to fail *silently* (a plausible wrong number, not a raise or NaN).
`CLAUDE.md` states: don't simplify the re-orthogonalizations. A fused kernel that
reordered or collapsed a pass would be precisely the failure class this repo has
spent several commits eliminating, and it would run in fp32 on Metal, with less
headroom than the f64 path those guards were tuned in.

If the measurements below show `_compute_sas` paying off and `_project_out`
dominating what remains, that is the moment to revisit it — with evidence, not
from the op-count table.

## The kernel

### What it computes

`_compute_sas(vectors, mvs)` builds the matrix of ⟨vᵢ|A|vⱼ⟩. Called twice:

- `ground_locg_mlx.py:313` — 2 vectors (iteration 1's `{x, p}` basis)
- `ground_locg_mlx.py:350` — 3 vectors (the loop body's `{x, y, p}` basis)

Present implementation: n² `mx.sum(v * mv)` calls, an `mx.stack` per row, an
outer `mx.stack`, then `(sas + sas.T) * 0.5`. For n=3 that is 16 launches to
produce nine numbers, of which only 6 are distinct.

`_compute_sas_metal` computes the 6 distinct products (n=3) in one launch:

- One threadgroup per (i, j) pair with i ≤ j — 6 threadgroups for n=3, 3 for n=2.
- Each thread strides over N accumulating `v[i][k] * mv[j][k]` in a register.
- A threadgroup-memory tree reduction with
  `threadgroup_barrier(mem_flags::mem_threadgroup)` collapses the partials.
- Thread 0 writes **both** `out[i*n + j]` and `out[j*n + i]`.

### Two design points confirmed with the user

**Writing both triangles.** Because thread 0 writes the same value to both
off-diagonal slots, the output is exactly symmetric by construction and the
`(sas + sas.T) * 0.5` step disappears — not merely fused but eliminated. This is
*stronger* than the current code, which averages two values that differ by
rounding.

**Templating on basis size.** n ∈ {2, 3} only. The kernel templates on n rather
than having a separate two-vector kernel, keeping one code path and one place to
be correct. Inputs are passed as a stacked `(n, N)` array so the signature is
fixed regardless of n.

### Accumulation order

A tree reduction changes summation order relative to `mx.sum`. For dot products
of unit-norm vectors this is benign and typically *more* accurate than sequential
summation (error grows as log n rather than n). It is still a change, so it is
measured rather than asserted — see Validation.

### fp32 only

Metal has no float64. `_compute_sas_metal` raises `ValueError` on non-float32
input, mirroring `apply_h_xz_mlx_metal`'s existing guard verbatim. The f64 arms
keep the op-graph `_compute_sas`.

Consequence: **both implementations stay in the file and must stay in step.**
This is the same standing obligation `CLAUDE.md` already records for
`ground_locg.py` / `ground_locg_mlx.py` ("when you change one, change both"), now
applying within the MLX file too. Rather than leaving that as an exhortation, the
static checker asserts the two agree on the same inputs (Validation case 2) — so
a divergence fails a check instead of relying on a reader remembering. That
assertion is the enforcement mechanism, and it runs headless.

The selection happens in `ground_locg_mlx`, dispatched on dtype. The intent is
that reading `xinit.dtype` is metadata-only — no evaluation, hence no host sync
and no Python branch on a device *value* — so the dispatch composes with
`mx.compile`, unlike the seed step's `float(norm_r)` guard which branches on a
device value and does sync.

That `.dtype`-does-not-force-eval property could not be confirmed in this session
(no Metal device: even `mx.array([1.0])` raises), and the module's existing
comments note the port deliberately tests dtypes via `str(...)`/direct comparison
to stay shim-compatible. **Confirm on hardware that the dispatch does not
introduce a per-call sync** — if it does, hoist it out of the compiled body by
selecting the implementation once, before the loop, which is correct regardless
since the dtype is fixed for the whole solve.

## Scaling the correctness gate

`examples/_bench_common.dense_reference` allocates `(N, N)` float64 and calls
`np.linalg.eigvalsh`. At the POC's N=893 that is 6 MB; at N=30 000 it is 7.2 GB;
at N=1e5 it is 80 GB. **The gate cannot follow the benchmark to large N.**

The operator is sparse by construction — built by `np.add.at` over J X-groups, so
at most J·N nonzeros. Add a sparse path:

- Build the same operator as a `scipy.sparse` CSR matrix from the X-groups.
- Take the algebraically-smallest eigenvalue with
  `scipy.sparse.linalg.eigsh(..., which='SA')`.
- Threshold at **N = 5000** (a 200 MB dense matrix — comfortably allocatable, and
  above the POC's 893 and the current `--num-states 4000` default, so existing
  invocations keep taking the dense path and stay bit-for-bit reproducible).
  At or below it, run **both** dense and sparse and assert agreement, so the
  sparse path is validated against the dense one it replaces rather than trusted
  on arrival. Above it, sparse only.
- Agreement tolerance: `RTOL["f64"]` (1e-9), the same constant the existing
  brute-force cross-check uses, scaled by `max(1.0, abs(reference))` as that
  check does.
- `eigsh` needs explicit convergence handling: it is iterative, so it can fail to
  converge or return a non-finite value. Treat that as a gate failure with a
  clear message, not a silently-accepted reference — the POC found a bug where a
  NaN eigenvalue passed the gate because every NaN comparison is false.
- `brute_force_reference` (full 2ⁿ construction) stays as the third, fully
  independent check at small n.

This satisfies `CLAUDE.md`'s rule to prefer an independent reference: `eigsh` is a
different algorithm in a different library, not a second run of the solver under
test. scipy is already a hard dependency (`pyproject.toml:21`, scipy 1.18.0
present), so this adds no new requirement.

Explicitly rejected: using the converged `jax-cpu-f64` eigenvalue as the
reference. That is self-consistency between two runs of the same algorithm —
exactly what `CLAUDE.md` warns about, having found bugs where "every internal
code path agree[d] on the same wrong number."

## Validation

Per the user's decision, the two existing checkers are extended; no
`tests/test_ground_locg_mlx.py` is added. This preserves the arrangement
`CLAUDE.md` documents, where this module is checked by `examples/` scripts
because MLX cannot initialize without a Metal device.

### `examples/check_ground_locg_mlx_static.py` (no device needed)

`_shim_metal_kernel` currently hardcodes the matvec's signature — it unpacks
exactly `(vec, xsources, diagonals, num_groups, num_states)` and reimplements
that one formula. It must dispatch on the kernel `name`.

Added cases:

1. The sas kernel's shim implementation **simulates the real per-thread and
   per-threadgroup indexing, including the reduction tree**, not just the
   mathematical result — the same standard by which
   `apply_h_xz_mlx_metal`'s arithmetic was validated (max abs diff 2.7e-15).
2. Assert agreement with the op-graph `_compute_sas` and with a direct numpy
   `v @ mv`, for both n=2 and n=3.
3. Assert exact symmetry of the output (the both-triangles claim above).
4. Assert the float64 guard fires, mirroring the matvec's guard test.

### `examples/check_ground_locg_mlx_mlx.py` (real device)

Real-device cases for both precisions and both devices, matching the existing
structure.

### What this does and does not prove

The shim validates indexing arithmetic, caller wiring, shapes, dtypes, and grid
setup. **It cannot prove the Metal source compiles or that the barriers are
correct** — it is numpy, with no threadgroups and no memory model. Only a
real-device run establishes that. This is the same limitation the existing shim
already discloses in its own docstring, and it is why the real-device checker
exists.

## Measurement

`bench_mlx.py` gains `--sas {ops,metal}`, parallel to the existing `--matvec`:
default `ops` so existing measured results stay reproducible, and refusing
`metal` on JAX arms and on f64 arms exactly as `--matvec metal` does (a row must
not silently time a different kernel than it claims).

Then an N sweep, at fixed J, with `--matvec metal --compile-body` held constant
so the only variable is `--sas`.

### Expected shape of the result

This is a launch-count and memory-pass optimization. Per the POC's own cost model
— **syncs ≫ launches ≫ op construction** — it should help most where launch
overhead dominates and less as N grows and O(N) gather work takes over. The
31.4%-of-launches figure is from N=893, where the vector is ~7 KiB and only 893
threads launch on a 7-core GPU.

A plausible outcome is a solid win at N≈10³ shading toward noise at N≈10⁵. That
would itself be the useful finding, and it is why the sweep matters more than any
single number.

### Constraint on how far N can go

The POC records that the f32 arms fail the eigenvalue gate above ~1000 states at
n=14 (5.4e-3 relative error against rtol 1e-4). Metal is fp32-only. So fp32
convergence at large subspace dimension is a real prerequisite, and the sweep may
hit it before it hits a memory limit. Where an arm fails the gate, **no timing
row is reported for it** — the POC's standing rule that no measurement comes from
an arm that failed the gate.

### Results

Not yet measured. This session is sandboxed with no Metal device:
`mlx.core` 0.32.0 imports, but `metal::load_device` fails on even
`mx.array([1.0]) * 2` ("headless, sandboxed, or virtualized macOS session"), and
it is not a filesystem grant the sandbox can widen — `/dev` is already allowed.

The sweep must therefore be run by the user, or from a session started outside
the sandbox. **No performance number will be recorded here without a real
measurement behind it**; the table below is left with empty cells rather than
estimates.

(Correctness on hardware has since been confirmed by two device runs — see
"Correctness on real hardware" below. Only the *timing* below is still open.)

### Measured — the optimization does NOT pay off

`mlx-gpu-f32`, `--matvec metal --compile-body`, J≈100, n=14, `--repeat 15`.
Rows that tripped the benchmark's own steady-state spread guard (>25%) are
reported as unusable rather than quoted, per this repo's rule that a warned run
must not be compared:

| N requested | subspace_dim | `--sas ops` | `--sas metal` | ratio |
|---|---|---|---|---|
| 893 | ~800 | 0.593 ms | 0.697 ms | **1.18× SLOWER** |
| 2 000 | ~1 800 | 0.730 ms | *unusable (196% spread)* | — |
| 4 000 | 3 571 | 0.593 ms | *unusable (35% spread)* | — |
| 20 000 | 11 533 | gate failed | gate failed | — |

**One clean paired comparison exists, and the fused kernel loses it.** An earlier
sweep on a busy machine produced contradictory ratios (1.47× faster at N=893,
2.32× slower at N=4000) with 48–186% spreads; those numbers are discarded
entirely.

Two secondary observations pinning down what was actually measured:

* `ops` costs **0.593 ms at both N=893 and N=4000** — identical to three digits
  across a 4.5× problem-size increase. Per-iteration cost is not tracking N, so
  this regime is still dominated by fixed dispatch overhead rather than work.
* N=20 000 fails the gate on *both* arms at ~1.2e-4 relative error against
  `rtol=1e-4`, with the two arms agreeing to 7 digits. `jax-cpu-f64` converges to
  the correct value there (841 iterations, matvec_err 1.2e-14), so this is the
  **documented fp32 convergence ceiling**, not a kernel defect. Metal is
  fp32-only, so the large-N regime this design called "the decisive measurement"
  is unreachable by this kernel in principle.

### Why it loses: fixed occupancy

The launch-count model predicted the win and was wrong by 0.278 ms in the wrong
direction (predicted 0.419 ms, measured 0.697 ms). The mechanism is structural,
not a tuning miss:

`_compute_sas_metal` launches **one threadgroup per (i, j) pair — six, for
n=3 — regardless of N.** Serial work per thread grows linearly with N while
parallelism stays pinned:

| N | threadgroups | total threads | serial steps per thread |
|---|---|---|---|
| 800 | 6 | 1 536 | 3.1 |
| 3 571 | 6 | 1 536 | 13.9 |
| 11 533 | 6 | 1 536 | 45.1 |

The op-graph path instead calls MLX's own reduction kernels, which are tuned to
spread a reduction across the whole GPU. So the fusion trades 15 launches for a
drastically under-parallelized reduction, and at every measurable N the
reduction loses more than the launches save.

This is exactly why the *matvec* fusion succeeded where this one fails:
`apply_h_xz_mlx_metal` launches one thread **per output element** (46
threadgroups at N=11 533), because it has N outputs. The Rayleigh–Ritz step has
**nine**. There is no output parallelism to exploit, which is the same property
that made this quantity cheap in op-count terms and expensive in launch terms.

**Conclusion: keep `sas="ops"` as the default — it is the faster path.** The
`sas="metal"` kernel is correct, tested, and retained as a verified negative
result plus a starting point should the occupancy problem be addressed (e.g.
splitting each pair's reduction across many threadgroups with a second
combining pass, which reintroduces launches and is unlikely to win at these N).
The design's own prediction — "a plausible outcome is a solid win at N≈10³
shading toward noise at N≈10⁵" — was too optimistic: it is a loss at N≈10³ and
unmeasurable beyond N≈4×10³ because fp32 will not converge.

Command (to be run on hardware):

```bash
uv run --extra mlx python examples/bench_mlx.py --arm mlx-gpu-f32 \
    --matvec metal --compile-body --sas metal \
    --num-qubits 14 --num-paulis 100 --num-states <N>
```

## Correctness on real hardware — CONFIRMED

The kernel compiles and is numerically correct on a real Metal device. Two runs
of `examples/check_ground_locg_mlx_mlx.py` were needed.

**Run 1 failed to build the kernel**, and the cause is the most transferable
finding in this document: `half` is a **reserved built-in scalar type in Metal
Shading Language** (16-bit float), so the tree reduction's
`for (uint half = lanes / 2; ...)` failed with

```
error: cannot combine with previous 'type-name' declaration specifier
```

plus eight cascading errors from that single line. Renamed to `stride` in
`482b5aa`.

Nothing headless could have caught this, and it is worth being precise about
why, because three separate layers of verification passed it:

* The numpy shim reimplements the kernel's indexing *in Python*, where `half` is
  an ordinary identifier. It never compiles the Metal text, so it is blind by
  construction.
* A review specifically re-derived the kernel against MLX 0.32.0's own C++
  signature generator and cleared the threadgroup-size invariant, barrier
  uniformity under divergence, the attribute table, exact output coverage, and
  index overflow — without flagging a reserved-word collision. It was not in
  anyone's threat model.
* An *earlier* review had already caught one fatal compile error in this same
  kernel by reading (threadgroup memory used but never declared), which likely
  created false confidence that careful reading was sufficient.

A regression check now guards it: case `3l` in
`examples/check_ground_locg_mlx_static.py` scans both kernels' `source` strings
for identifiers colliding with MSL reserved type and qualifier names. It is the
only check in that file that reads `source` directly rather than the shim's
reimplementation, comments are stripped first (the prose legitimately discusses
`half` to explain this bug), and it was verified to fail when `half` is
reintroduced. The same run's `-Wsign-compare` warnings were also fixed:
`n_basis`/`n_states` arrive as `const constant int32_t` because MLX passes
Python ints as int32.

**Run 2 passed cleanly**, at n=10 / 20 paulis / 200 states (N=200), reference
ground energy `-2.496495741801`:

| arm | eigenvalue | iters | verdict |
|---|---|---|---|
| `mlx-cpu-f64` | -2.4964957418 | 89 | OK (matvec_err exactly 0.00e+00) |
| `mlx-cpu-f32` | -2.4964947701 | 26 | OK |
| `mlx-cpu-f32` `sas=metal` | -2.4964950085 | 26 | **OK** |
| `mlx-cpu-f32` `metal-both` | -2.4964947701 | 26 | **OK** |
| `mlx-gpu-f32` | -2.4964950085 | 26 | OK |
| `mlx-gpu-f32` `sas=metal` | -2.4964947701 | 26 | **OK** |
| `mlx-gpu-f32` `metal-both` | -2.4964947701 | 26 | **OK** |

`FAILURES: none`. Every f32 arm converges in **exactly 26 iterations**, matching
the POC's recorded f32 behaviour, and the fused-kernel eigenvalues land on the
same two values the op-graph path produces (`-2.4964947701` /
`-2.4964950085` — the two orderings of f32 rounding, not a systematic error).
`metal-both` — both custom kernels resident in one process, the configuration
the benchmark measures — matches the op-graph value bit-for-bit, so the fusions
compose without drift.

This discharges the correctness obligation. **The performance obligation is
still open:** the sweep below has not been run, so no speed claim exists.

## Implementation status

Implemented in twelve commits, `6049457..482b5aa`, gate-green (ruff / ruff
format / ty / 357 pytest) and reviewed. `sas="ops"` and `--sas ops` remain the
defaults, so the new path is unreachable unless explicitly requested and merging
carries no regression risk to existing measured results.

All six success criteria are now met, including criterion 6 — the sweep ran and
returned a negative result, which is a met criterion, not a failed one.

Re-run `uv run --extra mlx python examples/check_ground_locg_mlx_mlx.py` after
any edit to either kernel's source: it is the only check that compiles them.

Loose ends, none blocking:

* The sparse gate path **is** now exercised (`reference_path: "sparse"` at
  `subspace_dim=11533`, agreeing with the dense-equivalent reference to
  1.9e-15) — but only on `jax-cpu-f64`. No `mlx-*` arm has completed a run
  above the threshold, because fp32 fails the gate there. `_time_mlx`'s
  `matrix @ matvec_probe` against a CSR reference therefore remains untested,
  and is untestable until fp32 convergence at large N is addressed.
* A **pre-existing** latent bound in `apply_h_xz_mlx_metal` (not introduced by
  this work): its flat index `j * n_states + i` reaches `J*N`, which overflows
  32-bit Metal `uint` at N=1e7 once J > 429. Outside the documented J≈100
  regime. The sas kernel's own index reaches only `3N-1`, safe past N≈1.4e9.
* `--repeat 15` still left 2 of 6 rows above the spread guard on an otherwise
  idle machine, so this benchmark needs either more samples or a quieter host
  to resolve differences at the 0.1 ms scale.

## Success criteria

1. `check_ground_locg_mlx_static.py` passes, including the new sas cases, on a
   machine with no Metal device.
2. `check_ground_locg_mlx_mlx.py` passes on real hardware, both devices, both
   precisions.
3. The sparse gate agrees with the dense gate at small N, and with
   `brute_force_reference` at small n.
4. `ruff check`, `ruff format`, and `ty check` stay clean over
   `rqutils/ examples/` (`CLAUDE.md`: all three are clean, keep them that way).
5. `uv run --extra dev pytest` stays green — no existing test regresses.
6. The N sweep is measured and recorded, with gate status per row.

Criterion 6 cannot be met in a sandboxed session; see Results.

## Out of scope

- `eigenpair_3x3` in Metal (see Scope decisions).
- `_project_out` in Metal (see Scope decisions).
- Tuning `threadgroup` (256) and `compile_chunk` (10) — flag-only, and the POC
  already lists them separately.
- The `eigenpair_3x3` fp32 near-degeneracy robustness issue, which the POC
  correctly identifies as a pre-existing `rqutils` library characteristic
  deserving its own issue rather than a port change.
- Anything in `rqutils/ground_locg.py` (JAX). This design does not change the
  JAX path, so the "change both" obligation is not triggered for it.
