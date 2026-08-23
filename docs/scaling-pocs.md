# SQD scaling: six ideas, measured

Six ideas for improving `sqd`'s scaling, each built as a POC under `examples/scaling/` and measured.
Two are worth adopting, one is worth knowing about, two are dead, and one attempt found a real bug in
the library.

**Machine and its limits.** Apple M1, 16 GB, `jax.default_backend() == "cpu"`, one device, x64
enabled. Every number below is CPU unless a section says otherwise. Quoting a CPU ratio as a GPU ratio
is the error `CLAUDE.md` warns about, so it is not done here.

**GPU status (NVIDIA GH200 120GB, single device, 2026-08-06).** Two of the three GPU-specific claims
in `poc8_gpu_unverified.py` are now settled and one remains unrun:

- **searchsorted speedup: 5.15× at N = 64M**, rising with `N` (§1 below). Not the CPU 12–25×.
- **`lax.sort` memory leak: does not reproduce.** The in-tree note was stale (§1 below).
- **Multi-GPU speedup: UNRUN.** The box has one GH200. `--devices` sets `CUDA_VISIBLE_DEVICES`, a
  *filter* over devices the driver already exposes, so it cannot conjure a second GPU — this is unrun,
  not unresolved, and needs a physically multi-GPU machine. `poc7_sharding.py` already covers sharding
  *correctness* on virtual CPU devices, so only speed is missing.

**A POC's baseline must be pinned, not read from the library.** POC 1's proposal was adopted in
commit 23fb226, which made `get_xsource` *be* the searchsorted — and silently turned both POC 1's and
POC 8's timing arms into searchsorted-versus-searchsorted comparisons. The first GPU run of POC 8
duly reported 1.002×/1.000×/1.000× and a flat memory profile, and produced no information about
either claim; POC 1e read 0.26× "SLOWER" for the same reason. `fmt_ratio` was right every time, which
is precisely what made it unalarming. Both files now time against `poc1.xsource_sort_legacy`, a
verbatim copy of the pre-23fb226 sort, and their *correctness* arms still compare against the library
(that agreement is now a regression test rather than a proposal). Restoring the baseline recovered
12.1×/18.3× and, for the lex variant, 3.57×/3.18× — matching what 23fb226 recorded.

Three measurement defects in POC 8 were fixed at the same time, all of the same character as the
library bugs this suite is organized around: a plausible reading rather than an error. Its leak test
sampled `bytes_in_use` *after* `del`, and printed `peak_bytes_in_use`, a high-water mark that never
decreases — neither column could move regardless of the truth. Its `--devices` flag sets
`CUDA_VISIBLE_DEVICES`, a filter over devices the driver already exposes, so it cannot conjure a
second GPU: Claim 3 on a one-GPU box is **unrun**, not unresolved. And the sort arm now asserts it
grows with `N`; the stale run's 84/93/159 ms across a 25× `N` increase was the visible tell, and the
guard was verified to fire on exactly those numbers and stay quiet on healthy ones.

Timings are min-of-N-trials with the measured spread reported alongside; `_scaling_common.fmt_ratio`
refuses to call a difference a win when it falls inside that spread. Several rows below are therefore
labelled UNRESOLVED rather than given a number.

## Which ceiling actually binds

`baseline.py`. Three limits are documented — the `N ≤ 2**31` sort, cache memory, and `O(J·N)`
matvec — and the question is which one is in front. It is not the one the module docstring emphasises.

| component | runs | cost at N=200k, J=50 | scaling |
|---|---|---|---|
| `uniquify_states` | once | 48.6 ms | N^1.15 |
| `get_xsource` × J | once (cached) | **3064 ms** | N^0.88 |
| `apply_h` matvec | per iteration | 8.35 ms | N^1.03 |

Weighted by call count, `get_xsource` setup is **66–97 %** of a whole solve (97.5 % at 10 iterations,
66.4 % at 200). `matvec/J` is flat at ~0.16 ms across J ∈ {10, 25, 50, 100}, confirming the `O(J·N)`
model exactly. So the sort is not merely a ceiling on N — it is the dominant cost at every size
measured, which reorders the priorities.

## Results

| # | idea | verdict | measured |
|---|---|---|---|
| 1 | `searchsorted` replaces the 2N sort | **ADOPTED — now in `get_xsource`** | CPU 12–25× per signature; **GPU 5.15× at N=64M**, rising |
| 3 | `cache_level=(1,1)` vs `(1,2)` | **know about it** | 16× less memory, 2.4–2.6× slower matvec |
| 4 | real-symmetric f64 path | **already works** | 1.47–1.80× per solve; nothing to implement |
| 2 | partial-J caching dial | **marginal** | curve is linear, but endpoints dominate |
| 6 | mixed-precision matvec | **reject** | 1.23× per iteration, **0.44× end to end** |
| 5 | Gray-code X ordering | **dead** | silently drops 91 % of matrix elements |
| — | multi-device sharding | **bug found and fixed** | `sqd` failed on *any* mesh |

### 1. searchsorted instead of the sort — ADOPTED

**This is now what `get_xsource` does**; `poc1_searchsorted.py` remains as the exploratory record. The
integrated version re-measured **12.1×, 18.7×, 16.9×** on the J-fold precompute at N = 100k/200k/500k,
consistent with the POC.

#### Measured on GPU (NVIDIA GH200 120GB, 2026-08-06)

**5.15× at N = 64M, single signature, and still rising.** The CPU figure does not transfer, exactly as
predicted: a GPU sort is well optimized relative to its gather, so the ratio compresses while the
direction holds.

| N | legacy sort | searchsorted | speedup | noise |
|---|---|---|---|---|
| 1 000 000 | 1.13 ms | 0.44 ms | 2.56× | 9.4 % |
| 4 000 000 | 4.70 ms | 1.39 ms | 3.38× | 2.9 % |
| 16 000 000 | 21.5 ms | 4.93 ms | 4.36× | 1.2 % |
| 64 000 000 | 104 ms | 20.2 ms | **5.15×** | 0.3 % |

`alpha = 1.09` (sort) against `0.92` (searchsorted) from a log-log fit. That gap is why the ratio
climbs monotonically and why 5.15× is a **lower bound** rather than a plateau — the sort is mildly
superlinear, the gather mildly sublinear.

**Two GPU numbers from the same run are artifacts; do not quote them.** POC 1c (J = 50) reported
12.5–14×, suspiciously close to the CPU figure, with a sort arm *flat* at 1141/1239/1201 ms across a
5× `N` range — launch-latency bound, so it is a ratio of two overheads. POC 1b below N = 1M gives
2.56× for the same reason. On a GH200 the launch-bound regime extends past N = 1M at J = 1 **and**
covers N = 500k at J = 50, so it is per-call latency × call count, not `N` alone: reaching
compute-bound needs large `N` *per call*, which is what `--sweep-to` does. `check_scaling` now fits
`alpha` and refuses to call the ratios quotable below 0.6, because the launch-bound numbers are
reproducible and meaningless at the same time — the same failure mode as the stale-baseline 1.00×,
one layer down.

#### The `lax.sort` GPU memory leak does not reproduce — the note was stale

Measured on the same GH200 at n = 28, N = 5M, B = 4, J = 50, five repetitions per arm: the legacy
sort allocated ~0.95 GB of live transients and returned to the 0.019 GB baseline **every** repetition,
`retained_vs_baseline` and `drift_vs_rep0` both +0.000 GB. The searchsorted arm was identical. So on
this backend and JAX version, `lax.sort` does not leak, and the original note (up to 5 GB at shape
`(5M, 9)`) was either version-specific or has been fixed upstream. Reported rather than dropped, per
the script's own instruction.

Note this is now a claim about `lax.sort` itself: the sort left the library in 23fb226, so it is
answerable only against the pinned legacy arm, and a flat result there is a finding about JAX rather
than about `sqd`. The removal still stands on the other three grounds (speed, the `2N` allocation
behind the `N ≤ 2**31` ceiling, and shardability). Tests are in `tests/test_sqd.py::TestGetXsource`, verified to fail against
three injected defects: reversed byte significance (7 failures), the `uint64` path used beyond 8 bytes
(3 failures, exactly the `B > 8` cases), and a non-negative absent-source sentinel (13 failures).


`poc1_searchsorted.py`. `S` comes out of `uniquify_states` already lex-sorted, so finding
`A[i]` with `S[A[i]] == S[i] ^ x` is a binary search into `S`, not a reason to sort a `2N` stack.

| N | sort (library) | searchsorted | speedup |
|---|---|---|---|
| 10 000 | 2.49 ms | 0.23 ms | 10.8× |
| 200 000 | 80.7 ms | 3.18 ms | 25.4× |
| 1 000 000 | 421 ms | 31.8 ms | 13.2× |

J-fold precompute (the cost `baseline.py` found dominant): **12.0×, 16.9×, 14.9×** at
N = 100k/200k/500k with J = 50. Transient allocation drops 2.2–3.4×.

Three properties beyond speed: it removes the `2N` allocation that sets the `N ≤ 2**31` ceiling; a
gather **shards** where a sort does not; and it removes the `lax.sort` GPU leak — which, measured on a
GH200, turned out not to exist on that backend anyway (see the GPU subsection above). The removal
stands on the other two grounds plus speed.

Two costs to be honest about. The fast path packs a state row into a `uint64`, so it is exact only
while `n + 1 ≤ 64` bits — a hard correctness boundary, asserted rather than documented. An
arbitrary-width lexicographic variant removes the limit at 3.0–3.7× instead of 12–25× on CPU. **The
lex variant has no quotable GPU number:** POC 1e times a single signature at N = 200k, which is inside
the GH200's launch-bound regime (it read 1.60×/1.62×, and the u64 path reads 2.56× at N = 1M for the
same reason). Sizing it up the way `--sweep-to` does for POC 1b would fix that; nobody has.

**The correctness gate had to be rewritten, and the reason is the interesting part.** "Bit-identical"
rejected a correct implementation: on a fill-in row the library's `idx_sorted[1:] - size` lands on
assorted negatives (−12, −11, −10…) where a searchsorted returns −1. Both are consumed identically,
since `apply_xgrp` gathers with `mode="fill", wrap_negative_indices=False`. The gate is now: valid-row
indices bit-identical **and** the gathered result bit-identical — the latter being the only property a
consumer can observe.

### 3. `cache_level=(1,1)` — a real tradeoff, not a free lunch

`poc23_caching.py`. The predicted memory saving holds **exactly**: 15.99× (complex), 8.00× (real),
matching `16JN/κ̄JN` and `8JN/κ̄JN`. Results are bit-identical (`maxdiff = 0.0`).

But the popcount is **not** nearly free — the matvec costs **2.4–2.6× more**. My earlier
characterisation of it as nearly free was wrong. It is still a good trade when memory-bound (16× memory
for 2.6× time) and it needs no new code, but it is a tradeoff to choose deliberately.

### 4. real-symmetric — already correct, worth 1.5–1.8×

`poc4_real_symmetric.py`. Checked the premise before building: `apply_h` **already** propagates
float64 when `PauliSumXZ.c` narrows, and `eigvec` comes back float64 through a full `ground_locg`
solve. There is nothing to implement.

What it is worth, at fixed iteration count (`tol=0`, so convergence differences cannot confound):
**1.47× at N=50k and 1.80× at N=200k**, against noise floors of 3.6 % and 3.0 %. Diagonal cache is
exactly 2.00× smaller. The lever is Hamiltonian construction — even Y-count per string — not solver
configuration.

### 2. partial-J dial — real but marginal

The curve is genuinely linear in J′ (0 → 3757 ms, 12 → 2881, 25 → 1834, 38 → 932, 50 → 9.5), so it is
a usable continuous tradeoff rather than a step function. But the endpoint gap is enormous: full
caching is ~400× faster than none with the library sort, ~20× with searchsorted. **Only worth building
if the full cache genuinely does not fit** — otherwise always cache everything.

### 6. mixed precision — reject, and note the trap

`poc6_mixed_precision.py`. The per-iteration win is real (1.29–1.74× matvec-only, 1.23–1.24× at fixed
iterations). It does not survive convergence: three of four converged solves hit `maxiter=300` and
reported `converged=False`, because the f32 residual floor (~1e-7 relative) makes the f64 convergence
test unsatisfiable. End to end, **0.44× and 0.74× — slower**.

The more valuable result is POC 6d, which is the failure the `ground_locg` docstring predicts. Let the
matvec *return* f32 and `work_dtype` collapses, loosening `tol` by 5.4e8×. The solver then converges
in **9 iterations, reports `converged=True`, and is wrong by 4.4e-2 relative**. A silent 4 % error.

### 5. Gray-code ordering — dead on correctness

`poc5_graycode.py`. The composition `A2 = Ad ∘ A1` is only valid when the **intermediate** state
`S[i] ^ x1` is itself in the sampled subspace. In a sparse SQD subspace it usually is not: `S[i] ^ x2`
can be present while `S[i] ^ x1` is absent, and the chain cannot reach an element that genuinely
exists. Measured at n=20, N=100k, J=50 — reference finds 8720 sources, the chain finds **791**,
silently dropping **90.9 %** of the real matrix elements. The projected matrix stays symmetric, so the
eigenvalue would be plausible and wrong.

An early premise check appeared to confirm the composition; it was measuring a 400-row sample that
happened to be closed under the intermediate. **A composition identity that holds on a subset is not
an identity.**

Independently, the timing is UNRESOLVED against J independent lookups, since `Ad` still needs a full
lookup per step. Two reasons not to revisit it.

## The bug this found

`poc7_sharding.py` uses `XLA_FLAGS=--xla_force_host_platform_device_count=4` to get virtual CPU
devices, which exercises every sharding code path without a GPU. On first run, `sqd` **raised
`ShardingTypeError` immediately** — it could not run on a multi-device mesh at all.

Cause: the scatter at `sqd.py:474` in `vinit_from_min_diag` omitted `out_sharding`. A scatter into a
sharded operand cannot resolve its output sharding unambiguously, so JAX refuses to guess. Every
neighbouring array op in the module passes `out_sharding=`; this one did not. Fixed in one line.

This is exactly the gap `CLAUDE.md` predicted — "nothing exercises a multi-device mesh, so
`ground_locg`'s `out_sharding` contract, `sqd`'s mesh-size padding, and `svsim`'s `out_sharding` are
all unverified." After the fix:

- sharded vs single-device agree to **8.9e-16**, against an independent dense `eigvalsh` reference;
- mesh padding is transparent across all four residues of `N mod 4`;
- the `return_eigvec=True` reshard round-trip returns a genuine eigenvector (‖Hv−θv‖/‖v‖ ≈ 5e-12).

`pytest` remains 343 passed; ruff, ruff format, and ty are all clean.

## Recommended order

1. **Land POC 1** (`searchsorted`), keeping the `n+1 ≤ 64` assert and the lex fallback for wider
   problems. Largest single win, and it attacks the ceiling, the memory, and the shardability together.
2. **Keep the sharding fix** and consider promoting `poc7_sharding.py` into `tests/` under the
   virtual-device flag — it is the only coverage the multi-device contract has.
3. **Document `(1,1)`** as the memory-constrained option, with its real 2.6× cost.
4. **Note the real-symmetric 1.5–1.8×** as a Hamiltonian-construction concern.
5. Skip 2, 5, 6.

## What is still unverified

- The `lax.sort` GPU memory leak, and whether searchsorted removes it.
- POC 1's speedup **on GPU** — expect it to compress, since a GPU sort is better optimised relative to
  its gather than a CPU one.
- Whether sharding is *faster* on real devices, and whether per-device memory divides as intended.
- Anything at N near 2**31. The largest N measured is 10^6; the source-index cache alone is 400 GB at
  N = 2**31 with J = 50, so that regime is multi-node and outside anything reachable here.
