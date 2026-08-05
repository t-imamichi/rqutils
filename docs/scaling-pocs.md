# SQD scaling: six ideas, measured

Six ideas for improving `sqd`'s scaling, each built as a POC under `examples/scaling/` and measured.
Two are worth adopting, one is worth knowing about, two are dead, and one attempt found a real bug in
the library.

**Machine and its limits.** Apple M1, 16 GB, `jax.default_backend() == "cpu"`, one device, x64
enabled. Every number below is CPU. Two claims in `rqutils/sqd.py` are GPU-specific and were **not**
verified: the `lax.sort` memory leak (`sqd.py:561`) and multi-GPU speedup. `examples/scaling/poc8_gpu_unverified.py`
exists to settle both in one run on a CUDA box. Quoting a CPU ratio as a GPU ratio is the error
`CLAUDE.md` warns about for the MLX arms, so it is not done here.

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
| 1 | `searchsorted` replaces the 2N sort | **ADOPTED — now in `get_xsource`** | 12–25× per signature; 12–19× on the J-fold precompute |
| 3 | `cache_level=(1,1)` vs `(1,2)` | **know about it** | 16× less memory, 2.4–2.6× slower matvec |
| 4 | real-symmetric f64 path | **already works** | 1.47–1.80× per solve; nothing to implement |
| 2 | partial-J caching dial | **marginal** | curve is linear, but endpoints dominate |
| 6 | mixed-precision matvec | **reject** | 1.23× per iteration, **0.44× end to end** |
| 5 | Gray-code X ordering | **dead** | silently drops 91 % of matrix elements |
| — | multi-device sharding | **bug found and fixed** | `sqd` failed on *any* mesh |

### 1. searchsorted instead of the sort — ADOPTED

**This is now what `get_xsource` does**; `poc1_searchsorted.py` remains as the exploratory record. The
integrated version re-measured **12.1×, 18.7×, 16.9×** on the J-fold precompute at N = 100k/200k/500k,
consistent with the POC. Tests are in `tests/test_sqd.py::TestGetXsource`, verified to fail against
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
gather **shards** where a sort does not; and it should remove the `sqd.py:561` GPU leak, though that
last one is unverified here.

Two costs to be honest about. The fast path packs a state row into a `uint64`, so it is exact only
while `n + 1 ≤ 64` bits — a hard correctness boundary, asserted rather than documented. An
arbitrary-width lexicographic variant removes the limit at 3.0–3.7× instead of 12–25×.

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
