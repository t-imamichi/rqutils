# Scaling `sqd` past `N = 2^31`: what binds, what was measured, what is impossible

A session's investigation into lifting the `N <= 2^31 - 1` subspace ceiling, prompted by wanting more
unique bitstring samples than an int32 index can address. Four candidate approaches were examined and
three were measured. One is adopted-worthy, one was rejected on its own numbers, one is a partial
result whose limits matter more than its speedup, and the final target turned out to be unreachable by
a margin no engineering closes.

**Read `scaling-pocs.md` first** for the six earlier POCs. This document continues that record and
follows its conventions: timings are min-of-N-trials with the spread reported, and a difference inside
the spread is UNRESOLVED rather than a number.

**Machine.** Apple M1, 16 GB, `jax.default_backend() == "cpu"`, one device, x64 enabled. All numbers
below are CPU and single-process. No multi-node hardware was available, which bounds what could be
verified and is flagged wherever it matters.

## Summary

| # | question | verdict |
|---|---|---|
| 1 | Remove the int32 constraint cheaply? | **Yes, but it is not the binding limit** |
| 2 | A better lookup algorithm than binary search? | **No — hashing measured 4.3x SLOWER** |
| 3 | Out-of-core uniquify to lift the ceiling? | **Prototyped, 4.95x faster / 5.9x less peak — but `B <= 8` only, single-node** |
| 4 | Does it work multi-node? | **No, and the prototype's design cannot** |
| 5 | 100-site XXZ, N > 2^31? | **`xsource` is 91% of memory; cut J before raising N** |
| 6 | N = 10^30? | **Impossible: 65 million times world storage, and 10x the Hilbert sector** |

## 1. The int32 constraint is real but is not what stops you

`_MAX_STATES = 2**31 - 1` (`sqd.py:200`) guards exactly two things: `uniquify_states`' int32 iota
(`sqd.py:913`) and `get_xsource`'s returned positions (`sqd.py:1132`, `-1` as the absent marker).

Neither limits **sample generation** or qubit count. States are packed `uint8` rows of
`ceil((n+1)/8)` bytes, so `n` is unbounded by this. The cap is purely on `N`, the number of *distinct*
basis states in the projected subspace. Drawing more than `2^31` samples is fine; the ceiling binds
only on unique states surviving uniquification.

Widening the dtype is mechanical (`np.int32 -> np.int64` at the iota, the `bincount`/`cumsum`
accumulators, and `get_xsource`'s `lo`/`hi`/`invalid`), and should be a **static `index_dtype`
parameter defaulting to int32**, not a global switch. Both functions are already `@jax.jit` with static
`states_size`, so the choice costs nothing per call and the int32 path stays byte-identical. Two costs
argue against flipping it globally:

- **Memory.** `xsource` doubles from 4 to 8 bytes per state *per X group*. It is already the dominant
  array (§5), so this is the expensive change, not a formality.
- **Bandwidth.** `apply_xgrp` (`sqd.py:1274`) is a pure gather-and-multiply, memory-bound. Doubling
  index width doubles traffic in the innermost kernel — a regression for every user below `2^31`,
  which is all of them.

## 2. No better lookup algorithm — hashing is 4.3x slower

The obvious candidate: replace `get_xsource`'s `O(log N)` binary search with `O(1)` hashing. Built as
an open-addressed linear-probe table, correctness-gated against `searchsorted`.

The per-lookup microbenchmark is encouraging and wrong: **10.3 ns single-probe gather against 97 ns for
`searchsorted`**, apparently 9x. End to end it inverts.

| N | `searchsorted` | hash (build + lookup) | verdict |
|---|---|---|---|
| 4 M | 103 ms | 206 + 179 = **385 ms** | 3.7x slower |
| 16 M | 447 ms | 1035 + 892 = **1927 ms** | 4.3x slower |

Three causes: collision resolution costs **47 probe rounds** at load factor 0.5 (vectorized insertion
cannot do per-element probe loops, so every round reprocesses stragglers); memory is **4.5x worse**
(544 MiB against 122 MiB at 16 M), actively harmful when memory is the real constraint; and the build
is paid `J` times or held resident.

**Why no algorithm wins much.** `scaling-pocs.md:92` fits `searchsorted` at **alpha = 0.92 —
sublinear**. Direct measurement confirms the mechanism: ~97 ns/lookup, **flat** from 4 M to 64 M rather
than growing like `log N`. The lookup is **cache-miss-bound, not comparison-bound** — essentially one
unavoidable random memory access. Hashing's asymptotic advantage is unrealizable when both cost one
cache miss, and it adds a build phase and 4.5x memory to buy it.

Two variants also measured, both no better: batching all `J*N` targets into one call (6738 ms against
6484 ms) and sorting targets first for probe locality (7957 ms — the sort costs more than the locality
saves).

**The microbenchmark trap is the transferable lesson**, and it is the same shape as the defects this
repo is organized around: a warm-table single probe is a real measurement of the wrong quantity. The
gap between "9x faster" and "0.23x" was entirely build cost and collisions.

## 3. Out-of-core uniquify: `examples/scaling/poc9_ooc_uniquify.py`

`uniquify_states`' `jax.lax.sort` must run on one device, and since POC 1 replaced `get_xsource`'s
`2N` stack-and-sort with a binary search, **that sort is the sole remaining cause of the ceiling**
(`sqd.py:157-163`).

The observation the POC rests on: **nothing downstream requires the sort to be in JAX.**
`get_xsource`, the diagonal builders and the matvec need `states_u` lex-sorted and unique; how it got
that way is invisible to them. So the sort moves host-side, in chunks that each fit, merged by a tree.

Three arms, all gated **byte-for-byte** against `uniquify_states` on the full padded array — including
the `255` filler tail, since `_is_filler` and `subspace_dim` depend on it.

| N | incumbent (`lax.sort`) | host (one sort) | OOC (chunk + merge) | OOC vs incumbent |
|---|---|---|---|---|
| 800 k | 359 ms | 118 ms | 106 ms | **3.39x** |
| 2 M | 1312 ms | 431 ms | 265 ms | **4.95x** |

Peak sort-phase working set at N = 2 M: **host 244 MiB -> OOC 41 MiB, 5.9x lower**, and **flat** across
`chunk_rows` from 8 k to 512 k. That flatness is the result: peak is set by the chunk size, not by `N`.
It also improves with `N` (0.66x at 200 k to 2.32x at 12.8 M against the host arm), the right shape for
large problems.

### The `union1d` trap — two rejected merges

Both first attempts were *slower than the incumbent*, and the second failure is subtle enough to be
worth recording:

- **Scalar cursor loop** (textbook k-way merge): **0.12x**, ~8x slower. One Python iteration per
  output row.
- **`np.union1d` per pair**: degraded *with* `N`, **0.30x at 12.8 M**. It looks like exactly the right
  primitive — a sorted-unique merge of two sorted unique inputs — and is **80x slower than it should
  be**: 2459 ms against 30 ms for a concatenate-and-radix-sort of the same two 4 M arrays. It calls
  `np.sort` on the concatenation with the default **comparison** sort, discarding both its inputs'
  sortedness and the uint64 radix path.

`np.concatenate` + in-place `sort(kind="stable")` + a neighbour-comparison dedup is what turned 0.30x
into 4.95x. **A library function whose name matches your intent is not evidence about its complexity.**

### What the prototype does NOT do

Stated plainly, because an earlier draft of the POC docstring said it "removes the single-device
constraint," which over-claims:

- It removes the single-device **sort**, bounding the sort's working set. It **distributes nothing**.
- **`B <= 8` only.** `B = ceil((n+1)/8)`, so this covers `n <= 63`. Wider rows take an unchunked
  `np.unique(axis=0)` fallback, correctly measured at 1.0x since both arms run the same code.
  **A 100-site problem is `B = 13` and does not benefit at all** — the headline numbers do not apply
  to the stated target.
- Largest `N` measured is 12.8 M. The disk-spill path is exercised but never where it is load-bearing.
- CPU host timings against a CPU `lax.sort`. On GPU the incumbent is far better optimized, so the
  speedup will compress — the same caveat that turned POC 1's CPU 12–25x into GPU 5.15x. The *memory*
  result should hold.

## 4. Multi-node: the prototype cannot be extended to it

It returns a plain `np.ndarray` with no `.sharding` — single-process host code throughout. Three
blockers, each fatal on its own:

1. **The merge tree is sequential.** One Python process; 8 nodes take the same wall-clock as 1.
   Extrapolating the 12.8 M measurement to `2^31` puts the merge alone in the **~5 minute range on one
   node** while the rest of the solve is distributed — an Amdahl bottleneck, not a solution.
2. **It materializes the full result on one host.** `_pad_to` allocates `[states_size, B]` — **16 GiB
   at `N = 2^31`** — which is exactly what multi-node exists to avoid. Note the 5.9x memory result
   deliberately excluded this allocation to isolate the sort's working set; that was right for the
   question asked, but it means **5.9x does not describe peak process memory**.
3. **Downstream expects a sharded JAX array.** `run_sqd` feeds `states_u` to `get_xsource`
   (`sqd.py:754`) and reshards under `if sharding:` (`sqd.py:768`). Host numpy forces a `device_put` of
   the whole array through one process.

**What a multi-node uniquify requires** — a different algorithm, with the chunked merge demoted to the
per-node kernel:

1. **Sample-based range partitioning.** Sample keys, pick `P-1` splitters; node `p` owns a key range.
   Sublinear.
2. **All-to-all shuffle.** Each node sends keys to their owner. `O(N/P)` per node. The prototype has
   no analogue of this step.
3. **Local sort + dedup**, which is §3's kernel on `N/P` keys.
4. **Boundary dedup is free** — disjoint ranges mean duplicates cannot straddle nodes. This is why
   range- beats hash-partitioning here.

**JAX has no distributed sort primitive**, so step 2 needs `mpi4py` (imported by `examples/` but
declared nowhere) or `jax.experimental.multihost_utils`. It is **untestable on this machine**:
`--xla_force_host_platform_device_count=4` gives virtual devices in *one* process, exercising sharding
specs but never a real inter-host transfer, and per `CLAUDE.md` its timings are meaningless regardless.
Writing code whose central step is unverifiable is a poor trade in a library whose failure mode is
silently-wrong results.

## 5. The 100-site XXZ target: `xsource` is the bottleneck, not the sort

Measured from the real `PauliSumXZ.from_paulisum`, not hand-built:

| n | terms | J (X groups) | max K^(j) | B |
|---|---|---|---|---|
| 20 | 57 | 20 | 19 | 3 |
| 50 | 147 | 50 | 49 | 7 |
| **100** | **297** | **100** | **99** | **13** |

Two consequences: **J = 100** (large — `xsource` memory is linear in it), and **B = 13** so the uint64
fast path does not apply anywhere.

| N | states | `xsource` int64, J=100 | vectors (x4) | total |
|---|---|---|---|---|
| 2^31 | 0.03 TiB | **1.56 TiB** | 0.12 TiB | 1.71 TiB |
| 4x10^9 | 0.05 TiB | **2.91 TiB** | 0.23 TiB | 3.19 TiB |
| 10^10 | 0.12 TiB | **7.28 TiB** | 0.58 TiB | 7.98 TiB |

**`xsource` is ~91% of the footprint.** At N = 4x10^9 that is **~35 GPUs at 100 GB each just to hold
indices**. So the sort ceiling is the wrong thing to attack for this target; `xsource` memory is.

### Sz conservation means N is not set by the physics

XXZ conserves total Sz, so the subspace lives in one magnetization sector:

| sector | dimension |
|---|---|
| 50 up (half filling) | 1.009x10^29 |
| 30 up | 2.94x10^25 |
| 20 up | 5.36x10^20 |

`N = 2^31` is **10^-20 of the half-filling sector**. Going to 10^10 buys 5x more of a space sampled at
10^-19 density. For SKQD, accuracy comes from *which* states the Krylov rungs surface, not how many —
so raising `N` is unlikely to be where accuracy comes from.

### Recommended order for this target

1. **Measure the convergence curve first.** n=100 XXZ at N = 10^6, 10^7, 10^8; plot energy error vs N.
   If it is flat by 10^8, the whole `> 2^31` project is unnecessary and the answer is better sampling.
   Cheap, runs on one machine.
2. **Cut J before raising N.** `cache_level=(1,1)` is **16x less `xsource` memory for 2.4–2.6x slower
   matvec** (`scaling-pocs.md:66`). That turns 1.56 TiB into ~100 GiB — one node instead of 35. Far
   better than distributed sorting, and **it already exists**.
3. **Only then int64 + distributed**, remembering that int64 doubles the dominant array.
4. **If `B > 8` chunking is needed**: pack to **two** uint64s (B=13 fits 16 bytes) and merge
   lexicographically on the pair, keeping radix sort on the high word.
5. **Speculative, unmeasured:** a combinatorial rank within the fixed-magnetization sector maps states
   to a dense integer range, potentially replacing lex-sort-plus-binary-search with arithmetic. The one
   idea that could beat the alpha=0.92 gather, since it removes the memory lookup. Ranking C(100,50)
   needs big integers or careful fixed-point.

## 6. N = 10^30 is impossible, and not for engineering reasons

| quantity | value |
|---|---|
| packed states at B=13 | **1.3x10^31 bytes** (~10^7 YiB) |
| vs all world data storage (~200 ZB) | **65,000,000x** |
| C(100,50), the half-filling sector | 1.009x10^29 |
| **N = 10^30 vs that sector** | **~10x LARGER** |
| full 2^100 Hilbert space | 1.27x10^30 |

No ordering of bottlenecks is meaningful here. Every array is ~10^30 entries, exceeding world storage
by seven orders of magnitude; distributed sorting, int64 indices and combinatorial ranking are all
irrelevant.

The substantive point is not the storage: **N = 10^30 is ~10x the entire half-filling sector and
comparable to the full 2^100 Hilbert space.** At that N you are not doing SQD — SQD's premise is a
subspace *much smaller* than the full space. Exact diagonalization of a 100-site chain is precisely
what SQD exists to avoid.

**The practical frontier is N = 10^9–10^10**, with `cache_level=(1,1)` as the lever inside it.

**If the goal is the physics rather than benchmarking SQD**: for a 1D XXZ chain, DMRG is near-exact at
100 sites at a tiny fraction of the cost, exploiting the low entanglement of a gapped 1D ground state.
If the goal *is* to stress SQD, N = 10^8–10^9 with good sampling is the meaningful experiment.

## Status and what is unverified

**Nothing in `rqutils/` was modified.** The only artifact is
`examples/scaling/poc9_ooc_uniquify.py` (new, untracked). `pytest` 549 passed; ruff, ruff format and ty
clean.

Open, in rough priority order:

- The convergence curve of §5.1 — the measurement that decides whether any of this is needed.
- POC 9 on GPU (expect the speedup to compress, the memory result to hold).
- POC 9 chunking for `B > 8`, without which `n >= 64` gains nothing.
- The distributed shuffle of §4, **unverifiable without real multi-node hardware**.
- Anything at `N` near `2^31`: largest measured here is 12.8 M, and `scaling-pocs.md` already notes
  that regime is multi-node and out of reach.
- The combinatorial-rank idea of §5.5 is speculation, not a result.
