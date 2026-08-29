# Cutting the `xsources` cache: partial-J caching and a Bloom pre-filter

A design note, not a change. Every number here is measured in-tree on one laptop CPU and recorded in
`NOTES.md`; no library code implements any of it yet. Written after profiling an n=100 solve showed the
scaling bottleneck is not where the rest of the scaling work has been aimed.

## Contents

- [1. The problem](#1-the-problem)
- [2. Partial-J caching](#2-partial-j-caching)
- [3. The Bloom pre-filter](#3-the-bloom-pre-filter)
- [4. The two compose](#4-the-two-compose)
- [5. The capacity, which is the blocker](#5-the-capacity-which-is-the-blocker)
- [6. What an API would look like](#6-what-an-api-would-look-like)
- [7. What is not claimed](#7-what-is-not-claimed)

## 1. The problem

`cache_level=(source_indices, diagonals)` picks among six matvec strategies. The source axis is binary:
`1` stores a `[J, N]` int32 array of source indices, `0` recomputes them inside every matvec. Budgeting an
n=100 solve at `B = 13`, `J = 50`:

| N | states | vectors (~6) | **xsources `[J,N]` int32** | total |
| --- | --- | --- | --- | --- |
| 2^24 (17M) | 0.2 GB | 1.6 GB | **3.4 GB** | 5.2 GB |
| 2^28 (268M) | 3.5 GB | 25.8 GB | **53.7 GB** | 82.9 GB |
| 2^31 (2147M) | 27.9 GB | 206.2 GB | **429.5 GB** | 663.6 GB |

The cache is **65% of the footprint** — the largest single object, 8× the state list. So `CLAUDE.md`'s
"prefer `cache_level[0] = 1`" is right at the sizes it was measured at and **becomes impossible exactly
where scaling matters**: on a 16 GB node at `N = 2^28`, 14 of 50 groups fit.

Turning it off is worse at n=100 than the recorded 7–11×. Measured at n=100, N=200k, J=16 with the
word-based search: an uncached matvec is **59.8× slower** (106.4 ms against 1.8 ms), and the precompute
breaks even after **1.5 matvecs** against a solve's 100–300. The wide-row search is dearer per call, so
the two settings degrade to *"does not fit"* and *"60× slower"*.

Nothing between them is expressible today. That is the gap.

## 2. Partial-J caching

Cache `J'` of the `J` groups and recompute the rest. `docs/scaling-pocs.md` §2 measured this and gated it
precisely: **"only worth building if the full cache genuinely does not fit — otherwise always cache
everything."** At n=100 with large `N` that condition is now met; it was not when the POC ran.

Re-measured with the current implementation, n=100, N=200k, J=16:

| `J'` | cache | matvec |
| --- | --- | --- |
| 0 | 0 MB | 106.0 ms |
| 4 | 3.2 MB | 79.6 ms |
| 8 | 6.4 MB | 54.6 ms |
| 12 | 9.6 MB | 28.2 ms |
| 16 | 12.8 MB | 1.7 ms |

Linear in `J'`, so a genuine continuous dial rather than a step function. Note the last step is
disproportionate: **a partial cache never reaches full-cache time.**

## 3. The Bloom pre-filter

On a realistic SQD subspace almost every search finds nothing — the hit rate tracks subspace density, and
at n=100 that is well under 1%. Over 99% of targets pay `log2(N)` serialized loads to discover a partner
is absent. A membership filter rejects them in one.

**False positives are safe here, and that is not generally true of a filter.** `get_xsource` ends with an
explicit equality test on both paths (`found = keys[pos] == target_keys`; `jnp.all(W[pos] == Wt)` on the
wide path), so a false positive costs one wasted `searchsorted` that then correctly reports absent. **The
output stays exact.** False negatives would be fatal, and a Bloom filter cannot produce them.

### Why Bloom rather than an exact bitmap

An exact membership bitmap is `2^n / 8` bytes — it indexes the *Hilbert space*, so it dies at n≈34 however
sparse the subspace is. **A Bloom filter sizes by `N`**, which removes the exponential term entirely:

| target FP | bits/item | k | at N=24M | at N=2^31 | exact bitmap |
| --- | --- | --- | --- | --- | --- |
| 10% | 4.79 | 3 | 14 MB | 1.3 GB | n=30: 0.13 GB |
| 1% | 9.59 | 7 | 29 MB | 2.6 GB | n=34: 2.15 GB |
| 0.1% | 14.38 | 10 | 43 MB | 3.9 GB | n=40: **137 GB** |

At n=100 the exact bitmap would need 1.6e29 bytes. The filter needs 1.2 MB at N=1M.

### Why not a binary fuse filter

`arxiv.org/abs/2201.01174` — within 13% of the storage lower bound against Bloom's 44%, and a query is a
fixed 3 gathers regardless of FP rate where Bloom needs `k = -log2(p)` hashes. Implemented and verified
(1.130n array, 9.04 bits/key, measured FP 0.389% against the 2^-8 = 0.391% theory, no false negatives).
**Its query is better than Bloom's** — 3.62× against 2.76× at n=30, N=4M, at a *lower* FP rate.

It is ruled out by construction cost, not by query performance. Peeling is a sequential graph algorithm
with a data-carried dependency: measured ~2.4 us/key, projecting to **~60 s at N=24M** against the ~3.4 s
`J = 50` precompute it would accelerate. Round-based vectorization degenerates — at n=20k the peel rate
falls from 4612 in round 1 to ~80 by round 30 — because at 1.125n the hypergraph sits deliberately near
the peelability threshold, which is exactly what buys the 13% space overhead. **The space efficiency and
the sequential construction are the same design choice.** Bloom wins here only because its build is one
vectorized `np.bitwise_or.at`: no loop, no failure mode, no reseed path.

## 4. The two compose

Cache the `J'` groups that fit, filter the recompute for the rest. Measured together at n=100, N=600k,
J=16, with a `p = 1%` filter at **0.72 MB** against a full cache of 38.4 MB. Fully-cached matvec is the
reference at 6.2 ms. **Every arm verified exact against it.**

| cached `J'` | cache | recompute, plain | recompute, **+ BF** | BF gain | vs full cache |
| --- | --- | --- | --- | --- | --- |
| 0 | 0 MB | 447.4 ms | **47.4 ms** | **9.43×** | 7.71× |
| 4 | 9.6 MB | 336.5 ms | **38.6 ms** | **8.73×** | 6.27× |
| 8 | 19.2 MB | 162.6 ms | **28.5 ms** | **5.70×** | 4.64× |
| 12 | 28.8 MB | 112.6 ms | **20.0 ms** | **5.64×** | 3.25× |

**The filter earns its 0.72 MB at every point on the dial**, not only at `J' = 0`: 5.6–9.4× on whatever
share is recomputed. Because it is built once per subspace and shared by every group, the `+BF` column is
**flat** while the cache column grows linearly. That is what makes these a composition rather than two
alternatives — the filter has no per-group cost to trade against the cache.

As a budget, at n=100, J=50:

| N | full cache | 16 GB | 64 GB | 256 GB |
| --- | --- | --- | --- | --- |
| 24M | 4.8 GB | 50/50 cached | 50/50 | 50/50 |
| 268M | 53.7 GB | 14/50 cached | 50/50 | 50/50 |
| 2^31 | 429.5 GB | 1/50 cached | 7/50 | 29/50 |

## 5. The capacity, which is the blocker

The filter's compaction is `jnp.nonzero(mask, size=cap)`, which needs a **static** size. An undersized
`cap` drops hits with **no error**. Three findings, and they determine the shape of any implementation.

### An analytic bound does not exist

`candidates = hits + FP`. The FP tail is tight — a 6-sigma binomial bound is within **0.1–6%** of the mean,
because `Binomial(N, p)` concentrates hard. But it needs `hits`, which is the unknown being computed, and
`hits` can legitimately be `N`: a subspace closed under a hop has a 100% hit rate for that hop. So any
bound not derived from the data collapses to `cap = N`, which is correct and worthless.

### The check is free; deriving the cap is not

| variant | time | vs baseline | overhead |
| --- | --- | --- | --- |
| baseline `searchsorted` | 67.2 ms | 1.00× | — |
| BF, cap given, no check | 24.8 ms | 2.70× | — |
| BF, cap given, **with check** | 24.8 ms | 2.71× | **-0.3%** (noise) |
| BF, cap **derived** + check | 31.7 ms | 2.12× | +27.6% |

The check is `mask.sum()` over a mask already computed — **0.04 ms** at N=4M. The 27.6% is not the count;
it is the *second pass*, since deriving runs the mask once to count and again to search.

### So derive once per sweep, not per group

The naive version is slower than no filter at all. Over a J=16 sweep at N=4M:

| strategy | time | vs baseline |
| --- | --- | --- |
| 16 plain searches | 1083.8 ms | 1.00× |
| derive per group | 2403.8 ms | **0.45×** |
| derive once, check every group, retry on overflow | 518.5 ms | **2.09×** |

Per-group derivation loses because each distinct `cap` is a separate `jit` compilation. Deriving once from
the first group and letting the free check catch a later miss is **4.64× better** and equally exact: worst
case is one extra kernel for the offending group, never a wrong answer.

**The retry path is verified, not assumed.** On a fixture mixing a dense half (closed under bit-0 flips)
with a sparse half, group 0 derives a 65,536 cap and group 1 needs 519,752 candidates: the check fires, it
re-runs at 524,288, and the result is exact. Power-of-two rounding bounds the compilation count — a
16-group sweep on a uniform subspace hits **one** cap value.

Two behaviours any implementation must keep:

- **Raise, do not clamp.** An undersized explicit `cap` must raise, including off-by-one (55,347 against
  55,348 candidates), naming the deficit and the sufficient value.
- **`cap = N` is the safe degenerate case.** On a hop-closed subspace the derived cap clamps to `N`, the
  filter stops paying, and the answer stays correct. Verified at 2,000,000 of 2,000,000 candidates.

### The failure inflates its own speedup

Worth stating because it bit twice during this work. A run with `ccap = 16384` against a true worst case of
17,913 candidates across the J groups silently dropped 933 real hits and reported **5.7× instead of the
honest 8.0×** — the truncated arm does less work, so the bug *flatters* the result. Size the capacity from
the worst case over *all* `J` groups, and verify against `get_xsource` rather than trusting a timing.

## 6. What an API would look like

Sketch, not a specification. The surface is smaller than it first appears: `apply_h`, `get_xsource` and
`uniquify_states` do not take `cache_level` at all — `apply_h` *derives* it from which keywords are passed
— so only `sqd()` and `run_sqd()` are affected.

**A budget is a different kind of parameter from a mode.** `cache_level` is a mode: the caller declares
what they want and must know whether it fits. A budget is a constraint: the caller states what they have
and the library derives the mode. The second cannot be declared inconsistently, which follows the
principle behind the `apply_h` break — make the invalid combinations unconstructible rather than validated.

Constraints any design must respect:

- **`cache_level` must keep working.** `tests/test_sqd.py` sweeps `CACHE_LEVELS` across seven test classes,
  and `CLAUDE.md` records that three bugs hid behind the default `(1, 0)`, each masked by the one before.
  That sweep is load-bearing.
- **A new static parameter follows the `states_size` pattern**, not the `cache_level` one: public kw-only
  with a `None` default, computed default, range-validated at the boundary, forwarded positionally,
  declared in `static_argnames` at *every* jit boundary it crosses, re-guarded inside the innermost jitted
  consumer where the check is free. `cache_level` needs `functools.partial` only because it rides outside
  the positionally-splatted `args`.
- **`sqd` is `@overload`ed** on `return_eigvec`, so a new parameter lands in both stubs plus the
  implementation, and `_check_*` runs at both `sqd` and `run_sqd` — the latter deliberately, for callers
  who bypass the entry point.
- **The derive/check/retry policy is caller-side sequencing**, not one traced function: it is a kernel
  call, a host-side decision, and possibly a second kernel call. It cannot live inside a single `jit`.

Two mechanics of that design were open questions and are now measured. **Hoisting the precompute out of
`run_sqd`'s trace is affordable** — it doubles the compiled-variant count (two jitted functions instead of
one) but costs 1.03× compile time and 1.01× warm run time, and the `states_size` padding still collapses
many input lengths to one compilation. **The per-group overflow sync costs 2.0% at J=50**, or 0.2% if the
`ncand` reads are batched into one host transfer. Batching, though, cannot retry a group until all `J` have
run — fine for a precompute, wrong inside a matvec — which is a second independent reason the filter
belongs on the precompute.

**Minimal viable subset:** partial-J alone. It is a pure memory/time dial with no probabilistic structure,
no capacity parameter and no silent-failure surface, and it delivers the linear curve in §2 on its own. The
filter is strictly additive on top and can follow once the capacity policy is settled.

## 7. What is not claimed

- **No GPU measurement.** All of the above is one laptop CPU. A GPU gather is better optimized relative to
  its sort, so the filter's advantage may compress.
- **The filter shrinks the per-device cache; it does not make the subspace distributable.**
  `jnp.nonzero`'s compaction does *not* shard — it needs a `cumsum` over the masked axis, and on a
  partitioned mask that raises. It works here only because `get_xsource` already requires a **replicated**
  `states` (a partitioned one fails on the baseline), so the mask is replicated by construction. Verified
  under a 4-device mesh: zero collectives, output spec identical to the baseline, exact. But `states`
  still costs `13 * N` bytes on every device — 27.9 GB per device at N=2^31 — which is a separate ceiling
  no filter can touch.
- **Nothing measured through a full `sqd()` solve.** These are matvec and setup-path figures. The
  composition into a solve is inferred from the module docstring's 66–97% setup share, not measured.
- **The linear model is not reliable enough to promise a runtime.**
  `t = J'*t_cached + (J-J')*t_bf` (ratio 7.6×) fits the endpoints exactly and the middle to 4–6% but
  drifts to **17.5% at `J' = 12`** (20.0 ms measured against 16.5 predicted). A budget can size the cache
  by exact arithmetic; it should not advertise a speed.
- **Fixtures are synthetic.** Subspaces closed under one bit flip at fixed Hamming weight, not sampled from
  a real circuit at n=100. Hit rate and prefix structure both differ in reality, and both affect the
  filter.
- **No implementation exists.** Every figure comes from a prototype outside `rqutils/`.

Evidence and post-mortems: `NOTES.md`, under the `N ≤ 2^31 - 1` ceiling section. Related:
`docs/scaling-pocs.md` §2 (the partial-J gate) and §11, `docs/wide-row-packing-session.md` §4.4.
