# Words, not bytes: what verifying the multi-observable request actually found

`rqutils` 0.2.0, branch `dev`, `2c5f3e7` → `b03edb4` (11 commits, 2026-08-29). 615 tests, `ruff` and
`ty` clean, all sharded subprocess harnesses passing. **Nothing pushed** — and `git fetch` fails in this
environment, so verify against the remote from a networked shell before trusting the local ref.

The task was to verify `docs/rqutils-multiobs-request.md` — a drafted change request against
`get_xsource`, written from the `spinchain` side. All three of its asks turned out to be wrong. But the
profiling done to check them found a real defect in two places, and not where the request was pointing.

Two code changes shipped; five ideas were measured and rejected; one prototype landed. Everything below
is measured in-tree on one laptop CPU. Nothing is measured on GPU, and nothing through a full `sqd()`
solve — see [What is not claimed](#what-is-not-claimed).

## Contents

- [1. The request, verified](#1-the-request-verified)
- [2. What shipped](#2-what-shipped)
- [3. Two coverage gaps, found by mutation](#3-two-coverage-gaps-found-by-mutation)
- [4. Scaling to n=100](#4-scaling-to-n100)
- [5. Why the literature does not transfer](#5-why-the-literature-does-not-transfer)
- [What is not claimed](#what-is-not-claimed)
- [Commits](#commits)

## 1. The request, verified

| Ask | Outcome |
| --- | --- |
| An entry point taking a **stack** of X-signatures | **Declined — it already exists.** `_apply_h_kernel` *is* a `lax.scan` over stacked signatures. Measured that form directly: **~1.0×** |
| Whether the **weight-2 case admits a closed form** | **Declined — the premise is false.** `S ^ X` is not a local permutation |
| Whether the gather can be **shared rather than batched** | **No.** It is latency-bound on a `log2(N)` dependent-load chain |

The batched form the request asked for is not a missing capability, so we measured the thing itself
rather than building a wrapper for it. Baseline is one `get_xsource` call per signature; `nsig=16`
weight-2 signatures; **all three forms bit-identical** to the baseline:

| form | 250k | 1M | 4M (3 seeds) |
| --- | --- | --- | --- |
| `jax.vmap` over signatures | 1.32× | 1.09× | 0.84–1.10× (median 0.87×) |
| `jax.lax.scan` over stacked signatures | 0.91× | 0.96× | 0.75–0.97× (median 0.83×) |

Run-to-run spread is wide enough that neither column is distinguishable from 1.0× at 4M. The defensible
claim is **~1.0× with no trend toward a win as N grows**.

### The locality claim is false

The request argued that because every spin-current operator is 2-local, `S ^ X` flips two bits and is
therefore "a *local* permutation of the state list", so a low-weight specialization might beat any
batching. Measured on a dense fixture — all `2^20` states, so the hit rate is 100% and `A[i] = i ^ X`
holds exactly:

| bond (bit pair) | Pauli weight | median `|A[i] - i|` | as fraction of N |
| --- | --- | --- | --- |
| (0, 1) | 2 | 2 | 0.0% |
| (9, 10) | 2 | 1,024 | 0.1% |
| (14, 15) | 2 | 32,768 | 3.1% |
| (18, 19) | 2 | 524,288 | **50.0%** |

Every row is a nearest-neighbour, weight-2 bond — identical Pauli weight. Displacement is the **XOR
distance** `2^b2 ± 2^b1`, set by **bit position, not weight**. A spin-current sweep covers all `n-1`
bonds, so roughly half its operators gather across the whole array however local the operator is
physically.

### Why nothing above works: the chain, not the misses

| N | ns/target | log2 N | **ns/level** |
| --- | --- | --- | --- |
| 250k | 14.7 | 17.9 | 0.82 |
| 1M | 21.1 | 19.9 | 1.06 |
| 4M | 22.7 | 21.9 | 1.03 |
| 16M | 26.9 | 23.9 | 1.12 |

`ns/level` is constant across a 64× range of N: the cost is a **dependent-load chain of `log2(N)`
levels**, where each probe's address depends on the previous probe's result. A breakdown at N=4M
confirms nothing else matters — `_pack_state_keys` is 1.0%, the XOR 0.4%, and `searchsorted` plus the
hit test **95.6%**.

That single fact explains every negative result, including three the request never proposed:

| approach | result |
| --- | --- |
| Batching (`vmap`, `scan`) | ~1.0× — batching cannot shorten a dependency chain |
| Pre-sorting targets for sequential access | 0.88–1.04× — access *order* is not the constraint |
| Top-k-bit bucketing (22 levels → 8) | **0.31–0.51×**, i.e. 2–3× *slower* |

## 2. What shipped

Byte-wise comparison was costing `O(B)` per operation in two places. Both now work on packed `uint64`
words via a new `_pack_state_words` helper.

### 2.1 `get_xsource`'s wide path (`409f779`)

`get_xsource` picks its path statically on packed width: `B <= 8` packs each row into one `uint64` and
uses `jnp.searchsorted`; wider rows fell back to a row-wise lexicographic search comparing **one byte at
a time**. The fast path survives to n=60 (`B = 8`); above it the fallback takes over:

| n | B | path | ns/state |
| --- | --- | --- | --- |
| 30 | 4 | `uint64` | 22 |
| 60 | 8 | `uint64` | 15 |
| 64 | 9 | **lexicographic** | **122** |
| 100 | 13 | **lexicographic** | **189** |

An **~8× per-state cliff** at the boundary — invisible in a profile taken entirely at n=30, which is
what the request's numbers are. A/B against the pre-change module, N=300k, **bit-identical at every
width**:

| n | B | words | before | after | speedup |
| --- | --- | --- | --- | --- | --- |
| 30 | 4 | 1 | 3.9 ms | 4.0 ms | 0.97× |
| 60 | 8 | 1 | 4.1 ms | 4.3 ms | 0.95× |
| 64 | 9 | 2 | 36.8 ms | 9.7 ms | **3.81×** |
| 80 | 11 | 2 | 47.0 ms | 9.6 ms | **4.88×** |
| 100 | 13 | 2 | 56.7 ms | 9.7 ms | **5.81×** |
| 127 | 16 | 2 | 75.8 ms | 9.8 ms | **7.71×** |
| 200 | 26 | 4 | 126.9 ms | 17.6 ms | **7.22×** |

Read this as **~3.6–7.7× across n=64–200**; a repeat run read 3.56× / 4.40× / 5.41× / 6.48× / 7.08×, so
per-row values move ~15% while the trend does not. The `B <= 8` rows are the control — that path is
untouched.

The speedup **grows with n**: "after" is flat at ~9.7 ms from n=64 to n=127, because everything from
`B = 9` to `B = 16` is two words. Cost scales as `ceil((n+1)/64)` words — logarithmic in packed width,
with no `2^n` term.

**The adversarial case holds.** A real subspace shares long prefixes, so a leading-word discriminator
should be less effective than on random rows. It is, and the win survives:

| n=100 fixture | first word unique | before | after | speedup |
| --- | --- | --- | --- | --- |
| uniform random rows | 100.0% | 72.5 ms | 9.8 ms | **7.37×** |
| Hamming weight 50 + 1-hop closure | **37.0%** | 76.6 ms | 13.2 ms | **5.80×** |

### 2.2 `uniquify_states`' lexsort — the bigger cost (`9da8dee`)

Profiling the rest of the pipeline showed §2.1 had optimized the **smaller** of the two costs. At n=100,
N=200k:

| component | time |
| --- | --- |
| **`uniquify_states`** | **143 ms** |
| `get_xsource` (one signature) | 6.6 ms |
| `get_diagonal` | 0.5 ms |
| `get_diag_signs` | 0.4 ms |

`uniquify_states` is **14–27× `get_xsource`** at every width from n=30 to n=127, and unlike
`get_xsource` it had no fast path at all — cost grew linearly in `B` throughout. Its lexsort was
`jax.lax.sort((*states.T, iota), num_keys=B)`, i.e. all `B` uint8 columns as separate key operands, and
XLA compares key operands one at a time.

A/B against the pre-change module, N=220k **including duplicates**, output identical at every width:

| n | B | keys before | keys after | before | after | speedup |
| --- | --- | --- | --- | --- | --- | --- |
| 30 | 4 | 4 | 1 | 62.9 ms | 35.1 ms | **1.79×** |
| 60 | 8 | 8 | 1 | 89.7 ms | 35.5 ms | **2.53×** |
| 64 | 9 | 9 | 2 | 102.6 ms | 42.4 ms | **2.42×** |
| 100 | 13 | 13 | 2 | 158.4 ms | 43.2 ms | **3.67×** |
| 127 | 16 | 16 | 2 | 204.6 ms | 43.7 ms | **4.68×** |
| 200 | 26 | 26 | 4 | 286.1 ms | 58.5 ms | **4.89×** |

These are stable, unlike §2.1's: worst per-row spread across five runs is **3.4%**. The difference is
structural — `uniquify_states` is dominated by one large `lax.sort`, a single long-running kernel, where
§2.1's path is a `lax.scan` of `ceil(log2 N) + 1` short steps far more exposed to scheduling jitter.

**This one helps at n=30 too** — the key count drops at every width. Combined with §2.1 on the setup
path the module docstring measures at **66–97% of a solve**:

| n | before | after | speedup |
| --- | --- | --- | --- |
| 30 | 80.4 ms | 56.1 ms | **1.43×** |
| 64 | 273.5 ms | 88.0 ms | **3.11×** |
| 100 | 418.5 ms | 89.1 ms | **4.70×** |
| 127 | 554.5 ms | 92.2 ms | **6.01×** |

### The cost of both: packing widens the data

`_pack_state_words` rounds `B` up to a multiple of 8, so it adds `8*ceil(B/8) - B` bytes per row — worst
at n=64 (`B = 9` → 16 bytes, +7/row), free at n=127. Measured **1.09–1.69× compiled temp bytes** at
N=1M. Negligible at working sizes (0.07–0.17 GB at N=24M) but **6–15 GB at the `2^31` ceiling**, so it
is the wrong trade for any design whose purpose is bounding memory — see §4.2.

This was **not measured before the commit**; it surfaced only when the out-of-core work forced the
question. It is now documented on the code.

## 3. Two coverage gaps, found by mutation

Both fixes had a plausible-looking wrong version that the existing suite passed.

| mutation | defect | existing tests |
| --- | --- | --- |
| Word loop reversed to LSW-first | Returns a *permutation* of the right answer | **12/12 passed** |
| `num_keys=1` on the lexsort | Output not lex-sorted → breaks `get_xsource`'s precondition | **189/189 passed** |

The first survives because the byte-width reference test concentrates its fixture's variation in the
*trailing* bytes, so the leading word rarely decides a comparison. The second survives because a group
must share an entire 8-byte word before `num_keys=1` can reorder anything, and most fixtures vary the
leading bytes. At `B = 13` the first word holds byte 0 alone, so rows agreeing on byte 0 are exactly
such a group.

Two tests were added, each verified by mutation to fail against the defect it targets and pass against
the fix:

- `test_wide_rows_compare_most_significant_word_first` — forces byte 0 and the tail to disagree on
  order, where MSW-first and LSW-first give different answers.
- `test_wide_rows_sort_on_every_word_not_just_the_first` — asserts sortedness directly on a fixture
  where four rows share byte 0.

### One invariant nearly shipped, and it was false

A draft docstring for `_pack_state_words` claimed that trailing-padding would reorder rows. **It does
not** — appending a constant number of zero bytes is a left-shift by `8 * pad`, and a constant
left-shift is monotonic. Verified exhaustively at `B = 3` and over 20,000 random pairs at `B = 9`. The
padding end is a *compatibility* choice (it makes `nwords == 1` reproduce `_pack_state_keys` bit for
bit), not a correctness one, and the docstring now says so.

## 4. Scaling to n=100

### 4.1 Hilbert-space indexing — rejected

A bitmap over all `2^n` basis states plus a per-word cumulative popcount returns a state's *position* in
two loads and one `popcount` — no search at all. It is the largest speedup measured, emits **0** `sort`
ops against the baseline's 117, and shards cleanly with zero collectives:

| fixture | hit rate | baseline | rank-select | speedup |
| --- | --- | --- | --- | --- |
| uniform random | 0.37% | 1607 ms | 71.7 ms | **22.4×** |
| clustered (1-hop closed) | 6.77% | 1556 ms | 70.1 ms | **22.2×** |
| dense low block | 36.8% | 1581 ms | 27.4 ms | **57.7×** |

**It dies at n≈34.** Memory is `2^n / 4` bytes:

| n | 30 | 32 | 34 | 36 | 40 | 100 |
| --- | --- | --- | --- | --- | --- | --- |
| memory | 0.27 GB | 1.07 GB | 4.29 GB | 17.2 GB | 275 GB | 3e29 bytes |

The same wall applies to the two weaker variants measured on the way — a membership pre-filter
(`2^n / 8` bytes, 2.66–4.09× by skipping the search for the >99% of targets that cannot hit) and a
direct index array (`2^n * 4` bytes, 69×). All three index the *Hilbert space* rather than the subspace.

The pre-filter additionally has a **silent-wrong-answer** hazard: its compaction needs a static `cap`,
an undersized `cap` drops hits with no error (883 returned against 884 true candidates), and there is no
provable bound below `N` — a subspace closed under a hop has a 100% hit rate for that hop.

### 4.2 Out-of-core chunk-and-merge — rejected

`examples/scaling/poc9_ooc_uniquify.py` bounds the sort's working set by chunking, but bails out at
`B > 8` citing "no uint64 equivalence available" — which `_pack_state_words` now supplies. So the wide
case was built and measured. At n=100, B=13, N=8M host-side:

| approach | time | peak RSS |
| --- | --- | --- |
| `np.unique(rows, axis=0)` | 7.3 s | **365 MB** |
| word-packed `np.unique` | **4.9 s** | 1655 MB |
| chunked sort + merge tree | 31.3 s | 1048 MB |

**4.3× slower and 2.9× more peak memory** than the incumbent — it fails on both axes it exists to win.
The cause is the widening from §2.3: buying speed with memory is right for the in-JAX sort, whose own
working set dominates, and exactly wrong here. **The shipped changes and an out-of-core design pull
opposite ways**, which is worth stating because §2.2 otherwise reads as a step toward this one.

### 4.3 Range-partitioned sample sort — prototype (`poc11_range_partition.py`)

POC 9's docstring names the missing algorithm: a range-partitioned shuffle, for which chunk-local
sorting is the per-node kernel. That shuffle was built, and **it is the one design that removes the
single-device sort.** Sample sort with data-derived splitters → scatter into fixed-capacity buckets →
`NSH` **independent** `lax.sort`s under `vmap`. Splitters guarantee bucket `i` < bucket `i+1`, so
concatenating in order is globally sorted with no cross-bucket comparison.

Output bit-identical to `np.unique(rows, axis=0)`, and **zero `all-gather` / `all-reduce` /
`collective-permute`** in the compiled HLO. It reads 1.70–2.32× against the incumbent at N=0.4M–3.4M,
but that is on **four virtual CPU devices** where this repo's own rule makes timings meaningless. The
claim is **shardability**, not speed.

Four traps, each from a wrong turn:

| trap | measurement |
| --- | --- |
| Equal-range splitting on the packed MSW | **4.00×/8.00× imbalance** at NSH=4/8 — every row in one bucket, because at n=100 that word is 7 bytes of leading pad. Data-derived quantiles give 1.07–1.19× |
| `argsort(bucket)` for within-bucket rank | Reinstates the global sort being removed: **256 ms against 8 ms** at N=2M, and 208 HLO `sort` ops against the incumbent's 29 |
| `[N, NSH]` one-hot cumsum instead | Same speed but `4*N*NSH` bytes — **34 GB at `N = 2^31, NSH = 4`**, growing *with* the device count. Use `NSH` sequential `[N]` cumsums: `O(N)`, same time |
| Undersized bucket capacity | Drops rows (16,090 lost at slack 1.05) — but the kernel **returns the overflow count**, so a caller can raise. That detectability is what separates this from §4.1's `cap` |

**A version measured "1.15× faster" while still containing the global `argsort`.** Only reading the
compiled HLO caught it; a timing-only check would have shipped a broken design.

Two pieces of real work remain before it could replace `uniquify_states`: splitter selection is
host-side numpy (`O(1)` in `N`, so it does not reintroduce the ceiling, but it is a centralization
point), and reassembly into the `[states_size, B]` contract is unimplemented.

### 4.4 But the sort is not what binds

Budgeting an actual n=100 solve at `B = 13`, `J = 50`, `cache_level[0] = 1`:

| N | states | vectors (~6) | **xsources `[J,N]` int32** | total |
| --- | --- | --- | --- | --- |
| 2^24 (17M) | 0.2 GB | 1.6 GB | **3.4 GB** | 5.2 GB |
| 2^28 (268M) | 3.5 GB | 25.8 GB | **53.7 GB** | 82.9 GB |
| 2^31 (2147M) | 27.9 GB | 206.2 GB | **429.5 GB** | 663.6 GB |

The cache is **65% of the footprint** — the largest single object, 8× the state list. So `CLAUDE.md`'s
"prefer `cache_level[0] = 1`" is right at the sizes behind it and **becomes impossible exactly where
scaling matters**: on a 16 GB node at `N = 2^28`, 14 of 50 groups fit.

And the penalty for turning it off is far worse at n=100 than the recorded 7–11×. Measured at n=100,
N=200k, J=16 with the word-based search: an uncached matvec is **59.8× slower** (106.4 ms against
1.8 ms), and the precompute breaks even after **1.5 matvecs** against a solve's 100–300. The wide-row
search is dearer per call, so at n=100 the two settings are *"does not fit"* and *"60× slower"*.

**The partial-J dial is the answer**, and `docs/scaling-pocs.md` §2 already scoped it correctly: *"only
worth building if the full cache genuinely does not fit — otherwise always cache everything."* At n=100
with large `N` that condition is now met, which it was not when that POC ran. Re-measured with the
current implementation, caching `J'` of `J = 16` groups and recomputing the rest:

| `J'` | cache | matvec |
| --- | --- | --- |
| 0 | 0 MB | 106.0 ms |
| 4 | 3.2 MB | 79.6 ms |
| 8 | 6.4 MB | 54.6 ms |
| 12 | 9.6 MB | 28.2 ms |
| 16 | 12.8 MB | 1.7 ms |

Linear in `J'`, so a genuine continuous dial. The API shape this wants is a **memory budget**, not a
mode: cache `floor(budget / (4N))` groups. Note the last step is disproportionate, so a partial cache
never reaches full-cache time.

**Not implemented.** It is an API change (`cache_level` → a budget) touching `run_sqd`, `apply_h` and
`_apply_h_kernel`, and `CLAUDE.md`'s rule about `static_argnames` versus `functools.partial` makes it
delicate. It deserves its own pass.

## 5. Why the literature does not transfer

Four techniques were considered. All fail for the same reason.

| technique | source | why it fails here |
| --- | --- | --- |
| Lin two-sublattice tables | Lin (1990) | Requires a complete symmetry sector |
| Divide-and-conquer index lookup | DanceQ, `tmp/2407.14591v2.pdf` | Offsets/strides are binomial counts (Eqs. 15/19/20/23) |
| Residue arrays / dynamic bit masking | Selected-CI literature | Exists to *discover* unknown excitations |
| Rank-select | Succinct data structures | Indexes the Hilbert space (§4.1) |

All four assume the basis is **characterized** — every state at fixed particle number — so "how many
states precede this one" is a closed-form combinatorial count. An SQD subspace is a **sampled subset**:
that count exists only in the sampled list, so the offsets and strides those methods need do not exist,
and no subsystem partitioning creates them.

The selected-CI comparison is worth making explicitly, since that field has worked at 100+ orbitals for
decades. Its sorting-based paradigm was tested here and measures **0.41×** — the `argsort` of 16N keys
alone (429 ms) costs more than all 16 independent searches (296 ms). Their sorting wins because their
problem requires *discovering* which determinant pairs are connected; `get_xsource` is **given** `X`, so
a direct search is already the right shape. That asymmetry is also why `23fb226`'s removal of the old
sort was correct.

**What does transfer** is DanceQ §4.4: for a matrix-free matvec, *"the memory footprint of each worker
process should be the guiding principle"*, because runtime depends only weakly on the lookup scheme.
That is §4.4's partial-J recommendation, arrived at independently. For calibration, their state of the
art is **46 spins on ~256 nodes at 512 GiB each**, with 120 TiB for two wave functions — useful
confirmation that `N = 2^31` is inherently multi-node.

## What is not claimed

- **No GPU measurement.** Everything is one laptop CPU. A GPU sort is well optimized relative to its
  gather — `23fb226`'s own CPU-to-GH200 ratio compressed from 12–25× to 5.15× — so §2.2's ratio in
  particular may compress. This is the largest open risk.
- **Nothing measured through a full `sqd()` solve.** The combined 1.43–6.01× covers the *setup* path.
  The 66–97%-of-a-solve composition comes from the module docstring, not from a solve run here.
- **The n>60 path is thinly tested.** The byte-width reference test covers n=55/63/64/71/80 against an
  independent lookup table, but **no test runs a subspace anywhere near n=100** — `TestInt32Ceiling`
  bounds `N`, not `n`. A wrong answer on this path is a *permutation* of the right one, so it stays
  symmetric and finite and will not announce itself.
- **Fixtures above n=60 are synthetic.** Generated rows, not sampled from a real circuit at those sizes.
  The adversarial structure was tested for §2.1 (weight-50 plus 1-hop closure) and the win held; §2.2's
  A/B used random rows plus injected duplicates, and a real subspace's duplicate rate and prefix
  structure differ.
- **§4.3 is unmeasured on real devices.** Four virtual CPU devices only.
- **`spinchain`-side numbers were not verified** — the ~47% packing hoist, the 4.6e-19 agreement, the
  replay timings. They are consistent with everything checkable here.

## Commits

| commit | what |
| --- | --- |
| `c825b76` | Record the request, its response, and the n=30 profile |
| `1863e9f` | Lead the response with a scalable win; bound rank-select at n≤32 |
| `409f779` | **Code:** compare `uint64` words on `get_xsource`'s wide path |
| `ab3358d` | Requote §5.1 from the shipped A/B |
| `9da8dee` | **Code:** lexsort `uniquify_states` on `uint64` words |
| `2498cf1` | Add §5.2: `uniquify_states` was the bigger cost |
| `b561867` | State §5.2's noise band and why it is tighter |
| `0fdbc7d` | Reject the out-of-core replacement; document what packing costs |
| `d0a7c3f` | POC 11: range-partitioned uniquification |
| `e445fc4` | Add §5.3: the range-partitioned shuffle works |
| `b03edb4` | Record that the `xsources` cache is what binds, not the sort |

Code is **181 insertions / 32 deletions** across `rqutils/sqd.py` and `tests/test_sqd.py`. The other
~1,121 lines are `docs/`, `NOTES.md`, and the POC.

Related: `docs/rqutils-multiobs-request.md` (the request), `docs/rqutils-multiobs-response.md` (the
reply, from the `rqutils` side), `docs/scaling-pocs.md` §11, `NOTES.md` under the `N ≤ 2^31 - 1` ceiling.
