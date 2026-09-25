# Partitioning `states` and `uniquify_states`

Prototyped 2026-08-29/30 on virtual CPU devices (`--xla_force_host_platform_device_count`, 4 unless
stated), so every result is correctness and structure only: **no speed claim is made anywhere below**, and
nothing ran on a real interconnect. Nothing here is in the library. Scripts: `poc/hash_partition.py`,
`poc/hash_partition_jax.py`, `poc/range_partition.py`, `poc/uniquify_sharded.py` (§13), and
`test/sharded/diagonals.py` for §10.

## The idea

The `13 * N` replicated state list is the one term the `(0, 0)` floor cannot shed — 27.9 GB **per
device** at `N = 2^31` — and `sqd.py` (`:1042` when this was written) records it as a hard requirement: a
partitioned `[N, B]` "fails outright". Investigated with a breaking change permitted. **It is not a wall,
and the literature has the established solution**: give each device `N/d` states, route every target
`S ^ X` to the device that owns it, binary-search there, route the index back. That converts the
27.9 GB/device wall into `27.9/d`. `get_xsource` needs a *balanced* partition, and gets it from hashing
the whole key (§3–§4); `uniquify_states`, whose output feeds a binary search, needs a *globally ordered*
one and must use range splitting (§7–§8). Every mechanism is now built and exact; whether the routing pays
is the open question (§12).

## 1. What actually fails on a partitioned `states`

It is narrower than "states must be replicated." Per ingredient, on a `P('x', None)` state array under a
4-device mesh:

| operation | partitioned |
| --- | --- |
| `bitwise_xor` (build `S ^ X`) | **OK**, `P('x', None)` |
| `_pack_state_keys` | **OK**, `P('x',)` |
| `jnp.searchsorted(keys, targets)` | **fails** — "Unmapped values passed to vmap cannot be sharded" |
| `keys[pos]` gather | fixable with `out_sharding=` |

The *targets* shard fine. What fails is that a binary search needs the whole sorted haystack visible to
every query. That is a data dependency, not a JAX gap — and it is only unavoidable **given a
range-agnostic partition**.

**The established scheme is Wietek & Läuchli**, *Phys. Rev. E* **98**, 033309 (2018), described in
`awietek.github.io/assets/pdf/thesis_awietek.pdf` §3.3. Split each basis state into **prefix** and
**postfix** bits; states sharing a prefix live on one rank; **a hash of the prefix bits gives the owning
rank**, so — quoting — *"we also don't have to store any information about their distribution. This
information is all encoded in the hash function."* Within a rank states stay lexicographically ordered,
so the local lookup is still a binary search, over `N/d` rows. The matvec buffers `(target, coefficient)`
pairs locally, does one **`MPI_Alltoallv`**, then each rank searches its own list. The paper hashes rather
than range-partitions deliberately: a random distribution *"reduces load balance problems significantly
since the communication structure is randomized. This is in stark contrast to distributing the basis
states in a linear fashion,"* where single processes take a multiple of the workload. Range splitters
(`poc/range_partition.py`, natural here because each shard stays sorted for free) are the linear case.

**Not applicable: DanceQ's approach.** `arxiv.org/abs/2407.14591` reaches 46 spins over ~256 nodes and
120 TiB with *"thread-local lookup tables for fast and synchronization-free state-to-index mapping"* — no
routing at all. That works because its basis is a **complete** U(1) particle-number sector, so state →
index is a closed-form combinatorial map (enumerative encoding, Cover 1973). **rqutils' subspace is an
arbitrary sampled set**, for which no such formula exists — which is why `get_xsource` searches in the
first place. The distinction is structural; do not cite DanceQ as evidence that this is easy.

**All three ingredients work in JAX**, verified on a 4-device mesh: prefix-hash ownership is elementwise
(`P('x',)`, balance 1011/1041/1016/1028 over 4096), `jax.lax.all_to_all` composes inside `jax.shard_map`,
and — the load-bearing one — **`jnp.searchsorted` against the local slice inside `shard_map` works**,
which is what removes the replicated haystack.

**A minimal end-to-end version is exact.** Range-partitioned variant at n=30, N=2564, d=4, hit rate
40.4%: **bit-identical to `get_xsource`** (1036 hits both, `np.array_equal` True), load imbalance
**1.02×**, and a **4.0× per-device** reduction in the state array. The JAX form compiled to **zero
`all-gather` / `all-reduce` / `collective-permute`** and 18 `all-to-all`. The 1.02× is on an evenly-split
*sorted* array and says nothing about a real subspace; §3–§4 test that.

## 2. The cost is traffic, and it is only viable at `cache_level[0] = 1`

Routing a target and its answer is ~12 B/target (8 out as a uint64 key, 4 back as an int32 index),
asymptotically in `d`. At `N = 2^31`, n=100:

| `d` | states/device | routed per group/device |
| --- | --- | --- |
| 4 | **6.98 GB** (from 27.9) | 6.44 GB |
| 16 | 1.74 GB | 1.61 GB |
| 64 | 0.44 GB | 0.40 GB |
| 256 | 0.11 GB | 0.10 GB |

**The multiplier is `J`, not `J × niter`** — `get_xsource` runs once per solve in the precompute at
`cache_level[0] = 1` (`sqd.py:1008` when written), not inside the matvec. So at `d = 4` it is ~650 GB
routed **once** against 20.9 GB/device of permanent residency reclaimed, and it lands on the precompute
measured at only 4.5–8.4% of a solve. At `cache_level[0] = 0` the recompute *is* per matvec and the
traffic becomes ~84 TB per solve at `d=4` — unusable. **So this is only viable together with the source
cache**, the opposite of the Bloom filter's constraint. It composes where the filter did not because the
target is paid **once per solve**, not once per matvec — the same "how many times is it paid" question,
answered favourably.

## 3. Hash the whole key, not the prefix

**Prefix hashing collapses completely on a structured subspace.** Imbalance at n=60, N=200k, d=16, 12
prefix bits:

| fixture | prefix-hash | whole-key hash | range-split | distinct prefixes |
| --- | --- | --- | --- | --- |
| uniform | 1.12× | 1.01× | 1.00× | 2048 |
| fixed-weight n/2 | 1.16× | 1.02× | 1.00× | 2048 |
| low-weight n/8 | **3.81×** | 1.01× | 1.00× | 779 |
| banded (last n/4) | **16.00×** — total collapse | **1.09×** | 1.01× | **1** |

The last column is the mechanism: when excitations are confined to low qubits, **every state shares the
same high bits**, so the prefix hash is constant and one shard takes everything. It is the same root cause
as equal-range splitting's collapse (§7, *"at n=100 that word is 7 bytes of leading pad"*): **the high
bits of an SQD state carry almost no entropy.** A banded subspace is not contrived: it is what a circuit
acting on a subset of qubits produces. (The range-split column balances here because it is measured on
the states; §4 shows why that hides its failure.)

**Hashing the whole key fixes it and costs nothing.** The key is already computed by `_pack_state_keys`,
so it is one extra `mix64`. Wietek & Läuchli hash the *prefix* because their scheme needs prefix-grouping
for local enumeration; **rqutils has no such requirement**, so the constraint does not carry over.

**The one thing it gives up is the free local sort.** A range split hands each shard a contiguous sorted
block; a hash hands it a scattered subset that must be sorted per shard. That is affordable and already
prototyped: `d` independent sorts of `N/d` rows is exactly `poc/range_partition.py`'s phase 3 (`vmap` over
`lax.sort`, zero collectives), and it runs **once at setup**, not per `get_xsource` call. So the design is
*hash to assign owners → per-shard sort → local binary search*, with ownership still metadata-free.

**Verified exact on every fixture** against `get_xsource`, at n=60, d=16, with the hop chosen to act on
qubits each fixture populates (a hop on qubits 0–1 gives the banded fixture a 0% hit rate and tests
nothing):

| fixture | N | hit rate | imbalance | bit-identical | per-device |
| --- | --- | --- | --- | --- | --- |
| uniform | 74,944 | 39.9% | 1.03× | **yes** | 16× less |
| fixed-weight | 75,156 | 40.3% | 1.02× | **yes** | 16× less |
| banded | 6,435 | 53.3% | 1.14× | **yes** | 16× less |

**Balance holds as `d` grows, and the residual is Poisson, not structural.** Including a Zipf-sampled
fixture (2M shots → 148,976 unique, a **92.6% duplicate rate**, matching the 91–97.6% `NOTES.md` measured
for real shot distributions):

| fixture | N | d=4 | d=16 | d=64 | d=256 | d=1024 |
| --- | --- | --- | --- | --- | --- | --- |
| fixed-weight | 200,000 | 1.01× | 1.03× | 1.04× | 1.08× | 1.21× |
| banded | 6,435 | 1.01× | 1.09× | 1.22× | 1.63× | 2.86× |
| zipf-sampled | 148,976 | 1.01× | 1.01× | 1.04× | 1.09× | 1.22× |

**Checked against the balls-in-bins bound** `1 + sqrt(2 ln d / (N/d))` rather than asserted: predicted and
measured agree in magnitude at every point and **both depend only on `N/d`**, not on the fixture. Banded
degrades because its `N` is 6,435, so `d=1024` leaves ~6 states per shard — not because its structure
survives. **That is the hash working: it has erased the structure that destroyed prefix hashing, leaving
ordinary sampling noise**, which a capacity slack absorbs exactly as `poc/range_partition`'s `slack`
parameter does.

**Practical rule:** size shard capacity from `N/d` via the balls-in-bins bound, and keep `N/d` above
~1000 for a slack under 1.15×. At `N = 2^31` that permits `d` up to ~2×10^6 — far beyond any real mesh.

## 4. 1D XXZ decides it: prefix hashing fails at exactly `d`, range splitting fails on the targets

Re-run on a real 1D XXZ subspace, built the way SQD gets one — a Krylov expansion from |Néel⟩ under the
nearest-neighbour hop graph, so it lives in one magnetization sector and stays *local* rather than spread
over the weight shell. **Both alternatives to whole-key hashing fail, and one fails completely.**

**Prefix hashing measures exactly `d`.** At n=30, N=200k, 12 prefix bits:

| scheme | d=4 | d=16 | d=64 | d=256 |
| --- | --- | --- | --- | --- |
| prefix-hash | **4.00×** | **16.00×** | **64.00×** | **256.00×** |
| whole-key hash | 1.00× | 1.02× | 1.04× | 1.11× |

Imbalance equal to `d` means **one shard holds the entire subspace and the other `d-1` hold nothing**. The
cause is **1 distinct prefix**, an artifact of the packing width rather than the physics: `B = ceil(31/8)
= 4` bytes leaves **33 leading zero bits** in the uint64 key, so the top 12 bits are constant *by
construction*. At n=60 and n=100 there are 312 and 280 distinct prefixes and imbalance is still
**2.05–118.98×**, because a magnetization-conserving Krylov subspace concentrates near Néel and even the
non-pad high bits barely vary.

**Range splitting looked competitive and is not — measuring it on the states hides the failure.**
Splitters derived from the state list balance *that list* by construction (1.00–1.85× measured), but
**what gets routed is the targets `S ^ X`**. At n=60, d=64, per XXZ hop:

| hop | hit rate | range-split | whole-key hash |
| --- | --- | --- | --- |
| (45, 46) | 18.4% | 1.01× | 1.05× |
| (29, 30) | 18.5% | 2.48× | 1.03× |
| (15, 16) | 18.3% | 7.90× | 1.04× |
| (0, 1) | 18.4% | **13.68×** | 1.05× |
| (59, 0) | 18.5% | **14.95×** | 1.04× |

The failure is **hop-dependent**: swapping *high-order* bits moves a key far in lex order, piling targets
into few range buckets, while a swap deep in the string barely moves it. **XXZ has a hop on every bond, so
the worst case is always present** in the J-fold sweep.

**And splitters go stale where a hash cannot.** A real SQD run grows the subspace during configuration
recovery, so the set the splitters were derived from is not the set later queried. Splitters from half
the subspace applied to the full set: **32.51×**, against whole-key hashing's **1.04×**. A hash of the key
does not depend on the population at all.

**Verified exact on XXZ across every X group.** n=60, J=61 groups, K=60, N=80,000, d=64: **61 groups, 0
mismatches**, hit rates spanning **2.2%–100.0%** (the 100% group is the one where the subspace is closed
under that hop — the degenerate case where every target is present, handled at the same imbalance),
imbalance **1.04×–1.10×** throughout, 64× fewer state rows per device.

**So the choice is settled by physics, not preference.** On the Hamiltonian this library is most often
pointed at, prefix hashing and range splitting are both unusable; whole-key hashing is 1.03–1.11×
everywhere and invariant to hop, to `d`, and to the subspace growing.

## 5. Variable-length routing: a primitive exists, and the padded fallback's capacity is computable

A JAX implementation needs a routing step where **each shard receives a different, data-dependent
count**. MPI does this with `Alltoallv`; JAX requires static shapes.

**`jax.lax.ragged_all_to_all` exists** (JAX 0.11.1) and is the direct analogue — it takes `(operand,
output, input_offsets, send_sizes, output_offsets, recv_sizes)` and ships exactly the per-destination
counts. **Its known defect does not apply here.** FESOM2-JAX (`arxiv.org/abs/2608.01546`) reports it "has
a defective reverse-mode rule in JAX 10.1 and is usable **forward-only**", and fell back to a padded
variant to keep exact adjoints. `get_xsource` returns **integer indices** and is never differentiated —
`sqd` does not backprop through it — so rqutils can use the ragged path they had to abandon. **It cannot
run here**: `JaxRuntimeError: UNIMPLEMENTED: HLO opcode ragged-all-to-all is not supported by XLA:CPU
ThunkEmitter`. It is a GPU/TPU-only path, and nothing about it is measured.

**The padded fallback is affordable, and its capacity is computable — unlike the Bloom filter's.** Size
each send buffer by the balls-in-bins bound `cap = m/d + sqrt(2·(m/d)·ln d)` with `m = N/d`. Padding
overhead, as a fraction of the data sent:

| `N` | d=4 | d=16 | d=64 | d=256 |
| --- | --- | --- | --- | --- |
| 10^6 | 0.7% | 3.8% | 18.8% | **90.1%** |
| 24×10^6 | 0.1% | 0.8% | 3.8% | 17.4% |
| 2^31 | 0.0% | 0.1% | 0.4% | **1.8%** |

**It shrinks as `N` grows and grows with `d`** — the same `N/d` dependence as the imbalance, and
negligible exactly where the feature matters (0.4% at `N = 2^31, d = 64`). FESOM2-JAX measured padded
against ragged at **0.742 vs 0.589 s/step** on 64 GPUs, a 26% penalty rather than a cliff, so the fallback
is a real option and not a last resort.

**The contrast with the pre-filter's `cap` is structural.** There `cap = hits + FP` and **`hits` is the
unknown being computed**, so any data-independent bound collapses to `cap = N`, "correct and worthless" —
which "blocks the whole pre-filter family". Here the capacity depends only on `N/d`, **known at setup**,
so the bound is tight and derivable in advance. The overflow check is equally free — sum the
per-destination counts and compare — and the rule is **raise, do not clamp**, since an undersized
capacity drops states silently.

## 6. The JAX `get_xsource` composition is exact (`poc/hash_partition_jax.py`)

**Fully verifiable on CPU despite `ragged_all_to_all`**, because the dense `all_to_all` over
fixed-capacity buckets carries the identical dataflow — ragged is a *bandwidth* optimization of the same
routing step, not a different algorithm. **Bit-identical to `get_xsource`** on a 1D XXZ Krylov subspace,
n=30, N=21,716, hit rate 24.9%:

| D | hop | cap | exact | overflow | all-gather / all-reduce / collective-permute / all-to-all |
| --- | --- | --- | --- | --- | --- |
| 2 | (0,1) | 8826 | **yes** | 0 | 0 / 0 / 0 / 8 |
| 2 | (15,16) | 8826 | **yes** | 0 | 0 / 0 / 0 / 8 |
| 2 | (29,0) | 8826 | **yes** | 0 | 0 / 0 / 0 / 8 |
| 4 | (0,1) | 2270 | **yes** | 0 | 0 / 0 / 0 / 12 |
| 4 | (15,16) | 2270 | **yes** | 0 | 0 / 0 / 0 / 12 |
| 4 | (29,0) | 2270 | **yes** | 0 | 0 / 0 / 0 / 12 |

`(0,1)` and `(29,0)` touch the **high-order** bits, exactly where range splitting measured 13.68–14.95×
(§4); under hashing they are indistinguishable from the mid-string hop. **Zero all-gather, all-reduce and
collective-permute** in every case — only the intended `all_to_all`.

**Bucketing must be `D` passes of an `[n]` cumsum, not one `[n, D]` one-hot.** Measured at D=4, **24
B/slot for the one-hot against 13 for the loop**, and the gap widens in `D` since one is `O(N·D)` and the
other `O(N)` (§7 has the same finding at `N = 2^31`). Both give identical buckets — verified against a
numpy reference before either was used.

**The capacity guard works, and the check is sufficient.** `cap` must be static (`all_to_all` needs a
fixed shape), derived from the balls-in-bins bound `mu + sqrt(2·mu·ln D)` with `mu = (N/D)/D`. The kernel
returns a free overflow count (a sum of a mask already computed):

| slack | cap | overflow | exact |
| --- | --- | --- | --- |
| 1.6 | 2270 | 0 | **yes** |
| 0.9 | 1277 | **1284** | **no** |

**Exactness and a zero overflow count coincide**, which is what makes the free check a sufficient guard
rather than a partial one — and why a caller must **raise, not clamp**.

**Two JAX mechanics.** A rank-0 output cannot be concatenated across the mesh — `shard_map` rejects
`out_specs=P('x')` on a scalar with "which has rank 0 (and 0 < 1)", so the overflow count must be
`.reshape(1)`. And the sentinel is `0xFFFF...`, unreachable by construction: the packed keys carry a zero
pad bit at position 0, so no real key is all-ones.

The setup phase (owner assignment plus per-shard sort) is host-side numpy here; in the library it is
`poc/range_partition`'s phase 3.

## 7. `uniquify_states` must range-partition, and the single-round shuffle works

**It cannot use hash partitioning.** Its output feeds `get_xsource`, which binary-searches, so the result
must be **globally lex-sorted**, and a hash destroys global order by design. Range partitioning is
mandatory — splitters guarantee bucket `i` < bucket `i+1`, so concatenating in order is globally sorted.
**That is a real asymmetry**: `get_xsource` needs balance and gets it from hashing; `uniquify_states`
needs order and must accept range splitting's imbalance.

**The range-partitioned shuffle (`poc/range_partition.py`) removes the single-device sort**, and is
correct: output bit-identical to `np.unique(rows, axis=0)`, and **zero `all-gather` / `all-reduce` /
`collective-permute`** in the compiled HLO. Sample sort with data-derived splitters, fixed-capacity
buckets, then `NSH` independent `lax.sort`s under `vmap`; no cross-bucket comparison ever happens. Four
findings, each of which cost a wrong turn:

- **Equal-range splitting collapses.** Splitting the packed most-significant word's nominal `2^64` range
  gives **4.00x/8.00x imbalance at NSH=4/8** — every row in one bucket. At n=100 that word is 7 bytes of
  leading pad (a consequence of `_pack_state_words`' leading-pad choice), so its range carries almost no
  information. Data-derived quantiles give **1.07–1.19x** on both uniform and fixed-Hamming-weight
  fixtures.
- **A global `argsort` for within-bucket rank defeats the whole design.** It is a global sort, the exact
  thing being removed: **256 ms against 8 ms** at N=2M, and 208 `sort` ops in the HLO against the
  incumbent's 29.
- **The `[N, NSH]` one-hot cumsum is the wrong fix.** Same speed as the alternative but `4*N*NSH` bytes —
  **34 GB at `N = 2^31, NSH = 4`, 275 GB at NSH=32**. It grows *with* the device count, so adding devices
  to reach larger `N` makes it worse. `NSH` sequential `[N]` cumsums are `O(N)` memory, same time,
  identical result.
- **Capacity overflow is detectable, which is what makes this shippable.** An undersized capacity drops
  rows (16,090 lost at slack 1.05), but the kernel *returns the overflow count*, so a caller can raise.
  The rank-select prototype (`markdown/spinchain/rqutils-multiobs-response.md` §5.3) had an analogous
  `cap` with no detectable failure mode — that is why one is a candidate and the other is not. Slack must
  exceed the splitter imbalance; 1.35 was sufficient in every fixture, and the padded array is `slack`
  times the input.

Timings against the incumbent on 4 virtual CPU devices, only to show the structure is not pathologically
slow (**virtual-device timings are meaningless**; the claim is shardability, not speed):

| N | incumbent | range-partitioned | ratio |
| --- | --- | --- | --- |
| 420,000 | 109.2 ms | 49.8 ms | 2.19x |
| 1,680,000 | 482.1 ms | 225.0 ms | 2.14x |
| 3,360,000 | 863.5 ms | 496.9 ms | 1.74x |

(NOTES recorded the range across runs as 1.70–2.32x.) It left two gaps — splitter selection is host-side
numpy, and it returns `[NSH, cap, NW]` blocks rather than the `[states_size, B]` contract — and probing
them found a third. Each was closed separately:

**Gap 1 — in-graph splitters: works, but they must compare the full row.** A strided sample gathered with
`.at[idx].get(out_sharding=P(None, None))` lands replicated (the explicit spec is required; JAX cannot
infer it off a partitioned axis), and sorting `d*64` rows is negligible. **But ordering the sample by its
lead word alone collapses.** On a fixture with a constant lead word and the order carried entirely by the
tail: lead-word-only puts **4000 of 4000 rows in one bucket**, full-row lex gives 990/990/990/1030.
Neither produces an *ordering violation* — lead-word splitting is sound, never wrong — it simply cannot
see the tail. **An n=100 XXZ fixture does not catch this**: 94.1% of its rows share a lead word with
another row, yet enough distinct lead values remain that both schemes measured 0 violations and balance
looked fine at 1.30×. The adversarial fixture was necessary to separate them.

**Gap 2 — reassembly into `[states_size, B]`: works.** Bucket `k`'s rows land at `sum(unique counts of
buckets < k)`, and those counts are *traced*. A scatter at traced offsets is expressible: park dead slots
at index `states_size` in a `states_size + 1` buffer and drop them (`mode='drop'`), then slice. Verified —
output globally sorted, unique count exact, filler correct. `states_size` must be `static_argnums`, which
it already is in the real function. After dedupe the live rows inside a block are **not contiguous**
(interior duplicates are blanked), so the within-block destination needs a *re-rank* (`cumsum(live) -
1`), not the original slot index.

**Gap 3 — the global prefix sum, the blocker the single-round shuffle did not reach.** Ranking a row
within its bucket is `cumsum(bucket == k)` over the **sharded** axis, which raises `ShardingTypeError: The
input should be fully replicated when axis is not specified to cumsum` — the rule that *"everything that
reorders or compacts along the sharded axis fails; only elementwise ops and reductions survive."*
`poc/hash_partition_jax` escapes it because its cumsum runs **inside `shard_map`** over each shard's own
slice; these ranks must be *global*. **Resolved by a two-level prefix sum**, the standard distributed
counting-sort structure: each shard cumsums its own slice for a local rank, one `all_gather` of a `[d, d]`
per-shard-per-bucket count matrix, then add the exclusive prefix over lower-numbered shards. **Verified
against a sequential reference** — global within-bucket ranks bit-identical, per-shard counts summing to
the bucket totals, **1 `all_gather`, no all-reduce, no collective-permute, no all-to-all**. The volume is
`O(d^2)`, **independent of `N`**.

## 8. Sharded `uniquify_states` needs two routing rounds, and is exact (`poc/uniquify_sharded.py`)

**Bit-identical to `uniquify_states`** at n=100, N=209,400 with duplicates, `states_size` 262,144 — first
as a numpy specification at d=4 (bucket imbalance 1.15%), then in JAX at **D=2 and D=4**: **157,051 unique
rows, `np.array_equal` True**, including the `[states_size, B]` uint8 layout and its 255 filler.

**The second round is the finding.** The output is a *sharded* `[states_size, B]` array, so shard `s` owns
a contiguous block of rows. Bucket `k`'s unique rows land at a **data-dependent** global offset — the
prefix sum of earlier buckets' unique counts — which does **not** align with output-shard boundaries.
Measured: offsets `[0, 22279, 50782, 96614]` against a shard size of 65,536, so **2 of 4 buckets straddle
a boundary** and a bucket owner must send rows to more than one output shard. Zero crossings would be
luck, not a property. The structure is **route → sort/dedupe → route again**:

1. splitters from a strided sample, compared on the **full row** (§7 gap 1)
2. `all_to_all` #1: rows to their bucket's owner
3. local sort + dedupe — each owner ends up holding one sorted unique run
4. `all_gather` of the `d` unique counts, giving each bucket's global offset
5. `all_to_all` #2: rows from bucket owner to output-shard owner

**Round 2 is what makes the output spec honest.** After it, shard `s` holds exactly output rows
`[s·SS/d, (s+1)·SS/d)`, so `out_specs=P('x', None)` is **true**. Two dead ends, both of which looked
right:

- *Private per-shard blocks cannot be declared replicated.* Having every shard scatter into its own
  `[d, cap]` block with `out_specs=P(None, None, None)` is rejected: *"implies that the corresponding
  output value is replicated across mesh axis 'x', but could not infer replication over any axes."*
  **`check_vma` was correct and the spec was the lie** — those blocks are per-shard *partial* results,
  needing a cross-shard reduction to become replicated. `check_vma=False` would have produced a silently
  wrong answer. Routing to a single owner per bucket removes the reduction, which is why step 2 exists.
- *A value derived from an explicitly-sharded array cannot be closed over.* `NotImplementedError: Closing
  over inputs to shard_map where the input is sharded on 'Explicit' axes is not implemented.` The
  splitters are computed outside `shard_map` from the sharded `words`, so they must be **passed as an
  argument** with `P(None, None)`, not captured. The error names the workaround.

**Collectives, D=4:** 1 `all-gather`, 24 `all-to-all`, 4 `all-reduce`, 0 `collective-permute`. Two of the
`all-reduce`s are the caller's own `.sum()` over `P('x')` results. The rest is `u64[256,2]`: the
**splitter sample** — `words.at[idx].get(out_sharding=P(None, None))` gathers a replicated sample from a
partitioned array, and XLA implements that replication as an all-reduce. It is 4 KB and `O(nsample)`,
**independent of `N`** — so splitter selection is *cheap*, not *free* as first claimed.

**Two capacity defects.**

- *The guard fired on correct input.* The first working version reported **overflow 763,677 beside a
  bit-exact result**, because round 2 funnels every dead row to one bucket, which overflows by design and
  is then dropped harmlessly. Fixed by masking the count to **live** elements. **A guard that fires on
  correct input is worse than none** — it trains a caller to ignore the one signal that matters, which is
  how `poc/range_partition`'s `cap` bug shipped.
- *Balls-in-bins is the wrong model for a range partition.* §6's `mu + sqrt(2·mu·ln d)` is correct **for
  a hash**, where each element picks its destination independently at random. Under a range partition the
  destination is the element's *value* bucket, and bucket sizes are set by the data. Measured: buckets
  `[44557, 57006, 47400, 60437]`, so a shard sends `60437/d ≈ 15,109` rows to the largest bucket's owner,
  not the `N/d/d = 13,088` the bound predicts; a 1.2× slack over that mean still overflowed by 42,712.
  `poc/range_partition`'s rule was already right — slack must exceed the splitter imbalance — and 1.35 is
  exact here at both D. Round 2 needs a different baseline again: a bucket's unique rows land
  **contiguously**, reaching only **1–2 output shards**, so its worst per-destination is ~the whole bucket
  — size off `ss/d`, not `ss/d/d`, which understates it by a factor of `d`.

**The guard is verified in both directions.** Undersizing either round makes overflow nonzero *and*
`exact` False; at 1.35 both are clean. Exactness and a zero count coincide, so the free check is
sufficient — and a caller must **raise**, not clamp, in both rounds.

## 9. Widening the device sweep found a latent `all_to_all` bug

Moving `poc/hash_partition_jax`/`poc/uniquify_sharded` to real GPUs replaced their hardcoded
`XLA_FLAGS=...count=4` with `poc/gpu_unverified`'s `--devices` convention (argparse **before** `import
jax`, since `CUDA_VISIBLE_DEVICES` and `XLA_FLAGS` are both read at backend initialization) and derived
the shard sweep from `jax.device_count()` rather than pinning `(2, 4)`. **That immediately failed at 8
devices**: `ValueError: The size of all_to_all split_axis (4) has to be divisible by the size of the named
axis x (8)`. `run_case` took `mesh` and `num_shards` as **independent parameters that had to agree**; the
send buffer got `num_shards` rows while the mesh axis had `jax.device_count()` devices, and **a 4-device
box makes them coincide**, which is why every earlier run passed. Fixed by deriving `num_shards =
mesh.shape["x"]` inside `run_case`.

**A hardcoded device count is not a fixture, it is a coincidence.** Sweeping the parameter separated two
quantities that had been silently identical — the same reason `CLAUDE.md` says to sweep `cache_level`,
and the same shape as the three bugs that hid behind its default `(1, 0)`.

`d = 1` needed handling in both scripts, for *different* reasons. `poc/hash_partition_jax` runs it as a
meaningful degenerate case (zero collectives, still exact) but **skips its overflow section**: with one
bucket the derived capacity is ~`N`, so a 0.9x slack is still ample, nothing overflows, and the section's
premise is false rather than its assertion weak. `poc/uniquify_sharded` **raises** at `d = 1`: with one
bucket there is no second routing round, which is the mechanism it exists to verify.

## 10. The popcount diagonal path already shards — nothing to build

Every function on the path shards with **zero collectives** and returns bit-identical values. **It was
never at risk**: `_z_parity` is `sum(bitwise_count(states & z), axis=1) & 1`, two elementwise ops and a
reduction along the **byte** axis. For a `P('x', None)` state array the sharded axis is axis 0, so the
reduction runs within each device's own rows — the elementwise-and-reductions rule in its easiest form,
**the reduction is over the unsharded axis**. Contrast `uniquify_states`, whose `cumsum` reduces *along*
the sharded axis and needed §7's two-level prefix sum.

Measured on 1D XXZ, `P('x', None)` states, 4 devices — asserting the **spec and the values together**,
since a replicated run agrees to exactly 0.0 and "correct but silently unsharded" is invisible to value
comparison alone:

| function | out spec | values | max diff |
| --- | --- | --- | --- |
| `_z_parity` | `P('x',)` | match | 0.00e+00 |
| `get_diag_signs` | `P('x', None)` | match | 0.00e+00 |
| `get_diagonal` | `P('x',)` | match | 0.00e+00 |
| `compute_diagonal` | `P('x',)` | match | 0.00e+00 |

**Swept over every X group and both coefficient dtypes**, at n=40 (J=41, K=40, N=9,220 for real; J=42 for
the complex case, where a single odd-Y string makes `.c` complex128): **0 failures out of 41 and 42
groups** for all three builders, counting a missing `'x'` in the output spec as a failure alongside a
value mismatch. And **`apply_h` end-to-end at both cache levels that use this path** — `(1, 0)`, which
recomputes from `zsignatures`, and `(1, 1)`, which unpacks cached `diag_signs` — output spec `P('x',)` and
`max|diff| = 0.00e+00` on both dtypes.

**One pre-existing behaviour, correctly attributed.** `apply_h` with a **real** `vec` against
**complex128** coefficients raises `TypeError: scan body function carry input and carry output must have
equal types ... float64[N] but ... complex128[N]`. It looked like a sharding failure and is not: it
**reproduces on a single device with no mesh at all**. It is an undocumented dtype contract — `vec` must
be promotable to the coefficient dtype — and `PauliSumXZ` makes `.c` complex whenever any Pauli string has
an odd Y count. Unrelated to this work, and *not* fixed here.

**Pinned by `test/sharded/diagonals.py` and mutation-verified.** Dropping the single
`out_sharding=jax.typeof(states).sharding` on `get_diag_signs`' `init` accumulator makes the builder run
**correctly but unsharded**: `bad_value = 0` — every value still bit-identical — while `bad_spec` goes to
42/44. **A value-only test passes that mutant silently.** Placing the mutation *inside* the scan instead
raises on a carry-type mismatch, so the `init` line is the one that had to be mutated to produce the
silent form.

## 11. What it means

- **The distributed-`states` line has no remaining unverified mechanism.** `get_xsource` (§6),
  `uniquify_states` (§8) and the diagonal path (§10) all have working, exact, sharded forms. It is the only
  lever found that moves the `13 * N` ceiling — 27.9 GB/device becomes `27.9/d`.
- **Two partitions, not one**: whole-key hashing for `get_xsource` (1.03–1.11× on XXZ, invariant to hop,
  `d` and subspace growth; capacity from balls-in-bins in `N/d`), range splitting for `uniquify_states`
  (capacity from the splitter imbalance, slack 1.35). Prefix hashing and range-split `get_xsource` are
  both unusable on XXZ.
- **Only with `cache_level[0] = 1`**: routing is paid once per solve there, ~84 TB per solve at `d=4`
  without it.
- **Every capacity is a correctness parameter with a free, sufficient overflow count: raise, never
  clamp**, and count only live elements.
- **It is a project, not a patch**: a new partitioning contract, a routed `get_xsource`, and
  `uniquify_states` rebuilt on the two-round shuffle.

## 12. Open

1. **Whether the routing pays** — the whole question now. `get_xsource`'s one routed round per X group,
   and `uniquify_states`' two rounds with 24 `all-to-all` at D=4 (materially more communication than the
   single round this line originally assumed), against the 27.9 GB/device they save. Needs a real
   interconnect; both JAX scripts have a timing section that runs only off the CPU backend.
2. **`ragged_all_to_all`** — unrunnable on XLA:CPU, so unmeasured; the padded path's 26% FESOM2-JAX
   penalty is the only figure.
3. **In-graph setup for `get_xsource`**: owner assignment plus per-shard sort is host-side numpy in
   `poc/hash_partition_jax`; in the library it would be `poc/range_partition`'s phase 3.
4. **Library integration**: the prototypes live outside `rqutils/`; `states_size` and both capacities are
   `static_argnums`, and `N` must divide `d` (`sqd` already rounds `states_size` to a multiple of
   `mesh.size`, so the machinery exists). `uniquify_states`' word packing remains *"the wrong trade for an
   out-of-core design"* at the `2^31` ceiling (6–15 GB of extra buffer), unaffected by anything here.

## 13. The scripts

None has subcommands; each runs its sections top to bottom and asserts exactness.

- **`hash_partition.py`** — no arguments (n=60 fixed). Numpy: prefix vs whole-key imbalance (§3),
  exactness on three fixtures, imbalance at d=4/64/1024, XXZ prefixes at n=30/60, and all 60 hops at
  d=64 (§4).
- **`hash_partition_jax.py`** — `--devices` (GPU ids), `--num-qubits` (default 30), `--host-devices`
  (virtual CPU devices without `--devices`, default 4). JAX `get_xsource` over the shard sweep and three
  hops, the overflow guard at slack 1.6/0.9, and a timing section run only off the CPU backend (§6).
- **`range_partition.py`** — no arguments; run under `XLA_FLAGS=--xla_force_host_platform_device_count=4`
  (the mesh is fixed at 4). The single-round shuffle against `uniquify_states` at three sizes (§7).
- **`uniquify_sharded.py`** — `--devices` (GPU ids, or `mpi` for one GPU per MPI rank), `--num-qubits`
  (default 100), `--host-devices` (default 4). Two-round `uniquify_states` over the shard sweep, the
  guard at slack 1.35/0.30, and a timing section run only off the CPU backend (§8).

The JAX scripts sweep the powers of two dividing `jax.device_count()` (`uniquify_sharded` from 2, and it
exits below 2 devices). `--devices mpi` (`uniquify_sharded` only) needs `--extra mpi` under `mpirun`, and
is the only mode that exercises the multi-process gather. Not reproduced by any script: §1's ingredient
probes and range prototype, §2's traffic table, §3's range-split column, d=16/256 columns and Zipf
fixture, §4's n=100 prefixes, d-sweep, range-split and stale-splitter measurements, §5's padding table,
§6's one-hot memory, §7's gap probes and argsort/one-hot figures, and §8's numpy specification.
