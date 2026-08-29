# Response: the batched gather is declined on evidence, and two scalable wins landed elsewhere

Reply to `docs/rqutils-multiobs-request.md`, from the `rqutils` side. Branch `dev`, version still
`0.2.0` (unreleased).

**All three of your asks are declined, with measurements.** But the investigation turned up two real
speedups elsewhere in the same pipeline, and **both have shipped**: byte-wise comparison was costing
`O(B)` per operation in two places, and both now work on packed `uint64` words.

- **`get_xsource`'s wide path** (`B > 8`, i.e. n > 60): **3.6-7.7x** at n=64-200 (§5.1).
- **`uniquify_states`' lexsort**: **1.79-4.89x** from n=30 to n=200 (§5.2) — this one **helps you
  today**, at n=30, because the sort key count drops at every width.

Together, on the setup path this module's docstring measures at 66-97% of a solve: **1.43x at n=30**,
4.70x at n=100. Output is identical at every width and no API moved, so there is **nothing for you to
change** — but see §5.2 for the limit we did *not* remove, which is the one that matters if you are
heading to n=100 with large `N`.

| # | Ask | Outcome |
| --- | --- | --- |
| 1 | An entry point taking a **stack** of X-signatures | **Declined — it already exists.** `_apply_h_kernel` is a `lax.scan` over stacked signatures. Measured that form directly: **~1.0x**. |
| 2 | Whether the **weight-2 case admits a closed form** | **Declined — the premise is false.** `S ^ X` is *not* a local permutation. Measured below. |
| 3 | Whether the gather can be **shared rather than batched** | **Answered, and it reframes the problem.** The gather is latency-bound on a `log2(N)` dependent-load chain, not miss-bound across operators. Nothing shareable. |
| — | (not asked) | **Landed on `dev`:** `get_xsource`'s wide path compares `uint64` **words** instead of bytes — **3.6-7.7x** at n=64-200, removing an ~8x cliff at the `B > 8` boundary. See §5.1. |
| — | (not asked) | **Landed on `dev`:** `uniquify_states` lexsorts on words too — **1.79-4.89x**, n=30 included, and it was the *larger* of the two costs (14-27x `get_xsource`). See §5.2. |
| — | (not asked) | **Not pursued, `n <= 32` only:** a rank-select index measures **22–58x**, emits no sort, and shards — but is Hilbert-space-indexed and dies at n~34. See §5.3. |

**Thank you for the `vmap` measurement and for the framing of the status block.** "Treat the *what we
would like* section as a question, not a specification" is what made this productive: the specification
would have been implemented and would have delivered nothing. Your own suspicion — that the easy version
failing is a reason to distrust the hard version — was correct, and §3 explains why in a way that also
kills two ideas you did not propose.

Every number below is measured **in this tree**, single laptop CPU, `jax_enable_x64` on, at n=30 with
4-byte packed states. **Nothing is measured on GPU and nothing is measured through `apply_h` or your
`_expval_kernel`** — see §6 before acting on any of it.

---

## 1. The batched entry point exists, and it does not help

`sqd.py:1486` already expresses all six `cache_level` strategies as one `jax.lax.scan` over the X
groups. Your ask — "accepts a stack of X-signatures and resolves all of their destination index arrays
in one traversal" — is that scan with a different leading axis. There is no new capability to add, so we
measured the thing itself rather than building a wrapper for it.

Baseline is one `get_xsource` call per signature (what `spinchain` does today). `nsig=16` weight-2
signatures. **All three forms are bit-identical to the baseline** — only speed differs:

| form | 250k | 1M | 4M (3 seeds) |
| --- | --- | --- | --- |
| `jax.vmap` over signatures | 1.32x | 1.09x | 0.84–1.10x (median 0.87x) |
| `jax.lax.scan` over stacked signatures | 0.91x | 0.96x | 0.75–0.97x (median 0.83x) |

Run-to-run spread on this machine is wide: we saw the same 4M scan fixture read 1.25x once and 0.75x
another time. **The defensible claim is that both sit at ~1.0x with no trend toward a win as N grows.**
Your `vmap` numbers (2.5x at 141k, 1.5x at 1M, 1.04x at 6M) say the same thing from the other side, and
we reproduce the direction.

The reason is §3. A scan iteration performs its own full `searchsorted`; nothing stays resident between
iterations, so there were never shared misses for batching to amortize.

## 2. The weight-2 closed form does not exist — the premise is wrong

You wrote that because every spin-current operator is 2-local, `S ^ X` flips two bits and is therefore
"a *local* permutation of the state list", so a low-weight specialization might beat any batching. We
tested it on a dense fixture — all `2^20` states present, so the hit rate is 100% and `A[i] = i ^ X`
holds exactly:

| bond (bit pair) | Pauli weight | median `|A[i] - i|` | as fraction of N |
| --- | --- | --- | --- |
| (0, 1) | 2 | 2 | 0.0% |
| (4, 5) | 2 | 32 | 0.0% |
| (9, 10) | 2 | 1,024 | 0.1% |
| (14, 15) | 2 | 32,768 | 3.1% |
| (18, 19) | 2 | 524,288 | **50.0%** |

Every row is a nearest-neighbour, weight-2 bond — identical Pauli weight. Displacement is the **XOR
distance** `2^b2 ± 2^b1`, set by **bit position, not weight**. A spin-current sweep covers all `n-1`
bonds, so roughly half its operators gather across the whole array however local the operator is
physically. There is no low-weight structure for a specialization to exploit.

One structural fact we found while testing this, recorded because it is genuinely interesting and
genuinely useless: XOR by `X` splits the sorted key array into **already-sorted runs**, and the run
count tracks `2^(nq-1-high_bit)` — just **4 runs** at 4M for a bond on bits (28,29), against ~10^6 for
bits (7,8). So high bits displace far but preserve order; low bits displace little but shuffle locally.
That suggests replacing the search with a merge. It does not work — see §3.

## 3. The gather is latency-bound, which is why nothing above works

This is the part that reframes your request. First, where the time actually goes at N=4M:

| component | time | share |
| --- | --- | --- |
| `_pack_state_keys(states)` | 1.0 ms | 1.0% |
| `xor S^X` | 0.4 ms | 0.4% |
| **`searchsorted` + hit test** | **101.6 ms** | **95.6%** |

So caching the packed keys across your 87 operators — which looks like free redundancy, and which we
expected to be part of your "70 vector-passes" — saves **1%**. The streaming ops run at bandwidth
(0.4 ms for 4M elements); the search does not.

Then why the search is slow:

| N | ns/target | log2 N | **ns/level** |
| --- | --- | --- | --- |
| 250k | 14.7 | 17.9 | 0.82 |
| 1M | 21.1 | 19.9 | 1.06 |
| 4M | 22.7 | 21.9 | 1.03 |
| 16M | 26.9 | 23.9 | 1.12 |

**`ns/level` is constant across a 64x range of N.** The cost is a **dependent-load chain of `log2(N)`
levels** — each probe's address depends on the previous probe's result, so the loads serialize and no
amount of parallel work in flight hides them. Your "~70 vector-passes" figure is this: ~22 serialized
loads at ~1 ns each against a streaming pass at bandwidth.

That single fact explains every negative result, including two ideas you did not propose:

- **Batching (`vmap`, `scan`) — ~1.0x.** Batching does not shorten a dependency chain.
- **Merge instead of search — 0.88–1.04x.** We pre-sorted the targets so access order is sequential
  (using the run structure from §2). Same work, sequential instead of random: **no win**. Access order
  is not the constraint.
- **Bucketing on the top-k bits to cut levels — 0.31–0.51x.** Cut the chain from 22 levels to 8 and it
  ran **2–3x slower**: the per-level vectorized bookkeeping costs more than the levels saved. Level
  count was not the binding constraint; `searchsorted`'s tight inner loop was.

So there is nothing to share across operators, and the answer to your question "whether the gather can
be shared rather than merely batched" is **no** — not because the redundancy is hard to exploit, but
because the cost is per-target latency that every operator pays independently.

**We are therefore not pursuing a blocked/tiled gather either.** A loop interchange (block `S` outer,
signatures inner) targets cache residency, and the merge result above shows residency is not what is
being paid.

## 4. Two corrections to the report's numbers

Neither changes your conclusion; both would cost a reader time.

- **`230 ms/op x 87 = 20.0 s`, but your table says 23.5 s at dim 6M.** These reconcile: you note the 87
  operators fall into two shape classes (58 + 29), and the table's implied mean is **270 ms/op**. A
  230/350 split across the classes gives 23.5 s exactly. Please say which class the 230 ms figure is,
  or the arithmetic reads as a 17% contradiction.
- **Quote the 12M row, not the 24M one.** Your share column is correct against your own profile at every
  point (we checked all five). But the 24M share of 14% is *lower* than 12M's 21% only because that
  run's total is inflated to 754 s by something unrelated to this kernel, while per-state cost is still
  **rising** (0.048 -> 0.051 us). **21% at 12M is the representative figure.** Also the stated range
  `0.042–0.050` clips its own top: the five points span 0.0421–0.0505, so it should read
  **0.042–0.051**.

The `jnp.vdot` comparison is right as written (0.089 GiB / 3.3 ms = 27.1 GiB/s for one complex128
array), but "over the same array" is worth making explicit — a two-operand `vdot` would read 54 GiB/s
and someone rebuilding the check could land on either. It does not affect the ~70x conclusion.

## 5. Counter-proposals

Three. **§5.1 and §5.2 are implemented on `dev`** and scale to any qubit count; §5.3 is capped at
`n <= 32`, is not implemented, and is offered only because it is large where it applies. If you have a
long-term target past ~32 qubits, read §5.1 and §5.2 and treat §5.3 as a footnote. §5.4 records what the
literature suggested and measurement rejected.

### 5.1 Compare `uint64` words, not bytes — the scalable one

**This addresses a different bottleneck than your report describes, and at your stated qubit counts it
is the one that matters.** `get_xsource` has two paths, chosen statically on packed width: `B <= 8`
packs each row into a `uint64` key and uses `jnp.searchsorted`; `B > 8` falls back to an explicit
row-wise lexicographic binary search. Your n=30 case (`B = 4`) is on the fast path. The fast path
survives to **n = 60** (`B = 8`). Above that the fallback takes over, and it is far more expensive per
state:

| n | B | path | ns/state |
| --- | --- | --- | --- |
| 30 | 4 | `uint64` | 22 |
| 60 | 8 | `uint64` | 15 |
| 64 | 9 | **lexicographic** | **122** |
| 100 | 13 | **lexicographic** | **189** |

An **~8x cliff** at the boundary. The cause is that the fallback compares rows one **byte** at a time,
so a level of the binary search costs `O(B)` byte comparisons — 13 of them at n=100.

The fix is to compare `uint64` **words**: 13 bytes is 2 words, so a level costs 2 comparisons instead
of 13. Cost then scales as `ceil((n+1)/64)` words, i.e. **logarithmically in packed width**, with no
`2^n` term.

**This is implemented and measured in-tree, not projected.** The figures below are an A/B of the patched
`get_xsource` against the pre-change module loaded side by side in one process, N = 300k, all outputs
**bit-identical at every width**:

| n | B | words | before | after | speedup |
| --- | --- | --- | --- | --- | --- |
| 30 | 4 | 1 | 3.9 ms | 4.0 ms | 0.97x |
| 60 | 8 | 1 | 4.1 ms | 4.3 ms | 0.95x |
| 63 | 8 | 1 | 4.3 ms | 4.2 ms | 1.01x |
| 64 | 9 | 2 | 36.8 ms | 9.7 ms | **3.81x** |
| 80 | 11 | 2 | 47.0 ms | 9.6 ms | **4.88x** |
| 100 | 13 | 2 | 56.7 ms | 9.7 ms | **5.81x** |
| 127 | 16 | 2 | 75.8 ms | 9.8 ms | **7.71x** |
| 200 | 26 | 4 | 126.9 ms | 17.6 ms | **7.22x** |

Read the range as **~3.6-7.7x across n=64-200**: a repeat of the same A/B on this machine read
3.56x / 4.40x / 5.41x / 6.48x / 7.08x, so the per-row values move by ~15% run to run while the trend
does not. The `B <= 8` rows are the control -- that path is not touched, and it reads 0.95-1.06x.

The speedup **grows with n**, which is the point: the "after" column is flat at ~9.7 ms from n=64 to
n=127 while "before" grows 37 -> 76 ms, because everything from `B = 9` to `B = 16` is two words and
costs the same. The next step up is at `B = 17` (n >= 128), where a third word is needed.

We stress-tested the obvious objection — that a *real* subspace shares long prefixes, so a leading-word
discriminator would be less effective than on random rows. It is, and the technique survives it:

| n=100 fixture | first word unique | current | multi-word | speedup |
| --- | --- | --- | --- | --- |
| uniform random rows | 100.0% | 72.5 ms | 9.8 ms | **7.37x** |
| Hamming weight 50 + 1-hop closure | **37.0%** | 76.6 ms | 13.2 ms | **5.80x** |

Prefix sharing cuts the leading word's discriminating power from 100% to 37%, and the speedup only
falls from 7.4x to 5.8x — the second word resolves the remainder, and 2 word-comparisons still beat 13
byte-comparisons.

This is the one idea in this document with no exponential term, and it is the smallest change: it
replaces the comparison inside an existing loop and touches neither the API nor the `B <= 8` path.
Note the `B <= 8` boundary is a **correctness** limit for the single-`uint64` key (a wider row would
alias); a multi-word key has no such limit, so this generalizes the fast path rather than adding a
third one.

**Status: landed on `dev`.** `get_xsource` now packs wide rows with `_pack_state_words` and compares
with `_word_less_than`; the byte-wise `_row_less_than` is deleted, having had no other caller. The full
suite (614 tests), `ruff`, `ty`, and both sharded subprocess harnesses are clean, and the sharded runs
agree to the last digit with partition specs preserved. **Nothing for you to change** -- the signature,
the return contract and the `-1` absent marker are all unchanged, and your n=30 path is untouched.

Two notes from implementing it, both recorded because they are the kind of thing that bites later:

- **It exposed a coverage gap.** Reversing the word loop to LSW-first is a genuine permutation defect,
  and it left all 12 cases of the existing byte-width reference test green: that fixture concentrates
  variation in the *trailing* bytes, so the leading word rarely decides a comparison. A new test
  (`test_wide_rows_compare_most_significant_word_first`) forces byte 0 and the tail to disagree on
  order, where MSW-first and LSW-first differ, and is verified by mutation to fail against the reversed
  loop.
- **One invariant we nearly shipped is false.** A draft docstring claimed trailing-padding would
  reorder rows. It does not -- appending a constant number of zero bytes is a left-shift by `8 * pad`,
  and a constant left-shift is monotonic (verified exhaustively at `B = 3` and over 20000 random pairs
  at `B = 9`). The padding end is a *compatibility* choice, so that `nwords == 1` reproduces
  `_pack_state_keys` bit for bit. Do not read it as load-bearing.

### 5.2 The same fix in `uniquify_states`, which turned out to be the bigger cost

**We profiled the rest of the pipeline before stopping, and §5.1 had optimized the smaller of the two
costs.** At n=100, N=200k:

| component | time |
| --- | --- |
| **`uniquify_states`** | **143 ms** |
| `get_xsource` (one signature) | 6.6 ms |
| `get_diagonal` | 0.5 ms |
| `get_diag_signs` | 0.4 ms |

`uniquify_states` is **14-27x `get_xsource`** at every width from n=30 to n=127, and unlike
`get_xsource` it had no fast path at all — cost grew linearly in `B` throughout, 55 ms at n=30 to
186 ms at n=127.

Same root cause, in the more expensive function. Its lexsort was
`jax.lax.sort((*states.T, iota), num_keys=B)` — all `B` uint8 columns as separate key operands, and XLA
compares key operands one at a time. Keying on `ceil(B/8)` packed words instead cuts 13 keys to 2 at
n=100. The permutation is unchanged because the packing is order-preserving, and the uniqueness check
moves to the words for the same reason (the packing is injective: the pad bytes are constant).

A/B against the pre-change module, N = 220k **including duplicates**, output identical at every width:

| n | B | keys before | keys after | before | after | speedup |
| --- | --- | --- | --- | --- | --- | --- |
| 30 | 4 | 4 | 1 | 62.9 ms | 35.1 ms | **1.79x** |
| 60 | 8 | 8 | 1 | 89.7 ms | 35.5 ms | **2.53x** |
| 64 | 9 | 9 | 2 | 102.6 ms | 42.4 ms | **2.42x** |
| 100 | 13 | 13 | 2 | 158.4 ms | 43.2 ms | **3.67x** |
| 127 | 16 | 16 | 2 | 204.6 ms | 43.7 ms | **4.68x** |
| 200 | 26 | 26 | 4 | 286.1 ms | 58.5 ms | **4.89x** |

**This one helps you today.** Unlike §5.1, which only touches `B > 8`, the key count drops at every
width — including n=30, where it is worth **1.79x**.

Combined with §5.1 on the setup path the module docstring measures at **66-97% of a solve**
(`uniquify_states` plus a J-fold `get_xsource` precompute, J=8, N=200k, output identical):

| n | before | after | speedup |
| --- | --- | --- | --- |
| 30 | 80.4 ms | 56.1 ms | **1.43x** |
| 64 | 273.5 ms | 88.0 ms | **3.11x** |
| 100 | 418.5 ms | 89.1 ms | **4.70x** |
| 127 | 554.5 ms | 92.2 ms | **6.01x** |

**Status: landed on `dev`.** Same as §5.1 — signature, return contract and filler convention all
unchanged, nothing for you to change.

It exposed a third coverage gap, and this one is the most dangerous of the three. Keying the sort on
only the first word (`num_keys=1`) returns an array that is the right shape and contains every unique
row, but is **not lex-sorted** — which silently violates the precondition `get_xsource`'s binary search
depends on. **All 189 existing tests pass against that mutant**, because a group must share an entire
8-byte word before it can be reordered and most fixtures vary the leading bytes. A new test
(`test_wide_rows_sort_on_every_word_not_just_the_first`) asserts sortedness directly on a fixture where
four rows share byte 0, and is verified by mutation.

**What we did *not* fix, and it is the real n=100 wall.** That `lax.sort` is still there, still the
single largest cost in the pipeline, and still the reason two structural limits exist: it is what caps
`N` at `2^31` (the sort must run on one device) and it is why `uniquify_states` does not shard, where
`get_xsource` does. We made it ~4x cheaper; we did not make it scale. If your target is n=100 *with large
N*, the question that matters is replacing it with a sharded or out-of-core uniquification — sketched in
`examples/scaling/poc9_ooc_uniquify.py` — which is a much larger change than either of these two and one
we have not attempted.

### 5.3 A rank-select index — large, but capped at `n <= 32`

**Only read this if n <= 32 is a regime you care about.** It is the largest speedup we measured and it
is architecturally the cleanest, but it is fundamentally not scalable, so we lead with §5.1.

A bitmap over all `2^n` basis states plus a per-word cumulative popcount returns the *position* of a
state in **two loads and one `popcount`** — there is no search at all. Measured against the current
`uint64` path, 16 operators per fixture, N ~ 4M, all outputs **bit-identical**:

| fixture | hit rate | baseline | rank-select | speedup |
| --- | --- | --- | --- | --- |
| uniform random | 0.37% | 1607 ms | 71.7 ms | **22.4x** |
| clustered (1-hop closed) | 6.77% | 1556 ms | 70.1 ms | **22.2x** |
| dense low block | 36.8% | 1581 ms | 27.4 ms | **57.7x** |

Note it *improves* with hit rate, unlike a pre-filter: cost is independent of whether the state exists.
Properties that make it attractive where it fits:

- **It emits no sort.** The lowered HLO contains **0** `sort` ops against the baseline's **117**. It
  therefore does not reintroduce what `23fb226` removed, and the `2^31` ceiling's structural cause is
  absent from it.
- **It shards.** On a 4-device mesh with partitioned targets the output stays `P('x')` with **zero**
  `all-gather`/`all-reduce`, using this library's existing `.at[...].get(out_sharding=...)` convention.
  (Without `out_sharding` it raises `ShardingTypeError`, as the gather's sharding cannot be inferred.)
- **Build is cheaper than one search** — 0.37–0.91x the cost of a single `searchsorted`, paid once and
  amortized over all 87 operators.
- It makes the **lex-sortedness precondition irrelevant**, since nothing is searched.

**Why it is capped.** Memory is `2^n / 4` bytes — it indexes the *Hilbert space*, not the subspace:

| n | 30 | 32 | 34 | 36 | 40 | 100 |
| --- | --- | --- | --- | --- | --- | --- |
| memory | 0.27 GB | 1.07 GB | 4.29 GB | 17.2 GB | 275 GB | 3e29 bytes |

So it is viable to n=32, borderline at n=34, and impossible beyond. The same wall applies to the two
weaker variants we measured on the way — a membership bitmap pre-filter (`2^n / 8` bytes, 2.66–4.09x by
skipping the search for the >99% of targets that cannot hit) and a direct index array (`2^n * 4` bytes,
69x). All three are Hilbert-space-indexed and all three die at n ~ 34.

If it were ever wanted, it would have to be an `n`-gated strategy alongside the existing paths, not a
replacement — a new required input (`num_qubits`) and a precomputed structure threaded through
`cache_level`. The pre-filter variant additionally has a **silent-wrong-answer** hazard we would not
ship: its compaction needs a static `cap`, an undersized `cap` drops hits with no error (883 returned
against 884 true candidates), and there is no provable bound below `N` — a subspace closed under a hop
has a 100% hit rate for that hop. `cap = N` is safe and recovers the baseline exactly.

### 5.4 What we tried from the literature and rejected

Recorded so none of it is re-attempted. Every row measured in this tree.

| approach | source | result |
| --- | --- | --- |
| Interpolation search (one linear model) | Interpolation-search literature; `O(log log N)` on uniform keys | **0.91–0.99x.** Sound in principle — subspace keys are a near-uniform sample, so one linear model predicts position to within ~1244 rows at N=4M, cutting 22 levels to 14. But a hand-rolled `lax.scan` loop gives back more than the levels save. |
| Level reduction generally | — | **Ceiling is 1.77x.** `searchsorted` into 4096 elements vs 4M is only 1.77x faster, so cutting levels cannot pay much; the small array simply fits in cache. |
| Top-k-bit bucketing | Radix/learned-index style | **0.31–0.51x.** Cut the chain from 22 levels to 8 and ran **2–3x slower**. |
| Batched sort of all targets | Selected-CI literature ("sorting-based paradigm", residue arrays) | **0.41x.** The `argsort` of 16N keys alone (429 ms) costs more than all 16 independent searches (296 ms). |
| Hash table / tree replacement | — | Not measured here, and the selected-CI literature reports it directly: search trees and hash tables were *"found for the most part to be not competitive in any of our tests"* against cache-efficient sorted-array methods. |

The selected-CI comparison is worth stating explicitly, because that field has worked at 100+ orbitals
for decades and its problem looks like this one. Its **residue-array / dynamic-bit-masking** technique
does not transfer: it exists to *discover* which determinant pairs are connected when the excitation is
unknown, and solves that in `O(N log N)` by masking and sorting. `get_xsource` is **given** `X`, so
there is nothing to discover and a direct search is already the right shape. That asymmetry is also why
their sorting-based paradigm loses here, and why `23fb226`'s removal of the old sort was correct.

## 6. What we are not claiming

- **No GPU measurement.** All of the above is one laptop CPU. The gather is better optimized on GPU and
  `23fb226`'s own CPU-to-GH200 ratio compressed from 12–25x to 5.15x, so the §5 figures should be
  assumed optimistic until measured there. This is the largest open risk. It applies to §5.1 and §5.2
  as shipped: both are strict reductions in comparison count, so we expect the direction to hold on GPU,
  but the magnitude is unmeasured — and a GPU sort is well optimized relative to its gather, so §5.2's
  ratio in particular may compress.
- **The n>60 fixtures are synthetic.** The n=64–200 rows are generated, not sampled from a real circuit
  at those sizes. For §5.1 we did test the adversarial structure (weight-50 plus 1-hop closure, which
  drops the leading word's discriminating power to 37%) and the win held at 5.80x. §5.2's A/B used random
  rows plus injected duplicates; a real subspace's duplicate rate and prefix structure differ, and the
  sort's cost depends on both.
- **The n>60 path is still thinly tested.** §5.1 has landed and the existing byte-width reference test
  covers n=55/63/64/71/80 against an independent lookup table, but **no test runs a subspace anywhere
  near n=100** — `TestInt32Ceiling` bounds `N`, not `n`. If you are heading there, that gap is worth
  closing on your side too: a wrong answer on this path is a *permutation* of the right one, so it stays
  symmetric and finite and will not announce itself.
- **Nothing measured through `apply_h`, `_expval_kernel`, or a full `sqd()` solve.** §1–§5 exercise
  `get_xsource` and `uniquify_states` directly. The combined 1.43x–6.01x in §5.2 covers the *setup* path
  only; the module docstring puts that at 66–97% of a solve, but we did not measure a solve to confirm
  the composition, and your 105 s at dim 24M is an end-to-end figure we have not decomposed.
- **§5.3 is not implemented and we are not proposing to.** It is a candidate with a blocking design
  problem (the `cap` bound) on top of its `n <= 32` ceiling. §5.1 and §5.2 have landed.
- **The `lax.sort` scaling wall is untouched.** §5.2 made it cheaper, not scalable. It is still
  single-device, still what caps `N` at `2^31`, and still the largest single cost in the pipeline. We
  have not attempted a sharded or out-of-core replacement.
- **We did not verify your `spinchain`-side numbers** — the ~47% packing hoist, the 4.6e-19 agreement,
  or the replay timings. They are consistent with everything checkable here.

## 7. What we suggest you do

**Take the opt-out.** Your status block calls this the lowest-priority of three requests because
`spinchain` can make the contraction opt-out, and that is the right call for now: the batched gather is
not going to be built, and the pre-filter is not ready. `[solver] js = false` is a config key you would
rather not add, but it is available today and this is not.

**One thing worth reconsidering.** Your report frames the priority as low partly because the cost is a
fifth of a large replay. If you later establish that raising the subspace cap is spent as an accuracy
lever — and that better *selection* of subspace strings is the remaining line — then subspace states get
more valuable per dimension and multi-observable evaluation gets more central, not less. That would
change this request's priority without changing anything in it. Tell us if that happens.

**Tell us your target qubit count, and your target `N`.** §5.1 and §5.2 have landed, so nothing is
blocked on this, but the answer decides where we look next and whether §5.3 is worth anyone's time.
§5.3 is 22-58x and dies at n~34; §5.1 and §5.2 have no ceiling in `n`. Two things your report cannot
show us:

- The cliff §5.1 removes (~8x at the `B > 8` boundary, i.e. n > 60) is invisible at n=30 — every number
  in your report is on the fast path. If your roadmap crosses 60 qubits, the profile you sent is not the
  profile you will have.
- `N` matters more than `n` for what is left. The `lax.sort` in `uniquify_states` is now ~4x cheaper but
  still the largest single cost, still single-device, and still what caps `N` at `2^31`. If you need
  n=100 *and* large `N`, that is the next thing to attack and it is a much bigger change (§5.2).

## 8. Reproducing

Everything in §1–§5 needs **only `rqutils` and 64-bit jax** — no `spinchain`, no samples file. Prefix any
repro with `jax.config.update("jax_enable_x64", True)`.

- **loop / scan / vmap (§1):** lex-sorted unique `[N, 4]` uint8 states at n=30, `[16, 4]` weight-2
  signature stack. Compare `jnp.stack([get_xsource(sigs[k], states) ...])` against
  `jax.lax.scan(lambda _, x: (None, get_xsource(x, states)), None, sigs)[1]` and
  `jax.vmap(get_xsource, in_axes=(0, None), out_axes=0)(sigs, states)`. **`out_axes=0` is required** —
  without it `vmap` returns `[N, nsig]` and a shape-blind comparison reads as a numerical disagreement.
  `jit` all three, `min` of >=5 warm trials, repeat across seeds: single runs are not stable to better
  than ~1.5x here.
- **locality (§2):** `states = np.arange(2**20)` packed to 4 bytes, so every source exists. Check
  `get_xsource` returns exactly `i ^ X`, then histogram `|A[i] - i|` for weight-2 signatures at
  increasing bit positions.
- **breakdown and ns/level (§3):** time `_pack_state_keys`, the XOR, and a `searchsorted` with both key
  arrays precomputed, against whole `get_xsource`. Then sweep N over 250k–16M and divide ns/target by
  `log2(N)`.
- **multi-word compare (§5.1):** this is in the tree, so the repro is an A/B rather than a build. Load
  the pre-change `sqd.py` as a second module (`importlib.util.spec_from_file_location`) so both live in
  one process — a copied *repo* does not work, since the venv holds an editable install pointing at the
  original. Build `[N, B]` uint8 rows at the target `n` (`B = ceil((n+1)/8)`, pad bit clear,
  `np.unique(rows, axis=0)` for lex-sorted uniqueness), `jit` both `get_xsource`s, and compare outputs
  for equality before timing. Take `min` of >=3 warm trials and sweep `n` across the `B = 8`/`B = 9`
  boundary — the `B <= 8` rows are the control and should read ~1.0x. For the adversarial fixture, draw
  fixed-Hamming-weight integers and close them under one bit flip.
- **`uniquify_states` A/B (§5.2):** same two-module setup as above. Include duplicates in the input
  (`np.concatenate([rows, rows[:N//10]])`) so the uniqueness path is exercised, and pass the arrays as
  *arguments* to the `jit`ted function -- closing over them lets XLA constant-fold the sort at compile
  time, which reads as a 1.00x speedup and a `slow_operation_alarm` in the log rather than as an error.
  Compare full output arrays, not just timings.
- **rank-select (§5.3):** bitmap of `2^n` bits in `uint64` words via
  `np.bitwise_or.at(bm, keys >> 6, np.uint64(1) << (keys & 63))`, plus an exclusive prefix sum of
  `np.bitwise_count(bm)`. Lookup is `pc[w] + popcount(word & ((1 << b) - 1))` gated on the bit being
  set. Use `.at[w].get(out_sharding=jax.typeof(tk).sharding)` for both gathers or it raises under a
  mesh. The pre-filter variant additionally needs `jnp.nonzero(present, size=cap, fill_value=0)`;
  verify its truncation failure explicitly by passing `cap = true_count - 1`.
