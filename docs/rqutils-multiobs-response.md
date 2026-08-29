# Response: the batched gather is declined on evidence, and the cost is not what the report thinks

Reply to `docs/rqutils-multiobs-request.md`, from the `rqutils` side. Branch `dev`, version still
`0.2.0` (unreleased). **No code has changed.** This is a findings document: three of the four things
you asked about are answered with measurements, and there is a counter-proposal that is measured but
not committed.

| # | Ask | Outcome |
| --- | --- | --- |
| 1 | An entry point taking a **stack** of X-signatures | **Declined — it already exists.** `_apply_h_kernel` is a `lax.scan` over stacked signatures. Measured that form directly: **~1.0x**. |
| 2 | Whether the **weight-2 case admits a closed form** | **Declined — the premise is false.** `S ^ X` is *not* a local permutation. Measured below. |
| 3 | Whether the gather can be **shared rather than batched** | **Answered, and it reframes the problem.** The gather is latency-bound on a `log2(N)` dependent-load chain, not miss-bound across operators. Nothing shareable. |
| — | (not asked) | **Counter-proposal:** a membership pre-filter measures **2.66–4.09x** in JAX under `jit`, sharded and exact. It has one blocking design problem. See §5. |

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

## 5. Counter-proposal, measured but not committed: a membership pre-filter

The one thing that does work follows directly from §3. If the cost is `log2(N)` serialized loads per
target, the win is not making the search faster — it is **not running it**. On a realistic SQD subspace
almost every search finds nothing:

| N | subspace density | hit rate |
| --- | --- | --- |
| 1M | 0.09% of `2^30` | 0.09% |
| 4M | 0.37% of `2^30` | 0.37% |

Over 99% of targets pay 22 serialized loads to discover that `S[i] ^ X` is absent. A bitmap indexed by
state value answers that in **one** load. The bitmap is built once from `states` and reused across every
operator, so its cost amortizes exactly the way you hoped the gather would.

Measured **in JAX under `jit`**, 8 operators per fixture, `nsig` drawn at random, **all outputs
bit-identical to `get_xsource`**:

| fixture | N | hit rate | baseline | pre-filtered | speedup |
| --- | --- | --- | --- | --- | --- |
| uniform random | 2.0M | 0.19% | 409 ms | 100 ms | **4.09x** |
| clustered (1-hop closed) | 1.9M | 6.60% | 388 ms | 146 ms | **2.66x** |

A 16-operator sweep at N=4M reads **4.24x** with the per-operator counting pass included. Speedup
degrades monotonically with hit rate and **break-even is ~50%**, above which plain `searchsorted` wins:

| hit rate | 0.2% | 5.2% | 25.1% | 50.1% | 100% |
| --- | --- | --- | --- | --- | --- |
| speedup | 5.52x | 3.08x | 1.71x | 1.01x | 0.82x |

Three implementation notes, since the JAX details are where this nearly died:

- **Masking alone is 1.00x.** The win requires *compaction* — `jnp.nonzero(present, size=cap)` — so the
  search runs on `cap` elements instead of `N`. Simply masking the result still executes the full search.
- **`cap` must be static, but bucketing it to the next power of two collapses the recompilations.** All
  16 operators in the sweep landed in one bucket: **one compilation**.
- **It shards, and it does not reintroduce a sort.** Under a 4-device mesh (`--xla_force_host_platform_device_count=4`)
  the pre-filter emits **zero** `all-gather` and `all-reduce`, and produces identical output sharding to
  the flat path. The lowered HLO contains 117 `sort` ops — but so does the **unmodified baseline**
  (also 117): those come from `jnp.searchsorted`'s own lowering in this JAX version, not from `nonzero`.
  We flag this because a raw sort count here looks alarming against `23fb226`'s history and is not.
  Note `states` must stay **replicated** (`sqd.py:790`); sharding `keys` makes even the baseline
  `jnp.searchsorted` raise.

### Why this is not committed: `cap` has no safe static bound

**An undersized `cap` silently drops hits.** With a true candidate count of 884, `cap=883` returns a
wrong answer — 883 hits, no error, no warning. And there is no provable bound below `N`: a subspace
closed under a hop has a **100%** hit rate for that hop, and in general the hit rate tracks subspace
density (25% density measured 25.3% hits). `cap = N` is safe and recovers the baseline exactly.

So `cap` has to be derived from the data by a counting pass (cheap — ~1% of the search) and threaded as
a static argument. That is what we measured, and it works. But it makes this a two-step call whose
correctness depends on a parameter a caller could hardcode, and this library's failure history is
specifically silent wrong answers — `docs/locg.md`'s I1–I7, and a `vinit` sign error that returned a
wrong eigenvalue with `converged=True`. **We are not shipping a 4x speedup whose misuse is a silent
wrong answer** until the cap is either derived internally or the truncation is made to raise.

Secondary limit: the bitmap is `2^n / 8` bytes — 134 MB at n=30, **137 GB at n=40**. It needs an `n`
ceiling, or a hashed Bloom filter. False positives are safe here (they cost a wasted search, never a
wrong answer); false negatives are not, which is why the exact bitmap was measured rather than a Bloom.

## 6. What we are not claiming

- **No GPU measurement.** All of the above is one laptop CPU. The gather is better optimized on GPU and
  `23fb226`'s own CPU-to-GH200 ratio compressed from 12–25x to 5.15x, so the §5 figures should be
  assumed optimistic until measured there. This is the largest open risk.
- **Nothing measured through `apply_h` or `_expval_kernel`.** §1–§5 exercise `get_xsource` in isolation.
  Your 105 s at dim 24M is an end-to-end figure; we have not shown what fraction of it §5 would remove.
- **No commitment to implement.** §5 is a candidate with a blocking design problem, not a plan.
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
- **pre-filter (§5):** build `bm` as a `2^n / 8` uint8 bitmap with
  `np.bitwise_or.at(bm, keys >> 3, 1 << (keys & 7))`; in the kernel test the bit, `jnp.nonzero(present,
  size=cap, fill_value=0)`, `searchsorted` the compacted targets, and scatter back with `-1` elsewhere.
  Verify against `get_xsource` — and verify the truncation failure explicitly by passing
  `cap = true_count - 1`.
