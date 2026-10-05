# Proposal: `"pairs"` under a mesh

Status: **for review**, 2026-10-05. Nothing here is built. Figures marked *measured* come from the cited
write-ups; everything marked *estimated* is arithmetic on those, not a run.

## 1. Summary

`Matvec.PAIRS` is the fastest single-device kernel on both backends (*measured*: 3.9–5.6× `"indices"` on one
CPU, 1.1–1.8× on one GPU, `poc/sparse/gpu.md` §9) but raises under a mesh, so a sharded solve falls back to
`"indices"`. Two families of design would lift that, in order of effort:

| design | per-matvec communication | per-device memory | locality needed |
| --- | --- | --- | --- |
| §3 row-owned directed pairs | one all-gather, as `"indices"` | ~2× symmetric pairs / `P` | none |
| §3.1 the same as plain or bucketed ELL | same | §3 plus per-row padding | none |
| §4 its two dials | same | less (recomputed factors, hybrid) | §4.2 only |
| §5 push matvec | one `all_to_all` of hit values | no `N`-length buffer | none |
| §6 GF(2)-linear partition (randomized) | one `ppermute` per distinct offset | ~`(1 + c·log2 P)·N/P` values | yes, gains on local `H` |

§3 is the exact, low-risk step: the same collective `"indices"` pays, so the single-device kernel advantage
should carry over. §6 is the only design that removes the `N`-length gathered vector *and* the all-gather;
it works best on local Hamiltonians and degrades to §3's cost on molecular ones.
§9 predicts the winner: per-device bucketed ELL (§3.1) on one node, §6 across nodes.
§10 makes that ELL shardable and lean: a random hash of the state key, optionally with recomputed factors.

## 2. Today's sharded `"indices"`, the baseline

Each device owns a block of rows of `out`, matching `vec`'s `P('x')` blocks, and holds the `J × N/P`
source indices of those rows. A batched matvec does one all-gather of `vec` (`test/sharded/batch_matvec.py`
pins the count), then a local gather-multiply-add per X group. `states` stays replicated (`13·N` B per
device), and every device materializes the full gathered `vec` (`16·N` B per vector).

## 3. Exact: row-owned directed pairs

- **Storage.** Device `p` stores `(i_local, j, d)` for every transition whose row `i` it owns, **both
  directions** of each symmetric pair. Every update then lands in an owned row: no cross-device reduction,
  `out` stays `P('x')`.
- **Matvec.** All-gather `vec` (as `"indices"`), then today's chunked gather and scatter-add, locally. The
  per-entry arithmetic is unchanged: the single-device kernel already applies two updates per stored pair.
- **Build.** From the symmetric pairs the build already makes, emit both directions, bucket by owner
  `i // (N/P)`, and pad each device's list to the largest one's size class (`_size_class`). Multi-process:
  each process builds only its own rows against the replicated `states`, as `"indices"`' setup does; the
  padded shape needs one unconditional host exchange of the counts at build time (never inside a branch,
  `CLAUDE.md` "A collective inside a conditional deadlocks").
- **Memory, estimated.** Doubling the entries turns the measured ~0.6× of `"indices"` (one GPU) into ~1.2×
  on `type1`; on a low-hit-rate sampled subspace it stays below `"indices"`, which stores a slot for every
  `(group, state)` whether it hits or not.
- **Risk.** Hit rates differ by row block, so entry counts are uneven and padding is paid in memory and
  time. Unmeasured; §7 step 1 measures it.
- **Code.** The matvec is `_apply_pairs` on local shards plus the all-gather; new code is the owner
  bucketing and lifting the mesh guard. The residual check (`_sparse_residual`) stays independent of the
  operator, as now.

An alternative, symmetric storage with a `psum_scatter` of a full-length partial `out`, halves the entries
but doubles the communication and adds an `N`-length buffer. Not proposed.

### 3.1 The removed `"csr"` and `"ell"` already had §3's layout

Both stored entries by target row, both directions, so every update lands in an owned row: §3's layout,
which `"pairs"` must be converted to. Neither was ever mesh-capable (`dev-0.2.4` raised "single-device for
now" for all three sparse kernels), and single-device they lost to `"pairs"` (*measured*,
`poc/sparse/tune.md` §3): `"csr"` dominated on both backends (58.00 against 20.35 ms per GH200 iteration,
`type2` `2^22`, with its sorted hint; still dominated without it); `"ell"` was mixed on a GH200 (0.28×/1.50× `"pairs"` per iteration at `type1` `2^20`/`2^22`,
1.25×/0.70× at `type2`) at more memory and a 4.8 s build, a near-tie on CPU. So the case for them under a
mesh is structural, not speed. Three target-ordered variants, each built from today's `"pairs"` build
(zero-drop, device sort) rather than from `poc/sparse/legacy.py`:

| variant | shape per device | padding | multi-process count exchange |
| --- | --- | --- | --- |
| directed pairs (§3), CSR-like | ragged, padded to the largest device's size class | per device | yes |
| plain ELL | `(width, N/P)`, sharded on `N` with `P('x')` like `"indices"`' table | per row, to the global max width | no |
| ELL bucketed per device | each device's rows bucketed by width, scattered back locally | per bucket | yes, per bucket |

Plain ELL is the simplest to shard: a fixed shape, so no device needs its own count. Its cost is per-row
padding, and the tuned `"ell"`'s fix, bucketing rows by width, reorders rows across the whole vector and
breaks row ownership; under a mesh it must bucket within each device's row block instead. Which variant
wins depends on the row-width spread, which §7 step 1 also measures.

## 4. Exact: two dials on §3

1. **Recomputed factors.** Store `(j, group)` (~6 B) rather than `(i, j, d)` (24 B) and compute `d` from
   `states[i]`'s Z parities, the way `"indices"` computes its diagonal, but only on hits: the sparse twin of
   the `"tables"`/`"indices"` trade. It reads only the device's own rows of `states`, so the matvec needs no
   replicated `states`.
2. **Hybrid storage.** Keep a pair symmetric when both endpoints live on one device, directed only when it
   crosses. Recovers up to half of §3's doubling with no reduce-scatter. The saving depends on locality, and
   range-split locality is hop-dependent on XXZ (`poc/partition-states.md` §4), so measure before building.

## 5. Exact: a push matvec

The owner of `v_j` sends `d·v_j` to the owner of `i`, one `all_to_all` with counts fixed at build time; the
pair list is exactly that send list. Already identified as the per-matvec exchange the partitioned-`states`
line lacks (`poc/partition-states.md` §12 item 5, after Westerhout and Chamberlain, arXiv:2308.16712). It
drops the `N`-length gathered vector from every device, a cost `"indices"` pays too, but moves
~`J·h/P` of the all-gather's bytes (2–12× *more* at `P = 4`, *estimated* there), so it wins on memory
before traffic. Needs a real interconnect to judge and builds on the partitioned-`states` work.

## 6. Randomized: a GF(2)-linear partition, picked by search

**The partition.** Choose a sparse binary matrix `A` with `r = log2 P` rows and give state `s` to device
`A·s` (mod 2). Linearity gives, for an X group with signature `x`,

    owner(s ⊕ x) = A·s ⊕ A·x = owner(s) ⊕ Δ_x,   Δ_x = A·x fixed per group

so every transition of a group shifts the device by the same `Δ_x`, whatever the state:

- **Groups with `Δ_x = 0` never communicate**, and their pairs can stay symmetric (§4.2 for free).
- **Every other group is a fixed device permutation**: device `d` only exchanges with `d ⊕ Δ_x`. A matvec
  is one `ppermute` per distinct `Δ`, at most `P − 1` and usually far fewer, instead of an all-gather.
- **Each device receives only the sources its cross pairs read**, not the whole of `vec`.

**Why a search.** Two goals conflict. Fewer crossings wants `A·x = 0` for the heavily hit groups, but zero
crossings means `A·s` is constant on each connected block of `H`: under XXZ's bond hops the only such row
is the all-ones parity, and a fixed-magnetization subspace puts every state on one device. Balance wants
`A·s` spread evenly, and physical subspaces are correlated (near Néel, bits `k` and `m` are tied by the
parity of `k − m`, so naive few-qubit rows give lopsided devices). So sample many sparse candidates (rows of
weight 1–4 on well-separated qubits) and score each on the actual data, host-side numpy:

- crossing weight `Σ_g hits_g · [A·x_g ≠ 0]`;
- imbalance, the largest device count over the mean;
- the number of distinct `Δ`, i.e. neighbours per device.

Keep the Pareto-best. If no choice balances, hash to `2^r'` buckets with `r' > log2 P` and pack buckets
onto devices greedily: groups that stay inside a bucket still need no communication.

**Estimated, not measured.**

- XXZ at `n = 60`: a single-qubit row crosses only the bonds and fields touching that qubit, ~3 groups. At
  `P = 64` (6 rows) a few percent of groups cross, each device talks to ~6 neighbours (a hypercube), and
  receives ~`6N/P` values against the all-gather's `63N/P`: ~10× less traffic and per-device memory, with
  no `N`-length vector anywhere. Balance is the open risk.
- Molecular JW: a sparse row has odd overlap with about half of the many-orbital X signatures, so most
  groups cross and the cost falls back to about §3's. The win is for local Hamiltonians.

**What it costs the library.** States are laid out by owner, not in global lex order: the build's
`get_xsource` search still runs on the replicated sorted `states`, only the matvec layout changes.
`ground_locg` takes only inner products, so a permuted order is fine; `sqd` must un-permute `eigvec` and
the basis before returning them, and the initial vector's filler mask must follow the permutation. Device
blocks are padded to the largest, so imbalance is paid directly.

**Rejected: randomizing the matvec itself** (sampled entries, sketches). `sqd` needs an exact matvec: its
convergence at `rtol ≈ 4·eps` and the independent residual recomputed after every solve
(`EigenpairCheckError`) would both break, as the closed `Ax`-reuse investigation showed for a far milder
inexactness (`CLAUDE.md`, "Reusing `Ax` to cut `body()`'s 3 matvecs to 2").

## 7. Plan, with a gate at each step

1. **Host-only POC** (me, CPU, no library change): from today's `"pairs"` build on XXZ `type1`/`type2` at
   `n = 60` and molecular-like `n = 14–20`, for `P ∈ {4, 16, 64}`, report §3's per-device entry imbalance,
   §4.2's same-device fraction under a range split, and §6's search: imbalance, same-device fraction,
   distinct `Δ`, receive volume against an all-gather. Gate: §6 at imbalance ≤ 1.25 and ≥ 2× less receive
   volume on XXZ, else build §3 alone.

   For §3.1, the same build also reports the per-row width spread: stored entries against plain ELL's
   `width · N`, and against per-device buckets. Build §3 as plain ELL if its padding is ≤ 1.25× the
   directed entries, otherwise as directed pairs.

   For §10, it also reports ELLC's per-bucket padding when every device must hold one shape, under a
   random whole-key hash against today's lex-ordered row blocks, and §10.2's bytes per slot.
2. **§3, or §6 if step 1 passes**, in the library behind a `test/sharded/*.py` case on virtual CPU devices:
   values against single-device `"pairs"`, the sharding *spec* asserted, and the collective count from
   `.lower(...).compile().as_text()`. Correctness only: virtual-device timings are meaningless.
3. **Multi-GPU** (you): whole `sqd` calls against `"indices"` under the same mesh. Gate: ≥ `"indices"`' speed
   at ≤ its per-device memory. A multi-process run is required too, since virtual devices cannot reach the
   non-addressable-shard class of errors (`CLAUDE.md`, "Sharding tests").

## 8. Decisions for you

1. **Is a sharded `"pairs"` worth a new layout contract?** §6 changes the state order seen by the solver,
   and §3 lifts `"pairs"`' single-device restriction that `CLAUDE.md` and the docs state.
2. **§3 first, or straight to step 1's POC?** §3 is safe but buys no communication; §6 is the only design
   that shrinks the gathered vector, and only step 1 can say whether its balance holds.
3. **Molecules.** §6 helps local Hamiltonians most; if molecular `J` is the target, §4.1's recomputed
   factors may matter more than any partition.

## 9. Prediction: the fastest sparse mesh matvec

*Predicted, not measured.* **Row-sharded ELL, bucketed within each device's rows (§3.1), with one
all-gather per matvec, on one node; across many nodes the all-gather becomes the cost, and §6's partition
takes over for local Hamiltonians.**

**Why ELL over directed pairs.**

- **`"pairs"`' single-device edge over `"ell"` is gone under a mesh.** It stores each transition once and
  writes both ends, half ELL's both-directions entries; under a mesh every target-ordered design stores both
  directions, §3's directed pairs included, so the entry counts are equal.
- **ELL needs no scatter.** By target row it is `out[i] = Σ_w d[w,i]·v[j[w,i]]`, gathers and a reduction
  along the width, no atomics. `"pairs"`' GPU trouble was its scatter: atomics serialized by padding on one
  row (0.32–0.73×, `poc/sparse/prune.md` §6), nondeterministic iteration counts (`poc/sparse/gpu.md` §5),
  the carry split (`poc/sparse/split.md`). Directed pairs keep a scatter-add, or need a sorted segment sum.
- **The single-device data already leans this way.** With twice the entries, the tuned `"ell"` beat
  `"pairs"` per iteration in two of four GH200 cells, 1.50× at `type1` `2^22` and 1.25× at `type2` `2^20`
  (*measured*, `poc/sparse/tune.md` §3). With the doubling taken out of the comparison, it should match or
  beat directed pairs in most cells.
- **Bucketed rather than plain**, because after zero-drop the row widths of a Hamming-shell XXZ subspace
  vary, and plain ELL pads every row to the global maximum.

**When communication takes over** (*estimated* arithmetic). At `N = 2^24` one `complex128` vector is
256 MB.

- **Within a node**, NVLink at hundreds of GB/s: ~0.3–1 ms per all-gather, under a matvec's local compute
  at that size, so the local kernel decides and ELL wins.
- **Across nodes**, InfiniBand at ~25–50 GB/s: ~5–10 ms per vector, matching or exceeding the compute,
  with per-device memory capped by the full gathered vector. There §6's `ppermute` pattern should win by
  about its traffic cut (~10× at `P = 64` on XXZ, §6), if its balance holds; on molecular `H` it falls back
  to the all-gather and ELL's local kernel decides again.
- **On a CPU mesh** ELL's regular gathers vectorize well: close to directed pairs, possibly still behind
  `"tables"`, which gathers no factors.

**What would prove it wrong.**

- **The row-width spread** (§7 step 1): heavy-tailed widths make ELL lose to padding even bucketed, and
  directed pairs win.
- **XLA's lowering of the width reduction on a GPU** may not fuse as well as the scatter path; the
  single-device `"ell"`'s 4.8 s build hints at compile cost.
- **§6's balance on physical subspaces** decides whether the multi-node case ever leaves the all-gather.

## 10. Fast, simple ELL and CSR variants, with a randomized layout

What single-device measurement already settled (`poc/sparse/pairs.md` §4, §7 item 6, §10, *measured*): row
degrees are heavy-tailed (mean 3.7–11.6 entries against a maximum of 60–117), so **HYB loses** (42–78% of
entries spill into its CSR overflow) and a windowed `segment_sum` is 1.7–1.9× slower than C2R. **ELLC**,
rows bucketed by degree on a ×1.25 grid, 14–19 buckets at 2–7.7% padding, is 1.5–2.3× C2R per matvec on CPU.
The variants below keep ELLC's speed across devices; simplest first.

### 10.1 Hashed ELLC

- **The constraint.** An SPMD program has one shape on every device, so each degree bucket must be
  padded to the largest device's count. Today's lex-ordered row blocks have correlated degree histograms
  (inner-shell states, nearly full, sit apart from outer-shell ones), so padding each bucket to the worst
  device could cost a lot.
- **The randomized fix.** Assign each state to a device by a random hash of its whole key, the hashing
  `poc/partition-states.md` §3 measured at 1.03–1.11× balance on XXZ. Each device's histogram then
  concentrates on the global one, and padding a bucket to the largest device costs ~`√count` rows
  (*estimated*, balls in bins).
- **Matvec.** One all-gather of `vec`, as `"indices"`, then per bucket a gather, a multiply and a reduction
  along the width, written by `.at[rows].set(..., unique_indices=True)`: each row is in exactly one bucket,
  so there are no atomics and no scatter-add.
- **Cost.** States are laid out in hash order, so `sqd` un-permutes `eigvec` and the basis on return, as in
  §6. Locality is lost, which the all-gather never used.

### 10.2 Hashed ELLC with recomputed factors

Store `(j, group)` per entry (5–6 B) rather than `(j, d)` (20 B), and compute `d` from the local
`states[i]`'s Z parities by popcount (§4.1 in ELL form). Past the cache the matvec is bandwidth-bound, so
trading bytes for ALU work should pay on a GPU; XXZ has at most 2 Z terms per group. *Estimated* from the
measured mean degrees: ~22–70 B/slot, against `"indices"`' `4·J` = 248–480 B/slot, 4–10× less than the
format a mesh uses today. It is `"indices"` with the misses removed: `"indices"` stores a `-1` and computes a
diagonal for every `(group, state)` slot, hit or not.

### 10.3 Group-major directed lists, the simplest CSR-like form

One X group is a perfect matching (a state occurs in at most one pair per group, `poc/sparse/pairs.md`
§7 item 6), so store each group's directed entries per device and let each group's scatter declare
`unique_indices=True`: no atomics. Under §10.1's hash each group's per-device count is balls in bins, so
padding is small. The cost is `J` scans per matvec, and single-device the atomic-free scatter was mixed
(2.26× at `2^22`, 0.67× at `2^20`, *measured*, `poc/sparse/tune.md` §3). The fallback if ELLC's bucket
compile cost hurts: single-device its 19 scans cost ~+209 MiB of compile memory.

### 10.4 Combined with §6 across nodes

§10.1's random hash balances and §6's linear hash keeps locality; they conflict. Layer them: §6's `A·s`
picks the node and a random hash the device within it, so inter-node traffic follows the `ppermute`
pattern while each node runs the fast all-gather plus hashed ELLC.

### 10.5 Not worth it

- **HYB and a windowed `segment_sum`**: lost single-device (above).
- **`float64` factors for real groups**: −25–30% operator for 0.37–0.94× speed (`poc/sparse/tune.md` §3);
  §10.2 saves more.
- **Delta-encoded `j − i`**: a high-bit flip moves an index by up to half of `N`
  (`markdown/spinchain/rqutils-multiobs-response.md` §2), so deltas do not fit a narrow type.
- **A sampled degree histogram for the bucket grid**: speeds only the build, and ELLC's fixed ×1.25 grid
  needs none.

**Pick**: §10.1 as the first mesh kernel, §10.2 as its memory dial. It reuses a measured fastest format
and a measured balanced hash, and needs no scatter. **What could kill it**: per-bucket padding under one
shared shape, the §7 step 1 measurement above.
