# Proposal: `"pairs"` under a mesh

Status: **for review**, 2026-10-05. Nothing here is built. Figures marked *measured* come from the cited
write-ups or from `poc/split_layouts.py` (§7); everything marked *estimated* is arithmetic on those, not a
run.

## 1. Summary, and the algorithms ranked

`Matvec.PAIRS` is the fastest single-device kernel on both backends (*measured*: 3.9–5.6× `"indices"` on one
CPU, 1.1–1.8× on one GPU, `poc/sparse/gpu.md` §9) but raises under a mesh, so a sharded solve falls back to
`"indices"`. Every candidate, ranked by recommendation (simplicity first, then predicted speed and memory):

| rank | algorithm | §   | communication per matvec | operator per device | new code | for |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | term-parallel `"pairs"`, on named axes | 4.1 | all-gather + reduce-scatter (~2×) | symmetric pairs / `P` | `shard_map` wrapper, group packing | first step, one node |
| 2 | 2-D hashed, ELLC within each block | 4.2 | all-gather along `'r'` + reduce-scatter along `'c'`, `N/√P` each | directed entries / `P` | owner-pair bucketing, row hash | one node and many, any `H` |
| 3 | 1-D hashed ELLC | 4.3 | one all-gather | ELLC / `P`, ~`√count` padding | row hash, per-device buckets | one node; rank 2's local kernel |
| 4 | recomputed factors | 4.4 | as the kernel it dials | ~22–70 B/slot (*estimated*) | popcount factors | memory dial for 2, 3, 5 |
| 5 | row-owned directed pairs / plain ELL | 4.5 | one all-gather | ~2× symmetric pairs / `P` | owner bucketing | fallback for 3 |
| 6 | group-major directed lists | 4.6 | one all-gather | ~2× symmetric pairs / `P` | per-group lists | fallback if ELLC's compile hurts |
| 7 | GF(2)-linear partition | 4.7 | one `ppermute` per distinct offset | ~`(1 + c·log2 P)·N/P` values | partition search, layout permutation | only if rank 2's traffic dominates on a local `H` |
| 8 | linear partition across nodes, hashed ELLC within | 4.8 | `ppermute` between nodes, all-gather within | as 3 | 3 + 7 | many nodes, local `H` |
| 9 | push matvec | 4.9 | one `all_to_all` of hit values | no `N`-length buffer | routed exchange | memory first, needs partitioned `states` |
| 10 | product `S_α × S_β` layout | 4.10 | one all-gather of `V` | near zero | new input form | molecules only |
| 11 | hybrid symmetric/directed storage | 4.11 | one all-gather | up to half of rank 5's doubling | locality split | only if locality exists |

Traffic received or sent per device per vector, *estimated* (`(P_r−1)/P_r · N/P_c` along `'r'` plus
`(P_c−1)/P_c · N/P_r` along `'c'` for rank 2):

| | `P = 4` (2×2) | `P = 64` (8×8) | `N`-sized buffers |
| --- | --- | --- | --- |
| `"indices"` today | ~0.75 N | ~0.98 N | 1 full |
| rank 1 | ~1.5 N | ~1.97 N | 2 full |
| rank 2 | ~0.5 N | ~0.22 N | 2 of `N/√P` |
| rank 7 | — | ~0.09 N if balance holds, XXZ only | ~`6N/P` |

Rejected, with the measurement or reason, in §5. **The recommendation**: build rank 1 first, on named
mesh axes; it reuses today's kernel unchanged and answers whether `"pairs"` beats `"indices"` on a real
multi-GPU mesh before any new layout is built. Then rank 2, the 2-D layout with rank 3's ELLC as its local
kernel and rank 4 as its memory dial: on one node it should tie rank 3, and it scales to many nodes and to
molecules with no balance search. Rank 7 matters only if rank 2's traffic still dominates on a local `H`;
rank 10 is a separate molecular track.

## 2. Today's sharded `"indices"`, the baseline

Each device owns a block of rows of `out`, matching `vec`'s `P('x')` blocks, and holds the `J × N/P`
source indices of those rows. A batched matvec does one all-gather of `vec` (`test/sharded/batch_matvec.py`
pins the count), then a local gather-multiply-add per X group. `states` stays replicated (`13·N` B per
device), and every device materializes the full gathered `vec` (`16·N` B per vector). It needs nothing to
run under a mesh; every candidate below is measured against it.

## 3. What single-device measurement already settled

- **`"csr"` and `"ell"`, removed in `dev-0.2.5`, already stored by target row, both directions**, the layout
  ranks 2–6 need, which `"pairs"` must be converted to. Neither was ever mesh-capable (`dev-0.2.4` raised
  "single-device for now" for all three sparse kernels), and single-device they lost to `"pairs"`
  (*measured*, `poc/sparse/tune.md` §3): `"csr"` dominated on both backends (58.00 against 20.35 ms per GH200
  iteration, `type2` `2^22`, with its sorted hint; still dominated without it); `"ell"` was mixed on a GH200
  (0.28×/1.50× `"pairs"` per iteration at `type1` `2^20`/`2^22`, 1.25×/0.70× at `type2`) at more memory and a
  4.8 s build, a near-tie on CPU. So their case under a mesh is structural, not speed; any variant below is
  built from today's `"pairs"` build (zero-drop, device sort), not from `poc/sparse/legacy.py`.
- **Row degrees are heavy-tailed** (`poc/sparse/pairs.md` §4, §7 item 6, §10, *measured*): mean 3.7–11.6
  entries against a maximum of 60–117. So **HYB loses** (42–78% of entries spill into its CSR overflow) and a
  windowed `segment_sum` is 1.7–1.9× slower than C2R. **ELLC**, rows bucketed by degree on a ×1.25 grid, 14–19
  buckets at 2–7.7% padding, is 1.5–2.3× C2R per matvec on CPU.
- **A whole-key hash balances XXZ subspaces** at 1.03–1.11× (`poc/partition-states.md` §3–§4, *measured*);
  prefix hashing and range splits do not.

## 4. The algorithms, in rank order

### 4.1 Term-parallel `"pairs"` (rank 1)

Split the Hamiltonian, not the rows: `H = d0 + Σ_p H_p`, device `p` owning a subset of whole X groups.

- **Build.** Assign groups to devices by greedy bin packing on their entry counts. Each device runs today's
  `_sparse_operator`, unchanged, on its sub-Hamiltonian, its chunk count padded to the largest device's size
  class (`_size_class`) so the shapes agree. Multi-process is natural: each process builds only its own
  groups, and no rank holds the whole operator.
- **Matvec.** All-gather `vec` (as `"indices"`); each device runs today's `_apply_pairs` on its entries into
  a full-length local accumulator, both scatter directions, device sort and zero-drop unchanged; one
  `psum_scatter` sums the accumulators and leaves each device its `P('x')` row block; `d0 * vec` is
  elementwise and shards for free.
- **Cost.** Two collectives per matvec (~2× `"indices"`' traffic), and two `N`-length buffers per device
  (gathered `vec`, accumulator) against `"indices"`' one; the operator per device is the smallest of any
  candidate, symmetric pairs divided by `P`.
- **Estimated.** Within a node, NVLink at hundreds of GB/s, the extra reduce-scatter is ~0.3–1 ms per vector
  at `N = 2^24`, so most of single-device `"pairs"`' 1.1–1.8× over `"indices"` should survive.
- **Where it loses.** Many nodes, where both collectives grow with `N` and two full vectors per device cap
  memory (rank 2). And `P` approaching `J`, where a few groups dominate the entry count; splitting a large
  group's chunks across devices fixes that, since chunks are independent.
- **Built on named axes.** Rank 1 is rank 2 with all devices on the term axis and none on the row axis,
  so a `shard_map` over named mesh axes makes the later move to 2-D a change of mesh shape and entry
  bucketing, not a rewrite.
- **The batched pair stays batched.** `ground_locg` stacks two matvecs as `(2, N)`; the `shard_map` keeps that
  leading axis, so one all-gather and one reduce-scatter serve both vectors, the 3 → 2 collective cut
  `test/sharded/batch_matvec.py` pins for `"indices"`. This holds for every rank below.

An earlier draft dismissed this form ("halves the entries but doubles the communication and adds an
`N`-length buffer"). Those costs stand; it ranks first for simplicity, not for the least traffic.

### 4.2 2-D hashed layout (rank 2)

The classic distributed sparse-matvec layout: the `P` devices as a `P_r × P_c` grid (`√P × √P` when square)
with mesh axes `('r', 'c')`.

- **Ownership.** Hash each state to a block (§3's whole-key hash), and give device `(R, C)` every directed
  entry `(i, j, d)` with `i` in row block `R` and `j` in column block `C`.
- **Matvec.** All-gather `vec` along `'r'` only, so each device holds its column block (`N/P_c` values); run
  the local kernel with block-local indices; `psum_scatter` along `'c'` only, summing the partial results for
  row block `R` and splitting them back to `P(('r', 'c'))`.
- **Local kernel.** Today's `_apply_pairs` on the block's entries is the simplest; rank 3's ELLC (rows bucketed
  by degree, `.at[rows].set`, no atomics) is the fast form, unchanged inside a block.
- **Build.** Rank 5's owner bucketing keyed by the pair `(block(i), block(j))` rather than `block(i)`, each
  device's list padded to the largest device's size class. Multi-process builds stay per process.
- **Balance.** The hash makes each block's entry count balls in bins, the standard reason 2-D layouts use a
  random or hash assignment on irregular graphs (Boman et al., SC 2013).
- **Traffic.** ~`N/P_c + N/P_r` per device per vector, `√P` less than rank 1's two full collectives: 3–9×
  less at `P = 4`–`64`, below `"indices"` from `P = 4`, with buffers of `N/√P` (*estimated*, §1). It needs no
  locality, so unlike rank 7 it helps molecules; rank 7 can still beat it on a local `H` if its balance
  search succeeds.
- **Cost.** Two collectives on sub-axes, as rank 1, so on one node it should about tie rank 1; directed
  entries, twice the symmetric count, as ranks 3–6; states in hash order, so `sqd` un-permutes on return, as
  rank 3. A non-square `P_r × P_c` works at `N/P_c + N/P_r`.
- **Risk.** Each block's entry count under one shared shape, directed entries split `P_r × P_c` ways: §8
  step 2 measures it.

### 4.3 Hashed ELLC (rank 3)

- **The constraint.** An SPMD program has one shape on every device, so each degree bucket must be padded
  to the largest device's count. Today's lex-ordered row blocks have correlated degree histograms
  (inner-shell states, nearly full, sit apart from outer-shell ones), so padding each bucket to the worst
  device could cost a lot.
- **The randomized fix.** Assign each state to a device by a random hash of its whole key (§3). Each
  device's histogram then concentrates on the global one, and padding a bucket to the largest device costs
  ~`√count` rows (*estimated*, balls in bins).
- **Matvec.** One all-gather of `vec`, as `"indices"`, then per bucket a gather, a multiply and a reduction
  along the width, `out[i] = Σ_w d[w,i]·v[j[w,i]]`, written by `.at[rows].set(..., unique_indices=True)`: each
  row is in exactly one bucket, so there are no atomics and no scatter-add.
- **Cost.** States are laid out in hash order, so `sqd` un-permutes `eigvec` and the basis on return and the
  initial vector's filler mask follows the permutation. Locality is lost, which the all-gather never used.
  Single-device ELLC's 19 scans cost ~+209 MiB of compile memory.
- **Inside rank 2** it is the local kernel, unchanged, on each block's entries.
- **Why it is predicted fastest on one node**: §6.

### 4.4 Hashed ELLC with recomputed factors (rank 4)

Store `(j, group)` per entry (5–6 B) rather than `(j, d)` (20 B), and compute `d` from the local `states[i]`'s
Z parities by popcount, the way `"indices"` computes its diagonal but only on hits: the sparse twin of the
`"tables"`/`"indices"` trade. Past the cache the matvec is bandwidth-bound, so trading bytes for ALU work
should pay on a GPU; XXZ has at most 2 Z terms per group. *Estimated* from the measured mean degrees:
~22–70 B/slot, against `"indices"`' `4·J` = 248–480 B/slot, 4–10× less than the format a mesh uses today.
It is `"indices"` with the misses removed: `"indices"` stores a `-1` and computes a diagonal for every
`(group, state)` slot, hit or not. It reads only the device's own rows of `states`, so the matvec needs no
replicated `states`. Applies equally to rank 5 (`(j, group)` ~6 B against `(i, j, d)` 24 B), and to rank 2:
a device in grid row `R` needs only row block `R` of `states`, so it costs nothing extra there.

### 4.5 Row-owned directed pairs, or plain ELL (rank 5)

- **Storage.** Device `p` stores `(i_local, j, d)` for every transition whose row `i` it owns, **both
  directions** of each symmetric pair, so every update lands in an owned row: no cross-device reduction,
  `out` stays `P('x')`.
- **Matvec.** All-gather `vec`, then today's chunked gather and scatter-add, locally. The per-entry
  arithmetic is unchanged: the single-device kernel already applies two updates per stored pair.
- **Build.** From the symmetric pairs the build already makes, emit both directions, bucket by owner
  `i // (N/P)`, and pad each device's list to the largest one's size class. Multi-process: each process
  builds only its own rows against the replicated `states`, as `"indices"`' setup does; the padded shape
  needs one unconditional host exchange of the counts at build time (never inside a branch, `CLAUDE.md` "A
  collective inside a conditional deadlocks").
- **Memory, estimated.** Doubling the entries turns the measured ~0.6× of `"indices"` (one GPU) into ~1.2× on
  `type1`; on a low-hit-rate sampled subspace it stays below `"indices"`, which stores a slot for every
  `(group, state)` whether it hits or not.
- **Risk.** Hit rates differ by row block, so entry counts are uneven and padding is paid in memory and time.
- **Code.** `_apply_pairs` on local shards plus the all-gather; the residual check (`_sparse_residual`) stays
  independent of the operator.

Three target-ordered shapes for it:

| variant | shape per device | padding | multi-process count exchange |
| --- | --- | --- | --- |
| directed pairs, CSR-like | ragged, padded to the largest device's size class | per device | yes |
| plain ELL | `(width, N/P)`, sharded on `N` with `P('x')` like `"indices"`' table | per row, to the global max width | no |
| ELL bucketed per device | each device's rows bucketed by width, scattered back locally | per bucket | yes, per bucket |

Plain ELL is the simplest to shard: a fixed shape, so no device needs its own count. Its cost is per-row
padding, and the tuned `"ell"`'s fix, bucketing rows by width, reorders rows across the whole vector and
breaks row ownership; under a mesh it must bucket within each device's row block instead (which is rank 3,
with a hash for balance).

### 4.6 Group-major directed lists (rank 6)

One X group is a perfect matching (a state occurs in at most one pair per group, `poc/sparse/pairs.md` §7
item 6), so store each group's directed entries per device and let each group's scatter declare
`unique_indices=True`: no atomics. Under rank 3's hash each group's per-device count is balls in bins, so
padding is small. The cost is `J` scans per matvec, and single-device the atomic-free scatter was mixed
(2.26× at `2^22`, 0.67× at `2^20`, *measured*, `poc/sparse/tune.md` §3). The fallback if rank 3's bucket
compile cost hurts.

### 4.7 GF(2)-linear partition, picked by randomized search (rank 7)

Ranked below rank 2: build it only if rank 2's traffic still dominates on a local `H`.

**The partition.** Choose a sparse binary matrix `A` with `r = log2 P` rows and give state `s` to device `A·s`
(mod 2). Linearity gives, for an X group with signature `x`,

    owner(s ⊕ x) = A·s ⊕ A·x = owner(s) ⊕ Δ_x,   Δ_x = A·x fixed per group

so every transition of a group shifts the device by the same `Δ_x`, whatever the state:

- **Groups with `Δ_x = 0` never communicate**, and their pairs can stay symmetric (rank 11 for free).
- **Every other group is a fixed device permutation**: device `d` only exchanges with `d ⊕ Δ_x`. A matvec is
  one `ppermute` per distinct `Δ`, at most `P − 1` and usually far fewer, instead of an all-gather.
- **Each device receives only the sources its cross pairs read**, not the whole of `vec`.

**Why a search.** Two goals conflict. Fewer crossings wants `A·x = 0` for the heavily hit groups, but zero
crossings means `A·s` is constant on each connected block of `H`: under XXZ's bond hops the only such row is
the all-ones parity, and a fixed-magnetization subspace puts every state on one device. Balance wants `A·s`
spread evenly, and physical subspaces are correlated (near Néel, bits `k` and `m` are tied by the parity of
`k − m`, so naive few-qubit rows give lopsided devices). So sample many sparse candidates (rows of weight 1–4
on well-separated qubits) and score each on the actual data, host-side numpy:

- crossing weight `Σ_g hits_g · [A·x_g ≠ 0]`;
- imbalance, the largest device count over the mean;
- the number of distinct `Δ`, i.e. neighbours per device.

Keep the Pareto-best. If no choice balances, hash to `2^r'` buckets with `r' > log2 P` and pack buckets onto
devices greedily: groups that stay inside a bucket still need no communication.

**Estimated, not measured.**

- XXZ at `n = 60`: a single-qubit row crosses only the bonds and fields touching that qubit, ~3 groups. At
  `P = 64` (6 rows) a few percent of groups cross, each device talks to ~6 neighbours (a hypercube), and
  receives ~`6N/P` values against the all-gather's `63N/P`: ~10× less traffic and per-device memory, with no
  `N`-length vector anywhere. Balance is the open risk.
- Molecular JW: a sparse row has odd overlap with about half of the many-orbital X signatures, so most groups
  cross and the cost falls back to about rank 5's. The win is for local Hamiltonians.

**What it costs the library.** States are laid out by owner, not in global lex order: the build's
`get_xsource` search still runs on the replicated sorted `states`, only the matvec layout changes.
`ground_locg` takes only inner products, so a permuted order is fine; `sqd` must un-permute `eigvec` and the
basis before returning them, and the initial vector's filler mask must follow the permutation. Device blocks
are padded to the largest, so imbalance is paid directly.

### 4.8 Linear partition across nodes, hashed ELLC within (rank 8)

Rank 3's random hash balances and rank 7's linear hash keeps locality; they conflict. Layer them: rank 7's
`A·s` picks the node and a random hash the device within it, so inter-node traffic follows the `ppermute`
pattern while each node runs the fast all-gather plus hashed ELLC. Rank 2 reaches many nodes without the
linear partition's balance search, so this is for a local `H` where rank 7 beats rank 2's traffic.

### 4.9 Push matvec (rank 9)

The owner of `v_j` sends `d·v_j` to the owner of `i`, one `all_to_all` with counts fixed at build time; the
pair list is exactly that send list. Already identified as the per-matvec exchange the partitioned-`states`
line lacks (`poc/partition-states.md` §12 item 5, after Westerhout and Chamberlain, arXiv:2308.16712). It
drops the `N`-length gathered vector from every device, a cost `"indices"` pays too, but moves ~`J·h/P` of
the all-gather's bytes (2–12× *more* at `P = 4`, *estimated* there), so it wins on memory before traffic.
Needs a real interconnect to judge and builds on the partitioned-`states` work.

### 4.10 The product `S_α × S_β` layout, molecules only (rank 10)

Cut each bitstring into a left and a right half, `s = (a, b)`, and require the subspace to be a full product
`S_L × S_R`; then `vec` is a dense matrix `V[a, b]` and the operator factors.

- **X flips act on each axis separately.** `X^x`, `x = (x_L, x_R)`, is a row map `π_L` on `S_L` composed with
  a column map `π_R` on `S_R`. The searches run over ~`√N` entries rather than `N`, and all `J` maps take
  `J·(|S_L| + |S_R|)·4` B: small enough to store, so no per-matvec search at all.
- **Diagonals are rank-`K` outer products.** `(-1)^{z·s} = (-1)^{z_L·a}·(-1)^{z_R·b}`, so a Z term's diagonal
  is `u ⊗ w`, two vectors of ~`√N`. Nothing of size `N` is stored but `V`.
- **The matvec**, per group: a row gather of `V`, a column gather, a multiply by `Σ_k c_k u_k w_kᵀ`. Coalesced,
  regular gathers along one axis, which suits a GPU; groups sharing an `x_L` share the row gather.
- **Under a mesh**, `V` sharded by rows: column work is local, row maps need other devices' rows. The simplest
  form all-gathers `V` once per matvec, `"indices"`' communication, with the operator's memory near zero.

**Spinchain: dead** (*measured*, §7). The product of halves is far larger than the subspace:

| N | cut 30 | cut 20 |
| --- | --- | --- |
| `2^14` | 376× | 282× |
| `2^17` | 1156× | 815× |
| `2^20` | 3893× | 2415× |

**Chemistry SQD: native.** qiskit-addon-sqd already forms the subspace as `S_α × S_β` from the unique α and β
strings, and Jordan–Wigner puts the α and β blocks on separate qubit halves. It is the string-driven σ-vector
of FCI codes (Knowles–Handy; Olsen et al.): the search shrinks from `N` to ~`√N` per spin sector, and index
memory from `J·N` to `J·√N`, the molecular `J` problem `rqutils/sqd/__init__.py` documents. **Caveat**: αβ
two-body terms carry ~`n_α²` distinct `x_α`, so moving rows per pattern costs ~`n_α²·N/P` (*estimated*), which
loses to one all-gather unless `P` is large; hence the all-gather form. It needs the subspace as
`(S_α, S_β)` rather than a state list, an API addition. Unmeasured on a molecule.

### 4.11 Hybrid symmetric/directed storage (rank 11)

Keep a pair symmetric when both endpoints live on one device, directed only when it crosses: recovers up to
half of rank 5's doubling with no reduce-scatter. The saving depends on locality, and range-split locality is
hop-dependent on XXZ (`poc/partition-states.md` §4), and a random hash (rank 3) has none; it comes free only
under rank 7. Measure before building.

## 5. Rejected

- **Ownership by a random hash of the left half** (*measured*, §7). The aim was randomization for balance and
  locality together: every group flipping only right-half qubits stays on its device (2/3 of the bonds at
  cut 20), with no linearity constraint like rank 7's. **Dead beyond `P = 4`**: states near Néel share a few
  left halves, one alone owning 3–27% of the subspace. Largest device load over the mean, `2^17` and `2^20`:

  | | `P = 4` | `P = 16` | `P = 64` |
  | --- | --- | --- | --- |
  | cut 30 | 1.05–1.14× | 1.38–1.83× | 3.11–4.22× |
  | cut 20 | 1.24–1.50× | 2.46–2.71× | 7.95–8.39× |
  | cut 10 | 1.77–1.96× | 4.22–4.68× | 15.51–17.19× |

  Splitting a heavy left half by a few right-half bits rebalances, but every group flipping those bits then
  crosses devices, giving the locality back. The concentration near Néel that kills this and rank 10 on
  spinchain is the one that makes a range split fail (`poc/partition-states.md` §4), so ranks 3 and 7 stay the
  spinchain candidates.
- **Randomizing the matvec itself** (sampled entries, sketches). `sqd` needs an exact matvec: its convergence
  at `rtol ≈ 4·eps` and the independent residual recomputed after every solve (`EigenpairCheckError`) would
  both break, as the closed `Ax`-reuse investigation showed for a far milder inexactness (`CLAUDE.md`,
  "Reusing `Ax` to cut `body()`'s 3 matvecs to 2").
- **HYB and a windowed `segment_sum`**: lost single-device (§3).
- **`float64` factors for real groups**: −25–30% operator for 0.37–0.94× speed (`poc/sparse/tune.md` §3);
  rank 4 saves more.
- **Delta-encoded `j − i`**: a high-bit flip moves an index by up to half of `N`
  (`markdown/spinchain/rqutils-multiobs-response.md` §2), so deltas do not fit a narrow type.
- **A sampled degree histogram for the bucket grid**: speeds only the build, and ELLC's fixed ×1.25 grid needs
  none.

## 6. Prediction: the fastest sparse mesh matvec

*Predicted, not measured.* **ELL bucketed within each device's rows and balanced by a random hash, as the
local kernel of the 2-D layout (rank 2).** On one node it about ties 1-D hashed ELLC (rank 3), since
communication is cheap there; across many nodes its `√P`-smaller collectives keep it ahead, and rank 7's
partition overtakes it only on a local `H` whose balance search succeeds.

**Why ELL over directed pairs.**

- **`"pairs"`' single-device edge over `"ell"` is gone under a row-owned mesh layout.** It stores each
  transition once and writes both ends, half ELL's both-directions entries; under a mesh every
  target-ordered design stores both directions, rank 5's directed pairs included, so the entry counts are
  equal. (Rank 1 keeps the symmetric half by paying a reduce-scatter instead.)
- **ELL needs no scatter.** By target row it is gathers and a reduction along the width, no atomics.
  `"pairs"`' GPU trouble was its scatter: atomics serialized by padding on one row (0.32–0.73×,
  `poc/sparse/prune.md` §6), nondeterministic iteration counts (`poc/sparse/gpu.md` §5), the carry split
  (`poc/sparse/split.md`). Directed pairs keep a scatter-add, or need a sorted segment sum.
- **The single-device data already leans this way.** With twice the entries, the tuned `"ell"` beat `"pairs"`
  per iteration in two of four GH200 cells, 1.50× at `type1` `2^22` and 1.25× at `type2` `2^20` (*measured*,
  `poc/sparse/tune.md` §3). With the doubling taken out of the comparison, it should match or beat directed
  pairs in most cells.
- **Bucketed rather than plain**, because after zero-drop the row widths of a Hamming-shell XXZ subspace
  vary, and plain ELL pads every row to the global maximum.

**When communication takes over** (*estimated* arithmetic). At `N = 2^24` one `complex128` vector is 256 MB.

- **Within a node**, NVLink at hundreds of GB/s: ~0.3–1 ms per all-gather, under a matvec's local compute at
  that size, so the local kernel decides and ELL wins.
- **Across nodes**, InfiniBand at ~25–50 GB/s: ~5–10 ms per vector, matching or exceeding the compute, with
  per-device memory capped by the full gathered vector. There rank 2's sub-axis collectives cut that to
  ~0.22 N at `P = 64` (§1), for any `H`; rank 7's `ppermute` pattern can go further (~0.09 N, ~10× below the
  all-gather on XXZ) if its balance holds, and on molecular `H` it falls back to the all-gather.
- **On a CPU mesh** ELL's regular gathers vectorize well: close to directed pairs, possibly still behind
  `"tables"`, which gathers no factors.

**What would prove it wrong.**

- **The row-width spread** (§8 step 2): heavy-tailed widths make ELL lose to padding even bucketed, and
  directed pairs win.
- **Per-bucket padding under one shared shape**, if the random hash concentrates less than balls in bins.
- **XLA's lowering of the width reduction on a GPU** may not fuse as well as the scatter path; the
  single-device `"ell"`'s 4.8 s build hints at compile cost.
- **Rank 2's per-block balance** under one shared shape, directed entries split `P_r × P_c` ways.
- **Rank 7's balance on physical subspaces** decides whether it ever beats rank 2 on a local `H`.

## 7. The script behind §4.10 and §5

`poc/split_layouts.py`, host-only, against its argparse: `--num-qubits` (60), `--product-sizes` (`14 17 20`)
and `--product-cuts` (`30 20`) for §4.10, `--hash-sizes` (`17 20`), `--hash-cuts` (`30 20 10`) and
`--devices` (`4 16 64`) for §5's left-half hash. Fixture: `poc/eigenpair_check_scale`'s Hamming-shell
subspaces around both Néel states; the left half is a state's first `cut` columns; the hash is a random salt
per distinct left half, drawn in loop order from one seeded generator, so a run reproduces the tables. Run at
`49366a5` on an Apple M1, 2026-10-05, with the defaults.

## 8. Plan, with a gate at each step

1. **Rank 1 on virtual CPU devices** (me): a `test/sharded/*.py` case against single-device `"pairs"`, the
   sharding *spec* asserted, and the collective count from `.lower(...).compile().as_text()`. Correctness
   only: virtual-device timings are meaningless. Then **multi-GPU** (you): whole `sqd` calls against
   `"indices"` under the same mesh, and a multi-process run, since virtual devices cannot reach the
   non-addressable-shard class of errors (`CLAUDE.md`, "Sharding tests"). Gate: ≥ `"indices"`' speed at ≤ its
   per-device memory. If it passes and one node is the target, stop here.
2. **Host-only POC** (me, CPU, no library change), from today's `"pairs"` build on XXZ `type1`/`type2` at
   `n = 60` and molecular-like `n = 14–20`, for `P ∈ {4, 16, 64}`:
   - rank 2: each block's directed entry count against the mean, under the random hash, for
     `P_r × P_c ∈ {2×2, 4×4, 8×8}`;
   - rank 3: ELLC's per-bucket padding when every device must hold one shape, under a random whole-key hash
     against today's lex-ordered row blocks, and rank 4's bytes per slot;
   - rank 5: per-device entry imbalance, and the per-row width spread (stored entries against plain ELL's
     `width · N` and against per-device buckets);
   - rank 11: the same-device fraction under a range split;
   - rank 7: the search's imbalance, same-device fraction, distinct `Δ`, and receive volume against an
     all-gather.

   Gates: rank 2 with ELLC locally if its block imbalance and bucket padding both stay ≤ 1.25×; rank 3 if
   only the 1-D padding does; else rank 5 as plain ELL if its padding is ≤ 1.25× the
   directed entries, otherwise as directed pairs. Rank 7 only at imbalance ≤ 1.25 and ≥ 2× less receive
   volume on XXZ.
3. **The step 2 winner** in the library, behind the same sharded test, then the same multi-GPU gate against
   rank 1.
4. **Ranks 7–9** only if a multi-node run shows rank 2's collectives dominating.

## 9. Decisions for you

1. **Is a sharded `"pairs"` worth lifting its single-device restriction**, which `CLAUDE.md` and the docs
   state? Rank 1 does it with no layout change; ranks 2–8 also change the state order the solver sees.
2. **Rank 1 first, or straight to step 2's POC?** Rank 1 is the cheapest real measurement; step 2 decides
   between the layouts without a GPU.
3. **Molecules.** Rank 2 needs no locality, so it serves molecules as well as spin chains; ranks 7–8 help
   local Hamiltonians only. If molecular `J` is the target, rank 4's recomputed factors or rank 10's product
   layout may matter more than any partition.
