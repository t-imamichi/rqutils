# Proposal: `"pairs"` under a mesh

Status: **for review**, 2026-10-05; revised the same day after an independent review (§10). **Rank 1 is
merged** to `dev` (2026-10-06, `f5a981a`), correctness on virtual CPU devices only (§4.1, §8 step 1);
nothing else here is built. Figures marked *measured* come from the cited write-ups or from `poc/split_layouts.py` (§7);
everything marked *estimated* is arithmetic on those, not a run. **Post-drop** means after `_drop_zeros`
(`6d13bfb`) and its padding fix (`d0a9997`); several single-device comparisons predate them and are marked
*stale*.

## 1. Summary, and the algorithms ranked

`Matvec.PAIRS` is the fastest single-device kernel on both backends (*measured*: 3.9–5.6× `"indices"` on one
CPU, 1.08–1.82× on one GPU, `poc/sparse/gpu.md` §9) but raised under a mesh until rank 1, so a sharded solve fell
back to `"indices"`. The GPU figure predates the zero-drop, which made `"pairs"` itself 1.85–5.46× faster per solve
on a GH200 (`poc/sparse/prune.md` §6), so today's gap is likely wider; unmeasured. Every candidate, ranked by
recommendation (simplicity first, then predicted speed and memory):

| rank | algorithm | §   | communication per matvec | operator per device | new code | for |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | term-parallel `"pairs"` | 4.1 | all-gather + reduce-scatter (~2×) | symmetric pairs / `P` | `shard_map` wrapper, contiguous entry split | first step, one node |
| 2 | 1-D hashed rows, ELLC, all-gather then halo | 4.2 | one all-gather, then one padded `all_to_all` of the sources read | ELLC / `P` | row hash, buckets, then send lists | spin chains, one node and many |
| 3 | 2-D hashed, ELLC within each block | 4.3 | all-gather along `'r'` + reduce-scatter along `'c'`, `N/√P` each | directed entries / `P` | owner-pair bucketing | molecules; spin chains at small `P` |
| 4 | recomputed factors | 4.4 | as the kernel it dials | ~3–43 B/slot (*estimated*) | popcount factors | memory dial for 2, 3, 6 |
| 5 | ring-pipelined all-gather | 4.5 | the all-gather as `P − 1` `ppermute` steps, overlapped | entries bucketed by source owner | per-owner buckets | many nodes, directed pairs |
| 6 | row-owned directed pairs / plain ELL | 4.6 | as rank 2 | ~2× symmetric pairs / `P` | owner bucketing | fallback local kernel for 2 |
| 7 | group-major directed lists | 4.7 | as rank 2 | ~2× symmetric pairs / `P` | per-group lists | fallback if ELLC's compile hurts |
| 8 | product `S_α × S_β` layout | 4.8 | one all-gather of `V` | near zero | new input form | molecules only |
| 9 | GF(2)-linear partition | 4.9 | one `ppermute` per distinct offset | ~`(1 + c·log2 P)·N/P` values | partition search, layout permutation | only if rank 2's halo still dominates on a local `H` |
| 10 | hybrid symmetric/directed storage | 4.10 | as rank 2 | up to half of rank 6's doubling | locality split | only under rank 9 |

Traffic received or sent per device per vector, *estimated*. Rank 2's halo is bounded by the remote
directed entries a device's rows read, `deg·(P−1)/P²·N`, and by the distinct remote sources,
`(P−1)/P·N`, at the post-drop directed degree `deg` (0.51 `type1`, 7.1 `type2`, §3); rank 3's is
`(P_r−1)/P_r · N/P_c` along `'r'` plus `(P_c−1)/P_c · N/P_r` along `'c'`:

| | `P = 4` | `P = 64` | `N`-sized buffers |
| --- | --- | --- | --- |
| `"indices"` today | ~0.75 N | ~0.98 N | 1 full |
| rank 1 | ~1.5 N | ~1.97 N | 2 full |
| rank 2, all-gather | ~0.75 N | ~0.98 N | 1 full |
| rank 2, halo, `type1` | ≤ ~0.10 N | ≤ ~0.008 N | none |
| rank 2, halo, `type2` | ≤ ~0.75 N | ≤ ~0.11 N | none |
| rank 3 (2×2, 8×8) | ~0.5 N | ~0.22 N | 2 of `N/√P` |
| rank 9 | — | ~0.09 N if balance holds, XXZ only | ~`6N/P` |

Rejected, with the measurement or reason, in §5. **The recommendation**: build rank 1 first as the cheap
gate; it reuses today's kernel unchanged and answers whether `"pairs"` beats `"indices"` on a real
multi-GPU mesh. Then rank 2: the row-owned hashed layout with ELLC, first with the all-gather (`"indices"`'
own collective), then with the halo exchange, which needs no `N`-length buffer and, at spin-chain degrees,
moves less than every alternative but rank 3 at `type2` `P = 4`. Rank 3 is the molecular layout, where the
degree is in the hundreds and the halo approaches an all-gather. Rank 4 is a memory dial, not a speedup.
Rank 9 survives only if rank 2's halo still dominates on a local `H`; rank 8 is a separate molecular track.

## 2. Today's sharded `"indices"`, the baseline

Each device owns a block of rows of `out`, matching `vec`'s `P('x')` blocks, and holds the `J × N/P`
source indices of those rows. A batched matvec does one all-gather of `vec` (`test/sharded/batch_matvec.py`
pins the count), then a local gather-multiply-add per X group. `states` stays replicated (`13·N` B per
device), and every device materializes the full gathered `vec` (`16·N` B per vector). It needs nothing to
run under a mesh; every candidate below is measured against it.

## 3. What single-device measurement already settled

- **`"csr"` and `"ell"`, removed in `dev-0.2.5`, already stored by target row, both directions**, the layout
  ranks 2, 3, 5–7 need, which `"pairs"` must be converted to. Neither was ever mesh-capable (`dev-0.2.4`
  raised "single-device for now" for all three sparse kernels), and single-device they lost to `"pairs"`
  (*measured*, `poc/sparse/tune.md` §3): `"csr"` dominated on both backends (58.00 against 20.35 ms per GH200
  iteration, `type2` `2^22`, with its sorted hint; still dominated without it); `"ell"` was mixed on a GH200
  (0.28×/1.50× `"pairs"` per iteration at `type1` `2^20`/`2^22`, 1.25×/0.70× at `type2`) at more memory and a
  4.8 s build, a near-tie on CPU. **Stale**: measured at `1e6fbfc`/`4317d25`, against a `"pairs"` that still
  stored 86%/32% exact zeros and padded every entry onto row `size − 1`, a handicap ELL did not share. So
  their case under a mesh is structural, not speed; any variant below is built from today's `"pairs"` build
  (zero-drop, device sort), not from `poc/sparse/legacy.py`.
- **Row degrees are heavy-tailed** (`poc/sparse/pairs.md` §4, §7 item 6, §10, *measured* pre-drop): mean
  3.7–11.6 entries against a maximum of 60–117. So **HYB loses** (42–78% of entries spill into its CSR
  overflow) and a windowed `segment_sum` is 1.7–1.9× slower than C2R. **ELLC**, rows bucketed by degree on a
  ×1.25 grid, 14–19 buckets at 2–7.7% padding, is 1.5–2.3× C2R per matvec on CPU.
- **Post-drop the degree is far lower** (*measured* entry counts, `poc/sparse/prune.md` §3, `2^17`):
  33,722 and 467,176 stored pairs, so the mean directed degree is 2·33,722/2^17 = **0.51** (`type1`) and
  2·467,176/2^17 = **7.1** (`type2`). Every estimate below uses these; the post-drop degree histogram, and
  so ELLC's post-drop padding, is unmeasured.
- **A whole-key hash balances XXZ subspaces** at 1.03–1.11× (`poc/partition-states.md` §3–§4, *measured*,
  with its `mix64`); prefix hashing and range splits do not.
- **Molecular-like degree is in the hundreds**: hit rate 0.71 at `J = 872` (n = 14, `poc/sparse/pairs.md`
  §5), ~620 entries per row (*estimated*).

## 4. The algorithms, in rank order

Shared by every rank: `ground_locg` stacks two matvecs as `(2, N)`, and each `shard_map` keeps that leading
axis, so one collective of each kind serves both vectors, the 3 → 2 cut `test/sharded/batch_matvec.py` pins
for `"indices"`. Every layout that permutes states computes the initial vector (`_spread_seed`) in lex order
and permutes it in, so the trajectory matches single-device and tests can assert the iteration count
(`CLAUDE.md`, "assert the iteration count, not the energy"). Every multi-process build agrees its padded
shapes through **one unconditional host allgather of all per-group and per-bucket counts**, never inside a
branch (`CLAUDE.md`, "A collective inside a conditional deadlocks"), and every host read of a device scalar
goes through `_host_scalar`.

**The residual check, every rank.** `_sparse_residual` rebuilds each group's sources from the *global,
unfiltered* `pairs` (`_pair_xsources`, `rqutils/sqd/_sparse.py`), so a build that searches only its own
groups or rows cannot feed it. Either every process re-runs the full host search for the check, as today's
single-device check costs, or the per-process search results are gathered. The check stays independent of
the sharded operator either way.

### 4.1 Term-parallel `"pairs"` (rank 1)

Split the Hamiltonian's entries, not the rows: `H = d0 + Σ_p H_p`, device `p` owning a share of the stored
transitions.

- **Build.** `_sparse_operator(..., mesh)`: today's build, once, then `_compact` gives each device one
  contiguous slice of the `i`-sorted stored entries, the first `count % P` devices one more, each padded to
  one size class (`_size_class`) so the shapes agree. Any split is exact, since each device sums into a
  full-length accumulator; contiguous slices balance to one entry. (The first slice form gave each device
  `ceil(count / P)`, leaving the last short by up to `P − 1`, 2, 2, 2, 0 for 6 entries on 4: found by code
  review, fixed, and pinned by a test sweeping every remainder at 2–4 devices.) **Whole X groups,
  the first form built, did not balance** (*measured*, `poc/sparse/mesh_balance.py`, n = 60, `2^17`, largest
  device over the mean):

  | pattern | `J` | largest group's share of the kept entries | `P = 4` | `P = 16` | `P = 64` |
  | --- | --- | --- | --- | --- | --- |
  | `type1`, `type4`, whole groups | 61 | 22.3% | 1.41× | 3.86× | 14.26×, 3 devices idle |
  | `type2`, `type3`, whole groups | 119 | 1.6% | 1.00× | 1.07× | 1.07× |
  | all four, contiguous slices | | | 1.000× | 1.000× | 1.000× |

  Packing balanced the *searched* counts, but `_drop_zeros` keeps 33,722 of `type1`'s 249,104 pairs, unevenly
  by group, and one group's 22.3% caps any whole-group packing at `0.223·P` the mean. The slices pay 5–12% of
  padding slots. A random split balances too, but loses the slices' contiguous `out[i]` rows.
- **Matvec.** `_apply_pairs_mesh`, chosen by `_run_sparse` under a mesh: all-gather `vec` (as `"indices"`);
  each device runs today's `_apply_pairs` scan on its entries into
  a full-length local accumulator, both scatter directions, device sort and zero-drop unchanged; one
  `psum_scatter` sums the accumulators and leaves each device its `P('x')` row block; `d0 * vec` is
  elementwise and shards for free.
- **Cost.** Two collectives per matvec (~2× `"indices"`' traffic), and two `N`-length buffers per device
  (gathered `vec`, accumulator) against `"indices"`' one; the operator per device is the smallest of any
  candidate, symmetric pairs divided by `P`.
- **Estimated.** Within a node, NVLink at hundreds of GB/s, the extra reduce-scatter is ~0.3–1 ms per vector
  at `N = 2^24`. Each device still gathers from and scatters into full-length vectors, so its cache footprint
  is `N`, not `N/P`: expect the large-`N` end of single-device `"pairs"`' edge, 1.08× at `type1` `2^22`
  (`poc/sparse/gpu.md` §9, pre-drop), not the 1.82×.
- **Where it loses.** Many nodes, where both collectives grow with `N` and two full vectors per device cap
  memory (ranks 2–3).
- **Single-device `"pairs"` is unchanged** (*measured* against `dev`, M1, n = 60, `type1`/`type2`): the operator
  is bit-identical and the matvec's traced graph and lowered HLO identical at `2^17`; whole `sqd` calls
  interleaved against a `dev` worktree (4 alternations × 5 warm rounds, `2^14` and `2^17`) are 0.996–1.016×,
  paired wins split, eigenvalues bit-identical. That A/B ran before the remainder fix, which leaves the
  single-device operator and HLO unchanged; no committed script reproduces it.
- **Not a degenerate rank 3.** Its symmetric storage needs both `v_i` and `v_j` and writes both rows, which no
  proper 2-D block allows; it is a separate term axis, as in 2.5-D and 3-D layouts. Moving from rank 1 to rank
  2 or 3 changes the storage and the build, not just the mesh shape.

An earlier draft dismissed this form ("halves the entries but doubles the communication and adds an
`N`-length buffer"). Those costs stand; it ranks first for simplicity, not for the least traffic.

### 4.2 1-D hashed rows with ELLC: all-gather, then halo (rank 2)

- **Ownership.** Assign each state to a device by `mix64` of its whole key (§3). An SPMD program has one shape
  on every device, so each degree bucket must be padded to the largest device's count; today's lex-ordered
  row blocks have correlated degree histograms (inner-shell states, nearly full, sit apart from outer-shell
  ones), while under the hash each device's histogram concentrates on the global one, and padding a large
  bucket costs ~`√count` rows (*estimated*, balls in bins).
- **Local kernel.** Within each device, order its rows by degree bucket as well. Each bucket is a dense
  `(rows, k)` block, its sum a gather, a multiply and a reduction along the width,
  `out[i] = Σ_w d[w,i]·v[j[w,i]]`, and `out` is the **concatenation** of the buckets' outputs: no scatter at
  all, contiguous writes, SELL-C-σ for free (Kreutzer et al. 2014).
- **Collective, first form.** One all-gather of `vec`, as `"indices"`: the simplest, `"indices"`' traffic.
- **Collective, second form: the halo.** Each device reads only the sources its rows' entries name. At build
  time list them per owner, deduplicated; per matvec one padded `all_to_all` sends each device exactly those
  values (`poc/hash_partition_jax` already runs a padded `all_to_all`). Traffic is bounded as in §1:
  ≤ ~0.10 N / 0.008 N (`type1`, `P = 4` / `64`) and ≤ ~0.75 N / 0.11 N (`type2`), with no `N`-length buffer
  and no grid or balance search. It **subsumes the push matvec** (the owner of `v_j` sends `d·v_j` to the
  owner of `i`, `poc/partition-states.md` §12 item 5, after Westerhout and Chamberlain, arXiv:2308.16712):
  with sender-side combining by target, push moves the same volume. The earlier "2–12× *more* than an
  all-gather at `P = 4`" for push used pre-drop degrees; post-drop it is ~0.13× (`type1`) to ~1.8× (`type2`)
  before deduplication (*estimated*, `deg/P`).
- **Cost.** States are laid out in hash order, so `sqd` un-permutes `eigvec` and the basis on return and the
  filler mask follows the permutation. Single-device ELLC's 19 scans cost ~+209 MiB of compile memory.
- **Risks.** Tail buckets (degree 60–117 pre-drop, few rows) have `count/P ≲ 1` at `P = 64`, so padding them
  to the largest device can reach several times their size, on the widest rows: merge the tail into a few
  coarser buckets per device. And on molecules (~620 per row, §3) the halo approaches the all-gather, which
  is rank 3's case.

### 4.3 2-D hashed layout, ELLC within each block (rank 3)

The classic distributed sparse-matvec layout: the `P` devices as a `P_r × P_c` grid (`√P × √P` when square)
with mesh axes `('r', 'c')`.

- **Ownership.** Hash each state to a block (§3's whole-key hash), and give device `(R, C)` every directed
  entry `(i, j, d)` with `i` in row block `R` and `j` in column block `C`. Directed storage is required: a
  symmetric pair would need its transposed block's vector and output too, the extra collectives 2-D exists
  to save.
- **Matvec.** All-gather `vec` along `'r'` only, so each device holds its column block (`N/P_c` values); run
  the local kernel with block-local indices; `psum_scatter` along `'c'` only, summing the partial results for
  row block `R` and splitting them back to `P(('r', 'c'))`.
- **Local kernel.** Today's `_apply_pairs` on the block's entries is the simplest; rank 2's ELLC, rows ordered
  by bucket, is the fast form, unchanged inside a block.
- **Build.** Owner bucketing keyed by the pair `(block(i), block(j))`, each device's list padded to the
  largest device's size class, counts from the shared allgather.
- **Balance.** The hash makes each block's entry count balls in bins, the standard reason 2-D layouts use a
  random or hash assignment on irregular graphs (Boman et al., SC 2013).
- **Traffic.** ~`N/P_c + N/P_r` per device per vector, `√P` less than rank 1's two full collectives: 3–9×
  less at `P = 4`–`64`, below `"indices"` from `P = 4`, with buffers of `N/√P` (*estimated*, §1). It needs no
  locality and does not grow with the degree, so it is the molecular layout; on spin chains rank 2's halo
  beats it except at `type2` `P = 4` (0.5 N against ≤ 0.75 N).
- **Cost.** Two collectives on sub-axes; on one node it should about tie rank 2's all-gather form, since
  communication is cheap there. Directed entries, twice the symmetric count. States in hash order, as rank 2.
  A non-square `P_r × P_c` works at `N/P_c + N/P_r`.
- **Risk.** Each block's entry count under one shared shape, directed entries split `P_r × P_c` ways: §8 step
  2 measures it.

### 4.4 Recomputed factors (rank 4)

Store `(j, group)` per entry (5–6 B) rather than `(j, d)` (20 B), and compute `d` from `states[i]`'s Z parities
by popcount, the way `"indices"` computes its diagonal but only on hits: the sparse twin of the
`"tables"`/`"indices"` trade. `_entry_factors` reads `states[target]`, so a device needs only the rows it
owns, `13·N/P` of `states` (`13·N/P_r` in rank 3), and the matvec needs no replicated `states`.

- **Memory, *estimated* on post-drop degrees** (§3): ~2.6–3.1 B/slot (`type1`) and ~36–43 B/slot (`type2`),
  plus ELLC's padding, against `"indices"`' `4·J` = 248 B/slot (`J = 62`) and 480 B/slot (`J = 120`):
  ~80–95× and ~11–13× less. With stored factors (20 B) it is ~10 and ~143 B/slot. (An earlier draft's
  22–70 B/slot used pre-drop degrees.)
- **Speed: a memory dial, not a speedup.** Shrinking the factor already measured little: `uint8` factor
  codes, 16 B → 1 B, bought 1.03–1.09× on a GH200 (`poc/sparse/prune.md` §6), and the gather is
  latency-bound (`markdown/spinchain/rqutils-multiobs-response.md` §3). Expect ~1.0–1.09×, either way.

### 4.5 Ring-pipelined all-gather (rank 5)

Replace the all-gather by `P − 1` `ppermute` steps around a ring, computing on each arriving block of `vec`
while the next is in flight: the same traffic as an all-gather, but the receive buffer falls from `N` to
`2N/P` and the communication hides behind compute. It needs the entries bucketed by **source** owner as well,
so ELLC's 19 scans multiply by `P`; worth it only across nodes, with directed pairs (rank 6) as the local
kernel. Rank 2's halo moves less at spin-chain degrees, so this is for molecular degrees, beside rank 3.

### 4.6 Row-owned directed pairs, or plain ELL (rank 6)

The local-kernel fallbacks for rank 2 if ELLC's per-device buckets do not pay.

- **Storage.** Device `p` stores `(i_local, j, d)` for every transition whose row `i` it owns, **both
  directions** of each symmetric pair, so every update lands in an owned row: no cross-device reduction,
  `out` stays `P('x')`.
- **Matvec.** Rank 2's collective, then today's chunked gather and scatter-add, locally. The per-entry
  arithmetic is unchanged: the single-device kernel already applies two updates per stored pair.
- **Build.** From the symmetric pairs the build already makes, emit both directions and bucket by owner, each
  device's list padded to the largest one's size class.
- **Memory, estimated on operators** (post-drop, *measured* `poc/sparse/prune.md` §6): `type1` `2^20` stores
  28.0 MiB, doubled ~56 MiB against `"indices"`' 248 MiB of source indices (`4·62·2^20` B), ~0.23×; `type2`
  `2^20` 124.0 MiB, doubled ~248 against 480 MiB, ~0.52×. (An earlier draft's "~0.6× → ~1.2×" applied the
  whole-solve *peak* ratio, mostly solver vectors, to the operator.)
- **Risk.** Hit rates differ by row, so entry counts are uneven; the hash balances them.

Three target-ordered shapes for it:

| variant | shape per device | padding | multi-process count exchange |
| --- | --- | --- | --- |
| directed pairs, CSR-like | ragged, padded to the largest device's size class | per device | yes |
| plain ELL | `(width, N/P)`, sharded on `N` with `P('x')` like `"indices"`' table | per row, to the global max width | yes, one scalar (the global max width) |
| ELL bucketed per device | each device's rows bucketed by width | per bucket | yes, per bucket |

Plain ELL is the simplest to shard: a fixed shape once the global width is agreed. Its cost is per-row
padding, and the tuned `"ell"`'s fix, bucketing rows by width, reorders rows across the whole vector and
breaks row ownership; under a mesh it must bucket within each device's rows instead, which is rank 2.

### 4.7 Group-major directed lists (rank 7)

One X group is a perfect matching (a state occurs in at most one pair per group, `poc/sparse/pairs.md` §7
item 6), so store each group's directed entries per device and let each group's scatter declare
`unique_indices=True`: no atomics. Under rank 2's hash each group's per-device count is balls in bins, so
padding is small. The cost is `J` scans per matvec, and single-device the atomic-free scatter was mixed with
no switch point (*measured*, `poc/sparse/tune.md` §3, against `base@2^19`): `type1` 0.67× / 1.07× / 2.26× over
`2^20`–`2^22`, `type2` the reverse, 1.63× / 1.73× / 0.78×. The fallback if ELLC's bucket compile cost hurts.

### 4.8 The product `S_α × S_β` layout, molecules only (rank 8)

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

### 4.9 GF(2)-linear partition, picked by randomized search (rank 9)

Ranked below rank 2: rank 2's halo reaches comparable traffic on XXZ with no balance search, so build this
only if the halo still dominates on a local `H`.

**The partition.** Choose a sparse binary matrix `A` with `r = log2 P` rows and give state `s` to device `A·s`
(mod 2). Linearity gives, for an X group with signature `x`,

    owner(s ⊕ x) = A·s ⊕ A·x = owner(s) ⊕ Δ_x,   Δ_x = A·x fixed per group

so every transition of a group shifts the device by the same `Δ_x`, whatever the state:

- **Groups with `Δ_x = 0` never communicate**, and their pairs can stay symmetric (rank 10 for free).
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
  `P = 64` (6 rows) about 18 of `type2`'s 120 groups and 12 of `type1`'s 62 cross (15–19%), each device talks
  to ~6 neighbours (a hypercube), and receives ~`6N/P` values against the all-gather's `63N/P`: ~10× less
  traffic and per-device memory, with no `N`-length vector anywhere. Balance is the open risk.
- Molecular JW: a sparse row has odd overlap with about half of the many-orbital X signatures, so most groups
  cross and the cost falls back to about an all-gather. The win is for local Hamiltonians.

**Layered across nodes** (an earlier rank of its own): rank 9's `A·s` picks the node and a random hash the
device within it, so inter-node traffic follows the `ppermute` pattern while each node runs rank 2. Same
condition: only if rank 2's halo dominates across nodes.

**What it costs the library.** States are laid out by owner, not in global lex order: the build's
`get_xsource` search still runs on the replicated sorted `states`, only the matvec layout changes.
`ground_locg` takes only inner products, so a permuted order is fine; `sqd` must un-permute `eigvec` and the
basis before returning them, and the initial vector's filler mask must follow the permutation. Device blocks
are padded to the largest, so imbalance is paid directly.

### 4.10 Hybrid symmetric/directed storage (rank 10)

Keep a pair symmetric when both endpoints live on one device, directed only when it crosses: recovers up to
half of rank 6's doubling with no reduce-scatter. The saving depends on locality: range-split locality is
hop-dependent on XXZ (`poc/partition-states.md` §4), and a random hash (ranks 2–3) has none, so it comes
free only under rank 9. Measure before building.

## 5. Rejected

- **Ownership by a random hash of the left half** (*measured*, §7). The aim was randomization for balance and
  locality together: every group flipping only right-half qubits stays on its device (2/3 of the bonds at
  cut 20), with no linearity constraint like rank 9's. **Dead beyond `P = 4`**: states near Néel share a few
  left halves, one alone owning 3–27% of the subspace. Largest device load over the mean, `2^17` and `2^20`:

  | | `P = 4` | `P = 16` | `P = 64` |
  | --- | --- | --- | --- |
  | cut 30 | 1.05–1.14× | 1.38–1.83× | 3.11–4.22× |
  | cut 20 | 1.24–1.50× | 2.46–2.71× | 7.95–8.39× |
  | cut 10 | 1.77–1.96× | 4.22–4.68× | 15.51–17.19× |

  Splitting a heavy left half by a few right-half bits rebalances, but every group flipping those bits then
  crosses devices, giving the locality back. The concentration near Néel that kills this and rank 8 on
  spinchain is the one that makes a range split fail (`poc/partition-states.md` §4), so ranks 2, 3 and 9 stay
  the spinchain candidates.
- **Randomizing the matvec itself** (sampled entries, sketches). `sqd` needs an exact matvec: its convergence
  at `rtol ≈ 4·eps` and the independent residual recomputed after every solve (`EigenpairCheckError`) would
  both break, as the closed `Ax`-reuse investigation showed for a far milder inexactness (`CLAUDE.md`,
  "Reusing `Ax` to cut `body()`'s 3 matvecs to 2"). The same rules out reduced-precision or lossy-compressed
  exchange.
- **2.5-D / 1.5-D vector replication**: it pays for many right-hand sides; here there are 2.
- **Hypergraph partitioning**: the transition graphs are hypercube-like (reverse Cuthill–McKee found
  nothing, `poc/sparse/pairs.md` §4), the subspace changes every recovery round, and rank 9 is the
  structure-aware version.
- **Symmetric storage in 2-D**: needs the transposed block's vector and output, another all-gather and
  reduce-scatter pair, giving back the traffic 2-D exists to save.
- **s-step / matrix-powers for the prefilter**: ghost zones fill an expander-like graph within a few hops.
- **HYB and a windowed `segment_sum`**: lost single-device (§3).
- **`float64` factors for real groups**: −24–27% of the operator for 0.37–0.94× on a GH200
  (`poc/sparse/tune.md` §3); on CPU −25–30% at ~10% slower in cache (`poc/sparse/pairs.md` §4 item 5). Rank 4
  saves more.
- **Delta-encoded `j − i`**: a high-bit flip moves an index by up to half of `N`
  (`markdown/spinchain/rqutils-multiobs-response.md` §2), so deltas do not fit a narrow type.
- **A sampled degree histogram for the bucket grid**: speeds only the build, and ELLC's fixed ×1.25 grid needs
  none.

## 6. Prediction: the fastest sparse mesh matvec

*Predicted, not measured.* **Rank 2: hashed rows, ELLC with rows ordered by bucket, the halo exchange.** On
one node it about ties rank 3 and rank 2's all-gather form, since communication is cheap there; across
nodes the halo's spin-chain traffic (≤ 0.008–0.11 N at `P = 64`) undercuts rank 3's 0.22 N. On molecules,
degree in the hundreds, rank 3 takes over.

**Why ELL-like rows over directed pairs.**

- **`"pairs"`' single-device edge over `"ell"` is gone under a row-owned layout.** It stores each transition
  once and writes both ends, half ELL's both-directions entries; every row-owned design stores both
  directions, rank 6's directed pairs included, so the entry counts are equal. (Rank 1 keeps the symmetric
  half by paying a reduce-scatter instead.)
- **ELL needs no scatter-add.** By target row it is gathers and a reduction along the width, and with rows
  ordered by bucket the output is a concatenation, no scatter at all. `"pairs"`' GPU trouble was its
  scatter: atomics serialized by padding on one row (0.32–0.73×, `poc/sparse/prune.md` §6), nondeterministic
  iteration counts (`poc/sparse/gpu.md` §5), the carry split (`poc/sparse/split.md`). Directed pairs keep a
  scatter-add, or need a sorted segment sum.
- **The single-device evidence is stale.** With twice the entries, the tuned `"ell"` beat `"pairs"` per
  iteration in two of four GH200 cells, 1.50× at `type1` `2^22` and 1.25× at `type2` `2^20` (*measured*,
  `poc/sparse/tune.md` §3), but against a `"pairs"` before its zero-drop and padding fix, which made it
  1.85–5.46× faster (§1). **Remeasure** ELLC against today's `"pairs"` before this steers anything.
- **Bucketed rather than plain**, because row widths vary, and plain ELL pads every row to the global maximum.

**When communication takes over** (*estimated* arithmetic). At `N = 2^24` one `complex128` vector is 256 MB.

- **Within a node**, NVLink at hundreds of GB/s: ~0.3–1 ms per all-gather, under a matvec's local compute at
  that size, so the local kernel decides.
- **Across nodes**, InfiniBand at ~25–50 GB/s: ~5–10 ms per vector, matching or exceeding the compute, with
  per-device memory capped by the full gathered vector. There rank 2's halo moves ~0.008–0.11 N at `P = 64`
  on spin chains, rank 3's sub-axis collectives ~0.22 N for any `H`, and rank 9 ~0.09 N if its balance holds.
- **On a CPU mesh** ELL's regular gathers vectorize well: close to directed pairs, possibly still behind
  `"tables"`, which gathers no factors.

**What would prove it wrong.**

- **ELLC against today's `"pairs"`** single-device: the ELL evidence above predates the zero-drop.
- **The post-drop degree histogram** (§8 step 2): heavy tails make ELL lose to padding even bucketed, and
  directed pairs (rank 6) win; tail buckets at large `P` especially (§4.2).
- **Per-bucket and per-block padding under one shared shape**, if the hash concentrates less than balls in
  bins.
- **XLA's lowering of the width reduction on a GPU** may not fuse as well as the scatter path; the
  single-device `"ell"`'s 4.8 s build hints at compile cost.
- **The halo's real volume after deduplication**, against its bound.

## 7. The script behind §4.8 and §5

`poc/split_layouts.py`, host-only, against its argparse: `--num-qubits` (60), `--product-sizes` (`14 17 20`)
and `--product-cuts` (`30 20`) for §4.8, `--hash-sizes` (`17 20`), `--hash-cuts` (`30 20 10`) and
`--devices` (`4 16 64`) for §5's left-half hash. Fixture: `poc/eigenpair_check_scale`'s Hamming-shell
subspaces around both Néel states; the left half is a state's first `cut` columns; the hash is a random salt
per distinct left half, drawn in loop order from one seeded generator, so a run reproduces the tables. Run at
`49366a5` plus the script (committed in `9c1047e`), on an Apple M1, 2026-10-05, with the defaults; the review
re-ran it and both tables reproduced exactly.

## 8. Plan, with a gate at each step

1. **Rank 1 on virtual CPU devices** (me) — **done 2026-10-06**: `_sparse_operator(..., mesh)` and
   `_apply_pairs_mesh` (`rqutils/sqd/_sparse.py`), `test/sharded/pairs_mesh.py` (`TestShardedPairs`) and `"pairs"` in
   `sqd_grid.py`. Eigenvalues within 1e-12 of single-device, the batched product within 1.8e-15, entries
   `P('x', None, None)`, exactly one all-gather and one reduce-scatter, per-device entries within one at every
   remainder. Mutants killed: no reduce-scatter (the residual check raises), replicated entries, duplicated
   slices, every entry on one device, the `ceil` split. Reviewed by `/simplify`, a complexity pass and two
   `/code-review` runs (one finding, the remainder, fixed; the second clean, also on a `(2, 2)` mesh and a
   Hamiltonian with no off-diagonal pairs). Not done: the iteration count, which `sqd` does not return. Each
   process still builds the whole host operator. Then
   **multi-GPU** (you): whole `sqd` calls against `"indices"` under the same mesh, and a multi-process run,
   since virtual devices cannot reach the non-addressable-shard class of errors (`CLAUDE.md`, "Sharding
   tests"). Gate: ≥ `"indices"`' speed at ≤ its per-device memory. If it passes and one node is the target,
   stop here.
2. **Single-device and host-only measurements** (me, CPU, no library change), from today's `"pairs"` build on
   XXZ `type1`/`type2` at `n = 60` and molecular-like `n = 14–20`, for `P ∈ {4, 16, 64}`:
   - ELLC against today's `"pairs"`, single-device, so the §6 prediction rests on post-drop evidence;
   - the post-drop degree histogram, and ELLC's post-drop bucket padding;
   - rank 2: per-bucket padding when every device must hold one shape, under `mix64` against today's
     lex-ordered row blocks, tail buckets merged; and the halo's deduplicated receive volume per device;
   - rank 3: each block's directed entry count against the mean, for `P_r × P_c ∈ {2×2, 4×4, 8×8}`;
   - rank 4: bytes per slot;
   - rank 6: per-device entry imbalance, and plain ELL's `width · N` against the stored entries;
   - rank 10: the same-device fraction under a range split;
   - rank 9: the search's imbalance, same-device fraction, distinct `Δ`, and receive volume.

   Gates: rank 2 with ELLC if its padding stays ≤ 1.25× and ELLC is at least today's `"pairs"`' speed; else
   rank 2 with rank 6's directed pairs as the local kernel. Rank 3 for molecules if its block imbalance stays
   ≤ 1.25×. Rank 9 only at imbalance ≤ 1.25 and ≥ 2× less receive volume than rank 2's halo on XXZ.
3. **The step 2 winner** in the library, behind the same sharded test, all-gather form first, then the halo,
   then the same multi-GPU gate against rank 1.
4. **Ranks 5 and 9** only if a multi-node run shows rank 2's or rank 3's collectives dominating.

## 9. Decisions for you

1. **Merge rank 1 to `dev`?** *Decided: merged, `f5a981a`.* It lifts `"pairs"`' single-device restriction, with `CLAUDE.md` and the
   docs updated, at no single-device cost (§4.1); it is untimed on real devices and unrun multi-process. Ranks
   2, 3, 9 and 10 would also change the state order the solver sees (rank 6 keeps lex order unless it takes
   rank 2's hash).
2. **Rank 1 first, or straight to step 2's measurements?** Rank 1 is the cheapest real measurement; step 2
   decides between the layouts without a GPU, and re-grounds the ELL evidence.
3. **Molecules.** Rank 3 needs no locality and does not grow with the degree, so it is the molecular layout;
   rank 2's halo is for spin chains. If molecular `J` is the target, rank 4's recomputed factors or rank 8's
   product layout may matter more than any partition.

## 10. Review record

An independent review on 2026-10-05 checked every measured figure against its source and redid the
arithmetic. Corrected here: the stale ELL comparison (§3, §6); the pre-drop degrees behind rank 4's bytes and
the push estimate (§3, §4.2, §4.4); rank 6's memory, which had used a peak ratio (§4.6); rank 1 described as a
degenerate 2-D layout (§4.1); rank 1's predicted gain, now the large-`N` end (§4.1); the residual check's need
for the global `pairs`, and the missing count exchange (§4 preamble); plain ELL's count exchange (§4.6); the
atomic-free scatter's `type2` cells (§4.7); the `float64` sources (§5); rank 9's crossing fraction (§4.9);
inconsistent "tie" claims (§4.3, §6); "no scatter" where `.at[rows].set` is one (§6). Added: the halo
exchange (rank 2, absorbing the push matvec), the ring-pipelined all-gather (rank 5), rows ordered by bucket,
the initial vector in lex order, `mix64`, the shared count allgather, tail-bucket merging, and the rejected
2.5-D, hypergraph, symmetric-2-D and s-step forms. Verified correct: the §1 traffic arithmetic, 256 MB, the
link times, `4·J`, the 2-D collective pattern and its need for directed storage, the GF(2) owner shift, the
perfect matching, rank 1's reduce-scatter, rank 4 needing only owned rows of `states`, and every other
measured figure.

**Rank 1's build, 2026-10-06.** `/simplify`'s four passes moved the device split into `_compact`, so the mesh
path shares the zero-drop's padding and count, made `_apply_pairs_mesh` its own function rather than a rank
check inside `_apply_pairs`, and ran the sharded child once instead of three times. A complexity pass cut 17
lines. The first `/code-review` found the `ceil` split's short last device (§4.1); the second found nothing.
