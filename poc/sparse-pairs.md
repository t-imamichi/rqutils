# Sparse transition pairs

Item 8 of `markdown/improvement-ideas-2026-09-25.md`, prototyped on branch `sparse-pairs`, one laptop CPU
(10 cores, 64 GiB), 2026-09-25/26. Nothing here is in the library yet. The GPU is the open question.
Every number below comes from `poc/sparse_pairs.py` (§8).

## The idea

At `cache_level=(1, *)`, `sqd` stores one `int32` source index per `(X group, state)`: `4·J` B/slot, the
largest term of a `(1, 0)` solve at large `n`. On spinchain's open-XXZ Hamiltonians with Hamming-shell
subspaces only **8–22%** of those slots are real transitions; the rest are `−1`. XOR is an involution, so
each real transition of a group `g ≠ 0` is a pair `(i, j)`. Store each pair once, and compute its diagonal
factor once, since `H_ji = conj(H_ij)` holds *exactly* for one X signature (`conj(c_k s_k(i)) = c_k s_k(j)`
term by term). The identity-X group has no pairs and stays `d_0 ⊙ v`.

Fixtures are spinchain's `xxz` builder verbatim (`poc/eigenpair_check_scale.py`) at `δ = 0.5` with
Hamming-shell subspaces around both Néel states: n=60 `type2` (`J = 120`, uniform `Bx`) and `type1`
(`J = 32` at n=30, 62 at n=60, the pattern every shipped config uses); §5 adds `type3`/`type4` and a
subspace grown the way spinchain's recovery grows one. Every arm is checked against the
`(1, 0)` product (or its relabelled rows) before timing; every solve returned the same eigenvalue.

## The arms

| arm | layout | diagonal | stored per pair |
| --- | --- | --- | --- |
| `(1, 0)` / `(1, 2)` | dense `(J, N)` source indices, one gather per group | recomputed / cached | `4·J` / `20·J` B/slot |
| **P0** / **P2** | pairs `(i, j, g)`, two scatters | recomputed / cached per pair | 12 / 24 B |
| **C0** / **C2** | both directions `(t, s, g)` sorted by target (CSR), one sorted scatter | recomputed / cached per entry | 24 / 48 B |
| **C0i16** | C0 with `int16` group ids (`int32` if `J > 32768`) | recomputed | 20 B |
| **C2R** | C2 with `float64` diagonals for real groups, `complex128` only for the rest | cached | ~32 B |
| **seg** | C2 through one unchunked `segment_sum` | cached | as C2, plus temporaries |
| **+RCM** | a base arm after a reverse Cuthill–McKee relabelling of the states | as base | as base |

Pair and CSR arms scan fixed chunks of `2^15` entries, so kernel temporaries are `O(chunk)`; unchunked,
pair-sized temporaries put P0/P2 above `(1, 0)`'s memory (988 / 731 B/slot at `2^17`). Chunk `2^13` leaves
memory unchanged (the solver's own vectors dominate by then) and only costs speed. **`int16` wraps
silently** past `J = 32767` (`astype` turns 32768 into −32768, indexing another group's Z table), so the
width is chosen from `J` and an explicit narrow width asserts.

## 1. Whole solves at `2^17`

`ground_locg` driven as `run_sqd` drives it (same initial vector, prefilter `(32, 2)`, `Σ|c|` bound,
batched); memory is XLA `memory_analysis` of the whole solve (inputs + temp), per state slot:

| arm | n=60 `type2` | n=60 `type1` | n=30 `type1` |
| --- | --- | --- | --- |
| `(1, 0)` | 9.99 s, 666 B | 6.15 s, 434 B | 2.42 s, 309 B |
| `(1, 2)` | 3.93 s (2.5×), 2577 B | 2.08 s (3.0×), 1417 B | 1.24 s (2.0×), 817 B |
| P0 | **2.26 s (4.4×), 248 B (−63%)** | **1.16 s (5.3×), 209 B (−52%)** | 1.23 s (2.0×), 216 B (−30%) |
| P2 | **1.58 s (6.3×), 302 B (−55%)** | **0.92 s (6.7×), 224 B (−48%)** | 0.96 s (2.5×), 248 B (−20%) |
| C0i16 | 3.65 s (2.7×), 290 B (−56%) | 1.54 s (4.0×), 225 B (−48%) | 2.03 s (1.2×), 240 B (−22%) |
| C2R | 2.51 s (4.0×), 348 B (−48%) | 1.19 s (5.2×), 244 B (−44%) | 1.48 s (1.6×), 265 B (−14%) |

Iteration counts: 105–106 on `type2`; on n=60 `type1`, 96 for `(1, 0)`, `(1, 2)` and C0i16 against 105
for P0/P2/C2R — the pair kernels sum the same terms in another order, which shifts the trajectory, so
their per-iteration advantage is larger than the table shows. For scale, `(0, 0)` on the first column is
217 B/slot at **8.3× slower** than `(1, 0)`. The gain grows with the placeholder count `J·(1 − h)`: n=30
`type1`, with 32 groups, gains least.

## 2. Peak memory at scale, setup included (n=60 `type2`)

Each arm in a fresh process; on the CPU backend device memory is process memory, so this is peak RSS
above a post-startup baseline, pair construction included. `(1, 0)` is `run_sqd` itself.

| states | `(1, 0)` | P0 | P2 | C0i16 | C2R |
| --- | --- | --- | --- | --- | --- |
| `2^19` | 461 MiB | **297 (−36%)** | 313 (−32%) | 342 (−26%) | 404 (−12%) |
| `2^21` | 1278 MiB | **728 (−43%)** | 847 (−34%) | 940 (−26%) | 1234 (−3.5%) |

Getting here took three setup fixes, each measured:

- **Pairs are built one group at a time**, so only one group's `int32` row is transient — never the
  `(J, N)` array `(1, 0)` itself pays.
- **CSR is counting-sorted by target straight into padded chunks**: row counts, a prefix sum, then one
  conflict-free assignment per group (a row occurs at most once per group). The first version sorted with
  concatenations and an `int64` `argsort`, which put C0i16 at 471 MiB and C2R at 784 at `2^19` — at and
  well above `(1, 0)`.
- **No duplicate transients**: host pair arrays are freed once chunked when nothing else reads them, and
  C2R's real groups get `float64` diagonals directly from `c.real` rather than a complex array beside its
  `.real` copy (C2R at `2^21`: 1648 → 1234 MiB).

So **P0 is the memory choice** — it stores each pair once (89 operator B/slot at `2^21`, against C0i16's
144) — and C2R is a speed choice whose peak is about `(1, 0)`'s.

## 3. Why the speedup shrinks with N: cache locality, not threads

One `(2, N)` matvec, ns/state, n=60 `type2`:

| states | `(1, 0)` | P0 | P2 | C0 | C2 |
| --- | --- | --- | --- | --- | --- |
| `2^15` | 400 | 111 | 44 | 132 | 50 |
| `2^17` | 294 | 105 | 47 | 133 | 57 |
| `2^19` | 327 | 155 | 108 | 138 | 68 |
| `2^21` | 265 | 168 | 125 | 150 | 80 |

- **Not threading.** With XLA's CPU threading off (`--xla_cpu_multi_thread_eigen=false
  intra_op_parallelism_threads=1`) every number agrees within ~2%: neither kernel used the other cores.
- **Cache locality.** P2 more than doubles per state between `2^17` and `2^19` — where `vec` + `out`
  (64 B/state for a complex `(2, N)` batch) grow from 8 to 32 MiB, past the on-chip cache — while `(1, 0)`
  stays flat. `(1, 0)` reads `vec` at random too, but it recomputes all 120 groups' diagonals for every
  state, and that compute hides the latency. Pairs remove the compute, leaving a memory-bound kernel whose
  random `out[pj]` read-modify-writes miss.
- **CSR order recovers much of it**: sequential writes, only `vec[s]` reads random. Past the cache C2 is
  **1.6× faster than P2** (80 against 125 ns at `2^21`), at twice the entries.

## 4. Variants, both patterns

ns/state for one `(2, N)` matvec, and operator B/slot at `2^21` (solver vectors excluded):

| arm | `type2` `2^17` | `type2` `2^21` | `type2` B/slot | `type1` `2^17` | `type1` `2^21` | `type1` B/slot |
| --- | --- | --- | --- | --- | --- | --- |
| `(1, 0)` | 294 | 265 | 488 | 182 | 161 | 256 |
| P0 | 105 | 168 (1.6×) | **89** | 65 | 73 (2.2×) | **38** |
| P2 | 47 | 125 (2.1×) | 179 | 19 | 41 (3.9×) | 75 |
| C0 | 133 | 150 (1.8×) | 171 | 73 | 67 (2.4×) | 67 |
| C0i16 | 131 | 150 (1.8×) | 144 | 73 | 67 (2.4×) | 57 |
| C2 | 57 | 80 (3.3×) | 342 | 21 | 26 (6.2×) | 134 |
| **C2R** | 56 | **79 (3.4×)** | 236 | 20 | **26 (6.2×)** | 96 |
| seg | 53 | 70 (3.8×) | 341 + 411 temp | 20 | 23 (6.9×) | 134 + temp |
| C2 + RCM | 57 | 88 (3.0×) | 342 | 20 | 25 (6.5×) | 134 |

1. **`segment_sum` (seg): fastest, and not usable as is.** A sorted segment sum emits all `N` segments,
   so it cannot be chunked cheaply, and unchunked its temporaries are **411 B/slot** (XLA, `type2` `2^19`)
   against C2's 2 — more than `(1, 0)`'s whole operator. It needs a windowed form.
2. **`int16` group ids (C0i16): −16% of C0's operator for free**, speed unchanged.
3. **Real diagonals (C2R): −31% of C2's operator for free.** Only 2.2–4.3% of pairs sit in groups with
   complex coefficients (the end-site `Y` terms).
4. **Reverse Cuthill–McKee: nothing.** A Hamming-shell transition graph is hypercube-like — each state's
   neighbours span the whole index range — so there is no small bandwidth to find; lex order is as good.
5. **Real factors for pairs (P2R, 2026-09-27): −25–30% of the operator, ≤ 5% of the peak, ~10% slower in
   cache — not shipped.** C2R's split applied to P2: `float64` factors for real groups, two scans. At `2^21`
   the operator falls 75 → 56 (`type1`) and 179 → 126 B/slot (`type2`) at equal matvec speed; at `2^17` the
   second scan costs 10–17% per matvec and 9–10% per solve (0.93 → 1.02 s, 1.61 → 1.76 s). Whole-solve
   memory falls only 224 → 222 and 302 → 274 B/slot, since the solver's vectors dominate. Setup-inclusive
   peak (`type2`) is 326 → 310 MiB at `2^19` and 834 → 824 at `2^21` (−5% / −1%, within run-to-run noise)
   **when each set is filled group by group**; masking the two sets out of the full host pair list instead
   raised it to 934 MiB (+12%). Sharing one `(i, j)` array and slicing it inside the jit is worse still:
   XLA copies the slices, +18.5 / +48 B/slot of temp per matvec, so the whole solve rises to 238 / 316.

`type1` gains *more* past the cache than `type2` despite half the groups: with fewer groups the matvec is
less memory-bound, so C2R holds 6.2× at `2^21` where `type2`'s falls to 3.4×.

**Every figure above is batched, and batching never costs a pair arm** (`batch`: one `(2, N)` call against
two `(N,)` in one program, n=60 shells). Per matvec at `2^17`/`2^21` it is worth 1.98–2.15× to C2R,
1.54–1.77× to C0i16 and 1.55–1.75× to `(1, 0)`, but only 1.14–1.19× to P0 and 0.98–1.13× to P2: the pair
kernels' cost is their random scatters, which batching does not share. Whole solves at `2^17`
(`type1`/`type2`): C2R 1.27×/1.30×, C0i16 1.28×/1.34×, `(1, 0)` 1.27×/1.33×, `(1, 2)` 1.10×/1.09×, P0
1.02×/1.11×, P2 0.96×/1.03×. P2's 0.96× is iteration count, not speed: on `type1` the batched pair arms take
105 iterations against 96 unbatched (another summation order), and per iteration P2 is 1.04× and P0 1.11×
batched. Energies agree to 5.3e-15. **An `(N, 2)` layout**, putting both vectors' entries in one cache
line, measures 0.97–1.10× (best: P2 at `2^21`, 1.08–1.10×), so it does not recover the locality lost past
the cache.

## 5. Beyond these fixtures: the win tracks the hit rate

The pair form works for any Pauli Hamiltonian -- XOR is an involution for every X signature, `H_ji =
conj(H_ij)` for any Hermitian Pauli sum, and open vs periodic or 1D vs 2D only changes `J` -- but its gain
tracks the hit rate `h`, the fraction of `(group, state)` slots landing inside the subspace. Pairs cost
~`6·h·J` B/slot against `4·J`, so memory breaks even near `h ≈ 2/3`, and P0's work scales with `h` rather
than with every slot. Measured at the high-`h` end, operator bytes and matvec speed against `(1, 0)`,
interleaved (an earlier un-interleaved baseline read P0 at 0.89× on the molecular fixture; retracted):

| fixture | `J` | `h` | P0 | P2 | C2R |
| --- | --- | --- | --- | --- | --- |
| periodic XXZ n=24, Néel-Krylov (closed under hops) | 25 | 0.42 | −44%, 1.04× | −23%, 3.04× | **+46%**, 1.42× |
| molecular-like JW n=14, random states (71% of `2^14`) | 872 | 0.71 | −24%, 1.48× | ≈0%, 5.35× | **+100%**, 2.14× |

**At high `h` the CSR forms cost memory** — they store both directions — and P0's speed edge fades; P2 and
C2R stay faster (not compared against `(1, 2)` on these two rows). Sampled SQD subspaces at large `n`
(`10^5`–`10^7` states in a `2^n` space) are the low-`h` case.

**The other two patterns change nothing.** `type3` is `type2` plus end-site `By` and `type4` is `type1` plus
end-site `Bz`; neither adds an X signature, so `J` and `h` are their partners' and so are the numbers. At
`2^21`, `type3` gives P0 1.51× and C2R 3.27× (`type2`: 1.6×, 3.4×) and `type4` P0 2.12× and C2R 6.16×
(`type1`: 2.2×, 6.2×), with operator B/slot identical to the partner's. (One `type3` C2 cell read 137 ns;
re-run, it is 80.3, `type2`'s 80.)

**A recovery-grown subspace raises `h` less at n=60 than at n=20.** `--subspace recovery` grows the
subspace as spinchain's `recover_configurations` does: from a `2^12` Hamming-shell core, each round solves
`sqd`, scores H's one-hop reach by `|<c|H|v>|` and admits the top scorers, doubling it (scores checked
against a dense `H` at n=8 to ≤ 9e-16). It picks better rows, as it should — at n=12 and 512 states it
reaches −4.967 against the shells' −4.482 (exact −5.785) — and better rows are more coupled, so `h`
rises. At n=20 (`2^13`) it reaches 0.354, where the CSR forms already exceed `(1, 0)`'s memory (368–400
against 165 B/slot) and P0 still saves (101) at 1.57×. At n=60 it is only 1.3–1.8× the shells' `h`, at
`2^21`, ns/state (speedup) and operator B/slot:

| pattern | subspace | `h` | `(1, 0)` | P0 | P2 | C2R | B/slot `(1, 0)` / P0 / C2R |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `type1` | shells | 0.096 | 161 | 73 (2.2×) | 41 (3.9×) | 26 (6.2×) | 256 / 38 / 96 |
| `type1` | recovery | 0.170 | 196 | 163 (1.20×) | 122 (1.61×) | 69 (2.85×) | 256 / 65 / 169 |
| `type2` | shells | 0.121 | 265 | 168 (1.6×) | 125 (2.1×) | 79 (3.4×) | 488 / 89 / 236 |
| `type2` | recovery | 0.161 | 266 | 177 (1.51×) | 126 (2.12×) | 103 (2.58×) | 488 / 118 / 321 |

Within the cache (`2^17`) recovery still gives P0 2.40× / 2.75× and C2R 4.86× / 4.97× (`type1` /
`type2`, `h` 0.133 / 0.118). So on the subspace spinchain would actually build the memory win holds —
P0 −75%, C2R −34% of `(1, 0)`'s operator — while P0's speed past the cache thins to 1.20× on `type1`. Not
modelled: spinchain also prunes rows the eigenvector drives below `weight_tol`, freeing budget for more
coupled rows, which would push `h` further up, and its seed is Krylov samples in a Clifford frame, not
Hamming shells. Shell `h` for `type1`/`type2` is read off `type4`/`type3`, which share their X signatures.

## 6. What it means

- **For a memory-bound `(1, 0)` run** (spinchain at n ≥ 30, where `(1, 0)` already fills memory):
  **P0** is the memory choice — **−43% peak at `2^21`** (−52% whole-solve on n=60 `type1`) at 1.6–5.3×
  the speed. **C2R** is the speed choice — **3.4–6.2× past the cache, 4–5× whole solves** — with a peak
  about `(1, 0)`'s. C0i16 sits between them. At n=60 all beat `(1, 2)`, which buys its speed with 2.6–3.9× `(1, 0)`'s
  memory; at n=30 `type1` they only match it (C2R 1.48 s against 1.24 s), at a fraction of its memory.
  On a recovery-grown subspace (§5) the memory win holds (P0 −75%, C2R −34% of `(1, 0)`'s operator at
  `2^21`), but P0's speed past the cache falls to 1.20–1.51× and C2R's to 2.58–2.85×.
- **The partial diagonal cache (item 3) cannot help such a run**: it adds memory on top of `(1, 0)`.
- **CPU speedups at spinchain's sizes (N = 1.5M–20M) will be the `2^21` column or a little lower**, since
  past the cache the kernels are memory-bound. The GPU cannot be predicted from this: it has far more
  bandwidth, hides latency with many threads, and does scatters with atomics.

## 7. Open before a library version

1. **GPU timing** of P0 and C2R against `(1, 0)` and `(1, 2)` — the deciding measurement.
2. **Sharding.** A pair's endpoints can sit on different devices; CSR needs its sources gathered.
3. **API — done.** P2 and C2R ship as `sqd(matvec="pairs")` and `sqd(matvec="csr")` (§9); P0 and C0i16
   were not shipped.
4. **Pruned recovery and real samples.** §5's recovery subspace neither prunes nor starts from Krylov
   samples. Pruning pushes `h` up, and n=20's 0.354 shows how far `h` can go; a replayed spinchain run
   (`skqd/replay.py`) would give the true subspace.

## 8. The script

Everything above is `poc/sparse_pairs.py`, every arm built by one function (`operators`):

| subcommand | what it measures |
| --- | --- |
| `matvec` | ns/state of one `(2, N)` matvec per arm, each checked against `(1, 0)`; `XLA_FLAGS` for one thread |
| `solve` | whole `ground_locg` solves at one size, with XLA memory (inputs + temp) |
| `peak` | setup-inclusive peak RSS, one fresh process per arm and size |
| `general` | high hit rate: periodic Néel-Krylov XXZ and molecular-like JW |
| `batch` | one `(2, N)` call against two `(N,)` and an `(N, 2)` layout, then solves with `batch_matvec` on/off |

`--pattern type1`–`type4`, `--num-qubits` and `--subspace shells`/`recovery` select the fixture. A
recovery subspace is cached as `/tmp/sparse_pairs_recovery_n<n>_<pattern>_d<delta>_<size>.npy`, since
growing one to `2^21` runs `sqd` up to `2^20`.

## 9. In the library

`sqd(matvec="pairs"|"csr")`, 2026-09-26: built host-side in `sqd()` (per-group construction, the counting
sort, chunked factors, host arrays freed as they reach the device), chunk counts rounded to `m·2^k` with
8 ≤ m < 16 so the solve recompiles per size class, residual check on the `"onthefly"` kernel, single-device.
Warm `sqd` at n=60 `type1`, `2^17` (all −11.676532550657):

| `matvec` | `sqd` | construction | solve alone | POC solve |
| --- | --- | --- | --- | --- |
| `"indices"` | 6.25 s | — | — | 6.15 s |
| `"tables"` | 2.22 s | — | — | 2.08 s |
| `"pairs"` | 1.26 s | 0.17 s | 0.93 s | 0.92 s |
| `"csr"` | 1.63 s | 0.18 s | 1.27 s | 1.19 s |

The difference between `sqd` and the solve alone is construction plus the residual check (~0.15 s).
Peak RSS at `2^19`, n=60 `type2`, fresh processes: packed input 454 / 336 / 393 MiB for `"indices"` /
`"pairs"` / `"csr"` (POC: 461 / 313 / 404); unpacked input, which `sqd` packs itself, 627 / 451 / 469 MiB.

**Which split ships, and why the same trick gets opposite verdicts.** `"pairs"` is P2 and `"csr"` is C2R.
Real factors cut C2's operator 28–31% at no measured speed cost (§4, item 3), because CSR stores every
transition twice and so carries twice the factors; the same split on pairs (P2R, §4 item 5) halves as many
bytes, moves the peak only 1–5% (within noise), and costs 9–10% of an in-cache solve. Revisit P2R only if
a GPU run is bound by device memory, where the solve-memory row (up to −9%) is what counts.
