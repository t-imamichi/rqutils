# `"pairs"` without its exact-zero entries, and with coded factors

`poc/sparse/prune.py` (§8) at `0c137dd` plus the change, one Apple M1 (8 cores, 16 GiB), 2026-10-02;
the GH200 in §6.
Fixture as `poc/sparse/gpu.md`: spinchain's open-XXZ `xxz` at n=60, `δ = 0.5`, `type1` (`J = 62`) and
`type2` (`J = 120`), Hamming-shell subspaces around both Néel states. In the library since, as
`_drop_zeros`; the factor codes (§4) are not.

## 1. The zeros

An XX+YY hop's factor is `c_XX + c_YY·(-1)^{s_i ⊕ s_{i+1}}`, exactly zero on aligned spins (`00 ↔ 11`):
the search finds those pairs, and `"pairs"` stored them with a zero factor. Measured on the built operator:

| pattern | stored pairs at `2^17` | exactly zero | distinct factors |
| --- | --- | --- | --- |
| `type1`, `type4` | 249,104 | 86.5% (85.7% at `2^19`) | 4 |
| `type2`, `type3` | 682,558 | 31.6% (31.7% at `2^19`) | 4 (`type3` 5) |

The distinct values are `{0, ±0.5, 0.5i}` (`type1`) and `{0, ±0.5, −0.5+0.5i}` (`type2`), held as
`complex128`.

## 2. The filter

`_drop_zeros` runs after `_entry_factors`, on the sorted chunks: keep the entries with `d != 0` in order
(`jnp.nonzero` with a host-counted size), re-pad to a size class with the padding convention (`i = j =
size - 1`, `d = 0`). So the kept entries and their factors are exactly the unfiltered build's. The
residual check keeps reading the unfiltered `pairs`, so it vouches for the filter rather than sharing it.

## 3. Results

`all` patches the filter out; 3–5 interleaved rounds after a warm-up, every round won:

| pattern, N | stored pairs | operator | build, warm | per solve | iterations | eigenvalue diff |
| --- | --- | --- | --- | --- | --- | --- |
| `type1` `2^17` | 249,104 → 33,722 | 8.0 → 3.5 MiB | 17 → 19 ms | **1.53–1.55×** | 106 = 106 | 5.3e-15 |
| `type1` `2^19` | 1,149,220 → 164,655 | 35.0 → 12.5 MiB | 73 → 85 ms | **1.59×** | 64 = 64 | 5.3e-15 |
| `type2` `2^17` | 682,558 → 467,176 | 18.5 → 13.2 MiB | 37 → 48 ms | **1.33×** | 106 = 106 | 7.1e-15 |
| `type2` `2^19` | 3,103,160 → 2,118,595 | 80.0 → 62.0 MiB | 169 → 222 ms | **1.18–1.19×** | 103 = 103 | 3.6e-15 |

The solve's temp (18 / 72 MiB) is unchanged: it is the solver's vectors, not the operator. A first build
at a new size pays 0.15–0.36 s more, compiling the filter's eager ops; warm it is 2–53 ms, against
0.35–1.46 s saved per solve.

## 4. Factor codes

The `codes` arm stores each factor as an index into the operator's distinct values (4–5 here, §1), read
back as `table[code]` in the scan. One path for every Hamiltonian: `jnp.unique` builds the table, the
code is the narrowest unsigned type that fits (`np.min_scalar_type`: `uint8` here, so `d` 16 → 1 B per
pair), and the table is padded to 256 entries, or a size class above, so the solve compiles per class
rather than per count. Past ~256 distinct values the worst case is a 4 B code plus a 16 B table entry
per factor, about 1.25× today's bytes. Against `nonzero`, 3 rounds, bit-identical eigenvalue and
eigenvector:

| pattern, N | operator | per solve | build, warm |
| --- | --- | --- | --- |
| `type1` `2^17` | 3.5 → 2.6 MiB | 1.00× (0.652 against 0.651 s) | 19 → 28 ms |
| `type1` `2^19` | 12.5 → 9.7 MiB | 1.00× | 85 → 107 ms |
| `type2` `2^17` | 13.2 → 6.2 MiB | 1.01× | 49 → 109 ms |
| `type2` `2^19` | 62.0 → 28.3 MiB | 1.01× | 219 → 544 ms |

The build's extra is `jnp.unique`'s complex sort; a host `np.unique`, as a first version of the arm
did, cost 22–102 ms instead of 9–325 ms.

A memory lever only on CPU: −22% of the operator where `d0` and the indices dominate (`type1`), −54%
where the factors do (`type2`), at no solve cost. At `2^15` it read 0.77× in one round, unconfirmed.
Not shipped, for no CPU speed: the GPU, where a pair's 24 bytes feed a bandwidth-bound scatter, is what
would decide it. In the library it would be the arm's `encode` (4 lines) after `_drop_zeros` and one
`table[code]` line in `_apply_pairs`.

## 5. What it means

The time falls less than the entries (7× fewer at `type1` for 1.6×) because the solve's `O(N)` vector
work stays. Not bit-identical, unlike the expectation going in: dropping entries moves chunk boundaries,
and a chunk adds all its `out[i]` updates before its `out[j]`, so a row's terms are summed in another
order. The iteration counts match exactly.

## 6. On the GPU: the padding all on one row

One NVIDIA GH200 120GB, 2026-10-03, 5 rounds, every ratio 5/5 unless below 1. Per solve against `all`, at
`c0061df` (before the fix below) and at `d0a9997` (after):

| pattern, N | stored pairs | operator | `nonzero` before | `nonzero` after | fix alone | `codes` after |
| --- | --- | --- | --- | --- | --- | --- |
| `type1` `2^20` | 4,182,753 → 339,727 | 112.0 → 28.0 MiB | **0.38×** | 2.28× | **5.96×** | 2.34× |
| `type1` `2^22` | 9,909,391 → 1,912,744 | 304.0 → 112.0 MiB | 2.45× | 5.46× | **2.25×** | 5.70× |
| `type2` `2^20` | 8,165,072 → 4,322,046 | 208.0 → 124.0 MiB | **0.73×** | 3.69× | **4.95×** | 3.99× |
| `type2` `2^22` | 29,095,965 → 21,099,318 | 736.0 → 592.0 MiB | **0.32×** | 1.85× | **5.74×** | 2.02× |

"Fix alone" is `nonzero`'s own solve time before over after: 0.405 → 0.068 s, 0.774 → 0.344 s, 1.407 →
0.284 s and 5.624 → 0.980 s. Iteration counts within one, eigenvalues within 3.6e-15 throughout.

Before the fix, fewer entries ran *slower* in three of four cells. `_drop_zeros` re-padded with
`i = j = size - 1`, so every padding entry's two scatter-adds hit the same `out[size - 1]`, and a GPU
serializes atomics on one address. Rounding to `2^19`-entry chunks and a size class left ~185k such
entries at `type1` `2^20` and ~2.0M at `type2` `2^22`. The fix puts padding entry `k` on row `k mod size`,
still with equal endpoints and a zero factor, so it adds `0` to a distinct row; values are unchanged up to
the sign of a zero. That it recovered 2.25–5.96× confirms the cause, though no profile was taken. On the
M1 at `2^17` (3 rounds) it is harmless: `nonzero` 1.73× `type1` and 1.39× `type2` against `all`,
eigenvalue diffs 5.3e-15 and 7.1e-15 as in §3.

`all`'s own build padding (~11k–264k entries, from `_padded`) still sits on row `size - 1`, since that arm
skips `_drop_zeros`. So the "after" column overstates what dropping zeros buys by itself; the library
always drops, so "fix alone" is the figure that describes it.

**`codes` wins on the GPU once the padding is gone**: 1.03–1.09× `nonzero` (2.34/2.28 to 2.02/1.85), for
−27% (`type1`) to −56% (`type2`) of the operator. Before the fix it was 1.02–1.06×, inside the
contention's noise; on CPU it is 1.00–1.01× (§4).

## 7. Open

1. **Factor codes in the library** (§4, §6): 1.03–1.09× and up to −56% of the operator on a GH200,
   bit-identical on CPU. The build's `jnp.unique` sort costs 5–89 ms more than `nonzero`'s (0.024 → 0.029 s
   at `type1` `2^20`, 0.090 → 0.179 s at `type2` `2^22`), against 2–83 ms saved per solve, so it pays
   only over a solve or two at the larger sizes, more if one operator serves several. **Closed
   (2026-10-05)**: declined, with the sort-free variant in `markdown/parity-codes-proposal.md`, to keep the
   solver generic; codes pay only with few distinct factors and small `kmax`, as in spinchain's XXZ.
2. **The first-build compile: done.** The filter is one jitted program, `_compact`, 3.4–3.8× its eager
   first call on CPU, 0.23–0.36 s off a first build (`poc/sparse/drop-jit.md`).

## 8. The script

`poc/sparse/prune.py`, its argparse checked against this section:

| flag | default | meaning |
| --- | --- | --- |
| `--num-qubits`, `--delta` | `60`, `0.5` | the `xxz` fixture |
| `--patterns` | `type1 type2` | from `poc/eigenpair_check_scale.patterns` |
| `--log2-sizes` | `17 19` | subspace sizes `2^k` |
| `--rounds` | `5` | interleaved solves per arm after one warm-up |

One host search serves every arm; each build runs twice, the second timed. Eigenvalues must agree to
`1e-12`, and `codes` must equal `nonzero` bit for bit on CPU only: a GPU's atomic scatter-add
sums in no fixed order, which tripped that check on the GH200's first run. Runs here: two arms at the default (5 rounds), then
`--rounds 3` (§3's build column), then all three arms at `--rounds 3`, with the `encode` described in §4. §6: `--log2-sizes 20 22` on the
GH200 at `c0061df` and again at `d0a9997`, and `--log2-sizes 17 --rounds 3` on the M1 after the fix.
