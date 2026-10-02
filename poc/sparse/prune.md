# `"pairs"` without its exact-zero entries

`poc/sparse/prune.py` (§5) at `0c137dd` plus the change, one Apple M1 (8 cores, 16 GiB), 2026-10-02.
Fixture as `poc/sparse/gpu.md`: spinchain's open-XXZ `xxz` at n=60, `δ = 0.5`, `type1` (`J = 62`) and
`type2` (`J = 120`), Hamming-shell subspaces around both Néel states. In the library since, as
`_drop_zeros`.

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

## 4. What it means

The time falls less than the entries (7× fewer at `type1` for 1.6×) because the solve's `O(N)` vector
work stays. Not bit-identical, unlike the expectation going in: dropping entries moves chunk boundaries,
and a chunk adds all its `out[i]` updates before its `out[j]`, so a row's terms are summed in another
order. The iteration counts match exactly.

## 5. Open

1. **The GPU**, where the scatter dominates the matvec (`poc/sparse/split.md` §4): the gain should be
   nearer the entry ratio. `uv run python poc/sparse/prune.py --log2-sizes 20 22`.
2. **Factor codes**: the 4–5 distinct values fit a `uint8` index into a table, `d` 16 → 1 B per pair.
3. **The first-build compile**: a jitted `_drop_zeros` keyed on the size class would compile once per class.

## 6. The script

`poc/sparse/prune.py`, its argparse checked against this section:

| flag | default | meaning |
| --- | --- | --- |
| `--num-qubits`, `--delta` | `60`, `0.5` | the `xxz` fixture |
| `--patterns` | `type1 type2` | from `poc/eigenpair_check_scale.patterns` |
| `--log2-sizes` | `17 19` | subspace sizes `2^k` |
| `--rounds` | `5` | interleaved solves per arm after one warm-up |

One host search serves both arms; each build runs twice, the second timed. Eigenvalues must agree to
`1e-12`. Runs here: the default (5 rounds) and `--rounds 3`, whose build column is §3's.
