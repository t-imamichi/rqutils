# A tiled entry order for `"pairs"`

`poc/sparse/tiles.py` (§7), 2026-10-01/02: one Apple M1 (8 cores, 16 GiB) for §2, one NVIDIA GH200 120GB
for §3. Fixture as
`poc/sparse/gpu.md`: spinchain's open-XXZ `xxz` at n=60, `δ = 0.5`, `type1` (`J = 62`, `complex128`),
Hamming-shell subspaces around both Néel states. Nothing here is in the library.

## 1. The idea

`"pairs"` stores each transition once as `(i, j, d)`, sorted by `i` across groups since `d84c4a3`, so
`vec[i]`/`out[i]` stream while `vec[j]`/`out[j]` land anywhere in the vector. Sorting by
`(i >> s, j >> s, i)` instead makes consecutive chunks touch one `2^s`-state slice of each side. It was
`poc/sparse/gpu.md` §10.4's blocked-matvec lever for the L2 cliff. Only the data order changes, so one
compiled solve serves every arm. Arms:

- `i`: shipped.
- `group`: `(group, i)`, close to the order before `d84c4a3`.
- `tile12`, `tile14`, `tile16`: the tiled key at `s` = 12, 14, 16.

## 2. Results

Median of 5 interleaved rounds, each ratio against `i`, with how many rounds the arm won. `1-D` and
`(2, N)` are the bare kernel on a fixed vector of that shape; the solve runs both.

| order | N | solve/iter | ratio | wins | 1-D matvec | ratio | wins | `(2, N)` matvec | ratio | wins | iters |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `i` | `2^17` | 9.23 ms | — | — | 2.02 ms | — | — | 2.26 ms | — | — | 106 |
| `group` | `2^17` | 8.71 ms | 1.06× | 5/5 | 1.43 ms | 1.41× | 5/5 | 2.67 ms | 0.85× | 0/5 | 105 |
| `tile12` | `2^17` | 8.58 ms | **1.08×** | 5/5 | 1.68 ms | 1.20× | 5/5 | 2.15 ms | 1.05× | 5/5 | 105 |
| `tile14` | `2^17` | 8.88 ms | 1.04× | 4/5 | 1.77 ms | 1.14× | 5/5 | 2.17 ms | 1.04× | 4/5 | 96 |
| `tile16` | `2^17` | 9.32 ms | 0.99× | 0/5 | 1.98 ms | 1.02× | 4/5 | 2.27 ms | 0.99× | 3/5 | 96 |
| `i` | `2^19` | 51.58 ms | — | — | 8.49 ms | — | — | 10.35 ms | — | — | 64 |
| `group` | `2^19` | 62.69 ms | 0.82× | 0/5 | 8.77 ms | 0.97× | 3/5 | 20.10 ms | 0.52× | 0/5 | 64 |
| `tile12` | `2^19` | 47.68 ms | **1.08×** | 5/5 | 6.66 ms | **1.27×** | 5/5 | 10.07 ms | 1.03× | 4/5 | 64 |
| `tile14` | `2^19` | 48.82 ms | 1.06× | 5/5 | 7.44 ms | 1.14× | 5/5 | 10.24 ms | 1.01× | 4/5 | 64 |
| `tile16` | `2^19` | 49.46 ms | 1.04× | 5/5 | 7.76 ms | 1.09× | 5/5 | 10.39 ms | 1.00× | 3/5 | 64 |

- **`tile12` is 1.08× per solve iteration at both sizes**, winning 10 of 10 rounds, at the same 64
  iterations at `2^19`. An earlier run of the script, before the 1-D column, measured 1.08× and 1.11×.
- **The gain is in the 1-D matvec**: 1.20–1.27× against 1.03–1.05× on `(2, N)`. The solve's 1-D calls
  are the prefilter's 32 Chebyshev steps and `body()`'s third matvec. Why the batch gains so much less
  is not established.
- **Smaller tiles win, monotonically**: 12 > 14 > 16 in every column at both sizes. `s = 12` is the
  smallest tried, so the optimum may lie lower (§5). At 64 B/state for a batched vector and its output, a
  `2^12` slice is 256 KiB, far below the 12 MB L2 the sweep was sized for.
- **`group` is not a candidate**: it halves `(2, N)` throughput at `2^19` (0.52×) and loses 0.82× per
  solve iteration, despite a 1.41× 1-D matvec at `2^17`.
- Every arm holds the same entries as `i` and a different order (both asserted). Eigenvalues agree to
  2.9e-16; iteration counts differ at `2^17` (106/105/96) because scatter order changes rounding, which is
  why the solve is timed per iteration.

## 3. On the GPU: nothing before the carry fix, 1.01–1.03× after

**Before `0d25235`**, GH200, `--log2-sizes 20 21 22 --tiles 10 12 14 16 18`, the script unchanged from
§2's run; its `2^22` rows were never read. Ratios against `i`; `wins` as in §2:

| order | N | solve/iter | ratio | wins | 1-D matvec | ratio | wins | `(2, N)` matvec | ratio | wins |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `i` | `2^20` | 17.34 ms | — | — | 4.60 ms | — | — | 9.12 ms | — | — |
| `group` | `2^20` | 17.96 ms | 0.97× | 0/5 | 4.75 ms | 0.97× | 0/5 | 9.53 ms | 0.96× | 0/5 |
| `tile10` | `2^20` | 17.28 ms | 1.00× | 5/5 | 4.59 ms | 1.00× | 5/5 | 9.09 ms | 1.00× | 5/5 |
| `tile12` | `2^20` | 17.27 ms | 1.00× | 5/5 | 4.60 ms | 1.00× | 3/5 | 9.10 ms | 1.00× | 5/5 |
| `tile18` | `2^20` | 17.35 ms | 1.00× | 1/5 | 4.62 ms | 1.00× | 3/5 | 9.12 ms | 1.00× | 3/5 |
| `i` | `2^21` | 40.11 ms | — | — | 11.78 ms | — | — | 21.13 ms | — | — |
| `group` | `2^21` | 41.18 ms | 0.97× | 0/5 | 12.12 ms | 0.97× | 0/5 | 21.64 ms | 0.98× | 0/5 |
| `tile10` | `2^21` | 40.05 ms | 1.00× | 5/5 | 11.77 ms | 1.00× | 3/5 | 21.07 ms | 1.00× | 5/5 |
| `tile12` | `2^21` | 40.09 ms | 1.00× | 5/5 | 11.79 ms | 1.00× | 2/5 | 21.10 ms | 1.00× | 5/5 |
| `tile18` | `2^21` | 40.17 ms | 1.00× | 0/5 | 11.81 ms | 1.00× | 0/5 | 21.15 ms | 1.00× | 2/5 |

`tile14` and `tile16` sit between `tile12` and `tile18` in every column (0.99–1.00×).

- **Every tile is within 0.7% of `i`**, in all three columns at both sizes. `tile10`/`tile12` win
  consistently but by 0.1–0.4%: an order, not a lever. `group` loses 2–4%.
- **This is evidence against `poc/sparse/gpu.md` §3's L2 reading for `"pairs"`.** A `tile10` block touches
  two 1024-state slices, ~64 KiB, so its gathers and scatters should hit L2 throughout; if missed
  `vec[j]`/`out[j]` lines were the cost, this would have recovered most of the cliff.
- **The cost tracked the data, not the access pattern**: the `(2, N)` matvec was 1.79–2.01× the 1-D one
  in every arm. That was the per-step carry split, three full passes over `out` per scan step
  (`poc/sparse/split.md` §1), which no entry order can change.
- The shipped order reproduces `poc/sparse/gpu.md` §3 (17.45 and 40.15 ms per iteration). Eigenvalues agree
  to 3.9e-16; iteration counts vary by ±1 between repeats, as recorded there in §5.

**After `0d25235`**, the same GH200 and fixture, `--log2-sizes 20 21 --tiles 10 12 14 16 18`, with the
fixed kernel (the script calls the library's):

| order | N | solve/iter | ratio | wins | 1-D matvec | ratio | wins | `(2, N)` matvec | ratio | wins |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `i` | `2^20` | 4.25 ms | — | — | 1.43 ms | — | — | 1.60 ms | — | — |
| `group` | `2^20` | 4.36 ms | 0.98× | 0/5 | 1.46 ms | 0.98× | 1/5 | 1.66 ms | 0.97× | 0/5 |
| `tile10` | `2^20` | 4.12 ms | 1.03× | 5/5 | 1.41 ms | 1.02× | 5/5 | 1.51 ms | 1.06× | 5/5 |
| `tile18` | `2^20` | 4.20 ms | 1.01× | 5/5 | 1.42 ms | 1.01× | 5/5 | 1.57 ms | 1.02× | 5/5 |
| `i` | `2^21` | 6.60 ms | — | — | 2.26 ms | — | — | 2.52 ms | — | — |
| `group` | `2^21` | 6.96 ms | 0.95× | 0/5 | 2.33 ms | 0.97× | 0/5 | 2.75 ms | 0.91× | 0/5 |
| `tile10` | `2^21` | 6.46 ms | 1.02× | 5/5 | 2.22 ms | 1.02× | 5/5 | 2.43 ms | 1.04× | 5/5 |
| `tile18` | `2^21` | 6.56 ms | 1.01× | 5/5 | 2.25 ms | 1.00× | 5/5 | 2.49 ms | 1.01× | 5/5 |

`tile12`–`tile16` sit between `tile10` and `tile18` in every column (1.01–1.05×, all 5/5).

- **With the defect gone, order registers.** Every tile won all 5 rounds in every column at both sizes,
  1.01–1.03× per iteration and up to 1.06× on the `(2, N)` matvec, smaller tiles mostly better — the CPU's
  direction (§2) at a third of its size.
- **Skipping the cross-group sort loses**: `group` is 0.95–0.98× per iteration and 0.91× on the `(2, N)`
  matvec at `2^21`, while the sort is ~9–14% of a CPU `"pairs"` build (§4). It stays.
- Eigenvalues agree to 5.3e-16; iteration counts vary by ±1, as before.

## 4. What it costs

Device memory is unchanged: the same three arrays, permuted. The host build gains an `np.lexsort` over
the real entries, **unmeasured** — the timings above exclude the build. For scale, the shipped linear
counting sort by `i` is 9–14% of a CPU `"pairs"` build (M1, `type1`/`type2` `2^20`/`2^21`, warm; the
search is 80–87%, the device factors 4–6%), measured by wrapping `_sparse_operator`'s stages in a
one-off session — no committed script reproduces it.

## 5. What it means

A free per-iteration win on CPU at no memory cost, the first lever past the cache to measure positive
there (`poc/sparse/layout.md`'s state-major gather lost 0.95–0.98×). At 1.08× it does not change
`poc/sparse/pairs.md` §10's CPU ranking, where `"ell"` leads at ~2× `"csr"`.

**On the GPU it was aimed at the wrong cause**: the cliff was the per-step carry split
(`poc/sparse/split.md` §1), not locality. With that fixed it is a consistent 1.01–1.03× per iteration —
at `type2` `2^22`, ~0.1 s of a 6.38 s call (`poc/sparse/gpu.md` §7), which an `O(E log E)` `lexsort`
over 14M+ entries could cost back. Not worth shipping without a linear tile sort, measured.

## 6. Open

1. **A linear tile sort and its build cost**: a counting sort by tile id over the already `i`-sorted
   entries keeps `i` order within a tile; the bar is §5's ~2–3% of a GPU solve.
2. **`2^22` with the fixed kernel**, and below `tile10` on the GPU, since smaller tiles still win.
3. **Tiles below 12 on CPU**, since that sweep is monotone to its edge.
4. **`2^20` and up on CPU**, and the host build cost of the sort.
5. **The same key for `"csr"`/`"ell"`**, which sort by target row and so would lose `indices_are_sorted`
   and their one-write-per-row structure respectively.

## 7. The script

`poc/sparse/tiles.py`, its argparse checked against this section:

| flag | default | meaning |
| --- | --- | --- |
| `--num-qubits`, `--delta` | `60`, `0.5` | the `xxz` fixture |
| `--pattern` | `type1` | one of `poc/eigenpair_check_scale.patterns` |
| `--log2-sizes` | `17 19` | subspace sizes `2^k` |
| `--tiles` | `12 14 16` | the `s` of each `tileS` arm; `i` and `group` always run |
| `--rounds` | `5` | interleaved rounds after one warm-up per arm |

It wraps `_sort_by_target` on the host and permutes `"pairs"`' real entries (padding, `i == j`, stays
last), then builds each arm's operator through `_sparse_operator` itself. `solve` is `_run_sparse` with
`return_eigvec=False`, divided by that solve's iteration count, captured by wrapping `ground_locg` with a
host callback as `poc/sparse/gpu.py` does. Runs here: the default sweep on CPU, twice (§2 is the
second), and `--log2-sizes 20 21 22 --tiles 10 12 14 16 18` on the GH200 before `0d25235` and
`--log2-sizes 20 21` with the same tiles after it (§3).
