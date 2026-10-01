# Four `"pairs"` levers on a GPU

`poc/sparse_pairs_tune.py` (§6) at `909ebf5`, one NVIDIA GH200 120GB, 2026-10-02. Fixture as
`poc/sparse-gpu.md`: spinchain's open-XXZ `xxz` at n=60, `δ = 0.5`, `type1` (`J = 62`, `complex128`),
Hamming-shell subspaces around both Néel states. Nothing here is in the library.

## 1. The levers

After `0d25235` a GH200 `"pairs"` matvec is 84–89% scatters (`poc/sparse-split.md` §4), stepped
`_CHUNK = 32768` entries at a time: 128 steps at `2^20` and 320 at `2^22`, each kernel a fraction of the
GPU. Four levers, crossed with chunk sizes `2^15`, `2^17` and `2^19`:

- `base`: the library kernel, `_apply_pairs`.
- `sorted`: the `out[i]` scatter told `indices_are_sorted` — pairs are sorted by `i` — and `out[j]` not.
- `merged`: both directions as one scatter of the concatenated indices and values.
- `real`: `"csr"`'s split, the real groups' pairs with `float64` factors and the rest `complex128`, in
  two scans; a memory lever.

Every arm keeps the library's CUDA-only carry split.

## 2. Results

Per solve iteration against `base` at `2^15`, the shipped kernel, at `2^20` / `2^22`; every ratio won or
lost all 5 interleaved rounds:

| variant \ chunk | `2^15` | `2^17` | `2^19` |
| --- | --- | --- | --- |
| `base` | 4.49 / 20.67 ms | 2.09× / 1.34× | **2.73× / 1.39×** |
| `merged` | 1.65× / 1.23× | 2.70× / 1.40× | 2.70× / **1.41×** |
| `sorted` | 0.37× / 0.54× | 0.46× / 0.63× | 0.51× / 0.64× |
| `real` | 0.70× / 0.75× | 0.94× / 0.91× | 0.37× / 0.84× |

Operator bytes and the `(2, N)` matvec's temp, MiB, at `2^20` / `2^22`: `base` and `merged` 112 / 304 at
every chunk, `real` 82–92 / 232–236; temp 32 / 128 at `2^15` and `2^17`, 36–40 / 130–136 at `2^19`.
Every arm's eigenvalue agrees with the reference to 7.9e-16, and its iteration count to ±1.

## 3. Lever by lever

- **Chunk size is the lever**: 128 → 8 steps at `2^20` and 320 → 20 at `2^22`, for 2.73× and 1.39× per
  iteration at +8 MiB of temp. It fits the launch-bound, under-filled-GPU reading of §1; `2^19`, the
  largest tried, still edges `2^17`, so the optimum may lie higher (§5).
- **`merged` pays only while steps are many**: 1.65× / 1.23× at `2^15`, a tie with `base` at `2^19`
  (1.66 against 1.64 ms, 14.68 against 14.89 ms). Halving the scatter launches matters only when the
  launches do.
- **The sorted hint is a large slowdown**: 0.37–0.64× per iteration, 0.23–0.41× on the `(2, N)` matvec.
  On the GPU, XLA's scatter with `indices_are_sorted=True` is the slower one. The library's `"csr"`
  passes that flag, and `"csr"` is the slowest sparse kernel on the GH200 (58.00 ms per iteration
  against `"pairs"`' 20.35 at `type2` `2^22`, `poc/sparse-split.md` §4) — a suspect, untested (§5).
- **`real` costs more than it saves**: −24% to −27% operator memory, at 0.37–0.94× everywhere. Its
  0.37× at chunk `2^19`, `2^20` is unexplained.

## 4. What it means

At `type1` `2^22` the best arm's 14.7 ms per iteration is below `"tables"`' 21.06
(`poc/sparse-gpu.md` §7). With the single host search (`b38d48b`), a `"pairs"` call would land near
`"tables"`' 2.65 s at about a quarter of its memory — a projection from the solve stage, not a whole
call. A GPU-only chunk size is the change to make, once §5's sweep places it; `merged`, `sorted` and
`real` are not worth building.

## 5. Open

1. **Chunks past `2^19`**: `--chunks 19 20 21 --variants base`.
2. **`"csr"` without the sorted hint** on the GPU.
3. **The CPU's chunk size**, unmeasured here; `2^15` keeps its temporaries in cache.

## 6. The script

`poc/sparse_pairs_tune.py`, its argparse checked against this section:

| flag | default | meaning |
| --- | --- | --- |
| `--num-qubits`, `--delta` | `60`, `0.5` | the `xxz` fixture |
| `--pattern` | `type1` | one of `poc/eigenpair_check_scale.patterns` |
| `--log2-sizes` | `20 22` | subspace sizes `2^k` |
| `--chunks` | `15 17 19` | log2 of `_CHUNK`, the operator's shapes following it |
| `--matvec` | `pairs` | `pairs`, or `csr` for §5 item 2 |
| `--variants` | every one of `--matvec`'s | `base sorted merged real` for `pairs`, `base unsorted` for `csr`; each run at every chunk |
| `--rounds` | `5` | interleaved rounds after one warm-up per arm |

`base` at `2^15` is the reference and always runs. Each arm compiles its own solve, asserted pairwise
distinct, and must match the reference eigenvalue and `(2, N)` product to `1e-12`; one host search
serves every arm. Runs here: the default sweep on the GH200, and CPU smoke runs at `2^12` and `2^19`.
