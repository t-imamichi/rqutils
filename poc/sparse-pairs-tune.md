# Four `"pairs"` levers on a GPU

`poc/sparse_pairs_tune.py` (§6) at `909ebf5` (§2) and `1443f48` (§3's sweep and `"csr"` run), one NVIDIA
GH200 120GB, 2026-10-02; §3's CPU sweep at `a1fcde3` on one Apple M1 (8 cores, 16 GiB). Fixture as
`poc/sparse-gpu.md`: spinchain's open-XXZ `xxz` at n=60, `δ = 0.5`, `type1` (`J = 62`, `complex128`),
Hamming-shell subspaces around both Néel states. Two results are in the library since: the chunk size,
as `_GPU_PAIRS_CHUNK`, and dropping the sorted hint on CUDA (§3); the other levers are not.

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
  largest tried, still edges `2^17`. **A sweep past it finds the plateau there**:

  | chunk | `2^20`, per iteration (temp) | `2^22`, per iteration (temp) |
  | --- | --- | --- |
  | `2^15` | 4.30 ms (32 MiB) | 20.25 ms (128 MiB) |
  | `2^19` | 2.71× (40 MiB) | 1.37× (136 MiB) |
  | `2^20` | 2.56× (48 MiB) | 1.37× (144 MiB) |
  | `2^21` | 2.72× (64 MiB) | 1.39× (160 MiB) |

  All 5/5 rounds. Past `2^19` only the temp grows, so `2^19` — the smallest size on the plateau — ships
  for `"pairs"` on a GPU (`_GPU_PAIRS_CHUNK`); the CPU keeps `_CHUNK = 2^15`.
- **On CPU `2^15` is already the optimum**: `--chunks 13 15 17 19 --variants base` for `"pairs"` and
  `"csr"`, per iteration against `2^15`, at `2^17` / `2^19`:

  | chunk | `"pairs"` | `"csr"` | temp |
  | --- | --- | --- | --- |
  | `2^13` | 0.95× / 0.96× | 0.97× / 0.94× | 0.3–0.6 MiB |
  | `2^15` | 9.51 / 50.72 ms | 11.43 / 67.80 ms | 1.1–2.3 MiB |
  | `2^17` | 1.04× / 1.01× | 0.86× / 1.03× | 9 MiB |
  | `2^19` | 0.54× / 0.77× | 0.60× / 0.87× | 36 MiB |

  `2^17`'s best case is +4% at 4× the temp, and `"csr"` loses 14% with it at `2^17`; the GPU's `2^19` is
  13–46% slower, its temp spilling the cache (and at `2^17` mostly padding: one chunk holds the
  operator). Eigenvalues agree to 4.6e-16. Hence the GPU-only gate.
- **`merged` pays only while steps are many**: 1.65× / 1.23× at `2^15`, a tie with `base` at `2^19`
  (1.66 against 1.64 ms, 14.68 against 14.89 ms). Halving the scatter launches matters only when the
  launches do.
- **The sorted hint is a large slowdown**: 0.37–0.64× per iteration, 0.23–0.41× on the `(2, N)` matvec.
  On the GPU, XLA's scatter with `indices_are_sorted=True` is the slower one. The library's `"csr"`
  passes that flag, and `"csr"` is the slowest sparse kernel on the GH200 (58.00 ms per iteration
  against `"pairs"`' 20.35 at `type2` `2^22`, `poc/sparse-split.md` §4). **Confirmed** with
  `--matvec csr --chunks 15 19`, `type1`, per iteration against the shipped `"csr"`:

  | `"csr"` arm | `2^20` | `2^22` |
  | --- | --- | --- |
  | shipped (`2^15`, hint) | 20.27 ms | 58.32 ms |
  | no hint, `2^15` | **3.24×** | 2.12× |
  | no hint, `2^19` | 2.92× | **2.40×** |
  | hint, `2^19` | 1.07× | 1.09× |

  All 5/5; the `(2, N)` matvec gains 3.04–5.04×. `_scan_add` now drops the hint on CUDA, for every
  sparse kernel and a real carry too; the CPU keeps it, unmeasured there. The larger chunk is mixed for
  `"csr"` without the hint (0.90× at `2^20`, 1.13× at `2^22`), so it keeps `2^15`. Even fixed, `"csr"`
  trails `"pairs"`: 24.32 against 14.76 ms per iteration at `2^22`, at 408 against 304 MiB.
- **`real` costs more than it saves**: −24% to −27% operator memory, at 0.37–0.94× everywhere. Its
  0.37× at chunk `2^19`, `2^20` is unexplained.

## 4. What it means

At `type1` `2^22` the best arm's 14.7 ms per iteration is below `"tables"`' 21.06
(`poc/sparse-gpu.md` §7). With the single host search (`b38d48b`), a `"pairs"` call would land near
`"tables"`' 2.65 s at about a quarter of its memory — a projection from the solve stage, not a whole
call. The GPU-only chunk ships at `2^19`; `merged`, `sorted` and `real` are not worth building.

## 5. Open

1. **`"ell"` at a larger chunk** on the GPU; the script has no `ell` mode.
2. **The whole `"pairs"` call** against `"tables"` with the chunk and the single search (§4's projection):
   `poc/sparse_gpu.py --arms indices tables pairs --log2-sizes 20 21 22`.

## 6. The script

`poc/sparse_pairs_tune.py`, its argparse checked against this section:

| flag | default | meaning |
| --- | --- | --- |
| `--num-qubits`, `--delta` | `60`, `0.5` | the `xxz` fixture |
| `--pattern` | `type1` | one of `poc/eigenpair_check_scale.patterns` |
| `--log2-sizes` | `20 22` | subspace sizes `2^k` |
| `--chunks` | `15 17 19` | log2 of `_CHUNK`, the operator's shapes following it |
| `--matvec` | `pairs` | `pairs`, `csr`, or `ell` (whose variants are width grids) |
| `--variants` | every one of `--matvec`'s | `base sorted merged real` for `pairs`, `base sorted` for `csr`, `base grid1.5 grid2` for `ell`; each run at every chunk |
| `--rounds` | `5` | interleaved rounds after one warm-up per arm |

`base` at `2^15` is the reference and always runs. Each arm compiles its own solve, asserted pairwise
distinct, and must match the reference eigenvalue and `(2, N)` product to `1e-12`; one host search
serves every arm. It patches both `_CHUNK` and `_chunk`, so a GPU `"pairs"` arm takes its own chunk
rather than `_GPU_PAIRS_CHUNK`. Runs here: the default sweep on the GH200 (§2), `--chunks 19 20 21
--variants base` and `--matvec csr --chunks 15 19` there (§3), `--log2-sizes 17 19 --chunks 13 15 17 19
--variants base` on the M1 for both `--matvec` (§3), and CPU smoke runs at `2^12` and `2^19`.
Since the hint change, `--matvec csr`'s `base` is the library (no hint on CUDA) and `sorted` forces
it; the run above had `base` with the hint and an `unsorted` arm without.
