# `"pairs"`' cross-group sort on a GPU host

`poc/sparse/pairs_sort.py` (§4) at `ef00e77`, one NVIDIA GH200 120GB, 2026-10-02, `"pairs"` at the GPU chunk
`2^19`. Fixture as `poc/sparse/gpu.md`: spinchain's open-XXZ `xxz` at n=60, `δ = 0.5`, `type1` (`J = 62`)
and `type2` (`J = 120`), Hamming-shell subspaces around both Néel states. The device sort is in the
library since, for `"pairs"` on a GPU (`_pairs_sorted_on_device`).

## 1. The question

`"pairs"` counting-sorts its entries by `i` across groups (`d84c4a3`), which makes the solve's `i` side
local. On the GH200's host that sort is 48–63% of the build (`poc/sparse/gpu.md` §8), and skipping it cost
only 2–5% of the solve (`poc/sparse/tiles.md` §3), so whether it pays was open. Four builds from one shared
search, through the library's padding and device factors:

- `counting`: the library's `_sort_by_target(both=False)`.
- `none`: no cross-group sort, the groups concatenated.
- `argsort`: a stable `np.argsort` of the concatenation by `i`, on the host.
- `device`: a stable `jnp.argsort` on the device, the arrays permuted there.

`argsort` and `device` produce exactly `counting`'s arrays (asserted); `none` the same pairs in another
order. One compiled solve serves every arm.

## 2. Results

Build plus solve, against `counting`, all 5/5:

| arm | `type1` `2^20` | `type1` `2^22` | `type2` `2^20` | `type2` `2^22` |
| --- | --- | --- | --- | --- |
| `counting` | 0.242 s | 2.134 s | 1.214 s | 2.808 s |
| `none` | 1.27× | **0.95×** | 1.10× | **0.98×** |
| `argsort` | 1.22× | 1.04× | 1.08× | 1.15× |
| `device` | **1.44×** | **1.12×** | **1.15×** | **1.50×** |

- **The device sort wins everywhere**: its build is 4.96–10.20× `counting`'s (`type2` `2^22`: 1.035 →
  0.101 s) and its solve identical, the arrays being the same.
- **Dropping the sort loses at `2^22`**: the build gains 7.40–10.28× but the solve falls to 0.85×/0.64×
  per iteration (`type2` `2^22`: 13.74 → 21.42 ms), more than at `2^21` (`poc/sparse/tiles.md` §3).
- **The host `argsort` is in between**: 1.41–1.89× on the build, the solve unchanged.
- The search, shared and excluded, is 0.08–0.57 s. Eigenvalues agree to 5.3e-16.

## 3. What it means

**The sort stays, done on the device**: 1.12–1.50× per `"pairs"` build-plus-solve on the GH200, bit-identical
operator. On a CPU the device arm is XLA's CPU sort, slower than `counting` in a smoke run (0.67× on the
build at `2^12`–`2^14`), so the change belongs behind the GPU backend, beside `_GPU_PAIRS_CHUNK`. `"csr"`'s
and `"ell"`'s two-direction sorts are the same shape of work and unmeasured.

Open: the device sort for `"csr"`/`"ell"`, and its cost on a CPU at real sizes.

## 4. The script

`poc/sparse/pairs_sort.py`, its argparse checked against this section:

| flag | default | meaning |
| --- | --- | --- |
| `--num-qubits`, `--delta` | `60`, `0.5` | the `xxz` fixture |
| `--patterns` | `type1 type2` | `poc/eigenpair_check_scale.patterns` |
| `--log2-sizes` | `20 22` | subspace sizes `2^k` |
| `--rounds` | `5` | interleaved rounds after one warm-up |

`build` runs from the searched pairs to the device operator, `solve` is `_run_sparse`, `total` their sum.
Runs here: the default sweep on the GH200, and CPU smoke runs at `2^12` and `2^14`.
