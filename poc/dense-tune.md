# Three dense-kernel levers: scan unroll, a cached identity diagonal, a fixed-trip diagonal loop

`poc/dense_tune.py` (§6) at `1e6fbfc`, 2026-10-02: one NVIDIA GH200 120GB for §2, one Apple M1 (8 cores,
16 GiB) for §3. Fixture as `poc/sparse/gpu.md`: spinchain's open-XXZ `xxz` at n=60, `δ = 0.5`, `type1`
(`J = 62`, `complex128`, `kmax = 2` over the non-identity groups), Hamming-shell subspaces around both
Néel states. Nothing here is in the library.

## 1. The levers

A GH200 `"indices"` matvec is ~75% diagonal recompute (`poc/sparse/gpu.md` §6): `get_diagonal`'s
`while_loop` makes ΣK = 179 full passes per matvec at `type1`, 59 of them the identity group's, each step
syncing with the host. `"tables"` is one gather-multiply-add per group, memory-bound. Arms, crossed with
the scan over X groups' `unroll`:

- `plain`: the library's kernel.
- `id`: the identity group's diagonal computed once per solve and applied as `d0 * vec`, the scan over
  the other groups (`"indices"`, `"onthefly"`).
- `id-static`: as `id`, the other groups' diagonals summed over a fixed `kmax` terms, no `while_loop`.

`unroll` lets XLA fuse consecutive groups. The reference is `run_sqd`; each arm replicates its
single-device assembly around the library's `_solve`, and `plain` at `unroll=1` matches it exactly.

## 2. On the GH200

Per solve iteration against `run_sqd`, every ratio won 5/5; `temp` is the `(2, N)` kernel's:

| `"indices"` arm | `2^20` | `2^22` | temp at `2^22` |
| --- | --- | --- | --- |
| `run_sqd` | 27.79 ms | 76.59 ms | 68 MiB |
| `plain`, `unroll=4` / `8` | 1.04× / 1.10× | 0.99× / 1.11× | 1360 / 1824 MiB |
| `id`, `unroll=1` | 1.23× | 1.24× | 68 MiB |
| `id`, `unroll=8` | 1.39× | 1.40× | 1760 MiB |
| **`id-static`, `unroll=1`** | **4.44×** | **3.26×** | **12 MiB** |
| `id-static`, `unroll=4` | 8.19× | 6.59× | 1008 MiB |

| `"tables"` arm | `2^20` | `2^22` | temp at `2^22` |
| --- | --- | --- | --- |
| `run_sqd` | 5.69 ms | 20.39 ms | 4 MiB |
| `plain`, `unroll=4` | 1.55× | 1.66× | 2896 MiB |
| `plain`, `unroll=8` | 1.76× | 1.94× | 2720 MiB |

- **The `while_loop` is the cost, not the arithmetic.** `id-static` at `unroll=1` is 3.26–4.44× per
  iteration (1-D matvec 4.86–6.50×) and *lowers* temp, 68 → 12 MiB: no host sync per term, and the two
  passes fuse. Caching the identity diagonal alone is 1.23–1.24×.
- **`unroll` buys speed with temp**: XLA keeps each unrolled group's temporaries live, ~1 GB at `2^22` for
  `id-static` and 2.7–2.9 GB for `"tables"`, against megabytes. For kernels chosen for their memory that
  is the wrong trade as shipped.
- **`id-static` makes `"indices"` competitive**: 23.52 ms per iteration at `2^22` against `"tables"`'
  20.39 and the tuned `"pairs"`' 14.84 (`poc/sparse/pairs-tune.md` §3), at `"indices"`' memory plus one
  vector; 11.62 ms with `unroll=4`'s gigabyte.
- **A second run** (`--matvecs indices onthefly --variants plain id-static --unrolls 1 2`) reproduces
  `"indices"`' `id-static` at `unroll=1` (4.50× / 3.24×, temp 3 / 12 MiB); `unroll=2` reaches 6.74× /
  5.15× at 984 MiB of temp at `2^22`, no cheap middle. **`"onthefly"`** gains 1.45× with `id-static`
  at `2^20` (1.54× at `unroll=2`), its per-matvec search diluting it; its `2^22` was cut by the job's
  walltime.
- Iterations match `run_sqd` in every arm; eigenvalues agree exactly.

## 3. On CPU

`"indices"`, `unroll=1`, per iteration against `run_sqd`:

| arm | `2^17` | `2^19` | temp at `2^19` |
| --- | --- | --- | --- |
| `run_sqd` | 80.47 / 86.32 ms | 446.31 ms | 8.0 MiB |
| `id` | 1.27× (5/5) | 1.23× (3/3) | 8.0 MiB |
| `id-static` | **1.75×** (3/3) | **1.94×** (3/3) | 1.0 MiB |

`2^17` comes from two runs: `id` from the first, `id-static` from a rerun, each against its own
`run_sqd`. The first run stalled after printing its third row with no error, and was stopped; the cause
is unexplained, and `"onthefly"`, also in it, is unmeasured on CPU. The 1-D matvec gains 2.88–3.25×.
Iterations and eigenvalues match `run_sqd`.

## 4. What it means

**`id-static` at `unroll=1` is a win on both backends at no memory cost** — 3.26–4.44× per iteration on
the GH200, 1.75–1.94× on an M1 — so it needs no platform gate. It is the change to make in the library,
for `"indices"` and `"onthefly"`. One guard: the fixed loop pads every group to `kmax` terms, so a
Hamiltonian with one many-term group would pay `kmax` passes for every group; `type1` has `kmax = 2`. A
threshold, with the `while_loop` kept above it, is unmeasured and needs a large-`kmax` fixture.

## 5. Open

1. **The `kmax` threshold**, on a molecular-like fixture with many Z terms per group
   (`poc/sparse/pairs.py general`).
2. **`"onthefly"`** at `2^22` on the GPU, and on CPU, with `id-static` (1.45× at `2^20` on the GH200).
3. **`unroll` with bounded temp**: `unroll=2` already holds ~1 GB at `2^22`; anything above 1 trades
   memory for speed.

## 6. The script

`poc/dense_tune.py`, its argparse checked against this section:

| flag | default | meaning |
| --- | --- | --- |
| `--num-qubits`, `--delta` | `60`, `0.5` | the `xxz` fixture |
| `--pattern` | `type1` | one of `poc/eigenpair_check_scale.patterns` |
| `--log2-sizes` | `20 22` | subspace sizes `2^k` |
| `--matvecs` | `indices tables` | dense kernels; `onthefly` too |
| `--variants` | `plain id id-static` | `"tables"` takes `plain` only |
| `--unrolls` | `1 4` | the scan's `unroll` |
| `--rounds` | `5` | interleaved rounds after one warm-up per arm |

`solve` includes the setup, as `run_sqd`'s; `1-D` and `(2, N)` are the kernel alone. Traced solves are
asserted pairwise distinct and products equal to `run_sqd`'s to `1e-12`. Runs here: `--log2-sizes 20 22
--unrolls 1 4 8` and `--matvecs indices onthefly --variants plain id-static --unrolls 1 2` on the
GH200; on the M1, `--log2-sizes 17 19 --matvecs indices onthefly --unrolls 1`
(stalled), then `--log2-sizes 17 --variants id-static --unrolls 1 --rounds 3` and `--log2-sizes 19
--variants id id-static --unrolls 1 --rounds 3`, all `--matvecs indices` for the last two.
