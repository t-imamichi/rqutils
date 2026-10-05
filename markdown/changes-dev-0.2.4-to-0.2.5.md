# Changes from `dev-0.2.4` to `dev-0.2.5`

`dev-0.2.4` (`0d25235`, 2026-10-01) to `dev-0.2.5` (`6480db9`, 2026-10-05): 59 commits, 21 of them in
`rqutils/`. `pyproject.toml`'s version stays `0.2.0`. Figures are measured unless marked; each points to
the write-up that holds it.

## 1. Breaking changes

- **`Matvec.CSR` and `Matvec.ELL` are removed** (`5d5607a`). Both were dominated or mixed against
  `"pairs"` on CPU and GPU (`poc/sparse/tune.md` §3); `matvec=Matvec.CSR` is now an `AttributeError`.
  Their builders and kernels live on in `poc/sparse/legacy.py`, keyed by string name.
- **`PauliSumXZ` has three new required static fields**: `term_counts` (each X group's Z-term count),
  `identity_first` (group 0 is the identity) and `zfree_first` (per group, whether term 0 has no Z part).
  `from_paulisum` fills them, and now moves each group's Z-free term first. A `PauliSumXZ` constructed
  directly must pass them; there is no `while_loop` fallback when they are absent.

## 2. Faster dense kernels (`"indices"`, `"onthefly"`)

- **Fixed-trip diagonals, bucketed by term count, with the identity group's diagonal cached once per
  solve** (`e349e43`), replacing `get_diagonal`'s `while_loop`, which synced with the host per term on a
  GPU. `"indices"` per iteration: 2.96–4.32× on a GH200, 1.66–1.96× on an M1; `"onthefly"` 1.31–1.45× on
  the GH200 and 1.09–1.13× on the M1, where its per-matvec search dominates (`poc/dense-tune.md` §3, §4).
- **Z-free terms folded into a per-group constant** (`26e5b8d`): 1.28× (`type1`) and 1.50× (`type2`) per
  `"indices"` iteration on an M1, 1.13–1.18× on a GH200, same memory (`poc/dense-codes.md` §2, §5). On
  the GH200 this puts `"indices"` ahead of `"tables"`.
- `apply_h`, the residual check, `"tables"`' precompute and `hproj` keep the `while_loop` diagonal on
  purpose: each runs once per solve or call, and the check stays independent of the new static fields
  (`NOTES.md`, "sqd: `get_diagonal`'s `while_loop` stays outside the solve loop, deliberately").

## 3. Faster `"pairs"`

- **Exact-zero entries dropped** after the factor pass (`6d13bfb`): XX+YY hops cancel on aligned spins,
  86% of `type1`'s pairs and 32% of `type2`'s. 1.18–1.59× per solve on an M1, operator −23–64%, iteration
  counts unchanged (`poc/sparse/prune.md` §3).
- **The dropped entries' padding spread over distinct rows** (`d0a9997`): padding on one row serialized a
  GPU's atomic adds (0.32–0.73×); fixed, 2.25–5.96× faster solves on a GH200 (`poc/sparse/prune.md` §6).
- **The filter runs as one jitted program**, `_compact` (`3b50443`): first call 3.4–3.8× on an M1 and
  4.0–4.4× on a GH200, halving a first build there; bit-identical. JAX's persistent compile cache at its
  default 1 s threshold does not hide the eager cost (`poc/sparse/drop-jit.md` §2–§4).
- **The cross-group sort by `i` runs on the device** (`472e731`, all backends since `4317d25`): 1.12–1.50×
  per build plus solve on a GH200, 0.98–0.99× on an M1 (`poc/sparse/pairs-sort.md`).
- **`2^19` entries per scanned chunk on a GPU** (`96fb70f`, `_GPU_PAIRS_CHUNK`; `2^15` on CPU): 2.71×/1.37×
  per iteration at `2^20`/`2^22` on a GH200, +8 MiB temp (`poc/sparse/tune.md` §2).
- **No scatter is marked `indices_are_sorted`** (`a1fcde3`, `4317d25`): the hint slowed a GPU scatter
  2–3× and bought a CPU nothing (`poc/sparse/tune.md` §3).
- **One host search serves the build and the residual check** (`b38d48b`, `6c5b021`): build plus check
  1.41–1.53× on an M1, residual bit-identical.

## 4. Which kernel to use, now

Documented in the `sqd` module docstring (`3f05b4e`, `c745312`, `d05344d`):

- **One device, CPU or GPU: `Matvec.PAIRS`.** On one CPU it is 3.9–5.6× `"indices"` and 3.1–4.0×
  `"tables"` at `"indices"`' memory; on one GPU 1.1–1.8× `"indices"` at ~0.6× its memory
  (`poc/sparse/gpu.md` §9).
- **Under a mesh: `Matvec.INDICES`**, since `"pairs"` is single-device. Inferred from single-device runs;
  multi-GPU and CPU meshes are unmeasured, and a CPU mesh may favour `"tables"`.
- **`J`**, the number of distinct X signatures, is set by the Hamiltonian: `O(n)` for a spin chain, `O(n^4)`
  for Jordan–Wigner fermions. At large `J`, `"indices"`' `4·J·N` bytes outgrow one device.

## 5. Measured, not shipped

- **Factor codes** (`uint8` indices into a factor table) for `"pairs"`: operator −22–56%, 1.00–1.01× on
  CPU, 1.03–1.09× on a GH200 (`poc/sparse/prune.md` §4). The sort-free variant
  (`markdown/parity-codes-proposal.md`) is **declined**: it pays only for Hamiltonians with few distinct
  factors, and the solver stays generic.
- **Coded `"tables"` diagonals**: 0.80–0.84× on an M1, 0.88–0.97× on a GH200; dropped, since `"tables"` is
  behind `"pairs"` on one device (`poc/dense-codes.md`).
- **An atomic-free scatter and `unique-exact`** for `"pairs"` on a GPU: mixed by size, not shipped
  (`poc/sparse/tune.md` §3, `poc/sparse/pairs-sort.md` §2).
- **A tiled `"pairs"` order**: 1.01–1.03× on a GH200 (`poc/sparse/tiles.md` §3).

## 6. Proposals awaiting review

- `markdown/pairs-mesh-proposal.md`, on branch `pairs-mesh-proposal` (not in `dev-0.2.5`): `"pairs"` under
  a mesh, exact (row-owned directed pairs, a push matvec) and randomized (a GF(2)-linear partition).

## 7. Tooling and layout

- The sparse POCs moved to `poc/sparse/` (`d9fb5a4`); `pairs_tune` became `tune` (`bbb9ec4`).
- New POCs: `poc/dense_tune.py`, `poc/dense_codes.py`, `poc/sparse/prune.py`, `poc/sparse/pairs_sort.py`,
  `poc/sparse/drop_jit.py`, each with its write-up.
- `poc/sparse/gpu.py` reports the child's RSS peak where a backend has no `memory_stats` (`17965cc`).
