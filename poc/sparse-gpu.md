# Sparse matvec kernels on a GPU

`poc/sparse_gpu.py` (§8), one NVIDIA GH200 120GB, 2026-10-01: the GPU timing that `poc/sparse-pairs.md` §7
left open, through the shipped `sqd(matvec=...)` kernels. Fixture as there: spinchain's open-XXZ `xxz` at
n=60, `δ = 0.5`, Hamming-shell subspaces around both Néel states; `type1` (`J = 62`) and `type2`
(`J = 120`), both `complex128` (61 of 62 and 118 of 120 groups real). Every timing is a warm median of 3;
every sparse eigenvalue agreed with `"indices"` to 7.1e-15. No size here is at or below `--oracle-log2`,
so `"indices"` is the only reference. `type3`/`type4` and `type2` at `2^20` are unrun.

## 1. Whole solves against `"indices"`

`"indices"`' time over each arm's (median, s):

| pattern | N | `"indices"` | `"pairs"` | `"csr"` | `"ell"` |
| --- | --- | --- | --- | --- | --- |
| `type1` | `2^17` | 1.232 s | **9.33×** | 5.36× | 5.09× |
| `type1` | `2^19` | 1.337 s | **3.33×** | 1.45× | 1.78× |
| `type1` | `2^20` | 2.552 s | **1.30×** | 0.49× | 0.63× |
| `type1` | `2^21` | 5.339 s | 0.99× | 0.41× | 0.49× |
| `type2` | `2^17` | 1.930 s | **6.77×** | 3.23× | 4.20× |
| `type2` | `2^19` | 2.895 s | **2.58×** | 0.95× | 1.24× |
| `type2` | `2^21` | 9.919 s | 0.57× | 0.24× | 0.27× |

- **`"pairs"` is the fastest sparse kernel at every size**, inverting the CPU ranking, where `"ell"` led
  at ~2× `"csr"` and `"pairs"` trailed (`poc/sparse-pairs.md` §10). `"ell"` beats `"csr"` by only
  1.14–1.31×, and loses at `type1` `2^17` (0.95×).
- **Every sparse win shrinks with N and is gone by `2^21`.** §3 is why.
- The `type2` `2^21` row is from the script's previous revision: one process for all arms, no stage split
  or per-arm memory. Its solve path is the same `sqd` call.

## 2. The host build is not the cost

`type1` `2^21`, median per stage (§8 defines them):

| arm | build | solve | check | build share |
| --- | --- | --- | --- | --- |
| `"pairs"` | 0.322 s | 4.819 s | 0.191 s | 6.0% |
| `"csr"` | 0.472 s | 12.230 s | 0.190 s | 3.6% |
| `"ell"` | 0.922 s | 9.667 s | 0.193 s | 8.5% |

Build and check scale with N (`"pairs"` build 0.076 → 0.201 → 0.322 s over `2^19`–`2^21`); the solve
does not.

## 3. An L2 cliff between `2^19` and `2^20`

`type1`, solve time per LOBPCG iteration; ns are per state:

| arm | `2^19` | `2^20` | `2^21` | `2^19` → `2^20` | `2^20` → `2^21` |
| --- | --- | --- | --- | --- | --- |
| `"pairs"` | 4.00 ms (7.6 ns) | 17.45 ms (16.6 ns) | 40.15 ms (19.1 ns) | **4.4×** | 2.3× |
| `"csr"` | 11.66 ms (22.2 ns) | 50.87 ms (48.5 ns) | 101.92 ms (48.6 ns) | **4.4×** | 2.0× |
| `"ell"` | 7.16 ms (13.7 ns) | 35.93 ms (34.3 ns) | 80.56 ms (38.4 ns) | **5.0×** | 2.2× |
| `"indices"`* | 20.90 ms | 27.44 ms | 44.49 ms | 1.3× | 1.6× |

\* Setup included: it runs inside the one jitted call.

- **One step, then linear.** Per state, every sparse kernel costs 2.2–2.5× more at `2^20` than at
  `2^19`, then is flat to within 15%.
- **Iterations grow too, the same in every arm to ±1**: 64 → 93 → 120. `"pairs"`' 19× from `2^19` to `2^21`
  is 1.9× iterations times 10× per iteration.
- **The step is where the batched vectors leave L2.** A `complex128` `(2, N)` input plus its output is
  64 B/state: 32 MiB at `2^19`, 64 MiB at `2^20`, against the H100's 50 MB L2 (NVIDIA's specification, not
  measured here). This is the CPU's §3 finding at a larger cache: there P2 doubled per state from `2^17` to
  `2^19`, as the same arrays grew from 8 to 32 MiB. **Inferred, not measured** — the size of the step and
  its location fit, but no L2 hit rate was read (§7).
- **`"indices"` degrades less** because its per-state diagonal recomputation hides the latency, as on the
  CPU. That is why `"pairs"`' lead goes from 3.3× to a tie.

## 4. Memory

`peak_bytes_in_use`, one fresh process per arm (§8), GiB:

| pattern | N | `"indices"` | `"pairs"` | `"csr"` | `"ell"` |
| --- | --- | --- | --- | --- | --- |
| `type1` | `2^17` | 0.09 | 0.10 | 0.10 | 0.11 |
| `type1` | `2^19` | 0.22 | 0.13 | 0.14 | 0.14 |
| `type1` | `2^20` | 0.45 | 0.30 | 0.34 | 0.32 |
| `type1` | `2^21` | 0.89 | **0.54** | 0.59 | 0.56 |
| `type2` | `2^17` | 0.10 | 0.11 | 0.12 | 0.12 |
| `type2` | `2^19` | 0.34 | 0.18 | 0.22 | 0.20 |

From `2^19` on, every sparse arm is lighter than `"indices"`: −34% to −39% at `type1` `2^21`, −35% to
−47% at `type2` `2^19`. At `2^17` all four sit at a ~0.1 GiB floor. The peak covers the uploaded fixture,
the build, the warm-up and the timed solves.

## 5. Repeated solves are not bit-identical

The warm-up and the timed solves differed by one iteration in 3 of 12 sparse cells (`type1` `2^20`
`"csr"`: 92/93; `type2` `2^19` `"pairs"` and `"csr"`: 103/104), and `type1` `2^20` `"ell"` stopped at 92
against the others' 93. Energies still agree to 7.1e-15. **The cause is unverified**; a GPU scatter-add
summing in nondeterministic order would produce exactly this, and `"ell"`'s per-row reduce has a fixed
order, which fits it being the one kernel that never varied across repeats. CPU runs of the same script
at `2^10`–`2^13` gave identical counts.

## 6. What it means

- **Up to `2^19` on this GPU, `"pairs"` is the kernel**: 2.6–9.3× `"indices"`, at −41% to −47% memory by
  `2^19`.
- **Past the L2 at `complex128`, only the memory win survives.** At `2^21` `"pairs"` ties `"indices"` on
  `type1` (0.99×) and loses on `type2` (0.57×), at −39% memory. Spinchain's sizes (N = 1.5M–20M) are all
  past the cliff.
- **`"csr"` and `"ell"` lose past the cliff**, by 2–4×. On the GPU, the CPU's case for them (sequential
  writes) does not carry.

**For spinchain-scale runs, keep `"indices"`** (the default), and reach for `"pairs"` only when a single
GPU's memory binds. Three reasons, the last two projections from `2^21`:

- Past the cliff `"pairs"` buys memory, not time (above), and `"csr"`/`"ell"` buy neither.
- Memory does not bind at n=60: `"indices"` peaked at 0.89 GiB at `2^21`, so ~9 GiB at 20M if it scales
  linearly — not measured.
- Per iteration from `2^20` to `2^21`, `"indices"` grew 1.6× against the sparse kernels' 2.0–2.3×, setup
  included in its figure, so the gap should widen beyond `2^21`. Two points, not a trend.

Under a mesh the sparse kernels raise, which leaves `"onthefly"`, `"indices"` and `"tables"`. **`"tables"`
is unmeasured on any GPU** and is the open candidate (§7): it caches the diagonal factors whose
recomputation is what hides `"indices"`' latency past the cliff, so caching them could win or lose.

## 7. Open

1. **Confirm the L2 cause**: Nsight Compute's `lts__t_sector_hit_rate.pct` on one `"pairs"` solve at
   `2^19` and `2^20`. A collapse across the step settles §3.
2. **`type3`/`type4`**, and `type2` at `2^20`/`2^21` with this revision.
3. **`"tables"`, and sizes past `2^21`**, which §6's recommendation rests on:
   `--arms indices tables pairs --log2-sizes 21 22 23`. `"tables"` stores ~`20·J` B/state
   (`poc/sparse-pairs.md`'s arms table), ~20 GB for `type2` at `2^23`.
4. **The levers past the cliff, unmeasured**: a locality-preserving state order (RCM did nothing on CPU,
   `poc/sparse-pairs.md` §4, since these graphs are hypercube-like); a matvec blocked so each block's
   slice of `vec` fits L2; `"pairs"` without atomics (`poc/sparse-pairs.md` §7.6, which also lists the
   scan-step and layout sweeps).
5. **Nondeterministic iteration counts** (§5): confirm the scatter-add cause, and whether
   XLA's `--xla_gpu_deterministic_ops` removes it, at what cost.
6. **Only one GPU.** A different L2 size moves the cliff, by this account in proportion (A100: 40 MB).
   An A100 attempt gave no number: its child processes fell back to CPU on `cuInit(0)`'s
   `CUDA_ERROR_NO_DEVICE` — undiagnosed; the likeliest cause is `--device` overriding a scheduler's
   `CUDA_VISIBLE_DEVICES`.

## 8. The script

`poc/sparse_gpu.py`, its argparse checked against this section:

| flag | default | meaning |
| --- | --- | --- |
| `--patterns` | `type1 type2 type3 type4` | fixtures, from `poc/eigenpair_check_scale.patterns` |
| `--log2-sizes` | `17 19 21` | subspace sizes `2^k` |
| `--arms` | `indices pairs csr ell` | `"indices"` always runs first, as the reference |
| `--repeats` | `3` | timed solves after one warm-up |
| `--oracle-log2` | `16` | largest size also checked against `hproj` + `eigsh` |
| `--num-qubits`, `--delta` | `60`, `0.5` | the `xxz` fixture |
| `--device` | none | sets `CUDA_VISIBLE_DEVICES` before JAX initializes |

Each `(pattern, size, arm)` runs in a fresh subprocess (a hidden `--child`), because `peak_bytes_in_use` is
a process-wide high-water mark with no reset; the parent is pinned to CPU, so it holds no GPU memory while a
child runs. A sparse arm replays `sqd._core._solve_sqd`'s sparse branch with a sync between stages:
**build** is `uniquify_states` plus `_sparse_operator`, **solve** `_run_sparse`, **check**
`_sparse_residual` plus `_checked_eigval` — private calls that must track `_solve_sqd` if it changes.
`iters` is captured by wrapping `ground_locg` with one host callback per solve (`poc/real_groups.py`'s
pattern), and the script warns when the warm-up and timed solves disagree on it.

Runs here: `--device 0` (the default sweep, read through `type2` `2^19`) and `--patterns type1 type2
--log2-sizes 19 20 21 --device 0` (read through `type2` `2^19`).
