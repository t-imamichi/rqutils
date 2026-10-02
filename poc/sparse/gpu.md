# Sparse matvec kernels on a GPU

`poc/sparse/gpu.py` (§11), one NVIDIA GH200 120GB, 2026-10-01/02: the GPU timing that `poc/sparse/pairs.md` §7
left open, through the shipped `sqd(matvec=...)` kernels. Fixture as there: spinchain's open-XXZ `xxz` at
n=60, `δ = 0.5`, Hamming-shell subspaces around both Néel states; `type1` (`J = 62`) and `type2`
(`J = 120`), both `complex128` (61 of 62 and 118 of 120 groups real). Every timing is a warm median of 3;
every sparse eigenvalue agreed with `"indices"` to 7.1e-15. No size here is at or below `--oracle-log2`,
so `"indices"` is the only reference. `type3`/`type4` and `type2` at `2^20` are unrun. The `2^22` row,
and a repeat of `type1` `2^20`/`2^21`, come from a third run at `c63f067`, `"indices"` and `"pairs"` only;
the script is unchanged since, and the library's change between (`605ad4a`) touches only the host build.

**§1–§5's sparse times are the pre-`0d25235` kernels**, which on CUDA split and rejoined their complex
scan carry at every step (`poc/sparse/split.md` §1); their speeds and rankings describe that defect, while
memory (§4) and the build (§2) do not depend on it. §6 profiles the dense kernels, §7 is a fourth run
with the fix, and §8 a fifth with the tuned sparse kernels, which §9's recommendations rest on.

## 1. Whole solves against `"indices"`

`"indices"`' time over each arm's (median, s):

| pattern | N | `"indices"` | `"pairs"` | `"csr"` | `"ell"` |
| --- | --- | --- | --- | --- | --- |
| `type1` | `2^17` | 1.232 s | **9.33×** | 5.36× | 5.09× |
| `type1` | `2^19` | 1.337 s | **3.33×** | 1.45× | 1.78× |
| `type1` | `2^20` | 2.552 s | **1.30×** | 0.49× | 0.63× |
| `type1` | `2^21` | 5.339 s | 0.99× | 0.41× | 0.49× |
| `type1` | `2^22` | 9.657 s | **0.49×** | — | — |
| `type2` | `2^17` | 1.930 s | **6.77×** | 3.23× | 4.20× |
| `type2` | `2^19` | 2.895 s | **2.58×** | 0.95× | 1.24× |
| `type2` | `2^21` | 9.919 s | 0.57× | 0.24× | 0.27× |

- **`"pairs"` is the fastest sparse kernel at every size**, inverting the CPU ranking, where `"ell"` led
  at ~2× `"csr"` and `"pairs"` trailed (`poc/sparse/pairs.md` §10). `"ell"` beats `"csr"` by only
  1.14–1.31×, and loses at `type1` `2^17` (0.95×).
- **Every sparse win shrinks with N and is gone by `2^21`**; at `2^22` `"pairs"` runs at half
  `"indices"`' speed. §3 is why.
- The third run reproduces `type1` `2^20`/`2^21` within 1.4% (`"indices"` 2.543 and 5.266 s; `"pairs"`
  1.31× and 0.98×).
- The `type2` `2^21` row is from the script's previous revision: one process for all arms, no stage split
  or per-arm memory. Its solve path is the same `sqd` call.

## 2. The host build is not the cost

`type1` `2^21`, median per stage (§11 defines them):

| arm | build | solve | check | build share |
| --- | --- | --- | --- | --- |
| `"pairs"` | 0.322 s | 4.819 s | 0.191 s | 6.0% |
| `"csr"` | 0.472 s | 12.230 s | 0.190 s | 3.6% |
| `"ell"` | 0.922 s | 9.667 s | 0.193 s | 8.5% |

Build and check scale with N (`"pairs"` build 0.076 → 0.201 → 0.322 s over `2^19`–`2^21`); the solve
does not.

## 3. Two cliffs: `2^19` → `2^20`, then `2^21` → `2^22`

`type1`, solve time per LOBPCG iteration; ns are per state:

| arm | `2^19` | `2^20` | `2^21` | `2^22` | `2^19` → `2^20` | `2^20` → `2^21` | `2^21` → `2^22` |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `"pairs"` | 4.00 ms (7.6 ns) | 17.45 ms (16.6 ns) | 40.15 ms (19.1 ns) | 145.56 ms (34.7 ns) | **4.4×** | 2.3× | **3.6×** |
| `"csr"` | 11.66 ms (22.2 ns) | 50.87 ms (48.5 ns) | 101.92 ms (48.6 ns) | — | **4.4×** | 2.0× | — |
| `"ell"` | 7.16 ms (13.7 ns) | 35.93 ms (34.3 ns) | 80.56 ms (38.4 ns) | — | **5.0×** | 2.2× | — |
| `"indices"`* | 20.90 ms | 27.44 ms | 44.49 ms | 76.64 ms (18.3 ns) | 1.3× | 1.6× | 1.7× |

\* Setup included: it runs inside the one jitted call. `2^22` is the third run's (§1).

- **Two steps, not one.** Per state, every sparse kernel costs 2.2–2.5× more at `2^20` than at `2^19` and
  is flat to within 15% to `2^21`; then `"pairs"` steps again, 1.81× per state from `2^21` to `2^22`
  (`"csr"`/`"ell"` unrun there). The first-run reading, "one step, then linear", stood on sizes that
  stopped at `2^21`, and is retracted.
- **Iterations grow too, the same in every arm to ±1**: 64 → 93 → 120 → 126/127. `"pairs"`' 19× from
  `2^19` to `2^21` is 1.9× iterations times 10× per iteration.
- **The first step is where the batched vectors leave L2.** A `complex128` `(2, N)` input plus its output is
  64 B/state: 32 MiB at `2^19`, 64 MiB at `2^20`, against the H100's 50 MB L2 (NVIDIA's specification, not
  measured here). This is the CPU's §3 finding at a larger cache: there P2 doubled per state from `2^17` to
  `2^19`, as the same arrays grew from 8 to 32 MiB. **Inferred, not measured** — the size of the step and
  its location fit, but no L2 hit rate was read. **Retracted for the sparse kernels**: a profile puts
  72–88% of a `"pairs"` matvec from `2^20` in XLA splitting and rejoining the complex scan carry at every
  step, so both steps are the scan's step count, not L2 (`poc/sparse/split.md` §1, which also has the fix).
- **The second step is the same defect**: steps double, 160 → 320, from `2^21` to `2^22`.
- **`"indices"` degrades less** because it never scatters, so it pays no per-step carry split; three
  quarters of its matvec is recomputing the diagonal (§6). Its cost per state *falls* (26.1 → 20.9 → 18.3 ns over `2^20`–`2^22` in the third run), setup
  amortizing. That is why `"pairs"`' lead goes from 3.3× to a tie and then to 0.53× per iteration.

## 4. Memory

`peak_bytes_in_use`, one fresh process per arm (§11), GiB:

| pattern | N | `"indices"` | `"pairs"` | `"csr"` | `"ell"` |
| --- | --- | --- | --- | --- | --- |
| `type1` | `2^17` | 0.09 | 0.10 | 0.10 | 0.11 |
| `type1` | `2^19` | 0.22 | 0.13 | 0.14 | 0.14 |
| `type1` | `2^20` | 0.45 | 0.30 | 0.34 | 0.32 |
| `type1` | `2^21` | 0.89 | **0.54** | 0.59 | 0.56 |
| `type1` | `2^22` | 1.79 | **1.09** | — | — |
| `type2` | `2^17` | 0.10 | 0.11 | 0.12 | 0.12 |
| `type2` | `2^19` | 0.34 | 0.18 | 0.22 | 0.20 |

From `2^19` on, every sparse arm is lighter than `"indices"`: −34% to −39% at `type1` `2^21`, −35% to
−47% at `type2` `2^19`, and `"pairs"` still −39% at `type1` `2^22`. At `2^17` all four sit at a ~0.1 GiB floor. The peak covers the uploaded fixture,
the build, the warm-up and the timed solves.

## 5. Repeated solves are not bit-identical

The warm-up and the timed solves differed by one iteration in 3 of 12 sparse cells (`type1` `2^20`
`"csr"`: 92/93; `type2` `2^19` `"pairs"` and `"csr"`: 103/104; in the third run, `type1` `2^22`
`"pairs"`: 126/127), and `type1` `2^20` `"ell"` stopped at 92
against the others' 93. Energies still agree to 7.1e-15. **The cause is unverified**; a GPU scatter-add
summing in nondeterministic order would produce exactly this, and `"ell"`'s per-row reduce has a fixed
order, which fits it being the one kernel that never varied across repeats. CPU runs of the same script
at `2^10`–`2^13` gave identical counts.

## 6. The dense kernels' matvec, profiled

`poc/sparse/kernel_profile.py --matvec indices` and `--matvec tables` at `bc596f8`, `type1`, device time per
matvec over 10 calls. **No `wrapped_real`/`wrapped_imag`/`wrapped_complex` appears in either**: their scan
gathers and adds elementwise, never scatters, so `poc/sparse/split.md` §1's defect does not reach them.

`"indices"`, share of the 1-D matvec at `2^20` / `2^22` (6.38 / 24.07 ms):

| work | kernels | share |
| --- | --- | --- |
| diagonal: a full pass per Z term, ΣK = 179 per call | `loop_add_fusion` + `input_reduce_fusion` (179/call) | 51% / 57% |
| diagonal: zeroing each group's accumulator | `loop_broadcast_fusion` (63/call) | 12% / 11% |
| `get_diagonal`'s `while_loop`: host round trip per step | `MemcpyD2H` + `memcpy32_post` + `loop_and_fusion` (186–241/call) | 13% / 4% |
| gather, scale and add per group | `loop_add_fusion_2` (62/call) | 20% / 25% |

- **About three quarters is recomputing the diagonal**, the per-term streaming loop `poc/parity-xor.md`
  §2 found optimal on CPU. The gather is the minority; it alone doubles from 1-D to `(2, N)`.
- **The `while_loop` costs a device-to-host copy per step** on the GPU, 241 per call: fixed latency
  (0.8–1.0 ms per call), so 13% at `2^20` and 4% at `2^22`. It bounds what a sync-free loop could recover.

`"tables"` caches those diagonals, leaving one fused gather-multiply-add per group (61 per call, no host
copies):

| matvec | `"indices"` | `"tables"` | ratio | `"pairs"`, fixed (`poc/sparse/split.md` §4) |
| --- | --- | --- | --- | --- |
| `2^20` 1-D | 6.38 ms | 1.14 ms | 5.6× | 1.37 ms |
| `2^20` `(2, N)` | 7.70 ms | 2.69 ms | 2.9× | 1.58 ms |
| `2^22` 1-D | 24.07 ms | 5.13 ms | 4.7× | 10.77 ms |
| `2^22` `(2, N)` | 28.35 ms | 9.19 ms | 3.1× | 9.52 ms |

- **At `2^22` `"tables"`' matvec matches the fixed `"pairs"`**, and it works under a mesh, where the sparse
  kernels raise. Its price is ~`20·J` B/state (`poc/sparse/pairs.md`'s arms table): ~5 GB at `type1`
  `2^22`, ~25 GB at 20M states, divided across a mesh's devices but for the replicated states.
- These are device times per matvec, not solves; §7 is the solve run.

## 7. Whole solves after the fix

A fourth run after `0d25235` (the library is unchanged through `e62e172`): `--patterns type1 type2 --arms
indices tables pairs ell --log2-sizes 20 21 22`, all four kernels in one process per size. Median whole
`sqd` call in s (the sparse ones include their host build and check), and the ratio to `"indices"`:

| pattern | N | `"indices"` | `"tables"` | `"pairs"` | `"ell"` |
| --- | --- | --- | --- | --- | --- |
| `type1` | `2^20` | 2.572 | **0.567 (4.54×)** | 0.727 (3.54×) | 1.349 (1.91×) |
| `type1` | `2^21` | 5.357 | 1.382 (3.88×) | **1.361 (3.94×)** | 2.097 (2.55×) |
| `type1` | `2^22` | 9.675 | **2.653 (3.65×)** | 3.757 (2.58×) | 4.353 (2.22×) |
| `type2` | `2^20` | 6.353 | **1.646 (3.86×)** | 2.394 (2.65×) | 3.240 (1.96×) |
| `type2` | `2^21` | 10.179 | **3.073 (3.31×)** | 4.176 (2.44×) | 5.974 (1.70×) |
| `type2` | `2^22` | 15.554 | **5.100 (3.05×)** | 6.376 (2.44×) | 12.245 (1.27×) |

- **`"tables"` is fastest in 5 of 6 cells**; `"pairs"` takes `type1` `2^21` by 1.5%. Every kernel beats
  `"indices"` everywhere, and `type2`, where the pre-fix `"pairs"` was 0.57×, behaves like `type1`.
- **Peak memory against `"indices"`**: `"tables"` 2.1–2.4×, `"pairs"` 0.53–0.60×, `"ell"` 0.56–0.64×
  (`type2` `2^22`: 2.69, 6.44, 1.42, 1.62 GiB). Linear in N, `"tables"` would reach ~31 GiB for `type2`
  at 20M states — a projection.
- **The sparse kernels are now bound by their host stages**: at `type2` `2^22` the build and check are
  1.60 + 0.58 s of `"pairs"`' 6.38 s (34%) and 4.75 + 0.58 s of `"ell"`'s 12.25 s (44%). On its solve
  stage alone `"ell"` is the fastest kernel at `type1` `2^22` (15.91 ms per iteration against `"pairs"`'
  20.48), and the build gives it back.
- **A first call reorders them**: compile included, `type2` `2^22` reads `"indices"` 17.41 s, `"tables"`
  6.98 (2.49×), `"pairs"` 8.94 (1.95×), `"ell"` **20.84 (0.84×)** — `"ell"`'s bucket shapes are costly to
  compile, and a workflow calling `sqd` once per subspace pays it every time.
- Iteration counts match across arms (one `"pairs"` 126/127 repeat at `type1` `2^22`, §5); every
  eigenvalue agrees with `"indices"` to 5.3e-15.

## 8. Whole solves with the tuned sparse kernels

A fifth run at `a1fcde3` (the library unchanged through `140170f`): one host search for the build and the
check (`b38d48b`), `"pairs"` in `2^19`-entry chunks on a GPU (`96fb70f`) and no sorted-scatter hint on
CUDA (`a1fcde3`). `--patterns type1 type2 --arms indices tables pairs csr ell --log2-sizes 20 21 22`,
median whole `sqd` call in s and the ratio to `"indices"`:

| pattern | N | `"indices"` | `"tables"` | `"pairs"` | §7's `"pairs"` | `"csr"` | `"ell"` |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `type1` | `2^20` | 2.533 | 0.563 (4.50×) | **0.434 (5.84×)** | 0.727 | 0.971 (2.61×) | 1.297 (1.95×) |
| `type1` | `2^21` | 5.164 | 1.352 (3.82×) | **0.894 (5.78×)** | 1.361 | 1.802 (2.87×) | 1.983 (2.60×) |
| `type1` | `2^22` | 9.400 | **2.599 (3.62×)** | 2.786 (3.37×) | 3.757 | 4.719 (1.99×) | 4.073 (2.31×) |
| `type2` | `2^20` | 6.339 | 1.621 (3.91×) | **1.539 (4.12×)** | 2.394 | 3.295 (1.92×) | 3.179 (1.99×) |
| `type2` | `2^21` | 9.939 | 3.020 (3.29×) | **2.808 (3.54×)** | 4.176 | 5.870 (1.69×) | 5.821 (1.71×) |
| `type2` | `2^22` | 15.617 | 5.008 (3.12×) | **3.897 (4.01×)** | 6.376 | 9.880 (1.58×) | 11.862 (1.32×) |

- **`"pairs"` is fastest in 5 of 6 cells**, 1.35–1.68× its §7 call; `"tables"` keeps `type1` `2^22` by 7%,
  at 3.9× its memory. Its solve stage gains up to 2.30× (`type2` `2^22`: 4.07 → 1.77 s), the chunk paying
  most where pairs per state, and so scan steps, are many.
- **The check lost its search**: 0.401 → 0.165 s and 0.581 → 0.369 s at `2^22`, within the 0.58 s
  `b38d48b` bounded it by.
- **Peak memory**: `"pairs"` 0.53–0.64× `"indices"`' and 0.22–0.31× `"tables"`' (`type2` `2^22`: 1.43
  against 2.69 and 6.44 GiB); `"csr"` and `"ell"` 0.56–0.69× `"indices"`'.
- **First calls follow**: compile included, `type2` `2^22` reads `"pairs"` 6.46 s, `"tables"` 6.96,
  `"indices"` 17.49, `"ell"` 20.60.
- **`"csr"` is 1.6–2.9× `"indices"`** (its `type1` `2^22` solve per iteration halves, 58 → 27 ms, from the
  dropped hint) and behind `"pairs"` everywhere; **`"ell"` barely moves**, neither change touching it, its
  build up to 40% of a call (4.77 of 11.86 s).
- **`"pairs"`' host build is its limit now**: 1.63 of 3.90 s at `type2` `2^22` (42%), nearly all search.
- Every eigenvalue agrees with `"indices"` to 7.1e-15. A `cuda_timer` "Delay kernel timed out" line is an
  XLA autotuner warning; its row's numbers are in line with the rest.

## 9. What it means

Before `0d25235`, `"pairs"` led the sparse kernels to `2^19` and lost past it (0.49× at `type1` `2^22`),
and "keep `"indices"`" followed; both were the per-step carry split, and that advice is retracted. With
the tuned kernels (§8), on one GH200:

| setting | kernel | measured |
| --- | --- | --- |
| one GPU | `"pairs"` | 3.37–5.84× `"indices"` end to end, fastest in 5 of 6, at 0.53–0.64× its memory and 0.22–0.31× `"tables"`' |
| under a mesh | `"tables"` | 3.12–4.50×; the sparse kernels raise there; its multi-GPU solve is unmeasured |
| any | not `"csr"`/`"ell"` | both behind `"pairs"` at about its memory |

`"indices"` stays the library default: that is a CPU choice, and switching per backend is a separate
decision.

## 10. Open

1. **The L2 cause — settled otherwise**: both of §3's steps are the per-step carry split
   (`poc/sparse/split.md` §1).
2. **`type3`/`type4`**; `type2` at `2^20`/`2^21` is §7's.
3. **Past `2^22`**, toward spinchain's 20M: `--arms indices tables pairs --log2-sizes 23 24`, where
   `"tables"`' memory (~`20·J` B/state) starts to matter.
4. **The levers past the cliff, unmeasured**: a locality-preserving state order (RCM did nothing on CPU,
   `poc/sparse/pairs.md` §4, since these graphs are hypercube-like); a matvec blocked so each block's
   slice of `vec` fits L2 (a tiled `"pairs"` order: 1.08× per iteration on CPU, 1.00× here before the fix and
   1.01–1.03× after, `poc/sparse/tiles.md` §3); `"pairs"` without atomics (`poc/sparse/pairs.md` §7.6, which also lists the
   scan-step and layout sweeps).
5. **The second step — settled**: the same defect, steps doubling 160 → 320 (§3).
6. **Nondeterministic iteration counts** (§5): confirm the scatter-add cause, and whether
   XLA's `--xla_gpu_deterministic_ops` removes it, at what cost.
7. **Only one GPU.** An A100 attempt gave no number: its child processes fell back to CPU on `cuInit(0)`'s
   `CUDA_ERROR_NO_DEVICE` — undiagnosed; the likeliest cause is `--device` overriding a scheduler's
   `CUDA_VISIBLE_DEVICES`.
8. **The sparse kernels' host build** (§8), 42% of a `"pairs"` call at `type2` `2^22` and nearly all
   search: the device search on a GPU is the lever.
9. **`"ell"`'s compile cost** (§7's first calls), and whether the persistent compile cache recovers it.
10. **`"tables"` on a multi-GPU mesh**, the setting §9 recommends it for.

## 11. The script

`poc/sparse/gpu.py`, its argparse checked against this section:

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

Runs here: `--device 0` (the default sweep, read through `type2` `2^19`), `--patterns type1 type2
--log2-sizes 19 20 21 --device 0` (read through `type2` `2^19`), and, at `c63f067`, `--patterns type1
--arms indices pairs --log2-sizes 20 21 22` (§1's third run); after `0d25235`, `--patterns type1 type2
--arms indices tables pairs ell --log2-sizes 20 21 22` (§7); at `a1fcde3`, the same with `csr` added
(§8). §6 is `poc/sparse/kernel_profile.py --matvec
indices` and `--matvec tables`, `--log2-sizes 20 22`, at `bc596f8`.
