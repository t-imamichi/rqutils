# The sparse kernels' complex scan carry on a GPU

`poc/sparse_profile.py` and `poc/sparse_split.py` (§7), 2026-10-01/02: one NVIDIA GH200 120GB for §1 and
§4, one Apple M1 (8 cores, 16 GiB) for §3. Fixture as `poc/sparse-gpu.md`: spinchain's open-XXZ `xxz` at n=60,
`δ = 0.5`, `type1` (`J = 62`, `complex128`), Hamming-shell subspaces around both Néel states. The fix
(§2) is in the library at `0d25235`, and §4 measures it.

## 1. The defect: three per-step passes over `out`

`poc/sparse_profile.py` at `474aafb` on the GH200, `"pairs"`, device time per matvec over 10 calls:

| N | pairs/state | scan steps | 1-D | `(2, N)` | `wrapped_real`+`imag`+`complex` share, 1-D / `(2, N)` |
| --- | --- | --- | --- | --- | --- |
| `2^19` | 2.2 | 36 | 1.23 ms | 1.63 ms | 45% / 60% |
| `2^20` | 4.0 | 128 | 4.76 ms | 9.10 ms | 72% / 82% |
| `2^21` | 2.5 | 160 | 12.06 ms | 21.16 ms | 78% / 87% |
| `2^22` | 2.4 | 320 | 47.44 ms | 78.46 ms | 79% / 88% |

- **XLA has no GPU scatter-add for `complex128`**: each complex scatter becomes two real ones (four
  `input_scatter_fusion*` kernels for `"pairs"`' two), and the scan's complex carry — the whole `out` —
  is split by `wrapped_real`/`wrapped_imag` and rejoined by `wrapped_complex` **at every step**.
- **Each is a full pass over `out`**: `wrapped_real` at `2^22` takes 33.5 µs per step to read a 64 MiB
  vector, ~2 TB/s, HBM speed. The scatters themselves are 11–25% of the time from `2^20` up.
- **So the cost is `O(N × steps)`, and steps grow with N** (pairs / 32768). From `2^20` the 1-D matvec
  costs a flat ~0.035 ns per state per step (4.5, 5.8, 11.3 ns/state at 128, 160, 320 steps); `2^19`, the
  one size whose vectors fit L2, is 0.065. Both of `poc/sparse-gpu.md` §3's "cliffs" are the step count:
  36 → 128 from `2^19` to `2^20` (pairs per state also jump, 2.2 → 4.0, in this fixture), then 160 → 320.
- It accounts for everything the L2 reading could not: entry order is irrelevant (`poc/sparse-tiles.md`
  §3), the `(2, N)` matvec costs ~2× the 1-D one (twice the carry), `"indices"` does not degrade (it
  gathers, never scatters), and the CPU is unaffected (it scatters complex natively).

## 2. The fix

`_scan_add` in `rqutils/sqd/_sparse.py`, shared by all three kernels: on CUDA a complex `out` is carried
as `(out.real, out.imag)`, each update scattered as its real and imaginary parts (an `O(chunk)` split of
the chunk's values, not of `out`), and rejoined once after the scan. `jax.lax.platform_dependent`
chooses, so every other platform keeps the single complex carry (§3). A real `out` is untouched.
`TestComplexScanCarry` checks both branches as traced; dropping the split from the CUDA branch, leaking it
into the default, or reverting the kernel each fails all six cases.

## 3. On CPU the split is slower

Measured with the **ungated** prototype — the gate now makes it unreachable on CPU, so no committed
script reproduces these. `poc/sparse_split.py`, 5 interleaved rounds, ratio old / split (below 1 is the
split slower), split wins in brackets; `temp` is the `(2, N)` matvec's `temp_size_in_bytes`:

| kernel | N | solve/iter | 1-D matvec | `(2, N)` matvec | temp old / split |
| --- | --- | --- | --- | --- | --- |
| `"pairs"` | `2^17` | 0.89× (0/5) | 0.78× (1/5) | 0.82× (0/5) | 2.3 / 6.8 MiB |
| `"csr"` | `2^17` | 0.68× (0/5) | 0.65× (0/5) | 0.52× (0/5) | 2.3 / 6.3 MiB |
| `"ell"` | `2^17` | 0.87× (0/5) | 0.77× (0/5) | 0.89× (0/5) | 4.8 / 9.3 MiB |
| `"pairs"` | `2^19` | 0.87× (0/5) | 0.80× (0/5) | 0.72× (0/5) | 2.3 / 18.8 MiB |
| `"csr"` | `2^19` | 0.71× (0/5) | 0.65× (0/5) | 0.52× (0/5) | 1.1 / 17.1 MiB |
| `"ell"` | `2^19` | 0.90× (0/5) | 0.76× (0/5) | 0.80× (0/5) | 3.2 / 19.3 MiB |

- Two real scatters cost more than one native complex scatter, in every cell.
- **The split's extra temp grows with N**: +16.5 MiB at `2^19`, about one more `(2, N)` `complex128`
  (16 MiB) — the parts held beside `out`. The GPU branch will likely pay it too (§4).
- Gated, the CPU path is the previous kernel: bit-identical on 42 cases (three kernels, five coefficient
  layouts, real and complex vectors, 1-D and `(2, N)`), and `poc/sparse_split.py` reads 1.00× there.

## 4. On the GPU: 2.5–17.9× per iteration

GH200 at `0d25235`, `poc/sparse_split.py --log2-sizes 20 21 22`, 5 interleaved rounds; the split won
every round of every cell. Per solve iteration, old → split; `"indices"` is from `poc/sparse-gpu.md`'s
third run, another process, so that column is cross-run:

| kernel | `2^20` | `2^21` | `2^22` |
| --- | --- | --- | --- |
| `"pairs"` | 17.32 → **4.28 ms** (4.05×) | 39.49 → **6.59 ms** (6.00×) | 145.62 → 20.35 ms (7.15×) |
| `"ell"` | 35.95 → 6.36 ms (5.66×) | 79.24 → 7.63 ms (10.39×) | 276.44 → **15.48 ms** (17.85×) |
| `"csr"` | 50.79 → 20.23 ms (2.51×) | 100.57 → 26.92 ms (3.74×) | 329.68 → 58.00 ms (5.68×) |
| `"indices"` | 27.35 ms | 43.88 ms | 76.64 ms |

- **The speedup grows with N**, as `O(N × steps)` predicts: the `(2, N)` matvec gains 5.58/8.30/9.72× on
  `"pairs"` and 7.97/14.23/23.55× on `"ell"` over `2^20`–`2^22`.
- **Every sparse kernel now beats `"indices"`** per iteration at every size: `"pairs"` 6.4/6.7/3.8×,
  `"ell"` 4.3/5.8/5.0×, `"csr"` 1.35/1.63/1.32×. The fastest is `"pairs"` to `2^21` and `"ell"` at `2^22`,
  where `"pairs"`' per-state cost rises (4.1, 3.1, 4.9 ns) and `"ell"`'s stays flat (6.1, 3.6, 3.7).
- **No memory cost on the GPU**: the `(2, N)` matvec's temp is equal in every cell (32/64/128 MiB), unlike
  §3's CPU split. Iterations match (one 126/127 repeat at `"pairs"` `2^22`); matvecs agree to 6.4e-16,
  eigenvalues to 1.1e-14.
- **The profile confirms the mechanism.** `poc/sparse_profile.py` after the fix: `wrapped_complex` and
  `loop_imag_real_fusion` run once per call (1–4% together); the four scatters are 84–89% and the
  per-step add 9–12%. Device time at `2^22` is 10.77 ms (1-D) and 9.52 ms (`(2, N)`), against §1's 47.44
  and 78.46. Its 1-D figure exceeds the A/B's timed 7.35 ms at `2^22` — unresolved, likely profiler
  overhead.

## 5. What it means

The sparse kernels' GPU result in `poc/sparse-gpu.md` was this defect. Fixed, `"pairs"` and `"ell"` are
3.8–6.7× `"indices"` per iteration on one GH200, `"pairs"` at −39% memory — the opposite of that
write-up's "keep `"indices"`". The CPU keeps its complex carry and is unchanged.

## 6. Open

1. **A same-process run against `"indices"`, and `type2` — done**: `poc/sparse-gpu.md` §7, end to end
   and with `"tables"`, which is fastest there in 5 of 6 cells.
2. **Past `2^22`**, where `"pairs"`' per-state rise decides it against `"ell"`.
3. **ROCm**, which the gate does not cover; unmeasured whether its XLA scatter splits the same way.

## 7. The scripts

`poc/sparse_profile.py`:

| flag | default | meaning |
| --- | --- | --- |
| `--num-qubits`, `--delta` | `60`, `0.5` | the `xxz` fixture |
| `--pattern` | `type1` | one of `poc/eigenpair_check_scale.patterns` |
| `--log2-sizes` | `19 20 21 22` | subspace sizes `2^k` |
| `--calls` | `10` | traced matvecs per size and shape |
| `--top` | `10` | kernels listed per shape |

`poc/sparse_split.py`:

| flag | default | meaning |
| --- | --- | --- |
| `--num-qubits`, `--delta` | `60`, `0.5` | the `xxz` fixture |
| `--pattern` | `type1` | one of `poc/eigenpair_check_scale.patterns` |
| `--log2-sizes` | `20 21 22` | subspace sizes `2^k` |
| `--arms` | `pairs csr ell` | sparse kernels, each run with the old complex carry and the library's |
| `--rounds` | `5` | interleaved rounds after one warm-up per arm |

`sparse_profile.py` reads device events from the trace's `/device:` processes and falls back to host XLA
ops with no device (labelled so). `sparse_split.py`'s old arm is the kernels as of `474aafb`, copied
verbatim. Runs here: the profile on the GH200 at `474aafb` (§1) and at `0d25235` with `--log2-sizes 20 22` (§4);
the split A/B on CPU at `2^17 19` ungated (§3) and at `2^12` gated, and on the GH200 at `0d25235` (§4).
