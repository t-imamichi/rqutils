# The Chebyshev prefilter on a GPU, and compile memory in sweeps

`poc/prefilter_gpu.py`, `poc/prefilter_cycles_e2e.py` and `poc/gpu_unverified.py`, one CUDA device (a
71 GB GPU for the sweeps; the earlier `gpu_unverified` run was a GH200), 2026-09-04 and 2026-09-12. CPU
comparisons are the Apple M1 CPU backend. Full per-cell tables are in
`markdown/locg-chebyshev-prefilter.md` §3.2–§3.4; the scripts are described in §10.

## 1. The first GPU sweep: the peak transfers, its location does not (2026-09-04)

`prefilter_gpu.py`'s default 3x3 grid (`degree` 8/16/32 × `cycles` 2/4/8) on `n=26`, `N=1048576`
(padded), `J=30`, driving `ground_locg` on a pre-assembled `apply_h` matvec with setup excluded. The GPU
tops out at **1.38x at `(32, 8)`** against the CPU median of 1.36x, so the prefilter pays about as well
on GPU as on CPU — but CPU peaks near `(16, 4)`, and that same setting gives only **1.08x** here. The CPU
recipe's "do not exceed `cycles ≈ 4`" is a CPU statement. `sqd`'s `(32, 2)` default measures **1.07x**,
under a fifth of what is available on this fixture. The "optimum near `(32, 8)`" reading is retracted
in §2.

Two process lessons, both about reading a truncated result:

- **A sweep that dies partway invites the wrong conclusion, and one was drawn.** With `(32,4)` and
  `(32,8)` missing (see §5 for why they died), the reading was "the CPU 1.36x does not transfer" and
  "only `(16,8)` clears 1.28x". Both were wrong, in the *optimistic-for-the-hypothesis* direction: the
  two absent configurations were the two best in the grid. The measurement had no error in it — the
  inference from a grid with a hole did. **A per-item `try` that prints a SKIPPED row is worth more than
  the rows it saves**, because it makes the hole visible as a hole.
- **The two configurations that looked like they exhausted the GPU are the two fastest.** Believing the
  failure was about their cost would have removed exactly the settings worth using.

## 2. The optimum is fixture-dependent; `(32, 8)` was a boundary artifact (2026-09-12)

**Retracted:** "the optimum is near `(32, 8)`". That grid's maximum sat on its own upper corner in
*both* axes — a truncation, not a maximum, the same misreading as the SKIPPED rows one axis over. An
upper-corner run (`(32,16)` 1.12x, `(64,8)` 1.08x, `(64,16)` **0.74x**) and a 4x5 interior grid bracket
it. **Also retracted:** a rule proposed from the 3x3 grid, a ridge at `extra mv = cycles·(degree+1)` ≈
200–350. A second Hamiltonian falsified it; recorded as a correction rather than deleted, because the
over-fit is the lesson.

Matched 4x5 grids, left `n=26`/`J=30` (baseline 298 iterations / 279.2 ms), right `n=22`/`J=8` (baseline
167 iterations / 72.7 ms), both `N=1048576` padded, `|dE|` ≤ 8.9e-16 and ≤ 3.6e-15 in all twenty:

| degree | cycles=2 | cycles=4 | cycles=8 | cycles=12 | cycles=16 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 1.01 / 1.12 | 1.08 / 1.33 | 1.29 / **1.43** | 1.40 / 1.36 | **1.41** / 1.25 |
| 24 | 1.01 / 1.19 | 1.22 / 1.34 | **1.42** / 1.28 | 1.38 / 1.12 | 1.28 / **0.98** |
| 32 | 1.07 / 1.25 | 1.34 / 1.30 | 1.38 / 1.28 | 1.27 / 1.05 | 1.11 / **0.89** |
| 40 | 1.11 / **1.38** | 1.40 / 1.23 | 1.34 / **0.98** | 1.15 / **0.81** | 1.007 / **0.70** |

- `n=26`/`J=30` peaks along an **anti-diagonal** — `(16,16)` 1.41x, `(24,8)` 1.42x, `(32,8)` 1.38x,
  `(40,4)` 1.40x, all inside the 1.3% noise floor, so a tie. The bottom-right still holds 1.07–1.28x.
- `n=22`/`J=8` peaks at `(16,8)` 1.43x and `(40,2)` 1.38x, then **collapses**: 5 of 20 configurations
  lose outright, `(40,16)` at **0.70x**.
- **`extra mv` is not the controlling variable.** The same `extra mv` gives opposite verdicts: 283 →
  1.41x on the first fixture, 1.25x on the second; 411 → 1.28x there, **0.98x** here. `degree = 40` is
  near-best at `cycles = 2` on the second fixture and a **loss** at `cycles = 8`.
- **The two surfaces are near-transposes, so no `(degree, cycles)` is best on both** — `(16,8)`
  1.29/1.43, `(24,8)` 1.42/1.28, `(40,2)` 1.11/1.38 — and the gaps are far outside the 0.2–1.4% noise
  floors.

**Retracted third: "the sweep contradicts §3.1".** §3.1 (the CPU sweep `sqd`'s default rests on) reasons
that `degree` is high-leverage and `cycles` saturates past 2. On `n=26`/`J=30` the reverse holds (2 → 8
cycles is 1.07x → 1.38x at degree 32, while `degree` at fixed `cycles = 2` moves only 1.01 → 1.11). On
`n=22`/`J=8` **§3.1's recipe is correct**: `(40, 2)` tops that column at 1.38x. §3.1's 27 configurations
are XXZ chains on connected subspaces; `prefilter_gpu`'s is a random 100-term operator. **The knob
ordering inverts between the regimes** — the disagreement was the fixture, and the XXZ regime is the one
closer to a real SQD workflow. Two intermediate readings were both wrong: first that §3.1 predicted the
ridge (coincidence — both accounts penalize large `extra mv` for unrelated reasons), then that §3.1 was
contradicted (fixture, not error).

**Iteration count is anti-correlated with wall clock out here.** Iterations fall monotonically with
`extra mv` across the whole grid, to 106, and the *fewest* measured (106, at `(64,16)`) belong to the
*slowest* configuration measured. (The dated record gave the range as 212 → 106; §3.2's tables give 285
at `(16,2)`.) Third independent instance of the quote-end-to-end rule.

## 3. CPU against GPU: the iteration counts are identical

**Claim 1 holds exactly.** The four upper-corner configurations return **identical** iteration counts on
CPU and GPU — 138 / 115 / 125 / 106, not merely within an iteration — `|dE| ≤ 1.8e-15`, and wall-clock
ratios agree to 1–3% at every point (`(32,8)` 1.41x on both). Keep the claim narrow: §3.1's CPU peak near
`(16, 4)` against the GPU's 1.08x there is a real divergence and still stands. **The backends disagree
about where the ridge begins and agree about where it ends.**

## 4. End-to-end through `sqd`, `(32, 4)` loses by 45% (2026-09-12)

`(32, 4)` (1.34/1.30 on the two fixtures of §2) was the best worst-case cell measured and the one
candidate worth taking end-to-end. `prefilter_cycles_e2e.py`, one CUDA device, `n=20`, `N=3586`, XXZ
Krylov subspaces from `|Neel>`, **setup inside the timed region**, arms interleaved, 9 rounds per
configuration:

| `delta` | `(32, 2)` median | `(32, 4)` median | ratio | paired |
| ---: | ---: | ---: | ---: | ---: |
| 0.5 | 274.9–288.0 ms | 405.2–423.6 ms | 0.68x | 0/27 |
| 1.0 | 291.6–292.0 ms | 422.9–423.2 ms | 0.69x | 0/27 |
| 1.5 | 296.4–296.9 ms | 422.5–423.3 ms | 0.70x | 0/27 |

**0.68–0.71x at every anisotropy, 0 of 81 paired rounds won, spreads 0.3–1.6%.** `(32, 4)` is ~45%
slower end-to-end.

**The dilution was not symmetric.** The stated prediction was that both arms compress toward 1.0x — a
solver-side gap diluted by the setup term. Instead `(32, 4)`'s extra 66 filter matvecs cost *more*
end-to-end than the iterations they remove save, so **1.30x on the solver became 0.69x through `sqd`**.
§3.1's mechanism accounts for it: past cycle 2 you pay full cost for little separation, and end-to-end
there is no solver-side surplus left to absorb it. `CLAUDE.md`'s Amdahl rule says to ask how many times
per solve the target is paid before believing a ratio; this is the sharper version — **a ratio measured
on a 4.5–8.4% slice can change sign when the excluded 66–97% is restored, not merely shrink toward
1.0x.** The Bloom entry's 1.09x cap was the benign case.

**A fixture defect, worth knowing before reusing the script.** `xxz_krylov` at `rungs=4, cap=4000` never
reaches the cap, so `rng.choice` never fires and the fixture is **seed-independent** — identical
`N=3586` and energies identical to 10 digits across all three seeds. So the sweep is **3 configurations
measured 27 times each, not 9**; the `delta` sweep is real, the seed sweep is not. It does not change
this verdict (0/81 across a 45% gap needs no seed variation), but it would silently narrow a *close*
result. Raise `rungs` or lower `cap` until the subspace exceeds it (`CLAUDE.md`: "check the fixture
exercises the thing under test").

## 5. Sweeping a static argument exhausts the GPU on compiled modules, not tensors (2026-09-04)

`prefilter` is a `static_argnames` entry on `ground_locg` and `run_sqd`, so each `(degree, cycles)`
retraces and the process retains one more executable. The 8th configuration of the 3x3 grid failed on a
**71 GB** GPU with `RESOURCE_EXHAUSTED: Failed to load in-memory CUBIN`, while the device held under 1 GB
of tensors. `jax.clear_caches()` between configurations fixes it; `timeit`'s single warmup absorbs the
forced recompile (ratios within noise of the pre-fix run, iteration counts identical).

What made the diagnosis, and two wrong turns before it:

- **Wrong turn 1: retention.** Every `ground_locg` return holds an `O(N)` eigenvector and eight were
  reachable at the failure — a real defect, fixed — but freeing them **left the failure at exactly the
  same configuration**. A *reproducible* boundary is not a leak; a leak creeps. Fixing a real problem that
  is not *the* problem is the trap: the fix looked justified and the symptom was unmoved.
- **Wrong turn 2: graph growth.** Plausible because `degree` bounds a Chebyshev recurrence. Refuted by
  measurement: **compiled HLO size and `temp_size_in_bytes` are identical across all nine configurations
  (1836504 B, HLO within 2 characters)**, because both parameters are `lax.scan` trip counts and change
  neither graph size nor working set. Per `CLAUDE.md`, ask XLA rather than a formula — that applies to
  *excluding* a hypothesis, not only to sizing.
- **The error message named the layer and was read past.** "Failed to load in-memory **CUBIN**" is a
  module load, not an allocation. The solve never began.

**Scope: sweeps only, not the library.** Measured with `run_sqd._cache_size()`: five `sqd` calls at the
default `prefilter` hold the cache at **1**; six distinct values grow it 1→6. The `(32,2)` step does not
increment, since an earlier default call already created that entry — growth tracks *distinct static
values*, not call count. Generic to every static argname (`cache_level`, `maxiter`, `states_size`,
`xcache_groups`), not to `prefilter`. `poc/caching.py` is not exposed: it builds cache variants as arrays
and hand-assembles matvecs rather than passing `cache_level` through a jit boundary.

## 6. Harness defects the runs exposed, fixed in `prefilter_gpu.py`

- **The CPU banner claimed every CPU number was already documented.** True of the default grid only —
  the 2026-09-12 CPU corner run was new, and the banner told its reader to discard it. A "this is already
  known" message must scope itself to the parameters it was written for.
- **Claim 3's sharded arm hardcodes `(16, 4)`**, now known to be off-ridge. It asserts the output *spec*
  survives sharding; its ratio is not a recommendation and its 298→254 iterations are not comparable to
  the sweep table. Commented in place.

The 4-virtual-device Claim 3 arm reported 8340 ms → 7617 ms. **Discard those milliseconds** — one physical
backend, so it measures contention, not scaling (`CLAUDE.md`). The `spec=P('x',)` on both arms and
`|dE| ≤ 1.8e-15` are the results; sharding-transparency holds.

## 7. `gpu_unverified.py`'s re-run: the `lax.sort` leak still does not reproduce (2026-09-04)

The non-reproduction is already stated in `get_xsource`'s docstring from the GH200 run; this second CUDA
device reproduces that non-reproduction (flat at +0.000 GB retained and +0.000 GB drift across 5 reps in
both arms, 0.950 GB live and identical between them) and needs no separate record. Its Claims 2 and 3 stay
open (§9).

## 8. What it means

- **`sqd`'s `(32, 2)` default stays, on a direct measurement.** It measures 1.07x and 1.25x solver-side on
  the two fixtures of §2: never best, never a loss, mid-surface on both. Every alternative beating it on
  one fixture is mediocre or losing on the other, which is what a default across unknown Hamiltonians
  must avoid and what §3.1's paired sweep selected for. A change was requested three times across the
  session; the second fixture turned the answer from "insufficient evidence" into a positive reason. Had
  it agreed with the first, the switch would have been to `(24, 8)` — **1.28x** on the second fixture
  against `(32,2)`'s 1.25x, inside the noise floor, so it would have bought nothing and been justified by
  a single fixture. The best worst-case cell, `(32, 4)`, then lost 45% end-to-end (§4).
- **Solver-side ratios do not predict `sqd`.** `prefilter_gpu.py` excludes setup; a solver-side win can
  invert end-to-end (§4). The documented end-to-end figure for the default remains 1.49x median.
- **A one-fixture optimization surface produced two confident wrong rules in a row** (`extra mv` ≈
  200–350; "§3.1 is contradicted"), both caught by a *single* run at a different `K`/`J` rather than by
  any refinement of the original grid. `CLAUDE.md`'s "a quantity measured at one size is not a law"
  applies to the *shape* of an optimization surface, not just to scalars — and since `K` sets which
  axis dominates, it is the parameter to vary first, not last.
- **Sweeps over a static argument need `jax.clear_caches()`**; the library's own calls do not (§5).

## 9. Open

1. **`gpu_unverified.py` Claim 2** (searchsorted against the legacy sort) reports 16.0x/14.6x/10.8x at
   `N=200k/1M/5M`, but the two larger sizes carry the script's own not-kernel-dominated warning — the
   sort arm grew 1.03x and 1.23x for a 5x `N` increase, against a fixed ~1.46 s floor. A dominating floor
   *suppresses* the ratio, so those are lower bounds on an unmeasured value and **should not be quoted**
   until the floor is identified.
2. **`gpu_unverified.py` Claim 3** (multi-GPU speed) is unrun: one physical device. So is
   `prefilter_gpu.py`'s Claim 3 on real devices.
3. **Genuine seed variation end-to-end** needs `prefilter_cycles_e2e.py` with the cap reached (§4).

## 10. The scripts

**`poc/prefilter_gpu.py`** — the `(degree, cycles)` sweep of `ground_locg` on a pre-assembled
`cache_level=(1, 2)` `apply_h` matvec (setup excluded), on a random real 100-term operator from
`make_problem`. Prints iterations, ms, `|dE|` and a `fmt_ratio` verdict per cell, a SKIPPED row on a
runtime error, and calls `jax.clear_caches()` before each cell. With ≥ 2 devices it runs Claim 3 (plain
and `(16, 4)` on a `P('x')` mesh, asserting the output spec).

| argument | default | meaning |
| --- | --- | --- |
| `--devices` | none | `"0,1,2,3"` (`CUDA_VISIBLE_DEVICES`) or `mpi` (`jax.distributed`, needs `--extra mpi`) |
| `--num-qubits` | 26 | `n` |
| `--num-states` | 1000000 | states drawn; padded to a power of two |
| `--num-xgroups` | 30 | `J` |
| `--degrees` | `8,16,32` | comma-separated filter degrees |
| `--cycles` | `2,4,8` | comma-separated cycle counts |

§2's grids are `--degrees 16,24,32,40 --cycles 2,4,8,12,16`, the second fixture adding
`--num-qubits 22 --num-xgroups 8`.

**`poc/prefilter_cycles_e2e.py`** — whole `sqd(..., return_eigvec=False)` calls, setup included, arms
warmed then interleaved, each energy gated against `eigsh` on `hproj` (`|dE| > 1e-9` excludes the
configuration). Reports min/median, spread and paired wins per configuration and overall.

| argument | default | meaning |
| --- | --- | --- |
| `--num-qubits` | 20 | periodic XXZ chain length |
| `--rungs` | 4 | hops from the Néel state in `xxz_krylov` |
| `--cap` | 4000 | subspace cap (not reached at the defaults, §4) |
| `--bx` | 0.5 | transverse field breaking magnetization conservation |
| `--deltas` | `0.5,1.0,1.5` | XXZ anisotropies |
| `--seeds` | 3 | seeds per anisotropy |
| `--rounds` | 9 | interleaved A/B rounds per configuration |
| `--arms` | `32,2 32,4` | `degree,cycles` settings; the first is the baseline |

**`poc/gpu_unverified.py`** — three GPU-only claims, each against `searchsorted.xsource_sort_legacy`
rather than the library: Claim 1 samples live `bytes_in_use` over 5 `lax.sort` / searchsorted sweeps
(skipped on CPU); Claim 2 times both at `N` = 200k, 1M and `min(--num-states, 5M)` with a
not-kernel-dominated warning; Claim 3 times `sqd` single-device against a mesh of all devices (skipped
below 2). Arguments: `--devices` (`CUDA_VISIBLE_DEVICES`, a filter that cannot add a GPU),
`--num-qubits` (28), `--num-states` (5000000), `--num-xgroups` (50, used by Claim 2).

**`poc/caching.py`** takes no arguments; it is cited in §5 only as unaffected by the cache growth.
