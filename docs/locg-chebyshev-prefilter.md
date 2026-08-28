# A Chebyshev prefilter for `ground_locg`: measured 1.1–3.9×, no accuracy loss

**Direction: inbound.** Unlike `docs/skqd-basis-opt-optimization.md` and
`docs/skqd-sqd-solve-tolerance.md`, this proposes a change to **`rqutils` itself** —
`rqutils/ground_locg.py`. Nothing in `spinchain` changes; every caller benefits automatically.

**Date.** 2026-08-27, CPU, float64 (`jax_enable_x64`), branch `dev`.

> **SUPERSEDED IN PART (2026-08-28) — read this before trusting anything below about `hi`.**
>
> `_lambda_max_bound` **is deleted.** Its 10-step power iteration converged to the largest-*magnitude*
> eigenvalue, so on a negative-leaning spectrum the filter interval inverted and `ground_locg` returned
> an **excited** eigenpair with `converged=True`. Report and reproduction:
> `docs/rqutils-prefilter-bug.md`; reply and migration: `docs/rqutils-prefilter-bug-response.md`.
>
> What that changes in the text below:
>
> * Every mention of `hi` coming from power iteration with a ×1.05 margin (§2 line 56, §3 line 89,
>   §5 lines 188 and 201) describes **deleted** code. `hi` is now the required `prefilter_hi` argument:
>   Gershgorin `max_i Σ_j |A_ij|` derived automatically for an array, `Σ|c_k|` passed by `sqd`, and a
>   **raise** for an opaque callable. No matvec-only estimate can be rigorous — a theorem, not a
>   tuning problem.
> * The **77 matvecs** figure (§2) no longer includes "~11 for the `λ_max` estimate"; the bound costs
>   no matvec at all.
> * **The §3/§3.1 measurements were taken with the unsound bound.** They remain valid as relative
>   comparisons, but the headline improved once the estimate's matvecs were gone: **3.29× median**
>   against the 1.88× recorded here.
> * The recommended range **`degree` 32–64 (§3.1, §4) is narrowed to `(32, 2)` only.** An independent
>   sweep measured `degree=64` as the *weakest* arm (0.74–0.80× dense).
> * §4 item 4's "The filter cannot change the answer — every convergence test reads the true residual"
>   is **the exact claim the bug report demolished**: the residual test certifies that *an* eigenpair
>   was found, not the lowest one. It holds only when `prefilter_hi` is a valid bound.
>
> The sharding deferral in the paragraph below is now **closed** — see the note there.

**Status.** **Implemented** on branch `locg-chebyshev` as `ground_locg(prefilter=(degree, cycles))`,
default `None`. 12 tests in `tests/test_ground_locg.py::TestChebyshevPrefilter` plus
`tests/_sharded_prefilter.py`; suite 549 → 560. In-tree measurement reproduces the prototype: 18/18
configurations faster, median **1.36×**, range 1.11–3.07×, no regressions — at `(16, 4)`, which the
`(degree, cycles)` sweep in §3.1 has since superseded: **use `(32, 2)`**, measured 1.88× median over 27
configurations at fewer matvecs. Sharding verified on a
4-device mesh (202 → 51 iterations, spec `P('x',)` preserved, energies agreeing to 1e-13).

Sharding coverage is 1/2/4 devices x partitioned/replicated (12 cases), asserting the output *spec*.
Ragged mesh splits are **not** swept because they are unreachable: explicit sharding rejects
`dim % mesh.size != 0` at `device_put`, before `ground_locg` runs. That is `sqd`'s concern, where
`uniquify_states` pads to a power of two — **now covered** by
`tests/_sharded_sqd_prefilter.py` / `test_sqd.py::TestShardedSqdPrefilter` (2026-08-28): the padded
subspace and `apply_h`'s gather-heavy matvec on a mesh, swept over 1/2/4 devices x all six
`cache_level`s, asserting the prefilter's output *spec* as well as the energy.

**Still not the default**, and §5 item 4 stands: every figure here is single-device CPU.
`examples/scaling/poc9_prefilter_gpu.py` exists to settle the GPU question in one run on a CUDA box --
it sweeps degree x cycles, reports through `fmt_ratio` (which refuses to call a difference inside the
measured spread a win), and asserts the sharding spec on real devices. Its docstring states why the
*direction* is genuinely uncertain rather than merely unmeasured: the prefilter trades ~79 matvecs for
roughly half the iterations, and whether that pays depends on the matvec-to-bookkeeping cost ratio,
which measured ~25% on CPU and is exactly what differs between backends. `apply_h`'s matvec is a
gather-heavy irregular kernel while the LOBPCG bookkeeping is bandwidth-bound streaming, so the ratio
can move either way.

The prefilter is GPU-*safe* as written, which is checkable without a GPU: its jaxpr contains three
`scan`s and **zero** callbacks (`pure_callback`, `io_callback`) or `device_put`s, so there is no
per-cycle host synchronisation. An earlier prototype computed the Rayleigh quotient with a Python
`float(...)` each cycle, which would have stalled the device once per cycle; the shipped version is
fully traced.

---

## 1. What it is

Apply a low-degree Chebyshev polynomial filter to the initial vector before handing it to
`ground_locg`, then let `ground_locg` run unchanged to machine precision.

```
x  = v0
repeat 4 times:
    θ  = ⟨x|Ax⟩ / ⟨x|x⟩                    # 1 matvec
    x  = T_deg( (A - c)/e ) x  / ‖·‖        # deg matvecs, c=(hi+θ)/2, e=(hi-θ)/2
ground_locg(A, x)                            # unchanged
```

`hi` is an upper bound on `λ_max` from 10 steps of randomized power iteration (×1.05 margin).
`degree = 32`, 2 cycles → **77 matvecs** of prefilter (`cycles · (degree + 1)` plus ~11 for the
`lambda_max` estimate). See §3.1 for why this beats the originally-shipped `(16, 4)` and how the two
knobs differ.

**No prior spectral knowledge is required.** The filter's lower edge is the *current Rayleigh
quotient*, updated each cycle — this is what production ChFSI does. Using the exact `λ₁` instead is
both unavailable in practice and, measured, worse (see §4).

## 2. Why it fits this codebase where the other candidates did not

`docs/skqd-sqd-solve-tolerance.md` §8 records six rejected alternatives. The constraints they each
violated:

| constraint | why it binds | Chebyshev prefilter |
| --- | --- | --- |
| sharding-transparent | `ground_locg`'s documented contract; ARPACK cannot shard | **matvec + axpy only** — inherits `apply_h`'s `out_sharding` |
| fixed memory | block Lanczos needed ~420 MB at dim=200k | **3 vectors**, independent of iteration count |
| must attack the *gap* | per-iteration work is already near-minimal and load-bearing | a degree-`p` filter damps `[θ, λ_max]` by `1/T_p`, so it **changes the effective gap** |
| must not deplete the residual | a power-iteration start measured 177 iters vs 77 | **measured: it does not** — see §3 |

That last row was the real risk. A previously-measured result in this repo is that a *better* start
makes `ground_locg` *worse*: a shifted-power start with a far better Rayleigh quotient (−9.36 vs 0.02)
and smaller residual needed 2.3× **more** iterations, because block-size-1 LOBPCG spans only
`{x, r, p}` and convergence tracks what the residual can still expose. A Chebyshev-filtered start
does **not** have that pathology — it suppresses the unwanted band multiplicatively rather than
collapsing onto the dominant direction, so the residual stays rich in the directions that matter.
This is the one non-obvious fact that makes the approach work, and it is why the prefilter must be a
*filter* and not extra power iterations.

## 3. Measurement

`degree = 16`, 4 prefilter cycles (the original setting; §3.1 sweeps the surface and recommends
`(32, 2)`), `hi` from randomized power iteration. XXZ chains grown to connected
subspaces by one-hop expansion (**not** `rng.choice` — see the tolerance doc §4.4 for why that regime
is wrong). 3 seeds × 3 anisotropies × 2 sizes:

| n | seed | Jz | `ground_locg` | its iters | hybrid | its iters | speedup |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 18 | 0 | 0.5 | 55.0 ms | 128 | 29.2 ms | 47 | **1.88×** |
| 18 | 0 | 1.0 | 59.7 ms | 140 | 42.5 ms | 77 | 1.41× |
| 18 | 0 | 1.5 | 69.7 ms | 162 | 46.7 ms | 86 | 1.49× |
| 18 | 1 | 0.5 | 46.6 ms | 106 | 35.8 ms | 62 | 1.30× |
| 18 | 1 | 1.0 | 47.3 ms | 112 | 36.0 ms | 63 | 1.31× |
| 18 | 1 | 1.5 | 55.5 ms | 129 | 48.6 ms | 92 | 1.14× |
| 18 | 2 | 0.5 | 33.3 ms | 76 | 24.8 ms | 38 | 1.35× |
| 18 | 2 | 1.0 | 34.2 ms | 79 | 27.7 ms | 45 | 1.23× |
| 18 | 2 | 1.5 | 41.7 ms | 94 | 33.5 ms | 56 | 1.25× |
| 20 | 0 | 0.5 | 74.7 ms | 88 | 41.5 ms | 38 | 1.80× |
| 20 | 0 | 1.0 | 74.7 ms | 94 | 67.1 ms | 61 | 1.11× |
| 20 | 0 | 1.5 | 98.1 ms | 116 | 61.1 ms | 54 | 1.61× |
| 20 | 1 | 0.5 | 201.0 ms | 216 | 59.8 ms | 59 | **3.36×** |
| 20 | 1 | 1.0 | 246.8 ms | 249 | 63.7 ms | 65 | **3.87×** |
| 20 | 1 | 1.5 | 156.1 ms | 186 | 115.8 ms | 110 | 1.35× |
| 20 | 2 | 0.5 | 69.6 ms | 88 | 43.4 ms | 40 | 1.61× |
| 20 | 2 | 1.0 | 68.2 ms | 89 | 53.0 ms | 49 | 1.29× |
| 20 | 2 | 1.5 | 91.5 ms | 119 | 46.5 ms | 47 | 1.97× |

**min 1.11×, median 1.38×, max 3.87×, and 0 of 18 cases lose.**

### 3.1 Tuning `(degree, cycles)` — `(16, 4)` is not the best setting

`(16, 4)` was chosen early, before the surface was swept, and it is suboptimal. Across **27**
configurations (3 sizes n=18/20/22 × 3 seeds × 3 anisotropies; every arm converged, every energy correct
to <1e-9):

| setting | median | min | max | losses | filter matvecs |
| --- | --- | --- | --- | --- | --- |
| `(16, 4)` — originally shipped | 1.41× | 1.08× | 3.25× | 0 | 79 |
| **`(32, 2)` — recommended** | **1.88×** | **1.25×** | 3.95× | 0 | **77** |
| `(48, 2)` | 1.79× | 1.14× | 4.17× | 0 | 109 |

`(32, 2)` dominates `(16, 4)` on every axis: higher median, higher floor, and one fewer matvec. Paired
per-configuration, it is 1.27× median faster and slower in only **2 of 27** cases (worst 0.91×).

**The two knobs are not interchangeable.**

- `degree` sets how sharply **one** cycle separates. Amplification outside the damped band grows like
  `cosh(degree · arccosh|x|)` — roughly exponential — while costing only `degree` matvecs. High leverage.
- `cycles` sets how many times the interval re-tightens around the descending Rayleigh quotient. Cycle 1
  does most of the work (growth factor 1e8–1e12, §5a of `docs/locg-next-candidates.md`), cycle 2 refines
  once, and past that `θ` is already near `λ₀` so further cycles pay full cost for little separation.

So **raise `degree`, keep `cycles = 2`** — which inverts the intuition behind the original `(16, 4)`.

A narrower sweep (6 configurations) mapped the saturation point: `(64, 2)` reached 2.29× median,
`(96, 2)` 1.95×, and `(128, 2)` fell to 1.68× with a 1.01× floor. So the useful range is
**`degree` 32–64**, declining past ~96. Longer solves favour the higher end — the 249- and
573-iteration cases measured 3.6–5.1× at `degree` 48–96.

Practical recipe:

1. Start at **`(32, 2)`**.
2. If `ground_locg` runs 200+ iterations unfiltered, try `(48, 2)` or `(64, 2)`.
3. Do not exceed `degree ≈ 64` or `cycles ≈ 4`.
4. On the first run of a new problem class, assert the energy against a reference and check `converged`.
   The filter cannot change the answer — every convergence test reads the true residual — but that is
   the assertion worth making once.

**Caveat.** All of this is single-device CPU. The ordering could shift on a GPU, where the
matvec-to-bookkeeping cost ratio differs; `examples/scaling/poc9_prefilter_gpu.py` sweeps this grid for
exactly that reason.

Correctness, every case: energy agrees with `scipy.sparse.linalg.eigsh(tol=0)` to **1.8e-15–2.8e-14**,
eigenvector overlap with the unfiltered `ground_locg` result is **1.0000000**, and `converged` is
`True`. So this is not a speed-for-accuracy trade — it is the same eigenpair, sooner.

The prefilter roughly **halves** the LOBPCG iteration count (128→47, 249→65), and the biggest wins land
where `ground_locg` iterated most — which is consistent with the gap dependence documented in the
tolerance doc: the filter is doing exactly what a wider gap would do.

Larger sizes, single configuration (`Jz=0.8`, seed 0), for scale: n=22 1.50×, n=24 1.68×.

## 4. Two things that do *not* work, measured

- **Filtering with the exact `λ₁` as the lower edge.** Tempting, since it is the "correct" filter
  interval. With `λ₁` from ARPACK the filter is dramatically effective where the gap is comfortable
  (n=18: 8.1× at 2.6e-12; n=20: 6.9× at error exactly 0.0) but **fails outright** at n=22, where
  relgap is 4.0e-05: error 1.5e+01 at every degree and cycle count tried. With `λ₁ − λ₀` = 1.2e-03 the
  interval `[λ₁, λ_max]` begins essentially *at* `λ₀`, so the filter damps the ground state along with
  the rest. The self-consistent Rayleigh-quotient edge starts *above* `λ₀` and descends, so it never
  brackets the target out — and it needs no spectral input. Use it.
- **Chebyshev filtering alone, to convergence.** Accuracy plateaus at ~1e-5 to 1e-7 and does not reach
  machine precision even at 816 matvecs (only one configuration reached 8.0e-12, at 1.32×). As `θ → λ₀`
  the filter's lower edge approaches `λ₀` and it begins damping the ground state, so the method
  self-limits exactly where the last digits are wanted. This is a known ChFSI property — it is used in
  DFT as an outer solver at ~1e-6, not as a machine-precision eigensolver. **The hybrid is the point**:
  filter cheaply to get close, then let `ground_locg`'s guards deliver the last digits.

## 5. Implementation as shipped on `locg-chebyshev`

Two module-level helpers plus one call site, all in `rqutils/ground_locg.py`:
`_lambda_max_bound` (10 power steps, ×1.05 multiplicative margin) and `_chebyshev_prefilter`
(the `lax.scan` recurrence). The call site sits **after** the work-dtype promotion, so the recurrence
runs at the operator's precision, and **before** `body_iter0`, so `rho_init` reflects the filtered
vector and `maxiter=0` still returns something meaningful. `prefilter` is a static argument on both
`_ground_locg_matrix` and `_ground_locg_callable`, so `None` leaves the traced graph byte-identical
(pinned by `test_prefilter_is_static_so_it_adds_no_traced_argument`).

The original plan, retained for the reasoning:

1. `ground_locg(..., prefilter: tuple[int, int] | None = None)` — `(degree, cycles)`, `None` keeping
   today's behaviour exactly so no existing caller changes. Follow the shape of the existing
   `precond: Callable | None = None` argument, which is already a static argument resolved at trace
   time.
2. The `λ_max` estimate needs its own small helper (10 power steps, ×1.05). It must use the same
   `out_sharding` discipline as every `apply_*` in `rqutils/sqd.py`, or the prefilter breaks the
   sharding-transparency contract that `ground_locg`'s docstring states for `mat` and `precond`.
3. The filter is a `lax.scan` of `2(A−c)/e · T_k − T_{k−1}` — three vectors live at once. It must be
   inside the same jit as the solve, or the per-call trace overhead eats the win at small `N`.
4. **Do not** make it the default until it is measured on GPU and on a mesh. All figures here are
   single-device CPU, and `CLAUDE.md` warns that timings under virtual devices are meaningless.
5. `sqd()` would need to thread the option through `run_sqd`; note `cache_level` must stay static there
   because `ground_locg` splats `args` positionally (`CLAUDE.md`), so a new static argument needs the
   same care.

## 6. Verification

1. **Same eigenpair, not merely the same energy.** Assert eigenvector overlap with the unfiltered
   result is 1.0 to ~1e-12, not just that energies agree — a filter that converged to a different
   member of a near-degenerate pair would pass an energy check. §3 reports 1.0000000 for all 18 cases.
2. **Independent oracle.** Compare against `scipy.sparse.linalg.eigsh(k=1, which="SA", tol=0)`, not
   against `ground_locg` alone. Per `CLAUDE.md`, use `eigsh`/`eigvalsh`, never `eigh`.
3. **Connected subspaces only.** Build fixtures by one-hop expansion and assert their density; a
   `rng.choice` subspace is 3.6–6.1% dense against 32–44% for a real one, and iteration counts differ
   by 3–5× between the regimes.
4. **The small-gap case is the one that breaks filters.** Include a configuration with relgap ≲ 1e-4
   (n=22 at `Jz=0.8`, dim=30000 gives 4.0e-05) and assert the energy is still correct — that is the
   case where the exact-`λ₁` variant returned an answer off by 15 with no error raised.
5. **Assert it never loses.** The value of this change is that it is monotone; a regression on any
   configuration is the finding, not noise. Sweep seeds and anisotropies, not one instance.
6. **Sharding.** Run under `--xla_force_host_platform_device_count=4` in a subprocess and assert the
   output *sharding spec*, not only the values — per `CLAUDE.md`, a replicated run agrees to exactly
   0.0, so "correct but silently unsharded" is invisible to value comparison.

## 7. Provenance

Chebyshev-filtered subspace iteration (ChFSI) as an alternative to LOBPCG:
Banerjee, Lin, Suryanarayana, Yang, Pask, *Two-Level Chebyshev Filter Based Complementary Subspace
Method*, J. Chem. Theory Comput. 14 (2018); used in production in DFT-FE. The single-vector
prefilter-then-LOBPCG hybrid measured here is the adaptation to this codebase's constraints, not a
construction from those papers.
