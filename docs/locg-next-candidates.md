# Faster `ground_locg`: every candidate after the Chebyshev prefilter, all rejected

**Status: search closed. Nothing implemented, and nothing left to implement.** Written 2026-08-28 on
branch `locg-chebyshev`, after `docs/locg-chebyshev-prefilter.md` shipped. Records what a literature
search turned up (little), what the session's measurements *constrain* (a lot), and the four candidates
that were then built and measured — all of which lost to the shipped prefilter.

Read §1 first: the four constraints are what make this space so narrow, and they explain every rejection
below.

**Two candidates have since been tested and rejected**, for opposite reasons:

- **§3, filtered residual via `precond`** — 4–20× slower *and* wrong energies. The reasoning was sound
  and the prediction wrong, so §3 keeps both and names the mechanism: filtering the residual destroys
  the search direction's alignment with the gradient. That is the depleted-residual failure this repo
  has now hit three times in different guises.
- **§3b, Davidson** — *correct* (1e-11 or better) and 1.05–2.62× over baseline, but the shipped
  Chebyshev prefilter beats it in 3 of 4 cases while needing no basis and no tuning knob. Rejected on
  cost, not correctness.

**§5's two "cheap, safe" wins have also been tested and rejected** — adaptive scheduling made things
*worse* (median 1.31× → 1.11×, 6 losses where the fixed schedule had none), and bounds continuation is
worth ~5% for real plumbing. §5a records the one reusable finding: the filter's growth factor is a clean,
free stopping signal, and over-filtering is genuinely unstable (growth to 1e+23, a Rayleigh quotient
excursion to +4.60) — just not in a way that matters, since `ground_locg` repairs it.

**§4, Jackson damping, has also been tested and rejected** — 0.96× median, slower than no filter at all
in half the configurations, because damping trades away four to five orders of magnitude of the
amplification that is the prefilter's entire mechanism. Its premise was also wrong: the prefilter applies
a single `T_degree` and has no expansion coefficients to damp.

**Every candidate in this document has now been measured and rejected.** The prefilter's 1.11–3.07× is
banked and the search space around it is mapped empty. The recommendation is to stop: further work on
`ground_locg` should wait for the GPU numbers (`examples/scaling/poc9_prefilter_gpu.py`), which could
change the cost balance that decides several of these results.

All the rejections carry one methodological warning, hit three times: **timing JAX code outside the jit
boundary it normally lives inside measures the boundary, not the code.** A host-side Davidson read
0.39–0.86× and became 1.05–2.62× once jitted; `_lambda_max_bound` timed standalone read 103% of a solve
against a true ~5%. Warming does not fix it.

---

## 1. The constraints, which are what actually narrow the field

Every candidate must satisfy all four. This is why most of the standard toolkit is already ruled out —
see `docs/skqd-sqd-solve-tolerance.md` §8 for the seven measured rejections.

1. **Sharding-transparent** — matvec plus elementwise only. `ground_locg`'s docstring states this
   contract for `mat` and `precond`. It eliminates ARPACK (measured 2.0–2.7× but host-side and
   unshardable), and anything needing a QR or `eigh` over a tall basis.
2. **Fixed memory** — no growing basis. Block Lanczos needed ~420 MB at dim=200k, against the
   ~88 MB of cached `xsources`/`diagonals`. The single-vector design exists for this reason.
3. **Must attack the spectral gap** — per-iteration work is ~75% `O(N)` safeguards (≈94
   dot-equivalents), already inside one `@jax.jit` with a `lax.while_loop`, and `docs/locg.md`
   catalogues seven defects (I1–I7) from removing them, each failing *silently*. Iteration **count** is
   the only lever.
4. **Must not deplete the residual** — the non-obvious one, and the constraint that killed the most
   plausible-looking ideas. Block-size-1 LOBPCG spans only `{x, y, p}`, so convergence tracks what the
   residual can still *expose*, not proximity to `v_0`. Measured: a shifted-power start with a far
   better Rayleigh quotient (−9.36 against 0.02, true −14.36) and a *smaller* residual needed **177**
   iterations against 77.

The Chebyshev prefilter passed all four, which is why it is the only thing that worked.

---

## 2. Literature search: no new family

Six searches (randomized eigensolvers 2024–2025, polynomial-filtered LOBPCG, deflation-accelerated
LOBPCG, s-step / communication-avoiding LOBPCG, sketched Rayleigh–Ritz, Jackson-damped KPM filters).
Nothing surfaced that both beats the tested set and satisfies §1. Two families are directly relevant
and both are adjacent to work already done:

- **Polynomial-preconditioned eigensolvers** (Embree/Loe/Morgan line; also the ChFSI literature
  already cited in `docs/locg-chebyshev-prefilter.md` §7). This is the published formalization of
  candidate A below, and those papers treat reusing the filter beyond the initial vector as the
  standard move.
- **s-step / communication-avoiding Krylov** (Hoemmen, Carson lineage). Genuinely aimed at the
  latency-bound regime that matters on GPU, but every published variant either grows a basis or
  replaces the orthogonalization with a block QR — both fail constraint 2. Recorded rather than
  pursued.

Recorded as a negative search result: the remaining ideas are refinements of the filter that already
works, not a new algorithm.

---

## 3. Candidate A: Chebyshev-filtered residual via `precond` — **TESTED AND REJECTED**

Apply a low-degree polynomial filter to the residual *each iteration* — use `p(A) r` as the search
direction instead of `r` — rather than filtering only the initial vector.

**Measured 2026-08-28: 0.05–0.23×, i.e. 4–20× SLOWER, and wrong.** Six connected XXZ configurations,
degrees 2/3/4, `precond=lambda r: p(A) r` with the interval's lower edge just above the true `λ₀`
(itself generous — an implementation could not know it):

| n | plain | prefilter (16,4) | `precond` deg 2 | deg 3 | deg 4 |
| --- | --- | --- | --- | --- | --- |
| 18, seed 0 | 63.9 ms / 140 it | 46.1 ms / 77 it | 0.09× / 1000 it | 0.08× / 1000 it | 0.07× / 1000 it |
| 20, seed 1 | 247.4 ms / 249 it | 104.5 ms / 67 it | 0.23× / 581 it | 0.12× / 1000 it | 0.09× / 1000 it |

Nearly every arm hit `maxiter=1000`, and the energies are off by **0.17 to 10.0** — not a slow correct
answer but a wrong one.

**Why the structural argument was insufficient.** The reasoning below — that `p(A)` commutes with `A`
and so cannot reproduce Jacobi's mixed-sign pathology — is *correct*, and it is still not enough.
Commuting with `A` preserves the eigenbasis; it says nothing about preserving **alignment with the
gradient**, which is what makes LOBPCG a descent method. Measured on a mid-convergence iterate,
`|⟨r|p(A)r⟩| / (‖r‖‖p(A)r‖)`:

| degree | alignment with `r` | alignment with `x` |
| --- | --- | --- |
| 1 | 0.0106 | 0.0333 |
| 2 | **0.0010** | 0.0749 |
| 3 | 0.1542 | 0.1023 |
| 4 | 0.2245 | 0.1375 |

`p(A)r` is nearly **orthogonal to `r`**. The reason is structural: `r` is orthogonal to `x` by
construction, so it is spread across the remaining spectrum and has almost no component in the
direction the filter amplifies. What survives filtering is therefore close to noise relative to the
true gradient, and a direction with 0.001 alignment is not a preconditioned gradient at all.

**This is the depleted-residual failure again**, in a new place. `docs/skqd-sqd-solve-tolerance.md` §8
records that a shifted-power *start* measured 177 iterations against 77 despite a far better Rayleigh
quotient, because block-size-1 LOBPCG spans only `{x, y, p}` and convergence tracks what the residual
can still expose. Filtering the residual attacks that same quantity directly. **The shipped prefilter
works precisely because it acts on the iterate before the iteration begins, where there is no gradient
to corrupt** — that distinction is the transferable lesson, and it was not obvious in advance.

The original argument is preserved below, since it is a good argument that happens to be wrong, and the
failure mode is worth recognizing rather than re-deriving:

> - **Zero new plumbing.** `ground_locg(precond=...)` already takes a callable on the residual, is a
>   static argument resolved at trace time, and carries the same sharding-preservation contract
>   `_chebyshev_prefilter` already satisfies.
> - **It is not the Jacobi failure repeated.** Jacobi measured **0.29–0.35×** (iterations 77 → 258)
>   because `diag(H)^-1` has mixed signs on an indefinite operator and destroys the descent direction.
>   A polynomial in `A` commutes with `A` and preserves the eigenbasis, so that specific pathology is
>   structurally absent.
> - **It needs only an upper bound.** `precond` measured 1.79× median on a *shifted* operator, and the
>   shift is unobtainable. A polynomial filter needs only `lambda_max`, which `_lambda_max_bound`
>   supplies in ~11 matvecs.
> - **The known risk:** degree `d` multiplies matvecs per iteration by `d`, so `d` must be 2–4 and must
>   remove more than roughly `d/4` of the iterations to break even.

The cost objection turned out to be irrelevant — the method fails on correctness before cost matters.

---

## 3b. Davidson — **TESTED AND REJECTED**, but on cost grounds, not correctness

Restarted Davidson: expand the subspace with the correction `t = -(diag(A) - θI)^{-1} r`, Rayleigh–Ritz
over the accumulated basis, restart from the Ritz vector at a fixed basis size.

**It works, which distinguishes it from every other preconditioner tried here.** Energies correct to
1e-11 or better in all cases. The reason is the shift: Davidson uses `(diag(A) - θI)^{-1}` with `θ` the
*current* eigenvalue estimate, which is free, where the rejected Jacobi arm used `diag(A)^{-1}`, whose
mixed signs on an indefinite operator destroy descent (0.29–0.35×). `docs/rqutils-precond-request.md`
closed the *lower-bound* shift; this one needs no bound at all.

**But it does not beat the shipped prefilter.** Best setting `m=16`, 6 sweeps (connected XXZ):

| n | seed | `ground_locg` | prefilter (16,4) | Davidson m=16 |
| --- | --- | --- | --- | --- |
| 18 | 0 | 63.2 ms / 140 mv | **45.9 ms** / 77 mv | 42.5 ms / 96 mv — 1.49× |
| 18 | 1 | 52.6 ms / 112 mv | **38.7 ms** / 63 mv | 42.2 ms / 96 mv — 1.25× |
| 20 | 0 | 79.0 ms / 94 mv | **53.3 ms** / 49 mv | 75.1 ms / 96 mv — 1.05× |
| 20 | 1 | 201.6 ms / 249 mv | 70.9 ms / 67 mv | **76.9 ms** / 96 mv — 2.62× |

Davidson is 1.05–2.62× over baseline, and the prefilter beats it in **3 of 4** cases — while already
being implemented, tested, and sharding-verified.

Three reasons not to pursue it:

- **The basis size is an unprincipled tuning knob.** `m=16` wins; `m=24` and `m=32` measured
  0.46–1.54×, frequently *slower than baseline*. Choosing `m` requires knowing the spectrum, which is
  what is being computed.
- **It abandons the single-vector design on purpose.** `CLAUDE.md` states `ground_locg` is a
  block-size-1 specialization with analytic `eigenpair_2x2`/`eigenpair_3x3` "to keep memory down for
  huge vectors." Davidson holds `V` and `AV`: `2·m·N·8` bytes, so 67 MB at `m=16`, dim=200k, on top of
  the ~88 MB of cached `xsources`/`diagonals`. Not fatal, but it is exactly the trade that design
  refused.
- **The gain is not additive with the prefilter.** Both attack the same quantity — how good the iterate
  is when Rayleigh–Ritz runs — so they compete rather than compose. The prefilter is the cheaper of the
  two and needs no basis.

**Two corrections to my own analysis, recorded because both were nearly written up as findings.**

*A host-side implementation measured 0.39–0.86× and almost became the verdict.* Every matvec crossed
the JAX/NumPy boundary at a measured **2.0×** overhead (0.557 ms device-resident vs 1.111 ms via
`np.asarray`), and the Rayleigh–Ritz and orthogonalization ran as un-jitted NumPy on `N`-vectors.
Jitting it moved the result to 1.05–2.62×. This is the same artifact that made the ARPACK comparison in
`docs/skqd-sqd-solve-tolerance.md` §8 look mixed: **a host-side prototype of a device-resident algorithm
measures the boundary, not the algorithm.** Re-test only with a jitted, fixed-shape version.

*A claim that Davidson "should fare relatively worse on GPU" was wrong and is withdrawn.* It rested on
Davidson using more matvecs than the prefilter. Measured, it uses **fewer** — 96 against the
prefilter's 146 (67 iterations + 79 filter matvecs) — with the same implied `O(N)` work per matvec:

| method | matvecs | wall | ms/matvec | implied `O(N)` ops per matvec |
| --- | --- | --- | --- | --- |
| `ground_locg` | 249 | 201.7 ms | 0.810 | ~31 dot-equivalents |
| prefilter (16,4) | 146 | 81.3 ms | 0.557 | ~18 |
| Davidson m=16 | 96 | 76.9 ms | 0.801 | ~31 |

The prefilter's advantage is that it is `O(N)`-*light* per matvec, not matvec-light. Whether that
survives a GPU, where the matvec is a gather-heavy irregular kernel and the `O(N)` work is
bandwidth-bound streaming, **is not derivable from these numbers** — the same caveat
`examples/scaling/poc9_prefilter_gpu.py` carries.

---

## 4. Candidate B: Jackson damping — **TESTED AND REJECTED**

The original claim: a per-coefficient multiplier on the Chebyshev expansion, standard in the kernel
polynomial method, suppressing the Gibbs oscillation at the interval edge — "a few lines inside the
existing recurrence."

**First correction: the premise was wrong. The prefilter has no coefficients to damp.** It applies a
*single* `T_degree`, taking only the last term of the three-term recurrence. Jackson damping is a
multiplier `g_k` on an *expansion* `Σ_k c_k T_k`; with no expansion there is nothing to weight. Applying
it properly means restructuring into a damped step-function expansion — a different filter, not a few
lines.

Both the true expansion and a cheaper edge-shift variant were built and measured. **The Gibbs
oscillation is real** — `|T_16|` inside the band reaches 1.0 at 16 points, so an eigenvalue just inside
the lower edge is damped only to ~1 — but suppressing it makes things *worse*:

| variant | median | min |
| --- | --- | --- |
| **shipped, single `T_16`** | **1.34×** | **1.29×** |
| edge shift `β = 0.05` | 1.08× | 0.93× |
| edge shift `β = 0.20` | 1.15× | 1.02× |
| **Jackson-damped expansion** | **0.96×** | 0.89× |

Jackson damping is slower than *no filter at all* in half the configurations. Energies stayed correct
(≤5.3e-15), so this is purely a cost result.

**Why, quantified.** Damping smooths the transition, and the amplification outside the band is what the
prefilter exists for. Comparing the two filters at positions where the ground state sits (`x < -1` is
outside the band):

| `x` | single `T_16` | Jackson expansion | ratio |
| --- | --- | --- | --- |
| −1.50 | 2.44e+06 | 2.41e+02 | 1.0e-04 |
| −1.20 | 1.06e+04 | 7.19e-01 | 6.8e-05 |
| −1.05 | 7.71e+01 | 2.34e-03 | 3.0e-05 |

Jackson gives up **four to five orders of magnitude of amplification** to buy a cleaner edge. That is
the right trade in the kernel polynomial method, whose purpose is estimating a smooth spectral density
where ringing is the error. It is the wrong trade here: **the ringing is harmless**, because components
inside the band are already suppressed relative to the ground state's 1e2–1e5 growth, and the growth is
the entire mechanism.

The transferable point: KPM's Gibbs problem and a prefilter's separation problem pull in opposite
directions. A technique being standard in one does not make it applicable to the other, and §4's original
framing — that this was worth trying because the edge is "where the filter is weakest" — inverted the
objective. The filter is *supposed* to be weak at the edge and strong beyond it.

Note the original section already said this only mattered "if someone wants filtering to converge on its
own," which the shipped hybrid does not. That caveat was the correct instinct; the measurement confirms
it and adds the mechanism.

---

## 5. Two cheap, safe, small wins — **BOTH TESTED AND REJECTED**

Implemented and measured 2026-08-28, then reverted. Neither pays, and the reasons are more useful than
the ideas were.

### 5a. Adaptive prefilter scheduling — rejected

The original claim: `(16, 4)` is fixed, the win correlates with how much `ground_locg` was going to
iterate anyway, so stopping early on a per-problem signal "converts a fixed 72-matvec cost into a
problem-adaptive one."

**A clean stopping signal does exist**, and it is worth recording because it is free. The filter's
**growth factor** `‖T_degree(...)v‖` — already computed, since `normalize` divides by it — behaves as
a sharp indicator across all 18 configurations:

| cycle | growth factor |
| --- | --- |
| 1 | 1e8 – 1e12 |
| 2 | 15 – 330 |
| 3+ | collapses toward 1.0 |

The mechanism: the recurrence amplifies whatever lies outside `[θ, hi]`, so while `θ` is far above `λ₀`
the ground state is well outside the band and grows enormously. Once `θ` has descended to `λ₀` the
ground state sits *at* the interval edge where `T_degree ≈ 1`, and the filter separates nothing further.

**Continuing past that point is genuinely unstable.** A marginally-stable recursion run on a vector with
no signal left amplifies its own rounding: measured growth factors up to **1e+23** and a Rayleigh
quotient excursion from −9.97 to **+4.60** at cycle 6 on one configuration. `(16, 4)` stops before this,
so the shipped default was safe by luck rather than by construction — which is worth knowing.

**But the change made things worse**, measured over 18 configurations:

| setting | min | median | max | losses |
| --- | --- | --- | --- | --- |
| **`(16, 4)`, shipped** | **1.06×** | **1.31×** | 3.10× | **0** |
| `(16, 8)` with adaptive latch | 0.88× | 1.11× | 2.26× | 6 |
| `(16, 16)` with adaptive latch | 0.67× | 0.83× | 2.00× | 14 |

Two errors in the original reasoning, both mine:

1. **The premise is impossible under `lax.scan`.** A scan has a *static* trip count, so a latch stops
   the vector updating but every matvec still executes. `(16, 16)` pays 16 cycles regardless. Only a
   `lax.while_loop` would actually save the work, and that adds a traced loop for a benefit that only
   materializes when `cycles` was set too high to begin with.
2. **The floor cases are not over-filtered.** They stop being useful after cycle 3–4 — which is what
   `(16, 4)` already does. Adaptive stopping can only recover waste that the default does not create.

**The safety benefit is also not real.** Pre-change code at `cycles` = 4, 8, 16, 24 across six
configurations returned energies correct to **3.6e-15 in every case**. `ground_locg` repairs whatever the
filter hands it, so over-filtering degrades the prefilter without ever producing a wrong answer — the
same conclusion the shipped suite's mutation testing reached (§7 item 4). So there is nothing to protect
against.

### 5b. Carrying spectral bounds across a solve sequence — rejected on size

`_lambda_max_bound` costs 11 matvecs. The prefilter costs 79, and the solve 49–249. So carrying
`lambda_max` across a monotone sequence saves at most **~5%** of a solve — and it would mean threading
spectral state through `sqd` → `run_sqd` → `ground_locg`, where `cache_level` must already stay static
because `ground_locg` splats `args` positionally (`CLAUDE.md`). The measured drift across a growing
sequence is small (2.4% on the first step, then 0.3–0.5%), so the *idea* is sound; it is the payoff that
is not worth the plumbing.

### A measurement trap, hit three times

A first attempt reported `_lambda_max_bound` at **103% of a solve**, which is impossible. It is not
jitted standalone — only when called from inside `_ground_locg_callable` — so timing it directly measures
**re-tracing**: 48.23 ms traced against 19.23 ms jitted, where 11 matvecs is 18.62 ms. Jitted, it costs
exactly what it should.

This is the third instance of the same class of error in this investigation, after the host-side Davidson
(0.39–0.86× → 1.05–2.62× once jitted) and the ARPACK comparison. **Timing a JAX helper outside the jit
boundary it normally lives inside measures the boundary, not the code.** Warming does not fix it; the
call must be jitted the way the caller jits it.

---

## 6. Deprioritized, with reasons

- **Growing-basis methods** — Davidson, Jacobi–Davidson, thick-restart Lanczos. Fail constraint 2;
  thick-restart still needs `eigh` on the restart basis.
- **Randomized SVD-style sketching** — built for low-rank approximation. The projected `H` is not
  low-rank.
- **Shift-invert** — needs linear solves with `H - sigma I`, indefinite and unpreconditionable here.
- **Stochastic / SGD Rayleigh-quotient minimization** — for problems where the matvec is itself a
  sampled expectation. `apply_h` is exact and cheap; sampling it adds variance for nothing.
- **s-step LOBPCG** — see §2. The only idea here that might behave qualitatively differently on GPU,
  but no variant satisfies constraint 2.

---

## 7. Verification any candidate must pass

The failure mode in this module is consistently *plausible-but-wrong*, so an energy check is not
sufficient.

1. **Eigenvector overlap**, not just energy. A spectral transformation can converge to a neighbouring
   eigenvector while the energy still looks right. `TestChebyshevPrefilter` asserts overlap > 1 − 1e-9.
2. **A tiny-gap fixture** (relgap ≲ 1e-4). The exact-`lambda_1` filter variant returned an energy off
   by **15** with no error raised, and only that regime exposes it.
3. **Keep `lambda_0` away from zero in fixtures.** The criterion is
   `|r| < tol * (|Ax| + |theta|) * N * 10`, so a spectrum starting at 0.0 makes it unsatisfiable — an
   *unfiltered* run then reports `converged=False` while returning the right energy, which looks like a
   defect in the thing under test and is not one. Cost me a debugging cycle; see
   `TestChebyshevPrefilter._gapped`.
4. **The suite will not catch filter arithmetic.** Mutation testing on the shipped prefilter: a
   discarded filter is caught, but a flipped three-term recurrence sign and a 20%-raised interval edge
   both survive, because `ground_locg` repairs the start (100.00% vs 99.99% of the distance closed). A
   change to the recurrence needs the wall-clock XXZ batch, not pytest.
5. **Sharding on the spec**, under `--xla_force_host_platform_device_count`, sweeping partitioned *and*
   replicated. A replicated run agrees with single-device to exactly 0.0, so a silently unsharded result
   is invisible to value comparison.
