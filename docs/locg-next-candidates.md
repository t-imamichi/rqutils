# Faster `ground_locg`: remaining candidates after the Chebyshev prefilter

**Status: investigation notes, nothing implemented.** Written 2026-08-28 on branch `locg-chebyshev`,
after `docs/locg-chebyshev-prefilter.md` shipped. Records what a literature search turned up (little),
what the session's measurements *constrain* (a lot), and the candidates worth trying next.

**Candidate A has since been tested and rejected** (§3) — 4–20× slower *and* returning wrong energies.
Its reasoning was sound and its prediction wrong, so §3 keeps both and names the mechanism: filtering
the residual destroys the search direction's alignment with the gradient, which is the
depleted-residual failure this repo has now hit three times in different guises. Candidate B (§4) is the
leading untested option.

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

## 4. Candidate B (now the leading untested one): Jackson damping on the prefilter's coefficients

A per-coefficient multiplier on the Chebyshev expansion, standard in the kernel polynomial method,
suppressing the Gibbs oscillation at the interval edge.

Why it is worth a try: the current filter's lower edge is the running Rayleigh quotient, which creeps
toward `lambda_0` as it converges — so the edge is exactly where the filter is weakest, and Gibbs
ringing there is the plausible mechanism behind the measured accuracy plateau at 1e-5 to 1e-7
(`docs/locg-chebyshev-prefilter.md` §4). It is a few lines inside the existing recurrence.

Note this only matters if someone wants filtering to converge *on its own*. For the shipped hybrid the
plateau is irrelevant, since `ground_locg` delivers the last digits. So this is lower value than
candidate A unless the prefilter is being pushed toward a standalone solver.

---

## 5. Two cheap, safe, small wins

- **Carry the spectral bounds across a solve sequence.** The recovery loop solves a monotone sequence.
  Eigenvector continuation failed (constraint 4 — a zero-padded eigenvector is the extreme
  depleted-residual case, measured 79 → 129 iterations and one size converging to a *different*
  eigenvalue). But continuation of `lambda_max` and the previous `theta` moves no vector, so it cannot
  reproduce that failure: it saves the ~11 power-iteration matvecs per solve and starts the filter
  tighter. Worth a few percent, and safe.
- **Adaptive prefilter scheduling.** `(16, 4)` is fixed today, but the measured win correlates with how
  much `ground_locg` was going to iterate anyway (biggest gains where it ran 216–249 iterations,
  smallest at 1.11× where it ran ~129). Running 2 cycles, checking the Rayleigh-quotient improvement,
  and deciding whether to continue converts a fixed 72-matvec cost into a problem-adaptive one and
  should lift the floor cases.

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
