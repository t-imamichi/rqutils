# Response: the prefilter bound is fixed, and how to adapt

Reply to `docs/rqutils-prefilter-bug.md`, from the `rqutils` side. Branch `dev`, version still
`0.2.0` (unreleased). Fix in `568b173`; the report was recorded verbatim first, in `0873a4c`, so the
premise was reproduced against the unfixed tree before anything changed.

**Thank you — this was a real silent-wrong-answer bug and the diagnosis was exactly right.** Every
number in the report reproduced: n=2 Heisenberg returning `+0.25` for `-0.75`, the bound invalid in
16/25 configurations, wrong answers in 2/25, the coefficient-sum bound valid 25/25. Nothing in it
needed correction.

Two things you should read before re-enabling the wiring. The callable path is a **breaking change**
(§2) — though it does not affect your `sqd()` call sites. And we found a **second, unrelated defect in
`sqd`** while validating against your `skqd` regime, which affected every `sqd` call and not just
prefiltered ones; that is **also fixed** (§5). **Nothing is left blocked: both `exact.py` and
`skqd/sqd_backend.py` are safe to re-enable.**

---

## 1. What changed

`_lambda_max_bound` is **deleted**, not repaired. The bound now comes from the operator's structure:

| caller | where `hi` comes from | you change |
| --- | --- | --- |
| `sqd(..., prefilter=...)` | `sum|c_k|`, computed internally from the Hamiltonian | **nothing** |
| `ground_locg(array, ..., prefilter=...)` | Gershgorin `max_i sum_j |A_ij|`, derived from `mat` | **nothing** |
| `ground_locg(callable, ..., prefilter=...)` | the new `prefilter_hi=` argument | **must pass it** |

`_chebyshev_prefilter` takes `hi` as a required positional argument (it is private; noted only because
the report cites its line numbers).

Results: wrong answers **2/25 → 0/25**. Iteration reduction *improved* to a **3.29× median** from the
1.88× recorded with the unsound bound, because the estimate's ~11 matvecs per solve are gone as well.
Suite 581 → 585, including two regression tests built from your reproducer.

## 2. Migration

Your two `sqd()` call sites need **no change** — re-enable `prefilter=(32, 2)` as it was. `sqd` holds
the Pauli sum, so it computes `sum|c_k|` itself with zero matvecs, and the bound is rigorous because
every Pauli string is unitary (`||H|| <= sum|c_k| ||P_k|| = sum|c_k|`) and projecting onto a subspace
can only shrink the spectral radius (`P H P` with `P` an orthogonal projector).

If you call `ground_locg` directly with a matvec closure anywhere, that now raises:

```python
# before -- now raises ValueError
ground_locg(matvec, xinit, prefilter=(32, 2))

# after -- for a Pauli sum you already hold
ground_locg(matvec, xinit, prefilter=(32, 2),
            prefilter_hi=float(np.abs(ham.coeffs).sum()))    # SparsePauliOp

# after -- for a dense operator, pass the array as `mat` and the bound is automatic
ground_locg(A, xinit, prefilter=(32, 2))
```

All three forms above are tested. The raise is deliberate rather than a fallback to a repaired
estimate — see §3.

## 3. Why not your suggested fallback

The report offers `hi = max(power_iteration_estimate, -power_iteration_estimate)` as the minimum fix
if the estimate must stay iterative. **We measured it invalid in 4 of 25** of your own XXZ
configurations. Fixing the sign leaves a fixed 10-step iteration simply under-converged, which is the
same wall your rejected `A - sI` shift variant hit. Other candidates:

| candidate | your XXZ 25 | adversarial | matvec-only |
| --- | --- | --- | --- |
| `max(est, -est)` | 21/25 | — | yes |
| `sqrt` of power iteration on `A^2` | 24/25 | — | yes |
| Lanczos `mu_max + beta_k` | **25/25** | **988/1000** | yes |
| EVSL's `mu_max + |beta_k s_k|` | — | **3433/4000** | yes |
| Gershgorin / `sum|c_k|` | 25/25 | rigorous | no |

Lanczos passing 25/25 on the physics cases and failing adversarially is worth dwelling on: it is the
same "validated on the easy regime" trap that let the original bug ship, and it is why we did not adopt
the tighter Krylov bound despite it being tempting.

**The obstruction is a theorem, not a tuning problem.** Kuczynski & Wozniakowski (SIAM J. Matrix Anal.
Appl. 13(4):1094–1122, 1992) prove that with fewer than `N` matvecs there is always another operator
consistent with every observation whose `lambda_max` is arbitrarily larger. Constructively: for
block-diagonal `A` with a start vector inside one block, the Krylov space never leaves that block — we
measured a true `lambda_max` of 1000.0 against a Lanczos bound of 4.68, and **16 random restarts do not
help** (200/200 still invalid), because the restarts share the invariant subspace. Cauchy interlacing
makes the top Ritz value a *lower* bound on `lambda_max`, so no amount of `abs()` converts it into an
upper one. Production Chebyshev codes (ChASE, EVSL, ChebFD) use Ritz-plus-residual forms and call them
*estimates*; none claims rigour.

So a matvec-only caller genuinely cannot be served safely, and guessing on its behalf is what produced
`converged=True` on an excited state. Hence the raise.

## 4. One correction to the report — and to our own first measurement

The report says a looser `hi` "costs no speedup", supported by identical iteration counts. Our first
measurement agreed, and went further: counts were unchanged from 1× to **1660×** the coefficient sum.
**Both of us were reading the wrong quantity.**

The filtered vector's ground-state overlap does degrade with looseness:

| `hi` relative to true `lambda_max` | overlap after filtering |
| --- | --- |
| 1.6× | 0.779 |
| 26× | 0.559 |
| 415× | 0.095 |
| 6600× | 0.018 |

That follows the `sqrt(width)` law for the degree needed at a given damping. Iteration counts hide it
because the prefilter only has to get the iterate into the ground state's basin — LOBPCG does the rest,
and at these sizes it absorbs the difference entirely.

Practical consequence: your conclusion (use the coefficient sum) is right, but the *reason* matters if
anyone is ever tempted to inflate `hi` for safety. `sum|c_k|` is ~1.8× loose, which is comfortably in
the flat region. Don't pad it by orders of magnitude. The asymmetry still favours loose over tight —
over-estimating degrades smoothly, under-estimating changes the answer — so a deliberately generous
bound is safe, just not a wildly generous one.

## 5. A second defect in `sqd`'s initial vector — found here, **now also fixed**

**Update: fixed. `skqd/sqd_backend.py` is no longer blocked.** Both paths are safe to re-enable.
Left in place below as the record of what it was, since it affected every `sqd` call and not just
prefiltered ones, on every revision that has `_spread_seed`.


Found while validating this fix against your `skqd` regime (sampled subspaces, `Bx = 0`). It is
**unrelated to the prefilter** — it predates it, and `568b173` did not address it; the fix is
`5170290`, described at the end of this section. Reproducer, against a tree without that commit:

```python
import numpy as np, jax
jax.config.update("jax_enable_x64", True)
from rqutils.sqd import sqd

n = 4
strings, coeffs = [], []
for i in range(n - 1):
    for p in "XY":
        s = ["I"] * n; s[i] = s[i + 1] = p; strings.append("".join(s)); coeffs.append(0.25)
    s = ["I"] * n; s[i] = s[i + 1] = "Z"; strings.append("".join(s)); coeffs.append(0.25)

states = np.array([[0, 1, 0, 1], [1, 1, 0, 1]], dtype=np.uint8)   # a 2-state sampled subspace
print(sqd((strings, coeffs), states, return_eigvec=False))        # -0.25, true answer is -0.75
```

Returns `-0.25` against a true `-0.75`, `converged=True`, **with or without `prefilter`**. The
projected operator is correct (we probed it: exactly `diag(-0.75, -0.25)`, symmetric), and
`ground_locg` solves that matrix correctly from any start we tried. The fault is `sqd`'s initial
vector:

- `_spread_seed`'s bit-mixing hash maps index 0 to **exactly `-1.0`** — verified at every
  `states_size` from 2 to 1024, because the Murmur-style mixer maps 0 to 0 and the affine map sends
  that to `-1.0`.
- `vinit_from_min_diag` then does `seed.at[imin].add(1.0)`.
- So whenever `argmin(diagonal) == 0`, the two **cancel exactly** and the iterate starts with a zero
  component on basis state 0, violating `ground_locg`'s documented non-vanishing-overlap precondition.
  Here the diagonal is `[0.75, 0.75]`, `argmin` is 0, `vinit` becomes `[0.0, -0.183]`, and since the
  operator is diagonal the solver converges in **0 iterations** to the wrong eigenvalue.

This hits your regime specifically: it needs `argmin(diagonal) == 0`, which is likelier on small,
degenerate-diagonal subspaces — exactly what `skqd/sqd_backend.py::ground_state` samples, and exactly
where you have no independent oracle. Measured 1 in 18 random sampled subspaces of `Bx = 0` Heisenberg
at n = 4–8.

**The fix**: the weight now carries the seed component's own sign, so it reinforces instead of
subtracting — `seed.at[imin].add(jnp.sign(seed[imin]))`, giving `|vinit[imin]|` in `[1, 2)` whatever the
seed. `jnp.sign` rather than `copysign` because the seed is complex whenever the coefficients are.

Structural rather than a special case on index 0, deliberately: exact cancellation is only reachable
there, but **511 of `2**20` indices carry a seed within 1e-3 of −1.0**, and each of those would have
lost all but a thousandth of the component — a slow-convergence or wrong-answer risk that no test would
have attributed to this. After the fix, **0 of 676** randomly sampled subspaces are wrong (previously 1
in 18 on the `Bx = 0` family), and the reported 2-state case returns `-0.75`.

Three regression tests in `TestSqdMinDiagWeightCancellation`, verified against the restored `+1.0`. Two
pin the *preconditions* rather than the symptom — that `seed[0]` is exactly −1.0, and that
`|seed[i] + sign(seed[i])| >= 1` for every `i` — because the symptom test alone would keep passing if a
future change to the mixer moved the cancellation site somewhere the fixture does not reach.

## 6. Accepted: `degree=64`

Your sweep (median 1.35× through `sqd`, 0.74–0.80× dense) contradicted the docstring's "useful range
32–64". Narrowed: `ground_locg`'s docstring now records that an independent sweep measured 64 as the
weakest arm, and recommends only `(32, 2)` without qualification.

We also narrowed the claim your report correctly attacked. Both docstrings said the prefilter "cannot
change the answer, only the path", resting on the residual test — and as you noted, the premise is true
while the conclusion does not follow, since a converged eigenpair of a *different* eigenvalue also has
a small residual. That sentence now carries the caveat that it holds only when `prefilter_hi` is a
valid bound.

## 7. Why our own tests missed it

For the record, since it bears on what to trust. `TestSqdPrefilter`'s 20 tests passed throughout: its
fixtures are random Pauli strings, whose spectra do not lean negative enough to invert the interval —
the same blind spot your report names as "a transverse field masks the bug".

Worse, our first regression test *also* passed against the unfixed code. It used an arbitrary seed
(20260828). The bound is invalid for every seed we tried, but whether the ground state additionally
lands inside the damped band varies: seeds 0 and 2 return `+0.25`, seeds 1, 3 and 20260828 return the
correct `-0.75`. Only a full revert of `ground_locg.py` surfaced this — a hand-written mutant of the
bound did not, because it fed the power iteration a different start vector than the original did. The
committed tests pin seed 0 and assert `|lambda_min| > |lambda_max|` on the fixture, so they cannot
silently stop testing the failure.
