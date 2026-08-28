# rqutils bug report: `ground_locg`'s Chebyshev prefilter can return an excited state

> ## Disposition (2026-08-28): **confirmed in full, FIXED**
>
> **Reply to the reporter, with migration steps: `docs/rqutils-prefilter-bug-response.md`.**
> It also carries one correction to this report (§4) and a **second, unrelated `sqd` defect**
> found while validating the fix, which is *not* fixed and still blocks `skqd/sqd_backend.py`
> (§5).
>
> Fixed by deleting `_lambda_max_bound` outright and taking the bound from the operator's structure
> instead. `_chebyshev_prefilter` now requires `hi`; `ground_locg` gains `prefilter_hi=`, derived
> automatically as Gershgorin `max_i sum_j |A_ij|` when `mat` is an array and **required** when it is a
> callable; `sqd` passes `sum|c_k|`. Wrong answers 2/25 -> **0/25**, and the n=2 case returns -0.75 on
> both entry points. Iteration reduction *improved* to a 3.29x median (from the 1.88x recorded with the
> unsound bound), since the 11 wasted matvecs are gone too. Suite 581 -> 585.
>
> **Breaking change**: `ground_locg(callable, ..., prefilter=...)` now raises without `prefilter_hi`.
> Deliberately a raise rather than a fallback -- see the impossibility result below. `sqd` callers and
> array callers are unaffected.
>
> ### Where this report's proposed fix was not enough
>
> The report's fallback suggestion, `hi = max(estimate, -estimate)`, was measured **invalid in 4 of
> 25** configurations: fixing the sign leaves a fixed 10-step iteration simply under-converged. A
> `sqrt`-of-`A^2` variant reached 24/25. Lanczos `mu_max + beta_k` looked perfect on these physics cases
> (25/25) and fails **12/1000** adversarially -- the same "validated on the easy regime" trap that let
> the original bug ship.
>
> **No matvec-only method can be rigorous, and this is a theorem, not a tuning problem.** Kuczynski &
> Wozniakowski (SIAM J. Matrix Anal. Appl. 13(4):1094-1122, 1992) prove that with fewer than `N`
> matvecs another operator consistent with every observation has an arbitrarily larger `lambda_max`.
> Constructively: for block-diagonal `A` with a start vector inside one block, Krylov never leaves it --
> measured, a true `lambda_max` of 1000.0 against a Lanczos bound of 4.68, and **16 random restarts do
> not help** (200/200 still invalid), because the restarts share the invariant subspace. Production
> libraries (ChASE, EVSL, ChebFD) use Ritz-plus-residual forms and call them *estimates*; EVSL's own
> `mu_max + |beta_k s_k|` measured invalid in **14% of 4000** random Hermitian cases. Cauchy
> interlacing makes the top Ritz value a *lower* bound, so no amount of `abs()` can convert it.
>
> ### One correction to this report, and one to our own earlier reasoning
>
> The report says a looser `hi` "costs no speedup", and our first measurement agreed -- iteration
> counts were identical from 1x to 1660x the coefficient sum. **Both were reading the wrong quantity.**
> The filtered vector's ground-state overlap does degrade with looseness (0.78 at 1.6x `lambda_max`,
> 0.094 at 415x, 0.018 at 6600x), following the `sqrt(width)` law for the degree needed at a target
> damping. Iteration counts hid it because the prefilter only has to get the iterate into the ground
> state's basin before LOBPCG takes over. So tightness *does* matter, which is why `sum|c_k|` (~1.8x
> loose) is preferred over a deliberately inflated fallback -- and why the asymmetry still favours
> loose over tight: over-estimating degrades smoothly, under-estimating changes the answer.
>
> ### `degree=64`
>
> Accepted. `ground_locg`'s docstring no longer advertises a "useful range 32-64"; it now records that
> an independent sweep measured 64 as the weakest arm and recommends only `(32, 2)`.
>
> Every claim reproduced against `dev` at `f249ce1` before any code was changed. Nothing in the report
> needed correction:
>
> | Claim | Reported | Measured here |
> | --- | --- | --- |
> | n=2 Heisenberg returns an excited state | `+0.25` for `-0.75`, `converged=True` | identical |
> | `_lambda_max_bound` invalid | 16 of 25 | **16 of 25** |
> | Silent wrong answer | 2 of 25 (both n=2, Bx=0) | **2 of 25**, same two |
> | Coefficient-sum bound valid | 25 of 25 | **25 of 25** |
>
> **The root cause is as diagnosed.** Power iteration converges to the eigenvalue of largest
> *magnitude*; on a negative-leaning spectrum that is `lambda_min`, so `hi` lands at or below
> `lambda_0`, `half = (hi - theta) / 2` goes negative, and the filter damps its own target.
> `_lambda_max_bound`'s docstring anticipates the sign question for `margin` -- and is right that
> `|estimate| * (margin - 1)` widens either way -- but widening cannot repair an estimate that
> converged to the opposite end.
>
> **Two additions from this side.**
>
> 1. **It reaches `sqd`, not only dense `ground_locg`.** Confirmed on a full-basis Heisenberg subspace
>    through `sqd(prefilter=(32, 2))`: n=2 returns `+0.25` against a true `-0.75`; n=3 and n=4 are
>    correct. So both public entry points carry it.
> 2. **The report's critique of our own docstrings is correct.** `sqd` and `ground_locg` both claimed
>    the prefilter "cannot change the answer, only the path", resting on the residual test. That
>    premise is true and the conclusion does not follow: a converged eigenpair of a *different*
>    eigenvalue also has a small true residual, so the test proves *an* eigenpair was found, not the
>    lowest. The claim is being narrowed rather than kept.
>
> **Why `TestSqdPrefilter` (20 tests) passes while this is broken:** its fixtures are random Pauli
> strings, whose spectra do not lean negative enough to invert the interval -- the same blind spot the
> report names as "a transverse field masks the bug". A regression test therefore has to pin a
> *specific* negative-leaning spectrum, not a random one.
>
> **Also accepted:** the `degree=64` measurements here (0.74-0.80x on dense) contradict
> `ground_locg`'s advertised "useful range 32-64". That recommendation needs narrowing too; it is
> tracked separately from the correctness fix.


One bug report against `rqutils` on branch `dev` (installed rev `a013322`, reporting version 0.2.0),
written from the `spinchain` side. It concerns the `prefilter` option added to
`ground_locg`/`sqd` after `docs/rqutils-precond-request.md` was filed, and is independent of every
request in `docs/rqutils-requests.md`.

**Bug in one line:** `_lambda_max_bound` returns an estimate of the eigenvalue of largest
*magnitude*, not of the algebraic *maximum*, so on an operator whose spectrum leans negative the
Chebyshev filter interval is inverted and can damp the ground state — `ground_locg` then returns an
**excited** eigenpair with `converged=True` and no error raised.

| | |
| --- | --- |
| Where | `rqutils/ground_locg.py:290`, `_lambda_max_bound`; consumed at `:353` by `_chebyshev_prefilter` |
| Reached from | `ground_locg(..., prefilter=)` (`:769`) and `sqd(..., prefilter=)` (`sqd.py:917`) |
| Kind | **silent wrong answer** — wrong eigenvalue, wrong eigenvector, `converged=True` |
| Trigger | `abs(lambda_min) > abs(lambda_max)` **and** the ground state falls inside the damped band |
| Severity | opt-in path, so no default caller is affected; but the default guidance is `(32, 2)` |
| Bound validity | **invalid in 16 of 25** XXZ configurations tested, including n=6 and n=8 |
| Wrong answers | **2 of 25** configurations tested (both `n=2`, `Bx=0`) |
| Suggested fix | replace the power iteration with a coefficient-sum bound — valid 25/25, **zero** matvecs, and measured to cost no speedup |

Everything below was measured on that revision. Reproductions import `rqutils`, `jax`, `numpy`,
`scipy` and `qiskit` only — no `spinchain` — but they assume **64-bit jax**, so they call
`jax.config.update("jax_enable_x64", True)` before the first array is created.

---

## Symptom: the n=2 Heisenberg chain returns `+0.25` instead of `-0.75`

`spinchain` wired `prefilter=(32, 2)` into its two `sqd()` call sites and its test suite failed on the
smallest case it has:

```text
FAILED test/test_exact.py::test_eigval_negative[rqutils-2]
AssertionError: Expected negative eigval, got 0.25
```

The antiferromagnetic Heisenberg spectrum at n=2 is `[-0.75, 0.25, 0.25, 0.25]`. The solver returned
`+0.25` — a genuine eigenvalue, but the wrong one, and it reported convergence.

```python
import jax, jax.numpy as jnp, numpy as np
jax.config.update("jax_enable_x64", True)
from qiskit.quantum_info import SparsePauliOp
from rqutils.ground_locg import ground_locg

# n=2 antiferromagnetic Heisenberg, spinchain's normalization: 0.25*(XX + YY + ZZ)
ham = SparsePauliOp.from_list([("XX", 0.25), ("YY", 0.25), ("ZZ", 0.25)])
A = jnp.asarray(ham.to_matrix())
x0 = jnp.asarray(np.random.default_rng(0).normal(size=4) + 0j)

print(np.linalg.eigvalsh(np.asarray(A)))          # [-0.75  0.25  0.25  0.25]
print(float(ground_locg(A, x0, maxiter=2000)[0]))                      # -0.75  correct
print(ground_locg(A, x0, maxiter=2000, prefilter=(32, 2))[0::3])       # +0.25, converged=True
```

## Root cause: power iteration finds the wrong end of the spectrum

`_lambda_max_bound` (`:290`) runs `steps=10` of power iteration and scales by `margin=1.05`:

```python
def step(vec, _):
    return normalize(matvec(vec, *args)), None

vec = jax.lax.scan(step, normalize(vector), None, length=steps)[0]
estimate = jnp.sum(vec.conjugate() * matvec(vec, *args)).real
return estimate + jnp.abs(estimate) * (margin - 1.0)      # ground_locg.py:317
```

Power iteration converges to the eigenvalue of largest **absolute value**. For an antiferromagnetic
XXZ chain that is the most *negative* eigenvalue, so `estimate` approaches `lambda_min`, and the
returned "upper bound on `lambda_max`" lands at or below `lambda_min`:

| case | spectrum | returned bound | true `lambda_max` | valid? |
| --- | --- | --- | --- | --- |
| Heisenberg n=2 | -0.7500 … +0.2500 | **-0.7125** | +0.2500 | no |
| n=3 | -1.0000 … +0.5000 | **-0.9500** | +0.5000 | no |
| n=4 | -1.6160 … +0.7500 | **-1.5349** | +0.7500 | no |
| n=8 | -3.3749 … +1.7500 | **-3.0941** | +1.7500 | no |

`_chebyshev_prefilter` (`:320`) then builds its interval from that bound:

```python
hi = _lambda_max_bound(matvec, args, vector)        # ground_locg.py:353
...
theta  = jnp.sum(vec.conjugate() * matvec(vec, *args)).real
centre = (hi + theta) / 2
half   = (hi - theta) / 2
```

With `hi < theta` the interval `[theta, hi]` is **inverted** and `half` is negative. At n=2 the first
cycle gets `[theta, hi] = [-0.4067, -0.7125]`, which does not contain the ground state at `-0.75` —
so the filter damps its own target instead of amplifying it, exactly the failure the docstring warns
about for a different reason.

### Why the existing docstring reasoning does not cover this

`_lambda_max_bound`'s docstring says:

> power iteration converges to the dominant eigenvalue from below, so the `margin` covers the
> shortfall

That is true for magnitude and irrelevant for sign. `margin` is applied as
`estimate + abs(estimate) * (margin - 1.0)`, and the docstring correctly explains this widens the
interval for either sign — but widening cannot repair an estimate that converged to the *opposite
end* of the spectrum. A 5% multiplicative slack on `-1.425` gives `-1.496`, moving it further from
`+0.5`.

`_chebyshev_prefilter`'s docstring is careful about the *lower* edge, and its warning is sound:

> THE LOWER EDGE IS THE CURRENT RAYLEIGH QUOTIENT, RE-READ EACH CYCLE, AND THAT CHOICE IS
> LOAD-BEARING […] A Rayleigh quotient starts *above* `lambda_0` and descends toward it, so it can
> never bracket the target out.

The reasoning is right and the invariant holds — but it protects only the edge the Rayleigh quotient
supplies. Nothing establishes the corresponding property for `hi`, and when `hi` drops below
`lambda_0` the bracket-out the docstring rules out for `theta` happens anyway, from the other side.

### Why the convergence test does not catch it

Both `ground_locg`'s and `sqd`'s docstrings state that the prefilter cannot affect the result:

> **It cannot change the answer, only the path**: every convergence test still reads the true
> residual

The premise is true and the conclusion does not follow. A converged eigenpair of a *different*
eigenvalue also has a small true residual: `A v = 0.25 v` holds exactly for the returned vector. The
residual test verifies that **an** eigenpair was found, not that it is the lowest one. So this
guarantee holds against a preconditioner (which only rescales a search direction) but not against a
prefilter (which can remove the target from the starting vector's effective span).

This also defeats the independent oracle on the `spinchain` side:
`exact.py::_RQUTILS_RESIDUAL_TOLERANCE` recomputes `<v|H|v>` and compares it against the reported
eigenvalue. For the `+0.25` result those agree to machine precision, so the guard passes.

## Scope: the bound is invalid almost everywhere; the wrong answer is rarer

Two distinct questions, with different answers. Reproducer:

```python
import jax, jax.numpy as jnp, numpy as np
jax.config.update("jax_enable_x64", True)
from qiskit.quantum_info import SparsePauliOp
from rqutils.ground_locg import ground_locg, _lambda_max_bound

def xxz(n, delta=1.0, bx=0.0):
    terms = []
    for i in range(n - 1):
        for p in "XY":
            s = ["I"] * n; s[i] = s[i + 1] = p; terms.append(("".join(s[::-1]), 0.25))
        s = ["I"] * n; s[i] = s[i + 1] = "Z"; terms.append(("".join(s[::-1]), 0.25 * delta))
    for i in range(n) if bx else ():
        s = ["I"] * n; s[i] = "X"; terms.append(("".join(s[::-1]), bx))
    return SparsePauliOp.from_list(terms)

for n in (2, 3, 4, 6, 8):
    for delta, bx in ((1.0, 0.0), (1.0, 0.5), (2.0, 0.0), (0.5, 0.1), (-1.0, 0.0)):
        ham = xxz(n, delta, bx)
        A = jnp.asarray(ham.to_matrix())
        w = np.linalg.eigvalsh(np.asarray(A))
        x0 = jnp.asarray(np.random.default_rng(0).normal(size=2**n) + 0j)
        hi = float(_lambda_max_bound(lambda v: A @ v, (), x0))
        ev = float(ground_locg(A, x0, maxiter=2000, prefilter=(32, 2))[0])
        print(f"n={n} delta={delta} Bx={bx}  bound_valid={hi >= w[-1]}  "
              f"wrong={abs(ev - w[0]) > 1e-6}")
```

- **The bound is invalid in 16 of 25** `(n, delta, Bx)` combinations across n=2–8. The valid cases
  are accidents rather than the intended behaviour: the ferromagnetic sign (`delta=-1.0`) puts
  `abs(lambda_max)` above `abs(lambda_min)` at every size, and a transverse field does the same at
  some sizes.
- **The wrong answer appears in 2 of 25** configurations — both at `n=2` with `Bx=0`
  (`delta=1.0` returns `+0.25` for `-0.75`; `delta=2.0` returns `0.0` for `-1.0`).

The gap between those two numbers is what makes this dangerous rather than obvious: the precondition
is violated almost everywhere, and the *consequence* only surfaces when the ground state additionally
lands inside the damped band. Small or strongly degenerate spectra hit it; the ferromagnetic sign
(`delta=-1.0`) never does, because there `abs(lambda_max) > abs(lambda_min)` and the bound is valid by
luck.

**A transverse field masks the bug.** Every configuration in our first verification sweep carried
`Bx >= 0.1`, and all 22 instances passed with eigenvector overlap `1.0000000` against the unfiltered
solve. The failing corner is pure `Bx = 0` Heisenberg — so a test matrix that always includes a field
will report the prefilter as exact.

## Suggested fix: bound `lambda_max` from the coefficients, not by iterating

The cheapest correct bound needs no matvec at all. For `H = sum_k c_k P_k` with Pauli strings `P_k`,
every eigenvalue satisfies `abs(lambda) <= sum_k abs(c_k)`, so
`hi = sum(abs(ham.coeffs))` is always a valid upper bound. Equivalently, for a general matrix, the
1-norm bound `max_i sum_j abs(A_ij)`.

Measured against the same 25 configurations: **valid in 25/25**, versus 9/25 for the current
estimate.

The obvious objection is that a looser `hi` widens the damped band and costs separation. Measured, it
does not:

| bound | median speedup vs unfiltered | iteration counts |
| --- | --- | --- |
| current (invalid) | 1.29x | 14, 38, 19, 22 |
| coefficient-sum | **1.28x** | 14, 38, 19, 22 — **identical** |

Dense `ground_locg`, `prefilter=(32, 2)`, best-of-3, n=8 and n=10 at `Bx` 0.0/0.5. The iteration
counts are unchanged in every case, so the looser bound costs nothing measurable while removing the
failure mode and ~11 matvecs per solve.

A variant that keeps the power iteration — shifting to `A - s*I` with `s` a valid lower bound so the
shifted spectrum is one-signed — was also tested and **rejected**: at `steps=10` it under-estimates on
larger chains and was invalid in 4 of 15 cases tried (n=6 and n=8). Raising `steps` would cost the matvecs
the coefficient-sum bound avoids entirely.

If the estimate must stay iterative for operators supplied only as a `matvec`, then the correct
minimum is to take `hi = max(power_iteration_estimate, -power_iteration_estimate)` — or better, to
document `prefilter` as requiring a caller-supplied `lambda_max` upper bound, since the caller
usually knows it in closed form. A `hi` argument on `prefilter` would let `spinchain` pass
`sum(abs(coeffs))` directly.

## What `spinchain` does in the meantime

**The wiring is reverted; `spinchain` does not pass `prefilter` on any path.** Both call sites
(`exact.py::_rqutils_ground_state` and `skqd/sqd_backend.py::ground_state`) reach LOBPCG through
`sqd()`, and neither can afford this failure mode:

- `exact.py` is the reference solver other backends are checked against, and its residual guard
  cannot see the bug (above).
- `skqd/sqd_backend.py::ground_state` has no independent oracle at all — avoiding a reference solve is
  the point of the matrix-free path — and it runs on sampled subspaces whose spectra are small and
  frequently degenerate, which is the regime where the bug fires.

The measured payoff is worth returning to once the bound is fixed: **1.76x median** (range 1.36–2.24x)
through `sqd()` on sampled subspaces at n=14–18, dim≈2000, best-of-5, and 1.92x on the full basis.
Notably that is *better* than `sqd`'s own docstring predicts — it warns the figures were taken on
dense `ground_locg` and that `apply_h`'s gather-heavy matvec might not pay for the extra ~79 matvecs.
It does.

On knob choice, our measurements disagree with the docstring in one place: `degree=64` was the
weakest arm on every path here (median 1.35x through `sqd`, and 0.74x/0.80x on dense), rather than the
top of a useful 32–64 range. `(32, 2)` is the right default, chosen for having no regression on any
measured path.
