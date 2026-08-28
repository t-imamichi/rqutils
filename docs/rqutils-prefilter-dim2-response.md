# Response: the dim-2 raise is fixed at the root, and nothing to adapt

Reply to `docs/rqutils-prefilter-dim2-request.md`, from the `rqutils` side. Branch `dev`, version still
`0.2.0` (unreleased). Fix in `8358180`.

**You asked for an explicit answer, so: neither (1) nor (2) — we took (3).** The underlying
`vinit`/filter interaction you could not localize from outside the jit boundary turned out to be four
lines, so documenting a limitation or gating on dimension were both unnecessary. Please note in
`spinchain`'s `options.SQD_PREFILTER` comment that **dim-2 positive subspaces are fixed, not a
documented limitation**.

**Nothing for you to change.** No API moved, no default changed on this account, and `prefilter=(32, 2)`
at both your `sqd()` call sites is correct as-is. It arrives with your next lockfile bump.

Thank you for the report. The `1.887e-16` observation in §1 is what made this findable — a "did not
converge" with a plausible-looking energy would have read as a hard problem; a value driven to the
rounding floor said the iterate had been destroyed, which is a different search.

---

## 1. One correction: the fault is in `ground_locg`, not `sqd`

Your §2 concluded the fault was in `sqd`'s path because `ground_locg` solved the same projected operator
correctly at every prefilter arm. That inference was sound but the conclusion is inverted — the
difference was not `sqd`, it was *which vector* each path handed the solver. `sqd` supplied the input
that exposes an existing `ground_locg` defect:

```python
import numpy as np, jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from rqutils.ground_locg import ground_locg

mat = np.array([[2.9, 1.0], [1.0, 2.9]])          # your projected operator, eigs [1.9, 3.9]
w, V = np.linalg.eigh(mat)
v0, v1 = jnp.asarray(V[:, 0]), jnp.asarray(V[:, 1])

ground_locg(jnp.asarray(mat), v0)                  # BEFORE: theta 2.0e-16, converged=False
ground_locg(jnp.asarray(mat), v0 + 1e-16 * v1)     # BEFORE: theta 1.9, converged=True, 1 iteration
```

No `sqd`, no prefilter. The trigger is an `xinit` that already *is* an eigenvector — and with
`prefilter` on, a 2-dimensional iterate lands there routinely, which is why your reproducer needed
`sqd`.

**Mechanism.** `body_iter1` formed its search direction as a bare `normalize(rcurr, norm_r)`, dividing
by the residual norm however small it is. An `xinit` that is an eigenvector *in floating point* leaves a
residual at the **rounding floor** — 3.1e-16 here — not exactly zero, so the pre-existing
`norm_r == 0.0` guard missed it. Dividing by that floor amplified pure rounding noise until `tmp_p` came
back **parallel to `xcurr`**, and the projected matrix degenerated:

```
sas = [[ 1.9, -1.9],
       [-1.9,  4.8]]      lowest eigenvalue 0.96  ->  theta, for a true 1.9
```

At iteration 0 theta was still correct (1.9, residual 2.4e-16) with `converged=False`; iteration 1
destroyed it. Every subsequent iteration then re-derived noise, which is why raising `maxiter` did not
help and why the message's advice was inapplicable, exactly as you observed.

Your §3 reading — "a relative residual test whose denominator vanishes" — was close. The `λ₀ > 0.45·width`
boundary is real but incidental: it is the regime where the filter can drive a 2-dimensional iterate onto
the eigenvector *to within rounding*, not a property of the residual test.

## 2. Two fixes we tried and rejected

Recording these because both look correct, and the second produced a worse failure than the one you
reported.

**(a) Masking `sas[1, 1]`.** `body_iter1` already had an `r_is_zero` path that lifts the `p` diagonal out
of contention. It *fires correctly* — we verified `sas[1, 1] = 4.8 = 2|ρ| + 1` — and is still
insufficient, because the surviving **off-diagonal** keeps coupling `x` to the noise direction. Lifting a
diagonal only works when the off-diagonals are already negligible, which is precisely what a parallel
`tmp_p` breaks.

**(b) A scale-relative residual floor**, `eps · dim · max(|ρ|, 1)`. This was our first fix. It repaired
every cell of your §3 table and then **broke a cell that previously passed** — `dim=2, shift=-0.1`:

| | |
| --- | --- |
| residual norm | `8.07e-16` |
| floor | `7.99e-16` |

A 1% margin deciding correctness. And loosening the floor is worse than useless: it pins `theta = ρ` when
the iterate is *not* an eigenvector, which returned **0.96 silently** on your original case. We reverted
it — a silent wrong answer is strictly worse than your loud failure, and this is the same trap as the
previous report.

## 3. The fix

`body_iter1` now forms its direction with `_project_out((xcurr,), rcurr)` — which `body()` has always
used, and which this step was simply missing. It needs **no threshold**: it renormalizes, subtracts the
basis again, and returns *exactly* zero when the norm collapses below 0.99, which is "this direction was
rounding noise" expressed structurally rather than as a tolerance. It also drops a redundant norm
reduction, so the step is marginally cheaper than before.

## 4. Verification

Your reproducer, all six arms from your §2 table:

| arm | eigval | arm | eigval |
| --- | --- | --- | --- |
| `None` | 1.9000000000 | `(16, 2)` | 1.9000000000 |
| `(2, 1)` | 1.9000000000 | `(32, 2)` | 1.9000000000 |
| `(8, 1)` | 1.9000000000 | `(64, 2)` | 1.9000000000 |

Your §3 dimension × shift sweep: **0 failures in 42 cells** (dims 2, 3, 4, 6, 8, 12, 16 × shifts +2, +1,
+0.1, 0, −0.1, −1). Your magnitude axis: ok at every `λ₀` from 0 to +11.9.

Beyond your cases: **243 random `sqd` subspaces** (nq 2–6, dims 2 to full) pass, and `ground_locg` fed
the exact ground eigenvector is **0/120 failures across dims 2–40**, real and complex operators.

Test suite 592 → 600. Six of the eight new tests fail against the restored bare `normalize`, so they pin
the defect rather than merely covering the area.

## 5. One place your framing was too narrow

**This was not dim-2-specific.** Your §4 argues the regime is marginal — "dim 2 is one above the `dim == 1`
case, and a strictly positive projected spectrum means the subspace excludes the ground state entirely".
That is right about *your* trigger, but the underlying defect has **no dimension dependence**: any caller
passing a near-exact eigenvector as `xinit` hit it at any size. dim 2 is only where `sqd`'s prefilter
lands on the eigenvector routinely.

So the affected-caller class is wider than the report suggests — it includes anyone warm-starting
`ground_locg` from a previous solve's eigenvector, which is a normal thing to do. That is why we fixed it
rather than documenting it, and it is a fair correction to your own priority assessment: the report was
right that *you* were not blocked, and understated who else could be.

Your §4 note that 240 dim-2 XXZ subspaces produced 0 raises still holds and is consistent — a
quantum-sampled basis is seeded from low-energy configurations, so the filter rarely lands exactly on an
eigenvector there.

## 6. On §6 (`precond`), since you raised it

Withdrawn on your side, and now moot: `precond` is **deleted** as of `0381ea0`, so it will not appear in
your next bump either.

For the record, your instinct was right and better-founded than you claimed. You wrote that the 1.79×
"was iteration count on `scipy`'s `lobpcg`, which is exactly the kind of proxy that §4 of your last reply
showed can mislead". Measured end-to-end through `sqd`, that is what happened: the prefilter's benefit
falls from 5.02× counting iterations to 1.49× in wall clock, and `precond` cannot be used from `sqd` at
all — `sqd` solves the raw indefinite projected `H`, where literal Jacobi does not merely regress but
**fails to converge** (8000-iteration cap, wrong answer, 3/3 sizes we tried). We also measured that the
two **anti-compose**: adding `precond` to a prefiltered run halves the gain from `(32, 2)` up.

One thing that did change on your behalf without your asking: **`sqd` now defaults to
`prefilter=(32, 2)`**, so your explicit argument is now redundant (harmless, and worth keeping if you want
the default pinned against future changes). Measured 1.49× median end-to-end, min 1.15×, 0 regressions.
