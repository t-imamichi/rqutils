# rqutils bug report: `prefilter` raises on a dim-2 subspace whose spectrum is strictly positive

> ## Disposition (2026-08-28): **confirmed, root-caused, and FIXED at the root**
>
> Not option (1) or (2) — the real fix turned out to be four lines, so we took it. Thank you for the
> report; the reproducer is exact and the `1.887e-16` observation is what made it findable.
>
> **§2's heading is the one thing to correct: the fault is in `ground_locg`, not `sqd`.** `sqd` only
> supplied the input that exposes it. It reproduces with no `sqd` involved:
>
> ```python
> ground_locg(jnp.asarray([[2.9, 1.0], [1.0, 2.9]]), v0)   # v0 = exact ground eigenvector
> # -> theta 2.0e-16, converged=False        (true 1.9)
> ground_locg(..., v0 + 1e-16 * v1)
> # -> theta 1.9, converged=True, 1 iteration
> ```
>
> **Mechanism.** `body_iter1` formed its search direction as a bare `normalize(rcurr, norm_r)`, which
> divides by the residual norm however small it is. An `xinit` that *is* an eigenvector in floating
> point leaves a residual at the **rounding floor** — 3.1e-16 here — not exactly zero, so the existing
> `norm_r == 0.0` guard missed it and the division amplified pure noise until `tmp_p` came back
> **parallel to `xcurr`**. `sas` degenerated to `[[1.9, -1.9], [-1.9, 4.8]]`, whose lowest eigenvalue is
> **0.96**, and Rayleigh–Ritz duly selected it. At iteration 0 theta was still correct at 1.9 with
> `converged=False`; iteration 1 destroyed it.
>
> Your reading in §3 — "a relative residual test whose denominator vanishes" — was close. The trigger is
> `lambda_0 > 0.45 * width` because that is the regime where the filter can drive a 2-dimensional
> iterate onto the eigenvector to within rounding; it is not a property of the residual test itself.
>
> **Two fixes we tried and rejected**, both of which look right:
>
> 1. **Masking `sas[1, 1]`** — the pre-existing `r_is_zero` path. It fires correctly (`sas[1, 1] = 4.8 =
>    2|rho| + 1`, verified) and is still insufficient, because the surviving *off-diagonal* keeps
>    coupling `x` to the noise.
> 2. **A scale-relative residual threshold.** This is what we wrote first. It fixed every cell in your
>    §3 table and then **failed at `dim=2, shift=-0.1`**, a cell that previously passed — `|r| = 8.07e-16`
>    against a floor of `7.99e-16`, i.e. a 1% margin deciding correctness. Loosening the floor makes it
>    pin `theta = rho` when the iterate is *not* an eigenvector, which returned **0.96 silently**. We
>    reverted it: a wrong answer is worse than your loud failure.
>
> **The fix.** `body_iter1` now uses `_project_out((xcurr,), rcurr)` — which `body()` has always used and
> which this step was simply missing. It needs no threshold: it renormalizes, subtracts the basis again,
> and returns *exactly* zero when the norm collapses below 0.99, which is precisely "this direction was
> rounding noise". Four lines, and it drops a redundant norm reduction.
>
> **Verification.** Your §3 sweep is 0 failures in 42 cells (dims 2–16 × six shifts), every prefilter arm
> in §2 returns 1.9000000000, and 243 random `sqd` subspaces pass. `ground_locg` fed the exact ground
> eigenvector: 0/120 failures across dims 2–40, real and complex. Suite 592 → 600.
>
> **On priority: you were right that it is low, and wrong that it is dim-2-specific.** The underlying
> defect has no dimension dependence — any caller passing a near-exact eigenvector as `xinit` hits it at
> any size. dim 2 is just where `sqd`'s filter lands there routinely. So the class of affected callers is
> wider than your §4 suggests, which is part of why we fixed it rather than documenting it.
>
> **§6 (`precond`), for completeness:** withdrawn on your side, and now moot — `precond` is deleted as of
> `0381ea0`. Your instinct about the 1.79× being an iteration-count proxy was correct: measured
> end-to-end it is the prefilter that wins, and `precond` cannot be used from `sqd` at all, since `sqd`
> solves the raw indefinite `H` where Jacobi fails to converge outright.


One low-priority bug report against `rqutils` on branch `dev` (installed rev `4b7be94`, version
`0.2.0`), written from the `spinchain` side. It is a follow-up to
`tmp/rqutils-prefilter-bug.md`, whose fix is verified and adopted — `spinchain` now passes
`prefilter=(32, 2)` at both `sqd()` call sites.

**Bug in one line:** with `prefilter` enabled, `sqd()` raises `RuntimeError: LOBPCG did not converge`
on a **two-state** subspace whose projected spectrum is strictly positive, for an operator
`ground_locg` itself solves correctly at every prefilter setting.

| | |
| --- | --- |
| Where | `rqutils/sqd.py`; the fault is in `sqd`'s own path, **not** `_chebyshev_prefilter` (see §2) |
| Kind | **fail-loud availability**, not a wrong answer — it raises rather than returning a plausible number |
| Trigger | `dim == 2` **and** `lambda_0 / (lambda_max - lambda_0)` above ~0.45 |
| Not triggered by | `dim >= 4` at any shift; `dim == 2` with `lambda_0 <= 0`; any prefilter arm on the dense path |
| Reaches `spinchain`? | **No.** 240 dim-2 XXZ subspaces (52 with `lambda_0 > 0`): 0 raises, 0 wrong |
| Severity | low — it costs an unnecessary raise on an easy problem, and never a silent wrong answer |
| Priority | **your call; we are not blocked.** Filed for completeness, not as an ask we need |

Everything below was measured on that revision. The reproducer imports `rqutils`, `numpy`, `jax` and
`qiskit` only — no `spinchain` — and assumes **64-bit jax**, so it calls
`jax.config.update("jax_enable_x64", True)` before the first array is created.

---

## 1. Reproducer

```python
import numpy as np, jax
jax.config.update("jax_enable_x64", True)
from qiskit.quantum_info import SparsePauliOp
from rqutils.sqd import sqd, hproj

# IIIX couples bit 0; the Z terms give the two states different diagonal energies.
ham = SparsePauliOp.from_list([("IIIX", 1.0), ("IIZI", 0.7), ("IZII", 0.9), ("ZIII", 1.3)])
states = np.array([[0, 0, 0, 0], [0, 0, 0, 1]], dtype=np.uint8)   # dim 2, connected

print(np.linalg.eigvalsh(hproj(ham, states).toarray()))    # [1.9  3.9] -- strictly positive
print(sqd(ham, states, return_eigvec=False))               # +1.9, correct
print(sqd(ham, states, return_eigvec=False,
          prefilter=(32, 2)))                              # RuntimeError: did not converge
```

The raised message is the `maxiter` one, and it is misleading here rather than wrong: raising
`maxiter` does not help, because the iteration is not slowly converging. The subspace is dim 2 with a
relative gap of **1.0** — the easiest problem in the space.

The value it reports reaching is the tell: **`1.887e-16`**, not a slightly-wrong energy near `+1.9`.
The iterate has been driven to numerical zero, so the Rayleigh quotient is `0/0` noise rather than a
variational upper bound — which makes the message's "variational upper bound" wording wrong in this
case specifically, since `1.9e-16` is far *below* the true `+1.9` and no upper bound could be. That is
also why the suggested remedies do not apply: there is no iterate left to converge.

## 2. The fault is in `sqd`, not in the filter

`ground_locg` solves the *same projected operator* correctly at **every** prefilter arm, including the
one `sqd` fails with. Passing the dense 2x2 directly, with `prefilter_hi` set to the same Gershgorin
bound `sqd` computes internally:

| arm | eigval | converged | iters |
| --- | --- | --- | --- |
| `None` | +1.900000000 | True | 1 |
| `(2, 1)` | +1.900000000 | True | 1 |
| `(8, 1)` | +1.900000000 | True | 1 |
| `(16, 2)` | +1.900000000 | True | 1 |
| `(32, 2)` | +1.900000000 | True | 1 |
| `(64, 2)` | +1.900000000 | True | 1 |

So `_chebyshev_prefilter` and the new `sum|c_k|` bound are both fine on this operator. What differs on
the `sqd` path is the initial vector: `vinit` there comes from `_spread_seed` /
`vinit_from_min_diag` rather than from a caller-supplied `xinit`, and at dim 2 that seed interacts with
the filter in a way the dense path never sees. We did not localize it further — `run_sqd` is
`@jax.jit`-wrapped, so a spy on the traced `vinit` hits
`TracerArrayConversionError` and reading it needs an upstream-side probe.

Worth noting because §5 of your reply fixed a closely related interaction: an exact cancellation
between `_spread_seed`'s hash and `vinit_from_min_diag`'s weight. This is a *different* symptom
(raise, not wrong answer) and the earlier fix is present in the tree we measured, so it is not a
regression of it — but the same two components are involved, and dim 2 is the smallest case where
`vinit_from_min_diag` has anything to rank.

## 3. Trigger boundary

Two knobs, swept independently. Shifting the identity moves `lambda_0` without changing the
eigenvectors or the width:

| dim | `lambda_0=+2` | `+1` | `+0.1` | `0` | `-0.1` | `-1` |
| --- | --- | --- | --- | --- | --- | --- |
| **2** | **RAISE** | **RAISE** | ok | ok | ok | ok |
| 4 | ok | ok | ok | ok | ok | ok |
| 6 | ok | ok | ok | ok | ok | ok |
| 8 | ok | ok | ok | ok | ok | ok |
| 12 | ok | ok | ok | ok | ok | ok |
| 16 | ok | ok | ok | ok | ok | ok |

**`dim == 2` is the whole story on the dimension axis** — dim 4 is clean at every shift we tried.

On the magnitude axis it is not a sign test but a scale one. At fixed width 2.0:

| `lambda_0` | `lambda_0 / width` | result |
| --- | --- | --- |
| +0.000 … +0.500 | 0.000 … 0.250 | ok |
| **+0.900** | **0.450** | **RAISE** |
| +1.000 | 0.500 | RAISE |
| +2.000 | 1.000 | RAISE |
| +10.000 | 5.000 | RAISE |

So the boundary sits near `lambda_0 ≈ 0.45 * width`, and `lambda_0` exactly 0 or slightly negative is
always fine — consistent with a *relative* residual test whose denominator vanishes as the spectrum
straddles zero, though we did not confirm that reading.

Frequency, for sizing: 5 of 64 randomly drawn multi-block subspaces of the toy Hamiltonian raised,
and **all 5 were dim 2 with a strictly positive spectrum**. No case at any dimension returned a wrong
answer, filtered or not.

## 4. Why this is low priority, stated plainly

**It does not reach `spinchain`, and we are not asking for it on our own behalf.** Measured on 240
randomly drawn dim-2 subspaces of real XXZ chains (n = 6, 8, 10; `Bx = 0.5`; `By` 0.0 and 0.3),
through `spinchain.skqd.sqd_backend.ground_state` with `prefilter=(32, 2)` live: **0 raises, 0 wrong
answers**, with `lambda_0 > 0` in 52 of them. So the positive-spectrum half of the trigger does occur
in our regime and still does not fire, which suggests the toy's structure contributes something we
have not isolated.

Two further reasons not to rush it:

- **It fails loud.** The whole reason the previous report was urgent is that a bad bound returned a
  converged excited state. This is the opposite: no wrong number is produced, and a caller cannot
  mistake a `RuntimeError` for an answer.
- **The regime is marginal.** dim 2 is one above the `dim == 1` case, and a strictly positive projected
  spectrum means the subspace excludes the ground state entirely — uncommon for a sampled Krylov
  basis, which is seeded from low-energy configurations.

What would raise the priority: any caller that legitimately diagonalizes tiny positive-definite
subspaces — a positive-definite Gram or overlap matrix, or a shifted operator `H + cI` chosen to make
the spectrum positive — since for those the trigger is the normal case rather than a corner.

## 5. Suggested handling

In rough order of cost:

1. **Nothing, deliberately** — record the boundary in `sqd`'s `prefilter` docstring ("may fail to
   converge at `dim == 2` when the projected spectrum is strictly positive; pass `prefilter=None`
   there") and leave the code alone. Given §4 this is a defensible choice, and it is the one we would
   make.
2. **Skip the filter below a dimension threshold.** `dim <= 2` is solved in one iteration anyway — the
   table in §2 shows `iters=1` unfiltered — so `cycles * (degree + 1)` matvecs of prefilter buy
   nothing there even when they work. A static guard costs no traced ops for the sizes that matter and
   removes the failure class rather than documenting it. This is what we would suggest if you do want a
   code change.
3. **Fix the underlying `vinit`/filter interaction**, if it turns out to be the same family as §5 of
   your reply and is cheap once located. We could not localize it from outside the jit boundary.

Whichever you pick, one request: if the answer is (1) or (2), please say so explicitly rather than
leaving it open, so we can note in `spinchain`'s `options.SQD_PREFILTER` comment that dim-2 positive
subspaces are a documented limitation rather than an unknown.

## 6. Not requested: forwarding `precond` through `sqd`

For the record, since an earlier draft of this file asked for it and
`docs/rqutils-precond-request.md` is still in our tree. We had observed that `precond` was reachable on
`ground_locg` but not through `sqd()`, which is the only path `spinchain` uses, and were about to ask
for the same one-line forward that `prefilter` received.

**Withdrawn** — you have since told us the preconditioner was measured useless and removed. We have not
re-measured that and are not disputing it; the 1.79x in `docs/rqutils-precond-request.md` was iteration
count on `scipy`'s `lobpcg`, which is exactly the kind of proxy that §4 of your last reply showed can
mislead about the quantity that matters. Note the removal is **not** in the revision we have installed
(`4b7be94` still carries `precond` on `ground_locg`), so we will see it arrive with the next lockfile
bump; no action needed. We will drop `docs/rqutils-precond-request.md` or mark it withdrawn on our side.
