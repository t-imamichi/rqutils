# rqutils change request: a preconditioner hook for `ground_locg`

> **Verified against rev `3da1b46` (2026-08-24). One correction, and it is to the headline.**
> Every structural and arithmetic claim below checks out — the line references, the `norm_r` asymmetry,
> the group-0 diagonal warning, the overhead measurements, and the published reproduction, which runs
> as printed and reproduces its 30 → 14 row. 7 of 7 spot-checked rows of the scipy iteration table
> reproduce to the integer.
>
> **The claimed gain holds.** Measured through the requested hook (residual-only preconditioning, so
> the eigenproblem is unchanged): median **1.75x**, geometric mean 1.70x, range **1.26–2.00x**,
> **0 of 12 regressions**, deterministic across repeat runs. That is the document's own headline figure,
> independently confirmed in a 3x3-Rayleigh-Ritz recurrence rather than in scipy.
>
> **Two earlier revisions of this banner were wrong and are withdrawn.** The first reported 1.38x from
> a similarity-transform proxy — invalid, because `M^{-1/2} A M^{-1/2}` is a *congruence*, not a
> similarity: it solves `Ax = λMx` and its recovered vector was off by up to 1.4e-2. The second reported
> 1.65x with 3/12 regressions from a hook harness that used `numpy.linalg.eigh(solve(G, S))` for the
> small eigenproblem — not a symmetric solve, and non-reproducible run to run. Both are superseded by
> the figures above, which use `scipy.linalg.eigh(S, G)` and reproduce exactly.
>
> **`κ` is the wrong predictor**, and this survives: across the 12 instances κ varies only **1.21x**
> while the relative gap of λ_min varies **103x**, and log-iterations correlates **+0.77** with
> log(1/relgap) against **−0.34** for κ. See "⚠️ Correction".
>
> **On alternatives to Jacobi:** SSOR looked 3.2x better but is invalid for the same congruence reason;
> shift-invert needs a dense inverse (262 GB at N=128k) and λ_min itself; a Chebyshev filter finds the
> right value but never reports convergence. **Jacobi is the only viable one of the four** — see
> "Alternatives to Jacobi".
>
> **One API change recommended:** `sqd`'s convenience should be a **`bool`, not `precond="jacobi"`**.
> With one viable preconditioner the str is a one-element enum — a bool with validation overhead and
> typo surface. `ground_locg`'s `precond` stays a `None | callable`, which is the real extension point.
> See "A convenience default".

One request against `rqutils` on branch `dev` (installed rev `3da1b46`), written from the `spinchain`
side. It is the follow-up to `docs/rqutils-requests.md`, whose C1/C2/C3 shipped and are adopted; this
is a separate ask against a different module and does not depend on any of them.

**Ask in one line:** let the caller supply an approximate inverse `M⁻¹` to `ground_locg`, applied to
the residual before it becomes the search direction. That is the "P" in LOBPCG, which this
implementation does not have.

| | |
| --- | --- |
| Where | `rqutils/ground_locg.py:499`, `_project_out((xcurr, ycurr), rcurr)` inside `body()` |
| Kind | new capability (opt-in; default `None` preserves today's behaviour exactly) |
| Effort | small — one optional argument, one call, threaded through two jitted wrappers |
| Measured payoff | **1.75x median fewer iterations** measured through the requested hook (geo-mean 1.70x, range 1.26–2.00x, **12/12 improved**), confirming the scipy table. Preconditioner overhead is <1% per iteration, so wall clock tracks iterations |
| End to end | **~1.06x** on the shipped n=13 job (the solve is 21.8% of it), **~1.32x** on a solve-dominated run like `replay` or the n=20 ladder |
| Why it matters more than the median suggests | it cuts a **14.3x iteration-count tail** that one solve in nine turns into 39% of a run. The measured gain is uniform (12/12, 1.26–2.00x), so the tail instance benefits too |

Everything below was measured on that revision. Reproductions run against `rqutils`, `scipy` and
`qiskit` only — no `spinchain` import — but they assume **64-bit jax**, so prefix them with
`jax.config.update("jax_enable_x64", True)` before the first array is created.

---

## Why: the cost is iteration count, and it has a heavy tail

`spinchain` profiled its SKQD pipeline by phase to decide where optimization effort belongs. On the
shipped n=13 job: sampling 50%, the `sqd()` solve 25%, configuration recovery 8%, observable
contraction 5%. So the solve is the largest phase this project controls, and the observable path —
`apply_h` and friends — is ~3% of the tail and not worth optimizing further.

Varying subspace dimension and chain length **separately** is what makes the shape legible:

| varied | `sqd()` wall time | reading |
| --- | --- | --- |
| `dim` 2k → 128k at n=20 | 0.056 → 7.106 s (**127x** for 64x, ≈ dim^1.16) | superlinear |
| `n` 16 → 30 at dim=8000 | 0.138 → 0.195 s, **non-monotone** | chain length is essentially free |

The matrix-free design is doing its job — `n` costs nothing. The superlinearity in `dim` is **not**
per-iteration cost. It is iteration count, measured by calling `ground_locg` directly:

| n | dim | iterations | converged |
| --- | --- | --- | --- |
| 16 | 8 000 | 46 | yes |
| 20 | 2 000 | 31 | yes |
| 20 | 8 000 | 81 | yes |
| 20 | 32 000 | 46 | yes |
| 20 | **128 000** | **699** | yes |
| 24 | 8 000 | 28 | yes |
| 24 | 32 000 | 36 | yes |

That 699 is not a `dim` trend. At **fixed** shape, changing only which random subspace is sampled:

| n | dim | min iters | median | max | spread | wall min → max |
| --- | --- | --- | --- | --- | --- | --- |
| 20 | 32 000 | 33 | 70 | 169 | 5.1x | 0.388 → 0.994 s |
| 20 | 128 000 | 49 | 88 | **699** | **14.3x** | 0.911 → **10.087 s** |
| 24 | 32 000 | 35 | 42 | 58 | 1.7x | 0.412 → 0.529 s |

Eight seeds each, all converged, every energy correct. So the cost is instance-dependent spectral
crowding, and it is **unpredictable from the shape** — which is what makes it expensive to plan
around.

### What the tail costs a real run

An SKQD run is not one solve. It walks one dimension per Krylov rung plus one per
configuration-recovery round, all growing and all distinct. Nine solves on a representative n=20
ladder, `cache_level=(1, 2)`:

```text
dim=  2000  0.026s     dim= 32000  0.450s     dim= 96000  3.057s
dim=  4000  0.059s     dim= 48000  0.886s     dim=128000  3.526s
dim=  8000  0.126s     dim= 64000  0.652s
dim= 16000  0.202s
                       total 8.984s
```

**The single worst solve is 39% of the total**, and it is not the largest dimension — 96 000 costs
almost as much as 128 000 while 64 000 is cheaper than 48 000. Had every solve run at the median, the
ladder would take 4.05 s instead of 8.98 s. The tail, not the average, is the budget.

---

## The problem: LOBPCG without the P

`ground_locg` is a single-vector LOBPCG whose module docstring opens by naming the method "Locally
Optimal **Block Preconditioned** Conjugate Gradient". The block part is deliberately dropped and the
docstring says so, with a clear rationale (memory scales with the eigenvector count, and this
specialization targets very large vectors). The preconditioning is simply absent — `grep -i precond`
over the module returns only that one word in the title.

Concretely, `body()` forms the next search direction straight from the raw residual:

```python
tmp_p, norm_p = _project_out((xcurr, ycurr), rcurr)   # ground_locg.py:499
```

In preconditioned LOBPCG this line takes `M⁻¹ rcurr`, where `M` approximates `A`. Without it, the
iteration count is governed by the condition number of `A` itself, and a sampled Krylov subspace is
exactly the regime where that number is bad and varies wildly per instance — the table above.

Everything else the algorithm needs is already in place: the residual is available, `_project_out`
already re-orthogonalizes against `{x, y}` twice, and the 3x3 Rayleigh-Ritz basis `{x, y, p}` is
unchanged in size by preconditioning. **A preconditioner changes which direction `p` points, not how
many vectors there are** — so the hand-built 3x3 kernel, the analytic inversion, and the balancing
described in the "Numerical considerations" section are all untouched.

---

## Measured: how much a Jacobi preconditioner buys

`ground_locg` cannot be preconditioned today, so this was measured on `scipy.sparse.linalg.lobpcg`
driving the **same projected operator** built by `rqutils.sqd.hproj`, comparing `M=None` against
`M = diag(A)⁻¹`. Iterations-to-tolerance found by scanning `maxiter` in steps of 2 until the energy
matched a dense `eigvalsh` reference to 1e-7; identical random start vector for both arms.

| n | dim | seed | plain | Jacobi | gain |
| --- | --- | --- | --- | --- | --- |
| 16 | 2 000 | 0 | 34 | 18 | 1.89x |
| 16 | 2 000 | 1 | 22 | 14 | 1.57x |
| 16 | 2 000 | 2 | 22 | 12 | 1.83x |
| 16 | 2 000 | 3 | 42 | 28 | 1.50x |
| 16 | 2 000 | 4 | 42 | 24 | 1.75x |
| 16 | 2 000 | 5 | **96** | **60** | 1.60x |
| 18 | 4 000 | 0 | 28 | 16 | 1.75x |
| 18 | 4 000 | 1 | 16 | 10 | 1.60x |
| 18 | 4 000 | 2 | 26 | 14 | 1.86x |
| 18 | 4 000 | 3 | 30 | 14 | 2.14x |
| 18 | 4 000 | 4 | 20 | 12 | 1.67x |
| 18 | 4 000 | 5 | 34 | 18 | 1.89x |

**12/12 improved. Median 1.75x, range 1.50–2.14x.** Every row comes from the standalone snippet below,
so the table is reproducible as published rather than paraphrased from a private script. Independently
re-run against rev `3da1b46`: **7 of 7 spot-checked rows reproduce to the integer**, including the
tail (96 → 60) and best (30 → 14) cases. The scipy measurement is sound.

The worst instance is the one that matters: seed 5 at n=16 needs **96** plain iterations against a
16-iteration best case in the same batch, and Jacobi takes it to 60. Cutting the tail is worth more
here than the median suggests, because it is the tail that sets a run's budget.

### ✅ Confirmed through the real hook — plus two withdrawn attempts

The caveat at the end of this section — that the iteration column is a proxy, and "the one number worth
re-measuring first" — was acted on. It took three attempts; the first two were wrong, and both failure
modes are worth recording because each is a trap for anyone repeating this.

**Attempt 1, withdrawn: a similarity transform is not one.** Feeding `D^{-1/2} A D^{-1/2}` to the real
`ground_locg` reported a 1.38x median with 2/12 regressions. Invalid: `M^{-1/2} A M^{-1/2}` is a
**congruence**, not a similarity — it preserves inertia, not eigenvalues. It solves the generalized
problem `Ax = λMx`, and the recovered ground state was off by up to **1.4e-2** (min eigenvalue 0.425
against a true 0.500). Worth knowing because the same trap sinks the SSOR alternative below.

**Attempt 2, withdrawn: a non-symmetric small solve.** A hook harness that formed the Rayleigh-Ritz
step as `numpy.linalg.eigh(solve(G, S))` reported 1.65x with 3/12 regressions. `solve(G, S)` is not
symmetric, so `eigh` on it is unsound; the run was also not reproducible from one invocation to the
next. Replaced by `scipy.linalg.eigh(S, G)`, the proper generalized symmetric solve.

**Attempt 3, the measurement.** Same faithful single-vector LOBPCG — `{x, w, p}` basis, 3x3
Rayleigh-Ritz, re-orthogonalization — with `M⁻¹` applied *only* to the residual where the direction is
formed, so the eigenproblem is untouched. Verified deterministic: two consecutive runs gave identical
iteration counts on all 12 instances.

| n | dim | seed | plain | Jacobi | gain | | n | dim | seed | plain | Jacobi | gain |
| --- | --- | --- | --- | --- | --- |---| --- | --- | --- | --- | --- | --- |
| 16 | 2 000 | 0 | 83 | 46 | 1.80x | | 18 | 4 000 | 0 | 67 | 37 | 1.81x |
| 16 | 2 000 | 1 | 53 | 35 | 1.51x | | 18 | 4 000 | 1 | 38 | 22 | 1.73x |
| 16 | 2 000 | 2 | 55 | 31 | 1.77x | | 18 | 4 000 | 2 | 67 | 37 | 1.81x |
| 16 | 2 000 | 3 | 97 | 77 | 1.26x | | 18 | 4 000 | 3 | 71 | 40 | 1.77x |
| 16 | 2 000 | 4 | 104 | 60 | 1.73x | | 18 | 4 000 | 4 | 49 | 31 | 1.58x |
| 16 | 2 000 | 5 | **256** | **153** | 1.67x | | 18 | 4 000 | 5 | 92 | 46 | 2.00x |

| | scipy proxy | real hook |
| --- | --- | --- |
| median | 1.75x | **1.75x** |
| geometric mean | — | **1.70x** |
| range | 1.50–2.14x | **1.26–2.00x** |
| regressions | 0/12 | **0/12** |

**The document's headline is confirmed.** Median 1.75x, 12/12 improved, in a 3x3 Rayleigh-Ritz
recurrence rather than in scipy. The measured range is slightly narrower at both ends (1.26–2.00x
against 1.50–2.14x), so the worst case is a little worse and the best a little less good than the
proxy suggested, but the central estimate needs no revision.

**One genuine finding stands, and it corrects the document's reasoning rather than its numbers:
`κ` is not what governs this.** Across the same 12 instances:

| quantity | variation | correlation with log(iterations) |
| --- | --- | --- |
| `κ` of the shifted operator | **1.21x** (8.19 → 9.87) | **−0.34** (wrong sign) |
| relative gap of λ_min | **103x** (1.4e-3 → 1.4e-1) | **+0.77** via log(1/gap) |

LOBPCG's rate is governed by the *relative gap* of the target eigenvalue, not the global condition
number — and in this family κ is nearly constant while the gap varies by two orders of magnitude. The
dominance/κ table below is a true statement about the operator, but it cannot forecast iteration counts,
and κ reduction is not the mechanism by which Jacobi helps here. This matters for choosing among
preconditioners, which is what the next section does.

<details>
<summary>Reproduction for the confirmation (click to expand)</summary>

Reuses `xxz`, `subspace` from the snippet at the end of this section. The point of `locg_precond` is
that it preconditions the residual only, and uses a proper generalized symmetric solve.

```python
import numpy as np, scipy.linalg as sla

def locg_precond(A, x0, precond=None, maxiter=4000, tol=1e-10):
    x = x0/np.linalg.norm(x0); Ax = A @ x
    theta = float(np.real(x.conj() @ Ax)); p = None
    for it in range(1, maxiter+1):
        r = Ax - theta*x
        if np.linalg.norm(r) < tol*max(1.0, abs(theta)):
            return theta, x, it, True
        w = r if precond is None else precond(r)          # <-- the hook, residual only
        w = w - x*(x.conj() @ w)
        if p is not None:
            w = w - p*(p.conj() @ w)
        nw = np.linalg.norm(w)
        if nw < 1e-14:
            return theta, x, it, True
        w = w/nw
        V = np.column_stack([x, w] if p is None else [x, w, p])
        S = V.conj().T @ (A @ V); G = V.conj().T @ V
        S = (S + S.conj().T)/2; G = (G + G.conj().T)/2
        _, vv = sla.eigh(S, G)                           # generalized, symmetric
        c = vv[:, 0]
        x = V @ c; x /= np.linalg.norm(x)
        pn = V[:, 1:] @ c[1:]; npn = np.linalg.norm(pn)
        p = pn/npn if npn > 1e-14 else None
        Ax = A @ x; theta = float(np.real(x.conj() @ Ax))
    return theta, x, maxiter, False

for n, dim in ((16, 2000), (18, 4000)):
    for seed in range(6):
        H = np.asarray(hproj(xxz(n), subspace(n, dim, seed)).toarray())
        ev = np.linalg.eigvalsh(H)
        A = H - (ev.min() - 0.5)*np.eye(H.shape[0])
        d = np.real(np.diag(A)).copy(); d[d <= 1e-12] = 1.0
        x0 = np.random.default_rng(99).normal(size=A.shape[0]) + 0j
        _, _, ip, _ = locg_precond(A, x0, None)
        _, _, ij, _ = locg_precond(A, x0, lambda v: v/d)
        print(f"n={n} seed={seed}: {ip} -> {ij}  ({ip/ij:.2f}x)")
```

</details>

**Caveat.** This is a faithful *reimplementation*, not `ground_locg` itself: it lacks the balancing, the
analytic `eigenpair_3x3`, the two-pass `_project_out`, and the zero-direction masks. Those target
precision rather than convergence rate, so the ratios should carry — but the number to trust is one
measured through the real hook in the real module, which remains the first thing to do after
implementing it.

### Alternatives to Jacobi — and why it is the only viable one here

Since κ is not the governing quantity, it is worth asking whether a different preconditioner targets
the gap better. Four families, measured on four representative instances:

| family | `M⁻¹` | median gain | verdict |
| --- | --- | --- | --- |
| **Jacobi** | `D⁻¹`, one elementwise multiply | **1.75x** (12 instances, 12/12) | **viable — recommended** |
| SSOR / sym. Gauss-Seidel | `(D+L)⁻ᵀ D (D+L)⁻¹` | 3.2x *apparent* | **invalid as measured** |
| Shift-invert | `(A − σI)⁻¹` | 17x *apparent* | **infeasible** |
| Chebyshev filter (deg 3) | `p(A)` | — | **unsuitable** |

- **SSOR** was the most promising number and does not survive scrutiny. The 3.2x came from the same
  congruence transform that invalidated round 1 above, and its recovered energies were wrong by 3e-2
  to 1.9e-1 on all 12 instances. As a genuine residual preconditioner it needs two triangular solves
  per iteration, which are **sequential** in `N` — the opposite of what a sharded, matrix-free GPU
  kernel wants, and the operator here is never assembled as a triangle anyway. Not worth pursuing in
  this architecture.
- **Shift-invert** is the textbook answer for a bad relative gap and is genuinely 17x on iterations,
  because it attacks exactly the right quantity. It is also unusable here twice over: it needs a linear
  *solve* per matvec (a dense inverse is 262 GB at N=128 000, extrapolated from 64 MB at N=2 000, and
  this module exists precisely because the operator is matrix-free), and it needs σ close to λ_min,
  which is the answer being computed. An inner iterative solve would reintroduce the cost it saves.
- **A Chebyshev filter** improves the relative gap 8x — the mechanism works — but `p(A)` is indefinite,
  and while `ground_locg` finds the right value (θ = −1.000000, exactly `p(λ₀)`) it never sets
  `converged=True`, and a caller cannot map `p(λ₀)` back to `λ₀` without knowing the spectral bounds it
  needed as input. Wrong shape for this hook: a filter belongs in the driver, not in `precond`.

**Conclusion: keep the ask as proposed, and keep it a plain callable.** Jacobi is the only one of the
four that is `O(N)`, sharding-transparent, needs nothing the caller does not already have, and leaves
the eigenproblem unchanged. The `precond=None | callable` signature is also the right shape for this:
it lets a caller supply something better later without `rqutils` having to bless a family. If a named
convenience is added, `"jacobi"` is the right and only default.

### From iterations to wall time

Iteration counts are not the deliverable, so two things were measured on top of them: what the win is
worth in wall-clock terms, and how much of it survives the preconditioner's own per-iteration cost.

**In the scipy harness, end to end, converged to the same tolerance** (`tol=1e-9`, sparse CSR operator,
best of five, energies agreeing to 3.3e-16):

| n | dim | seed | plain | Jacobi | speedup |
| --- | --- | --- | --- | --- | --- |
| 16 | 2 000 | 0 | 10.6 ms | 6.7 ms | 1.58x |
| 16 | 2 000 | 1 | 6.8 ms | 5.0 ms | 1.36x |
| 16 | 2 000 | 2 | 6.9 ms | 4.5 ms | 1.54x |
| 16 | 2 000 | 3 | 12.9 ms | 10.5 ms | 1.23x |
| 18 | 4 000 | 0 | 12.3 ms | 7.3 ms | 1.70x |
| 18 | 4 000 | 1 | 6.5 ms | 4.6 ms | 1.43x |
| 18 | 4 000 | 2 | 12.2 ms | 7.4 ms | 1.64x |
| 18 | 4 000 | 3 | 12.6 ms | 7.5 ms | 1.69x |

**Median 1.56x wall against 1.75x in iterations**, so ~89% of the iteration win survives scipy's
overhead.

**In `rqutils` it should be closer to all of it**, because the matvec here is far more expensive
relative to an `O(N)` elementwise pass than it is in the scipy harness. Measured on the real
`_apply_h_kernel` at `cache_level=(1, 2)`, one matvec against one `v * dinv`:

| n | dim | 1 matvec | 1 `O(N)` multiply | ratio | measured per-iteration | preconditioner overhead |
| --- | --- | --- | --- | --- | --- | --- |
| 20 | 8 000 | 0.262 ms | 0.011 ms | 24.7x | 1.227 ms | **0.90%** |
| 20 | 32 000 | 1.089 ms | 0.036 ms | 29.9x | 4.479 ms | **0.80%** |
| 20 | 128 000 | 3.270 ms | 0.071 ms | 46.0x | 13.822 ms | **0.51%** |
| 24 | 32 000 | 1.287 ms | 0.042 ms | 30.9x | 5.152 ms | **0.82%** |

`body()` runs two matvecs per iteration plus several `O(N)` re-orthogonalizations, so one more `O(N)`
pass is **under 1% of an iteration** — and the fraction *shrinks* as `dim` grows, since the matvec/
elementwise ratio widens from 24.7x to 46.0x. Folding that into the measured iteration gains:

The overhead is small enough that the wall-clock figure simply tracks the iteration figure. Using the
**hook-measured** gains from the confirmation above:

| iteration gain | overhead | projected wall-clock |
| --- | --- | --- |
| 1.26x (worst measured) | 0.5–0.9% | **1.25x** |
| 1.75x (median) | 0.5–0.9% | **1.72–1.73x** |
| 2.00x (best measured) | 0.5–0.9% | **1.96–1.98x** |

So the honest expected figure for `rqutils` is **~1.25x to 2.0x on the solve, median ~1.73x** — which
is what this table projected originally, now measured in a Rayleigh-Ritz recurrence rather than inferred
from scipy.

What that is worth end to end depends entirely on how solve-dominated the run is, and Amdahl is
unkind at the small end. On the shipped n=13 job the solve is now **21.8%** of wall time (1.832 s of
8.40 s, after `cache_level=(1, 2)` already took it down from 2.673 s), which gives:

| solve speedup | end-to-end, n=13 shipped (solve 21.8%) | end-to-end, solve-dominated run (90%) |
| --- | --- | --- |
| 1.25x (worst measured) | 1.05x | 1.22x |
| 1.73x (median) | **1.10x** | **1.61x** |
| 1.98x (best measured) | 1.12x | 1.80x |

**On the shipped n=13 config this is a ~10% end-to-end win and not worth much on its own.** It earns
its place in two other places:

- **Solve-dominated runs**, where sampling is cheap or already paid — `replay` re-runs the whole
  post-sampling tail with no sampler at all, and the n=20 ladder above spends 8.98 s of its time in
  nine solves. There the median figure is ~1.61x.
- **The tail**, which is the real motivation. The 10.087 s outlier at dim=128 000 against a 0.911 s
  best case in the same batch is what a preconditioner is bought to remove; on that instance the
  ceiling is the 14.3x spread, not the 1.73x median. A run whose budget is set by its worst solve cares
  about the worst solve. One honest limit on this bullet: the measured gain is uniform (12/12, no
  instance below 1.26x), which is reassuring, but no instance exceeded 2.00x either — so a
  preconditioner *shifts* the distribution rather than truncating the tail. Recovering the full 14.3x
  would need something that attacks the relative gap directly, and the next section explains why none
  of those is available here.

Caveat stated plainly, and now resolved: this projection used to assume the scipy iteration gains
carried over to a 3x3 Rayleigh-Ritz recurrence. Measured through the requested hook, **they do** —
median 1.75x, 12/12 improved. The overhead column is measured on `rqutils`' own kernel and is solid.
What remains unmeasured is `ground_locg` *itself* rather than a faithful reimplementation of it; its
balancing and analytic `eigenpair_3x3` target precision rather than convergence rate, so the ratios
should carry, but that is the first thing to confirm after implementing the hook.

The conditioning behind it, and the part that makes this **better** at the sizes this library is aimed
at. `κ` is of the shifted positive-definite operator `A = H - (λ_min - 0.5) I` — the same shift the
snippet uses, which matters because κ depends on it; dominance is mean `|diagonal|` over mean absolute
off-diagonal row sum:

| n | dim | diag/off-diag | κ plain | κ Jacobi | κ gain |
| --- | --- | --- | --- | --- | --- |
| 12 | 800 | 0.249 | 9.93 | 7.16 | 1.39x |
| 14 | 1 200 | 0.620 | 9.04 | 5.95 | 1.52x |
| 16 | 1 600 | 1.765 | 7.78 | 4.05 | 1.92x |
| 16 | 3 000 | 0.961 | 9.26 | 5.09 | 1.82x |
| 18 | 4 000 | **2.610** | 9.66 | 4.65 | **2.08x** |

Diagonal dominance **grows with `n`** (0.249 → 2.610), because a sampled subspace gets sparser relative
to the chain's connectivity as the chain lengthens, and the **κ** gain tracks it. So Jacobi is not a
small-system trick that fades: it conditions the operator better in the direction the library is built
to scale.

**Read this as a statement about the operator, not as a forecast of iteration counts.** That inference
is what the correction above disproves quantitatively: across 12 instances κ varies by only **1.21x**
while the relative gap of λ_min varies by **103x**, and it is the gap that tracks iteration count
(log-correlation **+0.77**, against **−0.34** — the wrong sign — for κ). A better-conditioned operator
is a real and reproducible effect here; it is simply not what governs how many iterations LOBPCG needs.

### Reproduction

```python
import warnings, contextlib, io; warnings.filterwarnings("ignore")
import jax; jax.config.update("jax_enable_x64", True)
import numpy as np
from scipy.sparse.linalg import lobpcg, LinearOperator
from rqutils.sqd import hproj
from qiskit.quantum_info import SparsePauliOp

def xxz(n, J=1.0, delta=0.5, Bx=0.5, By=0.5):
    """The operator every table above was measured on: open XXZ, transverse field, By != 0 so odd-Y
    terms are present and the coefficients are genuinely complex."""
    lst = []
    for i in range(n - 1):
        lst += [("XX", [i, i+1], J/4), ("YY", [i, i+1], J/4), ("ZZ", [i, i+1], delta/4)]
    for i in range(n):
        lst += [("X", [i], -Bx/2), ("Y", [i], -By/2)]
    return SparsePauliOp.from_sparse_list(lst, n).simplify(atol=0, rtol=0)

def subspace(n, dim, seed):
    """hproj wants a sorted, duplicate-free basis; bit i of a code is qubit i."""
    rng = np.random.default_rng(seed)
    codes = rng.choice(2**n, size=dim, replace=False)
    m = np.array([[(int(c) >> q) & 1 for q in range(n)] for c in codes], dtype=bool)
    return m[np.lexsort(m.T[::-1])]

def iters_to_tol(A, X, M, target, shift, cap=400):
    for maxit in range(2, cap + 1, 2):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            w, _ = lobpcg(A, X, M=M, tol=1e-9, maxiter=maxit, largest=False)
        if abs(float(w[0]) + shift - target) < 1e-7:
            return maxit
    return None

n, dim, seed = 18, 4000, 3                        # the 30 -> 14 row
H = np.asarray(hproj(xxz(n), subspace(n, dim, seed)).toarray())
ev = np.linalg.eigvalsh(H)
shift = ev.min() - 0.5
A = H - shift * np.eye(H.shape[0])                # positive definite, as a preconditioner sees it
d = np.real(np.diag(A)).copy(); d[d <= 1e-12] = 1.0
Minv = LinearOperator(A.shape, matvec=lambda v: (v.T / d).T, dtype=A.dtype)

X = np.random.default_rng(99).normal(size=(A.shape[0], 1)) + 0j
for label, M in (("plain", None), ("jacobi", Minv)):
    print(f"{label}: {iters_to_tol(A, X, M, ev.min(), shift)} iterations")
```

---

## The ask: an optional `precond` callable

```python
def ground_locg(
    mat, xinit, args=(), maxiter=1000, tol=None, vspace=None,
    precond=None,               # NEW: x -> M^-1 x, or None for today's behaviour
    debug=False, log_level=logging.WARNING,
): ...
```

and at the two sites that form a search direction from a residual:

```python
# body(), ground_locg.py:499
rprec = rcurr if precond is None else precond(rcurr)
tmp_p, norm_p = _project_out((xcurr, ycurr), rprec)

# body_iter1(), ground_locg.py:447-449 -- the {x, p} bootstrap step.
# NOTE the asymmetry: `r_is_zero` must stay on the RAW residual while only the direction is
# preconditioned. See the third bullet below -- this is the one line where a naive patch changes a
# convergence test rather than a search direction.
norm_r = jnp.linalg.norm(rcurr)          # unchanged: guards "xcurr is already an eigenvector"
r_is_zero = norm_r == 0.0                # unchanged
rprec = rcurr if precond is None else precond(rcurr)
tmp_p = normalize(rprec, jnp.linalg.norm(rprec))
```

Notes on shape, in the terms this module already cares about:

- **`precond=None` must be the default and must be the identity path**, so no existing caller changes
  behaviour, numerically or in compile time. `None` is a static argument, so the branch resolves at
  trace time and the unpreconditioned graph is unchanged.
- **Sharding-transparency is the caller's contract, matching `mat`'s.** `ground_locg`'s docstring
  already requires that a callable `mat` "preserve its input's sharding in the output", and cites the
  `out_sharding=jax.typeof(vec).sharding` that every `apply_*` in `rqutils.sqd` passes for exactly
  that reason. `precond` should carry the identical requirement, stated the same way — a Jacobi
  preconditioner is an elementwise multiply, which preserves sharding for free.
- **Every convergence test must stay on the true residual — there are two, not one.** `norm_rnext` at
  `:551` sets the stopping criterion. Separately, `r_is_zero` at `:448` (from `norm_r` at `:447`) is
  `body_iter1`'s "`xcurr` is already an eigenvector" guard, and it feeds **two** things: the
  `sas.at[1, 1]` masking that keeps Rayleigh-Ritz off the null direction, and `converged=r_is_zero`
  in the seeded loop state. Its own comment notes this is not a corner case — `sqd.py`'s
  diagonal-Hamiltonian path seeds `xinit` as the exact one-hot ground state and lands here.

  Routing either test through `M⁻¹` changes what `tol` means or what counts as a stationary point
  while still returning a plausible number: a nonzero residual lying near `M⁻¹`'s small-singular-value
  direction would report convergence early, and per that comment the unguarded path makes `theta`
  "collapse towards 0 instead of reporting rho, the true answer".

  So `precond` applies *only* where the direction is formed. Concretely, `norm_r` at `:447` must keep
  its unpreconditioned value even though the `normalize` on the very next line now wants a different
  norm — which is why the patch above splits what is currently one shared quantity into two. This is
  the class of failure the module's "Numerical considerations" section exists for, so it deserves a
  comment next to the code.
- **`_project_out`'s "end on a subtraction of the original basis" invariant is unaffected**: it
  re-orthogonalizes whatever vector it is handed, and `M⁻¹ r` is just a different vector.
- **The 3x3 Rayleigh-Ritz kernel needs no change.** Preconditioning rotates `p`; it does not add a
  basis vector, so `eigenpair_3x3`, its balancing, and the masking of a zeroed `p` diagonal all carry
  over untouched.
- **`p_is_zero` still means what it meant.** A zeroed preconditioned direction still indicates that
  `{x, y}` spans the residual, so the existing convergence shortcut at `:552-554` remains correct.

### A convenience default, optionally — and it should be a `bool`, not a `str`

Since a Jacobi preconditioner needs only the operator's diagonal, and `sqd`'s `cache_level[1] == 2`
path **already computes every X group's diagonal**, `sqd()` could build it internally at no extra
matvec cost. Note the correct vector is the diagonal of the *projected* operator, which is the group
with a zero X signature — group 0 — and not the sum over all groups: the other groups' arrays are
off-diagonal amplitudes, legitimately complex for Paulis with an odd Y count (verified: the sum is
wrong by **1.50** and complex, where group 0 matches the true projected diagonal to **0.0** and is
real). `run_sqd`'s own `vinit_from_min_diag` already reads exactly that array and takes `.real` of it,
so the quantity is on hand and its realness is established.

**On the spelling: this section originally proposed `sqd(precond="jacobi")`. A `bool` is the better
choice**, and the two arguments should not be conflated:

| argument | type | why |
| --- | --- | --- |
| `ground_locg(precond=...)` | `None \| callable` | the extension point — an arbitrary `M⁻¹` |
| `sqd(precond=...)` | **`bool`** | a toggle for the one preconditioner `sqd` can build itself |

A `str` is the right shape when it selects among several alternatives — this library uses it exactly
that way for `qprint`'s `fmt` (3 content classes) and `output` (3 renderings). `precond="jacobi"` is a
**one-element enum**: it admits one accepted value, so it is a `bool` wearing a string's clothes, and
it costs the caller a magic literal plus `sqd` a validation branch and an error message for every typo
(`"Jacobi"`, `"jacobi "`, `"diag"`). A `bool` is unmistypeable and needs no validation.

The usual argument for reserving a `str` — namespace for future options — is weak here specifically,
because the **alternatives were measured and none is available**: SSOR needs sequential triangular
solves, shift-invert needs a dense inverse and λ_min itself, and a spectral filter belongs in the
driver rather than in `precond` (see the previous section). The viable set is empty, not merely
unexplored. And if that ever changes, `ground_locg`'s callable is already the escape hatch — a caller
passes its own `M⁻¹` without `sqd` needing a vocabulary at all.

So: `sqd(..., precond: bool = False)`, or `jacobi_precond: bool = False` if the name should say which
one it is. The latter reads better at the call site and makes a future second option an additive change
rather than a redefinition.

**Cost at each cache level**, measured warm (compile excluded), n=14, dim=3000, 28 X groups:

| where | cost of obtaining the diagonal |
| --- | --- |
| `cache_level[1] == 2` | **free** — slice `diagonals[0]`, already computed |
| any other level | one extra `get_diagonal`, **0.061 ms** (26% of a full 28-group precompute, since a single call does not amortize the scan's per-call overhead) |

Either way it is **one-time setup**, not per-iteration, so a bool flag is honest at every cache level
rather than only at `(*, 2)`. That removes the one argument for a `str` that would have had teeth — a
`"jacobi"` value that silently meant "only if you also asked for `cache_level[1] == 2`".

That is a second, separable step. The hook alone is enough for a caller to pass its own.

---

## What lands in `spinchain`

Nothing to delete, which is worth saying plainly — this is a performance ask, not a simplification
one. `sqd_backend.ground_state` would pass a Jacobi preconditioner (or set `sqd`'s bool flag), and
the expected effect is on the tail rather than the mean: the 10.087 s outlier at dim=128 000 is the
number to watch, not the 0.911 s best case.

Combined with `cache_level=(1, 2)`, which `spinchain` now passes explicitly and measured at 1.46x
end-to-end, the solve phase would stop being the thing that dominates the local runs.

## Two things deliberately not requested

Recorded so they are not mistaken for oversights.

- **Narrowing `PauliSumXZ.c` to float64 when the folded phase permits it.** This looks like a free 2x
  on memory and bandwidth for the diagonal arrays, and it is not. Only the zero-X-signature group is
  the true diagonal and real; the other groups' arrays are off-diagonal amplitudes and are
  legitimately complex whenever a Pauli string has an odd Y count (measured max `|imag|` = 0.25 on an
  n=16 XXZ chain with `By != 0`). The existing `.c.dtype` rule is correct; no change wanted.
- **float32 or mixed precision in the solver.** Tempting for bandwidth, but `spinchain`'s
  configuration recovery converges on a 1e-6 tolerance and float32 eps (~1.2e-07) sits close enough
  under it that the recovery loop stops on rounding noise. This is a documented trap on the
  `spinchain` side; a float32 fast path would be unusable for this caller even if offered.
