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
> **Status: the `ground_locg` hook is implemented and shipped. The `sqd` convenience is rejected.**
> The hook lands as `precond=None | callable` and measures **1.79x** median in the real module. The
> `sqd(jacobi_precond=...)` convenience was implemented on a branch and **discarded: it makes `sqd`
> about 3x slower.** Jacobi needs a positive-definite operator; `sqd` solves the *unshifted* `H`, which
> is indefinite (~50% of diagonal entries ≤ 0, λ_min ≈ −2.4). Every figure in this document was measured
> on the shifted `A = H − (λ_min − 0.5)I`, which `sqd` never builds. A caller that knows its own
> spectral range can shift and pass `precond` itself; `sqd` cannot do it on the caller's behalf. See
> "❌ A convenience default on `sqd`".

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
    """hproj wants a sorted, duplicate-free basis, with bit q of a code on qubit q.

    Note the `[::-1]`: `hproj`'s `states` columns are indexed by Pauli-string CHARACTER
    position, not by qubit number, so bit q belongs in column n-1-q. `xxz` above builds its
    operator with `SparsePauliOp.from_sparse_list`, which is indexed by qubit -- so pairing
    it with a `bit q -> column q` table silently projects onto the bit-reversed subspace.
    Measured: `Z` on qubit 0 over codes {0, 1} gives diag [1, 1] with the naive pairing
    (no dependence on qubit 0 at all) against the correct [1, -1]. See docs/gotchas.md
    item 2; the earlier version of this helper had the bug.
    """
    rng = np.random.default_rng(seed)
    codes = rng.choice(2**n, size=dim, replace=False)
    m = np.array([[(int(c) >> q) & 1 for q in range(n)][::-1] for c in codes], dtype=bool)
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

### ❌ A convenience default on `sqd` — implemented, measured, and **rejected**

**Do not build this.** It was implemented on a branch, measured, and discarded. The flag works
correctly and makes `sqd` roughly **3x slower**.

**Why.** Jacobi preconditioning requires a positive-definite operator, and `sqd` does not shift: it
solves the raw projected `H`, which is **indefinite**. Measured on XXZ subspaces:

| n | dim | diagonal range | entries ≤ 0 | λ_min |
| --- | --- | --- | --- | --- |
| 12 | 800 | [−1.1250, +1.1250] | **398/800 (50%)** | −2.379 |
| 14 | 1 500 | [−1.3750, +1.3750] | **745/1500 (50%)** | −2.252 |

Half the diagonal entries must be masked to 1.0 to avoid dividing by zero or by a negative, so `M⁻¹`
degenerates into an arbitrary half-scaling that *destroys* conditioning rather than improving it.
Measured through `ground_locg` on the unshifted operator: **0.28–0.36x, i.e. 3x worse, 8 of 8
instances**.

**This does not invalidate the 1.79x figure**, and the distinction is the whole point. Every gain in
this document was measured on `A = H − (λ_min − 0.5)I` — the shift the reproduction snippet performs
explicitly, and which the κ table's own caption names. `sqd` never constructs that operator. The hook
in `ground_locg` is fine; the *convenience* is what fails, because `sqd` has nothing to build a usable
`M⁻¹` from.

**Why an internal shift does not rescue it.** A tight shift needs a lower bound on λ_min, which is the
answer being computed. A bound from data `sqd` already has — the diagonal alone — was tried
(`σ = min(diag) − 2·max|diag|`): it restores correctness and positive-definiteness, and yields
**1.04–1.24x**. A crude bound over-shifts and flattens the spectrum, so most of the benefit is lost.
Gershgorin would give a tighter bound but needs row sums, which a matrix-free operator cannot supply
without `J` extra matvecs.

**So the caller keeps the responsibility, which is the right place for it.** A caller that knows its
own spectral range — `spinchain` does; it is what makes `hproj` + `eigvalsh` a viable reference at
these sizes — can shift, build `diag(A)⁻¹`, and pass it through `ground_locg`'s `precond`. That path is
measured at 1.79x. `sqd` cannot do it on the caller's behalf.

Recorded rather than deleted because the implementation was straightforward and the trap is not
obvious: the flag produces **correct energies at every `cache_level`** (verified against a dense
reference to 5.8e-15, including the degenerate no-diagonal-terms case), so nothing about its output
signals that it is a pessimization.

<details>
<summary>The original proposal, kept for the record (click to expand)</summary>

The section below argued for spelling the convenience as a `bool` rather than `precond="jacobi"`. That
reasoning still holds *if* such a flag were ever viable — and the shape argument is worth keeping,
because it applies to any future single-option flag on this module.

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

</details>

### ❌ Two XXZ-specific candidates for the *unshifted* operator — both measured and rejected

Follow-ups to the section above, asked as: *given 1D open-boundary XXZ **with a transverse field**
(so `Sz` is not conserved and sector decomposition is unavailable), is there a preconditioner that
helps the operator `sqd` actually builds?* Both cheap candidates are now closed. Measured through the
real `ground_locg` hook, `precond=None` against the candidate, same `x0`, `maxiter=4000`, `tol=1e-10`,
on the same 12-instance batch as the 1.79x figure (`(n,dim) = (16,2000), (18,4000)`, seeds 0-5).

**Candidate A — `M⁻¹ = |diag(H)|⁻¹`, no shift.** The appeal was that taking the absolute value gives a
positive-definite `M⁻¹` with no shift at all, sidestepping the reason the `sqd` flag failed. **Measured
0.30x median (0.16-0.88x), 12/12 regressions**, every energy verified against `eigvalsh` to 1e-6 and
every arm converged.

| n | seeds 0-5, `none` -> abs-diagonal |
| --- | --- |
| 16 | 44->183, 28->87, 29->100, 45->132, 54->193, 122->138 |
| 18 | 35->102, 19->91, 34->99, 37->233, 24->75, 46->226 |

This is *worse* than the masked Jacobi it was meant to improve on (0.28-0.36x), and the mechanism is
the point: `|diag|⁻¹` amplifies the components with the **smallest** `|diag|`, which for an indefinite
operator sit in the middle of the spectrum rather than near `λ_min`. It boosts exactly the directions
the eigensolver should be suppressing. Positive-definiteness of `M⁻¹` was never the binding
constraint -- correlation with `A⁻¹` was, and `|·|` destroys it. Note again that all 12 returned the
**correct energy**: same trap as the rejected flag, a silent pessimization with nothing in the output
to signal it.

**Candidate B — a shift from the analytic diagonal plus a structural row-sum bound.** `σ = min(diag) −
B` with `B = (n−1)·(J/4)·2 + n·(Bx+By)/2` from XXZ's structure rather than the assembled matrix.
**Rejected without an iteration measurement**, because the bound is disqualifying on its own:

| n | dim | `min(diag)` | structural `B` | true max off-diag row sum | over-shift |
| --- | --- | --- | --- | --- | --- |
| 10 | 300 | −0.8750 | 9.5000 | 4.6213 | **16.05x** |
| 12 | 800 | −1.1250 | 11.5000 | 3.7678 | **20.49x** |

Worse than the coefficient-sum bound already rejected at 4.14-8.14x, and degrading faster with `n`.
The cause is that a *sampled* subspace projects most couplings away: the structural bound assumes
every state couples through all `n−1` bonds and all `n` field terms, where the true max row sum is
~3x smaller. Even the unattainable bound computed from the assembled matrix (`sigma_true`, which a
matrix-free operator cannot have) still over-shifts **5.03-6.29x**. So the whole row-sum family is
structurally incapable of a tight shift on a sampled subspace, however the bound is obtained.

**Confirmed in passing:** the analytic diagonal `diag = Δ/4 · Σᵢ sᵢsᵢ₊₁` with `s = 1 − 2·bit` matches
the true projected diagonal to **0.000e+00** with the field on, at n=10 and n=12 -- the transverse
field is purely off-diagonal and contributes nothing. That holds and is `O(N·n)` matvec-free; it is
the *shift*, not the diagonal, that has no viable estimator here.

**What remains untried** is the third candidate: a two-level / deflation preconditioner exploiting the
near-block structure of a sampled 1D chain (measured 0.12-0.58% nonzero at n=12-14). It is
substantially more work than either of the above and remains speculative. Any candidate should still be
judged on whether it **opens the relative gap**, not on `κ` -- across these same 12 instances `κ`
varies 1.21x while the relative gap varies 103x, and log-iterations correlates +0.77 with `log(1/gap)`
against −0.34 (wrong sign) for `κ`.

### ❌ The product-state solver as a shift source — three routes, all closed

Asked as: *could a Jacobi preconditioner use an eigenvalue from the product-state solver
(`rqutils/product.py`, SCIP)?* It is a better idea than the two candidates above -- every previous
rejection traced back to having **no viable lower bound on `λ_min`**, and `solve_product` returns
SCIP's branch-and-bound `lower_bound`, which is *certified* rather than a norm inequality. It still
fails, for one root cause worth stating once:

> **`solve_product` optimizes over a manifold on the wrong side of `λ_min`.** Product states are a
> strict subset of Hilbert space, so its minimum is a variational **upper** bound on the ground
> energy. `lower_bound` certifies the *product-state* optimum, not the spectrum -- it is SCIP's own
> optimality gap (`eigval` and `lower_bound` agree to ~2e-4 at `tol=1e-4`), not a spectral bound.

| n | `λ_min(H_full)` | `prod_eigval` | `prod_lower_bound` | `lb ≤ λ_min`? |
| --- | --- | --- | --- | --- |
| 8 | −3.19678 | −2.41108 | −2.41130 | **no** |
| 10 | −4.00079 | −3.03639 | −3.03657 | **no** |
| 12 | −4.79110 | −3.66146 | −3.66182 | **no** |

**Route 1 — `σ = lower_bound − ε` against the projected operator. Rejected, with the worst possible
failure profile.** Against the *projected* `H` the bound initially appeared to hold, because a randomly
sampled subspace misses the entangled ground state badly enough that `λ_proj` lands above the product
optimum. It is a coincidence of bad subspaces, not a theorem. Sweeping subspace quality (`top-amp` =
the true ground state's largest-amplitude basis states) breaks it everywhere it matters:

| n | dim | subspace | `λ_proj` | `prod_lb` | valid? |
| --- | --- | --- | --- | --- | --- |
| 8 | 12 | random | −0.62500 | −2.41130 | ✅ |
| 8 | 64 | top-amp | −2.89506 | −2.41130 | ❌ |
| 8 | 128 | top-amp | −3.18222 | −2.41130 | ❌ |
| 12 | 204 | random | −1.52630 | −3.66182 | ✅ |
| 12 | 1024 | top-amp | −4.74143 | −3.66182 | ❌ |
| 12 | 2048 | top-amp | −4.78817 | −3.66182 | ❌ |

**Every `random` row passes and every decent-sized `top-amp` row fails.** The bound holds exactly when
the subspace is worse than a product state and inverts the moment it beats one -- which is the entire
purpose of SQD, since the ground state is entangled. At n=12/dim=2048 the projection has captured
−4.788 of a true −4.791 while the bound sits 1.13 too high. The failure is silent: `H − σI` comes out
*indefinite* rather than raising, so Jacobi divides by negatives and returns the ~3x pessimization with
correct energies. Same signature as every rejection above.

**Route 2 — relax the Bloch sphere to the Bloch ball (`x²+y²+z² == 1` → `<= 1`). Structurally
impossible, not merely ineffective.** The intent was that the ball is the set of single-qubit *density
matrices*, so minimizing over it might relax toward a genuine spectral bound. **Measured: the identical
objective**, −2.41108 at n=8, matching the sphere to 5 decimals at n=8/10/12. The reason is that the
objective is a sum of products with **one factor per qubit**, hence linear in each qubit's three
variables individually -- and a linear function over a convex ball attains its optimum on the boundary.
The ball optimum *is* the sphere optimum, for every Hamiltonian. Do not retry this.

**Route 3 — drop the constraint entirely (box `[-1,1]^{3n}`). Valid bound, same dead end already
rejected.** This *does* bound from below (−4.375 ≤ −3.197 at n=8, holding at n=8/10/12), but it is the
coefficient-sum bound in disguise: each component independently saturates ±1, so the optimum is exactly
`−Σ|coeff|` (verified). Over-shift against the projected `λ_min` that `sqd` actually solves:

| n | dim | box bound (= minus the coefficient sum) | `λ_proj` | over-shift |
| --- | --- | --- | --- | --- |
| 10 | 300 | −10.6250 | −2.3510 | **16.55x** |
| 12 | 800 | −12.8750 | −2.3787 | **20.99x** |
| 14 | 1 500 | −15.1250 | −2.3943 | **25.46x** |

Worsening with `n`, and marginally worse than the structural row-sum bound rejected in the section
above. Nothing new.

**Conclusion on shifts: stop looking in this direction.** Three independent routes, one cause. A
certified lower bound on `λ_min` needs a fundamentally different model -- a moment/SOS relaxation over the
Pauli algebra rather than a product-state parameterization.

**That was subsequently built and measured, and it closes the shift question for good — see
`docs/sdp-lower-bound.md`.** A level-1 SDP bound is valid and ~1.45x loose on `H`, and yields **1.29x
median with 0/12 regressions** through the real hook. Note it *is* by far the tightest bound anyone here
has produced: **0.64-1.06x over-shift** against the projected operator, versus 4.14-8.14x for the
coefficient-sum bound rejected above and 5.03-6.29x for the one this document called unattainable --
5-8x tighter, from the Pauli list alone with no assembled matrix and no matvec. It is also **superseded by the free option**: the
same 1.29x is available from the diagonal-only `σ = min(diag) − 2·max|diag|` at `O(N)`, with no conic
solve. Two findings there are worth reading even though the verdict is negative.

First, the bound has a **closed form**: `σ` is exactly linear in `n` (max residual 5.2e-08 over n=4..14),
with the slope *equal to the single-bond `λ_min`*, so the conic solve recovers a linear function that two
solves determine. Its value is in establishing the constant for a new coupling family, not per-instance
evaluation.

Second, and this **retracts the sentence this section used to end with** — a tighter bound on `H` was *not*
the route to the 1.79x. `ground_locg` sees `hproj(H, subspace)`, whose minimum sits 0.64-1.06x of the
projected spectral width **above** `λ_min(H)`; that gap is a property of the random projection, so **no
bound on `H`, however tight, can reach the shift the 1.79x was measured at.** The 1.79x used the
*projected* `λ_min`. So the remaining upside lives in estimating the **projected** operator's minimum, not
in tightening a bound on `H` — and the only untried candidate for that is still the two-level/deflation
preconditioner named as speculative above.

### ⚠️ `solve_product`'s state vector as an `xinit` seed — 1.09x, too weak to ship

What survives is the part needing no bound at all: `Solution.vec` is the optimal product state's Bloch
vectors, a genuine variational approximation to the ground state, and `solve_product` is **fast** --
measured 0.03 s / 0.04 s / 0.17 s / 0.78 s at n=8/12/16/18 with `tol=1e-4`. Amplitude on a bitstring is
`Π_q amp[q][bit_q]` with `θ_q = arccos(z_q)`, `φ_q = atan2(y_q, x_q)`: `O(N·n)`, matvec-free. Unlike a
preconditioner it cannot affect correctness -- `xinit` needs only non-vanishing overlap with `v_0`.

Measured through the real `ground_locg`, random `x0` against product-state `x0`, same 12-instance batch
(`(n,dim) = (16,2000), (18,4000)`, seeds 0-5), `maxiter=4000`, `tol=1e-10`:

| n | seeds 0-5, `rand` -> `prod` | ratios |
| --- | --- | --- |
| 16 | 44->35, 28->27, 29->26, 45->49, 54->46, 122->66 | 1.26, 1.04, 1.12, **0.92**, 1.17, **1.85** |
| 18 | 35->27, 19->17, 34->33, 37->47, 24->24, 46->43 | 1.30, 1.12, 1.03, **0.79**, 1.00, 1.07 |

**Median 1.09x, range 0.79-1.85x, 2/12 regressions**, all 12 converged to the correct energy. Recorded
as ⚠️ rather than ❌ because the effect is real and the structure is interesting: the best case is
n=16 seed=5, **122->66 (1.85x)**, which was the *worst* instance in the batch -- consistent with a seed
cutting the pathological tail, which is the thing worth buying per this document's own framing. But
0.79x on n=18 seed=3 means it **cannot be a default**, and one tail point is not evidence of
tail-cutting; that needs a batch selected for pathology. Compare Jacobi's 1.79x with 0/12 regressions:
a seed only changes the starting distance, where a preconditioner changes the per-iteration contraction
rate.

A regression is mechanically unsurprising -- a product state can have *worse* overlap with the
**projected** ground state than a random vector does, because projection onto a randomly sampled
subspace distorts it, and a random vector has no structure to lose.

**The untried extension, and the only one that addresses the real limitation:** a richer ansatz in the
SCIP model -- e.g. bond-dimension-2 MPS, `~12n` variables with per-site normalization -- would capture
the nearest-neighbour entanglement a 1D chain has and a product state structurally cannot. At 0.78 s
for n=18 there is a large budget before it stops being cheap against the SQD solve, and it would
improve both surviving uses at once (a better seed *and* a better sampling distribution for subspace
selection). Caveat: the non-convexity worsens (spatial branch-and-bound over `12n` variables with
cross-site products), so its scaling past n≈16 is an open question, not an assumption.

### ❌ Bloch-marginal subspace sampling — uninformative at `Δ=0.5` by symmetry

The remaining use of `solve_product` was to pick the *subspace* rather than to shift or seed: its Bloch
vectors give a per-qubit marginal `p_q = P(bit=1) = (1 − z_q)/2`, so bitstrings could be drawn from that
product distribution as a classical stand-in for SQD's quantum sampling step. Motivated by the
break-test data above -- at n=12/dim=2048 an oracle subspace reached −4.788 of a true −4.791 where a
random one got −3.545, so subspace quality dominates anything the eigensolver does.

**It carries no information for this Hamiltonian class.** Measured `max|z_q| = 0.0000` **exactly**, so
`p_q = 0.5` at every site and the "sampled" distribution *is* uniform:

| parameters | largest abs z-component | `p_q` range |
| --- | --- | --- |
| `Δ=0.5, Bx=By=0.5` (the request's operator) | **0.0000** | [0.500, 0.500] |
| `Δ=0.5, Bx=2.0, By=0` | **0.0000** | [0.500, 0.500] |
| `Δ=0.5, Bx=By=0` (no field at all) | **0.0000** | [0.500, 0.500] |
| `Δ=2.0, Bx=By=0.5` | 0.9852 | [0.007, 0.993] |
| `Δ=−1.0, Bx=By=0.1` | 0.6443 | [0.178, 0.213] |

The cause is symmetry, not the field: at `Δ=0.5` the XY coupling dominates ZZ, so the product optimum
is in-plane (Néel order in x/y) and the z-magnetization vanishes identically -- it holds with the field
switched **off**, which rules out "the transverse field polarizes in-plane" as the explanation. Only in
the Ising-dominated regime (`Δ=2.0`) does the marginal become informative.

There is a structural point underneath, and it also explains the weak seeding result. **Sampling
computational-basis states reads only `|amplitude|²` in the z-basis, i.e. only `z_q`.** An in-plane
product state is *maximally uninformative* in that basis while being a perfectly good variational
state: all of its content sits in the relative phases `φ_q = atan2(y_q, x_q)`, which basis-state
sampling discards. That is a mismatch between what `solve_product` optimizes and what subspace
selection can read, not a harness defect -- and it is consistent with the `xinit` seeding measuring
only 1.09x with 2/12 regressions, since the phases are all it had to offer.

**A retracted intermediate result, recorded because it was nearly believed.** A first version of this
experiment reported bloch-sampling beating random in **9/9 cells, capturing 13.6-50.3%** of the oracle's
advantage. That was an artifact of the harness: the `random` arm drew with
`rng.choice(2**n, replace=False)` while the `bloch` arm drew iid bits and applied `np.unique`. Those are
different **sampling mechanisms**, and since the marginals were all 0.5 the two arms had *identical
distributions* -- so the entire measured gap came from the mechanism, not the operator. Two controls
caught it: sampling from a deliberately **mismatched** Hamiltonian's product state did just as well
(−1.628 vs −1.513 at n=10/dim=51, i.e. slightly better), and printing the marginals showed
`[0.5, 0.5, ...]`. **When two arms differ in more than the variable under test, a large effect is
evidence of the confound, not of the hypothesis** -- and a mismatched-input control is the cheap way to
find out.

### ⛔ Subspace selection by weight shell + diagonal ranking — measured, then REJECTED by the user

**Do not build on this.** The results below are sound and are kept as a record, but the direction was
rejected on 2026-08-25 after review. The deciding gap: every measurement compares against **uniform
random** subspaces, which is a weak baseline -- a real SQD workflow samples from a quantum circuit, and
those samples are already biased toward the ground state. Whether this improves on *quantum-sampled*
subspaces was never measured, so the payoff in practice is unestablished. It is also a change to how
callers choose `states`, not an `sqd` improvement, so it does not belong in this library's scope.

Read the rest of this section as a record of what was measured and why, not as a recommendation.

### Low-Hamming-weight sampling bias — and the mechanism is *connectivity*, not amplitude

This started as a throwaway control against the Bloch confound above ("does *any* bias beat uniform?")
and turned into the session's one positive result. Draw each bit iid with `P(1) = p` instead of `p = 0.5`.

**First harness was itself confounded, twice** -- recorded because both traps are easy to repeat. It
reported a wandering optimum (best `p` = 0.30, 0.40, 0.20 with no pattern, and at n=12/dim=1638 `p=0.30`
came *last* of five). Causes: (a) low-`p` arms ran out of distinct states and were topped up with
**uniform** draws, so the arm that "won" was the one that had cheated back toward uniform; and (b)
`np.unique` then `lex`-sort then `[:dim]` keeps a *contiguous block in lexicographic order*, which
selects on leading bits -- kept-weight means came out 2.68-4.45 where the draw means were 1.2-6.0, i.e.
**truncation, not `p`, was setting the final distribution.** Different `p` gave different unique-counts
gave different truncation severity: the arms differed in more than the variable under test, again.

**Clean harness** (rejection-sample until exactly `dim` distinct iid(`p`) states; no top-up, no
truncation) gives a **monotonic** result in all five cells, 8 seeds, non-overlapping error bars:

| n | dim | `p=0.20` | `p=0.30` | `p=0.40` | `p=0.50` (uniform) |
| --- | --- | --- | --- | --- | --- |
| 12 | 204 | **−1.6872**±0.029 | −1.5699±0.034 | −1.4708±0.044 | −1.3866±0.047 |
| 12 | 614 | **−2.7667**±0.019 | −2.4175±0.016 | −2.1408±0.053 | −2.0854±0.044 |
| 12 | 1 638 | **−3.8621**±0.020 | −3.6018±0.022 | −3.1697±0.026 | −3.0374±0.034 |
| 14 | 819 | **−2.2224**±0.028 | −1.8897±0.053 | −1.8698±0.070 | −1.8610±0.035 |
| 14 | 2 457 | **−3.3924**±0.022 | −2.9667±0.032 | −2.6179±0.034 | −2.4851±0.034 |

Lower is better, and lower `p` wins monotonically every time. At n=12/dim=1638 that is **−3.862 against
uniform's −3.037**, against top-amplitude selection (the true ground state's largest-`|amplitude|`
states) at −4.783 -- roughly half that gap, from a one-line change to how bitstrings are drawn. Note
top-amplitude selection is a *reference*, not a ceiling: it maximizes fidelity with the ground state,
where `λ_min` of the projection is variational and rewards connectivity, and the hybrid section below
beats it at n=14.

**It appears to contradict the ground state's own statistics, and the resolution is the point.** The
true ground state's mean Hamming weight is *exactly* `n/2` (measured 6.000 at n=12, 7.000 at n=14), so
the amplitude-matched choice would be `p = 0.5` -- uniform. Low-`p` sampling deliberately samples *away*
from where the amplitude mass sits, and wins anyway. Why:

| `p` | `λ_min` | off-diagonal nnz | nnz/row | mean pairwise Hamming distance |
| --- | --- | --- | --- | --- |
| 0.20 | **−2.8329** | 4 972 | **8.10** | 5.128 |
| 0.30 | −2.4760 | 3 558 | 5.79 | 5.444 |
| 0.40 | −2.0488 | 2 108 | 3.43 | 5.801 |
| 0.50 | −1.9420 | 1 672 | **2.72** | 6.004 |

(n=12, dim=614, one seed.) **Connectivity tracks the energy in lockstep** -- a 3x range in nonzeros per
row across the same 3x range in `λ_min` improvement. The mechanism: XXZ's `XX`/`YY` terms connect states
at Hamming distance **2** and the field's `X`/`Y` terms at distance **1**, so a matrix element exists
only when the subspace contains *both* endpoints. Low-`p` draws concentrate in a small region of the
hypercube (weight 0-5 rather than 0-9), making near-neighbour pairs common; uniform draws scatter `dim`
states thinly over all `2^n`, where typical pairs sit ~6 apart and contribute **nothing**.

So the quantity a projected variational energy needs is **connected** states, not individually
high-amplitude ones: a subspace of important-but-mutually-unconnected states is nearly diagonal, and its
`λ_min` degenerates to its best diagonal entry. This is the same principle that put `_spread_seed` in
`rqutils/sqd.py` -- a one-hot cannot leave the connected component containing it -- applied to subspace
selection rather than to the initial vector.

**Extending `p` below 0.20: the parameter stops controlling anything, and the question dissolves.**
Swept `p` over 0.03-0.50 at n=12/dim=204, n=12/dim=614, n=14/dim=819, 8 seeds, with saturation
diagnostics. The energy does flatten and marginally reverse around `p ≈ 0.03-0.05` -- but that is the
**sampler**, not an optimum:

| p | `λ_min` (n=12, dim=204) | nnz/row | realized mean weight | target `n·p` | draws per state |
| --- | --- | --- | --- | --- | --- |
| 0.03 | −1.9705±0.015 | 7.46 | **2.61** | 0.36 | **183.2** |
| 0.05 | **−1.9747**±0.009 | 7.22 | **2.64** | 0.60 | 55.2 |
| 0.10 | −1.8711±0.013 | 6.56 | 2.74 | 1.20 | 20.1 |
| 0.15 | −1.8032±0.016 | 5.65 | 2.93 | 1.80 | 20.1 |
| 0.20 | −1.6872±0.029 | 4.65 | 3.14 | 2.40 | 20.1 |
| 0.30 | −1.5699±0.034 | 2.57 | 3.82 | 3.60 | 20.1 |
| 0.50 | −1.3866±0.047 | 0.83 | 6.02 | 6.00 | 20.1 |

**The realized weight detaches from the target below `p ≈ 0.20`.** At `p=0.03` the target mean weight is
0.36 while the subspace actually has **2.61** -- there are only ~13 states of weight ≤ 1 at n=12, so
filling 204 distinct states is impossible and the sampler reaches up to weight ~4.6, grinding through
**183 draws per kept state** (2 296 at dim=614). Across `p` = 0.03-0.15 the realized mean weight moves
only 2.61 → 2.93 while `n·p` moves 0.36 → 1.80: those arms have converged to nearly the same
distribution, which is why the curve flattens. **`p` below ~0.15 is an expensive approximation to a
weight-shell subspace**, not a measurement of `p`.

So the right parameterization is the shell itself -- take all states of Hamming weight ≤ k,
lowest-weight first, filling to `dim` (which fixes `k` automatically). Measured against the best sampled
arm, 8 seeds:

| n | dim | weight-shell | `k` reached | best sampled `p` | top-amplitude |
| --- | --- | --- | --- | --- | --- |
| 12 | 204 | **−1.9958**±0.014 | 3 | −1.9747 | −3.9083 |
| 12 | 614 | **−3.0537**±0.012 | 4 | −3.0399 | −4.5878 |
| 14 | 819 | **−2.7655**±0.012 | 4 | −2.7418 | −4.5411 |

The shell construction **wins in 3/3 cells** (small margins, consistently signed) with the highest
connectivity measured (nnz/row 7.79 / 10.46 / 9.90), has **no free parameter**, and costs one
enumeration instead of 100-2000x oversampling. So: there is no useful turnover in `p`; the low-`p` limit
*is* the weight-shell subspace, and it should be built directly.

**The hybrid — shell for connectivity, ranking within it — works, and the *deployable* variant wins.**
A pure shell forces most of the subspace (at n=12/dim=204, 79 of 204 states are fixed by the full shells
≤ 2; only the 125 drawn from the 220-state weight-3 shell are free), so the experiment widens the
candidate window by `w` shells and ranks inside it. Ranking by the **analytic diagonal**
(`Δ/4 · Σᵢ sᵢsᵢ₊₁`, exact, matvec-free, needs no solve) against ranking by true ground-state
`|amplitude|`:

| n | dim | shell | shell+diag | shell+amp | **wind3+diag** | **wind4+diag** | top-amplitude |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 12 | 204 | −2.0504 | −2.5080 | −2.5447 | **−3.6572** | −3.5725 | −3.9083 |
| 12 | 614 | −3.0530 | −3.4637 | −3.3063 | **−4.2597** | −4.2600 | −4.5878 |
| 14 | 819 | −2.7373 | −3.3334 | −3.1598 | −4.5449 | **−4.5519** | −4.5411 |

Three findings, in increasing order of consequence.

**1. The analytic diagonal outranks true amplitude** (`shell+diag` beats `shell+amp` in 2 of 3 cells:
−3.4637 vs −3.3063, −3.3334 vs −3.1598). So the practical criterion is the *better* one, not a
compromise. Coherent with the connectivity mechanism: amplitude picks individually-important states with
no guarantee they couple, while a low diagonal in a chain means locally-alternating spins, which are
Hamming-neighbours of each other -- so the diagonal is *implicitly* a connectivity criterion. The
`nnz/row` diagnostic confirms it, higher for `+diag` than `+amp` in all three cells.

**2. Widening has a genuine interior optimum at `w = 3-4`**, unlike the `p` sweep which had none.
Measured across `w = 1..8`: n=12/dim=204 peaks at `w=3` (−3.6572, then −3.57, −3.51, plateauing ~−3.55);
dim=614 at `w=3-4` (−4.2597/−4.2600, then −4.19, −4.14); n=14 at `w=4` (−4.5519, then −4.49 flat). Too
narrow starves ranking, too wide dilutes the low-diagonal preference over a window so large that
selection scatters. An interior maximum is the signature of a real trade-off.

**3. It beats top-amplitude selection at n=14** -- `wind4+diag` gives **−4.5519** against **−4.5411** --
and that is legitimate rather than a defect. Top-amplitude selection maximizes *fidelity* with the ground
state; `λ_min` of the projection is a *variational* quantity that rewards **connectivity**. Different
objectives, so neither dominates, and the amplitude arm should not have been called an "oracle" or a
ceiling (corrected above). **Verified independently**: both subspaces are exactly `dim` states, distinct
and lex-sorted, and `hproj`'s `λ_min` agrees with a from-scratch dense `Vᵀ H V` to **5.9e-08** for both
arms -- far tighter than the 0.011 margin claimed.

So the deployable recipe reaches **93.5% / 92.8% / 100.2%** of top-amplitude selection's energy using
only the analytic diagonal: no ground state, no solve, no SCIP, `O(N·n)` bit arithmetic.

**Characterizing the optimum: `w` is the wrong parameter, and `kmax` is anchored to `n`.** Swept 13 cells
over n=12-20 and dim=204-10 000, varying `n` and `dim` **independently** (the first three cells had
confounded them). `w*` ranges **1 to 8** -- so the earlier "3-4" was an artifact of three nearby cells --
and it varies systematically: *decreasing* with `dim` at fixed `n` (n=12: 3→2→1; n=14: 4→3→2; n=16:
7→4→3) and increasing with `n`. But the **absolute** top weight `kmax = k + w*` is stable at
`kmax/n ≈ 0.5`, exactly 0.500 in all six n=12/14 cells. That is the ground state's mean Hamming weight
(`n/2`, measured exactly earlier), confirming the physical reading: **the window needs to reach the
weight shell where the ground state lives, and no further.** Two competing hypotheses die here -- the
selection ratio `window/dim` is *not* constant (2.0 to 166.4), and a rule anchored to `k` rather than `n`
(`kmax = k+2`) is badly wrong (up to **+18.31%** shortfall).

Two closed-form rules, each optimal on a different half of the grid, shortfall against the swept optimum:

| strategy | worst | median | mean | projected solves |
| --- | --- | --- | --- | --- |
| `kmax = ceil(n/2)` | 2.77% | 0.07% | 0.86% | 1 |
| `kmax = ceil(n/2) + 1` | 5.21% | 0.17% | 0.89% | 1 |
| `kmax = k + 2` | 18.31% | 4.94% | 6.37% | 1 |
| **better of the two `n/2` rules** | **0.26%** | **0.00%** | **0.02%** | 2 |

`ceil(n/2)` is exact at n=12/14 and n=20/dim=3000 but loses 1.5-2.8% at n=16/18 and large `dim`;
`ceil(n/2)+1` is the mirror image, optimal at n=16/18 and losing 5.21% at n=12/dim=204. **Probing both
and keeping the better costs one extra projected eigensolve and is essentially optimal** (0.26% worst
case) -- that is the recommendation, and it works *because* the optimum is broad rather than sharp: the
region within 1% of best spans `w` = 4..9 at n=16/dim=800 and 3..9 at n=16/dim=8000, so two well-chosen
probes bracket the basin. A sharp optimum would have needed a real search. Note this also explains an
earlier over-reading: at n=16/18 the raw argmin picks among values differing by ~0.003, so `w*` itself is
poorly determined there while `kmax` is not.

Use `kmax = max(k, ...)` in either rule -- the window must be large enough to hold `dim`, and at large
`dim` the shell `k` where `dim` runs out has already passed `n/2`, which is precisely where the plain
`n/2` rule's floor binds and it forfeits its 1.5-2.8%.

**Checked across `Δ`: the anchor shifts by exactly one shell and then saturates.** Swept `kmax` over its
full legal range at `Δ` = 0.5, 2.0, 4.0 on seven cells (n=12-16, dim=204-3 000), including `kmax` values
*below* `ceil(n/2)` which no rule variant allowed. Two corrections come out of it.

First, a **retraction**: the caveat previously here said the `Δ=2.0` ground state is z-polarized (citing
`max|z| = 0.985` from the Bloch section), so the anchor should move. That conflated two different objects.
The **product-state approximation** becomes z-polarized; the **true ground state** does not move at all --
its mean Hamming weight is *exactly* `n/2` at every `Δ` measured (0.5, 1.0, 2.0, 4.0), pinned by the
spin-flip symmetry (conjugating by `X` on all sites maps weight `w → n−w`, so any non-degenerate
eigenstate's weight distribution is symmetric about `n/2`). What changes with `Δ` is the **spread**, not
the mean: std falls 0.741 → 0.343 as `Δ` goes 0.5 → 4.0 at n=12, and the top-2 states' mass rises
**0.052 → 0.715** as the state concentrates onto the two Néel configurations (which themselves sit at
weight `n/2`).

Second, the anchor does shift -- by one shell, discretely, then stops. Reading the **effective** optimum
(smallest `kmax` within 0.1% of best) rather than the raw argmin, which is essential here because the
curves are flat past the optimum and the argmin picks among ties (at `Δ=2.0`, n=12/dim=204 reads
`7:−6.905 8:−6.905 9:−6.905 10:−6.905` and the raw argmin reported `kmax*=10`, i.e. `kmax/n = 0.833`):

| `Δ` | effective `kmax` (n=12 / 14 / 16) | as a fraction of `n` | rule that is exact |
| --- | --- | --- | --- |
| 0.5 | 6 / 7 / 8 | 0.500 | `ceil(n/2)` |
| 2.0 | 7 / 8 / 9 | 0.583 / 0.571 / 0.562 | `ceil(n/2) + 1` |
| 4.0 | 7 / 8 / 9 | 0.583 / 0.571 / 0.562 | `ceil(n/2) + 1` |

`n/2 + 1` in **all six cells at both** `Δ=2.0` and `Δ=4.0` -- so the shift saturates rather than tracking
`Δ`. Consistent with the mechanism: at `Δ=0.5` the weight distribution is broad enough (std 0.74-1.08)
that `n/2` already spans the mass, while at `Δ ≥ 2` the state concentrates onto the two weight-`n/2` Néel
states and the window needs **one shell beyond** to include their single-flip neighbours -- the states
that supply the off-diagonal coupling. Connectivity again, which is what the whole mechanism predicts.

**The recommendation is unchanged and now validated over an 8x range in `Δ`.** Best-of-two-probes is
**≤0.06% everywhere** across all three `Δ`. The individual rules swap which is exact -- `ceil(n/2)` is
optimal at `Δ=0.5` and costs +0.54% to +1.64% at `Δ=2.0`; `ceil(n/2)+1` is optimal at `Δ≥2.0` and costs up
to +5.21% at `Δ=0.5` -- which is precisely why probing both is the right answer rather than picking one.

**Remaining caveats.** Subspace fractions span 0.6-40%; `Δ` covered at 0.5/2.0/4.0 with `J=1`,
`Bx=By=0.5` fixed throughout, so the field's role is untested. n ≤ 20. `λ_min` for the larger cells came from sparse `eigsh` at `tol=1e-10`,
validated against dense `eigvalsh` to **7.9e-08** at n=16/dim=3000 (11.24 s → 0.02 s), three orders below
the inter-arm differences being compared.

**Remaining caveats.** It is measured only at n=12/14, on `Δ=0.5`, with subspace
fractions 5-40%. And it owes **nothing to SCIP or `product.py`**: it is a property of the sampling
distribution's locality, so it applies to any subspace-selection scheme, and would compose with rather
than replace amplitude-based selection.

---

## What lands in `spinchain`

Nothing to delete, which is worth saying plainly — this is a performance ask, not a simplification
one. `sqd_backend.ground_state` would build its own shifted Jacobi preconditioner and pass it through
`ground_locg`'s `precond` -- **not** via an `sqd` flag, which was measured to be a 3x pessimization
because `sqd` does not shift (see the rejected-convenience section). `spinchain` already computes the
spectral information a correct shift needs. The expected effect is on the tail rather than the mean, and
the 10.087 s outlier at dim=128 000 is the number to watch, not the 0.911 s best case.

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
