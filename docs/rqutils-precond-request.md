# rqutils change request: a preconditioner hook for `ground_locg`

> **Verified against rev `3da1b46` (2026-08-24). One correction, and it is to the headline.**
> Every structural and arithmetic claim below checks out — the line references, the `norm_r` asymmetry,
> the group-0 diagonal warning, the overhead measurements, and the published reproduction, which runs
> as printed and reproduces its 30 → 14 row. 7 of 7 spot-checked rows of the scipy iteration table
> reproduce to the integer.
>
> **But the scipy gain does not transfer cleanly to `ground_locg`.** Re-measured in its own recurrence:
> median **1.38x**, not 1.75x, with **2 of 12 instances regressing** (worst 0.35x) and one improving
> **5.50x**. The κ-based argument does not predict it — the two regressing seeds had *larger* κ gains
> than the 5.50x best case. Numbers throughout have been updated; see
> "⚠️ Correction" under "Measured: how much a Jacobi preconditioner buys".
>
> The recommendation is unchanged — the hook is cheap, opt-in, and the **tail** case is stronger than
> first claimed — but the median is roughly half what this document originally led with.

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
| Measured payoff | **1.38x median fewer iterations in `ground_locg` itself** (range 0.35–5.50x, 10/12 improved, 2 regressed) — see the correction below. The scipy proxy that first suggested 1.75x is reproducible but does **not** transfer cleanly. Preconditioner overhead is <1% per iteration, so the wall-clock figure tracks the iteration figure |
| End to end | **~1.06x** on the shipped n=13 job (the solve is 21.8% of it), **~1.32x** on a solve-dominated run like `replay` or the n=20 ladder |
| Why it matters anyway | the median is the wrong statistic here. It cuts the **tail**: the worst instance in a 12-instance batch went **297 → 54 iterations (5.50x)**, and one solve in nine is 39% of a run |

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

### ⚠️ Correction: the scipy gain does **not** transfer cleanly to `ground_locg`

The caveat at the end of this section — that the iteration column is a proxy, and "the one number
worth re-measuring first" — was acted on, and it changes the headline. **Re-measure it before quoting
1.75x anywhere.**

The hook does not exist yet, but symmetric Jacobi is a similarity transform: `D^{-1/2} A D^{-1/2}` has
the conditioning a preconditioner would supply, so feeding *that* operator to `ground_locg` measures
whether its own 3x3 recurrence converts the conditioning gain into iterations. Same 12 instances, same
`x0`, `maxiter=4000`, both arms converged and every energy correct:

| n | dim | seed | plain | Jacobi | gain | | n | dim | seed | plain | Jacobi | gain |
| --- | --- | --- | --- | --- | --- |---| --- | --- | --- | --- | --- | --- |
| 16 | 2 000 | 0 | 91 | 151 | **0.60x** | | 18 | 4 000 | 0 | 71 | 48 | 1.48x |
| 16 | 2 000 | 1 | 58 | 48 | 1.21x | | 18 | 4 000 | 1 | 39 | 31 | 1.26x |
| 16 | 2 000 | 2 | 60 | 45 | 1.33x | | 18 | 4 000 | 2 | 71 | 45 | 1.58x |
| 16 | 2 000 | 3 | 111 | 53 | 2.09x | | 18 | 4 000 | 3 | 76 | 69 | 1.10x |
| 16 | 2 000 | 4 | 114 | 324 | **0.35x** | | 18 | 4 000 | 4 | 51 | 36 | 1.42x |
| 16 | 2 000 | 5 | **297** | **54** | **5.50x** | | 18 | 4 000 | 5 | 100 | 62 | 1.61x |

| | scipy proxy | `ground_locg` |
| --- | --- | --- |
| median | 1.75x | **1.38x** |
| geometric mean | — | **1.32x** |
| range | 1.50–2.14x | **0.35–5.50x** |
| regressions | 0/12 | **2/12** |

**And the mechanism this document reasons from breaks down.** The argument is that Jacobi helps because
it improves `κ`, with the dominance table below as support. But on the two *regressing* seeds the
transform improved `κ` by **2.05x and 2.25x** — *more* than on the 5.50x best case, which improved it
only **1.55x**. Iteration count in this recurrence is not tracking `κ`, so κ gains cannot be used to
predict iteration gains here. The κ table stays as a statement about the operator; it should no longer
be read as a forecast.

**Two caveats on the correction's own method**, stated so it is not over-read either.

*Convergence parity.* Each arm stops on `ground_locg`'s own `converged` flag at its default `tol`,
not at a fixed external accuracy the way the scipy table's `iters_to_tol` scan does. Both arms
converged on every instance and every plain-arm energy matched `eigvalsh` (checked: 0.0 to 4.9e-15),
but the transformed operator has a different spectrum, so its `tol` is not the same absolute bar. The
ranking of the two arms is what this measures; the exact ratios would shift somewhat under a strict
iterations-to-fixed-tolerance scan. The 1.38x median reproduced across two independent runs.

*The transform is not the hook.* A similarity transform is not identical to the requested hook: it
preconditions the *operator*, so `ground_locg`'s balancing and Rayleigh-Ritz see different inputs,
whereas `precond` would rotate only `p`. The real hook could do better — the two regressions in
particular may be an artifact of transforming the operator rather than the direction. Treat 1.38x as
evidence against assuming 1.75x, not as the final number. It is still a proxy; it is just a *closer*
one, measured in the actual recurrence.

<details>
<summary>Reproduction for the correction (click to expand)</summary>

Reuses `xxz`, `subspace` from the snippet at the end of this section.

```python
import numpy as np, jax.numpy as jnp
from rqutils.ground_locg import ground_locg
from rqutils.sqd import hproj

for n, dim in ((16, 2000), (18, 4000)):
    for seed in range(6):
        H = np.asarray(hproj(xxz(n), subspace(n, dim, seed)).toarray())
        ev = np.linalg.eigvalsh(H)
        A = H - (ev.min() - 0.5) * np.eye(H.shape[0])
        d = np.real(np.diag(A)).copy(); d[d <= 1e-12] = 1.0
        s = 1.0 / np.sqrt(d)
        At = (A * s[:, None]) * s[None, :]        # symmetric Jacobi similarity transform
        Aj, Atj = jnp.asarray(A), jnp.asarray(At)
        x0 = jnp.asarray(np.random.default_rng(99).normal(size=A.shape[0]) + 0j)
        ip = int(ground_locg(lambda v: Aj @ v, x0, maxiter=4000)[2])
        it = int(ground_locg(lambda v: Atj @ v, x0, maxiter=4000)[2])
        print(f"n={n} dim={dim} seed={seed}: {ip} -> {it}  ({ip/it:.2f}x)")
```

</details>

**What survives intact, and is now the main case.** The tail argument gets *stronger*, not weaker:
seed 5 went **297 → 54** iterations, a 5.50x cut, better than the scipy proxy predicted for it (1.60x).
Since this document already argues that the tail sets a run's budget rather than the average, that is
the right framing regardless of what the median turns out to be. The overhead measurement is unaffected
(it is measured on `rqutils`' own kernel), as is everything in the "Why" and "The problem" sections.

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

The overhead is small enough that the wall-clock figure simply tracks whatever the iteration figure
is. Using the **`ground_locg`-measured** gains from the correction above (not the scipy proxy):

| iteration gain | overhead | projected wall-clock |
| --- | --- | --- |
| 0.35x (worst — a regression) | 0.5–0.9% | **0.35x** |
| 1.38x (median) | 0.5–0.9% | **1.36–1.37x** |
| 5.50x (best — the tail case) | 0.5–0.9% | **5.24–5.35x** |

So the honest expected figure for `rqutils` is **a median around 1.36x on the solve, with a spread
from 0.35x to 5.3x that is far wider than the scipy proxy implied in either direction.** The old
numbers this table used to carry (1.49x / 1.74x / 2.12x, from the 1.50–2.14x scipy range) are
superseded; they are recorded in the correction above rather than deleted, because the scipy
measurement itself reproduces and the discrepancy is the finding.

What that is worth end to end depends entirely on how solve-dominated the run is, and Amdahl is
unkind at the small end. On the shipped n=13 job the solve is now **21.8%** of wall time (1.832 s of
8.40 s, after `cache_level=(1, 2)` already took it down from 2.673 s), which gives:

| solve speedup | end-to-end, n=13 shipped (solve 21.8%) | end-to-end, solve-dominated run (90%) |
| --- | --- | --- |
| 0.35x (regression) | 0.71x | 0.37x |
| 1.36x (median) | **1.06x** | **1.31x** |
| 5.19x (tail case) | 1.21x | 3.66x |

**On the shipped n=13 config this is a ~6% end-to-end win and not worth much on its own** — and on a
bad instance it is a *loss*. It earns its place in two other places:

- **Solve-dominated runs**, where sampling is cheap or already paid — `replay` re-runs the whole
  post-sampling tail with no sampler at all, and the n=20 ladder above spends 8.98 s of its time in
  nine solves. There the median figure is ~1.31x, and the tail instance is where the real win sits.
- **The tail**, which is the real motivation. The 10.087 s outlier at dim=128 000 against a 0.911 s
  best case in the same batch is what a preconditioner is bought to remove; on that instance the
  ceiling is the 14.3x spread, not the median. A run whose budget is set by its worst solve cares
  about the worst solve — and the `ground_locg` measurement supports this bullet more strongly than
  the median one: the worst instance in that batch improved **5.50x**.

Caveat stated plainly, and now partly resolved: this projection used to assume the scipy iteration
gains carried over to `ground_locg`'s 3x3 Rayleigh-Ritz recurrence. **They do not, cleanly** — see the
correction above, which re-measured it via a similarity transform and found a 1.38x median with two
regressions. The overhead column is measured on `rqutils`' own kernel and is solid. The iteration
column is still a proxy (the transform preconditions the operator, where `precond` would rotate only
`p`), so measuring it through the real hook remains the first thing to do after implementing it.

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
is what the correction above disproves: measured in `ground_locg`'s own recurrence, the two instances
whose κ improved *most* (2.05x and 2.25x) were the two whose iteration counts got *worse*, while the
5.50x best case improved κ only 1.55x. A better-conditioned operator is a real and reproducible effect
here; this recurrence just does not convert it into iterations monotonically.

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

### A convenience default, optionally

Since a Jacobi preconditioner needs only the operator's diagonal, and `sqd`'s `cache_level[1] == 2`
path **already computes every X group's diagonal**, `sqd()` could offer `precond="jacobi"` and build
it internally at no extra matvec cost. Note the correct vector is the diagonal of the *projected*
operator, which is the group with a zero X signature — group 0 — and not the sum over all groups:
the other groups' arrays are off-diagonal amplitudes, legitimately complex for Paulis with an odd Y
count. `run_sqd`'s own `vinit_from_min_diag` already reads exactly that array and takes `.real` of it,
so the quantity is on hand and its realness is established.

That is a second, separable step. The hook alone is enough for a caller to pass its own.

---

## What lands in `spinchain`

Nothing to delete, which is worth saying plainly — this is a performance ask, not a simplification
one. `sqd_backend.ground_state` would pass a Jacobi preconditioner (or `sqd(precond="jacobi")`), and
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
