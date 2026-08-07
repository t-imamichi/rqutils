# Robustness, accuracy, and speed review of `rqutils/ground_locg.py`

> ## ⚠️ STALE — historical reference only
>
> **Every defect described in this document has been fixed.** It audits the *pre-rewrite*
> `ground_locg.py` as of `143601c` (2026-08-04); the module was replaced in `8fa6be2` and has been
> revised repeatedly since. Do not use this document to decide whether the current code is correct,
> and do not cite it as a live description of behaviour.
>
> **Specifically stale:**
>
> - **All line numbers.** They refer to `143601c`. The file has since roughly doubled in length and
>   every function moved. `eigenpair_3x3:453` and friends point at unrelated code today.
> - **The "Scope and limitations" section's central claim.** It says "There is no pytest suite in
>   this repository […] no tests were left behind, and no file under `rqutils/` was modified."
>   Both halves are now false: `tests/test_ground_locg.py` exists and pins most of these items, and
>   the module was rewritten.
> - **The API gaps A1–A5.** A1 (no convergence flag) and A5 (`debug` unreachable) are implemented —
>   `ground_locg` returns a 4-tuple ending in `converged` and forwards `debug`. A2 (`maxiter=0`
>   returning `0.0`), A3 (`tol` from `xinit.dtype`) and A4 (unguarded zero `xinit`) are fixed too.
> - **The severity of I5.** Still guarded, but no longer reachable from outside: neutering the
>   re-orthogonalization leaves the whole test suite green, and dense operators at shifts 1e9–1e12
>   agree to 4e-15 either way. The measured `|⟨x|y⟩| = 1.0` below required the *other* defects to
>   compound with it — chiefly I4, which drove every run to 2000 iterations. With those fixed the
>   same problems converge in 8–46 iterations and never enter the regime. See
>   `_reorthogonalize`'s docstring.
> - **S1 is done**, and the current loop carries `ax` in a `_State` NamedTuple. The 4-matvec
>   figure describes code that no longer exists; today `body()` traces exactly 3.
> - **The `jnp.linalg.eigh` compile-time argument** was already re-measured and reversed *within*
>   this document; the surviving reason to keep the closed form is GPU fusability, not compile time.
>
> **What it is still good for:** the measurements. The randomized sweep counts, the failure modes
> and the concrete replacement formulations — especially the `eigenpair_2x2` δ-sign warning and the
> methodological note that aggregate random sampling does not exercise data-dependent branches
> evenly — are why `tests/test_ground_locg.py` is organized by defect. The binding invariants now
> live in `rqutils/ground_locg.py`'s module docstring, next to the code, and the tests enforce them.

Assessment date: 2026-08-04. Investigated against branch `metal`; `ground_locg.py` last touched in
`143601c`, all line numbers below refer to that revision. **The findings below are written in the
present tense against that revision; read every "is" as "was".**

## Short answer

Six independent defects, three in the analytic eigenpair kernels and three in the LOBPCG iteration,
plus one redundant matrix-vector product. On a randomized sweep of 90 Hermitian matrices spanning
shifts, clustered spectra, exact degeneracies, and wide dynamic range:

| | failures | median iterations |
|---|---|---|
| current | **29 / 90** | 76 |
| all fixes applied | **0 / 90** | 65 |

Separately, removing the redundant matvec cuts wall-clock by **23%** (7438 ms → 5710 ms at
$n = 3000$) with bit-identical output.

The defects compound. Fixing only the kernels leaves 4 of 9 targeted cases failing, because the
iteration-level bugs dominate whenever the operator carries a large trace. Any subset is a partial
fix.

## Summary of findings

| # | Location | Defect | Severity |
|---|---|---|---|
| I1 | `eigenpair_3x3:453` | Trace cancellation → `sqrt` of a negative radicand → `NaN` | Critical |
| I2 | `eigenpair_3x3:447-456` | No scaling; `c0` is cubic in the entries → overflow/underflow | High |
| I3 | `eigenpair_3x3:465` | Fixed column pair in `cross` → wrong eigenvector at rank deficiency | High |
| I4 | `body:329` | `reltol` sign error vs. upstream → convergence test never fires | Critical |
| I5 | `body:302` | $x$/$y$ orthogonality loss → Rayleigh–Ritz returns $\theta$ **below** the true minimum | Critical |
| I6 | `_project_out:421` | Returns norm $\ge 0.99$, not $1$; Rayleigh–Ritz assumes orthonormal | Medium |
| I7 | `body` / `compute_sas` | Zeroed $P$ → $0/0$ → `NaN` | Medium |
| S1 | `body:298,306` | 4 matvecs per iteration where 3 suffice | 23% wall-clock |

`eigenpair_2x2` is hit by I1 and I2 as well, plus a `0/0` of its own at `:431` — **2353 `NaN`s in
40000 random inputs**. It has its own section below, with the concrete replacement formulation and a
warning about the branch that is easy to get wrong.

## Kernel defects

### I1. Trace cancellation in `eigenpair_3x3` (line 453)

`p = c2² - 3·c1` and `q` are built from the **unshifted** characteristic-polynomial coefficients.
For $H = A + sI$, $c_2 \approx -3s$, so $c_2^2$ and $3 c_1$ both grow as $s^2$ while their
difference stays $O(\mathrm{spread}^2)$. At $s = 10^6$ roughly twelve digits are gone; the radicand
under `jnp.sqrt` on line 456 goes **negative** and the function returns `NaN`.

This is not an exotic input. It is the ordinary case for a physical Hamiltonian, which is rarely
traceless — precisely what `sqd.py` feeds this solver.

Fix: subtract $\mathrm{tr}/3$ before forming the coefficients. The eigenvector is unchanged by the
shift, so only the returned eigenvalue needs the shift added back.

### I2. No balancing (lines 447–456)

`c0` is cubic in the matrix entries, so matrices well inside the float64 range are destroyed by
intermediate overflow or underflow. Measured relative eigenvalue error: `7.8e-01` at scale
$10^{-160}$, `NaN` at $10^{150}$.

Fix: divide by $\max |{\rm mat}|$ up front, multiply the eigenvalue back afterwards.

### I3. Fixed column pair in the cross product (line 465)

The null vector is taken as `cross(mat[:,1], mat[:,2])`. When that particular pair is rank
deficient the cross product vanishes and the routine returns a garbage direction. This reproduces on
an input as innocuous as `diag(5, 6, 1)`: the eigenvalue is exact but the returned eigenvector has
residual **0.67**.

Across 5000 random matrices the current kernel returns a wrong-or-`NaN` eigenvector **1148** times,
with worst overlap deficit `6.6e-01`.

Fix: form all three cross products and take the largest, with a rank-1 fallback (degenerate lowest
eigenvalue, where the null space is the orthogonal complement of the largest column) and a rank-0
fallback (a multiple of the identity, where any unit vector is valid).

### Kernel results

With I1–I3 fixed plus a Rayleigh-quotient polish (see below), over 20000 random Hermitian matrices:

| | `NaN` results | worst eigenvalue error | wrong eigenvector (of 5000) |
|---|---|---|---|
| current | **4066 / 20000** | 5.7e-04 | **1148** |
| fixed | 0 | 1.5e-15 | 0 |
| `jnp.linalg.eigh` | 0 | 1.6e-15 | 0 |

The fixed closed form matches LAPACK to within a bit. The Rayleigh-quotient polish
($\theta \leftarrow v^\dagger B v$ on the balanced matrix) is second-order in the eigenvector error
and lifts near-degenerate cases from $\sqrt{\epsilon} \approx 3\times10^{-9}$ to full
$10^{-16}$ — worth its cost, since it is three flops on a 3-vector. Note this fixes the
*eigenvalue* only; see "The polish rescues $\theta$, not $v$" below.

Cost: compile 46 ms → 85 ms, per-call 5.0 µs → 8.2 µs.

**Why not `jnp.linalg.eigh`, restated.** This was originally justified on compile time: `eigh` cost
126 ms against the closed form's 85 ms. **That is no longer true** — re-measured on jax 0.11.0,
`eigh` compiles in **20.3 ms (f64) / 21.8 ms (complex128)** against the closed form's unchanged
83.9 / 106.6 ms, and is *faster* per call (5.91 vs 6.93 µs). End-to-end first-call compile for
`ground_locg` drops 279 ms → 3.1 ms at $n = 200$. Anyone re-deriving the decision from compile time
alone will now reach the opposite conclusion, so the argument that actually survives is:

- **`eigh` is a fusion barrier inside the `while_loop` body.** On GPU it lowers to an FFI custom
  call, `{cuda,rocm}solver_syevd_ffi` (`jax/_src/lax/linalg.py:1298-1306`), which XLA cannot fuse
  into the surrounding elementwise graph. The closed form is nine-number scalar arithmetic that
  folds into neighbouring kernels. This is the real content of the "`jnp.sum` hates small arrays"
  design intent.
- **cuSOLVER `syevd` on a 3×3 is pathological**: a blocked algorithm with workspace queries, sized
  for $n$ in the hundreds-to-thousands. Compare the sibling MAGMA path's own docstring warning
  (`linalg.py:150-152`), "typically slower than the equivalent LAPACK implementation for small
  matrices (less than about 2048)" — three orders of magnitude above $n = 3$. There is also an
  `info` round trip per call (`linalg.py:1313-1318`).
- `implementation=JACOBI` does not help: still a cuSOLVER custom call, still unfusable.

Measured end-to-end with `eigh` swapped in for both kernels (CPU, f64): six cases spanning
$n = 200$–800, shifts $0$ and $\pm 10^{6}$, real and complex operators — **bit-identical
eigenvalues and identical iteration counts** in every case. At these sizes the matvec dominates and
the projected solve is not where accuracy or time is decided; the GPU fusability argument above is
the whole reason to prefer one over the other.

### The polish rescues $\theta$, not $v$

The Rayleigh-quotient polish is second order in the eigenvector *angle* error, so it repairs the
eigenvalue while leaving the eigenvector as computed. `_nullvec_3x3`'s cross products are the
cancellation-prone step (Kopp §3.2: "the subtractions … are very prone to cancellation errors"), and
under near-degeneracy they lose the eigenvector itself — there is then nothing left for the polish
to recover from.

Measured, complex128, on a matrix whose two lowest eigenvalues are separated by
$g = \mathrm{gap}/(\epsilon\|A\|) \approx 6.5\times10^{5}$:

| | $\theta$ error | residual | $\lvert\langle v_{\mathrm{true}} \vert v\rangle\rvert$ |
|---|---|---|---|
| closed form | 1.16e-10 | 7.18e-11 | **0.447** |
| `jnp.linalg.eigh` | 2.4e-17 | 3.4e-17 | 0.999999999999 |

The returned eigenvector is nearly orthogonal to the truth while $\theta$ still looks fine to ten
digits. This matters here specifically because LOBPCG propagates the *vector*: $\kappa$ becomes the
next iterate's search direction, which is the same class of hazard as I5. Sweeping $g$ in
half-decade bins, the closed form's p99 residual grows monotonically with $g$ (4.9e-16 at
$g \sim 1$ to 3.6e-09 at $g \sim 10^{6}$) with p99 eigenspace angle $\approx 1$ across
$3 \le g \le 3\times10^{3}$; a fixed-4-sweep Hermitian Jacobi holds ~1e-15 residual and
$\le 0.15$ angle throughout.

It has not been shown that `ground_locg` ever *builds* such a `sas`: `_project_out` keeps
$\{x, y, p\}$ orthonormal by construction, and across ten controlled end-to-end runs (real and
complex, $n = 200$–800) every kernel variant tried returned bit-identical eigenvalues with
identical iteration counts. Recorded because the failure is silent and the diagnostic — check the
eigenvector, not just $\theta$ — is not obvious from the code.

### `eigenpair_2x2` shares I1 and I2

`tr*tr - 4.*det` on line 430 is the same cancellation as I1; relative error reaches `5.8e-01` at
shift $10^9$. The cubic-free but unbalanced intermediates are the same I2 problem. And line 431
computes `0/0`, returning `NaN` for any multiple of the identity — including the plain identity
matrix and, more insidiously, any already-diagonal input whose lower eigenvalue sits in position 0
(`diag(1, 5)` returns `NaN`).

Over 40000 random 2×2 Hermitian matrices spanning scales $10^{-40}$ to $10^{40}$, the shipped
kernel returns **2353 `NaN`s**.

The fix works from the traceless part after balancing by the largest entry:

$$\lambda_- = \left(\tfrac{\mathrm{tr}'}{2} - \mathrm{hypot}(\delta, |b'|)\right)\cdot\mathrm{scale},
\qquad \delta = \tfrac{d'_0 - d'_1}{2}$$

For the eigenvector, note that $T + \mathrm{rad}\,I =
\begin{pmatrix}\delta + \mathrm{rad} & \bar b\\ b & \mathrm{rad} - \delta\end{pmatrix}$ is singular,
so **either** row yields the null vector: row 1 gives $[-\bar b,\ \delta + \mathrm{rad}]$ and row 2
gives $[\mathrm{rad} - \delta,\ -b]$. These are parallel, but each cancels when its own pivot is
small, so select on the sign of $\delta$ — take row 1 when $\delta \ge 0$ and row 2 otherwise. That
way $\delta$ is never subtracted from a nearly equal $\mathrm{rad}$. Guard $\mathrm{rad} = 0$
(a multiple of the identity) by returning an arbitrary unit vector.

Results over the same 40000 matrices:

| | `NaN` | worst error | worse than 1e-10 |
|---|---|---|---|
| current | **2353** | 1.3e-12 | 0 |
| fixed | 0 | 9.3e-16 | 0 |

Balancing is what rescues the extreme scales specifically: at $10^{-160}$ the unbalanced form still
carries `4.9e-02` relative error and at $10^{160}$ it returns `NaN`, whereas the balanced form is
exact on both. Cost is 5.20 µs → 5.38 µs per call, 44 ms → 56 ms to compile.

**A caution, from getting this wrong myself.** My first version of this fix had the row-2 branch as
$[\mathrm{rad} - \delta,\ +b]$ instead of $-b$. It passed every test I had, because those tests
happened to only generate $\delta > 0$ inputs, which never reach that branch — and the end-to-end
solver calls `eigenpair_2x2` exactly once, in `body_iter1`, so a wrong eigenvector there is quietly
repaired by subsequent iterations. Forcing $\delta < 0$ exposed it immediately at **4000/4000
failures**, residual `3.8e-01`. Any replacement for this function must be tested on both signs of
$\delta$ separately; aggregate random sampling and end-to-end convergence both hide it.

## Iteration defects

### I4. `reltol` sign error (line 329)

Upstream `jax.experimental.sparse.linalg.lobpcg_standard:218` computes

```python
reltol = jnp.linalg.norm(AX, ord=2, axis=0) + theta[:k]
```

Line 329 of this file has `- theta`, while the explanatory comment block above it is copied
verbatim from upstream. For a positive-definite operator $\|Ax\| \approx \theta$, so the difference
is pure cancellation noise; it was measured going **negative** (`-2.3e-07`), which makes
`converged = norm_rnext < tol * reltol` unsatisfiable. The solver silently never converges and
always burns `maxiter` iterations.

Fixing this sign alone is a **24–33× iteration reduction**: 2000 → 83 at shift $10^3$, 2000 → 60 at
$10^6$.

Use `+ jnp.abs(theta)` rather than upstream's `+ theta`. `theta` is negative for a
negative-definite operator — the normal situation for a Hamiltonian ground state — where upstream's
own form would cancel instead.

### I5. Loss of $x$/$y$ orthogonality (line 302)

> **Severity retracted.** The guard is in place (`_reorthogonalize`) and stays, but this item is not
> independently reachable: neutering that function leaves the full test suite green, and dense
> operators at shifts 1e9–1e12 return a $\theta$ matching `eigvalsh` to 4e-15 with or without it. The
> 1.0 measurement below needed I4's 2000-iteration runs to develop; at the 8–46 iterations the fixed
> solver takes, the drift never accumulates. So it could not be pinned by a test — recorded in
> `_reorthogonalize`'s docstring so the passing suite is not read as evidence the guard is dead code.

Line 302 builds

```python
tmp_t = tmp_s * (kappa[0] / norm_s) - xcurr * norm_s
```

Near convergence $\kappa_0 \to 1$ and $|s| \to 0$, so this is a difference of two quantities both
nearly parallel to $x$. Catastrophic cancellation lets $y$ drift into $x$. Measured worst
$|\langle x | y \rangle|$ over 40 iterations:

| shift | worst $\lvert\langle x\vert y\rangle\rvert$ |
|---|---|
| 0 | 1.3e-16 |
| $10^6$ | 2.5e-12 |
| $10^9$ | **1.0** |

At $10^9$ the two vectors become parallel and the nominally 3-dimensional search space collapses.

The consequence is worse than a crash, and is the most serious finding in this review.
`rayleigh_ritz` solves a **standard** (non-generalized) 3×3 eigenproblem, which presumes an
orthonormal basis. Once the basis is not orthonormal it returns a $\theta$ **below the true minimum
eigenvalue** — impossible for a genuine Rayleigh quotient, so the failure is silently wrong rather
than merely imprecise. Directly observed: $\theta_{\rm std} = 602348154$ against a true minimum of
$999999972$, i.e. the solver reports an energy far beneath the actual ground state. A caller
checking only "did it return a finite number" would accept this.

Fix: re-orthogonalize `tmp_t` against the new `xnext` twice before normalizing.

### I6. `_project_out` does not return a unit vector (line 421)

The routine renormalizes, then subtracts the basis again **without** renormalizing, then zeroes the
result if the norm fell below 0.99. So its contract is $\|p\| \ge 0.99$, not $\|p\| = 1$.

Feeding a short $p$ into a standard eigensolve scales `sas[2,2]` by $|p|^2$. For a large positive
shift that is a spuriously **low** diagonal, so Rayleigh–Ritz selects the $p$ direction and returns
a $\theta$ far below the true minimum. At shift $10^9$, a $|p|$ of 0.999 displaced $\theta$ by
$2\times10^6$:

| $\lvert p\rvert$ | `sas[2,2]` | $\theta$ from standard `eigh` | below true min? |
|---|---|---|---|
| 1.0000 | 1.000000e+09 | 999999998.054387 | no |
| 0.9990 | 9.980010e+08 | 998001001.915311 | **yes** |
| 0.9950 | 9.900250e+08 | 990025001.900006 | **yes** |
| 0.9900 | 9.801000e+08 | 980100001.880957 | **yes** |

Fix: renormalize $p$ before Rayleigh–Ritz. (Equivalently, solve the generalized problem against the
Gram matrix — but renormalizing is far cheaper and sufficient once I5 is fixed.)

### I7. Zeroed $P$ produces `NaN`

When `_project_out` deliberately returns exactly zero (line 421, the documented guard against
re-introducing $\{x, y\}$ components), row and column 2 of `sas` vanish. For a positive-definite
$A$ that zero diagonal is then the **smallest** eigenvalue, so Rayleigh–Ritz selects the null
direction, giving `kappa = [0, 0, 1]`, hence `norm_s == 0` and `xnext = 0/0`. Traced at shift
$10^9$, iteration 18.

Fix: mask that diagonal so it cannot be selected, and treat the condition as convergence — a zeroed
$P$ means $\{x, y\}$ already spans the residual, so no new search direction exists and further
iterations cannot lower $\theta$.

**Open design question.** Treating this as converged is a judgement call; the alternative is to
restart with a fresh random $p$. Which is preferable depends on the `sqd.py` use case and has not
been settled.

## S1. Redundant matrix-vector product

`body` calls `rayleigh_ritz(xcurr, ycurr, tmp_p)` (line 298), and `compute_sas` computes
`matvec(xcurr)` — but `xcurr` is the previous iteration's `xnext`, whose `axnext` was already
computed on line 306. Verified by counting traced invocations against the shipped code: **8 matvec
calls per trace** (1 in `body_iter0`, 3 in `body_iter1`, 4 in `body`), i.e. **4 per iteration where
3 suffice**.

Threading `ax` through the loop carry:

| | iterations | wall clock ($n = 3000$) |
|---|---|---|
| current | 343 | 7438 ms |
| `ax` reused | 343 | **5710 ms** |

Bit-identical eigenvalues and eigenvectors, 23% faster. For the matrix-free `sqd.py` path, where
`matvec` dominates and is the entire reason this specialization exists, this is the full 25% of
iteration cost.

## API and usability gaps

**All five are fixed.** `ground_locg` now returns `(eigval, eigvec, niter, converged)` and forwards
`debug`; `maxiter=0` reports the seed Rayleigh quotient; `tol` derives from the promoted operator
dtype; and the `xinit` normalization is guarded. The descriptions below are the pre-fix behaviour,
kept because each records a measured way the old surface misled a caller.

These are not numerical defects — they surfaced while instrumenting the solver and are recorded
because each one can mislead a caller. Severity is lower than I1–I7 and none of them require the
change set above.

### A1. No convergence flag in the return value (line 136)

`ground_locg` returns `(eigval, eigvec, niter)`. There is no boolean saying whether the residual
test actually fired, so a caller cannot distinguish "converged" from "gave up at `maxiter`". The
only available signal is `niter == maxiter`, and that is ambiguous: a run that converges on exactly
the last permitted iteration is indistinguishable from one that ran out of budget.

Measured with `maxiter=5`, the function returns a confident-looking eigenvalue whose true relative
residual is `1.1e-01`. Given I4 above — which makes *every* run silently exhaust `maxiter` — this
is what allowed that bug to stay invisible. Returning the `converged` boolean already present in
the loop state (`state[1]`) would cost nothing.

### A2. `maxiter=0` returns `0.0` as the eigenvalue (line 358)

The loop state is initialized with `theta = 0.` (line 358) and `eigval = state[2]` is returned
verbatim (line 373). With `maxiter=0` the `while_loop` body never runs, so the function returns
`eigval = 0.0` — not a Rayleigh quotient of anything, just the initializer. For a matrix whose true
lowest eigenvalue is $-21.35$, `ground_locg(H, x0, maxiter=0)` reports `0.000000` with a normalized
eigenvector and `niter=0`.

The `body_iter1` step has already computed a valid `theta` by that point, so seeding the state with
it instead of `0.` makes the degenerate case return something meaningful.

### A3. `tol=None` derives $\epsilon$ from `xinit.dtype`, not the operator's (line 209)

```python
tol = float(jnp.finfo(xinit.dtype).eps)
```

A caller who passes a float32 initial vector for a complex128 operator silently gets
`tol = 1.2e-07` instead of `2.2e-16`. Measured effect on an otherwise identical problem:

| `tol` | iterations | eigenvalue error | residual |
|---|---|---|---|
| `eps(complex128)` | 56 | 3.6e-16 | 1.6e-13 |
| `eps(float32)` | 18 | 2.7e-08 | 1.2e-04 |

Eight orders of magnitude of accuracy, quietly traded away by the dtype of the *starting guess*.
Deriving `tol` from the matvec output dtype, or from `jnp.result_type` of both, would be more
predictable. (The integer `xinit` one-hot path is safe: the conversion on line 205 precedes the
`finfo` call, so `finfo` never sees an integer dtype.)

### A4. Zero `xinit` yields `NaN` with no diagnostic (line 346)

`xinit /= jnp.linalg.norm(xinit)` has no zero guard, unlike the analogous normalizations elsewhere
in the file which all use `jnp.where(norm == 0., 1., norm)`. A zero initial vector returns
`eigval = nan` and a non-finite eigenvector. Cheap to guard for consistency with the rest of the
module.

### A5. `debug=True` is unreachable from the public entry point

`debug` exists on `_ground_locg_matrix` and `_ground_locg_callable` but not on `ground_locg`
(line 128), which does not forward it — `ground_locg(..., debug=True)` raises `TypeError`. The
diagnostics machinery is therefore reachable only by calling the private functions directly, which
is how all instrumentation in this investigation was done.

Two related notes on that path, for anyone using it: the `debug` branch uses `jax.lax.scan` with
`length=maxiter` (lines 360–361) rather than the `while_loop`, so it has **no early exit** and
always runs the full `maxiter` iterations — a `maxiter=600` debug run reports `niter=600` where the
production path converges at 82. And because `scan` keeps iterating past convergence, the tail
diagnostic rows are post-convergence noise rather than part of the solve.

## Recommended change set

**All of the below has been implemented.** Kept as the record of what was proposed and in what
order; `8fa6be2` did items 1–6 and A1 together rather than in the two commits suggested here.

Ordered by payoff. I4 and I5 are the highest-value items and are independent of the kernel work.

1. **I4** — flip the `reltol` sign to `+ jnp.abs(theta)`. One line; 24–33× fewer iterations.
2. **I5** — re-orthogonalize `tmp_t` against `xnext`. Two lines; eliminates the
   silently-below-minimum failure mode.
3. **I1–I3** — rewrite `eigenpair_3x3` with shift/scale balancing, best-of-three cross products
   with rank-1 and rank-0 fallbacks, and a Rayleigh-quotient polish.
4. **I1–I2 in `eigenpair_2x2`** — balance by the largest entry, solve from the traceless part, and
   select the null-vector row on the sign of $\delta$. Test both signs of $\delta$ explicitly.
5. **I6, I7** — renormalize $p$; mask and stop on zeroed $P$.
6. **S1** — thread `ax` through the loop carry.

Suggested split into two commits: robustness (I1–I7) separately from the S1 optimization, since I5
changes convergence behaviour and merits isolated scrutiny, whereas S1 is provably output-identical.

**A1 is worth folding into the first commit**, out of order with its low severity. Returning the
`converged` flag that already exists in the loop state is a two-line change, and it is the reason I4
could hide for as long as it did: with no convergence signal, a solver that never converges looks
exactly like one that does. Fixing the numerics without exposing that flag leaves the next such
regression equally invisible.

## Scope and limitations of this investigation

**Superseded.** At the time of writing there was no pytest suite in this repository, so everything
below was verified with throwaway `uv run python` scripts under `/tmp`; no tests were left behind and
no file under `rqutils/` was modified *by this investigation*. Both conditions have since changed:
the module was rewritten in `8fa6be2` and `tests/test_ground_locg.py` now covers most of these items
(and is deliberately organized by defect, following this document's closing warning). The numbers
below are measurements from those throwaway scripts, not projections — that much still holds, and is
the reason to keep this file.

Verified: analytic-kernel accuracy against `numpy.linalg.eigh` over 20000 random Hermitian matrices
(complex and real, scales $10^{-160}$ to $10^{300}$, exact and near degeneracies, rank-deficient and
block-structured inputs); eigenvector correctness by overlap with the true lowest eigenspace,
handling degeneracy via the spectral gap; per-iteration tracing of $\theta$, $\kappa$, `sas`,
$|r|$, `reltol`, and $|\langle x|y\rangle|$; matvec counts by wrapping the callable and counting
traced invocations; end-to-end solves at $n = 120$–$3000$ with shifts $0$ to $\pm 10^{14}$.

Checked and found **sound** (recorded so the boundary of the audit is explicit, and so nobody
re-derives these):

- **Returned $\theta$ is self-consistent with the returned $x$.** `theta` carried in the loop state
  comes from the Rayleigh–Ritz minimizer over $\{x, y, p\}$, which is `xnext`; the returned
  eigenvalue matches $x^\dagger A x$ to `0.00e+00`. There is no off-by-one between the reported
  eigenvalue and the reported vector.
- **Integer `xinit`** (the one-hot convenience path) works for both index 0 and interior indices.
- **Non-normalized `xinit`.** Line 346 normalizes, so `|xinit|` scaled by $10^{\pm 8}$ gives
  bit-identical results.
- **`xinit` is not mutated.** Despite the in-place-looking `xinit /= ...` on line 346, JAX arrays
  are immutable and the statement rebinds; the caller's array is untouched.
- **Real (non-complex) operator dtype** solves correctly and returns `float64`.
- **`xinit` exactly orthogonal to the true ground state.** The documented precondition is a
  non-vanishing overlap with $v_0$; starting exactly at $v_1$ still converged to $v_0$ (overlap
  `1.00e+00`) because rounding reintroduces a component. This is luck, not a guarantee — the
  precondition still stands — but it is not a latent failure in practice.
- **The `resid=inf` entry at scale $10^{300}$** in early kernel testing was a test artifact:
  forming `m @ u` to *check* the result overflows when $|m| \sim 10^{300}$. The kernel itself
  returns the correct eigenvalue there.

Not verified:

- **Multi-device / sharded execution.** All measurements are single-device. The `out_sharding`
  contract described in `CLAUDE.md` was not exercised, and the added re-orthogonalization in I5
  introduces two new inner products per iteration whose sharding behaviour is untested.
- **float32 / complex64.** The fixed kernel is dtype-generic and was smoke-tested there
  (relative error `3.6e-08` in float32, as expected for the precision), but the iteration-level
  fixes were only tested in float64. The `jax_enable_x64` path is the documented one.
- **Tight clusters.** Cases with relative gap below $\approx 10^{-10}$ still degrade: eigenvalue
  error stays near $10^{-10}$ while the eigenvector residual worsens. This is the expected
  gap-dependent conditioning of LOBPCG, not a defect, but it means "0 / 90 failures" above is
  reported against a sweep whose cluster gaps range $10^{-6}$ to $10^{-2}$. An earlier sweep with
  gaps down to $10^{-12}$ left 3 of 60 cases outside tolerance.
- **Real `sqd.py` workloads.** All operators here were dense random matrices, not projected Pauli
  sums. The interaction with `cache_level` matvec kernels is untested.

A methodological warning that cost me a wrong candidate: **aggregate random sampling does not
exercise data-dependent branches evenly.** A `jnp.where` selecting on the sign of a quantity is
invisible to a sweep that happens to generate one sign, and end-to-end convergence testing hides
errors in any routine the solver calls only during setup — `eigenpair_2x2` runs exactly once, in
`body_iter1`. Both of my safety nets missed a sign error that a two-line targeted test caught at
4000/4000 failures. Kernel changes here need per-branch tests, not just volume.

One further process note, since it affected intermediate conclusions: `_ground_locg_matrix` and
`_ground_locg_callable` are `@jax.jit`-decorated, so monkey-patching `eigenpair_2x2` /
`eigenpair_3x3` after the first trace silently reuses the **old** kernels. An early A/B comparison
was invalidated this way and showed both variants failing identically. All figures in this document
come from runs in separate subprocesses with patching applied before any tracing.
