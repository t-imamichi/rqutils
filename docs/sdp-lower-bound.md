# A level-1 SDP lower bound on `lambda_min`, for use as a preconditioner shift

**Status: spike complete, formulation verified, integration not recommended.** Probes are
`examples/scaling/poc_sdp_lower_bound.py` (level 1) and `examples/scaling/poc_sdp_level2.py`
(level 2, the chain-structured variants, and the 100-qubit scaling arm). Everything below is measured
on this machine unless labelled otherwise.

## Why this was asked

`docs/rqutils-precond-request.md` records a shifted-Jacobi preconditioner for `ground_locg` whose
blocker is the *shift*: the operator `sqd` builds is indefinite, so `1/diag(H)` flips sign and a
shift `sigma <= lambda_min` is needed to make `H - sigma*I` positive definite. Every cheap estimator
for `sigma` has been rejected as too loose -- a coefficient-sum bound at 4.14-8.14x over-shift, a
structural row-sum bound at 16.05-20.49x, and even the *unattainable* bound computed from the
assembled matrix at 5.03-6.29x. That doc's conclusion was that "the whole row-sum family is
structurally incapable of a tight shift on a sampled subspace".

An SDP moment relaxation is a different family. It bounds from below by construction, for a reason
worth stating because it is the opposite of `rqutils/product.py`:

* `product.py` **restricts** to product states (`x^2+y^2+z^2 = 1` per qubit), a subset of physical
  states. Its optimum is therefore an **upper** bound on `lambda_min`, and its `Solution.lower_bound`
  field is SCIP's branch-and-bound dual bound *of that nonconvex product-state problem* -- it bounds
  the product-state optimum, and its relation to `lambda_min` is not fixed. **It is not a lower bound
  on the eigenvalue and must not be used as a shift.** Measured, same XXZ family, `tol=1e-4`:

  | n | `lambda_min` | `Solution.eigval` | `Solution.lower_bound` | `lower_bound <= lambda_min`? |
  | --- | --- | --- | --- | --- |
  | 4 | −1.484558 | −1.152680 | −1.152789 | **False** |
  | 6 | −2.367203 | −1.784568 | −1.784723 | **False** |
  | 8 | −3.196779 | −2.411077 | −2.411296 | **False** |

  It sits *above* `lambda_min` at every size, so `H - lower_bound*I` is still indefinite -- the exact
  failure the shift exists to prevent. Note also how close `lower_bound` is to `eigval` (~1e-4, the
  requested gap): it is converging to the product-state optimum, confirming what it actually bounds.
* A moment relaxation **relaxes** to matrices satisfying only *some* of the constraints a real
  density matrix satisfies -- a superset. Its optimum is a genuine **lower** bound.

## The formulation

Write `H = sum_k c_k P_k` over Pauli monomials of degree <= 2. Level-1 monomial basis
`B = [I, X_0..X_{n-1}, Y_0.., Z_0..]`, so `m = 1 + 3n`. Moment matrix `M_ij = <B_i B_j>` (each `B_i`
is a Hermitian Pauli, so `B_i^dag = B_i`). Constraints: `M >= 0`, `M_00 = 1`, objective
`sum_k c_k <P_k>`.

Two encoding details, both of which fail *silently* if got wrong -- they return a plausible finite
number rather than raising, which is the failure mode this repo keeps hitting:

* **The moment matrix is complex Hermitian, not real symmetric.** Same-qubit products are
  anti-commuting: `X_0 Y_0 = i Z_0`, so `M_ij = i <Z_0>` is purely imaginary. Clarabel's
  `PSDTriangleConeT` is a *real* cone, so `M >= 0` is encoded as the exact `2m x 2m` real embedding
  `[[Re M, -Im M], [Im M, Re M]] >= 0`. Constraining `Re M >= 0` alone is also a valid bound and is
  `m x m`, but it is a *weaker* relaxation -- it discards the same-qubit entries. The embedding was
  chosen so the number reported here is level 1 and not a pessimistic variant of it. At these sizes
  the 4x cone cost is irrelevant (`n=14` is a 86x86 cone).
* **Clarabel's sign convention and scaling.** The form is `A x + s = b`, `s in K`. To put the moment
  matrix *itself* in the cone you need `s = Mre`, so with `b = 0` that forces `A = -(coeffs of Mre)`.
  Get the sign backwards and you constrain `-M >= 0`, which silently yields an *upper* bound.
  Separately, `PSDTriangleConeT` takes the upper triangle in column-major order with off-diagonals
  scaled by `sqrt(2)`; omitting the scale still gives a valid cone but a different matrix, so the
  bound comes out wrong-but-finite.

Both are pinned by a self-check that does not depend on any of this being right: for **one qubit**
the level-1 feasible set *is* the Bloch ball, so the relaxation is exact and must reproduce a closed
form. `H = Z` must give exactly `-1`; `H = X + Z` must give exactly `-sqrt(2)`. Both hold to `1e-6`.
The algebra table is separately checked entry-by-entry against explicit 2x2 matrices.

The Pauli-algebra linking constraints are *structural* rather than a constraint block: each entry is
emitted as `+-1` or `+-i` times a single moment **variable**, so two entries mapping to the same
Pauli string reference the same variable and are equal by construction. Hence only one equality row
(`<I> = 1`).

## Measured: the bound is valid and ~1.45x loose on `H`

Open XXZ + transverse field, the builder from `rqutils-precond-request.md` line 440
(`J=1, delta=0.5, Bx=By=0.5`, so odd-Y terms are present). `lambda_min` from dense `eigvalsh` --
an independent reference, built by Kronecker products and sharing no code path with the SDP.

| n | cone | nvar | sigma | lambda_min | trivial `-sum|c_k|` | sigma/lambda_min | slack recovered | t_sdp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | 26 | 67 | −1.955591 | −1.484558 | −3.8750 | 1.317 | 0.803 | 0.04 s |
| 6 | 38 | 154 | −3.205591 | −2.367203 | −6.1250 | 1.354 | 0.777 | 0.18 s |
| 8 | 50 | 277 | −4.455591 | −3.196779 | −8.3750 | 1.394 | 0.757 | 0.70 s |
| 10 | 62 | 436 | −5.705591 | −4.000791 | −10.6250 | 1.426 | 0.743 | 1.45 s |
| 12 | 74 | 631 | −6.955592 | −4.791102 | −12.8750 | 1.452 | 0.732 | 2.98 s |

`sigma <= lambda_min` holds at every size (the primary correctness assertion; the script exits
nonzero on violation). "Slack recovered" is `(sigma - trivial)/(lambda_min - trivial)`: the fraction
of the free bound's slack that the SDP closes, 0.73-0.80.

## The structural finding: sigma is *exactly linear in n*, with a closed form

This is the result that governs whether the approach is worth integrating. Across `n = 4..14` the
increments are `-0.625000` per site to six decimals, and a linear fit has max residual **5.2e-08**:

| couplings | fit | max linear residual | single-bond `lambda_min` |
| --- | --- | --- | --- |
| `J=1, delta=0.5, Bx=By=0.5` | `sigma = -0.625000 n + 0.544409` | 5.2e−08 | −0.625000 |
| `J=1, delta=1, B=0` | `sigma = -0.750000 n + 0.750000` | 7.7e−09 | −0.750000 |
| `J=0.7, delta=0.3, Bx=0.9, By=0.2` | `sigma = -0.463397 n + 0.190459` | 3.2e−06 | −0.425000 |

In the first two rows the slope **equals the single-bond `lambda_min` exactly**, so on the
`n`-dependence level 1 is optimizing one bond at a time. It is *not* naive decoupling -- a bound that
also decoupled the field would give slope −0.9786 for row 1, and measured level-1 beats a fully
decoupled per-term bound by 0.64 (n=2) to 2.75 (n=8) -- the field is handled jointly, showing up in
the intercept. The third row's slope differs from the bare bond minimum, so with an asymmetric field
the field does mix into the per-site rate.

**Consequence: the conic solve is buying a quantity with a closed form.** For a uniform chain, two
solves at different `n` determine slope and intercept, and every larger `n` follows by arithmetic.
Paying 3-14 s of Clarabel per instance to recover a linear function is the wrong trade if the
coupling set is fixed; the SDP's value is in *establishing* the constant for a new coupling family,
not in per-instance evaluation.

## Measured against the operator a preconditioner actually sees

The decisive arm, and the one that killed prior candidates: `ground_locg` is handed
`hproj(H, subspace)`, not `H`. By the variational principle `lambda_min(H) <= lambda_min(P^dag H P)`,
so even a *perfect* bound on `H` over-shifts the projected operator. Over-shift below is
`(lambda_min_proj - sigma)` in units of the projected spectral width; identical to 4 decimals with
and without `JAX_ENABLE_X64=1`.

| n | dim | seed | sigma | `lambda_min` proj | min diag | gap | width | gap/width |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 10 | 300 | 0 | −5.7056 | −2.3510 | −0.8750 | 3.3546 | 4.5932 | 0.730 |
| 10 | 300 | 1 | −5.7056 | −2.6267 | −1.1250 | 3.0789 | 4.8194 | 0.639 |
| 10 | 300 | 2 | −5.7056 | −2.1708 | −0.8750 | 3.5348 | 4.6357 | 0.763 |
| 12 | 800 | 0 | −6.9556 | −2.3787 | −1.1250 | 4.5769 | 4.4663 | 1.025 |
| 12 | 800 | 1 | −6.9556 | −2.2806 | −1.1250 | 4.6750 | 4.4294 | 1.055 |
| 12 | 800 | 2 | −6.9556 | −2.5643 | −1.3750 | 4.3913 | 4.7625 | 0.922 |

**0.64-1.06x over-shift, against 4.14-8.14x for the rejected coefficient-sum bound and 5.03-6.29x
for the bound previously labelled unattainable.** On this normalization the SDP shift is roughly
5-8x tighter than anything previously available, and it needs only the Pauli list -- no assembled
matrix, no matvec.

**But the trend is adverse and is the reason to stop here.** `sigma` is linear in `n` while the
projected `lambda_min` barely moves with `n` at these subspace sizes (−2.35 at n=10, −2.38 at n=12),
because a random subspace of a few hundred to a few thousand states does not reach the true ground
state. So `gap/width` grows: 0.73 at n=10, 1.03 at n=12. Extrapolating the fitted `sigma` to n=18 --
where the 1.79x preconditioner figure was measured -- gives `sigma ~ -10.7` against a projected
`lambda_min` still near −2.5, i.e. `gap/width ~ 1.8-1.9`. The bound is tight on `H` and loosens on a
sampled subspace exactly as the prior analysis predicted, for the same reason: **projection discards
most couplings, and no bound derived from `H`'s full structure can know which ones.**

## Recommendation

1. **Do not use `product.py`'s `Solution.lower_bound` as a shift.** It is not a lower bound on
   `lambda_min`. Worth a docstring note there regardless of what happens to this work.
2. **Do not integrate the level-1 SDP as a per-instance shift oracle.** It computes a closed-form
   linear quantity at 3-14 s per solve, and on the projected operator it lands at 0.64-1.06x
   over-shift, degrading with `n`. Better than everything previously measured, but the pattern of
   prior results says an over-shift near or above 1x has not translated into iteration wins.
3. **The cheap way to capture its value**, if a shift is wanted: fit `sigma = a*n + b` once per
   coupling family from two SDP solves at small `n`, then evaluate arithmetically. That is `O(1)` and
   gives the same number to 1e-8.
4. **Before any of that, measure iterations.** Every rejected candidate in
   `rqutils-precond-request.md` returned the *correct energy* while silently pessimizing convergence,
   and two were rejected on iteration counts that a tightness figure did not predict. The next step
   for this line is a `ground_locg` A/B with `sigma` from item 3 -- not more SDP work. If that shows
   no gain, the whole shifted-Jacobi direction closes and level 2 is not worth attempting.

## Not measured

* **Validity at n >= 14** against dense `eigvalsh`. Two runs were started and had not finished; the
  `n=16` reference alone is a 65536^2 complex diagonalization. Validity is established at n <= 12 plus
  the exact 1-qubit and 2-qubit checks, and nothing about the construction is `n`-dependent, but the
  larger sizes are unrun rather than confirmed.
* ~~**Anything about iteration counts.**~~ Every number in *this* section is bound tightness. Iteration
  counts **were** subsequently measured through the real hook -- see "Measured: iteration counts
  through the real `ground_locg` hook" below, which is the section that settles the question.

Level 2 and iteration counts **were** both subsequently measured; see the sections below. Item 4 of
the recommendation above ("measure iterations") has been carried out, and its conclusion held: the
gain is real but is not attributable to the SDP.
## Level 2 vs level 1, and whether level 2 reaches 100 qubits

Probe: `examples/scaling/poc_sdp_level2.py`. Four variants, all from one generalized builder so the
verified encoding (real embedding, `sqrt(2)` scaling, `A`-sign convention) lives in one place:

| variant | monomial set | cone (after real embedding) |
| --- | --- | --- |
| `level1` | `{I} u {P_i}` | `2(1+3n)`, O(n) |
| `level2` | `level1 u {P_i Q_j : i<j}` | `2(1+3n+9C(n,2))`, O(n^2) |
| `chain2` | `level1 u {P_i Q_{i+1}}` | `2(1+3n+9(n-1))`, O(n) |
| `chain2blk` | same as `chain2`, but one PSD cone **per bond** | `(n-1)` cones of **fixed size 32** |

`chain2blk` is the sparse/chordal Lasserre structure: requiring each per-bond principal submatrix to
be PSD is weaker than requiring the whole matrix to be PSD, so it is a valid but looser bound. Both
forms are measured rather than assuming the cheap one suffices.

**New self-check, stronger than level 1's.** Over 2 qubits the level-2 monomial set spans the entire
16-dimensional operator algebra, so the relaxation is not a relaxation -- it must reproduce
`eigvalsh` exactly. Asserted for `ZZ` and for an asymmetric-coefficient case (so a bug preserving
symmetry cannot hide). Both hold to `1e-5`.

### Answer 1: level 2 is dramatically tighter than level 1

Open XXZ + transverse field, `J=1, delta=0.5, Bx=By=0.5`. `gain` is the fraction of level 1's
remaining slack that the richer relaxation closes, `(sigma - sigma_L1)/(lambda_min - sigma_L1)`.

| n | variant | cone | nvar | sigma | lambda_min | ratio | gain | t_sdp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | level1 | 26 | 67 | −1.955591 | −1.484558 | 1.317 | — | 0.03 s |
| 4 | chain2 | 80 | 256 | −1.484558 | −1.484558 | **1.000** | **1.000** | 3.60 s |
| 4 | chain2blk | 32x3 | 40 | −1.875000 | −1.484558 | 1.263 | 0.171 | 0.19 s |
| 4 | level2 | 134 | 256 | −1.484558 | −1.484558 | **1.000** | **1.000** | 29.69 s |
| 6 | level1 | 38 | 154 | −3.205591 | −2.367203 | 1.354 | — | 0.18 s |
| 6 | chain2 | 128 | 1072 | −2.369642 | −2.367203 | **1.001** | **0.997** | 40.61 s |
| 6 | chain2blk | 32x5 | 64 | −3.125000 | −2.367203 | 1.320 | 0.096 | 0.40 s |

Two findings, both sharp:

* **Level 2 is essentially exact on this family** -- ratio 1.000 at n=4, 1.001 at n=6, against level
  1's 1.317-1.354. So yes, level 2 is much better: it closes 99.7-100% of the gap.
* **The chain restriction costs nothing in accuracy, and full level 2 buys nothing over it.**
  `chain2` matches `level2` to 6 decimals at n=4 while using a smaller cone and 8x less time. For a
  nearest-neighbour Hamiltonian the distant-pair monomials in full level 2 are dead weight, exactly
  as the structure argues.
* **But the blocked decomposition loses almost all of it**: gain 0.171 at n=4, *falling* to 0.096 at
  n=6. Per-block PSD is a much weaker condition than global PSD, and the loss worsens with n.

### Answer 2: no, level 2 does not reach 100 qubits -- and the variant that does is not better

Cone sizes and memory, by counting alone (`e^2 * 8` bytes to store the moment matrix **once**; an
interior-point solver needs 3-5x that for workspace and the Schur complement):

| n | level1 cone | level2 cone | chain2 cone | chain2blk | level2 store | chain2 store |
| --- | --- | --- | --- | --- | --- | --- |
| 12 | 74 | 1262 | 272 | 32x11 | 0.01 GB | 0.6 MB |
| 40 | 242 | 14282 | 944 | 32x39 | 1.63 GB | 7.1 MB |
| 100 | 602 | **89702** | 2384 | 32x99 | **64.4 GB** | 45.5 MB |

Full level 2 at 100 qubits needs 64 GB for one copy of the moment matrix, so 200-300 GB in a real
solve, with interior-point cost ~O(cone^3). Structurally out of reach; this is not a "slow but
doable". Measured, full level 2 already costs 29.7 s at **n=4** and 40.6 s at n=6 for the equivalent
`chain2`.

`chain2blk` **does** reach 100 qubits, with cost linear in n as designed:

| n | blocks | cone | nvar | sigma | sigma/n | t_sdp | status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 10 | 9 | 32 | 112 | −5.625000 | −0.562500 | 0.81 s | Solved |
| 20 | 19 | 32 | 232 | −11.875000 | −0.593750 | 1.95 s | Solved |
| 40 | 39 | 32 | 472 | −24.374998 | −0.609375 | 4.80 s | **AlmostSolved** |
| 60 | 59 | 32 | 712 | −36.875000 | −0.614583 | 6.58 s | **AlmostSolved** |
| 80 | 79 | 32 | 952 | −49.374998 | −0.617187 | 8.94 s | **AlmostSolved** |
| 100 | 99 | 32 | 1192 | −61.874999 | −0.618750 | 10.55 s | **AlmostSolved** |

**But it converges to level 1.** Fitting the n=20..100 rows gives
`sigma = -0.625000 n + 0.625000` (max residual 1.2e−06), against level 1's
`sigma = -0.625000 n + 0.544409`. **The slopes differ by 1.0e−08 -- solver precision -- and only the
intercept differs, by 0.0806, a constant.** So the relative advantage decays as `1/n`:

| n | `chain2blk` | `level1` | relative gain |
| --- | --- | --- | --- |
| 20 | −11.8750 | −11.9556 | 0.674% |
| 60 | −36.8750 | −36.9556 | 0.218% |
| 100 | −61.8750 | −61.9556 | **0.130%** |
| 1000 | −624.3750 | −624.4556 | 0.013% |

At n=100 that is a 0.13% better bound for 10.5 s of conic solving, against a closed form that costs
nothing. Note the sign of the difference: `chain2blk` is the *tighter* of the two (−61.875 is closer
to zero, hence closer to `lambda_min`), so the comparison is fair -- it wins, negligibly.

Two further cautions on the large-n rows. `AlmostSolved` from n=40 on means Clarabel reached only
reduced accuracy -- for a *bound* that is not cosmetic, since an inexact dual value is not a
rigorous certificate; a trustworthy number would need the dual checked explicitly or tolerances
tightened. And validity at these sizes rests on the construction plus the small-n checks, not on any
direct verification: `2^100` admits no dense reference. Note `chain2` reported `AlmostSolved` even at
n=4 despite agreeing with `eigvalsh` to 6 decimals -- accuracy and certification are separate things
here.

### What this means for the shift

The useful accuracy sits in `chain2` (monolithic), which is near-exact but whose solve time grew
3.60 s -> 40.61 s from n=4 to n=6 -- roughly 11-12x per two qubits, far steeper than its O(n) cone
suggests, because a near-exact optimum sits on the cone boundary where interior-point methods
struggle. **The n=8 wall is measured, not extrapolated: both `chain2` and full `level2` at n=8 were
killed after ~15 minutes without returning**, in two independent runs. On that growth rate n=12 is
hours and n=16-18 -- the range where the preconditioner work was measured -- is out of reach, let
alone 100.

So the two variants split cleanly, and neither is what a shift oracle wants:

* **Accurate and unaffordable**: `chain2`/`level2`, ratio 1.000-1.001, dead by ~n=10.
* **Affordable and asymptotically pointless**: `chain2blk`, reaches n=100 in 10.5 s but converges to
  level 1's own bound, beating it by a constant 0.081.

Combined with the level-1 finding that `sigma` is exactly linear in n with a closed form, the
recommendation is unchanged and now better supported: **do not build an SDP shift oracle.** If a
shift is wanted, fit `sigma = a*n + b` from two small solves. The one genuinely new option this
opens is different in kind -- `chain2` is near-*exact* at small n, which makes it a candidate for a
cheap high-accuracy **ground-state energy estimate** on small chains, not a preconditioner shift.
## Measured: iteration counts through the real `ground_locg` hook

Probe: `examples/scaling/poc_sdp_iterations.py`. This is the measurement the sections above kept
deferring, and the only one that decides the question, since two candidates in
`rqutils-precond-request.md` returned the correct energy while silently pessimizing convergence.

Protocol copied from the batch that produced the recorded 1.79x: `(n, dim) = (16, 2000), (18, 4000)`,
seeds 0-5, same `x0` (seed 99) across arms, `maxiter=4000`, `tol=1e-10`, every energy checked against
`eigvalsh`. **The harness reproduces that batch exactly** -- its `precond=None` counts are
44, 28, 29, 45, 54, 122 / 35, 19, 34, 37, 24, 46, matching the baseline column of the rejected
candidate's table in `rqutils-precond-request.md` value for value. That agreement is what makes the
rest of this section comparable to the figures already on record.

| n | dim | seed | none | ideal | fitted | mindiag | x_ideal | x_fitted | x_mindiag |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 16 | 2000 | 0 | 44 | 26 | 33 | 33 | 1.69 | 1.33 | 1.33 |
| 16 | 2000 | 1 | 28 | 20 | 23 | 23 | 1.40 | 1.22 | 1.22 |
| 16 | 2000 | 2 | 29 | 18 | 24 | 24 | 1.61 | 1.21 | 1.21 |
| 16 | 2000 | 3 | 45 | 41 | 39 | 40 | 1.10 | 1.15 | 1.12 |
| 16 | 2000 | 4 | 54 | 34 | 41 | 42 | 1.59 | 1.32 | 1.29 |
| 16 | 2000 | 5 | 122 | 85 | 93 | 97 | 1.44 | 1.31 | 1.26 |
| 18 | 4000 | 0 | 35 | 21 | 27 | 27 | 1.67 | 1.30 | 1.30 |
| 18 | 4000 | 1 | 19 | 12 | 15 | 15 | 1.58 | 1.27 | 1.27 |
| 18 | 4000 | 2 | 34 | 20 | 25 | 25 | 1.70 | 1.36 | 1.36 |
| 18 | 4000 | 3 | 37 | 22 | 29 | 28 | 1.68 | 1.28 | 1.32 |
| 18 | 4000 | 4 | 24 | 17 | 19 | 19 | 1.41 | 1.26 | 1.26 |
| 18 | 4000 | 5 | 46 | 25 | 34 | 32 | 1.84 | 1.35 | 1.44 |

| arm | shift | median gain | range | regressions |
| --- | --- | --- | --- | --- |
| `ideal` | `eigvalsh.min() - 0.5` (unattainable) | **1.60x** | 1.10-1.84 | 0/12 |
| `fitted` | `-0.625n + 0.544409` (SDP closed form) | **1.29x** | 1.15-1.36 | 0/12 |
| `mindiag` | `min(diag) - structural B` | 1.28x | 1.12-1.44 | 0/12 |

All 48 solves converged to the `eigvalsh` minimum (max error 4.2e-08).

**On `ideal` reading 1.60x where `rqutils-precond-request.md` records 1.79x: the definitions differ,
and both are right for what they measure.** That doc's reproduction snippet builds
`A = H - (lambda_min - 0.5)I` once and runs *both* arms on `A`, so its denominator is
`precond=None` **on the shifted operator**. This harness uses `precond=None` on unshifted `H`.
Measured side by side on the same 12 instances:

| n | seed | none(H) | none(A) | jacobi(A) | ratio vs H | ratio vs A |
| --- | --- | --- | --- | --- | --- | --- |
| 16 | 0 | 44 | 48 | 26 | 1.69 | 1.85 |
| 16 | 1 | 28 | 31 | 20 | 1.40 | 1.55 |
| 16 | 2 | 29 | 32 | 18 | 1.61 | 1.78 |
| 16 | 3 | 45 | 57 | 41 | 1.10 | 1.39 |
| 16 | 4 | 54 | 59 | 34 | 1.59 | 1.74 |
| 16 | 5 | 122 | 142 | 85 | 1.44 | 1.67 |
| 18 | 0 | 35 | 38 | 21 | 1.67 | 1.81 |
| 18 | 1 | 19 | 22 | 12 | 1.58 | 1.83 |
| 18 | 2 | 34 | 36 | 20 | 1.70 | 1.80 |
| 18 | 3 | 37 | 40 | 22 | 1.68 | 1.82 |
| 18 | 4 | 24 | 27 | 17 | 1.41 | 1.59 |
| 18 | 5 | 46 | 51 | 25 | 1.84 | 2.04 |
| | | | | **median** | **1.60x** | **1.79x** |

Baseline-on-`A` reproduces the recorded figure **exactly -- 1.79x median**, with a range of 1.39-2.04x
against the documented 1.29-2.04x. So the harness is faithful and the discrepancy was entirely the
choice of denominator.

Baseline-on-`A` isolates "what does Jacobi buy at a fixed operator", which is the right question for
evaluating the *hook*, and reproduces the recorded band. Baseline-on-`H` answers "what does this whole
scheme buy against doing nothing", which is the right question for evaluating a *shift oracle*, since
not shifting is part of doing nothing. This section uses baseline-on-`H` throughout for that reason,
which makes every gain quoted here **smaller** than the equivalent figure in the older doc. The two
tables are not in conflict; compare like with like before carrying a number between them.

**So the fitted sigma does work: 1.29x median, no regressions.** That is a real result and the first
positive one in this line -- unlike the `|diag|^-1` candidate (0.30x, 12/12 regressions) it never
makes things worse. But three findings qualify it, and together they say the SDP is not what earned it.

### Half the gain is the shift, not the preconditioner

A control arm the earlier sections lacked: shift the operator by the fitted `sigma` but pass
`precond=None`. Whatever this gains is not attributable to Jacobi. Full 12-instance batch:

| n | seed | none | shift only | fitted (shift+Jacobi) | ideal |
| --- | --- | --- | --- | --- | --- |
| 16 | 0 | 44 | 39 (1.13x) | 33 (1.33x) | 26 (1.69x) |
| 16 | 1 | 28 | 26 (1.08x) | 23 (1.22x) | 20 (1.40x) |
| 16 | 2 | 29 | 27 (1.07x) | 24 (1.21x) | 18 (1.61x) |
| 16 | 3 | 45 | 43 (1.05x) | 39 (1.15x) | 41 (1.10x) |
| 16 | 4 | 54 | 47 (1.15x) | 41 (1.32x) | 34 (1.59x) |
| 16 | 5 | 122 | 108 (1.13x) | 93 (1.31x) | 85 (1.44x) |
| 18 | 0 | 35 | 31 (1.13x) | 27 (1.30x) | 21 (1.67x) |
| 18 | 1 | 19 | 18 (1.06x) | 15 (1.27x) | 12 (1.58x) |
| 18 | 2 | 34 | 30 (1.13x) | 25 (1.36x) | 20 (1.70x) |
| 18 | 3 | 37 | 33 (1.12x) | 29 (1.28x) | 22 (1.68x) |
| 18 | 4 | 24 | 21 (1.14x) | 19 (1.26x) | 17 (1.41x) |
| 18 | 5 | 46 | 40 (1.15x) | 34 (1.35x) | 25 (1.84x) |
| | **median** | | **1.13x** | **1.29x** | **1.60x** |

**The split is 1.13x from the shift and 1.14x from the preconditioner** (1.29/1.13), i.e. very nearly
even -- so the Jacobi multiply, the thing the shift exists to enable, is worth about 14%. At the ideal
shift the same multiply is worth 1.60/1.13 ~ 1.42x on top, so the over-shift costs the preconditioner
roughly two thirds of its effect. That is the cost of using a bound on `H` where the operator is
`hproj(H, subspace)`, expressed in iterations.

Note that in exact arithmetic a shift cannot change LOBPCG's trajectory -- `H - sigma I` has the same
eigenvectors and every Rayleigh quotient moves equally -- so the shift-only column is a
finite-precision effect, presumably the balancing and orthogonalization guards behaving differently on
a definite operator. It is a stable one (1.05-1.15x, 0/12 regressions), but it is not a mechanism to
design against, and it is not what an SDP bound was supposed to contribute.

### The over-shift flattens `M^-1` toward the identity, which is why the rest is lost

The mechanism behind `fitted` reaching only 1.29x where `ideal` reaches 1.60x. `M^-1 = 1/diag(A)`,
and `diag(H)` spans only ~3.0 on these instances, so subtracting a large constant makes `diag(A)`
into `constant +- 1.5`:

| n | seed | arm | shift | diag(A) range | `M^-1` max/min |
| --- | --- | --- | --- | --- | --- |
| 16 | 0 | ideal | −2.2877 | 0.913 – 3.913 | **4.29** |
| 16 | 0 | fitted | −9.4556 | 8.081 – 11.081 | 1.37 |
| 16 | 0 | mindiag | −16.8750 | 15.500 – 18.500 | 1.19 |
| 18 | 0 | ideal | −2.6712 | 0.796 – 4.296 | **5.40** |
| 18 | 0 | fitted | −10.7056 | 8.831 – 12.331 | 1.40 |
| 18 | 0 | mindiag | −19.3750 | 17.500 – 21.000 | 1.20 |

At `ideal` the preconditioner has a 4.3-5.4x dynamic range to work with; at `fitted` it has 1.37-1.44.
A `M^-1` that is nearly a scalar multiple of `I` cannot precondition -- LOBPCG normalizes the search
direction, so a uniform rescale is invisible. This is the same "over-shifts and flattens the spectrum"
mechanism `rqutils-precond-request.md` names, now measured as a number.

### The result was already on record, from a free bound

`rqutils-precond-request.md` had already run this experiment with the cheapest possible bound:

> A bound from data `sqd` already has -- the diagonal alone -- was tried
> (`sigma = min(diag) - 2*max|diag|`): it restores correctness and positive-definiteness, and yields
> **1.04-1.24x**.

`fitted`'s 1.29x (1.15-1.36) sits just above that 1.04-1.24x band, and `mindiag` measured here at
1.28x confirms the two are the same regime. **So the SDP bound buys perhaps 0.05-0.1x over a bound
computable from the diagonal in O(N), and both sit far below the 1.60x an exact shift gives.** The
level-1 SDP has not found a new operating point; it is a marginally better crude bound.

### Verdict

The honest summary is that this line is closed, on a positive-but-insufficient result:

* An SDP-derived shift **does** reduce iterations -- 1.29x median, 0/12 regressions, correct energies
  throughout. It is not a pessimization, which distinguishes it from two prior candidates.
* But **the gain splits almost evenly: 1.13x from the shift, 1.14x from the preconditioner.** The
  over-shift flattens `M^-1` to within 1.34-1.45x of the identity, so the Jacobi multiply delivers 14%
  where at the ideal shift it delivers ~42%. And the whole 1.29x is available from a diagonal-only
  bound at O(N) cost -- no conic solve, no `clarabel`, no closed-form fit per coupling family.
* The gap to `ideal` (1.60x) is not closable by a better *bound on `H`*. `ground_locg` sees
  `hproj(H, subspace)`, whose minimum sits 0.64-1.06x of the projected spectral width above
  `lambda_min(H)`; that gap is a property of the random projection, so no bound on `H` -- however
  tight, SDP or otherwise -- can reach the shift `ideal` uses.

**Recommendation: do not integrate.** If a shift is wanted, use `min(diag) - 2*max|diag|` for its
1.04-1.24x at O(N). The remaining upside lives in estimating `lambda_min` of the *projected* operator,
not in tightening a bound on `H`, and the untried candidate for that is still the two-level/deflation
preconditioner `rqutils-precond-request.md` names as speculative.
