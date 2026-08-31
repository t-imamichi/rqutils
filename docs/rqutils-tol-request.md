# rqutils change request: let `tol` mean the eigen-residual

One request against `rqutils` on branch `dev` (installed rev `c400fae`), written from the `spinchain`
side. A separate ask from `docs/rqutils-requests.md` (whose C1/C2/C3 shipped and are adopted, and whose
A1 was rejected) and from `docs/rqutils-precond-request.md` (withdrawn); it depends on none of them.

> **Status: NOT SENT.** Drafted from the `tol` measurements recorded in `NOTES.md`'s seed-variance
> entry. Nothing in `spinchain` passes `tol` today, so every timing in that entry and in `CONFIG.md`
> is at the machine-epsilon default this file is about.
>
> Read `docs/rqutils-precond-request.md`'s status block before acting on this one. That request was
> built on iteration counts measured with **scipy's lobpcg** — a different solver on a different
> operator — and upstream found the capability unusable on the only path `spinchain` has. The figures
> below are measured through `sqd()` itself on this repo's own operators, which is the difference that
> matters; but the "Not verified" list is deliberately explicit about what still is not.

## The problem

`sqd(tol=None)` — the default, and what `spinchain` passes at every call site — resolves to the
**machine epsilon of the operator dtype**, ~2.2e-16 for the `complex128` that any `By != 0` XXZ chain
produces. That is the tightest target available, and it is ~10 orders of magnitude tighter than what
this caller can actually verify: `skqd/sqd_backend.py::_RESIDUAL_TOLERANCE` is **1e-6**, the threshold
above which `ground_state_packed` raises.

The cost of that overshoot is not uniform, which is what makes it hard to see. Holding `n`, `dim` and
the Hamiltonian fixed and varying only which subspace is drawn, `sqd()` spans **13.5x** in wall time
(1.89 s to 25.58 s at n=50, dim=200000; 19.8x across nine draws at n in {50, 70, 100}). The slow draws
are the ones where the last digits are expensive, so the variance *is* the overshoot.

Measured on two slow draws, `cache_level=(1, 2)`, `prefilter=(32, 2)`, dim=200000:

| case | `tol` | `sqd()` | ΔE vs default | eigen-residual |
| --- | --- | --- | --- | --- |
| n=50, seed 99 | `None` (eps) | 27.28 s | — | 1.2e-09 |
| | `1e-12` | **13.03 s** | +1.0e-08 | 5.0e-06 |
| | `1e-09` | **0.91 s** | +9.2e-03 | 4.1e-03 |
| n=100, seed 7 | `None` (eps) | 35.37 s | — | 1.2e-09 |
| | `1e-12` | **14.31 s** | +5.3e-07 | 4.4e-06 |
| | `1e-09` | **2.07 s** | +4.0e-03 | 1.5e-03 |

**There is no value that is both faster and admissible.** `1e-12` is 2.1-2.5x with the energy intact to
~1e-07 — but its eigen-residual is 4.4-5.0e-06, so `spinchain`'s own guard raises. `1e-09` is 17-30x and
moves the energy by 4-9e-03, a real accuracy loss rather than rounding. The two criteria disagree about
what "converged" means, and the caller can only express its requirement in the one `sqd()` does not
accept.

## The ask: let `tol` mean the eigen-residual

Either interpret `tol` as a bound on `‖Hv − λv‖` (normalized however upstream prefers, documented), or
add a separate keyword that does — `residual_tol=1e-6` alongside the existing `tol`. A caller that
needs 1e-6 could then ask for 1e-6 and stop paying for 2.2e-16.

The expected win is the gap between the two rows above: **~2x on the median solve and far more on the
slow tail**, since the tail is exactly where the surplus digits cost the most. That would also compress
the 13.5x subspace variance, which is currently the dominant term in any cost projection for this
workload — it is why a single-seed timing here cannot be scaled.

## Why the obvious workaround is wrong

Passing `tol=1e-12` from `spinchain` and loosening `_RESIDUAL_TOLERANCE` to match would trade a
guard for a speedup. That guard is not decoration: it exists because a Rayleigh quotient cannot detect
non-convergence (`NOTES.md`), and it has already caught a real upstream regression. Loosening it to
5e-06 to accommodate a knob would remove the check that made the knob's effect visible.

And a looser `tol` cannot be validated on energy alone. `NOTES.md`'s prefilter entry measured iteration
counts identical from 1x to 1660x the true spectral bound while the filtered vector's ground-state
**overlap** fell to 0.095 — the energy looked fine throughout. Here the eigenvector is consumed by
`js_projection` and by the occupancy prior that steers every proposal, so a vector degraded below what
the energy shows would surface as a worse subspace one round later, with nothing raised.

## Verification performed

Two slow draws and one fast draw, four `tol` values each, on the synthetic low-Hamming-weight subspaces
described in `NOTES.md`'s seed-variance entry. Energies and residuals in the table above; the residual
is `sqd_backend._eigen_residual_packed`, i.e. the same quantity the shipped guard tests. `spinchain`
passes no `tol` today, so every figure in that entry and in `CONFIG.md` is at the machine-epsilon
default.

Not verified: whether the 2.1-2.5x at `1e-12` holds on **real** sampled subspaces rather than synthetic
ones, and whether a residual-targeted `tol` would compress the variance as much as the tail figures
suggest. Both need the API before they can be measured honestly.

## What lands in spinchain

`ground_state`/`ground_state_packed` would pass `residual_tol=_RESIDUAL_TOLERANCE`, making the solver's
convergence criterion and the guard's threshold **the same number** rather than two unrelated ones ten
orders apart. The guard stays — it would then be checking the contract the solver was asked to meet,
which is what it always claimed to do.
