# `sqd` solve tolerance: a measured 1.4–1.9×, and the `tau` coupling that gates it

**Direction: outbound.** Written from `rqutils`, describing a change available to `spinchain`. It asks
for **no `rqutils` change** — `sqd(tol=...)` already exposes everything needed. Companion to
`docs/skqd-basis-opt-optimization.md`, which is also outbound.

**Date.** 2026-08-27. All figures measured on CPU, float64 (`jax_enable_x64`), against
`rqutils` on branch `basis_opt`.

---

## 1. Summary

`rqutils.sqd.sqd`'s `tol` defaults to `None`, which means **the operator dtype's machine epsilon**
(~2.2e-16). `spinchain` never overrides it: neither `skqd/sqd_backend.py` nor `exact.py` passes `tol`,
so every SKQD solve converges ~10 orders tighter than the surrounding algorithm needs.

Loosening it is worth **1.4–1.9×** on the solve. But it is **not** a free win, because the eigenvector's
small components are what `recovery._carry_over(tau=...)` thresholds, and their noise floor scales
linearly with `tol`. Adopting the speedup requires moving `tau` with it. The measured rule is
`tau ≈ 1e5 * tol`.

This is a caller-side decision in `spinchain`, and it must be made **per call site** — the full-basis
path in `exact.py` needs machine epsilon and must not inherit a loosened default.

---

## 2. Why the solve is over-converged

Three tolerances are in play, and they are four to ten orders apart:

| what | where | value |
|---|---|---|
| inner eigensolve | `ground_locg` via `sqd(tol=None)` | ~2.2e-16 (dtype eps) |
| recovery's stopping criterion | `RecoveryOptions.tol` (`options.py:129`) | **1e-6** |
| carry-over threshold | `RecoveryOptions.tau` (`options.py:130`) | 1e-8 |

`RecoveryOptions.tol = 1e-6` stops a recovery round when the energy improves by less than that. Each
round's *inner* solve nonetheless converges to 2e-16 — ten orders tighter than the criterion deciding
whether the round mattered.

Separately, the quantity SKQD reports is capped by the **basis**, not the solve. At n=14 with a 12%
subspace the basis error is 12 Hartree; the difference between an eps solve and a `tol=1e-12` solve is
5e-13.

---

## 3. Measured speedup

Six-round sequence, n=20, dims 2000→2282, `cache_level=(1,2)` (`DiagCache.SPEED`, the default).

**The subspace sequence is held fixed across arms** — precomputed once from the reference run — so
only the tolerance varies. This matters: an earlier uncontrolled version of this experiment let each
arm choose its own subspaces and produced final energies of −13.5, −18.1 and −15.8, which measured
trajectory divergence (see §4), not solve time.

| solve `tol` | total | speedup | max \|dE\| vs eps |
|---|---|---|---|
| `None` (eps) | 0.083 s | 1.00× | 0.0 |
| `1e-12` | 0.060 s | **1.38×** | 2.4e-13 |
| `1e-11` | 0.053 s | **1.55×** | 1.7e-11 |
| `1e-10` | 0.047 s | **1.77×** | 1.6e-09 |
| `1e-09` | 0.043 s | **1.93×** | 9.0e-08 |

Single-solve figures agree: at n=20/dim=20000, `tol=1e-12` is 1.63× and `tol=1e-10` is 2.40×; at
n=24/dim=50000, 1.40× and 1.76×.

**There is a cliff.** `tol=1e-6` returns −6.47 against a true −14.36 — off by 7.9 Hartree, with no
error raised. Do not treat the trend as continuing; anything looser than ~1e-9 was not validated here
and 1e-6 is catastrophically wrong.

### Why it is 1.4–1.9× and not more

The solve is ~75% `ground_locg`, and within an iteration the matvec is only ~25% — the rest is the
`O(N)` re-orthogonalization, balancing, zero-direction-mask and Rayleigh–Ritz work (~94
dot-equivalents per iteration, scaling linearly in `N` at 30–36 µs per 1k elements across a 32× size
range). Loosening `tol` reduces the iteration **count**; it cannot make an iteration cheaper. Per
`CLAUDE.md` and `docs/locg.md`, that per-iteration work is load-bearing — seven defects (I1–I7) that
each failed silently — so it is not a target.

---

## 4. The blocker: `tol` moves the protected set

`recovery._carry_over(bitstring_matrix, eigvec, tau=options.tau)` keeps states whose `|eigvec|`
exceeds `tau = 1e-8`. Those surviving states are protected from truncation when `max_dim` is hit.

The eigenvectors barely move — but the count above `tau` moves by three orders of magnitude
(n=20, dim=2000, `tau` fixed at 1e-8):

| `tol` | overlap with eps solution | states above `tau=1e-8` |
|---|---|---|
| `None` | 1.0 | **1** |
| `1e-12` | 1.000000000000 | 22 |
| `1e-11` | 0.999999999993 | 194 |
| `1e-10` | 0.999999999943 | 1390 |
| `1e-09` | 0.999999954723 | 1914 |
| `1e-08` | 0.999998519277 | 1987 |

At eps the small components are genuinely converged to ~1e-16, so nothing crosses 1e-8. Loosen `tol`
and the residual noise floor rises above the threshold, so states cross it spuriously.

**The energy is unaffected while the algorithm's trajectory changes.** That is the failure shape this
repo repeatedly documents — right number, wrong internals — and it is why the naive version of this
change is unsafe rather than merely approximate.

### The noise floor is linear in `tol`

Measured across six orders of magnitude, n=20, dim=2000:

| `tol` | median \|v\| | p90 \|v\| | median/`tol` |
|---|---|---|---|
| `None` (2.2e-16) | 5.2e-14 | 2.1e-13 | 2.4e+02 |
| `1e-12` | 2.0e-10 | 8.0e-10 | 2.0e+02 |
| `1e-11` | 2.4e-09 | 9.8e-09 | 2.4e+02 |
| `1e-10` | 2.1e-08 | 9.2e-08 | 2.1e+02 |
| `1e-09` | 2.5e-07 | 1.0e-06 | 2.5e+02 |
| `1e-08` | 2.0e-06 | 8.7e-06 | 2.0e+02 |

`median|v| ≈ 200–250 · tol`, stable to within 25% over 10^4 in `tol`. So the floor is **linear**, not
`sqrt(tol)` — a `sqrt(tol)` rule over-suppresses (it keeps 1–19 states where the reference keeps 1,
discarding genuinely protected ones at tight `tol`).

### The pairing that preserves the protected set exactly

Reference: `tol=eps`, `tau=1e-8` → 1 protected state. Symmetric difference against that set:

| `tol` | `tau` | protected | symdiff | reproduces reference |
|---|---|---|---|---|
| 1e-12 | 1e-09 | 153 | 152 | no |
| 1e-12 | 1e-08 | 22 | 21 | no |
| **1e-12** | **1e-07** | **1** | **0** | **yes** |
| 1e-11 | 1e-08 | 194 | 193 | no |
| 1e-11 | 1e-07 | 27 | 26 | no |
| 1e-11 | 1e-06 | 2 | 1 | no |
| 1e-10 | 1e-07 | 179 | 178 | no |
| 1e-10 | 1e-06 | 24 | 23 | no |
| **1e-10** | **1e-05** | **1** | **0** | **yes** |

**Rule: `tau ≈ 1e5 · tol`.** It reproduces the reference protected set exactly at both tested
tolerances. Note the current shipped pairing (`tau=1e-8`, `tol=eps`) has a margin of ~1e8 — far above
the rule, which is why it is safe today and why nobody had to think about the coupling.

`1e5` is ~400× the measured median floor and ~100× p90, i.e. deliberately conservative. It was
validated at two `tol` values on one Hamiltonian and one subspace; treat it as a starting point to
re-validate on real SKQD subspaces, not a derived constant.

---

## 5. The full-basis path must keep machine epsilon

`spinchain.exact._rqutils_ground_state` (`exact.py:72`) hands `sqd` the **full** `2**n` basis, as its
docstring says. There the trade inverts — n=14, exact energy −22.556950561257:

| subspace | fraction of 2^n | basis error | solver error (`tol=1e-12`) |
|---|---|---|---|
| 500 | 3.1% | 1.5e+01 | 1.8e-15 |
| 2,000 | 12.2% | 1.2e+01 | 4.9e-13 |
| 8,000 | 48.8% | 6.7e+00 | 5.7e-13 |
| **16,384** | **100%** | **1.5e-13** | **9.1e-12** |

At the full basis the solver error is ~60× *larger* than the basis error. That path exists to be a
reference, and `exact.py:134` additionally checks the returned Rayleigh quotient against `<v|H|v>`
with `_RQUTILS_RESIDUAL_TOLERANCE = 1e-6` (`exact.py:35`), raising `RuntimeError` if they disagree —
its comment notes `sqd()` discards `ground_locg`'s `converged` flag, so this is the only thing
standing between a non-converged solve and a plausible-looking energy. A loosened `tol` eats directly
into that 1e-6 margin.

**So this must not become a global default.** Thread it per call site.

---

## 6. Suggested implementation

1. Add a `solve_tol: float | None = None` field to `SolverOptions` (or `RecoveryOptions`), `None`
   meaning "inherit `rqutils`' dtype epsilon" — the current behaviour, so existing configs are
   unchanged.
2. Thread it through `sqd_backend.ground_state(..., tol=...)` into the existing `sqd(...)` call at
   `sqd_backend.py:230`. `ground_state` already takes `diag_cache` as an optional override; follow
   that shape.
3. **Do not** thread it into `exact._rqutils_ground_state`. Leave that path on the default, and say
   why in a comment pointing at §5.
4. If `solve_tol` is set, validate the `tau` coupling in `RecoveryOptions.__post_init__`: warn (or
   raise) when `tau < 1e5 * solve_tol`. The existing `__post_init__` already validates `tol >= 0`
   and `tau`, so this is one more check in a place that has them.
5. Leave the default `None`. The speedup is real but the coupling is subtle; make it opt-in.

---

## 7. Verification

1. **Fixed-subspace timing.** Reproduce §3 with the subspace sequence precomputed once and reused
   across arms. Varying subspaces per arm measures trajectory divergence, not tolerance.
2. **Protected-set equivalence.** For the chosen `(tol, tau)`, assert the set
   `{i : |eigvec[i]| > tau}` equals the set from `(eps, 1e-8)` on the same matrix. This is the
   assertion that would have caught the naive change; §4's table is its expected output.
3. **End-to-end recovery trajectory.** Run the full recovery loop at both settings with a fixed seed
   and assert the per-round energies and `dim`s match. The energies alone are insufficient — they
   agreed to 1.6e-09 while the trajectory diverged.
4. **Full-basis guard.** Assert `exact_backend="rqutils"` still passes its own residual check
   (`exact.py:133-136`) and that its energy matches the dense/sparse backends to their existing
   tolerance.
5. **Do not test only at `theta`-like small values.** The `tol=1e-6` cliff (−6.47 vs −14.36, no
   error raised) means a test sweeping tolerances should include a known-bad value and assert the
   energy is rejected, not silently accepted.

---

## 8. Ideas tested and rejected — do not re-propose

Recorded so these are not mistaken for unexplored options. All measured on this branch, 2026-08-27.

- **Eigenvector continuation across the growing subspace.** Both `sqd` driver loops grow the subspace
  monotonically (`core.py:203-212` cumulative Krylov rungs; `recovery.py:517` `subspace |= new`,
  explicitly union-monotone), and each round re-solves from a fresh `_spread_seed` start. Seeding with
  the previous round's eigenvector, zero-padded onto the new states, **made it worse**: iterations
  79→129 and 112→136 at two sizes, and at dim=12000 it converged to a *different eigenvalue*
  (\|dE\| = 7.4e-01). A zero-padded eigenvector is closer to a one-hot than to a spread — exactly zero
  on every newly added state — so single-vector LOBPCG has no direction to escape with. This
  reintroduces the disconnected-component failure `CLAUDE.md` records `_spread_seed` as existing to
  prevent.
- **Jacobi preconditioner on the raw indefinite projected `H`.** `ground_locg(precond=...)` shipped
  and measures 1.79× median on a *shifted* operator, and the diagonal is already computed for free at
  `cache_level[1]=2`. Applied to the unshifted operator it hits `maxiter=1000` at every size and lands
  12–14 Hartree off: the mixed-sign diagonal destroys the descent direction. Consistent with
  `docs/rqutils-precond-request.md` closing that line.
- **Host-side `ints_to_matrix` in the recovery loop.** Rebuilt on the whole subspace every round, so
  it looked like a candidate. Measured **1.4–5.1%** of a solve (2.9 ms vs 201 ms at dim=20000; 42.6 ms
  vs 1262 ms at dim=200000, the shipped `max_dim`). Not worth touching.
- **Fusing `ground_locg`'s per-iteration vector work.** The ~75%-of-iteration bookkeeping looked like
  a fusion opportunity, but the iteration body is already `@jax.jit` with a `lax.while_loop`
  (`ground_locg.py:396,653`), so XLA already fuses what it can. The residual scales linearly in `N`,
  confirming real arithmetic rather than dispatch overhead.

Two claims were also **checked and confirmed correct**, so they need no work:

- **`NOTES.md`'s "`get_xsource` setup is 66–97% of a solve".** Correct as stated — it is about
  `cache_level[0] = 0` vs `1`, weighted by call count. Reproduced end-to-end: `(0,0)`/`(1,0)` = 7.2×
  at J=24, against the note's stated 7.2× at its J=23, with the ratio scaling in J (5.7× at J=12 up to
  9.3× at J=52), as the `O(J·N)` model predicts. An earlier "3–23%" figure measured one `get_xsource`
  call as a fraction of a solve — a different quantity.
- **`DiagCache`'s mapping.** `SPEED = (1,2)`, `MEMORY = (1,0)`, default `SPEED`. Measured fastest of
  all six levels at both n=14/N=5k and n=18/N=20k, with all six returning identical energies.
  `(1,1)` is the only level on the `[0]=1` row slower than `(1,0)` — it caches packed diagonal signs
  and re-derives the diagonal per matvec, paying unpack without the full precompute win — which is the
  empirical justification for `spinchain` exposing only two of the six.
