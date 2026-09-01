# Response: `atol` and `rtol` shipped, but `rtol` is not the scale you asked for

Reply to `docs/rqutils-atol-rtol-request.md`, from the `rqutils` side. Branch `atol-rtol`, version still
`0.2.0` (unreleased).

> **Status: shipped, with one deliberate departure from the ask.** The pair exists,
> `max(atol, rtol · scale)` is the test, and either arm suffices — your §3 form, adopted as written.
>
> **The departure: `scale = ‖Hv‖ + |E|`, with no `N · 10`.** So `rtol` is a fraction of the operator
> magnitude — the `np.allclose` meaning — and is **dimension-independent**. Your §3 table's row 2 ("`0`,
> `eps·10` recovers `c400fae` behaviour exactly") is **not delivered**, and cannot be with this scale.
>
> **Your §4 question is answered, and the answer is no.** You asked whether `scale = ‖Ax‖ + |θ|` without
> `N` is enough to reproduce your locked constants. Measured: one `rtol` now delivers `2.03e-14` at
> N=64 and `2.19e-14` at N=1024 — flat across 16x, where your constants encode a bound that *grows*
> with N. **Expect them to still fail.** §3 has what to do instead.
>
> **Why we did not just give you the `N` factor**, since you asked for it and we had already shipped it
> once: it produced a *silent wrong answer*. At `rtol=1e-8`, n=2²⁰, the bound came to **4.2 against
> ‖H‖ = 20** — 21% of the operator norm — so the first iterate reported convergence and the returned
> eigenpair was arbitrary, with `converged=True`. §2 has the measurement. We are not willing to ship a
> tolerance parameter whose useful range depends on a dimension the caller may not have in hand.
>
> **`tol` is removed** (no alias, `TypeError`), so this is a break rather than the pure widening your §3
> proposed. §5.

---

## 1. What shipped

```text
converged  when  ‖Hv − Ev‖₂  <  max(atol, rtol · (‖Hv‖ + |E|))
```

| parameter | default | `None`? | meaning |
| --- | --- | --- | --- |
| `atol` | `0.0` | **rejected** | absolute residual bound; `0.0` disables the arm |
| `rtol` | `None` → `4·eps` | accepted | fraction of `‖Hv‖ + |E|` (≈ `2‖H‖`), so ≈ `8·eps·‖H‖` |

Measured behaviour, real 1D XXZ, verified against an independent dense `eigvalsh`:

| configuration | delivered residual | ΔE |
| --- | --- | --- |
| defaults (`atol=0, rtol=None`) | 2.185e-14 | 3.2e-14 |
| `atol=1e-6, rtol=0` | 4.674e-07 | 3.6e-15 |
| `atol=1e-6, rtol=None` (your row 3) | 4.674e-07 | 3.6e-15 |
| `rtol=1e-8` | 1.880e-07 | 1.8e-14 |

Your §3 row 3 — the one you said you would actually run — works exactly as you described: the looser arm
binds, and it is `atol` here.

## 2. Why `rtol` has no `N` factor

We shipped it with `N · 10` first, exactly as your §3 asked. Then we probed the parameter across its
range, which we should have done before shipping, and it fails in three ways that a caller cannot see.

**It can exceed the operator norm.** Every normalized `v` satisfies `‖Hv − Ev‖ ≤ ‖H‖`, so once the bound
reaches `‖H‖` the test carries no information — the first iterate "converges" and the eigenpair is
arbitrary. With `N` in the scale that is reachable at ordinary-looking values:

| `rtol` | bound at N=1024 | bound at N=2²⁰ | vs `‖H‖ = 20` |
| --- | --- | --- | --- |
| `1e-12` | 4.1e-07 | 8.4e-04 | fine |
| `1e-8` | 4.1e-03 | **8.4e+00** | **42% of ‖H‖** |

Confirmed end-to-end: `atol=100` against `‖H‖ = 17` converged in **one iteration** with
`converged=True`. This repo has a documented history of hunting silent wrong answers with that exact
signature, and it will not add a knob that produces them.

**It saturated.** `rtol=1e-6` and `rtol=1e-4` returned **bit-identical** answers (residual `8.458e-04`
both) — a 100x range of the dial with one outcome.

**It was not predictable.** `rtol=1e-8` meant `4.1e-03` at N=1024 and `8.4e+00` at N=2²¹. A caller cannot
state what a value will do without knowing N, which is not what "relative tolerance" means anywhere else.

The `‖Hv‖ + |E|` factor stays because it is load-bearing — `‖r‖` has units of `‖H‖`, so a dimensionless
`rtol` needs it. The `N` and the bare `10` do not: as your §4 already granted, the achievable floor is
`eps·‖H‖` with no `N` term, so neither was a rounding budget. Both arms are now guarded (`rtol ≥ 0.5`,
`atol ≥ Σ|c_k|`); the `atol` half was **missing in our first cut of this change** and is only there
because the parameter got probed a second time.

## 3. What this means for your locked constants

**They will probably still fail, and re-locking is the honest path.** Your §1 diagnosis is right that the
constants encode a per-dimension bound; we have removed the mechanism that produced one. The measurement
that settles your §4:

| live states | N (padded) | residual at `rtol=None` | residual at `rtol=1e-10` |
| --- | --- | --- | --- |
| 50 | 64 | 2.033e-14 | 1.692e-09 |
| 200 | 256 | 2.041e-14 | 1.916e-09 |
| 800 | 1024 | 2.185e-14 | 3.527e-09 |
| 859 | 1024 | 1.620e-14 | 2.031e-09 |

One Hamiltonian, one `rtol`, four subspace sizes: **flat**. That is the property we wanted and the
opposite of the one you did.

Three routes, in the order we would try them:

1. **Re-lock at `atol=_RESIDUAL_TOLERANCE`, one value, and drop the per-dimension expectation.** Your
   guard checks 1e-6; an `atol` of 1e-6 now means 1e-6 at every dimension in the run, so the constants
   become reproducible for a *different* reason — the bound no longer varies, so there is nothing for
   them to track. This is the route we would pick, and it is also the fastest (§4).
2. **Set `atol` per call site**, from the dimension you are about to solve at. This reproduces an
   arbitrary per-N schedule, including the old `eps·(‖Hv‖+|E|)·N·10` exactly if you want it —
   `atol = 2.22e-16 * 2 * sum(abs(c)) * N * 10` is a one-line helper on your side. It is more explicit
   than a magic scale and you control it.
3. **Ask us to reinstate the `N` factor** behind a separate, named parameter, so it cannot be reached by
   a plausible-looking `rtol`. We would want the accept-anything guard on it and a reason route 1 or 2 is
   insufficient, but the request is not unreasonable and this document is not the last word.

On your §4's second point — whether this is worth it to anyone but you — `rtol` in its current form is
useful to a caller who does not know `‖H‖`, which is a broader audience than the pipeline case. So the
parameter earns its place; it is the `N` factor that did not.

## 4. Performance

`atol=1e-6` against the default is **1.85x** (4.60 → 2.49 ms warm, best of 5, real 1D XXZ, N=800). Note
the baseline: the default is `atol=0, rtol=4·eps`, which targets ~8x the achievable floor, so it is a
*tight* default. The win your original `tol` request was after is still there.

An earlier draft of this section said 1.96x. That figure was measured against the **superseded**
`n · 10` default and does not survive the scale change — the current default is a different quantity, so
the ratio had to be re-measured rather than carried over. Same lesson as your §1: a tolerance figure is
only meaningful beside the definition it was taken under.

Two things we are **not** claiming. We have not measured at your `dim = 2,723,958` — the dense
`eigvalsh` that validates these runs stops fitting near N≈3000 — so treat the ratio as measured only in
the range shown. And your `~2.1x` figure for the absolute-`tol` default regression no longer applies:
that default is gone, replaced by the relative pair, and the current default is a different quantity
again.

## 5. The break

**`tol` is removed with no alias.** `tol=` raises `TypeError`. This is *not* the pure widening your §3
offered, and the reason is that `tol` has meant two different things in two months — relative against an
`N`-scaled bound until 2026-08-31, absolute after. A third revision that silently resolved it to one of
the pair would be the worst of the three options. Migration:

| you had | you want |
| --- | --- |
| `tol=x` on the absolute form (`d55f067`) | `atol=x, rtol=0.0` |
| `tol=x` on the relative form (`c400fae`) | no exact equivalent — see §3 |
| nothing (the default) | nothing; the default is the relative pair |

It is also an **arity** change: `ground_locg`'s positional signature gained a slot, so any caller passing
through `maxiter` positionally must add one. Our own test suite caught this immediately, which is what its
all-positional arm exists for.

## 6. Verification

- **644 tests pass** (was 636), full extras so none of the 23 optional-dep tests skipped.
- **ruff check / ruff format / ty check** clean; docs build clean (its 1 warning is pre-existing).
- **`poc7_sharding`** agrees at all six `cache_level`s, worst deviation 8.9e-16.
- **Four mutants killed**, each by reverting the change in place in a fresh subprocess:

  | mutant | result |
  | --- | --- |
  | `max` → `min` | 9 failures |
  | drop the `rtol ≥ 0.5` guard | 1 |
  | drop the `atol ≥ Σ\|c_k\|` guard | 1 |
  | hardcode `rtol`'s default as the float64 literal | 1 (the float32 arm stops converging) |

**Three process notes, because two of them are corrections to our own work.**

The `atol` guard was **missing from our first implementation of this change**. We measured the
accept-anything failure on `rtol`, wrote a guard whose justification is stated in terms of the *bound*
("every normalized vector satisfies `‖Hv − Ev‖ ≤ ‖H‖`"), and then applied it to only one arm. That
argument was never about which parameter produced the bound. It was caught by re-probing `atol` after
someone asked whether it had been reviewed as carefully as `rtol`; it had not.

A test we wrote for the dimension property **passed for the wrong reason**. It compared two fixtures
differing in *both* qubit count and subspace size, so `‖H‖` moved too — and it passed against a formula
with no dimension term at all. Rewritten to vary one axis at a time, which is the only form that can
attribute the difference.

And one boundary test asserted `atol == Σ|c_k|` exactly, which **did not raise**: `sqd` sums the padded
coefficient rectangle, so its `Σ|c_k|` is `6.47309024676594` against the test's `6.473090246765939` — one
ulp apart, from a different summation order. The test was asserting floating-point associativity rather
than the guard. It now tests strictly inside the rejection region.
