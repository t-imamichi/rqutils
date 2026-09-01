# Response: `tol` now *is* the eigen-residual, and the floor has no dimension dependence

Reply to `docs/rqutils-tol-request.md`, from the `rqutils` side. Branch `dev`, version still `0.2.0`
(unreleased).

> # ⚠️ SUPERSEDED 2026-09-01 — `tol` no longer exists
>
> **Do not copy code from this document.** `tol` was removed and replaced by the pair `atol`/`rtol`,
> so every `tol=` call below raises `TypeError`. Convergence is now
>
> ```text
> ‖Hv − Ev‖₂  <  max(atol, rtol · (‖Hv‖ + |E|))
> ```
>
> satisfying **either** arm. See `docs/rqutils-atol-rtol-request.md` and its response for the reasoning;
> §6 below has been rewritten with the working call, and §1 and §5.1 carry inline notes where their
> conclusions no longer hold.
>
> **What this document still gets right, and why it is kept:** the residual-floor measurement of §2
> (`eps·‖H‖`, no `N` dependence, 27 samples) is unchanged and is the evidence base for the current
> design. So is §3's guard rationale and §4.2's warning about scaling tolerance figures across
> definitions — a warning this document then violated itself, twice, which §7 records.
>
> **The one conclusion that reversed:** §5.1 measured the absolute-`tol` default as 1.18–1.49x slower
> than `c400fae`'s and told you to pass an explicit `tol` to recover it. That default is gone. The
> current default (`atol=0.0, rtol=None`) is relative again, so the regression §5.1 describes no longer
> applies — but it is *also* not `c400fae`, because `rtol`'s scale dropped the `N · 10` factor.

> **Status: shipped.** `tol` is now an **absolute bound on `‖Hv − Ev‖₂`** on both `sqd` and
> `ground_locg`, taken as a rename-in-place rather than a new keyword. You can pass
> `atol=_RESIDUAL_TOLERANCE` (this said `tol=` when written — see the banner above) and the solver's
> convergence criterion and your guard's threshold become the same number, which is what you asked for.
>
> **One correction, and it is to the mechanism, not the ask.** The request attributes the cost to
> "surplus digits" below what the caller can verify. That is right about the *symptom*. The cause is
> that the old `tol` was multiplied internally by `(‖Hv‖ + |E|)·N·10`, and the `N·10` factor was
> **measured to be slack, not a rounding budget**: the achievable residual floor has **no `N`
> dependence** at all. So the reason "no value is both faster and admissible" is not that 1e-12 and
> 1e-6 sit on either side of a real limit — it is that the old `tol` could not *name* a residual,
> because the same value meant a different absolute residual at every `N`.
>
> **Your ~2x projection is confirmed, at a different operating point than your table's.** Measured
> warm on a real 1D XXZ subspace (N=800): **1.96x at `tol=1e-6`** against the default, monotonic across
> the range (§4.1). But your table's `1e-12`/`1e-9` rows were taken under the *old relative* `tol` and
> do not carry over to the new semantics — §4.2.
>
> **⚠️ Correction, and it withdraws a reassurance an earlier revision of this document gave you.** That
> revision said `tol=None` behaviour is "essentially unchanged" and that you are therefore "unaffected
> today". **Both are wrong.** The new default is **1.18–1.49x slower**, A/B'd warm against the
> pre-change revision, and **the slowdown grows with `N`** — 1.22x at N=200 rising to 1.49x at N=9460.
> The default is also 3–4 decades tighter (1e-14 against 1e-10), which is where the time goes; energies
> are bit-identical. The claim was made without measuring, and it was the one claim in this document
> that you would have relied on to do nothing. **You are not unaffected. §5.1 has the table and what to
> do about it** — the short version is: pass the tolerance explicitly rather than relying on a default.
> (The regression itself is gone with the absolute default; §6 has the call that works today.)

---

## 1. What shipped

| Change | Where |
| --- | --- |
| ~~`tol` is an absolute bound on `‖Hv − Ev‖₂`~~ — **superseded**, now `atol`; see the banner | `ground_locg` (the convergence test), `sqd`, `run_sqd` |
| ~~`tol=None` resolves to `4·eps·max(‖Ax₀‖, 1)`~~ — **gone**; the default is now `atol=0.0, rtol=None` | `ground_locg` |
| A below-floor `tol` raises `ValueError` naming a value that works | `sqd` |
| `residual_floor(opnorm_bound, dtype)` — public, so you can compute the floor yourself | `ground_locg` |

The old test was `‖r‖ < tol · (‖Ax‖ + |θ|) · N · 10`. It is now `‖r‖ < tol`.

`p_is_zero` remains a second, independent convergence route — a zeroed search direction means
`{x, y}` already spans the residual, so no further iteration can lower `θ`. It ignores `tol`, by
design, and predates this change. **A converged result can therefore have a residual above your
`tol`** in that one case. It is a genuine stationary point rather than a tolerance crossing, but if
your guard treats residual as a hard contract, that is the case to know about.

## 2. The measurement that changed the design

The request's framing implies the floor is `N`-dependent — that is the natural reading of "the last
digits are expensive" on large subspaces, and it is what the old formula asserted. We swept it before
building anything, because an absolute `tol` is only safe if it stays satisfiable as `N` grows.

Method: run with `debug=True` (a `scan` over the full `maxiter`, so it does **not** stop at
convergence) and `tol=0`, then read where `‖r‖` plateaus. 27 samples, `N = 70 … 32768`, `‖H‖₂` over six
decades, float64 and complex128, dense **and** matrix-free.

| model | min | median | max | spread |
| --- | --- | --- | --- | --- |
| `floor / (eps·‖H‖)` | 0.494 | **0.839** | 1.260 | **2.6x** |
| `floor / (eps·‖H‖·N)` | 1.9e-05 | 2.4e-04 | 5.8e-03 | 306x |

```
floor(‖Hv − Ev‖)  ≈  eps(dtype) · ‖H‖₂        no N dependence
```

Two arms worth naming separately:

- **Scale invariance.** Holding `N = 800` and scaling `H` by 10⁻³…10³, `floor/(eps·‖H‖)` stayed in
  [1.13e-16, 2.58e-16] — i.e. `floor/(eps·‖H‖)` ∈ [0.51, 1.16] with no trend. That isolates the
  mechanism from the size sweep.
- **The matrix-free path matches.** On the same subspaces, `floor_matfree / floor_dense` ∈
  [0.66, 1.57], straddling 1.0 with no bias. So the packed-scan `Ax` carries the same rounding
  constant as a dense matvec, and the floor you get through `sqd()` is the one measured here. Extended
  matrix-free-only to `N = 32768`.

**Consequence for you:** at your `‖H‖ ≈ 20`, the floor is ~4e-15. `residual_tol = 1e-6` has **nine
decades of headroom**, and that headroom does not shrink at `dim = 200000`. The concern that motivated
our first look — that an absolute tolerance becomes unsatisfiable at scale — does not arise.

## 3. The guard, and why it raises

`sqd` rejects a `tol` below `4·eps·Σ|c_k|` (3.2x margin over the worst measured constant):

```
ValueError: atol=1.000e-30 is below the achievable eigen-residual floor 2.931e-14 for this
operator (4 * eps * sum|c_k|, with sum|c_k|=33 bounding ||H||_2) and rtol=0 leaves no other
arm, so the solve could never converge and would exhaust maxiter. The floor is eps*||H||_2
and does not shrink with subspace size -- measured over n=70..32768 and six decades of
||H||. Pass atol >= 2.931e-14, or a non-zero rtol.
```

*(Text as of 2026-09-01. The original said `tol=`; the `and rtol=0 leaves no other arm` clause was added
with the pair, because a below-floor `atol` is harmless when the relative arm can still fire — a guard
must not fire on correct input.)*

`Σ|c_k|` is already computed for `prefilter_hi`, so the guard is free. It is a **1.56–1.90x
over-estimate** of `‖H‖₂` on 1D XXZ fixtures, which is the safe direction: it raises the reported
floor, so anything admitted is comfortably reachable.

Raised, not clamped. The floor is computable from the operator alone, and clamping would silently
deliver a criterion other than the one requested — the same reasoning as the `poc13` capacity note.

## 4. On the speedup — read this before quoting a number

### 4.1 The measured figure: 1.96x at `tol=1e-6`

The request's §"The ask" projects "~2x on the median solve". **Confirmed.** Real 1D XXZ, n=12, N=800,
`cache_level=(1, 2)`, `prefilter=(32, 2)`, every arm warm, best of 3:

| `tol` | wall time | vs default |
| --- | --- | --- |
| `None` (default) | 5.06 ms | 1.00x |
| `1e-12` | 4.21 ms | 1.20x |
| `1e-9` | 3.21 ms | 1.58x |
| `1e-6` | 2.58 ms | **1.96x** |

Monotonic, which is the shape to expect: the residual falls geometrically, so each decade of `tol` buys
a roughly constant number of iterations.

**Note the baseline.** "vs default" here means against the *new* `tol=None`, which is itself 1.18–1.49x
slower than the old default (§5.1). Against the **old** default the 1.96x nets to roughly 1.5x. Quote
whichever you mean, but say which.

One caveat on how to reproduce this. `tol` is a **traced** argument of `run_sqd`, not a static one
(`run_sqd._cache_size()` stays at 1 across three distinct values), so changing it does **not** retrace
the solver — but the *first* call in a process still pays the trace. An earlier draft of this document
reported 0.327 s at `tol=1e-6` against 0.003 s at `tol=1e-9` and blamed per-`tol` retracing; both the
number and the explanation were wrong. That first figure was one cold call. Warm every arm before
comparing.

What the change *guarantees*, as distinct from what it saves:

| requested `tol` | delivered residual | ΔE vs `eigvalsh` |
| --- | --- | --- |
| `None` (default) | 1.020e-14 | 2.8e-14 |
| `1e-6` | 4.674e-07 | 3.6e-15 |
| `1e-9` | 5.568e-10 | 3.6e-14 |
| `1e-12` | 9.583e-13 | 3.9e-14 |

Real 1D XXZ, N=800, verified against an independent dense `eigvalsh`. Every request is met.

### 4.2 Why your table's figures should not be scaled

The 1.96x above is ours, on our fixture. Your table's rows are a different matter, for two reasons —
both about what the old `tol` meant:

1. **Your `1e-12` and `1e-9` rows were measured under the relative form**, so they correspond to
   *different absolute residuals at every `dim`* — your own table shows it: `1e-12` gave residuals of
   4.4e-06 and 5.0e-06 at two different `n`. Under the new semantics `tol=1e-12` means 1e-12. The rows
   do not carry over.
2. **Treat the 13.5x variance claim as still open.** The request says "the slow draws are the ones
   where the last digits are expensive, so the variance *is* the overshoot", and projects that a
   residual-targeted `tol` would compress it. We could not confirm that, and our data points the other
   way: iterations-to-`‖r‖<1e-8` on 1D XXZ track the **relative spectral gap** more closely than `N`
   —

   | live states | `N` | relgap | iters to `‖r‖<1e-8` |
   | --- | --- | --- | --- |
   | 200 | 256 | 8.09e-02 | 40 |
   | 248 | 256 | 7.69e-02 | 42 |
   | 800 | 1024 | 5.48e-02 | 53 |
   | 859 | 1024 | 5.47e-02 | 76 |
   | 2898 | 4096 | 4.12e-02 | 92 |

   `N` rises 16x while iterations rise 2.3x, tracking the 2.0x fall in relgap. **But relgap and `N` are
   correlated in this family, so this does not cleanly separate them** — it is suggestive, not decisive.
   The honest statement: a `tol` that stops earlier truncates every draw, so expect *some* compression;
   if the slow draws are slow because their gap is small, the variance is a property of the subspaces
   and will survive. Your §"Not verified" listed exactly this. It is still not verified, and our sweep
   was not designed to settle it — a draw-to-draw sweep at fixed `N` and fixed Hamiltonian, which is
   your fixture and not ours, is what would.

Your §"Not verified" flagged both of these as needing the API first. It now exists, so both are
measurable on your own sampled subspaces — which is where they have to be measured, since the
synthetic-vs-real distinction is the one your own status block called the difference that matters.

## 5. The break

**This is a silent semantic change on a published parameter, taken deliberately.** `tol` keeps its
name, so an existing explicit `tol=…` call keeps working and **changes criterion with nothing raised**:
at `dim = 2e5` the old `tol=1e-6` admitted a residual of order 1e0 where it now demands 1e-6.

- **Anyone who tuned `tol` empirically must re-derive it.** There is no CHANGELOG in this repo; the
  notice is a `.. warning::` block in both `sqd`'s and `ground_locg`'s published docstrings.
- **Anyone relying on the default is slower, not unaffected** — §5.1. This corrects an earlier
  revision of this document.

`ground_locg` with a **callable** `mat` cannot check the floor — it has no coefficients to bound `‖H‖`
with — so a below-floor `tol` there exhausts `maxiter` rather than raising. `sqd` always checks.

### 5.1 The default got slower, by a factor that grows with `N`

An earlier revision of this document told you `tol=None` was "essentially unchanged" and that you were
"unaffected today". That was asserted without measuring it. Measured — `tol=None` on both sides, warm,
best of 5, real 1D XXZ, A/B against a worktree of the pre-change revision `c400fae`:

| `N` | before | after | **slower** | residual before | residual after |
| --- | --- | --- | --- | --- | --- |
| 200 | 1.06 ms | 1.29 ms | **1.22x** | 1.58e-11 | 8.82e-15 |
| 800 | 3.48 ms | 4.64 ms | **1.33x** | 7.34e-11 | 1.02e-14 |
| 859 | 4.03 ms | 4.73 ms | **1.18x** | 8.36e-11 | 1.62e-14 |
| 2898 | 18.68 ms | 26.32 ms | **1.41x** | 2.35e-10 | 1.23e-14 |
| 9460 | 81.19 ms | 120.79 ms | **1.49x** | — | — |

Median **1.33x**, and **rising with `N`**. Energies are bit-identical at every size, so this is not an
accuracy trade — it is the same answer, converged 3–4 decades further than before, paid for in
iterations.

**Why.** The old default was `eps`, but it was compared against `tol·(‖Ax‖ + |θ|)·N·10`, so the
*effective* absolute bound it admitted was `eps·(‖Ax‖ + |θ|)·N·10` — carrying `N`. The new default is
`4·eps·‖Ax₀‖`, which does not. The ratio between them is therefore ~`N·10/4`:

| `N` | old bound | new bound | old/new |
| --- | --- | --- | --- |
| 800 | 9.4e-11 | 1.8e-14 | ~5,100x |
| 2898 | 3.4e-10 | 1.8e-14 | ~18,000x |
| 200000 | 1.8e-08 | 1.8e-14 | ~1,000,000x |

**At your `dim = 200000` the new default is ~10⁶x tighter than the old one.** The 1.49x at N=9460 is a
*lower bound* on what you would see. We did not measure at your size — the dense `eigvalsh` reference
that validates these runs does not fit past N≈3000, and extrapolating a wall-clock ratio from a bound
ratio is exactly the kind of inference this document tells you not to trust elsewhere. Treat the
`dim=200000` row as arithmetic on the bounds, not a measurement.

**What to do about it, and why this is still a net win for you.**

> **Superseded 2026-09-01.** The regression this section measures was in the *absolute* `tol=None`
> default, which no longer exists — the current default is relative again, so there is nothing here to
> recover. The advice below is kept because the *reasoning* still applies to choosing a criterion: pass
> the tolerance rather than relying on a default whose value you did not choose. Substitute
> `atol=_RESIDUAL_TOLERANCE, rtol=0.0` for the `tol=…` in the table.

Do not stay on the default. Pass `tol=_RESIDUAL_TOLERANCE` (1e-6) explicitly, which was the point of
the request:

| what you run | residual delivered | vs the *old* default |
| --- | --- | --- |
| old default (before this change) | ~1e-10 at N=800, ~1e-8 at N=2e5 | 1.00x |
| new default (`tol=None`) | ~1e-14 | **0.67–0.82x** — slower, worst at large `N` |
| `tol=1e-6`, as you asked for | ~5e-07 | **~1.5x faster** |

The middle row is the regression. The bottom row is the ask, and it is *faster than where you started*
while delivering a residual your guard can accept — 1.96x against the new default (§4.1), which nets to
roughly 1.5x against the old one. **The change is only a win for you if you pass the parameter.** If
you ship it and keep `tol=None`, you take a 1.3–1.5x loss for accuracy you do not use.

**Why the default was not left alone.** Under absolute semantics a bare `eps` sits *below* the
achievable floor `eps·‖H‖` for any `‖H‖ > 1`, so the convergence test becomes unsatisfiable and every
default call would exhaust `maxiter` and raise — mutation-tested, `RuntimeError: did not converge in
maxiter=8000` at `‖H‖ = 219` (§7). The default had to acquire the operator's scale. What it could
*also* have acquired is the `N` factor, reproducing the old effective bound exactly; we did not, because
a default that means 1e-10 at one size and 1e-8 at another is the property this whole change existed to
remove, and reintroducing it in the default brings it back through a side door. That is a judgement
call, not a measurement — **if you would rather have the old timing on the default than a default that
names a residual, say so and it is a two-line change.**

## 6. What lands in spinchain

> **Rewritten 2026-09-01.** The original text of this section said to pass `tol=_RESIDUAL_TOLERANCE`.
> That raises `TypeError` now. The working call is below.

```python
ground_state_packed(..., atol=_RESIDUAL_TOLERANCE, rtol=0.0)
```

`atol` is the absolute bound on `‖Hv − Ev‖₂`, so the solver's criterion and `_RESIDUAL_TOLERANCE` are
the same quantity — which was the whole point of the original request, and it is delivered.

**Why `rtol=0.0` explicitly.** Convergence is `max(atol, rtol · scale)`, so leaving `rtol` at its
default (`None` → `4·eps`) means the *looser* of the two arms wins. On a complex128 XXZ subspace with
your reported `Σ|c_k| ≈ 19` the relative arm targets ~1e-14 — far tighter than 1e-6, so `atol` binds
either way and the default is harmless. Verified on such a fixture: `atol=1e-6, rtol=0.0` and
`atol=1e-6` alone both deliver a residual of **9.497e-07**, bit-identical. Stating `rtol=0.0` makes the
criterion exactly the one your guard checks, with nothing else able to satisfy it; drop it if you would
rather have "whichever is looser".

**Keep the guard.** §1's `p_is_zero` note is one reason it still earns its place, and the request's own
argument (a Rayleigh quotient cannot detect non-convergence; it has already caught a real upstream
regression) is the other. It is now checking the contract the solver was asked to meet.

**On the timing advice this section used to give:** it said passing the parameter was necessary to
recover a 1.18–1.49x default regression. That regression is gone with the absolute default, so passing
`atol` is now a choice about *criterion*, not about speed. If you want the speed figure, it is 1.85x for
`atol=1e-6` against the current default — measured in the atol/rtol response, not here.

Two things the request raised that this change does **not** give you:

- **A residual bound is not an accuracy guarantee.** `|E − λ_min| ≤ ‖r‖²/gap` asymptotically, so the
  energy error at `tol=1e-6` depends on a gap you do not have. Our table above shows ΔE *better* than
  the residual by 1–7 decades at every arm, but that is the well-conditioned regime, not a promise.
- **The eigenvector-degradation concern stands unaddressed.** The request is right that `js_projection`
  and the occupancy prior consume the vector, and that `NOTES.md`'s prefilter entry measured overlap
  falling to 0.095 while the energy looked fine. A residual bound constrains the vector more tightly
  than an energy check does, so this is an improvement — but if the prior is the sensitive consumer,
  measure overlap directly at your chosen `tol` before trusting it.

## 7. Verification

- **636 tests pass** (was 627), full extras so none of the 23 optional-dep tests skipped.
- **ruff check / ruff format / ty check** clean. Docs build clean (its 1 warning is pre-existing,
  confirmed by building against stashed changes).
- **`poc7_sharding`**: sharded and single-device agree at all six `cache_level`s, worst deviation
  8.9e-16. Its own residual check independently reports `‖Hv−ev‖/‖H‖` of 5.5e-16 and 6.6e-16 —
  a third confirmation of the `eps·‖H‖` floor, from a script written for another purpose.
- **Three mutants killed**, each verified by reverting the fix in place in a fresh subprocess:

  | mutant | result |
  | --- | --- |
  | restore the relative test | 2 failures — `4.967e-05` delivered against a requested `1e-8`; `4.006e-05` vs `6.331e-11` across two dimensions from one `tol` |
  | remove the guard | 4 failures |
  | revert default to bare `eps` | `RuntimeError: did not converge in maxiter=8000` |

**A methodological note, because it bears on how to read the request's own table.** Two of our new
tests initially passed against the restored-relative-form mutant. Cause: at `N=21` the old threshold
was ~4200x looser, so the solver overshot it and satisfied the absolute assertion anyway. The tests
only discriminate on a fixture large enough for the `N` scaling to bite (n=10, 200 states) or one that
varies dimension explicitly. This is the same effect that makes single-`dim` `tol` comparisons
unreliable, and it is why §4.2 asks you not to scale the request's rows.

**And the compatibility claim was asserted, not measured.** "`tol=None` behaviour is essentially
unchanged" appeared in three places — this document, `CLAUDE.md`, and the `sqd` docstring — on the
strength of reading the two code paths rather than timing them. It is wrong by 1.18–1.49x (§5.1). The
reasoning that should have caught it needs no benchmark at all: the old default's *effective* bound
carried an `N` factor and the new one does not, so they cannot be equivalent, and the discrepancy has to
grow with `N`. A claim about behaviour being unchanged is a claim about a measurement, and this repo's
own rule is to A/B whole calls against a worktree of the pre-change revision. We did that only once
asked.

Separately, our first large-`N` floor sweep reported constants from 1.2e3 to 4.8e7 — non-monotonic
across five decades, which no rounding model produces. `maxiter=120` had left the residual still
descending 799–2096x within its final quartile, so the tail median was sampling a live trajectory. A
plateau gate (`r[75%]/r[-1] < 3`) is now enforced in the harness and those rows were discarded rather
than published. Had we only run large `N`, the artifact would have read as "the matrix-free path has a
much higher floor" — plausible and entirely wrong.
