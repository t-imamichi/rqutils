# Warm-starting the SQD solve across rounds: measured, and not worth building

**Audience.** Anyone on the `utokyo-saito` / `spinchain` side who is considering reusing the previous
round's eigenvector to speed up the next `sqd(...)` call. You do not need the `rqutils` repo to read
this; the conclusion is self-contained.

**Direction.** *Outbound* — written from `rqutils`, reporting a negative result that concerns a
`spinchain`-side idea. **Nothing here asks for a `spinchain` change, and nothing was changed in
`rqutils`.** It exists so the idea is not re-proposed a third time, and so that whoever does re-propose
it knows what a valid test requires.

**Bottom line.** Both `skqd` driver loops grow the subspace monotonically, so warm-starting the
eigensolver looks like free money. It is not. The natural shape was measured and rejected in 2026-08
(`markdown/skqd-sqd-solve-tolerance.md` §8); the refined shape was measured on 2026-09-17 and produced **no
usable verdict** across three fixtures and 19 rounds. There is currently **no fixture in either repo
that can evaluate this idea**. `sqd` exposes no seam to try it through, and **a `vinit` parameter is
declined** — see §6.3 for why that is a design position rather than a backlog item.

**Status of the measurements.** CPU, float64 (`jax_enable_x64`), `n = 14..20`, subspace dimensions 32 to
104k. Iteration counts are `ground_locg`'s own reported count, not a wrapper's. Energies are checked
against a dense/sparse oracle every round. Harness: `poc/warmstart.py` in `rqutils`
(committed `2ff269b`), runnable standalone.

---

## 1. Why the idea keeps coming back

Both driver loops union states in monotonically:

- `core.py:203-212` — cumulative Krylov rungs.
- `recovery.py:517` — `subspace |= new`, explicitly union-monotone.

Each round then calls `sqd(...)`, which re-solves from a **fresh** start vector. Round *k*'s converged
eigenvector is discarded, and round *k+1* begins from a deterministic pseudo-random spread
(`_spread_seed`, a bit-mixing hash of the state index). Throwing away a converged answer to a
near-identical problem reads as obvious waste, which is why the idea recurs.

It is not waste, for a reason specific to the solver.

## 2. The constraint that decides it

`rqutils.ground_locg` is a **block-size-1 LOBPCG** specialization. Its search space at each iteration is
only `{x, residual, previous direction}` — three vectors. Convergence therefore tracks **how much
remaining error the residual can still expose**, not how close `x` already is to the answer.

That inverts the usual intuition:

> **A better start in the Rayleigh sense can be a worse start in the Krylov sense.**

Measured in 2026-08 (§8's table), same subspace, four starts:

| start | Rayleigh quotient (true −14.36) | residual | iterations |
| --- | --- | --- | --- |
| plain random | 0.020 | 5.14 | **77** |
| `_spread_seed` (shipped) | 0.007 | 3.61 | 86 |
| shifted-power `k=10` | **−9.36** | **2.98** | **177** |
| diagonal-weighted random | −2.19 | 11.23 | 140 |

The shifted-power start is far closer to the answer by every scalar you would report, and needs **2.3×
more iterations**. Power iteration damps precisely the subdominant components the residual needs to
expose. `_spread_seed`'s docstring says "the point is coverage, not quality" — that is what it means.

This is the same mechanism that has now sunk three separate ideas in `rqutils`
(`markdown/locg-next-candidates.md` §1.4 records it as a standing constraint on any candidate).

## 3. What was rejected in 2026-08: zero-padded continuation

Seed round *k+1* with round *k*'s eigenvector, zero on the newly added states.

- Iterations **79 → 129** and **112 → 136** at two sizes.
- At dim=12000 it converged to a **different eigenvalue**, `|dE| = 7.4e-01`.

Stated mechanism: a zero-padded eigenvector is closer to a **one-hot** than to a spread — exactly zero
on every new state — so single-vector LOBPCG has no direction to escape with. This reintroduces the
disconnected-component failure `_spread_seed` exists to prevent (that failure mode returned −1.293
against a true −2.191, reporting `converged=True`).

## 4. What was tested on 2026-09-17: spread-on-new-states

If §3's mechanism is the whole story, the fix is obvious: **carry the eigenvector on states that
survive, and put `_spread_seed` values on the newly added ones**, so the new directions are populated.

```text
vinit[old] = prev_eigvec        # rescaled
vinit[new] = _spread_seed(new)  # rather than 0.0
```

Two implementation details that matter if anyone rebuilds this:

- **The join must be by state bytes, not by index.** `uniquify_states` lex-sorts, so round *k*'s index
  *i* and round *k+1*'s index *i* are different basis states in general. A positional carry is
  numerically plausible and physically meaningless.
- **Filler slots must stay exactly zero.** Padding slots carry no basis state; weight there puts the
  iterate partly outside the subspace.

### 4.1 It beats the shipped baseline, and that number is not usable

Prefilter off, so iteration counts reflect the start rather than the filter. XXZ Krylov subspaces from
|Néel⟩, `delta=1.0`:

| dim | carried | cold (`_spread_seed`) | warm (new shape) | **zeropad (§3, rejected)** | valid? |
| --- | --- | --- | --- | --- | --- |
| 128 | 17 | 16 | 13 | **5** | no |
| 512 | 121 | 24 | 19 | **12** | no |
| 2048 | 489 | 34 | 23 | **16** | no |
| 4096 | 1325 | 44 | 25 | **19** | no |
| 8192 | 2733 | 42 | 29 | **20** | no |
| 8192 | 4645 | 47 | 31 | **23** | no |

Warm beats cold in 6/6, median **1.46×**, energies exact to 4.3e-14. **And zero-padding — the shape §3
rejected as harmful — beats both arms in every round.**

So the fixture cannot reproduce §3's rejection, and therefore cannot evaluate a replacement for it. The
1.46× is real arithmetic measured against the wrong opponent.

### 4.2 The recovery-style fixture does not help

Suspecting the hop-rung growth pattern was too geometric, the loop was rebuilt to match
`recovery.py`'s actual algorithm — solve, square the eigenvector for per-site occupancies
(`_site_occupancy`, arXiv:2605.29521 §II.A), resample bitstrings, union, repeat.

It made things **worse**: ground-state weight landing on newly added states fell to **0.4–3.7%**
(against 18%→0.8% for hop rungs). Occupancy resampling is self-reinforcing — it draws where weight
already is. Zero-padding still won 5/5.

## 5. Why no fixture worked: the ground state is concentrated

The mechanism is not about growth patterns at all.

| measurement (n=16, 6000 states) | value |
| --- | --- |
| weight in top-64 states | **89.9%** |
| weight in top-256 states | **98.8%** |

Any subspace of a few hundred states already captures nearly all of the ground state. So round *k*'s
converged eigenvector is already ~99% of round *k+1*'s answer, **whatever the growth rule** — which is
exactly why zero-padding's "no escape direction" defect never bites and every continuation scheme wins.

### 5.1 Anisotropy is the knob that moves this, and the transverse field is not

`delta` (the ZZ coefficient) controls concentration cleanly, and survives projection because ZZ
conserves magnetization. Participation ratio = effective number of contributing states, dim=2038:

| Δ | top-64 | participation |
| --- | --- | --- |
| 0.0 (XY) | 26.7% | **440.9** |
| 0.5 | 41.2% | 173.7 |
| 1.0 (Heisenberg) | 58.8% | 48.2 |
| 2.0 | 88.8% | 5.0 |

Zero-padding's margin over the warm arm shrinks monotonically along that axis — 2–3 iterations at
Δ=1.0, exactly 1 at Δ=0.5, and a **tie at 17** at Δ=0 where weight-on-new peaks at 11.9%. It never
inverts.

**A transverse field cannot be used to delocalize a hop-generated subspace.** Such a subspace is a
single Hamming-weight sector, and single-site `X` changes weight by ±1, so the projector annihilates
every field term: `nnz` and `E0` bit-identical at `bx = 0.0`, `0.5` and `3.0` (n=12, dim=380,
E0=−20.883220316338; n=16, dim=1325, E0=−27.524421980960). The `bx = 0.0` arm is the decisive one — the
term contributes *nothing*, not merely the same thing at every magnitude. Worth flagging because
`rqutils`' own `poc/davidson_xxz.py` asserted the opposite until 2026-09-17 — true of the
*Hamiltonian*, false of *its projection*. If `spinchain` has a fixture relying on a transverse field to
break conservation **within a fixed-weight subspace**, that assumption is worth checking.

### 5.2 Near-degeneracy is absent, so that explanation is out too

§3's wrong-eigenvalue outcome at dim=12000 suggested degeneracy (and `recovery.py`'s reproducibility
does depend on which degenerate member gets picked — see `markdown/rqutils-requests-2.md`). Measured:

| dim | relgap, `(E₁−E₀)` over `abs(E₀)` |
| --- | --- |
| 489 | 2.06e-01 |
| 2733 | 9.90e-02 |
| 6885 | 5.35e-02 |
| 23353 | 4.23e-02 |
| 103876 | **3.95e-02** |

The gap narrows with dimension then **saturates around 4e-02** — the early narrowing is a finite-size
effect, not a route to a hard regime. Degeneracy does occur, but at E₂/E₃ (exact pairs), which the
solver never has to resolve. For scale, prefilter tuning in `rqutils` breaks at relgap **4.0e-05**, a
thousand times tighter than anything this operator family reaches.

## 6. What this means for `spinchain`

**Do not build warm-starting on the current evidence.** Concretely:

1. **§8's rejection stands.** Zero-padded continuation is still the measured-harmful option.
2. **The refined shape has no supporting evidence** — only a 1.3–1.9× against a baseline that is not
   the relevant comparison.
3. **There is no seam, and `rqutils` will not add one.** `sqd()` does not accept a start vector; it is
   built inside jitted `run_sqd` via `jax.lax.cond` on `jnp.all(hamiltonian.x[0] == 0)`. **A `vinit`
   parameter is declined, not unimplemented** — it would add public API surface to buy a measured
   non-result, and the two branches it would bypass each exist because they fixed a silent wrong answer
   (a one-hot cannot leave its connected component: measured −1.293 against a true −2.191, and 0.0
   against a true −1.0, both with `converged=True`). A caller-supplied vector re-opens exactly that
   class of failure, with the library unable to validate it — any vector is a *plausible* start, so
   there is nothing to check against.

   It would also break something `spinchain` currently relies on. `recovery.py::recover_configurations`
   has *observed* reproducibility only because `_spread_seed` makes the chosen eigenvector a
   deterministic function of the subspace (`markdown/rqutils-requests-2.md`; an early version gave 5
   different recovered bases in 6 identically-seeded runs). A caller-supplied start makes it a function
   of the caller's history instead, so round *k*'s vector decides round *k+1*'s degenerate member — the
   reproducibility regression that file asks to have *strengthened*, arriving through a new door.
4. **The round-over-round cost you are seeing is probably elsewhere.** Two measured alternatives that
   do pay, both already available: `prefilter=(32, 2)` (`sqd`'s default; 1.49× median end-to-end), and
   sizing `states_size` by hand past N≈1e5 — the power-of-two default wastes 62.4% of every per-slot
   term where a hand-chosen multiple wastes 3.3%, for one extra compilation and no measurable time. At
   N=24M with `cache_level=(0,2)` that is **7.7 GB**.

## 7. If you want to test it properly anyway

The declined parameter does not block this: **drive `ground_locg` directly**, assembling the matvec with
`functools.partial(apply_h, xsources=..., diagonals=...)` — that is how the harness here does it, and it
is the supported way to test a start vector without one. §6.3 declines the parameter on `sqd`; it does
not decline the experiment. Bring a positive result from that route and the API question can be
reopened on evidence.

Two requirements, both learned the hard way:

**Include the rejected shape as a third arm.** This is the reusable part of the exercise. The harness
reports a `valid` column that requires zero-padding to *lose* before the warm arm is read at all, and
it withheld the verdict in **19 of 19 rounds**. Comparing only against the shipped baseline would have
reported a clean 1.46× win. Any continuation-scheme measurement needs cold, warm **and** zeropad.

**Find an operator with relgap ≲ 1e-04.** That is the regime where §3's failure lives and where start
quality plausibly matters. The XXZ family does not reach it at any Δ, `n`, or dimension measured up to
dim=104k. This needs a different Hamiltonian — a `spinchain` production instance would be far more
informative than anything synthetic, and is the one thing `rqutils` cannot supply.

Also note the verdict metric: **iteration count, not energy or wall clock.** Every arm here converged
to the correct eigenvalue (max `|dE|` 4.3e-14), so energy cannot separate them; and with `sqd`'s
prefilter on, iteration counts drop to 1–9 and every ratio quantizes to 3.00×, making the arms
unresolvable. Measure with the prefilter off, then re-check end-to-end — `rqutils` has a recorded case
of a solver-side 1.30× becoming 0.69× once setup was restored.

---

## Appendix: reproducing

```bash
# in the rqutils repo
uv run python poc/warmstart.py          # self-check, then the hop-rung arm
uv run python -c "import examples.scaling.poc/warmstart as m; m.run_recovery(delta=0.5)"
```

`run()` and `run_recovery()` both take `nq`, `delta`, `bx`, `rtol`, `seed`. `run(prefilter=(32, 2))`
reproduces the unresolvable-quantization effect described above. Full evidence tables are in
`rqutils`' `NOTES.md` under "Warm-starting the growing subspace"; the 2026-08 predecessor is
`markdown/skqd-sqd-solve-tolerance.md` §8.
