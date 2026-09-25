# Warm-starting the growing subspace

Moved from `NOTES.md` ("Warm-starting the growing subspace", 2026-09-17); CPU, float64, `n=14..20`, XXZ
Krylov and recovery-style subspaces. Negative result: no verdict, and the fixture gate is the finding.
The script is `poc/warmstart.py` (§6); the `bx` check in §4 also ran `poc/davidson_xxz.py`'s fixture.

## 1. The proposal and the verdict

`markdown/skqd-sqd-solve-tolerance.md` §8 rejected *zero-padded* eigenvector continuation (iterations
79→129 and 112→136, a different eigenvalue at dim=12000 with `|dE| = 7.4e-01`). This tested the shape §8's
stated mechanism suggests instead — **carry the converged eigenvector on surviving states, put
`_spread_seed` values on the newly added ones**, so the new directions are populated rather than zero.

**The verdict is withheld, and that is the finding.** The new shape beats a cold `_spread_seed` by a
stable **1.3–1.9× on iteration count** in all 19 rounds measured, across every configuration. But
**zero-padding — the shape §8 rejected — beat *both* arms in every one of those rounds**, so no fixture
built here can reproduce §8 and none can judge a replacement for it. A fixture that cannot reproduce the
rejection it is meant to overturn measures nothing, however clean its own numbers look.

## 2. The gate is the reusable part

`run_recovery` reports a `valid` column that requires zero-padding to *lose* before the warm arm is read
at all, and it refused a verdict 19/19. Without it the 1.3–1.9× would have read as a win — the arm is
genuinely faster than the baseline it was compared against, just not than the one that matters. **Any
continuation-scheme measurement needs the rejected shape as a third arm, not the shipped baseline
alone.**

## 3. Four hypotheses for the non-reproduction, all eliminated

| # | hypothesis | measurement | outcome |
| --- | --- | --- | --- |
| 1 | growth pattern (hop rungs too geometric) | recovery-style occupancy resampling, `new wt` 0.4–3.7% vs 18%→0.8% | **worse** — resampling is self-reinforcing |
| 2 | ground-state weight concentration | top-64 = 89.9%, top-256 = 98.8% of weight (n=16, 6000 states) | root cause, but a property of the physics |
| 3 | delocalization via `delta` | participation 48.2 → 440.9 states as Δ 1.0 → 0.0 | right dial, never crosses over |
| 4 | near-degeneracy | relgap 6.9–8.0e-02, flat in Δ; plateaus at ~4e-02 to dim=104k | **not present** |

**Hypothesis 2 is the mechanism.** The XXZ ground state is intrinsically concentrated, so a converged
eigenvector from round `k` is already ~99% of the answer at round `k+1` no matter how the subspace grows
— which is why zero-padding's "no escape direction" defect never bites and every continuation scheme
wins.

**`delta` is the knob that moves it, and `bx` is not.** Anisotropy controls concentration cleanly
(participation 5.0 at Δ=2.0, 48.2 at Δ=1.0, 173.7 at Δ=0.5, 440.9 at Δ=0.0, dim=2038), and it survives
projection because `ZZ` conserves magnetization. Zero-padding's margin over the warm arm shrinks
monotonically along that axis — 2–3 iterations at Δ=1.0, exactly 1 at Δ=0.5, **a tie at 17 at Δ=0**
where `new wt` peaks at 11.9% — but never inverts.

## 4. Fixture trap: the transverse field is inert on a hop-generated subspace

`xxz_rungs` produces a single Hamming-weight sector (verified: all weight 6 at n=12), and single-site `X`
changes weight by ±1, so **every `bx` term projects to exactly zero** — `nnz=4208` and
`E0=-21.0756622399` bit-identical at `bx=0.3` and `bx=3.0`. So `bx` cannot be used to delocalize here.
`poc/davidson_xxz.py` carried the same dead knob and claimed the opposite in its docstring — "Bx breaks
magnetization conservation; without it the hop-generated subspace is closed under H and the projection is
trivially block-diagonal" — which is true of the *Hamiltonian* but **not of its projection onto that
subspace**; corrected 2026-09-17, and its results are unaffected since `bx` is never swept there.
Verified through `poc/davidson_xxz.py`'s own functions at its own defaults, including a `bx=0.0` arm:
`nnz` and `E0` bit-identical across bx=0.0/0.5/3.0 (n=12, dim=380, E0=-20.883220316338; n=16, dim=1325,
E0=-27.524421980960). **Inferring a property of the projection from a property of the operator is the
error** — the projector onto a fixed-weight subspace annihilates exactly the terms that break the
conservation. Use `delta`.

## 5. What it means

- **§8's rejection stands, better characterized rather than overturned.** On physically-motivated SQD
  subspaces the previous eigenvector is *so* good that the open question is not "does a warm start help"
  but "why does anything beat zero-padding" — and nothing here did.
- **`sqd` exposes no `vinit` seam anyway** (it is built inside jitted `run_sqd` via `jax.lax.cond` on
  `jnp.all(hamiltonian.x[0] == 0)`), so shipping any of this would mean adding a parameter to buy a
  measured non-result.

## Open: a genuine retest of §8 needs relgap ≲ 1e-04

This operator family does not reach it at any Δ, `n`, or dimension measured. The gap narrows with
dimension then **saturates** — 2.06e-01 at dim=489, 5.35e-02 at dim=6885, then flat at 3.95e-02 through
dim=103876 — so the early narrowing is a finite-size effect, not a path to the tight-gap regime. For
scale, `markdown/locg-next-candidates.md` records prefilter tuning breaking at relgap 4.0e-05, a thousand
times tighter. A different Hamiltonian, not a fixture tweak.

## 6. The script

`poc/warmstart.py` has no command-line arguments; `uv run python poc/warmstart.py` runs `demo()` (a
self-check of the byte-keyed carry, zero filler slots and occupancy resampling) then `run()` at its
defaults. Both drivers solve each round with three starts — cold (`_spread_seed`), warm, and
`zero_pad_start` (§8's rejected shape) — check `|dE|` against a dense `hproj` oracle, and print the
`valid` gate through the shared `report`:

| function | subspace growth | keyword defaults |
| --- | --- | --- |
| `run` | cumulative periodic-XXZ hop rungs from Néel (`xxz_rungs`) | `nq=16, max_rungs=7, cap=60000, delta=1.0, bx=0.3, seed=0, rtol=1e-8, prefilter=None` |
| `run_recovery` | resample from the eigenvector's site occupancies, Hamming-repaired, unioned | `nq=16, rounds=6, num_draws=400, delta=1.0, bx=0.3, seed=0, rtol=1e-8` (prefilter off) |

`prefilter` stays `None` deliberately: with `sqd`'s `(32, 2)` iteration counts fall to 1–9 and every ratio
quantizes to 3.00×. `bx` is kept only for signature parity with `poc/davidson_xxz.py` (§4). The recovery
arm runs as `uv run python -c "import poc.warmstart as m; m.run_recovery(delta=0.5)"`.
