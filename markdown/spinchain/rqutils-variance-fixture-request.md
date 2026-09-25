# Request to `spinchain`: one draw-to-draw sweep, to settle the 13.5x variance claim

**Direction — note this is the unusual one.** `docs/rqutils-requests*.md` and
`docs/rqutils-*-request.md` are *inbound*: written from the `spinchain` side asking `rqutils` to change.
`markdown/skqd-basis-opt-optimization.md` and `markdown/skqd-warmstart-negative-result.md` are *outbound*:
`rqutils` describing work or results on the `spinchain` side. **This is a third thing — `rqutils` asking
`spinchain` for a measurement, not a code change.** Nothing here asks either repo to change behaviour.

**What is asked.** One artifact: a table of per-draw iteration counts at **fixed `N` and fixed
Hamiltonian**, at production `n` and `dim ≈ 128k`, across ~10 subspace draws and 4 tolerance settings.
Roughly 40 `sqd()` calls, no code change, ~15 minutes of compute on hardware that already runs this.
§3 has the exact columns; the appendix has a 4-call version if that is too much.

**Why `rqutils` cannot produce it.** The fixture is a production SKQD subspace. Every `rqutils` fixture is
1D XXZ, where the relative spectral gap and `N` are **correlated**, so no sweep here can separate them —
this is stated as the blocking reason in `markdown/spinchain/rqutils-tol-response.md` §2 and left explicitly open in
`NOTES.md`. It has now blocked a second investigation (`markdown/skqd-warmstart-negative-result.md`), which
is why it is being asked for directly rather than worked around a third time.

**Status: DRAFT, not sent.**

---

## 1. Why this one measurement is worth asking for

`spinchain`'s own profile of the n=13 job: sampling 50%, **`sqd()` solve 25%**, configuration recovery
8%, observable contraction 5%. The solve is the largest phase the project controls.

And its cost is not where a shape-based estimate would put it. From `markdown/spinchain/rqutils-precond-request.md`,
at **fixed** shape, varying only which subspace is drawn:

| n | dim | min iters | median | max | spread | wall min → max |
| --- | --- | --- | --- | --- | --- | --- |
| 20 | 32 000 | 33 | 70 | 169 | 5.1x | 0.388 → 0.994 s |
| 20 | **128 000** | **49** | **88** | **699** | **14.3x** | 0.911 → **10.087 s** |
| 24 | 32 000 | 35 | 42 | 58 | 1.7x | 0.412 → 0.529 s |

Every draw converged; every energy correct. On the 9-solve n=20 ladder in that same document, **the
single worst solve is 39% of an 8.98 s total**, and it is not the largest dimension — 96k costs nearly
what 128k does while 64k is cheaper than 48k. At the median throughout, the ladder would be 4.05 s.

**So the budget is the tail, not the average.** A 20% median improvement is worth ~5% of the run;
removing the 699-iteration case is worth ~40% of the solve phase. Every optimization measured in
`rqutils` to date has targeted the median.

## 2. The specific claim to settle

`markdown/spinchain/rqutils-tol-request.md` argued that the slow draws are slow because *the last digits are
expensive*, so a residual-targeted `tol` should compress the variance. `markdown/spinchain/rqutils-tol-response.md` §2
could not confirm it, and the data pointed the other way — iterations track the **relative spectral gap**
rather than `N`:

| live states | `N` | relgap | iters to `‖r‖<1e-8` |
| --- | --- | --- | --- |
| 200 | 256 | 8.09e-02 | 40 |
| 800 | 1024 | 5.48e-02 | 53 |
| 2898 | 4096 | 4.12e-02 | 92 |

`N` rises 16x, iterations 2.3x, tracking a 2.0x fall in relgap. **But relgap and `N` are correlated in
1D XXZ, so this does not separate them.**

The two hypotheses make opposite predictions, and one measurement distinguishes them:

| | slow draws are **overshoot** | slow draws are **small-gap** |
| --- | --- | --- |
| loosening `tol` | variance **compresses** — the tail shortens toward the median | variance **survives** — every draw shortens by a similar factor |
| what to build | tolerance guidance; possibly a `sqd` default change | nothing on the tolerance axis; the tail is a property of the subspaces |

This matters because `tol` is **traced, not static** — `run_sqd._cache_size()` stays at 1 across values,
so loosening it costs no recompilation. Measured warm on `rqutils`' own fixture:
`None → 1e-12 → 1e-9 → 1e-6` gives **5.06 → 4.21 → 3.21 → 2.58 ms**, monotonic, **1.96x** at `1e-6`. If
the tail is overshoot, this is close to free. If it is the gap, no tolerance setting helps and the effort
belongs elsewhere.

## 3. What to capture

One table. For **~10 subspace draws** at fixed `n` and fixed `dim` (production values; `dim ≈ 128k` is
where the 14.3x was seen), and each of `rtol = None`, `1e-11`, `1e-9`, `1e-6`. The `1e-11` row is the one
whose result is directly usable (§4); the looser two are there to make the *trend* legible, since a single
tight row cannot show whether compression is happening at all.

> **Run this as a standalone sweep, not through the recovery pipeline.** `markdown/skqd-sqd-solve-tolerance.md`
> §6.4 records that at `1e-9` the energy error reaches 1.8e-05 on connected subspaces — past
> `RecoveryOptions.tol = 1e-6` — which is why that document asks for `solve_tol <= 1e-11` as an assertion.
> **This request does not contradict that.** The loose settings here are an *instrument* for measuring how
> iteration count responds to the tolerance, on a fixed subspace, with the energy recorded and discarded.
> They are not a proposal to run production at `1e-6`, and nothing downstream should consume these
> eigenvalues. If the variance does compress, the usable setting is a separate question bounded by that
> `1e-11` assertion — which is precisely the guidance §4 says would then be worth writing.

| column | how |
| --- | --- |
| `draw` | the seed or identifier for the sampled subspace |
| `n`, `dim` | must be **identical across all rows** — this is the whole point |
| `rtol` | the setting for this row |
| `iterations` | see §3.1 — `sqd()` does not return it |
| `wall` | seconds for the `sqd()` call, warm |
| `eigval` | to confirm every arm found the same state |

Two things that make the table interpretable, and one that would waste it:

- **Fixed `N` and fixed Hamiltonian across every row.** If `dim` varies, the result is another
  `N`-vs-relgap confound and settles nothing.
- **Warm calls only.** A cold call includes compilation; `rqutils` has a recorded case of one cold run
  reading 125 s against a warm 20 s. Discard the first call per shape.
- **Do not average the draws.** The distribution *is* the measurement — report every row. A mean over a
  14.3x spread destroys exactly the signal being asked for.

### 3.1 Getting the iteration count

`sqd()` does not return it (3 values with `return_eigvec=True`, a bare `float` otherwise). Two routes:

```python
# Route A -- no rqutils change: call the solver directly, which returns it as the 3rd value.
from rqutils.ground_locg import ground_locg
eigval, eigvec, niter, converged = ground_locg(matvec, vinit, rtol=..., maxiter=...)
```

Route A needs the matvec `sqd` builds internally, so it is only convenient if `sqd_backend` already has
a path to it. If not, **wall-clock alone is sufficient** — at fixed shape and warm, wall time is
proportional to iteration count, and the ratio between draws is what the test reads. Iteration counts
are preferable because they are noise-free, not because wall clock fails.

There is a third route worth naming so it is not chosen by accident: `ground_locg(debug=True)` returns
per-iteration diagnostics, but it switches `while_loop` to `scan` and therefore **runs the full
`maxiter`** rather than stopping at convergence. It is an instrument for residual trajectories, not a
way to count iterations.

## 4. What `rqutils` will do with each outcome

Stated in advance so the request is not open-ended:

- **Variance compresses** → tolerance guidance goes into `sqd`'s docstring with these numbers, and the
  `rtol` default is re-examined. **Note the ceiling**: the usable setting is bounded by
  `markdown/skqd-sqd-solve-tolerance.md`'s `solve_tol <= 1e-11` assertion, so the payoff is whatever
  compression is available between `None` and `1e-11` — **not** the 1.96x measured at `1e-6`. The sweep
  should therefore include a `1e-11` row if the loose rows show any compression at all, since that is the
  only row whose speedup is actually bankable for this pipeline.
- **Variance survives** → the tolerance axis is closed and recorded as such, and the tail becomes a
  *subspace-selection* question, which is `spinchain`-side. `rqutils` would then have a concrete reason
  to look at cheap early detection of a crowded instance (the prefilter's growth factor was measured to
  be a clean, free signal) so a caller can resample rather than pay 699 iterations. That work is not
  worth starting on speculation.

Either way the open claim in `NOTES.md` and `markdown/spinchain/rqutils-tol-response.md` §2 gets closed with a
measurement instead of staying open through a third investigation.

## 5. What this request is not

- **Not a request for a code change.** No `rqutils` API change is proposed here, and none is needed to
  produce the table.
- **Not a request to re-open preconditioning.** That is closed on six measured routes with `precond`
  deleted, and deflation separately at 0.68–0.98x, 8/8 losses. The tail is not a conditioning problem
  anyone here can fix with a preconditioner on the raw indefinite projected `H`.
- **Not a warm-start request.** See `markdown/skqd-warmstart-negative-result.md` — measured 2026-09-17, no
  valid verdict, and a `vinit` parameter is declined.
- **Not blocking.** If the fixture is inconvenient to produce, say so and the claim stays open; it has
  been open since 2026-08-31 without causing harm. The cost of leaving it open is that two future
  investigations will hit the same wall, which is the argument for spending the ten minutes now.

---

## Appendix: the smallest version

If ~40 calls is too many, the minimum that still discriminates is **the two extreme draws only** — the
fastest and slowest known subspace at one `dim` — each at `rtol=None` and `rtol=1e-6`. Four calls. If
loosening `tol` pulls the slow draw's ratio toward the fast one's, the variance is overshoot; if both
shorten by the same factor, it is the gap. Use the loose `1e-6` here deliberately: it is the strongest
available signal for the binary question, and the standalone-sweep caveat in §3 is what makes it safe to
use. The 10-draw version is better because a 2-draw version cannot show the *shape* of the distribution,
but 4 calls settle the binary question.
