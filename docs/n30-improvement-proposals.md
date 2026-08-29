# Where the n=30 SKQD run spends itself, and seven things to do about it

Profiled on `tmp/n30-mps/samples-2b63cb.npz` (20k shots/circuit, frame `YX`, `diag_cache = "memory"`),
serially on one laptop. Energies are scored against the DMRG reference −11.7356188484 (`dmrg/`).

All timing claims here are measurements, not tests: nothing in `test/` asserts on time.

## The profile is the whole argument

| run | final dim | energy | recovery | rung solves | js | total |
| --- | --- | --- | --- | --- | --- | --- |
| 800k | 800,000 | −11.1316455 | 15.1 s | 18.8 s | 3.0 s | 37.0 s |
| 3M | 3,000,000 | −11.4706366 | 45.4 s | 19.3 s | 11.0 s | 76.0 s |
| 6M | 6,000,000 | −11.4955812 | 85.4 s | 19.3 s | 23.5 s | 128.4 s |
| 12M | 12,000,000 | −11.5350765 | 164.6 s | 19.4 s | 50.3 s | 234.5 s |
| 24M | 24,000,000 | −11.5602961 | 628.4 s | 20.4 s | 105.4 s | 754.4 s |

**The 12 per-rung diagonalizations are flat at ~19 s** regardless of the cap, because they run on the
sampled rungs (141,440 states at most) before recovery grows anything. All the scaling is in
`recovery` and `js_projection`.

**Round 2 of the 24M run is the worst trade in the pipeline.** Round 1 reaches dim 18,793,320 at
−11.5576517 in 290 s; round 2 spends **310 more seconds** to add 5.2M strings and buys **0.0026 Ha** —
41% of the run (49% of recovery) for 0.5% of the gain the cap bought over 400k. Marginal energy per
second, against the 3M point:

| point | dim | error | wall | mHa/s |
| --- | --- | --- | --- | --- |
| 3M | 3,000,000 | 0.2650 | 76.0 s | — |
| 6M | 6,000,000 | 0.2400 | 128.4 s | 0.476 |
| 12M | 12,000,000 | 0.2005 | 234.5 s | 0.407 |
| 24M round 1 | 18,793,320 | 0.1780 | 292.3 s | 0.402 |
| 24M round 2 | 24,000,000 | 0.1753 | 754.4 s | **0.132** |

## 1. Cost-aware stopping in recovery — recommend

`RecoveryOptions.tol` is an *absolute* energy threshold (1e-6), so a 0.0026 Ha gain correctly
continues; there is no way to express "stop when a round stops being worth its time". Add a rate
criterion — energy gain per second, or per added dimension, below a floor.

On the 24M run that stops after round 1: **292 s instead of 754 s (2.6x) for 0.0026 Ha**, and
−11.5576517 at 292 s beats every other measured configuration on energy per second.

Local, ~30 lines, and a floor of 0 reproduces today's behaviour exactly. It can change `stop` and
`iterations` in metadata, so `make test-slow` needs re-running.

## 2. Hold the subspace as packed integer codes — recommend

Recovery keeps the subspace as a Python `set` of ints and rebuilds a full bool matrix every round. At
n=30, dim 24M that matrix is **720 MB** (numpy bool is one byte per element), and 24M Python ints cost
more again in object overhead. Measured at 8M codes — a third of the 24M run, so scale by ~3:

| representation | time | memory |
| --- | --- | --- |
| `set(int)` + `sorted()` | 6.04 s | 538 MiB peak |
| `np.unique` on uint64 | 3.81 s | 61 MiB |
| bool matrix, n=30 | 0.78 s | 228 MiB |
| packed uint32 | — | **30 MiB (7.5x)** |

Holding a sorted `uint64` array and merging with `np.union1d` removes the Python-set cost and defers
the bool expansion to the `sqd()` call that needs it. This raises the reachable `max_dim` on a fixed
machine, so it is a **scaling** change as much as a memory one.

**Mind the endianness contract**: `skqd/recovery.py` states the bit-order convention lives in exactly
two functions. A packed path must route through those rather than becoming a third encode site.

## 3. rqutils: a fused multi-observable kernel — upstream

The 87 spin-current operators (3 axes x 29 bonds) each get their own `_expval_kernel` call, so the
state array is traversed 87 times: 105 s at dim 24M, linear in `dim * n_ops` (0.042–0.050 us per state
per operator across five caps).

**The obvious fix does not work, measured.** `jax.vmap` over the two shape classes (58 + 29 operators)
is correct to 4.6e-19 but gives 2.5x at dim 141k, 1.5x at 1M and **1.04x at 6M** — the win is per-call
dispatch overhead, which vanishes exactly where it would matter.

The cost is elsewhere: one operator costs **70 vector-passes' worth of time** (230 ms/op at dim 6M
against 3.3 ms for a `vdot` over the same array), so this is not bandwidth. It is the `searchsorted`
in `get_xsource` — a random-access gather of `S ^ X` into a 24M-element sorted array, once per
operator. Request written up in [`rqutils-multiobs-request.md`](rqutils-multiobs-request.md).

## 4. Make the js contraction opt-out — recommend

Independent of #3 and available today: `js_projection` is 105 s of the 24M run and contributes
**nothing to the energy** — it is a post-hoc observable. A run sweeping `max_dim` pays it every
iteration for a number nobody reads.

A `[solver] js = false` switch, combined with #1, lands the 24M configuration at **~292 s against
754 s** for 0.0026 Ha. The samples file keeps everything needed to recover the currents later in one
cheap replay.

## 5. Bound the expansion by locality, not by a global cap — speculative, needs a POC

The measured wall is neither shots nor the solver: error falls as roughly `dim^-0.21..-0.35` (see #7),
so closing the gap needs 35–455x more dimension — against a full `2^30` that is only 44.7x more than
24M. Everything above buys time or memory; only a better *choice* of strings changes that exponent, and
#7 shows raising the cap cannot substitute for it at any value.

`_expand_once` admits every one-hop child of every parent in count order until the cap stops it — at
24M, round 1 alone proposes 18.6M strings. An alternative: weight each child by its parent's
eigenvector amplitude **times** the magnitude of the Hamiltonian term reaching it, which is a
first-order perturbative estimate of the child's weight in the true ground state, and strictly more
information than a parent's sampled count.

**No number claimed, deliberately.** The adjacent idea — reordering the truncation — measured
bit-identical, and the ordering effect turned out to be instance-dependent (descending wins 9 of 12
random instances at n=8–12, ascending wins at n=30; `NOTES.md`). So: mechanism plausible, cheap
version already failed, and this needs a `poc/` script measuring energy at *matched dimension* across
several instances before anything in `recovery.py` moves.

## 6. Batch js with `jax.vmap` — measured and rejected

Correct to 4.6e-19, genuinely 2.5x at dim 141k, and **1.04x at 6M**. The speedup inverts with problem
size, so it optimizes only the regime where js already costs seconds. Recorded so it is not re-tried.

## 7. Raise `max_dim` further — exhausted, not recommended

It produced the 75% error reduction and is **exhausted, not merely diminishing** — the dimension the
extrapolation asks for does not exist at n=30.

Least squares on `log10(err)` against `log2(dim)`, from the profile table above:

| window | fit | exponent | R² |
| --- | --- | --- | --- |
| last four caps (3M–24M) | `log10(err) = 0.7557 - 0.0616*log2(dim)` | `dim^-0.205` | 0.989 |
| all five (800k–24M) | `log10(err) = 1.7983 - 0.1062*log2(dim)` | `dim^-0.353` | 0.912 |

The exponent drifts with the fit window — the 800k point sits well off the line, which both steepens
the five-point fit and costs it the R² — so treat **0.21–0.35** as the range rather than a constant, and
prefer the four-point fit for extrapolating from the current regime. The three measured doublings buy
9%, 16% and 13% error reduction for 1.69x, 1.83x and 3.22x the wall time.

**Why the lever is spent, and not by a small margin.** `2^30` is only **44.7x** more dimension than
24M. Reaching 0.05 Ha needs 455x more at `dim^-0.205`, or 35x at `dim^-0.353` — so the target straddles
the full Hilbert space depending on which fit you believe, and the better-conditioned fit puts it an
order of magnitude beyond it. Worse, the power law is self-evidently invalid that far out: at the full
`2^30` it predicts 0.080 Ha (or 0.046 Ha), where the true answer is *exact*. Any extrapolation of 10x
or more is reading the fit past the point where it means anything, which is the real reason to stop
here — stronger than "diminishing returns" and not a statement about wall time at all.

At n=20 this lever converges because one-hop expansion eventually reaches all `2^20` states; at n=30,
24M is 2.2% of `2^30`, and the gap is not closable by raising the cap. **That leaves §5 as the only
lever that can move the exponent**, which is a stronger argument for building its POC than the section
itself makes.

Proposal 2 raises the ceiling on a fixed machine, which is worth having as headroom — not as an
accuracy strategy.

## Suggested order

1. **#1 + #4 together** — ~40 lines, no new dependencies, and they turn the best-energy configuration
   from a 754 s run into a 292 s one. Both default to current behaviour.
2. **#2** — the memory work, which is what makes larger caps reachable and pays back in every solver
   that touches a subspace.
3. **#3 upstream**, with the 70-vector-passes figure attached.
4. **#5 as a `poc/`** — the only idea that could move the scaling exponent, and the one that must not
   reach `recovery.py` without a multi-instance measurement.
