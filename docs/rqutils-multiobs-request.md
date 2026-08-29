# rqutils change request: a batched gather for many observables on one subspace

One request against `rqutils` 0.2.0 (`github.com/t-imamichi/rqutils`, branch `dev`, installed rev
`2c5f3e7`), written from the `spinchain` side. It is a sibling of
[`rqutils-requests.md`](rqutils-requests.md) — whose C1/C2/C3 shipped and are adopted, and whose A1 was
rejected — and of the withdrawn [`rqutils-precond-request.md`](rqutils-precond-request.md). It depends
on none of them and is against a different call site.

> **Status: DRAFTED, not yet sent.** Every number below is measured in-tree at n=30; the profile it
> comes from is [`n30-improvement-proposals.md`](n30-improvement-proposals.md).
>
> **This is the lowest-priority request of the three**, and deliberately so: `spinchain` can sidestep the
> whole cost by making the contraction opt-out, because the observable in question does not enter the
> energy. It is filed because that is a workaround for us, not a fix for the API — any caller that
> genuinely wants many observables on a large subspace pays this today and has no way out.
>
> One thing this request cannot see from outside, stated up front: whether the batched gather is a real
> algorithmic win or merely moves the same cache misses around. The `vmap` measurement below is evidence
> that the *easy* version of the idea does not work, which is a reason to be suspicious of the hard
> version too. `rqutils-precond-request.md` asked for a capability that turned out to be unusable on the
> only path we have; treat the "what we would like" section here as a question, not a specification.

## Where

`rqutils/sqd.py`, `get_xsource` — reached once per X group by `apply_h`, therefore once per observable
by any caller evaluating several observables against one subspace.

## What we do now

`spinchain`'s spin-current contraction evaluates **87 operators** (3 axes x `n-1` bonds at n=30) that
share one subspace, one eigenvector, and one packed state list. `sqd_backend.expvals` already hoists the
state packing out of the loop — worth a measured ~47% of a warm sweep, and the reason the remaining cost
is not setup — so what is left is one `_expval_kernel` call per operator.

## The cost, measured

Linear in `dim * n_ops`, at **0.042–0.050 us per state per operator** across five subspace sizes
(dim 0.8M through 24M, on the n=30 replay). At dim 24M that is **105 s, or 14% of the whole run**:

| final dim | js contraction | share of run |
| --- | --- | --- |
| 800,000 | 3.0 s | 8% |
| 3,000,000 | 11.0 s | 14% |
| 6,000,000 | 23.5 s | 18% |
| 12,000,000 | 50.3 s | 21% |
| 24,000,000 | 105.4 s | 14% |

### The obvious fix does not work, and that is the most useful datum here

`jax.vmap` over the operator axis is expressible — the 87 operators fall into exactly two shape classes
(58 + 29, differing in X-group count) — and it is numerically exact, agreeing with the current path to
**4.6e-19**. It is also useless where it matters:

| dim | vmap speedup |
| --- | --- |
| 141,440 | 2.5x |
| 1,000,000 | 1.5x |
| 6,000,000 | **1.04x** |

The win is per-call dispatch overhead, and it evaporates as dimension grows. Anyone reading the
87-operator loop and reaching for `vmap` should know this was tried.

### It is not bandwidth-bound either

One operator costs **70 vector-passes' worth of time**: 230 ms/op at dim 6M, against 3.3 ms for a
`jnp.vdot` over the same complex128 array (~27 GiB/s effective on this laptop). So the kernel is doing
~70x the memory traffic of a single stream, which rules out "already at bandwidth" as the explanation.

That points at the `searchsorted` in `get_xsource` — a random-access gather of `S ^ X` into an
N-element sorted array, repeated per operator over the *same* `S`. At n=30 the packed states are 4
bytes, so this takes the `searchsorted` path rather than the explicit row-wise binary search.

## What we would like

An entry point that accepts a **stack** of X-signatures and resolves all of their destination index
arrays in one traversal of the state list, so the cache misses are amortized across observables instead
of repeating per observable. Shape-wise it is the batched analogue of today's single-signature function,
and the caller already holds the signatures stacked.

Two things we cannot settle from outside, and which may well determine whether this is worth building:

- **Whether the weight-2 case admits a closed form.** Every spin-current operator is a 2-local Pauli, so
  `S ^ X` flips two bits — a *local* permutation of the state list. A general sorted-array search may be
  strictly more than that case needs, and a specialization for low-weight `X` could beat any batching.
- **Whether the gather can be shared rather than merely batched.** Distinct operators have distinct `X`
  and so cannot share a destination array, but they do share `S`. Answering many membership queries in
  one pass over sorted data is a different algorithm from N independent binary searches; whether it wins
  on a GPU, where the gather is already well optimized, is not something our CPU numbers predict.

This is the next step along an axis upstream has already moved once: replacing the concatenate-and-sort
implementation with `searchsorted` was measured at 12–25x per signature on CPU and 5.15x at N=64M on a
GH200. We are asking for a comparable win *across* operators rather than within one — while noting that
the earlier change removed an asymptotically worse algorithm, whereas this one would only remove
redundancy, so a similar payoff should not be assumed.

## What lands in spinchain if it ships

`skqd/core.py`'s `js_projection` phase stops being a fifth of a large replay, which removes the
motivation for the `[solver] js = false` opt-out in
[`n30-improvement-proposals.md`](n30-improvement-proposals.md) §4 — a config key we would rather not add.
`sqd_backend.expvals` would pass its operator stack straight through instead of looping, and its
docstring paragraph about hoisting the packing per operator would shrink to a note that the batching is
upstream's.

## Reproducing the measurements

The timings come from replaying `tmp/n30-mps/samples-2b63cb.npz` at several `max_dim` values
(`docs/n30-improvement-proposals.md` has the configs). The two standalone checks — the `vmap` A/B and the
vector-pass comparison — need only `rqutils`, `qiskit` and 64-bit jax, and are quick to rebuild from the
description above: stack `PauliSumXZ.from_paulisum(op).arrays` per shape class, `jax.vmap` the expectation
kernel over the leading axis with the state list and vector held fixed, and compare against a
`jnp.vdot` over an array of the same length. Prefix any repro with
`jax.config.update("jax_enable_x64", True)`; in 32-bit the agreement figures above do not hold.
