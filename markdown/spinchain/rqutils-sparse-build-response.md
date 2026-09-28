# Response: the sparse check no longer repeats the search; `ELL`'s retrace stays, with a figure

Reply to spinchain's `rqutils-sparse-build-request.md`, from the `rqutils` side. The changes are on branch
`sparse-followup` (not yet on `dev`), on top of `e4da18a`. Measured on one laptop CPU (10 cores, 64 GiB),
without the persistent compile cache.

> **Status: asks 1, 3 and 4 done; ask 5 answered with a figure; ask 2 not taken up.**
>
> - **Ask 3 was worth more than it looked.** The sparse kernels' residual check re-ran the full
>   per-group device search: 0.31 s of a 0.74 s `ELL` solve at n=60, `2^17`. It now runs on the host.
>   Warm `sqd` takes 0.23–0.25 s less on all three kernels, with bit-identical energies (§2).
> - **The check may be most of your unexplained ~2 s.** `TABLES` never paid that search, because its
>   check reuses cached source indices (§2.2).
> - **Ask 5 has no cheap fix for growing bases.** A shape memory with headroom still retraced on every
>   growing call. Constant-size turnover retraced nothing even without it (§4).

## 1. Your report, checked

Everything we could reproduce matched.

- **The build (ask 1)** is `28005b0` / `e5f241e`, the host search. Your §5 figures agree with ours.
- **The timer (ask 4)** logged before the device work finished. A dense solve logged 4.39 s with
  16.4 s of solving still to come.
- **`ELL` retraces per subspace (ask 5).** `_run_sparse` recompiled on 2 of 3 same-bucket calls at n=30,
  12k–15.5k states. It also compiled a few small eager ops from the factor assembly each time. `TABLES`
  compiled once.
- **Bit-identity across kernels.** `rqutils-matvec.md` said "All kernels return the same energy". It
  now says "to within the solve's tolerance, not bit for bit", since each kernel sums in its own order.
  Your 1e-12 comparison across kernels is the right one.
- **The 1.8e-15 shift from `698616a`** is the real-group reordering moving the energy in the last ulp,
  as its commit message records. That behaviour is intended.

## 2. Ask 3: the residual check on the sparse kernels

### 2.1 What changed

The check used to run the `"onthefly"` kernel inside the jitted solve. That meant a fresh device search
for every X group, and XLA runs that search on one core: the same cost the build had before
`28005b0`.

`sqd` now solves first, frees the sparse operator, and checks one group at a time:

- **Sources** come from the same threaded host search the build uses.
- **Diagonals** are recomputed. They are never read from the operator.

It still reads none of the operator. What it no longer does is search independently. That is the policy
the dense kernels already follow, since they reuse their cached source indices.

The check still catches a wrong pair. A sign-flipped eigenvector raises `EigenpairCheckError` on all
three kernels, and a test pins that.

| warm `sqd`, n=60 `type2`, `2^17` | before | after |
| --- | --- | --- |
| `ELL` | 0.857 s | **0.629 s** |
| `PAIRS` | 1.062 s | 0.825 s |
| `CSR` | 1.599 s | 1.348 s |

### 2.2 Your recovery gap

This is an estimate, not a measurement on your data. Scaling our 0.31 s linearly in `N` and in the
number of groups to your n=30 952k subspace (31 X groups) gives about 0.6 s per sparse solve.

That comes to about 2.3–2.9 s over 4–5 recovery rounds, close to your unexplained ~2 s. `TABLES` paid
almost none of it, because its check reuses cached source indices.

**Your re-run of the recovery sweep on this branch is the test.** If the `ELL` gap does not shrink by
about that much, the estimate is wrong, and we would like the numbers.

## 3. Ask 4: the timer

`sqd` now blocks on the result before logging `Found ground eigenpair`, for every kernel. The INFO line
is the real solve time, including the check.

## 4. Ask 5: `ELL`'s compiled shapes

We built what the ask suggested: a per-`states_size` memory of the largest bucket shapes built so far,
with every later build padded up to them. Padding adds exactly 0.0, so energies were unchanged. At n=30,
in a `2^17` bucket:

| call pattern | retraces without the memory | retraces with it |
| --- | --- | --- |
| growing ~10% per call (your rungs) | every call | every call, also with 1.25–1.5× headroom |
| constant size, 5% random turnover (recovery) | none after the first | none after the first |
| shrinking after growth | one | none |

**Growth is the obstacle, not the rounding.** Adding outer-shell states gives existing states new
neighbours inside the subspace. Rows then move to wider buckets. Width 19 went from 1 to 4 to 8 pieces
over +12% steps, so no headroom that stays small covers it.

The memory saved one compile in nine calls, so we did not keep it: it would have been module-level state
making each build depend on earlier calls. A retrace costs about 1–2 s per growing call without the
persistent cache. Your §5 figure with it is 0.34–0.54 s.

Our simulated recovery had no retrace, but yours had 26–85 cache misses per solve. So something in your
recovery differs from a random 5% swap. It could be the size changing between rounds, or H-expansion
moving the degree histogram. If you can share one round's `(states before, states after)`, we can find
which bucket moves.

## 5. Ask 2: an incremental build

Not attempted this round. With the check off the search, a build of your n=30 952k subspace is 0.39 s
(your §5), against a 3.0–3.1 s warm call. Reuse across bases could save at most that fraction.

## 6. What to re-measure

- **The n=30 pipeline, `ELL` against `TABLES`, recovery especially.** That is the check estimate in §2.2.
- **Your n=60 fixed-subspace row.** The check was also part of the 5.8 s non-build time there.
