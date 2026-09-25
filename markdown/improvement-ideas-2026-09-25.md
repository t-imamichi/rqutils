# Improvement ideas, 2026-09-25

A review of what is still open after `markdown/sqd-locg-improvement-ideas.md` (revised 2026-09-16). None of
that list's top items had landed since. Grouped by the axis each one moves; unmeasured ideas are
labelled as such. The closed investigations in `CLAUDE.md` are not repeated here -- the remaining levers
are the **operator's storage** and the **collective count**, not the solver's arithmetic, whose 7-vector
working set is already the minimum for a 3-dim Rayleigh--Ritz basis.

## Speed (multi-node)

### 1. Combine the norm reductions -- **DONE, and the premise was wrong** (2026-09-25)

The 2026-09-16 section proposed that `body()`'s seven single-scalar all-reduces were isolated norms
hidden from XLA's combiner by `jnp.linalg.norm`'s boundary. Measured on a 4-device CPU mesh
(`test/sharded/allreduce_count.py`):

- **Baseline: 13 all-reduces per iteration**, arities `[1×7, 2×3, 3×2, 5]` -- the doc's figure reproduces.
- **All 13 lie on a single dependency chain; zero independent pairs.** The combiner had already merged
  everything it could. The singletons are sequential, not isolated.
- **Inlining the norm formula changes nothing** (still 13): XLA inlines the nested `@jit` anyway.
- **One pair was independent and missed**: `norm_s = ‖tmp_s‖` and `norm_u = ‖tmp_u‖`
  (`tmp_u = x·κ₀ + tmp_s` does not need `norm_s`). Reordering the source leaves it at 13; one stacked
  reduction over `(tmp_s, tmp_u)` gives **12**, the chain's true depth.

Shipped as that stacked reduction. **Bit-identical** θ, iteration count and eigenvector checksums over
N ∈ {64, 1000, 5000} × {f32, f64, c128} × `batch_matvec` on/off; XLA temp memory **flat** (+0 to
+192 B, constant from N=10k to 1M -- the stack fuses into the reduction). Pinned by
`TestAllReduceCount`, mutation-checked (the original form gives 13).

**Expected payoff: ~1/13 of per-iteration collective latency, unmeasured.** Virtual devices cannot time
it; `poc/sqd_multinode` on real nodes is the harness. Going below 12 needs an *algorithmic* change: the
`_project_out` passes, the two re-orthogonalization passes and the final `norm_y` are each sequential
by construction, and every one is load-bearing (`CLAUDE.md`, "Every guard in it is load-bearing").

### 2. One collective for `sqd`'s two host scalars -- **dropped, the premise was wrong** (2026-09-25)

A pytree does not merge the gathers: `process_allgather` is `jax.tree.map` over a per-leaf handler, so
`(eigval, converged)` is still two collectives from one call. And the common case may move no data --
a `P()` scalar on the full mesh takes the non-addressable branch, a `jit(identity)` onto the same `P()`
spec. Only a fully addressable input (a 1-device solve on a multi-process world) really gathers across
ranks. Stacking into one `(2,)` array would merge them, but saves one dispatch per solve against
hundreds of iterations, and the multi-process path is unverifiable in the sandbox (localhost bind is
denied). `NOTES.md` has the entry.

### 7. A cheaper residual check -- **new, decision first**

Item 6's check costs **+3.2%** at `(1, 0)` and **+7.9%** at `(1, 2)` (`J=120`, `N=30k`, measured) because
it always runs an uncached `(0, 0)` matvec, paying the `J`-fold `get_xsource` search again. When the solve
cached source indices, checking at `(1, 0)` would reuse them and still recompute every diagonal from its
signatures -- dropping the search, but no longer independent of an `xsources` defect. That trade is a
correctness-policy decision before it is a performance one; decide it, then A/B warm and interleaved.

## Memory

### 3. Partial diagonal cache

The only unbuilt item measured through real `sqd()` solves: half the diagonal memory for 2.45× the
full-cache solve time, bit-identical (`NOTES.md`, 2026-08-30). `xcache_groups` is the pattern on the
other axis. Test whether caching the **largest-`K_g` groups first** beats a prefix at equal bytes before
fixing the API. Limits: net-negative below about `K = 7`, and must not be combined with `xcache_groups`.

### 4. Return device arrays from `sqd`

`sqd` re-replicates both O(N) outputs (+7 all-gathers, measured), which makes sharding `states` pointless
until fixed. Return padded arrays plus `subspace_dim`. `run_sqd` now returns a `SqdResult` carrying exactly
those fields, so this is an `sqd`-level API choice; verifying it still needs a real multi-process run.

### 5. Split the solve by symmetry sector -- **new, unmeasured**

If `H` conserves a quantity readable from a bitstring (Hamming weight for XXZ, Z₂ parity), the projected
`H` is block-diagonal by sector. Solving each block separately makes peak memory scale with the
**largest sector**, not `N`, and turns the disconnected-block hazard `_spread_seed` exists for
(`NOTES.md`, "why the initial vector is a spread") into a saving. No prior record under "sector" or
"symmetry". **Gate:** it only pays when real sampler output spans several sectors -- the warm-start
fixture was a single sector (`NOTES.md`, 2026-09-17), so check an actual sampler first.

### 8. Sparse transition pairs instead of dense source indices -- **unmeasured**

The source cache stores one `int32` per `(X group, state)`, `-1` for an absent transition, while measured
hit rates run 2--100%. XOR is an involution, so present transitions pair up; storing each pair once costs
about `4·h·N` bytes per group against `4·N` -- **~10× less at a 10% hit rate**. Risks recorded in
`markdown/sqd-locg-improvement-ideas.md` §5: scatter slower than gather, ragged group lengths, and
distributed ownership of the two directed updates. Needs a prototype before any API.

### 9. Distributed `states` -- **built, blocked on hardware**

Hash ownership plus a local search is verified bit-identical at **16× less** per-device `states` memory,
and range-partitioned `uniquify_states` is built too (`CLAUDE.md`, "`states` must be replicated today").
What is unmeasured is whether the routing pays, which needs a real interconnect; virtual devices cannot
time it (`markdown/sqd-locg-improvement-ideas.md` §3).

## Accuracy

### 6. Independent final-residual check -- **DONE** (2026-09-25)

Shipped as `EigenpairCheckError`, always on in `sqd`: one `(0, 0)` matvec after the solve, raising
above 10× the convergence bound. Converged solves measure at most 0.96 of the bound; the cost is
+1.5–7.9% by cache level. Lets spinchain drop its own `_eigen_residual` guard. `NOTES.md` has the entry.

## Not reopened

The solver's arithmetic is closed: its basis and carried-vector count, reusing `Ax`, reduced precision,
compensated sums, adaptive re-orthogonalization, preconditioning and Davidson at matched memory are all
measured and recorded (`CLAUDE.md`, "Closed investigations"; `markdown/sqd-locg-improvement-ideas.md`,
"Ideas not to reopen"). So are the Bloom pre-filter, `cache_level[1] = 1`, the identity-X special case and
a named `states_size` policy. The levers left are the operator's storage (3, 8) and data movement (4, 9).

## Order

1. **7** -- the only item that is small and measurable on one machine; blocked only on the policy call.
2. **3** -- one measurement (largest-`K_g` first against a prefix, at equal bytes) settles the API.
3. **8** -- prototype and measure; drop it if scatter loses to gather.
4. **4**, **9** -- need a real multi-process run; **5** needs real sampler output.
