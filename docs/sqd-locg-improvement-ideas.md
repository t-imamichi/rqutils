# SQD and LOCG improvement candidates

This note reviews `rqutils/sqd.py` and `rqutils/ground_locg.py` for credible opportunities to improve
speed, memory, scaling, or accuracy. It is a proposal list, not a benchmark report. Measurements quoted
below come from `NOTES.md`; unmeasured ideas are labelled as such.

The strongest opportunities are in SQD's operator representation and distributed data flow.
`ground_locg` is already close to the algorithmic memory floor of its three-dimensional
Rayleigh--Ritz basis, and several attractive-looking solver changes have measured poorly or harmed
residual reliability.

## Priorities

### 1. Add a paired-RHS matvec path

Each steady-state LOCG iteration evaluates the operator on `ycurr`, `tmp_p`, and then `xnext` in
`ground_locg.py`. For SQD, the first two calls independently repeat the same per-X-group preparation:
source lookup when it is uncached, diagonal construction when it is uncached, and a scan over every X
group.

A specialized optional interface could expose

```python
matvec_pair(ycurr, tmp_p, *args) -> (ay, ap)
```

and let SQD scan each X group once, prepare its source indices and diagonal once, then apply those data
to both vectors. The implementation should carry two separate output vectors through the scan rather
than materializing a stacked `(2, N)` array. Ordinary `ground_locg` callers would retain the existing
single-vector callable.

The likely benefit is largest for `(0, 0)`, `(0, 2)`, and `(1, 0)`, where source or diagonal setup is
recomputed for every matvec. It may turn three operator-setup sweeps per LOCG iteration into two.

The third application, `matvec(xnext)`, must remain fresh. Reconstructing it from previous images is the
closed `Ax`-reuse investigation: it produced a reported residual up to 59 times smaller than the true
one in float32 and thereby allowed false convergence.

Validation should:

- measure warm, whole solves across all six cache levels rather than timing the predicate alone;
- use XLA `memory_analysis()` to ensure the paired path does not increase peak live vectors;
- compare the reported residual with an independently evaluated `H @ x`;
- run `examples/scaling/poc7_sharding.py` and inspect output shardings and collectives.

### 2. Implement the measured partial diagonal cache

The diagonal cache is currently all-or-nothing. The already-measured two-arm form uses `(1, 2)` over
`J'` groups and `(1, 0)` over the remainder. It produced bit-identical results and measured:

- half the diagonal storage with a 2.45x solve-time ratio versus no diagonal cache;
- a stable 2.60--2.83x matvec ratio at `J/2` across a tenfold range in `N`;
- 16 bytes per slot of temporary overhead, or `4/J` of the memory returned by a half split.

This is a better memory-for-speed dial than partial source caching. Uncached diagonal construction is
about five times slower than a cached diagonal, whereas uncached source lookup has measured 40--60
times slower.

A possible API is `dcache_groups`, valid only when `cache_level[1] == 2`. The matvec would combine:

- `(cache_level[0], 2)` for the cached prefix; and
- `(cache_level[0], 0)` for the uncached tail.

Do not automatically split both cache axes. The four-arm combination was measured substantially worse
because the uncached source-search arm dominates. Also document that each distinct `J'` creates a new
compiled variant; this is a memory-budget setting, not a parameter to sweep casually in one process.

See `NOTES.md`, "A partial *diagonal* cache works" and "The diagonal split at large `N`".

### 3. Integrate the distributed-state prototypes after real-interconnect measurement

`states_u` remains replicated whenever source lookup is required. This leaves approximately `13 * N`
bytes resident per device even while the solver vectors and diagonal cache shard.

The repository now has exact prototypes for every necessary mechanism:

- whole-key hash ownership and routed local lookup in
  `examples/scaling/poc13_hash_partition_jax.py`;
- globally sorted, two-round distributed uniquification in
  `examples/scaling/poc14_uniquify_sharded.py`;
- diagonal construction over partitioned states, already verified with zero collectives.

Together these can change state storage from `O(N)` per device to `O(N / d)`. Whole-key hashing is
required: prefix hashing and range ownership both collapse on physical XXZ subspaces. The routed lookup
is viable only with source caching, so its communication is paid once per X group per solve rather than
once per group per matvec.

This should remain a capacity proposal until measured on a real interconnect. Existing multi-node data
show a four-GPU solve at `N = 2^20` running 32 times slower than one GPU because LOCG's reductions cross
the network. The distributed lookup adds `all_to_all` traffic, and virtual-device timings cannot decide
whether the memory saving pays.

A staged implementation would:

1. integrate distributed `uniquify_states`;
2. integrate hash-owned `get_xsource` for `cache_level[0] == 1` only;
3. keep the replicated path below a measured size/device threshold;
4. measure NVLink and network-connected meshes separately.

### 4. Preserve distributed outputs instead of always replicating them

With `return_eigvec=True`, `run_sqd` currently reshards both the eigenvector and state list to a
replicated `PartitionSpec(None)` before the public wrapper converts them to host arrays. That preserves
the existing NumPy-returning API but defeats distributed scaling at the final step: every participant
must be able to hold the full basis and eigenvector.

Add a separate device-returning API, or an explicit option that returns partitioned JAX arrays. Do not
silently alter the existing return type. A device-returning path would let callers compute distributed
observables or write local shards without ever materializing the complete result on one host.

This should follow the established `examples/svsim.py` convention for distributed output rather than
introducing a second addressability protocol.

## Further experiments

### 5. Store sparse transition pairs instead of dense source indices

The source cache stores one `int32` for every `(X group, state)`, including `-1` for absent transitions.
Physical fixtures have measured hit rates from roughly 2% to 100%. For a fixed X signature, XOR is an
involution, so present transitions form pairs. A compact cache could store each `(i, j)` pair once and
apply both directed contributions:

```text
out[i] += diagonal[i] * vec[j]
out[j] += diagonal[j] * vec[i]
```

At hit fraction `h`, two int32 endpoints per pair cost approximately `4 * h * N` bytes per group,
against the current `4 * N`. This could reduce the retained source cache by about 10x at a 10% hit rate.

This is unmeasured and has substantial risks:

- scatter performance may be worse than dense gather;
- group lengths vary, requiring offsets or checked static capacities;
- high-hit groups approach the current storage cost;
- distributed ownership complicates the two directed updates.

Benchmark whole solves over the recorded 2--100% hit-rate range. A broken or truncated pair list does
less work and will falsely look faster, so verify every matvec against the dense source-cache form before
accepting timing results.

### 6. Special-case the identity-X group

The all-zero X signature sorts first when it exists, and SQD already uses its diagonal to choose the
initial vector. The normal matvec nevertheless computes or stores an identity source array, gathers
`vec[xsource]`, and processes the group like every other X signature.

A direct `vec * diagonal` arm could remove one `int32[N]` source row and one gather per matvec. The
expected gain is modest, roughly one group out of `J`, but the change is conceptually narrow and may
matter when the identity group contains many Z terms.

Before implementing it, inspect compiled HLO to determine whether XLA already reduces the identity
gather to a no-op. Benchmark a physical Hamiltonian rather than a one-group synthetic operator.

### 7. Add a safe continuation start for growing subspaces

A normal SQD workflow repeatedly solves related, growing subspaces, but each call currently constructs a
new deterministic spread seed. A previous eigenvector can be mapped into the newly sorted basis and
blended with a spread component:

```text
xinit = normalize(embedded_previous_vector + alpha * spread_seed)
```

The spread component is mandatory. A pure embedded vector can have zero overlap with a newly introduced
lower-energy connected component, reproducing the same excited-state failure that ruled out one-hot
starts.

This could reduce LOCG iterations across Krylov or configuration-recovery rounds. It requires an
explicit state-to-state mapping API, duplicate handling, and deterministic behavior under padding. The
right benchmark is an entire growing-subspace sequence, including mapping and compilation costs, not a
single solve.

### 8. Investigate reducing LOCG's collective count without changing its residual meaning

On multi-node hardware, reductions rather than arithmetic dominate. Each iteration computes both
`norm(rnext)` and `norm(axnext)` separately. Mathematically, with normalized `x` and a residual
orthogonal to it,

```text
||Ax|| = hypot(|theta|, ||r||).
```

Using the right-hand side could remove one global reduction from the relative-tolerance scale. It must
not be adopted from the identity alone: finite-precision loss of orthogonality could make the derived
scale underestimate or overestimate the direct norm and thereby change convergence.

The experiment should first count collectives in compiled HLO, then compare both scales over shifted,
near-degenerate, float32, float64, complex, and sharded fixtures. Reject the change if it ever loosens the
criterion enough to accept a larger independently measured true residual. Any speed claim must come
from a real multi-node run; virtual devices are suitable only for correctness and HLO inspection.

The same principle applies to the several scalar inner products in `compute_sas`: check whether XLA has
already combined their reductions before designing a custom fused reduction. The previously tested
stacked one-matmul form is not a candidate; it created two large temporaries and measured 98.7 ms versus
27.7 ms for the current scatter form at `N = 16.8M`.

### 9. Re-evaluate the GPU prefilter operating point end to end

A recent GPU sweep over `ground_locg` driven by `apply_h` found `(32, 8)` at 1.38x versus only 1.07x for
the current `(32, 2)` SQD default. That harness excludes SQD setup, so it is not enough to change the
default.

Run the full grid through warm, end-to-end `sqd` calls on representative physical subspaces. Likely
outcomes are retaining `(32, 2)` as the portable default while documenting a GPU recommendation, or
exposing a profile chosen by the caller. Avoid runtime auto-tuning: `prefilter` is static, so every
distinct tuple retains another compiled executable.

### 10. Expose a size policy for `states_size`

Power-of-two padding is beneficial for small growing solves because it coalesces compilations, but it can
inflate every per-slot allocation by almost 2x. A measured finer policy, rounding in increments near
`pow2 / 8`, reduced padding from 62.4% to 3.3% in a large-size sweep with no measurable wall-clock cost,
while the same policy regressed a small-size sweep by 57%.

Instead of replacing the current default, expose or document named policies such as `"pow2"`,
`"pow2-div-8"`, and `"exact"`, with the crossover explicitly described as approximate and
workload-dependent. The existing integer `states_size` escape hatch should remain the primitive API.

### 11. Gather scalar results once per solve

The public `sqd` wrapper reads the eigenvalue and convergence flag through two separate `_host_scalar`
calls. In a multi-process job each invokes `process_allgather`. Packing the two scalar values into one
small array or pytree before host transfer could remove one process-wide collective per solve.

This is not an iteration-level improvement, but it is straightforward to benchmark and useful on
high-latency multi-node systems. The implementation must preserve `_host_scalar`'s rule that every rank
enters the same collective; branching on local addressability can deadlock.

## Ideas not to reopen without new evidence

The following have already been measured or ruled out structurally:

- reducing LOCG's three-dimensional basis or its carried vector count;
- reconstructing `Ax` instead of performing the fresh third matvec;
- mixed-precision storage of `x` or `Ax`;
- adaptive or single-pass reorthogonalization;
- compensated reductions;
- Jacobi or deflation preconditioning for the raw indefinite SQD operator;
- Bloom-filter acceleration of the one-off source precompute;
- `cache_level[1] == 1`, which is slower than level 0 and often larger than level 2;
- replacing `compute_sas` with a stacked one-matmul form;
- switching to Davidson at the same memory budget.

## Recommended order

1. Prototype the paired-RHS matvec and measure whole solves.
2. Implement the already-measured partial diagonal cache.
3. Add a distributed/device return path.
4. Run end-to-end GPU prefilter tuning.
5. Measure the distributed-state prototypes on real interconnects before integration.
6. Explore compact transition pairs and continuation starts as larger follow-up projects.
