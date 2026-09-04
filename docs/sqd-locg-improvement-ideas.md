# SQD and LOCG improvement candidates

This note reviews `rqutils/sqd.py` and `rqutils/ground_locg.py` for credible opportunities to improve
speed, memory, scaling, or accuracy. It is a proposal list, not a benchmark report. Measurements quoted
below come from `NOTES.md`; unmeasured ideas are labelled as such.

**Section 1 has since been implemented**, and its plan was wrong in the specifics — the interface it
proposed measured worse than the alternative it explicitly ruled out. Treat every unmeasured
prescription here as a hypothesis about *mechanism* rather than a design to follow.

The strongest opportunities are in SQD's operator representation and distributed data flow.
`ground_locg` is already close to the algorithmic memory floor of its three-dimensional
Rayleigh--Ritz basis, and several attractive-looking solver changes have measured poorly or harmed
residual reliability.

## Priorities

### 1. Add a paired-RHS matvec path — **DONE, and the prescription below was wrong**

Implemented 2026-09-05 as `batch_matvec` (`ground_locg`, forwarded by `run_sqd`). Two of this section's
three prescriptions were falsified by measurement, so read the outcome rather than the plan:

- **The proposed `matvec_pair(ycurr, tmp_p, *args) -> (ay, ap)` interface was built and rejected.** It
  worked, but needed a whole new `_apply_h_pair_kernel` in `sqd.py` and measured only **1.14--1.21x**
  end-to-end.
- **"Carry two separate output vectors rather than materializing a stacked `(2, N)` array" is backwards.**
  The stacked form is the one that shipped: **1.61--1.81x on the matvec pair**, 1.15--1.21x end-to-end,
  and it needs **no kernel change at all** — `apply_xgrp` already indexes `vec.at[..., xsource]` and
  scales elementwise, so leading-axis broadcast is implicit. The whole change is ~9 lines in `sqd.py`.
- **"May turn three operator-setup sweeps into two" understates it.** The larger win is fusing the
  *gather*: on a 4-device mesh the all-gathers drop **6 to 3**, because the operator's gather is paid
  once per group instead of twice. That is why the stacked form beats the paired-callable one, and why
  the benefit grows with device count. `jnp.stack` on a `P('x')` vector gives `P(None, 'x')`, so the data
  axis keeps its partitioning and nothing reshards.

The section was right that the third application must stay fresh, and it is: `xnext` is downstream of
the Rayleigh--Ritz result, so 3 matvecs become 2 calls and never 1.

**Measured across the full grid** (whole warm `run_sqd` calls, interleaved arms, energies gated to
~1e-16 first). The prediction that `(0, 0)`, `(0, 2)` and `(1, 0)` benefit most is roughly right, but the
grid also found the one level where batching **loses**:

| `cache_level` | ratio (N=2k) | note |
|---|---|---|
| `(0, 0)` | 1.24x | |
| `(0, 1)` | 1.24x | |
| `(0, 2)` | 1.22x | |
| `(1, 0)` | 1.20x | the default |
| `(1, 1)` | 1.22x | |
| `(1, 2)` | **0.93x** | see below |

`(1, 2)` caches both source indices *and* diagonals, so there is no per-matvec setup left to share and
only the stack cost remains. It is size-dependent, not a flat regression: 0.94x at N=2k, then 1.04x at
N=8k and 1.04--1.09x at N=30k as the fixed cost amortizes. `run_sqd` defaults `batch_matvec=True`
regardless, since `(1, 2)` is the rarest level (it needs the full diagonal cache resident) and the loss
is confined to small subspaces.

Two findings worth carrying forward, both of which contradicted an initial reading:

- **Memory has opposite signs by operator.** Through `run_sqd` temp *falls* a flat -16.00 B/slot (0.942x,
  N=4000--60000), because the unbatched arm holds two gather results live where the batched arm holds one
  `(2, N)` buffer. Against an elementwise operator with no gather to save, it rises a vector. So this is
  neither a "pure win" nor a memory-for-speed trade; the sign depends on the fixture.
- **Bit-identity depends on `mat`, and only `theta` is worth comparing.** XLA may contract a `(k, N)`
  operand in a different order than an `(N,)`. An elementwise operator gives exactly 0.0 on every
  diagnostic; a dense `einsum` moves `y` by 1.1e-9. The magnitude carries no information — on a
  near-degenerate subspace `y` moved 0.56 while `theta` agreed to 2.2e-15, the eigenvector being free to
  rotate within an invariant subspace.

The contract is that `mat` broadcasts over a leading axis of **any** size, not just 2: `debug=True`'s
diagnostics send three. `ground_locg` defaults to `False` (an arbitrary callable need not batch) and an
array `mat` raises, its matvec being a `jax.lax.dot` that rejects a rank-2 rhs. See `CLAUDE.md`'s
`ground_locg` section, and `tests/_sharded_batch_matvec.py` for the sharding case.

**Not batchable, checked:** the Chebyshev prefilter (its terms are a sequential recurrence) and
`body()`'s third matvec (downstream of Rayleigh--Ritz).

### 2. Implement the measured partial diagonal cache

The diagonal cache is currently all-or-nothing. The already-measured two-arm form uses `(1, 2)` over
`J'` groups and `(1, 0)` over the remainder. It produced bit-identical results and measured:

- half the diagonal storage for a **2.45x slowdown** against the full cache. Note the direction: every
  ratio in `NOTES.md`' table is "vs all-cached", so this is the cost you accept to halve the memory, not
  a speedup. No cache at all is the 5.13x row, so the dial buys back 2.68x of that 5.13x for half the
  bytes.
- a stable 2.60--2.83x matvec ratio at `J/2` across a tenfold range in `N` (same direction: vs all);
- 16 bytes per slot of temporary overhead, or `4/J` of the memory returned by a half split -- **but that
  favourable form only holds at large `J`**: it is 4.0% at `J=100` and 40% at `J=10`, so the dial is
  cheap only when there are many groups.

This is still a better memory-for-speed dial than partial source caching, because the two axes are not
equally expensive to leave uncached. Matvec-to-matvec, an uncached diagonal is 5.0--8.2x while an
uncached source lookup is 41--60x. (Do not compare the 5.13x *solve* ratio against the 41--60x *matvec*
ratio, as an earlier draft of this note did -- that overstates the asymmetry.)

A possible API is `dcache_groups`, valid only when `cache_level[1] == 2`. The matvec would combine:

- `(cache_level[0], 2)` for the cached prefix; and
- `(cache_level[0], 0)` for the uncached tail.

Do not automatically split both cache axes. The four-arm combination was measured substantially worse
because the uncached source-search arm dominates. Also document that each distinct `J'` creates a new
compiled variant; this is a memory-budget setting, not a parameter to sweep casually in one process.

**Three obstacles this note originally missed, all verified against `rqutils/sqd.py`:**

1. **Truncate the diagonal *precompute*, not just what is retained.** `xcache_groups` limits
   `hamiltonian.x[:ncached]` so only `J'` sources are ever built. A `dcache_groups` must likewise scan
   only `hamiltonian.z[:ncached]`/`.c[:ncached]`; otherwise it computes the whole diagonal, pays the peak
   memory the option exists to avoid, and merely *retains* half. The answer stays correct either way,
   so this fails silently -- exactly the class of defect this repo keeps guarding against.
2. **The precondition is necessary but not sufficient: `dcache_groups` must *exclude* `xcache_groups`,
   not merely require `cache_level[1] == 2`.** Both splits at once is the four-arm form already measured
   at 39.37 ms against the two-arm 4.97. `_check_xcache_groups` already rejects the mirror case, so the
   validator pattern exists to copy.
3. **A diagonal split reintroduces the `13 * N` state array at the one level that had eliminated it.**
   `needs_states` is `cache_level[0] == 0 or cache_level[1] == 0 or partial_xcache`, so `(1, 2)` is
   uniquely able to drop `S` entirely -- and a `(1, 0)` tail arm searches `states_u` every matvec, which
   turns that back on. The trade still wins at large `K`: a half split returns `~4*K` B/slot against
   13 B/slot of states plus 16 B/slot of temp, so +371 B/slot at `K=100`. But it **goes net-negative
   below about `K = 7`**, which is where this dial should refuse rather than silently cost memory. Quote
   `K` with any figure here, as `CLAUDE.md` requires.

See `NOTES.md`, "A partial *diagonal* cache works" and "The diagonal split at large `N`".

### 3. Integrate the distributed-state prototypes after real-interconnect measurement

**Gated measurement is in: DO NOT INTEGRATE (2026-09-05).** `poc15` ran on real nodes, 1D XXZ n=26,
N=400000, one GPU per node: **961.8 ms on 2 devices against 1728.6 ms on 4 -- 1.80x slower per
doubling**, at `|dE| = 0.0e+00`, so pure communication. This is the real-interconnect measurement this
section made itself conditional on, and it answers against integration: routed lookup adds `all_to_all`
on top of a solve that already loses 1.80x per doubling. Still a network rather than NVLink, so the
in-one-box question remains genuinely open -- but nothing here should be built until that is measured.
See `NOTES.md`, "poc15 on real nodes".

**Verified: accurate, but roughly half of this section restates `CLAUDE.md`'s `sqd` section**, which
already says states-replication is not fundamental, that whole-key hashing is required, that
`uniquify_states` needs range partitioning, that the diagonal path shards with zero collectives, and that
what remains unverified is whether the routing pays. Its unique content is the prototype file paths, the
32x multi-node datum as the gate, and the staged threshold plan below. One caveat: "the routed lookup is
viable only with source caching, so its communication is paid once per X group per solve" is a sound
*inference*, not a measurement -- nothing in `NOTES.md` measures it.

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

**Verified 2026-09-05.** The premise is exact: `run_sqd` reshards both arrays to `PartitionSpec(None)`,
and on a 4-device mesh `return_eigvec=True` costs **7 extra all-gathers** (30 to 37) with both outputs
returning `P(None)` -- every rank holds the whole eigenvector and basis. This is a genuine
`O(N)`-per-device wall on the one path that is supposed to scale, and unlike section 3 it needs no
interconnect to fix or to justify: it is an API shape, not a performance question. **Rank this above
section 3** -- section 3's memory relief is pointless while the return path re-replicates.

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

**Verified 2026-09-05 -- DO NOT BUILD.** The storage claim and the involution are both correct (`a[a[i]]
== i` checked on every X group of a 14-qubit Heisenberg subspace, and the ~10x at h=10% arithmetic is
right). Three findings kill it:

- **The sharding risk is not "complicated", it is measured fatal.** Gather form: **3 all-gathers**.
  Scatter form: **24 all-gathers and 58 scatters**, and it only compiles at all after forcing
  `out_sharding` on both the reads and the scatters -- otherwise `ShardingTypeError: out sharding could
  not be resolved unambiguously`. This is the same defect `sqd.py` already documents at its scatter site
  ("`sqd()` fails outright on ANY multi-device mesh"). The pair list's length is also data-dependent, so
  it does not divide a mesh.
- **A variable-length list cannot ride the existing scan**, whose axis must be rectangular. Padding to
  capacity restores the `4*N` the proposal exists to shed; a host-side capacity guess is the recorded
  silent-wrong-answer mode.
- **Self-pairs at `X = 0` are missed.** The identity group gives `a[i] == i` for every state -- measured
  590 self-pairs in group 0, none elsewhere -- so the identity group is simultaneously 100% hit and 100%
  self-pair, this scheme's worst case. Note that also makes sections 5 and 6 collide: 6 peels off exactly
  the group 5 handles worst.

It also does not escape the closed Bloom finding. It avoids the Amdahl cap (it attaches to the retained
cache, not the precompute), but `cache_level[0] = 0` already sheds the whole xsource cache for a measured
4.0 GB against 4.1 GB, so the ~10x is against `(1, 0)` rather than the honest floor. `CLAUDE.md`'s
**"the memory lever is the diagonal axis, not this one"** stands, which is section 2.

### 6. Special-case the identity-X group

The all-zero X signature sorts first when it exists, and SQD already uses its diagonal to choose the
initial vector. The normal matvec nevertheless computes or stores an identity source array, gathers
`vec[xsource]`, and processes the group like every other X signature.

A direct `vec * diagonal` arm could remove one `int32[N]` source row and one gather per matvec. The
expected gain is modest, roughly one group out of `J`, but the change is conceptually narrow and may
matter when the identity group contains many Z terms.

Before implementing it, inspect compiled HLO to determine whether XLA already reduces the identity
gather to a no-op. Benchmark a physical Hamiltonian rather than a one-group synthetic operator.

**Verified 2026-09-05 -- the HLO question is malformed, and the measured gain is zero or negative.**
The all-zero signature does sort first (`np.unique(xbits, axis=0)` returns lex-sorted rows), and it does
not always exist -- `sqd.py` already branches on that dynamically for the initial vector.

But `apply_h` lowers to **one `while` loop with a single `gather` in the body, shared by all `J` groups**;
the group index is the loop induction variable, so there is no per-group HLO for XLA to specialize. There
is no "identity gather" to eliminate. Measured on a J=20 Heisenberg fixture (n=20, N=131072, both arms
warm, arrays passed as arguments): baseline 0.854/0.931 ms against 0.899/1.03 ms peeled, at identical temp
bytes -- **noise to slightly worse**. Two further costs the note omits: the static branch means a **second
compiled matvec variant**, and `vec * diagonal[0]` is *not* equivalent to the gather at filler slots
(`255 ^ 0 == 255` is present in `states`, so `get_xsource` returns a valid index rather than `-1`;
measured max difference **91.1** before masking). It is safe inside `sqd` only because `_spread_seed`
happens to zero fillers -- an accident of the seed, not an invariant.

**Drop it, or keep this paragraph as the recorded negative.**

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

**Verified 2026-09-05 -- accurate throughout, and the best of the "further experiments".** The seed
description is right, including the detail that `vinit_from_min_diag` adds its weight *on top of* the
spread seed carrying the seed's own sign. The "spread component is mandatory" warning is load-bearing,
not a hedge: an embedded eigenvector is zero on every newly added state, so it reproduces the one-hot
defect exactly if a new connected component appears (`sqd.py` records the measured case, -1.293 against a
true -2.191 on a subspace splitting 4+10). It is **not** a repeat of the closed weight-shell
investigation, which is about which *states* to pick rather than the initial vector given them.

The API change is smaller than this note assumes: `ground_locg` already takes `xinit`, so only `run_sqd`
and `sqd` gain a keyword. Three real obstacles: an array-valued kwarg fits neither branch of `CLAUDE.md`'s
static-vs-`partial` rule (it must ride the traced arguments); the old-to-new state mapping is the actual
work and must avoid elementwise indexing of a sharded array (a `searchsorted` into the new `states_u` is
the natural form); and **`states_size`'s power-of-two padding exists to pin shapes**, so a continuation
API that changes shape per round pays compilation each time and can easily lose. Verify eigenvalues every
round, not just iteration counts -- a badly mapped `xinit` converges to the *wrong* answer faster.

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

**Verified 2026-09-05 -- the identity is exact, the primary proposal is marginal, and the secondary
one is the real win.** The algebra holds to 0.0 relative error (`<x|r> = 0` by construction of the
Rayleigh quotient, and it stays exact under a stale `theta` and a denormalized `x`; measured
`|<x|r>| <= 1.2e-15` and agreement to `<= 3.8e-16` at every iteration including the residual floor).

But counting the collectives -- which this section itself asks for first -- inverts its priorities. On a
4-device mesh the LOCG loop body issues **13 distinct `all-reduce` ops carrying 24 logical scalars**, and
the arity histogram is the finding:

| operands per `all-reduce` | count | what they are |
| --- | --- | --- |
| 1 | **7** | `jnp.linalg.norm` calls |
| 2 | 3 | `compute_sas` / reorthogonalization sums |
| 3 | 1 | ditto |
| 4 | 2 | ditto |

**XLA's collective combiner has already merged the `compute_sas` sums** into multi-operand reductions,
while every `jnp.linalg.norm` stays a **separate single-scalar** all-reduce -- `norm` is its own jit
boundary, which blocks the combiner from folding it into its neighbours. So:

- The proposal as written removes **one of 13 ops**, and buys it by making the convergence scale derived
  rather than measured. The error term `2*theta*Re<x|r> / (2||Ax||)` is sign-indefinite, so it can
  *loosen* the test -- though its magnitude is bounded by an orthogonality defect the two fixed
  reorthogonalization passes already hold at `<= 1.3e-16`, far below `rtol`'s own slack. This is not the
  `* n * 10` class of hazard (that factor grew with dimension); still, poor value for the risk.
- **The unnamed win is making the norms combinable at all** -- computing `||.||^2` as `jnp.sum(abs2(.))`
  so they fuse with the adjacent sums. That targets 7 of 13 ops instead of 1, needs no change to the
  residual's meaning, and needs no custom fused reduction, since the combiner demonstrably does this
  already for the un-jitted sums.

**Rescope this section to the norms' jit boundary and drop the `hypot` identity.** Also note: the
`compute_sas` half of the section is already answered -- XLA has combined them, so there is nothing to
design there.

**Promoted 2026-09-05 by the `poc15` multi-node run**, which measured 4 devices at **1.80x slower** than
2 at fixed `N` with identical energies. On a topology where adding devices costs that much, cutting 7 of
13 per-iteration `all-reduce` ops stops being a micro-optimization -- this becomes the highest-value item
for multi-node, ahead of section 2. Section 2 remains first if the binding constraint is single-device
memory.

### 9. Re-evaluate the GPU prefilter operating point end to end

A recent GPU sweep over `ground_locg` driven by `apply_h` found `(32, 8)` at 1.38x versus only 1.07x for
the current `(32, 2)` SQD default. That harness excludes SQD setup, so it is not enough to change the
default.

Run the full grid through warm, end-to-end `sqd` calls on representative physical subspaces. Likely
outcomes are retaining `(32, 2)` as the portable default while documenting a GPU recommendation, or
exposing a profile chosen by the caller. Avoid runtime auto-tuning: `prefilter` is static, so every
distinct tuple retains another compiled executable.

**Verified 2026-09-05: accurate on every checkable claim, and blocked on hardware.** Both figures match
`NOTES.md` exactly, the harness really does exclude setup (its own docstring says so), and the
static-`prefilter` warning is not speculative -- the 8th of 9 configurations hit `RESOURCE_EXHAUSTED:
Failed to load in-memory CUBIN` on a 71 GB GPU holding under 1 GB of tensors. The section also keeps the
1.07x GPU/`apply_h` figure and the 1.49x end-to-end CPU figure properly distinct, which is the trap
`CLAUDE.md` warns about. Nothing to do on a CPU-only machine; this is a faithful hand-off.

### 10. Expose a size policy for `states_size`

Power-of-two padding is beneficial for small growing solves because it coalesces compilations, but it can
inflate every per-slot allocation by almost 2x. A measured finer policy, rounding in increments near
`pow2 / 8`, reduced padding from 62.4% to 3.3% in a large-size sweep with no measurable wall-clock cost,
while the same policy regressed a small-size sweep by 57%.

Instead of replacing the current default, expose or document named policies such as `"pow2"`,
`"pow2-div-8"`, and `"exact"`, with the crossover explicitly described as approximate and
workload-dependent. The existing integer `states_size` escape hatch should remain the primitive API.

**Verified 2026-09-05 -- the documentation half is already done, and the API half should be declined.**
Both figures match `NOTES.md`. But the guidance this section asks for already shipped in `sqd`'s own
docstring ("**At large subspace dimensions, size this by hand**", with the 4.9%/39.8%/82.0% padding
figures, the 57% regression and the 62.4%->3.3% comparison), so section 10 adds nothing documentary beyond
a possible one-line `CLAUDE.md` pointer.

The named-policy enum is the wrong shape, and `NOTES.md` says so directly: "what is missing is not a
parameter -- `states_size` is already public and overridable -- but the *knowledge*", and "if a default
ever changes, make it size-dependent rather than replacing one fixed rule with another". Two of the three
names are also already spellable (`"exact"` is `states_size=states.shape[0]`, `"pow2"` is `None`), leaving
only `pow2-div-8` -- a one-liner whose crossover `NOTES.md` calls "an order of magnitude rather than a
boundary", measured on one laptop CPU with the large-`N` rows being arithmetic rather than runs. Giving an
under-measured heuristic a public name is the part to resist.

### 11. Gather scalar results once per solve

The public `sqd` wrapper reads the eigenvalue and convergence flag through two separate `_host_scalar`
calls. In a multi-process job each invokes `process_allgather`. Packing the two scalar values into one
small array or pytree before host transfer could remove one process-wide collective per solve.

This is not an iteration-level improvement, but it is straightforward to benchmark and useful on
high-latency multi-node systems. The implementation must preserve `_host_scalar`'s rule that every rank
enters the same collective; branching on local addressability can deadlock.

**Verified 2026-09-05: correct, and easier than the note assumes.** Both calls are real
(`sqd.py`'s `float(_host_scalar(result[0]))` and `bool(_host_scalar(result[-1]))`), and single-process
short-circuits before the collective, so this costs only multi-process. The float64/bool dtype mismatch
is a non-issue: `process_allgather` accepts a **mixed-dtype pytree in one call**, verified. So the change
is mechanical -- one collective instead of two, no cast, no new protocol. Small but genuinely free;
worth doing opportunistically the next time `_host_scalar` is touched.

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
- switching to Davidson at the same memory budget;
- a `matvec_pair`-style paired-callable interface for the two independent matvecs — built and measured
  at 1.14--1.21x against plain `(2, N)` batching's larger win, and it needed a second kernel (section 1).

## Recommended order

**Revised 2026-09-05 after verifying every item against the code.** Four items moved, two should be
dropped, and one section's own instruction ("first count collectives in HLO") is what demoted it.

1. **Section 2 -- the partial diagonal cache.** The only unimplemented item already measured through real
   `sqd()` solves, bit-identical across a 10x `N` range, on the axis both `CLAUDE.md` and `sqd.py` name as
   the larger lever, with the `xcache_groups` API precedent in place. Roughly a day, mostly validator and
   docstring. Read the three obstacles added above first -- especially that it reintroduces the `13 * N`
   state array at `(1, 2)` and goes net-negative below about `K = 7`.
2. **Section 4 -- a device-returning path.** Promoted above section 3. It is an API shape rather than a
   performance question, so unlike everything else distributed it needs no interconnect to justify *or* to
   fix, and section 3's memory relief is pointless while the return path re-replicates. Measured: +7
   all-gathers and `P(None)` on both outputs.
3. **Section 8, rescoped -- make the norms combinable.** *(First instead, if the target is multi-node:
   `poc15` measured 4 devices at 1.80x slower than 2, which makes the collective count the binding
   constraint there rather than memory.)* Not the `hypot` identity (1 of 13 ops, and it
   puts the convergence test at risk) but the finding underneath it: 7 of 13 `all-reduce` ops are
   single-scalar `jnp.linalg.norm` calls whose jit boundary blocks XLA's combiner, which has already
   merged every neighbouring sum. Same residual semantics, ~7x the target.
4. **Section 7 -- the continuation start.** The best of the "further experiments" and a smaller API change
   than the note assumed, but gate it on the shape-recompilation question before building.
5. **Section 11 -- one collective for the two scalars.** Mechanical, verified free (`process_allgather`
   takes a mixed-dtype pytree), multi-process only. Do it opportunistically.
6. **Section 9 -- GPU prefilter tuning.** Blocked on CUDA hardware; nothing to do here.
7. **Section 3 -- distributed states.** Correctly deferred behind a real-interconnect measurement, and
   roughly half of it restates `CLAUDE.md`.
8. **Sections 5, 6, 10 -- drop.** Section 5's scatter is measured fatal to sharding (24 all-gathers
   against 3, and `ShardingTypeError` without forced `out_sharding`). Section 6 measured zero-to-negative
   with a filler-mask trap worth 91.1 in absolute error. Section 10's documentation ask already shipped and
   its enum is the wrong shape by `NOTES.md`'s own argument. Keep their paragraphs as recorded negatives.
