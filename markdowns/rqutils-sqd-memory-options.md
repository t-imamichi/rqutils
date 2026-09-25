# Two new `sqd` options for memory: `packed` and `xcache_groups`

From the `rqutils` side, for `skqd`. Version still `0.2.0` (unreleased). Both options are on `dev`:
`xcache_groups` in `ae4bdee`, `packed` in `8630c76`. Neither is released, so they arrive with your next
lockfile bump against this branch.

**Both are additive and keyword-only, so no existing call site changes.** Every parameter after `states`
is keyword-only, both default to the pre-existing behaviour, and both trace the pre-existing graph at
their defaults. Adopt either, both, or neither.

| option | what it does | adopt when |
| --- | --- | --- |
| `packed=True` | Takes `PauliSumXZ.pack_states`' output directly instead of unpacked `[N, n]` | You can build the packed array without materializing the unpacked one |
| `xcache_groups=J'` | Caches source indices for `J'` of the `J` X groups, recomputes the rest | The full source cache does not fit and `cache_level[0]=0` is too slow |

Neither changes any answer. Eigenvalue, eigenvector and returned basis are bit-identical at every
setting, pinned by `TestPackedStatesInput` and `TestPartialXCache`.

## 1. `packed=True`

```python
# before
states = build_unpacked_bitstrings()          # [N, num_qubits] uint8, one byte per qubit
energy, vec, basis = sqd(hamiltonian, states)

# after
from rqutils.paulis.symplectic import PauliSumXZ
packed = PauliSumXZ.pack_states(states)        # [N, ceil((num_qubits + 1) / 8)] uint8
energy, vec, basis = sqd(hamiltonian, packed, packed=True)
```

**Superseded 2026-08-30: `packed` now governs the returned `basis` too.** `packed=True` returns the
`ceil((num_qubits + 1) / 8)`-wide rows the solver searched, so a round trip needs no re-pack;
`packed=False` is unchanged and unpacks to `num_qubits`. This is a behavioural change for a caller that
passed `packed=True` and compared the result against an unpacked array — that comparison now sees a
shape mismatch, and `np.array_equal` returns `False` on differing shapes, so it fails loudly rather than
silently. One flag rather than a second `return_packed`: two flags make four combinations, two of which
are format conversions, and `pack_states`/`unpack_states` already are the conversion utilities.

**The saving is yours, not ours.** `sqd` has always worked on the packed form internally; what this
removes is the 8x expansion you were doing to satisfy the signature, plus the transient peak while both
arrays were live. Measured at n=100, N=400k with a caller that packs chunk by chunk and never holds the
full unpacked array:

| | build | peak RSS through `sqd` |
| --- | --- | --- |
| unpacked | +75 MB | 514.7 MB |
| packed | **+16 MB** | **456.2 MB** |

Arrays are 40.0 MB unpacked against 5.2 MB packed (7.7x at n=100).

**It only pays if you avoid building the unpacked array at all.** Calling `pack_states` on a full
unpacked array and then passing `packed=True` saves nothing — you have already paid the peak. The pattern
that wins is packing incrementally from whatever compact form you keep (integers, per-chunk arrays) and
never allocating `[N, num_qubits]`. If your recovery loop holds a set of ints, pack straight from it.

**One hazard, and it is undetectable at exactly one qubit count.** At `num_qubits == 1` the unpacked and
packed widths are both 1, so passing the wrong form is not caught: unpacked `[[0], [1]]` with
`packed=True` returns `+1.0` where the truth is `-1.0`, silently. Every other qubit count is rejected on
width, and the reverse mistake (packed array, flag omitted) is always caught by `pack_states`' binary
check. Pass the flag only for an array that came out of `pack_states`.

This is also why it is a flag rather than shape inference on our side: the widths are genuinely
undecidable there.

## 2. `xcache_groups=J'`

```python
# before -- an all-or-nothing choice
sqd(hamiltonian, states, cache_level=(1, 0))   # 4*J*states_size bytes of source indices
sqd(hamiltonian, states, cache_level=(0, 0))   # none, ~60x slower per matvec at n=100

# after -- anything in between
sqd(hamiltonian, states, cache_level=(1, 0), xcache_groups=8)
```

Requires `cache_level[0] == 1`; it raises otherwise, because there is no cache to make partial and a
silent no-op would read as "partial caching does not help on my problem". `None` (the default) and the
full group count give the same answer, but only `None` traces the single-arm graph.

Measured through `sqd()` at nq=18, J=16, N=28344 — energy bit-identical at every setting:

| `J'` | cache array | solve |
| --- | --- | --- |
| `None` | 2.1 MB | 0.18 s |
| 12 | 1.6 MB | 0.46 s |
| 8 | 1.0 MB | 0.69 s |
| 4 | 0.5 MB | 1.00 s |
| 0 | 0 MB | 1.28 s |

**Read the limits before sizing anything from this.** Three of them, all measured, and each one would
otherwise mislead:

- **The cache array is not the footprint.** Dropping groups shrinks `4 * J' * states_size` linearly, but
  the state list and the solver's vectors stay. On 1D Heisenberg at n=100 (`J = 101`, `K = 100`,
  `N = 24M`) the whole `cache_level=(1, 0)` footprint is 16.5 GB of which the cache is 13.6 GB, so
  `xcache_groups=0` saves 12.4 GB — not everything.
- **`K` decides whether this is your lever at all.** At `K = 100` the *diagonal* arrays dominate:
  `diag_signs` is 1313 bytes per state slot against the source cache's 404. So on that Hamiltonian the
  source cache is 22% of `(1, 0)`'s footprint and 5–8% of `(1, 1)`'s, and **`cache_level[1]` is the
  bigger lever** — `(0, 0)` measures 4.0 GB against `(1, 1)`'s 60.6 GB, a 15x difference available with
  no new API. At small `K` the proportions reverse and `xcache_groups` becomes the dominant dial. Quote
  your `K` when comparing figures.
- **An intermediate `J'` can *raise* peak memory.** The split runs two matvec kernels, and below a
  break-even in `J` the second kernel's intermediates cost more than the cache saves — measured 9.0 MB
  for the full cache against 10.4 MB at `J' = 8` with `J = 16`, `N = 28k`. `xcache_groups=0` always
  saves (that arm is single-kernel). Intermediate values pay off once `4 * J * states_size` is large
  next to one kernel's working set.

`floor(budget / (4 * states_size))` sizes the *cache* to a byte budget exactly. Use it for that, then
**measure** peak and speed: the time does not follow a linear model closely enough to promise (17.5% off
at the midpoint) and neither does the peak.

## 3. Sharding note

If you run on a mesh: a partial cache keeps `states` **replicated**, where a full cache reshards it after
the precompute. That is not a regression, it is required — the uncached groups search `states` inside
every matvec and `get_xsource` cannot take a partitioned array. So on a mesh, `xcache_groups` trades the
source cache against a replicated state list rather than against nothing. `tests/_sharded_partial_xcache.py`
sweeps 21 `(cache_level, J')` cells on a 4-device mesh against the single-device baseline.

## 4. What we suggest

**Take `packed=True` if your recovery loop already holds a compact representation** — it is a two-line
change, it cannot alter an answer at your qubit counts, and it removes an 8x allocation you never needed.

**Reach for `xcache_groups` only when the cache genuinely does not fit**, and check `cache_level[1]`
first: on a high-`K` Hamiltonian the diagonal axis is where the memory is, and choosing `(0, 0)` over
`(1, 1)` is worth more than any partial cache. Tell us your `K` and target `N` and we can say which
applies.

Evidence for every figure here is in `NOTES.md` — the sections on the real n=100 Hamiltonian, the
partial-J measurements, and the `xsources` budget.
