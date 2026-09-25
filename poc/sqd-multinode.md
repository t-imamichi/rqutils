# `sqd` across nodes, one GPU per node

A 4-node cluster with a single GPU per node, networked (not NVLink), 2026-09-04 to 2026-09-07. Scripts:
`poc/sharding.py`, `poc/prefilter_gpu.py`, `poc/uniquify_sharded.py` and `poc/sqd_multinode.py`, all
through `poc/_scaling_common.py` (§8). Multi-node **correctness** is settled; multi-node **speed** is
negative at every size measured. The library's own multi-process defect (a scalar host read) is recorded
separately in `NOTES.md`, "`sqd` could not return its own eigenvalue multi-process, and I audited past it
once"; the rules behind it are in "What multi-process *does not* require of the library, and why the
POCs still broke" and "A local gather is not a gather".

## 1. Five harness failures before the library's own surfaced

`mpirun` is not optional on this cluster -- it is the only way to reach 4 GPUs -- and every POC in `poc/`
assumed a single process owning several local GPUs. Five distinct failures, in the order they appeared,
each hidden behind the previous one. The library's own defect took one more run to surface, so read these
as the *harness* failures they were, not as evidence the library was clean.

1. **`jax.distributed.initialize` is the whole difference, and its absence is not an error.** Under
   `mpirun -n 4` without it, each rank comes up as an *independent* JAX client that sees all 4 GPUs, so
   `jax.devices()` prints `[CudaDevice(0..3)]` four times and everything looks right. `examples/sqd.py`
   already had the fix (`--gpus mpi` → `initialize(cluster_detection_method="mpi4py")`); the scaling POCs
   never got it. `_scaling_common.init_devices` now covers all three modes and **raises** when a launcher
   put several ranks in the job without `--devices mpi`, because the symptom otherwise arrives minutes
   later as failure 2.
2. **`jax.make_mesh` rejects multi-slice topologies outright** -- *after* `initialize` has correctly
   formed the cluster, which makes it read as an initialization problem. Its source settles it: it routes
   through `mesh_utils.create_device_mesh`, then raises whenever the chosen devices carry more than one
   distinct `slice_index`. **One GPU per node means one slice per node**, so an N-node job hits this
   unconditionally. Its message names `create_hybrid_device_mesh`, which wants a `dcn_mesh_shape` split.
   For a **1-D** mesh none of that is needed: `create_device_mesh`'s topology optimization orders devices
   for *multi-dimensional* meshes on a torus, and a 1-D mesh has no ordering choice.
   `_scaling_common.make_1d_mesh` constructs `Mesh` directly over `jax.devices()`, verified bit-identical
   to `make_mesh` on a single slice (same devices, shape, axis_types, and `Mesh` equality), so it is a
   drop-in.
3. **A module-scope `jnp` call initializes the backend at import**, after which `initialize` refuses with
   "must be called before any JAX calls that might initialise the XLA backend". `poc/uniquify_sharded` had
   `SENTINEL = jnp.uint64(0xFFFFFFFFFFFFFFFF)` at module level. The fix is `np.uint64`, not a bare Python
   int -- `0xFFFFFFFFFFFFFFFF` exceeds int64 and `jnp.where` raises `OverflowError` on the untyped literal,
   which is how the first attempt failed. Pinned by asserting `xla_bridge._backends` is empty after
   importing the module.
4. **Closing over a sharded array is legal single-process and illegal across processes.** A rank
   addresses only its own shard, so a `functools.partial` over globally-sharded arrays raises "Closing
   over jax.Array that spans non-addressable (non process local) devices". `poc/prefilter_gpu` built its
   own partial over `apply_h`; the fix is `ground_locg`'s `args`, which it splats as `matvec(vec, *args)`.
   This is the *same* split `run_sqd` already uses for another reason (`cache_level` must stay static or
   the kernel retraces every matvec): the library's performance requirement and multi-process correctness
   want the identical shape.
5. **Getting a sharded array to the host took three tries**, each fix exposing the next constraint. Read
   this before touching any gather:
   - `np.asarray` on a globally-sharded array raises "Fetching value for jax.Array that spans
     non-addressable devices".
   - `process_allgather`'s **default `tiled=False` stacks a *fully addressable* array into a new leading
     axis** -- `(4, 2)` becomes `(1, 4, 2)`. Single-process, that silently made `np.array_equal` against
     the reference return False, and `poc/uniquify_sharded`'s exactness flipped True → False at `d=2`. The
     docstring distinguishes the addressable and non-addressable cases; only the latter ignores `tiled`.
   - `process_allgather` **still fails on a sub-mesh**. `poc/uniquify_sharded` sweeps
     `jax.devices()[:num_shards]`, so at `d=2` on a 4-process job ranks 2 and 3 hold no shard and its
     internal `addressable_data(0)` raises `FullyReplicatedShard: Array has no addressable shards`.
     Measured: ranks 0-1 printed `True`, ranks 2-3 crashed, then the shutdown barrier timed out at
     **2/4 tasks**.

   The working form is a **non-collective** gather: concatenate `addressable_shards` sorted by global
   index (`sh.index[0].start`), and return None off-mesh so a rank with nothing to say skips the
   comparison instead of asserting on an absent value. Sorting matters -- shard arrival order is not
   global order.

**Two diagnostic notes.** Output from 4 ranks interleaves, so one failure appears four times and a partial
failure (2 of 4) is easy to misread as a flake -- count the tracebacks. And the gRPC "failed to connect to
... Connection refused" noise at the end of a failed run is the coordination service being torn down,
*downstream* of the real error; it is never the cause.

## 2. Correctness on 4 nodes: settled

`poc/sharding.py` passed in full on **4 real GPUs across 4 nodes**: all six `cache_level` cells, every
`N mod mesh.size` in 0-3, and the `return_eigvec` reshard round trip, worst `|sharded - single|` =
**4.441e-16** and `‖Hv-ev‖/‖v‖` = 2.670e-15. The six-cell sweep matters because two sharding bugs once
hid in the `cache_level[0] == 0` cells and the first masked the second. `poc/prefilter_gpu`'s Claim 3
also passed, asserting the output *spec* `P('x',)` rather than only the value -- the point, since a
silently replicated run agrees with single-device to exactly 0.0.

## 3. The first wall clock: 32x slower at `N = 2^20`

In the same 2026-09-04 run, the sharded solve took **9010 ms against 279 ms single-device, 32x SLOWER**,
at *identical* iteration counts (298 plain, 253 prefiltered). Identical counts mean pure communication,
not a different computation. At `N = 2^20` each device holds ~262k elements while every LOBPCG
iteration's `O(N)` reductions cross a **network** rather than NVLink.

It is a real measurement of the pessimistic topology, and it locates the crossover far above this size --
but it is **one size on one interconnect**, and a quantity measured at one size is not a law. It says
nothing about several GPUs in one box, where the same collectives run over NVLink. Record which
interconnect produced any such figure.

## 4. Device-count sweep at fixed N: memory shards, speed does not

`poc/sqd_multinode.py` sweeps device count at fixed `N` and reports per-device memory, asked of XLA
rather than a formula, beside the wall clock. Fixture: 1D XXZ `n=26`, `Jz=0.8`, N=400000 (J=27,
maxK=26), float64, `cache_level=(1, 0)`, one GPU per node, `--devices mpi`. The final run (2026-09-07)
is the full `for n in 1 2 4` sweep with `--reference-energy` threaded from the 1-device row:

| devices | temp MB | vs 1-dev | ideal | excess | ms | vs 1-dev | vs prev |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 87.02 | -- | 87.02 | -- | 367.7 | -- | -- |
| 2 | 48.28 | 1.80x | 43.51 | +4.77 | 871.5 | 2.37x slower | 2.37x |
| 4 | 27.16 | **3.20x** | 21.75 | +5.41 | 1491.2 | **4.06x slower** | 1.71x |

`|dE| = 7.1e-15` at both multi-rank points against the 1-device reference, so the energy is right and
every millisecond of the regression is communication.

**Claim 1, memory: the working set shards.** `temp MB` falls 3.20x over 4 devices -- "falling but not
halving", as the script predicts. The excess over ideal is **flat at ~+5 MB** rather than growing: the
signature of the known replicated `states` term (`13 * N` per device, the one the `(0,0)` floor cannot
shed), not of a leak. On an n=14 rehearsal fixture `temp MB` falls 0.52 → 0.32 → 0.23 MB across 1/2/4
devices (2.23x), the same shape.

**Claim 2, speed: the cost is crossing the network at all, not each device.** 1→2 is 2.37x and 2→4 only
1.71x, so the dominant term is collective **count**, not payload per device or hop count.

**Retracted: the 1.80x-per-doubling slope.** The first real run (2026-09-05) got only the 2- and 4-device
points, because the 1-device rank exited non-zero and `mpirun` aborted the job:

| devices | ms | vs 2-device |
| --- | --- | --- |
| 2 | 961.8 | -- |
| 4 | 1728.6 | **1.80x SLOWER** |

Both numbers stand as measured, but the ratio was computed across separate runs rather than reported by
the script (every printed `speedup` column said "baseline", each rank count being its own job), and the
*slope* reading does not survive the 1-device anchor: 2→4 is the cheaper of the two doublings. Its
`|dE| = 0.0e+00` checked nothing either -- without `--reference-energy` a single-row job compares its
eigenvalue to itself (§5). The 7.1e-15 above is the real check.

Caveat on the whole section: multi-**node** over a network, the pessimistic topology; nothing about
NVLink. `N` is fixed across all rows, so this is a device-count curve for one fixture, not a scaling law.

## 5. Instrument failures in `poc/sqd_multinode`, and their fixes

- **The first run's memory columns all read `0.0` -- an instrument failure, never measured.** The probe
  bracketed a `solve()` returning `float(sqd(...))`, so every device array was freed before the second
  reading and both samples took the resting allocator value -- structurally zero on any backend. The
  script's own VERDICT could not catch it: its advice was "a FLAT delta means nothing sharded", and flat
  and absent both print `0.0`. Fixed by sampling while the arrays are still referenced, which
  `poc/gpu_unverified`'s docstring already recorded as the fix for the identical defect.
- **Routing `return_eigvec=True` through `sqd` does not fix it**: `sqd` converts on the way out
  (`np.array(eigvec[...])`, `np.asarray(basis_states)`), so holding those keeps no device memory alive
  and the delta stays `0.0` with more machinery in the way. `run_sqd` is the innermost layer whose
  outputs are still `jax.Array`. An exact zero now prints `0?`, since a real solve allocates `O(N)` vectors
  and 0 B can only mean the reading missed them. That fix was first verified only under virtual devices,
  where the CPU backend has no allocator accounting and correctly reports `n/a`.
- **Even fixed, `delta MB` cannot answer Claim 1; `temp MB` does.** `delta MB` sees only what `run_sqd`
  returns, and `eigvec`/`basis` come back replicated, so it sat at exactly 6.00 on every row (`eigvec`
  float64[524288] = 4.00 MB plus `basis` uint8[524288,4] = 2.00 MB). `temp MB` is `memory_analysis()`'s
  `temp_size_in_bytes`, the per-device scratch where the solver's `O(N)` working set lives.
- **The `-n 1` job must exit 0.** It used to exit 1 through the "only one device" bail-out, aborting the
  whole `for n in 1 2 4` loop on its first iteration. It is now a sweep point.
- **The energy check needs `--reference-energy`**, since each job measures one mesh size. A row with no
  second arm prints `n/a` for `|dE|` and the VERDICT says invariance was not checked. Paste the reference
  at full `repr` precision (details in the script's docstring).
- **`jax_enable_x64` omitted reads exactly like a sharding bug.** Measured first-hand writing the script:
  it moved the 1-device energy from -23.782182463507 to -23.782182693481 and fired the cross-device
  assertion at 3.8e-06. With x64 the spread is 3.6e-15 across 1/2/4 devices.

## 6. What it means

- **Multi-node correctness for `sqd` is settled** (§2); multi-node speed is **negative** at both sizes
  measured (32x at `N = 2^20`, 4.06x at N=400000 on 4 nodes).
- **Memory does shard** -- 3.20x over 4 devices, with the replicated `states` term as a flat ~+5 MB -- so
  multi-node is a capacity lever, not a speed one, on this interconnect.
- **Collective count is the measured cause.** `markdown/sqd-locg-improvement-ideas.md` §3 (routing
  hash-partitioned state lookup) was gated on this measurement and stays **do not integrate**: it adds
  `all_to_all` to a solve already 4.06x underwater. §8 (cutting 7 of the 13 per-iteration `all-reduce`
  ops) is the *only* lever aimed at the measured cause, since more than half the collectives per iteration
  would go; it is no longer a micro-optimization.

## 7. Open

1. **Several GPUs in one box over NVLink** -- unmeasured; nothing here transfers to it.
2. **More than one `N`** -- every speed figure is one fixture at one size; the crossover lies somewhere
   above `N = 2^20` on this network.
3. **§8 of `markdown/sqd-locg-improvement-ideas.md`**, measured on this cluster.
4. **`poc/uniquify_sharded`'s routing cost** on a real interconnect -- only its correctness and gather were
   exercised here.

## 8. The scripts

Every script parses its flags before `import jax` (`CUDA_VISIBLE_DEVICES`, `XLA_FLAGS` and
`jax.distributed.initialize` are read at backend initialization), then calls
`_scaling_common.init_devices`, which has three modes: `--devices mpi` (one GPU per MPI rank, needs
`--extra mpi`), `--devices 0,1,2,3` (one process, several local GPUs, via `CUDA_VISIBLE_DEVICES`), or no
`--devices` (virtual CPU devices, correctness only). Meshes come from `_scaling_common.make_1d_mesh`.

| script | what it measures | multi-node launch |
| --- | --- | --- |
| `sharding.py` | correctness: six `cache_level` cells, `N mod mesh.size`, `return_eigvec` round trip; no timings | `mpirun -n 4 uv run --extra mpi python poc/sharding.py --devices mpi` |
| `prefilter_gpu.py` | prefilter iteration counts, wall clock, Claim 3 output spec | `mpirun -n 4 uv run --extra mpi python poc/prefilter_gpu.py --devices mpi` |
| `uniquify_sharded.py` | sharded `uniquify_states`, bit-identical; only multi-process mode exercises the gather | `mpirun -n 4 uv run --extra mpi python poc/uniquify_sharded.py --devices mpi` |
| `sqd_multinode.py` | per-device `temp MB` / `delta MB` and wall clock against device count | one job per rank count, below |

`sqd_multinode.py` measures **one mesh size per job**: every rank must take part in every mesh, so a
sub-mesh over `jax.devices()[:k]` would exclude whole processes (measured: the 2-device row raised
`FullyReplicatedShard` on exactly the 2 excluded ranks). Get the curve from one job per rank count, and
pass the `-n 1` eigenvalue to the rest:

```bash
mpirun -n 1 uv run --extra mpi python poc/sqd_multinode.py --devices mpi   # prints E_ref
for n in 2 4; do
    mpirun -n $n uv run --extra mpi python poc/sqd_multinode.py --devices mpi --reference-energy <E_ref>
done
```

Its other flags: `--num-qubits` (default 26), `--max-states` (400000), `--jz` (0.8), `--cache-level`
(`1,0`), `--host-devices` (4, virtual devices when no `--devices`). `prefilter_gpu.py` takes
`--num-qubits` (26), `--num-states` (1000000), `--num-xgroups` (30), `--degrees` (`8,16,32`) and
`--cycles` (`2,4,8`); `uniquify_sharded.py` takes `--num-qubits` (100) and `--host-devices` (4).
