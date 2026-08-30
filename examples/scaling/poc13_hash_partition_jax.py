"""POC 13: the JAX composition of POC 12 -- a distributed ``get_xsource`` under ``shard_map``.

``poc12_hash_partition.py`` validated the *algorithm* in numpy: hash each state's packed key to pick
an owning shard, sort within each shard, route each target ``S ^ X`` to its owner, binary-search the
owner's local slice, route the index back. This POC composes that in JAX with real collectives, and
verifies it is **bit-identical to** :func:`rqutils.sqd.get_xsource`.

Why this is testable here at all. ``jax.lax.ragged_all_to_all`` -- the ``MPI_Alltoallv`` analogue that
would ship exactly the per-destination counts -- is ``UNIMPLEMENTED`` on XLA:CPU, so it cannot run in
this environment. But it is a *bandwidth optimization of the same routing step*, not a different
algorithm: the dense ``all_to_all`` over fixed-capacity buckets carries the identical dataflow, and
that does run on CPU. So the composition below is complete and verified, with ragged left as an
untested GPU/TPU fast path. Its published defect (a broken reverse-mode rule, "usable forward-only",
FESOM2-JAX arXiv:2608.01546) does not apply here: ``get_xsource`` returns integer indices and is never
differentiated.

**Bucketing uses ``D`` passes of an ``[n]`` cumsum, not one ``[n, D]`` one-hot.** ``poc11`` rejected the
one-hot shape for costing ``4*N*NSH`` bytes ("34 GB at N=2^31, NSH=4 ... defeats the purpose"), and that
holds here: measured **13 B/slot for the loop against 24 for the one-hot** at ``D = 4``, with the gap
widening in ``D`` since the one-hot is ``O(N*D)`` and the loop ``O(N)``.

**The capacity is a correctness parameter, and unlike the Bloom pre-filter's it is derivable.** Each
shard sends a Poisson-distributed count to each destination, so a dense buffer must be sized for the
worst case: ``cap = m/D + sqrt(2*(m/D)*ln D)`` with ``m = N/D``, the balls-in-bins bound ``poc12``
validated. That depends only on ``N/D``, known at setup -- where the pre-filter's ``cap = hits + FP``
needed ``hits``, the unknown being computed, and so collapsed to ``cap = N``. The kernel returns an
**overflow count**, which is free (it sums a mask already computed), and a caller **must raise** on a
nonzero value rather than clamp: an undersized capacity silently drops targets. Demonstrated below --
at 0.9x the derived capacity the count reports the deficit and the result is no longer exact.

What this POC does **not** establish. Nothing ran on a real interconnect; per ``CLAUDE.md`` timings
under virtual devices are meaningless, so **no speed claim is made** and none is reported. The setup
phase (owner assignment plus the per-shard sort) is host-side numpy here -- in the library it would be
``poc11``'s phase 3, ``D`` independent ``lax.sort``s under ``vmap``. And ``uniquify_states`` remains a
separate blocker.

Run on virtual CPU devices (the default -- correctness only)::

    uv run python examples/scaling/poc13_hash_partition_jax.py

Run on real GPUs, which is the only way the timing question can be answered::

    uv run python examples/scaling/poc13_hash_partition_jax.py --devices 0,1,2,3
"""

import argparse
import os

# `argparse` before `import jax`, as in poc8/poc9: CUDA_VISIBLE_DEVICES and XLA_FLAGS are both read
# at backend initialization, so neither can be set after jax is imported. That is also why this
# preamble is duplicated per script rather than living in `_scaling_common` -- that module imports jax.
parser = argparse.ArgumentParser()
parser.add_argument("--devices", help='Comma-separated GPU ids, e.g. "0,1,2,3".')
parser.add_argument("--num-qubits", type=int, default=30)
parser.add_argument(
    "--host-devices", type=int, default=4, help="Virtual CPU devices when no --devices."
)
options = parser.parse_args()

if options.devices:
    # A filter over what the driver already exposes -- it cannot conjure a GPU, so a one-GPU box
    # correctly still reports one device and the sweep below shrinks to match.
    os.environ["CUDA_VISIBLE_DEVICES"] = options.devices
else:
    os.environ.setdefault(
        "XLA_FLAGS", f"--xla_force_host_platform_device_count={options.host_devices}"
    )

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax.sharding import AxisType
from jax.sharding import PartitionSpec as P

from rqutils.sqd import _pack_state_keys, get_xsource

# Sorts above every real key, so a padded slot never matches a target. The packed keys carry a zero
# pad bit at position 0 (see `sqd`), so all-ones is unreachable by construction.
SENTINEL = np.uint64(0xFFFFFFFFFFFFFFFF)
NUM_QUBITS = options.num_qubits


def mix64(x, xnp=jnp):
    """splitmix64 finalizer. Table-free, and the same expression must run on host and device.

    The whole-key hash is the correctness argument for the routing: a state and any target equal to
    it must land on the same shard, which holds because both sides apply *this* function to the same
    packed key. ``poc12`` records why the published prefix hash cannot be used -- on a 1D XXZ
    subspace it collapses to an imbalance of exactly ``D``.
    """
    u64 = xnp.uint64
    x = x.astype(u64)
    x = (x ^ (x >> u64(30))) * u64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> u64(27))) * u64(0x94D049BB133111EB)
    return x ^ (x >> u64(31))


def build_kernel(mesh, num_shards, cap):
    """Return a jitted distributed ``get_xsource`` for a fixed shard count and capacity.

    ``cap`` is static because ``all_to_all`` needs a fixed shape; that is the whole reason the
    capacity exists. Both are closed over rather than passed, so a change in either retraces -- which
    is correct, since each is part of the array shapes.
    """

    def kernel(local_keys, local_gidx, local_targets):
        # local_keys: this shard's OWNED keys, sorted, padded to a common length with SENTINEL.
        # local_gidx: their indices in the ORIGINAL lex order -- what get_xsource returns, and what
        #             the eigenvector is indexed by, so the mapping back is not optional.
        # local_targets: this shard's contiguous slice of S ^ X, owners arbitrary.
        owner = (mix64(local_targets) % jnp.uint64(num_shards)).astype(jnp.int32)

        # Rank of each target within its own destination bucket, as D passes of an [n] cumsum.
        # An [n, D] one-hot cumsum is equally correct and costs O(N*D) -- see the module docstring.
        slot = jnp.zeros_like(owner)
        for dest in range(num_shards):
            is_dest = (owner == dest).astype(jnp.int32)
            slot = jnp.where(owner == dest, jnp.cumsum(is_dest) - 1, slot)

        fits = slot < cap
        # Free overflow count: `fits` is already computed. A caller MUST raise on nonzero -- the
        # scatter below drops any target that does not fit, silently and with a plausible result.
        overflow = jnp.sum(~fits).reshape(1)
        safe_owner = jnp.where(fits, owner, 0)
        safe_slot = jnp.where(fits, slot, cap - 1)

        send = jnp.full((num_shards, cap), jnp.uint64(SENTINEL), local_targets.dtype)
        send = send.at[safe_owner, safe_slot].set(
            jnp.where(fits, local_targets, jnp.uint64(SENTINEL))
        )

        # Route: row j of shard i's buffer goes to shard j. This is the only communication.
        received = jax.lax.all_to_all(send, "x", 0, 0, tiled=True).reshape(num_shards, cap)

        # The step the whole design exists for: a binary search against THIS SHARD'S slice only.
        # local_keys is N/D rows, not N -- which is what removes the replicated state array.
        flat = received.reshape(-1)
        pos = jnp.minimum(jnp.searchsorted(local_keys, flat), local_keys.shape[0] - 1)
        answer = jnp.where(local_keys[pos] == flat, local_gidx[pos], jnp.int32(-1))

        # Route answers back, then read each target's answer from the coordinate it was sent at.
        back = jax.lax.all_to_all(answer.reshape(num_shards, cap), "x", 0, 0, tiled=True)
        out = back.reshape(num_shards, cap)[safe_owner, safe_slot]
        return jnp.where(fits, out, jnp.int32(-1)), overflow

    def wrapped(local_keys, local_gidx, local_targets):
        return jax.shard_map(kernel, mesh=mesh, in_specs=(P("x"),) * 3, out_specs=(P("x"), P("x")))(
            local_keys, local_gidx, local_targets
        )

    return jax.jit(wrapped)


def derive_capacity(num_targets, num_shards, slack=1.0):
    """Balls-in-bins capacity per destination: ``mu + sqrt(2*mu*ln D)``, ``mu = m/D``.

    Depends only on ``num_targets / num_shards``, both known at setup -- the property the Bloom
    pre-filter's capacity lacked. ``slack`` multiplies the bound for headroom; the overflow check
    catches an underestimate either way.
    """
    mu = num_targets / num_shards
    bound = mu + np.sqrt(2.0 * mu * np.log(max(num_shards, 2)))
    return int(np.ceil(bound * slack))


def partition(states, num_shards):
    """Assign owners by whole-key hash and sort within each shard.

    Host-side here. In the library this is ``poc11``'s phase 3 -- ``D`` independent ``lax.sort``s
    under ``vmap``, zero collectives -- and it runs once per subspace, not per ``get_xsource`` call.
    A hash scatters, so unlike a range split this sort is not free; that is the cost of the balance.
    """
    keys = np.asarray(_pack_state_keys(jnp.asarray(states)))
    owner = (mix64(keys, xnp=np) % np.uint64(num_shards)).astype(np.int64)
    per_shard = np.bincount(owner, minlength=num_shards)
    width = int(per_shard.max())
    local_keys = np.full((num_shards, width), SENTINEL)
    local_gidx = np.full((num_shards, width), -1, np.int32)
    for shard in range(num_shards):
        mine = np.nonzero(owner == shard)[0]
        order = np.argsort(keys[mine], kind="stable")
        local_keys[shard, : len(mine)] = keys[mine][order]
        local_gidx[shard, : len(mine)] = mine[order]
    return local_keys, local_gidx, per_shard


def xxz_krylov(num_qubits, max_states):
    """A real 1D XXZ subspace: states reachable from |Neel> by nearest-neighbour hops.

    ``poc12`` establishes why this fixture rather than random draws -- it is the one that exposes the
    prefix hash's collapse, because magnetization conservation keeps the high bits nearly constant.
    """
    neel = np.zeros(num_qubits, np.uint8)
    neel[::2] = 1
    frontier = {neel.tobytes()}
    seen = set(frontier)
    for _ in range(9):
        nxt = set()
        for state_bytes in frontier:
            state = np.frombuffer(state_bytes, np.uint8)
            for a in range(num_qubits):
                b = (a + 1) % num_qubits
                if state[a] != state[b]:
                    hopped = state.copy()
                    hopped[a], hopped[b] = hopped[b], hopped[a]
                    key = hopped.tobytes()
                    if key not in seen:
                        seen.add(key)
                        nxt.add(key)
        frontier = nxt
        if len(seen) > max_states:
            break
    rows = np.frombuffer(b"".join(sorted(seen)), np.uint8).reshape(-1, num_qubits)
    padded = np.concatenate([np.zeros((len(rows), 1), np.uint8), rows], axis=1)
    return np.unique(np.packbits(padded, axis=1), axis=0)


def hop_signature(num_qubits, a, b):
    """The X signature of a nearest-neighbour hop on qubits ``a`` and ``b``."""
    eye = np.eye(num_qubits, dtype=np.uint8)
    return np.packbits(np.concatenate([np.zeros(1, np.uint8), eye[a] + eye[b]]))


def run_case(mesh, states, xsig, slack=1.6):
    """Run the distributed kernel and compare against ``get_xsource``.

    ``num_shards`` is derived from the mesh rather than passed alongside it. They were separate
    parameters that had to agree, and on a box with more devices than the swept shard count they did
    not: the send buffer got ``num_shards`` rows while the mesh axis had ``jax.device_count()``, and
    ``all_to_all`` raised "The size of all_to_all split_axis (4) has to be divisible by the size of
    the named axis x (8)". A 4-device box hid it because the two coincided there.
    """
    num_shards = mesh.shape["x"]
    reference = np.asarray(jax.jit(get_xsource)(xsig, jnp.asarray(states)))
    targets = np.asarray(_pack_state_keys(jnp.bitwise_xor(jnp.asarray(states), xsig)))
    local_keys, local_gidx, _ = partition(states, num_shards)
    cap = derive_capacity(len(states) // num_shards, num_shards, slack)
    with jax.set_mesh(mesh):
        fn = build_kernel(mesh, num_shards, cap)
        shard = jax.NamedSharding(mesh, P("x"))
        args = (
            jax.device_put(jnp.asarray(local_keys.reshape(-1)), shard),
            jax.device_put(jnp.asarray(local_gidx.reshape(-1)), shard),
            jax.device_put(jnp.asarray(targets), shard),
        )
        got, overflow = fn(*args)
        got.block_until_ready()
        hlo = fn.lower(*args).compile().as_text()
    collectives = tuple(
        hlo.count(name) for name in ("all-gather", "all-reduce", "collective-permute", "all-to-all")
    )
    return {
        "exact": np.array_equal(reference, np.asarray(got)),
        "overflow": int(np.asarray(overflow).sum()),
        "hit_rate": float((reference >= 0).mean()),
        "cap": cap,
        "collectives": collectives,
    }


def shard_counts():
    """Shard counts to sweep: the powers of two dividing the visible device count.

    Derived rather than hardcoded so the same script is meaningful on 1, 2, 4 or 8 devices --
    `--devices 0,1` on a two-GPU box sweeps (1, 2), an eight-GPU box sweeps (1, 2, 4, 8). Explicit
    sharding rejects a mesh larger than the device count, so a hardcoded 4 turns a smaller box into
    a crash inside `make_mesh` rather than a smaller run.
    """
    total = jax.device_count()
    counts = [d for d in (1, 2, 4, 8, 16, 32) if d <= total and total % d == 0]
    return tuple(counts)


def backend_note():
    """One line naming the backend, and whether timings from it would mean anything."""
    backend = jax.default_backend()
    kind = "REAL" if backend != "cpu" else "virtual"
    return f"{jax.device_count()} {kind} {backend} device(s)"


def main():
    print("POC 13: distributed get_xsource under shard_map\n")
    print(f"running on {backend_note()}; sweeping shard counts {shard_counts()}")
    if jax.default_backend() == "cpu":
        print("  virtual devices: correctness only -- per CLAUDE.md, timings here are meaningless")
    states = xxz_krylov(NUM_QUBITS, 20_000)
    print(f"1D XXZ Krylov subspace, nq={NUM_QUBITS}, N={len(states)}\n")

    print("1. exactness across shard counts and hops")
    print(
        f"   {'D':>3} {'hop':>9} {'hit%':>6} {'cap':>7} {'exact':>6} {'ovf':>4} {'ag/ar/cp/a2a':>14}"
    )
    for num_shards in shard_counts():
        mesh = jax.make_mesh((num_shards,), ("x",), axis_types=(AxisType.Explicit,))
        trimmed = states[: (len(states) // num_shards) * num_shards]
        # (0,1) and (nq-1,0) touch the high-order bits, which is where a range split fails
        for a, b in ((0, 1), (NUM_QUBITS // 2, NUM_QUBITS // 2 + 1), (NUM_QUBITS - 1, 0)):
            r = run_case(mesh, trimmed, hop_signature(NUM_QUBITS, a, b))
            print(
                f"   {num_shards:>3} {(a, b)!s:>9} {r['hit_rate'] * 100:>5.1f}% {r['cap']:>7} "
                f"{r['exact']!s:>6} {r['overflow']:>4} {r['collectives']!s:>14}"
            )
            assert r["exact"], f"D={num_shards} hop={(a, b)}: differs from get_xsource"
            assert r["overflow"] == 0, "derived capacity should not overflow"
    print("   zero all-gather / all-reduce / collective-permute in every case.")

    print("\n2. the overflow check fires rather than truncating silently")
    widest = shard_counts()[-1]
    if widest == 1:
        # With one bucket the derived capacity is ~N, so even a 0.9x slack is ample and nothing
        # overflows -- the section's premise ("an undersized capacity drops targets") is false at
        # d=1, so skip it rather than weaken the assertion.
        print("   SKIPPED -- one shard means one bucket, so no capacity is undersized.")
        return
    mesh = jax.make_mesh((widest,), ("x",), axis_types=(AxisType.Explicit,))
    trimmed = states[: (len(states) // widest) * widest]
    xsig = hop_signature(NUM_QUBITS, 0, 1)
    for slack in (1.6, 0.9):
        r = run_case(mesh, trimmed, xsig, slack=slack)
        note = "<- a caller MUST raise" if r["overflow"] else ""
        print(
            f"   slack={slack}: cap={r['cap']:>6} overflow={r['overflow']:>6} "
            f"exact={r['exact']!s:>5}  {note}"
        )
    under = run_case(mesh, trimmed, xsig, slack=0.9)
    assert under["overflow"] > 0, "undersized capacity must be detected"
    assert not under["exact"], "an undersized capacity does drop targets -- hence the check"
    print("\n   Exactness and a zero overflow count coincide, which is what makes the free")
    print("   check a sufficient guard.")

    print("\n3. timing: routed search against the replicated baseline")
    if jax.default_backend() == "cpu":
        print("   SKIPPED -- virtual devices share one physical backend, so per CLAUDE.md these")
        print("   timings are meaningless. Re-run with --devices on a multi-GPU box.")
        return
    # The question the whole distributed-states line is blocked on: routing costs one all_to_all
    # per X group, and buys `13 * N` bytes per device instead of on every device. Only a real
    # interconnect can say whether that trades well. Both arms warm -- per CLAUDE.md a cold run
    # measures compilation, which reads as a catastrophic regression and is not one.
    import time

    widest = shard_counts()[-1]
    if widest == 1:
        print("   SKIPPED -- one device, so there is no routing to measure.")
        return
    mesh = jax.make_mesh((widest,), ("x",), axis_types=(AxisType.Explicit,))
    trimmed = states[: (len(states) // widest) * widest]
    xsig = hop_signature(NUM_QUBITS, 0, 1)
    reference = jax.jit(get_xsource)
    print(f"   {'arm':>26} {'ms':>9} {'per-device states':>18}")
    for label, thunk, per_device in (
        (
            "replicated get_xsource",
            lambda: reference(xsig, jnp.asarray(trimmed)),
            trimmed.nbytes,
        ),
        (
            f"hash-routed, d={widest}",
            lambda: run_case(mesh, trimmed, xsig)["exact"],
            trimmed.nbytes // widest,
        ),
    ):
        thunk()  # warm: compile before timing, never measure the trace
        start = time.perf_counter()
        for _ in range(3):
            jax.block_until_ready(thunk())
        elapsed = (time.perf_counter() - start) / 3 * 1e3
        print(f"   {label:>26} {elapsed:>8.2f}m {per_device / 1e6:>17.2f}M")
    print("   Report BOTH columns: the routed arm buys per-device memory and spends interconnect,")
    print("   so a time ratio alone cannot say whether the trade is good.")


if __name__ == "__main__":
    main()
