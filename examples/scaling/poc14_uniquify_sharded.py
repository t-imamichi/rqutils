"""POC 14: a sharded ``uniquify_states`` -- two routing rounds, bit-identical output.

``uniquify_states``' ``jax.lax.sort`` is the last unsharded operation in the ``sqd`` path and the sole
remaining cause of the ``N <= 2^31`` ceiling: a sort refuses a partitioned sorting axis, so the whole
state list must live on one device. ``poc11_range_partition.py`` made the *sort* shardable and named two
gaps in its own docstring -- host-side splitter selection, and no reassembly into the
``[states_size, B]`` contract. This POC closes both, plus a third they hid.

**Range partitioning, not the hashing of poc12/poc13.** This function's output feeds
:func:`rqutils.sqd.get_xsource`, which binary-searches, so it must be **globally lex-sorted** -- and a
hash destroys global order by construction. Range splitters guarantee bucket ``i`` < bucket ``i+1``, so
concatenating buckets in order is globally sorted. That is a real asymmetry between the two functions:
``get_xsource`` needs *balance* and gets it from hashing, ``uniquify_states`` needs *order* and must
accept range splitting's imbalance.

**Why two routing rounds.** The output is a sharded array, so shard ``s`` owns a contiguous block of
rows. But bucket ``k``'s unique rows land at a **data-dependent** global offset -- the prefix sum of
earlier buckets' unique counts -- which does not align with output-shard boundaries. Measured on the
fixture below: offsets ``[0, 22279, 50782, 96614]`` against a shard size of 65,536, so **2 of 4 buckets
straddle a boundary** and their owner must send rows to more than one output shard. Zero crossings would
be luck, not a property. Hence: route to bucket owner, sort/dedupe, route to output-shard owner.

Round 2 is also what makes the output spec *honest*. An earlier attempt had each shard build a private
``[d, cap]`` block and declared ``out_specs=P(None, None, None)``; ``check_vma`` rejected it, correctly,
because those blocks are per-shard **partial** results. Passing ``check_vma=False`` would have silenced a
real error and produced a silently wrong answer. After round 2 shard ``s`` holds exactly output rows
``[s*SS/d, (s+1)*SS/d)``, so ``P('x', None)`` is true rather than suppressed.

**The capacity model is NOT balls-in-bins, and that is the main lesson here.** ``poc13`` sizes its
routing capacity as ``mu + sqrt(2*mu*ln d)``, which is right **for a hash**: each element picks its
destination independently at random. A range partition violates that assumption outright -- the
destination is the element's *value* bucket, and bucket sizes are set by the data distribution. Measured
buckets ``[44557, 57006, 47400, 60437]`` mean a shard sends ``60437/d ~ 15,109`` rows to the largest
bucket's owner, not the ``N/d/d = 13,088`` the bound predicts, and a 1.2x slack over that mean still
overflowed by 42,712. ``poc11`` already had the right rule: *"slack must exceed the splitter imbalance;
1.35 was sufficient in every fixture here."*

Round 2 needs a different baseline again. A bucket's unique rows land **contiguously**, so they reach only
1-2 output shards rather than all ``d``; its worst per-destination is therefore ~the whole bucket, so size
off ``ss/d``, not ``ss/d/d`` -- understating it by a factor of ``d``.

**And an overflow guard must count only LIVE elements.** The first working version reported
**overflow 763,677 beside a bit-exact result**, because round 2 funnels every dead row into one bucket,
which overflows by design and is then dropped harmlessly. The count conflated discarded padding with lost
data. A guard that fires on correct input is worse than no guard -- it trains a caller to ignore the one
signal that matters, which is how ``poc11``'s ``cap`` bug shipped.

What this POC does **not** establish. Virtual devices only, so per ``CLAUDE.md`` timings are meaningless
and **no speed claim is made**. Two routing rounds with 24 ``all-to-all`` at ``d = 4`` is materially more
communication than the single round this line originally assumed; whether it pays for the 27.9 GB/device
of replicated ``states`` it removes needs real interconnect measurement. ``states_size`` and both
capacities are ``static_argnums``, and ``N`` must divide ``d``.

Run on virtual CPU devices (the default -- correctness only)::

    uv run python examples/scaling/poc14_uniquify_sharded.py

Run on real GPUs, which is the only way the routing-cost question can be answered::

    uv run python examples/scaling/poc14_uniquify_sharded.py --devices 0,1,2,3
"""

import argparse
import functools

# `argparse` before `import jax`, as in poc8/poc9: CUDA_VISIBLE_DEVICES and XLA_FLAGS are both read
# at backend initialization, so neither can be set after jax is imported. That is also why this
# preamble is duplicated per script rather than living in `_scaling_common` -- that module imports jax.
parser = argparse.ArgumentParser()
parser.add_argument(
    "--devices",
    help='Comma-separated GPU ids, e.g. "0,1,2,3", or "mpi" for one GPU per MPI rank.',
)
parser.add_argument("--num-qubits", type=int, default=100)
parser.add_argument(
    "--host-devices", type=int, default=4, help="Virtual CPU devices when no --devices."
)
options = parser.parse_args()

# CUDA_VISIBLE_DEVICES / XLA_FLAGS / jax.distributed.initialize all have to precede backend
# initialization, so device setup is deferred to init_devices, called first thing in main().

import jax
import numpy as np
from _scaling_common import init_devices, make_1d_mesh
from jax.experimental.multihost_utils import process_allgather

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax.sharding import PartitionSpec as P

from rqutils.sqd import _pack_state_words, uniquify_states

# Sorts above every real packed word, so a padding slot never compares equal to real data. The
# packed rows carry a zero pad bit at position 0 (see `sqd`), so all-ones is unreachable.
# np.uint64, NOT jnp.uint64: a jnp constructor at module scope initializes the XLA backend at import,
# after which jax.distributed.initialize refuses with "must be called before any JAX calls that might
# initialise the XLA backend" -- which is how the 4-node run failed. A bare Python int is not a fix
# either: 0xFFFFFFFFFFFFFFFF exceeds int64, so jnp.where raises OverflowError on the untyped literal.
# numpy carries the unsigned dtype without touching XLA.
SENTINEL = np.uint64(0xFFFFFFFFFFFFFFFF)
NUM_QUBITS = options.num_qubits


def ge_lex(rows, splitter):
    """``rows >= splitter`` in full lexicographic order over the word axis.

    Full-row, not lead-word. Comparing only the leading word is *sound* -- it never produces an
    ordering violation -- but it cannot see the tail, and on a fixture whose lead word is constant it
    puts every row in one bucket. That is the same collapse ``poc11`` documented for equal-range
    splitting on the most-significant word.
    """
    greater = jnp.zeros(rows.shape[0], bool)
    equal = jnp.ones(rows.shape[0], bool)
    for j in range(rows.shape[1]):
        greater = greater | (equal & (rows[:, j] > splitter[j]))
        equal = equal & (rows[:, j] == splitter[j])
    return greater | equal


def bucketize(vals, num_buckets, key, cap, fill, live=None):
    """Scatter ``vals`` into ``[num_buckets, cap]`` by ``key``; return the blocks and an overflow count.

    Rank within a bucket comes from ``num_buckets`` passes of an ``[n]`` cumsum rather than one
    ``[n, num_buckets]`` one-hot: ``poc11`` rejected the one-hot for costing ``O(N*d)`` where this is
    ``O(N)``.

    ``live`` marks which elements are real data, and the overflow count covers **only** those. Round 2
    funnels every dead row into a single bucket, which overflows it by design and then drops those rows
    harmlessly -- counting them reported 763,677 alongside a bit-exact result. A guard that fires on
    correct input trains a caller to ignore it.
    """
    if live is None:
        live = jnp.ones(key.shape[0], bool)
    slot = jnp.zeros_like(key)
    for k in range(num_buckets):
        slot = jnp.where(key == k, jnp.cumsum((key == k).astype(jnp.int32)) - 1, slot)
    fits = slot < cap
    safe_key = jnp.where(fits, key, 0)
    safe_slot = jnp.where(fits, slot, cap - 1)
    blocks = jnp.full((num_buckets, cap) + vals.shape[1:], fill, vals.dtype)
    blocks = blocks.at[safe_key, safe_slot].set(jnp.where(fits[:, None], vals, fill))
    return blocks, jnp.sum(~fits & live)


def capacities(num_states, states_size, num_shards, slack=1.35):
    """Per-destination capacities for the two routing rounds.

    **Not balls-in-bins.** That bound assumes independent random destinations, which holds for
    ``poc13``'s hash and fails for range splitters, whose bucket sizes are set by the data. ``poc11``'s
    rule is the applicable one: slack must exceed the splitter imbalance, and 1.35 sufficed in every
    fixture there and here.

    Round 1: a shard sends up to ``max_bucket / d`` rows to the largest bucket's owner.
    Round 2: a bucket's unique rows are contiguous, so they reach only 1-2 output shards rather than
    all ``d`` -- its worst per-destination is ~the whole bucket, hence ``states_size / d`` and not
    ``states_size / d / d``, which understates it by a factor of ``d``.
    """
    return (
        int(np.ceil(num_states / num_shards * slack)),
        int(np.ceil(states_size / num_shards * slack)),
    )


@functools.partial(jax.jit, static_argnums=(2, 3, 4, 5, 6, 7))
def sharded_uniquify(states_p, words, states_size, num_shards, cap1, cap2, nsample, mesh):
    """Sharded ``uniquify_states``: returns ``([states_size, B], unique_count, overflow)``.

    ``states_p`` and ``words`` are partitioned ``P('x', None)``; the output is partitioned the same
    way. A nonzero overflow means data was dropped -- a caller must **raise**, not clamp.
    """
    num_states, num_bytes = states_p.shape
    num_words = words.shape[1]
    shard_out = states_size // num_shards

    # Splitters are computed here, outside shard_map, and PASSED IN. They cannot be closed over:
    # "Closing over inputs to shard_map where the input is sharded on `Explicit` axes is not
    # implemented" -- the error names the workaround.
    #
    # The sample is tiny and must be replicated (every shard needs every splitter) while `words` is
    # partitioned, so the gather needs an explicit out_sharding. XLA implements that replication as
    # an all-reduce: 4 KB here, and O(nsample) -- independent of N, so cheap rather than free.
    idx = (jax.lax.broadcasted_iota(jnp.int32, (nsample,), 0) * (num_states // nsample)).astype(
        jnp.int32
    )
    sample = words.at[idx].get(out_sharding=P(None, None))
    cols = jax.lax.sort(
        tuple(sample[:, j] for j in range(num_words)), dimension=0, num_keys=num_words
    )
    sample = jnp.stack(cols, axis=1)
    quantiles = (
        (jax.lax.broadcasted_iota(jnp.int32, (num_shards - 1,), 0) + 1) * (nsample // num_shards)
    ).astype(jnp.int32)
    splitters = sample[quantiles]

    def local(states_p, words, splitters):
        # --- round 1: route rows to the owner of their value bucket ---
        bucket = jnp.zeros(words.shape[0], jnp.int32)
        for s in range(num_shards - 1):
            bucket = bucket + ge_lex(words, splitters[s]).astype(jnp.int32)
        send_w, over1 = bucketize(words, num_shards, bucket, cap1, SENTINEL)
        send_r, _ = bucketize(states_p, num_shards, bucket, cap1, jnp.uint8(255))
        recv_w = jax.lax.all_to_all(send_w, "x", 0, 0, tiled=True).reshape(
            num_shards * cap1, num_words
        )
        recv_r = jax.lax.all_to_all(send_r, "x", 0, 0, tiled=True).reshape(
            num_shards * cap1, num_bytes
        )

        # --- I now own one whole bucket: sort and dedupe it locally, no cross-shard comparison ---
        order = jax.lax.sort(
            (
                *[recv_w[:, j] for j in range(num_words)],
                jax.lax.broadcasted_iota(jnp.int32, (num_shards * cap1,), 0),
            ),
            dimension=0,
            num_keys=num_words,
        )[-1]
        recv_w = recv_w[order]
        recv_r = recv_r[order]
        live = jnp.any(recv_w != SENTINEL, axis=1)
        # Each bucket gets its own "element 0 is unique" -- correct only because range splitters
        # partition by VALUE, so this bucket's first row differs from the previous bucket's last.
        fresh = jnp.concatenate([jnp.ones(1, bool), jnp.any(recv_w[1:] != recv_w[:-1], axis=1)])
        keep = live & fresh
        unique_count = keep.sum()

        # --- my bucket's global output offset = unique counts of all lower-numbered buckets ---
        all_counts = jax.lax.all_gather(unique_count.reshape(1), "x", axis=0, tiled=True)
        me = jax.lax.axis_index("x")
        base = jnp.where(jnp.arange(num_shards) < me, all_counts, 0).sum()
        global_row = base + jnp.cumsum(keep.astype(jnp.int32)) - 1

        # --- round 2: route to whichever output shard owns that global row ---
        dest = jnp.clip(jnp.where(keep, global_row // shard_out, 0), 0, num_shards - 1).astype(
            jnp.int32
        )
        # The destination row cannot ride inside the uint8 payload, so it travels as its own int32
        # buffer: the offset WITHIN the destination shard.
        local_row = jnp.where(keep, global_row - dest * shard_out, -1).astype(jnp.int32)
        key2 = jnp.where(keep, dest, num_shards - 1)
        send2_r, over2 = bucketize(
            jnp.where(keep[:, None], recv_r, jnp.uint8(255)),
            num_shards,
            key2,
            cap2,
            jnp.uint8(255),
            live=keep,
        )
        send2_i, _ = bucketize(
            jnp.where(keep, local_row, -1).reshape(-1, 1),
            num_shards,
            key2,
            cap2,
            jnp.int32(-1),
            live=keep,
        )
        rows2 = jax.lax.all_to_all(send2_r, "x", 0, 0, tiled=True).reshape(
            num_shards * cap2, num_bytes
        )
        idx2 = jax.lax.all_to_all(send2_i, "x", 0, 0, tiled=True).reshape(num_shards * cap2)

        # --- place into MY slice of the output; filler stays 255 ---
        out = jnp.full((shard_out + 1, num_bytes), jnp.uint8(255), states_p.dtype)
        where = jnp.where((idx2 >= 0) & (idx2 < shard_out), idx2, shard_out)
        out = out.at[where].set(rows2, mode="drop")
        return out[:shard_out], unique_count.reshape(1), (over1 + over2).reshape(1)

    return jax.shard_map(
        local,
        mesh=mesh,
        in_specs=(P("x", None), P("x", None), P(None, None)),
        out_specs=(P("x", None), P("x"), P("x")),
    )(states_p, words, splitters)


def xxz_krylov(num_qubits, max_states):
    """A real 1D XXZ subspace: states reachable from |Neel> by nearest-neighbour hops."""
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
    return np.packbits(padded, axis=1)


def run(mesh, num_shards, states, states_size, slack=1.35):
    """Run the sharded version and compare against ``uniquify_states``."""
    reference = np.asarray(uniquify_states(jnp.asarray(states), states_size))
    words = _pack_state_words(jnp.asarray(states))
    cap1, cap2 = capacities(len(states), states_size, num_shards, slack)
    with jax.set_mesh(mesh):
        shard = jax.NamedSharding(mesh, P("x", None))
        got, counts, overflow = sharded_uniquify(
            jax.device_put(jnp.asarray(states), shard),
            jax.device_put(words, shard),
            states_size,
            num_shards,
            cap1,
            cap2,
            256,
            mesh,
        )
        # Multi-process: np.asarray on a globally-sharded array raises "Fetching value for
        # jax.Array that spans non-addressable (non process local) devices". process_allgather
        # replicates it to every rank.
        #
        # tiled=True is REQUIRED, not a tuning choice. With the default tiled=False a *fully
        # addressable* array (the single-process case) is stacked into a new leading axis --
        # (4, 2) becomes (1, 4, 2) -- so np.array_equal against the reference silently returned
        # False. Measured: exactness flipped True -> False at d=2 until this was set. tiled=True
        # concatenates instead, matching np.asarray's shape in both regimes.
        got, counts, overflow = process_allgather((got, counts, overflow), tiled=True)
    return {
        "exact": np.array_equal(reference, got),
        "unique": int(np.asarray(counts).sum()),
        "overflow": int(np.asarray(overflow).sum()),
        "caps": (cap1, cap2),
        "reference_unique": int(np.sum(~np.all(reference == 255, axis=1))),
    }


def shard_counts():
    """Shard counts to sweep: the powers of two dividing the visible device count.

    Derived rather than hardcoded so the same script is meaningful on 1, 2, 4 or 8 devices. Note
    ``d = 1`` is skipped here, unlike ``poc13``: with one bucket there is no second routing round to
    exercise and the run would assert nothing this POC exists for.
    """
    total = jax.device_count()
    return tuple(d for d in (2, 4, 8, 16, 32) if d <= total and total % d == 0)


def backend_note():
    """One line naming the backend, and whether timings from it would mean anything."""
    backend = jax.default_backend()
    kind = "REAL" if backend != "cpu" else "virtual"
    return f"{jax.device_count()} {kind} {backend} device(s)"


def main():
    init_devices(options.devices, options.host_devices)
    print("POC 14: sharded uniquify_states, two routing rounds\n")
    print(f"running on {backend_note()}; sweeping shard counts {shard_counts()}")
    if jax.default_backend() == "cpu":
        print("  virtual devices: correctness only -- per CLAUDE.md, timings here are meaningless")
    if not shard_counts():
        raise SystemExit(
            "needs at least 2 devices: with one bucket there is no second routing round, which is "
            "the mechanism this POC exists to verify. Pass --devices, or --host-devices 2."
        )
    states = xxz_krylov(NUM_QUBITS, 12_000)
    # Duplicate a third of the rows so the dedupe path is actually exercised, then shuffle: the
    # input to uniquify_states is unsorted by contract.
    states = np.concatenate([states, states[: len(states) // 3]])
    states = states[np.random.default_rng(0).permutation(len(states))]
    states_size = 1 << int(np.ceil(np.log2(len(states))))
    print(f"1D XXZ Krylov, nq={NUM_QUBITS}, N={len(states)} (with duplicates), ss={states_size}")

    print("\n1. exactness across shard counts")
    print(f"   {'d':>3} {'cap1':>8} {'cap2':>8} {'unique':>8} {'ovf':>7} {'exact':>6}")
    for num_shards in shard_counts():
        mesh = make_1d_mesh(devices=jax.devices()[:num_shards])
        trimmed = states[: (len(states) // num_shards) * num_shards]
        r = run(mesh, num_shards, trimmed, states_size)
        print(
            f"   {num_shards:>3} {r['caps'][0]:>8} {r['caps'][1]:>8} {r['unique']:>8} "
            f"{r['overflow']:>7} {r['exact']!s:>6}"
        )
        assert r["exact"], f"d={num_shards}: output differs from uniquify_states"
        assert r["overflow"] == 0, f"d={num_shards}: slack 1.35 should not overflow"
        assert r["unique"] == r["reference_unique"], "unique count disagrees"

    print("\n2. the overflow guard fires when either round is undersized")
    widest = shard_counts()[-1]
    mesh = make_1d_mesh(devices=jax.devices()[:widest])
    trimmed = states[: (len(states) // widest) * widest]
    print(f"   {'slack':>7} {'ovf':>8} {'exact':>6}")
    for slack in (1.35, 0.30):
        r = run(mesh, widest, trimmed, states_size, slack=slack)
        note = "   <- guard fires; a caller MUST raise" if r["overflow"] else ""
        print(f"   {slack:>7} {r['overflow']:>8} {r['exact']!s:>6}{note}")
    under = run(mesh, widest, trimmed, states_size, slack=0.30)
    assert under["overflow"] > 0, "an undersized capacity must be detected"
    assert not under["exact"], "an undersized capacity does drop rows -- hence the guard"
    print("\n   Exactness and a zero overflow count coincide, which is what makes the free check")
    print("   sufficient.")

    print("\n3. timing: two-round routing against the single-device sort")
    if jax.default_backend() == "cpu":
        print("   SKIPPED -- virtual devices share one physical backend, so per CLAUDE.md these")
        print("   timings are meaningless. Re-run with --devices on a multi-GPU box.")
        return
    # THE question this line is blocked on. Two rounds is 24 all_to_all at d=4, against removing a
    # `13 * N`-byte replicated state list from every device. Both arms warm: a cold run measures
    # compilation, and one such run read 125 s against a warm 20 s elsewhere in this repo.
    import time

    widest = shard_counts()[-1]
    trimmed = states[: (len(states) // widest) * widest]
    mesh = make_1d_mesh(devices=jax.devices()[:widest])
    print(f"   {'arm':>28} {'ms':>9} {'per-device states':>18}")
    for label, thunk, per_device in (
        (
            "uniquify_states (1 device)",
            lambda: uniquify_states(jnp.asarray(trimmed), states_size),
            trimmed.nbytes,
        ),
        (
            f"sharded, two rounds, d={widest}",
            lambda: run(mesh, widest, trimmed, states_size)["exact"],
            trimmed.nbytes // widest,
        ),
    ):
        thunk()
        start = time.perf_counter()
        for _ in range(3):
            jax.block_until_ready(thunk())
        elapsed = (time.perf_counter() - start) / 3 * 1e3
        print(f"   {label:>28} {elapsed:>8.2f}m {per_device / 1e6:>17.2f}M")
    print("   A ratio above 1.0 is not automatically a failure: the sharded arm is what lifts the")
    print("   N <= 2^31 ceiling, so the question is whether the cost is affordable, not whether it")
    print("   is zero. Report both columns.")


if __name__ == "__main__":
    main()
