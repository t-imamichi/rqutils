"""POC 12: hash-partitioning ``states`` so ``get_xsource`` needs only ``N/d`` rows per device.

``states`` is the one term the ``(0, 0)`` memory floor cannot shed -- ``13 * N`` bytes on **every**
device, 27.9 GB at ``N = 2**31`` -- because ``get_xsource`` binary-searches it and ``sqd.py`` requires
it replicated. Per-ingredient testing shows only ``searchsorted`` actually fails on a partitioned
``[N, B]``: ``bitwise_xor`` and ``_pack_state_keys`` shard fine. A binary search needs the whole sorted
haystack visible to every query -- unavoidable only given a *range-agnostic* partition.

Wietek & Lauchli (*Phys. Rev. E* **98**, 033309, and thesis SS3.3) solve this for exact diagonalization:
states sharing a **prefix** live on one rank, ownership is a **hash of the prefix bits** (so no
distribution metadata is stored at all), each rank keeps its states sorted, and the matvec buffers
targets locally then does one ``MPI_Alltoallv``.

**This POC's finding is that the prefix must be replaced by the whole key.** An SQD subspace whose
excitations are confined to low qubits -- what a circuit acting on a subset of qubits produces -- has
*one* distinct prefix, so the prefix hash is constant and one shard takes everything: measured
**16.00x imbalance at d=16, i.e. total collapse**. That is the same low-entropy-high-bits failure
``poc11_range_partition.py`` recorded for equal-range splitting ("at n=100 that word is 7 bytes of
leading pad"). Hashing the whole key measures **1.01-1.14x** on every fixture here and costs one extra
``mix64`` on a key ``_pack_state_keys`` already computed. The published scheme hashes the prefix because
it needs prefix-grouping for local *enumeration*; rqutils has no such requirement.

What whole-key hashing gives up is the free local sort: a range split hands each shard a contiguous
sorted block, a hash hands it a scattered subset. That is ``poc11``'s phase 3 (``d`` independent
``lax.sort``s under ``vmap``, zero collectives) and runs **once at setup**, not per call.

**Not applicable: DanceQ** (arXiv:2407.14591), which reaches 46 spins over ~256 nodes with
*synchronization-free* thread-local lookup. Its basis is a **complete** U(1) sector, so state-to-index
is a closed-form combinatorial map (enumerative encoding, Cover 1973). A sampled subspace has no such
formula -- which is why ``get_xsource`` searches at all. Do not cite it as precedent.

This script validates the **algorithm** in numpy: ownership, per-shard sort, routed search, and the
mapping of a per-shard slot back to the global lex index. It is deliberately not a JAX implementation --
the three JAX ingredients (``shard_map``, ``all_to_all``, local ``searchsorted``) are verified separately
in ``NOTES.md`` but not yet composed with hashing. **No speed claim is made**: per ``CLAUDE.md``, timings
under virtual devices are meaningless, and nothing here ran on a real interconnect.

Run::

    uv run python examples/scaling/poc12_hash_partition.py
"""

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from rqutils.sqd import _pack_state_keys, get_xsource

NQ = 60
RNG = np.random.default_rng(11)


def mix64(x):
    """splitmix64-style finalizer. Any good avalanche works; this one is table-free."""
    x = np.asarray(x, dtype=np.uint64)
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


def pack(rows):
    """Pack unpacked bit rows with the mandatory pad bit at position 0, as ``sqd`` does."""
    padded = np.concatenate([np.zeros((len(rows), 1), np.uint8), rows], axis=1)
    return np.unique(np.packbits(padded, axis=1), axis=0)


def fixture(kind, num_draws, rng):
    """Subspaces spanning the range from easy to adversarial for prefix hashing.

    ``banded`` is the one that matters: excitations confined to the last ``n/4`` qubits leave every
    state sharing its high bits, which is what destroys a prefix hash.
    """
    rows = np.zeros((num_draws, NQ), np.uint8)
    if kind == "uniform":
        rows = rng.integers(0, 2, (num_draws, NQ), dtype=np.uint8)
    elif kind == "fixed-weight":
        for i in range(num_draws):
            rows[i, rng.choice(NQ, NQ // 2, replace=False)] = 1
    elif kind == "banded":
        band = NQ // 4
        for i in range(num_draws):
            rows[i, NQ - band + rng.choice(band, band // 2, replace=False)] = 1
    else:
        raise ValueError(kind)
    # Hop partners, so the hit rate is realistic. The hop must act on qubits the fixture actually
    # populates: a hop on qubits 0-1 gives `banded` a 0% hit rate, which tests nothing.
    lo = NQ - NQ // 4 if kind == "banded" else 0
    partners = rows[: num_draws // 2].copy()
    partners[:, [lo, lo + 1]] = partners[:, [lo + 1, lo]]
    return pack(np.concatenate([rows, partners]))


def xsignature(kind):
    """The X signature matching :func:`fixture`'s hop."""
    eye = np.eye(NQ, dtype=np.uint8)
    lo = NQ - NQ // 4 if kind == "banded" else 0
    return np.packbits(np.concatenate([np.zeros(1, np.uint8), eye[lo] + eye[lo + 1]]))


def hash_partitioned_xsource(states, xsig, num_shards):
    """``get_xsource`` with the state list split across ``num_shards`` devices.

    Each shard holds ``N / num_shards`` rows and nothing else. Returns the same ``[N]`` int32 array
    of global lex indices that :func:`get_xsource` returns, with -1 where the source is absent.
    """
    keys = np.asarray(_pack_state_keys(jnp.asarray(states)))
    targets = np.asarray(_pack_state_keys(jnp.bitwise_xor(jnp.asarray(states), xsig)))
    mod = np.uint64(num_shards)

    # Setup, once per subspace: assign owners, then sort within each shard. The local slice must be
    # sorted because the lookup is still a binary search; a hash scatters, so unlike a range split
    # this sort is not free. `shard_gidx` maps a local slot back to the global lex index, which is
    # what `get_xsource` returns and what the eigenvector is indexed by.
    owner_state = (mix64(keys) % mod).astype(np.int64)
    shard_keys, shard_gidx = [], []
    for shard in range(num_shards):
        mine = np.nonzero(owner_state == shard)[0]
        order = np.argsort(keys[mine], kind="stable")
        shard_keys.append(keys[mine][order])
        shard_gidx.append(mine[order].astype(np.int32))

    # Query: route each target to its owner -- the same hash on the same key, so a state and any
    # target equal to it land on the same shard, which is the whole correctness argument.
    out = np.full(len(states), -1, np.int32)
    owner_target = (mix64(targets) % mod).astype(np.int64)
    for shard in range(num_shards):
        mine = np.nonzero(owner_target == shard)[0]
        if len(mine) == 0:
            continue
        local = shard_keys[shard]
        pos = np.minimum(np.searchsorted(local, targets[mine]), len(local) - 1)
        found = local[pos] == targets[mine]
        out[mine] = np.where(found, shard_gidx[shard][pos], -1)
    return out, np.bincount(owner_target, minlength=num_shards)


def main():
    print(f"POC 12: hash-partitioned get_xsource, nq={NQ}\n")

    print("1. prefix hashing collapses on a structured subspace (d=16, 12 prefix bits)")
    print(f"   {'fixture':>14} {'prefix-hash':>12} {'whole-key':>10} {'distinct prefixes':>18}")
    for kind in ("uniform", "fixed-weight", "banded"):
        rows = np.zeros((200_000, NQ), np.uint8)
        if kind == "uniform":
            rows = RNG.integers(0, 2, (200_000, NQ), dtype=np.uint8)
        elif kind == "fixed-weight":
            for i in range(len(rows)):
                rows[i, RNG.choice(NQ, NQ // 2, replace=False)] = 1
        else:
            band = NQ // 4
            for i in range(len(rows)):
                rows[i, NQ - band + RNG.choice(band, band // 2, replace=False)] = 1
        keys = np.asarray(_pack_state_keys(jnp.asarray(pack(rows))))
        prefix = keys >> np.uint64(64 - 12)
        pre = np.bincount((mix64(prefix) % np.uint64(16)).astype(np.int64), minlength=16)
        whole = np.bincount((mix64(keys) % np.uint64(16)).astype(np.int64), minlength=16)
        mean = len(keys) / 16
        print(
            f"   {kind:>14} {pre.max() / mean:>11.2f}x {whole.max() / mean:>9.2f}x "
            f"{len(np.unique(prefix)):>18}"
        )

    print("\n2. whole-key hashing is exact against get_xsource")
    print(f"   {'fixture':>14} {'N':>8} {'hit rate':>9} {'imbalance':>10} {'bit-identical':>14}")
    for kind in ("uniform", "fixed-weight", "banded"):
        states = fixture(kind, 60_000, RNG)
        xsig = xsignature(kind)
        reference = np.asarray(jax.jit(get_xsource)(xsig, jnp.asarray(states)))
        got, counts = hash_partitioned_xsource(states, xsig, 16)
        exact = np.array_equal(reference, got)
        print(
            f"   {kind:>14} {len(states):>8} {(reference >= 0).mean() * 100:>8.1f}% "
            f"{counts.max() / (len(states) / 16):>9.2f}x {exact!s:>14}"
        )
        assert exact, f"{kind}: hash-partitioned result differs from get_xsource"

    print("\n3. imbalance is Poisson in N/d, not structural")
    print("   compared against the balls-in-bins bound 1 + sqrt(2 ln d / (N/d))")
    print(f"   {'fixture':>14} {'N':>8}" + "".join(f"{'d=' + str(d):>9}" for d in (4, 64, 1024)))
    for kind in ("fixed-weight", "banded"):
        states = fixture(kind, 200_000, RNG)
        keys = np.asarray(_pack_state_keys(jnp.asarray(states)))
        line = f"   {kind:>14} {len(keys):>8}"
        for d in (4, 64, 1024):
            counts = np.bincount((mix64(keys) % np.uint64(d)).astype(np.int64), minlength=d)
            line += f"{counts.max() / (len(keys) / d):>8.2f}x"
        print(line)
    print("\n   Keep N/d above ~1000 for a slack under 1.15x; at N=2^31 that allows d up to ~2e6.")
    print("   Sizing shard capacity from that bound is the same role poc11's `slack` plays.")


if __name__ == "__main__":
    main()
