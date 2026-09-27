"""XOR-fold the Z parity before one popcount, instead of a popcount per byte and a sum -- spike.

``_z_parity`` computes ``popcount(s & z) mod 2`` as ``sum(bitwise_count(s & z), axis=1) & 1`` over the
``B`` packed bytes. Parity is linear over XOR, so ``bitwise_count(xor_reduce(s & z)) & 1`` gives the same
bit with one popcount per state; with ``B % 8 == 0`` the bytes can be bitcast to ``uint64`` words first.
``matvec=Matvec.INDICES`` rebuilds every diagonal per matvec, ~64% of its solve against ``TABLES``
(``poc/sparse-pairs.md`` §9), so that is where a faster parity would show.

Arms swap ``rqutils.sqd._diagonal._z_parity`` (looked up at trace time) and clear jax's caches, so each
retraces; rounds alternate the arms and report min/median plus a paired win count.

``onepass`` asks whether reading the states once per group beats ``get_diagonal``'s one pass per Z term:
every group's diagonal (one ``INDICES`` matvec's diagonal work) three ways -- the library's per-group loop,
and with the identity group split off (so the rest need not be padded to its ``K``) as a chunked broadcast
popcount or an ``int8`` dot of unpacked bits (bits assumed cached, ``n`` B/slot).

Run: uv run python poc/parity_xor.py fold [--num-qubits 60] [--patterns type1 type2] [--log2-size 17]
     uv run python poc/parity_xor.py onepass [...]
"""

import argparse
import time

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from sparse_pairs import spinchain_problem

import rqutils.sqd._diagonal as diagonal_mod
from rqutils.sqd import Matvec, get_diagonal, sqd, uniquify_states
from rqutils.sqd._states import _pad_states

BASELINE = diagonal_mod._z_parity


def parity_xor_bytes(states, zsignature):
    """XOR the masked bytes together, then one popcount."""
    folded = jax.lax.reduce(states & zsignature, np.uint8(0), jax.lax.bitwise_xor, (1,))
    return jnp.bitwise_count(folded) & 1


def parity_xor_words(states, zsignature):
    """As ``parity_xor_bytes`` over ``uint64`` words; falls back to bytes unless ``B % 8 == 0``."""
    num_bytes = states.shape[1]
    if num_bytes % 8:
        return parity_xor_bytes(states, zsignature)
    words = jax.lax.bitcast_convert_type(
        (states & zsignature).reshape(states.shape[0], num_bytes // 8, 8), np.uint64
    )
    folded = jax.lax.reduce(words, np.uint64(0), jax.lax.bitwise_xor, (1,))
    return (jnp.bitwise_count(folded) & 1).astype(np.uint8)


ARMS = {"base": BASELINE, "xor8": parity_xor_bytes, "xor64": parity_xor_words}


def use(arm):
    # A deliberate monkeypatch: the arms are untyped stand-ins with the same signature.
    diagonal_mod._z_parity = ARMS[arm]  # ty: ignore[invalid-assignment]
    jax.clear_caches()


def check_identical(ham, states_u):
    """Every group's diagonal, bit-identical across arms."""
    ref = None
    for arm in ARMS:
        use(arm)
        got = [np.asarray(get_diagonal(z, c, states_u)) for z, c in zip(ham.z, ham.c, strict=True)]
        if ref is None:
            ref = got
        else:
            assert all(np.array_equal(a, b) for a, b in zip(ref, got, strict=True)), arm
    print(f"  diagonals bit-identical across {list(ARMS)} ({len(ref)} groups)")


def time_diagonal(ham, states_u, arm, trials=5):
    """The identity group's diagonal alone (largest K), warm, arrays passed as arguments."""
    use(arm)
    z, c = ham.z[0], ham.c[0]
    get_diagonal(z, c, states_u).block_until_ready()
    best = np.inf
    for _ in range(trials):
        start = time.perf_counter()
        get_diagonal(z, c, states_u).block_until_ready()
        best = min(best, time.perf_counter() - start)
    return best


def time_sqd(ham, states, arm, matvec):
    use(arm)
    sqd(ham, states, return_eigvec=False, matvec=matvec)  # compile
    start = time.perf_counter()
    eigval = sqd(ham, states, return_eigvec=False, matvec=matvec)
    return time.perf_counter() - start, eigval


CHUNK = 1 << 13


def chunked(fn, states, *rest):
    """``fn(chunk, *rest)`` over fixed state chunks, flattened back to ``(N,)``."""
    parts = states.reshape(-1, CHUNK, *states.shape[1:])
    return jax.lax.map(lambda part: fn(part, *rest), parts).reshape(-1)


def onepass_bcast(part, z, c):
    """All K parities of one chunk at once: ``(chunk, K, B)`` AND, popcount, sum over bytes."""
    counts = jnp.sum(jnp.bitwise_count(part[:, None, :] & z[None]), axis=2, dtype=np.uint8)
    return (1.0 - 2.0 * (counts & 1)) @ c


def onepass_dot(part, zbits, c):
    """The same parities as one integer product of unpacked bits: exact, since each count <= n."""
    counts = jnp.dot(part, zbits, preferred_element_type=np.int32)
    return (1.0 - 2.0 * (counts & 1)) @ c


@jax.jit
def diag_library(z, c, states):
    return jax.lax.map(lambda zc: get_diagonal(zc[0], zc[1], states), (z, c))


@jax.jit
def diag_bcast(z0, c0, zr, cr, states):
    d0 = chunked(onepass_bcast, states, z0, c0)
    rest = jax.lax.map(lambda zc: chunked(onepass_bcast, states, *zc), (zr, cr))
    return jnp.concatenate([d0[None], rest])


@jax.jit
def diag_dot(z0bits, c0, zrbits, cr, bits):
    d0 = chunked(onepass_dot, bits, z0bits, c0)
    rest = jax.lax.map(lambda zc: chunked(onepass_dot, bits, *zc), (zrbits, cr))
    return jnp.concatenate([d0[None], rest])


def unpack_bits(packed):
    """``(..., B)`` uint8 to ``(..., 8B)`` int8 bits, one order for states and signatures alike."""
    return np.unpackbits(np.asarray(packed), axis=-1).astype(np.int8)


def onepass(args):
    for pattern in args.patterns:
        ham, states = spinchain_problem(args.num_qubits, pattern, args.delta, 1 << args.log2_size)
        states_size = 1 << args.log2_size
        states_u = uniquify_states(_pad_states(ham.pack_states(states), states_size), states_size)
        k = np.array([int(np.count_nonzero(c)) for c in ham.c])
        assert np.all(k[1:] <= k[1:].max()) and ham.x[0].any() == 0, (
            "group 0 must be the identity X"
        )
        kr = int(k[1:].max())
        z, c = jnp.asarray(ham.z), jnp.asarray(ham.c)
        z0, c0 = z[0], c[0]
        zr, cr = z[1:, :kr], c[1:, :kr]
        bits = jnp.asarray(unpack_bits(states_u))
        z0bits = jnp.asarray(unpack_bits(ham.z[0]).T)
        zrbits = jnp.asarray(np.swapaxes(unpack_bits(ham.z[1:, :kr]), 1, 2))
        print(
            f"n={args.num_qubits} {pattern} 2^{args.log2_size}: J={len(k)}, sum K={k.sum()}, "
            f"identity K={k[0]}, other groups K<={kr}; per-group-loop passes {k.sum()}, "
            f"padded rectangle {len(k) * k.max()}"
        )
        arms = {
            "library": (diag_library, (z, c, states_u)),
            "bcast": (diag_bcast, (z0, c0, zr, cr, states_u)),
            "dot": (diag_dot, (z0bits, c0, zrbits, cr, bits)),
        }
        ref = np.asarray(diag_library(z, c, states_u))
        for name, (fn, fn_args) in arms.items():
            got = np.asarray(fn(*fn_args))
            err = np.max(np.abs(got - ref))
            assert err < 1e-12 * np.abs(ref).max(), (name, err)
            print(f"  {name:>8}: max |diff| vs library {err:.1e}")
        times = {name: [] for name in arms}
        for _ in range(args.rounds * 3):
            for name, (fn, fn_args) in arms.items():
                start = time.perf_counter()
                jax.block_until_ready(fn(*fn_args))
                times[name].append(time.perf_counter() - start)
        base = np.median(times["library"])
        print(
            f"  all {len(k)} diagonals (one INDICES matvec's diagonal work), {args.rounds * 3} rounds:"
        )
        for name, ts in times.items():
            wins = sum(t < b for t, b in zip(ts, times["library"], strict=True))
            print(
                f"    {name:>8}: min {min(ts) * 1e3:7.2f} ms  median {np.median(ts) * 1e3:7.2f} ms  "
                f"{base / np.median(ts):5.2f}x  wins {wins}/{len(ts)}"
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("fold", "onepass"))
    parser.add_argument("--num-qubits", type=int, default=60)
    parser.add_argument("--patterns", nargs="+", default=["type1", "type2"])
    parser.add_argument("--delta", type=float, default=0.5)
    parser.add_argument("--log2-size", type=int, default=17)
    parser.add_argument("--rounds", type=int, default=5)
    args = parser.parse_args()
    if args.mode == "onepass":
        return onepass(args)

    for pattern in args.patterns:
        ham, states = spinchain_problem(args.num_qubits, pattern, args.delta, 1 << args.log2_size)
        states_size = 1 << args.log2_size
        states_u = uniquify_states(_pad_states(ham.pack_states(states), states_size), states_size)
        k = [int(np.count_nonzero(c)) for c in ham.c]
        print(
            f"n={args.num_qubits} {pattern} 2^{args.log2_size}: J={len(k)}, K per group "
            f"{min(k)}..{max(k)} (identity {k[0]}), B={states_u.shape[1]}"
        )
        check_identical(ham, states_u)

        print("  identity-group get_diagonal, warm min of 5:")
        base_d = time_diagonal(ham, states_u, "base")
        for arm in ARMS:
            t = base_d if arm == "base" else time_diagonal(ham, states_u, arm)
            print(f"    {arm:>6}: {t * 1e3:8.2f} ms  ({base_d / t:.2f}x)")

        times = {arm: [] for arm in ("base", "xor64")}
        eigvals = {}
        for _ in range(args.rounds):
            for arm, ts in times.items():
                t, eigvals[arm] = time_sqd(ham, states, arm, Matvec.INDICES)
                ts.append(t)
        use("base")
        tables, _ = time_sqd(ham, states, "base", Matvec.TABLES)
        wins = sum(x < b for b, x in zip(times["base"], times["xor64"], strict=True))
        assert eigvals["base"] == eigvals["xor64"], eigvals
        print(
            f"  warm sqd(matvec=INDICES), {args.rounds} interleaved rounds (eigval {eigvals['base']!r}):"
        )
        for arm, ts in times.items():
            print(f"    {arm:>6}: min {min(ts):6.2f} s  median {np.median(ts):6.2f} s")
        print(
            f"    xor64 vs base: {np.median(times['base']) / np.median(times['xor64']):.2f}x median, "
            f"wins {wins}/{args.rounds}; TABLES (ceiling) {tables:.2f} s"
        )
    use("base")


if __name__ == "__main__":
    main()
