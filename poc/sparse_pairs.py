"""Sparse transition pairs instead of dense source indices (item 8, ``markdown/improvement-ideas-2026-09-25.md``).

At ``cache_level=(1, *)`` the source cache holds one ``int32`` per ``(X group, state)`` -- ``4*J`` B/slot,
the largest term of a ``(1, 0)`` solve at large ``n`` -- yet only a fraction ``h`` are real transitions,
the rest ``-1``. ``i ^ x ^ x = i``, so every real transition of a group ``g != 0`` is a pair ``(i, j)``
with ``j = src_g(i)`` and ``i = src_g(j)``; storing ``(i, j, g)`` once costs ``~6*h*J`` B/slot. And since a
group has one X signature, ``H_ji = conj(H_ij)`` exactly (each folded term satisfies
``conj(c_k s_k(i)) = c_k s_k(j)``), so a pair needs its diagonal factor once, not twice.

Two variants, checked against the ``(1, 0)`` product and timed against ``(1, 0)`` and ``(1, 2)``:

* **P0** -- recompute each pair's diagonal every matvec (the ``(1, 0)`` analogue);
* **P2** -- cache one complex diagonal per pair (the ``(1, 2)`` analogue, sized by pairs, not states).

The identity-X group has no pairs and stays ``d_0 * v``. Fixture: spinchain's open-XXZ Hamiltonians and
Hamming-shell subspaces (``poc/eigenpair_check_scale.py``). Prototype only: single device, not sharded.

Run: uv run python poc/sparse_pairs.py [--num-qubits 60] [--pattern type2] [--log2-size 17]
"""

import argparse
import time

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from eigenpair_check_scale import hamming_shells, patterns, xxz

from rqutils.ground_locg import ground_locg
from rqutils.paulis.symplectic import PauliSumXZ
from rqutils.sqd import (
    _apply_h_kernel,
    _pack_scanned,
    _pad_states,
    _spread_seed,
    _z_parity,
    get_diagonal,
    get_xsource,
    uniquify_states,
)


def build_pairs(ham: PauliSumXZ, states_u: jax.Array, dim: int):
    """``(i, j, g)`` for every real transition of every non-identity group, each pair once (``i < j``)."""
    xs = np.asarray(jax.lax.scan(lambda _, x: (None, get_xsource(x, states_u)), None, ham.x)[1])
    rows = np.arange(xs.shape[1])
    identity = bool(np.all(np.asarray(ham.x[0]) == 0))
    pi, pj, pg = [], [], []
    for g in range(1 if identity else 0, xs.shape[0]):
        j = xs[g]
        keep = (j > rows) & (rows < dim)  # j > i >= 0: each pair once, filler rows excluded
        assert np.array_equal(xs[g][j[keep]], rows[keep]), f"group {g} is not an involution"
        pi.append(rows[keep])
        pj.append(j[keep])
        pg.append(np.full(keep.sum(), g))
    cat = lambda parts, dt: np.concatenate(parts).astype(dt)
    hit = float((xs[:, :dim] >= 0).mean())
    return cat(pi, np.int32), cat(pj, np.int32), cat(pg, np.int32), identity, xs, hit


def pair_diagonal(states_i, zpair, cpair):
    """``sum_k c_k (1 - 2*parity(state_i & z_k))`` per pair, over the (small) per-group term axis."""
    signs = 1.0 - 2.0 * jax.vmap(_z_parity, in_axes=(None, 1), out_axes=1)(states_i, zpair)
    return jnp.sum(cpair * signs, axis=-1)


def chunked(pairs, chunk):
    """Pad every per-pair array to a multiple of ``chunk`` and reshape to ``(chunks, chunk, ...)``.

    Padding pairs are ``(0, 0)`` with a zero diagonal, so they add nothing. Scanning over chunks keeps the
    kernel's temporaries ``O(chunk)`` instead of ``O(pairs)``.
    """
    total = -(-len(pairs[0]) // chunk) * chunk
    return tuple(
        jnp.pad(jnp.asarray(a), [(0, total - len(a))] + [(0, 0)] * (a.ndim - 1)).reshape(
            -1, chunk, *a.shape[1:]
        )
        for a in pairs
    )


def scatter_pairs(vec, out, pi, pj, d):
    out = out.at[..., pi].add(d * vec[..., pj])
    return out.at[..., pj].add(jnp.conj(d) * vec[..., pi])


@jax.jit
def matvec_p0(vec, pi, pj, pg, z, c, states, kmax):
    """``pi``/``pj``/``pg`` chunked; ``z``/``c`` the full (tiny) group tables, gathered per chunk."""

    def body(out, chunk):
        i, j, g = chunk
        d = pair_diagonal(states[i], z[g][:, : kmax.shape[0]], c[g][:, : kmax.shape[0]])
        d = jnp.where(i == j, 0.0, d)  # padding pairs are (0, 0)
        return scatter_pairs(vec, out, i, j, d), None

    out = get_diagonal(z[0], c[0], states) * vec
    return jax.lax.scan(body, out, (pi, pj, pg))[0]


@jax.jit
def matvec_p2(vec, pi, pj, dpair, d0):
    """``pi``/``pj``/``dpair`` chunked; padding pairs carry a zero diagonal."""

    def body(out, chunk):
        return scatter_pairs(vec, out, *chunk), None

    return jax.lax.scan(body, d0 * vec, (pi, pj, dpair))[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-qubits", type=int, default=60)
    parser.add_argument("--pattern", default="type2")
    parser.add_argument("--delta", type=float, default=0.5)
    parser.add_argument("--log2-size", type=int, default=17)
    parser.add_argument("--rounds", type=int, default=15)
    parser.add_argument("--solve", action="store_true", help="also time whole ground_locg solves")
    parser.add_argument(
        "--chunk-log2", type=int, default=0, help="pairs per scan chunk; 0 = states_size"
    )
    args = parser.parse_args()

    n, size = args.num_qubits, 1 << args.log2_size
    ham = PauliSumXZ.from_paulisum(xxz(n, args.delta, *patterns(n)[args.pattern]))
    states = hamming_shells(n, size, np.random.default_rng(0))
    su = uniquify_states(_pad_states(PauliSumXZ.pack_states(states), size), size)
    pi, pj, pg, identity, xs, hit = build_pairs(ham, su, len(states))
    assert identity, "the XXZ fixtures always have an identity-X group"
    num_groups = ham.x.shape[0]
    kmax = int(
        (np.abs(np.asarray(ham.c[1:])) > 0).sum(axis=1).max()
    )  # terms in a non-identity group
    z, c = jnp.asarray(ham.z), jnp.asarray(ham.c)
    d0 = get_diagonal(ham.z[0], ham.c[0], su)
    dpair = np.asarray(pair_diagonal(su[pi], z[pg, :kmax], c[pg, :kmax]))
    chunk = 1 << (args.chunk_log2 or args.log2_size)
    cpi, cpj, cpg, cdp = chunked((pi, pj, pg, dpair), chunk)
    kdummy = jnp.zeros(kmax)  # its static shape carries kmax into the jitted kernel
    print(
        f"n={n} {args.pattern} delta={args.delta}: J={num_groups}, states_size={size}, hit rate {hit:.3f}, "
        f"{len(pi)} pairs ({len(pi) / size:.2f} per slot), non-identity K_max={kmax}, chunk {chunk}"
    )

    dg = jax.lax.scan(lambda _, v: (None, get_diagonal(v[0], v[1], su)), None, (ham.z, ham.c))[1]
    kernel = jax.jit(
        lambda v, sc, st, lvl: _apply_h_kernel(v, sc, st, cache_level=lvl), static_argnums=3
    )
    arms = {
        "(1,0)": (kernel, (_pack_scanned((1, 0), jnp.asarray(xs), ham.z, ham.c), su, (1, 0))),
        "(1,2)": (kernel, (_pack_scanned((1, 2), jnp.asarray(xs), dg, ham.c), None, (1, 2))),
        "P0": (matvec_p0, (cpi, cpj, cpg, z, c, su, kdummy)),
        "P2": (matvec_p2, (cpi, cpj, cdp, d0)),
    }
    # Bytes of the per-slot operator arrays each arm keeps (the vectors are common to all).
    stored = {
        "(1,0)": xs.nbytes + su.nbytes,
        "(1,2)": xs.nbytes + dg.nbytes,
        "P0": cpi.nbytes + cpj.nbytes + cpg.nbytes + su.nbytes + z.nbytes + c.nbytes,
        "P2": cpi.nbytes + cpj.nbytes + cdp.nbytes + d0.nbytes,
    }
    vec = jnp.asarray(np.random.default_rng(1).normal(size=(2, size)) * (1 + 0.5j))
    vec = vec.at[:, len(states) :].set(0.0)  # filler slots carry no amplitude, as in a solve
    ref = np.asarray(arms["(1,0)"][0](vec, *arms["(1,0)"][1]))
    for name, (fn, a) in arms.items():
        got = np.asarray(fn(vec, *a))[:, : len(states)]
        assert np.allclose(got, ref[:, : len(states)], rtol=1e-12, atol=1e-12), name
    print("every arm matches the (1,0) product on a batched (2, N) vec")

    times = {k: [] for k in arms}
    for _ in range(args.rounds):
        for name, (fn, a) in arms.items():
            t0 = time.perf_counter()
            jax.block_until_ready(fn(vec, *a))
            times[name].append(time.perf_counter() - t0)
    base = np.median(times["(1,0)"])
    print(f"\n{'arm':6} {'operator B/slot':>16} {'(2,N) matvec':>13} {'vs (1,0)':>9}")
    for name in arms:
        m = np.median(times[name])
        print(f"{name:6} {stored[name] / size:>16.1f} {1e3 * m:>10.2f} ms {base / m:>8.2f}x")

    if args.solve:
        vinit = _spread_seed(size, su, ham.c.dtype, None)
        bound = float(np.abs(np.asarray(ham.c)).sum())
        print(
            f"\n{'arm':6} {'solve':>9} {'iters':>5} {'eigval':>20} {'XLA inputs+temp B/slot':>23}"
        )
        for name, (fn, a) in arms.items():

            def solve(v, *a, fn=fn):
                return ground_locg(
                    fn, v, args=a, prefilter=(32, 2), prefilter_hi=bound, batch_matvec=True
                )

            # The (1,x) kernel's static level cannot ride in ground_locg's traced args.
            if fn is kernel:
                level = a[2]
                a = a[:2]

                def solve(v, *a, level=level):
                    return ground_locg(
                        lambda x, sc, st: _apply_h_kernel(x, sc, st, cache_level=level),
                        v,
                        args=a,
                        prefilter=(32, 2),
                        prefilter_hi=bound,
                        batch_matvec=True,
                    )

            jitted = jax.jit(solve)
            mem = jitted.lower(vinit, *a).compile().memory_analysis()
            jax.block_until_ready(jitted(vinit, *a))
            ts = []
            for _ in range(3):
                t0 = time.perf_counter()
                r = jax.block_until_ready(jitted(vinit, *a))
                ts.append(time.perf_counter() - t0)
            total = (mem.argument_size_in_bytes + mem.temp_size_in_bytes) / size
            print(
                f"{name:6} {np.median(ts):8.2f}s {int(r[2]):>5} {float(r[0]):>20.12f} {total:>23.1f}"
            )


if __name__ == "__main__":
    main()
