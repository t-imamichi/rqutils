"""Partial diagonal cache: which groups to cache at equal bytes -- a prefix, or the largest ``K_g`` first.

Item 3 of ``markdown/improvement-ideas-2026-09-25.md``. A partial diagonal cache runs ``(1, 2)`` over the
cached groups and ``(1, 0)`` over the rest, summed (the 2026-08-30 prototype in ``NOTES.md``). Every
cached group costs the same bytes (one diagonal per state), but recomputing group ``g`` costs ``K_g``
iterations -- ``_accumulate_diagonal`` stops at the first zero-padded term -- so at equal bytes, caching
the largest ``K_g`` first removes the most work. This measures whether that shows up in time.

The fixture is a molecular-like Jordan-Wigner Hamiltonian from random integrals: the structure of a
chemistry Hamiltonian (two-body X patterns with JW Z decorations, so ``K_g`` varies and the large groups
are spread through the sort order), with random numbers. A spin chain would not test this: all its
diagonal terms fall in the identity-X group, which sorts first, so a prefix already picks it.

Run: uv run python poc/diag_cache_order.py [--num-qubits 18] [--num-states 16384] [--density 0.25]
"""

import argparse
import time

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from qiskit.quantum_info import SparsePauliOp

from rqutils.ground_locg import ground_locg
from rqutils.paulis.symplectic import PauliSumXZ
from rqutils.sqd import (
    _apply_h_kernel,
    _pack_scanned,
    _pad_states,
    _spread_seed,
    get_diagonal,
    get_xsource,
    uniquify_states,
)

FRACTIONS = (0.125, 0.25, 0.5)


def annihilator(p: int, n: int) -> SparsePauliOp:
    """Jordan-Wigner ``a_p`` on ``n`` qubits (qiskit little-endian labels)."""

    def label(op: str) -> str:
        s = ["I"] * n
        for q in range(p):
            s[n - 1 - q] = "Z"
        s[n - 1 - p] = op
        return "".join(s)

    return SparsePauliOp([label("X"), label("Y")], [0.5, 0.5j])


def molecular_like(n: int, rng: np.random.Generator, density: float) -> SparsePauliOp:
    """``sum h a+_p a_q + sum g a+_p a+_q a_r a_s``, random integrals, a ``density`` of two-body terms."""
    a = [annihilator(p, n) for p in range(n)]
    ad = [op.adjoint() for op in a]
    h = rng.normal(size=(n, n))
    h = (h + h.T) / 2
    terms = [(ad[p] @ a[q]) * h[p, q] for p in range(n) for q in range(n)]
    for p in range(n):
        for q in range(p + 1, n):
            for r in range(n):
                for s in range(r + 1, n):
                    if rng.random() < density:
                        terms.append((ad[p] @ ad[q] @ a[r] @ a[s]) * (0.1 * rng.normal()))
    total = SparsePauliOp.sum(terms)
    return ((total + total.adjoint()) / 2).simplify(atol=1e-12)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-qubits", type=int, default=18)
    parser.add_argument("--num-states", type=int, default=16384)
    parser.add_argument("--density", type=float, default=0.25)
    parser.add_argument("--rounds", type=int, default=15)
    args = parser.parse_args()

    rng = np.random.default_rng(0)
    t0 = time.perf_counter()
    ham = PauliSumXZ.from_paulisum(molecular_like(args.num_qubits, rng, args.density))
    num_groups = ham.x.shape[0]
    k = (np.abs(np.asarray(ham.c)) > 0).sum(axis=1)
    print(
        f"n={args.num_qubits}: J={num_groups} groups, K_max={ham.z.shape[1]}, sum K={int(k.sum())}, "
        f"K_g quartiles {np.percentile(k, [0, 25, 50, 75, 100]).astype(int).tolist()} "
        f"({time.perf_counter() - t0:.0f} s to build)"
    )

    bits = rng.integers(0, 2, size=(args.num_states, args.num_qubits), dtype=np.uint8)
    size = 1 << (args.num_states - 1).bit_length()
    states_u = uniquify_states(_pad_states(PauliSumXZ.pack_states(bits), size), size)
    xs = jax.lax.scan(lambda _, x: (None, get_xsource(x, states_u)), None, ham.x)[1]
    dg = jax.lax.scan(
        lambda _, v: (None, get_diagonal(v[0], v[1], states_u)), None, (ham.z, ham.c)
    )[1]
    vec = jnp.asarray(rng.normal(size=size))
    print(f"states_size={size}, full diagonal store {dg.nbytes / 2**20:.1f} MiB")

    two_arm = jax.jit(
        lambda v, cached, rest, st: (
            _apply_h_kernel(v, cached, None, cache_level=(1, 2))
            + _apply_h_kernel(v, rest, st, cache_level=(1, 0))
        )
    )

    def arm(order: np.ndarray, num_cached: int) -> tuple:
        sel, rest = np.sort(order[:num_cached]), np.sort(order[num_cached:])
        cached = _pack_scanned((1, 2), xs[sel], dg[sel], ham.c[sel])
        uncached = _pack_scanned((1, 0), xs[rest], ham.z[rest], ham.c[rest])
        return cached, uncached, int(k[rest].sum())

    prefix = np.arange(num_groups)
    largest = np.argsort(-k, kind="stable")
    reference = np.asarray(
        _apply_h_kernel(vec, _pack_scanned((1, 2), xs, dg, ham.c), None, cache_level=(1, 2))
    )

    arms = {}
    for frac in FRACTIONS:
        num_cached = round(frac * num_groups)
        for name, order in (("prefix", prefix), ("largest", largest)):
            cached, uncached, left = arm(order, num_cached)
            out = np.asarray(two_arm(vec, cached, uncached, states_u))
            assert np.allclose(out, reference, rtol=1e-11, atol=1e-11), (frac, name)
            arms[(frac, name)] = (cached, uncached, left)

    times = {key: [] for key in arms}
    for _ in range(args.rounds):  # interleaved, so drift hits both orders of a pair alike
        for key, (cached, uncached, _left) in arms.items():
            t0 = time.perf_counter()
            jax.block_until_ready(two_arm(vec, cached, uncached, states_u))
            times[key].append(time.perf_counter() - t0)

    print(f"\n{'cached':>12} {'order':>8} {'K left':>7} {'matvec ms':>10}  paired: largest faster")
    for frac in FRACTIONS:
        for name in ("prefix", "largest"):
            left = arms[(frac, name)][2]
            print(
                f"{frac:>12.3f} {name:>8} {left:>7} {1e3 * np.median(times[(frac, name)]):>10.2f}",
                end="",
            )
            if name == "largest":
                wins = sum(b < a for a, b in zip(times[(frac, "prefix")], times[(frac, "largest")]))
                ratio = np.median(times[(frac, "prefix")]) / np.median(times[(frac, "largest")])
                print(f"  {wins}/{args.rounds}, {ratio:.3f}x")
            else:
                print()

    # Whole solves at the smallest cache, where the orders differ most, as run_sqd drives ground_locg.
    frac = FRACTIONS[0]
    bound = float(np.abs(np.asarray(ham.c)).sum())
    vinit = _spread_seed(size, states_u, ham.c.dtype, None)
    print(f"\nwhole solves at cached={frac}:")
    for name in ("prefix", "largest"):
        cached, uncached, _ = arms[(frac, name)]

        def solve(cached=cached, uncached=uncached):
            return ground_locg(
                two_arm,
                vinit,
                args=(cached, uncached, states_u),
                prefilter=(32, 2),
                prefilter_hi=bound,
                batch_matvec=True,
            )

        jax.block_until_ready(solve())
        ts = []
        for _ in range(3):
            t0 = time.perf_counter()
            result = jax.block_until_ready(solve())
            ts.append(time.perf_counter() - t0)
        print(
            f"  {name:>8}: {1e3 * np.median(ts):8.1f} ms, eigval {float(result[0]):.12f}, "
            f"iterations {int(result[2])}, converged {bool(result[3])}"
        )


if __name__ == "__main__":
    main()
