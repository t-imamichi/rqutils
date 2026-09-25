"""Sparse transition pairs beyond open XXZ: a non-XXZ and a high-hit-rate fixture.

Companion to ``poc/sparse_pairs.py``, whose win was measured on spinchain's open-XXZ Hamiltonians with
Hamming-shell subspaces (hit rate 8--25%). The pair form is general -- XOR is an involution for every X
signature and ``H_ji = conj(H_ij)`` for any Hermitian Pauli sum -- but its gain tracks the hit rate. So this
runs it where the rate is high: a molecular-like JW Hamiltonian on random states, and periodic XXZ on a
Neel-Krylov subspace closed under hops. Operator bytes and matvec speed only, against ``(1, 0)``.

Run: uv run python poc/sparse_pairs_general.py
"""

import jax

jax.config.update("jax_enable_x64", True)
import sys
import time

sys.argv = sys.argv[:1]  # sqd_multinode parses argv at import
import jax.numpy as jnp
import numpy as np
from diag_cache_order import molecular_like
from sparse_pairs import build_pairs, chunked, matvec_p0, matvec_p2, pair_diagonal
from sqd_multinode import xxz_hamiltonian, xxz_krylov_states

from rqutils.paulis.symplectic import PauliSumXZ
from rqutils.sqd import _apply_h_kernel, _pack_scanned, _pad_states, get_diagonal, uniquify_states


def run(label, ham, states):
    """Operator bytes per slot and (2, N) matvec speed of P0 and P2 against (1, 0), each checked first."""
    size = 1 << (len(states) - 1).bit_length()
    su = uniquify_states(_pad_states(PauliSumXZ.pack_states(states), size), size)
    pi, pj, pg, identity, xs, hit = build_pairs(ham, su, len(states))
    kmax = int((np.abs(np.asarray(ham.c[1:])) > 0).sum(axis=1).max())
    z, c = jnp.asarray(ham.z), jnp.asarray(ham.c)
    d0 = get_diagonal(ham.z[0], ham.c[0], su) if identity else jnp.zeros(size, ham.c.dtype)
    dpair = np.asarray(pair_diagonal(su[pi], z[pg, :kmax], c[pg, :kmax]))
    chunk = min(1 << 15, size)
    cpi, cpj, cpg, cdp = chunked((pi, pj, pg, dpair), chunk)
    k10 = jax.jit(lambda v, sc, st: _apply_h_kernel(v, sc, st, cache_level=(1, 0)))
    sc = _pack_scanned((1, 0), jnp.asarray(xs), ham.z, ham.c)
    vec = (
        jnp.asarray(np.random.default_rng(1).normal(size=(2, size)))
        .astype(ham.c.dtype)
        .at[:, len(states) :]
        .set(0)
    )
    arms = {
        "(1,0)": lambda: k10(vec, sc, su),
        "P0": lambda: matvec_p0(vec, cpi, cpj, cpg, z, c, su, jnp.zeros(kmax)),
        "P2": lambda: matvec_p2(vec, cpi, cpj, cdp, d0),
    }
    ref = np.asarray(arms["(1,0)"]())[:, : len(states)]
    for k, f in arms.items():
        assert np.allclose(np.asarray(f())[:, : len(states)], ref, rtol=1e-11, atol=1e-11), k
    t = {k: [] for k in arms}
    for _ in range(9):
        for k, f in arms.items():
            t0 = time.perf_counter()
            jax.block_until_ready(f())
            t[k].append(time.perf_counter() - t0)
    b = {
        "(1,0)": xs.nbytes + su.nbytes,
        "P0": cpi.nbytes + cpj.nbytes + cpg.nbytes + su.nbytes,
        "P2": cpi.nbytes + cpj.nbytes + cdp.nbytes + d0.nbytes,
    }
    base = np.median(t["(1,0)"])
    print(
        f"{label}: J={ham.x.shape[0]}, N={len(states)}, hit rate {hit:.3f} | "
        + " | ".join(f"{k} {b[k] / size:.0f} B/slot, {base / np.median(t[k]):.2f}x" for k in arms)
    )


def main() -> None:
    rng = np.random.default_rng(0)
    mol = PauliSumXZ.from_paulisum(molecular_like(14, rng, 0.2))
    bits = np.unique(rng.integers(0, 2, size=(20000, 14), dtype=np.uint8), axis=0)
    run("molecular-like JW n=14, random states", mol, bits)
    run(
        "periodic XXZ n=24, Neel-Krylov (closed under hops)",
        xxz_hamiltonian(24, 1.0),
        xxz_krylov_states(24, 60000),
    )


if __name__ == "__main__":
    main()
