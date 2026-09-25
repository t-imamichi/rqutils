"""POC 2: the caching axis -- a partial-J source-index dial.

``cache_level`` is all-or-nothing across the J X-groups: either every group's source indices are
cached or none are. A partial dial -- cache the first ``J'`` groups, recompute the rest -- turns the
discrete choice into a continuous memory/time curve. Whether that is *useful* depends entirely on the
shape of the curve, which is the thing to measure: if time is flat in ``J'`` until it collapses at the
end, the dial is worthless because only the endpoints matter.

POC 2 builds on POC 1's searchsorted, since after that result recomputation is 12-25x cheaper on CPU
(5.15x on a GH200) than the old sort, and the tradeoff shifts substantially. Both are reported. The
smaller GPU ratio does not change POC 2's "marginal" verdict, which rests on the *shape* of the curve
-- flat in ``J'`` until it collapses at the end, so only the endpoints matter -- not on the magnitude.

POC 3, ``cache_level`` (1,1) sign bits against (1,2) full diagonals, is gone with the level: (1,1) was
removed from ``sqd`` as dominated by (1,2) on both memory and time.

Run: uv run --extra qiskit python poc/caching.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax

jax.config.update("jax_enable_x64", True)


import jax.numpy as jnp
import numpy as np
from _scaling_common import header, make_problem, max_abs_diff, timeit
from searchsorted import xsource_searchsorted_u64

from rqutils.sqd import (
    apply_h,
    apply_xgrp,
    get_diagonal,
    get_xsource,
    uniquify_states,
)


def build_caches(problem, states_u):
    """Build every cache variant once, returning the arrays and their true byte sizes."""
    ham = problem.hamiltonian
    xsources = jax.block_until_ready(
        jax.lax.scan(lambda _, x: (None, get_xsource(x, states_u)), None, ham.x)[1]
    )
    diagonals = jax.block_until_ready(
        jax.lax.scan(lambda _, v: (None, get_diagonal(v[0], v[1], states_u)), None, (ham.z, ham.c))[
            1
        ]
    )
    return {
        "xsources": xsources,
        "diagonals": diagonals,
        "states_u": states_u,
    }


def nbytes(arr) -> int:
    return int(np.asarray(arr).nbytes)


def _partial_matvec_factory(xsources_cached, xsigs_uncached, diagonals, states_u, use_searchsorted):
    """Matvec caching the first J' source-index arrays and recomputing the rest.

    Two scans rather than one: the cached and uncached halves carry different leading-axis types
    (int32 indices versus uint8 signatures), so a single scan cannot cover both.
    """
    xsource_fn = xsource_searchsorted_u64 if use_searchsorted else get_xsource
    ncached = xsources_cached.shape[0] if xsources_cached is not None else 0

    @jax.jit
    def matvec(vec):
        out = jnp.zeros_like(vec)
        if ncached:

            def cached_step(acc, val):
                return acc + apply_xgrp(val[0], val[1], vec), None

            out = jax.lax.scan(cached_step, out, (xsources_cached, diagonals[:ncached]))[0]
        if xsigs_uncached is not None and xsigs_uncached.shape[0]:

            def uncached_step(acc, val):
                xsrc = xsource_fn(val[0], states_u)
                return acc + apply_xgrp(xsrc, val[1], vec), None

            out = jax.lax.scan(uncached_step, out, (xsigs_uncached, diagonals[ncached:]))[0]
        return out

    return matvec


def partial_j():
    header("POC 2: partial-J source-index caching -- shape of the memory/time curve")
    num_qubits, num_states, j = 24, 200_000, 50
    p = make_problem(num_qubits, num_states, num_terms=200, num_xgroups=j, seed=22)
    size = p.states_p.shape[0]
    states_u = jax.block_until_ready(uniquify_states(p.states_p, size))
    c = build_caches(p, states_u)
    vec = jnp.asarray(np.random.default_rng(0).normal(size=size).astype(p.hamiltonian.c.dtype))
    j_actual = p.num_xgroups
    print(f"  {p.describe()}")

    # Keyword form: this is a one-shot call, so naming the arrays is both clearer and unmispairable.
    # The benchmark thunks above stay on the positional form deliberately -- they are measuring the
    # shape the solver actually calls, which splats a tuple.
    ref = jax.block_until_ready(apply_h(vec, xsources=c["xsources"], diagonals=c["diagonals"]))

    for use_ss in [False, True]:
        label = "searchsorted (POC 1)" if use_ss else "library sort"
        print(f"\n  recomputation via {label}:")
        print(
            f"  {'J_cached':>9s}  {'cache MB':>9s}  {'matvec ms':>10s}  "
            f"{'vs full-cache':>14s}  {'maxdiff':>10s}"
        )
        for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
            ncached = round(j_actual * frac)
            xs_c = c["xsources"][:ncached] if ncached else None
            xs_u = p.hamiltonian.x[ncached:] if ncached < j_actual else None
            mv = _partial_matvec_factory(xs_c, xs_u, c["diagonals"], states_u, use_ss)
            got = jax.block_until_ready(mv(vec))
            diff = max_abs_diff(ref, got)
            t = timeit(lambda mv=mv, vec=vec: mv(vec), f"J'={ncached}", trials=3)
            cache_mb = (nbytes(xs_c) if ncached else 0) / 2**20
            print(
                f"  {ncached:>9d}  {cache_mb:>9.2f}  {t.min_s * 1e3:>10.2f}  "
                f"{t.min_s * 1e3 / 1.0:>14.2f}  {diff:>10.2e}"
            )

    print()
    print("  Read the curve shape, not the endpoints: a dial is only useful if intermediate")
    print("  points are on a smooth tradeoff. Cache size is exactly linear in J' (4*J'*N bytes),")
    print(
        "  so if time is also linear the dial is real; if time collapses only at J'=J, it is not."
    )


if __name__ == "__main__":
    partial_j()
