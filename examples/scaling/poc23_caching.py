"""POC 2 and 3: the caching axis -- a partial-J dial, and (1,1) versus (1,2).

Two related questions about ``cache_level``, measured together because they share a setup.

**POC 3 (measurement only, no new code).** ``rqutils/sqd.py``'s module docstring predicts sign-bit
caching (``cache_level[1] == 1``) costs ``kappa_bar * J * N`` bytes against ``8 J N`` or ``16 J N``
for full diagonals (``== 2``) -- an 8-16x memory saving for a popcount-and-shift that should be
nearly free. If that holds, ``(1, 1)`` is an underused option rather than a new idea. This measures
the actual arrays and the actual matvec time.

**POC 2 (new code).** ``cache_level`` is all-or-nothing across the J X-groups: either every group's
source indices are cached or none are. A partial dial -- cache the first ``J'`` groups, recompute the
rest -- turns a 6-point discrete choice into a continuous memory/time curve. Whether that is *useful*
depends entirely on the shape of the curve, which is the thing to measure: if time is flat in ``J'``
until it collapses at the end, the dial is worthless because only the endpoints matter.

POC 2 builds on POC 1's searchsorted, since after that result recomputation is 12-25x cheaper on CPU
(5.15x on a GH200) than the old sort, and the tradeoff shifts substantially. Both are reported. The
smaller GPU ratio does not change POC 2's "marginal" verdict, which rests on the *shape* of the curve
-- flat in ``J'`` until it collapses at the end, so only the endpoints matter -- not on the magnitude.

Run: uv run --extra qiskit python examples/scaling/poc23_caching.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax

jax.config.update("jax_enable_x64", True)

import functools

import jax.numpy as jnp
import numpy as np
from _scaling_common import header, make_problem, max_abs_diff, timeit
from poc1_searchsorted import xsource_searchsorted_u64

from rqutils.sqd import (
    apply_h,
    apply_xgrp,
    get_diag_signs,
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
    diag_signs = jax.block_until_ready(
        jax.lax.scan(lambda _, z: (None, get_diag_signs(z, states_u)), None, ham.z)[1]
    )
    diagonals = jax.block_until_ready(
        jax.lax.scan(lambda _, v: (None, get_diagonal(v[0], v[1], states_u)), None, (ham.z, ham.c))[
            1
        ]
    )
    return {
        "xsources": xsources,
        "diag_signs": diag_signs,
        "diagonals": diagonals,
        "states_u": states_u,
    }


def nbytes(arr) -> int:
    return int(np.asarray(arr).nbytes)


def poc3_signbits_vs_diagonals():
    header("POC 3: cache_level (1,1) sign bits vs (1,2) full diagonals")
    print("Claim under test: (1,1) is an 8-16x memory saving for a nearly-free popcount+shift.")
    print()
    for num_qubits, num_states, j, real in [
        (24, 200_000, 50, False),
        (24, 200_000, 50, True),
        (24, 200_000, 200, False),
    ]:
        p = make_problem(
            num_qubits,
            num_states,
            num_terms=max(j * 4, 200),
            num_xgroups=j,
            real_only=real,
            seed=21,
        )
        size = p.states_p.shape[0]
        states_u = jax.block_until_ready(uniquify_states(p.states_p, size))
        c = build_caches(p, states_u)
        vec = jnp.asarray(np.random.default_rng(0).normal(size=size).astype(p.hamiltonian.c.dtype))

        mv11 = functools.partial(
            apply_h, xsources=c["xsources"], diag_signs=c["diag_signs"], coeffs=p.hamiltonian.c
        )
        mv12 = functools.partial(apply_h, xsources=c["xsources"], diagonals=c["diagonals"])

        r11 = jax.block_until_ready(mv11(vec))
        r12 = jax.block_until_ready(mv12(vec))
        diff = max_abs_diff(r11, r12)

        t11 = timeit(lambda mv11=mv11, vec=vec: mv11(vec), "(1,1)", trials=5)
        t12 = timeit(lambda mv12=mv12, vec=vec: mv12(vec), "(1,2)", trials=5)

        b_signs, b_diags = nbytes(c["diag_signs"]), nbytes(c["diagonals"])
        # (1,1) additionally needs the coefficient array; (1,2) does not. Both keep xsources.
        b11 = b_signs + nbytes(p.hamiltonian.c)
        b12 = b_diags
        print(f"  {p.describe()}")
        print(
            f"    memory: (1,1)={b11 / 2**20:8.2f}MB  (1,2)={b12 / 2**20:8.2f}MB  "
            f"saving={b12 / b11:5.2f}x"
        )
        print(
            f"    matvec: (1,1)={t11.min_s * 1e3:7.2f}ms  (1,2)={t12.min_s * 1e3:7.2f}ms  "
            f"cost={t11.min_s / t12.min_s:5.2f}x   maxdiff={diff:.2e}"
        )
        print()


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


def poc2_partial_j():
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
    poc3_signbits_vs_diagonals()
    poc2_partial_j()
