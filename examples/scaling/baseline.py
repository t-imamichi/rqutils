"""Decompose SQD's cost to find which ceiling binds, before optimizing any of them.

Three separate limits are documented in ``rqutils/sqd.py``: the ``N <= 2**31`` single-device sort,
per-device memory for the caches, and the ``O(J*N)`` per-matvec cost. They scale differently, so
"improve scaling" means nothing until we know which one is in front. This script measures the three
components against N and against J independently:

- ``uniquify_states`` -- runs **once** per solve.
- ``get_xsource`` -- runs J times, and holds the ``2N`` sort that sets the hard ceiling.
- ``apply_h`` -- runs once per solver iteration, tens to hundreds of times.

The per-call costs are then weighted by call count, which is the only comparison that matters: a
component that is individually expensive but runs once can be irrelevant next to a cheap one in the
iteration loop.

Run: uv run --extra qiskit python examples/scaling/baseline.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax

jax.config.update("jax_enable_x64", True)

import functools

import jax.numpy as jnp
import numpy as np
from _scaling_common import header, make_problem, timeit

from rqutils.sqd import apply_h, get_diagonal, get_xsource, uniquify_states


def component_costs(problem, cache_level=(1, 2)):
    """Time each component of one SQD solve at a fixed problem size."""
    ham = problem.hamiltonian
    size = problem.states_p.shape[0]
    states_u = uniquify_states(problem.states_p, size)
    jax.block_until_ready(states_u)

    out = {}
    out["uniquify"] = timeit(
        lambda: uniquify_states(problem.states_p, size), "uniquify_states", trials=5
    )
    # One X signature only: the J-fold cost is this times J, and timing all J at once would hide
    # whether the per-signature cost itself scales.
    out["xsource_1"] = timeit(
        lambda: get_xsource(ham.x[0], states_u), "get_xsource (1 signature)", trials=5
    )
    out["diagonal_1"] = timeit(
        lambda: get_diagonal(ham.z[0], ham.c[0], states_u), "get_diagonal (1 group)", trials=5
    )

    # Full precomputation of all J source indices, as run_sqd does under cache_level[0]==1.
    def all_xsources():
        return jax.lax.scan(lambda _, x: (None, get_xsource(x, states_u)), None, ham.x)[1]

    out["xsource_all"] = timeit(all_xsources, f"all {problem.num_xgroups} xsources", trials=3)

    # One matvec under the fully-cached strategy, which is the steady-state solver cost.
    xsources = jax.block_until_ready(all_xsources())
    diagonals = jax.block_until_ready(
        jax.lax.scan(lambda _, v: (None, get_diagonal(v[0], v[1], states_u)), None, (ham.z, ham.c))[
            1
        ]
    )
    vec = jnp.asarray(np.random.default_rng(0).normal(size=size).astype(ham.c.dtype))
    mv = functools.partial(apply_h, cache_level=(1, 2))
    out["matvec"] = timeit(
        lambda: mv(vec, (xsources, diagonals), None), "apply_h (1,2) matvec", trials=5
    )
    return out


def main():
    header("BASELINE 1: component cost vs N (n=20, J=50)")
    print(f"{'N':>9s}  {'uniquify':>12s}  {'xsrc x1':>12s}  {'xsrc xJ':>12s}  {'matvec':>12s}")
    n_rows = []
    for num_states in [10_000, 50_000, 200_000, 500_000]:
        p = make_problem(20, num_states, num_terms=50, num_xgroups=50, seed=3)
        c = component_costs(p)
        n_rows.append((num_states, c))
        print(
            f"{num_states:>9d}  "
            f"{c['uniquify'].min_s * 1e3:>10.2f}ms  "
            f"{c['xsource_1'].min_s * 1e3:>10.2f}ms  "
            f"{c['xsource_all'].min_s * 1e3:>10.2f}ms  "
            f"{c['matvec'].min_s * 1e3:>10.2f}ms"
        )

    print()
    print("Scaling exponents (log-log fit, cost ~ N^alpha):")
    ns = np.array([r[0] for r in n_rows], dtype=float)
    for key in ["uniquify", "xsource_1", "xsource_all", "matvec"]:
        ts = np.array([r[1][key].min_s for r in n_rows])
        alpha = np.polyfit(np.log(ns), np.log(ts), 1)[0]
        print(f"  {key:<14s} alpha = {alpha:.2f}")

    header("BASELINE 2: component cost vs J (n=20, N=200k)")
    print(f"{'J':>5s}  {'xsrc xJ':>12s}  {'matvec':>12s}  {'matvec/J':>12s}")
    for j in [10, 25, 50, 100]:
        p = make_problem(20, 200_000, num_terms=max(j, 100), num_xgroups=j, seed=4)
        c = component_costs(p)
        print(
            f"{p.num_xgroups:>5d}  "
            f"{c['xsource_all'].min_s * 1e3:>10.2f}ms  "
            f"{c['matvec'].min_s * 1e3:>10.2f}ms  "
            f"{c['matvec'].min_s * 1e3 / p.num_xgroups:>10.3f}ms"
        )

    header("BASELINE 3: weighted totals -- what actually dominates a solve")
    # A solve is: 1 uniquify + 1 all-xsource precompute + niter matvecs.
    p = make_problem(20, 200_000, num_terms=100, num_xgroups=50, seed=5)
    c = component_costs(p)
    print(f"problem: {p.describe()}")
    for niter in [10, 50, 200]:
        setup = c["uniquify"].min_s + c["xsource_all"].min_s
        loop = c["matvec"].min_s * niter
        total = setup + loop
        print(
            f"  niter={niter:>4d}: setup={setup * 1e3:8.1f}ms ({setup / total * 100:4.1f}%)  "
            f"loop={loop * 1e3:8.1f}ms ({loop / total * 100:4.1f}%)  total={total * 1e3:8.1f}ms"
        )

    header("BASELINE 4: memory footprint of the caches (bytes, analytic)")
    print("Per rqutils/sqd.py's module docstring: 4JN for source indices, 8JN/16JN for diagonals.")
    print(f"{'N':>12s}  {'J':>4s}  {'S':>10s}  {'xsrc 4JN':>12s}  {'diag 16JN':>12s}")
    for num_states in [1_000_000, 100_000_000, 2**31]:
        for j in [50, 500]:
            nbytes_s = int(np.ceil(21 / 8)) * num_states
            print(
                f"{num_states:>12d}  {j:>4d}  "
                f"{nbytes_s / 2**30:>8.2f}GB  "
                f"{4 * j * num_states / 2**30:>10.2f}GB  "
                f"{16 * j * num_states / 2**30:>10.2f}GB"
            )


if __name__ == "__main__":
    main()
