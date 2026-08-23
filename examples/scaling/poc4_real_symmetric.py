"""POC 4: what the real-symmetric (all-even-Y) path is worth, end to end.

The premise needed checking before building anything, and it changed the question. ``apply_h``
already propagates float64 correctly when ``PauliSumXZ.c`` narrows to float64 -- the dtype flows from
``.c`` through ``get_diagonal`` to the output, verified directly. So there is no promotion bug to fix
and nothing to implement.

What is left is a measurement worth having: **how much does realness actually buy**, across the
matvec, the cache footprint, and a full ``ground_locg`` solve? And does ``ground_locg``'s
``work_dtype`` promotion (``rqutils/ground_locg.py:518``) preserve realness end to end, or quietly
promote to complex somewhere in the Rayleigh-Ritz path?

The comparison is deliberately awkward to make fair. A real and a complex Hamiltonian are *different
operators* with different spectra and different convergence behaviour, so a raw solve-time ratio
conflates "complex arithmetic is slower" with "this problem needed more iterations". Two controls:

- **Fixed iteration count** (``maxiter`` pinned, ``tol=0``) isolates per-iteration cost. This is the
  number to quote, and it mirrors why ``CLAUDE.md`` says to read ``per_it_ms`` from ``fixed_s``
  rather than ``solve_s``.
- **Same X/Z structure**: the real and complex problems are generated from the same seed and the same
  ``num_xgroups``, so J, N, and maxK match. Only the Y parity differs.

Run: uv run --extra qiskit python examples/scaling/poc4_real_symmetric.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax

jax.config.update("jax_enable_x64", True)

import functools

import jax.numpy as jnp
import numpy as np
from _scaling_common import fmt_ratio, header, make_problem, timeit

from rqutils.ground_locg import ground_locg
from rqutils.sqd import apply_h, get_diagonal, get_xsource, uniquify_states


def setup(problem):
    size = problem.states_p.shape[0]
    states_u = jax.block_until_ready(uniquify_states(problem.states_p, size))
    ham = problem.hamiltonian
    xsources = jax.block_until_ready(
        jax.lax.scan(lambda _, x: (None, get_xsource(x, states_u)), None, ham.x)[1]
    )
    diagonals = jax.block_until_ready(
        jax.lax.scan(lambda _, v: (None, get_diagonal(v[0], v[1], states_u)), None, (ham.z, ham.c))[
            1
        ]
    )
    vec = jnp.asarray(np.random.default_rng(0).normal(size=size).astype(ham.c.dtype))
    return states_u, xsources, diagonals, vec, size


def main():
    header("POC 4a: dtype propagation -- does realness survive the whole solve?")
    print("Checked directly rather than assumed: apply_h's output dtype follows PauliSumXZ.c, and")
    print("ground_locg promotes xinit to result_type(xinit, matvec output).")
    print()
    for real in [False, True]:
        p = make_problem(18, 8_000, num_terms=60, num_xgroups=30, real_only=real, seed=41)
        _states_u, xs, dg, vec, size = setup(p)
        mv = functools.partial(apply_h, cache_level=(1, 2))
        hv = jax.block_until_ready(mv(vec, (xs, dg), None))
        eigval, eigvec, niter, conv = ground_locg(
            lambda v, *a, mv=mv: mv(v, *a), vec, args=((xs, dg), None), maxiter=60
        )
        print(
            f"  real_only={real!s:<5s}  c={p.hamiltonian.c.dtype!s:<10s} "
            f"Hv={hv.dtype!s:<10s} eigvec={eigvec.dtype!s:<10s} "
            f"eigval={float(eigval):+.8f} niter={int(niter)} conv={bool(conv)}"
        )
    print()
    print("  If eigvec stays float64 on the real problem, realness survives end to end and there")
    print("  is nothing to implement -- only a cost to quantify below.")

    header("POC 4b: per-matvec cost, real vs complex, matched J/N/maxK")
    print(f"{'N':>9s}  {'J':>4s}  {'complex ms':>12s}  {'real ms':>10s}  {'verdict':>34s}")
    for num_states in [50_000, 200_000, 500_000]:
        ts = {}
        info = {}
        for real in [False, True]:
            p = make_problem(24, num_states, num_terms=200, num_xgroups=50, real_only=real, seed=42)
            _states_u, xs, dg, vec, size = setup(p)
            mv = functools.partial(apply_h, cache_level=(1, 2))
            ts[real] = timeit(
                lambda mv=mv, vec=vec, xs=xs, dg=dg: mv(vec, (xs, dg), None),
                f"real={real}",
                trials=5,
            )
            info[real] = (p.num_xgroups, size, nbytes_of(dg))
        assert info[False][0] == info[True][0], "J mismatch: comparison is not controlled"
        print(
            f"{info[True][1]:>9d}  {info[True][0]:>4d}  "
            f"{ts[False].min_s * 1e3:>10.2f}ms  {ts[True].min_s * 1e3:>8.2f}ms  "
            f"{fmt_ratio(ts[False], ts[True]):>34s}"
        )
        print(
            f"           diagonal cache: complex={info[False][2] / 2**20:7.2f}MB  "
            f"real={info[True][2] / 2**20:7.2f}MB  ({info[False][2] / info[True][2]:.2f}x)"
        )

    header("POC 4c: full solve at FIXED iteration count (tol=0) -- per-iteration cost")
    print("tol=0 forces exactly maxiter iterations, so this cannot be confounded by the two")
    print("operators needing different iteration counts to converge.")
    print(f"{'N':>9s}  {'iters':>6s}  {'complex ms':>12s}  {'real ms':>10s}  {'verdict':>34s}")
    maxiter = 40
    for num_states in [50_000, 200_000]:
        ts = {}
        for real in [False, True]:
            p = make_problem(24, num_states, num_terms=200, num_xgroups=50, real_only=real, seed=43)
            _states_u, xs, dg, vec, size = setup(p)
            mv = functools.partial(apply_h, cache_level=(1, 2))

            def solve(mv=mv, vec=vec, xs=xs, dg=dg):
                return ground_locg(
                    lambda v, *a, mv=mv: mv(v, *a),
                    vec,
                    args=((xs, dg), None),
                    maxiter=maxiter,
                    tol=0.0,
                )

            _ = jax.block_until_ready(solve())
            ts[real] = timeit(solve, f"real={real}", trials=3)
        print(
            f"{num_states:>9d}  {maxiter:>6d}  "
            f"{ts[False].min_s * 1e3:>10.2f}ms  {ts[True].min_s * 1e3:>8.2f}ms  "
            f"{fmt_ratio(ts[False], ts[True]):>34s}"
        )
        print(
            f"           per-iteration: complex={ts[False].min_s / maxiter * 1e3:6.3f}ms  "
            f"real={ts[True].min_s / maxiter * 1e3:6.3f}ms"
        )


def nbytes_of(arr) -> int:
    return int(np.asarray(arr).nbytes)


if __name__ == "__main__":
    main()
