"""POC 6: float32 matvec with a float64 solver -- accuracy against speed.

The matvec dominates the per-iteration cost, so running it in float32 halves its bandwidth. The
Rayleigh-Ritz step must stay float64: ``docs/locg.md``'s I1-I7 are all cases where reduced precision
in *that* step returns a plausible wrong number rather than failing, and the ``eigenpair_3x3``
balancing exists specifically because the intermediates lose significance.

Two obstacles, both real and both measured rather than asserted:

1. ``ground_locg`` derives its convergence tolerance from ``work_dtype`` (line ~534,
   ``jnp.finfo(work_dtype).eps``), which is ``result_type(xinit, matvec output)``. Feed it an f32
   matvec and the tolerance loosens by ~9 orders of magnitude -- the module docstring already warns
   about exactly this ("a float32 xinit on a complex128 problem would otherwise silently loosen this
   by nine orders of magnitude"). So a naive f32 matvec does not merely lose accuracy, it also stops
   the solver early and *reports converged*. The POC therefore casts back to f64 inside the matvec
   wrapper, keeping ``work_dtype`` f64 while the arithmetic happens in f32.

2. The residual ``r = Ax - theta x`` is a difference of nearly equal quantities near convergence. With
   an f32 matvec its error floor is ~1e-7 relative, so the f64 convergence test can become
   unsatisfiable -- the solver would run to ``maxiter`` and report not-converged. Measured below as
   the achieved residual, not assumed.

The fp32 trap ``CLAUDE.md`` records for an eigensolve is relevant here too: a large dynamic
range makes balancing destroy small eigenvalues outright. That is why accuracy is reported as the
eigenvalue error against a full-f64 reference on the *same* operator, with the residual normalized by
the operator norm rather than by |theta|.

Run: uv run --extra qiskit python examples/scaling/poc6_mixed_precision.py
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
from rqutils.sqd import _diag_from_z, apply_h, uniquify_states, xsource


def setup(problem):
    size = problem.states_p.shape[0]
    states_u = jax.block_until_ready(uniquify_states(problem.states_p, size))
    ham = problem.hamiltonian
    xsources = jax.block_until_ready(
        jax.lax.scan(lambda _, x: (None, xsource(x, states_u)), None, ham.x)[1]
    )
    diagonals = jax.block_until_ready(
        jax.lax.scan(lambda _, v: (None, _diag_from_z(v[0], v[1], states_u)), None, (ham.z, ham.c))[
            1
        ]
    )
    vec = jnp.asarray(np.random.default_rng(0).normal(size=size).astype(ham.c.dtype))
    return states_u, xsources, diagonals, vec, size


def make_matvecs(xsources, diagonals):
    """Return (f64 matvec, mixed-precision matvec).

    The mixed version casts inputs down to f32, does the gather-and-multiply there, and casts the
    result back to f64. Casting back is what keeps ``ground_locg``'s ``work_dtype`` -- and therefore
    its convergence tolerance -- at f64; without it the tolerance loosens by ~1e9 and the solver
    stops early while reporting success.
    """
    base = functools.partial(apply_h, cache_level=(1, 2))
    diag32 = diagonals.astype(jnp.complex64 if jnp.iscomplexobj(diagonals) else jnp.float32)

    def mv64(v, *args):
        return base(v, (xsources, diagonals), None)

    def mv_mixed(v, *args):
        v32 = v.astype(diag32.dtype)
        out32 = base(v32, (xsources, diag32), None)
        return out32.astype(v.dtype)

    return mv64, mv_mixed


def main():
    header("POC 6a: matvec-only cost, f64 vs f32 (CPU -- understates GPU)")
    print("On a GPU the f32 win comes from halved bandwidth and 2x FLOP throughput; on this CPU")
    print("backend only the bandwidth half applies, so treat these as a LOWER bound.")
    print(f"{'N':>9s}  {'dtype':>9s}  {'f64 ms':>9s}  {'f32 ms':>9s}  {'verdict':>34s}")
    for real in [True, False]:
        for num_states in [200_000, 500_000]:
            p = make_problem(24, num_states, num_terms=200, num_xgroups=50, real_only=real, seed=61)
            _, xs, dg, vec, size = setup(p)
            mv64, mvmx = make_matvecs(xs, dg)
            _ = jax.block_until_ready(mv64(vec))
            _ = jax.block_until_ready(mvmx(vec))
            t64 = timeit(lambda mv64=mv64, vec=vec: mv64(vec), "f64", trials=5)
            tmx = timeit(lambda mvmx=mvmx, vec=vec: mvmx(vec), "mixed", trials=5)
            print(
                f"{size:>9d}  {'real' if real else 'complex':>9s}  "
                f"{t64.min_s * 1e3:>7.2f}ms  {tmx.min_s * 1e3:>7.2f}ms  "
                f"{fmt_ratio(t64, tmx):>34s}"
            )

    header("POC 6b: accuracy -- eigenvalue error and achieved residual")
    print("Reference is a full-f64 solve on the SAME operator. Residual is normalized by ||A||_est")
    print("(estimated as max|diagonal| + row-sum bound) rather than |theta|, per CLAUDE.md's note")
    print("that normalizing by |theta| misleads when the spectrum straddles zero.")
    print()
    print(
        f"{'N':>8s}  {'dtype':>8s}  {'ref eigval':>15s}  {'mixed eigval':>15s}  "
        f"{'rel err':>10s}  {'iters':>11s}  {'conv':>11s}"
    )
    for real in [True, False]:
        for num_states in [50_000, 200_000]:
            p = make_problem(24, num_states, num_terms=200, num_xgroups=50, real_only=real, seed=62)
            _, xs, dg, vec, size = setup(p)
            mv64, mvmx = make_matvecs(xs, dg)

            e64, _v64, n64, c64 = ground_locg(lambda v, *a, mv64=mv64: mv64(v), vec, maxiter=300)
            emx, _vmx, nmx, cmx = ground_locg(lambda v, *a, mvmx=mvmx: mvmx(v), vec, maxiter=300)
            e64, emx = float(e64), float(emx)
            rel = abs(emx - e64) / max(abs(e64), 1e-300)
            print(
                f"{size:>8d}  {'real' if real else 'complex':>8s}  "
                f"{e64:>15.10f}  {emx:>15.10f}  {rel:>10.2e}  "
                f"{int(n64):>4d}/{int(nmx):<6d}  {bool(c64)!s:>5s}/{bool(cmx)!s:<5s}"
            )
    print()
    print("  'iters' is f64/mixed and 'conv' is f64/mixed. A mixed run that needs MORE iterations")
    print("  or fails to converge has given back the per-iteration saving -- read both columns")
    print("  together with 6c before believing any speedup.")

    header("POC 6c: end-to-end solve time -- does the per-matvec win survive?")
    print("Fixed iteration count (tol=0) isolates per-iteration cost; the converged solve shows")
    print("whether extra iterations eat the gain.")
    print(
        f"{'N':>8s}  {'dtype':>8s}  {'mode':>10s}  {'f64 ms':>10s}  {'mixed ms':>10s}  "
        f"{'verdict':>34s}"
    )
    for real in [True, False]:
        num_states = 200_000
        p = make_problem(24, num_states, num_terms=200, num_xgroups=50, real_only=real, seed=63)
        _, xs, dg, vec, size = setup(p)
        mv64, mvmx = make_matvecs(xs, dg)

        for mode, kw in [
            ("fixed(40)", {"maxiter": 40, "tol": 0.0}),
            ("converge", {"maxiter": 300}),
        ]:

            def s64(mv64=mv64, vec=vec, kw=kw):
                return ground_locg(lambda v, *a, mv64=mv64: mv64(v), vec, **kw)

            def smx(mvmx=mvmx, vec=vec, kw=kw):
                return ground_locg(lambda v, *a, mvmx=mvmx: mvmx(v), vec, **kw)

            _ = jax.block_until_ready(s64())
            _ = jax.block_until_ready(smx())
            t64 = timeit(s64, "f64", trials=3)
            tmx = timeit(smx, "mixed", trials=3)
            print(
                f"{size:>8d}  {'real' if real else 'complex':>8s}  {mode:>10s}  "
                f"{t64.min_s * 1e3:>8.2f}ms  {tmx.min_s * 1e3:>8.2f}ms  "
                f"{fmt_ratio(t64, tmx):>34s}"
            )

    header("POC 6d: the tolerance trap -- what a NAIVE f32 matvec does")
    print(
        "Demonstrates obstacle (1) concretely: if the matvec RETURNS f32 instead of casting back,"
    )
    print("ground_locg's work_dtype becomes f32 and tol = finfo(f32).eps -- ~1e9 looser.")
    p = make_problem(24, 50_000, num_terms=200, num_xgroups=50, real_only=True, seed=64)
    _, xs, dg, vec, size = setup(p)
    base = functools.partial(apply_h, cache_level=(1, 2))
    dg32 = dg.astype(jnp.float32)

    def mv_naive(v, *a):
        # No cast back: work_dtype collapses to f32 and the tolerance loosens with it.
        return base(v.astype(jnp.float32), (xs, dg32), None)

    e_ref = float(ground_locg(lambda v, *a: base(v, (xs, dg), None), vec, maxiter=300)[0])
    en, _, nn, cn = ground_locg(mv_naive, vec.astype(jnp.float32), maxiter=300)
    print(f"  f64 reference : eigval={e_ref:.10f}")
    print(f"  naive f32     : eigval={float(en):.10f}  iters={int(nn)}  converged={bool(cn)}")
    print(f"  relative error: {abs(float(en) - e_ref) / abs(e_ref):.2e}")
    print(
        f"  eps(f32)={float(jnp.finfo(jnp.float32).eps):.2e} vs "
        f"eps(f64)={float(jnp.finfo(jnp.float64).eps):.2e}  "
        f"-> tolerance {float(jnp.finfo(jnp.float32).eps / jnp.finfo(jnp.float64).eps):.1e}x looser"
    )


if __name__ == "__main__":
    main()
