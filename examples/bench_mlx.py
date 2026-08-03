"""Benchmark the SQD eigensolver loop across JAX and MLX, CPU and GPU.

Only the solver loop is compared. Setup (uniquification, X-source lookup, diagonal
composition) always runs in JAX on CPU and is not timed, so every arm consumes identical
arrays -- see docs/superpowers/specs/2026-08-03-mlx-sqd-poc-design.md. The one exception is
``jax-metal-f32``: JAX's x64 flag is process-global, and Metal supports neither float64 nor
complex128, so that arm's *setup* (not just its solve) runs at reduced precision and is
therefore not solving byte-identical arrays to every other arm. This is disclosed via a
``setup_precision`` field in every result and a footnote in the text report -- see I3 in
.superpowers/sdd/2026-08-03-mlx-sqd-poc/final-review.md.

JAX's platform and x64 flag are process-global and must be set before importing jax, so each
JAX arm needs its own process. --all re-executes this script once per arm and collates.

.. code-block:: sh

    uv run python examples/bench_mlx.py --arm mlx-gpu-f32
    uv run python examples/bench_mlx.py --all --num-qubits 10
    uv run python examples/bench_mlx.py --all --num-qubits 10 --json > results.json

Two metrics are reported per arm: per-iteration cost at a fixed iteration count (identical
work per arm, so a clean speed comparison) and time-to-convergence with its iteration count
(what production actually pays). Reporting both makes it visible when fp32 is faster per
iteration but needs more iterations.

CAVEAT -- MLX per-call graph reconstruction is not subtracted out. ``ground_locg_mlx`` is
plain Python: every timed call re-walks the iteration loop and re-constructs MLX's op graph
from scratch, in Python, before any device work happens. ``ground_locg`` (the JAX original)
is ``@jax.jit``: tracing happens once and is reported in ``compile_s``, and every subsequent
timed call dispatches an already-compiled executable with no further Python-level graph
construction. So the MLX arms' ``fixed_s`` / ``per_it_ms`` / ``solve_s`` numbers include a
per-call Python graph-construction cost that the JAX arms' numbers do not pay in the same
column. This biases the comparison AGAINST MLX (JAX's steady-state number is cleaner than
MLX's), which is the safer direction for a benchmark whose main risk is a bogus MLX win --
but the magnitude has not been measured here (it would require timing graph construction
without ``mx.eval``, which this PoC does not do; see I4 in the final review). Do not read
"MLX is slower per iteration" as a verdict on MLX's kernels without accounting for this.
"""
import argparse
import json
import os
import subprocess
import sys

ARMS = ('jax-cpu-f64', 'jax-cpu-f32', 'jax-metal-f32',
        'mlx-cpu-f64', 'mlx-cpu-f32', 'mlx-gpu-f32')

# Relative tolerance for the correctness gate, by precision.
RTOL = {'f64': 1e-9, 'f32': 1e-4}

# Above this many qubits, the 2^n-by-2^n brute-force cross-check (build the full dense matrix
# by Kronecker products) is skipped by default: n=12 measures ~4 s / ~270 MB per Pauli string
# here, which is tolerable, but n=13 is already ~20 s and n=14 is ~470 s and >4 GB per string
# (measured on this machine; see final-review.md I2). n=10 -- the spec's gate size -- and n=12
# both always run it. The dense-from-solver-inputs gate (dense_reference) is NOT size-limited
# and always runs; only this independent 2^n cross-check is skipped, and skipping is always
# disclosed (see run_arm). --skip-brute-force remains available as an explicit override at any
# size.
BRUTE_FORCE_MAX_QUBITS = 12

# Absolute tolerance on matvec_err (||ported_matvec(v) - H @ v||_inf for a random v), by
# precision. Probing with a fixed-seed random vector (not the one-hot vinit) is required by the
# design spec and is what gives this gate its sensitivity to gather bugs -- see I1 in
# final-review.md. Re-measured directly with the random probe (jax-cpu-f64/f32) across
# n=10/12 x num_paulis=20/100 x num_states=200/1000 (see final-fix-report.md for the exact
# runs): f64 error tops out at 3.55e-15 (now that dozens of O(1) terms accumulate per output
# entry instead of one, the floor rose off the one-hot probe's exact 0.0, but is still far
# below the threshold -- a >280,000x margin), f32 error tops out at 1.69e-6, a >59x margin
# under 1e-4 and still comfortably above f32 eps (~1.19e-7) times a modest accumulation
# factor. Both margins stay far below "the matvec is just wrong" territory (mismatches from a
# real bug, e.g. bad mx.take indexing, show up as order-1 errors, not order-1e-6/1e-15).
MATVEC_ATOL = {'f64': 1e-9, 'f32': 1e-4}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--arm', choices=ARMS, help='Single arm to run.')
    parser.add_argument('--all', action='store_true',
                        help='Run every arm, one subprocess each, and collate.')
    parser.add_argument('--num-qubits', type=int, default=14)
    parser.add_argument('--num-paulis', type=int, default=100)
    parser.add_argument('--num-states', type=int, default=4000)
    parser.add_argument('--repeat', type=int, default=3,
                        help='Timed iterations after warmup.')
    parser.add_argument('--fixed-iters', type=int, default=100,
                        help='Iteration count for the fixed-work measurement.')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--json', action='store_true', help='Emit JSON instead of a table.')
    parser.add_argument('--skip-brute-force', action='store_true',
                        help='Explicitly skip the 2^n reference at any --num-qubits. It is '
                             f'auto-skipped above --num-qubits {BRUTE_FORCE_MAX_QUBITS} '
                             'regardless of this flag; either way the skip is reported.')
    parser.add_argument('--self-test-break-gate', action='store_true',
                        help=argparse.SUPPRESS)  # corrupts the problem to prove the gate bites
    options = parser.parse_args(argv)
    if not options.arm and not options.all:
        parser.error('specify --arm or --all')
    return options


def run_arm(arm, options):
    """Run one arm in this process. Returns a result dict."""
    framework, device, precision = arm.split('-')

    # JAX must be configured before import, so do it here rather than at module scope.
    #
    # Setup (_bench_common) should run in float64 whenever the backend allows it, regardless
    # of the arm's target solve precision -- that is the whole point of "every arm consumes
    # identical arrays". Forcing jax_enable_x64 off to match an f32 arm would silently corrupt
    # the shared setup stage itself: get_diagonal accumulates in coeffs.dtype, so a float32
    # PauliSumXZ produces different (less precise) xsources/diagonals than a float64 one, and
    # then no two arms would actually be comparing the same problem. The one exception is the
    # metal backend, which supports neither float64 nor complex128 at all (see examples/bench.py);
    # jax_enable_x64 is process-global, so a metal arm's setup necessarily runs at reduced
    # precision too, same as its timed solve, matching this repo's existing
    # JAX_PLATFORMS=metal convention.
    if framework == 'jax':
        os.environ['JAX_PLATFORMS'] = 'cpu' if device == 'cpu' else device
    import jax
    jax.config.update('jax_enable_x64', device != 'metal')

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from _bench_common import (generate_problem, build_solver_inputs, dense_reference,
                               brute_force_reference)
    import time
    import numpy as np

    if framework == 'jax' and device == 'metal':
        try:
            backend = jax.default_backend()
        except RuntimeError as exc:
            return {'arm': arm, 'status': 'skipped',
                    'reason': f'jax metal backend unavailable: {exc}'}
        if backend != 'metal':
            return {'arm': arm, 'status': 'skipped',
                    'reason': f'jax backend is {backend}, not metal'}

    setup_start = time.perf_counter()
    pauli_strings, coeffs, states = generate_problem(
        options.num_qubits, options.num_paulis, options.num_states, options.seed
    )
    inputs = build_solver_inputs(pauli_strings, coeffs, states)
    setup_s = time.perf_counter() - setup_start

    if options.self_test_break_gate:
        # Corrupt the diagonals so the solver cannot reach the reference eigenvalue.
        inputs.diagonals = inputs.diagonals * 2.5 + 1.0

    # Matvec probe vector: a fixed-seed random vector, per the design spec (line 213), shared
    # verbatim between _time_jax and _time_mlx so their matvec_err values are comparable. A
    # one-hot vector (e.g. inputs.vinit) has a single nonzero entry and is nearly blind to
    # gather bugs such as a dropped X group or an off-by-one `take` index -- see I1 in
    # final-review.md, which measured 1% vs 100% detection of injected perturbations. The seed
    # is fixed (not options.seed) so the probe is reproducible independent of the problem seed.
    matvec_probe = np.random.default_rng(20260803).normal(size=inputs.subspace_dim)

    # setup_precision reflects the precision the SHARED SETUP stage ran at, which is float64
    # for every arm except jax-metal-f32 (see the jax_enable_x64 comment above). This is
    # independent of `precision`, the arm's target SOLVE precision: jax-cpu-f32's setup is
    # still float64, only its solve is float32. Disclosed in every result (I3) because an arm
    # whose setup precision differs is not solving byte-identical arrays to the rest.
    setup_precision = 'f32' if device == 'metal' else 'f64'

    matrix, reference = dense_reference(inputs)
    brute_force_note = None
    skip_brute_force = options.skip_brute_force
    if not skip_brute_force and options.num_qubits > BRUTE_FORCE_MAX_QUBITS:
        skip_brute_force = True
        brute_force_note = (
            f'brute-force 2^n cross-check auto-skipped: --num-qubits {options.num_qubits} > '
            f'{BRUTE_FORCE_MAX_QUBITS} (the 2^n dense matrix becomes too large/slow -- see '
            'BRUTE_FORCE_MAX_QUBITS in this module). The dense-from-solver-inputs gate '
            '(dense_reference) still ran.'
        )
        print(f'NOTE: {brute_force_note}', file=sys.stderr)
    elif skip_brute_force:
        brute_force_note = 'brute-force 2^n cross-check skipped: --skip-brute-force was passed.'
        print(f'NOTE: {brute_force_note}', file=sys.stderr)

    if not skip_brute_force:
        brute = brute_force_reference(pauli_strings, coeffs, states)
        # The threshold scales with setup precision, not solve precision: a float32 setup
        # (jax-metal-f32 only) accumulates rounding error in the setup chain itself (measured
        # |H_metal - H_cpu|_inf ~= 2.79e-08 in the final review), which a fixed 1e-9 threshold
        # sits right at the edge of -- passing or failing depending on n by accident. Scaling by
        # RTOL[setup_precision] makes the gate's behaviour depend on a disclosed, principled
        # quantity instead of luck.
        brute_force_rtol = RTOL[setup_precision]
        if abs(reference - brute) > brute_force_rtol * max(1., abs(brute)):
            raise SystemExit(f'gate failed: dense reference {reference} disagrees with '
                             f'brute force {brute} by more than rtol={brute_force_rtol} '
                             f'(setup_precision={setup_precision}) -- the setup chain is wrong')

    rtol = RTOL[precision]
    if framework == 'jax':
        result = _time_jax(arm, inputs, precision, options, matrix, matvec_probe)
    else:
        result = _time_mlx(arm, inputs, device, precision, options, matrix, matvec_probe)

    # Gate: the ported matvec (apply_h_xz_mlx, or apply_h_xz_cached re-checked for symmetry)
    # must agree with the dense H @ v built straight from the same solver inputs. This isolates
    # a matvec bug (e.g. mx.take behaving unlike jax's fill-mode gather on out-of-bounds
    # indices) from a solver-loop bug -- without it, a broken matvec would only ever be visible
    # as a wrong eigenvalue, which is much harder to root-cause. Must fail before any timing
    # number is reported, same as the eigenvalue gate below.
    # Explicit non-finite check, ahead of the comparison-based gates below. A comparison-based
    # guard cannot catch NaN: `nan > x` and `abs(nan - y) > z` are both False for every x, y, z
    # under IEEE 754, so a NaN result silently falls through a `if err > atol: raise` gate
    # instead of tripping it. np.isfinite is required here specifically because it is not a
    # comparison -- do not "simplify" this back into the threshold checks below. The iteration
    # count is the diagnostic: hitting maxiter (iters == the configured cap) means the solver
    # never converged, as opposed to a numerical blow-up partway through.
    if not np.isfinite(result['matvec_err']):
        raise SystemExit(f'gate failed for {arm}: matvec error is non-finite '
                         f'({result["matvec_err"]!r}) after {result["iters"]} solver '
                         'iterations -- solver did not converge cleanly')

    if not np.isfinite(result['eigval']):
        raise SystemExit(f'gate failed for {arm}: eigenvalue is non-finite '
                         f'({result["eigval"]!r}) after {result["iters"]} solver iterations '
                         '-- solver did not converge cleanly')

    matvec_atol = MATVEC_ATOL[precision]
    if result['matvec_err'] > matvec_atol:
        raise SystemExit(f'gate failed for {arm}: matvec error {result["matvec_err"]} exceeds '
                         f'atol={matvec_atol}')

    if abs(result['eigval'] - reference) > rtol * max(1., abs(reference)):
        raise SystemExit(f'gate failed for {arm}: eigenvalue {result["eigval"]} differs from '
                         f'reference {reference} by more than rtol={rtol}')

    result['reference'] = reference
    result['setup_precision'] = setup_precision
    result['brute_force_note'] = brute_force_note
    result['setup_s'] = setup_s
    result['status'] = 'ok'
    return result


def _time_jax(arm, inputs, precision, options, matrix, matvec_probe):
    import numpy as np
    import jax
    import jax.numpy as jnp
    from rqutils.ground_locg import ground_locg
    from rqutils.sqd import apply_h_xz_cached
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from _bench_common import timeit

    dtype = np.float64 if precision == 'f64' else np.float32
    xsources = jnp.asarray(inputs.xsources)
    diagonals = jnp.asarray(inputs.diagonals, dtype=dtype)
    vinit = jnp.asarray(inputs.vinit, dtype=dtype)

    # Gate: the ported matvec and the original must agree on the same input, probed with a
    # random vector (not vinit) -- see the matvec_probe comment in run_arm.
    probe = jnp.asarray(matvec_probe, dtype=dtype)
    matvec_out = np.asarray(apply_h_xz_cached(probe, xsources, diagonals), dtype=np.float64)
    matvec_err = float(np.abs(matvec_out - matrix @ matvec_probe).max())

    def fixed():
        return ground_locg(apply_h_xz_cached, vinit, args=(xsources, diagonals),
                           maxiter=options.fixed_iters, tol=0.)

    compile_s, fixed_s = timeit(fixed, options.repeat, jax.block_until_ready)

    def solve():
        return ground_locg(apply_h_xz_cached, vinit, args=(xsources, diagonals))

    _, solve_s = timeit(solve, options.repeat, jax.block_until_ready)
    eigval, _, iters = solve()

    return {'arm': arm, 'compile_s': compile_s, 'fixed_s': fixed_s,
            'per_it_ms': fixed_s / options.fixed_iters * 1e3,
            'solve_s': solve_s, 'iters': int(iters), 'eigval': float(eigval),
            'matvec_err': matvec_err}


def _time_mlx(arm, inputs, device, precision, options, matrix, matvec_probe):
    import numpy as np
    import mlx.core as mx
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from _bench_common import timeit
    from ground_locg_mlx import apply_h_xz_mlx, ground_locg_mlx

    mx.set_default_device(mx.cpu if device == 'cpu' else mx.gpu)
    dtype = mx.float64 if precision == 'f64' else mx.float32
    xsources = mx.array(inputs.xsources)
    diagonals = mx.array(inputs.diagonals).astype(dtype)
    vinit = mx.array(inputs.vinit).astype(dtype)

    # Same random probe as _time_jax (identical values, only the array framework differs), so
    # matvec_err is comparable across the two paths -- see the matvec_probe comment in run_arm.
    probe = mx.array(matvec_probe).astype(dtype)
    matvec_out = np.asarray(apply_h_xz_mlx(probe, xsources, diagonals), dtype=np.float64)
    matvec_err = float(np.abs(matvec_out - matrix @ matvec_probe).max())

    def sync(result):
        # MLX is lazy: without this we would time graph construction, not computation.
        mx.eval(result[1])
        return result

    def fixed():
        return ground_locg_mlx(apply_h_xz_mlx, vinit, args=(xsources, diagonals),
                               maxiter=options.fixed_iters, tol=0.)

    compile_s, fixed_s = timeit(fixed, options.repeat, sync)

    def solve():
        return ground_locg_mlx(apply_h_xz_mlx, vinit, args=(xsources, diagonals))

    _, solve_s = timeit(solve, options.repeat, sync)
    eigval, _, iters = solve()

    return {'arm': arm, 'compile_s': compile_s, 'fixed_s': fixed_s,
            'per_it_ms': fixed_s / options.fixed_iters * 1e3,
            'solve_s': solve_s, 'iters': int(iters), 'eigval': float(eigval),
            'matvec_err': matvec_err}


def run_all(options):
    """Run every arm in its own subprocess and collate the results."""
    results = []
    for arm in ARMS:
        argv = [sys.executable, os.path.abspath(__file__), '--arm', arm, '--json',
                '--num-qubits', str(options.num_qubits),
                '--num-paulis', str(options.num_paulis),
                '--num-states', str(options.num_states),
                '--repeat', str(options.repeat),
                '--fixed-iters', str(options.fixed_iters),
                '--seed', str(options.seed)]
        if options.skip_brute_force:
            argv.append('--skip-brute-force')
        proc = subprocess.run(argv, capture_output=True, text=True)
        if proc.returncode != 0:
            results.append({'arm': arm, 'status': 'failed',
                            'reason': (proc.stderr or proc.stdout).strip().split('\n')[-1]})
            continue
        try:
            results.append(json.loads(proc.stdout))
        except json.JSONDecodeError:
            results.append({'arm': arm, 'status': 'failed',
                            'reason': f'unparseable output: {proc.stdout[:200]}'})
    return results


def report(results, as_json):
    if as_json:
        print(json.dumps({'results': results}, indent=2))
        return

    header = (f'{"arm":<15}{"setup_s":>9}{"compile_s":>10}{"fixed_s":>10}{"per_it_ms":>11}'
              f'{"solve_s":>10}{"iters":>7}{"matvec_err":>12}  eigval')
    print(header)
    print('-' * len(header))
    for row in results:
        if row.get('status') != 'ok':
            print(f'{row["arm"]:<15}{row.get("status", "?"):>10}  {row.get("reason", "")}')
            continue
        print(f'{row["arm"]:<15}{row["setup_s"]:>9.4f}{row["compile_s"]:>10.4f}'
              f'{row["fixed_s"]:>10.4f}{row["per_it_ms"]:>11.3f}{row["solve_s"]:>10.4f}'
              f'{row["iters"]:>7d}{row["matvec_err"]:>12.2e}  {row["eigval"]:.10f}')

    # Disclosure footnotes (I2, I3): a silently skipped correctness check or a silently
    # reduced-precision setup is exactly the failure mode this benchmark exists to avoid.
    for row in results:
        if row.get('status') != 'ok':
            continue
        if row.get('setup_precision') == 'f32':
            print(f'NOTE [{row["arm"]}]: setup ran at reduced precision (float32, Metal has no '
                  'float64) -- this arm solved a measurably different problem from the '
                  'f64-setup arms and its eigval/matvec_err are not strictly comparable to '
                  'theirs.')
        if row.get('brute_force_note'):
            print(f'NOTE [{row["arm"]}]: {row["brute_force_note"]}')


def main():
    options = parse_args()
    if options.all:
        report(run_all(options), options.json)
    else:
        result = run_arm(options.arm, options)
        if options.json:
            print(json.dumps(result, indent=2))
        else:
            report([result], False)


if __name__ == '__main__':
    main()
