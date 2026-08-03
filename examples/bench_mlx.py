"""Benchmark the SQD eigensolver loop across JAX and MLX, CPU and GPU.

Only the solver loop is compared. Setup (uniquification, X-source lookup, diagonal
composition) always runs in JAX on CPU and is not timed, so every arm consumes identical
arrays -- see docs/superpowers/specs/2026-08-03-mlx-sqd-poc-design.md.

JAX's platform and x64 flag are process-global and must be set before importing jax, so each
JAX arm needs its own process. --all re-executes this script once per arm and collates.

.. code-block:: sh

    uv run python examples/bench_mlx.py --arm mlx-gpu-f32
    uv run python examples/bench_mlx.py --all
    uv run python examples/bench_mlx.py --all --json > results.json

Two metrics are reported per arm: per-iteration cost at a fixed iteration count (identical
work per arm, so a clean speed comparison) and time-to-convergence with its iteration count
(what production actually pays). Reporting both makes it visible when fp32 is faster per
iteration but needs more iterations.
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
                        help='Skip the 2^n reference (needed above ~n=14).')
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
                               brute_force_reference, timeit)

    if framework == 'jax' and device == 'metal':
        try:
            backend = jax.default_backend()
        except RuntimeError as exc:
            return {'arm': arm, 'status': 'skipped',
                    'reason': f'jax metal backend unavailable: {exc}'}
        if backend != 'metal':
            return {'arm': arm, 'status': 'skipped',
                    'reason': f'jax backend is {backend}, not metal'}

    pauli_strings, coeffs, states = generate_problem(
        options.num_qubits, options.num_paulis, options.num_states, options.seed
    )
    inputs = build_solver_inputs(pauli_strings, coeffs, states)

    if options.self_test_break_gate:
        # Corrupt the diagonals so the solver cannot reach the reference eigenvalue.
        inputs.diagonals = inputs.diagonals * 2.5 + 1.0

    matrix, reference = dense_reference(inputs)
    if not options.skip_brute_force:
        brute = brute_force_reference(pauli_strings, coeffs, states)
        if abs(reference - brute) > 1e-9 * max(1., abs(brute)):
            raise SystemExit(f'gate failed: dense reference {reference} disagrees with '
                             f'brute force {brute} -- the setup chain is wrong')

    rtol = RTOL[precision]
    if framework == 'jax':
        result = _time_jax(arm, inputs, precision, options)
    else:
        result = _time_mlx(arm, inputs, device, precision, options)

    if abs(result['eigval'] - reference) > rtol * max(1., abs(reference)):
        raise SystemExit(f'gate failed for {arm}: eigenvalue {result["eigval"]} differs from '
                         f'reference {reference} by more than rtol={rtol}')

    result['reference'] = reference
    result['status'] = 'ok'
    return result


def _time_jax(arm, inputs, precision, options):
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

    # Gate: the ported matvec and the original must agree on the same input.
    matvec_out = np.asarray(apply_h_xz_cached(vinit, xsources, diagonals), dtype=np.float64)
    matrix, _ = _reference_matrix(inputs)
    matvec_err = float(np.abs(matvec_out - matrix @ inputs.vinit).max())

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


def _time_mlx(arm, inputs, device, precision, options):
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

    matvec_out = np.asarray(apply_h_xz_mlx(vinit, xsources, diagonals), dtype=np.float64)
    matrix, _ = _reference_matrix(inputs)
    matvec_err = float(np.abs(matvec_out - matrix @ inputs.vinit).max())

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


def _reference_matrix(inputs):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from _bench_common import dense_reference
    return dense_reference(inputs)


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

    header = (f'{"arm":<15}{"compile_s":>10}{"fixed_s":>10}{"per_it_ms":>11}'
              f'{"solve_s":>10}{"iters":>7}  eigval')
    print(header)
    print('-' * len(header))
    for row in results:
        if row.get('status') != 'ok':
            print(f'{row["arm"]:<15}{row.get("status", "?"):>10}  {row.get("reason", "")}')
            continue
        print(f'{row["arm"]:<15}{row["compile_s"]:>10.4f}{row["fixed_s"]:>10.4f}'
              f'{row["per_it_ms"]:>11.3f}{row["solve_s"]:>10.4f}{row["iters"]:>7d}'
              f'  {row["eigval"]:.10f}')


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
