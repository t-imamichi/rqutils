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

ARMS = ("jax-cpu-f64", "jax-cpu-f32", "jax-metal-f32", "mlx-cpu-f64", "mlx-cpu-f32", "mlx-gpu-f32")

# Relative tolerance for the correctness gate, by precision.
RTOL = {"f64": 1e-9, "f32": 1e-4}

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
MATVEC_ATOL = {"f64": 1e-9, "f32": 1e-4}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--arm", choices=ARMS, help="Single arm to run.")
    parser.add_argument(
        "--all", action="store_true", help="Run every arm, one subprocess each, and collate."
    )
    parser.add_argument("--num-qubits", type=int, default=14)
    parser.add_argument("--num-paulis", type=int, default=100)
    parser.add_argument("--num-states", type=int, default=4000)
    parser.add_argument("--repeat", type=int, default=3, help="Timed iterations after warmup.")
    parser.add_argument(
        "--fixed-iters",
        type=int,
        default=100,
        help="Iteration count for the fixed-work measurement.",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--matvec",
        choices=("loop", "chunked", "metal"),
        default="loop",
        help="Matvec kernel, applied to BOTH jax and mlx arms so the comparison "
        "stays about the solver loop rather than the matvec (Optimization "
        '1). "loop" (default) is the original group-at-a-time gather -- '
        "existing measured results are only reproducible with this default. "
        '"chunked" gathers --chunk X-groups per flat take, cutting op count '
        "roughly 3*J -> 3*ceil(J/chunk).",
    )
    parser.add_argument(
        "--chunk",
        type=int,
        default=16,
        help="Chunk size for --matvec chunked (default 16 -> ~14.3x fewer ops "
        "at J=100, temporary bounded to chunk*N). Ignored for --matvec loop.",
    )
    parser.add_argument(
        "--compile-body",
        action="store_true",
        help="MLX arms only (Optimization 2): wrap the LOBPCG iteration body in "
        "mx.compile so its ~1260 ops/iteration are traced once instead of "
        "reconstructed every call. A no-op for jax arms -- reported as a "
        "note rather than an error, since JAX already amortizes graph "
        "construction via jax.jit/lax.while_loop (see compile_s).",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of a table.")
    parser.add_argument(
        "--skip-brute-force",
        action="store_true",
        help="Explicitly skip the 2^n reference at any --num-qubits. It is "
        f"auto-skipped above --num-qubits {BRUTE_FORCE_MAX_QUBITS} "
        "regardless of this flag; either way the skip is reported.",
    )
    parser.add_argument(
        "--self-test-break-gate", action="store_true", help=argparse.SUPPRESS
    )  # corrupts the problem to prove the gate bites
    options = parser.parse_args(argv)
    if not options.arm and not options.all:
        parser.error("specify --arm or --all")
    return options


def run_arm(arm, options):
    """Run one arm in this process. Returns a result dict."""
    framework, device, precision = arm.split("-")

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
    if framework == "jax":
        os.environ["JAX_PLATFORMS"] = "cpu" if device == "cpu" else device
    import jax

    jax.config.update("jax_enable_x64", device != "metal")

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import time

    import numpy as np
    from _bench_common import (
        brute_force_reference,
        build_solver_inputs,
        dense_reference,
        generate_problem,
    )

    if framework == "jax" and device == "metal":
        try:
            backend = jax.default_backend()
        except RuntimeError as exc:
            return {
                "arm": arm,
                "status": "skipped",
                "reason": f"jax metal backend unavailable: {exc}",
            }
        if backend != "metal":
            return {
                "arm": arm,
                "status": "skipped",
                "reason": f"jax backend is {backend}, not metal",
            }

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
    setup_precision = "f32" if device == "metal" else "f64"

    matrix, reference = dense_reference(inputs)
    brute_force_note = None
    skip_brute_force = options.skip_brute_force
    if not skip_brute_force and options.num_qubits > BRUTE_FORCE_MAX_QUBITS:
        skip_brute_force = True
        brute_force_note = (
            f"brute-force 2^n cross-check auto-skipped: --num-qubits {options.num_qubits} > "
            f"{BRUTE_FORCE_MAX_QUBITS} (the 2^n dense matrix becomes too large/slow -- see "
            "BRUTE_FORCE_MAX_QUBITS in this module). The dense-from-solver-inputs gate "
            "(dense_reference) still ran."
        )
        print(f"NOTE: {brute_force_note}", file=sys.stderr)
    elif skip_brute_force:
        brute_force_note = "brute-force 2^n cross-check skipped: --skip-brute-force was passed."
        print(f"NOTE: {brute_force_note}", file=sys.stderr)

    if not skip_brute_force:
        brute = brute_force_reference(pauli_strings, coeffs, states)
        # The threshold scales with setup precision, not solve precision: a float32 setup
        # (jax-metal-f32 only) accumulates rounding error in the setup chain itself (measured
        # |H_metal - H_cpu|_inf ~= 2.79e-08 in the final review), which a fixed 1e-9 threshold
        # sits right at the edge of -- passing or failing depending on n by accident. Scaling by
        # RTOL[setup_precision] makes the gate's behaviour depend on a disclosed, principled
        # quantity instead of luck.
        brute_force_rtol = RTOL[setup_precision]
        if abs(reference - brute) > brute_force_rtol * max(1.0, abs(brute)):
            raise SystemExit(
                f"gate failed: dense reference {reference} disagrees with "
                f"brute force {brute} by more than rtol={brute_force_rtol} "
                f"(setup_precision={setup_precision}) -- the setup chain is wrong"
            )

    compile_body_note = None
    if framework == "jax" and options.compile_body:
        # --compile-body is Optimization 2, an MLX-only concept (mx.compile over the LOBPCG
        # iteration body). JAX already amortizes Python-level graph construction via jax.jit
        # over the whole solver loop (jax.lax.while_loop/scan) -- that is exactly what compile_s
        # measures for jax arms -- so there is nothing for this flag to do here. Reported as a
        # note, not an error: per the WIRING requirements, a jax arm must not fail just because
        # --compile-body was passed (e.g. under --all, which passes the same flags to every arm).
        compile_body_note = (
            f"--compile-body is MLX-only (Optimization 2, mx.compile over the LOBPCG iteration "
            f"body); ignored for {arm} because JAX already amortizes graph construction via "
            "jax.jit (see compile_s)."
        )
        print(f"NOTE: {compile_body_note}", file=sys.stderr)

    rtol = RTOL[precision]
    if framework == "jax":
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
    if not np.isfinite(result["matvec_err"]):
        raise SystemExit(
            f"gate failed for {arm}: matvec error is non-finite "
            f"({result['matvec_err']!r}) after {result['iters']} solver "
            "iterations -- solver did not converge cleanly"
        )

    if not np.isfinite(result["eigval"]):
        raise SystemExit(
            f"gate failed for {arm}: eigenvalue is non-finite "
            f"({result['eigval']!r}) after {result['iters']} solver iterations "
            "-- solver did not converge cleanly"
        )

    matvec_atol = MATVEC_ATOL[precision]
    if result["matvec_err"] > matvec_atol:
        raise SystemExit(
            f"gate failed for {arm}: matvec error {result['matvec_err']} exceeds atol={matvec_atol}"
        )

    if abs(result["eigval"] - reference) > rtol * max(1.0, abs(reference)):
        raise SystemExit(
            f"gate failed for {arm}: eigenvalue {result['eigval']} differs from "
            f"reference {reference} by more than rtol={rtol}"
        )

    result["reference"] = reference
    result["setup_precision"] = setup_precision
    result["brute_force_note"] = brute_force_note
    result["setup_s"] = setup_s
    result["status"] = "ok"
    # Self-describing options: a saved per_it_ms/eigval is useless for comparison unless the
    # matvec/compile settings that produced it travel with it.
    result["matvec"] = options.matvec
    result["chunk"] = options.chunk if options.matvec == "chunked" else None
    result["compile_body"] = bool(options.compile_body) and framework == "mlx"
    result["compile_body_note"] = compile_body_note
    return result


def _time_jax(arm, inputs, precision, options, matrix, matvec_probe):
    import functools

    import jax
    import jax.numpy as jnp
    import numpy as np

    from rqutils.ground_locg import ground_locg
    from rqutils.sqd import apply_h_xz_cached

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from _bench_common import apply_h_xz_chunked, timeit

    dtype = np.float64 if precision == "f64" else np.float32
    xsources = jnp.asarray(inputs.xsources)
    diagonals = jnp.asarray(inputs.diagonals, dtype=dtype)
    vinit = jnp.asarray(inputs.vinit, dtype=dtype)

    # --matvec selects the kernel used by the solver loop, applied identically to the jax and
    # mlx arms (Optimization 1) -- see apply_h_xz_chunked's docstring in _bench_common.py for
    # the op-count/memory tradeoff. "loop" (default) is apply_h_xz_cached unchanged, so the
    # existing measured results stay reproducible.
    if options.matvec == "metal":
        # Fail loudly rather than silently substituting a different kernel: a "metal" row that
        # actually timed apply_h_xz_cached would be a fabricated comparison.
        raise SystemExit(
            f"{arm}: --matvec metal is an MLX-only custom Metal kernel and has no JAX "
            "equivalent. Use --matvec loop or --matvec chunked for the jax arms."
        )
    if options.matvec == "chunked":
        matvec_fn = functools.partial(apply_h_xz_chunked, chunk=options.chunk)
    else:
        matvec_fn = apply_h_xz_cached

    # Gate: the ported matvec and the original must agree on the same input, probed with a
    # random vector (not vinit) -- see the matvec_probe comment in run_arm.
    probe = jnp.asarray(matvec_probe, dtype=dtype)
    matvec_out = np.asarray(matvec_fn(probe, xsources, diagonals), dtype=np.float64)
    matvec_err = float(np.abs(matvec_out - matrix @ matvec_probe).max())

    def fixed():
        return ground_locg(
            matvec_fn, vinit, args=(xsources, diagonals), maxiter=options.fixed_iters, tol=0.0
        )

    compile_s, fixed_s = timeit(fixed, options.repeat, jax.block_until_ready)

    def solve():
        return ground_locg(matvec_fn, vinit, args=(xsources, diagonals))

    _, solve_s = timeit(solve, options.repeat, jax.block_until_ready)
    eigval, _, iters, _ = solve()

    return {
        "arm": arm,
        "compile_s": compile_s,
        "fixed_s": fixed_s,
        "per_it_ms": fixed_s / options.fixed_iters * 1e3,
        "solve_s": solve_s,
        "iters": int(iters),
        "eigval": float(eigval),
        "matvec_err": matvec_err,
    }


def _time_mlx(arm, inputs, device, precision, options, matrix, matvec_probe):
    import functools

    import mlx.core as mx
    import numpy as np

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from _bench_common import timeit

    from rqutils.ground_locg_mlx import (
        apply_h_xz_mlx,
        apply_h_xz_mlx_chunked,
        apply_h_xz_mlx_metal,
        ground_locg_mlx,
    )

    # --matvec selects the kernel, mirroring _time_jax exactly (Optimization 1) -- see
    # apply_h_xz_mlx_chunked's docstring for the op-count/memory tradeoff.
    if options.matvec == "metal":
        # Optimization 3: one fused custom Metal kernel instead of a sequence of MLX ops.
        # Metal has no float64, so this path is f32-only; refuse rather than silently running
        # a different kernel than the one the row claims to be timing.
        if precision != "f32":
            raise SystemExit(
                f"{arm}: --matvec metal requires float32 (Metal has no float64). Use an f32 "
                "arm, or --matvec chunked for the f64 arms."
            )
        matvec_fn = apply_h_xz_mlx_metal
    elif options.matvec == "chunked":
        matvec_fn = functools.partial(apply_h_xz_mlx_chunked, chunk=options.chunk)
    else:
        matvec_fn = apply_h_xz_mlx

    mx.set_default_device(mx.cpu if device == "cpu" else mx.gpu)
    dtype = mx.float64 if precision == "f64" else mx.float32
    # Pass dtype at CONSTRUCTION, never construct-then-cast: MLX's docs state that "NumPy
    # arrays with type float64 will be default converted to MLX arrays with type float32"
    # (https://ml-explore.github.io/mlx/build/html/usage/numpy.html). mx.array(x).astype(dtype)
    # truncates to float32 in the first call and only then casts back up in the second -- the
    # low bits are already gone, so an "f64" arm silently ends up doing float64 arithmetic on
    # float32-precision data (this is exactly what happened before this fix: mlx-cpu-f64 showed
    # the f32 matvec-error floor of 2.07e-08, not the ~1e-15 a genuine f64 matvec gives).
    # mx.array(x, dtype) builds at the target precision directly, with no lossy intermediate.
    # xsources is int32 (see _bench_common.py), not floating point: MLX's default integer type
    # is also int32 and int64 is a full native dtype (unlike float64, it is not narrowed on
    # numpy ingest), so this array was never at risk -- dtype is still passed explicitly here
    # for symmetry and so the _time_mlx dtype guard below has something to check.
    xsources = mx.array(inputs.xsources, mx.int32)
    diagonals = mx.array(inputs.diagonals, dtype)
    vinit = mx.array(inputs.vinit, dtype)

    # Guard against a silent precision downgrade recurring: without this, a future
    # construct-then-cast regression (or any other dtype bug) would produce an "f64" arm that
    # quietly runs at f32 precision, visible only if someone reads the eigenvalue/matvec_err
    # closely enough to notice the f32 error floor -- exactly how this bug hid before. Assert
    # actual .dtype against the requested dtype right after construction, for every array.
    for name, arr, expected in (
        ("xsources", xsources, mx.int32),
        ("diagonals", diagonals, dtype),
        ("vinit", vinit, dtype),
    ):
        assert arr.dtype == expected, (
            f"{arm}: {name} was constructed with dtype {arr.dtype}, expected {expected} -- "
            "a construct-then-cast (mx.array(x).astype(...)) or similar ingest bug silently "
            "downgraded precision. Pass dtype at construction: mx.array(x, dtype)."
        )

    # Same random probe as _time_jax (identical values, only the array framework differs), so
    # matvec_err is comparable across the two paths -- see the matvec_probe comment in run_arm.
    probe = mx.array(matvec_probe, dtype)
    assert probe.dtype == dtype, (
        f"{arm}: matvec probe was constructed with dtype {probe.dtype}, expected {dtype} -- "
        "a construct-then-cast ingest bug silently downgraded precision."
    )
    matvec_out = np.asarray(matvec_fn(probe, xsources, diagonals), dtype=np.float64)
    matvec_err = float(np.abs(matvec_out - matrix @ matvec_probe).max())

    def sync(result):
        # MLX is lazy: without this we would time graph construction, not computation.
        mx.eval(result[1])
        return result

    # --compile-body is Optimization 2: mx.compile over the LOBPCG iteration body. Off by
    # default (compile_body=False), which reproduces this function's behaviour exactly as it
    # was before the parameter existed -- see ground_locg_mlx's docstring.
    def fixed():
        return ground_locg_mlx(
            matvec_fn,
            vinit,
            args=(xsources, diagonals),
            maxiter=options.fixed_iters,
            tol=0.0,
            compile_body=options.compile_body,
        )

    compile_s, fixed_s = timeit(fixed, options.repeat, sync)

    def solve():
        return ground_locg_mlx(
            matvec_fn, vinit, args=(xsources, diagonals), compile_body=options.compile_body
        )

    _, solve_s = timeit(solve, options.repeat, sync)
    eigval, _, iters, _ = solve()

    return {
        "arm": arm,
        "compile_s": compile_s,
        "fixed_s": fixed_s,
        "per_it_ms": fixed_s / options.fixed_iters * 1e3,
        "solve_s": solve_s,
        "iters": int(iters),
        "eigval": float(eigval),
        "matvec_err": matvec_err,
    }


def run_all(options):
    """Run every arm in its own subprocess and collate the results."""
    results = []
    for arm in ARMS:
        argv = [
            sys.executable,
            os.path.abspath(__file__),
            "--arm",
            arm,
            "--json",
            "--num-qubits",
            str(options.num_qubits),
            "--num-paulis",
            str(options.num_paulis),
            "--num-states",
            str(options.num_states),
            "--repeat",
            str(options.repeat),
            "--fixed-iters",
            str(options.fixed_iters),
            "--seed",
            str(options.seed),
            "--matvec",
            options.matvec,
            "--chunk",
            str(options.chunk),
        ]
        if options.skip_brute_force:
            argv.append("--skip-brute-force")
        if options.compile_body:
            argv.append("--compile-body")
        # check=False: a failing arm is recorded as a "failed" result below rather than aborting
        # the whole sweep, so the remaining arms still get benchmarked.
        proc = subprocess.run(argv, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            results.append(
                {
                    "arm": arm,
                    "status": "failed",
                    "reason": (proc.stderr or proc.stdout).strip().split("\n")[-1],
                }
            )
            continue
        try:
            results.append(json.loads(proc.stdout))
        except json.JSONDecodeError:
            results.append(
                {
                    "arm": arm,
                    "status": "failed",
                    "reason": f"unparseable output: {proc.stdout[:200]}",
                }
            )
    return results


def report(results, as_json):
    if as_json:
        print(json.dumps({"results": results}, indent=2))
        return

    header = (
        f"{'arm':<15}{'setup_s':>9}{'compile_s':>10}{'fixed_s':>10}{'per_it_ms':>11}"
        f"{'solve_s':>10}{'iters':>7}{'matvec_err':>12}  {'eigval':<15}  options"
    )
    print(header)
    print("-" * len(header))
    for row in results:
        if row.get("status") != "ok":
            print(f"{row['arm']:<15}{row.get('status', '?'):>10}  {row.get('reason', '')}")
            continue
        # Self-describing options string: a saved per_it_ms/eigval is useless for comparison
        # unless the matvec/compile settings that produced it travel with it (WIRING requirement).
        opt = f"matvec={row.get('matvec', 'loop')}"
        if row.get("matvec") == "chunked":
            opt += f"(chunk={row.get('chunk')})"
        if row.get("compile_body"):
            opt += " compile_body"
        print(
            f"{row['arm']:<15}{row['setup_s']:>9.4f}{row['compile_s']:>10.4f}"
            f"{row['fixed_s']:>10.4f}{row['per_it_ms']:>11.3f}{row['solve_s']:>10.4f}"
            f"{row['iters']:>7d}{row['matvec_err']:>12.2e}  {row['eigval']:<15.10f}  {opt}"
        )

    # Disclosure footnotes (I2, I3): a silently skipped correctness check or a silently
    # reduced-precision setup is exactly the failure mode this benchmark exists to avoid.
    for row in results:
        if row.get("status") != "ok":
            continue
        if row.get("setup_precision") == "f32":
            print(
                f"NOTE [{row['arm']}]: setup ran at reduced precision (float32, Metal has no "
                "float64) -- this arm solved a measurably different problem from the "
                "f64-setup arms and its eigval/matvec_err are not strictly comparable to "
                "theirs."
            )
        if row.get("brute_force_note"):
            print(f"NOTE [{row['arm']}]: {row['brute_force_note']}")
        if row.get("compile_body_note"):
            print(f"NOTE [{row['arm']}]: {row['compile_body_note']}")


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


if __name__ == "__main__":
    main()
