"""Localize the large ``compile_s`` on the MLX *CPU* arms of ``bench_mlx.py``.

Observed in a real six-arm run at ``--num-qubits 10 --matvec chunked --chunk 16
--compile-body`` (N=1001):

    mlx-gpu-f32   compile_s =  0.41 s
    mlx-cpu-f32   compile_s = 10.78 s     26x the GPU
    mlx-cpu-f64   compile_s = 20.39 s     50x the GPU, and 1.9x the cpu-f32

**A first version of this script failed to reproduce any of that** -- it measured 0.44 s for
cpu-f64 with compilation, 46x cheaper than the benchmark, and found no CPU/GPU gap at all for a
bare compiled matvec (ratio 1x). So the cost is NOT ``mx.compile`` on the CPU backend, and it is
not the chunked matvec: both of those hypotheses are dead. It is something the benchmark does
that a direct call does not.

Two candidates remain, and this script tests them:

1. **Device-switch cost.** ``bench_mlx.py`` calls ``mx.set_default_device`` once and then builds
   arrays; a fresh process targeting only one device may never pay whatever the switch costs.
   Section 1 times repeated solves with and without an intervening device switch.
2. **State accumulated by the benchmark's own setup.** Before the first timed call, ``run_arm``
   has already built the JAX-side setup, a dense reference, and (at n<=12) a 2^n x 2^n
   brute-force Kronecker cross-check. Section 2 replays that ordering.

Also note a flaw in the first version, corrected here: ``ground_locg_mlx`` ends with
``return float(theta)``, which forces a device sync, so its "construct_only vs first_call"
columns were necessarily identical and separated nothing. Those first-call numbers already
included full execution -- which is what makes the 0.44 s result meaningful rather than an
artifact of laziness.

REQUIRES A REAL METAL DEVICE -- MLX cannot initialize without one, even to target ``mx.cpu``.

    uv run --extra mlx --extra qiskit python examples/diagnose_mlx_compile_s.py

Send back the whole output.
"""

import os
import sys
import time

import jax
import mlx.core as mx
import numpy as np

jax.config.update("jax_enable_x64", True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _bench_common import build_solver_inputs, dense_reference, generate_problem

from rqutils.ground_locg_mlx import apply_h_xz_mlx_chunked, ground_locg_mlx

ps, cs, states = generate_problem(10, 100, 4000, seed=1)
inputs = build_solver_inputs(ps, cs, states)
print(f"problem: N={inputs.subspace_dim}  J={inputs.xsources.shape[0]}")
print("benchmark reported compile_s: gpu-f32 0.41s, cpu-f32 10.78s, cpu-f64 20.39s")
print()


def matvec(vec, xsources, diagonals):
    return apply_h_xz_mlx_chunked(vec, xsources, diagonals, chunk=16)


def build(device, dtype):
    """Set the device and build the arrays, exactly as _time_mlx does."""
    mx.set_default_device(device)
    xs = mx.array(inputs.xsources, mx.int32)
    dg = mx.array(np.asarray(inputs.diagonals, dtype=np.float64), dtype)
    v0 = mx.array(np.asarray(inputs.vinit, dtype=np.float64), dtype)
    mx.eval(xs, dg, v0)
    return xs, dg, v0


def timed_solve(xs, dg, v0, **kwargs):
    start = time.perf_counter()
    out = ground_locg_mlx(matvec, v0, args=(xs, dg), **kwargs)
    mx.eval(out[1])
    return time.perf_counter() - start, out


CONFIGS = [(mx.cpu, "cpu", mx.float32, "f32"), (mx.cpu, "cpu", mx.float64, "f64")]
CONFIGS += [(mx.gpu, "gpu", mx.float32, "f32")]

print("=" * 78)
print("SECTION 1 -- is the cost paid once per process, or once per device switch?")
print("=" * 78)
print("The benchmark's compile_s is the FIRST call after mx.set_default_device. If call 1 is")
print("expensive and call 2 is cheap, it is a one-time cost. If re-switching the device makes")
print("call 3 expensive again, the switch itself is what costs.")
print()
print(f"{'device':6s} {'dtype':6s} {'call1':>9s} {'call2':>9s} {'after switch':>13s}")
for device, dname, dtype, tname in CONFIGS:
    try:
        xs, dg, v0 = build(device, dtype)
        t1, _ = timed_solve(xs, dg, v0, maxiter=100, tol=0.0, compile_body=True)
        t2, _ = timed_solve(xs, dg, v0, maxiter=100, tol=0.0, compile_body=True)
        # Switch away and back, then rebuild -- mirroring a fresh arm.
        other = mx.gpu if device is mx.cpu else mx.cpu
        mx.set_default_device(other)
        mx.eval(mx.array([1.0], mx.float32))
        xs, dg, v0 = build(device, dtype)
        t3, _ = timed_solve(xs, dg, v0, maxiter=100, tol=0.0, compile_body=True)
        print(f"{dname:6s} {tname:6s} {t1:9.4f} {t2:9.4f} {t3:13.4f}")
    except Exception as exc:  # noqa: BLE001
        print(f"{dname:6s} {tname:6s} ERROR {type(exc).__name__}: {str(exc)[:50]}")

print()
print("=" * 78)
print("SECTION 2 -- replay the benchmark's setup ordering before the first solve")
print("=" * 78)
print("run_arm builds the dense reference and, at n<=12, a 2^n x 2^n brute-force Kronecker")
print("cross-check BEFORE the first timed call. If that is what inflates compile_s, doing it")
print("here first should reproduce the 10-20 s figure.")
print()
start = time.perf_counter()
matrix, reference = dense_reference(inputs)
print(f"dense_reference built in {time.perf_counter() - start:.4f}s  (ref={reference:.10f})")
print()
print(f"{'device':6s} {'dtype':6s} {'first solve after setup':>24s}")
for device, dname, dtype, tname in CONFIGS:
    try:
        xs, dg, v0 = build(device, dtype)
        t, _ = timed_solve(xs, dg, v0, maxiter=100, tol=0.0, compile_body=True)
        print(f"{dname:6s} {tname:6s} {t:24.4f}")
    except Exception as exc:  # noqa: BLE001
        print(f"{dname:6s} {tname:6s} ERROR {type(exc).__name__}: {str(exc)[:50]}")

print()
print("=" * 78)
print("SECTION 3 -- run the real benchmark path in-process, three arms in one process")
print("=" * 78)
print("bench_mlx.py --all spawns one subprocess per arm, so each arm's compile_s is a")
print("first-call-in-a-fresh-process number. This runs run_arm directly for three arms in ONE")
print("process, which is the one thing a --all run never does. If compile_s is only large for")
print("arms after the first, the cost belongs to switching devices within a live process.")
print()
try:
    from bench_mlx import parse_args, run_arm

    for arm in ("mlx-gpu-f32", "mlx-cpu-f32", "mlx-cpu-f64"):
        options = parse_args(
            [
                "--arm",
                arm,
                "--num-qubits",
                "10",
                "--matvec",
                "chunked",
                "--chunk",
                "16",
                "--compile-body",
                "--repeat",
                "1",
            ]
        )
        try:
            result = run_arm(arm, options)
            print(
                f"  {arm:12s} compile_s={result['compile_s']:8.4f}  "
                f"per_it_ms={result['per_it_ms']:7.3f}  iters={result['iters']}"
            )
        except SystemExit as exc:
            print(f"  {arm:12s} gate/exit: {exc}")
        except Exception as exc:  # noqa: BLE001
            print(f"  {arm:12s} ERROR {type(exc).__name__}: {str(exc)[:60]}")
except Exception as exc:  # noqa: BLE001
    print(f"  could not import bench_mlx.run_arm: {type(exc).__name__}: {exc}")

print()
print("Interpretation guide:")
print("  * call1 >> call2, and 'after switch' >> call2  -> the device switch costs")
print("  * call1 ~ call2 ~ 0.4s everywhere             -> not reproducible in isolation;")
print("    Section 3 is then the deciding test")
print("  * Section 3 shows arm 1 cheap, arms 2-3 costly -> per-process device-switch cost,")
print("    which a --all run pays because each subprocess switches once. Then the benchmark's")
print("    compile_s is measuring MLX device setup, not this port, and should be relabelled.")
