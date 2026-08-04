"""Localize the large ``compile_s`` on the MLX *CPU* arms of ``bench_mlx.py``.

Observed at ``--num-qubits 10 --matvec chunked --chunk 16 --compile-body`` (N=1001, J=100):

    mlx-gpu-f32   compile_s =  0.41 s
    mlx-cpu-f32   compile_s = 10.78 s     26x the GPU
    mlx-cpu-f64   compile_s = 20.39 s     50x the GPU, and 1.9x the cpu-f32

``compile_s`` is the FIRST timed call, so it bundles three costs that this script separates:

1. Python-level MLX graph construction (``chunk_body`` unrolls ``compile_chunk`` copies of
   ``iter_body``, each with ~63 chunked-matvec ops plus the eigenpair kernels).
2. ``mx.compile``'s own tracing/fusion pass.
3. Actually executing 100 iterations of real work.

The f64-vs-f32 ratio is the reason this is worth isolating. The traced graph STRUCTURE is
identical between those two arms -- same ops, same chunk, same N -- so a 2x cost difference
cannot come from graph construction (cost 1) or from tracing (cost 2), both of which are
structural. It must come from per-element work or memory traffic. That points at cost 3, or at
``mx.compile`` performing a data-dependent compilation pass on the CPU backend that the Metal
backend either skips or does far more cheaply.

REQUIRES A REAL METAL DEVICE -- MLX cannot initialize without one, even to target ``mx.cpu``.

    uv run --extra mlx --extra qiskit python examples/diagnose_mlx_compile_s.py

Send back the whole output. The interesting comparison is column-wise: whether
``construct_only`` (no eval) is flat across devices while ``first_call`` diverges, and whether
``compile_chunk`` scaling is linear (graph size) or flat (a fixed per-compile cost).
"""

import os
import sys
import time

import jax
import mlx.core as mx
import numpy as np

jax.config.update("jax_enable_x64", True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _bench_common import apply_h_xz_chunked, build_solver_inputs, generate_problem

from rqutils.ground_locg_mlx import apply_h_xz_mlx_chunked, ground_locg_mlx

ps, cs, states = generate_problem(10, 100, 4000, seed=1)
inputs = build_solver_inputs(ps, cs, states)
print(f"problem: N={inputs.subspace_dim}  J={inputs.xsources.shape[0]}  (the failing-gate size)")
print(f"apply_h_xz_chunked available for reference: {apply_h_xz_chunked is not None}")
print()


def prepare(device, dtype):
    mx.set_default_device(device)
    xs = mx.array(inputs.xsources, mx.int32)
    dg = mx.array(np.asarray(inputs.diagonals, dtype=np.float64), dtype)
    v0 = mx.array(np.asarray(inputs.vinit, dtype=np.float64), dtype)
    mx.eval(xs, dg, v0)
    return xs, dg, v0


def matvec(vec, xsources, diagonals):
    return apply_h_xz_mlx_chunked(vec, xsources, diagonals, chunk=16)


DEVICES = [(mx.cpu, "cpu"), (mx.gpu, "gpu")]
DTYPES = [(mx.float32, "f32"), (mx.float64, "f64")]

print("=" * 78)
print("SECTION 1 -- separate graph CONSTRUCTION from EXECUTION")
print("=" * 78)
print("construct_only builds the op graph and never evals it (MLX is lazy), so it isolates")
print("Python-level construction. first_call adds the eval. If construction is flat across")
print("devices but first_call is not, the cost is in compile/execute, not in Python.")
print()
print(f"{'device':6s} {'dtype':6s} {'compile':>8s} {'construct_only':>15s} {'first_call':>12s}")
for device, dname in DEVICES:
    for dtype, tname in DTYPES:
        if device is mx.gpu and dtype is mx.float64:
            print(f"{dname:6s} {tname:6s} {'--':>8s} {'(Metal has no float64)':>28s}")
            continue
        for compile_body in (False, True):
            try:
                xs, dg, v0 = prepare(device, dtype)
                start = time.perf_counter()
                out = ground_locg_mlx(
                    matvec, v0, args=(xs, dg), maxiter=100, tol=0.0, compile_body=compile_body
                )
                construct = time.perf_counter() - start
                mx.eval(out[1])
                first = time.perf_counter() - start
                print(f"{dname:6s} {tname:6s} {compile_body!s:>8s} {construct:15.4f} {first:12.4f}")
            except Exception as exc:  # noqa: BLE001 (probing which configs fail is the point)
                print(
                    f"{dname:6s} {tname:6s} {compile_body!s:>8s}  ERROR {type(exc).__name__}: {exc}"
                )

print()
print("=" * 78)
print("SECTION 2 -- does the cost scale with compile_chunk (graph size) or is it fixed?")
print("=" * 78)
print("chunk_body unrolls compile_chunk copies of iter_body. If the first-call cost is")
print("proportional to compile_chunk, it is graph size. If it is roughly flat, it is a fixed")
print("per-mx.compile overhead. maxiter is held at 100 so the WORK is identical throughout.")
print()
print(f"{'device':6s} {'dtype':6s} {'chunk':>6s} {'first_call':>12s} {'per_chunk':>11s}")
for device, dname in DEVICES:
    for dtype, tname in DTYPES:
        if device is mx.gpu and dtype is mx.float64:
            continue
        for chunk in (1, 5, 10, 20):
            try:
                xs, dg, v0 = prepare(device, dtype)
                start = time.perf_counter()
                out = ground_locg_mlx(
                    matvec,
                    v0,
                    args=(xs, dg),
                    maxiter=100,
                    compile_body=True,
                    compile_chunk=chunk,
                )
                mx.eval(out[1])
                elapsed = time.perf_counter() - start
                print(f"{dname:6s} {tname:6s} {chunk:6d} {elapsed:12.4f} {elapsed / chunk:11.4f}")
            except Exception as exc:  # noqa: BLE001
                print(f"{dname:6s} {tname:6s} {chunk:6d}  ERROR {type(exc).__name__}: {exc}")

print()
print("=" * 78)
print("SECTION 3 -- is it mx.compile at all, or the chunked matvec?")
print("=" * 78)
print("Compile a bare chunked matvec (no solver) and time the first vs second call. This")
print("removes the eigenpair kernels and the unrolled loop entirely.")
print()
for device, dname in DEVICES:
    for dtype, tname in DTYPES:
        if device is mx.gpu and dtype is mx.float64:
            continue
        try:
            xs, dg, v0 = prepare(device, dtype)
            compiled = mx.compile(lambda v, x=xs, d=dg: matvec(v, x, d))
            start = time.perf_counter()
            mx.eval(compiled(v0))
            first = time.perf_counter() - start
            start = time.perf_counter()
            mx.eval(compiled(v0))
            second = time.perf_counter() - start
            print(
                f"{dname:6s} {tname:6s} bare chunked matvec: first={first:.4f}s "
                f"second={second:.6f}s  ratio={first / max(second, 1e-9):.0f}x"
            )
        except Exception as exc:  # noqa: BLE001
            print(f"{dname:6s} {tname:6s} ERROR {type(exc).__name__}: {exc}")

print()
print("Interpretation guide:")
print("  * construct_only flat, first_call diverging  -> mx.compile or execution, not Python")
print("  * first_call linear in compile_chunk         -> graph size drives it (expected)")
print("  * first_call flat in compile_chunk           -> fixed per-compile cost")
print("  * bare matvec already shows the cpu/gpu gap  -> it is mx.compile on the CPU backend,")
print("    independent of anything this port does")
