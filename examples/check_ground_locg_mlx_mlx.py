"""Verify ground_locg_mlx under real MLX, on both devices and both precisions.

Must be run by the user: mlx.core loads a Metal device even for mx.cpu arrays, so this
fails in a headless session with
  RuntimeError: [metal::load_device] No Metal device available

THIS SCRIPT REQUIRES A REAL METAL DEVICE (a Mac with MLX installed and GPU access) AND
CANNOT RUN HEADLESS. It was written by an agent that could not execute it -- the numpy-shim
counterpart in check_ground_locg_mlx_static.py is what validated the algorithm instead.

Run with:
    uv run python examples/check_ground_locg_mlx_mlx.py
"""

import os
import sys

import jax
import mlx.core as mx
import numpy as np

jax.config.update("jax_enable_x64", True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _bench_common import build_solver_inputs, dense_reference, generate_problem

from rqutils.ground_locg_mlx import apply_h_xz_mlx, ground_locg_mlx

ps, cs, states = generate_problem(10, 20, 200, seed=1)
inputs = build_solver_inputs(ps, cs, states)
H, ref = dense_reference(inputs)
print(f"reference ground energy = {ref:.12f}")

failures = []
for device, name in ((mx.cpu, "cpu"), (mx.gpu, "gpu")):
    for dtype, dtname, rtol in ((mx.float64, "f64", 1e-9), (mx.float32, "f32", 1e-4)):
        arm = f"mlx-{name}-{dtname}"
        if name == "gpu" and dtname == "f64":
            print(f"{arm}: skipped (Metal has no float64)")
            continue
        try:
            mx.set_default_device(device)
            # Pass dtype at CONSTRUCTION, never construct-then-cast: MLX's docs state that
            # "NumPy arrays with type float64 will be default converted to MLX arrays with
            # type float32" (https://ml-explore.github.io/mlx/build/html/usage/numpy.html).
            # mx.array(inputs.diagonals).astype(dtype) truncates to float32 in the first call
            # and only then casts back up in the second -- the low bits are already gone, so
            # the "f64" arm ends up doing float64 arithmetic on float32-precision data. Passing
            # dtype directly (mx.array(x, dtype)) builds at the target precision with no lossy
            # intermediate. xsources is int32 (see _bench_common.py); MLX's default integer
            # type is also int32 and int64 is a full native dtype (not narrowed on ingest like
            # float64 is), so no explicit dtype is required here -- passed anyway for symmetry.
            xs = mx.array(inputs.xsources, mx.int32)
            dg = mx.array(inputs.diagonals, dtype)
            v0 = mx.array(inputs.vinit, dtype)

            mv = np.asarray(apply_h_xz_mlx(v0, xs, dg), dtype=np.float64)
            mverr = np.abs(mv - H @ inputs.vinit).max()

            eig, _, iters, _ = ground_locg_mlx(apply_h_xz_mlx, v0, args=(xs, dg))
            ok = abs(eig - ref) < rtol * max(1.0, abs(ref))
            print(
                f"{arm}: eig={eig:.10f} iters={iters} matvec_err={mverr:.2e} "
                f"{'OK' if ok else 'FAIL'}"
            )
            if not ok:
                failures.append(arm)

            # sas='metal' is f32-only (Metal has no float64), so only exercise it on
            # the f32 arms. This is the ONLY check that can establish the Metal source
            # actually compiles and that its barriers are correct -- the numpy shim
            # cannot.
            if dtname == "f32":
                eig_sas, _, iters_sas, _ = ground_locg_mlx(
                    apply_h_xz_mlx, v0, args=(xs, dg), sas="metal"
                )
                ok_sas = abs(eig_sas - ref) < rtol * max(1.0, abs(ref))
                print(
                    f"{arm} sas=metal: eig={eig_sas:.10f} iters={iters_sas} "
                    f"{'OK' if ok_sas else 'FAIL'}"
                )
                if not ok_sas:
                    failures.append(f"{arm}-sas-metal")
        except Exception as exc:  # noqa: BLE001 (a checker must survive any arm's failure)
            print(f"{arm}: ERROR {type(exc).__name__}: {exc}")
            failures.append(arm)

print("\nFAILURES:", failures if failures else "none")
