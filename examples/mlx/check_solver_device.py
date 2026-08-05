"""Verify ground_locg_mlx under real MLX, on both devices and both precisions.

Must be run by the user: mlx.core loads a Metal device even for mx.cpu arrays, so this
fails in a headless session with
  RuntimeError: [metal::load_device] No Metal device available

THIS SCRIPT REQUIRES A REAL METAL DEVICE (a Mac with MLX installed and GPU access) AND
CANNOT RUN HEADLESS. It was written by an agent that could not execute it -- the numpy-shim
counterpart in check_solver_headless.py is what validated the algorithm instead.

Device status as of 2026-08-05: both surviving Metal kernels -- `_apply_h_xz_metal` and
`_eigenpair_3x3_metal` -- have been exercised on a real M1 GPU and passed, so their MSL is known
to compile and to be correct. The fused Rayleigh-Ritz kernel that used to be checked here is gone;
it measured slower than the op-graph path (see docs/mlx-metal-kernels.md).

The arms below were rewritten when the solver's option surface collapsed to a single `device`
parameter, so the *combinations* are new even though both kernels are device-validated. Re-run
after any change here.

Run with:
    uv run python examples/mlx/check_solver_device.py
"""

import os
import sys

import jax
import mlx.core as mx
import numpy as np

jax.config.update("jax_enable_x64", True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _bench_common import build_solver_inputs, dense_reference, generate_problem

from rqutils.ground_locg_mlx import (
    _apply_h_xz_metal,
    _eigenpair_3x3_metal,
    apply_h_xz,
    eigenpair_3x3,
    ground_locg_mlx,
)

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

            mv = np.asarray(apply_h_xz(v0, xs, dg), dtype=np.float64)
            mverr = np.abs(mv - H @ inputs.vinit).max()

            # device="cpu": the portable op-graph path. The only route for f64, and the baseline
            # every Metal arm is compared against.
            eig, _, iters, _ = ground_locg_mlx(apply_h_xz, v0, args=(xs, dg), device="cpu")
            ok = abs(eig - ref) < rtol * max(1.0, abs(ref))
            print(
                f"{arm} device=cpu: eig={eig:.10f} iters={iters} matvec_err={mverr:.2e} "
                f"{'OK' if ok else 'FAIL'}"
            )
            if not ok:
                failures.append(f"{arm}-device-cpu")

            # device="gpu" is f32-only (Metal has no float64), so only the f32 arms below. These
            # are the ONLY checks that can establish the Metal sources actually compile -- the
            # numpy shim never reads them.
            if dtname == "f32":
                # Fused eigensolve, op-graph matvec. Isolates _eigenpair_3x3_metal: it is the only
                # kernel here that calls math functions (metal::sqrt/cos/sin/atan2), and the static
                # checker's metal::-qualification guard catches their spelling, not availability.
                eig_e, _, iters_e, _ = ground_locg_mlx(apply_h_xz, v0, args=(xs, dg), device="gpu")
                ok_e = abs(eig_e - ref) < rtol * max(1.0, abs(ref))
                print(
                    f"{arm} device=gpu: eig={eig_e:.10f} iters={iters_e} {'OK' if ok_e else 'FAIL'}"
                )
                if not ok_e:
                    failures.append(f"{arm}-device-gpu")

                # Both surviving Metal kernels resident at once: _apply_h_xz_metal for the matvec
                # plus the fused eigensolve. This is exactly what bench.py measures with
                # `--matvec metal` on a gpu arm, and the configuration that matters most.
                mv_metal = np.asarray(_apply_h_xz_metal(v0, xs, dg), dtype=np.float64)
                mverr_metal = np.abs(mv_metal - H @ inputs.vinit).max()
                eig_both, _, iters_both, _ = ground_locg_mlx(
                    _apply_h_xz_metal, v0, args=(xs, dg), device="gpu"
                )
                ok_both = abs(eig_both - ref) < rtol * max(1.0, abs(ref))
                print(
                    f"{arm} metal-both: eig={eig_both:.10f} iters={iters_both} "
                    f"matvec_err={mverr_metal:.2e} {'OK' if ok_both else 'FAIL'}"
                )
                if not ok_both:
                    failures.append(f"{arm}-metal-both")

                # device="gpu" must refuse f64 rather than silently narrowing. Checked on device
                # because the guard lives in ground_locg_mlx, not in the kernel.
                try:
                    ground_locg_mlx(
                        apply_h_xz,
                        mx.array(inputs.vinit, mx.float64),
                        args=(xs, mx.array(inputs.diagonals, mx.float64)),
                        device="gpu",
                    )
                except ValueError:
                    print(f"{arm} f64-guard: device='gpu' rejects float64 OK")
                else:
                    print(f"{arm} f64-guard: device='gpu' accepted float64 FAIL")
                    failures.append(f"{arm}-f64-guard")

                # Direct kernel-vs-op-graph comparison on the matrix classes the solver actually
                # produces, so a transcription error shows up as a wrong eigenvalue on a specific
                # case rather than only as a drifted solve. Mirrors case 3k of
                # check_solver_headless.py, but against the real compiled MSL.
                eig3_ops = eigenpair_3x3
                eig3_met = _eigenpair_3x3_metal
                rng_e = np.random.default_rng(20260805)
                worst_e = 0.0
                for _ in range(40):
                    a = rng_e.normal(size=(3, 3))
                    a = ((a + a.T) * 0.5).astype(np.float32)
                    am = mx.array(a, mx.float32)
                    t_ops, _ = eig3_ops(am)
                    t_met, kp_met = eig3_met(am)
                    ref_min = float(np.linalg.eigvalsh(a.astype(np.float64)).min())
                    den = max(1.0, abs(ref_min))
                    worst_e = max(worst_e, abs(float(t_met) - float(t_ops)) / den)
                    # Independent of the shared Cardano formulation: both rqutils paths would
                    # agree with each other on a root-selection error, but eigh would not.
                    if abs(float(t_met) - ref_min) / den > 1e-4:
                        print(
                            f"{arm} eig3-metal: {float(t_met)} is not the minimum "
                            f"eigenvalue {ref_min} -- FAIL"
                        )
                        failures.append(f"{arm}-eig3-vs-eigh")
                        break
                    kp = np.asarray(kp_met, dtype=np.float64)
                    if abs(np.linalg.norm(kp) - 1.0) > 1e-4:
                        print(f"{arm} eig3-metal: eigenvector not unit norm -- FAIL")
                        failures.append(f"{arm}-eig3-norm")
                        break
                else:
                    print(
                        f"{arm} eig3-metal: matches op-graph and numpy eigh on 40 random "
                        f"symmetric matrices (worst rel {worst_e:.2e}) OK"
                    )
        except Exception as exc:  # noqa: BLE001 (a checker must survive any arm's failure)
            print(f"{arm}: ERROR {type(exc).__name__}: {exc}")
            failures.append(arm)

print("\nFAILURES:", failures if failures else "none")
