"""Verify ground_locg_mlx under real MLX, at both precisions.

Must be run by the user: mlx.core loads a Metal device even for mx.cpu arrays, so this
fails in a headless session with
  RuntimeError: [metal::load_device] No Metal device available

THIS SCRIPT REQUIRES MLX AND A REAL METAL DEVICE (a Mac with MLX installed and GPU access) AND
CANNOT RUN HEADLESS -- importing `mlx.core` is enough to require one. The numpy-shim counterpart in
check_solver_headless.py is what validates the algorithm without hardware.

Its remaining job is narrow: confirm that the port's real-MLX arithmetic agrees with an independent
dense reference at both precisions. It no longer verifies any Metal Shading Language, because the
module no longer contains any. The two fused kernels (`_apply_h_xz_metal`, `_eigenpair_3x3_metal`)
and the `device="cpu"|"gpu"` parameter that selected them were removed when the port was deprecated
-- the JAX solver measured faster even on the MLX GPU backend. `docs/mlx-metal-kernels.md` records
what they were worth and the earlier M1 validation runs.

If a Metal kernel is ever revived here, revive with it the two static MSL guards that used to live
in check_solver_headless.py (the `metal::`-qualification scan and the reserved-identifier scan) --
a numpy shim never compiles the MSL text, so nothing headless can cover it.

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
from solver import apply_h_xz, ground_locg_mlx

ps, cs, states = generate_problem(10, 20, 200, seed=1)
inputs = build_solver_inputs(ps, cs, states)
H, ref = dense_reference(inputs)
print(f"reference ground energy = {ref:.12f}")

failures = []
for dtype, dtname, rtol in ((mx.float64, "f64", 1e-9), (mx.float32, "f32", 1e-4)):
    arm = f"mlx-{dtname}"
    try:
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

        # The matvec against an independent dense H, separately from the solve: this is what
        # distinguishes a matvec bug from a solver-loop bug.
        mv = np.asarray(apply_h_xz(v0, xs, dg), dtype=np.float64)
        mverr = np.abs(mv - H @ inputs.vinit).max()

        eig, _, iters, _ = ground_locg_mlx(apply_h_xz, v0, args=(xs, dg))
        ok = abs(eig - ref) < rtol * max(1.0, abs(ref))
        print(
            f"{arm}: eig={eig:.10f} iters={iters} matvec_err={mverr:.2e} {'OK' if ok else 'FAIL'}"
        )
        if not ok:
            failures.append(arm)
    except Exception as exc:  # noqa: BLE001 (a checker must survive any arm's failure)
        print(f"{arm}: ERROR {type(exc).__name__}: {exc}")
        failures.append(arm)

print("\nFAILURES:", failures if failures else "none")
