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
import numpy as np
import mlx.core as mx
import jax
jax.config.update('jax_enable_x64', True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _bench_common import generate_problem, build_solver_inputs, dense_reference
from ground_locg_mlx import apply_h_xz_mlx, ground_locg_mlx

ps, cs, states = generate_problem(10, 20, 200, seed=1)
inputs = build_solver_inputs(ps, cs, states)
H, ref = dense_reference(inputs)
print(f'reference ground energy = {ref:.12f}')

failures = []
for device, name in ((mx.cpu, 'cpu'), (mx.gpu, 'gpu')):
    for dtype, dtname, rtol in ((mx.float64, 'f64', 1e-9), (mx.float32, 'f32', 1e-4)):
        arm = f'mlx-{name}-{dtname}'
        if name == 'gpu' and dtname == 'f64':
            print(f'{arm}: skipped (Metal has no float64)')
            continue
        try:
            mx.set_default_device(device)
            xs = mx.array(inputs.xsources)
            dg = mx.array(inputs.diagonals).astype(dtype)
            v0 = mx.array(inputs.vinit).astype(dtype)

            mv = np.asarray(apply_h_xz_mlx(v0, xs, dg), dtype=np.float64)
            mverr = np.abs(mv - H @ inputs.vinit).max()

            eig, _, iters = ground_locg_mlx(apply_h_xz_mlx, v0, args=(xs, dg))
            ok = abs(eig - ref) < rtol * max(1., abs(ref))
            print(f'{arm}: eig={eig:.10f} iters={iters} matvec_err={mverr:.2e} '
                  f'{"OK" if ok else "FAIL"}')
            if not ok:
                failures.append(arm)
        except Exception as exc:
            print(f'{arm}: ERROR {type(exc).__name__}: {exc}')
            failures.append(arm)

print('\nFAILURES:', failures if failures else 'none')
