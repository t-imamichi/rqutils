"""Verify ground_locg_mlx without a Metal device.

Parses the module, checks its API, then re-executes its source with a numpy shim bound to
the name `mx` to validate the numerics. This catches algorithm transcription errors -- the
kind that matter most -- without needing MLX to initialize.

Run with:
    uv run python examples/check_ground_locg_mlx_static.py

This script does NOT require MLX or a Metal device: it substitutes a numpy shim for
`mlx.core` and re-executes examples/ground_locg_mlx.py's own source text against that shim.
See examples/check_ground_locg_mlx_mlx.py for the real-MLX counterpart, which requires a
Metal device and cannot run headless.
"""
import ast
import inspect
import os
import sys
import types
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, 'ground_locg_mlx.py')
source = open(SRC).read()

# 1. It must parse, and must define the two public functions.
tree = ast.parse(source)
defined = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
for name in ('apply_h_xz_mlx', 'ground_locg_mlx'):
    assert name in defined, f'{name} not defined in {SRC}'
print('OK  module parses and defines both public functions')

# 2. Build a numpy shim exposing the mlx.core surface the port uses.
shim = types.ModuleType('mx')
for fn in ('sum', 'sqrt', 'where', 'stack', 'zeros_like', 'take', 'array', 'abs',
           'minimum', 'maximum', 'argmin', 'real', 'imag', 'conjugate', 'zeros',
           'arange', 'cos', 'sin', 'arctan2', 'diagonal', 'roll', 'prod', 'square',
           'insert', 'concatenate', 'min', 'max'):
    if hasattr(np, fn):
        setattr(shim, fn, getattr(np, fn))
shim.linalg = types.SimpleNamespace(norm=np.linalg.norm, cross=np.cross)
shim.float32, shim.float64 = np.float32, np.float64
shim.eval = lambda *a, **k: None
shim.compile = lambda f: f
shim.Dtype = type(np.dtype('float64'))
shim.finfo = np.finfo


class _CPU:
    pass


shim.cpu = _CPU()
shim.gpu = _CPU()
shim.set_default_device = lambda d: None
shim.default_device = lambda: shim.cpu

module = types.ModuleType('ground_locg_mlx_shimmed')
module.__dict__['np'] = np
sys.modules['mlx'] = types.ModuleType('mlx')
sys.modules['mlx.core'] = shim
exec(compile(source, SRC, 'exec'), module.__dict__)
print('OK  module executes against the numpy shim')

apply_h_xz_mlx = module.apply_h_xz_mlx
ground_locg_mlx = module.ground_locg_mlx

sig = inspect.signature(ground_locg_mlx)
for param in ('mat', 'xinit', 'args', 'maxiter', 'tol'):
    assert param in sig.parameters, f'ground_locg_mlx missing parameter {param}'
print('OK  ground_locg_mlx signature matches ground_locg')

# 3. Numerics, against the same problem Task 1 verified.
import jax
jax.config.update('jax_enable_x64', True)
sys.path.insert(0, HERE)
from _bench_common import generate_problem, build_solver_inputs, dense_reference

ps, cs, states = generate_problem(10, 20, 200, seed=1)
inputs = build_solver_inputs(ps, cs, states)
H, ref = dense_reference(inputs)

# 3a. matvec must equal H @ v
rng = np.random.default_rng(7)
v = rng.normal(size=inputs.subspace_dim)
got = np.asarray(apply_h_xz_mlx(v, inputs.xsources, inputs.diagonals))
err = np.abs(got - H @ v).max()
assert err < 1e-12, f'apply_h_xz_mlx disagrees with H @ v by {err}'
print(f'OK  matvec matches H @ v (max err {err:.2e})')

# 3b. full solve must reach the reference eigenvalue
eigval, eigvec, iters = ground_locg_mlx(apply_h_xz_mlx, inputs.vinit,
                                        args=(inputs.xsources, inputs.diagonals))
assert abs(eigval - ref) < 1e-9 * max(1., abs(ref)), \
    f'solver got {eigval}, reference {ref}'
assert 0 < iters <= 1000, f'implausible iteration count {iters}'
print(f'OK  solve: eig={eigval:.12f} ref={ref:.12f} iters={iters}')

# 3c. tol=0. must run exactly maxiter iterations (fixed-iteration mode)
_, _, fixed_iters = ground_locg_mlx(apply_h_xz_mlx, inputs.vinit,
                                    args=(inputs.xsources, inputs.diagonals),
                                    maxiter=100, tol=0.)
assert fixed_iters == 100, f'tol=0. ran {fixed_iters} iterations, expected exactly 100'
print('OK  tol=0. gives fixed-iteration mode')
print('\nALL STATIC CHECKS PASSED (numpy shim; MLX itself still unverified)')
