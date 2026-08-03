"""Localize where float64 precision is lost inside ground_locg_mlx under real MLX.

Context: after fixing the float64->float32 truncation on array ingest, mlx-cpu-f64's
matvec_err went to exactly 0.0 (the matvec is now exact), but the solver still stalls at
maxiter=1000 with an eigenvalue error of ~4.1e-08 instead of converging like the JAX f64
path does in 89 iterations. Since matvec_err is measured OUTSIDE the solver, it cannot see
anything the loop does internally. This script instruments the inside.

REQUIRES A REAL METAL DEVICE -- cannot run headless. Run with:

    uv run python examples/diagnose_mlx_f64.py

It prints one section per candidate cause. Send back the whole output; no single line is
conclusive on its own, and the point is to find which boundary loses the bits rather than
to guess.
"""
import os
import sys
import numpy as np
import mlx.core as mx
import jax
jax.config.update('jax_enable_x64', True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _bench_common import generate_problem, build_solver_inputs, dense_reference
from ground_locg_mlx import apply_h_xz_mlx, eigenpair_2x2, eigenpair_3x3, _project_out

mx.set_default_device(mx.cpu)
F64 = mx.float64

print('=' * 78)
print('SECTION 1 -- does MLX float64 survive the basic ops the loop uses?')
print('=' * 78)
# A value whose low bits are destroyed by any f32 round-trip: 1 + 2^-40 is
# representable in f64 but rounds to exactly 1.0 in f32.
probe = 1.0 + 2.0 ** -40
a = mx.array(np.array([probe], dtype=np.float64), F64)
print(f'ingest            : dtype={a.dtype} value_minus_1={float(a[0]) - 1.0:.6e} '
      f'(expect ~9.09e-13, NOT 0.0)')

checks = [
    ('mx.sum', lambda x: mx.sum(x)),
    ('mx.sqrt', lambda x: mx.sqrt(x)),
    ('x * x', lambda x: x * x),
    ('x + x', lambda x: x + x),
    ('mx.linalg.norm', lambda x: mx.linalg.norm(x)),
    ('x / norm(x)', lambda x: x / mx.linalg.norm(x)),
    ('mx.where', lambda x: mx.where(x > 0, x, x)),
    ('mx.stack', lambda x: mx.stack([x[0], x[0]])),
    ('mx.take', lambda x: mx.take(x, mx.array(np.array([0], dtype=np.int32), mx.int32))),
    ('x * 0.5 (py float)', lambda x: x * 0.5),
    ('x * mx.array(0.5)', lambda x: x * mx.array(0.5, F64)),
]
for name, fn in checks:
    try:
        out = fn(a)
        mx.eval(out)
        print(f'{name:<20}: dtype={out.dtype}'
              f'{"  <-- LOST f64!" if out.dtype != F64 else ""}')
    except Exception as exc:
        print(f'{name:<20}: ERROR {type(exc).__name__}: {exc}')

print()
print('=' * 78)
print('SECTION 2 -- mx.linalg.norm accuracy in f64')
print('=' * 78)
# norm of a vector whose true norm needs f64 to distinguish from 1.0
v = mx.array(np.array([1.0, 2.0 ** -30], dtype=np.float64), F64)
got = float(mx.linalg.norm(v))
expected = float(np.linalg.norm(np.array([1.0, 2.0 ** -30], dtype=np.float64)))
print(f'norm dtype        : {mx.linalg.norm(v).dtype}')
print(f'norm(v) MLX       : {got!r}')
print(f'norm(v) numpy f64 : {expected!r}')
print(f'difference        : {abs(got - expected):.6e}   (expect ~0, f32 would give 1.0)')
print(f'is MLX norm exactly 1.0 (f32 behaviour)? {got == 1.0}')

print()
print('=' * 78)
print('SECTION 3 -- mx.linalg.cross (used by eigenpair_3x3) in f64')
print('=' * 78)
c0 = np.array([1.0, 2.0 ** -30, 0.0], dtype=np.float64)
c1 = np.array([0.0, 1.0, 2.0 ** -30], dtype=np.float64)
cr = mx.linalg.cross(mx.array(c0, F64), mx.array(c1, F64))
mx.eval(cr)
print(f'cross dtype       : {cr.dtype}{"  <-- LOST f64!" if cr.dtype != F64 else ""}')
print(f'cross MLX         : {np.asarray(cr, dtype=np.float64)!r}')
print(f'cross numpy f64   : {np.cross(c0, c1)!r}')

print()
print('=' * 78)
print('SECTION 4 -- the real problem: dtype of every value inside one iteration')
print('=' * 78)
ps, cs, states = generate_problem(10, 20, 200, seed=1)
inputs = build_solver_inputs(ps, cs, states)
H, ref = dense_reference(inputs)
xs = mx.array(inputs.xsources, mx.int32)
dg = mx.array(inputs.diagonals, F64)
x = mx.array(inputs.vinit, F64)
print(f'reference eigenvalue = {ref!r}')
print(f'xsources dtype={xs.dtype}  diagonals dtype={dg.dtype}  vinit dtype={x.dtype}')

x = x / mx.linalg.norm(x)
ax = apply_h_xz_mlx(x, xs, dg)
rho = mx.sum(x * ax)
r = ax - rho * x
for label, arr in (('x/|x|', x), ('A@x', ax), ('rho', rho), ('r', r)):
    print(f'  {label:<8} dtype={arr.dtype}{"  <-- LOST f64!" if arr.dtype != F64 else ""}')

p = _project_out((x, mx.zeros_like(x)), r)
print(f'  _project_out -> dtype={p.dtype}'
      f'{"  <-- LOST f64!" if p.dtype != F64 else ""}')

# Build the Rayleigh-Ritz matrix the way the solver does and inspect it.
mvs = [apply_h_xz_mlx(v, xs, dg) for v in (x, mx.zeros_like(x), p)]
rows = [mx.stack([mx.sum(v1 * mvs[i2]) for i2 in range(3)]) for v1 in (x, mx.zeros_like(x), p)]
sas = mx.stack(rows)
sas = (sas + sas.T) * 0.5
mx.eval(sas)
print(f'  Rayleigh-Ritz 3x3 dtype={sas.dtype}'
      f'{"  <-- LOST f64!" if sas.dtype != F64 else ""}')
theta3, kappa3 = eigenpair_3x3(sas)
mx.eval(theta3, kappa3)
print(f'  eigenpair_3x3 -> theta dtype={theta3.dtype} kappa dtype={kappa3.dtype}'
      f'{"  <-- LOST f64!" if theta3.dtype != F64 else ""}')
print(f'  eigenpair_3x3 theta   = {float(theta3)!r}')
sas_np = np.asarray(sas, dtype=np.float64)
print(f'  numpy eigvalsh lowest = {np.linalg.eigvalsh(sas_np)[0]!r}')
print(f'  eigenpair_3x3 abs err = {abs(float(theta3) - np.linalg.eigvalsh(sas_np)[0]):.6e}')

theta2, kappa2 = eigenpair_2x2(sas[:2, :2])
mx.eval(theta2, kappa2)
sas2 = sas_np[:2, :2]
print(f'  eigenpair_2x2 -> theta dtype={theta2.dtype}'
      f'{"  <-- LOST f64!" if theta2.dtype != F64 else ""}')
print(f'  eigenpair_2x2 abs err = {abs(float(theta2) - np.linalg.eigvalsh(sas2)[0]):.6e}')

print()
print('=' * 78)
print('SECTION 5 -- residual trajectory: does it stall, and where?')
print('=' * 78)
print('Running the solver loop manually, printing the residual norm every 10 iterations.')
print('Convergence needs norm(r) < eps_f64 * reltol; with reltol ~9e3 that is ~2e-12.')
print()


def sas_of(*vectors):
    mvs_ = [apply_h_xz_mlx(v, xs, dg) for v in vectors]
    rows_ = [mx.stack([mx.sum(v1 * mvs_[i2]) for i2 in range(len(vectors))])
             for v1 in vectors]
    out = mx.stack(rows_)
    return (out + out.T) * 0.5


xc = mx.array(inputs.vinit, F64)
xc = xc / mx.linalg.norm(xc)
axc = apply_h_xz_mlx(xc, xs, dg)
rc = axc - mx.sum(xc * axc) * xc
nr = mx.linalg.norm(rc)
pp = rc / mx.where(nr == 0., mx.array(1., nr.dtype), nr)
th, ka = eigenpair_2x2(sas_of(xc, pp))
tt = pp * ka[0] - xc * ka[1]
uu = xc * ka[0] + pp * ka[1]
xc = uu / mx.linalg.norm(uu)
yc = tt / mx.linalg.norm(tt)
rc = apply_h_xz_mlx(xc, xs, dg) - th * xc

for it in range(1, 201):
    pp = _project_out((xc, yc), rc)
    th, ka = eigenpair_3x3(sas_of(xc, yc, pp))
    ss = yc * ka[1] + pp * ka[2]
    ns = mx.linalg.norm(ss)
    tt = ss * (ka[0] / ns) - xc * ns
    uu = xc * ka[0] + ss
    xc = uu / mx.linalg.norm(uu)
    yc = tt / mx.linalg.norm(tt)
    axn = apply_h_xz_mlx(xc, xs, dg)
    rc = axn - xc * th
    if it % 10 == 0 or it <= 3:
        reltol = float((mx.linalg.norm(axn) - th) * xc.shape[0] * 10)
        need = float(np.finfo(np.float64).eps) * reltol
        print(f'  iter {it:>3}: |r|={float(mx.linalg.norm(rc)):.6e}  '
              f'theta={float(th):.15f}  theta_err={abs(float(th) - ref):.3e}  '
              f'need |r|<{need:.3e}')

print()
print('Send the ENTIRE output back. Any "LOST f64!" marker localizes the bug immediately;')
print('if none appear, Section 5 shows whether the residual stalls at a floor (a precision')
print('loss somewhere) or decreases steadily (a convergence-criterion problem instead).')
