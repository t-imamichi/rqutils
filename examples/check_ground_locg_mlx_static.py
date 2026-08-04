"""Verify ground_locg_mlx without a Metal device.

Parses the module, checks its API, then re-executes its source with a numpy shim bound to
the name `mx` to validate the numerics. This catches algorithm transcription errors -- the
kind that matter most -- without needing MLX to initialize.

Run with:
    uv run python examples/check_ground_locg_mlx_static.py

This script does NOT require MLX or a Metal device: it substitutes a numpy shim for
`mlx.core` and re-executes rqutils/ground_locg_mlx.py's own source text against that shim.
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
# The module under test lives in the package; HERE is examples/, so go up one level.
SRC = os.path.join(os.path.dirname(HERE), "rqutils", "ground_locg_mlx.py")
with open(SRC) as source_file:
    source = source_file.read()

# 1. It must parse, and must define the two public functions.
tree = ast.parse(source)
defined = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
for name in ("apply_h_xz_mlx", "ground_locg_mlx"):
    assert name in defined, f"{name} not defined in {SRC}"
print("OK  module parses and defines both public functions")

# 2. Build a numpy shim exposing the mlx.core surface the port uses.
shim = types.ModuleType("mx")
for fn in (
    "sum",
    "sqrt",
    "where",
    "stack",
    "zeros_like",
    "take",
    "array",
    "abs",
    "minimum",
    "maximum",
    "argmin",
    "real",
    "imag",
    "conjugate",
    "zeros",
    "arange",
    "cos",
    "sin",
    "arctan2",
    "diagonal",
    "roll",
    "prod",
    "square",
    "insert",
    "concatenate",
    "min",
    "max",
    # Added when the port was synced with the hardened JAX algorithms: balancing needs eye/sign,
    # the rank-aware null-vector search needs argmax, and the zero-direction guards need
    # zeros_like/logical_or.
    "eye",
    "sign",
    "argmax",
    "logical_or",
    "matmul",
):
    if hasattr(np, fn):
        setattr(shim, fn, getattr(np, fn))
shim.linalg = types.SimpleNamespace(norm=np.linalg.norm, cross=np.cross)
shim.float32, shim.float64 = np.float32, np.float64
shim.eval = lambda *a, **k: None
shim.compile = lambda f: f
shim.Dtype = type(np.dtype("float64"))
shim.finfo = np.finfo


def _shim_metal_kernel(name, input_names, output_names, source, **kwargs):
    """Stand in for mx.fast.metal_kernel by interpreting each kernel's arithmetic in numpy.

    Dispatches on `name`: this module now has two kernels. Each closure below is an independent
    numpy reimplementation of the kernel's *intended* per-thread/per-threadgroup indexing -- it
    does not read the `source` argument at all. So it can catch a divergence between the CALLER's
    contract (shapes, dtypes, grid/threadgroup setup, flat row-major indexing) and that intent,
    but it is blind to a bug in the Metal source text itself (e.g. a transposed index inside
    `source`): such a bug would not change what this shim computes, since the shim never looks at
    `source`. It also cannot verify that the Metal source compiles, that the barriers are
    correct, or that it is right on device -- only the user's real-hardware run can.
    """

    def call_matvec(inputs, output_dtypes=None, **kw):
        vec, xsources, diagonals, num_groups, num_states = inputs
        xs_flat = np.asarray(xsources).reshape(-1)
        dg_flat = np.asarray(diagonals).reshape(-1)
        vec_np = np.asarray(vec)
        out = np.zeros(num_states, dtype=np.dtype(output_dtypes[0]))
        for j in range(num_groups):
            off = j * num_states
            out = out + vec_np[xs_flat[off : off + num_states]] * dg_flat[off : off + num_states]
        return [out]

    def call_sas(inputs, grid=None, threadgroup=None, output_dtypes=None, **kw):
        vectors, mvs, num_basis, num_states = inputs
        vec_np = np.asarray(vectors).reshape(-1)
        mv_np = np.asarray(mvs).reshape(-1)
        lanes = threadgroup[0]
        num_pairs = num_basis * (num_basis + 1) // 2
        assert grid[0] == num_pairs * lanes, (
            f"grid {grid[0]} != num_pairs*lanes {num_pairs * lanes}: the caller must launch one "
            "threadgroup per (i, j) pair"
        )
        assert lanes & (lanes - 1) == 0, f"threadgroup {lanes} is not a power of two"
        out = np.zeros((num_basis, num_basis), dtype=np.dtype(output_dtypes[0]))
        # Reproduce the kernel's *intended* pair-unranking scan, strided partial sums, and tree
        # reduction -- not just the mathematical result -- so a mismatch between the caller's
        # grid/threadgroup contract and that intent shows up here. This is independent of
        # `source`, so it cannot catch a bug written into the Metal source text itself.
        pairs = [(a, b) for a in range(num_basis) for b in range(a, num_basis)]
        for i, j in pairs:
            partials = np.zeros(lanes, dtype=np.dtype(output_dtypes[0]))
            for lane in range(lanes):
                acc = np.dtype(output_dtypes[0]).type(0)
                for k in range(lane, num_states, lanes):
                    acc += vec_np[i * num_states + k] * mv_np[j * num_states + k]
                partials[lane] = acc
            half = lanes // 2
            while half > 0:
                partials[:half] += partials[half : half * 2]
                half //= 2
            out[i, j] = partials[0]
            out[j, i] = partials[0]
        return [out]

    if name == "sqd_apply_h_xz":
        return call_matvec
    if name == "sqd_compute_sas":
        return call_sas
    raise AssertionError(f"no shim implementation for Metal kernel {name!r}")


shim.fast = types.SimpleNamespace(metal_kernel=_shim_metal_kernel)


class _CPU:
    pass


shim.cpu = _CPU()
shim.gpu = _CPU()
shim.set_default_device = lambda d: None
shim.default_device = lambda: shim.cpu

module = types.ModuleType("ground_locg_mlx_shimmed")
module.__dict__["np"] = np
sys.modules["mlx"] = types.ModuleType("mlx")
sys.modules["mlx.core"] = shim
# Executing the module's own source against a numpy shim is this script's entire purpose: it
# validates the MLX port's algorithm on a machine with no Metal device.
exec(compile(source, SRC, "exec"), module.__dict__)  # noqa: S102
print("OK  module executes against the numpy shim")

apply_h_xz_mlx = module.apply_h_xz_mlx
ground_locg_mlx = module.ground_locg_mlx

sig = inspect.signature(ground_locg_mlx)
for param in ("mat", "xinit", "args", "maxiter", "tol"):
    assert param in sig.parameters, f"ground_locg_mlx missing parameter {param}"
print("OK  ground_locg_mlx signature matches ground_locg")

# 3. Numerics, against the same problem Task 1 verified.
import jax

jax.config.update("jax_enable_x64", True)
sys.path.insert(0, HERE)
from _bench_common import build_solver_inputs, dense_reference, generate_problem

ps, cs, states = generate_problem(10, 20, 200, seed=1)
inputs = build_solver_inputs(ps, cs, states)
H, ref = dense_reference(inputs)

# 3a. matvec must equal H @ v
rng = np.random.default_rng(7)
v = rng.normal(size=inputs.subspace_dim)
got = np.asarray(apply_h_xz_mlx(v, inputs.xsources, inputs.diagonals))
err = np.abs(got - H @ v).max()
assert err < 1e-12, f"apply_h_xz_mlx disagrees with H @ v by {err}"
print(f"OK  matvec matches H @ v (max err {err:.2e})")

# 3b. full solve must reach the reference eigenvalue
eigval, eigvec, iters, converged = ground_locg_mlx(
    apply_h_xz_mlx, inputs.vinit, args=(inputs.xsources, inputs.diagonals)
)
assert abs(eigval - ref) < 1e-9 * max(1.0, abs(ref)), f"solver got {eigval}, reference {ref}"
assert 0 < iters <= 1000, f"implausible iteration count {iters}"
print(f"OK  solve: eig={eigval:.12f} ref={ref:.12f} iters={iters}")

# 3c. tol=0. must run exactly maxiter iterations (fixed-iteration mode)
_, _, fixed_iters, _ = ground_locg_mlx(
    apply_h_xz_mlx, inputs.vinit, args=(inputs.xsources, inputs.diagonals), maxiter=100, tol=0.0
)
assert fixed_iters == 100, f"tol=0. ran {fixed_iters} iterations, expected exactly 100"
print("OK  tol=0. gives fixed-iteration mode")

# 3d. apply_h_xz_mlx_chunked must agree with apply_h_xz_mlx and with H @ v, for several chunk
# sizes -- this is the restructured-control-flow check for Optimization 1 (chunked matvec). The
# shim's mx.take/mx.sum are plain numpy, so this exercises the chunking logic itself, not MLX
# kernels.
apply_h_xz_mlx_chunked = module.apply_h_xz_mlx_chunked
for chunk in (1, 4, 8, 16, 32, 128):
    got_chunked = np.asarray(apply_h_xz_mlx_chunked(v, inputs.xsources, inputs.diagonals, chunk))
    err_vs_loop = np.abs(got_chunked - got).max()
    err_vs_h = np.abs(got_chunked - H @ v).max()
    assert err_vs_loop < 1e-9, f"chunk={chunk}: disagrees with loop matvec by {err_vs_loop}"
    assert err_vs_h < 1e-9, f"chunk={chunk}: disagrees with H @ v by {err_vs_h}"
print("OK  apply_h_xz_mlx_chunked matches apply_h_xz_mlx and H @ v for chunk in {1,4,8,16,32,128}")

# 3g. apply_h_xz_mlx_metal's CALLER logic: correct flat row-major indexing, shapes, grid setup,
# and the float32-only guard. The shim interprets the kernel's arithmetic in numpy, so this
# checks that the call is wired up correctly -- it CANNOT check that the Metal source compiles
# or is numerically right on device. Only the user's real-hardware run establishes that.
apply_h_xz_mlx_metal = module.apply_h_xz_mlx_metal
xs32 = inputs.xsources.astype(np.int32)
got_metal = np.asarray(
    apply_h_xz_mlx_metal(v.astype(np.float32), xs32, inputs.diagonals.astype(np.float32))
)
err_metal = np.abs(got_metal - (H @ v)).max()
assert err_metal < 1e-3, f"metal kernel caller logic disagrees with H @ v by {err_metal}"
print(f"OK  apply_h_xz_mlx_metal caller logic matches H @ v (f32, max err {err_metal:.2e})")

# The float64 guard must fire rather than silently producing a wrong-precision result: Metal
# has no float64, so an f64 arm must be routed to the chunked path instead.
try:
    apply_h_xz_mlx_metal(v, xs32, inputs.diagonals)
except ValueError as exc:
    assert "float32" in str(exc), f"unexpected guard message: {exc}"
    print("OK  apply_h_xz_mlx_metal rejects float64 input (Metal has no float64)")
else:
    raise AssertionError("apply_h_xz_mlx_metal accepted float64 input -- guard did not fire")

# 3e. compile_body=True must produce the restructured control flow's eigenvalue, and must NOT
# alter default (compile_body=False) behaviour. The shim stubs mx.compile as identity, so this
# does not validate real MLX compilation -- only that the chunked-convergence-check control
# flow this port adds is numerically sound. Use tol=0. (fixed-iteration) first: no convergence
# check happens at all, so the compiled body must reproduce the uncompiled tol=0. trajectory
# exactly (same ops, same order).
eigval_default, _, iters_default, _ = ground_locg_mlx(
    apply_h_xz_mlx, inputs.vinit, args=(inputs.xsources, inputs.diagonals), maxiter=100, tol=0.0
)
eigval_compiled, _, iters_compiled, _ = ground_locg_mlx(
    apply_h_xz_mlx,
    inputs.vinit,
    args=(inputs.xsources, inputs.diagonals),
    maxiter=100,
    tol=0.0,
    compile_body=True,
)
assert iters_compiled == iters_default == 100, (
    f"compile_body fixed-iteration mismatch: {iters_compiled} vs {iters_default}"
)
assert abs(eigval_compiled - eigval_default) < 1e-12, (
    f"compile_body changed the fixed-iteration eigenvalue: {eigval_compiled} vs {eigval_default}"
)
print(
    "OK  compile_body=True, tol=0. matches compile_body=False bit-for-bit "
    f"(eig={eigval_compiled:.12f}, iters={iters_compiled})"
)

# 3f. compile_body=True with convergence checking (chunked between compile_chunk iterations)
# must still reach the reference eigenvalue, and compile_body=False must be completely
# unaffected by compile_body's existence (same call as 3b, repeated to prove no state leaks).
eigval_chunked_conv, _, iters_chunked_conv, _ = ground_locg_mlx(
    apply_h_xz_mlx,
    inputs.vinit,
    args=(inputs.xsources, inputs.diagonals),
    compile_body=True,
    compile_chunk=10,
)
assert abs(eigval_chunked_conv - ref) < 1e-9 * max(1.0, abs(ref)), (
    f"compile_body=True with convergence checking got {eigval_chunked_conv}, reference {ref}"
)
print(
    f"OK  compile_body=True with chunked convergence checking: "
    f"eig={eigval_chunked_conv:.12f} iters={iters_chunked_conv}"
)

eigval_default2, _, iters_default2, _ = ground_locg_mlx(
    apply_h_xz_mlx, inputs.vinit, args=(inputs.xsources, inputs.diagonals)
)
assert iters_default2 == iters, (
    f"default (compile_body=False) iteration count changed after exercising compile_body: "
    f"{iters_default2} vs original {iters} -- compile_body must be a strict no-op when unset"
)
assert abs(eigval_default2 - eigval) < 1e-12
print(
    "OK  default path (compile_body=False) unaffected -- byte-identical to the pre-existing "
    f"behaviour (iters={iters_default2})"
)

# 3h. _compute_sas_metal's CALLER logic and arithmetic, for both basis sizes. Same standing
# caveat as 3g: the shim interprets the kernel in numpy, so this validates indexing, shapes,
# grid setup and the both-triangles symmetry claim -- NOT that the Metal source compiles.
_compute_sas = module._compute_sas
_compute_sas_metal = module._compute_sas_metal

rng_sas = np.random.default_rng(11)
for nbasis in (2, 3):
    vecs = [rng_sas.normal(size=inputs.subspace_dim) for _ in range(nbasis)]
    mvs = [np.asarray(apply_h_xz_mlx(vv, inputs.xsources, inputs.diagonals)) for vv in vecs]
    want = np.asarray(_compute_sas(tuple(vecs), tuple(mvs)))
    vecs32 = [vv.astype(np.float32) for vv in vecs]
    mvs32 = [mm.astype(np.float32) for mm in mvs]
    got_sas = np.asarray(_compute_sas_metal(tuple(vecs32), tuple(mvs32)))

    assert got_sas.shape == (nbasis, nbasis), f"n={nbasis}: shape {got_sas.shape}"
    # Exact symmetry, not symmetry-to-tolerance: thread 0 writes the same value to both
    # triangles, so the two must be bit-identical. This is stronger than the op-graph path's
    # (sas + sas.T) * 0.5, which averages two values differing by rounding.
    asym = np.abs(got_sas - got_sas.T).max()
    assert asym == 0.0, f"n={nbasis}: output not exactly symmetric (|S - S.T| = {asym})"
    err_sas = np.abs(got_sas - want).max()
    scale = max(1.0, np.abs(want).max())
    assert err_sas < 1e-4 * scale, f"n={nbasis}: disagrees with _compute_sas by {err_sas}"

    # Independent reference: the inner products computed directly, not via either code path.
    direct = np.array([[vv @ mm for mm in mvs] for vv in vecs])
    direct = (direct + direct.T) * 0.5
    err_direct = np.abs(got_sas - direct).max()
    assert err_direct < 1e-4 * scale, f"n={nbasis}: disagrees with direct v@m by {err_direct}"
    print(
        f"OK  _compute_sas_metal n={nbasis} matches _compute_sas ({err_sas:.2e}) and "
        f"direct v@m ({err_direct:.2e}), exactly symmetric"
    )

# 3h continued: the loop above uses mvs = H @ v for real symmetric H, so v_i . mv_j ==
# v_j . mv_i mathematically -- a transposed `vectors[j]*mvs[i]` bug in the kernel source would be
# invisible even on real hardware with that data, and the (direct + direct.T) * 0.5 above
# symmetrizes away the only remaining signal in the shim comparison too. Use genuinely asymmetric
# mvs (independent random vectors, not images of any operator) to break that degeneracy: now
# v_i . mv_j != v_j . mv_i in general, so a transposed index would show up.
#
# The kernel only ever promises the i <= j inner products (both triangles get the *same* write),
# so the right assertion is against those specific dot products, not against a full v @ M matrix:
# out[i, j] == out[j, i] == v_i . mv_j for every i <= j -- exactly what the docstring claims.
rng_asym = np.random.default_rng(23)
for nbasis in (2, 3):
    vecs_a = [rng_asym.normal(size=inputs.subspace_dim).astype(np.float32) for _ in range(nbasis)]
    mvs_a = [rng_asym.normal(size=inputs.subspace_dim).astype(np.float32) for _ in range(nbasis)]
    got_asym = np.asarray(_compute_sas_metal(tuple(vecs_a), tuple(mvs_a)))

    asym2 = np.abs(got_asym - got_asym.T).max()
    assert asym2 == 0.0, f"n={nbasis} (asymmetric mvs): output not exactly symmetric ({asym2})"

    scale_a = max(1.0, np.abs(np.stack(vecs_a)).max() * np.abs(np.stack(mvs_a)).max() * 100)
    for i in range(nbasis):
        for j in range(i, nbasis):
            want_ij = float(vecs_a[i] @ mvs_a[j])
            err_ij = abs(float(got_asym[i, j]) - want_ij)
            assert err_ij < 1e-4 * scale_a, (
                f"n={nbasis} (asymmetric mvs): out[{i},{j}]={got_asym[i, j]} disagrees with "
                f"v_{i}.mv_{j}={want_ij} by {err_ij}"
            )
            err_ji = abs(float(got_asym[j, i]) - want_ij)
            assert err_ji < 1e-4 * scale_a, (
                f"n={nbasis} (asymmetric mvs): out[{j},{i}]={got_asym[j, i]} disagrees with "
                f"v_{i}.mv_{j}={want_ij} by {err_ji} -- both triangles must equal v_i.mv_j, "
                "not v_j.mv_i"
            )
    print(
        f"OK  _compute_sas_metal n={nbasis} with asymmetric mvs: out[i,j]==out[j,i]==v_i.mv_j "
        "for all i<=j (discriminates a transposed index)"
    )

# The float64 guard must fire, mirroring apply_h_xz_mlx_metal's (Metal has no float64).
try:
    _compute_sas_metal((vecs[0], vecs[1]), (mvs[0], mvs[1]))
except ValueError as exc:
    assert "float32" in str(exc), f"unexpected guard message: {exc}"
    print("OK  _compute_sas_metal rejects float64 input (Metal has no float64)")
else:
    raise AssertionError("_compute_sas_metal accepted float64 input -- guard did not fire")

print("\nALL STATIC CHECKS PASSED (numpy shim; MLX itself still unverified)")
