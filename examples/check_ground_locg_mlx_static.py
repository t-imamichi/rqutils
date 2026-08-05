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
import re
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
    # Added when _nullvec_3x3's candidate construction was batched: the orthogonal-complement
    # candidates are now built as `col x e_k` via one broadcast cross product instead of by
    # indexing out individual scalars and negating them one at a time.
    "broadcast_to",
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

    def call_eig3(inputs, grid=None, threadgroup=None, output_dtypes=None, **kw):
        (mat,) = inputs
        m = np.asarray(mat)
        assert m.shape == (3, 3), f"eigenpair_3x3 kernel expects a (3, 3) matrix, got {m.shape}"
        assert grid == (1, 1, 1), (
            f"grid {grid} != (1, 1, 1): the 3x3 eigensolve has no output parallelism, so the "
            "caller must launch exactly one thread"
        )
        assert threadgroup == (1, 1, 1), f"threadgroup {threadgroup} != (1, 1, 1)"
        dtype = np.dtype(output_dtypes[0])
        # Reproduce the kernel's *intended* single-thread arithmetic, in the same order: balance,
        # Cardano, the rank-aware seven-candidate search with first-extremum tie-breaking, then the
        # closing Rayleigh quotient. Independent of `source`, so a bug in the Metal text itself is
        # invisible here -- only the real-device run can catch that.
        shift = (m[0, 0] + m[1, 1] + m[2, 2]) / 3.0
        scale = np.abs(m).max()
        if not scale > 0.0:
            scale = 1.0
        b = (m - shift * np.eye(3)) / scale
        bd = np.diagonal(b)
        od = np.array([b[1, 0], b[2, 0], b[2, 1]]) ** 2
        c1 = (bd[0] * bd[2] + bd[1] * bd[0] + bd[2] * bd[1]) - od.sum()
        c0 = (
            (bd[0] * od[2] + bd[1] * od[1] + bd[2] * od[0])
            - bd[0] * bd[1] * bd[2]
            - 2.0 * (b[0, 2] * b[1, 0] * b[2, 1])
        )
        p = max(-3.0 * c1, 0.0)
        disc = max(-27.0 * c1**3 - 182.25 * c0 * c0, 0.0)
        phi = np.arctan2(np.sqrt(disc), -13.5 * c0) / 3.0
        cphi, sphi = np.cos(phi), np.sin(phi)
        sqrt3 = np.sqrt(3.0)
        xmin = min(2.0 * cphi, -cphi - sqrt3 * sphi, -cphi + sqrt3 * sphi) * np.sqrt(p) / 3.0

        s = b - xmin * np.eye(3)
        cands = np.zeros((7, 3))
        for k, (a, bcol) in enumerate(((0, 1), (1, 2), (2, 0))):
            cands[k] = np.cross(s[:, a], s[:, bcol])
        colnorms = np.sum(np.square(s), axis=0)
        # First maximum, matching the kernel's strict-> ascending scan and argmax.
        col_index = int(np.argmax(colnorms))
        col = s[:, col_index]
        cands[3] = [0.0, col[2], -col[1]]
        cands[4] = [-col[2], 0.0, col[0]]
        cands[5] = [col[1], -col[0], 0.0]
        cands[6] = [1.0, 0.0, 0.0]
        norms = np.linalg.norm(cands, axis=1)
        cands = cands / np.where(norms == 0.0, 1.0, norms)[:, None]
        resid = np.linalg.norm(cands @ s.T, axis=1)
        resid = resid + (norms == 0.0) * (resid.max() + 1.0)
        # First minimum, matching the kernel's strict-< ascending scan and argmin.
        vec = cands[int(np.argmin(resid))]
        rq = float(vec @ (b @ vec))
        return [
            np.array([rq * scale + shift], dtype=dtype),
            np.asarray(vec, dtype=np.dtype(output_dtypes[1])),
        ]

    if name == "sqd_apply_h_xz":
        return call_matvec
    if name == "sqd_compute_sas":
        return call_sas
    if name == "sqd_eigenpair_3x3":
        return call_eig3
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

# 3i. sas="metal" must be a strict no-op on the trajectory relative to sas="ops" under the
# shim, where both reduce to numpy arithmetic. Fixed-iteration mode (tol=0.) so no convergence
# check can mask a divergence: same ops, same order, same iteration count.
inputs32_v = inputs.vinit.astype(np.float32)
xs32_i = inputs.xsources.astype(np.int32)
dg32_i = inputs.diagonals.astype(np.float32)
eig_ops, _, it_ops, _ = ground_locg_mlx(
    apply_h_xz_mlx, inputs32_v, args=(xs32_i, dg32_i), maxiter=30, tol=0.0, sas="ops"
)
eig_met, _, it_met, _ = ground_locg_mlx(
    apply_h_xz_mlx, inputs32_v, args=(xs32_i, dg32_i), maxiter=30, tol=0.0, sas="metal"
)
assert it_ops == it_met == 30, f"iteration counts differ: {it_ops} vs {it_met}"
scale_eig = max(1.0, abs(eig_ops))
# Measured exact agreement (0.0) under the shim, where both paths reduce to the same float32
# numpy arithmetic on this small problem -- tightened from an earlier 1e-4*scale, which would
# have passed a genuine ~4e-4 kernel defect. This numerical check is defence-in-depth on top of
# the call-spy check below, which is what actually proves the two branches are wired up.
assert abs(eig_met - eig_ops) < 1e-8 * scale_eig, (
    f"sas='metal' changed the fixed-iteration eigenvalue: {eig_met} vs {eig_ops}"
)
print(f"OK  sas='metal' tracks sas='ops' (eig {eig_met:.6f} vs {eig_ops:.6f}, {it_met} iters)")

# 3i continued: the check above proves "metal doesn't diverge from ops", but not that sas="metal"
# actually dispatches to _compute_sas_metal -- under the shim the two paths agree to the bit on
# this problem, so a mis-wired call site (e.g. both branches hardcoded to _compute_sas) would
# pass it silently. Wrap each module-level function with a counting spy and confirm sas="metal"
# calls only the metal implementation, and sas="ops" only the ops implementation. The binding
# `compute_sas = _compute_sas_metal if sas == "metal" else _compute_sas` reads these module
# globals at call time, so patching module._compute_sas / module._compute_sas_metal intercepts it.
_orig_compute_sas = module._compute_sas
_orig_compute_sas_metal = module._compute_sas_metal
_spy_counts = {"ops": 0, "metal": 0}


def _spy_ops(*a, **k):
    _spy_counts["ops"] += 1
    return _orig_compute_sas(*a, **k)


def _spy_metal(*a, **k):
    _spy_counts["metal"] += 1
    return _orig_compute_sas_metal(*a, **k)


module._compute_sas = _spy_ops
module._compute_sas_metal = _spy_metal
try:
    _spy_counts["ops"] = 0
    _spy_counts["metal"] = 0
    ground_locg_mlx(
        apply_h_xz_mlx, inputs32_v, args=(xs32_i, dg32_i), maxiter=2, tol=0.0, sas="metal"
    )
    assert _spy_counts["metal"] > 0 and _spy_counts["ops"] == 0, (
        f"sas='metal' did not dispatch exclusively to _compute_sas_metal: {_spy_counts}"
    )
    metal_calls_for_2_iters = _spy_counts["metal"]

    _spy_counts["ops"] = 0
    _spy_counts["metal"] = 0
    ground_locg_mlx(
        apply_h_xz_mlx, inputs32_v, args=(xs32_i, dg32_i), maxiter=2, tol=0.0, sas="ops"
    )
    assert _spy_counts["ops"] > 0 and _spy_counts["metal"] == 0, (
        f"sas='ops' did not dispatch exclusively to _compute_sas: {_spy_counts}"
    )
finally:
    module._compute_sas = _orig_compute_sas
    module._compute_sas_metal = _orig_compute_sas_metal
print(
    f"OK  sas='metal'/sas='ops' dispatch exclusively to their own implementation "
    f"({metal_calls_for_2_iters} metal calls, 0 ops calls, for 2 iterations)"
)

# sas="metal" must refuse float64 rather than silently running a different kernel.
try:
    ground_locg_mlx(
        apply_h_xz_mlx, inputs.vinit, args=(inputs.xsources, inputs.diagonals), sas="metal"
    )
except ValueError as exc:
    assert "float32" in str(exc), f"unexpected guard message: {exc}"
    print("OK  ground_locg_mlx(sas='metal') rejects float64 input")
else:
    raise AssertionError("ground_locg_mlx(sas='metal') accepted float64 -- guard did not fire")

# An unknown sas value must fail loudly, not fall through to a default.
try:
    ground_locg_mlx(apply_h_xz_mlx, inputs32_v, args=(xs32_i, dg32_i), maxiter=2, sas="bogus")
except ValueError as exc:
    assert "bogus" in str(exc), f"unexpected message: {exc}"
    print("OK  ground_locg_mlx rejects an unknown sas value")
else:
    raise AssertionError("ground_locg_mlx accepted sas='bogus'")

# _compute_sas_metal must refuse a threadgroup exceeding _METAL_SAS_MAX_THREADGROUP (256), since
# the kernel's `partials` threadgroup array is sized by that literal at compile time. Mirrors the
# float64-guard case above in style.
_METAL_SAS_MAX_THREADGROUP = module._METAL_SAS_MAX_THREADGROUP
vecs32_tg = [vv.astype(np.float32) for vv in vecs]
mvs32_tg = [mm.astype(np.float32) for mm in mvs]
try:
    _compute_sas_metal(tuple(vecs32_tg[:2]), tuple(mvs32_tg[:2]), threadgroup=512)
except ValueError as exc:
    assert "256" in str(exc), f"unexpected guard message: {exc}"
    print("OK  _compute_sas_metal rejects threadgroup=512 (exceeds _METAL_SAS_MAX_THREADGROUP)")
else:
    raise AssertionError("_compute_sas_metal accepted threadgroup=512 -- guard did not fire")
_ = _compute_sas_metal(tuple(vecs32_tg[:2]), tuple(mvs32_tg[:2]), threadgroup=256)
print("OK  _compute_sas_metal still works at threadgroup=256")

# 3k. eigenpair_3x3_metal must agree with the op-graph eigenpair_3x3 on the matrix classes the
# solver actually produces, INCLUDING the rank-deficient and near-degenerate ones the seven-
# candidate search and the balancing exist for. This is the check that discriminates a transcription
# error in the fused kernel's algorithm: a wrong permutation in the characteristic polynomial, a
# dropped balancing step, or a tie-breaking difference in the argmax/argmin scans.
eigenpair_3x3_metal = module.eigenpair_3x3_metal
eigenpair_3x3_ops = module.eigenpair_3x3
_rng_eig = np.random.default_rng(20260805)


def _sym32(matrix):
    matrix = (matrix + matrix.T) * 0.5
    return matrix.astype(np.float32)


_eig_cases = []
# Generic well-separated spectra.
for _ in range(40):
    _eig_cases.append(("generic", _sym32(_rng_eig.normal(size=(3, 3)))))
# Large trace: the case balancing exists for (docs/locg.md I1/I2). Without balancing the
# characteristic polynomial's coefficients lose all significance and disc goes negative -> NaN.
for shift_mag in (1e3, 1e5, 1e7):
    _eig_cases.append(("large-trace", _sym32(_rng_eig.normal(size=(3, 3)) + shift_mag * np.eye(3))))
# Exactly degenerate and near-degenerate lowest pairs: rank-1 and rank-2 null spaces, which is
# what the seven-candidate search is for (item I3).
for eps in (0.0, 1e-7, 1e-4):
    basis = np.linalg.qr(_rng_eig.normal(size=(3, 3)))[0]
    for lows in ((1.0, 1.0 + eps, 5.0), (-2.0, -2.0 - eps, 3.0)):
        _eig_cases.append(("degenerate", _sym32(basis @ np.diag(lows) @ basis.T)))
# Multiples of the identity (rank 0 after the shift) and the exact zero matrix: every candidate
# collapses, so the e_0 fallback must win.
_eig_cases.append(("identity", np.eye(3, dtype=np.float32) * np.float32(3.5)))
_eig_cases.append(("zero", np.zeros((3, 3), dtype=np.float32)))
# A diagonal matrix carrying the p_is_zero exclusion shift that iter_body applies, which is
# max(diag_xy) + sum(|diag_xy|) + 1 -- bounded by the matrix's own scale, NOT an arbitrary huge
# value. Using 1e9 here instead would test something the solver never produces AND would fail for
# both implementations equally: after balancing by scale=1e9, -1.5 and -1.0 collapse to bit-
# identical float32 (measured difference exactly 0.0, against eps=1.19e-07), so the true minimum is
# unrecoverable in fp32 by any algorithm. The op-graph eigenpair_3x3 returns 0.0 for that input
# too. That is the documented fp32 dynamic-range limit of this balancing, not a kernel defect, so
# the case is written the way the solver actually generates it.
_excl_diag = np.array([-1.5, -1.0])
_excl = _excl_diag.max() + np.abs(_excl_diag).sum() + 1.0
_eig_cases.append(("excluded-p", np.diag([-1.5, -1.0, _excl]).astype(np.float32)))

_worst_eig = 0.0
_worst_case = None
for _label, _mat in _eig_cases:
    _th_ops, _kp_ops = eigenpair_3x3_ops(_mat)
    _th_met, _kp_met = eigenpair_3x3_metal(_mat)
    assert np.isfinite(_th_met), f"{_label}: metal eigenvalue is not finite ({_th_met})"
    assert np.all(np.isfinite(_kp_met)), f"{_label}: metal eigenvector is not finite ({_kp_met})"
    # Compare eigenvalues, which are basis-independent. The eigenvectors can legitimately differ
    # by sign, and for a degenerate lowest pair by an arbitrary rotation within the eigenspace, so
    # asserting on them directly would be wrong -- check the Rayleigh quotient instead.
    _den = max(1.0, abs(float(_th_ops)))
    _err = abs(float(_th_met) - float(_th_ops)) / _den
    if _err > _worst_eig:
        _worst_eig, _worst_case = _err, _label
    assert _err < 1e-5, f"{_label}: metal eigenvalue {_th_met} vs ops {_th_ops} (rel {_err:.2e})"
    # The returned vector must actually be a unit eigenvector for the returned eigenvalue: this is
    # what catches a null-vector selection bug that happens to leave the eigenvalue intact.
    _nrm = float(np.linalg.norm(_kp_met))
    assert abs(_nrm - 1.0) < 1e-5, f"{_label}: metal eigenvector norm {_nrm} != 1"
    # Normalize the residual by the MATRIX norm, not by |theta|: |Av - theta v| is an absolute
    # quantity whose float32 rounding floor scales with |A|, so dividing by a small |theta| on a
    # large-norm matrix would demand accuracy the arithmetic cannot deliver.
    _mnorm = max(1.0, float(np.abs(_mat).max()))
    _resid = np.linalg.norm(_mat @ _kp_met - float(_th_met) * _kp_met) / _mnorm
    assert _resid < 1e-4, f"{_label}: |Av - theta v|/|A| = {_resid:.2e} -- not an eigenpair"
print(
    f"OK  eigenpair_3x3_metal matches eigenpair_3x3 on {len(_eig_cases)} matrices spanning "
    f"generic/large-trace/degenerate/identity/zero (worst rel {_worst_eig:.2e} on {_worst_case}), "
    "and every returned pair satisfies |Av - theta v| ~ 0"
)

# The fused kernel must return the SMALLEST eigenvalue, not just some eigenvalue -- an independent
# check against numpy's eigh, since both rqutils paths share the same Cardano formulation and would
# agree with each other on a sign or root-selection error.
_worst_vs_eigh = 0.0
for _label, _mat in _eig_cases:
    _th_met, _ = eigenpair_3x3_metal(_mat)
    _ref = float(np.linalg.eigvalsh(_mat.astype(np.float64)).min())
    _err = abs(float(_th_met) - _ref) / max(1.0, abs(_ref))
    _worst_vs_eigh = max(_worst_vs_eigh, _err)
    assert _err < 1e-4, f"{_label}: metal {_th_met} is not the minimum eigenvalue {_ref}"
print(
    f"OK  eigenpair_3x3_metal returns the MINIMUM eigenvalue (vs numpy eigh, worst rel "
    f"{_worst_vs_eigh:.2e}) -- independent of the shared Cardano formulation"
)

# eigenpair_3x3_metal must refuse float64 rather than silently narrowing.
try:
    eigenpair_3x3_metal(np.eye(3, dtype=np.float64))
except ValueError as exc:
    assert "float32" in str(exc), f"unexpected guard message: {exc}"
    print("OK  eigenpair_3x3_metal rejects float64 input (Metal has no float64)")
else:
    raise AssertionError("eigenpair_3x3_metal accepted float64 input -- guard did not fire")

# 3l. eig="metal" end-to-end: same trajectory as eig="ops" in fixed-iteration mode, and dispatch
# proven by a call spy (the numerical check alone would pass a mis-wired call site, exactly as for
# sas above).
eig_e_ops, _, it_e_ops, _ = ground_locg_mlx(
    apply_h_xz_mlx, inputs32_v, args=(xs32_i, dg32_i), maxiter=30, tol=0.0, eig="ops"
)
eig_e_met, _, it_e_met, _ = ground_locg_mlx(
    apply_h_xz_mlx, inputs32_v, args=(xs32_i, dg32_i), maxiter=30, tol=0.0, eig="metal"
)
assert it_e_ops == it_e_met == 30, f"iteration counts differ: {it_e_ops} vs {it_e_met}"
assert abs(eig_e_met - eig_e_ops) < 1e-5 * max(1.0, abs(eig_e_ops)), (
    f"eig='metal' changed the fixed-iteration eigenvalue: {eig_e_met} vs {eig_e_ops}"
)
print(
    f"OK  eig='metal' tracks eig='ops' (eig {eig_e_met:.6f} vs {eig_e_ops:.6f}, {it_e_met} iters)"
)

_orig_eig3 = module.eigenpair_3x3
_orig_eig3_metal = module.eigenpair_3x3_metal
_eig_spy = {"ops": 0, "metal": 0}


def _spy_eig_ops(*a, **k):
    _eig_spy["ops"] += 1
    return _orig_eig3(*a, **k)


def _spy_eig_metal(*a, **k):
    _eig_spy["metal"] += 1
    return _orig_eig3_metal(*a, **k)


module.eigenpair_3x3 = _spy_eig_ops
module.eigenpair_3x3_metal = _spy_eig_metal
try:
    _eig_spy["ops"] = _eig_spy["metal"] = 0
    ground_locg_mlx(
        apply_h_xz_mlx, inputs32_v, args=(xs32_i, dg32_i), maxiter=2, tol=0.0, eig="metal"
    )
    assert _eig_spy["metal"] > 0 and _eig_spy["ops"] == 0, (
        f"eig='metal' did not dispatch exclusively to eigenpair_3x3_metal: {_eig_spy}"
    )
    _eig_metal_calls = _eig_spy["metal"]

    _eig_spy["ops"] = _eig_spy["metal"] = 0
    ground_locg_mlx(
        apply_h_xz_mlx, inputs32_v, args=(xs32_i, dg32_i), maxiter=2, tol=0.0, eig="ops"
    )
    assert _eig_spy["ops"] > 0 and _eig_spy["metal"] == 0, (
        f"eig='ops' did not dispatch exclusively to eigenpair_3x3: {_eig_spy}"
    )
finally:
    module.eigenpair_3x3 = _orig_eig3
    module.eigenpair_3x3_metal = _orig_eig3_metal
print(
    f"OK  eig='metal'/eig='ops' dispatch exclusively to their own implementation "
    f"({_eig_metal_calls} metal calls, 0 ops calls, for 2 iterations)"
)

# eig="metal" must refuse an f64 solve, and an unknown value must fail loudly.
try:
    ground_locg_mlx(
        apply_h_xz_mlx, inputs.vinit, args=(inputs.xsources, inputs.diagonals), eig="metal"
    )
except ValueError as exc:
    assert "float32" in str(exc), f"unexpected guard message: {exc}"
    print("OK  ground_locg_mlx(eig='metal') rejects float64 input")
else:
    raise AssertionError("ground_locg_mlx(eig='metal') accepted float64 -- guard did not fire")

try:
    ground_locg_mlx(apply_h_xz_mlx, inputs32_v, args=(xs32_i, dg32_i), maxiter=2, eig="bogus")
except ValueError as exc:
    assert "bogus" in str(exc), f"unexpected message: {exc}"
    print("OK  ground_locg_mlx rejects an unknown eig value")
else:
    raise AssertionError("ground_locg_mlx accepted eig='bogus'")

# 3j. The r_is_zero / seed_converged guard (ground_locg_mlx.py:503-517): a one-hot xinit against a
# diagonal operator is an exact eigenvector, so the residual after the seed step is exactly zero.
# Without the guard, eigenpair_2x2 sees a sas_mat whose row/col 1 (the p direction) vanishes and
# selects that null direction, collapsing theta towards 0 instead of reporting the true diagonal
# entry. Ports tests/test_ground_locg.py::TestZeroResidualAfterSeedStep's
# test_diagonal_operator_one_hot_xinit fixture -- this path had zero coverage in either MLX
# checker despite being renamed (sas -> sas_mat) in this task. sqd.py's diagonal-Hamiltonian path
# produces exactly this input shape, so it is not a contrived corner case.
#
# Index 0 and an interior index (5) are both covered, mirroring the JAX fixture's parametrize:
# nothing in the seed step singles out position 0, but the original audit's reproduction used
# index 0, so an interior index guards against the guard being accidentally position-dependent.
diag = np.arange(1.0, 61.0)
diag32 = diag.astype(np.float32)


def _diag_matvec(vec, dvec):
    return vec * dvec


for index in (0, 5):
    for dtype_name, dvec, sas_kw in (
        ("float64/ops", diag, "ops"),
        ("float32/metal", diag32, "metal"),
    ):
        one_hot = np.zeros(60, dtype=dvec.dtype)
        one_hot[index] = 1.0
        eig_seed, vec_seed, iters_seed, converged_seed = ground_locg_mlx(
            _diag_matvec, one_hot, args=(dvec,), sas=sas_kw
        )
        assert converged_seed is True, f"{dtype_name} index={index}: not converged at seed step"
        assert iters_seed == 0, f"{dtype_name} index={index}: expected 0 iters, got {iters_seed}"
        err_eig = abs(eig_seed - diag[index])
        assert err_eig < 1e-5, (
            f"{dtype_name} index={index}: eig={eig_seed}, expected diag[{index}]={diag[index]} "
            f"(err {err_eig})"
        )
        expected_vec = np.zeros(60)
        expected_vec[index] = 1.0
        err_vec = np.abs(np.asarray(vec_seed) - expected_vec).max()
        assert err_vec < 1e-5, (
            f"{dtype_name} index={index}: eigenvector drifted from the one-hot seed by {err_vec}"
        )
        print(
            f"OK  r_is_zero guard ({dtype_name}): one-hot seed at index {index} against a "
            f"diagonal operator converges in 0 iterations with eig={eig_seed}"
        )

# 3k. Large-magnitude seed-guard cases. 3j's diagonal entries are O(1-60), so a theta collapsed
# towards 0 by a defeated guard and the true theta are hard to tell apart numerically. At 1e9 they
# are unmistakable -- this is what actually discriminates a scaling bug in the guard's `excluded =
# mx.abs(rho) * 2.0 + 1.0` (ground_locg_mlx.py:511): that expression's `+ 1.0` term is negligible
# at rho~1e9 and dominant at rho~8, so a scaling error there would be invisible in 3j alone.
# Mirrors tests/test_ground_locg.py::TestZeroResidualAfterSeedStep::test_one_by_one_large_magnitude.
#
# 1x1 case, sas="ops" (float64): the JAX fixture uses rel=1e-13. float64 has ~15-16 significant
# decimal digits, so 1e-13 leaves comfortable headroom above eps (~2.2e-16) for the Rayleigh
# quotient's rounding.
eig_1x1, vec_1x1, iters_1x1, converged_1x1 = ground_locg_mlx(
    _diag_matvec, np.array([1.0]), args=(np.array([1e9]),), sas="ops"
)
assert converged_1x1 is True, "1x1 large-magnitude (ops): not converged at seed step"
assert iters_1x1 == 0, f"1x1 large-magnitude (ops): expected 0 iters, got {iters_1x1}"
rel_err_1x1 = abs(eig_1x1 - 1e9) / 1e9
assert rel_err_1x1 < 1e-13, f"1x1 large-magnitude (ops): rel err {rel_err_1x1}, eig={eig_1x1}"
err_vec_1x1 = abs(float(np.asarray(vec_1x1)[0]) - 1.0)
assert err_vec_1x1 < 1e-13, f"1x1 large-magnitude (ops): eigenvector drifted by {err_vec_1x1}"
print(
    f"OK  1x1 large-magnitude seed guard (ops, float64): eig={eig_1x1:.1f} "
    f"(rel err {rel_err_1x1:.1e})"
)

# 1x1 case, sas="metal" (float32): float32 has ~7 significant decimal digits (eps ~1.19e-7), so a
# float64-scale rel=1e-13 tolerance is unmeetable by construction -- assert against what f32
# arithmetic can actually deliver instead (a small multiple of eps), not a borrowed f64 bound.
eig_1x1_f32, _, iters_1x1_f32, converged_1x1_f32 = ground_locg_mlx(
    _diag_matvec,
    np.array([1.0], dtype=np.float32),
    args=(np.array([1e9], dtype=np.float32),),
    sas="metal",
)
assert converged_1x1_f32 is True, "1x1 large-magnitude (metal): not converged at seed step"
assert iters_1x1_f32 == 0, f"1x1 large-magnitude (metal): expected 0 iters, got {iters_1x1_f32}"
rel_err_1x1_f32 = abs(eig_1x1_f32 - 1e9) / 1e9
assert rel_err_1x1_f32 < 1e-5, (
    f"1x1 large-magnitude (metal): rel err {rel_err_1x1_f32}, eig={eig_1x1_f32}"
)
print(
    f"OK  1x1 large-magnitude seed guard (metal, float32): eig={eig_1x1_f32:.1f} "
    f"(rel err {rel_err_1x1_f32:.1e})"
)

# Large-magnitude diagonal/one-hot case: same shape as 3j but scaled by 1e9, so the seed-step
# guard's excluded-diagonal arithmetic runs at rho~O(1e9-6e10) instead of rho~O(1-60).
diag_big = diag * 1e9
diag_big32 = diag_big.astype(np.float32)
for index in (0, 5):
    for dtype_name, dvec, sas_kw, rel_tol in (
        ("float64/ops", diag_big, "ops", 1e-12),
        ("float32/metal", diag_big32, "metal", 1e-5),
    ):
        one_hot = np.zeros(60, dtype=dvec.dtype)
        one_hot[index] = 1.0
        eig_big, _, iters_big, converged_big = ground_locg_mlx(
            _diag_matvec, one_hot, args=(dvec,), sas=sas_kw
        )
        assert converged_big is True, f"{dtype_name} index={index}: large-mag not converged"
        assert iters_big == 0, f"{dtype_name} index={index}: large-mag expected 0 iters"
        rel_err_big = abs(eig_big - diag_big[index]) / abs(diag_big[index])
        assert rel_err_big < rel_tol, (
            f"{dtype_name} index={index}: large-mag eig={eig_big}, expected "
            f"diag_big[{index}]={diag_big[index]} (rel err {rel_err_big})"
        )
        print(
            f"OK  large-magnitude diagonal seed guard ({dtype_name}): index={index} "
            f"eig={eig_big:.3e} (rel err {rel_err_big:.1e})"
        )

# 3l. Metal Shading Language reserved-identifier scan over the kernel SOURCE TEXT.
#
# This is the one check here that reads the kernels' `source` strings rather than the shim's
# reimplementation of them. It exists because a real-device run found `uint half = lanes / 2` in
# _METAL_SAS_SOURCE: `half` is a reserved built-in scalar type in MSL (16-bit float), so that line
# failed to compile with "cannot combine with previous 'type-name' declaration specifier" and
# cascaded into eight further errors. Nothing headless could have caught it -- the numpy shim never
# compiles the Metal text, and `half` is a perfectly ordinary Python identifier.
#
# A name scan cannot verify that the Metal compiles (only hardware can), but it does pin the
# specific failure mode that got through, which is the whole point of a regression check.
_MSL_RESERVED = {
    # Scalar and vector built-in types that are also legal Python/C identifiers.
    "half",
    "short",
    "long",
    "char",
    "uchar",
    "ushort",
    "ulong",
    "size_t",
    "ptrdiff_t",
    "float2",
    "float3",
    "float4",
    "int2",
    "int3",
    "int4",
    "uint2",
    "uint3",
    "uint4",
    "bool2",
    "bool3",
    "bool4",
    "half2",
    "half3",
    "half4",
    # Address-space, function and resource qualifiers.
    "device",
    "constant",
    "threadgroup",
    "thread",
    "kernel",
    "vertex",
    "fragment",
    "matrix",
    "sampler",
    "texture",
    "atomic",
    "simdgroup",
}
_kernel_sources = {
    "_METAL_SAS_SOURCE": module._METAL_SAS_SOURCE,
    "_METAL_MATVEC_SOURCE": module._METAL_MATVEC_SOURCE,
    "_METAL_EIG3_SOURCE": module._METAL_EIG3_SOURCE,
}
for _src_name, _src_text in _kernel_sources.items():
    # Strip // comments first: prose legitimately discusses `half` to explain this very bug.
    _code_only = re.sub(r"//[^\n]*", "", _src_text)
    _declared = set(re.findall(r"\b(?:uint|int|float|bool|T)\s+(\w+)\s*(?:=|;|\[)", _code_only))
    _collisions = sorted(_declared & _MSL_RESERVED)
    assert not _collisions, (
        f"{_src_name} declares variable(s) named after MSL reserved types: {_collisions}. "
        "Metal will fail to compile this with a 'cannot combine with previous type-name' error. "
        "Rename them (e.g. `half` -> `stride`)."
    )
    print(f"OK  {_src_name} declares no MSL-reserved identifiers ({len(_declared)} names checked)")

# Every math function called from a kernel must be namespace-qualified (`metal::sqrt`, not bare
# `sqrt`). Metal Shading Language puts these in the `metal` namespace, and MLX wraps the `source`
# text in a function body without a `using namespace metal;`, so an unqualified call fails to
# compile -- another defect class the numpy shim is structurally blind to, since Python resolves
# `np.sqrt` regardless of what the Metal text says. _METAL_EIG3_SOURCE is the first kernel here to
# call any math function at all, so this had no in-repo precedent to copy.
_MSL_MATH = (
    "sqrt",
    "abs",
    "max",
    "min",
    "cos",
    "sin",
    "atan2",
    "atan",
    "exp",
    "log",
    "pow",
    "fabs",
    "fmax",
    "fmin",
    "hypot",
    "rsqrt",
    "floor",
    "ceil",
)
for _src_name, _src_text in _kernel_sources.items():
    _code_only = re.sub(r"//[^\n]*", "", _src_text)
    _unqualified = set()
    for _fn in _MSL_MATH:
        # A call not preceded by `metal::` (or by another identifier character, which would make it
        # part of a longer name such as `best_colnorm` or a member call).
        for _match in re.finditer(rf"(?<![\w:]){_fn}\s*\(", _code_only):
            _start = _match.start()
            if not _code_only[:_start].endswith("metal::"):
                _unqualified.add(_fn)
    assert not _unqualified, (
        f"{_src_name} calls math function(s) {sorted(_unqualified)} without the `metal::` "
        "namespace qualifier. MLX does not emit `using namespace metal;`, so Metal will fail to "
        "compile these. Prefix them with `metal::`."
    )
print(f"OK  all {len(_kernel_sources)} kernel sources qualify their math calls with `metal::`")

print("\nALL STATIC CHECKS PASSED (numpy shim; MLX itself still unverified)")
