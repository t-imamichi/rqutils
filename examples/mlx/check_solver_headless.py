"""Verify ground_locg_mlx without a Metal device.

Parses the module, checks its API, then re-executes its source with a numpy shim bound to
the name `mx` to validate the numerics. This catches algorithm transcription errors -- the
kind that matter most -- without needing MLX to initialize.

Run with:
    uv run python examples/mlx/check_solver_headless.py

This script does NOT require MLX or a Metal device: it substitutes a numpy shim for
`mlx.core` and re-executes solver.py's own source text against that shim. See
examples/mlx/check_solver_device.py for the real-MLX counterpart, which needs MLX itself
(importing mlx.core initializes a Metal device) and so cannot run headless.

The module under test is a deprecated reference implementation -- the JAX solver is faster even
on the MLX GPU backend -- so this checker's job is to keep it *correct*, not to compare
configurations. There is only one path left to check.

Scope, after the deprecation trimmed this file: what remains either compares against an INDEPENDENT
reference (the dense H, numpy's eigvalsh, the pinned eigenvalue) or pins a guard that was measured to
fail silently (the r_is_zero seed cases). Dropped as no-longer-discriminating: the f32 duplicates of
every seed case (f32 existed only because the removed Metal path was f32-only, and it catches nothing
f64 misses -- verified by neutering the guard), the anti-revert name/parameter absence lists, and a
fixed-iteration case that re-ran an identical solve. Prefer adding an independent reference over a
self-consistency check if you extend this.

What this file DOES catch, verified by reverting each guard in place (the only valid way to check --
see CLAUDE.md): _nullvec_3x3's rank-aware selection and its zero-collapse penalty, eigenpair_3x3's
balancing and closing Rayleigh polish, eigenpair_2x2's balancing, the r_is_zero seed guard, the
`excluded` scaling, and the chunked matvec's indexing. Eight for eight.

Two KNOWN GAPS, both pre-existing (confirmed against the pre-deprecation checker, which was equally
green on them) -- recorded rather than papered over, since a passing suite is not evidence a guard is
dead:

* **I5**, the ``tmp_t`` re-orthogonalization: neutering it leaves everything green. CLAUDE.md already
  documents why -- the drift it prevents needed I4's 2000-iteration runs to develop, so no test at
  this scale discriminates it. Keep the guard.
* **I6/I7**, ``_project_out``'s final zeroing and the ``p_is_zero`` exclusion mask: reachable only
  when the search direction actually collapses, which none of the problems here produce. Constructing
  such an input is the open piece of work if this file is ever extended.
"""

import ast
import inspect
import os
import sys
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# The module under test sits beside this checker: it was moved out of the package into examples/mlx/
# when the MLX port was deprecated, which is why this is a plain local join rather than the
# two-levels-up walk to rqutils/ that it used to be.
SRC = os.path.join(HERE, "solver.py")
with open(SRC) as source_file:
    source = source_file.read()

# 1. It must parse and define the two names its callers import.
#
# There used to be an accompanying list of ~8 REMOVED names asserted absent -- fused Metal kernels,
# their builders, and matvec variants deleted as measured losses. Those tripwires existed to defend
# a live optimization history against a well-meaning revert. The module is deprecated and nobody is
# extending it, so a name that reappeared would be someone deliberately reviving the port, which is
# a decision to make in review rather than a regression to catch here. docs/mlx-metal-kernels.md
# records what each was worth, which is the durable form of that warning.
tree = ast.parse(source)
defined = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
for name in ("apply_h_xz", "ground_locg_mlx"):
    assert name in defined, f"{name} not defined in {SRC}"
print("OK  module parses and defines its public surface")

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


module = types.ModuleType("ground_locg_mlx_shimmed")
module.__dict__["np"] = np
sys.modules["mlx"] = types.ModuleType("mlx")
sys.modules["mlx.core"] = shim
# Executing the module's own source against a numpy shim is this script's entire purpose: it
# validates the MLX port's algorithm on a machine with no Metal device.
exec(compile(source, SRC, "exec"), module.__dict__)  # noqa: S102
print("OK  module executes against the numpy shim")

apply_h_xz = module.apply_h_xz
ground_locg_mlx = module.ground_locg_mlx

sig = inspect.signature(ground_locg_mlx)
# The signature is the caller-facing contract: bench.py and check_solver_device.py both call this by
# keyword, so a renamed or dropped parameter breaks them. (The companion list asserting the removed
# knobs -- sas/eig/compile_body/compile_chunk/device -- is gone for the same reason as the removed
# function names above: it defended an option surface that no longer has anything to select.)
for param in ("mat", "xinit", "args", "maxiter", "tol"):
    assert param in sig.parameters, f"ground_locg_mlx missing parameter {param}"
print("OK  ground_locg_mlx signature is (mat, xinit, args, maxiter, tol)")

# 3. Numerics, against the same problem Task 1 verified.
import jax

jax.config.update("jax_enable_x64", True)
sys.path.insert(0, HERE)
from _bench_common import build_solver_inputs, dense_reference, generate_problem

ps, cs, states = generate_problem(10, 20, 200, seed=1)
inputs = build_solver_inputs(ps, cs, states)
H, ref = dense_reference(inputs)

# 3a. matvec must equal H @ v. The single most load-bearing check in this file: an independent
# dense reference, not self-consistency.
rng = np.random.default_rng(7)
v = rng.normal(size=inputs.subspace_dim)
got = np.asarray(apply_h_xz(v, inputs.xsources, inputs.diagonals))
err = np.abs(got - H @ v).max()
assert err < 1e-12, f"apply_h_xz disagrees with H @ v by {err}"
print(f"OK  matvec matches H @ v (max err {err:.2e})")

# 3b. full solve must reach the reference eigenvalue
eigval, eigvec, iters, converged = ground_locg_mlx(
    apply_h_xz, inputs.vinit, args=(inputs.xsources, inputs.diagonals)
)
assert abs(eigval - ref) < 1e-9 * max(1.0, abs(ref)), f"solver got {eigval}, reference {ref}"
assert 0 < iters <= 1000, f"implausible iteration count {iters}"
print(f"OK  solve: eig={eigval:.12f} ref={ref:.12f} iters={iters}")

# 3c. tol=0. must run exactly maxiter iterations, AND land on the pinned eigenvalue.
#
# The iteration count checks that tol=0. selects the fixed-iteration compiled branch (compilation is
# unconditional, so this is the only branch that skips the convergence sync). The literal is a
# REGRESSION PIN, captured from the pre-refactor `compile_body=True, tol=0.0` run: it survives
# refactors that a self-consistency check would wave through, since a self-comparison passes even
# when both sides drift together. The shim stubs mx.compile as identity, so this validates the
# control flow rather than real MLX compilation. If it fires, the loop changed its trajectory; do
# not update the literal without understanding why.
#
# These two assertions used to be separate cases (3c and 3f) running the identical call twice with
# identical arguments. Merged: one solve, both assertions.
_PINNED_FIXED_EIG = -2.496495741801  # 100 iterations, tol=0.0, f64, seed=1 problem
eigval_fixed, _, fixed_iters, _ = ground_locg_mlx(
    apply_h_xz, inputs.vinit, args=(inputs.xsources, inputs.diagonals), maxiter=100, tol=0.0
)
assert fixed_iters == 100, f"tol=0. ran {fixed_iters} iterations, expected exactly 100"
assert abs(eigval_fixed - _PINNED_FIXED_EIG) < 1e-11, (
    f"fixed-iteration eigenvalue drifted: {eigval_fixed:.12f} vs pinned {_PINNED_FIXED_EIG:.12f}"
)
print(
    f"OK  tol=0. gives fixed-iteration mode (zero syncs) and matches the pinned "
    f"eigenvalue ({eigval_fixed:.12f})"
)

# 3d. The chunked gather must agree with H @ v at every chunk size, and all chunk sizes must agree
# with each other. The second half matters because chunking is pure index arithmetic -- a reshape or
# flat-take bug would show up as a chunk-size-dependent answer while any single chunk still looked
# plausible. (This used to compare against a separate unchunked loop matvec, which is now gone; the
# cross-chunk comparison recovers that signal without it.)
per_chunk = {}
for chunk in (1, 4, 8, 16, 32, 128):
    got_chunked = np.asarray(apply_h_xz(v, inputs.xsources, inputs.diagonals, chunk))
    per_chunk[chunk] = got_chunked
    err_vs_h = np.abs(got_chunked - H @ v).max()
    assert err_vs_h < 1e-9, f"chunk={chunk}: disagrees with H @ v by {err_vs_h}"
spread = max(np.abs(per_chunk[c] - per_chunk[1]).max() for c in per_chunk)
assert spread < 1e-9, f"chunk sizes disagree with each other by {spread}"
print(f"OK  apply_h_xz matches H @ v for chunk in {{1,4,8,16,32,128}} (cross-chunk {spread:.2e})")

# 3e. eigenpair_3x3 against numpy's eigvalsh, over the matrix CLASSES the solver actually produces.
#
# This covers _nullvec_3x3's rank-aware seven-candidate search (item I3) and eigenpair_3x3's
# balancing (I1/I2). It is not redundant with the end-to-end solve above: the solve's own 3x3 matrices
# are generic and well-separated, so `return cands[mx.argmin(resid)]` -> `return cands[0]` leaves the
# full solve GREEN -- verified by making exactly that edit. The degenerate and rank-deficient classes
# below are what discriminate it.
#
# Restored deliberately. This coverage used to exist only as a side effect of comparing the fused
# Metal eigensolve against the op-graph one across these same classes; deleting the kernel would have
# silently taken the I3 guard's only test with it. eigvalsh is an INDEPENDENT reference -- comparing
# the two rqutils paths could never have caught a shared root-selection error anyway.
_rng_eig = np.random.default_rng(20260805)


def _sym(matrix):
    return (matrix + matrix.T) * 0.5


_eig_cases = [("generic", _sym(_rng_eig.normal(size=(3, 3)))) for _ in range(40)]
# Large trace: what balancing exists for. Without it the characteristic polynomial's coefficients
# lose all significance and the radicand goes negative -> NaN.
_eig_cases += [
    ("large-trace", _sym(_rng_eig.normal(size=(3, 3)) + mag * np.eye(3))) for mag in (1e3, 1e5, 1e7)
]
# Exactly degenerate and near-degenerate lowest pairs: rank-1 and rank-2 null spaces, which is what
# the seven-candidate search is for. For a degenerate eigenvalue the cross products do not vanish but
# decay to O(eps |M|^2), which no fixed magnitude cutoff separates from a genuine rank-2 result.
for _eps in (0.0, 1e-7, 1e-4):
    _basis = np.linalg.qr(_rng_eig.normal(size=(3, 3)))[0]
    for _lows in ((1.0, 1.0 + _eps, 5.0), (-2.0, -2.0 - _eps, 3.0)):
        _eig_cases.append(("degenerate", _sym(_basis @ np.diag(_lows) @ _basis.T)))
# Rank 0 after the shift: every candidate collapses, so the e_0 fallback must win.
_eig_cases.append(("identity", np.eye(3) * 3.5))
_eig_cases.append(("zero", np.zeros((3, 3))))
# A diagonal carrying the p_is_zero exclusion shift iter_body actually produces, which is bounded by
# the matrix's own scale rather than being an arbitrary huge value.
_excl_diag = np.array([-1.5, -1.0])
_eig_cases.append(
    ("excluded-p", np.diag([-1.5, -1.0, _excl_diag.max() + np.abs(_excl_diag).sum() + 1.0]))
)

_worst_eig, _worst_case = 0.0, None
for _label, _mat in _eig_cases:
    _theta, _kappa = module.eigenpair_3x3(_mat)
    assert np.isfinite(_theta), f"{_label}: eigenvalue is not finite ({_theta})"
    assert np.all(np.isfinite(_kappa)), f"{_label}: eigenvector is not finite ({_kappa})"
    _ref_min = float(np.linalg.eigvalsh(_mat).min())
    _err = abs(float(_theta) - _ref_min) / max(1.0, abs(_ref_min))
    if _err > _worst_eig:
        _worst_eig, _worst_case = _err, _label
    assert _err < 1e-9, f"{_label}: eigenpair_3x3 gave {_theta}, eigvalsh minimum is {_ref_min}"
    # The returned vector must be a unit eigenvector FOR that eigenvalue: this is what catches a
    # null-vector selection bug that happens to leave the eigenvalue intact.
    assert abs(float(np.linalg.norm(_kappa)) - 1.0) < 1e-9, f"{_label}: eigenvector not unit norm"
    _mnorm = max(1.0, float(np.abs(_mat).max()))
    _resid = float(np.linalg.norm(_mat @ _kappa - float(_theta) * _kappa)) / _mnorm
    assert _resid < 1e-8, f"{_label}: |Av - theta v|/|A| = {_resid:.2e} -- not an eigenpair"
print(
    f"OK  eigenpair_3x3 returns the minimum eigenpair on {len(_eig_cases)} matrices spanning "
    f"generic/large-trace/degenerate/identity/zero vs numpy eigvalsh "
    f"(worst rel {_worst_eig:.2e} on {_worst_case})"
)

# 3f. Two successive default solves must agree exactly -- no state leaks through _CONST_CACHE or
# _INDEX_CACHE, both module-level mutable dicts populated on first use.
eigval_default2, _, iters_default2, _ = ground_locg_mlx(
    apply_h_xz, inputs.vinit, args=(inputs.xsources, inputs.diagonals)
)
assert iters_default2 == iters, (
    f"repeated default solve changed its iteration count: {iters_default2} vs {iters} -- "
    "something in the module-level caches is not idempotent"
)
assert abs(eigval_default2 - eigval) < 1e-12
print(f"OK  repeated default solve is identical (iters={iters_default2}) -- caches are idempotent")

# 3j. The r_is_zero / seed_converged guard (grep `r_is_zero` in the module): a one-hot xinit against a
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
# f64 only. These cases used to run at both precisions, because f32 was the sole dtype the removed
# device="gpu" path accepted and the arm therefore had to exist. Measured after the kernels went:
# neutering the guard (`if r_is_zero:` -> `if False:`) is caught identically by both dtypes, in all
# four index/precision combinations, and so is a scaling bug in `excluded`. The f32 arm discriminated
# nothing the f64 arm missed, so it was duplication once its device rationale disappeared.
diag = np.arange(1.0, 61.0)


def _diag_matvec(vec, dvec):
    return vec * dvec


for index in (0, 5):
    one_hot = np.zeros(60)
    one_hot[index] = 1.0
    eig_seed, vec_seed, iters_seed, converged_seed = ground_locg_mlx(
        _diag_matvec, one_hot, args=(diag,)
    )
    assert converged_seed is True, f"index={index}: not converged at seed step"
    assert iters_seed == 0, f"index={index}: expected 0 iters, got {iters_seed}"
    err_eig = abs(eig_seed - diag[index])
    assert err_eig < 1e-5, (
        f"index={index}: eig={eig_seed}, expected diag[{index}]={diag[index]} (err {err_eig})"
    )
    expected_vec = np.zeros(60)
    expected_vec[index] = 1.0
    err_vec = np.abs(np.asarray(vec_seed) - expected_vec).max()
    assert err_vec < 1e-5, f"index={index}: eigenvector drifted from the one-hot seed by {err_vec}"
    print(
        f"OK  r_is_zero guard: one-hot seed at index {index} against a diagonal operator "
        f"converges in 0 iterations with eig={eig_seed}"
    )

# 3k. Large-magnitude seed-guard cases. 3j's diagonal entries are O(1-60), so a theta collapsed
# towards 0 by a defeated guard and the true theta are hard to tell apart numerically. At 1e9 they
# are unmistakable -- this is what actually discriminates a scaling bug in the guard's `excluded =
# mx.abs(rho) + mx.abs(rho) + 1.0` (grep `excluded` in the module): that `+ 1.0` term is negligible
# at rho~1e9 and dominant at rho~8, so a scaling error there would be invisible in 3j alone.
# Mirrors tests/test_ground_locg.py::TestZeroResidualAfterSeedStep::test_one_by_one_large_magnitude.
#
# f64 only here too, for the reason recorded above. 1x1 case: the JAX fixture uses rel=1e-13, and
# float64's ~15-16 significant decimal digits leave comfortable headroom above eps (~2.2e-16) for the
# Rayleigh quotient's rounding.
eig_1x1, vec_1x1, iters_1x1, converged_1x1 = ground_locg_mlx(
    _diag_matvec, np.array([1.0]), args=(np.array([1e9]),)
)
assert converged_1x1 is True, "1x1 large-magnitude: not converged at seed step"
assert iters_1x1 == 0, f"1x1 large-magnitude: expected 0 iters, got {iters_1x1}"
rel_err_1x1 = abs(eig_1x1 - 1e9) / 1e9
assert rel_err_1x1 < 1e-13, f"1x1 large-magnitude: rel err {rel_err_1x1}, eig={eig_1x1}"
err_vec_1x1 = abs(float(np.asarray(vec_1x1)[0]) - 1.0)
assert err_vec_1x1 < 1e-13, f"1x1 large-magnitude: eigenvector drifted by {err_vec_1x1}"
print(f"OK  1x1 large-magnitude seed guard: eig={eig_1x1:.1f} (rel err {rel_err_1x1:.1e})")

# Large-magnitude diagonal/one-hot: same shape as the group above but scaled by 1e9, so the
# guard's excluded-diagonal arithmetic runs at rho~O(1e9-6e10) instead of rho~O(1-60).
#
# Both indices are load-bearing, and index=0 alone would be VACUOUS at the small magnitudes above:
# replacing `excluded` with the constant 1.0 (dropping its rho-proportional term) returns exactly
# 1.0, which is the correct answer for diag[0]==1.0. Measured -- of the four small-magnitude
# index/dtype combinations, only index=5 catches that bug, while every large-magnitude case catches
# it at rel=1.0. This is why the 1e9 scaling exists.
diag_big = diag * 1e9
for index in (0, 5):
    one_hot = np.zeros(60)
    one_hot[index] = 1.0
    eig_big, _, iters_big, converged_big = ground_locg_mlx(_diag_matvec, one_hot, args=(diag_big,))
    assert converged_big is True, f"index={index}: large-mag not converged"
    assert iters_big == 0, f"index={index}: large-mag expected 0 iters"
    rel_err_big = abs(eig_big - diag_big[index]) / abs(diag_big[index])
    assert rel_err_big < 1e-12, (
        f"index={index}: large-mag eig={eig_big}, expected "
        f"diag_big[{index}]={diag_big[index]} (rel err {rel_err_big})"
    )
    print(
        f"OK  large-magnitude diagonal seed guard: index={index} "
        f"eig={eig_big:.3e} (rel err {rel_err_big:.1e})"
    )

print("\nALL STATIC CHECKS PASSED (numpy shim; MLX itself still unverified)")
