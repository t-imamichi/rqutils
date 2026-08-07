r"""
=========================================================
Single-vector LOBPCG on MLX (deprecated reference example)
=========================================================

**Deprecated.** This is a readable reference implementation, kept as an example rather than as
library code. The JAX implementation in ``rqutils.ground_locg`` measured faster than this port even
on the MLX GPU backend, so there is no configuration in which running it is the right choice. It
used to live at ``rqutils/ground_locg_mlx.py``; it was moved here because nothing in the package
imported it, no pytest exercises it, and it is not part of the published API.

Two fused Metal kernels (a gather-multiply-accumulate matvec and a 3x3 eigensolve) and the
``device="cpu"|"gpu"`` parameter that selected them were removed when the port was deprecated,
along with the two static MSL-source guards that covered them. What they were worth, the two
verified *negative* results (a fused Rayleigh-Ritz reduction that ran slower, and a host-side
per-iteration ``eigh`` that ran 1.37x slower), the full op-count history by backend, and the cost
model behind all of it are recorded in ``docs/mlx-metal-kernels.md``, now a historical document.

Overview
========

An MLX port of ``rqutils.ground_locg`` and of ``rqutils.sqd.apply_h`` at ``cache_level=(1, 2)``.
The algorithm is the same single-vector LOBPCG -- see ``rqutils.ground_locg`` for the
derivation, the Rayleigh-Ritz specialization, and the numerical analysis behind every guard
reproduced here.

``mlx`` is an optional, darwin-only extra (``uv run --extra mlx ...``); importing this module
without it raises ``ImportError``.

What remains is one path: the portable op-graph implementation, usable at float32 or float64, with
``mx.compile`` applied unconditionally (it was the single largest measured win, 2.71x, and was
never worth making selectable). Since the Metal kernels are gone, so is the float32-only
constraint they imposed -- an f64 solve is no longer a special case.

Relationship to the JAX implementation
======================================

The numerics were kept deliberately in step with ``rqutils.ground_locg``: both eigenpair
kernels balance before forming their characteristic polynomials, ``eigenpair_3x3`` takes its
null vector from the rank-aware ``_nullvec_3x3``, both close with a Rayleigh-quotient polish,
the iteration re-orthogonalizes ``t`` against the new ``x``, renormalizes the search direction,
masks a zeroed direction out of Rayleigh-Ritz contention, and reports that condition as
convergence. Those are items I1-I7 of ``docs/locg.md``; each was measured to fail *silently* --
a plausible wrong number rather than a raise or a ``NaN``.

The "when editing either file, change both" rule that used to govern this pair is **retired** with
the deprecation: ``rqutils.ground_locg`` is now free to evolve without a parity obligation
here. Every guard below is still load-bearing for *this* file, so don't strip them from it either;
just don't expect the two to stay aligned.

``docs/locg.md`` is a **stale historical audit** of the pre-rewrite JAX module: its line numbers
and severity rankings no longer apply, and it is cited here (and below) only for the I-numbers,
which remain a stable index into the measured failure modes.

Five structural differences, all forced by MLX:

* ``jax.lax.scan`` over X groups becomes a Python loop. J is static and small (tens), so
  unrolling into MLX's lazy graph is fine.
* ``jax.lax.while_loop`` becomes a Python loop. MLX has no while_loop/cond/scan, and reading
  the convergence flag forces ``mx.eval`` -- a device sync per iteration. That cost is real
  for an MLX user, not a measurement artifact. Pass ``tol=0.`` to skip the check entirely
  and get a sync-free fixed-iteration run.
* All ``out_sharding=`` arguments are dropped; MLX has unified memory and no sharding.
* The zero-residual guard at the seed step branches in Python on a synced ``float()`` rather
  than staying in the graph as a ``jnp.where``, since there is no ``lax.cond`` to fold it into.
* **Everything is real-valued.** MLX has no complex128 anywhere, so ``eigenpair_2x2`` and
  ``eigenpair_3x3`` drop the ``.conjugate()`` calls and real/imag norm splits their JAX
  counterparts need for the general complex-Hermitian case. ``ground_locg_mlx`` therefore
  **rejects complex input** rather than silently returning wrong eigenvalues: an earlier version
  of this port relied on an upstream even-Y constraint to keep coefficients real and admitted in
  its own docstring that nothing in the repo would catch a regression there.

The analytic 2x2/3x3 Rayleigh-Ritz step carries over directly, which is what made this port
feasible at all.

A sixth difference is not structural but a performance consequence of the same asymmetry, and it
is why several expressions here look less like their JAX counterparts than a straight
transcription would: **in JAX an unrolled Python loop over scalars is free, and in MLX it is
not.** XLA fuses ``jnp.stack([normalize(c) for c in cands])`` into the surrounding ``jit``, so
``rqutils.ground_locg`` pays nothing for building its seven null-vector candidates one at a
time; MLX constructs one lazily-evaluated op per call, so the identical source shape became 18%
of this port's entire per-iteration op count. The Rayleigh-Ritz step here therefore uses batched
whole-array forms -- one broadcast cross product for all three orthogonal-complement candidates,
one ``(7, 3)`` normalization, ``mx.diagonal``/``mx.roll`` instead of element-by-element
``mx.stack`` -- and hoists its dtype-dependent constants into ``_CONST_CACHE`` instead of
rebuilding them every iteration. Each was verified *bit-identical* to the form it replaced, not
merely close. Measured with ``examples/mlx/count_ops.py``, the LOBPCG body went from 116 to 65.0 op
constructions per iteration. These batched forms are an MLX-specific concern: porting them back to
JAX would be churn, since XLA already fuses what they hand-fuse.

Verification
============

MLX cannot initialize without a Metal device, which rules out headless testing.
``examples/mlx/check_solver_headless.py`` re-executes this module's source against a numpy shim
bound to the name ``mx`` (no MLX, no GPU -- this is what validates the algorithm and the
caller-facing contract), and ``examples/mlx/check_solver_device.py`` runs the real thing at both
precisions, which requires a real Metal device.
"""

import math

import mlx.core as mx
import numpy as np

_SQRT3 = math.sqrt(3.0)

# How many LOBPCG iterations run inside one compiled call before control returns to Python for a
# convergence check. Only relevant when `tol != 0.`; a fixed-iteration solve never checks and so
# never syncs. Checking after every iteration (chunk of 1) would sync as often as an uncompiled loop
# and defeat the point of compiling, so this trades a slightly coarser iteration count for ~10x
# fewer syncs -- see ground_locg_mlx's Returns section for what "coarser" costs. A module constant
# rather than a parameter: 10 was the default nothing in the tree ever overrode.
_COMPILE_CHUNK = 10

# Per-dtype constant cache.
#
# Every `mx.array(...)` call constructs a fresh array -- a host-to-device allocation and, in the
# lazy graph, another node -- so the scalar and small-vector constants the eigenpair kernels need
# were being rebuilt on every iteration. `eigenpair_3x3` alone constructed 7 of them per call
# (measured with examples/mlx/count_ops.py --by-op), which made it the single largest op-count
# contributor in the LOBPCG body. They depend only on the dtype, which is fixed for a whole solve,
# so they are built once per dtype here and reused.
#
# Keyed by the dtype object rather than its name so this works identically under real MLX and under
# the numpy shim in examples/mlx/check_solver_headless.py, whose dtypes are numpy types.
# Unbounded in principle, bounded by {float32, float64} in practice -- this module rejects complex
# input, so at most two entries are ever created.
_CONST_CACHE = {}


def _consts(dtype):
    """Return the cached constants for ``dtype``, building them on first use."""
    entry = _CONST_CACHE.get(dtype)
    if entry is not None:
        return entry
    entry = _CONST_CACHE[dtype] = {
        # Only values that must be ARRAYS of the solve dtype belong here -- chiefly mx.where
        # operands, which have to match their branch's dtype. A bare Python float is promoted
        # in place by mx.maximum/mx.minimum and needs no entry (see eigenpair_3x3's clamps).
        "one": mx.array(1.0, dtype),
        # eigenpair_3x3's root coefficients: the three Cardano roots are the same linear
        # combination a*cos(phi) + b*sin(phi) with these (a, b) pairs.
        "cphi_coeff": mx.array([2.0, -1.0, -1.0], dtype),
        "sphi_coeff": mx.array([0.0, -_SQRT3, _SQRT3], dtype),
        "eye3": mx.eye(3, dtype=dtype),
        # eigenpair_2x2's fallback eigenvector for a multiple of the identity.
        "e0_2": mx.array([1.0, 0.0], dtype),
        # iter_body's selector for the p direction's diagonal entry, used to lift a zeroed
        # search direction out of Rayleigh-Ritz contention (docs/locg.md item I7). A fixed
        # constant, so it does not need rebuilding per iteration.
        "p_mask": mx.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype),
        # The complement of p_mask, cached rather than recomputed as `1.0 - p_mask` inside the
        # per-iteration blend: it is loop-invariant, so building it in the hot path was one op
        # construction per iteration for a constant.
        "p_keep": mx.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 0.0]], dtype),
    }
    return entry


# Integer index arrays, cached the same way but keyed separately from the float constants above
# because they are dtype-independent: the same indices serve an f32 and an f64 solve.
_INDEX_CACHE = {}


def _indices():
    """Return the cached index arrays, building them on first use.

    ``lower_rows``/``lower_cols`` address a 3x3's strict lower triangle in the order (1,0), (2,0),
    (2,1) -- ``eigenpair_3x3``'s off-diagonal triple in one gather. ``col_pair`` rolls the columns
    by one so that a single batched cross product over ``(matT, matT[col_pair])`` yields
    ``_nullvec_3x3``'s three rank-2 candidates (c0xc1, c1xc2, c2xc0).
    """
    if not _INDEX_CACHE:
        _INDEX_CACHE["lower_rows"] = mx.array([1, 2, 2])
        _INDEX_CACHE["lower_cols"] = mx.array([0, 0, 1])
        _INDEX_CACHE["col_pair"] = mx.array([1, 2, 0])
    return _INDEX_CACHE


def normalize(vector, norm=None):
    """Divide by the norm, leaving a zero vector untouched instead of producing NaN.

    Pass ``norm`` when the caller has already computed it -- several call sites need the norm
    itself for a separate zero test, and recomputing it here would cost an extra reduction per
    iteration.

    Module-level, mirroring ``rqutils.ground_locg.normalize``. It was previously a closure inside
    ``ground_locg_mlx`` over that solve's cached ``one``, which meant the zero-norm guard --
    load-bearing against NaN, and one of the invariants ``docs/locg.md`` covers -- had to be
    re-inlined at every other call site rather than shared.
    """
    if norm is None:
        norm = mx.linalg.norm(vector)
    return vector / mx.where(norm == 0.0, _consts(norm.dtype)["one"], norm)


def apply_h_xz(vec, xsources, diagonals, chunk=16):
    """Return Hv from precomputed X sources and diagonals, via chunked batched gather.

    Mirrors ``rqutils.sqd.apply_h`` at ``cache_level=(1, 2)``. ``xsources`` must already be
    sanitized (no negative entries, with the corresponding diagonals zeroed) -- see
    ``examples/mlx/_bench_common.build_solver_inputs``.

    The obvious formulation loops over the J X-groups doing one ``take``+multiply+add per group,
    which is 3J Python-level MLX op constructions per matvec. Since ``xsources``/``diagonals`` are
    dense ``(J, N)`` arrays, groups are instead processed in chunks: gather a whole chunk with one
    flat ``take``, reshape, and reduce with a single weighted sum, cutting the op count from
    ``3*J`` to roughly ``3*ceil(J/chunk)``. The group-at-a-time version is gone -- it measured
    8.106 ms/iter against 4.393 for this one, the slowest of every configuration tried
    (``docs/mlx-metal-kernels.md``).

    ``chunk`` bounds the size of the gathered temporary to ``chunk * N`` elements, rather than
    the full ``(J, N)`` a fully-batched (``chunk=J``) version would materialize -- at the large N
    this benchmark is ultimately meant to probe (SQD's matrix-free design targets N up to
    ~10**7), a full ``(J, N)`` gather would cost ~8 GB versus ~80 MB for the unchunked
    group-at-a-time loop. This is a deliberate tradeoff, not an oversight: do not "simplify"
    this to full batching (``chunk=J``).

    Op-count / temporary-size tradeoff measured at J=100:

    ========  ========================  ===================
    chunk     ops per matvec (3*ceil)   temporary (chunk*N)
    ========  ========================  ===================
    1 (loop)  300                       N        (baseline)
    8         39   (7.7x fewer)         8*N
    16        21   (14.3x fewer)        16*N   <- default
    32        12   (25x fewer)          32*N
    ========  ========================  ===================

    See ``examples/mlx/_bench_common.apply_h_xz_chunked`` for the JAX equivalent used by the JAX
    arms of the benchmark -- batching the gather is applied symmetrically to both frameworks so
    the comparison stays about the solver loop, not about who has the better matvec.

    Verified (design doc, Optimization 1): max abs diff across chunk in
    {1, 4, 8, 16, 32, 50, 100, 128} is <= 2.7e-15, and every chunk matches ``H @ v`` to 1.8e-15.

    Args:
        vec: The vector to multiply, shape ``(N,)``.
        xsources: Sanitized X-source indices, shape ``(J, N)``.
        diagonals: Sanitized diagonals, shape ``(J, N)``, matching ``vec``'s dtype.
        chunk: Number of X-groups to gather per flat ``take``. J is static and small, so this
            is a plain Python int controlling how many groups are unrolled per flat gather.

    Returns:
        ``H @ vec``.
    """
    num_groups = xsources.shape[0]
    out = mx.zeros_like(vec)
    for start in range(0, num_groups, chunk):
        xc = xsources[start : start + chunk]
        dc = diagonals[start : start + chunk]
        gathered = mx.take(vec, xc.reshape(-1)).reshape(xc.shape)
        out = out + mx.sum(gathered * dc, axis=0)
    return out


def ground_locg_mlx(mat, xinit, args=(), maxiter=1000, tol=None):
    """Single-vector LOBPCG in MLX.

    Args:
        mat: Callable mapping ``(vec, *args)`` to ``A @ vec`` -- ``apply_h_xz`` for an SQD
            projected Hamiltonian. This function never substitutes a matvec of its own.
        xinit: Initial vector. Must have nonvanishing overlap with the ground state.
        args: Extra arguments forwarded to ``mat``.
        maxiter: Maximum gradient-descent iterations.
        tol: Convergence tolerance. ``None`` uses the dtype epsilon. ``0.`` disables the
            check, running exactly ``maxiter`` iterations with no device sync at all.

    Returns:
        ``(eigenvalue, eigenvector, iterations, converged)``. Check the fourth value rather than
        comparing the third against ``maxiter``, which is ambiguous when convergence happens on
        the final permitted iteration.

        One caveat on ``iterations`` when ``tol != 0.``: the convergence test runs at
        ``_COMPILE_CHUNK`` boundaries, so a solve that terminates because its search direction
        zeroed out mid-chunk reports the boundary count rather than the exact iteration. Never
        wrong -- a zeroed direction means no further iteration can lower the eigenvalue, so the
        remainder of the chunk is wasted work, not incorrect work -- but ``iterations`` can round
        up to a multiple of ``_COMPILE_CHUNK`` in that case.

    Raises:
        ValueError: If ``xinit`` is complex. Every kernel here assumes real symmetric input.
    """
    xinit = mx.array(xinit)
    # Test the dtype's name rather than comparing against mx.complex64: this holds under real MLX
    # and under the numpy shim in examples/mlx/check_solver_headless.py, which defines only the
    # dtypes the port actually uses.
    if "complex" in str(xinit.dtype):
        # The kernels below drop the .conjugate() calls their JAX counterparts need, so complex
        # input would silently produce wrong eigenvalues. MLX has no complex128 at all, so this
        # port is real-only by construction -- fail loudly instead.
        raise ValueError(
            f"ground_locg_mlx is real-symmetric only, got {xinit.dtype}. MLX has no complex128 "
            "and the eigenpair kernels here omit the conjugations the complex case requires; use "
            "rqutils.ground_locg (JAX) for a complex-Hermitian operator."
        )
    check_convergence = tol != 0.0
    if tol is None:
        # Compare the dtype object directly rather than parsing its repr: this works
        # identically under real MLX and under the numpy shim used by the static check.
        tol = float(np.finfo(np.float32 if xinit.dtype == mx.float32 else np.float64).eps)

    def matvec(vec):
        return mat(vec, *args)

    # Constants fetched once per solve rather than rebuilt on every iter_body call (see _consts).
    # The dtype is fixed for the whole solve, so this lookup cannot change mid-run.
    _solve_consts = _consts(xinit.dtype)
    one = _solve_consts["one"]
    p_mask = _solve_consts["p_mask"]
    p_keep = _solve_consts["p_keep"]
    # Static Python int, fixed for the whole solve, so converged() does not need a vector argument
    # just to read a shape off it.
    num_states = xinit.shape[0]

    xinit = normalize(xinit)

    # Iteration 0: no previous direction yet. Keep ax -- iteration 1 reuses it (item S1).
    ax = matvec(xinit)
    rho = mx.sum(xinit * ax)
    xcurr = xinit
    rcurr = ax - rho * xcurr

    # Iteration 1: two-vector Rayleigh-Ritz over {x, r}.
    #
    # The zero-residual guard mirrors the JAX version's body_iter1: an exactly-zero residual means
    # xinit is already an eigenvector (sqd.py's diagonal-Hamiltonian path seeds exactly that), and
    # without the guard eigenpair_2x2 sees a sas_mat whose row/col 1 vanish and selects that null
    # direction, collapsing theta towards 0 instead of reporting rho -- the true answer.
    norm_r = mx.linalg.norm(rcurr)
    r_is_zero = float(norm_r) == 0.0
    tmp_p = normalize(rcurr, norm_r)
    # Reuse ax from iteration 0 rather than recomputing it inside compute_sas.
    sas_mat = _compute_sas((xcurr, tmp_p), (ax, matvec(tmp_p)))
    if r_is_zero:
        # Lift the p diagonal out of contention so Rayleigh-Ritz cannot pick the null direction.
        # With p excluded the 2x2 solve collapses onto x alone, giving theta = rho and kappa =
        # [1, 0], so xcurr is unchanged and no new search direction is introduced.
        #
        # Assign the entry directly, mirroring ground_locg's `sas.at[1, 1].set(excluded)`. MLX
        # supports item assignment, so the nested mx.stack tower this used to build (four ops to
        # add `excluded - sas_mat[1, 1]` into one slot) was both longer and *less* exact -- the
        # subtract-then-re-add does not cancel, measured at 5.7e-14 versus 0 for a direct set.
        # This branch runs once per solve, not per iteration, so no launch count depends on it.
        excluded = mx.abs(rho) + mx.abs(rho) + 1.0
        sas_mat[1, 1] = excluded
    theta, kappa = eigenpair_2x2(sas_mat)
    tmp_t = tmp_p * kappa[0] - xcurr * kappa[1]
    tmp_u = xcurr * kappa[0] + tmp_p * kappa[1]
    xcurr = normalize(tmp_u)
    # Re-orthogonalize for the same reason as in iter_body below (item I5).
    for _ in range(2):
        tmp_t = tmp_t - xcurr * mx.sum(xcurr * tmp_t)
    ycurr = normalize(tmp_t)
    axnext = matvec(xcurr)
    rcurr = axnext - theta * xcurr
    # A zeroed post-seed residual means {x} already spans the relevant space, so no further
    # iteration can lower theta: report convergence without entering the loop.
    seed_converged = r_is_zero

    def iter_body(xcurr, ycurr, rcurr, axcurr):
        """One LOBPCG step. Returns (theta, xcurr, ycurr, rcurr, axnext, p_is_zero)."""
        tmp_p = _project_out((xcurr, ycurr), rcurr)
        # _project_out guarantees only |tmp_p| >= 0.99, but the Rayleigh-Ritz step below solves a
        # standard eigenproblem and so assumes an orthonormal basis: a short tmp_p scales
        # sas_mat[2, 2] by |tmp_p|^2, which for a large positive shift is a spuriously low diagonal
        # that gets selected in place of the true minimizer (item I6).
        norm_p = mx.linalg.norm(tmp_p)
        p_is_zero = norm_p == 0.0
        tmp_p = normalize(tmp_p, norm_p)
        # xcurr's image is already known from the previous iteration -- three matvecs, not four.
        sas_mat = _compute_sas((xcurr, ycurr, tmp_p), (axcurr, matvec(ycurr), matvec(tmp_p)))
        # A zeroed tmp_p leaves sas_mat row/col 2 empty, and for a positive-definite A that zero
        # diagonal is the smallest eigenvalue, so Rayleigh-Ritz would pick the null direction and
        # the normalizations below would divide by zero. Lift it out of contention (item I7); the
        # p_is_zero case is reported as convergence by the caller.
        # The x/y diagonal via one strided read of the diagonal rather than two scalar gathers
        # plus a stack, and the [2, 2] selector mask from the per-dtype constant cache instead of
        # a fresh zeros_like plus an in-place write every iteration -- the mask is a fixed
        # constant, not a function of sas_mat.
        diag_xy = mx.diagonal(sas_mat)[:2]
        excluded = mx.max(diag_xy) + mx.sum(mx.abs(diag_xy)) + 1.0
        # ground_locg spells this `jnp.where(p_is_zero, sas.at[2, 2].set(excluded), sas)`. Here it
        # stays a masked blend rather than an item assignment: this runs every iteration inside
        # mx.compile, so the selection must remain part of the traced graph, and MLX has no
        # functional `.at[].set()` that composes with a traced predicate.
        sas_mat = mx.where(p_is_zero, sas_mat * p_keep + excluded * p_mask, sas_mat)
        theta, kappa = eigenpair_3x3(sas_mat)
        tmp_s = ycurr * kappa[1] + tmp_p * kappa[2]
        norm_s = mx.linalg.norm(tmp_s)
        tmp_t = tmp_s * (kappa[0] / mx.where(norm_s == 0.0, one, norm_s)) - xcurr * norm_s
        tmp_u = xcurr * kappa[0] + tmp_s
        xnext = normalize(tmp_u)
        # tmp_t is a difference of two quantities both nearly parallel to xcurr as norm_s -> 0, so
        # catastrophic cancellation lets ynext drift into xnext. Once <x|y> is O(1) the basis is no
        # longer orthonormal and the standard Rayleigh-Ritz above returns a theta *below* the true
        # minimum eigenvalue -- a silent wrong answer rather than a visible failure (item I5).
        for _ in range(2):
            tmp_t = tmp_t - xnext * mx.sum(xnext * tmp_t)
        ynext = normalize(tmp_t)
        axnext = matvec(xnext)
        rnext = axnext - xnext * theta
        return theta, xnext, ynext, rnext, axnext, p_is_zero

    def converged(theta, rcurr, axnext, p_is_zero):
        # A zeroed search direction means {x, y} already spans the residual: we are at a stationary
        # point and no further iteration can lower theta. Test it FIRST, before building reltol --
        # that early return discards reltol's two op constructions, so computing them above the
        # branch was pure waste on the terminating iteration.
        #
        # This bool() forces a device sync -- the price of MLX having no while_loop.
        if bool(p_is_zero):
            return True
        # Compare the residual norm against the floating-point error we'd expect from forming the
        # residual at all. abs(theta) rather than +theta, and a sum rather than the difference the
        # first version of this port inherited: norm(Ax) - theta is a cancellation of two nearly
        # equal large positive numbers for a positive-definite operator, which was measured going
        # negative and made the test unsatisfiable, so the solver never converged and always burned
        # maxiter (item I4). A ground-state search is typically negative-definite, where +theta
        # would cancel in turn.
        reltol = (mx.linalg.norm(axnext) + mx.abs(theta)) * num_states * 10
        # ONE float() sync, not two: the comparison is folded into a single on-device subtraction
        # and only its sign is read back. `norm(r) < tol * reltol` as two separate float() calls
        # drains the MLX pipeline twice per iteration, and syncs are the top item in this repo's
        # cost model (see docs/mlx-metal-kernels.md). Exactly
        # equivalent -- 0 disagreements over 400k random magnitude pairs spanning 1e-30..1e10 in
        # both f32 and f64, since tol is a Python float and both operands are non-negative.
        return float(mx.linalg.norm(rcurr) - tol * reltol) < 0.0

    niter = 0
    is_converged = False
    if seed_converged:
        # {x} already spans the residual; the loop cannot improve on the seed pair.
        return float(theta), xcurr, 0, True
    # Compilation is unconditional. mx.compile was the single largest measured win here (2.71x on
    # the MLX CPU backend) and its trajectory was pinned bit-for-bit against the uncompiled one
    # before that path was removed, so there was no configuration in which not compiling was right.
    # What remains is the ONE distinction that is not a performance knob: whether a convergence
    # check happens at all. MLX has no while_loop/cond, so a convergence check is necessarily a
    # Python branch on a synced value, and mx.compile traces a fixed array-in/array-out computation
    # that cannot contain one. Hence two loop shapes rather than one.
    if not check_convergence:
        # tol=0.: no convergence check, so the compiled body runs every requested iteration with
        # ZERO device syncs -- the clean per-iteration measurement the benchmark reads. Compile
        # iter_body directly rather than a chunk: maxiter need not be a multiple of _COMPILE_CHUNK,
        # and with no sync to amortize there is nothing for chunking to buy.
        compiled_body = mx.compile(iter_body)
        for niter in range(1, maxiter + 1):
            theta, xcurr, ycurr, rcurr, axnext, _p_is_zero = compiled_body(
                xcurr, ycurr, rcurr, axnext
            )
    else:
        # Run _COMPILE_CHUNK iterations inside one compiled function, then sync once to check
        # convergence. This amortizes the unavoidable float()-sync over _COMPILE_CHUNK iterations
        # rather than paying it every iteration; syncs are the top item in this repo's cost model
        # (~389 us, ~53x a kernel launch -- see docs/mlx-metal-kernels.md).
        def chunk_body(xcurr, ycurr, rcurr, axcurr):
            # No theta/p_is_zero initializers: _COMPILE_CHUNK is a module constant of 10, so the
            # loop always runs at least once and any initial value would be dead. (They were not
            # merely redundant -- a zero-trip loop would have returned them, reporting theta = 0.)
            #
            # NOTE, a real behaviour consequence: only the LAST iteration's p_is_zero survives, so
            # a zeroed search direction arising mid-chunk is not reported until the next boundary.
            # Still correct, never wrong -- p_is_zero means no further iteration can lower theta, so
            # the rest of the chunk is wasted work rather than wrong work, is_converged is still
            # True, and only the reported niter can round up (see this function's Returns section).
            # Widening the compiled signature to an OR-accumulated flag would make niter exact at
            # the cost of an extra array in the traced carry; that trade was declined deliberately.
            for _ in range(_COMPILE_CHUNK):
                theta, xcurr, ycurr, rcurr, axcurr, p_is_zero = iter_body(
                    xcurr, ycurr, rcurr, axcurr
                )
            return theta, xcurr, ycurr, rcurr, axcurr, p_is_zero

        compiled_chunk = mx.compile(chunk_body)
        # `range`, not a hand-advanced `while niter < maxiter`: the old shape derived its progress
        # from `this_chunk = min(chunk, maxiter - niter)` and advanced by it three lines later, so a
        # non-positive chunk made the advance a no-op and the loop spun forever. Python's own range
        # semantics guarantee termination instead. That hang is now unconstructible anyway --
        # _COMPILE_CHUNK is a module constant -- which is why the guard that used to validate it is
        # gone, but keep this form rather than reintroducing the hand-advanced one.
        for start in range(0, maxiter, _COMPILE_CHUNK):
            this_chunk = min(_COMPILE_CHUNK, maxiter - start)
            if this_chunk == _COMPILE_CHUNK:
                theta, xcurr, ycurr, rcurr, axnext, p_is_zero = compiled_chunk(
                    xcurr, ycurr, rcurr, axnext
                )
            else:
                # Final partial chunk: run the body uncompiled so niter lands exactly on maxiter
                # rather than overshooting it.
                for _ in range(this_chunk):
                    theta, xcurr, ycurr, rcurr, axnext, p_is_zero = iter_body(
                        xcurr, ycurr, rcurr, axnext
                    )
            niter = start + this_chunk
            if converged(theta, rcurr, axnext, p_is_zero):
                is_converged = True
                break

    return float(theta), xcurr, niter, is_converged


def _compute_sas(vectors, mvs):
    """Return the (n x n) matrix of <v_i | A | v_j> for n in {2, 3}.

    Takes the images ``mvs`` rather than computing them, so a caller holding an already-known
    ``A x`` can pass it in instead of paying for a fourth matrix-vector product per iteration
    (``docs/locg.md`` item S1).

    All n^2 inner products come from ONE stacked ``(n, N) @ (N, n)`` matmul rather than n^2
    separate ``mx.sum(v1 * mv)`` reductions plus n+1 ``mx.stack`` calls -- 9 sums and 4.5 stacks
    per iteration at n=3 (measured, ``examples/mlx/count_ops.py --by-op``) collapsing to 3 ops.

    This is *more* accurate than the form it replaces, not merely faster: against a longdouble
    reference over 3000 random unit-norm triples at N=900, the matmul is exact (0.0 max error)
    while the n^2-sums version carries up to 7.1e-15, because BLAS accumulates in blocks
    (pairwise, error growing as log N) where ``mx.sum`` over a product array accumulates
    sequentially. The symmetrization is kept: it is what makes the two triangles agree exactly,
    and it costs 2 ops against the 3x3 result rather than anything O(N).

    Unlike ``_project_out``, this function carries no accumulation-order contract -- its own
    symmetrization already averages away the rounding difference between the two triangles -- so
    reassociating the sum here is safe in a way it would *not* be there. Do not apply the same
    transformation to ``_project_out``: those passes guard against catastrophic cancellation
    (``docs/locg.md`` items I5/I6) and a matmul reassociates their summation order, measured to
    shift results by ~1e-14.

    One visible consequence, expected and benign: because the accumulation order changes, **f32
    eigenvalues shift in their last one or two digits** relative to runs recorded before this
    change (e.g. -5.3960409164 versus -5.3960399628, ~1.8e-7 relative, against an f32 eps of
    1.19e-7). Iteration counts are unchanged and the correctness gate passes with a >500x margin at
    ``rtol=1e-4``. f64 is unaffected at the printed precision. Do not treat an f32 last-digit
    difference against a pre-matmul recorded value as a regression.
    """
    stacked_v = mx.stack(vectors)
    stacked_m = mx.stack(mvs)
    sas = stacked_v @ stacked_m.T
    # Symmetrize: the two triangles differ only by rounding for real symmetric A.
    return (sas + sas.T) * 0.5


def _project_out(basis, vector):
    """Orthogonalize ``vector`` against ``basis``, ending on a subtraction.

    The repeated passes and the final zeroing are load-bearing; see the comments in
    ``rqutils.ground_locg._project_out``, which this mirrors. Near convergence, ending on a
    normalization can reintroduce basis components through catastrophic cancellation and wreck the
    Rayleigh-Ritz conditioning.

    **Deliberately NOT batched into a matmul, unlike ``_compute_sas``.** The eight ``mx.sum``
    reductions here (four passes x two basis vectors, 13 ops/iteration total) could become one
    stacked ``B @ v`` per pass, which is what made ``_compute_sas`` 14.2 -> 2.3 ops. It is not
    done here because a matmul reassociates the summation order, and this function's whole purpose
    is resisting catastrophic cancellation (``docs/locg.md`` items I5/I6, both measured to fail
    silently). Tested rather than assumed: over 4000 adversarial cases with ``r`` placed almost
    entirely inside ``span(x, y)`` plus an orthogonal part of size 1e-14..1e-6, both forms hold
    residual orthogonality at machine epsilon, but the matmul is consistently *worse* -- worst
    ``|<b|p>|`` of 8.3e-17 versus 6.2e-17. That margin is small and neither form is broken, so this
    is a judgement call, not a measured failure: ~4 ops/iteration is not worth a 33% erosion of the
    quantity these guards exist to protect. Do not "optimize" this without re-running that
    comparison.
    """
    for _ in range(2):
        ips = [mx.sum(vb * vector) for vb in basis]
        for vb, ip in zip(basis, ips):
            vector = vector - vb * ip
        vector = normalize(vector)

    for _ in range(2):
        ips = [mx.sum(vb * vector) for vb in basis]
        for vb, ip in zip(basis, ips):
            vector = vector - vb * ip

    return vector * (mx.linalg.norm(vector) >= 0.99).astype(vector.dtype)


def eigenpair_2x2(mat):
    """Lowest eigenpair of a real symmetric 2x2 matrix.

    Mirrors ``rqutils.ground_locg.eigenpair_2x2``, including its balancing: the matrix is scaled
    by its largest entry and reduced to its traceless part before the quadratic is solved. Without
    that, ``tr*tr - 4*det`` cancels catastrophically for a large trace (relative error 5.8e-1 at
    shift 1e9) and the unbalanced intermediates overflow at extreme scale. The eigenvector row is
    selected on the sign of ``delta`` so ``delta`` is never cancelled against a nearly equal
    radius, and the closing Rayleigh quotient is second order in the eigenvector error. See
    ``docs/locg.md`` items I1/I2.

    Assumes ``mat`` is real symmetric, not just Hermitian: the JAX original's ``.conjugate()``
    calls are no-ops here and are dropped. ``ground_locg_mlx`` rejects complex input up front, so
    that assumption is enforced rather than merely documented.
    """
    consts = _consts(mat.dtype)
    scale = mx.max(mx.abs(mat))
    scale = mx.where(scale > 0.0, scale, consts["one"])
    balanced = mat / scale
    d = mx.diagonal(balanced)
    delta = (d[0] - d[1]) * 0.5
    offd = balanced[1, 0]
    rad = mx.sqrt(delta * delta + offd * offd)
    # Null vector of T + rad I = [[delta + rad, offd], [offd, rad - delta]], which is singular.
    # Row 1 gives [-offd, delta + rad] and row 2 gives [rad - delta, -offd]; the two are parallel,
    # but each cancels when its own pivot is small, so select on the sign of delta.
    vec = mx.where(
        delta >= 0.0,
        mx.stack([-offd, delta + rad]),
        mx.stack([rad - delta, -offd]),
    )
    # rad == 0 means a multiple of the identity, for which any unit vector is an eigenvector.
    norm = mx.linalg.norm(vec)
    vec = mx.where(
        norm > 0.0,
        vec / mx.where(norm > 0.0, norm, consts["one"]),
        consts["e0_2"],
    )
    # Rayleigh quotient: recovers full precision where the closed form alone reaches only
    # sqrt(eps), as for a near-degenerate lowest pair.
    return mx.sum(vec * (balanced @ vec)) * scale, vec


def _nullvec_3x3(mat):
    """Unit null vector of a singular real symmetric 3x3 matrix, robust to any rank.

    Mirrors ``rqutils.ground_locg._nullvec_3x3``. Seven candidates are offered and the one with
    the smallest residual ``|Mv|`` wins. Selecting on the measured residual rather than on a
    magnitude threshold matters because the rank-2 and rank-1 constructions fail in ways no fixed
    cutoff separates cleanly: for a degenerate eigenvalue the cross products do not vanish but
    decay only to O(eps |M|^2).
    """
    # Rank 2 (simple eigenvalue): the null vector is col_i x col_j. Any single pair can be rank
    # deficient, in which case its cross product points nowhere useful, so all three are offered.
    #
    # All three pairings go through ONE batched cross product over stacked (3, 3) operands rather
    # than three separate mx.linalg.cross calls. `matT[k]` is column k of `mat`, so the pairs are
    # (c0,c1), (c1,c2), (c2,c0) -- the same three, in the same order, as the unrolled version.
    matT = mat.T
    crosses = mx.linalg.cross(matT, matT[_indices()["col_pair"]])
    # Rank 1 (degenerate lowest eigenvalue): every cross product is numerical noise and the null
    # space is the orthogonal complement of the largest column; any member of it is an eigenvector.
    # mx.square alone, not mx.square(mx.abs(...)): |x|^2 == x^2 for real x, and this module rejects
    # complex input, so the abs the JAX original needs for the complex-Hermitian case is a pure
    # extra op launch here. Bit-identical, and argmax picks the same column -- verified at 0.0 max
    # difference over 5000 random matrices spanning scales 1e-8..1e8.
    col_index = mx.argmax(mx.sum(mx.square(mat), axis=0))
    col = mat[:, col_index]
    # The three complement vectors are exactly `col x e_k` for the standard basis vectors e_k:
    # col x e_0 = [0, col_2, -col_1], col x e_1 = [-col_2, 0, col_0], col x e_2 = [col_1, -col_0, 0],
    # which is what the unrolled version built by indexing out individual scalars and negating them
    # one at a time -- roughly a dozen launches for nine numbers. One broadcast cross replaces all
    # of it, bit-identically (verified at 0.0 max relative difference over 3000 random matrices
    # spanning scales 1e-8..1e8, including rank-deficient ones with a zeroed column).
    consts = _consts(mat.dtype)
    eye = consts["eye3"]
    comps = mx.linalg.cross(mx.broadcast_to(col, (3, 3)), eye)
    # Rank 0 (a multiple of the identity): every candidate above is zero, so offer an arbitrary
    # unit vector as the last resort. It has residual 0 and wins by default.
    cands = [crosses, comps, eye[:1]]

    # Stack FIRST, then normalize the whole (7, 3) block in one pass. Normalizing candidate by
    # candidate -- what the JAX original does, and what this port copied -- costs 3 op launches per
    # candidate, 21 per iteration (measured: 18% of the whole LOBPCG body, see
    # examples/mlx/count_ops.py). In JAX that loop is free because XLA fuses it into the surrounding
    # jit; in MLX every op is its own launch, so the identical source shape has a completely
    # different cost. The batched form is *bit-identical* to the per-candidate one, not merely close
    # -- verified at 0.0 max abs difference over 2000 random triples spanning scales 1e-12..1e12
    # plus the exact-zero row the guard below exists for -- because each row's divisor is unchanged;
    # only the number of launches differs. Do not unroll this back into a Python loop for symmetry
    # with the JAX file.
    # concatenate, not stack: the three entries are already (3, 3), (3, 3) and (1, 3) blocks rather
    # than seven separate (3,) vectors, so this joins them into the same (7, 3) array the
    # per-candidate version produced -- in the same row order (crosses, complements, fallback).
    cands = mx.concatenate(cands)
    norms = mx.linalg.norm(cands, axis=1, keepdims=True)
    # `consts` from above, not a second `_consts(norms.dtype)` lookup: norms is the norm of a real
    # array, so its dtype is mat.dtype by construction, and re-deriving it invited the reader to
    # wonder whether the two could differ.
    cands = cands / mx.where(norms == 0.0, consts["one"], norms)
    # `cands @ mat`, not `cands @ mat.T`: this function's contract is a real *symmetric* matrix
    # (see the docstring, and eigenpair_3x3 only ever passes `balanced - xmin*eye`, symmetric by
    # construction), so the transpose is an exact no-op -- verified identical -- and costs an op
    # launch per iteration. The JAX original writes the einsum in the equivalent `mat @ cands_i`
    # order. If this ever needs to accept an asymmetric matrix, the transpose must come back.
    resid = mx.linalg.norm(cands @ mat, axis=1)
    # A candidate that collapsed to zero is not a valid eigenvector; disqualify it with a large
    # FINITE penalty proportional to the residual scale -- the comparison only needs an ordering.
    #
    # `mx.inf` does exist, so `mx.where(alive, resid, mx.inf)` would mirror the JAX original more
    # closely and in one fewer op. It is avoided deliberately: the rank-0 case (a multiple of the
    # identity) zeroes the first six candidates, and the surviving e_0 fallback has residual exactly
    # 0, so an inf-filled vector would be six infs and one zero. That works, but it puts non-finite
    # values into an array this function then reduces with argmin, and every other guard in this
    # module is written to keep NaN/inf out of the graph entirely rather than rely on comparison
    # semantics. A finite penalty is the same ordering with no non-finite intermediates.
    #
    # Test the PRE-normalization norms rather than re-reducing the normalized rows: after the
    # division above every surviving row has norm exactly 1 and every zero row is still exactly 0,
    # so `norms > 0` and `norm(cands, axis=1) > 0.5` select the same rows while the former is
    # already computed. The threshold moves from 0.5 to 0 because it is now applied to the
    # unnormalized magnitudes, where 0.5 would be a scale-dependent cutoff on the candidates
    # themselves rather than the "did this collapse to zero" test intended -- the JAX version's 0.5
    # is only meaningful because it tests post-normalization rows, whose norms are exactly 1 or 0.
    alive = (norms[:, 0] > 0.0).astype(mat.dtype)
    resid = resid + (1.0 - alive) * (mx.max(resid) + 1.0)
    return cands[mx.argmin(resid)]


def eigenpair_3x3(mat):
    """Lowest eigenpair of a real symmetric 3x3 matrix via Cardano's method.

    Mirrors ``rqutils.ground_locg.eigenpair_3x3``. The matrix is balanced -- shifted traceless and
    scaled by its largest entry -- before the characteristic polynomial is formed: without that,
    the coefficients of a large-trace matrix lose all significance and the radicand of the square
    root goes negative, yielding NaN (``docs/locg.md`` items I1/I2). The null vector comes from
    ``_nullvec_3x3``, which is rank-aware rather than using one fixed column pair (item I3),
    and the closing Rayleigh quotient recovers the precision Cardano alone loses.

    Assumes ``mat`` is real symmetric, not just Hermitian: the JAX original's ``.conjugate()``
    calls and real/imag norm splits are no-ops here and are dropped. ``ground_locg_mlx`` rejects
    complex input up front, so that assumption is enforced rather than merely documented.

    Reference: J. Kopp, Int. J. Mod. Phys. C. 19, 523 (2008).

    ``mx.linalg.eigh`` does exist and would accept this input directly; the analytic route is kept
    because a per-iteration eigensolve that crosses the device boundary measured 1.37x slower, not
    because MLX lacks an eigensolver. An earlier note here claimed Cardano's near-degeneracy error
    costs the f64 CPU arm 217 iterations against JAX's 89, which would have made an ``eigh``-backed
    f64 path a large win; a controlled run retired that claim (220 vs 217 -- both frameworks use the
    same formulation, so they converge alike). ``docs/mlx-metal-kernels.md`` has the numbers.
    """
    consts = _consts(mat.dtype)
    eye = consts["eye3"]
    # Whole-array extraction rather than element-by-element stacking. `mx.stack([mat[0, 0],
    # mat[1, 1], mat[2, 2]])` costs four launches (three scalar gathers plus the stack) to obtain
    # what mx.diagonal gets in one, and the two hand-permuted stacks below are mx.roll and a
    # reversed slice. All of it is index arithmetic, not algebra, so the reformulation is exact:
    # c1 and c0 agree with the element-wise version at 0.0 max relative difference over 5000
    # random symmetric matrices spanning scales 1e-8..1e8. This also brings the source closer to
    # rqutils.ground_locg.eigenpair_3x3, which already uses diagonal/roll/prod (the "change one,
    # change both" rule in CLAUDE.md).
    d = mx.diagonal(mat)
    shift = mx.sum(d) / 3.0
    scale = mx.max(mx.abs(mat))
    scale = mx.where(scale > 0.0, scale, consts["one"])
    balanced = (mat - shift * eye) / scale

    bd = mx.diagonal(balanced)
    # The strict lower triangle, in the order (1,0), (2,0), (2,1) -- one gather, not three.
    indices = _indices()
    modod = mx.square(balanced[indices["lower_rows"], indices["lower_cols"]])
    # Characteristic polynomial of the traceless balanced matrix: x^3 + c1 x + c0.
    # roll(bd, 1) is [bd2, bd0, bd1] and modod[::-1] is [modod2, modod1, modod0], exactly the
    # permutations the explicit stacks spelled out.
    c1 = mx.sum(bd * mx.roll(bd, 1)) - mx.sum(modod)
    c0 = (
        mx.sum(bd * modod[::-1])
        - mx.prod(bd)
        - 2.0 * (balanced[0, 2] * balanced[1, 0] * balanced[2, 1])
    )
    # Both radicands are non-negative for a symmetric matrix; clamp them against rounding.
    # Bare 0.0, as ground_locg does: mx.maximum promotes a Python scalar without constructing an
    # array, so this needs no cached constant (unlike the mx.where operands, which must be arrays
    # to match dtype).
    p = mx.maximum(-3.0 * c1, 0.0)
    disc = mx.maximum(-27.0 * c1 * c1 * c1 - 182.25 * c0 * c0, 0.0)
    phi = mx.arctan2(mx.sqrt(disc), -13.5 * c0) / 3.0
    cphi = mx.cos(phi)
    sphi = mx.sin(phi)
    # Roots are (sqrt(p) / 3) {2 cos(phi), 2 cos(phi -+ 2pi/3)}. Built as one length-3 expression
    # over stacked coefficients instead of three separately-computed scalars: the three roots are
    # the same linear combination a*cphi + b*sphi with different (a, b), so stacking the
    # coefficients turns six scalar ops plus a stack into two multiplies and an add. Exact -- the
    # same products in the same order, only batched. The coefficient vectors come from the
    # per-dtype cache so they are not rebuilt every iteration.
    xmin = mx.min(consts["cphi_coeff"] * cphi + consts["sphi_coeff"] * sphi)
    xmin = xmin * mx.sqrt(p) / 3.0

    vec = _nullvec_3x3(balanced - xmin * eye)
    # Rayleigh quotient: recovers full precision where Cardano alone reaches only sqrt(eps).
    return mx.sum(vec * (balanced @ vec)) * scale + shift, vec
