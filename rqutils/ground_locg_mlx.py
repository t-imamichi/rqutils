r"""
=========================================
Single-vector LOBPCG on MLX (Apple Metal)
=========================================

.. currentmodule:: rqutils.ground_locg_mlx

Overview
========

An MLX port of :mod:`rqutils.ground_locg` and of ``rqutils.sqd.apply_h_xz_cached``, for running
the SQD eigensolver loop on Apple GPUs. The algorithm is the same single-vector LOBPCG -- see
:mod:`rqutils.ground_locg` for the derivation, the Rayleigh-Ritz specialization, and the
numerical analysis behind every guard reproduced here.

``mlx`` is an optional, darwin-only extra (``uv run --extra mlx ...``); importing this module
without it raises ``ImportError``. Nothing else in the package imports it.

Relationship to the JAX implementation
======================================

The numerics are kept deliberately in step with :mod:`rqutils.ground_locg`: both eigenpair
kernels balance before forming their characteristic polynomials, :func:`eigenpair_3x3` takes its
null vector from the rank-aware :func:`_nullvec_3x3`, both close with a Rayleigh-quotient polish,
the iteration re-orthogonalizes ``t`` against the new ``x``, renormalizes the search direction,
masks a zeroed direction out of Rayleigh-Ritz contention, and reports that condition as
convergence. Those are items I1-I7 of ``docs/locg.md``; each was measured to fail *silently* --
a plausible wrong number rather than a raise or a ``NaN``. **When editing either file, change
both.**

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
* **Everything is real-valued.** MLX has no complex128 anywhere and Metal has no float64 at
  all, so :func:`eigenpair_2x2` and :func:`eigenpair_3x3` drop the ``.conjugate()`` calls and
  real/imag norm splits their JAX counterparts need for the general complex-Hermitian case.
  :func:`ground_locg_mlx` therefore **rejects complex input** rather than silently returning
  wrong eigenvalues: an earlier version of this port relied on an upstream even-Y constraint to
  keep coefficients real and admitted in its own docstring that nothing in the repo would catch
  a regression there.

The analytic 2x2/3x3 Rayleigh-Ritz step carries over directly, which is what makes this port
feasible at all -- MLX has no ``eigh``.

Verification
============

MLX cannot initialize without a Metal device, which rules out headless testing. Two checkers
cover the gap: ``examples/check_ground_locg_mlx_static.py`` re-executes this module's source
against a numpy shim bound to the name ``mx`` (no MLX, no GPU -- this is what validates the
algorithm), and ``examples/check_ground_locg_mlx_mlx.py`` runs the real thing on both devices
and both precisions.

MLX LOBPCG API
==============

.. autofunction:: ground_locg_mlx

.. autofunction:: eigenpair_2x2

.. autofunction:: eigenpair_3x3

.. autofunction:: apply_h_xz_mlx
"""

import math

import mlx.core as mx
import numpy as np

_SQRT3 = math.sqrt(3.0)


def apply_h_xz_mlx(vec, xsources, diagonals):
    """Return Hv from precomputed X sources and diagonals.

    Mirrors ``rqutils.sqd.apply_h_xz_cached``. ``xsources`` must already be sanitized (no
    negative entries, with the corresponding diagonals zeroed) -- see
    ``_bench_common.build_solver_inputs``.
    """
    out = mx.zeros_like(vec)
    for igroup in range(xsources.shape[0]):
        out = out + mx.take(vec, xsources[igroup]) * diagonals[igroup]
    return out


def apply_h_xz_mlx_chunked(vec, xsources, diagonals, chunk=16):
    """Return Hv via chunked batched gather -- the MLX counterpart of ``apply_h_xz_mlx``.

    ``apply_h_xz_mlx`` above loops over the J X-groups doing one ``take``+multiply+add per
    group -- 3J Python-level MLX op constructions per matvec. Since ``xsources``/``diagonals``
    are dense ``(J, N)`` arrays, groups can instead be processed in chunks: gather a whole chunk
    with one flat ``take``, reshape, and reduce with a single weighted sum, cutting the op count
    from ``3*J`` to roughly ``3*ceil(J/chunk)``.

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

    See ``examples/_bench_common.apply_h_xz_chunked`` for the JAX equivalent used by the JAX
    arms of the benchmark -- batching the gather is applied symmetrically to both frameworks so
    the comparison stays about the solver loop, not about who has the better matvec.

    Verified (design doc, Optimization 1): max abs diff vs the unchunked loop matvec is
    <= 2.7e-15 for chunk in {1, 4, 8, 16, 32, 50, 100, 128}, and matches ``H @ v`` to 1.8e-15.

    Args:
        vec: The vector to multiply, shape ``(N,)``.
        xsources: Sanitized X-source indices, shape ``(J, N)``.
        diagonals: Sanitized diagonals, shape ``(J, N)``, matching ``vec``'s dtype.
        chunk: Number of X-groups to gather per flat ``take``. J is static and small, so this
            is a plain Python int controlling how many groups are unrolled per flat gather.

    Returns:
        ``H @ vec``, algebraically identical to ``apply_h_xz_mlx``.
    """
    num_groups = xsources.shape[0]
    out = mx.zeros_like(vec)
    for start in range(0, num_groups, chunk):
        xc = xsources[start : start + chunk]
        dc = diagonals[start : start + chunk]
        gathered = mx.take(vec, xc.reshape(-1)).reshape(xc.shape)
        out = out + mx.sum(gathered * dc, axis=0)
    return out


_METAL_MATVEC_SOURCE = """
    uint i = thread_position_in_grid.x;
    if (i >= n_states) {
        return;
    }
    T acc = 0;
    for (uint j = 0; j < n_groups; ++j) {
        // xsources/diagonals are row-major (J, N), so column i of group j lives at j*N + i.
        // Adjacent threads read adjacent addresses, so these loads coalesce; only the
        // vec[] gather is irregular, and vec is small enough to sit in cache.
        uint off = j * n_states + i;
        acc += vec[xsources[off]] * diagonals[off];
    }
    out[i] = acc;
"""

_METAL_MATVEC_KERNEL = None


def _get_metal_matvec_kernel():
    """Build (once) the fused gather-multiply-accumulate Metal kernel."""
    global _METAL_MATVEC_KERNEL
    if _METAL_MATVEC_KERNEL is None:
        _METAL_MATVEC_KERNEL = mx.fast.metal_kernel(
            name="sqd_apply_h_xz",
            input_names=["vec", "xsources", "diagonals", "n_groups", "n_states"],
            output_names=["out"],
            source=_METAL_MATVEC_SOURCE,
        )
    return _METAL_MATVEC_KERNEL


def apply_h_xz_mlx_metal(vec, xsources, diagonals, threadgroup=256):
    """Return Hv via a single fused custom Metal kernel.

    Both ``apply_h_xz_mlx`` and ``apply_h_xz_mlx_chunked`` express the matvec as a sequence of
    MLX ops, so each one launches its own kernel and materializes a full intermediate array.
    This version computes ``out[i] = sum_j vec[xsources[j, i]] * diagonals[j, i]`` in one launch
    with the accumulator held in a per-thread register, so there are no intermediates at all.

    One thread owns one output element, which means no atomics are needed: thread ``i`` is the
    only writer of ``out[i]``.

    At J=100, N=893 this replaces 21 op launches (chunk=16) plus ~0.4 MB of intermediate
    traffic per matvec with a single launch and none. The gather into ``vec`` stays irregular --
    that is inherent to SQD -- but ``vec`` is only ~3.5 KiB at this N and sits in cache.

    Metal is float32-only for this purpose (it has no float64 at all), so this path is usable
    only by the f32 arms. Callers must check ``vec.dtype``; passing float64 raises.

    The kernel's arithmetic was validated against ``apply_h_xz_cached`` by simulating the exact
    per-thread indexing in numpy: max abs diff 2.7e-15, and 3.6e-15 against a dense ``H @ v``.

    Args:
        vec: The vector to multiply, shape ``(N,)``, float32.
        xsources: Sanitized X-source indices, shape ``(J, N)``, int32.
        diagonals: Sanitized diagonals, shape ``(J, N)``, float32.
        threadgroup: Threads per threadgroup. 256 is a reasonable default on Apple GPUs.

    Returns:
        ``H @ vec``, algebraically identical to ``apply_h_xz_mlx``.

    Raises:
        ValueError: If ``vec`` is not float32, since Metal has no float64.
    """
    if vec.dtype != mx.float32:
        raise ValueError(
            f"apply_h_xz_mlx_metal requires float32 (Metal has no float64), got {vec.dtype}. "
            "Use apply_h_xz_mlx_chunked for the f64 arms."
        )

    num_groups, num_states = xsources.shape
    kernel = _get_metal_matvec_kernel()
    outputs = kernel(
        inputs=[vec, xsources, diagonals, num_groups, num_states],
        template=[("T", mx.float32)],
        grid=(num_states, 1, 1),
        threadgroup=(min(threadgroup, num_states), 1, 1),
        output_shapes=[(num_states,)],
        output_dtypes=[mx.float32],
    )
    return outputs[0]


# Threadgroup memory arrays need a compile-time-constant size in Metal -- they cannot be sized by
# the runtime `lanes` value, since MLX generates the kernel's signature from input_names/
# output_names plus a fixed table of attributes, with no way to thread a runtime size into a
# `threadgroup` declaration. So `partials` is declared with this literal size, and the Python
# wrapper below must raise rather than silently clamp if a caller's `threadgroup` would exceed it.
_METAL_SAS_MAX_THREADGROUP = 256

_METAL_SAS_SOURCE = """
    // One threadgroup per (i, j) pair with i <= j; one thread per stride-slice of the vectors.
    // Fixed-size threadgroup memory: see _METAL_SAS_MAX_THREADGROUP above for why this must be a
    // compile-time literal rather than `lanes`. Only the first `lanes` slots are ever touched.
    threadgroup T partials[_METAL_SAS_MAX_THREADGROUP_LITERAL_];

    uint pair = threadgroup_position_in_grid.x;
    uint lane = thread_position_in_threadgroup.x;
    uint lanes = threads_per_threadgroup.x;

    // Unrank `pair` into (i, j) with i <= j. n_basis is 2 or 3, so a short scan is cheaper
    // than any closed form and avoids integer-sqrt rounding concerns entirely.
    uint i = 0;
    uint j = 0;
    uint seen = 0;
    for (uint a = 0; a < n_basis; ++a) {
        for (uint b = a; b < n_basis; ++b) {
            if (seen == pair) {
                i = a;
                j = b;
            }
            seen += 1;
        }
    }

    // Strided partial sum in a register. Stride `lanes` keeps adjacent lanes on adjacent
    // addresses, so these loads coalesce.
    T acc = 0;
    for (uint k = lane; k < n_states; k += lanes) {
        acc += vectors[i * n_states + k] * mvs[j * n_states + k];
    }
    partials[lane] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tree reduction over the threadgroup. `lanes` is a power of two (the caller rounds down),
    // so the halving is exact and no lane reads past the written region.
    for (uint half = lanes / 2; half > 0; half /= 2) {
        if (lane < half) {
            partials[lane] += partials[lane + half];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lane == 0) {
        // Write BOTH triangles with the identical value, so the result is exactly symmetric by
        // construction and no separate symmetrization op is needed. For i == j this writes the
        // same slot twice, which is harmless.
        out[i * n_basis + j] = partials[0];
        out[j * n_basis + i] = partials[0];
    }
""".replace("_METAL_SAS_MAX_THREADGROUP_LITERAL_", str(_METAL_SAS_MAX_THREADGROUP))

_METAL_SAS_KERNEL = None


def _get_metal_sas_kernel():
    """Build (once) the fused Rayleigh-Ritz inner-product kernel."""
    global _METAL_SAS_KERNEL
    if _METAL_SAS_KERNEL is None:
        _METAL_SAS_KERNEL = mx.fast.metal_kernel(
            name="sqd_compute_sas",
            input_names=["vectors", "mvs", "n_basis", "n_states"],
            output_names=["out"],
            source=_METAL_SAS_SOURCE,
        )
    return _METAL_SAS_KERNEL


def _compute_sas_metal(vectors, mvs, threadgroup=256):
    """Return the (n x n) matrix of <v_i | A | v_j> in a single fused Metal launch.

    The op-graph :func:`_compute_sas` costs 16 op launches per iteration (measured) to produce
    nine numbers, of which only six are distinct. This computes all six distinct inner products
    in one launch: one threadgroup per (i, j) pair with i <= j, a strided per-thread partial sum,
    then a threadgroup-memory tree reduction.

    Thread 0 of each threadgroup writes **both** ``out[i*n + j]`` and ``out[j*n + i]``, so the
    result is exactly symmetric by construction -- stronger than the op-graph path's
    ``(sas + sas.T) * 0.5``, which averages two values that differ by rounding -- and the
    symmetrization op disappears rather than merely being fused.

    The tree reduction changes summation order relative to ``mx.sum``. For dot products of
    unit-norm vectors this is benign and typically *more* accurate than sequential summation
    (error growing as log n rather than n), but it is a change: see
    ``examples/check_ground_locg_mlx_static.py`` case 3h, which pins agreement with both the
    op-graph path and a direct ``v @ m``.

    Metal has no float64, so this path is f32-only, exactly like :func:`apply_h_xz_mlx_metal`.
    The f64 arms must keep using :func:`_compute_sas`.

    Args:
        vectors: Basis vectors, a tuple of 2 or 3 arrays of shape ``(N,)``, float32.
        mvs: Their images under A, same length and shapes, float32.
        threadgroup: Maximum threads per threadgroup. Rounded down to a power of two so the
            tree reduction halves exactly, and must not exceed
            :data:`_METAL_SAS_MAX_THREADGROUP` -- the kernel's ``partials`` array is sized by
            that literal at compile time, not by this runtime value.

    Returns:
        The ``(n, n)`` matrix of ``<v_i | A | v_j>``, exactly symmetric.

    Raises:
        ValueError: If the inputs are not float32, since Metal has no float64.
        ValueError: If ``vectors`` and ``mvs`` differ in length, or the length is not 2 or 3.
        ValueError: If ``threadgroup`` exceeds :data:`_METAL_SAS_MAX_THREADGROUP`.
    """
    if len(vectors) != len(mvs):
        raise ValueError(f"vectors and mvs must have equal length, got {len(vectors)}/{len(mvs)}")
    num_basis = len(vectors)
    if num_basis not in (2, 3):
        raise ValueError(f"_compute_sas_metal supports a basis of 2 or 3, got {num_basis}")
    for name, arrays in (("vectors", vectors), ("mvs", mvs)):
        for array in arrays:
            if array.dtype != mx.float32:
                raise ValueError(
                    f"_compute_sas_metal requires float32 (Metal has no float64), got "
                    f"{array.dtype} in {name}. Use _compute_sas for the f64 arms."
                )
    # The kernel's `partials` threadgroup array is declared with a compile-time-constant size
    # (see _METAL_SAS_MAX_THREADGROUP above); raise loudly rather than silently clamping, since a
    # silent clamp here would mean the caller's requested threadgroup size is not what actually
    # ran, which is exactly the class of "plausible wrong number" bug this repo's guards exist to
    # prevent.
    if threadgroup > _METAL_SAS_MAX_THREADGROUP:
        raise ValueError(
            f"threadgroup={threadgroup} exceeds _METAL_SAS_MAX_THREADGROUP="
            f"{_METAL_SAS_MAX_THREADGROUP}, the compile-time size of the kernel's `partials` "
            "threadgroup array"
        )

    num_states = vectors[0].shape[0]
    stacked_v = mx.stack(vectors)
    stacked_m = mx.stack(mvs)
    num_pairs = num_basis * (num_basis + 1) // 2

    # Round the threadgroup size down to a power of two: the tree reduction halves `lanes` until
    # it reaches 1, which only visits every written lane exactly once if it starts as a power of
    # two. Also cap it at num_states so no lane sits idle with nothing to accumulate.
    lanes = 1
    while lanes * 2 <= min(threadgroup, num_states):
        lanes *= 2

    # No output initialization is needed: the pairs (i, j) with i <= j cover every slot of the
    # (n, n) output once the j > i writes are mirrored, so every element is written before it is
    # read. Do not "fix" this by zero-filling first; that would add back a launch.
    kernel = _get_metal_sas_kernel()
    outputs = kernel(
        inputs=[stacked_v, stacked_m, num_basis, num_states],
        template=[("T", mx.float32)],
        # grid is in THREADS, not threadgroups: one threadgroup of `lanes` threads per pair.
        grid=(num_pairs * lanes, 1, 1),
        threadgroup=(lanes, 1, 1),
        output_shapes=[(num_basis, num_basis)],
        output_dtypes=[mx.float32],
    )
    return outputs[0]


def ground_locg_mlx(
    mat, xinit, args=(), maxiter=1000, tol=None, compile_body=False, compile_chunk=10
):
    """Single-vector LOBPCG in MLX.

    Args:
        mat: Callable mapping ``(vec, *args)`` to ``A @ vec``.
        xinit: Initial vector. Must have nonvanishing overlap with the ground state.
        args: Extra arguments forwarded to ``mat``.
        maxiter: Maximum gradient-descent iterations.
        tol: Convergence tolerance. ``None`` uses the dtype epsilon. ``0.`` disables the
            check, running exactly ``maxiter`` iterations with no per-iteration device sync.
        compile_body: If True, wrap the per-iteration body in ``mx.compile`` so the ~1260
            MLX ops it constructs are traced once instead of on every call. Opt-in and OFF by
            default: with ``compile_body=False`` this function's behaviour, including its
            exact iteration count, is unchanged from before this parameter existed.
        compile_chunk: When ``compile_body`` is True, the number of raw iterations the compiled
            body runs before control returns to Python for a convergence check. MLX has no
            ``while_loop``/``cond``, so checking convergence at all forces a ``float()``
            device sync; checking after every single iteration (chunk=1) would sync exactly as
            often as the uncompiled path and defeat the point of compiling. Running a chunk of
            iterations per compiled call amortizes that sync over ``compile_chunk`` iterations
            instead of paying it every time. In fixed-iteration mode (``tol=0.``) no
            convergence check happens at all, so the compiled body runs start-to-finish with no
            per-iteration sync regardless of this value -- see the fixed-iteration branch below.

    Returns:
        ``(eigenvalue, eigenvector, iterations, converged)``. Check the fourth value rather than
        comparing the third against ``maxiter``, which is ambiguous when convergence happens on
        the final permitted iteration.

    Raises:
        ValueError: If ``xinit`` is complex. Every kernel here assumes real symmetric input.
    """
    xinit = mx.array(xinit)
    # Test the dtype's name rather than comparing against mx.complex64: this holds under real MLX
    # and under the numpy shim in examples/check_ground_locg_mlx_static.py, which defines only the
    # dtypes the port actually uses.
    if "complex" in str(xinit.dtype):
        # The kernels below drop the .conjugate() calls their JAX counterparts need, so complex
        # input would silently produce wrong eigenvalues. MLX has no complex128 at all and Metal
        # no float64, so this port is real-only by construction -- fail loudly instead.
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

    def normalize(vector, norm=None):
        """Divide by the norm, leaving a zero vector untouched instead of producing NaN."""
        if norm is None:
            norm = mx.linalg.norm(vector)
        return vector / mx.where(norm == 0.0, mx.array(1.0, norm.dtype), norm)

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
    # without the guard eigenpair_2x2 sees a sas whose row/col 1 vanish and selects that null
    # direction, collapsing theta towards 0 instead of reporting rho -- the true answer.
    norm_r = mx.linalg.norm(rcurr)
    r_is_zero = bool(float(norm_r) == 0.0)
    tmp_p = normalize(rcurr, norm_r)
    # Reuse ax from iteration 0 rather than recomputing it inside _compute_sas.
    sas = _compute_sas((xcurr, tmp_p), (ax, matvec(tmp_p)))
    if r_is_zero:
        # Lift the p diagonal out of contention so Rayleigh-Ritz cannot pick the null direction.
        # With p excluded the 2x2 solve collapses onto x alone, giving theta = rho and kappa =
        # [1, 0], so xcurr is unchanged and no new search direction is introduced.
        excluded = mx.abs(rho) * 2.0 + 1.0
        sas = sas + mx.stack(
            [
                mx.stack([mx.array(0.0, sas.dtype), mx.array(0.0, sas.dtype)]),
                mx.stack([mx.array(0.0, sas.dtype), excluded - sas[1, 1]]),
            ]
        )
    theta, kappa = eigenpair_2x2(sas)
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
        # sas[2, 2] by |tmp_p|^2, which for a large positive shift is a spuriously low diagonal
        # that gets selected in place of the true minimizer (item I6).
        norm_p = mx.linalg.norm(tmp_p)
        p_is_zero = norm_p == 0.0
        tmp_p = normalize(tmp_p, norm_p)
        # xcurr's image is already known from the previous iteration -- three matvecs, not four.
        sas = _compute_sas((xcurr, ycurr, tmp_p), (axcurr, matvec(ycurr), matvec(tmp_p)))
        # A zeroed tmp_p leaves sas row/col 2 empty, and for a positive-definite A that zero
        # diagonal is the smallest eigenvalue, so Rayleigh-Ritz would pick the null direction and
        # the normalizations below would divide by zero. Lift it out of contention (item I7); the
        # p_is_zero case is reported as convergence by the caller.
        diag_xy = mx.stack([sas[0, 0], sas[1, 1]])
        excluded = mx.max(diag_xy) + mx.sum(mx.abs(diag_xy)) + 1.0
        mask = mx.zeros_like(sas)
        mask[2, 2] = 1.0
        sas = mx.where(p_is_zero, sas * (1.0 - mask) + excluded * mask, sas)
        theta, kappa = eigenpair_3x3(sas)
        tmp_s = ycurr * kappa[1] + tmp_p * kappa[2]
        norm_s = mx.linalg.norm(tmp_s)
        tmp_t = tmp_s * (kappa[0] / mx.where(norm_s == 0.0, mx.array(1.0, norm_s.dtype), norm_s))
        tmp_t = tmp_t - xcurr * norm_s
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

    def converged(xcurr, theta, rcurr, axnext, p_is_zero):
        # Compare the residual norm against the floating-point error we'd expect from forming the
        # residual at all. abs(theta) rather than +theta, and a sum rather than the difference the
        # first version of this port inherited: norm(Ax) - theta is a cancellation of two nearly
        # equal large positive numbers for a positive-definite operator, which was measured going
        # negative and made the test unsatisfiable, so the solver never converged and always burned
        # maxiter (item I4). A ground-state search is typically negative-definite, where +theta
        # would cancel in turn.
        reltol = (mx.linalg.norm(axnext) + mx.abs(theta)) * xcurr.shape[0] * 10
        # These float() calls force a device sync -- the price of MLX having no while_loop.
        if bool(p_is_zero):
            return True
        return float(mx.linalg.norm(rcurr)) < tol * float(reltol)

    niter = 0
    is_converged = False
    if seed_converged:
        # {x} already spans the residual; the loop cannot improve on the seed pair.
        return float(theta), xcurr, 0, True
    if not compile_body:
        for niter in range(1, maxiter + 1):
            theta, xcurr, ycurr, rcurr, axnext, p_is_zero = iter_body(xcurr, ycurr, rcurr, axnext)
            if check_convergence and converged(xcurr, theta, rcurr, axnext, p_is_zero):
                is_converged = True
                break
    elif not check_convergence:
        # Fixed-iteration mode: no convergence check at all, so the compiled body can run every
        # requested iteration with zero per-iteration device sync -- the clean per-iteration
        # speed measurement the design calls for. mx.compile traces iter_body's op graph once
        # (over the (xcurr, ycurr, rcurr, axcurr) array tree) instead of once per maxiter call.
        compiled_body = mx.compile(iter_body)
        for niter in range(1, maxiter + 1):
            theta, xcurr, ycurr, rcurr, axnext, _p_is_zero = compiled_body(
                xcurr, ycurr, rcurr, axnext
            )
    else:
        # Convergence checking mode with compilation: run compile_chunk raw iterations inside a
        # single compiled function, then sync once to check convergence, instead of syncing
        # after every single iteration. A compiled chunk-of-iterations body is what amortizes
        # the unavoidable float()-sync cost over compile_chunk iterations rather than paying it
        # every time -- compiling a body that itself contains a Python-level convergence branch
        # is not an option, since mx.compile traces a fixed array-in/array-out computation and
        # has no equivalent of jax.lax.while_loop/cond to make that branch part of the graph.
        def chunk_body(xcurr, ycurr, rcurr, axcurr):
            theta = mx.array(0.0, xcurr.dtype)
            p_is_zero = mx.array(False)
            for _ in range(compile_chunk):
                theta, xcurr, ycurr, rcurr, axcurr, p_is_zero = iter_body(
                    xcurr, ycurr, rcurr, axcurr
                )
            return theta, xcurr, ycurr, rcurr, axcurr, p_is_zero

        compiled_chunk = mx.compile(chunk_body)
        niter = 0
        while niter < maxiter:
            this_chunk = min(compile_chunk, maxiter - niter)
            if this_chunk == compile_chunk:
                theta, xcurr, ycurr, rcurr, axnext, p_is_zero = compiled_chunk(
                    xcurr, ycurr, rcurr, axnext
                )
            else:
                # Final partial chunk: fall back to the uncompiled body so niter lands exactly
                # on maxiter, matching the uncompiled path's semantics one iteration at a time.
                for _ in range(this_chunk):
                    theta, xcurr, ycurr, rcurr, axnext, p_is_zero = iter_body(
                        xcurr, ycurr, rcurr, axnext
                    )
            niter += this_chunk
            if converged(xcurr, theta, rcurr, axnext, p_is_zero):
                is_converged = True
                break

    return float(theta), xcurr, niter, is_converged


def _compute_sas(vectors, mvs):
    """Return the (n x n) matrix of <v_i | A | v_j> for n in {2, 3}.

    Takes the images ``mvs`` rather than computing them, so a caller holding an already-known
    ``A x`` can pass it in instead of paying for a fourth matrix-vector product per iteration
    (``docs/locg.md`` item S1).
    """
    rows = []
    for v1 in vectors:
        rows.append(mx.stack([mx.sum(v1 * mv) for mv in mvs]))
    sas = mx.stack(rows)
    # Symmetrize: the two triangles differ only by rounding for real symmetric A.
    return (sas + sas.T) * 0.5


def _project_out(basis, vector):
    """Orthogonalize ``vector`` against ``basis``, ending on a subtraction.

    The repeated passes and the final zeroing are load-bearing; see the comments in
    ``rqutils/ground_locg.py:390-410``. Near convergence, ending on a normalization can
    reintroduce basis components through catastrophic cancellation and wreck the
    Rayleigh-Ritz conditioning.
    """
    for _ in range(2):
        ips = [mx.sum(vb * vector) for vb in basis]
        for vb, ip in zip(basis, ips):
            vector = vector - vb * ip
        norm = mx.linalg.norm(vector)
        vector = vector / mx.where(norm == 0.0, mx.array(1.0, norm.dtype), norm)

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
    scale = mx.max(mx.abs(mat))
    scale = mx.where(scale > 0.0, scale, mx.array(1.0, mat.dtype))
    balanced = mat / scale
    d = mx.stack([balanced[0, 0], balanced[1, 1]])
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
        vec / mx.where(norm > 0.0, norm, mx.array(1.0, norm.dtype)),
        mx.stack([mx.array(1.0, mat.dtype), mx.array(0.0, mat.dtype)]),
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
    cands = [
        mx.linalg.cross(mat[:, 0], mat[:, 1]),
        mx.linalg.cross(mat[:, 1], mat[:, 2]),
        mx.linalg.cross(mat[:, 2], mat[:, 0]),
    ]
    # Rank 1 (degenerate lowest eigenvalue): every cross product is numerical noise and the null
    # space is the orthogonal complement of the largest column; any member of it is an eigenvector.
    col_index = mx.argmax(mx.sum(mx.square(mx.abs(mat)), axis=0))
    col = mat[:, col_index]
    zero = mx.array(0.0, mat.dtype)
    cands += [
        mx.stack([zero, col[2], -col[1]]),
        mx.stack([-col[2], zero, col[0]]),
        mx.stack([col[1], -col[0], zero]),
    ]
    # Rank 0 (a multiple of the identity): every candidate above is zero, so offer an arbitrary
    # unit vector as the last resort. It has residual 0 and wins by default.
    cands.append(mx.stack([mx.array(1.0, mat.dtype), zero, zero]))

    cands = mx.stack([_normalize_or_zero(c) for c in cands])
    resid = mx.linalg.norm(cands @ mat.T, axis=1)
    # A candidate that collapsed to zero is not a valid eigenvector; disqualify it. MLX has no
    # inf-filling idiom as terse as jnp.where(..., jnp.inf), so add a large finite penalty
    # proportional to the residual scale instead -- the comparison only needs an ordering.
    alive = (mx.linalg.norm(cands, axis=1) > 0.5).astype(mat.dtype)
    resid = resid + (1.0 - alive) * (mx.max(resid) + 1.0)
    return cands[mx.argmin(resid)]


def _normalize_or_zero(vector):
    norm = mx.linalg.norm(vector)
    return vector / mx.where(norm == 0.0, mx.array(1.0, norm.dtype), norm)


def eigenpair_3x3(mat):
    """Lowest eigenpair of a real symmetric 3x3 matrix via Cardano's method.

    Mirrors ``rqutils.ground_locg.eigenpair_3x3``. The matrix is balanced -- shifted traceless and
    scaled by its largest entry -- before the characteristic polynomial is formed: without that,
    the coefficients of a large-trace matrix lose all significance and the radicand of the square
    root goes negative, yielding NaN (``docs/locg.md`` items I1/I2). The null vector comes from
    :func:`_nullvec_3x3`, which is rank-aware rather than using one fixed column pair (item I3),
    and the closing Rayleigh quotient recovers the precision Cardano alone loses.

    Assumes ``mat`` is real symmetric, not just Hermitian: the JAX original's ``.conjugate()``
    calls and real/imag norm splits are no-ops here and are dropped. ``ground_locg_mlx`` rejects
    complex input up front, so that assumption is enforced rather than merely documented.

    Reference: J. Kopp, Int. J. Mod. Phys. C. 19, 523 (2008). MLX has no eigh, so the analytic
    route is the only route.
    """
    eye = mx.eye(3, dtype=mat.dtype)
    d = mx.stack([mat[0, 0], mat[1, 1], mat[2, 2]])
    shift = mx.sum(d) / 3.0
    scale = mx.max(mx.abs(mat))
    scale = mx.where(scale > 0.0, scale, mx.array(1.0, mat.dtype))
    balanced = (mat - shift * eye) / scale

    bd = mx.stack([balanced[0, 0], balanced[1, 1], balanced[2, 2]])
    modod = mx.stack([balanced[1, 0], balanced[2, 0], balanced[2, 1]]) ** 2
    # Characteristic polynomial of the traceless balanced matrix: x^3 + c1 x + c0.
    c1 = mx.sum(bd * mx.stack([bd[2], bd[0], bd[1]])) - mx.sum(modod)
    c0 = (
        mx.sum(bd * mx.stack([modod[2], modod[1], modod[0]]))
        - bd[0] * bd[1] * bd[2]
        - 2.0 * (balanced[0, 2] * balanced[1, 0] * balanced[2, 1])
    )
    # Both radicands are non-negative for a symmetric matrix; clamp them against rounding.
    p = mx.maximum(-3.0 * c1, mx.array(0.0, mat.dtype))
    disc = mx.maximum(
        -27.0 * c1 * c1 * c1 - 182.25 * c0 * c0,
        mx.array(0.0, mat.dtype),
    )
    phi = mx.arctan2(mx.sqrt(disc), -13.5 * c0) / 3.0
    cphi = mx.cos(phi)
    sphi = mx.sin(phi)
    # Roots are (sqrt(p) / 3) {2 cos(phi), 2 cos(phi -+ 2pi/3)}.
    xmin = mx.min(mx.stack([2.0 * cphi, -cphi - _SQRT3 * sphi, -cphi + _SQRT3 * sphi]))
    xmin = xmin * mx.sqrt(p) / 3.0

    vec = _nullvec_3x3(balanced - xmin * eye)
    # Rayleigh quotient: recovers full precision where Cardano alone reaches only sqrt(eps).
    return mx.sum(vec * (balanced @ vec)) * scale + shift, vec
