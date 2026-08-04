"""MLX port of the single-vector LOBPCG solver and the cached SQD matvec.

A transcription of ``rqutils.ground_locg`` and ``rqutils.sqd.apply_h_xz_cached`` onto MLX,
for benchmarking against the JAX originals. See
``docs/superpowers/specs/2026-08-03-mlx-sqd-poc-design.md``.

Four structural differences from the JAX version, all forced by MLX:

* ``jax.lax.scan`` over X groups becomes a Python loop. J is static and small (tens), so
  unrolling into MLX's lazy graph is fine.
* ``jax.lax.while_loop`` becomes a Python loop. MLX has no while_loop/cond/scan, and reading
  the convergence flag forces ``mx.eval`` -- a device sync per iteration. That cost is real
  for an MLX user, not a measurement artifact. Pass ``tol=0.`` to skip the check entirely
  and get a sync-free fixed-iteration run.
* All ``out_sharding=`` arguments are dropped; MLX has unified memory and no sharding.
* Everything is real-valued: MLX has no complex128 anywhere and no float64 on Metal, so this
  port relies on the even-Y constraint enforced by ``_bench_common.build_solver_inputs`` to
  keep every coefficient real. ``eigenpair_2x2`` and ``eigenpair_3x3`` in particular drop the
  ``.conjugate()`` calls and real/imag norm splits the JAX originals need for the general
  (complex-Hermitian) case -- see their docstrings.

The analytic 2x2/3x3 Rayleigh-Ritz step carries over directly, which is what makes this port
feasible at all -- MLX has no ``eigh``.
"""

import mlx.core as mx
import numpy as np


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
        (eigenvalue, eigenvector, iterations).
    """
    xinit = mx.array(xinit)
    check_convergence = tol != 0.0
    if tol is None:
        # Compare the dtype object directly rather than parsing its repr: this works
        # identically under real MLX and under the numpy shim used by the static check.
        tol = float(np.finfo(np.float32 if xinit.dtype == mx.float32 else np.float64).eps)

    def matvec(vec):
        return mat(vec, *args)

    def rayleigh_ritz(*vectors):
        sas = _compute_sas(matvec, *vectors)
        if len(vectors) == 2:
            return eigenpair_2x2(sas)
        return eigenpair_3x3(sas)

    xinit = xinit / mx.linalg.norm(xinit)

    # Iteration 0: no previous direction yet.
    ax = matvec(xinit)
    rho = mx.sum(xinit * ax)
    xcurr = xinit
    rcurr = ax - rho * xcurr

    # Iteration 1: two-vector Rayleigh-Ritz over {x, r}.
    norm_r = mx.linalg.norm(rcurr)
    tmp_p = rcurr / mx.where(norm_r == 0.0, mx.array(1.0, norm_r.dtype), norm_r)
    theta, kappa = rayleigh_ritz(xcurr, tmp_p)
    tmp_t = tmp_p * kappa[0] - xcurr * kappa[1]
    tmp_u = xcurr * kappa[0] + tmp_p * kappa[1]
    xcurr = tmp_u / mx.linalg.norm(tmp_u)
    ycurr = tmp_t / mx.linalg.norm(tmp_t)
    rcurr = matvec(xcurr) - theta * xcurr

    def iter_body(xcurr, ycurr, rcurr):
        """One LOBPCG gradient-descent step. Returns (theta, xcurr, ycurr, rcurr, axnext)."""
        tmp_p = _project_out((xcurr, ycurr), rcurr)
        theta, kappa = rayleigh_ritz(xcurr, ycurr, tmp_p)
        tmp_s = ycurr * kappa[1] + tmp_p * kappa[2]
        norm_s = mx.linalg.norm(tmp_s)
        tmp_t = tmp_s * (kappa[0] / norm_s) - xcurr * norm_s
        tmp_u = xcurr * kappa[0] + tmp_s
        xcurr = tmp_u / mx.linalg.norm(tmp_u)
        ycurr = tmp_t / mx.linalg.norm(tmp_t)
        axnext = matvec(xcurr)
        rcurr = axnext - xcurr * theta
        return theta, xcurr, ycurr, rcurr, axnext

    def converged(xcurr, theta, rcurr, axnext):
        # Same heuristic as the JAX version: compare the residual norm against the
        # floating-point error we'd expect from forming the residual at all.
        reltol = (mx.linalg.norm(axnext) - theta) * xcurr.shape[0] * 10
        # This float() forces a device sync -- the price of MLX having no while_loop.
        return float(mx.linalg.norm(rcurr)) < tol * float(reltol)

    niter = 0
    if not compile_body:
        for niter in range(1, maxiter + 1):
            theta, xcurr, ycurr, rcurr, axnext = iter_body(xcurr, ycurr, rcurr)
            if check_convergence and converged(xcurr, theta, rcurr, axnext):
                break
    elif not check_convergence:
        # Fixed-iteration mode: no convergence check at all, so the compiled body can run every
        # requested iteration with zero per-iteration device sync -- the clean per-iteration
        # speed measurement the design calls for. mx.compile traces iter_body's op graph once
        # (over the (xcurr, ycurr, rcurr) array tree) instead of once per maxiter call.
        compiled_body = mx.compile(iter_body)
        for niter in range(1, maxiter + 1):
            theta, xcurr, ycurr, rcurr, _axnext = compiled_body(xcurr, ycurr, rcurr)
    else:
        # Convergence checking mode with compilation: run compile_chunk raw iterations inside a
        # single compiled function, then sync once to check convergence, instead of syncing
        # after every single iteration. A compiled chunk-of-iterations body is what amortizes
        # the unavoidable float()-sync cost over compile_chunk iterations rather than paying it
        # every time -- compiling a body that itself contains a Python-level convergence branch
        # is not an option, since mx.compile traces a fixed array-in/array-out computation and
        # has no equivalent of jax.lax.while_loop/cond to make that branch part of the graph.
        def chunk_body(xcurr, ycurr, rcurr):
            theta = mx.array(0.0, xcurr.dtype)
            axnext = xcurr
            for _ in range(compile_chunk):
                theta, xcurr, ycurr, rcurr, axnext = iter_body(xcurr, ycurr, rcurr)
            return theta, xcurr, ycurr, rcurr, axnext

        compiled_chunk = mx.compile(chunk_body)
        niter = 0
        while niter < maxiter:
            this_chunk = min(compile_chunk, maxiter - niter)
            if this_chunk == compile_chunk:
                theta, xcurr, ycurr, rcurr, axnext = compiled_chunk(xcurr, ycurr, rcurr)
            else:
                # Final partial chunk: fall back to the uncompiled body so niter lands exactly
                # on maxiter, matching the uncompiled path's semantics one iteration at a time.
                for _ in range(this_chunk):
                    theta, xcurr, ycurr, rcurr, axnext = iter_body(xcurr, ycurr, rcurr)
            niter += this_chunk
            if converged(xcurr, theta, rcurr, axnext):
                break

    return float(theta), xcurr, niter


def _compute_sas(matvec, *vectors):
    """Return the (n x n) matrix of <v_i | A | v_j> for n in {2, 3}."""
    mvs = [matvec(v) for v in vectors]
    rows = []
    for iv1, v1 in enumerate(vectors):
        rows.append(mx.stack([mx.sum(v1 * mvs[iv2]) for iv2 in range(len(vectors))]))
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

    Assumes ``mat`` is real symmetric, not just Hermitian. The JAX original
    (``rqutils.ground_locg.eigenpair_2x2``) handles the general complex-Hermitian case with a
    ``.conjugate()`` on the off-diagonal element and a real/imag norm split; both are dropped
    here because they are no-ops when everything is real. That is guaranteed only by the
    even-Y constraint enforced in ``_bench_common.build_solver_inputs`` -- if that constraint
    is ever relaxed, this function must regain the conjugate and the norm split.
    """
    d = mx.stack([mat[0, 0], mat[1, 1]])
    off = mat[1, 0]
    det = d[0] * d[1] - off * off
    tr = d[0] + d[1]
    eigval = (tr - mx.sqrt(tr * tr - 4.0 * det)) * 0.5
    first = (d[1] - eigval + off) / (d[0] - eigval + off)
    eigvec = mx.stack([first, mx.array(-1.0, first.dtype)])
    return eigval, eigvec / mx.sqrt(first * first + 1.0)


def eigenpair_3x3(mat):
    """Lowest eigenpair of a real symmetric 3x3 matrix via Cardano's method.

    Reference: J. Kopp, Int. J. Mod. Phys. C. 19, 523 (2008). Ported from
    ``rqutils.ground_locg.eigenpair_3x3``; MLX has no eigh, so the analytic route is the
    only route.

    Assumes ``mat`` is real symmetric, not just Hermitian. The JAX original ends with
    ``jnp.cross(v0, v1).conjugate()`` and then normalizes via a real/imag norm split
    (``re[0]*re[0] + ... + im[0]*im[0] + ...``); both are dropped here -- the ``.conjugate()``
    call and the imaginary half of the norm -- because they vanish identically once every
    input is real. That is true only because of the even-Y constraint enforced upstream in
    ``_bench_common.build_solver_inputs`` (odd-Y Pauli strings produce complex coefficients,
    which MLX cannot represent at any precision). If that constraint is ever relaxed, this
    function silently returns wrong eigenvalues -- there is no real-input-only assertion
    here, and the numpy-shim static check exercises only real inputs by construction, so
    nothing in this repo would catch the regression.
    """
    d = mx.stack([mat[0, 0], mat[1, 1], mat[2, 2]])
    modod = mx.stack([mat[1, 0], mat[2, 0], mat[2, 1]]) ** 2
    c2 = -mx.sum(d)
    c1 = mx.sum(d * mx.stack([d[2], d[0], d[1]])) - mx.sum(modod)
    c0 = mx.sum(d * mx.stack([modod[2], modod[1], modod[0]]))
    c0 = c0 - d[0] * d[1] * d[2]
    c0 = c0 - 2.0 * (mat[0, 2] * mat[1, 0] * mat[2, 1])
    p = c2 * c2 - 3.0 * c1
    q = -13.5 * c0 - c2 * c2 * c2 + 4.5 * c2 * c1
    phi = mx.arctan2(mx.sqrt(27.0 * (0.25 * c1 * c1 * (p - c1) + c0 * (q + 6.75 * c0))), q) / 3.0
    cphi = mx.cos(phi)
    sphi = mx.sin(phi)
    root3 = float(np.sqrt(3.0))
    xmin = mx.min(mx.stack([2.0 * cphi, -cphi - root3 * sphi, -cphi + root3 * sphi]))
    eigval = mx.sqrt(p) / 3.0 * xmin - c2 / 3.0
    v0 = mx.stack([mat[0, 1], mat[1, 1] - eigval, mat[2, 1]])
    v1 = mx.stack([mat[0, 2], mat[1, 2], mat[2, 2] - eigval])
    eigvec = mx.linalg.cross(v0, v1)
    return eigval, eigvec / mx.linalg.norm(eigvec)
