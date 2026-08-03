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
import numpy as np
import mlx.core as mx


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


def ground_locg_mlx(mat, xinit, args=(), maxiter=1000, tol=None):
    """Single-vector LOBPCG in MLX.

    Args:
        mat: Callable mapping ``(vec, *args)`` to ``A @ vec``.
        xinit: Initial vector. Must have nonvanishing overlap with the ground state.
        args: Extra arguments forwarded to ``mat``.
        maxiter: Maximum gradient-descent iterations.
        tol: Convergence tolerance. ``None`` uses the dtype epsilon. ``0.`` disables the
            check, running exactly ``maxiter`` iterations with no per-iteration device sync.

    Returns:
        (eigenvalue, eigenvector, iterations).
    """
    xinit = mx.array(xinit)
    check_convergence = tol != 0.
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
    tmp_p = rcurr / mx.where(norm_r == 0., mx.array(1., norm_r.dtype), norm_r)
    theta, kappa = rayleigh_ritz(xcurr, tmp_p)
    tmp_t = tmp_p * kappa[0] - xcurr * kappa[1]
    tmp_u = xcurr * kappa[0] + tmp_p * kappa[1]
    xcurr = tmp_u / mx.linalg.norm(tmp_u)
    ycurr = tmp_t / mx.linalg.norm(tmp_t)
    rcurr = matvec(xcurr) - theta * xcurr

    niter = 0
    for niter in range(1, maxiter + 1):
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

        if check_convergence:
            # Same heuristic as the JAX version: compare the residual norm against the
            # floating-point error we'd expect from forming the residual at all.
            reltol = (mx.linalg.norm(axnext) - theta) * xcurr.shape[0] * 10
            # This float() forces a device sync -- the price of MLX having no while_loop.
            if float(mx.linalg.norm(rcurr)) < tol * float(reltol):
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
        vector = vector / mx.where(norm == 0., mx.array(1., norm.dtype), norm)

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
    eigval = (tr - mx.sqrt(tr * tr - 4. * det)) * 0.5
    first = (d[1] - eigval + off) / (d[0] - eigval + off)
    eigvec = mx.stack([first, mx.array(-1., first.dtype)])
    return eigval, eigvec / mx.sqrt(first * first + 1.)


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
    c0 = c0 - 2. * (mat[0, 2] * mat[1, 0] * mat[2, 1])
    p = c2 * c2 - 3. * c1
    q = -13.5 * c0 - c2 * c2 * c2 + 4.5 * c2 * c1
    phi = mx.arctan2(
        mx.sqrt(27. * (0.25 * c1 * c1 * (p - c1) + c0 * (q + 6.75 * c0))),
        q
    ) / 3.
    cphi = mx.cos(phi)
    sphi = mx.sin(phi)
    root3 = float(np.sqrt(3.))
    xmin = mx.min(mx.stack([2. * cphi, -cphi - root3 * sphi, -cphi + root3 * sphi]))
    eigval = mx.sqrt(p) / 3. * xmin - c2 / 3.
    v0 = mx.stack([mat[0, 1], mat[1, 1] - eigval, mat[2, 1]])
    v1 = mx.stack([mat[0, 2], mat[1, 2], mat[2, 2] - eigval])
    eigvec = mx.linalg.cross(v0, v1)
    return eigval, eigvec / mx.linalg.norm(eigvec)
