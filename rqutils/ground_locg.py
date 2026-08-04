r"""
====================
Single-vector LOBPCG
====================

.. currentmodule:: rqutils.ground_locg

Overview
========

This module defines a single-vector version of the Locally Optimal Block Preconditioned Conjugate
Gradient (LOBPCG) solver[1]. The structure heavily borrows from
``jax.experimental.sparse.linalg.lobpcg_standard``, with optimizations for single-vector (ground
eigenpair) calculation.

LOBPCG is a matrix-free method for finding the extremal eigenpairs of a generalized eigenvalue
problem

.. math::

    A x = \lambda B x

with :math:`(A, B)` Hermitian. Our implementation only solves the non-generalized (:math:`B = I`)
problem and finds the minimum eigenvalue :math:`\lambda_0` and the corresponding eigenvector
:math:`v_0`.

The basic arguments to the function are

- Matrix :math:`A`, either as a JAX Array or a function that takes the vector :math:`x` (as a JAX
  Array) as input and returns :math:`Ax`.
- Initial vector, which must have a non-vanishing overlap with :math:`v_0`.

Algorithm
=========

See reference [1] for details. The goal is to minimize the Rayleigh quotient

.. math::

    \{\lambda_0, v_0\} = \{\min_{x}, \mathrm{argmin}_{x}\}
    \left(\rho (x) := \frac{x^{\dagger} A x}{|x|^2} \right) .


Conceptually, the algorithm consists of gradient descent iterations

.. math::

    x_{i+1} = x_{i} + \alpha_{i} r_{i},

where :math:`r_{i} := A x_{i} - \rho (x_{i}) x_{i}` can be proven to be proportional to the gradient
of :math:`\rho (x_{i})`. Note that :math:`r_{i}` is orthogonal to :math:`x_{i}`:

.. math::

    x_{i}^{\dagger} r_{i} & = x_{i}^{\dagger} A x_{i}
                              - \frac{x_{i}^{\dagger} A x_{i}}{|x_{i}|^2} x_{i}^{\dagger} x_{i} \\
                          & = 0.


In practice, instead of finding the optimal step size :math:`\alpha_{i}`, we can directly minimize
:math:`\rho` in the space spanned by :math:`\{x_{i}, r_{i}\}` via the Rayleigh-Ritz method and
identify the minimizing vector as :math:`x_{i+1}`. Furthermore, it is known that convergence of the
algorithm is drastically improved if we search :math:`x_{i+1}` in the extended space spanned by
:math:`\{x_{i}, x_{i-1}, r_{i}\}`. Thus, one iteration of gradient descent is given by the following
steps, with orthogonal :math:`\{x_{i}, y_{i}, r_{i}\}` (:math:`x_{i}, y_{i}` normal) as the
carryover from the previous iteration and :math:`R_A` indicating the Rayleigh-Ritz routine over
matrix :math:`A`:

.. math::

    p & \leftarrow \frac{r_{i}}{|r_{i}|} \\
    \theta, \kappa & \leftarrow R_A[x_{i}, y_{i}, p] \\
    s & \leftarrow \kappa_1 y_{i} + \kappa_2 p \\
    t & \leftarrow \frac{\kappa_0}{|s|} s - |s| x_{i} \\
    u & \leftarrow \kappa_0 x_{i} + s \\
    x_{i+1} & \leftarrow \frac{u}{|u|} \\
    y_{i+1} & \leftarrow \frac{t}{|t|} \\
    r_{i+1} & \leftarrow A x_{i+1} - \theta x_{i+1}.

The normal vector :math:`y_{i}` is orthogonal to :math:`x_{i}` and :math:`r_{i}` and lies in the
space spanned by :math:`\{x_{i}, x_{i-1}, r_{i}\}`.

Single-vector optimization
==========================

The B of LOBPCG refers to the algorithm's ability to determine multiple eigenvectors simultaneously
as a block. We have however chosen to compute just the ground state vector in this implementation,
eyeing running on extremely large vectors (memory requirement of LOBPCG scales with the number of
eigenvectors to compute). This choice opens up further memory-footprint optimizations in the
Rayleigh-Ritz subroutine.

In the Rayleigh-Ritz subroutine, we form the matrix

.. math::

    R_{jk} = w_{j}^{\dagger} A w_{k},

where :math:`w = {x, y, p}`, and diagonalize it. With an undetermined number of simultaneous vectors
in :math:`w`, we'd have to concatenate :math:`x`, :math:`y`, and :math:`p` (thus creating their
copies) and then numerically invert :math:`R`. Since we know that there are only three vectors, we
can construct :math:`R_{jk}` "by hand" and analytically invert the 3x3 matrix.

The matrix-vector product is the dominant cost in the matrix-free regime this specialization exists
to serve, so :math:`Ax_{i}` is carried through the loop state rather than recomputed when the
projected matrix is formed -- three products per iteration instead of four.

Numerical considerations
========================

Naive transcriptions of the steps above are numerically fragile in ways that fail *silently*: they
return a plausible number that is simply wrong, rather than raising or producing ``NaN``. The
measurements behind each item below are recorded in ``docs/locg.md``.

Analytic eigenpair kernels
--------------------------

:func:`eigenpair_2x2` and :func:`eigenpair_3x3` must not build the characteristic-polynomial
coefficients from the *unshifted, unscaled* matrix. For :math:`H = A + sI` the leading coefficients
grow as powers of :math:`s` while the quantities of interest stay :math:`O(\mathrm{spread})`, so a
large trace destroys the result -- for the 3x3 kernel the radicand of a square root goes negative
and produces ``NaN``. Since a physical Hamiltonian is rarely traceless, this is the ordinary case
rather than an edge case.

Both kernels therefore **balance** first: subtract :math:`\mathrm{tr}/3` (2x2: :math:`\mathrm{tr}/2`)
to work with the traceless part, and divide by :math:`\max_{ij} |A_{ij}|` so the intermediates stay
:math:`O(1)`. The eigenvector is invariant under both operations; only the eigenvalue is mapped
back at the end.

.. math::

    \lambda_{\mathrm{min}}(A) = \sigma \,
    \lambda_{\mathrm{min}}\!\left(\frac{A - \tau I}{\sigma}\right) + \tau,
    \qquad \tau = \frac{\mathrm{tr} A}{n}, \quad \sigma = \max_{ij} |A_{ij}| .


The eigenvector extraction is rank-aware. :func:`eigenpair_3x3` takes the largest of all three
column cross products rather than one fixed pair (any single pair can be rank deficient, in which
case its cross product points nowhere useful), and falls back to the orthogonal complement of the
largest column when the lowest eigenvalue is degenerate (rank 1), or to an arbitrary unit vector
when the input is a multiple of the identity (rank 0). :func:`eigenpair_2x2` selects between the two
rows of the singular shifted matrix on the sign of :math:`\delta = (d_0 - d_1)/2`, so :math:`\delta`
is never cancelled against a nearly equal radius.

Finally both kernels close with a Rayleigh-quotient polish, :math:`\theta \leftarrow v^{\dagger} B
v` on the balanced matrix. This is second order in the eigenvector error and recovers full
precision where the closed form alone reaches only :math:`\sqrt{\epsilon}` (a near-degenerate lowest
pair).

Iteration
---------

- **Convergence threshold.** The relative tolerance is
  :math:`(\|Ax\| + |\theta|)\,n \cdot 10`. The natural-looking ``norm(Ax) - theta`` is a difference
  of two nearly equal large positive numbers for a positive-definite operator, and was measured
  going *negative*, which makes the test unsatisfiable so the solver never converges and always
  exhausts ``maxiter``. Note :math:`|\theta|` rather than :math:`+\theta`: for the
  negative-definite operators typical of a ground-state search, :math:`+\theta` cancels in turn.

- **Basis orthogonality.** :math:`t = \kappa_0 s / |s| - |s| x` is a difference of two quantities
  both nearly parallel to :math:`x` as :math:`|s| \to 0`, so :math:`y` drifts into :math:`x` until
  the nominally three-dimensional search space collapses. Because the Rayleigh-Ritz step solves a
  *standard* (non-generalized) eigenproblem, which presumes an orthonormal basis, the consequence is
  a returned :math:`\theta` **below the true minimum eigenvalue** -- silently wrong rather than
  merely imprecise. :math:`t` is re-orthogonalized against the new :math:`x` before normalization.

- **Search direction normalization.** :func:`_project_out` guarantees only
  :math:`\|p\| \ge 0.99`, and a short :math:`p` scales :math:`\mathrm{sas}_{22}` by :math:`|p|^2`,
  which for a large positive shift is a spuriously low diagonal that Rayleigh-Ritz then selects.
  :math:`p` is renormalized before the projected eigensolve.

- **Exhausted search space.** When :func:`_project_out` returns exactly zero, row and column 2 of
  the projected matrix vanish; for a positive-definite :math:`A` that zero diagonal is the
  *smallest* eigenvalue, so Rayleigh-Ritz would select the null direction and the subsequent
  normalization would divide by zero. That diagonal is masked out of contention, and the condition
  is reported as convergence: :math:`\{x, y\}` already spans the residual, so no new search
  direction exists and no further iteration can lower :math:`\theta`.

Every division by a norm in this module is guarded, so a zero or degenerate input yields a
well-defined result instead of ``NaN``.

Distributed arrays
==================

This function works transparently over distributed (sharded) input :math:`v_0` if the callable
passed as the ``mat`` argument preserves the sharding in the output. The re-orthogonalization
described above introduces two additional inner products per iteration; these follow the same
reduction pattern as the existing ones but have not been exercised on a multi-device mesh.

References
==========

[1]: https://en.wikipedia.org/wiki/LOBPCG

[2]: J. Kopp, *Efficient numerical diagonalization of hermitian 3 x 3 matrices*,
Int. J. Mod. Phys. C. **19**, 523 (2008).

Single-vector LOBPCG API
========================

.. autofunction:: ground_locg

.. autofunction:: eigenpair_2x2

.. autofunction:: eigenpair_3x3
"""
from collections.abc import Callable
import logging
import math
from typing import Optional
from numpy.typing import DTypeLike, NDArray
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec, get_abstract_mesh

_SQRT3 = math.sqrt(3.)


def ground_locg(
    mat: Callable[[jax.Array], jax.Array] | jax.Array,
    xinit: jax.Array | int,
    args: tuple = (),
    maxiter: int = 1000,
    tol: Optional[float] = None,
    vspace: tuple[int, DTypeLike] | None = None,
    debug: bool = False,
    log_level: int = logging.WARNING
) -> tuple[float, NDArray, int, bool]:
    r"""Single-vector LOBPCG.

    Args:
        mat: Matrix :math:`A`, either as an Array or a function :math:`x \mapsto Ax`.
        xinit: Initial vector. If given as an integer (requires ``vspace`` if ``mat`` is callable),
            a one-hot vector is created internally. Must have a non-vanishing overlap with
            :math:`v_0`.
        args: Additional arguments to callable ``mat``.
        maxiter: Maximum number of gradient descent iterations.
        tol: Convergence condition. If None, the machine epsilon of the operator dtype is used.
        vspace: Specification (dimension, dtype) of the vector space. Required only when ``mat`` is
            a callable and ``xinit`` is an integer.
        debug: If True, additionally return per-iteration diagnostics. Note that the diagnostic
            path uses ``jax.lax.scan`` to collect fixed-size output, and therefore always runs the
            full ``maxiter`` iterations with no early exit; rows past convergence are
            post-convergence noise.
        log_level: Verbosity level.

    Returns:
        The smallest eigenvalue, its eigenvector, the number of gradient descent iterations
        performed, and whether the convergence criterion was met. Check the fourth value rather
        than comparing the third against ``maxiter``, which is ambiguous.
    """
    if callable(mat):
        return _ground_locg_callable(mat, xinit, args, maxiter, tol, vspace=vspace, debug=debug,
                                     log_level=log_level)
    return _ground_locg_matrix(mat, xinit, maxiter, tol, debug=debug, log_level=log_level)


@jax.jit(static_argnames=['maxiter', 'debug', 'log_level'])
def _ground_locg_matrix(
    mat: jax.Array,
    xinit: jax.Array,
    maxiter: int,
    tol: jax.Array | float | None,
    debug: bool = False,
    log_level: int = logging.WARNING
):
    vspace = None
    if jnp.issubdtype(xinit.dtype, jnp.integer):
        vspace = (mat.shape[1], mat.dtype)

    def matvec(x):
        return jax.lax.dot(mat, x,
                           precision=(jax.lax.Precision.HIGHEST,) * 2,
                           out_sharding=jax.typeof(x).sharding)

    return _ground_locg_callable(matvec, xinit, (), maxiter, tol,
                                 vspace=vspace, debug=debug, log_level=log_level)


@jax.jit(
    static_argnames=[
        'matvec',
        'maxiter',
        'vspace',
        'debug',
        'log_level'
    ]
)
def _ground_locg_callable(
    matvec: Callable[[jax.Array], jax.Array],
    xinit: jax.Array | int,
    args: tuple,
    maxiter: int,
    tol: jax.Array | float | None,
    vspace: tuple[int, DTypeLike] | None = None,
    debug: bool = False,
    log_level: int = logging.WARNING
):
    if jnp.issubdtype(xinit.dtype, jnp.integer):
        sharding = None
        if not (mesh := get_abstract_mesh()).empty:
            sharding = PartitionSpec(mesh.axis_names)
        xinit = (jax.lax.broadcasted_iota(xinit.dtype, (vspace[0],), 0, out_sharding=sharding)
                 == xinit).astype(vspace[1])

    def normalize(vector, norm=None):
        """Divide by the norm, leaving a zero vector untouched instead of producing NaN."""
        if norm is None:
            norm = jnp.linalg.norm(vector)
        return vector / jnp.where(norm == 0., 1., norm)

    def compute_sas(vectors, mvs):
        """Projected matrix over ``vectors``, given their (possibly precomputed) images."""
        nv = len(vectors)
        sas = jnp.zeros((nv, nv), dtype=vectors[0].dtype)
        for iv1, mv1 in enumerate(mvs):
            for iv2 in range(iv1 + 1, nv):
                sas = sas.at[iv2, iv1].set(jnp.sum(vectors[iv2].conjugate() * mv1))
        sas += sas.conjugate().T
        for iv1, (v1, mv1) in enumerate(zip(vectors, mvs)):
            sas = sas.at[iv1, iv1].set(jnp.sum(v1.conjugate() * mv1))

        return sas

    def diagnostics(xcurr, ycurr, rcurr, theta, kappa=None, reltol=None, converged=None):
        sas = compute_sas((xcurr, ycurr, rcurr),
                          tuple(matvec(v, *args) for v in (xcurr, ycurr, rcurr)))
        axcurr = matvec(xcurr, *args)
        rho = jnp.sum(xcurr.conjugate() * axcurr).real  # turns out to be faster than dot()

        if kappa is None:
            kappa = jnp.zeros(3, dtype=xcurr.dtype)
        if reltol is None:
            reltol = jnp.array(0.)
        if converged is None:
            converged = jnp.array(False)

        return {
            'x': xcurr,
            'y': ycurr,
            'r': rcurr,
            'theta': theta,
            'rho': rho,
            'kappa': kappa,
            'sas': sas,
            'reltol': reltol,
            'converged': converged
        }

    def body_iter0(xcurr):
        """Steepest-descent seed. Returns Ax so no later step recomputes it."""
        xnext = xcurr
        ax = matvec(xcurr, *args)
        rho = jnp.sum(xcurr.conjugate() * ax).real
        rnext = ax - rho * xnext
        if debug:
            diag = diagnostics(xnext, jnp.zeros_like(xnext), rnext, rho)
            return xnext, rnext, ax, rho, diag
        return xnext, rnext, ax, rho

    def body_iter1(xcurr, rcurr, axcurr):
        tmp_p = normalize(rcurr)
        # Reuse Ax from body_iter0 rather than recomputing it inside compute_sas.
        sas = compute_sas((xcurr, tmp_p), (axcurr, matvec(tmp_p, *args)))
        theta, kappa = eigenpair_2x2(sas)
        tmp_t = tmp_p * kappa[0] - xcurr * kappa[1]
        tmp_u = xcurr * kappa[0] + tmp_p * kappa[1]
        xnext = normalize(tmp_u)
        # Re-orthogonalize for the same reason as in body(); see the module docstring.
        for _ in range(2):
            tmp_t -= xnext * jnp.sum(xnext.conjugate() * tmp_t)
        ynext = normalize(tmp_t)
        axnext = matvec(xnext, *args)
        rnext = axnext - theta * xnext
        if debug:
            diag = diagnostics(xnext, ynext, rnext, theta, jnp.insert(kappa, 1, 0.))
            return xnext, ynext, rnext, axnext, theta, diag
        return xnext, ynext, rnext, axnext, theta

    def body(state):
        xcurr, ycurr, rcurr, axcurr = state[-4:]
        if log_level <= logging.DEBUG:
            jax.debug.print('LOCG iteration {}', state[0])

        # Residual basis selection.
        # R is supposed to be already orthogonal to X, but we find that it's necessary to project
        # out with respect to both X and P to get good convergence of the residual.
        tmp_p = _project_out((xcurr, ycurr), rcurr)
        # _project_out only guarantees |tmp_p| >= 0.99, but the Rayleigh-Ritz step below solves a
        # standard eigenproblem and so assumes an orthonormal basis: a short tmp_p scales sas[2, 2]
        # by |tmp_p|^2, which for a large positive shift is a spuriously low diagonal that gets
        # selected in place of the true minimizer.
        norm_p = jnp.linalg.norm(tmp_p)
        p_is_zero = norm_p == 0.
        tmp_p = normalize(tmp_p, norm_p)
        # Projected eigensolve. xcurr is the previous iteration's xnext, so its image is already
        # known -- three matvecs per iteration instead of four.
        sas = compute_sas((xcurr, ycurr, tmp_p),
                          (axcurr, matvec(ycurr, *args), matvec(tmp_p, *args)))
        # A zeroed tmp_p leaves sas row/col 2 empty, and for a positive-definite A the resulting
        # zero diagonal is the smallest eigenvalue, so Rayleigh-Ritz would pick the null direction
        # and the normalizations below would divide by zero. Lift it out of contention; the
        # p_is_zero case is reported as convergence instead (see below).
        diag_xy = jnp.diagonal(sas).real[:2]
        excluded = jnp.max(diag_xy) + jnp.sum(jnp.abs(diag_xy)) + 1.
        sas = jnp.where(p_is_zero, sas.at[2, 2].set(excluded.astype(sas.dtype)), sas)
        theta, kappa = eigenpair_3x3(sas)
        # New vectors
        tmp_s = ycurr * kappa[1] + tmp_p * kappa[2]
        norm_s = jnp.linalg.norm(tmp_s)
        tmp_t = tmp_s * (kappa[0] / jnp.where(norm_s == 0., 1., norm_s)) - xcurr * norm_s
        tmp_u = xcurr * kappa[0] + tmp_s
        xnext = normalize(tmp_u)
        # tmp_t is a difference of two quantities both nearly parallel to xcurr as norm_s -> 0, so
        # catastrophic cancellation lets ynext drift into xnext. Once <x|y> is O(1) the basis is no
        # longer orthonormal and the standard Rayleigh-Ritz above returns a theta *below* the true
        # minimum eigenvalue -- a silent wrong answer rather than a visible failure.
        for _ in range(2):
            tmp_t -= xnext * jnp.sum(xnext.conjugate() * tmp_t)
        ynext = normalize(tmp_t)
        axnext = matvec(xnext, *args)
        rnext = axnext - xnext * theta
        # Use the intermediate AX for relative tolerance.
        #
        # Comments from lobpcg_standard:
        # =========
        # I tried many variants of hard and soft locking [3]. All of them seemed
        # to worsen performance relative to no locking.
        #
        # Further, I found a more experimental convergence formula compared to what
        # is suggested in the literature, loosely based on floating-point
        # expectations.
        #
        # [2] discusses various strategies for this in Sec 5.3. The solution
        # they end up with, which estimates operator norm |A| via Gaussian
        # products, was too crude in practice (and overly-lax). The Gaussian
        # approximation seems like an estimate of the average eigenvalue.
        #
        # Instead, we test convergence via self-consistency of the eigenpair
        # i.e., the residual norm |r| should be small, relative to the floating
        # point error we'd expect from computing just the residuals given
        # candidate vectors.
        # =========
        # abs(theta) rather than +theta: the sum must not cancel for either sign of theta, and a
        # ground-state search is typically negative-definite.
        reltol = jnp.linalg.norm(axnext) + jnp.abs(theta)
        reltol *= xcurr.shape[0]
        # Allow some margin for a few element-wise operations.
        reltol *= 10
        norm_rnext = jnp.linalg.norm(rnext)
        # A zeroed search direction means {x, y} already spans the residual: we are at a stationary
        # point of the Rayleigh quotient and no further iteration can lower theta.
        converged = jnp.logical_or(norm_rnext < tol * reltol, p_is_zero)
        if log_level <= logging.DEBUG:
            jax.debug.print('Residual {}, reltol {}, converged: {}', norm_rnext, reltol, converged)

        state = (state[0] + 1, converged, theta, xnext, ynext, rnext, axnext)
        if debug:
            return state, diagnostics(xnext, ynext, rnext, theta, kappa, reltol, converged)
        return state

    if log_level <= logging.DEBUG:
        jax.debug.print('Performing first LOBPCG steps')

    xinit = normalize(xinit)

    # The projected matrix inherits xinit's dtype at the seed step but the operator's inside the
    # loop, so a lower-precision xinit makes while_loop's carry types disagree on theta. Promote up
    # front. eval_shape reads the operator dtype without spending a matrix-vector product, and
    # astype on a matching dtype is a no-op, so the common path is unaffected.
    work_dtype = jnp.result_type(xinit.dtype,
                                 jax.eval_shape(lambda vec: matvec(vec, *args), xinit).dtype)
    xinit = xinit.astype(work_dtype)

    vs_iter0 = body_iter0(xinit)
    if debug:
        diag0 = jax.tree.map(lambda a: jnp.expand_dims(a, 0), vs_iter0[-1])
    # Seed theta with the Rayleigh quotient of xinit so that maxiter=0 returns a meaningful value
    # rather than the state initializer.
    rho_init = vs_iter0[3]

    if tol is None:
        # Derive the tolerance from the operator, not from the initial guess: a float32 xinit on a
        # complex128 problem would otherwise silently loosen this by nine orders of magnitude.
        # work_dtype above is already that promotion.
        tol = float(jnp.finfo(work_dtype).eps)

    vs_iter1 = body_iter1(vs_iter0[0], vs_iter0[1], vs_iter0[2])
    if debug:
        diag1 = jax.tree.map(lambda a: jnp.expand_dims(a, 0), vs_iter1[-1])

    if maxiter == 0:
        # No iteration is permitted, so report the seed pair.
        empty = jnp.array(False)
        if debug:
            diagnostics_out = jax.tree.map(lambda d0, d1: jnp.concatenate([d0, d1], axis=0),
                                           diag0, diag1)
            return rho_init, xinit, 0, empty, diagnostics_out
        return rho_init, xinit, 0, empty

    state = (0, jnp.array(False), vs_iter1[4]) + vs_iter1[:4]
    if debug:
        state, diagnostics_out = jax.lax.scan(
            lambda s, _: body(s), state, length=maxiter
        )
        diagnostics_out = jax.tree.map(lambda d0, d1, dr: jnp.concatenate([d0, d1, dr], axis=0),
                                       diag0, diag1, diagnostics_out)
    else:
        state = jax.lax.while_loop(
            lambda s: jnp.logical_and(s[0] < maxiter, ~s[1]),
            body,
            state
        )

    niter = state[0]
    converged = state[1]
    eigval = state[2]
    xfinal = state[3]
    if debug:
        return eigval, xfinal, niter, converged, diagnostics_out
    return eigval, xfinal, niter, converged


def _project_out(basis, vector):
    for _ in range(2):
        ips = []
        for vb in basis:
            ips.append(jnp.sum(vb.conjugate() * vector))
        for vb, ip in zip(basis, ips):
            vector -= vb * ip
        norm = jnp.linalg.norm(vector)
        vector /= jnp.where(norm == 0., 1., norm)

    # Comments from the original function:
    # ================
    # It's crucial to end on a subtraction of the original basis.
    # This seems to be a detail not present in [2], possibly because of
    # of reliance on soft locking.
    #
    # Near convergence, if the residuals R are 0 and our last
    # operation when projecting (X, P) out from R is the orthonormalization
    # done above, then due to catastrophic cancellation we may re-introduce
    # (X, P) subspace components into U, which can ruin the Rayleigh-Ritz
    # conditioning.
    #
    # We zero out any columns that are even remotely suspicious, so the invariant
    # that [basis, U] is zero-or-orthogonal is ensured.
    # ================
    for _ in range(2):
        ips = []
        for vb in basis:
            ips.append(jnp.sum(vb.conjugate() * vector))
        for vb, ip in zip(basis, ips):
            vector -= vb * ip

    # Note the postcondition: the returned vector is either exactly zero or has norm >= 0.99. It is
    # NOT normalized -- callers feeding it to a standard Rayleigh-Ritz step must renormalize.
    return vector * (jnp.linalg.norm(vector) >= 0.99).astype(vector.dtype)


@jax.jit
def eigenpair_2x2(mat: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Return the lowest eigenpair of a 2x2 Hermitian matrix.

    The matrix is balanced by its largest entry and reduced to its traceless part before the
    quadratic is solved, so that neither the ``tr^2 - 4 det`` cancellation nor over/underflow of the
    intermediates can occur.

    Args:
        mat: A 2x2 Hermitian matrix. Only the diagonal and the lower off-diagonal entry are read.

    Returns:
        The smaller eigenvalue and its normalized eigenvector.
    """
    scale = jnp.max(jnp.abs(mat))
    scale = jnp.where(scale > 0., scale, 1.).astype(jnp.diagonal(mat).real.dtype)
    balanced = mat / scale
    d = jnp.diagonal(balanced).real
    half_tr = jnp.sum(d) * 0.5
    delta = (d[0] - d[1]) * 0.5
    offd = balanced[1, 0]
    rad = jnp.hypot(delta, jnp.abs(offd))
    # Null vector of T + rad I = [[delta + rad, conj(offd)], [offd, rad - delta]], which is
    # singular. Row 1 gives [-conj(offd), delta + rad] and row 2 gives [rad - delta, -offd]; the two
    # are parallel, but each cancels when its own pivot is small, so select on the sign of delta.
    vec = jnp.where(
        delta >= 0.,
        jnp.array([-offd.conjugate(), (delta + rad).astype(mat.dtype)]),
        jnp.array([(rad - delta).astype(mat.dtype), -offd])
    )
    # rad == 0 means a multiple of the identity, for which any unit vector is an eigenvector.
    norm = jnp.linalg.norm(vec)
    vec = jnp.where(norm > 0., vec / jnp.where(norm > 0., norm, 1.),
                    jnp.array([1., 0.], dtype=mat.dtype))
    # Rayleigh quotient: second order in the eigenvector error, so it recovers full precision where
    # the closed form alone reaches only sqrt(eps).
    return jnp.vdot(vec, jnp.dot(balanced, vec)).real * scale, vec


def _nullvec_3x3(mat: jax.Array) -> jax.Array:
    """Return a unit null vector of a singular 3x3 Hermitian matrix, robust to any rank.

    Six candidates are generated and the one with the smallest residual :math:`|Mv|` is returned.
    Selecting on the measured residual rather than on a magnitude threshold matters because the
    rank-2 and rank-1 constructions below fail in ways that a threshold cannot cleanly separate: for
    a degenerate eigenvalue the cross products do not vanish but decay only to
    :math:`O(\\epsilon \\|M\\|^2)`, close enough to a genuinely small rank-2 cross product that any
    fixed cutoff misclassifies one case or the other.
    """
    # Rank 2 (simple eigenvalue): the null vector is conj(col_i x col_j). Any single pair can be
    # rank deficient, in which case its cross product vanishes and points nowhere useful, so all
    # three pairings are offered.
    cands = [jnp.cross(mat[:, 0], mat[:, 1]).conjugate(),
             jnp.cross(mat[:, 1], mat[:, 2]).conjugate(),
             jnp.cross(mat[:, 2], mat[:, 0]).conjugate()]
    # Rank 1 (degenerate lowest eigenvalue): every cross product is numerical noise and the null
    # space is the orthogonal complement of the largest column; any member of it is an eigenvector.
    col = mat[:, jnp.argmax(jnp.sum(jnp.square(jnp.abs(mat)), axis=0))].conjugate()
    zero = jnp.zeros((), dtype=mat.dtype)
    cands += [jnp.stack([zero, col[2], -col[1]]),
              jnp.stack([-col[2], zero, col[0]]),
              jnp.stack([col[1], -col[0], zero])]
    # Rank 0 (a multiple of the identity): every candidate above is zero, so offer an arbitrary
    # unit vector as the last resort. It has residual 0 and wins by default.
    cands.append(jnp.array([1., 0., 0.], dtype=mat.dtype))

    cands = jnp.stack([_normalize_or_zero(c) for c in cands])
    resid = jnp.linalg.norm(jnp.einsum('ij,cj->ci', mat, cands), axis=1)
    # A candidate that collapsed to zero is not a valid eigenvector; disqualify it.
    resid = jnp.where(jnp.linalg.norm(cands, axis=1) > 0.5, resid, jnp.inf)
    return cands[jnp.argmin(resid)]


def _normalize_or_zero(vector: jax.Array) -> jax.Array:
    norm = jnp.linalg.norm(vector)
    return vector / jnp.where(norm == 0., 1., norm)


@jax.jit
def eigenpair_3x3(mat: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Return the lowest eigenpair of a 3x3 Hermitian matrix computed via Cardano's method.

    The matrix is balanced -- shifted to be traceless and scaled by its largest entry -- before the
    characteristic polynomial is formed. Without this, the coefficients of a matrix with a large
    trace lose all significance to cancellation and the radicand of the square root below goes
    negative, yielding NaN.

    Args:
        mat: A 3x3 Hermitian matrix. Only the diagonal and the lower triangle are read.

    Returns:
        The smallest eigenvalue and its normalized eigenvector.

    Reference:
        J. Kopp, Efficient numerical diagonalization of hermitian 3 x 3 matrices,
        Int. J. Mod. Phys. C. 19, 523 (2008).
    """
    eye = jnp.eye(3, dtype=mat.dtype)
    d = jnp.diagonal(mat).real
    shift = jnp.sum(d) / 3.
    scale = jnp.max(jnp.abs(mat))
    scale = jnp.where(scale > 0., scale, 1.).astype(d.dtype)
    balanced = (mat - shift * eye) / scale

    bd = jnp.diagonal(balanced).real
    modod = jnp.square(jnp.abs(balanced[jnp.array([1, 2, 2]), jnp.array([0, 0, 1])]))
    # Characteristic polynomial of the traceless balanced matrix: x^3 + c1 x + c0.
    c1 = jnp.sum(bd * jnp.roll(bd, 1)) - jnp.sum(modod)
    c0 = jnp.sum(bd * modod[::-1]) - jnp.prod(bd) - 2. * (
        balanced[0, 2] * balanced[1, 0] * balanced[2, 1]).real
    # Both radicands are non-negative for a Hermitian matrix; clamp them against rounding.
    p = jnp.maximum(-3. * c1, 0.)
    disc = jnp.maximum(-27. * c1 * c1 * c1 - 182.25 * c0 * c0, 0.)
    phi = jnp.atan2(jnp.sqrt(disc), -13.5 * c0) / 3.
    cphi = jnp.cos(phi)
    sphi = jnp.sin(phi)
    # Roots are (sqrt(p) / 3) {2 cos(phi), 2 cos(phi -+ 2pi/3)}.
    xmin = jnp.min(jnp.array([2. * cphi, -cphi - _SQRT3 * sphi, -cphi + _SQRT3 * sphi]))
    xmin *= jnp.sqrt(p) / 3.

    vec = _nullvec_3x3(balanced - xmin * eye)
    # Rayleigh quotient: second order in the eigenvector error, so it recovers full precision where
    # Cardano alone reaches only sqrt(eps) (a near-degenerate lowest pair).
    return jnp.vdot(vec, jnp.dot(balanced, vec)).real * scale + shift, vec
