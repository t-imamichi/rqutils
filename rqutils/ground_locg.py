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
return a plausible number that is simply wrong, rather than raising or producing ``NaN``.

The measurements behind each item below were originally recorded in ``docs/locg.md``. **That
document describes the pre-rewrite module and is now stale** -- its line numbers, its "no pytest
suite exists" scope note, and several of its severity claims no longer hold, and at least one
failure mode it measured is no longer reachable now that the defects it compounded with are fixed
(see :func:`_reorthogonalize`). Read it as history; the invariant that is actually binding is the
one stated here next to the code, and ``tests/test_ground_locg.py`` is what enforces it.

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

Being second order in the eigenvector *angle* error, the polish repairs the eigenvalue and leaves
the eigenvector as computed. Those are not equally good: for a near-degenerate lowest pair the
returned :math:`v` can be nearly orthogonal to the true eigenvector while :math:`\theta` is still
accurate to ten digits (measured :math:`|\langle v_{\mathrm{true}} | v \rangle| = 0.447` against a
:math:`\theta` error of 1.2e-10). :func:`_nullvec_3x3`'s cross products are the fragile step, and
once they lose the eigenvector the polish has nothing to recover from. So when auditing this module,
check the eigenvector and not only :math:`\theta` -- the caller propagates the vector, since
:math:`\kappa` becomes the next iteration's search direction. It has *not* been shown that the
iteration ever builds such a projected matrix, since :func:`_project_out` keeps the basis
orthonormal by construction.

Iteration
---------

- **Convergence threshold.** The relative tolerance is
  :math:`(\|Ax\| + |\theta|)\,n \cdot 10`. The natural-looking ``norm(Ax) - theta`` is a difference
  of two nearly equal large positive numbers for a positive-definite operator, and was measured
  going *negative*, which makes the test unsatisfiable so the solver never converges and always
  exhausts ``maxiter``. Note :math:`|\theta|` rather than :math:`+\theta`: for the
  negative-definite operators typical of a ground-state search, :math:`+\theta` cancels in turn.

- **Basis orthogonality.** :math:`t` is re-orthogonalized against the new :math:`x` before
  normalization, or :math:`y` drifts into :math:`x` and the standard Rayleigh-Ritz step returns a
  :math:`\theta` below the true minimum. See :func:`_reorthogonalize`, whose docstring records the
  measured drift and how to A/B it correctly.

- **Search direction normalization.** :func:`_project_out` guarantees only
  :math:`\|p\| \ge 0.99`, and a short :math:`p` scales :math:`\mathrm{sas}_{22}` by :math:`|p|^2`,
  which for a large positive shift is a spuriously low diagonal that Rayleigh-Ritz then selects.
  :math:`p` is renormalized before the projected eigensolve, using the norm :func:`_project_out`
  returns alongside it rather than a second reduction over a vector of up to :math:`10^8` elements.

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
described above introduces two additional inner products per iteration, following the same reduction
pattern as the existing ones. ``examples/scaling/poc7_sharding.py`` exercises this on a four-device
mesh (virtual CPU devices via ``XLA_FLAGS=--xla_force_host_platform_device_count=4``), agreeing with
the single-device result to 8.9e-16; real multi-GPU behaviour remains unverified.

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

import logging
import math
from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec, get_abstract_mesh
from numpy.typing import DTypeLike, NDArray

_SQRT3 = math.sqrt(3.0)

# ground_locg's return is (eigval, eigvec, niter, converged), with the per-iteration diagnostics
# appended when debug=True. Deliberately plain tuple aliases rather than a dataclass: the arity, not
# the anonymity of the positions, is what a type checker needs to see here, and `ground_locg` is
# published API that every caller destructures positionally (sqd.py, two scaling POCs, a benchmark
# under examples/). A caller reading the fifth element must narrow first -- `assert len(result) == 5`
# -- which is also the only way a checker can tell the debug path was requested, since `debug` is
# static.
_Result = tuple[float, NDArray, int, bool]
_DebugResult = tuple[float, NDArray, int, bool, dict[str, jax.Array]]


class _Seed(NamedTuple):
    """Output of the steepest-descent seed step, before a y direction exists."""

    x: jax.Array
    r: jax.Array
    ax: jax.Array
    rho: jax.Array


class _State(NamedTuple):
    """The ``while_loop`` / ``scan`` carry.

    A NamedTuple rather than a bare tuple because this is a registered pytree -- it flattens to the
    identical carry with no extra ops -- and the fields were previously read positionally as
    ``state[0]``, ``state[1]``, ``state[-4:]`` and assembled by splicing one function's return tuple
    into a different order. Inserting a field silently shifted every index, with all four vector
    entries the same shape and both scalars interchangeable, so nothing would have failed loudly.

    ``ax`` is carried so that the projected matrix can reuse the image of ``x`` computed at the end
    of the previous iteration -- three matrix-vector products per iteration instead of four.
    """

    niter: jax.Array | int
    converged: jax.Array
    theta: jax.Array
    x: jax.Array
    y: jax.Array
    r: jax.Array
    ax: jax.Array


def normalize(vector: jax.Array, norm: jax.Array | None = None) -> jax.Array:
    """Divide by the norm, leaving a zero vector untouched instead of producing NaN.

    Pass ``norm`` when the caller has already computed it -- several call sites need the norm itself
    for a separate zero test, and recomputing it here would cost an extra reduction per iteration.
    """
    if norm is None:
        norm = jnp.linalg.norm(vector)
    return vector / jnp.where(norm == 0.0, 1.0, norm)


def _lambda_max_bound(
    matvec: Callable[..., jax.Array],
    args: tuple,
    vector: jax.Array,
    steps: int = 10,
    margin: float = 1.05,
) -> jax.Array:
    r"""Upper bound on :math:`\lambda_{\max}` from ``steps`` of power iteration, times ``margin``.

    Only an *upper* bound is needed, which is why this is cheap where a lower bound on
    :math:`\lambda_0` is not: power iteration converges to the dominant eigenvalue from below, so the
    ``margin`` covers the shortfall. ``docs/rqutils-precond-request.md`` closes the shift line for want
    of a usable lower bound; nothing here needs one.

    ``margin`` is a *multiplicative* slack on the magnitude, added as ``|estimate| * (margin - 1)``
    rather than as ``estimate * margin``, so it widens the interval for either sign -- an operator whose
    dominant eigenvalue is negative would otherwise have its bound moved the wrong way.

    Preserves sharding by construction: every operation is the caller's ``matvec`` or an elementwise
    scale, so the result inherits ``vector``'s sharding the same way :func:`normalize` does.
    """

    def step(vec, _):
        return normalize(matvec(vec, *args)), None

    vec = jax.lax.scan(step, normalize(vector), None, length=steps)[0]
    estimate = jnp.sum(vec.conjugate() * matvec(vec, *args)).real
    return estimate + jnp.abs(estimate) * (margin - 1.0)


def _chebyshev_prefilter(
    matvec: Callable[..., jax.Array],
    args: tuple,
    vector: jax.Array,
    degree: int,
    cycles: int,
) -> jax.Array:
    r"""Damp the unwanted band of the spectrum before the LOBPCG iteration starts.

    Applies :math:`T_{\text{degree}}` of :math:`(A - c)/e` mapping ``[theta, hi]`` onto
    :math:`[-1, 1]`, ``cycles`` times. Eigenvalues inside that band are damped by
    :math:`1/T_{\text{degree}}`; anything below ``theta`` sits outside :math:`[-1, 1]` and grows like
    :math:`\cosh`, so the ground state comes out amplified relative to everything else. Cost is
    ``cycles * (degree + 1)`` matrix-vector products and three live vectors, independent of ``degree``.

    THE LOWER EDGE IS THE CURRENT RAYLEIGH QUOTIENT, RE-READ EACH CYCLE, AND THAT CHOICE IS LOAD-BEARING
    -- not an approximation to a better bound. Using an accurate ``lambda_1`` instead is faster where the
    gap is comfortable and **returns a wrong answer** where it is not: measured 8.1x at relgap 1.3e-2 but
    an energy off by 15, silently, at relgap 4.0e-05, because the filter interval then begins at
    ``lambda_0`` and damps the ground state along with the rest. A Rayleigh quotient starts *above*
    ``lambda_0`` and descends toward it, so it can never bracket the target out.

    Filtering alone does not converge: as ``theta`` approaches ``lambda_0`` the lower edge does too, so
    the filter begins attacking its own target and accuracy plateaus around 1e-5 to 1e-7. That is why
    this is a *prefilter* handing off to the full iteration rather than a solver -- see
    ``docs/locg-chebyshev-prefilter.md`` for both measurements.

    Note this does **not** reproduce the depleted-residual failure that makes a power-iteration start
    *worse* than a random one (measured 177 LOBPCG iterations against 77). Power iteration collapses onto
    the dominant direction, leaving a residual with nothing left to expose; a polynomial filter
    suppresses the unwanted band multiplicatively and leaves the residual rich in the directions
    block-size-1 LOBPCG can actually search.
    """
    hi = _lambda_max_bound(matvec, args, vector)

    def cycle(vec, _):
        theta = jnp.sum(vec.conjugate() * matvec(vec, *args)).real
        centre = (hi + theta) / 2
        half = (hi - theta) / 2
        # A degenerate interval would divide by zero. It means theta has reached hi, i.e. the iterate
        # is at the top of the spectrum rather than the bottom, so there is nothing to filter.
        half = jnp.where(half == 0.0, 1.0, half)

        def term(carry, _):
            previous, current = carry
            nxt = 2.0 * (matvec(current, *args) - centre * current) / half - previous
            return (current, nxt), None

        first = (matvec(vec, *args) - centre * vec) / half
        return normalize(jax.lax.scan(term, (vec, first), None, length=degree - 1)[0][1]), None

    return jax.lax.scan(cycle, normalize(vector), None, length=cycles)[0]


def ground_locg(
    mat: Callable[[jax.Array], jax.Array] | jax.Array,
    xinit: jax.Array | int,
    args: tuple = (),
    maxiter: int = 1000,
    tol: float | None = None,
    vspace: tuple[int, DTypeLike] | None = None,
    precond: Callable[[jax.Array], jax.Array] | None = None,
    prefilter: tuple[int, int] | None = None,
    debug: bool = False,
    log_level: int = logging.WARNING,
) -> _Result | _DebugResult:
    r"""Single-vector LOBPCG.

    Args:
        mat: Matrix :math:`A`, either as an Array or a function :math:`x \mapsto Ax`. **On a mesh, a
            callable must preserve its input's sharding in the output** -- this routine is
            sharding-transparent only through that contract, which is why every ``apply_*`` in
            :mod:`rqutils.sqd` passes ``out_sharding=jax.typeof(vec).sharding``.
        xinit: Initial vector, or an integer index selecting a one-hot vector (which requires
            ``vspace`` if ``mat`` is callable). A plain Python ``int`` is accepted -- the
            implementations inspect ``xinit.dtype``, but both are ``jax.jit``-wrapped, so an ``int``
            arrives as a 0-d traced array. Must have a non-vanishing overlap with :math:`v_0`.
        args: Additional arguments to callable ``mat``.
        maxiter: Maximum number of gradient descent iterations.
        tol: Convergence condition. If None, the machine epsilon of the operator dtype is used.
        vspace: Specification (dimension, dtype) of the vector space. Required only when ``mat`` is
            a callable and ``xinit`` is an integer.
        precond: Optional approximate inverse :math:`M^{-1}`, applied to the residual where the search
            direction is formed -- the "P" in LOBPCG. ``None`` (the default) is the identity path and
            leaves the traced graph unchanged, so no existing caller is affected; it is a static
            argument, so the branch resolves at trace time. **The callable must preserve its input's
            sharding in the output**, the same contract ``mat`` carries above; a Jacobi preconditioner
            is an elementwise multiply, which does so for free. It must not be relied on to change the
            answer -- only the path: every convergence test still reads the **true** residual, so a
            preconditioner that shrinks a direction cannot make the routine stop early. Measured on a
            12-instance XXZ batch, :math:`M^{-1} = \mathrm{diag}(A)^{-1}` gave a median **1.79x**
            reduction in iterations (range 1.29-2.04x, no regressions), at an :math:`O(N)` cost of
            under 1% of an iteration.
        prefilter: Optional ``(degree, cycles)`` Chebyshev prefilter applied to ``xinit`` before the
            iteration starts, damping the unwanted band of the spectrum so the LOBPCG loop begins
            closer to :math:`v_0`. ``None`` (the default) leaves the traced graph unchanged, so no
            existing caller is affected; it is a static argument, so the branch resolves at trace
            time. Costs ``cycles * (degree + 1)`` extra matrix-vector products plus ~11 for the
            :math:`\lambda_{\max}` estimate, and three live vectors -- **no growing basis**, which is
            what makes it compatible with this module's single-vector memory budget. Like ``mat`` and
            ``precond`` it is sharding-transparent, since it only calls ``mat`` and scales
            elementwise. **It cannot change the answer, only the path**: every convergence test still
            reads the true residual, and the returned eigenpair is the same one to the tolerance the
            solver was going to reach anyway (measured: eigenvector overlap 1.0000000 against the
            unfiltered result, energies agreeing with ``eigsh(tol=0)`` to 2.8e-14 across 18
            configurations). ``(16, 4)`` measured a 1.11-3.87x wall-clock reduction, median 1.38x,
            with no configuration slower; see ``docs/locg-chebyshev-prefilter.md`` for the
            measurements, why the filter's lower edge must be the running Rayleigh quotient rather
            than an accurate :math:`\lambda_1`, and why filtering alone does not converge.
        debug: If True, additionally return per-iteration diagnostics. Note that the diagnostic
            path uses ``jax.lax.scan`` to collect fixed-size output, and therefore always runs the
            full ``maxiter`` iterations with no early exit; rows past convergence are
            post-convergence noise.
        log_level: Verbosity level.

    Returns:
        ``(eigval, eigvec, niter, converged)`` -- the smallest eigenvalue, its eigenvector, the
        number of gradient descent iterations performed, and whether the convergence criterion was
        met. Check the fourth value rather than comparing the third against ``maxiter``, which is
        ambiguous. With ``debug=True`` a fifth element is appended, a dict of stacked per-iteration
        diagnostics; narrow on ``len(result) == 5`` before reading it, since ``debug`` is a static
        flag that a type checker cannot follow into the return arity.

    Raises:
        ValueError: If ``xinit`` is an integer and ``mat`` is a callable but ``vspace`` is None. The
            vector space cannot be inferred from a callable, and without this the one-hot
            construction would fail with an opaque "NoneType is not subscriptable".
    """
    if callable(mat):
        return _ground_locg_callable(
            mat,
            xinit,
            args,
            maxiter,
            tol,
            vspace=vspace,
            precond=precond,
            prefilter=prefilter,
            debug=debug,
            log_level=log_level,
        )
    return _ground_locg_matrix(
        mat,
        xinit,
        maxiter,
        tol,
        precond=precond,
        prefilter=prefilter,
        debug=debug,
        log_level=log_level,
    )


@jax.jit(static_argnames=["maxiter", "precond", "prefilter", "debug", "log_level"])
def _ground_locg_matrix(
    mat: jax.Array,
    xinit: jax.Array,
    maxiter: int,
    tol: jax.Array | float | None,
    precond: Callable[[jax.Array], jax.Array] | None = None,
    prefilter: tuple[int, int] | None = None,
    debug: bool = False,
    log_level: int = logging.WARNING,
):
    vspace = None
    if jnp.issubdtype(xinit.dtype, jnp.integer):
        vspace = (mat.shape[1], mat.dtype)

    def matvec(x):
        return jax.lax.dot(
            mat, x, precision=(jax.lax.Precision.HIGHEST,) * 2, out_sharding=jax.typeof(x).sharding
        )

    return _ground_locg_callable(
        matvec,
        xinit,
        (),
        maxiter,
        tol,
        vspace=vspace,
        precond=precond,
        prefilter=prefilter,
        debug=debug,
        log_level=log_level,
    )


@jax.jit(
    static_argnames=[
        "matvec",
        "maxiter",
        "vspace",
        "precond",
        "prefilter",
        "debug",
        "log_level",
    ]
)
def _ground_locg_callable(
    matvec: Callable[[jax.Array], jax.Array],
    xinit: jax.Array | int,
    args: tuple,
    maxiter: int,
    tol: jax.Array | float | None,
    vspace: tuple[int, DTypeLike] | None = None,
    precond: Callable[[jax.Array], jax.Array] | None = None,
    prefilter: tuple[int, int] | None = None,
    debug: bool = False,
    log_level: int = logging.WARNING,
):
    if jnp.issubdtype(xinit.dtype, jnp.integer):
        if vspace is None:
            # Without this, the subscripts below raise an opaque "NoneType is not subscriptable".
            raise ValueError(
                "vspace (dimension, dtype) is required when xinit is an integer and mat is a "
                "callable, since the vector space cannot be inferred from a matrix in that case"
            )
        sharding = None
        if not (mesh := get_abstract_mesh()).empty:
            sharding = PartitionSpec(mesh.axis_names)
        xinit = (
            jax.lax.broadcasted_iota(xinit.dtype, (vspace[0],), 0, out_sharding=sharding) == xinit
        ).astype(vspace[1])

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
        sas = compute_sas(
            (xcurr, ycurr, rcurr), tuple(matvec(v, *args) for v in (xcurr, ycurr, rcurr))
        )
        # rho is <x|Ax>, which compute_sas has just computed as the [0, 0] entry. Recomputing it
        # here cost a fourth matvec that XLA did not eliminate.
        rho = sas[0, 0].real

        if kappa is None:
            kappa = jnp.zeros(3, dtype=xcurr.dtype)
        if reltol is None:
            reltol = jnp.array(0.0)
        if converged is None:
            converged = jnp.array(False)

        return {
            "x": xcurr,
            "y": ycurr,
            "r": rcurr,
            "theta": theta,
            "rho": rho,
            "kappa": kappa,
            "sas": sas,
            "reltol": reltol,
            "converged": converged,
        }

    def body_iter0(xcurr):
        """Steepest-descent seed. Returns Ax so no later step recomputes it."""
        xnext = xcurr
        ax = matvec(xcurr, *args)
        rho = jnp.sum(xcurr.conjugate() * ax).real
        rnext = ax - rho * xnext
        seed = _Seed(x=xnext, r=rnext, ax=ax, rho=rho)
        diag = diagnostics(xnext, jnp.zeros_like(xnext), rnext, rho) if debug else None
        return seed, diag

    def body_iter1(xcurr, rcurr, axcurr, rho):
        # Same zero-search-direction guard as body(), adapted from 3x3 to this step's 2x2 projected
        # matrix (basis {x, p} rather than {x, y, p}). An exactly-zero residual means xcurr is
        # already an eigenvector -- e.g. sqd.py's diagonal-Hamiltonian path seeds xinit as the exact
        # one-hot ground state, so this is not a corner case. Without the guard, eigenpair_2x2 sees a
        # sas whose row/col 1 vanish and, for a positive-definite operator, spuriously selects that
        # null direction: theta collapses towards 0 instead of reporting rho, the true answer.
        #
        # The guard reads the RAW residual and the search direction reads the preconditioned one.
        # These were one quantity before `precond` existed, and splitting them is mandatory, not
        # stylistic: `r_is_zero` feeds both the sas[1, 1] masking above and `converged` in the
        # returned state, so routing it through M^-1 would change what counts as a stationary point.
        # A nonzero residual lying near M^-1's small-singular-value direction would then report
        # convergence early. Only the direction is preconditioned.
        norm_r = jnp.linalg.norm(rcurr)
        r_is_zero = norm_r == 0.0
        if precond is None:
            tmp_p = normalize(rcurr, norm_r)
        else:
            rprec = precond(rcurr)
            tmp_p = normalize(rprec, jnp.linalg.norm(rprec))
        # Reuse Ax from body_iter0 rather than recomputing it inside compute_sas.
        sas = compute_sas((xcurr, tmp_p), (axcurr, matvec(tmp_p, *args)))
        # Lift the p diagonal out of contention, serving the same purpose as body()'s mask on
        # sas[2, 2]: the excluded value strictly exceeds any entry still in play, so Rayleigh-Ritz
        # cannot pick it. With p excluded, the 2x2 solve collapses onto x alone, returning
        # theta = rho (the Rayleigh quotient of xcurr, already computed in body_iter0 and passed in)
        # and kappa = [1, 0], so xnext == xcurr below and no new search direction is introduced.
        #
        # Note this bound is *not* body()'s formula specialized to one surviving entry: that form
        # gives rho + |rho| + 1, which is merely 1.0 for the negative rho of a ground-state search.
        # Both bounds are valid -- each exceeds the only retained entry, sas[0, 0] = rho -- but they
        # are different expressions, so don't "unify" them without redoing the bound argument.
        excluded = 2.0 * jnp.abs(rho) + 1.0
        sas = jnp.where(r_is_zero, sas.at[1, 1].set(excluded.astype(sas.dtype)), sas)
        theta, kappa = eigenpair_2x2(sas)
        tmp_t = tmp_p * kappa[0] - xcurr * kappa[1]
        tmp_u = xcurr * kappa[0] + tmp_p * kappa[1]
        xnext = normalize(tmp_u)
        ynext = normalize(_reorthogonalize(tmp_t, xnext))
        axnext = matvec(xnext, *args)
        rnext = axnext - theta * xnext
        # As in body(): a zeroed residual means {x} (here, in place of {x, y}) already spans the
        # relevant space, so no further iteration can lower theta. Report convergence immediately.
        #
        # This flag must seed the loop state rather than a hardcoded False, or while_loop would
        # spend an iteration re-deriving what is already known -- and, worse, feed a zeroed search
        # direction into body()'s Rayleigh-Ritz step.
        state = _State(
            niter=0, converged=r_is_zero, theta=theta, x=xnext, y=ynext, r=rnext, ax=axnext
        )
        diag = (
            diagnostics(xnext, ynext, rnext, theta, jnp.insert(kappa, 1, 0.0), converged=r_is_zero)
            if debug
            else None
        )
        return state, diag

    def body(state):
        xcurr, ycurr, rcurr, axcurr = state.x, state.y, state.r, state.ax
        if log_level <= logging.DEBUG:
            jax.debug.print("LOCG iteration {}", state.niter)

        # Residual basis selection. R should already be orthogonal to X, but projecting out both X
        # and P is needed for good residual convergence. _project_out only guarantees |tmp_p| >= 0.99
        # while the Rayleigh-Ritz step below assumes an orthonormal basis, so tmp_p is renormalized:
        # a short one scales sas[2, 2] by |tmp_p|^2, a spuriously low diagonal that gets selected in
        # place of the true minimizer under a large positive shift.
        #
        # Preconditioning applies here, where the search direction is formed, and nowhere else -- the
        # convergence test below reads `rnext` and must stay on the true residual.
        rdir = rcurr if precond is None else precond(rcurr)
        tmp_p, norm_p = _project_out((xcurr, ycurr), rdir)
        p_is_zero = norm_p == 0.0
        tmp_p = normalize(tmp_p, norm_p)
        # Projected eigensolve. xcurr is the previous iteration's xnext, so its image is already
        # known -- three matvecs per iteration instead of four.
        sas = compute_sas(
            (xcurr, ycurr, tmp_p), (axcurr, matvec(ycurr, *args), matvec(tmp_p, *args))
        )
        # A zeroed tmp_p leaves sas row/col 2 empty, and for a positive-definite A the resulting
        # zero diagonal is the smallest eigenvalue, so Rayleigh-Ritz would pick the null direction
        # and the normalizations below would divide by zero. Lift it out of contention; the
        # p_is_zero case is reported as convergence instead (see below).
        diag_xy = jnp.diagonal(sas).real[:2]
        excluded = jnp.max(diag_xy) + jnp.sum(jnp.abs(diag_xy)) + 1.0
        sas = jnp.where(p_is_zero, sas.at[2, 2].set(excluded.astype(sas.dtype)), sas)
        theta, kappa = eigenpair_3x3(sas)
        # New vectors
        tmp_s = ycurr * kappa[1] + tmp_p * kappa[2]
        norm_s = jnp.linalg.norm(tmp_s)
        tmp_t = tmp_s * (kappa[0] / jnp.where(norm_s == 0.0, 1.0, norm_s)) - xcurr * norm_s
        tmp_u = xcurr * kappa[0] + tmp_s
        xnext = normalize(tmp_u)
        ynext = normalize(_reorthogonalize(tmp_t, xnext))
        axnext = matvec(xnext, *args)
        rnext = axnext - xnext * theta
        # Relative tolerance from the intermediate AX. Convergence is tested by self-consistency of
        # the eigenpair -- |r| small relative to the floating-point error expected from computing the
        # residual -- rather than via an estimated operator norm, which measured too lax (upstream
        # lobpcg_standard; locking of either kind also measured worse there than none).
        #
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
            jax.debug.print("Residual {}, reltol {}, converged: {}", norm_rnext, reltol, converged)

        state = _State(
            niter=state.niter + 1,
            converged=converged,
            theta=theta,
            x=xnext,
            y=ynext,
            r=rnext,
            ax=axnext,
        )
        if debug:
            return state, diagnostics(xnext, ynext, rnext, theta, kappa, reltol, converged)
        return state

    if log_level <= logging.DEBUG:
        jax.debug.print("Performing first LOBPCG steps")

    xinit = normalize(xinit)

    # The projected matrix inherits xinit's dtype at the seed step but the operator's inside the
    # loop, so a lower-precision xinit makes while_loop's carry types disagree on theta. Promote up
    # front. eval_shape reads the operator dtype without spending a matrix-vector product, and
    # astype on a matching dtype is a no-op, so the common path is unaffected.
    #
    # This promotion is also deliberately what fixes a second, independent problem for a complex
    # operator paired with a real xinit -- e.g. complex128 mat with float64 xinit, a natural and
    # previously-working way to call this function. Without it, xinit stayed real through
    # compute_sas's scatter, which raised FutureWarning/ComplexWarning and silently discarded the
    # imaginary part of the projected matrix -- a correctness bug for a genuinely complex operator,
    # not merely a noisy warning. Promoting xinit up front means it is never a carry-type workaround
    # to remove later; both problems share the same fix.
    work_dtype = jnp.result_type(
        xinit.dtype, jax.eval_shape(lambda vec: matvec(vec, *args), xinit).dtype
    )
    xinit = xinit.astype(work_dtype)

    # After the dtype promotion, so the filter's Chebyshev recurrence runs at the operator's
    # precision rather than a lower-precision xinit's, and before body_iter0, so rho_init below is
    # the filtered vector's Rayleigh quotient and maxiter=0 still reports something meaningful.
    if prefilter is not None:
        degree, cycles = prefilter
        if degree > 1 and cycles > 0:
            xinit = _chebyshev_prefilter(matvec, args, xinit, degree, cycles)

    seed, diag0 = body_iter0(xinit)
    # Seed theta with the Rayleigh quotient of xinit so that maxiter=0 returns a meaningful value
    # rather than the state initializer.
    rho_init = seed.rho

    if tol is None:
        # Derive the tolerance from the operator, not from the initial guess: a float32 xinit on a
        # complex128 problem would otherwise silently loosen this by nine orders of magnitude.
        # work_dtype above is already that promotion.
        tol = float(jnp.finfo(work_dtype).eps)

    state, diag1 = body_iter1(seed.x, seed.r, seed.ax, rho_init)
    if debug:
        diag0, diag1 = jax.tree.map(lambda a: jnp.expand_dims(a, 0), (diag0, diag1))

    if maxiter == 0:
        # No iteration is permitted, so report the seed pair.
        empty = jnp.array(False)
        if debug:
            diagnostics_out = jax.tree.map(
                lambda d0, d1: jnp.concatenate([d0, d1], axis=0), diag0, diag1
            )
            return rho_init, xinit, 0, empty, diagnostics_out
        return rho_init, xinit, 0, empty

    if debug:
        state, diagnostics_out = jax.lax.scan(lambda s, _: body(s), state, length=maxiter)
        diagnostics_out = jax.tree.map(
            lambda d0, d1, dr: jnp.concatenate([d0, d1, dr], axis=0), diag0, diag1, diagnostics_out
        )
    else:
        state = jax.lax.while_loop(
            lambda s: jnp.logical_and(s.niter < maxiter, ~s.converged), body, state
        )

    if debug:
        return state.theta, state.x, state.niter, state.converged, diagnostics_out
    return state.theta, state.x, state.niter, state.converged


def _reorthogonalize(vector, against, passes=2):
    """Re-orthogonalize ``vector`` against a single unit vector, repeatedly.

    :math:`t = \\kappa_0 s / |s| - |s| x` is a difference of two quantities both nearly parallel to
    :math:`x` as :math:`|s| \\to 0`, so catastrophic cancellation lets :math:`y` drift into
    :math:`x`. Once :math:`\\langle x | y \\rangle` is :math:`O(1)` the basis is no longer
    orthonormal, and because the Rayleigh-Ritz step solves a *standard* eigenproblem it then returns
    a :math:`\\theta` **below** the true minimum eigenvalue -- a silent wrong answer rather than a
    visible failure. Measured :math:`|\\langle x | y \\rangle| = 1.0` at shift 1e9 without this.

    One pass is not enough for the same reason :func:`_project_out` runs twice; the second removes
    what the first pass's own rounding reintroduced.

    **Measurably load-bearing, and pinned by**
    ``tests/test_ground_locg.py::TestBasisOrthogonality``. Removing it degrades the worst
    :math:`|\\langle x | y \\rangle|` over 60 iterations from ~5e-17 to 2.5e-12 at shift 1e6 and
    **1.0e-08 at shift 1e9** -- eight orders of magnitude -- and that test fails 3 of its 4 arms as a
    result. Note theta still matches ``eigvalsh`` throughout, so *nothing else* in the suite notices:
    the drift is underway but has not yet collapsed the basis, and the audit's
    :math:`|\\langle x|y\\rangle| = 1.0` needed the 2000-iteration runs that the ``reltol`` sign
    error (item I4) used to force, where the fixed solver converges in 8-46. That is why the
    invariant is asserted directly off the ``debug=True`` per-iteration diagnostics rather than by
    waiting for a wrong eigenvalue.

    When A/B-ing this function, patch it in a **fresh subprocess before any tracing**. Both callers
    are ``@jax.jit``-decorated, so reassigning it in a live session silently reuses the compiled
    kernel and both arms return bit-identical numbers that look like "no effect".
    """
    for _ in range(passes):
        vector = vector - against * jnp.sum(against.conjugate() * vector)
    return vector


def _subtract_projections(basis, vector):
    """Subtract the projection of ``vector`` onto each basis element.

    All inner products are taken before any subtraction, so a multi-element basis is projected out
    in one pass rather than sequentially. Deliberately *not* batched into a matmul: reassociating the
    summation order measured consistently worse in the near-degenerate regime this exists for. Over
    4000 adversarial cases with ``r`` placed almost entirely inside ``span(x, y)`` plus an orthogonal
    part of size 1e-14..1e-6, both forms hold residual orthogonality at machine epsilon, but the
    matmul is consistently worse -- worst ``|<b|p>|`` of 8.3e-17 against 6.2e-17. Neither form is
    broken, so this is a judgement call rather than a measured failure: a few ops are not worth a
    33% erosion of the quantity these guards exist to protect. Re-run that comparison before
    "optimizing" this.
    """
    ips = [jnp.sum(vb.conjugate() * vector) for vb in basis]
    for vb, ip in zip(basis, ips):
        vector = vector - vb * ip
    return vector


def _project_out(basis, vector):
    for _ in range(2):
        vector = normalize(_subtract_projections(basis, vector))

    # Must end on a subtraction of the original basis, not the orthonormalization: near convergence
    # with R = 0, catastrophic cancellation re-introduces (X, P) components into U and ruins the
    # Rayleigh-Ritz conditioning. Suspicious vectors are zeroed to keep [basis, U] zero-or-orthogonal.
    for _ in range(2):
        vector = _subtract_projections(basis, vector)

    # Note the postcondition: the returned vector is either exactly zero or has norm >= 0.99. It is
    # NOT normalized -- callers feeding it to a standard Rayleigh-Ritz step must renormalize. The
    # norm is returned alongside it because every caller needs it for exactly that, plus the
    # zero test; recomputing it outside would be a second O(N) reduction over a huge vector.
    norm = jnp.linalg.norm(vector)
    return vector * (norm >= 0.99).astype(vector.dtype), jnp.where(norm >= 0.99, norm, 0.0)


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
    scale = jnp.where(scale > 0.0, scale, 1.0).astype(jnp.diagonal(mat).real.dtype)
    balanced = mat / scale
    d = jnp.diagonal(balanced).real
    delta = (d[0] - d[1]) * 0.5
    offd = balanced[1, 0]
    rad = jnp.hypot(delta, jnp.abs(offd))
    # Null vector of T + rad I = [[delta + rad, conj(offd)], [offd, rad - delta]], which is
    # singular. Row 1 gives [-conj(offd), delta + rad] and row 2 gives [rad - delta, -offd]; the two
    # are parallel, but each cancels when its own pivot is small, so select on the sign of delta.
    vec = jnp.where(
        delta >= 0.0,
        jnp.array([-offd.conjugate(), (delta + rad).astype(mat.dtype)]),
        jnp.array([(rad - delta).astype(mat.dtype), -offd]),
    )
    # rad == 0 means a multiple of the identity, for which any unit vector is an eigenvector.
    norm = jnp.linalg.norm(vec)
    vec = jnp.where(norm > 0.0, normalize(vec, norm), jnp.array([1.0, 0.0], dtype=mat.dtype))
    # Rayleigh quotient: second order in the eigenvector error, so it recovers full precision where
    # the closed form alone reaches only sqrt(eps).
    return jnp.vdot(vec, jnp.dot(balanced, vec)).real * scale, vec


def _nullvec_3x3(mat: jax.Array) -> jax.Array:
    """Return a unit null vector of a singular 3x3 Hermitian matrix, robust to any rank.

    Seven candidates are generated and the one with the smallest residual :math:`|Mv|` is returned.
    Selecting on the measured residual rather than on a magnitude threshold matters because the
    rank-2 and rank-1 constructions below fail in ways that a threshold cannot cleanly separate: for
    a degenerate eigenvalue the cross products do not vanish but decay only to
    :math:`O(\\epsilon \\|M\\|^2)`, close enough to a genuinely small rank-2 cross product that any
    fixed cutoff misclassifies one case or the other.
    """
    # Rank 2 (simple eigenvalue): the null vector is conj(col_i x col_j). Any single pair can be
    # rank deficient, in which case its cross product vanishes and points nowhere useful, so all
    # three pairings are offered.
    cands = [
        jnp.cross(mat[:, 0], mat[:, 1]).conjugate(),
        jnp.cross(mat[:, 1], mat[:, 2]).conjugate(),
        jnp.cross(mat[:, 2], mat[:, 0]).conjugate(),
    ]
    # Rank 1 (degenerate lowest eigenvalue): every cross product is numerical noise and the null
    # space is the orthogonal complement of the largest column; any member of it is an eigenvector.
    col = mat[:, jnp.argmax(jnp.sum(jnp.square(jnp.abs(mat)), axis=0))].conjugate()
    zero = jnp.zeros((), dtype=mat.dtype)
    cands += [
        jnp.stack([zero, col[2], -col[1]]),
        jnp.stack([-col[2], zero, col[0]]),
        jnp.stack([col[1], -col[0], zero]),
    ]
    # Rank 0 (a multiple of the identity): every candidate above is zero, so offer an arbitrary
    # unit vector as the last resort. It has residual 0 and wins by default.
    cands.append(jnp.array([1.0, 0.0, 0.0], dtype=mat.dtype))

    cands = jnp.stack([normalize(c) for c in cands])
    resid = jnp.linalg.norm(jnp.einsum("ij,cj->ci", mat, cands), axis=1)
    # A candidate that collapsed to zero is not a valid eigenvector; disqualify it.
    resid = jnp.where(jnp.linalg.norm(cands, axis=1) > 0.5, resid, jnp.inf)
    return cands[jnp.argmin(resid)]


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
    shift = jnp.sum(d) / 3.0
    scale = jnp.max(jnp.abs(mat))
    scale = jnp.where(scale > 0.0, scale, 1.0).astype(d.dtype)
    balanced = (mat - shift * eye) / scale

    bd = jnp.diagonal(balanced).real
    modod = jnp.square(jnp.abs(balanced[jnp.array([1, 2, 2]), jnp.array([0, 0, 1])]))
    # Characteristic polynomial of the traceless balanced matrix: x^3 + c1 x + c0.
    c1 = jnp.sum(bd * jnp.roll(bd, 1)) - jnp.sum(modod)
    c0 = (
        jnp.sum(bd * modod[::-1])
        - jnp.prod(bd)
        - 2.0 * (balanced[0, 2] * balanced[1, 0] * balanced[2, 1]).real
    )
    # Both radicands are non-negative for a Hermitian matrix; clamp them against rounding.
    # disc is Cardano's p^3 - q^2 for q = -13.5 * c0 (182.25 == 13.5**2), deliberately written out
    # in c1 and c0 rather than as the recognizable p*p*p - q*q: this grouping is measurably *more
    # accurate*, 1.16e-16 mean relative error against 1.92e-16 for the factored form over 200k
    # random inputs versus exact rational arithmetic, and it holds that ~1.7x margin all the way
    # down the near-degenerate sweep where disc -> 0 and cancellation is worst. Reason: p = -3*c1
    # rounds once and cubing triples that error, whereas 27.0 is exact in binary so c1 enters the
    # cube unrounded. Don't "simplify" this into the textbook form.
    p = jnp.maximum(-3.0 * c1, 0.0)
    disc = jnp.maximum(-27.0 * c1 * c1 * c1 - 182.25 * c0 * c0, 0.0)
    phi = jnp.atan2(jnp.sqrt(disc), -13.5 * c0) / 3.0
    cphi = jnp.cos(phi)
    sphi = jnp.sin(phi)
    # Roots are (sqrt(p) / 3) {2 cos(phi), 2 cos(phi -+ 2pi/3)}.
    xmin = jnp.min(jnp.array([2.0 * cphi, -cphi - _SQRT3 * sphi, -cphi + _SQRT3 * sphi]))
    xmin *= jnp.sqrt(p) / 3.0

    vec = _nullvec_3x3(balanced - xmin * eye)
    # Rayleigh quotient: second order in the eigenvector error, so it recovers full precision where
    # Cardano alone reaches only sqrt(eps) (a near-degenerate lowest pair).
    return jnp.vdot(vec, jnp.dot(balanced, vec)).real * scale + shift, vec
