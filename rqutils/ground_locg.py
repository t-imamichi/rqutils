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


Instead of an optimal step size :math:`\alpha_{i}`, Rayleigh-Ritz minimizes :math:`\rho` directly,
and over the extended space :math:`\{x_{i}, x_{i-1}, r_{i}\}`, which converges drastically faster
than :math:`\{x_{i}, r_{i}\}`. With orthogonal :math:`\{x_{i}, y_{i}, r_{i}\}` (:math:`x_{i}, y_{i}`
normal) carried over and :math:`R_A` the Rayleigh-Ritz routine over :math:`A`, one iteration is:

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

The measurements behind each item below are in ``markdown/locg.md``, which audits the pre-rewrite
module and is stale; the binding invariant is the one stated here, enforced by
``test/test_ground_locg.py`` (``NOTES.md``, "``ground_locg``: every guard is load-bearing").

Analytic eigenpair kernels
--------------------------

:func:`eigenpair_2x2` and :func:`eigenpair_3x3` must not build the characteristic polynomial from the
*unshifted, unscaled* matrix: for :math:`H = A + sI` its coefficients grow as powers of :math:`s`, so
the large trace of an ordinary physical Hamiltonian destroys the result (``NaN`` for the 3x3 kernel).

Both kernels therefore **balance** first: subtract :math:`\mathrm{tr}/3` (2x2: :math:`\mathrm{tr}/2`)
and divide by :math:`\max_{ij} |A_{ij}|` so the intermediates stay :math:`O(1)`. The eigenvector is
invariant under both; only the eigenvalue is mapped back.

.. math::

    \lambda_{\mathrm{min}}(A) = \sigma \,
    \lambda_{\mathrm{min}}\!\left(\frac{A - \tau I}{\sigma}\right) + \tau,
    \qquad \tau = \frac{\mathrm{tr} A}{n}, \quad \sigma = \max_{ij} |A_{ij}| .


The eigenvector extraction is rank-aware. :func:`eigenpair_3x3` takes the largest of all three
column cross products (any one pair can be rank deficient), falling back to the orthogonal complement
of the largest column at rank 1 (degenerate lowest eigenvalue) and to an arbitrary unit vector at
rank 0 (a multiple of the identity). :func:`eigenpair_2x2` selects between the two rows of the
singular shifted matrix on the sign of :math:`\delta = (d_0 - d_1)/2`, so :math:`\delta` never cancels.

Finally both kernels close with a Rayleigh-quotient polish, :math:`\theta \leftarrow v^{\dagger} B
v` on the balanced matrix. This is second order in the eigenvector error and recovers full
precision where the closed form alone reaches only :math:`\sqrt{\epsilon}` (a near-degenerate lowest
pair). It repairs :math:`\theta`, not :math:`v`, so audit the eigenvector too (``NOTES.md``,
"ground_locg module: the polish repairs the eigenvalue, not the eigenvector").

Iteration
---------

- **Convergence threshold.** Two independent tolerances, satisfied by **either**:

  .. math:: \|r\| < \max\bigl(\mathrm{atol},\ \mathrm{rtol}\,(\|Ax\| + |\theta|)\bigr)

  ``atol`` is an absolute bound on :math:`\|Ax - \theta x\|_2`, holding at every :math:`n`. ``rtol``
  is a fraction of the operator magnitude, as in :func:`numpy.allclose`: its bound is
  :math:`\approx 2\,\mathrm{rtol}\|A\|_2`, independent of the dimension. The achievable floor is

  .. math:: \mathrm{floor}(\|r\|) \approx \varepsilon(\mathrm{dtype}) \cdot \|A\|_2

  with **no** dependence on :math:`n`; ``rtol=None`` targets :math:`8\varepsilon\|A\|_2`, 8x that
  floor (``NOTES.md``, "The eigen-residual floor is ``eps·‖H‖`` with no dimension dependence"). Two
  earlier forms, an :math:`n \cdot 10`-scaled relative ``tol`` and a purely absolute one, are
  superseded (``NOTES.md``, "``atol``/``rtol``: the pair is right").

- **Basis orthogonality.** :math:`t` is re-orthogonalized against the new :math:`x` before
  normalization, or :math:`y` drifts into :math:`x` and the standard Rayleigh-Ritz step returns a
  :math:`\theta` below the true minimum. See :func:`_reorthogonalize`.

- **Search direction normalization.** :func:`_project_out` guarantees only
  :math:`\|p\| \ge 0.99`, and a short :math:`p` scales :math:`\mathrm{sas}_{22}` by :math:`|p|^2`,
  which for a large positive shift is a spuriously low diagonal that Rayleigh-Ritz then selects.
  :math:`p` is renormalized before the projected eigensolve, using the norm :func:`_project_out`
  returns alongside it rather than a second reduction over a vector of up to :math:`10^8` elements.

- **Exhausted search space.** When :func:`_project_out` returns exactly zero, the zero diagonal it
  leaves would be the *smallest* eigenvalue of a positive-definite projected matrix, and normalizing
  the null direction would divide by zero. It is masked out of contention and reported as
  convergence: :math:`\{x, y\}` already spans the residual, so :math:`\theta` cannot fall further.

Every division by a norm in this module is guarded, so a zero or degenerate input yields a
well-defined result instead of ``NaN``.

Chebyshev prefilter
===================

:func:`ground_locg` takes an optional ``prefilter=(degree, cycles)`` that applies a Chebyshev
polynomial filter to the initial vector, damping the band :math:`[\theta, \lambda_{\max}]` by
:math:`1/T_{\mathrm{degree}}` so the ground direction comes out amplified. It is the single-vector,
prefilter-then-LOBPCG specialization of Chebyshev-filtered subspace iteration[3][4][5], not the
two-level method of [5] (``NOTES.md``, "ground_locg module: Chebyshev prefilter provenance").

Two properties are load-bearing: the filter's lower edge is the *current Rayleigh quotient*, re-read
each cycle, not an estimate of :math:`\lambda_1`; and ``prefilter_hi`` must be a true upper bound on
:math:`\lambda_{\max}`, which no matvec-based iteration can supply. :func:`_chebyshev_prefilter` has
both; ``markdown/locg-chebyshev-prefilter.md`` has the tuning tables.

Distributed arrays
==================

This function works transparently over distributed (sharded) input :math:`v_0` if the callable
passed as the ``mat`` argument preserves the sharding in the output (``NOTES.md``, "ground_locg
module: sharding verification").

References
==========

[1]: https://en.wikipedia.org/wiki/LOBPCG

[2]: J. Kopp, *Efficient numerical diagonalization of hermitian 3 x 3 matrices*,
Int. J. Mod. Phys. C. **19**, 523 (2008).

[3]: A. S. Banerjee, L. Lin, W. Hu, C. Yang, J. E. Pask, *Chebyshev polynomial filtered subspace
iteration in the discontinuous Galerkin method for large-scale electronic structure calculations*,
J. Chem. Phys. **145**, 154101 (2016).

[4]: Y. Zhou, Y. Saad, M. L. Tiago, J. R. Chelikowsky, *Self-consistent-field calculations using
Chebyshev-filtered subspace iteration*, J. Comput. Phys. **219**, 172 (2006).

[5]: A. S. Banerjee, L. Lin, P. Suryanarayana, C. Yang, J. E. Pask, *Two-level Chebyshev filter based
complementary subspace method*, J. Chem. Theory Comput. **14**, 2930 (2018).

Single-vector LOBPCG API
========================

.. autofunction:: ground_locg

.. autofunction:: eigenpair_2x2

.. autofunction:: eigenpair_3x3
"""

import logging
import math
from collections.abc import Callable
from typing import Any, Literal, NamedTuple, overload

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import PartitionSpec, get_abstract_mesh
from numpy.typing import DTypeLike, NDArray

_SQRT3 = math.sqrt(3.0)

# Plain tuple aliases, not a dataclass: `ground_locg` is published API that callers destructure
# positionally; the overloads below give a checker the arity (5 elements only with debug=True).
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


class _LocgContext(NamedTuple):
    """What the step functions share, closed over rather than carried: static fields stay static."""

    matvec: Callable[..., jax.Array]
    args: tuple
    atol: jax.Array | float
    rtol: jax.Array | float
    batch_matvec: bool
    debug: bool
    log_level: int


def normalize(vector: jax.Array, norm: jax.Array | None = None) -> jax.Array:
    """Divide by the norm, leaving a zero vector untouched instead of producing NaN.

    Pass ``norm`` when the caller has already computed it -- several call sites need the norm itself
    for a separate zero test, and recomputing it here would cost an extra reduction per iteration.
    """
    if norm is None:
        norm = jnp.linalg.norm(vector)
    return vector / jnp.where(norm == 0.0, 1.0, norm)


def residual_floor(opnorm_bound: float, dtype: DTypeLike) -> float:
    r"""Return the smallest eigen-residual a solve on this operator can reach.

    The achievable floor of :math:`\|Ax - \theta x\|_2` is :math:`\varepsilon \cdot \|A\|_2`,
    **independent of the dimension**; this returns **4** times that, a 3.2x margin over the worst
    measured constant (``NOTES.md``, "ground_locg.residual_floor: the measured constant").

    ``opnorm_bound`` may be any upper bound on :math:`\|A\|_2`; :func:`rqutils.sqd.sqd` passes
    :math:`\sum_k |c_k|`. Over-estimating is the safe direction: it raises the reported floor.

    Args:
        opnorm_bound: An upper bound on the spectral norm of the operator.
        dtype: The operator's dtype; its machine epsilon sets the scale.

    Returns:
        The smallest ``tol`` worth passing. A solve requesting less cannot converge.
    """
    return 4.0 * float(np.finfo(np.dtype(dtype)).eps) * float(opnorm_bound)


def _check_tols(atol: Any, rtol: Any, opnorm_bound: float, dtype: DTypeLike) -> None:
    r"""Raise unless ``atol``/``rtol`` are usable tolerances for :func:`ground_locg`.

    The convergence test is ``||r|| < max(atol, rtol * scale)`` -- satisfied by **either** arm -- so the
    two are validated differently:

    * ``atol`` is an absolute residual bound. ``None`` is **rejected**: a derived absolute bound is the
      unintuitive construct this pair replaced, and 0.0 already expresses "no absolute arm". Negative is
      rejected; 0.0 is legal and means exactly that.
    * ``rtol`` is a fraction of ``||Ax|| + |theta|``, i.e. dimension-independent. ``None`` is accepted
      and resolves to ``4 * eps`` (see :func:`ground_locg`). Negative is rejected, 0.0 disables the arm,
      and ``>= 0.5`` is rejected because the bound would then reach ``||A||`` and any vector would pass.

    Only ``rtol`` takes ``None`` because its default is the promoted dtype's epsilon, which no literal
    can express; ``atol`` has no dtype-derived value (``NOTES.md``, "ground_locg._check_tols: why only
    rtol takes None").

    The below-floor check fires only when ``rtol == 0``: with a live relative arm a below-floor ``atol``
    is harmless, and rejecting it would fail a working call. Both arms zero is rejected outright, since
    no residual satisfies ``||r|| < 0``.

    Sited here beside the test it guards (``NOTES.md``, "Validation belongs to the module that owns the
    gate"), but called by :func:`rqutils.sqd.sqd`, the outermost point where :math:`\sum_k |c_k|` is
    concrete: ``converged`` is traced inside a ``while_loop``, so nothing there can raise.

    Args:
        atol: The caller's absolute tolerance, unvalidated.
        rtol: The caller's relative tolerance, unvalidated.
        opnorm_bound: An upper bound on the operator's spectral norm.
        dtype: The operator dtype whose epsilon sets the floor.

    Raises:
        TypeError: If ``atol`` is not a real number, or ``rtol`` is neither None nor a real number.
            Note ``atol=None`` raises :class:`ValueError`, not this -- it is a rejected *value* with a
            documented replacement (``0.0``), not an unusable type.
        ValueError: If ``atol`` is ``None``; if either is negative; if both are zero; if ``atol`` is
            below the achievable residual floor while ``rtol`` is zero; or if ``rtol`` is at least 0.5,
            where its bound reaches ``||A||`` and any vector would report convergence.
    """
    if atol is None:
        raise ValueError(
            "atol must be a number, not None -- pass 0.0 to disable the absolute arm and rely on "
            "rtol. Only rtol accepts None (it resolves to the operator dtype's epsilon)."
        )
    for name, value in (("atol", atol), ("rtol", rtol)):
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float, np.floating, np.integer)):
            raise TypeError(
                f"{name} must be {'None or ' if name == 'rtol' else ''}a real number, got "
                f"{type(value).__name__}"
            )
        if float(value) < 0.0:
            raise ValueError(f"{name} must be non-negative, got {float(value)!r}")

    atol = float(atol)
    # None means "derive from dtype", which is always positive, so it cannot make both arms zero.
    rtol_is_zero = rtol is not None and float(rtol) == 0.0
    if atol == 0.0 and rtol_is_zero:
        raise ValueError(
            "atol and rtol are both 0.0, so no residual can satisfy the convergence test and the "
            "solve would exhaust maxiter. Set atol to the residual you need, or rtol=None to use "
            "4*eps as a fraction of (||Hv|| + |E|)."
        )
    # A bound reaching ||H|| accepts any vector (||Hv - Ev|| <= ||H||), so both arms are checked,
    # loosely (NOTES.md, "ground_locg._check_tols: the accept-anything cutoffs").
    if rtol is not None and float(rtol) >= 0.5:
        raise ValueError(
            f"rtol={float(rtol):.3e} makes the convergence bound reach the operator norm "
            f"(the scale is ||Hv|| + |E| <= 2||H||, so the bound is up to {2 * float(rtol):.2f}*||H||). "
            "Every normalized vector satisfies ||Hv - Ev|| <= ||H||, so the first iterate would report "
            "convergence and the returned eigenpair would be arbitrary. rtol is a *fraction* of the "
            "operator magnitude: pass something well below 0.5, or use atol for an absolute bound."
        )
    if atol >= float(opnorm_bound):
        raise ValueError(
            f"atol={atol:.3e} is at or above the operator norm bound sum|c_k|="
            f"{float(opnorm_bound):.4g}, so it accepts anything: every normalized vector satisfies "
            "||Hv - Ev|| <= ||H||, and the solve would report convergence on its first iterate with an "
            "arbitrary eigenpair (measured: atol=100 against ||H||=17 converged in 1 iteration). Pass "
            "an atol well below the operator scale."
        )
    if atol > 0.0 and rtol_is_zero:
        floor = residual_floor(opnorm_bound, dtype)
        if atol < floor:
            raise ValueError(
                f"atol={atol:.3e} is below the achievable eigen-residual floor {floor:.3e} for this "
                f"operator (4 * eps * sum|c_k|, with sum|c_k|={float(opnorm_bound):.4g} bounding "
                f"||H||_2) and rtol=0 leaves no other arm, so the solve could never converge and "
                f"would exhaust maxiter. The floor is eps*||H||_2 and does not shrink with subspace "
                f"size -- measured over n=70..32768 and six decades of ||H||. Pass "
                f"atol >= {floor:.3e}, or a non-zero rtol."
            )


def _check_prefilter(prefilter: Any) -> None:
    """Raise unless ``prefilter`` is None or a ``(degree, cycles)`` pair of non-negative ints.

    Lives here because this module owns the ``degree > 1 and cycles > 0`` gate, which absorbs an
    out-of-range value into a silent no-op rather than reporting it (``NOTES.md``,
    "ground_locg._check_prefilter: what the gate absorbed").

    ``bool`` is rejected because it is an ``int`` subclass, so ``(True, 2)`` would otherwise pass as
    ``(1, 2)``, i.e. as a documented no-op.

    The **intentional** no-ops stay legal: ``degree <= 1`` or ``cycles == 0`` disables the filter
    without restructuring a sweep (pinned by ``TestChebyshevPrefilter``). Only negative values,
    non-ints and wrong shapes are errors.

    Args:
        prefilter: The caller's value, unvalidated.

    Raises:
        TypeError: If it is not None or a length-2 sequence of ints.
        ValueError: If either entry is negative.
    """
    if prefilter is None:
        return
    try:
        degree, cycles = prefilter
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"`prefilter` must be None or a (degree, cycles) pair of ints, got {prefilter!r}"
        ) from exc
    if not all(isinstance(v, int) and not isinstance(v, bool) for v in (degree, cycles)):
        raise TypeError(
            f"`prefilter` must be None or a (degree, cycles) pair of ints, got {prefilter!r}"
        )
    if degree < 0 or cycles < 0:
        raise ValueError(
            f"`prefilter` is {prefilter!r}, but both entries must be non-negative. A negative value "
            "is silently absorbed as a no-op by the filter's `degree > 1 and cycles > 0` gate, so it "
            "returns the unfiltered energy at zero speedup rather than reporting anything. To disable "
            "the filter deliberately, pass None (or degree<=1 / cycles=0, which stay legal)."
        )


def _gershgorin_bound(mat: jax.Array) -> jax.Array:
    r"""Rigorous upper bound on :math:`\lambda_{\max}` of a Hermitian array: ``max_i sum_j |A_ij|``.

    Gershgorin: every eigenvalue lies within :math:`\sum_j |A_{ij}|` of some :math:`A_{ii}`, so this
    bounds the spectral radius. **Rigorous for every Hermitian input**, with no iteration: one
    :math:`O(N^2)` reduction (``NOTES.md``, "ground_locg._gershgorin_bound: cost").

    Preserves sharding: an elementwise ``abs`` and two reductions, so the scalar result carries no
    partitioning to conflict with the caller's vector.
    """
    return jnp.abs(mat).sum(axis=-1).max()


def _chebyshev_prefilter(
    matvec: Callable[..., jax.Array],
    args: tuple,
    vector: jax.Array,
    degree: int,
    cycles: int,
    hi: jax.Array | float,
) -> jax.Array:
    r"""Damp the unwanted band of the spectrum before the LOBPCG iteration starts.

    ``hi`` MUST BE A TRUE UPPER BOUND ON :math:`\lambda_{\max}`; an under-estimate makes the filter
    damp its own target, returning an *excited* eigenpair with ``converged=True``. No matvec-only upper
    bound exists (a theorem), so it comes from structure: Gershgorin for an array
    (:func:`_gershgorin_bound`), :math:`\sum_k |c_k|` for a Pauli sum (``NOTES.md``, "No matvec-only
    upper bound on ``λ_max`` exists"; "ground_locg._chebyshev_prefilter: the measurements").

    Prefer a loose bound to a tight estimate: over-estimating costs resolution smoothly, while
    under-estimating flips which eigenvector is amplified most and returns the wrong answer.

    Applies :math:`T_{\text{degree}}` of :math:`(A - c)/e` mapping ``[theta, hi]`` onto
    :math:`[-1, 1]`, ``cycles`` times. Eigenvalues inside that band are damped by
    :math:`1/T_{\text{degree}}`; anything below ``theta`` sits outside :math:`[-1, 1]` and grows like
    :math:`\cosh`, so the ground state comes out amplified relative to everything else. Cost is
    ``cycles * (degree + 1)`` matrix-vector products and three live vectors, independent of ``degree``.

    THE LOWER EDGE IS THE CURRENT RAYLEIGH QUOTIENT, RE-READ EACH CYCLE, AND THAT IS LOAD-BEARING: it
    starts above ``lambda_0`` and descends, so it never brackets the target out, where an accurate
    ``lambda_1`` returned a wrong answer on a tight gap. Filtering alone plateaus as ``theta`` nears
    ``lambda_0``, hence a *prefilter* handing off to the iteration. Unlike a power-iteration start it
    leaves the residual rich in searchable directions (``markdown/locg-chebyshev-prefilter.md``).
    """

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


# Annotation only; the overloads repeat the full positional order because callers pass `maxiter`
# positionally (`poc/mixed_precision.py`), and a non-literal bool gets the union.
@overload
def ground_locg(
    mat: Callable[[jax.Array], jax.Array] | jax.Array,
    xinit: jax.Array | int,
    args: tuple = ...,
    maxiter: int = ...,
    atol: float = ...,
    rtol: float | None = ...,
    vspace: tuple[int, DTypeLike] | None = ...,
    prefilter: tuple[int, int] | None = ...,
    prefilter_hi: float | None = ...,
    debug: Literal[False] = ...,
    log_level: int = ...,
    batch_matvec: bool = ...,
) -> _Result: ...


@overload
def ground_locg(
    mat: Callable[[jax.Array], jax.Array] | jax.Array,
    xinit: jax.Array | int,
    args: tuple = ...,
    maxiter: int = ...,
    atol: float = ...,
    rtol: float | None = ...,
    vspace: tuple[int, DTypeLike] | None = ...,
    prefilter: tuple[int, int] | None = ...,
    prefilter_hi: float | None = ...,
    *,
    debug: Literal[True],
    log_level: int = ...,
    batch_matvec: bool = ...,
) -> _DebugResult: ...


@overload
def ground_locg(
    mat: Callable[[jax.Array], jax.Array] | jax.Array,
    xinit: jax.Array | int,
    args: tuple,
    maxiter: int,
    atol: float,
    rtol: float | None,
    vspace: tuple[int, DTypeLike] | None,
    prefilter: tuple[int, int] | None,
    prefilter_hi: float | None,
    debug: Literal[True],
    log_level: int = ...,
    batch_matvec: bool = ...,
) -> _DebugResult: ...


@overload
def ground_locg(
    mat: Callable[[jax.Array], jax.Array] | jax.Array,
    xinit: jax.Array | int,
    args: tuple = ...,
    maxiter: int = ...,
    atol: float = ...,
    rtol: float | None = ...,
    vspace: tuple[int, DTypeLike] | None = ...,
    prefilter: tuple[int, int] | None = ...,
    prefilter_hi: float | None = ...,
    debug: bool = ...,
    log_level: int = ...,
    batch_matvec: bool = ...,
) -> _Result | _DebugResult: ...


def ground_locg(
    mat: Callable[[jax.Array], jax.Array] | jax.Array,
    xinit: jax.Array | int,
    args: tuple = (),
    maxiter: int = 1000,
    atol: float = 0.0,
    rtol: float | None = None,
    vspace: tuple[int, DTypeLike] | None = None,
    prefilter: tuple[int, int] | None = None,
    prefilter_hi: float | None = None,
    debug: bool = False,
    log_level: int = logging.WARNING,
    batch_matvec: bool = False,
) -> _Result | _DebugResult:
    r"""Single-vector LOBPCG.

    Args:
        mat: Matrix :math:`A`, either as an Array or a function :math:`x \mapsto Ax`. **On a mesh, a
            callable must preserve its input's sharding in the output** -- this routine is
            sharding-transparent only through that contract, which is why every ``apply_*`` in
            :mod:`rqutils.sqd` passes ``out_sharding=jax.typeof(vec).sharding``.
        xinit: Initial vector, or an integer index selecting a one-hot vector (which requires
            ``vspace`` if ``mat`` is callable). A plain Python ``int`` is accepted. Must have a
            non-vanishing overlap with :math:`v_0`.
        args: Additional arguments to callable ``mat``.
        maxiter: Maximum number of gradient descent iterations.
        atol: **Absolute** bound on the eigen-residual :math:`\|Ax - \theta x\|_2`, holding at
            **every** :math:`n`: ``atol=1e-6`` means :math:`\|r\| < 10^{-6}`. Default ``0.0``, which
            disables this arm. **``None`` is rejected**; pass ``0.0`` to disable the arm.

            The achievable floor is :math:`\mathrm{eps} \cdot \|A\|_2` with **no** :math:`n`
            dependence. A below-floor ``atol`` is unreachable **when ``rtol`` is zero**, and
            :func:`rqutils.sqd.sqd` rejects that combination; with a non-zero ``rtol`` it is harmless.
        rtol: **Relative** tolerance -- a fraction of the operator magnitude:

            .. math:: \|r\| < \mathrm{rtol}\,(\|Ax\| + |\theta|)

            Dimensionless, as in :func:`numpy.allclose`: the bound is
            :math:`\approx 2\,\mathrm{rtol}\|A\|_2`, **independent of the dimension**. If ``None``
            (the default), :math:`4\varepsilon` of the promoted dtype, targeting
            :math:`8\varepsilon\|A\|_2`, 8x the floor. Pass ``0.0`` to disable the arm.

            :math:`|\theta|`, not :math:`+\theta`, so the scale cannot cancel for either sign
            (``NOTES.md``, "ground_locg.body: the convergence test's scale").

            **Convergence is** ``||r|| < max(atol, rtol * scale)`` **-- either arm suffices**, so
            ``atol=x, rtol=0.0`` is absolute-only, ``atol=0.0`` with a non-zero ``rtol`` is
            relative-only, and setting both takes whichever is looser. One ``rtol`` does not scale
            itself across dimensions; set ``atol`` per call for that. :func:`rqutils.sqd.sqd`
            rejects ``rtol >= 0.5``, where the bound reaches :math:`\|A\|`.

            .. warning::

               **``tol`` is gone, and it had two meanings.** Relative (against an
               :math:`n \cdot 10`-scaled bound) until 2026-08-31, absolute after. ``tol=`` raises
               ``TypeError``. From the absolute form, ``tol=x`` becomes ``atol=x``; from the
               relative form there is **no exact equivalent** (``NOTES.md``, "``atol``/``rtol``: the
               pair is right").
        vspace: Specification (dimension, dtype) of the vector space. Required only when ``mat`` is
            a callable and ``xinit`` is an integer.
        prefilter: Optional ``(degree, cycles)`` Chebyshev prefilter applied to ``xinit`` before the
            iteration, damping the unwanted band of the spectrum. Static; ``None`` (the default)
            leaves the traced graph unchanged. Costs ``cycles * (degree + 1)`` matvecs and three live
            vectors, with no growing basis, and is sharding-transparent. ``degree <= 1`` or
            ``cycles == 0`` is a legal no-op.

            **It changes only the path, not the answer -- provided ``prefilter_hi`` is a true upper
            bound.** An under-estimate removes the target from the iterate's span and returns an
            excited eigenpair with ``converged=True``, since the residual test certifies *an*
            eigenpair, not the lowest (``markdown/spinchain/rqutils-prefilter-bug.md``).

            **Start with ``(32, 2)``**, and raise ``degree`` rather than ``cycles``: ``degree`` sets
            how sharply one cycle separates, while cycle 1 does most of the work and cycle 2 refines
            once. All figures are single-device CPU (``NOTES.md``, "ground_locg.ground_locg:
            prefilter tuning"; ``markdown/locg-chebyshev-prefilter.md``).
        prefilter_hi: Upper bound on :math:`\lambda_{\max}`, the filter's upper interval edge;
            ignored unless ``prefilter`` runs. **Required when ``mat`` is a callable**, with no
            fallback, since no matvec-only estimate is rigorous; derived by Gershgorin when ``mat``
            is an array. :mod:`rqutils.sqd` passes ``sum|c_k|``. A loose bound only costs resolution,
            while an *under*-estimate changes the answer. See :func:`_chebyshev_prefilter`.
        debug: If True, additionally return per-iteration diagnostics. Note that the diagnostic
            path uses ``jax.lax.scan`` to collect fixed-size output, and therefore always runs the
            full ``maxiter`` iterations with no early exit; rows past convergence are
            post-convergence noise.
        log_level: Verbosity level.
        batch_matvec: Send each group of *independent* operator applications as one stacked
            ``(k, n)`` array, so ``mat`` is invoked once per group. ``mat`` must broadcast over a
            leading axis of **any** size and return a matching ``(k, n)``: the iteration sends 2,
            ``debug=True``'s diagnostics 3. Default ``False``, since an arbitrary callable need not
            accept a batch; rejected when ``mat`` is an array.

            The eigen*value* is unaffected, but the eigen*vector* need not be bit-identical to the
            unbatched arm, since XLA may contract a ``(k, n)`` operand in a different order; compare
            eigenvalues, not vectors (``NOTES.md``, "ground_locg.body: the batched matvec pair").

    Returns:
        ``(eigval, eigvec, niter, converged)`` -- the smallest eigenvalue, its eigenvector, the
        number of gradient descent iterations performed, and whether the convergence criterion was
        met. Check the fourth value rather than comparing the third against ``maxiter``.

        With ``debug=True`` a fifth element is appended, a dict of stacked per-iteration diagnostics
        keyed ``x``, ``y``, ``r``, ``theta``, ``rho``, ``kappa``, ``sas``, ``rtol_scale`` and
        ``converged``; narrow on ``len(result) == 5``, since a type checker cannot follow the static
        ``debug`` flag into the return arity.

        ``rtol_scale`` holds :math:`\|Ax\| + |\theta|`, the quantity ``rtol`` multiplies
        (:math:`\approx 2|\lambda_{\min}|` once converged, not :math:`2\|A\|_2`). It was ``reltol``
        before 2026-09-01; ``diag["reltol"]`` raises ``KeyError`` (``NOTES.md``,
        "ground_locg.ground_locg: the rtol_scale key").

    Raises:
        ValueError: If ``xinit`` is an integer and ``mat`` is a callable but ``vspace`` is None,
            since the vector space cannot be inferred from a callable.

            Also if an integer ``xinit`` is out of range (negative included), which would otherwise
            yield the zero vector and a converged ``0.0`` (``NOTES.md``, "ground_locg.ground_locg:
            the out-of-range one-hot").

            Also if ``batch_matvec`` is set while ``mat`` is an array, rather than silently ignoring
            the flag; if ``prefilter`` runs on a callable ``mat`` with no ``prefilter_hi``; or if a
            ``prefilter`` entry is negative.
        TypeError: If ``prefilter`` is neither None nor a ``(degree, cycles)`` pair of ints.
    """
    _check_prefilter(prefilter)
    # Host-side: inside jit the index is traced and cannot raise. Negative is equally wrong -- iota is
    # non-negative, so it matches nothing rather than counting from the end.
    if isinstance(xinit, int):
        dim = vspace[0] if vspace is not None else (mat.shape[1] if not callable(mat) else None)
        if dim is not None and not 0 <= xinit < dim:
            raise ValueError(
                f"integer xinit {xinit} is out of range for a vector space of dimension {dim}; it "
                "selects a one-hot vector, so an out-of-range index yields the zero vector and a "
                "converged 0.0 rather than an error"
            )
    if callable(mat):
        return _ground_locg_callable(
            mat,
            xinit,
            args,
            maxiter,
            atol,
            rtol,
            vspace=vspace,
            prefilter=prefilter,
            prefilter_hi=prefilter_hi,
            debug=debug,
            log_level=log_level,
            batch_matvec=batch_matvec,
        )
    if batch_matvec:
        # Rejected, not ignored: this path's matvec is a `jax.lax.dot`, which rejects a rank-2 rhs,
        # so batching needs a different contraction rather than a flag.
        raise ValueError("`batch_matvec` applies only when `mat` is a callable, not an array")
    return _ground_locg_matrix(
        mat,
        xinit,
        maxiter,
        atol,
        rtol,
        prefilter=prefilter,
        prefilter_hi=prefilter_hi,
        debug=debug,
        log_level=log_level,
    )


@jax.jit(static_argnames=["maxiter", "prefilter", "debug", "log_level"])
def _ground_locg_matrix(
    mat: jax.Array,
    xinit: jax.Array,
    maxiter: int,
    atol: jax.Array | float,
    rtol: jax.Array | float,
    prefilter: tuple[int, int] | None = None,
    prefilter_hi: jax.Array | float | None = None,
    debug: bool = False,
    log_level: int = logging.WARNING,
):
    vspace = None
    if jnp.issubdtype(xinit.dtype, jnp.integer):
        vspace = (mat.shape[1], mat.dtype)

    # Gershgorin, never an iterative estimate (see _chebyshev_prefilter), and only when the filter
    # runs (NOTES.md, "Validation belongs to the module that owns the gate").
    if prefilter is not None and prefilter[0] > 1 and prefilter[1] > 0 and prefilter_hi is None:
        prefilter_hi = _gershgorin_bound(mat)

    def matvec(x):
        return jax.lax.dot(
            mat, x, precision=(jax.lax.Precision.HIGHEST,) * 2, out_sharding=jax.typeof(x).sharding
        )

    return _ground_locg_callable(
        matvec,
        xinit,
        (),
        maxiter,
        atol,
        rtol,
        vspace=vspace,
        prefilter=prefilter,
        prefilter_hi=prefilter_hi,
        debug=debug,
        log_level=log_level,
    )


@jax.jit(
    static_argnames=[
        "matvec",
        "maxiter",
        "vspace",
        "prefilter",
        "debug",
        "log_level",
        "batch_matvec",
    ]
)
def _ground_locg_callable(
    matvec: Callable[[jax.Array], jax.Array],
    xinit: jax.Array | int,
    args: tuple,
    maxiter: int,
    atol: jax.Array | float,
    rtol: jax.Array | float,
    vspace: tuple[int, DTypeLike] | None = None,
    prefilter: tuple[int, int] | None = None,
    prefilter_hi: jax.Array | float | None = None,
    debug: bool = False,
    log_level: int = logging.WARNING,
    batch_matvec: bool = False,
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

    if log_level <= logging.DEBUG:
        jax.debug.print("Performing first LOBPCG steps")

    xinit = normalize(xinit)

    # Promote xinit to the operator dtype up front (eval_shape, no matvec): it keeps the carry types
    # agreeing and a complex sas complex (NOTES.md, "ground_locg: promote xinit up front").
    work_dtype = jnp.result_type(
        xinit.dtype, jax.eval_shape(lambda vec: matvec(vec, *args), xinit).dtype
    )
    xinit = xinit.astype(work_dtype)

    # After the dtype promotion (the recurrence runs at operator precision) and before body_iter0
    # (so rho_init is the filtered vector's Rayleigh quotient and maxiter=0 stays meaningful).
    if prefilter is not None:
        degree, cycles = prefilter
        if degree > 1 and cycles > 0:
            if prefilter_hi is None:
                # A raise, never an iterative-estimate fallback; Pauli-sum callers have sum|c_k|
                # (NOTES.md, "No matvec-only upper bound on `λ_max` exists").
                raise ValueError(
                    "prefilter_hi is required when mat is a callable: it must be a true upper bound "
                    "on lambda_max, and nothing computable from matvec alone can guarantee that. "
                    "For a Pauli sum, sum(abs(coeffs)) is a valid bound; for an explicit array, pass "
                    "the array as `mat` and it is derived automatically. A loose bound is safe -- an "
                    "under-estimate makes the filter damp its own target and silently return an "
                    "excited eigenpair."
                )
            xinit = _chebyshev_prefilter(matvec, args, xinit, degree, cycles, prefilter_hi)

    if rtol is None:
        # Only rtol takes None: 4*eps of the operator dtype (never xinit's), 8x the residual floor
        # (NOTES.md, "ground_locg: the `rtol=None` default").
        rtol = 4.0 * float(jnp.finfo(work_dtype).eps)

    ctx = _LocgContext(
        matvec=matvec,
        args=args,
        atol=atol,
        rtol=rtol,
        batch_matvec=batch_matvec,
        debug=debug,
        log_level=log_level,
    )
    seed, diag0 = _body_iter0(ctx, xinit)
    # Seed theta with the Rayleigh quotient of xinit so that maxiter=0 returns a meaningful value
    # rather than the state initializer.
    rho_init = seed.rho

    state, diag1 = _body_iter1(ctx, seed.x, seed.r, seed.ax, rho_init)
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
        state, diagnostics_out = jax.lax.scan(lambda s, _: _body(ctx, s), state, length=maxiter)
        diagnostics_out = jax.tree.map(
            lambda d0, d1, dr: jnp.concatenate([d0, d1, dr], axis=0), diag0, diag1, diagnostics_out
        )
    else:
        state = jax.lax.while_loop(
            lambda s: jnp.logical_and(s.niter < maxiter, ~s.converged),
            lambda s: _body(ctx, s),
            state,
        )

    if debug:
        return state.theta, state.x, state.niter, state.converged, diagnostics_out
    return state.theta, state.x, state.niter, state.converged


def _compute_sas(vectors, mvs):
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


def _diagnostics(ctx, xcurr, ycurr, rcurr, theta, kappa=None, scale=None, converged=None):
    """One ``debug=True`` diagnostics row, at three extra matvecs."""
    # Batched like body()'s pair, and it cannot change the trajectory: `diagnostics` is scan's
    # output, never its carry, so any debug=True divergence comes from body()'s pair.
    if ctx.batch_matvec:
        mvs = tuple(ctx.matvec(jnp.stack((xcurr, ycurr, rcurr)), *ctx.args))
    else:
        mvs = tuple(ctx.matvec(v, *ctx.args) for v in (xcurr, ycurr, rcurr))
    sas = _compute_sas((xcurr, ycurr, rcurr), mvs)
    # rho is <x|Ax>, which compute_sas has just computed as the [0, 0] entry. Recomputing it
    # here cost a fourth matvec that XLA did not eliminate.
    rho = sas[0, 0].real

    if kappa is None:
        kappa = jnp.zeros(3, dtype=xcurr.dtype)
    if scale is None:
        scale = jnp.array(0.0)
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
        # `rtol_scale`, formerly "reltol": the scale rtol multiplies, never a tolerance
        # (NOTES.md, "ground_locg debug diagnostics: `reltol` became `rtol_scale`").
        "rtol_scale": scale,
        "converged": converged,
    }


def _body_iter0(ctx, xcurr):
    """Steepest-descent seed. Returns Ax so no later step recomputes it."""
    xnext = xcurr
    ax = ctx.matvec(xcurr, *ctx.args)
    rho = jnp.sum(xcurr.conjugate() * ax).real
    rnext = ax - rho * xnext
    seed = _Seed(x=xnext, r=rnext, ax=ax, rho=rho)
    diag = _diagnostics(ctx, xnext, jnp.zeros_like(xnext), rnext, rho) if ctx.debug else None
    return seed, diag


def _body_iter1(ctx, xcurr, rcurr, axcurr, rho):
    """First Rayleigh-Ritz step, on {x, p}: no y direction exists yet. Returns the initial carry."""
    # Zero-direction guard as in body(), on {x, p}: `_project_out`, never a bare normalize
    # (NOTES.md, "A rounding-floor residual is not zero, and `== 0.0` is the wrong guard").
    tmp_p, norm_p = _project_out((xcurr,), rcurr)
    r_is_zero = norm_p == 0.0
    tmp_p = normalize(tmp_p, norm_p)
    # Reuse Ax from body_iter0 rather than recomputing it inside compute_sas.
    sas = _compute_sas((xcurr, tmp_p), (axcurr, ctx.matvec(tmp_p, *ctx.args)))
    # Lift p out of contention so theta = rho and xnext == xcurr; not body()'s bound specialized
    # (NOTES.md, "`ground_locg`: `body_iter1`'s exclusion bound is not `body()`'s specialized").
    excluded = 2.0 * jnp.abs(rho) + 1.0
    sas = jnp.where(r_is_zero, sas.at[1, 1].set(excluded.astype(sas.dtype)), sas)
    theta, kappa = eigenpair_2x2(sas)
    tmp_t = tmp_p * kappa[0] - xcurr * kappa[1]
    tmp_u = xcurr * kappa[0] + tmp_p * kappa[1]
    xnext = normalize(tmp_u)
    ynext = normalize(_reorthogonalize(tmp_t, xnext))
    axnext = ctx.matvec(xnext, *ctx.args)
    rnext = axnext - theta * xnext
    # A zeroed residual means {x} already spans the relevant space: seed the flag, not False, or
    # while_loop feeds a zeroed search direction into body()'s Rayleigh-Ritz step.
    state = _State(niter=0, converged=r_is_zero, theta=theta, x=xnext, y=ynext, r=rnext, ax=axnext)
    diag = (
        _diagnostics(
            ctx, xnext, ynext, rnext, theta, jnp.insert(kappa, 1, 0.0), converged=r_is_zero
        )
        if ctx.debug
        else None
    )
    return state, diag


def _body(ctx, state):
    """One LOBPCG iteration on {x, y, p}: the ``while_loop`` / ``scan`` body."""
    xcurr, ycurr, rcurr, axcurr = state.x, state.y, state.r, state.ax
    if ctx.log_level <= logging.DEBUG:
        jax.debug.print("LOCG iteration {}", state.niter)

    # Project out both X and P, then renormalize: _project_out only guarantees |tmp_p| >= 0.99,
    # and a short tmp_p scales sas[2, 2] into a spurious minimum under a large positive shift.
    tmp_p, norm_p = _project_out((xcurr, ycurr), rcurr)
    p_is_zero = norm_p == 0.0
    tmp_p = normalize(tmp_p, norm_p)
    # xcurr's image is carried, so three matvecs, not four; the independent pair batches as one
    # (2, N) call, judged by theta (NOTES.md, "ground_locg.body: the batched matvec pair").
    if ctx.batch_matvec:
        aycurr, ap = ctx.matvec(jnp.stack((ycurr, tmp_p)), *ctx.args)
    else:
        aycurr, ap = ctx.matvec(ycurr, *ctx.args), ctx.matvec(tmp_p, *ctx.args)
    sas = _compute_sas((xcurr, ycurr, tmp_p), (axcurr, aycurr, ap))
    # A zeroed tmp_p leaves a zero diagonal Rayleigh-Ritz would pick for a positive-definite A,
    # then divide by; lift it out, and report p_is_zero as convergence below.
    diag_xy = jnp.diagonal(sas).real[:2]
    excluded = jnp.max(diag_xy) + jnp.sum(jnp.abs(diag_xy)) + 1.0
    sas = jnp.where(p_is_zero, sas.at[2, 2].set(excluded.astype(sas.dtype)), sas)
    theta, kappa = eigenpair_3x3(sas)
    # New vectors
    tmp_s = ycurr * kappa[1] + tmp_p * kappa[2]
    tmp_u = xcurr * kappa[0] + tmp_s
    # One joint reduction: XLA's combiner leaves two separate norms as two chained all-reduces.
    su = jnp.stack((tmp_s, tmp_u))
    norm_s, norm_u = jnp.sqrt(jnp.sum(jnp.real(su * jnp.conj(su)), axis=-1))
    tmp_t = tmp_s * (kappa[0] / jnp.where(norm_s == 0.0, 1.0, norm_s)) - xcurr * norm_s
    xnext = normalize(tmp_u, norm_u)
    ynext = normalize(_reorthogonalize(tmp_t, xnext))
    axnext = ctx.matvec(xnext, *ctx.args)
    rnext = axnext - xnext * theta
    norm_rnext = jnp.linalg.norm(rnext)
    # ||r|| < max(atol, rtol * (||Ax|| + |theta|)), either arm sufficing: no `n` factor, and
    # `abs` so it cannot cancel (NOTES.md, "ground_locg.body: the convergence test's scale").
    scale = jnp.linalg.norm(axnext) + jnp.abs(theta)
    # A zeroed search direction means {x, y} already spans the residual: we are at a stationary
    # point of the Rayleigh quotient and no further iteration can lower theta.
    converged = jnp.logical_or(norm_rnext < jnp.maximum(ctx.atol, ctx.rtol * scale), p_is_zero)
    if ctx.log_level <= logging.DEBUG:
        jax.debug.print("Residual {}, scale {}, converged: {}", norm_rnext, scale, converged)

    state = _State(
        niter=state.niter + 1,
        converged=converged,
        theta=theta,
        x=xnext,
        y=ynext,
        r=rnext,
        ax=axnext,
    )
    if ctx.debug:
        return state, _diagnostics(ctx, xnext, ynext, rnext, theta, kappa, scale, converged)
    return state


def _reorthogonalize(vector, against, passes=2):
    """Re-orthogonalize ``vector`` against a single unit vector, repeatedly.

    :math:`t = \\kappa_0 s / |s| - |s| x` cancels catastrophically as :math:`|s| \\to 0`, letting
    :math:`y` drift into :math:`x`; the *standard* Rayleigh-Ritz step on the non-orthonormal basis then
    returns a :math:`\\theta` **below** the true minimum. One pass is not enough, for the same reason
    :func:`_project_out` runs twice: the second removes what the first's rounding reintroduced.

    Pinned by ``test/test_ground_locg.py::TestBasisOrthogonality`` off the ``debug=True``
    diagnostics, since ``theta`` stays correct while the drift builds. A/B it in a **fresh subprocess
    before any tracing**, or the jitted callers reuse one kernel (``NOTES.md``,
    "ground_locg._reorthogonalize: the measured drift").
    """
    for _ in range(passes):
        vector = vector - against * jnp.sum(against.conjugate() * vector)
    return vector


def _subtract_projections(basis, vector):
    """Subtract the projection of ``vector`` onto each basis element.

    All inner products are taken before any subtraction, so a multi-element basis is projected out
    in one pass. Deliberately *not* batched into a matmul, whose summation order measured
    consistently worse near degeneracy; re-run that comparison before "optimizing" this (``NOTES.md``,
    "ground_locg._subtract_projections: why not a matmul").
    """
    ips = [jnp.sum(vb.conjugate() * vector) for vb in basis]
    for vb, ip in zip(basis, ips):
        vector = vector - vb * ip
    return vector


def _project_out(basis, vector):
    # Algorithm 5 of Duersch et al. (arXiv:1704.07458) at block size 1, two passes by "twice is
    # enough" (NOTES.md, "ground_locg._project_out: sources for the two-pass form").
    for _ in range(2):
        vector = normalize(_subtract_projections(basis, vector))

    # End on a subtraction of the basis, not a normalization: near convergence (R = 0) cancellation
    # re-introduces (X, P) components; suspicious vectors are zeroed to keep [basis, U] orthogonal.
    for _ in range(2):
        vector = _subtract_projections(basis, vector)

    # Postcondition: exactly zero or norm >= 0.99, NOT normalized; the norm is returned so callers
    # renormalize and zero-test without a second O(N) reduction.
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
    # Null vector of singular T + rad I: rows give [-conj(offd), delta + rad], [rad - delta, -offd],
    # parallel but each cancelling when its pivot is small, so select on the sign of delta.
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
    Not a magnitude threshold: at a degenerate eigenvalue the cross products decay only to
    :math:`O(\\epsilon \\|M\\|^2)`, close enough to a genuinely small rank-2 cross product that any
    fixed cutoff misclassifies one case or the other.
    """
    # Rank 2: the null vector is conj(col_i x col_j), but any one pair can be rank deficient and
    # vanish, so all three pairings are offered.
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
    # Radicands clamped against rounding; disc is Cardano's p^3 - q^2 (q = -13.5*c0) kept in c1, c0,
    # never p*p*p - q*q (NOTES.md, "ground_locg.eigenpair_3x3: the discriminant form").
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
