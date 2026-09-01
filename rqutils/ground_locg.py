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

- **Convergence threshold.** Two independent tolerances, satisfied by **either**:

  .. math:: \|r\| < \max\bigl(\mathrm{atol},\ \mathrm{rtol}\,(\|Ax\| + |\theta|)\bigr)

  ``atol`` is an absolute bound on :math:`\|Ax - \theta x\|_2`, so a caller whose consumer checks a
  fixed residual can name it and have it hold at every :math:`n`. ``rtol`` is a *fraction of the
  operator magnitude* -- the conventional meaning, as in :func:`numpy.allclose` -- and since
  :math:`\|Ax\| \approx |\theta|` at convergence its bound is :math:`\approx 2\,\mathrm{rtol}\|A\|_2`,
  independent of the dimension. The ``max`` is what makes the pair strictly more expressive than either
  alone: a purely relative test cannot name a residual, and a purely absolute one cannot track an
  operator whose scale the caller does not know.

  The achievable floor is

  .. math:: \mathrm{floor}(\|r\|) \approx \varepsilon(\mathrm{dtype}) \cdot \|A\|_2

  with **no** dependence on :math:`n` -- measured over :math:`n = 70` to :math:`32768` and
  :math:`\|A\|_2` over six decades, on both the dense and matrix-free paths and both real and complex
  coefficients (27 samples: the constant spans 0.49-1.26 with median 0.84, while the :math:`n`-scaled
  form :math:`\varepsilon\|A\|n` spans 306x). ``rtol=None`` targets :math:`8\varepsilon\|A\|_2`, 8x that
  floor.

  **Two earlier forms, both superseded, and the reasons matter.** A relative test
  :math:`\|r\| < \mathrm{tol}\,(\|Ax\| + |\theta|)\,n \cdot 10` carried an :math:`n \cdot 10` factor that
  was 700x-94600x looser than the floor -- slack, not a rounding budget -- so one ``tol`` meant a
  different absolute residual at every :math:`n` and no single value could be both fast and admissible
  for a caller with a fixed requirement. Replacing it with a *purely* absolute ``tol`` then removed the
  ability of one value to track the operator at all, and made the default 1.18-1.49x slower. Neither
  factor of that :math:`n \cdot 10` survives here: folding a dimension count into a "relative"
  tolerance made it unpredictable and, at large :math:`n`, dangerous -- ``rtol=1e-8`` at
  :math:`n = 2^{20}` gave a bound of 4.2 against :math:`\|A\| = 20`, so the first iterate reported
  convergence and returned a wrong answer. :func:`rqutils.sqd.sqd` now rejects ``rtol >= 0.5``.

  Two traps the relative form recorded, still worth keeping visible because they constrain any future
  change to the scale: the natural-looking ``norm(Ax) - theta`` is a difference of two nearly equal
  large positive numbers for a positive-definite operator and was measured going *negative*, making the
  test unsatisfiable; and :math:`|\theta|` rather than :math:`+\theta` is needed because for the
  negative-definite operators typical of a ground-state search :math:`+\theta` cancels in turn. The
  second still applies -- the scale here is that same sum.

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

Chebyshev prefilter
===================

:func:`ground_locg` takes an optional ``prefilter=(degree, cycles)`` that applies a Chebyshev
polynomial filter to the initial vector before the iteration starts, damping the band
:math:`[\theta, \lambda_{\max}]` by :math:`1/T_{\mathrm{degree}}` so the ground direction comes out
amplified relative to everything else. The technique is Chebyshev-filtered subspace iteration
(ChFSI), standard in large-scale electronic-structure codes[3][4][5]; what is used here is the
single-vector, prefilter-then-LOBPCG specialization rather than a construction from those papers,
which filter a whole subspace inside a self-consistent loop. In particular the two-level
complementary-subspace method of [5] is **not** implemented here and would not fit: it filters a
subspace and solves its complement, against this module's three-vector memory budget. A two-level
*preconditioner* was separately measured and rejected (0.68-0.98x, ``docs/deflation-preconditioner.md``)
-- it improved conditioning without opening the gap, which is what the iteration count tracks.

Two properties are load-bearing and neither is inherited from the references. The filter's lower edge
is the *current Rayleigh quotient*, re-read each cycle, not an estimate of :math:`\lambda_1` -- an
accurate :math:`\lambda_1` measured faster on a comfortable gap and silently wrong on a tight one.
And the upper edge ``prefilter_hi`` must be a true bound on :math:`\lambda_{\max}`, which no
matvec-based iteration can supply; a power-iteration estimate returned an *excited* eigenpair with
``converged=True``. :func:`_chebyshev_prefilter` records both measurements, and
``docs/locg-chebyshev-prefilter.md`` has the tuning tables.

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

[3]: A. S. Banerjee, L. Lin, W. Hu, C. Yang, J. E. Pask, *Chebyshev polynomial filtered subspace
iteration in the discontinuous Galerkin method for large-scale electronic structure calculations*,
J. Chem. Phys. **145**, 154101 (2016).

[4]: Y. Zhou, Y. Saad, M. L. Tiago, J. R. Chelikowsky, *Self-consistent-field calculations using
Chebyshev-filtered subspace iteration*, J. Comput. Phys. **219**, 172 (2006).

[5]: A. S. Banerjee, L. Lin, P. Suryanarayana, C. Yang, J. E. Pask, *Two-level Chebyshev filter based
complementary subspace method*, J. Chem. Theory Comput. **14**, 2930 (2018). The provenance
``docs/locg-chebyshev-prefilter.md`` cites; used in production in DFT-FE. Its two-level
complementary-subspace split is **not** what this module does -- see the note below.

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

# ground_locg's return is (eigval, eigvec, niter, converged), with the per-iteration diagnostics
# appended when debug=True. Deliberately plain tuple aliases rather than a dataclass: the arity, not
# the anonymity of the positions, is what a type checker needs to see here, and `ground_locg` is
# published API that every caller destructures positionally (sqd.py, two scaling POCs, a benchmark
# under examples/). A caller reading the fifth element must narrow first, `assert len(result) == 5`.
#
# `@overload` on `debug: Literal[True]/[False]` would give a checker the arity without narrowing, the
# way `sqd` does it for `return_eigvec`. Untried here: it is what the `invalid-assignment` ignore in
# pyproject.toml currently absorbs (19 diagnostics, mostly this destructuring), so it is a real
# candidate rather than a rejected one.
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


def residual_floor(opnorm_bound: float, dtype: DTypeLike) -> float:
    r"""Return the smallest eigen-residual a solve on this operator can reach.

    The achievable floor of :math:`\|Ax - \theta x\|_2` is :math:`\varepsilon \cdot \|A\|_2`,
    **independent of the dimension** -- measured over :math:`n = 70` to :math:`32768` and six decades
    of :math:`\|A\|_2`, on the dense and matrix-free paths and both coefficient dtypes (27 samples:
    the constant spans 0.49-1.26, median 0.84; the :math:`n`-scaled form spans 306x and is therefore
    not the mechanism). The returned value multiplies that by **4**, a 3.2x margin over the worst
    constant observed.

    ``opnorm_bound`` may be any upper bound on :math:`\|A\|_2`; :func:`rqutils.sqd.sqd` passes
    :math:`\sum_k |c_k|`, measured a 1.56-1.90x over-estimate on 1D XXZ fixtures. Over-estimating is
    the safe direction -- it raises the reported floor, so a ``tol`` this function admits is
    comfortably reachable.

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

    **Why only one of them takes ``None``**, since the asymmetry invites the question. ``rtol``'s default
    is the *promoted operator dtype's* epsilon, which cannot be written as a literal in the signature: a
    hardcoded ``8.88e-16`` is right for float64 and unsatisfiable by 1.3e8x on a float32 problem, which
    ``tests/test_ground_locg.py`` exercises. ``None`` is the only way to defer that to runtime. ``atol``
    has no such excuse -- there is no dtype-derived absolute residual a caller would want -- so it takes
    a plain 0.0 and ``None`` is an error rather than a synonym for it.

    **The floor check fires only when ``atol`` is the sole arm.** The achievable residual floor is
    ``eps * ||H||_2``, so a below-floor ``atol`` cannot be met -- but with ``rtol > 0`` the relative arm
    still can, and rejecting that configuration would fail a *working* call. This repo has already paid
    for a guard that fired on correct input (an overflow count that included discarded padding, reported
    763,677 beside a bit-exact result); the condition is `rtol == 0`, not `atol < floor` alone.

    Both arms zero is rejected outright: no residual satisfies ``|| r || < 0``, so the solve would run to
    ``maxiter`` and raise. That is diagnosable here and only a symptom there.

    Sited in this module, beside the test it guards, for the reason :func:`_check_prefilter` gives. It
    cannot be enforced *at* that test: ``converged`` is a traced boolean inside a ``jax.lax.while_loop``,
    so nothing there can raise. :func:`rqutils.sqd.sqd` is the outermost point where
    :math:`\sum_k |c_k|` is concrete, which is why it is the caller.

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
    # A bound that reaches ||H|| is not a tolerance, it is an accept-anything: every normalized vector
    # satisfies ||Hv - Ev|| <= ||H|| (since |E| <= ||H||), so the first iterate reports convergence and
    # the returned eigenpair is arbitrary -- with converged=True, the failure mode this module keeps
    # guarding against. Measured on BOTH arms: atol=100 against ||H||=17 converged in *one* iteration,
    # and on the superseded `* n * 10` rtol scale, rtol=1e-8 at n=2^20 gave a bound of 4.2 vs ||H||=20.
    #
    # The condition is on the bound, not on which parameter produced it, so both arms are checked. For
    # `rtol` the scale is `||Hv|| + |E| <= 2||H||`, so the cutoff is rtol >= 0.5. For `atol` the bound is
    # the value itself, compared against `opnorm_bound` -- which is sum|c_k|, an over-estimate of ||H||_2
    # (measured 1.56-1.90x on 1D XXZ), so this errs toward accepting. Both are deliberately loose: they
    # catch the accept-anything case without second-guessing a caller who wants a sloppy solve.
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

    Lives here rather than in :mod:`rqutils.sqd` because this module owns the gate the check exists
    to compensate for: :func:`_chebyshev_prefilter` runs only ``if degree > 1 and cycles > 0``, an
    equality-style
    branch with an implicit ``else``, so an out-of-range value is **absorbed into a silent no-op**
    rather than reported: measured, ``(2, -1)``, ``(-4, 2)`` and ``(True, 2)`` all returned the exact
    unfiltered energy at zero speedup, which reads as "the prefilter does not help on my problem".
    That misdiagnosis is the one thing this option cannot afford, because its docstring tells callers
    to A/B it on their own subspaces. Malformed *types* were no better: ``(2,)``, ``"32,2"`` and ``32``
    reached ``ground_locg``'s tuple unpack and surfaced its ``ValueError``/``TypeError`` from inside a
    public entry point.

    ``bool`` is rejected for the reason :func:`rqutils.sqd._check_cache_level` gives -- it is an
    ``int`` subclass, so ``(True, 2)`` would otherwise pass as ``(1, 2)``, i.e. as a documented no-op.

    The **intentional** no-ops stay legal: ``degree <= 1`` or ``cycles == 0`` is how a caller disables
    the filter without restructuring a sweep, and ``TestChebyshevPrefilter`` pins that contract. Only
    negative values, non-ints and wrong shapes are errors -- a distinction only expressible here,
    since that single ``degree > 1 and cycles > 0`` test cannot make it.

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
    row-magnitude maximum bounds the spectral radius and therefore :math:`\lambda_{\max}`. **Rigorous
    for every Hermitian input**, needs no iteration, and costs one :math:`O(N^2)` reduction -- measured
    1.1-2.7x a single matvec at N=512-4096, against the 11 matvecs the power iteration this replaced
    spent on an estimate that was not a bound at all.

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

    ``hi`` MUST BE A TRUE UPPER BOUND ON :math:`\lambda_{\max}`, and is now a required argument
    because nothing computable from ``matvec`` can guarantee that. It previously came from 10 steps of
    power iteration, which is wrong twice over: power iteration converges to the eigenvalue of largest
    *magnitude*, so on a negative-leaning spectrum it returns something near :math:`\lambda_{\min}`
    and the interval **inverts**; and even with the sign repaired a fixed step count merely
    under-estimates. The consequence was a silent wrong answer -- the filter damps its own target and
    the solver returns an *excited* eigenpair with ``converged=True`` (measured: the n=2 Heisenberg
    chain returned +0.25 for a true -0.75; the bound was invalid in 16 of 25 XXZ configurations, with
    wrong answers in 2). ``docs/rqutils-prefilter-bug.md`` has the report and the reproduction.

    **No cheap matvec-only upper bound exists** -- a theorem, not a tuning problem (Kuczynski &
    Wozniakowski, SIAM J. Matrix Anal. Appl. 13(4):1094-1122, 1992). So rigour has to come from the
    operator's structure: Gershgorin for an array (:func:`_gershgorin_bound`),
    :math:`\sum_k |c_k|` for a Pauli sum, which is what :mod:`rqutils.sqd` passes. ``NOTES.md`` has
    the measured candidate table, the adversarial construction, and why the Ritz-plus-residual forms
    production libraries use are estimates rather than bounds.

    Prefer a loose bound to a tight estimate: over-estimating costs resolution smoothly, while
    under-estimating flips which eigenvector is amplified most and returns the wrong answer.

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


# Overloads so a caller destructuring the 4-tuple does not have to narrow on `len(result) == 5` first.
# Annotation only: the implementation signature below is unchanged and is the one sphinx documents.
# `debug` stays positional-or-keyword here (unlike `sqd`'s keyword-only block), so the overloads repeat
# the full positional order -- `examples/scaling/poc6_mixed_precision.py` and the benchmark pass
# `maxiter` positionally. A `bool` that is not a literal still matches the `bool` overload and gets the
# union, which is what a runtime-computed flag needs.
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
        atol: **Absolute** bound on the eigen-residual :math:`\|Ax - \theta x\|_2`. Default ``0.0``,
            which disables this arm and leaves ``rtol`` to decide. Set it when a downstream consumer has
            a fixed residual requirement: ``atol=1e-6`` means :math:`\|r\| < 10^{-6}` at **every**
            :math:`n`, which no relative tolerance can express.

            **``None`` is rejected** -- pass ``0.0`` to disable the arm. A *derived* absolute bound is
            the unintuitive construct this pair replaced: an absolute residual is either a number the
            caller wants or it is not wanted at all.

            The achievable floor is :math:`\mathrm{eps} \cdot \|A\|_2` with **no** :math:`n` dependence
            (measured over :math:`n = 70..32768` and six decades of :math:`\|A\|`, dense and
            matrix-free, both dtypes: the constant spans 0.49-1.26 while the :math:`n`-scaled form
            spans 306x). An ``atol`` below that floor is unreachable **when ``rtol`` is zero**;
            :func:`rqutils.sqd.sqd` rejects that combination. With a non-zero ``rtol`` it is harmless,
            because the relative arm can still fire.
        rtol: **Relative** tolerance -- a fraction of the operator magnitude:

            .. math:: \|r\| < \mathrm{rtol}\,(\|Ax\| + |\theta|)

            This is the conventional meaning, as in :func:`numpy.allclose` and :mod:`scipy`: ``rtol`` is
            dimensionless, and :math:`(\|Ax\| + |\theta|)` supplies the units, since :math:`\|r\|` scales
            with :math:`\|A\|`. Since :math:`\|Ax\| \approx |\theta|` at convergence the bound is
            :math:`\approx 2\,\mathrm{rtol}\|A\|_2` -- **independent of the dimension**, so one value
            means the same thing at every :math:`n`.

            If ``None`` (the default), :math:`4\varepsilon` is used, targeting :math:`8\varepsilon\|A\|_2`
            -- 8x the measured floor of :math:`\varepsilon\|A\|_2`, the same 3.2x margin over the worst
            observed floor constant that :func:`residual_floor` applies. Pass ``0.0`` to disable the arm.

            :math:`|\theta|` rather than :math:`+\theta` because the sum must not cancel for either sign,
            and a ground-state search is typically negative-definite; the natural-looking
            ``norm(Ax) - theta`` was measured going *negative* for a positive-definite operator, which
            makes the test unsatisfiable.

            **An earlier form multiplied this by** :math:`n \cdot 10`, **and that is gone.** Neither
            factor was a rounding budget -- the floor has no :math:`n` term (measured over
            :math:`n = 70..32768`: the :math:`\varepsilon\|A\|` constant spans 2.6x where the
            :math:`n`-scaled form spans 306x) -- and folding a dimension count and a bare 10 into a
            "relative" tolerance made it unpredictable and, at large :math:`n`, dangerous: ``rtol=1e-8``
            at :math:`n = 2^{20}` produced a bound of 4.2 against :math:`\|A\| = 20`, so the solve
            converged on the first iterate and returned a wrong answer with ``converged=True``. The dial
            also saturated, ``rtol=1e-6`` and ``1e-4`` giving bit-identical results.

            The cost is real and worth stating: **one ``rtol`` no longer scales itself across
            dimensions.** A caller that needs a different bound per subspace size sets ``atol`` per call
            instead. :func:`rqutils.sqd.sqd` rejects ``rtol >= 0.5``, where the bound would reach
            :math:`\|A\|` and every vector would "converge".

            **Convergence is** ``||r|| < max(atol, rtol * scale)`` **-- either arm suffices**, so
            ``atol=x, rtol=0.0`` is absolute-only, ``atol=0.0`` with a non-zero ``rtol`` is
            relative-only, and setting both takes whichever is looser.

            .. warning::

               **``tol`` is gone, and it had two meanings.** Relative (against an
               :math:`n`-scaled bound) until 2026-08-31, absolute after. There is no alias:
               ``tol=`` raises ``TypeError`` rather than silently resolving to one of the pair.
               From the absolute form, ``tol=x`` becomes ``atol=x``. From the relative form there is
               **no exact equivalent**, because the :math:`n \cdot 10` factor is not reproduced;
               ``rtol`` gives per-operator scaling only.
        vspace: Specification (dimension, dtype) of the vector space. Required only when ``mat`` is
            a callable and ``xinit`` is an integer.
        prefilter_hi: Upper bound on :math:`\lambda_{\max}`, used as the filter's upper interval
            edge. Ignored unless ``prefilter`` is set. **Required when ``mat`` is a callable** -- there
            is no fallback, because no matvec-only estimate can be rigorous and the estimate this
            replaced silently returned excited eigenpairs. When ``mat`` is an array the Gershgorin
            bound ``max_i sum_j |A_ij|`` is derived automatically, so no caller of the array path is
            affected. :mod:`rqutils.sqd` passes ``sum|c_k|``, valid because every Pauli string is
            unitary. Prefer a loose bound: over-estimating costs resolution smoothly, while
            *under*-estimating changes the answer. See :func:`_chebyshev_prefilter`.
        prefilter: Optional ``(degree, cycles)`` Chebyshev prefilter applied to ``xinit`` before the
            iteration starts, damping the unwanted band of the spectrum so the LOBPCG loop begins
            closer to :math:`v_0`. ``None`` (the default) leaves the traced graph unchanged, so no
            existing caller is affected; it is a static argument, so the branch resolves at trace
            time. Costs ``cycles * (degree + 1)`` extra matrix-vector products -- the ~11 for a
            :math:`\lambda_{\max}` estimate are gone, since ``prefilter_hi`` needs no iteration --
            and three live vectors: **no growing basis**, which is
            what makes it compatible with this module's single-vector memory budget. Like ``mat`` and
            ``mat`` it is sharding-transparent, since it only calls ``mat`` and scales
            elementwise. **It cannot change the answer, only the path -- provided ``prefilter_hi``
            is a true upper bound.** That caveat is load-bearing and was originally missing: the
            residual test every convergence check reads certifies that *an* eigenpair was found, not
            that it is the lowest, so a filter that removed the target from the iterate's span
            returned an excited eigenpair with ``converged=True``
            (``docs/rqutils-prefilter-bug.md``). With a valid bound the returned eigenpair is the same
            one to the tolerance the solver was going to reach anyway (measured: eigenvector overlap 1.0000000 against the
            unfiltered result, energies agreeing with ``eigsh(tol=0)`` to 2.8e-14).

            **Start with ``(32, 2)``.** Across 27 connected-subspace configurations (3 sizes x 3
            seeds x 3 anisotropies, every arm converged and correct to <1e-9) it measured a median
            **1.88x** wall-clock reduction, range 1.25-3.95x, at *fewer* matvecs than the
            alternatives below. The two knobs are not interchangeable:

            - ``degree`` sets how sharply **one** cycle separates. Amplification outside the damped
              band grows like :math:`\cosh(\mathrm{degree} \cdot \mathrm{arccosh}|x|)`, i.e.
              roughly exponentially, while costing only ``degree`` matrix-vector products. This is
              the high-leverage knob.
            - ``cycles`` sets how many times the interval re-tightens around the descending Rayleigh
              quotient. Cycle 1 does most of the work (measured growth factor 1e8-1e12), cycle 2
              refines once, and past that :math:`\theta` is already near :math:`\lambda_0` so
              further cycles pay full cost for little separation.

            So **raise ``degree``, keep ``cycles`` at 2**. Measured medians: ``(16, 4)`` 1.41x,
            ``(32, 2)`` **1.88x**, ``(48, 2)`` 1.79x; on a narrower sweep ``(64, 2)`` reached 2.29x
            and ``(128, 2)`` fell to 1.68x with a 1.01x floor. **Treat the upper half of that range with
            suspicion**: an independent sweep from the ``spinchain`` side measured ``degree=64`` as
            the *weakest* arm on every path it tried (median 1.35x through ``sqd``, 0.74-0.80x dense),
            so ``(32, 2)`` is the only value recommended without qualification. Longer solves favour the higher end:
            the 249- and 573-iteration cases measured 3.6-5.1x at ``degree`` 48-96.

            All figures are single-device CPU; the ordering may differ on a GPU, where the
            matvec-to-bookkeeping cost ratio differs -- ``examples/scaling/poc9_prefilter_gpu.py``
            sweeps this grid to settle it. See ``docs/locg-chebyshev-prefilter.md`` for the tables,
            why the filter's lower edge must be the running Rayleigh quotient rather than an
            accurate :math:`\lambda_1`, and why filtering alone does not converge.
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
        diagnostics keyed ``x``, ``y``, ``r``, ``theta``, ``rho``, ``kappa``, ``sas``,
        ``rtol_scale`` and ``converged``; narrow on ``len(result) == 5`` before reading it, since
        ``debug`` is a static flag that a type checker cannot follow into the return arity.

        **The ``rtol_scale`` key was named ``reltol`` before 2026-09-01.** It holds
        :math:`\|Ax\| + |\theta|`, the quantity ``rtol`` multiplies -- never a tolerance, and never
        the residual floor the old name suggested. Once converged :math:`x` is the ground eigenvector,
        so this is :math:`\approx 2|\lambda_{\min}|` (verified 3.9990 against
        :math:`|\lambda_{\min}| = 2`), **not** :math:`2\|A\|_2` -- those coincide only when the
        ground state is also the largest-magnitude one. A caller reading ``diag["reltol"]`` now gets a
        ``KeyError``, which is the intended failure: the old name was off by roughly
        :math:`1/\varepsilon` against what it implied.

    Raises:
        ValueError: If ``xinit`` is an integer and ``mat`` is a callable but ``vspace`` is None. The
            vector space cannot be inferred from a callable, and without this the one-hot
            construction would fail with an opaque "NoneType is not subscriptable".

            Also if an integer ``xinit`` is out of range (negative included). The one-hot is built as
            ``iota == xinit``, so such an index matches nothing and yields the **zero vector**, from
            which the solver returns ``0.0`` with ``converged=True`` -- measured ``xinit=16`` on a
            dimension-16 operator whose true minimum was -1.5.
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
        )
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

    # The array path can always supply its own rigorous bound, so a caller never has to. Gershgorin
    # rather than anything iterative: see _chebyshev_prefilter on why no matvec-based estimate is one.
    # Gated on the filter actually running, matching `run_sqd`: `degree <= 1` or `cycles == 0` is a
    # documented no-op, and computing an O(N^2) reduction for those values put it in the traced graph
    # anyway -- measured +6.1% on a 2048-dim solve with `prefilter=(16, 0)`.
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

    def diagnostics(xcurr, ycurr, rcurr, theta, kappa=None, scale=None, converged=None):
        sas = compute_sas(
            (xcurr, ycurr, rcurr), tuple(matvec(v, *args) for v in (xcurr, ycurr, rcurr))
        )
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
            # Renamed from "reltol" 2026-09-01: the value is the scale `||Ax|| + |theta|` that `rtol`
            # multiplies, never a tolerance. A `debug=True` caller reading `diag["reltol"]` now gets a
            # KeyError rather than a number ~2|lambda_min| where the old name promised a floor near
            # eps*||A|| -- loud, and the old name was off by ~1e16 against what it claimed.
            #
            # `rtol_scale` here, not the local's bare `scale`: the local sits three lines from
            # `rtol * scale` so context disambiguates it, while a key is read on its own out of a dict
            # of nine, where "scale" could be any of several quantities in this solver.
            "rtol_scale": scale,
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
        # `_project_out` rather than a bare `normalize(rcurr)`, which is what `body()` has always
        # used and what this step was missing. A bare normalize divides by the residual norm however
        # small it is: an `xinit` that *is* an eigenvector in floating point leaves a residual at the
        # rounding floor (measured 3.1e-16 on the 2x2 `[[2.9, 1], [1, 2.9]]`), and dividing by that
        # amplifies pure noise until `tmp_p` comes back **parallel to `xcurr`**. `sas` then degenerates
        # -- measured `[[1.9, -1.9], [-1.9, 4.8]]`, whose lowest eigenvalue is 0.96 for a true 1.9 --
        # and the caller saw a `RuntimeError` naming `maxiter` on a problem solved in one iteration
        # (`docs/rqutils-prefilter-dim2-request.md`).
        #
        # Masking `sas[1, 1]` alone does NOT fix it: the mask fired correctly and the surviving
        # off-diagonal still coupled `x` to the noise. Nor does a scale-relative residual threshold --
        # measured, `|r| = 8.07e-16` against a floor of `7.99e-16` on a neighbouring instance, i.e. a
        # 1% margin deciding correctness, and a looser floor pins `theta = rho` when the iterate is
        # *not* an eigenvector. `_project_out` needs no threshold: it renormalizes, subtracts again,
        # and returns exactly zero when the norm collapses below 0.99, which is precisely "this
        # direction was rounding noise". It also returns the norm, so no separate reduction is needed.
        tmp_p, norm_p = _project_out((xcurr,), rcurr)
        r_is_zero = norm_p == 0.0
        tmp_p = normalize(tmp_p, norm_p)
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
        tmp_p, norm_p = _project_out((xcurr, ycurr), rcurr)
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
        norm_rnext = jnp.linalg.norm(rnext)
        # Two independent tolerances, satisfied by EITHER -- hence `max`, not `min`:
        #
        #     ||r|| < max(atol, rtol * (||Ax|| + |theta|))
        #
        # `atol` bounds the residual absolutely, so a caller with a fixed requirement (a downstream
        # guard at 1e-6) can name it and have it hold at every dimension. `rtol` is a *fraction of the
        # operator magnitude* -- the conventional meaning, as in `np.allclose` and `scipy` -- so it needs
        # the `(||Ax|| + |theta|)` factor and nothing else: ||r|| has units of ||A||, and dividing by
        # something with those units is what makes `rtol` dimensionless.
        #
        # Do NOT reintroduce the pre-2026-08-31 `* n * 10` factor on the scale: it made `rtol=1e-8` at
        # n=2^20 a bound of 4.2 against ||A|| = 20, so the first iterate "converged" on a wrong answer.
        # The module docstring has the measurements; `sqd` rejects an `rtol` whose bound reaches ||A||.
        #
        # `abs(theta)` rather than `+theta`: the sum must not cancel for either sign of theta, and a
        # ground-state search is typically negative-definite. The natural-looking `norm(Ax) - theta` was
        # measured going *negative* for a positive-definite operator, making the test unsatisfiable.
        scale = jnp.linalg.norm(axnext) + jnp.abs(theta)
        # A zeroed search direction means {x, y} already spans the residual: we are at a stationary
        # point of the Rayleigh quotient and no further iteration can lower theta.
        converged = jnp.logical_or(norm_rnext < jnp.maximum(atol, rtol * scale), p_is_zero)
        if log_level <= logging.DEBUG:
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
        if debug:
            return state, diagnostics(xnext, ynext, rnext, theta, kappa, scale, converged)
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
            if prefilter_hi is None:
                # Deliberately a raise, not a fallback to an iterative estimate: the estimate is what
                # returned an excited eigenpair with converged=True, and no matvec-only method can be
                # made rigorous (see _chebyshev_prefilter). Callers holding a Pauli sum have the bound
                # for free as sum|c_k|, which is what rqutils.sqd passes.
                raise ValueError(
                    "prefilter_hi is required when mat is a callable: it must be a true upper bound "
                    "on lambda_max, and nothing computable from matvec alone can guarantee that. "
                    "For a Pauli sum, sum(abs(coeffs)) is a valid bound; for an explicit array, pass "
                    "the array as `mat` and it is derived automatically. A loose bound is safe -- an "
                    "under-estimate makes the filter damp its own target and silently return an "
                    "excited eigenpair."
                )
            xinit = _chebyshev_prefilter(matvec, args, xinit, degree, cycles, prefilter_hi)

    seed, diag0 = body_iter0(xinit)
    # Seed theta with the Rayleigh quotient of xinit so that maxiter=0 returns a meaningful value
    # rather than the state initializer.
    rho_init = seed.rho

    if rtol is None:
        # Only `rtol` takes None; `atol` defaults to 0.0 instead, because a "derived absolute bound" is
        # exactly the unintuitive thing this pair replaced -- an absolute residual is either a number the
        # caller wants or it is not wanted at all.
        #
        # 4*eps, not eps: the scale is now `||Ax|| + |theta|` ~ 2*||A||, so `rtol = eps` would target
        # 2*eps*||A||, only 2x the measured floor of eps*||A|| -- and that floor's constant spans
        # 0.49-1.26 over 27 samples, so a 2x target sits inside the noise. 4*eps gives 8x the floor,
        # which is the same 3.2x margin over the worst observed constant that `residual_floor` uses.
        #
        # Derive it from the operator dtype, not from the initial guess: a float32 xinit on a complex128
        # problem would otherwise silently loosen this by nine orders of magnitude. `work_dtype` is
        # already that promotion.
        rtol = 4.0 * float(jnp.finfo(work_dtype).eps)

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
