"""
======================
Math utility functions
======================

.. currentmodule:: rqutils.math

Math API
========

.. autoclass:: Symmetry
   :members:
.. autofunction:: matrix_ufunc
.. autofunction:: matrix_exp
.. autofunction:: matrix_angle
"""

import enum
from collections.abc import Callable
from types import ModuleType

import numpy as np
from numpy.typing import ArrayLike, NDArray


class Symmetry(enum.IntEnum):
    """Which diagonalization route :func:`matrix_ufunc` should take.

    An ``IntEnum`` rather than a plain ``Enum``, so the historical ``0``/``1``/``-1`` (and
    ``True``/``False``) keep working -- this names the values, it does not replace them.

    Attributes:
        GENERAL: No symmetry assumed; uses ``eig``.
        HERMITIAN: Uses ``eigh``, which reads only one triangle. **Asserted by the caller, not
            verified**: on a non-Hermitian input this returns a well-formed spectrum of a
            *different* operator, and validating it would cost an ``O(n^2)`` comparison per call.
        ANTI_HERMITIAN: Uses ``eigh`` on ``i*mat`` and undoes the rotation.
    """

    GENERAL = 0
    HERMITIAN = 1
    ANTI_HERMITIAN = -1


def _check_symmetry(hermitian: object) -> None:
    """Raise unless ``hermitian`` names one of the three diagonalization routes.

    The dispatch in :func:`matrix_ufunc` is an if/elif chain whose ``else`` catches everything, so an
    out-of-range value used to select the general ``eig`` path silently -- slower, less accurate, and
    indistinguishable from having asked for it. ``hermitian=2`` looked like a hint and was one only
    by accident.

    Args:
        hermitian: The caller's value.

    Raises:
        TypeError: If it is not an int (``bool`` and :class:`Symmetry` are ints, so both pass).
        ValueError: If it is an int outside ``{-1, 0, 1}``.
    """
    if isinstance(hermitian, bool | Symmetry):
        return
    if not isinstance(hermitian, int):
        raise TypeError(
            f"`hermitian` must be an int in {{-1, 0, 1}} or a `Symmetry`, got {hermitian!r}"
        )
    if hermitian not in (-1, 0, 1):
        raise ValueError(
            f"`hermitian` is {hermitian!r}, but must be 1/True (Hermitian, `Symmetry.HERMITIAN`), "
            "-1 (anti-Hermitian, `Symmetry.ANTI_HERMITIAN`), or 0/False (general, "
            "`Symmetry.GENERAL`). Out-of-range values used to fall through to the general `eig` "
            "path silently."
        )


def matrix_ufunc(
    operator: Callable,
    mat: ArrayLike,
    *,
    hermitian: int | bool | Symmetry = Symmetry.GENERAL,
    with_diagonals: bool = False,
    npmod: ModuleType = np,
) -> NDArray | tuple[NDArray, NDArray]:
    """Apply a unitary-invariant unary matrix operator to an array of normal matrices.

    The argument `mat` must be an array of normal (i.e. square diagonalizable) matrices in the last
    two dimensions. This function unitary-diagonalizes the matrices, applies `operator` to the
    diagonals, and inverts the diagonalization.

    **Diagonalization and gradient**

    When using this function with an autodiff library (e.g. JAX), the gradient diverges when an
    input parameter controls off-diagonal elements of ``mat`` but ``mat`` is diagonal. Use an
    alternative function (that is hopefully available) in such cases:

    .. code-block:: python

        # Reshape the matrix to gather all off-diagonal elements to a block ([:, 1:])
        mat_dim = mat.shape[-1]
        diag_checker = mat.reshape(-1, mat_dim ** 2)
        # The very last element is a part of diagonal -> can ignore for this purpose
        diag_checker = diag_checker[:, :-1].reshape(-1, mat_dim - 1, mat_dim + 1)
        is_diagonal = ~jnp.any(diag_checker[:, :, 1:], axis=(1, 2))
        has_diagonal = jnp.any(is_diagonal)

        result = jax.lax.cond(has_diagonal,
                              alternative_X,
                              functools.partial(matrix_ufunc, X),
                              mat)

    Everything after ``mat`` is **keyword-only**. ``hermitian`` used to be the second positional
    parameter and ``with_diagonals`` the third, so ``matrix_exp(mat, True)`` read naturally as "with
    diagonals" and meant "Hermitian" -- and since ``with_diagonals`` changes the return *arity*, the
    mistake surfaced as an unpacking error somewhere else, if at all.

    Args:
        operator: Unary operator to be applied to the diagonals of ``mat``.
        mat: Array of normal matrices (shape (..., n, n)). No check on normality is performed.
        hermitian: Which diagonalization route to take: 1/True or :attr:`Symmetry.HERMITIAN` ->
            ``mat`` is Hermitian, -1 or :attr:`Symmetry.ANTI_HERMITIAN` -> anti-Hermitian, 0/False or
            :attr:`Symmetry.GENERAL` -> no assumption. Note this is an **assertion by the caller**
            that is not verified: ``hermitian=1`` routes to ``eigh``, which reads only one triangle,
            so a non-Hermitian input yields a well-formed spectrum of a *different* operator.
            Checking it would cost an ``O(n^2)`` comparison per call, so the contract stands -- see
            ``tests/test_math.py::TestHermitianHint``.
        with_diagonals: If True, also return the array ``operator(eigenvalues)``.

    Returns:
        An array corresponding to `operator(mat)`. If `with_diagonals==True`, another array
        corresponding to `operator(eigvals)`.

    Raises:
        TypeError: If ``hermitian`` is not an int (``bool`` and :class:`Symmetry` are ints).
        ValueError: If ``hermitian`` is an int outside ``{-1, 0, 1}``. Such values used to fall
            through the dispatch's implicit ``else`` to the general ``eig`` path, silently.
    """
    _check_symmetry(hermitian)
    if hermitian in (1, True):
        eigvals, eigcols = npmod.linalg.eigh(mat)
    elif hermitian == -1:
        # numpy's stubs have no complex * ArrayLike overload; correct under the npmod convention.
        eigvals, eigcols = npmod.linalg.eigh(1.0j * mat)  # ty: ignore[unsupported-operator]
        eigvals = -1.0j * eigvals
    else:
        eigvals, eigcols = npmod.linalg.eig(mat)

    eigrows = npmod.conjugate(npmod.moveaxis(eigcols, -2, -1))
    op_eigvals = operator(eigvals)
    op_mat = npmod.matmul(eigcols * op_eigvals[..., None, :], eigrows)
    if with_diagonals:
        return op_mat, op_eigvals

    return op_mat


def matrix_exp(
    mat: ArrayLike,
    *,
    hermitian: int | bool | Symmetry = Symmetry.GENERAL,
    with_diagonals: bool = False,
    npmod: ModuleType = np,
) -> NDArray:
    """`matrix_ufunc(exp, ...)`"""
    return matrix_ufunc(
        npmod.exp, mat, hermitian=hermitian, with_diagonals=with_diagonals, npmod=npmod
    )


def matrix_angle(
    mat: ArrayLike,
    *,
    hermitian: int | bool | Symmetry = Symmetry.GENERAL,
    with_diagonals: bool = False,
    npmod: ModuleType = np,
) -> NDArray:
    """`matrix_ufunc(angle, ...)`"""
    return matrix_ufunc(
        npmod.angle, mat, hermitian=hermitian, with_diagonals=with_diagonals, npmod=npmod
    )
