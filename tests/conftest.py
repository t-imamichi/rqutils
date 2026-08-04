"""Shared configuration and helpers for the rqutils test suite.

``jax_enable_x64`` is set here, at conftest import time, so that it takes effect before any test
module imports ``rqutils``. This is load-bearing rather than incidental: without x64 JAX silently
produces float32/complex64 and every tolerance in this suite is wrong by nine orders of magnitude.
"""
import jax
jax.config.update('jax_enable_x64', True)

import numpy as np  # noqa: E402  (must follow the x64 config)


def herm(n, rng, complex_=True):
    """Return a random ``(n, n)`` Hermitian matrix drawn from ``rng``."""
    mat = rng.normal(size=(n, n))
    if complex_:
        mat = mat + 1.j * rng.normal(size=(n, n))
    return mat + mat.conjugate().T


def symmetrize(mat):
    """Mirror the lower triangle of ``mat`` over the diagonal.

    ``eigenpair_2x2`` and ``eigenpair_3x3`` read only the diagonal and the lower triangle, so a
    reference eigendecomposition must be taken of *this* matrix, not of the raw input. Comparing
    against ``eigvalsh`` of an unsymmetrized input compares against a different matrix.
    """
    lower = np.tril(mat)
    return lower + np.tril(mat, -1).conjugate().T


def lowest(mat):
    """Reference lowest eigenvalue of ``mat``, via LAPACK, after symmetrization."""
    return float(np.linalg.eigvalsh(symmetrize(mat))[0])


def rel_resid(mat, val, vec):
    """Eigenpair residual ``|Av - λv|``, scaled by ``max|A|``.

    The scaling is what makes a single tolerance usable across the shifted and extreme-scale cases:
    the 1e9-shifted 2x2 input has an absolute residual of 6e-8 but a relative residual of 6e-17.
    """
    mat = symmetrize(mat)
    vec = np.asarray(vec)
    return float(np.linalg.norm(mat @ vec - val * vec) / np.abs(mat).max())
