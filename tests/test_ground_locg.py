"""Tests for :mod:`rqutils.ground_locg`.

Organized by defect. Most tests correspond to a numbered item in ``docs/locg.md``, the audit of the
previous implementation of this module; each such test names its item and the measured old failure.
That audit's closing warning shapes the design here:

    Aggregate random sampling does not exercise data-dependent branches evenly. [...] Both of my
    safety nets missed a sign error that a two-line targeted test caught at 4000/4000 failures.

So targeted per-branch tests are the backbone and randomized sweeps are a supplement.
"""
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import herm, lowest, rel_resid, symmetrize

from rqutils.ground_locg import _project_out, eigenpair_2x2


class TestProjectOut:
    """``_project_out`` returns either exactly zero or a vector of norm >= 0.99 (item I6)."""

    def test_vector_in_basis_span_returns_exactly_zero(self):
        """A residual lying wholly in span{x, y} must come back as exact zero, not as noise.

        The zeroing guard is what item I7's convergence check keys off: a zeroed search direction
        means {x, y} already spans the residual, so no further iteration can lower theta.
        """
        basis = (jnp.array([1., 0., 0.]), jnp.array([0., 1., 0.]))
        out = np.asarray(_project_out(basis, jnp.array([1., 1., 0.])))
        assert np.array_equal(out, np.zeros(3))

    def test_orthogonal_vector_is_not_normalized_to_unity(self):
        """The postcondition is norm >= 0.99, NOT norm == 1.

        Item I6: callers feeding this to a standard Rayleigh-Ritz step must renormalize themselves.
        At shift 1e9 a |p| of 0.999 displaced theta by 2e6, below the true minimum. This test exists
        so that a future edit "tidying" the trailing subtraction into a normalization fails loudly.

        Two assertions: (1) the orthonormal case must return exactly [0,0,1], and (2) a tilted
        (non-orthonormal) basis must return a result with norm strictly between 0.99 and 1.0,
        proving the vector is not renormalized to unity.
        """
        # Orthonormal basis: _project_out subtracts <e_i|v> e_i, leaving exactly [0, 0, 1].
        basis = (jnp.array([1., 0., 0.]), jnp.array([0., 1., 0.]))
        out = np.asarray(_project_out(basis, jnp.array([0., 0., 2.])))
        assert np.linalg.norm(out) >= 0.99
        assert np.allclose(out, [0., 0., 1.])

        # Non-orthonormal basis: _project_out subtracts <b|v> b without a Gram solve, so the
        # result falls short of unit norm. This discriminates "masked to >= 0.99" from
        # "renormalized to unity". The result must be strictly less than 1.0.
        basis = (jnp.array([1., 0., 0.]), jnp.array([0.3, np.sqrt(1. - 0.3 ** 2), 0.]))
        out = np.asarray(_project_out(basis, jnp.array([0.3, 0.4, 1.0])))
        norm = np.linalg.norm(out)
        assert norm >= 0.99, f"Norm {norm} dropped below 0.99"
        assert norm < 1.0, f"Norm {norm} is not strictly less than 1.0; likely renormalized"


def two_by_two(rng, delta_sign):
    """Return a 2x2 complex Hermitian matrix whose delta = (d0 - d1) / 2 has ``delta_sign``.

    ``eigenpair_2x2`` selects which row of the singular shifted matrix yields the null vector on the
    sign of delta, so the two signs are genuinely different code paths and must be tested apart.
    """
    offd = rng.normal() + 1.j * rng.normal()
    diag = rng.normal(size=2)
    if np.sign(diag[0] - diag[1]) != delta_sign:
        diag = diag[::-1]
    return np.array([[diag[0], offd.conjugate()], [offd, diag[1]]])


class TestEigenpair2x2:
    """Lowest eigenpair of a 2x2 Hermitian matrix.

    The shipped predecessor returned 2353 NaNs over 40000 random inputs.
    """

    @pytest.mark.parametrize('mat', [np.diag([1., 5.]), np.diag([5., 1.]), np.eye(2)])
    def test_diagonal_and_identity(self, mat):
        """Old kernel returned NaN here: its eigenvector formula computed 0/0.

        Any multiple of the identity hit it, and so did any already-diagonal input whose lower
        eigenvalue sat in position 0 -- diag(1, 5) returned NaN.
        """
        eigval, eigvec = eigenpair_2x2(jnp.asarray(mat))
        eigval = float(eigval)
        assert np.isfinite(eigval)
        assert np.all(np.isfinite(np.asarray(eigvec)))
        assert eigval == pytest.approx(lowest(mat), abs=1e-13 * np.abs(mat).max())
        assert rel_resid(mat, eigval, eigvec) < 1e-13
        assert np.linalg.norm(np.asarray(eigvec)) == pytest.approx(1.)

    def test_large_shift(self):
        """Item I1: ``tr*tr - 4*det`` cancelled, reaching relative error 5.8e-1 at shift 1e9."""
        mat = np.array([[1., 0.5], [0.5, 2.]]) + 1e9 * np.eye(2)
        eigval, eigvec = eigenpair_2x2(jnp.asarray(mat))
        eigval = float(eigval)
        assert eigval == pytest.approx(lowest(mat), rel=1e-13)
        assert rel_resid(mat, eigval, eigvec) < 1e-13

    @pytest.mark.parametrize('exponent', [-160, 160])
    def test_extreme_scale(self, exponent):
        """Item I2: unbalanced intermediates carried 4.9e-2 error at 1e-160 and NaN at 1e160."""
        mat = np.array([[1., 0.5], [0.5, 2.]]) * 10. ** exponent
        eigval, eigvec = eigenpair_2x2(jnp.asarray(mat))
        eigval = float(eigval)
        assert np.isfinite(eigval)
        assert eigval == pytest.approx(lowest(mat), rel=1e-13)
        assert rel_resid(mat, eigval, eigvec) < 1e-13

    @pytest.mark.parametrize('delta_sign', [1., -1.])
    def test_both_delta_branches(self, delta_sign):
        """Each sign of delta separately -- the audit's own sign error hid from aggregate sampling.

        Its first fix had the row-2 branch as ``[rad - delta, +b]`` instead of ``-b``. That passed
        every test it had, because those tests happened to generate only delta > 0. Forcing delta < 0
        exposed it at 4000/4000 failures, residual 3.8e-1. End-to-end convergence hides it too:
        ``eigenpair_2x2`` is called exactly once, in ``body_iter1``, so a wrong eigenvector there is
        quietly repaired by later iterations.
        """
        rng = np.random.default_rng(20260804)
        worst_eigval = worst_residual = 0.
        for _ in range(2000):
            mat = two_by_two(rng, delta_sign)
            diag = np.diagonal(mat).real
            assert np.sign((diag[0] - diag[1]) / 2.) == delta_sign  # the branch really is forced
            eigval, eigvec = eigenpair_2x2(jnp.asarray(mat))
            eigval = float(eigval)
            assert np.isfinite(eigval)
            scale = np.abs(mat).max()
            worst_eigval = max(worst_eigval, abs(eigval - lowest(mat)) / scale)
            worst_residual = max(worst_residual, rel_resid(mat, eigval, eigvec))
        assert worst_eigval < 1e-13, f'worst relative eigenvalue error {worst_eigval:.2e}'
        assert worst_residual < 1e-13, f'worst relative residual {worst_residual:.2e}'
