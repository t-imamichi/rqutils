"""Tests for :mod:`rqutils.math`.

``matrix_ufunc`` unitary-diagonalizes an array of normal matrices, applies a scalar function to the
eigenvalues, and reassembles. ``scipy.linalg`` provides an independent reference for the two wrappers
(``matrix_exp``, ``matrix_angle``), so most assertions here are against scipy rather than against
hand-derived values.

No bugs were found in this module -- unlike every other suite in this repo, this one is pure
regression locking. The one behaviour worth pinning deliberately is that a wrong ``hermitian`` hint
produces a *silently wrong answer* rather than a raise: see
:meth:`TestHermitianHint.test_wrong_hint_is_silently_wrong`.
"""

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.linalg as sla

import rqutils.math as rm

DIMS = [1, 2, 3, 5]


def hermitian(dim, rng):
    mat = rng.normal(size=(dim, dim)) + 1.0j * rng.normal(size=(dim, dim))
    return mat + mat.conjugate().T


def unitary(dim, rng):
    """A unitary matrix: normal but neither Hermitian nor anti-Hermitian."""
    return sla.expm(1.0j * hermitian(dim, rng))


class TestMatrixExp:
    """``matrix_exp`` against ``scipy.linalg.expm``."""

    @pytest.mark.parametrize("dim", DIMS)
    def test_hermitian_input_with_hint(self, dim):
        rng = np.random.default_rng(20260804)
        mat = hermitian(dim, rng)
        got = np.asarray(rm.matrix_exp(mat, hermitian=1))
        assert np.abs(got - sla.expm(mat)).max() < 1e-11

    @pytest.mark.parametrize("dim", DIMS)
    def test_hermitian_input_without_hint(self, dim):
        """``hermitian=0`` uses the general ``eig`` path, which must agree with the ``eigh`` one."""
        rng = np.random.default_rng(20260804)
        mat = hermitian(dim, rng)
        got = np.asarray(rm.matrix_exp(mat, hermitian=0))
        assert np.abs(got - sla.expm(mat)).max() < 1e-10

    @pytest.mark.parametrize("dim", DIMS)
    def test_anti_hermitian_input(self, dim):
        """``hermitian=-1`` means the argument is ``i`` times a Hermitian matrix.

        The implementation diagonalizes ``1j * mat`` with ``eigh`` and undoes the factor, so this is
        a genuinely different code path from both ``hermitian=1`` and ``hermitian=0``.
        """
        rng = np.random.default_rng(20260804)
        mat = 1.0j * hermitian(dim, rng)
        got = np.asarray(rm.matrix_exp(mat, hermitian=-1))
        assert np.abs(got - sla.expm(mat)).max() < 1e-12

    def test_unitary_input(self):
        """A unitary matrix is normal but not Hermitian, so only the general path applies."""
        rng = np.random.default_rng(20260804)
        mat = unitary(4, rng)
        got = np.asarray(rm.matrix_exp(mat, hermitian=0))
        assert np.abs(got - sla.expm(mat)).max() < 1e-12

    def test_diagonal_input(self):
        """A diagonal matrix is the degenerate case the docstring warns about for gradients.

        The value must still be correct even though the eigenvectors are ambiguous.
        """
        mat = np.diag([1.0, 2.0, 3.0]).astype(np.complex128)
        got = np.asarray(rm.matrix_exp(mat, hermitian=1))
        assert np.abs(got - np.diag(np.exp([1.0, 2.0, 3.0]))).max() < 1e-13

    def test_zero_matrix_gives_the_identity(self):
        got = np.asarray(rm.matrix_exp(np.zeros((3, 3), dtype=np.complex128), hermitian=1))
        assert np.abs(got - np.eye(3)).max() < 1e-13

    @pytest.mark.parametrize("hint", [0, 1, -1])
    def test_batched_input(self, hint):
        """Leading axes are batch dimensions; each slice must be exponentiated independently."""
        rng = np.random.default_rng(20260804)
        base = hermitian(4, rng)
        if hint == -1:
            base = 1.0j * base
        batch = np.stack([base, 2.0 * base, -0.5 * base])
        got = np.asarray(rm.matrix_exp(batch, hermitian=hint))
        assert got.shape == (3, 4, 4)
        for index, scale in enumerate((1.0, 2.0, -0.5)):
            assert np.abs(got[index] - sla.expm(scale * base)).max() < 1e-9, f"slice {index}"


class TestMatrixAngle:
    """``matrix_angle`` recovers the generator of a unitary, within the principal branch."""

    def test_recovers_the_generator(self):
        """``matrix_angle(expm(i H)) == H`` only when H's eigenvalues lie in ``(-pi, pi]``.

        ``np.angle`` returns the principal value, so an eigenvalue outside that range comes back
        wrapped -- with an unscaled random H the discrepancy is O(pi), which looks like a failure but
        is the branch cut. Scaling H into the branch first is what makes this a real test.
        """
        rng = np.random.default_rng(20260804)
        generator = hermitian(5, rng)
        # Scale so every eigenvalue sits strictly inside (-pi, pi).
        generator *= np.pi * 0.9 / np.abs(np.linalg.eigvalsh(generator)).max()
        got = np.asarray(rm.matrix_angle(sla.expm(1.0j * generator)))
        assert np.abs(got - generator).max() < 1e-12

    def test_eigenvalues_outside_the_branch_wrap(self):
        """Documents the wrapping rather than pretending it is a bug.

        A caller who does not scale into the branch gets a different (still correct) representative
        of the same rotation, so this asserts the *rotation* matches even when the generator does not.
        """
        rng = np.random.default_rng(20260804)
        generator = hermitian(4, rng)
        assert np.abs(np.linalg.eigvalsh(generator)).max() > np.pi, "fixture must exceed the branch"
        recovered = np.asarray(rm.matrix_angle(sla.expm(1.0j * generator)))
        assert np.abs(recovered - generator).max() > 1.0, "expected branch wrapping"
        # But exponentiating the recovered generator returns the same unitary.
        assert np.abs(sla.expm(1.0j * recovered) - sla.expm(1.0j * generator)).max() < 1e-11


class TestWithDiagonals:
    """``with_diagonals=True`` additionally returns ``operator`` applied to the eigenvalues."""

    def test_returns_the_transformed_eigenvalues(self):
        rng = np.random.default_rng(20260804)
        mat = hermitian(5, rng)
        result = rm.matrix_exp(mat, hermitian=1, with_diagonals=True)
        assert len(result) == 2
        matrix_result, diagonals = np.asarray(result[0]), np.asarray(result[1])
        assert matrix_result.shape == (5, 5)
        assert diagonals.shape == (5,)
        # The returned diagonals must be exp() of the eigenvalues, matching the reassembled matrix.
        assert np.allclose(np.sort(diagonals.real), np.sort(np.exp(np.linalg.eigvalsh(mat))))
        assert np.allclose(
            np.sort(np.linalg.eigvals(matrix_result).real), np.sort(diagonals.real), atol=1e-10
        )

    def test_single_return_without_the_flag(self):
        rng = np.random.default_rng(20260804)
        got = rm.matrix_exp(hermitian(3, rng), hermitian=1)
        assert np.asarray(got).shape == (3, 3), "expected a bare array, not a tuple"


class TestHermitianHint:
    """The ``hermitian`` flag is an assertion by the caller, not something the code verifies."""

    def test_wrong_hint_is_silently_wrong(self):
        """``hermitian=1`` on a non-Hermitian matrix returns a wrong answer with no warning.

        Pinned deliberately: the function takes the caller's word and calls ``eigh``, which reads
        only one triangle, so the result is the exponential of a *different* matrix. Measured here at
        an absolute error of order 10 -- not a rounding artifact, a different answer. Anyone tempted
        to add validation should note this test asserts the current contract, so changing it to raise
        is a deliberate behaviour change and will fail here first.
        """
        rng = np.random.default_rng(20260804)
        anti_hermitian = 1.0j * hermitian(5, rng)
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # no warning is emitted, so this must not trip
            got = np.asarray(rm.matrix_exp(anti_hermitian, hermitian=1))
        assert np.abs(got - sla.expm(anti_hermitian)).max() > 1.0, (
            "a wrong hermitian hint is expected to give a wrong answer; if this now agrees, the "
            "function started validating its input and this test should be rewritten"
        )

    def test_hint_true_equals_hint_one(self):
        """``hermitian`` is annotated ``int | bool``, so ``True`` must behave as ``1``."""
        rng = np.random.default_rng(20260804)
        mat = hermitian(4, rng)
        assert np.allclose(
            np.asarray(rm.matrix_exp(mat, hermitian=True)),
            np.asarray(rm.matrix_exp(mat, hermitian=1)),
        )


class TestNpmodParity:
    """``npmod=jax.numpy`` must agree with numpy, and survive ``jax.jit``."""

    @pytest.mark.parametrize("hint", [0, 1, -1])
    def test_matrix_exp_parity(self, hint):
        rng = np.random.default_rng(20260804)
        mat = hermitian(4, rng)
        if hint == -1:
            mat = 1.0j * mat
        from_np = np.asarray(rm.matrix_exp(mat, hermitian=hint))
        from_jnp = np.asarray(rm.matrix_exp(jnp.asarray(mat), hermitian=hint, npmod=jnp))
        assert np.abs(from_np - from_jnp).max() < 1e-10

    def test_matrix_exp_under_jit(self):
        """The point of ``npmod`` is traceability, which only ``jit`` actually proves."""
        rng = np.random.default_rng(20260804)
        mat = hermitian(4, rng)
        traced = jax.jit(lambda m: rm.matrix_exp(m, hermitian=1, npmod=jnp))
        assert (
            np.abs(
                np.asarray(traced(jnp.asarray(mat))) - np.asarray(rm.matrix_exp(mat, hermitian=1))
            ).max()
            < 1e-11
        )

    def test_matrix_angle_parity(self):
        rng = np.random.default_rng(20260804)
        mat = unitary(4, rng)
        from_np = np.asarray(rm.matrix_angle(mat))
        from_jnp = np.asarray(rm.matrix_angle(jnp.asarray(mat), npmod=jnp))
        assert np.abs(from_np - from_jnp).max() < 1e-11

    def test_with_diagonals_under_jit(self):
        """The tuple return has to survive tracing too, not just the single-array one."""
        rng = np.random.default_rng(20260804)
        mat = hermitian(4, rng)
        traced = jax.jit(lambda m: rm.matrix_exp(m, hermitian=1, with_diagonals=True, npmod=jnp))
        matrix_result, diagonals = traced(jnp.asarray(mat))
        assert np.asarray(matrix_result).shape == (4, 4)
        assert np.asarray(diagonals).shape == (4,)


class TestMatrixUfunc:
    """The general entry point, with operators other than exp/angle."""

    def test_arbitrary_operator(self):
        """``matrix_ufunc(sqrt, ...)`` on a positive-definite matrix is a matrix square root."""
        rng = np.random.default_rng(20260804)
        mat = hermitian(4, rng)
        positive = mat @ mat.conjugate().T  # positive semidefinite, so sqrt is real
        root = np.asarray(rm.matrix_ufunc(np.sqrt, positive, hermitian=1))
        assert np.abs(root @ root - positive).max() < 1e-9

    def test_identity_operator_returns_the_input(self):
        """Applying the identity to the eigenvalues must reassemble the original matrix.

        This isolates the diagonalize/reassemble machinery from any particular operator: any error in
        the unitary inversion shows up here with nothing else to hide behind.
        """
        rng = np.random.default_rng(20260804)
        mat = hermitian(5, rng)
        assert (
            np.abs(np.asarray(rm.matrix_ufunc(lambda x: x, mat, hermitian=1)) - mat).max() < 1e-12
        )

    @pytest.mark.parametrize(
        "bad,match",
        [(np.zeros((2, 3)), "square"), (np.zeros(4), "dimension")],
    )
    def test_bad_shapes_raise(self, bad, match):
        with pytest.raises(np.linalg.LinAlgError, match=match):
            rm.matrix_exp(bad)
