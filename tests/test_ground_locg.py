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

from rqutils.ground_locg import _project_out, eigenpair_2x2, eigenpair_3x3, ground_locg


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


class TestEigenpair3x3:
    """Lowest eigenpair of a 3x3 Hermitian matrix, via Cardano's method.

    The shipped predecessor returned 4066 NaNs over 20000 random inputs, and a wrong eigenvector
    1148 times in 5000.
    """

    def test_rank_deficient_column_pair(self):
        """Item I3, the sharpest case in the audit: diag(5, 6, 1).

        The old kernel took the null vector as ``cross(mat[:, 1], mat[:, 2])``. When that particular
        pair is rank deficient the cross product vanishes and the result points nowhere useful. On
        this innocuous input it returned an *exact eigenvalue* alongside an eigenvector with
        residual 0.67 -- so a test asserting only on the eigenvalue would have passed. Assert both.
        """
        mat = np.diag([5., 6., 1.])
        eigval, eigvec = eigenpair_3x3(jnp.asarray(mat))
        eigval = float(eigval)
        assert eigval == pytest.approx(lowest(mat), abs=1e-13)
        assert rel_resid(mat, eigval, eigvec) < 1e-13
        assert np.linalg.norm(np.asarray(eigvec)) == pytest.approx(1.)

    def test_identity_rank_zero(self):
        """Item I3 rank-0 fallback: every cross product vanishes for a multiple of the identity."""
        mat = np.eye(3)
        eigval, eigvec = eigenpair_3x3(jnp.asarray(mat))
        eigval = float(eigval)
        assert eigval == pytest.approx(1., abs=1e-13)
        assert rel_resid(mat, eigval, eigvec) < 1e-13
        assert np.linalg.norm(np.asarray(eigvec)) == pytest.approx(1.)

    def test_degenerate_lowest_rank_one(self):
        """Item I3 rank-1 fallback: a degenerate lowest eigenvalue.

        Every cross product is numerical noise here; the null space is the orthogonal complement of
        the largest column, and any member of it is a valid eigenvector.
        """
        mat = np.diag([1., 1., 7.])
        eigval, eigvec = eigenpair_3x3(jnp.asarray(mat))
        eigval = float(eigval)
        assert eigval == pytest.approx(1., abs=1e-13)
        assert rel_resid(mat, eigval, eigvec) < 1e-13
        assert np.linalg.norm(np.asarray(eigvec)) == pytest.approx(1.)

    def test_large_shift(self):
        """Item I1: at shift 1e9 the radicand under ``sqrt`` went negative and returned NaN.

        Not an exotic input -- this is the ordinary case for a physical Hamiltonian, which is rarely
        traceless, and is exactly what ``sqd.py`` feeds this solver.
        """
        mat = np.diag([1., 2., 3.]) + 1e9 * np.eye(3)
        eigval, eigvec = eigenpair_3x3(jnp.asarray(mat))
        eigval = float(eigval)
        assert np.isfinite(eigval)
        assert eigval == pytest.approx(lowest(mat), rel=1e-13)
        assert rel_resid(mat, eigval, eigvec) < 1e-13

    @pytest.mark.parametrize('exponent', [-160, 150])
    def test_extreme_scale(self, exponent):
        """Item I2: ``c0`` is cubic in the entries, so unbalanced it overflows or underflows.

        Measured on the old kernel: relative error 7.8e-1 at 1e-160, NaN at 1e150.
        """
        mat = np.diag([1., 2., 3.]) * 10. ** exponent
        eigval, eigvec = eigenpair_3x3(jnp.asarray(mat))
        eigval = float(eigval)
        assert np.isfinite(eigval)
        assert eigval == pytest.approx(lowest(mat), rel=1e-13)
        assert rel_resid(mat, eigval, eigvec) < 1e-13

    @pytest.mark.parametrize('complex_', [True, False])
    def test_random_sweep(self, complex_):
        """Aggregate accuracy over seeded random input. Supplements the targeted cases above.

        The old kernel produced 4066 NaNs in 20000 such matrices.
        """
        rng = np.random.default_rng(20260804)
        worst_eigval = worst_residual = 0.
        for _ in range(2000):
            mat = herm(3, rng, complex_=complex_)
            eigval, eigvec = eigenpair_3x3(jnp.asarray(mat))
            eigval = float(eigval)
            assert np.isfinite(eigval)
            assert np.all(np.isfinite(np.asarray(eigvec)))
            scale = np.abs(mat).max()
            worst_eigval = max(worst_eigval, abs(eigval - lowest(mat)) / scale)
            worst_residual = max(worst_residual, rel_resid(mat, eigval, eigvec))
        assert worst_eigval < 1e-13, f'worst relative eigenvalue error {worst_eigval:.2e}'
        assert worst_residual < 1e-13, f'worst relative residual {worst_residual:.2e}'


class TestDtypes:
    """Operator and ``xinit`` dtype combinations.

    A float32 ``xinit`` against a float64 operator raised ``TypeError`` from ``while_loop``: theta
    entered the carry as float32 from ``body_iter1``, whose projected matrix takes its dtype from
    ``xinit``, and returned float64 from ``body``, whose projected matrix is built from the matvec
    output. Only a lower-precision ``xinit`` failed; a float32 operator with a float64 ``xinit`` was
    always fine.

    This is adjacent to audit item A3, which concerned the same configuration: the rewrite fixed the
    *tolerance* derivation for it but left the state dtype inconsistent.
    """

    @pytest.mark.parametrize('operator_dtype,xinit_dtype', [
        (jnp.float64, jnp.float64),
        (jnp.float32, jnp.float32),
        (jnp.float64, jnp.float32),
        (jnp.complex128, jnp.float32),
    ])
    def test_dtype_combinations_solve(self, operator_dtype, xinit_dtype):
        rng = np.random.default_rng(20260804)
        mat = herm(60, rng, complex_=False)
        xinit = rng.normal(size=60)
        eigval, eigvec, _, converged = ground_locg(jnp.asarray(mat, dtype=operator_dtype),
                                                  jnp.asarray(xinit, dtype=xinit_dtype))
        assert bool(converged)
        assert np.all(np.isfinite(np.asarray(eigvec)))
        # float32 arithmetic reaches only ~1e-6 relative; float64 operators reach ~1e-13.
        tolerance = 1e-5 if jnp.finfo(operator_dtype).bits == 32 else 1e-12
        assert float(eigval) == pytest.approx(lowest(mat), rel=tolerance)

    def test_low_precision_xinit_does_not_degrade_result(self):
        """Audit item A3's concern: a low-precision initial guess must not cost accuracy.

        A4/A3 measured the old ``tol`` derivation trading eight orders of magnitude of accuracy on
        the dtype of the *starting guess* alone.
        """
        rng = np.random.default_rng(20260804)
        mat = jnp.asarray(herm(60, rng, complex_=False), dtype=jnp.float64)
        xinit = rng.normal(size=60)
        from_f64 = float(ground_locg(mat, jnp.asarray(xinit, dtype=jnp.float64))[0])
        from_f32 = float(ground_locg(mat, jnp.asarray(xinit, dtype=jnp.float32))[0])
        assert from_f32 == pytest.approx(from_f64, rel=1e-12)


class TestGroundLocg:
    """End-to-end LOBPCG solves."""

    @pytest.mark.parametrize('shift', [0., 1e3, -1e3, 1e6, 1e9])
    @pytest.mark.parametrize('complex_', [True, False])
    def test_solves_and_converges(self, shift, complex_):
        """Three assertions per case, each keyed to a different audit item.

        Item I4 -- ``reltol`` used ``norm(Ax) - theta``, a cancellation measured going *negative*,
        which made the convergence test unsatisfiable: the solver never converged and always burned
        ``maxiter``. Fixing that sign alone was a 24-33x iteration reduction.

        Item I5, the audit's most serious finding -- loss of x/y orthogonality made the standard
        Rayleigh-Ritz step return a theta *beneath* the true minimum (observed 6.0e8 against a true
        minimum of 1.0e9), which is impossible for a genuine Rayleigh quotient. A caller checking
        only "is it finite" would have accepted it.
        """
        rng = np.random.default_rng(20260804)
        mat = herm(200, rng, complex_=complex_) + shift * np.eye(200)
        reference = lowest(mat)
        scale = np.abs(mat).max()
        xinit = rng.normal(size=200)
        if complex_:
            xinit = xinit + 0.j
        eigval, eigvec, _, converged = ground_locg(jnp.asarray(mat), jnp.asarray(xinit))
        eigval = float(eigval)

        assert bool(converged), 'solver exhausted maxiter (item I4)'
        assert abs(eigval - reference) / scale < 1e-12
        assert eigval > reference - 1e-8 * scale, (
            f'theta {eigval!r} is below the true minimum {reference!r} (item I5)'
        )

    def test_maxiter_too_small_reports_not_converged(self):
        """Item A1: ``niter == maxiter`` is ambiguous, so the flag is the only usable signal.

        Measured on the old code with ``maxiter=5``: a confident-looking eigenvalue whose true
        relative residual was 1.1e-1, and no way for the caller to tell.
        """
        rng = np.random.default_rng(20260804)
        mat = herm(60, rng, complex_=False)
        eigval, _, niter, converged = ground_locg(jnp.asarray(mat),
                                                  jnp.asarray(rng.normal(size=60)), maxiter=5)
        assert not bool(converged)
        assert int(niter) == 5
        assert np.isfinite(float(eigval))

    def test_maxiter_zero_returns_rayleigh_quotient(self):
        """Item A2: the old code returned the literal state initializer, 0.0.

        For a matrix whose true lowest eigenvalue was -21.35 it reported 0.000000 with a normalized
        eigenvector and niter=0.
        """
        rng = np.random.default_rng(20260804)
        mat = herm(60, rng, complex_=False)
        xinit = rng.normal(size=60)
        eigval, _, niter, converged = ground_locg(jnp.asarray(mat), jnp.asarray(xinit), maxiter=0)
        normalized = xinit / np.linalg.norm(xinit)
        assert int(niter) == 0
        assert not bool(converged)
        assert float(eigval) == pytest.approx(normalized @ symmetrize(mat) @ normalized)

    def test_zero_xinit_is_finite(self):
        """Item A4: the old ``xinit /= norm(xinit)`` had no zero guard and returned NaN."""
        rng = np.random.default_rng(20260804)
        mat = herm(60, rng, complex_=False)
        eigval, eigvec, _, _ = ground_locg(jnp.asarray(mat), jnp.zeros(60))
        assert np.isfinite(float(eigval))
        assert np.all(np.isfinite(np.asarray(eigvec)))

    def test_integer_xinit_one_hot(self):
        """The integer convenience path builds a one-hot vector internally."""
        rng = np.random.default_rng(20260804)
        mat = herm(60, rng, complex_=False)
        eigval, _, _, converged = ground_locg(jnp.asarray(mat), jnp.asarray(3))
        assert bool(converged)
        assert float(eigval) == pytest.approx(lowest(mat), rel=1e-12)

    def test_xinit_scale_invariance(self):
        """``xinit`` is normalized on entry, so its magnitude cannot matter."""
        rng = np.random.default_rng(20260804)
        mat = jnp.asarray(herm(60, rng, complex_=False))
        xinit = jnp.asarray(rng.normal(size=60))
        assert float(ground_locg(mat, xinit * 1e8)[0]) == float(ground_locg(mat, xinit)[0])

    def test_xinit_not_mutated(self):
        """Despite the in-place-looking ``xinit = normalize(xinit)``, JAX arrays are immutable."""
        rng = np.random.default_rng(20260804)
        mat = jnp.asarray(herm(60, rng, complex_=False))
        xinit = jnp.asarray(rng.normal(size=60))
        before = np.asarray(xinit).copy()
        ground_locg(mat, xinit)
        assert np.array_equal(before, np.asarray(xinit))

    def test_debug_diagnostics(self):
        """Item A5: ``debug=True`` used to raise TypeError, being unreachable from the entry point.

        The debug path uses ``jax.lax.scan`` with ``length=maxiter`` and so has no early exit; it
        returns ``maxiter + 2`` rows (the two seed steps plus every scanned iteration) and the rows
        past convergence are post-convergence noise. Assert the shape and that the eigenvalue is
        still right -- not that every row is meaningful.
        """
        rng = np.random.default_rng(20260804)
        mat = herm(40, rng, complex_=False)
        maxiter = 12
        result = ground_locg(jnp.asarray(mat), jnp.asarray(rng.normal(size=40)),
                             maxiter=maxiter, debug=True)
        assert len(result) == 5
        diagnostics = result[4]
        for key in ('x', 'y', 'r', 'theta', 'rho', 'kappa', 'sas', 'reltol', 'converged'):
            assert np.shape(diagnostics[key])[0] == maxiter + 2, f'{key} row count'
