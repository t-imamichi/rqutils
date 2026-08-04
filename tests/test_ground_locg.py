"""Tests for :mod:`rqutils.ground_locg`.

Organized by defect. Most tests correspond to a numbered item in ``docs/locg.md``, the audit of the
previous implementation of this module; each such test names its item and the measured old failure.
That audit's closing warning shapes the design here:

    Aggregate random sampling does not exercise data-dependent branches evenly. [...] Both of my
    safety nets missed a sign error that a two-line targeted test caught at 4000/4000 failures.

So targeted per-branch tests are the backbone and randomized sweeps are a supplement.
"""

import warnings

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
        basis = (jnp.array([1.0, 0.0, 0.0]), jnp.array([0.0, 1.0, 0.0]))
        out = np.asarray(_project_out(basis, jnp.array([1.0, 1.0, 0.0])))
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
        basis = (jnp.array([1.0, 0.0, 0.0]), jnp.array([0.0, 1.0, 0.0]))
        out = np.asarray(_project_out(basis, jnp.array([0.0, 0.0, 2.0])))
        assert np.linalg.norm(out) >= 0.99
        assert np.allclose(out, [0.0, 0.0, 1.0])

        # Non-orthonormal basis: _project_out subtracts <b|v> b without a Gram solve, so the
        # result falls short of unit norm. This discriminates "masked to >= 0.99" from
        # "renormalized to unity". The result must be strictly less than 1.0.
        basis = (jnp.array([1.0, 0.0, 0.0]), jnp.array([0.3, np.sqrt(1.0 - 0.3**2), 0.0]))
        out = np.asarray(_project_out(basis, jnp.array([0.3, 0.4, 1.0])))
        norm = np.linalg.norm(out)
        assert norm >= 0.99, f"Norm {norm} dropped below 0.99"
        assert norm < 1.0, f"Norm {norm} is not strictly less than 1.0; likely renormalized"


def two_by_two(rng, delta_sign):
    """Return a 2x2 complex Hermitian matrix whose delta = (d0 - d1) / 2 has ``delta_sign``.

    ``eigenpair_2x2`` selects which row of the singular shifted matrix yields the null vector on the
    sign of delta, so the two signs are genuinely different code paths and must be tested apart.
    """
    offd = rng.normal() + 1.0j * rng.normal()
    diag = rng.normal(size=2)
    if np.sign(diag[0] - diag[1]) != delta_sign:
        diag = diag[::-1]
    return np.array([[diag[0], offd.conjugate()], [offd, diag[1]]])


class TestEigenpair2x2:
    """Lowest eigenpair of a 2x2 Hermitian matrix.

    The shipped predecessor returned 2353 NaNs over 40000 random inputs.
    """

    @pytest.mark.parametrize("mat", [np.diag([1.0, 5.0]), np.diag([5.0, 1.0]), np.eye(2)])
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
        assert np.linalg.norm(np.asarray(eigvec)) == pytest.approx(1.0)

    def test_large_shift(self):
        """Item I1: ``tr*tr - 4*det`` cancelled, reaching relative error 5.8e-1 at shift 1e9."""
        mat = np.array([[1.0, 0.5], [0.5, 2.0]]) + 1e9 * np.eye(2)
        eigval, eigvec = eigenpair_2x2(jnp.asarray(mat))
        eigval = float(eigval)
        assert eigval == pytest.approx(lowest(mat), rel=1e-13)
        assert rel_resid(mat, eigval, eigvec) < 1e-13

    @pytest.mark.parametrize("exponent", [-160, 160])
    def test_extreme_scale(self, exponent):
        """Item I2: unbalanced intermediates carried 4.9e-2 error at 1e-160 and NaN at 1e160."""
        mat = np.array([[1.0, 0.5], [0.5, 2.0]]) * 10.0**exponent
        eigval, eigvec = eigenpair_2x2(jnp.asarray(mat))
        eigval = float(eigval)
        assert np.isfinite(eigval)
        assert eigval == pytest.approx(lowest(mat), rel=1e-13)
        assert rel_resid(mat, eigval, eigvec) < 1e-13

    @pytest.mark.parametrize("delta_sign", [1.0, -1.0])
    def test_both_delta_branches(self, delta_sign):
        """Each sign of delta separately -- the audit's own sign error hid from aggregate sampling.

        Its first fix had the row-2 branch as ``[rad - delta, +b]`` instead of ``-b``. That passed
        every test it had, because those tests happened to generate only delta > 0. Forcing delta < 0
        exposed it at 4000/4000 failures, residual 3.8e-1. End-to-end convergence hides it too:
        ``eigenpair_2x2`` is called exactly once, in ``body_iter1``, so a wrong eigenvector there is
        quietly repaired by later iterations.
        """
        rng = np.random.default_rng(20260804)
        worst_eigval = worst_residual = 0.0
        for _ in range(2000):
            mat = two_by_two(rng, delta_sign)
            diag = np.diagonal(mat).real
            assert np.sign((diag[0] - diag[1]) / 2.0) == delta_sign  # the branch really is forced
            eigval, eigvec = eigenpair_2x2(jnp.asarray(mat))
            eigval = float(eigval)
            assert np.isfinite(eigval)
            scale = np.abs(mat).max()
            worst_eigval = max(worst_eigval, abs(eigval - lowest(mat)) / scale)
            worst_residual = max(worst_residual, rel_resid(mat, eigval, eigvec))
        assert worst_eigval < 1e-13, f"worst relative eigenvalue error {worst_eigval:.2e}"
        assert worst_residual < 1e-13, f"worst relative residual {worst_residual:.2e}"

    @pytest.mark.parametrize("delta_sign", [1.0, -1.0])
    def test_tiny_offdiagonal_relative_to_delta(self, delta_sign):
        """The delta-sign branch only matters when |offd| << |delta|.

        ``two_by_two`` above never generates an off-diagonal/delta ratio below 2.4e-2, so it cannot
        exercise the regime this branch exists for. Reviewer-measured: removing the branch entirely
        (using only the ``delta >= 0`` row unconditionally) gives residual 8.9e-11 on
        ``[[-1, 1e-6], [1e-6, 1]]`` (the delta < 0 case; the delta > 0 case happens to keep using the
        surviving row and stays accurate). The real code reaches ~1e-22 on both signs here, so 1e-10
        would slip right past the mutant -- verified directly: 8.9e-11 < 1e-10. Use the same 1e-13
        relative tolerance as the rest of this class, four orders of magnitude below the mutant's
        residual and comfortably above the real code's.
        """
        offd = 1e-6
        if delta_sign > 0:
            mat = np.array([[1.0, offd], [offd, -1.0]])
        else:
            mat = np.array([[-1.0, offd], [offd, 1.0]])
        diag = np.diagonal(mat).real
        assert np.sign((diag[0] - diag[1]) / 2.0) == delta_sign
        eigval, eigvec = eigenpair_2x2(jnp.asarray(mat))
        eigval = float(eigval)
        assert np.isfinite(eigval)
        assert eigval == pytest.approx(lowest(mat), rel=1e-13)
        assert rel_resid(mat, eigval, eigvec) < 1e-13


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
        mat = np.diag([5.0, 6.0, 1.0])
        eigval, eigvec = eigenpair_3x3(jnp.asarray(mat))
        eigval = float(eigval)
        assert eigval == pytest.approx(lowest(mat), abs=1e-13)
        assert rel_resid(mat, eigval, eigvec) < 1e-13
        assert np.linalg.norm(np.asarray(eigvec)) == pytest.approx(1.0)

    def test_identity_rank_zero(self):
        """Item I3 rank-0 fallback: every cross product vanishes for a multiple of the identity."""
        mat = np.eye(3)
        eigval, eigvec = eigenpair_3x3(jnp.asarray(mat))
        eigval = float(eigval)
        assert eigval == pytest.approx(1.0, abs=1e-13)
        assert rel_resid(mat, eigval, eigvec) < 1e-13
        assert np.linalg.norm(np.asarray(eigvec)) == pytest.approx(1.0)

    def test_degenerate_lowest_rank_one(self):
        """Item I3 rank-1 fallback: a degenerate lowest eigenvalue.

        Every cross product is numerical noise here; the null space is the orthogonal complement of
        the largest column, and any member of it is a valid eigenvector.

        This axis-aligned matrix does NOT actually reach the rank-1 fallback: for diag(1, 1, 7) the
        rank-0 constant candidate [1, 0, 0] happens to already be a true eigenvector and wins the
        residual selection outright, so this case alone leaves the rank-1 branch unexercised (see
        ``test_degenerate_lowest_rank_one_rotated`` below, which forces it). Kept because it still
        documents the axis-aligned case.
        """
        mat = np.diag([1.0, 1.0, 7.0])
        eigval, eigvec = eigenpair_3x3(jnp.asarray(mat))
        eigval = float(eigval)
        assert eigval == pytest.approx(1.0, abs=1e-13)
        assert rel_resid(mat, eigval, eigvec) < 1e-13
        assert np.linalg.norm(np.asarray(eigvec)) == pytest.approx(1.0)

    def test_degenerate_lowest_rank_one_rotated(self):
        """Item I3 rank-1 fallback, actually forced: no axis-aligned vector lies in the null space.

        ``diag(1, 1, 7)`` does not reach the rank-1 fallback in ``_nullvec_3x3`` (the rank-0 constant
        candidate wins by accident), so deleting the fallback entirely still leaves all other tests
        green. Rotating the degenerate matrix by a random orthogonal Q removes that accident: no
        candidate other than the rank-1 orthogonal-complement construction can land in the null
        space. Reviewer-measured over 200 such draws: worst relative residual 2.2e-01 with the
        fallback deleted, versus 6.5e-16 for the real code.
        """
        rng = np.random.default_rng(20260804)
        worst_residual = 0.0
        for _ in range(50):
            lo = rng.normal()
            hi = lo + abs(rng.normal()) + 0.5  # strictly above lo: lo stays the degenerate lowest
            axes = np.array([lo, lo, hi])
            q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
            mat = q @ np.diag(axes) @ q.T
            eigval, eigvec = eigenpair_3x3(jnp.asarray(mat))
            eigval = float(eigval)
            assert np.isfinite(eigval)
            assert eigval == pytest.approx(lo, abs=1e-13)
            worst_residual = max(worst_residual, rel_resid(mat, eigval, eigvec))
        assert worst_residual < 1e-13, f"worst relative residual {worst_residual:.2e}"

    def test_large_shift(self):
        """Item I1: at shift 1e9 the radicand under ``sqrt`` went negative and returned NaN.

        Not an exotic input -- this is the ordinary case for a physical Hamiltonian, which is rarely
        traceless, and is exactly what ``sqd.py`` feeds this solver.
        """
        mat = np.diag([1.0, 2.0, 3.0]) + 1e9 * np.eye(3)
        eigval, eigvec = eigenpair_3x3(jnp.asarray(mat))
        eigval = float(eigval)
        assert np.isfinite(eigval)
        assert eigval == pytest.approx(lowest(mat), rel=1e-13)
        assert rel_resid(mat, eigval, eigvec) < 1e-13

    @pytest.mark.parametrize("exponent", [-160, 150])
    def test_extreme_scale(self, exponent):
        """Item I2: ``c0`` is cubic in the entries, so unbalanced it overflows or underflows.

        Measured on the old kernel: relative error 7.8e-1 at 1e-160, NaN at 1e150.
        """
        mat = np.diag([1.0, 2.0, 3.0]) * 10.0**exponent
        eigval, eigvec = eigenpair_3x3(jnp.asarray(mat))
        eigval = float(eigval)
        assert np.isfinite(eigval)
        assert eigval == pytest.approx(lowest(mat), rel=1e-13)
        assert rel_resid(mat, eigval, eigvec) < 1e-13

    @pytest.mark.parametrize("complex_", [True, False])
    def test_random_sweep(self, complex_):
        """Aggregate accuracy over seeded random input. Supplements the targeted cases above.

        The old kernel produced 4066 NaNs in 20000 such matrices.
        """
        rng = np.random.default_rng(20260804)
        worst_eigval = worst_residual = 0.0
        for _ in range(2000):
            mat = herm(3, rng, complex_=complex_)
            eigval, eigvec = eigenpair_3x3(jnp.asarray(mat))
            eigval = float(eigval)
            assert np.isfinite(eigval)
            assert np.all(np.isfinite(np.asarray(eigvec)))
            scale = np.abs(mat).max()
            worst_eigval = max(worst_eigval, abs(eigval - lowest(mat)) / scale)
            worst_residual = max(worst_residual, rel_resid(mat, eigval, eigvec))
        assert worst_eigval < 1e-13, f"worst relative eigenvalue error {worst_eigval:.2e}"
        assert worst_residual < 1e-13, f"worst relative residual {worst_residual:.2e}"


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

    @pytest.mark.parametrize(
        "operator_dtype,xinit_dtype",
        [
            (jnp.float64, jnp.float64),
            (jnp.float32, jnp.float32),
            (jnp.float64, jnp.float32),
            (jnp.complex128, jnp.float32),
            (jnp.complex128, jnp.float64),
        ],
    )
    def test_dtype_combinations_solve(self, operator_dtype, xinit_dtype):
        rng = np.random.default_rng(20260804)
        mat = herm(60, rng, complex_=False)
        xinit = rng.normal(size=60)
        eigval, eigvec, _, converged = ground_locg(
            jnp.asarray(mat, dtype=operator_dtype), jnp.asarray(xinit, dtype=xinit_dtype)
        )
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

    def test_complex_operator_real_xinit_emits_no_warnings(self):
        """A complex128 operator with a real float64 ``xinit`` is a natural, previously-working call.

        Before the ``work_dtype`` promotion in ``_ground_locg_callable``, this path emitted
        ``FutureWarning: cannot safely cast complex128 to float64`` and ``ComplexWarning: Casting
        complex values discards the imaginary part`` from ``compute_sas``'s scatter, and silently
        discarded the imaginary part of the projected matrix -- a real bug, not just a noisy warning,
        since a genuinely complex Hamiltonian's off-diagonal imaginary parts would be dropped from the
        Rayleigh-Ritz step. The promotion added for the dtype-mismatch crash (see ``TestDtypes``'s
        class docstring) fixes this as a side effect. Guard both properties: no warnings, and a
        genuinely complex (non-Hermitian-as-real) operator still solves correctly.
        """
        rng = np.random.default_rng(20260804)
        mat = herm(30, rng, complex_=True)  # genuinely complex off-diagonal, not just complex dtype
        xinit = rng.normal(size=30)  # real float64, deliberately not complex
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            eigval, eigvec, _, converged = ground_locg(
                jnp.asarray(mat, dtype=jnp.complex128), jnp.asarray(xinit, dtype=jnp.float64)
            )
        assert bool(converged)
        assert float(eigval) == pytest.approx(lowest(mat), rel=1e-12)
        assert rel_resid(mat, float(eigval), np.asarray(eigvec)) < 1e-12

    def test_maxiter_zero_complex_operator_promotes_eigenvector_dtype(self):
        """Documents a visible consequence of the promotion: ``maxiter=0``'s eigenvector dtype.

        ``_ground_locg_callable`` returns the promoted ``xinit`` verbatim when ``maxiter=0``, so a
        complex128 operator with a real float64 ``xinit`` now gets back a complex128 eigenvector
        (all-real-valued, but complex dtype) instead of float64. This is the flip side of the fix
        above being deliberate rather than accidental: the imaginary part is no longer discarded
        anywhere in the pipeline, including trivially at ``maxiter=0``.
        """
        rng = np.random.default_rng(20260804)
        mat = herm(10, rng, complex_=True)
        xinit = rng.normal(size=10)
        _, eigvec, _, _ = ground_locg(
            jnp.asarray(mat, dtype=jnp.complex128), jnp.asarray(xinit, dtype=jnp.float64), maxiter=0
        )
        assert np.asarray(eigvec).dtype == np.complex128


class TestGroundLocg:
    """End-to-end LOBPCG solves.

    Coverage gap, recorded honestly rather than silently: items I5 (x/y orthogonality loss in
    ``body()``) and I6 (``_project_out``'s non-unit-norm postcondition, in the ``p_is_zero`` branch
    of ``body()`` specifically, as opposed to ``TestProjectOut`` which covers ``_project_out`` in
    isolation) have NO end-to-end mutation coverage in this suite. A mutation-testing pass confirmed
    that deleting *both* re-orthogonalization loops in ``body()`` (the I5 fix) leaves every test
    below passing -- the ``herm(200) + shift*I`` fixtures below converge in ~63 iterations and simply
    do not induce the x/y collapse the audit measured (worst observed |<x|y>| 1.7e-16 with the
    fix deleted, versus the audit's 1.0 at shift 1e9). The ``eigval > reference - 1e-8*scale``
    assertion in ``test_solves_and_converges`` below was, before this comment, believed to cover I5;
    it does not -- it was actually passing only because ``test_xinit_scale_invariance``'s old
    bit-exact-equality assertion happened to also break under the same mutant (see that test and
    Finding 2 of the 2026-08-04 review). Reproducing I5 for real needs a fixture that drives |s| -> 0
    near convergence for a large-shift operator; nobody has constructed one yet.
    """

    @pytest.mark.parametrize("shift", [0.0, 1e3, -1e3, 1e6, 1e9])
    @pytest.mark.parametrize("complex_", [True, False])
    def test_solves_and_converges(self, shift, complex_):
        """Three assertions per case, each keyed to a different audit item.

        Item I4 -- ``reltol`` used ``norm(Ax) - theta``, a cancellation measured going *negative*,
        which made the convergence test unsatisfiable: the solver never converged and always burned
        ``maxiter``. Fixing that sign alone was a 24-33x iteration reduction.

        The eigval-below-reference assertion below is *aimed at* item I5, the audit's most serious
        finding (loss of x/y orthogonality making the standard Rayleigh-Ritz step return a theta
        *beneath* the true minimum -- observed 6.0e8 against a true minimum of 1.0e9, impossible for
        a genuine Rayleigh quotient). It is kept as a sanity check but does NOT actually discriminate
        the I5 mutant on these fixtures -- see the class docstring above.
        """
        rng = np.random.default_rng(20260804)
        mat = herm(200, rng, complex_=complex_) + shift * np.eye(200)
        reference = lowest(mat)
        scale = np.abs(mat).max()
        xinit = rng.normal(size=200)
        if complex_:
            xinit = xinit + 0.0j
        eigval, _, _, converged = ground_locg(jnp.asarray(mat), jnp.asarray(xinit))
        eigval = float(eigval)

        assert bool(converged), "solver exhausted maxiter (item I4)"
        assert abs(eigval - reference) / scale < 1e-12
        assert eigval > reference - 1e-8 * scale, (
            f"theta {eigval!r} is below the true minimum {reference!r} (item I5)"
        )

    def test_maxiter_too_small_reports_not_converged(self):
        """Item A1: ``niter == maxiter`` is ambiguous, so the flag is the only usable signal.

        Measured on the old code with ``maxiter=5``: a confident-looking eigenvalue whose true
        relative residual was 1.1e-1, and no way for the caller to tell.
        """
        rng = np.random.default_rng(20260804)
        mat = herm(60, rng, complex_=False)
        eigval, _, niter, converged = ground_locg(
            jnp.asarray(mat), jnp.asarray(rng.normal(size=60)), maxiter=5
        )
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
        """``xinit`` is normalized on entry, so its eigenvalue is insensitive to ``|xinit|``.

        Not bit-exact: an earlier version of this test asserted ``==`` on the floats, which passes
        only by coincidence. Measured directly: at a fixed seed, scale 1e8 happens to agree bit-for-
        bit with scale 1 (both exact powers of two), but 1e7, 1e9, 3.0, 7.0, and 1e-8 all differ by
        3.5e-15 to 7.1e-15 relative to max|mat|. Sweeping 13 seeds x 5 matrix sizes (10-200) x 8
        scales (1e-8 to 1e9) puts the worst observed relative spread at 4.9e-15. This assertion uses
        1e-12, about 200x that margin, so it is insensitive to JAX/XLA version or backend changes
        while still catching a genuine scale-dependence regression.
        """
        rng = np.random.default_rng(20260804)
        mat = jnp.asarray(herm(60, rng, complex_=False))
        xinit = jnp.asarray(rng.normal(size=60))
        scale_of_mat = float(jnp.abs(mat).max())
        val_1 = float(ground_locg(mat, xinit)[0])
        val_scaled = float(ground_locg(mat, xinit * 1e8)[0])
        assert abs(val_scaled - val_1) / scale_of_mat < 1e-12

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
        result = ground_locg(
            jnp.asarray(mat), jnp.asarray(rng.normal(size=40)), maxiter=maxiter, debug=True
        )
        assert len(result) == 5
        diagnostics = result[4]
        for key in ("x", "y", "r", "theta", "rho", "kappa", "sas", "reltol", "converged"):
            assert np.shape(diagnostics[key])[0] == maxiter + 2, f"{key} row count"


class TestZeroResidualAfterSeedStep:
    """``body_iter1`` must guard a zeroed post-seed residual the same way ``body()`` does.

    ``body()`` (the main iteration) masks its projected matrix's search-direction diagonal and folds
    a zeroed direction into ``converged`` -- see items I6/I7 in ``docs/locg.md``. ``body_iter1`` (the
    one-shot seed step run before the main loop) had no equivalent guard: when the residual after
    ``body_iter0`` is exactly zero, xinit is already an eigenvector and the ``{x, p}`` projected
    matrix has a vanishing row/column 1, which ``eigenpair_2x2`` would otherwise resolve into
    selecting the null direction and collapsing theta towards 0 rather than reporting the correct
    Rayleigh quotient.

    This is not a contrived corner case: ``sqd.py`` seeds ``vinit`` as an exact one-hot vector at the
    minimum-diagonal index whenever the Hamiltonian is diagonal (``jnp.all(hamiltonian.x[0] == 0)``),
    which is exactly this input shape -- a one-hot vector against a diagonal operator is an exact
    eigenvector, giving an exactly-zero residual after the seed step.

    These fixtures also give end-to-end coverage of items I6 and I7 (the zeroed-search-direction
    guard and convergence-on-exhausted-search-space), which no other fixture in this suite reaches:
    ``p_is_zero``/the residual-is-zero condition is never True in any of the random or shifted-``herm``
    fixtures elsewhere in this file, since those all use generic, non-eigenvector initial guesses.
    """

    def test_one_by_one(self):
        eigval, eigvec, niter, converged = ground_locg(jnp.array([[3.0]]), jnp.array([1.0]))
        assert bool(converged)
        assert int(niter) == 0
        assert float(eigval) == pytest.approx(3.0, abs=1e-13)
        assert np.asarray(eigvec) == pytest.approx([1.0])

    def test_one_by_one_large_magnitude(self):
        """The large-magnitude case: theta must be the operator's value, not a value near 0."""
        eigval, eigvec, niter, converged = ground_locg(jnp.array([[1e9]]), jnp.array([1.0]))
        assert bool(converged)
        assert int(niter) == 0
        assert float(eigval) == pytest.approx(1e9, rel=1e-13)
        assert np.asarray(eigvec) == pytest.approx([1.0])

    @pytest.mark.parametrize("index", [0, 5])
    def test_diagonal_operator_one_hot_xinit(self, index):
        """A one-hot ``xinit`` against a diagonal operator is an exact eigenvector at any index.

        Index 0 and an interior index (5) are both covered: nothing in ``body_iter1`` singles out
        position 0, but the audit's own reproduction used index 0, so an interior index is added to
        make sure the guard is not accidentally position-dependent.
        """
        mat = jnp.diag(jnp.arange(1.0, 61.0))
        eigval, eigvec, niter, converged = ground_locg(mat, jnp.asarray(index))
        assert bool(converged)
        assert int(niter) == 0
        assert float(eigval) == pytest.approx(float(index) + 1.0, abs=1e-13)
        expected = np.zeros(60)
        expected[index] = 1.0
        assert np.asarray(eigvec) == pytest.approx(expected)
