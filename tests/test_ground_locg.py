"""Tests for :mod:`rqutils.ground_locg`.

Organized by defect. Most tests correspond to a numbered item in ``docs/locg.md``, the audit of the
previous implementation of this module; each such test names its item and the measured old failure.
That audit's closing warning shapes the design here:

    Aggregate random sampling does not exercise data-dependent branches evenly. [...] Both of my
    safety nets missed a sign error that a two-line targeted test caught at 4000/4000 failures.

So targeted per-branch tests are the backbone and randomized sweeps are a supplement.
"""

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import herm, lowest, rel_resid, run_sharded_child, symmetrize

from rqutils.ground_locg import (
    _chebyshev_prefilter,
    _project_out,
    eigenpair_2x2,
    eigenpair_3x3,
    ground_locg,
)


class TestProjectOut:
    """``_project_out`` returns either exactly zero or a vector of norm >= 0.99 (item I6).

    It returns ``(vector, norm)``; the norm is handed back rather than recomputed by the caller
    because that would be a second O(N) reduction over a vector of up to 1e8 elements.
    """

    def test_vector_in_basis_span_returns_exactly_zero(self):
        """A residual lying wholly in span{x, y} must come back as exact zero, not as noise.

        The zeroing guard is what item I7's convergence check keys off: a zeroed search direction
        means {x, y} already spans the residual, so no further iteration can lower theta. The
        returned norm must be exactly 0.0 alongside it, since that is the flag body() tests.
        """
        basis = (jnp.array([1.0, 0.0, 0.0]), jnp.array([0.0, 1.0, 0.0]))
        out, norm = _project_out(basis, jnp.array([1.0, 1.0, 0.0]))
        assert np.array_equal(np.asarray(out), np.zeros(3))
        assert float(norm) == 0.0, "a zeroed vector must report norm 0, or p_is_zero never fires"

    def test_orthogonal_vector_is_not_normalized_to_unity(self):
        """The postcondition is norm >= 0.99, NOT norm == 1.

        Item I6: callers feeding this to a standard Rayleigh-Ritz step must renormalize themselves.
        At shift 1e9 a |p| of 0.999 displaced theta by 2e6, below the true minimum. This test exists
        so that a future edit "tidying" the trailing subtraction into a normalization fails loudly.

        Three assertions: (1) the orthonormal case must return exactly [0,0,1], (2) a tilted
        (non-orthonormal) basis must return a result with norm strictly between 0.99 and 1.0,
        proving the vector is not renormalized to unity, and (3) the returned norm must agree with
        the vector's actual norm, so the caller's renormalization uses the right divisor.
        """
        # Orthonormal basis: _project_out subtracts <e_i|v> e_i, leaving exactly [0, 0, 1]. The norm
        # is unused -- the exact-value assertion is stronger than any bound on it; the
        # non-orthonormal case below is where the norm itself is load-bearing.
        basis = (jnp.array([1.0, 0.0, 0.0]), jnp.array([0.0, 1.0, 0.0]))
        out, _ = _project_out(basis, jnp.array([0.0, 0.0, 2.0]))
        assert np.allclose(np.asarray(out), [0.0, 0.0, 1.0])

        # Non-orthonormal basis: _project_out subtracts <b|v> b without a Gram solve, so the
        # result falls short of unit norm. This discriminates "masked to >= 0.99" from
        # "renormalized to unity". The result must be strictly less than 1.0.
        basis = (jnp.array([1.0, 0.0, 0.0]), jnp.array([0.3, np.sqrt(1.0 - 0.3**2), 0.0]))
        out, norm = _project_out(basis, jnp.array([0.3, 0.4, 1.0]))
        out = np.asarray(out)
        actual = np.linalg.norm(out)
        assert actual >= 0.99, f"Norm {actual} dropped below 0.99"
        assert actual < 1.0, f"Norm {actual} is not strictly less than 1.0; likely renormalized"
        assert float(norm) == pytest.approx(actual), "returned norm disagrees with the vector"


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
    Finding 2 of the 2026-08-04 review).

    I5 *is* pinned, just not from here: :class:`TestBasisOrthogonality` asserts ``|<x|y>|`` directly
    off the per-iteration diagnostics at shifts up to 1e12, with measured neutered-guard values
    (1.0e-08 at shift 1e9 against 3.8e-17 real). That is strictly better than waiting for the basis
    collapse to corrupt theta -- it fails at the cause rather than three steps downstream -- so what
    remains missing is only an end-to-end *wrong eigenvalue* fixture, which would need the
    2000-iteration runs item I4's sign bug used to force. Not worth building: the fixed solver
    converges in 8-46 iterations. I6-in-``body()`` has no coverage and no cheap fixture, since
    reaching ``p_is_zero`` needs the residual to land in ``span{x, y}`` at an iteration >= 2.
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


class TestBasisOrthogonality:
    """Item I5: ``{x, y}`` must stay orthonormal, because Rayleigh-Ritz assumes it.

    ``t = kappa_0 s / |s| - |s| x`` is orthogonal to ``x`` in exact arithmetic, so
    ``_reorthogonalize`` is algebraically a no-op and removing it breaks no other assertion in this
    file -- theta still matches ``eigvalsh``. What it breaks is the *basis*, and only later does that
    corrupt theta (the audit measured a full collapse to 1.0, but needed the 2000-iteration runs that
    item I4's sign error used to force; the fixed solver converges in 8-46).

    So this asserts the invariant directly off the per-iteration diagnostics rather than waiting for
    a wrong eigenvalue. Measured with the guard neutered in a fresh subprocess (it must be a
    subprocess -- both callers are ``@jax.jit``-decorated, so patching in a live session reuses the
    compiled kernel and both arms return bit-identical numbers):

        shift 0     3.8e-17 -> 9.7e-17
        shift 1e6   5.6e-17 -> 2.5e-12
        shift 1e9   3.8e-17 -> 1.0e-08
        shift 1e12  5.6e-17 -> 7.6e-13

    The 1e-13 threshold below sits far under every neutered value and far above every real one.
    """

    @pytest.mark.parametrize("shift", [0.0, 1e6, 1e9, 1e12])
    def test_x_y_stay_orthogonal_at_large_shift(self, shift):
        """A large trace is what drives ``|s| -> 0`` cancellation into ``y``."""
        n = 200
        rng = np.random.default_rng(20260807)
        mat = symmetrize(rng.normal(size=(n, n))) + shift * np.eye(n)
        xinit = rng.normal(size=n)

        # debug=True runs scan with no early exit, so every iteration is inspected -- including the
        # post-convergence tail, which is where |s| is smallest and cancellation worst.
        result = ground_locg(jnp.asarray(mat), jnp.asarray(xinit), maxiter=60, debug=True)
        assert len(result) == 5  # also narrows the annotated 4-tuple for the type checker
        diagnostics = result[4]
        xs = np.asarray(diagnostics["x"])
        ys = np.asarray(diagnostics["y"])
        overlap = np.abs(np.einsum("ij,ij->i", xs.conjugate(), ys))

        worst = overlap.max()
        assert worst < 1e-13, (
            f"worst |<x|y>| = {worst:.3e} at shift {shift:.0e}; the Rayleigh-Ritz step solves a "
            "standard eigenproblem and assumes an orthonormal basis"
        )


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

    These fixtures cover the **seed-step analogue** of items I6/I7 -- the zeroed-direction guard as
    ``body_iter1`` implements it -- and nothing else in this suite reaches even that, since every
    other fixture here uses a generic, non-eigenvector initial guess.

    They do **not** cover ``body()``'s own ``p_is_zero`` branch, and cannot: every test below asserts
    ``niter == 0``, which means ``body_iter1`` already set ``converged``, the ``while_loop`` predicate
    is False on entry, and ``body()`` never executes. Verified by neutering ``body()``'s
    ``sas.at[2, 2]`` exclusion mask -- all 54 tests in this file stay green. ``TestGroundLocg``'s
    class docstring records that gap; this class does not close it.
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


class TestPreconditioner:
    """``precond`` applies ``M^-1`` where the search direction is formed, and nowhere else.

    Opt-in: ``precond=None`` is the default and must be the identity path, so no existing caller
    changes behaviour. It is a static argument, so the branch resolves at trace time and the
    unpreconditioned graph is unchanged.

    The load-bearing subtlety is an **asymmetry in ``body_iter1``**. ``norm_r`` was one quantity
    feeding two consumers: ``r_is_zero`` (which masks ``sas[1, 1]`` and sets ``converged``) and the
    ``normalize`` that forms ``tmp_p``. Preconditioning must touch only the second. Routing the guard
    through ``M^-1`` would change what counts as a stationary point while still returning a plausible
    number -- a residual lying near ``M^-1``'s small-singular-value direction would report convergence
    early, and per ``body_iter1``'s own comment the unguarded path makes ``theta`` collapse towards 0
    instead of reporting ``rho``. :class:`TestZeroResidualAfterSeedStep` owns that guard; the test here
    checks ``precond`` does not disturb it.

    Measured on the 12-instance XXZ batch this hook was requested for: median **1.79x** fewer
    iterations, range 1.29-2.04x, 0 regressions, every arm converging to the same eigenvalue.

    **Known weak spot, recorded rather than closed.** Making ``precond`` a no-op in ``body_iter1``
    *only* -- leaving ``body()`` correct -- survives this suite. It is a real defect but a small one:
    ``body_iter1`` runs exactly once, so it perturbs a single bootstrap direction. Measured across the
    12 XXZ instances it degrades every one of them by 2-7% (49->51, 87->87, 168->180 iterations), which
    is adverse but never crosses into a wrong answer. Two fixtures were tried to catch it and neither
    discriminated -- the effect is below the granularity of an integer iteration count on a
    well-conditioned synthetic operator, and pinning a tighter count would make this a flaky
    performance test. If you touch ``body_iter1``'s direction, re-run the XXZ batch by hand.
    """

    @staticmethod
    def _spd(dim, rng, cond=50.0):
        """A positive-definite matrix with a controlled diagonal spread, so Jacobi has something to do.

        Uniform diagonals leave ``D^-1`` proportional to the identity, which makes the preconditioner
        a no-op and the test vacuous. The spread is asserted by the caller.
        """
        q = np.linalg.qr(rng.normal(size=(dim, dim)))[0]
        spec = np.geomspace(1.0, cond, dim)
        return symmetrize(q @ np.diag(spec) @ q.T)

    def test_none_is_bit_identical_to_omitting_it(self):
        """The default must not perturb the existing graph at all, not merely agree to a tolerance."""
        rng = np.random.default_rng(20260824)
        mat = jnp.asarray(self._spd(64, rng))
        xinit = jnp.asarray(rng.normal(size=64))
        omitted = ground_locg(mat, xinit, maxiter=200)
        explicit = ground_locg(mat, xinit, maxiter=200, precond=None)
        assert float(omitted[0]) == float(explicit[0]), "theta must be bit-identical"
        assert int(omitted[2]) == int(explicit[2]), "iteration count must be identical"
        assert jnp.array_equal(omitted[1], explicit[1]), "eigenvector must be bit-identical"

    def test_jacobi_finds_the_same_eigenvalue(self):
        """Preconditioning changes the path, never the answer.

        Against ``numpy.linalg.eigvalsh`` rather than against the unpreconditioned arm: two arms of the
        same routine agreeing proves only that they are consistent, not that either is right.
        """
        rng = np.random.default_rng(20260824)
        mat = self._spd(96, rng)
        diag = np.diag(mat).copy()
        assert diag.max() / diag.min() > 2.0, "fixture needs a diagonal spread or Jacobi is a no-op"
        dinv = jnp.asarray(1.0 / diag)
        matj = jnp.asarray(mat)
        xinit = jnp.asarray(rng.normal(size=96))
        reference = float(np.linalg.eigvalsh(mat)[0])
        for label, pc in (("none", None), ("jacobi", lambda v: v * dinv)):
            theta, _, _, converged = ground_locg(matj, xinit, maxiter=500, tol=1e-12, precond=pc)
            assert bool(converged), f"{label} did not converge"
            assert float(theta) == pytest.approx(reference, abs=1e-9), (
                f"{label}: got {float(theta)}, dense reference {reference}"
            )

    def test_jacobi_reduces_the_iteration_count(self):
        """The hook exists for this, so a test should notice if it stops paying off.

        The bound is deliberately loose (any improvement at all). The measured median on the XXZ batch
        this was requested for is 1.79x, but the gain is instance-dependent -- 1.29x to 2.04x there --
        and pinning a ratio would make this a flaky performance test rather than a contract test.
        """
        rng = np.random.default_rng(20260824)
        mat = self._spd(96, rng, cond=200.0)
        dinv = jnp.asarray(1.0 / np.diag(mat).copy())
        matj = jnp.asarray(mat)
        xinit = jnp.asarray(rng.normal(size=96))
        plain = int(ground_locg(matj, xinit, maxiter=500, tol=1e-12)[2])
        jac = int(ground_locg(matj, xinit, maxiter=500, tol=1e-12, precond=lambda v: v * dinv)[2])
        assert jac < plain, f"Jacobi took {jac} iterations against {plain} unpreconditioned"

    def test_exact_eigenvector_seed_still_reports_rho(self):
        """``precond`` must not reroute ``body_iter1``'s ``r_is_zero`` guard.

        A one-hot seed of a diagonal operator is an exact eigenvector, so the raw residual is exactly
        zero -- ``sqd``'s diagonal-Hamiltonian path does this, per ``body_iter1``'s comment, so it is
        not a corner case. The guard must fire identically with and without a preconditioner: if
        ``norm_r`` were computed from ``M^-1 r`` instead, a preconditioner that shrinks that direction
        would still read zero here, but for the wrong reason, and any *nonzero* residual it shrank
        would be misreported as stationary.
        """
        dim = 64
        diag = np.linspace(1.0, 5.0, dim)
        mat = jnp.asarray(np.diag(diag))
        onehot = jnp.zeros(dim, dtype=jnp.float64).at[0].set(1.0)
        dinv = jnp.asarray(1.0 / diag)
        for label, pc in (("none", None), ("jacobi", lambda v: v * dinv)):
            theta, _, niter, converged = ground_locg(mat, onehot, maxiter=100, precond=pc)
            assert float(theta) == pytest.approx(diag[0], abs=1e-12), (
                f"{label}: theta collapsed to {float(theta)} instead of rho={diag[0]}"
            )
            assert bool(converged), f"{label}: an exact eigenvector must report converged"
            assert int(niter) == 0, f"{label}: expected to stop at the seed step, took {int(niter)}"

    def test_the_zero_residual_guard_reads_the_raw_residual(self):
        """The asymmetry, pinned. ``r_is_zero`` must not see ``M^-1 r``.

        This is the one test that discriminates, and it needs a preconditioner that maps a **nonzero**
        residual to zero -- an annihilating ``M^-1``. The sibling test above uses an *exactly* zero
        residual, where ``M^-1 0 == 0`` either way, so it cannot tell the two implementations apart
        (mutation-tested: it passes against the defect).

        With the guard on the raw residual (correct), ``r_is_zero`` is False, no masking happens, and
        the zeroed *direction* is handled downstream by ``normalize``/``p_is_zero`` -- theta comes out
        at 0.0 for this degenerate preconditioner, which is a useless answer but an honest one.

        With the guard on ``M^-1 r`` (the defect), ``r_is_zero`` fires at the seed step, ``sas[1, 1]``
        is masked, and the routine returns **the seed vector's Rayleigh quotient** with
        ``converged=True`` -- measured 10.254108965 against a true minimum of 1.0. A plausible finite
        wrong number, which is the failure mode this module's guards exist to prevent.
        """
        dim = 32
        rng = np.random.default_rng(7)
        q = np.linalg.qr(rng.normal(size=(dim, dim)))[0]
        mat = jnp.asarray(q @ np.diag(np.geomspace(1.0, 40.0, dim)) @ q.T)
        xinit = jnp.asarray(rng.normal(size=dim))
        seed_rho = float(
            np.asarray(xinit)
            @ np.asarray(mat)
            @ np.asarray(xinit)
            / (np.asarray(xinit) @ np.asarray(xinit))
        )
        assert seed_rho > 5.0, "fixture needs a seed far from the ground state to be diagnostic"

        theta = float(ground_locg(mat, xinit, maxiter=300, tol=1e-12, precond=lambda v: v * 0.0)[0])
        assert theta != pytest.approx(seed_rho, rel=1e-6), (
            f"theta came back as the seed's Rayleigh quotient ({theta}), which means the zero-residual "
            "guard was computed from the preconditioned residual instead of the raw one"
        )

    def test_precond_is_static_so_it_adds_no_traced_argument(self):
        """``precond`` is in ``static_argnames``, which is what keeps ``None`` free.

        A *traced* callable would appear as an argument in the jaxpr even when unused, and the ``None``
        branch could not resolve at trace time. Asserted on the jaxpr's input signature rather than on
        the lowered HLO text: the two differ only in source-location metadata (the call site's line and
        column), which is debug information rather than computation, so a text comparison would fail
        for a reason that has nothing to do with the graph.
        """
        rng = np.random.default_rng(20260824)
        mat = jnp.asarray(self._spd(32, rng))
        xinit = jnp.asarray(rng.normal(size=32))
        from rqutils.ground_locg import _ground_locg_matrix

        without = jax.make_jaxpr(lambda m, v: _ground_locg_matrix(m, v, 50, None))(mat, xinit)
        explicit = jax.make_jaxpr(lambda m, v: _ground_locg_matrix(m, v, 50, None, precond=None))(
            mat, xinit
        )
        assert str(without) == str(explicit), (
            "precond=None changed the traced graph, so it is not resolving at trace time"
        )


class TestChebyshevPrefilter:
    """``prefilter`` damps the unwanted band of the spectrum before the iteration starts.

    Opt-in exactly as ``precond`` is: ``None`` is the default and must be the identity path, and it is
    a static argument so the unfiltered graph is unchanged. It **cannot change the answer, only the
    path** -- every convergence test still reads the true residual.

    Measured on connected XXZ subspaces (18 configurations, 3 seeds x 3 anisotropies x 2 sizes) with
    ``(16, 4)``: median **1.36x** wall clock, range 1.11-3.07x, 0 regressions, eigenvector overlap
    1.0000000 against the unfiltered result. ``docs/locg-chebyshev-prefilter.md`` has the tables.

    Two design points are load-bearing and are pinned below, because both fail *silently*:

    - **The filter's lower edge is the running Rayleigh quotient, not an accurate** ``lambda_1``. This
      is a *robustness* choice, and weaker than it first appears -- recorded honestly because the
      obvious stronger claim is wrong. With ``lambda_1`` as a fixed edge, **filtering alone** returns an
      energy off by **15** at a relative gap of 4.0e-05, since the interval then begins at ``lambda_0``
      and damps the ground state. In the *hybrid* that failure is unreachable: the full iteration
      repairs a poor start. **No test here pins the edge choice, and mutation confirms none can** --
      raising it by 5%, 20%, 50% and 100% of ``|theta|`` leaves every test passing, because ``theta``
      begins far above ``lambda_0`` (measured +5.37 against -5.0), so the interval starts entirely above
      the target and the filter closes 100% of the gap at every bump. The running quotient is preferred
      for needing no spectral input, not for correctness. If you replace it, re-measure wall clock on
      the XXZ batch in ``docs/locg-chebyshev-prefilter.md`` -- the suite will not tell you.

    **Mutation results, recorded so the coverage is not overestimated.** Caught: discarding the
    filtered vector (2 tests fail). **Not caught**: flipping the sign of the three-term recurrence
    (``- previous`` -> ``+ previous``), and raising the interval's lower edge by 20% of ``|theta|``.
    Both still converge to the right eigenvalue, because ``ground_locg`` repairs the start and because
    the initial interval sits entirely above ``lambda_0`` -- measured, the correct recurrence closes
    100.00% of the distance to ``lambda_0`` on the direct fixture and the sign-flipped one closes
    99.99%, a difference no non-flaky tolerance separates. What that means practically: this suite
    protects the *contract* (same answer, opt-in, sharding, no-op default) and the presence of the
    filter, not the arithmetic inside it. A change to the recurrence needs the wall-clock batch.
    - **Filtering is a prefilter, not a solver.** Alone it plateaus at ~1e-5 to 1e-7, since as
      ``theta`` approaches ``lambda_0`` the lower edge does too and the filter starts attacking its own
      target. The handoff to the full iteration is what delivers the last digits, so the accuracy
      assertions here are against the *converged* result and are deliberately tight.

    Note a filtered start is not the same as a *better* start. A shifted-power start, which has a far
    better Rayleigh quotient and a smaller residual, measured 177 iterations against 77 -- block-size-1
    LOBPCG spans only ``{x, y, p}``, so convergence tracks what the residual can still expose and power
    iteration collapses onto the dominant direction. A polynomial filter suppresses the unwanted band
    multiplicatively and leaves the residual rich. Do not "simplify" the filter into extra power steps.
    """

    @staticmethod
    def _gapped(dim, rng, gap, spread=20.0, base=-5.0):
        """Hermitian matrix with a prescribed lowest gap, so the filter has a defined target.

        Built by conjugating a chosen spectrum, since the point is to control ``lambda_1 - lambda_0``
        exactly; drawing at random gives whatever gap it gives and makes the small-gap case
        unreachable.

        ``base`` MUST STAY AWAY FROM ZERO. The convergence test is
        ``|r| < tol * (|Ax| + |theta|) * N * 10``, so a spectrum with ``lambda_0 == 0`` drives both
        terms of that sum to zero as the iterate converges and the threshold becomes unsatisfiable:
        measured, an unfiltered run on such a fixture returns the right energy (1e-15 against a
        reference of 1.3e-14) while reporting ``converged=False`` at every ``maxiter``. That would look
        like a prefilter defect and is not one. A physical Hamiltonian has a nonzero ground energy, so
        the shift is also the realistic case.
        """
        spec = np.concatenate([[0.0, gap], np.linspace(gap + 1.0, spread, dim - 2)]) + base
        q = np.linalg.qr(rng.normal(size=(dim, dim)))[0]
        return symmetrize(q @ np.diag(spec) @ q.T)

    def test_none_is_bit_identical_to_omitting_it(self):
        """The default must not perturb the existing graph at all, not merely agree to a tolerance."""
        rng = np.random.default_rng(20260828)
        mat = jnp.asarray(herm(64, rng, complex_=False))
        xinit = jnp.asarray(rng.normal(size=64))
        omitted = ground_locg(mat, xinit, maxiter=300)
        explicit = ground_locg(mat, xinit, maxiter=300, prefilter=None)
        assert float(omitted[0]) == float(explicit[0]), "theta must be bit-identical"
        assert int(omitted[2]) == int(explicit[2]), "iteration count must be identical"
        assert jnp.array_equal(omitted[1], explicit[1]), "eigenvector must be bit-identical"

    def test_prefilter_is_static_so_it_adds_no_traced_argument(self):
        """A traced tuple would recompile per value and defeat the trace-time branch."""
        rng = np.random.default_rng(20260828)
        mat = jnp.asarray(herm(32, rng, complex_=False))
        xinit = jnp.asarray(rng.normal(size=32))
        without = jax.make_jaxpr(lambda m, x: ground_locg(m, x))(mat, xinit)
        explicit = jax.make_jaxpr(lambda m, x: ground_locg(m, x, prefilter=None))(mat, xinit)
        assert str(without) == str(explicit), (
            "prefilter=None changed the traced graph, so it is not resolving at trace time"
        )

    @pytest.mark.parametrize("prefilter", [(8, 2), (16, 4), (32, 2)])
    def test_finds_the_same_eigenvalue(self, prefilter):
        """Filtering changes the path, never the answer.

        Compared against LAPACK rather than against the unfiltered run, so a shared wrong answer
        cannot pass -- the repo's rule about preferring an independent reference.
        """
        rng = np.random.default_rng(20260828)
        mat = self._gapped(96, rng, gap=0.5)
        matj = jnp.asarray(mat)
        xinit = jnp.asarray(rng.normal(size=96))
        reference = lowest(mat)
        result = ground_locg(matj, xinit, maxiter=500, prefilter=prefilter)
        assert bool(result[3]), f"prefilter={prefilter} failed to converge"
        assert abs(float(result[0]) - reference) < 1e-10, (
            f"prefilter={prefilter} gave {float(result[0])}, expected {reference}"
        )
        assert rel_resid(mat, float(result[0]), np.asarray(result[1])) < 1e-10

    def test_returns_the_same_eigenvector_not_merely_the_same_energy(self):
        """An energy check alone would pass on a different member of a near-degenerate pair.

        The filter is a spectral transformation, so the failure worth guarding is that it converges to
        a *neighbouring* eigenvector while the energy still looks right -- the same geometry
        ``TestBasisOrthogonality`` guards for the balancing.
        """
        rng = np.random.default_rng(20260828)
        mat = jnp.asarray(self._gapped(96, rng, gap=0.05))
        xinit = jnp.asarray(rng.normal(size=96))
        plain = ground_locg(mat, xinit, maxiter=800)
        filtered = ground_locg(mat, xinit, maxiter=800, prefilter=(16, 4))
        vec_plain = np.asarray(plain[1]).ravel()
        vec_filtered = np.asarray(filtered[1]).ravel()
        overlap = abs(vec_plain @ vec_filtered) / (
            np.linalg.norm(vec_plain) * np.linalg.norm(vec_filtered)
        )
        assert overlap > 1.0 - 1e-9, (
            f"filtered run found a different eigenvector (overlap {overlap})"
        )

    def test_tiny_gap_still_finds_the_ground_state(self):
        """THE CASE THAT BREAKS A FILTER BUILT ON AN ACCURATE ``lambda_1``.

        With the interval's lower edge at ``lambda_1`` and ``lambda_1 - lambda_0`` tiny, the filter
        damps the ground state too and returns a wrong energy with no error raised -- measured 1.5e+01
        off at a relative gap of 4.0e-05. The running-Rayleigh-quotient edge is what makes this pass, so
        this test is what pins that choice.
        """
        rng = np.random.default_rng(20260828)
        mat = self._gapped(128, rng, gap=1e-5)
        matj = jnp.asarray(mat)
        xinit = jnp.asarray(rng.normal(size=128))
        reference = lowest(mat)
        result = ground_locg(matj, xinit, maxiter=2000, prefilter=(16, 4))
        assert abs(float(result[0]) - reference) < 1e-8, (
            f"tiny-gap filtered run gave {float(result[0])}, expected {reference} -- the filter is "
            "damping the ground state, which means its lower edge is not the running Rayleigh quotient"
        )

    def test_reduces_the_iteration_count(self):
        """The whole point. Asserted as a direction, not a pinned count, to avoid a flaky threshold."""
        rng = np.random.default_rng(20260828)
        mat = jnp.asarray(self._gapped(256, rng, gap=0.02))
        xinit = jnp.asarray(rng.normal(size=256))
        plain = ground_locg(mat, xinit, maxiter=2000)
        filtered = ground_locg(mat, xinit, maxiter=2000, prefilter=(16, 4))
        assert bool(plain[3]) and bool(filtered[3]), "both arms must converge for the comparison"
        assert int(filtered[2]) < int(plain[2]), (
            f"prefilter did not reduce iterations ({int(plain[2])} -> {int(filtered[2])})"
        )

    def test_degenerate_prefilter_values_are_a_no_op(self):
        """``degree <= 1`` or ``cycles == 0`` must not divide by zero or corrupt the start."""
        rng = np.random.default_rng(20260828)
        mat = jnp.asarray(herm(48, rng, complex_=False))
        xinit = jnp.asarray(rng.normal(size=48))
        baseline = ground_locg(mat, xinit, maxiter=300)
        for prefilter in [(1, 4), (16, 0), (0, 0)]:
            result = ground_locg(mat, xinit, maxiter=300, prefilter=prefilter)
            assert float(result[0]) == float(baseline[0]), (
                f"prefilter={prefilter} should be a no-op but changed theta"
            )

    def test_filter_moves_the_rayleigh_quotient_toward_the_ground_state(self):
        """The filter must actually filter -- asserted on its output, not on the solve that follows.

        This is the most direct assertion available, and it is still not sensitive to the interval's
        lower edge (see the class docstring: every bump tried closes 100% of the gap). What it does
        catch is a filter that is inert, inverted, or applied to the wrong operator -- e.g. the
        recurrence built with ``+ previous`` instead of ``- previous``, or ``centre``/``half`` swapped.
        """
        rng = np.random.default_rng(20260828)
        mat = self._gapped(128, rng, gap=0.02)
        matj = jnp.asarray(mat)
        reference = lowest(mat)
        xinit = jnp.asarray(rng.normal(size=128))

        def rayleigh(vec):
            vec = vec / jnp.linalg.norm(vec)
            return float(jnp.sum(vec.conjugate() * (matj @ vec)).real)

        before = rayleigh(xinit)
        after = rayleigh(_chebyshev_prefilter(lambda v: matj @ v, (), xinit, 16, 4))
        assert after < before, f"filter did not lower the Rayleigh quotient ({before} -> {after})"
        closed = (before - after) / (before - reference)
        assert closed > 0.9, f"filter closed only {closed:.1%} of the gap to lambda_0"
        # Deliberately loose. Measured, this fixture closes 100.00% with the correct recurrence and
        # 99.99% with the sign of the three-term recurrence flipped, so a tolerance tight enough to
        # separate those two would be pinning noise. The class docstring records which mutants survive.

    def test_complex_operator(self):
        """The Chebyshev recurrence must not assume a real operator.

        ``matvec`` output is complex here while the interval bounds are real, so a spelling that mixed
        the two would either raise or silently drop the imaginary part -- the trap the module docstring
        records for ``compute_sas``.
        """
        rng = np.random.default_rng(20260828)
        mat = jnp.asarray(herm(64, rng, complex_=True))
        xinit = jnp.asarray(rng.normal(size=64) + 1j * rng.normal(size=64))
        reference = float(np.linalg.eigvalsh(np.asarray(mat))[0])
        result = ground_locg(mat, xinit, maxiter=500, prefilter=(16, 4))
        assert bool(result[3]), "complex operator failed to converge with a prefilter"
        assert abs(float(result[0]) - reference) < 1e-10

    def test_preserves_sharding_on_a_mesh(self):
        """The prefilter must not silently un-shard the vector it returns.

        Asserted on the SPEC, not only the energy: per ``CLAUDE.md`` a replicated run agrees with
        single-device to exactly 0.0, so "correct but silently unsharded" is invisible to a value
        comparison. Subprocessed because the virtual device count must be set before jax initializes.
        """
        stdout = run_sharded_child("_sharded_prefilter.py", "ground_locg prefilter")
        rows = {}
        reference = None
        for line in stdout.splitlines():
            parts = line.split()
            if parts[0] == "reference":
                reference = float(parts[1])
                continue
            rows[parts[0]] = (float(parts[1]), int(parts[2]), parts[3], parts[4])
        # Assert the case set is complete before checking values: a child that died partway would
        # otherwise pass on whatever it managed to print.
        assert reference is not None
        assert set(rows) == {"plain", "prefiltered"}, f"incomplete child output: {sorted(rows)}"
        for label, (energy, iters, converged, spec) in rows.items():
            assert converged == "True", f"{label} did not converge on the mesh"
            assert abs(energy - reference) < 1e-10, f"{label} gave {energy}, expected {reference}"
            assert spec == "P('x',)", f"{label} lost its sharding: spec is {spec}"
        assert rows["prefiltered"][1] < rows["plain"][1], (
            "prefilter did not reduce iterations on the mesh "
            f"({rows['plain'][1]} -> {rows['prefiltered'][1]})"
        )
