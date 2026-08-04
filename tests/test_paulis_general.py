"""Tests for :mod:`rqutils.paulis.general`.

The module's own docstring is the specification, and its two normalization conventions are what
CLAUDE.md calls "the most bug-prone invariant here":

- ``lambda_0 = sqrt(2/n) I``, **not** ``I``.
- ``tr(lambda_k lambda_l) = 2 delta_kl`` for a single subsystem, and
  ``2 / 2**(s-1) prod(delta)`` for a product of ``s`` subsystems.

Basis-index ordering is the other load-bearing property: it is fixed by a shell-by-shell
construction loop and *relied on* by :func:`symmetry`, :func:`l0_projector`, and :func:`truncate`.
Nothing pinned it before this suite, so a reordering would have silently broken all three consumers
while every individual function still looked self-consistent. :class:`TestBasisOrdering` exists for
that.

One bug was found and fixed while writing these: ``truncate(..., npmod=jnp)`` could not run at all
for any input form. See :class:`TestTruncate` for what was wrong and how it presented.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import rqutils.paulis.general as pg

SINGLE_DIMS = [2, 3, 4, 5]
PRODUCT_DIMS = [(2, 2), (2, 3), (3, 2), (2, 2, 2)]


def hermitian(dim, rng):
    """Return a random ``dim``-by-``dim`` Hermitian matrix."""
    mat = rng.normal(size=(dim, dim)) + 1.0j * rng.normal(size=(dim, dim))
    return mat + mat.conjugate().T


def basis_matrices(dim):
    """Return the Pauli basis for ``dim`` flattened to ``(num_basis, N, N)``."""
    total = int(np.prod(np.atleast_1d(dim)))
    return np.asarray(pg.paulis(dim)).reshape(-1, total, total)


class TestNormalization:
    """The normalization conventions the rest of the module depends on."""

    @pytest.mark.parametrize("dim", SINGLE_DIMS)
    def test_lambda_0_is_not_the_identity(self, dim):
        """``lambda_0 = sqrt(2/n) I``.

        The module docstring calls this out explicitly ("note that lambda_0 is *not* the
        n-dimensional identity matrix"), and CLAUDE.md names it the most bug-prone invariant in the
        package. Asserting the scale factor rather than just proportionality is the point: a basis
        normalized to ``I`` would still be a valid basis, but every ``components``/``compose``
        round-trip through it would be off by ``sqrt(n/2)``.
        """
        lambda_0 = np.asarray(pg.paulis(dim))[0]
        assert np.allclose(lambda_0, np.sqrt(2.0 / dim) * np.eye(dim))
        if dim != 2:
            # sqrt(2/2) == 1, so for a qubit lambda_0 genuinely *is* the identity -- the invariant
            # is the scale factor, and dim=2 is the one case where it happens to be unity. Asserting
            # "not the identity" unconditionally would be asserting arithmetic is false.
            assert not np.allclose(lambda_0, np.eye(dim)), "lambda_0 must not be the bare identity"

    @pytest.mark.parametrize("dim", SINGLE_DIMS)
    def test_single_subsystem_orthonormality(self, dim):
        """``tr(lambda_k lambda_l) = 2 delta_kl``, checked as a full Gram matrix.

        Checking the whole matrix rather than a few pairs also proves the basis is complete (there
        are exactly ``n**2`` of them) and that no two entries are duplicates -- a construction-loop
        error that produced the same matrix twice would leave an off-diagonal 2.
        """
        matrices = basis_matrices(dim)
        assert matrices.shape[0] == dim**2
        gram = np.einsum("kij,lji->kl", matrices, matrices)
        assert np.allclose(gram, 2.0 * np.eye(dim**2))

    @pytest.mark.parametrize("dim", PRODUCT_DIMS)
    def test_product_normalization_carries_the_half_power(self, dim):
        """For ``s`` subsystems the norm is ``2 / 2**(s-1)``, the "rather awkward" one.

        This is a different constant from the single-subsystem case, so a test that only covered
        one subsystem would miss a wrong power of two entirely. Measured: 1.0 for two subsystems,
        0.5 for three.
        """
        matrices = basis_matrices(dim)
        num_subsystems = len(dim)
        gram = np.einsum("kij,lji->kl", matrices, matrices)
        expected = 2.0 / 2.0 ** (num_subsystems - 1)
        assert np.allclose(gram, expected * np.eye(matrices.shape[0]))

    @pytest.mark.parametrize("dim", SINGLE_DIMS)
    def test_basis_is_hermitian(self, dim):
        """Every basis element must be Hermitian: they span the Hermitian matrices."""
        for index, matrix in enumerate(basis_matrices(dim)):
            assert np.allclose(matrix, matrix.conjugate().T), f"lambda_{index} is not Hermitian"

    @pytest.mark.parametrize("dim", SINGLE_DIMS)
    def test_traceless_except_the_zeroth(self, dim):
        """All but ``lambda_0`` are traceless -- the standard generalized-Pauli property."""
        matrices = basis_matrices(dim)
        assert np.trace(matrices[0]).real == pytest.approx(np.sqrt(2.0 / dim) * dim)
        for index, matrix in enumerate(matrices[1:], start=1):
            assert abs(np.trace(matrix)) < 1e-12, f"lambda_{index} is not traceless"


class TestComponentsCompose:
    """``components`` and ``compose`` must invert each other exactly."""

    @pytest.mark.parametrize("dim", SINGLE_DIMS)
    def test_round_trip_single(self, dim):
        rng = np.random.default_rng(20260804)
        matrix = hermitian(dim, rng)
        components = pg.components(matrix, dim=dim)
        assert np.allclose(np.asarray(pg.compose(components, dim=dim)), matrix)

    @pytest.mark.parametrize("dim", PRODUCT_DIMS)
    def test_round_trip_product(self, dim):
        """The product case, where the ``1/2**(s-1)`` factor has to cancel in both directions."""
        rng = np.random.default_rng(20260804)
        matrix = hermitian(int(np.prod(dim)), rng)
        components = pg.components(matrix, dim=dim)
        assert np.asarray(components).shape == tuple(d**2 for d in dim)
        assert np.allclose(np.asarray(pg.compose(components, dim=dim)), matrix)

    @pytest.mark.parametrize("dim", SINGLE_DIMS)
    def test_components_of_a_hermitian_matrix_are_real(self, dim):
        """The basis is Hermitian and orthogonal, so a Hermitian input has real components.

        A complex component would mean the decomposition had leaked an ``i`` -- the same class of
        error that broke ``svsim``.
        """
        rng = np.random.default_rng(20260804)
        components = np.asarray(pg.components(hermitian(dim, rng), dim=dim))
        assert np.abs(components.imag).max() < 1e-12

    def test_components_matches_the_documented_trace_formula(self):
        """``nu_k = tr(lambda_k H) / 2`` -- the docstring's own formula, computed by hand.

        Independent of ``components``'s implementation (which uses einsum over a memoized basis), so
        this pins the factor of 2 that the normalization forces.
        """
        rng = np.random.default_rng(20260804)
        dim = 4
        matrix = hermitian(dim, rng)
        by_hand = np.array([np.trace(m @ matrix) / 2.0 for m in basis_matrices(dim)])
        assert np.allclose(np.asarray(pg.components(matrix, dim=dim)), by_hand)


class TestBasisOrdering:
    """Basis-index ordering is relied on by ``symmetry``, ``l0_projector``, and ``truncate``.

    CLAUDE.md: "Basis-index ordering is fixed by a shell-by-shell construction loop and is relied on
    by ``symmetry``, ``l0_projector``, and ``truncate``." Nothing pinned it before this class, so a
    reordering would have broken all three while each still looked internally consistent.
    """

    def test_qubit_basis_is_i_x_y_z(self):
        """For ``dim=2`` the order is exactly I, X, Y, Z (up to the ``sqrt(2/2) = 1`` scale).

        This is the anchor: every higher dimension is built shell-by-shell on top of it.
        """
        matrices = basis_matrices(2)
        assert np.allclose(matrices[0], np.eye(2))
        assert np.allclose(matrices[1], [[0.0, 1.0], [1.0, 0.0]])
        assert np.allclose(matrices[2], [[0.0, -1.0j], [1.0j, 0.0]])
        assert np.allclose(matrices[3], [[1.0, 0.0], [0.0, -1.0]])

    def test_label_order_and_product_normalization(self):
        """``labels`` names basis positions, from an independent symbol table.

        It is generated from a symbol list rather than derived from the matrices, so it *cannot*
        detect a matrix reordering -- verified by mutation: swapping X and Y in the basis fails
        :meth:`test_qubit_basis_is_i_x_y_z` and :meth:`test_symmetry_classifies_by_reality` while
        leaving ``labels`` untouched. What this test pins is the labelling convention users read,
        and the ``/2`` suffix that reflects the product normalization.
        """
        assert list(np.asarray(pg.labels(2, fmt="text")).ravel()) == ["I", "X", "Y", "Z"]
        # Two subsystems: row-major over (k1, k2), and the product picks up the /2 normalization.
        two_qubit = np.asarray(pg.labels((2, 2), fmt="text"))
        assert two_qubit.shape == (4, 4)
        assert list(two_qubit[0]) == ["II/2", "IX/2", "IY/2", "IZ/2"]
        assert list(two_qubit[:, 0]) == ["II/2", "XI/2", "YI/2", "ZI/2"]

    @pytest.mark.parametrize("dim", SINGLE_DIMS)
    def test_symmetry_classifies_by_reality(self, dim):
        """``symmetry`` returns +1 for real off-diagonal, -1 for imaginary, 0 for diagonal.

        Verified against the matrices themselves rather than against a hardcoded list, so the test
        states the *meaning* of the classification and stays valid at any dimension. This is the
        clearest consumer of the ordering: ``symmetry`` indexes by basis position alone.
        """
        symmetries = np.asarray(pg.symmetry(dim))
        matrices = basis_matrices(dim)
        assert symmetries.shape == (dim**2,)
        for index, (sym, matrix) in enumerate(zip(symmetries, matrices)):
            is_real = np.allclose(matrix.imag, 0.0)
            is_imaginary = np.allclose(matrix.real, 0.0)
            if sym == -1:
                assert is_imaginary, f"lambda_{index} marked antisymmetric but is not imaginary"
            else:
                assert is_real, f"lambda_{index} marked {sym} but is not real"
            if sym == 0:
                assert np.allclose(matrix, np.diag(np.diagonal(matrix))), (
                    f"lambda_{index} marked 0 but is not diagonal"
                )

    @pytest.mark.parametrize("dim", SINGLE_DIMS)
    def test_symmetry_counts(self, dim):
        """An ``n``-dim basis has ``n`` diagonal, and ``n(n-1)/2`` each of real and imaginary.

        A dimension-independent count, so it catches a shell that was built with the wrong number of
        off-diagonal pairs without depending on where in the order they land.
        """
        symmetries = np.asarray(pg.symmetry(dim))
        assert np.count_nonzero(symmetries == 0) == dim
        assert np.count_nonzero(symmetries == 1) == dim * (dim - 1) // 2
        assert np.count_nonzero(symmetries == -1) == dim * (dim - 1) // 2

    @pytest.mark.parametrize("reduced,original", [(2, 3), (2, 4), (3, 4), (2, 5)])
    def test_l0_projector_maps_lambda_0_between_dimensions(self, reduced, original):
        """``l0_projector`` expresses the reduced ``lambda_0`` in the original basis.

        Its nonzero entries land only on the diagonal (``symmetry == 0``) basis indices, which is
        precisely an ordering dependency: the projector is a flat vector indexed by basis position.
        """
        projector = np.asarray(pg.l0_projector(reduced, original))
        assert projector.shape == (original**2,)
        diagonal_indices = np.nonzero(np.asarray(pg.symmetry(original)) == 0)[0]
        nonzero_indices = np.nonzero(np.abs(projector) > 1e-12)[0]
        assert set(nonzero_indices) <= set(diagonal_indices), (
            "l0_projector put weight on a non-diagonal basis element"
        )
        # Contracting it with the original basis must give the reduced lambda_0, embedded.
        composed = np.einsum("k,kij->ij", projector, basis_matrices(original))
        expected = np.zeros((original, original), dtype=np.complex128)
        expected[:reduced, :reduced] = np.sqrt(2.0 / reduced) * np.eye(reduced)
        assert np.allclose(composed, expected)


class TestTruncate:
    """``truncate`` restricts components to a submatrix, and must agree across backends.

    The ``npmod=jnp`` path was broken for every input form, in three separate ways, so it had
    evidently never been executed:

    1. The scalar-to-tuple normalization of ``reduced_dim`` was gated on ``npmod is np``, so the jnp
       path hit ``len(reduced_dim)`` on a bare int -- ``TypeError: object of type 'int' has no
       len()``, naming nothing.
    2. ``npmod.sqrt(original_shape)`` was called on a Python tuple. numpy accepts that; JAX raises
       ``sqrt requires ndarray or scalar arguments``. Shape arithmetic is static, so this is now
       plain ``np``.
    3. The jnp branch used ``jax.lax.fori_loop`` over subsystems, whose traced index cannot
       subscript the static ``reduced_dim`` tuple -- ``TracerIntegerConversionError``. The trip count
       is static, so it is now an unrolled Python loop for both backends.

    Also fixed: the scalar form built its tuple as ``(dim,) * len(components.shape)``, which
    over-counted whenever the component array carried leading axes (the docstring's time-series
    case). A scalar means one subsystem.
    """

    @pytest.mark.parametrize("reduced_dim", [2, (2,)])
    def test_scalar_and_tuple_forms_agree(self, reduced_dim):
        """A scalar ``reduced_dim`` is shorthand for a one-subsystem tuple."""
        rng = np.random.default_rng(20260804)
        components = pg.components(hermitian(3, rng), dim=3)
        truncated = np.asarray(pg.truncate(components, reduced_dim))
        assert truncated.shape == (4,)

    @pytest.mark.parametrize("original,reduced", [(3, 2), (4, 2), (4, 3), (5, 2)])
    def test_truncation_equals_the_submatrix(self, original, reduced):
        """The defining property: ``compose(truncate(components(H))) == H[:r, :r]``.

        Checked against the raw submatrix rather than against another call to ``truncate``, so this
        is a real specification and not a self-consistency check.
        """
        rng = np.random.default_rng(20260804)
        matrix = hermitian(original, rng)
        truncated = pg.truncate(pg.components(matrix, dim=original), reduced)
        assert np.allclose(
            np.asarray(pg.compose(truncated, dim=reduced)), matrix[:reduced, :reduced]
        )

    def test_multi_subsystem(self):
        """Truncating one subsystem of a product leaves the other alone."""
        rng = np.random.default_rng(20260804)
        matrix = hermitian(6, rng)
        components = pg.components(matrix, dim=(2, 3))
        truncated = np.asarray(pg.truncate(components, (2, 2)))
        assert truncated.shape == (4, 4)
        composed = np.asarray(pg.compose(truncated, dim=(2, 2)))
        # (2,3) -> (2,2) keeps subsystem-1 levels 0..1, i.e. rows/cols {0,1,3,4} of the 6x6.
        keep = np.array([0, 1, 3, 4])
        assert np.allclose(composed, matrix[np.ix_(keep, keep)])

    def test_no_op_when_dimensions_match(self):
        rng = np.random.default_rng(20260804)
        components = np.asarray(pg.components(hermitian(3, rng), dim=3))
        assert np.allclose(np.asarray(pg.truncate(components, (3,))), components)

    def test_leading_axes_are_preserved(self):
        """The docstring's time-series case: extra front axes pass through untouched.

        This is what the old ``(dim,) * len(components.shape)`` scalar expansion got wrong -- with a
        leading axis it built a 2-tuple for a 1-subsystem array.
        """
        rng = np.random.default_rng(20260804)
        matrices = [hermitian(3, rng) for _ in range(4)]
        series = np.stack([np.asarray(pg.components(m, dim=3)) for m in matrices])
        assert series.shape == (4, 9)
        truncated = np.asarray(pg.truncate(series, (2,)))
        assert truncated.shape == (4, 4)
        for index, matrix in enumerate(matrices):
            composed = np.asarray(pg.compose(truncated[index], dim=2))
            assert np.allclose(composed, matrix[:2, :2]), f"time slice {index}"

    def test_rejects_growing_the_dimension(self):
        rng = np.random.default_rng(20260804)
        components = pg.components(hermitian(2, rng), dim=2)
        with pytest.raises(ValueError, match="greater than original"):
            pg.truncate(components, (3,))


class TestNpmodParity:
    """The ``npmod`` convention: passing ``jax.numpy`` must give the same numbers as numpy.

    CLAUDE.md documents this as a hard rule for the package ("numeric functions take
    ``npmod: ModuleType = np`` ... so callers can pass ``jax.numpy`` for traceable execution"), and
    the whole point is traceability -- so each function is also checked inside ``jax.jit``, which is
    the only thing that proves no Python-level branch on a traced value survives.
    """

    @pytest.mark.parametrize("dim", [3, (2, 2), (2, 3)])
    def test_components_parity(self, dim):
        rng = np.random.default_rng(20260804)
        matrix = hermitian(int(np.prod(np.atleast_1d(dim))), rng)
        from_np = np.asarray(pg.components(matrix, dim=dim))
        from_jnp = np.asarray(pg.components(jnp.asarray(matrix), dim=dim, npmod=jnp))
        assert np.abs(from_np - from_jnp).max() < 1e-13

    @pytest.mark.parametrize("dim", [3, (2, 2), (2, 3)])
    def test_compose_parity(self, dim):
        rng = np.random.default_rng(20260804)
        matrix = hermitian(int(np.prod(np.atleast_1d(dim))), rng)
        components = np.asarray(pg.components(matrix, dim=dim))
        from_np = np.asarray(pg.compose(components, dim=dim))
        from_jnp = np.asarray(pg.compose(jnp.asarray(components), dim=dim, npmod=jnp))
        assert np.abs(from_np - from_jnp).max() < 1e-13

    @pytest.mark.parametrize("reduced_dim", [2, (2,)])
    def test_truncate_parity(self, reduced_dim):
        """Both input forms, both backends -- the combination that used to raise three ways."""
        rng = np.random.default_rng(20260804)
        components = np.asarray(pg.components(hermitian(3, rng), dim=3))
        from_np = np.asarray(pg.truncate(components, reduced_dim))
        from_jnp = np.asarray(pg.truncate(jnp.asarray(components), reduced_dim, npmod=jnp))
        assert np.abs(from_np - from_jnp).max() < 1e-13

    def test_truncate_under_jit(self):
        """``npmod=jnp`` exists for tracing, so it has to survive ``jax.jit``.

        The old jnp branch failed here specifically: ``fori_loop``'s traced index cannot index the
        static ``reduced_dim`` tuple. A non-jit parity test alone would not have caught that.
        """
        rng = np.random.default_rng(20260804)
        components = np.asarray(pg.components(hermitian(3, rng), dim=3))
        traced = jax.jit(lambda c: pg.truncate(c, (2,), npmod=jnp))
        assert np.allclose(
            np.asarray(traced(jnp.asarray(components))),
            np.asarray(pg.truncate(components, (2,))),
        )

    def test_components_under_jit(self):
        rng = np.random.default_rng(20260804)
        matrix = hermitian(4, rng)
        traced = jax.jit(lambda m: pg.components(m, dim=4, npmod=jnp))
        assert np.allclose(
            np.asarray(traced(jnp.asarray(matrix))), np.asarray(pg.components(matrix, dim=4))
        )


class TestDocumentedLimits:
    """Limits CLAUDE.md records as known rough edges, pinned so they fail loudly, not obscurely."""

    def test_too_many_subsystems_raises(self):
        """``paulis`` uses ``np.einsum`` with 3 letters per subsystem, capping at ~17.

        The point of the test is the error *type*: an einsum letter exhaustion would otherwise
        surface as an opaque numpy message.
        """
        with pytest.raises(NotImplementedError, match="Too many subsystems"):
            pg.paulis((2,) * 18)

    def test_sparse_product_raises(self):
        with pytest.raises(NotImplementedError):
            pg.paulis((2, 2), sparse=True)

    def test_sparse_single_subsystem_matches_dense(self):
        """``sparse=True`` is supported for a single subsystem, and must agree with the dense form."""
        dense = np.asarray(pg.paulis(3))
        sparse = pg.paulis(3, sparse=True)
        assert len(sparse) == 9
        for index, (dense_matrix, sparse_matrix) in enumerate(zip(dense, sparse)):
            assert np.allclose(sparse_matrix.toarray(), dense_matrix), f"lambda_{index}"


class TestShapesAndMemoization:
    """Shape conventions, which CLAUDE.md warns are transposed between the two directions."""

    @pytest.mark.parametrize("dim", PRODUCT_DIMS)
    def test_paulis_puts_basis_axes_first(self, dim):
        """``paulis(dim)`` -> ``(d1**2, ..., D, D)``: basis axes first, matrix axes last."""
        total = int(np.prod(dim))
        assert np.asarray(pg.paulis(dim)).shape == tuple(d**2 for d in dim) + (total, total)
        assert pg.paulis_shape(dim) == tuple(d**2 for d in dim) + (total, total)

    @pytest.mark.parametrize("dim", PRODUCT_DIMS)
    def test_components_puts_component_axes_last(self, dim):
        """Component arrays are ``(..., d1**2, ...)``: the opposite convention from ``paulis``."""
        rng = np.random.default_rng(20260804)
        matrix = hermitian(int(np.prod(dim)), rng)
        assert np.asarray(pg.components(matrix, dim=dim)).shape == tuple(d**2 for d in dim)

    def test_memoization_returns_consistent_values(self):
        """Results are memoized in module-level dicts keyed by a tuple-normalized ``dim``.

        Two calls must agree, and an equivalent ``dim`` spelling must hit the same entry -- a cache
        keyed on the raw argument would treat ``3`` and ``(3,)`` as different problems.
        """
        first = np.asarray(pg.paulis(3))
        second = np.asarray(pg.paulis(3))
        assert np.array_equal(first, second)
        assert np.array_equal(first, np.asarray(pg.paulis((3,))))

    def test_pauli_matrices_matches_paulis_for_one_subsystem(self):
        """``pauli_matrices`` is the single-subsystem primitive ``paulis`` builds on."""
        assert np.allclose(np.asarray(pg.pauli_matrices(4)), np.asarray(pg.paulis(4)))
