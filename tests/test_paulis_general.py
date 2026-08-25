"""Tests for :mod:`rqutils.paulis.general`.

The module's own docstring is the specification, and its two normalization conventions are what
CLAUDE.md calls "the most bug-prone invariant here":

- ``lambda_0 = sqrt(2/n) I``, **not** ``I``.
- ``tr(lambda_k lambda_l) = 2 delta_kl`` for a single subsystem, and
  ``2 / 2**(s-1) prod(delta)`` for a product of ``s`` subsystems.

Basis-index ordering is the other load-bearing property: it is fixed by a shell-by-shell
construction loop, and :func:`components` and :func:`labels` both index by basis position, so a
reordering would silently disagree with the labels users read. :class:`TestBasisOrdering` pins it.

(``symmetry``, ``l0_projector``, ``truncate``, ``compose`` and ``paulis_shape`` were removed as dead
code, along with their tests. ``truncate``'s ``npmod=jnp`` path had been broken for every input form
in three separate ways -- the ``npmod`` rule those defects established is still enforced here by
:class:`TestNpmodParity` against ``components``, which is the last remaining ``npmod`` consumer.)
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import herm

import rqutils.paulis.general as pg

SINGLE_DIMS = [2, 3, 4, 5]
PRODUCT_DIMS = [(2, 2), (2, 3), (3, 2), (2, 2, 2)]


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
        normalized to ``I`` would still be a valid basis, but every ``components`` decomposition
        through it would be off by ``sqrt(n/2)``.
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


class TestComponents:
    """``components`` decomposes a matrix onto the basis, with the factor of 2 the norm forces."""

    @pytest.mark.parametrize("dim", PRODUCT_DIMS)
    def test_component_shape_for_products(self, dim):
        """The product case, where the ``1/2**(s-1)`` normalization factor enters."""
        rng = np.random.default_rng(20260804)
        matrix = herm(int(np.prod(dim)), rng)
        components = pg.components(matrix, dim=dim)
        assert np.asarray(components).shape == tuple(d**2 for d in dim)

    @pytest.mark.parametrize("dim", SINGLE_DIMS)
    def test_components_of_a_hermitian_matrix_are_real(self, dim):
        """The basis is Hermitian and orthogonal, so a Hermitian input has real components.

        A complex component would mean the decomposition had leaked an ``i`` -- the same class of
        error that broke ``svsim``.
        """
        rng = np.random.default_rng(20260804)
        components = np.asarray(pg.components(herm(dim, rng), dim=dim))
        assert np.abs(components.imag).max() < 1e-12

    def test_components_matches_the_documented_trace_formula(self):
        """``nu_k = tr(lambda_k H) / 2`` -- the docstring's own formula, computed by hand.

        Independent of ``components``'s implementation (which uses einsum over a memoized basis), so
        this pins the factor of 2 that the normalization forces.
        """
        rng = np.random.default_rng(20260804)
        dim = 4
        matrix = herm(dim, rng)
        by_hand = np.array([np.trace(m @ matrix) / 2.0 for m in basis_matrices(dim)])
        assert np.allclose(np.asarray(pg.components(matrix, dim=dim)), by_hand)


class TestBasisOrdering:
    """Basis-index ordering, which ``components`` and ``labels`` both index by position.

    CLAUDE.md: "Basis-index ordering is fixed by a shell-by-shell construction loop." A reordering
    would leave every function internally consistent while disagreeing with the labels users read,
    so the order is pinned against the matrices themselves.
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
        matrix = herm(int(np.prod(np.atleast_1d(dim))), rng)
        from_np = np.asarray(pg.components(matrix, dim=dim))
        from_jnp = np.asarray(pg.components(jnp.asarray(matrix), dim=dim, npmod=jnp))
        assert np.abs(from_np - from_jnp).max() < 1e-13

    def test_components_under_jit(self):
        rng = np.random.default_rng(20260804)
        matrix = herm(4, rng)
        traced = jax.jit(lambda m: pg.components(m, dim=4, npmod=jnp))
        assert np.allclose(
            np.asarray(traced(jnp.asarray(matrix))), np.asarray(pg.components(matrix, dim=4))
        )


class TestDimIsRequired:
    """``components`` requires ``dim``; it must not infer a basis from the matrix shape.

    With ``dim=None`` a 4x4 matrix inferred ``(4,)`` -- one 4-level qudit -- where the caller may have
    meant ``(2, 2)``, two qubits. Both pass the ``prod(dim)`` check (``4 == 4`` and ``2*2 == 4``) and
    both return 16 valid complex coefficients, but they are decompositions in *different bases*: the
    normalization factor ``2**(len(dim) - 2)`` is ``0.5`` for one subsystem against ``1.0`` for two, a
    2x difference, so the coefficient vectors differ in norm by ``sqrt(2)``. Measured 1.4142135623730951.

    Nothing signalled which one the caller got. The shapes do differ -- ``(16,)`` against ``(4, 4)`` --
    but a caller who flattens, or who only sums squares, sees two equally plausible answers.

    Under ``npmod=jnp`` it was worse: the ``prod(dim)`` check is gated on ``npmod is np``, so a wrong
    ``dim`` gave an opaque ``dot_general`` error rather than a ``ValueError``. Requiring ``dim`` does
    not fix that gating, but it removes the case where no ``dim`` was supplied at all.

    Every in-tree caller already passes ``dim``, so this is a downstream-only break.
    """

    def test_omitting_dim_raises(self):
        matrix = np.eye(4, dtype=np.complex128)
        with pytest.raises(TypeError, match="dim"):
            pg.components(matrix)  # ty: ignore[missing-argument]

    def test_the_two_readings_of_a_4x4_really_do_differ(self):
        """The premise: this is a genuine ambiguity, not a hypothetical one."""
        rng = np.random.default_rng(20260825)
        matrix = herm(4, rng)
        one_qudit = np.asarray(pg.components(matrix, dim=4))
        two_qubit = np.asarray(pg.components(matrix, dim=(2, 2)))
        assert one_qudit.shape == (16,)
        assert two_qubit.shape == (4, 4)
        ratio = np.linalg.norm(two_qubit.ravel()) / np.linalg.norm(one_qudit.ravel())
        assert abs(ratio - np.sqrt(2.0)) < 1e-12, ratio

    @pytest.mark.parametrize("dim", [2, 3, 4, (2, 2), (2, 3)])
    def test_explicit_dim_still_works(self, dim):
        rng = np.random.default_rng(20260825)
        total = int(np.prod(dim))
        matrix = herm(total, rng)
        expected = tuple(d**2 for d in ((dim,) if isinstance(dim, int) else dim))
        assert np.asarray(pg.components(matrix, dim=dim)).shape == expected

    def test_dim_may_still_be_passed_positionally(self):
        """Required, not keyword-only -- ``components(matrix, dim)`` reads unambiguously."""
        rng = np.random.default_rng(20260825)
        matrix = herm(4, rng)
        assert np.allclose(
            np.asarray(pg.components(matrix, (2, 2))),
            np.asarray(pg.components(matrix, dim=(2, 2))),
        )

    def test_a_mismatched_dim_still_raises_under_numpy(self):
        """The existing ``prod(dim)`` guard must survive the signature change."""
        rng = np.random.default_rng(20260825)
        with pytest.raises(ValueError):
            pg.components(herm(4, rng), dim=(2, 3))


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


class TestSparseCacheImmutability:
    """The cached ``sparse=True`` bases must not be mutable by the caller.

    ``pauli_matrices`` memoizes in a module-level dict and returns the cached object directly. The
    dense path is protected -- ``matrices.setflags(write=False)`` makes an in-place write raise -- but
    the sparse path was not, and the source comment conceded that ``setflags`` on an object array of
    ``csr_array`` s "would not protect its elements".

    So every caller received *the same* CSR objects, and an in-place ``/=`` for a different
    normalization convention corrupted the cache for the **process lifetime**. Measured before the
    fix: ``pauli_matrices(3, sparse=True)[1] /= 2`` changed the cached basis by 0.5 max abs, and the
    result stayed Hermitian -- so every later :func:`components` call returned plausible,
    consistently wrong coefficients. Normalization is the invariant ``CLAUDE.md`` calls "the most
    bug-prone" in this module.

    Fixed by making the CSR buffers read-only rather than copying on return: a copy per call would
    pay for every read to protect against a rare write, where ``setflags`` on ``.data``/``.indices``/
    ``.indptr`` blocks the mutation at its source and costs nothing.
    """

    @pytest.mark.parametrize("dim", [2, 3, 4])
    def test_in_place_division_raises(self, dim):
        """The exact corruption: renormalizing a cached basis in place."""
        matrices = pg.pauli_matrices(dim, sparse=True)
        with pytest.raises(ValueError, match="read-only"):
            matrices[1] /= 2.0

    def test_in_place_multiplication_raises(self):
        matrices = pg.pauli_matrices(3, sparse=True)
        with pytest.raises(ValueError, match="read-only"):
            matrices[1] *= 2.0

    def test_writing_the_data_buffer_raises(self):
        """The lower-level route: reaching past the operator into ``.data``."""
        matrices = pg.pauli_matrices(3, sparse=True)
        with pytest.raises(ValueError, match="read-only"):
            matrices[1].data[0] = 9.0

    def test_the_cache_survives_an_attempted_mutation(self):
        """The property that actually matters: a failed write must leave the cache intact."""
        before = pg.pauli_matrices(3, sparse=True)[1].toarray().copy()
        with pytest.raises(ValueError):
            pg.pauli_matrices(3, sparse=True)[1] /= 2.0
        after = pg.pauli_matrices(3, sparse=True)[1].toarray()
        assert np.allclose(before, after)

    @pytest.mark.parametrize("dim", [2, 3, 4, 5])
    def test_reads_still_work(self, dim):
        """The guard must not break the operations the basis exists for."""
        matrices = pg.pauli_matrices(dim, sparse=True)
        vec = np.ones(dim)
        for mat in matrices:
            assert mat.toarray().shape == (dim, dim)
            assert (mat @ vec).shape == (dim,)
        # A caller who *wants* to rescale can still copy first.
        scaled = matrices[1].copy()
        scaled /= 2.0
        assert np.allclose(scaled.toarray() * 2.0, matrices[1].toarray())

    @pytest.mark.parametrize("dim", [2, 3, 4, 5, 6])
    def test_sparse_still_agrees_with_dense(self, dim):
        """Guarding the buffers must not change any value."""
        sparse = pg.pauli_matrices(dim, sparse=True)
        dense = np.asarray(pg.pauli_matrices(dim, sparse=False))
        for isparse, idense in zip(sparse, dense, strict=True):
            assert np.abs(isparse.toarray() - idense).max() == 0.0

    def test_the_dense_path_was_already_protected(self):
        """Pinned for contrast -- the asymmetry is what made the sparse gap easy to miss."""
        with pytest.raises(ValueError, match="read-only|assignment destination"):
            pg.pauli_matrices(3, sparse=False)[1] /= 2.0


class TestShapesAndMemoization:
    """Shape conventions, which CLAUDE.md warns are transposed between the two directions."""

    @pytest.mark.parametrize("dim", PRODUCT_DIMS)
    def test_paulis_puts_basis_axes_first(self, dim):
        """``paulis(dim)`` -> ``(d1**2, ..., D, D)``: basis axes first, matrix axes last."""
        total = int(np.prod(dim))
        assert np.asarray(pg.paulis(dim)).shape == tuple(d**2 for d in dim) + (total, total)

    @pytest.mark.parametrize("dim", PRODUCT_DIMS)
    def test_components_puts_component_axes_last(self, dim):
        """Component arrays are ``(..., d1**2, ...)``: the opposite convention from ``paulis``."""
        rng = np.random.default_rng(20260804)
        matrix = herm(int(np.prod(dim)), rng)
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
