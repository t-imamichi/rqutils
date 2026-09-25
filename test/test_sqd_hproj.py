"""Tests for :func:`rqutils.sqd.hproj`, the dense projection the suite uses as its debug path.
Organized by defect, like ``test_sqd.py``.
"""

import numpy as np
import pytest
from conftest import (
    eigval_of,
    lowest_projected,
    project_dense,
    real_pauli_strings,
    unique_states,
)

from rqutils.sqd import (
    _is_lex_sorted,
    hproj,
    uniquify_states,
)


class TestHproj:
    """``hproj`` builds the projected Hamiltonian densely (sparse), as a debug/reference path."""

    def test_subspace_above_the_int32_ceiling_raises(self):
        """``hproj`` reaches ``get_xsource`` too, so it shares ``sqd``'s int32 ceiling.

        ``2**31`` rows cannot be allocated, so the shape is produced by ``np.broadcast_to`` (a view,
        no allocation) -- enough to reach a guard that reads ``states.shape[0]``.

        The guard's *placement* matters as much as its presence: it sits before the O(N) sortedness
        scan and the ``np.unique``, so a doomed call reports the real problem instead of spending time
        first -- this test is where that shows up, since the scan would run over all 2^31 rows.
        """
        states = np.broadcast_to(np.zeros(2, dtype=np.uint8), (2**31, 2))
        with pytest.raises(ValueError, match="exceeds the .* limit imposed by int32"):
            hproj((["ZI"], [1.0]), states, unique_states=True)

    def test_states_columns_are_character_indexed_not_qubit_indexed(self):
        """A ``states`` column is a Pauli-string *character* position, not a qubit number.

        The convention crossing this pins: ``SparsePauliOp.from_sparse_list`` is indexed by **qubit**
        (``("Z", [q], 1.0)`` puts Z on qubit ``q``), while ``states[:, j]`` is character ``j`` of the
        Pauli string. Since character ``j`` is qubit ``n-1-j``, a caller pairing the two must reverse:
        bit ``q`` of a basis code belongs in column ``n-1-q``.

        Getting it backwards is silent. The projection stays symmetric and the eigenvalue stays a
        genuine variational bound -- of the bit-reversed subspace -- so it reads as a poor sample
        rather than a bug. Measured with the naive ``bit q -> column q`` pairing, ``Z`` on qubit 0 over
        codes ``{0, 1}`` gives ``diag == [1, 1]``: no dependence on qubit 0 whatsoever, against the
        correct ``[1, -1]``. That exact defect shipped in the ``subspace`` helper in
        ``markdown/rqutils-precond-request.md`` and propagated to a POC that copied it.

        Uncovered until now because every other qiskit test here builds operators from *strings*
        (``SparsePauliOp(["ZI"], ...)``), where character order is what the caller already wrote. Only
        the index-based constructor exposes the flip.
        """
        qiskit = pytest.importorskip("qiskit")
        num_qubits = 4

        for qubit in range(num_qubits):
            op = qiskit.quantum_info.SparsePauliOp.from_sparse_list(
                [("Z", [qubit], 1.0)], num_qubits
            )
            # Two codes differing only in bit `qubit`, packed with the correct reversal.
            codes = [0, 1 << qubit]
            states = np.array(
                [[(code >> k) & 1 for k in range(num_qubits)][::-1] for code in codes],
                dtype=np.uint8,
            )
            states = states[np.lexsort(states.T[::-1])]
            diagonal = np.real(np.diag(hproj(op, states).toarray()))
            # <s|Z_q|s> is +1 when bit q is 0 and -1 when it is 1, so the pair must straddle zero.
            assert sorted(diagonal) == pytest.approx([-1.0, 1.0]), (
                f"Z on qubit {qubit} gave diag {diagonal}; a column/qubit mismatch would give "
                "[1, 1] or [-1, -1], i.e. no dependence on that qubit"
            )

        # And the naive pairing really is wrong, so the reversal above is load-bearing rather than
        # cosmetic. Asserted rather than assumed: without this, the test above would pass for a
        # convention-free implementation too.
        op = qiskit.quantum_info.SparsePauliOp.from_sparse_list([("Z", [0], 1.0)], num_qubits)
        naive = np.array(
            [[(code >> k) & 1 for k in range(num_qubits)] for code in [0, 1]], dtype=np.uint8
        )
        naive = naive[np.lexsort(naive.T[::-1])]
        naive_diagonal = np.real(np.diag(hproj(op, naive).toarray()))
        assert sorted(naive_diagonal) == pytest.approx([1.0, 1.0]), (
            f"expected the naive pairing to be insensitive to qubit 0, got {naive_diagonal}"
        )

    def test_unsorted_input_with_unique_states_raises(self):
        """``unique_states=True`` rejects unsorted states instead of projecting them wrongly.

        ``get_xsource`` requires a lex-sorted ``S``. Both production callers satisfy it -- ``run_sqd``
        via ``uniquify_states``, ``hproj`` via ``np.unique(..., axis=0)`` -- but ``hproj``'s
        ``unique_states=True`` shortcut skips that ``np.unique``, so a caller who has already
        deduplicated *without* sorting violates the precondition.

        This is the flipped form of ``test_unsorted_input_with_unique_states_is_wrong``, which pinned
        the old behaviour: the returned matrix was **not symmetric**, which a Hermitian projection
        must always be. That was a silent wrong answer, so the shortcut now validates sortedness and
        raises instead.

        The sorted arm below is what keeps this honest: a guard that rejected *everything* would also
        satisfy the ``raises`` assertions.
        """
        # Unique but NOT sorted: row order is 1000, 0000, 0001.
        unsorted = np.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]], dtype=np.uint8)
        with pytest.raises(ValueError, match="uniquified and lex-sorted"):
            hproj((["IIIX"], [1.0]), unsorted, unique_states=True)

        # Duplicate rows are rejected too: "uniquified" is the other half of the precondition.
        dup = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]], dtype=np.uint8)
        with pytest.raises(ValueError, match="uniquified and lex-sorted"):
            hproj((["IIIX"], [1.0]), dup, unique_states=True)

        # The same states, sorted, give the correct symmetric coupling 0000 <-> 0001.
        good = hproj((["IIIX"], [1.0]), np.unique(unsorted, axis=0), unique_states=True)
        good = good.toarray().real
        assert np.allclose(good, good.T)
        expected = np.zeros((3, 3))
        expected[0, 1] = expected[1, 0] = 1.0
        assert np.allclose(good, expected)

    def test_is_lex_sorted_discriminates(self):
        """``_is_lex_sorted`` must agree with ``np.unique``, including where bytes tie.

        Row lex order is decided by the *first differing byte*, so the traps are pairs that agree on
        a prefix. A check written as ``np.all(np.diff(keys) > 0)`` over per-row sums, or one comparing
        only byte 0, passes the obvious cases and still admits ``[[0, 9], [0, 3]]``.
        """
        cases = [
            (np.array([[0, 0], [0, 1], [1, 0]], np.uint8), True),
            (np.array([[0, 9], [0, 3]], np.uint8), False),  # ties on byte 0, decided by byte 1
            (np.array([[0, 3], [0, 9]], np.uint8), True),
            (np.array([[1, 0], [0, 255]], np.uint8), False),  # larger row sum, still unsorted
            (np.array([[5, 5]], np.uint8), True),  # single row is trivially sorted
            (np.array([[7, 7], [7, 7]], np.uint8), False),  # duplicates are not *strictly* sorted
        ]
        for rows, expected in cases:
            assert _is_lex_sorted(rows) is expected, f"{rows.tolist()} -> expected {expected}"
            # Cross-check against the ordering hproj's default path actually produces.
            if expected:
                assert np.array_equal(rows, np.unique(rows, axis=0))

        # Zero rows: no adjacent pair exists, so vacuously sorted rather than an IndexError.
        assert _is_lex_sorted(np.zeros((0, 4), np.uint8)) is True

    def test_padded_uniquify_output_is_rejected(self):
        """A padded ``uniquify_states`` result must NOT pass, since ``hproj`` cannot mask fillers.

        This is the one way the sortedness guard could surprise a caller -- ``uniquify_states`` output
        is otherwise exactly what ``get_xsource`` wants -- so it is pinned rather than left to be
        rediscovered. ``sqd`` trims fillers before returning its basis, which is why
        ``poc/sharding.py`` can hand that basis straight to ``hproj``.

        **This test used to assert the opposite of its own title for the single-filler case**, which is
        how the parity hole (``markdown/gotchas.md`` item 14) survived: rejection rested on two or more
        ``255`` rows being *duplicates*, so exactly one filler was still strictly increasing and
        passed, and the assertion below read ``is True``. The docstring's stated intent was right and
        the assertion was wrong. There is now an explicit high-bit test in ``_is_lex_sorted``,
        independent of sortedness, so any number of fillers is rejected --
        see :class:`TestSingleFillerRow` for the measured consequence.
        """
        from rqutils.paulis.symplectic import PauliSumXZ

        rng = np.random.default_rng(3)
        states = rng.integers(0, 2, size=(12, 4), dtype=np.uint8)
        packed = PauliSumXZ.pack_states(states)

        # states_size=12 happens to leave a single filler: strictly increasing, so only the explicit
        # filler test catches it.
        one_filler = np.asarray(uniquify_states(packed, 12))
        assert int((one_filler[:, 0] >> 7).sum()) == 1
        assert _is_lex_sorted(one_filler) is False

        # Two or more are also duplicates, so they fail either way.
        many_fillers = np.asarray(uniquify_states(packed, 16))
        assert int((many_fillers[:, 0] >> 7).sum()) > 1
        assert _is_lex_sorted(many_fillers) is False

    def test_matches_dense_reference(self):
        """``hproj`` packed states WITHOUT the pad bit while padding the Hamiltonian.

        ``PauliSumXZ`` shifts every X/Z signature one bit right for the pad bit, so unpadded states
        disagree with them on alignment and every matrix element lands in the wrong column. Measured
        before the fix on this input: lowest eigenvalue -1.398 against a true -2.191. The padding was
        an opt-in ``add_padding`` flag at the time, which is what let the two sides disagree; it is
        now unconditional, so they cannot. A benchmark under ``examples/`` builds its own dense
        reference instead of calling ``hproj`` -- once because of this bug, now for independence,
        since a gate that reruns the code under test proves nothing.
        """
        rng = np.random.default_rng(20260804)
        num_qubits = 5
        strings = real_pauli_strings(num_qubits, 6, rng)
        coeffs = rng.normal(size=len(strings))
        states = rng.integers(0, 2, size=(20, num_qubits)).astype(np.uint8)

        matrix = hproj((strings, coeffs.tolist()), states).toarray()
        expected = project_dense(strings, coeffs, states)
        assert matrix.shape == expected.shape
        assert np.abs(matrix - expected.real).max() < 1e-12

    def test_agrees_with_sqd_on_a_subspace_with_a_decoupled_state(self):
        """``hproj`` and ``sqd`` must agree, since both align states against the same pad bit.

        This is the pairing that the old ``add_padding`` flag left unenforced: each path decided
        independently whether to pad, and when they disagreed every matrix element moved one column
        and the answer was still symmetric, so nothing downstream could notice. With the padding
        intrinsic to ``PauliSumXZ`` there is no flag to set inconsistently, and this pins the two
        paths together.

        The subspace deliberately includes a state whose X-partner is absent, which is the case that
        also exercised the missing ``shape=`` on ``hproj``'s ``coo_array``: the two defects lived on
        the same input, so a passing assertion here covers alignment and extent at once.
        """
        rng = np.random.default_rng(20260805)
        num_qubits = 5
        strings = real_pauli_strings(num_qubits, 6, rng)
        coeffs = rng.normal(size=len(strings))
        # Sparse draw from a 32-state space, so some X-partners necessarily fall outside.
        states = unique_states(7, num_qubits, rng)

        matrix = hproj((strings, coeffs.tolist()), states).toarray()
        assert matrix.shape == (states.shape[0], states.shape[0])
        from_hproj = float(np.linalg.eigvalsh(matrix.real)[0])
        from_sqd = eigval_of(strings, coeffs, states)
        reference = lowest_projected(strings, coeffs, states)
        assert from_hproj == pytest.approx(reference, abs=1e-9)
        assert from_sqd == pytest.approx(reference, abs=1e-6)

    def test_shape_is_subspace_dim_when_top_column_unreachable(self):
        """``coo_array((data, (rows, cols)))`` was built with no ``shape=``.

        scipy then infers the extent from the largest index actually present, so any trailing
        basis state that no term couples into is dropped from the matrix entirely. Here qubit-0
        flip couples states 0<->1 but the partner of the highest state is absent from the
        subspace, so its column never appears: measured (2, 2) for a 3-state subspace before the
        fix. The same shortfall was seen as 41x41 for a 53-state subspace with this repo's local
        two-site ``js`` operators.

        This failed *silently* -- a truncated matrix is still a valid symmetric matrix, so
        ``eigvalsh`` returns a plausible wrong ground energy rather than raising. The reference
        is the dense Kronecker projection, which shares no code with the packing path.
        """
        strings = ["IIIIIX"]
        coeffs = np.array([1.0])
        states = np.array(
            [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0]], dtype=np.uint8
        )
        matrix = hproj((strings, coeffs.tolist()), states)
        assert matrix.shape == (3, 3), "trailing uncoupled basis state was dropped"
        expected = project_dense(strings, coeffs, states)
        assert np.abs(matrix.toarray() - expected.real).max() < 1e-12

    def test_empty_projection_returns_zero_matrix(self):
        """No in-subspace matrix element at all must give a zero matrix, not a raise.

        With zero surviving elements both index arrays are empty, and scipy cannot infer any
        extent from them: measured ``ValueError: cannot infer dimensions from zero sized index
        arrays`` before the fix. A fully off-diagonal operator on a subspace closed under none of
        its terms is a legitimate (if degenerate) input -- the projection is genuinely zero.
        """
        strings = ["XXXXXX"]
        coeffs = np.array([1.0])
        states = np.array([[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1]], dtype=np.uint8)
        matrix = hproj((strings, coeffs.tolist()), states)
        assert matrix.shape == (2, 2)
        assert matrix.nnz == 0
        expected = project_dense(strings, coeffs, states)
        assert np.abs(matrix.toarray() - expected.real).max() < 1e-12

    def test_is_symmetric(self):
        """A real Hermitian projection must come back exactly symmetric.

        ``eigsh``/``eigvalsh`` read one triangle only, so an asymmetric result would be silently
        half-ignored rather than raising.
        """
        rng = np.random.default_rng(20260804)
        strings = real_pauli_strings(4, 5, rng)
        coeffs = rng.normal(size=len(strings))
        states = rng.integers(0, 2, size=(12, 4)).astype(np.uint8)
        matrix = hproj((strings, coeffs.tolist()), states).toarray()
        assert np.abs(matrix - matrix.T).max() == 0.0
