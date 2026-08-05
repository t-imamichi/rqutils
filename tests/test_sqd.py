"""Tests for :mod:`rqutils.sqd`.

Organized by defect, like ``test_ground_locg.py``. Three bugs were found while writing this suite
and fixed alongside it; each has a test named for it that reproduces the exact input and records the
measured wrong value, so a regression names itself:

- ``compute_diagonal``'s ``ibit = iterm & 255`` (should be ``& 7``), which made the
  ``cache_level[1] == 1`` kernels wrong once an X group held more than 8 Z terms.
- ``hproj`` building the Hamiltonian *with* the signature pad bit while packing states *without* it,
  so its bit alignment disagreed with the ``sqd`` path. The padding was an opt-in ``add_padding``
  flag then; it is now intrinsic to ``PauliSumXZ``, so the two sides cannot disagree.
- ``run_sqd``'s one-hot initial vectors, which cannot leave the connected component of the
  projected Hamiltonian that contains the seed, and which violate ``ground_locg``'s
  non-vanishing-overlap precondition outright when the seed state is decoupled.

The reference eigenvalue always comes from ``conftest.lowest_projected``: a dense ``2**n``
Kronecker construction that shares no code with the packing/uniquification/matvec chain under test.
Cross-kernel agreement is asserted too, but it is deliberately not the only check -- the two
initial-vector bugs affected all six kernels identically, so a consistency-only suite would have
passed while every kernel returned the same wrong number.
"""

import numpy as np
import pytest
from conftest import lowest_projected, project_dense, real_pauli_strings

from rqutils.sqd import (
    apply_h,
    compute_diagonal,
    get_diag_signs,
    get_diagonal,
    get_xsource,
    hproj,
    run_sqd,
    sqd,
    uniquify_states,
)

# Every (source_indices, diagonals) combination, i.e. all six matvec kernels.
CACHE_LEVELS = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]


def pack_padded(states):
    """Pack states with the leading zero pad bit, exactly as ``sqd`` does internally."""
    return np.packbits(np.pad(np.asarray(states, dtype=np.uint8), {1: (1, 0)}), axis=1)


def eigval_of(pauli_strings, coeffs, states, **kwargs):
    """Return ``sqd``'s eigenvalue as a plain float, whatever shape it comes back as."""
    result = sqd((pauli_strings, list(coeffs)), states, return_eigvec=False, **kwargs)
    return float(np.asarray(result).ravel()[0])


class TestComputeDiagonal:
    """``compute_diagonal`` composes a diagonal from packed sign bits and coefficients."""

    @pytest.mark.parametrize("num_terms", [1, 7, 8, 9, 17])
    def test_matches_direct_sum(self, num_terms):
        """The bit offset must wrap at 8, not at 256.

        ``ibit = iterm & 255`` made the shift ``7 - ibit`` negative from ``iterm == 8`` onward, so
        every term past the first byte read garbage. Measured on 9 terms: 0.71 absolute error. The
        parametrization brackets the byte boundary deliberately -- 7 passes even with the bug, 8 is
        the first failure, and 17 crosses into a third byte.
        """
        rng = np.random.default_rng(20260804)
        num_states = 6
        negative = rng.integers(0, 2, size=(num_states, num_terms)).astype(np.uint8)
        packed = np.zeros((num_states, (num_terms + 7) // 8), dtype=np.uint8)
        for term in range(num_terms):
            packed[:, term // 8] |= (negative[:, term] << (7 - (term % 8))).astype(np.uint8)
        coeffs = rng.normal(size=num_terms)

        got = np.asarray(compute_diagonal(packed, coeffs))
        expected = ((1.0 - 2.0 * negative) * coeffs).sum(axis=1)
        assert np.abs(got - expected).max() < 1e-13

    def test_agrees_with_get_diagonal(self):
        """The two diagonal paths (``cache_level[1]`` 1 vs 2) must produce the same numbers.

        ``get_diagonal`` recomputes signs from the Z signatures; ``compute_diagonal`` reads them
        from precomputed packed bits. They are the same quantity by two routes, so this pins the
        pair together independently of any end-to-end solve.
        """
        rng = np.random.default_rng(20260804)
        num_qubits = 6
        # Pure-Z strings all share the all-identity X signature, which is how one X group ends up
        # holding many Z terms -- the regime the & 255 bug lived in.
        strings = ["I" * num_qubits]
        while len(strings) < 11:
            candidate = "".join(rng.choice(["I", "Z"], size=num_qubits))
            if candidate not in strings:
                strings.append(candidate)
        coeffs = rng.normal(size=len(strings))
        states = rng.integers(0, 2, size=(20, num_qubits)).astype(np.uint8)

        from rqutils.paulis.symplectic import PauliSumXZ

        hamiltonian = PauliSumXZ.from_paulisum((strings, coeffs.tolist()))
        states_p = pack_padded(states)
        states_u = uniquify_states(states_p, states_p.shape[0])
        assert hamiltonian.z.shape[0] == 1, "expected a single X group for pure-Z input"

        signs = get_diag_signs(hamiltonian.z[0], states_u)
        from_signs = np.asarray(compute_diagonal(signs, hamiltonian.c[0]))
        direct = np.asarray(get_diagonal(hamiltonian.z[0], hamiltonian.c[0], states_u).real)
        assert np.abs(from_signs - direct).max() < 1e-13


class TestGetXsource:
    """``get_xsource`` finds, for each state, the index of ``state ^ xsignature``."""

    def test_partner_indices_and_absent_marker(self):
        """A partner outside the subspace must be ``-1``, which the matvec turns into a zero.

        Also pins the bit alignment between a padded Hamiltonian signature and padded states: an
        off-by-one there sends every matrix element to the wrong column, which is exactly the
        ``hproj`` bug in :class:`TestHproj`.
        """
        from rqutils.paulis.symplectic import PauliSumXZ

        states = np.array([[0, 0, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0]], dtype=np.uint8)
        states_u = uniquify_states(pack_padded(states), 3)
        # IIIX flips the last qubit: state 0 <-> state 1, and state 2's partner is absent.
        hamiltonian = PauliSumXZ.from_paulisum((["IIIX"], [1.0]))
        assert np.array_equal(np.asarray(get_xsource(hamiltonian.x[0], states_u)), [1, 0, -1])
        # XIII flips the first qubit: state 0 <-> state 2, and state 1's partner is absent.
        hamiltonian = PauliSumXZ.from_paulisum((["XIII"], [1.0]))
        assert np.array_equal(np.asarray(get_xsource(hamiltonian.x[0], states_u)), [2, -1, 0])

    @pytest.mark.parametrize("num_qubits", [6, 7, 8, 15, 16, 23, 24, 55, 63, 64, 71, 80])
    def test_matches_dense_partner_map_across_byte_widths(self, num_qubits):
        """Defect: a source-index array that is a *permutation* of the right answer.

        ``get_xsource`` is a binary search into a lex-sorted ``S``, with a ``uint64``-key fast path
        for ``B <= 8`` bytes and a lexicographic fallback beyond. Two ways that goes wrong silently:

        - Packing bytes in the wrong significance order makes integer order disagree with row lex
          order, so the search lands on a *different but valid* index. Every consumer still gets a
          finite number and the projected matrix stays symmetric.
        - At ``B > 8`` a ``uint64`` key cannot hold the row, so distinct states alias onto one key
          and unrelated states are reported as partners. This is why the width boundary is a
          correctness check and not a tuning knob.

        The reference is a dict from state bytes to row index, built with plain Python -- it shares
        no code with the packing, the search, or the sort the search replaced. The parametrization
        straddles every byte boundary the packing crosses: ``n+1`` at 7/8/9 bits, 15/16/17, and the
        ``B = 8`` -> ``B = 9`` transition at ``n = 63``/``64`` where the fast path must hand over to
        the fallback.
        """
        from rqutils.paulis.symplectic import PauliSumXZ

        rng = np.random.default_rng(num_qubits)
        # The subspace is CONSTRUCTED to be partly closed under flipping the last qubit, not sampled
        # and hoped over. Two reasons, both learned by getting it wrong:
        #
        # - If no signature has a partner inside the subspace, the expected answer is "-1
        #   everywhere" and an implementation that finds nothing agrees with it. A random wide
        #   signature essentially never has a partner, so the test would pass vacuously.
        # - The variation is concentrated in the TRAILING bytes of the packed row so the leading 8
        #   bytes collide across states. That is what makes a truncating uint64 key alias distinct
        #   states, which is the ``B > 8`` failure this parametrization exists to catch. Note the
        #   orientation: ``packbits`` fills from the most significant end and the pad bit is at
        #   position 0, so *low* qubit indices land in the *leading* bytes -- the reverse of the
        #   little-endian qubit numbering.
        nvary = min(num_qubits, 12)
        base = np.zeros((200, num_qubits), dtype=np.uint8)
        base[:, num_qubits - nvary :] = rng.integers(0, 2, size=(200, nvary), dtype=np.uint8)
        # Include each state's last-qubit partner for half the rows, so partners both exist (those)
        # and are absent (the rest).
        partners = base[: base.shape[0] // 2].copy()
        partners[:, -1] ^= 1
        states = np.unique(np.concatenate([base, partners], axis=0), axis=0)
        states_p = pack_padded(states)
        states_u = np.asarray(uniquify_states(states_p, states_p.shape[0]))

        # Independent reference: explicit lookup table over the packed rows.
        row_of = {row.tobytes(): i for i, row in enumerate(states_u)}

        # At least one signature must have partners that genuinely EXIST in the subspace, or the
        # expected answer is "-1 everywhere" and any implementation returning nothing agrees with
        # it. A random wide signature almost never has a partner in a sampled subspace, so include a
        # low-weight one -- flipping one of the last qubits keeps ~half the pairs inside a subspace
        # whose variation lives in those bits.
        low_weight = "I" * (num_qubits - 1) + "X"
        labels = [low_weight]
        labels += ["".join(rng.choice(list("IXYZ"), size=num_qubits)) for _ in range(2)]
        hamiltonian = PauliSumXZ.from_paulisum((labels, [1.0] * len(labels)))
        # Guard the guard: if no signature couples anything, this test proves nothing.
        n_present = max(
            int(np.sum(np.asarray(get_xsource(np.asarray(xs), states_u)) >= 0))
            for xs in hamiltonian.x
        )
        assert n_present > 0, (
            f"n={num_qubits}: no signature has any partner in the subspace, so the expected "
            "answer is -1 everywhere and this test cannot distinguish implementations"
        )

        for xsig in hamiltonian.x:
            got = np.asarray(get_xsource(np.asarray(xsig), states_u))
            expected = np.array(
                [row_of.get(np.bitwise_xor(row, xsig).tobytes(), -1) for row in states_u],
                dtype=np.int32,
            )
            # Fill-in rows (all-255) have no source; both sides agree they are absent, but only the
            # sign is contractual there -- see the note in get_xsource's docstring.
            is_fill = states_u[:, 0] == 255
            assert np.array_equal(got[~is_fill], expected[~is_fill]), (
                f"n={num_qubits} B={states_u.shape[1]}: index array disagrees with the dense "
                f"partner map on {int(np.sum(got[~is_fill] != expected[~is_fill]))} valid rows"
            )
            assert np.all(got[is_fill] < 0), "fill rows must report no source"

    def test_absent_source_is_negative_not_wrapped(self):
        """Defect: an absent source that indexes a real vector entry instead of gathering zero.

        The contract consumers rely on is only that an absent source is *negative*, since
        ``apply_xgrp`` gathers with ``wrap_negative_indices=False``. A non-negative sentinel (0, or
        ``N``) would silently add a spurious matrix element. Pinned here because the sort-based
        implementation returned assorted negatives while the search returns exactly -1, so the
        *sign* is the invariant, not the value.
        """
        from rqutils.paulis.symplectic import PauliSumXZ

        # Single state whose IIIX partner (0001) is not in the subspace.
        states_u = uniquify_states(pack_padded(np.array([[0, 0, 0, 0]], dtype=np.uint8)), 1)
        hamiltonian = PauliSumXZ.from_paulisum((["IIIX"], [1.0]))
        got = np.asarray(get_xsource(hamiltonian.x[0], states_u))
        assert got[0] < 0, f"absent source must be negative, got {got[0]}"


class TestUniquifyStates:
    """``uniquify_states`` sorts, deduplicates, and pads to a fixed size with 255 fillers."""

    def test_dedupes_and_marks_fillers(self):
        """Filler slots must be detectable via ``states_u[:, 0] >> 7``.

        That marker is what lets ``run_sqd`` keep fillers out of the argmin and out of the initial
        vector; if fillers were indistinguishable from real states the solver could place weight
        outside the subspace.
        """
        states = np.array([[0, 1], [1, 0], [0, 1], [1, 1]], dtype=np.uint8)
        states_p = pack_padded(states)
        num_unique = len(np.unique(states_p, axis=0))
        assert num_unique == 3
        out = np.asarray(uniquify_states(states_p, 6))
        assert out.shape[0] == 6
        real = out[(out[:, 0] >> 7) == 0]
        assert real.shape[0] == num_unique
        assert np.array_equal(real, np.unique(states_p, axis=0)), "real slots must be sorted-unique"
        assert np.all(out[(out[:, 0] >> 7) == 1] == 255)


class TestHproj:
    """``hproj`` builds the projected Hamiltonian densely (sparse), as a debug/reference path."""

    def test_unsorted_input_with_unique_states_is_wrong(self):
        """Defect: ``unique_states=True`` accepts unsorted states and returns a wrong matrix.

        ``get_xsource`` requires a lex-sorted ``S``. Both production callers satisfy it -- ``run_sqd``
        via ``uniquify_states``, ``hproj`` via ``np.unique(..., axis=0)`` -- but ``hproj``'s
        ``unique_states=True`` shortcut skips that ``np.unique``, so a caller who has already
        deduplicated *without* sorting silently violates the precondition.

        This predates the search-based ``get_xsource``: the sort-based implementation was equally
        wrong on unsorted input, just wrong differently. The test pins the *detectable signature*
        rather than the specific wrong numbers, since those are implementation-dependent: the
        returned matrix is **not symmetric**, which a Hermitian projection must always be. It exists
        so that anyone tightening this into a raise has a named test to flip, and so the asymmetry is
        not mistaken for a regression in the search.
        """
        # Unique but NOT sorted: row order is 1000, 0000, 0001.
        unsorted = np.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]], dtype=np.uint8)
        bad = hproj((["IIIX"], [1.0]), unsorted, unique_states=True).toarray().real
        assert not np.allclose(bad, bad.T), (
            "unsorted input happened to give a symmetric matrix; this test's premise is stale"
        )

        # The same states, sorted, give the correct symmetric coupling 0000 <-> 0001.
        good = hproj((["IIIX"], [1.0]), np.unique(unsorted, axis=0), unique_states=True)
        good = good.toarray().real
        assert np.allclose(good, good.T)
        expected = np.zeros((3, 3))
        expected[0, 1] = expected[1, 0] = 1.0
        assert np.allclose(good, expected)

    def test_matches_dense_reference(self):
        """``hproj`` packed states WITHOUT the pad bit while padding the Hamiltonian.

        ``PauliSumXZ`` shifts every X/Z signature one bit right for the pad bit, so unpadded states
        disagree with them on alignment and every matrix element lands in the wrong column. Measured
        before the fix on this input: lowest eigenvalue -1.398 against a true -2.191.
        ``examples/_bench_common`` had worked around it by not using ``hproj`` at all, noting it
        "raises a shape-mismatch TypeError". The padding was an opt-in ``add_padding`` flag at the
        time, which is what let the two sides disagree; it is now unconditional.
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
        states = np.unique(rng.integers(0, 2, size=(7, num_qubits)).astype(np.uint8), axis=0)

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


class TestSqdInitialVector:
    """``run_sqd``'s initial vector must have a non-vanishing overlap with the ground state.

    A one-hot seed cannot leave the connected component of the projected Hamiltonian that contains
    it: Krylov iteration only reaches states linked by a nonzero matrix element. Both of the
    original seeds were one-hots, so both could return a genuine eigenvalue that was not the lowest
    -- with ``converged=True``, and with nothing downstream able to notice.
    """

    @pytest.mark.parametrize("pauli", ["IIIX", "XXII", "YYII", "XIII"])
    def test_decoupled_seed_state(self, pauli):
        """``vinit_nodiag``'s one-hot at index 0, where state 0 is decoupled.

        For this 9-state subspace every ``xsource`` out of state 0 lands outside, so row 0 of the
        projected Hamiltonian is identically zero and ``e_0`` is a true eigenvector with eigenvalue
        0. The zero-residual guard in ``ground_locg`` then correctly reports convergence and ``sqd``
        returned 0.0 against a true -1.0. ``XIII`` is included as the control: its seed state is
        coupled, so it passed even before the fix.
        """
        rng = np.random.default_rng(11)
        states = np.unique(rng.integers(0, 2, size=(12, 4)).astype(np.uint8), axis=0)
        reference = lowest_projected([pauli], [1.0], states)
        assert eigval_of([pauli], [1.0], states) == pytest.approx(reference, abs=1e-10)

    def test_disconnected_components(self):
        """``vinit_from_min_diag``'s one-hot at the minimum-diagonal index, in a split subspace.

        The projected Hamiltonian here splits into two components of size 4 and 10. The
        minimum-diagonal state sits in the size-4 block, whose own minimum is -1.293; the true
        minimum is -2.191, in the other block. ``sqd`` returned -1.293 -- an exact eigenvalue of the
        projected Hamiltonian, just not the lowest one, which is what made it undetectable without
        an external reference.
        """
        rng = np.random.default_rng(3)
        num_qubits = 5
        strings = real_pauli_strings(num_qubits, 6, rng)
        coeffs = rng.normal(size=len(strings))
        states = rng.integers(0, 2, size=(20, num_qubits)).astype(np.uint8)

        # Verify the fixture really is disconnected, so this test cannot silently stop testing it.
        import scipy.sparse as sp

        matrix = project_dense(strings, coeffs, states).real
        num_components = sp.csgraph.connected_components(
            sp.csr_matrix(matrix != 0), directed=False
        )[0]
        assert num_components == 2, f"fixture is no longer disconnected ({num_components} blocks)"

        reference = lowest_projected(strings, coeffs, states)
        assert eigval_of(strings, coeffs, states) == pytest.approx(reference, rel=1e-10)


class TestSqdEndToEnd:
    """``sqd`` against an independent dense reference, over all six matvec kernels."""

    @pytest.mark.parametrize("cache_level", CACHE_LEVELS)
    def test_all_kernels_agree_with_reference(self, cache_level):
        """The six ``cache_level`` kernels are six routes to one number.

        They trade memory for speed and must be numerically interchangeable. Asserting each against
        the external reference (rather than only against each other) is what catches an error common
        to all six, which is precisely how both initial-vector bugs presented.
        """
        rng = np.random.default_rng(20260804)
        num_qubits = 5
        strings = real_pauli_strings(num_qubits, 7, rng)
        coeffs = rng.normal(size=len(strings))
        states = rng.integers(0, 2, size=(24, num_qubits)).astype(np.uint8)
        reference = lowest_projected(strings, coeffs, states)
        assert eigval_of(strings, coeffs, states, cache_level=cache_level) == pytest.approx(
            reference, rel=1e-10
        )

    @pytest.mark.parametrize("cache_level", CACHE_LEVELS)
    def test_many_z_terms_per_x_group(self, cache_level):
        """Pure-Z input puts every term in one X group, exercising the byte-boundary path.

        With 13 Z terms the ``cache_level[1] == 1`` kernels returned -6.520 against a true -8.699 --
        a 25% error in a physical eigenvalue, silently. The other four kernels agreed on -8.699,
        which is why cross-kernel comparison catches this one even without a reference.
        """
        rng = np.random.default_rng(2)
        num_qubits = 6
        strings = ["I" * num_qubits]
        while len(strings) < 13:
            candidate = "".join(rng.choice(["I", "Z"], size=num_qubits))
            if candidate not in strings:
                strings.append(candidate)
        coeffs = rng.normal(size=len(strings))
        states = rng.integers(0, 2, size=(40, num_qubits)).astype(np.uint8)
        reference = lowest_projected(strings, coeffs, states)
        assert eigval_of(strings, coeffs, states, cache_level=cache_level) == pytest.approx(
            reference, rel=1e-10
        )

    @pytest.mark.parametrize("seed", [5, 6, 7, 8])
    def test_random_hamiltonians(self, seed):
        """Aggregate check over seeded random input, supplementing the targeted cases."""
        rng = np.random.default_rng(seed)
        num_qubits = 5
        strings = real_pauli_strings(num_qubits, 7, rng)
        coeffs = rng.normal(size=len(strings))
        states = rng.integers(0, 2, size=(24, num_qubits)).astype(np.uint8)
        reference = lowest_projected(strings, coeffs, states)
        assert eigval_of(strings, coeffs, states) == pytest.approx(reference, rel=1e-10)

    def test_duplicate_states_are_uniquified(self):
        """Duplicated input states must not change the answer: the subspace is the same."""
        rng = np.random.default_rng(20260804)
        strings = real_pauli_strings(4, 5, rng)
        coeffs = rng.normal(size=len(strings))
        unique = np.unique(rng.integers(0, 2, size=(10, 4)).astype(np.uint8), axis=0)
        duplicated = np.concatenate([unique, unique[:3]], axis=0)
        assert eigval_of(strings, coeffs, duplicated) == pytest.approx(
            eigval_of(strings, coeffs, unique), rel=1e-10
        )

    def test_eigenvector_and_basis_states(self):
        """``return_eigvec=True`` must return an eigenvector over the returned basis states.

        The basis rows come back through ``np.unpackbits`` with the pad bit stripped, so this also
        pins that round-trip: a misaligned slice would return the wrong bitstrings for a correct
        eigenvector, which no eigenvalue check would notice.
        """
        rng = np.random.default_rng(20260804)
        num_qubits = 4
        strings = real_pauli_strings(num_qubits, 5, rng)
        coeffs = rng.normal(size=len(strings))
        states = np.unique(rng.integers(0, 2, size=(12, num_qubits)).astype(np.uint8), axis=0)

        eigval, eigvec, basis = sqd((strings, coeffs.tolist()), states, return_eigvec=True)
        assert basis.shape == (len(eigvec), num_qubits)
        assert np.array_equal(basis, np.unique(basis, axis=0)), "basis must be sorted-unique"
        assert np.array_equal(basis, np.unique(states, axis=0))
        assert np.linalg.norm(eigvec) == pytest.approx(1.0, rel=1e-8)

        # The returned pair must satisfy H v = lambda v on the matrix built over those same rows.
        matrix = project_dense(strings, coeffs, basis).real
        residual = np.linalg.norm(matrix @ eigvec - eigval * eigvec)
        assert residual < 1e-8 * max(1.0, np.abs(matrix).max())

    def test_states_size_padding_does_not_change_the_answer(self):
        """``states_size`` only pins array shapes to avoid JIT recompilation.

        Padding to a larger size adds 255-filler slots, which must stay out of the result entirely.
        """
        rng = np.random.default_rng(20260804)
        strings = real_pauli_strings(4, 5, rng)
        coeffs = rng.normal(size=len(strings))
        states = rng.integers(0, 2, size=(12, 4)).astype(np.uint8)
        baseline = eigval_of(strings, coeffs, states)
        for states_size in (16, 24):
            assert eigval_of(strings, coeffs, states, states_size=states_size) == pytest.approx(
                baseline, rel=1e-10
            )

    def test_states_size_below_input_length_raises(self):
        rng = np.random.default_rng(20260804)
        states = rng.integers(0, 2, size=(12, 4)).astype(np.uint8)
        with pytest.raises(ValueError, match="states_size smaller"):
            sqd((["ZIII"], [1.0]), states, states_size=4)

    def test_states_size_actually_prevents_recompilation(self):
        """``states_size`` pinned the internal arrays but not the input, so it never worked.

        ``sqd`` packed ``states`` to its raw length and handed that to ``run_sqd``, where
        ``states_p`` is a *traced* argument -- so its leading dimension entered the jit cache key and
        every distinct ``len(states)`` retraced the whole solver despite the pin. That is the exact
        thing the parameter is documented to prevent, and it failed silently: results stayed correct,
        only ~7x slower (measured 0.44 s per call versus 0.064 s once the shape repeats, n=16
        N=4096). The companion test above covers the numbers; this one covers the contract.

        Asserting on cache misses rather than wall-clock keeps it deterministic on a loaded machine.
        """
        rng = np.random.default_rng(20260804)
        strings = real_pauli_strings(4, 5, rng)
        coeffs = rng.normal(size=len(strings))
        states = rng.integers(0, 2, size=(12, 4)).astype(np.uint8)
        states_size = 16

        # Warm the cache at the pinned shape, then count misses across shorter inputs.
        eigval_of(strings, coeffs, states, states_size=states_size)
        before = run_sqd._cache_size()
        for length in (11, 10, 9):
            eigval_of(strings, coeffs, states[:length], states_size=states_size)
        assert run_sqd._cache_size() == before, (
            "run_sqd retraced for a shorter input despite states_size being pinned"
        )


class TestMatvecKernels:
    """The matvec kernels, checked directly against a dense matrix-vector product."""

    def test_apply_h_matches_dense(self):
        """``apply_h`` (no caching) is the reference kernel the cached ones must match."""
        from rqutils.paulis.symplectic import PauliSumXZ

        rng = np.random.default_rng(20260804)
        num_qubits = 4
        strings = real_pauli_strings(num_qubits, 5, rng)
        coeffs = rng.normal(size=len(strings))
        states = np.unique(rng.integers(0, 2, size=(12, num_qubits)).astype(np.uint8), axis=0)

        hamiltonian = PauliSumXZ.from_paulisum((strings, coeffs.tolist()))
        states_u = uniquify_states(pack_padded(states), states.shape[0])
        matrix = project_dense(strings, coeffs, states).real
        vector = rng.normal(size=states.shape[0])

        got = np.asarray(
            apply_h(vector, (hamiltonian.x, hamiltonian.z, hamiltonian.c), states_u, (0, 0))
        ).real
        assert np.abs(got - matrix @ vector).max() < 1e-12

    @pytest.mark.parametrize("cache_level", CACHE_LEVELS)
    def test_every_cache_level_matches_dense(self, cache_level):
        """All six resolution paths of the unified kernel, each against the dense product.

        ``apply_h`` replaced six near-identical functions with one ``cache_level``-indexed kernel.
        The risk in that collapse is a mis-wired argument slot -- feeding a Z signature where a
        coefficient belongs, say -- which would still produce a plausible finite vector. Checking
        every cell of the 2x3 grid against ``project_dense`` (an independent Kronecker construction)
        rather than against the other kernels is what catches it: cross-kernel agreement alone would
        pass if the collapse broke all six identically.
        """
        from rqutils.paulis.symplectic import PauliSumXZ

        rng = np.random.default_rng(20260805)
        num_qubits = 4
        strings = real_pauli_strings(num_qubits, 6, rng)
        coeffs = rng.normal(size=len(strings))
        states = np.unique(rng.integers(0, 2, size=(12, num_qubits)).astype(np.uint8), axis=0)

        hamiltonian = PauliSumXZ.from_paulisum((strings, coeffs.tolist()))
        states_u = uniquify_states(pack_padded(states), states.shape[0])
        vector = rng.normal(size=states.shape[0])
        matrix = project_dense(strings, coeffs, states).real

        xgroup = hamiltonian.x
        if cache_level[0] == 1:
            xgroup = np.stack([np.asarray(get_xsource(x, states_u)) for x in hamiltonian.x])
        match cache_level[1]:
            case 0:
                scanned = (xgroup, hamiltonian.z, hamiltonian.c)
            case 1:
                signs = np.stack([np.asarray(get_diag_signs(z, states_u)) for z in hamiltonian.z])
                scanned = (xgroup, signs, hamiltonian.c)
            case 2:
                diagonals = np.stack(
                    [
                        np.asarray(get_diagonal(z, c, states_u).real)
                        for z, c in zip(hamiltonian.z, hamiltonian.c)
                    ]
                )
                scanned = (xgroup, diagonals)

        needs_states = cache_level[0] == 0 or cache_level[1] == 0
        got = np.asarray(
            apply_h(vector, scanned, states_u if needs_states else None, cache_level)
        ).real
        assert np.abs(got - matrix @ vector).max() < 1e-12

    @pytest.mark.parametrize("cache_level", [(0, 0), (0, 1), (0, 2), (1, 0)])
    def test_omitting_states_raises(self, cache_level):
        """Only ``(1, 2)`` can run without the state list; the rest must say so, not crash later.

        ``(1, 1)`` and ``(1, 2)`` read neither signature array, which is what lets a caller drop S
        after caching. For the other four, a missing S would otherwise surface as an opaque failure
        deep inside ``get_xsource``/``get_diagonal``.
        """
        with pytest.raises(ValueError, match="states is required"):
            apply_h(np.zeros(4), (np.zeros((1, 1), dtype=np.uint8),) * 3, None, cache_level)

    def test_fully_cached_level_matches_dense(self):
        """``cache_level=(1, 2)``, the level ``examples/bench_mlx.py`` and the MLX port mirror.

        Overlaps :meth:`test_every_cache_level_matches_dense` by design: this one fixes the input
        that ``ground_locg_mlx.apply_h_xz_mlx`` was validated against, so it stays a named pin for
        the ported kernel even as the grid test's parametrization changes.
        """
        from rqutils.paulis.symplectic import PauliSumXZ

        rng = np.random.default_rng(20260804)
        num_qubits = 4
        strings = real_pauli_strings(num_qubits, 5, rng)
        coeffs = rng.normal(size=len(strings))
        states = np.unique(rng.integers(0, 2, size=(12, num_qubits)).astype(np.uint8), axis=0)

        hamiltonian = PauliSumXZ.from_paulisum((strings, coeffs.tolist()))
        states_u = uniquify_states(pack_padded(states), states.shape[0])
        xsources = np.stack([np.asarray(get_xsource(x, states_u)) for x in hamiltonian.x])
        diagonals = np.stack(
            [
                np.asarray(get_diagonal(z, c, states_u).real)
                for z, c in zip(hamiltonian.z, hamiltonian.c)
            ]
        )
        matrix = project_dense(strings, coeffs, states).real
        vector = rng.normal(size=states.shape[0])

        got = np.asarray(apply_h(vector, (xsources, diagonals), None, (1, 2))).real
        assert np.abs(got - matrix @ vector).max() < 1e-12
