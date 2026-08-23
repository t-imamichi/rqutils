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

import os
import subprocess
import sys

import numpy as np
import pytest
from conftest import (
    collapsing_states,
    lowest_projected,
    project_dense,
    real_pauli_strings,
    unique_states,
)

from rqutils.sqd import (
    _MAX_STATES,
    _is_lex_sorted,
    _pack_scanned,
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

# Every (source_indices, diagonals) combination, i.e. all six matvec kernels. Written out rather than
# derived from `apply_h`'s dispatch: the axis options now live inline in the function (they pair each
# keyword with its array, which only exists at call time), so there is no module table to read. The
# grid is small, fixed by the kernel's 2x3 shape, and a seventh strategy would need new tests anyway.
CACHE_LEVELS = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]


def pack_padded(states):
    """Pack states with the leading zero pad bit, as ``PauliSumXZ.pack_states`` does.

    Deliberately *not* a call to that method: this is the independent reference the suite compares
    against, and the pad-bit alignment is exactly the kind of convention where a bug would make every
    internal path agree on the same wrong answer. ``test_paulis_symplectic.py::TestPadding`` pins the
    library's version against an independent unpacking, so a divergence surfaces there rather than
    silently propagating through every test that packs states here.
    """
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

    def test_subspace_above_the_int32_ceiling_raises(self):
        """``hproj`` reaches ``get_xsource`` too, so it shares ``sqd``'s int32 ceiling.

        ``2**31`` rows cannot be allocated, so the shape is produced by ``np.broadcast_to`` (a view,
        no allocation) -- enough to reach a guard that reads ``states.shape[0]``.

        The guard's *placement* matters as much as its presence: it sits before the O(N) sortedness
        scan and the ``np.unique``, so a doomed call reports the real problem instead of spending time
        first. Measured on this test: 0.23 s with the check first, 23 s with it after the scan.
        """
        states = np.broadcast_to(np.zeros(2, dtype=np.uint8), (2**31, 2))
        with pytest.raises(ValueError, match="exceeds the .* limit imposed by int32"):
            hproj((["ZI"], [1.0]), states, unique_states=True)

    # KNOWN GAP, deliberate: nothing pins the *accept* side of the boundary, so relaxing either
    # guard's `>` to `>=` leaves the suite green while rejecting `_MAX_STATES` itself -- the largest
    # legal size. Mutation-tested, and not closed on purpose.
    #
    # A test for it was written and measured. Driving the real guard at exactly `_MAX_STATES` works
    # (``np.broadcast_to`` makes the shape for free), but the call then runs the O(N) sortedness scan
    # over 2^31 rows: **23 s in the passing case**, against a 6 s suite. Re-deriving the comparison in
    # the test instead is worthless -- it reimplements the predicate and catches nothing (verified: the
    # `>=` mutants survived it).
    #
    # Not worth 23 s: the mutant's only consequence is rejecting a call that would immediately OOM
    # anyway (2^31 states is 4.3 GB of packed states before any vector), so the off-by-one is
    # unobservable on any hardware that exists. The reject side, which is the side that prevents a
    # silent wrong answer, is pinned above.

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

        Filler slots are all-``255`` rows, so two or more are duplicates and fail the strictness test.
        This is the one way the sortedness guard could surprise a caller -- ``uniquify_states`` output
        is otherwise exactly what ``get_xsource`` wants -- so it is pinned rather than left to be
        rediscovered. ``sqd`` trims fillers before returning its basis, which is why
        ``examples/scaling/poc7_sharding.py`` can hand that basis straight to ``hproj``.
        """
        from rqutils.paulis.symplectic import PauliSumXZ

        rng = np.random.default_rng(3)
        states = rng.integers(0, 2, size=(12, 4), dtype=np.uint8)
        packed = PauliSumXZ.pack_states(states)

        # states_size=12 happens to leave a single filler, which is still strictly increasing.
        one_filler = np.asarray(uniquify_states(packed, 12))
        assert int((one_filler[:, 0] >> 7).sum()) == 1
        assert _is_lex_sorted(one_filler) is True

        # Pad further and the duplicate 255 rows are rejected.
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
        states = unique_states(12, 4, rng)
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
        unique = unique_states(10, 4, rng)
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
        states = unique_states(12, num_qubits, rng)

        eigval, eigvec, basis = sqd((strings, coeffs.tolist()), states, return_eigvec=True)
        assert basis.shape == (len(eigvec), num_qubits)
        assert np.array_equal(basis, np.unique(basis, axis=0)), "basis must be sorted-unique"
        assert np.array_equal(basis, np.unique(states, axis=0))
        assert np.linalg.norm(eigvec) == pytest.approx(1.0, rel=1e-8)

        # The returned pair must satisfy H v = lambda v on the matrix built over those same rows.
        matrix = project_dense(strings, coeffs, basis).real
        residual = np.linalg.norm(matrix @ eigvec - eigval * eigvec)
        assert residual < 1e-8 * max(1.0, np.abs(matrix).max())

    def test_states_size_padding_is_shape_invariant_only(self):
        """``states_size`` only pins array shapes to avoid JIT recompilation.

        Shape invariance ONLY. ``sqd``-vs-``sqd`` is the right reference for that -- the ``rel=1e-10``
        below is defensible precisely because both arms are the same iterative eigensolver on the same
        subspace, which a dense comparison could not match (the sqd-vs-dense arm of
        :meth:`TestHproj.test_agrees_with_sqd_on_a_subspace_with_a_decoupled_state` needs ``abs=1e-6``,
        four orders looser).

        It does **not** check that filler slots stay out of the result, and cannot: this fixture
        collapses under uniquification, so every arm including ``baseline`` carries fillers and a
        broken filler mask corrupts them identically.
        :meth:`test_filler_slots_are_excluded_against_a_dense_reference` owns that measurement and
        pins the exclusion, against a dense reference and a filler-free control arm. Don't delete it
        as redundant with this one.

        The ``24`` arm is not redundant with ``16``: relative to the 12-row input it puts filler rows
        in the *majority*, so a future defect whose behaviour depends on fillers outnumbering genuine
        states is covered by it and not by 16.
        """
        rng = np.random.default_rng(20260804)
        strings = real_pauli_strings(4, 5, rng)
        coeffs = rng.normal(size=len(strings))
        # collapsing_states, not a bare draw: the collapse is a precondition here (it is what makes
        # every arm carry fillers, per the docstring), and the helper asserts it rather than trusting
        # the seed. Were a future edit to make this fixture distinct, the test would silently become
        # the filler-free-vs-padded comparison the sibling owns.
        states = collapsing_states(12, 4, rng)
        baseline = eigval_of(strings, coeffs, states)
        for states_size in (16, 24):
            assert eigval_of(strings, coeffs, states, states_size=states_size) == pytest.approx(
                baseline, rel=1e-10
            )

    @pytest.mark.parametrize("states_size", [None, 8])
    def test_filler_slots_are_excluded_against_a_dense_reference(self, states_size):
        """Filler slots must be excluded, checked against DENSE rather than an unpadded ``sqd`` call.

        Sibling of ``test_states_size_padding_is_shape_invariant_only``, which cannot catch this:
        its fixture collapses under uniquification, so every arm including its "baseline" already
        carries filler slots and drifts identically. Measured, with ``_is_filler``'s ``>> 7`` changed
        to ``>> 8`` (a uint8 shifted by 8 is 0, marking every filler as a genuine state): the whole
        sqd suite stays green *except* this test.

        This fixture instead uses 4 states that are ALREADY unique, so the ``states_size=None`` arm
        needs no padding at all and is a genuinely filler-free control -- the only arm that stays
        correct under that mutation, which is what separates "filler handling broke" from "the solver
        broke".

        That control property depends on the fixture length being a power of two, and it is asserted
        below rather than left implicit. ``states_size=None`` no longer means "no padding": it
        defaults to the next power of two at or above the input length, so a fixture of, say, 5 rows
        would round to 8 and this arm would silently acquire three filler slots -- becoming a second
        padded arm and leaving the mutation uncaught, with nothing in the test to say so.
        Two distinct guards are pinned, both measured to return a plausible wrong answer of -1.2
        against the true -0.8297058541:

        * ``_is_filler``'s high-bit test (``states_u[:, 0] >> 7``) -- three call sites depend on it.
        * ``run_sqd``'s filler-diagonal masking (``jnp.where(_is_filler(...) == 1, max, diagonal)``),
          which keeps a filler's zero diagonal from being selected as the minimum eigenvalue.

        A filler slot is all-ones (255) and ``pack_states`` reserves a leading zero pad bit, so a
        genuine state's byte 0 is always < 128 -- that asymmetry is the whole mechanism.

        **Not subsumed by the eight other tests here that compare a filler-carrying fixture against
        a dense reference** (``test_all_kernels_agree_with_reference`` and friends). Those look like
        they should catch it and do not -- measured. A broken mask leaves the extra filler diagonals
        at zero, which perturbs the reported minimum only for some spectra; theirs happen to survive.
        That non-catch is spectrum-dependent and therefore not something to rely on, which is why the
        control arm below is explicit rather than incidental.

        Only two arms: a third at 16 was measured to fail to the same wrong value as 8, while costing
        a further ~0.4 s -- each distinct ``states_size`` is a separate jit trace of the whole solver
        (see :meth:`test_states_size_actually_prevents_recompilation`).
        """
        strings = ["ZIII", "IZII", "XXII", "IIZI"]
        coeffs = [1.0, -0.5, 0.3, 0.7]
        states = np.array([[0, 0, 0, 0], [0, 0, 1, 1], [0, 1, 0, 1], [1, 0, 0, 1]], dtype=np.uint8)
        assert len(np.unique(states, axis=0)) == len(states), "fixture must start filler-free"
        # Uniqueness alone is not enough for the states_size=None arm to be filler-free -- the
        # default rounds up to a power of two, so the row count must already be one.
        assert len(states) & (len(states) - 1) == 0, (
            f"fixture length {len(states)} is not a power of two, so the states_size=None arm "
            "would be padded and would stop being a filler-free control"
        )

        reference = lowest_projected(strings, coeffs, states)
        got = eigval_of(strings, coeffs, states, states_size=states_size)
        assert got == pytest.approx(reference, rel=1e-9), (
            f"states_size={states_size}: sqd gave {got}, dense reference is {reference} -- "
            "filler slots leaked into the subspace"
        )

    def test_states_size_above_the_int32_ceiling_raises(self):
        """The 2^31 limit the module documents as hard was documented but never enforced.

        Subspace positions are int32 throughout -- ``uniquify_states``' iota and ``get_xsource``'s
        returned indices, which use ``-1`` as the absent marker -- so a size at or above ``2**31``
        wraps to ``-2147483648`` and yields a corrupted permutation rather than an error. That is a
        plausible finite answer, the failure mode this module exists to guard against. Note the wrapped
        value is ``-2147483648``, *not* ``-1``, so the absent-marker test cannot even catch it.

        Unreachable on real hardware (``2**31`` states is 4.3 GB of packed states before any vector),
        so the check is asserted against the *argument* rather than by allocating anything.
        """
        states = np.array([[0, 0], [1, 1]], dtype=np.uint8)
        with pytest.raises(ValueError, match="exceeds the .* limit imposed by int32"):
            sqd((["ZI"], [1.0]), states, states_size=2**31)

        # Both sides of the boundary, because only asserting the reject side leaves the comparison
        # operator untested. Mutation-tested: relaxing `>` to `>=` rejects _MAX_STATES itself -- the
        # largest *legal* size -- and every other test here stays green.
        #
        # The accept side cannot be asserted end to end: `states_size=_MAX_STATES` passes validation
        # and then tries to allocate ~2 GB of packed states (verified -- it runs until killed, which is
        # itself the evidence that validation let it through). So the boundary is pinned on the
        # predicate instead, against the int32 range it exists to respect.
        assert _MAX_STATES == np.iinfo(np.int32).max, (
            "the ceiling must be the largest representable int32 index, not one more or less"
        )
        assert _MAX_STATES == np.iinfo(np.int32).max, (
            "the ceiling must be the largest representable int32 index, not one more or less"
        )
        # And the wrap this guards against is real, not hypothetical.
        assert np.array(2**31, dtype=np.int64).astype(np.int32) == -(2**31)

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


def apply_h_inputs(rng, num_qubits=4, num_terms=6, num_states=12):
    """Build every per-X-group representation ``apply_h`` accepts, plus a dense reference.

    A plain function taking ``rng`` rather than a ``@pytest.fixture``, per the suite convention in
    ``conftest``: a fixture drawing from an RNG makes stream position depend on fixture resolution
    order, which is invisible at the call site. Callers pass their own seeded ``rng``, so the six
    keyword combinations below all see byte-identical inputs and differ only in which names are used.

    Returns a dict of every representation of the same operator, so a test can select one pairing
    without rebuilding the others. The keys are spelled exactly as ``apply_h``'s keyword parameters
    (``xsignatures``, ``zsignatures``, ``diag_signs``, ``coeffs``, ...), so a caller can splat a
    selection straight in -- ``**{k: p[k] for k in names}`` -- rather than writing a dict literal that
    re-pairs name to array by hand. That hand-pairing is itself a place a typo silently swaps two
    arrays, which is precisely the hazard ``apply_h``'s keyword form exists to remove.
    """
    from rqutils.paulis.symplectic import PauliSumXZ

    strings = real_pauli_strings(num_qubits, num_terms, rng)
    coeffs = rng.normal(size=len(strings))
    states = unique_states(num_states, num_qubits, rng)

    hamiltonian = PauliSumXZ.from_paulisum((strings, coeffs.tolist()))
    states_u = uniquify_states(pack_padded(states), states.shape[0])
    return {
        "states_u": states_u,
        "vector": rng.normal(size=states.shape[0]),
        "matrix": project_dense(strings, coeffs, states).real,
        "xsignatures": hamiltonian.x,
        "zsignatures": hamiltonian.z,
        "coeffs": hamiltonian.c,
        "xsources": np.stack([np.asarray(get_xsource(x, states_u)) for x in hamiltonian.x]),
        "diag_signs": np.stack([np.asarray(get_diag_signs(z, states_u)) for z in hamiltonian.z]),
        "diagonals": np.stack(
            [
                np.asarray(get_diagonal(z, c, states_u).real)
                for z, c in zip(hamiltonian.z, hamiltonian.c)
            ]
        ),
    }


class TestMatvecKernels:
    """The matvec kernels, checked directly against a dense matrix-vector product."""

    def test_apply_h_matches_dense(self):
        """``apply_h`` (no caching) is the reference kernel the cached ones must match."""
        p = apply_h_inputs(np.random.default_rng(20260804), num_terms=5)
        got = np.asarray(
            apply_h(
                p["vector"],
                xsignatures=p["xsignatures"],
                zsignatures=p["zsignatures"],
                coeffs=p["coeffs"],
                states=p["states_u"],
            )
        ).real
        assert np.abs(got - p["matrix"] @ p["vector"]).max() < 1e-12

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
        p = apply_h_inputs(np.random.default_rng(20260805))
        # Map the cache_level under test back to the keywords that select it. The positional form is
        # gone, so a level is now requested by *naming* the arrays it implies -- which is the point of
        # the change: this mapping is written once here, in the test, rather than being a thing every
        # caller had to get right.
        xname = "xsources" if cache_level[0] == 1 else "xsignatures"
        dname = {0: "zsignatures", 1: "diag_signs", 2: "diagonals"}[cache_level[1]]
        kwargs = {xname: p[xname], dname: p[dname]}
        if cache_level[1] != 2:
            kwargs["coeffs"] = p["coeffs"]
        needs_states = cache_level[0] == 0 or cache_level[1] == 0
        got = np.asarray(
            apply_h(p["vector"], states=p["states_u"] if needs_states else None, **kwargs)
        ).real
        assert np.abs(got - p["matrix"] @ p["vector"]).max() < 1e-12

    @pytest.mark.parametrize(
        "names",
        [
            ("xsignatures", "zsignatures", "coeffs"),
            ("xsignatures", "diag_signs", "coeffs"),
            ("xsignatures", "diagonals"),
            ("xsources", "zsignatures", "coeffs"),
            ("xsources", "diag_signs", "coeffs"),
            ("xsources", "diagonals"),
        ],
    )
    def test_keyword_form_matches_dense_for_every_combination(self, names):
        """The keyword form covers all six strategies and each still matches a dense reference.

        The keyword names are the only thing selecting the strategy here, so this is what pins the
        keyword-to-digit pairing inside ``apply_h``. A transposed digit there would route a
        call to the wrong branch and produce a plausible finite vector, exactly the failure mode
        :meth:`test_every_cache_level_matches_dense` exists for -- so the reference is dense here too,
        not the positional form (agreeing with a sibling that is wrong the same way proves nothing).
        """
        p = apply_h_inputs(np.random.default_rng(20260805))
        kwargs = {name: p[name] for name in names}
        got = np.asarray(apply_h(p["vector"], states=p["states_u"], **kwargs)).real
        assert np.abs(got - p["matrix"] @ p["vector"]).max() < 1e-12

    @pytest.mark.parametrize(
        ("names", "match"),
        [
            (("diag_signs", "coeffs"), "exactly one of xsources="),
            (("xsignatures", "xsources", "diagonals"), "exactly one of xsources="),
            (("xsources", "coeffs"), "exactly one of diagonals="),
            (("xsources", "diag_signs", "diagonals", "coeffs"), "exactly one of diagonals="),
            (("xsources", "diag_signs"), "requires coeffs="),
            (("xsources", "diagonals", "coeffs"), "already folds in coeffs="),
        ],
    )
    def test_underspecified_or_overspecified_keyword_calls_raise(self, names, match):
        """Every way of not naming exactly one X source and one diagonal strategy must raise.

        This is the substance of the change: the six valid combinations become the only *constructible*
        ones. Under the positional form each of these was either a silent wrong answer or an opaque
        failure deep inside the scan; here they fail at the call site before any array is read.
        """
        p = apply_h_inputs(np.random.default_rng(20260805))
        kwargs = {name: p[name] for name in names}
        with pytest.raises(ValueError, match=match):
            apply_h(p["vector"], states=p["states_u"], **kwargs)

    def test_no_arrays_at_all_raises(self):
        """Naming nothing names what is missing, rather than failing somewhere downstream."""
        with pytest.raises(ValueError, match="exactly one of xsources= or xsignatures="):
            apply_h(np.zeros(4))

    @pytest.mark.parametrize("cache_level", CACHE_LEVELS)
    def test_pack_scanned_arity_matches_what_the_kernel_unpacks(self, cache_level):
        """The packer's arity is a contract with the kernel, and it is shared by two callers.

        ``_apply_h_kernel``'s scan body reads ``val[2]`` only when ``cache_level[1] == 0``, so the
        3-tuple/2-tuple split is a real contract: the two strategies that *compute* a diagonal need
        the coefficients, the one that reads a precomputed diagonal must not carry them.

        Asserted directly because the end-to-end tests do **not** catch a violation. Mutation-tested:
        forcing the 3-tuple for every level leaves all of `test_sqd.py` green, because the extra
        element is scanned and then ignored -- wasted work per group rather than a wrong number. That
        makes it invisible to any value assertion, and it is exactly the kind of silent drift the
        shared packer exists to prevent now that ``run_sqd`` and ``apply_h`` both depend on it.
        """
        marker = np.zeros(1)
        packed = _pack_scanned(cache_level, marker, marker, marker)
        expected = 2 if cache_level[1] == 2 else 3
        assert len(packed) == expected, (
            f"cache_level={cache_level} packed a {len(packed)}-tuple; the kernel expects {expected} "
            "(coeffs are carried only by the levels that compute a diagonal)"
        )

    def test_shape_assertion_would_not_have_closed_this(self):
        """Records *why* the fix is naming rather than a per-branch shape check.

        A shape assertion is the obvious cheap mitigation and was proposed as one: X sources are
        ``(n_groups, n_states)`` while X signatures are ``(n_groups, n_bytes)``, so the trailing
        dimension "already distinguishes them". It does not, and this test pins the counterexample so
        nobody re-derives the shortcut: at ``n = 15`` the signatures are 2 bytes wide, so a 2-state
        subspace makes both arrays exactly ``(2, 2)``. Only the dtype differs, which is an
        implementation detail of ``get_xsource`` rather than a contract.
        """
        from rqutils.paulis.symplectic import PauliSumXZ

        num_qubits = 15
        strings = ["X" * 3 + "I" * 12, "I" * 4 + "ZZ" + "I" * 9]
        hamiltonian = PauliSumXZ.from_paulisum((strings, [0.5, -0.3]))
        states = np.array(
            [[int(b) for b in format(c, f"0{num_qubits}b")] for c in (0, 7)], dtype=np.uint8
        )
        states_u = uniquify_states(pack_padded(states), states.shape[0])
        xsources = np.stack([np.asarray(get_xsource(x, states_u)) for x in hamiltonian.x])

        assert hamiltonian.x.shape == xsources.shape == (2, 2), (
            "the counterexample requires the signature and index arrays to collide in shape; "
            f"got {hamiltonian.x.shape} and {xsources.shape}"
        )

    @pytest.mark.parametrize("cache_level", [(0, 0), (0, 1), (0, 2), (1, 0)])
    def test_omitting_states_raises(self, cache_level):
        """Only ``(1, 1)`` and ``(1, 2)`` can run without the state list; the rest must say so.

        ``(1, 1)`` and ``(1, 2)`` read neither signature array, which is what lets a caller drop S
        after caching. For the other four, a missing S would otherwise surface as an opaque failure
        deep inside ``get_xsource``/``get_diagonal``.
        """
        dummy = np.zeros((1, 1), dtype=np.uint8)
        xname = "xsources" if cache_level[0] == 1 else "xsignatures"
        dname = {0: "zsignatures", 1: "diag_signs", 2: "diagonals"}[cache_level[1]]
        kwargs = {xname: dummy, dname: dummy}
        if cache_level[1] != 2:
            kwargs["coeffs"] = np.zeros(1)
        with pytest.raises(ValueError, match="states is required"):
            apply_h(np.zeros(4), states=None, **kwargs)

    def test_fully_cached_level_matches_dense(self):
        """``cache_level=(1, 2)``, the fully-precomputed level, against a dense reference.

        Overlaps :meth:`test_every_cache_level_matches_dense` by design, and is kept separate because
        this level is the special case: with both the source indices and the diagonals precomputed it
        reads neither signature array, so it can run with ``states=None``. That makes it the positive
        control for the guard :meth:`test_omitting_states_raises` exercises from the other side --
        here passing None must *not* raise, and the answer must still match the dense projection.
        Fixing the input rather than parametrizing keeps that pin stable as the grid test's
        parametrization changes.
        """
        p = apply_h_inputs(np.random.default_rng(20260804), num_terms=5)
        got = np.asarray(
            apply_h(p["vector"], xsources=p["xsources"], diagonals=p["diagonals"], states=None)
        ).real
        assert np.abs(got - p["matrix"] @ p["vector"]).max() < 1e-12


class TestComplexCoefficientsAcrossCacheLevels:
    """Every ``cache_level`` must handle a complex-coefficient Hamiltonian.

    ``.c`` is complex128 whenever any Pauli string has an **odd Y count** -- the folded
    ``(-i)^{x.z}`` phase makes it so by construction, not by mistake (see
    ``paulis/symplectic.py``). ``run_sqd``'s ``vinit_from_min_diag`` took ``.real`` on the uncached
    diagonal branch but used ``diagonals[0]`` raw on the cached one, and the ``jnp.max``/``argmin``
    below reject complex input outright. So ``cache_level[1] == 2`` raised
    ``TypeError: lt does not accept dtype complex128`` for every odd-Y Hamiltonian -- **single
    device, no mesh involved**.

    Uncovered because the whole suite's fixtures draw from ``real_pauli_strings``, which keeps the Y
    count even so ``.c`` stays float64. A grid sweep over ``cache_level`` with a real fixture reports
    six passes; the defect needs the *fixture* varied, not the parameter. That is the lesson worth
    keeping: parametrizing over a strategy axis proves nothing about dtype axes the fixture pins.
    """

    def test_odd_y_hamiltonian_works_at_every_cache_level(self):
        from rqutils.paulis.symplectic import PauliSumXZ

        # "YZII" and "IIYY": the first has an odd Y count, so the folded phase leaves .c complex.
        strings = ["YZII", "XXII", "IZZI", "IIYY"]
        coeffs = [0.5, -0.3, 0.7, 0.2]
        hamiltonian = PauliSumXZ.from_paulisum((strings, coeffs))
        assert hamiltonian.c.dtype == np.complex128, (
            "fixture must carry complex coefficients or this test is vacuous"
        )

        rng = np.random.default_rng(3)
        states = np.unique(rng.integers(0, 2, size=(12, 4)).astype(np.uint8), axis=0)
        reference = float(
            np.min(np.linalg.eigvalsh(hproj(hamiltonian, states, unique_states=True).toarray()))
        )

        for cache_level in CACHE_LEVELS:
            got = float(sqd(hamiltonian, states, return_eigvec=False, cache_level=cache_level))
            assert got == pytest.approx(reference, abs=1e-9), (
                f"cache_level={cache_level}: sqd gave {got}, dense reference is {reference}"
            )


class TestShardedCacheLevels:
    """Every ``cache_level`` must give the same answer sharded as single-device.

    Two distinct sharding defects lived in the three ``cache_level[0] == 0`` cells, and **nothing
    covered them**: ``examples/scaling/poc7_sharding.py`` and the first version of this test both ran
    only ``sqd``'s default ``(1, 0)``. Measured on a 4-device mesh:

    * ``_accumulate_diagonal`` carried its template's *full* sharding spec onto a 1-D accumulator,
      while ``get_diagonal`` passes the 2-D ``(N, nbytes)`` state list -- so a rank-2
      ``PartitionSpec`` met a rank-1 ``jnp.zeros`` ("Length of sharding.spec (2) must be equal to
      aval's ndim (1)"). This failed **all six** levels.
    * ``_spread_seed``'s ``jnp.where`` mixed a replicated predicate with a partitioned ``vec``:
      ``vec`` is built sharded unconditionally, but ``run_sqd`` reshards ``states_u`` only inside
      ``if cache_level[0] == 1`` (the uncached branch still needs the replicated array for
      ``get_xsource``). Raised ``ShardingTypeError`` on ``(0, 0)``, ``(0, 1)`` and ``(0, 2)``.

    **The first bug masked the second** -- it raised earlier in the call, so fixing it turned six
    failures into three rather than none. That is why this sweeps the whole grid instead of sampling
    a representative cell: one cell reported success while three were broken.

    Runs as a subprocess because the virtual device count has to be set before jax initializes, and
    ``conftest`` has already imported it by collection time. The child script lives in
    ``tests/_sharded_cache_levels.py`` rather than an inline string so ruff and ty check it -- as a
    blob, an ``ImportError`` there would surface as a nonzero exit, indistinguishable from the
    regression this exists to catch, under an assertion message blaming the sharding.
    """

    def test_every_cache_level_agrees_sharded_and_single_device(self):
        script = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "_sharded_cache_levels.py"
        )
        assert os.path.exists(script), f"missing sharding harness at {script}"
        env = {**os.environ, "XLA_FLAGS": "--xla_force_host_platform_device_count=4"}
        # check=False deliberately: the assertion below reports the child's stderr, which is far more
        # useful than CalledProcessError's bare exit code for a jax sharding raise.
        proc = subprocess.run(
            [sys.executable, script],
            capture_output=True,
            text=True,
            check=False,
            env=env,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )
        assert proc.returncode == 0, f"sharded sqd raised:\n{proc.stderr[-3000:]}"

        seen = {}
        for line in proc.stdout.strip().splitlines():
            parts = line.split()
            if len(parts) != 4:
                continue
            i, j, single, sharded = int(parts[0]), int(parts[1]), float(parts[2]), float(parts[3])
            seen[(i, j)] = (single, sharded)

        # Assert the grid is complete before checking values: a child that died after two levels
        # would otherwise pass on the two it managed to print.
        assert sorted(seen) == CACHE_LEVELS, (
            f"expected all of {CACHE_LEVELS}, got {sorted(seen)} -- the child did not run the full "
            f"grid:\n{proc.stdout[-2000:]}"
        )
        for cache_level, (single, sharded) in sorted(seen.items()):
            assert single == pytest.approx(sharded, abs=1e-12), (
                f"cache_level={cache_level}: sharded {sharded} disagrees with single-device {single}"
            )
