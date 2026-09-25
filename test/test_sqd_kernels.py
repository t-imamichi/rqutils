"""Tests for :mod:`rqutils.sqd`'s building blocks: diagonals, source-index search, uniquification
and the six matvec kernels. Organized by defect, like ``test_sqd.py``.
"""

import warnings

import numpy as np
import pytest
from conftest import (
    CACHE_LEVELS,
    apply_h_inputs,
    apply_h_kwargs,
    pack_padded,
)

from rqutils.sqd import (
    _is_lex_sorted,
    _pack_scanned,
    _pack_state_keys,
    apply_h,
    compute_diagonal,
    get_diag_signs,
    get_diagonal,
    get_xsource,
    hproj,
    uniquify_states,
)

# Every keyword `apply_h_kwargs` may ask for, so a caller with no real arrays can fill them all.
_APPLY_H_ARRAY_KEYS = (
    "xsources",
    "xsignatures",
    "zsignatures",
    "diag_signs",
    "diagonals",
    "coeffs",
)


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
        # and hoped over (see the low-weight label below for why a sampled one passes vacuously).
        # Variation is concentrated in the TRAILING bytes so the leading 8 bytes collide across
        # states -- that is what makes a truncating uint64 key alias distinct states, the ``B > 8``
        # failure this parametrization exists to catch. Note the orientation: ``packbits`` fills from
        # the most significant end and the pad bit is at position 0, so *low* qubit indices land in
        # the *leading* bytes, the reverse of the little-endian qubit numbering.
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

    def test_wide_rows_compare_most_significant_word_first(self):
        """Defect: word significance reversed on the ``B > 8`` path, giving a *permutation*.

        The wide path packs rows into ``ceil(B/8)`` uint64 words and compares them MSW-first. If the
        word loop runs LSW-first the search still terminates and still returns in-range indices, so
        the result is a plausible permutation rather than an error --  the same silent shape as the
        byte-order defect ``test_matches_dense_partner_map_across_byte_widths`` covers.

        That test does **not** catch this: its fixture concentrates variation in the trailing bytes,
        so the leading word rarely decides a comparison and MSW-first and LSW-first agree. This one
        forces disagreement. At ``B = 9`` the padding puts byte 0 alone in word 1 and bytes 1..8 in
        word 2, so a pair differing in *both* -- with byte 0 saying "less" and the tail saying
        "greater" -- is ordered one way by MSW-first and the other by LSW-first. Only MSW-first
        matches byte-wise lexicographic order, which is what ``states`` is sorted by and what
        ``get_xsource``'s binary search requires.

        Note the *padding end* is deliberately not pinned here: leading- and trailing-padding are
        both order-preserving (a constant left-shift is monotonic), so a mutation there is not a
        defect. See ``_pack_state_words``.
        """
        # Nine-byte rows (the n=64 width) so there are two words and the word axis exists at all.
        # Chosen so byte 0 and the tail disagree on order; the values matter, not which qubits they
        # correspond to.
        rows = np.array(
            [
                [1, 255, 255, 255, 255, 255, 255, 255, 255],
                [2, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
            dtype=np.uint8,
        )
        rows[:, 0] &= 0x7F  # keep the pad bit clear so these are not read as fillers
        states_u = np.asarray(uniquify_states(rows, rows.shape[0]))
        assert not np.any(states_u[:, 0] >> 7), "fixture must contain no filler rows"

        # Independent reference: a dict over packed rows, sharing no code with the search.
        row_of = {row.tobytes(): i for i, row in enumerate(states_u)}
        # The signature that maps one row onto the other is just their XOR, so both sources exist
        # and a permuted answer is distinguishable from the right one.
        xsig = np.bitwise_xor(states_u[0], states_u[1])
        expected = np.array(
            [row_of.get(np.bitwise_xor(row, xsig).tobytes(), -1) for row in states_u],
            dtype=np.int32,
        )
        assert np.all(expected >= 0), (
            f"both partners must exist or this test cannot see a permutation: {expected}"
        )
        got = np.asarray(get_xsource(xsig, states_u))
        assert np.array_equal(got, expected), (
            f"expected {expected}, got {got} -- word comparison is not MSW-first, so integer word "
            "order disagrees with byte-wise row order"
        )


class TestUniquifyStates:
    """``uniquify_states`` sorts, deduplicates, and pads to a fixed size with 255 fillers."""

    def test_wide_rows_sort_on_every_word_not_just_the_first(self):
        """Defect: output not lex-sorted when rows share their leading uint64 word.

        The lexsort runs on ``ceil(B/8)`` packed words with ``num_keys`` equal to the word count. Drop
        it to 1 and only the most significant word orders the rows, so any group sharing that word
        comes back in arbitrary order. Nothing raises: the array is still the right shape, still
        contains every unique row, and still has its fillers in place -- but it is no longer sorted,
        which silently breaks the precondition ``get_xsource``'s binary search depends on.

        Existing coverage misses this because most fixtures vary the leading bytes, and a group has to
        share an entire 8-byte word before ``num_keys=1`` can reorder anything. At ``B = 13`` (the
        n=100 width) the first word holds byte 0 alone, so rows agreeing on byte 0 are exactly such a
        group.

        Asserting sortedness directly rather than comparing against a reference: this is the invariant
        the downstream search needs, and it is the thing that goes wrong.
        """
        nbytes = 13  # the n=100 packed width: two words, first holding byte 0 only
        # Four rows sharing byte 0 (hence the whole first word) and differing only in the last byte,
        # deliberately supplied out of order.
        rows = np.zeros((4, nbytes), dtype=np.uint8)
        rows[:, 0] = 1
        rows[:, nbytes - 1] = [9, 3, 7, 1]
        got = np.asarray(uniquify_states(rows, rows.shape[0]))

        tails = [int(r[nbytes - 1]) for r in got]
        assert sorted(tails) == [1, 3, 7, 9], f"every input row must survive, got {tails}"
        assert tails == sorted(tails), (
            f"output must be lex-sorted, got trailing bytes {tails} -- the lexsort is not keyed on "
            "every packed word, so rows sharing the leading word come back unordered and "
            "get_xsource's binary search sees unsorted input"
        )

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


class TestUint64KeyWidthBoundary:
    """``_pack_state_keys`` must reject ``B > 8`` rather than silently aliasing distinct states.

    ``get_xsource`` selects a ``uint64``-key search for ``B <= 8`` and an explicit lexicographic
    search beyond, and that dispatch is correct -- so there is no live wrong answer through the public
    path. What was missing is the guard at the packing function itself. Its docstring said "Only valid
    while ``B <= 8``; :func:`get_xsource` checks that before calling", which is a *comment*: nothing
    enforced it, and ``NOTES.md`` calls the limit "a correctness limit" while
    ``markdown/scaling-pocs.md`` calls it "a hard correctness boundary, asserted rather than documented".
    It was in fact neither asserted nor enforced.

    The failure is worse than truncation. Byte 0 is the most significant, so at ``B = 9`` its shift is
    ``8 * (9 - 1) = 64`` bits on a ``uint64`` -- the byte vanishes entirely rather than being
    coarsened, destroying lex order rather than merely weakening it. Measured: two 9-byte rows
    differing *only* in byte 0 both pack to key ``0``, as does an all-zero row.

    ``B = 9`` is reachable: ``B = ceil((n + 1) / 8)``, so ``n >= 64`` crosses it, and
    ``markdown/scaling-pocs.md`` measures at ``n = 64`` and beyond.

    The wrapper-type fix ``markdown/gotchas.md`` proposes (encoding width in item 7's packed-states type)
    is deferred, so this is defence-in-depth on a private function: it converts a silent wrong answer
    into a raise for anyone who reaches past ``get_xsource``.
    """

    @pytest.mark.parametrize("nbytes", [9, 10, 16])
    def test_wide_rows_raise(self, nbytes):
        with pytest.raises(ValueError, match="8 bytes|uint64|width"):
            _pack_state_keys(np.zeros((2, nbytes), dtype=np.uint8))

    def test_the_aliasing_it_prevents_is_real(self):
        """The premise, at the widest legal width, so the guard is not protecting a non-problem.

        Rather than call the guarded function, reproduce its arithmetic at ``B = 9`` to show that
        byte 0's 64-bit shift loses the byte outright.
        """
        shifts = np.array([8 * (9 - 1 - i) for i in range(9)], dtype=np.uint64)
        distinct = np.zeros((1, 9), dtype=np.uint8)
        distinct[0, 0] = 1
        allzero = np.zeros((1, 9), dtype=np.uint8)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            key_a = (distinct.astype(np.uint64) << shifts).sum(axis=1)
            key_b = (allzero.astype(np.uint64) << shifts).sum(axis=1)
        assert key_a[0] == key_b[0], (key_a, key_b)

    @pytest.mark.parametrize("nbytes", [1, 2, 4, 8])
    def test_legal_widths_still_pack_and_preserve_lex_order(self, nbytes):
        """The guard must not narrow the fast path, and the keys must stay order-preserving."""
        rng = np.random.default_rng(20260825)
        rows = np.unique(rng.integers(0, 128, size=(64, nbytes), dtype=np.uint8), axis=0)
        rows = rows[np.lexsort(rows.T[::-1])]
        keys = np.asarray(_pack_state_keys(rows))
        assert keys.shape == (rows.shape[0],)
        # Lex order on rows must equal integer order on keys -- the whole point of the packing.
        assert np.all(np.diff(keys) > 0), keys

    def test_get_xsource_still_handles_both_sides_of_the_boundary(self):
        """The public path is unaffected: 8 bytes takes the fast path, 9 the lexicographic one."""
        from rqutils.paulis.symplectic import PauliSumXZ

        for num_qubits in (63, 64):  # B = 8 and B = 9
            hamiltonian = PauliSumXZ.from_paulisum((["X" + "I" * (num_qubits - 1)], [1.0]))
            states = np.zeros((2, num_qubits), dtype=np.uint8)
            states[1, 0] = 1
            states_p = np.asarray(PauliSumXZ.pack_states(states))
            states_u = uniquify_states(states_p, 2)
            sources = np.asarray(get_xsource(hamiltonian.x[0], states_u))
            assert sources.shape == (2,), (num_qubits, sources)
            # The X flips the character-0 qubit, so the two states are each other's source.
            assert sorted(sources.tolist()) == [0, 1], (num_qubits, sources)


class TestSingleFillerRow:
    """``_is_lex_sorted`` must reject *one* filler row, not just two or more.

    The parity hole ``markdown/gotchas.md`` item 14 names. Filler slots are all-``255`` rows, so **two**
    are duplicates and fail the strictness test -- which is what ``_is_lex_sorted``'s docstring
    claimed made it reject padded input "by design". But a **single** filler row is still strictly
    increasing and passed. Measured: ``uniquify_states(..., 3)`` on a 2-state subspace gives
    ``[[32], [64], [255]]``, and ``_is_lex_sorted`` returned True on it while returning False for the
    4-slot version. The guard rejected the easy case and admitted the hard one.

    ``hproj`` has no filler-masking step, so that row becomes a spurious basis state in the dense
    ``[N, N]`` projection: one row and column too large, still symmetric, plausible wrong eigenvalue.
    Measured end to end -- **-1.118034 against a true -1.0**.

    Note item 1's binary check cannot cover this. Unpacking a ``255`` filler at n=2 yields ``[1, 1]``,
    a perfectly legitimate binary state, so a caller who round-trips through ``unpack_states`` gets a
    silently enlarged subspace. The guard has to sit on the *packed* side, where ``255`` is
    unambiguous because ``pack_states`` makes byte 0 of every genuine state ``< 128``.
    """

    def test_one_filler_row_is_rejected(self):
        from rqutils.paulis.symplectic import PauliSumXZ

        states = np.array([[0, 1], [1, 0]], dtype=np.uint8)
        padded = np.asarray(uniquify_states(PauliSumXZ.pack_states(states), 3))
        assert padded[-1, 0] == 255, "fixture must actually contain one filler row"
        assert not _is_lex_sorted(padded)

    def test_duplicate_rows_are_rejected(self):
        """Duplicate rows must be rejected, on filler-free input so the filler branch is not what does it.

        Was ``test_two_filler_rows_are_still_rejected``, using two all-255 rows -- which the high-bit
        filler check rejects first, making it a second copy of
        :meth:`test_one_filler_row_is_rejected`. This fixture keeps byte 0 < 128 so the rejection has
        to come from the sortedness pass instead.

        Note the ``np.all(np.any(differs, axis=1))`` line it reaches is an **early-out, not an
        independent guard**: for a duplicate pair the final ``lhs < rhs`` comparison is False too, so
        disabling the early-out leaves the suite green (measured). Nothing to pin there -- the
        behaviour is what this asserts.
        """
        from rqutils.paulis.symplectic import PauliSumXZ

        duplicated = np.asarray(PauliSumXZ.pack_states(np.array([[0, 1], [0, 1]], dtype=np.uint8)))
        assert duplicated[-1, 0] < 128, "fixture must clear the filler branch to reach sortedness"
        assert not _is_lex_sorted(duplicated)

    def test_unpack_states_silently_launders_a_filler_into_a_real_state(self):
        """A *separate* hazard, recorded rather than fixed here -- and not reachable by this guard.

        ``unpack_states`` destroys the filler marker: ``255`` unpacks to ``[1, 1]`` and repacks to
        ``96``, not ``255``. So a caller who round-trips a padded ``uniquify_states`` result hands
        ``hproj`` three *legitimately* distinct, sorted, filler-free states and gets a 3x3 projection
        where 2x2 was meant. ``_is_lex_sorted`` cannot catch that and should not try -- by then the
        input really is a valid subspace, one state too large.

        Pinned so the boundary of the filler fix is explicit: it protects callers passing **packed**
        arrays, which is the form ``uniquify_states`` returns and the form the marker survives in.
        Slice with ``~_is_filler(states)`` before unpacking.
        """
        from rqutils.paulis.symplectic import PauliSumXZ

        states = np.array([[0, 1], [1, 0]], dtype=np.uint8)
        padded = np.asarray(uniquify_states(PauliSumXZ.pack_states(states), 3))
        unpacked = PauliSumXZ.unpack_states(padded, 2)
        repacked = np.asarray(PauliSumXZ.pack_states(unpacked))
        assert padded[-1, 0] == 255 and repacked[-1, 0] == 96, (padded, repacked)
        # Accepted, because it genuinely is a valid 3-state subspace by this point.
        assert hproj((["ZI", "XI"], [1.0, 0.5]), unpacked, unique_states=True).shape == (3, 3)

    def test_hproj_cannot_reach_this_guard_and_that_is_dimension_independent(self):
        """The guard's real boundary, which two reviewers were right to question.

        An earlier version of this test asserted ``not _is_lex_sorted(padded)`` -- a byte-for-byte
        repeat of :meth:`test_one_filler_row_is_rejected` that never called ``hproj``, while its name
        and docstring claimed end-to-end coverage.

        Trying to write the honest version showed the coverage cannot exist. ``hproj`` packs
        internally, so it only ever sees packed-from-unpacked rows -- and ``pack_states`` inserts the
        pad bit at position 0, making byte 0 of *every* genuine state ``< 128`` by construction. So an
        all-ones unpacked row repacks to 127, never 255: the round trip launders a filler into a
        legitimate state at every width, not just at ``n = 2``. ``hproj`` therefore receives a valid
        subspace one state too large, and no check on its input could tell.

        The filler guard protects callers who hand an already-packed array to ``_is_lex_sorted`` --
        the form ``uniquify_states`` returns and the only form in which the marker survives.
        """
        from rqutils.paulis.symplectic import PauliSumXZ

        for num_qubits in (2, 8, 16):
            states = np.zeros((2, num_qubits), dtype=np.uint8)
            states[1, 0] = 1
            padded = np.asarray(uniquify_states(PauliSumXZ.pack_states(states), 3))
            assert padded[-1, 0] == 255, (num_qubits, padded)
            repacked = np.asarray(
                PauliSumXZ.pack_states(PauliSumXZ.unpack_states(padded, num_qubits))
            )
            assert repacked[-1, 0] < 128, (num_qubits, repacked)

    def test_genuine_sorted_input_is_still_accepted(self):
        """The guard must not reject a legitimately sorted, unique, filler-free basis."""
        from rqutils.paulis.symplectic import PauliSumXZ

        states = np.array([[0, 0], [0, 1], [1, 0]], dtype=np.uint8)
        assert _is_lex_sorted(np.asarray(PauliSumXZ.pack_states(states)))
        assert hproj((["ZI", "XI"], [1.0, 0.5]), states, unique_states=True).shape == (3, 3)

    def test_an_all_ones_state_is_not_mistaken_for_a_filler(self):
        """``[1, 1, ...]`` is a legitimate state; only the *packed* 255 marks a filler.

        At n=7 a genuine all-ones row packs to byte 0 = 127 (the pad bit keeps it under 128), so the
        high-bit test distinguishes them. This is why the check belongs on the packed side.
        """
        from rqutils.paulis.symplectic import PauliSumXZ

        all_ones = np.ones((1, 7), dtype=np.uint8)
        packed = np.asarray(PauliSumXZ.pack_states(all_ones))
        assert packed[0, 0] == 127, packed
        assert _is_lex_sorted(packed)


class TestApplyHArrayRoles:
    """``apply_h`` rejects an array passed under the wrong *name*, where dtype can tell.

        Going keyword-only removed *mispairing* -- declaring one strategy while having packed the arrays
        for another -- but not *misnaming*: ``apply_h(vec, xsources=x)`` where ``x`` is a signature array
        was still accepted. ``markdown/rqutils-requests.md`` concedes that residue is "much smaller... but it
        is not zero".

    ``apply_h``'s own docstring records why a **shape** assertion cannot close it, and that is
        correct: at ``n = 15`` (2 bytes) with a 2-state subspace, X signatures and X sources are *both*
        exactly ``(2, 2)``. That counterexample is pinned by
        :meth:`TestMatvecKernels.test_shape_assertion_would_not_have_closed_this`, which also
        asserts the dtype difference this fix relies on -- rather than rebuilding the fixture here.

        What the docstring generalized too far is "naming was the only fix available". **Dtype
        discriminates precisely where shape collides**, and structurally rather than by luck: packed
        signatures are ``uint8`` (``np.packbits`` output) while source indices are ``int32`` positions
        carrying ``-1`` as the absent marker -- a ``uint8`` cannot hold ``-1``, so the two dtypes cannot
        converge. Same for ``diagonals`` (inexact) against ``diag_signs``/``zsignatures`` (``uint8``).

        Still not closed, and deliberately not claimed: swapping two arrays of the *same* role class --
        ``xsignatures`` for ``zsignatures``, say, both ``uint8`` -- remains undetectable here.
    """

    def test_signatures_passed_as_xsources_raise(self):
        """The exact misnaming the residue names: packed signatures under ``xsources=``."""
        from rqutils.paulis.symplectic import PauliSumXZ

        hamiltonian = PauliSumXZ.from_paulisum((["XZ"], [1.0]))
        vec = np.ones(2)
        with pytest.raises(ValueError, match="xsources"):
            apply_h(
                vec,
                xsources=hamiltonian.x,  # uint8 signatures where int32 indices are meant
                diagonals=np.zeros(2),
            )

    def test_sources_passed_as_xsignatures_raise(self):
        from rqutils.paulis.symplectic import PauliSumXZ

        hamiltonian = PauliSumXZ.from_paulisum((["XZ"], [1.0]))
        states = np.array([[0, 1], [1, 0]], dtype=np.uint8)
        states_u = uniquify_states(PauliSumXZ.pack_states(states), 2)
        xsources = np.asarray(get_xsource(hamiltonian.x[0], states_u))[None]
        with pytest.raises(ValueError, match="xsignatures"):
            apply_h(
                np.ones(2),
                xsignatures=xsources,  # int32 indices where uint8 signatures are meant
                diagonals=np.zeros(2),
                states=states_u,
            )

    def test_signatures_passed_as_diagonals_raise(self):
        """``diagonals`` is inexact; ``uint8`` there is a misnamed sign-bit or signature array."""
        from rqutils.paulis.symplectic import PauliSumXZ

        hamiltonian = PauliSumXZ.from_paulisum((["XZ"], [1.0]))
        states = np.array([[0, 1], [1, 0]], dtype=np.uint8)
        states_u = uniquify_states(PauliSumXZ.pack_states(states), 2)
        xsources = np.asarray(get_xsource(hamiltonian.x[0], states_u))[None]
        with pytest.raises(ValueError, match="diagonals"):
            apply_h(np.ones(2), xsources=xsources, diagonals=hamiltonian.z[0])

    @pytest.mark.parametrize("cache_level", CACHE_LEVELS)
    def test_every_valid_input_set_is_still_accepted(self, cache_level):
        """The guard must not reject any of the six the kernel implements."""
        arrays = apply_h_inputs(np.random.default_rng(20260825))
        got = np.asarray(
            apply_h(
                arrays["vector"],
                states=arrays["states_u"],
                **apply_h_kwargs(cache_level, arrays),
            )
        )
        expected = arrays["matrix"] @ arrays["vector"]
        assert np.abs(got - expected).max() < 1e-10


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
        kwargs = apply_h_kwargs(cache_level, p)
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
        # The discriminator a shape check cannot provide, and the premise `_check_array_role` rests
        # on: uint8 packed signatures against int32 positions, which cannot converge because a uint8
        # cannot hold the -1 absent marker. Asserted on the same fixture rather than in a second copy
        # of it (see TestApplyHArrayRoles).
        assert hamiltonian.x.dtype != xsources.dtype, (hamiltonian.x.dtype, xsources.dtype)

    @pytest.mark.parametrize("cache_level", [(0, 0), (0, 1), (0, 2), (1, 0)])
    def test_omitting_states_raises(self, cache_level):
        """Only ``(1, 1)`` and ``(1, 2)`` can run without the state list; the rest must say so.

        ``(1, 1)`` and ``(1, 2)`` read neither signature array, which is what lets a caller drop S
        after caching. For the other four, a missing S would otherwise surface as an opaque failure
        deep inside ``get_xsource``/``get_diagonal``.
        """
        # Shapes are irrelevant here -- the guard fires before any array is read -- so one dummy
        # stands in for every name the level asks for.
        dummy = np.zeros((1, 1), dtype=np.uint8)
        kwargs = apply_h_kwargs(cache_level, dict.fromkeys(_APPLY_H_ARRAY_KEYS, dummy))
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
