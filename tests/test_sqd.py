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

import ast
import inspect
import textwrap
import warnings

import jax
import numpy as np
import pytest
from conftest import (
    assert_imports_without,
    assert_type_checks,
    collapsing_states,
    lowest_projected,
    project_dense,
    real_pauli_strings,
    run_sharded_child,
    unique_states,
)

from rqutils.sqd import (
    _MAX_STATES,
    _host_scalar,
    _is_lex_sorted,
    _pack_scanned,
    _pack_state_keys,
    _spread_seed,
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


def run_sqd_jaxpr(rng, **kwargs):
    """Return ``run_sqd``'s traced graph as a string, for static-argument comparisons.

    ``run_sqd`` is the jit boundary that owns the ``static_argnames`` entries, so a staticness claim
    has to be asserted there -- ``sqd`` is not jitted, and a jaxpr comparison against it would prove
    nothing about where the staticness actually has to hold.

    A plain function taking ``rng``, not a ``@pytest.fixture``: the prohibition in ``conftest`` is
    about RNG stream position depending on fixture ordering, and ``unique_states`` is the pattern.
    Returns ``str`` rather than the ``ClosedJaxpr`` because every caller compares text.
    """
    from rqutils.paulis.symplectic import PauliSumXZ

    strings = real_pauli_strings(4, 6, rng)
    hamiltonian = PauliSumXZ.from_paulisum((strings, list(rng.normal(size=len(strings)))))
    states_p = pack_padded(unique_states(12, 4, rng))
    return str(
        jax.make_jaxpr(lambda h, s: run_sqd(h, s, 16, False, (1, 0), maxiter=50, **kwargs))(
            hamiltonian, states_p
        )
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


class TestPackedStatesInput:
    """``sqd(packed=True)`` takes ``pack_states``' output directly, skipping an 8x round trip.

    Unpacked states are one byte per qubit against ``ceil((n + 1) / 8)`` packed, and both arrays are
    live during the pack, so a caller already holding the packed form was paying an 8x expansion plus a
    transient peak for nothing. Nothing internal changes -- ``run_sqd`` has always taken the packed
    form -- so what these pin is the *boundary*.
    """

    def test_packed_and_unpacked_agree_on_everything_returned(self):
        """Defect: a boundary that accepts packed input but reports a different subspace.

        All three return values must describe the same subspace in the same order, not just the
        eigenvalue: the eigenvector is indexed by basis position, and the basis's qubit count used to
        come from ``states.shape[1]`` -- the *packed* width when the caller passes packed states,
        which would unpack to the wrong number of qubits while still returning a plausible array.

        Since 2026-08-30 ``packed`` governs the returned width too, so the two bases are no longer
        directly comparable. The invariant asserted is the stronger one that survives: unpacking the
        packed return must reproduce the unpacked return exactly. That still catches the original
        defect -- a wrong qubit count cannot round-trip to the right rows -- and additionally pins
        both widths, so neither branch can quietly start returning the other form.
        """
        from rqutils.paulis.symplectic import PauliSumXZ

        rng = np.random.default_rng(21)
        num_qubits = 20
        labels = ["".join(rng.choice(list("IXYZ"), size=num_qubits)) for _ in range(8)]
        hamiltonian = PauliSumXZ.from_paulisum((labels, rng.normal(size=len(labels)).tolist()))
        states = np.unique(rng.integers(0, 2, size=(4000, num_qubits), dtype=np.uint8), axis=0)
        packed = PauliSumXZ.pack_states(states)
        assert packed.shape[1] < states.shape[1], "fixture must actually be narrower when packed"

        val_u, vec_u, basis_u = sqd(hamiltonian, states)
        val_p, vec_p, basis_p = sqd(hamiltonian, packed, packed=True)

        assert val_p == pytest.approx(val_u, abs=1e-12)
        assert np.array_equal(vec_p, vec_u), (
            "eigenvectors differ, so the bases are not the same order"
        )
        assert basis_u.shape[1] == num_qubits, (
            f"packed=False must return unpacked rows, got width {basis_u.shape[1]}"
        )
        assert basis_p.shape[1] == packed.shape[1], (
            f"packed=True must return the packed width {packed.shape[1]}, got {basis_p.shape[1]}"
        )
        assert np.array_equal(np.asarray(PauliSumXZ.unpack_states(basis_p, num_qubits)), basis_u), (
            "unpacking the packed return does not reproduce the unpacked return"
        )

    def test_packed_return_round_trips_with_no_repack(self):
        """The behaviour ``packed``'s return side exists for: feed the output straight back in.

        Before 2026-08-30 the return was unpacked regardless, so a caller holding packed states had
        to re-pack after every solve. ``pack_states`` is **not idempotent**, so that re-pack was also
        a live hazard: feeding the returned array back with ``packed=True`` would previously have
        declared unpacked rows as packed and silently solved a different subspace.

        Asserted on the *second* solve rather than only on shapes, because a returned array of the
        right width could still be the wrong rows -- and a wrong subspace changes the eigenvalue.
        """
        from rqutils.paulis.symplectic import PauliSumXZ

        rng = np.random.default_rng(831)
        num_qubits = 14
        labels = real_pauli_strings(num_qubits, 6, rng)
        hamiltonian = PauliSumXZ.from_paulisum((labels, rng.normal(size=len(labels)).tolist()))
        states = unique_states(600, num_qubits, rng)
        packed = PauliSumXZ.pack_states(states)

        first_val, _, first_basis = sqd(hamiltonian, packed, packed=True)
        assert first_basis.shape[1] == packed.shape[1], "returned width must be the packed width"

        # The round trip: no pack_states call between the two solves.
        second_val, _, second_basis = sqd(hamiltonian, first_basis, packed=True)
        assert second_val == pytest.approx(first_val, abs=1e-12), (
            "the round trip solved a different subspace, so the returned basis is not the one the "
            "solver searched"
        )
        assert np.array_equal(first_basis, second_basis), "basis not stable across the round trip"

    def test_mismatched_flag_is_rejected_on_width(self):
        """Each form must be rejected under the wrong flag, at every width where it can be."""
        from rqutils.paulis.symplectic import PauliSumXZ

        rng = np.random.default_rng(22)
        num_qubits = 12
        hamiltonian = PauliSumXZ.from_paulisum((["Z" * num_qubits], [1.0]))
        states = np.unique(rng.integers(0, 2, size=(30, num_qubits), dtype=np.uint8), axis=0)
        packed = PauliSumXZ.pack_states(states)

        with pytest.raises(ValueError, match="packed=True"):
            sqd(hamiltonian, states, packed=True, return_eigvec=False)
        with pytest.raises(ValueError, match="unpacked"):
            sqd(hamiltonian, packed, return_eigvec=False)
        with pytest.raises(ValueError, match="must be uint8"):
            sqd(hamiltonian, packed.astype(np.uint16), packed=True, return_eigvec=False)

    def test_single_qubit_is_the_one_width_the_flag_cannot_check(self):
        """Defect the flag exists for: at ``num_qubits == 1`` the two widths coincide.

        Unpacked ``[[0], [1]]`` and packed ``[[0], [64]]`` are both ``(2, 1)`` uint8, so no shape
        inference can tell them apart -- which is why this is a declared flag and not sniffed. Measured:
        passing the unpacked array with ``packed=True`` returns ``+1.0`` where the truth is ``-1.0``,
        silently, because the unpacked array is a *legal* packed array meaning something else.

        The reverse direction is closed by ``pack_states``' binary check, so only this one needs the
        caller's word. Pinned rather than fixed: there is nothing to fix, and a reader who assumes the
        widths are always distinguishable would drop the flag in favour of inference.
        """
        from rqutils.paulis.symplectic import PauliSumXZ

        hamiltonian = PauliSumXZ.from_paulisum((["Z"], [1.0]))
        unpacked = np.array([[0], [1]], dtype=np.uint8)
        packed = PauliSumXZ.pack_states(unpacked)
        assert unpacked.shape == packed.shape, "the premise: both forms are the same shape at n=1"

        truth = sqd(hamiltonian, unpacked, return_eigvec=False)
        assert truth == pytest.approx(-1.0, abs=1e-12)
        assert sqd(hamiltonian, packed, packed=True, return_eigvec=False) == pytest.approx(
            -1.0, abs=1e-12
        )
        # The undetectable misuse, recorded so nobody replaces the flag with a width check.
        assert sqd(hamiltonian, unpacked, packed=True, return_eigvec=False) == pytest.approx(
            1.0, abs=1e-12
        ), (
            "if this stops being +1.0 the n=1 ambiguity has changed and the docstring needs revisiting"
        )

    def test_hproj_still_requires_unpacked(self):
        """``hproj`` is deliberately not given the flag: its own preconditions are unpacked-only."""
        from rqutils.paulis.symplectic import PauliSumXZ

        hamiltonian = PauliSumXZ.from_paulisum((["ZZII"], [1.0]))
        states = np.array([[0, 0, 0, 0], [0, 0, 0, 1], [1, 1, 0, 0]], dtype=np.uint8)
        packed = PauliSumXZ.pack_states(states)
        with pytest.raises(ValueError, match="unpacked"):
            hproj(hamiltonian, packed)


class TestPartialXCache:
    """``xcache_groups`` caches the first J' X groups and recomputes the rest.

    The dial exists because ``cache_level[0]``'s two settings are ``4 * J * N`` bytes and nothing, and
    at n=100 that is "does not fit" against 59.8x slower (``NOTES.md``), so the intermediate values are
    the useful ones. Memory is linear in the count.
    """

    @pytest.mark.parametrize("cache_level", [(1, 0), (1, 1), (1, 2)])
    def test_every_split_matches_the_full_cache(self, cache_level):
        """Defect: a split that drops or double-counts a group, i.e. a wrong energy.

        The cached and uncached arms carry different X arrays -- int32 indices against uint8
        signatures -- so they cannot share one scanned leading axis and the matvec becomes a sum of two
        kernels. That sum is where a group can go missing or be applied twice, and either shows up as a
        plausible finite eigenvalue rather than an error, since a subspace with a term dropped is still
        a valid variational problem. So the reference is the *full* cache at the same
        ``cache_level``, and every J' from 0 to J must reproduce it.

        The diagonal axis is swept too because it is sliced by the same index: ``hamiltonian.z``,
        ``diag_signs`` and ``diagonals`` all carry the X group on their leading axis, so a
        transposed or unsliced diagonal would survive ``cache_level[1] == 2`` and fail the others.
        """
        from rqutils.paulis.symplectic import PauliSumXZ

        rng = np.random.default_rng(11)
        num_qubits = 8
        labels = ["".join(rng.choice(list("IXYZ"), size=num_qubits)) for _ in range(8)]
        hamiltonian = PauliSumXZ.from_paulisum((labels, rng.normal(size=len(labels)).tolist()))
        num_groups = hamiltonian.x.shape[0]
        states = np.unique(rng.integers(0, 2, size=(40, num_qubits), dtype=np.uint8), axis=0)

        reference = sqd(hamiltonian, states, cache_level=cache_level, return_eigvec=False)
        for ncached in range(num_groups + 1):
            got = sqd(
                hamiltonian,
                states,
                cache_level=cache_level,
                xcache_groups=ncached,
                return_eigvec=False,
            )
            assert got == pytest.approx(reference, abs=1e-10), (
                f"cache_level={cache_level}, xcache_groups={ncached}: {got} against {reference} for "
                f"the full cache -- a group is dropped, double-counted, or paired with the wrong "
                f"diagonal slice"
            )

    def test_none_and_full_count_agree(self):
        """``None`` and ``num_groups`` are the same subspace, reached by different graphs.

        ``None`` keeps the single-arm path -- one kernel, no tail tuple -- while an explicit full count
        would take the two-arm path with an empty second arm. The library resolves the latter to the
        former (``ncached < njgroups`` is false), so this pins that they agree rather than that one is
        a special case of the other.
        """
        from rqutils.paulis.symplectic import PauliSumXZ

        rng = np.random.default_rng(12)
        num_qubits = 6
        labels = ["".join(rng.choice(list("IXYZ"), size=num_qubits)) for _ in range(4)]
        hamiltonian = PauliSumXZ.from_paulisum((labels, [1.0] * len(labels)))
        states = np.unique(rng.integers(0, 2, size=(20, num_qubits), dtype=np.uint8), axis=0)
        num_groups = hamiltonian.x.shape[0]

        implicit = sqd(hamiltonian, states, xcache_groups=None, return_eigvec=False)
        explicit = sqd(hamiltonian, states, xcache_groups=num_groups, return_eigvec=False)
        assert implicit == pytest.approx(explicit, abs=1e-12)

    def test_rejects_values_that_would_otherwise_clamp_or_no_op(self):
        """Three silent misuses, each returning the right answer at the wrong cost.

        Out of range the slice would clamp -- ``J' > J`` caches everything, a negative caches nothing.
        With ``cache_level[0] == 0`` there is no cache to make partial, so the argument is a pure
        no-op that reads as "partial caching does not help on my problem". And ``True`` would slice as
        ``1`` through Python's bool-is-int rule, caching exactly one group.
        """
        from rqutils.paulis.symplectic import PauliSumXZ

        hamiltonian = PauliSumXZ.from_paulisum((["IIXX", "ZZII"], [1.0, 1.0]))
        states = np.array([[0, 0, 0, 0], [0, 0, 0, 1], [1, 1, 0, 0]], dtype=np.uint8)
        num_groups = hamiltonian.x.shape[0]

        with pytest.raises(ValueError, match="X groups"):
            sqd(hamiltonian, states, xcache_groups=num_groups + 1, return_eigvec=False)
        with pytest.raises(ValueError, match="X groups"):
            sqd(hamiltonian, states, xcache_groups=-1, return_eigvec=False)
        with pytest.raises(ValueError, match="no source-index cache"):
            sqd(hamiltonian, states, cache_level=(0, 0), xcache_groups=1, return_eigvec=False)
        with pytest.raises(TypeError, match="must be None or an int"):
            sqd(hamiltonian, states, xcache_groups=True, return_eigvec=False)

    def test_partial_cache_keeps_states_for_the_uncached_arm(self):
        """Defect: ``needs_states`` false on a partial cache, so the uncached arm gets ``None``.

        ``cache_level=(1, 2)`` reads neither signature array, so the full-cache path drops ``states_u``
        entirely -- that is the documented point of the most aggressive level. A partial cache breaks
        that: the uncached groups search ``states`` inside every matvec. If ``needs_states`` were left
        as the level's own value the kernel would receive ``None`` and raise, so this pins the
        override rather than the absence of a crash.
        """
        from rqutils.paulis.symplectic import PauliSumXZ

        rng = np.random.default_rng(13)
        num_qubits = 6
        labels = ["".join(rng.choice(list("IXYZ"), size=num_qubits)) for _ in range(4)]
        hamiltonian = PauliSumXZ.from_paulisum((labels, [1.0] * len(labels)))
        states = np.unique(rng.integers(0, 2, size=(20, num_qubits), dtype=np.uint8), axis=0)

        # (1, 2) is the level that would otherwise pass states=None.
        reference = sqd(hamiltonian, states, cache_level=(1, 2), return_eigvec=False)
        got = sqd(hamiltonian, states, cache_level=(1, 2), xcache_groups=1, return_eigvec=False)
        assert got == pytest.approx(reference, abs=1e-10)


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

        assert sorted(int(r[nbytes - 1]) for r in got) == [1, 3, 7, 9], (
            f"every input row must survive, got {[int(r[nbytes - 1]) for r in got]}"
        )
        tails = [int(r[nbytes - 1]) for r in got]
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


class TestCacheLevelValidation:
    """``cache_level`` digits are validated instead of falling through an implicit ``else``.

    Every branch on ``cache_level`` is an equality test with no ``else``, so before this:

    - an out-of-range **first** digit was silently ignored -- ``(2, 0)`` behaved exactly as
      ``(0, 0)``, returning the same energy at 7.2x the cost;
    - an out-of-range **second** digit surfaced as
      ``UnboundLocalError: cannot access local variable 'diagonals'`` -- an internal error, not a
      validation error, from a public entry point.

    The likelier mistake is neither: it is the **transposition**. ``(0, 1)`` and ``(1, 0)`` are both
    legal and return the same energy, differing only in cost -- ``NOTES.md`` measures ``(0, 2)`` at
    10.9x slower than ``(1, 2)`` and ``(0, 0)`` at 7.2x slower than ``(1, 0)``. A transposed tuple
    reads as "SQD is slow", never as an error, so validation cannot catch it; what the message can do
    is name the axes so the call site is readable. Kept as a tuple rather than split into two enum
    parameters because ``cache_level`` is bound **static** into the jit'd kernel via
    ``functools.partial`` (``ground_locg`` splats ``args`` positionally, so ``static_argnames`` would
    never see it) -- the validation belongs at the public boundary, not in the jit plumbing.
    """

    @pytest.mark.parametrize("bad", [(2, 0), (-1, 0), (3, 1)])
    def test_out_of_range_first_digit_raises(self, bad):
        """Was silently equivalent to ``(0, 0)``: same answer, 7.2x the cost."""
        states = np.array([[0, 1], [1, 0]], dtype=np.uint8)
        with pytest.raises(ValueError, match="cache_level"):
            sqd((["ZI"], [1.0]), states, return_eigvec=False, cache_level=bad)

    @pytest.mark.parametrize("bad", [(1, 5), (1, 3), (0, -1)])
    def test_out_of_range_second_digit_raises_a_value_error(self, bad):
        """Was ``UnboundLocalError``, an internal error leaking from a public entry point."""
        states = np.array([[0, 1], [1, 0]], dtype=np.uint8)
        with pytest.raises(ValueError, match="cache_level"):
            sqd((["ZI"], [1.0]), states, return_eigvec=False, cache_level=bad)

    @pytest.mark.parametrize("bad", [(1,), (1, 0, 0), 1, "10"])
    def test_malformed_cache_level_raises(self, bad):
        states = np.array([[0, 1], [1, 0]], dtype=np.uint8)
        with pytest.raises((ValueError, TypeError), match="cache_level"):
            sqd((["ZI"], [1.0]), states, return_eigvec=False, cache_level=bad)

    def test_the_message_names_both_axes(self):
        """A transposed tuple is legal, so the message has to make the axes readable."""
        states = np.array([[0, 1], [1, 0]], dtype=np.uint8)
        with pytest.raises(ValueError) as excinfo:
            sqd((["ZI"], [1.0]), states, return_eigvec=False, cache_level=(2, 0))
        message = str(excinfo.value)
        assert "source" in message.lower() and "diagonal" in message.lower()

    @pytest.mark.parametrize("cache_level", CACHE_LEVELS)
    def test_every_valid_level_is_still_accepted(self, cache_level):
        """The guard must accept exactly the six the kernel implements."""
        states = np.array([[0, 1], [1, 0]], dtype=np.uint8)
        assert isinstance(
            float(sqd((["ZI"], [1.0]), states, return_eigvec=False, cache_level=cache_level)), float
        )

    def test_run_sqd_validates_too(self):
        """``run_sqd`` is public and takes the same argument, so it needs the same guard."""
        from rqutils.paulis.symplectic import PauliSumXZ

        states = np.array([[0, 1], [1, 0]], dtype=np.uint8)
        hamiltonian = PauliSumXZ.from_paulisum((["ZI"], [1.0]))
        with pytest.raises(ValueError, match="cache_level"):
            run_sqd(hamiltonian, pack_padded(states), 2, False, (2, 0))


class TestStatesWidthCheck:
    """``states.shape[1]`` must equal ``hamiltonian.num_qubits`` on both entry points.

    The realistic failure is that ``pack_states`` is **not idempotent**. ``sqd`` takes unpacked
    ``(N, n)`` states and *returns* unpacked ones, but the natural intermediate a caller keeps -- from
    ``uniquify_states``, or from ``pack_states`` called directly as the docstring encourages -- is
    *packed*, shape ``(N, ceil((n+1)/8))``. Feeding that back in re-packs it: ``astype(uint8)`` is a
    no-op and ``packbits`` then treats each byte as one bit via nonzero-to-1, yielding a different
    subspace. Nothing caught it, because both inputs are 2-D uint8 and the width was never compared
    against the Hamiltonian. This is a realistic loop -- run ``sqd``, do configuration recovery, run
    ``sqd`` again.

    One ``O(1)`` comparison closes double-packing, a transposed array, and a mismatched Hamiltonian at
    once. Note it does not close *every* re-feed: at ``n <= 7`` a packed row is 1 byte wide, so a
    1-qubit Hamiltonian would accept it -- the shape genuinely matches there. Item 1's binary check is
    what catches that case, since packed bytes exceed 1.
    """

    def test_sqd_rejects_packed_states(self):
        """The measured loop: pack the states, feed them back, get a different subspace."""
        states = np.array([[0, 1, 0, 1], [1, 0, 1, 0]], dtype=np.uint8)
        packed = pack_padded(states)
        assert packed.shape[1] != states.shape[1], "fixture must actually change width"
        with pytest.raises(ValueError, match="num_qubits|width|shape"):
            sqd((["ZZII"], [1.0]), packed, return_eigvec=False)

    def test_hproj_rejects_packed_states(self):
        states = np.array([[0, 1, 0, 1], [1, 0, 1, 0]], dtype=np.uint8)
        with pytest.raises(ValueError, match="num_qubits|width|shape"):
            hproj((["ZZII"], [1.0]), pack_padded(states))

    def test_a_transposed_array_is_rejected(self):
        """Same check, second payoff: (n, N) instead of (N, n)."""
        states = np.array([[0, 1, 0, 1], [1, 0, 1, 0]], dtype=np.uint8)
        with pytest.raises(ValueError, match="num_qubits|width|shape"):
            sqd((["ZZII"], [1.0]), states.T.copy(), return_eigvec=False)

    def test_a_mismatched_hamiltonian_is_rejected(self):
        """Third payoff: right shape family, wrong qubit count."""
        states = np.array([[0, 1, 0, 1], [1, 0, 1, 0]], dtype=np.uint8)
        with pytest.raises(ValueError, match="num_qubits|width|shape"):
            sqd((["ZZ"], [1.0]), states, return_eigvec=False)

    def test_the_error_names_both_widths(self):
        states = np.array([[0, 1, 0, 1]], dtype=np.uint8)
        with pytest.raises(ValueError) as excinfo:
            sqd((["ZZ"], [1.0]), states, return_eigvec=False)
        message = str(excinfo.value)
        assert "4" in message and "2" in message

    def test_matching_widths_still_work(self):
        """The guard must not narrow what already worked."""
        states = np.array([[0, 1, 0, 1], [1, 0, 1, 0]], dtype=np.uint8)
        assert isinstance(float(sqd((["ZZII"], [1.0]), states, return_eigvec=False)), float)
        assert hproj((["ZZII"], [1.0]), states).shape == (2, 2)


class TestKeywordOnlyEntryPoints:
    """Everything after ``states`` is keyword-only on both public entry points.

    The slip this closes is ``sqd(ham, states, True)``. Every parameter after ``states`` used to be
    positional-or-keyword, and the three are semantically unrelated (``states_size: int | None``,
    ``return_eigvec: bool``, ``cache_level: tuple``). Since ``True == 1``, that call was a *valid*
    ``states_size`` and did not raise -- it pinned the array to size 1. ``hproj(ham, states, True)``
    is the same shape one function over, where the third parameter is ``unique_states``.

    ``apply_h`` already received this treatment (``docs/rqutils-requests.md`` C1); the public entry
    points were missed. No in-tree caller passed these positionally, so this is a downstream-only
    break.
    """

    def test_sqd_rejects_a_third_positional_argument(self):
        states = np.array([[0, 1], [1, 0]], dtype=np.uint8)
        with pytest.raises(TypeError, match="positional"):
            sqd((["ZI"], [1.0]), states, True)

    def test_hproj_rejects_a_third_positional_argument(self):
        states = np.array([[0, 1], [1, 0]], dtype=np.uint8)
        with pytest.raises(TypeError, match="positional"):
            hproj((["ZI"], [1.0]), states, True)  # ty: ignore[too-many-positional-arguments]

    def test_the_keyword_forms_still_work(self):
        """The two arguments a caller actually wants must remain reachable by name."""
        states = np.array([[0, 1], [1, 0]], dtype=np.uint8)
        assert isinstance(float(sqd((["ZI"], [1.0]), states, return_eigvec=False)), float)
        assert hproj((["ZI"], [1.0]), states, unique_states=False).shape == (2, 2)

    def test_states_size_one_is_still_expressible_by_name(self):
        """The guard must not remove the behaviour, only the accidental way of reaching it."""
        states = np.array([[0, 1], [1, 0]], dtype=np.uint8)
        # states_size=1 is legal but degenerate: one slot for a two-state subspace.
        with pytest.raises((ValueError, IndexError, RuntimeError)):
            sqd((["ZI"], [1.0]), states, states_size=1, return_eigvec=False)


class TestInt32Ceiling:
    """``_MAX_STATES`` is enforced where the int32 index is created, not only in the entry points.

    ``uniquify_states`` and ``get_xsource`` are un-underscored and are called directly by six scripts
    under ``examples/scaling/`` -- exactly the code that pushes N -- so those call sites reached the
    int32 iota with neither ``sqd()``'s nor ``hproj()``'s guard in the chain. The guard now also sits
    on ``uniquify_states``' static ``states_size``, which fires at trace time and costs nothing.

    Putting it there is also what makes **both** sides of the boundary cheap to test. Reaching the
    guard through ``hproj`` cost 23 s (the passing case runs the O(N) sortedness scan over 2^31 rows),
    which is why the accept side was previously left unpinned and recorded as a known gap. Through
    ``jax.eval_shape`` the guard traces without allocating: measured ~5 ms per side.
    """

    def test_oversized_states_size_raises_at_the_source(self):
        """The bypass path: ``uniquify_states`` called directly, as the scaling POCs do."""
        packed = np.zeros((4, 2), dtype=np.uint8)
        with pytest.raises(ValueError, match="exceeds the .* limit imposed by the int32 index"):
            jax.eval_shape(lambda s: uniquify_states(s, 2**31), packed)

    def test_the_largest_legal_size_is_accepted(self):
        """The accept side, which pins the comparison operator rather than just the constant.

        Mutation-tested: relaxing either ``>`` to ``>=`` rejects ``_MAX_STATES`` itself -- the largest
        representable int32 index -- and without this arm every other test stays green. Asserting the
        arithmetic instead would only reimplement the predicate; the guard has to run.
        """
        packed = np.zeros((4, 2), dtype=np.uint8)
        jax.eval_shape(lambda s: uniquify_states(s, _MAX_STATES), packed)
        assert _MAX_STATES == np.iinfo(np.int32).max


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
        ``docs/rqutils-precond-request.md`` and propagated to a POC that copied it.

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
        ``examples/scaling/poc7_sharding.py`` can hand that basis straight to ``hproj``.

        **This test used to assert the opposite of its own title for the single-filler case**, which is
        how the parity hole (``docs/gotchas.md`` item 14) survived: rejection rested on two or more
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


class TestSqdMinDiagWeightCancellation:
    """``vinit_from_min_diag``'s weight must reinforce the spread seed, never cancel it.

    A bare ``+1.0`` subtracts where the seed component is negative, and ``_spread_seed`` maps index 0
    to exactly -1.0 -- so ``argmin(diagonal) == 0`` zeroed the component at the very index the
    heuristic had just declared the best guess available. Near-cancellation at other indices is what
    makes the fix structural rather than a special case on index 0, and why this class asserts the
    invariant as well as the symptom. ``NOTES.md`` has the measurements. Each test names its own
    defect.
    """

    def test_two_state_diagonal_subspace(self):
        """THE REPORTED CASE, from ``docs/rqutils-prefilter-bug-response.md`` section 5.

        A 2-state subspace of the Bx=0 n=4 Heisenberg chain whose projected Hamiltonian is
        ``diag(-0.75, -0.25)``. The diagonal is ``[0.75, 0.75]``, so ``argmin`` is 0, the weight
        cancelled seed[0] to zero, and ``vinit`` became ``[0.0, -0.183]``. Because the operator is
        diagonal that surviving component *is* an eigenvector, so the solver returned **-0.25** in
        **0 iterations** with ``converged=True`` -- a genuine eigenvalue, just not the lowest.

        Found while validating the prefilter fix, and independent of it: this reproduces with
        ``prefilter=None``, and on revisions predating the prefilter entirely.
        """
        num_qubits = 4
        strings, coeffs = [], []
        for site in range(num_qubits - 1):
            for pauli in "XY":
                term = ["I"] * num_qubits
                term[site] = term[site + 1] = pauli
                strings.append("".join(term))
                coeffs.append(0.25)
            term = ["I"] * num_qubits
            term[site] = term[site + 1] = "Z"
            strings.append("".join(term))
            coeffs.append(0.25)
        states = np.array([[0, 1, 0, 1], [1, 1, 0, 1]], dtype=np.uint8)

        # Pin the two fixture properties the defect needs, so this cannot silently stop testing it.
        dense = project_dense(strings, np.array(coeffs), states)
        assert np.count_nonzero(np.abs(dense - np.diag(np.diag(dense)))) == 0, (
            "fixture must be diagonal -- that is what makes the surviving component an eigenvector"
        )
        reference = lowest_projected(strings, np.array(coeffs), states)
        assert reference == pytest.approx(-0.75), "fixture is no longer the reported subspace"

        got = eigval_of(strings, np.array(coeffs), states)
        assert got == pytest.approx(reference, abs=1e-10), (
            f"got {got}, expected {reference} -- the min-diagonal weight cancelled the spread seed"
        )

    def test_seed_index_zero_is_exactly_minus_one(self):
        """The precondition behind the defect, asserted directly rather than assumed.

        If a future change to ``_spread_seed``'s mixer moved this value, the test above would keep
        passing while no longer exercising a cancellation -- so pin the property itself. Any index
        whose seed is exactly -1.0 is a cancellation site under the old ``+1.0`` weight.
        """
        for states_size in (2, 16, 1024):
            states = np.zeros((states_size, 1), dtype=np.uint8)
            states_u = uniquify_states(pack_padded(states), states_size)
            seed = np.asarray(_spread_seed(states_size, states_u, np.dtype(np.float64), None))
            assert seed[0] == -1.0, (
                f"states_size={states_size}: seed[0] is {seed[0]}, not -1.0 -- the cancellation "
                "this class guards is no longer reachable, so its fixture needs revisiting"
            )

    def test_weight_reinforces_at_every_possible_argmin(self):
        """The invariant, swept over every index rather than sampled at 0.

        Near-cancellation is the general hazard (511 of ``2**20`` seeds lie within 1e-3 of -1.0), so
        the guarantee has to be that ``|vinit[imin]| >= 1`` for *any* ``imin``, not merely that index 0
        survives. Asserted on a subspace whose diagonal is engineered to place the minimum at each
        index in turn, via a pure-Z Hamiltonian: a single Z term's projected diagonal is +-c per state,
        so choosing the states fixes which index is the argmin.
        """
        num_qubits = 4
        states = np.array(
            [[int(b) for b in format(k, f"0{num_qubits}b")] for k in range(2**num_qubits)],
            dtype=np.uint8,
        )
        states_size = 16
        states_u = uniquify_states(pack_padded(states), states_size)
        seed = np.asarray(_spread_seed(states_size, states_u, np.dtype(np.float64), None))
        # The fix is `seed[imin] + sign(seed[imin])`, so |component| = |seed| + 1 >= 1 always.
        for imin in range(states_size):
            direction = np.sign(seed[imin]) if seed[imin] != 0 else 1.0
            assert abs(seed[imin] + direction) >= 1.0 - 1e-12, (
                f"imin={imin}: weighted component is {abs(seed[imin] + direction)}, below 1 -- the "
                "weight is not reinforcing"
            )
            # And the old form is what this replaces: assert it WOULD have failed at index 0.
            if imin == 0:
                assert abs(seed[imin] + 1.0) == 0.0, "index 0 no longer demonstrates the old defect"


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


# Every keyword `apply_h_kwargs` may ask for, so a caller with no real arrays can fill them all.
_APPLY_H_ARRAY_KEYS = (
    "xsources",
    "xsignatures",
    "zsignatures",
    "diag_signs",
    "diagonals",
    "coeffs",
)


def apply_h_kwargs(cache_level, arrays):
    """Map a ``cache_level`` back to the ``apply_h`` keywords that select it.

    The positional form is gone, so a level is requested by *naming* the arrays it implies. That
    mapping is the thing under test in several places here, so it lives in one function: written out
    per test it was copy-paste-with-variation, which is the hazard the keyword API exists to reduce.

    Args:
        cache_level: The ``(source_indices, diagonals)`` pair to express.
        arrays: Anything indexable by keyword name -- :func:`apply_h_inputs`' dict keys are already
            spelled as the keywords, so it can be passed directly.

    Returns:
        The keyword dict, including ``coeffs`` for the two strategies that compute a diagonal.
    """
    xname = "xsources" if cache_level[0] == 1 else "xsignatures"
    dname = {0: "zsignatures", 1: "diag_signs", 2: "diagonals"}[cache_level[1]]
    kwargs = {xname: arrays[xname], dname: arrays[dname]}
    if cache_level[1] != 2:
        kwargs["coeffs"] = arrays["coeffs"]
    return kwargs


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


class TestUint64KeyWidthBoundary:
    """``_pack_state_keys`` must reject ``B > 8`` rather than silently aliasing distinct states.

    ``get_xsource`` selects a ``uint64``-key search for ``B <= 8`` and an explicit lexicographic
    search beyond, and that dispatch is correct -- so there is no live wrong answer through the public
    path. What was missing is the guard at the packing function itself. Its docstring said "Only valid
    while ``B <= 8``; :func:`get_xsource` checks that before calling", which is a *comment*: nothing
    enforced it, and ``NOTES.md`` calls the limit "a correctness limit" while
    ``docs/scaling-pocs.md`` calls it "a hard correctness boundary, asserted rather than documented".
    It was in fact neither asserted nor enforced.

    The failure is worse than truncation. Byte 0 is the most significant, so at ``B = 9`` its shift is
    ``8 * (9 - 1) = 64`` bits on a ``uint64`` -- the byte vanishes entirely rather than being
    coarsened, destroying lex order rather than merely weakening it. Measured: two 9-byte rows
    differing *only* in byte 0 both pack to key ``0``, as does an all-zero row.

    ``B = 9`` is reachable: ``B = ceil((n + 1) / 8)``, so ``n >= 64`` crosses it, and
    ``docs/scaling-pocs.md`` measures at ``n = 64`` and beyond.

    The wrapper-type fix ``docs/gotchas.md`` proposes (encoding width in item 7's packed-states type)
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


class TestConvergenceIsReported:
    """``sqd`` must not return a non-converged eigenvalue as though it were the answer.

    ``run_sqd`` unpacked ``ground_locg``'s result as ``eigval, eigvec, _, _``, discarding
    ``converged``. A non-converged LOBPCG run still returns ``state.theta`` -- a valid *variational
    upper bound*, so finite and entirely plausible -- and ``sqd`` wrapped it in ``float()`` and
    returned it as "Calculated ground state energy" with no indication.

    ``docs/locg.md`` records that this absence "is the reason I4 could hide": a sign error made the
    convergence test unsatisfiable, so the solver silently never converged and every answer was the
    iteration cap's best guess. ``sqd`` also exposed no ``maxiter`` or ``tol``, so a caller could
    neither detect the situation nor retry.

    Narrow fix, deliberately: ``maxiter``/``tol`` are exposed and non-convergence raises. ``sqd``'s
    *return shape* is unchanged -- returning a status object is item 11 in ``docs/gotchas.md`` and a
    much wider break.

    The raise lives in ``sqd``, not ``run_sqd``: the latter is ``@jax.jit``-wrapped, so ``converged``
    is a traced boolean there and cannot be branched on at trace time.
    """

    def test_a_tight_maxiter_raises_instead_of_returning_a_guess(self):
        rng = np.random.default_rng(20260825)
        strings = real_pauli_strings(6, 8, rng)
        coeffs = rng.normal(size=len(strings))
        states = unique_states(20, 6, rng)
        with pytest.raises(RuntimeError, match="converge"):
            sqd((strings, coeffs.tolist()), states, return_eigvec=False, maxiter=1)

    def test_the_message_names_maxiter_and_tol(self):
        """A caller who hits this needs to know which knobs exist."""
        rng = np.random.default_rng(20260825)
        strings = real_pauli_strings(6, 8, rng)
        coeffs = rng.normal(size=len(strings))
        states = unique_states(20, 6, rng)
        with pytest.raises(RuntimeError) as excinfo:
            sqd((strings, coeffs.tolist()), states, return_eigvec=False, maxiter=1)
        message = str(excinfo.value)
        assert "maxiter" in message and "tol" in message

    def test_the_default_path_still_converges_and_is_unchanged(self):
        """The guard must not start rejecting the runs that always worked.

        Also pins that exposing the parameters did not change the answer: the default result must
        equal the reference, not merely avoid raising.
        """
        rng = np.random.default_rng(20260825)
        strings = real_pauli_strings(6, 8, rng)
        coeffs = rng.normal(size=len(strings))
        states = unique_states(20, 6, rng)
        got = eigval_of(strings, coeffs, states)
        expected = lowest_projected(strings, coeffs, states)
        assert abs(got - expected) < 1e-6

    def test_near_degenerate_subspace_needs_maxiter_above_the_default(self):
        """A `maxiter=1000` non-convergence can mean a small gap, NOT an ill-conditioned subspace.

        This 37-state subspace has a relative gap of 5.5e-04 -- its three lowest excited states are
        degenerate to 4e-16 and sit 3.2e-03 above the ground state. LOBPCG's eigenvalue converges
        quadratically while its eigenvector converges at a rate set by the gap, so measured here
        ``theta`` is already correct to 4.4e-16 by iteration 500, and the *residual* only crosses the
        threshold at iteration **1091** -- just past the default cap. So the default raises and
        ``maxiter=2000`` returns an answer accurate to 4.4e-16.

        Pinned because the raise is easy to misread as a defect (it was, in this repo's own stress
        testing) and because the error message's advice matters: raising ``maxiter`` is the first
        thing to try, not inspecting the subspace. Rare rather than systematic -- 0 of 140 further
        random subspaces failed at the default, including 18 with a relative gap below 1e-4.
        """
        strings = ["XIXZXX", "IIXZIZ", "YXYXII", "XZXXZZ"]
        coeffs = np.array([2.107755, 0.453263, 0.410334, 1.867813])
        # The 37 basis states written out as integers rather than redrawn from a seed: a random draw
        # gives whatever gap it gives (a nearby seed measured 8.7e-03, too large to reproduce this),
        # and CLAUDE.md's rule is that a fixture picked for a specific pathology must keep it.
        basis = [
            0,
            1,
            2,
            4,
            5,
            6,
            9,
            10,
            12,
            14,
            15,
            16,
            19,
            23,
            24,
            25,
            31,
            32,
            33,
            34,
            37,
            38,
            40,
            41,
            42,
            44,
            45,
            46,
            47,
            48,
            49,
            50,
            55,
            56,
            59,
            60,
            61,
        ]
        states = np.array([[int(b) for b in format(k, "06b")] for k in basis], dtype=np.uint8)

        spectrum = np.linalg.eigvalsh(project_dense(strings, coeffs, states))
        relgap = (spectrum[1] - spectrum[0]) / (spectrum[-1] - spectrum[0])
        assert relgap < 5e-3, (
            f"fixture's relative gap is {relgap:.2e}, too large to need more than the default "
            "maxiter -- this test is no longer exercising a near-degenerate subspace"
        )

        # prefilter=None explicitly: `(32, 2)` is the default now and *resolves* this case within the
        # default cap (measured, converged to 4.4e-16 at maxiter=1000), which is a real bonus of the
        # default change but would make this test assert nothing. The diagnosis being pinned here is
        # about the unfiltered solver's convergence, so pin the unfiltered path.
        with pytest.raises(RuntimeError, match="did not converge"):
            eigval_of(strings, coeffs, states, maxiter=1000, prefilter=None)
        got = eigval_of(strings, coeffs, states, maxiter=4000, prefilter=None)
        assert got == pytest.approx(float(spectrum[0]), abs=1e-10), (
            f"got {got}, expected {spectrum[0]} with a generous maxiter"
        )

    def test_a_generous_maxiter_is_accepted(self):
        rng = np.random.default_rng(20260825)
        strings = real_pauli_strings(6, 8, rng)
        coeffs = rng.normal(size=len(strings))
        states = unique_states(20, 6, rng)
        got = eigval_of(strings, coeffs, states, maxiter=2000)
        expected = lowest_projected(strings, coeffs, states)
        assert abs(got - expected) < 1e-6

    def test_a_loose_atol_converges_sooner_without_changing_the_answer(self):
        """``atol`` must be plumbed through, not accepted and ignored."""
        rng = np.random.default_rng(20260825)
        strings = real_pauli_strings(6, 8, rng)
        coeffs = rng.normal(size=len(strings))
        states = unique_states(20, 6, rng)
        loose = eigval_of(strings, coeffs, states, atol=1e-6)
        expected = lowest_projected(strings, coeffs, states)
        assert abs(loose - expected) < 1e-4


class TestAtolAndRtol:
    """Convergence is ``||r|| < max(atol, rtol * (||Hv|| + |E|))`` -- either arm suffices.

    Two defects are locked down here, one per arm.

    ``atol`` exists because a purely *relative* tolerance cannot name a residual: the old ``tol`` was
    multiplied by ``(||Hv|| + |E|) * N * 10``, so one value meant a different absolute residual at every
    subspace size and a caller with a fixed 1e-6 requirement could not express it. The achievable floor
    is ``eps*||H||`` with **no** N dependence -- measured over n=70..32768 and six decades of ``||H||``
    (27 samples, dense and matrix-free, both dtypes), ``floor/(eps||H||)`` spans 2.6x where
    ``floor/(eps||H||N)`` spans 306x -- so the ``N`` factor was slack, not a rounding budget.

    ``rtol`` exists because a purely *absolute* tolerance cannot scale: a pipeline solving at several N
    in one run needs a per-dimension bound from one value, and the slack the floor measurement exposed
    is exactly that scaling property. So it is retained deliberately rather than corrected away.

    The ``max`` is what makes the pair strictly more expressive than either alone, and it is why the
    below-floor guard is conditioned on ``rtol == 0``: with a live relative arm, an unreachable ``atol``
    is harmless, and a guard that rejected it would fire on correct input.
    """

    def _problem(self, seed=20260831, num_qubits=6, num_terms=8, num_states=24):
        rng = np.random.default_rng(seed)
        strings = real_pauli_strings(num_qubits, num_terms, rng)
        coeffs = rng.normal(size=len(strings))
        states = unique_states(num_states, num_qubits, rng)
        return strings, coeffs, states

    def _residual(self, strings, coeffs, states, **kwargs):
        """``||Hv - Ev||`` from an independent dense construction, not from the solver's own report."""
        from rqutils.paulis.symplectic import PauliSumXZ

        eigval, eigvec, _basis = sqd((strings, list(coeffs)), states, return_eigvec=True, **kwargs)
        ham = PauliSumXZ.from_paulisum((strings, list(coeffs)))
        dense = hproj(ham, np.unique(states, axis=0)).toarray()
        vec = np.asarray(eigvec).ravel()[: dense.shape[0]]
        # sqd returns the padded basis; trim and renormalize so the residual is on a unit vector.
        nrm = np.linalg.norm(vec)
        assert nrm > 0.0, "eigenvector came back zero -- the fixture, not the tolerance, is broken"
        vec = vec / nrm
        return float(np.linalg.norm(dense @ vec - float(eigval) * vec))

    def test_the_requested_residual_is_actually_achieved(self):
        """A caller asking for 1e-8 must get a residual below 1e-8, verified independently.

        The subspace is deliberately **large** (n=10, 200 draws). The old relative form multiplied
        ``tol`` by ``(||Hv|| + |E|) * N * 10``, so its threshold grew with N: at N~180 that is a factor
        of ~3.6e4, admitting a residual of ~3.6e-4 for this request. A small fixture does **not**
        discriminate -- at N=21 the solver overshoots the loose threshold and lands under 1e-6 anyway,
        so the assertion passes under both semantics and pins nothing. Verified by mutation: with the
        relative form restored this fixture reports **4.967e-05** against the requested 1e-8, while the
        N=21 fixture still passes.
        """
        strings, coeffs, states = self._problem(num_qubits=10, num_terms=12, num_states=200)
        resid = self._residual(strings, coeffs, states, atol=1e-8, rtol=0.0, maxiter=6000)
        assert resid < 1e-8, f"asked for 1e-8, got {resid:.3e}"

    def test_a_tighter_tol_gives_a_smaller_residual(self):
        """Monotonicity: the knob must actually move the delivered residual, not just be accepted."""
        strings, coeffs, states = self._problem()
        loose = self._residual(strings, coeffs, states, atol=1e-6, rtol=0.0, maxiter=4000)
        tight = self._residual(strings, coeffs, states, atol=1e-11, rtol=0.0, maxiter=4000)
        assert tight < loose, f"atol=1e-11 gave {tight:.3e}, not below atol=1e-6's {loose:.3e}"

    def test_the_same_tol_means_the_same_residual_at_two_dimensions(self):
        """The defect, stated directly: the old form's threshold scaled with N, so this failed.

        Two subspaces differing ~4x in size must deliver residuals of the same order for one ``tol``.
        Under ``tol * (||Hv|| + |E|) * N * 10`` the larger subspace was admitted ~4x looser.
        """
        small = self._problem(num_states=12)
        large = self._problem(num_states=48, num_qubits=8)
        r_small = self._residual(*small, atol=1e-7, rtol=0.0, maxiter=4000)
        r_large = self._residual(*large, atol=1e-7, rtol=0.0, maxiter=4000)
        assert r_small < 1e-7 and r_large < 1e-7, (
            f"one arm missed the requested bound: small={r_small:.3e} large={r_large:.3e}"
        )

    def test_a_tol_below_the_floor_raises_rather_than_exhausting_maxiter(self):
        """An unreachable request is a diagnosable input error, not a 1000-iteration timeout.

        Rejected rather than clamped: the floor is computable from the operator alone, and clamping
        would silently deliver a criterion other than the one asked for.
        """
        strings, coeffs, states = self._problem()
        with pytest.raises(ValueError, match="below the achievable eigen-residual floor"):
            eigval_of(strings, coeffs, states, atol=1e-30, rtol=0.0)

    def test_the_floor_message_names_a_value_that_is_accepted(self):
        """The error must be actionable: the number it suggests has to actually work."""
        from rqutils.ground_locg import residual_floor
        from rqutils.paulis.symplectic import PauliSumXZ

        strings, coeffs, states = self._problem()
        ham = PauliSumXZ.from_paulisum((strings, list(coeffs)))
        floor = residual_floor(float(np.abs(ham.c).sum()), ham.c.dtype)
        # Just above the floor must be accepted, just below must not.
        eigval_of(strings, coeffs, states, atol=floor * 1.001, rtol=0.0, maxiter=8000)
        with pytest.raises(ValueError, match="below the achievable"):
            eigval_of(strings, coeffs, states, atol=floor * 0.999, rtol=0.0)

    def test_a_negative_tolerance_raises(self):
        """Negative is meaningless as a norm bound. Zero is **legal** and means "disable this arm"."""
        strings, coeffs, states = self._problem()
        with pytest.raises(ValueError, match="atol must be non-negative"):
            eigval_of(strings, coeffs, states, atol=-1e-6)
        with pytest.raises(ValueError, match="rtol must be non-negative"):
            eigval_of(strings, coeffs, states, rtol=-1e-6)

    def test_atol_none_raises_but_rtol_none_is_the_default(self):
        """The asymmetry is the point: only ``rtol`` has a derivable value.

        ``atol=None`` would have to mean "derive an absolute bound", which is the unintuitive construct
        this pair replaced -- 0.0 already expresses "no absolute arm".
        """
        strings, coeffs, states = self._problem()
        with pytest.raises(ValueError, match="atol must be a number, not None"):
            eigval_of(strings, coeffs, states, atol=None)
        # rtol=None is the default and must simply work.
        eigval_of(strings, coeffs, states, rtol=None, maxiter=8000)

    def test_both_tolerances_zero_raises(self):
        """No residual satisfies ``|r| < 0``, so this would exhaust maxiter. Diagnosable here."""
        strings, coeffs, states = self._problem()
        with pytest.raises(ValueError, match="both 0.0"):
            eigval_of(strings, coeffs, states, atol=0.0, rtol=0.0)

    def test_a_bound_that_reaches_the_operator_norm_raises_on_either_arm(self):
        """A bound at ``||H||`` accepts anything, so the first iterate "converges" on a wrong answer.

        Every normalized ``v`` satisfies ``||Hv - Ev|| <= ||H||`` (since ``|E| <= ||H||``), so once the
        bound reaches the operator norm the test carries no information. **Both arms are checked**,
        because the condition is on the bound and not on which parameter produced it -- an earlier
        revision guarded only ``rtol``, and ``atol=100`` against ``||H||=17`` was accepted and converged
        in **one iteration** with ``converged=True``.

        Measured on the superseded ``* n * 10`` rtol scale too: ``rtol=1e-8`` at n=2^20 gave a bound of
        4.2 against ``||H||=20``. The scale is now ``||Hv|| + |E| <= 2||H||``, so the rtol cutoff is 0.5;
        the atol cutoff is ``sum|c_k|``, an over-estimate of ``||H||_2`` so it errs toward accepting.
        """
        strings, coeffs, states = self._problem()
        sumabs = float(np.abs(np.asarray(coeffs)).sum())
        for bad in (0.5, 0.9, 2.0):
            with pytest.raises(ValueError, match="reach the operator norm"):
                eigval_of(strings, coeffs, states, rtol=bad)
        # Strictly inside the rejection region, not exactly on its edge: `sqd` sums the *padded*
        # coefficient rectangle, so its sum|c_k| differs from this one in the last ulp (measured
        # 6.473090246765939 here against 6.47309024676594 there) and an `atol == sumabs` arm would be
        # asserting floating-point associativity rather than the guard.
        for bad in (sumabs * 1.01, sumabs * 3.0):
            with pytest.raises(ValueError, match="accepts anything"):
                eigval_of(strings, coeffs, states, atol=bad, rtol=0.0)
        # Neither guard may fire on legal input.
        eigval_of(strings, coeffs, states, rtol=0.49, maxiter=8000)
        eigval_of(strings, coeffs, states, atol=sumabs * 0.5, rtol=0.0, maxiter=8000)

    def test_a_non_numeric_tolerance_raises_typeerror(self):
        """``bool`` is rejected for the reason ``_check_cache_level`` gives: it is an int subclass."""
        strings, coeffs, states = self._problem()
        for bad in ("1e-6", (1e-6,), True):
            with pytest.raises(TypeError, match="must be a real number"):
                eigval_of(strings, coeffs, states, atol=bad)
            with pytest.raises(TypeError, match="must be None or a real number"):
                eigval_of(strings, coeffs, states, rtol=bad)

    def test_a_below_floor_atol_is_accepted_when_rtol_can_still_fire(self):
        """**A guard must not fire on correct input.**

        With ``rtol > 0`` a below-floor ``atol`` is harmless -- the relative arm still converges the
        solve -- so rejecting it would fail a working configuration. That is the defect class recorded
        in ``CLAUDE.md`` (an overflow count that included padding, reported 763,677 beside a bit-exact
        result). The guard is conditioned on ``rtol == 0``, not on ``atol < floor`` alone.
        """
        strings, coeffs, states = self._problem()
        # Unreachable as an absolute bound, but rtol=None (the default) carries the solve.
        got = eigval_of(strings, coeffs, states, atol=1e-30, maxiter=8000)
        expected = lowest_projected(strings, coeffs, states)
        assert abs(got - expected) < 1e-6, f"got {got!r}, expected {expected!r}"
        # And the same value with rtol explicitly zero *must* raise.
        with pytest.raises(ValueError, match="below the achievable"):
            eigval_of(strings, coeffs, states, atol=1e-30, rtol=0.0)

    def test_convergence_is_the_looser_of_the_two_arms(self):
        """``max``, not ``min``: satisfying **either** tolerance converges the solve.

        Asserted through the delivered residual rather than a timing, since a ``max`` and a ``min`` differ
        by which arm binds. A loose ``atol`` beside a tight ``rtol`` must deliver the *loose* residual.
        """
        strings, coeffs, states = self._problem()
        # atol=1e-4 is far looser than the rtol arm; max() must pick it, so the residual lands near 1e-4
        # rather than at the ~1e-11 the relative arm alone would reach.
        loose = self._residual(strings, coeffs, states, atol=1e-4, rtol=2.22e-16, maxiter=8000)
        rel_only = self._residual(strings, coeffs, states, atol=0.0, rtol=2.22e-16, maxiter=8000)
        assert loose > rel_only, (
            f"max() did not take the looser arm: atol=1e-4 gave {loose:.3e}, "
            f"rtol-only gave {rel_only:.3e} -- a min() would make these equal"
        )
        assert loose < 1e-4, f"the loose arm should still bound the residual, got {loose:.3e}"

    def test_rtol_scales_with_the_operator_and_not_with_dimension(self):
        """``rtol`` is a fraction of ``||Hv|| + |E|``, so it tracks ``||H||`` and **not** ``N``.

        Both halves are asserted, because the earlier form conflated them. Scaling the coefficients 100x
        must move the delivered residual ~100x; growing ``N`` at fixed coefficients must **not** move it
        materially. A test varying both at once cannot attribute the difference to either -- an earlier
        revision of this test did exactly that (6 vs 10 qubits *and* 12 vs 200 states) and passed against
        a formula with no dimension term at all.
        """
        strings, coeffs, states = self._problem(num_qubits=8, num_states=40)
        r_1x = self._residual(strings, coeffs, states, maxiter=8000)
        r_100x = self._residual(strings, np.asarray(coeffs) * 100.0, states, maxiter=8000)
        ratio = r_100x / r_1x
        assert 20.0 < ratio < 500.0, (
            f"rtol should track ||H||: 100x coefficients moved the residual {ratio:.1f}x, "
            f"expected ~100x ({r_1x:.3e} -> {r_100x:.3e})"
        )

        # Same Hamiltonian, ~7x the subspace. The bound is N-independent now, so the residuals must
        # stay within an order of magnitude -- under the old `* n * 10` scale this was ~7x by construction.
        small = self._problem(num_qubits=10, num_states=30)
        large = self._problem(num_qubits=10, num_states=200)
        r_s = self._residual(*small, maxiter=8000)
        r_l = self._residual(*large, maxiter=8000)
        assert 0.1 < r_l / r_s < 10.0, (
            f"rtol must not scale with N: {len(np.unique(small[2], axis=0))} states gave {r_s:.3e}, "
            f"{len(np.unique(large[2], axis=0))} gave {r_l:.3e} ({r_l / r_s:.1f}x)"
        )

    def test_the_default_converges_and_is_accurate(self):
        """A bare call must still work, and match an independent reference."""
        strings, coeffs, states = self._problem()
        coeffs = np.asarray(coeffs) * 100.0
        got = eigval_of(strings, coeffs, states, maxiter=8000)
        expected = lowest_projected(strings, coeffs, states)
        assert abs(got - expected) < 1e-6 * abs(expected), (
            f"default tolerances gave {got!r}, expected {expected!r}"
        )

    def test_the_returned_value_would_have_been_plausible(self):
        """Records *why* this was silent: the discarded result is a valid upper bound.

        Asserts the failure mode rather than the fix, so the reason the guard exists stays visible --
        a non-converged theta is finite, real, and above the true minimum, i.e. indistinguishable
        from a correct answer by inspection.
        """
        rng = np.random.default_rng(20260825)
        strings = real_pauli_strings(6, 8, rng)
        coeffs = rng.normal(size=len(strings))
        states = unique_states(20, 6, rng)
        reference = lowest_projected(strings, coeffs, states)
        # Reach past sqd's guard to see what it would have returned.
        from rqutils.paulis.symplectic import PauliSumXZ

        hamiltonian = PauliSumXZ.from_paulisum((strings, coeffs.tolist()))
        states_p = PauliSumXZ.pack_states(states)
        result = run_sqd(hamiltonian, states_p, states_p.shape[0], False, (1, 0), maxiter=1)
        theta, converged = float(result[0]), bool(result[-1])
        assert not converged
        assert np.isfinite(theta) and theta > reference, (theta, reference)


class TestPublicHelperPreconditions:
    """The un-underscored helpers state preconditions; these check the ones that *can* be checked.

    ``uniquify_states``, ``get_xsource`` and ``get_diag_signs`` are public and called directly by six
    scripts under ``examples/scaling/`` -- i.e. exactly the code that pushes ``N`` past where the
    entry-point guards would have fired. ``NOTES.md`` records that this is how the int32 iota was
    reached "with neither entry-point guard in the chain".

    What is and is not reachable here, stated exactly, because two of the three preconditions cannot
    be validated at this boundary:

    - ``uniquify_states``' int32 ceiling **is** guarded, on the static ``states_size`` where the iota
      is actually created. Already fixed; re-pinned here so the bypass path stays covered.
    - ``get_xsource``'s **lex-sortedness** requirement cannot be checked. It is ``@jax.jit``-wrapped,
      so ``states`` arrives as a tracer and its values are unavailable; a host-side scan like
      ``_is_lex_sorted`` is impossible there. This is a structural limit, not an oversight, and it is
      why ``docs/gotchas.md`` item 10 proposed wrapper types rather than validation.
    - **Rank and dtype are static under jit**, so those *are* checkable -- and ``get_diag_signs``
      silently accepted a 1-D ``zsignatures`` array, returning a wrongly shaped result rather than
      raising.
    """

    def test_uniquify_states_ceiling_is_guarded_on_the_bypass_path(self):
        """The guard sits on the static ``states_size``, where the int32 iota is created."""
        with pytest.raises(ValueError, match="_MAX_STATES|int32|ceiling|limit"):
            jax.eval_shape(
                lambda st: uniquify_states(st, _MAX_STATES + 1),
                jax.ShapeDtypeStruct((4, 2), np.uint8),
            )

    def test_get_diag_signs_rejects_a_rank_1_zsignature_array(self):
        """Was accepted, returning shape (4, 1) from a 1-D input that should be (n_terms, n_bytes)."""
        with pytest.raises((ValueError, TypeError), match="zsignatures|rank|2-D|dimension"):
            get_diag_signs(np.zeros(2, dtype=np.uint8), np.zeros((4, 2), dtype=np.uint8))

    def test_get_diagonal_rejects_a_rank_1_zsignature_array_too(self):
        """The peer with the identical hazard: both index ``zsignatures``' leading axis.

        Measured before the shared guard: ``get_diagonal`` returned ``(4,)`` of ``[2., 2., 2., 2.]``
        from a 1-D input -- a plausible finite diagonal. It is public and is ``cache_level=(*, 0)``'s
        diagonal source, so it is in exactly the bypass population item 10 is about.
        """
        with pytest.raises(ValueError, match="zsignatures|rank|2-D|dimension"):
            get_diagonal(np.zeros(2, dtype=np.uint8), np.ones(2), np.zeros((4, 2), dtype=np.uint8))

    def test_get_diagonal_still_accepts_a_proper_2d_array(self):
        diagonal = np.asarray(
            get_diagonal(
                np.zeros((3, 2), dtype=np.uint8), np.ones(3), np.zeros((4, 2), dtype=np.uint8)
            )
        )
        assert diagonal.shape == (4,)

    def test_get_diag_signs_still_accepts_a_proper_2d_array(self):
        signs = np.asarray(
            get_diag_signs(np.zeros((3, 2), dtype=np.uint8), np.zeros((4, 2), dtype=np.uint8))
        )
        assert signs.shape[0] == 4

    def test_get_xsource_sortedness_is_documented_as_uncheckable(self):
        """Pinned so the limit is explicit: unsorted input gives wrong indices, silently.

        ``get_xsource`` binary-searches into ``states``, so sortedness is load-bearing -- but the
        function is jit'd and the values are traced, so it cannot verify it. This asserts the failure
        mode rather than a raise, which is what makes the gap visible in the suite.
        """
        from rqutils.paulis.symplectic import PauliSumXZ

        hamiltonian = PauliSumXZ.from_paulisum((["IX"], [1.0]))
        states = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.uint8)
        packed_sorted = np.asarray(PauliSumXZ.pack_states(states))
        packed_unsorted = packed_sorted[::-1].copy()

        good = np.asarray(get_xsource(hamiltonian.x[0], packed_sorted))
        bad = np.asarray(get_xsource(hamiltonian.x[0], packed_unsorted))
        # Both return plausible index arrays; only one is correct, and nothing signals which.
        assert good.shape == bad.shape
        assert not np.array_equal(good, bad), (
            "if these now agree, either the fixture stopped being unsorted or get_xsource became "
            "order-independent -- both would make this test vacuous"
        )


class TestSingleFillerRow:
    """``_is_lex_sorted`` must reject *one* filler row, not just two or more.

    The parity hole ``docs/gotchas.md`` item 14 names. Filler slots are all-``255`` rows, so **two**
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

    def test_two_filler_rows_are_still_rejected(self):
        """The case that already worked, via the duplicate test -- must not regress."""
        from rqutils.paulis.symplectic import PauliSumXZ

        states = np.array([[0, 1], [1, 0]], dtype=np.uint8)
        padded = np.asarray(uniquify_states(PauliSumXZ.pack_states(states), 4))
        assert not _is_lex_sorted(padded)

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
        was still accepted. ``docs/rqutils-requests.md`` concedes that residue is "much smaller... but it
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
        states = unique_states(12, 4, rng)
        # lowest_projected, not hproj: an independent Kronecker construction rather than library code
        # that shares get_xsource with sqd. Verified identical here (-0.7976882234986247), but
        # agreeing with a sibling that could be wrong the same way proves nothing.
        reference = lowest_projected(strings, coeffs, states)

        for cache_level in CACHE_LEVELS:
            got = float(sqd(hamiltonian, states, return_eigvec=False, cache_level=cache_level))
            assert got == pytest.approx(reference, abs=1e-9), (
                f"cache_level={cache_level}: sqd gave {got}, dense reference is {reference}"
            )


class TestShardedSqdPrefilter:
    """``sqd(prefilter=...)`` must agree sharded and single-device, and keep the vector partitioned.

    ``tests/_sharded_prefilter.py`` already covers the prefilter on a mesh, but only through
    ``ground_locg`` with a dense ``einsum`` matvec on an unpadded power-of-two vector.
    ``docs/locg-chebyshev-prefilter.md`` states that gap and defers it here.

    So this covers the one configuration only reachable through ``sqd``: a **padded** subspace whose
    filler slots are masked to zero, partitioned across a mesh, driven through ``apply_h``'s
    gather-heavy irregular kernel instead of a dense matmul. The filter calls that matvec
    ``cycles * (degree + 1)`` times before the solver's first iteration, so a sharding fault there gets
    far more exposure than one LOBPCG step would give it.

    Swept over all six ``cache_level`` values, not sampled: ``cache_level`` selects which kernel the
    filter calls, and ``TestShardedCacheLevels`` records two sharding bugs that lived in the three
    ``cache_level[0] == 0`` cells where one representative cell reported success.

    **Asserts the spec, not only the energy.** Measured, all 18 energy cases agree to 4e-16 or better
    whether or not partitioning survives -- per ``CLAUDE.md`` a replicated run agrees with
    single-device to exactly 0.0, so "correct but silently unsharded" is invisible to value
    comparison. The child therefore also prints the prefilter's output sharding.
    """

    def test_every_cache_level_agrees_sharded_and_single_device(self):
        stdout = run_sharded_child("_sharded_sqd_prefilter.py", "sqd prefilter")

        energies = {}
        specs = {}
        for line in stdout.strip().splitlines():
            parts = line.split()
            if parts[:1] == ["energy"] and len(parts) == 6:
                key = (int(parts[1]), int(parts[2]), int(parts[3]))
                energies[key] = (float(parts[4]), float(parts[5]))
            elif parts[:1] == ["spec"] and len(parts) == 5:
                specs[(int(parts[1]), parts[2])] = (parts[3], parts[4])

        # Assert both case sets are complete before checking values: a child that died partway would
        # otherwise pass on whatever it managed to print.
        expected_energies = sorted(
            (devices, *level) for devices in (1, 2, 4) for level in CACHE_LEVELS
        )
        expected_specs = sorted(
            (devices, label) for devices in (1, 2, 4) for label in ("part", "repl")
        )
        for name, got, want in (
            ("energy", sorted(energies), expected_energies),
            ("spec", sorted(specs), expected_specs),
        ):
            assert got == want, (
                f"child did not run the full {name} grid: got {got}, expected {want}\n"
                f"{stdout[-2000:]}"
            )

        for key, (single, sharded) in sorted(energies.items()):
            assert single == pytest.approx(sharded, abs=1e-12), (
                f"devices={key[0]} cache_level={key[1:]}: sharded {sharded} disagrees with "
                f"single-device {single}"
            )
        for (devices, label), (vinit_spec, filtered_spec) in sorted(specs.items()):
            assert filtered_spec == vinit_spec, (
                f"devices={devices} {label}: the prefilter returned {filtered_spec} for a "
                f"{vinit_spec} input -- it is not sharding-transparent"
            )
        # And the partitioned arm must actually be partitioned, or the check above is vacuous.
        for devices in (2, 4):
            assert specs[(devices, "part")][1] == "P('x',)", (
                f"devices={devices}: the partitioned arm came back "
                f"{specs[(devices, 'part')][1]}, so nothing was sharded and this test proves nothing"
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
        stdout = run_sharded_child("_sharded_cache_levels.py", "sqd")

        seen = {}
        for line in stdout.strip().splitlines():
            parts = line.split()
            if len(parts) != 4:
                continue
            i, j, single, sharded = int(parts[0]), int(parts[1]), float(parts[2]), float(parts[3])
            seen[(i, j)] = (single, sharded)

        # Assert the grid is complete before checking values: a child that died after two levels
        # would otherwise pass on the two it managed to print.
        assert sorted(seen) == CACHE_LEVELS, (
            f"expected all of {CACHE_LEVELS}, got {sorted(seen)} -- the child did not run the full "
            f"grid:\n{stdout[-2000:]}"
        )
        for cache_level, (single, sharded) in sorted(seen.items()):
            assert single == pytest.approx(sharded, abs=1e-12), (
                f"cache_level={cache_level}: sharded {sharded} disagrees with single-device {single}"
            )


class TestHostScalar:
    """``_host_scalar`` must accept every scalar form ``sqd`` can hand it, sharded or not.

    The defect it fixes is only reachable **multi-process**: ``float(result[0])`` raised "Fetching
    value for `jax.Array` that spans non-addressable (non process local) devices" on a 4-node mesh, on
    the default ``return_eigvec=False`` path. Virtual devices are one process, so every device is
    addressable and *that error cannot be produced here at all*.

    **Mutation-verified as NOT pinning the fix, and kept anyway.** Replacing the whole body with
    ``return value`` -- undoing the fix completely -- leaves all three of these green, because
    single-process ``float()`` succeeds either way. So this class pins the helper's *contract* (it must
    accept a device scalar, a Python float, a bool and a numpy scalar, and must not perturb the value)
    and its *premise* (the scalar really is fully replicated, so reading one shard is exact rather than
    partial). The defect itself is only catchable on a real multi-process run; recorded here so nobody
    reads a green suite as multi-node coverage.

    A reduction over a partitioned vector is the shape that matters: a rank-0 array whose sharding
    still names the whole mesh. ``jax.reshard`` is not the fix (its spec is already ``P()``), so the
    assertion worth making is that the value survives the local-shard read exactly -- a replicated
    scalar holds the same number on every device, so this is exact rather than approximate.
    """

    def test_passes_through_host_scalars_unchanged(self):
        # A plain float, a bool and a numpy scalar must survive untouched: `sqd` reaches this helper
        # on the single-device path too, where `result[0]` may already be host-side.
        assert float(_host_scalar(3.5)) == 3.5
        assert bool(_host_scalar(True)) is True
        assert float(_host_scalar(np.float64(1.25))) == 1.25

    def test_reads_a_device_scalar_exactly(self):
        # Rank-0 outputs of reductions, which is the shape run_sqd returns for eigval and converged.
        vec = jax.numpy.arange(8.0)
        assert float(_host_scalar(jax.numpy.sum(vec))) == 28.0
        assert bool(_host_scalar(jax.numpy.all(vec >= 0.0))) is True

    def test_the_multi_process_branch_does_not_depend_on_this_rank(self):
        """The branch must be rank-uniform, or the ranks disagree and the job hangs.

        Two earlier versions were wrong in different ways, both only visible on real nodes:

        1. ``if not shards: return value`` read an **empty** ``addressable_shards`` as "already
           host-side" and handed the unreadable array back, so the caller's ``float()`` raised the very
           error the helper exists to prevent.
        2. Gating the fast path on ``is_fully_replicated and addressable_shards`` fixed that but
           branched on a **per-rank** property. Measured on 4 nodes, two ranks read the value while two
           raised from the same call -- so that gate would have sent two ranks into a collective and let
           two return early, hanging at the barrier instead of failing. Strictly worse.

        Single-process cannot construct either state, so what is assertable here is the guard's
        *condition*, read off the source: the multi-process decision must come from
        ``jax.process_count()``, which is identical on every rank, and must not consult this rank's
        shards. Not a substitute for a real multi-process run.
        """
        # Strip the docstring and comments: both *quote* the historical wrong forms in order to warn
        # about them, so a naive substring search matches the explanation rather than the code. This
        # test caught itself doing exactly that.
        full = inspect.getsource(_host_scalar)
        tree = ast.parse(textwrap.dedent(full))
        func = tree.body[0]
        assert isinstance(func, ast.FunctionDef)
        if (
            func.body
            and isinstance(func.body[0], ast.Expr)
            and isinstance(func.body[0].value, ast.Constant)
        ):
            del func.body[0]  # the docstring
        source = ast.unparse(func)
        assert "process_count()" in source, (
            "the multi-process branch must be decided by jax.process_count(), which is the same on "
            "every rank; a per-rank condition makes the ranks disagree and hang at the barrier"
        )
        assert "if not shards" not in source, (
            "a bare `if not shards: return value` returns an unreadable array unchanged, which is the "
            "4-node failure this helper exists to prevent"
        )
        # The fast/slow decision must not be gated on a per-rank view of the array. `addressable_shards`
        # may still appear -- it is how the value is finally read -- but not as the branch condition.
        branch_lines = [
            line
            for line in source.splitlines()
            if line.strip().startswith(("if ", "elif "))
            and ("addressable_shards" in line or "is_fully_replicated" in line)
        ]
        assert not branch_lines, (
            f"branching on a per-rank property: {branch_lines}. Ranks that can see the value would "
            "skip the collective the others enter, and the job hangs."
        )
        # The multi-process read must go through process_allgather, not a hand-rolled
        # jit(out_shardings=...). Both hand-rolled forms failed on real nodes: a bare PartitionSpec
        # needs a context mesh ("jit requires a non-empty mesh in context"), and a NamedSharding from
        # value.sharding.mesh replicates only across that array's own mesh -- one device on a 1-device
        # solve, so the result still had no addressable shard on 3 of 4 ranks and `[0]` raised
        # IndexError.
        assert "process_allgather" in source, (
            "the multi-process read must use process_allgather, which handles the mesh-in-context and "
            "array-mesh-scope cases that broke two hand-rolled jit(out_shardings=...) attempts"
        )
        assert "tiled=True" in source, (
            "process_allgather's default tiled=False stacks a fully addressable input into a new "
            "leading axis, so a rank-0 array comes back with shape (1,) instead of a scalar"
        )

    def test_the_scalar_it_reads_is_fully_replicated(self):
        # The premise the helper rests on. If a future change made `eigval` genuinely partitioned,
        # reading one shard would silently return part of the answer -- so pin the premise, not just
        # the behaviour.
        total = jax.numpy.sum(jax.numpy.arange(8.0))
        assert total.is_fully_replicated
        assert len({float(shard.data) for shard in total.addressable_shards}) == 1


class TestShardedEigvecRoundtrip:
    """``return_eigvec=True`` on a mesh must return a genuine eigenvector of its own basis.

    Runs as a subprocess for the same reason as ``TestShardedCacheLevels``: the virtual device count
    has to be set before jax initializes.

    **The gap this closes.** Every other ``tests/_sharded_*.py`` calls ``sqd`` with
    ``return_eigvec=False``, so the branch that reshards ``eigvec`` and ``states_u`` back to
    ``PartitionSpec(None)`` had no coverage at all -- only
    ``examples/scaling/poc7_sharding.py``'s POC 7c, which is measured at 59.7 s subprocessed against
    ~1 s here. The POC stays the thorough arm; this is the distilled one.

    **It asserts the eigenvector equation, not shapes.** A reshard that dropped or reordered rows
    still returns an array of the right shape and dtype, and the eigenvalue is computed separately --
    so ``‖H v - E v‖ / ‖v‖`` against a dense projection *of the returned basis* is what couples the
    two arrays, which are resharded independently and would otherwise each look plausible alone. The
    eigenvalue is checked against the dense minimum as well, since a residual test alone is satisfied
    by any eigenpair, including an excited one.
    """

    def test_returned_eigenvector_satisfies_its_own_projection(self):
        stdout = run_sharded_child("_sharded_eigvec_roundtrip.py", "sqd")
        assert "OK" in stdout, stdout
        assert "FAIL" not in stdout, stdout


class TestShardedPartialXCache:
    """A partial source-index cache must agree with single-device on a 4-device mesh.

    Runs as a subprocess for the same reason as ``TestShardedCacheLevels``: the virtual device count
    has to be set before jax initializes.

    **This is the only test that can see the guard it exists for.** ``run_sqd`` reshards ``states_u``
    after the precompute because no further searches happen -- true for a full cache, false for a
    partial one, whose uncached groups search ``states`` inside every matvec, and ``get_xsource``
    requires that array replicated. Deleting the ``not partial_xcache`` condition on that reshard
    leaves all six ``TestPartialXCache`` cases **green** and raises on any mesh. Verified by mutation,
    which is why this exists rather than being folded into the in-process class.
    """

    def test_partial_cache_agrees_with_single_device_on_a_mesh(self):
        stdout = run_sharded_child("_sharded_partial_xcache.py", "sqd")

        single = None
        seen = {}
        for line in stdout.strip().splitlines():
            parts = line.split()
            if parts[:1] == ["single"] and len(parts) == 2:
                single = float(parts[1])
            elif parts[:1] == ["mesh"] and len(parts) == 5:
                seen[(int(parts[1]), int(parts[2]), int(parts[3]))] = float(parts[4])

        assert single is not None, f"child printed no single-device baseline:\n{stdout[-2000:]}"
        # Assert the grid is complete before checking values, so a child that died partway through
        # cannot pass on the cells it managed to print.
        expected = {(1, j, n) for j in (0, 1, 2) for n in range(7)}
        assert set(seen) == expected, (
            f"expected {len(expected)} (cache_level, xcache_groups) cells, got {len(seen)} -- the "
            f"child did not run the full sweep:\n{stdout[-2000:]}"
        )
        for key, value in sorted(seen.items()):
            assert value == pytest.approx(single, abs=1e-12), (
                f"cache_level=({key[0]}, {key[1]}), xcache_groups={key[2]}: sharded {value} "
                f"disagrees with single-device {single}"
            )


class TestSqdPrefilter:
    """``sqd(prefilter=...)`` must change the path and never the answer.

    The option is plumbing: it is forwarded verbatim to :func:`rqutils.ground_locg.ground_locg`,
    whose own ``TestChebyshevPrefilter`` covers the filter's numerics (the three-term recurrence, the
    running-Rayleigh-quotient lower edge, complex operators, degenerate knobs). What is *only*
    testable here, and untested by that class, is the interaction with the three things ``sqd`` puts
    between the caller and the solver:

    * **Filler slots.** ``uniquify_states`` pads the subspace to ``states_size`` with 255 rows. The
      prefilter normalizes and takes Rayleigh quotients over the *full padded* vector, and it calls
      the matvec ``cycles * (degree + 1)`` times before the solver's first iteration, so any leakage
      between the padding and the genuine subspace gets far more exposure in a filtered run than in
      an unfiltered one. Nothing in ``ground_locg``'s ``TestChebyshevPrefilter`` can cover this: its
      fixtures are dense and unpadded, and ragged mesh splits are explicitly out of scope there
      *because* padding is ``sqd``'s concern.

      The padded operator is block-diagonal here, so what this pins is the *energy*, not a
      no-leakage mechanism, and it does **not** pin ``_spread_seed``'s filler mask -- removing that
      mask leaves every assertion green. ``NOTES.md`` has the measurements and why.
    * **The spread seed.** ``run_sqd`` starts from ``_spread_seed``, not a one-hot, so that a subspace
      whose projected Hamiltonian splits into disconnected blocks cannot silently return one block's
      minimum. A filter is a spectral transformation applied to exactly that vector, so it is capable
      of depleting the very overlap the spread seed exists to provide.
    * **The static-argument plumbing.** ``prefilter`` reaches ``run_sqd``'s ``static_argnames`` while
      ``cache_level`` deliberately cannot (``ground_locg`` splats ``args`` positionally), so the two
      travel by different routes and the new one needs its own pin.

    Every value assertion is against :func:`lowest_projected`, the dense reference -- not against the
    unfiltered ``sqd`` arm -- so a defect common to both arms cannot pass.

    **No timing or iteration-count assertion**: the published figures were taken on dense
    ``ground_locg``, not on ``apply_h``, so pinning one here would pin an unmeasured claim. See
    ``sqd``'s ``prefilter`` docstring.
    """

    def test_the_default_is_32_2_and_disabling_it_agrees(self):
        """``(32, 2)`` is the default; ``None`` must reach the same answer, not a bit-identical one.

        Inverted when the default changed (2026-08-28). Bit-identity is the wrong assertion in this
        direction: the filter moves the starting vector, so the two arms take different paths and land
        on the same eigenpair to the solver's tolerance rather than to the last ulp. What *is* still
        exact is that omitting the argument equals passing the default explicitly.
        """
        rng = np.random.default_rng(20260828)
        num_qubits = 5
        strings = real_pauli_strings(num_qubits, 7, rng)
        coeffs = rng.normal(size=len(strings))
        states = unique_states(24, num_qubits, rng)
        reference = lowest_projected(strings, coeffs, states)

        omitted = sqd((strings, list(coeffs)), states)
        explicit = sqd((strings, list(coeffs)), states, prefilter=(32, 2))
        assert float(omitted[0]) == float(explicit[0]), (
            "omitting `prefilter` must equal passing the default explicitly, bit for bit"
        )
        assert np.array_equal(omitted[1], explicit[1]), "eigenvector must be bit-identical"

        disabled = sqd((strings, list(coeffs)), states, prefilter=None)
        assert float(disabled[0]) == pytest.approx(reference, rel=1e-10)
        assert float(omitted[0]) == pytest.approx(reference, rel=1e-10)

    def test_prefilter_none_adds_no_traced_argument(self):
        """A traced tuple would recompile per value and defeat the trace-time branch.

        Asserted on ``run_sqd``, the jit boundary that owns the ``static_argnames`` entry. ``sqd``
        itself is not jitted, so making the jaxpr comparison there would prove nothing about where
        the staticness actually has to hold.
        """
        # The baseline is `prefilter=None`, stated explicitly: `(32, 2)` is the default now, so a bare
        # call traces *with* the filter and would compare the wrong pair.
        unfiltered = run_sqd_jaxpr(np.random.default_rng(20260828), prefilter=None)
        assert run_sqd_jaxpr(np.random.default_rng(20260828), prefilter=(1, 4)) == unfiltered, (
            "a degenerate prefilter changed the traced graph, so it is not resolving at trace time"
        )
        # The converse: a real value must reach the graph. Without this, a `prefilter` silently
        # dropped on the way to `ground_locg` would pass the equality above for the wrong reason --
        # every arm identical because the option does nothing at all.
        assert run_sqd_jaxpr(np.random.default_rng(20260828), prefilter=(16, 2)) != unfiltered, (
            "prefilter=(16, 2) left the traced graph unchanged, so it is not reaching ground_locg"
        )
        # And the default really is (32, 2): omitting the argument must match passing it.
        assert run_sqd_jaxpr(np.random.default_rng(20260828)) == run_sqd_jaxpr(
            np.random.default_rng(20260828), prefilter=(32, 2)
        ), "omitting `prefilter` did not trace as the documented (32, 2) default"

    @pytest.mark.parametrize("cache_level", CACHE_LEVELS)
    def test_agrees_with_reference_across_every_kernel(self, cache_level):
        """Swept over ``cache_level``, not sampled at the default.

        Per ``CLAUDE.md`` three bugs have hidden behind the default ``(1, 0)``, each masked by the one
        before. The axes are not independent here: ``cache_level`` selects which of the six matvec
        kernels the Chebyshev recurrence calls, and the recurrence calls it ``cycles * (degree + 1)``
        times rather than once per iteration, so a kernel-specific defect gets a different amount of
        exposure in the filtered arm than in the unfiltered one.
        """
        rng = np.random.default_rng(20260828)
        num_qubits = 5
        strings = real_pauli_strings(num_qubits, 7, rng)
        coeffs = rng.normal(size=len(strings))
        states = unique_states(24, num_qubits, rng)
        reference = lowest_projected(strings, coeffs, states)
        got = eigval_of(strings, coeffs, states, cache_level=cache_level, prefilter=(16, 2))
        assert got == pytest.approx(reference, rel=1e-10), (
            f"cache_level={cache_level} with a prefilter gave {got}, expected {reference}"
        )

    def test_filler_slots_do_not_contaminate_the_filtered_vector(self):
        """Padding meets a normalizing filter -- an interaction that only exists in ``sqd``.

        Uses ``collapsing_states`` so filler rows are present in the padded arm, and an unpadded
        control so "does padding change the answer?" has an arm where padding is truly absent -- per
        ``CLAUDE.md``, a control whose filler slots exist in both arms is not a control, and the
        padding test that read like one passed against a mutant returning -1.2 for a true -0.83.

        ``states_size`` is pinned well above the unique count, so the padded arm is mostly filler:
        this fixture collapses 40 draws over 4 qubits to 14 uniques, leaving 50 of 64 slots filler.

        Asserts an energy, not a no-leakage mechanism, and does not pin ``_spread_seed``'s filler
        mask -- see the class docstring and ``NOTES.md``.
        """
        rng = np.random.default_rng(20260828)
        num_qubits = 4
        strings = real_pauli_strings(num_qubits, 6, rng)
        coeffs = rng.normal(size=len(strings))
        states = collapsing_states(40, num_qubits, rng)
        unique = np.unique(states, axis=0)

        # The control arm: states_size == the exact unique count, so there are NO filler slots.
        padded_size = 64
        reference = lowest_projected(strings, coeffs, states)

        unpadded = eigval_of(strings, coeffs, unique, states_size=len(unique), prefilter=(16, 2))
        padded = eigval_of(strings, coeffs, states, states_size=padded_size, prefilter=(16, 2))
        assert unpadded == pytest.approx(reference, rel=1e-10), (
            f"filler-free arm gave {unpadded}, expected {reference}"
        )
        assert padded == pytest.approx(reference, rel=1e-10), (
            f"{padded_size - len(unique)} filler slots contaminated the filtered vector: got "
            f"{padded}, expected {reference}"
        )

    def test_disconnected_components_survive_the_filter(self):
        """The filter must not deplete the overlap ``_spread_seed`` exists to provide.

        Reuses ``TestSqdInitialVector::test_disconnected_components``' fixture and seed: the projected
        Hamiltonian splits into blocks of 4 and 10, the minimum-diagonal state sits in the size-4
        block whose own minimum is -1.293, and the true minimum is -2.191 in the other block. A
        one-hot seed returned -1.293 -- an exact eigenvalue, just not the lowest, with
        ``converged=True``.

        This is the case where a filter could plausibly *reintroduce* that defect rather than merely
        fail: the spread seed's whole job is a non-vanishing overlap with every block, and a spectral
        filter is applied to exactly that vector. If it collapsed the iterate toward the dominant
        block the way power iteration does, the answer would come back as the wrong block's minimum
        -- a genuine eigenvalue, converged, and undetectable without this external reference.
        """
        rng = np.random.default_rng(3)
        num_qubits = 5
        strings = real_pauli_strings(num_qubits, 6, rng)
        coeffs = rng.normal(size=len(strings))
        states = rng.integers(0, 2, size=(20, num_qubits)).astype(np.uint8)

        import scipy.sparse as sp

        matrix = project_dense(strings, coeffs, states).real
        num_components = sp.csgraph.connected_components(
            sp.csr_matrix(matrix != 0), directed=False
        )[0]
        assert num_components == 2, f"fixture is no longer disconnected ({num_components} blocks)"

        reference = lowest_projected(strings, coeffs, states)
        got = eigval_of(strings, coeffs, states, prefilter=(16, 2))
        assert got == pytest.approx(reference, rel=1e-10), (
            f"filtered run on a disconnected subspace gave {got}, expected {reference} -- the filter "
            "depleted the spread seed's overlap with the block holding the true minimum"
        )

    def test_returns_the_same_eigenvector_not_merely_the_same_energy(self):
        """An energy check alone would pass on a different member of a near-degenerate pair.

        The same geometry ``ground_locg``'s ``TestChebyshevPrefilter`` guards, asserted here through
        ``sqd``'s return path, which additionally trims the eigenvector to the genuine unique rows.
        A filter that returned a neighbouring eigenvector, or a trim that lost alignment with the
        basis, would both surface as a fallen overlap.
        """
        rng = np.random.default_rng(20260828)
        num_qubits = 6
        strings = real_pauli_strings(num_qubits, 8, rng)
        coeffs = rng.normal(size=len(strings))
        states = unique_states(48, num_qubits, rng)
        plain = sqd((strings, list(coeffs)), states)
        filtered = sqd((strings, list(coeffs)), states, prefilter=(32, 2))
        assert np.array_equal(plain[2], filtered[2]), "the two arms returned different bases"
        # Length, not just overlap: `sqd` trims the eigenvector to `subspace_dim`, and this fixture
        # pads 36 uniques to 64, so 28 filler slots are trimmed away. A comparison of the two arms
        # cannot see that trim -- both are trimmed identically, so dropping it leaves the overlap at
        # 1.0 (verified against a mutant returning the untrimmed vector). Pinning the length against
        # the basis is what makes the padded tail's absence an assertion rather than an assumption.
        assert np.asarray(filtered[1]).shape[0] == np.asarray(filtered[2]).shape[0], (
            f"eigenvector ({np.asarray(filtered[1]).shape[0]}) and basis "
            f"({np.asarray(filtered[2]).shape[0]}) disagree -- the padded tail was not trimmed"
        )
        first = np.asarray(plain[1]).ravel()
        second = np.asarray(filtered[1]).ravel()
        overlap = abs(np.vdot(first, second)) / (np.linalg.norm(first) * np.linalg.norm(second))
        assert overlap > 1.0 - 1e-9, (
            f"filtered run found a different eigenvector (overlap {overlap})"
        )

    @pytest.mark.parametrize(
        "prefilter",
        [(2, -1), (-4, 2), (True, 2), (1.5, 2), (32, 2.0), (2,), "32,2", 32],
    )
    def test_malformed_values_raise_rather_than_no_op(self, prefilter):
        """A malformed value must be reported, not absorbed.

        ``ground_locg`` gates the filter on ``degree > 1 and cycles > 0``, an equality-style branch
        with an implicit ``else``, so before ``_check_prefilter`` the out-of-range values here
        returned the exact unfiltered energy at zero speedup -- measured -3.533932511396397 against a
        working ``(32, 2)``'s -3.533932511396396. That reads as "the prefilter does not help on my
        problem", which is the one misdiagnosis this option cannot afford given its docstring tells
        callers to A/B it themselves. The malformed *types* were worse: they surfaced
        ``ground_locg``'s own tuple-unpack ``ValueError``/``TypeError`` from inside a public entry
        point. ``(True, 2)`` is the ``bool``-is-an-``int`` hole ``_check_cache_level`` also closes.
        """
        rng = np.random.default_rng(20260828)
        strings = real_pauli_strings(4, 6, rng)
        coeffs = rng.normal(size=len(strings))
        states = unique_states(12, 4, rng)
        with pytest.raises((TypeError, ValueError)):
            eigval_of(strings, coeffs, states, prefilter=prefilter)

    def test_negative_leaning_hamiltonian_finds_the_ground_state(self):
        """The reported bug, through ``sqd`` rather than dense ``ground_locg``.

        ``sqd`` supplies the filter's upper bound as ``sum|c_k|``, which is rigorous because every
        Pauli string is unitary (``||H|| <= sum|c_k|``) and projecting onto the subspace can only
        shrink the spectral radius. Before that, ``ground_locg`` estimated it by power iteration,
        which converges to the eigenvalue of largest *magnitude*: on this antiferromagnetic
        Heisenberg subspace that is ``lambda_min``, so the interval inverted and ``sqd`` returned
        **+0.25** against a true **-0.75**, with ``converged=True``.

        The n=2 full basis is the smallest reproducer, and ``|lambda_min| > |lambda_max|`` is the
        precondition that makes it one -- asserted, since a fixture that stopped leaning negative
        would silently stop testing this. ``docs/rqutils-prefilter-bug.md`` has the report.
        """
        num_qubits = 2
        strings, coeffs = [], []
        for pauli in "XYZ":
            strings.append(pauli * num_qubits)
            coeffs.append(0.25)
        states = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.uint8)
        dense = project_dense(strings, np.array(coeffs), states)
        spectrum = np.linalg.eigvalsh(dense)
        assert abs(spectrum[0]) > abs(spectrum[-1]), (
            "fixture must lean negative, or the old power-iteration bound was valid by luck"
        )
        reference = lowest_projected(strings, np.array(coeffs), states)
        assert reference == pytest.approx(-0.75), "fixture is no longer the n=2 Heisenberg chain"
        got = eigval_of(strings, np.array(coeffs), states, prefilter=(32, 2))
        assert got == pytest.approx(reference, abs=1e-10), (
            f"got {got}, expected {reference} -- an excited eigenpair means the bound sqd passes as "
            "prefilter_hi is not a true upper bound on lambda_max"
        )

    def test_degenerate_prefilter_values_are_a_no_op(self):
        """``degree <= 1`` or ``cycles == 0`` must reach the guard, not divide by zero.

        ``ground_locg`` pins this on its own entry point; repeated here because the value travels
        through ``sqd``'s validation and ``run_sqd``'s ``static_argnames`` first, and a plumbing layer
        that normalized or rejected the degenerate tuples would break the no-op contract without
        touching the filter itself.

        **Asserted on the traced graph, not on the energy.** An energy comparison cannot do this job
        here: measured on this fixture, a *working* ``(16, 1)`` also returns a bit-identical energy
        (only ``(32, 2)`` moves the last ulp), so "same energy as the baseline" is satisfied by a
        genuine filter and cannot distinguish one from a no-op. Verified against a mutant that
        coerces ``cycles=0`` to ``1`` in ``sqd``: the energy form passed, this form fails.
        """
        unfiltered = run_sqd_jaxpr(np.random.default_rng(20260828), prefilter=None)
        for prefilter in [(1, 4), (16, 0), (0, 0)]:
            got = run_sqd_jaxpr(np.random.default_rng(20260828), prefilter=prefilter)
            assert got == unfiltered, (
                f"prefilter={prefilter} is degenerate and must not add filter ops to the graph"
            )


class TestHamiltonianInputIsCheckable:
    """``HamiltonianInput``'s ``SparsePauliOp`` arm must be visible to a static type checker.

    The alias was built by runtime mutation -- ``HamiltonianInput |= SparsePauliOp`` after the ``type``
    statement -- which a checker never executes, so the arm was invisible **whether or not qiskit was
    installed** and every correct ``sqd(SparsePauliOp, ...)`` call was an ``invalid-argument-type``
    error downstream (reported from `spinchain`, on calls that were right and documented as supported).
    ``svsim.CircuitInput`` and ``qprint.PrintReturnType`` had the same defect and are fixed alongside.

    See ``conftest.assert_type_checks`` for why this shells out to ``ty`` and what makes it easy to
    turn into a silent no-op.
    """

    def test_sparsepauliop_is_an_accepted_arm(self):
        pytest.importorskip("qiskit")
        assert_type_checks(
            "from qiskit.quantum_info import SparsePauliOp\n"
            "from rqutils.sqd import HamiltonianInput\n"
            "def take(h: HamiltonianInput) -> None: ...\n"
            'take(SparsePauliOp.from_list([("IIZZ", 0.5)]))\n',
            "sqd.HamiltonianInput",
        )

    def test_module_imports_without_qiskit(self):
        """The ``TYPE_CHECKING``-only qiskit import must not become a runtime dependency.

        The risk this alias takes on: it names ``SparsePauliOp`` while importing it only for the
        checker, which is safe solely because a ``type`` statement is lazy and nothing reads
        ``__value__``. That "nothing" is the kind of claim that rots, so it is pinned.
        """
        assert_imports_without(
            "rqutils.sqd",
            ["qiskit"],
            'assert type(m.HamiltonianInput).__name__ == "TypeAliasType"\n',
        )


class TestShardedDiagonals:
    """The popcount diagonal path on a mesh, with states partitioned rather than replicated.

    ``_z_parity`` is ``sum(bitwise_count(states & z), axis=1) & 1``, so it reduces along the **byte**
    axis while ``P('x', None)`` shards axis 0. That should make the whole path free of collectives --
    the easy half of the rule that only elementwise ops and reductions survive a partitioned axis,
    unlike ``uniquify_states``' ``cumsum``, which reduces *along* the sharded axis and cannot.

    Carried as the last unverified mechanism of the distributed-``states`` design and measured rather
    than assumed, because "should be free" is exactly the claim this repo requires evidence for. The
    child asserts the **spec and the values together**: a replicated run agrees to exactly 0.0, so a
    silently unsharded builder is invisible to value comparison, and a spec check alone would not
    catch a wrong sign.
    """

    def test_diagonal_builders_shard_and_agree_with_single_device(self):
        pytest.importorskip("qiskit")
        stdout = run_sharded_child("_sharded_diagonals.py", "diagonal builders")

        seen = {}
        for line in stdout.strip().splitlines():
            parts = line.split()
            if len(parts) == 5:
                seen[(parts[0], int(parts[1]))] = (int(parts[2]), int(parts[3]), int(parts[4]))

        # Assert the grid is complete before checking it, so a child that died partway through
        # cannot pass on the cells it managed to print.
        expected = {(dtype, n) for dtype in ("real", "complex") for n in (2, 4)}
        assert set(seen) == expected, f"child printed {sorted(seen)}:\n{stdout[-2000:]}"

        for (dtype, num_devices), (groups, bad_spec, bad_value) in sorted(seen.items()):
            assert groups > 1, (
                f"{dtype}/{num_devices}: only {groups} X groups, fixture is degenerate"
            )
            assert bad_spec == 0, (
                f"{dtype}/{num_devices}: {bad_spec} outputs lost their 'x' spec -- the builder ran "
                "correctly but unsharded, which value comparison alone cannot see"
            )
            assert bad_value == 0, (
                f"{dtype}/{num_devices}: {bad_value} outputs differ from single-device"
            )
