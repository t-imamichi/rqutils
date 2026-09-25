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

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import (
    CACHE_LEVELS,
    assert_imports_without,
    assert_type_checks,
    collapsing_states,
    eigval_of,
    lowest_projected,
    pack_padded,
    project_dense,
    real_pauli_strings,
    unique_states,
)

from rqutils.sqd import (
    _MAX_STATES,
    EigenpairCheckError,
    _spread_seed,
    get_diag_signs,
    get_diagonal,
    get_xsource,
    hproj,
    run_sqd,
    sqd,
    uniquify_states,
)


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

    @pytest.mark.parametrize(
        "bad",
        [
            # First digit out of range: was silently equivalent to (0, 0) -- same answer, 7.2x cost.
            (2, 0),
            (-1, 0),
            (3, 1),
            # Second digit out of range: was `UnboundLocalError`, an internal error leaking from a
            # public entry point.
            (1, 5),
            (1, 3),
            (0, -1),
            # Malformed altogether -- wrong arity, wrong type.
            (1,),
            (1, 0, 0),
            1,
            "10",
        ],
    )
    def test_invalid_cache_level_raises(self, bad):
        """Every rejected shape must name ``cache_level`` rather than fall through the implicit else."""
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

    STATES = np.array([[0, 1, 0, 1], [1, 0, 1, 0]], dtype=np.uint8)

    @pytest.mark.parametrize(
        "entry",
        [
            # The measured loop: pack the states, feed them back, get a different subspace.
            lambda s: sqd((["ZZII"], [1.0]), pack_padded(s), return_eigvec=False),
            lambda s: hproj((["ZZII"], [1.0]), pack_padded(s)),
            # Second payoff: (n, N) instead of (N, n).
            lambda s: sqd((["ZZII"], [1.0]), s.T.copy(), return_eigvec=False),
            # Third payoff: right shape family, wrong qubit count.
            lambda s: sqd((["ZZ"], [1.0]), s, return_eigvec=False),
        ],
        ids=["sqd_packed", "hproj_packed", "transposed", "mismatched_hamiltonian"],
    )
    def test_wrong_width_is_rejected(self, entry):
        """All four mistakes reach the one shared comparison; mutating it kills every case."""
        assert pack_padded(self.STATES).shape[1] != self.STATES.shape[1], (
            "fixture must actually change width"
        )
        with pytest.raises(ValueError, match="num_qubits|width|shape"):
            entry(self.STATES)

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

    ``apply_h`` already received this treatment (``markdown/rqutils-requests.md`` C1); the public entry
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
    under ``poc/`` -- exactly the code that pushes N -- so those call sites reached the
    int32 iota with neither ``sqd()``'s nor ``hproj()``'s guard in the chain. The guard now also sits
    on ``uniquify_states``' static ``states_size``, which fires at trace time and costs nothing.

    Putting it there is also what makes **both** sides of the boundary cheap to test. Reaching the
    guard through ``hproj`` was slow enough to be impractical (the passing case runs the O(N)
    sortedness scan over 2^31 rows), which is why the accept side was previously left unpinned and
    recorded as a known gap. Through ``jax.eval_shape`` the guard traces without allocating.
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
        """THE REPORTED CASE, from ``markdown/rqutils-prefilter-bug-response.md`` section 5.

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


class TestConvergenceIsReported:
    """``sqd`` must not return a non-converged eigenvalue as though it were the answer.

    ``run_sqd`` unpacked ``ground_locg``'s result as ``eigval, eigvec, _, _``, discarding
    ``converged``. A non-converged LOBPCG run still returns ``state.theta`` -- a valid *variational
    upper bound*, so finite and entirely plausible -- and ``sqd`` wrapped it in ``float()`` and
    returned it as "Calculated ground state energy" with no indication.

    ``markdown/locg.md`` records that this absence "is the reason I4 could hide": a sign error made the
    convergence test unsatisfiable, so the solver silently never converged and every answer was the
    iteration cap's best guess. ``sqd`` also exposed no ``maxiter`` or ``tol``, so a caller could
    neither detect the situation nor retry.

    Narrow fix, deliberately: ``maxiter``/``tol`` are exposed and non-convergence raises. ``sqd``'s
    *return shape* is unchanged -- returning a status object is item 11 in ``markdown/gotchas.md`` and a
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
        expected = lowest_projected(strings, coeffs, states)
        # Both the default cap and a generous one: this fixture converges well inside either, so a
        # loose `maxiter` must be accepted and must not perturb the answer.
        for maxiter in (None, 2000):
            kwargs = {} if maxiter is None else {"maxiter": maxiter}
            got = eigval_of(strings, coeffs, states, **kwargs)
            assert abs(got - expected) < 1e-6, f"maxiter={maxiter} gave {got}, expected {expected}"

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

    def test_a_loose_atol_converges_sooner_without_changing_the_answer(self):
        """``atol`` must be plumbed through, not accepted and ignored."""
        rng = np.random.default_rng(20260825)
        strings = real_pauli_strings(6, 8, rng)
        coeffs = rng.normal(size=len(strings))
        states = unique_states(20, 6, rng)
        loose = eigval_of(strings, coeffs, states, atol=1e-6)
        expected = lowest_projected(strings, coeffs, states)
        assert abs(loose - expected) < 1e-4


class TestEigenpairCheck:
    """A pair that converged but is not an eigenpair must raise, not return.

    Defect injected: ``ground_locg`` returns its eigenvector with the dominant component's sign
    flipped and ``converged=True`` -- a shape a downstream caller (spinchain) measured at ~1e+00 and
    checked for itself. Components 0 and 1 would not do: they carry ~1e-17 of this ground state, so
    swapping them is a no-op. Measured here: 5.7e+00 against a threshold of 7.8e-14.
    """

    def test_a_sign_flipped_eigenvector_raises_the_subclass(self, monkeypatch):
        import rqutils.sqd as sqd_module

        real = sqd_module.ground_locg

        def flipped(*args, **kwargs):
            eigval, eigvec, iters, converged = real(*args, **kwargs)
            dominant = jnp.arange(eigvec.shape[-1]) == jnp.argmax(jnp.abs(eigvec))
            return eigval, jnp.where(dominant, -eigvec, eigvec), iters, converged

        rng = np.random.default_rng(20260825)
        strings = real_pauli_strings(6, 8, rng)
        coeffs = rng.normal(size=len(strings))
        states = unique_states(20, 6, rng)
        monkeypatch.setattr(sqd_module, "ground_locg", flipped)
        run_sqd.clear_cache()  # `ground_locg` is read at trace time
        try:
            with pytest.raises(EigenpairCheckError) as excinfo:
                sqd((strings, coeffs.tolist()), states, return_eigvec=False)
        finally:
            monkeypatch.undo()
            run_sqd.clear_cache()  # or later tests reuse the defective trace
        # Callers retry on this substring (non-convergence); a wrong pair must not match it.
        assert "did not converge" not in str(excinfo.value)


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
        theta, converged = float(result.eigval), bool(result.converged)
        assert not converged
        assert np.isfinite(theta) and theta > reference, (theta, reference)


class TestPublicHelperPreconditions:
    """The un-underscored helpers state preconditions; these check the ones that *can* be checked.

    ``uniquify_states``, ``get_xsource`` and ``get_diag_signs`` are public and called directly by six
    scripts under ``poc/`` -- i.e. exactly the code that pushes ``N`` past where the
    entry-point guards would have fired. ``NOTES.md`` records that this is how the int32 iota was
    reached "with neither entry-point guard in the chain".

    What is and is not reachable here, stated exactly, because two of the three preconditions cannot
    be validated at this boundary:

    - ``uniquify_states``' int32 ceiling **is** guarded, on the static ``states_size`` where the iota
      is actually created. Already fixed; re-pinned here so the bypass path stays covered.
    - ``get_xsource``'s **lex-sortedness** requirement cannot be checked. It is ``@jax.jit``-wrapped,
      so ``states`` arrives as a tracer and its values are unavailable; a host-side scan like
      ``_is_lex_sorted`` is impossible there. This is a structural limit, not an oversight, and it is
      why ``markdown/gotchas.md`` item 10 proposed wrapper types rather than validation.
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
        would silently stop testing this. ``markdown/rqutils-prefilter-bug.md`` has the report.
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
