"""Tests for :mod:`rqutils.paulis.symplectic`.

``PauliSumXZ`` is the bit-packed qubit-only Pauli representation that ``sqd`` and ``svsim`` both
build on. Its two conventions are the ones worth pinning, because getting either wrong is silent:

- ``Q = (-i)^{x.z} Z^z X^x`` with the ``(-i)^{popcount(x & z)}`` phase folded into the
  coefficients. **This is the same convention whose omission broke** ``svsim`` -- it dropped the
  factor entirely and corrupted every ``y``/``ry`` gate, which is exactly the class of error a test
  here would have caught earlier (see ``docs/skqd.md``).
- **Little-endian qubit ordering**: Qiskit's ``.x``/``.z`` are reversed on ingest, so bit ``q`` of a
  packed signature is qubit ``q``. ``sqd``'s tests cover this transitively; these cover it directly.

One gap was found and fixed while writing these: the old ``force_real`` flag validated only the
*input* coefficients, so an odd-Y Pauli string came back complex128 with no warning at all. The flag
has since been removed entirely -- it could not deliver what its name promised, and its only real
effect was silently discarding the imaginary part of a non-Hermitian operator. See
:class:`TestCoefficientDtype`.
"""

import warnings

import numpy as np
import pytest
from conftest import dense_pauli_sum, gate_unitary

from rqutils.paulis.symplectic import PauliSumXZ

# The dense references come from conftest (``dense_pauli_sum`` for the Kronecker sum, ``gate_unitary``
# for a single-qubit Pauli embedded at a little-endian index). Both were previously reimplemented here
# -- verified bit-exact against the conftest versions before the switch. The independence that matters
# is from ``PauliSumXZ``, the code under test, and conftest shares no code with it either; a second
# local copy bought nothing and duplicated the trickiest convention in this file (the
# ``reversed(range(num_qubits))`` Kronecker ordering) twice over.


def signature_bits(packed, num_qubits):
    """Unpack a bit-packed signature into a qubit-0-first bit array.

    ``np.packbits`` fills each byte from the most significant end, so the payload bits are the
    *first* entries of the unpacked array, not the last. Bit 0 is the intrinsic pad bit (a dummy
    identity factor that aligns signatures with the pad bit in consumers' state bitstrings), so the
    ``num_qubits`` payload bits start at index 1. Within that slice the order matches the Pauli
    string's character order (leftmost character first), which is the opposite of the little-endian
    qubit indexing -- hence the reversal here to index by qubit number.
    """
    bits = np.unpackbits(np.atleast_1d(np.asarray(packed, dtype=np.uint8)))
    return bits[1 : 1 + num_qubits][::-1]


class TestPhaseConvention:
    """The ``(-i)^{popcount(x & z)}`` factor folded into the coefficients."""

    @pytest.mark.parametrize(
        "string,expected",
        [
            ("XX", 1.0),  # x.z == 0
            ("ZZ", 1.0),  # x.z == 0
            ("XZ", 1.0),  # x and z on different qubits, no overlap
            ("XY", -1.0j),  # one overlap
            ("YX", -1.0j),  # one overlap
            ("YY", -1.0),  # two overlaps: (-i)^2
            ("YYY", 1.0j),  # three: (-i)^3
            ("YYYY", 1.0),  # four: back to 1
        ],
    )
    def test_phase_matches_the_popcount(self, string, expected):
        """``(-i)^k`` for ``k = popcount(x & z)``, i.e. the number of Y factors.

        ``svsim`` omitted this factor entirely and so corrupted every circuit containing a ``y`` or
        ``ry`` gate. Tabulating all four values of ``k mod 4`` pins the cycle rather than a single
        case, so a sign error or a conjugated phase fails somewhere in the table.
        """
        hamiltonian = PauliSumXZ.from_paulisum(([string], [1.0]))
        assert hamiltonian.c[0][0] == pytest.approx(expected)

    @pytest.mark.parametrize("string", ["XY", "YY", "YYY", "XZY"])
    def test_dense_reconstruction_matches(self, string):
        """The folded phase must reproduce the actual Pauli operator, not just a plausible number.

        Reconstructs ``(-i)^{x.z} Z^z X^x`` from the stored signatures and compares against the
        Kronecker product of the named Paulis -- the end-to-end statement of the convention.
        """
        num_qubits = len(string)
        hamiltonian = PauliSumXZ.from_paulisum(([string], [1.0]))
        xbits = signature_bits(hamiltonian.x[0], num_qubits)
        zbits = signature_bits(hamiltonian.z[0][0], num_qubits)
        coeff = complex(hamiltonian.c[0][0])

        # Z^z X^x with little-endian qubit indexing, then the coefficient (which carries the phase).
        # gate_unitary("x"/"z", [q], n) returns the bare Pauli embedded at little-endian index q,
        # which is exactly the single-qubit factor needed here.
        operator = np.eye(2**num_qubits, dtype=np.complex128)
        for qubit in range(num_qubits):
            if xbits[qubit]:
                operator = gate_unitary("x", [qubit], num_qubits) @ operator
        for qubit in range(num_qubits):
            if zbits[qubit]:
                operator = gate_unitary("z", [qubit], num_qubits) @ operator
        assert np.allclose(coeff * operator, dense_pauli_sum([string], [1.0]))


class TestQubitOrdering:
    """Little-endian ingest: bit ``q`` of a packed signature is qubit ``q``."""

    @pytest.mark.parametrize(
        "string,expected_x",
        [("XII", [0, 0, 1]), ("IIX", [1, 0, 0]), ("IXI", [0, 1, 0]), ("XXI", [0, 1, 1])],
    )
    def test_x_signature_is_reversed_from_the_string(self, string, expected_x):
        """The leftmost character is the highest qubit, so the bit array is the reverse.

        CLAUDE.md: "little-endian qubit ordering (Qiskit's ``.x``/``.z`` get reversed on ingest)".
        This is the convention ``sqd``'s state packing has to agree with, and a silent mismatch there
        sends every matrix element to the wrong column.
        """
        hamiltonian = PauliSumXZ.from_paulisum(([string], [1.0]))
        assert list(signature_bits(hamiltonian.x[0], len(string))) == expected_x

    @pytest.mark.parametrize(
        "string,expected_z", [("ZII", [0, 0, 1]), ("IIZ", [1, 0, 0]), ("IZI", [0, 1, 0])]
    )
    def test_z_signature_is_reversed_from_the_string(self, string, expected_z):
        hamiltonian = PauliSumXZ.from_paulisum(([string], [1.0]))
        assert list(signature_bits(hamiltonian.z[0][0], len(string))) == expected_z

    def test_matches_qiskit_ingest(self):
        """A ``SparsePauliOp`` must produce the same signatures as the equivalent string list."""
        qiskit = pytest.importorskip("qiskit")
        strings = ["XIZ", "IYY"]
        coeffs = [1.5, -0.5]
        from_strings = PauliSumXZ.from_paulisum((strings, coeffs))
        from_qiskit = PauliSumXZ.from_paulisum(qiskit.quantum_info.SparsePauliOp(strings, coeffs))
        assert np.array_equal(from_strings.x, from_qiskit.x)
        assert np.array_equal(from_strings.z, from_qiskit.z)
        assert np.allclose(from_strings.c, from_qiskit.c)


class TestGrouping:
    """Terms are grouped by unique X signature, Z groups zero-padded to a rectangle."""

    def test_shared_x_signature_groups_together(self):
        """``XIZ`` and ``XZI`` share the X signature, so they form one group with two Z rows."""
        hamiltonian = PauliSumXZ.from_paulisum((["XIZ", "XZI"], [1.0, 2.0]))
        assert hamiltonian.x.shape[0] == 1, "expected a single X group"
        assert hamiltonian.z.shape[1] == 2, "expected two Z signatures in the group"
        assert hamiltonian.c.shape == (1, 2)

    def test_distinct_x_signatures_make_distinct_groups(self):
        hamiltonian = PauliSumXZ.from_paulisum((["XII", "IXI"], [1.0, 2.0]))
        assert hamiltonian.x.shape[0] == 2

    def test_ragged_groups_are_zero_padded(self):
        """One X group with two Z terms and another with one: the array is padded to a rectangle.

        ``sqd``'s kernels iterate until they hit ``coeff == 0``, so the padding slots must be exactly
        zero -- a nonzero filler would be silently summed into the diagonal.
        """
        hamiltonian = PauliSumXZ.from_paulisum((["XIZ", "XZI", "IXI"], [1.0, 2.0, 3.0]))
        assert hamiltonian.z.shape[1] == 2
        coeffs = np.asarray(hamiltonian.c)
        # Exactly one padding slot, and it must be zero.
        assert np.count_nonzero(coeffs == 0.0) == 1

    def test_pure_z_input_makes_one_group(self):
        """Pure-Z strings all share the all-identity X signature.

        That is how one X group ends up holding many Z terms, which is the regime where ``sqd``'s
        ``compute_diagonal`` had its byte-boundary bug.
        """
        strings = ["III", "ZII", "IZI", "ZZI", "IIZ"]
        hamiltonian = PauliSumXZ.from_paulisum((strings, [1.0] * len(strings)))
        assert hamiltonian.x.shape[0] == 1
        assert hamiltonian.z.shape[1] == len(strings)


class TestCoefficientDtype:
    """``.c`` is float64 exactly when the folded phase is real, and there is no flag to force it.

    This replaces a ``force_real`` parameter that could not do what its name promised. Measured, it
    had exactly one effect: on a *complex* input coefficient it silently took ``.real``, discarding
    the imaginary part of a non-Hermitian operator. For real input it was a no-op in the even-Y case
    (the unconditional narrowing below already handles that) and impossible in the odd-Y case, where
    it emitted a warning saying so. The flag is gone; a complex coefficient now raises, matching what
    the Qiskit ingest branch always did.
    """

    @pytest.mark.parametrize("string", ["XX", "YY", "ZZ", "XZ", "YYYY"])
    def test_even_y_narrows_to_float64(self, string):
        """An even number of Ys keeps the folded phase real, so ``.c`` is float64.

        Silence matters as much as the dtype: this is the ordinary case and must not warn.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            hamiltonian = PauliSumXZ.from_paulisum(([string], [1.0]))
        assert hamiltonian.c.dtype == np.float64

    @pytest.mark.parametrize("string", ["XY", "YYY", "XZY"])
    def test_odd_y_stays_complex_silently(self, string):
        """An odd-Y string cannot be real in this convention, and that is not an error.

        The ``(-i)^{x.z}`` phase makes real input complex by construction, so complex128 here is the
        correct answer rather than a failure to force realness -- hence no warning. Callers restricted
        to float64 (a backend with no complex128, say) check ``.c.dtype`` and raise on exactly that;
        there is deliberately no flag to request realness, since no flag could grant it.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            hamiltonian = PauliSumXZ.from_paulisum(([string], [1.0]))
        assert hamiltonian.c.dtype == np.complex128

    def test_complex_coefficient_raises(self):
        """A complex coefficient means a non-Hermitian operator, which every consumer assumes away.

        This used to warn and silently truncate to ``.real`` under ``force_real=True`` -- turning an
        invalid Hamiltonian into a plausible valid one, the failure mode this package keeps hitting.
        """
        with pytest.raises(ValueError, match="must be real for the Hamiltonian to be Hermitian"):
            PauliSumXZ.from_paulisum((["XX"], [1.0 + 1.0j]))

    def test_complex_coefficient_raises_for_qiskit_input_too(self):
        """Both ingest paths must agree; the check is hoisted so neither can drift.

        The Qiskit branch always raised while the tuple branch warned and truncated -- two answers to
        the same question depending on which type you passed in.
        """
        qiskit = pytest.importorskip("qiskit")
        with pytest.raises(ValueError, match="must be real for the Hamiltonian to be Hermitian"):
            PauliSumXZ.from_paulisum(qiskit.quantum_info.SparsePauliOp(["XX"], [1.0 + 1.0j]))


class TestPadding:
    """The pad bit at position 0 is intrinsic: every signature is shifted right by one bit.

    These assertions are deliberately *absolute* rather than relative. They used to compare a
    ``add_padding=True`` build against an ``add_padding=False`` one, but with the flag gone there is
    no unpadded build to difference against -- and a relative test rewritten naively becomes two
    identical calls, which passes no matter what the padding does. So each check names the bit
    position or the phase value it expects, computed independently of ``from_paulisum``.
    """

    @pytest.mark.parametrize(
        "string,qubit",
        [("IIX", 0), ("IXI", 1), ("XII", 2), ("IIIIIIIIX", 0), ("XIIIIIIII", 8)],
    )
    def test_pad_bit_is_always_reserved(self, string, qubit):
        """The set bit must sit one position right of where an unpadded packing would put it.

        ``sqd`` packs its states with a leading zero pad bit and needs the Hamiltonian to match. A
        mismatch is exactly the ``hproj`` bug: the Hamiltonian was padded while the states were not,
        so every matrix element landed in the wrong column -- silently, since the result stays
        symmetric. Making the padding unconditional is what removes that failure mode, and this pins
        it. The 9-qubit cases cross a byte boundary, where an off-by-one in the pad shift would move
        the bit into the wrong byte entirely.
        """
        num_qubits = len(string)
        packed = np.atleast_1d(np.asarray(PauliSumXZ.from_paulisum(([string], [1.0])).x[0]))
        bits = np.unpackbits(packed)
        # Character order is most-significant-qubit-first, so qubit q is character num_qubits-1-q;
        # the pad bit occupies index 0, pushing the payload one place right.
        expected_index = 1 + (num_qubits - 1 - qubit)
        assert np.flatnonzero(bits).tolist() == [expected_index]

    @pytest.mark.parametrize("string,expected", [("XY", -1.0j), ("YY", -1.0), ("XX", 1.0)])
    def test_pad_bit_does_not_change_the_coefficients(self, string, expected):
        """The pad bit is a dummy identity, so it must not perturb the ``(-i)^{x.z}`` phase.

        Checked against the phase table in :class:`TestPhaseConvention` rather than against a second
        build: an identity factor contributes nothing to ``popcount(x & z)``, so the padded
        coefficient must equal the bare convention value.
        """
        hamiltonian = PauliSumXZ.from_paulisum(([string], [1.5]))
        assert hamiltonian.c[0][0] == pytest.approx(1.5 * expected)

    @pytest.mark.parametrize("num_qubits", [1, 4, 7, 8, 9, 16, 17])
    def test_pack_states_puts_the_payload_one_bit_right(self, num_qubits):
        """``pack_states`` is the states half of the alignment contract, so it pads identically.

        Checked bit-by-bit against an independent unpacking rather than against ``from_paulisum``:
        both sides agreeing on a *wrong* shift is the failure mode this contract exists to prevent,
        so a test that only compares them to each other would pass through it. The widths straddle
        byte boundaries, where an off-by-one moves the payload into the wrong byte.
        """
        rng = np.random.default_rng(20260807 + num_qubits)
        states = rng.integers(0, 2, size=(5, num_qubits), dtype=np.uint8)
        packed = PauliSumXZ.pack_states(states)

        assert packed.shape == (5, -(-(num_qubits + 1) // 8))
        bits = np.unpackbits(packed, axis=1)
        assert np.all(bits[:, 0] == 0), "pad bit must be zero"
        assert np.array_equal(bits[:, 1 : 1 + num_qubits], states)
        # Trailing bits are packbits' own zero fill, not payload.
        assert np.all(bits[:, 1 + num_qubits :] == 0)

    @pytest.mark.parametrize("num_qubits", [1, 4, 7, 8, 9, 16, 17])
    def test_unpack_states_inverts_pack_states(self, num_qubits):
        """The ``+1`` offset must agree between the two directions, or the round trip shifts."""
        rng = np.random.default_rng(20260807 - num_qubits)
        states = rng.integers(0, 2, size=(6, num_qubits), dtype=np.uint8)
        recovered = PauliSumXZ.unpack_states(PauliSumXZ.pack_states(states), num_qubits)
        assert np.array_equal(recovered, states)

    @pytest.mark.parametrize("string", ["XIIXXIXII", "XI", "XXXXXXXX", "IIIIIIIIX"])
    def test_pack_states_agrees_with_the_signature_padding(self, string):
        """A state and the X signature of the same bitstring must land on identical bytes.

        This is the contract itself: ``sqd`` XORs packed states against packed signatures, so if the
        two paddings diverge the operation is nonsense. A Pauli string of Xs and Is has an X signature
        that *is* the bitstring, in character order.

        Both are compared against a third, independently computed byte sequence rather than only
        against each other. ``from_paulisum`` pads its X side by *calling* ``pack_states``, so the two
        agreeing proves only that one function is self-consistent -- break it and this assertion would
        still hold while every packed state silently shifted. The expected bytes here are built from
        the bit positions the "Bit layout" docstring specifies, so the test pins the layout itself.
        """
        num_qubits = len(string)
        bits = [1 if ch == "X" else 0 for ch in string]
        # Pad bit at index 0, payload at 1..n in character order, packbits' zero fill after.
        expected_bits = np.array([0] + bits, dtype=np.uint8)
        expected = np.packbits(np.pad(expected_bits, (0, -len(expected_bits) % 8)))

        packed_state = PauliSumXZ.pack_states(np.array([bits], dtype=np.uint8))[0]
        signature = np.atleast_1d(np.asarray(PauliSumXZ.from_paulisum(([string], [1.0])).x[0]))

        assert np.array_equal(packed_state, expected), f"pack_states for {num_qubits} qubits"
        assert np.array_equal(signature, expected), f"signature for {num_qubits} qubits"


class TestDataclass:
    """``PauliSumXZ`` is a frozen, JAX-registered dataclass."""

    def test_arrays_property_returns_the_three_fields(self):
        hamiltonian = PauliSumXZ.from_paulisum((["XX"], [1.0]))
        arrays = hamiltonian.arrays
        assert len(arrays) == 3
        assert arrays[0] is hamiltonian.x
        assert arrays[1] is hamiltonian.z
        assert arrays[2] is hamiltonian.c

    def test_is_frozen(self):
        hamiltonian = PauliSumXZ.from_paulisum((["XX"], [1.0]))
        with pytest.raises((AttributeError, TypeError)):
            hamiltonian.num_qubits = 5

    def test_is_a_valid_jax_pytree(self):
        """Registered via ``register_dataclass``, so it can cross a ``jit`` boundary.

        ``sqd`` passes one straight into a jitted function, so this is load-bearing rather than
        incidental.
        """
        import jax

        hamiltonian = PauliSumXZ.from_paulisum((["XX", "ZZ"], [1.0, 2.0]))
        leaves, treedef = jax.tree.flatten(hamiltonian)
        assert len(leaves) == 3, "x, z, c are leaves; num_qubits is static metadata"
        rebuilt = jax.tree.unflatten(treedef, leaves)
        assert rebuilt.num_qubits == hamiltonian.num_qubits

    @pytest.mark.parametrize("strings,expected", [(["X"], 1), (["XX"], 2), (["XIZI"], 4)])
    def test_num_qubits_is_inferred(self, strings, expected):
        hamiltonian = PauliSumXZ.from_paulisum((strings, [1.0]))
        assert hamiltonian.num_qubits == expected


class TestValidation:
    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError, match="Lengths of Pauli and coeff"):
            PauliSumXZ.from_paulisum((["XX", "ZZ"], [1.0]))

    def test_non_uniform_string_lengths_raise(self):
        with pytest.raises(ValueError, match="non-uniform lengths"):
            PauliSumXZ.from_paulisum((["XX", "ZZZ"], [1.0, 2.0]))

    def test_zero_coefficients_are_dropped(self):
        """A zero coefficient contributes nothing and is removed before grouping."""
        hamiltonian = PauliSumXZ.from_paulisum((["XX", "ZZ"], [1.0, 0.0]))
        assert hamiltonian.x.shape[0] == 1

    def test_lowercase_strings_are_accepted(self):
        """``from_paulisum`` upper-cases the strings, so lowercase input must work."""
        lower = PauliSumXZ.from_paulisum((["xy"], [1.0]))
        upper = PauliSumXZ.from_paulisum((["XY"], [1.0]))
        assert np.array_equal(lower.x, upper.x)
        assert np.allclose(lower.c, upper.c)
