"""Tests for :mod:`rqutils.paulis.symplectic`.

``PauliSumXZ`` is the bit-packed qubit-only Pauli representation that ``sqd`` and ``svsim`` both
build on. Its two conventions are the ones worth pinning, because getting either wrong is silent:

- ``Q = (-i)^{x.z} Z^z X^x`` with the ``(-i)^{popcount(x & z)}`` phase folded into the
  coefficients. **This is the same convention whose omission broke** ``svsim`` -- it dropped the
  factor entirely and corrupted every ``y``/``ry`` gate, which is exactly the class of error a test
  here would have caught earlier (see ``docs/skqd.md``).
- **Little-endian qubit ordering**: Qiskit's ``.x``/``.z`` are reversed on ingest, so bit ``q`` of a
  packed signature is qubit ``q``. ``sqd``'s tests cover this transitively; these cover it directly.

One gap was found and fixed while writing these: ``force_real=True`` validated only the *input*
coefficients, so an odd-Y Pauli string came back complex128 with no warning at all -- see
:class:`TestForceReal`.
"""

import functools
import warnings

import jax.numpy as jnp
import numpy as np
import pytest
from conftest import _PAULI_MATRICES

from rqutils.paulis.symplectic import PauliSumXZ


def dense_from_strings(pauli_strings, coeffs):
    """Return ``sum(c * Q)`` as a dense matrix, via Kronecker products.

    Character order is most-significant-qubit-first, matching Qiskit's ``SparsePauliOp``. Independent
    of the bit-packing under test.
    """
    total = 0.0
    for string, coeff in zip(pauli_strings, coeffs):
        operator = functools.reduce(
            np.kron, [_PAULI_MATRICES[char] for char in string], np.array([[1.0 + 0.0j]])
        )
        total = total + coeff * operator
    return total


def signature_bits(packed, num_qubits):
    """Unpack a bit-packed signature into a qubit-0-first bit array.

    ``np.packbits`` fills each byte from the most significant end, so the ``num_qubits`` payload bits
    are the *first* entries of the unpacked array, not the last. Within that slice the order matches
    the Pauli string's character order (leftmost character first), which is the opposite of the
    little-endian qubit indexing -- hence the reversal here to index by qubit number.
    """
    bits = np.unpackbits(np.atleast_1d(np.asarray(packed, dtype=np.uint8)))
    return bits[:num_qubits][::-1]


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
        hamiltonian = PauliSumXZ.from_paulisum(([string], [1.0]), add_padding=False)
        assert hamiltonian.c[0][0] == pytest.approx(expected)

    @pytest.mark.parametrize("string", ["XY", "YY", "YYY", "XZY"])
    def test_dense_reconstruction_matches(self, string):
        """The folded phase must reproduce the actual Pauli operator, not just a plausible number.

        Reconstructs ``(-i)^{x.z} Z^z X^x`` from the stored signatures and compares against the
        Kronecker product of the named Paulis -- the end-to-end statement of the convention.
        """
        num_qubits = len(string)
        hamiltonian = PauliSumXZ.from_paulisum(([string], [1.0]), add_padding=False)
        xbits = signature_bits(hamiltonian.x[0], num_qubits)
        zbits = signature_bits(hamiltonian.z[0][0], num_qubits)
        coeff = complex(hamiltonian.c[0][0])

        # Z^z X^x with little-endian qubit indexing, then the coefficient (which carries the phase).
        dimension = 2**num_qubits
        operator = np.eye(dimension, dtype=np.complex128)
        for qubit in range(num_qubits):
            if xbits[qubit]:
                operator = single_qubit(dimension, qubit, "X") @ operator
        for qubit in range(num_qubits):
            if zbits[qubit]:
                operator = single_qubit(dimension, qubit, "Z") @ operator
        assert np.allclose(coeff * operator, dense_from_strings([string], [1.0]))


def single_qubit(dimension, qubit, letter):
    """Return the ``dimension``-sized operator applying ``letter`` to little-endian ``qubit``."""
    num_qubits = int(np.log2(dimension))
    factors = [
        _PAULI_MATRICES[letter if index == qubit else "I"] for index in reversed(range(num_qubits))
    ]
    return functools.reduce(np.kron, factors, np.array([[1.0 + 0.0j]]))


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
        hamiltonian = PauliSumXZ.from_paulisum(([string], [1.0]), add_padding=False)
        assert list(signature_bits(hamiltonian.x[0], len(string))) == expected_x

    @pytest.mark.parametrize(
        "string,expected_z", [("ZII", [0, 0, 1]), ("IIZ", [1, 0, 0]), ("IZI", [0, 1, 0])]
    )
    def test_z_signature_is_reversed_from_the_string(self, string, expected_z):
        hamiltonian = PauliSumXZ.from_paulisum(([string], [1.0]), add_padding=False)
        assert list(signature_bits(hamiltonian.z[0][0], len(string))) == expected_z

    def test_matches_qiskit_ingest(self):
        """A ``SparsePauliOp`` must produce the same signatures as the equivalent string list."""
        qiskit = pytest.importorskip("qiskit")
        strings = ["XIZ", "IYY"]
        coeffs = [1.5, -0.5]
        from_strings = PauliSumXZ.from_paulisum((strings, coeffs), add_padding=False)
        from_qiskit = PauliSumXZ.from_paulisum(
            qiskit.quantum_info.SparsePauliOp(strings, coeffs), add_padding=False
        )
        assert np.array_equal(from_strings.x, from_qiskit.x)
        assert np.array_equal(from_strings.z, from_qiskit.z)
        assert np.allclose(from_strings.c, from_qiskit.c)


class TestGrouping:
    """Terms are grouped by unique X signature, Z groups zero-padded to a rectangle."""

    def test_shared_x_signature_groups_together(self):
        """``XIZ`` and ``XZI`` share the X signature, so they form one group with two Z rows."""
        hamiltonian = PauliSumXZ.from_paulisum((["XIZ", "XZI"], [1.0, 2.0]), add_padding=False)
        assert hamiltonian.x.shape[0] == 1, "expected a single X group"
        assert hamiltonian.z.shape[1] == 2, "expected two Z signatures in the group"
        assert hamiltonian.c.shape == (1, 2)

    def test_distinct_x_signatures_make_distinct_groups(self):
        hamiltonian = PauliSumXZ.from_paulisum((["XII", "IXI"], [1.0, 2.0]), add_padding=False)
        assert hamiltonian.x.shape[0] == 2

    def test_ragged_groups_are_zero_padded(self):
        """One X group with two Z terms and another with one: the array is padded to a rectangle.

        ``sqd``'s kernels iterate until they hit ``coeff == 0``, so the padding slots must be exactly
        zero -- a nonzero filler would be silently summed into the diagonal.
        """
        hamiltonian = PauliSumXZ.from_paulisum(
            (["XIZ", "XZI", "IXI"], [1.0, 2.0, 3.0]), add_padding=False
        )
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
        hamiltonian = PauliSumXZ.from_paulisum((strings, [1.0] * len(strings)), add_padding=False)
        assert hamiltonian.x.shape[0] == 1
        assert hamiltonian.z.shape[1] == len(strings)


class TestForceReal:
    """``force_real=True`` is best-effort: it warns rather than raising, and cannot always succeed."""

    @pytest.mark.parametrize("string", ["XX", "YY", "ZZ"])
    def test_even_y_gives_real_coefficients(self, string):
        """An even number of Ys keeps the phase real, so the result is float64."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            hamiltonian = PauliSumXZ.from_paulisum(
                ([string], [1.0]), force_real=True, add_padding=False
            )
        assert hamiltonian.c.dtype == np.float64

    def test_odd_y_warns_and_stays_complex(self):
        """``force_real=True`` cannot force an odd-Y string real, and now says so.

        The check originally ran on the *input* coefficients only, but the ``(-i)^{x.z}`` phase is
        folded in afterwards -- so real input became complex output and ``force_real=True`` returned
        complex128 silently. Callers were left to notice on their own:
        ``examples/_bench_common.build_solver_inputs`` raises on ``.c.dtype != np.float64`` for
        exactly this reason. It now warns; it deliberately still does not raise, matching the
        pre-existing check's best-effort semantics.
        """
        with pytest.warns(UserWarning, match="force_real=True"):
            hamiltonian = PauliSumXZ.from_paulisum(
                (["XY"], [1.0]), force_real=True, add_padding=False
            )
        assert hamiltonian.c.dtype == np.complex128

    def test_complex_input_warns(self):
        """The original pre-phase check: a genuinely complex input coefficient."""
        with pytest.warns(UserWarning, match="imaginary part"):
            PauliSumXZ.from_paulisum((["XX"], [1.0 + 1.0j]), force_real=True, add_padding=False)

    def test_force_real_false_is_silent(self):
        """Without the flag, complex coefficients are expected and must not warn."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            hamiltonian = PauliSumXZ.from_paulisum(
                (["XY"], [1.0]), force_real=False, add_padding=False
            )
        assert hamiltonian.c.dtype == np.complex128


class TestPadding:
    """``add_padding=True`` shifts every signature right by one bit."""

    def test_padding_shifts_the_signatures(self):
        """``sqd`` packs its states with a leading zero pad bit and needs the Hamiltonian to match.

        A mismatch here is exactly the ``hproj`` bug: the Hamiltonian was padded while the states were
        not, so every matrix element landed in the wrong column.
        """
        unpadded = PauliSumXZ.from_paulisum((["IIX"], [1.0]), add_padding=False)
        padded = PauliSumXZ.from_paulisum((["IIX"], [1.0]), add_padding=True)
        assert int(np.asarray(unpadded.x[0])[0]) == 2 * int(np.asarray(padded.x[0])[0]) or (
            # Equivalently: the padded signature has its bit one position further right.
            np.argmax(np.unpackbits(np.atleast_1d(np.asarray(padded.x[0]))))
            == np.argmax(np.unpackbits(np.atleast_1d(np.asarray(unpadded.x[0])))) + 1
        )

    def test_padding_does_not_change_the_coefficients(self):
        """The pad bit is a dummy identity, so it must not touch the phase."""
        unpadded = PauliSumXZ.from_paulisum((["XY"], [1.5]), add_padding=False)
        padded = PauliSumXZ.from_paulisum((["XY"], [1.5]), add_padding=True)
        assert np.allclose(np.asarray(unpadded.c), np.asarray(padded.c))


class TestMatmul:
    """``matmul`` applies the Pauli sum to a vector, matrix-free."""

    @pytest.mark.parametrize(
        "strings,coeffs",
        [
            (["XX"], [1.0]),
            (["ZZ"], [2.0]),
            (["XX", "ZZ"], [1.0, 2.0]),
            (["XI", "IZ", "YY"], [0.5, -1.0, 2.0]),
        ],
    )
    def test_matches_the_dense_product(self, strings, coeffs):
        """Against ``dense @ v``, which shares no code with the bit-packed gather."""
        hamiltonian = PauliSumXZ.from_paulisum((strings, coeffs), add_padding=False)
        dimension = 2 ** len(strings[0])
        rng = np.random.default_rng(20260804)
        vector = rng.normal(size=dimension) + 1.0j * rng.normal(size=dimension)
        got = np.asarray(hamiltonian.matmul(jnp.asarray(vector)))
        expected = dense_from_strings(strings, coeffs) @ vector
        assert np.abs(got - expected).max() < 1e-12

    def test_wrong_vector_length_raises(self):
        hamiltonian = PauliSumXZ.from_paulisum((["XX"], [1.0]), add_padding=False)
        with pytest.raises(ValueError, match="incompatible with num_qubits"):
            hamiltonian.matmul(jnp.zeros(8))

    def test_requires_a_jax_array(self):
        """A numpy array fails on ``.at[]`` with an opaque AttributeError.

        Recorded rather than fixed: the method is JAX-only by design (it is the matrix-free kernel
        ``sqd`` traces), but the error names ``.at`` rather than the argument, so this test is the
        documentation.
        """
        hamiltonian = PauliSumXZ.from_paulisum((["XX"], [1.0]), add_padding=False)
        with pytest.raises(AttributeError, match="'at'"):
            hamiltonian.matmul(np.zeros(4))


class TestDataclass:
    """``PauliSumXZ`` is a frozen, JAX-registered dataclass."""

    def test_arrays_property_returns_the_three_fields(self):
        hamiltonian = PauliSumXZ.from_paulisum((["XX"], [1.0]), add_padding=False)
        arrays = hamiltonian.arrays
        assert len(arrays) == 3
        assert arrays[0] is hamiltonian.x
        assert arrays[1] is hamiltonian.z
        assert arrays[2] is hamiltonian.c

    def test_is_frozen(self):
        hamiltonian = PauliSumXZ.from_paulisum((["XX"], [1.0]), add_padding=False)
        with pytest.raises((AttributeError, TypeError)):
            hamiltonian.num_qubits = 5

    def test_is_a_valid_jax_pytree(self):
        """Registered via ``register_dataclass``, so it can cross a ``jit`` boundary.

        ``sqd`` passes one straight into a jitted function, so this is load-bearing rather than
        incidental.
        """
        import jax

        hamiltonian = PauliSumXZ.from_paulisum((["XX", "ZZ"], [1.0, 2.0]), add_padding=False)
        leaves, treedef = jax.tree.flatten(hamiltonian)
        assert len(leaves) == 3, "x, z, c are leaves; num_qubits is static metadata"
        rebuilt = jax.tree.unflatten(treedef, leaves)
        assert rebuilt.num_qubits == hamiltonian.num_qubits

    def test_num_qubits_is_inferred(self):
        for strings, expected in ((["X"], 1), (["XX"], 2), (["XIZI"], 4)):
            hamiltonian = PauliSumXZ.from_paulisum((strings, [1.0]), add_padding=False)
            assert hamiltonian.num_qubits == expected


class TestValidation:
    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError, match="Lengths of Pauli and coeff"):
            PauliSumXZ.from_paulisum((["XX", "ZZ"], [1.0]), add_padding=False)

    def test_non_uniform_string_lengths_raise(self):
        with pytest.raises(ValueError, match="non-uniform lengths"):
            PauliSumXZ.from_paulisum((["XX", "ZZZ"], [1.0, 2.0]), add_padding=False)

    def test_zero_coefficients_are_dropped(self):
        """A zero coefficient contributes nothing and is removed before grouping."""
        hamiltonian = PauliSumXZ.from_paulisum((["XX", "ZZ"], [1.0, 0.0]), add_padding=False)
        assert hamiltonian.x.shape[0] == 1

    def test_lowercase_strings_are_accepted(self):
        """``from_paulisum`` upper-cases the strings, so lowercase input must work."""
        lower = PauliSumXZ.from_paulisum((["xy"], [1.0]), add_padding=False)
        upper = PauliSumXZ.from_paulisum((["XY"], [1.0]), add_padding=False)
        assert np.array_equal(lower.x, upper.x)
        assert np.allclose(lower.c, upper.c)
