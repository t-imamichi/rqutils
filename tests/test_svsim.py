"""Tests for :mod:`rqutils.svsim`.

Written from ``docs/skqd.md``, which recorded ``svsim`` as *the* blocker for sample-based Krylov
quantum diagonalization: ``do_svsim`` never applied the ``(-i)^{x.z}`` factor from the module's own
documented convention ``Q = (-i)^{x.z} Z^z X^x``. That hit exactly the gates whose X and Z
signatures overlap -- ``y`` and ``ry`` -- and since the documented workflow transpiles to
``['rx','ry','rz','rzz']``, which emits ``ry`` constantly, it corrupted essentially every nontrivial
circuit. Measured before the fix: isolated ``ry(0.7)`` on |0> gave ``[0.939, +0.343j]`` instead of
``[0.939, +0.343]``, a 5-qubit GHZ had ``|overlap|`` 0.5 against Qiskit, and a 6-qubit 4-rep Trotter
step had 1e-16.

The reference is ``conftest.simulate_dense``: dense Kronecker-product gate matrices applied by
matrix multiplication, sharing no code with the symplectic ``CircuitXZ`` representation or the
``lax.scan`` kernel under test. Every gate unitary it builds was validated against
``qiskit.quantum_info.Operator`` before being trusted, but the tests themselves need no qiskit,
which is only an optional extra here. The qiskit-dependent tests are marked and skip cleanly.

A note on global phases. Most assertions here are *exact*, not up-to-phase: a global phase stops
being global the moment a caller superposes two simulations, and skqd.md's whole point is that a
phase error was hiding in plain sight. The one exception is ``cz``, whose decomposition carries an
irreducible ``exp(i*pi/4)`` -- see :meth:`TestCz.test_cz_matches_up_to_global_phase`.
"""

import numpy as np
import pytest
from conftest import gate_unitary, phaseless_distance, run_sharded_child, simulate_dense

from rqutils.svsim import CircuitXZ, svsim, to_circuitxz

SINGLE_QUBIT_GATES = ["x", "y", "z"]
ROTATION_GATES = ["rx", "ry", "rz"]


def spec_of(name, qubits, angle=None):
    return (name, qubits) if angle is None else (name, qubits, angle)


class TestSymplecticPhase:
    """The ``(-i)^{x.z}`` factor of ``Q = (-i)^{x.z} Z^z X^x`` must actually be applied.

    Its absence was ``docs/skqd.md``'s blocker. These tests target the phase directly rather than
    through a whole circuit, because a single missing factor of ``i`` on one gate type is invisible
    in any check that only looks at magnitudes -- which is exactly how it survived.
    """

    def test_ry_is_real_on_a_real_input(self):
        """``ry`` on |0> must stay real: the documented failure, in its simplest form.

        Before the fix this returned ``[0.939373, +0.342898j]`` -- correct magnitudes, rotated into
        the imaginary axis. Asserting on ``abs()`` would have passed.
        """
        state = np.asarray(svsim([("ry", [0], 0.7)]))
        assert np.abs(state.imag).max() < 1e-15, f"ry produced imaginary amplitudes: {state}"
        assert state[0].real == pytest.approx(np.cos(0.35), abs=1e-12)
        assert state[1].real == pytest.approx(np.sin(0.35), abs=1e-12)

    @pytest.mark.parametrize("name", ["y", "ry"])
    def test_overlapping_signature_gates(self, name):
        """``y`` and ``ry`` are the only supported gates with ``popcount(x & z) == 1``.

        Everything else has ``x.z == 0``, so the missing phase was a no-op for them; these two
        carried the entire discrepancy. Verified here through the built ``CircuitXZ`` as well, so
        the test states *why* these gates are the sensitive ones rather than just that they work.
        """
        angle = 0.7
        circuit = to_circuitxz([spec_of(name, [0], None if name == "y" else angle)])
        popcount = int(np.bitwise_count(int(circuit.x[0]) & int(circuit.z[0])))
        assert popcount == 1, f"{name} was expected to have overlapping X/Z signatures"

        expected = simulate_dense([spec_of(name, [0], None if name == "y" else angle)], 1)
        assert np.allclose(np.asarray(svsim(circuit)), expected, atol=1e-12)

    @pytest.mark.parametrize("name", ["x", "z", "rx", "rz"])
    def test_non_overlapping_signature_gates(self, name):
        """The control group: ``x.z == 0``, so these were correct even before the fix.

        Keeping them explicit documents the boundary of the bug and guards the opposite regression
        -- a fix that over-applied the phase would break precisely these.
        """
        angle = 0.7
        circuit = to_circuitxz([spec_of(name, [0], None if name in ("x", "z") else angle)])
        assert int(circuit.x[0]) & int(circuit.z[0]) == 0

        expected = simulate_dense([spec_of(name, [0], None if name in ("x", "z") else angle)], 1)
        assert np.allclose(np.asarray(svsim(circuit)), expected, atol=1e-12)

    def test_sin_is_complex(self):
        """``CircuitXZ.sin`` must be complex to hold the phase at all.

        ``skqd.md`` noted the fix "is not a one-liner: ``sin`` is a float64 array, so the complex
        phase cannot be folded into it as-is". Widening it is what made the fix possible, so a
        future narrowing back to float64 would silently reintroduce the bug -- catch it here.
        """
        circuit = to_circuitxz([("ry", [0], 0.7)])
        assert np.iscomplexobj(circuit.sin)
        # For ry the folded factor is i * (-i) = 1, so the amplitude is real-valued but complex-typed.
        assert circuit.sin[0].imag == pytest.approx(0.0, abs=1e-15)
        # For rx it is i * (-i)^0 = i, so the amplitude is genuinely imaginary.
        assert to_circuitxz([("rx", [0], 0.7)]).sin[0].real == pytest.approx(0.0, abs=1e-15)


class TestPauliGates:
    """``x``/``y``/``z`` are bare Pauli gates, not pi rotations."""

    @pytest.mark.parametrize("name", SINGLE_QUBIT_GATES)
    @pytest.mark.parametrize("index", [0, 1, 2])
    def test_matches_dense(self, name, index):
        """Exact, including phase.

        The ``"pi"`` shortcut in ``to_circuitxz`` builds these from the rotation formula, and
        ``R_P(pi) = -i P``, so without compensation ``x`` and ``z`` came back with a ``-i`` global
        phase relative to the Pauli (pre-existing, independent of the ``(-i)^{x.z}`` bug) while
        ``y`` matched only because the two errors happened to cancel. All three are now exact.
        """
        num_qubits = 3
        # An rx(0) on the top qubit is an exact identity, and it pins num_qubits to 3 regardless of
        # which qubit the gate under test acts on -- so the dense reference and svsim agree on
        # dimension without the test having to skip for index < 2.
        specs = [spec_of(name, [index]), ("rx", [num_qubits - 1], 0.0)]
        assert to_circuitxz(specs).num_qubits == num_qubits
        expected = simulate_dense(specs, num_qubits)
        assert np.allclose(np.asarray(svsim(specs)), expected, atol=1e-12)

    @pytest.mark.parametrize("name", SINGLE_QUBIT_GATES)
    def test_applied_twice_is_identity(self, name):
        """``P^2 = I`` exactly, which pins the phase without needing an external reference.

        A residual ``-i`` per application would show up here as ``-1`` on the state, so this is a
        self-contained check on the Pauli normalization.
        """
        initial = np.array([0.6, 0.8], dtype=np.complex128)
        state = np.asarray(svsim([spec_of(name, [0]), spec_of(name, [0])], initial_state=initial))
        assert np.allclose(state, initial, atol=1e-12)


class TestRotationGates:
    """The rotation gates, against the dense reference over several angles and qubits."""

    @pytest.mark.parametrize("name", ROTATION_GATES)
    @pytest.mark.parametrize("angle", [0.0, 0.3, np.pi / 2, np.pi, 2.5, -0.8])
    def test_single_rotation(self, name, angle):
        """Includes ``angle=0`` (identity) and ``angle=pi``, where ``cos`` vanishes.

        ``do_svsim`` branches on ``gate.cos == 0.0`` via ``lax.cond``, so ``pi`` is the input that
        exercises the other side of that branch.
        """
        expected = simulate_dense([spec_of(name, [0], angle)], 1)
        assert np.allclose(np.asarray(svsim([spec_of(name, [0], angle)])), expected, atol=1e-12)

    @pytest.mark.parametrize("name", ROTATION_GATES)
    def test_rotation_on_a_superposition(self, name):
        """A superposed input makes every amplitude's phase observable.

        Starting from |0> leaves half the entries zero, which can hide a phase error on those
        entries entirely.
        """
        initial = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.complex128)
        specs = [spec_of(name, [0], 0.9), spec_of(name, [1], -0.4)]
        expected = simulate_dense(specs, 2, initial_state=initial)
        got = np.asarray(svsim(specs, initial_state=initial))
        assert np.allclose(got, expected, atol=1e-12)

    def test_rzz_two_qubit(self):
        """``rzz`` acts on a qubit pair; its X signature is zero so it is diagonal."""
        initial = np.ones(8, dtype=np.complex128) / np.sqrt(8)
        specs = [("rzz", [0, 2], 0.7)]
        expected = simulate_dense(specs, 3, initial_state=initial)
        got = np.asarray(svsim(specs, initial_state=initial))
        assert np.allclose(got, expected, atol=1e-12)

    def test_rotation_is_unitary(self):
        """Norm preservation, over a random circuit -- a necessary condition the phase bug passed.

        Included precisely because it is weak: the pre-fix code was norm-preserving too. It guards
        a different failure class (a dropped or double-counted amplitude) than the phase tests do.
        """
        rng = np.random.default_rng(20260804)
        num_qubits = 4
        specs = [
            spec_of(str(rng.choice(ROTATION_GATES)), [int(rng.integers(num_qubits))], rng.normal())
            for _ in range(12)
        ]
        initial = rng.normal(size=2**num_qubits) + 1.0j * rng.normal(size=2**num_qubits)
        initial /= np.linalg.norm(initial)
        got = np.asarray(svsim(specs, initial_state=initial))
        assert np.linalg.norm(got) == pytest.approx(1.0, abs=1e-12)


class TestCircuitStructure:
    """``to_circuitxz`` translation, independent of the simulation kernel."""

    def test_qubit_count_inference(self):
        """Without a QuantumCircuit, ``num_qubits`` is inferred as ``max(qubit) + 1``."""
        assert to_circuitxz([("x", [0])]).num_qubits == 1
        assert to_circuitxz([("x", [3])]).num_qubits == 4
        assert to_circuitxz([("rzz", [1, 5], 0.3)]).num_qubits == 6

    def test_unsupported_gate_raises(self):
        with pytest.raises(ValueError, match="Unsupported gate name"):
            to_circuitxz([("h", [0])])

    @pytest.mark.parametrize("name", ["rx", "ry", "rz", "rzz"])
    def test_rotation_without_angle_raises(self, name):
        """A parameterized gate missing its angle must say so, not raise a bare IndexError."""
        with pytest.raises(ValueError, match="requires an angle"):
            to_circuitxz([(name, [0])])

    def test_circuitxz_passes_through(self):
        """``svsim`` accepts a prebuilt ``CircuitXZ``, and ``to_circuitxz`` is idempotent on one."""
        circuit = to_circuitxz([("ry", [0], 0.7)])
        assert to_circuitxz(circuit) is circuit
        expected = simulate_dense([("ry", [0], 0.7)], 1)
        assert np.allclose(np.asarray(svsim(circuit)), expected, atol=1e-12)

    def test_empty_circuit_is_identity(self):
        """No gates means the initial state comes back untouched."""
        initial = np.array([0.6, 0.0, 0.0, 0.8], dtype=np.complex128)
        circuit = CircuitXZ(
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.complex128),
            2,
        )
        assert np.allclose(np.asarray(svsim(circuit, initial_state=initial)), initial, atol=1e-12)


class TestInitialState:
    """``initial_state`` accepts a one-hot index or a full vector."""

    @pytest.mark.parametrize("index", [0, 1, 5, 7])
    def test_one_hot_index(self, index):
        """An integer selects a basis state; the identity circuit must return exactly that."""
        state = np.asarray(svsim([("rx", [0], 0.0)], initial_state=index))
        expected = np.zeros(8, dtype=np.complex128)
        expected[index] = 1.0
        # rx(0) is the identity, so this also pins the bit order of the one-hot construction.
        assert np.allclose(state, expected[: len(state)], atol=1e-12)

    def test_vector_initial_state(self):
        rng = np.random.default_rng(20260804)
        initial = rng.normal(size=4) + 1.0j * rng.normal(size=4)
        initial /= np.linalg.norm(initial)
        specs = [("ry", [0], 0.5), ("rzz", [0, 1], 0.9)]
        expected = simulate_dense(specs, 2, initial_state=initial)
        got = np.asarray(svsim(specs, initial_state=initial))
        assert np.allclose(got, expected, atol=1e-12)

    def test_qubit_index_is_bit_index(self):
        """Qubit ``q`` is bit ``q`` of the statevector index (little-endian).

        This is the convention ``sqd`` has to agree with for SKQD to work at all -- ``skqd.md``
        checked it explicitly ("no endianness trap between the two modules"), so pin it here.
        """
        # x on qubit 0 maps |00> -> |01>, i.e. index 0 -> index 1.
        state = np.asarray(svsim([("x", [0])], initial_state=0))
        assert int(np.argmax(np.abs(state))) == 1
        # x on qubit 1 maps |00> -> |10>, i.e. index 0 -> index 2.
        state = np.asarray(svsim([("x", [1])], initial_state=0))
        assert int(np.argmax(np.abs(state))) == 2


class TestMultiGateCircuits:
    """Whole circuits, where the phase bug actually bit."""

    def test_random_circuit_matches_dense(self):
        """A mixed random circuit over all gate types, exact against the dense reference."""
        rng = np.random.default_rng(20260804)
        num_qubits = 4
        specs = []
        for _ in range(20):
            name = str(rng.choice(["x", "y", "z", "rx", "ry", "rz", "rzz"]))
            if name == "rzz":
                pair = rng.choice(num_qubits, size=2, replace=False)
                specs.append(("rzz", [int(pair[0]), int(pair[1])], float(rng.normal())))
            elif name in ("x", "y", "z"):
                specs.append((name, [int(rng.integers(num_qubits))]))
            else:
                specs.append((name, [int(rng.integers(num_qubits))], float(rng.normal())))
        # Force the qubit count so the dense reference and svsim agree on dimension.
        specs.append(("rx", [num_qubits - 1], 0.0))
        expected = simulate_dense(specs, num_qubits)
        got = np.asarray(svsim(specs))
        assert np.allclose(got, expected, atol=1e-11)

    def test_ry_heavy_circuit(self):
        """Many ``ry`` gates, the case the missing phase corrupted worst.

        With N sequential ``ry`` gates the pre-fix error compounded as ``i**N``, so this fails
        dramatically rather than marginally if the phase is ever dropped again.
        """
        specs = [("ry", [i % 3], 0.3 + 0.1 * i) for i in range(9)]
        expected = simulate_dense(specs, 3)
        got = np.asarray(svsim(specs))
        assert np.allclose(got, expected, atol=1e-12)
        # An all-ry circuit on |0> stays real; a dropped phase shows up as imaginary weight.
        assert np.abs(got.imag).max() < 1e-14


class TestCz:
    """``cz`` is decomposed into ``rzz`` + two ``rz`` gates, and only on the QuantumCircuit path."""

    def test_cz_as_a_gate_spec_is_rejected(self):
        """The docstring lists ``cz`` among the gate-specifier names, but ``to_circuitxz``'s match
        has no ``cz`` case -- only the QuantumCircuit branch above it decomposes one. Recorded as a
        test rather than silently fixed: the decomposition needs a qubit *pair*, and it is not clear
        that a 2-tuple gate spec should expand into three gates behind the caller's back.
        """
        with pytest.raises(ValueError, match="Unsupported gate name cz"):
            to_circuitxz([("cz", [0, 1])])

    def test_cz_matches_up_to_global_phase(self):
        """``cz`` via a QuantumCircuit is correct only up to ``exp(i*pi/4)``.

        Its decomposition ``rzz(pi/2) rz(-pi/2) rz(-pi/2)`` equals ``CZ`` times a uniform
        ``exp(i*pi/4)``, and a pure global phase cannot be expressed through the ``cos``/``sin``
        amplitudes of Pauli rotations -- fixing it exactly would need a separate phase field on
        ``CircuitXZ``. Left as-is because a *uniform* phase is unobservable for the sampling and
        expectation values SKQD needs; asserted up-to-phase here so the limitation is explicit
        rather than a surprise, and so a change in its magnitude is still caught.
        """
        pytest.importorskip("qiskit")
        from qiskit import QuantumCircuit

        circuit = QuantumCircuit(2)
        circuit.cz(0, 1)
        initial = np.ones(4, dtype=np.complex128) / 2.0
        got = np.asarray(svsim(circuit, initial_state=initial))
        expected = initial.copy()
        expected[3] *= -1.0  # CZ flips the |11> amplitude
        assert phaseless_distance(got, expected) < 1e-12
        # And pin the phase itself, so a change to the decomposition is not silently absorbed.
        ratio = got[0] / expected[0]
        assert ratio == pytest.approx(np.exp(1.0j * np.pi / 4.0), abs=1e-12)


class TestAgainstQiskit:
    """Cross-checks against Qiskit, the authority ``skqd.md`` measured against.

    The dense reference above is the primary one -- it needs no optional dependency -- but Qiskit
    independently pins the gate conventions and the transpiled-circuit path, which is the documented
    workflow and the one that was broken.
    """

    @pytest.mark.parametrize("num_qubits", [3, 5])
    def test_transpiled_ghz(self, num_qubits):
        """The shipped example's circuit, which previously reported a GHZ it had not produced.

        ``examples/svsim.py``'s only check was ``num_nonzero == 2``, which passed -- but the two
        nonzero states were ``00000`` and ``00001``, not ``00000`` and ``11111``, and the overlap
        with Qiskit was 0.5. Assert the actual states and an exact overlap, not just the count.
        """
        pytest.importorskip("qiskit")
        from qiskit import QuantumCircuit, transpile
        from qiskit.quantum_info import Statevector

        circuit = QuantumCircuit(num_qubits)
        circuit.h(0)
        for qubit in range(num_qubits - 1):
            circuit.cx(qubit, qubit + 1)
        transpiled = transpile(circuit, basis_gates=["rx", "ry", "rz", "rzz"], optimization_level=0)
        got = np.asarray(svsim(transpiled))
        reference = np.asarray(Statevector(circuit))
        assert phaseless_distance(got, reference) < 1e-12
        nonzero = set(np.nonzero(np.abs(got) > 1e-9)[0].tolist())
        assert nonzero == {0, 2**num_qubits - 1}, (
            f"expected |0...0> and |1...1>, got indices {sorted(nonzero)}"
        )

    def test_trotter_step(self):
        """A Trotterized evolution step: ``skqd.md``'s headline measurement.

        Before the fix a 6-qubit 4-rep step overlapped Qiskit's ``Statevector`` at 1e-16; restoring
        the phase gave 0.9999999999999991. This is the SKQD workload, so it is the test that decides
        whether the module is usable for it.
        """
        pytest.importorskip("qiskit")
        from qiskit import QuantumCircuit, transpile
        from qiskit.quantum_info import Statevector

        num_qubits, reps = 6, 4
        rng = np.random.default_rng(20260804)
        couplings = rng.normal(size=num_qubits - 1)
        fields = rng.normal(size=num_qubits)

        circuit = QuantumCircuit(num_qubits)
        for _ in range(reps):
            for qubit in range(num_qubits - 1):
                circuit.rzz(0.1 * couplings[qubit], qubit, qubit + 1)
            for qubit in range(num_qubits):
                circuit.rx(0.1 * fields[qubit], qubit)
        transpiled = transpile(circuit, basis_gates=["rx", "ry", "rz", "rzz"], optimization_level=0)

        got = np.asarray(svsim(transpiled))
        reference = np.asarray(Statevector(circuit))
        assert phaseless_distance(got, reference) < 1e-12
        assert np.allclose(got, reference, atol=1e-12), "expected exact agreement, not just overlap"

    @pytest.mark.parametrize("name", ["x", "y", "z", "rx", "ry", "rz"])
    def test_single_gates_match_qiskit_exactly(self, name):
        """Gate-by-gate, including phase: the conventions must agree, not merely be similar."""
        pytest.importorskip("qiskit")
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import Statevector

        angle = 0.7
        circuit = QuantumCircuit(1)
        if name in ("x", "y", "z"):
            getattr(circuit, name)(0)
            spec = (name, [0])
        else:
            getattr(circuit, name)(angle, 0)
            spec = (name, [0], angle)
        got = np.asarray(svsim([spec]))
        assert np.allclose(got, np.asarray(Statevector(circuit)), atol=1e-12)


def test_gate_unitary_reference_matches_dense_simulation():
    """The reference checks itself: applying unitaries must equal ``simulate_dense``.

    Guards the test suite's own foundation -- if the reference drifted, every assertion above would
    still pass while testing nothing.
    """
    specs = [("ry", [0], 0.4), ("rzz", [0, 1], 0.9), ("x", [1])]
    by_hand = np.zeros(4, dtype=np.complex128)
    by_hand[0] = 1.0
    for spec in specs:
        name, qubits, *rest = spec
        by_hand = gate_unitary(name, qubits, 2, *rest) @ by_hand
    assert np.allclose(simulate_dense(specs, 2), by_hand, atol=1e-15)


class TestShardedOutput:
    """``svsim``'s ``out_sharding`` must not change the answer, at any spec or gate.

    This path was exercised by **nothing** -- CLAUDE.md recorded it as the one remaining
    ``out_sharding`` contract with no test. It is checked here because the same axis in
    :mod:`rqutils.sqd` hid three defects, two of which masked the next: a rank-2 ``PartitionSpec`` on
    a rank-1 accumulator, a ``jnp.where`` mixing a replicated predicate with a partitioned operand,
    and a complex diagonal reaching ``argmin``. Measured, ``svsim`` has none of them -- every case
    below agrees with the single-device answer to exactly 0.0, and the returned specs confirm the
    output is genuinely distributed (4 shards of 16 amplitudes at n=6) rather than replicated and
    coincidentally right.

    ``svsim`` is structurally safer than ``sqd`` here: it takes ``out_sharding`` as an explicit
    parameter and threads it through every op that creates an array, rather than resharding
    conditionally partway through. The ``array_initial_state`` case is the one that mirrors ``sqd``'s
    ``_spread_seed`` defect -- a replicated caller-supplied vector meeting a partitioned scan body --
    and it passes.

    **Not covered, deliberately:** a mesh size that does not divide ``2**num_qubits``. That raises
    from jax, and it is a documented hard constraint rather than a defect -- a state vector cannot be
    padded the way ``sqd``'s state list can, because its indices *are* the basis states. Note the
    constraint bites harder than "small circuits": a 3- or 6-device mesh never divides a power of two,
    so it fails at *every* qubit count. See ``svsim``'s docstring.

    Runs as a subprocess because the virtual device count has to be set before jax initializes, and
    ``conftest`` has already imported it by collection time.
    """

    # frozenset, not set: a mutable class attribute is shared across every test instance, so an
    # accidental mutation in one test would silently change what the others assert.
    EXPECTED_CASES = frozenset(
        {
            "implicit",
            "explicit_partitioned",
            "explicit_replicated",
            "array_initial_state",
            "sharded_initial_state",
        }
    )

    def test_sharding_does_not_change_the_state_vector(self):
        stdout = run_sharded_child("_sharded_svsim.py", "svsim")

        seen = {}
        for line in stdout.strip().splitlines():
            parts = line.split(maxsplit=2)
            if len(parts) != 3:
                continue
            seen[parts[0]] = (float(parts[1]), parts[2])

        # Completeness before values: a child dying after two cases would otherwise pass on those two.
        assert set(seen) == self.EXPECTED_CASES, (
            f"expected {sorted(self.EXPECTED_CASES)}, got {sorted(seen)} -- the child did not run "
            f"every case:\n{stdout[-2000:]}"
        )
        for label, (diff, spec) in sorted(seen.items()):
            assert diff == pytest.approx(0.0, abs=1e-13), (
                f"{label}: sharded state vector differs from single-device by {diff}"
            )
        # The partitioned arms must actually be partitioned. This is not redundant with the value
        # checks above: an explicitly *replicated* run agrees with the single-device reference to
        # exactly 0.0 (measured), so "correct but silently unsharded" is invisible to any value
        # comparison. Only the spec distinguishes them.
        #
        # Mutation-tested. Neutering the mesh lookup or discarding a caller-supplied `out_sharding`
        # is caught, though via the child's exit code rather than here -- the `sharded_initial_state`
        # case then feeds a partitioned input into a replicated body and jax raises. These assertions
        # are what would catch a subtler regression that kept every op self-consistent while dropping
        # the partitioning.
        assert seen["explicit_replicated"][1] == "P(None,)", seen["explicit_replicated"]
        for label in self.EXPECTED_CASES - {"explicit_replicated"}:
            assert seen[label][1] == "P('x',)", f"{label} was not partitioned: {seen[label][1]}"
