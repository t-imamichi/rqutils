"""
===============================
Scalable state vector simulator
===============================

**svsim** is a GPU-accelerated state vector simulator implemented in
`JAX <https://docs.jax.dev/en/latest/index.html>`__. The implementation focuses on gate execution
speed and scalability. In practice this means:

* Only a very limited gate set is supported. When given as a qiskit QuantumCircuit, the input
  circuit can only contain ``x``, ``y``, ``z``, ``cz``, ``rx``, ``ry``, ``rz``, and ``rzz`` gates.
* Using the `Multi-device <https://docs.jax.dev/en/latest/parallel.html>`__ and
  `Multi-controller <https://docs.jax.dev/en/latest/multi_process.html>`__ features of JAX,
  circuits for large (32+) numbers of qubits can be simulated.

Usage examples can be found at
`examples/svsim.py <https://github.com/UTokyo-ICEPP/rqutils/tree/main/examples/svsim.py>`__.

.. currentmodule:: rqutils.svsim

.. autofunction:: svsim
.. autoclass:: CircuitXZ
.. autofunction:: to_circuitxz
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec, get_abstract_mesh
from jax.tree_util import register_dataclass
from numpy.typing import NDArray

try:
    from qiskit.circuit import QuantumCircuit

    HAS_QISKIT = True
except ImportError:
    HAS_QISKIT = False


@register_dataclass
@dataclass
class CircuitXZ:
    """Symplectic (XZ) representation of a series of rotation gates."""

    x: np.ndarray[tuple[int], np.dtype[np.int64]]
    z: np.ndarray[tuple[int], np.dtype[np.int64]]
    cos: np.ndarray[tuple[int], np.dtype[np.floating]]
    # Complex, not real: this carries i * (-i)^popcount(x & z), i.e. the sine amplitude with both
    # the leading i of the rotation and the symplectic phase of the convention
    # Q = (-i)^{x.z} Z^z X^x already folded in. Every supported gate has popcount(x & z) in {0, 1},
    # so the folded factor is exactly +i or +1 and no rounding is introduced. Keeping it real would
    # make the phase unrepresentable, which is how it came to be omitted entirely -- see
    # ``to_circuitxz`` and ``docs/skqd.md``.
    sin: np.ndarray[tuple[int], np.dtype[np.complexfloating]]
    num_qubits: int = field(metadata={"static": True})


type GateSpec = tuple[str, int | Sequence[int]] | tuple[str, int | Sequence[int], Any]
# Name the QuantumCircuit arm here rather than appending it under HAS_QISKIT -- see the
# optional-dependency rule in CLAUDE.md. No TYPE_CHECKING guard: `to_circuitxz`'s isinstance needs the
# runtime import anyway.
type CircuitInput = CircuitXZ | list[GateSpec] | QuantumCircuit


# name -> (x bit, z bit, takes an angle). The supported gate set in one place: which gates exist,
# their symplectic signature, and which are parameterized. Keeping the parameterized set as a
# separate tuple alongside a per-gate dispatch meant the same fact was spelled twice, and a drift
# between them would leave `angle` silently stale from the previous loop iteration rather than
# raising. Note `cz` is absent deliberately -- it is decomposed on the QuantumCircuit path only, and
# rejected as a raw gate spec (tests/test_svsim.py::TestCz::test_cz_as_a_gate_spec_is_rejected).
_GATE_XZ = {
    "x": (1, 0, False),
    "y": (1, 1, False),
    "z": (0, 1, False),
    "rx": (1, 0, True),
    "ry": (1, 1, True),
    "rz": (0, 1, True),
    "rzz": (0, 1, True),
}


def svsim(
    circuit: CircuitInput,
    initial_state: NDArray[np.complex128] | int = 0,
    out_sharding: NamedSharding | PartitionSpec | None = None,
) -> jax.Array:
    """Simulate the quantum circuit.

    The ``circuit`` argument can be given in three formats:

    * Qiskit ``QuantumCircuit``
    * ``CircuitXZ``
    * A list of gate specifiers

    A gate specifier is a 2-tuple ``(name, qubit)`` (for nonparametric gates) or a 3-tuple
    ``(name, qubit, angle)`` (rotation gates). The gate name must be one of ``x``, ``y``, ``z``,
    ``rx``, ``ry``, ``rz``, or ``rzz``. Note ``cz`` is *not* valid here: it is decomposed only on the
    ``QuantumCircuit`` path and is rejected as a raw gate spec.

    Args:
        circuit: Quantum circuit to simulate.
        initial_state: Initial state vector or the one-hot index.
        out_sharding: Manual specification of the sharding of the final state vector. Defaults to
            ``PartitionSpec(mesh.axis_names)`` when a mesh is active, i.e. the state vector is
            partitioned over every mesh axis. **The mesh size must divide** ``2**num_qubits``, which
            is a hard requirement rather than a tuning choice: a state vector cannot be padded the way
            :func:`rqutils.sqd.sqd`'s state list can, because its indices *are* the basis states and a
            filler amplitude would change the operator. Since the dimension is a power of two, this
            holds automatically for a power-of-two mesh with ``2**num_qubits >= mesh.size``, and never
            holds for a mesh of, say, 3 or 6 devices -- there it fails at *every* qubit count, not just
            small ones. The raise comes from jax and names both shapes. Pass
            ``PartitionSpec(None)`` to replicate instead, at the cost of a full copy per device.

    Returns:
        Final state vector as a (sharded) JAX Array.
    """
    if not isinstance(circuit, CircuitXZ):
        circuit = to_circuitxz(circuit)

    return do_svsim(circuit, initial_state, out_sharding)


@jax.jit(static_argnames=["out_sharding"])
def do_svsim(
    circuit: CircuitXZ,
    initial_state: NDArray[np.complex128] | int = 0,
    out_sharding: NamedSharding | PartitionSpec | None = None,
) -> jax.Array:
    """JIT-Compiled core of the simulator."""
    if out_sharding is None and not (mesh := get_abstract_mesh()).empty:
        out_sharding = PartitionSpec(mesh.axis_names)

    dim = 2**circuit.num_qubits

    # A separate name rather than rebinding `initial_state`: the parameter is declared as the int index
    # *or* the vector, and reassigning the vector onto it makes the declared type wrong from here down.
    state_vector = initial_state
    if len(initial_state.shape) == 0:
        one_hot_indices = jnp.arange(dim, dtype=np.int64, out_sharding=out_sharding)
        state_vector = (one_hot_indices == initial_state).astype(np.complex128)

    def apply_gate(state, gate):
        # Build the index iota inside the body, not once outside it. Closing over it instead makes
        # XLA hoist it into the scan's carry, where a 2^n int64 array stays resident for the whole
        # simulation on top of the two complex128 statevector buffers the loop already needs --
        # measured peak temp 2.5x the statevector, invariant in n, against 2.0x when built here (an
        # iota is loop-invariant, so XLA rematerializes it for free rather than keeping it live).
        # That is 8 GiB at n=30 and 32 GiB at n=32, on a module whose docstring advertises 32+
        # qubits. Output is bit-identical either way.
        indices = jnp.arange(dim, dtype=np.int64, out_sharding=out_sharding)
        signs = 1.0 - 2.0 * (jnp.bitwise_count(indices & gate.z) & 1)
        xstate = jax.lax.cond(
            jnp.all(gate.x == 0),
            lambda: state,
            lambda: state.at[indices ^ gate.x].get(out_sharding=out_sharding),
        )
        # No leading 1.0j here: gate.sin already carries i * (-i)^popcount(x & z) from
        # to_circuitxz. Multiplying by 1.0j again would double-count the rotation's i and drop the
        # symplectic phase, which is exactly the bug documented in docs/skqd.md.
        out = gate.sin * signs * xstate
        out = jax.lax.cond(gate.cos == 0.0, lambda: out, lambda: out + gate.cos * state)
        return out, None

    return jax.lax.scan(apply_gate, state_vector, circuit)[0]


def to_circuitxz(circuit: CircuitInput) -> CircuitXZ:
    """Translate circuit data given as a list of GateSpecs or a QuantumCircuit into signatures.

    A ``CircuitXZ`` is returned unchanged, so this is idempotent and cheap to call defensively.

    Args:
        circuit: A qiskit ``QuantumCircuit``, an existing :class:`CircuitXZ`, or a list of gate
            specifiers. See :func:`svsim` for the gate-specifier format and the supported names;
            note ``cz`` is valid only on the ``QuantumCircuit`` path, where it is decomposed.

    Returns:
        The circuit as ``(x, z, cos, sin)`` signature arrays.

    Raises:
        ValueError: If a gate name is not one of the supported set, or if a rotation gate is given
            as a 2-tuple with no angle.
    """
    if isinstance(circuit, CircuitXZ):
        return circuit

    num_qubits = None

    if HAS_QISKIT and isinstance(circuit, QuantumCircuit):

        def qidx(qubits):
            if isinstance(qubits, tuple):
                return np.array(list(map(circuit.qregs[0].index, qubits)))
            else:
                return np.array([circuit.qregs[0].index(qubits)])

        gate_specs = []
        for datum in circuit.data:
            if (op := datum.operation).name == "cz":
                gate_specs.extend(
                    [
                        ("rzz", qidx(datum.qubits), np.pi / 2.0),
                        ("rz", qidx(datum.qubits[0]), -np.pi / 2.0),
                        ("rz", qidx(datum.qubits[1]), -np.pi / 2.0),
                    ]
                )
            elif op.params:
                gate_specs.append((op.name, qidx(datum.qubits), op.params[0]))
            else:
                gate_specs.append((op.name, qidx(datum.qubits)))

        num_qubits = circuit.num_qubits
        circuit = gate_specs

    xarr = np.zeros(len(circuit), dtype=np.int64)
    zarr = np.zeros(len(circuit), dtype=np.int64)
    cosarr = np.zeros(len(circuit), dtype=np.float64)
    sinarr = np.zeros(len(circuit), dtype=np.complex128)
    qmax = 0
    for igate, gate in enumerate(circuit):
        # Unpack rather than index: GateSpec is a union of a 2-tuple and a 3-tuple, and a
        # parameterized gate must carry its angle. Indexing gate[2] directly would raise a bare
        # IndexError further down, with nothing to say which gate was malformed.
        name, qubit_spec, *rest = gate
        try:
            xbit, zbit, needs_angle = _GATE_XZ[name]
        except KeyError:
            raise ValueError(f"Unsupported gate name {name}") from None
        if needs_angle:
            if not rest:
                raise ValueError(f"Gate {name} requires an angle as its third element")
            spec = (xbit, zbit, rest[0])
        else:
            spec = (xbit, zbit, "pi")

        qubits = np.asarray(qubit_spec)
        xarr[igate] = np.sum(np.array(spec[0], dtype=np.int64) << qubits)
        zarr[igate] = np.sum(np.array(spec[1], dtype=np.int64) << qubits)
        # The module's convention is Q = (-i)^{x.z} Z^z X^x, so the symplectic phase belongs to
        # every gate whose X and Z signatures overlap -- from the match above, that is y and ry
        # (x, z, rx, rz, rzz, cz all have x.z == 0). Fold it into the sine amplitude together with
        # the rotation's own leading i, which is why sinarr is complex. Omitting it left ry off by
        # exactly a factor of i, and since transpiling to ['rx','ry','rz','rzz'] emits ry
        # constantly, that corrupted essentially every nontrivial circuit: a 5-qubit GHZ came back
        # with |overlap| 0.5 against qiskit, and a 6-qubit 4-rep Trotter step with 1e-16.
        phase = 1.0j * (-1.0j) ** int(np.bitwise_count(xarr[igate] & zarr[igate]))
        if spec[2] == "pi":
            # x/y/z are the bare Pauli gates, so they must NOT carry the -i that a pi rotation
            # would: R_P(pi) = exp(-i pi P / 2) = -i P. Multiplying the pi-rotation amplitude
            # (-phase) by the compensating +i gives exactly the Pauli. Without this, x and z came
            # back with a -i global phase relative to qiskit's X and Z -- harmless for a single
            # circuit, but a global phase stops being global as soon as a caller superposes two
            # simulations, and these are documented as gate names rather than as rotations.
            sinarr[igate] = -phase * 1.0j
        else:
            cosarr[igate] = np.cos(-spec[2] * 0.5)
            sinarr[igate] = phase * np.sin(-spec[2] * 0.5)

        qmax = max(qmax, np.max(qubits) + 1)

    if num_qubits is None:
        num_qubits = qmax

    return CircuitXZ(xarr, zarr, cosarr, sinarr, num_qubits)
