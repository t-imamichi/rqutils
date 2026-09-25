"""``svsim`` single-device against a 4-device mesh across gates and specs; see ``TestShardedOutput``."""

import jax
import numpy as np
from common import emit, mesh
from jax.sharding import PartitionSpec

from rqutils.svsim import GateSpec, svsim

NUM_QUBITS = 6


def base_circuit() -> list[GateSpec]:
    """A circuit touching every supported gate, so one sweep covers the whole gate set."""
    # Annotated, or the first comprehension pins the element type and the heterogeneous `+=` fails ty.
    circuit: list[GateSpec] = [("ry", q, 0.3 * (q + 1)) for q in range(NUM_QUBITS)]
    circuit += [("rzz", (q, q + 1), 0.2) for q in range(NUM_QUBITS - 1)]
    circuit += [("x", 0), ("y", 1), ("z", 2), ("rx", 3, 0.4), ("rz", 4, 0.5)]
    return circuit


def main() -> None:
    circuit = base_circuit()
    reference = np.asarray(svsim(circuit, 0))
    the_mesh = mesh(4)
    one_hot = np.eye(2**NUM_QUBITS, dtype=np.complex128)[0]
    with jax.set_mesh(the_mesh):
        cases = {
            "implicit": lambda: svsim(circuit, 0),
            "explicit_partitioned": lambda: svsim(circuit, 0, PartitionSpec("x")),
            "explicit_replicated": lambda: svsim(circuit, 0, PartitionSpec(None)),
            # A caller-supplied vector must not fight the scan carry's sharding (sqd's _spread_seed).
            "array_initial_state": lambda: svsim(circuit, one_hot),
            "sharded_initial_state": lambda: svsim(
                circuit, jax.device_put(one_hot, jax.NamedSharding(the_mesh, PartitionSpec("x")))
            ),
        }
        result = {}
        for label, fn in cases.items():
            got = fn()
            result[label] = [
                float(np.abs(np.asarray(got) - reference).max()),
                str(got.sharding.spec),
            ]
    emit(result)


if __name__ == "__main__":
    main()
