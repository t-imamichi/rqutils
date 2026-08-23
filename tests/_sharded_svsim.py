"""Run ``svsim`` single-device and on a 4-device mesh across gates and sharding specs.

Driven as a subprocess by ``test_svsim.py::TestShardedOutput``, which owns the rationale. It lives in
a file rather than an inline string so ruff and ty check it, and it is not a pytest module (the
leading underscore keeps it uncollected): the virtual device count must be set before jax initializes,
which ``conftest.py`` has already done by collection time.

Prints one line per case: ``<label> <maxabsdiff> <spec>``. The caller asserts the case list is
complete before checking values, so a child that dies partway cannot pass on what it managed to print.

Covers what nothing did before: ``svsim``'s ``out_sharding`` was exercised by no test at all. The
axes here are the ones that found real defects in ``rqutils.sqd`` -- every supported gate, an
explicitly replicated spec, an explicit partitioned spec, and a caller-supplied initial state (whose
sharding must not fight the scan carry's).

Deliberately *not* covered, because it is a documented hard constraint rather than a bug: a mesh size
that does not divide ``2**num_qubits`` raises from jax. A state vector cannot be padded -- its indices
are the basis states -- so there is nothing to fix. See ``svsim``'s docstring.
"""

import jax

# Must precede the first array creation, exactly as conftest.py does for the suite.
jax.config.update("jax_enable_x64", True)

import numpy as np
from jax.sharding import AxisType, PartitionSpec

from rqutils.svsim import svsim

NUM_QUBITS, MESH_SIZE = 6, 4


def base_circuit() -> list:
    """A circuit touching every supported gate, so one sweep covers the whole gate set."""
    circuit = [("ry", q, 0.3 * (q + 1)) for q in range(NUM_QUBITS)]
    circuit += [("rzz", (q, q + 1), 0.2) for q in range(NUM_QUBITS - 1)]
    circuit += [("x", 0), ("y", 1), ("z", 2), ("rx", 3, 0.4), ("rz", 4, 0.5)]
    return circuit


def main() -> None:
    circuit = base_circuit()
    reference = np.asarray(svsim(circuit, 0))
    mesh = jax.make_mesh((MESH_SIZE,), ("x",), (AxisType.Explicit,))
    dim = 2**NUM_QUBITS
    one_hot = np.eye(dim, dtype=np.complex128)[0]

    with jax.set_mesh(mesh):
        cases = {
            # Implicit: out_sharding defaults to PartitionSpec(mesh.axis_names).
            "implicit": lambda: svsim(circuit, 0),
            "explicit_partitioned": lambda: svsim(circuit, 0, PartitionSpec("x")),
            "explicit_replicated": lambda: svsim(circuit, 0, PartitionSpec(None)),
            # A caller-supplied vector: its sharding must not fight the scan carry's. This is the
            # shape of the _spread_seed defect in rqutils.sqd -- a replicated input meeting a
            # partitioned body -- so it is checked here rather than assumed.
            "array_initial_state": lambda: svsim(circuit, one_hot),
            "sharded_initial_state": lambda: svsim(
                circuit,
                jax.device_put(one_hot, jax.NamedSharding(mesh, PartitionSpec("x"))),
            ),
        }
        for label, fn in cases.items():
            got = fn()
            diff = float(np.abs(np.asarray(got) - reference).max())
            print(f"{label} {diff!r} {got.sharding.spec}")


if __name__ == "__main__":
    main()
