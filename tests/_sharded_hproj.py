"""Check that ``hproj`` rejects a live mesh, and still works once the mesh context exits.

Driven by ``test_sqd.py::TestShardedHproj``. A file rather than an inline blob so ruff and ty check
it; not collected by pytest (leading underscore) because the virtual device count must be set before
jax initializes.
"""

import jax

# Must precede the first array creation, exactly as conftest.py does for the suite.
jax.config.update("jax_enable_x64", True)

import numpy as np
from jax.sharding import AxisType
from qiskit.quantum_info import SparsePauliOp

from rqutils.sqd import hproj

NUM_QUBITS, MESH_SIZE = 6, 4


def basis(count: int) -> np.ndarray:
    """The first `count` nonzero bitstrings, deterministic so the parent can check by hand."""
    return np.array(
        [[(i >> k) & 1 for k in reversed(range(NUM_QUBITS))] for i in range(1, count + 1)],
        dtype=np.uint8,
    )


def main() -> None:
    ham = SparsePauliOp(["XXIIII", "IIZZII", "IXIZII"], [0.7, -1.3, 0.4])
    # 23 is indivisible by 4, 24 divisible: neither is supported under a mesh, and both must work
    # without one -- the divisibility of the subspace is irrelevant to hproj either way.
    reference = {n: hproj(ham, basis(n)).toarray() for n in (23, 24)}
    print(f"no_mesh_shapes {[reference[n].shape[0] for n in (23, 24)]}")

    mesh = jax.make_mesh((MESH_SIZE,), ("x",), (AxisType.Explicit,))

    # Scoped mesh: `hproj` inside must raise, outside must work. This is poc7_sharding.py's pattern,
    # which builds its dense reference outside the `with` block.
    with jax.set_mesh(mesh):
        try:
            hproj(ham, basis(24))
            print("scoped_raised False")
        except ValueError as exc:
            print(f"scoped_raised {'does not support sharding' in str(exc)}")
    after = hproj(ham, basis(24)).toarray()
    print(f"after_scope_agrees {float(np.abs(after - reference[24]).max()):.15e}")

    # Globally set mesh: both sizes rejected, so a divisible count is no loophole.
    jax.set_mesh(mesh)
    for n in (23, 24):
        try:
            hproj(ham, basis(n))
            print(f"global_raised_{n} False")
        except ValueError as exc:
            print(f"global_raised_{n} {'does not support sharding' in str(exc)}")


if __name__ == "__main__":
    main()
