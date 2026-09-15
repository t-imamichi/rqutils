"""Run ``hproj`` on a 4-device mesh, printing the projection's agreement with a single-device one.

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

    # Single-device reference first, before any mesh exists.
    single = {n: hproj(ham, basis(n)).toarray() for n in (MESH_SIZE * 5, MESH_SIZE * 6)}

    jax.set_mesh(jax.make_mesh((MESH_SIZE,), ("x",), (AxisType.Explicit,)))

    # Divisible counts must work and agree exactly: a boolean-mask gather on the sharded `columns`
    # raised ShardingTypeError, so every mesh-enabled hproj call failed at any size.
    for n, ref in single.items():
        got = hproj(ham, basis(n)).toarray()
        print(f"agrees_{n} {float(np.abs(got - ref).max()):.15e}")
        print(f"symmetric_{n} {float(np.abs(got - got.T).max()):.15e}")

    # An indivisible count must name uniquify_states rather than raising jax's own message.
    try:
        hproj(ham, basis(MESH_SIZE * 5 + 3))
        print("named False")
    except ValueError as exc:
        print(f"named {str(exc).startswith('hproj:')}")


if __name__ == "__main__":
    main()
