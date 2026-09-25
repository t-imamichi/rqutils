"""``hproj`` must reject a live mesh and work once it exits; see ``TestShardedHproj``."""

import jax
import numpy as np
from common import emit, mesh
from qiskit.quantum_info import SparsePauliOp

from rqutils.sqd import hproj

NUM_QUBITS = 6


def basis(count: int) -> np.ndarray:
    """The first `count` nonzero bitstrings, deterministic so the parent can check by hand."""
    return np.array(
        [[(i >> k) & 1 for k in reversed(range(NUM_QUBITS))] for i in range(1, count + 1)],
        dtype=np.uint8,
    )


def rejects(n: int) -> bool:
    try:
        hproj(HAM, basis(n))
    except ValueError as exc:
        return "does not support sharding" in str(exc)
    return False


HAM = SparsePauliOp(["XXIIII", "IIZZII", "IXIZII"], [0.7, -1.3, 0.4])


def main() -> None:
    # 23 is indivisible by 4 and 24 divisible: both must work without a mesh and fail under one.
    reference = {n: hproj(HAM, basis(n)).toarray() for n in (23, 24)}
    result = {"no_mesh_shapes": [reference[n].shape[0] for n in (23, 24)]}
    # Scoped mesh, then outside it: poc/sharding.py's pattern of building the dense oracle outside.
    with jax.set_mesh(mesh(4)):
        result["scoped_rejects"] = rejects(24)
    result["after_scope_diff"] = float(
        np.abs(hproj(HAM, basis(24)).toarray() - reference[24]).max()
    )
    # A global mesh rejects both sizes, so a divisible count is no loophole.
    jax.set_mesh(mesh(4))
    result["global_rejects"] = [rejects(n) for n in (23, 24)]
    emit(result)


if __name__ == "__main__":
    main()
