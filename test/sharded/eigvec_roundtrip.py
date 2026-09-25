"""``sqd(return_eigvec=True)`` on a 4-device mesh against a dense projection of its own basis; see
``TestShardedEigvecRoundtrip``."""

import jax
import numpy as np
from common import emit, mesh
from conftest import real_pauli_strings, unique_states

from rqutils.sqd import hproj, sqd

NUM_QUBITS = 8
NUM_STATES = 30  # not a multiple of 4, so sqd's mesh-size padding runs


def main() -> None:
    rng = np.random.default_rng(20260904)
    strings = real_pauli_strings(NUM_QUBITS, 12, rng)
    coeffs = rng.uniform(-1.0, 1.0, size=len(strings))
    states = unique_states(NUM_STATES, NUM_QUBITS, rng)
    with jax.set_mesh(mesh(4)):
        eigval, eigvec, basis = sqd((strings, coeffs), states, return_eigvec=True)
    eigvec, basis = np.asarray(eigvec), np.asarray(basis)
    # Built from the *returned* basis, which is what couples the two independently resharded arrays.
    dense = hproj((strings, coeffs), basis).toarray()
    emit(
        {
            "eigvec_len": len(eigvec),
            "basis_rows": len(basis),
            "relative_residual": float(
                np.linalg.norm(dense @ eigvec - eigval * eigvec) / np.linalg.norm(eigvec)
            ),
            "eigval": float(eigval),
            "reference": float(np.linalg.eigvalsh(dense)[0]),
        }
    )


if __name__ == "__main__":
    main()
