"""The popcount diagonal builders with states partitioned ``P('x', None)``; see ``TestShardedDiagonals``."""

import jax
import jax.numpy as jnp
import numpy as np
from common import emit, mesh
from jax.sharding import NamedSharding, PartitionSpec
from qiskit.quantum_info import SparsePauliOp

from rqutils.paulis.symplectic import PauliSumXZ
from rqutils.sqd import compute_diagonal, get_diag_signs, get_diagonal


def xxz(num_qubits, odd_y):
    """1D periodic XXZ; ``odd_y`` adds one odd-Y string so the folded coefficients go complex."""
    labels, coeffs = [], []
    for q in range(num_qubits):
        for letter in "XYZ":
            term = ["I"] * num_qubits
            term[q] = term[(q + 1) % num_qubits] = letter
            labels.append("".join(term))
            coeffs.append(1.0)
    if odd_y:
        labels.append("Y" + "I" * (num_qubits - 1))
        coeffs.append(0.5)
    return PauliSumXZ.from_paulisum(SparsePauliOp(labels, coeffs).simplify())


def subspace(num_qubits, num_rows, rng):
    """Fixed-Hamming-weight rows, packed with the mandatory pad bit at position 0."""
    rows = np.zeros((num_rows, num_qubits), np.uint8)
    for i in range(num_rows):
        rows[i, rng.choice(num_qubits, num_qubits // 2, replace=False)] = 1
    padded = np.concatenate([np.zeros((num_rows, 1), np.uint8), rows], axis=1)
    return np.unique(np.packbits(padded, axis=1), axis=0)


def main():
    num_qubits = 20
    rng = np.random.default_rng(20260830)
    result = {}
    for odd_y in (False, True):
        ham = xxz(num_qubits, odd_y)
        label = "complex" if np.iscomplexobj(ham.c) else "real"
        for num_devices in (2, 4):
            the_mesh = mesh(num_devices)
            states = subspace(num_qubits, 512, rng)
            # Explicit sharding rejects a ragged split at device_put, so trim rather than sweep it.
            states = states[: (len(states) // num_devices) * num_devices]
            bad_spec = bad_value = 0
            with jax.set_mesh(the_mesh):
                partitioned = jax.device_put(
                    jnp.asarray(states), NamedSharding(the_mesh, PartitionSpec("x", None))
                )
                for zsig, coeff in zip(ham.z, ham.c):
                    ref_signs = np.asarray(get_diag_signs(zsig, jnp.asarray(states)))
                    refs = (
                        ref_signs,
                        np.asarray(get_diagonal(zsig, coeff, jnp.asarray(states))),
                        np.asarray(compute_diagonal(jnp.asarray(ref_signs), coeff)),
                    )
                    got_signs = get_diag_signs(zsig, partitioned)
                    gots = (
                        got_signs,
                        get_diagonal(zsig, coeff, partitioned),
                        compute_diagonal(got_signs, coeff),
                    )
                    for ref, got in zip(refs, gots):
                        bad_spec += "x" not in str(jax.typeof(got).sharding.spec)
                        bad_value += not np.array_equal(ref, np.asarray(got))
            result[f"{label} {num_devices}"] = [int(ham.x.shape[0]), bad_spec, bad_value]
    emit(result)


if __name__ == "__main__":
    main()
