"""``run_sqd`` with ``batch_matvec`` on and off, sharded and single-device; see ``TestShardedBatchMatvec``."""

import jax
import numpy as np
from common import emit, mesh
from conftest import real_pauli_strings, unique_states

from rqutils.paulis.symplectic import PauliSumXZ
from rqutils.sqd import run_sqd

# 37 draws collapse to ~34 unique rows; STATES_SIZE is a constant, not sqd's formula, so the padding
# below asserts the bound rather than trusting it.
NUM_QUBITS, NUM_STATES, NUM_TERMS, STATES_SIZE = 8, 37, 6, 64


def main() -> None:
    rng = np.random.default_rng(23)
    labels = real_pauli_strings(NUM_QUBITS, NUM_TERMS, rng)
    hamiltonian = PauliSumXZ.from_paulisum((labels, rng.normal(size=NUM_TERMS).tolist()))
    # Padded here because this calls run_sqd directly; 255 is the filler uniquify_states uses.
    states_p = PauliSumXZ.pack_states(unique_states(NUM_STATES, NUM_QUBITS, rng))
    deficit = STATES_SIZE - states_p.shape[0]
    assert deficit >= 0, f"{states_p.shape[0]} unique rows exceed STATES_SIZE={STATES_SIZE}"
    states_p = np.append(states_p, np.full((deficit, states_p.shape[1]), 255, np.uint8), axis=0)

    the_mesh = mesh(4)
    args = (hamiltonian, states_p, STATES_SIZE, False)
    energies = {}
    for batch in (False, True):
        single = float(run_sqd(*args, batch_matvec=batch)[0])
        with jax.set_mesh(the_mesh):
            energies[str(batch)] = [single, float(run_sqd(*args, batch_matvec=batch)[0])]
    with jax.set_mesh(the_mesh):
        vec = jax.device_put(
            jax.numpy.arange(float(STATES_SIZE)), jax.NamedSharding(the_mesh, jax.P("x"))
        )
        stacked = jax.numpy.stack((vec, vec))
        specs = [str(jax.typeof(vec).sharding.spec), str(jax.typeof(stacked).sharding.spec)]
    emit({"energies": energies, "specs": specs})


if __name__ == "__main__":
    main()
