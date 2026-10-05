"""Term-parallel ``sqd(matvec=Matvec.PAIRS)`` under a mesh against single-device; see ``TestShardedPairs``."""

import re

import jax
import jax.numpy as jnp
import numpy as np
from common import emit, mesh
from conftest import real_pauli_strings, unique_states
from jax.sharding import NamedSharding, PartitionSpec

from rqutils.paulis.symplectic import PauliSumXZ
from rqutils.sqd import Matvec, hproj, sqd, uniquify_states
from rqutils.sqd._sparse import _apply_pairs, _apply_pairs_mesh, _group_pairs, _sparse_operator
from rqutils.sqd._states import _pad_states

NUM_QUBITS, NUM_TERMS, NUM_DRAWS, STATES_SIZE = 10, 40, 300, 512
COLLECTIVES = ("all-gather", "reduce-scatter", "all-reduce", "all-to-all", "collective-permute")


def fixture(letters, seed):
    rng = np.random.default_rng(seed)
    if letters == "IXZ":
        strings = real_pauli_strings(NUM_QUBITS, NUM_TERMS, rng, letters=letters)
    else:  # odd Y counts allowed, so `.c` is complex
        strings = ["".join(rng.choice(list(letters), size=NUM_QUBITS)) for _ in range(NUM_TERMS)]
    coeffs = rng.normal(size=len(strings)).tolist()
    return (strings, coeffs), unique_states(NUM_DRAWS, NUM_QUBITS, rng)


def stored(operator):
    """Each device's stored ``(i, j, d)``, its padding (zero factor) dropped, in device order."""
    i, j, d = (np.asarray(a).reshape(np.asarray(a).shape[0], -1) for a in operator[1:])
    keep = d != 0
    return [np.concatenate([a[p][keep[p]] for p in range(len(d))]) for a in (i, j, d)]


def case(letters, seed):
    ham, states = fixture(letters, seed)
    hamiltonian = PauliSumXZ.from_paulisum(ham)
    dense = float(np.linalg.eigvalsh(hproj(ham, states).toarray())[0])
    single = float(sqd(ham, states, return_eigvec=False, matvec=Matvec.PAIRS))
    states_u = uniquify_states(
        _pad_states(PauliSumXZ.pack_states(states), STATES_SIZE), STATES_SIZE
    )
    pairs = _group_pairs(hamiltonian, states_u)
    flat = _sparse_operator(hamiltonian, states_u, pairs)
    flat_stored = [np.asarray(a).ravel()[np.asarray(flat[3]).ravel() != 0] for a in flat[1:]]
    vec = np.random.default_rng(seed + 1).normal(size=(2, STATES_SIZE)) + 0j
    reference = np.asarray(_apply_pairs(jnp.asarray(vec), *flat))
    got = {"dense": dense, "single": single, "devices": {}}
    for num_devices in (2, 4):
        m = mesh(num_devices)
        with jax.set_mesh(m):
            sharded = float(sqd(ham, states, return_eigvec=False, matvec=Matvec.PAIRS))
            operator = _sparse_operator(hamiltonian, states_u, pairs, m)
            placed = jax.device_put(jnp.asarray(vec), NamedSharding(m, PartitionSpec(None, "x")))
            compiled = jax.jit(_apply_pairs_mesh).lower(placed, *operator).compile()
            product = compiled(placed, *operator)
        text = compiled.as_text()
        got["devices"][str(num_devices)] = {
            "eigval": sharded,
            "specs": [str(jax.typeof(a).sharding.spec) for a in operator],
            "product_spec": str(jax.typeof(product).sharding.spec),
            "product_diff": float(np.max(np.abs(np.asarray(product) - reference))),
            "shards": len(operator[3].addressable_shards),
            "entries_per_device": [
                int(np.count_nonzero(np.asarray(s.data))) for s in operator[3].addressable_shards
            ],
            "same_entries": all(
                np.array_equal(a, b) for a, b in zip(stored(operator), flat_stored, strict=True)
            ),
            "collectives": {k: len(re.findall(rf" {k}(?:-start)?\(", text)) for k in COLLECTIVES},
        }
    return got


def main() -> None:
    emit({"real": case("IXZ", 5), "complex": case("IXYZ", 3)})


if __name__ == "__main__":
    main()
