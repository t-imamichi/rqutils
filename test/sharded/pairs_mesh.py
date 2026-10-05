"""Term-parallel ``sqd(matvec=Matvec.PAIRS)`` under a mesh against single-device; see ``TestShardedPairs``."""

import jax
import jax.numpy as jnp
import numpy as np
from common import emit, mesh
from jax.sharding import NamedSharding, PartitionSpec

from rqutils.paulis.symplectic import PauliSumXZ
from rqutils.sqd import Matvec, hproj, sqd, uniquify_states
from rqutils.sqd._sparse import _apply_pairs, _group_pairs, _mesh_operator, _sparse_operator
from rqutils.sqd._states import _pad_states

NUM_QUBITS, NUM_TERMS, NUM_DRAWS, STATES_SIZE = 10, 40, 300, 512
COLLECTIVES = ("all-gather", "reduce-scatter", "all-reduce", "all-to-all", "collective-permute")


def fixture(alphabet, seed):
    rng = np.random.default_rng(seed)
    strings = ["".join(rng.choice(list(alphabet), size=NUM_QUBITS)) for _ in range(NUM_TERMS)]
    coeffs = rng.normal(size=NUM_TERMS).tolist()
    states = np.unique(rng.integers(0, 2, size=(NUM_DRAWS, NUM_QUBITS)).astype(np.uint8), axis=0)
    return (strings, coeffs), states


def entries(operator):
    """The stored ``(i, j, d)`` with a nonzero factor, sorted: padding has a zero factor."""
    i, j, d = (np.asarray(a).ravel() for a in operator[1:])
    keep = d != 0
    order = np.lexsort((j[keep], i[keep]))
    return i[keep][order], j[keep][order], d[keep][order]


def case(alphabet, seed):
    ham, states = fixture(alphabet, seed)
    hamiltonian = PauliSumXZ.from_paulisum(ham)
    dense = float(np.linalg.eigvalsh(hproj(ham, states).toarray())[0])
    single = float(sqd(ham, states, return_eigvec=False, matvec=Matvec.PAIRS))
    states_u = uniquify_states(
        _pad_states(PauliSumXZ.pack_states(states), STATES_SIZE), STATES_SIZE
    )
    pairs = _group_pairs(hamiltonian, states_u)
    flat = _sparse_operator(hamiltonian, states_u, pairs)
    vec = np.random.default_rng(seed + 1).normal(size=(2, STATES_SIZE)) + 0j
    reference = np.asarray(_apply_pairs(jnp.asarray(vec), *flat))
    got = {"groups": len(pairs), "dense": dense, "single": single, "devices": {}}
    for num_devices in (2, 4):
        m = mesh(num_devices)
        with jax.set_mesh(m):
            sharded = float(sqd(ham, states, return_eigvec=False, matvec=Matvec.PAIRS))
            operator = _mesh_operator(hamiltonian, states_u, pairs, m)
            placed = jax.device_put(jnp.asarray(vec), NamedSharding(m, PartitionSpec(None, "x")))
            text = jax.jit(_apply_pairs).lower(placed, *operator).compile().as_text()
            product = _apply_pairs(placed, *operator)
        per_device = [
            int(np.count_nonzero(np.asarray(s.data))) for s in operator[3].addressable_shards
        ]
        got["devices"][str(num_devices)] = {
            "eigval": sharded,
            "specs": [str(jax.typeof(a).sharding.spec) for a in operator],
            "product_spec": str(jax.typeof(product).sharding.spec),
            "product_diff": float(np.max(np.abs(np.asarray(product) - reference))),
            "shards": len(operator[3].addressable_shards),
            "entries_per_device": per_device,
            "same_entries": all(
                np.array_equal(a, b) for a, b in zip(entries(operator), entries(flat), strict=True)
            ),
            "collectives": {k: text.count(k + "(") for k in COLLECTIVES},
        }
    return got


def main() -> None:
    # I/X/Z keeps `.c` real; adding Y makes it complex, the path spinchain's end-site fields take.
    emit({"real": case("IXZ", 5), "complex": case("IXYZ", 3)})


if __name__ == "__main__":
    main()
