"""``sqd`` with a partial source-index cache on a 4-device mesh; see ``TestShardedPartialXCache``."""

import jax
import numpy as np
from common import emit, mesh

from rqutils.paulis.symplectic import PauliSumXZ
from rqutils.sqd import sqd


def main() -> None:
    rng = np.random.default_rng(3)
    num_qubits = 8
    labels = ["".join(rng.choice(list("IXYZ"), size=num_qubits)) for _ in range(6)]
    hamiltonian = PauliSumXZ.from_paulisum((labels, [1.0] * len(labels)))
    states = np.unique(rng.integers(0, 2, size=(32, num_qubits), dtype=np.uint8), axis=0)

    sharded = {}
    with jax.set_mesh(mesh(4)):
        for level in [(1, 0), (1, 1), (1, 2)]:
            for ncached in range(hamiltonian.x.shape[0] + 1):
                sharded[f"{level} {ncached}"] = float(
                    sqd(
                        hamiltonian,
                        states,
                        cache_level=level,
                        xcache_groups=ncached,
                        return_eigvec=False,
                    )
                )
    emit({"single": float(sqd(hamiltonian, states, return_eigvec=False)), "sharded": sharded})


if __name__ == "__main__":
    main()
