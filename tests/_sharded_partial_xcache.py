"""Run ``sqd`` with a partial source-index cache on a 4-device mesh, printing every energy.

Driven as a subprocess by ``test_sqd.py::TestShardedPartialXCache``, which owns the rationale. It lives
in a file rather than an inline string so ruff and ty check it, matching ``_sharded_cache_levels.py``.

Not a pytest module (the leading underscore keeps it uncollected): the virtual device count must be set
before jax initializes.

**Why this cannot be an in-process test.** ``run_sqd`` reshards ``states_u`` after the precompute, on
the grounds that no further searches happen -- true for a full cache, false for a partial one, whose
uncached groups search ``states`` inside every matvec. ``get_xsource`` requires that array
**replicated**: a partitioned ``[N, B]`` raises "Unmapped values passed to vmap cannot be sharded along
the mesh axis you are vmapping over". So the guard on that reshard is **invisible single-device** --
removing it leaves all six ``TestPartialXCache`` cases green and crashes on any mesh. Measured, not
assumed.
"""

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

from jax.sharding import AxisType

from rqutils.paulis.symplectic import PauliSumXZ
from rqutils.sqd import sqd


def main() -> None:
    rng = np.random.default_rng(3)
    num_qubits = 8
    labels = ["".join(rng.choice(list("IXYZ"), size=num_qubits)) for _ in range(6)]
    hamiltonian = PauliSumXZ.from_paulisum((labels, [1.0] * len(labels)))
    num_groups = hamiltonian.x.shape[0]
    states = np.unique(rng.integers(0, 2, size=(32, num_qubits), dtype=np.uint8), axis=0)

    single = sqd(hamiltonian, states, return_eigvec=False)
    print(f"single {single!r}")

    mesh = jax.make_mesh((4,), ("x",), axis_types=(AxisType.Explicit,))
    with jax.sharding.set_mesh(mesh):
        for cache_level in [(1, 0), (1, 1), (1, 2)]:
            for ncached in range(num_groups + 1):
                value = sqd(
                    hamiltonian,
                    states,
                    cache_level=cache_level,
                    xcache_groups=ncached,
                    return_eigvec=False,
                )
                print(f"mesh {cache_level[0]} {cache_level[1]} {ncached} {value!r}")


if __name__ == "__main__":
    main()
