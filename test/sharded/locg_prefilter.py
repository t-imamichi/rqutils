"""``ground_locg``'s Chebyshev prefilter on meshes; see ``TestChebyshevPrefilter.test_preserves_sharding_on_a_mesh``."""

import jax
import jax.numpy as jnp
import numpy as np
from common import emit, mesh
from jax.sharding import NamedSharding, PartitionSpec

from rqutils.ground_locg import ground_locg


def main():
    dim = 512
    rng = np.random.default_rng(20260828)
    dense = rng.normal(size=(dim, dim))
    dense = (dense + dense.T) / 2
    start = rng.normal(size=dim)
    rows = {}
    # Both specs, not only partitioned: a replicated input meeting a partitioned body is the shape of
    # sqd's `_spread_seed` defect. Ragged splits are unreachable -- explicit sharding rejects them.
    for num_devices in (1, 2, 4):
        the_mesh = mesh(num_devices)
        with jax.set_mesh(the_mesh):
            operator = jax.device_put(
                jnp.asarray(dense), NamedSharding(the_mesh, PartitionSpec(None, None))
            )
            for label, spec in (("part", PartitionSpec("x")), ("repl", PartitionSpec(None))):
                xinit = jax.device_put(jnp.asarray(start), NamedSharding(the_mesh, spec))

                def matvec(vec, mat=operator, out_spec=spec):
                    return jnp.einsum("ij,j->i", mat, vec, out_sharding=out_spec)

                for kind, prefilter in (("plain", None), ("prefiltered", (16, 4))):
                    # A callable needs an explicit bound; the dense operator gives a Gershgorin one.
                    eigval, eigvec, iters, converged = ground_locg(
                        matvec,
                        xinit,
                        prefilter=prefilter,
                        prefilter_hi=float(np.abs(dense).sum(axis=-1).max()),
                    )
                    rows[f"{num_devices}:{label}:{kind}"] = [
                        float(eigval),
                        int(iters),
                        bool(converged),
                        str(jax.typeof(eigvec).sharding.spec),
                    ]
    emit({"reference": float(np.linalg.eigvalsh(dense)[0]), "rows": rows})


if __name__ == "__main__":
    main()
