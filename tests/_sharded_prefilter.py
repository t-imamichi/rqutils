"""Sharding coverage for ``ground_locg``'s Chebyshev prefilter, under virtual devices.

Subprocessed by ``tests/test_ground_locg.py`` because the virtual device count comes from
``XLA_FLAGS=--xla_force_host_platform_device_count``, which XLA reads at backend initialization --
long before an in-process test could set it.

The prefilter only calls the caller's ``matvec`` and scales elementwise, so it *should* be
sharding-transparent for free. That is exactly the kind of "should" this repo requires measuring: a
replicated run agrees with single-device to exactly 0.0, so a silently unsharded prefilter is
invisible to value comparison. Hence the spec is printed and asserted, not just the energy.

Line protocol, one per case: ``<label> <energy> <iterations> <converged> <eigvec spec>``.
"""

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax.sharding import AxisType, NamedSharding, PartitionSpec

from rqutils.ground_locg import ground_locg


def main():
    dim = 512
    rng = np.random.default_rng(20260828)
    dense = rng.normal(size=(dim, dim))
    dense = (dense + dense.T) / 2
    reference = float(np.linalg.eigvalsh(dense)[0])
    print(f"reference {reference!r}")

    start = rng.normal(size=dim)
    # Sweep mesh size AND both specs. A single 4-device partitioned case would miss the
    # replicated-input pairing, which is the shape of the `_spread_seed` ShardingTypeError in
    # `rqutils.sqd`: a replicated predicate meeting a partitioned vector. Ragged splits are NOT
    # swept because they are unreachable -- explicit sharding rejects `dim % mesh.size != 0` at
    # `device_put`, before any of this code runs.
    for num_devices in (1, 2, 4):
        mesh = jax.make_mesh((num_devices,), ("x",), axis_types=(AxisType.Explicit,))
        with jax.sharding.set_mesh(mesh):
            operator = jax.device_put(
                jnp.asarray(dense), NamedSharding(mesh, PartitionSpec(None, None))
            )
            for spec_label, spec in (("part", PartitionSpec("x")), ("repl", PartitionSpec(None))):
                xinit = jax.device_put(jnp.asarray(start), NamedSharding(mesh, spec))

                # Both loop variables bound as defaults, per CLAUDE.md's B023 note: the fix is to
                # bind, not to restructure the loop.
                def matvec(vec, mat=operator, out_spec=spec):
                    return jnp.einsum("ij,j->i", mat, vec, out_sharding=out_spec)

                for kind, prefilter in (("plain", None), ("prefiltered", (16, 4))):
                    # A callable now requires an explicit bound; the dense operator behind this
                    # matvec supplies a rigorous Gershgorin one.
                    result = ground_locg(
                        matvec,
                        xinit,
                        prefilter=prefilter,
                        prefilter_hi=float(np.abs(dense).sum(axis=-1).max()),
                    )
                    out_spec = jax.typeof(result[1]).sharding.spec
                    print(
                        f"{num_devices}:{spec_label}:{kind} {float(result[0])!r} "
                        f"{int(result[2])} {bool(result[3])!r} {out_spec}"
                    )


if __name__ == "__main__":
    main()
