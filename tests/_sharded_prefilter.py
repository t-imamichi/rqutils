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

    mesh = jax.make_mesh((4,), ("x",), axis_types=(AxisType.Explicit,))
    with jax.sharding.set_mesh(mesh):
        operator = jax.device_put(
            jnp.asarray(dense), NamedSharding(mesh, PartitionSpec(None, None))
        )
        xinit = jax.device_put(
            jnp.asarray(rng.normal(size=dim)), NamedSharding(mesh, PartitionSpec("x"))
        )

        def matvec(vec):
            return jnp.einsum("ij,j->i", operator, vec, out_sharding=PartitionSpec("x"))

        for label, prefilter in (("plain", None), ("prefiltered", (16, 4))):
            result = ground_locg(matvec, xinit, prefilter=prefilter)
            spec = jax.typeof(result[1]).sharding.spec
            print(f"{label} {float(result[0])!r} {int(result[2])} {bool(result[3])!r} {spec}")


if __name__ == "__main__":
    main()
