"""Run ``sqd(prefilter=...)`` single-device and on meshes, at every ``cache_level``.

Driven as a subprocess by ``test_sqd.py::TestShardedSqdPrefilter``, which owns the rationale -- read it
there rather than restating it here. In a file rather than an inline string so ruff and ty check it,
and underscore-prefixed so pytest does not collect it: the virtual device count comes from
``XLA_FLAGS``, which XLA reads at backend initialization.

Two line protocols, since value agreement alone cannot detect an unsharded filter:

* ``energy <devices> <i> <j> <single> <sharded>``
* ``spec <devices> <label> <vinit_spec> <filtered_spec>``
"""

import itertools

import jax

# Must precede the first array creation, exactly as conftest.py does for the suite.
jax.config.update("jax_enable_x64", True)

import functools

import jax.numpy as jnp
import numpy as np
from jax.sharding import AxisType, PartitionSpec

import rqutils.sqd as sqd_module
from rqutils.ground_locg import _chebyshev_prefilter
from rqutils.paulis.symplectic import PauliSumXZ
from rqutils.sqd import sqd

# 37 genuine states pad to 64, which every MESH_SIZES entry divides -- the arrangement
# `uniquify_states` exists to produce, and the one `_sharded_prefilter.py` cannot reach.
NUM_QUBITS, NUM_STATES, NUM_TERMS, STATES_SIZE = 8, 37, 5, 64
PREFILTER = (16, 2)
MESH_SIZES = (1, 2, 4)
CACHE_LEVELS = sorted(itertools.product((0, 1), (0, 1, 2)))


def main() -> None:
    rng = np.random.default_rng(11)
    # I/X/Z only: an even Y count keeps the folded phase real, so `.c` stays float64 and this
    # exercises the real-symmetric path a Hamiltonian actually takes. The complex path is covered
    # single-device by `TestComplexCoefficientsAcrossCacheLevels`.
    strings = ["".join(rng.choice(list("IXZ"), size=NUM_QUBITS)) for _ in range(NUM_TERMS)]
    coeffs = rng.normal(size=NUM_TERMS).tolist()
    states = rng.integers(0, 2, size=(NUM_STATES, NUM_QUBITS)).astype(np.uint8)

    for num_devices in MESH_SIZES:
        mesh = jax.make_mesh((num_devices,), ("x",), (AxisType.Explicit,))
        for cache_level in CACHE_LEVELS:
            single = float(
                sqd(
                    (strings, coeffs),
                    states,
                    return_eigvec=False,
                    cache_level=cache_level,
                    prefilter=PREFILTER,
                )
            )
            with jax.set_mesh(mesh):
                sharded = float(
                    sqd(
                        (strings, coeffs),
                        states,
                        return_eigvec=False,
                        cache_level=cache_level,
                        prefilter=PREFILTER,
                    )
                )
            print(f"energy {num_devices} {cache_level[0]} {cache_level[1]} {single!r} {sharded!r}")

    _report_specs(strings, coeffs, states)


def _report_specs(strings: list[str], coeffs: list[float], states: np.ndarray) -> None:
    """Print the prefilter's output spec, assembled exactly as ``run_sqd`` assembles its matvec.

    Reaches into the module's helpers rather than calling ``sqd``: ``sqd`` reshards the eigenvector to
    ``PartitionSpec(None)`` before returning it, so the partitioning the filter has to preserve is not
    observable from the public result. Everything here mirrors ``run_sqd``'s ``cache_level == (1, 0)``
    path -- the replicated ``states_u`` for the ``get_xsource`` searches, the partitioned ``vinit``.
    """
    hamiltonian = PauliSumXZ.from_paulisum((strings, coeffs))
    # `pack_states` rather than the packbits idiom open-coded: its docstring says consumers must call
    # it, because a disagreement over the pad bit is silent. STATES_SIZE is a constant here rather
    # than `sqd`'s bit_length formula, which would be a second copy of it that could drift.
    states_p = PauliSumXZ.pack_states(states)
    padding = np.full((STATES_SIZE - len(states_p), states_p.shape[1]), 255, dtype=np.uint8)
    states_p = np.append(states_p, padding, axis=0)

    for num_devices in MESH_SIZES:
        mesh = jax.make_mesh((num_devices,), ("x",), (AxisType.Explicit,))
        with jax.set_mesh(mesh):
            states_u = jax.reshard(
                sqd_module.uniquify_states(states_p, STATES_SIZE), PartitionSpec(None, None)
            )
            xsources = jnp.stack([sqd_module.get_xsource(x, states_u) for x in hamiltonian.x])
            diagonals = jnp.stack(
                [
                    sqd_module.get_diagonal(z, c, states_u)
                    for z, c in zip(hamiltonian.z, hamiltonian.c)
                ]
            )
            matvec = functools.partial(sqd_module.apply_h, xsources=xsources, diagonals=diagonals)
            for label, spec in (("part", PartitionSpec("x")), ("repl", PartitionSpec(None))):
                vinit = sqd_module._spread_seed(STATES_SIZE, states_u, hamiltonian.c.dtype, spec)
                filtered = _chebyshev_prefilter(
                    matvec, (), vinit, PREFILTER[0], PREFILTER[1], jnp.abs(hamiltonian.c).sum()
                )
                print(
                    f"spec {num_devices} {label} {jax.typeof(vinit).sharding.spec} "
                    f"{jax.typeof(filtered).sharding.spec}"
                )


if __name__ == "__main__":
    main()
