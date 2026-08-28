"""Run ``sqd(prefilter=...)`` single-device and on meshes, at every ``cache_level``.

Driven as a subprocess by ``test_sqd.py::TestShardedSqdPrefilter``, which owns the rationale. In a
file rather than an inline string so ruff and ty check it, and underscore-prefixed so pytest does not
collect it -- the virtual device count comes from ``XLA_FLAGS``, which XLA reads at backend
initialization, long before an in-process test could set it.

**Why this exists when ``_sharded_prefilter.py`` already covers the prefilter on a mesh:** that script
drives ``ground_locg`` with a dense ``einsum`` matvec on an unpadded power-of-two vector.
``docs/locg-chebyshev-prefilter.md`` says so explicitly and defers the rest here -- *"Ragged mesh
splits are not swept because they are unreachable: explicit sharding rejects ``dim % mesh.size != 0``
at ``device_put``. That is ``sqd``'s concern, where ``uniquify_states`` pads to a power of two."* So
``sqd`` is the only place two things meet: a **padded** subspace whose filler slots are masked to zero,
and a **partitioned** vector fed through ``apply_h``'s gather-heavy irregular kernel rather than a
dense matmul.

The prefilter calls that matvec ``cycles * (degree + 1)`` times before the solver's first iteration, so
it gets far more exposure to a sharding fault than one LOBPCG step does. ``cache_level`` selects which
of the six kernels it calls, and per ``CLAUDE.md`` three bugs have hidden behind the default ``(1, 0)``
-- so this sweeps the grid rather than sampling it, exactly as ``_sharded_cache_levels.py`` does after
one representative cell reported success at three broken ones.

Both mesh sizes and ``NUM_STATES`` are chosen so the padded size divides the mesh while the *genuine*
row count does not: 37 real states pad to 64, which 2 and 4 both divide. That is the arrangement
``uniquify_states`` exists to produce, and it is unreachable in ``_sharded_prefilter.py``.

Two line protocols, because value agreement alone cannot detect the failure that matters. Per
``CLAUDE.md`` a replicated run agrees with single-device to *exactly* 0.0, so a silently unsharded
prefilter is invisible to a value comparison -- and measured, every one of the 18 energy cases below
agrees to 4e-16 or better whether or not the filter preserves partitioning. So:

* ``energy <devices> <i> <j> <single> <sharded>`` -- the end-to-end answer, swept over the grid.
* ``spec <devices> <spec_label> <vinit_spec> <filtered_spec>`` -- the prefilter's output sharding,
  read off ``jax.typeof`` inside ``sqd``'s own assembled matvec. Both the partitioned and the
  replicated input pairing are swept: the replicated one is the shape of the ``_spread_seed``
  ``ShardingTypeError`` recorded in ``_sharded_cache_levels.py``, where a replicated predicate met a
  partitioned vector.
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

# 37 genuine states pad to 64, which every mesh size below divides -- see the module docstring.
NUM_QUBITS, NUM_STATES, NUM_TERMS = 8, 37, 5
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
    states_p = np.packbits(np.pad(states, {1: (1, 0)}), axis=1)
    states_size = 1 << (len(states_p) - 1).bit_length()
    padding = np.full((states_size - len(states_p), states_p.shape[1]), 255, dtype=np.uint8)
    states_p = np.append(states_p, padding, axis=0)

    for num_devices in MESH_SIZES:
        mesh = jax.make_mesh((num_devices,), ("x",), (AxisType.Explicit,))
        with jax.set_mesh(mesh):
            states_u = jax.reshard(
                sqd_module.uniquify_states(states_p, states_size), PartitionSpec(None, None)
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
                vinit = sqd_module._spread_seed(states_size, states_u, hamiltonian.c.dtype, spec)
                filtered = _chebyshev_prefilter(
                    matvec, (), vinit, PREFILTER[0], PREFILTER[1], jnp.abs(hamiltonian.c).sum()
                )
                print(
                    f"spec {num_devices} {label} {jax.typeof(vinit).sharding.spec} "
                    f"{jax.typeof(filtered).sharding.spec}"
                )


if __name__ == "__main__":
    main()
