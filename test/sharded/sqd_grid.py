"""``sqd`` sharded against single-device over every cache level and mesh size; see ``TestShardedSqd``."""

import functools
import itertools

import jax
import jax.numpy as jnp
import numpy as np
from common import emit, mesh
from jax.sharding import PartitionSpec

import rqutils.sqd as sqd_module
from rqutils.ground_locg import _chebyshev_prefilter
from rqutils.paulis.symplectic import PauliSumXZ
from rqutils.sqd import sqd

# 37 states, indivisible by every mesh size, pad to 64, which each divides.
NUM_QUBITS, NUM_STATES, NUM_TERMS, STATES_SIZE = 8, 37, 5, 64
PREFILTER = (16, 2)
MESH_SIZES = (1, 2, 4)
CACHE_LEVELS = sorted(itertools.product((0, 1), (0, 1, 2)))


def main() -> None:
    rng = np.random.default_rng(11)
    # I/X/Z only: an even Y count keeps `.c` float64, the real-symmetric path a Hamiltonian takes.
    strings = ["".join(rng.choice(list("IXZ"), size=NUM_QUBITS)) for _ in range(NUM_TERMS)]
    coeffs = rng.normal(size=NUM_TERMS).tolist()
    states = rng.integers(0, 2, size=(NUM_STATES, NUM_QUBITS)).astype(np.uint8)

    def solve(cache_level):
        return float(
            sqd(
                (strings, coeffs),
                states,
                return_eigvec=False,
                cache_level=cache_level,
                prefilter=PREFILTER,
            )
        )

    single = {str(level): solve(level) for level in CACHE_LEVELS}
    sharded = {}
    for num_devices in MESH_SIZES:
        with jax.set_mesh(mesh(num_devices)):
            sharded[num_devices] = {str(level): solve(level) for level in CACHE_LEVELS}
    emit({"single": single, "sharded": sharded, "specs": prefilter_specs(strings, coeffs, states)})


def prefilter_specs(strings, coeffs, states):
    """``{devices: {label: [vinit spec, filtered spec]}}``, with the matvec assembled as ``run_sqd`` does.

    Not through ``sqd``: it reshards the eigenvector to ``P(None)`` on return, which hides the
    partitioning the filter must preserve. Mirrors ``cache_level == (1, 0)``.
    """
    hamiltonian = PauliSumXZ.from_paulisum((strings, coeffs))
    states_p = PauliSumXZ.pack_states(states)
    padding = np.full((STATES_SIZE - len(states_p), states_p.shape[1]), 255, dtype=np.uint8)
    states_p = np.append(states_p, padding, axis=0)
    specs = {}
    for num_devices in MESH_SIZES:
        specs[num_devices] = {}
        with jax.set_mesh(mesh(num_devices)):
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
                specs[num_devices][label] = [
                    str(jax.typeof(vinit).sharding.spec),
                    str(jax.typeof(filtered).sharding.spec),
                ]
    return specs


if __name__ == "__main__":
    main()
