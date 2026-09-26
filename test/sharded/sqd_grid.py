"""``sqd`` sharded against single-device over every ``matvec`` and mesh size; see ``TestShardedSqd``."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
from common import emit, mesh
from jax.sharding import PartitionSpec

import rqutils.sqd as sqd_module
from rqutils.ground_locg import _chebyshev_prefilter
from rqutils.paulis.symplectic import PauliSumXZ
from rqutils.sqd import _MATVECS, _SPARSE_MATVECS, sqd

# 37 states, indivisible by every mesh size, pad to 64, which each divides.
NUM_QUBITS, NUM_STATES, NUM_TERMS, STATES_SIZE = 8, 37, 5, 64
PREFILTER = (16, 2)
MESH_SIZES = (1, 2, 4)
# pairs/csr are single-device and raise under a mesh (sparse_mesh.py), so they are excluded here.
DENSE = [name for name in _MATVECS if name not in _SPARSE_MATVECS]


def main() -> None:
    rng = np.random.default_rng(11)
    # I/X/Z only: an even Y count keeps `.c` float64, the real-symmetric path a Hamiltonian takes.
    strings = ["".join(rng.choice(list("IXZ"), size=NUM_QUBITS)) for _ in range(NUM_TERMS)]
    coeffs = rng.normal(size=NUM_TERMS).tolist()
    states = rng.integers(0, 2, size=(NUM_STATES, NUM_QUBITS)).astype(np.uint8)

    def solve(matvec):
        return float(
            sqd(
                (strings, coeffs),
                states,
                return_eigvec=False,
                matvec=matvec,
                prefilter=PREFILTER,
            )
        )

    single = {name: solve(name) for name in DENSE}
    sharded = {}
    for num_devices in MESH_SIZES:
        with jax.set_mesh(mesh(num_devices)):
            sharded[num_devices] = {name: solve(name) for name in DENSE}
    emit({"single": single, "sharded": sharded, "specs": prefilter_specs(strings, coeffs, states)})


def prefilter_specs(strings, coeffs, states):
    """``{devices: {label: [vinit spec, filtered spec]}}``, with the matvec assembled as ``run_sqd`` does.

    Not through ``sqd``: it reshards the eigenvector to ``P(None)`` on return, which hides the
    partitioning the filter must preserve. Mirrors ``matvec="indices"``.
    """
    hamiltonian = PauliSumXZ.from_paulisum((strings, coeffs))
    states_p = PauliSumXZ.pack_states(states)
    states_p = sqd_module._pad_states(states_p, STATES_SIZE)
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
