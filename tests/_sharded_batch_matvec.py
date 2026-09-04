"""Run ``run_sqd`` with ``batch_matvec`` on and off, on a 4-device mesh and single-device.

Driven as a subprocess by ``test_sqd.py::TestShardedBatchMatvec``, which owns the rationale. It lives
in a file rather than a ``textwrap.dedent`` blob so ruff and ty check it: as a blob, an ``ImportError``
from a rename in :mod:`rqutils.sqd` would surface as a nonzero exit, indistinguishable from the
sharding regression it exists to catch.

Not a pytest module (the leading underscore keeps it uncollected): the virtual device count must be set
before jax initializes, which ``conftest.py`` has already done by collection time.

Why this needs a sharded arm at all. Batching stacks two ``(N,)`` vectors into ``(2, N)``, which moves
the partitioned axis from position 0 to position 1. ``jnp.stack`` on a ``P('x')`` vector yields
``P(None, 'x')`` -- batch axis replicated, data axis still partitioned -- so nothing reshards and no
collective appears. That is the whole correctness argument, and it is invisible single-device: a
resharding bug here would show up as an all-gather (a slowdown, not a wrong answer) or as a spec of
``P('x', None)`` (a wrong answer). Values alone cannot see either, which is why this prints the spec.

``batch_matvec`` goes through ``run_sqd`` rather than ``sqd``: the public entry point does not forward
it, so the flag has no other reachable call site.
"""

import jax

# Must precede the first array creation, exactly as conftest.py does for the suite.
jax.config.update("jax_enable_x64", True)

import numpy as np
from jax.sharding import AxisType

from rqutils.paulis.symplectic import PauliSumXZ
from rqutils.sqd import run_sqd

# N mod 4 != 0, so the mesh-size rounding in `run_sqd` is exercised rather than skipped.
NUM_QUBITS, NUM_STATES, NUM_TERMS, MESH_SIZE = 8, 37, 6, 4


def main() -> None:
    rng = np.random.default_rng(23)
    # I/X/Z only: an even Y count keeps the folded phase real, so `.c` stays float64 and the run
    # exercises the real-symmetric path a physical Hamiltonian takes.
    labels = ["".join(rng.choice(list("IXZ"), size=NUM_QUBITS)) for _ in range(NUM_TERMS)]
    coeffs = rng.normal(size=NUM_TERMS).tolist()
    hamiltonian = PauliSumXZ.from_paulisum((labels, coeffs))
    states = np.unique(rng.integers(0, 2, size=(NUM_STATES, NUM_QUBITS), dtype=np.uint8), axis=0)
    states_p = PauliSumXZ.pack_states(states)
    states_size = 1 << (int(states.shape[0] - 1).bit_length())

    mesh = jax.make_mesh((MESH_SIZE,), ("x",), (AxisType.Explicit,))

    def run(batch: bool, sharded: bool) -> float:
        kwargs = {
            "hamiltonian": hamiltonian,
            "states_p": states_p,
            "states_size": states_size,
            "return_eigvec": False,
            "batch_matvec": batch,
        }
        if not sharded:
            return float(run_sqd(**kwargs)[0])
        with jax.set_mesh(mesh):
            return float(run_sqd(**kwargs)[0])

    for batch in (False, True):
        single = run(batch, sharded=False)
        sharded = run(batch, sharded=True)
        print(f"energy {int(batch)} {single!r} {sharded!r}")

    # The spec check, which is the half a value comparison cannot make. A stacked pair must keep the
    # data axis partitioned and leave the new batch axis replicated; `P('x', None)` would mean the
    # stack partitioned the wrong axis, and a fully replicated spec would mean sharding was silently
    # dropped -- both of which agree with single-device to exactly 0.0.
    with jax.set_mesh(mesh):
        vec = jax.device_put(
            jax.numpy.arange(float(states_size)), jax.NamedSharding(mesh, jax.P("x"))
        )
        stacked = jax.numpy.stack((vec, vec))
        print(
            f"spec {tuple(jax.typeof(vec).sharding.spec)!r} "
            f"{tuple(jax.typeof(stacked).sharding.spec)!r}"
        )


if __name__ == "__main__":
    main()
