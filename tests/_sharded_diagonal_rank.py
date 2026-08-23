"""Run ``sqd`` single-device and on a 4-device mesh, and print both energies.

Driven as a subprocess by ``test_sqd.py::TestShardedDiagonalRank``, which owns the rationale. It
lives in a file rather than an inline string so ruff and ty check it: as a ``textwrap.dedent`` blob
it was invisible to both, and an ``ImportError`` from a rename in :mod:`rqutils.sqd` would surface as
a nonzero exit -- indistinguishable from the sharding regression the test exists to catch, and
reported under an assertion message blaming the sharding.

Not a pytest module (the leading underscore keeps it uncollected): it must set the virtual device
count before jax initializes, which ``conftest.py`` has already done by collection time.

``examples/scaling/poc7_sharding.py`` covers the same contract far more thoroughly and is the right
place to extend that coverage. This is deliberately the narrow arm -- subprocessing the POC instead
was measured at 59.7 s against 0.96 s here, a 60x cost for a full sweep this regression does not
need.
"""

import jax

# Must precede the first array creation, exactly as conftest.py does for the suite.
jax.config.update("jax_enable_x64", True)

import numpy as np
from jax.sharding import AxisType

from rqutils.sqd import sqd

# N mod 4 != 0, so the mesh-size rounding in `sqd` is exercised rather than skipped.
NUM_QUBITS, NUM_STATES, NUM_TERMS, MESH_SIZE = 8, 37, 5, 4


def main() -> None:
    rng = np.random.default_rng(11)
    # I/X/Z only: an even Y count keeps the folded phase real, so `.c` stays float64 and the run
    # exercises the real-symmetric path the solver actually takes for a Hamiltonian.
    strings = ["".join(rng.choice(list("IXZ"), size=NUM_QUBITS)) for _ in range(NUM_TERMS)]
    coeffs = rng.normal(size=NUM_TERMS).tolist()
    states = rng.integers(0, 2, size=(NUM_STATES, NUM_QUBITS)).astype(np.uint8)

    single = float(sqd((strings, coeffs), states, return_eigvec=False))
    mesh = jax.make_mesh((MESH_SIZE,), ("x",), (AxisType.Explicit,))
    with jax.set_mesh(mesh):
        sharded = float(sqd((strings, coeffs), states, return_eigvec=False))
    print(f"{single!r} {sharded!r}")


if __name__ == "__main__":
    main()
