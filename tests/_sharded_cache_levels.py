"""Run ``sqd`` single-device and on a 4-device mesh at every ``cache_level``, printing both energies.

Driven as a subprocess by ``test_sqd.py::TestShardedCacheLevels``, which owns the rationale. It lives
in a file rather than an inline string so ruff and ty check it: as a ``textwrap.dedent`` blob it was
invisible to both, and an ``ImportError`` from a rename in :mod:`rqutils.sqd` would surface as a
nonzero exit -- indistinguishable from the sharding regression it exists to catch.

Not a pytest module (the leading underscore keeps it uncollected): it must set the virtual device
count before jax initializes, which ``conftest.py`` has already done by collection time.

**All six cache levels, not just the default.** Two distinct sharding bugs lived in the three
``cache_level[0] == 0`` cells, which nothing covered -- ``examples/scaling/poc7_sharding.py`` and the
first version of this script both ran only ``sqd``'s default ``(1, 0)``:

* ``_accumulate_diagonal`` put a rank-2 ``PartitionSpec`` on a rank-1 accumulator, which failed
  **all six** levels.
* With that fixed, ``_spread_seed``'s ``jnp.where`` mixed a replicated predicate with a partitioned
  ``vec``, because ``run_sqd`` reshards ``states_u`` only inside ``if cache_level[0] == 1``. The first
  bug masked the second: it raised earlier in the call, so fixing it turned 6 failures into 3.

That masking is the reason this sweeps the grid rather than sampling it -- one representative cell
would have reported success at three broken ones.

``examples/scaling/poc7_sharding.py`` remains the thorough arm (it also covers ``return_eigvec``
round-trips and more sizes). Subprocessing it from pytest was measured at 59.7 s against ~1 s here.
"""

import itertools

import jax

# Must precede the first array creation, exactly as conftest.py does for the suite.
jax.config.update("jax_enable_x64", True)

import numpy as np
from jax.sharding import AxisType

from rqutils.sqd import sqd

# N mod 4 != 0, so the mesh-size rounding in `sqd` is exercised rather than skipped.
NUM_QUBITS, NUM_STATES, NUM_TERMS, MESH_SIZE = 8, 37, 5, 4
CACHE_LEVELS = sorted(itertools.product((0, 1), (0, 1, 2)))


def main() -> None:
    rng = np.random.default_rng(11)
    # I/X/Z only: an even Y count keeps the folded phase real, so `.c` stays float64 and the run
    # exercises the real-symmetric path the solver actually takes for a Hamiltonian.
    strings = ["".join(rng.choice(list("IXZ"), size=NUM_QUBITS)) for _ in range(NUM_TERMS)]
    coeffs = rng.normal(size=NUM_TERMS).tolist()
    states = rng.integers(0, 2, size=(NUM_STATES, NUM_QUBITS)).astype(np.uint8)

    mesh = jax.make_mesh((MESH_SIZE,), ("x",), (AxisType.Explicit,))
    for cache_level in CACHE_LEVELS:
        single = float(sqd((strings, coeffs), states, return_eigvec=False, cache_level=cache_level))
        with jax.set_mesh(mesh):
            sharded = float(
                sqd((strings, coeffs), states, return_eigvec=False, cache_level=cache_level)
            )
        # One line per level, parsed by the caller. Printed as repr so no precision is lost.
        print(f"{cache_level[0]} {cache_level[1]} {single!r} {sharded!r}")


if __name__ == "__main__":
    main()
