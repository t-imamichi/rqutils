"""Run ``sqd(return_eigvec=True)`` on a 4-device mesh and check the vector is a genuine eigenvector.

Driven as a subprocess by ``test_sqd.py::TestShardedEigvecRoundtrip``, which owns the rationale. In a
file rather than an inline string so ruff and ty check it, for the reason
``_sharded_cache_levels.py`` gives: as a ``textwrap.dedent`` blob it is invisible to both, and an
``ImportError`` from a rename would surface as a nonzero exit indistinguishable from the regression
this exists to catch.

**Why this arm was missing.** Every other ``tests/_sharded_*.py`` calls ``sqd`` with
``return_eigvec=False``, so nothing covered the branch that reshards ``eigvec`` and ``states_u`` back
to ``PartitionSpec(None)`` before returning. ``examples/scaling/poc7_sharding.py`` covers it (as POC
7c) but is measured at 59.7 s subprocessed against ~1 s here, so the POC stays the thorough arm and
this is the distilled one.

**Checking the vector, not just that the call returned.** A reshard that dropped or reordered rows
still yields an array of the right shape and dtype, and the eigenvalue is computed separately, so
comparing shapes or reading ``eigval`` cannot see it. The assertion is the eigenvector *equation*:
``‖H v - E v‖ / ‖v‖`` against the dense projection, which is only small if every row of the returned
basis pairs with the right amplitude.

**And the basis states are checked against the projection they index.** ``eigvec`` and ``basis`` are
resharded separately, so a bug that permuted one relative to the other would leave both individually
plausible; building ``H`` from the *returned* basis and testing the *returned* vector against it is
what couples them.

``N mod 4 != 0`` deliberately, so the mesh-size padding in ``sqd`` runs rather than being skipped --
the padded slots are exactly what the ``subspace_dim`` slice on the return path has to remove, and a
reshard that mishandled filler rows would show up here rather than at a divisible length.
"""

import os
import sys

import jax
import numpy as np
from jax.sharding import AxisType

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from conftest import real_pauli_strings, unique_states

from rqutils.sqd import hproj, sqd

NUM_QUBITS = 8
# Not a multiple of the 4-device mesh, so `sqd`'s mesh-size padding is exercised.
NUM_STATES = 30


def main() -> int:
    if jax.device_count() < 4:
        print(f"FAIL: expected 4 devices, got {jax.device_count()}")
        return 1

    rng = np.random.default_rng(20260904)
    strings = real_pauli_strings(NUM_QUBITS, 12, rng)
    coeffs = rng.uniform(-1.0, 1.0, size=len(strings))
    states = unique_states(NUM_STATES, NUM_QUBITS, rng)

    mesh = jax.make_mesh((jax.device_count(),), ("x",), (AxisType.Explicit,))
    with jax.sharding.set_mesh(mesh):
        eigval, eigvec, basis = sqd((strings, coeffs), states, return_eigvec=True)

    eigvec = np.asarray(eigvec)
    basis = np.asarray(basis)

    # Shape first, so a length mismatch reports as itself rather than as a residual blow-up.
    if eigvec.shape != (len(basis),):
        print(f"FAIL: eigvec shape {eigvec.shape} does not match basis length {len(basis)}")
        return 1
    if len(basis) > NUM_STATES:
        print(f"FAIL: basis has {len(basis)} rows, more than the {NUM_STATES} states given")
        return 1

    # `hproj` returns a scipy csr_array: `.toarray()`, never `np.asarray`, which yields a 0-d object
    # array and fails frames later as an IndexError.
    dense = hproj((strings, coeffs), basis).toarray()
    residual = dense @ eigvec - eigval * eigvec
    relative = float(np.linalg.norm(residual) / max(np.linalg.norm(eigvec), 1e-300))

    # Compare the eigenvalue against the dense reference too: the residual alone is satisfied by any
    # eigenpair, so a solve that returned an excited state would pass it.
    reference = float(np.linalg.eigvalsh(dense)[0])

    print(f"eigval={eigval:.12f} reference={reference:.12f} relative_residual={relative:.3e}")
    print(f"basis_rows={len(basis)} eigvec_norm={float(np.linalg.norm(eigvec)):.12f}")
    if relative > 1e-10:
        print(f"FAIL: returned vector is not an eigenvector of its own basis ({relative:.3e})")
        return 1
    if abs(eigval - reference) > 1e-9 * max(abs(reference), 1.0):
        print(f"FAIL: eigenvalue disagrees with the dense minimum by {abs(eigval - reference):.3e}")
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
