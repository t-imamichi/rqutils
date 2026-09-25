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

import os
import sys

import jax

# Must precede the first array creation, exactly as conftest.py does for the suite.
jax.config.update("jax_enable_x64", True)

import numpy as np
from jax.sharding import AxisType

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from conftest import real_pauli_strings, unique_states

from rqutils.paulis.symplectic import PauliSumXZ
from rqutils.sqd import run_sqd

# `NUM_STATES` is the draw count, not the row count: `unique_states` collapses duplicates, so this
# fixture yields 34 rows (29-37 over 200 seeds) and `STATES_SIZE = 64` bounds it with room to spare.
# The 64 is a constant rather than `sqd`'s `bit_length` formula, for the reason
# `_sharded_sqd_prefilter.py` gives: a second copy of that formula could drift from the original. That
# makes it independent of `NUM_STATES`, so the padding below asserts the bound rather than trusting it
# -- raising `NUM_STATES` past 64 would otherwise hand `np.full` a negative dimension inside a
# subprocess whose stderr the caller truncates.
NUM_QUBITS, NUM_STATES, NUM_TERMS, STATES_SIZE, MESH_SIZE = 8, 37, 6, 64, 4


def main() -> None:
    rng = np.random.default_rng(23)
    # `real_pauli_strings` keeps the Y count even, so the folded phase stays real, `.c` narrows to
    # float64, and the run exercises the real-symmetric path a physical Hamiltonian takes.
    labels = real_pauli_strings(NUM_QUBITS, NUM_TERMS, rng)
    coeffs = rng.normal(size=NUM_TERMS).tolist()
    hamiltonian = PauliSumXZ.from_paulisum((labels, coeffs))
    states = unique_states(NUM_STATES, NUM_QUBITS, rng)
    # Padded up to STATES_SIZE here because this calls `run_sqd` directly: `sqd` is what normally
    # does this, and 255 is the filler for the reason `uniquify_states` uses it -- an all-ones row
    # sorts to the end and its high bit in byte 0 is what `_is_filler` tests.
    states_p = PauliSumXZ.pack_states(states)
    deficit = STATES_SIZE - states_p.shape[0]
    assert deficit >= 0, (
        f"{states_p.shape[0]} unique rows exceed STATES_SIZE={STATES_SIZE}; raise the constant"
    )
    states_p = np.append(
        states_p, np.full((deficit, states_p.shape[1]), 255, dtype=np.uint8), axis=0
    )

    mesh = jax.make_mesh((MESH_SIZE,), ("x",), (AxisType.Explicit,))

    args = (hamiltonian, states_p, STATES_SIZE, False)
    for batch in (False, True):
        single = float(run_sqd(*args, batch_matvec=batch)[0])
        with jax.set_mesh(mesh):
            sharded = float(run_sqd(*args, batch_matvec=batch)[0])
        print(f"energy {int(batch)} {single!r} {sharded!r}")

    # The spec check, which is the half a value comparison cannot make. A stacked pair must keep the
    # data axis partitioned and leave the new batch axis replicated; `P('x', None)` would mean the
    # stack partitioned the wrong axis, and a fully replicated spec would mean sharding was silently
    # dropped -- both of which agree with single-device to exactly 0.0.
    with jax.set_mesh(mesh):
        vec = jax.device_put(
            jax.numpy.arange(float(STATES_SIZE)), jax.NamedSharding(mesh, jax.P("x"))
        )
        stacked = jax.numpy.stack((vec, vec))
        print(
            f"spec {tuple(jax.typeof(vec).sharding.spec)!r} "
            f"{tuple(jax.typeof(stacked).sharding.spec)!r}"
        )


if __name__ == "__main__":
    main()
