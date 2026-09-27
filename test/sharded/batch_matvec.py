"""``ground_locg`` on ``sqd``'s own matvec, batched and unbatched, sharded and single-device.

See ``TestShardedBatchMatvec``. ``run_sqd`` always batches, so the unbatched arm is only reachable here.
"""

import functools
import re

import jax
import jax.numpy as jnp
import numpy as np
from common import emit, mesh
from conftest import real_pauli_strings, unique_states
from jax.sharding import PartitionSpec, get_abstract_mesh

from rqutils.ground_locg import ground_locg
from rqutils.paulis.symplectic import PauliSumXZ
from rqutils.sqd import get_xsource, uniquify_states
from rqutils.sqd._dense import _apply_h_kernel, _pack_scanned
from rqutils.sqd._solve import _spread_seed
from rqutils.sqd._states import _pad_states

# 37 draws collapse to ~34 unique rows; `_pad_states` raises if that ever exceeds STATES_SIZE.
NUM_QUBITS, NUM_STATES, NUM_TERMS, STATES_SIZE = 8, 37, 6, 64
MATVEC = "indices"  # run_sqd's default


@jax.jit
def operator(hamiltonian, states_p):
    """``run_sqd``'s ``"indices"`` matvec args and spread seed, built as it builds them."""
    sharding = None if (m := get_abstract_mesh()).empty else PartitionSpec(m.axis_names)
    states_u = uniquify_states(states_p, STATES_SIZE)
    xsources = jax.lax.scan(lambda _, x: (None, get_xsource(x, states_u)), None, hamiltonian.x)[1]
    if sharding:
        states_u = jax.reshard(states_u, sharding)
    scanned = _pack_scanned(MATVEC, xsources, hamiltonian.z, hamiltonian.c)
    return (scanned, states_u), _spread_seed(STATES_SIZE, states_u, hamiltonian.c.dtype, sharding)


# No prefilter: its matvecs are sequential, so batching cannot touch them, and without it the solve
# runs enough iterations for the iteration-count comparison to have teeth.
@functools.partial(jax.jit, static_argnames="batch")
def solve(vinit, args, batch):
    matvec = functools.partial(_apply_h_kernel, matvec=MATVEC)
    return ground_locg(matvec, vinit, args=args, batch_matvec=batch)


def loop_gathers(text):
    """All-gathers in the busiest ``while`` body, i.e. one ``ground_locg`` iteration."""
    counts = [0]
    for name in set(re.findall(r"while\(.*?body=(%?[\w.\-]+)", text)):
        start = re.search(rf"^{re.escape(name)} .*\{{$", text, re.MULTILINE).start()
        counts.append(
            len(re.findall(r" all-gather(?:-start)?\(", text[start : text.index("\n}", start)]))
        )
    return max(counts)


def run(hamiltonian, states_p, batch):
    """``[theta, iterations, all-gathers in one loop iteration]`` on whatever mesh is live."""
    args, vinit = operator(hamiltonian, states_p)
    theta, _, niter, converged = solve(vinit, args, batch)
    assert bool(converged), f"batch={batch} did not converge"
    text = solve.lower(vinit, args, batch).compile().as_text()
    return [float(theta), int(niter), loop_gathers(text)]


def main() -> None:
    rng = np.random.default_rng(23)
    labels = real_pauli_strings(NUM_QUBITS, NUM_TERMS, rng)
    hamiltonian = PauliSumXZ.from_paulisum((labels, rng.normal(size=NUM_TERMS).tolist()))
    states = PauliSumXZ.pack_states(unique_states(NUM_STATES, NUM_QUBITS, rng))
    states_p = _pad_states(states, STATES_SIZE)

    the_mesh = mesh(4)
    runs = {}
    for batch in (False, True):
        single = run(hamiltonian, states_p, batch)
        with jax.set_mesh(the_mesh):
            runs[str(batch)] = [single, run(hamiltonian, states_p, batch)]
    with jax.set_mesh(the_mesh):
        vec = jax.device_put(
            jnp.arange(float(STATES_SIZE)), jax.NamedSharding(the_mesh, jax.P("x"))
        )
        stacked = jnp.stack((vec, vec))
        specs = [str(jax.typeof(vec).sharding.spec), str(jax.typeof(stacked).sharding.spec)]
    emit({"runs": runs, "specs": specs})


if __name__ == "__main__":
    main()
