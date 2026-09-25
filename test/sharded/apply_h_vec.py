"""A host ``vec`` fed to ``apply_h`` on a 4-device mesh; see ``TestShardedApplyHVec``."""

import jax
import numpy as np
from common import emit, mesh
from qiskit.quantum_info import SparsePauliOp

from rqutils.paulis.symplectic import PauliSumXZ
from rqutils.sqd import apply_h, get_diag_signs, get_xsource, uniquify_states

NUM_QUBITS, NUM_STATES = 6, 23  # 23 is indivisible by 4; 24 is the rounded length


def raised(fn):
    """The ``ValueError`` message ``fn`` raises, or ``None`` if it returns."""
    try:
        fn()
    except ValueError as exc:
        return str(exc)
    return None


def main() -> None:
    jax.set_mesh(mesh(4))
    # Fixed rather than drawn, so the parent's expected value is reproducible by hand.
    x, z, c = PauliSumXZ.from_paulisum(
        SparsePauliOp(["XXIIII", "IIZZII", "IXIZII"], [0.7, -1.3, 0.4])
    ).arrays
    bits = np.array(
        [[(i >> k) & 1 for k in reversed(range(NUM_QUBITS))] for i in range(1, NUM_STATES + 1)],
        dtype=np.uint8,
    )
    packed = PauliSumXZ.pack_states(bits)
    size = NUM_STATES + 4 - NUM_STATES % 4
    states = uniquify_states(packed, size)
    vec = np.zeros(size)
    vec[:NUM_STATES] = np.random.default_rng(0).normal(size=NUM_STATES)
    vec /= np.linalg.norm(vec)

    out = apply_h(vec, xsignatures=x, zsignatures=z, coeffs=c, states=states)
    result = {
        "placed": float(np.asarray(out * vec).sum()),
        "spec": str(jax.typeof(out).sharding.spec),
    }
    # A single-device-committed jax.Array carries an empty mesh exactly as a host array does.
    committed = apply_h(
        jax.device_put(vec, jax.devices()[0]), xsignatures=x, zsignatures=z, coeffs=c, states=states
    )
    result["committed_diff"] = float(np.abs(np.asarray(committed) - np.asarray(out)).max())

    # An indivisible length must name the size for every diagonal strategy.
    st = uniquify_states(packed, NUM_STATES)
    short = vec[:NUM_STATES]
    strategies = {
        "zsignatures": {"zsignatures": z, "coeffs": c},
        "diagonals": {"diagonals": np.ones((x.shape[0], NUM_STATES))},
        "diag_signs": {"diag_signs": np.stack([get_diag_signs(zg, st) for zg in z]), "coeffs": c},
    }
    result["size"] = size
    result["raised"] = {
        name: raised(lambda kw=kw: apply_h(short, xsignatures=x, states=st, **kw))
        for name, kw in strategies.items()
    }
    # A divisible vec against an indivisible states: the check must read `states`, not `vec`.
    result["raised"]["mismatch"] = raised(
        lambda: apply_h(vec, xsignatures=x, zsignatures=z, coeffs=c, states=st)
    )
    # A batched (k, N) vec: the length check must read shape[-1].
    batched = np.asarray(
        apply_h(np.stack([vec, vec * 2.0]), xsignatures=x, zsignatures=z, coeffs=c, states=states)
    )
    result["batched_shape"] = list(batched.shape)
    result["batched_diff"] = float(np.abs(batched[0] - np.asarray(out)).max())
    # `xsources=` does no search, so no reshard and no divisibility requirement.
    xs = np.stack([np.asarray(get_xsource(xi, states)) for xi in x])[:, :NUM_STATES]
    result["xsources_len"] = int(
        np.asarray(apply_h(short, xsources=xs, diagonals=np.ones((len(x), NUM_STATES)))).shape[0]
    )
    emit(result)


if __name__ == "__main__":
    main()
