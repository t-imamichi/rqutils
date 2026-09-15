"""Feed a host ``vec`` to ``apply_h`` on a 4-device mesh, printing the result and the raise.

Driven by ``test_sqd.py::TestShardedApplyHVec``. A file rather than an inline blob so ruff and ty
check it; not collected by pytest (leading underscore) because the virtual device count must be set
before jax initializes.
"""

import jax

# Must precede the first array creation, exactly as conftest.py does for the suite.
jax.config.update("jax_enable_x64", True)

import numpy as np
from jax.sharding import AxisType
from qiskit.quantum_info import SparsePauliOp

from rqutils.paulis.symplectic import PauliSumXZ
from rqutils.sqd import apply_h, get_diag_signs, get_xsource, uniquify_states

# 23 is deliberately indivisible by 4; 24 is the rounded length.
NUM_QUBITS, NUM_STATES, MESH_SIZE = 6, 23, 4


def main() -> None:
    jax.set_mesh(jax.make_mesh((MESH_SIZE,), ("x",), (AxisType.Explicit,)))

    # Fixed, not drawn: this exercises mesh placement, which is agnostic to the operator's content, so
    # a random draw would only make the expected value in the parent test unreproducible by hand.
    x, z, c = PauliSumXZ.from_paulisum(
        SparsePauliOp(["XXIIII", "IIZZII", "IXIZII"], [0.7, -1.3, 0.4])
    ).arrays

    bits = np.array(
        [[(i >> k) & 1 for k in reversed(range(NUM_QUBITS))] for i in range(1, NUM_STATES + 1)],
        dtype=np.uint8,
    )
    packed = PauliSumXZ.pack_states(bits)

    # A host vec at the rounded length must now work rather than raising on the empty mesh.
    size = NUM_STATES + MESH_SIZE - NUM_STATES % MESH_SIZE
    states = uniquify_states(packed, size)
    vec = np.zeros(size)
    vec[:NUM_STATES] = np.random.default_rng(0).normal(size=NUM_STATES)
    vec /= np.linalg.norm(vec)

    out = apply_h(vec, xsignatures=x, zsignatures=z, coeffs=c, states=states)
    print(f"placed {float(np.asarray(out * vec).sum()):.15e}")
    print(f"spec {jax.typeof(out).sharding.spec}")

    # A jax.Array committed to ONE device carries an empty mesh exactly as a host array does, so an
    # `isinstance(vec, jax.Array)` guard passed it through to the same "Resource axis" raise. It must
    # be placed too, and must agree with the host-array arm bit-for-bit.
    committed = apply_h(
        jax.device_put(vec, jax.devices()[0]),
        xsignatures=x,
        zsignatures=z,
        coeffs=c,
        states=states,
    )
    print(f"committed {float(np.abs(np.asarray(committed) - np.asarray(out)).max()):.15e}")

    # An UNROUNDED length is the point of the change: a bare vector plus
    # `uniquify_states(packed, packed.shape[0])` must work and agree with the hand-padded arm above
    # over the real rows, with a zero pad row.
    rounded = apply_h(
        vec[:NUM_STATES],
        xsignatures=x,
        zsignatures=z,
        coeffs=c,
        states=uniquify_states(packed, NUM_STATES),
    )
    rounded = np.asarray(rounded)
    print(f"rounded_len {rounded.shape[0]}")
    print(f"rounded_pad {float(np.abs(rounded[NUM_STATES:]).max()):.15e}")

    # `vec` was normalized at the padded width, so renormalize the trimmed reference before comparing.
    trimmed = vec[:NUM_STATES] / np.linalg.norm(vec[:NUM_STATES])
    ref = np.asarray(
        apply_h(
            np.append(trimmed, np.zeros(size - NUM_STATES)),
            xsignatures=x,
            zsignatures=z,
            coeffs=c,
            states=states,
        )
    )
    scaled = rounded / np.linalg.norm(vec[:NUM_STATES])
    print(f"rounded_agrees {float(np.abs(scaled - ref).max()):.15e}")

    # An `xsources=` strategy does no search, so its length must be left alone.
    xs = np.stack([np.asarray(get_xsource(xi, states)) for xi in x])[:, :NUM_STATES]
    kept = apply_h(vec[:NUM_STATES], xsources=xs, diagonals=np.ones((len(x), NUM_STATES)))
    print(f"xsources_len {np.asarray(kept).shape[0]}")

    # A vec genuinely disagreeing with `states` must still fail rather than be padded into agreement:
    # the rounding pads both to ONE length, it does not reconcile two different ones.
    try:
        apply_h(
            vec[: NUM_STATES - 5],
            xsignatures=x,
            zsignatures=z,
            coeffs=c,
            states=uniquify_states(packed, NUM_STATES),
        )
        print("mismatch_raised False")
    except (ValueError, TypeError, IndexError):
        print("mismatch_raised True")

    # `diagonals=`/`diag_signs=` are per-state with different state axes, so an unrounded length must
    # name the size rather than reaching the scan's broadcast TypeError.
    st = uniquify_states(packed, NUM_STATES)
    for name, kwargs in (
        ("diagonals", {"diagonals": np.ones((x.shape[0], NUM_STATES))}),
        (
            "diag_signs",
            {
                "diag_signs": np.stack([np.asarray(get_diag_signs(zg, st)) for zg in z]),
                "coeffs": c,
            },
        ),
    ):
        try:
            apply_h(vec[:NUM_STATES], xsignatures=x, states=st, **kwargs)
            print(f"{name}_named False")
        except ValueError as exc:
            print(f"{name}_named {str(size) in str(exc)}")


if __name__ == "__main__":
    main()
