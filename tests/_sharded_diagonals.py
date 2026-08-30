"""Sharding coverage for the popcount diagonal path, under virtual devices.

Subprocessed by ``tests/test_sqd.py`` because the virtual device count comes from
``XLA_FLAGS=--xla_force_host_platform_device_count``, which XLA reads at backend initialization --
long before an in-process test could set it.

``_z_parity`` is ``sum(bitwise_count(states & z), axis=1) & 1``: two elementwise ops and a reduction
along the **byte** axis. With states partitioned ``P('x', None)`` the sharded axis is axis 0, so the
reduction runs entirely within each device's own rows and no collective is needed. That is the easy
half of the rule in ``NOTES.md`` -- only elementwise ops and reductions survive a partitioned axis --
and it is exactly the kind of "should be free" this repo requires measuring rather than assuming.

**The spec is asserted alongside the values, not instead of them.** A replicated run agrees with
single-device to exactly 0.0, so a silently *unsharded* builder is invisible to value comparison; and a
spec check alone would not catch a wrong sign. Both, over every X group, and over both coefficient
dtypes -- ``PauliSumXZ`` narrows ``.c`` to float64 only when every string has an even Y count, so the
odd-Y case exercises the complex path that ``vinit_from_min_diag`` once broke on.

Line protocol, one per case: ``<label> <num_devices> <groups> <bad_spec> <bad_value>``.
"""

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax.sharding import AxisType, NamedSharding, PartitionSpec

from rqutils.paulis.symplectic import PauliSumXZ
from rqutils.sqd import compute_diagonal, get_diag_signs, get_diagonal


def xxz_terms(num_qubits, odd_y):
    """1D XXZ, periodic. ``odd_y`` appends one odd-Y string so the folded coefficients go complex."""
    labels, coeffs = [], []
    for q in range(num_qubits):
        a, b = q, (q + 1) % num_qubits
        for letter in "XYZ":
            term = ["I"] * num_qubits
            term[a] = term[b] = letter
            labels.append("".join(term))
            coeffs.append(1.0)
    if odd_y:
        term = ["I"] * num_qubits
        term[0] = "Y"
        labels.append("".join(term))
        coeffs.append(0.5)
    return labels, coeffs


def hamiltonian(num_qubits, odd_y):
    """Build a ``PauliSumXZ`` without requiring qiskit -- this script runs in the dev extra only."""
    from rqutils.paulis.symplectic import PauliSumXZ as _P  # noqa: F401  (import-time check)

    labels, coeffs = xxz_terms(num_qubits, odd_y)
    try:
        from qiskit.quantum_info import SparsePauliOp
    except ImportError:
        return None
    return PauliSumXZ.from_paulisum(SparsePauliOp(labels, coeffs).simplify())


def subspace(num_qubits, num_rows, rng):
    """Fixed-Hamming-weight rows, packed with the mandatory pad bit at position 0."""
    rows = np.zeros((num_rows, num_qubits), np.uint8)
    for i in range(num_rows):
        rows[i, rng.choice(num_qubits, num_qubits // 2, replace=False)] = 1
    padded = np.concatenate([np.zeros((num_rows, 1), np.uint8), rows], axis=1)
    return np.unique(np.packbits(padded, axis=1), axis=0)


def main():
    num_qubits = 20
    rng = np.random.default_rng(20260830)
    for odd_y in (False, True):
        ham = hamiltonian(num_qubits, odd_y)
        if ham is None:
            print(f"odd_y={odd_y} skipped-no-qiskit")
            continue
        label = "complex" if np.iscomplexobj(ham.c) else "real"
        for num_devices in (2, 4):
            mesh = jax.make_mesh(
                (num_devices,),
                ("x",),
                devices=jax.devices()[:num_devices],
                axis_types=(AxisType.Explicit,),
            )
            states = subspace(num_qubits, 512, rng)
            # Explicit sharding rejects a ragged split at device_put, so trim rather than sweep it.
            states = states[: (len(states) // num_devices) * num_devices]
            bad_spec = bad_value = 0
            with jax.set_mesh(mesh):
                partitioned = jax.device_put(
                    jnp.asarray(states), NamedSharding(mesh, PartitionSpec("x", None))
                )
                for group in range(ham.x.shape[0]):
                    zsig, coeff = ham.z[group], ham.c[group]
                    ref_signs = np.asarray(get_diag_signs(zsig, jnp.asarray(states)))
                    ref_diag = np.asarray(get_diagonal(zsig, coeff, jnp.asarray(states)))
                    ref_comp = np.asarray(compute_diagonal(jnp.asarray(ref_signs), coeff))

                    got_signs = get_diag_signs(zsig, partitioned)
                    got_diag = get_diagonal(zsig, coeff, partitioned)
                    got_comp = compute_diagonal(got_signs, coeff)
                    for value in (got_signs, got_diag, got_comp):
                        if "x" not in str(jax.typeof(value).sharding.spec):
                            bad_spec += 1
                    for ref, got in (
                        (ref_signs, got_signs),
                        (ref_diag, got_diag),
                        (ref_comp, got_comp),
                    ):
                        if not np.array_equal(ref, np.asarray(got)):
                            bad_value += 1
            print(f"{label} {num_devices} {ham.x.shape[0]} {bad_spec} {bad_value}")


if __name__ == "__main__":
    main()
