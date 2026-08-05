"""Shared configuration and helpers for the rqutils test suite.

``jax_enable_x64`` is set here, at conftest import time, so that it takes effect before any test
module imports ``rqutils``. This is load-bearing rather than incidental: without x64 JAX silently
produces float32/complex64 and every tolerance in this suite is wrong by nine orders of magnitude.
"""

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np


def herm(n, rng, complex_=True):
    """Return a random ``(n, n)`` Hermitian matrix drawn from ``rng``."""
    mat = rng.normal(size=(n, n))
    if complex_:
        mat = mat + 1.0j * rng.normal(size=(n, n))
    return mat + mat.conjugate().T


def symmetrize(mat):
    """Mirror the lower triangle of ``mat`` over the diagonal.

    ``eigenpair_2x2`` and ``eigenpair_3x3`` read only the diagonal and the lower triangle, so a
    reference eigendecomposition must be taken of *this* matrix, not of the raw input. Comparing
    against ``eigvalsh`` of an unsymmetrized input compares against a different matrix.
    """
    lower = np.tril(mat)
    return lower + np.tril(mat, -1).conjugate().T


def lowest(mat):
    """Reference lowest eigenvalue of ``mat``, via LAPACK, after symmetrization."""
    return float(np.linalg.eigvalsh(symmetrize(mat))[0])


def rel_resid(mat, val, vec):
    """Eigenpair residual ``|Av - λv|``, scaled by ``max|A|``.

    The scaling is what makes a single tolerance usable across the shifted and extreme-scale cases:
    the 1e9-shifted 2x2 input has an absolute residual of 6e-8 but a relative residual of 6e-17.
    """
    mat = symmetrize(mat)
    vec = np.asarray(vec)
    return float(np.linalg.norm(mat @ vec - val * vec) / np.abs(mat).max())


_PAULI_MATRICES = {
    "I": np.eye(2, dtype=np.complex128),
    "X": np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128),
    "Y": np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128),
    "Z": np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128),
}


def dense_pauli_sum(pauli_strings, coeffs):
    """Return the full ``2**n``-by-``2**n`` matrix of ``sum(c * Q)``, by Kronecker products.

    Character order is most-significant-qubit-first, matching Qiskit's ``SparsePauliOp`` (verified:
    ``ZI``, ``IZ``, and ``XY`` all agree with ``SparsePauliOp(s).to_matrix()``). Only for small
    ``n`` -- this is ``4**n`` work.
    """
    num_qubits = len(pauli_strings[0])
    full = np.zeros((2**num_qubits,) * 2, dtype=np.complex128)
    for string, coeff in zip(pauli_strings, coeffs):
        operator = np.array([[1.0]], dtype=np.complex128)
        for char in string:
            operator = np.kron(operator, _PAULI_MATRICES[char])
        full += coeff * operator
    return full


def project_dense(pauli_strings, coeffs, states):
    """Return the Pauli sum projected onto the subspace spanned by ``states``, densely.

    Independent of ``rqutils.sqd``'s entire packing/padding/uniquification/matvec chain, so
    agreement with it is real evidence rather than a tautology. This mirrors
    ``examples/_bench_common.brute_force_reference``, which is gated against the sqd path in
    ``examples/check_bench_common.py`` and agreed to 3.6e-15; it is duplicated here rather than
    imported so the test suite does not depend on ``examples/`` (script territory, and those
    modules import qiskit and mlx).
    """
    num_qubits = states.shape[1]
    full = dense_pauli_sum(pauli_strings, coeffs)
    unique = np.unique(np.asarray(states, dtype=np.uint8), axis=0)
    # Row bits are most-significant-first, matching the Pauli string character order.
    indices = unique.dot(1 << np.arange(num_qubits)[::-1])
    return full[np.ix_(indices, indices)]


def lowest_projected(pauli_strings, coeffs, states):
    """Reference lowest eigenvalue of the projected Pauli sum. See :func:`project_dense`."""
    return float(np.linalg.eigvalsh(project_dense(pauli_strings, coeffs, states))[0].real)


def gate_unitary(name, qubits, num_qubits, angle=None):
    """Return the ``2**num_qubits`` unitary for one ``svsim`` gate spec.

    Built from Kronecker products of the 2x2 Paulis, independent of both ``rqutils.svsim`` and
    qiskit. ``qubits`` are indices in ``svsim``'s convention, where qubit ``q`` is bit ``q`` of the
    statevector index (bit 0 = least significant), i.e. the reverse of the Pauli-string character
    order used by :func:`dense_pauli_sum`.

    ``x``/``y``/``z`` are the bare Pauli gates. The rotations are ``exp(-i * angle * P / 2)``,
    matching Qiskit's ``rx``/``ry``/``rz``/``rzz``.
    """
    letters = {"x": "X", "y": "Y", "z": "Z", "rx": "X", "ry": "Y", "rz": "Z", "rzz": "Z"}
    if name not in letters:
        raise ValueError(f"unsupported gate {name}")
    qubits = np.atleast_1d(np.asarray(qubits))
    # Build the Pauli operator as a tensor product over all qubits, identity except on `qubits`.
    # Index bit q is qubit q, and np.kron's first factor is the MOST significant bit, so the
    # per-qubit factors go in reverse qubit order.
    factors = []
    for qubit in reversed(range(num_qubits)):
        factors.append(_PAULI_MATRICES[letters[name] if qubit in qubits else "I"])
    operator = factors[0]
    for factor in factors[1:]:
        operator = np.kron(operator, factor)
    if name in ("x", "y", "z"):
        return operator
    identity = np.eye(2**num_qubits, dtype=np.complex128)
    return np.cos(angle / 2.0) * identity - 1.0j * np.sin(angle / 2.0) * operator


def simulate_dense(gate_specs, num_qubits, initial_state=0):
    """Apply ``gate_specs`` to a statevector by dense matrix multiplication.

    The independent reference for :func:`rqutils.svsim.svsim`: it shares no code with the
    symplectic ``CircuitXZ`` representation or the ``lax.scan`` kernel under test.
    """
    state = np.zeros(2**num_qubits, dtype=np.complex128)
    if np.ndim(initial_state) == 0:
        state[int(initial_state)] = 1.0
    else:
        state = np.asarray(initial_state, dtype=np.complex128).copy()
    for spec in gate_specs:
        name, qubits, *rest = spec
        state = gate_unitary(name, qubits, num_qubits, *rest) @ state
    return state


def phaseless_distance(first, second):
    """Return the distance between two state vectors, minimized over a global phase.

    ``1 - |<a|b>| / (|a| |b|)`` -- zero exactly when the two agree up to a global phase. Used where
    a global phase is genuinely unobservable; assert exact equality instead wherever it is not.
    """
    first = np.asarray(first).ravel()
    second = np.asarray(second).ravel()
    norms = np.linalg.norm(first) * np.linalg.norm(second)
    if norms == 0.0:
        return 0.0 if np.linalg.norm(first) == np.linalg.norm(second) else 1.0
    return float(1.0 - abs(np.vdot(first, second)) / norms)


def real_pauli_strings(num_qubits, count, rng, letters="IXYZ"):
    """Return ``count`` distinct Pauli strings with an even number of Ys.

    ``PauliSumXZ`` narrows ``.c`` to float64 exactly when the folded ``(-i)^{x.z}`` phase is real,
    which holds precisely when each string has an even Y count. An odd-Y string leaves the
    coefficients complex128 -- correct, but not the real-arithmetic path these tests want.
    """
    strings, seen = [], set()
    while len(strings) < count:
        candidate = "".join(rng.choice(list(letters), size=num_qubits))
        if candidate.count("Y") % 2 or candidate in seen:
            continue
        seen.add(candidate)
        strings.append(candidate)
    return strings
