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


def real_pauli_strings(num_qubits, count, rng, letters="IXYZ"):
    """Return ``count`` distinct Pauli strings with an even number of Ys.

    ``PauliSumXZ.from_paulisum(..., force_real=True)`` requires real coefficients after the
    ``(-i)^{x.z}`` phase is folded in, which holds exactly when each string has an even Y count.
    An odd-Y string makes the coefficients complex and ``sqd`` raises rather than silently
    returning a wrong answer -- but these tests want the supported path.
    """
    strings, seen = [], set()
    while len(strings) < count:
        candidate = "".join(rng.choice(list(letters), size=num_qubits))
        if candidate.count("Y") % 2 or candidate in seen:
            continue
        seen.add(candidate)
        strings.append(candidate)
    return strings
