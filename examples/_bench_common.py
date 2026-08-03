"""Shared problem generation, setup, and timing for the JAX-vs-MLX SQD benchmarks.

The setup stage (uniquification, X-source lookup, diagonal composition) always runs in JAX on
CPU and is deliberately *not* timed: every benchmark arm consumes the identical arrays this
module produces, so timing differences are attributable to the solver alone.

This module must not import mlx at module level -- the JAX-only arms run in processes where
mlx may be unavailable.
"""
from dataclasses import dataclass
import time
from collections.abc import Callable
from typing import Any
import numpy as np
import jax
import jax.numpy as jnp
from rqutils.paulis.symplectic import PauliSumXZ
from rqutils.sqd import uniquify_states, get_xsource, get_diagonal

_PAULI_MATRICES = {
    'I': np.eye(2, dtype=np.complex128),
    'X': np.array([[0., 1.], [1., 0.]], dtype=np.complex128),
    'Y': np.array([[0., -1.j], [1.j, 0.]], dtype=np.complex128),
    'Z': np.diag([1., -1.]).astype(np.complex128)
}


@dataclass
class SolverInputs:
    """Everything the solver loop needs, with invalid X sources already neutralized."""
    xsources: np.ndarray
    diagonals: np.ndarray
    vinit: np.ndarray
    subspace_dim: int
    num_xgroups: int


def generate_problem(
    num_qubits: int,
    num_paulis: int,
    num_states: int,
    seed: int = 0
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Generate a random Hamiltonian with real coefficients, plus a subspace of states.

    Every Pauli string carries an even number of Ys. This makes the phased coefficients
    ``alpha * (-i)^{x.z}`` real, which matters because MLX has no complex128: a general
    (odd-Y) Hamiltonian cannot be represented in MLX at any precision.
    """
    rng = np.random.default_rng(seed)
    pauli_strings = []
    while len(pauli_strings) < num_paulis:
        chars = rng.choice(list('IXYZ'), size=num_qubits)
        if np.count_nonzero(chars == 'Y') % 2:
            continue
        pauli_strings.append(''.join(chars))

    coeffs = rng.uniform(-1., 1., num_paulis)
    states = rng.choice(2, size=(num_states, num_qubits)).astype(np.uint8)
    return pauli_strings, coeffs, states


def build_solver_inputs(
    pauli_strings: list[str],
    coeffs: np.ndarray,
    states: np.ndarray
) -> SolverInputs:
    """Run the JAX setup stage and return sanitized solver inputs.

    Sanitization: JAX's ``.at[].get(mode='fill', fill_value=0.)`` maps ``xsource == -1``
    (no source state in the subspace) to zero. MLX's ``take`` has no fill mode and its
    out-of-bounds behavior is undocumented, so instead of relying on it we clamp the index
    to 0 and zero the matching diagonal. The gathered value is then arbitrary but multiplied
    by 0., which is algebraically identical and costs nothing inside the solver loop.
    Both frameworks receive these same arrays, so neither is advantaged.
    """
    hamiltonian = PauliSumXZ.from_paulisum((pauli_strings, coeffs), force_real=True,
                                           add_padding=True)
    if hamiltonian.c.dtype != np.float64:
        raise ValueError(f'Hamiltonian coefficients are {hamiltonian.c.dtype}, expected float64.'
                         ' Pauli strings must have an even number of Ys.')

    states_p = np.packbits(np.pad(states.astype(np.uint8), {1: (1, 0)}), axis=1)
    subspace_dim = int(np.unique(states_p, axis=0).shape[0])
    states_u = uniquify_states(states_p, subspace_dim)

    xsources = np.stack([np.asarray(get_xsource(x, states_u)) for x in hamiltonian.x])
    diagonals = np.stack([np.asarray(get_diagonal(z, c, states_u))
                          for z, c in zip(hamiltonian.z, hamiltonian.c)])

    valid = xsources >= 0
    xsources = np.where(valid, xsources, 0).astype(np.int32)
    diagonals = np.where(valid, diagonals, 0.).real.astype(np.float64)

    # Mirror sqd's vinit_from_min_diag: one-hot at the minimum diagonal entry.
    if np.all(hamiltonian.x[0] == 0):
        start = int(np.argmin(diagonals[0]))
    else:
        start = 0
    vinit = np.zeros(subspace_dim, dtype=np.float64)
    vinit[start] = 1.

    return SolverInputs(xsources, diagonals, vinit, subspace_dim, xsources.shape[0])


def dense_reference(inputs: SolverInputs) -> tuple[np.ndarray, float]:
    """Build the projected Hamiltonian densely from the solver inputs and diagonalize it.

    Used instead of ``rqutils.sqd.hproj``, which raises a shape-mismatch TypeError: it builds
    the Hamiltonian with add_padding=True but packs the states without the pad bit.
    """
    dim = inputs.subspace_dim
    matrix = np.zeros((dim, dim), dtype=np.float64)
    rows = np.arange(dim)
    for xsource, diagonal in zip(inputs.xsources, inputs.diagonals):
        np.add.at(matrix, (rows, xsource), diagonal)
    return matrix, float(np.linalg.eigvalsh(matrix)[0])


def brute_force_reference(
    pauli_strings: list[str],
    coeffs: np.ndarray,
    states: np.ndarray
) -> float:
    """Ground energy via the full 2^n matrix, projected onto the unique states.

    Independent of the whole packing/padding/uniquification chain, so agreement with
    dense_reference validates that chain. Costs ~0.13 s at n=10; do not call it for large n.
    """
    num_qubits = states.shape[1]
    full = np.zeros((2 ** num_qubits,) * 2, dtype=np.complex128)
    for string, coeff in zip(pauli_strings, coeffs):
        operator = np.array([[1.]], dtype=np.complex128)
        for char in string:
            operator = np.kron(operator, _PAULI_MATRICES[char])
        full += coeff * operator

    unique = np.unique(states, axis=0)
    # Row bits are most-significant-first, matching the Pauli string character order.
    indices = unique.dot(1 << np.arange(num_qubits)[::-1])
    projected = full[np.ix_(indices, indices)]
    return float(np.linalg.eigvalsh(projected)[0].real)


def timeit(
    fn: Callable[[], Any],
    repeat: int,
    sync: Callable[[Any], Any]
) -> tuple[float, float]:
    """Return (compile_seconds, mean_steady_seconds).

    ``sync`` must force the computation to complete: ``jax.block_until_ready`` for JAX,
    ``mx.eval`` for MLX. Without it both frameworks return before the work is done -- MLX is
    lazy and JAX is async -- and the measurement is meaningless.
    """
    start = time.perf_counter()
    sync(fn())
    compile_time = time.perf_counter() - start

    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        sync(fn())
        times.append(time.perf_counter() - start)
    return compile_time, float(np.mean(times))
