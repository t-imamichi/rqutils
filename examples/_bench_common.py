"""Shared problem generation, setup, and timing for the JAX-vs-MLX SQD benchmarks.

The setup stage (uniquification, X-source lookup, diagonal composition) always runs in JAX on
CPU and is deliberately *not* timed: every benchmark arm consumes the identical arrays this
module produces, so timing differences are attributable to the solver alone.

This module must not import mlx at module level -- the JAX-only arms run in processes where
mlx may be unavailable.
"""

import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np

from rqutils.paulis.symplectic import PauliSumXZ
from rqutils.sqd import get_diagonal, get_xsource, uniquify_states

_PAULI_MATRICES = {
    "I": np.eye(2, dtype=np.complex128),
    "X": np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128),
    "Y": np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128),
    "Z": np.diag([1.0, -1.0]).astype(np.complex128),
}

# Above this subspace dimension, dense_reference's (N, N) float64 matrix stops being
# allocatable: 200 MB at N=5000, 7.2 GB at N=30000, 80 GB at N=1e5. The sparse path takes over
# there. Chosen above the bench default --num-states 4000 so existing invocations keep using
# the dense path and stay bit-for-bit reproducible.
DENSE_REFERENCE_MAX_DIM = 5000


@dataclass
class SolverInputs:
    """Everything the solver loop needs, with invalid X sources already neutralized."""

    xsources: np.ndarray
    diagonals: np.ndarray
    vinit: np.ndarray
    subspace_dim: int
    num_xgroups: int


def generate_problem(
    num_qubits: int, num_paulis: int, num_states: int, seed: int = 0
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Generate a random Hamiltonian with real coefficients, plus a subspace of states.

    Every Pauli string carries an even number of Ys. This makes the phased coefficients
    ``alpha * (-i)^{x.z}`` real, which matters because MLX has no complex128: a general
    (odd-Y) Hamiltonian cannot be represented in MLX at any precision.
    """
    rng = np.random.default_rng(seed)
    pauli_strings = []
    while len(pauli_strings) < num_paulis:
        chars = rng.choice(list("IXYZ"), size=num_qubits)
        if np.count_nonzero(chars == "Y") % 2:
            continue
        pauli_strings.append("".join(chars))

    coeffs = rng.uniform(-1.0, 1.0, num_paulis)
    states = rng.choice(2, size=(num_states, num_qubits)).astype(np.uint8)
    return pauli_strings, coeffs, states


def build_solver_inputs(
    pauli_strings: list[str], coeffs: np.ndarray, states: np.ndarray
) -> SolverInputs:
    """Run the JAX setup stage and return sanitized solver inputs.

    Sanitization: JAX's ``.at[].get(mode='fill', fill_value=0.)`` maps ``xsource == -1``
    (no source state in the subspace) to zero. MLX's ``take`` has no fill mode and its
    out-of-bounds behavior is undocumented, so instead of relying on it we clamp the index
    to 0 and zero the matching diagonal. The gathered value is then arbitrary but multiplied
    by 0., which is algebraically identical and costs nothing inside the solver loop.
    Both frameworks receive these same arrays, so neither is advantaged.
    """
    hamiltonian = PauliSumXZ.from_paulisum((pauli_strings, coeffs), force_real=True)
    if hamiltonian.c.dtype != np.float64:
        raise ValueError(
            f"Hamiltonian coefficients are {hamiltonian.c.dtype}, expected float64."
            " Pauli strings must have an even number of Ys."
        )

    states_p = np.packbits(np.pad(states.astype(np.uint8), {1: (1, 0)}), axis=1)
    subspace_dim = int(np.unique(states_p, axis=0).shape[0])
    states_u = uniquify_states(states_p, subspace_dim)

    xsources = np.stack([np.asarray(get_xsource(x, states_u)) for x in hamiltonian.x])
    diagonals = np.stack(
        [np.asarray(get_diagonal(z, c, states_u)) for z, c in zip(hamiltonian.z, hamiltonian.c)]
    )

    valid = xsources >= 0
    xsources = np.where(valid, xsources, 0).astype(np.int32)
    diagonals = np.where(valid, diagonals, 0.0).astype(np.float64)

    # Mirror sqd's vinit_from_min_diag: one-hot at the minimum diagonal entry.
    if np.all(hamiltonian.x[0] == 0):
        start = int(np.argmin(diagonals[0]))
    else:
        start = 0
    vinit = np.zeros(subspace_dim, dtype=np.float64)
    vinit[start] = 1.0

    return SolverInputs(xsources, diagonals, vinit, subspace_dim, xsources.shape[0])


def apply_h_xz_chunked(
    vec: jnp.ndarray, xsources: jnp.ndarray, diagonals: jnp.ndarray, chunk: int = 16
) -> jnp.ndarray:
    """Return Hv via chunked batched gather, the JAX counterpart of ``apply_h_xz_mlx_chunked``.

    ``rqutils.sqd.apply_h_xz_cached`` (and ``ground_locg_mlx.apply_h_xz_mlx``) loop over the J
    X-groups doing one take+multiply+add per group -- 3J elementary ops per matvec. Since
    ``xsources``/``diagonals`` are dense ``(J, N)`` arrays, groups can instead be processed in
    chunks: gather a whole chunk with one flat ``take``, reshape, and reduce with a single
    weighted sum, cutting the op count from ``3*J`` to roughly ``3*ceil(J/chunk)``.

    ``chunk`` bounds the size of the temporary gathered block to ``chunk * N`` elements rather
    than the full ``J * N`` a fully-batched (``chunk=J``) version would materialize. At the large
    N this benchmark is ultimately meant to probe (SQD's matrix-free design targets N up to
    ~10**7), a full ``(J, N)`` gather would cost ~8 GB versus ~80 MB for the unchunked
    group-at-a-time loop -- exactly the memory blowup the matrix-free loop exists to avoid.
    Chunking keeps the op-count win while capping the temporary at ``chunk * N``. This is why
    the implementation deliberately chunks rather than fully batching, even though full batching
    (``chunk=J``) would look "simpler."

    Op-count / temporary-size tradeoff measured at J=100 (see the design doc's Results section
    for the corresponding per-iteration timings):

    ========  ========================  =================
    chunk     ops per matvec (3*ceil)   temporary (chunk*N)
    ========  ========================  =================
    1 (loop)  300                       N        (baseline)
    8         39   (7.7x fewer)         8*N
    16        21   (14.3x fewer)        16*N   <- default
    32        12   (25x fewer)          32*N
    ========  ========================  =================

    This same function is used by the JAX arms of the benchmark (see ``--matvec chunked`` in
    ``bench_mlx.py``): batching the gather is an algorithmic improvement independent of the
    framework, so restricting it to the MLX arm would confound "MLX got faster" with "the matvec
    got better." Applying it symmetrically keeps the JAX-vs-MLX comparison about the solver loop,
    not about who has the better matvec.

    Verified (design doc, Optimization 1): max abs diff vs the unchunked loop matvec is
    <= 2.7e-15 for chunk in {1, 4, 8, 16, 32, 50, 100, 128}, and matches ``H @ v`` to 1.8e-15.

    Args:
        vec: The vector to multiply, shape ``(N,)``.
        xsources: Sanitized X-source indices, shape ``(J, N)``, dtype int32.
        diagonals: Sanitized diagonals, shape ``(J, N)``, matching ``vec``'s dtype.
        chunk: Number of X-groups to gather per flat ``take``. Static (Python int) -- it
            controls how many trace-time loop iterations ``jax.jit`` unrolls, exactly like the
            existing ``xsources.shape[0]`` group loop in ``apply_h_xz_cached``.

    Returns:
        ``H @ vec``, algebraically identical to ``apply_h_xz_cached``.
    """
    num_groups = xsources.shape[0]
    out = jnp.zeros_like(vec)
    for start in range(0, num_groups, chunk):
        xc = xsources[start : start + chunk]
        dc = diagonals[start : start + chunk]
        gathered = jnp.take(vec, xc.reshape(-1)).reshape(xc.shape)
        out = out + jnp.sum(gathered * dc, axis=0)
    return out


def dense_reference(inputs: SolverInputs) -> tuple[np.ndarray, float]:
    """Build the projected Hamiltonian densely from the solver inputs and diagonalize it.

    Deliberately independent of ``rqutils.sqd.hproj`` rather than a wrapper around it: a benchmark
    gate must not be a second run of the code under test. Past bugs in this package had every
    internal path agreeing on the same wrong number, so self-consistency proves nothing here.

    (This once carried a note that ``hproj`` was unusable because it raised a shape-mismatch
    TypeError -- it built the Hamiltonian with the signature pad bit but packed the states without
    it, back when that padding was an opt-in flag. That bug and a missing ``shape=`` on its
    ``coo_array`` are both fixed; the reason to keep this separate is independence, not breakage.)
    """
    dim = inputs.subspace_dim
    matrix = np.zeros((dim, dim), dtype=np.float64)
    rows = np.arange(dim)
    for xsource, diagonal in zip(inputs.xsources, inputs.diagonals):
        np.add.at(matrix, (rows, xsource), diagonal)
    asym = np.abs(matrix - matrix.T).max()
    assert asym == 0.0, (
        f"gate failed: dense reference is not symmetric (|H - H.T|_inf = {asym}); eigvalsh "
        "would silently ignore the upper triangle and return a wrong reference. This should be "
        "impossible for even-Y (real) input -- see _bench_common.build_solver_inputs."
    )
    return matrix, float(np.linalg.eigvalsh(matrix)[0])


def sparse_reference(inputs: SolverInputs) -> float:
    """Ground energy of the projected Hamiltonian via sparse eigsh, for large subspaces.

    ``dense_reference`` allocates an (N, N) float64 matrix, which is 80 GB at N=1e5. The
    operator is sparse by construction -- ``np.add.at`` over J X-groups, so at most J*N
    nonzeros -- so the same eigenvalue is reachable without ever densifying it.

    An independent algorithm in an independent library, not a second run of the solver under
    test: CLAUDE.md records bugs where every internal code path agreed on the same wrong number,
    so self-consistency is not an acceptable reference.

    Raises:
        SystemExit: If eigsh fails to converge or returns a non-finite value. An unconverged
            reference must fail the gate loudly -- a NaN eigenvalue once passed this benchmark's
            gate silently, because every IEEE 754 NaN comparison is false.
    """
    import scipy.sparse
    import scipy.sparse.linalg

    dim = inputs.subspace_dim
    rows = np.tile(np.arange(dim), inputs.xsources.shape[0])
    cols = inputs.xsources.reshape(-1)
    data = inputs.diagonals.reshape(-1)
    # coo_matrix sums duplicate (row, col) entries, matching dense_reference's np.add.at.
    matrix = scipy.sparse.coo_matrix((data, (rows, cols)), shape=(dim, dim)).tocsr()

    try:
        # 'SA' = smallest algebraic. Not 'SM' (smallest magnitude), which would return the
        # eigenvalue nearest zero rather than the ground state.
        eigvals = scipy.sparse.linalg.eigsh(
            matrix, k=1, which="SA", return_eigenvectors=False, tol=0.0
        )
    except scipy.sparse.linalg.ArpackNoConvergence as exc:
        raise SystemExit(
            f"gate failed: sparse_reference eigsh did not converge at N={dim} ({exc}); the "
            "reference eigenvalue is unusable, so no timing row may be reported"
        ) from exc
    value = float(eigvals[0])
    if not np.isfinite(value):
        raise SystemExit(
            f"gate failed: sparse_reference returned non-finite {value} at N={dim}. Every NaN "
            "comparison is false, so this must be rejected explicitly rather than compared"
        )
    return value


def brute_force_reference(
    pauli_strings: list[str], coeffs: np.ndarray, states: np.ndarray
) -> float:
    """Ground energy via the full 2^n matrix, projected onto the unique states.

    Independent of the whole packing/padding/uniquification chain, so agreement with
    dense_reference validates that chain. Costs ~0.13 s at n=10; do not call it for large n.
    """
    num_qubits = states.shape[1]
    full = np.zeros((2**num_qubits,) * 2, dtype=np.complex128)
    for string, coeff in zip(pauli_strings, coeffs):
        operator = np.array([[1.0]], dtype=np.complex128)
        for char in string:
            operator = np.kron(operator, _PAULI_MATRICES[char])
        full += coeff * operator

    unique = np.unique(states, axis=0)
    # Row bits are most-significant-first, matching the Pauli string character order.
    indices = unique.dot(1 << np.arange(num_qubits)[::-1])
    projected = full[np.ix_(indices, indices)]
    return float(np.linalg.eigvalsh(projected)[0].real)


# A steady-state spread wider than this fraction of the minimum means the machine was doing
# something else while we measured, so the number should not be compared against another run.
_SPREAD_WARN = 0.25

# ...but only once the work is long enough for that ratio to mean anything. Below this, timer
# granularity and scheduler jitter dominate: a 0.4 ms sample next to a 0.2 ms one is a 100%
# "spread" that says nothing about machine load. The benchmark's own per-iteration figures are
# milliseconds and its fixed_s values tens of milliseconds, so this floor only silences the
# sub-millisecond smoke-test calls (e.g. check_bench_common.py's) that were never comparable
# across runs anyway.
_SPREAD_WARN_MIN_SECONDS = 5e-3


def timeit(fn: Callable[[], Any], repeat: int, sync: Callable[[Any], Any]) -> tuple[float, float]:
    """Return (first_call_seconds, best_steady_seconds).

    ``sync`` must force the computation to complete: ``jax.block_until_ready`` for JAX,
    ``mx.eval`` for MLX. Without it both frameworks return before the work is done -- MLX is
    lazy and JAX is async -- and the measurement is meaningless.

    The steady-state figure is the MINIMUM of ``repeat`` samples, not the mean. Noise is
    one-sided -- the machine can steal time from a sample but never give it back -- so the
    minimum is the least-contaminated estimate of the work's actual cost, and it is what makes
    two runs comparable. If the samples spread by more than 25% of the minimum, a warning goes to
    stderr: that run was disturbed and its timings should not be quoted against another.
    (Previously this returned the mean, so figures recorded before this change are very slightly
    higher and are not exactly comparable.)

    THE FIRST VALUE IS A SINGLE SAMPLE AND CANNOT BE REPEATED. Compilation happens once per
    process, so there is no way to average it. It therefore includes not just tracing/compilation
    but whatever else the machine was doing at that instant -- shader-cache population, thermal
    state, contention from sibling benchmark subprocesses. It has been observed varying 53x
    (20.39 s vs 0.38 s) between two runs of the *same* command whose steady-state numbers agreed
    to 3% and whose results were bit-identical. Treat it as indicative of magnitude only, never
    as a measurement to compare across runs or frameworks.
    """
    start = time.perf_counter()
    sync(fn())
    first_call_time = time.perf_counter() - start

    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        sync(fn())
        times.append(time.perf_counter() - start)

    best = float(np.min(times))
    if len(times) > 1 and best >= _SPREAD_WARN_MIN_SECONDS:
        spread = (float(np.max(times)) - best) / best
        if spread > _SPREAD_WARN:
            sys.stderr.write(
                f"WARNING: steady-state timing spread {spread:.0%} over {len(times)} samples "
                f"(min {best * 1e3:.3f} ms, max {max(times) * 1e3:.3f} ms). The machine was busy; "
                "do not compare this run's timings against another.\n"
            )
    return first_call_time, best
