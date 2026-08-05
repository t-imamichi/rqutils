"""Shared problem generation and timing for the scaling POCs.

Every script in this directory reaches this module through
``sys.path.insert(0, dirname(__file__))``, the same pattern ``examples/mlx/`` uses for
``_bench_common.py``. Names here are unqualified by design: the directory already says
``scaling``.

Two things live here because getting either wrong silently invalidates a comparison.

**Problem generation** goes through the real ``PauliSumXZ.from_paulisum``, not a hand-built
symplectic array. The bit layout has a pad bit, a folded ``(-i)^{x.z}`` phase, and a
group-by-X-signature rectangle, and a POC that reconstructs any of that by hand is measuring its
own reconstruction. Generation is also parameterized by ``num_xgroups`` rather than by term count,
because ``J`` (distinct X signatures) is what the matvec cost is linear in, and a random Pauli list
gives ``J == num_terms`` almost surely -- which makes the ``K^{(j)}`` axis unmeasurable.

**Timing** reports the min of repeated trials and, separately, the spread. The spread is not
decoration: ``CLAUDE.md`` records a 3.9% noise floor on this machine for the MLX arm and a case
where two runs of identical code looked like a valid comparison. Any difference under the measured
spread is unresolved, and ``fmt_ratio`` refuses to call such a pair a win.
"""

import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from qiskit.quantum_info import SparsePauliOp

from rqutils.paulis.symplectic import PauliSumXZ


@dataclass
class Problem:
    """A generated SQD problem, plus the derived arrays the POCs need."""

    hamiltonian: PauliSumXZ
    states: np.ndarray  # (N, n) uint8, unpacked, NOT uniquified
    states_p: np.ndarray  # (N, B) uint8, packed with the leading pad bit
    num_qubits: int
    num_states: int

    @property
    def num_xgroups(self) -> int:
        """J -- the matvec cost is linear in this."""
        return self.hamiltonian.x.shape[0]

    @property
    def max_zterms(self) -> int:
        """max_j K^(j) -- the width of the zero-padded Z rectangle."""
        return self.hamiltonian.z.shape[1]

    @property
    def is_real(self) -> bool:
        """Whether the folded phase left the coefficients real (all-even-Y)."""
        return not np.iscomplexobj(self.hamiltonian.c)

    def describe(self) -> str:
        return (
            f"n={self.num_qubits} N={self.num_states} J={self.num_xgroups} "
            f"maxK={self.max_zterms} dtype={self.hamiltonian.c.dtype}"
        )


def pack_states(states: np.ndarray) -> np.ndarray:
    """Pack states with the leading pad bit, exactly as ``sqd`` and ``hproj`` do.

    Duplicated from ``sqd`` deliberately: a POC that imports a private helper would break when the
    library moves, and this one line is the alignment contract every consumer restates.
    """
    return np.packbits(np.pad(states.astype(np.uint8), {1: (1, 0)}), axis=1)


def make_problem(
    num_qubits: int,
    num_states: int,
    num_terms: int = 100,
    num_xgroups: int | None = None,
    real_only: bool = False,
    seed: int = 0,
) -> Problem:
    """Generate a random SQD problem with independent control over N, J, and realness.

    Args:
        num_qubits: Qubit count ``n``.
        num_states: Subspace sample count ``N`` before uniquification. Sampled with replacement
            from ``2**num_qubits``, so for small ``n`` the uniquified count is lower; the POCs that
            care report the post-uniquification size.
        num_terms: Number of Pauli strings in the Hamiltonian.
        num_xgroups: If given, force the terms into this many distinct X signatures by drawing
            ``num_xgroups`` X patterns and reusing them. This is the knob that separates the ``J``
            axis from the ``K`` axis; without it a random list gives ``J == num_terms``.
        real_only: Build every string with an even number of Ys, so the folded phase stays real and
            ``.c`` narrows to float64. This is the only way to get the real-symmetric regime, and
            it cannot be requested after the fact -- see ``PauliSumXZ``'s note on why no flag can
            grant realness.
        seed: PRNG seed. Fixed by default so a comparison is reproducible.
    """
    rng = np.random.default_rng(seed)

    if num_xgroups is None:
        # Each term gets its own X pattern; J == num_terms almost surely.
        xbits = rng.choice(2, size=(num_terms, num_qubits)).astype(bool)
    else:
        # Draw J distinct X patterns, then assign terms to them round-robin so every group is
        # populated even when num_terms is close to num_xgroups.
        base = rng.choice(2, size=(num_xgroups, num_qubits)).astype(bool)
        base = np.unique(base, axis=0)
        xbits = base[np.arange(num_terms) % base.shape[0]]

    zbits = rng.choice(2, size=(num_terms, num_qubits)).astype(bool)

    if real_only:
        # A Y sits where x and z are both set. Realness needs popcount(x & z) even for every
        # string, so clear the lowest set bit of the overlap on the odd rows. Clearing rather than
        # setting keeps the X signature untouched, which matters because num_xgroups is a
        # controlled variable here.
        overlap = xbits & zbits
        odd = (overlap.sum(axis=1) & 1).astype(bool)
        for irow in np.nonzero(odd)[0]:
            first = np.nonzero(overlap[irow])[0][0]
            zbits[irow, first] = False

    labels = []
    for xrow, zrow in zip(xbits, zbits):
        # "IXYZ" indexed by (x + 2z): I=0, X=1, Y=3, Z=2 -> reorder to match.
        chars = []
        for xb, zb in zip(xrow, zrow):
            chars.append({(0, 0): "I", (1, 0): "X", (0, 1): "Z", (1, 1): "Y"}[(int(xb), int(zb))])
        labels.append("".join(chars))

    coeffs = rng.uniform(-1.0, 1.0, size=num_terms)
    hamiltonian = PauliSumXZ.from_paulisum(SparsePauliOp(labels, coeffs))

    if real_only and np.iscomplexobj(hamiltonian.c):
        raise RuntimeError(
            "real_only=True still produced complex coefficients -- the even-Y construction is "
            "broken, and every downstream 'real path' number would be measuring the complex path."
        )

    states = rng.choice(2, size=(num_states, num_qubits)).astype(np.uint8)
    return Problem(
        hamiltonian=hamiltonian,
        states=states,
        states_p=pack_states(states),
        num_qubits=num_qubits,
        num_states=num_states,
    )


@dataclass
class Timing:
    """Result of a timed measurement."""

    label: str
    min_s: float
    median_s: float
    spread_frac: float  # (max - min) / min
    trials: int

    def __str__(self) -> str:
        return (
            f"{self.label:<40s} min={self.min_s * 1e3:9.3f} ms  "
            f"med={self.median_s * 1e3:9.3f} ms  spread={self.spread_frac * 100:5.1f}%"
        )


def timeit(
    fn: Callable[[], object],
    label: str = "",
    trials: int = 5,
    warmup: int = 1,
    block: bool = True,
) -> Timing:
    """Time ``fn``, reporting min and spread.

    ``warmup`` calls are discarded, which for JAX means the trace-and-compile cost is not folded
    into the measurement. ``block`` forces completion of the returned arrays: JAX dispatches
    asynchronously, so without it a "measurement" can time the dispatch and nothing else.
    """
    import jax

    for _ in range(warmup):
        out = fn()
        if block:
            jax.block_until_ready(out)

    samples = []
    for _ in range(trials):
        start = time.perf_counter()
        out = fn()
        if block:
            jax.block_until_ready(out)
        samples.append(time.perf_counter() - start)

    lo = min(samples)
    return Timing(
        label=label or getattr(fn, "__name__", "fn"),
        min_s=lo,
        median_s=statistics.median(samples),
        spread_frac=(max(samples) - lo) / lo if lo > 0 else 0.0,
        trials=trials,
    )


def fmt_ratio(baseline: Timing, candidate: Timing, noise_floor: float | None = None) -> str:
    """Describe candidate-vs-baseline, refusing to claim a win inside the noise.

    ``noise_floor`` defaults to the larger of the two measured spreads. A speedup smaller than that
    is reported as UNRESOLVED rather than as a number, which is the discipline ``CLAUDE.md`` records
    for the MLX arms after two runs of identical code looked like a valid comparison.
    """
    if noise_floor is None:
        noise_floor = max(baseline.spread_frac, candidate.spread_frac)
    ratio = baseline.min_s / candidate.min_s
    delta = abs(ratio - 1.0)
    verdict = "UNRESOLVED (within noise)" if delta < noise_floor else f"{ratio:.2f}x"
    direction = "faster" if ratio > 1.0 else "SLOWER"
    if verdict.startswith("UNRESOLVED"):
        return f"{ratio:.3f}x -> {verdict}; noise floor {noise_floor * 100:.1f}%"
    return f"{verdict} {direction}; noise floor {noise_floor * 100:.1f}%"


def max_abs_diff(a, b) -> float:
    """Max absolute difference, for correctness gates."""
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


def header(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)
