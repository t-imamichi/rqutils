"""Shared problem generation and timing for the scaling POCs.

Every script in this directory reaches this module through
``sys.path.insert(0, dirname(__file__))``, so they run as scripts rather than being imported. Names
here are unqualified by design: the directory already says ``scaling``.

Two things live here because getting either wrong silently invalidates a comparison.

**Problem generation** goes through the real ``PauliSumXZ.from_paulisum``, not a hand-built
symplectic array. The bit layout has a pad bit, a folded ``(-i)^{x.z}`` phase, and a
group-by-X-signature rectangle, and a POC that reconstructs any of that by hand is measuring its
own reconstruction. Generation is also parameterized by ``num_xgroups`` rather than by term count,
because ``J`` (distinct X signatures) is what the matvec cost is linear in, and a random Pauli list
gives ``J == num_terms`` almost surely -- which makes the ``K^{(j)}`` axis unmeasurable.

**Timing** reports the min of repeated trials and, separately, the spread. The spread is not
decoration: a **3.9%** noise floor was measured on this machine, and two runs of identical code once
looked like a valid comparison. Any difference under the measured spread is unresolved, and
``fmt_ratio`` refuses to call such a pair a win. (The 3.9% figure came from the since-deleted MLX
benchmark arm; treat it as the order of magnitude to expect, not a constant for these POCs, and rely
on the per-run spread that ``timeit`` actually reports.)
"""

import os
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
        states_p=PauliSumXZ.pack_states(states),
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
    is reported as UNRESOLVED rather than as a number -- the discipline adopted after two runs of
    identical code once looked like a valid comparison.
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


def init_devices(devices: str | None, host_devices: int = 4) -> str:
    """Bring up the JAX backend for a POC and return a one-line description of what it got.

    Three launch modes, because each node in the target cluster has a **single** GPU:

    - ``devices="mpi"`` -- multi-process. Calls ``jax.distributed.initialize`` so the ranks form one
      JAX program: each process owns its local GPU and ``jax.devices()`` reports all of them
      globally. **This is the only way to reach N GPUs when they sit on N nodes.**
    - ``devices="0,1"`` -- single-process, several local GPUs, via ``CUDA_VISIBLE_DEVICES``.
    - ``devices=None`` -- virtual CPU devices, correctness only.

    Without the ``initialize`` call, ``mpirun -n 4`` gives four **independent** clients that each see
    every GPU. That topology is multi-slice, and ``jax.make_mesh`` rejects it with "does not support
    multi-slice topologies. Please use jax.experimental.mesh_utils.create_hybrid_device_mesh" -- a
    message naming a replacement API rather than the missing initialization. Measured: four ranks
    each ran a complete prefilter sweep and then all four died at ``make_mesh``, so the failure
    arrives minutes in, four times interleaved, with gRPC "failed to connect" noise from the
    coordination service the ranks never joined.

    Must be called **before** the first ``jax.devices()``/``jax.device_count()``, since
    ``jax.distributed.initialize`` has to precede backend initialization.

    Args:
        devices: ``"mpi"``, a ``CUDA_VISIBLE_DEVICES`` list, or None for virtual CPU devices.
        host_devices: Virtual CPU device count when ``devices`` is None.

    Returns:
        A description such as ``"4 real gpu device(s) over 4 processes"``, for the script to print.

    Raises:
        RuntimeError: If ``devices="mpi"`` but ``mpi4py`` is missing, or if a launcher put several
            ranks in the job without ``devices="mpi"`` -- the case that produces the multi-slice
            error above.
    """
    ranks = next(
        (
            int(v)
            for v in (
                os.environ.get("OMPI_COMM_WORLD_SIZE"),
                os.environ.get("PMI_SIZE"),
                os.environ.get("SLURM_NTASKS"),
            )
            if v and v.isdigit()
        ),
        1,
    )

    if devices == "mpi":
        try:
            import mpi4py  # noqa: F401
        except ImportError as exc:
            # mpi4py is a required dependency now, so reaching this means the install is broken rather
            # than incomplete -- most likely it could not build against the host MPI.
            raise RuntimeError(
                "--devices mpi needs mpi4py, which is a required dependency, so this import failing "
                "means the install is broken -- usually a build against the host MPI. Try "
                "`uv sync --reinstall-package mpi4py` with an MPI compiler on PATH."
            ) from exc
        import jax

        jax.distributed.initialize(cluster_detection_method="mpi4py")
    else:
        if ranks > 1:
            raise RuntimeError(
                f"Launched with {ranks} ranks but --devices is {devices!r}, so each rank would come "
                "up as an independent JAX client seeing every GPU. That topology is multi-slice and "
                "jax.make_mesh rejects it with a message about create_hybrid_device_mesh that does "
                "not name this cause.\n\nFor one GPU per node, pass --devices mpi so the ranks "
                "form a single distributed program:\n"
                "    mpirun -n 4 uv run --extra qiskit python <script> --devices mpi"
            )
        if devices:
            os.environ["CUDA_VISIBLE_DEVICES"] = devices
        else:
            os.environ.setdefault(
                "XLA_FLAGS", f"--xla_force_host_platform_device_count={host_devices}"
            )

    import jax

    ndev, nproc = jax.device_count(), jax.process_count()
    backend = jax.devices()[0].platform
    kind = "virtual" if backend == "cpu" else "real"
    over = f" over {nproc} processes" if nproc > 1 else ""
    return f"{ndev} {kind} {backend} device(s){over}"


def make_1d_mesh(axis: str = "x", devices: "list | None" = None):
    """Build a 1-D mesh over every device, including across a multi-slice (multi-node) topology.

    ``jax.make_mesh`` **rejects multi-slice topologies outright**: it routes through
    ``mesh_utils.create_device_mesh`` and then raises "does not support multi-slice topologies" when
    the chosen devices carry more than one distinct ``slice_index``. One GPU per node means one slice
    per node, so a 4-node job hits that unconditionally -- and it does so *after*
    ``jax.distributed.initialize`` has correctly formed the cluster, which makes the failure look
    like an initialization problem rather than a mesh-construction one. Measured: 4 processes
    reporting ``process_count() == 4`` still failed here.

    ``create_device_mesh``'s topology optimization exists to order devices well for
    **multi-dimensional** meshes on a torus interconnect. For a 1-D mesh there is no ordering choice
    to make -- every device is one axis element -- so constructing ``Mesh`` directly over
    ``jax.devices()`` is equivalent and works irrespective of slice count.

    Args:
        axis: Mesh axis name.
        devices: Devices to include; defaults to all of ``jax.devices()``.

    Returns:
        A ``jax.sharding.Mesh`` with one ``AxisType.Explicit`` axis, matching what the POCs and
        ``rqutils``' implicit-mesh convention expect.
    """
    import jax
    import numpy as _np
    from jax.sharding import AxisType, Mesh

    devs = list(jax.devices()) if devices is None else list(devices)
    return Mesh(_np.asarray(devs), (axis,), axis_types=(AxisType.Explicit,))


def header(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)
