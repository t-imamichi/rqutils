"""POC 15: ``sqd`` at scale on a real multi-node GPU mesh -- memory and speed, not correctness.

``poc7_sharding.py`` already settles multi-node **correctness**: it passed on 4 GPUs across 4 nodes at
all six ``cache_level`` cells, every ``N mod mesh.size``, and the ``return_eigvec`` round trip, worst
``|sharded - single|`` = 4.441e-16. It does so at ``n <= 18``, ``N <= 5000`` -- fixtures small enough
that every arm fits on one device, which is what makes them safe to compare against a single-device
reference. This script exists for the two questions that *cannot* be asked there:

**Claim 1: does per-device memory actually fall as devices are added?** ``CLAUDE.md`` records the
cost model -- ``states`` is replicated today at ``13 * N`` bytes per device, the one term the ``(0,0)``
floor cannot shed, while the solver's ``O(N)`` vectors and the diagonal cache do shard. Those are
*predictions from a formula*, and per ``CLAUDE.md`` memory should be asked of XLA rather than derived:
byte-count formulas have predicted a saving where the measured peak **rose**. Here the mesh is swept
over ``1, 2, 4, ...`` devices at fixed ``N`` and ``bytes_in_use`` is read per device, so the answer is
a measured curve. A replicated term shows as a flat component; a sharded one halves per doubling.

**Claim 2: is the sharded solve faster, and where does it stop being faster?** ``poc7``'s docstring is
explicit that virtual devices "cannot speak to interconnect cost, per-device memory limits, or whether
the sharded solve is actually *faster*". On a **multi-node** mesh the collectives cross a network
rather than NVLink, so this is the pessimistic topology -- and the honest place to find the crossover
below which communication dominates. ``fmt_ratio`` refuses to call a difference inside the measured
spread a win, so a result under the noise floor is reported as unresolved.

**What this script does not claim.** It is not a correctness harness -- it asserts the energy against
the 1-device run so a broken arm cannot post a good time (``CLAUDE.md``: a broken arm flatters its own
benchmark), but the systematic correctness sweep is ``poc7``'s and is not repeated. Both arms are run
warm, per-call, whole-solve, with the arrays passed as arguments rather than closed over.

The fixture is the 1D XXZ Krylov subspace, not ``rng.choice`` rows: a random subspace is 3.6-6.1%
dense against 32-44% for a physical one, and ``CLAUDE.md`` records that a physically-motivated fixture
has *inverted* a conclusion a synthetic one reached. The Hamiltonian is the XXZ chain whose ground
state that subspace is built to span, so the solve is the one a real SKQD workflow performs.

Run on a multi-node cluster with one GPU per node (the only way to reach N GPUs on N nodes)::

    mpirun -n 4 uv run --extra mpi python examples/scaling/poc15_sqd_multinode.py --devices mpi

Run on one node holding several GPUs::

    uv run --extra qiskit python examples/scaling/poc15_sqd_multinode.py --devices 0,1,2,3

Correctness-only rehearsal on virtual CPU devices (timings are meaningless there and are suppressed)::

    uv run --extra qiskit python examples/scaling/poc15_sqd_multinode.py --host-devices 4
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# `argparse` before `import jax`, as in poc8/poc9/poc13/poc14: CUDA_VISIBLE_DEVICES, XLA_FLAGS and
# jax.distributed.initialize are all read at backend initialization, so none can be set afterwards.
parser = argparse.ArgumentParser()
parser.add_argument(
    "--devices",
    help='Comma-separated GPU ids, e.g. "0,1,2,3", or "mpi" for one GPU per MPI rank.',
)
parser.add_argument(
    "--host-devices", type=int, default=4, help="Virtual CPU devices when no --devices."
)
parser.add_argument("--num-qubits", type=int, default=26, help="Chain length n.")
parser.add_argument(
    "--max-states", type=int, default=400_000, help="Cap on the Krylov subspace size N."
)
parser.add_argument("--jz", type=float, default=0.8, help="XXZ anisotropy.")
parser.add_argument(
    "--cache-level",
    default="1,0",
    help="cache_level as 'a,b'. Default (1,0) is sqd's own default.",
)
options = parser.parse_args()

import jax

# Mandatory, and its absence is silent: without x64 every array narrows to float32/int32 and the
# energies drift in the 7th digit. Measured while writing this script -- the 1-device result moved
# from -23.782182463507 to -23.782182693481 and the cross-device assertion below fired at 3.8e-06,
# reading exactly like a sharding bug. CLAUDE.md states the rule; this is what breaking it looks like.
jax.config.update("jax_enable_x64", True)

import numpy as np
from _scaling_common import fmt_ratio, header, init_devices, make_1d_mesh, timeit
from qiskit.quantum_info import SparsePauliOp

from rqutils.paulis.symplectic import PauliSumXZ
from rqutils.sqd import sqd

CACHE_LEVEL = tuple(int(x) for x in options.cache_level.split(","))


def xxz_hamiltonian(num_qubits: float, jz: float) -> PauliSumXZ:
    """The periodic 1D XXZ chain: ``sum_i XX + YY + Jz * ZZ`` over nearest neighbours.

    Built through ``PauliSumXZ.from_paulisum`` rather than by assembling the symplectic arrays by
    hand, for the reason ``_scaling_common`` gives: the bit layout has a pad bit, a folded
    ``(-i)^{x.z}`` phase and a group-by-X-signature rectangle, so a POC that reconstructs any of it
    is measuring its own reconstruction.

    Every term has an even Y count (``YY`` has two, ``XX``/``ZZ`` none), so the folded phase stays
    real and ``.c`` narrows to float64 -- the real-symmetric regime. Asserted, not assumed.
    """
    n = int(num_qubits)
    labels, coeffs = [], []
    for i in range(n):
        j = (i + 1) % n
        for pauli, coeff in (("X", 1.0), ("Y", 1.0), ("Z", jz)):
            row = ["I"] * n
            row[i] = row[j] = pauli
            labels.append("".join(row))
            coeffs.append(coeff)
    hamiltonian = PauliSumXZ.from_paulisum(SparsePauliOp(labels, coeffs))
    if np.iscomplexobj(hamiltonian.c):
        raise RuntimeError(
            "XXZ coefficients came out complex -- every term has an even Y count, so the folded "
            "phase must be real. A complex .c here means the construction is wrong and every "
            "'real path' number below would be measuring the complex path."
        )
    return hamiltonian


def xxz_krylov_states(num_qubits: int, max_states: int) -> np.ndarray:
    """Unpacked ``(N, n)`` uint8 states reachable from |Neel> by nearest-neighbour hops.

    The hop conserves magnetization, so this spans the physical sector the XXZ ground state lives in
    -- which is the point: a ``rng.choice`` subspace is 3.6-6.1% dense against 32-44% here, and the
    two regimes differ by 3-5x in iteration count.

    Unpacked rather than ``poc14``'s packed form, because ``sqd`` is called with ``packed=False``;
    the states it returns are not lex-sorted either, which is correct -- ``sqd`` uniquifies and sorts
    internally, and only ``hproj(unique_states=True)`` requires pre-sorted input.
    """
    neel = np.zeros(num_qubits, np.uint8)
    neel[::2] = 1
    frontier = {neel.tobytes()}
    seen = set(frontier)
    while len(seen) <= max_states:
        nxt = set()
        for state_bytes in frontier:
            state = np.frombuffer(state_bytes, np.uint8)
            for a in range(num_qubits):
                b = (a + 1) % num_qubits
                if state[a] != state[b]:
                    hopped = state.copy()
                    hopped[a], hopped[b] = hopped[b], hopped[a]
                    key = hopped.tobytes()
                    if key not in seen:
                        seen.add(key)
                        nxt.add(key)
        if not nxt:  # The sector is exhausted before the cap; N is then smaller than asked for.
            break
        frontier = nxt
    rows = np.frombuffer(b"".join(sorted(seen)), np.uint8).reshape(-1, num_qubits)
    return rows[:max_states]


def per_device_bytes() -> dict:
    """``bytes_in_use`` per addressable device, or an empty dict where XLA has no such accounting.

    Only *addressable* devices are read: on a multi-process mesh a rank cannot query a peer's
    allocator, so summing across ``jax.devices()`` would either fail or silently report this rank's
    numbers as global. Per-rank output is the honest form, and rank 0's is what gets printed.

    ``bytes_in_use``, not ``peak_bytes_in_use``: the peak is a high-water mark that never decreases,
    so it cannot show a *reduction* from adding devices -- ``poc8`` recorded that exact mistake.
    """
    out = {}
    for dev in jax.local_devices():
        try:
            stats = dev.memory_stats()
        except (AttributeError, RuntimeError):
            continue
        if stats and "bytes_in_use" in stats:
            out[str(dev)] = stats["bytes_in_use"]
    return out


def solve(hamiltonian, states, mesh=None):
    """One whole ``sqd`` solve, on ``mesh`` if given. Returns the eigenvalue as a float.

    The mesh is set around the call rather than passed in: ``rqutils`` reads
    ``jax.sharding.get_abstract_mesh()``, so establishing it is the caller's job and the *same* call
    serves both arms. Any divergence between them is therefore a sharding effect, not a different
    computation.
    """
    if mesh is None:
        return float(sqd(hamiltonian, states, return_eigvec=False, cache_level=CACHE_LEVEL))
    with jax.sharding.set_mesh(mesh):
        return float(sqd(hamiltonian, states, return_eigvec=False, cache_level=CACHE_LEVEL))


def mesh_sizes(total: int) -> tuple:
    """Powers of two up to ``total``, so each step doubles the device count.

    Doubling is what makes the memory curve readable: a replicated term stays flat while a sharded
    one halves, and a non-power-of-two step confounds the two.
    """
    sizes, size = [], 1
    while size <= total:
        sizes.append(size)
        size *= 2
    return tuple(sizes)


def main():
    desc = init_devices(options.devices, options.host_devices)
    virtual = jax.devices()[0].platform == "cpu"
    print(f"POC 15: sqd at scale on a real mesh\n\nrunning on {desc}")
    if virtual:
        print(
            "\n*** VIRTUAL CPU DEVICES: correctness is meaningful, TIMINGS ARE NOT (they share one\n"
            "physical backend, per CLAUDE.md) and are suppressed below. Pass --devices to measure. ***"
        )

    hamiltonian = xxz_hamiltonian(options.num_qubits, options.jz)
    states = xxz_krylov_states(options.num_qubits, options.max_states)
    coeff_sum = float(np.abs(hamiltonian.c).sum())
    header(f"fixture: 1D XXZ n={options.num_qubits} Jz={options.jz} cache_level={CACHE_LEVEL}")
    print(
        f"  N={len(states)} states (unpacked, unsorted -- sqd uniquifies internally)\n"
        f"  J={hamiltonian.x.shape[0]} X-groups, maxK={hamiltonian.z.shape[1]}, "
        f"dtype={hamiltonian.c.dtype}, sum|c_k|={coeff_sum:.4f}"
    )
    if hamiltonian.x.shape[0] == 0:
        raise RuntimeError("empty Hamiltonian -- the fixture is broken, not the solver")

    sizes = mesh_sizes(jax.device_count())
    if len(sizes) < 2:
        print(
            "\nOnly one device, so there is no scaling curve to measure -- this script needs >= 2.\n"
            "  mpirun -n 4 uv run --extra mpi python <this script> --devices mpi   # 1 GPU/node\n"
            "  uv run --extra qiskit python <this script> --devices 0,1,2,3           # 1 node\n"
            "poc7_sharding.py covers single-device-vs-sharded correctness; this covers memory/speed."
        )
        return 1

    header("Claim 1+2: per-device memory and wall clock against device count")
    print(
        "Each row adds devices at FIXED N. A replicated term stays flat as devices double; a\n"
        "sharded one halves. Speedup is against the 1-device arm, both warm. The energy is\n"
        "asserted, not printed for inspection: a broken arm does less work and posts a better time."
    )
    print(
        f"\n{'devices':>8} {'baseline MB':>12} {'solve MB':>10} {'delta MB':>9} "
        f"{'ms':>9} {'|dE|':>10}  speedup"
    )

    reference, base_timing = None, None
    for size in sizes:
        mesh = make_1d_mesh(devices=jax.devices()[:size]) if size > 1 else None

        # Baseline before the solve, so `delta` isolates what this solve allocated from whatever the
        # process already held. Read on this rank only -- see per_device_bytes.
        before = per_device_bytes()
        eigval = solve(hamiltonian, states, mesh)
        after = per_device_bytes()

        if reference is None:
            reference = eigval
        dE = abs(eigval - reference)
        # Assert rather than report. The tolerance is loose in absolute terms but the observed
        # agreement on 4 real GPUs was 4.441e-16 (poc7), so anything near this bound is a real defect.
        assert dE < 1e-9 * max(abs(reference), 1.0), (
            f"{size} devices: energy moved by {dE:.3e} from the 1-device result {reference:.12f}. "
            "A sharding bug, not a tolerance question -- poc7 measures 4.441e-16 here."
        )

        # `n/a` rather than nan where XLA exposes no allocator accounting (the CPU backend): a
        # printed nan reads as a failed measurement, and this is an absent one.
        local = next(iter(sorted(before))) if before else None
        have_mem = local is not None and local in after
        base_s = f"{before[local] / 2**20:.1f}" if have_mem else "n/a"
        solve_s = f"{after[local] / 2**20:.1f}" if have_mem else "n/a"
        delta_s = f"{(after[local] - before[local]) / 2**20:.1f}" if have_mem else "n/a"

        if virtual:
            print(
                f"{size:>8} {base_s:>12} {solve_s:>10} {delta_s:>9} "
                f"{'--':>9} {dE:>10.1e}  (timings suppressed)"
            )
            continue

        timing = timeit(lambda m=mesh: solve(hamiltonian, states, m), f"{size}dev", trials=3)
        if base_timing is None:
            base_timing = timing
            verdict = "baseline"
        else:
            verdict = fmt_ratio(base_timing, timing)
        print(
            f"{size:>8} {base_s:>12} {solve_s:>10} {delta_s:>9} "
            f"{timing.min_s * 1e3:>9.1f} {dE:>10.1e}  {verdict}"
        )

    header("VERDICT")
    print(f"  energy invariant across {sizes} devices: max |dE| within assertion bound.")
    if virtual:
        print("  Memory numbers above are CPU-allocator numbers and the timings were suppressed.")
        print(
            "  Re-run with --devices (or --devices mpi) for the measurement this script exists for."
        )
    else:
        print(
            "  READ THE MEMORY COLUMN AS A CURVE, not as two points: `states` is replicated today"
        )
        print("  (13*N per device, the term the (0,0) floor cannot shed) while the solver's O(N)")
        print("  vectors shard, so the honest expectation is a falling-but-not-halving delta.")
        print("  A FLAT delta means nothing sharded -- check the spec, not just the energy.")
        if options.devices == "mpi":
            print()
            print("  Multi-NODE topology: these collectives crossed a network, not NVLink. A ratio")
            print(
                "  measured here is the pessimistic case and does NOT transfer to several GPUs in"
            )
            print("  one box -- the interconnect is the variable under test, so record which one.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
