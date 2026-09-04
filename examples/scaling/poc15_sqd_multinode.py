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

**Multi-process measures one point, not a curve.** Every rank has to take part in every mesh, so a
sub-mesh over ``jax.devices()[:k]`` is not available: with one GPU per node it would exclude whole
processes, and those ranks then call ``sqd`` on a mesh they hold no shard of -- measured on 4 nodes, the
2-device row raised ``FullyReplicatedShard: Array has no addressable shards`` from inside
``process_allgather``, on exactly the 2 excluded ranks. Get the curve from one job per rank count::

    for n in 1 2 4; do
        mpirun -n $n uv run --extra mpi python examples/scaling/poc15_sqd_multinode.py --devices mpi
    done

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
from rqutils.sqd import _host_scalar, run_sqd, sqd

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


def solve(hamiltonian, states, mesh=None, retain=False):
    """One whole ``sqd`` solve, on ``mesh`` if given. Returns the eigenvalue as a float.

    The mesh is set around the call rather than passed in: ``rqutils`` reads
    ``jax.sharding.get_abstract_mesh()``, so establishing it is the caller's job and the *same* call
    serves both arms. Any divergence between them is therefore a sharding effect, not a different
    computation.

    ``retain=True`` goes through ``run_sqd`` instead and hands back its **device** arrays, so a caller
    can read the allocator while they are still referenced. Both halves of that are load-bearing:

    * Sampling around a call that returns only a scalar reads the *resting* allocator value twice, so
      the delta is structurally 0.0 whatever the truth -- which is exactly what the first 2- and 4-GPU
      rows printed. ``poc8``'s docstring records the identical defect.
    * ``sqd`` cannot serve this, because it converts on the way out: ``np.array(eigvec[...])`` and
      ``np.asarray(basis_states)``. Holding *those* keeps no device memory alive, so routing
      ``return_eigvec=True`` through ``sqd`` looks like a fix and measures the same 0.0. ``run_sqd`` is
      the innermost layer whose outputs are still ``jax.Array``.

    Do not "simplify" either half back.
    """
    if not retain:
        if mesh is None:
            return float(sqd(hamiltonian, states, return_eigvec=False, cache_level=CACHE_LEVEL))
        with jax.sharding.set_mesh(mesh):
            return float(sqd(hamiltonian, states, return_eigvec=False, cache_level=CACHE_LEVEL))

    # `run_sqd` takes packed states and a static size, which `sqd` would otherwise derive. Mirrors
    # `sqd`'s own defaulting (power-of-two bucketing, then the mesh round-up) so the measured arm is
    # the same shape the normal path would solve.
    states_p = PauliSumXZ.pack_states(states)
    states_size = 1 << max((states_p.shape[0] - 1).bit_length(), 1)
    if mesh is not None and (resid := states_size % mesh.size) != 0:
        states_size += mesh.size - resid
    if (deficit := states_size - states_p.shape[0]) > 0:
        states_p = np.append(
            states_p, np.full((deficit, states_p.shape[1]), 255, dtype=np.uint8), axis=0
        )

    def run():
        # eigval, eigvec, basis, subspace_dim, converged -- every one a live jax.Array.
        out = run_sqd(hamiltonian, states_p, states_size, True, cache_level=CACHE_LEVEL)
        assert bool(_host_scalar(out[-1])), "run_sqd did not converge"
        return float(_host_scalar(out[0])), out[1], out[2]

    if mesh is None:
        return run()
    with jax.sharding.set_mesh(mesh):
        return run()


def mesh_sizes(total: int) -> tuple:
    """Powers of two up to ``total``, so each step doubles the device count.

    Doubling is what makes the memory curve readable: a replicated term stays flat while a sharded
    one halves, and a non-power-of-two step confounds the two.

    **Single process only.** A sub-mesh over ``jax.devices()[:k]`` is fine when one process owns every
    device, and impossible across processes: with one GPU per node, device count equals process count,
    so a mesh of size ``k < total`` excludes ``total - k`` processes entirely -- and those ranks then
    call ``sqd`` on a mesh they hold no shard of. Measured on 4 nodes: the 2-device row raised
    ``FullyReplicatedShard: Array has no addressable shards`` from *inside* ``process_allgather``, on
    exactly the 2 ranks left out. The multi-process path therefore measures **one** point, and the
    sweep comes from launching separate jobs -- see :func:`main`.
    """
    sizes, size = [], 1
    while size <= total:
        sizes.append(size)
        size *= 2
    return tuple(sizes)


def main():
    desc = init_devices(options.devices, options.host_devices)
    virtual = jax.devices()[0].platform == "cpu"

    # Rank 0 prints; every other rank stays silent. Without this, `mpirun -n 4` emitted four copies of
    # every header and table, and two ranks' VERDICT blocks interleaved mid-line into text that read as
    # one mangled paragraph -- the 4-GPU log is the record. `examples/svsim.py` gates on
    # `jax.process_index()` the same way, and this follows it rather than inventing a second
    # convention. Bound to distinct names (`emit`/`section`) rather than shadowing the builtin, which
    # ruff rejects as a forward reference in this scope, and *not* pushed into
    # `_scaling_common.header`, which every other (single-process) script shares.
    #
    # Printing only, never computation: every rank still runs the whole solve and the assertion. A
    # collective inside a rank-0 branch would deadlock, which is why the gate wraps output alone.
    rank0 = jax.process_index() == 0
    emit = print if rank0 else lambda *a, **kw: None
    section = header if rank0 else lambda title: None

    emit(f"POC 15: sqd at scale on a real mesh\n\nrunning on {desc}")
    if virtual:
        emit(
            "\n*** VIRTUAL CPU DEVICES: correctness is meaningful, TIMINGS ARE NOT (they share one\n"
            "physical backend, per CLAUDE.md) and are suppressed below. Pass --devices to measure. ***"
        )

    hamiltonian = xxz_hamiltonian(options.num_qubits, options.jz)
    states = xxz_krylov_states(options.num_qubits, options.max_states)
    coeff_sum = float(np.abs(hamiltonian.c).sum())
    section(f"fixture: 1D XXZ n={options.num_qubits} Jz={options.jz} cache_level={CACHE_LEVEL}")
    emit(
        f"  N={len(states)} states (unpacked, unsorted -- sqd uniquifies internally)\n"
        f"  J={hamiltonian.x.shape[0]} X-groups, maxK={hamiltonian.z.shape[1]}, "
        f"dtype={hamiltonian.c.dtype}, sum|c_k|={coeff_sum:.4f}"
    )
    if hamiltonian.x.shape[0] == 0:
        raise RuntimeError("empty Hamiltonian -- the fixture is broken, not the solver")

    # Multi-process cannot sweep: every rank must participate in every mesh, so the only mesh available
    # is the full one. Sweeping is done by launching one job per rank count (mpirun -n 1, -n 2, -n 4)
    # and comparing the single rows they print.
    multiprocess = jax.process_count() > 1
    sizes = (jax.device_count(),) if multiprocess else mesh_sizes(jax.device_count())
    if multiprocess:
        emit(
            f"\nMULTI-PROCESS: measuring the {jax.device_count()}-device point only. A sub-mesh would\n"
            "exclude whole processes, which then hold no shard of the array `sqd` gathers -- measured,\n"
            "that raises FullyReplicatedShard inside process_allgather on exactly the excluded ranks.\n"
            "For the curve, run this once per rank count and compare rows:\n"
            "  for n in 1 2 4; do mpirun -n $n uv run --extra mpi python <this script> "
            "--devices mpi; done"
        )
    if len(sizes) < 2 and not multiprocess:
        emit(
            "\nOnly one device, so there is no scaling curve to measure -- this script needs >= 2.\n"
            "  mpirun -n 4 uv run --extra mpi python <this script> --devices mpi   # 1 GPU/node\n"
            "  uv run python <this script> --devices 0,1,2,3                         # 1 node\n"
            "poc7_sharding.py covers single-device-vs-sharded correctness; this covers memory/speed."
        )
        return 1

    section("Claim 1+2: per-device memory and wall clock against device count")
    emit(
        "Each row adds devices at FIXED N. A replicated term stays flat as devices double; a\n"
        "sharded one halves. Speedup is against the 1-device arm, both warm. The energy is\n"
        "asserted, not printed for inspection: a broken arm does less work and posts a better time."
    )
    emit(
        f"\n{'devices':>8} {'baseline MB':>12} {'solve MB':>10} {'delta MB':>9} "
        f"{'ms':>9} {'|dE|':>10}  speedup"
    )

    reference, base_timing = None, None
    for size in sizes:
        mesh = make_1d_mesh(devices=jax.devices()[:size]) if size > 1 else None

        # Baseline before the solve, so `delta` isolates what this solve allocated from whatever the
        # process already held. Read on this rank only -- see per_device_bytes.
        #
        # `after` must be read while the solve's arrays are STILL REFERENCED, which is why this asks
        # for the eigenvector and basis and holds them in `live` across the reading. Bracketing a
        # `return_eigvec=False` call reads the resting value twice and reports 0.0 whatever the truth
        # -- measured, that is what the first 2- and 4-GPU rows printed. See `solve`'s docstring and
        # `poc8`'s, which records the identical defect.
        before = per_device_bytes()
        eigval, *live = solve(hamiltonian, states, mesh, retain=True)
        after = per_device_bytes()
        del live

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
        if not have_mem:
            delta_s = "n/a"
        else:
            delta_b = after[local] - before[local]
            # An exactly-zero delta is reported as `0?` rather than `0.0`, because the two causes are
            # not distinguishable from the number and only one of them is a finding. A real solve
            # allocates O(N) vectors, so a true 0 B means the reading missed them -- the defect this
            # script shipped with. The VERDICT's advice was "a FLAT delta means nothing sharded", which
            # cannot catch this: flat and absent both print 0.0. Any nonzero value is a measurement.
            delta_s = "0?" if delta_b == 0 else f"{delta_b / 2**20:.1f}"

        if virtual:
            emit(
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
        emit(
            f"{size:>8} {base_s:>12} {solve_s:>10} {delta_s:>9} "
            f"{timing.min_s * 1e3:>9.1f} {dE:>10.1e}  {verdict}"
        )

    section("VERDICT")
    emit(f"  energy invariant across {sizes} devices: max |dE| within assertion bound.")
    if virtual:
        emit("  Memory numbers above are CPU-allocator numbers and the timings were suppressed.")
        emit(
            "  Re-run with --devices (or --devices mpi) for the measurement this script exists for."
        )
    else:
        emit("  READ THE MEMORY COLUMN AS A CURVE, not as two points: `states` is replicated today")
        emit("  (13*N per device, the term the (0,0) floor cannot shed) while the solver's O(N)")
        emit("  vectors shard, so the honest expectation is a falling-but-not-halving delta.")
        emit("  A FLAT delta means nothing sharded -- check the spec, not just the energy.")
        emit("  A `0?` delta means the reading MISSED the arrays, not that nothing was allocated:")
        emit("  a real solve allocates O(N) vectors, so an exact 0 B is an instrument failure.")
        if options.devices == "mpi":
            emit()
            emit("  Multi-NODE topology: these collectives crossed a network, not NVLink. A ratio")
            emit("  measured here is the pessimistic case and does NOT transfer to several GPUs in")
            emit("  one box -- the interconnect is the variable under test, so record which one.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
