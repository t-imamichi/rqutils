"""POC 15: ``sqd`` at scale on a real multi-node GPU mesh -- memory and speed, not correctness.

``sharding.py`` already settles multi-node **correctness**: it passed on 4 GPUs across 4 nodes at
all six ``cache_level`` cells, every ``N mod mesh.size``, and the ``return_eigvec`` round trip, worst
``|sharded - single|`` = 4.441e-16. It does so at ``n <= 18``, ``N <= 5000`` -- fixtures small enough
that every arm fits on one device, which is what makes them safe to compare against a single-device
reference. This script exists for the two questions that *cannot* be asked there:

**Claim 1: does per-device memory actually fall as devices are added?** ``CLAUDE.md`` records the
cost model -- ``states`` is replicated today at ``13 * N`` bytes per device, the one term the ``(0,0)``
floor cannot shed, while the solver's ``O(N)`` vectors and the diagonal cache do shard. Those are
*predictions from a formula*, and per ``CLAUDE.md`` memory should be asked of XLA rather than derived:
byte-count formulas have predicted a saving where the measured peak **rose**. Here the mesh is swept
over ``1, 2, 4, ...`` devices at fixed ``N``. A replicated term shows as a flat component; a sharded
one halves per doubling.

**Read the ``temp MB`` column for this claim, not ``delta MB``.** Two instruments, and only one can
answer it. ``delta MB`` brackets the call with ``bytes_in_use``, so it can only see arrays that outlive
it -- i.e. what ``run_sqd`` *returns* -- and both of those come back ``P(None,)`` / ``P(None, None)``,
**replicated**. It is therefore flat in device count by construction: measured at N=400000 it read
exactly 6.00 MB on both 2 and 4 GPUs, which is ``eigvec`` float64[524288] = 4.00 MB plus ``basis``
uint8[524288,4] = 2.00 MB. That flatness is not evidence about sharding, and the VERDICT's original
advice ("a FLAT delta means nothing sharded") misfired on its own instrument. ``temp MB`` is
``temp_size_in_bytes`` from ``memory_analysis()``, the per-device scratch peak where the solver's
``O(N)`` working set actually lives; on an n=14 fixture it falls 0.52 -> 0.32 -> 0.23 MB across 1/2/4
devices (2.23x), the shape the cost model predicts. For the sharding itself, assert the **spec**.

**Claim 1 is answered on real GPUs (2026-09-07):** the n=26 N=400000 fixture measured 87.02 -> 48.28 ->
27.16 MB across 1/2/4 nodes, a 3.20x fall, with the excess over ideal halving flat at ~+5 MB -- the
replicated ``states`` term, not a leak. See ``NOTES.md``, "sqd_multinode anchored".

**Claim 2: is the sharded solve faster, and where does it stop being faster?** ``sharding``'s docstring is
explicit that virtual devices "cannot speak to interconnect cost, per-device memory limits, or whether
the sharded solve is actually *faster*". On a **multi-node** mesh the collectives cross a network
rather than NVLink, so this is the pessimistic topology -- and the honest place to find the crossover
below which communication dominates. ``fmt_ratio`` refuses to call a difference inside the measured
spread a win, so a result under the noise floor is reported as unresolved.

**What this script does not claim.** It is not a correctness harness -- it asserts the energy against
the 1-device run so a broken arm cannot post a good time (``CLAUDE.md``: a broken arm flatters its own
benchmark), but the systematic correctness sweep is ``sharding``'s and is not repeated. Both arms are run
warm, per-call, whole-solve, with the arrays passed as arguments rather than closed over.

The fixture is the 1D XXZ Krylov subspace, not ``rng.choice`` rows: a random subspace is 3.6-6.1%
dense against 32-44% for a physical one, and ``CLAUDE.md`` records that a physically-motivated fixture
has *inverted* a conclusion a synthetic one reached. The Hamiltonian is the XXZ chain whose ground
state that subspace is built to span, so the solve is the one a real SKQD workflow performs.

Run on a multi-node cluster with one GPU per node (the only way to reach N GPUs on N nodes)::

    mpirun -n 4 uv run --extra mpi python poc/sqd_multinode.py --devices mpi

**Multi-process measures one point, not a curve.** Every rank has to take part in every mesh, so a
sub-mesh over ``jax.devices()[:k]`` is not available: with one GPU per node it would exclude whole
processes, and those ranks then call ``sqd`` on a mesh they hold no shard of -- measured on 4 nodes, the
2-device row raised ``FullyReplicatedShard: Array has no addressable shards`` from inside
``process_allgather``, on exactly the 2 excluded ranks. Get the curve from one job per rank count::

    for n in 1 2 4; do
        mpirun -n $n uv run --extra mpi python poc/sqd_multinode.py --devices mpi
    done

Because each job measures one mesh size, **the energy check needs the reference passed in**. Without
it a single-row job compares its eigenvalue to itself, ``|dE|`` is zero by construction, and the
assertion passes vacuously -- both real multi-node logs printed that ``0.0e+00`` while checking
nothing. The ``-n 1`` job prints its eigenvalue; hand it to the rest::

    mpirun -n 1 ... --devices mpi                              # prints E_ref
    mpirun -n 4 ... --devices mpi --reference-energy <E_ref>   # asserts against it

A row with no second arm prints ``n/a`` in the ``|dE|`` column rather than ``0.0e+00``, and the VERDICT
says the invariance was not checked. Note the ``-n 1`` job is a sweep point and exits 0: it used to
exit 1 through the "only one device" bail-out, which aborted the whole ``mpirun`` on the first
iteration of the loop above.

**Paste the reference at full precision.** The VERDICT prints it via ``repr``; a fixed ``.12f`` is not
enough. Measured under 2 and 4 local MPI ranks at n=14, N=2000: the truncated ``-22.986068174156``
reported ``|dE|`` = 4.0e-13 at both rank counts -- the rounding error of the *printed string*, ~450x
above the 4.441e-16 ``sharding`` measures, and nothing to do with sharding. The full
``-22.986068174155598`` reports 0.0e+00 (2 ranks, bit-identical to 1 rank) and 3.6e-15 (4 ranks). A
figure near 1e-13 with a hand-shortened reference is that artifact, not a finding.

Run on one node holding several GPUs::

    uv run --extra qiskit python poc/sqd_multinode.py --devices 0,1,2,3

Correctness-only rehearsal on virtual CPU devices (timings are meaningless there and are suppressed)::

    uv run --extra qiskit python poc/sqd_multinode.py --host-devices 4
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# `argparse` before `import jax`, as in gpu_unverified/prefilter_gpu/hash_partition_jax/uniquify_sharded: CUDA_VISIBLE_DEVICES, XLA_FLAGS and
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
parser.add_argument(
    "--reference-energy",
    type=float,
    help="Eigenvalue from the 1-device job, to assert this job's energy against. Multi-process "
    "measures one mesh size, so without this the check compares a row to itself and passes "
    "vacuously; the 1-device job prints the value to paste here.",
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

    Unpacked rather than ``uniquify_sharded``'s packed form, because ``sqd`` is called with ``packed=False``;
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
    so it cannot show a *reduction* from adding devices -- ``gpu_unverified`` recorded that exact mistake.
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


def peak_temp_bytes(hamiltonian, states, mesh=None) -> int | None:
    """XLA's per-device scratch high-water mark for one whole ``run_sqd``, or None if unavailable.

    **This is the number Claim 1 is actually about**, and the allocator probe cannot see it. The
    solver's ``O(N)`` working set -- ``ground_locg``'s 7 carried vectors, the term that *does* shard --
    lives and dies inside the jitted call, so it is already freed by the time `per_device_bytes` reads
    `bytes_in_use` after the call returns. What survives to be sampled is only what `run_sqd` returns.

    Measured, that mattered: the 2- and 4-GPU rows both printed a flat delta of exactly 6.00 MB at
    N=400000, which decomposes as ``eigvec`` float64[524288] = 4.00 MB plus ``basis`` uint8[524288,4] =
    2.00 MB -- both of which come back ``P(None,)`` / ``P(None, None)``, i.e. **replicated**, so the
    figure is flat in device count by construction and says nothing about sharding either way. The
    VERDICT's advice ("a FLAT delta means nothing sharded") therefore misfired on its own instrument.

    ``temp_size_in_bytes`` is per-device and comes from the compiler rather than a byte formula, which
    is what ``CLAUDE.md`` prescribes for memory. On the same fixture at n=14 it falls 0.52 -> 0.32 ->
    0.23 MB across 1/2/4 devices (2.23x), the falling-but-not-halving shape the cost model predicts.
    """
    states_p = PauliSumXZ.pack_states(states)
    states_size = 1 << max((states_p.shape[0] - 1).bit_length(), 1)
    if mesh is not None and (resid := states_size % mesh.size) != 0:
        states_size += mesh.size - resid
    if (deficit := states_size - states_p.shape[0]) > 0:
        states_p = np.append(
            states_p, np.full((deficit, states_p.shape[1]), 255, dtype=np.uint8), axis=0
        )
    fn = jax.jit(lambda h, s: run_sqd(h, s, states_size, True, cache_level=CACHE_LEVEL))
    try:
        if mesh is None:
            return int(
                fn.lower(hamiltonian, states_p).compile().memory_analysis().temp_size_in_bytes
            )
        with jax.sharding.set_mesh(mesh):
            return int(
                fn.lower(hamiltonian, states_p).compile().memory_analysis().temp_size_in_bytes
            )
    except (AttributeError, RuntimeError, NotImplementedError):
        # Some backends expose no memory_analysis; absent is not zero, so say so with None.
        return None


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
      rows printed. ``gpu_unverified``'s docstring records the identical defect.
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
    # `--devices mpi` is a sweep *point*, whatever its rank count: the documented curve is one job per
    # count, and its `-n 1` job is the 1-device reference the other jobs are read against. So the
    # single-device bail-out below must not fire for it -- keyed on the launch mode rather than on
    # `process_count() > 1`, which is False under `mpirun -n 1` and made the documented
    # `for n in 1 2 4` loop exit 1 on its first iteration and abort the whole mpirun.
    launched_mpi = options.devices == "mpi"
    multiprocess = jax.process_count() > 1
    sizes = (
        (jax.device_count(),) if (multiprocess or launched_mpi) else mesh_sizes(jax.device_count())
    )
    if multiprocess:
        emit(
            f"\nMULTI-PROCESS: measuring the {jax.device_count()}-device point only. A sub-mesh would\n"
            "exclude whole processes, which then hold no shard of the array `sqd` gathers -- measured,\n"
            "that raises FullyReplicatedShard inside process_allgather on exactly the excluded ranks.\n"
            "For the curve, run this once per rank count and compare rows:\n"
            "  for n in 1 2 4; do mpirun -n $n uv run --extra mpi python <this script> "
            "--devices mpi; done"
        )
    if len(sizes) == 1 and launched_mpi and not multiprocess:
        emit(
            f"\nSINGLE RANK under --devices mpi: measuring the {jax.device_count()}-device point, which\n"
            "is a sweep row like any other -- and the one the other jobs take as reference. Not a\n"
            "failure: exits 0 so the documented `for n in 1 2 4` loop reaches the multi-rank jobs."
        )
    elif len(sizes) < 2 and not multiprocess and not launched_mpi:
        # Both guards matter: under `--devices mpi` *every* job measures a single size (a sub-mesh
        # would exclude ranks), so a bare `len(sizes) < 2` bail-out would reject the -n 2 and -n 4
        # sweep jobs too -- which is a regression this branch introduced and the decision table caught.
        emit(
            "\nOnly one device, so there is no scaling curve to measure -- this script needs >= 2.\n"
            "  mpirun -n 4 uv run --extra mpi python <this script> --devices mpi   # 1 GPU/node\n"
            "  uv run python <this script> --devices 0,1,2,3                         # 1 node\n"
            "sharding.py covers single-device-vs-sharded correctness; this covers memory/speed."
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
        f"{'temp MB':>9} {'ms':>9} {'|dE|':>10}  speedup"
    )

    reference, base_timing, self_ref = None, None, False
    for size in sizes:
        mesh = make_1d_mesh(devices=jax.devices()[:size]) if size > 1 else None

        # Baseline before the solve, so `delta` isolates what this solve allocated from whatever the
        # process already held. Read on this rank only -- see per_device_bytes.
        #
        # `after` must be read while the solve's arrays are STILL REFERENCED, which is why this asks
        # for the eigenvector and basis and holds them in `live` across the reading. Bracketing a
        # `return_eigvec=False` call reads the resting value twice and reports 0.0 whatever the truth
        # -- measured, that is what the first 2- and 4-GPU rows printed. See `solve`'s docstring and
        # `gpu_unverified`'s, which records the identical defect.
        before = per_device_bytes()
        eigval, *live = solve(hamiltonian, states, mesh, retain=True)
        after = per_device_bytes()
        del live

        # The `delta` columns above can only see what `run_sqd` RETURNS, and both of those arrays come
        # back replicated -- so they are flat in device count whatever the sharding does. `temp MB` is
        # the working set that actually shards, read from the compiler. See `peak_temp_bytes`.
        temp_b = peak_temp_bytes(hamiltonian, states, mesh)

        # In multi-process mode `sizes` holds ONE size, so a reference taken from this loop is this
        # row's own energy and `dE` is |x - x| == 0 by construction -- the assertion then passes
        # vacuously and prints 0.0e+00, which reads exactly like a verified invariance. Both real
        # multi-node logs printed that. `--reference-energy` carries the 1-device job's value in so
        # the check is against another arm; without it a single-row job says `n/a`, never 0.0e+00.
        if reference is None:
            reference = options.reference_energy if options.reference_energy is not None else eigval
        self_ref = len(sizes) == 1 and options.reference_energy is None
        dE = abs(eigval - reference)
        # Assert rather than report. The tolerance is loose in absolute terms but the observed
        # agreement on 4 real GPUs was 4.441e-16 (sharding), so anything near this bound is a real defect.
        assert self_ref or dE < 1e-9 * max(abs(reference), 1.0), (
            f"{size} devices: energy moved by {dE:.3e} from the 1-device result {reference!r}. "
            "A sharding bug, not a tolerance question -- sharding measures 4.441e-16 here."
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

        # `n/a` where the only reference available was this row itself: a printed 0.0e+00 there claims
        # an invariance that was never tested. Only a comparison against another arm earns a number.
        dE_s = "n/a" if self_ref else f"{dE:.1e}"
        temp_s = "n/a" if temp_b is None else f"{temp_b / 2**20:.2f}"

        if virtual:
            emit(
                f"{size:>8} {base_s:>12} {solve_s:>10} {delta_s:>9} {temp_s:>9} "
                f"{'--':>9} {dE_s:>10}  (timings suppressed)"
            )
            continue

        timing = timeit(lambda m=mesh: solve(hamiltonian, states, m), f"{size}dev", trials=3)
        if base_timing is None:
            base_timing = timing
            verdict = "baseline"
        else:
            verdict = fmt_ratio(base_timing, timing)
        emit(
            f"{size:>8} {base_s:>12} {solve_s:>10} {delta_s:>9} {temp_s:>9} "
            f"{timing.min_s * 1e3:>9.1f} {dE_s:>10}  {verdict}"
        )

    section("VERDICT")
    # Only claim the invariance that was actually checked. This line read "energy invariant across
    # (2,) devices" on a run whose only reference was that same row -- the strongest-sounding sentence
    # in the output, backed by |x - x|.
    if reference is not None and not self_ref:
        emit(f"  energy invariant across {sizes} devices: max |dE| within assertion bound.")
    else:
        emit(f"  energy NOT cross-checked: {sizes} is one mesh size and no --reference-energy was")
        emit("  given, so there was no second arm to compare against. This row's eigenvalue is")
        # `repr`, not a fixed number of decimals. This value exists to be pasted into the next job's
        # --reference-energy, and `.12f` truncates it: measured, the 2- and 4-rank rows then reported
        # |dE| = 4.0e-13 -- the rounding error of the printed string, ~450x above the 4.441e-16 sharding
        # measures and entirely an artifact of this line. With the full repr the same runs report
        # 0.0e+00 and 3.6e-15. A reference the script itself prints must not blunt the comparison it
        # exists to sharpen, so never reformat this to a fixed precision.
        emit(f"  {reference!r} -- pass it to the other jobs in the sweep to make the check real:")
        emit(f"    mpirun -n <k> ... --devices mpi --reference-energy {reference!r}")
    if virtual:
        emit("  Memory numbers above are CPU-allocator numbers and the timings were suppressed.")
        emit(
            "  Re-run with --devices (or --devices mpi) for the measurement this script exists for."
        )
    else:
        emit("  READ `temp MB` FOR CLAIM 1, NOT `delta MB`. `temp MB` is the compiler's per-device")
        emit("  scratch high-water mark, which is where the solver's O(N) working set lives -- the")
        emit("  term that actually shards. Expect falling-but-not-halving: `states` is replicated")
        emit("  today (13*N per device, the term the (0,0) floor cannot shed).")
        emit("  `delta MB` CANNOT answer Claim 1 and a flat value there is not a finding: it sees")
        emit("  only what run_sqd returns, and eigvec/basis come back P(None,)/P(None,None) --")
        emit(
            "  replicated -- so it is flat in device count however well the solve shards. Measured"
        )
        emit(
            "  at N=400000 it read exactly 6.00 MB on both 2 and 4 GPUs = eigvec 4.00 + basis 2.00."
        )
        emit("  A `0?` delta still means the reading MISSED even those, which is a real instrument")
        emit("  failure. For the sharding itself, assert the spec (CLAUDE.md), not any byte count.")
        if options.devices == "mpi":
            emit()
            emit("  Multi-NODE topology: these collectives crossed a network, not NVLink. A ratio")
            emit("  measured here is the pessimistic case and does NOT transfer to several GPUs in")
            emit("  one box -- the interconnect is the variable under test, so record which one.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
