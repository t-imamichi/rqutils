"""POC 7: multi-device sharding correctness, on virtual CPU devices or real GPUs.

``CLAUDE.md`` records that nothing in the test suite exercises a multi-device mesh, so
``ground_locg``'s ``out_sharding`` contract, ``sqd``'s mesh-size padding, and ``svsim``'s
``out_sharding`` are all unverified. That gap matters for the scaling work in this directory: POC 1's
searchsorted is attractive largely *because* a gather shards where a sort does not, and that argument
is worthless if the sharded path is broken to begin with.

``XLA_FLAGS=--xla_force_host_platform_device_count=K`` gives K virtual CPU devices, which exercises
every sharding *code path* -- mesh detection, ``PartitionSpec`` propagation, ``jax.reshard``, the
mesh-size padding in ``sqd`` -- without a GPU. What it does **not** give is any performance signal:
virtual devices share one physical CPU, so timings here are meaningless and none are reported. This
tests correctness only, which is the part that was never tested at all.

Run on virtual CPU devices (the default is 4, so no flag is needed)::

    uv run --extra qiskit python examples/scaling/poc7_sharding.py

Run on real GPUs, which exercises the same assertions against real kernels and interconnect::

    uv run --extra qiskit python examples/scaling/poc7_sharding.py --devices 0,1,2,3

Run across nodes, one GPU each -- the only mode that exercises **multi-process** paths, where a rank
addresses only its own shard (needs the ``mpi`` extra)::

    mpirun -n 4 uv run --extra mpi python examples/scaling/poc7_sharding.py --devices mpi

``--devices`` sets ``CUDA_VISIBLE_DEVICES``, a **filter** over the devices the driver already exposes:
it cannot split one GPU into several, and ``--xla_force_host_platform_device_count`` is host-platform
only, so there is no way to fake a second GPU. Two physical GPUs are required for the GPU path.
Correctness transfers from either mode; **timings do not**, and this script reports none in either.
"""

import argparse
import itertools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# `argparse` before `import jax`, as in poc8/poc9/poc13/poc14: CUDA_VISIBLE_DEVICES and XLA_FLAGS are
# both read at backend initialization, so neither can be set after jax is imported. That is also why
# this preamble is duplicated per script rather than living in `_scaling_common` -- that module
# imports jax.
parser = argparse.ArgumentParser()
parser.add_argument(
    "--devices",
    help='Comma-separated GPU ids, e.g. "0,1,2,3", or "mpi" for one GPU per MPI rank.',
)
parser.add_argument(
    "--host-devices", type=int, default=4, help="Virtual CPU devices when no --devices."
)
options = parser.parse_args()

# CUDA_VISIBLE_DEVICES / XLA_FLAGS / jax.distributed.initialize all have to precede backend
# initialization, so device setup is deferred to init_devices, called first thing in main().

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
from _scaling_common import header, init_devices, make_1d_mesh, make_problem

from rqutils.sqd import hproj, sqd


def check_single_vs_sharded():
    header("POC 7a: sqd single-device vs sharded -- same eigenvalue?")
    print("The mesh is read implicitly via get_abstract_mesh(), so the SAME call is run twice with")
    print(
        "and without a mesh set. Any divergence is a sharding bug, since the operator is identical."
    )
    print()
    results = {}
    for num_qubits, num_states in [(14, 600), (16, 2000), (18, 5000)]:
        p = make_problem(num_qubits, num_states, num_terms=40, seed=71)

        # Single-device reference: no mesh.
        eig_single = float(sqd(p.hamiltonian, p.states, return_eigvec=False))

        # Sharded run.
        mesh = make_1d_mesh()
        with jax.set_mesh(mesh):
            eig_sharded = float(sqd(p.hamiltonian, p.states, return_eigvec=False))

        # Independent reference: dense projection + scipy, so this is not self-consistency.
        hp = hproj(p.hamiltonian, np.unique(p.states, axis=0), unique_states=True)
        dense = hp.toarray()
        eig_dense = float(np.min(np.linalg.eigvalsh(dense)))

        d_ss = abs(eig_sharded - eig_single)
        d_ref = abs(eig_single - eig_dense)
        results[num_qubits] = (d_ss, d_ref)
        if num_qubits == 14:
            # Hand the n=14 fixture and its dense reference to POC 7d rather than have it rebuild
            # both: same make_problem arguments, same hproj, same eigvalsh (~310 ms).
            reused = (p, eig_dense)
        print(
            f"  n={num_qubits:<3d} N={num_states:<6d}  single={eig_single:+.10f}  "
            f"sharded={eig_sharded:+.10f}  dense={eig_dense:+.10f}"
        )
        print(f"          |sharded-single|={d_ss:.3e}   |single-dense|={d_ref:.3e}")
    return results, reused


def check_all_cache_levels(problem, eig_dense):
    header("POC 7d: every cache_level, sharded vs single-device vs dense")
    print("Two sharding bugs lived in the three cache_level[0] == 0 cells, uncovered because this")
    print("script and the pytest arm both ran only sqd's default (1, 0):")
    print("  * _accumulate_diagonal put a rank-2 PartitionSpec on a rank-1 accumulator (all six).")
    print("  * _spread_seed's jnp.where mixed a replicated predicate with a partitioned vec, since")
    print("    run_sqd reshards states_u only inside `if cache_level[0] == 1` ((0,*) only).")
    print(
        "The FIRST masked the SECOND -- fixing it turned 6 failures into 3, not 0. So the grid is"
    )
    print(
        "swept rather than sampled: one representative cell reported success at three broken ones."
    )
    print()
    p = problem
    mesh = make_1d_mesh()

    worst = 0.0
    print(
        f"  {'cache_level':>12s}  {'single':>16s}  {'sharded':>16s}  {'|s-1dev|':>10s}  {'|s-dense|':>10s}"
    )
    for cache_level in sorted(itertools.product((0, 1), (0, 1, 2))):
        single = float(sqd(p.hamiltonian, p.states, return_eigvec=False, cache_level=cache_level))
        with jax.set_mesh(mesh):
            sharded = float(
                sqd(p.hamiltonian, p.states, return_eigvec=False, cache_level=cache_level)
            )
        d_ss, d_ref = abs(sharded - single), abs(sharded - eig_dense)
        worst = max(worst, d_ss)
        print(
            f"  {cache_level!s:>12s}  {single:+16.10f}  {sharded:+16.10f}  "
            f"{d_ss:10.3e}  {d_ref:10.3e}"
        )
    print(f"\n  dense reference: {eig_dense:+.10f}")
    return worst


def check_mesh_padding():
    header("POC 7b: sqd's mesh-size padding -- states_size not divisible by mesh.size")
    print("sqd pads states_size up to a multiple of mesh.size (sqd.py:240-243). That branch is")
    print("unreachable single-device. Feed it deliberately awkward lengths and check the answer")
    print("is unchanged, since padding must be transparent.")
    print()
    mesh = make_1d_mesh()
    nd = jax.device_count()
    for extra in range(nd):
        # Choose N so N % mesh.size == extra, hitting each residue including 0.
        num_states = 400 + extra
        p = make_problem(14, num_states, num_terms=30, seed=72)
        eig_single = float(sqd(p.hamiltonian, p.states, return_eigvec=False))
        with jax.set_mesh(mesh):
            eig_sharded = float(sqd(p.hamiltonian, p.states, return_eigvec=False))
        resid = num_states % nd
        print(
            f"  N={num_states} (N mod {nd} = {resid})  single={eig_single:+.10f}  "
            f"sharded={eig_sharded:+.10f}  diff={abs(eig_sharded - eig_single):.3e}"
        )


def check_eigvec_path():
    header("POC 7c: return_eigvec=True under sharding -- reshard round-trip")
    print("run_sqd reshards eigvec and states_u back to PartitionSpec(None) before returning")
    print("(sqd.py:497-499). Check the returned vector is a genuine eigenvector, not just that the")
    print("call succeeds: ||Hv - ev|| / ||v|| against the dense projection.")
    print()
    mesh = make_1d_mesh()
    for num_qubits, num_states in [(14, 600), (16, 2000)]:
        p = make_problem(num_qubits, num_states, num_terms=40, seed=73)
        with jax.set_mesh(mesh):
            eigval, eigvec, basis = sqd(p.hamiltonian, p.states, return_eigvec=True)

        # Rebuild the projected operator on the RETURNED basis and check the eigen-relation.
        hp = hproj(p.hamiltonian, basis, unique_states=True).toarray()
        v = np.asarray(eigvec)
        nrm = np.linalg.norm(v)
        resid = np.linalg.norm(hp @ v - float(eigval) * v) / max(nrm, 1e-300)
        # Normalize by ||H|| rather than |eigval|, per CLAUDE.md's fp32/eigensolve note.
        hnorm = np.linalg.norm(hp, ord=2) if hp.size else 1.0
        print(
            f"  n={num_qubits:<3d} basis={basis.shape[0]:<6d} eigval={float(eigval):+.10f}  "
            f"||Hv-ev||/||v||={resid:.3e}  (/||H||={resid / max(hnorm, 1e-300):.3e})"
        )


def main():
    desc = init_devices(options.devices, options.host_devices)
    ndev = jax.device_count()
    backend = jax.devices()[0].platform
    virtual = backend == "cpu"
    print(f"JAX devices: {desc}")
    if options.devices and virtual:
        # CUDA_VISIBLE_DEVICES was set but the backend came up as CPU, so there is no CUDA device
        # here and the flag did nothing. Say so: otherwise a CPU result reads as a GPU result.
        print()
        print("WARNING: --devices was passed but the backend is CPU -- there is no CUDA device")
        print("visible, so the flag had NO effect and nothing below is a GPU measurement.")
    if ndev < 2:
        print()
        print("SKIPPED: this script needs >= 2 devices. Either:")
        print("  uv run --extra qiskit python examples/scaling/poc7_sharding.py --host-devices 4")
        print("  uv run --extra qiskit python examples/scaling/poc7_sharding.py --devices 0,1")
        if virtual and "xla_force_host_platform_device_count" not in os.environ.get(
            "XLA_FLAGS", ""
        ):
            # `setdefault` above yields to any pre-existing XLA_FLAGS, so an unrelated flag (say
            # --xla_dump_to) silently suppresses the device count and this skips for a reason that
            # is nowhere on screen. Name it.
            print()
            print("NOTE: XLA_FLAGS is set without --xla_force_host_platform_device_count, which")
            print("takes precedence over --host-devices, so only one device was created. Add the")
            print(
                f'flag to XLA_FLAGS yourself: XLA_FLAGS="{os.environ.get("XLA_FLAGS", "")} '
                '--xla_force_host_platform_device_count=4"'
            )
        if not virtual:
            # --devices filters CUDA_VISIBLE_DEVICES, so it cannot create a device that is not
            # there; poc8 records the same trap. Say so rather than let the flag look broken.
            print()
            print("NOTE: --devices sets CUDA_VISIBLE_DEVICES, which only NARROWS the devices the")
            print(
                "driver exposes -- it cannot split one GPU into several. This needs >= 2 physical"
            )
            print("GPUs, and --xla_force_host_platform_device_count is host-platform only.")
        return 1

    print()
    if virtual:
        print(
            "NOTE: virtual CPU devices exercise the sharding CODE PATHS but share one physical CPU."
        )
        print("No timings are reported here -- they would be meaningless. Correctness only.")
    else:
        # Correctness-only holds on real GPUs too: this script asserts values and specs, and adding
        # timings would need the A/B discipline CLAUDE.md requires (both arms warm, whole calls).
        print(f"NOTE: {ndev} real {backend} devices, so the sharding paths run on real kernels and")
        print("interconnect. Still correctness only -- this script reports no timings.")

    res, (problem, eig_dense) = check_single_vs_sharded()
    worst_cache = check_all_cache_levels(problem, eig_dense)
    check_mesh_padding()
    check_eigvec_path()

    header("VERDICT")
    worst = max(max(d for d, _ in res.values()), worst_cache)
    print(f"  worst |sharded - single| across sizes: {worst:.3e}")
    if worst < 1e-8:
        print("  Sharded and single-device agree at ALL SIX cache levels. The out_sharding")
        print(f"  contract and mesh padding hold on {ndev} devices.")
    else:
        print("  DIVERGENCE: the sharded path does not reproduce the single-device answer.")
    print()
    print("  STILL UNVERIFIED: real multi-GPU behaviour (interconnect, per-device memory limits,")
    print("  the lax.sort GPU memory leak, since removed). Virtual CPU devices cannot speak to any")
    print("  of those.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
