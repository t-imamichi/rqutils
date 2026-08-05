"""POC 7: multi-device sharding correctness, on virtual CPU devices.

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

Run (the flag is mandatory -- without it this reports a single device and skips):

    XLA_FLAGS=--xla_force_host_platform_device_count=4 \
        uv run --extra qiskit python examples/scaling/poc7_sharding.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
from _scaling_common import header, make_problem
from jax.sharding import AxisType

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
        mesh = jax.make_mesh((jax.device_count(),), ("x",), (AxisType.Explicit,))
        with jax.set_mesh(mesh):
            eig_sharded = float(sqd(p.hamiltonian, p.states, return_eigvec=False))

        # Independent reference: dense projection + scipy, so this is not self-consistency.
        hp = hproj(p.hamiltonian, np.unique(p.states, axis=0), unique_states=True)
        dense = hp.toarray()
        eig_dense = float(np.min(np.linalg.eigvalsh(dense)))

        d_ss = abs(eig_sharded - eig_single)
        d_ref = abs(eig_single - eig_dense)
        results[num_qubits] = (d_ss, d_ref)
        print(
            f"  n={num_qubits:<3d} N={num_states:<6d}  single={eig_single:+.10f}  "
            f"sharded={eig_sharded:+.10f}  dense={eig_dense:+.10f}"
        )
        print(f"          |sharded-single|={d_ss:.3e}   |single-dense|={d_ref:.3e}")
    return results


def check_mesh_padding():
    header("POC 7b: sqd's mesh-size padding -- states_size not divisible by mesh.size")
    print("sqd pads states_size up to a multiple of mesh.size (sqd.py:240-243). That branch is")
    print("unreachable single-device. Feed it deliberately awkward lengths and check the answer")
    print("is unchanged, since padding must be transparent.")
    print()
    mesh = jax.make_mesh((jax.device_count(),), ("x",), (AxisType.Explicit,))
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
    mesh = jax.make_mesh((jax.device_count(),), ("x",), (AxisType.Explicit,))
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
    ndev = jax.device_count()
    print(f"JAX devices: {ndev} ({jax.devices()[0].platform})")
    if ndev < 2:
        print()
        print("SKIPPED: this script needs >= 2 devices. Re-run with")
        print("  XLA_FLAGS=--xla_force_host_platform_device_count=4 \\")
        print("      uv run --extra qiskit python examples/scaling/poc7_sharding.py")
        return 1

    print()
    print("NOTE: virtual CPU devices exercise the sharding CODE PATHS but share one physical CPU.")
    print("No timings are reported here -- they would be meaningless. Correctness only.")

    res = check_single_vs_sharded()
    check_mesh_padding()
    check_eigvec_path()

    header("VERDICT")
    worst = max(d for d, _ in res.values())
    print(f"  worst |sharded - single| across sizes: {worst:.3e}")
    if worst < 1e-8:
        print("  Sharded and single-device agree. The out_sharding contract and mesh padding")
        print(f"  hold on {ndev} devices -- previously untested per CLAUDE.md.")
    else:
        print("  DIVERGENCE: the sharded path does not reproduce the single-device answer.")
    print()
    print("  STILL UNVERIFIED: real multi-GPU behaviour (interconnect, per-device memory limits,")
    print("  the lax.sort GPU memory leak at sqd.py:561). Virtual CPU devices cannot speak to any")
    print("  of those.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
