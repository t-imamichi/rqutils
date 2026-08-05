"""POC 8: the claims that need a real GPU. NOT VERIFIED on the machine this was written on.

Everything else in this directory was measured on an Apple M1, ``jax.default_backend() == "cpu"``,
one device. Three claims cannot be reached from there, and this script exists so they can be settled
in one run on a CUDA box rather than re-derived.

**Claim 1: the ``lax.sort`` memory leak.** ``rqutils/sqd.py:561`` records "lax.sort seems to leak GPU
memory; can lose as much as 5 GB when sorting x of shape (5M,9)". That is an allocator behaviour of a
GPU backend; a CPU backend has no comparable accounting to observe. If POC 1's searchsorted is
adopted, the leak should disappear with the sort -- but "should" is not "does".

**Claim 2: POC 1's speedup on GPU.** Measured 12-25x on CPU. The direction should hold, since a
binary-search gather is strictly less work than a sort, but the magnitude will differ: a GPU sort is
far better optimized relative to its gather than a CPU one, so **expect the GPU speedup to be
smaller**. Quoting the CPU number as a GPU number would be exactly the error ``CLAUDE.md`` warns
about for the MLX arms ("a flat CPU result does not mean a change is worthless" -- and its converse).

**Claim 3: real multi-GPU sharding.** ``poc7_sharding.py`` validated the sharding code paths on
virtual CPU devices and, in doing so, found and fixed a real bug (a scatter in ``vinit_from_min_diag``
missing ``out_sharding``, which made ``sqd`` fail on *any* mesh). Virtual devices cannot speak to
interconnect cost, per-device memory limits, or whether the sharded solve is actually *faster*.

Run on a CUDA machine:

    uv run --extra qiskit python examples/scaling/poc8_gpu_unverified.py
    # multi-GPU:
    uv run --extra qiskit python examples/scaling/poc8_gpu_unverified.py --devices 0,1,2,3
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

parser = argparse.ArgumentParser()
parser.add_argument("--devices", help='Comma-separated GPU ids, e.g. "0,1,2,3".')
parser.add_argument("--num-qubits", type=int, default=28)
parser.add_argument("--num-states", type=int, default=5_000_000)
parser.add_argument("--num-xgroups", type=int, default=50)
options = parser.parse_args()

if options.devices:
    os.environ["CUDA_VISIBLE_DEVICES"] = options.devices

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
from _scaling_common import fmt_ratio, header, make_problem, timeit
from jax.sharding import AxisType
from poc1_searchsorted import FILL_BYTE, xsource_searchsorted_u64

from rqutils.sqd import get_xsource, sqd, uniquify_states


def device_memory_stats(label):
    """Print per-device allocator stats, which only a GPU backend provides."""
    for dev in jax.devices():
        try:
            st = dev.memory_stats()
        except Exception:  # noqa: BLE001 - backend-specific errors; a diagnostic printer must not crash the run
            print(f"  [{label}] {dev}: memory_stats() unavailable (expected on CPU)")
            continue
        if st is None:
            print(f"  [{label}] {dev}: memory_stats() returned None")
            continue
        print(
            f"  [{label}] {dev}: in_use={st.get('bytes_in_use', 0) / 2**30:7.3f}GB  "
            f"peak={st.get('peak_bytes_in_use', 0) / 2**30:7.3f}GB  "
            f"limit={st.get('bytes_limit', 0) / 2**30:7.3f}GB"
        )


def claim1_sort_leak():
    header("CLAIM 1: does lax.sort leak GPU memory, and does searchsorted avoid it?")
    print("Reproduces the in-tree note at rqutils/sqd.py:561 as closely as the shape allows:")
    print(f"  n={options.num_qubits}, N={options.num_states} (note cites (5M, 9))")
    print()
    if jax.default_backend() == "cpu":
        print("  BACKEND IS CPU -- this claim is UNVERIFIABLE here. Skipping.")
        print("  memory_stats() has no GPU-style accounting on CPU, so any number printed would")
        print("  be meaningless rather than merely imprecise.")
        return

    p = make_problem(options.num_qubits, options.num_states, num_terms=8, num_xgroups=4, seed=81)
    size = p.states_p.shape[0]
    states_u = jax.block_until_ready(uniquify_states(p.states_p, size))
    xsig = p.hamiltonian.x[0]

    device_memory_stats("baseline")
    # Repeat the sort path many times; a leak shows as monotonically rising bytes_in_use.
    for rep in range(5):
        out = jax.block_until_ready(get_xsource(xsig, states_u))
        del out
        device_memory_stats(f"sort rep{rep}")
    print()
    for rep in range(5):
        out = jax.block_until_ready(xsource_searchsorted_u64(xsig, states_u))
        del out
        device_memory_stats(f"ssorted rep{rep}")
    print()
    print("  READ: rising bytes_in_use across 'sort repN' with flat 'ssorted repN' confirms both")
    print("  the leak and that searchsorted avoids it. Flat in both means the note is stale or")
    print("  backend-version-specific -- report that, do not quietly drop it.")


def claim2_speedup_on_gpu():
    header("CLAIM 2: POC 1's 12-25x CPU speedup -- what is it on GPU?")
    print("EXPECT A SMALLER NUMBER than the CPU measurement. A GPU sort is well optimized relative")
    print("to its gather, so the ratio should compress. The direction should still favour")
    print("searchsorted, since it is strictly less work.")
    print()
    print(f"{'N':>10s}  {'J':>4s}  {'sort ms':>10s}  {'ssorted ms':>11s}  {'verdict':>34s}")
    for num_states in [200_000, 1_000_000, min(options.num_states, 5_000_000)]:
        p = make_problem(
            options.num_qubits, num_states, num_terms=200, num_xgroups=options.num_xgroups, seed=82
        )
        size = p.states_p.shape[0]
        states_u = jax.block_until_ready(uniquify_states(p.states_p, size))
        xs = p.hamiltonian.x

        def all_sort(states_u=states_u, xs=xs):
            return jax.lax.scan(lambda _, x: (None, get_xsource(x, states_u)), None, xs)[1]

        def all_ss(states_u=states_u, xs=xs):
            return jax.lax.scan(
                lambda _, x: (None, xsource_searchsorted_u64(x, states_u)), None, xs
            )[1]

        # Same equivalence gate as POC 1: valid rows identical, gathers identical.
        ref, got = np.asarray(all_sort()), np.asarray(all_ss())
        is_fill = np.asarray(states_u)[:, 0] == FILL_BYTE
        assert np.array_equal(ref[:, ~is_fill], got[:, ~is_fill]), "valid-row index mismatch on GPU"

        t_sort = timeit(all_sort, "sort", trials=3)
        t_ss = timeit(all_ss, "ssorted", trials=3)
        print(
            f"{size:>10d}  {p.num_xgroups:>4d}  {t_sort.min_s * 1e3:>8.2f}ms  "
            f"{t_ss.min_s * 1e3:>9.2f}ms  {fmt_ratio(t_sort, t_ss):>34s}"
        )


def claim3_multi_gpu():
    header("CLAIM 3: real multi-GPU sharding -- correctness AND speed")
    ndev = jax.device_count()
    print(f"  devices visible: {ndev} ({jax.devices()[0].platform})")
    if ndev < 2:
        print("  Fewer than 2 devices; pass --devices 0,1,... on a multi-GPU box.")
        return

    for num_qubits, num_states in [(20, 200_000), (24, 1_000_000)]:
        p = make_problem(num_qubits, num_states, num_terms=100, num_xgroups=50, seed=83)
        eig_1 = float(sqd(p.hamiltonian, p.states, return_eigvec=False))
        t_1 = timeit(
            lambda p=p: sqd(p.hamiltonian, p.states, return_eigvec=False),
            "1 dev",
            trials=2,
            block=False,
        )

        mesh = jax.make_mesh((ndev,), ("x",), (AxisType.Explicit,))
        with jax.set_mesh(mesh):
            eig_n = float(sqd(p.hamiltonian, p.states, return_eigvec=False))
            t_n = timeit(
                lambda p=p: sqd(p.hamiltonian, p.states, return_eigvec=False),
                f"{ndev} dev",
                trials=2,
                block=False,
            )
        print(
            f"  n={num_qubits} N={num_states}: eig_1={eig_1:+.10f} eig_{ndev}={eig_n:+.10f} "
            f"diff={abs(eig_n - eig_1):.2e}"
        )
        print(
            f"    time: 1dev={t_1.min_s * 1e3:9.1f}ms  {ndev}dev={t_n.min_s * 1e3:9.1f}ms  "
            f"speedup={t_1.min_s / t_n.min_s:.2f}x  (ideal {ndev}.00x)"
        )
    print()
    print("  NOTE: poc7_sharding.py already validated sharding CORRECTNESS on virtual CPU devices")
    print("  and fixed a real bug found that way. What only this script can add is the SPEEDUP and")
    print("  whether per-device memory actually divides as intended.")


def main():
    print(f"backend={jax.default_backend()}  devices={jax.device_count()}")
    if jax.default_backend() == "cpu":
        print()
        print("=" * 78)
        print("WARNING: running on CPU. Claims 1 and 3's speed halves are NOT measurable here.")
        print("This script is intended for a CUDA machine. Nothing it prints on CPU should be")
        print("quoted as a GPU result.")
        print("=" * 78)
    claim1_sort_leak()
    claim2_speedup_on_gpu()
    claim3_multi_gpu()


if __name__ == "__main__":
    main()
