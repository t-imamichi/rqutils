"""POC 8: the claims that need a real GPU. NOT VERIFIED on the machine this was written on.

Everything else in this directory was measured on an Apple M1, ``jax.default_backend() == "cpu"``,
one device. Three claims cannot be reached from there, and this script exists so they can be settled
in one run on a CUDA box rather than re-derived.

**Both sort-related claims below are measured against ``poc1.xsource_sort_legacy``, a pinned copy of
the pre-23fb226 implementation -- not against ``rqutils.sqd.get_xsource``.** The library *is* the
searchsorted now, so it can no longer serve as its own baseline. The first GPU run of this script
predated that swap and compared the library against POC 1: two binary searches, reported as
1.002x/1.000x/1.000x, and a memory claim aimed at a ``lax.sort`` that had already been deleted. The
run was honest and informative about nothing. A POC whose baseline is "whatever the library does"
stops being a comparison the moment the library changes, which is the transferable lesson here.

**Claim 1: the ``lax.sort`` memory leak.** ``rqutils/sqd.py`` records "lax.sort seems to leak GPU
memory; can lose as much as 5 GB when sorting x of shape (5M,9)". That is an allocator behaviour of a
GPU backend; a CPU backend has no comparable accounting to observe. Since the sort is gone from the
library, this is now a claim about ``lax.sort`` itself, answered with the legacy arm -- and a flat
result is a finding about JAX, not about ``sqd``. Sample ``bytes_in_use`` while the result is still
live: an earlier version read it after ``del`` and watched ``peak_bytes_in_use`` (a high-water mark
that never decreases), so neither number it printed could move regardless of the truth.

**Claim 2: POC 1's speedup on GPU.** Measured 12-25x on CPU, and reproduced here at 12.1x/18.3x
(N=100k/200k, J=50) with the legacy arm restored, matching 23fb226's recorded 12.1x/18.7x. The
direction should hold on GPU, since a binary-search gather is strictly less work than a sort, but the
magnitude will differ: a GPU sort is far better optimized relative to its gather than a CPU one, so
**expect the GPU speedup to be smaller**. Quoting the CPU number as a GPU number would be exactly the
error this repo has hit before: a result measured on one backend says nothing about another, in either
direction -- a flat CPU result does not mean a change is worthless, and a large one does not mean it
carries over. The sort arm is asserted to grow with ``N``; if it goes flat the
measurement is not kernel-dominated and the ratio is suppressed rather than printed.

**Claim 3: real multi-GPU sharding.** ``poc7_sharding.py`` validated the sharding code paths on
virtual CPU devices and, in doing so, found and fixed a real bug (a scatter in ``vinit_from_min_diag``
missing ``out_sharding``, which made ``sqd`` fail on *any* mesh). Virtual devices cannot speak to
interconnect cost, per-device memory limits, or whether the sharded solve is actually *faster*.
**Requires a physically multi-GPU machine.** ``--devices`` sets ``CUDA_VISIBLE_DEVICES``, which is a
filter over the devices the driver already exposes -- it cannot conjure a second GPU, so passing
``--devices 0,1,2,3`` on a one-GPU box correctly still reports one device and this claim is skipped.

Run on a CUDA machine:

    uv run --extra qiskit python examples/scaling/poc8_gpu_unverified.py
    # multi-GPU (needs >= 2 physical GPUs; verify with nvidia-smi -L first):
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
from poc1_searchsorted import FILL_BYTE, xsource_searchsorted_u64, xsource_sort_legacy

from rqutils.sqd import sqd, uniquify_states


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


def bytes_in_use():
    """Live ``bytes_in_use`` summed over devices, or None if the backend has no accounting.

    Sampled while the result of the op under test is **still referenced**. The previous version of
    this script printed stats after ``del out``, so it read the post-free baseline every repetition
    and no leak could possibly register; it then told the reader to watch ``peak_bytes_in_use``,
    which is a high-water mark that never decreases and pins after the first repetition. Both
    columns were incapable of moving, which is why the first GPU run showed flat memory and was
    misread as evidence against the in-tree note.
    """
    total = 0
    for dev in jax.devices():
        try:
            st = dev.memory_stats()
        except Exception:  # noqa: BLE001 - diagnostics must not crash the run
            return None
        if st is None:
            return None
        total += st.get("bytes_in_use", 0)
    return total


def claim1_sort_leak():
    header("CLAIM 1: does lax.sort leak GPU memory, and does searchsorted avoid it?")
    print(
        "Reproduces the note now recorded in get_xsource's docstring, as closely as shape allows:"
    )
    print(f"  n={options.num_qubits}, N={options.num_states} (note cites (5M, 9))")
    print()
    if jax.default_backend() == "cpu":
        print("  BACKEND IS CPU -- this claim is UNVERIFIABLE here. Skipping.")
        print("  memory_stats() has no GPU-style accounting on CPU, so any number printed would")
        print("  be meaningless rather than merely imprecise.")
        return

    # J must be large enough that the sort runs many times per sweep: a leak is cumulative, and the
    # previous num_xgroups=4 / one-call-per-rep version applied nowhere near enough pressure to
    # surface a 5 GB artifact.
    p = make_problem(options.num_qubits, options.num_states, num_terms=200, num_xgroups=50, seed=81)
    size = p.states_p.shape[0]
    states_u = jax.block_until_ready(uniquify_states(p.states_p, size))
    xs = p.hamiltonian.x
    print(f"  post-uniquification N={size}, B={states_u.shape[1]}, J={p.num_xgroups}")
    print()

    # The subject of this claim is lax.sort, which commit 23fb226 REMOVED from get_xsource. Timing
    # or profiling the library here would exercise a binary search and find no leak by construction,
    # so the arm under test is poc1's pinned legacy sort.
    def sweep(fn):
        return jax.lax.scan(lambda _, x: (None, fn(x, states_u)), None, xs)[1]

    device_memory_stats("baseline")
    base = bytes_in_use()
    if base is None:
        print("  No allocator accounting on this backend; claim unmeasurable here.")
        return

    for name, fn in [("sort", xsource_sort_legacy), ("ssorted", xsource_searchsorted_u64)]:
        print()
        first = None
        for rep in range(5):
            out = jax.block_until_ready(sweep(fn))
            live = bytes_in_use()  # sampled while `out` is still alive
            del out
            after = bytes_in_use()
            if first is None:
                first = live
            print(
                f"  [{name} rep{rep}] live={live / 2**30:7.3f}GB  "
                f"after_free={after / 2**30:7.3f}GB  "
                f"retained_vs_baseline={(after - base) / 2**30:+7.3f}GB  "
                f"drift_vs_rep0={(live - first) / 2**30:+7.3f}GB"
            )
    print()
    print(
        "  READ: 'retained_vs_baseline' rising across reps is the leak -- memory still held after"
    )
    print("  the result is dropped. 'drift_vs_rep0' rising means live footprint grows per sweep.")
    print("  Flat in both arms means the note is stale or backend-version-specific: report that,")
    print("  do not quietly drop it. Note the leak was reported against a lax.sort that no longer")
    print(
        "  exists in the library, so a flat 'sort' arm here is a finding about JAX, not about sqd."
    )


def claim2_speedup_on_gpu():
    header("CLAIM 2: POC 1's 12-25x CPU speedup -- what is it on GPU?")
    print("EXPECT A SMALLER NUMBER than the CPU measurement. A GPU sort is well optimized relative")
    print("to its gather, so the ratio should compress. The direction should still favour")
    print("searchsorted, since it is strictly less work.")
    print()
    print(
        "Baseline is poc1's xsource_sort_legacy, NOT rqutils.sqd.get_xsource. Commit 23fb226 made"
    )
    print("the library BE the searchsorted, so timing against it compares two binary searches: the")
    print("first GPU run of this script reported 1.002x/1.000x/1.000x for exactly that reason.")
    print()
    print(f"{'N':>10s}  {'J':>4s}  {'sort ms':>10s}  {'ssorted ms':>11s}  {'verdict':>34s}")
    # Sorted and de-duplicated: the growth check below compares each row to the previous one, so an
    # out-of-order list (--num-states below a hardcoded size) would compare against a LARGER N and
    # warn on a decrease. Caught exactly that way at --num-states 60000.
    prev = None
    sizes = sorted({200_000, 1_000_000, min(options.num_states, 5_000_000)})
    for num_states in sizes:
        p = make_problem(
            options.num_qubits, num_states, num_terms=200, num_xgroups=options.num_xgroups, seed=82
        )
        size = p.states_p.shape[0]
        states_u = jax.block_until_ready(uniquify_states(p.states_p, size))
        xs = p.hamiltonian.x

        def all_sort(states_u=states_u, xs=xs):
            return jax.lax.scan(lambda _, x: (None, xsource_sort_legacy(x, states_u)), None, xs)[1]

        def all_ss(states_u=states_u, xs=xs):
            return jax.lax.scan(
                lambda _, x: (None, xsource_searchsorted_u64(x, states_u)), None, xs
            )[1]

        # Same equivalence gate as POC 1: valid rows identical, gathers identical. Assert the gate
        # is non-vacuous -- an all-fill subspace would make it pass while comparing nothing.
        ref, got = np.asarray(all_sort()), np.asarray(all_ss())
        is_fill = np.asarray(states_u)[:, 0] == FILL_BYTE
        assert (~is_fill).sum() > 0, "no valid rows; the index gate would pass vacuously"
        assert np.array_equal(ref[:, ~is_fill], got[:, ~is_fill]), "valid-row index mismatch on GPU"

        t_sort = timeit(all_sort, "sort", trials=5)
        t_ss = timeit(all_ss, "ssorted", trials=5)
        print(
            f"{size:>10d}  {p.num_xgroups:>4d}  {t_sort.min_s * 1e3:>8.2f}ms  "
            f"{t_ss.min_s * 1e3:>9.2f}ms  {fmt_ratio(t_sort, t_ss):>34s}"
        )
        # The failure this guards against is a FLAT arm: the stale run showed 84/93/159ms across a
        # 25x N increase, the signature of a measurement dominated by dispatch rather than the
        # kernel. So the threshold tests for flatness, not for a particular exponent -- an N log N
        # sort measured 1.69x for a 2.0x N increase here, which is healthy, and an earlier
        # sqrt-based threshold flagged it as a problem. Anything at least half-linear is fine.
        if prev is not None:
            pn, pt = prev
            grew, ngrew = t_sort.min_s / pt, size / pn
            if grew < 0.5 * ngrew:
                print(
                    f"    WARNING: sort arm grew only {grew:.2f}x for a {ngrew:.1f}x N increase "
                    "-- measurement may not be kernel-dominated; corroborate before quoting."
                )
                # This fires reproducibly at J=50 on every GPU tried, and the answer is already
                # recorded rather than open: docs/scaling-pocs.md "Measured on GPU (NVIDIA GH200
                # 120GB)" diagnosed the identical pattern
                # on a GH200 (sort arm flat at 1141/1239/1201 ms) as launch-latency bound, making
                # the ratio a quotient of two overheads that lands misleadingly near the CPU
                # 12-25x. The quotable GPU figure comes from poc1's check_scaling --sweep-to, which
                # fits alpha and refuses ratios below 0.6: 5.15x at N=64M, rising with N.
                print(
                    "    -> Already diagnosed: launch-bound at J=50, NOT a new finding. See "
                    'docs/scaling-pocs.md "Measured on GPU"; quotable GPU number is 5.15x at '
                    "N=64M from "
                    "poc1_searchsorted.py check_scaling --sweep-to."
                )
        prev = (size, t_sort.min_s)


def claim3_multi_gpu():
    header("CLAIM 3: real multi-GPU sharding -- correctness AND speed")
    ndev = jax.device_count()
    print(f"  devices visible: {ndev} ({jax.devices()[0].platform})")
    if ndev < 2:
        print("  Fewer than 2 devices, so this claim did NOT run -- it is unrun, not unresolved.")
        print(
            "  CUDA_VISIBLE_DEVICES (what --devices sets) is a FILTER, not a request: it can only"
        )
        print("  narrow the devices the driver already exposes. Passing --devices 0,1,2,3 on a")
        print("  single-GPU box therefore still yields one device, which is not a failure of the")
        print("  flag. This claim needs a physically multi-GPU machine; poc7_sharding.py already")
        print("  covers sharding CORRECTNESS on virtual CPU devices, so only speed is missing.")
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
