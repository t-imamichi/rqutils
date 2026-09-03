"""POC 9: does ``ground_locg``'s Chebyshev prefilter still pay on a GPU? NOT VERIFIED on CPU-only hardware.

``docs/locg-chebyshev-prefilter.md`` measures the prefilter at a median **1.36x** over 18 XXZ
configurations, on an Apple M1 with ``jax.default_backend() == "cpu"``, one device. That number must
not be quoted as a GPU number. POC 8 states the rule this script inherits: *a result measured on one
backend says nothing about another, in either direction* -- a flat CPU result does not mean a change is
worthless, and a large one does not mean it carries over.

**Why the direction is genuinely uncertain here, rather than merely unmeasured.** The prefilter trades
matrix-vector products for LOBPCG iterations: ``(16, 4)`` spends ~79 extra matvecs to remove roughly
half the iterations. Whether that is a win depends on the cost *ratio* between a matvec and the ~40
``O(N)`` vector operations an iteration performs around it, and that ratio is exactly what changes
between backends. On CPU the matvec measured only ~25% of an iteration, so trading matvecs for
iterations is cheap. A GPU has far more arithmetic throughput relative to its memory bandwidth, and
``apply_h``'s matvec is a gather-heavy irregular kernel while the LOBPCG bookkeeping is
bandwidth-bound streaming -- so **the ratio can move either way**, and with it the sign of the trade.

Three claims to settle:

**Claim 1: the iteration reduction is backend-independent.** It should be, since it is arithmetic:
the same filtered vector goes into the same iteration. If the iteration counts here differ from CPU by
more than a couple of iterations, something is wrong with the port, not with the GPU -- suspect
precision (``jax_enable_x64`` is set below; without it ``ground_locg``'s dtype-derived tolerance
loosens by ~9 orders and the counts are meaningless).

**Claim 2: the wall-clock speedup on GPU.** Unknown. Reported here per configuration with
``fmt_ratio``, which refuses to call a difference inside the measured spread a win. A result under the
noise floor is the honest answer, not a failure of the script.

**Claim 3: it stays sharding-transparent on real devices.** ``tests/_sharded_prefilter.py`` verifies
the output *spec* is preserved across 1/2/4 virtual devices and both partitioned and replicated
inputs, but virtual devices share one physical backend -- per ``CLAUDE.md``, timings under them are
meaningless and only correctness transfers. This script asserts the spec on real devices and reports
the multi-device timing separately.

Run on a CUDA box::

    uv run --extra qiskit python examples/scaling/poc9_prefilter_gpu.py
    uv run --extra qiskit python examples/scaling/poc9_prefilter_gpu.py --devices 0,1,2,3
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

parser = argparse.ArgumentParser()
parser.add_argument("--devices", help='Comma-separated GPU ids, e.g. "0,1,2,3".')
parser.add_argument("--num-qubits", type=int, default=26)
parser.add_argument("--num-states", type=int, default=1_000_000)
parser.add_argument("--num-xgroups", type=int, default=30)
parser.add_argument(
    "--degrees",
    default="8,16,32",
    help="Comma-separated filter degrees to sweep.",
)
parser.add_argument("--cycles", default="2,4,8", help="Comma-separated cycle counts to sweep.")
options = parser.parse_args()

# CUDA_VISIBLE_DEVICES / XLA_FLAGS / jax.distributed.initialize all have to precede backend
# initialization, so device setup is deferred to init_devices, called first thing in main().
import functools

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
from _scaling_common import fmt_ratio, header, init_devices, make_1d_mesh, make_problem, timeit
from jax.sharding import PartitionSpec

from rqutils.ground_locg import ground_locg
from rqutils.sqd import apply_h, get_diagonal, get_xsource, uniquify_states

DEGREES = tuple(int(x) for x in options.degrees.split(","))
CYCLES = tuple(int(x) for x in options.cycles.split(","))


def assemble(problem):
    """Build the cached matvec the prefilter will be measured against.

    ``cache_level=(1, 2)`` equivalent: X sources and diagonals both precomputed, which is
    ``DiagCache.SPEED`` and the level a real SKQD run uses. Assembled by hand rather than through
    ``sqd()`` so the solve can be timed without the setup, since the prefilter only affects the solve.
    """
    hamiltonian, states = problem.hamiltonian, problem.states_p
    packed = uniquify_states(states, 1 << int(np.ceil(np.log2(states.shape[0]))))
    arrays = hamiltonian.arrays
    sources = jax.numpy.stack(
        [get_xsource(arrays.x[group], packed) for group in range(arrays.x.shape[0])]
    )
    diagonals = jax.numpy.stack(
        [
            get_diagonal(arrays.z[group], arrays.c[group], packed)
            for group in range(arrays.x.shape[0])
        ]
    )
    return sources, diagonals, packed.shape[0]


def main():
    print(f"devices: {init_devices(options.devices)}")
    if jax.default_backend() == "cpu":
        print(
            "\n*** THIS IS A CPU RUN. Every number below is already in "
            "docs/locg-chebyshev-prefilter.md; this script exists to be run on a GPU. ***"
        )

    problem = make_problem(
        options.num_qubits,
        options.num_states,
        num_terms=100,
        num_xgroups=options.num_xgroups,
        real_only=True,
    )
    header(f"Prefilter sweep: n={options.num_qubits} J={options.num_xgroups}")
    print("Claim 1 (iteration counts) must match the CPU figures; Claim 2 (wall clock) is unknown.")

    sources, diagonals, dim = assemble(problem)
    matvec = functools.partial(apply_h, xsources=sources, diagonals=diagonals)
    # A callable mat cannot yield a bound on lambda_max from matvecs alone (Kuczynski-Wozniakowski),
    # so ground_locg requires it explicitly. sum|c_k| is the bound sqd itself passes; loose is safe.
    prefilter_hi = float(np.abs(problem.hamiltonian.c).sum())
    rng = np.random.default_rng(0)
    xinit = jax.numpy.asarray(rng.normal(size=dim))
    xinit = xinit / jax.numpy.linalg.norm(xinit)

    baseline = timeit(lambda: jax.block_until_ready(ground_locg(matvec, xinit)), "plain")
    plain = ground_locg(matvec, xinit)
    # Keep only the scalars: an O(N) eigenvector per config has no reason to outlive the number it
    # is reduced to. This alone did NOT fix the GPU OOM below -- freeing eight vectors left the
    # failure at the same config -- so it is hygiene, not the cure.
    reference, plain_iters = float(plain[0]), int(plain[2])
    del plain
    print(
        f"\nN(padded)={dim}  plain: {baseline.min_s * 1e3:.1f}ms  {plain_iters} iters  E={reference:.12f}"
    )
    print(f"{'degree':>7} {'cycles':>7} {'extra mv':>9} {'iters':>6} {'ms':>9} {'|dE|':>10}  ratio")
    for degree in DEGREES:
        for cycles in CYCLES:
            prefilter = (degree, cycles)
            # Each (degree, cycles) is static, so ground_locg retraces and the process retains one
            # more CUBIN per config. On a 71 GB GPU the 8th config failed to *load its module*
            # (RESOURCE_EXHAUSTED / "Failed to load in-memory CUBIN"), not to allocate a tensor.
            # Measured: HLO size and temp_size_in_bytes are identical across all nine configs
            # (1836504 B; degree and cycles are lax.scan trip counts and change neither), so the
            # boundary is cumulative executables, not this config's demand. Hence clear the cache.
            jax.clear_caches()
            # Per-config isolation: one OOM used to abort the sweep, so the configs after the
            # failure were reported as nothing at all rather than as unrun. A skipped row is data.
            try:
                result = ground_locg(matvec, xinit, prefilter=prefilter, prefilter_hi=prefilter_hi)
                iters, dE = int(result[2]), abs(float(result[0]) - reference)
                del result
                timing = timeit(
                    lambda p=prefilter: jax.block_until_ready(
                        ground_locg(matvec, xinit, prefilter=p, prefilter_hi=prefilter_hi)
                    ),
                    f"({degree},{cycles})",
                )
            except jax.errors.JaxRuntimeError as exc:
                first = str(exc).strip().splitlines()[0][:60]
                print(f"{degree:>7} {cycles:>7} {'':>9} {'--':>6} {'SKIPPED':>9}  {first}")
                continue
            # ~11 matvecs for the lambda_max estimate, plus cycles*(degree+1) for the filter.
            extra = cycles * (degree + 1) + 11
            print(
                f"{degree:>7} {cycles:>7} {extra:>9} {iters:>6} "
                f"{timing.min_s * 1e3:>9.2f} {dE:>10.1e}"
                f"  {fmt_ratio(baseline, timing)}"
            )

    if len(jax.devices()) < 2:
        print("\nSingle device: skipping the sharded arm. Re-run with --devices to reach Claim 3.")
        return

    header("Claim 3: sharding on real devices")
    mesh = make_1d_mesh()
    with jax.sharding.set_mesh(mesh):
        sharded_sources = jax.device_put(
            sources, jax.sharding.NamedSharding(mesh, PartitionSpec(None, "x"))
        )
        sharded_diagonals = jax.device_put(
            diagonals, jax.sharding.NamedSharding(mesh, PartitionSpec(None, "x"))
        )

        # Multi-process: a functools.partial closing over these arrays raises "Closing over jax.Array
        # that spans non-addressable (non process local) devices". Each rank addresses only its own
        # shard, so a globally-sharded array must arrive as an *argument*. ground_locg splats
        # matvec(vec, *args), so `args` is that channel -- and apply_h being keyword-only (CLAUDE.md)
        # is why this takes a positional adapter rather than another partial.
        def sharded_matvec(vec, xsources, diagonals):
            return apply_h(vec, xsources=xsources, diagonals=diagonals)

        margs = (sharded_sources, sharded_diagonals)
        sharded_init = jax.device_put(xinit, jax.sharding.NamedSharding(mesh, PartitionSpec("x")))
        for label, prefilter in (("plain", None), ("prefiltered", (16, 4))):
            result = ground_locg(
                sharded_matvec,
                sharded_init,
                args=margs,
                prefilter=prefilter,
                prefilter_hi=prefilter_hi,
            )
            timing = timeit(
                lambda p=prefilter: jax.block_until_ready(
                    ground_locg(
                        sharded_matvec,
                        sharded_init,
                        args=margs,
                        prefilter=p,
                        prefilter_hi=prefilter_hi,
                    )
                ),
                label,
            )
            spec = jax.typeof(result[1]).sharding.spec
            # Assert rather than print: a silently replicated output agrees with single-device to
            # exactly 0.0, so only the spec distinguishes sharded from accidentally-not.
            assert spec == PartitionSpec("x"), f"{label} lost its sharding: {spec}"
            print(
                f"  {label:>12}: {timing.min_s * 1e3:>8.2f}ms  {int(result[2]):>4} iters  "
                f"spec={spec}  |dE|={abs(float(result[0]) - reference):.1e}"
            )


if __name__ == "__main__":
    main()
