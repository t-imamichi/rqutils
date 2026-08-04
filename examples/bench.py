"""Benchmark svsim and sqd across JAX backends.

Compares wall-clock time of the two main kernels on whichever backend JAX selects.
Run once per backend and diff the numbers:

.. code-block:: sh

    JAX_PLATFORMS=cpu   python examples/bench.py 14
    JAX_PLATFORMS=metal python examples/bench.py 14   # requires a working jax-metal

Compile time is reported separately from steady-state run time, since the first
call pays for JIT tracing and lowering.

Note that ``--x64`` (the default, and what the other examples use) is
incompatible with the Metal backend, which supports neither float64 nor
complex128. Metal runs therefore require ``--no-x64`` and are not
numerically comparable to the x64 CPU baseline.
"""

import argparse
import time

import jax
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("num_qubits", type=int, help="Number of qubits.")
parser.add_argument("--repeat", type=int, default=5, help="Timed iterations after warmup.")
parser.add_argument(
    "--num-paulis", type=int, default=100, help="Pauli terms for the sqd benchmark."
)
parser.add_argument("--subspace-frac", type=float, default=0.01, help="Subspace fraction for sqd.")
parser.add_argument(
    "--no-x64",
    dest="x64",
    action="store_false",
    help="Disable float64/int64 (required by the Metal backend).",
)
parser.add_argument("--skip-sqd", action="store_true", help="Benchmark svsim only.")
options = parser.parse_args()

jax.config.update("jax_enable_x64", options.x64)

print(f"backend={jax.default_backend()} devices={jax.devices()} x64={options.x64}")

from rqutils.svsim import svsim, to_circuitxz


def timeit(fn, repeat):
    """Return (compile_seconds, mean_steady_seconds)."""
    start = time.perf_counter()
    jax.block_until_ready(fn())
    compile_time = time.perf_counter() - start

    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        jax.block_until_ready(fn())
        times.append(time.perf_counter() - start)
    return compile_time, float(np.mean(times))


# svsim: a GHZ-like chain of native gates, built without qiskit so the benchmark
# has no optional dependency.
gates = [("ry", [0], np.pi / 2.0)]
for iq in range(options.num_qubits - 1):
    gates.extend([("rzz", [iq, iq + 1], np.pi / 2.0), ("rx", [iq + 1], np.pi / 2.0)])
circuit = to_circuitxz(gates)

compile_s, run_s = timeit(lambda: svsim(circuit), options.repeat)
print(f"svsim  n={options.num_qubits} gates={len(gates)} compile={compile_s:.4f}s run={run_s:.6f}s")

if not options.skip_sqd:
    from rqutils.sqd import sqd

    rng = np.random.default_rng(0)
    paulis = rng.choice(["I", "X", "Y", "Z"], size=(options.num_paulis, options.num_qubits))
    pauli_strings = ["".join(row) for row in paulis]
    coeffs = rng.uniform(-1.0, 1.0, options.num_paulis)
    num_samples = max(2, int(2**options.num_qubits * options.subspace_frac))
    states = rng.choice(2, size=(num_samples, options.num_qubits)).astype(np.uint8)

    compile_s, run_s = timeit(
        lambda: sqd((pauli_strings, coeffs), states, return_eigvec=False), options.repeat
    )
    print(
        f"sqd    n={options.num_qubits} terms={options.num_paulis} states={num_samples} "
        f"compile={compile_s:.4f}s run={run_s:.6f}s"
    )
