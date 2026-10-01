"""Per-kernel time of one ``"pairs"`` matvec, from a ``jax.profiler`` trace: where the GPU cliff goes.

``poc/sparse-tiles.md`` §3 found the GH200's ``"pairs"`` cost insensitive to entry order, so the cost is
not locality; this splits it by XLA kernel (gather, scatter, the ``scan``'s per-step work) at each size,
for a 1-D and a ``(2, N)`` vector. Per size it prints the entry and scan-step counts, then per shape the
device time per call, kernel launches per call, and the top kernels by time.

Device events are read from the trace's ``/device:`` processes; on a CPU-only run there are none, so it
falls back to host XLA ops (a smoke test, not a measurement). Fixture as ``poc/sparse_gpu.py``.

Run: uv run python poc/sparse_profile.py [--log2-sizes 19 20 21 22] [--calls 10] [--top 10]
"""

import argparse
import collections
import glob
import gzip
import json
import os
import sys
import tempfile

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from eigenpair_check_scale import hamming_shells, patterns, xxz

from rqutils.paulis.symplectic import PauliSumXZ
from rqutils.sqd import Matvec, uniquify_states
from rqutils.sqd._core import _sqd_inputs
from rqutils.sqd._sparse import _apply_pairs, _sparse_operator

parser = argparse.ArgumentParser()
parser.add_argument("--num-qubits", type=int, default=60)
parser.add_argument("--pattern", default="type1")
parser.add_argument("--delta", type=float, default=0.5)
parser.add_argument("--log2-sizes", type=int, nargs="+", default=[19, 20, 21, 22])
parser.add_argument("--calls", type=int, default=10)
parser.add_argument("--top", type=int, default=10)
options = parser.parse_args()


def kernel_times(trace_dir):
    """``{kernel: (total µs, count)}`` over device events, or host XLA ops when there is no device."""
    (path,) = glob.glob(f"{trace_dir}/**/*.trace.json.gz", recursive=True)
    with gzip.open(path) as fh:
        events = json.load(fh)["traceEvents"]
    names = {
        e["pid"]: e["args"]["name"]
        for e in events
        if e.get("ph") == "M" and e.get("name") == "process_name"
    }
    device = {pid for pid, name in names.items() if name.startswith("/device:")}
    out = collections.defaultdict(lambda: [0.0, 0])
    for e in events:
        if e.get("ph") != "X":
            continue
        if device and e["pid"] not in device:
            continue
        if not device and (e["name"].startswith("$") or "::" in e["name"] or "(" in e["name"]):
            continue
        out[e["name"]][0] += e["dur"]
        out[e["name"]][1] += 1
    return out, bool(device)


ham = PauliSumXZ.from_paulisum(
    xxz(options.num_qubits, options.delta, *patterns(options.num_qubits)[options.pattern])
)
arm = Matvec.PAIRS
matvec = jax.jit(_apply_pairs)
print(f"{jax.devices()[0].device_kind}, n={options.num_qubits} {options.pattern}, matvec=pairs")
for log2 in options.log2_sizes:
    states = hamming_shells(options.num_qubits, 1 << log2, np.random.default_rng(0))
    h, states_p, size = _sqd_inputs(ham, states, None, False, arm, 0.0, None, (32, 2))
    operator = jax.block_until_ready(_sparse_operator(h, uniquify_states(states_p, size), arm))
    pi, pj = (np.asarray(a) for a in operator[1:3])
    real = int(np.count_nonzero(pi != pj))
    print(
        f"\n2^{log2}: {real} pairs ({real / size:.1f}/state), {pi.shape[0]} scan steps of {pi.shape[1]}"
    )
    for shape in ((size,), (2, size)):
        vec = jax.random.normal(jax.random.key(0), shape, jnp.complex128)
        jax.block_until_ready(matvec(vec, *operator))  # compile
        trace_dir = tempfile.mkdtemp()
        with jax.profiler.trace(trace_dir):
            for _ in range(options.calls):
                jax.block_until_ready(matvec(vec, *operator))
        kernels, on_device = kernel_times(trace_dir)
        total = sum(t for t, _ in kernels.values())
        launches = sum(n for _, n in kernels.values())
        source = "device" if on_device else "host XLA ops (no device trace)"
        print(
            f"  {'1-D' if len(shape) == 1 else '(2, N)'}: {total / options.calls / 1e3:.2f} ms/call"
            f" {source}, {launches / options.calls:.0f} launches/call"
        )
        for name, (t, n) in sorted(kernels.items(), key=lambda kv: -kv[1][0])[: options.top]:
            print(
                f"    {t / options.calls / 1e3:8.3f} ms  {100 * t / total:5.1f}%"
                f"  {n / options.calls:6.0f}/call  {name[:70]}"
            )
