"""The sparse build's search on the device against the host: is ``get_xsource`` on a GPU the build lever?

With the tuned kernels a GH200 ``"pairs"`` call is 42% host build at ``type2`` ``2^22``, nearly all the
host search (``poc/sparse/gpu.md`` §8). The host search exists because XLA runs ``get_xsource`` on one
CPU core (``NOTES.md``, "sqd sparse kernels: the residual check runs on the host"); a GPU runs it wide.
Arms, each producing ``_group_pairs``' pairs for every group but the identity:

- ``host``: ``_search_pairs``, the build's search.
- ``device``: ``get_xsource`` per group on the device, each group's sources copied back and reduced to
  pairs on the host as ``_search_pairs`` does.
- ``device-all``: every group in one jitted ``scan``, one ``(J, N)`` copy back, then the same reduction.

Every arm must return the host's pairs exactly. One warm-up, then the median of ``--repeats``; ``copy``
is ``device-all``'s device-to-host transfer alone. On a CPU the device arms run XLA's one-core search, so
only a GPU run answers the question. Fixture as ``poc/sparse/gpu.py``.

Run: uv run python poc/sparse/device_search.py [--patterns type1 type2] [--log2-sizes 20 22]
"""

import argparse
import os
import statistics
import sys
import time

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)  # poc/, for its fixtures
from eigenpair_check_scale import hamming_shells, patterns, xxz

import rqutils.sqd._sparse as sm
from rqutils.paulis.symplectic import PauliSumXZ
from rqutils.sqd import Matvec, get_xsource, uniquify_states
from rqutils.sqd._core import _sqd_inputs

parser = argparse.ArgumentParser()
parser.add_argument("--num-qubits", type=int, default=60)
parser.add_argument("--delta", type=float, default=0.5)
parser.add_argument("--patterns", nargs="+", default=["type1", "type2"])
parser.add_argument("--log2-sizes", type=int, nargs="+", default=[20, 22])
parser.add_argument("--repeats", type=int, default=3)
options = parser.parse_args()


def reduce(j, rows):
    """``_search_pairs``' reduction: each pair once, ``j > i``."""
    keep = j > rows
    return rows[keep], j[keep]


search_one = jax.jit(get_xsource)


@jax.jit
def search_all(x, states_u):
    return jax.lax.scan(lambda _, xg: (None, get_xsource(xg, states_u)), None, x)[1]


def host(x, states_u):
    return sm._search_pairs(np.asarray(x), states_u)


def device(x, states_u):
    rows = np.arange(states_u.shape[0], dtype=np.int32)
    return [reduce(np.asarray(search_one(xg, states_u)), rows) for xg in x]


def device_all(x, states_u):
    rows = np.arange(states_u.shape[0], dtype=np.int32)
    sources = np.asarray(jax.block_until_ready(search_all(x, states_u)))
    return [reduce(j, rows) for j in sources]


ARMS = {"host": host, "device": device, "device-all": device_all}

print(f"{jax.devices()[0].device_kind}, n={options.num_qubits}")
print("pattern N    groups | host (s) | device (s)  x | device-all (s)  x | copy (s) | pairs")
for pattern in options.patterns:
    ham = PauliSumXZ.from_paulisum(
        xxz(options.num_qubits, options.delta, *patterns(options.num_qubits)[pattern])
    )
    for log2 in options.log2_sizes:
        states = hamming_shells(options.num_qubits, 1 << log2, np.random.default_rng(0))
        h, states_p, size = _sqd_inputs(ham, states, None, False, Matvec.PAIRS, 0.0, None, (32, 2))
        states_u = jax.block_until_ready(uniquify_states(states_p, size))
        first = int(not np.asarray(h.x[0]).any())
        x = jax.numpy.asarray(np.asarray(h.x)[first:])  # every group but a leading identity
        ref = host(x, states_u)
        for name, fn in ARMS.items():
            got = fn(x, states_u)  # warm-up, and the check
            assert len(got) == len(ref), name
            for g, ((a, b), (c, d)) in enumerate(zip(got, ref, strict=True)):
                assert np.array_equal(a, c) and np.array_equal(b, d), (name, g)
        times = {name: [] for name in ARMS}
        copies = []
        for _ in range(options.repeats):
            for name, fn in ARMS.items():
                t0 = time.perf_counter()
                fn(x, states_u)
                times[name].append(time.perf_counter() - t0)
            sources = jax.block_until_ready(search_all(x, states_u))
            t0 = time.perf_counter()
            np.asarray(sources)
            copies.append(time.perf_counter() - t0)
        med = {name: statistics.median(v) for name, v in times.items()}
        pairs = sum(len(i) for i, _ in ref)
        print(
            f"{pattern} 2^{log2} {x.shape[0]:6} | {med['host']:8.3f} |"
            f" {med['device']:8.3f} {med['host'] / med['device']:5.2f}x |"
            f" {med['device-all']:8.3f} {med['host'] / med['device-all']:5.2f}x |"
            f" {statistics.median(copies):8.3f} | {pairs}",
            flush=True,
        )
