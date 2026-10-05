"""``_drop_zeros`` eager against the library's jitted ``_compact``: first call, warm call, whole build.

Fixture as ``poc/sparse/prune.py``. Each ``(pattern, size, arm)`` runs in a fresh subprocess, so ``first``
pays every compile; ``warm`` is the min of ``--rounds`` further calls on the same inputs. Both sync once for
the count, and the eager arm must return the library's arrays exactly.

Run: uv run python poc/sparse/drop_jit.py [--patterns type1 type2] [--log2-sizes 17 19] [--rounds 5]
"""

import argparse
import json
import os
import subprocess
import sys
import time

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # poc/
from eigenpair_check_scale import hamming_shells, patterns, xxz

import rqutils.sqd._sparse as sm
from rqutils.paulis.symplectic import PauliSumXZ
from rqutils.sqd import Matvec, uniquify_states
from rqutils.sqd._core import _sqd_inputs

parser = argparse.ArgumentParser()
parser.add_argument("--num-qubits", type=int, default=60)
parser.add_argument("--delta", type=float, default=0.5)
parser.add_argument("--patterns", nargs="+", default=["type1", "type2"])
parser.add_argument("--log2-sizes", type=int, nargs="+", default=[17, 19])
parser.add_argument("--rounds", type=int, default=5)
parser.add_argument("--child", nargs=3, metavar=("PATTERN", "LOG2", "ARM"), help=argparse.SUPPRESS)
options = parser.parse_args()


def drop_eager(t, s, d, chunk, size):
    """The library's ``_drop_zeros`` before ``_compact``: the same ops, dispatched one at a time."""
    keep = (d != 0).ravel()
    count = int(keep.sum())
    length = sm._size_class(-(-count // chunk)) * chunk
    (idx,) = jnp.nonzero(keep, size=length, fill_value=0)
    live = jnp.arange(length) < count
    pad = jnp.arange(length, dtype=t.dtype) % size
    t, s = (jnp.where(live, a.ravel()[idx], pad).reshape(-1, chunk) for a in (t, s))
    return t, s, jnp.where(live, d.ravel()[idx], 0).reshape(-1, chunk)


def child(pattern, log2, arm):
    n = options.num_qubits
    ham = PauliSumXZ.from_paulisum(xxz(n, options.delta, *patterns(n)[pattern]))
    states = hamming_shells(n, 1 << int(log2), np.random.default_rng(0))
    h, states_p, size = _sqd_inputs(ham, states, None, False, Matvec.PAIRS, 0.0, None, (32, 2))
    states_u = uniquify_states(states_p, size)
    pairs = sm._group_pairs(h, states_u)
    library = sm._drop_zeros
    fn = library if arm == "jit" else drop_eager
    seen = []

    def timed(*args):
        t0 = time.perf_counter()
        out = jax.block_until_ready(fn(*args))
        seen.append((time.perf_counter() - t0, args, out))
        return out

    sm._drop_zeros = timed  # ty: ignore[invalid-assignment]
    t0 = time.perf_counter()
    jax.block_until_ready(sm._sparse_operator(h, states_u, pairs))
    build = time.perf_counter() - t0
    first, args, out = seen[0]
    warm = []
    for _ in range(options.rounds):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        warm.append(time.perf_counter() - t0)
    ref = library(*args)
    assert all(np.array_equal(a, b) for a, b in zip(out, ref, strict=True)), "arms differ"
    print(json.dumps({"first": first, "warm": min(warm), "build": build, "entries": out[0].size}))


if options.child:
    child(*options.child)
    sys.exit()

print(
    "| pattern, N | entries kept | eager first / warm | jit first / warm | first build eager / jit |"
)
print("| --- | --- | --- | --- | --- |")
for pattern in options.patterns:
    for log2 in options.log2_sizes:
        r = {}
        for arm in ("eager", "jit"):
            cmd = [sys.executable, __file__, "--rounds", str(options.rounds)]
            cmd += ["--num-qubits", str(options.num_qubits), "--delta", str(options.delta)]
            out = subprocess.run(
                [*cmd, "--child", pattern, str(log2), arm],
                capture_output=True,
                text=True,
                check=False,
            )
            if out.returncode:
                sys.exit(out.stderr)
            r[arm] = json.loads(out.stdout.strip().splitlines()[-1])
        e, j = r["eager"], r["jit"]
        print(
            f"| {pattern} 2^{log2} | {e['entries']} | {e['first'] * 1e3:.0f} / {e['warm'] * 1e3:.1f} ms "
            f"| {j['first'] * 1e3:.0f} / {j['warm'] * 1e3:.1f} ms "
            f"| {e['build']:.2f} / {j['build']:.2f} s |",
            flush=True,
        )
