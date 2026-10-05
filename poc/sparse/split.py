"""A/B: the sparse kernels' scan carrying ``out`` as real and imaginary parts, against one complex carry.

``poc/sparse/kernel_profile.py`` on the GH200 put 75-88% of a ``"pairs"`` matvec from ``2^20`` in three kernels
(``wrapped_real``/``wrapped_imag``/``wrapped_complex``) run once per scan step: XLA's GPU scatter splits
a ``complex128`` carry and rejoins it every step, a full pass over ``out`` each. ``split`` is the library
kernel, whose ``_scan_add`` carries the two parts itself; ``complex`` is the previous kernel, copied
verbatim below. The split is CUDA-only (``jax.lax.platform_dependent``), so on CPU the two arms run the
same carry and must agree bit for bit and time alike (``poc/sparse/split.md``).

Each arm gets a function of its own to jit (``legacy.APPLY`` is read at trace time, and a second jit of
one function reuses the first's trace), and the lowered solves are asserted to differ. Arms are warm and
interleaved; ``solve`` is per iteration, since GPU scatter order makes iteration counts vary. ``temp`` is
XLA's ``temp_size_in_bytes`` for the ``(2, N)`` matvec. Fixture as ``poc/sparse/gpu.py``.

Run: uv run python poc/sparse/split.py [--log2-sizes 20 21 22] [--arms pairs csr ell] [--rounds 5]
"""

import argparse
import functools
import os
import statistics
import sys
import time

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)  # poc/, for its fixtures
import legacy  # "csr"/"ell", removed from the library
from eigenpair_check_scale import hamming_shells, patterns, xxz

import rqutils.sqd._solve as solve_mod
from rqutils.paulis.symplectic import PauliSumXZ
from rqutils.sqd import Matvec, uniquify_states
from rqutils.sqd._core import _sqd_inputs

parser = argparse.ArgumentParser()
parser.add_argument("--num-qubits", type=int, default=60)
parser.add_argument("--pattern", default="type1")
parser.add_argument("--delta", type=float, default=0.5)
parser.add_argument("--log2-sizes", type=int, nargs="+", default=[20, 21, 22])
parser.add_argument("--arms", nargs="+", default=["pairs", "csr", "ell"])
parser.add_argument("--rounds", type=int, default=5)
options = parser.parse_args()

ITERATIONS = []
_ground_locg = solve_mod.ground_locg


def counted_ground_locg(*args, **kwargs):
    """``ground_locg``, recording each solve's iteration count on the host."""
    out = _ground_locg(*args, **kwargs)
    jax.debug.callback(lambda niter: ITERATIONS.append(int(niter)), out[2])
    return out


solve_mod.ground_locg = counted_ground_locg  # ty: ignore[invalid-assignment]


# The kernels as of 474aafb, one complex carry each.
def apply_pairs(vec, d0, pi, pj, d):
    sharding = jax.typeof(vec).sharding

    def body(out, chunk):
        i, j, di = chunk
        out = out.at[..., i].add(di * vec.at[..., j].get(out_sharding=sharding))
        return out.at[..., j].add(jnp.conj(di) * vec.at[..., i].get(out_sharding=sharding)), None

    return jax.lax.scan(body, d0 * vec, (pi, pj, d))[0]


def apply_csr(vec, d0, *entries):
    sharding = jax.typeof(vec).sharding

    def body(acc, chunk):
        ti, si, di = chunk
        gathered = vec.at[..., si].get(out_sharding=sharding)
        return acc.at[..., ti].add(di * gathered, indices_are_sorted=True), None

    out = d0 * vec
    for k in range(0, len(entries), 3):
        out = jax.lax.scan(body, out, entries[k : k + 3])[0]
    return out


def apply_ell(vec, d0, *buckets):
    sharding = jax.typeof(vec).sharding

    def body(acc, piece):
        rows, src, fac = piece
        val = jnp.sum(fac * vec.at[..., src].get(out_sharding=sharding), axis=-1)
        return acc.at[..., rows].add(val), None

    out = d0 * vec
    for k in range(0, len(buckets), 3):
        out = jax.lax.scan(body, out, buckets[k : k + 3])[0]
    return out


KERNELS = {
    "split": dict(legacy.APPLY),
    "complex": {"pairs": apply_pairs, "csr": apply_csr, "ell": apply_ell},
}


def timed(fn):
    t0 = time.perf_counter()
    out = jax.block_until_ready(fn())
    return time.perf_counter() - t0, out


ham = PauliSumXZ.from_paulisum(
    xxz(options.num_qubits, options.delta, *patterns(options.num_qubits)[options.pattern])
)
print(f"{jax.devices()[0].device_kind}, n={options.num_qubits} {options.pattern}")
print(
    "arm   N    | complex / split (ms), speedup, split wins: solve per iter | 1-D matvec |"
    " (2, N) matvec | temp complex / split (MiB) | iters complex / split"
)
for log2 in options.log2_sizes:
    states = hamming_shells(options.num_qubits, 1 << log2, np.random.default_rng(0))
    for name in options.arms:
        arm = name
        h, states_p, size = _sqd_inputs(ham, states, None, False, Matvec.PAIRS, 0.0, None, (32, 2))
        states_u = uniquify_states(states_p, size)
        operator = jax.block_until_ready(legacy.operator(h, states_u, arm))
        vecs = [
            jax.random.normal(jax.random.key(0), shape, jnp.complex128)
            for shape in ((size,), (2, size))
        ]
        calls, lowered, temp, products = {}, {}, {}, {}
        for kind, table in KERNELS.items():
            legacy.APPLY[arm] = table[arm]
            solve = jax.jit(lambda *a: legacy.run.__wrapped__(*a), static_argnums=(3, 4, 5))
            lowered[kind] = solve.lower(h, states_u, operator, size, False, arm).as_text()
            matvec = jax.jit(table[arm])
            calls[kind] = [functools.partial(solve, h, states_u, operator, size, False, arm)]
            calls[kind] += [functools.partial(matvec, vec, *operator) for vec in vecs]
            compiled = matvec.lower(vecs[1], *operator).compile()
            temp[kind] = compiled.memory_analysis().temp_size_in_bytes / 2**20
            for call in calls[kind]:
                call()  # compile
            products[kind] = np.asarray(calls[kind][2]())
        legacy.APPLY[arm] = KERNELS["split"][arm]
        assert lowered["split"] != lowered["complex"], "both solve arms compiled the same kernel"
        ref = products["complex"]
        err = np.abs(products["split"] - ref).max() / np.abs(ref).max()
        times = {kind: ([], [], []) for kind in KERNELS}
        eig, iters = [], {kind: set() for kind in KERNELS}
        for _ in range(options.rounds):
            for kind in KERNELS:
                ITERATIONS.clear()
                t, result = timed(calls[kind][0])
                times[kind][0].append(t / ITERATIONS[-1])
                for k in (1, 2):
                    times[kind][k].append(timed(calls[kind][k])[0])
                eig.append(float(result.eigval))
                iters[kind].add(ITERATIONS[-1])
        spread = max(eig) - min(eig)
        assert spread < 1e-12 * abs(eig[0]), eig
        cols = []
        for old, new in zip(times["complex"], times["split"], strict=True):
            a, b = statistics.median(old), statistics.median(new)
            wins = sum(n < o for o, n in zip(old, new, strict=True))
            cols.append(f"{a * 1e3:7.2f} / {b * 1e3:7.2f} {a / b:5.2f}x {wins}/{options.rounds}")
        print(
            f"{name:5} 2^{log2} | {' | '.join(cols)} | {temp['complex']:.1f} / {temp['split']:.1f}"
            f" | {sorted(iters['complex'])} / {sorted(iters['split'])}"
            f"  (matvec rel diff {err:.1e}, eigval spread {spread:.1e})",
            flush=True,
        )
