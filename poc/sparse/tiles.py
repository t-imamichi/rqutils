"""A/B: ``"pairs"`` entry orders, the shipped sort by ``i`` against tiled and per-group orders.

``"pairs"`` sorts its ``(i, j)`` entries by ``i`` across groups, so ``vec[i]``/``out[i]`` stream while
``vec[j]``/``out[j]`` land anywhere. Arms, by sort key:

- ``i``: shipped.
- ``group``: ``(group, i)``, close to the order before ``d84c4a3``, where ``j`` also sweeps within a group.
- ``tileS``: ``(i >> S, j >> S, i)``, so consecutive chunks touch one ``2^S``-state slice per side.

Only the data order changes: one compiled solve serves every arm. ``matvec`` is the bare kernel on a
1-D and a ``(2, N)`` vector, since the solve runs both (the prefilter and ``body()``'s third matvec are
1-D). Each arm holds the same entries as ``i`` (asserted), its solve is timed per iteration (scatter
order changes rounding, so iteration counts can differ), arms are warm and interleaved, and eigenvalues
must agree to ``1e-12`` relative.
Fixture as ``poc/sparse/gpu.py``.

Run: uv run python poc/sparse/tiles.py [--log2-sizes 17 19] [--tiles 12 14 16] [--rounds 5]
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
from eigenpair_check_scale import hamming_shells, patterns, xxz

import rqutils.sqd._solve as solve_mod
import rqutils.sqd._sparse as sparse_mod
from rqutils.paulis.symplectic import PauliSumXZ
from rqutils.sqd import Matvec, uniquify_states
from rqutils.sqd._core import _sqd_inputs

parser = argparse.ArgumentParser()
parser.add_argument("--num-qubits", type=int, default=60)
parser.add_argument("--pattern", default="type1")
parser.add_argument("--delta", type=float, default=0.5)
parser.add_argument("--log2-sizes", type=int, nargs="+", default=[17, 19])
parser.add_argument("--tiles", type=int, nargs="+", default=[12, 14, 16])
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

ORDER = None  # a callable (i, j, g) -> permutation, or None for the shipped order
_pairs_sorted_on_device = sparse_mod._pairs_sorted_on_device


def reordered_pairs(pairs, groups, alloc):
    """``_pairs_sorted_on_device``, then the real entries permuted by ``ORDER``; padding stays last."""
    out = _pairs_sorted_on_device(pairs, groups, alloc)
    if ORDER is None:
        return out
    i, j, g = (np.array(a) for a in out)
    real = int(np.count_nonzero(i != j))  # padding has i == j, and sorts last
    perm = ORDER(i[:real], j[:real], g[:real])
    for array in (i, j, g):
        array[:real] = array[:real][perm]
    return [i, j, g]


sparse_mod._pairs_sorted_on_device = reordered_pairs  # ty: ignore[invalid-assignment]


def tile_order(shift, i, j, g):
    return np.lexsort((i, j >> shift, i >> shift))


ORDERS = {
    "i": None,
    "group": lambda i, j, g: np.lexsort((i, g)),
    **{f"tile{s}": functools.partial(tile_order, s) for s in options.tiles},
}


def timed(fn):
    t0 = time.perf_counter()
    out = jax.block_until_ready(fn())
    return time.perf_counter() - t0, out


def entries(operator):
    """The real ``(i, j)`` entries, as one sorted int64 key array."""
    i, j = (np.asarray(a).ravel() for a in operator[1:3])
    keep = i != j
    return np.sort(i[keep].astype(np.int64) << 32 | j[keep])


ham = PauliSumXZ.from_paulisum(
    xxz(options.num_qubits, options.delta, *patterns(options.num_qubits)[options.pattern])
)
arm = Matvec.PAIRS
solve = sparse_mod._run_sparse
print(f"{jax.devices()[0].device_kind}, n={options.num_qubits} {options.pattern}, matvec=pairs")
print(
    "order   N    | solve per iter (ms) x wins | 1-D matvec (ms) x wins | (2, N) matvec (ms) x wins | iters | eigval diff"
)
for log2 in options.log2_sizes:
    states = hamming_shells(options.num_qubits, 1 << log2, np.random.default_rng(0))
    h, states_p, size = _sqd_inputs(ham, states, None, False, arm, 0.0, None, (32, 2))
    states_u = uniquify_states(states_p, size)
    vecs = [
        jax.random.normal(jax.random.key(0), shape, jnp.complex128)
        for shape in ((size,), (2, size))
    ]
    matvec = jax.jit(sparse_mod._apply_pairs)
    ops, calls = {}, {}
    for name, order in ORDERS.items():
        ORDER = order
        ops[name] = jax.block_until_ready(sparse_mod._sparse_operator(h, states_u, arm))
        calls[name] = [functools.partial(solve, h, states_u, ops[name], size, False, arm)]
        calls[name] += [functools.partial(matvec, vec, *ops[name]) for vec in vecs]
        for call in calls[name]:
            call()  # warm
    ref = entries(ops["i"])
    for name in ORDERS:
        assert np.array_equal(entries(ops[name]), ref), f"{name} changed the entries"
        if name != "i":
            assert not np.array_equal(np.asarray(ops[name][1]), np.asarray(ops["i"][1])), name
    times = {name: ([], [], []) for name in ORDERS}
    eig, iters = {}, {}
    for _ in range(options.rounds):
        for name in ORDERS:
            ITERATIONS.clear()
            t, result = timed(calls[name][0])
            times[name][0].append(t / ITERATIONS[-1])
            for k in (1, 2):
                times[name][k].append(timed(calls[name][k])[0])
            eig.setdefault(name, []).append(float(result.eigval))
            iters.setdefault(name, set()).add(ITERATIONS[-1])
    base = eig["i"][0]
    for name in ORDERS:
        diff = max(abs(e - base) for e in eig[name]) / abs(base)
        assert diff < 1e-12, (name, eig[name])
        cols = []
        for mine, ref_times in zip(times[name], times["i"], strict=True):
            med, med0 = statistics.median(mine), statistics.median(ref_times)
            wins = sum(a < b for a, b in zip(mine, ref_times, strict=True))
            cols.append(f"{med * 1e3:7.2f}  {med0 / med:.2f}x {wins}/{options.rounds}")
        print(
            f"{name:7} 2^{log2} | {' | '.join(cols)} | {sorted(iters[name])} | {diff:.1e}",
            flush=True,
        )
