"""``"pairs"``' cross-group sort by ``i`` on a GPU host: keep it, drop it, or do it faster?

On the GH200 the sort is 48-63% of a ``"pairs"`` build (``poc/sparse/build.py``), about 1.0 s of a 3.90 s
``type2`` ``2^22`` call, while dropping it cost only 2-5% of the solve (``poc/sparse/tiles.md`` §3).
Arms, each from one shared host search and through the same padding and device factors as the library:

- ``counting``: the library's ``_sort_by_target(both=False)``.
- ``none``: no cross-group sort, the groups concatenated (each already ascending in ``i``).
- ``argsort``: a stable ``np.argsort`` of the concatenation by ``i``, on the host.
- ``device``: the same with a stable ``jnp.argsort`` on the device, the arrays permuted there.

``argsort`` and ``device`` must equal ``counting`` exactly and ``none`` hold the same pairs. Every arm
shares one compiled solve (only the data differ). ``build`` is the pairs to the device operator, the
search excluded (printed once); ``solve`` is ``_run_sparse``, per call and per iteration; ``total`` is
their sum. Arms are warm and interleaved. Fixture as ``poc/sparse/gpu.py``.

Run: uv run python poc/sparse/pairs_sort.py [--patterns type1 type2] [--log2-sizes 20 22] [--rounds 5]
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
import rqutils.sqd._sparse as sm
from rqutils.paulis.symplectic import PauliSumXZ
from rqutils.sqd import Matvec, get_diagonal, uniquify_states
from rqutils.sqd._core import _sqd_inputs

parser = argparse.ArgumentParser()
parser.add_argument("--num-qubits", type=int, default=60)
parser.add_argument("--delta", type=float, default=0.5)
parser.add_argument("--patterns", nargs="+", default=["type1", "type2"])
parser.add_argument("--log2-sizes", type=int, nargs="+", default=[20, 22])
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
ARM = Matvec.PAIRS


def build(h, states_u, pairs, how):
    """``_sparse_operator``'s ``"pairs"`` branch with the cross-group order chosen by ``how``."""
    size, chunk = states_u.shape[0], sm._chunk(ARM)
    z, c = jnp.asarray(h.z), jnp.asarray(h.c)
    first = int(0 not in pairs)
    d0 = get_diagonal(z[0], c[0], states_u) if first else jnp.zeros(size, c.dtype)
    groups = range(first, h.x.shape[0])
    coeffs = np.asarray(h.c)
    kmax = max((int(np.count_nonzero(coeffs[g])) for g in groups), default=1)

    def alloc(count):
        return [sm._padded(count, f, chunk) for f in (size - 1, size - 1, 0)]

    if how == "counting":
        t, s, g = (
            jnp.asarray(a)
            for a in sm._sort_by_target(dict(pairs), list(groups), size, alloc, both=False)[0]
        )
    else:
        i = np.concatenate([pairs[k][0] for k in groups])
        j = np.concatenate([pairs[k][1] for k in groups])
        grp = np.repeat(
            np.arange(first, h.x.shape[0], dtype=np.int32), [len(pairs[k][0]) for k in groups]
        )
        n = len(i)
        if how == "argsort":
            order = np.argsort(i, kind="stable")
            i, j, grp = i[order], j[order], grp[order]
        if how == "device":
            i, j, grp = (jnp.asarray(a) for a in (i, j, grp))
            order = jnp.argsort(i, stable=True)
            i, j, grp = i[order], j[order], grp[order]
        t, s, g = (jnp.asarray(a) for a in alloc(n))
        t, s, g = t.at[:n].set(i), s.at[:n].set(j), g.at[:n].set(grp)
    t, s, g = (a.reshape(-1, chunk) for a in (t, s, g))
    return d0, t, s, sm._entry_factors(t, s, g, z, c, states_u, kmax)


def timed(fn):
    t0 = time.perf_counter()
    out = jax.block_until_ready(fn())
    return time.perf_counter() - t0, out


HOWS = ("counting", "none", "argsort", "device")
print(f"{jax.devices()[0].device_kind}, n={options.num_qubits}, chunk={sm._chunk(ARM)}")
print(
    "how      pattern N    | build (s) x | solve (s) per iter (ms) x | total (s) x wins | iters | eigval diff"
)
for pattern in options.patterns:
    ham = PauliSumXZ.from_paulisum(
        xxz(options.num_qubits, options.delta, *patterns(options.num_qubits)[pattern])
    )
    for log2 in options.log2_sizes:
        states = hamming_shells(options.num_qubits, 1 << log2, np.random.default_rng(0))
        h, states_p, size = _sqd_inputs(ham, states, None, False, ARM, 0.0, None, (32, 2))
        states_u = jax.block_until_ready(uniquify_states(states_p, size))
        search, pairs = timed(functools.partial(sm._group_pairs, h, states_u))
        ops = {how: jax.block_until_ready(build(h, states_u, pairs, how)) for how in HOWS}
        ref = [np.asarray(a) for a in ops["counting"][1:3]]
        for how in ("argsort", "device"):
            assert all(
                np.array_equal(np.asarray(a), b) for a, b in zip(ops[how][1:3], ref, strict=True)
            ), how

        def key(op):
            i, j = (np.asarray(a).ravel() for a in op[1:3])
            keep = i != j
            return np.sort(i[keep].astype(np.int64) << 32 | j[keep])

        assert np.array_equal(key(ops["none"]), key(ops["counting"])), "none changed the pairs"
        assert not np.array_equal(np.asarray(ops["none"][1]), ref[0]), "none must differ in order"
        for how in HOWS:  # warm: one compiled solve serves every arm
            jax.block_until_ready(sm._run_sparse(h, states_u, ops[how], size, False, ARM))
        times = {how: ([], [], []) for how in HOWS}
        eig, iters = {how: [] for how in HOWS}, {how: set() for how in HOWS}
        for _ in range(options.rounds):
            for how in HOWS:
                b, op = timed(functools.partial(build, h, states_u, pairs, how))
                ITERATIONS.clear()
                sv, result = timed(
                    functools.partial(sm._run_sparse, h, states_u, op, size, False, ARM)
                )
                times[how][0].append(b)
                times[how][1].append(sv)
                times[how][2].append(b + sv)
                eig[how].append(float(result.eigval))
                iters[how].add(ITERATIONS[-1])
        base = eig["counting"][0]
        print(
            f"{pattern} 2^{log2}: search {search:.3f} s, {sum(len(p[0]) for p in pairs.values())} pairs"
        )
        for how in HOWS:
            med = [statistics.median(v) for v in times[how]]
            ref_med = [statistics.median(v) for v in times["counting"]]
            wins = sum(a < b for a, b in zip(times[how][2], times["counting"][2], strict=True))
            per_iter = med[1] / statistics.median(iters[how])
            diff = max(abs(e - base) for e in eig[how]) / abs(base)
            assert diff < 1e-12, (how, eig[how])
            print(
                f"{how:8} {pattern} 2^{log2} | {med[0]:7.3f} {ref_med[0] / med[0]:5.2f}x |"
                f" {med[1]:7.3f} {per_iter * 1e3:7.2f} {ref_med[1] / med[1]:5.2f}x |"
                f" {med[2]:7.3f} {ref_med[2] / med[2]:5.2f}x {wins}/{options.rounds} |"
                f" {sorted(iters[how])} | {diff:.1e}",
                flush=True,
            )
