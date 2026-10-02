"""A/B: the sparse kernels gathering from ``vec`` as ``(N, k)`` rather than ``(k, N)``.

``run_sqd`` batches LOBPCG's matvec pair as a ``(2, N)`` array, so each random index in ``"pairs"``,
``"csr"`` and ``"ell"`` touches two places ``16·N`` B apart. The ``col`` arm moves the state axis to the
front for the gathers and scatters, so the batch's values share a cache line (or GPU sector), and moves
it back after. ``row`` is the shipped kernel, and ``col`` keeps its CUDA rules (split complex carry, no sorted hint). Both arms are warm and interleaved, one round each. The
matvecs agree bit for bit, but fused into the solve the iteration count can differ, so ``solve`` is per
iteration and the eigenvalues must agree to ``1e-12`` relative (``poc/sparse/layout.md``).

Fixture as ``poc/sparse/gpu.py``. ``solve`` is ``_run_sparse`` alone; ``matvec`` the jitted kernel on a
``(2, N)`` complex vector passed as an argument.

Run: uv run python poc/sparse/layout.py [--log2-sizes 17 19] [--arms pairs ell] [--rounds 5]
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


def state_major(vec):
    """``vec`` with its state axis first, and the inverse."""
    return jnp.moveaxis(vec, -1, 0), lambda out: jnp.moveaxis(out, 0, -1)


def col(fac, ndim):
    """``fac`` broadcast against a state-major gather of a rank-``ndim`` ``vec``."""
    return fac.reshape(fac.shape + (1,) * (ndim - 1))


def scan_add(updates, out, xs, ordered=False):
    """``sparse_mod._scan_add`` on a state-major ``out``: ``updates`` index its leading axis.

    The same platform rule as the library: on CUDA a complex carry is split into real and imaginary
    parts and the sorted hint dropped (``poc/sparse/split.md``, ``poc/sparse/pairs-tune.md``), else
    the GPU arm would measure that defect rather than the layout.
    """

    def scan(parts, ordered):
        def body(parts, chunk):
            for index, value in updates(chunk):
                values = (value.real, value.imag) if len(parts) == 2 else (value,)
                parts = tuple(
                    p.at[index].add(v, indices_are_sorted=ordered)
                    for p, v in zip(parts, values, strict=True)
                )
            return parts, None

        return jax.lax.scan(body, parts, xs)[0]

    def default(out):
        return scan((out,), ordered)[0]

    def cuda(out):
        if not jnp.iscomplexobj(out):
            return scan((out,), False)[0]
        re, im = scan((out.real, out.imag), False)
        return jax.lax.complex(re, im)

    return jax.lax.platform_dependent(out, default=default, cuda=cuda)


def apply_pairs(vec, d0, pi, pj, d):
    v, back = state_major(vec)

    def updates(chunk):
        i, j, di = chunk
        return [(i, col(di, vec.ndim) * v[j]), (j, col(jnp.conj(di), vec.ndim) * v[i])]

    return back(scan_add(updates, state_major(d0 * vec)[0], (pi, pj, d)))


def apply_csr(vec, d0, *entries):
    v, back = state_major(vec)

    def updates(chunk):
        ti, si, di = chunk
        return [(ti, col(di, vec.ndim) * v[si])]

    out = state_major(d0 * vec)[0]
    for k in range(0, len(entries), 3):
        out = scan_add(updates, out, entries[k : k + 3], ordered=True)
    return back(out)


def apply_ell(vec, d0, *buckets):
    v, back = state_major(vec)

    def updates(piece):
        rows, src, fac = piece
        return [(rows, jnp.sum(col(fac, vec.ndim) * v[src], axis=-vec.ndim))]

    out = state_major(d0 * vec)[0]
    for k in range(0, len(buckets), 3):
        out = scan_add(updates, out, buckets[k : k + 3])
    return back(out)


COL = {Matvec.PAIRS: apply_pairs, Matvec.CSR: apply_csr, Matvec.ELL: apply_ell}
ROW = dict(sparse_mod._SPARSE_APPLY)


def timed(fn):
    t0 = time.perf_counter()
    out = jax.block_until_ready(fn())
    return time.perf_counter() - t0, out


ham = PauliSumXZ.from_paulisum(
    xxz(options.num_qubits, options.delta, *patterns(options.num_qubits)[options.pattern])
)
print(f"{jax.devices()[0].device_kind}, n={options.num_qubits} {options.pattern}")
print(
    "arm   N    | solve per iter row / col (ms) x wins | matvec row / col (ms) x wins | iters row / col"
)
for log2 in options.log2_sizes:
    states = hamming_shells(options.num_qubits, 1 << log2, np.random.default_rng(0))
    for name in options.arms:
        arm = Matvec(name)
        h, states_p, size = _sqd_inputs(ham, states, None, False, arm, 0.0, None, (32, 2))
        states_u = uniquify_states(states_p, size)
        operator = jax.block_until_ready(sparse_mod._sparse_operator(h, states_u, arm))
        vec = jax.random.normal(jax.random.key(0), (2, size), jnp.complex128)
        solves, matvecs, lowered = {}, {}, {}
        for layout, table in (("row", ROW), ("col", COL)):
            # _SPARSE_APPLY is read at trace time, and a second jit of the same function reuses the
            # first's trace, so each layout gets a function of its own.
            sparse_mod._SPARSE_APPLY[arm] = table[arm]
            solve = jax.jit(
                lambda *a: sparse_mod._run_sparse.__wrapped__(*a), static_argnums=(3, 4, 5)
            )
            solves[layout] = functools.partial(solve, h, states_u, operator, size, False, arm)
            lowered[layout] = solve.lower(h, states_u, operator, size, False, arm).as_text()
            matvecs[layout] = functools.partial(jax.jit(table[arm]), vec, *operator)
            solves[layout]()  # compile
            matvecs[layout]()
        sparse_mod._SPARSE_APPLY[arm] = ROW[arm]
        assert lowered["row"] != lowered["col"], "both solve arms compiled the same kernel"
        ref = np.asarray(matvecs["row"]())
        err = np.abs(np.asarray(matvecs["col"]()) - ref).max() / np.abs(ref).max()
        times = {key: [] for key in ("srow", "scol", "mrow", "mcol")}
        eig, iters = {}, {}
        for _ in range(options.rounds):
            for layout in ("row", "col"):
                ITERATIONS.clear()
                t, result = timed(solves[layout])
                times["s" + layout].append(t / ITERATIONS[-1])
                eig.setdefault(layout, []).append(float(result.eigval))
                iters.setdefault(layout, set()).add(ITERATIONS[-1])
                times["m" + layout].append(timed(matvecs[layout])[0])
        spread = max(eig["row"] + eig["col"]) - min(eig["row"] + eig["col"])
        assert spread < 1e-12 * abs(eig["row"][0]), eig
        med = {key: statistics.median(v) for key, v in times.items()}
        swins = sum(c < r for r, c in zip(times["srow"], times["scol"], strict=True))
        mwins = sum(c < r for r, c in zip(times["mrow"], times["mcol"], strict=True))
        print(
            f"{name:5} 2^{log2} | {med['srow'] * 1e3:7.2f} / {med['scol'] * 1e3:7.2f}  {med['srow'] / med['scol']:.2f}x"
            f" {swins}/{options.rounds} | {med['mrow'] * 1e3:7.2f} / {med['mcol'] * 1e3:7.2f}"
            f"  {med['mrow'] / med['mcol']:.2f}x {mwins}/{options.rounds} | {sorted(iters['row'])} / {sorted(iters['col'])}"
            f"  (matvec rel diff {err:.1e}, eigval spread {spread:.1e})",
            flush=True,
        )
