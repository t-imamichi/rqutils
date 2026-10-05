"""``"pairs"`` with and without its exact-zero entries (``_drop_zeros``): build, solve, memory.

Fixture as ``poc/sparse/gpu.py``: spinchain's open-XXZ ``xxz`` with Hamming-shell subspaces. ``all`` patches
``_drop_zeros`` out, ``nonzero`` is the library; one host search serves both. Each arm warms up once, then
``--rounds`` interleaved solves. The eigenvalues must agree to ``1e-12``, not bit for bit: dropping
entries moves chunk boundaries, and a chunk adds its ``out[i]`` updates before its ``out[j]``, so a row's
sum is reordered. ``codes`` is ``nonzero`` with each factor a ``uint8`` index into a table of the distinct
values, and must match it bit for bit on CPU; a GPU's atomic scatter-add has no fixed order, so
there it only meets the ``1e-12``. ``iters`` is LOBPCG's count, by a host callback as ``poc/sparse/gpu.py``.

Run: uv run python poc/sparse/prune.py [--patterns type1 type2] [--log2-sizes 17 19] [--rounds 5]
"""

import argparse
import logging
import os
import sys
import time

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # poc/
from eigenpair_check_scale import hamming_shells, patterns, xxz

import rqutils.sqd._solve as solve_mod
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
options = parser.parse_args()

ITERATIONS = []
_ground_locg = solve_mod.ground_locg


def counted_ground_locg(*args, **kwargs):
    out = _ground_locg(*args, **kwargs)
    jax.debug.callback(lambda niter: ITERATIONS.append(int(niter)), out[2])
    return out


solve_mod.ground_locg = counted_ground_locg  # ty: ignore[invalid-assignment]
library_drop = sm._drop_zeros
ARMS = {
    "all": lambda t, s, d, chunk, size: (t, s, d),
    "nonzero": library_drop,
    "codes": library_drop,
}


def encode(operator):
    """``(d0, i, j, code, table)``: each factor an index into its distinct values, ``d == table[code]``.

    The code is the narrowest unsigned type that fits, so there is no fallback; the table is padded to
    256, or a size class above that, so its shape changes with the class rather than the count.
    """
    d0, t, s, d = operator
    table, code = jnp.unique(d, return_inverse=True)
    width = max(256, sm._size_class(len(table)))
    code = code.reshape(d.shape).astype(np.min_scalar_type(len(table) - 1))
    return d0, t, s, code, jnp.pad(table, (0, width - len(table)))


def apply_codes(vec, d0, pi, pj, code, table):
    """``sm._apply_pairs`` with ``d = table[code]``."""
    sharding = jax.typeof(vec).sharding

    def updates(chunk):
        i, j, k = chunk
        di = table[k]
        return [
            (i, di * vec.at[..., j].get(out_sharding=sharding)),
            (j, jnp.conj(di) * vec.at[..., i].get(out_sharding=sharding)),
        ]

    return sm._scan_add(updates, d0 * vec, (pi, pj, code))


@jax.jit(static_argnames=["states_size", "return_eigvec"])
def run_codes(hamiltonian, states_u, operator, states_size, return_eigvec):
    return sm._solve(
        hamiltonian,
        states_u,
        apply_codes,
        operator,
        lambda: operator[0],
        None,
        None,
        return_eigvec=return_eigvec,
        maxiter=1000,
        atol=0.0,
        rtol=None,
        prefilter=(32, 2),
        log_level=logging.INFO,
        check_residual=False,
    )


RUN = {"all": sm._run_sparse, "nonzero": sm._run_sparse, "codes": run_codes}


def build(h, states_u, pairs, arm):
    sm._drop_zeros = ARMS[arm]  # ty: ignore[invalid-assignment]
    try:
        t0 = time.perf_counter()
        operator = sm._sparse_operator(h, states_u, pairs)
        operator = jax.block_until_ready(encode(operator) if arm == "codes" else operator)
        return operator, time.perf_counter() - t0
    finally:
        sm._drop_zeros = library_drop


def solve(h, states_u, operator, size, arm):
    t0 = time.perf_counter()
    result = jax.block_until_ready(RUN[arm](h, states_u, operator, size, True))
    jax.effects_barrier()
    return result, time.perf_counter() - t0, ITERATIONS[-1]


n = options.num_qubits
print(f"{jax.default_backend()}, n={n}, chunk={sm._chunk()}")
print(
    "arm      pattern N    | entries   MiB  temp MiB | build (s) | solve (s)     x wins | iters | eigval diff"
)
for pattern in options.patterns:
    ham = PauliSumXZ.from_paulisum(xxz(n, options.delta, *patterns(n)[pattern]))
    for log2 in options.log2_sizes:
        states = hamming_shells(n, 1 << log2, np.random.default_rng(0))
        h, states_p, size = _sqd_inputs(ham, states, None, False, Matvec.PAIRS, 0.0, None, (32, 2))
        states_u = jax.block_until_ready(uniquify_states(states_p, size))
        pairs = sm._group_pairs(h, states_u)
        ops, builds, results, times, iters = {}, {}, {}, {}, {}
        for arm in ARMS:
            build(h, states_u, pairs, arm)  # compile the build's eager ops, so it times warm
            ops[arm], builds[arm] = build(h, states_u, pairs, arm)
            results[arm] = solve(h, states_u, ops[arm], size, arm)[0]  # warm-up
        for _ in range(options.rounds):
            for arm in ARMS:
                _, elapsed, count = solve(h, states_u, ops[arm], size, arm)
                times.setdefault(arm, []).append(elapsed)
                iters.setdefault(arm, set()).add(count)
        ref = results["all"]
        same = all(
            np.array_equal(
                np.asarray(getattr(results["codes"], f)), np.asarray(getattr(results["nonzero"], f))
            )
            for f in ("eigval", "eigvec")
        )
        # Atomic scatter-adds on a GPU sum in no fixed order, so only the CPU is bit-reproducible.
        assert same or jax.default_backend() != "cpu", "codes must match nonzero bit for bit"
        for arm in ARMS:
            op, r = ops[arm], results[arm]
            diff = float(r.eigval) - float(ref.eigval)
            assert abs(diff) <= 1e-12, f"{arm} differs from all: {r.eigval} against {ref.eigval}"
            temp = (
                RUN[arm]
                .lower(h, states_u, op, size, True)
                .compile()
                .memory_analysis()
                .temp_size_in_bytes
            )
            entries = int(np.count_nonzero(np.asarray(op[1]) != np.asarray(op[2])))
            ratio = np.median(times["all"]) / np.median(times[arm])
            wins = sum(a < b for a, b in zip(times[arm], times["all"], strict=True))
            print(
                f"{arm:8s} {pattern} 2^{log2} | {entries:8d} {sum(a.nbytes for a in op) / 2**20:6.1f}"
                f" {temp / 2**20:8.1f} | {builds[arm]:9.3f} | {np.median(times[arm]):7.3f}"
                f" {ratio:5.2f}x {wins}/{options.rounds} | {sorted(iters[arm])} | {diff:+.1e}",
                flush=True,
            )
