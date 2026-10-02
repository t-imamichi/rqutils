"""A/B of ``"tables"`` against ``codes``, its diagonals stored as ``uint8`` codes.

``codes`` stores each group's diagonal as an index into its distinct values, encoded on the device by
``jnp.unique(size=256)`` per group in a scan, so no ``(J, N)`` diagonal is ever live: 1 B/slot instead of
8 (float64) or 16 (complex128); real groups keep a float64 table. The z = 0 ``fold`` arm this script
also ran is in the library since (``PauliSumXZ.zfree_first``); run it from ``81406ee``, where the
library predates it.

The reference is ``run_sqd(matvec=Matvec.TABLES)``. Arms are warm and interleaved; ``solve``
is per iteration with setup inside, as ``run_sqd``'s; ``1-D`` and ``(2, N)`` are the kernel alone; ``temp``
is the whole solve's ``temp_size_in_bytes``, ``op`` the operator arrays' bytes per ``J x N`` slot.
Matvecs must agree to ``1e-12`` and eigenvalues to ``1e-12`` relative; ``bit-identical`` reports whether
both matvec widths match the reference exactly.
Fixture as ``poc/dense_tune.py``.

Run: uv run python poc/dense_codes.py [--patterns type1 type2] [--log2-sizes 14] [--rounds 3]
"""

import argparse
import functools
import logging
import os
import statistics
import sys
import time

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from eigenpair_check_scale import hamming_shells, patterns, xxz

import rqutils.sqd._solve as solve_mod
from rqutils.paulis.symplectic import PauliSumXZ
from rqutils.sqd import Matvec, get_diagonal, get_xsource, uniquify_states
from rqutils.sqd._core import _sqd_inputs
from rqutils.sqd._dense import _pack_scanned, apply_xgrp
from rqutils.sqd._solve import _apply_parts, _group_parts, _solve, run_sqd

CODES = 256  # uint8
parser = argparse.ArgumentParser()
parser.add_argument("--num-qubits", type=int, default=60)
parser.add_argument("--delta", type=float, default=0.5)
parser.add_argument("--patterns", nargs="+", default=["type1", "type2"])
parser.add_argument("--log2-sizes", type=int, nargs="+", default=[14])
parser.add_argument("--rounds", type=int, default=3)
options = parser.parse_args()

ITERATIONS = []
_ground_locg = solve_mod.ground_locg


def counted_ground_locg(*args, **kwargs):
    out = _ground_locg(*args, **kwargs)
    jax.debug.callback(lambda niter: ITERATIONS.append(int(niter)), out[2])
    return out


solve_mod.ground_locg = counted_ground_locg  # ty: ignore[invalid-assignment]


def codes_kernel(vec, parts):
    """``"tables"``' scan with ``diagonal = table[code]``, over each group part as ``_apply_parts``."""

    def fn(out, val):
        xsource, code, table = val
        return out + apply_xgrp(xsource, table[code], vec), None

    out = jnp.zeros_like(vec)
    for part in parts:
        out = jax.lax.scan(fn, out, part)[0]
    return out


def encode(h, states_u):
    """``(xsources, codes, tables)`` per ``_group_parts`` part, one group's diagonal live at a time."""

    def fn(_, v):
        xsource = get_xsource(v[0], states_u)
        diag = get_diagonal(v[1], v[2], states_u)
        table, code = jnp.unique(diag, return_inverse=True, size=CODES, fill_value=diag[0])
        return None, (xsource, code.reshape(-1).astype(np.uint8), table)

    return tuple(jax.lax.scan(fn, None, part)[1] for part in _group_parts(h))


def solve_tail(h, states_u, apply, args, diag0):
    return _solve(
        h, states_u, apply, args, diag0, None, None, return_eigvec=False, maxiter=1000,
        atol=0.0, rtol=None, prefilter=(32, 2), log_level=logging.INFO, check_residual=False,
    )  # fmt: skip


@functools.partial(jax.jit, static_argnums=(2,))
def codes_args(h, states_p, size):
    states_u = uniquify_states(states_p, size)
    return states_u, (encode(h, states_u),)


@functools.partial(jax.jit, static_argnums=(2,))
def codes_solve(h, states_p, size):
    states_u, args = codes_args(h, states_p, size)
    return solve_tail(h, states_u, codes_kernel, args, lambda: args[0][0][2][0][args[0][0][1][0]])


@functools.partial(jax.jit, static_argnums=(2,))
def library_args(h, states_p, size):
    """``run_sqd``'s ``"tables"`` arguments for ``_apply_parts``."""
    states_u = uniquify_states(states_p, size)
    scanned = tuple(
        _pack_scanned(
            "tables",
            jax.lax.scan(lambda _, x: (None, get_xsource(x, states_u)), None, x)[1],
            jax.lax.scan(lambda _, v: (None, get_diagonal(v[0], v[1], states_u)), None, (z, c))[1],
            None,
        )
        for x, z, c in _group_parts(h)
    )
    return scanned, None


def timed(fn):
    t0 = time.perf_counter()
    out = jax.block_until_ready(fn())
    return time.perf_counter() - t0, out


def operator_bytes(args, states_shape):
    """Bytes of the arrays a matvec reads, ``states`` excluded."""
    return sum(x.nbytes for x in jax.tree.leaves(args) if x.shape != states_shape)


print(f"{jax.devices()[0].device_kind}, n={options.num_qubits}, delta={options.delta}")
print(
    "arm               N    | solve/iter (ms) x wins | 1-D (ms) x wins | (2, N) (ms) x wins |"
    " temp (MiB) | op (B/slot) | iters | bit-identical | eigval diff"
)
for pattern in options.patterns:
    ham = PauliSumXZ.from_paulisum(
        xxz(options.num_qubits, options.delta, *patterns(options.num_qubits)[pattern])
    )
    num_groups = len(ham.term_counts)
    print(f"-- {pattern}: J={num_groups}, {ham.c.dtype}")
    for log2 in options.log2_sizes:
        states = hamming_shells(options.num_qubits, 1 << log2, np.random.default_rng(0))
        matvec = Matvec.TABLES
        h, states_p, size = _sqd_inputs(ham, states, None, False, matvec, 0.0, None, (32, 2))
        ref_args = library_args(h, states_p, size)
        states_u, arm_args = codes_args(h, states_p, size)
        distinct = max(len(np.unique(row)) for part in arm_args[0] for row in np.asarray(part[2]))
        assert distinct < CODES, f"{distinct} distinct values overflow uint8"
        ref_mv = jax.jit(functools.partial(_apply_parts, matvec="tables"))
        arm_mv = jax.jit(codes_kernel)
        arm_solve = functools.partial(codes_solve, h, states_p, size)
        arm = f"codes (<={distinct})"
        ref_solve = functools.partial(run_sqd, h, states_p, size, False, matvec)
        vecs = [
            jax.random.normal(jax.random.key(0), shape, jnp.complex128)
            for shape in ((size,), (2, size))
        ]
        calls = {
            "ref": [ref_solve] + [functools.partial(ref_mv, v, *ref_args) for v in vecs],
            arm: [arm_solve] + [functools.partial(arm_mv, v, *arm_args) for v in vecs],
        }
        temp = {
            "ref": run_sqd.lower(h, states_p, size, False, matvec).compile(),
            arm: codes_solve.lower(h, states_p, size).compile(),
        }
        temp = {k: v.memory_analysis().temp_size_in_bytes for k, v in temp.items()}
        slots = num_groups * size
        op = {
            k: operator_bytes(a, states_u.shape) / slots
            for k, a in (("ref", ref_args), (arm, arm_args))
        }
        for fns in calls.values():
            for call in fns:
                call()  # compile
        ref_out, arm_out = np.asarray(calls["ref"][1]()), np.asarray(calls[arm][1]())
        identical = all(np.array_equal(calls["ref"][k](), calls[arm][k]()) for k in (1, 2))
        err = np.abs(ref_out - arm_out).max() / np.abs(ref_out).max()
        assert err <= 1e-12, (arm, err)
        times = {k: ([], [], []) for k in calls}
        eig, iters = {k: [] for k in calls}, {k: set() for k in calls}
        for _ in range(options.rounds):
            for key, fns in calls.items():
                ITERATIONS.clear()
                t, result = timed(fns[0])
                times[key][0].append(t / ITERATIONS[-1])
                for k in (1, 2):
                    times[key][k].append(timed(fns[k])[0])
                eig[key].append(float(result.eigval))
                iters[key].add(ITERATIONS[-1])
        base = eig["ref"][0]
        for key in calls:
            diff = max(abs(e - base) for e in eig[key]) / abs(base)
            assert diff < 1e-12, (key, eig[key])
            cols = []
            for mine, theirs in zip(times[key], times["ref"], strict=True):
                med, med0 = statistics.median(mine), statistics.median(theirs)
                wins = sum(a < b for a, b in zip(mine, theirs, strict=True))
                cols.append(f"{med * 1e3:7.2f} {med0 / med:5.2f}x {wins}/{options.rounds}")
            label = f"tables:{key}"
            same = "-" if key == "ref" else ("yes" if identical else f"no ({err:.1e})")
            print(
                f"{label:17} 2^{log2} | {' | '.join(cols)} | {temp[key] / 2**20:8.1f} |"
                f" {op[key]:6.2f} | {sorted(iters[key])} | {same} | {diff:.1e}",
                flush=True,
            )
