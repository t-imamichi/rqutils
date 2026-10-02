"""A/B of three dense-kernel levers: scan unroll, a cached identity diagonal, a fixed-trip diagonal loop.

A GH200 ``"indices"`` matvec is ~75% diagonal recompute (``poc/sparse/gpu.md`` §6): ``get_diagonal``'s
``while_loop`` makes ΣK = 179 full passes per matvec at ``type1``, 59 of them the identity group's, with a
device-to-host sync per step; ``"tables"`` is one gather-multiply-add per group, memory-bound, ``out``
read and written once per group. Arms are ``--matvecs`` × ``--variants`` × ``--unrolls``:

- ``plain``: the library's kernel, its scan over X groups given ``unroll``.
- ``id``: the identity group's diagonal computed once per solve and applied as ``d0 * vec``; the scan
  covers the other groups (``"indices"``/``"onthefly"`` only).
- ``id-static``: as ``id``, the other groups' diagonals summed over a fixed ``kmax`` terms instead of the
  ``while_loop``, so no step syncs with the host.

``unroll`` lets XLA fuse consecutive groups' updates, reading and writing ``out`` once per ``unroll``
groups. Each arm replicates ``run_sqd``'s single-device assembly around the library's ``_solve``; the
reference is ``run_sqd`` itself. Before the library adopted ``id-static`` (bucketed by term count, so
without ``kmax`` padding), ``plain`` at ``unroll=1`` matched it exactly; since, ``plain`` is the old
kernel, and summation order can shift its iteration count. The reference's ``1-D`` and ``(2, N)``
columns time the library's ``_apply_buckets`` on ``run_sqd``'s own arguments (``library_args``). Arms are
warm and interleaved; ``solve`` is per iteration, its setup inside (as ``run_sqd``'s); ``1-D`` and
``(2, N)`` are the kernel alone, setup outside; ``temp`` is the ``(2, N)`` kernel's
``temp_size_in_bytes``. Eigenvalues must agree to ``1e-12`` relative, traced solves be pairwise distinct.
Fixture as ``poc/sparse/gpu.py``.

Run: uv run python poc/dense_tune.py [--log2-sizes 20 22] [--matvecs indices tables]
     [--variants plain id id-static] [--unrolls 1 4] [--rounds 5]
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
from rqutils.sqd._dense import _apply_buckets, _bucket_args, _pack_scanned, apply_xgrp
from rqutils.sqd._diagonal import _z_parity
from rqutils.sqd._solve import _group_parts, _solve, run_sqd

VARIANTS = ("plain", "id", "id-static")
parser = argparse.ArgumentParser()
parser.add_argument("--num-qubits", type=int, default=60)
parser.add_argument("--pattern", default="type1")
parser.add_argument("--delta", type=float, default=0.5)
parser.add_argument("--log2-sizes", type=int, nargs="+", default=[20, 22])
parser.add_argument(
    "--matvecs", nargs="+", default=["indices", "tables"], choices=["indices", "tables", "onthefly"]
)
parser.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=VARIANTS)
parser.add_argument("--unrolls", type=int, nargs="+", default=[1, 4])
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


def kernel(vec, scanned, states, d0, *, matvec, unroll, kmax):
    """``_apply_h_kernel`` with ``unroll``, an optional cached ``d0`` and, given ``kmax``, a fixed-trip diagonal."""

    def fn(out, val):
        xsource = get_xsource(val[0], states) if matvec == "onthefly" else val[0]
        if matvec == "tables":
            diagonal = val[1]
        elif kmax:
            z, c = val[1], val[2]
            diagonal = sum(c[k] * (1.0 - 2.0 * _z_parity(states, z[k])) for k in range(kmax))
        else:
            diagonal = get_diagonal(val[1], val[2], states)
        return out + apply_xgrp(xsource, diagonal, vec), None

    init = jnp.zeros_like(vec) if d0 is None else d0 * vec
    return jax.lax.scan(fn, init, scanned, unroll=unroll)[0]


def apply_parts(vec, parts, states, d0, *, matvec, unroll, kmax):
    """``_apply_parts`` over :func:`kernel`; ``d0`` seeds the first part."""
    out = None
    for part in parts:
        seed = d0 if out is None else None
        part_out = kernel(vec, part, states, seed, matvec=matvec, unroll=unroll, kmax=kmax)
        out = part_out if out is None else out + part_out
    return out


def prepare(h, states_p, states_size, matvec, variant, kmax):
    """``run_sqd``'s single-device assembly: ``(states_u, args, groups, d0)`` for :func:`apply_parts`."""
    states_u = uniquify_states(states_p, states_size)
    parts = _group_parts(h) if matvec == "tables" else (h.arrays,)
    groups = None
    if matvec != "onthefly":
        groups = tuple(
            (
                jax.lax.scan(lambda _, x: (None, get_xsource(x, states_u)), None, part[0])[1],
                *part[1:],
            )
            for part in parts
        )
    d0 = None
    if matvec == "tables":
        diagonals = tuple(
            jax.lax.scan(lambda _, v: (None, get_diagonal(v[0], v[1], states_u)), None, part[1:])[1]
            for part in parts
        )
        scanned = tuple(
            _pack_scanned(matvec, g[0], d, None) for g, d in zip(groups, diagonals, strict=True)
        )
    else:
        x, z, c = h.arrays if groups is None else groups[0]
        if variant == "plain":
            scanned = (_pack_scanned(matvec, x, z, c),)
        else:  # the identity group is group 0, checked on the host
            d0 = get_diagonal(z[0], c[0], states_u)
            k = kmax if variant == "id-static" else z.shape[1]
            scanned = (_pack_scanned(matvec, x[1:], z[1:, :k], c[1:, :k]),)
    return states_u, (scanned, None if matvec == "tables" else states_u, d0), groups, d0


@functools.partial(jax.jit, static_argnums=(2, 3))
def library_args(h, states_p, states_size, matvec):
    """``run_sqd``'s arguments for ``_apply_buckets`` through the library's ``_bucket_args``."""
    states_u = uniquify_states(states_p, states_size)
    x = h.x
    if matvec != "onthefly":
        x = jax.lax.scan(lambda _, xg: (None, get_xsource(xg, states_u)), None, x)[1]
    return _bucket_args(h, x, states_u)


@functools.partial(jax.jit, static_argnums=(2, 3, 4, 5, 6))
def dense_solve(h, states_p, states_size, matvec, variant, unroll, kmax):
    states_u, args, groups, d0 = prepare(h, states_p, states_size, matvec, variant, kmax)
    static_k = kmax if variant == "id-static" else 0
    apply = functools.partial(apply_parts, matvec=matvec, unroll=unroll, kmax=static_k)

    def diag0():
        if matvec == "tables":
            return args[0][0][1][0]
        return d0 if d0 is not None else get_diagonal(h.z[0], h.c[0], states_u)

    return _solve(
        h,
        states_u,
        apply,
        args,
        diag0,
        groups,
        None,
        return_eigvec=False,
        maxiter=1000,
        atol=0.0,
        rtol=None,
        prefilter=(32, 2),
        log_level=logging.INFO,
        check_residual=False,
    )


def timed(fn):
    t0 = time.perf_counter()
    out = jax.block_until_ready(fn())
    return time.perf_counter() - t0, out


ham = PauliSumXZ.from_paulisum(
    xxz(options.num_qubits, options.delta, *patterns(options.num_qubits)[options.pattern])
)
identity = not np.asarray(ham.x[0]).any()
coeffs = np.asarray(ham.c)
kmax = max(int(np.count_nonzero(row)) for row in coeffs[1:])  # over the non-identity groups
assert identity, "the id variants need a leading identity group"
print(f"{jax.devices()[0].device_kind}, n={options.num_qubits} {options.pattern}, kmax={kmax}")
print(
    "arm                  N    | solve/iter (ms) x wins | 1-D (ms) x wins | (2, N) (ms) x wins |"
    " temp (MiB) | iters | eigval diff"
)
for log2 in options.log2_sizes:
    states = hamming_shells(options.num_qubits, 1 << log2, np.random.default_rng(0))
    for name in options.matvecs:
        matvec = Matvec(name)
        h, states_p, size = _sqd_inputs(ham, states, None, False, matvec, 0.0, None, (32, 2))
        arms = [("ref", 1)] + [
            (v, u)
            for v in options.variants
            if v == "plain" or matvec != "tables"
            for u in options.unrolls
        ]
        vecs = [
            jax.random.normal(jax.random.key(0), shape, jnp.complex128)
            for shape in ((size,), (2, size))
        ]
        calls, traced, temp = {}, {}, {}
        for arm in arms:
            variant, unroll = arm
            if variant == "ref":
                solve = functools.partial(run_sqd, h, states_p, size, False, matvec)
                traced[arm] = str(run_sqd.trace(h, states_p, size, False, matvec).jaxpr)
                prep = ("plain", 0)
            else:
                solve = functools.partial(
                    dense_solve, h, states_p, size, matvec, variant, unroll, kmax
                )
                traced[arm] = str(
                    dense_solve.trace(h, states_p, size, matvec, variant, unroll, kmax).jaxpr
                )
                prep = (variant, kmax if variant == "id-static" else 0)
            if variant == "ref" and matvec != "tables":
                args = library_args(h, states_p, size, matvec)
                mv = jax.jit(functools.partial(_apply_buckets, matvec=matvec))
            else:
                _, args, _, _ = jax.jit(prepare, static_argnums=(2, 3, 4, 5))(
                    h, states_p, size, matvec, prep[0], kmax
                )
                mv = jax.jit(
                    functools.partial(apply_parts, matvec=matvec, unroll=unroll, kmax=prep[1])
                )
            calls[arm] = [solve] + [functools.partial(mv, vec, *args) for vec in vecs]
            temp[arm] = mv.lower(vecs[1], *args).compile().memory_analysis().temp_size_in_bytes
            for call in calls[arm]:
                call()  # compile
        assert len(set(traced.values())) == len(arms), "two arms traced the same solve"
        ref = np.asarray(calls[("ref", 1)][2]())
        for arm in arms:
            got = np.asarray(calls[arm][2]())
            assert np.abs(got - ref).max() <= 1e-12 * np.abs(ref).max(), arm
        times = {arm: ([], [], []) for arm in arms}
        eig, iters = {arm: [] for arm in arms}, {arm: set() for arm in arms}
        for _ in range(options.rounds):
            for arm in arms:
                ITERATIONS.clear()
                t, result = timed(calls[arm][0])
                times[arm][0].append(t / ITERATIONS[-1])
                for k in (1, 2):
                    times[arm][k].append(timed(calls[arm][k])[0])
                eig[arm].append(float(result.eigval))
                iters[arm].add(ITERATIONS[-1])
        base = eig[("ref", 1)][0]
        for arm in arms:
            diff = max(abs(e - base) for e in eig[arm]) / abs(base)
            assert diff < 1e-12, (arm, eig[arm])
            cols = []
            for mine, theirs in zip(times[arm], times[("ref", 1)], strict=True):
                med, med0 = statistics.median(mine), statistics.median(theirs)
                wins = sum(a < b for a, b in zip(mine, theirs, strict=True))
                cols.append(f"{med * 1e3:7.2f} {med0 / med:5.2f}x {wins}/{options.rounds}")
            label = f"{name}:{arm[0]}@u{arm[1]}"
            print(
                f"{label:20} 2^{log2} | {' | '.join(cols)} | {temp[arm] / 2**20:7.1f} |"
                f" {sorted(iters[arm])} | {diff:.1e}",
                flush=True,
            )
