"""A/B of four ``"pairs"`` levers on a GPU: chunk size, a sorted ``i`` scatter, one merged scatter, real factors.

After ``0d25235`` a GH200 ``"pairs"`` matvec is 84-89% scatters (``poc/sparse-split.md`` §4), stepping
``_CHUNK = 32768`` entries at a time: 320 steps and 1603 launches per call at ``type2`` ``2^22``, each
kernel a fraction of the GPU. Arms are every ``--chunks`` × ``--variants``:

- ``base``: the library kernel, ``_apply_pairs``.
- ``sorted``: the ``out[i]`` scatter told ``indices_are_sorted`` (pairs are sorted by ``i``); ``out[j]``
  unchanged.
- ``merged``: both directions as one scatter of the concatenated indices and values.
- ``real``: ``"csr"``'s split, real groups' pairs with ``float64`` factors and the rest ``complex128``,
  two scans; the memory lever.

The reference is ``base`` at ``2^15``, the shipped kernel. Every arm keeps the library's CUDA-only carry
split (``_scan_add``'s rule), gets a function of its own to jit, and the lowered solves are asserted
pairwise distinct. Arms are warm and interleaved; ``solve`` is per iteration (GPU scatter order varies
iteration counts); eigenvalues must agree to ``1e-12`` relative. ``op`` is the operator's device bytes,
``temp`` XLA's ``temp_size_in_bytes`` for the ``(2, N)`` matvec. Fixture as ``poc/sparse_gpu.py``.

Run: uv run python poc/sparse_pairs_tune.py [--log2-sizes 20 22] [--chunks 15 17 19]
     [--variants base sorted merged real] [--rounds 5]
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

sys.path.insert(0, os.path.dirname(__file__))
from eigenpair_check_scale import hamming_shells, patterns, xxz

import rqutils.sqd._solve as solve_mod
import rqutils.sqd._sparse as sm
from rqutils.paulis.symplectic import PauliSumXZ
from rqutils.sqd import Matvec, get_diagonal, uniquify_states
from rqutils.sqd._core import _sqd_inputs

VARIANTS = ("base", "sorted", "merged", "real")
parser = argparse.ArgumentParser()
parser.add_argument("--num-qubits", type=int, default=60)
parser.add_argument("--pattern", default="type1")
parser.add_argument("--delta", type=float, default=0.5)
parser.add_argument("--log2-sizes", type=int, nargs="+", default=[20, 22])
parser.add_argument("--chunks", type=int, nargs="+", default=[15, 17, 19], help="log2 of _CHUNK")
parser.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=VARIANTS)
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


def scan_add(updates, out, xs):
    """``sm._scan_add`` with a sorted flag per update rather than per call."""

    def scan(*parts):
        def body(parts, chunk):
            for index, value, ordered in updates(chunk):
                values = (value.real, value.imag) if len(parts) == 2 else (value,)
                parts = tuple(
                    p.at[..., index].add(v, indices_are_sorted=ordered)
                    for p, v in zip(parts, values, strict=True)
                )
            return parts, None

        return jax.lax.scan(body, parts, xs)[0]

    def fused(out):
        return scan(out)[0]

    if not jnp.iscomplexobj(out):
        return fused(out)

    def split(out):
        re, im = scan(out.real, out.imag)
        return jax.lax.complex(re, im)

    return jax.lax.platform_dependent(out, default=fused, cuda=split)


def apply_sorted(vec, d0, pi, pj, d):
    def updates(chunk):
        i, j, di = chunk
        return [(i, di * vec[..., j], True), (j, jnp.conj(di) * vec[..., i], False)]

    return scan_add(updates, d0 * vec, (pi, pj, d))


def apply_merged(vec, d0, pi, pj, d):
    def updates(chunk):
        i, j, di = chunk
        both = jnp.concatenate([di * vec[..., j], jnp.conj(di) * vec[..., i]], axis=-1)
        return [(jnp.concatenate([i, j]), both, False)]

    return scan_add(updates, d0 * vec, (pi, pj, d))


def apply_real(vec, d0, *sets):
    def updates(chunk):
        i, j, di = chunk
        return [(i, di * vec[..., j]), (j, jnp.conj(di) * vec[..., i])]

    out = d0 * vec
    for k in range(0, len(sets), 3):
        out = sm._scan_add(updates, out, sets[k : k + 3])
    return out


KERNELS = {
    "base": sm._apply_pairs,
    "sorted": apply_sorted,
    "merged": apply_merged,
    "real": apply_real,
}


def real_operator(h, states_u, pairs):
    """``(d0, ri, rj, rd, qi, qj, qd)``: ``"pairs"``' layout per coefficient set, as ``"csr"`` splits."""
    size = states_u.shape[0]
    z, c = jnp.asarray(h.z), jnp.asarray(h.c)
    first = int(0 not in pairs)
    d0 = get_diagonal(z[0], c[0], states_u) if first else jnp.zeros(size, c.dtype)
    groups = range(first, h.x.shape[0])
    coeffs = np.asarray(h.c)
    kmax = max((int(np.count_nonzero(coeffs[g])) for g in groups), default=1)
    real = np.isreal(coeffs).all(axis=1)
    pairs = dict(pairs)

    def alloc(count):
        return [sm._padded(count, f) for f in (size - 1, size - 1, 0)]

    arrays = [d0]
    for subset, c_set in (
        ([g for g in groups if real[g]], c.real),
        ([g for g in groups if not real[g]], c),
    ):
        host = sm._sort_by_target(pairs, subset, size, alloc, both=False)[0]
        t, s, g = (jnp.asarray(a.reshape(-1, sm._CHUNK)) for a in host)
        arrays += [t, s, sm._entry_factors(t, s, g, z, c_set, states_u, kmax)]
    return tuple(arrays)


def timed(fn):
    t0 = time.perf_counter()
    out = jax.block_until_ready(fn())
    return time.perf_counter() - t0, out


ham = PauliSumXZ.from_paulisum(
    xxz(options.num_qubits, options.delta, *patterns(options.num_qubits)[options.pattern])
)
arm = Matvec.PAIRS
shipped = sm._CHUNK
arms = [(c, v) for c in options.chunks for v in options.variants]
ref = (15, "base")
if ref not in arms:
    arms.insert(0, ref)
print(f"{jax.devices()[0].device_kind}, n={options.num_qubits} {options.pattern}, matvec=pairs")
print(
    "arm          N    | solve/iter (ms) x wins | 1-D (ms) x wins | (2, N) (ms) x wins |"
    " op / temp (MiB) | iters | eigval diff | steps"
)
for log2 in options.log2_sizes:
    states = hamming_shells(options.num_qubits, 1 << log2, np.random.default_rng(0))
    h, states_p, size = _sqd_inputs(ham, states, None, False, arm, 0.0, None, (32, 2))
    states_u = uniquify_states(states_p, size)
    pairs = sm._group_pairs(h, states_u)  # one search for every arm
    vecs = [
        jax.random.normal(jax.random.key(0), shape, jnp.complex128)
        for shape in ((size,), (2, size))
    ]
    calls, lowered, info = {}, {}, {}
    for chunk, variant in arms:
        sm._CHUNK = 1 << chunk  # read at build time: the operator's shapes follow it
        if variant == "real":
            operator = jax.block_until_ready(real_operator(h, states_u, pairs))
        else:
            operator = jax.block_until_ready(sm._sparse_operator(h, states_u, arm, pairs))
        kernel = KERNELS[variant]
        sm._SPARSE_APPLY[arm] = kernel
        solve = jax.jit(lambda *a: sm._run_sparse.__wrapped__(*a), static_argnums=(3, 4, 5))
        key = (chunk, variant)
        lowered[key] = solve.lower(h, states_u, operator, size, False, arm).as_text()
        matvec = jax.jit(kernel)
        calls[key] = [functools.partial(solve, h, states_u, operator, size, False, arm)]
        calls[key] += [functools.partial(matvec, vec, *operator) for vec in vecs]
        temp = matvec.lower(vecs[1], *operator).compile().memory_analysis().temp_size_in_bytes
        steps = sum(a.shape[0] for a in operator[1::3])
        info[key] = (sum(a.nbytes for a in operator) / 2**20, temp / 2**20, steps)
        for call in calls[key]:
            call()  # compile
    sm._CHUNK = shipped
    sm._SPARSE_APPLY[arm] = KERNELS["base"]
    assert len(set(lowered.values())) == len(arms), "two arms compiled the same solve"
    products = {key: np.asarray(calls[key][2]()) for key in arms}
    times = {key: ([], [], []) for key in arms}
    eig, iters = {key: [] for key in arms}, {key: set() for key in arms}
    for _ in range(options.rounds):
        for key in arms:
            ITERATIONS.clear()
            t, result = timed(calls[key][0])
            times[key][0].append(t / ITERATIONS[-1])
            for k in (1, 2):
                times[key][k].append(timed(calls[key][k])[0])
            eig[key].append(float(result.eigval))
            iters[key].add(ITERATIONS[-1])
    base_eig = eig[ref][0]
    for key in arms:
        diff = max(abs(e - base_eig) for e in eig[key]) / abs(base_eig)
        assert diff < 1e-12, (key, eig[key])
        err = np.abs(products[key] - products[ref]).max() / np.abs(products[ref]).max()
        assert err < 1e-12, (key, err)
        cols = []
        for mine, theirs in zip(times[key], times[ref], strict=True):
            med, med0 = statistics.median(mine), statistics.median(theirs)
            wins = sum(a < b for a, b in zip(mine, theirs, strict=True))
            cols.append(f"{med * 1e3:7.2f} {med0 / med:5.2f}x {wins}/{options.rounds}")
        op, temp, steps = info[key]
        name = f"{key[1]}@2^{key[0]}"
        print(
            f"{name:12} 2^{log2} | {' | '.join(cols)} | {op:7.1f} / {temp:6.1f} |"
            f" {sorted(iters[key])} | {diff:.1e} | {steps}",
            flush=True,
        )
