"""A/B of sparse-kernel levers on a GPU: ``"pairs"``' chunk, sort hint, merged scatter, real factors and atomics.

After ``0d25235`` a GH200 ``"pairs"`` matvec is 84-89% scatters (``poc/sparse/split.md`` §4), stepping
``_CHUNK = 32768`` entries at a time: 320 steps and 1603 launches per call at ``type2`` ``2^22``, each
kernel a fraction of the GPU. Arms are every ``--chunks`` × ``--variants``:

- ``base``: the library kernel, ``_apply_pairs``.
- ``sorted``: the ``out[i]`` scatter told ``indices_are_sorted`` (pairs are sorted by ``i``); ``out[j]``
  unchanged.
- ``merged``: both directions as one scatter of the concatenated indices and values.
- ``real``: ``"csr"``'s split, real groups' pairs with ``float64`` factors and the rest ``complex128``,
  two scans; the memory lever.
- ``unique``: one X group per scan step, both scatters ``unique_indices=True`` -- a group's pairs are a
  perfect matching (asserted), so no atomics are needed; padding takes distinct out-of-bounds indices,
  dropped. It ignores ``--chunks`` (run once, at the first) and pays the padding to the largest group.
- ``unique-exact``: ``unique`` without that padding -- no scan, a Python loop over groups, each sized to
  its own pair count rounded to ``_size_class``; more compiled shapes, no padding to the largest group.

``--matvec csr`` runs ``"csr"`` instead, with ``base`` (the library, which since this ran drops
``indices_are_sorted`` on CUDA) and ``sorted`` (the hint forced back, as the library had it).
``--matvec ell`` runs ``"ell"``, whose variants are width grids: ``base`` (the library's ×1.25),
``grid1.5`` and ``grid2`` — fewer buckets, so fewer scans and compiled shapes, at more padding.

The reference is ``base`` at ``2^15``, the CPU's chunk (the GPU ships ``"pairs"`` at ``2^19`` since
this ran, ``_GPU_PAIRS_CHUNK``), for either ``--matvec``. Every arm keeps the library's CUDA-only carry
split (``_scan_add``'s rule), gets a function of its own to jit, and the solves are asserted
pairwise distinct as traced. Arms are warm and interleaved; ``solve`` is per iteration (GPU scatter order varies
iteration counts); eigenvalues must agree to ``1e-12`` relative. ``op`` is the operator's device bytes,
``temp`` XLA's ``temp_size_in_bytes`` for the ``(2, N)`` matvec. Fixture as ``poc/sparse/gpu.py``.

Run: uv run python poc/sparse/tune.py [--log2-sizes 20 22] [--chunks 15 17 19]
     [--variants base sorted merged real] [--rounds 5] [--matvec csr|ell]
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

VARIANTS = {
    "pairs": ("base", "sorted", "merged", "real", "unique", "unique-exact"),
    "csr": ("base", "sorted"),
    "ell": ("base", "grid1.5", "grid2"),
}
parser = argparse.ArgumentParser()
parser.add_argument("--num-qubits", type=int, default=60)
parser.add_argument("--pattern", default="type1")
parser.add_argument("--delta", type=float, default=0.5)
parser.add_argument("--log2-sizes", type=int, nargs="+", default=[20, 22])
parser.add_argument("--chunks", type=int, nargs="+", default=[15, 17, 19], help="log2 of _CHUNK")
parser.add_argument("--matvec", default="pairs", choices=list(VARIANTS))
parser.add_argument("--variants", nargs="+", help="default: every variant of --matvec")
parser.add_argument("--rounds", type=int, default=5)
options = parser.parse_args()
options.variants = options.variants or list(VARIANTS[options.matvec])
if unknown := set(options.variants) - set(VARIANTS[options.matvec]):
    parser.error(f"--matvec {options.matvec} has no variants {sorted(unknown)}")

ITERATIONS = []
_ground_locg = solve_mod.ground_locg


def counted_ground_locg(*args, **kwargs):
    """``ground_locg``, recording each solve's iteration count on the host."""
    out = _ground_locg(*args, **kwargs)
    jax.debug.callback(lambda niter: ITERATIONS.append(int(niter)), out[2])
    return out


solve_mod.ground_locg = counted_ground_locg  # ty: ignore[invalid-assignment]


def scan_add(updates, out, xs):
    """``sm._scan_add`` with ``.add`` keywords per update -- ``(index, value, kwargs)`` -- not per call."""

    def scan(*parts):
        def body(parts, chunk):
            for index, value, kwargs in updates(chunk):
                values = (value.real, value.imag) if len(parts) == 2 else (value,)
                parts = tuple(
                    p.at[..., index].add(v, **kwargs) for p, v in zip(parts, values, strict=True)
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
        return [
            (i, di * vec[..., j], {"indices_are_sorted": True}),
            (j, jnp.conj(di) * vec[..., i], {}),
        ]

    return scan_add(updates, d0 * vec, (pi, pj, d))


def apply_merged(vec, d0, pi, pj, d):
    def updates(chunk):
        i, j, di = chunk
        both = jnp.concatenate([di * vec[..., j], jnp.conj(di) * vec[..., i]], axis=-1)
        return [(jnp.concatenate([i, j]), both, {})]

    return scan_add(updates, d0 * vec, (pi, pj, d))


def apply_real(vec, d0, *sets):
    def updates(chunk):
        i, j, di = chunk
        return [(i, di * vec[..., j]), (j, jnp.conj(di) * vec[..., i])]

    out = d0 * vec
    for k in range(0, len(sets), 3):
        out = sm._scan_add(updates, out, sets[k : k + 3])
    return out


def apply_csr_sorted(vec, d0, *entries):
    def updates(chunk):
        ti, si, di = chunk
        return [(ti, di * vec[..., si], {"indices_are_sorted": True})]

    out = d0 * vec
    for k in range(0, len(entries), 3):
        out = scan_add(updates, out, entries[k : k + 3])
    return out


def apply_unique(vec, d0, pi, pj, d):
    unique = {"unique_indices": True, "mode": "drop"}

    def updates(chunk):
        i, j, di = chunk
        vj, vi = (vec.at[..., k].get(mode="fill", fill_value=0) for k in (j, i))
        return [(i, di * vj, unique), (j, jnp.conj(di) * vi, unique)]

    return scan_add(updates, d0 * vec, (pi, pj, d))


def apply_unique_exact(vec, d0, *sets):
    """``apply_unique`` over per-group ``(i, j, d)`` sets, unrolled in Python rather than scanned."""
    unique = {"unique_indices": True, "mode": "drop"}

    def run(parts):
        for k in range(0, len(sets), 3):
            i, j, d = sets[k : k + 3]
            vj, vi = (vec.at[..., n].get(mode="fill", fill_value=0) for n in (j, i))
            for index, value in ((i, d * vj), (j, jnp.conj(d) * vi)):
                values = (value.real, value.imag) if len(parts) == 2 else (value,)
                parts = tuple(
                    p.at[..., index].add(v, **unique) for p, v in zip(parts, values, strict=True)
                )
        return parts

    def default(out):
        return run((out,))[0]

    def cuda(out):
        if not jnp.iscomplexobj(out):
            return run((out,))[0]
        re, im = run((out.real, out.imag))
        return jax.lax.complex(re, im)

    return jax.lax.platform_dependent(d0 * vec, default=default, cuda=cuda)


KERNELS = {
    "pairs": {
        "base": sm._apply_pairs,
        "sorted": apply_sorted,
        "merged": apply_merged,
        "real": apply_real,
        "unique": apply_unique,
        "unique-exact": apply_unique_exact,
    },
    "csr": {"base": sm._apply_csr, "sorted": apply_csr_sorted},
    "ell": dict.fromkeys(VARIANTS["ell"], sm._apply_ell),  # one kernel; the grid is the build's
}[options.matvec]


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
        return [sm._padded(count, f, sm._CHUNK) for f in (size - 1, size - 1, 0)]

    arrays = [d0]
    for subset, c_set in (
        ([g for g in groups if real[g]], c.real),
        ([g for g in groups if not real[g]], c),
    ):
        host = sm._sort_by_target(pairs, subset, size, alloc, both=False)[0]
        t, s, g = (jnp.asarray(a.reshape(-1, sm._CHUNK)) for a in host)
        arrays += [t, s, sm._entry_factors(t, s, g, z, c_set, states_u, kmax)]
    return tuple(arrays)


def unique_operator(h, states_u, pairs):
    """``(d0, i, j, d)`` with one X group per row, padded to the largest with distinct out-of-bounds indices."""
    size = states_u.shape[0]
    z, c = jnp.asarray(h.z), jnp.asarray(h.c)
    first = int(0 not in pairs)
    d0 = get_diagonal(z[0], c[0], states_u) if first else jnp.zeros(size, c.dtype)
    groups = sorted(pairs)
    coeffs = np.asarray(h.c)
    kmax = max((int(np.count_nonzero(coeffs[g])) for g in groups), default=1)
    width = max(1, max((len(pairs[g][0]) for g in groups), default=0))
    t, s, grp = (np.zeros((len(groups), width), np.int32) for _ in range(3))
    real = np.zeros((len(groups), width), bool)
    for row, g in enumerate(groups):
        i, j = pairs[g]
        both = np.concatenate([i, j])
        assert len(np.unique(both)) == len(both), f"group {g} is not a matching"
        t[row, : len(i)], s[row, : len(i)], grp[row] = i, j, g
        real[row, : len(i)] = True
    # Padding is t == s == 0, which _entry_factors zeroes; only then do the indices go out of bounds.
    d = sm._entry_factors(jnp.asarray(t), jnp.asarray(s), jnp.asarray(grp), z, c, states_u, kmax)
    lane = np.arange(width, dtype=np.int32)
    t = np.where(real, t, size + lane)
    s = np.where(real, s, size + width + lane)
    return d0, jnp.asarray(t), jnp.asarray(s), d


def unique_exact_operator(h, states_u, pairs):
    """``(d0, i_g, j_g, d_g, ...)``: ``unique_operator``'s rows, each cut to its own size class."""
    d0, t, s, d = unique_operator(h, states_u, pairs)
    counts = [len(pairs[g][0]) for g in sorted(pairs)]
    arrays = [d0]
    for row, n in enumerate(counts):
        m = min(max(1, sm._size_class(n)), t.shape[1])  # the padding past n is out of bounds
        arrays += [t[row, :m], s[row, :m], d[row, :m]]
    return tuple(arrays)


def width_grid(factor):
    """``_ELL_WIDTHS``' construction for another ratio, kept below ``2^31``."""
    steps = int(np.log(2**31) / np.log(factor))
    return np.unique(np.ceil(factor ** np.arange(steps)).astype(np.int64))


def timed(fn):
    t0 = time.perf_counter()
    out = jax.block_until_ready(fn())
    return time.perf_counter() - t0, out


ham = PauliSumXZ.from_paulisum(
    xxz(options.num_qubits, options.delta, *patterns(options.num_qubits)[options.pattern])
)
arm = Matvec(options.matvec)
shipped = sm._CHUNK, sm._chunk, sm._ELL_WIDTHS
arms = [
    (c, v)
    for c in options.chunks
    for v in options.variants
    if not v.startswith("unique") or c == options.chunks[0]
]
ref = (15, "base")
if ref not in arms:
    arms.insert(0, ref)
print(f"{jax.devices()[0].device_kind}, n={options.num_qubits} {options.pattern}, matvec={arm}")
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
        # Read at build time, and the operator's shapes follow; _chunk too, or a GPU "pairs" build
        # would take _GPU_PAIRS_CHUNK in every arm.
        sm._CHUNK = 1 << chunk
        sm._chunk = lambda matvec: sm._CHUNK  # ty: ignore[invalid-assignment]
        sm._ELL_WIDTHS = (
            width_grid(float(variant[4:])) if variant.startswith("grid") else shipped[2]
        )
        if variant == "real":
            operator = jax.block_until_ready(real_operator(h, states_u, pairs))
        elif variant == "unique":
            operator = jax.block_until_ready(unique_operator(h, states_u, pairs))
        elif variant == "unique-exact":
            operator = jax.block_until_ready(unique_exact_operator(h, states_u, pairs))
        else:
            operator = jax.block_until_ready(sm._sparse_operator(h, states_u, arm, pairs))
        kernel = KERNELS[variant]
        sm._SPARSE_APPLY[arm] = kernel
        solve = jax.jit(lambda *a: sm._run_sparse.__wrapped__(*a), static_argnums=(3, 4, 5))
        key = (chunk, variant)
        # The traced jaxpr keeps every platform branch, which lowering for one backend drops.
        lowered[key] = str(solve.trace(h, states_u, operator, size, False, arm).jaxpr)
        matvec = jax.jit(kernel)
        calls[key] = [functools.partial(solve, h, states_u, operator, size, False, arm)]
        calls[key] += [functools.partial(matvec, vec, *operator) for vec in vecs]
        temp = matvec.lower(vecs[1], *operator).compile().memory_analysis().temp_size_in_bytes
        # A 1-D set is one unscanned group (unique-exact); a 2-D one scans its leading axis.
        steps = sum(a.shape[0] if a.ndim == 2 else 1 for a in operator[1::3])
        info[key] = (sum(a.nbytes for a in operator) / 2**20, temp / 2**20, steps)
        for call in calls[key]:
            call()  # compile
    sm._CHUNK, sm._chunk, sm._ELL_WIDTHS = shipped
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
