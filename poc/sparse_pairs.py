"""Sparse transition pairs instead of dense source indices -- every prototype and measurement in one script.

Item 8 of ``markdown/improvement-ideas-2026-09-25.md``; the write-up with every result is
``poc/sparse-pairs.md``. At ``matvec="indices"``/``"tables"`` ``sqd`` stores one ``int32`` source per
``(X group, state)`` (``4*J`` B/slot) though on sampled subspaces most are ``-1``. XOR is an involution, so
each real transition of a group ``g != 0`` is a pair; storing it once, and computing its diagonal once since
``H_ji = conj(H_ij)`` exactly for one X signature, removes the placeholders and half the diagonal work.

Arms (the identity-X group is always a plain ``d_0 * v``; every other arm scans fixed ``2^15``-entry chunks):

* ``(1,0)``, ``(1,2)`` -- today's kernels, the baselines;
* ``P0`` / ``P2``  -- pairs ``(i, j, g)``, two scatters, diagonal recomputed / cached;
* ``P2R``          -- P2 with ``float64`` factors for real groups, ``complex128`` only for the rest;
* ``C0`` / ``C2``  -- both directions ``(t, s, g)`` sorted by target (CSR), one sorted scatter;
* ``C0i16``        -- C0 with ``int16`` group ids (``int32`` if ``J > 32768``: ``astype`` would wrap);
* ``C2R``          -- C2 with ``float64`` diagonals for real groups, ``complex128`` only for the rest;
* ``seg``          -- C2 as one unchunked ``segment_sum``;
* ``C2W``          -- C2R as row-window reductions: row-aligned chunks, a ``ROWS``-segment sum each;
* ``P2F``/``C2RF``  -- P2/C2R with every gather and scatter declared ``promise_in_bounds``;
* ``ELL``          -- rows bucketed by exact degree: scatter-free row sums, one write per row;
* ``HYB``          -- one natural-order ELL block of width ``W`` (contiguous writes) + CSR overflow;
* ``ELLC``         -- ELL with degrees rounded up to geometric classes: far fewer buckets and compiles;
* ``ELLD``         -- ELL with classes chosen per histogram by a DP (padding bytes + per-bucket cost);
* ``...+RCM``      -- states relabelled by reverse Cuthill--McKee first.

Subcommands (fixture: spinchain's open-XXZ ``xxz``, ``poc/eigenpair_check_scale``; ``--subspace`` picks
Hamming shells or ``recovery``, grown by spinchain-style ranked H-expansion from a small shell core):

* ``matvec``  -- ns/state of one ``(2, N)`` matvec per arm over sizes, each checked against ``(1,0)``;
* ``solve``   -- whole ``ground_locg`` solves at one size, with XLA memory (inputs + temp);
* ``peak``    -- setup-inclusive peak RSS, one fresh process per arm and size;
* ``general`` -- P0/P2/C2R at high hit rate: periodic Neel-Krylov XXZ, molecular-like JW;
* ``batch``   -- one ``(2, N)`` application against two ``(N,)`` and an ``(N, 2)`` layout, then whole
  solves with ``ground_locg``'s ``batch_matvec`` on and off.

Run: uv run python poc/sparse_pairs.py matvec [--log2-sizes 15 17 19 21] [--arms ...]
     uv run python poc/sparse_pairs.py solve [--log2-size 17]
     uv run python poc/sparse_pairs.py peak [--log2-sizes 17 19 21]
     uv run python poc/sparse_pairs.py general
Set XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" for one thread.
"""

import argparse
import functools
import json
import os
import resource
import subprocess
import sys
import time

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
from eigenpair_check_scale import hamming_shells, patterns, xxz
from scipy.sparse.csgraph import reverse_cuthill_mckee

from rqutils.ground_locg import ground_locg
from rqutils.paulis.symplectic import PauliSumXZ
from rqutils.sqd import Matvec, get_diagonal, get_xsource, run_sqd, sqd, uniquify_states
from rqutils.sqd._dense import _apply_h_kernel, _pack_scanned
from rqutils.sqd._diagonal import _z_parity
from rqutils.sqd._solve import _spread_seed
from rqutils.sqd._states import _is_filler, _pad_states

CHUNK = 1 << 15
ARMS = (
    "(1,0)",
    "(1,2)",
    "P0",
    "P2",
    "P2R",
    "C0",
    "C0i16",
    "C2",
    "C2R",
    "seg",
    "C2W",
    "P2F",
    "C2RF",
    "ELL",
    "HYB",
    "ELLC",
    "ELLD",
    "C0i16+RCM",
    "C2+RCM",
    "C2R+RCM",
)


# ---------------------------------------------------------------------------------------------- pairs


def build_pairs(ham: PauliSumXZ, states_u, dim: int):
    """``(pi, pj, pg, hit)``: each real transition of each non-identity group once (``i < j``).

    One group's search at a time, so the only transient is that group's ``int32`` row (~4 B/slot) --
    never ``(1, 0)``'s full ``(J, N)`` array. ``hit`` is the fraction of ``(group, state)`` slots that land
    inside the subspace.
    """
    assert bool(np.all(np.asarray(ham.x[0]) == 0)), "group 0 must be the identity-X group"
    rows = np.arange(states_u.shape[0], dtype=np.int32)
    pi, pj, pg, hits = [], [], [], 0
    for g in range(1, ham.x.shape[0]):
        j = np.asarray(get_xsource(ham.x[g], states_u))
        hits += int((j[:dim] >= 0).sum())
        keep = (j > rows) & (rows < dim)  # j > i >= 0: each pair once, filler rows excluded
        i_g, j_g = rows[keep], j[keep].astype(np.int32)
        assert np.array_equal(j[j_g], i_g), f"group {g} is not an involution"
        pi.append(i_g)
        pj.append(j_g)
        pg.append(np.full(len(i_g), g, np.int32))
    hit = (hits + dim) / (ham.x.shape[0] * dim)  # the identity group hits every state
    return np.concatenate(pi), np.concatenate(pj), np.concatenate(pg), hit


def directed_by_target(pi, pj, pg):
    """Both directions as ``(t, s, g)`` sorted by target: ``out[t] += d_g(t) * vec[s]``, CSR order."""
    t, s, g = np.concatenate([pi, pj]), np.concatenate([pj, pi]), np.concatenate([pg, pg])
    order = np.argsort(t, kind="stable")
    return t[order], s[order], g[order]


def csr_chunked(pi, pj, pg, groups, num_groups, size, gdtype):
    """Both directions of the pairs in ``groups``, counting-sorted by target straight into padded chunks.

    No concatenation, no ``argsort`` index, no second copy for padding: the outputs are allocated once at
    their chunked length and filled in place. Within one X group every state belongs to exactly one pair,
    so a row occurs at most once per group and each group's fill is one conflict-free assignment.
    Returns device ``(t, s, g)`` of shape ``(chunks, CHUNK)``; padding is ``(size-1, size-1, 0)``.
    """
    bounds = np.searchsorted(pg, np.arange(num_groups + 1))  # pairs are built group by group
    spans = [(bounds[g], bounds[g + 1]) for g in groups]
    start = np.zeros(size + 1, np.int64)
    for a, b in spans:
        start[pi[a:b] + 1] += 1
        start[pj[a:b] + 1] += 1
    np.cumsum(start, out=start)  # start[r] is row r's first slot
    total = -(-int(start[-1]) // CHUNK) * CHUNK
    assert total < np.iinfo(np.int32).max, "entry count overflows int32 targets"
    t = np.full(total, size - 1, np.int32)
    s = np.full(total, size - 1, np.int32)
    g_out = np.zeros(total, gdtype)
    for g, (a, b) in zip(groups, spans):
        for rows, srcs in ((pi[a:b], pj[a:b]), (pj[a:b], pi[a:b])):
            pos = start[rows]
            t[pos], s[pos], g_out[pos] = rows, srcs, g
            start[rows] += 1
    # One host array alive beside its device copy at a time (on CPU they share memory).
    ct = jnp.asarray(t.reshape(-1, CHUNK))
    del t
    cs = jnp.asarray(s.reshape(-1, CHUNK))
    del s
    cg = jnp.asarray(g_out.reshape(-1, CHUNK))
    del g_out
    return ct, cs, cg


def chunked(arrays, pad):
    """Pad each 1-D array with ``pad`` to a multiple of ``CHUNK`` and reshape to ``(chunks, CHUNK)``.

    Pairs pad with ``(0, 0)``, CSR entries with the last filler row so targets stay sorted; either way a
    padding entry has equal endpoints, which is how its diagonal is zeroed.
    """
    total = -(-len(arrays[0]) // CHUNK) * CHUNK
    return tuple(
        jnp.pad(jnp.asarray(a), (0, total - len(a)), constant_values=pad).reshape(-1, CHUNK)
        for a in arrays
    )


def pair_diagonal(states_t, z_t, c_t):
    """``sum_k c_k (1 - 2*parity(state_t & z_k))`` per entry, over the (small) per-group term axis."""
    signs = 1.0 - 2.0 * jax.vmap(_z_parity, in_axes=(None, 1), out_axes=1)(states_t, z_t)
    return jnp.sum(c_t * signs, axis=-1)


@functools.partial(jax.jit, static_argnames="kmax")
def chunked_diagonals(ct, cs, cg, z, c, states, kmax):
    """``d_g(t)`` per chunked entry from the target's state, zero on padding: one compile, O(chunk) temp."""

    def one(chunk):
        t, s, g = chunk
        return jnp.where(t == s, 0.0, pair_diagonal(states[t], z[g][:, :kmax], c[g][:, :kmax]))

    return jax.lax.map(one, (ct, cs, cg))


# -------------------------------------------------------------------------------------------- kernels


@functools.partial(jax.jit, static_argnames="kmax")
def matvec_p0(vec, pi, pj, pg, z, c, states, kmax):
    def body(out, chunk):
        i, j, g = chunk
        d = jnp.where(i == j, 0.0, pair_diagonal(states[i], z[g][:, :kmax], c[g][:, :kmax]))
        out = out.at[..., i].add(d * vec[..., j])
        return out.at[..., j].add(jnp.conj(d) * vec[..., i]), None

    return jax.lax.scan(body, get_diagonal(z[0], c[0], states) * vec, (pi, pj, pg))[0]


@jax.jit
def matvec_p2(vec, pi, pj, d, d0):
    return _pair_scan(vec, d0 * vec, pi, pj, d)


def _pair_scan(vec, out, pi, pj, d):
    """Per pair ``out[i] += d * vec[j]`` and ``out[j] += conj(d) * vec[i]``; conj is a no-op on real d."""

    def body(acc, chunk):
        i, j, di = chunk
        acc = acc.at[..., i].add(di * vec[..., j])
        return acc.at[..., j].add(jnp.conj(di) * vec[..., i]), None

    return jax.lax.scan(body, out, (pi, pj, d))[0]


@jax.jit
def matvec_p2r(vec, ri, rj, rd, qi, qj, qd, d0):
    """P2 over a ``float64``-factor set (real groups) and a ``complex128`` one (the rest), as C2R does."""
    return _pair_scan(vec, _pair_scan(vec, d0 * vec, ri, rj, rd), qi, qj, qd)


def real_split(pi, pj, pg, real_group):
    """Chunked ``(i, j, g)`` for the real groups' pairs and for the rest, each filled group by group
    into its final padded arrays: masking copies out of the full list measured +12% peak at ``2^21``."""
    bounds = np.searchsorted(pg, np.arange(len(real_group) + 1))  # pairs are built group by group
    sets = []
    for want in (True, False):
        gs = [g for g in range(1, len(real_group)) if real_group[g] == want]
        total = -(-int(sum(bounds[g + 1] - bounds[g] for g in gs)) // CHUNK) * CHUNK
        out = [np.zeros(total, np.int32) for _ in range(3)]
        pos = 0
        for g in gs:
            a, b = bounds[g], bounds[g + 1]
            for dst, src in zip(out, (pi, pj, pg)):
                dst[pos : pos + b - a] = src[a:b]
            pos += b - a
        sets.append([jnp.asarray(a.reshape(-1, CHUNK)) for a in out])
    return sets


@functools.partial(jax.jit, static_argnames="kmax")
def matvec_c0(vec, t, s, g, z, c, states, kmax):
    def body(out, chunk):
        ti, si, gi = chunk
        d = jnp.where(ti == si, 0.0, pair_diagonal(states[ti], z[gi][:, :kmax], c[gi][:, :kmax]))
        return out.at[..., ti].add(d * vec[..., si], indices_are_sorted=True), None

    return jax.lax.scan(body, get_diagonal(z[0], c[0], states) * vec, (t, s, g))[0]


def _sorted_scatter(vec, out, t, s, d):
    def body(acc, chunk):
        ti, si, di = chunk
        return acc.at[..., ti].add(di * vec[..., si], indices_are_sorted=True), None

    return jax.lax.scan(body, out, (t, s, d))[0]


@jax.jit
def matvec_c2(vec, t, s, d, d0):
    return _sorted_scatter(vec, d0 * vec, t, s, d)


@jax.jit
def matvec_c2r(vec, rt, rs, rd, qt, qs, qd, d0):
    """C2 over a ``float64``-diagonal set (real groups) and a ``complex128`` one (the rest)."""
    return _sorted_scatter(vec, _sorted_scatter(vec, d0 * vec, rt, rs, rd), qt, qs, qd)


@jax.jit
def matvec_seg(vec, t, s, d, d0):
    """One sorted ``segment_sum`` over every entry; unchunkable, since it always emits all N segments."""
    summed = jax.ops.segment_sum(
        (d * vec[..., s]).T, t, num_segments=vec.shape[-1], indices_are_sorted=True
    )
    return d0 * vec + summed.T


IB = "promise_in_bounds"  # every index is in bounds by construction: skip XLA's clamping


@jax.jit
def matvec_p2f(vec, pi, pj, d, d0):
    """P2 with every gather and scatter declared in bounds."""

    def body(acc, chunk):
        i, j, di = chunk
        acc = acc.at[..., i].add(di * vec.at[..., j].get(mode=IB), mode=IB)
        return acc.at[..., j].add(jnp.conj(di) * vec.at[..., i].get(mode=IB), mode=IB), None

    return jax.lax.scan(body, d0 * vec, (pi, pj, d))[0]


def _sorted_scatter_f(vec, out, t, s, d):
    def body(acc, chunk):
        ti, si, di = chunk
        gathered = vec.at[..., si].get(mode=IB)
        return acc.at[..., ti].add(di * gathered, indices_are_sorted=True, mode=IB), None

    return jax.lax.scan(body, out, (t, s, d))[0]


@jax.jit
def matvec_c2rf(vec, rt, rs, rd, qt, qs, qd, d0):
    """C2R with every gather and scatter declared in bounds."""
    return _sorted_scatter_f(vec, _sorted_scatter_f(vec, d0 * vec, rt, rs, rd), qt, qs, qd)


@functools.partial(jax.jit, static_argnames="kmax")
def bucket_factors(rows, grp, n, z, c, states, kmax):
    """``d_g(row)`` for every ``(piece, R, k)`` bucket entry, zero on the padding rows past ``n``."""
    per, k = grp.shape[1], grp.shape[2]

    def one(args):
        p, r, g = args
        tt, gg = jnp.repeat(r, k), g.reshape(-1)
        fac = pair_diagonal(states[tt], z[gg][:, :kmax], c[gg][:, :kmax]).reshape(per, k)
        return jnp.where((p * per + jnp.arange(per) < n)[:, None], fac, jnp.zeros_like(fac))

    return jax.lax.map(one, (jnp.arange(rows.shape[0]), rows, grp))


def host_csr(pi, pj, pg, groups, num_groups, size):
    """``(deg, indptr, src, grp)`` of the ``groups``' transitions, counting-sorted by target on the host.

    The target is implied by ``indptr``; within one group every row occurs at most once per direction.
    """
    bounds = np.searchsorted(pg, np.arange(num_groups + 1))  # pairs are built group by group
    deg = np.zeros(size, np.int64)
    for g in groups:
        deg[pi[bounds[g] : bounds[g + 1]]] += 1
        deg[pj[bounds[g] : bounds[g + 1]]] += 1
    indptr = np.zeros(size + 1, np.int64)
    np.cumsum(deg, out=indptr[1:])
    src = np.empty(int(indptr[-1]), np.int32)
    grp = np.empty(int(indptr[-1]), np.int32)
    pos = indptr[:-1].copy()
    for g in groups:
        a, b = bounds[g], bounds[g + 1]
        for rows, srcs in ((pi[a:b], pj[a:b]), (pj[a:b], pi[a:b])):
            p = pos[rows]
            src[p], grp[p] = srcs, g
            pos[rows] += 1
    return deg, indptr, src, grp


def hyb_width(deg, factor_bytes, size):
    """The ELL width minimising bytes: ``size * W`` padded slots against the overflow's CSR entries."""
    hist = np.bincount(deg)
    ks = np.arange(len(hist))
    above = np.cumsum(hist[::-1])[::-1]  # rows with degree >= k
    mass = np.cumsum((hist * ks)[::-1])[::-1]  # entries in rows with degree >= k
    over = np.array([mass[w + 1] - w * above[w + 1] if w + 1 < len(hist) else 0 for w in ks])
    cost = size * ks * (4 + factor_bytes) + over * (8 + factor_bytes)
    return int(np.argmin(cost))


@functools.partial(jax.jit, static_argnames="kmax")
def ell_block_factors(grp, deg, z, c, states, kmax):
    """``d_g(row)`` over an ``(pieces, R, W)`` ELL block whose rows are ``p * R + r``; zero past ``deg``."""
    _, rows, width = grp.shape

    def one(args):
        p, g, d = args
        tt = jnp.repeat(p * rows + jnp.arange(rows), width)
        gg = g.reshape(-1)
        fac = pair_diagonal(states[tt], z[gg][:, :kmax], c[gg][:, :kmax]).reshape(rows, width)
        return jnp.where(jnp.arange(width) < d[:, None], fac, jnp.zeros_like(fac))

    return jax.lax.map(one, (jnp.arange(grp.shape[0]), grp, deg))


def hyb_set(pi, pj, pg, groups, num_groups, size, z, c, states, kmax):
    """One HYB set: an ELL block over every row in natural order, width ``W``, plus the overflow as
    target-sorted CSR. Returns ``(ell_src, ell_fac, ovt, os, od)``; factors in one jitted call each."""
    deg, indptr, src, grp = host_csr(pi, pj, pg, groups, num_groups, size)
    width = hyb_width(deg, 8 if np.isrealobj(np.asarray(c)) else 16, size) if len(src) else 0
    rows = 1 << max(0, int(np.log2(max(1, CHUNK // max(width, 1)))))  # a power of two divides size
    rows = min(rows, size)
    es = np.zeros((size, width), np.int32)
    eg = np.zeros((size, width), np.int32)
    for w in range(width):
        r = np.flatnonzero(deg > w)
        es[r, w], eg[r, w] = src[indptr[r] + w], grp[indptr[r] + w]
    over = np.flatnonzero(deg > width)
    counts = deg[over] - width
    idx = np.repeat(indptr[over] + width - np.concatenate([[0], np.cumsum(counts)[:-1]]), counts)
    idx = (idx + np.arange(len(idx))).astype(np.int64)
    ovt, os_, og = np.repeat(over, counts).astype(np.int32), src[idx], grp[idx]
    del src, grp, idx
    shape = (size // rows, rows, width)
    es_d, eg_d = jnp.asarray(es.reshape(shape)), jnp.asarray(eg.reshape(shape))
    del es, eg
    ef = ell_block_factors(eg_d, jnp.asarray(deg.reshape(shape[:2])), z, c, states, kmax)
    del eg_d
    ct, cs, cg = chunked((ovt, os_, og), pad=size - 1)
    od = chunked_diagonals(ct, cs, cg, z, c, states, kmax)
    return es_d, ef, ct, cs, od


def ell_buckets(pi, pj, pg, groups, num_groups, size, z, c, states, kmax):
    """The ``groups``' transitions bucketed by row degree, straight from the per-group pairs.

    Per degree ``k``: ``(rows, src, fac)`` of shape ``(pieces, R)``, ``(pieces, R, k)``, ``(pieces, R, k)``,
    ``R ~ CHUNK // k``. A counting sort by target builds host ``src``/``grp`` only (the target is implied
    by the row offsets); padding rows repeat a real row with ``fac = 0``, so ``out`` needs no dummy slot.
    """
    deg, indptr, src, grp = host_csr(pi, pj, pg, groups, num_groups, size)
    buckets = []
    for k in np.unique(deg[deg > 0]):
        k = int(k)
        rows = np.flatnonzero(deg == k).astype(np.int32)
        n = len(rows)
        per = min(max(1, CHUNK // k), n)  # a small bucket is one unpadded piece
        total = -(-n // per) * per
        r = np.full(total, rows[0], np.int32)
        r[:n] = rows
        idx = indptr[rows].astype(np.int32)[:, None] + np.arange(k, dtype=np.int32)
        sb, gb = np.zeros((total, k), np.int32), np.zeros((total, k), np.int32)
        sb[:n], gb[:n] = src[idx], grp[idx]
        del idx
        rd, sd, gd = (jnp.asarray(x.reshape(-1, per, *x.shape[1:])) for x in (r, sb, gb))
        del r, sb, gb
        buckets += [rd, sd, bucket_factors(rd, gd, n, z, c, states, kmax)]
        del gd
    return buckets


@functools.partial(jax.jit, static_argnames="kmax")
def chunk_factors(t, s, g, z, c, states, kmax):
    """``d_g(t)`` for one flat ``(CHUNK,)`` chunk, zero where ``t == s``: one compile per dtype."""
    return jnp.where(t == s, 0.0, pair_diagonal(states[t], z[g][:, :kmax], c[g][:, :kmax]))


def geometric_classes(deg, ratio=1.25):
    """Class tops on a ``ratio``-geometric grid up to the largest degree (ELLC)."""
    top = int(deg.max())
    grid = np.ceil(ratio ** np.arange(int(np.log(max(top, 2)) / np.log(ratio)) + 2)).astype(
        np.int64
    )
    grid = np.unique(grid)
    return grid[: np.searchsorted(grid, top) + 1]


def optimal_classes(deg, slot_bytes, bucket_bytes):
    """Class tops minimising padding bytes + ``bucket_bytes`` per class (ELLD): an O(m^2) DP over the
    sorted distinct degrees, each class a contiguous run padded to its largest degree."""
    d, h = np.unique(deg[deg > 0], return_counts=True)
    m = len(d)
    best, cut = np.full(m + 1, np.inf), np.zeros(m + 1, np.int64)
    best[0] = 0.0
    hd, hc = np.concatenate([[0], np.cumsum(h * d)]), np.concatenate([[0], np.cumsum(h)])
    for b in range(1, m + 1):  # class d[a:b], padded to d[b - 1]
        a = np.arange(b)
        pad = (d[b - 1] * (hc[b] - hc[a]) - (hd[b] - hd[a])) * slot_bytes
        cost = best[a] + pad + bucket_bytes
        k = int(np.argmin(cost))
        best[b], cut[b] = cost[k], k
    tops, b = [], m
    while b:
        tops.append(int(d[b - 1]))
        b = int(cut[b])
    return np.array(sorted(tops), np.int64)


def ell_class_buckets(pi, pj, pg, groups, num_groups, size, z, c, states, kmax, classes):
    """ELL with each row's degree rounded up to a class top from ``classes(deg, slot_bytes)``.

    Factors are computed per bucket in fixed ``(CHUNK,)`` flat chunks: a padded or invalid slot gets
    source = target, so ``chunk_factors``' ``t == s`` rule zeroes it with no mask and no flat round trip.
    """
    deg, indptr, src, grp = host_csr(pi, pj, pg, groups, num_groups, size)
    if not len(src):
        return []
    slot = 4 + (8 if np.isrealobj(np.asarray(c)) else 16)
    tops = classes(deg, slot)
    cls = np.where(deg > 0, tops[np.minimum(np.searchsorted(tops, deg), len(tops) - 1)], 0)
    buckets = []
    for w in np.unique(cls[cls > 0]):
        w = int(w)
        rows = np.flatnonzero(cls == w).astype(np.int32)
        n = len(rows)
        per = min(max(1, CHUNK // w), n)
        total = -(-n // per) * per
        r = np.full(total, rows[0], np.int32)
        r[:n] = rows
        valid = np.zeros((total, w), bool)
        valid[:n] = np.arange(w)[None, :] < deg[rows][:, None]
        idx = np.zeros((total, w), np.int64)
        idx[:n] = indptr[rows][:, None] + np.arange(w)[None, :]
        sb = np.where(valid, src[np.where(valid, idx, 0)], 0).astype(np.int32)
        gb = np.where(valid, grp[np.where(valid, idx, 0)], 0).astype(np.int32)
        del idx
        tt = np.repeat(r, w)
        ss = np.where(valid.reshape(-1), sb.reshape(-1), tt)  # invalid slot: s == t, zero factor
        pad = -len(tt) % CHUNK
        flat = [
            np.concatenate([x, np.zeros(pad, np.int32)]).reshape(-1, CHUNK)
            for x in (tt, ss, gb.reshape(-1))
        ]
        del tt, ss, valid, gb
        fac = jnp.concatenate(
            [
                chunk_factors(*(jnp.asarray(x[k]) for x in flat), z, c, states, kmax)
                for k in range(len(flat[0]))
            ]
        )[: total * w].reshape(-1, per, w)
        del flat
        buckets += [jnp.asarray(r.reshape(-1, per)), jnp.asarray(sb.reshape(-1, per, w)), fac]
    return buckets


def _bucket_scan(vec, out, rows, src, fac):
    """Per piece: each row's sum ``fac . vec[src]`` along its own entries, written once per row."""

    def body(acc, chunk):
        r, sj, dj = chunk
        val = jnp.sum(dj * vec.at[..., sj].get(mode=IB), axis=-1)
        return acc.at[..., r].add(val, mode=IB), None

    return jax.lax.scan(body, out, (rows, src, fac))[0]


def _ell_block(vec, out, src, fac):
    """Per piece of ``R`` consecutive rows: row sums ``fac . vec[src]``, added into ``out`` in place."""
    rows = src.shape[1]

    def body(acc, chunk):
        p, sj, dj = chunk
        val = jnp.sum(dj * vec[..., sj], axis=-1)
        win = jax.lax.dynamic_slice_in_dim(acc, p * rows, rows, axis=-1)
        return jax.lax.dynamic_update_slice_in_dim(acc, win + val, p * rows, axis=-1), None

    return jax.lax.scan(body, out, (jnp.arange(src.shape[0]), src, fac))[0]


@jax.jit
def matvec_hyb(vec, d0, *parts):
    """HYB: per set, a natural-order ELL block (contiguous writes) plus a sorted-scatter overflow."""
    out = d0 * vec
    for k in range(0, len(parts), 5):
        es, ef, ovt, os_, od = parts[k : k + 5]
        out = _sorted_scatter(vec, _ell_block(vec, out, es, ef), ovt, os_, od)
    return out


@jax.jit
def matvec_ell(vec, d0, *buckets):
    """Degree-bucketed ELL: scatter-free row sums, then one write per row."""
    out = d0 * vec
    for k in range(0, len(buckets), 3):
        out = _bucket_scan(vec, out, *buckets[k : k + 3])
    return out


# C2W: rows one window may span, so its segment_sum emits O(window) segments, never N. 2^15 is the
# best of 2^11..2^15 measured, and still slower than C2R.
ROWS = int(os.environ.get("C2W_ROWS", str(1 << 15)))


def row_windows(t, s, d):
    """Target-sorted entries re-chunked on row boundaries: ``(base, local, s, d)``, ``(chunks, CHUNK)``.

    Each window holds whole rows, at most ``CHUNK`` entries and ``ROWS`` consecutive rows from ``base``,
    so ``local = t - base < ROWS``. Padding has ``d = 0``. Built on the host from C2R's chunks.
    """
    keep = t != s  # padding entries have equal endpoints; genuine ones never do
    t, s, d = t[keep], s[keep], d[keep]
    cuts, e0 = [], 0
    while e0 < len(t):
        e1 = min(e0 + CHUNK, int(np.searchsorted(t, t[e0] + ROWS)))
        if e1 < len(t) and t[e1 - 1] == t[e1]:  # never split a row across windows
            e1 = int(np.searchsorted(t, t[e1]))
        cuts.append((e0, e1))
        e0 = e1
    n = max(len(cuts), 1)
    base = np.zeros(n, np.int32)
    local, src = np.zeros((n, CHUNK), np.int32), np.zeros((n, CHUNK), np.int32)
    fac = np.zeros((n, CHUNK), d.dtype)
    for k, (a, b) in enumerate(cuts):
        base[k] = t[a]
        local[k, : b - a], src[k, : b - a], fac[k, : b - a] = t[a:b] - t[a], s[a:b], d[a:b]
    return tuple(jnp.asarray(x) for x in (base, local, src, fac))


def _window_scan(vec, out, base, local, src, fac):
    """Per window: ``segment_sum`` of ``fac * vec[src]`` over ``ROWS`` local rows, added at ``base``."""

    def body(acc, chunk):
        b, lt, sj, dj = chunk
        msg = jnp.moveaxis(dj * vec[..., sj], -1, 0)  # segment axis first
        seg = jnp.moveaxis(
            jax.ops.segment_sum(msg, lt, num_segments=ROWS, indices_are_sorted=True), 0, -1
        )
        win = jax.lax.dynamic_slice_in_dim(acc, b, ROWS, axis=-1)
        return jax.lax.dynamic_update_slice_in_dim(acc, win + seg, b, axis=-1), None

    return jax.lax.scan(body, out, (base, local, src, fac))[0]


@jax.jit
def matvec_c2w(vec, rb, rl, rs, rd, qb, ql, qs, qd, d0):
    """C2R as row-window reductions instead of per-entry scatters; ``out`` is padded by ``ROWS``."""
    pad = [(0, 0)] * (vec.ndim - 1) + [(0, ROWS)]
    out = jnp.pad(d0 * vec, pad)
    out = _window_scan(vec, _window_scan(vec, out, rb, rl, rs, rd), qb, ql, qs, qd)
    return out[..., : vec.shape[-1]]


# (N, k) layout: row-indexed, so one random access fetches every vector's entry from one cache line.


@functools.partial(jax.jit, static_argnames="kmax")
def matvec_p0t(vec, pi, pj, pg, z, c, states, kmax):
    def body(out, chunk):
        i, j, g = chunk
        d = jnp.where(i == j, 0.0, pair_diagonal(states[i], z[g][:, :kmax], c[g][:, :kmax]))[
            :, None
        ]
        out = out.at[i].add(d * vec[j])
        return out.at[j].add(jnp.conj(d) * vec[i]), None

    return jax.lax.scan(body, get_diagonal(z[0], c[0], states)[:, None] * vec, (pi, pj, pg))[0]


@jax.jit
def matvec_p2t(vec, pi, pj, d, d0):
    def body(out, chunk):
        i, j, di = chunk
        out = out.at[i].add(di[:, None] * vec[j])
        return out.at[j].add(jnp.conj(di)[:, None] * vec[i]), None

    return jax.lax.scan(body, d0[:, None] * vec, (pi, pj, d))[0]


def _sorted_scatter_t(vec, out, t, s, d):
    def body(acc, chunk):
        ti, si, di = chunk
        return acc.at[ti].add(di[:, None] * vec[si], indices_are_sorted=True), None

    return jax.lax.scan(body, out, (t, s, d))[0]


@jax.jit
def matvec_c2rt(vec, rt, rs, rd, qt, qs, qd, d0):
    return _sorted_scatter_t(vec, _sorted_scatter_t(vec, d0[:, None] * vec, rt, rs, rd), qt, qs, qd)


TRANSPOSED = {"P0": None, "P2": matvec_p2t, "C2R": matvec_c2rt}  # P0's needs its kmax, bound below


# ------------------------------------------------------------------------------------ operator assembly


def rcm_permutation(t, s, dim, size):
    """Reverse Cuthill--McKee order of the genuine states; filler rows keep their places."""
    graph = sp.csr_matrix((np.ones(len(t), np.int8), (t, s)), shape=(dim, dim))
    order = reverse_cuthill_mckee(graph, symmetric_mode=True)
    return np.concatenate([order, np.arange(dim, size)]).astype(np.int64)


def relabel(t, s, g, d, perm):
    """Entries under ``new = inverse(perm)[old]``, re-sorted by the new target."""
    inv = np.empty_like(perm)
    inv[perm] = np.arange(len(perm))
    nt, ns = inv[t].astype(np.int32), inv[s].astype(np.int32)
    order = np.argsort(nt, kind="stable")
    return nt[order], ns[order], g[order], d[order]


def term_masks(op):
    """``(x, z, coef)`` per Pauli term of a ``SparsePauliOp``, as ``uint64`` masks (qubit q = bit q).

    ``coef`` folds in ``(-i)^{|x&z|}``, so ``<s^x|P|s> = coef * (-1)^{|z & (s^x)|}``.
    """
    w = np.uint64(1) << np.arange(op.num_qubits, dtype=np.uint64)
    x = (op.paulis.x.astype(np.uint64) * w).sum(axis=1).astype(np.uint64)
    z = (op.paulis.z.astype(np.uint64) * w).sum(axis=1).astype(np.uint64)
    phase = (-1j) ** ((op.paulis.x & op.paulis.z).sum(axis=1) + op.paulis.phase)
    return x, z, op.coeffs * phase


def to_codes(states):
    return (states.astype(np.uint64) << np.arange(states.shape[1], dtype=np.uint64)).sum(axis=1)


def recovery_scores(masks, codes, v, signed=False):
    """``(c, |<c|H|v>|)`` for every ``c`` in H's one-hop reach of sorted ``codes`` and outside it.

    ``signed=True`` returns ``<c|H|v>`` itself, for a first-order start on the admitted states.
    """
    x, z, coef = masks
    cand, amp = [], []
    for xg in np.unique(x[x != 0]):
        c = codes ^ xg
        pos = np.searchsorted(codes, c).clip(max=len(codes) - 1)
        out = codes[pos] != c
        c, vs = c[out], v[out]
        a = np.zeros(len(c), complex)
        for zk, ck in zip(z[x == xg], coef[x == xg]):
            a += (
                ck * (1 - 2 * (np.bitwise_count(c & zk) & 1).astype(np.int8)) * vs
            )  # uint8: 1-2 wraps
        cand.append(c)
        amp.append(a)
    uniq, inv = np.unique(np.concatenate(cand), return_inverse=True)
    score = np.zeros(len(uniq), complex)
    np.add.at(score, inv, np.concatenate(amp))
    return uniq, score if signed else np.abs(score)


def recovery_subspace(op, n, size, seed_size=1 << 12):
    """``size`` states grown like spinchain's recovery: Hamming shells, then ranked H-expansion rounds.

    Each round solves ``sqd`` on the current subspace and admits the top ``|<c|H|v>|`` of its one-hop
    reach, doubling it (the last round fills to ``size``). No pruning: spinchain also drops rows the
    eigenvector drives below ``weight_tol``, which would free budget for more of the same coupled rows.
    """
    ham = PauliSumXZ.from_paulisum(op)
    masks = term_masks(op)
    codes = np.sort(to_codes(hamming_shells(n, min(seed_size, size), np.random.default_rng(0))))
    while len(codes) < size:
        states = ((codes[:, None] >> np.arange(n, dtype=np.uint64)) & np.uint64(1)).astype(np.uint8)
        _, vec, basis = sqd(ham, states, rtol=1e-8)
        order = np.argsort(to_codes(np.asarray(basis)))
        v = np.asarray(vec)[: len(codes)][order]
        cand, score = recovery_scores(masks, codes, v)
        take = min(len(codes), size - len(codes), len(cand))
        assert take > 0, "the H-reach is exhausted before reaching size"
        codes = np.sort(np.concatenate([codes, cand[np.argsort(-score, kind="stable")[:take]]]))
    return ((codes[:, None] >> np.arange(n, dtype=np.uint64)) & np.uint64(1)).astype(np.uint8)


def spinchain_problem(n, pattern, delta, size, subspace="shells"):
    op = xxz(n, delta, *patterns(n)[pattern])
    if subspace == "recovery":  # cached: growing to 2^21 solves sqd up to 2^20
        path = f"/tmp/sparse_pairs_recovery_n{n}_{pattern}_d{delta}_{size}.npy"
        if not os.path.exists(path):
            np.save(path, recovery_subspace(op, n, size))
        return PauliSumXZ.from_paulisum(op), np.load(path)
    return PauliSumXZ.from_paulisum(op), hamming_shells(n, size, np.random.default_rng(0))


def operators(ham, states, size, arms):
    """``{arm: (kernel, args, perm or None)}`` for ``vec`` supplied first; ``perm`` marks a relabelled arm.

    Built once per size so every subcommand times identical operators. Also returns the diagnostics a
    report needs: ``(ops, info)``.
    """
    dim = len(states)
    su = uniquify_states(_pad_states(PauliSumXZ.pack_states(states), size), size)
    z, c = jnp.asarray(ham.z), jnp.asarray(ham.c)
    kmax = int((np.abs(np.asarray(ham.c[1:])) > 0).sum(axis=1).max())
    real_group = ~(np.abs(np.asarray(ham.c).imag) > 0).any(axis=1)
    d0 = get_diagonal(ham.z[0], ham.c[0], su)
    ops, info = {}, {"kmax": kmax, "su": su, "d0": d0}
    wanted = set(arms)
    rcm_arms = {"C0i16+RCM", "C2+RCM", "C2R+RCM"} & wanted
    num_groups = ham.x.shape[0]
    small = np.int16 if num_groups - 1 <= np.iinfo(np.int16).max else np.int32  # C0i16's id width
    # Build only what the requested arms use, and drop host copies once chunked: on CPU they share the
    # device's memory, so a stray reference is exactly what `peak` would misreport as operator cost.
    t = s = g = dflat = None
    if wanted - {"(1,0)", "(1,2)"}:
        pi, pj, pg, info["hit"] = build_pairs(ham, su, dim)
        info["pairs_per_slot"] = len(pi) / size
        if "P2R" in wanted:
            (ri, rj, rg), (qi, qj, qg) = real_split(pi, pj, pg, real_group)
            rd = chunked_diagonals(ri, rj, rg, z, c.real, su, kmax)
            qd = chunked_diagonals(qi, qj, qg, z, c, su, kmax)
            del rg, qg
            ops["P2R"] = (matvec_p2r, (ri, rj, rd, qi, qj, qd, d0), None)
        if {"P0", "P2"} & wanted:
            cpi, cpj, cpg = chunked((pi, pj, pg), pad=0)
            if "P0" in wanted:
                ops["P0"] = (
                    functools.partial(matvec_p0, kmax=kmax),
                    (cpi, cpj, cpg, z, c, su),
                    None,
                )
            if "P2" in wanted:
                cdp = chunked_diagonals(cpi, cpj, cpg, z, c, su, kmax)
                ops["P2"] = (matvec_p2, (cpi, cpj, cdp, d0), None)
            del cpi, cpj, cpg
        needs_host = bool(
            (
                {"C0", "C0i16", "C2", "C2R", "C2W", "C2RF", "ELL", "HYB", "ELLC", "ELLD", "seg"}
                | rcm_arms
            )
            & wanted
        )
        if (
            not needs_host
        ):  # nothing else reads the host pairs: free them before anything else is built
            del pi, pj, pg
        every = range(1, num_groups)

        def csr(groups, dtype):
            return csr_chunked(pi, pj, pg, groups, num_groups, size, dtype)

        k0 = functools.partial(matvec_c0, kmax=kmax)
        for arm, dtype in (("C0", np.int32), ("C0i16", small)):
            if arm in wanted:
                # astype would wrap silently past the dtype's range, and a wrapped id indexes another
                # group's Z table; `small` already falls back to int32, so this guards explicit widths.
                assert num_groups - 1 <= np.iinfo(dtype).max, (
                    f"J={num_groups} overflows {dtype.__name__}"
                )
                ops[arm] = (k0, (*csr(every, dtype), z, c, su), None)
        if {"C2", "seg"} & wanted:
            ct, cs, cg = csr(every, np.int32)
            cd = chunked_diagonals(ct, cs, cg, z, c, su, kmax)
            del cg
            if "C2" in wanted:
                ops["C2"] = (matvec_c2, (ct, cs, cd, d0), None)
            if "seg" in wanted:  # padding entries are (size-1, size-1) with d = 0: sorted and inert
                ops["seg"] = (
                    matvec_seg,
                    (ct.reshape(-1), cs.reshape(-1), cd.reshape(-1), d0),
                    None,
                )
        if "C2R" in wanted:
            parts = []
            for groups, real in (
                ([g for g in every if real_group[g]], True),
                ([g for g in every if not real_group[g]], False),
            ):
                qt, qs, qg = csr(groups, np.int32)
                # Real groups: exact real coefficients give float64 diagonals directly, with no
                # complex intermediate alive beside its `.real` copy.
                qd = chunked_diagonals(qt, qs, qg, z, c.real if real else c, su, kmax)
                del qg
                parts += [qt, qs, qd]
            ops["C2R"] = (matvec_c2r, (*parts, d0), None)
        if "P2F" in wanted:
            fi, fj, fg = chunked((pi, pj, pg), pad=0)
            ops["P2F"] = (
                matvec_p2f,
                (fi, fj, chunked_diagonals(fi, fj, fg, z, c, su, kmax), d0),
                None,
            )
            del fg
        if "C2RF" in wanted:
            parts = []
            for groups, real in (
                ([g for g in every if real_group[g]], True),
                ([g for g in every if not real_group[g]], False),
            ):
                qt, qs, qg = csr(groups, np.int32)
                parts += [qt, qs, chunked_diagonals(qt, qs, qg, z, c.real if real else c, su, kmax)]
                del qg
            ops["C2RF"] = (matvec_c2rf, (*parts, d0), None)
            del parts
        if "ELL" in wanted:
            buckets = []
            for groups, real in (
                ([g for g in every if real_group[g]], True),
                ([g for g in every if not real_group[g]], False),
            ):
                c_set = c.real if real else c
                buckets += ell_buckets(pi, pj, pg, groups, num_groups, size, z, c_set, su, kmax)
            ops["ELL"] = (matvec_ell, (d0, *buckets), None)
            del buckets
        if "ELLC" in wanted:
            buckets = []
            for groups, real in (
                ([g for g in every if real_group[g]], True),
                ([g for g in every if not real_group[g]], False),
            ):
                c_set = c.real if real else c
                buckets += ell_class_buckets(
                    pi,
                    pj,
                    pg,
                    groups,
                    num_groups,
                    size,
                    z,
                    c_set,
                    su,
                    kmax,
                    lambda deg, slot: geometric_classes(deg),
                )
            ops["ELLC"] = (matvec_ell, (d0, *buckets), None)
            del buckets
        if "ELLD" in wanted:
            bucket_bytes = float(os.environ.get("ELLD_BUCKET_MIB", "5.7")) * 2**20
            buckets = []
            for groups, real in (
                ([g for g in every if real_group[g]], True),
                ([g for g in every if not real_group[g]], False),
            ):
                c_set = c.real if real else c
                buckets += ell_class_buckets(
                    pi,
                    pj,
                    pg,
                    groups,
                    num_groups,
                    size,
                    z,
                    c_set,
                    su,
                    kmax,
                    lambda deg, slot: optimal_classes(deg, slot, bucket_bytes),
                )
            ops["ELLD"] = (matvec_ell, (d0, *buckets), None)
            del buckets
        if "HYB" in wanted:
            parts = []
            for groups, real in (
                ([g for g in every if real_group[g]], True),
                ([g for g in every if not real_group[g]], False),
            ):
                c_set = c.real if real else c
                parts += hyb_set(pi, pj, pg, groups, num_groups, size, z, c_set, su, kmax)
            ops["HYB"] = (matvec_hyb, (d0, *parts), None)
            del parts
        if "C2W" in wanted:
            windows = []
            for groups, real in (
                ([g for g in every if real_group[g]], True),
                ([g for g in every if not real_group[g]], False),
            ):
                qt, qs, qg = csr(groups, np.int32)
                qd = chunked_diagonals(qt, qs, qg, z, c.real if real else c, su, kmax)
                del qg
                flat = (np.asarray(x).reshape(-1) for x in (qt, qs, qd))
                del qt, qs, qd
                windows += row_windows(*flat)
            ops["C2W"] = (matvec_c2w, (*windows, d0), None)
        if rcm_arms:
            t, s, g = directed_by_target(pi, pj, pg)
            ct, cs, cg = chunked((t, s, g), pad=size - 1)
            dflat = np.asarray(chunked_diagonals(ct, cs, cg, z, c, su, kmax)).reshape(-1)[: len(t)]
            del ct, cs, cg
        if needs_host:
            del pi, pj, pg

    def c0(tt, ss, gg, states_, dtype):
        # astype wraps silently past the dtype's range, and a wrapped id indexes another group's Z table.
        assert ham.x.shape[0] - 1 <= np.iinfo(dtype).max, (
            f"J={ham.x.shape[0]} overflows {dtype.__name__}"
        )
        a, b, e = chunked((tt, ss, gg.astype(dtype)), pad=size - 1)
        return (a, b, e, z, c, states_)

    def c2(tt, ss, dd, d_0):
        a, b = chunked((tt, ss), pad=size - 1)
        return (a, b, chunked((dd,), pad=0)[0], d_0)

    def c2r(tt, ss, gg, dd, d_0):
        r = real_group[gg]
        rt, rs = chunked((tt[r], ss[r]), pad=size - 1)
        qt, qs = chunked((tt[~r], ss[~r]), pad=size - 1)
        rd = chunked((dd[r].real.astype(np.float64),), pad=0)[0]
        return (rt, rs, rd, qt, qs, chunked((dd[~r],), pad=0)[0], d_0)

    if {"(1,0)", "(1,2)"} & set(arms):
        xs = jax.lax.scan(lambda _, x: (None, get_xsource(x, su)), None, ham.x)[1]
        if "(1,0)" in arms:
            k10 = functools.partial(_apply_h_kernel, matvec=Matvec.INDICES)
            ops["(1,0)"] = (
                jax.jit(k10),
                (_pack_scanned(Matvec.INDICES, xs, ham.z, ham.c), su),
                None,
            )
        if "(1,2)" in arms:
            dg = jax.lax.scan(
                lambda _, v: (None, get_diagonal(v[0], v[1], su)), None, (ham.z, ham.c)
            )[1]
            k12 = functools.partial(_apply_h_kernel, matvec=Matvec.TABLES)
            ops["(1,2)"] = (jax.jit(k12), (_pack_scanned(Matvec.TABLES, xs, dg, ham.c), None), None)
    k0 = functools.partial(matvec_c0, kmax=kmax)
    if rcm_arms:
        t0 = time.perf_counter()
        perm = rcm_permutation(t, s, dim, size)
        info["rcm_s"] = time.perf_counter() - t0
        pt, ps, pgr, pdv = relabel(t, s, g, dflat, perm)
        psu, pd0 = su[perm], d0[perm]
        if "C0i16+RCM" in arms:
            ops["C0i16+RCM"] = (k0, c0(pt, ps, pgr, psu, small), perm)
        if "C2+RCM" in arms:
            ops["C2+RCM"] = (matvec_c2, c2(pt, ps, pdv, pd0), perm)
        if "C2R+RCM" in arms:
            ops["C2R+RCM"] = (matvec_c2r, c2r(pt, ps, pgr, pdv, pd0), perm)
    return ops, info


def operator_bytes(args):
    return sum(x.nbytes for x in jax.tree.leaves(args) if hasattr(x, "nbytes"))


def check(ops, vec, dim):
    """Every arm against ``(1,0)``'s product, under its relabelling where it has one."""
    ref = None
    for name, (fn, a, perm) in ops.items():
        if name == "(1,0)":
            ref = np.asarray(fn(vec, *a))
    assert ref is not None, "the check needs the (1,0) arm"
    for name, (fn, a, perm) in ops.items():
        got = np.asarray(fn(vec if perm is None else vec[:, perm], *a))
        want = ref if perm is None else ref[:, perm]
        keep = (np.arange(len(ref[0])) if perm is None else perm) < dim
        assert np.allclose(got[:, keep], want[:, keep], rtol=1e-11, atol=1e-11), name


def run_sqd_vinit(ham, states_u, size, d0):
    """``run_sqd``'s initial vector: the spread seed plus a signed weight on the min-diagonal state."""
    seed = _spread_seed(size, states_u, ham.c.dtype, None)
    diagonal = jnp.where(_is_filler(states_u) == 1, jnp.max(d0.real), d0.real)
    sign = jnp.sign(seed)
    selected = jnp.arange(size) == jnp.argmin(diagonal)
    return seed + jnp.where(selected, jnp.where(sign == 0, 1.0, sign), 0.0)


def solve_fn(fn, bound, batch=True):
    """A ``ground_locg`` solve as ``run_sqd`` drives it: prefilter ``(32, 2)``, ``Σ|c|`` bound, batched."""

    def solve(vinit, *args):
        return ground_locg(
            fn, vinit, args=args, prefilter=(32, 2), prefilter_hi=bound, batch_matvec=batch
        )

    return jax.jit(solve)


# ------------------------------------------------------------------------------------------ subcommands


def timed(fn, args, rounds):
    ts = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def cmd_matvec(args) -> None:
    arms = args.arms or [a for a in ARMS if a != "(1,2)"]
    if "(1,0)" not in arms:
        arms = ["(1,0)", *arms]
    print(
        f"n={args.num_qubits} {args.pattern} {args.subspace} delta={args.delta}, chunk {CHUNK}, "
        f"XLA_FLAGS={os.environ.get('XLA_FLAGS', 'default')}: ns/state, then operator B/slot"
    )
    print(f"{'size':>6} " + " ".join(f"{a:>10}" for a in arms))
    for log2 in args.log2_sizes:
        size = 1 << log2
        ham, states = spinchain_problem(
            args.num_qubits, args.pattern, args.delta, size, args.subspace
        )
        ops, info = operators(ham, states, size, arms)
        rng = np.random.default_rng(1)
        vec = jnp.asarray(rng.normal(size=(2, size)) * (1 + 0.5j)).at[:, len(states) :].set(0)
        check(ops, vec, len(states))
        inputs = {a: (vec if p is None else vec[:, p], *x) for a, (_, x, p) in ops.items()}
        times = {a: [] for a in arms}
        for _ in range(args.rounds):  # interleaved, so drift hits every arm alike
            for a in arms:
                times[a].append(timed(ops[a][0], inputs[a], 1))
        ns = {a: 1e9 * np.median(v) / size for a, v in times.items()}
        print(f"  2^{log2} " + " ".join(f"{ns[a]:>10.1f}" for a in arms), flush=True)
        print(
            f"{'B/slot':>8} "
            + " ".join(f"{operator_bytes(ops[a][1]) / size:>10.0f}" for a in arms)
            + f"   hit {info['hit']:.3f}"
            + (f", RCM {info['rcm_s']:.1f} s" if "rcm_s" in info else ""),
            flush=True,
        )


def cmd_solve(args) -> None:
    arms = args.arms or ["(1,0)", "(1,2)", "P0", "P2", "C0i16", "C2R"]
    size = 1 << args.log2_size
    ham, states = spinchain_problem(args.num_qubits, args.pattern, args.delta, size, args.subspace)
    ops, info = operators(ham, states, size, set(arms) | {"(1,0)"})
    vinit = run_sqd_vinit(ham, info["su"], size, info["d0"])
    bound = float(np.abs(np.asarray(ham.c)).sum())
    print(
        f"n={args.num_qubits} {args.pattern} {args.subspace} delta={args.delta}, states_size 2^{args.log2_size}, "
        f"hit {info['hit']:.3f}; whole solves, memory = XLA inputs + temp"
    )
    print(f"{'arm':>7} {'solve s':>8} {'iters':>5} {'eigval':>20} {'B/slot':>7}")
    for a in arms:
        fn, x, _ = ops[a]
        solve = solve_fn(fn, bound)
        mem = solve.lower(vinit, *x).compile().memory_analysis()
        result = jax.block_until_ready(solve(vinit, *x))
        seconds = timed(solve, (vinit, *x), 3)
        total = (mem.argument_size_in_bytes + mem.temp_size_in_bytes) / size
        print(
            f"{a:>7} {seconds:>8.2f} {int(result[2]):>5} {float(result[0]):>20.12f} {total:>7.0f}",
            flush=True,
        )


def cmd_batch(args) -> None:
    """One ``(2, N)`` application against two ``(N,)`` and an ``(N, 2)`` layout, then whole solves."""
    arms = args.arms or ["(1,0)", "(1,2)", "P0", "P2", "C0i16", "C2R"]
    print(
        f"n={args.num_qubits} {args.pattern} {args.subspace} delta={args.delta}: ns/state for two "
        "vectors; 'x' columns are speedups over the batched (2, N) call"
    )
    print(
        f"{'size':>6} {'arm':>7} {'(2,N)':>8} {'2x(N,)':>8} {'unbat x':>8} {'(N,2)':>8} {'(N,2) x':>8}"
    )
    for log2 in args.log2_sizes:
        size = 1 << log2
        ham, states = spinchain_problem(
            args.num_qubits, args.pattern, args.delta, size, args.subspace
        )
        ops, info = operators(ham, states, size, set(arms) | {"(1,0)"})
        rng = np.random.default_rng(1)
        vec = jnp.asarray(rng.normal(size=(2, size)) * (1 + 0.5j)).at[:, len(states) :].set(0)
        vt = jnp.asarray(np.ascontiguousarray(np.asarray(vec).T))
        check(ops, vec, len(states))
        runs = {}
        for a in arms:
            fn, x, _ = ops[a]
            pair = jax.jit(lambda v0, v1, *xs, fn=fn: (fn(v0, *xs), fn(v1, *xs)))
            want = np.asarray(fn(vec, *x))
            assert np.allclose(np.stack(pair(vec[0], vec[1], *x)), want, rtol=1e-12, atol=1e-12), a
            runs[a] = {"b": (fn, (vec, *x)), "u": (pair, (vec[0], vec[1], *x))}
            if a in TRANSPOSED:
                ft = TRANSPOSED[a] or functools.partial(matvec_p0t, kmax=info["kmax"])
                assert np.allclose(np.asarray(ft(vt, *x)).T, want, rtol=1e-12, atol=1e-12), a
                runs[a]["t"] = (ft, (vt, *x))
        times = {(a, k): [] for a in arms for k in runs[a]}
        for _ in range(args.rounds):  # interleaved over arms and layouts
            for a in arms:
                for k, (f, fa) in runs[a].items():
                    times[a, k].append(timed(f, fa, 1))
        for a in arms:
            ns = {k: 1e9 * np.median(times[a, k]) / size for k in runs[a]}
            t_cols = (
                f"{ns['t']:>8.1f} {ns['b'] / ns['t']:>7.2f}x" if "t" in ns else f"{'':>8} {'':>8}"
            )
            print(
                f"  2^{log2} {a:>7} {ns['b']:>8.1f} {ns['u']:>8.1f} {ns['u'] / ns['b']:>7.2f}x {t_cols}"
            )
        print(f"{'':>8} hit {info['hit']:.3f}", flush=True)
    if not args.solve_log2:
        return
    size = 1 << args.solve_log2
    ham, states = spinchain_problem(args.num_qubits, args.pattern, args.delta, size, args.subspace)
    ops, info = operators(ham, states, size, set(arms) | {"(1,0)"})
    vinit = run_sqd_vinit(ham, info["su"], size, info["d0"])
    bound = float(np.abs(np.asarray(ham.c)).sum())
    print(f"whole solves at 2^{args.solve_log2}: batched vs unbatched ground_locg")
    print(
        f"{'arm':>7} {'batched s':>10} {'unbat s':>8} {'unbat x':>8} {'iters':>9} {'eigval diff':>12}"
    )
    for a in arms:
        fn, x, _ = ops[a]
        solves = {b: solve_fn(fn, bound, b) for b in (True, False)}
        res = {b: jax.block_until_ready(f(vinit, *x)) for b, f in solves.items()}
        secs = {True: [], False: []}
        for _ in range(3):
            for b, f in solves.items():
                secs[b].append(timed(f, (vinit, *x), 1))
        med = {b: float(np.median(v)) for b, v in secs.items()}
        print(
            f"{a:>7} {med[True]:>10.2f} {med[False]:>8.2f} {med[False] / med[True]:>7.2f}x "
            f"{int(res[True][2]):>4}/{int(res[False][2]):<4} "
            f"{float(res[False][0]) - float(res[True][0]):>12.1e}",
            flush=True,
        )


def peak_child(args) -> dict:
    """One arm, in its own process: peak RSS above a post-startup baseline (CPU: device memory = RSS)."""
    peak = lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    ham = PauliSumXZ.from_paulisum(
        xxz(args.num_qubits, args.delta, *patterns(args.num_qubits)[args.pattern])
    )
    states = np.load(args.states)
    size = 1 << (len(states) - 1).bit_length()
    states_p = _pad_states(PauliSumXZ.pack_states(states), size)
    jnp.zeros(1).block_until_ready()
    baseline = peak()
    t0 = time.perf_counter()
    if args.child == "(1,0)":  # production: run_sqd precomputes its own (J, N) source array
        solve, x, setup = (lambda: run_sqd(ham, states_p, size, False, Matvec.INDICES)), (), 0.0
    else:
        ops, info = operators(ham, states, size, [args.child])
        fn, xargs, _ = ops[args.child]
        vinit = run_sqd_vinit(ham, info["su"], size, info["d0"])
        f = solve_fn(fn, float(np.abs(np.asarray(ham.c)).sum()))
        solve, x, setup = (lambda: f(vinit, *xargs)), (), time.perf_counter() - t0
    jax.block_until_ready(solve())
    warm = timed(lambda: solve(), x, 1)
    result = solve()
    eigval = float(result.eigval if args.child == "(1,0)" else result[0])
    return {
        "arm": args.child,
        "size": size,
        "peak_mib": (peak() - baseline) / 2**20,
        "setup_s": setup,
        "warm_s": warm,
        "eigval": eigval,
    }


def cmd_peak(args) -> None:
    arms = args.arms or ["(1,0)", "P0", "P2", "C0i16", "C2R"]
    print(
        f"n={args.num_qubits} {args.pattern} {args.subspace} delta={args.delta}: peak RSS above baseline, fresh process per arm"
    )
    print(
        f"{'size':>9} {'arm':>7} {'peak MiB':>9} {'B/slot':>7} {'setup s':>8} {'warm s':>8} {'eigval':>18}"
    )
    workdir = args.workdir
    os.makedirs(workdir, exist_ok=True)
    for log2 in args.log2_sizes:
        size = 1 << log2
        path = os.path.join(workdir, f"sparse_pairs_states_{log2}.npy")
        np.save(
            path,
            spinchain_problem(args.num_qubits, args.pattern, args.delta, size, args.subspace)[1],
        )
        for arm in arms:
            cmd = [
                sys.executable,
                __file__,
                "peak",
                "--child",
                arm,
                "--states",
                path,
                "--num-qubits",
                str(args.num_qubits),
                "--pattern",
                args.pattern,
                "--delta",
                str(args.delta),
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if proc.returncode:
                print(f"{size:>9} {arm:>7} FAILED\n{proc.stderr[-1500:]}", flush=True)
                continue
            r = json.loads(proc.stdout.strip().splitlines()[-1])
            print(
                f"{size:>9} {arm:>7} {r['peak_mib']:>9.1f} {r['peak_mib'] * 2**20 / size:>7.0f} "
                f"{r['setup_s']:>8.1f} {r['warm_s']:>8.2f} {r['eigval']:>18.12f}",
                flush=True,
            )
        os.remove(path)


def cmd_general(args) -> None:
    from diag_cache_order import molecular_like

    saved, sys.argv = sys.argv, sys.argv[:1]  # sqd_multinode parses the command line at import
    try:
        from sqd_multinode import xxz_hamiltonian, xxz_krylov_states
    finally:
        sys.argv = saved
    rng = np.random.default_rng(0)
    fixtures = [
        (
            "molecular-like JW n=14, random states",
            PauliSumXZ.from_paulisum(molecular_like(14, rng, 0.2)),
            np.unique(rng.integers(0, 2, size=(20000, 14), dtype=np.uint8), axis=0),
        ),
        (
            "periodic XXZ n=24, Neel-Krylov (closed under hops)",
            xxz_hamiltonian(24, 1.0),
            xxz_krylov_states(24, 60000),
        ),
    ]
    arms = ["(1,0)", "P0", "P2", "C2R"]
    for label, ham, states in fixtures:
        size = 1 << (len(states) - 1).bit_length()
        ops, info = operators(ham, states, size, arms)
        vec = jnp.asarray(np.random.default_rng(1).normal(size=(2, size))).astype(ham.c.dtype)
        vec = vec.at[:, len(states) :].set(0)
        check(ops, vec, len(states))
        times = {a: [] for a in arms}
        for _ in range(args.rounds):  # interleaved, so the baseline shares every arm's drift
            for a in arms:
                times[a].append(timed(ops[a][0], (vec, *ops[a][1]), 1))
        base = np.median(times["(1,0)"])
        cells = [
            f"{a} {operator_bytes(ops[a][1]) / size:.0f} B/slot, {base / np.median(times[a]):.2f}x"
            for a in arms
            if a != "(1,0)"
        ]
        print(
            f"{label}: J={ham.x.shape[0]}, N={len(states)}, hit {info['hit']:.3f} | "
            + " | ".join(cells)
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("matvec", "solve", "peak", "general", "batch"):
        p = sub.add_parser(name)
        p.add_argument("--num-qubits", type=int, default=60)
        p.add_argument("--pattern", default="type2")
        p.add_argument("--delta", type=float, default=0.5)
        p.add_argument("--subspace", choices=("shells", "recovery"), default="shells")
        p.add_argument("--arms", nargs="+", choices=ARMS)
        p.add_argument("--rounds", type=int, default=7)
        if name in ("matvec", "peak", "batch"):
            p.add_argument(
                "--log2-sizes",
                type=int,
                nargs="+",
                default=[15, 17, 19, 21] if name == "matvec" else [17, 19, 21],
            )
        if name == "solve":
            p.add_argument("--log2-size", type=int, default=17)
        if name == "batch":
            p.add_argument("--solve-log2", type=int, default=17, help="0 skips the whole solves")
        if name == "peak":
            p.add_argument("--workdir", default="/tmp")
            p.add_argument("--child", choices=ARMS, help=argparse.SUPPRESS)
            p.add_argument("--states", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.cmd == "peak" and args.child:
        print(json.dumps(peak_child(args)))
        return
    {
        "matvec": cmd_matvec,
        "solve": cmd_solve,
        "peak": cmd_peak,
        "general": cmd_general,
        "batch": cmd_batch,
    }[args.cmd](args)


if __name__ == "__main__":
    main()
