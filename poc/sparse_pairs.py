"""Sparse transition pairs instead of dense source indices -- every prototype and measurement in one script.

Item 8 of ``markdown/improvement-ideas-2026-09-25.md``; the write-up with every result is
``poc/sparse-pairs.md``. At ``cache_level=(1, *)`` ``sqd`` stores one ``int32`` source per
``(X group, state)`` (``4*J`` B/slot) though on sampled subspaces most are ``-1``. XOR is an involution, so
each real transition of a group ``g != 0`` is a pair; storing it once, and computing its diagonal once since
``H_ji = conj(H_ij)`` exactly for one X signature, removes the placeholders and half the diagonal work.

Arms (the identity-X group is always a plain ``d_0 * v``; every other arm scans fixed ``2^15``-entry chunks):

* ``(1,0)``, ``(1,2)`` -- today's kernels, the baselines;
* ``P0`` / ``P2``  -- pairs ``(i, j, g)``, two scatters, diagonal recomputed / cached;
* ``C0`` / ``C2``  -- both directions ``(t, s, g)`` sorted by target (CSR), one sorted scatter;
* ``C0i16``        -- C0 with ``int16`` group ids (``int32`` if ``J > 32768``: ``astype`` would wrap);
* ``C2R``          -- C2 with ``float64`` diagonals for real groups, ``complex128`` only for the rest;
* ``seg``          -- C2 as one unchunked ``segment_sum``;
* ``...+RCM``      -- states relabelled by reverse Cuthill--McKee first.

Subcommands (fixture: spinchain's open-XXZ ``xxz`` with Hamming-shell subspaces, ``poc/eigenpair_check_scale``):

* ``matvec``  -- ns/state of one ``(2, N)`` matvec per arm over sizes, each checked against ``(1,0)``;
* ``solve``   -- whole ``ground_locg`` solves at one size, with XLA memory (inputs + temp);
* ``peak``    -- setup-inclusive peak RSS, one fresh process per arm and size;
* ``general`` -- P0/P2/C2R at high hit rate: periodic Neel-Krylov XXZ, molecular-like JW.

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
from rqutils.sqd import (
    _apply_h_kernel,
    _is_filler,
    _pack_scanned,
    _pad_states,
    _spread_seed,
    _z_parity,
    get_diagonal,
    get_xsource,
    run_sqd,
    uniquify_states,
)

CHUNK = 1 << 15
ARMS = (
    "(1,0)",
    "(1,2)",
    "P0",
    "P2",
    "C0",
    "C0i16",
    "C2",
    "C2R",
    "seg",
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
    def body(out, chunk):
        i, j, di = chunk
        out = out.at[..., i].add(di * vec[..., j])
        return out.at[..., j].add(jnp.conj(di) * vec[..., i]), None

    return jax.lax.scan(body, d0 * vec, (pi, pj, d))[0]


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


def spinchain_problem(n, pattern, delta, size):
    ham = PauliSumXZ.from_paulisum(xxz(n, delta, *patterns(n)[pattern]))
    return ham, hamming_shells(n, size, np.random.default_rng(0))


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
        needs_host = bool(({"C0", "C0i16", "C2", "C2R", "seg"} | rcm_arms) & wanted)
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
            k10 = functools.partial(_apply_h_kernel, cache_level=(1, 0))
            ops["(1,0)"] = (jax.jit(k10), (_pack_scanned((1, 0), xs, ham.z, ham.c), su), None)
        if "(1,2)" in arms:
            dg = jax.lax.scan(
                lambda _, v: (None, get_diagonal(v[0], v[1], su)), None, (ham.z, ham.c)
            )[1]
            k12 = functools.partial(_apply_h_kernel, cache_level=(1, 2))
            ops["(1,2)"] = (jax.jit(k12), (_pack_scanned((1, 2), xs, dg, ham.c), None), None)
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


def solve_fn(fn, bound):
    """A ``ground_locg`` solve as ``run_sqd`` drives it: prefilter ``(32, 2)``, ``Σ|c|`` bound, batched."""

    def solve(vinit, *args):
        return ground_locg(
            fn, vinit, args=args, prefilter=(32, 2), prefilter_hi=bound, batch_matvec=True
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
        f"n={args.num_qubits} {args.pattern} delta={args.delta}, chunk {CHUNK}, "
        f"XLA_FLAGS={os.environ.get('XLA_FLAGS', 'default')}: ns/state, then operator B/slot"
    )
    print(f"{'size':>6} " + " ".join(f"{a:>10}" for a in arms))
    for log2 in args.log2_sizes:
        size = 1 << log2
        ham, states = spinchain_problem(args.num_qubits, args.pattern, args.delta, size)
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
    ham, states = spinchain_problem(args.num_qubits, args.pattern, args.delta, size)
    ops, info = operators(ham, states, size, set(arms) | {"(1,0)"})
    vinit = run_sqd_vinit(ham, info["su"], size, info["d0"])
    bound = float(np.abs(np.asarray(ham.c)).sum())
    print(
        f"n={args.num_qubits} {args.pattern} delta={args.delta}, states_size 2^{args.log2_size}, "
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
        solve, x, setup = (lambda: run_sqd(ham, states_p, size, False, (1, 0))), (), 0.0
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
        f"n={args.num_qubits} {args.pattern} delta={args.delta}: peak RSS above baseline, fresh process per arm"
    )
    print(
        f"{'size':>9} {'arm':>7} {'peak MiB':>9} {'B/slot':>7} {'setup s':>8} {'warm s':>8} {'eigval':>18}"
    )
    workdir = args.workdir
    os.makedirs(workdir, exist_ok=True)
    for log2 in args.log2_sizes:
        size = 1 << log2
        path = os.path.join(workdir, f"sparse_pairs_states_{log2}.npy")
        np.save(path, spinchain_problem(args.num_qubits, args.pattern, args.delta, size)[1])
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
    for name in ("matvec", "solve", "peak", "general"):
        p = sub.add_parser(name)
        p.add_argument("--num-qubits", type=int, default=60)
        p.add_argument("--pattern", default="type2")
        p.add_argument("--delta", type=float, default=0.5)
        p.add_argument("--arms", nargs="+", choices=ARMS)
        p.add_argument("--rounds", type=int, default=7)
        if name in ("matvec", "peak"):
            p.add_argument(
                "--log2-sizes",
                type=int,
                nargs="+",
                default=[15, 17, 19, 21] if name == "matvec" else [17, 19, 21],
            )
        if name == "solve":
            p.add_argument("--log2-size", type=int, default=17)
        if name == "peak":
            p.add_argument("--workdir", default="/tmp")
            p.add_argument("--child", choices=ARMS, help=argparse.SUPPRESS)
            p.add_argument("--states", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.cmd == "peak" and args.child:
        print(json.dumps(peak_child(args)))
        return
    {"matvec": cmd_matvec, "solve": cmd_solve, "peak": cmd_peak, "general": cmd_general}[args.cmd](
        args
    )


if __name__ == "__main__":
    main()
