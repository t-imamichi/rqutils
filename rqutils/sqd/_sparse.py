"""The sparse kernel, ``"pairs"``: host-side construction and the solve."""

import functools
import logging
import os
from collections import deque
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor

import jax
import jax.numpy as jnp
import numpy as np

from rqutils.paulis.symplectic import PauliSumXZ
from rqutils.sqd._dense import _apply_h_kernel, _pack_scanned
from rqutils.sqd._diagonal import _z_parity, get_diagonal
from rqutils.sqd._solve import _SOLVE_STATIC, Matvec, SqdResult, _solve
from rqutils.sqd._states import _MAX_STATES, StateList, _is_filler

#: Entries per scanned chunk, so the sparse kernels' temporaries are ``O(chunk)`` (``poc/sparse/pairs.md``).
_CHUNK = 1 << 15
#: ``"pairs"``' chunk on a GPU, which ``_CHUNK`` under-fills: 2.71x/1.37x per iteration on a GH200 at
#: ``2^20``/``2^22``, the smallest size on the plateau, +8 MiB temp (``poc/sparse/tune.md``).
_GPU_PAIRS_CHUNK = 1 << 19


def _chunk() -> int:
    """Entries per scanned chunk on the default backend."""
    return _GPU_PAIRS_CHUNK if jax.default_backend() == "gpu" else _CHUNK


def _pairs_sorted_on_device(
    pairs: dict[int, tuple[np.ndarray, np.ndarray]], groups: range, alloc: Callable
) -> list[jax.Array]:
    """``"pairs"``' ``(i, j, group)``, each pair once, stably sorted by ``i`` on the device.

    Stable, so a row's pairs keep group order. 1.12-1.50x per build-plus-solve over a host counting sort
    on a GH200, 0.98-0.99x on an M1 (``poc/sparse/pairs-sort.md``).
    """
    sizes = [len(pairs[g][0]) for g in groups]
    i, j, grp = alloc(count := sum(sizes))
    for k, out in enumerate((i, j)):  # the empty array: no group but the identity
        np.concatenate([np.empty(0, np.int32), *(pairs[g][k] for g in groups)], out=out[:count])
    grp[:count] = np.repeat(np.asarray(groups, np.int32), sizes)
    # Padding's i is size - 1, above every pair's i < j, so it sorts last.
    i, j, grp = map(jnp.asarray, (i, j, grp))
    order = jnp.argsort(i, stable=True)
    return [i[order], j[order], grp[order]]


def _size_class(chunks: int) -> int:
    """Round a chunk count up to ``m * 2**k`` with ``8 <= m < 16`` (exact below 16), at least 1.

    The jitted solve recompiles per class rather than per subspace, wasting at most 12.5%.
    """
    shift = max(chunks.bit_length() - 4, 0)
    return max(-(-chunks >> shift) << shift, 1)


def _check_entries(count: int) -> None:
    """Raise ValueError if ``count`` entries reach :math:`2^{31}`, past int32 indexing."""
    if count > _MAX_STATES:
        raise ValueError(
            f"the sparse operator has {count} entries, beyond the 2^31 - 1 addressable with int32 "
            "indices; use matvec=Matvec.INDICES or a smaller subspace"
        )


#: Rows per block of a host search, bounding each thread's temporaries.
_SEARCH_ROWS = 1 << 18


def _words(rows: np.ndarray) -> np.ndarray:
    """MSW-first uint64 words per row, the host twin of :func:`~rqutils.sqd._states._pack_state_words`."""
    return np.pad(rows, ((0, 0), (-rows.shape[1] % 8, 0))).view(">u8").astype(np.uint64)


def _host_sources[T](
    x: np.ndarray, states_u: StateList, reduce: Callable[[np.ndarray], T]
) -> Iterator[T]:
    """Per X signature, in order, ``reduce`` of :func:`get_xsource`'s int32 sources, one group a thread.

    Folds each row's words, MSW-first, into its rank among the states' sorted rows: ``(rank << 32) |
    word rank`` stays in uint64 while both are below :math:`2^{31}`. A state's rank is its row index;
    any other target gets some index, which the word comparison rejects. Fillers sort last and never
    pair, so only real rows are searched. ``reduce`` runs in the thread; one window of groups is live.
    """
    host = np.asarray(states_u)
    real = len(host) - int(np.count_nonzero(_is_filler(host)))
    words = _words(host[:real])

    def pack(hi, lo):
        return (hi.astype(np.uint64) << np.uint64(32)) | lo.astype(np.uint64)

    def distinct(ordered):  # sorted input: distinct values and ranks in one linear pass
        new = np.r_[True, ordered[1:] != ordered[:-1]]
        return ordered[new], np.cumsum(new) - 1

    # Rows are lex-sorted, so the first word and every rank pair are sorted; only later words sort.
    # `own` keeps each state's word ranks: a word an X signature leaves unchanged needs no search.
    first, rank = distinct(words[:, 0])
    levels, own = [], [rank]
    for w in words.T[1:]:
        values, inverse = np.unique(w, return_inverse=True)
        pairs, rank = distinct(pack(rank, inverse))
        levels.append((values, pairs))
        own.append(inverse)
    values = [first, *(v for v, _ in levels)]

    def one(xw):
        xsource = np.full(len(host), -1, np.int32)
        for lo in range(0, real, _SEARCH_ROWS):
            target = words[lo : lo + _SEARCH_ROWS] ^ xw
            rows = slice(lo, lo + len(target))
            ranks = [
                own[k][rows] if xw[k] == 0 else np.searchsorted(v, target[:, k])
                for k, v in enumerate(values)
            ]
            rank = ranks[0]
            for (_, pairs), word in zip(levels, ranks[1:], strict=True):
                rank = np.searchsorted(pairs, pack(rank, word))
            j = np.minimum(rank, real - 1).astype(np.int32)
            hit = np.all(words[j] == target, axis=1)
            xsource[lo : lo + len(target)][hit] = j[hit]
        return reduce(xsource)

    # np.searchsorted releases the GIL, so threads scale where the jitted search runs on one core.
    # A sliding window, not batches: at most `workers` groups in flight, none waiting on the slowest.
    # ponytail: Executor.map(buffersize=workers) replaces this once the floor is Python 3.14.
    workers = os.cpu_count() or 1
    with ThreadPoolExecutor(workers) as pool:
        pending = deque()
        for xw in _words(np.asarray(x)):
            pending.append(pool.submit(one, xw))
            if len(pending) >= workers:
                yield pending.popleft().result()
        while pending:
            yield pending.popleft().result()


def _search_pairs(x: np.ndarray, states_u: StateList) -> list[tuple[np.ndarray, np.ndarray]]:
    """Per X signature, the ``(i, j)`` with ``j > i`` that :func:`get_xsource` finds."""
    rows = np.arange(states_u.shape[0], dtype=np.int32)

    def pairs(j):
        keep = j > rows  # each pair once; absent sources (-1) and fillers drop
        return rows[keep], j[keep]

    return list(_host_sources(x, states_u, pairs))


def _group_pairs(
    hamiltonian: PauliSumXZ, states_u: StateList
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """:func:`_search_pairs` keyed by group, for every group but a leading identity."""
    groups = range(int(hamiltonian.identity_first), hamiltonian.x.shape[0])
    return dict(
        zip(groups, _search_pairs(np.asarray(hamiltonian.x)[groups], states_u), strict=True)
    )


def _pair_xsources(
    x: np.ndarray, states_u: StateList, pairs: dict[int, tuple[np.ndarray, np.ndarray]]
) -> Iterator[np.ndarray]:
    """:func:`_host_sources`' sources per X signature, rebuilt from ``pairs`` rather than searched.

    XOR is an involution, so ``(i, j)`` gives both ``xsource[i] = j`` and ``xsource[j] = i``; the
    identity group, the one :func:`_group_pairs` skips, maps each real row to itself.
    """
    size = states_u.shape[0]
    real = size - int(np.count_nonzero(_is_filler(np.asarray(states_u))))
    for g in range(len(x)):
        xsource = np.full(size, -1, np.int32)
        if g not in pairs:
            xsource[:real] = np.arange(real, dtype=np.int32)
        else:
            i, j = pairs[g]
            xsource[i], xsource[j] = j, i
        yield xsource


def _sparse_residual(
    hamiltonian: PauliSumXZ,
    states_u: StateList,
    eigval: jax.Array,
    eigvec: jax.Array,
    pairs: dict[int, tuple[np.ndarray, np.ndarray]],
) -> tuple[jax.Array, jax.Array]:
    """``(||Hv - Ev||, ||Hv||)`` by the ``"indices"`` kernel one group at a time, from the build's ``pairs``.

    Reads none of the sparse operator, and reuses the search (:func:`_group_pairs`) as the dense kernels
    reuse cached xsources (``NOTES.md``, "sqd sparse kernels: the residual check runs on the host").
    """
    x, z, c = hamiltonian.arrays
    ax = jnp.zeros_like(eigvec)
    for g, xsource in enumerate(_pair_xsources(x, states_u, pairs)):
        scanned = _pack_scanned(Matvec.INDICES, xsource[None], z[g : g + 1], c[g : g + 1])
        ax = _apply_h_kernel(eigvec, scanned, states_u, matvec=Matvec.INDICES, init=ax)
    return jnp.linalg.norm(ax - eigval * eigvec), jnp.linalg.norm(ax)


def _padded(count: int, fill: int, chunk: int) -> np.ndarray:
    """A flat int32 array of ``count`` entries rounded up to whole ``chunk``s of a size class, all ``fill``.

    Raises:
        ValueError: See :func:`_check_entries`.
    """
    _check_entries(count)
    return np.full(_size_class(-(-count // chunk)) * chunk, fill, dtype=np.int32)


@functools.partial(jax.jit, static_argnames="kmax")
def _entry_factors(
    target: jax.Array,
    source: jax.Array,
    group: jax.Array,
    z: jax.Array,
    c: jax.Array,
    states: StateList,
    kmax: int,
) -> jax.Array:
    """``H[target, source] = d_group(target)`` per chunked entry, zero where the endpoints coincide."""

    def one(chunk):
        t, s, g = chunk
        parity = jax.vmap(_z_parity, in_axes=(None, 1), out_axes=1)(states[t], z[g][:, :kmax])
        factor = jnp.sum(c[g][:, :kmax] * (1.0 - 2.0 * parity), axis=-1)
        return jnp.where(t == s, jnp.zeros_like(factor), factor)

    return jax.lax.map(one, (target, source, group))


def _drop_zeros(
    t: jax.Array, s: jax.Array, d: jax.Array, chunk: int, size: int
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """``(t, s, d)`` without the entries whose factor is exactly zero, order kept, re-padded to a size class.

    XX+YY hops cancel on aligned spins: 86% of spinchain's ``type1`` pairs, 32% of ``type2``'s
    (``poc/sparse/prune.md``). The check reads the unfiltered ``pairs``, so it vouches for this filter.
    """
    keep = (d != 0).ravel()
    count = int(keep.sum())
    length = _size_class(-(-count // chunk)) * chunk
    (idx,) = jnp.nonzero(keep, size=length, fill_value=0)
    live = jnp.arange(length) < count
    # Padding on distinct rows: one shared row serialized a GPU's atomic adds (poc/sparse/prune.md §6).
    pad = jnp.arange(length, dtype=t.dtype) % size
    t, s = (jnp.where(live, a.ravel()[idx], pad).reshape(-1, chunk) for a in (t, s))
    return t, s, jnp.where(live, d.ravel()[idx], 0).reshape(-1, chunk)


def _sparse_operator(
    hamiltonian: PauliSumXZ,
    states_u: StateList,
    pairs: dict[int, tuple[np.ndarray, np.ndarray]] | None = None,
) -> tuple[jax.Array, ...]:
    """Build ``"pairs"``' ``(d0, i, j, d)`` on the host, each entry array ``(chunks, _chunk())``.

    Each transition once, sorted by ``i`` across groups so ``out[i]`` and ``vec[i]`` are local
    (``NOTES.md``, "sqd sparse kernels: pairs sorted by i"); padding entries have equal endpoints and a
    zero factor, and so does no stored pair (:func:`_drop_zeros`). ``pairs`` is :func:`_group_pairs`'
    output, searched here when not given.

    Raises:
        ValueError: If the entry count reaches :math:`2^{31}` -- see :func:`_check_entries`.
    """
    size, chunk = states_u.shape[0], _chunk()
    z, c = jnp.asarray(hamiltonian.z), jnp.asarray(hamiltonian.c)
    pairs = _group_pairs(hamiltonian, states_u) if pairs is None else pairs
    first = int(hamiltonian.identity_first)
    d0 = get_diagonal(z[0], c[0], states_u) if first else jnp.zeros(size, c.dtype)
    groups = range(first, hamiltonian.x.shape[0])
    coeffs = np.asarray(hamiltonian.c)
    kmax = max((int(np.count_nonzero(coeffs[g])) for g in groups), default=1)

    def alloc(count):  # i, j, group
        return [_padded(count, f, chunk) for f in (size - 1, size - 1, 0)]

    t, s, g = (a.reshape(-1, chunk) for a in _pairs_sorted_on_device(pairs, groups, alloc))
    return d0, *_drop_zeros(t, s, _entry_factors(t, s, g, z, c, states_u, kmax), chunk, size)


def _scan_add(
    updates: Callable[[tuple[jax.Array, ...]], list[tuple[jax.Array, jax.Array]]],
    out: jax.Array,
    xs: tuple[jax.Array, ...],
) -> jax.Array:
    """``out`` after a scan adding each chunk's ``updates`` -- ``(index, value)`` -- into it.

    On CUDA a complex ``out`` is carried as its real and imaginary parts: XLA's GPU scatter otherwise
    splits the carry itself, a full pass over ``out`` per scan step. Elsewhere that costs 0.68-0.90x
    and an extra ``out`` of temp, so the carry stays complex (``poc/sparse/split.md``). No scatter is
    marked sorted: that slows a GPU scatter 2.1-3.2x and buys a CPU nothing (``poc/sparse/tune.md``).
    """

    def scan(*parts):
        def body(parts, chunk):
            for index, value in updates(chunk):
                values = (value.real, value.imag) if len(parts) == 2 else (value,)
                parts = tuple(p.at[..., index].add(v) for p, v in zip(parts, values, strict=True))
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


def _apply_pairs(
    vec: jax.Array, d0: jax.Array, pi: jax.Array, pj: jax.Array, d: jax.Array
) -> jax.Array:
    """``"pairs"``: ``d0 * vec``, then per pair ``out[i] += d * vec[j]`` and ``out[j] += conj(d) * vec[i]``."""
    sharding = jax.typeof(vec).sharding

    def updates(chunk):
        i, j, di = chunk
        return [
            (i, di * vec.at[..., j].get(out_sharding=sharding)),
            (j, jnp.conj(di) * vec.at[..., i].get(out_sharding=sharding)),
        ]

    return _scan_add(updates, d0 * vec, (pi, pj, d))


@jax.jit(static_argnames=[s for s in _SOLVE_STATIC if s != "matvec"])
def _run_sparse(
    hamiltonian: PauliSumXZ,
    states_u: StateList,
    operator: tuple[jax.Array, ...],
    states_size: int,
    return_eigvec: bool,
    maxiter: int = 1000,
    atol: float = 0.0,
    rtol: float | None = None,
    prefilter: tuple[int, int] | None = (32, 2),
    log_level: int = logging.INFO,
) -> SqdResult:
    """:func:`run_sqd` for ``"pairs"``, given :func:`_sparse_operator`'s arrays.

    It runs no residual check: :func:`sqd` checks with :func:`_sparse_residual` after it returns.
    """
    return _solve(
        hamiltonian,
        states_u,
        _apply_pairs,
        operator,
        lambda: operator[0],
        None,
        None,
        return_eigvec=return_eigvec,
        maxiter=maxiter,
        atol=atol,
        rtol=rtol,
        prefilter=prefilter,
        log_level=log_level,
        check_residual=False,
    )
