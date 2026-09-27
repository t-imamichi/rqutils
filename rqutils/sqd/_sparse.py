"""The sparse kernels, ``"pairs"``, ``"csr"`` and ``"ell"``: host-side construction and the solve."""

import functools
import logging
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np

from rqutils.paulis.symplectic import PauliSumXZ
from rqutils.sqd._diagonal import _z_parity, get_diagonal
from rqutils.sqd._solve import _SOLVE_STATIC, Matvec, SqdResult, _solve
from rqutils.sqd._states import _MAX_STATES, StateList, get_xsource

#: Entries per scanned chunk, so the sparse kernels' temporaries are ``O(chunk)`` (``poc/sparse-pairs.md``).
_CHUNK = 1 << 15


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


def _padded(count: int, fill: int) -> np.ndarray:
    """A flat int32 array of ``count`` entries rounded up to whole chunks of a size class, all ``fill``.

    Raises:
        ValueError: See :func:`_check_entries`.
    """
    _check_entries(count)
    return np.full(_size_class(-(-count // _CHUNK)) * _CHUNK, fill, dtype=np.int32)


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


#: ``"ell"``'s row widths, a x1.25 geometric grid: few buckets (each its own compiled scan) at a few
#: percent padding (``poc/sparse-pairs.md``, section 10).
_ELL_WIDTHS = np.unique(np.ceil(1.25 ** np.arange(100)).astype(np.int64))


def _flat_factors(
    t: np.ndarray,
    s: np.ndarray,
    g: np.ndarray,
    z: jax.Array,
    c: jax.Array,
    states: StateList,
    kmax: int,
) -> jax.Array:
    """:func:`_entry_factors` over flat host entries, one ``(1, _CHUNK)`` chunk per call.

    A fixed chunk shape, so it compiles once per coefficient dtype rather than per bucket shape.
    """
    out = []
    for k in range(0, len(t), _CHUNK):
        chunk = [a[k : k + _CHUNK] for a in (t, s, g)]
        n = len(chunk[0])
        if n < _CHUNK:
            chunk = [np.pad(a, (0, _CHUNK - n)) for a in chunk]
        f = _entry_factors(*(jnp.asarray(a[None]) for a in chunk), z, c, states, kmax)
        out.append(f[0, :n])
    return jnp.concatenate(out)


def _sort_by_target(
    pairs: dict[int, tuple[np.ndarray, np.ndarray]],
    subset: list[int],
    size: int,
    alloc: Callable[[int], list[np.ndarray]],
) -> tuple[list[np.ndarray], np.ndarray]:
    """Counting-sort ``subset``'s transitions, both directions, by target into ``alloc(count)``.

    ``alloc`` returns arrays for ``(target, source, group)`` or ``(source, group)``. A row occurs at
    most once per group, so each group's fill is conflict-free (``poc/sparse-pairs.md``, section 2);
    each group is popped from ``pairs`` once written. Returns the arrays and each row's end offset.
    """
    end = np.zeros(size + 1, np.int64)
    for g in subset:
        for rows in pairs[g]:
            end[rows + 1] += 1
    np.cumsum(end, out=end)
    out = alloc(int(end[-1]))
    for g in subset:
        i, j = pairs.pop(g)
        for target, source in ((i, j), (j, i)):
            pos = end[target]
            for array, value in zip(out, (target, source, g)[-len(out) :]):
                array[pos] = value
            end[target] += 1
    return out, end[:-1]


def _ell_buckets(
    pairs: dict[int, tuple[np.ndarray, np.ndarray]],
    subset: list[int],
    size: int,
    z: jax.Array,
    c_set: jax.Array,
    states_u: StateList,
    kmax: int,
) -> list[jax.Array]:
    """``"ell"``'s ``(rows, src, fac)`` per width for one coefficient set; see :func:`_sparse_operator`.

    Raises:
        ValueError: If the padded slot count reaches :math:`2^{31}` -- see :func:`_check_entries`.
    """
    (src, grp), end = _sort_by_target(
        pairs, subset, size, lambda count: [np.empty(count, np.int32) for _ in range(2)]
    )
    deg = np.diff(end, prepend=0)
    width = np.where(deg > 0, _ELL_WIDTHS[np.searchsorted(_ELL_WIDTHS, deg)], 0)
    # Rows per piece fixed per width and pieces size-classed, so shapes come from a bounded set.
    plan = []
    for w in np.unique(width[width > 0]).tolist():
        per = max(1, _CHUNK // w)
        rows = np.flatnonzero(width == w).astype(np.int32)
        plan.append((w, per, _size_class(-(-len(rows) // per)), rows))
    del width
    _check_entries(sum(w * per * pieces for w, per, pieces, _ in plan))
    pos = (end - deg).astype(np.int32)  # row starts
    del end
    buckets = []
    for w, per, pieces, rows in plan:
        pad = (0, pieces * per - len(rows))
        # Padding rows repeat a real row with no entries: a dummy output slot costs two (2, N) copies.
        r = np.pad(rows, pad, constant_values=rows[0]).reshape(pieces, per)
        lane = np.arange(w, dtype=np.int32)
        valid = lane < np.pad(deg[rows], pad).reshape(pieces, per, 1)
        slot = np.where(valid, pos[r][..., None] + lane, 0)
        # An empty slot gets source = target, which _entry_factors' t == s rule zeroes.
        s, g = np.where(valid, src[slot], r[..., None]), grp[slot]
        del slot
        fac = _flat_factors(np.repeat(r, w), s.ravel(), g.ravel(), z, c_set, states_u, kmax)
        del g
        fac.block_until_ready()  # the host-to-device copies of s are asynchronous
        s[~valid] = 0
        del valid
        buckets += [jnp.asarray(r), jnp.asarray(s), fac.reshape(s.shape)]
    return buckets


def _sparse_operator(
    hamiltonian: PauliSumXZ, states_u: StateList, matvec: Matvec
) -> tuple[jax.Array, ...]:
    """Build a sparse ``matvec``'s operator arrays on the host, one X group's search at a time.

    Returns ``(d0, i, j, d)`` for ``"pairs"`` and ``(d0, rt, rs, rd, qt, qs, qd)`` for ``"csr"``, each
    entry array ``(chunks, _CHUNK)``; ``r``/``q`` are the real-coefficient groups (float64 factors)
    and the rest. Padding entries have equal endpoints and a zero factor.

    ``"ell"`` returns ``d0`` then, per set (real first) and ascending row width ``w``, ``rows``
    ``(pieces, R)``, ``src`` and ``fac`` ``(pieces, R, w)``, ``R = max(1, _CHUNK // w)``; slots past a
    row's degree, and padding rows, have source 0 and factor 0.

    Raises:
        ValueError: If the entry count reaches :math:`2^{31}` -- see :func:`_check_entries`.
    """
    size = states_u.shape[0]
    z, c = jnp.asarray(hamiltonian.z), jnp.asarray(hamiltonian.c)
    first = int(np.all(np.asarray(hamiltonian.x[0]) == 0))
    d0 = get_diagonal(z[0], c[0], states_u) if first else jnp.zeros(size, c.dtype)
    groups = range(first, hamiltonian.x.shape[0])
    coeffs = np.asarray(hamiltonian.c)
    kmax = max((int(np.count_nonzero(coeffs[g])) for g in groups), default=1)
    rows = np.arange(size, dtype=np.int32)
    pairs = {}
    for g in groups:
        j = np.asarray(get_xsource(hamiltonian.x[g], states_u))
        keep = j > rows  # each pair once; absent sources (-1) and filler rows (always absent) drop
        pairs[g] = (rows[keep], j[keep])
    del rows

    def on_device(host, c_set):
        # Pops each host array as it is copied, so no host entry array outlives its device copy.
        t, s, g = (jnp.asarray(host.pop(0).reshape(-1, _CHUNK)) for _ in range(3))
        return t, s, _entry_factors(t, s, g, z, c_set, states_u, kmax)

    if matvec == "pairs":
        count = sum(len(i) for i, _ in pairs.values())
        host = [_padded(count, fill) for fill in (size - 1, size - 1, 0)]  # i, j, group
        pos = 0
        for g in groups:
            i, j = pairs.pop(g)
            for array, value in zip(host, (i, j, g)):
                array[pos : pos + len(i)] = value
            pos += len(i)
        return (d0, *on_device(host, c))

    real = np.isreal(coeffs).all(axis=1)
    arrays = [d0]
    for subset, c_set in (
        ([g for g in groups if real[g]], c.real),
        ([g for g in groups if not real[g]], c),
    ):
        if matvec == "ell":
            arrays += _ell_buckets(pairs, subset, size, z, c_set, states_u, kmax)
            continue
        host = _sort_by_target(
            pairs, subset, size, lambda count: [_padded(count, f) for f in (size - 1, size - 1, 0)]
        )[0]
        arrays += on_device(host, c_set)
    return tuple(arrays)


def _apply_pairs(
    vec: jax.Array, d0: jax.Array, pi: jax.Array, pj: jax.Array, d: jax.Array
) -> jax.Array:
    """``"pairs"``: ``d0 * vec``, then per pair ``out[i] += d * vec[j]`` and ``out[j] += conj(d) * vec[i]``."""
    sharding = jax.typeof(vec).sharding

    def body(out, chunk):
        i, j, di = chunk
        out = out.at[..., i].add(di * vec.at[..., j].get(out_sharding=sharding))
        return out.at[..., j].add(jnp.conj(di) * vec.at[..., i].get(out_sharding=sharding)), None

    return jax.lax.scan(body, d0 * vec, (pi, pj, d))[0]


def _apply_csr(vec: jax.Array, d0: jax.Array, *entries: jax.Array) -> jax.Array:
    """``"csr"``: ``d0 * vec``, then ``out[t] += d * vec[s]`` over each target-sorted ``(t, s, d)`` set."""
    sharding = jax.typeof(vec).sharding

    def body(acc, chunk):
        ti, si, di = chunk
        gathered = vec.at[..., si].get(out_sharding=sharding)
        return acc.at[..., ti].add(di * gathered, indices_are_sorted=True), None

    out = d0 * vec
    for k in range(0, len(entries), 3):
        out = jax.lax.scan(body, out, entries[k : k + 3])[0]
    return out


def _apply_ell(vec: jax.Array, d0: jax.Array, *buckets: jax.Array) -> jax.Array:
    """``"ell"``: ``d0 * vec``, then per ``(rows, src, fac)`` piece ``out[rows] += sum(fac * vec[src])``."""
    sharding = jax.typeof(vec).sharding

    def body(acc, piece):
        rows, src, fac = piece
        val = jnp.sum(fac * vec.at[..., src].get(out_sharding=sharding), axis=-1)
        return acc.at[..., rows].add(val), None

    out = d0 * vec
    for k in range(0, len(buckets), 3):
        out = jax.lax.scan(body, out, buckets[k : k + 3])[0]
    return out


_SPARSE_APPLY = {"pairs": _apply_pairs, "csr": _apply_csr, "ell": _apply_ell}


@jax.jit(static_argnames=_SOLVE_STATIC)
def _run_sparse(
    hamiltonian: PauliSumXZ,
    states_u: StateList,
    operator: tuple[jax.Array, ...],
    states_size: int,
    return_eigvec: bool,
    matvec: Matvec,
    maxiter: int = 1000,
    atol: float = 0.0,
    rtol: float | None = None,
    prefilter: tuple[int, int] | None = (32, 2),
    log_level: int = logging.INFO,
    check_residual: bool = False,
) -> SqdResult:
    """:func:`run_sqd` for the sparse kernels, given :func:`_sparse_operator`'s arrays.

    The residual check runs the ``"onthefly"`` kernel, so it reads none of ``operator``.
    """
    return _solve(
        hamiltonian,
        states_u,
        _SPARSE_APPLY[matvec],
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
        check_residual=check_residual,
    )
