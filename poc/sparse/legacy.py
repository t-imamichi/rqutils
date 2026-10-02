"""``"csr"`` and ``"ell"``, removed from the library and kept here so the ``poc/sparse`` scripts still run.

Both left ``rqutils`` in favour of ``"pairs"`` (``poc/sparse/tune.md`` §3): ``"csr"`` is dominated on both
backends, and ``"ell"`` tuned is mixed against ``"pairs"`` per iteration and behind it end to end. This is
the library code as it was, verbatim but for reading the library's internals through ``sm``, plus a
uniform front end over all three kernels: :func:`operator`, :data:`APPLY` and :func:`run`, keyed by
the kernel's name as a string (``Matvec`` lists ``"pairs"`` only now).
"""

import logging
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np

import rqutils.sqd._sparse as sm
from rqutils.paulis.symplectic import PauliSumXZ
from rqutils.sqd._diagonal import get_diagonal
from rqutils.sqd._solve import SqdResult, _solve
from rqutils.sqd._states import StateList

#: The sparse kernels by name, ``"pairs"`` from the library and the two kept here.
SPARSE = ("pairs", "csr", "ell")

#: ``"ell"``'s row widths, a x1.25 geometric grid: few buckets (each its own compiled scan) at a few
#: percent padding (``poc/sparse/pairs.md``, section 10).
ELL_WIDTHS = np.unique(np.ceil(1.25 ** np.arange(100)).astype(np.int64))


def flat_factors(
    t: np.ndarray,
    s: np.ndarray,
    g: np.ndarray,
    z: jax.Array,
    c: jax.Array,
    states: StateList,
    kmax: int,
) -> jax.Array:
    """:func:`~rqutils.sqd._sparse._entry_factors` over flat host entries, one ``(1, sm._CHUNK)`` chunk per call.

    A fixed chunk shape, so it compiles once per coefficient dtype rather than per bucket shape.
    """
    out = []
    for k in range(0, len(t), sm._CHUNK):
        chunk = [a[k : k + sm._CHUNK] for a in (t, s, g)]
        n = len(chunk[0])
        if n < sm._CHUNK:
            chunk = [np.pad(a, (0, sm._CHUNK - n)) for a in chunk]
        f = sm._entry_factors(*(jnp.asarray(a[None]) for a in chunk), z, c, states, kmax)
        out.append(f[0, :n])
    return jnp.concatenate(out)


def sort_by_target(
    pairs: dict[int, tuple[np.ndarray, np.ndarray]],
    subset: list[int],
    size: int,
    alloc: Callable[[int], list[np.ndarray]],
) -> tuple[list[np.ndarray], np.ndarray]:
    """Counting-sort ``subset``'s transitions, both directions, by target into ``alloc(count)``.

    ``alloc`` returns arrays for ``(target, source, group)`` or ``(source, group)``. A row occurs at
    most once per group, so each group's fill is conflict-free (``poc/sparse/pairs.md``, section 2);
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


def ell_buckets(
    pairs: dict[int, tuple[np.ndarray, np.ndarray]],
    subset: list[int],
    size: int,
    z: jax.Array,
    c_set: jax.Array,
    states_u: StateList,
    kmax: int,
) -> list[jax.Array]:
    """``"ell"``'s ``(rows, src, fac)`` per width for one coefficient set; see :func:`operator`.

    Raises:
        ValueError: If the padded slot count reaches :math:`2^{31}` -- see ``_check_entries``.
    """
    (src, grp), end = sort_by_target(
        pairs, subset, size, lambda count: [np.empty(count, np.int32) for _ in range(2)]
    )
    deg = np.diff(end, prepend=0)
    width = np.where(deg > 0, ELL_WIDTHS[np.searchsorted(ELL_WIDTHS, deg)], 0)
    # Rows per piece fixed per width and pieces size-classed, so shapes come from a bounded set.
    plan = []
    for w in np.unique(width[width > 0]).tolist():
        per = max(1, sm._CHUNK // w)
        rows = np.flatnonzero(width == w).astype(np.int32)
        plan.append((w, per, sm._size_class(-(-len(rows) // per)), rows))
    del width
    sm._check_entries(sum(w * per * pieces for w, per, pieces, _ in plan))
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
        # An empty slot gets source = target, which sm._entry_factors' t == s rule zeroes.
        s, g = np.where(valid, src[slot], r[..., None]), grp[slot]
        del slot
        fac = flat_factors(np.repeat(r, w), s.ravel(), g.ravel(), z, c_set, states_u, kmax)
        del g
        fac.block_until_ready()  # the host-to-device copies of s are asynchronous
        s[~valid] = 0
        del valid
        buckets += [jnp.asarray(r), jnp.asarray(s), fac.reshape(s.shape)]
    return buckets


def operator(
    hamiltonian: PauliSumXZ,
    states_u: StateList,
    matvec: str,
    pairs: dict[int, tuple[np.ndarray, np.ndarray]] | None = None,
) -> tuple[jax.Array, ...]:
    """Build a sparse ``matvec``'s operator arrays on the host, one X group's search at a time.

    Returns ``(d0, i, j, d)`` for ``"pairs"`` and ``(d0, rt, rs, rd, qt, qs, qd)`` for ``"csr"``, each
    entry array ``(chunks, sm._CHUNK)``; ``r``/``q`` are the real-coefficient groups (float64 factors)
    and the rest. Padding entries have equal endpoints and a zero factor.

    ``"ell"`` returns ``d0`` then, per set (real first) and ascending row width ``w``, ``rows``
    ``(pieces, R)``, ``src`` and ``fac`` ``(pieces, R, w)``, ``R = max(1, sm._CHUNK // w)``; slots past a
    row's degree, and padding rows, have source 0 and factor 0.

    ``pairs`` is ``sm._group_pairs``' output, searched here when not given; it is left unconsumed.

    Raises:
        ValueError: If the entry count reaches :math:`2^{31}` -- see ``sm._check_entries``.
    """
    if matvec == "pairs":
        return sm._sparse_operator(hamiltonian, states_u, pairs)
    size = states_u.shape[0]
    chunk = sm._CHUNK
    z, c = jnp.asarray(hamiltonian.z), jnp.asarray(hamiltonian.c)
    # A copy: _sort_by_target pops each group as it writes it.
    pairs = dict(sm._group_pairs(hamiltonian, states_u) if pairs is None else pairs)
    first = int(hamiltonian.identity_first)
    d0 = get_diagonal(z[0], c[0], states_u) if first else jnp.zeros(size, c.dtype)
    groups = range(first, hamiltonian.x.shape[0])
    coeffs = np.asarray(hamiltonian.c)
    kmax = max((int(np.count_nonzero(coeffs[g])) for g in groups), default=1)

    def on_device(host, c_set):
        # Pops each host array as it is copied, so no host entry array outlives its device copy.
        t, s, g = (jnp.asarray(host.pop(0).reshape(-1, chunk)) for _ in range(3))
        return t, s, sm._entry_factors(t, s, g, z, c_set, states_u, kmax)

    def alloc(count):  # i, j, group
        return [sm._padded(count, f, chunk) for f in (size - 1, size - 1, 0)]

    real = np.isreal(coeffs).all(axis=1)
    arrays = [d0]
    for subset, c_set in (
        ([g for g in groups if real[g]], c.real),
        ([g for g in groups if not real[g]], c),
    ):
        if matvec == "ell":
            arrays += ell_buckets(pairs, subset, size, z, c_set, states_u, kmax)
            continue
        host = sort_by_target(pairs, subset, size, alloc)[0]
        arrays += on_device(host, c_set)
    return tuple(arrays)


def apply_csr(vec: jax.Array, d0: jax.Array, *entries: jax.Array) -> jax.Array:
    """``"csr"``: ``d0 * vec``, then ``out[t] += d * vec[s]`` over each target-sorted ``(t, s, d)`` set."""
    sharding = jax.typeof(vec).sharding

    def updates(chunk):
        ti, si, di = chunk
        return [(ti, di * vec.at[..., si].get(out_sharding=sharding))]

    out = d0 * vec
    for k in range(0, len(entries), 3):
        out = sm._scan_add(updates, out, entries[k : k + 3])
    return out


def apply_ell(vec: jax.Array, d0: jax.Array, *buckets: jax.Array) -> jax.Array:
    """``"ell"``: ``d0 * vec``, then per ``(rows, src, fac)`` piece ``out[rows] += sum(fac * vec[src])``."""
    sharding = jax.typeof(vec).sharding

    def updates(piece):
        rows, src, fac = piece
        return [(rows, jnp.sum(fac * vec.at[..., src].get(out_sharding=sharding), axis=-1))]

    out = d0 * vec
    for k in range(0, len(buckets), 3):
        out = sm._scan_add(updates, out, buckets[k : k + 3])
    return out


#: Each kernel's matvec by name; :func:`run` reads it at trace time.
APPLY = {"pairs": sm._apply_pairs, "csr": apply_csr, "ell": apply_ell}


@jax.jit(
    static_argnames=("states_size", "return_eigvec", "matvec", "maxiter", "prefilter", "log_level")
)
def run(
    hamiltonian: PauliSumXZ,
    states_u: StateList,
    operator: tuple[jax.Array, ...],
    states_size: int,
    return_eigvec: bool,
    matvec: str,
    maxiter: int = 1000,
    atol: float = 0.0,
    rtol: float | None = None,
    prefilter: tuple[int, int] | None = (32, 2),
    log_level: int = logging.INFO,
) -> SqdResult:
    """``_run_sparse`` for any of the three kernels, given :func:`operator`'s arrays.

    It runs no residual check: check with ``sm._sparse_residual``, which reads none of the operator.
    """
    return _solve(
        hamiltonian,
        states_u,
        APPLY[matvec],
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
