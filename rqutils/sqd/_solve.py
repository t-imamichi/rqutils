"""The kernel names, the jitted solve every kernel shares, and :func:`run_sqd` for the dense ones."""

import functools
import logging
from collections.abc import Callable
from enum import StrEnum
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.multihost_utils import process_allgather
from jax.sharding import PartitionSpec, get_abstract_mesh
from numpy.typing import DTypeLike

from rqutils.ground_locg import _check_prefilter, ground_locg, residual_floor
from rqutils.paulis.symplectic import PauliSumXZ
from rqutils.sqd._dense import _apply_h_kernel, _pack_scanned
from rqutils.sqd._diagonal import get_diagonal
from rqutils.sqd._states import StateList, _is_filler, get_xsource, uniquify_states


class Matvec(StrEnum):
    """A :func:`~rqutils.sqd.sqd` matvec kernel, named by what it stores (module documentation).

    Pass a member (``Matvec.ELL``); a plain string is rejected. Members are ``str`` subclasses only so
    that the kernels' internal equality tests stay plain.
    """

    ONTHEFLY = "onthefly"
    INDICES = "indices"
    TABLES = "tables"
    PAIRS = "pairs"
    CSR = "csr"
    ELL = "ell"


#: The kernels whose operator arrays :func:`sqd` builds host-side; single-device for now.
_SPARSE_MATVECS = (Matvec.PAIRS, Matvec.CSR, Matvec.ELL)


def _check_matvec(matvec: Any) -> None:
    """Raise unless ``matvec`` is a :class:`Matvec` member.

    Every branch on ``matvec`` is an equality test with an implicit ``else``, so an unvalidated value
    would be absorbed into some kernel rather than reported.

    Raises:
        TypeError: If it is not a :class:`Matvec` member: a plain string such as ``"ell"``, or the
            removed ``cache_level`` tuple, included.
    """
    if not isinstance(matvec, Matvec):
        names = ", ".join(f"Matvec.{m.name}" for m in Matvec)
        raise TypeError(f"`matvec` must be a Matvec member, one of {names}; got {matvec!r}")


def _residual_floor_of(hamiltonian: PauliSumXZ) -> float:
    """The achievable eigen-residual floor for this Hamiltonian, for guards and error messages."""
    return residual_floor(float(np.abs(hamiltonian.c).sum()), hamiltonian.c.dtype)


#: Headroom of :class:`EigenpairCheckError` over the convergence bound: converged solves measure at
#: most 0.96 of it, so this only fires on a pair that is wrong rather than marginal (``NOTES.md``).
_RESIDUAL_SLACK = 10.0


def _host_scalar(value: jax.Array | float | bool) -> jax.Array | float | bool:
    """Return a replicated scalar in a form the host can read, on any process topology.

    ``float()``/``bool()`` **raise across processes** on a rank-0 reduction over a partitioned vector,
    whose sharding still names the whole mesh ("spans non-addressable devices"). ``jax.reshard`` does
    not help (the spec is already ``P()``), so multi-process goes through
    :func:`jax.experimental.multihost_utils.process_allgather`.

    **The branch is on ``jax.process_count()``, not on this rank's view of the array**: that is
    identical on every rank, so no rank enters the collective alone and hangs the job. Single process
    short-circuits (``NOTES.md``, "sqd._host_scalar: one path on every rank").

    Args:
        value: A rank-0 ``jax.Array``, or an already-host scalar (returned unchanged).

    Returns:
        Something ``float()`` or ``bool()`` accepts.
    """
    if not isinstance(value, jax.Array):
        # Already a Python or numpy scalar.
        return value
    if jax.process_count() == 1:
        # Single process addresses every device, so `float()` works directly and no collective is
        # needed. This is also the only branch a CPU test suite can reach -- see TestHostScalar.
        return value

    # Multi-process: gather unconditionally, branching only on process_count (the same on every
    # rank), then read reshape(-1)[0] (NOTES.md, "sqd._host_scalar: one path on every rank").
    return np.asarray(process_allgather(value, tiled=True)).reshape(-1)[0]


def _spread_seed(
    states_size: int, states_u: StateList, dtype: DTypeLike, sharding: PartitionSpec | None
) -> jax.Array:
    """Return a deterministic pseudo-random unit-scale vector over the subspace.

    Used as (part of) ``run_sqd``'s initial vector. The point is coverage, not quality: LOBPCG
    requires a non-vanishing overlap with the ground state, and a one-hot seed can fail that
    outright -- it cannot leave the connected component of the projected Hamiltonian that contains
    it, so a subspace whose Hamiltonian splits into disconnected blocks yields that block's
    minimum rather than the global one. See the comments at both call sites for the measured cases.

    The values come from a fixed bit-mixing hash of the index rather than from ``jax.random``, so
    this stays a pure function of ``states_size`` with no PRNG key to thread through the public
    signature, and is reproducible run to run. An unreproducible eigensolver seed would make
    convergence itself irreproducible, which is a poor trade for a slightly better spread.

    Fill-in slots produced by uniquification (marked by the high bit of byte 0) are zeroed: they
    carry no basis state, so weight there would place the iterate partly outside the subspace.
    """
    index = jax.lax.broadcasted_iota(jnp.uint32, (states_size,), 0, out_sharding=sharding)
    # Two xorshift-multiply rounds (Murmur-style constants): enough to decorrelate consecutive
    # indices, so the seed is not accidentally orthogonal to any one eigenvector.
    mixed = index ^ (index >> 16)
    mixed = mixed * jnp.uint32(0x7FEB352D)
    mixed = mixed ^ (mixed >> 15)
    mixed = mixed * jnp.uint32(0x846CA68B)
    mixed = mixed ^ (mixed >> 16)
    # Map to [-1, 1). The distribution does not matter, only that no entry is systematically zero.
    vec = mixed.astype(dtype) * (2.0 / float(2**32)) - 1.0
    # Reshard the mask to `vec`: states_u arrives replicated under "onthefly", and this
    # function owns vec's sharding (NOTES.md, "sqd._spread_seed: reshard the filler mask").
    filler = _is_filler(states_u) == 1
    if sharding is not None:
        filler = jax.reshard(filler, sharding)
    return jnp.where(filler, jnp.zeros_like(vec), vec)


class SqdResult(NamedTuple):
    """What :func:`run_sqd` returns; the optional fields are ``None`` unless requested."""

    eigval: jax.Array
    converged: jax.Array
    eigvec: jax.Array | None = None
    states: jax.Array | None = None
    subspace_dim: jax.Array | None = None
    residual: jax.Array | None = None
    ax_norm: jax.Array | None = None


#: Static arguments of both jitted solve entry points, :func:`run_sqd` and :func:`_run_sparse`.
_SOLVE_STATIC = [
    "states_size",
    "return_eigvec",
    "matvec",
    "maxiter",
    "prefilter",
    "log_level",
    "check_residual",
]


@jax.jit(static_argnames=_SOLVE_STATIC)
def run_sqd(
    hamiltonian: PauliSumXZ,
    states_p: StateList,
    states_size: int,
    return_eigvec: bool,
    matvec: Matvec = Matvec.INDICES,
    maxiter: int = 1000,
    atol: float = 0.0,
    rtol: float | None = None,
    prefilter: tuple[int, int] | None = (32, 2),
    log_level: int = logging.INFO,
    check_residual: bool = False,
) -> SqdResult:
    """JIT-compiled part of the SQD function; returns a :class:`SqdResult`.

    ``converged`` is returned rather than checked, being traced here; :func:`sqd` raises on it once
    concrete. The solver always runs with ``batch_matvec=True``, since every kernel here broadcasts
    over a leading batch axis, bit-identically to separate calls.

    Args:
        matvec: The kernel name, as in :func:`sqd`. Static, bound into the kernel via
            :func:`functools.partial` because ``ground_locg`` splats ``args`` positionally.
        maxiter: Maximum LOBPCG iterations, forwarded to :func:`rqutils.ground_locg.ground_locg`.
            Static, as it is there.
        atol: Absolute bound on the eigen-residual ``||Hv - Ev||``. Validated in :func:`sqd`, which is
            where ``sum|c_k|`` is concrete -- this function is jitted, so it cannot raise.
        rtol: Relative tolerance on ``||Hv|| + |E|``. Convergence is the ``max`` of the two arms, so
            either suffices; see :func:`rqutils.ground_locg.ground_locg`, which both are forwarded to.
        prefilter: Optional ``(degree, cycles)`` Chebyshev prefilter, forwarded by keyword to
            :func:`rqutils.ground_locg.ground_locg`, and static as it is there.
        check_residual: Recompute ``||Hv - Ev||`` and ``||Hv||`` after the solve, into ``residual``
            and ``ax_norm``, from recomputed diagonals (and a fresh search unless ``xsources`` are
            cached). :func:`sqd` turns it on and raises on the result.

    Raises:
        TypeError: If ``matvec`` is not a :class:`Matvec` member.
        ValueError: If ``matvec`` is a sparse kernel (see the module documentation), which only
            :func:`sqd` can build.
    """
    # Static, so this runs once per trace; sqd validates too, and this covers direct poc/ callers.
    _check_matvec(matvec)
    if matvec in _SPARSE_MATVECS:
        raise ValueError(
            f"run_sqd cannot build matvec={matvec!r}: its entry counts are data-dependent, so the "
            "operator is built host-side before the solve. Call sqd(..., matvec=...) instead."
        )
    _check_prefilter(prefilter)
    sharding = None
    if not (mesh := get_abstract_mesh()).empty:
        sharding = PartitionSpec(mesh.axis_names)

    if log_level <= logging.DEBUG:
        jax.debug.print("Uniquifying states (size {})", states_size)

    states_u = uniquify_states(states_p, states_size)

    if matvec != "onthefly":
        if log_level <= logging.DEBUG:
            jax.debug.print("Precomputing xsources")

        xsources = jax.lax.scan(lambda _, x: (None, get_xsource(x, states_u)), None, hamiltonian.x)[
            1
        ]
        if sharding:
            # No sort or search follows, so states_u can now be sharded.
            if log_level <= logging.DEBUG:
                jax.debug.print("Sharding states array")

            states_u = jax.reshard(states_u, sharding)

    if matvec == "tables":
        if log_level <= logging.DEBUG:
            jax.debug.print("Precomputing diagonals")

        diagonals = jax.lax.scan(
            lambda _, v: (None, get_diagonal(v[0], v[1], states_u)),
            None,
            (hamiltonian.z, hamiltonian.c),
        )[1]
        scanned = _pack_scanned(matvec, xsources, diagonals, None)
    else:
        xgroup = xsources if matvec == "indices" else hamiltonian.x
        scanned = _pack_scanned(matvec, xgroup, hamiltonian.z, hamiltonian.c)
    # Bind matvec via partial, not static_argnames: ground_locg splats args positionally, so it
    # would be traced and retrace the kernel every matvec. "tables" reads no states.
    apply = functools.partial(_apply_h_kernel, matvec=matvec)
    args = (scanned, None if matvec == "tables" else states_u)

    def diag0():
        if matvec == "tables":
            return diagonals[0]
        return get_diagonal(hamiltonian.z[0], hamiltonian.c[0], states_u)

    return _solve(
        hamiltonian,
        states_u,
        apply,
        args,
        diag0,
        None if matvec == "onthefly" else xsources,
        sharding,
        return_eigvec=return_eigvec,
        maxiter=maxiter,
        atol=atol,
        rtol=rtol,
        prefilter=prefilter,
        log_level=log_level,
        check_residual=check_residual,
    )


def _solve(
    hamiltonian: PauliSumXZ,
    states_u: StateList,
    apply: Callable[..., jax.Array],
    args: tuple,
    diag0: Callable[[], jax.Array],
    xsources: jax.Array | None,
    sharding: PartitionSpec | None,
    *,
    return_eigvec: bool,
    maxiter: int,
    atol: float,
    rtol: float | None,
    prefilter: tuple[int, int] | None,
    log_level: int,
    check_residual: bool,
) -> SqdResult:
    """The solve every kernel shares, traced inside :func:`run_sqd` or :func:`_run_sparse`.

    ``apply(vec, *args)`` is the operator and ``diag0()`` the identity-X group's diagonal;
    ``xsources`` is ``None`` unless cached source indices exist for the residual check to reuse.
    """
    states_size = states_u.shape[0]

    def vinit_from_min_diag():
        # `.real`: `diagonals` is complex128 for odd-Y strings, and max/argmin reject complex
        # (NOTES.md, "sqd.vinit_from_min_diag: `.real` on both branches").
        diagonal = diag0().real
        # Filler slots get the max so argmin sees only genuine entries; no reshard, unlike
        # _spread_seed, since `diagonal` derives from states_u (verified P(None) and P('x')).
        diagonal = jnp.where(_is_filler(states_u) == 1, jnp.max(diagonal), diagonal)
        imin = jnp.argmin(diagonal)
        # Min-diagonal weight takes the seed's sign, over the spread, never a one-hot; out_sharding
        # is mandatory (NOTES.md, "sqd.run_sqd: the initial-vector guards").
        seed = _spread_seed(states_size, states_u, hamiltonian.c.dtype, sharding)
        # Mask, not `seed.at[imin]` (an all-gather); jnp.sign, not copysign (complex seed); keep the
        # unreachable zero branch (NOTES.md, "sqd.run_sqd: the initial-vector guards").
        sign = jnp.sign(seed)
        direction = jnp.where(sign == 0, 1.0, sign)
        selected = jax.lax.broadcasted_iota(imin.dtype, (states_size,), 0, out_sharding=sharding)
        return seed + jnp.where(selected == imin, direction, jnp.zeros_like(seed))

    def vinit_nodiag():
        # The spread seed, never e_0: a decoupled e_0 is an exact eigenvector at 0, so sqd returned
        # 0.0 with converged=True (NOTES.md, "sqd.run_sqd: the initial-vector guards").
        return _spread_seed(states_size, states_u, hamiltonian.c.dtype, sharding)

    if log_level <= logging.DEBUG:
        jax.debug.print("Generating vinit")

    vinit = jax.lax.cond(jnp.all(hamiltonian.x[0] == 0), vinit_from_min_diag, vinit_nodiag)

    if log_level <= logging.DEBUG:
        jax.debug.print("Starting minimization")

    # sum|c_k| rigorously bounds lambda_max (Pauli strings are unitary; projecting only shrinks it),
    # which a callable cannot supply (NOTES.md, "No matvec-only upper bound on `λ_max` exists").
    filter_runs = prefilter is not None and prefilter[0] > 1 and prefilter[1] > 0
    prefilter_hi = jnp.abs(hamiltonian.c).sum() if filter_runs else None
    eigval, eigvec, _, converged = ground_locg(
        apply,
        vinit,
        args=args,
        maxiter=maxiter,
        atol=atol,
        rtol=rtol,
        prefilter=prefilter,
        prefilter_hi=prefilter_hi,
        log_level=log_level,
        batch_matvec=True,
    )
    result = SqdResult(eigval, converged)
    if check_residual:
        # Diagonals always recomputed, so no cached one vouches for itself; cached xsources are reused,
        # since redoing the J-fold search was ~90% of the check (NOTES.md, "`EigenpairCheckError`").
        ref, xgroup = (
            (Matvec.ONTHEFLY, hamiltonian.x) if xsources is None else (Matvec.INDICES, xsources)
        )
        scanned_ref = _pack_scanned(ref, xgroup, hamiltonian.z, hamiltonian.c)
        ax = _apply_h_kernel(eigvec, scanned_ref, states_u, matvec=ref)
        result = result._replace(
            residual=jnp.linalg.norm(ax - eigval * eigvec), ax_norm=jnp.linalg.norm(ax)
        )
    if return_eigvec:
        if sharding:
            eigvec = jax.reshard(eigvec, PartitionSpec(None))
            states_u = jax.reshard(states_u, PartitionSpec(None))
        subspace_dim = jnp.searchsorted(_is_filler(states_u), 1)
        result = result._replace(eigvec=eigvec, states=states_u, subspace_dim=subspace_dim)
    return result
