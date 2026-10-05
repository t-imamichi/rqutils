"""The body of :mod:`rqutils.sqd`: :func:`sqd`, :func:`hproj` and :class:`EigenpairCheckError`."""

import logging
import time
from collections.abc import Sequence
from numbers import Number
from typing import TYPE_CHECKING, Literal, overload

import jax
import numpy as np
from jax.sharding import get_abstract_mesh
from scipy.sparse import coo_array, csr_array

from rqutils.ground_locg import _check_prefilter, _check_tols
from rqutils.paulis.symplectic import PauliSumXZ
from rqutils.sqd._diagonal import get_diagonal
from rqutils.sqd._solve import (
    _RESIDUAL_SLACK,
    Matvec,
    SqdResult,
    _check_matvec,
    _host_scalar,
    _residual_floor_of,
    run_sqd,
)
from rqutils.sqd._sparse import (
    _group_pairs,
    _mesh_operator,
    _run_sparse,
    _sparse_operator,
    _sparse_residual,
)
from rqutils.sqd._states import (
    _MAX_STATES,
    StateList,
    _check_states_shape,
    _is_lex_sorted,
    _pad_states,
    get_xsource,
    uniquify_states,
)

LOG = logging.getLogger("rqutils.sqd")


if TYPE_CHECKING:
    from qiskit.quantum_info import SparsePauliOp

# Name all three arms here, never via a later `|=`: a `type` statement is static, and lazy, so the
# TYPE_CHECKING-only import is never resolved. Pinned by TestHamiltonianInputIsCheckable.
type HamiltonianInput = PauliSumXZ | tuple[Sequence[str], Sequence[Number]] | SparsePauliOp
type Vector = np.ndarray[tuple[int], np.dtype[np.inexact]]


class EigenpairCheckError(RuntimeError):
    """:func:`sqd` returned ``converged=True`` for a pair that is not an eigenpair.

    Distinct from non-convergence, which raises a plain ``RuntimeError``: that one is fixed by raising
    ``maxiter``, this one never is, so a caller retrying on non-convergence must not catch it.
    """


# Overloads so `eigval, eigvec, basis = sqd(...)` type-checks without narrowing; annotation only,
# sphinx documents the implementation signature below.
@overload
def sqd(
    hamiltonian: HamiltonianInput,
    states: StateList,
    *,
    states_size: int | None = ...,
    return_eigvec: Literal[True] = ...,
    packed: bool = ...,
    matvec: Matvec = ...,
    maxiter: int = ...,
    atol: float = ...,
    rtol: float | None = ...,
    prefilter: tuple[int, int] | None = ...,
) -> tuple[float, Vector, StateList]: ...


@overload
def sqd(
    hamiltonian: HamiltonianInput,
    states: StateList,
    *,
    states_size: int | None = ...,
    return_eigvec: Literal[False],
    packed: bool = ...,
    matvec: Matvec = ...,
    maxiter: int = ...,
    atol: float = ...,
    rtol: float | None = ...,
    prefilter: tuple[int, int] | None = ...,
) -> float: ...


def sqd(
    hamiltonian: HamiltonianInput,
    states: StateList,
    *,
    states_size: int | None = None,
    return_eigvec: bool = True,
    packed: bool = False,
    matvec: Matvec = Matvec.INDICES,
    maxiter: int = 1000,
    atol: float = 0.0,
    rtol: float | None = None,
    prefilter: tuple[int, int] | None = (32, 2),
) -> float | tuple[float, Vector, StateList]:
    r"""Perform a sample-based quantum diagonalization of the Hamiltonian.

    The Hamiltonian can be given in three different forms:

    * A tuple of two lists, where the first list enumerates the Pauli strings :math:`Q` as strs and
      the second contains the coefficients :math:`\alpha`.
    * Qiskit SparsePauliOp
    * PauliSumXZ (From :mod:`rqutils.paulis.symplectic`)

    States must have binary values and can be passed as an array of integers or booleans.

    Internally, the states are bit-packed and represented by :math:`\lceil (n+1)/8 \rceil`
    ``uint8`` s, where the extra bit is placed at position 0 and serves as the indicator for
    spurious (fill-in) entries. ``PauliSumXZ`` reserves the same bit in its signatures
    unconditionally, so the two are aligned by construction.

    ``matvec`` names the matrix-vector kernel: ``Matvec.ONTHEFLY`` caches nothing, ``INDICES`` the
    per-group source indices, ``TABLES`` the source indices and the diagonals, and the
    sparse kernels (see the module documentation) store only the transitions inside the subspace.

    Everything after ``states`` is **keyword-only** (``NOTES.md``, "sqd.sqd: why keyword-only").

    Args:
        hamiltonian: Hamiltonian to be projected and diagonalized.
        states: Binary array of computational basis states to project the Hamiltonian onto. Shape
            (subspace_dim, num_qubits). Entries must be 0 or 1 --
            :meth:`~rqutils.paulis.symplectic.PauliSumXZ.pack_states` raises otherwise, since a
            :math:`\{-1, +1\}` spin encoding would silently collapse the subspace.
        states_size: Pad the states array to this size, so calls with slightly different sizes share
            one compilation. Must be at least ``states.shape[0]`` and at most :math:`2^{31} - 1`.
            Defaults to the next power of two at or above ``states.shape[0]``, suiting growing,
            all-distinct subspace dimensions; pass ``states.shape[0]`` for no padding. The padding
            is not observable in the result.

            On a non-empty mesh it is rounded **up** to a multiple of ``mesh.size``, so the value
            used may exceed the one passed (``NOTES.md``, "sqd.sqd: states_size on a mesh").

            **At large subspace dimensions, size this by hand.** The power-of-two default inflates
            every per-slot term -- states, the solver's vectors and every cache -- by up to 2x, and
            past N ~ 1e5 a finer bucket costs no measurable time (``NOTES.md``, "``states_size``'s
            power-of-two padding").
        return_eigvec: Whether to return the eigenvector (coefficients and unique state bitstrings).
        maxiter: Maximum LOBPCG iterations. **Non-convergence raises** -- see ``Raises``.
        atol: **Absolute** bound on the eigen-residual :math:`\|Hv - Ev\|_2`, forwarded to
            :func:`rqutils.ground_locg.ground_locg`, holding at **every** :math:`N`: ``atol=1e-6``
            means :math:`\|Hv - Ev\| < 10^{-6}`. Default ``0.0``, which disables this arm. ``None``
            is **rejected**; pass ``0.0``.
        rtol: **Relative** tolerance -- a fraction of the operator magnitude,
            :math:`\|r\| < \mathrm{rtol}\,(\|Hv\| + |E|)`. The conventional meaning, as in
            :func:`numpy.allclose`: dimensionless, with the bound
            :math:`\approx 2\,\mathrm{rtol}\|H\|_2` **independent of** :math:`N`. ``None`` (the
            default) uses :math:`4\varepsilon`, 8x the achievable floor; ``0.0`` disables the arm.

            **Convergence is** ``||r|| < max(atol, rtol * scale)`` **-- either arm suffices.** So
            ``atol=x, rtol=0.0`` is absolute-only, ``atol=0.0`` with a non-zero ``rtol`` is
            relative-only, and setting both takes whichever is looser -- usually what a caller wants.
            A caller needing a different bound per subspace size sets ``atol`` per call.

            .. warning::

               **``tol`` is removed, and it had two different meanings.** It was *relative* against an
               :math:`N`-scaled bound up to 2026-08-31, and *absolute* after. ``tol=`` raises
               ``TypeError``. From the absolute form, ``tol=x`` becomes ``atol=x``; from the
               relative form there is **no exact equivalent** (``NOTES.md``, "``atol``/``rtol``: the
               pair is right").

            ``rtol >= 0.5`` is rejected, and so is an ``atol`` below the floor
            :math:`4\,\varepsilon\sum_k|c_k|` **when ``rtol`` is zero** (otherwise the relative arm
            can still fire). See ``Raises`` and :func:`rqutils.ground_locg.residual_floor`.
        packed: Whether ``states`` is already bit-packed, i.e. the output of
            :meth:`~rqutils.paulis.symplectic.PauliSumXZ.pack_states`. Default ``False``, which takes
            the unpacked ``(subspace_dim, num_qubits)`` form and packs it internally. Set it when you
            hold the packed array, to skip an ~8x round trip (``NOTES.md``, "sqd.sqd: ``packed=True``
            skips the pack").

            **It governs the returned basis too**, so a round trip needs no re-pack: ``packed=True``
            returns ``ceil((num_qubits + 1) / 8)``-wide rows, ``packed=False`` unpacks them to
            ``num_qubits``. Before 2026-08-30 the return was unpacked either way (``NOTES.md``,
            "sqd.sqd: one packed flag for both directions").

            It is a declaration, and a wrong one is not always caught: at ``num_qubits == 1`` the two
            widths coincide, so unpacked states with ``packed=True`` silently return a different
            eigenvalue (and a wrong-width basis). Every other qubit count is rejected on width.
        matvec: A :class:`Matvec` member: ``ONTHEFLY``, ``INDICES`` (default) or ``TABLES``, which
            of the source indices and diagonals to cache; or a sparse kernel, which stores the
            in-subspace transitions instead. See the module documentation for the kernels and
            their resource tradeoff.

            A sparse ``matvec`` is built on the host before the solve (logged as its own phase),
            its array shapes rounded up to size classes so the solve recompiles per class rather
            than per subspace. Under a mesh ``Matvec.PAIRS`` runs term-parallel: each device applies a
            share of the X groups to the all-gathered vector, and one reduce-scatter sums them.
            Every process builds the whole operator on the host.
        prefilter: ``(degree, cycles)`` Chebyshev prefilter, forwarded verbatim to
            :func:`rqutils.ground_locg.ground_locg` (see there) and validated by
            :func:`rqutils.ground_locg._check_prefilter`. Static; ``None`` disables it and restores
            the unfiltered graph exactly. ``sqd`` supplies the filter's upper bound itself as
            :math:`\sum_k |c_k|`, so there is no bound to get wrong.

            **``(32, 2)`` is the default**, measured at a 1.49x median end-to-end, below the dense
            ``ground_locg`` figures (``NOTES.md``, "sqd.sqd: the prefilter default"). Pass ``None`` if
            your subspaces do not benefit; A/B rather than assuming, since all figures are
            single-device CPU.

    Returns:
        Calculated ground state energy, or a tuple of energy, ground state vector, and sorted
        uniquified states (if return_eigvec=True). The returned states are the genuine unique rows
        only, never the filler slots, so their count can be below ``states_size``. Their **width
        follows** ``packed``: ``num_qubits`` columns by default, ``ceil((num_qubits + 1) / 8)`` when
        ``packed=True``; a type checker will not catch a caller assuming the wrong width.

        **On a degenerate ground eigenvalue the eigenvector is one arbitrary member of the eigenspace,
        and nothing in the return marks that case.** The eigenvalue is still correct; detection needs
        a second opinion such as ``eigvalsh`` on :func:`hproj` (``NOTES.md``, "sqd.sqd: degenerate
        ground states").

        **Which member is returned is deterministic in the arguments** at a fixed rqutils version: the
        start vector is a fixed hash of the subspace index (:func:`_spread_seed`). It is **not** stable
        across versions; pin the version rather than fingerprinting the vector.

    Raises:
        RuntimeError: If LOBPCG does not converge within ``maxiter``. The unconverged value is a
            finite variational upper bound, indistinguishable from a correct result, so it is never
            returned (``NOTES.md``, "sqd.sqd: non-convergence raises"). Raise ``maxiter``, or loosen
            ``atol`` / ``rtol``, to proceed.
        EigenpairCheckError: If the solve converged but ``||Hv - Ev||``, recomputed after it without
            any cached diagonal, exceeds 10x the convergence bound (or the residual floor).
        ValueError: If ``states_size`` is smaller than ``states.shape[0]``, or exceeds
            :math:`2^{31} - 1`, the int32 ceiling (beyond it the subspace is silently permuted); or
            if ``states`` has the wrong shape (see ``packed``) or non-binary entries.

            If either ``prefilter`` entry is negative, which the filter's own gate would absorb as a
            silent no-op (:func:`rqutils.ground_locg._check_prefilter`).

            If ``atol`` is ``None`` or either tolerance is negative; if **both** are zero; if ``atol``
            is at or above :math:`\sum_k|c_k|`, or below the floor :math:`4\,\varepsilon\sum_k|c_k|`
            **while** ``rtol`` is zero; or if ``rtol`` is at least 0.5, where any vector would report
            convergence.

            If a sparse ``matvec``'s operator reaches :math:`2^{31}` entries.
        TypeError: If ``matvec`` is not a :class:`Matvec` member, a plain string included; if
            ``prefilter`` is neither None nor a ``(degree, cycles)`` pair of ints; or if ``atol`` is
            not a real number, or ``rtol`` neither None nor one.
    """
    hamiltonian, states_p, states_size = _sqd_inputs(
        hamiltonian, states, states_size, packed, matvec, atol, rtol, prefilter
    )
    result = _solve_sqd(
        hamiltonian, states_p, states_size, return_eigvec, matvec, maxiter, atol, rtol, prefilter
    )
    eigval = _checked_eigval(result, hamiltonian, maxiter, atol, rtol)
    if return_eigvec:
        return (eigval, *_eigvec_and_basis(result, hamiltonian, packed))
    return eigval


def _sqd_inputs(
    hamiltonian: HamiltonianInput,
    states: StateList,
    states_size: int | None,
    packed: bool,
    matvec: Matvec,
    atol: float,
    rtol: float | None,
    prefilter: tuple[int, int] | None,
) -> tuple[PauliSumXZ, StateList, int]:
    """Validate :func:`sqd`'s arguments; return the Hamiltonian, padded packed states, ``states_size``."""
    _check_matvec(matvec)
    _check_prefilter(prefilter)
    if states_size is None:
        # Next power of two: growing distinct sizes are the normal SQD pattern, so O(log N) retraces
        # (NOTES.md, "`states_size`'s power-of-two padding").
        states_size = 1 << max((states.shape[0] - 1).bit_length(), 1)
    if states_size < states.shape[0]:
        raise ValueError("states_size smaller than the states array length")
    # A wrapped int32 index is -2147483648, not the -1 absent marker, so nothing downstream catches
    # it (NOTES.md, "The `N ≤ 2^31 - 1` ceiling: why it is enforced where it is").
    if states_size > _MAX_STATES:
        raise ValueError(
            f"states_size {states_size} exceeds the {_MAX_STATES} limit imposed by int32 subspace "
            "indexing; see the scaling-limits section of the module documentation"
        )
    if not isinstance(hamiltonian, PauliSumXZ):
        hamiltonian = PauliSumXZ.from_paulisum(hamiltonian)
    # After the conversion above (it supplies `.c`) and not in run_sqd, which is jitted: sum|c_k| is
    # traced there, and a traced value cannot raise.
    _check_tols(atol, rtol, float(np.abs(hamiltonian.c).sum()), hamiltonian.c.dtype)
    states = _check_states_shape(states, hamiltonian.num_qubits, packed)

    if not (mesh := get_abstract_mesh()).empty and (resid := states_size % mesh.size) != 0:
        LOG.debug("Adjusting states_size to make the array divisible by %d", mesh.size)
        states_size += mesh.size - resid

    # `packed=True` skips the pack: pack_states is not idempotent, and re-expanding costs the caller
    # ~8x (NOTES.md, "sqd.sqd: `packed=True` skips the pack").
    states_p = states if packed else PauliSumXZ.pack_states(states)
    # Pad the input to states_size too: its leading dimension is part of run_sqd's jit cache key
    # (NOTES.md, "sqd.sqd: pad the input states too").
    states_p = _pad_states(states_p, states_size)
    return hamiltonian, states_p, states_size


def _solve_sqd(
    hamiltonian: PauliSumXZ,
    states_p: StateList,
    states_size: int,
    return_eigvec: bool,
    matvec: Matvec,
    maxiter: int,
    atol: float,
    rtol: float | None,
    prefilter: tuple[int, int] | None,
) -> SqdResult:
    """Build a sparse operator if ``matvec`` names one, then solve with the residual check on."""
    LOG.debug("Starting SQD with array size %s", states_size)
    start = time.time()
    tols = {"maxiter": maxiter, "atol": atol, "rtol": rtol, "prefilter": prefilter}
    if matvec is Matvec.PAIRS:
        # run_sqd is jitted and the entry counts are data-dependent, so the operator is built here.
        states_u = uniquify_states(states_p, states_size)
        pairs = _group_pairs(hamiltonian, states_u)  # one search, for the build and the check
        mesh = jax.sharding.get_mesh()
        operator = (
            _sparse_operator(hamiltonian, states_u, pairs)
            if mesh.empty
            else _mesh_operator(hamiltonian, states_u, pairs, mesh)
        )
        LOG.info("Built the %s operator in %f seconds.", matvec, time.time() - start)
        result = _run_sparse(hamiltonian, states_u, operator, states_size, True, **tols)
        del operator  # the check reads none of it
        residual, ax_norm = _sparse_residual(
            hamiltonian, states_u, result.eigval, result.eigvec, pairs
        )
        result = result._replace(residual=residual, ax_norm=ax_norm)
    else:
        result = run_sqd(
            hamiltonian, states_p, states_size, return_eigvec, matvec, check_residual=True, **tols
        )
    # Dispatch is asynchronous: without the wait this logs before the solve has run.
    jax.block_until_ready(result)
    LOG.info("Found ground eigenpair in %f seconds.", time.time() - start)
    return result


def _checked_eigval(
    result: SqdResult, hamiltonian: PauliSumXZ, maxiter: int, atol: float, rtol: float | None
) -> float:
    """The eigenvalue as a host float, after raising on non-convergence or a failed residual check."""
    eigval = float(_host_scalar(result.eigval))
    # Raise here because run_sqd is jitted: an unconverged theta is a finite upper bound, not
    # distinguishable by inspection; markdown/locg.md's I4 hid behind the discarded flag.
    if not bool(_host_scalar(result.converged)):
        raise RuntimeError(
            f"LOBPCG did not converge in maxiter={maxiter} iterations (atol={atol!r}, "
            f"rtol={rtol!r}). The value it "
            f"reached, {eigval!r}, is a variational upper bound rather than the ground energy -- "
            "finite and plausible, which is why this raises instead of returning it. Raise `maxiter` "
            "first: a near-degenerate ground state converges in the eigenVALUE long before the "
            "residual test is satisfied, so this often means the default cap was simply too low "
            "rather than that anything is wrong. Measured on a 37-state subspace with a relative gap "
            "of 5.5e-04, theta was already correct to 4e-16 by iteration 500 while the residual only "
            "crossed the threshold at 1091. Loosening a tolerance is the other lever: `atol` is an "
            f"absolute residual bound whose floor for this operator is "
            f"{_residual_floor_of(hamiltonian):.3e}, so any value above that is reachable in "
            "principle; `rtol` is a fraction of (||Hv|| + |E|) instead. A genuinely "
            "ill-conditioned subspace is the rarer cause."
        )
    residual, ax_norm = (float(_host_scalar(v)) for v in (result.residual, result.ax_norm))
    eps = float(np.finfo(hamiltonian.c.dtype).eps)
    bound = max(atol, (4.0 * eps if rtol is None else rtol) * (ax_norm + abs(eigval)))
    threshold = _RESIDUAL_SLACK * max(bound, _residual_floor_of(hamiltonian))
    LOG.info("Independent eigen-residual %.3e (threshold %.3e).", residual, threshold)
    if not residual <= threshold:  # `not <=` so a NaN residual raises too
        raise EigenpairCheckError(
            f"LOBPCG reported convergence, but the returned pair fails an independent check: "
            f"||Hv - Ev|| recomputed after the solve is {residual:.3e}, above {threshold:.3e} "
            f"({_RESIDUAL_SLACK:g}x the convergence bound {bound:.3e}). The eigenvector is "
            "inconsistent with its eigenvalue, which raising `maxiter` or loosening a tolerance "
            "cannot fix; please report it with the Hamiltonian and states."
        )
    return eigval


def _eigvec_and_basis(
    result: SqdResult, hamiltonian: PauliSumXZ, packed: bool
) -> tuple[np.ndarray, np.ndarray]:
    """The eigenvector and the basis it is expressed in, packed exactly when the input was."""
    eigvec, states_u, subspace_dim = result.eigvec, result.states, result.subspace_dim
    assert eigvec is not None and states_u is not None and subspace_dim is not None  # return_eigvec
    # One `packed` flag governs both directions, so a round trip needs no re-pack (sqd is not a
    # converter). num_qubits, not states.shape[1], which is the packed width on that path.
    basis_states = (
        states_u[:subspace_dim]
        if packed
        else PauliSumXZ.unpack_states(states_u[:subspace_dim], hamiltonian.num_qubits)
    )
    return np.array(eigvec[:subspace_dim]), np.asarray(basis_states)


def hproj(
    hamiltonian: HamiltonianInput, states: StateList, *, unique_states: bool = False
) -> csr_array:
    r"""Return the Hamiltonian projected onto the given subspace.

    The Hamiltonian can be given in three different forms:

    * A tuple of two lists, where the first list enumerates the Pauli strings :math:`Q` as strs and
      the second contains the coefficients :math:`\alpha`.
    * Qiskit SparsePauliOp
    * PauliSumXZ (From :mod:`rqutils.paulis.symplectic`)

    States must have binary values and can be passed as an array of integers or booleans.

    ``unique_states`` is **keyword-only**, for the reason given on :func:`sqd`: as a third positional
    it made ``hproj(ham, states, True)`` read as a plausible but unrelated argument.

    Args:
        hamiltonian: Hamiltonian to be projected and diagonalized.
        states: Binary array of computational basis states to project the Hamiltonian onto. Shape
            (subspace_dim, num_qubits). Entries must be 0 or 1 --
            :meth:`~rqutils.paulis.symplectic.PauliSumXZ.pack_states` raises otherwise.
        unique_states: Whether ``states`` is already uniquified **and lex-sorted**, skipping the
            internal ``np.unique(..., axis=0)``; :func:`get_xsource` binary-searches it, so both are
            required. Validated on this path at 12-14% of the call, raising rather than returning a
            wrong, non-symmetric matrix. Leave it ``False`` to have ``hproj`` sort for you.

    Returns:
        The projected Hamiltonian as a sparse matrix.

    Raises:
        ValueError: If a mesh is set -- ``hproj`` is single-device only; if ``unique_states=True`` and
            ``states`` is not strictly increasing in lexicographic order (unsorted, or containing
            duplicate rows); if ``states`` is not 2-D with ``num_qubits`` columns, or has non-binary
            entries; or if the subspace exceeds :math:`2^{31} - 1` states, the ceiling imposed by
            the int32 indices :func:`get_xsource` returns.
    """
    if not isinstance(hamiltonian, PauliSumXZ):
        hamiltonian = PauliSumXZ.from_paulisum(hamiltonian)
    states = _check_states_shape(states, hamiltonian.num_qubits)
    # Rejected rather than supported: the return is a host scipy matrix, so a mesh buys nothing. Here
    # rather than below for the same reason as the ceiling check: O(1), so it precedes the O(N) sort.
    if not get_abstract_mesh().empty:
        raise ValueError(
            "hproj does not support sharding: it builds a host-side scipy matrix, so a mesh buys "
            "nothing. Call it outside the mesh context, or use sqd() for a sharded solve."
        )
    # sqd()'s int32 ceiling, checked before the O(N) scan and np.unique below
    # (NOTES.md, "The `N ≤ 2^31 - 1` ceiling: why it is enforced where it is").
    if states.shape[0] > _MAX_STATES:
        raise ValueError(
            f"subspace of {states.shape[0]} states exceeds the {_MAX_STATES} limit imposed by int32 "
            "subspace indexing; see the scaling-limits section of the module documentation"
        )
    if not unique_states:
        states = np.unique(states, axis=0)
    else:
        # get_xsource binary-searches `states`, so unsorted rows gave a wrong non-symmetric matrix;
        # checked at 12-14% of hproj, on this opt-in path only.
        if not _is_lex_sorted(states):
            raise ValueError(
                "unique_states=True requires `states` to be uniquified and lex-sorted, but the "
                "rows given are not strictly increasing. get_xsource binary-searches into this "
                "array, so an unsorted subspace yields a wrong (non-symmetric) projection rather "
                "than an error deeper in. Pass np.unique(states, axis=0), or leave "
                "unique_states=False to have hproj do it."
            )
    states_p = PauliSumXZ.pack_states(states)

    columns, elements = _hproj_cols_elems(hamiltonian, states_p)
    valid = columns != -1
    rows = np.tile(np.arange(states.shape[0])[None, :], (columns.shape[0], 1))[valid]
    data = np.array(elements[valid])
    cols = np.array(columns[valid])
    # shape= is mandatory: scipy otherwise infers the extent and silently drops uncoupled trailing
    # states (NOTES.md, "sqd.hproj: `shape=` is mandatory").
    dim = states.shape[0]
    return csr_array(coo_array((data, (rows, cols)), shape=(dim, dim)))


@jax.jit
def _hproj_cols_elems(hamiltonian: PauliSumXZ, states_p: StateList) -> tuple[jax.Array, jax.Array]:
    """Scan every Pauli term, returning its column indices and matrix elements.

    Module scope is load-bearing: ``jax.jit`` keys its cache on the function *object*, so a closure
    inside ``hproj`` retraced and recompiled on every call. For the same reason ``states_p`` must stay
    an argument, not a capture (``NOTES.md``, "sqd._hproj_cols_elems: module scope").
    ``PauliSumXZ`` is a registered dataclass pytree, so it passes through as a jit argument.
    """

    def get_from_one(_, ham):
        # Read by name: x and z are same-dtype integer arrays, so swapped positions type-check and
        # silently compute with X and Z exchanged.
        columns = get_xsource(ham.x, states_p)
        diagonals = get_diagonal(ham.z, ham.c, states_p)
        return None, (columns, diagonals)

    return jax.lax.scan(get_from_one, None, hamiltonian.arrays)[1]
