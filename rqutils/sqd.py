r"""
======================================================================
Sample-based quantum diagonalization of general Pauli-sum Hamiltonians
======================================================================

.. currentmodule:: rqutils.sqd

Overview
========

SQD is an algorithm for finding an approximate ground eigenpair of a very large (computationally
intractable) Hamiltonian by projecting it onto a subspace. In our case, the Hamiltonian is expressed
as a linear combination of Pauli strings, and the subspace is identified from a (possibly redundant)
set of bitstrings (states).

Typically, the projected Hamiltonian is itself still too large to be stored in memory as a matrix,
requiring some matrix-free method to solve the eigenvalue problem. The technical challenge is then
threefold:

- Extracting the set :math:`S` of unique bitstrings from the input list.
- Computing (on the fly) the matrix elements :math:`\langle j | Q | k \rangle` for all
  :math:`j, k \in S` for each term :math:`Q` of the Hamiltonian.
- Solving the eigenvalue problem.

The first point will have to be done with ``np.unique`` or an equivalent accelerated function. For
the last point, we can use the `ground_locg` solver provided in this package, which takes a
matrix-vector application function and an initial-guess vector as inputs. The central task here is
thus providing the matvec function that covers the second point.

Usage examples can be found at
`examples/sqd.py <https://github.com/UTokyo-ICEPP/rqutils/tree/main/examples/sqd.py>`__.

Algorithm
=========

The algorithm takes advantage of the symplectic representation of Pauli strings. A term in the
Hamiltonian is a product of a real coefficient :math:`\alpha` and a Pauli string :math:`Q`. In the
symplectic representation,

.. math::

    \alpha Q = \alpha (-i)^{xz} \left(Z^{z_{n-1}} \otimes \cdots Z^{z_{0}}\right)
                                  \left(X^{x_{n-1}} \otimes \cdots X^{x_{0}}\right)

where :math:`x` (X signature) and :math:`z` (Z signature) are binary vectors of length :math:`n`
(number of qubits) and :math:`xz` represents their inner product. For a given bitstring
:math:`s = [s_{n-1}, \dots, s_{0}]`, the X signature of the Hamiltonian term determines the
existence and location of the matrix element, and the Z signature gives the sign.

In the preparation stage of the algorithm, the terms of the input Hamiltonian are grouped by the X
signature (for example, XIZ and YZI will belong to the same group). For each X signature, there will
be multiple Z signatures and the corresponding phased coefficients (:math:`\alpha (-i)^{xz}` above).
The X and Z signatures are bit-packed into arrays of 8-bit integers.

The input states are then similarly bit-packed, to allow bitwise operations between the states and
the X/Z signatures, and sorted. The resulting array :math:`S = [s^{0}, \dots, s^{N-1}]` is the basis
on which the Hamiltonian is projected. We then define the initial vector of length :math:`N` as the
input to the LOBPCG function.

That initial vector is a deterministic pseudo-random spread over the subspace, with the
minimum-diagonal state weighted heavily on top of it when the Hamiltonian has a diagonal part. It is
deliberately *not* a one-hot vector: LOBPCG requires a non-vanishing overlap with the ground state,
and a one-hot cannot leave the connected component of the projected Hamiltonian that contains it, so
a subspace whose Hamiltonian splits into disconnected blocks would silently yield that block's
minimum instead of the global one. See :func:`_spread_seed`.

Let :math:`J` be the number of distinct X signatures in the Hamiltonian, and :math:`K^{(j)}` be the
number of Z signatures and coefficients associated with the :math:`j` th X signature. The
matrix-vector operation to be passed to the solver acts on the length-:math:`N` vector :math:`v` of
coefficients as

.. math::

    v' = \sum_{j=1}^{J} \left( \sum_{k=1}^{K^{(j)}} \alpha^{(j,k)} (-i)^{x^{(j)}z^{(j,k)}}
                                            D[z^{(j,k)}] \right) \circ B[x^{(j)}](v).

The operation

.. math::

    w = B[x](v)

consists of the following steps:

#. Compute the source state :math:`t^{i} \leftarrow s^{i} \oplus x` of :math:`w^{i}`.
#. If a source index :math:`j^{i}` exists such that :math:`s^{j^{i}} = t^{i}`,
   :math:`w^{i} \leftarrow v^{j^{i}}`. Otherwise :math:`w^{i} \leftarrow 0`.

The operation :math:`D[z]` is a diagonal operation that applies a sign factor to each vector entry:

.. math::

    D[z](w^{i}) = (-1)^{zs^{i}} w^{i}.

Caching
-------

In the expressions above, source indices :math:`[j^{i}]` and sign factors :math:`[(-1)^{zs^{i}}]` do
not depend on the coefficient vector :math:`v` and can be determined once :math:`S` is given. In
fact, the composition of the sign factors with the coefficients

.. math::

    C^{(j)} = \sum_{k=1}^{K^{(j)}} \alpha^{(j,k)} (-i)^{x^{(j)}z^{(j,k)}} [(-1)^{z^{(j,k)}s^{i}}]

is entirely static in the same way. It is therefore possible to consider caching these vectors and
reusing them in the repeated call to the matrix-vector function. There is however a tradeoff between
the compute time and memory footprint, as is always the case with caching.

Concretely, caching the source indices :math:`[j^{i}]` requires :math:`4 J N` bytes of memory,
assuming :math:`N \leq 2^{31}` (which is actually the hard limit set by other constraints; see the
next section) and therefore 32-bit (4-byte) integers are used for vector indexing. Caching the
composed diagonals :math:`C^{(j)}` takes :math:`8 J N` or :math:`16 J N` bytes, depending on whether
the vector is real or complex (i.e., if there are terms in the Hamiltonian with odd numbers of Ys).
Caching both also frees :math:`S`, which is no longer read, so at small :math:`n` the full cache can
be the cheaper option in memory too.

``matvec`` names the kernel by what it stores:

- ``"onthefly"``: nothing; the sources are searched and the diagonals computed per matvec.
- ``"indices"`` (the default): the source indices :math:`[j^{i}]` of every X group.
- ``"tables"``: the source indices and the composed diagonals :math:`C^{(j)}`.
- ``"pairs"``: each transition :math:`(a, b)`, :math:`a < b`, of a non-identity X group once, with
  :math:`d = C^{(j)}_a`, applied as :math:`v'_a \mathrel{+}= d v_b` and
  :math:`v'_b \mathrel{+}= \bar{d} v_a` -- exact, since :math:`H_{ba} = \overline{H_{ab}}` per X signature.
- ``"csr"``: both directions of every transition, sorted by target, with ``float64`` factors for
  the all-real groups and ``complex128`` for the rest.

The other dense storage combinations are dominated on both memory and time (``NOTES.md``). The last
two store only the transitions that land inside the subspace, where the source indices mostly hold
the ``-1`` absent marker; :func:`sqd` builds them host-side before the solve, they are single-device
for now, and ``poc/sparse-pairs.md`` has the measurements.

**The source-index setup dominates the solve, so this is not a symmetric memory-for-speed dial.**
Weighted by call count, the :math:`J`-fold :func:`get_xsource` precompute measured **66-97%** of an
entire solve -- 97.5% at 10 iterations, 66.4% at 200 (3064 ms of setup against 8.35 ms per matvec
iteration, N=200k, J=50; see ``markdown/scaling-pocs.md``). Turning source-index caching *off* pays
that cost once per matvec rather than once per solve: measured end-to-end at N=3k, n=12, J=23,
``"onthefly"`` is 7.2x slower than ``"indices"``, returning the same energy. Prefer ``"indices"`` or
``"tables"`` unless the memory genuinely will not fit.

Distributed arrays and scaling limits
=====================================

When the SQD function is called within a context where the global mesh is set via
``jax.set_mesh(mesh)``, the state vector is distributed (sharded) among the devices in the mesh,
and accordingly all arrays with an axis with size :math:`N` follow the same sharding. Even the most
aggressive caching strategy described above will be possible this way.

However, there is a limit to scaling in :math:`N` (SQD subspace dimension) imposed by the need to
sort the states list during the initial uniquification. That sort must take place within a single
device, with at most :math:`2^{32}` elements involved, so it caps the achievable :math:`N` at
:math:`2^{31}`. Source index identification no longer contributes: it is a binary search into the
already-sorted state list rather than a sort of a stacked :math:`2N` array, which is why the cap
comes from :func:`uniquify_states` alone -- see :func:`get_xsource` for that change and what it was
measured to be worth. A comparable limit is set by the GPU memory, which is at most O(100)GB per
device as of mid-2026.

When the source indices are cached, the state list :math:`S` is also sharded once they are computed.

SQD API
=======

.. autofunction:: sqd
.. autoexception:: EigenpairCheckError
.. autofunction:: hproj

States are packed with :meth:`~rqutils.paulis.symplectic.PauliSumXZ.pack_states`, which inserts the
pad bit that aligns them with the Hamiltonian's signatures, and recovered with
:meth:`~rqutils.paulis.symplectic.PauliSumXZ.unpack_states`.
"""

import functools
import logging
import time
from collections.abc import Callable, Sequence
from numbers import Number
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, overload

import jax
import jax.core
import jax.numpy as jnp
import numpy as np
from jax.experimental.multihost_utils import process_allgather
from jax.sharding import PartitionSpec, get_abstract_mesh
from numpy.typing import DTypeLike, NDArray
from scipy.sparse import coo_array, csr_array

from rqutils.ground_locg import (
    _check_prefilter,
    _check_tols,
    ground_locg,
    residual_floor,
)
from rqutils.paulis.symplectic import PauliSumXZ

LOG = logging.getLogger(__name__)
# Subspace positions are int32 throughout, so the size is capped and enforced to raise on overflow
# (NOTES.md, "The `N ≤ 2^31 - 1` ceiling: why it is enforced where it is").
_MAX_STATES = 2**31 - 1

if TYPE_CHECKING:
    from qiskit.quantum_info import SparsePauliOp

# Name all three arms here, never via a later `|=`: a `type` statement is static, and lazy, so the
# TYPE_CHECKING-only import is never resolved. Pinned by TestHamiltonianInputIsCheckable.
type HamiltonianInput = PauliSumXZ | tuple[Sequence[str], Sequence[Number]] | SparsePauliOp
type Vector = np.ndarray[tuple[int], np.dtype[np.inexact]]
type StateList = np.ndarray[tuple[int, int], np.dtype[np.uint8]]


# Dtype kind per `apply_h` keyword: `u` packed bytes, `i` positions (-1 = absent), `fc` float or
# complex. It separates arrays whose shapes collide, (2, 2) at n=15 with 2 states.
_ARRAY_ROLE_KINDS = {
    "xsignatures": ("u", "packed X signatures (uint8, from PauliSumXZ)"),
    "xsources": ("i", "X source indices (int32, from get_xsource; -1 marks absent)"),
    "zsignatures": ("u", "packed Z signatures (uint8, from PauliSumXZ)"),
    "diagonals": ("fc", "precomputed diagonals (float or complex, from get_diagonal)"),
}


def _check_array_role(name: str, array: Any) -> None:
    """Raise if the array under ``name`` has the wrong dtype kind for that role.

    Closes part of the misnaming residue :func:`apply_h` documents. Deliberately *not* a shape check:
    at ``n = 15`` (2 bytes) with a 2-state subspace, X signatures and X sources are both exactly
    ``(2, 2)``, so a shape assertion sails through the mispairing it exists to trip -- which is why
    the positional form was deleted rather than asserted.

    Dtype separates them structurally at exactly that point. Packed signatures are ``uint8``
    (:func:`numpy.packbits` output); source indices are ``int32`` positions carrying ``-1`` as the
    absent marker, and a ``uint8`` cannot hold ``-1``. Diagonals are float or complex.

    What it still does **not** reach: swapping two arrays of the same kind, e.g. ``xsignatures`` for
    ``zsignatures`` (both ``uint8``). That residue stays open.

    Args:
        name: The keyword the caller used.
        array: The array they passed under it.

    Raises:
        ValueError: If the dtype kind does not match the role.
    """
    expected, described = _ARRAY_ROLE_KINDS[name]
    # `.dtype` directly, not `np.asarray(array).dtype`: both numpy and jax arrays expose it, and the
    # asarray round-trip measured 13x slower (1.08 us against 0.083 us) for the same information.
    dtype = array.dtype
    if dtype.kind not in expected:
        raise ValueError(
            f"apply_h: {name}= expects {described}, but got dtype {dtype}. Check the keyword names "
            "against the arrays -- a shape check cannot catch this (X signatures and X sources are "
            "both (2, 2) at n=15 with 2 states), so the dtype is what distinguishes them."
        )


type Matvec = Literal["onthefly", "indices", "tables", "pairs", "csr"]
_MATVECS = ("onthefly", "indices", "tables", "pairs", "csr")


def _residual_floor_of(hamiltonian: PauliSumXZ) -> float:
    """The achievable eigen-residual floor for this Hamiltonian, for guards and error messages."""
    return residual_floor(float(np.abs(hamiltonian.c).sum()), hamiltonian.c.dtype)


def _check_matvec(matvec: Any) -> None:
    """Raise unless ``matvec`` is one of the kernel names in ``_MATVECS``.

    Every branch on ``matvec`` is an equality test with an implicit ``else``, so an unvalidated value
    would be absorbed into some kernel rather than reported.

    Args:
        matvec: The caller's value, unvalidated.

    Raises:
        TypeError: If it is not a ``str`` (the removed ``cache_level`` tuple included).
        ValueError: If it is not one of those names.
    """
    if not isinstance(matvec, str):
        raise TypeError(f"`matvec` must be a str, one of {_MATVECS}; got {matvec!r}")
    if matvec not in _MATVECS:
        raise ValueError(f"`matvec` is {matvec!r}, but must be one of {_MATVECS}")


def _check_zsignatures_rank(zsignatures: Any) -> None:
    """Raise unless ``zsignatures`` is one X group's 2-D ``(num_zterms, num_bytes)`` array.

    :func:`get_diagonal` is public and indexes the array's leading axis, so handed a 1-D array it
    reads *scalars* and silently returns a wrongly shaped result rather than raising. Measured:
    ``(4,)`` of ``[2., 2., 2., 2.]``, a plausible finite diagonal. Rank is static under ``jax.jit``,
    so unlike ``get_xsource``'s lex-sortedness precondition this one is checkable here.

    Args:
        zsignatures: The caller's Z-signature array.

    Raises:
        ValueError: If it is not 2-D.
    """
    if np.ndim(zsignatures) != 2:
        raise ValueError(
            "`zsignatures` must be 2-D with shape (num_zterms, num_bytes) -- one X group's Z "
            f"signatures -- but has rank {np.ndim(zsignatures)}. A 1-D array would be scanned as "
            "scalars and silently return the wrong shape. Pass `hamiltonian.z[igroup]`, not "
            "`hamiltonian.z[igroup][iterm]`."
        )


def _check_states_shape(states: Any, num_qubits: int, packed: bool = False) -> StateList:
    """Raise unless ``states`` is ``(subspace_dim, num_qubits)``, or packed-width when ``packed``.

    One ``O(1)`` look at a shape, shared by :func:`sqd` and :func:`hproj` so the two cannot drift.
    It closes three distinct mistakes that all used to produce a plausible finite answer:

    * **Re-feeding packed states.** :meth:`PauliSumXZ.pack_states` is not idempotent, and ``sqd``
      takes *unpacked* states while the natural intermediate a caller keeps -- from
      :func:`uniquify_states`, or from ``pack_states`` called directly -- is *packed*. Feeding that
      back re-packs it: ``astype(uint8)`` is a no-op and ``packbits`` then reads each byte as one bit
      via nonzero-to-1, so the subspace silently changes. Realistic loop: run ``sqd``, do
      configuration recovery, run ``sqd`` again.
    * **A transposed array**, ``(num_qubits, subspace_dim)``.
    * **A mismatched Hamiltonian**, right shape family and wrong qubit count.

    Note it does not close *every* re-feed on its own: at ``num_qubits <= 7`` a packed row is one
    byte wide, so a 1-qubit Hamiltonian would accept it -- the shape genuinely matches. What catches
    that is :meth:`PauliSumXZ.pack_states`' binary check, since packed bytes exceed 1.

    ``packed=True`` expects ``ceil((num_qubits + 1) / 8)`` columns instead, for a caller handing over
    :meth:`PauliSumXZ.pack_states`' own output. It is a **declaration, not a shape inference**, and
    that is not merely stylistic: at ``num_qubits == 1`` both widths are 1, so the shapes are
    genuinely undecidable and the flag is the only discriminator. Measured there --
    ``sqd(unpacked, packed=True)`` returns ``+1.0`` where the truth is ``-1.0``, silently, because an
    unpacked ``[[0], [1]]`` is a legal *packed* array meaning something else. The reverse direction
    (packed array, flag omitted) is caught by :meth:`PauliSumXZ.pack_states`' binary check, since
    packed bytes exceed 1. So the flag closes the one direction nothing else can.

    Coerces with :func:`numpy.asarray` and returns the result, rather than only inspecting: both
    entry points document ``states`` as passable "as an array of integers or booleans", and
    :func:`hproj` accepted a list of lists. Reading ``.ndim`` off the raw argument would narrow that
    to arrays only, and would do so with an ``AttributeError`` rather than this function's documented
    ``ValueError``. Coercing here also makes the two entry points agree, where ``sqd`` previously
    required an array and ``hproj`` did not.

    Args:
        states: The caller's states, as anything :func:`numpy.asarray` accepts.
        num_qubits: The Hamiltonian's qubit count.
        packed: Whether ``states`` is already bit-packed, so the expected width is
            ``ceil((num_qubits + 1) / 8)`` rather than ``num_qubits``.

    Returns:
        ``states`` as an array, for the caller to use in place of its argument.

    Raises:
        ValueError: If ``states`` is not 2-D, or its second axis is not the expected width.
    """
    states = np.asarray(states)
    if states.ndim != 2:
        raise ValueError(
            f"`states` must be 2-D with shape (subspace_dim, num_qubits), got shape {states.shape}"
        )
    if packed:
        width = -(-(num_qubits + 1) // 8)
        if states.shape[1] != width:
            raise ValueError(
                f"`states` has {states.shape[1]} columns but `packed=True` with "
                f"{num_qubits} qubits expects {width} (= ceil(({num_qubits} + 1) / 8)). Pass the "
                "output of `PauliSumXZ.pack_states`, or drop `packed=True` for unpacked states."
            )
        if states.dtype != np.uint8:
            raise ValueError(
                f"packed `states` must be uint8, got {states.dtype}. `PauliSumXZ.pack_states` "
                "returns uint8; a wider dtype means the array is not its output."
            )
        return states
    if states.shape[1] != num_qubits:
        raise ValueError(
            f"`states` has {states.shape[1]} columns but the Hamiltonian has {num_qubits} qubits; "
            "`states` must be (subspace_dim, num_qubits) and *unpacked*. Note "
            "`PauliSumXZ.pack_states` is not idempotent, so a packed array kept from a previous call "
            "(or from `uniquify_states`) cannot be fed back in -- it would re-pack into a different "
            "subspace. Also check for a transposed array or a mismatched Hamiltonian."
        )
    return states


#: Headroom of :class:`EigenpairCheckError` over the convergence bound: converged solves measure at
#: most 0.96 of it, so this only fires on a pair that is wrong rather than marginal (``NOTES.md``).
_RESIDUAL_SLACK = 10.0


class EigenpairCheckError(RuntimeError):
    """:func:`sqd` returned ``converged=True`` for a pair that is not an eigenpair.

    Distinct from non-convergence, which raises a plain ``RuntimeError``: that one is fixed by raising
    ``maxiter``, this one never is, so a caller retrying on non-convergence must not catch it.
    """


def _host_scalar(value: jax.Array | float | bool) -> jax.Array | float | bool:
    """Return a replicated scalar in a form the host can read, on any process topology.

    ``float()``/``bool()`` on a ``jax.Array`` fetch its value, which **raises across processes**:
    "Fetching value for `jax.Array` that spans non-addressable (non process local) devices". A
    reduction over a partitioned vector produces exactly that -- a rank-0 array whose sharding still
    names the whole mesh -- so on a multi-process mesh ``sqd`` could not return its own eigenvalue,
    nor read its own convergence flag, on the *default* ``return_eigvec=False`` path.

    ``jax.reshard`` does not help: the spec is already ``P()``, so the problem is addressability rather
    than layout. Multi-process therefore goes through
    :func:`jax.experimental.multihost_utils.process_allgather`, which puts the value on every process
    irrespective of the mesh in context or the mesh the array came from -- both of which broke
    hand-rolled ``jit(out_shardings=...)`` attempts (see the comments in the body).

    **The branch is on ``jax.process_count()``, deliberately, not on this rank's view of the array.**
    Whether a given rank can see the value is a per-rank property -- measured on 4 nodes, two ranks read
    it successfully while two raised from the same call -- and branching on that would send some ranks
    into the collective and let others return early, hanging the job at the barrier. ``process_count()``
    is identical on every rank, so every rank takes the same path. Single process short-circuits, since
    it addresses every device and no communication is required.

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
    matvec: Matvec = "indices",
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

    ``matvec`` names the matrix-vector kernel: ``"onthefly"`` caches nothing, ``"indices"`` caches
    the per-group source indices, ``"tables"`` caches the source indices and the diagonals, and
    ``"pairs"``/``"csr"`` store only the transitions inside the subspace.

    Everything after ``states`` is **keyword-only**. It used to be positional-or-keyword, which made
    ``sqd(ham, states, True)`` a valid ``states_size`` of 1 (``True == 1``) rather than the
    ``return_eigvec`` the caller meant -- no error, the array pinned to one slot. The three
    parameters are semantically unrelated, so no reading of a bare positional was worth preserving.

    Args:
        hamiltonian: Hamiltonian to be projected and diagonalized.
        states: Binary array of computational basis states to project the Hamiltonian onto. Shape
            (subspace_dim, num_qubits). Entries must be 0 or 1 --
            :meth:`~rqutils.paulis.symplectic.PauliSumXZ.pack_states` raises otherwise, since a
            :math:`\{-1, +1\}` spin encoding would silently collapse the subspace.
        states_size: Fix the size of the states array used in computation to the specified value so
            that compilation is not triggered at each call with slightly different array sizes. Must
            be at least ``states.shape[0]``. Defaults to the next power of two at or above
            ``states.shape[0]``, which is the rule a caller feeding growing, all-distinct subspace
            dimensions needs (the normal SQD access pattern -- one dimension per Krylov rung plus
            one per configuration-recovery round, so no two calls share a size). Pass
            ``states.shape[0]`` for no padding at all. On a non-empty mesh this is rounded **up**
            to the next multiple of ``mesh.size``, so the value used internally may exceed the one
            passed; that widens the coalescing this argument exists for rather than defeating it
            (measured on a 4-device mesh, ``states_size`` 33 through 36 all share one compiled
            kernel -- 707 ms to compile, then ~16 ms -- while 37 rounds to 40 and compiles afresh).
            The padding is not observable in the result: filler slots are excluded from the
            projection and trimmed from the returned basis.

            **At large subspace dimensions, size this by hand.** The power-of-two default inflates
            *every* per-slot term at once -- states, the solver's vectors and every cache -- by up to
            2x, and how much depends on where ``states.shape[0]`` falls: 4.9% at N=1M, 39.8% at
            N=24M, 82.0% at N=144k. The default is right at small dimensions, where a compilation is
            most of a solve (97% of a cold solve at N=2000), and a finer bucket there measured **57%
            slower** over a growing sweep. But compile time is roughly fixed while the waste is a
            fraction, so past N ~ 1e5 the trade inverts: rounding to a multiple of the largest power
            of two at or below ``N/8`` measured **one extra compilation and no measurable time**
            (5.09 s against 5.10 s over five growing dimensions at n=20) while cutting waste from
            62.4% to 3.3%. At N=24M with the since-removed sources-searched, diagonals-cached kernel that was **7.7 GB**.
            See ``NOTES.md``, "``states_size``'s power-of-two padding".
        return_eigvec: Whether to return the eigenvector (coefficients and unique state bitstrings).
        maxiter: Maximum LOBPCG iterations. **Non-convergence now raises** rather than returning
            the iteration cap's best guess -- see ``Raises``.
        atol: **Absolute** bound on the eigen-residual :math:`\|Hv - Ev\|_2`, forwarded to
            :func:`rqutils.ground_locg.ground_locg`. Default ``0.0``, which disables this arm.

            Set it when a downstream consumer has a fixed residual requirement: ``atol=1e-6`` means
            :math:`\|Hv - Ev\| < 10^{-6}` at **every** :math:`N`, which no relative tolerance can
            express. ``None`` is **rejected** -- pass ``0.0`` to disable the arm, since a *derived*
            absolute bound is the unintuitive construct this pair replaced.
        rtol: **Relative** tolerance -- a fraction of the operator magnitude,
            :math:`\|r\| < \mathrm{rtol}\,(\|Hv\| + |E|)`. The conventional meaning, as in
            :func:`numpy.allclose`: dimensionless, with the bracket supplying the units. Since
            :math:`\|Hv\| \approx |E|` at convergence the bound is :math:`\approx 2\,\mathrm{rtol}\|H\|_2`
            -- **independent of** :math:`N`, so one value means the same thing at every subspace size.

            ``None`` (the default) uses :math:`4\varepsilon`, targeting :math:`8\varepsilon\|H\|_2`, 8x
            the measured achievable floor. Pass ``0.0`` to disable the arm.

            **Convergence is** ``||r|| < max(atol, rtol * scale)`` **-- either arm suffices.** So
            ``atol=x, rtol=0.0`` is absolute-only, ``atol=0.0`` with a non-zero ``rtol`` is
            relative-only, and setting both takes whichever is looser -- usually what a caller wants.

            .. warning::

               **``tol`` is removed, and it had two different meanings.** It was *relative* against an
               :math:`N`-scaled bound up to 2026-08-31, and *absolute* after. There is no alias:
               ``tol=`` raises ``TypeError`` rather than silently resolving to one of the pair.
               From the absolute form, ``tol=x`` becomes ``atol=x``. From the relative form there is
               **no exact equivalent** -- see below.

            The pre-2026-08-31 relative form multiplied the scale by a further :math:`N \cdot 10`, and
            that is **not** reproduced. Neither factor was a rounding budget (the floor has no :math:`N`
            term, measured), and folding a dimension count into a "relative" tolerance made it
            unpredictable and, at large :math:`N`, dangerous -- ``rtol=1e-8`` at :math:`N = 2^{20}` gave
            a bound of 4.2 against :math:`\|H\| = 20`, so the solve converged on the first iterate and
            returned a wrong answer with ``converged=True``. **A caller needing a different bound per
            subspace size sets ``atol`` per call.** ``rtol >= 0.5`` is rejected, being the range where
            the bound reaches :math:`\|H\|`.

            An ``atol`` below the achievable floor :math:`4\,\varepsilon\sum_k|c_k|` is rejected **only
            when ``rtol`` is zero**, since otherwise the relative arm can still fire. See ``Raises`` and
            :func:`rqutils.ground_locg.residual_floor`.
        packed: Whether ``states`` is already bit-packed, i.e. the output of
            :meth:`~rqutils.paulis.symplectic.PauliSumXZ.pack_states`. Default ``False``, which takes
            the unpacked ``(subspace_dim, num_qubits)`` form and packs it internally.

            Set it when you already hold the packed array, to skip an 8x round trip: unpacked states
            are one byte per qubit against ``ceil((num_qubits + 1) / 8)`` bytes packed -- 2.40 GB
            against 0.31 GB at ``num_qubits=100``, ``subspace_dim=24M`` -- and both arrays are live
            at once during the pack, so the transient peak is their sum. The packed form is what the
            solver has always used internally.

            **It governs the returned basis too**, so a round trip needs no re-pack: ``packed=True``
            returns the ``ceil((num_qubits + 1) / 8)``-wide rows the solver searched, ``packed=False``
            unpacks them to ``num_qubits``. **This is a behavioural change** -- before 2026-08-30 the
            return was unpacked either way, so a caller passing ``packed=True`` and comparing the
            result against an unpacked array now gets a shape mismatch. That comparison fails loudly
            (``np.array_equal`` is ``False`` on differing shapes) rather than silently, which is why
            the flag governs both directions instead of a second ``return_packed``: two flags make
            four combinations, two of which are format conversions, and ``sqd`` is not a conversion
            utility -- :meth:`~rqutils.paulis.symplectic.PauliSumXZ.pack_states` and
            :meth:`~rqutils.paulis.symplectic.PauliSumXZ.unpack_states` are.

            It is a declaration, and a wrong one is not always caught. At ``num_qubits == 1`` the two
            widths coincide, so passing unpacked states with ``packed=True`` silently returns a
            different eigenvalue; every other qubit count is rejected on width. Pass the flag only for
            an array that came from ``pack_states``. Note the returned width is *also* wrong in that
            case, which gives a second chance to notice.
        matvec: ``"onthefly"``, ``"indices"`` (default) or ``"tables"``: which of the source indices
            and diagonals to cache; or ``"pairs"``/``"csr"``, which store the in-subspace transitions
            instead. See the module documentation for the resource tradeoff involved.

            ``"pairs"`` and ``"csr"`` are built on the host before the solve (logged as their own
            phase), with the entry count rounded up to a size class so the solve recompiles per
            class rather than per subspace. Single-device only for now.
        prefilter: ``(degree, cycles)`` Chebyshev prefilter, forwarded verbatim to
            :func:`rqutils.ground_locg.ground_locg` -- see its docstring for the semantics, the cost
            and the knob-choosing guidance. Validated by
            :func:`rqutils.ground_locg._check_prefilter`, which rejects the malformed values the
            filter's own gate would absorb as a silent no-op. Static, so the branch resolves at trace
            time; ``None`` disables it and restores the pre-2026-08-28 graph exactly.

            **``(32, 2)`` is the default**, measured end-to-end through ``sqd`` at a **1.49x median**
            wall-clock reduction (range 1.15-1.70x, 6 sampled XXZ subspaces at n=14-18, dim
            978-3982, every arm correct to <1e-9 against ``eigsh(tol=0)``). ``sqd`` supplies the
            filter's required upper bound itself as :math:`\sum_k |c_k|`, so the option costs the
            caller nothing to use and there is no bound to get wrong.

            That 1.49x is **below** the 1.88x median ``markdown/locg-chebyshev-prefilter.md`` measured on
            dense ``ground_locg``, and the gap is the point: the filter spends
            ``cycles * (degree + 1)`` matvecs up front, and :func:`apply_h`'s sparse gather-heavy
            kernel is cheap enough that those cost proportionally more here. Counting *iterations*
            instead would report 5.02x -- the wrong unit for a caller. Pass ``None`` if your subspaces
            do not benefit; A/B rather than assuming, since all figures are single-device CPU.

    Returns:
        Calculated ground state energy, or a tuple of energy, ground state vector, and sorted
        uniquified states (if return_eigvec=True). The returned states are the genuine unique rows
        only, never the filler slots, so their count can be below ``states_size``. Their **width
        follows** ``packed``: ``num_qubits`` columns by default, ``ceil((num_qubits + 1) / 8)`` when
        ``packed=True``. Both overloads annotate this as ``StateList``, which cannot express the
        difference, so a type checker will not catch a caller that assumes the wrong width.

        **On a degenerate ground eigenvalue the eigenvector is one arbitrary member of the eigenspace,
        and nothing in the return marks that case.** The eigenvalue is still correct. Anything not
        basis-independent -- per-site occupancies from :math:`|v_i|^2`, say -- therefore gets an
        arbitrary member's value rather than the eigenspace average. Detection needs a second opinion
        (``eigvalsh`` on :func:`hproj`, or a deflate-and-resolve); the solver cannot report it, since
        the Rayleigh-Ritz 3x3 spans the search basis :math:`\{x, y, p\}` and its spacing reflects that
        basis, not the multiplicity -- measured 2.0, not 0, on a 2-fold degenerate operator.

        **Which member is returned is deterministic in the arguments** and may be relied on across
        runs and processes at a fixed rqutils version: the start vector is a fixed hash of the subspace
        index (:func:`_spread_seed`), with no PRNG key and no host-order dependence. It is **not**
        stable across versions -- a change to the seed, the prefilter default or the iteration moves
        it, and none of those is treated as breaking. Pin the version rather than fingerprinting the
        vector.

    Raises:
        RuntimeError: If LOBPCG does not converge within ``maxiter``. Previously the convergence flag
            was discarded and the non-converged value was returned as the answer: it is
            ``state.theta``, a valid variational **upper bound**, so finite, real and above the true
            minimum -- indistinguishable from a correct result by inspection. ``markdown/locg.md`` records
            that this absence "is the reason I4 could hide", a sign error that made the convergence
            test unsatisfiable so the solver silently never converged. Raise ``maxiter``, or loosen
            ``atol`` / ``rtol``, to proceed.
        EigenpairCheckError: If the solve converged but ``||Hv - Ev||``, recomputed after it without
            any cached diagonal, exceeds 10x the convergence bound (or the residual floor).
        ValueError: If ``states_size`` is smaller than ``states.shape[0]``, or if it exceeds
            :math:`2^{31} - 1`, the ceiling imposed by the int32 indices used for subspace positions
            (beyond it an index wraps negative and the subspace is silently permuted); or if either
            ``prefilter`` entry is negative -- see :func:`rqutils.ground_locg._check_prefilter`, which explains why a
            negative value would otherwise be absorbed as a silent no-op; if ``atol`` is ``None`` or
            either tolerance is negative; if **both** tolerances are zero, leaving no satisfiable
            criterion; if ``atol`` is below the achievable eigen-residual floor
            :math:`4\,\varepsilon\sum_k|c_k|` **while** ``rtol`` is zero, so no arm can fire; or if
            ``rtol`` is at least 0.5, where its bound reaches :math:`\|H\|_2` and any vector would
            report convergence; if ``matvec`` is not a kernel name; if ``matvec`` is ``"pairs"`` or
            ``"csr"`` under a mesh, or their operator reaches :math:`2^{31}` entries.
        TypeError: If ``matvec`` is not a ``str``, or ``prefilter`` is neither None nor a
            ``(degree, cycles)`` pair of ints.
    """
    _check_matvec(matvec)
    if matvec in _SPARSE_MATVECS and not get_abstract_mesh().empty:
        raise ValueError(
            f"matvec={matvec!r} is single-device for now; call sqd outside the mesh context, or use "
            "a dense kernel (onthefly, indices, tables) for a sharded solve"
        )
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

    LOG.debug("Starting SQD with array size %s", states_size)
    start = time.time()
    if matvec in _SPARSE_MATVECS:
        # run_sqd is jitted and the entry counts are data-dependent, so the operator is built here.
        states_u = uniquify_states(states_p, states_size)
        operator = _sparse_operator(hamiltonian, states_u, matvec)
        LOG.info("Built the %s operator in %f seconds.", matvec, time.time() - start)
        solve = functools.partial(_run_sparse, hamiltonian, states_u, operator)
    else:
        solve = functools.partial(run_sqd, hamiltonian, states_p)
    result = solve(
        states_size,
        return_eigvec,
        matvec,
        maxiter=maxiter,
        atol=atol,
        rtol=rtol,
        prefilter=prefilter,
        check_residual=True,
    )
    LOG.info("Found ground eigenpair in %f seconds.", time.time() - start)
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
    if return_eigvec:
        eigvec, states_u, subspace_dim = result.eigvec, result.states, result.subspace_dim
        # One `packed` flag governs both directions, so a round trip needs no re-pack (sqd is not a
        # converter). num_qubits, not states.shape[1], which is the packed width on that path.
        basis_states = (
            states_u[:subspace_dim]
            if packed
            else PauliSumXZ.unpack_states(states_u[:subspace_dim], hamiltonian.num_qubits)
        )
        return (eigval, np.array(eigvec[:subspace_dim]), np.asarray(basis_states))
    return eigval


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
        unique_states: Whether ``states`` can be assumed to be already uniquified **and
            lex-sorted**, skipping the internal ``np.unique(..., axis=0)``. Both halves are
            required, because :func:`get_xsource` binary-searches into ``states``; violating either
            raises :class:`ValueError`. Sortedness is validated on this path (host-side numpy,
            measured 12-14% of the call), so an unsorted subspace is rejected rather than silently
            projected into a wrong, non-symmetric matrix as it was before. Leave this ``False`` to
            skip the check and have ``hproj`` sort for you.

    Returns:
        The projected Hamiltonian as a sparse matrix.

    Raises:
        ValueError: If a mesh is set -- ``hproj`` is single-device only; if ``unique_states=True`` and
            ``states`` is not strictly increasing in
            lexicographic order (unsorted, or containing duplicate rows); or if the subspace exceeds
            :math:`2^{31} - 1` states, the ceiling imposed by the int32 indices
            :func:`get_xsource` returns.
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

    Module scope is load-bearing, not style. ``jax.jit`` keys its trace cache on the wrapped
    function *object*, so defining this inside ``hproj`` -- as it used to be -- made a fresh key on
    every call and the cache never hit: each ``hproj`` call paid a full retrace and XLA compile,
    measured at 0.098 s versus 0.0001 s once the cache is warm, i.e. essentially the entire
    steady-state cost of ``hproj``. Nothing else in ``hproj`` is worth hoisting for speed, and not
    because of where it sits: ``PauliSumXZ.from_paulisum`` is 0.4 ms and the ``packbits`` below
    noise. The distinction is compile cost versus host work, and only this closure was compile cost.

    ``states_p`` must stay an argument rather than a captured closure variable, or the retrace
    returns -- a capture is part of the function object, so a new array means a new key again. As an
    argument it is traced, and the cache keys on shape and dtype, which repeat across calls.
    ``PauliSumXZ`` is a registered dataclass pytree, so it passes through as a jit argument.
    """

    def get_from_one(_, ham):
        # Read by name: x and z are same-dtype integer arrays, so swapped positions type-check and
        # silently compute with X and Z exchanged.
        columns = get_xsource(ham.x, states_p)
        diagonals = get_diagonal(ham.z, ham.c, states_p)
        return None, (columns, diagonals)

    return jax.lax.scan(get_from_one, None, hamiltonian.arrays)[1]


def _is_filler(states_u: StateList) -> jax.Array:
    """Return the 0/1 fill-in marker bit of each row of a uniquified state list.

    Filler slots are all-ones rows (``255``); genuine states have byte 0 ``< 128`` because
    :meth:`PauliSumXZ.pack_states` inserts a leading zero pad bit. So the high bit of byte 0 identifies fillers,
    and testing it is equivalent to testing ``states_u[:, 0] == 255`` -- one spelling for all three
    consumers, rather than two that a reader has to re-derive as equal.

    Returned as the bit itself, not a bool, because the fillers sort to the end: the result is a
    non-decreasing 0/1 array, which is what lets ``run_sqd`` locate the subspace boundary with a
    ``searchsorted`` instead of a count.
    """
    return states_u[:, 0] >> 7


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


def _pad_states(states_p: StateList, states_size: int) -> StateList:
    """``states_p`` padded to ``states_size`` rows with the ``255`` filler :func:`run_sqd` expects.

    An all-ones row sorts last and sets byte 0's high bit, which ``_is_filler`` tests; a genuine
    state never collides, since the pad bit keeps its byte 0 below 128.
    """
    deficit = states_size - states_p.shape[0]
    if deficit < 0:
        raise ValueError(f"{states_p.shape[0]} states exceed states_size={states_size}")
    filler = np.full((deficit, states_p.shape[1]), 255, dtype=np.uint8)
    return np.append(states_p, filler, axis=0)


class SqdResult(NamedTuple):
    """What :func:`run_sqd` returns; the optional fields are ``None`` unless requested."""

    eigval: jax.Array
    converged: jax.Array
    eigvec: jax.Array | None = None
    states: jax.Array | None = None
    subspace_dim: jax.Array | None = None
    residual: jax.Array | None = None
    ax_norm: jax.Array | None = None


@jax.jit(
    static_argnames=[
        "states_size",
        "return_eigvec",
        "matvec",
        "maxiter",
        "prefilter",
        "log_level",
        "check_residual",
    ]
)
def run_sqd(
    hamiltonian: PauliSumXZ,
    states_p: StateList,
    states_size: int,
    return_eigvec: bool,
    matvec: Matvec = "indices",
    maxiter: int = 1000,
    atol: float = 0.0,
    rtol: float | None = None,
    prefilter: tuple[int, int] | None = (32, 2),
    log_level: int = logging.INFO,
    check_residual: bool = False,
) -> SqdResult:
    """JIT-compiled part of the SQD function; returns a :class:`SqdResult`.

    ``converged`` is returned rather than checked because this function is ``@jax.jit``-wrapped, so it
    is a traced boolean here; :func:`sqd` raises on it once concrete. It used to be discarded, which
    let a non-converged ``theta`` -- a valid variational upper bound, so plausible -- pass as the answer.

    The solver always runs with ``batch_matvec=True``: every kernel here broadcasts over a leading
    batch axis, so the stacked call is bit-identical to two separate ones.

    Args:
        matvec: The kernel name, as in :func:`sqd`. Static, bound into the kernel via
            :func:`functools.partial` because ``ground_locg`` splats ``args`` positionally.
        maxiter: Maximum LOBPCG iterations, forwarded to :func:`rqutils.ground_locg.ground_locg`.
            Static, as it is there.
        atol: Absolute bound on the eigen-residual ``||Hv - Ev||``. Validated in :func:`sqd`, which is
            where ``sum|c_k|`` is concrete -- this function is jitted, so it cannot raise.
        rtol: Relative tolerance on ``||Hv|| + |E|``. Convergence is the ``max`` of the two arms, so
            either suffices; see :func:`rqutils.ground_locg.ground_locg`, which both are forwarded to.
        prefilter: Optional ``(degree, cycles)`` Chebyshev prefilter, forwarded to
            :func:`rqutils.ground_locg.ground_locg`. Static, as it is there -- passed by keyword, so
            unlike ``matvec`` it needs no :func:`functools.partial` binding. See :func:`sqd` on
            why this option's published speedups do not transfer to this path.
        check_residual: Recompute ``||Hv - Ev||`` and ``||Hv||`` after the solve, into ``residual``
            and ``ax_norm``, from recomputed diagonals (and a fresh search unless ``xsources`` are
            cached). :func:`sqd` turns it on and raises on the result.

    Raises:
        ValueError: If ``matvec`` is ``"pairs"`` or ``"csr"``, which only :func:`sqd` can build.
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

    # Diagonals always recomputed, so no cached one vouches for itself; cached xsources are reused,
    # since redoing the J-fold search was ~90% of the check (NOTES.md, "`EigenpairCheckError`").
    check_ref = ("onthefly", hamiltonian.x) if matvec == "onthefly" else ("indices", xsources)
    return _solve(
        hamiltonian,
        states_u,
        states_size,
        apply,
        args,
        diag0,
        check_ref,
        matvec,
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
    states_size: int,
    apply: Callable[..., jax.Array],
    args: tuple,
    diag0: Callable[[], jax.Array],
    check_ref: tuple[Matvec, NDArray],
    matvec: Matvec,
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

    ``apply(vec, *args)`` is the operator, ``diag0()`` the identity-X group's diagonal, and
    ``check_ref`` the ``(kernel, X groups)`` the residual check recomputes :math:`Hv` with.
    """

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
        jax.debug.print(f"Starting minimization with matvec {matvec}")

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
        ref, xgroup = check_ref
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


#: The kernels whose operator arrays :func:`sqd` builds host-side; single-device for now.
_SPARSE_MATVECS = ("pairs", "csr")
#: Entries per scanned chunk, so the sparse kernels' temporaries are ``O(chunk)`` (``poc/sparse-pairs.md``).
_CHUNK = 1 << 15


def _size_class(chunks: int) -> int:
    """Round a chunk count up to ``m * 2**k`` with ``8 <= m < 16`` (exact below 16), at least 1.

    The jitted solve recompiles per class rather than per subspace, wasting at most 12.5%.
    """
    shift = max(chunks.bit_length() - 4, 0)
    return max(-(-chunks >> shift) << shift, 1)


def _check_entry_count(count: int) -> None:
    """Raise unless ``count`` sparse entries are addressable by the int32 indices they are stored in.

    Raises:
        ValueError: If ``count`` is at least :math:`2^{31}`.
    """
    if count >= 2**31:
        raise ValueError(
            f"the sparse operator has {count} entries, beyond the 2^31 - 1 addressable with int32 "
            'indices; use matvec="indices" or a smaller subspace'
        )


def _padded(count: int, fill: int, dtype: DTypeLike) -> np.ndarray:
    """A flat array of ``count`` entries rounded up to whole chunks of a size class, all ``fill``."""
    _check_entry_count(count)
    return np.full(_size_class(-(-count // _CHUNK)) * _CHUNK, fill, dtype=dtype)


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


def _sparse_operator(
    hamiltonian: PauliSumXZ, states_u: StateList, matvec: Matvec
) -> tuple[jax.Array, ...]:
    """Build the ``"pairs"`` or ``"csr"`` operator arrays on the host, one X group's search at a time.

    Returns ``(d0, i, j, d)`` for ``"pairs"`` and ``(d0, rt, rs, rd, qt, qs, qd)`` for ``"csr"``, each
    entry array ``(chunks, _CHUNK)``; ``r``/``q`` are the real-coefficient groups (float64 factors)
    and the rest. Padding entries have equal endpoints and a zero factor.

    Raises:
        ValueError: If the entry count reaches :math:`2^{31}` -- see :func:`_check_entry_count`.
    """
    size = states_u.shape[0]
    z, c = jnp.asarray(hamiltonian.z), jnp.asarray(hamiltonian.c)
    first = int(np.all(np.asarray(hamiltonian.x[0]) == 0))
    d0 = get_diagonal(z[0], c[0], states_u) if first else jnp.zeros(size, c.dtype)
    groups = range(first, hamiltonian.x.shape[0])
    coeffs = np.asarray(hamiltonian.c)
    kmax = max([int(np.count_nonzero(coeffs[g])) for g in groups], default=1)
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
        host = [_padded(count, 0, np.int32) for _ in range(3)]  # i, j, group
        pos = 0
        for g in groups:
            i, j = pairs.pop(g)
            for array, value in zip(host, (i, j, g)):
                array[pos : pos + len(i)] = value
            pos += len(i)
        return (d0, *on_device(host, c))

    real = ~np.any(coeffs.imag != 0, axis=1)
    arrays = [d0]
    for subset, c_set in (
        ([g for g in groups if real[g]], c.real),
        ([g for g in groups if not real[g]], c),
    ):
        # Counting sort by target straight into padded chunks: a row occurs at most once per group,
        # so each group's fill is conflict-free (poc/sparse-pairs.md, section 2).
        start = np.zeros(size + 1, np.int64)
        for g in subset:
            i, j = pairs[g]
            start[i + 1] += 1
            start[j + 1] += 1
        np.cumsum(start, out=start)
        host = [_padded(int(start[-1]), fill, np.int32) for fill in (size - 1, size - 1, 0)]
        for g in subset:
            i, j = pairs.pop(g)
            for target, source in ((i, j), (j, i)):
                pos = start[target]
                host[0][pos], host[1][pos], host[2][pos] = target, source, g
                start[target] += 1
        del start
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


def _sorted_scatter(
    vec: jax.Array, out: jax.Array, t: jax.Array, s: jax.Array, d: jax.Array
) -> jax.Array:
    """``out[t] += d * vec[s]`` over target-sorted chunks."""
    sharding = jax.typeof(vec).sharding

    def body(acc, chunk):
        ti, si, di = chunk
        gathered = vec.at[..., si].get(out_sharding=sharding)
        return acc.at[..., ti].add(di * gathered, indices_are_sorted=True), None

    return jax.lax.scan(body, out, (t, s, d))[0]


def _apply_csr(vec: jax.Array, d0: jax.Array, *entries: jax.Array) -> jax.Array:
    """``"csr"``: ``d0 * vec`` plus one sorted scatter per ``(t, s, d)`` entry set."""
    out = d0 * vec
    for t, s, d in zip(entries[::3], entries[1::3], entries[2::3]):
        out = _sorted_scatter(vec, out, t, s, d)
    return out


@jax.jit(
    static_argnames=[
        "states_size",
        "return_eigvec",
        "matvec",
        "maxiter",
        "prefilter",
        "log_level",
        "check_residual",
    ]
)
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
    """:func:`run_sqd` for ``"pairs"``/``"csr"``, given :func:`_sparse_operator`'s arrays.

    The residual check runs the ``"onthefly"`` kernel, so it reads none of ``operator``.
    """
    apply = _apply_pairs if matvec == "pairs" else _apply_csr
    return _solve(
        hamiltonian,
        states_u,
        states_size,
        apply,
        operator,
        lambda: operator[0],
        ("onthefly", hamiltonian.x),
        matvec,
        None,
        return_eigvec=return_eigvec,
        maxiter=maxiter,
        atol=atol,
        rtol=rtol,
        prefilter=prefilter,
        log_level=log_level,
        check_residual=check_residual,
    )


@jax.jit(static_argnames=["states_size"])
def uniquify_states(states_p: StateList, states_size: int) -> StateList:
    """A stripped-down implementation of jnp.unique.

    The returned array will have shape (states_size, states_p.shape[1]). If states_size is greater
    than the number of unique states, the residual entries at the end are filled with 255.

    Raises:
        ValueError: If ``states_size`` exceeds :data:`_MAX_STATES`, the ceiling imposed by the int32
            iota below.
    """
    # Also checked here, where the int32 iota is created: poc/ calls this directly, and it is free
    # at trace time (NOTES.md, "The `N ≤ 2^31 - 1` ceiling: why it is enforced where it is").
    if states_size > _MAX_STATES:
        raise ValueError(
            f"states_size {states_size} exceeds the {_MAX_STATES} limit imposed by the int32 index "
            "below; beyond it the iota wraps negative and the subspace is silently permuted"
        )
    # Lexsort on uint64 words, not uint8 columns: lax.sort pays per key and the order is identical;
    # it trades memory for speed (NOTES.md, "sqd.uniquify_states: lexsort on uint64 words").
    words = _pack_state_words(states_p)
    iota = jax.lax.broadcasted_iota(np.int32, (states_p.shape[0],), 0)
    perm = jax.lax.sort((*words.T, iota), dimension=0, num_keys=words.shape[1])[-1]
    states_srt = states_p[perm]
    # Uniqueness on the packed words, not the rows: same answer (the packing is injective -- the pad
    # bytes are constant) over ceil(B/8) columns instead of B.
    words_srt = words[perm]
    is_unique = jnp.any(jax.lax.ne(words_srt[1:], words_srt[:-1]), axis=1)
    # Element 0 is always considered unique -> add 1
    total_unique = jnp.sum(is_unique, dtype=np.int32) + 1
    # This cumsum(bincount(cumsum)) accounts for the uniqueness of the 0th element
    idx_unique = jnp.cumsum(
        jnp.bincount(jnp.cumsum(is_unique, dtype=np.int32), length=states_size), dtype=np.int32
    )
    # Finally flag out filler slots by setting total_unique: to -1
    if states_size != states_p.shape[0]:
        iota = jax.lax.broadcasted_iota(np.int32, (states_size,), 0)
    idx_unique = jnp.where(iota < total_unique, idx_unique, -1)
    # With wrap_negative_indices=False we'll have 255 for filler slots
    return states_srt.at[idx_unique].get(mode="fill", fill_value=255, wrap_negative_indices=False)


def _pack_state_keys(states: StateList) -> jax.Array:
    """Pack `[N, B]` uint8 state rows into `[N]` uint64 scalar keys, preserving lex order.

    Byte 0 becomes the most significant, so integer order on the keys is identical to row lex order.
    That equivalence is the whole point: it lets a scalar binary search stand in for a lexicographic
    one.

    Args:
        states: Packed state rows, shape ``[N, B]`` with ``B <= 8``.

    Returns:
        One ``uint64`` key per row, ordered identically to the rows.

    Raises:
        ValueError: If ``B > 8``. :func:`get_xsource` already routes wide input to the lexicographic
            path, so this is defence-in-depth for anyone reaching past it -- but the limit was
            previously only *described* here, and the failure is worse than truncation: byte 0 is the
            most significant, so at ``B = 9`` its shift is ``8 * (9 - 1) = 64`` bits on a ``uint64``
            and the byte vanishes outright rather than being coarsened. Measured, two 9-byte rows
            differing only in byte 0 both pack to key ``0``, aliasing distinct states and destroying
            the lex-order equivalence the search depends on. ``B = ceil((n + 1) / 8)``, so ``n >= 64``
            reaches it.
    """
    nbytes = states.shape[1]
    if nbytes > 8:
        raise ValueError(
            f"`_pack_state_keys` needs at most 8 bytes per row to fit a uint64 key, got {nbytes}. "
            "Byte 0 is the most significant, so a wider row shifts it out entirely and distinct "
            "states alias onto one key. `get_xsource` routes B > 8 to the lexicographic search."
        )
    shifts = jnp.asarray([8 * (nbytes - 1 - i) for i in range(nbytes)], dtype=jnp.uint64)
    return jnp.sum(states.astype(jnp.uint64) << shifts, axis=1)


def _is_lex_sorted(states: NDArray[np.uint8]) -> bool:
    """Return whether `[N, B]` uint8 rows are in strictly increasing lexicographic order.

    Host-side numpy, deliberately: the one caller (:func:`hproj`) is eager, and the point is to
    ``raise`` before any tracing, which a traced predicate cannot do. Compares adjacent rows at their
    first differing byte -- equal rows count as *unsorted*, since `get_xsource`'s precondition is
    uniquified-and-sorted and a duplicate row makes the projection ambiguous either way.

    One vectorized pass, no early exit, so unsorted input costs the same as sorted. Measured at 12-14%
    of `hproj` (A/B'd end-to-end; both are `O(N)`, so the ratio is flat in N) and ~20 ms standalone at
    N=1M, flat in `B`. Cheap enough to be unconditional on a debug/reference path, in exchange for
    turning a silent wrong answer into a raise. :func:`sqd` never reaches `hproj`, so it is unaffected.

    **Rejects a padded :func:`uniquify_states` result, by design** -- and now for *any* number of
    filler slots. Fillers are all-``255`` rows, so two or more are duplicates and fail the strictness
    test; that is what this paragraph used to rest on, but a **single** filler row is strictly
    increasing and passed. `hproj` builds a dense `[N, N]` operator with no filler-masking step, so
    that row became a spurious basis state: one row and column too large, still symmetric, plausible
    wrong eigenvalue (measured **-1.118034 against a true -1.0**). There is now an explicit high-bit
    test, independent of sortedness.

    It reads the *packed* byte deliberately. :meth:`PauliSumXZ.pack_states` makes byte 0 of every
    genuine state ``< 128``, so ``255`` is unambiguous there -- whereas unpacking a filler at
    ``n = 2`` yields ``[1, 1]``, a perfectly legitimate state, so no check on the unpacked form could
    distinguish them.

    Slice to the real rows first (`~_is_filler(states)`) if you hold a padded array; `sqd` already
    trims before returning its basis.
    """
    # Any filler disqualifies, tested apart from sortedness (a lone filler is strictly increasing);
    # fillers sort last, so the last row decides (NOTES.md, "sqd._is_lex_sorted: the filler test").
    if states.shape[0] and states[-1, 0] >= 128:
        return False
    if states.shape[0] < 2:
        return True
    lhs, rhs = states[:-1], states[1:]
    differs = lhs != rhs
    # A row pair with no differing byte is a duplicate -> not strictly increasing.
    if not bool(np.all(np.any(differs, axis=1))):
        return False
    first = np.argmax(differs, axis=1)
    rows = np.arange(lhs.shape[0])
    return bool(np.all(lhs[rows, first] < rhs[rows, first]))


def _pack_state_words(states: StateList) -> jax.Array:
    """Pack `[N, B]` uint8 rows into `[N, ceil(B/8)]` uint64 words, preserving lex order.

    The wide counterpart to :func:`_pack_state_keys`, which is limited to a single word. Words are
    big-endian, MSW-first, and the most significant word is left-padded with zero bytes, so word-wise
    lexicographic order matches byte-wise row order exactly.

    Leading-padding is chosen so that ``nwords == 1`` reproduces :func:`_pack_state_keys` bit for bit,
    which is what lets the two paths be compared directly. It is *not* required for correctness:
    trailing-padding is also order-preserving, since appending a constant number of zero bytes is a
    left-shift by ``8 * pad`` and a constant left-shift is monotonic. Verified exhaustively at
    ``B = 3`` and over 20000 random pairs at ``B = 9``; a mutation to the other end leaves the whole
    suite green, so do not read the choice as load-bearing.

    Returns:
        Shape ``[N, ceil(B/8)]`` uint64. ``B <= 8`` yields one column, identical in value to
        :func:`_pack_state_keys`' output.
    """
    nbytes = states.shape[1]
    nwords = -(-nbytes // 8)
    pad = nwords * 8 - nbytes
    padded = (
        states
        if not pad
        else jnp.concatenate([jnp.zeros((states.shape[0], pad), dtype=jnp.uint8), states], axis=1)
    )
    shifts = jnp.asarray([8 * (7 - i) for i in range(8)], dtype=jnp.uint64)
    words = [
        jnp.sum(padded[:, w * 8 : (w + 1) * 8].astype(jnp.uint64) << shifts, axis=1)
        for w in range(nwords)
    ]
    return jnp.stack(words, axis=1)


def _word_less_than(rows: jax.Array, targets: jax.Array) -> jax.Array:
    """Elementwise lexicographic `rows[i] < targets[i]` over uint64 words, MSW-first.

    Lexicographic row order, but on packed words: a search level costs ``ceil(B/8)`` comparisons
    instead of ``B``. At n=100 (``B = 13``) that is 2 rather than 13, measured 3.6-7.7x on the whole
    search across n=64..200 and bit-identical to the byte-wise form it replaced.

    The unrolled Python loop is deliberate: ``nwords`` is static, and a `lax` loop would need a
    traced index into a static shape.

    Accumulating ``lt``/``eq`` in bool is safe here, but do not reintroduce a ``jnp.cumprod`` prefix
    over the word axis to "vectorize" it. The byte-wise predecessor this replaced did exactly that and
    had to pin ``dtype=jnp.uint8``: `cumprod` rejects a bool accumulator and promotes to int64,
    materializing the ``[N, B]`` mask at 8 bytes per element where 1 suffices -- measured 192 MB of
    transients against 23 MB at N=1M, B=12, and 1.53x on the J-fold precompute this path feeds. Two
    or four scalar comparisons need no prefix at all.
    """
    lt = jnp.zeros(targets.shape[0], dtype=bool)
    eq = jnp.ones(targets.shape[0], dtype=bool)
    for w in range(rows.shape[1]):
        a, b = rows[:, w], targets[:, w]
        lt = jnp.logical_or(lt, jnp.logical_and(eq, a < b))
        eq = jnp.logical_and(eq, a == b)
    return lt


@jax.jit
def get_xsource(xsignature: NDArray[np.uint8], states: StateList) -> jax.Array:
    """Return an index array into the source of an X operation.

    Let `V` be a vector of complex or float values with shape `[N]`, `S` be a lex-sorted 2-d array
    of uint8 with shape `[N, B]` where `B = ceil(Q/8)`, and `X` be a vector of uint8 with shape
    `[B]`. An unpacked (truncated to `Q` bits) `X` is a bitstring that represents the location of X
    being applied to the states in `S`; X (I) is applied to qubit `q` if `Q-q-1`th bit is 1 (0). Let
    `P` be the projector of the shape `[2 ** Q]` state vector `W` onto `V`.

    We want to find a vector of indices `A` where `(PXW)[i] = V[A[i]]`. This is trivial if `P` is
    the identity (or equivalently if `S` contains all bitstrings from `0` to `2^Q-1`), because then
    we know that `S[i] = i` and therefore `A[i] = i ^ X`. In the presence of a nontrivial
    projection, we must take care of not only the source location but also the existence of the
    source itself, since it is not guaranteed that `S[i] ^ X` is in `S`. When it is not, we set
    `A[i] = -1` so that `V[A[i]]` can default to a `fill_value` of 0.0 through `at[].get()` applied
    to `V`.

    **`states` must be lex-sorted.** This has always been required -- the previous sort-based
    implementation also silently returned a wrong answer otherwise -- but was never stated. Both
    in-tree callers satisfy it: `run_sqd` passes `uniquify_states`' output, and `hproj` passes
    `np.unique(..., axis=0)`'s. Note `hproj`'s `unique_states=True` shortcut skips that `np.unique`,
    so a caller passing unsorted-but-unique states gets a wrong (and non-symmetric) matrix; that
    predates this implementation and is pinned by
    `test/test_sqd_hproj.py::TestHproj::test_unsorted_input_with_unique_states_is_wrong` -- named for the
    behaviour, since nothing is rejected: the result is silently wrong.

    Since `S` is sorted, finding `A` is a **binary search** of `S ^ X` into `S` -- not a reason to
    sort anything. The former implementation concatenated `S` and `S ^ X` into a `[2N, B]` array and
    sorted that, which cost three things: the `2N` allocation is what caps `N` at `2^31` (the sort
    must run on one device), `lax.sort` was observed to leak GPU memory (up to 5 GB at shape
    `(5M, 9)`), and it dominated runtime -- measured 66-97% of an entire solve. A `searchsorted` is a
    pure gather, so it also shards, where a sort does not. Measured on CPU at 12-25x per signature and
    12-17x on the J-fold precompute, and **5.15x at N=64M on an NVIDIA GH200** (a GPU sort is
    well optimized relative to its gather, so the ratio compresses while the direction holds); see
    `markdown/scaling-pocs.md`.

    The memory leak, re-measured on that GH200 against a pinned copy of the old sort, **did not
    reproduce**: ~0.95 GB of transients at `(5M, 4)` were fully reclaimed after every repetition. That
    note was therefore stale or version-specific. It is recorded here as history because the removal
    never depended on it -- the `2N` ceiling, shardability and runtime each justify it alone.

    Two paths, selected statically on width. `B <= 8` packs each row into a `uint64` and uses
    `jnp.searchsorted` directly; wider inputs fall back to an explicit binary search over the rows
    packed into `ceil(B/8)` `uint64` words (:func:`_pack_state_words`), MSW-first. The boundary is a
    correctness limit, not a tuning parameter: at `B > 8` a *single* `uint64` key would silently
    truncate the row and alias distinct states onto one key.

    The wide path used to compare rows one **byte** at a time, which cost `O(B)` comparisons per
    search level and made this path scale poorly in `n`: measured **~8x per state** across the
    boundary (15 ns/state at n=60 against 189 at n=100, N=300k). Comparing words instead makes a
    level cost `ceil((n+1)/64)` comparisons -- 2 rather than 13 at n=100 -- so cost is logarithmic in
    packed width. Measured **3.6-7.7x** on the whole search across n=64..200, bit-identical to the
    byte-wise form at every width, and the `B <= 8` path is untouched (0.83-1.18x, i.e. noise).

    Returns `-1` at every position whose source is absent. Note the previous implementation returned
    *assorted* negative values there (it computed `I[k+1] - N` unconditionally) rather than exactly
    `-1`; consumers cannot tell, because `apply_xgrp` gathers with
    `mode="fill", wrap_negative_indices=False` and any negative index yields 0.0. Tests comparing
    against a stored index array must therefore compare only valid rows, or compare the gathered
    result.
    """
    size, nbytes = states.shape
    targets = jnp.bitwise_xor(states, xsignature)  # S^X
    invalid = np.array(-1, dtype=np.int32)

    if nbytes <= 8:
        keys = _pack_state_keys(states)
        # One name for the search key, used by both the search and the hit test: that they are the
        # same quantity is the whole correctness argument for the branch.
        target_keys = _pack_state_keys(targets)
        pos = jnp.searchsorted(keys, target_keys, side="left")
        # searchsorted returns N for a target above every key; clamp before gathering so the
        # equality test below stays in bounds.
        pos = jnp.minimum(pos, size - 1)
        found = keys[pos] == target_keys
    else:
        # Binary search on uint64 words, not bytes: ceil(B/8) comparisons per level instead of B
        # (NOTES.md, "sqd.get_xsource: search on uint64 words"). lo counts rows below the target.
        swords = _pack_state_words(states)
        twords = _pack_state_words(targets)

        def step(carry, _):
            lo, hi = carry
            mid = (lo + hi) // 2
            go_right = _word_less_than(swords[jnp.minimum(mid, size - 1)], twords)
            return (jnp.where(go_right, mid + 1, lo), jnp.where(go_right, hi, mid)), None

        nsteps = int(np.ceil(np.log2(max(size, 2)))) + 1
        lo = jnp.zeros(size, dtype=jnp.int32)
        hi = jnp.full(size, size, dtype=jnp.int32)
        (lo, _), _ = jax.lax.scan(step, (lo, hi), None, length=nsteps)
        pos = jnp.minimum(lo, size - 1)
        found = jnp.all(swords[pos] == twords, axis=1)

    xsource = jnp.where(found, pos, invalid).astype(np.int32)
    if not (mesh := get_abstract_mesh()).empty:
        xsource = jax.reshard(xsource, PartitionSpec(mesh.axis_names))
    return xsource


def _z_parity(states: StateList, zsignature: jax.Array) -> jax.Array:
    """Return `popcount(state & z) mod 2` per state, the sign bit of one Z term.

    Accumulated in uint8 and reduced with `& 1`: the parity is the only bit that matters.
    """
    return jnp.sum(jnp.bitwise_count(states & zsignature), axis=1, dtype=np.uint8) & 1


def _accumulate_diagonal(
    coeffs: NDArray[np.inexact], template: jax.Array, sign_bit: Callable[[jax.Array], jax.Array]
) -> jax.Array:
    """Sum ``coeff * (1 - 2 * sign_bit(iterm))`` over the Z terms of one X group.

    Null terms are removed by ``hamiltonian.simplify()`` on ingest, so a zero coefficient marks the
    end of the real terms in a zero-padded Z group and the loop stops there rather than scanning the
    full rectangle.

    Args:
        coeffs: Phased coefficients for this X group, shape ``(K,)``.
        template: Array whose leading axis length and sharding the output follows. May be of any
            rank -- only the leading axis's partitioning is carried over, since the output is 1-D.
        sign_bit: Maps a term index to that term's per-state sign bit (0 or 1).

    Returns:
        The composed diagonal, shape ``(template.shape[0],)``.
    """

    def cond_fn(val):
        iterm = val[1]
        return jnp.logical_and(iterm < coeffs.shape[0], jnp.not_equal(coeffs[iterm], 0.0))

    def add_diag(val):
        diagonal, iterm = val
        signs = 1.0 - 2.0 * sign_bit(iterm)
        return diagonal + coeffs[iterm] * signs, iterm + 1

    # Only the template's leading-axis sharding (2-D template, 1-D output), as a NamedSharding so it
    # works with no mesh (NOTES.md, "sqd._accumulate_diagonal: the output sharding").
    sharding = jax.typeof(template).sharding
    init = jnp.zeros(
        template.shape[0],
        dtype=coeffs.dtype,
        out_sharding=jax.sharding.NamedSharding(sharding.mesh, PartitionSpec(sharding.spec[0])),
    )
    return jax.lax.while_loop(cond_fn, add_diag, (init, 0))[0]


@jax.jit
def get_diagonal(
    zsignatures: NDArray[np.uint8], coeffs: NDArray[np.inexact], states: StateList
) -> jax.Array:
    """Return the fully composed diagonals for one X signature.

    Args:
        zsignatures: Packed Z signatures for one X group, shape ``(num_zterms, num_bytes)``.
        coeffs: Phase-folded coefficients for that group.
        states: Uniquified, lex-sorted packed state list.

    Returns:
        The composed diagonal, one entry per state.

    Raises:
        ValueError: If ``zsignatures`` is not 2-D -- see :func:`_check_zsignatures_rank`.
    """
    _check_zsignatures_rank(zsignatures)

    def sign_bit(iterm):
        return _z_parity(states, zsignatures[iterm])

    return _accumulate_diagonal(coeffs, states, sign_bit)


@jax.jit
def apply_xgrp(
    xsource: NDArray[np.int32], diagonal: NDArray[np.inexact], vec: NDArray[np.inexact]
) -> jax.Array:
    """Gather vector entries from the source indices and multiply them with diagonals."""
    xvec = vec.at[..., xsource].get(
        mode="fill",
        fill_value=0.0,
        wrap_negative_indices=False,
        out_sharding=jax.typeof(vec).sharding,
    )
    return xvec * diagonal


def _pack_scanned(
    matvec: Matvec, xgroup: NDArray, diagonal_arg: NDArray, coeffs: NDArray | None
) -> tuple[NDArray, ...]:
    """Lay out the tuple ``_apply_h_kernel`` scans over, for one resolved ``matvec``.

    The kernel unpacks positionally (``val[0]``, ``val[1]``, and ``val[2]`` only when ``matvec`` is
    not ``"tables"``), so the arity rule is a contract between packer and kernel: a 3-tuple carrying
    the coefficients for the two kernels that *compute* a diagonal, a 2-tuple for ``"tables"``,
    which reads a precomputed one. Both callers -- ``run_sqd`` and ``apply_h``'s keyword resolution --
    go through here so that rule is stated once rather than once per caller.
    """
    if matvec == "tables":
        return (xgroup, diagonal_arg)
    return (xgroup, diagonal_arg, coeffs)


def apply_h(
    vec: NDArray[np.inexact],
    *,
    states: StateList | None = None,
    xsignatures: NDArray | None = None,
    xsources: NDArray | None = None,
    zsignatures: NDArray | None = None,
    diagonals: NDArray | None = None,
    coeffs: NDArray | None = None,
) -> jax.Array:
    r"""Return :math:`Hv`, naming the per-X-group inputs so a mispairing cannot be expressed.

    Name the per-X-group arrays you have and the kernel follows from them. Exactly three input sets
    are accepted, one per dense :func:`sqd` ``matvec``; ``"pairs"``/``"csr"`` are ``sqd``-only:

    .. code-block:: python

        apply_h(vec, states=..., xsignatures=..., zsignatures=..., coeffs=...)  # "onthefly"
        apply_h(vec, states=..., xsources=..., zsignatures=..., coeffs=...)     # "indices"
        apply_h(vec, xsources=..., diagonals=...)                               # "tables"

    Every array parameter is keyword-only. **This replaced a positional ``(scanned, cache_level)``
    form, which is gone** -- a breaking change, because that tuple selected *positionally* how
    the members of ``scanned`` were interpreted and nothing checked that they matched: raw X
    signatures under a tuple promising X *sources* silently computed a different operator (measured
    max abs error 0.44 on a 5-state n=4 subspace).

    :func:`sqd` and :mod:`ground_locg` do not go through here: they call the private
    ``_apply_h_kernel`` with an assembled tuple and a static ``matvec`` name bound via
    ``functools.partial``, because the solver splats ``matvec(vec, *args)`` positionally.

    A shape check cannot separate the roles -- at ``n = 15`` with a 2-state subspace X sources and X
    signatures are both ``(2, 2)`` -- but a dtype check can, and is applied
    (:func:`_check_array_role`). What remains open is a swap between two roles of the **same** kind,
    ``xsignatures`` for ``zsignatures``, both ``uint8``.

    Each kernel is one ``jax.lax.scan`` over the X groups accumulating
    ``out + apply_xgrp(xsource, diagonal, vec)``, with ``xsource`` either ``get_xsource(x, states)``
    or ``xsources`` as given, and ``diagonal`` either ``get_diagonal(z, c, states)`` or ``diagonals``
    as given.

    **Under a mesh** ``vec`` is placed on the live mesh automatically, batch axis and all. Two
    constraints stay the caller's: with ``xsignatures=`` the state count must divide the device
    count, since ``get_xsource`` partitions per state -- build the subspace with
    ``uniquify_states(states, states_size)`` at a divisible ``states_size``, **not** from ``sqd``'s
    return, which is trimmed to the genuine uniques and so is the one input guaranteed to fail -- and
    a ``states`` passed already sharded must be replicated, its ``255`` filler being load-bearing.

    Args:
        vec: Vector to multiply. Placed on the live mesh unless already there.
        states: Uniquified state list. Required except with ``diagonals=``, which reads neither
            signature array.
        xsignatures: Packed X signatures per group; selects ``"onthefly"``.
        xsources: Precomputed X source indices per group; selects ``"indices"`` or ``"tables"``.
        zsignatures: Packed Z signatures per group; selects ``"onthefly"`` or ``"indices"``. Needs
            ``coeffs``.
        diagonals: Fully precomputed diagonals per group; selects ``"tables"``, so requires
            ``xsources=``. Must not be combined with ``coeffs``, which it makes redundant.
        coeffs: Pauli coefficients per group. Required by ``zsignatures``.

    Returns:
        :math:`Hv`.

    Raises:
        ValueError: If the named arrays do not select exactly one X source and one diagonal input
            (including naming none at all); if ``diagonals`` is combined with ``xsignatures``; if
            ``coeffs`` is missing where required or supplied alongside ``diagonals``; or if
            ``states`` is None without ``diagonals``. Under a mesh with ``xsignatures=``, also if
            ``states`` disagrees with ``vec``'s trailing axis, or if its row count does not divide
            the device count.
    """
    xgiven = [
        opt for opt in (("xsources", xsources), ("xsignatures", xsignatures)) if opt[1] is not None
    ]
    dgiven = [
        opt
        for opt in (("diagonals", diagonals), ("zsignatures", zsignatures))
        if opt[1] is not None
    ]
    if len(xgiven) != 1:
        raise ValueError(
            "apply_h: pass exactly one of xsources= or xsignatures= "
            f"(got {sorted(name for name, _ in xgiven) or 'neither'})"
        )
    if len(dgiven) != 1:
        raise ValueError(
            "apply_h: pass exactly one of diagonals= or zsignatures= "
            f"(got {sorted(name for name, _ in dgiven) or 'neither'})"
        )
    (xname, xarray), (dname, darray) = xgiven[0], dgiven[0]
    precomputed = dname == "diagonals"
    if precomputed and xname == "xsignatures":
        raise ValueError(
            "apply_h: diagonals= requires xsources=, not xsignatures= -- that pairing was removed as "
            'dominated by matvec="indices"; pass xsources= from get_xsource'
        )
    matvec = "tables" if precomputed else "indices" if xname == "xsources" else "onthefly"

    # coeffs is required by the computed diagonal and meaningless beside a precomputed one, so a stray
    # one is rejected rather than ignored.
    if precomputed and coeffs is not None:
        raise ValueError("apply_h: diagonals= already folds in coeffs=; do not pass both")
    if not precomputed and coeffs is None:
        raise ValueError(f"apply_h: {dname}= requires coeffs=")

    # Not redundant with the kernel's check: the missing-input-set error must precede the role check
    # below (TestMatvecKernels::test_omitting_states_raises); keep the messages in step.
    if matvec != "tables" and states is None:
        raise ValueError(f"states is required for matvec={matvec!r}")

    # Dtype separates what shape cannot (uint8 signatures vs int32 sources, both (2, 2) at n=15);
    # last, so it only adds errors and never displaces an input-set error raised above.
    _check_array_role(xname, xarray)
    _check_array_role(dname, darray)

    # `states`, not `vec`: `get_xsource` reshards one entry per state, and only "onthefly" gets there.
    vec = _place_vec(vec, states if matvec == "onthefly" else None)

    return _apply_h_kernel(vec, _pack_scanned(matvec, xarray, darray, coeffs), states, matvec)


def _check_mesh_divisible(num_states: int) -> None:
    """Raise unless the state count divides the mesh, naming the size to pass `uniquify_states`.

    `get_xsource` reshards one entry per state. jax's own message names neither the size nor the call
    that produces it.

    Raises:
        ValueError: If ``num_states`` is not a multiple of the device count.
    """
    mesh = get_abstract_mesh()
    if mesh.empty or num_states % mesh.size == 0:
        return
    size = -(-num_states // mesh.size) * mesh.size
    raise ValueError(
        f"apply_h: {num_states} states is not a multiple of the {mesh.size} mesh devices; size "
        f"every per-state array to {size}, e.g. uniquify_states(states, {size})"
    )


def _place_vec(vec: NDArray[np.inexact], divisible: StateList | None) -> NDArray[np.inexact]:
    """Put `vec` on the live mesh, replicated; return it unchanged when there is no mesh.

    Keyed on mesh *identity*, not `isinstance(vec, jax.Array)`: a committed `jax.Array` carries an
    empty mesh just as a host array does, so that guard passed it through to the "Resource axis" error
    this prevents. `get_mesh`, not `get_abstract_mesh`: `device_put` rejects an `AbstractMesh`. Skipped
    under tracing, where `get_mesh` raises.

    Args:
        vec: Vector to place.
        divisible: States whose row count must divide the device count, or None to skip that check.

    Raises:
        ValueError: If ``divisible``'s row count is not a multiple of the device count, or disagrees
            with ``len(vec)``.
    """
    if isinstance(vec, jax.core.Tracer) or (mesh := jax.sharding.get_mesh()).empty:
        return vec
    if divisible is not None:
        # shape[-1], not shape[0]: the kernel broadcasts over a leading batch axis of any size.
        if (n := divisible.shape[0]) != vec.shape[-1]:
            raise ValueError(f"apply_h: vec length {vec.shape[-1]} disagrees with {n} states")
        _check_mesh_divisible(n)
    if jax.typeof(vec).sharding.mesh is mesh.abstract_mesh:
        return vec
    return jax.device_put(vec, jax.sharding.NamedSharding(mesh, PartitionSpec()))


@jax.jit(static_argnames=["matvec"])
def _apply_h_kernel(
    vec: NDArray[np.inexact],
    scanned: tuple[NDArray, ...],
    states: StateList | None,
    matvec: Matvec,
) -> jax.Array:
    r"""Return :math:`Hv`, resolving the per-X-group inputs according to the ``matvec`` name.

    The jitted kernel behind :func:`apply_h`. Kept positional with a static ``matvec`` because
    :mod:`ground_locg` splats ``matvec(vec, *args)``: a ``static_argnames`` entry would never see a
    keyword, so it is bound via :func:`functools.partial` rather than passed through ``args``.
    The name resolution lives in the wrapper, in plain Python, so it costs nothing per call.

    See :func:`apply_h` for the input sets and argument semantics.

    Raises:
        ValueError: If ``states`` is None for a kernel other than ``"tables"``.
    """
    if matvec != "tables" and states is None:
        raise ValueError(f"states is required for matvec={matvec!r}")

    def fn(out, val):
        # val[0] is the X source for this group: either the precomputed index array or the X
        # signature it is derived from.
        xsource = get_xsource(val[0], states) if matvec == "onthefly" else val[0]
        diagonal = val[1] if matvec == "tables" else get_diagonal(val[1], val[2], states)
        return out + apply_xgrp(xsource, diagonal, vec), None

    return jax.lax.scan(fn, jnp.zeros_like(vec), scanned)[0]
