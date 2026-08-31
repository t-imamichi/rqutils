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
next section) and therefore 32-bit (4-byte) integers are used for vector indexing. Caching the sign
bits will require :math:`\kappa N` bytes, where

.. math::

    \kappa = \lceil \frac{\sum_{j} K^{(j)}}{8} \rceil

is the number of bytes required to pack the sign bits for each vector entry. This "dense" packing
however may cause inefficiencies in computation that defeats the purpose of caching. For a more
straightforward caching, the memory requirement is inflated to :math:`\bar{\kappa} J N` bytes, where

.. math::

    \bar{\kappa} = \lceil \frac{\max_{j} K^{(j)}}{8} \rceil.

If the entire composition of the coefficients are cached instead of the sign bits, :math:`8 J N` or
:math:`16 J N` bytes are used, depending on whether the vector is real or complex (i.e., if there
are terms in the Hamiltonian with odd numbers of Ys).

Given that identifying :math:`[j^{i}]` is an expensive operation, while the other diagonal
operations are not, the default behavior of this function is to cache only the source indices.
Note however that there are cases when further caching can actually be advantageous also in terms of
memory. This is because :math:`S` will not be used after caching both the source indices and sign
bits / coefficient sums. Since :math:`S` occupies :math:`\lceil n/8 \rceil N` bytes of memory,
caching setting should be adjusted according to the values of :math:`n` and :math:`\{K^{(j)}\}_j`.

**How expensive, concretely: the source-index setup dominates the solve, so this is not a symmetric
memory-for-speed dial.** Weighted by call count, the :math:`J`-fold :func:`get_xsource` precompute
measured **66-97%** of an entire solve -- 97.5% at 10 iterations, 66.4% at 200 (3064 ms of setup
against 8.35 ms per matvec iteration, N=200k, J=50; see ``docs/scaling-pocs.md``). Turning source-index
caching *off* therefore pays that cost once per matvec rather than once per solve, which is a far
larger effect than the :math:`4 J N` bytes it reclaims -- measured end-to-end at N=3k, n=12, J=23,
``(0, 2)`` is 10.9x slower than ``(1, 2)`` and ``(0, 0)`` is 7.2x slower than ``(1, 0)``, all four
returning the same energy. Prefer ``cache_level[0] = 1`` unless the memory genuinely will not fit; the
diagonal axis is where the real memory-versus-speed judgement lies.

**On that axis, ``cache_level[1] = 1`` is dominated -- prefer 0 or 2.** It caches ``diag_signs``, one
bit per (state, Z term), then re-derives the diagonal from it on every matvec. Measured end-to-end at
n=22, N=25k, over :math:`K \in \{16, 64, 128\}`: it is **16-31% slower than ``[1] = 0``** (which
stores nothing) and **2.2-7.6x slower than ``[1] = 2``**, at every :math:`K`, on both settings of
``cache_level[0]``, all six levels returning the same energy. The mechanism is that unpacking the
cached bits costs about what recomputing the parity does (3.16 ms against 2.83 ms per group at
:math:`K = 128`), so the cache is paid for in bytes and then largely redone in time.

It is also *memory*-dominated once :math:`\lceil K/8 \rceil` reaches the coefficient itemsize --
:math:`K \geq 64` for float64, :math:`K \geq 128` for complex128 -- since it stores
:math:`\lceil K/8 \rceil` bytes per state per group against level 2's fixed 8 or 16. That crossover is
exact arithmetic, not a fit. Below it level 1 is the smaller array, which is its only remaining claim,
and ``[1] = 0`` is smaller still at zero. Level 1 is retained because the ``cache_level`` sweep is
load-bearing in the test suite, not because a caller should select it.

When it will not fit, ``xcache_groups`` caches ``J'`` of the ``J`` X groups instead of all or none --
see :func:`sqd`. Two caveats measured after it shipped: an intermediate ``J'`` can *raise* peak memory,
since the split runs two matvec kernels, and on a Hamiltonian with many Z signatures per X group the
diagonal arrays dominate so ``cache_level[1]`` is the larger lever. ``NOTES.md`` has both.

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

When the source indices are cached but neither the sign bits nor the diagonals are, the state list
:math:`S` will also be sharded after the computation of the source indices are done.

SQD API
=======

.. autofunction:: sqd
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
from typing import TYPE_CHECKING, Any, Literal, overload

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import PartitionSpec, get_abstract_mesh
from numpy.typing import DTypeLike, NDArray
from scipy.sparse import coo_array, csr_array

from rqutils.ground_locg import (
    _check_prefilter,
    _check_tol,
    ground_locg,
    residual_floor,
)
from rqutils.paulis.symplectic import PauliSumXZ

LOG = logging.getLogger(__name__)
# Largest subspace size representable in the int32 indices used throughout (uniquify_states' iota,
# get_xsource's output). Documented in the module docstring as the hard scaling limit; enforced in
# sqd() and hproj() so an overflow raises instead of silently permuting the subspace.
_MAX_STATES = 2**31 - 1

if TYPE_CHECKING:
    from qiskit.quantum_info import SparsePauliOp

# All three arms must be named here, not added by a later `HamiltonianInput |= SparsePauliOp`: a `type`
# statement is evaluated statically, so the augmented assignment leaves the arm invisible to a checker.
# Safe without a runtime branch because the statement is lazy -- nothing reads `__value__`, so the
# TYPE_CHECKING-only import is never resolved. Both pinned by TestHamiltonianInputIsCheckable.
type HamiltonianInput = PauliSumXZ | tuple[Sequence[str], Sequence[Number]] | SparsePauliOp
type Vector = np.ndarray[tuple[int], np.dtype[np.inexact]]
type StateList = np.ndarray[tuple[int, int], np.dtype[np.uint8]]


# Which dtype kind each `apply_h` keyword's array must have. `u` is an unsigned integer (packed
# bytes), `i` a signed integer (positions, which use -1 as an absent marker), `fc` float or complex.
# This is the discriminator a shape check cannot provide: at n=15 with a 2-state subspace, X
# signatures and X sources are both (2, 2), but never both uint8.
_ARRAY_ROLE_KINDS = {
    "xsignatures": ("u", "packed X signatures (uint8, from PauliSumXZ)"),
    "xsources": ("i", "X source indices (int32, from get_xsource; -1 marks absent)"),
    "zsignatures": ("u", "packed Z signatures (uint8, from PauliSumXZ)"),
    "diag_signs": ("u", "packed diagonal sign bits (uint8, from get_diag_signs)"),
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


def _check_xcache_groups(xcache_groups: Any, cache_level: tuple[int, int], num_groups: int) -> None:
    """Raise unless ``xcache_groups`` is ``None`` or a group count legal for ``cache_level``.

    ``xcache_groups`` caches the source indices of the first ``J'`` X groups and recomputes the rest,
    a memory-for-speed dial between the two settings of ``cache_level[0]``. The cache array is
    ``4 * J' * states_size`` bytes and the uncached groups measured **59.8x** slower per matvec at
    n=100, N=200k, J=16 (``NOTES.md``), so the intermediate values exist to buy back memory without
    paying that whole factor.

    **How much it is worth depends on ``K``, the number of Z signatures per X group, and that is a
    property of the Hamiltonian rather than of the subspace.** On 1D Heisenberg at n=100 (``J = 101``,
    ``K = 100``) the source cache is 404 bytes per state slot against ``diag_signs``' 1313, so it is
    22% of the ``(1, 0)`` footprint and 5-8% of ``(1, 1)``'s -- the *diagonal* axis is the expensive
    one there, and ``cache_level[1]`` the larger lever. At small ``K`` the proportions reverse and this
    dial is the dominant one. See "Measured on a real n=100 Hamiltonian" in ``NOTES.md`` before sizing
    anything from ``4 * J * N`` alone.

    Three ways a bad value would otherwise pass silently:

    * with ``cache_level[0] == 0`` there is no cache to make partial, so any value is a no-op that
      *reads* as "partial caching does not help on my problem" -- the misdiagnosis an A/B-able option
      cannot afford. Rejected rather than ignored.
    * out of range, it would clamp: ``J' > J`` behaves as ``J`` and a negative as ``0``, both
      returning the right answer at the wrong cost.
    * ``True``/``False`` would slice as ``1``/``0`` through Python's bool-is-int rule, so ``True``
      would cache exactly one group -- mirroring the keyword-only change that fixed the same hazard
      for ``states_size``.

    ``None`` means "follow ``cache_level``" and is the only value that reproduces the pre-existing
    graph exactly; ``xcache_groups == num_groups`` is equivalent but traces as the partial path.
    """
    if xcache_groups is None:
        return
    if isinstance(xcache_groups, bool) or not isinstance(xcache_groups, int):
        raise TypeError(
            f"`xcache_groups` must be None or an int in [0, {num_groups}], got {xcache_groups!r}"
        )
    if cache_level[0] != 1:
        raise ValueError(
            f"`xcache_groups` is {xcache_groups} but `cache_level` is {cache_level}: there is no "
            "source-index cache to make partial when the first digit is 0. Either set "
            "`cache_level=(1, ...)` to enable caching, or drop `xcache_groups`."
        )
    if not 0 <= xcache_groups <= num_groups:
        raise ValueError(
            f"`xcache_groups` is {xcache_groups}, but this Hamiltonian has {num_groups} X groups, so "
            f"it must be in [0, {num_groups}]. Out of range it would clamp rather than raise: a "
            "larger value caches everything and a negative caches nothing, both returning the right "
            "answer at the wrong cost."
        )


def _residual_floor_of(hamiltonian: PauliSumXZ) -> float:
    """The achievable eigen-residual floor for this Hamiltonian, for guards and error messages."""
    return residual_floor(float(np.abs(hamiltonian.c).sum()), hamiltonian.c.dtype)


def _check_cache_level(cache_level: Any) -> None:
    """Raise unless ``cache_level`` is one of the six ``(source_indices, diagonals)`` pairs.

    Every branch on ``cache_level`` in this module is an equality test with an implicit ``else``, so
    an out-of-range value used to be absorbed rather than reported:

    * an out-of-range **first** digit was silently ignored -- ``(2, 0)`` behaved exactly as
      ``(0, 0)``, returning the same energy at 7.2x the cost;
    * an out-of-range **second** digit surfaced as ``UnboundLocalError`` on an internal variable,
      i.e. an internal error escaping a public entry point.

    The likelier mistake is the **transposition**, which validation cannot catch: ``(0, 1)`` and
    ``(1, 0)`` are both legal and return the same energy, differing only in cost (``(0, 2)`` measures
    10.9x slower than ``(1, 2)``, ``(0, 0)`` 7.2x slower than ``(1, 0)``), so it reads as "SQD is
    slow" rather than as an error. What this can do is name the axes in the message, so a reader of
    the call site can tell which digit is which.

    Kept as a tuple rather than split into two enum parameters: ``cache_level`` is bound **static**
    into the jit'd kernel through :func:`functools.partial`, because :func:`rqutils.ground_locg`
    splats ``args`` positionally and ``static_argnames`` would never see it. The validation belongs
    at the public boundary, not in that plumbing.

    Args:
        cache_level: The caller's value, unvalidated.

    Raises:
        TypeError: If it is not a length-2 sequence of ints.
        ValueError: If either digit is out of range -- ``source_indices`` in ``{0, 1}``,
            ``diagonals`` in ``{0, 1, 2}``.
    """
    try:
        source_indices, diagonals = cache_level
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"`cache_level` must be a (source_indices, diagonals) pair of ints, got {cache_level!r}"
        ) from exc
    # `bool` is an `int` subclass, so (True, 0) would otherwise pass as (1, 0). That is the same
    # True == 1 confusion `sqd`'s keyword-only change closed one level up, so reject it here too.
    if not all(isinstance(d, int) and not isinstance(d, bool) for d in (source_indices, diagonals)):
        raise TypeError(
            f"`cache_level` must be a (source_indices, diagonals) pair of ints, got {cache_level!r}"
        )
    if source_indices not in (0, 1) or diagonals not in (0, 1, 2):
        raise ValueError(
            f"`cache_level` is {cache_level!r}, but the first entry (source_indices: 0=recompute "
            "per matvec, 1=cache) must be 0 or 1 and the second (diagonals: 0=no caching, "
            "1=cache sign bits, 2=cache diagonals) must be 0, 1 or 2. Note the two axes are not "
            "interchangeable: caching source indices is near-free to enable and very expensive to "
            "disable, so a transposed pair is legal but runs 7.2-10.9x slower."
        )


def _check_zsignatures_rank(zsignatures: Any) -> None:
    """Raise unless ``zsignatures`` is one X group's 2-D ``(num_zterms, num_bytes)`` array.

    Shared by :func:`get_diag_signs` and :func:`get_diagonal`, which are both public, both consume the
    same array, and both index its leading axis -- so handed a 1-D array they read *scalars* and
    silently return a wrongly shaped result rather than raising. Measured on ``get_diagonal``:
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


# Overloads so a caller destructuring the 3-tuple does not have to narrow first. `return_eigvec` is a
# plain bool, so without these the declared return is the full union and every
# `eigval, eigvec, subdims = sqd(...)` reads as unpacking a `float`. Annotation only -- the
# implementation signature below is unchanged, and sphinx documents that one.
@overload
def sqd(
    hamiltonian: HamiltonianInput,
    states: StateList,
    *,
    states_size: int | None = ...,
    return_eigvec: Literal[True] = ...,
    packed: bool = ...,
    cache_level: tuple[int, int] = ...,
    xcache_groups: int | None = ...,
    maxiter: int = ...,
    tol: float | None = ...,
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
    cache_level: tuple[int, int] = ...,
    xcache_groups: int | None = ...,
    maxiter: int = ...,
    tol: float | None = ...,
    prefilter: tuple[int, int] | None = ...,
) -> float: ...


def sqd(
    hamiltonian: HamiltonianInput,
    states: StateList,
    *,
    states_size: int | None = None,
    return_eigvec: bool = True,
    packed: bool = False,
    cache_level: tuple[int, int] = (1, 0),
    xcache_groups: int | None = None,
    maxiter: int = 1000,
    tol: float | None = None,
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

    Cache level is a 2-tuple where the first element specifies the caching of the source indices
    (0=no caching, 1=cached) and the second specifies the caching of the diagonal elements (0=no
    caching, 1=cache sign bits, 2=cache diagonals).

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
            62.4% to 3.3%. At N=24M with ``cache_level=(0, 2)`` that is **7.7 GB**. See ``NOTES.md``,
            "``states_size``'s power-of-two padding".
        return_eigvec: Whether to return the eigenvector (coefficients and unique state bitstrings).
        maxiter: Maximum LOBPCG iterations. **Non-convergence now raises** rather than returning
            the iteration cap's best guess -- see ``Raises``.
        tol: **Absolute** bound on the eigen-residual :math:`\|Hv - Ev\|_2`, forwarded to
            :func:`rqutils.ground_locg.ground_locg`. ``None`` derives the achievable floor from the
            operator.

            .. warning::

               **This changed meaning.** ``tol`` was a *relative* tolerance, scaled internally by
               :math:`(\|Hv\| + |E|)\,N \cdot 10`. It is now the absolute residual itself, so an
               existing call passing an explicit ``tol`` **keeps working but changes criterion** and
               nothing raises -- at :math:`N = 2\times10^5` the old ``tol=1e-6`` admitted a residual
               of order :math:`10^0` where it now demands :math:`10^{-6}`. Re-derive any empirically
               tuned value.

               ``None`` still converges, but it is **not** unchanged: the default is 3-4 decades
               tighter than the old one and measured **1.18-1.49x slower**, the gap growing with
               :math:`N` (the old default's effective bound carried an :math:`N` factor; this one does
               not). Pass an explicit ``tol`` if the extra decades are not wanted --
               ``docs/rqutils-tol-response.md`` §5.1 has the table.

            A caller with a residual requirement can now state it directly: ``tol=1e-6`` means
            :math:`\|Hv - Ev\| < 10^{-6}`, at any :math:`N`. Values below the floor are **rejected**
            (see ``Raises``) rather than silently clamped or left to exhaust ``maxiter``; the floor is
            :math:`4\,\varepsilon \sum_k |c_k|`, available as
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
        cache_level: Switches for caching the results of source indices and sign bits / diagonals.
            See the module documentation for the detailed discussion of the resource tradeoff involved.
        xcache_groups: Number of X groups whose source indices to cache, or ``None`` (default) for all
            of them. A memory-for-speed dial between the two settings of ``cache_level[0]``: the cache
            array is ``4 * J' * states_size`` bytes, and the groups left out of it are searched inside
            every matvec, measured **59.8x** slower per matvec at n=100. Requires
            ``cache_level[0] == 1``. ``None`` and the full group count give the same answer, but only
            ``None`` traces the single-arm graph.

            **Two things to know before reaching for it.** Dropping groups shrinks the cache array
            linearly, but it does *not* shrink the footprint by as much: everything else -- the state
            list and the solver's vectors -- stays. On 1D Heisenberg at n=100 (``J = 101``,
            ``K = 100``, ``N = 24M``) the whole ``(1, 0)`` footprint measures 16.5 GB of which the
            cache is 13.6 GB, so ``xcache_groups=0`` saves 12.4 GB rather than "everything"; at
            ``cache_level[1] == 1`` the same 12.4 GB is only 20% of a 60.6 GB footprint, because at
            high ``K`` the *diagonal* arrays dominate and ``cache_level[1]`` is the larger lever.

            And an intermediate count can **raise** peak memory rather than lower it: the split runs
            two matvec kernels instead of one, and below a break-even in ``J`` the second kernel's
            intermediates cost more than the cache saves. Measured at ``J = 16``, ``N = 28k``: 9.0 MB
            for the full cache against 10.4 MB at ``J' = 8``. ``xcache_groups=0`` always saves (that
            arm is single-kernel); intermediate values pay off once ``4 * J * states_size`` is large
            next to one kernel's working set, which is the regime this exists for.

            ``floor(budget / (4 * states_size))`` sizes the *cache* to a budget exactly. Size it that
            way and then **measure** both peak memory and speed: the time does not follow a linear
            model closely enough to promise (17.5% off at the midpoint), and neither does the peak.
            ``NOTES.md`` has the measurements.
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

            That 1.49x is **below** the 1.88x median ``docs/locg-chebyshev-prefilter.md`` measured on
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
            minimum -- indistinguishable from a correct result by inspection. ``docs/locg.md`` records
            that this absence "is the reason I4 could hide", a sign error that made the convergence
            test unsatisfiable so the solver silently never converged. Raise ``maxiter`` or loosen
            ``tol`` to proceed.
        ValueError: If ``states_size`` is smaller than ``states.shape[0]``, or if it exceeds
            :math:`2^{31} - 1`, the ceiling imposed by the int32 indices used for subspace positions
            (beyond it an index wraps negative and the subspace is silently permuted); or if either
            ``prefilter`` entry is negative -- see :func:`rqutils.ground_locg._check_prefilter`, which explains why a
            negative value would otherwise be absorbed as a silent no-op; or if ``tol`` is
            non-positive or below the achievable eigen-residual floor
            :math:`4\,\varepsilon\sum_k|c_k|`, which no solve could reach.
        TypeError: If ``cache_level`` is not a pair of ints, or ``prefilter`` is neither None nor a
            ``(degree, cycles)`` pair of ints.
    """
    _check_cache_level(cache_level)
    _check_prefilter(prefilter)
    if states_size is None:
        # Default to the next power of two at or above the input length. All-distinct and growing
        # dimensions are the *normal* SQD access pattern -- an SKQD run walks one per Krylov rung plus
        # one per recovery round -- so a default of states.shape[0] pins no shape and retraces the
        # solver every call. Bucketing collapses that to O(log N) traces: measured 1.25x over five
        # dimensions 60..260 at n=10 and 1.43x over five rungs at n=13, energies bit-identical.
        # Padding is unobservable (filler is excluded from the projection and trimmed from the
        # returned basis); pass states.shape[0] explicitly for the old behaviour.
        states_size = 1 << max((states.shape[0] - 1).bit_length(), 1)
    if states_size < states.shape[0]:
        raise ValueError("states_size smaller than the states array length")
    # The 2^31 ceiling the module docstring calls a hard limit, actually enforced. Subspace positions
    # are int32 throughout -- uniquify_states' iota, and get_xsource's returned indices with -1 as the
    # absent marker -- so a size at or above 2^31 wraps to a negative index and yields a corrupted
    # permutation rather than an error: a plausible finite answer, the failure mode this module keeps
    # guarding against. Note a wrapped index is -2147483648, not -1, so the absent-marker test cannot
    # even catch it. Unreachable on current hardware (2^31 states is already 4.3 GB of packed states
    # before any vector), which is exactly why the check is cheap insurance rather than a cost.
    if states_size > _MAX_STATES:
        raise ValueError(
            f"states_size {states_size} exceeds the {_MAX_STATES} limit imposed by int32 subspace "
            "indexing; see the scaling-limits section of the module documentation"
        )
    if not isinstance(hamiltonian, PauliSumXZ):
        hamiltonian = PauliSumXZ.from_paulisum(hamiltonian)
    # `tol` is an absolute eigen-residual bound, so whether it is reachable depends on the operator's
    # scale. Checked here and not in `run_sqd` because that is jitted -- sum|c_k| is traced there, and
    # a traced value cannot raise. This is the outermost point where it is concrete, and it must come
    # after the PauliSumXZ conversion above, which is what supplies `.c`.
    _check_tol(tol, float(np.abs(hamiltonian.c).sum()), hamiltonian.c.dtype)
    states = _check_states_shape(states, hamiltonian.num_qubits, packed)

    if not (mesh := get_abstract_mesh()).empty and (resid := states_size % mesh.size) != 0:
        LOG.debug("Adjusting states_size to make the array divisible by %d", mesh.size)
        states_size += mesh.size - resid

    # `packed=True` hands over `pack_states`' own output, so skip the pack rather than repeat it --
    # it is not idempotent, and a second pass reads each byte as one bit. The saving is the caller's:
    # unpacked `[N, num_qubits]` is ~7.7x the packed width at n=100, and both arrays are live at once
    # during the pack, so a caller already holding the packed form was paying an 8x expansion plus a
    # transient peak (2.40 GB against 0.31 GB at n=100, N=24M) for nothing. Nothing downstream
    # changes: `run_sqd` has always taken the packed form.
    states_p = states if packed else PauliSumXZ.pack_states(states)
    # Pad the *input* up to states_size too, not just the internal arrays. states_p is a traced
    # argument of run_sqd, so its leading dimension is part of the jit cache key: leaving it at the
    # raw input length retraces the whole solver on every distinct len(states), which is precisely
    # what states_size exists to prevent (measured 0.44 s per call versus 0.064 s once the shape
    # repeats). uniquify_states already pins every array it derives, so this was the one shape that
    # still leaked the input length through.
    #
    # 255 is the correct filler, and for the same reason uniquify_states uses it: an all-ones row
    # sorts to the end of the lexsort, and its high bit in byte 0 is what _is_filler tests. A genuine
    # state can never collide with it, because PauliSumXZ.pack_states' pad bit forces byte 0 < 128.
    if (deficit := states_size - states_p.shape[0]) > 0:
        states_p = np.append(
            states_p, np.full((deficit, states_p.shape[1]), 255, dtype=np.uint8), axis=0
        )

    LOG.debug("Starting SQD with array size %s", states_size)
    start = time.time()
    result = run_sqd(
        hamiltonian,
        states_p,
        states_size,
        return_eigvec,
        cache_level,
        xcache_groups=xcache_groups,
        maxiter=maxiter,
        tol=tol,
        prefilter=prefilter,
    )
    LOG.info("Found ground eigenpair in %f seconds.", time.time() - start)
    eigval = float(result[0])
    # The convergence flag used to be discarded here, and a non-converged run still returns
    # `state.theta` -- a valid variational *upper bound*, so finite, real, and above the true minimum,
    # i.e. indistinguishable from a correct answer by inspection. docs/locg.md records that this
    # absence "is the reason I4 could hide": a sign error made the convergence test unsatisfiable, so
    # the solver silently never converged and every answer was the iteration cap's best guess.
    #
    # Raised here rather than in `run_sqd` because that function is @jax.jit-wrapped, so `converged`
    # is a traced boolean there and cannot be branched on at trace time.
    if not bool(result[-1]):
        raise RuntimeError(
            f"LOBPCG did not converge in maxiter={maxiter} iterations (tol={tol!r}). The value it "
            f"reached, {eigval!r}, is a variational upper bound rather than the ground energy -- "
            "finite and plausible, which is why this raises instead of returning it. Raise `maxiter` "
            "first: a near-degenerate ground state converges in the eigenVALUE long before the "
            "residual test is satisfied, so this often means the default cap was simply too low "
            "rather than that anything is wrong. Measured on a 37-state subspace with a relative gap "
            "of 5.5e-04, theta was already correct to 4e-16 by iteration 500 while the residual only "
            "crossed the threshold at 1091. Loosening `tol` is the other lever -- it is an absolute "
            f"residual bound, and the floor for this operator is {_residual_floor_of(hamiltonian):.3e}, "
            "so any value above that is reachable in principle. A genuinely ill-conditioned subspace "
            "is the rarer cause."
        )
    if return_eigvec:
        eigvec, states_u, subspace_dim = result[1:-1]
        # `packed` now governs BOTH directions: a caller who hands over packed states gets packed
        # states back, so a round trip through `sqd` needs no re-pack. The returned rows are the same
        # array `run_sqd` searched, sliced to the genuine uniques.
        #
        # Deliberately not a separate `return_packed` flag. Two independent flags make four
        # combinations of which two are round trips and two are conversions, and `sqd` is not a
        # conversion utility -- `PauliSumXZ.pack_states`/`unpack_states` are, and a caller wanting
        # the other width calls one of them. One flag keeps input and output in the same convention
        # by construction.
        #
        # `hamiltonian.num_qubits`, not `states.shape[1]`: the latter is the *packed* width on the
        # `packed=True` path, which would unpack to the wrong qubit count.
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
        ValueError: If ``unique_states=True`` and ``states`` is not strictly increasing in
            lexicographic order (unsorted, or containing duplicate rows); or if the subspace exceeds
            :math:`2^{31} - 1` states, the ceiling imposed by the int32 indices
            :func:`get_xsource` returns.
    """
    if not isinstance(hamiltonian, PauliSumXZ):
        hamiltonian = PauliSumXZ.from_paulisum(hamiltonian)
    states = _check_states_shape(states, hamiltonian.num_qubits)
    # Same int32 ceiling sqd() enforces, since hproj reaches get_xsource too and its returned
    # positions are int32 with -1 as the absent marker. Checked here, before the O(N) sortedness scan
    # and the np.unique below: it is an O(1) look at a shape, so it costs nothing to do first and
    # reports the real problem rather than letting a doomed call spend time first (measured on the
    # test that reaches it: 0.23 s with the check first, 23 s when it sits after the scan).
    if states.shape[0] > _MAX_STATES:
        raise ValueError(
            f"subspace of {states.shape[0]} states exceeds the {_MAX_STATES} limit imposed by int32 "
            "subspace indexing; see the scaling-limits section of the module documentation"
        )
    if not unique_states:
        states = np.unique(states, axis=0)
    else:
        # get_xsource binary-searches into `states`, so unsorted rows silently produced a wrong,
        # non-symmetric matrix. Validated rather than documented away, at 12-14% of hproj and only on
        # this opt-in path -- the branch above is sorted by construction and pays nothing.
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
    # shape= is mandatory here, not cosmetic: without it scipy infers the extent from the largest
    # index present, so a trailing basis state that no term couples into is dropped and the matrix
    # comes back too small (measured 41x41 for a 53-state subspace with local two-site js
    # operators). That truncation is silent -- the result is still symmetric, so eigvalsh returns a
    # plausible wrong ground energy. When nothing at all survives, the index arrays are empty and
    # scipy cannot infer any extent, raising instead ("cannot infer dimensions from zero sized
    # index arrays"); the projection is legitimately the zero matrix in that case.
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
        # `ham` is a PackedArrays, so these read by name. As a bare tuple this was
        # `ham[0]`/`ham[1]`/`ham[2]`, where swapping the first two type-checks and silently
        # computes with X and Z exchanged -- `x` and `z` are same-dtype integer arrays.
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
    # Two xorshift-multiply rounds (the constants are Murmur-style mixers): enough that consecutive
    # indices give uncorrelated outputs, which is all that is needed to avoid an accidental
    # orthogonality against any one eigenvector.
    mixed = index ^ (index >> 16)
    mixed = mixed * jnp.uint32(0x7FEB352D)
    mixed = mixed ^ (mixed >> 15)
    mixed = mixed * jnp.uint32(0x846CA68B)
    mixed = mixed ^ (mixed >> 16)
    # Map to [-1, 1). The distribution does not matter, only that no entry is systematically zero.
    vec = mixed.astype(dtype) * (2.0 / float(2**32)) - 1.0
    # Reshard the predicate to match `vec` rather than assuming it already does. `vec` is built
    # sharded unconditionally (the iota above takes `out_sharding`), but `states_u`'s sharding depends
    # on the caller: `run_sqd` reshards it only inside `if cache_level[0] == 1`, because the
    # uncached branch still needs the replicated array for the `get_xsource` searches. So at
    # cache_level[0] == 0 the predicate arrives replicated while `vec` is partitioned, and
    # `jnp.where` rejects the pair -- "select `which` must be scalar or have the same sharding as
    # cases". That raised on every mesh for (0, 0), (0, 1) and (0, 2), the three levels no sharding
    # test covered; it was previously masked by _accumulate_diagonal's rank bug, which failed all six
    # earlier in the call. Fixed here rather than by resharding states_u in the caller: this function
    # is what decides `vec`'s sharding, so it owns the requirement that the mask agree with it.
    filler = _is_filler(states_u) == 1
    if sharding is not None:
        filler = jax.reshard(filler, sharding)
    return jnp.where(filler, jnp.zeros_like(vec), vec)


@jax.jit(
    static_argnames=[
        "states_size",
        "return_eigvec",
        "cache_level",
        "xcache_groups",
        "maxiter",
        "prefilter",
        "log_level",
    ]
)
def run_sqd(
    hamiltonian: PauliSumXZ,
    states_p: StateList,
    states_size: int,
    return_eigvec: bool,
    cache_level: tuple[int, int] = (1, 0),
    xcache_groups: int | None = None,
    maxiter: int = 1000,
    tol: float | None = None,
    prefilter: tuple[int, int] | None = (32, 2),
    log_level: int = logging.INFO,
) -> tuple[float, bool] | tuple[float, jax.Array, jax.Array, int, bool]:
    """JIT-compiled part of the SQD function.

    Returns the eigenvalue, optionally the eigenvector/basis/dimension, and **the convergence flag as
    the last element**. The flag used to be discarded here, which is what let a non-converged
    ``theta`` -- a valid variational upper bound, so finite and plausible -- reach the caller as the
    answer. It is returned rather than checked because this function is ``@jax.jit``-wrapped, so
    ``converged`` is a traced boolean and cannot be branched on at trace time; :func:`sqd` raises on
    it once the value is concrete.

    Args:
        maxiter: Maximum LOBPCG iterations, forwarded to :func:`rqutils.ground_locg.ground_locg`.
            Static, as it is there.
        tol: Absolute bound on the eigen-residual ``||Hv - Ev||``, forwarded to
            :func:`rqutils.ground_locg.ground_locg`. ``None`` derives the achievable floor
            from the operator. Validated in :func:`sqd`, which is where ``sum|c_k|`` is
            concrete -- this function is jitted, so it cannot raise on it.
        prefilter: Optional ``(degree, cycles)`` Chebyshev prefilter, forwarded to
            :func:`rqutils.ground_locg.ground_locg`. Static, as it is there -- passed by keyword, so
            unlike ``cache_level`` it needs no :func:`functools.partial` binding. See :func:`sqd` on
            why this option's published speedups do not transfer to this path.
    """
    # `cache_level` is static, so this is a concrete tuple at trace time and the check runs once per
    # trace rather than once per call. `sqd` validates too; this covers the direct callers, which are
    # the six examples/scaling scripts -- i.e. the ones most likely to pass an experimental value.
    _check_cache_level(cache_level)
    _check_xcache_groups(xcache_groups, cache_level, hamiltonian.x.shape[0])
    _check_prefilter(prefilter)
    sharding = None
    if not (mesh := get_abstract_mesh()).empty:
        sharding = PartitionSpec(mesh.axis_names)

    if log_level <= logging.DEBUG:
        jax.debug.print("Uniquifying states (size {})", states_size)

    states_u = uniquify_states(states_p, states_size)

    # `xcache_groups` splits the X groups in two: the first `ncached` get precomputed source indices,
    # the rest keep their raw signatures and are searched inside every matvec. `None` means "all",
    # which is the pre-existing behaviour and the only value that traces the single-arm graph.
    njgroups = hamiltonian.x.shape[0]
    ncached = njgroups if xcache_groups is None else xcache_groups
    partial_xcache = cache_level[0] == 1 and ncached < njgroups

    if cache_level[0] == 1:
        if log_level <= logging.DEBUG:
            jax.debug.print("Precomputing xsources for {n} of {j} X groups", n=ncached, j=njgroups)

        xsources = jax.lax.scan(
            lambda _, x: (None, get_xsource(x, states_u)), None, hamiltonian.x[:ncached]
        )[1]
        if sharding and not partial_xcache:
            # We will not be performing sorts on states any more - shard the array.
            #
            # Skipped when the cache is partial: the uncached groups still search `states_u` inside
            # every matvec, and `get_xsource` requires it **replicated** -- a partitioned `[N, B]`
            # fails outright ("Unmapped values passed to vmap cannot be sharded along the mesh axis
            # you are vmapping over"). Resharding here would turn a memory dial into a crash on any
            # mesh, which is why the condition is `not partial_xcache`.
            if log_level <= logging.DEBUG:
                jax.debug.print("Sharding states array")

            states_u = jax.reshard(states_u, sharding)

    if cache_level[1] == 1:
        if log_level <= logging.DEBUG:
            jax.debug.print("Precomputing sign bits of diagonals")

        diag_signs = jax.lax.scan(
            lambda _, z: (None, get_diag_signs(z, states_u)), None, hamiltonian.z
        )[1]
    elif cache_level[1] == 2:
        if log_level <= logging.DEBUG:
            jax.debug.print("Precomputing diagonals")

        diagonals = jax.lax.scan(
            lambda _, v: (None, get_diagonal(v[0], v[1], states_u)),
            None,
            (hamiltonian.z, hamiltonian.c),
        )[1]

    # Assemble the per-X-group arrays apply_h scans over. This stays a Python-level branch on the
    # static cache_level: the *packing* must be static too, or the tuple structure would become part
    # of the traced arguments and retrace on every call.
    #
    # Selected with if/elif rather than a dict literal keyed on cache_level[1]: `diag_signs` and
    # `diagonals` are assigned only inside their own branches above, so a dict literal -- which
    # evaluates every value before indexing -- raises UnboundLocalError on the levels that skipped
    # them. The layout itself is shared with apply_h through _pack_scanned; only the choice of which
    # array to hand it is local.
    xgroup = xsources if cache_level[0] == 1 else hamiltonian.x
    if cache_level[1] == 0:
        diagonal_arg = hamiltonian.z
    elif cache_level[1] == 1:
        diagonal_arg = diag_signs
    else:
        diagonal_arg = diagonals
    scanned = _pack_scanned(cache_level, xgroup, diagonal_arg, hamiltonian.c)
    # A partial cache needs a *second* scanned tuple: the cached and uncached groups carry different
    # X arrays (int32 indices against uint8 signatures), so they cannot share one leading axis. The
    # matvec becomes the sum of two kernels, one per arm -- verified exact against the single-arm form
    # for all six cache_levels at every J' (tests/test_sqd.py::TestPartialXCache).
    #
    # The diagonal axis is sliced identically in both arms and is orthogonal to the split: it is
    # indexed by X group, so group k's diagonal data travels with whichever arm holds group k.
    scanned_tail = None
    if partial_xcache:
        scanned = _pack_scanned(
            cache_level, xgroup, diagonal_arg[:ncached], hamiltonian.c[:ncached]
        )
        scanned_tail = _pack_scanned(
            (0, cache_level[1]),
            hamiltonian.x[ncached:],
            diagonal_arg[ncached:],
            hamiltonian.c[ncached:],
        )
    # (1, 2) reads neither signature array, so it needs no states at all -- which is what lets the
    # caller drop S entirely under the most aggressive caching (see the module docstring).
    # `partial_xcache` forces states back on: the uncached arm searches them every matvec.
    needs_states = cache_level[0] == 0 or cache_level[1] == 0 or partial_xcache
    # cache_level is bound here rather than passed through args: ground_locg splats args
    # positionally (matvec(vec, *args)), so a static_argnames entry would never see it and the
    # tuple would be traced -- retracing the kernel on every matvec call in the solver loop.
    if partial_xcache:
        # Both arms' cache_levels are bound statically for the same reason, and the two kernels are
        # summed rather than fused: see the `scanned_tail` comment above. `args` gains the tail tuple,
        # so it stays traced data while both cache_levels stay static.
        def matvec(  # type: ignore[misc]
            vec: jax.Array,
            scanned: tuple,
            states: StateList | None,
            scanned_tail: tuple,
            _head: tuple[int, int] = cache_level,
            _tail: tuple[int, int] = (0, cache_level[1]),
        ) -> jax.Array:
            head = _apply_h_kernel(vec, scanned, states, cache_level=_head)
            return head + _apply_h_kernel(vec, scanned_tail, states, cache_level=_tail)

        args = (scanned, states_u, scanned_tail)
    else:
        matvec = functools.partial(_apply_h_kernel, cache_level=cache_level)
        args = (scanned, states_u if needs_states else None)

    def vinit_from_min_diag():
        if cache_level[1] == 2:
            diagonal = diagonals[0]
        else:
            diagonal = get_diagonal(hamiltonian.z[0], hamiltonian.c[0], states_u)
        # `.real` on both branches, not just the uncached one. A Hermitian operator's projected
        # diagonal is real by construction, but `diagonals` carries `hamiltonian.c`'s dtype, which is
        # complex128 whenever any Pauli string has an odd Y count (the folded `(-i)^{x.z}` phase makes
        # it so -- see PauliSumXZ). The `jnp.max`/`argmin` below reject complex input outright, so
        # cache_level[1] == 2 raised `TypeError: lt does not accept dtype complex128` for every such
        # Hamiltonian -- on a single device, with no mesh involved. The uncached branch took `.real`
        # and worked, which is what made the asymmetry easy to miss.
        diagonal = diagonal.real
        # Set the fill-in components to the maximum value so that argmin only sees the valid entries.
        # No reshard needed, unlike `_spread_seed`: `diagonal` derives from `states_u`, so predicate
        # and operand specs track by construction (verified P(None) and P('x') on both).
        diagonal = jnp.where(_is_filler(states_u) == 1, jnp.max(diagonal), diagonal)
        imin = jnp.argmin(diagonal)
        # Weight the minimum-diagonal state heavily -- it is the best single guess available, and
        # keeping it dominant preserves this heuristic's fast convergence -- but add the spread seed
        # underneath rather than returning a bare one-hot.
        #
        # THE WEIGHT CARRIES THE SEED'S OWN SIGN, and that is what stops it cancelling. A plain
        # `.add(1.0)` subtracts where the seed component is negative, and `_spread_seed` maps index 0
        # to *exactly* -1.0, so `argmin(diagonal) == 0` zeroed the component at the very index this
        # heuristic had declared most important -- sqd then returned a wrong eigenvalue with
        # converged=True. Structural rather than a special case on index 0, because near-cancellation
        # is reachable at many indices; the sign form bounds |vinit[imin]| into [1, 2) for any seed.
        # `NOTES.md` has the measurements.
        #
        # A pure one-hot cannot leave its own connected component: Krylov iteration only ever
        # reaches states linked to the seed by a nonzero matrix element, so if the projected
        # Hamiltonian splits into disconnected blocks (routine for a sampled subspace, where whole
        # groups of bitstrings may share no Pauli-induced transition), the solver returns that
        # block's minimum and reports convergence. Measured on a 14-state subspace that splits 4+10:
        # sqd returned -1.293, the exact minimum of the block holding the seed, against a true
        # minimum of -2.191 in the other block. The answer was a genuine eigenvalue, just not the
        # lowest one -- so nothing downstream could detect it.
        #
        # out_sharding is mandatory here, not decorative: a scatter into a sharded operand cannot
        # resolve its output sharding unambiguously (the update might land on any device), so JAX
        # raises ShardingTypeError rather than guessing. Without it, sqd() fails outright on ANY
        # multi-device mesh -- not subtly, but before the solver is ever reached. It went unnoticed
        # because nothing in the suite runs a mesh; `XLA_FLAGS=--xla_force_host_platform_device_count`
        # reproduces it on CPU, which is what examples/scaling/poc7_sharding.py does.
        seed = _spread_seed(states_size, states_u, hamiltonian.c.dtype, sharding)
        # Elementwise under a mask rather than `seed.at[imin].add(...)`, which is the same arithmetic
        # but reads one element out of a *sharded* array: measured on a 4-device mesh, indexing emitted
        # 3 `all-gather`s to fetch that scalar, materializing the whole `states_size` vector on every
        # device. At this module's sizes (`_MAX_STATES` is 2**31 - 1) that is exactly the full-vector
        # collective `ground_locg`'s single-vector memory budget exists to avoid. The mask form emits
        # none, and is bit-identical (verified at several `imin`).
        #
        # jnp.sign, not jnp.copysign: the seed is complex whenever `hamiltonian.c` is, and copysign
        # rejects complex input. sign(z) = z/|z| keeps the phase, so the update reinforces.
        # The zero branch is unreachable below the `_MAX_STATES` ceiling -- `NOTES.md` records why it
        # stays, so do not read a surviving mutant here as dead code.
        sign = jnp.sign(seed)
        direction = jnp.where(sign == 0, 1.0, sign)
        selected = jax.lax.broadcasted_iota(imin.dtype, (states_size,), 0, out_sharding=sharding)
        return seed + jnp.where(selected == imin, direction, jnp.zeros_like(seed))

    def vinit_nodiag():
        # No diagonal to rank states by, so the spread seed is all there is. A one-hot here is not
        # merely suboptimal but wrong: ground_locg's documented precondition is a non-vanishing
        # overlap with v0, and e_0 violates it outright whenever state 0 is decoupled from the rest
        # of the subspace (every xsource from it lands outside, so row 0 of the projected H is
        # identically zero). e_0 is then a true eigenvector with eigenvalue 0, the zero-residual
        # guard correctly reports convergence, and sqd returns 0.0 -- silently, with
        # converged=True. Reproduced on a 9-state IIIX subspace whose true answer is -1.
        return _spread_seed(states_size, states_u, hamiltonian.c.dtype, sharding)

    if log_level <= logging.DEBUG:
        jax.debug.print("Generating vinit")

    vinit = jax.lax.cond(jnp.all(hamiltonian.x[0] == 0), vinit_from_min_diag, vinit_nodiag)

    if log_level <= logging.DEBUG:
        jax.debug.print(f"Starting minimization with cache_level {cache_level}")

    # sum|c_k| bounds lambda_max rigorously -- every Pauli string is unitary, and projecting onto the
    # subspace only shrinks the spectral radius -- and costs no matvec. `ground_locg` cannot derive it
    # from a callable, and raises rather than guessing (docs/rqutils-prefilter-bug.md). Gated on the
    # filter actually running, since degree<=1 or cycles==0 is a documented no-op and computing the
    # bound anyway would add ops to the traced graph for those values.
    filter_runs = prefilter is not None and prefilter[0] > 1 and prefilter[1] > 0
    prefilter_hi = jnp.abs(hamiltonian.c).sum() if filter_runs else None
    eigval, eigvec, _, converged = ground_locg(
        matvec,
        vinit,
        args=args,
        maxiter=maxiter,
        tol=tol,
        prefilter=prefilter,
        prefilter_hi=prefilter_hi,
        log_level=log_level,
    )
    result = (eigval,)
    if return_eigvec:
        if sharding:
            eigvec = jax.reshard(eigvec, PartitionSpec(None))
            states_u = jax.reshard(states_u, PartitionSpec(None))
        subspace_dim = jnp.searchsorted(_is_filler(states_u), 1)
        result += (eigvec, states_u, subspace_dim)
    # Convergence last, so the existing positional unpackings above keep their meaning.
    return result + (converged,)


@jax.jit(static_argnames=["states_size"])
def uniquify_states(states_p: StateList, states_size: int) -> StateList:
    """A stripped-down implementation of jnp.unique.

    The returned array will have shape (states_size, states_p.shape[1]). If states_size is greater
    than the number of unique states, the residual entries at the end are filled with 255.

    Raises:
        ValueError: If ``states_size`` exceeds :data:`_MAX_STATES`, the ceiling imposed by the int32
            iota below.
    """
    # The int32 ceiling checked where the int32 index is actually created, not only in the public
    # entry points. `sqd()` and `hproj()` check it too -- earlier, with better messages, and before
    # their own O(N) work -- but this function and `get_xsource` are un-underscored and are called
    # directly by six scripts under examples/scaling/, which is exactly the code that pushes N. Those
    # call sites reach the iota with neither entry-point guard in the chain.
    #
    # Free: `states_size` is static (see the decorator), so this fires at trace time and costs nothing
    # per call. A traced value could not be compared at all.
    if states_size > _MAX_STATES:
        raise ValueError(
            f"states_size {states_size} exceeds the {_MAX_STATES} limit imposed by the int32 index "
            "below; beyond it the iota wraps negative and the subspace is silently permuted"
        )
    # Lexsort on uint64 words rather than the raw uint8 columns. `lax.sort` compares key operands one
    # at a time, so `num_keys=B` costs O(B) per comparison -- 13 columns at n=100. Packing to
    # ceil(B/8) words makes it 2, and the permutation is identical because `_pack_state_words` is
    # order-preserving. Measured 1.79x at n=30 through 5.06x at n=127 (N=200k); this sort dominates
    # the function, which in turn measured 14-27x `get_xsource` at every width.
    #
    # This trades memory for speed: the words are an extra `[N, ceil(B/8)]` uint64 buffer, and packing
    # rounds B up to a multiple of 8, so it widens the data by `8*ceil(B/8) - B` bytes per row. Worst
    # case is n=64 (B=9 -> 16 bytes, +7/row); n=127 (B=16) is free. Measured 1.09-1.69x compiled temp
    # bytes at N=1M. Negligible at the sizes here (0.07-0.17 GB at N=24M) but 6-15 GB at the 2^31
    # ceiling, so it is the wrong trade for an out-of-core design, whose whole point is bounding
    # memory -- see NOTES.md for the chunked-merge prototype this ruled out.
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
    # Any filler row disqualifies the array, and this must be tested *independently* of sortedness.
    # Two or more fillers are duplicates and so fail the strictness test below, which is what the
    # "rejects a padded result by design" claim rested on -- but a SINGLE filler is still strictly
    # increasing and used to pass. hproj has no filler-masking step, so that row became a spurious
    # basis state: one row and column too large, still symmetric, and measured -1.118034 against a
    # true -1.0. The test is on the packed byte because pack_states makes byte 0 < 128 for every
    # genuine state, so 255 is unambiguous -- unpacking a filler at n=2 gives [1, 1], a legitimate
    # state, which is why an unpacked-side check could not work.
    # O(1), not a pass over the column: fillers are all-255 rows and `pack_states` guarantees byte 0
    # < 128 for every genuine state, so they sort to the end -- if any filler is present the LAST row
    # carries one. A full `np.any(states[:, 0] >> 7)` measured 5.25 ms at N=10M against 0.00012 ms
    # here, i.e. ~4.4x the cost of the adjacent-row pass it was prepended to, for the same answer.
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
    `tests/test_sqd.py::TestHproj::test_unsorted_input_with_unique_states_is_wrong` -- named for the
    behaviour, since nothing is rejected: the result is silently wrong.

    Since `S` is sorted, finding `A` is a **binary search** of `S ^ X` into `S` -- not a reason to
    sort anything. The former implementation concatenated `S` and `S ^ X` into a `[2N, B]` array and
    sorted that, which cost three things: the `2N` allocation is what caps `N` at `2^31` (the sort
    must run on one device), `lax.sort` was observed to leak GPU memory (up to 5 GB at shape
    `(5M, 9)`), and it dominated runtime -- measured 66-97% of an entire solve. A `searchsorted` is a
    pure gather, so it also shards, where a sort does not. Measured on CPU at 12-25x per signature and
    12-17x on the J-fold precompute, and **5.15x at N=64M on an NVIDIA GH200** (a GPU sort is
    well optimized relative to its gather, so the ratio compresses while the direction holds); see
    `docs/scaling-pocs.md`.

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
        # Explicit binary search, on uint64 words rather than uint8 bytes: a level costs ceil(B/8)
        # comparisons instead of B (2 rather than 13 at n=100), and the packing is one pass. The
        # word form is what makes this path affordable past n=60 -- the byte form measured an ~8x
        # per-state cliff across the B > 8 boundary (15 ns/state at n=60, 189 at n=100).
        # Invariant unchanged: lo is the count of rows strictly less than the target, so after
        # ceil(log2(N)) + 1 halvings lo is the insertion point.
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

    Accumulated in uint8 and reduced with `& 1`: the parity is the only bit that matters, and both
    diagonal builders must spell it the same way, since a mismatch in the reduction dtype or the mask
    silently changes the sign of a term rather than raising.
    """
    return jnp.sum(jnp.bitwise_count(states & zsignature), axis=1, dtype=np.uint8) & 1


@jax.jit
def get_diag_signs(zsignatures: NDArray[np.uint8], states: StateList) -> jax.Array:
    """Return the packed sign bits, one per (state, Z term).

    Args:
        zsignatures: Packed Z signatures for one X group, shape ``(num_zterms, num_bytes)``.
        states: Uniquified, lex-sorted packed state list, shape ``(num_states, num_bytes)``.

    Returns:
        Packed sign bits, shape ``(num_states, ceil(num_zterms / 8))``.

    Raises:
        ValueError: If ``zsignatures`` is not 2-D. This function is public and called directly by
            scripts under ``examples/scaling/``, and the scan below iterates its leading axis: handed
            a 1-D array it scans *scalars* rather than rows, silently returning a wrongly shaped
            result (measured shape ``(4, 1)`` from a 2-element 1-D input) instead of raising. Rank is
            static under ``jax.jit``, so unlike the lex-sortedness precondition this one is
            checkable here.
    """
    _check_zsignatures_rank(zsignatures)

    def get_signs(carry, zsignature):
        out, ibyte, ibit = carry
        sign_bits = _z_parity(states, zsignature)
        # bits and bytes are counted from the left
        out = out.at[:, ibyte].add(sign_bits << (7 - ibit), out_sharding=jax.typeof(out).sharding)
        ibyte, ibit = jax.lax.cond(ibit == 7, lambda: (ibyte + 1, 0), lambda: (ibyte, ibit + 1))
        return (out, ibyte, ibit), None

    num_bytes = np.ceil(zsignatures.shape[0] / 8).astype(int)
    init = jnp.zeros(
        (states.shape[0], num_bytes), dtype=np.uint8, out_sharding=jax.typeof(states).sharding
    )
    return jax.lax.scan(get_signs, (init, 0, 0), zsignatures)[0][0]


def _accumulate_diagonal(
    coeffs: NDArray[np.inexact], template: jax.Array, sign_bit: Callable[[jax.Array], jax.Array]
) -> jax.Array:
    """Sum ``coeff * (1 - 2 * sign_bit(iterm))`` over the Z terms of one X group.

    The two public diagonal builders differ only in how they derive the sign bit -- from cached
    packed bits, or from ``popcount(state & z)`` -- so everything else lives here: the termination
    rule, the accumulator dtype, and the output sharding.

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

    # Carry over only the *leading* axis of the template's sharding. The output is 1-D of length
    # template.shape[0] while the template may be 2-D -- `get_diagonal` passes the (N, nbytes) state
    # list -- and jnp.zeros rejects a rank-2 spec on a rank-1 aval ("Length of sharding.spec (2) must
    # be equal to aval's ndim (1)"). That raise fired on *every* sharded sqd call, at every
    # cache_level, because run_sqd's vinit_from_min_diag reaches get_diagonal unconditionally.
    #
    # Rebuilt as a NamedSharding rather than a bare PartitionSpec: the spec alone is rejected when no
    # mesh context is active, which is the ordinary single-device path.
    sharding = jax.typeof(template).sharding
    init = jnp.zeros(
        template.shape[0],
        dtype=coeffs.dtype,
        out_sharding=jax.sharding.NamedSharding(sharding.mesh, PartitionSpec(sharding.spec[0])),
    )
    return jax.lax.while_loop(cond_fn, add_diag, (init, 0))[0]


@jax.jit
def compute_diagonal(diag_signs: NDArray[np.uint8], coeffs: NDArray[np.inexact]) -> jax.Array:
    """Compute the diagonals from the sign bits and coefficients."""

    def sign_bit(iterm):
        # iterm & 7, not iterm & 255: this is the bit offset WITHIN the selected byte, so it must
        # wrap at 8. With & 255 the shift 7 - ibit goes negative from iterm=8 onward, i.e. as soon as
        # an X group holds more than 8 Z terms, and the composed diagonal is silently wrong
        # (measured: 0.71 absolute error on 9 terms, and a 25% error in the end-to-end eigenvalue).
        return (diag_signs[:, iterm // 8] >> (7 - (iterm & 7))) & 1

    return _accumulate_diagonal(coeffs, diag_signs, sign_bit)


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
    cache_level: tuple[int, int], xgroup: NDArray, diagonal_arg: NDArray, coeffs: NDArray | None
) -> tuple[NDArray, ...]:
    """Lay out the tuple ``_apply_h_kernel`` scans over, for one resolved ``cache_level``.

    The kernel unpacks positionally (``val[0]``, ``val[1]``, and ``val[2]`` only when
    ``cache_level[1] == 0``), so the arity rule is a contract between packer and kernel: a 3-tuple
    carrying the coefficients for the two strategies that *compute* a diagonal, a 2-tuple for the one
    that reads a precomputed one. Both callers -- ``run_sqd`` and ``apply_h``'s keyword resolution --
    go through here so that rule is stated once rather than once per caller.
    """
    if cache_level[1] == 2:
        return (xgroup, diagonal_arg)
    return (xgroup, diagonal_arg, coeffs)


def apply_h(
    vec: NDArray[np.inexact],
    *,
    states: StateList | None = None,
    xsignatures: NDArray | None = None,
    xsources: NDArray | None = None,
    zsignatures: NDArray | None = None,
    diag_signs: NDArray | None = None,
    diagonals: NDArray | None = None,
    coeffs: NDArray | None = None,
) -> jax.Array:
    r"""Return :math:`Hv`, naming the per-X-group inputs so a mispairing cannot be expressed.

    Name the per-X-group arrays you have and the caching strategy follows from them:

    .. code-block:: python

        apply_h(vec, xsources=..., diag_signs=..., coeffs=..., states=states)   # was (1, 1)
        apply_h(vec, xsignatures=..., diagonals=..., states=states)             # was (0, 2)

    Every array parameter is keyword-only, so the six valid input sets are the only constructible
    ones. **This replaced a positional ``(scanned, cache_level)`` form, which is gone** -- a breaking
    change, and the reason it went: ``cache_level`` selected *positionally* how the members of
    ``scanned`` were interpreted, and nothing could check that the tuple matched the strategy
    declared. Passing raw X signatures while claiming ``cache_level[0] == 1`` -- which promises
    precomputed X *sources* -- raised nothing and silently computed a different operator (measured
    max abs error 0.44 on a 5-state n=4 subspace). Both are integer arrays, so the boundary could not
    tell an index array from a signature array.

    :func:`sqd` and :mod:`ground_locg` do not go through here: they call the private
    ``_apply_h_kernel`` with an assembled tuple and a static ``cache_level`` bound via
    ``functools.partial``, because the solver splats ``matvec(vec, *args)`` positionally.

    A per-branch **shape** assertion cannot help, which is worth recording because it is the obvious
    cheap alternative. X sources are ``(n_groups, n_states)`` and X signatures are
    ``(n_groups, n_bytes)``, so the trailing dimension usually separates them -- but at ``n = 15``
    (2 bytes) with a 2-state subspace both are exactly ``(2, 2)``, and a mispairing would sail through
    the assertion meant to trip it.

    A **dtype** assertion does help, and is now applied (:func:`_check_array_role`), which separates the
    confusable roles structurally at exactly the point shape fails. So ``xsources=<signature array>``
    now raises.

    What naming closes, stated exactly: it removes *mispairing* -- declaring one strategy while having
    packed the arrays for another -- because the strategy is no longer declared separately from the
    arrays. Combined with the dtype check above, an array passed under a wrong name of a *different*
    kind now raises too. What remains open is a swap between two roles of the **same** kind --
    ``xsignatures`` for ``zsignatures``, both ``uint8``. That residue is smaller again: the name sits
    at the call site immediately beside the array it labels, rather than in a positional tuple whose
    meaning is fixed by a separate argument several lines away.

    All six strategies are one ``jax.lax.scan`` over the X groups accumulating
    ``out + apply_xgrp(xsource, diagonal, vec)``; they differ only in where the two inputs come
    from. That is exactly the 2x3 grid ``cache_level`` names, so it is expressed as a grid
    rather than as six near-identical functions:

    =============  ==========================  =====================================
    cache_level    ``xsource``                 ``diagonal``
    =============  ==========================  =====================================
    ``(0, *)``     ``get_xsource(x, states)``  --
    ``(1, *)``     ``xsources`` as given       --
    ``(*, 0)``     --                          ``get_diagonal(z, c, states)``
    ``(*, 1)``     --                          ``compute_diagonal(signs, c)``
    ``(*, 2)``     --                          ``diagonals`` as given
    =============  ==========================  =====================================

    The keyword names correspond one-to-one: ``xsignatures``/``xsources`` select the first index,
    ``zsignatures``/``diag_signs``/``diagonals`` the second, and ``coeffs`` is required by the two
    diagonal strategies that compute rather than read a diagonal.

    Args:
        vec: Vector to multiply.
        states: Uniquified state list. Required whenever either element of the resolved
            ``cache_level`` is 0, i.e. for every combination except ``(1, 1)`` and ``(1, 2)`` -- those
            two read neither the X signatures nor the Z signatures, so they need no states at all.
        xsignatures: Packed X signatures per group; selects ``cache_level[0] == 0``.
        xsources: Precomputed X source indices per group; selects ``cache_level[0] == 1``.
        zsignatures: Packed Z signatures per group; selects ``cache_level[1] == 0``. Needs ``coeffs``.
        diag_signs: Precomputed diagonal sign bits per group; selects ``cache_level[1] == 1``. Needs
            ``coeffs``.
        diagonals: Fully precomputed diagonals per group; selects ``cache_level[1] == 2``. Must not be
            combined with ``coeffs``, which it makes redundant.
        coeffs: Pauli coefficients per group. Required by ``zsignatures`` and ``diag_signs``.

    Returns:
        :math:`Hv`.

    Raises:
        ValueError: If the named arrays do not select exactly one X source and one diagonal strategy
            (including naming none at all); if ``coeffs`` is missing where required or supplied
            alongside ``diagonals``; or if ``states`` is None while the resolved ``cache_level`` has a
            0 in either position.
    """
    # Each axis is one list of (keyword name, cache_level digit, array). Pairing the three together
    # means an axis is filtered, validated and unpacked from a single place -- no name-to-digit table
    # to keep in step with a separate name-to-array lookup, and no dict built only to be read back.
    xgiven = [
        opt
        for opt in (("xsources", 1, xsources), ("xsignatures", 0, xsignatures))
        if opt[2] is not None
    ]
    dgiven = [
        opt
        for opt in (
            ("diagonals", 2, diagonals),
            ("diag_signs", 1, diag_signs),
            ("zsignatures", 0, zsignatures),
        )
        if opt[2] is not None
    ]
    if len(xgiven) != 1:
        raise ValueError(
            "apply_h: pass exactly one of xsources= or xsignatures= "
            f"(got {sorted(name for name, _, _ in xgiven) or 'neither'})"
        )
    if len(dgiven) != 1:
        raise ValueError(
            "apply_h: pass exactly one of diagonals=, diag_signs= or zsignatures= "
            f"(got {sorted(name for name, _, _ in dgiven) or 'none'})"
        )
    (xname, xaxis, xarray), (dname, daxis, darray) = xgiven[0], dgiven[0]

    # coeffs is not optional-with-a-default: it is required by exactly two of the three diagonal forms
    # and meaningless in the third, so silently ignoring a stray one would hide a mistake.
    if daxis == 2 and coeffs is not None:
        raise ValueError("apply_h: diagonals= already folds in coeffs=; do not pass both")
    if daxis != 2 and coeffs is None:
        raise ValueError(f"apply_h: {dname}= requires coeffs=")

    cache_level = (xaxis, daxis)
    # `_apply_h_kernel` checks this too, so it looks gratuitous -- three independent reviewers read it
    # that way. It is not: the ORDER matters. The role check below must run after it, because a caller
    # who has passed the wrong arrays *and* omitted `states` needs to hear about the missing input set
    # first (it is what they must fix regardless), and the kernel's copy runs after both. Removing this
    # line makes the dtype error surface instead, which `TestMatvecKernels::test_omitting_states_raises`
    # catches. Keep them in step if either message changes.
    if (xaxis == 0 or daxis == 0) and states is None:
        raise ValueError(f"states is required for cache_level={cache_level}")

    # Dtype closes part of the misnaming residue -- see this function's docstring for what it does and
    # does not reach. A *shape* check cannot: X signatures and X sources are both exactly (2, 2) at
    # n=15 with a 2-state subspace, which is why the positional form was deleted rather than asserted.
    # Dtype separates them structurally at precisely that point: packed signatures are uint8 (packbits
    # output) while source indices are int32 positions using -1 as the absent marker, and a uint8
    # cannot hold -1.
    #
    # Ordered last on purpose. The checks above concern the input *set* -- which arrays were named, and
    # whether they are mutually consistent -- and a caller must fix those first regardless of dtypes.
    # Running the role check after them keeps it strictly additive: it never displaces an error that
    # was already being raised.
    _check_array_role(xname, xarray)
    _check_array_role(dname, darray)

    return _apply_h_kernel(
        vec, _pack_scanned(cache_level, xarray, darray, coeffs), states, cache_level
    )


@jax.jit(static_argnames=["cache_level"])
def _apply_h_kernel(
    vec: NDArray[np.inexact],
    scanned: tuple[NDArray, ...],
    states: StateList | None,
    cache_level: tuple[int, int],
) -> jax.Array:
    r"""Return :math:`Hv`, resolving the per-X-group inputs according to ``cache_level``.

    The jitted kernel behind :func:`apply_h`. Kept positional with a static ``cache_level`` because
    :mod:`ground_locg` splats ``matvec(vec, *args)``: a ``static_argnames`` entry would never see a
    keyword, and a traced ``cache_level`` tuple would retrace on every matvec call in the solver loop.
    The name resolution lives in the wrapper, in plain Python, so it costs nothing per call.

    All six caching strategies are one ``jax.lax.scan`` over the X groups accumulating
    ``out + apply_xgrp(xsource, diagonal, vec)``; they differ only in where the two inputs come from.
    ``cache_level`` is static, so each combination traces to the same code the six separate kernels
    did -- the selection happens once at trace time, not per group.

    See :func:`apply_h` for the grid, the argument semantics, and the tuple orderings; this docstring
    deliberately does not restate them, so the two cannot drift apart.

    Raises:
        ValueError: If ``states`` is None while ``cache_level`` has a 0 in either position.
    """
    if (cache_level[0] == 0 or cache_level[1] == 0) and states is None:
        raise ValueError(f"states is required for cache_level={cache_level}")

    def fn(out, val):
        # val[0] is the X source for this group: either the precomputed index array or the X
        # signature it is derived from.
        xsource = val[0] if cache_level[0] == 1 else get_xsource(val[0], states)
        if cache_level[1] == 0:
            diagonal = get_diagonal(val[1], val[2], states)
        elif cache_level[1] == 1:
            diagonal = compute_diagonal(val[1], val[2])
        else:
            diagonal = val[1]
        return out + apply_xgrp(xsource, diagonal, vec), None

    return jax.lax.scan(fn, jnp.zeros_like(vec), scanned)[0]
