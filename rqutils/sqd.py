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

**Selecting a strategy at the** :func:`apply_h` **boundary.** Name the arrays you have and the
strategy follows from them. The tuple form that took its meaning from ``cache_level`` positionally is
deprecated: nothing verified that the tuple matched the level declared, and supplying X signatures
while declaring precomputed sources silently computed a different operator (measured 0.44 max abs
error, no exception). A dtype or shape assertion cannot close that gap -- stacked Z signatures and
sign bits are both uint8 of rank 3 -- so the representations are distinguished by name instead.

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

When the source indices are cached -- ``cache_level[0] == 1``, whatever the diagonal setting -- the
state list :math:`S` is also sharded once the source indices are computed, since nothing sorts or
binary-searches it afterwards. At ``cache_level[0] == 0`` it stays replicated, because
:func:`get_xsource` runs inside the matvec and its binary search needs the whole sorted array; the
consumers that need a sharded view of it derive one themselves (see :func:`_spread_seed`).

SQD API
=======

.. autofunction:: sqd
.. autofunction:: hproj
.. autofunction:: all_diagonals

States are packed with :meth:`~rqutils.paulis.symplectic.PauliSumXZ.pack_states`, which inserts the
pad bit that aligns them with the Hamiltonian's signatures, and recovered with
:meth:`~rqutils.paulis.symplectic.PauliSumXZ.unpack_states`.
"""

import functools
import logging
import time
import warnings
from collections.abc import Callable, Sequence
from numbers import Number

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import PartitionSpec, get_abstract_mesh
from numpy.typing import DTypeLike, NDArray
from scipy.sparse import coo_array, csr_array

from rqutils.ground_locg import ground_locg
from rqutils.paulis.symplectic import PauliSumXZ

LOG = logging.getLogger(__name__)
# Largest subspace size representable in the int32 indices used throughout (uniquify_states' iota,
# get_xsource's output). Documented in the module docstring as the hard scaling limit; enforced in
# sqd() and hproj() so an overflow raises instead of silently permuting the subspace.
_MAX_STATES = 2**31 - 1


def _default_states_size(num_states: int) -> int:
    """Return the default ``states_size`` for a subspace of ``num_states`` rows.

    The next power of two at or above ``num_states``, floored at 2. Named rather than inlined so the
    tests that assert what the default *is* pin this rule instead of restating it -- a restatement
    would keep passing against a changed rule, silently turning those comparisons into tautologies.

    The floor matters only for degenerate input (0, 1 or 2 rows all give 2); it exists so a subspace
    can never be bucketed to a single slot, where the filler machinery has no room to work.
    """
    return 1 << max((num_states - 1).bit_length(), 1)


type HamiltonianInput = PauliSumXZ | tuple[Sequence[str], Sequence[Number]]
try:
    from qiskit.quantum_info import SparsePauliOp

    HamiltonianInput |= SparsePauliOp
except ImportError:
    pass
type Vector = np.ndarray[tuple[int], np.dtype[np.inexact]]
type StateList = np.ndarray[tuple[int, int], np.dtype[np.uint8]]


def sqd(
    hamiltonian: HamiltonianInput,
    states: StateList,
    states_size: int | None = None,
    return_eigvec: bool = True,
    cache_level: tuple[int, int] = (1, 0),
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

    On :func:`sqd` this tuple is the supported spelling -- the arrays are assembled internally, so
    there is nothing for a caller to mispair. A caller invoking :func:`apply_h` directly should name
    the arrays instead (``xsources=``/``xsignatures=``, ``diagonals=``/``diag_signs=``/
    ``zsignatures=``); the equivalent positional tuple is deprecated there because nothing could check
    it against the level declared.

    Args:
        hamiltonian: Hamiltonian to be projected and diagonalized.
        states: Binary array of computational basis states to project the Hamiltonian onto. Shape
            (subspace_dim, num_qubits).
        states_size: Fix the size of the states array used in computation to the specified value so
            that compilation is not triggered at each call with slightly different array sizes. Must
            be at least ``states.shape[0]``. Defaults to the next power of two at or above
            ``states.shape[0]``, which is the rule every caller with a growing subspace needs (a
            Krylov or configuration-recovery loop sees a fresh dimension per call, so an
            exact-length default would retrace on every one; measured 1.76x over five growing
            dimensions, energies agreeing to 8.9e-16 -- padding moves the last ULP by changing the
            reduction order, so this is not a bitwise-identical transformation). On a non-empty mesh
            this is rounded **up** to the next
            multiple of ``mesh.size``, so the value used internally may exceed the one passed; that
            widens the coalescing this argument exists for rather than defeating it (measured on a
            4-device mesh, ``states_size`` 33 through 36 all share one compiled kernel -- 707 ms to
            compile, then ~16 ms -- while 37 rounds to 40 and compiles afresh). The padding is not
            observable in the result: filler slots are excluded from the projection and trimmed from
            the returned basis.
        return_eigvec: Whether to return the eigenvector (coefficients and unique state bitstrings).
        cache_level: Switches for caching the results of source indices and sign bits / diagonals.
            See the module documentation for the detailed discussion of the resource tradeoff involved.

    Returns:
        Calculated ground state energy, or a tuple of energy, ground state vector, and sorted
        uniquified states (if return_eigvec=True). The returned states are the genuine unique rows
        only, never the filler slots, so their count can be below ``states_size``.

    Raises:
        ValueError: If ``states_size`` is smaller than ``states.shape[0]``, or if it exceeds
            :math:`2^{31} - 1`, the ceiling imposed by the int32 indices used for subspace positions
            (beyond it an index wraps negative and the subspace is silently permuted).
    """
    if states_size is None:
        # Bucket to the next power of two rather than using the raw length. Growing, all-distinct
        # subspace dimensions are the *normal* SQD access pattern, not an edge case: a Krylov or
        # configuration-recovery loop presents a dimension it has never seen before on every call, so
        # a raw-length default retraces the whole solver every time -- which is exactly what
        # states_size exists to prevent. Bucketing collapses those into one kernel per octave
        # (measured 1.76x over five growing dimensions 60..260 at n=10; a caller reported 1.25x on
        # the same shape of experiment and 1.43x over the first five rungs of an n=13 job).
        #
        # Energies agree to 8.9e-16, not bitwise: padding changes the reduction order over the filler
        # slots, so the last ULP can move. That is far below the eigensolver's own tolerance, but it
        # is a real difference -- don't assert bitwise equality across two states_size values.
        #
        # Purely a shape knob: the padding is not observable, since filler slots are excluded from the
        # projection and trimmed off the returned basis below. Pass an explicit value to override.
        states_size = _default_states_size(states.shape[0])
    if states_size < states.shape[0]:
        raise ValueError("states_size smaller than the states array length")
    # The 2^31 ceiling the module docstring calls a hard limit, actually enforced. Subspace positions
    # are int32 throughout -- uniquify_states' iota, and get_xsource's returned indices with -1 as the
    # absent marker -- so a size at or above 2^31 wraps to a negative index and yields a corrupted
    # permutation rather than an error: a plausible finite answer, the failure mode this module keeps
    # guarding against. Unreachable on current hardware (2^31 states is already 4.3 GB of packed
    # states before any vector), which is exactly why the check is cheap insurance rather than a cost.
    if states_size > _MAX_STATES:
        raise ValueError(
            f"states_size {states_size} exceeds the {_MAX_STATES} limit imposed by int32 subspace "
            "indexing; see the scaling-limits section of the module documentation"
        )
    if not isinstance(hamiltonian, PauliSumXZ):
        hamiltonian = PauliSumXZ.from_paulisum(hamiltonian)

    if not (mesh := get_abstract_mesh()).empty and (resid := states_size % mesh.size) != 0:
        LOG.debug("Adjusting states_size to make the array divisible by %d", mesh.size)
        states_size += mesh.size - resid

    states_p = PauliSumXZ.pack_states(states)
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
    # Built here rather than inside run_sqd: dropping the padding is a boolean-mask index, which
    # needs concrete shapes, and run_sqd sees `hamiltonian` as a pytree of tracers. Only the
    # cache_level[1] == 2 path reads it, so it is skipped otherwise -- the reshape is host-side work
    # proportional to the rectangle.
    flat_terms = hamiltonian.flat_terms if cache_level[1] == 2 else None
    result = run_sqd(
        hamiltonian, states_p, states_size, return_eigvec, cache_level, flat_terms=flat_terms
    )
    LOG.info("Found ground eigenpair in %f seconds.", time.time() - start)
    eigval = float(result[0])
    if return_eigvec:
        eigvec, states_u, subspace_dim = result[1:]
        basis_states = PauliSumXZ.unpack_states(states_u[:subspace_dim], states.shape[1])
        return (eigval, np.array(eigvec[:subspace_dim]), basis_states)
    return eigval


def hproj(
    hamiltonian: HamiltonianInput, states: StateList, unique_states: bool = False
) -> csr_array:
    r"""Return the Hamiltonian projected onto the given subspace.

    The Hamiltonian can be given in three different forms:

    * A tuple of two lists, where the first list enumerates the Pauli strings :math:`Q` as strs and
      the second contains the coefficients :math:`\alpha`.
    * Qiskit SparsePauliOp
    * PauliSumXZ (From :mod:`rqutils.paulis.symplectic`)

    States must have binary values and can be passed as an array of integers or booleans.

    Args:
        hamiltonian: Hamiltonian to be projected and diagonalized.
        states: Binary array of computational basis states to project the Hamiltonian onto. Shape
            (subspace_dim, num_qubits).
        unique_states: Whether ``states`` can be assumed to be already uniquified **and
            lex-sorted**, skipping the internal ``np.unique(..., axis=0)``. Both halves are
            required, because :func:`get_xsource` binary-searches into ``states``; violating either
            raises :class:`ValueError`. Sortedness is validated on this path (host-side numpy,
            measured 12-18% of the call), so an unsorted subspace is rejected rather than silently
            projected into a wrong, non-symmetric matrix as it was before. Leave this ``False`` to
            skip the check and have ``hproj`` sort for you.

            **Passing ``True`` with already-unique states is the faster path, not merely a way to
            avoid repeating work.** The validation is an O(N) adjacent-row comparison in place of
            ``np.unique``'s O(N log N) sort -- 0.12 ms against 2.96 ms at N=4064, ~24x cheaper -- so
            end-to-end ``hproj`` measures 1.4x faster at N=466, rising to 2.6x at N=2545 as the sort
            it displaces grows. Beware two wrong numbers when re-measuring: a cold-versus-warm
            comparison reads 55x, and timing the validation predicate alone reads 4.8-6.6% rather
            than the 12-18% an A/B of the whole call gives.

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
    # Same int32 ceiling sqd() enforces, since hproj reaches get_xsource too and its returned
    # positions are int32 with -1 as the absent marker. Checked here, before the O(N) sortedness scan
    # and the np.unique below: it is an O(1) look at a shape, so it costs nothing to do first and
    # reports the real problem rather than letting a doomed call spend time first.
    states = np.asarray(states)
    if states.shape[0] > _MAX_STATES:
        raise ValueError(
            f"subspace of {states.shape[0]} states exceeds the {_MAX_STATES} limit imposed by int32 "
            "subspace indexing; see the scaling-limits section of the module documentation"
        )
    if not unique_states:
        states = np.unique(states, axis=0)
    else:
        # get_xsource binary-searches into `states`, so unsorted rows silently produced a wrong,
        # non-symmetric matrix. Validated rather than documented away, and only on this opt-in path --
        # the branch above is sorted by construction and pays nothing.
        #
        # Cost provenance, which the docstring above does not carry: A/B'd against 2a89f94~1 (the
        # revision before this guard) per the whole-call rule -- 1.052 -> 1.239 ms at N=1580 and
        # 1.131 -> 1.276 ms at N=2545. See the `unique_states` docstring for what those percentages
        # mean and for the two ways of mis-measuring them.
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
        columns = get_xsource(ham[0], states_p)
        diagonals = get_diagonal(ham[1], ham[2], states_p)
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

    Args:
        states_size: Length of the vector to build, i.e. the padded subspace size.
        states_u: Uniquified state list, read only for its fill-in markers.
        dtype: Output dtype, normally ``hamiltonian.c.dtype``.
        sharding: Partition spec for the result, or ``None`` on an empty mesh. The fill-in predicate
            is resharded to match -- see the comment at the ``jnp.where`` for why that is not
            automatic.

    Returns:
        The seed vector, shape ``(states_size,)``.
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
    # Match the predicate's sharding to `vec`'s rather than assuming they agree. They do not at
    # cache_level[0] == 0: `vec` follows the `sharding` argument, while `states_u` is still unsharded
    # there because run_sqd defers its reshard until after the last get_xsource -- which on that level
    # happens inside the matvec, so it never runs before this point. jnp.where then raises
    # "select `which` must be scalar or have the same sharding as cases", which is why all three
    # cache_level[0] == 0 kernels failed on any non-empty mesh. Resharding the small 0/1 predicate is
    # cheaper than resharding the state list, and leaves the binary search's precondition untouched.
    is_filler = _is_filler(states_u)
    if sharding is not None:
        is_filler = jax.reshard(is_filler, sharding)
    return jnp.where(is_filler == 1, jnp.zeros_like(vec), vec)


@jax.jit(static_argnames=["states_size", "return_eigvec", "cache_level", "log_level"])
def run_sqd(
    hamiltonian: PauliSumXZ,
    states_p: StateList,
    states_size: int,
    return_eigvec: bool,
    cache_level: tuple[int, int] = (1, 0),
    log_level: int = logging.INFO,
    flat_terms: tuple[NDArray, NDArray, NDArray] | None = None,
) -> tuple[float] | tuple[float, jax.Array, jax.Array, int]:
    """JIT-compiled part of the SQD function.

    ``flat_terms`` is the padding-free ``(z, c, group_ids)`` layout from
    :attr:`~rqutils.paulis.symplectic.PauliSumXZ.flat_terms`, used by the
    ``cache_level[1] == 2`` precompute in place of a scan over the padded rectangle. It is built by
    the caller because the mask that drops the padding needs concrete shapes -- inside this trace
    ``hamiltonian`` is a pytree of tracers and the reshape raises
    ``TracerArrayConversionError``. ``None`` falls back to the scan, so the argument is optional.
    """
    sharding = None
    if not (mesh := get_abstract_mesh()).empty:
        sharding = PartitionSpec(mesh.axis_names)

    if log_level <= logging.DEBUG:
        jax.debug.print("Uniquifying states (size {})", states_size)

    states_u = uniquify_states(states_p, states_size)

    if cache_level[0] == 1:
        if log_level <= logging.DEBUG:
            jax.debug.print("Precomputing xsources")

        xsources = jax.lax.scan(lambda _, x: (None, get_xsource(x, states_u)), None, hamiltonian.x)[
            1
        ]
        if sharding:
            # We will not be performing sorts on states any more - shard the array
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

        if flat_terms is None:
            diagonals = jax.lax.scan(
                lambda _, v: (None, get_diagonal(v[0], v[1], states_u)),
                None,
                (hamiltonian.z, hamiltonian.c),
            )[1]
        else:
            # One pass over the real terms instead of J passes over the padded rectangle. Identical
            # to the last bit, and faster at every raggedness measured -- see all_diagonals.
            flat_z, flat_c, group_ids = flat_terms
            diagonals = all_diagonals(flat_z, flat_c, group_ids, states_u, hamiltonian.x.shape[0])

    # Assemble the per-X-group arrays apply_h scans over. This stays a Python-level match on the
    # static cache_level: the *packing* must be static too, or the tuple structure would become part
    # of the traced arguments and retrace on every call.
    xgroup = xsources if cache_level[0] == 1 else hamiltonian.x
    match cache_level[1]:
        case 0:
            scanned = (xgroup, hamiltonian.z, hamiltonian.c)
        case 1:
            scanned = (xgroup, diag_signs, hamiltonian.c)
        case 2:
            scanned = (xgroup, diagonals)
    # (1, 2) reads neither signature array, so it needs no states at all -- which is what lets the
    # caller drop S entirely under the most aggressive caching (see the module docstring).
    needs_states = cache_level[0] == 0 or cache_level[1] == 0
    # cache_level is bound here rather than passed through args: ground_locg splats args
    # positionally (matvec(vec, *args)), so a static_argnames entry would never see it and the
    # tuple would be traced -- retracing the kernel on every matvec call in the solver loop.
    # nterms is bound here for the same reason cache_level is: it must be static, and ground_locg
    # splats args positionally. max() rather than per-group, because apply_h's scan body traces once
    # -- one trip count has to serve every group, and the max is the tightest correct value.
    #
    # Bound only above _NTERMS_MIN_K, because the fixed-length scan is not free: measured against the
    # while_loop it loses below ~8 terms and wins above (see _NTERMS_MIN_K). The solver does not need
    # differentiability -- a caller who does passes nterms to apply_h directly -- so here the only
    # question is speed, and below the crossover the while_loop is the faster kernel.
    #
    # _apply_h_resolved, not the public apply_h, for a mechanical reason and not merely because the
    # pairing is safe here: this code already holds an assembled `scanned` tuple and must bind the
    # statics via functools.partial for ground_locg's positional splat. Routing through the public
    # wrapper would mean unpacking that tuple back into six keywords so the wrapper could reassemble
    # it -- pure churn on the matvec path. So don't "simplify" by deleting the private kernel and
    # jitting apply_h; tests would still pass and the solver would pay for it.
    solver_nterms = max(hamiltonian.nzterms)
    matvec_nterms = solver_nterms if solver_nterms >= _NTERMS_MIN_K else None
    matvec = functools.partial(_apply_h_resolved, cache_level=cache_level, nterms=matvec_nterms)
    args = (scanned, states_u if needs_states else None)

    def vinit_from_min_diag():
        if cache_level[1] == 2:
            diagonal = diagonals[0]
        else:
            # Same nterms the matvec uses: this computes the same diagonal by the same kernel, so
            # leaving it ungated would trace the accumulator a second time under a different static
            # key within this one trace, and would let the two drift if the accumulator changed.
            diagonal = get_diagonal(hamiltonian.z[0], hamiltonian.c[0], states_u, matvec_nterms)
        # Narrow AFTER the join, not in each branch. The diagonal of a Hermitian operator is real, but
        # `.c` is complex128 whenever any Pauli string has an odd number of Ys, so the composed
        # diagonal carries a zero imaginary part along and the argmin below cannot order it ("lt does
        # not accept dtype complex128"). Both branches used to narrow independently, and that is
        # exactly how the bug shipped: the cache_level[1] == 2 branch was added without copying the
        # other's `.real`, so that level raised on any odd-Y operator with an all-identity X group --
        # the group that makes this function reachable at all. One narrowing on the joined value means
        # a third diagonal source cannot reintroduce it.
        diagonal = diagonal.real
        # Set the fill-in components to the maximum value so that argmin only sees the valid entries
        diagonal = jnp.where(_is_filler(states_u) == 1, jnp.max(diagonal), diagonal)
        imin = jnp.argmin(diagonal)
        # Weight the minimum-diagonal state heavily -- it is the best single guess available, and
        # keeping it dominant preserves this heuristic's fast convergence -- but add the spread seed
        # underneath rather than returning a bare one-hot.
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
        return (
            _spread_seed(states_size, states_u, hamiltonian.c.dtype, sharding)
            .at[imin]
            .add(1.0, out_sharding=sharding)
        )

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

    eigval, eigvec, _, _ = ground_locg(matvec, vinit, args=args, log_level=log_level)
    result = (eigval,)
    if return_eigvec:
        if sharding:
            eigvec = jax.reshard(eigvec, PartitionSpec(None))
            states_u = jax.reshard(states_u, PartitionSpec(None))
        subspace_dim = jnp.searchsorted(_is_filler(states_u), 1)
        result += (eigvec, states_u, subspace_dim)
    return result


@jax.jit(static_argnames=["states_size"])
def uniquify_states(states_p: StateList, states_size: int) -> StateList:
    """A stripped-down implementation of jnp.unique.

    The returned array will have shape (states_size, states_p.shape[1]). If states_size is greater
    than the number of unique states, the residual entries at the end are filled with 255.
    """
    # Perform a lexsort
    iota = jax.lax.broadcasted_iota(np.int32, (states_p.shape[0],), 0)
    perm = jax.lax.sort((*states_p.T, iota), dimension=0, num_keys=states_p.shape[1])[-1]
    states_srt = states_p[perm]
    # Uniqueness flag for elements 1 to N-1
    is_unique = jnp.any(jax.lax.ne(states_srt[1:], states_srt[:-1]), axis=1)
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
    one. Only valid while `B <= 8`; :func:`get_xsource` checks that before calling.
    """
    nbytes = states.shape[1]
    shifts = jnp.asarray([8 * (nbytes - 1 - i) for i in range(nbytes)], dtype=jnp.uint64)
    return jnp.sum(states.astype(jnp.uint64) << shifts, axis=1)


def _is_lex_sorted(states: NDArray[np.uint8]) -> bool:
    """Return whether `[N, B]` uint8 rows are in strictly increasing lexicographic order.

    Host-side numpy, deliberately: the one caller (:func:`hproj`) is eager, and the point is to
    ``raise`` before any tracing, which a traced predicate cannot do. Compares adjacent rows at their
    first differing byte -- equal rows count as *unsorted*, since `get_xsource`'s precondition is
    uniquified-and-sorted and a duplicate row makes the projection ambiguous either way.

    One vectorized pass, no early exit, so unsorted input costs the same as sorted. Measured at 12-18%
    of `hproj` (A/B'd end-to-end; both are `O(N)`, so the ratio is flat in N) and ~20 ms standalone at
    N=1M, flat in `B`. Cheap enough to be unconditional on a debug/reference path, in exchange for
    turning a silent wrong answer into a raise. :func:`sqd` never reaches `hproj`, so it is unaffected.

    **Rejects a padded :func:`uniquify_states` result, by design.** Filler slots are all-``255`` rows,
    so two or more are duplicates and fail the strictness test. That is correct here: `hproj` builds a
    dense `[N, N]` operator with no filler-masking step, so filler rows would become spurious basis
    states. Slice to the real rows first (`~_is_filler(states)`) if you hold a padded array; `sqd`
    already trims before returning its basis.
    """
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


def _row_less_than(rows: jax.Array, targets: jax.Array) -> jax.Array:
    """Elementwise lexicographic `rows[i] < targets[i]` over uint8 rows, MSB-first.

    The first differing byte decides, so mask each byte position by "all higher bytes equal" and
    take any hit. Written without a loop carry so it stays one fused expression per search step.

    The prefix-AND is accumulated in uint8, not bool: `jnp.cumprod` rejects a bool accumulator and
    promotes to int64 under `jax_enable_x64`, materializing the `[N, B]` mask at 8 bytes per element
    where 1 suffices. That is a dtype decision fixed before HLO, so XLA cannot undo it -- measured
    192 MB of transients versus 23 MB at N=1M, B=12, and 1.53x on the J-fold precompute this path
    feeds. Pin `dtype=` explicitly; the promotion comes back without it.
    """
    eq_prefix = jnp.cumprod(
        jnp.concatenate(
            [
                jnp.ones((rows.shape[0], 1), jnp.uint8),
                (rows[:, :-1] == targets[:, :-1]).astype(jnp.uint8),
            ],
            axis=1,
        ),
        axis=1,
        dtype=jnp.uint8,
    ).astype(bool)
    return jnp.any(jnp.logical_and(eq_prefix, rows < targets), axis=1)


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
    `jnp.searchsorted` directly; wider inputs fall back to an explicit lexicographic binary search
    (measured 3.0-3.7x rather than 12-25x, so the fast path is worth keeping separate). The
    boundary is a correctness limit, not a tuning parameter: at `B > 8` a `uint64` key would silently
    truncate the row and alias distinct states onto one key.

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
        # Explicit binary search on the rows themselves. Invariant: lo is the count of rows strictly
        # less than the target, so after ceil(log2(N)) + 1 halvings lo is the insertion point.
        def step(carry, _):
            lo, hi = carry
            mid = (lo + hi) // 2
            go_right = _row_less_than(states[jnp.minimum(mid, size - 1)], targets)
            return (jnp.where(go_right, mid + 1, lo), jnp.where(go_right, hi, mid)), None

        nsteps = int(np.ceil(np.log2(max(size, 2)))) + 1
        lo = jnp.zeros(size, dtype=jnp.int32)
        hi = jnp.full(size, size, dtype=jnp.int32)
        (lo, _), _ = jax.lax.scan(step, (lo, hi), None, length=nsteps)
        pos = jnp.minimum(lo, size - 1)
        found = jnp.all(states[pos] == targets, axis=1)

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
    """Return the packed sign bits."""

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


# Term count at or above which a fixed-length scan beats the data-dependent while_loop. The scan
# carries a fixed per-call cost that the while_loop does not, so the crossover is a property of the
# kernel rather than of the padding: measured at N=2^16 with a *tight* count (no padding scanned at
# all), the ratio while/scan is 0.62x at K=2, 0.80x at 4, 0.88x at 6, 1.03x at 8, and stays above 1
# through K=32. Padding shifts the magnitude, not the crossover. The measurement noise band on this
# arm is 12%, so treat a ratio within that of 1.0 as unresolved and prefer the while_loop.
#
# This gates speed only. Differentiability always requires nterms, so apply_h honours whatever a
# caller passes; this constant only decides what run_sqd binds for itself.
# Every (source_indices, diagonals) combination, i.e. all six matvec kernels apply_h resolves to.
# Lives here rather than in the tests because the POCs need it too and cannot import from tests/;
# two independent copies of this list is how a sharding failure on the (0, *) levels went unnoticed
# while poc7_sharding.py swept only a subset.
CACHE_LEVELS = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]

_NTERMS_MIN_K = 8


def _accumulate_diagonal(
    coeffs: NDArray[np.inexact],
    template: jax.Array,
    sign_bit: Callable[[jax.Array], jax.Array],
    nterms: int | None = None,
) -> jax.Array:
    """Sum ``coeff * (1 - 2 * sign_bit(iterm))`` over the Z terms of one X group.

    The two public diagonal builders differ only in how they derive the sign bit -- from cached
    packed bits, or from ``popcount(state & z)`` -- so everything else lives here: the termination
    rule, the accumulator dtype, and the output sharding.

    Null terms are removed by ``hamiltonian.simplify()`` on ingest, so a zero coefficient marks the
    end of the real terms in a zero-padded Z group and the loop stops there rather than scanning the
    full rectangle.

    **Pass ``nterms`` to make this differentiable.** The default ``while_loop`` terminates on a
    condition that reads ``coeffs``, and reverse-mode autodiff rejects that outright (``ValueError:
    Reverse-mode differentiation does not work for lax.while_loop``) -- so with ``nterms=None``
    nothing built on this can be differentiated with respect to operator coefficients. Note the limit
    is specific to the coefficients: a gradient w.r.t. the *vector* already works, since the vector
    never appears in the termination condition.

    A static ``nterms`` replaces the data-dependent condition with a ``lax.scan`` of known extent,
    which is differentiable and, **above roughly 8 terms, also faster**. The output is identical to
    the last bit either way (``maxdiff`` exactly 0.0, measured at 8k, 64k and 400k states).

    **It is not uniformly faster, and the crossover is a property of the kernel rather than of the
    padding.** A ``lax.scan`` carries a fixed per-call cost the ``while_loop`` does not, so below
    roughly 8 terms the ``while_loop`` wins even with no padding to skip. :data:`_NTERMS_MIN_K`
    carries the measured series and is where :func:`run_sqd` draws the line for its own use. What is
    specific to this function: at ``K=29`` with 3 real terms the scan is 1.09x, and at 400k states
    0.279 ms against 0.382 ms. A caller who needs the gradient passes ``nterms`` regardless of
    ``K``, since correctness of the derivative is not a speed question.

    **Whether to unroll the X-group loop for per-group counts depends on skew, and the answer is not
    uniform.** ``apply_h`` scans over X groups, so one trip count serves all of them and every group
    pays the widest group's extent. Replacing that ``lax.scan`` with a Python loop lets each group use
    its own ``nzterms``, at the cost of a compile that grows with :math:`J`. The deciding quantity is
    the **skew**, ``max(nzterms) / mean(nzterms)`` -- not :math:`J`, which is what an earlier revision
    of this note swept and why it recorded a blanket "do not unroll".

    Measured, local two-body operators with long-range Z strings added, at :math:`N = 4096`:

    =====  ==========  ==============  ==================
    skew   real slots  unroll speedup  break-even (calls)
    =====  ==========  ==============  ==================
    4.2x   23.8%       1.19x           2120
    5.9x   16.9%       2.84x           166
    8.5x   11.8%       4.43x           89
    12.8x  7.8%        7.88x           49
    17.1x  5.9%        14.57x          6
    25.6x  3.9%        23.26x          4
    =====  ==========  ==============  ==================

    The extra compile cost is bounded (32 ms at :math:`J=10` to 183 ms at :math:`J=60`) while the
    per-call saving is not, so skew decides the asymptotics. A solve runs tens to low hundreds of
    matvec calls, so **below ~6x skew the scan wins and above ~13x the unroll wins**, with a marginal
    band between. Ordinary Hamiltonians sit at the low end -- a nearest-neighbour chain plus one
    long-range term measures ``(27, 1, 1, ...)``, skew ~9x, break-even ~89 -- which is why the shipped
    default is the single ``max``. **Rotated operators sit at the high end**: a caller reported a
    782x197 rectangle with median 2 terms per group, 1.4% of slots real, and measured routing a cost
    function through ``apply_h`` at 2x slower (n=20) to 9.7x slower (n=100) than a hand-rolled
    nonzero-only contraction. For that regime per-group counts, or a segment-sum over a flat nonzero
    list, is the right structure and this function's single static ``nterms`` is not.

    Two caveats that hold at any skew. Unrolling the one-shot ``cache_level[1] == 2`` precompute is
    always a loss -- net 0.69x at ``J=11`` falling to 0.04x at ``J=356`` -- since a single call cannot
    amortize the compile. And the unroll reassociates the accumulation, so results move by ~1e-15
    rather than staying bit-identical.

    Do **not** "simplify" this to a scan over the full rectangle instead. That is differentiable too
    and equally exact, but it discards the early exit, and the padding fraction is large -- 78.9% at
    13 qubits, 90.4% at 30. Measured 2.334 ms at 400k states, 8.4x the static scan.

    Args:
        coeffs: Phased coefficients for this X group, shape ``(K,)``.
        template: Array whose leading axis length and sharding the output follows.
        sign_bit: Maps a term index to that term's per-state sign bit (0 or 1).
        nterms: Number of real terms to sum, which **must be static** (a Python ``int``, not a
            tracer). When given, the accumulation is a fixed-length scan: differentiable w.r.t.
            ``coeffs``, and skipping the same padding the ``while_loop`` skips provided the value is
            the group's true term count -- :attr:`~rqutils.paulis.symplectic.PauliSumXZ.nzterms`, or
            its ``max`` when one trace must serve every group. Values above the true count still give
            the right answer (padding contributes exactly zero) but scan more slots than needed.
            ``None`` keeps the ``while_loop``, so no existing caller changes behaviour.

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

    # out_sharding is load-bearing, on both paths: ground_locg is sharding-transparent only if the
    # matvec preserves its output sharding, and this init is what sets it for the diagonal.
    #
    # Take only the leading axis of template's spec: template is 2-D (states is (N, B), diag_signs is
    # (N, ceil(K/8))) while this output is 1-D (N,), so copying the spec wholesale hands a rank-2
    # P('x', None) to a rank-1 array. On a single device the mesh is empty and the mismatch is
    # invisible, which is why the pytest suite never saw it; on any real mesh it raised
    # "Length of sharding.spec (2) must be equal to aval's ndim (1)" before this could return.
    # Rebuild against template's own mesh rather than handing over a bare PartitionSpec, which jax
    # rejects outside a mesh context ("Using PartitionSpec when you are not under a mesh context").
    template_sharding = jax.typeof(template).sharding
    init = jnp.zeros(
        template.shape[0],
        dtype=coeffs.dtype,
        out_sharding=jax.sharding.NamedSharding(
            template_sharding.mesh, PartitionSpec(template_sharding.spec[0])
        ),
    )
    if nterms is None:
        return jax.lax.while_loop(cond_fn, add_diag, (init, 0))[0]

    # Static extent, so no data-dependent termination for autodiff to choke on. jnp.arange(nterms)
    # rather than a fori_loop: the reverse-mode restriction is on dynamic start/stop, and a scan
    # carries the accumulator explicitly.
    def step(diagonal, iterm):
        return diagonal + coeffs[iterm] * (1.0 - 2.0 * sign_bit(iterm)), None

    return jax.lax.scan(step, init, jnp.arange(nterms))[0]


@jax.jit(static_argnames=["nterms"])
def compute_diagonal(
    diag_signs: NDArray[np.uint8], coeffs: NDArray[np.inexact], nterms: int | None = None
) -> jax.Array:
    """Compute the diagonals from the sign bits and coefficients.

    Args:
        diag_signs: Packed sign bits for this X group.
        coeffs: Phased coefficients for this X group.
        nterms: Static term count. Pass it to make this differentiable w.r.t. ``coeffs``; see
            :func:`_accumulate_diagonal`.

    Returns:
        The composed diagonal.
    """

    def sign_bit(iterm):
        # iterm & 7, not iterm & 255: this is the bit offset WITHIN the selected byte, so it must
        # wrap at 8. With & 255 the shift 7 - ibit goes negative from iterm=8 onward, i.e. as soon as
        # an X group holds more than 8 Z terms, and the composed diagonal is silently wrong
        # (measured: 0.71 absolute error on 9 terms, and a 25% error in the end-to-end eigenvalue).
        return (diag_signs[:, iterm // 8] >> (7 - (iterm & 7))) & 1

    return _accumulate_diagonal(coeffs, diag_signs, sign_bit, nterms)


@jax.jit(static_argnames=["nterms"])
def get_diagonal(
    zsignatures: NDArray[np.uint8],
    coeffs: NDArray[np.inexact],
    states: StateList,
    nterms: int | None = None,
) -> jax.Array:
    """Return the fully composed diagonals for one X signature.

    Args:
        zsignatures: Packed Z signatures for this X group.
        coeffs: Phased coefficients for this X group.
        states: Packed subspace states.
        nterms: Static term count. Pass it to make this differentiable w.r.t. ``coeffs``; see
            :func:`_accumulate_diagonal`.

    Returns:
        The composed diagonal.
    """

    def sign_bit(iterm):
        return _z_parity(states, zsignatures[iterm])

    return _accumulate_diagonal(coeffs, states, sign_bit, nterms)


@jax.jit(static_argnames=["num_groups"])
def all_diagonals(
    zsignatures: NDArray[np.uint8],
    coeffs: NDArray[np.inexact],
    group_ids: NDArray[np.integer],
    states: StateList,
    num_groups: int,
) -> jax.Array:
    r"""Return every X group's diagonal in one pass over the real Z terms only.

    The rectangular alternative to :func:`get_diagonal`. That one is called per X group under a
    ``lax.scan``, so a single trip count serves every group and each pays the widest group's extent.
    This one takes the *flat* layout -- :attr:`~rqutils.paulis.symplectic.PauliSumXZ.flat_terms`,
    which drops the padding -- computes every term's contribution at once, and scatter-adds them back
    to per-group rows. The work is proportional to the number of real terms rather
    than to :math:`J \times \max_j K^{(j)}`.

    **This is strictly better than a static ``nterms`` when the groups are ragged, and never worse.**
    Measured against the ``lax.scan`` form at :math:`N = 4096`, on local two-body operators with
    long-range Z strings added (``skew`` is :math:`\max(\mathrm{nzterms}) / \mathrm{mean}`):

    ======  ======  ===========  =========  ===============  ==================
    J       skew    real slots   speedup    compile (scan)   compile (segsum)
    ======  ======  ===========  =========  ===============  ==================
    10      4.2x    23.8%        1.68x      44 ms            30 ms
    20      8.5x    11.8%        5.13x      45 ms            32 ms
    40      17.1x   5.9%         52.5x      117 ms           41 ms
    60      25.6x   3.9%         66.0x      138 ms           47 ms
    100     42.8x   2.3%         134x       282 ms           52 ms
    ======  ======  ===========  =========  ===============  ==================

    Note the compile column: one kernel over the real terms compiles *faster* than a scan over the
    padded rectangle, and stays nearly flat in :math:`J` where the scan grows. That is what separates
    this from unrolling the group loop, which buys a similar speedup but pays a compile cost growing
    linearly with :math:`J` (4233 ms at :math:`J=257`) and only breaks even after thousands of calls.

    Output is **bit-identical** to the scan form (``maxdiff`` exactly 0.0 at every size above), because
    the scatter accumulates in the same term order. Differentiable w.r.t. ``coeffs`` with no
    special handling -- there is no data-dependent termination to begin with -- and verified against a
    closed-form gradient to 1.8e-15.

    **Sharding-transparent**, which took two changes rather than one. The per-term contribution is
    computed inside a ``vmap`` so each term is a single ``(num_states,)`` row inheriting ``states``'
    sharding -- multiplying outside, against a stacked ``(num_terms, num_states)`` parity, is a
    rank-mismatched broadcast between ``P(None, 'x')`` and ``P(None,)`` that jax refuses. And the
    reduction is an explicit ``.at[ids].add(..., out_sharding=...)`` rather than
    :func:`jax.ops.segment_sum`, which allocates its accumulator internally with no way to constrain
    it and so dropped the sharding even though the contributions reaching it were already correct.
    The two forms are bit-identical; only the sharding differs. Verified on a 4-device mesh across
    every residue of ``N mod mesh.size``, agreeing with single-device to 2.2e-15.

    Args:
        zsignatures: Real Z signatures, shape ``(num_terms, num_bytes)``.
        coeffs: Their phase-folded coefficients, shape ``(num_terms,)``.
        group_ids: The X group each term belongs to, shape ``(num_terms,)``.
        states: Uniquified state list, shape ``(num_states, num_bytes)``.
        num_groups: Number of X groups, i.e. the leading dimension of the result. **Must be static**
            (it is the segment count), hence a ``static_argnames`` entry.

    Returns:
        The diagonals, shape ``(num_groups, num_states)`` -- the same array
        ``cache_level[1] == 2`` consumes.
    """

    # vmap over the terms rather than restating the popcount: _z_parity is where the reduction dtype
    # and the mask are defined, and a mismatch there silently flips a term's sign instead of raising.
    #
    # vmap the whole per-term contribution, not just the parity: inside the vmap each term is a single
    # (num_states,) row that inherits `states`' sharding, so the coefficient is a scalar times a
    # correctly-sharded vector. Multiplying outside, against a stacked (num_terms, num_states) parity,
    # is a rank-mismatched broadcast between P(None, 'x') and P(None,) that jax refuses on a mesh.
    def contribution(zsig, coeff):
        return coeff * (1.0 - 2.0 * _z_parity(states, zsig))

    contributions = jax.vmap(contribution)(zsignatures, coeffs)

    # Scatter-add into an explicitly-sharded accumulator rather than calling jax.ops.segment_sum.
    # The two are bit-identical (verified at three sizes), but segment_sum allocates its own zeros
    # internally with no way to constrain them, so on a non-empty mesh it drops the sharding and
    # raises "Incompatible types for broadcasting: float64[5,32@x] vs float64[5,32]" -- measured, with
    # `contributions` already correctly P(None, 'x') at that point, so the loss was entirely inside
    # the helper. This is the same `.at[...].add(..., out_sharding=...)` form get_diag_signs uses, and
    # it is what makes these diagonals safe to feed ground_locg, which is sharding-transparent only if
    # the matvec preserves output sharding.
    #
    # group_ids is non-decreasing (flat_terms builds it with np.repeat), so the scatter is contiguous.
    out_sharding = jax.typeof(contributions).sharding
    accumulator = jnp.zeros(
        (num_groups, contributions.shape[1]), dtype=contributions.dtype, out_sharding=out_sharding
    )
    return accumulator.at[group_ids].add(contributions, out_sharding=out_sharding)


# Renamed in the next commit; aliased here so this commit is independently testable.
_diag_from_signs = compute_diagonal
_diag_from_z = get_diagonal
_diag_all_groups = all_diagonals


def diagonals(
    coeffs: NDArray[np.inexact],
    *,
    diag_signs: NDArray[np.uint8] | None = None,
    zsignatures: NDArray[np.uint8] | None = None,
    group_ids: NDArray[np.integer] | None = None,
    states: StateList | None = None,
    num_groups: int | None = None,
    nterms: int | None = None,
) -> jax.Array:
    r"""Compose the diagonal of one X group, or of every X group at once.

    One entry point over three sign sources, named the way :func:`apply_h` names its inputs. Which
    source you have is a fact about your data, so it is stated rather than encoded in a choice of
    function:

    .. code-block:: python

        diagonals(coeffs, diag_signs=signs)                       # from cached packed sign bits
        diagonals(coeffs, zsignatures=z, states=states)           # recomputed for one X group
        diagonals(flat_coeffs, zsignatures=flat_z, group_ids=ids,  # every group, padding-free
                  states=states, num_groups=J)

    The first two return one group's diagonal, shape ``(num_states,)``. The third returns every
    group's, shape ``(num_groups, num_states)``, and is the form to prefer for ragged operators --
    see :attr:`~rqutils.paulis.symplectic.PauliSumXZ.flat_terms` for the layout and
    :func:`_diag_all_groups` for the measured speedups.

    This function itself is not jitted: it is a host-side keyword dispatch over three kernels that
    are each already ``@jax.jit``-decorated, exactly as :func:`apply_h` dispatches over its own named
    forms. Jitting it would gain nothing and would force ``nterms``/``num_groups`` to be threaded
    through as static arguments of *this* function too, when they only need to be static on the
    inner kernels that actually trace on them.

    Args:
        coeffs: Phase-folded coefficients. Shape ``(num_terms,)`` for one group, or the flat
            ``(sum(nzterms),)`` for the all-groups form.
        diag_signs: Precomputed packed sign bits for this group, from :func:`get_diag_signs`. Mutually
            exclusive with ``zsignatures``, and needs no ``states``.
        zsignatures: Packed Z signatures. Requires ``states``, since the sign bits are recomputed.
        group_ids: The X group each flat term belongs to. Selects the all-groups form and requires
            ``num_groups``.
        states: Uniquified state list. Required with ``zsignatures``, rejected with ``diag_signs``.
        num_groups: Number of X groups. **Must be static.** Required with ``group_ids``.
        nterms: Static term count for the per-group forms; see :func:`_accumulate_diagonal` for why
            it matters and when it pays. Ignored by the all-groups form, which has no padding to skip.

    Returns:
        The composed diagonal(s).

    Raises:
        TypeError: If the keywords do not form one of the three valid sets -- neither or both sign
            sources, ``zsignatures`` without ``states``, ``diag_signs`` with ``states``, or
            ``group_ids`` without ``num_groups``.
    """
    given = [
        name
        for name, value in (("diag_signs", diag_signs), ("zsignatures", zsignatures))
        if value is not None
    ]
    if len(given) != 1:
        raise TypeError(
            f"diagonals: pass exactly one of diag_signs= or zsignatures= (got {given or 'neither'})"
        )
    if diag_signs is not None:
        if states is not None:
            raise TypeError("diagonals: diag_signs= is used without states=; the bits are cached")
        return _diag_from_signs(diag_signs, coeffs, nterms)
    if states is None:
        raise TypeError("diagonals: zsignatures= requires states= to recompute the sign bits")
    if group_ids is None:
        return _diag_from_z(zsignatures, coeffs, states, nterms)
    if num_groups is None:
        raise TypeError("diagonals: group_ids= requires num_groups=, which must be static")
    return _diag_all_groups(zsignatures, coeffs, group_ids, states, num_groups)


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


@jax.jit(static_argnames=["cache_level", "nterms"])
def _apply_h_resolved(
    vec: NDArray[np.inexact],
    scanned: tuple[NDArray, ...],
    states: StateList | None = None,
    cache_level: tuple[int, int] = (1, 2),
    nterms: int | None = None,
) -> jax.Array:
    r"""Return :math:`Hv`, resolving the per-X-group inputs according to ``cache_level``.

    All six caching strategies are one ``jax.lax.scan`` over the X groups accumulating
    ``out + apply_xgrp(xsource, diagonal, vec)``; they differ only in where the two inputs come
    from. That is exactly the 2x3 grid ``cache_level`` already names, so it is expressed as a grid
    rather than as six near-identical functions:

    =============  ==========================  =====================================
    cache_level    ``xsource``                 ``diagonal``
    =============  ==========================  =====================================
    ``(0, *)``     ``get_xsource(x, states)``  --
    ``(1, *)``     scanned (precomputed)       --
    ``(*, 0)``     --                          ``get_diagonal(z, c, states)``
    ``(*, 1)``     --                          ``compute_diagonal(signs, c)``
    ``(*, 2)``     --                          scanned (precomputed)
    =============  ==========================  =====================================

    ``cache_level`` is static, so each combination traces to the same code the six separate kernels
    did -- the selection happens once at trace time, not per group. It must stay static: passing it
    as a traced value would retrace on every matvec call inside the solver loop.

    This is the resolved kernel: ``cache_level`` and the tuple layout are already agreed. It is
    private because that agreement is unenforceable at this boundary -- see :func:`apply_h`, the
    public entry point, which derives ``cache_level`` from named arrays so a mismatch cannot be
    expressed.

    Args:
        vec: Vector to multiply.
        scanned: Per-X-group arrays to scan over, in the order the ``cache_level`` grid implies.
            ``(0, 0)``: ``(xsignatures, zsignatures, coeffs)``. ``(0, 1)``:
            ``(xsignatures, diag_signs, coeffs)``. ``(0, 2)``: ``(xsignatures, diagonals)``.
            ``(1, 0)``: ``(xsources, zsignatures, coeffs)``. ``(1, 1)``:
            ``(xsources, diag_signs, coeffs)``. ``(1, 2)``: ``(xsources, diagonals)``.
        states: Uniquified state list. Required whenever either element of ``cache_level`` is 0,
            i.e. for every combination except ``(1, 1)`` and ``(1, 2)`` -- those two read neither the
            X signatures nor the Z signatures, so they need no states at all.
        cache_level: Caching strategy, as documented on :func:`sqd`.
        nterms: Static number of Z terms to sum per X group. Pass it to make this differentiable with
            respect to the coefficients -- without it the diagonal accumulation terminates on a
            data-dependent condition that reverse-mode autodiff rejects. Use
            ``max(hamiltonian.nzterms)``: the scan body traces once, so one trip count must serve
            every group, and the max is the tightest value that is correct for all of them. Ignored
            under ``cache_level[1] == 2``, where the diagonals are precomputed and no accumulation
            happens here. See :func:`_accumulate_diagonal` for the measurements.

    Returns:
        :math:`Hv`.

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
            diagonal = get_diagonal(val[1], val[2], states, nterms)
        elif cache_level[1] == 1:
            diagonal = compute_diagonal(val[1], val[2], nterms)
        else:
            diagonal = val[1]
        return out + apply_xgrp(xsource, diagonal, vec), None

    return jax.lax.scan(fn, jnp.zeros_like(vec), scanned)[0]


def apply_h(
    vec: NDArray[np.inexact],
    scanned: tuple[NDArray, ...] | None = None,
    states: StateList | None = None,
    cache_level: tuple[int, int] | None = None,
    nterms: int | None = None,
    *,
    xsources: NDArray[np.int32] | None = None,
    xsignatures: NDArray[np.uint8] | None = None,
    zsignatures: NDArray[np.uint8] | None = None,
    diag_signs: NDArray[np.uint8] | None = None,
    diagonals: NDArray[np.inexact] | None = None,
    coeffs: NDArray[np.inexact] | None = None,
) -> jax.Array:
    r"""Return :math:`Hv`, with the caching strategy named rather than positional.

    Name the per-X-group arrays you have and the strategy follows from them:

    .. code-block:: python

        apply_h(vec, xsources=xsrc, diag_signs=signs, coeffs=c, states=states)   # was (1, 1)
        apply_h(vec, xsignatures=x, diagonals=diags, states=states)              # was (0, 2)

    **Why this replaces the positional form.** ``cache_level`` selected, *positionally*, how the
    members of a ``scanned`` tuple were interpreted, and nothing checked that the tuple matched the
    level declared. Passing raw X signatures while claiming ``cache_level[0] == 1`` -- which promises
    precomputed X *sources* -- raised nothing and silently computed a different operator (measured
    0.44 max abs error on a 5-state subspace). An index array and a signature array are
    indistinguishable at that boundary: both are integer-typed with compatible shapes. Worse, the
    obvious cheap mitigation does not work -- a dtype or rank assertion cannot separate
    ``zsignatures`` from ``diag_signs``, since stacked they are *both* uint8 of rank 3, and shapes
    collide outright when the subspace size equals the packed byte width.

    Naming each representation makes the six valid combinations the only constructible ones, so a
    mispairing is a :exc:`TypeError` here instead of a wrong number later. Note the honest limit: this
    removes the *mispairing* class, not a caller who passes the wrong array under the right name.

    Args:
        vec: Vector to multiply.
        scanned: Deprecated. The positional tuple form, whose members are interpreted by position
            according to ``cache_level``: ``(0, 0)`` ``(xsignatures, zsignatures, coeffs)``;
            ``(0, 1)`` ``(xsignatures, diag_signs, coeffs)``; ``(0, 2)``
            ``(xsignatures, diagonals)``; ``(1, 0)`` ``(xsources, zsignatures, coeffs)``;
            ``(1, 1)`` ``(xsources, diag_signs, coeffs)``; ``(1, 2)`` ``(xsources, diagonals)``.
            Emits a :exc:`DeprecationWarning` -- pass the arrays by name instead.
        states: Uniquified state list. Required whenever the X signatures or the Z signatures are
            supplied, i.e. for every combination except ``xsources`` with ``diag_signs`` and
            ``xsources`` with ``diagonals`` -- those two read neither, so they need no states at all.
        cache_level: Deprecated, and only meaningful with ``scanned``.
        nterms: Static number of Z terms to sum per X group. Pass it to make this differentiable with
            respect to the coefficients; use ``max(hamiltonian.nzterms)``. Ignored when ``diagonals``
            is given, since no accumulation happens then. See :func:`_accumulate_diagonal`.
        xsources: Precomputed source indices per X group, shape ``(num_xgroups, num_states)``. The
            cheap axis to enable and very expensive to disable -- prefer it (see the module docs).
        xsignatures: Packed X signatures, shape ``(num_xgroups, num_bytes)``. The sources are derived
            per call, which measured 66-97% of an entire solve.
        zsignatures: Packed Z signatures per X group. Needs ``coeffs`` and ``states``.
        diag_signs: Precomputed packed sign bits per X group. Needs ``coeffs``.
        diagonals: Fully precomputed diagonals per X group. Needs neither ``coeffs`` nor ``states``.
        coeffs: Phase-folded coefficients per X group. Required with ``zsignatures`` or
            ``diag_signs``, and rejected with ``diagonals``.

    Returns:
        :math:`Hv`.

    Raises:
        TypeError: If the named arrays do not form exactly one of the six valid combinations -- none
            or both of the X-axis arrays, none or several of the diagonal-axis arrays, ``coeffs``
            missing where it is needed or supplied where it is not, or the named form mixed with
            ``scanned``/``cache_level``.
        ValueError: If ``states`` is needed but None.
    """
    if scanned is not None or cache_level is not None:
        named = {
            name: value
            for name, value in (
                ("xsources", xsources),
                ("xsignatures", xsignatures),
                ("zsignatures", zsignatures),
                ("diag_signs", diag_signs),
                ("diagonals", diagonals),
                ("coeffs", coeffs),
            )
            if value is not None
        }
        if named:
            raise TypeError(
                "apply_h: the deprecated scanned/cache_level form cannot be mixed with the named "
                f"arrays {sorted(named)}; pass only one form"
            )
        if scanned is None:
            raise TypeError("apply_h: cache_level requires scanned")
        warnings.warn(
            "apply_h's positional (scanned, cache_level) form is deprecated: the pairing is "
            "unchecked, and mispairing it silently computes a different operator. Pass the arrays "
            "by name instead (xsources=/xsignatures=, diagonals=/diag_signs=/zsignatures=, "
            "coeffs=).",
            DeprecationWarning,
            stacklevel=2,
        )
        # cache_level omitted with a tuple is legal in the old API; let the kernel's own default
        # apply rather than restating it here, where it could drift from the signature.
        if cache_level is None:
            return _apply_h_resolved(vec, scanned, states, nterms=nterms)
        return _apply_h_resolved(vec, scanned, states, cache_level, nterms)

    # Each axis is (name, cache_level digit, array). Pairing the three here means the axis is
    # resolved, validated and unpacked from one list, with no lookup table to keep in step.
    xaxis_options = [("xsources", 1, xsources), ("xsignatures", 0, xsignatures)]
    daxis_options = [
        ("diagonals", 2, diagonals),
        ("diag_signs", 1, diag_signs),
        ("zsignatures", 0, zsignatures),
    ]
    xgiven = [opt for opt in xaxis_options if opt[2] is not None]
    dgiven = [opt for opt in daxis_options if opt[2] is not None]
    if len(xgiven) != 1:
        raise TypeError(
            "apply_h: pass exactly one of xsources= or xsignatures= "
            f"(got {sorted(name for name, _, _ in xgiven) or 'neither'})"
        )
    if len(dgiven) != 1:
        raise TypeError(
            "apply_h: pass exactly one of diagonals=, diag_signs= or zsignatures= "
            f"(got {sorted(name for name, _, _ in dgiven) or 'none'})"
        )
    (_, xaxis, xarray), (dname, daxis, darray) = xgiven[0], dgiven[0]

    # coeffs is not optional-with-a-default: it is required by exactly two of the three diagonal
    # forms and meaningless in the third, so silently ignoring a stray one would hide a mistake.
    if daxis == 2:
        if coeffs is not None:
            raise TypeError(f"apply_h: coeffs= is not used with {dname}=; drop it")
    elif coeffs is None:
        raise TypeError(f"apply_h: {dname}= requires coeffs=")

    scanned = (xarray, darray) if daxis == 2 else (xarray, darray, coeffs)
    return _apply_h_resolved(vec, scanned, states, (xaxis, daxis), nterms)
