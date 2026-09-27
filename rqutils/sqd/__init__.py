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

That initial vector is a deterministic pseudo-random spread over the subspace (:func:`_spread_seed`),
with the minimum-diagonal state weighted heavily on top when the Hamiltonian has a diagonal part --
not a one-hot, which cannot leave its connected component of the projected Hamiltonian (``NOTES.md``,
"``sqd``: why the initial vector is a spread, not a one-hot").

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

Caching the source indices :math:`[j^{i}]` takes :math:`4 J N` bytes (int32 indices, since
:math:`N < 2^{31}`; see the next section). Caching the composed diagonals :math:`C^{(j)}` takes
:math:`8 N` bytes for each X group with real coefficients and :math:`16 N` for each complex one (a
group holding a term with an odd number of Ys): the real groups come first
(:attr:`~rqutils.paulis.symplectic.PauliSumXZ.num_real_groups`) and are held as ``float64``. Caching
both also frees :math:`S`, which is no longer read, so at small :math:`n` the full cache can be the
cheaper option in memory too.

``matvec`` names the kernel by what it stores, as a :class:`Matvec` member:

- ``Matvec.ONTHEFLY``: nothing; the sources are searched and the diagonals computed per matvec.
- ``Matvec.INDICES`` (the default): the source indices :math:`[j^{i}]` of every X group.
- ``Matvec.TABLES``: the source indices and the composed diagonals :math:`C^{(j)}`.
- ``Matvec.PAIRS``: each transition :math:`(a, b)`, :math:`a < b`, of a non-identity X group once, with
  :math:`d = C^{(j)}_a`, applied as :math:`v'_a \mathrel{+}= d v_b` and
  :math:`v'_b \mathrel{+}= \bar{d} v_a` -- exact, since :math:`H_{ba} = \overline{H_{ab}}` per X signature.
- ``Matvec.CSR``: both directions of every transition, sorted by target, with ``float64`` factors for
  the all-real groups and ``complex128`` for the rest.
- ``Matvec.ELL``: the same transitions and split, with rows bucketed by degree rounded up on a
  :math:`\times 1.25` grid into dense ``(rows, width)`` blocks, so each row is one gathered sum rather
  than a scatter per entry (``poc/sparse-pairs.md``, section 10).

The other dense storage combinations are dominated on both memory and time (``NOTES.md``). The last
three store only the transitions that land inside the subspace, where the source indices mostly hold
the ``-1`` absent marker; :func:`sqd` builds them host-side before the solve, they are single-device
for now, and ``poc/sparse-pairs.md`` has the measurements.

**The source-index setup dominates the solve, so this is not a symmetric memory-for-speed dial:**
``Matvec.ONTHEFLY`` pays the :math:`J`-fold :func:`get_xsource` search once per matvec rather than
once per solve. Prefer ``Matvec.INDICES`` or ``Matvec.TABLES`` unless the memory genuinely will
not fit (``NOTES.md``, "``sqd``: ``get_xsource`` setup dominates a solve";
``markdown/scaling-pocs.md``).

Distributed arrays and scaling limits
=====================================

When the SQD function is called within a context where the global mesh is set via
``jax.set_mesh(mesh)``, the state vector is distributed (sharded) among the devices in the mesh,
and accordingly all arrays with an axis with size :math:`N` follow the same sharding. Even the most
aggressive caching strategy described above will be possible this way.

However, :math:`N` (the SQD subspace dimension) is capped at :math:`2^{31} - 1`: subspace positions
are int32, and :func:`uniquify_states`' sort runs on a single device over at most :math:`2^{32}`
elements (:func:`get_xsource` is a binary search and does not contribute). GPU memory, at most
O(100)GB per device as of mid-2026, sets a comparable limit (``NOTES.md``, "The ``N ≤ 2^31 - 1``
ceiling").

When the source indices are cached, the state list :math:`S` is also sharded once they are computed.

SQD API
=======

.. autofunction:: sqd
.. autoexception:: EigenpairCheckError
.. autoclass:: Matvec
   :members:
.. autofunction:: hproj

States are packed with :meth:`~rqutils.paulis.symplectic.PauliSumXZ.pack_states`, which inserts the
pad bit that aligns them with the Hamiltonian's signatures, and recovered with
:meth:`~rqutils.paulis.symplectic.PauliSumXZ.unpack_states`.
"""

from rqutils.sqd._core import LOG, EigenpairCheckError, HamiltonianInput, Vector, hproj, sqd
from rqutils.sqd._dense import apply_h, apply_xgrp
from rqutils.sqd._diagonal import get_diagonal
from rqutils.sqd._solve import Matvec, SqdResult, run_sqd
from rqutils.sqd._states import StateList, get_xsource, uniquify_states

__all__ = [
    "LOG",
    "EigenpairCheckError",
    "HamiltonianInput",
    "Matvec",
    "SqdResult",
    "StateList",
    "Vector",
    "apply_h",
    "apply_xgrp",
    "get_diagonal",
    "get_xsource",
    "hproj",
    "run_sqd",
    "sqd",
    "uniquify_states",
]
