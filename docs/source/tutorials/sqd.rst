=========================================
Sample-based quantum diagonalization
=========================================

.. note::

   Outline only -- prose not yet written.

The idea
========

.. todo::

   Project a large Pauli-sum Hamiltonian onto the subspace spanned by a list of computational-basis
   bitstrings, then solve matrix-free. This makes tractable a Hamiltonian whose full space is far too
   large to diagonalize, provided the sampled bitstrings capture the ground state's support.

A first solve
=============

.. todo::

   End-to-end: build a Pauli sum, obtain bitstrings, call ``sqd(...)``, read the eigenvalue. Point
   ``hproj(...)`` out as the dense/debug path for checking small cases against a direct
   ``eigvalsh``.

Preparing the state list
========================

.. todo::

   Two hard requirements, both of which return a plausible wrong answer rather than raising:

   - **States must be lex-sorted.** ``get_xsource`` is a binary search, so unsorted input yields a
     non-symmetric projected matrix. Note ``hproj(unique_states=True)`` skips its ``np.unique`` and
     can therefore violate this.
   - **One extra zero pad bit at position 0** before ``packbits``. ``PauliSumXZ`` reserves the same
     bit unconditionally, so the two sides align by construction -- see
     :doc:`paulis`.

   Mention that filler slots from uniquification are ``255``, detected via ``states_u[:, 0] >> 7``.

Choosing ``cache_level``
========================

.. todo::

   ``cache_level=(source_indices, diagonals)`` selects among six matvec strategies over a 2×3 grid.
   It **must** stay static (bound via ``functools.partial``).

   Frame the tradeoff honestly, because the parameter's name misleads: it is usually described as
   memory-versus-speed, but measurement shows ``get_xsource`` *setup* dominates a solve --
   66--97% weighted by call count -- while ``matvec/J`` is flat. So the interesting question is
   usually how often setup runs, not which caching tier is chosen.

Scaling limits
==============

.. todo::

   ``N ≤ 2^31`` subspace states. The ``uint64``-key path applies for ``B ≤ 8`` bytes with an explicit
   lexicographic search beyond; that boundary is a **correctness** limit, since a ``uint64`` key
   silently truncates a wider row and aliases distinct states.

   Cross-reference ``docs/scaling-pocs.md`` for the measured numbers, including that the GPU speedup
   is 5.15× rather than the 12--25× a CPU-only reading suggests.

The initial vector
==================

.. todo::

   ``run_sqd`` uses a deterministic pseudo-random spread, not a one-hot, and why that matters: a
   one-hot cannot leave the connected component of the projected Hamiltonian that contains it, so a
   subspace whose Hamiltonian splits into disconnected blocks would silently return that block's
   minimum with ``converged=True``.

Multi-device execution
======================

.. todo::

   Mesh-size padding and the ``out_sharding`` contract. Point at
   ``examples/scaling/poc7_sharding.py`` as the harness, and note that timings under virtual devices
   are meaningless -- use it for correctness only.
