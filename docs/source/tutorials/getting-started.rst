===============
Getting started
===============

.. note::

   Outline only -- prose not yet written.

Installation and the optional extras
====================================

.. todo::

   Cover: ``pip install rqutils``; the extras (``mpl``, ``qutip``, ``qiskit``, ``mlx``, ``docs``,
   ``dev``) and which module each one unlocks; that ``numpy``, ``scipy``, ``h5py`` and ``jax`` are
   hard dependencies so nothing needs an extra to import; and that ``mlx`` is darwin-only.

Enabling 64-bit precision
=========================

.. todo::

   ``jax.config.update('jax_enable_x64', True)`` must run before any ``rqutils`` import. Explain the
   failure it prevents -- without it you silently get ``complex64``/``int32`` and truncation
   warnings -- and that the test suite's tolerances all assume it.

The two Pauli representations
=============================

.. todo::

   The single most common source of confusion: :mod:`rqutils.paulis.general` (dense, any dimension,
   Gell-Mann-like) and :mod:`rqutils.paulis.symplectic` (bit-packed, qubits only, for JAX/GPU) are
   unrelated. Give a one-table decision guide and link to :doc:`paulis`.

Choosing a module for your problem
==================================

.. todo::

   A short decision guide: simulate a circuit → :doc:`svsim`; find a ground state in a sampled
   subspace → :doc:`sqd`; diagonalize a matrix-free operator directly → :doc:`ground-state-solver`;
   display a result → :doc:`qprint`.

A first end-to-end example
==========================

.. todo::

   Build a small Hamiltonian, find its ground state, and print it -- exercising
   :mod:`~rqutils.paulis.symplectic`, :mod:`~rqutils.sqd` and :mod:`~rqutils.qprint` in one snippet
   so the pieces are seen fitting together before any single module is covered in depth.

Running on multiple devices
===========================

.. todo::

   Sharding is implicit: the library reads ``jax.sharding.get_abstract_mesh()`` and it is the
   *caller's* job to set the mesh. Show the expected pattern -- a single axis named ``'x'`` with
   ``AxisType.Explicit`` -- and note that multi-device paths can be exercised on CPU with
   ``XLA_FLAGS=--xla_force_host_platform_device_count=4``.
