==========
Tutorials
==========

Task-oriented introductions to each module. The :doc:`API reference <../index>` documents *what*
every function accepts and returns; these pages cover *when* you would reach for it and how the
pieces fit together.

Each module's own docstring carries the mathematical conventions -- normalization, qubit ordering,
phase conventions -- and is the authority when a tutorial and a docstring disagree.

.. toctree::
   :maxdepth: 1
   :caption: Getting started

   getting-started

.. toctree::
   :maxdepth: 1
   :caption: Working with Pauli operators

   paulis

.. toctree::
   :maxdepth: 1
   :caption: Simulation and diagonalization

   svsim
   sqd
   ground-state-solver

.. toctree::
   :maxdepth: 1
   :caption: Presenting results

   qprint

Where to start
==============

- New to the library: :doc:`getting-started`, then the tutorial for whichever module you need.
- Diagonalizing a large Hamiltonian: :doc:`sqd`, which uses :doc:`ground-state-solver` as its
  eigensolver and :doc:`paulis` for its operator input.
- Simulating a circuit: :doc:`svsim`.
- Displaying a state or operator: :doc:`qprint`.

Conventions used in these tutorials
===================================

Every runnable example assumes 64-bit precision is enabled before any ``rqutils`` import, since
several modules silently produce ``complex64``/``int32`` results otherwise:

.. code-block:: python

   import jax
   jax.config.update('jax_enable_x64', True)

Examples that need an optional dependency say so at the top of the page. Install the corresponding
extra rather than the package directly, so the pinned version is respected:

.. code-block:: console

   (.venv) $ pip install 'rqutils[qiskit]'    # or mpl, qutip, mlx
