==========================
State vector simulation
==========================

.. note::

   Outline only -- prose not yet written.

.. note::

   The ``QuantumCircuit`` input path needs the ``qiskit`` extra.

What this simulator is for
==========================

.. todo::

   A JAX state-vector simulator built for scale: gates compile to a symplectic ``CircuitXZ``
   representation and are applied by a single ``jax.lax.scan``, so the whole circuit is one compiled
   loop rather than a Python loop over gate applications.

The supported gate set
======================

.. todo::

   Only ``x, y, z, cz, rx, ry, rz, rzz`` are supported, because every gate must compile to the
   ``(x, z, cos, sin)`` form. **Transpile first** to
   ``basis_gates=['rx', 'ry', 'rz', 'rzz']`` -- show the ``qiskit.transpile`` call, since an
   unsupported gate is the first thing a new user hits.

Running a circuit
=================

.. todo::

   Worked example: build a circuit, transpile it, simulate, inspect the state vector. Then the
   same thing written directly as a gate spec, bypassing qiskit.

Saving results
==============

.. todo::

   The HDF5 output path (``h5py`` is a hard dependency, so this needs no extra). Cross-reference
   ``examples/svsim.py``.

Conventions that matter
=======================

.. todo::

   Two that produce silently wrong answers if assumed away:

   - ``sin`` is **complex128**, not real: it carries
     :math:`i \cdot (-i)^{\mathrm{popcount}(x \wedge z)}`, folding in both the rotation's leading
     :math:`i` and the phase of the :math:`Q = (-i)^{x \cdot z} Z^z X^x` convention. Omitting it
     breaks every ``y``/``ry`` gate -- the only gates with overlapping X and Z signatures -- and so
     every transpiled circuit.
   - ``cz`` is decomposed only on the ``QuantumCircuit`` path, and is correct only up to a uniform
     :math:`e^{i\pi/4}` that its ``rzz``+``rz`` decomposition cannot express.

Multi-device execution
======================

.. todo::

   Set the mesh as described in :doc:`getting-started`. Flag honestly that ``svsim``'s
   ``out_sharding`` path is currently not covered by any test.
