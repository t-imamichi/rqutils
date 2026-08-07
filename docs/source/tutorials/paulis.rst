=========================
Working with Pauli bases
=========================

.. note::

   Outline only -- prose not yet written.

Which representation do you need?
=================================

.. todo::

   Lead with the decision, because the two modules share a name and nothing else:

   - :mod:`rqutils.paulis.general` -- dense generalized (Gell-Mann-like) basis for *arbitrary*
     subsystem dimension. Use for qutrits and higher, for decomposing a dense operator into
     components, and for human-readable labels.
   - :mod:`rqutils.paulis.symplectic` -- ``PauliSumXZ``, bit-packed and qubit-only, built for
     JAX/GPU matrix-free work. Use when feeding :mod:`~rqutils.sqd` or :mod:`~rqutils.svsim`.

The generalized (Gell-Mann) basis
=================================

.. todo::

   Cover ``paulis(dim)``, ``pauli_matrices(dim)``, ``components()`` and ``labels()``.

   Emphasize the normalization, which is the most bug-prone invariant in the module:
   :math:`\mathrm{tr}(\lambda_k \lambda_l) = 2\delta_{kl}`, so
   :math:`\lambda_0 = \sqrt{2/n}\,I` and **not** :math:`I`.

   Shapes are easy to transpose: ``paulis(dim)`` returns basis axes *first*
   (``(d1², …, D, D)``) while component arrays put component axes *last* (``(…, d1², …)``).

Decomposing an operator into components
=======================================

.. todo::

   Worked example with ``components()``, then ``labels()`` to read the result. Mention that
   basis-index ordering is fixed by a shell-by-shell construction and that ``components`` and
   ``labels`` both index by basis position, so they stay consistent with each other.

Traceable execution with ``npmod``
==================================

.. todo::

   ``components`` accepts ``npmod=jax.numpy`` for traceable/jit-compatible execution. Show it under
   ``jax.jit``. Note the limits: multi-subsystem ``paulis(dim)`` uses ``np.einsum`` with three
   letters per subsystem and so caps around 17 subsystems, and ``sparse=True`` for products raises
   ``NotImplementedError``.

The symplectic representation
=============================

.. todo::

   ``PauliSumXZ.from_paulisum`` and the ``arrays`` property. Explain the convention
   :math:`Q = (-i)^{x \cdot z} Z^z X^x` with **little-endian** qubit ordering, so Qiskit's ``.x`` /
   ``.z`` are reversed on ingest -- a silent wrong-answer source if assumed otherwise.

   Explain the storage: terms grouped by unique X signature, Z groups zero-padded to a rectangle,
   the :math:`(-i)^{\mathrm{popcount}(x \wedge z)}` phase folded into the coefficients, then
   ``np.packbits``.

Reading a packed signature
==========================

.. todo::

   The pad bit and the bit order are the two things that bite. Signatures always reserve one pad bit
   at position 0, aligning with the pad bit consumers put in their state bitstrings -- not optional.
   And ``packbits`` fills each byte from the *most significant* end, so payload bits are entries 1
   through ``num_qubits`` of ``np.unpackbits`` in string-character order, the reverse of the qubit
   numbering.

Hermiticity and coefficient dtype
=================================

.. todo::

   A complex coefficient raises: the sum would be non-Hermitian. ``.c`` narrows to ``float64``
   exactly when every string has an even Y count, so check ``.c.dtype`` rather than assuming.
