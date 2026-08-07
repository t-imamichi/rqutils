==================================
Pretty-printing quantum objects
==================================

.. note::

   Outline only -- prose not yet written.

.. note::

   ``output='mpl'`` needs the ``mpl`` extra; passing a QuTiP ``Qobj`` needs ``qutip``.

Two orthogonal choices
======================

.. todo::

   The API is a grid, and reading it as one axis is the usual mistake:

   - ``fmt`` picks the **content class** -- ``'braket'``, ``'pauli'`` or ``'matrix'``.
   - ``output`` picks the **rendering** -- ``'text'``, ``'latex'`` or ``'mpl'``.

   Any combination is valid. Present it as a table.

Printing a state vector
=======================

.. todo::

   ``fmt='braket'`` worked examples: a small state in text, then the same as LaTeX in a notebook.
   Cover amplitude formatting and how basis labels are chosen.

Printing an operator
====================

.. todo::

   ``fmt='pauli'`` for a Pauli decomposition (cross-reference :doc:`paulis` for the basis
   conventions) and ``fmt='matrix'`` for a matrix layout.

Choosing an output mode
=======================

.. todo::

   - ``'text'`` returns the object, rendering lazily via ``__repr__`` -- so it displays on its own in
     a REPL but needs ``print()`` or ``str()`` inside other code.
   - ``'latex'`` returns a string.
   - ``'mpl'`` returns a Figure you can further style or save.

Accepted inputs
===============

.. todo::

   numpy arrays, JAX arrays, and QuTiP ``Qobj`` when ``qutip`` is installed. Explain the guarded
   optional-dependency pattern: a missing package raises ``RuntimeError`` at call time rather than
   failing at import.

Formatting details worth knowing
================================

.. todo::

   An amplitude of exactly ``1`` is suppressed -- correct when a basis label follows, wrong when
   nothing does. Text-mode labels also carry the ``*`` separator as a prefix, so the text and LaTeX
   renderings can disagree while each looks fine in isolation.

Extending it
============

.. todo::

   For contributors: ``QPrintBase`` owns all the numerics and subclasses override only
   ``_qobj_data``, ``_add_labels`` and ``_format_lhs``. Note ``QPrintMatrix`` overrides
   ``_make_lines`` instead, positioning terms by row and column, which is why it does not implement
   ``_add_labels``.
