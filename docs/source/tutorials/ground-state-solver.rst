==========================
The ground state solver
==========================

.. note::

   Outline only -- prose not yet written.

When to use it
==============

.. todo::

   A single-vector (block-size-1) LOBPCG specialization: it finds the *lowest* eigenpair of a
   Hermitian operator supplied either as an array or as a matrix-free callable. Used directly when you
   have a large operator and want only the ground state; used indirectly as :mod:`~rqutils.sqd`'s
   eigensolver.

   Contrast with ``jnp.linalg.eigh``: use ``eigh`` for a small dense matrix or when you need the full
   spectrum. Use this when the operator is matrix-free, the vector is huge, or both.

Calling it
==========

.. todo::

   Both input forms -- a dense array, and a callable with ``args``. Cover ``xinit`` (including the
   integer one-hot convenience, which needs ``vspace`` when ``mat`` is callable), ``maxiter`` and
   ``tol``.

Reading the result
==================

.. todo::

   Returns ``(eigval, eigvec, niter, converged)``. **Check ``converged``**, not
   ``niter == maxiter`` -- the latter is ambiguous, since a run that converges on exactly the last
   permitted iteration is indistinguishable from one that ran out of budget.

   With ``debug=True`` a fifth element is appended, a dict of stacked per-iteration diagnostics.
   Note it uses ``jax.lax.scan`` with no early exit, so it always runs the full ``maxiter`` and rows
   past convergence are post-convergence noise.

Requirements on the initial vector
==================================

.. todo::

   Must have non-vanishing overlap with the target eigenvector. Explain what goes wrong otherwise and
   how :mod:`~rqutils.sqd` handles it (a pseudo-random spread rather than a one-hot; see
   :doc:`sqd`).

Matrix-free operators and sharding
==================================

.. todo::

   The contract that makes distributed execution work: the solver is sharding-transparent **only if
   the ``mat`` callable preserves output sharding**. Show the pattern -- pass
   ``out_sharding=jax.typeof(vec).sharding`` -- which is exactly what every ``apply_*`` in
   :mod:`~rqutils.sqd` does.

Why the Rayleigh-Ritz step is a closed form
===========================================

.. todo::

   The 3×3 projected eigenproblem is solved analytically (``eigenpair_2x2``, ``eigenpair_3x3`` via
   Cardano) rather than with ``eigh``. The reason is *not* compile time, which measurement reversed:
   ``eigh`` lowers to an unfusable FFI custom call inside the ``while_loop`` body, whereas
   nine-number scalar arithmetic folds into the surrounding kernels.

Numerical guards, and why not to remove them
============================================

.. todo::

   Aimed at contributors. Every guard is load-bearing and was measured; each defect they prevent
   failed *silently*, returning a plausible number rather than raising. Cover balancing in the
   eigenpair kernels, the re-orthogonalization, the search-direction renormalization, and the
   zeroed-direction masks.

   Warn that most are algebraically no-ops, so checking the algebra and stopping there is exactly the
   mistake to avoid. ``docs/locg.md`` records the original measurements but is a stale audit of the
   pre-rewrite module -- the module docstring is authoritative.

The MLX port
============

.. todo::

   :mod:`rqutils.ground_locg_mlx` duplicates this algorithm for Apple GPUs and is real-symmetric
   only. Its whole option surface is one ``device="cpu"|"gpu"`` parameter, and ``"cpu"`` is the only
   route to an f64 solve since Metal has no float64. Note it is a deliberate duplicate: change both
   together.
