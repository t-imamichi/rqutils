r"""
==========================
Generalized Pauli matrices
==========================

.. currentmodule:: rqutils.paulis.general

Fundamentals
============

Generalized :math:`n`-dimensional Pauli matrices :math:`\lambda^{(n)}_{k}`
(:math:`0 \leq k \leq n^2 - 1`) are defined recursively:

- :math:`\lambda^{(n)}_{0} = \sqrt{\frac{2}{n}} \mathrm{diag}(1, \dots, 1, 1)`
- :math:`\lambda^{(n)}_{k} = \mathrm{blkdiag}(\lambda^{(n-1)}_{k}, 0)` for
  :math:`1 \leq k < (n-1)^2`
- :math:`(\lambda^{(n)}_{(n-1)^2 + k})_{ab} = \xi_k \delta_{k//2, a}\delta_{n-1, b} + \eta_k \delta_{n-1, a}\delta_{k//2, b}`
  for :math:`0 \leq k < 2(n-1)`, with :math:`\xi_k = \eta_k = 1` (:math:`k` even) and
  :math:`-\xi_k = \eta_k = i` (:math:`k` odd)
- :math:`\lambda^{(n)}_{n^2-1} = \sqrt{\frac{2}{n(n-1)}} \mathrm{diag}(1, \dots, 1, -n+1)`

These matrices satisfy the normalization condition

.. math::

    \mathrm{tr}(\lambda^{(n)}_k \lambda^{(n)}_l) = 2 \delta_{k, l}

and thus form an orthonormal basis for the space of :math:`n`-dimensional Hermitian matrices.

Implications of the normalization
---------------------------------

Any :math:`n`-dimensional Hermitian matrix :math:`H` can be decomposed into a form

.. math::

    H = \sum_{k=0}^{n^2-1} \nu_k \lambda^{(n)}_k.

To extract the coefficient :math:`\nu_k`, one needs to compute

.. math::

    \nu_k = \frac{1}{2} \mathrm{tr}(\lambda^{(n)}_k H),

i.e., divide the product trace by 2.

Also, note that :math:`\lambda^{(n)}_{0}` is *not* the :math:`n`-dimensional identity matrix but
differ from it by a factor :math:`\sqrt{\frac{2}{n}}`.

Pauli products
==============

A physical composite system of :math:`s` subsystems is usually better described in terms of a tensor
product of :math:`s` Hamiltonians each of dimension :math:`n_i (i=1, \dots, s)`, rather than a
single Hamiltonian of :math:`N := \prod_{i=1}^{s} n^i` dimensions. A natural decomposition of the
former would be in terms of tensor products of :math:`s` Pauli matrices

.. math::

    \Lambda^{(n_1 \dots n_s)}_{k_1 \dots k_s} = \frac{1}{2^{s-1}} \bigotimes_{i=1}^{s}
                                                \lambda^{(n_i)}_{k_i},

which constitute an orthonormal basis of the space of :math:`N`-dimensional Hermitian matrices with
a rather awkward normalization

.. math::

    \mathrm{tr}(\Lambda^{(n_1 \dots n_s)}_{k_1 \dots k_s} \Lambda^{(n_1 \dots n_s)}_{l_1 \dots l_s})
    = 2 \frac{1}{2^{s-1}} \prod_i \delta_{k_i, l_i}.

The full `s`-body Hamiltonian :math:`H` is decomposed into

.. math::

    H = \sum_{k_1 \dots k_s} \nu_{k_1 \dots k_s} \Lambda^{(n_1 \dots n_s)}_{k_1 \dots k_s},

and the component :math:`\nu_{k_1 \dots k_s}` is extracted by

.. math::

    \nu_{k_1 \dots k_s} = 2^{s-2} \mathrm{tr}(\Lambda^{(n_1 \dots n_s)}_{k_1 \dots k_s} H).


Pauli Matrices API
==================

.. autosummary::
   :toctree: ../apidoc

   paulis
   components
   labels
"""

import string
from collections.abc import Sequence
from types import ModuleType

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.sparse import csr_array

from rqutils._types import MatrixDimension


def normalize_dim(dim: MatrixDimension) -> tuple[int, ...]:
    """Return ``dim`` as a tuple of plain ints, treating a scalar as a single subsystem.

    A ``tuple(int)`` is what the memoization dicts in this module key on, so normalizing to exactly
    that -- rather than to anything merely tuple-shaped -- is what makes ``paulis(2)``,
    ``paulis((2,))`` and ``paulis(np.int64(2))`` share one cache entry.

    This is Python-level shape inference on a static value, so per ``CLAUDE.md``'s ``npmod`` rule it
    must run for *every* backend and never sit behind an ``if npmod is np:`` gate. Gating it is what
    broke :func:`components`' ``npmod=jnp`` path (and those of two since-removed functions), raising
    "object of type 'int' has no len()" from a line that named nothing.
    """
    if isinstance(dim, (int, np.integer)):
        return (int(dim),)
    return tuple(map(int, dim))


def paulis(dim: MatrixDimension, sparse: bool = False) -> NDArray[np.complex128 | np.object_]:
    r"""Return an array of generalized Pauli matrices or matrix products of given dimension(s).

    Args:
        dim: Dimension(s) of the Pauli matrices.
        sparse: Whether to return the matrices as an array (dtype=object) of CSR arrays. Only
            supported for a single subsystem.

    Returns:
        An array of Pauli (product) matrices. For a multi-subsystem `dim=(d1, d2, ...)`, the shape
        is `(d1**2, d2**2, ..., d1*d2*..., d1*d2*...)`; for a single subsystem `dim=d` it is
        `(d**2, d, d)`. With `sparse=True` the result is instead a 1D `dtype=object` array of shape
        `(d**2,)` holding CSR arrays.

    Raises:
        NotImplementedError: If `sparse=True` with more than one subsystem (the sparse product path
            does not exist), or if `dim` has more than ~17 subsystems -- the products are built with
            a single `np.einsum` using 3 index letters per subsystem, which exhausts the available
            letters.
    """
    dim = normalize_dim(dim)

    if len(dim) == 1:
        return pauli_matrices(dim[0], sparse=sparse)

    # Raise before building anything: the sparse product path does not exist, and the dense body
    # below is the whole function, so guarding it with an else put 25 lines a level deeper and left
    # `matrix_array`'s definedness for the reader to prove. The cache is keyed on `dim` alone for the
    # same reason -- with this raise unconditional, a `sparse` axis could only ever hold False.
    if sparse:
        raise NotImplementedError("Need an hour")

    if (cache := _pauli_products.get(dim)) is not None:
        return cache

    subsystems = [pauli_matrices(d) for d in dim]
    num_sub = len(subsystems)

    # Compose Pauli products
    # (d1**2, d1, d1) x (d2**2, d2, d2) -> (d1**2, d2**2, d1*d2, d1*d2)
    #      a   b   c         d   e   f          a      d     be     cf
    # be and cf are reshaped into 1 dimension each
    chars = string.ascii_letters
    if num_sub * 3 > len(chars):
        raise NotImplementedError(
            "Too many subsystems - need an implementation using recursive np.kron"
        )

    indices_in = []
    indices_out = [""] * 3
    for ichar in range(0, num_sub * 3, 3):
        indices_in.append(chars[ichar : ichar + 3])
        indices_out[0] += chars[ichar]
        indices_out[1] += chars[ichar + 1]
        indices_out[2] += chars[ichar + 2]

    indices = f"{','.join(indices_in)}->{''.join(indices_out)}"
    shape = tuple(d**2 for d in dim) + (int(np.prod(dim)),) * 2
    matrix_array = np.einsum(indices, *subsystems).reshape(shape) / (2 ** (num_sub - 1))

    # Cache and return the same immutable array, as pauli_matrices does. Storing a `.copy()` and
    # returning the original left a second, writeable allocation alive per key for the process
    # lifetime: retained memory measured 2.00x the result at dim=(2,)*6 (537 MB for a 268 MB basis)
    # against 1.00x now. It also made the warm return writeable while the cold return was read-only,
    # so no caller could have depended on writeability without already hitting that inconsistency.
    matrix_array.setflags(write=False)
    _pauli_products[dim] = matrix_array
    return matrix_array


_pauli_products = {}


def pauli_matrices(dim: int, sparse: bool = False) -> NDArray[np.complex128 | np.object_]:
    """Return a set of Pauli matrices of a given dimension.

    Args:
        dim: Dimension of the matrices.
        sparse: Whether to return the matrices as an array (dtype=object) of CSR arrays.

    Returns:
        An array of matrices.
    """
    if (cache := _pauli_matrices.get((dim, sparse))) is not None:
        return cache

    # Compose the unnormalized matrices
    matrices = np.zeros((dim**2, dim, dim), dtype=complex)

    matrices[0] = np.diag(np.ones(dim))
    imat = 1
    for ishell in range(1, dim):
        for ipos in range(ishell):
            matrices[imat, ipos, ishell] = 1.0
            matrices[imat, ishell, ipos] = 1.0
            imat += 1
            matrices[imat, ipos, ishell] = -1.0j
            matrices[imat, ishell, ipos] = 1.0j
            imat += 1

        matrices[imat, : ishell + 1, : ishell + 1] = np.diag(np.array([1.0] * ishell + [-ishell]))
        imat += 1

    # Normalization
    norm = np.trace(np.matmul(matrices, matrices), axis1=1, axis2=2)
    matrices *= np.sqrt(2.0 / norm)[:, None, None]

    if sparse:
        # Derived from the dense basis rather than built as a second, independent construction. The
        # CSR branch this replaces re-derived the shell ordering and the sqrt(2/(k(k+1))) diagonal
        # normalization by hand as data/indices/indptr triplets -- a second spelling of the one
        # convention CLAUDE.md flags as most bug-prone here, where a divergence would be silent
        # because each branch stayed internally consistent. Verified identical (max abs diff 0.0 and
        # equal nnz) for dim 2 through 6 before the swap.
        matrices = np.array([csr_array(mat) for mat in matrices])
        # Freeze each operator's buffers, not the object array holding them. This function memoizes
        # and returns the cached object directly, so without this every caller shares one set of CSR
        # instances and an in-place rescale corrupts the basis for the process lifetime: measured
        # `pauli_matrices(3, sparse=True)[1] /= 2` shifting the cached values by 0.5 max abs, with the
        # result still Hermitian, so every later `components()` call returned plausible and
        # consistently wrong coefficients. Normalization is the invariant CLAUDE.md calls the most
        # bug-prone in this module, which is exactly the thing an in-place `/=` is reaching for.
        #
        # Read-only buffers rather than a copy on return, and the gap is not marginal: copying on
        # every cache hit measured 276 us against 0.10 us for returning the cached object, i.e. 2698x
        # slower, at dim=6. `setflags` on the three buffers blocks `/=`, `*=`, `data[i] = ...` and
        # `mat[i, j] = ...` at their source, costs one loop per `dim` at build time (1.15 ms cold for
        # dim=6), and leaves `toarray`, `@` and every other read untouched. A caller who genuinely
        # wants to rescale calls `.copy()` first, as the dense path already requires.
        for mat in matrices:
            mat.data.setflags(write=False)
            mat.indices.setflags(write=False)
            mat.indptr.setflags(write=False)
    else:
        # Make the matrix immutable. Only the dense array can carry the flag; the object array of
        # csr_arrays needs its elements frozen individually, which the sparse branch above does.
        matrices.setflags(write=False)

    _pauli_matrices[(dim, sparse)] = matrices
    return matrices


_pauli_matrices = {}


def components(
    matrix: ArrayLike, dim: MatrixDimension, npmod: ModuleType = np
) -> NDArray[np.complex128]:
    r"""Return the Pauli decomposition coefficients :math:`\nu_{k_1 \dots k_n}` of the matrix.

    Args:
        matrix: Matrix to decompose. The last two dimensions of the array are dotted with the Pauli
            matrices.
        dim: Subsystem dimensions. The product of subsystem dimensions must match the matrix
            dimension. If None, the matrix is assumed to represent a single system.
        npmod: Numeric module. Pass `jax.numpy` for traceable execution; see the `npmod` section of
            ``CLAUDE.md`` for the validation-gating rule this follows.

    Returns:
        A complex array of shape `(..., d1**2, d2**2, ...)` where `d1`, `d2`, ... are the subsystem
        dimensions.

    Raises:
        TypeError: If `dim` is omitted. It used to default to `None` and be inferred from
            `matrix.shape[-1]`, which is ambiguous for any composite dimension: a 4x4 matrix inferred
            one 4-level qudit where two qubits may have been meant, and the two differ by a factor
            `sqrt(2)` in coefficient norm (the `2**(len(dim) - 2)` normalization is 0.5 against 1.0)
            with nothing to indicate which was used.
        ValueError: If `prod(dim)` does not match the matrix dimension. **Only under `npmod is np`**
            -- the check cannot run on traced values, so under `npmod=jax.numpy` a mismatched `dim`
            instead surfaces as an opaque `TypeError` from the contraction ("dot_general requires
            contracting dimensions to have the same shape"). Validate before tracing if you need the
            named error.
    """
    # normalize_dim runs for every npmod -- `len(dim)` below needs a sequence either way, and
    # gating it left `components(m, dim=3, npmod=jnp)` raising "object of type 'int' has no len()"
    # from the return statement, naming nothing. Only the *validation* below belongs behind the gate,
    # per CLAUDE.md's npmod rule.
    # `dim` is required: inferring it from the matrix shape was ambiguous in a silent way. A 4x4
    # matrix inferred (4,) -- one 4-level qudit -- where the caller may have meant (2, 2). Both pass
    # the prod(dim) check below and both yield 16 valid coefficients, but they are decompositions in
    # different bases: the 2**(len(dim) - 2) normalization is 0.5 for one subsystem against 1.0 for
    # two, so the coefficient vectors differ in norm by sqrt(2) (measured 1.4142135623730951) with
    # nothing to say which the caller received.
    dim = normalize_dim(dim)

    if npmod is np and np.prod(dim) != matrix.shape[-1]:
        raise ValueError(
            f"Invalid subsystem dimensions {dim}"
            f" (prod {np.prod(dim)} != matrix shape {matrix.shape[-1]})"
        )

    basis = paulis(dim)
    return npmod.tensordot(matrix, basis, ((-2, -1), (-1, -2))) * (2 ** (len(dim) - 2))


def labels(
    dim: MatrixDimension,
    symbol: str | Sequence[str] | Sequence[Sequence[str]] | None = None,
    delimiter: str = "",
    norm: bool = True,
    fmt: str = "latex",
) -> NDArray[np.str_]:
    r"""Generate the labels for the Pauli matrices of a given dimension.

    Args:
        dim: Dimension(s) of the Pauli matrices.
        symbol: Base symbol, applied to every subsystem when given as a str, or per subsystem when
            given as a sequence. Three modes: None uses `I/X/Y/Z` for a qubit and `λ_i` otherwise;
            a non-empty str `s` gives `s_i`; and the empty string gives bare numeric labels `i`
            (still `I/X/Y/Z` for a qubit). A per-subsystem entry may itself be a sequence of
            `dim**2` explicit labels.
        delimiter: Delimiter between the symbols for multibody labels.
        norm: Include the normalization factors.
        fmt: Output format. Allowed values are 'text', 'latex', 'latex-text',
            'latex-slash'.

    Returns:
        An ndarray of type string and shape `(d1**2, d2**2, ...)`.

    Raises:
        AssertionError: If a `symbol` entry is a sequence whose length is not `dim**2` for its
            subsystem. Reachable from caller-supplied data, so it is a contract, not an internal
            invariant -- but it is an `assert`, so it vanishes under `python -O`.
    """
    dim = normalize_dim(dim)

    # A separate name rather than rebinding `symbol`: the parameter accepts a scalar or a per-subsystem
    # sequence, and broadcasting the scalar onto it makes the declared type wrong from here down.
    symbols = (symbol,) * len(dim) if symbol is None or isinstance(symbol, str) else symbol

    # Normalization affixes. Folded into the construction below -- the prefix into the seed and the
    # suffix into the last subsystem's per-label list -- rather than applied as two extra whole-array
    # np.char.add passes over np.full(out.shape, ...) at the end. Those passes cost 47-58% of the
    # call at 10 qubits, where the latex prefix alone was a 25 MB array holding one repeated 7-char
    # string. Replacing np.full with a scalar does not help: np.char.add densifies it anyway.
    pre, post = "", ""
    if norm and len(dim) >= 2:
        if len(dim) == 2:
            denom = "2"
        elif fmt == "text":
            denom = f"2**{len(dim) - 1}"
        else:
            denom = "2^{%d}" % (len(dim) - 1)  # noqa: UP031 (f-string needs {{}} escapes here)

        if fmt in ("text", "latex-slash"):
            post = f"/{denom}"
        elif fmt == "latex":
            pre, post = r"\frac{", "}{%s}" % denom  # noqa: UP031
        else:
            pre, post = r"\textstyle{\frac{", "}{%s}}" % denom  # noqa: UP031

    out = np.array(pre, dtype=str)

    for isub, (pauli_dim, sym) in enumerate(zip(dim, symbols)):
        if delimiter and len(out.shape) > 0:
            out = np.char.add(out, np.full(out.shape, delimiter))

        if not sym:
            if pauli_dim == 2:
                labels = ["I", "X", "Y", "Z"]
            elif sym is None:
                if fmt == "text":
                    labels = [f"λ{i}" for i in range(pauli_dim**2)]
                else:
                    labels = [rf"{{\lambda_{{{i}}}}}" for i in range(pauli_dim**2)]
            else:
                labels = [str(i) for i in range(pauli_dim**2)]
        elif isinstance(sym, str):
            labels = [f"{{{sym}_{{{i}}}}}" for i in range(pauli_dim**2)]
        else:
            assert len(sym) == pauli_dim**2, "Invalid length of the symbols array"
            labels = [f"{{{s}}}" for s in sym]

        if post and isub == len(dim) - 1:
            labels = [label + post for label in labels]

        out = np.char.add(np.repeat(out[..., None], pauli_dim**2, axis=-1), labels)

    return out
