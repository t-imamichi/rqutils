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


def _normalize_dim(dim: MatrixDimension) -> tuple[int, ...]:
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


def paulis(dim: MatrixDimension, sparse: bool = False) -> NDArray[np.complex128] | tuple[csr_array]:
    r"""Return an array of generalized Pauli matrices or matrix products of given dimension(s).

    Args:
        dim: Dimension(s) of the Pauli matrices.
        sparse: Whether to return the matrices as an array (dtype=object) of CSR arrays.

    Returns:
        An array of Pauli (product) matrices as an array. For `dim=(d1, d2, ...)`, the shape of
        the array is `(d1**2, d2**2, ..., d1*d2*..., d1*d2*...)`.
    """
    dim = _normalize_dim(dim)

    if len(dim) == 1:
        return pauli_matrices(dim[0], sparse=sparse)

    if (cache := _pauli_products.get((dim, sparse))) is not None:
        return cache.copy()

    subsystems = [pauli_matrices(d, sparse=sparse) for d in dim]
    num_sub = len(subsystems)
    if sparse:
        raise NotImplementedError("Need an hour")
    else:
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
        dim_array = np.asarray(dim)
        shape = np.concatenate(
            (np.square(dim_array), np.prod(np.repeat(dim_array[None, :], 2, axis=0), axis=1))
        )
        matrix_array = np.einsum(indices, *subsystems).reshape(*shape) / (2 ** (num_sub - 1))

    matrix_array.setflags(write=False)
    _pauli_products[(dim, sparse)] = matrix_array.copy()
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
        return cache.copy()

    if sparse:
        matrices = []
        shape = (dim, dim)

        data = np.full(dim, np.sqrt(2.0 / dim), dtype=complex)
        indices = np.arange(dim)
        indptr = np.arange(dim + 1)
        matrices.append(csr_array((data, indices, indptr), shape=shape))

        for ishell in range(1, dim):
            for ipos in range(ishell):
                indices = [ishell, ipos]
                indptr = [0] * (ipos + 1) + [1] * (ishell - ipos)
                indptr += [2] * (dim - ishell)

                matrices.append(csr_array(([1.0 + 0.0j, 1.0 + 0.0j], indices, indptr), shape=shape))
                matrices.append(csr_array(([-1.0j, 1.0j], indices, indptr), shape=shape))

            data = np.array([1.0] * ishell + [-ishell], dtype=complex)
            data *= np.sqrt(2.0 / ishell / (ishell + 1.0))
            indices = np.arange(ishell + 1)
            indptr = list(range(ishell + 1)) + [ishell + 1] * (dim - ishell)
            matrices.append(csr_array((data, indices, indptr), shape=shape))

        matrices = np.array(matrices)

    else:
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

            matrices[imat, : ishell + 1, : ishell + 1] = np.diag(
                np.array([1.0] * ishell + [-ishell])
            )
            imat += 1

        # Normalization
        norm = np.trace(np.matmul(matrices, matrices), axis1=1, axis2=2)
        matrices *= np.sqrt(2.0 / norm)[:, None, None]

    # Make the matrix immutable
    matrices.setflags(write=False)
    _pauli_matrices[(dim, sparse)] = matrices
    return matrices


_pauli_matrices = {}


def components(
    matrix: ArrayLike, dim: MatrixDimension | None = None, npmod: ModuleType = np
) -> NDArray[np.complex128]:
    r"""Return the Pauli decomposition coefficients :math:`\nu_{k_1 \dots k_n}` of the matrix.

    Args:
        matrix: Matrix to decompose. The last two dimensions of the array are dotted with the Pauli
            matrices.
        dim: Subsystem dimensions. The product of subsystem dimensions must match the matrix
            dimension. If None, the matrix is assumed to represent a single system.

    Returns:
        A complex array of shape `(..., d1**2, d2**2, ...)` where `d1`, `d2`, ... are the subsystem
        dimensions.

    Raises:
        ValueError: If `prod(dim)` does not match the matrix dimension.
    """
    # _normalize_dim runs for every npmod -- `len(dim)` below needs a sequence either way, and
    # gating it left `components(m, dim=3, npmod=jnp)` raising "object of type 'int' has no len()"
    # from the return statement, naming nothing. Only the *validation* below belongs behind the gate,
    # per CLAUDE.md's npmod rule.
    if dim is None:
        dim = (matrix.shape[-1],)
    else:
        dim = _normalize_dim(dim)

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
        symbol: Base symbol.
        delimiter: Delimiter between the symbols for multibody labels.
        norm: Include the normalization factors.
        fmt: Output format. Allowed values are 'text', 'latex', 'latex-text',
            'latex-slash'.

    Returns:
        An ndarray of type string and shape `(d1**2, d2**2, ...)`.
    """
    dim = _normalize_dim(dim)

    if symbol is None or isinstance(symbol, str):
        symbol = (symbol,) * len(dim)

    out = np.array("", dtype=str)

    for pauli_dim, sym in zip(dim, symbol):
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

        out = np.char.add(np.repeat(out[..., None], pauli_dim**2, axis=-1), labels)

    if norm and len(dim) >= 2:
        if len(dim) == 2:
            denom = "2"
        elif fmt == "text":
            denom = f"2**{len(dim) - 1}"
        else:
            denom = "2^{%d}" % (len(dim) - 1)  # noqa: UP031 (f-string needs {{}} escapes here)

        if fmt in ("text", "latex-slash"):
            post = np.full(out.shape, f"/{denom}")

        else:
            if fmt == "latex":
                pre = np.full(out.shape, r"\frac{")
                post = np.full(out.shape, "}{%s}" % denom)  # noqa: UP031
            else:
                pre = np.full(out.shape, r"\textstyle{\frac{")
                post = np.full(out.shape, "}{%s}}" % denom)  # noqa: UP031

            out = np.char.add(pre, out)

        out = np.char.add(out, post)

    return out
