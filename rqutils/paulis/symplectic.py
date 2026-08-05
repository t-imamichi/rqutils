r"""
========================================================================================
GPU-efficient symplectic representation of Pauli sums (:mod:`rqutils.paulis.symplectic`)
========================================================================================

.. currentmodule:: rqutils.paulis.symplectic

Symplectic representation of a Pauli string
===========================================

Any Pauli string :math:`Q` can be expressed as

.. math::

    Q = (-i)^{xz} \left(Z^{z_{n-1}} \otimes \cdots Z^{z_{0}}\right)
                                  \left(X^{x_{n-1}} \otimes \cdots X^{x_{0}}\right)

where :math:`x` (X signature) and :math:`z` (Z signature) are binary vectors of length :math:`n`
(number of qubits) and :math:`xz` represents their inner product.

Bit layout
==========

Signatures are bit-packed into :math:`\lceil (n+1)/8 \rceil` ``uint8`` s. The extra bit sits at
position 0 and is always zero: it is a dummy identity factor that aligns the signatures with the pad
bit consumers insert into their state bitstrings, where it marks spurious (fill-in) entries. This
padding is intrinsic to the representation rather than optional, so the two sides cannot disagree on
bit alignment -- a disagreement that is silent, since every matrix element lands in the wrong column
while the result stays symmetric.

Note that ``np.packbits`` fills each byte from the most significant end, so a signature's payload
occupies the *leading* entries of ``np.unpackbits`` rather than the trailing ones: bit 0 is the pad
bit, and the :math:`n` payload bits are entries 1 through :math:`n` in Pauli-string character order,
which is the reverse of the little-endian qubit numbering used on ingest. Code that decodes a packed
signature back to an integer must therefore shift by :math:`8 \lceil (n+1)/8 \rceil - (n+1)`,
counting the pad bit; dropping that :math:`+1` yields a *permutation* of the right answer, which is
finite and symmetric and therefore silent.

Symplectic Pauli sum representation API
=======================================

.. autoclass:: PauliSumXZ
"""

import warnings
from dataclasses import dataclass, field
from typing import Any

import jax
import numpy as np
from jax.tree_util import register_dataclass

try:
    from qiskit.quantum_info import SparsePauliOp

    HAS_QISKIT = True
except ImportError:
    HAS_QISKIT = False


@register_dataclass
@dataclass(frozen=True)
class PauliSumXZ:
    """Symplectic (XZ) representation of a sum of Pauli strings."""

    x: np.ndarray[tuple[int, int], np.dtype[np.uint8]]
    z: np.ndarray[tuple[int, int, int], np.dtype[np.uint8]]
    c: np.ndarray[tuple[int, int], np.dtype[np.inexact]]
    num_qubits: int = field(metadata={"static": True})

    @classmethod
    def from_paulisum(cls, paulisum: Any, force_real: bool = False) -> "PauliSumXZ":
        if isinstance(paulisum, tuple):  # ([paulis], [coeffs])
            paulis, coeffs = paulisum
            if len(paulis) != len(coeffs):
                raise ValueError("Lengths of Pauli and coeff lists do not match")
            if len({len(p) for p in paulis}) != 1:
                raise ValueError("Pauli strings have non-uniform lengths")

            coeffs = np.array(coeffs)
            # Sort and consolidate the pauli strings and coefficients
            paulis = np.array([list(p.upper()) for p in paulis])
            nonzero = np.nonzero(coeffs)
            coeffs = coeffs[nonzero]
            paulis = paulis[nonzero]
            paulis, indices = np.unique(paulis, axis=0, return_inverse=True)
            masks = (indices[None, :] == np.arange(paulis.shape[0])[:, None]).astype(int)
            coeffs = masks @ coeffs
            xbits = np.logical_or(paulis == "X", paulis == "Y")
            zbits = np.logical_or(paulis == "Y", paulis == "Z")
            num_qubits = paulis.shape[1]

        elif HAS_QISKIT and isinstance(paulisum, SparsePauliOp):
            if not np.allclose(paulisum.coeffs.imag, 0.0):
                raise ValueError(
                    "Coefficients of Paulis must be real for the Hamiltonian to be Hermitian."
                )

            # Remove null terms
            paulisum = paulisum.simplify()
            coeffs = paulisum.coeffs
            xbits = paulisum.paulis.x[:, ::-1]
            zbits = paulisum.paulis.z[:, ::-1]
            num_qubits = paulisum.num_qubits

        else:
            raise ValueError("Unsupported input type")

        if force_real:
            if np.any(coeffs.imag != 0.0):
                warnings.warn("Found nonzero imaginary part when force_real=True")
            coeffs = coeffs.real

        # Find unique X signatures together with correspondence pointers
        xuniq, indices, counts = np.unique(xbits, axis=0, return_inverse=True, return_counts=True)
        xsignatures = xuniq.astype(np.uint8)
        # Group the Z signatures and coeffs by X signatures
        shape = (xsignatures.shape[0], np.max(counts))
        zsignatures = np.zeros(shape + zbits.shape[-1:], dtype=np.uint8)
        phcoeffs = np.zeros(shape, dtype=np.complex128)
        for isig, xsig in enumerate(xsignatures):
            ipaulis = np.nonzero(indices == isig)[0]
            zsigs = zbits[ipaulis].astype(np.uint8)
            zsignatures[isig, : counts[isig]] = zsigs
            # Multiply the coeffs by (-i)^{n_zx}
            iphases = np.sum(xsig & zsigs, axis=1) & 3
            phases = np.array([1.0, -1.0j, -1.0, 1.0j])[iphases]
            phcoeffs[isig, : counts[isig]] = coeffs[ipaulis] * phases

        if np.all(phcoeffs.imag == 0.0):
            phcoeffs = phcoeffs.real
        elif force_real:
            # The check above on `coeffs` sees only the *input* coefficients, but the
            # (-i)^{x.z} phase is folded in afterwards, so a Pauli string with an odd number of Ys
            # turns real input complex again -- and force_real=True returned complex128 with no
            # warning at all. Callers were left to notice on their own:
            # examples/_bench_common.build_solver_inputs raises on `.c.dtype != np.float64` for
            # exactly this reason. Warn rather than raise, matching the pre-phase check's
            # best-effort semantics so no existing caller changes behaviour.
            warnings.warn(
                "force_real=True but the coefficients are complex after the (-i)^{x.z} phase is "
                "applied; a Pauli string with an odd number of Ys cannot have a real coefficient "
                "in this convention. The returned .c is complex128 -- check its dtype if your "
                "downstream code requires float64.",
                stacklevel=2,
            )

        # A dummy identity Pauli at bit position 0, aligning with the pad bit that consumers insert
        # into their state bitstrings.
        #
        # This is unconditional rather than opt-in. As a flag it was an invariant nothing could
        # enforce: sqd packs its states with a leading pad bit and needs the signatures shifted to
        # match, but the two decisions lived in separate call sites with no check that they agreed.
        # When they disagreed, every matrix element landed in the wrong column and the result was
        # still symmetric, so eigvalsh returned a plausible wrong ground energy -- that is exactly
        # how hproj shipped broken. Making the padding intrinsic removes the possibility.
        #
        # svsim, the only other consumer of this representation, is unaffected: it builds CircuitXZ
        # itself and never calls this method.
        xsignatures = np.pad(xsignatures, {1: (1, 0)})
        zsignatures = np.pad(zsignatures, {2: (1, 0)})

        # Pack the bit signatures
        xsignatures = np.packbits(xsignatures, axis=-1)
        zsignatures = np.packbits(zsignatures, axis=-1)
        return cls(xsignatures, zsignatures, phcoeffs, num_qubits)

    @property
    def arrays(self) -> tuple[jax.Array, jax.Array, jax.Array]:
        return self.x, self.z, self.c
