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

Both sides of that alignment live here: :meth:`PauliSumXZ.from_paulisum` pads the signatures, and
:meth:`PauliSumXZ.pack_states` pads the states consumers pair them with. Callers should reach for the
latter rather than open-coding the ``pad``-then-``packbits`` idiom, so there is one definition of the
layout rather than one per consumer.

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
   :members:
"""

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
    """Symplectic (XZ) representation of a sum of Pauli strings.

    Construct with :meth:`from_paulisum`; the fields are not meant to be assembled by hand, since the
    pad-bit alignment described under "Bit layout" above is what makes them consumable.

    Attributes:
        x: Packed unique X signatures, shape ``(num_xgroups, ceil((num_qubits + 1) / 8))``.
        z: Packed Z signatures per X group, zero-padded to a rectangle, shape
            ``(num_xgroups, max_zterms, ceil((num_qubits + 1) / 8))``.
        c: Phase-folded coefficients, shape ``(num_xgroups, max_zterms)``. **float64 exactly when
            every Pauli string has an even number of Ys, complex128 otherwise** -- an odd-Y string
            cannot be real in the :math:`Q = (-i)^{x \\cdot z} Z^z X^x` convention, because the folded
            phase turns real input complex. Check ``.c.dtype`` if you need float64; there is
            deliberately no flag to request it, since no flag could grant it.
        num_qubits: Number of qubits, static under JAX transforms.
        nzterms: Number of genuine (non-padding) Z terms in each X group, one entry per group, static
            under JAX transforms. ``z`` and ``c`` are rectangles padded to ``max(nzterms)``, so this
            is the only record of where the real terms end.

            **It cannot be recovered from the arrays afterwards**, which is why it is stored. A pad
            slot's Z signature packs to all-zero bytes -- and so does a genuine all-identity Z
            signature, so ``z`` cannot distinguish them. ``c``'s trailing zeros are only pad because
            :meth:`from_paulisum` drops zero-coefficient terms, which makes counting them a
            coincidence rather than a fact.

            Consumers needing a static trip count over the real terms use this instead of a
            data-dependent loop; see :func:`rqutils.sqd.apply_h`'s ``nterms``. Being static, it joins
            the JIT cache key: two Hamiltonians with identical shapes but different term counts
            compile separately, the same trade ``num_qubits`` already makes.
    """

    x: np.ndarray[tuple[int, int], np.dtype[np.uint8]]
    z: np.ndarray[tuple[int, int, int], np.dtype[np.uint8]]
    c: np.ndarray[tuple[int, int], np.dtype[np.inexact]]
    num_qubits: int = field(metadata={"static": True})
    nzterms: tuple[int, ...] = field(metadata={"static": True})

    @staticmethod
    def pack_states(
        states: np.ndarray[tuple[int, int], np.dtype[np.uint8]],
    ) -> np.ndarray[tuple[int, int], np.dtype[np.uint8]]:
        """Pack binary states into bytes with the leading pad bit this class's signatures carry.

        The states side of the bit-alignment contract described under "Bit layout" above. It lives on
        this class because this class is what *decides* the layout: :meth:`from_paulisum` inserts the
        signature pad bit unconditionally, and this method inserts the matching state pad bit, so the
        two halves cannot drift. Consumers call it rather than restating the idiom -- a disagreement
        is silent, putting every matrix element in the wrong column while the result stays symmetric,
        which is how ``rqutils.sqd.hproj`` once shipped a plausible wrong eigenvalue.

        The pad bit must be inserted with :func:`numpy.pad` before packing, not shifted in afterwards,
        because :func:`numpy.packbits` fills each byte from the most significant end.

        It also makes byte 0 of any genuine state ``< 128``, which is what lets ``255`` serve as an
        unambiguous fill-in marker for padded slots (see :func:`rqutils.sqd.uniquify_states`).

        Args:
            states: Binary array of computational basis states, shape ``(num_states, num_qubits)``.

        Returns:
            Packed states, shape ``(num_states, ceil((num_qubits + 1) / 8))``.
        """
        return np.packbits(np.pad(states.astype(np.uint8), {1: (1, 0)}), axis=1)

    @staticmethod
    def unpack_states(
        states_p: np.ndarray[tuple[int, int], np.dtype[np.uint8]], num_qubits: int
    ) -> np.ndarray[tuple[int, int], np.dtype[np.uint8]]:
        """Unpack states packed by :meth:`pack_states`, dropping the leading pad bit.

        The inverse of :meth:`pack_states`, defined alongside it so the ``+1`` offset the pad bit
        imposes cannot drift between the two directions.

        Args:
            states_p: Packed states, shape ``(num_states, num_bytes)``.
            num_qubits: Number of qubits to recover, i.e. the original trailing dimension.

        Returns:
            Binary array of states, shape ``(num_states, num_qubits)``.
        """
        return np.unpackbits(states_p, axis=-1)[:, 1 : 1 + num_qubits]

    @classmethod
    def from_paulisum(cls, paulisum: Any, atol: float = 1e-12) -> "PauliSumXZ":
        """Build the packed representation from a Pauli sum.

        The only constructor, and the signature half of the bit-alignment contract: it inserts the
        pad bit at position 0 unconditionally, matching what :meth:`pack_states` inserts on the state
        side. Terms are grouped by unique X signature, the Z groups zero-padded to a rectangle, the
        :math:`(-i)^{\\mathrm{popcount}(x \\wedge z)}` phase folded into the coefficients, and the
        result bit-packed.

        Duplicate Pauli strings are summed and zero-coefficient terms dropped, so the term count of
        the result can be below that of the input.

        Args:
            paulisum: Either a ``(paulis, coeffs)`` tuple of a Pauli-string sequence and a matching
                coefficient sequence, or a qiskit ``SparsePauliOp``. Qiskit's ``.x``/``.z`` are
                reversed on ingest, since this class is little-endian in qubit order.
            atol: Absolute tolerance on the imaginary part of each coefficient. A coefficient with
                ``abs(imag) <= atol`` is treated as real; anything above raises. The default of
                ``1e-12`` accepts operators that are Hermitian to float64 rounding -- numerically
                decomposed operators carry residue around ``1e-16`` -- while still rejecting genuinely
                complex signal by many orders. Tighten it to ``0.0`` to demand exact realness.

        Returns:
            The packed representation.

        Raises:
            ValueError: If the Pauli and coefficient sequences differ in length; if the Pauli strings
                are not all the same length; if ``paulisum`` is neither a tuple nor a
                ``SparsePauliOp``; or if any coefficient's imaginary part exceeds ``atol`` in
                absolute value, since a complex coefficient makes the operator non-Hermitian and
                every consumer of this class assumes Hermiticity. There is no flag to bypass the last
                check -- only ``atol``, which widens what counts as real without ever discarding a
                nonzero imaginary part silently.
        """
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
            # Sum the duplicate strings' coefficients with a scatter-add, not a one-hot matmul. The
            # matmul materialized a dense (n_unique, n_terms) mask to express a group-by: 64 MB and
            # 27.5 ms at 4000 terms / 2000 groups, growing quadratically (400 MB at 10000/5000, and
            # OOM past that), against 0.03 ms here. np.add.at rather than np.bincount because coeffs
            # is still complex at this point -- the Hermiticity check below is what narrows it -- and
            # bincount takes real weights only.
            summed = np.zeros(paulis.shape[0], dtype=coeffs.dtype)
            np.add.at(summed, indices, coeffs)
            coeffs = summed
            xbits = np.logical_or(paulis == "X", paulis == "Y")
            zbits = np.logical_or(paulis == "Y", paulis == "Z")
            num_qubits = paulis.shape[1]

        elif HAS_QISKIT and isinstance(paulisum, SparsePauliOp):
            # Remove null terms
            paulisum = paulisum.simplify()
            coeffs = paulisum.coeffs
            xbits = paulisum.paulis.x[:, ::-1]
            zbits = paulisum.paulis.z[:, ::-1]
            num_qubits = paulisum.num_qubits

        else:
            raise ValueError("Unsupported input type")

        # A complex coefficient on a Pauli string means the operator is not Hermitian, which every
        # consumer of this class assumes. Checked here rather than per-branch so both ingest paths
        # agree: the Qiskit branch always raised, while the tuple branch used to warn and silently
        # take .real under force_real=True -- discarding the imaginary part of a non-Hermitian
        # operator rather than rejecting it.
        #
        # Compared against atol rather than exactly, because a mathematically Hermitian operator whose
        # Pauli coefficients were obtained *numerically* carries rounding at the 1e-16 level. The
        # standard construction -- conjugate a Hermitian matrix by a non-Clifford circuit, then
        # decompose -- does exactly that: measured over 18 such operators (n = 3..5, 6 seeds), 18/18
        # were rejected by the exact check, with a largest coefficient |imag| of 3.3e-16 while the
        # operators' own hermiticity error reached 2.7e-15. The residue being an order of magnitude
        # *below* the property under test is what makes the exact form wrong.
        #
        # This is NOT a return to force_real, which took .real of genuinely complex signal. A
        # tolerance still rejects that loudly; 1e-12 sits ~4 orders above the observed rounding and
        # many orders below any physical coefficient, so the two cases stay cleanly separated.
        if np.any(np.abs(coeffs.imag) > atol):
            raise ValueError(
                "Coefficients of Paulis must be real for the Hamiltonian to be Hermitian."
            )
        coeffs = coeffs.real

        # Find unique X signatures together with correspondence pointers
        xuniq, indices, counts = np.unique(xbits, axis=0, return_inverse=True, return_counts=True)
        xsignatures = xuniq.astype(np.uint8)
        # Group the Z signatures and coeffs by X signatures
        shape = (xsignatures.shape[0], np.max(counts))
        zsignatures = np.zeros(shape + zbits.shape[-1:], dtype=np.uint8)
        phcoeffs = np.zeros(shape, dtype=np.complex128)
        # Bucket the terms by X signature with one stable sort instead of rescanning `indices` once
        # per signature. The rescan was the same quadratic shape as the mask matmul above -- 15.9 ms
        # at 5000 groups, against 0.3 ms here -- and it also re-ran the uint8 conversion per group.
        # `indices` is already the group id of each term, so sorting it groups the terms, and the
        # cumulative counts give each group's slice.
        order = np.argsort(indices, kind="stable")
        bounds = np.concatenate(([0], np.cumsum(counts)))
        zbits_u8 = zbits.astype(np.uint8)
        phase_table = np.array([1.0, -1.0j, -1.0, 1.0j])
        for isig, xsig in enumerate(xsignatures):
            ipaulis = order[bounds[isig] : bounds[isig + 1]]
            zsigs = zbits_u8[ipaulis]
            zsignatures[isig, : counts[isig]] = zsigs
            # Multiply the coeffs by (-i)^{n_zx}
            iphases = np.sum(xsig & zsigs, axis=1) & 3
            phcoeffs[isig, : counts[isig]] = coeffs[ipaulis] * phase_table[iphases]

        # Narrow to float64 when the folded phase left everything real, i.e. when every Pauli string
        # has an even number of Ys. An odd-Y string cannot be real in this convention -- the
        # (-i)^{x.z} phase turns real input complex -- so `.c` stays complex128 there by
        # construction, not by mistake. Callers restricted to float64 (a backend with no complex128,
        # say) check `.c.dtype`; there is deliberately no flag to request realness, since no flag can
        # grant it.
        if np.all(phcoeffs.imag == 0.0):
            phcoeffs = phcoeffs.real

        # Insert the dummy identity Pauli at bit position 0 and pack. This is unconditional rather
        # than opt-in: as a flag it was an invariant nothing could enforce, since the signature side
        # and the state side lived in separate call sites with no check that they agreed. When they
        # disagreed every matrix element landed in the wrong column, and the result stayed symmetric,
        # so eigvalsh returned a plausible wrong ground energy -- that is how hproj shipped broken.
        #
        # The X side goes through pack_states, the same method consumers pad their states with, so the
        # two halves of the alignment contract are one code path and not two that must agree. The Z
        # signatures are (n_xgroups, n_zterms, n_qubits), so they pad axis 2 rather than axis 1 and
        # cannot reuse it; the operation is otherwise identical.
        #
        # svsim, the only other consumer of this representation, is unaffected: it builds CircuitXZ
        # itself and never calls this method.
        xsignatures = cls.pack_states(xsignatures)
        zsignatures = np.packbits(np.pad(zsignatures, {2: (1, 0)}), axis=-1)
        # counts is the per-group real-term count; keep it rather than letting it fall out of scope.
        # Nothing downstream can rederive it (see the nzterms docstring), and a static trip count over
        # the real terms is what makes the diagonal accumulation differentiable.
        return cls(xsignatures, zsignatures, phcoeffs, num_qubits, tuple(counts.tolist()))

    @property
    def arrays(self) -> tuple[jax.Array, jax.Array, jax.Array]:
        """The packed ``(x, z, c)`` arrays, for splatting into a traced function.

        Returns:
            ``(x, z, c)``: the packed X signatures, the packed Z signatures, and the phase-folded
            coefficients. See the class docstring for shapes and for ``c``'s dtype rule.
        """
        return self.x, self.z, self.c
