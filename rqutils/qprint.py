r"""
==========================================================
Pretty-printer for quantum objects (:mod:`rqutils.qprint`)
==========================================================

.. currentmodule:: rqutils.qprint

QPrint API
==========

.. autofunction:: qprint
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import scipy
from numpy.typing import ArrayLike

try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
except ImportError:
    HAS_MPL = False
else:
    HAS_MPL = True
try:
    from qutip import Qobj
except ImportError:
    HAS_QUTIP = False
else:
    HAS_QUTIP = True
import rqutils.paulis.general as pmatrix
from rqutils._types import MatrixDimension

if HAS_MPL:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    MATPLOTLIB_INLINE_BACKENDS = {
        "module://ipykernel.pylab.backend_inline",
        "module://matplotlib_inline.backend_inline",
        "nbAgg",
    }
else:
    type Axes = Any
    type Figure = Any

# The Figure arm must be named here, not added by a later `PrintReturnType |= Figure` under HAS_MPL: a
# `type` statement is evaluated statically, so the augmented assignment leaves the arm invisible to a
# checker. Safe either way because the statement is lazy and `Figure` is bound on both branches above
# -- the real class with matplotlib, `Any` without, which correctly collapses the union to `Any`.
type PrintReturnType = str | Figure


def qprint(
    qobj: Any,
    fmt: str = "braket",
    amp_norm: complex | tuple[complex, str] | None = None,
    phase_norm: tuple[complex, str] | None = (np.pi, "π"),
    global_phase: complex | str | None = None,
    terms_per_row: int = 0,
    amp_format: str = ".3f",
    phase_format: str = ".2f",
    amp_cutoff: float = 5.0e-4,
    lhs_label: str | None = None,
    dim: ArrayLike | None = None,
    binary: bool = False,
    symbol: str | Sequence[str] | Sequence[Sequence[str]] | None = None,
    delimiter: str = "",
    output: str = "latex",
) -> PrintReturnType:
    """Pretty-print a quantum object.

    Available output formats are

    - `'braket'`: For a column vector, row vector, or matrix input. Prints out the mathematical
      expression of the linear combination of bras, kets, or ket-bras.
    - `'pauli'`: For an input representing a square matrix (shape `(d1*d2*..., d1*d2*...)`) or a
      components array (shape `(d1**2, d2**2, ...)`). Argument `dim` is required for the matrix
      interpretation.
    - `'matrix'`: For a square matrix input. Arguments `dim`, `binary`, `symbol`, and `delimiter`
      are ignored.

    Three printing formats are supported:

    - `'text'`: Print text to stdout.
    - `'latex'`: Return a LaTeX string.
    - `'mpl'`: Return a matplotlib figure.

    Args:
        qobj: Input quantum object.
        fmt: Content format (`'braket'`, `'pauli'`, or `'matrix'`).
        amp_norm: Specification of the normalization of amplitudes by (numeric devisor, unit in
            LaTeX).
        phase_norm: Specification of the normalization of phases by (numeric devisor, unit in
            LaTeX).
        global_phase: Specification of the phase to factor out. Give a numeric offset or 'mean'.
        terms_per_row: Number of terms to show per row.
        amp_format: Format for the numerical value of the amplitude absolute values.
        phase_format: Format for the numerical value of the phases.
        amp_cutoff: Ignore terms with absolute amplitudes less than ``max(abs(amplitudes))`` times
            this value.
        lhs_label: If not None, prepend 'label = ' to the printout.
        dim: Specification of the dimensions of the subsystems. For `fmt='pauli'`, used only when
            `qobj` is a square matrix or a 1D array.
        binary: Show bra and ket indices in binary. Only for `fmt='braket'`.
        symbol: Pauli matrix symbols. Only for `fmt='pauli'`.
        delimiter: Pauli product delimiter. Only for `fmt='pauli'`.
        output: Output method (`'text'`, `'latex'`, or `'mpl'`).

    Returns:
        Object to be printed: for `output='text'` the `QPrintBase` subclass instance itself, whose
        `__repr__` renders lazily; for `'latex'` a string; for `'mpl'` a matplotlib Figure.

    Raises:
        NotImplementedError: If `fmt` is not one of `'braket'`, `'pauli'`, `'matrix'`; if `output` is
            not one of `'text'`, `'latex'`, `'mpl'`; or if `qobj` is of a type none of the content
            classes accept.
        ValueError: Propagated from the content class for an input whose shape or `dim` is
            inconsistent -- see :class:`QPrintBraKet`, :class:`QPrintPauli`, :class:`QPrintMatrix`.
        RuntimeError: If `output='mpl'` and matplotlib is not installed.
    """
    if fmt == "braket":
        pobj = QPrintBraKet(
            qobj=qobj,
            amp_norm=amp_norm,
            phase_norm=phase_norm,
            global_phase=global_phase,
            terms_per_row=terms_per_row,
            amp_format=amp_format,
            phase_format=phase_format,
            amp_cutoff=amp_cutoff,
            lhs_label=lhs_label,
            dim=dim,
            binary=binary,
        )
        env = "split"

    elif fmt == "pauli":
        pobj = QPrintPauli(
            qobj=qobj,
            amp_norm=amp_norm,
            phase_norm=phase_norm,
            global_phase=global_phase,
            terms_per_row=terms_per_row,
            amp_format=amp_format,
            phase_format=phase_format,
            amp_cutoff=amp_cutoff,
            lhs_label=lhs_label,
            dim=dim,
            symbol=symbol,
            delimiter=delimiter,
        )
        env = "split"

    elif fmt == "matrix":
        pobj = QPrintMatrix(
            qobj=qobj,
            amp_norm=amp_norm,
            phase_norm=phase_norm,
            global_phase=global_phase,
            amp_format=amp_format,
            phase_format=phase_format,
            amp_cutoff=amp_cutoff,
            lhs_label=lhs_label,
        )
        env = None

    else:
        raise NotImplementedError(f"qprint with format {fmt} not implemented")

    if output == "text":
        return pobj
    if output == "latex":
        return pobj.latex(env)
    if output == "mpl":
        return pobj.mpl()

    raise NotImplementedError(f"qprint with output {output} not implemented")


class QPrintBase(ABC):
    """Helper class to compose an expression of a given quantum object.

    This is a base class for QPrint which performs numerical processing of the components in the
    input quantum object. Basis labeling is handled by the concrete subclasses.

    Args:
        qobj: Input quantum object.
        amp_norm: Specification of the normalization of amplitudes by (numeric devisor, unit in
            LaTeX).
        phase_norm: Specification of the normalization of phases by (numeric devisor, unit in
            LaTeX).
        global_phase: Specification of the phase to factor out. Give a numeric offset or ``'mean'``.
        terms_per_row: Number of terms to show per row.
        amp_format: Format for the numerical value of the amplitude absolute values.
        phase_format: Format for the numerical value of the phases.
        amp_cutoff: Ignore terms with absolute amplitudes less than ``max(abs(amplitudes))`` times
            this value.
        lhs_label: If not None, prepends ``'lhs_label = '`` to the printout.
        dim: Specification of the dimensions of the subsystems.
    """

    @dataclass
    class Term:
        index: tuple
        sign: int
        amp: str
        phase: str
        label: str = ""

    def __init__(
        self,
        qobj: Any,
        amp_norm: complex | tuple[complex, str] | None = None,
        phase_norm: tuple[complex, str] | None = (np.pi, "π"),
        global_phase: complex | str | None = None,
        terms_per_row: int = 0,
        amp_format: str = ".3f",
        phase_format: str = ".2f",
        amp_cutoff: float = 1.0e-6,
        lhs_label: str | None = None,
        dim: MatrixDimension | None = None,
    ):
        self.amp_norm = amp_norm
        self.phase_norm = phase_norm
        self.global_phase = global_phase
        self.terms_per_row = terms_per_row
        self.amp_format = amp_format
        self.phase_format = phase_format
        self.amp_cutoff = amp_cutoff
        self.lhs_label = lhs_label

        # One definition of what a MatrixDimension normalizes to, shared with paulis/general.py --
        # the module this one already calls components()/labels() from, and which keys its memoization
        # dicts on exactly that tuple(int) form.
        self._dim = None if dim is None else pmatrix.normalize_dim(dim)

        self._qobj, self._data = self._qobj_data(qobj)

    def __repr__(self):
        expr = self._format_lhs("text")

        if expr is None:
            expr = ""
        else:
            expr += " = "

        pre_expr, lines = self._make_lines("text")
        if pre_expr:
            expr += f"{pre_expr} ("

        indentation = " " * len(expr)
        expr += f"{lines[0]}\n"
        expr += "\n".join((indentation + l) for l in lines[1:])

        if pre_expr:
            expr += ")"

        return expr

    def latex(self, env="split") -> str:
        """Return a LaTeX expression."""
        pre_expr, lines = self._make_lines("latex")

        if pre_expr:
            lines[0] = rf" \left( {lines[0]}"
            lines[-1] += r" \right)"

            if len(lines) > 1:
                lines[0] += r" \right."
                lines[-1] = r"\left. " + lines[-1]

        if env == "split" and len(lines) > 1:
            lines = [f"& {line}" for line in lines]

        if pre_expr:
            lines[0] = f"{pre_expr} {lines[0]}"

        lhs = self._format_lhs("latex")

        if lhs is not None:
            lines[0] = f"{lhs} = {lines[0]}"

        expr = r" \\ ".join(lines)

        if env:
            return rf"\begin{{{env}}} {expr} \end{{{env}}}"

        return expr

    def mpl(self, ax: Axes | None = None) -> Figure | None:
        """Display or return the expression as a matplotlib figure."""
        if not HAS_MPL:
            raise RuntimeError("Matplotlib is not available")

        pre_expr, lines = self._make_lines("latex")

        if pre_expr:
            lines[0] = f"{pre_expr} ({lines[0]}"
            lines[-1] += ")"

        lhs = self._format_lhs("latex")

        if lhs is not None:
            lines[0] = f"{lhs} = {lines[0]}"

        if ax is None:
            fig, ax = plt.subplots(1, figsize=[10.0, 0.5 * len(lines)])
        else:
            fig = None

        ax.axis("off")

        num_rows = len(lines)
        for irow, line in enumerate(lines):
            ax.text(
                0.5,
                1.0 / num_rows * (num_rows - irow - 1),
                f"${line}$",
                fontsize="x-large",
                ha="right",
            )

        if fig is not None and mpl.get_backend() in MATPLOTLIB_INLINE_BACKENDS:
            plt.close(fig)

        return fig

    def _process(self) -> tuple[int, str, str, list[list[Term]]]:
        """Compose a list of QPrintTerms."""
        # Amplitude format template
        amp_template = f"{{:{self.amp_format}}}"

        # Phase format template
        phase_template = f"{{:{self.phase_format}}}"

        ## Preprocess self._data

        # Absolute value and phase of the amplitudes
        absamp = np.abs(self._data)
        phase = np.angle(self._data)

        # Normalize the abs amplitudes and identify integral values
        if self.amp_norm is not None:
            if isinstance(self.amp_norm, tuple):
                absamp /= self.amp_norm[0]
                global_amp = self.amp_norm[1]
            else:
                absamp /= self.amp_norm
                if np.isclose(np.round(self.amp_norm), self.amp_norm):
                    global_amp = f"{np.round(self.amp_norm)}"
                else:
                    global_amp = amp_template.format(self.amp_norm)

        else:
            global_amp = ""

        # Select the surviving terms *before* the phase pipeline below, not after. Only terms above
        # the cutoff are ever printed, so normalizing the phase of every element first and then
        # discarding all but a handful is pure waste -- and it is waste paid again on every repr,
        # since output='text' returns this object for a lazy __repr__. Measured on a 5-term printout
        # of a dim-2^20 input: 25 ms -> 5.3 ms. It also bounds the wrap-around loop below by the term
        # count rather than the input size.
        #
        # Show only terms with absamp < max(absamp) * amp_cutoff
        amp_atol = np.amax(absamp) * self.amp_cutoff
        amp_is_zero = np.isclose(np.zeros_like(absamp), absamp, atol=amp_atol)
        # convert into list of tuples
        kept = np.logical_not(amp_is_zero).nonzero()
        term_indices = list(zip(*kept))

        # Shift the phases. The offset under global_phase='mean' is a full-array reduction by
        # definition -- the mean is over every element, not just the printed ones -- so it has to be
        # taken before the compression below.
        phase_offset = 0.0
        if self.global_phase is not None:
            if self.global_phase == "mean":
                phase_offset = np.mean(phase)
            else:
                phase_offset = self.global_phase

            phase -= phase_offset

        # Compress to the surviving terms before normalizing anything further. Everything from here
        # on is read only at the kept positions, so running the wrap-around loop and the two
        # normalize_phase passes over all 2^n elements and then discarding all but a handful is pure
        # waste -- and waste paid again on every repr, since output='text' returns this object for a
        # lazy __repr__. `term_indices` still holds the original (possibly 2-d) indices, which is what
        # _add_labels needs; these flat arrays are indexed by term position instead.
        absamp = absamp[kept]
        phase = phase[kept]

        rounded_amp = np.round(absamp).astype(int)
        amp_is_int = np.isclose(rounded_amp, absamp)
        rounded_amp = np.where(amp_is_int, rounded_amp, -1)

        twopi = 2.0 * np.pi

        while np.any((phase < 0.0) | (phase >= twopi)):
            phase = np.where(phase >= 0.0, phase, phase + twopi)
            phase = np.where(phase < twopi, phase, phase - twopi)

        def normalize_phase(phase):
            reduced_phase = phase / (np.pi / 2.0)
            axis_proj = np.round(reduced_phase).astype(int)
            on_axis = np.isclose(axis_proj, reduced_phase)
            axis_proj = np.where(on_axis, axis_proj, -1)

            if self.phase_norm is not None:
                phase /= self.phase_norm[0]

            rounded_phase = np.round(phase).astype(int)
            phase_is_int = np.isclose(rounded_phase, phase)
            rounded_phase = np.where(phase_is_int, rounded_phase, -1)

            return phase, axis_proj, rounded_phase

        def sign_and_phase(phase, axis_proj, rounded_phase):
            if axis_proj == -1:
                # Not on Re or Im axis
                if rounded_phase == -1:
                    expr = phase_template.format(phase)
                else:
                    expr = f"{rounded_phase}"

                sign = 1

            else:
                if axis_proj % 2 == 1:
                    expr = "/"
                else:
                    expr = "0"

                if axis_proj >= 2:
                    sign = -1
                else:
                    sign = 1

            return sign, expr

        norm_offset, offset_proj, rounded_offset = normalize_phase(phase_offset)
        global_sign, global_phase = sign_and_phase(norm_offset, offset_proj, rounded_offset)

        norm_phase, axis_proj, rounded_phase = normalize_phase(phase)

        ## Compose the terms

        # List of terms. `iterm` indexes the compressed arrays above; `idx` is the corresponding
        # index into the original object, which is what Term carries for _add_labels.
        terms = []

        for iterm, idx in enumerate(term_indices):
            sign, phase_expr = sign_and_phase(
                norm_phase[iterm], axis_proj[iterm], rounded_phase[iterm]
            )

            if rounded_amp[iterm] == -1:
                amp_expr = amp_template.format(absamp[iterm])
            else:
                amp_expr = f"{rounded_amp[iterm]}"

            terms.append(QPrintBase.Term(index=idx, sign=sign, amp=amp_expr, phase=phase_expr))

        return global_sign, global_amp, global_phase, terms

    def _qobj_data(self, qobj):
        """Normalize a supported input to ``(qobj, data)``, where ``data`` holds the amplitudes.

        The type-dispatch shared by all three ``fmt`` classes; subclasses extend it and call up. A
        qutip ``Qobj`` also has its subsystem dimensions read off here when ``dim`` was not given.

        Raises:
            NotImplementedError: If ``qobj`` is not a qutip ``Qobj``, a scipy CSR matrix, or a numpy
                array. This is the error every subclass surfaces for an unsupported input type.
        """
        if HAS_QUTIP and isinstance(qobj, Qobj):
            if self._dim is None:
                # dims[0] is the row (ket) space and dims[1] the column (bra) space. For a bra,
                # dims[0] is the trivial [1] and the real subsystem structure is in dims[1], so
                # taking dims[0] unconditionally raised "Product of subsystem dimensions 1 and qobj
                # dimension 3 do not match" for every bra. Operators have both sides populated and
                # are unaffected either way.
                row_dims, column_dims = tuple(qobj.dims[0]), tuple(qobj.dims[1])
                self._dim = column_dims if np.prod(row_dims) == 1 else row_dims

            # Qobj.full(), not Qobj.data: in qutip 4 `.data` was a scipy sparse matrix, so
            # `qobj.data.data` reached its value buffer, but qutip 5 wraps the payload in its own
            # Dense/CSR class which has no `.data` -- the chained access raised
            # "'qutip.core.data.dense.Dense' object has no attribute 'data'" and made every Qobj
            # input fail. `.full()` returns a dense ndarray in both versions.
            qobj = np.asarray(qobj.full())
            data = qobj
        elif isinstance(qobj, scipy.sparse.csr_matrix):
            data = qobj.data
        elif isinstance(qobj, np.ndarray):
            data = qobj
        else:
            raise NotImplementedError(f"qprint not implemented for {type(qobj)}")

        return qobj, data

    @abstractmethod
    def _add_labels(self, terms, mode):
        pass

    def _format_lhs(self, mode):
        """Return the left-hand-side label, or None. Overridden by subclasses that decorate it.

        Concrete rather than abstract: two of the three subclasses want exactly this, and only
        ``QPrintBraKet`` varies it (bra/ket decoration).
        """
        return self.lhs_label

    def _format_pre_expr(self, global_sign, global_amp, global_phase, mode):
        """Compose the global sign / amplitude / phase prefix shared by every layout.

        ``QPrintMatrix`` overrides ``_make_lines`` for the row-and-column grid but composes this
        prefix identically, so it lives here rather than in two copies that can drift apart.
        """
        pre_expr = "-" if global_sign == -1 else ""
        pre_expr += global_amp
        return pre_expr + self._format_phase(global_phase, mode)

    def _make_lines(self, mode):
        global_sign, global_amp, global_phase, terms = self._process()
        self._add_labels(terms, mode)

        pre_expr = self._format_pre_expr(global_sign, global_amp, global_phase, mode)

        lines = []
        line_expr = ""
        num_terms = 0

        for term in terms:
            if lines or line_expr:
                if term.sign == -1:
                    line_expr += " - "
                else:
                    line_expr += " + "

            elif term.sign == -1:
                line_expr += "-"

            # Track whether anything numeric precedes the label, rather than inspecting the string
            # afterwards: text-mode labels carry the multiplication sign as a prefix
            # (QPrintPauli._add_labels), and both the amplitude and the phase can be absent, so only
            # the caller knows whether that "*" has a left operand.
            wrote_amp = term.amp != "1"
            if wrote_amp:
                line_expr += term.amp

            phase_expr = self._format_phase(term.phase, mode)
            line_expr += phase_expr

            label = term.label
            # A dangling separator: the amplitude is suppressed when it is exactly "1", which
            # rendered a unit-coefficient Pauli term as "- *IZ/2" instead of "- IZ/2". The latex path
            # was unaffected because its labels carry no separator, so the two renderers disagreed on
            # the same term -- which is why this survived: neither output looks wrong on its own.
            if not wrote_amp and not phase_expr and label.startswith("*"):
                label = label[1:]
            line_expr += label

            num_terms += 1
            if num_terms == self.terms_per_row:
                lines.append(line_expr)
                line_expr = ""
                num_terms = 0

        if num_terms != 0:
            lines.append(line_expr)

        if not lines:
            lines = ["0"]

        return pre_expr, lines

    def _format_phase(self, phase_expr, mode):
        if phase_expr == "0":
            return ""
        if phase_expr == "/":
            return "i"

        if mode == "text":
            expr = "["

            if self.phase_norm is not None and self.phase_norm[1]:
                if phase_expr == "1":
                    expr += self.phase_norm[1]
                elif self.phase_norm[1][0].isnumeric():
                    expr += f"{phase_expr}({self.phase_norm[1]})"
                else:
                    expr += f"{phase_expr}{self.phase_norm[1]}"
            else:
                expr += phase_expr

            expr += "]"

        elif mode == "latex":
            expr = "e^{"

            if phase_expr != "1":
                expr += phase_expr

            if self.phase_norm is not None:
                if self.phase_norm[1] and self.phase_norm[1][0].isnumeric():
                    expr += r" \cdot "

                expr += self.phase_norm[1]

            expr += " i}"

        return expr


class QPrintBraKet(QPrintBase):
    """Helper class to compose an expression of a given quantum object.

    Args:
        qobj: Input quantum object.
        amp_norm: Specification of the normalization of amplitudes by (numeric devisor, unit in
            LaTeX).
        phase_norm: Specification of the normalization of phases by (numeric devisor, unit in
            LaTeX).
        global_phase: Specification of the phase to factor out. Give a numeric offset or 'mean'.
        terms_per_row: Number of terms to show per row.
        amp_format: Format for the numerical value of the amplitude absolute values.
        phase_format: Format for the numerical value of the phases.
        amp_cutoff: Ignore terms with absolute amplitudes less than ``max(abs(amplitudes))`` times
            this value.
        lhs_label: If not None, prepend 'label = ' to the printout.
        dim: Subsystem dimensions. If None, the object is treated as a single system of its full
            dimension.
        binary: Show bra and ket indices in binary.

    Raises:
        ValueError: If the product of ``dim`` does not match the object's dimension, or if
            ``binary=True`` for subsystem dimensions that are not powers of two.
        NotImplementedError: If ``qobj`` is not a supported type (inherited from
            :meth:`QPrintBase._qobj_data`).
    """

    class QobjType(Enum):
        KET = 1
        BRA = 2
        OPER = 3

    def __init__(
        self,
        qobj: Any,
        amp_norm: complex | tuple[complex, str] | None = None,
        phase_norm: tuple[complex, str] | None = (np.pi, "π"),
        global_phase: complex | str | None = None,
        terms_per_row: int = 0,
        amp_format: str = ".3f",
        phase_format: str = ".2f",
        amp_cutoff: float = 1.0e-6,
        lhs_label: str | None = None,
        dim: MatrixDimension | None = None,
        binary: bool = False,
    ):
        super().__init__(
            qobj=qobj,
            amp_norm=amp_norm,
            phase_norm=phase_norm,
            global_phase=global_phase,
            terms_per_row=terms_per_row,
            amp_format=amp_format,
            phase_format=phase_format,
            amp_cutoff=amp_cutoff,
            lhs_label=lhs_label,
            dim=dim,
        )

        self.binary = binary

        if len(self._qobj.shape) == 1 or self._qobj.shape[1] == 1:
            self._objtype = QPrintBraKet.QobjType.KET
            self._objdim = self._qobj.shape[0]
        elif self._qobj.shape[0] == 1 and self._qobj.shape[1] != 1:
            self._objtype = QPrintBraKet.QobjType.BRA
            self._objdim = self._qobj.shape[1]
        else:
            self._objtype = QPrintBraKet.QobjType.OPER
            self._objdim = self._qobj.shape[0]

        if self._dim is None:
            self._dim = (self._objdim,)

        if np.prod(self._dim) != self._objdim:
            raise ValueError(
                f"Product of subsystem dimensions {np.prod(self._dim)} and qobj"
                f" dimension {self._objdim} do not match"
            )

    def _add_labels(self, terms, mode):
        has_ket = self._objtype in (QPrintBraKet.QobjType.KET, QPrintBraKet.QobjType.OPER)
        has_bra = self._objtype in (QPrintBraKet.QobjType.BRA, QPrintBraKet.QobjType.OPER)

        # State label format template
        if self.binary:
            log2_dims = np.log2(np.asarray(self._dim))
            if not np.allclose(log2_dims, np.round(log2_dims)):
                raise ValueError("Binary labels requested for dimensions not power-of-two")

            label_template = ",".join(f"{{:0{s}b}}" for s in log2_dims.astype(int))
        else:
            label_template = ",".join(["{}"] * len(self._dim))

        # Make tuples of quantum state labels and format the term indices
        if isinstance(self._qobj, scipy.sparse.csr_matrix):
            # CSR matrix: diff if indptr = number of elements in each row
            repeats = np.diff(self._qobj.indptr)
            row_labels_flat = np.repeat(np.arange(self._qobj.shape[0]), repeats)
            # unravel into row indices accounting for the tensor product. Sized by nnz, so these are
            # cheap to precompute -- unlike the dense case below.
            if has_ket:
                row_labels = np.unravel_index(row_labels_flat, self._dim)
            if has_bra:
                col_labels = np.unravel_index(self._qobj.indices, self._dim)
        else:
            # Dense: unravel per term rather than precomputing a len(dim) x objdim table to read one
            # element per term out of. That table costs 168 MB and 16.7 ms at dim 2^20 over 20
            # subsystems, against 0.009 ms for the handful of indices actually printed.
            row_labels = col_labels = None

        def subsystem_indices(labels, flat_index):
            if labels is None:
                return np.unravel_index(flat_index, self._dim)
            return tuple(axis[flat_index] for axis in labels)

        # Update the term objects with the basis labels
        for term in terms:
            if has_ket:
                ket_label = label_template.format(*subsystem_indices(row_labels, term.index[0]))

                if mode == "text":
                    term.label += f"|{ket_label}>"
                elif mode == "latex":
                    term.label += rf"| {ket_label} \rangle"

            if has_bra:
                # idx can be an 1- or 2-tuple depending on the type of self._qobj
                bra_label = label_template.format(*subsystem_indices(col_labels, term.index[-1]))

                if mode == "text":
                    term.label += f"<{bra_label}|"
                elif mode == "latex":
                    term.label += rf"\langle {bra_label} |"

    def _format_lhs(self, mode):
        if self.lhs_label:
            if mode == "text":
                if self._objtype == QPrintBraKet.QobjType.KET:
                    return f"|{self.lhs_label}>"
                if self._objtype == QPrintBraKet.QobjType.BRA:
                    return f"<{self.lhs_label}|"

            elif mode == "latex":
                if self._objtype == QPrintBraKet.QobjType.KET:
                    return rf"| {self.lhs_label} \rangle"
                if self._objtype == QPrintBraKet.QobjType.BRA:
                    return rf"\langle {self.lhs_label} |"

        return self.lhs_label


class QPrintPauli(QPrintBase):
    """Helper class to compose an expression for a Pauli decomposition from a matrix or components.

    Args:
        qobj: A square matrix (shape `(d1*d2*..., d1*d2*...)`), a structured components array
            (shape `(d1**2, d2**2, ...)`), or a fully flattened components array. Argument `dim` is
            required for the square matrix. For the other two it is optional: when omitted, each
            axis is taken to be one subsystem of dimension `sqrt(len(axis))`, so a flattened array
            is read as a single subsystem and a `ValueError` is raised if any axis length is not a
            perfect square. Pass `dim` to interpret a flattened array as multiple subsystems.
        amp_norm: Specification of the normalization of amplitudes by (numeric devisor, unit in
            LaTeX).
        phase_norm: Specification of the normalization of phases by (numeric devisor, unit in
            LaTeX).
        global_phase: Specification of the phase to factor out. Give a numeric offset or 'mean'.
        terms_per_row: Number of terms to show per row.
        amp_format: Format for the numerical value of the amplitude absolute values.
        phase_format: Format for the numerical value of the phases.
        amp_cutoff: Ignore terms with absolute amplitudes less than ``max(abs(amplitudes))`` times
            this value.
        lhs_label: If not None, prepend 'label = ' to the printout.
        dim: Specification of the dimensions of the subsystems. Used only when `qobj` is a square
            matrix or a 1D array.
        symbol: Pauli matrix symbols.
        delimiter: Pauli product delimiter.

    Raises:
        ValueError: If ``dim`` is None and any axis length of a components array is not a perfect
            square, so the subsystem dimensions cannot be inferred ("qobj shape is invalid").
        NotImplementedError: If ``qobj`` is not a supported type (inherited from
            :meth:`QPrintBase._qobj_data`).
    """

    def __init__(
        self,
        qobj: Any,
        amp_norm: complex | tuple[complex, str] | None = None,
        phase_norm: tuple[complex, str] | None = (np.pi, "π"),
        global_phase: complex | str | None = None,
        terms_per_row: int = 0,
        amp_format: str = ".3f",
        phase_format: str = ".2f",
        amp_cutoff: float = 1.0e-6,
        lhs_label: str | None = None,
        dim: MatrixDimension | None = None,
        symbol: str | Sequence[str] | Sequence[Sequence[str]] | None = None,
        delimiter: str = "",
    ):
        super().__init__(
            qobj=qobj,
            amp_norm=amp_norm,
            phase_norm=phase_norm,
            global_phase=global_phase,
            terms_per_row=terms_per_row,
            amp_format=amp_format,
            phase_format=phase_format,
            amp_cutoff=amp_cutoff,
            lhs_label=lhs_label,
            dim=dim,
        )

        self.symbol = symbol
        self.delimiter = delimiter

    def _qobj_data(self, qobj):
        # Convert all qobj to a components array (shape (d1**2, d2**2, ...))

        qobj, data = super()._qobj_data(qobj)

        if self._dim is not None:
            # Densify once, with the same isinstance test the base class classifies input by, rather
            # than duck-typing on .toarray() separately in each branch below.
            dense = qobj.toarray() if isinstance(qobj, scipy.sparse.csr_matrix) else qobj

            if len(qobj.shape) == 2 and qobj.shape[0] == qobj.shape[1]:
                # This is a matrix -> extract the components
                qobj = pmatrix.components(dense, dim=self._dim)
                data = qobj

            elif len(qobj.shape) == 1:
                # This is a 1D array of components
                qobj = dense.reshape(np.square(self._dim))
                data = qobj

        else:
            self._dim = np.around(np.sqrt(qobj.shape)).astype(int)
            if not np.allclose(np.square(self._dim), qobj.shape):
                raise ValueError("qobj shape is invalid")

        return qobj, data

    def _add_labels(self, terms, mode):
        labels = pmatrix.labels(self._dim, symbol=self.symbol, delimiter=self.delimiter, fmt=mode)

        # Update the term objects with the basis labels
        for term in terms:
            if mode == "text":
                term.label = f"*{labels[term.index]}"
            else:
                term.label = str(labels[term.index])


class QPrintMatrix(QPrintBase):
    """Helper class to lay a square matrix out as a bracketed grid of formatted elements.

    Unlike the other two ``fmt`` classes this one takes no ``dim``, ``symbol`` or ``delimiter``: a
    matrix element is identified by its ``(row, column)`` position, so there is no subsystem
    structure to name and no basis label to build. ``terms_per_row`` is likewise unused -- the row
    width is the matrix width. ``__init__`` is inherited from :class:`QPrintBase` rather than
    restated here, so those defaults live in exactly one place.

    Args:
        qobj: A square matrix, shape `(d1*d2*..., d1*d2*...)`. Anything else raises ``ValueError``.
        amp_norm: Specification of the normalization of amplitudes by (numeric devisor, unit in
            LaTeX).
        phase_norm: Specification of the normalization of phases by (numeric devisor, unit in
            LaTeX).
        global_phase: Specification of the phase to factor out. Give a numeric offset or 'mean'.
        amp_format: Format for the numerical value of the amplitude absolute values.
        phase_format: Format for the numerical value of the phases.
        amp_cutoff: Ignore terms with absolute amplitudes less than ``max(abs(amplitudes))`` times
            this value.
        lhs_label: If not None, prepend 'label = ' to the printout.

    Raises:
        ValueError: If ``qobj`` is not a square matrix.
        NotImplementedError: If ``qobj`` is not a supported type (inherited from
            :meth:`QPrintBase._qobj_data`).
    """

    def _qobj_data(self, qobj):
        # Convert all qobj to a square matrix

        qobj, data = super()._qobj_data(qobj)

        if not (len(qobj.shape) == 2 and qobj.shape[0] == qobj.shape[1]):
            raise ValueError("qobj is not a square matrix")

        return qobj, data

    def _add_labels(self, terms, mode):
        """No-op: a matrix element is identified by its position, not by a basis label.

        ``QPrintBase`` declares this abstract because its own ``_make_lines`` calls it to attach a
        basis label to each term. This class overrides ``_make_lines`` and lays the terms out by
        ``(row, column)`` instead, so there is nothing to label -- but the abstract declaration
        still had to be satisfied, and it was not: ``QPrintMatrix`` could not be instantiated at all
        ("Can't instantiate abstract class QPrintMatrix without an implementation for abstract
        method '_add_labels'"), so ``fmt='matrix'`` raised TypeError for every input and both output
        modes.
        """

    def _make_lines(self, mode):
        global_sign, global_amp, global_phase, terms = self._process()

        matrix_dim = self._data.shape[0]

        pre_expr = self._format_pre_expr(global_sign, global_amp, global_phase, mode)

        rows = [(["0"] * matrix_dim) for _ in range(matrix_dim)]

        for term in terms:
            irow, icolumn = term.index

            element = ""

            if term.sign == -1:
                element += "-"

            # Always emit the amplitude, even when it is exactly "1". Suppressing it is correct where
            # a basis label follows -- "\frac{IZ}{2}" reads better than "1\frac{IZ}{2}" -- but a
            # matrix element has no label, so dropping the 1 left the cell empty and produced
            # "\begin{pmatrix} & 0 & 0 & 0 \\ ...", a malformed matrix missing its first entry.
            element += term.amp

            element += self._format_phase(term.phase, mode)

            rows[irow][icolumn] = element

        lines = []

        if mode == "latex":
            for row in rows:
                lines.append(" & ".join(row))

            lines[0] = r"\begin{pmatrix}" + lines[0]
            lines[-1] += r"\end{pmatrix}"
        else:
            max_col_width = max(max(len(element) for element in row) for row in rows)
            template = f"{{:>{max_col_width}s}}"
            for row in rows:
                lines.append(" ".join(template.format(element) for element in row))

            lines[0] = "⎛" + lines[0] + "⎞"
            for iline in range(1, matrix_dim - 1):
                lines[iline] = "⎜" + lines[iline] + "⎟"
            lines[-1] = "⎝" + lines[-1] + "⎠"

        return pre_expr, lines
