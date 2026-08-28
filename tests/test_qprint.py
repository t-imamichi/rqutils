"""Tests for :mod:`rqutils.qprint`.

The module has two orthogonal axes -- ``fmt`` picks the content class (``QPrintBraKet`` /
``QPrintPauli`` / ``QPrintMatrix``) and ``output`` picks the rendering (``'text'`` / ``'latex'`` /
``'mpl'``) -- so the natural test shape is the product of the two, which is what
:class:`TestFormatOutputMatrix` covers. Four bugs were found and fixed while writing these, all of
them in cells of that product that nothing had ever exercised:

- ``fmt='matrix'`` raised ``TypeError`` for every input and both text modes: ``QPrintMatrix`` never
  implemented the abstract ``_add_labels``, so it could not be instantiated at all.
- ``fmt='matrix', output='latex'`` dropped an amplitude of exactly ``1``, emitting
  ``\\begin{pmatrix} & 0 & ...`` -- a matrix missing its first entry.
- ``fmt='pauli', output='text'`` rendered a unit-coefficient term as ``- *IZ/2``, a dangling
  multiplication sign, while the latex path rendered the same term correctly.
- Every ``qutip.Qobj`` input raised ``AttributeError``, and bras additionally raised ``ValueError``.

The recurring theme is that **the two renderings of the same content disagreed**, and neither looked
obviously wrong alone. So the strongest assertions here are cross-rendering: text and latex must
agree on structure (same term count, same signs, same numbers) even though they differ in syntax.
"""

import numpy as np
import pytest
from conftest import assert_imports_without, assert_type_checks

import rqutils.qprint as q

# A vector with a real, an imaginary, a negative, a fractional, and a below-cutoff amplitude, so a
# single fixture exercises sign handling, phase handling, and the amplitude cutoff at once.
MIXED_VECTOR = np.array([1.0 + 0.0j, 1.0j, -1.0, 0.5, 1e-9, 0.0, 0.0, 0.0])
DIAGONAL_MATRIX = np.diag([1.0, 2.0, 3.0, 4.0]).astype(np.complex128)
FORMATS = ["braket", "pauli", "matrix"]


def text_of(qobj, **kwargs):
    """Render to text. ``output='text'`` returns the object for lazy ``__repr__``, so stringify."""
    return str(q.qprint(qobj, output="text", **kwargs))


def latex_of(qobj, **kwargs):
    return q.qprint(qobj, output="latex", **kwargs)


class TestFormatOutputMatrix:
    """Every ``fmt`` x ``output`` combination must render. This is the grid the bugs hid in."""

    @pytest.mark.parametrize("fmt", FORMATS)
    @pytest.mark.parametrize("output", ["text", "latex"])
    def test_renders_without_raising(self, fmt, output):
        """``fmt='matrix'`` raised ``TypeError`` here for both outputs, for any input.

        ``QPrintBase`` declares ``_add_labels`` abstract because its own ``_make_lines`` calls it;
        ``QPrintMatrix`` overrides ``_make_lines`` and lays terms out by ``(row, column)`` instead, so
        it has nothing to label -- but it never declared that, and Python refused to instantiate it.
        One of three documented formats, dead for every caller.
        """
        kwargs = {"dim": (2, 2)} if fmt == "pauli" else {}
        rendered = q.qprint(DIAGONAL_MATRIX, fmt=fmt, output=output, **kwargs)
        assert str(rendered), f"fmt={fmt} output={output} rendered empty"

    @pytest.mark.parametrize("fmt", FORMATS)
    def test_mpl_returns_a_figure(self, fmt):
        """``output='mpl'`` returns a matplotlib Figure rather than a string."""
        pytest.importorskip("matplotlib")
        from matplotlib.figure import Figure

        kwargs = {"dim": (2, 2)} if fmt == "pauli" else {}
        assert isinstance(q.qprint(DIAGONAL_MATRIX, fmt=fmt, output="mpl", **kwargs), Figure)

    def test_unknown_fmt_and_output_raise(self):
        with pytest.raises(NotImplementedError, match="format"):
            q.qprint(MIXED_VECTOR, fmt="nonsense")
        with pytest.raises(NotImplementedError, match="output"):
            q.qprint(MIXED_VECTOR, output="nonsense")


class TestAmplitudeAndSeparator:
    """A suppressed unit amplitude must not leave a dangling operator or an empty cell.

    Both of these bugs come from the same decision -- omit the amplitude when it is exactly ``1`` --
    applied in two places with opposite consequences. Neither output looked wrong in isolation; the
    tell was that text and latex disagreed.
    """

    def test_unit_coefficient_pauli_term_has_no_stray_asterisk(self):
        """``fmt='pauli', output='text'`` rendered ``- *IZ/2`` instead of ``- IZ/2``.

        Text-mode labels carry the multiplication sign as a prefix, but the amplitude preceding it is
        dropped when it is exactly ``1``, leaving the operator with no left operand.
        """
        rendered = text_of(DIAGONAL_MATRIX, fmt="pauli", dim=(2, 2))
        assert "*IZ" not in rendered or "1*IZ" in rendered, f"dangling separator in {rendered!r}"
        assert " - IZ/2" in rendered, f"expected a bare unit term, got {rendered!r}"
        # And no term may begin with the separator, whatever the coefficients.
        for term in rendered.replace(" - ", " + ").split(" + "):
            assert not term.strip().startswith("*"), f"term {term!r} starts with a separator"

    def test_matrix_latex_keeps_a_unit_entry(self):
        """``fmt='matrix', output='latex'`` emitted ``\\begin{pmatrix} & 0 & ...``.

        Suppressing a unit amplitude is right when a basis label follows, but a matrix element has no
        label, so the cell rendered empty and the matrix lost its first entry.
        """
        rendered = latex_of(DIAGONAL_MATRIX, fmt="matrix")
        assert rendered.startswith(r"\begin{pmatrix}1 &"), f"lost the unit entry: {rendered!r}"
        # Every row must have the full complement of entries.
        body = rendered.replace(r"\begin{pmatrix}", "").replace(r"\end{pmatrix}", "")
        for row in body.split(r"\\"):
            entries = [cell.strip() for cell in row.split("&")]
            assert len(entries) == 4
            assert all(entries), f"empty cell in row {row!r}"

    def test_matrix_text_and_latex_agree_on_entries(self):
        """The two renderings must contain the same numbers, differing only in syntax."""
        matrix = np.array([[1.0, 2.0], [-2.0, 3.0]], dtype=np.complex128)
        text = text_of(matrix, fmt="matrix")
        latex = latex_of(matrix, fmt="matrix")
        for value in ("1", "2", "-2", "3"):
            assert value in text, f"{value} missing from text rendering {text!r}"
            assert value in latex, f"{value} missing from latex rendering {latex!r}"


class TestBraKet:
    """``fmt='braket'``, the default."""

    def test_basis_labels_and_signs(self):
        rendered = text_of(MIXED_VECTOR)
        assert rendered.strip() == "|0> + i|1> - |2> + 0.500|3>"

    def test_amplitude_cutoff_drops_small_terms(self):
        """``amp_cutoff`` is relative to ``max(abs(amplitudes))``.

        The 1e-9 entry in the fixture must be dropped by default and kept with a tiny cutoff, which
        pins the comparison as relative rather than absolute.
        """
        assert "|4>" not in text_of(MIXED_VECTOR)
        assert "|4>" in text_of(MIXED_VECTOR, amp_cutoff=1e-12)

    def test_binary_labels(self):
        """``binary=True`` writes the index in binary, sized by the vector dimension."""
        assert text_of(MIXED_VECTOR, binary=True).strip().startswith("|000>")

    def test_lhs_label(self):
        assert text_of(MIXED_VECTOR, lhs_label="psi").startswith("|psi> = ")

    def test_terms_per_row_splits_the_output(self):
        rendered = text_of(MIXED_VECTOR, terms_per_row=2)
        assert len(rendered.rstrip("\n").split("\n")) == 2

    def test_multi_subsystem_labels_are_comma_separated(self):
        """With ``dim``, indices become per-subsystem tuples."""
        vector = np.zeros(4, dtype=np.complex128)
        vector[1] = 1.0
        assert text_of(vector, dim=(2, 2)).strip() == "|0,1>"

    def test_row_vector_renders_as_a_bra(self):
        vector = np.array([[1.0 + 0.0j, 1.0 + 0.0j]])
        assert text_of(vector).strip() == "<0| + <1|"

    def test_matrix_renders_as_ket_bras(self):
        matrix = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.complex128)
        assert text_of(matrix).strip() == "|0><1|"

    def test_zero_vector_renders_as_zero(self):
        """An all-zero input must render ``0``, not an empty string."""
        assert text_of(np.zeros(4, dtype=np.complex128)).strip() == "0"


class TestPauli:
    """``fmt='pauli'``, which decomposes into the generalized Pauli basis."""

    def test_decomposition_labels(self):
        """The ``/2`` suffix reflects ``paulis``' product normalization for two subsystems."""
        rendered = text_of(DIAGONAL_MATRIX, fmt="pauli", dim=(2, 2))
        assert "II/2" in rendered
        assert "ZI/2" in rendered

    def test_matches_the_component_decomposition(self):
        """The printed coefficients must be the actual Pauli components, not a re-derivation.

        Cross-checks against ``rqutils.paulis.general.components``, which has its own test suite, so
        this catches a formatting layer that silently rescaled or reordered the numbers.
        """
        import rqutils.paulis.general as pg

        matrix = DIAGONAL_MATRIX
        components = np.asarray(pg.components(matrix, dim=(2, 2))).ravel()
        rendered = text_of(matrix, fmt="pauli", dim=(2, 2))
        for value in components[np.abs(components) > 1e-6]:
            # Coefficients print with the default '.3f' amplitude format; 1.0 is suppressed.
            magnitude = abs(value.real)
            if magnitude != 1.0:
                assert f"{magnitude:.3g}" in rendered.replace(".000", ""), (
                    f"coefficient {magnitude} missing from {rendered!r}"
                )

    def test_components_input_is_accepted_directly(self):
        """A components array (shape ``(d1**2, ...)``) is a valid input, not just a matrix."""
        components = np.zeros((4, 4), dtype=np.complex128)
        components[0, 3] = 2.0
        rendered = text_of(components, fmt="pauli")
        assert "IZ" in rendered

    def test_custom_symbol_and_delimiter(self):
        rendered = text_of(DIAGONAL_MATRIX, fmt="pauli", dim=(2, 2), symbol="s", delimiter=",")
        assert "s" in rendered


class TestMatrixFormat:
    """``fmt='matrix'``, the format that could not be instantiated at all."""

    def test_text_uses_bracket_glyphs(self):
        lines = text_of(DIAGONAL_MATRIX, fmt="matrix").rstrip("\n").split("\n")
        assert len(lines) == 4
        assert lines[0].startswith("⎛") and lines[0].endswith("⎞")
        assert lines[-1].startswith("⎝") and lines[-1].endswith("⎠")
        for line in lines[1:-1]:
            assert line.startswith("⎜") and line.endswith("⎟")

    def test_entries_are_positioned_by_row_and_column(self):
        """An off-diagonal entry must land in its own cell, not be collapsed into a term list."""
        matrix = np.array([[0.0, 5.0], [0.0, 0.0]], dtype=np.complex128)
        lines = text_of(matrix, fmt="matrix").rstrip("\n").split("\n")
        assert "5" in lines[0]
        assert "5" not in lines[1]

    def test_columns_are_right_aligned(self):
        """The text renderer pads every cell to a common width, so rows must align."""
        matrix = np.diag([1.0, 200.0]).astype(np.complex128)
        lines = text_of(matrix, fmt="matrix").rstrip("\n").split("\n")
        assert len({len(line) for line in lines}) == 1, f"ragged rows: {lines}"

    def test_non_square_input_raises(self):
        with pytest.raises(ValueError, match="square"):
            q.qprint(np.zeros((2, 3), dtype=np.complex128), fmt="matrix", output="text")

    def test_imaginary_entries_render(self):
        matrix = np.array([[1.0, 2.0j], [-2.0j, 3.0]], dtype=np.complex128)
        rendered = text_of(matrix, fmt="matrix")
        assert "i" in rendered


class TestNormalization:
    """``amp_norm``, ``phase_norm``, and ``global_phase`` factor common quantities out front."""

    def test_amp_norm_scalar_factors_out_a_divisor(self):
        rendered = text_of(MIXED_VECTOR, amp_norm=2.0)
        assert rendered.startswith("2.0 (")
        assert "0.500|0>" in rendered

    def test_amp_norm_tuple_uses_the_given_label(self):
        rendered = text_of(MIXED_VECTOR, amp_norm=(2.0, "a"))
        assert rendered.startswith("a (")

    def test_phase_norm_default_is_pi(self):
        """Phases print as multiples of pi by default, via the ``(np.pi, 'π')`` default."""
        vector = np.array([1.0 + 1.0j, 1.0 - 1.0j]) / np.sqrt(2.0)
        rendered = text_of(vector)
        assert "π" in rendered

    def test_phase_norm_none_prints_radians(self):
        vector = np.array([1.0 + 1.0j, 1.0 - 1.0j]) / np.sqrt(2.0)
        rendered = text_of(vector, phase_norm=None)
        assert "π" not in rendered

    def test_global_phase_mean_is_accepted(self):
        vector = np.array([1.0 + 1.0j, 1.0 - 1.0j]) / np.sqrt(2.0)
        assert text_of(vector, global_phase="mean")

    def test_global_phase_numeric_is_accepted(self):
        """A complex-typed default: ``global_phase`` was annotated ``numbers.Number``, an ABC that
        ``float`` does not statically satisfy, until the annotations were corrected to ``complex``.
        """
        vector = np.array([1.0 + 1.0j, 1.0 - 1.0j]) / np.sqrt(2.0)
        assert text_of(vector, global_phase=np.pi / 4.0)


class TestQutipInput:
    """``qutip.Qobj`` input, which raised for every object until fixed."""

    def test_ket(self):
        """``AttributeError: 'qutip.core.data.dense.Dense' object has no attribute 'data'``.

        ``qobj.data`` was a scipy sparse matrix in qutip 4, so ``qobj.data.data`` reached its value
        buffer; qutip 5 wraps the payload in its own Dense/CSR class with no ``.data``. Every Qobj
        input failed. ``.full()`` is dense ndarray in both versions.
        """
        qutip = pytest.importorskip("qutip")
        state = qutip.basis(3, 0) + qutip.basis(3, 1)
        assert text_of(state).strip() == "|0> + |1>"

    def test_bra(self):
        """Bras additionally hit ``ValueError: Product of subsystem dimensions 1 ...``.

        ``dims[0]`` is the row space, which for a bra is the trivial ``[1]`` -- the subsystem
        structure lives in ``dims[1]``. Taking ``dims[0]`` unconditionally made every bra fail.
        """
        qutip = pytest.importorskip("qutip")
        state = (qutip.basis(3, 0) + qutip.basis(3, 1)).dag()
        assert text_of(state).strip() == "<0| + <1|"

    def test_operator_pauli_and_matrix(self):
        """An operator populates both dims, so it picks up the subsystem structure automatically."""
        qutip = pytest.importorskip("qutip")
        operator = qutip.tensor(qutip.sigmax(), qutip.sigmaz())
        assert "XZ" in text_of(operator, fmt="pauli")
        assert "⎛" in text_of(operator, fmt="matrix")

    def test_agrees_with_the_equivalent_ndarray(self):
        """The Qobj path must be a pure ingest convenience: same content, same output.

        This is the assertion that would have caught the original bug class, since it compares the
        optional-dependency path against the one that always works.
        """
        qutip = pytest.importorskip("qutip")
        state = qutip.basis(3, 0) + qutip.basis(3, 1)
        as_array = np.array([[1.0], [1.0], [0.0]], dtype=np.complex128)
        assert text_of(state) == text_of(as_array)

    def test_multi_subsystem_dims_are_inferred(self):
        qutip = pytest.importorskip("qutip")
        state = qutip.tensor(qutip.basis(2, 0), qutip.basis(2, 1))
        assert text_of(state).strip() == "|0,1>"


class TestUnsupportedInput:
    def test_unsupported_type_raises(self):
        with pytest.raises(NotImplementedError, match="qprint not implemented"):
            q.qprint("not an array", output="text")


class TestPrintReturnTypeIsCheckable:
    """``PrintReturnType``'s ``Figure`` arm must be visible to a static type checker.

    Same defect as ``sqd.HamiltonianInput`` and ``svsim.CircuitInput``: the arm was added by
    ``PrintReturnType |= Figure`` under ``HAS_MPL``, invisible to a checker, so a caller annotating
    against ``qprint(..., output='mpl')``'s documented return could not accept it.

    ``Figure`` is bound on **both** branches of the ``HAS_MPL`` guard -- the real class with
    matplotlib, ``Any`` without -- so naming it unconditionally in the ``type`` statement is safe and
    collapses the union to ``Any`` when matplotlib is absent, which is the pre-existing convention for
    this module's optional types. See ``conftest.assert_type_checks`` for why this shells out to ``ty``.
    """

    def test_figure_is_an_accepted_arm(self):
        pytest.importorskip("matplotlib")
        assert_type_checks(
            "from matplotlib.figure import Figure\n"
            "from rqutils.qprint import PrintReturnType\n"
            "def take(x: PrintReturnType) -> None: ...\n"
            "take('some latex')\n"
            "take(Figure())\n",
            "qprint.PrintReturnType",
        )

    def test_module_works_without_matplotlib(self):
        """Naming ``Figure`` in the alias must not make matplotlib a hard dependency.

        Exercises the two renderings that do not need it, and asserts ``output='mpl'`` still raises
        ``RuntimeError`` rather than failing at import -- this module's documented contract for an
        absent optional dependency.
        """
        assert_imports_without(
            "rqutils.qprint",
            ["matplotlib", "qutip"],
            "import numpy as np\n"
            "assert m.HAS_MPL is False\n"
            "vec = np.array([1.0, 0.0])\n"
            "assert '|0>' in repr(m.qprint(vec, output='text'))\n"
            "assert isinstance(m.qprint(vec, output='latex'), str)\n"
            "try:\n"
            "    m.qprint(vec, output='mpl')\n"
            "except RuntimeError:\n"
            "    pass\n"
            "else:\n"
            "    raise AssertionError('output=mpl must raise RuntimeError without matplotlib')\n",
        )
