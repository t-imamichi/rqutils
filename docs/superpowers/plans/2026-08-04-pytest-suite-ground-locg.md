# First pytest suite (`ground_locg`) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish the repository's first pytest suite, covering `rqutils/ground_locg.py` with tests keyed to the defects in `docs/locg.md`, and fix a mixed-dtype crash found while designing it.

**Architecture:** Four tasks. Task 1 lays the scaffolding (`test` extra, pytest config, `conftest.py` with shared helpers) and proves it with the `_project_out` contract test. Tasks 2 and 3 add the analytic-kernel tests (2x2, then 3x3), each targeting specific audit items. Task 4 adds end-to-end solver tests, fixes the mixed-dtype crash in `rqutils/ground_locg.py`, and updates `CLAUDE.md`.

**Tech Stack:** Python 3.12+, pytest, JAX (hard dependency), NumPy. Managed by `uv`.

## Global Constraints

- **Always `uv run`, never bare `python`.** The venv at `.venv` is uv-managed.
- **Test invocation is `uv run --extra test pytest`.** `pytest` is declared in a `test` extra, matching the repo's `mpl`/`qutip`/`qiskit`/`docs` convention.
- **`jax_enable_x64` must be enabled before any `rqutils` import.** Without it JAX silently yields float32/complex64 and every tolerance below is wrong. This lives in `tests/conftest.py`.
- **All numerical tolerances are relative to `max(abs(mat))`.** Absolute tolerances are wrong here: the 1e9-shifted 2x2 case has absolute residual 6e-8 but relative residual 6e-17.
- **The kernels read only the diagonal and lower triangle.** Any reference comparison must symmetrize the input the same way first, or it compares against a different matrix.
- **Randomness uses seeded `numpy.random.default_rng`.** No `hypothesis` dependency.
- **Line width is 100 characters.** No linter is configured; match surrounding style.
- **Runtime budget: whole suite under ~30 s.** Measured components: 2000-matrix kernel sweep ≈ 0.01 s, full end-to-end set ≈ 1 s.
- **Do not claim tests pass without running them.** Every task has an explicit run step.

---

### Task 1: Scaffolding and `_project_out` contract

Establishes the suite's infrastructure and proves it end-to-end with one small test. The `_project_out` test is bundled here rather than split out because it is two assertions and needs no helpers beyond what the scaffolding provides — splitting it would create a task a reviewer could not meaningfully reject on its own.

**Files:**
- Modify: `pyproject.toml` (add `test` extra at line 28-32 block; add `[tool.pytest.ini_options]`)
- Create: `tests/conftest.py`
- Create: `tests/test_ground_locg.py`

**Interfaces:**
- Consumes: nothing (first task).
- Produces: four helpers importable from `conftest.py` by later tasks via pytest's automatic conftest injection — but note these are **plain module-level functions, not fixtures**, so test modules import them explicitly: `from conftest import herm, symmetrize, lowest, rel_resid`.
  - `herm(n: int, rng: np.random.Generator, complex_: bool = True) -> np.ndarray` — random Hermitian, shape `(n, n)`.
  - `symmetrize(mat: np.ndarray) -> np.ndarray` — mirrors the lower triangle over the diagonal.
  - `lowest(mat: np.ndarray) -> float` — `numpy.linalg.eigvalsh(symmetrize(mat))[0]`.
  - `rel_resid(mat: np.ndarray, val: float, vec: np.ndarray) -> float` — `norm(mat @ vec - val * vec) / max(abs(mat))`.

- [ ] **Step 1: Add the `test` extra and pytest config to `pyproject.toml`**

In the existing `[project.optional-dependencies]` block (currently lines 28-32), add the `test` line so the block reads:

```toml
[project.optional-dependencies]
mpl = ["matplotlib"]
qutip = ["qutip"]
qiskit = ["qiskit"]
docs = ["sphinx", "sphinx-rtd-theme"]
test = ["pytest"]
```

Then append a new block at the end of the file:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
```

`python_files` is explicit so collection never considers the three Jupyter notebooks in `tests/` or any other non-test file there.

- [ ] **Step 2: Write `tests/conftest.py`**

```python
"""Shared configuration and helpers for the rqutils test suite.

``jax_enable_x64`` is set here, at conftest import time, so that it takes effect before any test
module imports ``rqutils``. This is load-bearing rather than incidental: without x64 JAX silently
produces float32/complex64 and every tolerance in this suite is wrong by nine orders of magnitude.
"""
import jax
jax.config.update('jax_enable_x64', True)

import numpy as np  # noqa: E402  (must follow the x64 config)


def herm(n, rng, complex_=True):
    """Return a random ``(n, n)`` Hermitian matrix drawn from ``rng``."""
    mat = rng.normal(size=(n, n))
    if complex_:
        mat = mat + 1.j * rng.normal(size=(n, n))
    return mat + mat.conjugate().T


def symmetrize(mat):
    """Mirror the lower triangle of ``mat`` over the diagonal.

    ``eigenpair_2x2`` and ``eigenpair_3x3`` read only the diagonal and the lower triangle, so a
    reference eigendecomposition must be taken of *this* matrix, not of the raw input. Comparing
    against ``eigvalsh`` of an unsymmetrized input compares against a different matrix.
    """
    lower = np.tril(mat)
    return lower + np.tril(mat, -1).conjugate().T


def lowest(mat):
    """Reference lowest eigenvalue of ``mat``, via LAPACK, after symmetrization."""
    return float(np.linalg.eigvalsh(symmetrize(mat))[0])


def rel_resid(mat, val, vec):
    """Eigenpair residual ``|Av - λv|``, scaled by ``max|A|``.

    The scaling is what makes a single tolerance usable across the shifted and extreme-scale cases:
    the 1e9-shifted 2x2 input has an absolute residual of 6e-8 but a relative residual of 6e-17.
    """
    mat = symmetrize(mat)
    vec = np.asarray(vec)
    return float(np.linalg.norm(mat @ vec - val * vec) / np.abs(mat).max())
```

- [ ] **Step 3: Write the failing `_project_out` test**

Create `tests/test_ground_locg.py`:

```python
"""Tests for :mod:`rqutils.ground_locg`.

Organized by defect. Most tests correspond to a numbered item in ``docs/locg.md``, the audit of the
previous implementation of this module; each such test names its item and the measured old failure.
That audit's closing warning shapes the design here:

    Aggregate random sampling does not exercise data-dependent branches evenly. [...] Both of my
    safety nets missed a sign error that a two-line targeted test caught at 4000/4000 failures.

So targeted per-branch tests are the backbone and randomized sweeps are a supplement.
"""
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import herm, lowest, rel_resid, symmetrize

from rqutils.ground_locg import _project_out


class TestProjectOut:
    """``_project_out`` returns either exactly zero or a vector of norm >= 0.99 (item I6)."""

    def test_vector_in_basis_span_returns_exactly_zero(self):
        """A residual lying wholly in span{x, y} must come back as exact zero, not as noise.

        The zeroing guard is what item I7's convergence check keys off: a zeroed search direction
        means {x, y} already spans the residual, so no further iteration can lower theta.
        """
        basis = (jnp.array([1., 0., 0.]), jnp.array([0., 1., 0.]))
        out = np.asarray(_project_out(basis, jnp.array([1., 1., 0.])))
        assert np.array_equal(out, np.zeros(3))

    def test_orthogonal_vector_is_not_normalized_to_unity(self):
        """The postcondition is norm >= 0.99, NOT norm == 1.

        Item I6: callers feeding this to a standard Rayleigh-Ritz step must renormalize themselves.
        At shift 1e9 a |p| of 0.999 displaced theta by 2e6, below the true minimum. This test exists
        so that a future edit "tidying" the trailing subtraction into a normalization fails loudly.
        """
        basis = (jnp.array([1., 0., 0.]), jnp.array([0., 1., 0.]))
        out = np.asarray(_project_out(basis, jnp.array([0., 0., 2.])))
        assert np.linalg.norm(out) >= 0.99
        assert np.allclose(out, [0., 0., 1.])
```

- [ ] **Step 4: Run the tests and confirm they pass**

```bash
uv run --extra test pytest tests/test_ground_locg.py -v
```

Expected: 2 passed. Both behaviors were verified during design (`_project_out` returns `[0,0,0]` for the in-span case and `[0,0,1]` for the orthogonal case), so these pass immediately — they are regression locks on existing behavior, not TDD drivers. If either fails, the scaffolding is wrong (most likely `conftest.py` not importable, meaning `python_files`/`testpaths` is misconfigured), not the library.

- [ ] **Step 5: Confirm the notebooks are not collected**

```bash
uv run --extra test pytest --collect-only -q
```

Expected: only `tests/test_ground_locg.py` items listed; no mention of `paulis.ipynb`, `qprint.ipynb`, or `sqd.ipynb`.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml tests/conftest.py tests/test_ground_locg.py
git commit -m "Add pytest scaffolding and _project_out contract tests

First automated tests in the repository. pytest arrives as a 'test' extra,
matching the existing mpl/qutip/qiskit/docs convention. conftest.py enables
jax_enable_x64 before any rqutils import, without which every tolerance in
the suite would be wrong.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 2: `eigenpair_2x2` tests

The predecessor returned **2353 NaNs in 40000 random inputs**. Items I1 (trace cancellation), I2 (no balancing), and a 0/0 at the old line 431.

**Files:**
- Modify: `tests/test_ground_locg.py` (append `TestEigenpair2x2`; extend the import line)

**Interfaces:**
- Consumes: `herm`, `lowest`, `rel_resid`, `symmetrize` from `conftest` (Task 1).
- Produces: `two_by_two(rng, delta_sign)` — a module-level helper in `tests/test_ground_locg.py` returning a `(2, 2)` complex Hermitian `np.ndarray` whose `delta = (d0 - d1) / 2` has the requested sign. Not needed by later tasks.

- [ ] **Step 1: Extend the import in `tests/test_ground_locg.py`**

Change the existing import line to bring in the kernel:

```python
from rqutils.ground_locg import _project_out, eigenpair_2x2
```

- [ ] **Step 2: Add the `two_by_two` helper and `TestEigenpair2x2`**

Append to `tests/test_ground_locg.py`:

```python
def two_by_two(rng, delta_sign):
    """Return a 2x2 complex Hermitian matrix whose delta = (d0 - d1) / 2 has ``delta_sign``.

    ``eigenpair_2x2`` selects which row of the singular shifted matrix yields the null vector on the
    sign of delta, so the two signs are genuinely different code paths and must be tested apart.
    """
    offd = rng.normal() + 1.j * rng.normal()
    diag = rng.normal(size=2)
    if np.sign(diag[0] - diag[1]) != delta_sign:
        diag = diag[::-1]
    return np.array([[diag[0], offd.conjugate()], [offd, diag[1]]])


class TestEigenpair2x2:
    """Lowest eigenpair of a 2x2 Hermitian matrix.

    The shipped predecessor returned 2353 NaNs over 40000 random inputs.
    """

    @pytest.mark.parametrize('mat', [np.diag([1., 5.]), np.diag([5., 1.]), np.eye(2)])
    def test_diagonal_and_identity(self, mat):
        """Old kernel returned NaN here: its eigenvector formula computed 0/0.

        Any multiple of the identity hit it, and so did any already-diagonal input whose lower
        eigenvalue sat in position 0 -- diag(1, 5) returned NaN.
        """
        eigval, eigvec = eigenpair_2x2(jnp.asarray(mat))
        eigval = float(eigval)
        assert np.isfinite(eigval)
        assert np.all(np.isfinite(np.asarray(eigvec)))
        assert eigval == pytest.approx(lowest(mat), abs=1e-13 * np.abs(mat).max())
        assert rel_resid(mat, eigval, eigvec) < 1e-13
        assert np.linalg.norm(np.asarray(eigvec)) == pytest.approx(1.)

    def test_large_shift(self):
        """Item I1: ``tr*tr - 4*det`` cancelled, reaching relative error 5.8e-1 at shift 1e9."""
        mat = np.array([[1., 0.5], [0.5, 2.]]) + 1e9 * np.eye(2)
        eigval, eigvec = eigenpair_2x2(jnp.asarray(mat))
        eigval = float(eigval)
        assert eigval == pytest.approx(lowest(mat), rel=1e-13)
        assert rel_resid(mat, eigval, eigvec) < 1e-13

    @pytest.mark.parametrize('exponent', [-160, 160])
    def test_extreme_scale(self, exponent):
        """Item I2: unbalanced intermediates carried 4.9e-2 error at 1e-160 and NaN at 1e160."""
        mat = np.array([[1., 0.5], [0.5, 2.]]) * 10. ** exponent
        eigval, eigvec = eigenpair_2x2(jnp.asarray(mat))
        eigval = float(eigval)
        assert np.isfinite(eigval)
        assert eigval == pytest.approx(lowest(mat), rel=1e-13)
        assert rel_resid(mat, eigval, eigvec) < 1e-13

    @pytest.mark.parametrize('delta_sign', [1., -1.])
    def test_both_delta_branches(self, delta_sign):
        """Each sign of delta separately -- the audit's own sign error hid from aggregate sampling.

        Its first fix had the row-2 branch as ``[rad - delta, +b]`` instead of ``-b``. That passed
        every test it had, because those tests happened to generate only delta > 0. Forcing delta < 0
        exposed it at 4000/4000 failures, residual 3.8e-1. End-to-end convergence hides it too:
        ``eigenpair_2x2`` is called exactly once, in ``body_iter1``, so a wrong eigenvector there is
        quietly repaired by later iterations.
        """
        rng = np.random.default_rng(20260804)
        worst_eigval = worst_residual = 0.
        for _ in range(2000):
            mat = two_by_two(rng, delta_sign)
            diag = np.diagonal(mat).real
            assert np.sign((diag[0] - diag[1]) / 2.) == delta_sign  # the branch really is forced
            eigval, eigvec = eigenpair_2x2(jnp.asarray(mat))
            eigval = float(eigval)
            assert np.isfinite(eigval)
            scale = np.abs(mat).max()
            worst_eigval = max(worst_eigval, abs(eigval - lowest(mat)) / scale)
            worst_residual = max(worst_residual, rel_resid(mat, eigval, eigvec))
        assert worst_eigval < 1e-13, f'worst relative eigenvalue error {worst_eigval:.2e}'
        assert worst_residual < 1e-13, f'worst relative residual {worst_residual:.2e}'
```

- [ ] **Step 3: Run and confirm all pass**

```bash
uv run --extra test pytest tests/test_ground_locg.py::TestEigenpair2x2 -v
```

Expected: 8 passed (3 diagonal/identity + 1 shift + 2 scales + 2 delta branches). Measured during design: worst relative eigenvalue error 1.12e-15 (delta>0) and 8.33e-16 (delta<0); the 1e160 case sits at 7.8e-17, and the others are exact. All comfortably inside 1e-13.

- [ ] **Step 4: Confirm the delta assertion has teeth**

The `assert np.sign(...) == delta_sign` inside the loop guards against `two_by_two` silently failing to force the branch, which would make both parametrizations test the same path. Verify it fires when broken:

```bash
uv run --extra test python -c "
import numpy as np
sys_ok = True
rng = np.random.default_rng(0)
# Deliberately wrong helper: never swaps, so signs are random.
def bad(rng, sign):
    offd = rng.normal() + 1j*rng.normal(); diag = rng.normal(size=2)
    return np.array([[diag[0], offd.conjugate()], [offd, diag[1]]])
signs = [np.sign((np.diagonal(bad(rng, -1.)).real[0] - np.diagonal(bad(rng, -1.)).real[1])/2.) for _ in range(50)]
print('a non-forcing helper yields mixed signs:', len(set(signs)) > 1)
"
```

Expected: `True` — confirming that if `two_by_two` stopped forcing the sign, the in-loop assertion would catch it rather than the tests silently degenerating into one branch.

- [ ] **Step 5: Commit**

```bash
git add tests/test_ground_locg.py
git commit -m "Add eigenpair_2x2 tests, both delta branches separately

Covers the audit's I1 (trace cancellation at large shift), I2 (balancing at
extreme scale), and the 0/0 that returned NaN for any diagonal input with the
lower eigenvalue in position 0.

The delta branches are parametrized apart deliberately: the audit records a
sign error in the row-2 branch that passed every aggregate test because those
tests only generated delta > 0, and showed 4000/4000 failures once the sign
was forced.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 3: `eigenpair_3x3` tests

The predecessor returned **4066 NaNs in 20000 random inputs** and a wrong eigenvector **1148 times in 5000**. Items I1, I2, I3 (fixed column pair in the cross product).

**Files:**
- Modify: `tests/test_ground_locg.py` (append `TestEigenpair3x3`; extend the import line)

**Interfaces:**
- Consumes: `herm`, `lowest`, `rel_resid` from `conftest` (Task 1).
- Produces: nothing used by later tasks.

- [ ] **Step 1: Extend the import**

```python
from rqutils.ground_locg import _project_out, eigenpair_2x2, eigenpair_3x3
```

- [ ] **Step 2: Append `TestEigenpair3x3`**

```python
class TestEigenpair3x3:
    """Lowest eigenpair of a 3x3 Hermitian matrix, via Cardano's method.

    The shipped predecessor returned 4066 NaNs over 20000 random inputs, and a wrong eigenvector
    1148 times in 5000.
    """

    def test_rank_deficient_column_pair(self):
        """Item I3, the sharpest case in the audit: diag(5, 6, 1).

        The old kernel took the null vector as ``cross(mat[:, 1], mat[:, 2])``. When that particular
        pair is rank deficient the cross product vanishes and the result points nowhere useful. On
        this innocuous input it returned an *exact eigenvalue* alongside an eigenvector with
        residual 0.67 -- so a test asserting only on the eigenvalue would have passed. Assert both.
        """
        mat = np.diag([5., 6., 1.])
        eigval, eigvec = eigenpair_3x3(jnp.asarray(mat))
        eigval = float(eigval)
        assert eigval == pytest.approx(lowest(mat), abs=1e-13)
        assert rel_resid(mat, eigval, eigvec) < 1e-13
        assert np.linalg.norm(np.asarray(eigvec)) == pytest.approx(1.)

    def test_identity_rank_zero(self):
        """Item I3 rank-0 fallback: every cross product vanishes for a multiple of the identity."""
        mat = np.eye(3)
        eigval, eigvec = eigenpair_3x3(jnp.asarray(mat))
        eigval = float(eigval)
        assert eigval == pytest.approx(1., abs=1e-13)
        assert rel_resid(mat, eigval, eigvec) < 1e-13
        assert np.linalg.norm(np.asarray(eigvec)) == pytest.approx(1.)

    def test_degenerate_lowest_rank_one(self):
        """Item I3 rank-1 fallback: a degenerate lowest eigenvalue.

        Every cross product is numerical noise here; the null space is the orthogonal complement of
        the largest column, and any member of it is a valid eigenvector.
        """
        mat = np.diag([1., 1., 7.])
        eigval, eigvec = eigenpair_3x3(jnp.asarray(mat))
        eigval = float(eigval)
        assert eigval == pytest.approx(1., abs=1e-13)
        assert rel_resid(mat, eigval, eigvec) < 1e-13
        assert np.linalg.norm(np.asarray(eigvec)) == pytest.approx(1.)

    def test_large_shift(self):
        """Item I1: at shift 1e9 the radicand under ``sqrt`` went negative and returned NaN.

        Not an exotic input -- this is the ordinary case for a physical Hamiltonian, which is rarely
        traceless, and is exactly what ``sqd.py`` feeds this solver.
        """
        mat = np.diag([1., 2., 3.]) + 1e9 * np.eye(3)
        eigval, eigvec = eigenpair_3x3(jnp.asarray(mat))
        eigval = float(eigval)
        assert np.isfinite(eigval)
        assert eigval == pytest.approx(lowest(mat), rel=1e-13)
        assert rel_resid(mat, eigval, eigvec) < 1e-13

    @pytest.mark.parametrize('exponent', [-160, 150])
    def test_extreme_scale(self, exponent):
        """Item I2: ``c0`` is cubic in the entries, so unbalanced it overflows or underflows.

        Measured on the old kernel: relative error 7.8e-1 at 1e-160, NaN at 1e150.
        """
        mat = np.diag([1., 2., 3.]) * 10. ** exponent
        eigval, eigvec = eigenpair_3x3(jnp.asarray(mat))
        eigval = float(eigval)
        assert np.isfinite(eigval)
        assert eigval == pytest.approx(lowest(mat), rel=1e-13)
        assert rel_resid(mat, eigval, eigvec) < 1e-13

    @pytest.mark.parametrize('complex_', [True, False])
    def test_random_sweep(self, complex_):
        """Aggregate accuracy over seeded random input. Supplements the targeted cases above.

        The old kernel produced 4066 NaNs in 20000 such matrices.
        """
        rng = np.random.default_rng(20260804)
        worst_eigval = worst_residual = 0.
        for _ in range(2000):
            mat = herm(3, rng, complex_=complex_)
            eigval, eigvec = eigenpair_3x3(jnp.asarray(mat))
            eigval = float(eigval)
            assert np.isfinite(eigval)
            assert np.all(np.isfinite(np.asarray(eigvec)))
            scale = np.abs(mat).max()
            worst_eigval = max(worst_eigval, abs(eigval - lowest(mat)) / scale)
            worst_residual = max(worst_residual, rel_resid(mat, eigval, eigvec))
        assert worst_eigval < 1e-13, f'worst relative eigenvalue error {worst_eigval:.2e}'
        assert worst_residual < 1e-13, f'worst relative residual {worst_residual:.2e}'
```

- [ ] **Step 3: Run and confirm all pass**

```bash
uv run --extra test pytest tests/test_ground_locg.py::TestEigenpair3x3 -v
```

Expected: 8 passed (3 rank cases + 1 shift + 2 scales + 2 sweeps). Measured during design: 0 NaN, worst relative eigenvalue error 1.60e-15, worst relative residual 1.10e-15; the extreme-scale cases were exact.

- [ ] **Step 4: Confirm `test_rank_deficient_column_pair` targets what it claims**

The point of that test is the *eigenvector*, not the eigenvalue — the old kernel got the eigenvalue exactly right here. Verify the residual assertion is the load-bearing one by checking that a deliberately wrong eigenvector fails it while the eigenvalue assertion still passes:

```bash
uv run --extra test python -c "
import numpy as np
import sys; sys.path.insert(0, 'tests')
from conftest import lowest, rel_resid
mat = np.diag([5., 6., 1.])
good = np.array([0., 0., 1.]); bad = np.array([1., 0., 0.])
print('eigenvalue assertion passes either way:', abs(1.0 - lowest(mat)) < 1e-13)
print('residual, correct eigenvector:', rel_resid(mat, 1.0, good))
print('residual, wrong eigenvector:  ', rel_resid(mat, 1.0, bad))
"
```

Expected: eigenvalue assertion `True`; residual 0.0 for the correct eigenvector and ~0.67 for the wrong one — matching the audit's measured 0.67 and confirming the residual assertion is what catches I3.

- [ ] **Step 5: Commit**

```bash
git add tests/test_ground_locg.py
git commit -m "Add eigenpair_3x3 tests covering all three rank cases

Items I1-I3. diag(5, 6, 1) is the sharpest of these: the old kernel returned
an exact eigenvalue there alongside an eigenvector with residual 0.67, so
every kernel test asserts on both the eigenvalue and the residual.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 4: End-to-end solver tests, mixed-dtype fix, and `CLAUDE.md`

The only task that edits `rqutils/`. The dtype fix and its test ship together because the test cannot pass without the fix, and the fix has no other justification.

**Files:**
- Modify: `rqutils/ground_locg.py` (insert dtype promotion before the `body_iter0` call near line 466; simplify the `tol is None` branch at lines 473-476)
- Modify: `tests/test_ground_locg.py` (append `TestGroundLocg` and `TestDtypes`; extend the import line)
- Modify: `CLAUDE.md` (rewrite the "Testing" section, lines 30-32)

**Interfaces:**
- Consumes: `herm`, `lowest` from `conftest` (Task 1).
- Produces: `work_dtype` — a local in `_ground_locg_callable`, `jnp.dtype`. Not a public interface; named here because the `tol is None` branch below refers to it.

- [ ] **Step 1: Write the failing dtype test**

Extend the import first:

```python
from rqutils.ground_locg import _project_out, eigenpair_2x2, eigenpair_3x3, ground_locg
```

Then append:

```python
class TestDtypes:
    """Operator and ``xinit`` dtype combinations.

    A float32 ``xinit`` against a float64 operator raised ``TypeError`` from ``while_loop``: theta
    entered the carry as float32 from ``body_iter1``, whose projected matrix takes its dtype from
    ``xinit``, and returned float64 from ``body``, whose projected matrix is built from the matvec
    output. Only a lower-precision ``xinit`` failed; a float32 operator with a float64 ``xinit`` was
    always fine.

    This is adjacent to audit item A3, which concerned the same configuration: the rewrite fixed the
    *tolerance* derivation for it but left the state dtype inconsistent.
    """

    @pytest.mark.parametrize('operator_dtype,xinit_dtype', [
        (jnp.float64, jnp.float64),
        (jnp.float32, jnp.float32),
        (jnp.float64, jnp.float32),
        (jnp.complex128, jnp.float32),
    ])
    def test_dtype_combinations_solve(self, operator_dtype, xinit_dtype):
        rng = np.random.default_rng(20260804)
        mat = herm(60, rng, complex_=False)
        xinit = rng.normal(size=60)
        eigval, eigvec, _, converged = ground_locg(jnp.asarray(mat, dtype=operator_dtype),
                                                  jnp.asarray(xinit, dtype=xinit_dtype))
        assert bool(converged)
        assert np.all(np.isfinite(np.asarray(eigvec)))
        # float32 arithmetic reaches only ~1e-6 relative; float64 operators reach ~1e-13.
        tolerance = 1e-5 if jnp.finfo(operator_dtype).bits == 32 else 1e-12
        assert float(eigval) == pytest.approx(lowest(mat), rel=tolerance)

    def test_low_precision_xinit_does_not_degrade_result(self):
        """Audit item A3's concern: a low-precision initial guess must not cost accuracy.

        A4/A3 measured the old ``tol`` derivation trading eight orders of magnitude of accuracy on
        the dtype of the *starting guess* alone.
        """
        rng = np.random.default_rng(20260804)
        mat = jnp.asarray(herm(60, rng, complex_=False), dtype=jnp.float64)
        xinit = rng.normal(size=60)
        from_f64 = float(ground_locg(mat, jnp.asarray(xinit, dtype=jnp.float64))[0])
        from_f32 = float(ground_locg(mat, jnp.asarray(xinit, dtype=jnp.float32))[0])
        assert from_f32 == pytest.approx(from_f64, rel=1e-12)
```

- [ ] **Step 2: Run it and confirm the mixed-dtype cases fail**

```bash
uv run --extra test pytest tests/test_ground_locg.py::TestDtypes -v
```

Expected: 2 passed, 3 failed. The two same-dtype parametrizations pass; the `(float64, float32)` and `(complex128, float32)` parametrizations and `test_low_precision_xinit_does_not_degrade_result` fail with:

```
TypeError: while_loop body function carry input and carry output must have equal types
  The input carry component state[2] has type float32[] but the corresponding output carry
  component has type float64[]
```

**Do not proceed until you have seen this failure.** It is the evidence that the fix in Step 3 is load-bearing rather than decorative.

- [ ] **Step 3: Apply the dtype promotion in `rqutils/ground_locg.py`**

The current code around line 464 reads:

```python
    xinit = normalize(xinit)

    vs_iter0 = body_iter0(xinit)
```

Insert the promotion between them, so it reads:

```python
    xinit = normalize(xinit)

    # The projected matrix inherits xinit's dtype at the seed step but the operator's inside the
    # loop, so a lower-precision xinit makes while_loop's carry types disagree on theta. Promote up
    # front. eval_shape reads the operator dtype without spending a matrix-vector product, and
    # astype on a matching dtype is a no-op, so the common path is unaffected.
    work_dtype = jnp.result_type(xinit.dtype,
                                 jax.eval_shape(lambda vec: matvec(vec, *args), xinit).dtype)
    xinit = xinit.astype(work_dtype)

    vs_iter0 = body_iter0(xinit)
```

This must come *after* the integer one-hot conversion near line 298 (it does — that is far above) and *before* `body_iter0`, since that call is where the projected matrix's dtype originates.

- [ ] **Step 4: Simplify the `tol is None` branch to reuse `work_dtype`**

It currently reads:

```python
    if tol is None:
        # Derive the tolerance from the operator, not from the initial guess: a float32 xinit on a
        # complex128 problem would otherwise silently loosen this by nine orders of magnitude.
        tol = float(jnp.finfo(jnp.result_type(xinit.dtype, vs_iter0[2].dtype)).eps)
```

Replace with:

```python
    if tol is None:
        # Derive the tolerance from the operator, not from the initial guess: a float32 xinit on a
        # complex128 problem would otherwise silently loosen this by nine orders of magnitude.
        # work_dtype above is already that promotion.
        tol = float(jnp.finfo(work_dtype).eps)
```

- [ ] **Step 5: Run the dtype tests and confirm they now pass**

```bash
uv run --extra test pytest tests/test_ground_locg.py::TestDtypes -v
```

Expected: 5 passed. Measured during design: the `(float64, float32)` case solves to eigenvalue error 2.1e-14 in 65 iterations — identical iteration count to the all-float64 case — and `(float32, float32)` remains 21 iterations.

- [ ] **Step 6: Append the end-to-end solver tests**

```python
class TestGroundLocg:
    """End-to-end LOBPCG solves."""

    @pytest.mark.parametrize('shift', [0., 1e3, -1e3, 1e6, 1e9])
    @pytest.mark.parametrize('complex_', [True, False])
    def test_solves_and_converges(self, shift, complex_):
        """Three assertions per case, each keyed to a different audit item.

        Item I4 -- ``reltol`` used ``norm(Ax) - theta``, a cancellation measured going *negative*,
        which made the convergence test unsatisfiable: the solver never converged and always burned
        ``maxiter``. Fixing that sign alone was a 24-33x iteration reduction.

        Item I5, the audit's most serious finding -- loss of x/y orthogonality made the standard
        Rayleigh-Ritz step return a theta *beneath* the true minimum (observed 6.0e8 against a true
        minimum of 1.0e9), which is impossible for a genuine Rayleigh quotient. A caller checking
        only "is it finite" would have accepted it.
        """
        rng = np.random.default_rng(20260804)
        mat = herm(200, rng, complex_=complex_) + shift * np.eye(200)
        reference = lowest(mat)
        scale = np.abs(mat).max()
        xinit = rng.normal(size=200)
        if complex_:
            xinit = xinit + 0.j
        eigval, eigvec, _, converged = ground_locg(jnp.asarray(mat), jnp.asarray(xinit))
        eigval = float(eigval)

        assert bool(converged), 'solver exhausted maxiter (item I4)'
        assert abs(eigval - reference) / scale < 1e-12
        assert eigval > reference - 1e-8 * scale, (
            f'theta {eigval!r} is below the true minimum {reference!r} (item I5)'
        )

    def test_maxiter_too_small_reports_not_converged(self):
        """Item A1: ``niter == maxiter`` is ambiguous, so the flag is the only usable signal.

        Measured on the old code with ``maxiter=5``: a confident-looking eigenvalue whose true
        relative residual was 1.1e-1, and no way for the caller to tell.
        """
        rng = np.random.default_rng(20260804)
        mat = herm(60, rng, complex_=False)
        eigval, _, niter, converged = ground_locg(jnp.asarray(mat),
                                                  jnp.asarray(rng.normal(size=60)), maxiter=5)
        assert not bool(converged)
        assert int(niter) == 5
        assert np.isfinite(float(eigval))

    def test_maxiter_zero_returns_rayleigh_quotient(self):
        """Item A2: the old code returned the literal state initializer, 0.0.

        For a matrix whose true lowest eigenvalue was -21.35 it reported 0.000000 with a normalized
        eigenvector and niter=0.
        """
        rng = np.random.default_rng(20260804)
        mat = herm(60, rng, complex_=False)
        xinit = rng.normal(size=60)
        eigval, _, niter, converged = ground_locg(jnp.asarray(mat), jnp.asarray(xinit), maxiter=0)
        normalized = xinit / np.linalg.norm(xinit)
        assert int(niter) == 0
        assert not bool(converged)
        assert float(eigval) == pytest.approx(normalized @ symmetrize(mat) @ normalized)

    def test_zero_xinit_is_finite(self):
        """Item A4: the old ``xinit /= norm(xinit)`` had no zero guard and returned NaN."""
        rng = np.random.default_rng(20260804)
        mat = herm(60, rng, complex_=False)
        eigval, eigvec, _, _ = ground_locg(jnp.asarray(mat), jnp.zeros(60))
        assert np.isfinite(float(eigval))
        assert np.all(np.isfinite(np.asarray(eigvec)))

    def test_integer_xinit_one_hot(self):
        """The integer convenience path builds a one-hot vector internally."""
        rng = np.random.default_rng(20260804)
        mat = herm(60, rng, complex_=False)
        eigval, _, _, converged = ground_locg(jnp.asarray(mat), jnp.asarray(3))
        assert bool(converged)
        assert float(eigval) == pytest.approx(lowest(mat), rel=1e-12)

    def test_xinit_scale_invariance(self):
        """``xinit`` is normalized on entry, so its magnitude cannot matter."""
        rng = np.random.default_rng(20260804)
        mat = jnp.asarray(herm(60, rng, complex_=False))
        xinit = jnp.asarray(rng.normal(size=60))
        assert float(ground_locg(mat, xinit * 1e8)[0]) == float(ground_locg(mat, xinit)[0])

    def test_xinit_not_mutated(self):
        """Despite the in-place-looking ``xinit = normalize(xinit)``, JAX arrays are immutable."""
        rng = np.random.default_rng(20260804)
        mat = jnp.asarray(herm(60, rng, complex_=False))
        xinit = jnp.asarray(rng.normal(size=60))
        before = np.asarray(xinit).copy()
        ground_locg(mat, xinit)
        assert np.array_equal(before, np.asarray(xinit))

    def test_debug_diagnostics(self):
        """Item A5: ``debug=True`` used to raise TypeError, being unreachable from the entry point.

        The debug path uses ``jax.lax.scan`` with ``length=maxiter`` and so has no early exit; it
        returns ``maxiter + 2`` rows (the two seed steps plus every scanned iteration) and the rows
        past convergence are post-convergence noise. Assert the shape and that the eigenvalue is
        still right -- not that every row is meaningful.
        """
        rng = np.random.default_rng(20260804)
        mat = herm(40, rng, complex_=False)
        maxiter = 12
        result = ground_locg(jnp.asarray(mat), jnp.asarray(rng.normal(size=40)),
                             maxiter=maxiter, debug=True)
        assert len(result) == 5
        diagnostics = result[4]
        for key in ('x', 'y', 'r', 'theta', 'rho', 'kappa', 'sas', 'reltol', 'converged'):
            assert np.shape(diagnostics[key])[0] == maxiter + 2, f'{key} row count'
```

- [ ] **Step 7: Run the whole suite**

```bash
uv run --extra test pytest -v
```

Expected: all pass, in under ~30 s. Measured during design: the end-to-end set is about 1 s in total and the kernel sweeps about 0.01 s each; the dominant cost is JAX compilation, roughly 0.3 s per distinct shape.

- [ ] **Step 8: Update the "Testing" section of `CLAUDE.md`**

Lines 30-32 currently say there is no suite. Replace that section with:

```markdown
## Testing

```bash
uv run --extra test pytest              # whole suite
uv run --extra test pytest -v -x        # verbose, stop at first failure
```

`tests/test_ground_locg.py` covers `rqutils/ground_locg.py` only — the other six modules have no
automated tests yet, so for those still write a throwaway `uv run python -c ...` script or run the
matching `examples/` script. `tests/` also contains three Jupyter notebooks (`paulis.ipynb`,
`qprint.ipynb`, `sqd.ipynb`) used as interactive scratchpads; pytest does not collect them.

`tests/conftest.py` enables `jax_enable_x64` before any `rqutils` import — every tolerance in the
suite depends on it. Tests are organized by defect and keyed to the audit items in `docs/locg.md`;
when adding one, name the item it locks down and assert on the eigenvector residual as well as the
eigenvalue (the audit's I3 returned an exact eigenvalue with a garbage eigenvector).
```

- [ ] **Step 9: Verify the library still works outside the tests**

The dtype promotion touches the shared entry point, so confirm the dependent example still runs:

```bash
uv run --extra qiskit python examples/sqd.py 8 --num-paulis 10 --subspace-frac 0.5
```

Expected: completes with `INFO:rqutils.sqd:Found ground eigenpair in ... seconds.` and no traceback.

- [ ] **Step 10: Commit**

```bash
git add rqutils/ground_locg.py tests/test_ground_locg.py CLAUDE.md
git commit -m "Add end-to-end ground_locg tests and fix mixed-dtype crash

A float32 xinit against a float64 operator raised TypeError from while_loop:
theta entered the carry as float32 from body_iter1, whose projected matrix
takes its dtype from xinit, and returned float64 from body, whose projected
matrix comes from the matvec output. Promote xinit to the operator's dtype up
front, via eval_shape so no matvec is spent.

The end-to-end tests assert convergence (item I4) and that theta is never
below the true minimum (item I5) across shifts 0 to 1e9, real and complex.

CLAUDE.md's Testing section said there was no suite; it now documents one.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Notes for the implementer

**On tolerances.** Every numerical assertion is relative. If one fails marginally, do not loosen it without first checking whether the failure is scale-related — that is the symptom of comparing an absolute quantity, and the fix is `rel_resid`, not a bigger epsilon.

**On the notebooks.** `tests/*.ipynb` are scratchpads, not tests. Do not convert, delete, or collect them.

**On `pytest.approx`.** Use `rel=` for shifted and scaled inputs, `abs=` only where the expected value is a small exact number (such as the rank-case eigenvalues of 1.0). `pytest.approx` with neither defaults to `rel=1e-6`, far looser than anything here intends.

**Import style in the test module.** `from conftest import ...` works because pytest puts the rootdir's `tests/` directory on `sys.path` for test collection (rootdir-relative `conftest.py` insertion). If that import fails, the cause is `testpaths`/`python_files` misconfiguration from Task 1, not a missing `__init__.py` — do not add one, as it would change collection semantics.

**Do not add a CI workflow.** The repo has no `.github/workflows/`, and adding one is outside this plan's scope.
