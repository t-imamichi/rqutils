# SQD API Reorganization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Delete the deprecated MLX port, then reorganize `rqutils.sqd`'s public surface into three documented tiers with consistent names and one calling convention per entry point.

**Architecture:** Two independent changes, MLX first so the API rename never touches `examples/mlx/`. The API change publishes what is already used (an `__all__` plus autodoc directives for eight names), renames for consistency as a hard break, and collapses three diagonal builders into one keyword-dispatching `diagonals()` that mirrors `apply_h`'s existing convention. The three diagonal kernels survive as private functions — they are two genuinely different algorithms and merging their bodies would delete the abstraction rather than share it.

**Tech Stack:** Python 3.14, JAX (x64 mode), numpy, scipy, pytest, ruff, ty, Sphinx. Managed by `uv`.

**Spec:** `docs/superpowers/specs/2026-08-24-sqd-api-reorganization-design.md`

## Global Constraints

- Always `uv run python`, never bare `python`. The venv at `.venv` is uv-managed.
- Shell is fish: **quote grep globs** (`--include="*.py"`). Unquoted, fish fails with `(eval):1: no matches found` before grep runs — which looks like "no results", not "no command".
- No `timeout` command on macOS. Use the Bash tool's own timeout.
- Baseline before starting: **428 tests pass**. Never commit with a red suite.
- All three tools must stay clean: `ruff check`, `ruff format --check`, `ty check` over `rqutils/ tests/ examples/`.
- Docstrings containing a backslash **must** be raw strings (`r"""`), and must contain no literal tab characters — a tab inside an `r"""` docstring shipped once already and rendered as "J imes".
- Brace `:math:` exponents (`2^{31}`, not `2^31`).
- Docs build must end at **exactly one** warning (`rqutils.paulis.rst` not in any toctree). Anything else is new.
- `git rev-list --left-right --count origin/metal...HEAD` before amending. Never amend a pushed commit; there are currently 8 unpushed commits on `metal`.
- Mutation-test new guards in a **fresh subprocess** for `@jax.jit`-decorated functions — patching in a live session reuses the compiled kernel and both arms return bit-identical numbers.
- Every anchor-based edit must **assert the anchor exists** before substituting. A silent no-match reports a false "done".

---

### Task 1: Rescue the three non-MLX lessons from CLAUDE.md

Do this **before** deleting anything. Three lessons sit inside the MLX paragraph at `CLAUDE.md:266` but do not depend on MLX, and `examples/scaling/` prose cites CLAUDE.md as their source.

**Files:**
- Modify: `CLAUDE.md` (the MLX paragraph containing "noise floor of 3.9%")

**Interfaces:**
- Produces: a CLAUDE.md section holding the 3.9% noise floor, the fp32 eigensolve trap, and the "deleted comparison arm" lesson, which Task 3 reworded prose will cite.

- [ ] **Step 1: Locate the three lessons**

```bash
grep -n "noise floor of 3.9%" CLAUDE.md
grep -n "When you delete a comparison arm" CLAUDE.md
grep -n "f32 or f64" CLAUDE.md
```

Expected: all three on the same line (the MLX paragraph is one long line).

- [ ] **Step 2: Add the lessons to the testing/benchmarking guidance**

Insert into CLAUDE.md's "Testing" section, after the mutation-testing recipe paragraph. Use this exact text:

```markdown
**Three benchmarking lessons that outlived the code that taught them.** Each was measured on a port
that has since been deleted, but none depends on it:

- **When you delete a comparison arm, check what it was incidentally covering.** A guard was covered
  *only* as a side effect of comparing two implementations of an eigensolve; deleting one arm silently
  took the guard's only test with it. Applies to any A/B you remove.
- **Expect a ~3.9% noise floor when benchmarking a before/after on this machine.** Treat any
  difference under ~4% as unresolved, not as a result.
- **fp32 in an eigensolve is a dynamic-range trap**, not merely a precision one: a large shift
  destroys the small eigenvalue you are solving for. `examples/scaling/poc6_mixed_precision.py`
  measures it.
```

- [ ] **Step 3: Verify the lessons are findable at their new home**

```bash
grep -n "outlived the code that taught them" CLAUDE.md
grep -c "3.9%" CLAUDE.md   # expect 2 for now: old paragraph + new section
```

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md
git commit -m "Relocate three benchmarking lessons out of the MLX section

They are cited by examples/scaling/ prose and none depends on MLX --
notably 'when you delete a comparison arm, check what it was incidentally
covering', which applies to the deletion about to happen. Moved before
the MLX paragraph is removed so nothing is lost.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Delete the MLX tree and its doc

**Files:**
- Delete: `examples/mlx/` (8 files: `solver.py`, `bench.py`, `_bench_common.py`, `count_ops.py`, `check_bench.py`, `check_bench_common.py`, `check_solver_headless.py`, `check_solver_device.py`)
- Delete: `docs/mlx-metal-kernels.md`

**Interfaces:**
- Consumes: nothing.
- Produces: a tree with no MLX code, so Task 5's renames never touch an unrunnable file.

- [ ] **Step 1: Confirm nothing outside imports from the tree**

```bash
grep -rn "from solver\|import solver\|_bench_common\|examples/mlx" --include="*.py" rqutils/ tests/ examples/scaling/ 2>/dev/null
```

Expected: only `examples/scaling/_scaling_common.py` lines 4-5, which cite the `sys.path` *pattern* in prose (not an import). If any real import appears, stop and report.

- [ ] **Step 2: Confirm the doc is in no toctree**

```bash
grep -rn "mlx" docs/source/ 2>/dev/null
```

Expected: no output.

- [ ] **Step 3: Delete**

```bash
git rm -r examples/mlx/ && git rm docs/mlx-metal-kernels.md
```

- [ ] **Step 4: Verify the suite and tools are unaffected**

```bash
uv run --extra dev pytest -q
uv run --extra dev ruff check rqutils/ tests/ examples/
```

Expected: 428 passed; ruff clean. (pytest never collected `examples/mlx/`.)

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "Remove the deprecated MLX port

8 files and 2696 lines under examples/mlx/, plus the 349-line
docs/mlx-metal-kernels.md. The JAX solver measured faster even on MLX's
own GPU backend, so nothing should run it, and none of it is executable
here -- solver.py needs MLX to import and check_solver_device.py needs a
Metal device. Verified self-contained: every cross-reference pointed
within the tree, and the doc was in no toctree.

The three lessons worth keeping were relocated in the previous commit.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Strip MLX from the surviving prose

**Files:**
- Modify: `CLAUDE.md` (MLX sections: the eight-script coupling table, the parity-rule note, the deprecation narrative, the `examples/` description)
- Modify: `pyproject.toml:68` (comment naming `mlx` as a darwin-only extra)
- Modify: `examples/scaling/_scaling_common.py:4-5, 18, 203`
- Modify: `examples/scaling/poc4_real_symmetric.py:19`
- Modify: `examples/scaling/poc6_mixed_precision.py:23`
- Modify: `examples/scaling/poc8_gpu_unverified.py:28`

**Interfaces:**
- Consumes: the CLAUDE.md lessons section from Task 1.
- Produces: no `mlx`/`MLX` outside `docs/superpowers/{plans,specs}/` (dated historical records, deliberately left).

- [ ] **Step 1: List every remaining mention**

```bash
grep -rn -i "mlx" --include="*.py" --include="*.md" --include="*.toml" . 2>/dev/null | grep -v "\.venv\|docs/build\|docs/superpowers/"
```

Record the list. Each must be either reworded or deleted by the end of this task.

- [ ] **Step 2: Reword the four `examples/scaling/` citations**

Each keeps its measurement and drops the MLX attribution. Apply with anchor-asserting edits:

- `_scaling_common.py:4-5` — replace the `examples/mlx/` precedent with the directory's own convention: "Names here are unqualified by design: the directory already says `scaling`."
- `_scaling_common.py:18` — "``CLAUDE.md`` records a 3.9% noise floor on this machine" (drop "for the MLX arm").
- `_scaling_common.py:203` — "after two runs of identical code looked like a valid comparison" (drop "for the MLX arms").
- `poc4_real_symmetric.py:19` — "rather than ``solve_s``" (drop "for the MLX arms").
- `poc6_mixed_precision.py:23` — "The fp32 trap ``CLAUDE.md`` records for an eigensolve is relevant here too".
- `poc8_gpu_unverified.py:28` — "error ``CLAUDE.md`` warns about (\"a flat CPU result does not mean a change is neutral\")".

- [ ] **Step 3: Remove CLAUDE.md's MLX sections and fix the `examples/` description**

Delete the eight-script coupling table, the "change both" parity-rule paragraph, and the deprecation narrative. Rewrite the `examples/` sentence to describe only the three remaining demos (`sqd.py`, `svsim.py`, `bench.py`) and the `scaling/` POCs.

- [ ] **Step 4: Fix the pyproject comment**

```bash
uv run python - <<'EOF'
p='pyproject.toml'; s=open(p).read()
a="# mlx (darwin-only extra) and mpi4py (declared nowhere, installed manually) are optional at runtime."
assert s.count(a)==1, "ANCHOR MISSING"
open(p,'w').write(s.replace(a, "# mpi4py is imported by examples/ but declared nowhere -- install it manually if you need it."))
print("ok")
EOF
```

- [ ] **Step 5: Verify zero remaining mentions**

```bash
grep -rn -i "mlx" --include="*.py" --include="*.md" --include="*.toml" . 2>/dev/null | grep -v "\.venv\|docs/build\|docs/superpowers/"
```

Expected: no output.

- [ ] **Step 6: Verify tools and the POCs still run**

```bash
uv run --extra dev pytest -q
uv run --extra dev ruff check rqutils/ tests/ examples/ && uv run --extra dev ruff format --check rqutils/ tests/ examples/
uv run python examples/scaling/poc6_mixed_precision.py 2>&1 | tail -3
```

Expected: 428 passed, tools clean, POC runs.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "Strip MLX from the surviving prose

CLAUDE.md's MLX sections, the pyproject lint comment, and four
examples/scaling/ citations. The citations are reworded rather than
deleted: each carries a measurement that still applies to the JAX paths.
docs/superpowers/{plans,specs}/ MLX documents stay -- they are dated
records of completed work, not live documentation.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Add `diagonals()` alongside the three builders

Add the merged entry point **first**, with the old three still present, so the tests prove equivalence before anything is renamed.

**Files:**
- Modify: `rqutils/sqd.py` (add `diagonals` after `all_diagonals`)
- Test: `tests/test_sqd.py` (new `TestDiagonalsDispatch` class)

**Interfaces:**
- Consumes: `compute_diagonal(diag_signs, coeffs, nterms=None)`, `get_diagonal(zsignatures, coeffs, states, nterms=None)`, `all_diagonals(zsignatures, coeffs, group_ids, states, num_groups)`.
- Produces: `diagonals(coeffs, *, diag_signs=None, zsignatures=None, group_ids=None, states=None, num_groups=None, nterms=None) -> jax.Array`. Task 5 renames the three consumed kernels to `_diag_from_signs` / `_diag_from_z` / `_diag_all_groups`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_sqd.py`:

```python
class TestDiagonalsDispatch:
    """``diagonals`` is one entry point over three sign sources, named the way ``apply_h`` names its.

    The three predecessors (``compute_diagonal`` from cached bits, ``get_diagonal`` from Z signatures,
    ``all_diagonals`` over the flat layout) took the same ``(coeffs, ...)`` and differed only in what
    identified the sign source -- so which one to call was a fact the caller had to know rather than
    state. Equality with each predecessor is asserted **bitwise**: this is a dispatch change, not a
    numerical one, so anything else would signal a re-associated sum.
    """

    def test_diag_signs_path_matches_compute_diagonal(self):
        rng = np.random.default_rng(41)
        num_terms = 9
        diag_signs = rng.integers(0, 256, size=(24, 2), dtype=np.uint8)
        coeffs = np.zeros(num_terms)
        coeffs[:4] = rng.normal(size=4)
        want = compute_diagonal(diag_signs, coeffs, num_terms)
        got = diagonals(coeffs, diag_signs=diag_signs, nterms=num_terms)
        np.testing.assert_array_equal(np.asarray(got), np.asarray(want))

    def test_zsignatures_path_matches_get_diagonal(self):
        rng = np.random.default_rng(42)
        states = unique_states(9, 5, rng)
        hamiltonian = PauliSumXZ.from_paulisum((["ZIIII", "IZZII", "IIIZZ"], [1.0, -0.5, 0.25]))
        states_u = uniquify_states(pack_padded(states), states.shape[0])
        nterms = hamiltonian.nzterms[0]
        want = get_diagonal(hamiltonian.z[0], hamiltonian.c[0], states_u, nterms)
        got = diagonals(
            hamiltonian.c[0], zsignatures=hamiltonian.z[0], states=states_u, nterms=nterms
        )
        np.testing.assert_array_equal(np.asarray(got), np.asarray(want))

    def test_flat_path_matches_all_diagonals(self):
        rng = np.random.default_rng(43)
        strings = real_pauli_strings(5, 8, rng)
        coeffs = rng.normal(size=len(strings))
        states = unique_states(12, 5, rng)
        hamiltonian = PauliSumXZ.from_paulisum((strings, coeffs.tolist()))
        states_u = uniquify_states(pack_padded(states), states.shape[0])
        flat_z, flat_c, group_ids = hamiltonian.flat_terms
        num_groups = hamiltonian.x.shape[0]
        want = all_diagonals(flat_z, flat_c, group_ids, states_u, num_groups)
        got = diagonals(
            flat_c,
            zsignatures=flat_z,
            group_ids=group_ids,
            states=states_u,
            num_groups=num_groups,
        )
        np.testing.assert_array_equal(np.asarray(got), np.asarray(want))

    def test_invalid_combinations_raise(self):
        """Each invalid keyword set names the axis that is wrong, before any array is read."""
        coeffs = np.zeros((1, 1))
        dummy = np.zeros((1, 1), dtype=np.uint8)
        with pytest.raises(TypeError, match="exactly one of diag_signs= or zsignatures="):
            diagonals(coeffs)
        with pytest.raises(TypeError, match="exactly one of diag_signs= or zsignatures="):
            diagonals(coeffs, diag_signs=dummy, zsignatures=dummy)
        with pytest.raises(TypeError, match="zsignatures= requires states="):
            diagonals(coeffs, zsignatures=dummy)
        with pytest.raises(TypeError, match="diag_signs= is used without states="):
            diagonals(coeffs, diag_signs=dummy, states=dummy)
        with pytest.raises(TypeError, match="group_ids= requires num_groups="):
            diagonals(coeffs, zsignatures=dummy, states=dummy, group_ids=np.zeros(1, dtype=int))
```

Add `diagonals` to the `from rqutils.sqd import (...)` block in `tests/test_sqd.py`.

- [ ] **Step 2: Run the tests to verify they fail**

```bash
uv run --extra dev pytest tests/test_sqd.py -q -k DiagonalsDispatch
```

Expected: FAIL — `ImportError: cannot import name 'diagonals'`.

- [ ] **Step 3: Implement `diagonals`**

Insert into `rqutils/sqd.py` immediately after `all_diagonals`:

```python
def diagonals(
    coeffs: NDArray[np.inexact],
    *,
    diag_signs: NDArray[np.uint8] | None = None,
    zsignatures: NDArray[np.uint8] | None = None,
    group_ids: NDArray[np.integer] | None = None,
    states: StateList | None = None,
    num_groups: int | None = None,
    nterms: int | None = None,
) -> jax.Array:
    r"""Compose the diagonal of one X group, or of every X group at once.

    One entry point over three sign sources, named the way :func:`apply_h` names its inputs. Which
    source you have is a fact about your data, so it is stated rather than encoded in a choice of
    function:

    .. code-block:: python

        diagonals(coeffs, diag_signs=signs)                       # from cached packed sign bits
        diagonals(coeffs, zsignatures=z, states=states)           # recomputed for one X group
        diagonals(flat_coeffs, zsignatures=flat_z, group_ids=ids,  # every group, padding-free
                  states=states, num_groups=J)

    The first two return one group's diagonal, shape ``(num_states,)``. The third returns every
    group's, shape ``(num_groups, num_states)``, and is the form to prefer for ragged operators --
    see :attr:`~rqutils.paulis.symplectic.PauliSumXZ.flat_terms` for the layout and
    :func:`_diag_all_groups` for the measured speedups.

    Args:
        coeffs: Phase-folded coefficients. Shape ``(num_terms,)`` for one group, or the flat
            ``(sum(nzterms),)`` for the all-groups form.
        diag_signs: Precomputed packed sign bits for this group, from :func:`diag_signs`. Mutually
            exclusive with ``zsignatures``, and needs no ``states``.
        zsignatures: Packed Z signatures. Requires ``states``, since the sign bits are recomputed.
        group_ids: The X group each flat term belongs to. Selects the all-groups form and requires
            ``num_groups``.
        states: Uniquified state list. Required with ``zsignatures``, rejected with ``diag_signs``.
        num_groups: Number of X groups. **Must be static.** Required with ``group_ids``.
        nterms: Static term count for the per-group forms; see :func:`_accumulate_diagonal` for why
            it matters and when it pays. Ignored by the all-groups form, which has no padding to skip.

    Returns:
        The composed diagonal(s).

    Raises:
        TypeError: If the keywords do not form one of the three valid sets -- neither or both sign
            sources, ``zsignatures`` without ``states``, ``diag_signs`` with ``states``, or
            ``group_ids`` without ``num_groups``.
    """
    given = [
        name
        for name, value in (("diag_signs", diag_signs), ("zsignatures", zsignatures))
        if value is not None
    ]
    if len(given) != 1:
        raise TypeError(
            f"diagonals: pass exactly one of diag_signs= or zsignatures= (got {given or 'neither'})"
        )
    if diag_signs is not None:
        if states is not None:
            raise TypeError("diagonals: diag_signs= is used without states=; the bits are cached")
        return _diag_from_signs(diag_signs, coeffs, nterms)
    if states is None:
        raise TypeError("diagonals: zsignatures= requires states= to recompute the sign bits")
    if group_ids is None:
        return _diag_from_z(zsignatures, coeffs, states, nterms)
    if num_groups is None:
        raise TypeError("diagonals: group_ids= requires num_groups=, which must be static")
    return _diag_all_groups(zsignatures, coeffs, group_ids, states, num_groups)
```

Note this references the Task-5 private names. To keep this task independently green, add temporary aliases immediately above `diagonals`:

```python
# Renamed in the next commit; aliased here so this commit is independently testable.
_diag_from_signs = compute_diagonal
_diag_from_z = get_diagonal
_diag_all_groups = all_diagonals
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
uv run --extra dev pytest tests/test_sqd.py -q -k DiagonalsDispatch
uv run --extra dev pytest -q
```

Expected: 5 new tests pass; full suite 433 passed.

- [ ] **Step 5: Mutation-test the dispatch in a fresh subprocess**

For each of the three axis checks, replace its condition with `if False:`, run `pytest tests/test_sqd.py -q -k DiagonalsDispatch`, confirm a failure, then restore and `diff -q` to prove the restore took.

- [ ] **Step 6: Commit**

```bash
git add rqutils/sqd.py tests/test_sqd.py
git commit -m "Add diagonals(), one entry point over the three sign sources

The three builders took the same (coeffs, ...) and differed only in what
identified the sign source, so which to call was a fact the caller had to
know rather than state. diagonals() names the source instead, matching
apply_h's convention.

Added alongside the three, not replacing them, so the tests prove
bitwise equivalence to each predecessor before anything is renamed.
Dispatch mutation-tested in fresh subprocesses.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Rename for consistency

**Files:**
- Modify: `rqutils/sqd.py` (definitions and 46 internal references)
- Modify: `rqutils/paulis/symplectic.py` (1 reference, in `flat_terms`' docstring)
- Modify: `tests/test_sqd.py` (~65 references, including the import block)
- Modify: `tests/test_paulis_symplectic.py` (1 reference)
- Modify: `examples/scaling/` — `baseline.py`, `poc23_caching.py`, `poc6_mixed_precision.py`, `poc4_real_symmetric.py`, `poc7_sharding.py`, `poc1_searchsorted.py`, `poc5_graycode.py`, `poc8_gpu_unverified.py`
- Modify: `docs/scaling-pocs.md`, `CLAUDE.md`

**Interfaces:**
- Consumes: `diagonals()` from Task 4.
- Produces: `xsource`, `diag_signs`, `apply_xgroup`, `_diag_from_signs`, `_diag_from_z`, `_diag_all_groups`, `_run_sqd`. `uniquify_states` and `apply_h` unchanged.

| Before | After |
|---|---|
| `get_xsource` | `xsource` |
| `get_diag_signs` | `diag_signs` |
| `apply_xgrp` | `apply_xgroup` |
| `compute_diagonal` | `_diag_from_signs` |
| `get_diagonal` | `_diag_from_z` |
| `all_diagonals` | `_diag_all_groups` |
| `run_sqd` | `_run_sqd` |

- [ ] **Step 1: Record the pre-rename baseline**

```bash
uv run --extra dev pytest -q 2>&1 | tail -2   # expect 433 passed
```

- [ ] **Step 2: Rename, one symbol at a time, longest-first**

Longest-first matters: renaming `get_diagonal` before `get_diag_signs` would corrupt neither, but renaming `diagonals` before `all_diagonals` would. Use word-boundary substitution and assert a nonzero count per file:

```bash
uv run python - <<'EOF'
import re, pathlib
RENAMES = [  # order matters: no name may be a substring of a later target
    ("all_diagonals", "_diag_all_groups"),
    ("compute_diagonal", "_diag_from_signs"),
    ("get_diag_signs", "diag_signs"),
    ("get_diagonal", "_diag_from_z"),
    ("get_xsource", "xsource"),
    ("apply_xgrp", "apply_xgroup"),
    ("run_sqd", "_run_sqd"),
]
FILES = ["rqutils/sqd.py", "rqutils/paulis/symplectic.py", "tests/test_sqd.py",
         "tests/test_paulis_symplectic.py", "CLAUDE.md", "docs/scaling-pocs.md"]
FILES += [str(p) for p in pathlib.Path("examples/scaling").glob("*.py")]
total = 0
for f in FILES:
    s = orig = pathlib.Path(f).read_text()
    for old, new in RENAMES:
        s = re.sub(rf"(?<![\w.]){re.escape(old)}\b", new, s)
    if s != orig:
        pathlib.Path(f).write_text(s)
        n = sum(len(re.findall(rf"\b{re.escape(new)}\b", s)) for _, new in RENAMES)
        print(f"  {f}: rewritten")
        total += 1
print(f"{total} files rewritten")
EOF
```

The `(?<![\w.])` guard prevents renaming an attribute access like `m.get_xsource` into a broken form and prevents matching inside a longer identifier.

- [ ] **Step 3: Remove Task 4's temporary aliases**

```bash
uv run python - <<'EOF'
p='rqutils/sqd.py'; s=open(p).read()
a = """# Renamed in the next commit; aliased here so this commit is independently testable.
_diag_from_signs = _diag_from_signs
_diag_from_z = _diag_from_z
_diag_all_groups = _diag_all_groups
"""
assert s.count(a)==1, "ANCHOR MISSING -- the alias block may have been renamed too; inspect it"
open(p,'w').write(s.replace(a, ""))
print("aliases removed")
EOF
```

If the anchor does not match, the rename rewrote the alias block into self-assignments; find and delete it by inspection.

- [ ] **Step 4: Verify no old name survives**

```bash
grep -rn "get_xsource\|get_diag_signs\|apply_xgrp\|compute_diagonal\|get_diagonal\|all_diagonals\|\brun_sqd\b" --include="*.py" --include="*.md" rqutils/ tests/ examples/ CLAUDE.md docs/scaling-pocs.md 2>/dev/null
```

Expected: no output. `docs/rqutils-requests.md` is deliberately excluded — it is a dated record of a request written against 0.2.0 and its banner already says its line numbers have drifted.

- [ ] **Step 5: Verify the suite, tools, and a POC**

```bash
uv run --extra dev ruff check rqutils/ tests/ examples/ --fix
uv run --extra dev ruff format rqutils/ tests/ examples/
uv run --extra dev pytest -q                                  # expect 433 passed
uv run --extra dev ty check rqutils/ tests/ examples/
uv run python examples/scaling/baseline.py 2>&1 | tail -3
```

- [ ] **Step 6: Verify sharding still passes on 4 devices**

```bash
XLA_FLAGS=--xla_force_host_platform_device_count=4 uv run python examples/scaling/poc7_sharding.py 2>&1 | tail -8
```

Expected: all six cache levels agree, worst deviation ~4.4e-16.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "Rename sqd's kernel functions for consistency (breaking)

get_xsource -> xsource, get_diag_signs -> diag_signs, apply_xgrp ->
apply_xgroup. The three diagonal builders become private
(_diag_from_signs/_diag_from_z/_diag_all_groups) behind diagonals(), and
run_sqd -> _run_sqd since it is sqd's jitted body rather than an API.

Hard break, no aliases: an alias layer would recreate the two-surfaces
problem being removed from apply_h. uniquify_states and apply_h keep
their names -- both are imported downstream and neither carried a get_
prefix.

docs/rqutils-requests.md keeps the old names deliberately: it is a dated
record of a request against 0.2.0, and its banner already says its line
numbers have drifted.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Delete `apply_h`'s deprecated positional form

**Files:**
- Modify: `rqutils/sqd.py` (`apply_h`'s legacy branch, ~60 lines)
- Modify: `tests/test_sqd.py` (delete two tests, trim a third)

**Interfaces:**
- Consumes: nothing new.
- Produces: `apply_h(vec, *, xsources=..., xsignatures=..., zsignatures=..., diag_signs=..., diagonals=..., coeffs=..., states=..., nterms=...)`. The `scanned`/`cache_level` parameters are gone.

- [ ] **Step 1: Delete the two shim tests and trim the third**

Delete `test_legacy_tuple_form_still_works_but_warns` and `test_mixing_the_two_forms_raises` from `tests/test_sqd.py`. In `test_mispairing_is_now_unconstructible`, delete the `with pytest.warns(DeprecationWarning):` block and the `assert np.abs(mispaired - correct).max() > 0.1` that follows it, keeping both `TypeError` arms. Update its docstring: the legacy path no longer exists to demonstrate the old defect, so state the measured 0.44 error as history rather than as a live comparison.

- [ ] **Step 2: Run to confirm the suite is green before touching the source**

```bash
uv run --extra dev pytest tests/test_sqd.py -q
```

Expected: 431 passed (433 minus the 2 deleted).

- [ ] **Step 3: Delete the legacy branch from `apply_h`**

Remove the `scanned` and `cache_level` parameters from the signature, and the entire `if scanned is not None or cache_level is not None:` block (the mixing check, the `warnings.warn`, and both `_apply_h_resolved` returns). Make every array keyword-only. Drop the now-unused `import warnings` if nothing else uses it:

```bash
grep -n "warnings" rqutils/sqd.py
```

- [ ] **Step 4: Update `apply_h`'s docstring**

Delete the `scanned` and `cache_level` `Args:` entries and the paragraph explaining the deprecation. Keep the "why this replaces the positional form" rationale — it is the justification for the named API and still true — but state it in the past tense.

- [ ] **Step 5: Verify**

```bash
uv run --extra dev pytest -q                          # expect 431 passed
uv run --extra dev ruff check rqutils/ tests/ examples/
uv run --extra dev ty check rqutils/ tests/ examples/
grep -rn "cache_level=(1, 1))" --include="*.py" tests/ examples/   # expect no positional apply_h calls
```

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "Delete apply_h's deprecated positional form (breaking)

The named-keyword form is about to become a documented stable entry
point, and shipping it with a deprecated alternate path on day one
defeats the purpose. Removes ~60 lines: the mixing checks, the
DeprecationWarning, and both legacy returns.

_apply_h_resolved stays private and _run_sqd keeps calling it directly --
it holds an assembled scanned tuple and must bind statics via
functools.partial for ground_locg's positional splat.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: Publish the tiers — `__all__`, directives, CHANGELOG

**Files:**
- Modify: `rqutils/sqd.py` (add `__all__`; restructure the module docstring's API section)
- Create: `CHANGELOG.md`
- Modify: `CLAUDE.md` (architecture section: the tier distinction and new names)
- Test: `tests/test_sqd.py` (new `TestPublicSurface` class)

**Interfaces:**
- Consumes: every rename from Tasks 4-6.
- Produces: `rqutils.sqd.__all__ == ["sqd", "hproj", "uniquify_states", "xsource", "diag_signs", "apply_h", "diagonals", "apply_xgroup"]`.

- [ ] **Step 1: Write the failing test**

```python
class TestPublicSurface:
    """``__all__`` is the promise, and it must match what the module actually exports.

    Without it, ``from rqutils.sqd import *`` pulled in eight third-party names -- ``coo_array``,
    ``csr_array``, ``SparsePauliOp``, ``PartitionSpec``, ``get_abstract_mesh``, ``Callable``,
    ``Sequence``, ``Number`` -- because a module with no ``__all__`` exports every global that does
    not start with an underscore. ``rqutils/__init__.py`` already declares one; this module did not.
    """

    def test_all_names_exist(self):
        import rqutils.sqd as module

        for name in module.__all__:
            assert hasattr(module, name), f"__all__ names {name}, which does not exist"

    def test_all_matches_the_documented_tiers(self):
        import rqutils.sqd as module

        assert module.__all__ == [
            "sqd",
            "hproj",
            "uniquify_states",
            "xsource",
            "diag_signs",
            "apply_h",
            "diagonals",
            "apply_xgroup",
        ]

    def test_third_party_names_are_not_exported(self):
        """The leak this fixes: scipy and qiskit names were reachable as rqutils.sqd.X."""
        import rqutils.sqd as module

        for leaked in ("coo_array", "csr_array", "SparsePauliOp", "PartitionSpec", "Callable"):
            assert leaked not in module.__all__, f"{leaked} must not be part of the public surface"

    def test_every_public_name_is_in_all(self):
        """A public name absent from __all__ is either an oversight or should be private."""
        import inspect

        import rqutils.sqd as module

        defined_here = {
            name
            for name, obj in vars(module).items()
            if not name.startswith("_")
            and (inspect.isfunction(obj) or hasattr(obj, "__wrapped__"))
            and getattr(obj, "__module__", None) == "rqutils.sqd"
        }
        assert defined_here == set(module.__all__), (
            f"mismatch: {defined_here ^ set(module.__all__)}"
        )
```

- [ ] **Step 2: Run to verify it fails**

```bash
uv run --extra dev pytest tests/test_sqd.py -q -k PublicSurface
```

Expected: FAIL — `AttributeError: module 'rqutils.sqd' has no attribute '__all__'`.

- [ ] **Step 3: Add `__all__`**

Immediately after the module docstring in `rqutils/sqd.py`, before the imports:

```python
__all__ = [
    # Solvers.
    "sqd",
    "hproj",
    # Kernel API: enough to build a matvec by hand. Verified self-sufficient -- these four plus
    # PauliSumXZ compose a full Hv with no diagonal builder needed, since diag_signs feeds apply_h.
    "uniquify_states",
    "xsource",
    "diag_signs",
    "apply_h",
    # Lower-level building blocks: documented, but with no stability promise.
    "diagonals",
    "apply_xgroup",
]
```

- [ ] **Step 4: Run to verify it passes**

```bash
uv run --extra dev pytest tests/test_sqd.py -q -k PublicSurface
```

Expected: 4 passed. If `test_every_public_name_is_in_all` fails, the diff names a function that is public but unlisted — decide per the spec's tiers whether to list it or make it private.

- [ ] **Step 5: Restructure the module docstring's API section**

Replace the current `SQD API` section (`.. autofunction:: sqd` / `hproj` / `all_diagonals`) with:

```rst
SQD API
=======

Solvers
-------

.. autofunction:: sqd
.. autofunction:: hproj

Kernel API
----------

Enough to assemble a matrix-vector product by hand, which is what a caller with its own solver or
its own cost function needs. These four plus :class:`~rqutils.paulis.symplectic.PauliSumXZ` are
self-sufficient: ``diag_signs`` feeds ``apply_h`` directly, so no diagonal builder is required.

.. autofunction:: uniquify_states
.. autofunction:: xsource
.. autofunction:: diag_signs
.. autofunction:: apply_h

Lower-level building blocks
---------------------------

Documented because the in-tree POCs use them and because they explain how the kernel API is put
together, but **without a stability promise** -- prefer the kernel API above where it suffices.

.. autofunction:: diagonals
.. autofunction:: apply_xgroup

States are packed with :meth:`~rqutils.paulis.symplectic.PauliSumXZ.pack_states`, which inserts the
pad bit that aligns them with the Hamiltonian's signatures, and recovered with
:meth:`~rqutils.paulis.symplectic.PauliSumXZ.unpack_states`.
```

- [ ] **Step 6: Create `CHANGELOG.md`**

```markdown
# Changelog

All notable changes to `rqutils` are recorded here. This file starts at the 0.2.0 -> unreleased
boundary; for earlier history see the git log.

## Unreleased

### Removed

- **The deprecated MLX port.** `examples/mlx/` (8 files) and `docs/mlx-metal-kernels.md` are gone.
  The JAX solver measured faster even on MLX's own GPU backend, and none of the port was executable
  without a Metal device. Three benchmarking lessons it had taught were relocated into `CLAUDE.md`.

### Changed (breaking)

- **`rqutils.sqd` gained an `__all__`**, so `from rqutils.sqd import *` no longer re-exports
  `coo_array`, `csr_array`, `SparsePauliOp`, `PartitionSpec`, `get_abstract_mesh`, `Callable`,
  `Sequence` or `Number`. Import those from their own packages.
- **Renames.** No aliases were kept: an alias layer would recreate the dual-surface problem removed
  from `apply_h` in the same release.

  | Before | After |
  |---|---|
  | `get_xsource` | `xsource` |
  | `get_diag_signs` | `diag_signs` |
  | `apply_xgrp` | `apply_xgroup` |
  | `compute_diagonal` | `diagonals(coeffs, diag_signs=...)` |
  | `get_diagonal` | `diagonals(coeffs, zsignatures=..., states=...)` |
  | `all_diagonals` | `diagonals(coeffs, zsignatures=..., group_ids=..., states=..., num_groups=...)` |
  | `run_sqd` | `_run_sqd` (private; it is `sqd`'s jitted body) |

  `uniquify_states` and `apply_h` are unchanged.
- **`apply_h` no longer accepts the positional `(scanned, cache_level)` form.** Pass the arrays by
  name. The deprecated form silently computed a different operator when mispaired -- 0.44 measured
  error, no exception -- which is why it was deprecated; it is now removed.

### Added

- **A documented public surface**, in three tiers: solvers (`sqd`, `hproj`), a stable kernel API
  (`uniquify_states`, `xsource`, `diag_signs`, `apply_h`), and lower-level building blocks
  (`diagonals`, `apply_xgroup`) that are documented without a stability promise. Eight names now
  carry autodoc directives where three did.
- **`diagonals()`**, one entry point over the three sign sources, replacing the three separate
  builders. Bitwise identical to each predecessor.
- **`PauliSumXZ.flat_terms`** and the segment-scatter diagonal path behind
  `diagonals(..., group_ids=...)`: for ragged operators this is 1.4x to ~150x faster than the
  rectangular scan, bit-identical, and it compiles faster too.
```

- [ ] **Step 7: Update CLAUDE.md's architecture section**

Add the tier distinction and the new names to the `sqd.py` paragraph. State that `diagonals()` dispatches on the sign source and that the three kernels are private.

- [ ] **Step 8: Verify the docs build and every directive renders**

```bash
cd docs && uv run --extra docs make clean >/dev/null 2>&1 && uv run --extra docs make html 2>&1 | grep -iE "warning"
```

Expected: exactly one warning (`rqutils.paulis.rst` not in any toctree).

Then, from the repo root, confirm each new name produced an anchor:

```bash
for n in sqd hproj uniquify_states xsource diag_signs apply_h diagonals apply_xgroup; do
  grep -q "id=\"rqutils.sqd.$n\"" docs/build/html/apidoc/rqutils.sqd.html \
    && echo "  ok   $n" || echo "  MISSING anchor: $n"
done
```

Expected: all eight `ok`. Reading the source is not proof — grep the built HTML.

- [ ] **Step 9: Verify no cross-reference dangles after the renames**

```bash
grep -rn ":func:\`\(get_xsource\|get_diag_signs\|apply_xgrp\|compute_diagonal\|get_diagonal\|all_diagonals\|run_sqd\)\`" rqutils/ 2>/dev/null
```

Expected: no output.

- [ ] **Step 10: Run the full verification set**

```bash
uv run --extra dev pytest -q                                    # expect 435 passed
uv run --extra dev ruff check rqutils/ tests/ examples/ && uv run --extra dev ruff format --check rqutils/ tests/ examples/
uv run --extra dev ty check rqutils/ tests/ examples/
uv run python -c "
import rqutils.sqd, rqutils.ground_locg, rqutils.svsim, rqutils.qprint
import rqutils.math as rm, rqutils.paulis.general as pg, rqutils.paulis.symplectic as ps
bad = [(m.__name__, n) for m in (rqutils.sqd, rqutils.ground_locg, rqutils.svsim, rqutils.qprint, rm, pg, ps)
       for n, o in list(vars(m).items()) + [('<module>', m)]
       if isinstance(getattr(o, '__doc__', None), str) and any(c in o.__doc__ for c in '\x07\x08\x0b\x0c\t')]
print('control-char/tab docstrings:', bad)"
XLA_FLAGS=--xla_force_host_platform_device_count=4 uv run python examples/scaling/poc7_sharding.py 2>&1 | tail -6
```

Expected: green suite, clean tools, no control-char or tab docstrings, all six cache levels agreeing on 4 devices.

- [ ] **Step 11: Commit**

```bash
git add -A
git commit -m "Publish the sqd API in three documented tiers, and start a CHANGELOG

Adds __all__ (which also stops eight third-party names leaking as
rqutils.sqd.X), restructures the module docstring's API section into
solvers / kernel API / building blocks, and gives all eight published
names an autodoc directive where three had one.

The kernel API tier is the four names an external caller needs to build a
matvec by hand -- verified self-sufficient, since diag_signs feeds
apply_h and no diagonal builder is required. The building-block tier is
documented deliberately without a stability promise.

CHANGELOG.md is new. CLAUDE.md had noted its absence twice, once saying a
change belonged in it; with renames and a removed calling convention
landing together, downstream needs one place to read what broke.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage.** Every spec section maps to a task: MLX deletion → Tasks 2-3 (with Task 1 as the prerequisite the spec's self-review added); `__all__` and tiers → Task 7; renames → Task 5; `diagonals()` → Task 4; `apply_h` shim → Task 6; docs and CHANGELOG → Task 7. The spec's verification list is distributed across the tasks that can invalidate each item, with the full set re-run in Task 7 Step 10.

**One spec correction.** The spec claims `run_sqd` has "zero external callers". It has zero external *call sites*, but `tests/test_sqd.py:50` imports it for `_cache_size()` assertions, and `baseline.py`/`poc7_sharding.py` name it in prose. Task 5 handles all three; the risk note in the spec stands but is narrower than written.

**Two spec omissions, now covered.** The spec's call-site list missed `poc1_searchsorted.py` (9 `get_xsource` + 3 `apply_xgrp`), `poc5_graycode.py` (3), and `poc8_gpu_unverified.py` (4). Task 5's file list includes them, and its Step 4 grep would have caught any further misses.

**Placeholder scan.** No TBD/TODO. Every code step carries the actual code. Task 3's rewordings are specified as exact target text per line rather than "reword appropriately".

**Type consistency.** `diagonals`' signature in Task 4 matches its `__all__` entry and directive in Task 7. The private kernel names `_diag_from_signs`/`_diag_from_z`/`_diag_all_groups` are introduced in Task 4's interface block, created by Task 5's rename table, and referenced in Task 4's docstring — Task 4's temporary alias block bridges the gap so it is independently green, and Task 5 Step 3 removes it with a fallback if the rename rewrote it.

**Test-count arithmetic.** 428 baseline → 433 after Task 4 (+5) → 431 after Task 6 (−2) → 435 after Task 7 (+4). Each task states its expected count.
