# First pytest suite: `rqutils.ground_locg`

Design date: 2026-08-04. Branch `metal`.

## Goal

Establish a pytest suite in `tests/`, covering `rqutils/ground_locg.py`. The repository has no
automated tests today — `tests/` holds three Jupyter scratchpads (`paulis.ipynb`, `qprint.ipynb`,
`sqd.ipynb`) and `CLAUDE.md` states outright that there is no suite to run.

`ground_locg.py` is the target because it was just replaced by a numerically hardened rewrite. The
audit that motivated that rewrite, `docs/locg.md`, enumerates twelve specific ways the previous
implementation was **silently wrong** — returning a plausible finite number rather than raising —
each with a reproducing input and a measured old failure. Those reproducers are the backbone of
this suite.

Scope is deliberately one module. Other modules get their own suites later, following the pattern
established here.

## Approach

Tests are organized **by defect**, keyed to `docs/locg.md` items, rather than by public API surface
alone. Each test's docstring names its audit item and the measured old behavior.

The rationale is traceability: a future failure names the defect that regressed, and the reproducing
inputs cannot be quietly "simplified" into a random-matrix parametrization that no longer probes
what it was written for. Tests are still grouped into classes per function for navigability.

`docs/locg.md` closes with a methodological warning that shapes this design:

> Aggregate random sampling does not exercise data-dependent branches evenly. [...] Both of my
> safety nets missed a sign error that a two-line targeted test caught at 4000/4000 failures.
> Kernel changes here need per-branch tests, not just volume.

So targeted per-branch tests are the backbone and randomized sweeps are a supplement, not the
reverse. Randomness uses seeded `numpy.random.default_rng` for reproducibility; no `hypothesis`
dependency is added.

### Runtime budget

Under ~30 seconds total, so the suite is runnable on every edit. Kernel tests on 2x2/3x3 matrices
are microseconds. End-to-end solves are capped at n=200 (n=400 measured at 0.3 s); the audit's
n=3000 cases existed for timing, not correctness, and every correctness failure it documents
reproduces at small n.

## Scaffolding

### `pyproject.toml`

Add a `test` extra, matching the repository's existing convention — `mpl`, `qutip`, `qiskit`, and
`docs` are all extras, and `CLAUDE.md` documents the per-invocation `--extra` pattern:

```toml
[project.optional-dependencies]
test = ["pytest"]
```

Add a pytest configuration block:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
```

`python_files` is explicit so that collection never considers the three notebooks or anything else
in `tests/` that is not a `test_*.py` module.

Invocation: `uv run --extra test pytest`.

### `tests/conftest.py`

Calls `jax.config.update('jax_enable_x64', True)` at import time, before any test module imports
`rqutils`. This is load-bearing rather than incidental: without x64, JAX silently produces float32
and complex64, and every tolerance in this suite assumes float64. `CLAUDE.md` records this as the
expected pattern for the library generally.

It also holds four shared helpers:

| Helper | Purpose |
|---|---|
| `herm(n, rng, complex_=True)` | Random Hermitian matrix from a seeded generator |
| `symmetrize(mat)` | Mirror the lower triangle over the diagonal |
| `lowest(mat)` | `numpy.linalg.eigvalsh` reference, on the symmetrized matrix |
| `rel_resid(mat, val, vec)` | `norm(mat @ vec - val * vec) / max(abs(mat))` |

Two of these encode mistakes that are easy to repeat, and both were hit while validating this
design:

- **`symmetrize`.** `eigenpair_2x2` and `eigenpair_3x3` document that they read only the diagonal
  and the lower triangle. Comparing their output against `eigvalsh` of an unsymmetrized input
  compares against a *different matrix* and produces a spurious failure.
- **`rel_resid`.** Tolerances must be relative to `max|A|`. The 1e9-shifted 2x2 case has absolute
  residual 6e-8 but relative residual 6e-17. An absolute tolerance either fails spuriously here or
  is so loose elsewhere that it asserts nothing.

### `CLAUDE.md`

Its "Testing" section currently reads "**There is no pytest suite.**" and "Don't claim tests pass —
there are none to run." Both become false when this lands. Update to document
`uv run --extra test pytest`, note that the suite covers `ground_locg` only, and keep the
notebooks-as-scratchpads note.

## Test inventory

All in `tests/test_ground_locg.py`.

### `TestEigenpair2x2`

The shipped predecessor returned **2353 `NaN`s in 40000 random inputs**.

| Test | Input | Audit item | Old behavior |
|---|---|---|---|
| `test_diagonal_input` | `diag(1, 5)` | I1 / `:431` | `NaN` (0/0) |
| `test_identity` | `eye(2)` | I1 / `:431` | `NaN` |
| `test_large_shift` | `[[1, .5], [.5, 2]] + 1e9·I` | I1 | rel. error 5.8e-1 |
| `test_extreme_scale` | same, scaled 1e-160 and 1e160 | I2 | 4.9e-2 error; `NaN` |
| `test_delta_positive` | 2000 seeded matrices, `δ ≥ 0` forced | — | (passes) |
| `test_delta_negative` | 2000 seeded matrices, `δ < 0` forced | — | 4000/4000 failures |

The last two are separate parametrized cases, not one sweep. `δ = (d₀ - d₁)/2` selects which row of
the singular shifted matrix yields the null vector; the audit's own sign error in that branch passed
every aggregate test it had, because those tests happened to generate only one sign. Splitting the
branches is the whole point. Measured on the current implementation: worst relative eigenvalue error
1.19e-15 (δ>0) and 1.03e-15 (δ<0).

End-to-end convergence also hides errors here — `eigenpair_2x2` is called exactly once, in
`body_iter1`, so a wrong eigenvector is quietly repaired by later iterations.

### `TestEigenpair3x3`

The predecessor returned **4066 `NaN`s in 20000 random inputs** and a wrong eigenvector 1148 times
in 5000.

| Test | Input | Audit item | Old behavior |
|---|---|---|---|
| `test_rank_deficient_column_pair` | `diag(5, 6, 1)` | I3 | eigenvector residual **0.67** |
| `test_identity` | `eye(3)` | I3 rank 0 | garbage direction |
| `test_degenerate_lowest` | `diag(1, 1, 7)` | I3 rank 1 | garbage direction |
| `test_large_shift` | `diag(1, 2, 3) + 1e9·I` | I1 | `NaN` (negative radicand) |
| `test_extreme_scale` | scaled 1e-160 and 1e150 | I2 | 7.8e-1 error; `NaN` |
| `test_random_sweep` | 2000 seeded matrices | I1-I3 | 4066/20000 `NaN` |

`test_rank_deficient_column_pair` is the sharpest of these: `diag(5, 6, 1)` is an innocuous input on
which the old kernel returned an *exact eigenvalue* alongside an eigenvector with residual 0.67. A
test asserting only on the eigenvalue would have passed. Every kernel test asserts on both.

The sweep asserts no `NaN` and relative eigenvalue error < 1e-13. Measured on the current
implementation: 0 `NaN`, worst relative error 1.60e-15, worst relative residual 1.10e-15.

Both `test_extreme_scale` cases assert finiteness and relative eigenvalue error < 1e-13. Measured:
exact at 1e-160 and 1e150 for the 3x3, and 7.8e-17 at 1e160 for the 2x2 — the tolerance is relative
so it holds across all of them.

### `TestProjectOut`

One test for the postcondition its docstring states (I6): `_project_out` returns **either exactly
zero** (for a vector inside the basis span) **or a vector of norm ≥ 0.99** — notably *not* a unit
vector. Callers feeding it to a standard Rayleigh-Ritz step must renormalize; at shift 1e9 a `|p|` of
0.999 displaced θ by 2e6, below the true minimum.

This is a private function. Testing it is justified because the ≥ 0.99 contract is exactly the kind
of surprising invariant that a well-meaning future edit "tidies" into a normalization, silently
removing the zero-return guard that I7 depends on.

### `TestGroundLocg`

Parametrized over shifts `{0, +1e3, -1e3, 1e6, 1e9}` × `{real, complex}`, asserting three things
per case:

1. Eigenvalue matches `eigvalsh` relative to `max|A|`.
2. `converged is True` — I4. The old `reltol` used `norm(Ax) - theta`, a cancellation measured going
   *negative*, which made the test unsatisfiable: the solver never converged and always burned
   `maxiter`. Fixing that sign alone was a 24-33x iteration reduction.
3. **θ is not below the true minimum** — I5, the audit's most serious finding. Loss of x/y
   orthogonality made the standard Rayleigh-Ritz step return a θ beneath the actual ground state
   energy (observed: 6.0e8 against a true minimum of 1.0e9), which is impossible for a genuine
   Rayleigh quotient. A caller checking only "is it finite" would accept it.

Assertion 3 is the one worth stating explicitly even though assertion 1 nominally implies it. It
encodes *why* the failure mattered, and it is meaningful at looser tolerances where assertion 1
would pass.

Interface and edge cases:

| Test | Asserts | Audit item |
|---|---|---|
| `test_maxiter_too_small` | `converged is False` at `maxiter=5` | A1 |
| `test_maxiter_zero` | returns Rayleigh quotient of normalized `xinit`, not `0.0` | A2 |
| `test_zero_xinit` | finite result, not `NaN` | A4 |
| `test_integer_xinit` | one-hot convenience path solves correctly | — |
| `test_xinit_scale_invariance` | `xinit * 1e8` gives an identical result | — |
| `test_xinit_not_mutated` | caller's array untouched | — |
| `test_debug_diagnostics` | `debug=True` works from the public entry point | A5 |

`test_debug_diagnostics` asserts `maxiter + 2` diagnostic rows. The debug path uses `jax.lax.scan`
with `length=maxiter` and therefore has **no early exit**; rows past convergence are
post-convergence noise. The test asserts the shape and that the eigenvalue is still correct, not
that every row is meaningful.

## Mixed-dtype fix

A bug found while validating this design, not present in `docs/locg.md`.

A float32 `xinit` against a float64 operator raises:

```
TypeError: while_loop body function carry input and carry output must have equal types
  The input carry component state[2] has type float32[] but the corresponding output carry
  component has type float64[]
```

`state[2]` is θ. It enters as float32 from `body_iter1`, whose projected matrix takes its dtype from
`xinit`, and returns as float64 from `body`, whose projected matrix is built from the matvec output.
Only a *lower*-precision `xinit` fails; float32-operator with float64-`xinit` works.

This is adjacent to audit item A3, which concerned the same configuration: the rewrite fixed the
*tolerance* derivation for it but left the state dtype inconsistent.

### The change

In `_ground_locg_callable`, before `body_iter0`:

```python
# The projected matrix inherits xinit's dtype at the seed step but the operator's inside the
# loop, so a lower-precision xinit makes while_loop's carry types disagree. Promote up front.
work_dtype = jnp.result_type(xinit.dtype,
                             jax.eval_shape(lambda v: matvec(v, *args), xinit).dtype)
xinit = xinit.astype(work_dtype)
```

`jax.eval_shape` obtains the operator dtype without spending a matrix-vector product. The promotion
must precede `body_iter0`, since that is where the projected matrix's dtype originates. `astype` on
a matching dtype is a no-op, so the common path is unaffected.

This subsumes the existing `jnp.result_type` call in the `tol is None` branch, which computes the
same promotion; that becomes `jnp.finfo(work_dtype).eps`.

Validated: float32-`xinit` / float64-operator goes from `TypeError` to a correct solve, eigenvalue
error 2.1e-14, in the same 65 iterations as the all-float64 case. float32/float32 is unchanged at
21 iterations.

### Tests

`TestDtypes`, parametrized over `(operator, xinit)` in `{(f64, f64), (f32, f32), (f64, f32),
(c128, f32)}`: each solves, and the mixed cases match the all-float64 result to float64 tolerance —
i.e. a low-precision initial guess does not degrade the answer, which was A3's concern.

## Out of scope

Recorded so the boundary is explicit.

- **Sharded / multi-device execution.** `docs/locg.md` lists this as unverified, and the
  re-orthogonalization added by the rewrite introduces two inner products per iteration whose
  sharding behavior is untested. Exercising it needs a multi-device mesh.
- **`sqd.py` integration.** Deferred to a later suite. `examples/sqd.py` already exercises that path
  end to end.
- **Tight spectral clusters** below ~1e-10 relative gap. `docs/locg.md` documents the residual
  degradation there as expected gap-dependent LOBPCG conditioning rather than a defect. Asserting
  on it would encode a known-flaky boundary.
- **Other `rqutils` modules.** `paulis/general.py` in particular has invariants worth testing
  (`λ₀ = sqrt(2/n)·I`, basis-index ordering) but needs its own design pass.

## Verification

The suite passes with `uv run --extra test pytest`, and each defect-keyed test is confirmed to
actually exercise its target rather than pass trivially. For the mixed-dtype fix specifically: the
new `TestDtypes` case must be shown failing with `TypeError` before the fix is applied and passing
after, so the test is known to have teeth.
