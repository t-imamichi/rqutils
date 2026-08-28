# Typing: what was fixed, what remains, and what is not worth doing

Measured 2026-08-29. Companion to the `[tool.ty.rules]` block in `pyproject.toml`, which holds the
short version. Numbers come from `ty check -c 'rules.X="error"'` per rule, run over
`rqutils/ tests/ examples/` with all four extras installed.

## 1. State

Nine global ignores at the start of this work, **four** now. Every removal was earned by fixing the
sites, not by deleting the line, and each removed rule was then verified to catch a deliberately
planted error.

| rule | status | findings when enabled |
| --- | --- | --- |
| `unresolved-import` | **removed** | 1 → per-line ignore on `mpi4py`, deliberately undeclared |
| `unsupported-operator` | **removed** | 5 → 0 (one real defect, one per-line ignore) |
| `not-iterable` | **removed** | 7 → 0 (one real defect) |
| `invalid-assignment` | **removed** | 19 → 1 → per-line ignore on a read-only-property test |
| `not-subscriptable` | **removed** | 0 — dead config |
| `unresolved-attribute` | kept | 12 total, 8 in `rqutils/` |
| `no-matching-overload` | kept | 27 total, 6 in `rqutils/` |
| `invalid-argument-type` | kept | 47 total, 16 in `rqutils/` |
| `invalid-return-type` | kept | 5, all in `rqutils/` |

**Four real defects were hiding behind those ignores**, which is the argument against blanket
suppression:

1. `conftest.gate_unitary(..., angle=None)` reached `angle / 2.0` for any rotation gate and raised
   `TypeError: unsupported operand type(s) for /: 'NoneType' and 'float'`, naming neither the gate nor
   the argument.
2. `QPrintBase._process` was annotated `-> tuple[..., list[list[Term]]]` while returning a flat
   `list[Term]`. That is why `term.index` resolved to `list.index`, the bound method.
3. `matrix_exp` / `matrix_angle` declared `-> NDArray` while forwarding `with_diagonals`, so
   `matrix_exp(m, with_diagonals=True)` returned a 2-tuple against a declared array.
4. `svsim.do_svsim` and `paulis.general.labels` each rebound a parameter to a different type
   (`initial_state` int→vector, `symbol` scalar→tuple), which made the declared type wrong for the rest
   of the body.

## 2. The pattern that did most of the work

`@overload` on `Literal[True]` / `Literal[False]` for a **runtime flag that picks the return shape**.
Three applications so far:

| function | flag | before | after |
| --- | --- | --- | --- |
| `sqd` | `return_eigvec` | `float \| tuple[...]` | precise per arm |
| `ground_locg` | `debug` | 4-tuple or 5-tuple union | precise per arm |
| `matrix_exp`, `matrix_angle` | `with_diagonals` | wrongly `NDArray` | precise per arm |

Three rules to reuse it correctly, each learned here:

- **Include a `bool` fallback overload.** A runtime-computed flag (`debug=bool(n > 0)`) is not a
  `Literal`, and without the fallback that call stops type-checking. It must return the union, so such
  callers still narrow.
- **Match the real parameter kind.** `sqd`'s flags are keyword-only, `ground_locg`'s are
  positional-or-keyword — the latter needs an extra overload repeating the full positional order,
  because `examples/scaling/poc4` and `poc6` pass `maxiter` positionally.
- **Overloads are annotation-only, so prove it.** Each application was A/B'd against the pre-change
  revision: 16 `ground_locg` behaviors across nine calling conventions, 12 `matrix_*` outputs across
  three symmetry modes, all identical, plus `inspect.signature` unchanged so sphinx still documents the
  implementation.

## 3. What remains, and why most of it is not actionable

35 library-side findings across the four kept rules. They are **not** one population.

**Class A — `jax.jit` argument conversion (13 of 35). Not fixable locally.** A jitted function's
parameters are converted before the body runs, so a parameter annotated `int` genuinely holds an Array
with a `.dtype` by the time the body reads it. `ground_locg:739` reading `xinit.dtype` on
`Array | int`, and `svsim:143` reading `initial_state.shape` on `NDArray | int`, are both correct code
against an annotation that describes the *caller's* view. Writing the annotation as `Array` instead
would be a lie to callers, who really do pass an `int`. **Do not "fix" these** — they need jax to ship
stubs, or a `jax.jit`-aware checker.

**Class B — the `npmod` convention meeting numpy's stubs.** `ArrayLike` has no `.shape` and no
`complex * ArrayLike` overload, but `paulis/general.py` and `math.py` accept `ArrayLike` precisely so a
caller can pass `jax.numpy`. Two of these already carry per-line ignores. Widening the annotations to
`Any` would silence the rule and lose every real check in those functions.

**Class C — genuine narrowing gaps. Worth doing, small.** The tractable remainder:

- `svsim.to_circuitxz`'s `circuit.qregs` reads (`svsim.py:198,200`) sit after an `isinstance` check on a
  union arm the checker has not narrowed at that point. Hoisting the narrowed value into a local would
  fix it.
- `sqd.py:1410`'s `.at[...]` on an `NDArray`-declared value that is a `jax.Array` at runtime — an
  annotation that is simply too narrow for what the function accepts.

Class C is where the next pass should go. It will not retire `unresolved-attribute` on its own, because
Classes A and B share that rule.

## 4. Two suggestions beyond the ignore list

**Enable `possibly-unresolved-reference` — checked, and it finds a real bug.** Every rule in the
current list is a *type* rule; control-flow rules catch a different class and are not firing by default.
Enabling it reports **9 findings** (5 in `rqutils/`, 2 in `examples/scaling/`), and at least one is a
genuine latent defect:

```
rqutils/qprint.py:617  Name `expr` used when possibly not defined
```

`QPrintBase._format_phase` binds `expr` only inside `if mode == "text"` and the `latex` branch, then
returns it unconditionally. Confirmed reachable — `_format_phase("0.5", "mpl")` raises
`NameError: cannot access local variable 'expr' where it is not associated with a value`, where an
explicit "unsupported mode" error is what a caller needs. This is exactly the `fmt` x `output` grid
`CLAUDE.md` warns about, where four bugs previously hid in cells nothing exercised.

The other findings (`sqd.py:896-902`, `qprint.py:739,748`, `ground_locg.py:1013`) were not individually
triaged. **Recommended next step**: triage those 9, fix what is real, then enable the rule — it is
cheaper than any of the four remaining type rules and covers a class currently unchecked.

**`_Result` / `_DebugResult` could be `NamedTuple`s.** `ground_locg` returns plain tuple aliases, and
its own comment explains why: every caller destructures positionally, and the arity is what a checker
needs. That reasoning still holds, but a `NamedTuple` is arity-compatible with positional
destructuring while also giving `.converged` at the call sites that want it — `sqd.py:1000` currently
reads `_` for two of four elements. Not obviously worth the churn on published API; recorded because the
overload work made the return shapes precise enough that the question is now answerable rather than
speculative.

## 5. How to check a rule before adding one

```bash
uv run --extra dev --extra qiskit --extra mpl --extra qutip \
  ty check -c 'rules.<name>="error"' rqutils/ tests/ examples/
```

Count, then **read** the diagnostics — the count alone does not distinguish Class A from Class C. And
note `ty` silently checks *nothing* for a file outside the project root, so probes must be written
inside the tree (`conftest.assert_type_checks` handles this).
