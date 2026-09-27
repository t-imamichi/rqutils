# Refactor plan for `rqutils/` (2026-09-27)

Why: several files and functions have grown long. Measured on branch `matvec-names` (AST line counts),
excluding `qprint` and `svsim`, which were earlier ruled out of scope for refactoring.

## Where the length is

Most of it is docstring, not code:

| module | lines | docstrings | module docstring |
| --- | --- | --- | --- |
| `sqd.py` | 2090 | 917 (44%) | 172 |
| `ground_locg.py` | 1420 | 711 (50%) | 289 |
| `paulis/symplectic.py` | 323 | 174 (54%) | 47 |
| `paulis/general.py` | 368 | 173 (47%) | 91 |
| `math.py` | 239 | 92 (38%) | 14 |

The longest functions:

| function | lines | of which docstring | code |
| --- | --- | --- | --- |
| `sqd.sqd` | 314 | 198 | ~116 |
| `ground_locg.ground_locg` | 241 | 185 | ~56 |
| `ground_locg._ground_locg_callable` | 239 | 0 | 239 |
| `sqd.apply_h` | 120 | 61 | ~59 |
| `sqd.run_sqd` | 113 | 29 | ~84 |
| `paulis.symplectic.from_paulisum` | 110 | 31 | ~79 |
| `ground_locg._check_tols` | 102 | 46 | ~56 |
| `sqd.get_xsource` | 98 | 59 | ~39 |
| `sqd._solve` | 88 | 4 | 84 |

## The steps, highest value first

### 1. Split `sqd.py` into a package, with no API change

`rqutils/sqd/`, keeping `from rqutils.sqd import X` working for every public and currently imported
private name:

| submodule | contents |
| --- | --- |
| `__init__.py` | `sqd`, `hproj`, `EigenpairCheckError`, the `Matvec` types, `_check_matvec`, the published module docstring |
| `_states.py` | `pack_states` helpers, `uniquify_states`, `_is_lex_sorted`, `get_xsource`, `_pad_states`, `_check_states_shape` |
| `_diagonal.py` | `get_diagonal`, `_accumulate_diagonal`, `_z_parity` |
| `_dense.py` | `apply_h`, `_apply_h_kernel`, `_pack_scanned`, `apply_xgrp`, mesh placement |
| `_sparse.py` | size classes, construction and kernels of `"pairs"`/`"csr"`/`"ell"`, `_run_sparse` |
| `_solve.py` | `run_sqd`, `_solve`, `SqdResult`, `_spread_seed`, `_host_scalar` |

Acceptance: a pure move. `jax.make_jaxpr` hashes of every kernel (check on and off) and every sparse
operator array identical to the previous commit, the full suite, a clean docs build (1 known warning).
Risks to check first:

- **Monkeypatch targets move.** A test setting `rqutils.sqd._CHUNK`, or a script patching
  `rqutils.sqd.ground_locg`, would silently stop taking effect once the name lives in a submodule.
- **The published API reference** must not gain the private `_`-submodules; `docs/source/index.rst` may
  need a line (CLAUDE.md: a public module needs its directives and a toctree entry).
- **Path references** to `rqutils/sqd.py` in CLAUDE.md, NOTES.md, `markdown/` and `poc/` need updating;
  NOTES headings name functions (`sqd._host_scalar`), which stay valid.

### 2. Trim docstrings to CLAUDE.md's own ceiling

3–5 lines per docstring paragraph; evidence belongs in NOTES.md and the `poc/*.md` write-ups. Targets:
`sqd`'s 198-line docstring, `ground_locg`'s 185, `ground_locg`'s 289-line module docstring. `Args`,
`Returns` and `Raises` stay complete: they feed the published reference. A paragraph holding evidence
found nowhere else moves to NOTES rather than being cut. Likely the largest saving (several hundred
lines). Acceptance: a clean docs build and a read-through; the suite cannot see docstrings.

### 3. Break up `sqd()`'s ~116 lines of code

It validates, packs and pads the states, dispatches dense against sparse, runs the residual check and
formats the result inline. Named steps (`_prepare_states`, `_check_eigenpair`, ...) make it a short
sequence. Host-side Python only, no traced code: low risk.

### 4. Lift `_ground_locg_callable`'s nested functions to module level

239 lines of code: the seed steps, `body`, `compute_sas` and `diagnostics`, all closures inside one
function. Module-level functions with explicit arguments would be testable and readable. The highest
risk: CLAUDE.md says every guard here is load-bearing, with seven silent defects catalogued in
`markdown/locg.md`. Acceptance: jaxpr identity across `debug` and `batch_matvec`, `TestProjectOut`,
and the sharded all-reduce-count tests.

## Order and bar

1 → 3 → 2 → 4, one commit each. Steps 1 and 3 must prove identical traced graphs and bit-identical
operators; step 2 a clean docs build. The `"ell"` simplify fixes land first, since they touch `sqd.py`.
