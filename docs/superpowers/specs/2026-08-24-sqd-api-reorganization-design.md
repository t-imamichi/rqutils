# Removing MLX and reorganizing the `sqd` API

Design for two sequential changes: deleting the deprecated MLX port, then reorganizing
`rqutils.sqd`'s public surface into documented tiers. Breaking changes are in scope.

## Context

Two problems, unrelated in cause but overlapping in the files they touch.

**MLX.** The port was deprecated earlier (`e5703ac` removed the `mlx` extra) but the code stayed:
8 files and 2696 lines under `examples/mlx/`, plus a 349-line `docs/mlx-metal-kernels.md`. The JAX
solver measured faster even on MLX's own GPU backend, so nothing should run it. It cannot be
exercised here at all — `solver.py` needs MLX to import and `check_solver_device.py` needs a Metal
device — so it is unverifiable code carrying maintenance weight, and CLAUDE.md spends a long section
explaining how its eight coupled scripts interlock.

**The `sqd` API.** The surface grew by accretion and now misstates itself:

- **8 functions are public in Python but absent from the docs**, including `apply_h`. Meanwhile the
  SKQD/`spinchain` project genuinely builds its `expval` kernel out of four of them
  (`apply_h`, `get_diag_signs`, `get_xsource`, `uniquify_states`), so the real situation is a
  supported building-block layer that the docs deny exists.
- **No `__all__`**, so `Callable`, `Sequence`, `Number`, `coo_array`, `csr_array`, `SparsePauliOp`,
  `PartitionSpec` and `get_abstract_mesh` are all reachable as `rqutils.sqd.X`.
  `rqutils/__init__.py` already uses `__all__`; `sqd.py` does not follow it.
- **Three diagonal builders** (`compute_diagonal`, `get_diagonal`, `all_diagonals`) with overlapping
  jobs and inconsistent names. Their signatures differ only in what identifies the sign source.
- **`apply_h` carries two calling conventions** — the named-keyword form plus a deprecated
  positional `(scanned, cache_level)` tuple with mixing checks and a `DeprecationWarning`.
- **Inconsistent naming**: `get_xsource`/`get_diag_signs` carry a `get_` prefix that
  `uniquify_states`/`apply_h` do not, and `apply_xgrp` is abbreviated where nothing else is.

Intended outcome: one honest, minimal public surface, with everything it promises documented and
everything it does not promise labelled as such.

## Decisions taken

| Question | Decision |
|---|---|
| MLX scope | Delete the tree and its doc; **keep the measured lessons**, reworded |
| Sequencing | Two commits, **MLX first** |
| Primitive status | Publish as documented tiers |
| Tier shape | Two labelled groups: stable Kernel API, unpromised building blocks |
| Naming | Drop `get_` prefixes |
| Break style | Hard break, no aliases |
| Diagonal builders | Merge into one `diagonals()` with named inputs |
| `apply_h` shim | Delete it |
| Migration | Full docs restructure **plus** a new `CHANGELOG.md` |

---

## Change 1 — Remove MLX

### Delete

- `examples/mlx/` — all 8 files (`solver.py`, `bench.py`, `_bench_common.py`, `count_ops.py`,
  `check_bench.py`, `check_bench_common.py`, `check_solver_headless.py`, `check_solver_device.py`)
- `docs/mlx-metal-kernels.md` — verified to be in no `toctree`, so no docs reference breaks

Verified the tree is **self-contained**: every `examples/mlx/*` cross-reference points within
`examples/mlx/`, and nothing outside imports from it.

### Edit

- `CLAUDE.md` — drop the MLX sections (the eight-script coupling table, the parity-rule note, the
  deprecation narrative). Keep the `examples/` description of the three remaining demos.

  **Rescue three lessons first.** They currently sit *inside* the MLX paragraph at `CLAUDE.md:266`
  but none of them depend on MLX, and the `examples/scaling/` prose cites CLAUDE.md as their source —
  so deleting that paragraph without relocating them would break the citations and lose the findings:

  | Lesson | Why it outlives MLX |
  |---|---|
  | "When you delete a comparison arm, check what it was incidentally covering" | A general testing lesson, learned when deleting a kernel silently removed the only coverage of an unrelated guard. Directly relevant to this very change, which deletes several comparison arms. |
  | The **3.9% noise floor**; treat any difference under ~4% as unresolved | Cited by `_scaling_common.py`. A property of benchmarking on this machine, not of MLX. |
  | The fp32 dynamic-range trap in an eigensolve | Cited by `poc6_mixed_precision.py`, which is a JAX POC. |

  Move these into the general testing/benchmarking guidance in CLAUDE.md, then reword the
  `examples/scaling/` citations to point at the new location.
- `pyproject.toml:68` — the comment naming `mlx` as a darwin-only extra. The extra itself is
  **already gone** from `[project.optional-dependencies]`; only this lint comment remains. Reword it
  to cite `mpi4py` alone, which is still the reason `unresolved-import` is ignored.
- **Four `examples/scaling/` files** — reword prose so the measurement survives without the MLX
  attribution. These are findings that still hold for the JAX paths and were expensive to get:

  | File | Lesson to keep |
  |---|---|
  | `_scaling_common.py:18,203` | the 3.9% noise floor, and that two runs of identical code can look like a valid comparison |
  | `poc4_real_symmetric.py:19` | the `solve_s`-versus-eigensolve distinction |
  | `poc6_mixed_precision.py:23` | the fp32 dynamic-range trap in an eigensolve |
  | `poc8_gpu_unverified.py:28` | that a flat CPU result does not mean a change is neutral |

  `_scaling_common.py:4-5` also cites `examples/mlx/`'s `sys.path` pattern as precedent; restate it
  as the directory's own convention.

### Not doing

`docs/superpowers/{plans,specs}/*` MLX documents stay. They are dated historical records of
completed work, not live documentation, and rewriting history there would be dishonest.

---

## Change 2 — Reorganize the API

### The surface

```
Solvers (stable)
    sqd          hproj

Kernel API (stable) — enough to build a matvec by hand
    uniquify_states    xsource    diag_signs    apply_h

Building blocks (documented, no stability promise)
    diagonals    apply_xgroup
```

Verified the Kernel API is **self-sufficient**: a full matvec runs from those four plus
`PauliSumXZ`, with no diagonal builder needed (`diag_signs` feeds `apply_h` directly).

`sqd.py` gains an `__all__` listing exactly these eight, which also stops the eight leaked imports.

### Renames — hard break, no aliases

| Before | After | Why |
|---|---|---|
| `get_xsource` | `xsource` | drop `get_` |
| `get_diag_signs` | `diag_signs` | drop `get_` |
| `apply_xgrp` | `apply_xgroup` | unabbreviate |
| `compute_diagonal`, `get_diagonal`, `all_diagonals` | `diagonals` | merge |
| `run_sqd` | `_run_sqd` | `sqd`'s jitted body; zero external callers, in no docs |

`uniquify_states` and `apply_h` keep their names — SKQD imports both, and neither carries a `get_`
prefix. Dropping `get_` from `get_diagonal` would have produced `diagonal`, colliding awkwardly with
`compute_diagonal`/`all_diagonals`; merging removes the collision rather than renaming around it.

### `diagonals()` — one entry point

An un-jitted dispatcher over the three existing jitted kernels, mirroring what `apply_h` already
does, so the two published entry points read alike:

```python
diagonals(coeffs, diag_signs=...)                        # was compute_diagonal
diagonals(coeffs, zsignatures=..., states=...)           # was get_diagonal
diagonals(coeffs, zsignatures=..., group_ids=...,
          states=..., num_groups=...)                    # was all_diagonals
```

Prototyped and verified: each path reproduces its predecessor **exactly**, and every invalid
combination raises `TypeError` (neither source, both sources, `zsignatures=` without `states=`,
`group_ids=` without `num_groups=`). `nterms` and `num_groups` stay `static_argnames` on the inner
kernels. The three kernels remain as private `_diag_from_signs` / `_diag_from_z` / `_diag_all_groups`
— they are genuinely two different algorithms (per-group scan versus all-group scatter) and merging
their bodies would delete the abstraction rather than share it.

### `apply_h` — one calling convention

Delete the positional `(scanned, cache_level)` path, the named-versus-legacy mixing checks, and the
`DeprecationWarning`: about 60 lines, plus `test_legacy_tuple_form_still_works_but_warns` and
`test_mixing_the_two_forms_raises`. `test_mispairing_is_now_unconstructible` keeps its `TypeError`
arms and loses only its legacy arm.

`_apply_h_resolved` stays private and `_run_sqd` keeps calling it directly — it holds an already
assembled `scanned` tuple and must bind statics via `functools.partial` for `ground_locg`'s
positional splat.

### Docs

- Restructure `sqd.py`'s API section into the three labelled groups, with an `autofunction`
  directive for every one of the eight names. Today only `sqd`, `hproj` and `all_diagonals` have one.
- New **`CHANGELOG.md`** with an `Unreleased` section: every rename, the removed `apply_h` form, the
  merged builders, the MLX removal, and the newly documented tier. CLAUDE.md notes the missing
  CHANGELOG twice, once saying a specific change "belongs in it".
- Update CLAUDE.md's architecture section for the new names, and add the tier distinction.

### Call sites

Roughly 50, all in files that can be run here (MLX going first means the rename never touches
`examples/mlx/`, which was the one edit that would have shipped unverified):

- `tests/test_sqd.py` — the bulk
- `examples/scaling/` — `baseline.py`, `poc23_caching.py`, `poc6_mixed_precision.py`,
  `poc4_real_symmetric.py`, `poc7_sharding.py`, `poc1_searchsorted.py`, `poc5_graycode.py`
- `docs/*.md` prose naming the old symbols

---

## Verification

Both changes:

```bash
uv run --extra dev pytest -q                                  # expect 428 (minus 2 deleted shim tests, plus new diagonals tests)
uv run --extra dev ruff check rqutils/ tests/ examples/
uv run --extra dev ruff format --check rqutils/ tests/ examples/
uv run --extra dev ty check rqutils/ tests/ examples/
```

Beyond pytest, each covering something the suite does not:

1. **All six cache levels against the dense reference**, single-device — `diagonals` and `apply_h`
   are on every one of them.
2. **Sharding** — `XLA_FLAGS=--xla_force_host_platform_device_count=4 uv run python
   examples/scaling/poc7_sharding.py`. It sweeps all six levels; pytest is single-device only, so
   this is the only mesh coverage.
3. **Docs build** — `cd docs && uv run --extra docs make clean && make html`. Exactly **one**
   standing warning (`rqutils.paulis.rst` not in a toctree); anything else is new. Confirm each new
   `autofunction` renders by grepping the built HTML for an `id="rqutils.sqd.<name>"` anchor, and
   that no cross-reference dangles after the renames.
4. **`__all__` is honest** — import `rqutils.sqd` and assert every `__all__` name exists and that no
   third-party name (`coo_array`, `SparsePauliOp`, …) is reachable through it.
5. **Control-char/tab docstring sweep** — the snippet in CLAUDE.md, extended to tabs (a literal tab
   inside an `r"""` docstring shipped once already).
6. **Smoke-run the touched scaling POCs** — `baseline.py`, `poc23_caching.py`.
7. **Mutation-test the `diagonals` dispatch** — neutering each axis check must fail a test. Use a
   fresh subprocess for the jitted kernels: patching in a live session reuses the compiled kernel
   and both arms return bit-identical numbers.

## Risks

- **Hard break for SKQD.** Two import lines (`get_xsource` → `xsource`, `get_diag_signs` →
  `diag_signs`) plus any `get_diagonal`/`compute_diagonal` use. Listed explicitly in the CHANGELOG.
  Chosen over aliases deliberately: an alias would re-create the two-surfaces problem being deleted
  from `apply_h`.
- **`diagonals()` adds a dispatch layer** where three direct calls existed. Mitigated by it being
  un-jitted host-side dispatch, off the matvec hot path (`_run_sqd` binds the kernels directly), and
  by the same pattern already being proven in `apply_h`.
- **MLX deletion is irreversible in the working tree**, though recoverable from git history. The
  measured findings that outlive it are preserved by rewording rather than deletion.
- **`_run_sqd` rename** could break an unknown external caller. It has no in-repo callers, appears in
  no docs, and is `sqd`'s jitted body rather than an API — but it is the one rename with no
  visible consumer to check against.
