# Changelog

All notable changes to `rqutils` are recorded here. This file starts at the 0.2.0 -> unreleased
boundary; for earlier history see the git log.

## Unreleased

### Removed

- **The deprecated MLX port.** `examples/mlx/` (8 files) and `docs/mlx-metal-kernels.md` are gone.
  The JAX solver measured faster even on MLX's own GPU backend, and none of the port was executable
  without a Metal device. Three benchmarking lessons it had taught were relocated into `CLAUDE.md`.
- **Five dead functions in `rqutils.paulis.general`**: `compose`, `truncate`, `l0_projector`,
  `symmetry`, `paulis_shape`. None had a caller anywhere in `rqutils/` or `examples/` -- only their
  own tests. The published API for the module is now `paulis`, `pauli_matrices`, `components`,
  `labels`.

### Changed (breaking)

- **`PauliSumXZ.from_paulisum` lost its `force_real` flag.** A complex coefficient now always raises
  `ValueError`, rather than optionally being silently truncated to its real part. In its place,
  `atol` (default `1e-12`) sets how far a coefficient's imaginary part may sit from zero before it is
  no longer treated as real -- it widens what counts as real, but never discards a genuine imaginary
  component the way `force_real` did.
- **`PauliSumXZ` lost its `add_padding` flag.** The leading pad bit in signatures and packed states is
  now inserted unconditionally, since the two sides being optional was how they could disagree.
- **`hproj(unique_states=True)` now raises `ValueError` on unsorted or duplicate-containing input**,
  where it used to silently return a wrong, non-symmetric projection. `xsource` binary-searches into
  `states`, so both "uniquified" and "lex-sorted" are load-bearing preconditions; the new check
  validates the sortedness half host-side, at 12-14% of `hproj`'s cost on that opt-in path only. Pass
  `np.unique(states, axis=0)`, or leave `unique_states=False`.
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
