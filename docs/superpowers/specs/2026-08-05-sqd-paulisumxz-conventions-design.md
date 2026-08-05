# Simplifying `rqutils/paulis/`, `sqd.py`, and `ground_locg.py`

Date: 2026-08-05
Status: approved, not yet implemented
Breaking: yes, in one narrow respect (`add_padding` is removed from `PauliSumXZ.from_paulisum`)

## Scope

Five changes, ordered by risk. Four are pure duplication removal with no observable behaviour
change. The fifth removes an unenforceable cross-object invariant that has already caused one bug.

| # | Change | Removes | Observable change |
|---|---|---|---|
| 1 | Unify `get_diagonal` / `compute_diagonal` | duplicated `cond_fn` + `while_loop` | none |
| 2 | Merge the two normalize helpers in `ground_locg` | one function written twice | none |
| 3 | One `dim` normalization helper in `general.py` | six drifting copies of an idiom | none |
| 4 | Collapse six `apply_h_*` kernels into one | ~90 lines; 6 functions for a 2x3 grid | none |
| 5 | Remove `add_padding` | a silent cross-object invariant | none |

### Explicitly rejected: LSB-first packing and a trailing sentinel

An earlier revision of this document proposed also switching `PauliSumXZ` to LSB-first packing
(`bitorder='little'`) with the fill-in sentinel moved to a trailing bit at index `num_qubits`. That
is **not** being done. The reasoning, since the analysis is worth keeping:

- It *relocates* complexity rather than removing it. Today's filler test, `states_u[:, 0] >> 7`, is a
  constant expression. Under a trailing sentinel it becomes `is_filler(states_u, num_qubits)`, which
  requires threading `num_qubits` to five sites **including inside the JIT'd `run_sqd`**, which
  currently never references it. `sqd.py` would end up more coupled to `num_qubits`, not less, in
  exchange for deleting one flag.
- The double bit reversal it would fix is confined to two places — `signature_bits` in the tests and
  `matmul`'s `offset` shift. Both are correct, tested, and documented. That is a convention doing its
  job, not a maintenance burden. Contrast item 3, where the copies have already drifted apart.
- It is the only change in this document with a behavioural blast radius: it reorders the basis
  states `sqd` returns, and it *introduces* a failure mode that does not currently exist (under LSB
  packing a genuine state can have `byte0 == 255`, which silently truncated `subspace_dim` to 2
  against a true 3 at `nq=9` — measured). Spending new tests at `nq in {7,8,9,15,16}` to defend
  against a hazard we created is a poor trade against a flag rename.

Two invariants were verified while evaluating it, recorded here so the work is not repeated: the
all-255 filler row does still sort last for every `nq` from 1 to 19 under LSB packing (by
construction, since the reserved bit caps the top payload byte below 255), and `matmul` would
collapse to a plain positional decode (`powers = 256 ** arange(B)`, no offset, no reversal). Neither
is sufficient motivation given the above.

## Item 1: unify the diagonal accumulators (`sqd.py`)

`compute_diagonal` (`sqd.py:601`) and `get_diagonal` (`sqd.py:628`) contain a byte-identical
`cond_fn`:

```python
def cond_fn(val):
    iterm = val[1]
    return jnp.logical_and(iterm < coeffs.shape[0], jnp.not_equal(coeffs[iterm], 0.0))
```

and `while_loop` bodies that differ in exactly one line — how the sign bit is derived:

- `compute_diagonal`: `(diag_signs[:, iterm // 8] >> (7 - (iterm & 7))) & 1`
- `get_diagonal`: `jnp.sum(jnp.bitwise_count(states & zsignatures[iterm]), axis=1, dtype=uint8) & 1`

Extract one private accumulator parameterized by that derivation:

```python
def _accumulate_diagonal(coeffs, size, sharding, sign_fn):
    """Sum coeff * (1 - 2 * sign_fn(iterm)) over terms, stopping at the first zero coefficient."""
```

Both public functions keep their names, signatures, `@jax.jit`, and docstrings; each becomes a
three-line call. The `iterm & 7` comment at `sqd.py:613-616` (recording the measured 0.71 absolute
error and 25% eigenvalue error from `& 255`) moves with the `sign_fn` it documents — it must not be
lost, since it names a real defect.

The early-stop-on-zero-coefficient semantics ("null terms are removed with `simplify()` so we
iterate until we hit coeff=0") is shared and unchanged.

## Item 2: merge the normalize helpers (`ground_locg.py`)

`normalize` (a closure inside `_ground_locg_callable`, `:307`) and `_normalize_or_zero` (module
level, `:669`) are the same function: divide by the norm, leave a zero vector untouched rather than
producing `NaN`. The only difference is that the closure accepts an optional precomputed norm to
avoid recomputing it.

Merge into one module-level helper carrying the optional argument. The closure exists only because
it was written inside the function; it captures nothing. All call sites — `:311, 372, 385, 389, 417,
436, 443, 488, 662, 671` — keep their current behaviour.

This does not touch any guard, balancing step, or re-orthogonalization. Per `CLAUDE.md` and
`docs/locg.md` those are load-bearing and measured; only the two helpers merge.

## Item 3: one `dim` normalization helper (`general.py`)

The idiom appears six times, in three flavours:

| Site | Function | Form |
|---|---|---|
| `:158` | `paulis` | `if` scalar, `elif` non-tuple -> `tuple(map(int, dim))` |
| `:288` | `paulis_shape` | `if` scalar only |
| `:318` | `components` | `elif` scalar (follows a `None` check) |
| `:348` | `compose` | `elif` scalar (follows a `None` check) |
| `:486` | `symmetry` | `if` scalar, `elif` non-tuple -> `tuple(map(int, dim))` |
| `:557` | `labels` | `if` scalar only |

Replace with one `_normalize_dim(dim) -> tuple[int, ...]` implementing the fullest form (scalar ->
1-tuple, any other sequence -> `tuple(map(int, ...))`).

**The flavour differences must be checked per site before merging, not assumed accidental.** The
adopted form is a superset of the narrower ones, so `paulis_shape` and `labels` gain the
non-tuple-sequence fallback they currently lack. That is a widening — a list `dim` that previously
flowed through unnormalized now becomes a tuple. For `labels` this is the one site where behaviour
could change, because `dim` is consumed by `zip(dim, symbol)` and `len(dim)`, both of which accept a
list today. Verify the widening is inert there before committing to it; if it is not, `labels` keeps
its own narrower call.

`components` and `compose` keep their `None` handling inline (it is not part of the idiom) and call
the helper in the `elif` position. The `npmod` gating rule from `CLAUDE.md` is unaffected: this
normalization is Python-level shape inference on a static value and must continue to run for *every*
`npmod`, never behind an `if npmod is np:` gate. That is precisely the bug the comments at `:312-315`
and `:406-411` record, so the helper is called unconditionally at every site.

## Item 4: collapse the six `apply_h_*` kernels (`sqd.py`)

`apply_h`, `apply_h_s_cached`, `apply_h_z_cached`, `apply_h_x_cached`, `apply_h_xs_cached`, and
`apply_h_xz_cached` (`sqd.py:665-761`) are one function. Each is a `jax.lax.scan` accumulating
`out + apply_xgrp(xsource, diagonal, vec)` over per-X-group arguments. They differ only in how the
two inputs are resolved:

- `xsource`: from the scanned arguments (cached), or from `get_xsource(xpat, states)` (not cached)
- `diagonal`: from the scanned arguments (level 2), from `compute_diagonal(signs, cs)` (level 1), or
  from `get_diagonal(zpats, cs, states)` (level 0)

That is exactly the 2x3 grid the `cache_level` tuple already names. Replace the six functions and the
six-arm `match` at `sqd.py:433-451` with one kernel taking `cache_level` as a static argument.

Constraints:

- `cache_level` is already static in `run_sqd` (`static_argnames`), so trace-time specialization is
  preserved and each combination still compiles to the same code it does today. This is a
  prerequisite, not an aspiration — verify the unified kernel is `jax.jit`'d with `cache_level`
  static, or every matvec call in the solver loop pays a retrace.
- The `match` block also binds a different `args` tuple per level. The unified version must keep that
  packing static too; it stays a Python-level `match` on the static tuple, just building arguments
  for one kernel instead of selecting among six.
- `apply_xgrp` is unchanged.

**Gate:** `tests/test_sqd.py` already parametrizes `CACHE_LEVELS` over all six combinations and
asserts each against `conftest.lowest_projected`, an independent dense reference. Those tests must
pass unchanged. Additionally assert the six kernels agree with each other elementwise on a fixed
input before and after the change — `CLAUDE.md` warns that cross-kernel agreement alone is
insufficient evidence (the two initial-vector bugs fooled all six identically), which is why the
independent dense reference remains the primary check and the mutual-agreement check is secondary.

## Item 5: remove `add_padding` (`symplectic.py`, `sqd.py`)

`PauliSumXZ.from_paulisum(..., add_padding=True)` pads `xsignatures` on axis 1 and `zsignatures` on
axis 2, shifting every signature right by one bit to align with the leading pad bit `sqd` inserts
into its states. The flag is the problem:

- It exists only to serve a `sqd` sentinel and is meaningless to `svsim`, the representation's other
  consumer, which never calls `from_paulisum`.
- Nothing enforces that the two sides agree, and disagreement is silent. `hproj` shipped with
  `add_padding=True` against unpadded states, so every matrix element landed in the wrong column
  (`tests/test_sqd.py:9`).

**Change:** delete the parameter. `from_paulisum` always applies the padding. Both consumers then
share one layout by construction, and there is no flag to set wrongly.

**Deliberately unchanged:** MSB-first `packbits`, the leading pad bit at position 0, `255` as the
filler value, the `states_u[:, 0] >> 7` and `== 255` sentinel tests, and the returned basis-state
order. This is what keeps the change inert everywhere except the removed argument.

Byte counts are unchanged, since the padding was already applied on the `sqd` path — the only path
that reaches `run_sqd`.

**Consequence for `svsim`:** none. It builds `CircuitXZ` itself and never calls `from_paulisum`.

**Consequence for a caller that wanted unpadded signatures:** there is none in the repository. Every
in-repo call either passes `add_padding=True` (`sqd.py:237,300`, `examples/_bench_common.py:83`,
`examples/check_bench_common.py:30`, `tests/test_sqd.py:100,127,130,438,460`) or passes
`add_padding=False` purely because it was the default and the test does not care
(`tests/test_paulis_symplectic.py`, ~20 sites). The unpadded layout is not used for anything.

### Sites to update

- `rqutils/paulis/symplectic.py:57` — drop the parameter from the signature; `:134` — drop the `if`.
- `rqutils/sqd.py:237,300` — drop the argument. `:209-212` — the docstring sentence requiring
  `add_padding=True` becomes a statement that the padding is intrinsic. `:303-305` — the `hproj`
  comment explaining the alignment stays (the alignment requirement is still real), but stops
  referring to a flag.
- `examples/_bench_common.py:83` — drop the argument; `:185` — the docstring recounting the old
  `add_padding` bug is history and stays, reworded so it does not imply the flag still exists.
- `examples/check_bench_common.py:30` — drop the argument.
- `tests/test_sqd.py:100,127,130,438,460` — drop the argument. Docstrings at `:9,:162` reworded.
- `tests/test_paulis_symplectic.py` — drop `add_padding=False` from 28 calls.
  `TestPadding` (`:238-260`) is rewritten rather than deleted. Both its tests currently work by
  comparing a padded build against an unpadded one, which is exactly the comparison that ceases to
  exist once the flag is gone (all 3 `add_padding=True` calls in this file are here). They must be
  re-expressed against an *absolute* reference rather than a relative one:
  - `test_padding_shifts_the_signatures` becomes "the pad bit is always reserved": assert the packed
    signature for `"IIX"` has its set bit at the position implied by one leading pad bit, computed
    directly from `num_qubits` rather than by differencing two builds.
  - `test_padding_does_not_change_the_coefficients` keeps its subject — the pad bit is a dummy
    identity and must not perturb the `(-i)^{x.z}` phase — but asserts against the phase table
    value for an odd-Y string (`"XY"` -> `-1j`) instead of against an unpadded build.

## Testing

Items 1-4 are refactors with no intended behaviour change, so the gate is the existing suite passing
unchanged, plus the item-4 mutual-agreement check described above. No new tests are needed for them:
`CLAUDE.md`'s "name the defect it locks down" rule applies to bug fixes, and these fix no bug.

Item 5 gets one new test naming the defect it prevents: `hproj` and `sqd` must agree on a subspace
containing a trailing basis state that no Pauli term couples into — the configuration behind the
measured 41x41-for-53-states truncation. With the flag gone the two sides cannot disagree on
alignment, and the test pins that.

Per `CLAUDE.md`, that new test is verified to fail against the old code by reverting the change in
place (not in a copy — the venv holds an editable install pointing at the original).

## Definition of done

- `uv run --extra dev pytest` passes.
- `uv run --extra dev ruff check rqutils/ tests/ examples/` and `ruff format --check` clean.
- `uv run --extra dev ty check rqutils/ tests/ examples/` clean.
- `uv run python examples/check_ground_locg_mlx_static.py` passes (numpy shim, runs headless).
- No `add_padding` reference remains anywhere: `grep -rn add_padding` returns only historical prose.
- `docs/locg.md` and `docs/skqd.md` need no changes — items 1-4 preserve every documented invariant,
  and item 5 touches neither module's subject matter.
