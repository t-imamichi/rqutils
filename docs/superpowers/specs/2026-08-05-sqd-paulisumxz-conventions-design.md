# Simplifying the `sqd` / `PauliSumXZ` bit-layout conventions

Date: 2026-08-05
Status: approved, not yet implemented
Breaking: yes (deliberately)

## Problem

`rqutils/sqd.py` carries several workarounds whose sole purpose is to reconcile its own state
packing with `PauliSumXZ`'s signature packing. All of them trace to one fact:

> `np.packbits` fills each byte from the **most significant** end.

Three separate mechanisms exist because of it.

### 1. The `add_padding` cross-object coupling

`sqd` needs a sentinel to mark fill-in slots produced by uniquification. Because `packbits` is
MSB-first, the only bit a real state provably cannot reach is a *prepended* one. So states are
packed as `np.packbits(np.pad(states, {1: (1, 0)}), axis=1)`, and the Hamiltonian must
independently be built with `PauliSumXZ.from_paulisum(..., add_padding=True)`, which pads
`xsignatures` on axis 1 and `zsignatures` on axis 2 to shift every signature right by one bit.

This is a flag on `PauliSumXZ` that exists only to serve a `sqd` sentinel. It is meaningless for
`svsim`, the representation's other consumer, which never calls `from_paulisum` at all. Nothing
enforces that the two sides agree, and disagreement is silent: `hproj` shipped with
`add_padding=True` against unpadded states, so every matrix element landed in the wrong column
(`tests/test_sqd.py:9`).

### 2. Bit order is reversed twice

Qiskit's `.x`/`.z` are reversed on ingest to little-endian
(`symplectic.py:88-89`), then `packbits` reverses again. The composition means a signature's payload
bits are the *first* `num_qubits` entries of `np.unpackbits`, in Pauli-string character order —
the reverse of the qubit numbering. Consequences:

- `tests/test_paulis_symplectic.py:44-53` (`signature_bits`) exists purely to document the double
  reversal, and has to slice then reverse.
- `PauliSumXZ.matmul` must compute `offset = 8 * self.x.shape[1] - self.num_qubits` and
  right-shift by it, with `powers = 256 ** arange(B)[::-1]`, to undo the byte-level reversal.

### 3. Two spellings of one predicate

The filler marker is tested as `states_u[:, 0] >> 7` in two places and `states_u[:, 0] == 255` in
two others, plus `fill_value=255` at the source. Same question, four expressions, no shared helper.

### 4. Duplicated packing incantation

`np.packbits(np.pad(states.astype(np.uint8), {1: (1, 0)}), axis=1)` appears verbatim in
`rqutils/sqd.py:247`, `rqutils/sqd.py:306`, `examples/_bench_common.py:91`, and
`tests/test_sqd.py:45`. `PauliSumXZ` offers no state-packing helper, so the layout knowledge is
copy-pasted rather than owned.

## Chosen approach

Change two conventions at their root: **pack LSB-first** and **move the sentinel to a trailing
bit**. Expose the layout through helpers so exactly one place knows the bit positions.

Rejected alternatives:

- *Keep MSB packing, drop only `add_padding`* (always reserve the pad bit). Fixes the coupling but
  leaves the double reversal and `matmul`'s offset shift in place.
- *Separate boolean validity array.* Removes the bit tricks but threads an extra `(N,)` sharded
  array through all six matvec kernels — cost without a corresponding simplification.

## Design

### `paulis/symplectic.py`

**Packing becomes LSB-first.** Both `packbits` calls in `from_paulisum` take `bitorder='little'`.
The ingest reversal to little-endian qubit order stays. The two no longer compound, so:

> **bit `q` of the packed signature is qubit `q`**, for every `q`.

**`add_padding` is deleted** (parameter removed, not defaulted). Signatures always pack into
`B = ceil((num_qubits + 1) / 8)` bytes, reserving bit index `num_qubits` as the sentinel position.
No shift is applied to the signatures: the reserved bit sits *above* the payload, so X and Z
signatures have it clear by construction. This is what removes the coupling — there is no longer a
flag `sqd` must set and `svsim` must not.

Byte counts are unchanged from the current padded layout (`ceil((nq+1)/8)` either way), so memory
and sharding are unaffected.

**`matmul` loses its correction.** `powers = 256 ** np.arange(B)` (no `[::-1]`), and the
`offset` computation and `>> offset` both go away — byte 0 is now least significant, so the decode
is positional. Verified: for `nq` in {3, 8, 9} with qubits 0 and `nq-1` set, the reconstructed
integer equals `(1 << 0) | (1 << (nq-1))` exactly.

**Two new module-level helpers** own the layout:

```python
def pack_states(states: NDArray) -> NDArray[np.uint8]:
    """Pack binary states LSB-first, reserving the trailing sentinel bit. Returns (N, B) uint8."""

def is_filler(states_p, num_qubits) -> Array:
    """True where the sentinel bit is set, i.e. the row is a uniquification fill-in."""
```

`is_filler` tests byte `num_qubits // 8`, bit `num_qubits % 8`. It must work under `jax.jit` on
traced arrays (`sqd` calls it inside `run_sqd`) and on plain numpy (`examples/`, tests), so it uses
only ops common to both — shift, mask, compare.

### `sqd.py`

| Site | Before | After |
|---|---|---|
| `sqd:237` | `from_paulisum(..., add_padding=True)` | `from_paulisum(..., force_real=True)` |
| `sqd:247` | `packbits(pad(...))` | `pack_states(states)` |
| `sqd:271` | `unpackbits(...)[:, 1 : 1 + nq]` | `unpackbits(..., bitorder='little')[:, :nq]` |
| `hproj:300` | `from_paulisum(..., add_padding=True)` | `from_paulisum(hamiltonian)` |
| `hproj:306` | `packbits(pad(...))` | `pack_states(states)` |
| `_spread_seed:380` | `states_u[:, 0] == 255` | `is_filler(states_u, num_qubits)` |
| `vinit_from_min_diag:459` | `states_u[:, 0] == 255` | `is_filler(states_u, num_qubits)` |
| `run_sqd:499` | `searchsorted(states_u[:, 0] >> 7, 1)` | `searchsorted(is_filler(...), True)` |

`num_qubits` reaches these sites via `hamiltonian.num_qubits`, already a `static`-metadata pytree
field, so no signature changes. `_spread_seed` gains a `num_qubits` parameter.

`uniquify_states` is unchanged, including `fill_value=255`.

### Invariants verified before adopting this design

1. **The all-255 filler row still sorts last, for every `nq` from 1 to 19.** Checked directly. The
   reserved sentinel bit forces the top payload byte of any real state strictly below `255`, so the
   all-ones row is the lexicographic maximum *by construction* — a stronger guarantee than the
   current scheme, which relies on byte 0's high bit alone.

2. **The current boundary predicate breaks and must move.** Under LSB packing a genuine state can
   have `byte0 == 255` (any `nq >= 8` with the low 8 qubits set). Measured at `nq=9`: rows
   `[[0,0], [85,1], [255,1]]` plus filler `[255,255]` sort correctly, but `byte0 >> 7` yields
   `[0,0,1,1]`, so `searchsorted(..., 1)` reports `subspace_dim = 2` against a true 3 — a silent
   truncation of the returned eigenvector. This is why `is_filler` must key off byte `nq // 8`,
   bit `nq % 8`, and why the `nq % 8 == 0` cases are explicitly tested.

### Deliberately out of scope

- **`svsim.py`** — builds `CircuitXZ` itself, never calls `from_paulisum`. Untouched. Its
  `sin`-carries-`i(-i)^popcount` phase convention is **not** being changed.
- **`ground_locg.py`, `ground_locg_mlx.py`** — consume `xsources`/`diagonals`, which are index and
  value arrays with no bit layout. Untouched, so the "change one, change both" duplication rule
  does not trigger.
- **The `Q = (-i)^{x·z} Z^z X^x` phase convention** — unchanged everywhere.

### Observable behaviour change

Sort order changes: low qubits are now the most significant lexsort key, so `sqd` returns basis
states in a different order than before. Eigenvalues are unaffected. Eigenvector entries are
permuted consistently with the returned basis rows, so callers that zip the two are correct without
modification; a caller that hardcoded row positions is not.

Benchmark results committed in `docs/` remain valid — this changes bit layout, not arithmetic or op
counts.

## Consumers to update

- `examples/_bench_common.py:83,91` — `add_padding=True` and the hand-rolled packing.
- `examples/_bench_common.py:185` — docstring referencing the old `add_padding` bug.
- `examples/check_bench_common.py:30` — `add_padding=True`.
- `tests/test_sqd.py` — `pack_padded` helper (:45), the `add_padding=True` calls
  (:100, :127, :130, :438, :460), the `>> 7` assertions (:150, :153), and the docstrings at
  :9, :138, :162, :354 that describe the old layout.
- `tests/test_paulis_symplectic.py` — `signature_bits` (:44-53) collapses to
  `unpackbits(packed, bitorder='little')[:num_qubits]`; every `add_padding=False` call loses the
  argument. `TestPadding` (:238-260) is rewritten rather than deleted: its first test
  (`test_padding_shifts_the_signatures`) loses its subject, but its second
  (`test_padding_does_not_change_the_coefficients`) keeps a valid one — the reserved sentinel bit is
  still a dummy identity and must not perturb the `(-i)^{x·z}` phase, so that assertion becomes
  "reserving the sentinel bit leaves `.c` unchanged" and is checked against an odd-Y string.

## Testing

Following the repo's defect-oriented convention: each test names the defect it locks down, records
measured wrong values, and prefers an independent reference over self-consistency.

1. **`add_padding` removal.** `hproj` and `sqd` must agree on a subspace containing a trailing basis
   state that no term couples into — the configuration behind the measured 41×41-for-53-states
   truncation. With one shared `pack_states`, the two sides cannot disagree on alignment.
2. **LSB packing.** The existing dense-reconstruction tests
   (`TestPhaseConvention::test_dense_reconstruction_matches`) are the independent reference and must
   pass unchanged against the Kronecker-product construction; only `signature_bits` changes.
3. **`matmul`.** Compare against `dense_from_strings` for a multi-qubit sum including odd-Y strings,
   covering `nq` on both sides of a byte boundary.
4. **Sentinel position.** Parametrize `nq` over {7, 8, 9, 15, 16} — the byte-boundary cases where a
   real state's byte 0 reaches 255 and the old `>> 7` predicate misreports `subspace_dim`. Assert
   `subspace_dim` equals the true unique count and that no filler row survives into the result.
5. **All six cache levels.** The existing `CACHE_LEVELS` parametrization must pass unchanged; the
   two initial-vector bugs affected all six identically, so cross-kernel agreement alone is not
   sufficient evidence.

Per `CLAUDE.md`, each new test is verified to fail against the old convention by reverting the
change in place (not in a copy — the venv holds an editable install pointing at the original).

## Definition of done

- `uv run --extra dev pytest` passes.
- `ruff check`, `ruff format --check`, and `ty check` clean over `rqutils/ tests/ examples/`.
- `examples/check_ground_locg_mlx_static.py` passes (numpy shim, runs headless).
- Module docstrings in `sqd.py` and `symplectic.py` updated: the `add_padding=True` requirement at
  `sqd.py:212` is removed, and the LSB-first layout plus trailing sentinel are documented with the
  same rigour as the conventions they replace.
