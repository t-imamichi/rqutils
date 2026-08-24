# API gotchas, and the breaking changes that would close them

What a caller can get wrong such that `rqutils` returns a **plausible finite number** rather than
raising. Breaking changes are assumed to be allowed, so the fixes proposed here are structural (make
the mistake unrepresentable) rather than documentary.

**Coverage.** Only modules on this branch (`dev`) are covered, and every `file:line` citation below was
verified against the source here rather than trusted. Two exclusions:

* `rqutils/qprint.py` -- excluded at the user's instruction, unaudited.
* `rqutils/product.py` -- not on this branch. Its `Solution.lower_bound` is *not* a lower bound on
  `lambda_min` (measured above it at n=4, 6, 8, so `H - lower_bound*I` is still indefinite); that one is
  recorded in `docs/sdp-lower-bound.md`, which is where the measurement lives.

Method: two independent sweeps -- one over the library source, one over the ~4k lines of findings
under `docs/` plus `CLAUDE.md`/`NOTES.md` -- then hands-on verification of the severe items by
running them. Every "measured" figure below was produced on this machine unless it cites a doc.

The last section lists the gotchas **already closed** by earlier breaking changes, so nobody redoes
that work.

---

## Why this list is short on "just document it"

This repo has already made the make-it-unrepresentable move twice, and both worked:

* the state padding bit was converted from an opt-in `add_padding` flag into an unconditional
  property of `PauliSumXZ`, so the two halves of the alignment contract "cannot drift"
  (`symplectic.py:96-101`);
* `apply_h`'s positional `(scanned, cache_level)` form was **deleted** rather than deprecated,
  because `cache_level` selected positionally how `scanned`'s members were read and nothing could
  check the two agreed -- measured **0.44 max abs error** from one mispairing, and at `n = 15` with a
  2-state subspace even the *shapes* collide at `(2, 2)`, so no assertion could have closed it
  (`docs/rqutils-requests.md` C1).

Items 1, 2, 4, 5 and 6 below are that same move applied to states validation, column order, cache
level and array identity.

---

## Tier 1 -- silently wrong number, easily reached

### 1. `pack_states` maps any nonzero to 1, so `{-1,+1}` spin encoding collapses every state

`rqutils/paulis/symplectic.py:116` -- `np.packbits(np.pad(states.astype(np.uint8), {1: (1, 0)}), axis=1)`.

Two unchecked coercions in one expression. `np.packbits` maps **every** nonzero entry to 1, so a
caller holding spins in the standard `{-1, +1}` convention gets the all-ones bitstring for every row;
`uniquify_states` then collapses them to a single state and `sqd` returns its diagonal element.
Separately, `astype(np.uint8)` wraps, so `256` becomes `0`.

Measured end to end, 4-qubit XXZ, same physical states in both arms:

| encoding | returned energy |
| --- | --- |
| `{0, 1}` | −0.434017 |
| `{-1, +1}` | **0.000000** |
| true ground energy | −1.358047 |

No error, no warning. Also verified directly: `pack_states([[-1,1,-1,1]])` and
`pack_states([[1,1,1,1]])` both return `[[120]]`; `pack_states([[0,1,256]])` and
`pack_states([[0,1,0]])` both return `[[32]]`.

`sqd`'s docstring says "States must have binary values" -- a precondition nothing verifies, on
caller-supplied data, in the function documented as the canonical entry point.

**Breaking fix.** Raise in `pack_states` unless every entry is in `{0, 1}`. It is `O(N·n)` on an array
`packbits` is about to walk anyway, and it is the **single** choke point for both `sqd` and `hproj`.
Alternatively accept `np.bool_` only, which makes `{-1,+1}` a `TypeError` at the boundary.

### 2. `sqd` indexes `states` by Pauli-string character; `svsim` and qiskit index by qubit

`rqutils/sqd.py:241` (`states` shape `(subspace_dim, num_qubits)`) against `rqutils/svsim.py:238`
(qubit `q` is bit `q` of the state-vector index). `tests/conftest.py:124` pins the `sqd` side
(`unique.dot(1 << np.arange(num_qubits)[::-1])`); `tests/test_svsim.py:249` pins the other.

**What the convention actually is, measured rather than read off the docstring.** A `states` column is
a Pauli-string **character** position, and `PauliSumXZ`'s packed signatures are character-ordered too
-- both ingest branches agree bit for bit, so the class does *not* straddle two conventions internally.
Verified with the index-based constructor, which removes the character-counting ambiguity that a
string like `"ZII"` carries:

| `from_sparse_list([("Z", [q], 1.0)], 3)` | string | packed payload index |
| --- | --- | --- |
| q = 0 | `IIZ` | 3 |
| q = 1 | `IZI` | 2 |
| q = 2 | `ZII` | 1 |

Payload index is exactly character index + 1 (index 0 is the pad bit), for both the tuple and the
qiskit branch. So the `[:, ::-1]` in `from_paulisum` is an **adapter** -- it converts qiskit's
qubit-indexed `.x`/`.z` into the character order the packed layout uses -- not a declaration that the
layout is little-endian. The docstring at `symplectic.py:152` ("this class is little-endian in qubit
order") describes the reversal's intent and reads as a claim about layout, which is the opposite; the
wording at `symplectic.py:39` is the accurate one. An earlier revision of this document repeated the
misleading version.

**Only the caller-facing column order is incidental; the byte order is load-bearing.**
`_pack_state_keys` (`sqd.py:697`) requires byte 0 to be most significant so that integer order on the
packed keys equals row lex order -- which is what lets `get_xsource` use a scalar binary search
instead of a lexicographic one, a measured **12-19x** speedup on the routine `NOTES.md` puts at
**66-97%** of a solve. Reversing the *column* order at the boundary is free by comparison: measured
0.033 ms at N=3 000/n=14 (0.006% of the solve) and 5.4 ms at N=500 000/n=24 (0.18% of the setup cost).
So "switch to LSB-first" is cheap but buys nothing on its own; making the convention **explicit** is
the change with the payoff.

**A worked instance -- this bug was live in the repo's own harness.** The `subspace` helper in
`docs/rqutils-precond-request.md` paired `bit q -> column q` with an `xxz` operator built by
`SparsePauliOp.from_sparse_list`, which is qubit-indexed. Measured, `Z` on qubit 0 over codes
`{0, 1}`:

| states pairing | `diag(<s|Z_0|s>)` |
| --- | --- |
| `bit q -> column q` (as shipped) | `[1, 1]` -- **no dependence on qubit 0 at all** |
| `bit q -> column n-1-q` | `[1, -1]` -- correct |

It went unnoticed because the chain is uniform, so the mislabelled operator is the *bit-reversed* XXZ
-- a legitimate instance with qubits renumbered, giving self-consistent ratios. A site-dependent field
would have exposed it. The helper is now fixed, and
`tests/test_sqd.py::TestHproj::test_states_columns_are_character_indexed_not_qubit_indexed` pins it,
including an arm asserting the naive pairing really is insensitive to qubit 0 (so the reversal is
load-bearing rather than cosmetic). Verified by mutation: deleting the `[::-1]` fails the test with
`[1.0, 1.0]` against the expected `[-1.0, 1.0]`.

That path was **uncovered until now** because every other qiskit test in the suite builds operators
from *strings* (`SparsePauliOp(["ZI"], ...)`), where character order is what the caller already wrote.
Only the index-based constructor exposes the flip.

The documented SKQD workflow is *sample bitstrings from* `svsim`*, feed them to* `sqd`, which crosses
exactly this boundary. A column-reversed `states` array is still a legitimate subspace: uniquify
succeeds, the projection is symmetric, `ground_locg` converges, and the energy is a true variational
upper bound on the true ground energy -- just of the wrong subspace. It reads as "the sample wasn't
good enough".

Measured, n=5, all couplings drawn at random:

| `states` | energy |
| --- | --- |
| as given | −2.627596 |
| columns reversed | −4.550395 |

**A trap in testing this, hit while writing this document.** A first attempt used uniform `J`, `Bx`,
`Bz` and measured a difference of **exactly 0.000000** -- because a uniform chain is symmetric under
bit reversal, so reversing the columns maps the subspace onto an equivalent one and the energy really
is unchanged. That is a vacuous pass, indistinguishable from a real negative. Randomizing every
coupling exposed the gotcha immediately. **A regression test for this needs asymmetric couplings** or
it will pass while the bug is present. (Same shape as the `uint64`-overrun trap in `NOTES.md`, which
only fires when the leading bytes collide.)

**Breaking fix.** Put the convention in the type: `BitstringTable(rows, order=Literal['msb_first',
'lsb_first'])`, accepted by `sqd`/`hproj`/`pack_states`, reversed internally when needed. Or accept
integer basis indices plus `num_qubits` and do the unpacking in-library, so a caller never expresses a
bit order at all. Splitting the entry points (`sqd_from_character_ordered_states` /
`..._from_qubit_indexed_states`) also works -- and note the names should say *character vs qubit*, not
*MSB vs LSB*: the distinction is which axis the column index means, not the endianness of an integer.

## Tier 2 -- wrong result, or a silent 7-11x slowdown, from an ordinary slip

### 3. `sqd(ham, states, True)` sets `states_size=1`

`rqutils/sqd.py:212`. Everything after `states` is positional-or-keyword, and the three parameters are
semantically unrelated (`states_size: int | None`, `return_eigvec: bool`, `cache_level: tuple`). Since
`True == 1`, the call is a *valid* `states_size` and does not raise -- it pins the array to size 1.

**Breaking fix.** `*` after `states`. `apply_h` already received exactly this treatment; the public
entry point was missed.

### 4. `cache_level` is a bare `tuple[int, int]` with unvalidated digits

`rqutils/sqd.py:217`, branched at `:519`, `:533`, `:540`, `:559-565`, `:1153-1159`. Every branch is an
equality test with an implicit `else`. Verified:

| `cache_level` | result |
| --- | --- |
| `(1, 0)` | −0.574858 |
| `(0, 1)` | −0.574858 |
| `(2, 0)` | −0.574858 -- **silently behaves as `(0, 0)`** |
| `(1, 5)` | `UnboundLocalError: cannot access local variable 'diagonals'` |

So an out-of-range first digit is silently ignored, and an out-of-range second digit surfaces as an
internal error rather than a validation error. The likelier mistake is the transposition: `(0, 1)` and
`(1, 0)` are both legal, both return the same energy, and differ only in cost -- `NOTES.md` measures
`(0, 2)` at **10.9x slower** than `(1, 2)` and `(0, 0)` at **7.2x slower** than `(1, 0)`. A transposed
tuple reads as "SQD is slow", never as an error.

**Breaking fix.** Two keyword-only enum parameters, `source_cache=` and `diagonal_cache=`. Distinct
types make the transposition a type error and an out-of-range value a `ValueError` at the boundary.

### 5. `PauliSumXZ.arrays` returns a bare 3-tuple of same-typed arrays

`rqutils/paulis/symplectic.py:288`. `x, z, c = ham.arrays` is correct; `z, x, c = ham.arrays`
type-checks, runs, and computes with X and Z swapped. This is the hazard that got `apply_h`'s
positional form deleted, one abstraction lower.

**Breaking fix.** `NamedTuple`. Zero runtime cost and still splat-compatible for the traced-function
use case, so this one is nearly free.

### 6. `apply_h` still accepts an array passed under the wrong *name*

Verified signature: `(vec, states, xsignatures, xsources, zsignatures, diag_signs, diagonals,
coeffs)` -- eight parameters, most of them same-typed integer arrays. Going keyword-only removed
*mispairing* but not *misnaming*: `apply_h(vec, xsources=x)` where `x` is a signature array is still
accepted. `docs/rqutils-requests.md` concedes this: "That residue is much smaller... but it is not
zero."

**Breaking fix.** Distinct `NewType`s per array role (`XSignatures`, `XSources`, `ZSignatures`,
`DiagSigns`, `Diagonals`), produced only by the functions that build them.

### 7. `pack_states` is not idempotent, and nothing cross-checks the width

`rqutils/sqd.py:305` and `:395`. `sqd` takes unpacked `(N, n)` states and **returns** unpacked ones,
but the natural intermediate a caller keeps -- from `uniquify_states`, or from `pack_states` called
directly as the docstring encourages -- is *packed*, shape `(N, ceil((n+1)/8))`. Feeding that back in
re-packs it: `astype(uint8)` is a no-op and `packbits` then treats each byte as one bit via nonzero→1,
yielding a different subspace. No check catches it, because both inputs are 2-D uint8 and
`states.shape[1]` is never compared against `hamiltonian.num_qubits` anywhere. This is a realistic
loop: run `sqd`, do configuration recovery, run `sqd` again.

**Breaking fix.** Raise when `states.shape[1] != hamiltonian.num_qubits` in `sqd` and `hproj`. One
`O(1)` comparison that closes double-packing, a transposed array, and a mismatched Hamiltonian at
once. Give packed states their own type so `pack_states` cannot consume its own output.

### 8. `matrix_ufunc(hermitian=...)` is a positional tri-state with a silent `else`

`rqutils/math.py:26`, dispatched at `:68-74`. Valid values are `1`/`True`, `-1`, `0`; **everything
else falls through** to general `eig` -- slower, less accurate, no signal. `hermitian=1` routes to
`eigh`, which reads only the lower triangle, so on a non-Hermitian input it returns a well-formed
spectrum of a *different* operator (`tests/test_math.py:153` pins the error at `> 1.0`).

The severity multiplier is the signature: `hermitian` is the **second positional** parameter and
`with_diagonals: bool` the third, so `matrix_exp(mat, True)` reads naturally as "with diagonals" and
means "Hermitian". `with_diagonals` also changes the return arity, and `matrix_exp`/`matrix_angle`
annotate a bare `NDArray` -- wrong when it is set.

**Breaking fix.** Keyword-only after `mat`, plus a `Symmetry` enum (`GENERAL`, `HERMITIAN`,
`ANTI_HERMITIAN`) so `True`/`1`/`-1` are no longer accepted. Split the arity into two functions.

### 9. Lex-sortedness is required everywhere but enforced on one path

`sqd` sorts internally via `uniquify_states`; `hproj(..., unique_states=True)` requires the caller to
have done it and raises otherwise (`sqd.py:387`). Verified: `sqd` returns the same energy for sorted
and unsorted input. So `states` means "any list" to one function and "sorted, deduplicated list" to
its sibling -- same name, same type, same module, and the difference is carried by a boolean on the
*other* function. `CLAUDE.md` notes the requirement was "always required... but previously
undocumented".

**Breaking fix.** A `UniqueSortedStates` type returned only by a `prepare_states()` constructor.
`hproj` requires it, `sqd` accepts either, and `unique_states: bool` disappears -- the flag exists
purely to assert what a type could guarantee.

### 10. Public helpers bypass the entry-point guards

`uniquify_states`, `get_xsource` and `diag_signs` are un-underscored and called directly by six
scripts under `examples/scaling/` -- i.e. exactly the code that pushes `N`. `NOTES.md` records that
this is how the int32 iota was reached "with neither entry-point guard in the chain"; the response was
to push the guard down rather than narrow the API, so the bypass remains for anything not yet guarded.

**Breaking fix.** Underscore them and export a narrow façade, or keep them public but have them accept
only the wrapper types from items 8 and 10.

---

## Tier 3 -- real, but needs a refactor rather than a signature change

### 11. Return arity depends on a boolean

`sqd`'s `return_eigvec` (bare `float` vs 3-tuple, `sqd.py:218`) and `ground_locg`'s `debug` (4-tuple
vs 5-tuple, `ground_locg.py:300`). The latter's own docstring concedes a type checker "cannot follow"
a static flag into the return arity and tells callers to test `len(result) == 5`.

**Fix.** Always return a dataclass with optional fields populated; `debug` becomes "collect
diagnostics", not "change my type".

### 12. `sqd` discards `converged`

`sqd.py:642` does `eigval, eigvec, _, _ = ground_locg(...)`. A non-converged LOBPCG run still returns
`state.theta`, a valid variational upper bound -- finite and plausible. `sqd` wraps it in `float()` and
returns it as "Calculated ground state energy" with no indication. `docs/locg.md` records that this
absence "is the reason I4 could hide" (a sign error that made the convergence test unsatisfiable, so
the solver silently never converged). `sqd` exposes no `maxiter` or `tol`, so a caller cannot even
retry.

**Fix.** Return the flag (item 11 provides the place to put it) or raise on non-convergence, and expose
`maxiter`/`tol`.

### 13. `components(matrix, dim=None)` silently picks a basis

`rqutils/paulis/general.py:250`. On a 4x4 matrix `dim=None` infers `(4,)` -- one 4-level qudit -- where
the caller may have meant `(2, 2)`. The `prod(dim)` check passes both (`4 == 4`, `2*2 == 4`), both
return 16 valid complex coefficients, and the normalization factor `2**(len(dim) - 2)` differs by 2x
between them, so the numbers are incomparable but equally plausible. Under `npmod=jnp` even the shape
check is skipped (it is gated on `npmod is np`).

**Fix.** Make `dim` required; delete the single-system fallback.

### 14. `hproj(unique_states=True)` admits exactly one filler row

`_is_lex_sorted` rejects a padded `uniquify_states` result because two all-255 rows are duplicates --
but a **single** filler row is still strictly increasing and passes (`tests/test_sqd.py:397` records
`states_size=12` doing exactly this). `hproj` has no filler-masking step, so that row becomes a
spurious basis state in the dense `[N, N]` projection: one row and column too large, still symmetric,
plausible wrong eigenvalue. The guard rejects the easy case and admits the hard one.

**Fix.** Also reject any row whose byte 0 has the high bit set (i.e. `_is_filler`), which closes the
parity hole in one line, independent of how many fillers there are.

### 15. `svsim` infers `num_qubits` from the highest qubit touched

`rqutils/svsim.py:220`, `:260`, `:262`. On the gate-spec path, `num_qubits` is one past the largest
index appearing in any gate. A 10-qubit circuit whose qubit 9 receives no gate -- routine for a
layer-wise ansatz or a *slice* of a circuit -- yields a `2**9` state vector, returned normalized and
without error.

Related: `CircuitXZ` is a plain **mutable** dataclass while `PauliSumXZ` is `frozen=True`, and
`to_circuitxz` is advertised as idempotent and returns the *same object*. Two `svsim` calls therefore
share one mutable circuit, and `num_qubits` is baked into a jit cache key -- editing it between calls
silently reuses a kernel compiled for the old dimension.

**Fix.** Make `num_qubits` a required keyword on the gate-spec path, inferring it only from a
`QuantumCircuit` where the register width is authoritative. Add `frozen=True` to `CircuitXZ`.

### 16. `initial_state: NDArray | int` and `xinit: jax.Array | int`

`svsim.py:88`, `ground_locg.py:291`. An `int` means "one-hot index", an array means "amplitude
vector" -- two different arguments sharing one slot, dispatched on rank inside `@jax.jit`. Passing `0`
where `np.zeros(2**n)` was meant simulates a different initial state, legally. On the array path
nothing checks length against `2**num_qubits` (which compounds item 15), normalization, or dtype.

**Fix.** Split the parameter into mutually exclusive `initial_index=` and `initial_vector=`, and
validate shape and dtype on the vector path.

### 17. Cached sparse Pauli bases are handed out mutable

`rqutils/paulis/general.py:230-243`. The dense paths correctly `setflags(write=False)` before caching
-- returning a writeable copy *was* the previous defect -- but the `sparse=True` branch caches an
object array of `csr_array`s, and the source comment concedes `setflags` "would not protect its
elements". So `pauli_matrices(d, sparse=True)` returns the same CSR objects every call; an in-place
`/=` for a different normalization convention corrupts the process-lifetime cache, and every later
`components()` call silently uses the mutated basis. The result stays Hermitian and finite -- only the
normalization is wrong, which is the convention `CLAUDE.md` calls "the most bug-prone invariant here".

**Fix.** Copy on the sparse path (the arrays are small), or drop caching there.

### 18. `from_paulisum(op, 1e-3)` -- `atol` is positional and gates *discarding* signal

`rqutils/paulis/symplectic.py:137`, checked at `:227-235`. The design is otherwise careful (default
1e-12, ~4 orders above measured rounding; `atol=0.0` restores the exact test). But `atol` is
positional, so a caller who mistakes the second positional for a `simplify` tolerance or a coefficient
cutoff -- both plausible, since `simplify()` is called on ingest -- raises the Hermiticity threshold by
nine orders of magnitude. Line `:235` then executes `coeffs = coeffs.real`, **discarding** an imaginary
part up to `atol`: genuine signal, silently dropped, and the check that should have caught it is the
one that was loosened.

**Fix.** Keyword-only, and rename to `discard_imag_below` so the trade is visible at the call site.

### 19. The `uint64` fast path at `B <= 8` bytes is a correctness boundary, asserted not typed

`get_xsource` selects a `uint64`-key search for `B <= 8` and an explicit lexicographic search beyond.
`NOTES.md` is explicit that this is "a **correctness** limit -- a `uint64` key silently truncates a
wider row and aliases distinct states", and `docs/scaling-pocs.md` calls the `n+1 <= 64` limit "a hard
correctness boundary, asserted rather than documented".

**Fix.** Encode the width in the type of the packed-states wrapper from item 7 so the narrow path
cannot receive a wide table.

---

## Already closed by a breaking change -- do not redo

From `CLAUDE.md`, `NOTES.md`, `docs/rqutils-requests.md`, `docs/skqd.md`, `docs/locg.md`,
`docs/scaling-pocs.md`:

| gotcha | measured consequence | how it was closed |
| --- | --- | --- |
| `apply_h` positional `(scanned, cache_level)` | 0.44 max abs error; shapes collide at n=15 so no assertion could help | deleted; keyword-only |
| `hproj(unique_states=True)` on unsorted input | non-symmetric projection | now raises (behavioural break) |
| `add_padding` opt-in flag | how `hproj` shipped broken | pad bit made unconditional |
| `from_paulisum` exact `imag != 0` test | 18/18 valid operators rejected, residue 3.3e-16 | `atol=1e-12` |
| `states_size` default | 1.79 s → 1.43 s over five growing dims | next power of two |
| int32 subspace ceiling | wrapped to −2147483648, returned a corrupted permutation | `_MAX_STATES` enforced |
| `_accumulate_diagonal` rank-2 spec | **all** sharded `sqd` calls raised, every `cache_level` | rebuilt as `NamedSharding` |
| `vinit_from_min_diag` scatter | `ShardingTypeError` on any mesh | added `out_sharding` |
| `_spread_seed` where-mixing | failed `(0, *)` on any mesh | fixed |
| `diagonals[0]` vs `.real` | failed `(*, 2)` on any odd-Y Hamiltonian, single-device | fixed |
| `svsim` missing `(-i)^{x·z}` phase | overlap with Qiskit 1e-16; broke every `y`/`ry` | `sin` widened to complex128 |
| `iterm & 255` vs `& 7` | affected `cache_level[1] == 1` | fixed |
| `PauliSumXZ.matmul` signature shift | 2.07 max abs error | method removed |
| `ground_locg` I1-I7, S1, A1-A5 | NaNs; θ below the true ground state; 24-33x iteration cost | all fixed in `8fa6be2` |
| `npmod` gating of shape inference | entire `npmod=jnp` path broken in 3 places | `normalize_dim` unconditional |

---

## Inherent -- documentation only

* **No `force_real` on `PauliSumXZ`.** `.c` narrows to float64 exactly when every string has an even Y
  count; an odd-Y string is complex128 *by construction*. Measured max `|imag| = 0.25` on an n=16 XXZ
  chain with `By != 0`. Check `.c.dtype`.
* **`λ_0 = sqrt(2/n)·I`, not `I`** -- normalization is `tr(λ_k λ_l) = 2δ_kl`. Basis-index ordering is
  fixed by a shell-by-shell loop, and `components`/`labels` both index by position, so a reordering
  would disagree with the labels users read while every function stayed self-consistent.
* **Qiskit ingest is faithful.** `hproj` matches Qiskit's own dense projection element for element
  (diff 0.0); it is `qiskit-addon-sqd` that returns the complex conjugate on odd-Y terms. (The
  character-vs-qubit indexing of `states` is item 2, not this -- and it is *not* inherent.)
* **`svsim` mesh divisibility.** `mesh.size` must divide `2**num_qubits`; a 3- or 6-device mesh fails
  at every qubit count. A state vector cannot be padded -- its indices *are* the basis states.
* **`paulis(dim)` multi-subsystem einsum cap** at ~17 subsystems; `sparse=True` for products raises.
* **`cz` correct only up to a uniform `exp(iπ/4)`**, and decomposed only on the `QuantumCircuit` path.

---

## Suggested order

If only three are done: **1**, **3**, **5**. One validation check, one `*`, one `NamedTuple` -- between
them they close the most severe silent failure, the easiest slip, and a hazard class this repo has
already been bitten by twice. All three are local and low-risk.

**2** is the highest-value structural change and the largest, and the one with a demonstrated victim --
the repo's own `subspace` helper shipped the mistake, and no test covered the path. Two cautions from
fixing it: its regression test is easy to write vacuously (see the uniform-chain note under that item),
and the change to make is *explicitness*, not flipping the default -- reversing the column order costs
0.006-0.18% but buys nothing by itself, since neither convention is intrinsically better and a flip
breaks existing callers silently.

Items **7** and **14** are one-line checks with good severity-to-effort ratios and could ride along
with the first three.
