# rqutils change requests from the SKQD/basis-optimization side

> ## Disposition of C1, C2, C3 (2026-08-23)
>
> **All three shipped. A1 was deliberately skipped** -- it did not pay off, per the measurement already
> recorded in the banner below (the 54-line deletion cost 3-6% up to n=50 and 20% at n=100), so the
> scalar-`nterms` work was left as it stands.
>
> Each premise was reproduced against HEAD before being fixed, and each fix was mutation-tested.
> Test count 389 -> 421.
>
> | # | Fix | Verified |
> | --- | --- | --- |
> | C1 | `apply_h` gained a keyword form; the 2x3 grid is now derived from which arrays are named. The jitted kernel became `_apply_h_kernel`, unchanged and still positional. | All six combinations agree with a dense reference to 0.0; the six malformed calls raise; 7/7 mutants caught |
> | C2 | `from_paulisum(..., atol=1e-12)` compares `abs(coeffs.imag) > atol` instead of `!= 0.0` | 18/18 rejections -> 0/18; genuine signal (1e-11 and up) still raises; `atol=0.0` restores the exact test |
> | C3 | `states_size` defaults to the next power of two at or above the input length | 1.20x over five growing dimensions, energies agreeing to 4.4e-16; verified on a 4-device mesh |
>
> ### Three premise corrections
>
> **C1's suggested cheap mitigation does not work.** The request offers, as a fallback, "a dtype or
> shape assertion per branch -- X sources are `(n_groups, n_states)` index arrays while X signatures are
> `(n_groups, n_bytes)` packed bits, so the trailing dimension already distinguishes them." Measured: at
> `n = 15` the signatures are 2 bytes wide, so a 2-state subspace makes **both arrays exactly `(2, 2)`**
> and the mispairing passes the assertion that was meant to trip it. Only dtype differs, which is an
> implementation detail of `get_xsource` rather than a contract. The keyword form was therefore
> implemented as the primary fix rather than the fallback, and the counterexample is pinned by
> `TestMatvecKernels::test_shape_assertion_would_not_have_closed_this`.
>
> **C1 is narrower than "unconstructible-when-wrong" suggests.** The keyword form removes *mispairing* --
> declaring one strategy while having packed the arrays for another -- because the strategy is no longer
> declared separately from the arrays. It does **not** catch an array passed under the wrong name
> (`xsources=x` is still accepted). That residue is much smaller, since the name sits beside the array
> it labels, but it is not zero and the docstring says so.
>
> **C3's energies are not "bit-identical".** They agree to 4.4e-16, not exactly. The eigensolver is
> iterative and the filler-masked diagonals shift with the padding, so the last bit moves. Immaterial to
> any caller, but the stronger word was wrong. (C3's own measured speedup here was 1.20x against the
> 1.25x quoted.)
>
> ### A pre-existing bug found on the way, fixed
>
> Verifying C3 on a mesh required running `examples/scaling/poc7_sharding.py`, which **failed at stock
> HEAD, before any of these changes** -- confirmed by stashing them. `_accumulate_diagonal` carried its
> template's *full* sharding spec onto a 1-D accumulator, but `get_diagonal` passes the 2-D
> `(N, nbytes)` state list, so a rank-2 `PartitionSpec` met a rank-1 `jnp.zeros`:
> `Length of sharding.spec (2) must be equal to aval's ndim (1)`. This fired on **every** sharded `sqd`
> call at **every** `cache_level`, because `run_sqd`'s `vinit_from_min_diag` reaches `get_diagonal`
> unconditionally -- i.e. sharded `sqd` was entirely broken on `dev`.
>
> Fixed by carrying only the leading axis, rebuilt as a `NamedSharding` rather than a bare
> `PartitionSpec` (the bare spec is rejected when no mesh context is active, so that variant fixes the
> mesh path and breaks the single-device one -- 63 test failures). Both arms are now pinned.
>
> Why it went unnoticed is the reusable lesson: **reverting the fix leaves the entire pytest suite
> green** while the mesh path raises, because pytest was single-device only and `poc7_sharding.py` was
> the sole coverage. `tests/test_sqd.py::TestShardedDiagonalRank` now closes that gap, running
> `sqd` under `--xla_force_host_platform_device_count=4` in a subprocess (the device count must be set
> before jax initializes, which an in-process test cannot do after `conftest` has imported it).
>

> **Status: all four shipped, plus the segment-sum follow-up, and all are adopted by `spinchain`.**
> A1's payoff is realized: basis optimization's Eq. 8 routes through the shared
> `sqd_backend.expval_kernel_for`, and `_contraction_triples` (54 lines) plus `_pauli_entry` are
> deleted. The earlier verdict -- that the shared path was 9.7x slower at n=100 -- was a correct
> measurement of the *scalar* `nterms` API and no longer holds. On the flat path, measured A/B in one
> process with interleaved repeats:
>
> | n | shared `expval` | specialized triples | ratio |
> | --- | --- | --- | --- |
> | 20 | 17.8 ms | 16.9 ms | 1.06x |
> | 50 | 50.3 ms | 48.9 ms | 1.03x |
> | 100 | 107 ms | 89.0 ms | **1.20x** |
>
> Values identical to 9e-16. **This supersedes a "6-8%" figure an earlier revision of this file
> quoted**, which came from comparing separate processes and was noise-dominated; 3-6% up to n=50
> rising to 20% at n=100 is the honest shape.
>
> One argument for the swap was overstated and is corrected in A1 below: sharing `expval` was justified
> as protecting the odd-Y cancellation, but the deleted triples were verified to agree with `expval` to
> **5.6e-17** -- they carried the same convention, so they were a second *implementation*, not a second
> *library*. The real payoff is 54 fewer lines and one less copy of a sign convention that is invisible
> at `theta = 0`.
>
> ### ⚠️ The symbol names below are pre-0.3 and no longer import
>
> Everything under this banner is the **original request, preserved as written**, so its code blocks
> reference names that have since been renamed. Pasting one raises `ImportError`. The mapping:
> `get_xsource` -> `xsource`, `get_diag_signs` -> `diag_signs`, `apply_xgrp` -> `apply_xgroup`,
> `run_sqd` -> `_run_sqd` (now private), and `compute_diagonal`/`get_diagonal`/`all_diagonals` ->
> the single `diagonals()` with a named sign source. Its line-number references (`sqd.py:789`,
> `:827`, `:841`, `:867`, `:208`, `:442`) are against 0.2.0 and have drifted too. See `CHANGELOG.md`.
>
> ### ⚠️ Follow-up: A1's ragged-operator gap is now closed
>
> **The segment-sum layout that last paragraph asks for was built** (`rqutils.sqd.all_diagonals`,
> fed by `PauliSumXZ.flat_terms`), so the "would be needed" above is no longer the current state.
> It reduces per-term contributions to per-group rows in one pass over the *real* terms, so the work
> is proportional to `sum(nzterms)` rather than `J * max(nzterms)`.
>
> Measured against the rectangular scan at N=4096, **bit-identical** (`maxdiff` exactly 0.0) at every
> size, on local two-body operators with long-range Z strings added:
>
> | J | skew | real slots | speedup |
> | --- | --- | --- | --- |
> | 10 | 4.2x | 23.8% | 2.0x |
> | 40 | 17.1x | 5.9% | 47-49x |
> | 100 | 42.8x | 2.3% | **150-160x** |
>
> (Ranges, not point values: two runs of the same sweep gave 49.3x/156.9x and 47.0x/153.0x, so the
> last few percent is run-to-run variance rather than signal.)
>
> Compile time is *lower* than the scan's at every size (52 ms against 282 ms at J=100) and nearly
> flat in J, which is what makes this the right structure where unrolling the group loop was not --
> that unroll bought a similar speedup but paid compile growing linearly in J and only broke even
> after thousands of calls. Differentiable w.r.t. coefficients with no special handling; a full
> `expval` gradient matches central finite differences to 1.2e-10. Sharding-transparent, verified on
> a 4-device mesh across every residue of `N mod mesh.size` to 2.2e-15.
>
> So the `spinchain` measurement above stands as a correct verdict on the *scalar* `nterms` API, and
> is worth re-running against `all_diagonals`. Note the shipped default for `apply_h` is still the
> single `max(nzterms)` -- ordinary Hamiltonians sit at low skew, where the two are within noise --
> so the flat layout is opt-in via `cache_level[1] == 2`.
>
> Everything below this banner is the **original request**, preserved as written. Its line numbers
> refer to `rqutils` 0.2.0 and have drifted; cite it for the measurements and the reasoning, not for
> locations.

Four requests against `rqutils` 0.2.0 (`github.com/t-imamichi/rqutils`, branch `metal`), written from
the `spinchain` side. Every claim below was measured on that version; each item states the exact call
site, a self-contained reproduction, and what `spinchain` deletes or stops guarding if it lands.

Ordered by value. **A1** and **C1** are the two worth doing first: A1 unblocks a reuse that removes
54 lines of duplicated projection machinery, and C1 closes a path that returns a wrong answer with no
error. C2 and C3 are smaller and independent.

(On A1, with hindsight: the scalar `nterms` this section asks for was necessary but **not sufficient**
for that 54-line deletion — it took the segment-sum follow-up as well, and even then the reuse costs a
few percent. The banner above has the numbers.)

Every reproduction below runs against `rqutils` and `qiskit` alone — no `spinchain` import — but they
all assume **64-bit jax**, so prefix them with `jax.config.update("jax_enable_x64", True)` before the
first array is created. In 32-bit the residues C2 discusses sit above `float32` eps and the numbers
will not match.

| # | Where | Kind | Ask |
| --- | --- | --- | --- |
| A1 | `sqd.py:789` `_accumulate_diagonal` | new capability | make the diagonal accumulation differentiable without losing its early exit |
| C1 | `sqd.py:867` `apply_h` | convention | make the `cache_level`/`scanned` pairing unconstructible-when-wrong |
| C2 | `paulis/symplectic.py:207` `from_paulisum` | convention | accept coefficients that are Hermitian *to rounding* |
| C3 | `sqd.py:208` `sqd` | convention | default `states_size` to the bucketing every caller reimplements |

---

## A1. `_accumulate_diagonal` cannot be reverse-mode differentiated

**Where:** `rqutils/sqd.py:789`, `_accumulate_diagonal`. Reached by `compute_diagonal` (`:827`) and
`get_diagonal` (`:841`), therefore by `apply_h` (`:867`) at every `cache_level` except `(*, 2)`.

### The problem

The accumulator terminates on a data-dependent condition that reads the coefficient array:

```python
def cond_fn(val):
    iterm = val[1]
    return jnp.logical_and(iterm < coeffs.shape[0], jnp.not_equal(coeffs[iterm], 0.0))
...
return jax.lax.while_loop(cond_fn, add_diag, (init, 0))[0]
```

`jax.grad` rejects that outright:

```text
ValueError: Reverse-mode differentiation does not work for lax.while_loop or lax.fori_loop
with dynamic start/stop values. Try using lax.scan, or using fori_loop with static start/stop.
```

Since `coeffs` is exactly the array a variational parameter flows through, **nothing built on
`apply_h` can be differentiated with respect to operator coefficients.** Note the limitation is
specifically w.r.t. coefficients — grad w.r.t. the *vector* already works, because the vector does
not appear in `cond_fn`.

### Why the obvious fix is wrong

Replacing the `while_loop` with a `lax.scan` over the full padded rectangle is differentiable and
numerically identical (padding carries `coeffs == 0`, contributing exactly zero), but it throws away
the early exit — and the padding fraction is large:

| n | X groups | coeff rectangle | nonzero | padding |
| --- | --- | --- | --- | --- |
| 13 | 15 | (15, 12) | 38/180 | 78.9% |
| 20 | 22 | (22, 19) | 59/418 | 85.9% |
| 30 | 32 | (32, 29) | 89/928 | **90.4%** |

Measured cost of that (per call, `K=29`, 3 real terms, jitted, `block_until_ready`):

| states | `while_loop` | full-rectangle `scan` | `einsum` |
| --- | --- | --- | --- |
| 8 192 | 0.03 ms | 0.08 ms | 0.07 ms |
| 65 536 | 0.07 ms | 0.52 ms | 0.25 ms |
| 400 000 | 0.23 ms | 1.99 ms | 1.06 ms |

So the early exit is worth 2.6–8.6x and should be kept.

### The ask: a static trip count

`lax.scan` over `jnp.arange(nterms)` where `nterms` is a **Python int known at trace time** gives all
three properties at once. Measured against the current `while_loop`:

| | value vs `while_loop` | 400 000 states | `jax.grad` |
| --- | --- | --- | --- |
| `while_loop` (today) | — | 0.25 ms | **fails** |
| full-rectangle `scan` | identical | 1.99 ms | works |
| **static-length `scan`** | **identical (diff 0.0)** | **0.23 ms** | **works** |

The number is already available: `PauliSumXZ` knows the real term count per X group at ingest (it is
what the zero padding is padding *to*), so it can be stored alongside `arrays` and threaded to
`_accumulate_diagonal` rather than rediscovered per call. Because `nterms` must be static, it belongs
in `static_argnames` on the jitted wrappers, the way `cache_level` already is.

Suggested shape — keep the `while_loop` when the count is unknown, so no existing caller changes
behaviour:

```python
def _accumulate_diagonal(coeffs, template, sign_bit, nterms=None):
    init = jnp.zeros(template.shape[0], dtype=coeffs.dtype,
                     out_sharding=jax.typeof(template).sharding)
    if nterms is not None:                      # static: differentiable, same early exit
        def step(diagonal, iterm):
            return diagonal + coeffs[iterm] * (1.0 - 2.0 * sign_bit(iterm)), None
        return jax.lax.scan(step, init, jnp.arange(nterms))[0]
    return jax.lax.while_loop(cond_fn, add_diag, (init, 0))[0]
```

Per-group counts differ, so either pass the max (one trace, still skips most padding) or the
per-group count (one trace per distinct count). The max is the simpler default and already recovers
most of the win.

### Verification performed

A standalone reimplementation of `spinchain`'s `_expval_kernel` — built only from the public
`apply_xgrp`, `get_xsource`, `get_diag_signs`, `uniquify_states`, with the diagonal accumulated by a
static-length scan — reproduces `spinchain.skqd.sqd_backend.expval` at **diff 0.000e+00** on an n=8
XXZ chain over a 40-state subspace, and `jax.grad` w.r.t. the coefficients returns a finite gradient.
No `rqutils` source was modified to establish this.

### What lands in spinchain if this ships

`spinchain/skqd/basis_opt.py` hand-rolls the projected quadratic form as a precomputed
`(term, row, col, phase)` triple list — `_contraction_triples`, **54 lines** — purely because
`expval` is not differentiable. The intent was to delete it and route Eq. 8 through
`sqd_backend.expval` like every other observable.

**Measured outcome, first attempt: reverted.** Against the *scalar* `nterms` API the swap was exact
(~5e-16, value and gradient) but **2x slower at n=20 and 9.7x at n=100**, because `apply_h` took one
static `nterms` for every X group while a rotated operator's sparsity is the inverse of a
Hamiltonian's: at n=100 its 2151 labels form a 782x197 rectangle with median 2 terms per group, so
**1.4% of slots hold real terms** and every group paid the widest group's extent.

**Measured outcome, after the segment-sum follow-up: taken, but by a narrower margin than first
reported.** `PauliSumXZ.flat_terms` with `diagonals(group_ids=...)` reduces the real terms in one pass,
which takes the penalty from 970% to a few percent. Measured A/B in a single process with interleaved
repeats and the median of seven, subspace 500, `layers=2`:

| n | labels | shared `expval` | specialized triples | ratio | value diff |
| --- | --- | --- | --- | --- | --- |
| 20 | 391 | 17.8 ms | 16.9 ms | 1.06x | 1.4e-17 |
| 50 | 1051 | 50.3 ms | 48.9 ms | 1.03x | 0.0 |
| 100 | 2151 | 107 ms | 89.0 ms | **1.20x** | 8.9e-16 |

So: **3-6% up to n=50, 20% at n=100.** An earlier revision of this file said "6-8%", from separate
processes rather than an interleaved A/B; that figure was noise-dominated and is withdrawn.

The residual is the `(num_groups, num_states)` diagonal the flat path materializes -- 782 x 500 at
n=100, 391k entries -- that a contraction specialized to this operator never builds. It grows with the
group count, which is why the ratio widens with `n`.

### Correcting the justification

The swap was argued as restoring "one projection path", on the grounds that `sqd_backend`'s odd-Y
cancellation requires the eigenvector and every observable to come from one library. **That argument
does not hold, and the record should say so.** The deleted triples were checked against
`sqd_backend.expval` on an n=8 chain with odd-Y terms present: they agree to **5.6e-17**. They
reimplemented the *same* Qiskit/rqutils convention, so they were a second implementation, not a second
library. The hazard `sqd_backend` warns about is specifically `qiskit-addon-sqd`, which returns the
complex conjugate on odd-Y terms -- a genuinely different operator. That never applied here.

What the swap does buy, stated at its true weight:

- **54 fewer lines**, and one less copy of the row-vs-column sign convention. That convention is
  invisible at `theta = 0` and cost two debugging rounds during development, so a second copy is a real
  future-maintenance hazard even while both copies are correct.
- The `PauliSumXZ` layout knowledge lives in `sqd_backend`, next to the rest of it, rather than
  `basis_opt` owning a private subspace-projection scheme.
- `spinchain` and `rqutils` share a tested code path, so upstream improvements to `flat_terms` arrive
  for free. The specialized contraction would not benefit.

Both designs are exact, differentiable, and preserve the no-qubit-limit property (verified at n=70 and
n=100 with codes wider than int64). The decision is close enough that it turns on where basis
optimization is actually run: inside its measured working band (`dim/2^n` roughly 2-50%, so n about
16-22) the cost is 3-6%, and the 20% at n=100 is a size where the band has already collapsed to zero
gain. Reverting is one commit and remains reasonable for a caller who expects to run at n >= 50.

Also kept from A1: `_expval_kernel` takes `nterms` and `expvals` passes `max(nzterms)`, so plain
`expval` is differentiable too. On js operators `max(nzterms)` is 1-2, so that is speed-neutral
(13.4 -> 13.3 ms on the shipped 36-operator n=13 sweep, identical values).

---

## C1. `apply_h`'s positional `scanned` tuple is silently wrong when mispaired

**Where:** `rqutils/sqd.py:867`, `apply_h`.

### The problem

`cache_level` is static and selects, **positionally**, how the members of `scanned` are interpreted —
the 2x3 grid the docstring lays out. Nothing checks that the tuple the caller passed matches the
`cache_level` it declared. Supplying raw X signatures while claiming `cache_level[0] == 1` (which
promises precomputed X *sources*) raises nothing and silently computes a different operator:

```python
from rqutils.paulis.symplectic import PauliSumXZ
from rqutils.sqd import apply_h, get_xsource, get_diag_signs, uniquify_states
import jax, jax.numpy as jnp, numpy as np
from qiskit.quantum_info import SparsePauliOp

n = 4
ham = SparsePauliOp(["XXII", "IZZI", "IIYY", "ZIIZ"], [0.5, -0.3, 0.7, 0.2])
m = np.array([[int(b) for b in format(c, f"0{n}b")] for c in (0, 3, 5, 9, 12)], dtype=bool)
x, z, coeffs = PauliSumXZ.from_paulisum(ham).arrays
packed = PauliSumXZ.pack_states(m.astype(np.uint8))
states = uniquify_states(packed, packed.shape[0])
vec = jnp.asarray(np.arange(1, m.shape[0] + 1) / 5.0 + 0j)
xsrc = jax.lax.scan(lambda _, xx: (None, get_xsource(xx, states)), None, x)[1]
dsig = jax.lax.scan(lambda _, zz: (None, get_diag_signs(zz, states)), None, z)[1]

good = apply_h(vec, (xsrc, dsig, coeffs), states=states, cache_level=(1, 1))  # correct
bad = apply_h(vec, (x, dsig, coeffs), states=states, cache_level=(1, 1))  # x, not xsrc
```

```text
correct  : [ 0.2 +0.j -0.1 +0.j  0.46+0.j  0.22+0.j  0.2 +0.j]
mispaired: [-0.02+0.j  0.02+0.j  0.02+0.j -0.02+0.j  0.02+0.j]
no exception raised; max abs difference = 0.440000
```

Both arrays are integer-typed and have compatible shapes, so an index array and a signature array are
indistinguishable at the boundary. `spinchain/skqd/sqd_backend.py` documents the one correct pairing
in prose, with a warning that a mismatch "silently reads a signature array as an index array" —
prose is the only thing enforcing it today.

### The ask: name the inputs, derive `cache_level`

Accept the per-X-group arrays as **keywords** and let the supplied set determine the strategy:

```python
apply_h(vec, xsources=..., diag_signs=..., coeffs=..., states=...)   # today's (1, 1)
apply_h(vec, xsignatures=..., diagonals=..., states=...)             # today's (0, 2)
```

Each keyword names one specific representation, so the six valid combinations are the only
constructible ones and a mispairing becomes a `TypeError` at the call site instead of a wrong number.
`cache_level` can remain as a deprecated positional path for compatibility. If keeping one tuple is
preferable, a much cheaper mitigation is a dtype or shape assertion per branch — X sources are
`(n_groups, n_states)` index arrays while X signatures are `(n_groups, n_bytes)` packed bits, so the
trailing dimension already distinguishes them.

### What lands in spinchain

`sqd_backend._expval_kernel`'s docstring carries a paragraph explaining that
`cache_level=(1, 1)` pairs with `(xsources, diag_signs, coeffs)` and that the pairing is
"mandatory, not a preference". That paragraph exists only because the API cannot express the
constraint; it goes away.

---

## C2. `from_paulisum` rejects coefficients that are Hermitian to rounding

**Where:** `rqutils/paulis/symplectic.py:207`, `PauliSumXZ.from_paulisum`.

### The problem

The Hermiticity check is exact:

```python
if np.any(coeffs.imag != 0.0):
    raise ValueError("Coefficients of Paulis must be real for the Hamiltonian to be Hermitian.")
```

A mathematically Hermitian operator whose Pauli coefficients were obtained numerically carries
rounding at the 1e-16 level and is refused. Conjugating a Hermitian matrix by a non-Clifford circuit
and decomposing the result — the standard way to build a rotated Hamiltonian — does exactly that:

```python
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, SparsePauliOp
from rqutils.paulis.symplectic import PauliSumXZ

n, rng = 4, np.random.default_rng(1)
A = rng.normal(size=(2**n, 2**n)) + 1j * rng.normal(size=(2**n, 2**n))
H = A + A.conj().T                                  # Hermitian by construction
qc = QuantumCircuit(n)
for q in range(n):
    qc.ry(0.37 * (q + 1), q)
for q in range(n - 1):
    qc.cx(q, q + 1)
U = Operator(qc).to_matrix()
PauliSumXZ.from_paulisum(SparsePauliOp.from_operator(U @ H @ U.conj().T))   # raises
```

Measured over 18 such operators (n = 3, 4, 5 x 6 seeds): **18/18 rejected**, largest coefficient
`|imag|` **3.3e-16**, while the operators' own hermiticity error is at most **2.7e-15**. The
imaginary residue is an order of magnitude *smaller* than the hermiticity the check is testing for.

### The ask: a tolerance

```python
if np.any(np.abs(coeffs.imag) > atol):        # atol default ~1e-12
    raise ValueError(...)
coeffs = coeffs.real
```

This is deliberately **not** a return to the `force_real` behaviour the code comment at `:202-206`
describes removing. That took `.real` on operators with genuinely nonzero imaginary parts, discarding
real signal; a tolerance still rejects those loudly and only tolerates values that cannot be
distinguished from zero at float64. ~1e-12 sits about four orders above the observed residue and many
orders below any physical coefficient, so it separates the two cases cleanly. Exposing `atol` as a
keyword lets a caller tighten it.

### Scope note, stated honestly

`spinchain`'s current XXZ light-cone path does **not** trip this. Across 120 rotated operators
(Hamiltonian and spin currents, n = 6..13, layers 1-3) the imaginary residue was **exactly 0.0** and
none were rejected — those conjugations round cleanly because each window is small and the
coefficients come out on exact binary fractions. `spinchain/skqd/basis_opt.py:_real_coeffs` is
therefore insurance rather than a live workaround, and the request here is to remove a hazard that is
real in principle for any caller building operators numerically, not to unblock current work. Treat
this as lower priority than A1 and C1 accordingly.

---

## C3. `states_size` bucketing is boilerplate every caller reimplements

**Where:** `rqutils/sqd.py:208`, `sqd` (and `run_sqd` at `:442`).

### The problem

`states_size` pads the state array so jax does not retrace for every distinct subspace dimension. The
docstring introduces it for that reason, but leaves the choice of value to the caller — and the useful
choice is a fixed rule. `spinchain/skqd/sqd_backend.py:192` computes:

```python
states_size = 1 << max(int(bitstring_matrix.shape[0] - 1).bit_length(), 1)
```

Growing, all-distinct dimensions are the *normal* SQD access pattern rather than an edge case: an SKQD
run walks one dimension per Krylov rung plus one per configuration-recovery round, so every call sees
a dimension it has not seen before. Any caller with that pattern needs this rule, and each one has to
know to write it.

### Verification

Five growing dimensions (60, 90, 130, 190, 260) at n=10, identical subspaces in both passes:

| | wall clock | energy |
| --- | --- | --- |
| no `states_size` | 1.79 s | -0.9680703308 |
| bucketed | **1.43 s** | -0.9680703308 |

**1.25x, with bit-identical energies** — it is purely a performance knob, with no effect on results.
(`spinchain` separately measured 1.43x over the first five rungs of its shipped n=13 job, energies
agreeing to 1.3e-15.)

### The ask

Default `states_size` to the next power of two at or above the input dimension, keeping the argument
as an explicit override:

```python
if states_size is None:
    states_size = 1 << max((num_states - 1).bit_length(), 1)
```

`sqd()` already trims the filler slots before returning `basis_states`, so nothing downstream
observes the padding and the default is transparent to existing callers that pass their own value.

### What lands in spinchain

The comment block at `sqd_backend.py:184-192` — eleven lines explaining what the bucketing is for and
citing the measurement that justifies it — reduces to nothing, and the rule stops being something
every new caller has to rediscover.

---

## Summary

| # | Effort | Risk | Payoff |
| --- | --- | --- | --- |
| A1 | small — one function, plus threading a static count | low: identical values, opt-in via a default of `None` | unblocks `expval` for gradient-based callers. The 54-line deletion needed the *flat* path on top, and costs 3-6% (n<=50) to 20% (n=100) — see A1 |
| C1 | moderate — signature change, compatibility shim | low if `cache_level` is kept as deprecated | removes a silent-wrong-answer path (0.44 measured error) |
| C2 | trivial — one comparison plus a keyword | low: strictly widens what is accepted, still rejects real signal | removes a false rejection of Hermitian-to-rounding operators |
| C3 | trivial — one default | none: identical results, transparent to existing callers | 1.25x for the common access pattern; deletes caller boilerplate |

Two items deliberately **not** requested, recorded so they are not mistaken for oversights:

- **`svsim._GATE_XZ` has no generic multi-qubit Pauli rotation.** A `CX(a,b) Ry(t)_b CX(a,b)` block is
  exactly `exp(-i t/2 Y_b Z_a)`, one two-qubit Pauli rotation, and cannot be expressed by name today.
  The machinery is already generic (`to_circuitxz` folds the `(-i)^{x.z}` phase and sums bits over a
  qubit list; `rzz` proves multi-qubit entries work), and a hand-built entry matches a Qiskit
  `CX,Ry,CX` circuit to **2.8e-17** — so this is a table entry, not new machinery. Left out only
  because `spinchain`'s basis optimization no longer uses `svsim` at all; it would benefit other
  circuit-building callers.
- **The odd-Y sign convention is not an `rqutils` issue.** Verified `rqutils.sqd.hproj` matches
  Qiskit's own dense projection **element for element (diff 0.0)**. It is `qiskit-addon-sqd`'s
  `project_operator_to_subspace` that returns the complex conjugate on Pauli terms with an odd number
  of Y factors. No change wanted; `spinchain` handles it by never admitting that library. Note this is
  a narrower claim than the "one projection path" argument retracted in A1 above: what matters is not
  having a second *implementation* of the projection, but not mixing in one that uses a *different*
  convention.
