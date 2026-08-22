# rqutils change requests from the SKQD/basis-optimization side

> **Status: all four shipped** on `metal` (rev `d452c383`) and adopted by `spinchain`. One caveat that
> matters for anyone reading A1 as a template: A1 works exactly as specified -- `expval` is now
> differentiable w.r.t. coefficients -- but the *use* it was requested for did not survive
> measurement. Routing basis optimization's cost through the shared `expval` is 2x slower at n=20 and
> 9.7x slower at n=100, because `apply_h` takes one static `nterms` for all X groups while a rotated
> operator has median 2 terms per group and a max of 197 (1.4% of slots real). Per-group `nterms`, or
> a segment-sum layout, would be needed to close that. See the note at the end of A1.

Four requests against `rqutils` 0.2.0 (`github.com/t-imamichi/rqutils`, branch `metal`), written from
the `spinchain` side. Every claim below was measured on that version; each item states the exact call
site, a self-contained reproduction, and what `spinchain` deletes or stops guarding if it lands.

Ordered by value. **A1** and **C1** are the two worth doing first: A1 unblocks a reuse that removes
54 lines of duplicated projection machinery, and C1 closes a path that returns a wrong answer with no
error. C2 and C3 are smaller and independent.

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

**Measured outcome: that swap was tried and reverted.** It is exact (~5e-16 against a dense Qiskit
conjugation, value and gradient) but **2x slower at n=20 and 9.7x at n=100**. `apply_h` takes one
static `nterms` for every X group, and a rotated operator's sparsity is the opposite of a
Hamiltonian's: at n=100 its 2151 labels form a 782x197 rectangle with median 2 terms per group and a
max of 197, so **1.4% of slots hold real terms** and every group pays the widest group's extent. The
triple list enumerates only nonzeros and has no padding. Closing this needs **per-group `nterms`** (or
a segment-sum over a flat nonzero list) rather than a single scalar.

What `spinchain` did keep from A1: `_expval_kernel` now takes `nterms` and `expvals` passes
`max(nzterms)`, so `expval` is differentiable for any future caller. On js operators `max(nzterms)` is
1-2, so this is speed-neutral (13.4 -> 13.3 ms on the shipped 36-operator n=13 sweep, identical
values).

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
| A1 | small — one function, plus threading a static count | low: identical values, opt-in via a default of `None` | unblocks `expval` for gradient-based callers; deletes 54 lines in `spinchain` |
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
  of Y factors. No change wanted; `spinchain` handles it by having exactly one projection path.
