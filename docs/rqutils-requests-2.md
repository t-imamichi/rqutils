# rqutils change requests: a typed `HamiltonianInput`, and an eigenspace-aware `sqd`

Two requests against `rqutils` on branch `dev` (installed rev `f10b473`, version `0.2.0`), written from
the `spinchain` side. Both are pre-existing limitations rather than regressions — neither came from the
prefilter work, and the three bugs from `tmp/rqutils-prefilter-bug.md` and
`tmp/rqutils-prefilter-dim2-request.md` are all fixed and adopted.

Neither is urgent. **D1 is small, mechanical, and verified to work; D2 is a real capability request that
we are explicitly not blocked on.** Ordered by cost, not value.

| # | Ask | Kind | Effort | Our cost today |
| --- | --- | --- | --- | --- |
| D1 | Make `HamiltonianInput` visible to type checkers | annotation only, no runtime change | **~6 lines**, verified below | 2 `ty: ignore` suppressions on correct calls |
| D2 | Report the ground eigenvalue's multiplicity, or return an eigenspace basis | new capability | large | occupancy read off an arbitrary eigenspace member; reproducibility observed, not guaranteed |

Reproductions import `rqutils`, `numpy`, `jax` and `qiskit` only — no `spinchain` — and assume **64-bit
jax**.

---

## D1: `HamiltonianInput` hides its `SparsePauliOp` arm from type checkers

**Ask in one line:** declare the `SparsePauliOp` member of `HamiltonianInput` under `TYPE_CHECKING` so a
checker can see it, instead of adding it by runtime mutation.

| | |
| --- | --- |
| Where | `rqutils/sqd.py:202-208` |
| Kind | annotation only — **no runtime behaviour changes at all** |
| Symptom | every correct `sqd(SparsePauliOp, ...)` call is a type error |
| Our cost | 2 `ty: ignore[invalid-argument-type]` in `spinchain/` on calls that are right |

### The cause

```python
type HamiltonianInput = PauliSumXZ | tuple[Sequence[str], Sequence[Number]]
try:
    from qiskit.quantum_info import SparsePauliOp

    HamiltonianInput |= SparsePauliOp        # sqd.py:206
except ImportError:
    pass
```

A `type` statement is evaluated *statically*. The `|=` mutates the runtime alias object, which a
checker never executes, so the `SparsePauliOp` arm is invisible **whether or not qiskit is installed** —
this is not a "qiskit is optional" behaviour, it is unconditional.

Consequence, on the current build:

```text
error[invalid-argument-type]: Argument to function `sqd` is incorrect
   --> spinchain/skqd/sqd_backend.py:302
    | Expected `HamiltonianInput`, found `SparsePauliOp`
```

`spinchain` carries two suppressions for exactly this (`exact.py:112` and `skqd/sqd_backend.py:302`).
They are the bad kind: the calls are correct and *documented* as supported — `sqd`'s own signature
advertises `HamiltonianInput | SparsePauliOp` — so the suppression hides nothing and would also mask a
genuine argument-type error at that call in future.

(A third suppression at `skqd/sqd_backend.py:121` is **unrelated** and we are not asking about it:
`apply_h` annotates `vec`/`states` as numpy arrays while that call site runs inside a `jit`, so
everything is a traced `jax.Array`. A cast there would misdescribe the runtime type rather than fix
anything, so we would rather keep the suppression than have the annotation loosened.)

### Suggested fix, verified

```python
if TYPE_CHECKING:
    from qiskit.quantum_info import SparsePauliOp

    type HamiltonianInput = PauliSumXZ | tuple[Sequence[str], Sequence[Number]] | SparsePauliOp
else:
    type HamiltonianInput = PauliSumXZ | tuple[Sequence[str], Sequence[Number]]
```

Measured with the same checker `spinchain` gates on (`ty`, via `make check`), on a standalone probe
reproducing both shapes side by side:

| alias shape | passing a `SparsePauliOp` |
| --- | --- |
| current (`\|=` at runtime) | `error[invalid-argument-type]` — reproduces our failure exactly |
| proposed (`TYPE_CHECKING`) | **`All checks passed!`** |

Runtime import verified unaffected: the module loads with qiskit present, and the `else` arm keeps the
alias correct where it is absent. If you would rather not branch, a plain
`HamiltonianInput = Union[...]` assignment (not a `type` statement) with a `TYPE_CHECKING` import is
equivalent for checkers and shorter; the branch above is only to keep the runtime alias exactly as
narrow as it is today.

`spinchain` deletes two `ty: ignore` comments if this lands. Nothing else changes on our side.

## D2: nothing reports the ground eigenvalue's multiplicity

**Ask in one line:** let a caller learn that the returned eigenvector is one arbitrary member of a
degenerate ground eigenspace — ideally by returning a basis for it, at minimum by reporting the
multiplicity.

| | |
| --- | --- |
| Where | `rqutils/sqd.py:sqd` (3-tuple return); `ground_locg` returns a single vector by construction |
| Kind | new capability; the minimal form (a multiplicity count) may be much cheaper than an eigenspace solver |
| Blocking us? | **No.** Stated plainly in §"Why this is not urgent" below |

### What is missing

`sqd(..., return_eigvec=True)` returns `(eigval, eigvec, states)`. On a degenerate ground eigenspace
`eigvec` is an arbitrary member and nothing in the return distinguishes that case from a
non-degenerate one:

```python
import numpy as np, jax
jax.config.update("jax_enable_x64", True)
from qiskit.quantum_info import SparsePauliOp
from rqutils.sqd import sqd, hproj

ham = SparsePauliOp.from_list([("IIZZ", 0.5), ("IZZI", 0.5), ("ZZII", 0.5)])
n = 4
states = np.array([[(c >> (n - 1 - i)) & 1 for i in range(n)] for c in range(16)], dtype=np.uint8)

w = np.linalg.eigvalsh(hproj(ham, states).toarray())
print(w[:4])                                     # [-1.5 -1.5 -0.5 -0.5]  -- 2-fold degenerate
print(len(sqd(ham, states, return_eigvec=True)))  # 3 -- nothing says "multiplicity 2"
```

The eigenvector is **deterministic in practice** (4/4 repeated calls gave overlap 1.0 to 1e-12), which
matters below.

### What it costs `spinchain`

Two places, both documented as known limitations rather than worked around:

- `skqd/recovery.py::_site_occupancy` reads site occupancies by squaring the eigenvector's components
  (arXiv:2605.29521 §II.A). Under degeneracy that occupancy is arbitrary, where an **eigenspace average
  would be basis-independent**. Restoring the average needs an eigenspace solver, not a reshape, so the
  docstring records the limitation instead.
- `skqd/recovery.py::recover_configurations` therefore has *observed*, not guaranteed, reproducibility.
  It holds only because `sqd` builds its LOBPCG start vector as a fixed hash of the state index, making
  the chosen member a deterministic function of the subspace. **This has bitten before**: an early
  version gave 5 different recovered bases in 6 identically-seeded runs.

So today our reproducibility rests on an *implementation detail of `_spread_seed`* rather than on
anything either side promises. That is the concrete ask behind D2: we would rather depend on a
documented guarantee than on a hash.

### Two shapes, very different cost

1. **Report multiplicity only** — e.g. an optional count of eigenvalues within `tol` of `eigval`. Enough
   for a caller to *detect* the case and refuse, warn, or fall back. This is the one we would take if
   only one is possible, and it may be nearly free if the Rayleigh-Ritz step already has the
   information.
2. **Return an eigenspace basis** (block LOBPCG, or deflation-and-resolve). Strictly better and lets us
   restore the basis-independent occupancy average, but it is a real algorithmic change to a
   single-vector solver whose whole memory argument is that it holds three vectors. We are not asking
   for this lightly and would understand a refusal.

A third option that would help almost as much and costs nothing: **document the determinism**. If
`sqd`'s member choice is intended to be a deterministic function of `(hamiltonian, states)` and you are
willing to treat a change to it as breaking, saying so converts our "observed" into "guaranteed" without
any code. If it is *not* intended to be stable, we would very much like to know that instead — our
recovery fingerprint test depends on it, and it would be better to learn that from a docstring than
from a moved constant.

### Why this is not urgent

`ground_locg`'s own docstring already covers the degeneracy behaviour of its *internals* thoroughly
(rank-1 null space handling, near-degenerate precision, guarded norms), so this is a reporting gap at
the API boundary, not a correctness bug. And the practical determinism means our runs reproduce today.
We have lived with it since before the prefilter work and can continue to; it is on this list because
"our reproducibility depends on an upstream hash" is the kind of thing that should be written down
somewhere both sides can see.

## Not requested

For completeness, three things we checked and are **not** asking for:

- **Forwarding `precond` through `sqd`.** Moot — `precond` is deleted, and you measured that literal
  Jacobi on the raw indefinite projected `H` fails to converge outright and anti-composes with the
  prefilter. Our `docs/rqutils-precond-request.md` is marked withdrawn.
- **A `dim <= 2` prefilter guard.** Unnecessary after `8358180` fixed the eigenvector-`xinit` defect at
  the root. Verified here: 0/36 failures across dims 2-40, real and complex.
- **Anything about the convergence flag.** `sqd` raises on non-convergence, with a message that
  distinguishes itself from a residual-guard failure. Our `exact.py` and `sqd_backend.py` keep
  independent eigen-residual checks, but as second oracles on the *pair* — not because the flag is
  missing.
