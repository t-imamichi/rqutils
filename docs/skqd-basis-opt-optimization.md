# Implementation spec: two hot-path optimizations for `spinchain.skqd.basis_opt`

**Audience.** An agent implementing this in the `utokyo-saito` / `spinchain` package. You do not
need the `rqutils` repo to do this work; everything below is self-contained.

**Target file.** `spinchain/skqd/basis_opt.py` (906 lines as of 2026-08-27).

**Direction — note this differs from its neighbours.** `docs/rqutils-requests.md` and
`docs/rqutils-precond-request.md` are *inbound*: written from the `spinchain` side, asking `rqutils`
to change. This document is *outbound* — written from `rqutils`, describing work to be done in
`spinchain`. Nothing here asks for an `rqutils` change; see §6 for the one piece that might
eventually become one.

**Status of the measurements.** Every number in this document was measured out-of-tree against
standalone harnesses, on CPU, float64 (`jax_enable_x64`). The parity figures are exact-to-machine-
precision, not tolerances chosen to pass. Harness sources are reproduced in the appendix so you can
re-run them before and after.

> **Scope note.** These are optimizations to `basis_opt` **in place**. Do not move the module into
> `rqutils`: it imports upward (`spinchain.jax_config`, `spinchain.skqd.recovery._matrix_to_ints`,
> `spinchain.skqd.sqd_backend.ground_state`, `spinchain.options.BasisOptOptions`), and its public API
> is qiskit-typed, whereas `rqutils` treats qiskit as an optional extra. Win 1's table builder should
> be written as a self-contained function so it *can* be promoted later, but that is a separate
> decision requiring a second caller.

---

## 0. Before you start

### 0.1 There is one live copy of this file

`spinchain/skqd/basis_opt.py` is it. During the 2026-08-27 evaluation an untracked byte-identical
copy sat at `rqutils/rqutils/basis_opt.py` (906 lines, `diff` empty), copied in only to decide
whether basis optimization belonged in the library; the conclusion was that it does not (see the
scope note above) and that copy has since been deleted. Recorded here so that if it reappears it is
recognized as an evaluation artifact rather than a second implementation to keep in sync.

### 0.2 Establish test coverage first

The coverage `basis_opt` currently has was **not** verified when this spec was written — only the
`spinchain` package directory was available at the time, with no `tests/`. **Check this before
touching `_window_coeffs`.**

Win 1 is a rewrite of the numerical core. If there is no existing test that pins
`basis_optimize`'s returned energy end-to-end, **write one first** and let it fail against a
deliberately broken variant. Section 4 specifies the tests either way; if coverage is thin they are
mandatory rather than confirmatory.

### 0.3 The one trap that will bite you

`Pauli.compose` is **reverse-order**. `G.compose(P)` returns `P @ G`, not `G @ P`.

```
G.compose(P).to_matrix()  vs  G@P  ->  differs by 2.0
G.compose(P).to_matrix()  vs  P@G  ->  differs by 0.0
```

So the product you want, `G @ P_i`, is spelled **`P_i.compose(G)`**. This is exactly what the
existing `_reachable_subgroup` (line 323) already writes — `current.compose(gen)` — so follow that
spelling rather than re-deriving it.

**Why this is dangerous:** getting it backwards leaves `θ = 0` *exact* and corrupts only nonzero
angles. Measured on a 2-qubit window, `G = YZ`:

| term | θ | correct order | inverted order |
|---|---|---|---|
| `ZZ` | 0.0 | 0.0 | **0.0** |
| `ZZ` | 0.6 | 1.1e-16 | **1.1e+00** |
| `YY` | 0.0 | 0.0 | **0.0** |
| `YY` | 0.6 | 1.1e-16 | **1.1e+00** |

A test suite that only checks `C(0) == E_base` passes against the broken version. Section 4.3 is
non-negotiable for this reason.

---

## 1. Win 2 first — it is mechanical, and independent

Do this one first. It is ~10 lines, provably equivalent, and does not touch the numerical core, so
it de-risks the larger change.

### 1.1 The current code

`make_scalable_cost`, lines 549–574. The cost body:

```python
def cost(theta: jax.Array) -> jax.Array:
    coeffs = _rotated_coeffs(theta, plans, n_labels)
    return jnp.sum(jnp.conj(alpha[rows]) * coeffs[terms] * phases * alpha[cols]).real
```

`rows`, `cols`, `terms`, `phases` come from `_contraction_triples` (line 496) and all share one
length, which is

```
len = sum over distinct x_masks X of  |{labels with mask X}| * |{c in codes : c ^ X in codes}|
```

For an XXZ label universe the `x_mask == 0` group alone contributes `n_diag_labels * len(codes)`.
With `n_labels ~ 3000` and `len(codes) = 500` this is order 10^5–10^6 entries, each gathered and
multiplied **every L-BFGS iteration**.

### 1.2 The observation

`alpha` is **fixed** — that is the premise of Eq. 10 (`alpha` held FIXED, module docstring line 9).
It is captured by closure at line 550 and never varies across iterations or restarts. Therefore
`conj(alpha[rows]) * phases * alpha[cols]` is entirely `theta`-independent.

And because the expression is a **full reduction to one scalar** (`jnp.sum(...).real`, no
segment structure in the output), the `coeffs[terms]` gather can be transposed into a one-time
segment-sum over `terms`.

### 1.3 The change

```python
# --- once, at setup inside make_scalable_cost, after _contraction_triples ---
# theta-independent: alpha is fixed by Eq. 10, so fold it, the Pauli phases, and the
# term-index gather into one length-n_labels weight vector. Per-iteration work then
# tracks n_labels (thousands) instead of len(terms) (1e5-1e6).
weights = jnp.zeros(n_labels, dtype=jnp.complex128).at[terms].add(
    jnp.conj(alpha[rows]) * phases * alpha[cols]
)

def cost(theta: jax.Array) -> jax.Array:
    return jnp.sum(_rotated_coeffs(theta, plans, n_labels) * weights).real
```

`rows`, `cols`, `terms`, `phases` are then dead in `cost` and leave the device entirely.

### 1.4 Measured

`n_labels=3000`, `L=500`, `nnz=250,000`, jitted, 100 reps:

| | per iteration |
|---|---|
| current | 0.19–0.38 ms |
| folded | 0.0209–0.0210 ms |
| **speedup** | **9–18×** |

Value relative difference 3.2e-16 / 7.1e-16 across two runs; `jax.grad` maxdiff 2.2e-17 in both.

The folded timing is stable to three digits; the *baseline* is what varies with machine load, so
treat 9× as the conservative figure and re-run A.3 on your own hardware before quoting one. The
correctness figures do not move.

### 1.5 Follow-on (optional, same PR or later)

With `weights` as the only consumer, `_contraction_triples` (line 496) no longer needs to
materialize four Python lists of 10^5–10^6 entries and then `jnp.asarray` them (line 545 —
`jnp.asarray` on a Python list of `complex` is the slowest route into a device array). It can
accumulate directly into a NumPy `weights` array in the existing loop at lines 530–540 and return
that. This is setup cost, not per-iteration, so it is a convenience not a speedup — but it removes
the largest transient allocation in the module.

If you do this, keep `_pauli_entry` (line 210) as the single place the row-vs-column sign
convention lives. Do not inline it.

---

## 2. Win 1 — replace the dense window conjugation with a coefficient recursion

### 2.1 The current code and why it is expensive

`_window_coeffs`, lines 426–469. Per term, per L-BFGS iteration:

1. `eye = jnp.eye(dim)` — dense `(2^w, 2^w)`, `dim = 2^w`, `w = term_width + 2*layers`
2. `jax.lax.scan` carrying that dense matrix; each step is a gather + scale, `O(dim^2)` per block
3. `conj = unitary @ local @ unitary.conj().T` — **two dense matmuls, `O(dim^3)`**
4. project onto the subgroup, reading `len(reachable) * dim` entries of the `dim^2` result

Step 3 dominates and is almost entirely wasted. Measured sizes (n=20, interior term):

| layers | w | 2^w | 4^w | \|reach\| (ZZ) | fraction of result read |
|---|---|---|---|---|---|
| 1 | 4 | 16 | 256 | 2 | 0.78% |
| 2 | 6 | 64 | 4,096 | 6 | 0.15% |
| 3 | 8 | 256 | 65,536 | 8 | 0.012% |
| 4 | 10 | 1,024 | 1,048,576 | 10 | **0.001%** |

`|reach|` grows **linearly** in `layers`; the dense window grows as `4^(2 + 2*layers)`. At
`layers = 4` the code performs ~2 GFLOP of complex matmul per term to extract 10 numbers.

There is a second cost: the `lax.scan` at line 463 carries a `(dim, dim)` complex128 residual for
reverse mode. At `layers=4`, 18 blocks, `dim=1024`, that is `18 * 1024^2 * 16 B ≈ 300 MB` **per
vmap lane** unless XLA rematerializes. `jax.lax.scan` does not checkpoint by default. This may be
the real `layers=4` wall today, and Win 1 removes it.

### 2.2 The identity

For a Pauli generator `G` (which squares to the identity) and a Pauli `P`:

```
exp(-i t/2 G) P exp(+i t/2 G) = P                          if [G, P] = 0
                              = cos(t) P - i sin(t) (G P)   otherwise
```

`G P` is again a Pauli. So conjugating by `k` blocks maps a coefficient vector over the reachable
subgroup to another such vector — **a product of 2×2 rotations, never a matrix**.

The reachable set is the closure of `{P}` under left-multiplication by non-commuting generators.
It depends only on **commutation**, never on `theta` — which the module already relies on
(`_reachable_subgroup`, lines 296–328, already computed and cached at lines 352/359–360). So the
commutation pattern and the phase factors are precompute.

### 2.3 What to precompute

For each term, given its window Pauli `P`, its generator list `gens`, and its closure `labels`
(all of which `_term_plans` already has), build three tables of shape `(K, n_blocks)` where
`K = len(labels)`:

| table | dtype | meaning |
|---|---|---|
| `commutes[i, g]` | bool | whether `P_labels[i]` commutes with `gens[g]` |
| `target[i, g]` | int32 | index in `labels` of `phaseless(P_i · G_g)` |
| `mu[i, g]` | complex128 | phase such that `G_g @ P_i == mu * P_target[i,g]` |

Reference implementation (host-side NumPy + qiskit, runs once per distinct pattern):

```python
def _closure_tables(labels, generators):
    """(commutes, target, mu) for the closure `labels` under `generators`.

    mu[i, g] is the phase with `G_g @ P_i == mu[i, g] * P_target[i, g]`, so a single
    conjugation step is `cos(t) c_i` on element i plus `-i sin(t) mu c_i` on target.

    `Pauli.compose` is REVERSE-ORDER: `P_i.compose(G)` is `G @ P_i`, which is the product
    wanted here. Spelling it the other way leaves theta = 0 exact and corrupts only
    nonzero angles -- see `_pauli_entry` for the same class of trap.
    """
    index = {label: i for i, label in enumerate(labels)}
    n_labels, n_gens = len(labels), len(generators)
    commutes = np.zeros((n_labels, n_gens), dtype=bool)
    target = np.arange(n_labels)[:, None].repeat(n_gens, 1).copy()
    mu = np.ones((n_labels, n_gens), dtype=np.complex128)
    for i, label in enumerate(labels):
        element = Pauli(label)
        for g, gen in enumerate(generators):
            if element.commutes(gen):
                commutes[i, g] = True
                continue
            product = element.compose(gen)          # == gen @ element
            plain = product.to_label().lstrip("-i+")
            reference = Pauli(plain)
            # Global phase from the largest-magnitude entry, NOT from row 0: a Pauli's
            # row 0 has exactly one nonzero and it need not be the one sampled.
            full, ref = product.to_matrix(), reference.to_matrix()
            r, c = np.unravel_index(np.argmax(np.abs(ref)), ref.shape)
            mu[i, g] = full[r, c] / ref[r, c]
            target[i, g] = index[plain]
    return commutes, target, mu
```

**Two traps encoded above, both of which I hit while verifying:**

1. `element.compose(gen)`, not `gen.compose(element)` — see §0.3.
2. Recover `mu` from the **global** argmax of the reference matrix. Sampling row 0 only
   (`prod.to_matrix()[0:1, :]`) can divide by a zero entry.

A cheaper `mu` is available in closed form from the symplectic representation if you want to avoid
`to_matrix()` entirely (this is precompute, so it is not required): the phase of a Pauli product
follows from the `x`/`z` bit overlaps. I did not verify a closed form, so if you take that route,
test it against the `to_matrix()` version above rather than trusting it.

### 2.4 The width-group layout — read this carefully

`_term_plans` batches terms by light-cone **width** (see `_WidthGroup`, lines 259–294) so one
`vmap` covers each width. Keep that; it was measured and the docstring at lines 259–268 explains
why a per-term Python loop is worse.

But there is a subtlety that a naive rewrite gets wrong. The current code builds
`local_labels` as the **union** of per-term closures across the width group (lines 370–373), and
that union is much larger than any single term's closure. Measured, n=12, layers=2, XXZ terms:

| w | dim | n_terms | \|union\| | per-term \|reach\| | max \|reach\| |
|---|---|---|---|---|---|
| 4 | 16 | 6 | 28 | 4–6 | 6 |
| 5 | 32 | 6 | 35 | 5–10 | 10 |
| 6 | 64 | 21 | 27 | 6–15 | 15 |

If you size the recursion state by the union you carry 3–5× more state than needed. **Size it by
`max(|reach|)` within the width group instead**, and keep a per-term map from local closure index
into the shared label universe.

So per width group, the arrays become:

| array | shape | notes |
|---|---|---|
| `param_indices` | `(n_terms, n_blocks)` | unchanged from today |
| `active` | `(n_terms, n_blocks)` | unchanged; 1.0 real block, 0.0 padding |
| `commutes` | `(n_terms, K, n_blocks)` | **padded rows must be `True`** |
| `target` | `(n_terms, K, n_blocks)` | padded rows point at themselves |
| `mu` | `(n_terms, K, n_blocks)` | padded entries 1.0 |
| `start` | `(n_terms, K)` | one-hot at the term's own label, **scaled by its coefficient** |
| `slots` | `(n_terms, K)` | local index → universe index, for the existing scatter |

with `K = max(len(reach) for terms in group)`, `n_blocks = max(len(gens) ...)` as today.

**Padding rules.** Two independent kinds:

- **Block padding** (a term whose cone is clipped by a chain end has fewer blocks): keep the
  existing `active` mechanism, which zeroes the *angle*. `cos(0) = 1`, `sin(0) = 0`, so the step is
  the identity. Do not skip the factor.
- **Closure padding** (a term whose `|reach| < K`): set `commutes = True` on rows `k >= len(reach)`
  so they are carried unchanged, `target` to self, `mu` to 1.0, and `start = 0` there. A zero
  coefficient carried through an identity action stays zero.

`_WidthGroup.local` — the dense `(n_terms, 2^w, 2^w)` field (declared line 289, built line 364) —
**is no longer needed**. The term's coefficient enters as the scale on `start`. Deleting it also
removes the module's largest block of jit-captured constants (at `layers=4`, ~57 terms ×
`4^10` × 16 B ≈ 960 MB) and resolves the contradiction with `_WidthGroup`'s own docstring at lines
275–280, which claims every Pauli there is stored as a permutation and a phase.

`gen_cols` / `gen_phases` / `basis_cols` / `basis_phases` also become unnecessary in the hot path.
Keep `_sparse_pauli` — `_window_unitaries` (line 641) still uses it on the per-restart path.

### 2.5 The new `_window_coeffs`

```python
def _window_coeffs(theta, indices, commutes, target, mu, active, start):
    """One term's rotated coefficients on its window's Pauli subgroup. ``vmap``-ed per width group.

    ``U P U^dag`` stays inside the reachable subgroup, so this is a recursion on a length-K
    coefficient vector and never forms a ``2^w`` matrix. Per block: elements commuting with the
    generator are carried, the rest mix as ``cos(t) P - i sin(t) (G P)`` -- exact, since a Pauli
    generator squares to the identity.

    ``active`` zeroes a padding block's ANGLE (identity action), and padded closure rows carry
    ``commutes = True`` with a zero coefficient, so one scan shape serves the whole width group.
    """
    angles = active * theta[indices]        # NOTE: full angle, not half -- see below

    def step(coeffs, block):
        angle, comm, tgt, phase = block
        kept = jnp.where(comm, coeffs, jnp.cos(angle) * coeffs)
        moved = jnp.where(comm, 0.0, -1j * jnp.sin(angle) * phase * coeffs)
        return kept + jnp.zeros_like(coeffs).at[tgt].add(moved), None

    return jax.lax.scan(step, start, (angles, commutes.T, target.T, mu.T))[0]
```

**`angles` uses the full `theta`, not `theta / 2`.** The existing code halves it because it builds
`exp(-i t/2 G)` explicitly and applies it twice (once on each side). The conjugation identity
folds both sides into a single `cos(t)` / `sin(t)`. Getting this wrong halves every rotation angle
and is *not* caught by a `θ = 0` test.

`_rotated_coeffs` (line 472) changes only in its `vmap` `in_axes` and the arrays it passes; the
`acc.at[group.slots.ravel()].add(coeffs.ravel())` scatter at line 492 is unchanged.

### 2.6 Measured

Single interior `ZZ` term, n=20, jitted, float64.

Cost only, 200 reps:

| layers | w | dim | \|reach\| | blocks | dense | sparse | speedup | maxdiff |
|---|---|---|---|---|---|---|---|---|
| 1 | 4 | 16 | 2 | 1 | 0.0187 ms | 0.0039 ms | 4.8× | 5.6e-17 |
| 2 | 6 | 64 | 6 | 5 | 0.1603 ms | 0.0047 ms | 34× | 2.8e-16 |
| 3 | 8 | 256 | 8 | 10 | 2.0863 ms | 0.0052 ms | 403× | 2.8e-16 |
| 4 | 10 | 1024 | 10 | 18 | 68.2237 ms | 0.0061 ms | **11,125×** | 1.1e-16 |

`jax.value_and_grad` — what L-BFGS actually calls — 100 reps:

| layers | dense | sparse | speedup | grad maxdiff |
|---|---|---|---|---|
| 1 | 0.0432 ms | 0.0063 ms | 6.8× | 0.0 |
| 2 | 0.4743 ms | 0.0090 ms | 53× | 1.7e-16 |
| 3 | 6.9157 ms | 0.0482 ms | 143× | 1.1e-15 |
| 4 | 221.1915 ms | 0.0594 ms | **3,722×** | 1.7e-15 |

Full width-group `vmap` including padding, n=12, layers=2, all 33 XXZ terms across three width
groups, each compared against the dense per-term reference: **overall maxdiff 4.4e-16**.

**Calibrate expectations by depth.** At `layers=1` the win is ~5–7×; the payoff is superlinear in
depth and only becomes dramatic at 3–4. If the shipped `config/skqd.toml` uses `layers=1` or `2`,
the realistic end-to-end gain is the 34–53× range, further diluted by the per-restart
`rotate_operator` + `ground_state` tail (§3), which this does not touch.

---

## 3. Two adjacent wins this unlocks, not required

Both are **per-restart**, not per-iteration, so they matter only if the restart tail dominates
after Win 1. Measure before doing either.

1. **`_rotate_term` (line 613)** calls `SparsePauliOp.from_operator` on a `2^w` window, which
   enumerates the window's full `4^w` Pauli basis. At `layers=4` that is 1,048,576 labels when at
   most `|reach|` (≤45) can be nonzero — and `_reachable_subgroup` already knows which. Building the
   `SparsePauliOp` from the closure labels directly skips the enumeration. The module docstring at
   lines 61–64 already notes the full basis carries "~341× of exact zeros".

2. **`rotate_operator` at line 868** is called without a shared `window_unitary`, so each restart
   builds its own cache. That is *correct* — each restart has a different `theta`, as documented at
   lines 647–651 — so this is not a bug. But `rotate_observables` (line 905) already shares one
   cache across all `3*(n-1)` currents; nothing more to win there.

Do **not** attempt: `_reachable_subgroup`'s label round-trips (precompute-only, ~1247 conversions
worst case at layers=4, cached per distinct pattern — irrelevant to the hot path).

---

## 4. Verification

Run **all** of these. 4.3 is the one that catches the trap most likely to slip through.

### 4.1 Parity against the existing dense path

`projected_matrix` (line 115) is a deliberate independent reference — it uses a single whole-chain
window, sharing no code with the per-term path, and its docstring says so. Use it, plus a
direct dense reconstruction:

```
rebuilt = sum(c_k * Pauli(label_k).to_matrix()) == U @ (coeff * P_window) @ U^dag
```

Cover, for `layers` in 1..4 and at random `theta`:

- a 2-local interior term (`ZZ`, `XX`, `YY`)
- a 1-local term (`Z`, `X`) — different width, different `|reach|`
- a **chain-end term**, whose cone is clipped, so block padding is exercised
- an **identity term**, which returns the degenerate `(0, 0)` window (`_light_cone`, line 155)
- a width group containing terms with **different `|reach|`**, so closure padding is exercised
  (n=12, layers=2 gives `|reach|` 5 and 10 in one group — see §2.4)

Expected: ≤1e-15.

### 4.2 Gradient parity

Value parity is not sufficient. Check `jax.value_and_grad` against the pre-change implementation,
and against central finite differences. Expected ≤1e-15 vs the old code, ~1e-10 vs finite
differences.

### 4.3 The `θ = 0` trap — mandatory

`θ = 0` is exact under **both** the correct and the inverted `compose` order, and under both the
correct and the halved angle. A suite that only asserts `C(0) == E_base` passes against a broken
implementation.

Assert at **nonzero** `theta`, on a term with an odd number of `Y` factors, that the result differs
from the deliberately-inverted variant. Measured discriminating case: 2-qubit window, `G = YZ`,
`P = YY`, `t = 0.6` — correct 1.1e-16, inverted 1.1e+00.

Also assert the angle convention: compare against the old implementation at a single nonzero
`theta`. A halved angle is silent at `θ = 0` and differs by `O(θ²)` for small `θ`, so pick
`theta ~ 1.0`, not `1e-3`.

### 4.4 Mutation testing

Revert each change in place (not in a copy — an editable install points at the original) and
confirm the new tests fail. Mutate `@jax.jit` code in a **fresh subprocess**, or both arms reuse
one compiled kernel.

Specific mutants that must be caught:

- `element.compose(gen)` → `gen.compose(element)`
- `active * theta[indices]` → `active * theta[indices] / 2`
- `mu` recovered from row 0 instead of the global argmax
- padded closure rows set to `commutes = False`
- `start` one-hot not scaled by the term coefficient

### 4.5 End-to-end

`basis_optimize` on a small chain (n ≤ 10) with `restarts >= 1`:

- Eq. 12 still holds: `E' <= C(theta*) <= C(0) = eigval`. The existing guard at line 882 raises if
  the energy rises; it must not fire.
- Reported `eigval`, `best_start`, and `iterations` unchanged from the pre-change implementation
  at the same seed. `theta*` may differ in the last bits, which can change `iterations` — if it
  does, assert on the energy and `best_start` and note the tolerance rather than loosening silently.
- `rotate_observables` output unchanged: the rotated `js` must still travel with the eigenvector
  (`BasisOptResult` docstring, lines 772–784). An eigenvector of the rotated operator contracted
  against unrotated observables gives a plausible wrong spin current *with the energy still
  correct*, so check a current explicitly, not just the energy.

### 4.6 Lint and types

Run whatever the package uses (`ruff check` / `ruff format` / type checker) on the changed files.

Note **B023** (a lambda or closure in a `for` loop capturing the loop variable) is endemic to
table-building code like `_closure_tables`. Fix by binding as a default argument, not by
restructuring the loop. `ruff --fix` does not resolve it.

---

## 5. Suggested sequencing

1. Establish/extend test coverage (§0.2) — especially an end-to-end energy assertion.
2. **Win 2** (§1). Small, mechanical, 9×. Verify with §4.1–4.2 and §4.5.
3. **Win 1** (§2), in two steps:
   a. Add `_closure_tables` and its own unit test against a dense reference, with the table
      still unused by the hot path.
   b. Rewrite `_window_coeffs` / `_term_plans` to consume it; delete `_WidthGroup.local` and the
      now-dead `gen_*` / `basis_*` fields.
4. Re-measure end-to-end at the `layers` value your config actually ships (§2.6 caveat).
5. Only then consider §3.

Keep the two wins in separate commits — they are independent, and Win 2's correctness argument is
much simpler than Win 1's.

---

## 6. If any of this should later move into `rqutils`

Only one piece is a candidate, and **not yet**. The `(commutes, target, mu)` tables from §2.3 are
pure Pauli group theory — no subspace, no chain, no spin current, no `theta`. `rqutils` has no Pauli
product, commutation, or group-closure primitive today (`sqd._z_parity` is Z-only and private;
`paulis/general.py`'s "compose" mentions are comments on dense construction). The tables are also
host-side precompute, which is the one regime where `rqutils`' optional-qiskit guard pattern
(`HAS_QISKIT`, `pytest.importorskip`) applies cleanly.

Three reasons to build it in `spinchain` first regardless:

1. **The oracle that matters lives here.** The test worth trusting is not "does the closure table
   look right" but "does `basis_optimize` return the same energy and `best_start`" — which can only
   run where `basis_optimize` lives.
2. **`docs/rqutils-requests.md` is evidence-first by construction.** Every item there states the
   call site, a self-contained reproduction, and what `spinchain` deletes if it lands. This spec has
   the reproduction and the speedups but no integrated deletion count yet. A1's history is the
   cautionary case: it shipped, was recorded as adopted, and the 54-line deletion is still not in
   the live file.
3. **One caller is not a library.** `CLAUDE.md` records that `compose`, `truncate`, `l0_projector`
   and `symmetry` were removed from `paulis/general.py` as dead code, with an instruction not to
   reintroduce a caller expecting them. A closure API with exactly one consumer, in another repo,
   walks back into that.

**Promote when** either a second caller appears, or the integrated speedup is measured and you want
the primitive under `rqutils`' test suite and Sphinx reference. At that point it is an A1-style
request with real numbers attached.

Separately: `metal`'s segment-sum `diagonals()` is the `while_loop`-free replacement for
`_accumulate_diagonal` (`rqutils/sqd.py:1230` holds the single `lax.while_loop` that makes the whole
diagonal chain non-differentiable, and is the reason `_contraction_triples` is hand-rolled at all).
It is absent from `rqutils`' `dev`/`basis_opt` branches. Cherry-picking it is a larger, separate
question — and if it landed, §1's `weights` vector might be reachable through the shared `expval`
path instead of hand-built triples.

---

## Appendix: harnesses

These are standalone (numpy + qiskit + jax, no `spinchain` import). Re-run before and after.

### A.1 Reachable-subgroup sizes vs dense window (§2.1 table)

```python
import numpy as np
from qiskit.quantum_info import Pauli

def brickwork_pairs(nq, layers):
    pairs = []
    for layer in range(layers):
        pairs += [(i, i + 1) for i in range(layer % 2, nq - 1, 2)]
    return pairs

def block_generators(pairs, lo, hi):
    width = hi - lo + 1
    out = []
    for index, (a, b) in enumerate(pairs):
        if lo <= a and b <= hi:
            z = np.zeros(width, bool); x = np.zeros(width, bool)
            z[a - lo] = True
            z[b - lo] = x[b - lo] = True
            out.append((index, Pauli((z, x))))
    return out

def phaseless(p):
    return p.to_label().lstrip("-i+")

def closure(pauli, gens):
    seen = {phaseless(pauli)}
    frontier = [Pauli(phaseless(pauli))]
    while frontier:
        nxt = []
        for cur in frontier:
            for g in gens:
                if cur.commutes(g):
                    continue
                label = phaseless(cur.compose(g))
                if label not in seen:
                    seen.add(label); nxt.append(Pauli(label))
        frontier = nxt
    return sorted(seen)

def light_cone(p, layers, nq):
    s = np.flatnonzero(p.x | p.z)
    if not len(s):
        return 0, 0
    return max(0, int(s.min()) - layers), min(nq - 1, int(s.max()) + layers)

nq = 20
for layers in (1, 2, 3, 4):
    pairs = brickwork_pairs(nq, layers)
    p = Pauli("I" * (nq - 11) + "ZZ" + "I" * 9)
    lo, hi = light_cone(p, layers, nq)
    w = hi - lo + 1
    reach = closure(p[lo:hi + 1], [g for _, g in block_generators(pairs, lo, hi)])
    print(f"layers={layers} w={w} 2^w={2**w} 4^w={4**w} |reach|={len(reach)} "
          f"frac={len(reach)/4**w:.2e}")
```

### A.2 Dense-vs-sparse parity and timing (§2.6 tables)

Full source: reproduce `_window_coeffs` both ways over the same precomputed tables and compare.
The key pieces are `_closure_tables` from §2.3, `sparse_pauli` below, and the two jitted functions.

```python
def sparse_pauli(pauli, dim):
    x_mask = sum(1 << i for i, b in enumerate(pauli.x) if b)
    z_mask = sum(1 << i for i, b in enumerate(pauli.z) if b)
    lead = (-1.0j) ** ((x_mask & z_mask).bit_count() % 4)
    rows = np.arange(dim, dtype=np.int64)
    return rows ^ x_mask, lead * (1.0 - 2.0 * (np.bitwise_count(rows & z_mask) & 1))

def make_dense(d):
    dim = d['dim']; eye = jnp.eye(dim, dtype=jnp.complex128)
    def f(theta):
        def step(U, blk):
            a, cols, ph = blk
            return jnp.cos(a) * U - 1j * jnp.sin(a) * (ph[:, None] * U[cols]), None
        U = jax.lax.scan(step, eye, (theta / 2, d['gcols'], d['gph']))[0]
        conj = U @ d['local'] @ U.conj().T
        picked = conj[jnp.arange(dim)[None, :], d['bcols']]
        return jnp.sum(d['bph'] * picked, axis=1) / dim
    return jax.jit(f)

def make_sparse(d):
    def f(theta):
        def step(c, blk):
            t, cm, tg, m = blk
            keep = jnp.where(cm, c, jnp.cos(t) * c)
            moved = jnp.where(cm, 0.0, -1j * jnp.sin(t) * m * c)
            return keep + jnp.zeros_like(c).at[tg].add(moved), None
        return jax.lax.scan(step, d['start'],
                            (theta, d['comm'].T, d['tgt'].T, d['mu'].T))[0]
    return jax.jit(f)
```

Parity check: `rebuilt = sum(c_k * Pauli(label_k).to_matrix())` against
`U @ local @ U.conj().T`. Expect ≤1e-15. Set `jax.config.update('jax_enable_x64', True)` before
any array is created — in 32-bit these tolerances are meaningless.

### A.3 Weight-folding parity (§1.4)

```python
import numpy as np, jax, jax.numpy as jnp
jax.config.update('jax_enable_x64', True)
rng = np.random.default_rng(0)
n_labels, L = 3000, 500
nnz = n_labels * L // 6
T = jnp.asarray(rng.integers(0, n_labels, nnz))
R = jnp.asarray(rng.integers(0, L, nnz))
C = jnp.asarray(rng.integers(0, L, nnz))
P = jnp.asarray(rng.normal(size=nnz) + 1j * rng.normal(size=nnz))
a = rng.normal(size=L) + 1j * rng.normal(size=L); a /= np.linalg.norm(a)
A = jnp.asarray(a)

current = lambda c: jnp.sum(jnp.conj(A[R]) * c[T] * P * A[C]).real
W = jnp.zeros(n_labels, dtype=jnp.complex128).at[T].add(jnp.conj(A[R]) * P * A[C])
folded = lambda c: jnp.sum(c * W).real
# compare jax.jit of both, and jax.grad of both. Expect ~1e-16 / ~1e-17.
```

### A.4 The `θ = 0` trap (§4.3)

```python
import numpy as np
from qiskit.quantum_info import Pauli
w, dim = 2, 4
z = np.zeros(w, bool); x = np.zeros(w, bool)
z[0] = True; z[1] = x[1] = True
G = Pauli((z, x))
for label in ("ZZ", "YY"):
    P = Pauli(label)
    Gm, Pm = G.to_matrix(), P.to_matrix()
    for t in (0.0, 0.6):
        U = np.cos(t / 2) * np.eye(dim) - 1j * np.sin(t / 2) * Gm
        ref = U @ Pm @ U.conj().T
        good = np.cos(t) * Pm - 1j * np.sin(t) * (Gm @ Pm)
        bad = np.cos(t) * Pm - 1j * np.sin(t) * (Pm @ Gm)
        print(label, t, np.abs(good - ref).max(), np.abs(bad - ref).max())
```

Expected: at `t=0.0` both 0.0; at `t=0.6` correct ~1e-16 and inverted ~1.1.
