# `sqd` workflow: from input to checked eigenpair

A map of one `sqd(...)` call, stage by stage, with the measured reason for each choice behind a
pointer. The authoritative contract is the docstrings of `rqutils/sqd/__init__.py`,
`rqutils/sqd/_core.py` and `rqutils/ground_locg.py`; this file only puts them in order.

```text
sqd(hamiltonian, states, matvec=, prefilter=, atol=, rtol=, ...)
 ├─ 1. _sqd_inputs      validate, PauliSumXZ.from_paulisum, pack + pad states   (host)
 ├─ 2. _solve_sqd
 │    ├─ "pairs":  uniquify → host search + build operator → _run_sparse        (host build, jitted solve)
 │    └─ else:     run_sqd (jitted): uniquify → xsources? → diagonals? → _solve
 │         _solve:  initial vector → ground_locg(prefilter → LOBPCG) → residual check
 └─ 3. _checked_eigval  raise on non-convergence or a failed residual check     (host)
```

## 1. Inputs (`_sqd_inputs`, host)

- **Validation first**: `matvec` must be a `Matvec` member (a plain string raises `TypeError`),
  `prefilter` a `(degree, cycles)` pair of non-negative ints, and `atol`/`rtol` are checked against
  `Σ|c_k|` here — the outermost point where it is concrete, since `run_sqd` is jitted.
- **Hamiltonian** → `PauliSumXZ`: terms grouped by X signature (`J` groups, `K` Z signatures each), the
  `(-i)^{x·z}` phase folded into the coefficients, signatures bit-packed with one pad bit at position 0.
  Real groups first (`num_real_groups`); `term_counts`, `identity_first` and `zfree_first` are static
  metadata the kernels trust.
- **States** → packed `uint8` rows `ceil((n+1)/8)` wide, pad bit at position 0 aligned with the
  signatures. Skipped with `packed=True`.
- **`states_size`**: next power of two by default (bounds retraces to `O(log N)`), capped at
  `2^31 - 1` (int32 indices), rounded up to a multiple of `mesh.size` under a mesh. The input is padded
  to it, so the shape is part of the jit cache key.

## 2. Subspace (`uniquify_states`)

Sort and deduplicate on the device into a fixed `states_size` array. Leftover slots are filler rows
(`255`, detected by the pad bit `states_u[:, 0] >> 7`). The result **must be lex-sorted**, because
`get_xsource` is a binary search; two key paths are chosen statically on width (`uint64` for ≤ 8 bytes,
lexicographic beyond) — a correctness boundary.

## 3. The operator: four `matvec` kernels

The projected matvec is

```text
v' = Σ_j C^(j) ∘ B[x^(j)](v),    C^(j) = Σ_k α^(j,k) (-i)^{x·z} (-1)^{z^(j,k)·s}
```

where `B[x]` gathers `v` at the source index of `s ⊕ x` (zero when the source is outside the subspace),
and `C^(j)` is the composed diagonal of group `j`. Both depend only on the states, so they can be
cached. The kernels are named by what they store:

| `Matvec`             | Stores                                       | Per-matvec work                              | Memory                    |
|----------------------|----------------------------------------------|----------------------------------------------|---------------------------|
| `ONTHEFLY`           | nothing                                      | `J` binary searches + all diagonals          | floor, 120 B/slot         |
| `INDICES` (default)  | source indices `[j^i]` per group             | diagonals                                    | `+ 4·J·N`                 |
| `TABLES`             | indices **and** diagonals `C^(j)`            | gather × scale only                          | `+ 8 or 16 · J·N`, frees `S` |
| `PAIRS`              | each in-subspace transition `(a<b, d)` once  | scatter-add `v'_a += d v_b`, `v'_b += d̄ v_a` | ≈ hits only               |

Notes on choosing (module docstring "Caching"; `poc/sparse/gpu.md` §9):

- **Not a symmetric dial.** `get_xsource` setup is 66–97% of an `ONTHEFLY` solve because it is paid
  every matvec; a stored kernel pays it once (4.5–8.4% of the solve). Use `ONTHEFLY` only when nothing
  else fits.
- **`PAIRS` is the fastest single-device kernel**: 3.1–4.0× `TABLES` on CPU at `INDICES`' memory;
  1.1–1.8× `INDICES` on GPU at ≈0.6× its memory. It is `sqd`-only (`run_sqd`/`apply_h` reject it),
  because its entry count is data-dependent: `_solve_sqd` builds it **host-side before the jitted
  solve** in `2^15` (CPU) / `2^19` (GPU) entry chunks, the chunk count rounded to a size class so the
  solve recompiles per class. Exactly-zero factors are dropped (`_drop_zeros`; 86% of `type1`'s pairs).
- **Under a mesh `INDICES` is the measured choice.** `PAIRS` runs term-parallel there (one contiguous
  slice of entries per device, one all-gather + one reduce-scatter per matvec), but is untimed on real
  devices and every process still builds the whole host operator.
- **`K` and `J` decide the fit**, and are properties of the Hamiltonian: `J = O(n)` for a spin chain,
  `O(n^4)` for Jordan–Wigner fermions, where the `4·J·N` index table outgrows a device. Quote `K` with
  any memory figure.
- `INDICES`/`ONTHEFLY` sum diagonals over fixed trip counts bucketed by `term_counts`, with the identity
  group's diagonal cached once per solve (`_apply_buckets`; 3.2–4.5× GPU, 1.8–1.9× CPU,
  `poc/dense-tune.md`). `TABLES` alone splits real groups into float64 (`_group_parts`).
- On CUDA, `_scan_add` carries a complex accumulator as separate real/imaginary parts (`PAIRS`);
  `poc/sparse/split.md`.

Under a mesh, once the source indices are computed (no further sort or search), `states_u` is
resharded `P('x')`, and every `apply_*` preserves the input vector's sharding — the contract that makes
`ground_locg` sharding-transparent.

## 4. Initial vector (`_solve`)

A deterministic pseudo-random spread over the subspace (`_spread_seed`), **never a one-hot**: a one-hot
cannot leave its connected component, and returned a block's minimum on a disconnected Hamiltonian. When
group 0 is the identity, the minimum-diagonal state gets extra weight **with the seed's own sign**
(`jnp.sign`, since the seed may be complex), applied by an iota mask rather than `.at[i]`, which would
all-gather.

## 5. Chebyshev prefilter (`ground_locg._chebyshev_prefilter`)

`sqd` defaults to `prefilter=(32, 2)`: two cycles of a degree-32 Chebyshev polynomial applied to the
initial vector before LOBPCG starts, costing `cycles · (degree + 1)` = 66 matvecs and three live
vectors.

- Each cycle maps `[θ, hi]` onto `[-1, 1]` and applies `T_degree`, damping that band by `1/T_degree`;
  anything below `θ` grows like `cosh`, so the ground direction comes out amplified.
- **The lower edge `θ` is the current Rayleigh quotient, re-read every cycle.** It starts above `λ_0`
  and descends, so it never brackets the target out — an accurate `λ_1` estimate did, on a tight gap.
- **The upper edge `hi` must be a true bound on `λ_max`.** `sqd` passes `Σ|c_k|` (Pauli strings are
  unitary; projection only shrinks the spectrum). No matvec-only iteration can supply a bound, and an
  under-estimate returned an **excited** eigenpair with `converged=True`. Over-estimating only costs
  resolution.
- Measured gain through `sqd`: **1.49× median end-to-end (min 1.15×)**, single-device CPU. Quote that,
  not the 2.43× dense wall-clock or the 5.02× iteration count. `prefilter=None` restores the unfiltered
  graph exactly. Filtering alone plateaus as `θ → λ_0`, which is why it hands off to LOBPCG.

Tuning tables: `markdown/locg-chebyshev-prefilter.md`.

## 6. Eigensolver (`ground_locg`)

Single-vector LOBPCG: Rayleigh–Ritz over `{x_i, y_i, p = r_i/|r_i|}`, solved analytically
(`eigenpair_3x3`, Cardano) rather than by `eigh`. The working set is 7 `O(N)` vectors — the algorithmic
minimum for a 3-dim basis.

- **Convergence**: `‖r‖ < max(atol, rtol · (‖Ax‖ + |θ|))`, either arm suffices. `atol` defaults to
  `0.0`; `rtol=None` resolves to `4·eps` of the promoted dtype. No `n` factor in `rtol` (it would
  accept the first iterate at large `n`).
- **`batch_matvec=True`** (always, from `run_sqd`): each iteration's two independent applications are
  stacked into one `(2, N)` call — 1.15–1.21× end-to-end at `INDICES`, all-gathers per sharded loop
  body 3 → 2, and −16 B/slot temp because one gather is shared.
- Two re-orthogonalization passes, fixed; balancing and zero-direction masks are load-bearing
  (`markdown/locg.md`).
- LOBPCG rather than Davidson: at matched memory LOBPCG wins on the non-diagonally-dominant `H` that
  `sqd` projects (`poc/davidson.md`).

## 7. Independent check and return (`_checked_eigval`)

- **Non-convergence raises `RuntimeError`**: an unconverged `θ` is a finite variational upper bound,
  indistinguishable from a right answer. Raise `maxiter` first.
- **The residual `‖Hv − Ev‖` is recomputed after every solve**, with diagonals always rebuilt so no
  cached table vouches for itself (cached source indices are reused; `PAIRS` uses `_sparse_residual`
  from its unfiltered searched pairs on the host). Above `10 ×` the convergence bound (or the residual
  floor `4·eps·Σ|c_k|`) it raises `EigenpairCheckError`, whose message never says "did not converge".
- Host reads go through `_host_scalar`, so a multi-process mesh can read its own eigenvalue.
- **Return**: `eigval` alone, or `(eigval, eigvec, basis)` with `return_eigvec=True`; filler slots are
  stripped, and `basis` is packed exactly when the input was (`packed=`). A degenerate ground state
  yields one arbitrary but deterministic member of the eigenspace.

## Where the numbers live

| Topic                         | Pointer                                                         |
|-------------------------------|-----------------------------------------------------------------|
| Kernel memory/time tables     | `NOTES.md` (n=100 memory, n=22 timing); `poc/sparse/gpu.md` §9  |
| `PAIRS` design and scaling    | `poc/sparse/pairs.md`, `poc/sparse/tune.md` §3                  |
| `PAIRS` under a mesh          | `markdown/pairs-mesh-proposal.md` §4.1                          |
| Bucketed diagonals            | `poc/dense-tune.md`, `poc/dense-codes.md`                       |
| Prefilter                     | `markdown/locg-chebyshev-prefilter.md`; `NOTES.md` "sqd.sqd: the prefilter default" |
| Solver defects and guards     | `markdown/locg.md`                                              |
| Sharding the state list       | `poc/partition-states.md`                                       |
