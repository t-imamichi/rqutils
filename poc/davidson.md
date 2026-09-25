# Davidson vs `ground_locg`

Moved from `NOTES.md` (two entries of 2026-09-02, "The eigensolver"); single-device CPU. §1–§2 used a
synthetic harness that was never committed; §3–§4 come from `poc/davidson_xxz.py` (§7).

Asked because `diaglib` (Molecolab-Pisa, LGPL, `diaglib.f90`) ships both algorithms, so its paper's
headline claim ("Davidson remains the best choice for most applications; LOBPCG is competitive,
especially when memory is an issue") could be checked against this module rather than taken on trust.
**Comparison is by matvec count, not wall-clock**: a 50-line NumPy Davidson against a jitted JAX solver
says nothing about time, and matvecs are the currency for a matrix-free method anyway.

The Davidson in both harnesses is a faithful-but-minimal reimplementation — Jacobi preconditioner, thick
restart at `n_keep = max_dav - 2` (§2), no locking, no Cholesky `ortho_cd` with level shifting, single
eigenpair — adequate for counting matvecs and comparing eigenvalue error and nothing else.

## 1. Memory, read off `diaglib`'s allocations

Not a benchmark. `davidson_driver` allocates `space(n,lda)` *and* `aspace(n,lda)` with
`lda = dim_dav·n_max`, so its footprint is **linear in history depth**; `lobpcg_driver` allocates
`3·n_max` blocks plus temporaries. Per eigenpair:

| | vectors/eigenpair | at N = 18 360 640, 50 states |
| --- | --- | --- |
| Davidson, depth 25 | **51** | 384 GB (paper reports ~356) |
| `diaglib` LOBPCG | 9 | 68 GB (paper reports ~55) |
| **`ground_locg`, m=1** | **8 total** | **1.1 GB** |

The model reproduces the paper's published numbers within 8–23% (the gap is their locking shrinking
`n_act`), so it is sound. `ground_locg` is a further step past `diaglib`'s LOBPCG purely by being
block-size-1. This table does not depend on the reimplemented Davidson.

## 2. Synthetic dense fixtures, `n=512`, unfiltered

**Matvecs at matched memory — the comparison that is not obvious.** Davidson at depth 25 looks dominant,
but that is 54 O(N) vectors against 8. Sweeping depth, median over 5 seeds, `rtol=1e-10`, all arms 5/5
converged:

| | depth 3 (10 vec) | depth 5 (14) | depth 10 (24) | depth 25 (54) | **`ground_locg` (8 vec)** |
| --- | --- | --- | --- | --- | --- |
| diagonally dominant | **390** | 311 | 241 | 213 | **731** |
| dense non-dominant | **580** | 276 | 126 | 103 | **404** |

**The two regimes split, reproducing the paper's conclusion:** at comparable memory Davidson is **1.9×
better on the diagonally dominant** fixture (390 vs 731 — its Jacobi preconditioner exploits the
dominance) and **1.4× worse on the non-dominant** one (580 vs 404). Davidson's headline advantage is
bought with memory, very nearly linearly: on the dense fixture 580 → 103 matvecs (5.6×) costs 10 → 54
vectors (5.4×).

**Accuracy: comparable, and `ground_locg` is better at the floor.** Eigenvalue errors agree within ~1.5×
at usable tolerances. Driving `rtol` to `4·eps`, Davidson reports `conv=False` on both fixtures while
`ground_locg` converges, reaching a **10× lower true residual** on the dense case (2.42e-14 against
2.96e-13). That is the fixed 2-pass orthogonalization earning its keep (`NOTES.md`, "The fixed 2-pass
re-orthogonalization beats `diaglib`'s adaptive loop on its own metric").

**A thick restart is the one place `(AV)u` reuse is legitimate.** The first Davidson written here
restarted on a single Ritz vector; `diaglib` re-seeds with `n_max` (`dcopy(n_max*n,evec,1,space,1)`).
Keeping the `k` lowest Ritz vectors — `V ← V·S[:,:k]`, `AV ← AV·S[:,:k]` — is exact: measured
`‖VkᵀVk − I‖ = 1.8e-15` and `‖AVk − A·Vk‖ = 2.4e-14`. This is the same `(AV)u` form the `Ax`-reuse
investigation (`NOTES.md`, "Reusing `Ax` to cut the matvec count is closed") shows corrupting the
residual, and it is safe **because `S` is orthogonal *and* `V`/`AV` are current, not stale**.
Accumulated staleness is the defect, not the matmul; `‖u‖ = 1` alone was never the sufficient condition.

**Two negatives worth keeping.**

- **The thick restart barely matters at shallow depth** (384 vs 394 at depth 3, identical at 290 on the
  dense fixture), so the suspicion that a thin restart handicapped Davidson was **wrong** — it only helps
  at depth 10–25 (263 vs 266, 210 vs 226).
- **`n_keep = max_dav - 1` stalls outright**: the retained Ritz vectors plus one new direction refill the
  space every iteration, so the subspace never grows — measured 6001 matvecs and error **2.3** (wrong
  answer, non-convergent) at `max_dav=3, n_keep=2`, where `n_keep=1` converges in 374. Any thick restart
  needs `n_keep ≤ max_dav - 2`.

## 3. `xxz_krylov` through `apply_h`, prefilter on both arms

Closes §2's caveat — synthetic fixtures, no filter; CLAUDE.md warns a physically-motivated fixture can
invert a synthetic conclusion. The operator is a periodic XXZ chain (`XX + YY + delta*ZZ` per bond, a
transverse `Bx` per site) projected onto `poc/hash_partition`'s `xxz_krylov` subspace and applied through
`apply_h` at `cache_level=(1, 2)`. n=24, rungs=6, cap=8000 so the cap bites and each seed is a *distinct*
subspace, N=8000 (padded 8192), 5 seeds, `rtol=1e-10`, all arms 5/5 converged, every energy within 6e-13
of `eigsh(tol=0)`.

**`Bx` is inert, not load-bearing as first recorded.** The entry claimed that without `Bx`
magnetization is conserved and the projection is trivially block-diagonal. True of `H`, not of its
projection: `xxz_krylov` is one Hamming-weight sector, so every single-site `X` term projects to zero —
`nnz` and `E0` bit-identical at `bx` = 0.0, 0.5, 3.0 (n=12, dim=380; n=16, dim=1325). Corrected
2026-09-17 (`NOTES.md`, "Warm-starting the growing subspace: four hypotheses eliminated, and the fixture
gate is the result"); the results are unaffected, since `bx` is never swept.

**The filter goes to both arms, and that is the whole design of the comparison.** `_chebyshev_prefilter`
is an operator-agnostic start-vector transform — nothing about it is LOBPCG-specific, so handing it to
one arm measures the filter, not the algorithm. Both arms get the identical filtered vector and are
charged the identical `cycles*(degree+1) = 66` matvecs.

**Matvec counts, median over 5 seeds, excluding the shared 66:**

| | vectors | plain (delta=1.0) | filtered | plain (delta=0.2) | filtered |
| --- | --- | --- | --- | --- | --- |
| **`ground_locg`** | **8** | 156 | **57** | 129 | **48** |
| Davidson depth 2 | 8 | **112** | 67 | **125** | 68 |
| Davidson depth 3 | 10 | 68 | 40 | 74 | 40 |
| Davidson depth 10 | 24 | 47 | 24 | 49 | 25 |
| Davidson depth 25 | 54 | 41 | 23 | 42 | 23 |

**At matched memory the filter is what separates them, in `ground_locg`'s favour.** Depth 2 is the only
Davidson arm holding 8 vectors, and unfiltered it is a near-tie (112 vs 156; 125 vs 129 on the
non-dominant fixture). Filtered, `ground_locg` wins both (57 vs 67, 48 vs 68). Reading any deeper
Davidson column as a win is §2's mistake — depth 3 already buys its 68 with 10 vectors, and depth 25 with
54.

**The filter helps LOBPCG and *hurts* Davidson.** Speedup in matvecs including the filter's own 66,
median over seeds:

| | delta=1.0 | delta=0.2 |
| --- | --- | --- |
| `ground_locg` | **1.27x** (1.09-2.03) | **1.11x** (1.00-1.74) |
| Davidson depth 2 | 0.90x | 0.96x |
| Davidson depth 3 | 0.65x | 0.69x |
| Davidson depth 25 | 0.46x | 0.47x |

Every Davidson arm is a *regression*. The mechanism is the Amdahl argument that caps the Bloom filter:
Davidson converges in 41-125 matvecs, so a fixed 66-matvec precompute cannot pay for itself however much
it improves the start. `ground_locg` needs 129-156, so the same 66 does. **The filter is a poor
discriminator between the algorithms and a good one between operating points** — worth having exactly
when the solver is iteration-hungry, which is the regime the single-vector method is in by construction.

**The regime is set by the anisotropy, measured on the assembled sparse matrix.** The physical
fixture partly inverts the synthetic one, but not via the axis expected: `delta=1.0` gives 45% diagonally
dominant rows at a median `|d|/offsum` of **1.00** — borderline, not the non-dominant regime `sqd` is
claimed to be in — while `delta=0.2` (run at `Bx=2.0`, inert) gives 0.1% and **0.20**. So the
anisotropy, not the subspace, selects the regime, and a default `delta=1.0` XXZ is *not* a non-dominant
fixture. The matched-memory conclusion holds on both.

## 4. Two measurement traps, both silent

- **A Python-level counting wrapper cannot count matvecs through this module.** `body` runs under
  `jax.lax.while_loop` and the prefilter under `jax.lax.scan`, so a counting closure is invoked once per
  *trace*: measured **7 against a true 429** (niter=143), and 3 against the prefilter's 66. It does not
  error and the numbers look plausible. Counts must be analytic — `body_iter0` is 1 matvec, `body_iter1`
  2, `body()` 3, and `niter` counts `body()` calls only, so **matvecs = 3 + 3*niter** (verified by a
  `maxiter=0,1,2,5` sweep, where `niter` returns exactly `maxiter`). The NumPy Davidson's counter is
  real.
- **`cap` must actually bite or the seed does nothing.** At rungs=3 the reachable set is smaller than the
  cap, so every seed returned a bit-identical subspace and the same energy — a "median over 5 seeds" over
  one fixture repeated 5 times. The tell was identical `N=489 E=-26.7560572683` on every row.

## 5. What it means

- **`sqd` is in the row where `ground_locg` wins at matched memory.** Its projected `H` is not diagonally
  dominant — the structural reason `precond` is closed (CLAUDE.md, "Closed investigations") — and `N`
  is the binding constraint. Reassuring rather than surprising, but it had never been measured; §3
  confirms it on a physical fixture through `apply_h`.
- **Davidson's matvec advantage is memory**: 5.6× fewer matvecs for 5.4× the vectors (§2), 51 vectors
  per eigenpair at depth 25 against 8 (§1). At 8 vectors it ties unfiltered and loses filtered (§3).
- **The prefilter favours the single-vector method**: 1.11–1.27× for `ground_locg`, 0.46–0.96× for every
  Davidson arm.
- **Accuracy is not the discriminator**, except at the floor, where `ground_locg` converges at `4·eps`
  and Davidson does not.

## 6. Open

- Single-device CPU, one Hamiltonian family, one subspace size (N=8000). No wall clock.
- `delta=1.0` is borderline dominant (median `|d|/offsum` 1.00); only `delta=0.2` is the clearly
  non-dominant regime. A second Hamiltonian family in that regime is unmeasured.
- Davidson's Jacobi preconditioner reads the diagonal off the assembled sparse operator. A matrix-free
  caller would get it from `get_diagonal`'s identity-X group — free in `sqd`, but not counted as a matvec
  here.
- A full Davidson (locking, `ortho_cd`, multiple eigenpairs) is not compared.

## 7. The script

§3–§4 are `poc/davidson_xxz.py`. It builds the `(1, 2)` `apply_h` matvec per seed, a `hproj` sparse
reference for the Jacobi diagonal and `eigsh(tol=0)`, one shared prefiltered start, then runs
`ground_locg` (warm, analytic count) and the NumPy Davidson (real counter) on both starts. It prints
median matvecs, converged counts, max `|E - exact|` and the filter speedup per arm; Davidson's vector
count is reported as `2·depth + 4`.

| argument | default | meaning |
| --- | --- | --- |
| `--num-qubits` | 20 | chain length (§3 used 24) |
| `--rungs` | 4 | hops from Néel in `xxz_krylov` (§3 used 6) |
| `--cap` | 4000 | subspace cap; must bite, §4 (§3 used 8000) |
| `--delta` | 1.0 | `ZZ` anisotropy; selects the dominance regime |
| `--bx` | 0.5 | transverse field; inert after projection (§3) |
| `--seeds` | 5 | subspace draws, one per seed |
| `--rtol` | 1e-10 | both solvers' relative tolerance |
| `--depths` | `3,5,10,25` | Davidson `max_dav` sweep (§3 used `2,3,10,25`) |
| `--prefilter` | `32,2` | `degree,cycles` of the shared filter (66 matvecs) |
