# Two-level Chebyshev filtering (CS2CF): **not applicable — a paper review, no code written**

**Status: assessed 2026-08-28 by reading the source paper. Not implemented, and not a candidate.** The
method solves a problem this library does not have, and its cheap step is the one `sqd` already does
exactly. No measurement was taken, because the structural mismatch is decidable from the formulation
alone — recorded here so the question is not reopened from the title.

Reference: A. S. Banerjee, L. Lin, P. Suryanarayana, C. Yang, J. E. Pask, *Two-level Chebyshev filter
based complementary subspace method*, J. Chem. Theory Comput. **14**(6), 2930–2946 (2018). This is the
paper `docs/locg-chebyshev-prefilter.md` §7 names as the prefilter's provenance, now also cited as `[5]`
in `ground_locg`'s module docstring.

---

## 1. What the method actually is

Read from the paper (§2.1.3, §2.2.2), not from the title — the two natural guesses about "two-level"
are both wrong.

**It computes a density matrix, not eigenvectors.** The target is the projected density matrix
`P̃ = Σ_i f_i ψ̃_i ψ̃_i^T`, where `f_i` are occupation numbers. The complementary-subspace trick (§2.1.3)
rests on one observation: at electronic temperatures `Θe ≲ 3000 K`, *most* occupation numbers are exactly
1. Split the `Ns` states into `N1` fully-occupied ones and `Nt = Ns − N1` fractionally-occupied ones, and
the fully-occupied block contributes `Σ ψ̃_i ψ̃_i^T` — a projector, needing no individual eigenvectors at
all. So only the `Nt` *fractionally-occupied* states must be resolved, and `Nt << Ns`.

**The two levels are outer/inner, both over blocks:**

- **Outer:** CheFSI computes an orthonormal basis `Y` approximately spanning the occupied subspace of
  the full `H`.
- **Inner:** CheFSI computes the `Nt` topmost states of the projected `H̃ = Y^T H Y`, equivalently the
  `Nt` lowest of `−H̃`.

The inner filter is remarkably cheap — order 4 or lower, ≤5 cycles — because `H̃`'s spectral width is
already small. Its cost is `O(m̃ Ns² Nt)`, dominated by applying the filter to an `Ns × Nt` **block**,
chosen deliberately so the work is GEMM/PBLAS rather than LOBPCG-style vector operations.

## 2. Why it does not transfer

Three independent blockers; the first alone is sufficient.

| # | The method requires | This library |
| --- | --- | --- |
| 1 | A block of `Nt` vectors, filtered as an `Ns × Nt` GEMM | `ground_locg` is a **single-vector** specialization; its entire memory argument is three vectors regardless of iteration count |
| 2 | Many eigenpairs near a Fermi level, with fractional occupations | `sqd` wants **one** eigenpair, the ground state. There is no occupation structure and no `N1`/`Nt` split to exploit |
| 3 | Its outer level to build `Y` and project `H̃ = Y^T H Y` | `sqd` **already has the projected operator** — `hproj(H, states)` is the projection, done exactly, for free, from the sampled subspace |

Blocker 3 is the sharpest point and cuts the other way from what the title suggests: the paper's outer
level exists to *construct* a projected operator, which is precisely the step SQD gets for free from its
bitstring subspace. The paper's expensive machinery buys something `sqd` starts with. What remains — the
inner level — is a block eigensolver for `Nt > 1` states, which is blocker 1.

So the method is not "rejected on measurement"; it is answering a different question. Implementing the
inner level for `Nt = 1` degenerates to exactly what `ground_locg` already does: filter, then solve for
one extremal pair.

## 3. Relationship to what *was* measured

Two prior results are adjacent and neither is this paper. Keeping them distinct matters, because both
could be mistaken for having already answered the question.

- **`docs/deflation-preconditioner.md`** — a *two-level / deflation preconditioner*, tested 2026-08-28
  and rejected (0.68–0.98× wall clock, 8/8 losses, iteration count **up** in every configuration). Same
  "two-level" phrase, different mechanism: a coarse-space preconditioner, not Chebyshev filtering. Its
  failure mode is the transferable part — it improved conditioning without opening the **gap**, and
  iteration count tracks the gap (log-iterations correlates +0.77 with `log(1/gap)`, and −0.34, wrong
  sign, with `κ`).
- **`docs/locg-chebyshev-prefilter.md`** — the single-vector prefilter that shipped, `prefilter=(32, 2)`,
  1.49× median end-to-end through `sqd`. This is the adaptation of ChFSI's *filtering idea* to a
  single-vector solver, and the paper above is its provenance. It is not the paper's algorithm.

## 4. One transferable observation

The paper's inner filter uses **order 4 or lower** on the projected operator, because projection has
already narrowed the spectral width. `sqd` defaults to `degree=32` on an operator that is *also* a
projection. That is a real asymmetry, and the obvious question — is `(32, 2)` too aggressive for a
narrow projected spectrum? — is already answered in `docs/locg-chebyshev-prefilter.md` §3.1, which swept
the `(degree, cycles)` grid and found `(32, 2)` beat the original `(16, 4)`. The regimes are not
comparable: the paper's `H̃` is `Ns × Ns` from a converged occupied subspace, while `sqd`'s comes from
sampled bitstrings and can be far from a Krylov-optimal basis. No action, but the sweep is the reason to
trust `(32, 2)` rather than the paper's order-4 figure.

## 5. Provenance note

The WebFetch summary of this paper answered from metadata and was wrong on the two points that decide
applicability: it claimed the method holds a number of vectors "independent of system size" (it is
`Ns × Nt`) and that the two levels are "nested filtering ... onto occupied subspace properties" (they
are outer-full/inner-projected). Both were corrected by extracting the PDF text (`pdftotext`) and reading
§2.1.3 and §2.2.2. Worth recording as a method note: for a decision that hinges on memory footprint and
block size, read the formulation, not a summary of it.
