# Two-level / deflation preconditioner for `ground_locg` — **TESTED AND REJECTED**

**Status: measured 2026-08-28, rejected. 0.68–0.98× wall clock, 8/8 losses at the two smaller coarse
sizes, and it *increases* iteration count in every single configuration.** Harness:
`examples/scaling/poc10_deflation_precond.py`. CPU-only, see §5.

This was the last untried deterministic candidate. `docs/rqutils-precond-request.md:697` named it — "a
two-level / deflation preconditioner exploiting the near-block structure of a sampled 1D chain" — and
called it speculative. It is now closed, which closes the deterministic preconditioning line entirely.

---

## 1. Why it was worth trying, and why that reasoning was sound

Six routes to a Jacobi-style shift were previously measured and rejected, all for one root cause: no
viable *lower* bound on the projected operator's minimum. `docs/rqutils-precond-request.md:786` then
retracted the assumption that a tighter bound on `H` was ever the route — `ground_locg` sees
`hproj(H, subspace)`, whose minimum sits 0.64–1.06× of the projected spectral width **above**
`λ_min(H)`, so no bound on `H`, however tight, can reach the shift the 1.79× used.

Deflation escapes that blocker cleanly, and this is not a rationalization after the fact: it estimates
the **projected** operator's minimum directly, by building a small coarse space and solving the coarse
eigenproblem *exactly*. It needs no bound on anything. That is a genuinely different mechanism from
every rejected route, which is why it was the one candidate left.

**Construction.** Coarse space = the `k` lowest-diagonal basis states (`coarse_space`), which is the
"near-block structure" made concrete — the projected XXZ diagonal is free and exact (matches to
0.000e+00 per the request doc), and the ground state's weight concentrates there. Then
`M⁻¹r` replaces the residual's coarse component with `(A_c − σI)⁻¹` applied to it, `σ` just below the
coarse minimum, leaving the complement untouched. The `k×k` inverse is formed once, host-side.

## 2. The measurement

Connected XXZ subspaces via one-hop expansion, 8 configurations (2 sizes × 2 anisotropies × 2 seeds),
`k ∈ {16, 64, 256}`. Density 0.17–0.98% and relgap 2.2e-03–2.5e-02, both matching the ranges the
request doc reports for this regime — so this is the connected regime, **not** the uniform-random one
that `docs/skqd-sqd-solve-tolerance.md` §7.1 warns invalidates every iteration-count conclusion.

| `k` | iter-reduction median | wall median | wall min | losses | wrong | align median |
| --- | --- | --- | --- | --- | --- | --- |
| 16 | **0.86×** | 0.91× | 0.75× | **8/8** | 0/8 | 0.9867 |
| 64 | **0.84×** | 0.90× | 0.68× | **8/8** | 0/8 | 0.9329 |
| 256 | **0.86×** | 0.86× | 0.38× | 6/8 | 0/8 | 0.8609 |

Confirmed at a genuinely larger dimension (n=20, dim=14250, density 0.040%, relgap 1.43e-02), so this
is not a small-`N` artifact and there is no size crossover:

| arm | iters | wall |
| --- | --- | --- |
| plain | 122 | 11007 ms |
| `k=64` | 141 | 0.72× |
| `k=256` | 144 | 0.77× |
| `k=1024` | 121 | 0.90× |

## 3. Why it fails, and it is not the usual failure

**The iteration-reduction column is the finding: 0.84–0.86×, i.e. deflation makes `ground_locg` take
*more* iterations, not fewer.** That is the primary signal, per the acceptance criterion the request doc
set — judge on the relative gap, not on `κ`, because across 12 instances `κ` varies 1.21× while the gap
varies 103×, and log-iterations correlates +0.77 with `log(1/gap)` against **−0.34, the wrong sign**,
for `κ`. Deflation improves the conditioning of the coarse block and does not open the gap. It is
exactly the failure that criterion was written to predict, and the criterion earned its keep here.

**This is *not* the depleted-residual failure**, and that distinction is the transferable part. The
alignment `|⟨r|M⁻¹r⟩|/(‖r‖‖M⁻¹r‖)` is **0.86–0.99**, against **0.001** for the rejected
Chebyshev-filtered residual. So the search direction is a perfectly good gradient — the preconditioner
is well-behaved, and the wall-clock loss is real work buying nothing rather than a corrupted direction.
Three prior candidates failed by destroying alignment; this one failed while preserving it. A future
candidate scoring well on alignment has therefore cleared a necessary condition, not a sufficient one.

**Correctness is not the issue either.** Energies are exact to 3.6e-15–2.3e-14 and eigenvector overlap
is 1.0000000 in all 24 arms. Unusually for this module, there is no plausible-but-wrong mode to report.

The cost side is straightforward: the gather/matmul/scatter adds `O(N·k)`-ish work around each
iteration, and `k=256` at dim=3000 is where the 0.38× outlier comes from.

## 4. Note on the one apparent win

Two arms read faster — `k=256` at 1.21× and 1.05×, both at n=16 dim=512. Both are on the *smallest*
fixture, where `k=256` is half the subspace, and the same setting is 0.38× at dim=3000 and 0.77× at
dim=14250. Not a regime worth chasing: a coarse space that is a constant fraction of `N` is not a
two-level method, and it does not survive growing `N`.

## 5. CPU-only caveat

`jax.default_backend() == "cpu"`, single device. Deflation adds `O(N·k)` around each matvec, and that
matvec-to-`O(N)` ratio is exactly what shifts on GPU — the reason `poc9_prefilter_gpu.py` exists unrun.
Per POC 8's rule, a result on one backend says nothing about another in either direction. That said, the
**iteration counts are backend-independent arithmetic**, and they are unfavourable on their own (0.84–0.86×),
so a GPU run would have to overturn the mechanism, not just the constant. I would not expect it to.

**Sharding is unresolved and was not claimed.** The preconditioner is a gather, a small dense matmul and
a scatter — not elementwise, so on a mesh it would need `out_sharding` plumbing to satisfy
`ground_locg`'s documented contract. Measured single-device only. Since the candidate is rejected on
iteration count, this was left open rather than built.

## 6. What this closes

With this, every deterministic candidate in `docs/locg-next-candidates.md` and the preconditioning line
in `docs/rqutils-precond-request.md` is measured and rejected. Combined with the randomized families
(also all rejected — block Krylov, sketching, stochastic RQ minimization, randomized warm start), **the
search around `ground_locg` is closed on CPU.** The two live threads are unchanged and both are
measurement, not algorithm work:

- `examples/scaling/poc9_prefilter_gpu.py` — written, unrun, needs CUDA. It re-prices several
  rejections, including Davidson (correct to 1e-11, 1.05–2.62×, lost on cost) and ARPACK
  (2.15–2.72× where `ground_locg` iterates a lot, blocked on sharding rather than speed).
- The `iters ≈ relgap^−0.473` scaling law is **retracted**; refits swing between −0.15 and −0.82. A
  proper study needs relgap varied over several decades.
