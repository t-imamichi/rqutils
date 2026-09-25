# Reduced precision in the eigensolver

Moved from `NOTES.md` ("f32 *storage* for the solver's carried vectors", 2026-08-30), CPU. Both variants
rejected: f32 matvec arithmetic (POC 6, `markdown/scaling-pocs.md` §6) and f32 storage of carried
vectors. The script is `poc/mixed_precision.py` (§5); the storage measurement in §2 is not in it.

## 1. f32 arithmetic, f64 storage (POC 6): rejected, and it still reproduces

`poc/mixed_precision.py` runs the matvec *arithmetic* in f32 and casts back, keeping f64 *storage* — a
bandwidth optimization. Re-running it confirms the verdict (`markdown/scaling-pocs.md` §6, **reject**):
1.17–1.30× at fixed iterations, three of four converged solves hitting `maxiter=300`, **0.42×
end-to-end**, and 6d's naive form converging in 9 iterations with `converged=True` and a **4.42%
relative error**.

## 2. f32 storage: `ax` cannot be demoted, and it is the one that matters

Proposed as the one lever on the `(0, 0)` floor. That floor is **120 B/slot** measured, of which only 13
is the Hamiltonian: 32 B/slot is `_State`'s four carried vectors (`x, y, r, ax`) and ~75 is transients.
At `2^31` slots the floor alone is **258 GB**, and no `cache_level` setting touches it. Demoting a
carried vector is a memory optimization, a different proposal from §1, which the POC 6 verdict does not
cover.

The shared discriminator settles it cheaply. Take a converged eigenpair (1D Heisenberg n=40, N≈20k,
`theta = -28.236067977500`, 66 iterations, `‖A‖ ≤ Σ|c| = 117`) and round individual operands of
`r = Ax - θx` to f32:

| residual path | `‖r‖` | `‖r‖/‖A‖` |
| --- | --- | --- |
| f64 (today) | 2.813e-09 | 2.405e-11 |
| **`r` stored f32** | **2.813e-09** | **2.405e-11** |
| **`ax` stored f32** | **6.780e-07** | 5.795e-09 |

`ground_locg`'s default `tol` (at the time) is `eps(f64) = 2.22e-16`, so an f32 `ax` puts the residual
floor **3.1e6× above the tolerance** — the convergence test becomes unsatisfiable and the solver runs to
`maxiter`, which is precisely POC 6's measured failure arriving by a different route.

## 3. The asymmetry is structural

Storing `r` at f32 changes nothing, because `r` is already `O(1e-9)` and f32 carries ~7 significant
digits of *relative* precision. Storing `ax` is fatal because `ax` is `O(‖A‖) ≈ 117` while `r` is
`O(1e-9)`: the subtraction is a catastrophic cancellation of nearly equal `O(100)` quantities, and
rounding the operands destroys the digits the cancellation depends on. **That is a property of the
arithmetic, not a tunable tolerance** — no `work_dtype` handling or looser `tol` recovers it without
accepting POC 6d's silent error.

## 4. What it means

- **What is left is not worth the risk.** `x` is the answer and also an operand of the same subtraction;
  `ax` is ruled out. So at best `r` and `y` could be demoted: **8 of 120 B/slot, 6.7%**, or 17 GB of 258
  at `2^31`. That needs a mixed-dtype `_State` — `while_loop` requires the carry types to agree, and
  `ground_locg:951` (at the time) derives `work_dtype` from `result_type(xinit, matvec output)` and casts
  `xinit` to it, so a per-field dtype is a change to every operation in the iteration. Against
  `markdown/locg.md`'s seven defects that each failed *silently*, 6.7% of the floor is not a good trade.
- **The levers already available at that size dwarf it:** `cache_level` from `(1, 1)` to `(0, 0)` is
  1805 → 120 B/slot, **15.0×**, no code change; hand-sizing `states_size` at N=24M saves **7.7 GB** for
  one keyword. Neither risks a silent wrong answer.
- **The `(0, 0)` floor stands at 120 B/slot, and 258 GB at `2^31` is the honest ceiling** for this solver
  on one device. Lowering it needs a different eigensolver structure — fewer carried `O(N)` vectors, or a
  restart scheme that trades vectors for matvecs — not a dtype change. That is a separate investigation
  and nothing here bears on it.

## 5. The script

`poc/mixed_precision.py` ("POC 6") takes no arguments. It builds `make_problem(24, N, num_terms=200,
num_xgroups=50)` fixtures, real and complex, at `cache_level=(1, 2)`-style precomputed `xsources` and
`diagonals`, and compares an f64 `apply_h` with a mixed one that casts to f32, applies, and casts back:

| section | what it measures |
| --- | --- |
| 6a | matvec time f64 vs mixed, N = 200k, 500k (CPU: a lower bound on a GPU win) |
| 6b | eigenvalue error, iterations and convergence vs a full-f64 solve, N = 50k, 200k, `maxiter=300` |
| 6c | solve time at N = 200k: fixed 40 iterations (`atol=rtol=0`) and converged (`maxiter=300`) |
| 6d | the naive f32 matvec that does not cast back, so `work_dtype` and the tolerance drop to f32 |

It does not contain §2's storage experiment.
