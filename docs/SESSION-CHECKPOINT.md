# Session checkpoint — 2026-08-24

Written so this session can be resumed after a restart. Delete once picked up.

## Why the restart

`NONO_LOG_FILE` was added to `~/.claude/settings.json` to silence repeated
`WARN Network audit buffer full (4096 events)` messages. Those come from the **`nono`
sandbox plugin** (v0.74.0, `/opt/homebrew/bin/nono`) — verified: the literal string is in
that binary, and appears nowhere in this repo or `.venv`. It is a volume diagnostic (a
4096-event ring buffer overflowing), not an error. `env` is read at session start, hence
the restart.

Backup of the previous settings: `~/.claude/settings.json.bak`.
Logs now go to `~/.claude/nono.log` (will grow; truncate occasionally).

## Repo state — safe to resume from

Branch **`dev`** at **`8448794`**, working tree **clean**, `origin/dev` **behind by 5**
(nothing pushed). Verified at checkpoint time: **436 tests pass**, ruff and ty clean, docs
build at their one standing warning (`rqutils.paulis.rst` not in a toctree — pre-existing).

Unpushed commits, newest first:

| sha | what |
| --- | --- |
| `8448794` | Record the `sqd` convenience flag as measured and **rejected** |
| `8a9c9e5` | **Add a preconditioner hook to `ground_locg`** (the "P" in LOBPCG) |
| `08794e6` | Recommend a `bool` over `precond="jacobi"` |
| `4a90ee6` | Answer "is Jacobi the only option"; confirm 1.75x, retract two bad measurements |
| `a8e6f67` | Correct the precond request's headline |
| `3da1b46` | Cleanup review findings; enforce the int32 ceiling at its source |

Other branches: `metal` at `c1cffeb` (28 commits ahead on a parallel line — see "Open
threads"), `main` at `dc63162`.

## What shipped

**`ground_locg(precond=None | callable)`** — an approximate inverse `M⁻¹` applied to the
residual where the search direction is formed. Static argument, so `None` is the identity
path and leaves the traced graph unchanged. Measured **1.79x median** fewer iterations on a
12-instance XXZ batch (range 1.29–2.04x, 0 regressions).

Six tests in `tests/test_ground_locg.py::TestPreconditioner`. The load-bearing one is
`test_the_zero_residual_guard_reads_the_raw_residual`: `body_iter1`'s `norm_r`/`r_is_zero`
must keep reading the **raw** residual (it feeds the `sas[1,1]` masking *and* `converged`)
while only the direction is preconditioned. The obvious test — an exact-eigenvector seed —
does **not** discriminate, because `M⁻¹·0 = 0` either way; the discriminating fixture is an
annihilating `M⁻¹`, under which the defect returns the seed's Rayleigh quotient
(10.254108965 vs a true 1.0) with `converged=True`.

One recorded weak spot, in that class's docstring: making `precond` a no-op in `body_iter1`
*only* survives the suite. Real but small — that function runs once, costing 2–7% on all 12
instances, never a wrong answer. Two fixtures failed to discriminate at integer-count
granularity.

## What was rejected

**`sqd(jacobi_precond=...)`** — implemented on a branch, measured, discarded, branch deleted.
It is *correct* (same energy as dense to 5.8e-15 at all six `cache_level`s) and makes `sqd`
**~3x slower** (0.28–0.36x, 8/8 worse).

Cause: Jacobi needs a positive-definite operator; `sqd` solves the **unshifted** `H`, which
is indefinite — measured diagonal range `[−1.375, +1.375]` with ~50% of entries ≤ 0, and
`λ_min ≈ −2.4`. Half the diagonal must be masked to 1.0, so `M⁻¹` becomes an arbitrary
half-scaling that destroys conditioning. Every 1.79x figure was measured on the *shifted*
`A = H − (λ_min − 0.5)I`, which `sqd` never builds. Recorded in
`docs/rqutils-precond-request.md` under "❌ A convenience default on `sqd`".

## Open thread — resume here

**Question on the table:** *"1D open-boundary XXZ を前提とした preconditioner で有効なものはあるか?"*
with the explicit premise (from the user, mid-investigation) that **a transverse field is
present, so `Sz` conservation is broken.** That rules out sector decomposition, which would
have been the strongest structural lever.

Established so far, all measured:

- **The diagonal is analytically exact and matvec-free.** `diag = Δ/4 · Σᵢ sᵢsᵢ₊₁` with
  `s = 1 − 2·bit`, agreeing with the true projected diagonal to **0.0**. `O(N·n)` bit
  arithmetic. The transverse field contributes **nothing** to the diagonal (`X`/`Y` are
  purely off-diagonal), so this holds with the field on.
- **The projected operator is extremely sparse:** 0.12–0.58% nonzero at n=12–14.
- **Diagonal range is `n`-independent**, `[−1.375, +1.375]` for `J=1, Δ=0.5`.
- **Coefficient-sum shift bounds over-shift badly and worsen with `n`:** `‖H‖ ≤ (n−1)(J/2 +
  Δ/4) + n(Bx+By)/2` gives **4.14x / 6.18x / 8.14x** over-shift at n=10/12/14, because the
  bound grows with `n` while the true `λ_min` stays ≈ −2. This is the same mechanism that
  capped the crude internal shift at 1.04–1.24x.
- **Gershgorin would be tighter but needs row sums**, unavailable matrix-free without `J`
  extra matvecs.

**Candidates 1 and 2 are now measured and rejected** (2026-08-24, recorded in
`docs/rqutils-precond-request.md` under "❌ Two XXZ-specific candidates for the *unshifted*
operator"):

1. ~~A **shift from the analytic diagonal alone**~~ — **rejected.** The structural row-sum bound
   over-shifts **16.05x / 20.49x** at n=10/12, worse than the coefficient-sum bound's 4.14-8.14x and
   degrading faster with `n`. A sampled subspace projects most couplings away, so the bound assumes
   ~3x more coupling than exists. Even the unattainable bound from the assembled matrix over-shifts
   5.03-6.29x, so the whole row-sum family cannot give a tight shift here.
2. ~~**Preconditioning `|H|` rather than `H`** (`M⁻¹ = |diag|⁻¹`)~~ — **rejected.** Measured through
   the real hook: **0.30x median (0.16-0.88x), 12/12 regressions**, all energies correct and all arms
   converged. Worse than the masked Jacobi it was meant to beat. `|diag|⁻¹` amplifies the
   smallest-`|diag|` components, which for an indefinite operator sit mid-spectrum — it boosts exactly
   what should be suppressed. Positive-definiteness was never the binding constraint; correlation with
   `A⁻¹` was, and `|·|` destroys it.
3. **A two-level / deflation preconditioner** exploiting the near-block structure a sampled subspace
   of a 1D chain has — **still untried**, and now the only candidate left. Speculative, and the
   measured 0.12% sparsity is what makes it plausible. Substantially more work than either of the
   above.

Confirmed in passing: the analytic diagonal still matches the true projected diagonal to **0.0** with
the field on (n=10, n=12). It is the *shift*, not the diagonal, that has no viable estimator.

The prior finding that governs all of this: **`κ` is not what controls convergence here.**
Across 12 instances `κ` varies only **1.21x** while the relative gap of `λ_min` varies
**103x**, and log-iterations correlates **+0.77** with `log(1/gap)` against **−0.34** (wrong
sign) for `κ`. Any candidate should be judged on whether it opens the gap, not on `κ`.

## Reusable fixtures

Written this session, under `/tmp` (**gone after a reboot** — recreate from
`docs/rqutils-precond-request.md`'s reproduction blocks, which are self-contained):

- `/tmp/xxz_lib.py` — `xxz_pure`, `xxz_field`, `subspace`, `proj`
- `/tmp/precond_lib.py` — `xxz`, `subspace`, `shifted` (builds `A = H − (λ_min − 0.5)I`)
- `/tmp/absdiag.py` — candidate-2 harness: `|diag|⁻¹` through the real `ground_locg`, 12 instances.
  **Note it uses `debug=False` deliberately** — under `debug=True` the loop becomes a `jax.lax.scan`
  with no early exit, so `niter` reads `maxiter` on every arm and both arms look identical.
- `/tmp/anadiag.py` — candidate-1 harness: analytic-diagonal check plus the shift-bound sizing.

## Other open threads

- **`metal` is 28 commits ahead** on a parallel line that independently contains the
  `_spread_seed` fix, the complex-diagonal fix, the MLX removal, and a tighter C1 keyword
  API — plus an API reorganization and a CHANGELOG that `dev` lacks. Reconciliation is
  unresolved. The only thing unique to `dev` was the svsim `out_sharding` coverage.
- **Nothing is pushed.** `dev` is 5 ahead of `origin/dev`, and two of those commits are
  downstream-visible breaks (`apply_h` keyword-only; `hproj(unique_states=True)` now
  raising). Both are recorded in `CLAUDE.md` as CHANGELOG candidates.
