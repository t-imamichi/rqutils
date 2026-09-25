# `sqd` now checks its own eigenpair: `EigenpairCheckError`, and what it means for `sqd_backend.py`

From the `rqutils` side, unprompted. Commit `bed4757` on `dev`, version still `0.2.0` (unreleased).
**Revised the same day** on `reorg-layout`: the check now reuses a full `xsources` cache, cutting its
cost from +3.2–7.9% to +0.3–0.9% (§1, §5).

> **Status: shipped, always on.** After every solve, `sqd()` recomputes `‖Hv − Ev‖` independently and
> raises `EigenpairCheckError` if the pair it is about to return is not an eigenpair — the
> `converged=True`-but-wrong class your `_eigen_residual` guard exists to catch.
>
> **What you can delete** once you pin `>= bed4757`: `_residual_kernel`, `_eigen_residual`,
> `_eigen_residual_packed`, `_residual_threshold`, `_RESIDUAL_TOLERANCE`, and the residual block in
> both `ground_state` and `ground_state_packed` — about 120 lines. §4.
>
> **What you must keep**: the basis-ordering guards, `_check_operator_width`, and `sqd_with_retry`.
> Each covers a failure the new check does not. §4.
>
> **No change to `sqd_with_retry` is required**, and one constraint on our side exists for it: the new
> message never contains `"did not converge"`. §3.

---

## 1. What shipped

After LOBPCG returns, and before `sqd()` returns anything, one extra matvec runs:

```text
r  = ‖H v − E v‖          (recomputed, not the solver's in-loop value)
hv = ‖H v‖
bound     = max(atol, rtol · (hv + |E|))          # the solver's own convergence criterion
threshold = 10 · max(bound, 4·eps·Σ|c_k|)         # Σ|c_k| term = rqutils.ground_locg.residual_floor
raise EigenpairCheckError  if not r <= threshold   # `not <=`, so a NaN residual raises too
```

- **`rtol=None` resolves to `4·eps`** of the coefficient dtype, exactly as the solver resolves it, so
  the check tests the same criterion the solve used.
- **The check never uses a cached diagonal**: it rebuilds every diagonal from the Z signatures, so a
  defect in the `(1, 1)`/`(1, 2)` diagonal caches cannot vouch for itself. **It reuses the source
  indices when all of them are cached** (`cache_level[0] = 1`, no partial `xcache_groups`) and searches
  afresh otherwise. Redoing the search was ~90% of the original check's cost.
- **One difference from your guard, stated plainly**: `_apply_projected` pins `(0, 0)`, so it also
  re-derives the source-index array. The rqutils check no longer does when that array is cached. What
  that gives up is narrow: both call the same `get_xsource`, so only a defect in the precompute scan
  that builds the array would slip past, not one in the search itself.
- **The residual is logged at `INFO`** on the `rqutils.sqd` logger:
  `Independent eigen-residual 6.145e-16 (threshold 7.802e-14).` Your per-solve `residual=` log line
  can read from there instead, or stay — see §5.
- **Covers `return_eigvec=False` too.** The check runs inside the solve, so the energy-only path is
  guarded as well — which yours could not do, having no vector to check.

`EigenpairCheckError` is a public `RuntimeError` subclass in `rqutils.sqd`:

```python
from rqutils.sqd import EigenpairCheckError
```

## 2. Why the threshold is `10 × max(bound, floor)`

Measured, not chosen. Over the full `rqutils` test suite's **185 converged `sqd` solves**, the
recomputed residual sits at **max 0.96, median 0.27** of `bound`. The solve already guarantees its
in-loop residual is below `bound`, so the only thing separating the two numbers is the recomputation's
own rounding.

That tail sits at the **floor**: at the default `rtol`, `r = 2.9e-15` against `bound = 3.0e-15` with
floor `4.2e-15`. A bare multiple of `bound` would be testing floating-point noise there, so the floor
is inside the `max`. The `10×` leaves 10x headroom over the worst converged solve measured.

On the other side, the defect it is for:

| injected defect, `converged=True` | recomputed `r` | threshold |
|---|---|---|
| dominant eigenvector component sign-flipped | **5.7e+00** | 7.8e-14 |

13 orders of magnitude of separation. For comparison your `_RESIDUAL_TOLERANCE = 1e-6` (relative to
`max(1, |E|)`) is far looser — both catch the gross failures; ours additionally catches a pair that is
wrong at, say, 1e-10, which yours would accept.

**One caveat carried over from your docstring, unchanged**: a vector that stays inside a *degenerate*
ground eigenspace is still an eigenvector, so it passes. This bounds the eigenpair, not the choice of
vector within an eigenspace.

## 3. The exception, and `sqd_with_retry`

Two distinct failures, two responses, now two types:

| raised | means | fix |
|---|---|---|
| `RuntimeError("LOBPCG did not converge …")` | the iteration cap ran out | raise `maxiter` — your retry |
| `EigenpairCheckError("LOBPCG reported convergence, but …")` | the returned pair is inconsistent | none; it is a bug — report it |

`sqd_with_retry` matches `"did not converge"` in the message. **`EigenpairCheckError`'s message never
contains that substring** — pinned by an `rqutils` test — so it propagates through your wrapper
unretried, which is what your docstring requires ("Widening it to any `RuntimeError` is what must not
happen").

Being a subclass, it is still caught by any `except RuntimeError`. If you want to stop depending on
message text, the robust form is to let the subclass through explicitly:

```python
except EigenpairCheckError:
    raise
except RuntimeError as exc:
    if "did not converge" not in str(exc):
        raise
    ...
```

Optional — the current wrapper is already correct.

## 4. What to delete from `sqd_backend.py`, and what to keep

**Delete** (subsumed):

- `_residual_kernel`, `_eigen_residual`, `_eigen_residual_packed`
- `_residual_threshold` — and with it the `2·rtol·|E|` estimate of `‖Hv‖`: rqutils now has the exact
  `‖Hv‖` from the check's own matvec, so the unit conversion it did is no longer needed anywhere
- `_RESIDUAL_TOLERANCE`
- the `residual = …` / `if residual > threshold: raise …` blocks in `ground_state` and
  `ground_state_packed`

**Keep** (not covered by the new check):

- **Both basis-ordering guards** (`np.array_equal(basis_states, …)`). A permuted basis gives a
  *correct* eigenpair in the wrong index order — the residual is fine and `expvals` are wrong. The
  check cannot see that.
- **`_check_operator_width`** — still needed at `expvals_packed`, which calls the bare kernels.
- **`sqd_with_retry`** — non-convergence still raises the plain `RuntimeError`.
- `_pack_subspace*`, `_expval_kernel`, `expvals*` — unrelated to the residual.

If you log the residual yourselves today, either enable `INFO` on the `rqutils.sqd` logger or drop the
field from your log line.

## 5. Cost

One matvec per solve. Measured at `J=120` X groups, `N=30k` states, warm, 9 interleaved rounds:

| `cache_level` | spinchain `DiagCache` | time (first version) | time (revised) | XLA temp (revised) |
|---|---|---|---|---|
| `(1, 0)` | — (sqd default) | +3.2% | **+0.9%** | +0.25 MiB |
| `(1, 1)` | — | — | **+0.3%** | +0.35 MiB |
| `(1, 2)` | `SPEED` | +7.9% | **+0.6%** | +0.35 MiB |

**Net for spinchain**: your guard costs about 6% of a warm solve (its own repack, re-uniquify lexsort,
operator rebuild and a `(0, 0)` matvec), all of which goes away, so every level now comes out ahead,
`SPEED` included. These are CPU figures at one size; A/B on your own shipped job before quoting a number.

There is no flag to turn the check off in `sqd()`; the plumbing (`run_sqd(check_residual=…)`) exists.

## 6. Sharding and multi-process

- The check runs on the solve's own `states` layout, so it adds no reshard of `states`.
- Both scalars go through `_host_scalar`, the same non-collective read as the eigenvalue. They are
  replicated, so every rank takes the same branch and raises together — no collective sits inside the
  conditional.
- Verified on 4 virtual CPU devices (`poc/sharding.py`, all six cache levels agree). **Not verified on
  a real multi-process run** — the `rqutils` sandbox cannot bind a coordinator port. If you run SKQD
  across nodes, an `mpirun` smoke test after upgrading is worth the minute.

## 7. Upgrading

1. Pin rqutils `>= bed4757` and check the installed `commit_id` in
   `.venv/lib/*/rqutils-*.dist-info/direct_url.json` — the stale-venv gotcha applies.
2. Delete the §4 list. Run your suite: the `hproj`-based dense comparisons are unaffected.
3. If any test asserted your `"not an eigenpair"` message, point it at `EigenpairCheckError` instead.
