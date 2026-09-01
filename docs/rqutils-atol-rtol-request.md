# rqutils change request: keep both an absolute and a relative `tol`

One request against `rqutils` on branch `dev` (installed rev `d55f067`), written from the `spinchain`
side. It is a follow-up to `docs/rqutils-tol-request.md`, which asked for `tol` to become an absolute
eigen-residual bound and **got what it asked for** — this asks for the relative form back *alongside*
it, not instead of it.

> **Status: NOT SENT.** Drafted after adopting the absolute `tol` and finding one property the old
> relative form had that the new one cannot express. Both revisions are measured below; nothing here
> disputes the absolute form's value, and `spinchain` is keeping it.

## 1. The problem: neither form alone covers both callers

The absolute `tol` did exactly what was asked. On this repo's n=30 subspace (dim 2,723,958) it is
**2.41x at `tol=1e-6`** with the eigenvalue unchanged, and it lets a caller with a 1e-6 residual guard
finally *name* 1e-6. That half is settled.

What was lost is a property the old form had by accident: **one `tol` value scaled itself across the
several dimensions a single pipeline run solves at.** The old effective bound was
`tol·(‖Ax‖ + |θ|)·N·10`, so one value meant:

| call site | scale | `N` | old effective bound |
| --- | --- | --- | --- |
| `config/skqd.toml`, n=13 | ~9.5 | 4,596 | **9.7e-11** |
| `config/n30/replay.toml`, n=30 | ~30.6 | 2,723,958 | **1.9e-07** |

A 1900x spread from one number. **No single absolute `tol` can be both**, and that is not hypothetical
— it is why this repo cannot reproduce its own locked regression constants under the new semantics.
Measured, against a `1e-12`-tolerance end-to-end lock on `sum(js_Z)`:

| `solve_tol` | delta from the locked value | verdict |
| --- | --- | --- |
| `None` (upstream's derived floor) | 0 | passes |
| `1e-12` | 7.6e-12 | fails |
| `1e-10` | **1.9e-12** | fails (closest) |
| `1e-8` | 2.8e-10 | fails |
| `1e-6` | 2.1e-08 | fails |

**The deltas are non-monotone in `tol`, and they miss on opposite sides** — `1e-12` lands low where
`1e-8` lands high, and `1e-10` is nearer than `1e-12`. A converging sequence does not do that. The
signature says the target is not reachable by *any* absolute value, because the constants encode a
per-dimension bound and the pipeline solves at several dimensions in one run.

Our options today are therefore: keep `tol=None` and take the regression (§2), or set an absolute
`tol` and re-lock constants that have caught three accidental regressions during past refactors.
Neither is good, and a two-tolerance API makes both unnecessary.

## 2. Why `tol=None` is not a free fallback

Your §5.1 measured the new default at 1.3–1.5x slower, rising with `N`, and said the `dim=2e5` row was
arithmetic on bounds rather than a measurement. Measured on our real subspace at `dim = 2,723,958`,
`tol=None` on both revisions, same subspace, warm:

| revision | `sqd()` | delivered residual |
| --- | --- | --- |
| `c400fae` (relative) | **15.89–16.77 s** | ~1.9e-07 by your formula |
| `d55f067` (absolute) | **34.46 s** | 6.5e-16 |

**~2.1x slower**, worse than your largest measured point, for nine decades of convergence past
anything we check. So "stay on the default" costs a factor of two, and "set an absolute `tol`" costs
the regression anchor.

## 3. The ask: `atol` and `rtol`, either sufficient

```text
converged  when  ‖Hv − Ev‖₂  <  max(atol, rtol · scale)
```

with `scale` whatever you already compute (`‖Ax‖ + |θ|`, optionally the `N` factor — see below), and
each tolerance independently settable. This is `scipy.integrate`'s and `np.allclose`'s convention, and
it is chosen for the same reason: a purely relative test cannot bound a quantity near zero, and a
purely absolute one cannot scale across problem sizes.

The `max` matters — satisfy **either**, not both. It makes the pair strictly more expressive than
either form alone, and both current behaviours become special cases:

| caller wants | `atol` | `rtol` | recovers |
| --- | --- | --- | --- |
| name a residual the guard can check | `1e-6` | `0` | today's `d55f067` behaviour |
| per-dimension scaling across a pipeline | `0` | `eps·10` | `c400fae` behaviour exactly |
| whichever is looser (the usual want) | `1e-6` | `eps·10` | neither, and the useful default |

The third row is what we would actually run: the relative term binds on small subspaces where 1e-6
would be absurdly loose, and `atol` binds on large ones where the relative term would demand more
digits than any consumer reads.

**Backward compatibility is available and cheap.** `rtol=0` is exactly the current semantics, so
defaulting `rtol=0` makes this a pure widening — no existing caller changes behaviour, and the
`.. warning::` block you added stays accurate. If you would rather the default *be* the pair, that is
also fine by us, but it is a second silent semantic change and we are not asking for one.

## 4. What we cannot tell you

- **Whether the `N` factor belongs in `scale`.** Your §2 measured the achievable floor as `eps·‖H‖`
  with no `N` dependence, so `N·10` was slack rather than a rounding budget — we accept that. But the
  *reproducibility* property we want came from that slack. If `scale = ‖Ax‖ + |θ|` without `N`, `rtol`
  gives per-operator scaling but not per-dimension, and we do not know whether that is enough to
  reproduce our constants. It is one experiment on your side and we cannot run it without the API.
- **Whether this is worth it to anyone but us.** Our need is specific: an end-to-end lock spanning
  several subspace sizes in one run. A caller solving one dimension per process has no use for `rtol`.
  We are asking because the cost looks small and `rtol=0` makes it free for everyone else, not because
  we think it is universal.

## 5. Verification performed

All figures above are measured through `sqd()` on this repo's own operators (1D XXZ, `By != 0`, so
complex128 and odd-Y terms), `cache_level=(1, 2)`, `prefilter=(32, 2)`, warm, best-of-N. The `c400fae`
arm is an actual downgrade-and-rerun of the installed package, not an extrapolation.

The floor formula was checked independently: `residual_floor(sum|c|, complex128)` reproduces
`4·eps·Σ|c_k|` to the bit, and at our `Σ|c| = 19.1` gives 1.699e-14 — so our 1e-6 guard has ~8 decades
of headroom, and per your `N` sweep that headroom does not shrink with dimension. We have no quarrel
with any of §2 or §3 of your reply.

Not verified: whether a two-tolerance form reproduces our locked constants. That is the whole point of
the request and needs the API first — the same position your reply put us in on the variance question,
and we would rather say so than assert it.
