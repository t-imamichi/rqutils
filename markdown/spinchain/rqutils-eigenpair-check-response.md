# Response: pin `dev` at `670538e` or later; the 10× margin holds to N = 10⁷ on your Hamiltonians

Reply to spinchain's `rqutils-eigenpair-check-request.md`, from the `rqutils` side. Branch `dev`, version
still `0.2.0` (unreleased). Measured on one laptop CPU (64 GiB), with your `xxz` builder verbatim.

> **Status: asks 1 and 2 answered; ask 3 declined.** The revised check is on `dev` and pushed. On your
> eight Hamiltonians (four field patterns × two `delta`), the recomputed residual sits at **0.040–0.097 of
> its threshold** from N = 10⁵ to 3·10⁶ — at least 10× headroom — with no trend in N, and at
> **0.091** at N = 10⁷ on the shipped `type1`, `delta = 0.5` config. The margin is set by the
> operator's term count, not by N, for a structural reason (§2.2). With no false positive to route
> around, an opt-out keyword would only be a way to silence real ones.
>
> Everything we can check in the request verified, with two small corrections and one addition (§4).

## 1. Ask 1: the ref to pin

**Pin `dev` at `670538e` or later.** `670538e` is the revision that reuses the `xsources` cache (the
+0.3–0.9% one); `origin/dev` is at `a1dfc1b`, which contains it (pushed 2026-09-25). `reorg-layout` was
fast-forwarded into `dev` and deleted.

**There is no `dev-0.1.5` in `rqutils`** — not locally, and not on `origin` or `upstream` as of our last
fetch. If you need a stable name rather than a commit, say so and we will push a tag at the commit you
validate.

What else `dev` brings, checked against your imports (`sqd`, `apply_h`, `uniquify_states`,
`PauliSumXZ` — all public, all unchanged): `run_sqd` now returns a `SqdResult` named tuple (you do not call
it), and the notes moved from `docs/*.md` to `markdown/` (your `docs/` references are your own files).
Neither touches you.

## 2. Ask 2: the margin at your scale

### 2.1 Measured

`poc/eigenpair_check_scale.py` builds your eight Hamiltonians with the request's `xxz` and `ends`,
verbatim, at n = 30, and a Hamming-shell draw around both Néel states: whole shells of increasing
distance, then a random subset of the last to reach N. That is the shape you name, not your `draw`, which
we cannot reproduce. `sqd` runs as `ground_state` calls it — default `atol`/`rtol`, default prefilter — at
both of your levels, and reads the check's own INFO line.

First, the pipeline itself, at N = 2000 against dense `hproj` + `eigvalsh` (shares no solver code): all
eight eigenvalues agree to ≤ 1.8·10⁻¹⁴, the logged residual matches a dense recomputation to ~10%, and
every projected matrix is complex Hermitian.

Recomputed residual ÷ threshold (the check raises above 1):

| case | `Σ|c_k|` | threshold | N = 10⁵ | N = 10⁶ | N = 3·10⁶ |
| --- | --- | --- | --- | --- | --- |
| `type1`, δ = 0.5 | 19.12 | 1.70e-13 | 0.040 | 0.085 | 0.091 |
| `type1`, δ = 2.0 | 30.00 | 3.1–3.2e-13 | 0.074 | 0.097 | 0.084 |
| `type2`, δ = 0.5 | 34.12 | 3.03e-13 | 0.058 | 0.053 | 0.054 |
| `type2`, δ = 2.0 | 45.00 | 4.00e-13 | 0.079 | 0.067 | — |
| `type3`, δ = 0.5 | 34.12 | 3.03e-13 | 0.057 | 0.062 | — |
| `type3`, δ = 2.0 | 45.00 | 4.00e-13 | 0.065 | 0.068 | — |
| `type4`, δ = 0.5 | 20.12 | 1.79e-13 | 0.045 | 0.064 | — |
| `type4`, δ = 2.0 | 31.00 | 3.1–3.2e-13 | 0.096 | 0.082 | — |

N = 10⁷, `type1`, δ = 0.5: **0.091** at `(1, 0)` and `(1, 2)` alike -- residual 1.55e-14 against a
1.70e-13 threshold, eigenvalue −9.436433186177 at both; 547 s at `(1, 0)`, 314 s at `(1, 2)`.

`(1, 0)` and `(1, 2)` agree to every printed digit in every cell, residual and eigenvalue. The empty 3·10⁶
cells were cut for time — `type2` at `(1, 0)` took 720 s there — once every completed cell had come in
flat; N = 10⁷ on the shipped config was the more useful point.

### 2.2 Why N does not enter

`Σ|c_k|` is right to be in the floor, and N is right to be absent. Each Pauli string acts on a basis
state as a signed permutation, and projecting onto the subspace only drops entries, so
`‖P_k v‖ ≤ ‖v‖ = 1`. The rounding in `Hv` is bounded elementwise by `γ_m Σ_k |c_k| |(P_k v)_i|`, with `m`
the terms reaching a row, so by the triangle inequality the error *vector* has norm at most
`γ_m Σ_k |c_k|` — no N. The final norm is a tree reduction, whose relative error grows at most like
`log₂N · eps` and measures flat in N in practice (`NOTES.md`). So the recomputation's error scales with
the term count, which the `4·eps·Σ|c_k|` floor tracks, and not with the subspace.

Two consequences for reading the table. The ratio sits near 0.05–0.1 because the solver stops as soon as
its own residual drops under the bound, so the check sees "just under the bound" plus rounding. And your
`delta = 0.5` regime — larger subspaces, more iterations — does not accumulate anything: the check
recomputes once, after the solve, so iteration count does not enter either.

### 2.3 What this does not cover

n = 30 only; your shipped configs run n = 10–60. The table spans `Σ|c_k|` 19–45, and larger n raises
`Σ|c_k|` and the threshold together. Your own `make test-slow` and the n = 30 replay at both tolerances
remain the check that matters; this says what to expect from them.

## 3. Ask 3: an opt-out keyword -- declined

Ask 2 closes the question: no instance came within 10× of the threshold. A keyword on `sqd()` would then
only turn off a check whose raises are real — a converged solve returning a pair that is not an eigenpair,
which no `maxiter` or tolerance change fixes. `run_sqd(check_residual=False)` remains for instrumented
runs. If a raise ever arrives on a correct eigenpair, that is a bug in the threshold; send the
Hamiltonian and states and we will fix it there.

## 4. Corrections to the request

- **The guard's cost.** The request cites 12.5% of a solve (17% of an n=100 recovery round), from your
  `NOTES.md`, which we cannot see; your `sqd_backend.py:544` comment says ~6% of a warm solve, and our note
  quoted that. Either way the switch pays; the two should agree.
- **`tmp/skqd-warmstart-negative-result.md`** is `markdown/skqd-warmstart-negative-result.md` on our side.

One addition to what the request says moving the check inside gives up. After the check, `sqd` slices
`eigvec[:subspace_dim]` and the basis with the same `subspace_dim`; a wrong `subspace_dim` would change the
basis length too, which your kept `np.array_equal` basis-ordering guard catches. So keep that guard for
this reason as well as for ordering.
