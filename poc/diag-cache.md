# Partial diagonal cache

Item 3 of `markdown/improvement-ideas-2026-09-25.md`, measured on one laptop CPU 2026-08-30 (§1–5) and
2026-09-25 (§6–7). Nothing here is in the library. Only §6 has a committed script,
`poc/diag_cache_order.py` (§10); §1–5 and §7 came from throwaway code that was never committed (see
each section).

## The idea

`cache_level[1]` is all-or-nothing across X groups: level 2 at `J=101, K=100` is 808 B/slot, one composed
diagonal per (state, X group), irreducible in that form. The sibling of `xcache_groups` on the diagonal
axis does not exist. Prototyped outside the library as two summed `_apply_h_kernel` calls -- `(1, 2)`
over the first `J'` groups, `(1, 0)` over the rest -- so `J'` trades diagonal bytes for recompute time.
`cache_level[0]` stays at 1 (§5).

## 1. Real `sqd()` solves: it works and composes

n=100 1D Heisenberg, J=100, K=99, N=50,072, `states_size` 65536. Every arm returned `-52.23606798`,
identical to 8 decimals:

| `J'` | diagonal store | memory saved | solve | vs all-cached |
| --- | --- | --- | --- | --- |
| 0 | 0 MB | 100% | 3243.8 ms | 5.13× |
| 25 | 13.1 MB | 75% | 2004.9 ms | **3.17×** |
| 50 | 26.2 MB | 50% | 1552.2 ms | **2.45×** |
| 75 | 39.3 MB | 25% | 1091.0 ms | 1.73× |
| 100 | 52.4 MB | 0% | 632.3 ms | 1.00× |

So **half the diagonal memory for 2.45×**, or 75% of it for 3.17×. At `J=101, K=100` that is
808 → 404 B/slot; against the 24M-slot budget in `NOTES.md`, `(1, 2)`'s 43.5 GB → ~24 GB.

- **Why it composes where the Bloom pre-filter did not, which is the transferable part.** The implied
  matvec count is **129**, stable across n=40/60/100 (from `(s10 - s12) / (mv10 - mv12)`). The diagonal
  cache is *consumed* ~129 times per solve, so a per-matvec saving multiplies; the filter targeted a
  one-off precompute at 4.5–8.4% of a solve, so Amdahl divided it. Matvec ratios of 5.0–8.2× became
  solve ratios of **3.59–5.13×** — compression, not collapse. **Ask of any `sqd` optimization how many
  times per solve its target is paid**; it is the same "weighted by call count" distinction that made the
  filter look worth building.
- **Linear and monotonic**, no cliff. The X axis "never reaches full-cache time" with a disproportionate
  last step; this one lands on it.
- **Two-arm overhead is nil end-to-end**: `J' = J` measured 632.3 ms against the single-kernel 649.1
  (1.027×, noise). Peak memory is §3.
- **Bit-identical at every split** — `max |Δ|` exactly `0.00e+00` against the all-cached reference at
  every `J'`, at both n=60 and n=100.

Fixtures are 1D Heisenberg with a subspace half-closed under a weight-preserving hop, not sampled from a
circuit. The intermediate solves came from throwaway `run_sqd` instrumentation behind an env var, since
`sqd` has no such parameter; it was removed and the file restored from a `cp` backup (626 passed after).
No script was committed: commit `64dbe5f` touches only `NOTES.md` and `CLAUDE.md`, so this is not
recoverable from git.

## 2. The projection model: size by arithmetic, never advertise a runtime

Validated against points it was not calibrated on, since `NOTES.md` records the analogous X-axis model
drifting 17.5% mid-range. Calibrating `solve(J') = fixed + niter * matvec(J')` on the two endpoints only
(`niter = 129`, `fixed = 245.9 ms`) and testing the middle:

| `J'` | projected | measured | residual |
| --- | --- | --- | --- |
| 25 | 2050.7 | 2004.9 | +2.3% |
| 50 | 1665.6 | 1552.2 | **+7.3%** |
| 75 | 1207.3 | 1091.0 | **+10.7%** |

Worst non-calibration residual **10.7%**, erring *pessimistic* — real solves beat the projection. Better
than the X axis's 17.5%, and the same rule follows: **a budget API can size the cache by exact arithmetic
and must not advertise a runtime.**

## 3. Memory overhead is a *ratio*: 16 B/slot, 4.0% of the saving, `4/J`

A single `states_size` of 65536 first read peak temp as ~1.1 MB at an intermediate split against
0.0–0.5 MB at the endpoints (XLA `memory_analysis`; 3.5% of the 31.5 MB store at J=60), and that was
recorded as "flat at 1.1 MB across every intermediate split". **It is not flat.** Swept to `2^21`, peak
temp is exactly **16 B/slot** and scales linearly with `N`:

| `states_size` | full diagonal store | peak temp at `J/2` | peak B/slot |
| --- | --- | --- | --- |
| 2^16 | 52.4 MB | 1.05 MB | **16.0** |
| 2^18 | 209.7 MB | 4.20 MB | **16.0** |
| 2^20 | 838.9 MB | 16.78 MB | **16.0** |
| 2^21 | 1677.7 MB | 33.56 MB | **16.0** |

Two float64 output buffers, one per kernel arm — `O(N)`, not `O(J·N)`. **That is why the `xcache_groups`
hazard does not occur here**: its two-kernel overhead was a fraction of a `4·J·N` int32 cache and could
exceed the saving (9.0 MB against 10.4 MB), while this one is `O(N)` against an `8·J·N` store.

**So the invariant is a ratio**: overhead against the memory the split gives back is **4.0% at every
`states_size`**, a quotient of two terms both linear in `N` — a far stronger guarantee than a small number
at one size. It depends on `J` as `4/J`, measured at `states_size = 2^18`, `J' = J/2`:

| `J` | store B/slot | peak B/slot | overhead vs saved |
| --- | --- | --- | --- |
| 10 | 80 | 16.0 | **40.0%** |
| 25 | 200 | 16.0 | 16.0% |
| 50 | 400 | 16.0 | 8.0% |
| 100 | 800 | 16.0 | **4.0%** |
| 200 | 1600 | 16.0 | **2.0%** |

`16 / (4·J)` exactly. (A `2/J` prediction made along the way was wrong by 2× — it divided the 16 B/slot by
the *full* store rather than the half a `J' = J/2` split returns. The `1/J` shape was right, the constant
was not.) **The dial is only cheap at large `J`** — 40% overhead at `J = 10` — but `J = 10` is also where
the whole feature is pointless. Counting the `13·N` `states` array the split reintroduces as well, it goes
**net-negative in memory below about `J = 7` groups** (the arithmetic is per group; it was once written
`K = 7` because the fixture had `J = 100, K = 99`). `markdown/sqd-locg-improvement-ideas.md` has that
derivation.

No script was committed for this sweep or §4 (commit `d862361` touches only `NOTES.md` and `CLAUDE.md`).

## 4. The time ratios hold at scale

1D Heisenberg n=100, J=100, one matvec:

| `N` | `states_size` | store | all-cached | `J/2` | none | `J/2` vs all | exact |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 50,072 | 65536 | 52.4 MB | 4.49 ms | 12.49 | 28.04 | **2.78×** | `0.0e+00` |
| 187,846 | 262144 | 209.7 MB | 14.70 ms | 38.21 | 79.80 | **2.60×** | `0.0e+00` |
| 501,157 | 524288 | 419.4 MB | 19.99 ms | 56.51 | 123.97 | **2.83×** | `0.0e+00` |

Stable across a 10× range in `N`, slightly better than §1's 2.45×, bit-identical at every size.

**A trap: synthetic arrays gave a 25–39× phantom.** Shape-only arrays are right for peak memory, which
depends on shapes alone, but they timed the `J/2` split at 25–39× rather than 2.6–2.8×.
`_accumulate_diagonal` stops at the first zero coefficient in a zero-padded Z group, and `rng.normal`
coefficients are never zero, so all `K = 99` terms ran per group where a real Hamiltonian's structured
groups stop far sooner. **Synthetic arrays are valid for memory and invalid for time** whenever a
kernel's trip count is data-dependent — the converse of "a broken arm flatters its own benchmark": an
unrealistically *dense* fixture punishes the arm that depends on sparsity.

## 5. Two constraints any design must respect

**Do not split both axes.** Not because of the arm count, but because `cache_level[0] = 0` is
catastrophic per matvec: over all J groups at n=60, `(0, 2)` costs **73.97 ms against `(1, 2)`'s 1.80** —
**41×**, consistent with the documented 59.8×. Any X-axis split reintroduces it, so a shared split point
would pay 41× to save diagonal bytes. A four-arm prototype measured 39.37 ms against the two-arm 4.97 at
the same diagonal split, and the cause is that arm's presence, not the arm count. **The axes are not
symmetric: the diagonal axis is cheap to split (5.1× spread), the X axis expensive (41×).** So the dial
belongs on the diagonal axis alone, orthogonal to `xcache_groups`, with `cache_level[0]` left at 1.

**One compiled variant per distinct `J'`, and power-of-two rounding makes it worse.** 13 distinct splits
produced 13 variants; rounding `J'` up to a power of two produced **16**, because both arms' lengths vary
together (`xs[:jr]` and `xs[jr:]` are two shapes, and the complement is not a power of two) — unlike the
pre-filter's `cap`, where rounding bounded the count. Acceptable because `J'` is static per solve, but a
caller sweeping it pays one compile per value, which must be documented rather than discovered.

## 6. *Which* groups to cache barely matters, only how many

Every cached group costs the same bytes (one diagonal per state), but recomputing group `g` costs `K_g`
iterations -- `_accumulate_diagonal` stops at the first zero-padded term -- so at equal bytes, caching the
largest `K_g` first removes the most work. Measured with `poc/diag_cache_order.py` on a molecular-like
Jordan--Wigner Hamiltonian from random integrals, since a spin chain cannot test it: all its diagonal terms
land in the identity-X group, which sorts first, so a prefix already picks it (`K_g = [23, 2, 2, ...]` on
an XXZ chain with a field).

n=18, density 0.25: `J = 2641`, `K_g` quartiles 8/8/8 (min 6, max 62), `states_size` 16384, 330 MiB full
store; matvecs warm, 15 interleaved rounds, each arm checked against the full-cache product:

| cached | `K` left, prefix | `K` left, largest | matvec, prefix | matvec, largest | paired | ratio |
| --- | --- | --- | --- | --- | --- | --- |
| `J/8` | 19250 | 18486 | 167.4 ms | 162.5 ms | 14/15 | **1.030×** |
| `J/4` | 16412 | 15846 | 145.9 ms | 141.8 ms | 15/15 | **1.028×** |
| `J/2` | 10882 | 10566 | 102.1 ms | 99.8 ms | 15/15 | **1.023×** |

Whole `ground_locg` solves at `J/8` (prefilter `(32, 2)`, batched, as `run_sqd` drives it): 49.9 s against
48.8 s, **1.023×**, same eigenvalue to 12 digits and the same 107 iterations. n=14: 1.013--1.027×, 5/5
paired at every fraction.

- **The gain is the `K`-left difference, near one for one**: matvec time fits `a + b·(K left)` (n=14:
  `a` ≈ 3.1 ms, `b` ≈ 1.65 µs per term), so the order can only win what its `K` coverage differs by.
- **That difference is small because `K_g` is concentrated**: most groups here share `K = 8`, so ranking
  moves a few large groups. A denser n=12 variant (density 1.0, `K_g` 8/22/79) differs more -- 56% more
  `K` removed at `J/8`, 7% at `J/2` -- a work count, not timed.
- **So the API should expose the count, not an order.** Largest-first is never worse and costs a host
  argsort over `.c`, but it is worth ≤3% on these fixtures and 0% on spin chains; the dial that matters is
  `J'`.

## 7. XXZ: the identity group is the best byte, not most of the win

The 1D periodic XXZ chain (`Jz = 1`, `poc/sqd_multinode.py`'s `xxz_hamiltonian` and `xxz_krylov_states`,
the Néel-Krylov subspace) puts every diagonal term in the identity-X group: `K_g = [n, 2, 2, ...]` over
`J = n + 1` groups. Three arms, each checked against the full-cache product -- `J' = 0` (`(1, 0)`),
`J' = 1` (the identity group cached, the rest `(1, 0)`) and `J' = J` (`(1, 2)`) -- timed as whole
`ground_locg` solves driven as `run_sqd` drives them (prefilter `(32, 2)`, batched), memory from XLA
`memory_analysis` of the whole solve (inputs + temp). `N = 60000`, `states_size = 65536`, warm:

| n | `(1, 0)` | `J' = 1` | `(1, 2)` |
| --- | --- | --- | --- |
| 24 | 555 ms, 11.5 MiB | **441 ms (1.26×), 13.1 MiB** | 196 ms (2.83×), 23.8 MiB |
| 32 | 1123 ms, 13.6 MiB | **814 ms (1.38×), 15.1 MiB** | 325 ms (3.46×), 29.8 MiB |

Eigenvalue and iteration count identical across arms (44 and 59 iterations).

- **The identity group is a third of the recompute, not most of it.** It holds `n` of the `3n` terms (the
  `n` hops have `K = 2` each), and the solve fits §6's linear-in-`K`-left model: removing a third of the
  terms predicts 436 ms at n=24, measured 441. A prediction that it would recover "nearly all" of the
  full-cache speed was wrong for exactly this reason.
- **But it is the best byte by far**: ~73 ms saved per MiB at n=24, against ~29 ms for the whole cache,
  because it is `1/(n+1)` of the store and a third of the work. Further hop groups buy linearly.
- **Ordering is moot on XXZ**: a prefix *is* largest-first.
- **Cost**: `(1, 2)`'s store is `8·(n+1)` B/slot; `J' = 1` costs ~24 B/slot (its 8 B store plus §3's
  16 B/slot two-kernel temp).

No script was committed (commit `50c3bfe` touches only `NOTES.md` and two `markdown/` files); the fixture
builders are in `poc/sqd_multinode.py`, and the two-kernel form is `poc/diag_cache_order.py`'s `two_arm`.

## 8. What it means

- **The dial is `J'`, a count**, on the diagonal axis only, `cache_level[0] = 1`; half the diagonal memory
  costs 2.45× the full-cache solve (2.6–2.8× per matvec up to N = 501k), bit-identical. Largest-`K_g`-first
  ordering is worth ≤3%.
- **It only matters when `(1, 2)` does not fit but `(1, 0)` has room to spare.** It interpolates between
  the two and cannot go below `(1, 0)`: every cached group adds 8 B/slot, plus the 16 B/slot two-kernel
  temp, on top of `(1, 0)`'s footprint.
- **So it cannot help a memory-bound `(1, 0)` run** (spinchain at n ≥ 30, where `(1, 0)` already fills
  memory): there is no memory left to spend on it. Sparse transition pairs (`poc/sparse/pairs.md` §6) are
  the lever for that regime, since they shrink `(1, 0)`'s own source cache.
- **Cheap only at large `J`** (overhead `4/J` of the saving) and net-negative below about `J = 7`, where a
  dial should refuse rather than silently cost memory.

## 9. Open

1. **Building the dial**: a `J'` parameter, documented as one compile per distinct value, sized by exact
   byte arithmetic (§2), refusing below `J ≈ 7`, and rejected in combination with `xcache_groups`.
2. **GPU and sharding**: every number here is one laptop CPU, single-device.

## 10. The script

`poc/diag_cache_order.py` produces §6 only. It builds the molecular-like JW Hamiltonian (`molecular_like`:
one-body terms plus a random `--density` fraction of two-body terms, `rng` seed 0), draws random states,
precomputes the full `(1, 2)` source and diagonal caches, and times `two_arm` (`(1, 2)` over the cached
groups plus `(1, 0)` over the rest) for a prefix against largest-`K_g`-first at `J/8`, `J/4`, `J/2`, each
checked against the full-cache product; then three whole `ground_locg` solves per order at `J/8`.

| argument | default | meaning |
| --- | --- | --- |
| `--num-qubits` | 18 | qubits `n` of the JW Hamiltonian |
| `--num-states` | 16384 | random bitstrings drawn (`states_size` rounds up to a power of two) |
| `--density` | 0.25 | fraction of two-body `a+_p a+_q a_r a_s` terms kept |
| `--rounds` | 15 | interleaved timing rounds per arm for the matvec table |

Run: `uv run python poc/diag_cache_order.py [--num-qubits 18] [--num-states 16384] [--density 0.25]`.
