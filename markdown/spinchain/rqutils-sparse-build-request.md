# Sparse `Matvec` kernels on SKQD's pipeline: the host build decides the result

From the `spinchain` side, in reply to `rqutils-matvec.md`. Installed: `dev-0.2.0` at `bd51046`. We have
migrated `cache_level` to `matvec=`, and every kernel returns the right energy. This note covers
performance only.

> **Summary.** In our pipeline, `Matvec.TABLES` beats all three sparse kernels at n=30, by 1.19x
> (`ELL`) to 1.69x (`CSR`) end to end. At n=60, `PAIRS`/`ELL` beat it, but by less than your table
> suggests. On a fixed subspace the sparse solve itself behaves as your note says. The loss comes from
> the **host-side build**:
>
> - it costs about 1–1.8 µs per state;
> - it is paid on **every** call, and SKQD never solves the same subspace twice;
> - it grows with `N`, not with the number of stored entries.
>
> Asks 1 and 2 would move the result. Asks 3 and 4 are smaller.

## 1. What we measured

Machine: macOS, 10 CPU cores, 64 GiB, AC power, jax compile cache warm. `solve_atol = 1e-6` and
`prefilter = (32, 2)` throughout, via `sqd(packed=True)` from `ground_state_packed`.

### 1a. End to end: an n=30 SKQD replay from hardware samples

The replay has three Krylov rungs, up to 952,542 states. Recovery then runs at most 1,000,000 states,
with about 5 rounds. The two rounds were interleaved, and round 2 is reported:

| kernel | wall | vs `TABLES` | peak RSS | rung solves | recovery solves |
| --- | --- | --- | --- | --- | --- |
| `TABLES` | **43.4 s** | 1.00x | 4.81 GiB | **7.8 s** | **14.1 s** |
| `ELL` | 51.4 s | 1.19x | 5.35 GiB | 9.3 s | 20.0 s |
| `PAIRS` | 58.4 s | 1.35x | **4.35 GiB** | 9.2 s | 26.1 s |
| `INDICES` | 64.5 s | 1.49x | 4.65 GiB | 14.6 s | 28.2 s |
| `CSR` | 73.2 s | 1.69x | 4.68 GiB | 11.6 s | 39.2 s |

Round 1 ranked the kernels identically, and each figure moved by 10% or less.

Energies: `TABLES`/`INDICES`/`PAIRS` give −11.5808342832, and `CSR`/`ELL` give −11.5808342826. That is
within `atol`, so fine, but note it contradicts "every kernel returns the same energy" if a reader takes
that to mean bit-identical. Worth one word in your docs.

### 1b. One fixed subspace, repeated calls in one process (warm, calls 1–2)

At n=30, rung 2, 952,542 states:

| kernel | `Built the … operator` | rest of the call | total |
| --- | --- | --- | --- |
| `TABLES` | — | 3.3 s | **3.3 s** |
| `ELL` | 1.0–1.2 s | 2.5–3.0 s | 3.6–4.1 s |
| `PAIRS` | 0.9 s | 4.1 s | 5.0 s |
| `CSR` | 1.0 s | 5.1–5.5 s | 6.1–6.5 s |
| `INDICES` | — | 6.8–8.5 s | 6.8–8.5 s |

At n=60, rung 2, 2,396,047 states:

| kernel | build | rest of the call | total | ru_maxrss |
| --- | --- | --- | --- | --- |
| `TABLES` | — | 11.6 s, 20.6 s | 11.6–20.6 s | 4.53 GiB |
| `ELL` | 4.3–4.4 s | 5.8 s | 10.1 s | 3.90 GiB |
| `PAIRS` | 4.1–4.2 s | 5.8 s | 9.9 s | 3.90 GiB |

The two `TABLES` calls at n=60 differ by 1.8x, so treat that row as noisy.

### 1c. Why the n=30 and n=60 results differ: the hit rate

We counted `get_xsource(x[g], states) >= 0` per off-diagonal X group on the same subspaces:

| subspace | X groups | hit rate: mean (min–max) | stored transitions per state |
| --- | --- | --- | --- |
| n=30, 952k | 31 (1 diagonal) | **0.198** (0.081–0.311) | 5.95 |
| n=60, 2.4M | 61 (1 diagonal) | **0.001** (0.000–0.002) | 0.05 |

At n=30 the hit rate is about twice your fixtures' 8–12%. The sparse kernels save only about 5x of the
dense slots there, and `ELL`'s solve beats `TABLES` by only about 1.2x. The build erases that.

At n=60 the solve wins about 2x, as your note says. But the build is now over 40% of the call. The
remaining 5.8 s is almost all cost that does not scale with the entry count, since 0.05 entries per
state leaves the matvec with nearly nothing to do.

## 2. Why the build is never amortized in SKQD

Every `sqd()` call in an SKQD run is on a **new** subspace. But consecutive subspaces are nearly the
same set:

- **Krylov rungs are nested.** Rung `k`'s basis is the cumulative union of rungs `0..k`, so each solve's
  states are a superset of the previous one's (262k → 624k → 952k above).
- **Recovery rounds replace about 5% of the basis.** A logged n=30 round at `max_dim = 3M` read
  `dropped=154001, added=154001` against a 3,000,000-state basis, so 95% of the states carry over.

Today each call rebuilds the whole operator from the full state list.

## 3. Asks

1. **A faster build**, ideally on device or jitted in chunks. At about 1 µs/state (n=30) and 1.8 µs/state
   (n=60) it grows with `N` and is almost independent of the stored-entry count. At n=60 it takes
   4.3 s to store 123k entries, 0.05 per state. Can the build skip transitions that miss the subspace
   without scanning every `(group, state)` pair on the host? If most of the build is already that
   scan, then a faster scan is the ask.
2. **An incremental build: reuse an operator across a small change of basis.** For example, an optional
   `previous=(states, operator)` argument, or a returned handle, from which `sqd()` builds the new
   operator by adding only the rows and columns of the new states. The nested rungs make this pure
   growth, and recovery's 5% turnover is growth plus deletion. We would take either form. If the index
   shift makes it impractical, saying so closes the ask.
3. **The residual check on the sparse kernels.** Your matvec note says it repeats the full `J`-fold
   source search. At n=60 we cannot tell how much of the non-build 5.8 s that is. Two options:
   - a figure from your side;
   - a check that reuses the operator the solve just used.

   Our old guard went through the same trade-off, independence against cost. If the fresh search is the
   point, keeping it is fine and a figure is enough.
4. **The `Found ground eigenpair in … seconds` timer does not block.** On the dense kernels it reads
   `0.000085 s` for a warm 262k-state solve that takes 0.9 s of wall time. `run_sqd` returns before the
   result is ready, and the wait happens later in `_checked_eigval`'s host read. A
   `block_until_ready()` before the timestamp, or moving the log after `_checked_eigval`, would make the
   INFO line the real solve time. That line is the only per-phase timing `sqd()` exposes.

## 4. What we are doing meanwhile

- The default stays `Matvec.TABLES`.
- `PAIRS` is a candidate for our memory-bound config, since it beat `INDICES` on both time and memory
  in §1a. We will measure it at that config's actual 4M states before switching.
- `ELL` stays available through `[solver] matvec` for n=60-like subspaces, where §1b shows it ahead.
- We have not tested the sparse kernels under a mesh, since they are single-device by design.

## 5. Follow-up on `dev-0.2.1` (`e4da18a`): the build is fixed, and retracing is what remains

Re-measured with the same harness after `28005b0` ("Sparse builders search sources on the host in
threads"). Thank you: the build is 3–6x faster, and ask 1 is answered for our purposes.

| subspace | kernel | build before → after | total per warm call before → after |
| --- | --- | --- | --- |
| n=30, 952k | `ELL` | 1.0–1.2 s → **0.39 s** | 3.6–4.1 s → **3.0–3.1 s** (`TABLES`: 3.1–3.4 s) |
| n=30, 952k | `PAIRS` | 0.9 s → 0.15 s | 5.0 s → 4.3 s |
| n=30, 952k | `CSR` | 1.0 s → 0.26 s | 6.1–6.5 s → 5.4–5.5 s |
| n=60, 2.4M | `ELL` | 4.3 s → **0.75 s** | 10.1 s → **5.5–5.9 s** (`TABLES`: 11.3–11.7 s) |
| n=60, 2.4M | `PAIRS` | 4.1 s → 0.66 s | 9.9 s → 5.3–6.5 s |

In the n=30 pipeline, `TABLES` still wins, by 1.14x over `ELL` (40.9 s against 46.8 s). The rung solves
are now tied (7.9 s each). The whole remaining gap is in recovery: 17.5 s against 11.7 s in the sweep,
and 17.8 s against 12.3 s over 4 rounds in the rerun below, where we ran both kernels under
`JAX_LOG_COMPILES=1`:

| per solve | `ELL` | `TABLES` |
| --- | --- | --- |
| jit cache misses | 26–85 | 0–5 |
| lowering + XLA | 0.34–0.54 s | 0–0.13 s |
| tracing, whole run | 0.90 s over 402 traces | 0.23 s over 17 traces |

Every one of those misses was a **persistent-cache hit**, so XLA compilation itself is cheap. What recurs
is tracing and lowering `_run_sparse`'s pieces on every new subspace. The ELL bucket shapes depend on
the subspace's degree distribution, which changes with each basis. `TABLES` compiles one `run_sqd` per
`states_size` bucket and reuses it from rung 1 on.

The 5.5 s recovery gap splits as:

| cause | cost | status |
| --- | --- | --- |
| retrace + lower + XLA | ~2.0 s | measured |
| build (4 rounds × 0.4 s) | ~1.6 s | measured |
| **unexplained** | **~2 s** | not measured |

For the unexplained part, our guess is that recovery's H-expansion makes the basis closer to closed
under `H`, which raises the hit rate above the sampled rungs' ~20%. We have not measured a recovery
subspace's hit rate. We record it only as an open possibility, not a finding.

5. **Make `_run_sparse`'s compiled shapes independent of the subspace's degree distribution, so a new
   basis of the same `states_size` bucket reuses the compiled solve.** One option is to round the ELL
   bucket count and widths to a fixed ladder per `states_size`, as `states_size` itself is rounded to
   `m·2^k`. At SKQD's call pattern, where every call is a new basis of similar size, this is worth ~2 s
   of a 46.8 s n=30 run. It would also matter more at n=60, where the solve is shorter and fixed costs
   are a larger share. If the padding cost of a fixed ladder outweighs the retrace, a figure from your
   side closes the ask.

One behavioural note from the same bump, for the record rather than as an ask: `698616a` (real X groups
first, float64 cache) moved `TABLES` and `INDICES` apart by 1.8e-15 in energy. We had pinned
bit-identity across the two, which was our assumption and not your contract. We now compare kernels to
1e-12, and keep exact `==` only for a same-kernel replay.
