# Warm starts for a recovery-grown sqd sequence

`poc/warmstart_rounds.py`, one laptop CPU, float64, 2026-09-26. Asked because a spinchain recovery run calls
`sqd` repeatedly with the same Hamiltonian on a growing subspace, and a profile of that loop (n=60 `type1`,
`2^12` to `2^19`) put the cost in iterations: compilation is ~0.5 s a round, 2% of the `2^19` round's
13.6 s, while solves take 27–43 iterations each with the shipped prefilter.

## The question, and why the earlier fixtures could not answer it

Warm-starting was rejected twice (`markdown/skqd-warmstart-negative-result.md`): zero-padded continuation
slowed convergence and once found a different eigenvalue, and the 2026-09-17 retest found no fixture that
could judge anything, because the previous eigenvector was already ~99% of the next answer — even the
rejected shape won 19/19. This fixture is different: the subspace grows the way spinchain's recovery
grows one (ranked `|<c|H|v>|` expansion, doubling per round), so the energy moves every round and the new
states carry 5–30% of the ground-state weight.

## The arms

Each round, from the previous round's **cold** eigenvector (so warm-start error cannot compound), joined
by state code, with fillers exactly zero, at prefilter `(32, 2)` (shipped), `(16, 1)` and off:

| arm | start |
| --- | --- |
| `cold` | `run_sqd`'s own: spread seed plus the signed min-diagonal weight |
| `zeropad` | previous eigenvector, zero on new states: the rejected shape, kept as the control |
| `spnew` | previous eigenvector, spread on new states (the 2026-09-17 shape) |
| `mix` | unit previous eigenvector plus 0.3 × the unit cold start everywhere |
| `first` | previous eigenvector plus the first-order `<c\|H\|v> / (E − H_cc)` on new states |
| `first+sp` | `first` plus 0.1 × the unit cold start |

The verdict is total operator applications (prefilter plus solve, counted inside the jitted solve). Every
arm in every round reached the cold arm's energy within 1e-8.

## 1. Results (n=60 `type1`, δ = 0.5)

Applications (iterations); "shipped" is `cold` at `(32, 2)`:

| N | shipped | `spnew` `(32, 2)` | `spnew` `(16, 1)` | `mix` `(16, 1)` | `first` `(32, 2)` | `zeropad` `(16, 1)` | weight on new |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 8192 | 366 (99) | 177 (36) | 149 (43) | 149 (43) | 138 (23) | 92 (24) | 0.295 |
| 16384 | 162 (31) | 147 (26) | 131 (37) | 137 (39) | 171 (34) | 137 (39) | 0.130 |
| 32768 | 150 (27) | 141 (24) | 119 (33) | 128 (36) | 168 (33) | 128 (36) | 0.112 |
| 65536 | 162 (31) | 153 (28) | 131 (37) | 140 (40) | 183 (38) | 143 (41) | 0.066 |
| 131072 | 168 (33) | 165 (32) | 149 (43) | 146 (42) | 204 (45) | 161 (47) | 0.047 |

`cold` at `(16, 1)` and off is worse than at `(32, 2)` in every round (173–404 and 183–354 applications).
At `(32, 2)` the rejected `zeropad` loses to `cold` in 4 of the 5 rounds, so there the fixture passes
the memo's gate.

## 2. What it means

- **A warm start alone barely helps.** At the shipped prefilter, where the gate holds, `spnew` beats
  `cold` by 1.02–1.10× in the gated rounds.
- **The gain needs a warm start *and* a lighter prefilter.** `spnew` at `(16, 1)` is a steady
  1.13–1.26× over shipped; neither half pays alone, since a cold start at `(16, 1)` is worse.
- **The first-order start is the worst warm arm with the prefilter on**, slower than `cold` at `(32, 2)`
  in every round: an amplitude that is right to first order still depletes what the filter needs.
- **The gain shrinks as the rounds grow.** Weight on new states falls from 30% to 5% as recovery
  converges, the concentration of the earlier fixtures returns, and the largest (dearest) round gains
  1.13–1.15×. Weighted by solve cost that is about **1.15× per recovery run**.
- **The 3.98× in the first warm round is one bad cold solve** (99 iterations, also seen in the profile),
  which any warm start avoids — real, but a single round.

So on this evidence warm-starting is worth ~1.15×, below the 1.4–1.9× measured for loosening the solve
tolerance (blocked by recovery's pruning coupling, `markdown/skqd-sqd-solve-tolerance.md` §4), and it
does not by itself justify reopening the declined start-vector API.

## 3. Open

1. **Other patterns.** Only `type1`, one seed core, one growth rate. Pruning, which spinchain's recovery
   also does, would raise the weight on new states and may favour a warm start more.
2. **Seeding LOBPCG's `p` slot** (`x₀` = carried, `p₀` = spread) is untested: it needs an internal seam in
   `ground_locg`'s seed steps.
3. **A per-call prefilter choice.** `(16, 1)` only pays after a warm start, so if a warm start is ever
   exposed, the prefilter default would have to follow it.

## 4. The script

`poc/warmstart_rounds.py` grows the subspace with `sparse.pairs.recovery_scores` (`signed=True` for the
first-order arm) and drives `ground_locg` as `run_sqd` does (batched, `Σ|c|` bound, `(1, 0)` operator via
`apply_h`). Arguments: `--num-qubits` (60), `--pattern` (`type1`), `--delta` (0.5), `--seed-log2` (12, the
Hamming-shell core), `--top` (17, the last round's log2 size).

```bash
uv run python poc/warmstart_rounds.py --top 17
```
