# Proposal: sort-free "parity codes" for `"pairs"`

Status: **for review**, 2026-10-03. Nothing here is built. Every figure marked *measured* comes from
`poc/sparse/prune.md` or `poc/sparse/gpu.md`; every figure marked *estimated* is derived from those
measurements by the arithmetic in §4, not run.

**Scope: `"pairs"` only.** An earlier draft also proposed coded `"tables"` diagonals. That half is dropped.
`"tables"` is now behind `"pairs"` on one CPU (3.1–4.0× per `sqd` call, at more memory) and on one GPU, and
behind the folded `"indices"` on a GPU mesh (`poc/sparse/gpu.md` §9, `poc/dense-codes.md` §5). Improving
it would improve a kernel no measured setting picks. Its one remaining case, a CPU mesh, is unmeasured.
`poc/dense-codes.md` keeps the dense measurements.

## 1. Summary

`"pairs"` stores each nonzero transition as `(i, j, d)`, the factor `d` a `complex128` (16 B). The POC in
`poc/sparse/prune.md` §4 stores `d` as a `uint8` index into the operator's few distinct factors (4–5 for
spinchain's XXZ), and builds that index with `jnp.unique`, a sort over every entry. The sort is what makes
codes break even. This proposal derives the code from parities the build already computes, with no sort.

| GH200, `type1`/`type2` `2^20`–`2^22` | today (measured) | with parity codes (estimated) |
| --- | --- | --- |
| solve vs the shipped `nonzero` | 1.03–1.09× | 1.03–1.09× (kernel unchanged) |
| build vs `nonzero` | +5–89 ms | ~+0 ms |
| build + solve vs `nonzero` | 0.97–1.01× | **1.02–1.08×** |
| operator | −27–56% | −27–56% |

On CPU the solve is 1.00–1.01× (measured), so the gain there is the operator memory (−22–54%) and
removing the build's +9–325 ms (`prune.md` §4).

This revisits the earlier decision to keep factor codes POC-only, and is gated on measurements (§5).

## 2. The mechanism

`_entry_factors` computes each pair's factor as

    d = Σ_{t < kmax} c[g, t] · (1 − 2·p_t),   p_t = popcount(states[i] & z[g, t]) mod 2

so `d` depends only on the group `g` and the parity bits `b = Σ_t p_t << t`, both already in hand.

- **Raw code**: `g · 2^kmax + b`, from the parities `_entry_factors` computes anyway. No sort.
- **Raw table**: `(J, 2^kmax)` values, enumerated on the host from `c` alone, independent of `N`. That is
  480 entries at `type2` (`J = 120`, `kmax = 2`), more than a `uint8` holds.
- **Remap**: `np.unique` over that small raw table maps each raw code to its distinct value. There are
  4–5 here, so the stored code is `uint8`, and the stored table is the distinct values padded to 256 as
  in the POC. That is one gather per entry through a `J·2^kmax` lookup.
- **Zeros**: `_drop_zeros` reads `d != 0`, which becomes `table[code] != 0`. Its padding needs a code
  whose table entry is exactly `0`, which the remap must guarantee even when no stored pair is zero.
- **Bit-identity**: the raw table must be summed exactly as `_entry_factors` sums, a
  `jnp.sum(..., axis=-1)` over the same `kmax`-wide shape, so each table value equals the factor it
  replaces. The POC asserts it against today's `nonzero` arm.

**Limit**: `J·2^kmax` grows with `kmax`, the most Z terms in any off-diagonal group. Past ~`kmax = 16` the
raw table stops being small, and past 256 *distinct* values the code widens (`np.min_scalar_type`, as in
the POC) toward the ~1.25× worst case `prune.md` §4 records. Spinchain's patterns have `kmax = 2`.

## 3. What changes, and what doesn't

- **No API change.** `Matvec.PAIRS` keeps its name and contract; the operator tuple becomes
  `(d0, i, j, code, table)` internally.
- **Library size**, estimated ~15 lines in `rqutils/sqd/_sparse.py`: `_entry_factors` returns the raw
  code alongside `d`, a host remap, `_drop_zeros` keyed on the decoded factor, and one `table[code]` line
  in `_apply_pairs`.
- **The residual check** (`_sparse_residual`) reads the unfiltered `pairs` and recomputed diagonals, none
  of the cached operator, so it keeps vouching for the codes independently.
- **Single-device**, as `"pairs"` already is.

## 4. How the estimates were made

`poc/sparse/prune.md` §6, GH200 at `d0a9997`. The kernel is unchanged, so the solve keeps its measured
time. The assumption: the parity encode brings the build back to `nonzero`'s, since it replaces a sort with
one gather.

| cell | `nonzero` build + solve | codes today | codes est. | est. gain |
| --- | --- | --- | --- | --- |
| `type1` `2^20` | 0.024 + 0.068 s | 0.029 + 0.066 s | 0.024 + 0.066 s | 1.02× |
| `type1` `2^22` | 0.042 + 0.344 s | 0.052 + 0.330 s | 0.042 + 0.330 s | 1.04× |
| `type2` `2^20` | 0.032 + 0.284 s | 0.051 + 0.263 s | 0.032 + 0.263 s | 1.07× |
| `type2` `2^22` | 0.090 + 0.980 s | 0.179 + 0.897 s | 0.090 + 0.897 s | 1.08× |

The solve's temp is the solver's vectors (548–556 MiB either way), so the memory saving is the operator
itself: 28.0 → 20.5 MiB at `type1` `2^20`, 592 → 262 MiB at `type2` `2^22`.

**What would make these wrong**: the remap gather costing more than assumed, which step 1 measures; the
solve gain (1.03–1.09×, one GH200 run of 5 rounds) not holding on other GPUs; and every figure coming from
one fixture family. A Hamiltonian with many distinct factors gets a wider code and less saving.

## 5. Plan, with a gate at each step

1. **POC on CPU** (me). Replace `encode` in `poc/sparse/prune.py` with the parity encoder and time the
   build on its own. Gate: `codes` bit-identical to `nonzero` on CPU, and the build within +10% of
   `nonzero`'s. If it is slower, stop: §4's assumption fails.
2. **GH200** (you): `uv run python poc/sparse/prune.py --log2-sizes 20 22`. Gate: `codes` ≥ 1.02× `nonzero`
   build + solve in all four cells.
3. **Library** (only after your review of step 2), with tests by defect: bit-identity against float
   factors, a padding entry decoding to exactly `0` when no stored pair is zero, and the code widening
   past 256 distinct values.

## 6. Decisions for you

1. **Revisit "codes stay POC-only"?** The case is now a cheap encode plus measured GPU wins, not the
   1.00× CPU result that decision was made on.
2. **Is −27–56% of the `"pairs"` operator worth ~15 library lines** if step 2 lands at the low end
   (1.02×)? The memory is the more certain half of the gain.
