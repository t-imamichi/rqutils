# Proposal: sort-free "parity codes" for `"tables"` and `"pairs"`

Status: **for review**, 2026-10-03. Nothing here is built. Every figure marked *measured* comes from
`poc/dense-codes.md` (dense) or `poc/sparse/prune.md` (sparse); every figure marked *estimated* is
derived from those measurements by the arithmetic shown in §4, not run.

## 1. Summary

Both POCs store a cached value (a `"tables"` diagonal entry, a `"pairs"` factor) as a `uint8` index into
a small table of distinct values. Both pay for it in the **encode**: `jnp.unique`, a sort over every
entry. This proposal replaces the sort with a value that is already computed. A group's diagonal, and a
pair's factor, is `Σ_t c[g, t]·(1 − 2·p_t)`, so the parity bits `p_t` *are* the code, and the table is the
`2^K` sign patterns enumerated from the group's coefficients.

| | today (measured) | with parity codes (estimated) |
| --- | --- | --- |
| `"tables"` on a GH200, per iteration | 0.88–0.97× of itself | **1.05–1.10×** of itself, temp ~0.5× |
| `"tables"` on CPU (`2^14`), per iteration | 0.80–0.84× of itself | 0.88–0.90× of itself, but **1.18–1.51× the default `"indices"`** |
| `"pairs"` on a GH200, build + solve | 0.97–1.01× | **1.02–1.08×**, operator −27–56% |
| `"pairs"` on CPU | 1.00–1.01× solve, build +9–325 ms | ~1.00×, build cost gone, operator −22–54% |

The two outcomes I would act on:

- **On a GPU, coded `"tables"` would beat today's `"tables"` on both time and memory**, but would still
  not beat the default `"indices"` (0.93–1.04× of it, at +12–14% temp). It improves the second-best dense
  kernel, not the best.
- **On CPU, coded `"tables"` would become the fastest dense kernel** (1.18–1.51× `"indices"`) at +19–23%
  temp, a new point between `"indices"` and today's `"tables"`.

`"pairs"` gains less (1.02–1.08×), mostly as memory. This revisits the earlier decision to keep codes
POC-only, so it is gated on measurements (§5).

## 2. The mechanism

For X group `g` with `K_g` Z terms (after the fold, the Z-free term is a constant and not counted), the
value at state `s` is

    v_g(s) = const_g + Σ_{t < K_g} c[g, t] · (1 − 2·p_t(s)),   p_t(s) = popcount(s & z[g, t]) mod 2

so it depends on `s` only through the bit vector `b(s) = Σ_t p_t(s) << t`.

- **Code**: `b(s)`, computed from the parities the kernels already compute. One elementwise pass, no
  sort, no host round trip.
- **Table**: `table[g, b] = const_g + Σ_t c[g, t]·(1 − 2·bit_t(b))` for `b < 2^K_g`, a `(J, 2^Kmax)`
  array built from the coefficients alone, independent of `N`.
- **Bit-identity**: the table entry is the same sum as `get_diagonal`'s (or `_entry_factors`') for that
  state, so it must be summed **in the same order and shape**. `get_diagonal` adds sequentially from
  zero, which is easy to match. `_entry_factors` uses `jnp.sum(..., axis=-1)` over `kmax`, so the table
  must use that same reduction over that same shape. An assert in the POC checks it.

Per kernel:

- **Dense `"tables"`**: the identity group (~60 Z terms here) cannot be enumerated (`2^60`), so it stays a
  full diagonal, `d0`, cached once as `"indices"` already does. After the fold, every other group has
  `K ≤ 1` in both patterns (`type1`'s 61 groups keep 60 terms, `type2`'s 119 keep 61), so a table has 1–2
  entries. Storage is `xsources` (4 B) + code (1 B) per slot.
- **`"pairs"`**: the code is `(group, b)`, at most `J·2^kmax` values (480 at `type2`), over `uint8`. A
  host-side remap of that small table onto its distinct values (4–5 here) keeps the stored code `uint8`.
  The table needs an **exact `0`** entry for `_drop_zeros`' padding.

**Limit**: a non-identity group with `K > 8` overflows a `uint8` code. Those groups would keep a full
diagonal, or the kernel would fall back to today's storage. It doesn't happen in spinchain's patterns;
I would make it raise in the POC and decide the fallback only if a real Hamiltonian hits it.

## 3. What changes, and what doesn't

- **No API change.** `Matvec.TABLES` keeps its name and contract ("indices and factors cached"); only its
  storage changes. `"pairs"` likewise. `apply_h(diagonals=)` keeps accepting float diagonals; the coded
  form is internal to `run_sqd`/`sqd`.
- **Sharding**: codes `(J, N)` shard as the diagonals do, and the tables are small and replicated. It
  should be transparent, but it needs a `test/sharded/` case with the spec asserted (`CLAUDE.md`,
  "Sharding tests"). `"pairs"` stays single-device.
- **The residual check** already recomputes diagonals rather than reading the cache, so it keeps vouching
  for the codes independently.
- **Library size**, estimated: dense ~25 lines (encode in `run_sqd`'s `"tables"` branch, `table[code]` in
  `_apply_h_kernel`'s `"tables"` arm); `"pairs"` ~15 lines (code out of `_entry_factors`, remap, one line
  in `_apply_pairs`).

## 4. How the estimates were made

**Dense, GH200** (`poc/dense-codes.md` §5). The codes kernel already wins and the solve still loses, so
the encode is the cost. Assumption: a parity encode costs no more than `"tables"`' own diagonal
precompute, so the setup difference goes to zero. Each LOBPCG iteration runs one `(2, N)` and one 1-D
matvec, so the saving per iteration is `Δ(2, N) + Δ1-D` from the kernel columns. This excludes the
prefilter's matvecs, so it is conservative.

| cell | `"tables"` ms/iter | saving | coded est. | vs `"tables"` | folded `"indices"` | vs `"indices"` |
| --- | --- | --- | --- | --- | --- | --- |
| `type1` `2^20` | 5.66 | 0.38 + 0.14 | 5.14 | 1.10× | 5.33 | 1.04× |
| `type1` `2^22` | 19.97 | 0.28 + 0.87 | 18.82 | 1.06× | 19.42 | 1.03× |
| `type2` `2^20` | 9.61 | 0.48 + 0.28 | 8.85 | 1.09× | 8.77 | 0.99× |
| `type2` `2^22` | 38.19 | 0.44 + 1.26 | 36.49 | 1.05× | 34.01 | 0.93× |

Temp (measured, with the sort encode): coded 487–3044 MiB against `"tables"`' 929–6468 MiB (~0.5×) and
`"indices"`' 435–2664 MiB (+12–14%).

**Dense, CPU** (`poc/dense-codes.md` §2, `2^14`). Here the codes kernel itself loses at `(2, N)`, so only
the encode share is recoverable: `Δsolve − Δmatvec` = 0.16 ms/iter (`type1`), 0.93 ms/iter (`type2`).

| | coded today | coded est. | `"tables"` | folded `"indices"` | est. vs `"indices"` |
| --- | --- | --- | --- | --- | --- |
| `type1` | 4.18 ms | 4.02 ms | 3.52 ms | 6.08 ms | **1.51×** |
| `type2` | 8.38 ms | 7.45 ms | 6.72 ms | 8.80 ms | **1.18×** |

Temp: coded 8.1 / 12.4 MiB against `"indices"`' 6.6 / 10.4 MiB (+19–23%). These are `2^14` only.

**`"pairs"`, GH200** (`poc/sparse/prune.md` §6). Assumption: the parity encode brings the build back to
`nonzero`'s, with the solve unchanged, since the kernel doesn't change. Build + solve, `nonzero` against
coded:

| cell | `nonzero` | coded today | coded est. | est. gain |
| --- | --- | --- | --- | --- |
| `type1` `2^20` | 0.024 + 0.068 s | 0.029 + 0.066 s | 0.024 + 0.066 s | 1.02× |
| `type1` `2^22` | 0.042 + 0.344 s | 0.052 + 0.330 s | 0.042 + 0.330 s | 1.04× |
| `type2` `2^20` | 0.032 + 0.284 s | 0.051 + 0.263 s | 0.032 + 0.263 s | 1.07× |
| `type2` `2^22` | 0.090 + 0.980 s | 0.179 + 0.897 s | 0.090 + 0.897 s | 1.08× |

The solve's temp is the solver's vectors (548–556 MiB either way), so the memory saving is the operator
itself: 28.0 → 20.5 MiB up to 592 → 262 MiB.

**What would make these wrong**: the encode being slower than assumed, which §5's step 1 measures first;
the CPU `(2, N)` loss (`poc/dense-codes.md` §3, cause unknown) persisting at larger `N`; and every figure
coming from one fixture family (spinchain XXZ). A Hamiltonian with larger `K` per group gets bigger
tables, and past `K = 8` no code at all.

## 5. Plan, with a gate at each step

1. **Dense POC on CPU** (me). Replace `encode` in `poc/dense_codes.py` with the parity encoder, add the
   default `"indices"` as a second reference, and time the encode on its own. Gate: bit-identical to
   `"tables"`, and the encode no slower than `"tables"`' diagonal precompute. If it is slower, stop: §4's
   assumption fails.
2. **Dense on the GH200** (you): `uv run python poc/dense_codes.py --log2-sizes 20 22 --rounds 5`. Gate:
   coded ≥ 1.03× `"tables"` per iteration at ≤ 0.55× its temp, in all four cells.
3. **`"pairs"` POC** (me, CPU, then your GH200 run of `poc/sparse/prune.py --log2-sizes 20 22`). Gate:
   build within +10% of `nonzero`'s, bit-identical on CPU, and the solve ratio unchanged from §6.
4. **Library** (only after your review of steps 2–3): coded storage inside `Matvec.TABLES`, then
   `"pairs"`. Tests by defect: bit-identity against float diagonals, the `K > 8` raise, a group with no
   Z-free term, and a `test/sharded/` case asserting the codes' spec.

Steps 1 and 3 are independent and run on CPU; step 4 needs your sign-off.

## 6. Decisions for you

1. **Revisit "codes stay POC-only"?** The case is now a cheap encode plus measured GPU wins, not the
   1.00× CPU result that decision was made on.
2. **Dense scope**: coded `"tables"` improves `"tables"` everywhere, but on a GPU it still trails the
   default `"indices"`. Worth a library change for the second-best GPU kernel, or for CPU users only?
3. **Overflow (`K > 8`)**: raise, keep a full diagonal for that group, or fall back to uncoded storage?
