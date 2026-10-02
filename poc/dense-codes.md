# Dense kernels: `"tables"` diagonals as codes, and z = 0 terms folded out of `"indices"`

`poc/dense_codes.py` (§6) at `c0061df`, one Apple M1 (8 cores, 16 GiB), 2026-10-03. Fixture as
`poc/dense-tune.md`: spinchain's open-XXZ `xxz` at n=60, `δ = 0.5`, `type1` (`J = 62`) and `type2`
(`J = 120`), both `complex128`, Hamming-shell subspaces around both Néel states. Neither lever is in the
library.

## 1. The levers

A dense matvec scans the `J` X groups over all `N` states, but each group's diagonal takes few values,
and many of its terms have no Z part:

- **`codes`** (`"tables"`): each group's cached diagonal stored as a `uint8` index into its distinct
  values, read back as `table[code]`. Encoded on the device by `jnp.unique(size=256)` one group at a time
  in a scan, so no `(J, N)` diagonal is live at once. Real groups keep a float64 table, as `"tables"`
  keeps float64 diagonals. At most 7 distinct values per group at `2^14` (9 at `2^17`, `type1`).
- **`fold`** (`"indices"`): a term with `z = 0` has parity 0 in every state, so its coefficient is a
  constant. Each group's z = 0 terms are summed into one constant on the host, and the matvec computes
  parities only for the rest, bucketed by that remaining count. That folds 60 of `type1`'s 120
  non-identity terms (the XX of every hop, and the X fields) and 119 of `type2`'s 180.

The reference for each is `run_sqd` at the matching `matvec`.

## 2. Results

`2^14`, 3 interleaved rounds after a warm-up, per solve iteration with setup inside. `temp` is the whole
solve's `temp_size_in_bytes`; `operator` is the bytes the matvec reads per `J × N` slot, `states` excluded:

| arm | pattern | per iteration | wins | `(2, N)` matvec | temp | operator | iterations |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `"indices"` | `type1` | 7.80 ms | | 3.58 ms | 6.8 MiB | 4.26 B | 93 |
| **`fold`** | `type1` | **1.28×** | 3/3 | 1.38× | 6.6 MiB | 4.26 B | 93 |
| `"indices"` | `type2` | 13.19 ms | | 6.00 ms | 10.4 MiB | 4.14 B | 88 |
| **`fold`** | `type2` | **1.50×** | 3/3 | 1.69× | 10.4 MiB | 4.14 B | 88 |
| `"tables"` | `type1` | 3.52 ms | | 1.25 ms | 14.5 MiB | 12.13 B | 92 |
| `codes` | `type1` | 0.84× | 0/3 | 0.73× | 8.1 MiB | 5.13 B | 92 |
| `"tables"` | `type2` | 6.72 ms | | 2.72 ms | 25.5 MiB | 12.13 B | 88 |
| `codes` | `type2` | 0.80× | 0/3 | 0.77× | 12.4 MiB | 5.13 B | 88 |

`codes` matches `"tables"` bit for bit at both matvec widths, with zero eigenvalue difference. `fold`
does at `type2`. At `type1` it differs by 2.3e-16 relative and its eigenvalue by 4.8e-16, with the same
iteration count: each group's diagonal is exact, since its one z = 0 term comes first, but bucketing by
the remaining count changes the order the groups are summed in.

The only `2^17` figure is `type1` with §3's barrier variant of `codes`, which lost more at `2^14` than
the shipped arm: 0.88× per iteration and 0.89× on `(2, N)`, temp 116.3 → 63.9 MiB, operator 12.13 → 5.02
B/slot, 96 iterations each.

## 3. `codes`' cost is the batched matvec

The single-vector matvec is even (0.96–1.04×), so the table read itself is nearly free. The loss is in
`(2, N)`, which `run_sqd` always uses. An `optimization_barrier` on `table[code]`, to materialize the
diagonal once rather than once per batch row, made it worse (1-D 0.77×, `(2, N)` 0.72× at `2^14`), and is
not in the script. The cause is not established.

## 4. What it means

`fold` is a pure win, 1.28–1.50× at no memory cost, and it is bit-identical wherever the groups' order
survives. The gain scales with the folded share: half of `type1`'s terms, two-thirds of `type2`'s.

`codes` halves the solve's temp and cuts the operator 12.13 → 5.13 B/slot for 16–20% per iteration.
Against `fold`, not `"tables"`, it is still 1.45× (`type1`) and 1.05× (`type2`) faster for +0.87–0.99
B/slot. So with codes, `"indices"` is nearly dominated on time and memory together, as the checkpoint
guessed. That holds at `2^14` only.

## 5. Open

1. **Sizes `2^17`–`2^19`.** The defaults were cut to `2^14` for wall time. Only the one `2^17` `type1`
   barrier figure in §2 exists past that, and the ratios may move with `N`.
2. **The GPU.** `"indices"` on a GH200 already matches `"tables"` (`poc/sparse/gpu.md` §9), so `fold`
   could put it ahead, and `codes`' bandwidth saving is what a GPU rewards.
3. **Why `codes`' `(2, N)` matvec loses** (§3), before deciding on it.
4. **`fold` in the library:** a constant per group in `_bucket_args`, `const +` in `_apply_buckets`.
   Also applies to `"onthefly"` (`--matvecs onthefly`), unmeasured.

## 6. The script

`poc/dense_codes.py`, its argparse checked against this section:

| flag | default | meaning |
| --- | --- | --- |
| `--num-qubits`, `--delta` | `60`, `0.5` | the `xxz` fixture |
| `--patterns` | `type1 type2` | from `poc/eigenpair_check_scale.patterns` |
| `--log2-sizes` | `14` | subspace sizes `2^k` |
| `--matvecs` | `indices tables` | `indices`/`onthefly` run `fold`, `tables` runs `codes` |
| `--rounds` | `3` | interleaved solves per arm after one warm-up |

Matvecs must agree with the reference to `1e-12` relative, and eigenvalues too. The `bit-identical`
column compares both matvec widths exactly. Runs here: the defaults (§2), and before the defaults were
cut, `--patterns type1 --matvecs tables --log2-sizes 14 17 --rounds 3` with the barrier of §3 (its `2^17`
row is §2's `2^17` figure). No committed script reproduces the barrier arm: it was the one-line
`jax.lax.optimization_barrier(table[code])` in `codes_kernel`.
