# The Z parity of the diagonal: XOR-folding and one pass per group

Prompted by Ozaki scheme II (Ozaki, Uchino and Imamura, arXiv:2504.08009v4). That paper emulates FP64 GEMM
with exact INT8 tensor-core GEMMs combined through the Chinese Remainder Theorem. Prototyped on branch
`parity-xor` on one laptop CPU (Apple M1 Max, 10 cores, 64 GiB), 2026-09-27. Nothing here is in the
library. The GPU is the open question. Every number comes from `poc/parity_xor.py` (§5).

## The idea

The paper's technique itself needs a large, compute-bound GEMM, and neither `sqd` nor `ground_locg` has
one:

- the matvecs are gathers and scatters;
- the Rayleigh–Ritz step is 3×3;
- the inner products are `O(N)` reductions.

Its accuracy side is closed already: CLAUDE.md's closed investigations cite the same group's
Ozaki-scheme CG result under compensated summation. What carries over is narrower: **INT8 tensor cores
do exact integer GEMM fast.**

One piece of `sqd` has that shape, the diagonal. For each X group,

```text
D(s) = Σ_k c_k (−1)^{popcount(s & z_k)} = Σ_k c_k − 2 · Σ_k c_k · (popcount(s & z_k) & 1)
```

`get_diagonal` computes it in a `while_loop` over the K real terms. Each term is one fused pass over the
`(N, B)` packed states, running `_z_parity = sum(bitwise_count(s & z), axis=1) & 1`.
`matvec=Matvec.INDICES` rebuilds every diagonal in every matvec. At n=60 `type1`, `2^17` it takes 6.31 s
against `TABLES`' 2.25 s, and the diagonals are all `TABLES` caches, so **~64% of an `INDICES` solve is
the diagonal**. Two ways to make it cheaper were tried:

1. **XOR-fold (§1).** Parity is linear over XOR, so `bitwise_count(xor_reduce(s & z)) & 1` gives the same
   bit with one popcount per state. At `B % 8 == 0` the bytes can also be bitcast to `uint64` words
   first.
2. **One pass per group (§2).** Compute all K parities of a group together, so that the states are read
   once rather than K times. There are two forms:
   - a broadcast AND and popcount over `(chunk, K, B)`;
   - the paper's shape, an exact int8 `dot` of unpacked bits, `(chunk, 8B) @ (8B, K)` → int32.

   Either way, `(1 − 2·(P & 1)) @ c` finishes the diagonal.

The fixtures are spinchain's open XXZ at n=60 on Hamming-shell subspaces (`poc/sparse_pairs.py`'s
`spinchain_problem`), where B=8:

| fixture | groups J | real terms ΣK | identity group K | other groups K |
| --- | --- | --- | --- | --- |
| `type1` | 62 | 179 | 59 | ≤ 2 |
| `type2` | 120 | 239 | 59 | ≤ 2 |

## 1. XOR-folding changes nothing, and the `uint64` form is slower

At every setting, every group's diagonal is **bit-identical** across the three arms.

| `2^17` | `type1` | `type2` |
| --- | --- | --- |
| identity group's `get_diagonal`, current | 5.19 ms | 5.26 ms |
| XOR-fold over bytes | 5.16 ms (1.00×) | 5.15 ms (1.02×) |
| XOR-fold over `uint64` words | 9.14 ms (0.57×) | 8.38 ms (0.63×) |
| warm `sqd(INDICES)`, current, median of 5 | 6.31 s | 10.29 s |
| warm `sqd(INDICES)`, `uint64` words | 9.19 s (0.69×, won 0/5) | 14.54 s (0.71×, won 0/5) |
| `TABLES`, the ceiling | 2.25 s | 4.36 s |

- **The popcount was never the cost.** XLA already fuses the per-byte popcount and the sum into the pass.
- **The `uint64` form adds a copy.** Its reshape and bitcast put an extra copy in every one of the K
  passes.
- The two whole-solve arms return the same eigenvalue.

The byte fold was not run through a whole solve, since it ties on the diagonal.

## 2. One pass per group loses 4–9×, and the loss does not close with N

Each arm computes all J diagonals, which is one `INDICES` matvec's diagonal work.

- **Padding.** The one-pass arms split the identity group off, so the others are trimmed to their own
  largest K (2) rather than zero-padded to the rectangle's 59. That padding would cost 3658 or 7080
  term-passes against ΣK = 179 or 239.
- **Unpacked bits.** The `dot` arm is handed the unpacked bits, as if cached. That is 64 B/slot at n=60,
  against `INDICES`' 248 B/slot of sources.
- **Checks.** All three arms return exactly the same diagonals, `max |diff| = 0.0`.

The times are medians of 3 interleaved rounds, and the current code won every round.

| size | fixture | current | broadcast | int8 `dot` |
| --- | --- | --- | --- | --- |
| `2^15` | `type1` | 7.82 ms | 18.90 ms (0.41×) | 29.93 ms (0.26×) |
| `2^17` | `type1` | 22.9 ms | 79.9 ms (0.29×) | 124 ms (0.18×) |
| `2^17` | `type2` | 34.1 ms | 144 ms (0.24×) | 232 ms (0.15×) |
| `2^19` | `type1` | 74.1 ms | 315 ms (0.24×) | 499 ms (0.15×) |
| `2^19` | `type2` | 109 ms | 583 ms (0.19×) | 953 ms (0.11×) |
| `2^21` | `type1` | 277 ms | 1216 ms (0.23×) | 2012 ms (0.14×) |
| `2^21` | `type2` | 440 ms | 2265 ms (0.19×) | 3790 ms (0.12×) |

**The premise was wrong.** The premise was that the per-term loop wastes memory traffic by re-reading
the states K times. It does not: at `2^21` it does 179 passes over 16 MiB in 277 ms, about 1.35 G
state-terms/s. Each pass is one fused, streaming, elementwise loop, which is what a CPU core runs well.

The one-pass forms add work the loop never does:

- they materialize a `(chunk, K)` sign matrix;
- the broadcast arm also builds a `(chunk, K, B)` temporary, which spills out of cache at K=59;
- they finish with a small matmul;
- the `dot` arm reads 8× the bytes.

Nothing in this scales in their favour, and `type2`'s many small groups make it worse.

## 3. What it means

- **On CPU the diagonal is already about as fast as it gets.** Neither form of the parity beats the
  per-term streaming loop. The lever for an `INDICES` solve's diagonal share is still *caching* it:
  `TABLES`, or a partial diagonal cache (`NOTES.md`, "A partial *diagonal* cache works"). A faster
  recompute is not the lever.
- **Per-group lookup tables have a low ceiling on these Hamiltonians, so they were not built.**
  - The idea: a group whose Z terms touch r qubits has a 2^r-entry table.
  - But the groups other than the identity have K ≤ 2, so a table saves at most one pass each, and the
    identity group's Z_iZ_{i+1} chain touches every qubit.
  - Against the 179 or 239 passes that bounds the gain to about 1.25× of an `INDICES` solve. `TABLES`
    (2.8×) and `ELL` already beat that when they fit.
- **Nothing in the paper improves `ground_locg`.** Its work is memory-bound `O(N)` vector operations,
  and the precision directions are closed (CLAUDE.md, "Closed investigations").

## 4. Open

1. **The int8 `dot` on a GPU.**
   - The question is whether XLA lowers `jnp.dot(int8, int8, preferred_element_type=int32)` to an INT8
     tensor-core GEMM through cuBLASLt, fast enough to recover the 7–9× the CPU form loses.
   - It needs K padded to a multiple of 4: the identity group's 59 to 60 or 64.
   - Run `poc/parity_xor.py onepass` beside `poc/sparse-pairs.md` §7 item 6.
   - It only matters for sharded `INDICES` runs, since single-device runs have `ELL`.
2. **A Hamiltonian with large-K non-identity groups.** Here only the identity group has more than two
   terms. A fixture where many groups carry tens of Z terms would test the one-pass premise where it is
   strongest. That is molecular-like JW, which is `poc/sparse_pairs.py general`'s second fixture.

## 5. The script

Everything above comes from `poc/parity_xor.py`. Its first positional argument picks the mode:

| mode | what it measures |
| --- | --- |
| `fold` | swaps `rqutils.sqd._diagonal._z_parity` for each XOR-fold arm (clearing jax's caches so each retraces), checks every group's diagonal bit-identical, times the identity group's `get_diagonal`, then runs interleaved warm `sqd(matvec=Matvec.INDICES)` rounds (current against `uint64`) plus one `TABLES` solve |
| `onepass` | all J diagonals three ways (the library loop, the split broadcast, the split int8 `dot`), each checked against the library, over `3 × --rounds` interleaved rounds with a paired win count |

Flags:

- `--num-qubits` (60), `--patterns` (`type1 type2`), `--delta` (0.5) and `--log2-size` (17) select the
  fixture.
- `--rounds` (5) sets the round count.

The tables above ran `fold` at the defaults, and `onepass --rounds 1` at `--log2-size` 15, 17, 19 and 21.
