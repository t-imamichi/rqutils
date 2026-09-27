# Real X groups first, held as float64

Idea 1 of the `paulis/symplectic.py` review, 2026-09-27, built on branch `real-groups` and in the
library as of that branch. Measured on one laptop CPU (Apple M1 Max, 10 cores, 64 GiB). Whole-solve
numbers come from `poc/real_groups.py` (§6). The diagonal-only figures in §2 come from an ad-hoc run
that no committed script reproduces.

## The idea

`PauliSumXZ.c` has one dtype for all groups. It is complex128 if **any** Pauli string has an odd number
of Ys, because the folded `(-i)^{x·z}` phase is imaginary for that string. The dense kernels scan all X
groups in one `lax.scan`, which carries a single dtype. So one odd-Y term made every group's diagonal
complex:

- **`TABLES`** cached 16 B per state per group;
- **`INDICES` and `ONTHEFLY`** rebuilt every diagonal in complex arithmetic each matvec.

On spinchain's open XXZ, almost every group is real:

| fixture (n=60) | groups J | complex groups | real groups |
| --- | --- | --- | --- |
| `type1`, `type4` | 62 | 1 | 61 |
| `type2`, `type3` | 120 | 2 | 118 |

The complex groups are the ones holding the `By` field on the end sites.

The identity group is always real: its phase is `(-i)^0 = 1`.

## 1. What was built

- **`PauliSumXZ.from_paulisum`** orders the real groups first with a stable sort, so the identity group
  stays at index 0. It records the count in a static field, `num_real_groups`.
  - The default, `0`, promises nothing and gives the old single-part behaviour.
  - An all-real sum sets it to `J`.
- **`run_sqd`** calls `_group_parts`, which returns up to two parts:
  - the leading real groups, with `c[:R].real` as float64;
  - the complex rest.

  Source indices and `TABLES` diagonals are computed per part before the solve, so nothing is sliced
  inside the loop. The residual check uses the same parts.
- **`_apply_h_kernel`** gained `init`, the scan's starting value. `_apply_parts` chains one scan per part
  through it, with no extra vector add. `apply_h` and every direct caller of the kernel pass one part,
  as before.

## 2. Where the time goes: the diagonal

These are the real groups' diagonals (`type2`, 118 groups) with the same coefficients as complex128 and
as float64. The float64 result is exactly the real part of the complex one. Medians of 15, from the
ad-hoc run:

| size | diagonals only | diagonal × gathered vector |
| --- | --- | --- |
| `2^17` | 41.3 → 33.9 ms, 1.22× | 49.5 → 44.8 ms, 1.10× |
| `2^21` | 535.7 → 348.5 ms, 1.54× | 646.0 → 543.1 ms, 1.19× |

The gain grows with N because the complex accumulator is twice the bytes, and it leaves the cache first.
This is the same streaming loop `poc/parity-xor.md` §2 found already fast. It gets faster only by
touching fewer bytes.

## 3. Whole solves: 1.13–1.14×, and `TABLES` at 0.63× the memory

Warm `sqd` at n=60 `type2`, `2^17` states. Three rounds alternate processes between a `dev` worktree and
this branch, each process taking the min of 2 trials. Memory is `run_sqd`'s compiled XLA temp.

| | dev | real-groups | ratio (median) | wins |
| --- | --- | --- | --- | --- |
| `INDICES` | 10.15 / 10.09 / 10.18 s | 8.97 / 8.84 / 8.91 s | **1.14×** | 3/3 |
| `TABLES` | 4.42 / 4.40 / 4.32 s | 3.91 / 3.99 / 3.90 s | **1.13×** | 3/3 |
| `INDICES` temp | 665 B/slot | 666 B/slot | ≈ | — |
| `TABLES` temp | 2577 B/slot | **1633 B/slot** | 0.63× | — |

- **The memory saving is exactly the prediction.** It is 944 B/slot = 8 B × 118 real groups. At
  spinchain's `TABLES` size (N ≈ 1.5M, J=120) that is about 1.4 GB.
- **`INDICES` holds one diagonal at a time**, so its memory is unchanged.

## 4. Correctness

- **The split is bit-identical to no split.** A real factor times a complex entry equals a complex factor
  with a zero imaginary part times it, exactly. So with the group order fixed, all three dense kernels
  give the same eigenvalue and eigenvector bits with or without the split. `TestRealGroupSplit` asserts
  that against `num_real_groups=0`.
- **The new group order moves the energy in the last ulp only.** Against the old order it went from
  −15.898283219191192 to −15.898283219191189, since the order changes the summation across groups.
- **Mutation-checked.** Dropping the reorder puts a complex group in the float64 prefix, where `.real`
  silently discards its imaginary part. That fails 5 tests. Keeping the "real" part complex fails exactly
  the memory test, which is the one that proves the split is taken, not just computed.
- **Sharded.** `poc/sharding.py` passes on 4 virtual CPU devices with a mixed fixture (50 of 99 groups
  real), worst gap 8.9e-16.

## 5. What it means

- **Both spinchain modes gain.** `DiagCache.SPEED` (`TABLES`) gains mainly memory, and
  `DiagCache.MEMORY` (`INDICES`) gains speed. Spinchain needs no code change.
- **The win scales with the real fraction and with N.** A Hamiltonian whose groups are mostly complex
  gains nothing. Spinchain's XXZ is the favourable extreme, at 98–99% real.
- **The sparse kernels are untouched.** `CSR` and `ELL` already split their factors by group, and
  `PAIRS`' equivalent (P2R) was measured and rejected (`poc/sparse-pairs.md` §4 item 5).

## 6. Open

1. **Whole solves at large N.** The whole-solve A/B ran at `2^17` only. §2 suggests the `INDICES` gain
   grows at `2^21`, but no whole solve was measured there.
2. **`apply_h` does not split.** It is public and takes caller arrays with no `num_real_groups`, so a
   spinchain call through `apply_h` still runs one complex scan. Exposing the split there would be an
   API addition.
3. **GPU.** Unmeasured. Halving the accumulator bytes should matter at least as much on a
   bandwidth-bound device.

## 7. The script

`poc/real_groups.py` runs one arm and prints one JSON object. It holds, per `matvec`, the warm `sqd`
times, the eigenvalue, and `run_sqd`'s compiled argument and temp bytes per slot. It also records
`rqutils.__file__`, which proves which tree ran. Alternate the arms, with `PYTHONPATH` pointing the
baseline at a worktree of the pre-change revision (the docstring has the commands).

| flag | default |
| --- | --- |
| `--label` | required |
| `--num-qubits` | 60 |
| `--pattern` | `type2` |
| `--delta` | 0.5 |
| `--log2-size` | 17 |
| `--trials` | 2 |
| `--matvecs` | `indices tables` |

§3 is three alternating `dev`/`new` pairs at the defaults.
