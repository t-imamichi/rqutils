# Real X groups first, held as float64

Idea 1 of the `paulis/symplectic.py` review, 2026-09-27/28, built on branch `real-groups` on one laptop
CPU (Apple M1 Max, 10 cores, 64 GiB). The benefit is mainly `TABLES` memory, and the `INDICES` speed
depends on the Hamiltonian, so the library now **splits for `TABLES` only** (§6). §1–§5 measured the
split on every kernel. Whole-solve numbers come from `poc/real_groups.py` (§8). §4 and §5 are
ad-hoc runs that no committed script reproduces.

## The idea

`PauliSumXZ.c` has one dtype for all groups. It is complex128 if **any** Pauli string has an odd number
of Ys, because the folded `(-i)^{x·z}` phase is imaginary for that string. The dense kernels scan all X
groups in one `lax.scan`, which carries a single dtype. So one odd-Y term made every group's diagonal
complex:

- **`TABLES`** cached 16 B per state per group;
- **`INDICES` and `ONTHEFLY`** rebuilt every diagonal in complex arithmetic each matvec.

On spinchain's open XXZ, almost every group is real. The complex groups are the ones holding the `By`
field on the end sites, and the identity group is always real (its phase is `(-i)^0 = 1`).

| fixture (n=60) | groups J | complex groups | real groups |
| --- | --- | --- | --- |
| `type1`, `type4` | 62 | 1 | 61 |
| `type2`, `type3` | 120 | 2 | 118 |

## 1. What was built (branch `real-groups`, `ecb43c7`)

- **`PauliSumXZ.from_paulisum`** orders the real groups first with a stable sort, so the identity group
  stays at index 0. It records the count in a static field, `num_real_groups`.
  - The default, `0`, promises nothing and gives the old single-part behaviour.
  - An all-real sum sets it to `J`.
- **`run_sqd`** calls `_group_parts`, which returns up to two parts:
  - the leading real groups, with `c[:R].real` as float64;
  - the complex rest.

  Source indices and `TABLES` diagonals are computed per part before the solve, and the residual check
  uses the same parts.
- **`_apply_h_kernel`** gained `init`, the scan's starting value. `_apply_parts` chains one scan per part
  through it. `apply_h` and every direct caller of the kernel pass one part, as before.
- **Correctness.** The split is bit-identical to `num_real_groups=0`, because a real factor times a
  complex entry equals a zero-imaginary complex factor times it, exactly. Against the old group order
  the energy moves in the last ulp only.
- **Tests.** `TestRealGroupsFirst` and `TestRealGroupSplit`, mutation-checked:
  - dropping the reorder puts a complex group in the float64 prefix and fails 5 tests;
  - keeping the "real" part complex fails exactly the memory test.
- **Sharding.** `poc/sharding.py` passes on 4 virtual devices with a mixed fixture.

## 2. The final comparison: equal iteration counts

`run_sqd` at `atol=rtol=0` with `maxiter=100` never converges, so every arm runs exactly 100 iterations.
This is the only fair basis (§3 explains why).

The arms, all at n=60 and `2^17` states:

- **dev** is a `dev` worktree.
- **unsplit** is this branch at `num_real_groups=0`: the new group order, one complex scan.
- **split** is this branch as built.

| | dev | unsplit | split | split vs dev |
| --- | --- | --- | --- | --- |
| `type1` `INDICES` | 6.42 / 6.37 s | 6.38 / 6.39 s | 6.93 / 6.90 s | **0.92×** |
| `type1` `TABLES` | 2.25 / 2.29 s | 2.27 / 2.25 s | 2.21 / 2.17 s | 1.03× |
| `type2` `INDICES` | 9.76 / 9.75 s | 9.89 s | 8.27 s | **1.18×** |
| `type2` `TABLES` | 4.18 / 4.05 s | 4.23 s | 3.55 s | **1.16×** |

The run was stopped with 10 of its 12 processes done, which is why `type2` has one unsplit and one split
process.

XLA temp, B/slot:

| | dev | split |
| --- | --- | --- |
| `type1` `INDICES` | 433 | 457 (+24) |
| `type1` `TABLES` | 1417 | **930 (0.66×)** |
| `type2` `INDICES` | 665 | 666 |
| `type2` `TABLES` | 2577 | **1633 (0.63×)** |

- **The new group order costs nothing.** Unsplit matches dev on both fixtures.
- **`TABLES` wins on both fixtures.**
  - Memory drops by exactly 8 B per state per real group: −487 and −944 B/slot. At spinchain's N ≈ 1.5M
    with J=120, that is about 1.4 GB.
  - Time is flat to 1.16× better.
- **`INDICES` depends on the Hamiltonian.** It gains 1.18× on `type2` and loses 8% on `type1`.

## 3. Why converged solves misled

Converged warm `sqd` solves, min of 2 per process, three interleaved rounds. Both arms solve `type1` in
96 iterations and `type2` in 106.

| | dev | split | ratio | wins |
| --- | --- | --- | --- | --- |
| `type2` `INDICES` | 10.15 / 10.09 / 10.18 s | 8.97 / 8.84 / 8.91 s | 1.14× | 3/3 |
| `type2` `TABLES` | 4.42 / 4.40 / 4.32 s | 3.91 / 3.99 / 3.90 s | 1.13× | 3/3 |
| `type1` `INDICES` | 6.28 / 6.29 / 6.21 s | 6.64 / 6.57 / 6.51 s | 0.96× | 0/3 |
| `type1` `TABLES` | 2.26 / 2.20 / 2.24 s | 2.30 / 2.17 / 2.19 s | 1.02× | 2/3 |

The first write-up reported only `type2`, and so overstated the result.

Converged solves also mislead a second way. The new group order puts unsplit `type1` on the known
105-iteration trajectory (`poc/sparse-pairs.md` §10), against dev's and split's 96. The fixed costs are
then divided over different counts: the prefilter's 66 matvecs, the source-index precompute and the
residual check. So ms/iter flatters the arm that iterates longer:

- unsplit showed 63.7 ms/iter over 105 iterations;
- split showed 70.6 ms/iter over 96;
- in total time the two are nearly equal (6.72 against 6.78 s).

Hence §2's fixed-count basis.

## 4. The kernel alone is faster; inside the solve it is not

The same kernels were timed alone, jitted, with arrays passed as arguments. The medians below are for
one matvec at `2^17`. All arms agree exactly (max |diff| 0.0).

| matvec | unsplit | split | ratio |
| --- | --- | --- | --- |
| `type1` `(N,)` | 20.73 ms | 17.71 ms | 1.17× |
| `type1` `(2, N)` | 23.65 ms | 21.02 ms | 1.12× |
| `type2` `(N,)` | 31.58 ms | 30.89 ms | 1.02× |
| `type2` `(2, N)` | 37.39 ms | 32.92 ms | 1.14× |

Splitting into two parts that are **both complex** takes 23.68 against 23.69 ms (`type1`), so the second
scan itself is free. The float64 arithmetic is what buys the 12–17%.

`type1`'s `INDICES` loss therefore appears only once the kernel is compiled into `run_sqd`'s solver
loop. The compiled split program has 22 while loops against unsplit's 20, 5 more N-sized `u32` copies,
and 24 B/slot more temp. `type2`'s split program has 36 while loops.

Two ideas were tested and ruled out. Neither changed anything (whole solves, ad-hoc):

- **A length-1 scan.** `type1` has a single complex group. Padding that part with a zero group gave
  68–71 ms/iter, the same as before.
- **Fusion across the kernel boundary.** An `optimization_barrier` on the kernel's input and output
  gave 67.6–68.5 ms/iter against unsplit's 64.

**The cause was not found.**

## 5. Rejected: an explicit real × complex product

In `apply_xgrp`, `xvec * diagonal` promotes a float64 diagonal to complex128 and does a complex multiply.
Writing it as `lax.complex(xvec.real * d, xvec.imag * d)` gives two real products instead. Measured with
the harness in converged mode, per iteration, as a variant of `_dense.py` that was not kept:

| | split alone | + explicit product |
| --- | --- | --- |
| `type2` `INDICES` | 1.14× | 1.17× |
| `type2` `TABLES` | 1.10× | 1.18× |
| `type1` `TABLES` | 1.01× | 1.03× |
| `type1` `INDICES` | 0.97× | 0.94× |

It also breaks the split's exact bit-identity by one ulp (−6.068622773067091 against …092), most likely
because XLA contracts `out + xvec.real * d` into an FMA. It was reverted: a few percent, not worth losing
the exact test.

## 6. What it means

- **The robust result is `TABLES` memory:** 0.63–0.66× at equal or better speed, on both fixtures.
- **The `INDICES` result is not robust.** The kernel is faster, but on `type1` the whole solve is 8%
  slower, for a reason in XLA's compilation of the solver loop that was not found.
- **So only `TABLES` splits**; `INDICES` and `ONTHEFLY` keep one scan, pinned by
  `TestRealGroupSplit`'s HLO-equality test. That keeps the memory win and removes the `type1` risk.
  Finding the `type1` cause needs HLO-level work on the fused solver loop. Spinchain's
  `DiagCache.SPEED` is `TABLES`, so it is the mode that gains. The `TABLES`-only form was not re-timed;
  its kernel graph is the one §2 measured.

## 7. Open

1. The cause of the in-solve `type1` `INDICES` slowdown (§4).
2. Whole solves at `2^21`, since only `2^17` was measured end to end.
3. `apply_h` does not split. It takes caller arrays with no `num_real_groups`.
4. GPU.

## 8. The script

`poc/real_groups.py` runs one arm and prints one JSON object. It holds, per `matvec`:

- the warm times;
- the solver's iteration counts, read by wrapping `rqutils.sqd._solve.ground_locg` with a
  `jax.debug.callback`;
- the eigenvalue;
- `run_sqd`'s compiled argument and temp bytes per slot.

It also records `rqutils.__file__`, to prove which tree ran. Alternate the arms, with `PYTHONPATH`
pointing the baseline at a worktree of the pre-change revision (the docstring has the commands).

| flag | default | meaning |
| --- | --- | --- |
| `--label` | required | arm name in the output |
| `--num-qubits` | 60 | |
| `--pattern` | `type2` | |
| `--delta` | 0.5 | |
| `--log2-size` | 17 | |
| `--trials` | 2 | |
| `--matvecs` | `indices tables` | |
| `--unsplit` | off | this branch at `num_real_groups=0` |
| `--fixed-iterations` | 0 (off) | time `run_sqd` at `atol=rtol=0` and this `maxiter` |

The runs behind each section:

- **§2** is `--fixed-iterations 100` for dev, `--unsplit` and split, two rounds per fixture.
- **§3** is the default mode, three rounds (`--pattern type1` for its `type1` rows).
- **§5** is the default mode against the reverted `_dense.py` variant.
