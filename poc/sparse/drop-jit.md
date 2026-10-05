# `_drop_zeros` as one jitted program

`poc/sparse/drop_jit.py` (§4) at `ab4a2a3` plus the change, one Apple M1 (8 cores, 16 GiB), 2026-10-05.
Fixture as `poc/sparse/prune.md`: spinchain's open-XXZ `xxz` at n=60, `δ = 0.5`, `type1` and `type2`,
Hamming-shell subspaces around both Néel states. In the library since, as `_compact`.

## 1. The cost

`_drop_zeros` (`poc/sparse/prune.md` §2) ran op by op, so a first build at a new size class compiled
each of its ops: 0.15–0.36 s of first-build overhead (`prune.md` §3). Its output length depends on the
nonzero count, so it cannot join the build's other jitted steps; it syncs for the count, then
`_compact` runs the filter as one program, compiled per `(input shape, length, chunk, size)`, with the count traced.

## 2. Measured

One fresh process per arm, so `first` pays every compile; `warm` is the min of 5 further calls. The
eager arm returns the library's arrays exactly (asserted):

| pattern, N | entries kept | eager first / warm | jit first / warm | first build eager / jit |
| --- | --- | --- | --- | --- |
| type1 2^14 | 32768 | 325 / 1.3 ms | 92 / 0.4 ms | 0.56 / 0.34 s |
| type1 2^17 | 65536 | 484 / 3.2 ms | 129 / 1.8 ms | 0.78 / 0.43 s |
| type2 2^14 | 65536 | 354 / 1.5 ms | 99 / 0.6 ms | 0.61 / 0.36 s |
| type2 2^17 | 491520 | 474 / 11.3 ms | 140 / 5.2 ms | 0.77 / 0.44 s |

A run before shipping, with the jitted form local to the script, gave the same figures within 10%.

## 3. What it means, and open

The first call is 3.4–3.8× faster, 0.23–0.36 s saved per new size class, which is 39–45% of a first build;
warm it is 1.8–3.3×, but at 0.4–11 ms that hardly matters against a solve. Same output, so nothing to
trade. **Open**: the GH200, where `prune.md` §6 measured the warm filter; and whether the persistent
compile cache already hid the eager cost across processes, which this script does not enable.

## 4. The script

`poc/sparse/drop_jit.py`, against its argparse: `--num-qubits` (60), `--delta` (0.5), `--patterns`
(`type1 type2`), `--log2-sizes` (`17 19`; the table used `14 17`), `--rounds` (5). Each cell runs two child
processes (`--child`, internal). The eager arm is a copy of the pre-change `_drop_zeros`.
