# `_drop_zeros` as one jitted program

`poc/sparse/drop_jit.py` (§6) at `ab4a2a3` plus the change, one Apple M1 (8 cores, 16 GiB), 2026-10-05;
the NVIDIA GH200 120GB in §3, at `3b50443`.
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

## 3. On the GH200

`--log2-sizes 20 22`, `--patterns type1 type2`, default `--rounds`:

| pattern, N | entries kept | eager first / warm | jit first / warm | first build eager / jit |
| --- | --- | --- | --- | --- |
| type1 2^20 | 524288 | 1080 / 7.1 ms | 248 / 0.5 ms | 1.58 / 0.74 s |
| type1 2^22 | 2097152 | 1067 / 7.2 ms | 254 / 1.1 ms | 1.60 / 0.78 s |
| type2 2^20 | 4718592 | 1044 / 7.1 ms | 258 / 0.8 ms | 1.55 / 0.76 s |
| type2 2^22 | 23068672 | 1076 / 7.5 ms | 259 / 2.0 ms | 1.65 / 0.83 s |

The first call is 4.0–4.4× faster, 0.79–0.82 s saved, halving a first build (2.0–2.1×). Eager is flat at
~7 ms warm whatever the entry count, so its cost there is per-op dispatch, not work; jitted it is
3.8–14×.

## 4. A persistent compile cache

`--cache-min-secs S` (at `c3ba388`, M1, `--log2-sizes 14 17`): each child runs twice on a fresh cache
with that minimum compile time, and the second run is reported.

| pattern, N | `S = 0`: eager first / jit first | first build eager / jit | `S = 1.0`: eager first / jit first |
| --- | --- | --- | --- |
| type1 2^14 | 46 / 21 ms | 0.09 / 0.07 s | 328 / 94 ms |
| type1 2^17 | 64 / 24 ms | 0.11 / 0.07 s | 486 / 125 ms |
| type2 2^14 | 47 / 21 ms | 0.09 / 0.07 s | 364 / 96 ms |
| type2 2^17 | 76 / 32 ms | 0.13 / 0.09 s | 481 / 140 ms |

At JAX's default `S = 1.0` the cache stores nothing here: every compile in the build is under a second,
so both arms match §2. At `S = 0` it hides most of the cost, the whole build's compiles included, and the
jitted filter is still 2.2–2.7× faster on its first call. So the cache did not already hide the eager cost
for a caller using the defaults.

**On the GH200** (`--log2-sizes 20 22`, at `c3ba388`):

| pattern, N | `S = 0`: eager first / jit first | first build eager / jit | `S = 1.0`: eager first / jit first | first build eager / jit |
| --- | --- | --- | --- | --- |
| type1 2^20 | 176 / 36 ms | 0.29 / 0.15 s | 997 / 173 ms | 1.49 / 0.67 s |
| type1 2^22 | 192 / 38 ms | 0.33 / 0.18 s | 1028 / 177 ms | 1.56 / 0.71 s |
| type2 2^20 | 183 / 37 ms | 0.31 / 0.16 s | 1008 / 188 ms | 1.52 / 0.70 s |
| type2 2^22 | 179 / 38 ms | 0.37 / 0.23 s | 1029 / 187 ms | 1.61 / 0.77 s |

The same conclusion, more strongly: at `S = 0` the jitted first call is still 4.7–5.1× faster. At
`S = 1.0` eager is within 5% of §3, but jit's first call is 173–188 ms against §3's 248–259, and the build
0.67–0.77 s against 0.74–0.83. Unexplained: whether the default cache stored a compile over 1 s, or the
second process gained from something outside JAX (a CUDA driver cache), this design cannot separate.

## 5. What it means, and open

The first call is 3.4–3.8× faster, 0.23–0.36 s saved per new size class, which is 39–45% of a first build;
warm it is 1.8–3.3×, but at 0.4–11 ms that hardly matters against a solve. Same output, so nothing to
trade. On a GH200 the gain is larger (§3). A persistent cache at its default threshold hides none of it
(§4), on either backend. **Open**: the source of §4's GH200 `S = 1.0` jit gain.

## 6. The script

`poc/sparse/drop_jit.py`, against its argparse: `--num-qubits` (60), `--delta` (0.5), `--patterns`
(`type1 type2`), `--log2-sizes` (`17 19`; §2 and §4 used `14 17`, §3 `20 22`), `--rounds` (5),
`--cache-min-secs` (off; §4 used `0` and `1.0`). Each cell runs one child process per arm (`--child`,
internal), two with the cache. The eager arm is a copy of the pre-change `_drop_zeros`.
