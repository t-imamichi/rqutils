# A state-major gather layout for the sparse kernels

`poc/sparse/layout.py` (§6), one Apple M1 (8 cores, 16 GiB), CPU only, 2026-10-01. Fixture as
`poc/sparse/gpu.md`: spinchain's open-XXZ `xxz` at n=60, `δ = 0.5`, `type1` (`J = 62`, `complex128`),
Hamming-shell subspaces around both Néel states. Nothing here is in the library.

## 1. The idea

`run_sqd` batches LOBPCG's matvec pair as one `(2, N)` `complex128` array, so every random index in
`"pairs"`, `"csr"` and `"ell"` (`vec.at[..., j]`, `out.at[..., i].add`) touches two places `16·N` B apart:
two cache lines (or GPU sectors), each half used. Moving the state axis first, `(N, 2)`, puts both values
in one 32 B run. The `col` arm does that inside each kernel — `moveaxis` in, the same scans on the
state-major array, `moveaxis` out — and `row` is the shipped kernel. It was the first lever proposed for
`poc/sparse/gpu.md` §3's L2 cliff, tried on CPU first because a 64 B line holds the pair the same way.

## 2. Results

Median of 5 interleaved rounds; `wins` counts rounds where `col` was faster. A ratio below 1 is `col`
slower.

| kernel | N | solve/iter `row` | `col` | ratio | wins | matvec `row` | `col` | ratio | wins | iters `row` / `col` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `"pairs"` | `2^17` | 9.29 ms | 9.43 ms | 0.98× | 0/5 | 2.35 ms | 2.75 ms | 0.85× | 0/5 | 106 / 106 |
| `"csr"` | `2^17` | 11.19 ms | 11.65 ms | 0.96× | 0/5 | 2.83 ms | 3.26 ms | 0.87× | 0/5 | 105 / 96 |
| `"ell"` | `2^17` | 7.50 ms | 7.85 ms | 0.95× | 0/5 | 2.24 ms | 2.62 ms | 0.85× | 0/5 | 96 / 105 |
| `"pairs"` | `2^19` | 52.62 ms | 55.28 ms | 0.95× | 1/5 | 11.10 ms | 13.16 ms | 0.84× | 0/5 | 64 / 64 |
| `"csr"` | `2^19` | 69.04 ms | 70.94 ms | 0.97× | 0/5 | 13.98 ms | 16.78 ms | 0.83× | 0/5 | 64 / 64 |
| `"ell"` | `2^19` | 48.59 ms | 49.90 ms | 0.97× | 0/5 | 11.50 ms | 13.54 ms | 0.85× | 0/5 | 64 / 64 |

- **`col` loses in all 30 matvec rounds and 29 of 30 solve rounds**: 0.83–0.87× on the matvec, 0.95–0.98×
  per solve iteration, where LOBPCG's own vector work dilutes it.
- **The loss does not shrink with N.** Per state, `2^19` costs 1.2–1.3× `2^17` on the matvec, so the
  vectors are outgrowing the cache, yet the ratio goes 0.85–0.87× → 0.83–0.85×. Half-used lines are not what binds
  here; latency per random access is the likelier limit — inferred, not profiled.
- Every `col` matvec equals `row`'s bit for bit; eigenvalues agree to 5.3e-15.

## 3. Iteration counts differ inside the solve

Bit-identical matvecs do not give an identical trajectory once fused into the solve: `"csr"` and `"ell"`
at `2^17` swap 105 and 96 iterations between the layouts, and a `2^10` smoke run differed by one. XLA
fuses the `moveaxis` with the surrounding vector ops, so the solve's rounding changes — not verified past
that. A whole-solve ratio would then measure the iteration count, so §2 times the solve **per iteration**.

## 4. A first run compared the shipped kernel with itself

The first run built each arm as `jax.jit(_run_sparse.__wrapped__, ...)` after patching
`_SPARSE_APPLY`, which `_run_sparse` reads at trace time. The second `jax.jit` of the same function
reused the first's trace, so both solve arms ran `row`: their lowered text was identical, and the solves
agreed to the millisecond (3.385 against 3.384 s) while the matvec, two genuinely different functions,
read 0.76×. Its solve figures are void. The script now wraps each arm in a function of its own and asserts
the two lowered solves differ.

## 5. What it means and what is open

**Don't build it for CPU.** The transposes cost more than the shared lines return, at every size measured.

Open:

1. **The GPU**, which is where §1 aimed: a 32 B sector holds exactly the `(N, 2)` pair, and
   `poc/sparse/gpu.md` §3's cliff is bandwidth-shaped where this CPU's is not. No CPU evidence now
   favours it; `poc/sparse/gpu.md` §10.4's tiled order (pairs sorted by `(i >> s, j >> s, i)`) ranks first.
   Run: `uv run python poc/sparse/layout.py --log2-sizes 19 20 21` on the GPU host.
2. **A layout carried through the solver**, so no matvec transposes: `ground_locg` would hold `(N, 2)`
   throughout. Untried; it is the only form in which the transpose cost could vanish.
3. **`2^20` and up on CPU**, and `"indices"`, which gathers from the same `(2, N)` layout.

## 6. The script

`poc/sparse/layout.py`, its argparse checked against this section:

| flag | default | meaning |
| --- | --- | --- |
| `--num-qubits`, `--delta` | `60`, `0.5` | the `xxz` fixture |
| `--pattern` | `type1` | one of `poc/eigenpair_check_scale.patterns` |
| `--log2-sizes` | `17 19` | subspace sizes `2^k` |
| `--arms` | `pairs csr ell` | sparse kernels, each run `row` and `col` |
| `--rounds` | `5` | interleaved `row`/`col` rounds after one warm-up each |

`solve` is `_run_sparse` with `return_eigvec=False`, its time divided by that solve's iteration count
(captured by wrapping `ground_locg` with a host callback, as `poc/sparse/gpu.py`); `matvec` is the bare
kernel jitted on a fixed `(2, N)` random vector passed as an argument. The host build is outside both.
Runs here: the default sweep.
