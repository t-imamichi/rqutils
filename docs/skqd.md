# Feasibility: sample-based Krylov quantum diagonalization (SKQD) with rqutils

Assessment date: 2026-08-03. Investigated against branch `metal` (`a822a22`).

## Short answer

Yes — the library is close to purpose-built for it, and `sqd.py` handles more of SKQD than SQD
proper needs. But **`svsim` has a bug that breaks the Krylov half**, and you would hit it
immediately.

## Mapping SKQD onto the modules

| SKQD step | Library support |
|---|---|
| Reference state + Trotterized $e^{-iHk\Delta t}$ | `svsim` — **blocked, see below** |
| Sample bitstrings in the computational basis | Not present. Trivial to add (`abs(psi)**2` → `jax.random.categorical`) |
| Union samples across all $k$ into subspace $S$ | **Free** — `sqd` uniquifies internally |
| Diagonalize $H$ projected onto $S$ | `sqd` — verified correct |

That third row is the nice surprise: `sqd`'s docstring advertises a "possibly redundant set of
bitstrings", and `uniquify_states` does a lexsort-dedup inside the JIT. A union-of-samples across
timesteps is exactly that, so you concatenate and pass it straight in.

## What was verified

`sqd` itself is sound. Full-space, it reproduces `eigvalsh` exactly, and on a random 20-state
subspace it matches the dense projection $H[S,S]$ to 10 digits — which also confirms that
`format(i, '0nb')` bitstring rows line up with statevector index `i`, so there is no endianness
trap between the two modules.

## The blocker: `rqutils/svsim.py:110`

`do_svsim` never applies the $(-i)^{x \cdot z}$ factor from its own documented convention
$Q = (-i)^{xz} Z^z X^x$. It computes `1.j * gate.sin * signs * xstate` with no phase term.

This hits exactly the gates where `x` and `z` overlap — from the `match` in `to_circuitxz`, that is
**`ry` and `y`** (`rx`, `rz`, `rzz`, `x`, `z`, `cz` all have $x \cdot z = 0$ and are fine). Isolated
`ry(0.7)` on $|0\rangle$ returns `[0.939, +0.343j]` where it should be `[0.939, +0.343]` — right
magnitude, off by a factor of $i$.

Since the documented workflow is `transpile(basis_gates=['rx','ry','rz','rzz'])`, and that basis
emits `ry` constantly, this corrupts essentially every nontrivial circuit. A 6-qubit 4-rep Trotter
step gave `|overlap|` with Qiskit's `Statevector` of **1e-16**. Re-deriving the same step with
`(-i)^{popcount(x&z)}` restored gives **0.9999999999999991**, so that single missing factor is the
whole discrepancy.

Worth flagging: **the shipped `examples/svsim.py` masks this.** Its only check
(`examples/svsim.py:69`) is `num_nonzero == 2`, which passes — but the two nonzero states are
`00001` and `10001`, not `00000` and `11111`. It reports a GHZ state it did not produce.

Note the fix is not a one-liner: `sin` is a `float64` array, so the complex phase cannot be folded
into it as-is. Either widen `sin` to complex (for `ry`/`y`, `1.j * (-i) = 1`, so it stays
real-valued in practice) or add a separate `phase` field to `CircuitXZ`.

## Other constraints

- **Scale.** `svsim` is a dense statevector simulator ($2^n$), so Krylov-state sampling caps you
  near 32–40 qubits on HPC, well below `sqd`'s own $N \le 2^{31}$ subspace limit. Fine for
  algorithm validation; not a route to a hardware-scale demo.
- **`rqutils/sqd.py:544`** (`iterm & 255` where `iterm & 7` is intended) only affects
  `cache_level[1] == 1`, which is not the default `(1, 0)` — not a blocker.
- SKQD does not need SQD's self-consistent configuration recovery, so nothing is missing there.

## Scope of this investigation

Verified by throwaway scripts (there is no pytest suite in this repo): single-qubit rotation
correctness, Trotter-step overlap against `qiskit.quantum_info.Statevector`, the GHZ example's
actual output states, `sqd` full-space and projected-subspace eigenvalues, and the
statevector-index-to-bitstring-row convention.

Not tested: multi-device/sharded execution, sampling at scale, and any $n > 6$.
