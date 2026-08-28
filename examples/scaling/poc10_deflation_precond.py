"""POC 10: a two-level / deflation preconditioner for ``ground_locg``. CPU-only, see the caveat.

The last untried deterministic candidate. ``docs/rqutils-precond-request.md:697`` names it -- "a
two-level / deflation preconditioner exploiting the near-block structure of a sampled 1D chain
(measured 0.12-0.58% nonzero at n=12-14)" -- and calls it speculative. Everything else in that
document's preconditioning line is closed, and ``docs/locg-next-candidates.md`` closes the filter
line, so this script is the remaining deterministic option.

**Why it is not just another rejected preconditioner.** Six routes to a Jacobi-style shift were
measured and rejected, all for one root cause: no viable *lower* bound on the projected operator's
minimum. ``docs/rqutils-precond-request.md:786`` then retracts the assumption that a tighter bound on
``H`` was ever the route -- ``ground_locg`` sees ``hproj(H, subspace)``, whose minimum sits 0.64-1.06x
of the projected spectral width **above** ``lambda_min(H)``, so no bound on ``H`` can reach the shift
the 1.79x used. The remaining upside is in estimating the **projected** operator's minimum. Deflation
does exactly that, and needs no bound at all: it builds a small coarse space, solves the coarse
eigenproblem *exactly*, and uses that as a local spectral approximation.

**The acceptance criterion is the relative gap, not the condition number.** From the same document:
across 12 instances ``kappa`` varies 1.21x while the relative gap varies 103x, and log-iterations
correlates **+0.77** with ``log(1/gap)`` against **-0.34 -- the wrong sign -- for kappa**. So a
candidate that improves conditioning and not the gap will look principled and do nothing. This script
reports the gap-like quantity (iterations) as the primary signal and wall clock second.

**The four constraints from ``docs/locg-next-candidates.md`` §1**, which every candidate must satisfy:
sharding-transparency (matvec + elementwise only), fixed memory (no growing basis), attacking the
spectral gap (iteration count is the only lever), and **not depleting the residual**. The fourth is the
one that killed the most plausible ideas, including the Chebyshev-filtered residual (4-20x slower
*and* wrong, alignment 0.001). This preconditioner risks it too, so the alignment is measured
directly rather than inferred from the timing.

**CPU-ONLY CAVEAT.** ``jax.default_backend() == "cpu"`` here. Deflation adds ``O(N*k)`` projection
work around each matvec, and that ratio is exactly what shifts on GPU -- the same reason
``poc9_prefilter_gpu.py`` exists unrun. A CPU result here does not transfer in either direction.

Run::

    uv run --extra qiskit python examples/scaling/poc10_deflation_precond.py
"""

import os
import sys

import jax

jax.config.update("jax_enable_x64", True)  # must precede every other jax use

import jax.numpy as jnp
import numpy as np
import scipy.sparse.linalg as sla

sys.path.insert(0, os.path.dirname(__file__))

from _scaling_common import fmt_ratio, header, timeit

from rqutils.ground_locg import ground_locg
from rqutils.paulis.symplectic import PauliSumXZ
from rqutils.sqd import hproj

try:
    from qiskit.quantum_info import SparsePauliOp
except ImportError as exc:
    raise SystemExit("needs --extra qiskit") from exc


def xxz_hamiltonian(num_qubits, delta=1.0, field=0.0, seed=0):
    """Open XXZ chain, the fixture family every published prefilter number was measured on."""
    rng = np.random.default_rng(seed)
    labels, coeffs = [], []
    for i in range(num_qubits - 1):
        for pauli, coeff in (("X", 1.0), ("Y", 1.0), ("Z", delta)):
            s = ["I"] * num_qubits
            s[i] = s[i + 1] = pauli
            labels.append("".join(s))
            coeffs.append(coeff)
    if field:
        for i in range(num_qubits):
            s = ["I"] * num_qubits
            s[i] = "Z"
            labels.append("".join(s))
            coeffs.append(field * rng.uniform(-1.0, 1.0))
    return SparsePauliOp(labels, np.asarray(coeffs))


def connected_subspace(num_qubits, target_dim, hamiltonian, seed=0):
    """One-hop expansion from a seed state.

    ``docs/skqd-sqd-solve-tolerance.md`` §7.1: never ``rng.choice(2**n, size=dim)``, which is
    3.6-6.1% dense where a real subspace is 32-44%. Every conclusion about iteration counts differs
    between the two regimes, several by two orders of magnitude. The density is reported below.
    """
    rng = np.random.default_rng(seed)
    xsigs = np.unique(PauliSumXZ.from_paulisum(hamiltonian).x, axis=0)
    # Unpack the X signatures to integer masks. PauliSumXZ reserves one pad bit at position 0, so
    # the shift is 8*nbytes - (num_qubits + 1); dropping the +1 returns a permutation (NOTES.md).
    masks = set()
    for packed in xsigs:
        bits = np.unpackbits(np.asarray(packed, dtype=np.uint8))
        val = 0
        for b in bits[8 * len(packed) - (num_qubits + 1) :]:
            val = (val << 1) | int(b)
        masks.add(val)
    masks = [m for m in masks if m]

    frontier = {int(rng.integers(2**num_qubits))}
    seen = set(frontier)
    while len(seen) < target_dim and frontier:
        nxt = set()
        for s in frontier:
            for m in masks:
                t = s ^ m
                if t not in seen:
                    seen.add(t)
                    nxt.add(t)
                    if len(seen) >= target_dim:
                        break
            if len(seen) >= target_dim:
                break
        frontier = nxt
    ints = np.sort(np.fromiter(seen, dtype=np.uint64, count=len(seen)))
    shifts = np.arange(num_qubits - 1, -1, -1, dtype=np.uint64)
    return ((ints[:, None] >> shifts[None, :]) & np.uint64(1)).astype(np.uint8)


def build_case(num_qubits, target_dim, delta=1.0, field=0.0, seed=0):
    """Assemble the dense projected operator plus its reference spectrum."""
    ham = xxz_hamiltonian(num_qubits, delta=delta, field=field, seed=seed)
    states = connected_subspace(num_qubits, target_dim, ham, seed=seed)
    # hproj returns a scipy sparse matrix. Keep it sparse for the reference solve and the density
    # measure; densify only for the JAX matvec below.
    sp = hproj(ham, states, unique_states=False)
    sp = (sp + sp.conj().T) * 0.5
    density = sp.nnz / (sp.shape[0] ** 2)
    mat = np.asarray(sp.todense())
    # Reference via sparse eigsh: CLAUDE.md forbids eigh at these sizes (77 s vs 0.02 s).
    vals, vecs = sla.eigsh(sp, k=2, which="SA")
    order = np.argsort(vals)
    vals, vecs = vals[order], vecs[:, order]
    lam_max = float(sla.eigsh(sp, k=1, which="LA", return_eigenvectors=False)[0])
    relgap = (vals[1] - vals[0]) / (lam_max - vals[0])
    return {
        "n": num_qubits,
        "mat": mat,
        "dim": mat.shape[0],
        "density": density,
        "lam0": float(vals[0]),
        "vec0": vecs[:, 0],
        "relgap": float(relgap),
        "delta": delta,
        "seed": seed,
    }


def coarse_space(mat, k):
    """Pick the coarse space: the ``k`` lowest-diagonal basis states.

    This is the "near-block structure" the request names, made concrete. The diagonal of the
    projected XXZ operator is available for free (``docs/rqutils-precond-request.md`` confirms the
    analytic diagonal matches the true projected diagonal to 0.000e+00), and the lowest-diagonal
    states are where the ground state's weight concentrates. Selection is by diagonal only -- no
    matvec and no eigensolve on the fine operator.
    """
    return np.argsort(np.real(np.diag(mat)))[:k]


def make_deflation_precond(mat, idx, theta_shift=1.0):
    """Two-level preconditioner: exact coarse solve on ``idx``, identity on the complement.

    ``M^-1 r`` replaces the residual's coarse-space component with the coarse-corrected one and
    leaves the complement untouched. The coarse operator is inverted *exactly* (dense, k x k,
    host-side at build time), so no bound on the fine operator is needed -- which is the whole
    reason this candidate survives where the six shift routes died.

    Sharding note: the returned callable is a gather, a small dense matmul and a scatter. That is
    NOT elementwise, so on a mesh it would need ``out_sharding`` plumbing to satisfy
    ``ground_locg``'s contract. Measured single-device here; the sharding question is deferred and
    flagged in the writeup rather than claimed.
    """
    sub = mat[np.ix_(idx, idx)]
    # Shift the coarse operator below its own minimum so the inverse is definite on that space.
    w = np.linalg.eigvalsh(sub)
    shift = w[0] - theta_shift
    inv = np.linalg.inv(sub - shift * np.eye(len(idx)))
    inv_j = jnp.asarray(inv)
    idx_j = jnp.asarray(idx)

    def precond(r):
        return r.at[idx_j].set(inv_j @ r[idx_j])

    return precond


def alignment(mat_j, precond, vec):
    """Cosine between the preconditioned residual and the raw residual.

    The measurement that explains the Chebyshev-filtered-residual failure (0.001 alignment, 4-20x
    slower and wrong). A preconditioner whose output is near-orthogonal to the gradient is not a
    preconditioned gradient, and the timing alone would not say so.
    """
    theta = float((vec.conj() @ (mat_j @ vec)).real / (vec.conj() @ vec).real)
    r = mat_j @ vec - theta * vec
    rp = precond(r)
    den = float(jnp.linalg.norm(r) * jnp.linalg.norm(rp))
    return abs(complex(jnp.vdot(r, rp))) / den if den > 0 else 0.0


def main():
    header("POC 10: two-level / deflation preconditioner -- CPU ONLY, does not transfer to GPU")
    print(f"backend={jax.default_backend()} devices={jax.devices()}")
    print("Primary signal is ITERATIONS (gap proxy), not kappa: log-iters correlates +0.77 with")
    print("log(1/gap) and -0.34 -- wrong sign -- with kappa. See rqutils-precond-request.md:701.")

    cases = []
    for num_qubits, dim in ((14, 3000), (16, 6000)):
        for delta in (0.5, 1.0):
            for seed in (0, 1):
                cases.append(build_case(num_qubits, dim, delta=delta, seed=seed))

    ks = (16, 64, 256)
    rows = []
    for case in cases:
        mat_j = jnp.asarray(case["mat"])
        rng = np.random.default_rng(1234)
        xinit = jnp.asarray(rng.normal(size=case["dim"]))

        def matvec(v, m=mat_j):
            return m @ v

        base = ground_locg(matvec, xinit, maxiter=1000)
        base_t = timeit(
            lambda m=matvec, x=xinit: jax.block_until_ready(ground_locg(m, x, maxiter=1000)),
            "plain",
        )
        base_e, base_v, base_it = float(base[0]), base[1], int(base[2])
        base_err = abs(base_e - case["lam0"])

        print()
        print(
            f"--- n={case['n']} dim={case['dim']} delta={case['delta']} seed={case['seed']} "
            f"density={case['density'] * 100:.2f}% relgap={case['relgap']:.2e}"
        )
        print(f"    reference lam0={case['lam0']:.10f} (sparse eigsh)")
        print(f"    plain: {base_it} iters, E err {base_err:.2e}, {base_t.min_s * 1e3:.1f} ms")

        for k in ks:
            if k >= case["dim"]:
                continue
            idx = coarse_space(case["mat"], k)
            pre = make_deflation_precond(case["mat"], idx)
            align = alignment(mat_j, pre, base_v)
            res = ground_locg(matvec, xinit, maxiter=1000, precond=pre)
            cand_t = timeit(
                lambda m=matvec, x=xinit, p=pre: jax.block_until_ready(
                    ground_locg(m, x, maxiter=1000, precond=p)
                ),
                f"deflate k={k}",
            )
            energy, vec, iters = float(res[0]), res[1], int(res[2])
            err = abs(energy - case["lam0"])
            vnp = np.asarray(vec)
            overlap = float(abs(np.vdot(vnp, case["vec0"]))) / (
                float(np.linalg.norm(vnp)) * float(np.linalg.norm(case["vec0"]))
            )
            flag = ""
            if err > max(1e-9, 10 * base_err):
                flag = "  <-- WRONG ENERGY"
            elif overlap < 1 - 1e-6:
                flag = "  <-- WRONG EIGENVECTOR"
            print(
                f"    k={k:<4d} {iters:4d} iters (vs {base_it})  E err {err:.2e}  "
                f"overlap {overlap:.7f}  align {align:.4f}  "
                f"{fmt_ratio(base_t, cand_t)}{flag}"
            )
            rows.append(
                {
                    "k": k,
                    "iters": iters,
                    "base_iters": base_it,
                    "ratio": base_t.min_s / cand_t.min_s,
                    "err": err,
                    "overlap": overlap,
                    "align": align,
                }
            )

    header("Summary")
    if not rows:
        print("no rows")
        return
    for k in ks:
        sel = [r for r in rows if r["k"] == k]
        if not sel:
            continue
        it_ratio = [r["base_iters"] / r["iters"] for r in sel]
        wall = [r["ratio"] for r in sel]
        wrong = sum(1 for r in sel if r["err"] > 1e-9 or r["overlap"] < 1 - 1e-6)
        print(
            f"k={k:<4d} iter-reduction median {np.median(it_ratio):.2f}x  "
            f"wall median {np.median(wall):.2f}x  min {min(wall):.2f}x  "
            f"losses {sum(1 for w in wall if w < 1.0)}/{len(wall)}  wrong {wrong}/{len(sel)}  "
            f"align median {np.median([r['align'] for r in sel]):.4f}"
        )
    print()
    print("CPU-only. Deflation adds O(N*k) around each matvec; that ratio is what moves on GPU.")


if __name__ == "__main__":
    main()
