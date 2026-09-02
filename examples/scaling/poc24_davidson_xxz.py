"""Davidson vs ground_locg on a physical XXZ subspace, through apply_h, prefilter on both arms.

Closes the caveat in NOTES.md's "Davidson vs ground_locg" entry: the matvec ratios there are from
synthetic dense fixtures at n=512, unfiltered. Here the operator is a real 1D XXZ Hamiltonian
projected onto a Krylov subspace reachable from |Neel>, applied via `apply_h` at cache_level (1,2),
and the *identical* Chebyshev filter is given to both arms -- the filter is an operator-agnostic
start-vector transform, so giving it to only one arm measures the filter, not the algorithm.
"""

import argparse
import functools
import time
from typing import Any

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import scipy.sparse.linalg as spla

from rqutils.ground_locg import _chebyshev_prefilter, ground_locg
from rqutils.paulis.symplectic import PauliSumXZ
from rqutils.sqd import apply_h, get_diagonal, get_xsource, hproj, uniquify_states

parser = argparse.ArgumentParser()
parser.add_argument("--num-qubits", type=int, default=20)
parser.add_argument("--rungs", type=int, default=4)
parser.add_argument("--cap", type=int, default=4000)
parser.add_argument("--delta", type=float, default=1.0)
parser.add_argument("--bx", type=float, default=0.5)
parser.add_argument("--seeds", type=int, default=5)
parser.add_argument("--rtol", type=float, default=1e-10)
parser.add_argument("--depths", default="3,5,10,25")
parser.add_argument("--prefilter", default="32,2")
options = parser.parse_args()

DEPTHS = tuple(int(x) for x in options.depths.split(","))
PREFILTER = tuple(int(x) for x in options.prefilter.split(","))


def xxz_strings(nq, delta, bx):
    """Periodic XXZ (XX+YY+delta*ZZ per bond) plus a transverse field Bx on every site.

    Bx breaks magnetization conservation; without it the hop-generated subspace is closed under H
    and the projection is trivially block-diagonal.
    """
    strings, coeffs = [], []
    for q in range(nq):
        r = (q + 1) % nq
        for letter, coeff in (("X", 1.0), ("Y", 1.0), ("Z", delta)):
            s = ["I"] * nq
            s[q] = s[r] = letter
            strings.append("".join(s))
            coeffs.append(coeff)
    for q in range(nq):
        s = ["I"] * nq
        s[q] = "X"
        strings.append("".join(s))
        coeffs.append(bx)
    return strings, coeffs


def xxz_krylov(nq, rungs, cap, rng):
    """States reachable from |Neel> in `rungs` nearest-neighbour hops (poc12's fixture)."""
    neel = np.zeros(nq, np.uint8)
    neel[::2] = 1
    frontier = {neel.tobytes()}
    seen = set(frontier)
    bonds = [(q, (q + 1) % nq) for q in range(nq)]
    for _ in range(rungs):
        nxt = set()
        for state_bytes in frontier:
            state = np.frombuffer(state_bytes, np.uint8)
            for a, b in bonds:
                if state[a] != state[b]:
                    hopped = state.copy()
                    hopped[a], hopped[b] = hopped[b], hopped[a]
                    key = hopped.tobytes()
                    if key not in seen:
                        seen.add(key)
                        nxt.add(key)
        frontier = nxt
        if len(seen) > cap:
            break
    rows = np.frombuffer(b"".join(sorted(seen)), np.uint8).reshape(-1, nq)
    if len(rows) > cap:
        rows = rows[rng.choice(len(rows), cap, replace=False)]
    return np.unique(rows, axis=0)


def build(nq, rungs, cap, delta, bx, seed):
    """Assemble the apply_h matvec at cache_level (1,2), plus a sparse reference for the diagonal."""
    rng = np.random.default_rng(seed)
    strings, coeffs = xxz_strings(nq, delta, bx)
    states = xxz_krylov(nq, rungs, cap, rng)
    hamiltonian = PauliSumXZ.from_paulisum((strings, coeffs))
    states_p = PauliSumXZ.pack_states(states)
    size = 1 << int(np.ceil(np.log2(states_p.shape[0])))
    padding = np.full((size - len(states_p), states_p.shape[1]), 255, dtype=np.uint8)
    packed = uniquify_states(np.append(states_p, padding, axis=0), size)
    arrays = hamiltonian.arrays
    xsources = jax.numpy.stack([get_xsource(x, packed) for x in arrays.x])
    diagonals = jax.numpy.stack(
        [get_diagonal(arrays.z[g], arrays.c[g], packed) for g in range(arrays.x.shape[0])]
    )
    matvec = functools.partial(apply_h, xsources=xsources, diagonals=diagonals)
    sparse = hproj((strings, coeffs), states, unique_states=True)
    return (
        matvec,
        packed.shape[0],
        float(np.sum(np.abs(coeffs))),
        sparse,
        states.shape[0],
    )


# A Python-level wrapper CANNOT count matvecs here: `ground_locg` runs `body` under
# `jax.lax.while_loop` and the prefilter under `jax.lax.scan`, so a counting closure is invoked once
# per *trace*, not per execution. Measured: 7 counted against a true 429 (niter=143). Counts below
# are therefore analytic, from the solver's own reported iteration count.
#
#   ground_locg: body_iter0 = 1 matvec, body_iter1 = 2, body() = 3 each, niter counts body() calls
#                (verified by a maxiter=0,1,2,5 sweep) -> 3 + 3*niter
#   prefilter:   cycles * (degree + 1), as its docstring states
#   Davidson:    plain numpy, so its counter is real
LOCG_SEED_MV = 3


def locg_matvecs(niter):
    return LOCG_SEED_MV + 3 * int(niter)


def filter_matvecs(degree, cycles):
    return cycles * (degree + 1)


class Counter:
    """Counts matvec applications. Valid ONLY for the numpy Davidson -- see the note above."""

    def __init__(self, fn):
        self.fn, self.n = fn, 0

    def __call__(self, v, *args):
        self.n += 1
        return self.fn(v, *args)


def davidson(matvec, x0, diag, max_dav, rtol, scale, maxiter=4000):
    """Minimal Davidson with a Jacobi preconditioner and a thick restart.

    n_keep = max_dav - 2 because NOTES.md records max_dav - 1 stalling outright: the retained Ritz
    vectors plus one new direction refill the space every iteration, so the subspace never grows.
    The restart's (AV)u reuse is exact here -- S is orthogonal and V/AV are current, not stale.
    """
    dim = x0.shape[0]
    n_keep = max(1, max_dav - 2)
    space = np.zeros((max_dav, dim))
    aspace = np.zeros((max_dav, dim))
    space[0] = x0 / np.linalg.norm(x0)
    aspace[0] = np.asarray(matvec(jax.numpy.asarray(space[0])))
    used = 1
    theta = float(space[0] @ aspace[0])
    for _ in range(maxiter):
        gram = space[:used] @ aspace[:used].T
        gram = 0.5 * (gram + gram.T)
        vals, vecs = np.linalg.eigh(gram)
        theta, u = float(vals[0]), vecs[:, 0]
        ritz = u @ space[:used]
        aritz = u @ aspace[:used]  # exact: V and AV are current
        resid = aritz - theta * ritz
        if np.linalg.norm(resid) < rtol * (np.linalg.norm(aritz) + abs(theta)):
            return theta, ritz, True
        if used == max_dav:  # thick restart on the n_keep lowest Ritz vectors
            k = min(n_keep, used)
            sub = vecs[:, :k]
            newv = (sub.T @ space[:used]).copy()
            newa = (sub.T @ aspace[:used]).copy()
            space[:k], aspace[:k] = newv, newa
            used = k
            gram = space[:used] @ aspace[:used].T
            gram = 0.5 * (gram + gram.T)
            vals, vecs = np.linalg.eigh(gram)
            theta, u = float(vals[0]), vecs[:, 0]
            ritz, aritz = u @ space[:used], u @ aspace[:used]
            resid = aritz - theta * ritz
        denom = diag - theta  # Jacobi
        denom = np.where(np.abs(denom) < 1e-12 * scale, 1e-12 * scale, denom)
        direction = resid / denom
        direction -= space[:used].T @ (space[:used] @ direction)
        direction -= space[:used].T @ (space[:used] @ direction)  # twice, as ground_locg does
        norm = np.linalg.norm(direction)
        if norm < 1e-14:
            return theta, ritz, False
        space[used] = direction / norm
        aspace[used] = np.asarray(matvec(jax.numpy.asarray(space[used])))
        used += 1
    return theta, ritz, False


def main():
    print(f"backend={jax.default_backend()}  prefilter={PREFILTER}  rtol={options.rtol:g}")
    print(
        f"XXZ n={options.num_qubits} delta={options.delta} Bx={options.bx} "
        f"rungs={options.rungs} cap={options.cap}"
    )
    rows = []
    for seed in range(options.seeds):
        matvec, dim, hi, sparse, nstates = build(
            options.num_qubits, options.rungs, options.cap, options.delta, options.bx, seed
        )
        diag = sparse.diagonal()
        if diag.shape[0] < dim:  # pad to the power-of-two subspace apply_h expects
            diag = np.append(diag, np.full(dim - diag.shape[0], hi))
        exact = float(spla.eigsh(sparse.astype(float), k=1, which="SA", tol=0)[0][0])
        rng = np.random.default_rng(1000 + seed)
        x0 = jax.numpy.asarray(rng.normal(size=dim))
        x0 = x0 / jax.numpy.linalg.norm(x0)

        # The identical filtered start for both arms -- same cost charged to each.
        xf = jax.block_until_ready(
            _chebyshev_prefilter(matvec, (), x0, PREFILTER[0], PREFILTER[1], hi)
        )
        filter_mv = filter_matvecs(*PREFILTER)

        # Heterogeneous record (scalars alongside result tuples), so the value type is Any:
        # `ty` otherwise unions the tuple shapes with the scalars and rejects every subscript below.
        entry: dict[str, Any] = {
            "seed": seed,
            "dim": dim,
            "nstates": nstates,
            "exact": exact,
            "filter_mv": filter_mv,
        }

        for label, xstart in (("plain", x0), ("filtered", xf)):
            res = ground_locg(matvec, xstart, rtol=options.rtol, maxiter=4000)
            jax.block_until_ready(res[0])
            t0 = time.perf_counter()  # warm: the call above compiled it
            res = ground_locg(matvec, xstart, rtol=options.rtol, maxiter=4000)
            jax.block_until_ready(res[0])
            wall = time.perf_counter() - t0
            entry[f"locg_{label}"] = (
                locg_matvecs(res[2]),
                int(res[2]),
                bool(res[3]),
                float(res[0]),
                wall,
            )
            for depth in DEPTHS:
                dcounter = Counter(matvec)
                t0 = time.perf_counter()
                val, _, conv = davidson(dcounter, np.asarray(xstart), diag, depth, options.rtol, hi)
                wall = time.perf_counter() - t0
                entry[f"dav{depth}_{label}"] = (dcounter.n, conv, val, wall)
        rows.append(entry)
        print(f"  seed {seed}: N={nstates} padded={dim} E={exact:.10f} filter_mv={filter_mv}")

    print("\n=== matvec counts (median over seeds; excludes the shared filter cost) ===")

    def med(key, idx=0):
        return float(np.median([r[key][idx] for r in rows]))

    # Vector counts: Davidson holds space+aspace (2 per depth slot) plus Ritz/resid/direction
    # temporaries; ground_locg's working set is a measured 8 (NOTES.md). Matvecs are only comparable
    # at matched memory -- that is the whole point of the depth sweep.
    print("vectors/eigenpair: locg=8  " + "  ".join(f"dav{d}={2 * d + 4}" for d in DEPTHS))
    print(f"{'arm':>10} {'locg':>8} " + " ".join(f"{'dav' + str(d):>8}" for d in DEPTHS))
    for label in ("plain", "filtered"):
        cells = " ".join(f"{med(f'dav{d}_{label}'):8.0f}" for d in DEPTHS)
        print(f"{label:>10} {med(f'locg_{label}'):8.0f} {cells}")

    print("\n=== convergence (converged / total) ===")
    for label in ("plain", "filtered"):
        lc = sum(r[f"locg_{label}"][2] for r in rows)
        dc = " ".join(f"{sum(r[f'dav{d}_{label}'][1] for r in rows):8d}" for d in DEPTHS)
        print(f"{label:>10} {lc:8d} {dc}   (of {len(rows)})")

    print("\n=== max |E - exact| ===")
    for label in ("plain", "filtered"):
        le = max(abs(r[f"locg_{label}"][3] - r["exact"]) for r in rows)
        de = " ".join(
            f"{max(abs(r[f'dav{d}_{label}'][2] - r['exact']) for r in rows):8.1e}" for d in DEPTHS
        )
        print(f"{label:>10} {le:8.1e} {de}")

    print("\n=== filter speedup in matvecs (plain / filtered, incl. filter cost) ===")
    for name in ["locg"] + [f"dav{d}" for d in DEPTHS]:
        ratios = [r[f"{name}_plain"][0] / (r[f"{name}_filtered"][0] + r["filter_mv"]) for r in rows]
        print(
            f"{name:>10} median {np.median(ratios):.2f}x  min {min(ratios):.2f}x  max {max(ratios):.2f}x"
        )


if __name__ == "__main__":
    main()
