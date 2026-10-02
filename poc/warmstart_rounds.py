"""Warm starts for a *recovery-grown* sqd sequence, where the answer moves every round.

`poc/warmstart.py` could not judge any warm start: on its fixtures the previous eigenvector was ~99% of the
next answer, so even the rejected zero-padded start won 19/19 (`markdown/skqd-warmstart-negative-result.md`).
Here the subspace grows the way spinchain's recovery grows one -- ranked `|<c|H|v>|` expansion, doubling
per round (`sparse.pairs.recovery_subspace`) -- and the ground energy moves every round, so new states
carry real weight. The write-up is `poc/warmstart-rounds.md`.

Arms, each driving `ground_locg` as `run_sqd` does (batched, `Σ|c|` bound) at every prefilter setting:

* ``cold``   -- `run_sqd`'s own start: spread seed plus the signed min-diagonal weight;
* ``zeropad`` -- previous eigenvector, exactly zero on new states: the REJECTED shape, kept as the control
  that must lose before any warm arm is read;
* ``spnew``  -- previous eigenvector, spread values on new states (the 2026-09-17 shape);
* ``mix``    -- unit previous eigenvector plus ``w`` times the unit cold start, everywhere;
* ``first``     -- previous eigenvector plus the first-order amplitude ``<c|H|v> / (E - H_cc)`` on new states;
* ``first+sp``  -- ``first`` plus a small spread.

Every warm arm carries the previous round's COLD eigenvector, so warm-start error cannot compound, and
joins by state code, never position (`uniquify_states` lex-sorts). The verdict is total operator
applications (prefilter + solve, counted inside the jitted solve), with each arm's energy checked
against the cold arm's.

Run: uv run python poc/warmstart_rounds.py [--top 17] [--pattern type1]
"""

import argparse
import functools

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from eigenpair_check_scale import hamming_shells, patterns, xxz
from sparse.pairs import recovery_scores, run_sqd_vinit, term_masks, to_codes

from rqutils.ground_locg import ground_locg
from rqutils.paulis.symplectic import PauliSumXZ
from rqutils.sqd import apply_h, get_diagonal, get_xsource, uniquify_states
from rqutils.sqd._states import _pad_states

PREFILTERS = ((32, 2), (16, 1), None)
ARMS = ("cold", "zeropad", "spnew", "mix", "first", "first+sp")
MIX_W = 0.3  # weight of the unit cold start in `mix`
FO_SPREAD = 0.1  # weight of the unit cold start in `first+sp`


def unpack(codes, n):
    return ((codes[:, None] >> np.arange(n, dtype=np.uint64)) & np.uint64(1)).astype(np.uint8)


def row_codes(packed, n):
    """Each packed row's state code, and a filler mask (fillers have byte 0's high bit set)."""
    rows = np.asarray(packed)
    filler = (rows[:, 0] >> 7).astype(bool)
    bits = np.unpackbits(rows, axis=1)[:, 1 : n + 1]  # bit 0 is the pad bit
    return to_codes(bits), filler


def build_round(ham, codes, n):
    """The `(1, 0)` operator on one subspace, its packed states, row codes and cold start."""
    size = 1 << int(np.ceil(np.log2(len(codes))))
    packed = uniquify_states(_pad_states(PauliSumXZ.pack_states(unpack(codes, n)), size), size)
    rcodes, filler = row_codes(packed, n)
    assert np.array_equal(np.sort(rcodes[~filler]), codes), (
        "packed rows do not decode to the subspace"
    )
    xs = jnp.stack([get_xsource(x, packed) for x in ham.x])
    matvec = functools.partial(apply_h, xsources=xs, zsignatures=ham.z, coeffs=ham.c, states=packed)
    d0 = get_diagonal(ham.z[0], ham.c[0], packed)
    cold = np.array(run_sqd_vinit(ham, packed, size, d0))
    return matvec, rcodes, filler, cold, np.asarray(d0).real


def carried(rcodes, filler, prev_codes, prev_vec):
    """Previous eigenvector placed on this round's rows by code; ``new`` marks rows not carried."""
    pos = np.searchsorted(prev_codes, rcodes).clip(max=len(prev_codes) - 1)
    hit = (prev_codes[pos] == rcodes) & ~filler
    out = np.zeros(len(rcodes), complex)
    out[hit] = prev_vec[pos[hit]]
    return out, ~hit & ~filler


def unit(v):
    return v / np.linalg.norm(v)


def starts(cold, rcodes, filler, new, carry, fo_codes, fo_amp, e_prev, d0):
    """Every arm's start vector for one round; fillers are exactly zero in all of them."""
    base = unit(carry)
    spread = unit(np.where(filler, 0.0, cold))
    # As poc/warmstart.warm_start: the new block's norm equals the carried block's.
    spnew = base.copy()
    spnew[new] = cold[new] / max(np.linalg.norm(cold[new]), 1e-300)
    first = base.copy()
    pos = np.searchsorted(fo_codes, rcodes).clip(max=len(fo_codes) - 1)
    has = new & (fo_codes[pos] == rcodes)
    den = e_prev - d0[has]
    den = np.where(np.abs(den) < 1e-2, np.copysign(1e-2, den), den)  # guard a near-resonant state
    first[has] = fo_amp[pos[has]] / den / np.linalg.norm(carry)
    out = {
        "cold": cold,
        "zeropad": base,
        "spnew": spnew,
        "mix": base + MIX_W * spread,
        "first": first,
        "first+sp": unit(first) + FO_SPREAD * spread,
    }
    for name, v in out.items():
        assert not np.any(v[filler]), f"{name} puts weight on filler slots"
    return {k: jnp.asarray(v.astype(complex) if np.iscomplexobj(v) else v) for k, v in out.items()}


def counting(matvec):
    """The operator, plus a host counter of applications (a batched ``(k, N)`` call counts ``k``)."""
    count = [0]

    def fn(v):
        k = v.shape[0] if v.ndim == 2 else 1
        jax.debug.callback(lambda: count.__setitem__(0, count[0] + k))
        return matvec(v)

    return fn, count


def solve(matvec, x0, prefilter, bound):
    fn, count = counting(matvec)
    e, vec, niter, ok = ground_locg(
        fn, x0, prefilter=prefilter, prefilter_hi=bound, batch_matvec=True, maxiter=3000
    )
    jax.block_until_ready(e)
    return float(e), np.asarray(vec), int(niter), bool(ok), count[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-qubits", type=int, default=60)
    parser.add_argument("--pattern", default="type1")
    parser.add_argument("--delta", type=float, default=0.5)
    parser.add_argument("--seed-log2", type=int, default=12)
    parser.add_argument("--top", type=int, default=17, help="last round's log2 size")
    args = parser.parse_args()
    n = args.num_qubits
    op = xxz(n, args.delta, *patterns(n)[args.pattern])
    ham = PauliSumXZ.from_paulisum(op)
    masks = term_masks(op)
    bound = float(np.abs(np.asarray(ham.c)).sum())
    codes = np.sort(to_codes(hamming_shells(n, 1 << args.seed_log2, np.random.default_rng(0))))
    prev = (
        None  # (codes, cold eigvec aligned with codes, cold energy, first codes, first amplitudes)
    )
    print(f"n={n} {args.pattern} delta={args.delta}: operator applications (iterations) per arm")
    header = " ".join(f"{a:>13}" for a in ARMS)
    print(f"{'N':>8} {'prefilter':>9} {header}   w_new  zeropad loses?")
    while True:
        matvec, rcodes, filler, cold, d0 = build_round(ham, codes, n)
        row = {}
        if prev is None:
            e_c, v_c, it, ok, mv = solve(matvec, jnp.asarray(cold), (32, 2), bound)
            assert ok
            print(f"{len(codes):>8} {'(32, 2)':>9} {f'{mv} ({it})':>13}   (first round: cold only)")
        else:
            carry, new = carried(rcodes, filler, *prev[:2])
            x0 = starts(cold, rcodes, filler, new, carry, prev[3], prev[4], prev[2], d0)
            ref_e = None
            for pf in PREFILTERS:
                for a in ARMS:
                    e, v, it, ok, mv = solve(matvec, x0[a], pf, bound)
                    if a == "cold" and pf == (32, 2):
                        e_c, v_c, ref_e = e, v, e
                    row[pf, a] = (e, it, ok, mv)
            w = np.abs(v_c) ** 2
            w_new = float(w[new].sum() / w.sum())
            for pf in PREFILTERS:
                cells = []
                for a in ARMS:
                    e, it, ok, mv = row[pf, a]
                    bad = (not ok) or abs(e - ref_e) > 1e-8
                    cells.append(f"{mv} ({it}){'!' if bad else ''}")
                zp, cd = row[pf, "zeropad"], row[pf, "cold"]
                loses = zp[3] >= cd[3] or abs(zp[0] - ref_e) > 1e-8 or not zp[2]
                print(
                    f"{len(codes):>8} {pf!s:>9} "
                    + " ".join(f"{c:>13}" for c in cells)
                    + f"   {w_new:.3f}  {'yes' if loses else 'NO'}",
                    flush=True,
                )
        if len(codes) >= 1 << args.top:
            break
        # Carry the cold solution; grow by the top |<c|H|v>|, keeping the signed scores for `first`.
        v = v_c[~filler][np.argsort(rcodes[~filler])]  # aligned with sorted `codes`
        cand, score = recovery_scores(masks, codes, v, signed=True)
        take = np.argsort(-np.abs(score), kind="stable")[: len(codes)]
        prev = (codes, v, e_c, cand, score)
        codes = np.sort(np.concatenate([codes, cand[take]]))
    print("'!' marks an arm that did not converge or whose energy differs from cold's by > 1e-8.")


if __name__ == "__main__":
    main()
