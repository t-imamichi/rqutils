"""NEGATIVE RESULT. Warm-starting sqd's growing subspace: no verdict, and the gate is why.

Tests carrying the converged eigenvector onto surviving states with `_spread_seed` on the newly added
ones -- the shape `markdowns/skqd-sqd-solve-tolerance.md` §8's stated mechanism suggests after it rejected
*zero-padded* continuation. It beats the cold baseline 1.3-1.9x on iteration count and **that number is
not quotable**: zero-padding, the rejected shape, beat both arms in all 19 rounds measured, so no
fixture here reproduces §8 or can judge a replacement for it. `NOTES.md` has the four eliminated
hypotheses, the tables, and what a genuine retest would need (relgap <= 1e-04, unreachable in this
operator family).

Three rules for reusing this, each measured:

- **Keep `zero_pad_start` as a third arm.** The `valid` column requires it to LOSE before the warm arm
  is read; it refused a verdict 19/19. Comparing against the shipped baseline alone reads as a win.
- **Delocalize with `delta`, never `bx`.** A hop-generated subspace is one Hamming sector, so every
  transverse-field term projects to exactly zero -- bit-identical `nnz` and `E0` at bx=0.3 and bx=3.0.
- **Verdict is ITERATION COUNT**, per CLAUDE.md: every arm here converges to the right eigenvalue, so
  energy cannot separate them. `|dE|` against a dense oracle runs only to catch §8's wrong-eigenvalue
  failure.

Run:
    uv run python poc/warmstart.py       # demo() self-check, then xxz_rungs arm
    uv run python -c "import examples.scaling.warmstart as m; m.run_recovery(delta=0.5)"
"""

import functools

import jax

jax.config.update("jax_enable_x64", True)  # before any rqutils import: tolerances depend on it

import numpy as np
import scipy.sparse.linalg as sla

from rqutils.ground_locg import ground_locg
from rqutils.paulis.symplectic import PauliSumXZ
from rqutils.sqd import (
    _spread_seed,
    apply_h,
    get_diagonal,
    get_xsource,
    hproj,
    uniquify_states,
)


def xxz_strings(nq, delta, bx):
    """Periodic XXZ (XX+YY+delta*ZZ per bond) plus a transverse field Bx on every site.

    **`bx` does nothing on a hop-generated subspace** -- it is one Hamming sector and single-site X
    changes weight by +-1, so every field term projects to exactly zero (bit-identical nnz and E0 at
    0.3 and 3.0). Kept only for signature parity with davidson_xxz.py, which has the same dead
    knob; use `delta` to change the projected operator. `NOTES.md` has the measurement.
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


def xxz_rungs(nq, max_rungs, cap, rng):
    """Return the cumulative Krylov subspace after each hop rung -- a growing, union-monotone list.

    This is the real SQD access pattern (`core.py:203-212` grows cumulative rungs; `recovery.py:517`
    is `subspace |= new`), and the reason a synthetic random-growth fixture would not do: the whole
    question is whether a converged eigenvector on round k helps on round k+1, which depends on how
    much of round k's *weight* survives into k+1.
    """
    neel = np.zeros(nq, np.uint8)
    neel[::2] = 1
    frontier = {neel.tobytes()}
    seen = set(frontier)
    bonds = [(q, (q + 1) % nq) for q in range(nq)]
    out = []
    for _ in range(max_rungs):
        nxt = set()
        for state_bytes in frontier:
            state = np.frombuffer(state_bytes, np.uint8)
            for a, b in bonds:
                if state[a] != state[b]:  # a hop only acts between unlike neighbours
                    hopped = state.copy()
                    hopped[a], hopped[b] = hopped[b], hopped[a]
                    key = hopped.tobytes()
                    if key not in seen:
                        seen.add(key)
                        nxt.add(key)
        frontier = nxt
        rows = np.frombuffer(b"".join(sorted(seen)), np.uint8).reshape(-1, nq)
        out.append(rows)
        if len(seen) > cap or not frontier:
            break
    return out


def site_occupancy(states, vec):
    """Per-site occupation probabilities from the current eigenvector (arXiv:2605.29521 §II.A).

    This is what `skqd/recovery.py::_site_occupancy` computes: square the eigenvector's components and
    sum the weight of every state that has site q occupied. `markdowns/rqutils-requests-2.md` notes the
    occupancy is arbitrary under degeneracy -- irrelevant here, since the fixture only needs a
    distribution that MOVES between rounds, not a canonical one.
    """
    w = np.abs(np.asarray(vec)) ** 2
    w = w / w.sum()
    return (states * w[:, None]).sum(axis=0)


def recover_configurations(states, vec, num_draws, rng, hamming=None):
    """Resample bitstrings from the current eigenvector's site occupancies.

    THIS IS THE FIXTURE'S WHOLE POINT, and the structural difference from `xxz_rungs`: the new states
    depend on the current ANSWER rather than on a fixed geometric neighbourhood. Hop rungs expand a
    fixed neighbourhood, so the subspace converges and the ground state stops moving -- measured, the
    ground-state weight landing on newly added states decayed 18.13% -> 0.81% over six rungs, which
    made every continuation scheme win and let the zero-padding control BEAT the shipped baseline. A
    fixture that cannot reproduce §8's rejection cannot judge a replacement for it.

    Draws each site independently from its occupancy (a product distribution, as the reference does),
    then projects onto the fixed Hamming-weight sector when `hamming` is given, so the samples stay in
    the physical magnetization sector rather than spraying across it.
    """
    occ = np.clip(site_occupancy(states, vec), 1e-9, 1 - 1e-9)
    draws = (rng.random((num_draws, len(occ))) < occ).astype(np.uint8)
    if hamming is not None:
        # Repair the weight rather than rejecting: rejection sampling collapses at large n, and a
        # repaired draw is still concentrated where the occupancies point.
        for row in draws:
            excess = int(row.sum()) - hamming
            if excess > 0:
                ones = np.flatnonzero(row)
                row[rng.choice(ones, excess, replace=False)] = 0
            elif excess < 0:
                zeros = np.flatnonzero(row == 0)
                row[rng.choice(zeros, -excess, replace=False)] = 1
    return draws


def build_round(strings, coeffs, states):
    """Assemble the cache_level (1,2) matvec for one subspace, plus its dense oracle.

    Returns the packed states too: the warm start needs them as join keys, because `uniquify_states`
    lex-sorts and a state's index therefore MOVES between rounds. Slicing by position would silently
    scatter the previous eigenvector across unrelated basis states.
    """
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
    # hproj outside any mesh (it raises under one) and on the raw states: the dense oracle is what
    # catches §8's wrong-eigenvalue failure, so it must not share the packed path's assumptions.
    sparse = hproj((strings, coeffs), states, unique_states=False)
    # `eigsh(k=1)` needs k < N, and the first recovery round can be a handful of states. Dense
    # `eigvalsh` below the crossover -- CLAUDE.md's rule is eigvalsh or eigsh(k=1), never eigh.
    if sparse.shape[0] < 32:
        oracle = float(np.linalg.eigvalsh(sparse.toarray())[0])
    else:
        oracle = float(sla.eigsh(sparse, k=1, which="SA", return_eigenvectors=False)[0])
    return matvec, packed, float(np.sum(np.abs(coeffs))), oracle


def cold_start(packed, dtype):
    """The shipped baseline: `_spread_seed` over the whole subspace."""
    return _spread_seed(packed.shape[0], packed, dtype, None)


def warm_start(packed, dtype, prev_packed, prev_vec):
    """Previous eigenvector on carried-over states, `_spread_seed` values on the new ones.

    The join is by packed-state BYTES, not by index. `uniquify_states` lex-sorts, so round k's index
    i and round k+1's index i are different basis states in general; a positional carry would be
    numerically plausible and physically meaningless.

    Filler slots stay zero on both branches -- `_spread_seed` zeroes them (they carry no basis state,
    so weight there places the iterate partly outside the subspace) and the carried values must not
    reintroduce any.
    """
    # `np.array`, not `np.asarray`: a JAX array converts to a READ-ONLY view, so the carry below
    # raises "assignment destination is read-only" rather than copying.
    seed = np.array(cold_start(packed, dtype))
    prev = {row.tobytes(): val for row, val in zip(np.asarray(prev_packed), np.asarray(prev_vec))}
    rows = np.asarray(packed)
    carried = 0
    for i, row in enumerate(rows):
        if row[0] >> 7:  # filler slot, marked by the high bit of byte 0
            continue
        hit = prev.get(row.tobytes())
        if hit is not None:
            seed[i] = hit
            carried += 1
    # Scale so neither part swamps the other. The carried block is a unit vector over its own states
    # and the seed is unit-scale per entry, so without this the spread would dominate at large
    # new-state counts and the warm start would decay into the cold one.
    new_mask = np.ones(len(rows), bool)
    new_mask[[i for i, r in enumerate(rows) if r.tobytes() in prev]] = False
    new_mask[(rows[:, 0] >> 7).astype(bool)] = False
    if new_mask.any():
        seed[new_mask] *= np.linalg.norm(seed[~new_mask]) / max(
            np.linalg.norm(seed[new_mask]), 1e-300
        )
    return jax.numpy.asarray(seed), carried


def run(nq=16, max_rungs=7, cap=60000, delta=1.0, bx=0.3, seed=0, rtol=1e-8, prefilter=None):
    """The hop-rung arm: cumulative XXZ Krylov subspaces, all three starts, same gate as recovery.

    `prefilter` defaults to None deliberately. With `sqd`'s shipped `(32, 2)` the filter lands so close
    to converged that iteration counts fall to 1-9 and every ratio quantizes to 3.00x -- the arms are
    then unresolvable. Pass `(32, 2)` only to check the interaction, never to read a verdict.
    """
    rng = np.random.default_rng(seed)
    strings, coeffs = xxz_strings(nq, delta, bx)
    subspaces = xxz_rungs(nq, max_rungs, cap, rng)
    dtype = PauliSumXZ.from_paulisum((strings, coeffs)).c.dtype
    print(f"n={nq} delta={delta} bx={bx} rtol={rtol:g} prefilter={prefilter}")
    print(
        f"{'dim':>7} {'carried':>8} {'cold':>6} {'warm':>6} {'zeropad':>8} "
        f"{'w ratio':>8} {'|dE|w':>9}  valid"
    )

    prev = None  # (packed, eigvec) from the previous round's COLD solve
    rows = []
    for states in subspaces:
        matvec, packed, csum, oracle = build_round(strings, coeffs, states)
        hi = csum if prefilter else None

        cold = cold_start(packed, dtype)
        _, v_cold, n_cold, ok_cold = ground_locg(
            matvec, cold, rtol=rtol, maxiter=6000, prefilter=prefilter, prefilter_hi=hi
        )
        assert ok_cold, f"cold arm did not converge at dim={packed.shape[0]}"

        if prev is None:
            # Round 0 has nothing to carry: report the cold arm alone so the table's first row is
            # honest about there being no warm arm rather than comparing an arm against itself.
            print(f"{packed.shape[0]:>7} {'-':>8} {n_cold:>6} {'-':>6} {'-':>8} {'-':>8} {'-':>9}")
        else:
            warm, carried = warm_start(packed, dtype, prev[0], prev[1])
            e_warm, _, n_warm, ok_warm = ground_locg(
                matvec, warm, rtol=rtol, maxiter=6000, prefilter=prefilter, prefilter_hi=hi
            )
            assert ok_warm, f"warm arm did not converge at dim={packed.shape[0]}"
            zp = zero_pad_start(packed, dtype, prev[0], prev[1])
            e_zp, _, n_zp, ok_zp = ground_locg(
                matvec, zp, rtol=rtol, maxiter=6000, prefilter=prefilter, prefilter_hi=hi
            )
            valid = n_zp >= n_cold or abs(float(e_zp) - oracle) > 1e-6 or not ok_zp
            de = abs(float(e_warm) - oracle)
            print(
                f"{packed.shape[0]:>7} {carried:>8} {n_cold:>6} {n_warm:>6} {n_zp:>8} "
                f"{n_cold / n_warm:>7.2f}x {de:>9.1e}  {'YES' if valid else 'no'}"
            )
            rows.append((n_cold, n_warm, n_zp, de, valid, ok_warm))

        prev = (packed, v_cold)  # carry the COLD arm, so warm-start error cannot compound

    report(rows)
    return rows


def report(rows):
    """Print the verdict, or withhold it. Shared by both drivers -- the gate is not optional."""
    good = [r for r in rows if r[4]]
    print(f"\nvalid rounds (zeropad loses): {len(good)}/{len(rows)}")
    if good:
        print(
            f"warm beats cold in {sum(1 for r in good if r[1] < r[0])}/{len(good)}, "
            f"median {np.median([r[0] / r[1] for r in good]):.2f}x, "
            f"max |dE| {max(r[3] for r in good):.1e}"
        )
    else:
        print("  FIXTURE NOT DISCRIMINATING -- zeropad never lost; verdict withheld.")
        if rows:
            print(
                f"  (warm/cold would have read "
                f"{np.median([r[0] / r[1] for r in rows]):.2f}x median -- do not quote it)"
            )


def zero_pad_start(packed, dtype, prev_packed, prev_vec):
    """§8's REJECTED shape, kept as the fixture-validity control.

    Previous eigenvector on carried states, exactly zero on new ones. This must LOSE for a fixture to
    be able to judge any continuation scheme; on `xxz_rungs` it won everywhere, which is how that
    fixture was found invalid.
    """
    prev = {r.tobytes(): v for r, v in zip(np.asarray(prev_packed), np.asarray(prev_vec))}
    rows = np.asarray(packed)
    out = np.zeros(len(rows), dtype)
    for i, row in enumerate(rows):
        if row[0] >> 7:
            continue
        hit = prev.get(row.tobytes())
        if hit is not None:
            out[i] = hit
    return jax.numpy.asarray(out)


def weight_on_new(states, vec, prev_keys):
    """Fraction of ground-state probability on states absent from the previous round.

    The fixture-validity metric. High means the previous eigenvector is genuinely incomplete, which is
    §8's regime; decaying to ~1% means it is already the answer and every scheme wins.
    """
    w = np.abs(np.asarray(vec)) ** 2
    w = w / w.sum()
    keys = [r.tobytes() for r in np.asarray(PauliSumXZ.pack_states(states))]
    mask = np.array([k not in prev_keys for k in keys])
    return float(w[mask].sum()) if mask.any() else 0.0


def run_recovery(nq=16, rounds=6, num_draws=400, delta=1.0, bx=0.3, seed=0, rtol=1e-8):
    """The recovery-style loop: solve, read occupancies, resample, union, repeat.

    Reports the zero-padding control alongside, and the weight-on-new validity column. Read the
    verdict ONLY on rounds where zeropad loses -- otherwise the fixture is not discriminating.
    """
    rng = np.random.default_rng(seed)
    strings, coeffs = xxz_strings(nq, delta, bx)
    dtype = PauliSumXZ.from_paulisum((strings, coeffs)).c.dtype
    hamming = nq // 2

    # Bootstrap from noisy-sampler-like shots, NOT from a single reference state: one state's
    # occupancies are deterministic (0/1 per site), so every draw reproduces it and the subspace never
    # grows. Real SQD seeds round 0 from quantum-sampler shots; a Neel state with random hops is the
    # cheap stand-in for the same thing -- a spread of bitstrings in the target sector.
    neel = np.zeros(nq, np.uint8)
    neel[::2] = 1
    subspace = {neel.tobytes()}
    for _ in range(num_draws):
        row = neel.copy()
        for _ in range(rng.integers(1, 4)):  # a few random hops off the reference
            a, b = rng.choice(nq, 2, replace=False)
            row[a], row[b] = row[b], row[a]
        subspace.add(row.tobytes())

    print(f"n={nq} delta={delta} bx={bx} draws={num_draws} rtol={rtol:g}  (prefilter OFF)")
    print(
        f"{'dim':>7} {'new wt':>7} {'cold':>6} {'warm':>6} {'zeropad':>8} "
        f"{'w ratio':>8} {'|dE|w':>9}  valid"
    )
    prev = None
    prev_keys = set()
    rows = []
    for _ in range(rounds):
        states = np.frombuffer(b"".join(sorted(subspace)), np.uint8).reshape(-1, nq)
        matvec, packed, _, oracle = build_round(strings, coeffs, states)

        cold = cold_start(packed, dtype)
        _, v_cold, n_cold, ok = ground_locg(matvec, cold, rtol=rtol, maxiter=6000, prefilter=None)
        assert ok, f"cold arm did not converge at dim={packed.shape[0]}"

        if prev is None:
            print(f"{packed.shape[0]:>7} {'-':>7} {n_cold:>6} {'-':>6} {'-':>8} {'-':>8} {'-':>9}")
        else:
            warm, _ = warm_start(packed, dtype, prev[0], prev[1])
            e_warm, _, n_warm, okw = ground_locg(
                matvec, warm, rtol=rtol, maxiter=6000, prefilter=None
            )
            zp = zero_pad_start(packed, dtype, prev[0], prev[1])
            e_zp, _, n_zp, okz = ground_locg(matvec, zp, rtol=rtol, maxiter=6000, prefilter=None)
            wt = weight_on_new(states, np.asarray(v_cold)[: len(states)], prev_keys)
            valid = n_zp >= n_cold or abs(float(e_zp) - oracle) > 1e-6 or not okz
            de = abs(float(e_warm) - oracle)
            print(
                f"{packed.shape[0]:>7} {wt:>6.1%} {n_cold:>6} {n_warm:>6} {n_zp:>8} "
                f"{n_cold / n_warm:>7.2f}x {de:>9.1e}  {'YES' if valid else 'no'}"
            )
            rows.append((n_cold, n_warm, n_zp, de, valid, okw))

        # Recover from THIS round's eigenvector, on the states it was solved over.
        newstates = recover_configurations(
            states, np.asarray(v_cold)[: len(states)], num_draws, rng, hamming
        )
        prev_keys = {r.tobytes() for r in np.asarray(PauliSumXZ.pack_states(states))}
        prev = (packed, v_cold)
        before = len(subspace)
        subspace |= {row.tobytes() for row in newstates}
        if len(subspace) == before:
            print("  (subspace saturated -- recovery added nothing new, stopping)")
            break

    report(rows)
    return rows


def demo():
    """Self-check: the warm start must carry real weight and stay in the subspace."""
    strings, coeffs = xxz_strings(8, 1.0, 0.3)
    subs = xxz_rungs(8, 3, 500, np.random.default_rng(0))
    assert len(subs) >= 2, "fixture must grow at least once, or there is nothing to warm-start"
    _, p0, _, _ = build_round(strings, coeffs, subs[0])
    _, p1, _, _ = build_round(strings, coeffs, subs[1])
    assert p1.shape[0] > p0.shape[0], "subspace must actually grow between rounds"
    dtype = PauliSumXZ.from_paulisum((strings, coeffs)).c.dtype
    v0 = cold_start(p0, dtype)
    warm, carried = warm_start(p1, dtype, p0, v0)
    assert carried > 0, "join by bytes found nothing -- states are not surviving between rounds"
    # Filler slots must stay exactly zero, or the iterate leaves the subspace.
    filler = (np.asarray(p1)[:, 0] >> 7).astype(bool)
    assert np.all(np.asarray(warm)[filler] == 0.0), "warm start put weight on filler slots"
    assert np.linalg.norm(np.asarray(warm)) > 0, "warm start is identically zero"

    # Recovery must respect the Hamming sector and actually track the eigenvector it is given.
    rng = np.random.default_rng(0)
    st = subs[0]
    peaked = np.zeros(len(st))
    peaked[0] = 1.0  # all weight on one state -> occupancies must match that state's bits
    draws = recover_configurations(st, peaked, 200, rng, hamming=4)
    assert np.all(draws.sum(axis=1) == 4), "recovery left the Hamming-weight sector"
    occ = site_occupancy(st, peaked)
    assert np.allclose(occ, st[0]), f"occupancy ignores the eigenvector: {occ} vs {st[0]}"
    # A peaked eigenvector gives near-deterministic occupancies, so the draws should concentrate on
    # that state -- if they do not, the sampler is not following the distribution.
    assert (draws == st[0]).all(axis=1).mean() > 0.5, "draws do not track a peaked occupancy"
    print(f"demo ok: {carried} carried, {filler.sum()} filler zero, recovery tracks occupancy")


if __name__ == "__main__":
    demo()
    run()
