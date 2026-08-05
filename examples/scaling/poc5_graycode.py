"""POC 5: Gray-code X-signature ordering -- reuse group j's indices for group j+1.

The premise, verified before building anything: if ``x2 = x1 ^ d`` then the source maps compose,

    S[A1[i]] = S[i] ^ x1  and  S[A2[i]] = S[i] ^ x2 = S[A1[i]] ^ d   =>   A2 = Ad o A1

which was confirmed exactly on valid rows. So ordering the X signatures so consecutive ones differ
by few bits lets each group be a **gather** (O(N)) through the previous group's index array, instead
of an independent **search** (O(N log N)).

**RESULT: this idea is dead, and on correctness grounds rather than speed.** The composition is only
valid when the *intermediate* state ``S[i] ^ x1`` is itself in the sampled subspace. For a projected
Hamiltonian that is usually false: ``S[i] ^ x2`` can be present while ``S[i] ^ x1`` is absent, and the
chain then cannot reach a matrix element that genuinely exists. Measured at n=20, N=100k, J=50: the
reference finds 8720 sources, the chain finds 791 -- it **silently drops 91% of the real matrix
elements**. The projected matrix stays symmetric, so the resulting eigenvalue would be plausible and
wrong, which is precisely the failure class this repository documents throughout.

The premise check that appeared to confirm the composition was measuring a 400-row sample that
happened to be closed under the intermediate. A composition identity that holds on a subset is not
an identity.

Two further things would make it unattractive even if it were correct, and both are measured below:

1. ``Ad`` itself needs a full lookup. The chain only wins if one lookup plus one gather beats one
   lookup -- i.e. never, unless the same ``d`` is reused across many steps. A Gray-code ordering over
   *arbitrary* signatures does not give a repeated ``d``; it gives a different single-bit ``d`` at each
   step, so the ``Ad`` cost recurs. The win therefore hinges on whether a single-bit ``Ad`` lookup is
   cheaper than a general one (it is not -- the search cost is independent of the key's popcount).

2. **``-1`` propagation compounds.** ``A1[i] == -1`` means no source; composing then requires
   ``Ad[A1[i]]`` to also be masked. Each step can only lose validity, never regain it, so a long
   chain drives the valid fraction toward zero and the *result is still correct* -- it just means the
   gather did no useful work. The measurement below reports the valid fraction per chain depth,
   because a "fast" chain that has masked everything out is fast for the wrong reason.

The correctness gate is equivalence-after-gather against the library, per POC 1's rule, and it is the
gate that kills this. The timing sections are kept because they are the second, independent reason
not to revisit the idea: even ignoring correctness, the chain measured UNRESOLVED against J
independent lookups, since ``Ad`` still needs a full lookup per step.

Run: uv run --extra qiskit python examples/scaling/poc5_graycode.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from _scaling_common import fmt_ratio, header, make_problem, timeit
from poc1_searchsorted import FILL_BYTE, xsource_searchsorted_u64

from rqutils.sqd import get_xsource, uniquify_states


def hamming_order(xsigs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Greedy nearest-neighbour reordering of X signatures by Hamming distance.

    A true Gray code enumerates *all* 2^n patterns; here only J arbitrary patterns exist, so the
    analogue is a short Hamiltonian path through them. Greedy nearest-neighbour is not optimal but is
    O(J^2) and J is small (tens to hundreds); an optimal tour would not change the conclusion, since
    what matters below is whether the *mean* step distance is small enough to matter at all.

    Returns the permutation and the per-step Hamming distances along it.
    """
    bits = np.unpackbits(xsigs, axis=1).astype(np.int32)
    n = bits.shape[0]
    unvisited = set(range(1, n))
    order = [0]
    dists = []
    while unvisited:
        last = bits[order[-1]]
        best, bestd = None, None
        for cand in unvisited:
            d = int(np.sum(last != bits[cand]))
            if bestd is None or d < bestd:
                best, bestd = cand, d
        order.append(best)
        dists.append(bestd)
        unvisited.discard(best)
    return np.array(order), np.array(dists)


@jax.jit
def compose_sources(a_prev: jax.Array, a_delta: jax.Array) -> jax.Array:
    """Compose two source maps: ``(Ad o A1)[i] = Ad[A1[i]]``, masking where either is invalid.

    Validity can only be lost, never regained: an ``i`` with no source under ``A1`` has none under the
    composition either. That monotonicity is why chain depth degrades the valid fraction.
    """
    size = a_prev.shape[0]
    safe = jnp.clip(a_prev, 0, size - 1)
    composed = a_delta[safe]
    return jnp.where(jnp.logical_or(a_prev < 0, composed < 0), jnp.int32(-1), composed).astype(
        jnp.int32
    )


def gather_equiv(ref, got, states_u, rng):
    """POC 1's gate: valid-row indices identical AND gathered results identical."""
    is_fill = np.asarray(states_u)[:, 0] == FILL_BYTE
    valid = ~is_fill
    idx_same = bool(np.array_equal(np.asarray(ref)[valid], np.asarray(got)[valid]))
    probe = jnp.asarray(rng.normal(size=states_u.shape[0]))

    def g(idx):
        return np.asarray(
            probe.at[jnp.asarray(idx)].get(mode="fill", fill_value=0.0, wrap_negative_indices=False)
        )

    return idx_same, bool(np.array_equal(g(ref), g(got)))


def main():
    header("POC 5a: how close together ARE the X signatures? (premise check)")
    print("The chain only helps if consecutive step distances are small. If the mean step distance")
    print("is ~n/2, a reordering buys nothing because every step is as far as a random pair.")
    print(f"{'n':>4s}  {'J':>5s}  {'mean step d':>12s}  {'min':>5s}  {'max':>5s}  {'n/2':>6s}")
    for num_qubits, j in [(20, 50), (20, 200), (24, 50), (40, 50)]:
        p = make_problem(num_qubits, 2_000, num_terms=max(j * 2, 100), num_xgroups=j, seed=52)
        _, dists = hamming_order(np.asarray(p.hamiltonian.x))
        print(
            f"{num_qubits:>4d}  {p.num_xgroups:>5d}  {dists.mean():>12.2f}  "
            f"{dists.min():>5d}  {dists.max():>5d}  {num_qubits / 2:>6.1f}"
        )
    print()
    print("  A mean step distance near n/2 means the signatures are essentially random relative to")
    print("  each other and no ordering makes them neighbours.")

    header("POC 5b: chain correctness and valid-fraction decay with depth")
    p = make_problem(20, 100_000, num_terms=100, num_xgroups=50, seed=53)
    size = p.states_p.shape[0]
    states_u = jax.block_until_ready(uniquify_states(p.states_p, size))
    xsigs = np.asarray(p.hamiltonian.x)
    order, dists = hamming_order(xsigs)
    rng = np.random.default_rng(5)

    print(f"  {p.describe()}")
    print(f"{'depth':>6s}  {'valid frac':>11s}  {'idx match':>10s}  {'gather match':>13s}")
    a_curr = xsource_searchsorted_u64(xsigs[order[0]], states_u)
    all_ok = True
    for depth in range(1, min(9, len(order))):
        i_prev, i_curr = order[depth - 1], order[depth]
        delta = np.bitwise_xor(xsigs[i_prev], xsigs[i_curr])
        a_delta = xsource_searchsorted_u64(delta, states_u)
        a_curr = compose_sources(a_curr, a_delta)
        ref = get_xsource(xsigs[i_curr], states_u)
        i_same, g_same = gather_equiv(ref, a_curr, states_u, rng)
        all_ok = all_ok and i_same and g_same
        vf = float(np.mean(np.asarray(a_curr) >= 0))
        print(
            f"{depth:>6d}  {vf:>11.4f}  {'OK' if i_same else 'MISMATCH':>10s}  "
            f"{'OK' if g_same else 'MISMATCH':>13s}"
        )
    print(f"\n  chain correctness: {'OK' if all_ok else 'FAILURES'}")

    # Pin down WHY, so the negative result is self-documenting rather than just a FAIL column.
    i0, i1 = order[0], order[1]
    delta = np.bitwise_xor(xsigs[i0], xsigs[i1])
    a0 = np.asarray(xsource_searchsorted_u64(xsigs[i0], states_u))
    ad = np.asarray(xsource_searchsorted_u64(delta, states_u))
    comp = np.asarray(compose_sources(jnp.asarray(a0), jnp.asarray(ad)))
    ref1 = np.asarray(get_xsource(xsigs[i1], states_u))
    only_ref = np.logical_and(ref1 >= 0, comp < 0)
    print()
    print("  ROOT CAUSE (depth 1, the shallowest possible chain):")
    print(f"    reference finds        {int((ref1 >= 0).sum()):>7d} sources")
    print(f"    chain finds            {int((comp >= 0).sum()):>7d} sources")
    print(
        f"    elements chain MISSES  {int(only_ref.sum()):>7d} "
        f"({only_ref.sum() / max((ref1 >= 0).sum(), 1) * 100:.1f}% of real elements)"
    )
    print("    Composition requires the INTERMEDIATE state S[i]^x0 to be in the sampled subspace.")
    print("    S[i]^x1 can be present while S[i]^x0 is absent, and the chain cannot reach it.")
    print("    The projected matrix stays symmetric, so the eigenvalue would be plausible and")
    print("    WRONG -- the exact silent-failure class this repository documents throughout.")

    header("POC 5c: J-fold cost -- Gray chain vs J independent searchsorted lookups")
    print("The chain pays 1 searchsorted (for delta) + 1 gather per group; the baseline pays")
    print("1 searchsorted per group. So the chain can only win if a gather is cheaper than the")
    print(
        "searchsorted it does NOT avoid -- which it is not, since delta still needs a full lookup."
    )
    print()
    print(f"{'N':>9s}  {'J':>4s}  {'J x ssorted':>13s}  {'gray chain':>12s}  {'verdict':>34s}")
    for num_states in [100_000, 200_000]:
        p = make_problem(20, num_states, num_terms=100, num_xgroups=50, seed=54)
        size = p.states_p.shape[0]
        states_u = jax.block_until_ready(uniquify_states(p.states_p, size))
        xs = p.hamiltonian.x
        xsn = np.asarray(xs)
        order, _ = hamming_order(xsn)
        deltas = jnp.asarray(
            np.stack(
                [np.bitwise_xor(xsn[order[k - 1]], xsn[order[k]]) for k in range(1, len(order))]
            )
        )
        first = jnp.asarray(xsn[order[0]])

        def baseline(states_u=states_u, xs=xs):
            return jax.lax.scan(
                lambda _, x: (None, xsource_searchsorted_u64(x, states_u)), None, xs
            )[1]

        def gray(first=first, states_u=states_u, deltas=deltas):
            a0 = xsource_searchsorted_u64(first, states_u)

            def step(a_prev, d):
                a_d = xsource_searchsorted_u64(d, states_u)
                a_new = compose_sources(a_prev, a_d)
                return a_new, a_new

            return jax.lax.scan(step, a0, deltas)[1]

        t_base = timeit(baseline, "J x ssorted", trials=3)
        t_gray = timeit(gray, "gray chain", trials=3)
        print(
            f"{size:>9d}  {p.num_xgroups:>4d}  {t_base.min_s * 1e3:>11.2f}ms  "
            f"{t_gray.min_s * 1e3:>10.2f}ms  {fmt_ratio(t_base, t_gray):>34s}"
        )

    header("POC 5d: the one case where a chain WOULD win -- repeated delta")
    print("If many groups share the same delta (a structured Hamiltonian, e.g. a translation-")
    print("invariant chain where signatures are shifts of one pattern), Ad is computed ONCE and")
    print("amortized over the chain. Measured with an artificially structured signature set:")
    p = make_problem(20, 200_000, num_terms=100, num_xgroups=50, seed=55)
    size = p.states_p.shape[0]
    states_u = jax.block_until_ready(uniquify_states(p.states_p, size))
    # Structured set: x_k = x_0 ^ (k copies of a fixed single-bit delta) is not expressible as a
    # bitmask chain with a CONSTANT delta unless we xor the same bit repeatedly (which cancels).
    # The realistic structured case is a fixed delta applied cumulatively to DIFFERENT bits, so the
    # honest statement is: a constant delta chain visits only 2 distinct signatures. Report that
    # rather than fabricate a favourable case.
    print()
    print("  A single constant delta d generates only {x, x^d} -- two signatures -- because")
    print("  xoring d twice returns to x. So 'repeated delta' cannot generate J distinct")
    print("  signatures, and the amortization that would make a chain win is unavailable in")
    print("  general. This is a NEGATIVE result: the idea does not have a regime on this")
    print("  problem class.")
    _ = size, states_u, p


if __name__ == "__main__":
    main()
