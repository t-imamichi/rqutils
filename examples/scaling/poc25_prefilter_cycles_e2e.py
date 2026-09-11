"""POC 25: is ``prefilter=(32, 4)`` a better ``sqd`` default than the shipped ``(32, 2)``?

**The question, and why the existing measurements cannot answer it.** ``poc9_prefilter_gpu.py``
sweeps ``(degree, cycles)`` on two synthetic Hamiltonians and finds ``(32, 4)`` the better *worst
case* -- 1.34x / 1.30x against ``(32, 2)``'s 1.07x / 1.25x (``docs/locg-chebyshev-prefilter.md``
§3.2/§3.3). That is a real result and it is the wrong quantity twice over:

1. **It excludes setup.** poc9 drives ``ground_locg`` on a pre-assembled ``apply_h`` matvec. Per
   ``CLAUDE.md`` the ``get_xsource`` precompute is **66-97% of a solve**, so the solver is only a
   4.5-8.4% slice end-to-end and a 1.30x on it is an Amdahl fraction of that slice. This is the exact
   shape that capped the Bloom pre-filter at 1.09x after every mechanic checked out.
2. **It is the wrong fixture family.** poc9 uses random 100-term operators. §3.1's 27-configuration
   sweep -- the measurement the ``(32, 2)`` default actually rests on -- used XXZ chains on connected
   subspaces, and there ``cycles`` **saturates after 2**: ``(16, 4)`` measured 1.41x against
   ``(32, 2)``'s 1.88x. §3.3 showed the two families give near-transposed optimization surfaces, so
   two random-operator fixtures agreeing predicts nothing about the XXZ regime.

So this script measures ``(32, 2)`` against ``(32, 4)`` **through ``sqd``, setup included, on XXZ
subspaces**, sweeping seeds and anisotropies the way §3.1 did.

**§3.1's harness is not in the tree.** ``da7299e`` (the commit that changed the recommendation to
``(32, 2)``) touched only ``docs/locg-chebyshev-prefilter.md`` and ``rqutils/ground_locg.py``; no
sweep script was ever committed, and ``git log -S`` finds none in any branch. Its 27-configuration
table is therefore **not reproducible from this repository** -- only its conclusions survive, in
§3.1. This script does not reconstruct it (different fixture generator, different sizes, and it
compares two settings rather than three) and its numbers are **not** directly comparable to §3.1's
1.88x. What it can do is answer whether ``(32, 4)`` beats ``(32, 2)`` end-to-end on physical
subspaces, which is the open question.

**Stated expectation, recorded up front so a null result reads as a finding.** Amdahl says both arms
should compress toward 1.0x end-to-end: the filter's extra 66 matvecs land on a small slice of total
runtime. UNRESOLVED on most configurations is the *likely* outcome and is an answer -- it would mean
the poc9 surface does not reach the default, and ``(32, 2)`` stays on §3.1's evidence. A clear
``(32, 4)`` win across seeds and anisotropies is what would move the default.

**RESULT (2026-09-12): that expectation was wrong, and its being wrong is the finding.** ``(32, 4)``
measured **0.68-0.71x at every anisotropy, 0 of 81 paired rounds won**, spreads 0.3-1.6% -- ~45%
*slower*, not diluted-toward-1.0x. The extra filter matvecs cost more end-to-end than the iterations
they remove save, so a **1.30x on the solver became 0.69x through ``sqd``**. A ratio measured on a
4.5-8.4% slice can change *sign* when the excluded 66-97% is restored. Table in §3.4; ``(32, 2)``
stays, now on a direct measurement.

**Known fixture limitation.** At the defaults, ``xxz_krylov`` does not reach ``cap``, so ``rng.choice``
never fires and the fixture is **seed-independent** -- ``--seeds 3`` measures one fixture three times
(identical ``N`` and energies to 10 digits). The ``--deltas`` sweep is real. Raise ``--rungs`` or lower
``--cap`` until the subspace exceeds the cap if genuine seed variation is wanted; the 2026-09-12 run
did not need it, but a closer result would.

**Arms are interleaved, not timed in sequence.** Per ``CLAUDE.md``: ``fmt_ratio``'s noise floor is the
*max* of the two spreads, so one outlier in either arm suppresses the verdict however many trials are
added -- a recorded 1.21x win once sat at "UNRESOLVED, noise floor 66.6%" across three re-runs. The
fix is alternating the arms in one loop and reporting a paired win count alongside min/median.

Run::

    uv run --extra qiskit python examples/scaling/poc25_prefilter_cycles_e2e.py
    uv run --extra qiskit python examples/scaling/poc25_prefilter_cycles_e2e.py --rounds 15
    uv run --extra qiskit python examples/scaling/poc25_prefilter_cycles_e2e.py --arms 32,2 32,4 40,2
"""

import argparse
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import scipy.sparse.linalg as spla
from _scaling_common import header

# No E402 suppressions on these, despite the post-jax.config import order CLAUDE.md warns about: that
# rule is not in this project's enabled set, so the directives would themselves be flagged as unused
# (RUF100). Don't add them back. Writing one even inside a comment trips ruff's directive parser.
from rqutils.sqd import hproj, sqd

parser = argparse.ArgumentParser()
parser.add_argument("--num-qubits", type=int, default=20)
parser.add_argument("--rungs", type=int, default=4)
parser.add_argument("--cap", type=int, default=4000)
parser.add_argument("--bx", type=float, default=0.5)
parser.add_argument(
    "--deltas", default="0.5,1.0,1.5", help="XXZ anisotropies to sweep (§3.1 swept these three)."
)
parser.add_argument("--seeds", type=int, default=3)
parser.add_argument(
    "--rounds", type=int, default=9, help="Interleaved A/B rounds per configuration."
)
parser.add_argument(
    "--arms",
    nargs="+",
    default=["32,2", "32,4"],
    help='Prefilter settings as "degree,cycles". First is the baseline.',
)
options = parser.parse_args()

DELTAS = tuple(float(x) for x in options.deltas.split(","))
ARMS = tuple(tuple(int(x) for x in a.split(",")) for a in options.arms)


def xxz_strings(nq, delta, bx):
    """Periodic XXZ (XX+YY+delta*ZZ per bond) plus a transverse field Bx on every site.

    Copied from poc24 rather than imported: that module parses its own argv at import time, so
    importing it here would consume this script's flags. Bx breaks magnetization conservation --
    without it the hop-generated subspace is closed under H and the projection is block-diagonal.
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
    """States reachable from |Neel> in `rungs` nearest-neighbour hops (poc12's fixture, via poc24).

    This is the "connected subspace by one-hop expansion" regime §3.1 measured in, and the reason
    the comparison here is not another random-operator run. `CLAUDE.md`: a physically-motivated
    fixture can invert a conclusion a synthetic one reaches.
    """
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


def reference_energy(strings, coeffs, states):
    """Independent ground energy via sparse eigsh on hproj's projection.

    `CLAUDE.md`: prefer an independent reference over self-consistency, and use `eigsh(k=1)` rather
    than `eigh` (77 s vs 0.02 s at these sizes). `hproj` returns a csr_array, so `.toarray()` is
    required before anything that would otherwise get a 0-d object array.
    """
    projected = hproj((strings, coeffs), states, unique_states=True)
    return float(spla.eigsh(projected, k=1, which="SA", tol=0)[0][0])


def time_one(hamiltonian, states, prefilter):
    """One end-to-end `sqd` call, setup included. Returns (seconds, eigenvalue).

    `return_eigvec=False` so the O(N) eigenvector is not built -- the scalar is what is compared, and
    per `sqd`'s contract this returns a bare float rather than a 3-tuple. Nothing is pre-assembled
    and nothing is cached between calls: the `get_xsource` precompute that poc9 excludes is exactly
    what this measurement has to include, so it is paid inside the timed region every round.
    """
    start = time.perf_counter()
    eigval = sqd(hamiltonian, states, return_eigvec=False, prefilter=prefilter)
    # sqd's return is a host float already (it goes through _host_scalar), so there is no async
    # dispatch left to block on -- the value is materialized before perf_counter is read again.
    return time.perf_counter() - start, float(eigval)


def spread_frac(samples):
    """(max - min) / min, the same spread definition _scaling_common.Timing reports."""
    return (max(samples) - min(samples)) / min(samples)


def main():
    header("POC 25: prefilter cycles, end-to-end through sqd, on XXZ subspaces")
    print(f"n={options.num_qubits} rungs={options.rungs} cap={options.cap} bx={options.bx}")
    print(f"arms={[f'({d}, {c})' for d, c in ARMS]}  baseline={ARMS[0]}  rounds={options.rounds}")
    print(
        "Setup is INSIDE the timed region -- that is the whole point (poc9 excludes it; "
        "get_xsource is 66-97% of a solve).\n"
    )

    baseline_arm = ARMS[0]
    wins = {arm: 0 for arm in ARMS[1:]}
    totals = {arm: [] for arm in ARMS}

    for delta in DELTAS:
        for seed in range(options.seeds):
            rng = np.random.default_rng(seed)
            strings, coeffs = xxz_strings(options.num_qubits, delta, options.bx)
            states = xxz_krylov(options.num_qubits, options.rungs, options.cap, rng)
            reference = reference_energy(strings, coeffs, states)
            ham = (strings, coeffs)

            # Warm every arm before the paired loop: `prefilter` is a static argument, so each
            # setting traces its own executable and a cold first call would land entirely in
            # whichever arm the loop happened to start with.
            for arm in ARMS:
                time_one(ham, states, arm)

            samples = {arm: [] for arm in ARMS}
            energies = {arm: [] for arm in ARMS}
            for _ in range(options.rounds):
                # Interleaved, not arm-at-a-time: one slow outlier otherwise inflates that arm's
                # spread and fmt_ratio's max-of-spreads floor suppresses a real win (CLAUDE.md).
                for arm in ARMS:
                    secs, eigval = time_one(ham, states, arm)
                    samples[arm].append(secs)
                    energies[arm].append(eigval)

            # A broken arm flatters its own benchmark (CLAUDE.md): an arm that converged to the
            # wrong eigenvalue did less work and would report a better time. Gate before quoting.
            bad = [
                (arm, err)
                for arm in ARMS
                if (err := max(abs(e - reference) for e in energies[arm])) > 1e-9
            ]
            label = f"delta={delta} seed={seed} N={states.shape[0]}"
            if bad:
                arm, err = bad[0]
                print(f"{label}: WRONG ENERGY at {arm}, |dE|={err:.2e} -- excluded")
                continue

            base_med = statistics.median(samples[baseline_arm])
            base_min = min(samples[baseline_arm])
            for arm in ARMS:
                totals[arm].extend(samples[arm])
            print(
                f"{label}  E={reference:.10f}  "
                f"{baseline_arm} {base_min * 1e3:.1f}ms/{base_med * 1e3:.1f}ms "
                f"(spread {spread_frac(samples[baseline_arm]) * 100:.1f}%)"
            )
            for arm in ARMS[1:]:
                paired = sum(
                    1 for b, c in zip(samples[baseline_arm], samples[arm], strict=True) if c < b
                )
                wins[arm] += paired
                med = statistics.median(samples[arm])
                print(
                    f"    vs {arm}: {min(samples[arm]) * 1e3:.1f}ms/{med * 1e3:.1f}ms "
                    f"(spread {spread_frac(samples[arm]) * 100:.1f}%)  "
                    f"min {base_min / min(samples[arm]):.2f}x  median {base_med / med:.2f}x  "
                    f"paired {paired}/{options.rounds}"
                )

    if not totals[baseline_arm]:
        print("\nNo configuration produced a correct energy on every arm -- nothing to compare.")
        return

    header("Verdict")
    n_rounds = len(totals[baseline_arm])
    base_med = statistics.median(totals[baseline_arm])
    print(f"baseline {baseline_arm}: median {base_med * 1e3:.1f}ms over {n_rounds} rounds")
    for arm in ARMS[1:]:
        med = statistics.median(totals[arm])
        total_paired = wins[arm]
        possible = options.rounds * len(DELTAS) * options.seeds
        print(
            f"  {arm}: median {med * 1e3:.1f}ms  ratio {base_med / med:.2f}x  "
            f"paired wins {total_paired}/{possible}"
        )
    print(
        "\nRead the paired count with the ratio, not instead of it: a ratio inside the spreads with "
        "a lopsided paired count is a small real effect, and a ratio above 1 with a ~50% paired "
        "count is noise. Neither alone moves a default that rests on 27 paired configurations."
    )


if __name__ == "__main__":
    main()
