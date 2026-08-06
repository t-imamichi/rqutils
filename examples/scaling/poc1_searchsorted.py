"""POC 1: replace ``get_xsource``'s 2N sort with a searchsorted into the already-sorted S.

**This POC's proposal was adopted in commit 23fb226, so ``rqutils.sqd.get_xsource`` IS the
searchsorted now.** The sort baseline therefore lives here, as ``xsource_sort_legacy`` -- a verbatim
copy of the pre-23fb226 implementation -- and every *timing* arm compares against that, not against
the library. Timing against the library would compare two binary searches and report 1.00x, which is
exactly what happened on the first GPU run of ``poc8_gpu_unverified.py``. The *correctness* arms still
reference the library on purpose: agreeing with what ``sqd`` actually ships is the property that
matters, and it is now a regression test rather than a proposal.

``get_xsource`` finds, for each ``i``, the index ``A[i]`` with ``S[A[i]] == S[i] ^ x``, or -1 when no
such state is in the subspace. The former implementation did this by concatenating ``S`` and
``S ^ x`` into a ``2N`` array, sorting it, and reading off adjacent equal pairs. That sort is why
``N <= 2**31``, is noted in-tree as leaking up to 5 GB of GPU memory, and -- per ``baseline.py`` --
was 66-97% of an entire solve.

But ``S`` is **already lex-sorted**: ``uniquify_states`` returns it that way. So the lookup is a
binary search of ``S ^ x`` into ``S``, which needs no ``2N`` allocation and no sort at all. A
``searchsorted`` is also a pure gather, so unlike a sort it shards -- each device can search its
local slice against a replicated ``S``.

Two implementations are compared against the legacy sort (timing) and the library (correctness):

``searchsorted_u64`` packs each state row into a single uint64 key and searches. This is the fast
path and it is **exact only while n+1 <= 64 bits**, i.e. up to 63 qubits. That covers every problem
this library can actually run (``N <= 2**31`` subspace states means far fewer than 63 qubits of
useful subspace), but the limit is a hard correctness boundary, not a performance note, so it is
asserted rather than documented.

``searchsorted_lex`` handles arbitrary width by searching on a descending-significance tuple of
bytes. It is the general fallback; it is slower and included to show the width limit is removable.

Correctness gate, in two parts, because "bit-identical" turns out to be the wrong test and the
reason is worth recording.

On a **fill-in row** (all-255, produced by uniquification) the target ``255 ^ x`` is never in ``S``,
so the answer is "no source". The legacy sort reaches that verdict through
``idx_sorted[1:] - size`` and lands on assorted negative values (-12, -11, -10, ...) rather than
exactly -1; a searchsorted returns -1. Both are consumed identically, because ``apply_xgrp`` gathers
with ``mode="fill", wrap_negative_indices=False``, so *any* negative index yields 0.0. Demanding
equality here would reject a correct implementation over an unobservable difference.

So the gate is:

1. On **valid rows**, the index must be bit-identical. A permuted-but-plausible index array is
   exactly the silent failure mode ``symplectic.py`` warns about for signature decoding, so nothing
   less than equality will do.
2. On **all rows**, the *gathered result* ``v[A]`` must be bit-identical -- which is the only
   property any consumer can observe, and it covers the fill rows without over-constraining them.

Run: uv run --extra qiskit python examples/scaling/poc1_searchsorted.py
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from _scaling_common import fmt_ratio, header, make_problem, timeit

from rqutils.sqd import get_xsource, uniquify_states

# Fill-in rows from uniquification are all-255. They must never match a real source: a genuine
# packed state always has byte 0 < 128 because of the pad bit, so 255 is unreachable by construction.
FILL_BYTE = 255


@jax.jit
def xsource_sort_legacy(xsignature, states):
    """The pre-23fb226 ``get_xsource``: concatenate ``S`` and ``S ^ X`` into ``2N`` rows and sort.

    **This is the baseline arm, and it lives here because the library no longer provides one.**
    Commit 23fb226 replaced ``rqutils.sqd.get_xsource`` with the searchsorted below, so timing
    ``get_xsource`` against ``xsource_searchsorted_u64`` now compares two binary searches and
    measures 1.00x. That is exactly what happened on the first GPU run of ``poc8``: both arms were
    the same algorithm, ``fmt_ratio`` correctly refused to call a winner, and the run produced no
    information about the 12-25x claim it existed to check. A POC whose baseline is whatever the
    library currently does silently stops being a comparison the moment the library changes.

    Copied verbatim from ``git show 23fb226^:rqutils/sqd.py`` rather than reconstructed, including
    the ``lax.sort`` leak comment -- a hand-rewritten sort would measure the rewrite. The
    ``jax.reshard`` tail is dropped because these POCs time a single device; the sort is the subject.

    Returns assorted negative values (not exactly -1) on fill-in rows, since it computes
    ``I[k+1] - N`` unconditionally. Consumers cannot tell, because ``apply_xgrp`` gathers with
    ``wrap_negative_indices=False`` and any negative index yields 0.0 -- which is why the
    correctness gate here compares valid-row indices and gathered results, not raw index arrays.
    """
    size = states.shape[0]
    mapped_states = jnp.bitwise_xor(states, xsignature)  # S^X
    joined = jnp.concatenate([states, mapped_states], axis=0)
    idx = jax.lax.iota(np.int32, 2 * size)
    # lax.sort seems to leak GPU memory; can lose as much as 5 GB when sorting x of shape (5M,9)
    srt = jax.lax.sort(tuple(joined.T) + (idx,), num_keys=joined.shape[1])
    joined_sorted = jnp.stack(srt[:-1], axis=1)  # T
    idx_sorted = srt[-1]  # I
    invalid = np.array(-1, dtype=np.int32)

    source_idx = jnp.where(
        jnp.all(jnp.equal(joined_sorted[:-1], joined_sorted[1:]), axis=1),  # T[k] == T[k+1]
        idx_sorted[1:] - size,  # I[k+1] - N
        invalid,
    )
    tposition = jnp.cumsum(
        jnp.bincount(jnp.cumsum(idx_sorted < size, dtype=np.int32), length=size), dtype=np.int32
    )
    return source_idx.at[tposition].get(mode="fill", fill_value=invalid)


def _pack_u64(states: jax.Array) -> jax.Array:
    """Pack a (N, B) uint8 state array into (N,) uint64 keys, preserving lex order.

    Byte 0 is the most significant, matching the lexsort ``uniquify_states`` performs, so the
    resulting integer order is identical to the row lex order. That equivalence is what lets a
    scalar binary search stand in for a lexicographic one.
    """
    nbytes = states.shape[1]
    if nbytes > 8:
        raise ValueError(f"u64 packing needs B <= 8, got {nbytes} (n+1 > 64 bits)")
    shifts = jnp.asarray([8 * (nbytes - 1 - i) for i in range(nbytes)], dtype=jnp.uint64)
    return jnp.sum(states.astype(jnp.uint64) << shifts, axis=1)


@jax.jit
def xsource_searchsorted_u64(xsignature, states):
    """Source indices via a uint64 binary search. Requires ceil((n+1)/8) <= 8 bytes."""
    keys = _pack_u64(states)  # sorted by construction
    targets = _pack_u64(jnp.bitwise_xor(states, xsignature))
    pos = jnp.searchsorted(keys, targets, side="left")
    # Clamp before gathering: searchsorted returns N for a target above every key.
    pos_c = jnp.minimum(pos, keys.shape[0] - 1)
    found = keys[pos_c] == targets
    return jnp.where(found, pos_c, jnp.int32(-1)).astype(jnp.int32)


@jax.jit
def xsource_searchsorted_lex(xsignature, states):
    """Source indices via a byte-wise lexicographic binary search, any width.

    Implements the search manually because ``jnp.searchsorted`` takes scalar keys only. The
    invariant is the standard one: ``lo`` is the count of rows strictly less than the target.
    Comparison walks bytes most-significant first, which is the order ``uniquify_states`` sorted in.
    """
    size, _nbytes = states.shape
    targets = jnp.bitwise_xor(states, xsignature)

    def row_lt(a, b):
        """Lexicographic a < b over uint8 rows, evaluated MSB-first without a loop carry."""
        # First differing byte decides. Build a mask of "all previous bytes equal".
        eq_prefix = jnp.cumprod(
            jnp.concatenate([jnp.ones((a.shape[0], 1), bool), a[:, :-1] == b[:, :-1]], axis=1),
            axis=1,
        ).astype(bool)
        lt_here = jnp.logical_and(eq_prefix, a < b)
        return jnp.any(lt_here, axis=1)

    def body(carry, _):
        lo, hi = carry
        mid = (lo + hi) // 2
        mid_rows = states[jnp.minimum(mid, size - 1)]
        go_right = row_lt(mid_rows, targets)
        lo = jnp.where(go_right, mid + 1, lo)
        hi = jnp.where(go_right, hi, mid)
        return (lo, hi), None

    nsteps = int(np.ceil(np.log2(max(size, 2)))) + 1
    lo = jnp.zeros(size, dtype=jnp.int32)
    hi = jnp.full(size, size, dtype=jnp.int32)
    (lo, _), _ = jax.lax.scan(body, (lo, hi), None, length=nsteps)

    pos_c = jnp.minimum(lo, size - 1)
    found = jnp.all(states[pos_c] == targets, axis=1)
    return jnp.where(found, pos_c, jnp.int32(-1)).astype(jnp.int32)


def _equivalent(ref, got, states_u, rng):
    """Compare two index arrays under the two-part gate described in the module docstring.

    Returns (valid_rows_identical, gathered_identical, n_index_mismatches_on_valid_rows).
    """
    is_fill = np.asarray(states_u)[:, 0] == FILL_BYTE
    valid = ~is_fill
    idx_same = bool(np.array_equal(ref[valid], got[valid]))
    nbad = int(np.sum(ref[valid] != got[valid]))

    # The observable property: gather a probe vector through both index arrays exactly as
    # apply_xgrp does, and require the results to agree everywhere including the fill rows.
    probe = jnp.asarray(rng.normal(size=states_u.shape[0]))

    def gather(idx):
        return probe.at[jnp.asarray(idx)].get(
            mode="fill", fill_value=0.0, wrap_negative_indices=False
        )

    gathered_same = bool(np.array_equal(np.asarray(gather(ref)), np.asarray(gather(got))))
    return idx_same, gathered_same, nbad


def check_correctness():
    header("POC 1a: CORRECTNESS -- valid-row indices identical + gathered result identical")
    ok = True
    rng = np.random.default_rng(99)
    # Sweep qubit counts across the byte-width boundary (n+1 crossing 8, 16, 24 bits) and include a
    # near-complete subspace, where almost every source exists, and a sparse one, where almost none
    # do -- the -1 path is the half a naive implementation gets wrong.
    cases = [
        (6, 40, "tiny, near-complete subspace"),
        (7, 100, "n+1 = 8 bits exactly"),
        (8, 200, "n+1 = 9 bits, crosses to 2 bytes"),
        (15, 2000, "n+1 = 16 bits exactly"),
        (16, 3000, "n+1 = 17 bits, crosses to 3 bytes"),
        (20, 5000, "sparse subspace, most sources absent"),
        (24, 8000, "3 bytes"),
    ]
    for num_qubits, num_states, note in cases:
        p = make_problem(num_qubits, num_states, num_terms=12, seed=11)
        size = p.states_p.shape[0]
        states_u = uniquify_states(p.states_p, size)
        nfill = int(np.sum(np.asarray(states_u)[:, 0] == FILL_BYTE))

        case_ok = True
        for isig in range(min(4, p.num_xgroups)):
            xsig = p.hamiltonian.x[isig]
            ref = np.asarray(get_xsource(xsig, states_u))
            for name, got in [
                ("u64", np.asarray(xsource_searchsorted_u64(xsig, states_u))),
                ("lex", np.asarray(xsource_searchsorted_lex(xsig, states_u))),
            ]:
                idx_same, gathered_same, nbad = _equivalent(ref, got, states_u, rng)
                if not (idx_same and gathered_same):
                    case_ok = False
                    print(
                        f"  FAIL n={num_qubits} sig={isig} [{name}]: "
                        f"valid-row index mismatches={nbad} gathered_same={gathered_same}"
                    )
        ok = ok and case_ok
        nhit = int(np.sum(ref >= 0))
        print(
            f"  n={num_qubits:<3d} N={size:<6d} fill={nfill:<5d} hits={nhit:<6d} "
            f"({nhit / size * 100:5.1f}% sources present)  {'OK' if case_ok else 'FAIL'}  [{note}]"
        )

    print(
        f"\n  CORRECTNESS: {'valid-row indices identical, gathers identical' if ok else 'FAILURES PRESENT'}"
    )
    return ok


def check_scaling(sizes=None, num_qubits=24):
    """Per-signature cost vs N, plus the fitted exponent.

    ``alpha`` (from a log-log fit) is the load-bearing output, not the ratios: it says whether the
    measurement is in the regime where the ratio means anything. On a GH200 the default sizes give
    alpha = 0.29 (sort) and 0.10 (searchsorted) -- both far below the ~1 of a saturated device --
    because J=1 kernels at N <= 1M are **launch-latency bound**, so per-call overhead swamps the
    work and every ratio is a ratio of two overheads. Pass ``--sweep-to`` to push N until alpha
    approaches 1; the ratio at the largest size where alpha is near 1 is the quotable GPU number.
    """
    header(f"POC 1b: SCALING -- per-signature cost vs N (n={num_qubits}, single X signature)")
    print(f"{'N':>9s}  {'legacy sort':>13s}  {'searchsorted':>13s}  {'verdict':>34s}")
    rows = []
    for num_states in sizes or [10_000, 50_000, 200_000, 500_000, 1_000_000]:
        p = make_problem(num_qubits, num_states, num_terms=8, seed=12)
        size = p.states_p.shape[0]
        states_u = jax.block_until_ready(uniquify_states(p.states_p, size))
        xsig = p.hamiltonian.x[0]

        t_ref = timeit(
            lambda xsig=xsig, states_u=states_u: xsource_sort_legacy(xsig, states_u),
            "legacy sort",
            trials=5,
        )
        t_new = timeit(
            lambda xsig=xsig, states_u=states_u: xsource_searchsorted_u64(xsig, states_u),
            "u64",
            trials=5,
        )
        rows.append((num_states, t_ref, t_new))
        print(
            f"{size:>9d}  {t_ref.min_s * 1e3:>11.2f}ms  {t_new.min_s * 1e3:>11.2f}ms  "
            f"{fmt_ratio(t_ref, t_new):>34s}"
        )

    print()
    ns = np.array([r[0] for r in rows], dtype=float)
    alphas = {}
    for name, idx in [("legacy sort", 1), ("searchsorted", 2)]:
        ts = np.array([r[idx].min_s for r in rows])
        alphas[name] = np.polyfit(np.log(ns), np.log(ts), 1)[0]
        print(f"  {name:<14s} alpha = {alphas[name]:.2f}  (cost ~ N^alpha)")

    # alpha is the gate on whether the ratios above are quotable. A sort is >= N log N, so an alpha
    # well under 1 means fixed per-call cost dominates and the ratio is overhead/overhead. This is
    # the check that explains poc8's flat sort arm on a GH200 (1256/1313/1599 ms for 25x N) -- the
    # numbers were reproducible and meaningless at the same time.
    a_sort = alphas["legacy sort"]
    print()
    if a_sort < 0.6:
        print(
            f"  LAUNCH-BOUND: alpha = {a_sort:.2f} for a sort, which is at least linear in theory."
        )
        print(
            "  Fixed per-call cost dominates at these sizes, so the ratios above are ratios of two"
        )
        print(
            "  overheads and must NOT be quoted as the GPU speedup. Re-run with --sweep-to to push"
        )
        print("  N until alpha approaches 1, and read the ratio there.")
        largest = rows[-1]
        print(
            f"  Best available lower bound: {largest[1].min_s / largest[2].min_s:.2f}x at N={largest[0]}, "
            "rising with N."
        )
    else:
        print(
            f"  COMPUTE-BOUND ENOUGH: alpha = {a_sort:.2f}. The largest-N ratio above is quotable."
        )

    header("POC 1c: full J-fold precompute -- the cost baseline.py found dominant")
    print(f"{'N':>9s}  {'J':>4s}  {'legacy sort':>13s}  {'searchsorted':>13s}  {'verdict':>34s}")
    for num_states, j in [(100_000, 50), (200_000, 50), (500_000, 50)]:
        p = make_problem(24, num_states, num_terms=100, num_xgroups=j, seed=13)
        size = p.states_p.shape[0]
        states_u = jax.block_until_ready(uniquify_states(p.states_p, size))
        xs = p.hamiltonian.x

        def all_ref(states_u=states_u, xs=xs):
            return jax.lax.scan(lambda _, x: (None, xsource_sort_legacy(x, states_u)), None, xs)[1]

        def all_new(states_u=states_u, xs=xs):
            return jax.lax.scan(
                lambda _, x: (None, xsource_searchsorted_u64(x, states_u)), None, xs
            )[1]

        # Gate correctness on the full J-fold stack too, not just the single-signature test above,
        # under the same valid-row + gathered-result rule.
        rng = np.random.default_rng(7)
        ref_all, new_all = np.asarray(all_ref()), np.asarray(all_new())
        for isig in range(ref_all.shape[0]):
            i_same, g_same, nbad = _equivalent(ref_all[isig], new_all[isig], states_u, rng)
            assert i_same and g_same, f"J-fold mismatch at sig {isig}: nbad={nbad}"

        t_ref = timeit(all_ref, "legacy sort", trials=3)
        t_new = timeit(all_new, "new", trials=3)
        print(
            f"{size:>9d}  {p.num_xgroups:>4d}  {t_ref.min_s * 1e3:>11.2f}ms  "
            f"{t_new.min_s * 1e3:>11.2f}ms  {fmt_ratio(t_ref, t_new):>34s}"
        )

    header("POC 1d: peak transient allocation per X signature (analytic)")
    print("Analytic, since the CPU backend does not report a GPU-style peak. Counting only the")
    print("transients one call allocates on top of S, which both paths share.")
    print()
    print("  legacy sort path, per call:")
    print("    S^X            N x B  uint8")
    print("    joined         2N x B uint8   (concatenate)")
    print("    iota           2N     int32")
    print("    sorted tuple   2N x B uint8 + 2N int32   (lax.sort returns new arrays)")
    print("    joined_sorted  2N x B uint8   (stack)")
    print("  searchsorted path, per call:")
    print("    keys           N      uint64")
    print("    targets        N x B  uint8 (xor) then N uint64 (pack)")
    print()
    print(f"{'N':>12s}  {'B':>3s}  {'sort peak':>12s}  {'ssorted':>12s}  {'ratio':>7s}")
    for num_states in [1_000_000, 100_000_000, 2**31]:
        for nbytes in [4, 11]:
            n = num_states
            sort_peak = (
                n * nbytes  # S^X
                + 2 * n * nbytes  # joined
                + 2 * n * 4  # iota
                + 2 * n * nbytes
                + 2 * n * 4  # sort outputs
                + 2 * n * nbytes  # joined_sorted
            )
            ss_peak = n * 8 + n * nbytes + n * 8  # keys + xor + packed targets
            print(
                f"{num_states:>12d}  {nbytes:>3d}  {sort_peak / 2**30:>10.2f}GB  "
                f"{ss_peak / 2**30:>10.2f}GB  {sort_peak / ss_peak:>6.2f}x"
            )
    print()
    print("  NOTE: this is transient allocation arithmetic, NOT the 5 GB GPU leak once recorded in")
    print("  rqutils/sqd.py. That leak is an allocator behaviour of lax.sort on a GPU backend, so")
    print("  this analytic table cannot speak to it either way -- poc8_gpu_unverified.py measures")
    if jax.default_backend() == "cpu":
        print("  it, and on a CPU-only machine it is unverifiable.")
    else:
        print(
            "  it. Measured flat on a GH200 (2026-08-06): the sort allocated ~0.95GB of transients"
        )
        print(
            "  at N=5M/B=4 and returned to baseline after every repetition, so on that backend and"
        )
        print("  JAX version the leak does NOT reproduce and the original note is stale.")


def check_lex_variant():
    header("POC 1e: arbitrary-width lex variant (removes the 63-qubit limit)")
    print(f"{'n':>4s}  {'N':>9s}  {'legacy sort':>13s}  {'lex ssorted':>13s}  {'verdict':>34s}")
    for num_qubits, num_states in [(24, 200_000), (80, 200_000)]:
        p = make_problem(num_qubits, num_states, num_terms=8, seed=14)
        size = p.states_p.shape[0]
        states_u = jax.block_until_ready(uniquify_states(p.states_p, size))
        xsig = p.hamiltonian.x[0]
        ref = np.asarray(get_xsource(xsig, states_u))
        got = np.asarray(xsource_searchsorted_lex(xsig, states_u))
        i_same, g_same, nbad = _equivalent(ref, got, states_u, np.random.default_rng(8))
        assert i_same and g_same, f"lex mismatch at n={num_qubits}: nbad={nbad}"

        t_ref = timeit(
            lambda xsig=xsig, states_u=states_u: xsource_sort_legacy(xsig, states_u),
            "legacy sort",
            trials=3,
        )
        t_new = timeit(
            lambda xsig=xsig, states_u=states_u: xsource_searchsorted_lex(xsig, states_u),
            "lex",
            trials=3,
        )
        nb = p.states_p.shape[1]
        print(
            f"{num_qubits:>4d}  {size:>9d}  {t_ref.min_s * 1e3:>11.2f}ms  "
            f"{t_new.min_s * 1e3:>11.2f}ms  {fmt_ratio(t_ref, t_new):>34s}   (B={nb} bytes)"
        )


def _sweep_sizes(limit, num_qubits):
    """Geometric ladder up to ``limit``, for pushing N out of the launch-bound regime.

    Starts at 1M (where the default ladder ends) and quadruples. ``n`` is raised alongside so the
    subspace can actually hold the requested count: N distinct states need n > log2(N), and a
    too-small n silently caps the post-uniquification size, which would flatten the fit and read as
    "still launch-bound" no matter how large ``limit`` is.
    """
    sizes, n = [], 1_000_000
    while n <= limit:
        sizes.append(n)
        n *= 4
    need = int(np.ceil(np.log2(max(sizes) * 4))) if sizes else num_qubits
    return sizes, max(num_qubits, need)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--sweep-to",
        type=int,
        metavar="N",
        help="Extend POC 1b's N ladder up to N (e.g. 64000000) to leave the launch-bound regime. "
        "Raises n as needed so the subspace can hold that many distinct states.",
    )
    ap.add_argument("--skip-correctness", action="store_true", help="Skip POC 1a (timings only).")
    opts = ap.parse_args()

    if not opts.skip_correctness and not check_correctness():
        print("\nABORTING: correctness gate failed, timings would be meaningless.")
        sys.exit(1)
    if opts.sweep_to:
        sizes, nq = _sweep_sizes(opts.sweep_to, 24)
        print(f"\n(--sweep-to {opts.sweep_to}: N ladder {sizes}, n={nq})")
        check_scaling(sizes=sizes, num_qubits=nq)
    else:
        check_scaling()
    check_lex_variant()
