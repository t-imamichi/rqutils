"""Does the post-solve residual check stay inside its threshold as the subspace grows?

Answers ask 2 of spinchain's ``rqutils-eigenpair-check-request.md``: the 10x slack in
``EigenpairCheckError`` was calibrated on test-suite sizes (N <= 30k), while spinchain solves N = 1.5M-20M,
complex Hermitian, at the default tolerance. The threshold's floor term ``4*eps*sum|c_k|`` does not grow
with N, so the question is whether the recomputed residual does.

The Hamiltonians are the request's own: its ``xxz`` builder verbatim (open chain, ``J/4`` couplings, fields
``-f_i/2``) over its four field patterns at ``delta`` 0.5 and 2.0. Subspaces are a Hamming-shell draw
around the two Neel states -- whole shells of increasing distance, then a random subset of the last to hit
N -- the shape the request names for synthetic ones (spinchain's own ``draw`` is not reproduced exactly).
``sqd`` runs as spinchain calls it: default ``atol``/``rtol`` and prefilter, at both of its cache levels,
reading the check's own INFO line back.

Run: uv run python poc/eigenpair_check_scale.py [--num-qubits 30] [--sizes 100000 1000000]
     uv run python poc/eigenpair_check_scale.py --dense-check 2000   # eigval/residual vs dense, per case
"""

import argparse
import itertools
import logging
import re
import time

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
from qiskit.quantum_info import SparsePauliOp

from rqutils.sqd import hproj, sqd

DELTAS = (0.5, 2.0)


def xxz(n, delta, bx, by, bz, J=1.0):
    """The request's builder, verbatim: qubit ``i`` is site ``i``, Qiskit's little-endian order."""
    terms = [
        (p, [i, i + 1], c)
        for i in range(n - 1)
        for p, c in (("XX", J / 4), ("YY", J / 4), ("ZZ", delta / 4))
    ]
    terms += [(p, [i], -f[i] / 2) for i in range(n) for p, f in (("X", bx), ("Y", by), ("Z", bz))]
    return SparsePauliOp.from_sparse_list(terms, n).simplify(atol=0, rtol=0)


def ends(a, b, n):
    """The request's ``"a..b"``: ``a`` on site 0, ``b`` on site ``n-1``, zero between."""
    return [a] + [0.0] * (n - 2) + [b]


def patterns(n):
    """The request's four ``(Bx, By, Bz)`` patterns; a bare number is uniform."""
    zero, one = [0.0] * n, [1.0] * n
    return {
        "type1": (ends(0, 1, n), ends(1, 0, n), zero),
        "type2": (one, ends(1, 1, n), zero),
        "type3": (one, ends(1, -1, n), zero),
        "type4": (ends(0, 1, n), ends(1, 0, n), ends(1, 1, n)),
    }


def hamming_shells(n, size, rng):
    """``size`` unique states: whole Hamming shells around both Neel states, then part of the last."""
    neel = [sum(1 << q for q in range(start, n, 2)) for start in (0, 1)]
    chosen = np.array([], dtype=np.uint64)
    for d in range(n + 1):
        if d == 0:
            masks = np.zeros(1, np.uint64)  # the two Neel states themselves
        else:
            combos = np.array(list(itertools.combinations(range(n), d)), dtype=np.uint64)
            masks = np.bitwise_or.reduce(np.uint64(1) << combos, axis=1)
        shell = np.setdiff1d(
            np.unique(np.concatenate([np.uint64(r) ^ masks for r in neel])), chosen
        )
        if len(chosen) + len(shell) >= size:
            shell = rng.choice(shell, size - len(chosen), replace=False)
            chosen = np.concatenate([chosen, shell])
            break
        chosen = np.concatenate([chosen, shell])
    return ((chosen[:, None] >> np.arange(n, dtype=np.uint64)) & np.uint64(1)).astype(np.uint8)


class CheckLine(logging.Handler):
    """Captures the residual and threshold from ``sqd``'s "Independent eigen-residual" INFO line."""

    def __init__(self):
        super().__init__(logging.INFO)
        self.last = None

    def emit(self, record):
        m = re.search(r"eigen-residual ([\d.e+-]+) \(threshold ([\d.e+-]+)\)", record.getMessage())
        if m:
            self.last = (float(m.group(1)), float(m.group(2)))

    def take(self):
        assert self.last is not None, "sqd logged no 'Independent eigen-residual' line"
        last, self.last = self.last, None
        return last


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-qubits", type=int, default=30)
    parser.add_argument("--sizes", type=int, nargs="+", default=[100_000, 1_000_000])
    parser.add_argument("--patterns", nargs="+", default=["type1", "type2", "type3", "type4"])
    parser.add_argument("--levels", nargs="+", default=["1,0", "1,2"])
    parser.add_argument("--deltas", type=float, nargs="+", default=list(DELTAS))
    parser.add_argument(
        "--dense-check", type=int, default=0, help="N for a dense cross-check instead"
    )
    args = parser.parse_args()
    n = args.num_qubits
    levels = [tuple(int(v) for v in s.split(",")) for s in args.levels]
    handler = CheckLine()
    log = logging.getLogger("rqutils.sqd")
    log.addHandler(handler)
    log.setLevel(logging.INFO)

    if args.dense_check:
        states = hamming_shells(n, args.dense_check, np.random.default_rng(0))
        for name in args.patterns:
            for delta in args.deltas:
                ham = xxz(n, delta, *patterns(n)[name])
                eigval, eigvec, basis = sqd(ham, states, return_eigvec=True)
                logged, _ = handler.take()
                dense = hproj(ham, basis).toarray()
                ref = float(np.linalg.eigvalsh(dense)[0])
                res = float(np.linalg.norm(dense @ eigvec - eigval * eigvec))
                print(
                    f"{name} delta={delta}: eigval diff {abs(eigval - ref):.1e}, residual logged "
                    f"{logged:.3e} vs dense {res:.3e}, complex {np.iscomplexobj(dense)}"
                )
        return

    print(
        f"{'case':16} {'N':>9} {'level':>7} {'sum|c|':>7} {'eigval':>18} {'residual':>10} "
        f"{'threshold':>10} {'res/thr':>8} {'s':>6}"
    )
    for size in args.sizes:
        states = hamming_shells(n, size, np.random.default_rng(0))
        for name in args.patterns:
            for delta in args.deltas:
                ham = xxz(n, delta, *patterns(n)[name])
                sum_c = float(np.abs(ham.coeffs).sum())
                for level in levels:
                    t0 = time.perf_counter()
                    eigval = sqd(ham, states, return_eigvec=False, cache_level=level)
                    seconds = time.perf_counter() - t0
                    residual, threshold = handler.take()
                    print(
                        f"{name + ' d=' + str(delta):16} {len(states):>9} {level!s:>7} "
                        f"{sum_c:>7.2f} {eigval:>18.12f} {residual:>10.3e} {threshold:>10.3e} "
                        f"{residual / threshold:>8.4f} {seconds:>6.1f}",
                        flush=True,
                    )


if __name__ == "__main__":
    main()
