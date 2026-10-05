"""Two layouts that split each state at a qubit cut: how they fit spinchain's subspaces. Host-only.

For ``markdown/pairs-mesh-proposal.md`` §11. Fixture: ``poc/eigenpair_check_scale``'s Hamming-shell
subspaces around both Néel states at ``--num-qubits``; the left half is the state's first ``cut`` columns.

- ``product``: the full product ``S_L × S_R`` a matrix-shaped vector would need, as a multiple of ``N``.
- ``hash``: states owned by a random hash of their left half; the largest device's load over the mean,
  per device count. The hash draws come from one generator in loop order, so a run reproduces exactly.

Run: uv run python poc/split_layouts.py [--product-sizes 14 17 20] [--hash-sizes 17 20]
"""

import argparse

import numpy as np
from eigenpair_check_scale import hamming_shells

parser = argparse.ArgumentParser()
parser.add_argument("--num-qubits", type=int, default=60)
parser.add_argument("--product-sizes", type=int, nargs="+", default=[14, 17, 20])
parser.add_argument("--product-cuts", type=int, nargs="+", default=[30, 20])
parser.add_argument("--hash-sizes", type=int, nargs="+", default=[17, 20])
parser.add_argument("--hash-cuts", type=int, nargs="+", default=[30, 20, 10])
parser.add_argument("--devices", type=int, nargs="+", default=[4, 16, 64])
options = parser.parse_args()


def subspace(log2):
    states = hamming_shells(options.num_qubits, 1 << log2, np.random.default_rng(0))
    return np.unique(np.asarray(states, np.uint8), axis=0)


print("product: |S_L| x |S_R| over N")
for log2 in options.product_sizes:
    s = subspace(log2)
    for cut in options.product_cuts:
        left, right = (len(np.unique(half, axis=0)) for half in (s[:, :cut], s[:, cut:]))
        print(
            f"  2^{log2} N={len(s)} cut={cut}: |L|={left} |R|={right} product/N={left * right / len(s):.1f}"
        )

print("hash: largest device load over the mean, states owned by a random hash of their left half")
rng = np.random.default_rng(1)
for log2 in options.hash_sizes:
    s = subspace(log2)
    for cut in options.hash_cuts:
        counts = np.unique(s[:, :cut], axis=0, return_counts=True)[1]
        salt = rng.integers(0, 2**62, size=len(counts))
        loads = [
            np.bincount(salt % p, weights=counts, minlength=p).max() / (len(s) / p)
            for p in options.devices
        ]
        cells = "  ".join(f"P={p} {x:.2f}x" for p, x in zip(options.devices, loads, strict=True))
        print(
            f"  2^{log2} cut={cut}: |L|={len(counts)} largest left half {counts.max()} "
            f"({counts.max() / len(s):.1%})  {cells}"
        )
