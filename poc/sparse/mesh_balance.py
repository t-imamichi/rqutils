"""Term-parallel ``"pairs"`` balance across devices: whole X groups against the shipped contiguous split.

For ``markdown/pairs-mesh-proposal.md`` §4.1. Fixture as ``poc/sparse/prune.py``: spinchain's open-XXZ
``xxz`` at ``--num-qubits``, Hamming-shell subspaces around both Néel states, all four field patterns.

- ``groups``: whole X groups packed greedily on their *searched* counts, largest first (the first build),
  loaded with what ``_drop_zeros`` keeps of each group.
- ``split``: ``_sparse_operator`` under a mesh, the shipped form, its stored entries per device, and its slots over stored entries.

Each reports the largest device's load over the mean. Host-only arithmetic, except ``split``, which
places arrays on ``--devices``' largest virtual CPU mesh.

Run: XLA_FLAGS=--xla_force_host_platform_device_count=64 uv run python poc/sparse/mesh_balance.py
     [--log2-size 17] [--devices 4 16 64]
"""

import argparse
import os
import sys

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # poc/
from _scaling_common import make_1d_mesh
from eigenpair_check_scale import hamming_shells, patterns, xxz

import rqutils.sqd._sparse as sm
from rqutils.paulis.symplectic import PauliSumXZ
from rqutils.sqd import Matvec, uniquify_states
from rqutils.sqd._core import _sqd_inputs

parser = argparse.ArgumentParser()
parser.add_argument("--num-qubits", type=int, default=60)
parser.add_argument("--delta", type=float, default=0.5)
parser.add_argument("--log2-size", type=int, default=17)
parser.add_argument("--devices", type=int, nargs="+", default=[4, 16, 64])
options = parser.parse_args()

n = options.num_qubits
for name in ("type1", "type2", "type3", "type4"):
    ham = PauliSumXZ.from_paulisum(xxz(n, options.delta, *patterns(n)[name]))
    states = hamming_shells(n, 1 << options.log2_size, np.random.default_rng(0))
    h, states_p, size = _sqd_inputs(ham, states, None, False, Matvec.PAIRS, 0.0, None, (32, 2))
    states_u = uniquify_states(states_p, size)
    pairs = sm._group_pairs(h, states_u)
    # What _drop_zeros keeps of each group: the factors of its searched pairs, nonzero.
    z, c = jnp.asarray(h.z), jnp.asarray(h.c)
    kmax = max(int(np.count_nonzero(np.asarray(h.c)[g])) for g in pairs)
    kept = {}
    for g, (i, j) in pairs.items():
        factors = sm._entry_factors(
            i[None], j[None], np.full((1, len(i)), g, np.int32), z, c, states_u, kmax
        )
        kept[g] = int(np.count_nonzero(np.asarray(factors)))
    total = sum(kept.values())
    groups, split = [], []
    for p in options.devices:
        loads, post = [0] * p, [0] * p
        for g in sorted(pairs, key=lambda g: -len(pairs[g][0])):
            k = loads.index(min(loads))
            loads[k] += len(pairs[g][0])
            post[k] += kept[g]
        groups.append(f"P={p} {max(post) / (total / p):.2f}x, {post.count(0)} idle")
        mesh = make_1d_mesh(devices=jax.devices()[:p])
        d = np.asarray(sm._sparse_operator(h, states_u, pairs, mesh)[3]).reshape(p, -1)
        live = np.count_nonzero(d, axis=1)
        assert live.sum() == total, (name, p, live.sum(), total)
        split.append(f"P={p} {live.max() / live.mean():.3f}x, slots {d.size / total:.2f}x")
    print(
        f"{name}: J={len(pairs)}, kept {total} of {sum(len(v[0]) for v in pairs.values())}, "
        f"largest group {max(kept.values()) / total:.1%}"
    )
    print("  groups: " + " | ".join(groups))
    print("  split:  " + " | ".join(split))
