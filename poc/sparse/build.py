"""Where a sparse kernel's host build goes: search, sort, factors and the rest, per ``--matvec``.

On the GH200 ``"ell"``'s build is 4.77 s of an 11.86 s ``type2`` ``2^22`` call against ``"pairs"``' 1.63
(``poc/sparse/gpu.md`` §8), and on an M1 the search is ~60% of it, the two-direction sort ~20% and the
factors 6-8%. ``"ell"``'s ``_flat_factors`` calls ``_entry_factors`` once per ``_CHUNK``-entry chunk,
~1,900 device calls there, each with its own copies and launch: cheap on CPU, maybe not on a GPU.

Each stage is timed by wrapping the module function with a sync, so a stage includes its device work;
``factors`` is ``_flat_factors`` for ``"ell"`` and ``_entry_factors`` otherwise, with its device-call
count. ``rest`` is the build minus the named stages (bucket assembly, padding, copies). One warm-up build,
then the median of ``--repeats``. Fixture as ``poc/sparse/gpu.py``.

Run: uv run python poc/sparse/build.py [--matvec ell pairs] [--patterns type1 type2] [--log2-sizes 20 22]
"""

import argparse
import functools
import os
import statistics
import sys
import time

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)  # poc/, for its fixtures
from eigenpair_check_scale import hamming_shells, patterns, xxz

import rqutils.sqd._sparse as sm
from rqutils.paulis.symplectic import PauliSumXZ
from rqutils.sqd import Matvec, uniquify_states
from rqutils.sqd._core import _sqd_inputs

parser = argparse.ArgumentParser()
parser.add_argument("--num-qubits", type=int, default=60)
parser.add_argument("--delta", type=float, default=0.5)
parser.add_argument(
    "--matvec", nargs="+", default=["ell", "pairs"], choices=["pairs", "csr", "ell"]
)
parser.add_argument("--patterns", nargs="+", default=["type1", "type2"])
parser.add_argument("--log2-sizes", type=int, nargs="+", default=[20, 22])
parser.add_argument("--repeats", type=int, default=3)
options = parser.parse_args()

TIMES, CALLS = {}, {}


def timed(name, fn):
    """``fn`` with its synced wall time and call count added to ``TIMES``/``CALLS`` under ``name``."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        out = jax.block_until_ready(fn(*args, **kwargs))
        TIMES[name] = TIMES.get(name, 0.0) + time.perf_counter() - t0
        CALLS[name] = CALLS.get(name, 0) + 1
        return out

    return wrapper


sm._search_pairs = timed("search", sm._search_pairs)
sm._sort_by_target = timed("sort", sm._sort_by_target)
sm._flat_factors = timed("flat", sm._flat_factors)
# Inside _flat_factors for "ell", so only its call count is reported there.
sm._entry_factors = timed("entry", sm._entry_factors)

print(f"{jax.devices()[0].device_kind}, n={options.num_qubits}")
print("matvec pattern N    | build (s) | search | sort | factors (device calls) | rest")
for pattern in options.patterns:
    ham = PauliSumXZ.from_paulisum(
        xxz(options.num_qubits, options.delta, *patterns(options.num_qubits)[pattern])
    )
    for log2 in options.log2_sizes:
        states = hamming_shells(options.num_qubits, 1 << log2, np.random.default_rng(0))
        for name in options.matvec:
            arm = Matvec(name)
            h, states_p, size = _sqd_inputs(ham, states, None, False, arm, 0.0, None, (32, 2))
            states_u = jax.block_until_ready(uniquify_states(states_p, size))
            runs = []
            for _ in range(options.repeats + 1):  # the first is the warm-up
                TIMES.clear()
                CALLS.clear()
                t0 = time.perf_counter()
                operator = jax.block_until_ready(sm._sparse_operator(h, states_u, arm))
                runs.append((time.perf_counter() - t0, dict(TIMES), dict(CALLS)))
                del operator
            runs = runs[1:]
            build = statistics.median(r[0] for r in runs)
            factors = "flat" if arm == "ell" else "entry"
            stage = {
                k: statistics.median(r[1].get(k, 0.0) for r in runs)
                for k in ("search", "sort", factors)
            }
            rest = build - sum(stage.values())
            calls = runs[0][2].get("entry", 0)
            print(
                f"{name:6} {pattern} 2^{log2} | {build:7.3f} |"
                f" {stage['search']:.3f} ({100 * stage['search'] / build:.0f}%) |"
                f" {stage['sort']:.3f} ({100 * stage['sort'] / build:.0f}%) |"
                f" {stage[factors]:.3f} ({100 * stage[factors] / build:.0f}%, {calls}) |"
                f" {rest:.3f} ({100 * rest / build:.0f}%)",
                flush=True,
            )
