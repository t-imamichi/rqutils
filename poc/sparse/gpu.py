"""The sparse matvec kernels (``Matvec.PAIRS``/``CSR``/``ELL``) through ``sqd`` on one GPU.

``poc/sparse/pairs.md`` was measured on a laptop CPU; the GPU is its open question. Fixture: spinchain's
open-XXZ ``xxz`` with Hamming-shell subspaces, as ``poc/sparse/pairs.py``, over every ``--patterns``. Each
arm runs one warm-up solve (compile) then ``--repeats`` timed solves; every eigenvalue is checked against
``Matvec.INDICES``, and against a host ``hproj`` + ``eigsh`` oracle up to ``--oracle-log2``.

A sparse arm replays ``sqd._core._solve_sqd``'s sparse branch with a sync between stages, so its time
splits into ``build`` (``uniquify_states`` + the host-built operator), ``solve`` (the jitted LOBPCG) and
``check`` (``_sparse_residual`` + ``_checked_eigval``), each the median over ``--repeats``. ``iters`` is LOBPCG's count (one host callback per solve, as
``poc/real_groups.py``), and ``per_iter`` the solve time over it -- ``indices``' includes its setup.

Each arm runs in its own subprocess: ``peak_bytes_in_use`` is a process-wide high-water mark with no reset,
so a shared process would report the largest arm's peak for every later one. The CPU backend has no
``memory_stats``, so there ``peak`` is the child's RSS high-water mark, Python and JAX included. The parent is pinned to CPU
so it holds no GPU memory (nor XLA's preallocation) while a child runs.

Run: uv run python poc/sparse/gpu.py [--patterns type1 type2] [--log2-sizes 17 19 21]
     [--arms indices pairs csr ell] [--device 0]
"""

import argparse
import json
import os
import resource
import subprocess
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument("--num-qubits", type=int, default=60)
parser.add_argument("--patterns", nargs="+", default=["type1", "type2", "type3", "type4"])
parser.add_argument("--delta", type=float, default=0.5)
parser.add_argument("--log2-sizes", type=int, nargs="+", default=[17, 19, 21])
parser.add_argument("--arms", nargs="+", default=["indices", "pairs", "csr", "ell"])
parser.add_argument("--repeats", type=int, default=3)
parser.add_argument(
    "--oracle-log2", type=int, default=16, help="largest size checked against hproj"
)
parser.add_argument("--device", help="CUDA_VISIBLE_DEVICES, e.g. 0")
parser.add_argument("--child", nargs=3, help=argparse.SUPPRESS)  # pattern log2 arm
options = parser.parse_args()
if options.device is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = options.device  # before jax initializes
child_env = dict(os.environ)
if options.child is None:
    os.environ["JAX_PLATFORMS"] = "cpu"

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
from scipy.sparse.linalg import eigsh

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)  # poc/, for its fixtures
from eigenpair_check_scale import hamming_shells, patterns, xxz
from legacy import SPARSE
from legacy import operator as sparse_build
from legacy import run as run_sparse

import rqutils.sqd._solve as solve_mod
from rqutils.paulis.symplectic import PauliSumXZ
from rqutils.sqd import Matvec, hproj, sqd, uniquify_states
from rqutils.sqd._core import _checked_eigval, _sqd_inputs
from rqutils.sqd._sparse import _group_pairs, _sparse_residual

n = options.num_qubits
ITERATIONS = []
_ground_locg = solve_mod.ground_locg


def counted_ground_locg(*args, **kwargs):
    """``ground_locg``, recording each solve's iteration count on the host."""
    out = _ground_locg(*args, **kwargs)
    jax.debug.callback(lambda niter: ITERATIONS.append(int(niter)), out[2])
    return out


solve_mod.ground_locg = counted_ground_locg  # ty: ignore[invalid-assignment]


def fixture(pattern, log2):
    ham = PauliSumXZ.from_paulisum(xxz(n, options.delta, *patterns(n)[pattern]))
    return ham, hamming_shells(n, 1 << log2, np.random.default_rng(0))


def staged(ham, states, arm):
    """``(eigval, {stage: seconds})``: ``sqd``'s sparse path at its defaults, synced per stage."""
    ham, states_p, size = _sqd_inputs(ham, states, None, False, Matvec.PAIRS, 0.0, None, (32, 2))
    t0 = time.perf_counter()
    states_u = jax.block_until_ready(uniquify_states(states_p, size))
    pairs = _group_pairs(ham, states_u)  # one search for the build and the check, as sqd
    operator = jax.block_until_ready(sparse_build(ham, states_u, arm, pairs))
    t1 = time.perf_counter()
    result = jax.block_until_ready(run_sparse(ham, states_u, operator, size, True, arm))
    del operator
    t2 = time.perf_counter()
    residual, ax_norm = _sparse_residual(ham, states_u, result.eigval, result.eigvec, pairs)
    result = result._replace(residual=residual, ax_norm=ax_norm)
    eigval = _checked_eigval(result, ham, 1000, 0.0, None)
    t3 = time.perf_counter()
    return eigval, {"build": t1 - t0, "solve": t2 - t1, "check": t3 - t2}


def solve(pattern, log2, arm):
    ham, states = fixture(pattern, log2)
    if arm in SPARSE:  # "csr"/"ell" through poc/sparse/legacy.py, "pairs" through the library
        run = lambda: staged(ham, states, arm)
    else:
        run = lambda: (sqd(ham, states, return_eigvec=False, matvec=Matvec(arm)), {})
    t0 = time.perf_counter()
    eigval, _ = run()
    first = time.perf_counter() - t0
    times, stages = [], []
    for _ in range(options.repeats):
        t0 = time.perf_counter()
        stages.append(run()[1])
        times.append(time.perf_counter() - t0)
    jax.effects_barrier()
    stats = jax.devices()[0].memory_stats() or {}
    return {
        "backend": jax.default_backend(),
        "eigval": eigval,
        "first": first,
        "times": times,
        "stages": {k: float(np.median([st[k] for st in stages])) for k in (stages or [{}])[0]},
        "peak": stats.get("peak_bytes_in_use"),
        # The CPU backend reports no memory_stats; the child's RSS high-water mark, bytes on macOS.
        "rss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        * (1 if sys.platform == "darwin" else 1024),
        "iters": ITERATIONS[-1],
        "iter_counts": sorted(set(ITERATIONS)),
    }


if options.child is not None:
    pattern, log2, arm = options.child
    print(json.dumps(solve(pattern, int(log2), arm)), flush=True)
    sys.exit()

# INDICES first: the reference every other arm is checked against
arms = ["indices"] + [a for a in options.arms if a != "indices"]
warned = False

for pattern in options.patterns:
    for log2 in options.log2_sizes:
        ham, states = fixture(pattern, log2)
        print(f"\nN=2^{log2} ({len(states)}), n={n}, {pattern}, J={ham.x.shape[0]}", flush=True)
        oracle = None
        if log2 <= options.oracle_log2:
            oracle = float(eigsh(hproj(ham, states), k=1, which="SA", return_eigenvectors=False)[0])
            print(f"  oracle (hproj + eigsh) {oracle:.12f}", flush=True)
        ref = float("nan")
        for arm in arms:
            cmd = [sys.executable, __file__, *sys.argv[1:], "--child", pattern, str(log2), arm]
            out = subprocess.run(cmd, env=child_env, stdout=subprocess.PIPE, text=True, check=True)
            r = json.loads(out.stdout.splitlines()[-1])
            if r["backend"] != "gpu" and not warned:
                print("  WARNING: not a GPU backend; numbers below are not GPU numbers", flush=True)
                warned = True
            eigval, times = r["eigval"], r["times"]
            ref = eigval if arm == "indices" else ref
            delta = "" if oracle is None else f"  d_oracle={eigval - oracle:+.2e}"
            peak = (
                f"{r['peak'] / 2**30:.2f} GiB" if r["peak"] else f"rss {r['rss'] / 2**30:.2f} GiB"
            )
            solve_s = r["stages"].get("solve", float(np.median(times)))
            if len(r["iter_counts"]) > 1:
                print(f"  WARNING: {arm} iteration counts differ across runs: {r['iter_counts']}")
            print(
                f"  {arm:8s} E={eigval:.12f}  d_indices={eigval - ref:+.2e}{delta}"
                f"  first={r['first']:7.2f} s  min={min(times):7.3f} s"
                f"  median={np.median(times):7.3f} s  peak={peak}"
                f"  iters={r['iters']}  per_iter={1e3 * solve_s / r['iters']:.2f} ms"
                + "".join(f"  {k}={v:.3f} s" for k, v in r["stages"].items()),
                flush=True,
            )
