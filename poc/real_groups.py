"""Real X groups first, held as float64: one arm's warm ``sqd`` times and XLA memory, as JSON.

``PauliSumXZ.num_real_groups`` lets ``run_sqd`` scan the real groups with float64 coefficients (and
cache float64 diagonals under ``TABLES``), the complex rest as before. Run once per arm, alternating,
with ``PYTHONPATH`` pointing the baseline at a worktree of the pre-change revision:

    git worktree add --detach /tmp/claude/wt-dev dev
    PYTHONPATH=/tmp/claude/wt-dev uv run python poc/real_groups.py --label dev
    uv run python poc/real_groups.py --label new

Prints one JSON object: per ``matvec``, the warm ``sqd`` times, the eigenvalue, and ``run_sqd``'s
compiled argument/temp bytes per slot. ``rqutils.__file__`` is included, to prove which tree ran.
"""

import argparse
import json
import time

import jax

jax.config.update("jax_enable_x64", True)

from sparse_pairs import spinchain_problem

import rqutils
from rqutils.sqd import Matvec, run_sqd, sqd
from rqutils.sqd._states import _pad_states


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--num-qubits", type=int, default=60)
    parser.add_argument("--pattern", default="type2")
    parser.add_argument("--delta", type=float, default=0.5)
    parser.add_argument("--log2-size", type=int, default=17)
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument(
        "--matvecs", type=Matvec, choices=list(Matvec), nargs="+",
        default=[Matvec.INDICES, Matvec.TABLES],
    )  # fmt: skip
    args = parser.parse_args()

    size = 1 << args.log2_size
    ham, states = spinchain_problem(args.num_qubits, args.pattern, args.delta, size)
    states_p = _pad_states(ham.pack_states(states), size)
    # getattr: the baseline worktree predates the field.
    real = getattr(ham, "num_real_groups", None)
    out = {"label": args.label, "rqutils": rqutils.__file__, "num_real_groups": real}
    for matvec in args.matvecs:
        memory = (
            run_sqd.lower(ham, states_p, size, False, matvec=matvec).compile().memory_analysis()
        )
        sqd(ham, states, return_eigvec=False, matvec=matvec)  # compile
        times, eigval = [], None
        for _ in range(args.trials):
            start = time.perf_counter()
            eigval = sqd(ham, states, return_eigvec=False, matvec=matvec)
            times.append(time.perf_counter() - start)
        out[str(matvec)] = {
            "times": times,
            "eigval": eigval,
            "args_B_per_slot": memory.argument_size_in_bytes / size,
            "temp_B_per_slot": memory.temp_size_in_bytes / size,
        }
    print(json.dumps(out))


if __name__ == "__main__":
    main()
