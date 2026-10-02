"""``sqd(matvec=Matvec.PAIRS)`` must reject a live mesh and work once it exits; see
``TestShardedSparseRejects``."""

import jax
import numpy as np
from common import emit, mesh

from rqutils.sqd import Matvec, sqd

HAM = (["XXII", "IZZI", "ZIII"], [0.7, -1.3, 0.4])
STATES = np.array([[0, 0, 0, 0], [1, 1, 0, 0], [0, 1, 1, 0], [1, 0, 1, 0]], dtype=np.uint8)


def rejects(matvec: str) -> str:
    try:
        sqd(HAM, STATES, return_eigvec=False, matvec=matvec)
    except ValueError as exc:
        return str(exc)
    return ""


def main() -> None:
    with jax.set_mesh(mesh(4)):
        scoped = {name: rejects(name) for name in [Matvec.PAIRS]}
    after = {name: sqd(HAM, STATES, return_eigvec=False, matvec=name) for name in [Matvec.PAIRS]}
    emit({"scoped": scoped, "after": after, "dense": sqd(HAM, STATES, return_eigvec=False)})


if __name__ == "__main__":
    main()
