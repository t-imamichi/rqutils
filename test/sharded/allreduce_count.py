"""``all-reduce`` arities in ``ground_locg``'s compiled loop body on a mesh; see ``TestAllReduceCount``."""

import re

import jax
import jax.numpy as jnp
from common import emit, mesh
from jax.sharding import NamedSharding, PartitionSpec

from rqutils.ground_locg import ground_locg

N = 1024


def matvec(vec, diag, off):
    # Tridiagonal; the roll needs a replicated copy -- an all-gather, like sqd's -- and no all-reduce.
    full = jax.sharding.reshard(vec, PartitionSpec(*([None] * vec.ndim)))
    nbr = jnp.roll(full, 1, axis=-1) + jnp.roll(full, -1, axis=-1)
    return diag * vec + off * jax.sharding.reshard(nbr, jax.typeof(vec).sharding)


def while_body(text):
    """The HLO of the single ``while_loop``'s body computation: one steady-state iteration."""
    (name,) = set(re.findall(r"while\(.*?body=(%?[\w.\-]+)", text))
    start = re.search(rf"^{re.escape(name)} .*\{{$", text, re.MULTILINE).start()
    return text[start : text.index("\n}", start)]


def allreduce_arities(text):
    """Operand count of each ``all-reduce``, sorted; a combined reduction has arity > 1."""
    ops = re.findall(r"= (\([^)]*\)|\S+) all-reduce(?:-start)?\(", text)
    return sorted(o.count("[") for o in ops)


def main() -> None:
    the_mesh = mesh(4)
    arities = {}
    with jax.set_mesh(the_mesh):
        spec = NamedSharding(the_mesh, PartitionSpec("x"))
        diag = jax.device_put(jnp.linspace(-1.0, 1.0, N), spec)
        xinit = jax.device_put(jnp.ones(N) / N**0.5, spec)
        for batch in (False, True):
            solve = jax.jit(
                lambda x, d, batch=batch: ground_locg(
                    matvec, x, (d, 0.3), maxiter=50, batch_matvec=batch
                )
            )
            arities[str(batch)] = allreduce_arities(
                while_body(solve.lower(xinit, diag).compile().as_text())
            )
    emit(arities)


if __name__ == "__main__":
    main()
