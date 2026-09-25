"""Print the ``all-reduce`` arities in ``ground_locg``'s compiled loop body on a 4-device mesh.

Driven as a subprocess by ``test_ground_locg.py::TestAllReduceCount``, which owns the rationale. Not a
pytest module: the virtual device count must be set before jax initializes.
"""

import re

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax.sharding import AxisType, NamedSharding, PartitionSpec

from rqutils.ground_locg import ground_locg

N, MESH_SIZE = 1024, 4


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
    mesh = jax.make_mesh((MESH_SIZE,), ("x",), axis_types=(AxisType.Explicit,))
    with jax.set_mesh(mesh):
        spec = NamedSharding(mesh, PartitionSpec("x"))
        diag = jax.device_put(jnp.linspace(-1.0, 1.0, N), spec)
        xinit = jax.device_put(jnp.ones(N) / N**0.5, spec)
        for batch in (False, True):
            solve = jax.jit(
                lambda x, d, batch=batch: ground_locg(
                    matvec, x, (d, 0.3), maxiter=50, batch_matvec=batch
                )
            )
            body = while_body(solve.lower(xinit, diag).compile().as_text())
            print(f"arities {int(batch)} {','.join(map(str, allreduce_arities(body)))}")


if __name__ == "__main__":
    main()
