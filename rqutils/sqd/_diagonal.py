"""The composed diagonal of one X group, from the parity of each Z term."""

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import PartitionSpec
from numpy.typing import NDArray

from rqutils.sqd._states import StateList


def _check_zsignatures_rank(zsignatures: Any) -> None:
    """Raise unless ``zsignatures`` is one X group's 2-D ``(num_zterms, num_bytes)`` array.

    :func:`get_diagonal` is public and indexes the array's leading axis, so handed a 1-D array it
    reads *scalars* and silently returns a wrongly shaped result rather than raising. Measured:
    ``(4,)`` of ``[2., 2., 2., 2.]``, a plausible finite diagonal. Rank is static under ``jax.jit``,
    so unlike ``get_xsource``'s lex-sortedness precondition this one is checkable here.

    Args:
        zsignatures: The caller's Z-signature array.

    Raises:
        ValueError: If it is not 2-D.
    """
    if np.ndim(zsignatures) != 2:
        raise ValueError(
            "`zsignatures` must be 2-D with shape (num_zterms, num_bytes) -- one X group's Z "
            f"signatures -- but has rank {np.ndim(zsignatures)}. A 1-D array would be scanned as "
            "scalars and silently return the wrong shape. Pass `hamiltonian.z[igroup]`, not "
            "`hamiltonian.z[igroup][iterm]`."
        )


def _z_parity(states: StateList, zsignature: jax.Array) -> jax.Array:
    """Return `popcount(state & z) mod 2` per state, the sign bit of one Z term.

    Accumulated in uint8 and reduced with `& 1`: the parity is the only bit that matters.
    """
    return jnp.sum(jnp.bitwise_count(states & zsignature), axis=1, dtype=np.uint8) & 1


def _accumulate_diagonal(
    coeffs: NDArray[np.inexact], template: jax.Array, sign_bit: Callable[[jax.Array], jax.Array]
) -> jax.Array:
    """Sum ``coeff * (1 - 2 * sign_bit(iterm))`` over the Z terms of one X group.

    Null terms are removed by ``hamiltonian.simplify()`` on ingest, so a zero coefficient marks the
    end of the real terms in a zero-padded Z group and the loop stops there rather than scanning the
    full rectangle.

    Args:
        coeffs: Phased coefficients for this X group, shape ``(K,)``.
        template: Array whose leading axis length and sharding the output follows. May be of any
            rank -- only the leading axis's partitioning is carried over, since the output is 1-D.
        sign_bit: Maps a term index to that term's per-state sign bit (0 or 1).

    Returns:
        The composed diagonal, shape ``(template.shape[0],)``.
    """

    def cond_fn(val):
        iterm = val[1]
        return jnp.logical_and(iterm < coeffs.shape[0], jnp.not_equal(coeffs[iterm], 0.0))

    def add_diag(val):
        diagonal, iterm = val
        signs = 1.0 - 2.0 * sign_bit(iterm)
        return diagonal + coeffs[iterm] * signs, iterm + 1

    # Only the template's leading-axis sharding (2-D template, 1-D output), as a NamedSharding so it
    # works with no mesh (NOTES.md, "sqd._accumulate_diagonal: the output sharding").
    sharding = jax.typeof(template).sharding
    init = jnp.zeros(
        template.shape[0],
        dtype=coeffs.dtype,
        out_sharding=jax.sharding.NamedSharding(sharding.mesh, PartitionSpec(sharding.spec[0])),
    )
    return jax.lax.while_loop(cond_fn, add_diag, (init, 0))[0]


@jax.jit
def get_diagonal(
    zsignatures: NDArray[np.uint8], coeffs: NDArray[np.inexact], states: StateList
) -> jax.Array:
    """Return the fully composed diagonals for one X signature.

    Args:
        zsignatures: Packed Z signatures for one X group, shape ``(num_zterms, num_bytes)``.
        coeffs: Phase-folded coefficients for that group.
        states: Uniquified, lex-sorted packed state list.

    Returns:
        The composed diagonal, one entry per state.

    Raises:
        ValueError: If ``zsignatures`` is not 2-D -- see :func:`_check_zsignatures_rank`.
    """
    _check_zsignatures_rank(zsignatures)

    def sign_bit(iterm):
        return _z_parity(states, zsignatures[iterm])

    return _accumulate_diagonal(coeffs, states, sign_bit)
