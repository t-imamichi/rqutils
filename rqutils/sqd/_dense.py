"""The dense kernels, ``"onthefly"``, ``"indices"`` and ``"tables"``, and :func:`apply_h`."""

from typing import Any

import jax
import jax.core
import jax.numpy as jnp
import numpy as np
from jax.sharding import PartitionSpec, get_abstract_mesh
from numpy.typing import NDArray

from rqutils.sqd._diagonal import get_diagonal
from rqutils.sqd._matvec import Matvec
from rqutils.sqd._states import StateList, get_xsource

# Dtype kind per `apply_h` keyword: `u` packed bytes, `i` positions (-1 = absent), `fc` float or
# complex. It separates arrays whose shapes collide, (2, 2) at n=15 with 2 states.
_ARRAY_ROLE_KINDS = {
    "xsignatures": ("u", "packed X signatures (uint8, from PauliSumXZ)"),
    "xsources": ("i", "X source indices (int32, from get_xsource; -1 marks absent)"),
    "zsignatures": ("u", "packed Z signatures (uint8, from PauliSumXZ)"),
    "diagonals": ("fc", "precomputed diagonals (float or complex, from get_diagonal)"),
}


def _check_array_role(name: str, array: Any) -> None:
    """Raise if the array under ``name`` has the wrong dtype kind for that role.

    Closes part of the misnaming residue :func:`apply_h` documents. Deliberately *not* a shape check:
    at ``n = 15`` (2 bytes) with a 2-state subspace, X signatures and X sources are both exactly
    ``(2, 2)``, so a shape assertion sails through the mispairing it exists to trip -- which is why
    the positional form was deleted rather than asserted.

    Dtype separates them structurally at exactly that point. Packed signatures are ``uint8``
    (:func:`numpy.packbits` output); source indices are ``int32`` positions carrying ``-1`` as the
    absent marker, and a ``uint8`` cannot hold ``-1``. Diagonals are float or complex.

    What it still does **not** reach: swapping two arrays of the same kind, e.g. ``xsignatures`` for
    ``zsignatures`` (both ``uint8``). That residue stays open.

    Args:
        name: The keyword the caller used.
        array: The array they passed under it.

    Raises:
        ValueError: If the dtype kind does not match the role.
    """
    expected, described = _ARRAY_ROLE_KINDS[name]
    # `.dtype` directly, not `np.asarray(array).dtype`: both numpy and jax arrays expose it, and the
    # asarray round-trip measured 13x slower (1.08 us against 0.083 us) for the same information.
    dtype = array.dtype
    if dtype.kind not in expected:
        raise ValueError(
            f"apply_h: {name}= expects {described}, but got dtype {dtype}. Check the keyword names "
            "against the arrays -- a shape check cannot catch this (X signatures and X sources are "
            "both (2, 2) at n=15 with 2 states), so the dtype is what distinguishes them."
        )


@jax.jit
def apply_xgrp(
    xsource: NDArray[np.int32], diagonal: NDArray[np.inexact], vec: NDArray[np.inexact]
) -> jax.Array:
    """Gather vector entries from the source indices and multiply them with diagonals."""
    xvec = vec.at[..., xsource].get(
        mode="fill",
        fill_value=0.0,
        wrap_negative_indices=False,
        out_sharding=jax.typeof(vec).sharding,
    )
    return xvec * diagonal


def _pack_scanned(
    matvec: Matvec, xgroup: NDArray, diagonal_arg: NDArray, coeffs: NDArray | None
) -> tuple[NDArray, ...]:
    """Lay out the tuple ``_apply_h_kernel`` scans over, for one resolved ``matvec``.

    The kernel unpacks positionally (``val[0]``, ``val[1]``, and ``val[2]`` only when ``matvec`` is
    not ``"tables"``), so the arity rule is a contract between packer and kernel: a 3-tuple carrying
    the coefficients for the two kernels that *compute* a diagonal, a 2-tuple for ``"tables"``,
    which reads a precomputed one. Both callers -- ``run_sqd`` and ``apply_h``'s keyword resolution --
    go through here so that rule is stated once rather than once per caller.
    """
    if matvec == "tables":
        return (xgroup, diagonal_arg)
    return (xgroup, diagonal_arg, coeffs)


def apply_h(
    vec: NDArray[np.inexact],
    *,
    states: StateList | None = None,
    xsignatures: NDArray | None = None,
    xsources: NDArray | None = None,
    zsignatures: NDArray | None = None,
    diagonals: NDArray | None = None,
    coeffs: NDArray | None = None,
) -> jax.Array:
    r"""Return :math:`Hv`, naming the per-X-group inputs so a mispairing cannot be expressed.

    Name the per-X-group arrays you have and the kernel follows from them. Exactly three input sets
    are accepted, one per dense :func:`sqd` ``matvec``; the sparse kernels (see the module
    documentation) are ``sqd``-only:

    .. code-block:: python

        apply_h(vec, states=..., xsignatures=..., zsignatures=..., coeffs=...)  # "onthefly"
        apply_h(vec, states=..., xsources=..., zsignatures=..., coeffs=...)     # "indices"
        apply_h(vec, xsources=..., diagonals=...)                               # "tables"

    Every array parameter is keyword-only; the positional ``(scanned, cache_level)`` form is gone
    (``NOTES.md``, "``sqd``: why ``apply_h``'s positional form was deleted rather than deprecated").
    :func:`sqd` does not go through here: it binds a static ``matvec`` name to the private
    ``_apply_h_kernel`` via ``functools.partial``, since the solver splats ``matvec(vec, *args)``.

    A dtype check (:func:`_check_array_role`) separates roles a shape cannot -- at ``n = 15`` with 2
    states X sources and X signatures are both ``(2, 2)`` -- but not a swap between roles of the
    **same** kind, ``xsignatures`` for ``zsignatures``, both ``uint8``.

    Each kernel is one ``jax.lax.scan`` over the X groups accumulating
    ``out + apply_xgrp(xsource, diagonal, vec)``, with ``xsource`` either ``get_xsource(x, states)``
    or ``xsources`` as given, and ``diagonal`` either ``get_diagonal(z, c, states)`` or ``diagonals``
    as given.

    **Under a mesh** ``vec`` is placed on the live mesh automatically, batch axis and all. With
    ``xsignatures=`` the state count must divide the device count: build the subspace with
    ``uniquify_states(states, states_size)`` at a divisible size, **not** from ``sqd``'s trimmed
    return. A ``states`` passed already sharded must be replicated, its ``255`` filler being
    load-bearing.

    Args:
        vec: Vector to multiply. Placed on the live mesh unless already there.
        states: Uniquified state list. Required except with ``diagonals=``, which reads neither
            signature array.
        xsignatures: Packed X signatures per group; selects ``"onthefly"``.
        xsources: Precomputed X source indices per group; selects ``"indices"`` or ``"tables"``.
        zsignatures: Packed Z signatures per group; selects ``"onthefly"`` or ``"indices"``. Needs
            ``coeffs``.
        diagonals: Fully precomputed diagonals per group; selects ``"tables"``, so requires
            ``xsources=``. Must not be combined with ``coeffs``, which it makes redundant.
        coeffs: Pauli coefficients per group. Required by ``zsignatures``.

    Returns:
        :math:`Hv`.

    Raises:
        ValueError: If the named arrays do not select exactly one X source and one diagonal input
            (including naming none at all); if ``diagonals`` is combined with ``xsignatures``; if
            ``coeffs`` is missing where required or supplied alongside ``diagonals``; if ``states``
            is None without ``diagonals``; or if an array's dtype kind does not match its keyword.

            Under a mesh with ``xsignatures=``, also if ``states`` disagrees with ``vec``'s trailing
            axis, or if its row count does not divide the device count.
    """
    xgiven = [
        opt for opt in (("xsources", xsources), ("xsignatures", xsignatures)) if opt[1] is not None
    ]
    dgiven = [
        opt
        for opt in (("diagonals", diagonals), ("zsignatures", zsignatures))
        if opt[1] is not None
    ]
    if len(xgiven) != 1:
        raise ValueError(
            "apply_h: pass exactly one of xsources= or xsignatures= "
            f"(got {sorted(name for name, _ in xgiven) or 'neither'})"
        )
    if len(dgiven) != 1:
        raise ValueError(
            "apply_h: pass exactly one of diagonals= or zsignatures= "
            f"(got {sorted(name for name, _ in dgiven) or 'neither'})"
        )
    (xname, xarray), (dname, darray) = xgiven[0], dgiven[0]
    precomputed = dname == "diagonals"
    if precomputed and xname == "xsignatures":
        raise ValueError(
            "apply_h: diagonals= requires xsources=, not xsignatures= -- that pairing was removed as "
            "dominated by matvec=Matvec.INDICES; pass xsources= from get_xsource"
        )
    matvec = "tables" if precomputed else "indices" if xname == "xsources" else "onthefly"

    # coeffs is required by the computed diagonal and meaningless beside a precomputed one, so a stray
    # one is rejected rather than ignored.
    if precomputed and coeffs is not None:
        raise ValueError("apply_h: diagonals= already folds in coeffs=; do not pass both")
    if not precomputed and coeffs is None:
        raise ValueError(f"apply_h: {dname}= requires coeffs=")

    # Not redundant with the kernel's check: the missing-input-set error must precede the role check
    # below (TestMatvecKernels::test_omitting_states_raises); keep the messages in step.
    if matvec != "tables" and states is None:
        raise ValueError(f"states is required for matvec={matvec!r}")

    # Dtype separates what shape cannot (uint8 signatures vs int32 sources, both (2, 2) at n=15);
    # last, so it only adds errors and never displaces an input-set error raised above.
    _check_array_role(xname, xarray)
    _check_array_role(dname, darray)

    # `states`, not `vec`: `get_xsource` reshards one entry per state, and only "onthefly" gets there.
    vec = _place_vec(vec, states if matvec == "onthefly" else None)

    return _apply_h_kernel(vec, _pack_scanned(matvec, xarray, darray, coeffs), states, matvec)


def _check_mesh_divisible(num_states: int) -> None:
    """Raise unless the state count divides the mesh, naming the size to pass `uniquify_states`.

    `get_xsource` reshards one entry per state. jax's own message names neither the size nor the call
    that produces it.

    Raises:
        ValueError: If ``num_states`` is not a multiple of the device count.
    """
    mesh = get_abstract_mesh()
    if mesh.empty or num_states % mesh.size == 0:
        return
    size = -(-num_states // mesh.size) * mesh.size
    raise ValueError(
        f"apply_h: {num_states} states is not a multiple of the {mesh.size} mesh devices; size "
        f"every per-state array to {size}, e.g. uniquify_states(states, {size})"
    )


def _place_vec(vec: NDArray[np.inexact], divisible: StateList | None) -> NDArray[np.inexact]:
    """Put `vec` on the live mesh, replicated; return it unchanged when there is no mesh.

    Keyed on mesh *identity*, not `isinstance(vec, jax.Array)`: a committed `jax.Array` carries an
    empty mesh just as a host array does, so that guard passed it through to the "Resource axis" error
    this prevents. `get_mesh`, not `get_abstract_mesh`: `device_put` rejects an `AbstractMesh`. Skipped
    under tracing, where `get_mesh` raises.

    Args:
        vec: Vector to place.
        divisible: States whose row count must divide the device count, or None to skip that check.

    Raises:
        ValueError: If ``divisible``'s row count is not a multiple of the device count, or disagrees
            with ``len(vec)``.
    """
    if isinstance(vec, jax.core.Tracer) or (mesh := jax.sharding.get_mesh()).empty:
        return vec
    if divisible is not None:
        # shape[-1], not shape[0]: the kernel broadcasts over a leading batch axis of any size.
        if (n := divisible.shape[0]) != vec.shape[-1]:
            raise ValueError(f"apply_h: vec length {vec.shape[-1]} disagrees with {n} states")
        _check_mesh_divisible(n)
    if jax.typeof(vec).sharding.mesh is mesh.abstract_mesh:
        return vec
    return jax.device_put(vec, jax.sharding.NamedSharding(mesh, PartitionSpec()))


@jax.jit(static_argnames=["matvec"])
def _apply_h_kernel(
    vec: NDArray[np.inexact],
    scanned: tuple[NDArray, ...],
    states: StateList | None,
    matvec: Matvec,
) -> jax.Array:
    r"""Return :math:`Hv`, resolving the per-X-group inputs according to the ``matvec`` name.

    The jitted kernel behind :func:`apply_h`. Kept positional with a static ``matvec`` because
    :mod:`ground_locg` splats ``matvec(vec, *args)``: a ``static_argnames`` entry would never see a
    keyword, so it is bound via :func:`functools.partial` rather than passed through ``args``.
    The name resolution lives in the wrapper, in plain Python, so it costs nothing per call.

    See :func:`apply_h` for the input sets and argument semantics.

    Raises:
        ValueError: If ``states`` is None for a kernel other than ``"tables"``.
    """
    if matvec != "tables" and states is None:
        raise ValueError(f"states is required for matvec={matvec!r}")

    def fn(out, val):
        # val[0] is the X source for this group: either the precomputed index array or the X
        # signature it is derived from.
        xsource = get_xsource(val[0], states) if matvec == "onthefly" else val[0]
        diagonal = val[1] if matvec == "tables" else get_diagonal(val[1], val[2], states)
        return out + apply_xgrp(xsource, diagonal, vec), None

    return jax.lax.scan(fn, jnp.zeros_like(vec), scanned)[0]
